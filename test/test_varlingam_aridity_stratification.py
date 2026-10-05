from __future__ import annotations

import hashlib
import json
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

from confoundry.varlingam_aridity_stratification import (
    SOURCES, SIGNED, add_response_indices, align_aridity, annual_ai_name,
    assign_bins, download_aridity, fdr_bh, load_effects, load_qc, trend_tests,
    varlingam_aridity_stratification,
)


def write_raster(path, values, transform, crs="EPSG:4326", nodata=-9999):
    values = np.asarray(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0],
                       width=values.shape[1], count=1, dtype="float32",
                       transform=transform, crs=crs, nodata=nodata) as dest:
        dest.write(values, 1)


def test_alignment_averages_footprints_scales_and_preserves_original_indices(tmp_path):
    reference = tmp_path / "reference.tif"
    source = tmp_path / "ai.tif"
    write_raster(reference, np.zeros((3, 4)), from_origin(-9, 44, 1, 1))
    values = np.full((6, 8), 10000.0)
    values[2:4, 2:4] = [[2000, 4000], [6000, 8000]]
    values[2:4, 4:6] = [[-9999, -9999], [-9999, 10000]]
    values[4:6, 2:4] = 0  # A true zero is valid hyperaridity, not nodata.
    write_raster(source, values, from_origin(-9, 44, 0.5, 0.5))
    qc = pd.DataFrame({"row": [1, 1, 2], "col": [1, 2, 1], "qc_pass": True})
    result = align_aridity(qc, reference, source, 0.0001, 1, 0.5, tmp_path / "aligned.tif")
    np.testing.assert_allclose(result.aridity_index, [0.5, np.nan, 0], atol=1e-6)
    np.testing.assert_allclose(result.aridity_coverage, [1, 0.25, 1])
    np.testing.assert_allclose(result.longitude, [-7.5, -6.5, -7.5])
    np.testing.assert_allclose(result.latitude, [42.5, 42.5, 41.5])
    with rasterio.open(tmp_path / "aligned.tif") as aligned:
        assert aligned.width == aligned.height == 2
        assert aligned.tags()["reference_row_offset"] == "1"
        assert aligned.transform == from_origin(-8, 43, 1, 1)


def test_alignment_reprojects_to_a_different_crs(tmp_path):
    from rasterio.warp import transform
    reference, source = tmp_path / "reference.tif", tmp_path / "ai.tif"
    x, y = transform("EPSG:4326", "EPSG:3035", [-5], [40])
    write_raster(reference, [[0]], from_origin(x[0] - 5000, y[0] + 5000, 10000, 10000), "EPSG:3035")
    write_raster(source, np.full((20, 20), 6500), from_origin(-6, 41, 0.1, 0.1))
    qc = pd.DataFrame({"row": [0], "col": [0], "qc_pass": [True]})
    result = align_aridity(qc, reference, source, 0.0001, 1, 0.5, tmp_path / "out.tif")
    assert result.aridity_index.iloc[0] == pytest.approx(0.65)
    assert result.longitude.iloc[0] == pytest.approx(-5)
    assert result.latitude.iloc[0] == pytest.approx(40)
    # Avoid float32 rounding shifting official integer-encoded AI below 0.65.
    binned, _ = assign_bins(result, 5)
    assert binned.climate_class.iloc[0] == "Humid"


def test_bins_use_pre_qc_population_ties_and_documented_class_boundaries():
    frame = pd.DataFrame({"aridity_index": [0, .03, .2, .5, .65, 1, 2, 3, 4, np.nan],
                          "qc_pass": [True] * 5 + [False] * 5})
    binned, definitions = assign_bins(frame, 5)
    assert list(binned.climate_class.iloc[:5]) == ["Hyperarid", "Arid", "Semiarid", "Dry subhumid", "Humid"]
    expected = np.quantile(frame.aridity_index.dropna(), .2)
    assert definitions.iloc[0].upper == pytest.approx(expected)
    assert pd.isna(binned.iloc[-1]["quantile"])
    tied, bins = assign_bins(pd.DataFrame({"aridity_index": [0.5] * 20}), 5)
    assert tied["quantile"].nunique() == 1
    assert len(bins.loc[bins.scheme.eq("quantile")]) == 1


def test_response_indices_preserve_missingness_and_signed_cancellation():
    frame = pd.DataFrame([
        [1, -2, 3, -2, 2],  # W=6, A=4; mean magnitudes equal => share 0.5
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, np.nan, 1, 1, 1],
    ], columns=SIGNED)
    frame["qc_pass"], frame["aridity_index"] = True, 0.5
    result = add_response_indices(frame)
    assert result.loc[0, "water_sum"] == 6
    assert result.loc[0, "water_atmospheric_ratio"] == 1.5
    assert result.loc[0, "water_share"] == 0.5
    assert result.loc[0, "shallow_soil_moisture"] == -2
    assert np.isnan(result.loc[1, "water_atmospheric_ratio"])
    assert result.loc[1, "water_share"] == 1
    assert np.isnan(result.loc[2, "water_share"])
    assert result.loc[2, "analysis_eligible"]  # Pruned zeros are retained.
    assert not result.loc[3, "analysis_eligible"]
    assert np.isnan(result.loc[3, "water_sum"])


def test_qc_rejects_unknown_booleans_and_duplicate_coordinates(tmp_path):
    path = tmp_path / "qc.csv"
    pd.DataFrame({"row": [0], "col": [0], "primary_eligible": ["maybe"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="booleans"):
        load_qc(path, "primary_eligible")
    pd.DataFrame({"row": [0, 0], "col": [0, 0], "primary_eligible": [True, False]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_qc(path, "primary_eligible")


def effect_rows():
    rows = []
    for row in range(4):
        for col in range(5):
            for index, source in enumerate(SOURCES):
                rows.append(dict(row=row, col=col, source=source, target="ndvi_resid", horizon=12,
                                 scaled_cumulative_total_effect=(index + 1) * (1 + col) / (row + 1),
                                 cumulative_total_effect=index + 1, source_q_delta=1,
                                 target_q_delta=2, model_type="varlingam", error=None))
    return pd.DataFrame(rows)


def test_loader_filters_horizon_and_rejects_duplicates(tmp_path):
    frame = effect_rows()
    other = frame.copy()
    other["horizon"] = 1
    other["scaled_cumulative_total_effect"] = 999
    path = tmp_path / "effects.csv"
    pd.concat([frame, other]).to_csv(path, index=False)
    values = load_effects(None, path, "unused", "ndvi_resid", 12, -1)
    assert len(values) == 20
    assert values.potential_evaporation.max() < 0
    assert values.precipitation.max() < 999
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate effects"):
        load_effects(None, path, "unused", "ndvi_resid", 12, 1)


def test_block_tests_reproducible_and_do_not_gain_evidence_by_repeating_pixels():
    n = 18
    frame = pd.DataFrame({"easting": np.arange(n) * 100000 + 1,
                          "northing": np.zeros(n), "aridity_index": np.arange(n) / n + .1})
    for index, metric in enumerate(["water_share", *SIGNED, "water_atmospheric_ratio"]):
        frame[metric] = np.arange(n) * (-1 if index % 2 else 1)
    args = dict(sizes=[100.0], primary_size=100.0, bootstrap=99, permutations=199, min_blocks=10, seed=17)
    tests, _ = trend_tests(frame, **args)
    repeated, _ = trend_tests(pd.concat([frame] * 5, ignore_index=True), **args)
    np.testing.assert_allclose(tests.p_permutation, repeated.p_permutation)
    np.testing.assert_allclose(tests.ci_low, repeated.ci_low)
    assert (tests.n_blocks == n).all()
    assert (tests.p_permutation <= .01).all()
    np.testing.assert_allclose(tests.rho.abs(), 1)
    constant = frame.copy()
    constant["water_share"] = .5
    result, _ = trend_tests(constant, **args)
    assert result.iloc[0].status == "constant_block_values"
    assert np.isnan(result.iloc[0].p_permutation)
    result, _ = trend_tests(frame.iloc[:5], **args)
    assert result.status.eq("too_few_blocks").all()
    np.testing.assert_allclose(fdr_bh(np.array([.01, .04, .2, np.nan])), [.03, .06, .2, np.nan])


def test_cached_download_checks_checksum_and_extracts_safely(tmp_path, monkeypatch):
    archive = tmp_path / "ai_v3_yr.zip"
    with zipfile.ZipFile(archive, "w") as dest:
        dest.writestr("../../ai_v3_yr.tif", b"test raster")
    checksum = hashlib.md5(archive.read_bytes()).hexdigest()
    metadata = {"version": 5, "files": [{"name": archive.name, "computed_md5": checksum,
                                        "download_url": "https://example.invalid/no-network"}]}
    (tmp_path / "figshare_7504448_v5.json").write_text(json.dumps(metadata))
    def fail_network(*args, **kwargs):
        raise AssertionError("A valid cached archive should not trigger network access.")
    monkeypatch.setattr("confoundry.varlingam_aridity_stratification.requests.get", fail_network)
    raster, provenance = download_aridity(tmp_path)
    assert raster.parent == tmp_path
    assert raster.read_bytes() == b"test raster"
    assert provenance["archive_member"] == "../../ai_v3_yr.tif"
    assert annual_ai_name("ai_v3_yr.zip")
    assert not annual_ai_name("et0_v3_yr.zip")
    assert not annual_ai_name("ai_v3_monthly.zip")


def test_download_fetches_only_selected_annual_archive_and_validates_checksum(tmp_path, monkeypatch):
    content = b"mock annual raster"
    metadata = {"version": 5, "files": [
        {"name": "ai_v3_yr.tif", "computed_md5": hashlib.md5(content).hexdigest(),
         "download_url": "https://example.invalid/annual"},
        {"name": "ai_v3_monthly.zip", "download_url": "https://example.invalid/monthly"}]}
    calls = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def raise_for_status(self):
            pass
        def json(self):
            return metadata
        def iter_content(self, size):
            return iter([content])
    def fetch(url, **kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr("confoundry.varlingam_aridity_stratification.requests.get", fetch)
    raster, _ = download_aridity(tmp_path)
    assert raster.read_bytes() == content
    assert calls[-1] == "https://example.invalid/annual"
    assert len(calls) == 2
    metadata["files"][0]["computed_md5"] = "0" * 32
    (tmp_path / "figshare_7504448_v5.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum failed"):
        download_aridity(tmp_path)


@pytest.mark.parametrize("database", [False, True])
def test_cli_end_to_end_produces_publication_artifacts(tmp_path, database):
    reference, raster = tmp_path / "reference.tif", tmp_path / "aridity.tif"
    write_raster(reference, np.zeros((4, 5)), from_origin(-9, 44, .5, .5))
    write_raster(raster, np.linspace(.1, 1.5, 20).reshape(4, 5), from_origin(-9, 44, .5, .5))
    qc = pd.DataFrame({"row": np.repeat(np.arange(4), 5), "col": np.tile(np.arange(5), 4),
                       "primary_eligible": [False] + [True] * 19})
    qc.to_csv(tmp_path / "qc.csv", index=False)
    frame = effect_rows()
    # One QC-passing pixel has a missing source estimate and must not become a zero.
    frame = frame.loc[~((frame.row == 3) & (frame.col == 4) & (frame.source == "precipitation_resid"))]
    if database:
        effects = tmp_path / "effects.duckdb"
        with duckdb.connect(str(effects)) as con:
            con.register("frame", frame)
            con.execute("CREATE TABLE pixel_varlingam_effects AS SELECT * FROM frame")
        input_args = ["--effects-db", str(effects)]
    else:
        effects = tmp_path / "effects.csv"
        frame.to_csv(effects, index=False)
        input_args = ["--effects-csv", str(effects)]
    original_hash = hashlib.sha256(effects.read_bytes()).hexdigest()
    output = tmp_path / "results"
    args = [*input_args, "--qc-csv", str(tmp_path / "qc.csv"),
            "--reference-raster", str(reference), "--aridity-raster", str(raster),
            "--aridity-scale", "1", "--output-dir", str(output), "--bootstrap", "99",
            "--permutations", "99", "--block-km", "20", "--sensitivity-block-km", "50",
            "--min-blocks", "3"]
    if database:
        args.append("--no-plots")
    result = CliRunner().invoke(varlingam_aridity_stratification, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"]["analyzed"] == 18
    assert manifest["counts"]["all_fitted"] == 20
    assert manifest["counts"]["qc_pass_incomplete_effects"] == 1
    assert (output / "trend_tests.tex").exists()
    assert (output / "report.md").exists()
    assert len(pd.read_csv(output / "trend_tests.csv")) == 14
    assert hashlib.sha256(effects.read_bytes()).hexdigest() == original_hash
    coverage = pd.read_csv(output / "qc_coverage.csv")
    assert coverage.loc[coverage.scheme.eq("quantile"), "n_fitted"].sum() == 20
    if not database:
        for name in ["signed_effects_quantile", "signed_effects_climate_class",
                     "response_regime_quantile", "response_regime_climate_class",
                     "block_aridity_relationship", "aridity_coverage_map"]:
            assert (output / f"{name}.pdf").stat().st_size > 1000
            assert (output / f"{name}.png").stat().st_size > 1000
    # Existing results cannot be silently overwritten.
    result = CliRunner().invoke(varlingam_aridity_stratification, args)
    assert result.exit_code != 0
    assert "not empty" in result.output
