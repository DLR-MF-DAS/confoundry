from __future__ import annotations

import json
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

from confoundry.irrigation_covariates import align_fraction, download, download_gmia, download_worldcover, study_grid
from confoundry.varlingam_aridity_stratification import SOURCES
from confoundry.varlingam_irrigation_stratification import (
    association_test, bin_summaries, join_aridity, load_diagnostics,
    prepare_population, trend_tests, varlingam_irrigation_stratification,
)


def raster(path, values, transform, nodata=-9999, crs="EPSG:4326"):
    values = np.asarray(values, dtype="float64")
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float64", transform=transform, crs=crs, nodata=nodata) as target:
        target.write(values, 1)


def test_gmia_minus9_means_zero_but_outside_grid_is_missing(tmp_path):
    ref, source = tmp_path / "ref.tif", tmp_path / "gmia.tif"
    raster(ref, [[0, 0, 0]], from_origin(-9, 43, 1, 1))
    raster(source, [[-9, 25]], from_origin(-9, 43, 1, 1), nodata=-9, crs=None)
    pixels, grid = study_grid(pd.DataFrame({"row": [0, 0, 0], "col": [0, 1, 2]}), ref, "EPSG:3035")
    result = align_fraction(pixels, grid, [source], tmp_path / "out.tif", "irrigation_fraction", "gmia")
    np.testing.assert_allclose(result.irrigation_fraction, [0, .25, np.nan])
    np.testing.assert_allclose(result.irrigation_fraction_coverage, [1, 1, 0])


def test_native_categorical_fraction_coverage_and_disjoint_tile_seam(tmp_path):
    ref, left, right = (tmp_path / n for n in ["ref.tif", "left.tif", "right.tif"])
    raster(ref, [[0, 0]], from_origin(0, 2, 2, 2))
    # A reference pixel straddles a source-tile seam; classify before average.
    raster(left, [[40], [40]], from_origin(0, 2, 1, 1), nodata=0)
    raster(right, [[10, 40, 0], [10, 40, 0]], from_origin(1, 2, 1, 1), nodata=0)
    pixels, grid = study_grid(pd.DataFrame({"row": [0, 0], "col": [0, 1]}), ref, "EPSG:3035")
    result = align_fraction(pixels, grid, [left, right], tmp_path / "out.tif", "crop", "categorical", classes=(40,))
    np.testing.assert_allclose(result.crop, [.5, 1])
    np.testing.assert_allclose(result.crop_coverage, [1, .5])
    with pytest.raises(ValueError, match="overlap"):
        align_fraction(pixels, grid, [left, left], tmp_path / "bad.tif", "crop", "categorical", classes=(40,), chunk_cells=1)


def test_fraction_scaling_mask_original_offsets_and_reprojection(tmp_path):
    ref, source = tmp_path / "ref.tif", tmp_path / "source.tif"
    raster(ref, np.zeros((3, 4)), from_origin(-9, 44, 1, 1))
    values = np.full((6, 8), 50.)
    values[2:4, 2:4] = [[-9999, -9999], [-9999, 100]]
    raster(source, values, from_origin(-9, 44, .5, .5))
    pixels, grid = study_grid(pd.DataFrame({"row": [1, 2], "col": [1, 2]}), ref, "EPSG:3035")
    result = align_fraction(pixels, grid, [source], tmp_path / "out.tif", "crop", "fraction", .01)
    np.testing.assert_allclose(result.crop, [np.nan, .5])
    np.testing.assert_allclose(result.crop_coverage, [.25, 1])
    with rasterio.open(tmp_path / "out.tif") as target:
        assert target.tags()["reference_row_offset"] == "1"
        assert target.transform == from_origin(-8, 43, 1, 1)
    with pytest.raises(ValueError, match="outside"):
        align_fraction(pixels, grid, [source], tmp_path / "bad.tif", "crop", "fraction", 1)
    x, y = rasterio.warp.transform("EPSG:4326", "EPSG:3035", [-5], [40])
    raster(ref, [[0]], from_origin(x[0]-5000, y[0]+5000, 10000, 10000), crs="EPSG:3035")
    raster(source, np.full((20, 20), .3), from_origin(-6, 41, .1, .1))
    pixels, grid = study_grid(pd.DataFrame({"row": [0], "col": [0]}), ref, "EPSG:3035")
    result = align_fraction(pixels, grid, [source], tmp_path / "projected.tif", "irrigation", "fraction")
    assert result.irrigation.iloc[0] == pytest.approx(.3)


def diagnostic_frame(n=12):
    return pd.DataFrame({"row": np.arange(n), "col": np.zeros(n, dtype=int),
        "primary_eligible": np.arange(n) % 2 == 0, "var_stable": True,
        "residual_whiteness_bootstrap_status": "ok", "residual_whiteness_bootstrap_p": .2,
        "residual_max_abs_corr": .15, "residual_lag1_max_median_abs_autocorr": .1,
        "residual_crosslag_max_abs_corr": .2})


def test_invalid_whiteness_is_not_a_rejection_and_qc_failures_are_retained(tmp_path):
    path = tmp_path / "qc.csv"
    frame = diagnostic_frame(4)
    frame.loc[1, "residual_whiteness_bootstrap_status"] = "failed"
    frame.loc[1, "residual_whiteness_bootstrap_p"] = .001
    frame.loc[3, "residual_whiteness_bootstrap_p"] = .01
    frame.to_csv(path, index=False)
    result = load_diagnostics(path, "primary_eligible")
    assert len(result) == 4
    np.testing.assert_allclose(result.whiteness_rejected, [0, np.nan, 0, 1])
    assert result.whiteness_valid.tolist() == [1, 0, 1, 1]
    frame.loc[1, "primary_eligible"] = True
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="QC-passing"):
        load_diagnostics(path, "primary_eligible")


def population_frame(n=24):
    irrigation = np.linspace(0, .6, n)
    pixels = pd.DataFrame({"row": np.arange(n), "col": 0, "easting": np.arange(n)*100000 + 1,
        "northing": 1, "qc_pass": (np.arange(n) < n//2).astype(float), "point_stable": True,
        "irrigation_fraction": irrigation, "cropland_fraction": .8,
        "aridity_index": np.random.default_rng(4).uniform(.2, 1, n),
        "structural_correlation": irrigation, "lag1_correlation": irrigation/2,
        "crosslag_correlation": irrigation/3, "whiteness_rejected": (irrigation > .3).astype(float),
        "whiteness_p": 1-irrigation, "whiteness_valid": 1.})
    effects = pd.DataFrame({"row": pixels.row, "col": 0, "precipitation": 1-irrigation,
        "shallow_soil_moisture": irrigation-.3, "deep_soil_moisture": -irrigation})
    return prepare_population(pixels, effects, .5, (.01, .2))


def test_effects_preserve_zeros_missingness_signed_magnitude_and_selection():
    frame = population_frame()
    assert frame.primary_effect_eligible.sum() == 12
    assert frame.stable_effect_eligible.sum() == 24
    assert frame.abs_deep_soil_moisture.iloc[-1] == .6
    assert frame.deep_soil_moisture.iloc[-1] == -.6
    assert frame.stable_effect_eligible.iloc[0]  # Pruned zero is not missing.
    original = frame.drop(columns=["precipitation", "shallow_soil_moisture", "deep_soil_moisture"])
    effects = frame[["row", "col", "precipitation", "shallow_soil_moisture", "deep_soil_moisture"]].copy()
    effects.loc[2, "precipitation"] = np.nan
    result = prepare_population(original, effects, .5, (.01, .2))
    assert not result.effects_complete.iloc[2]
    assert result.irrigation_eligible.iloc[2]


def test_aridity_join_checks_coordinates_and_does_not_drop_qc_failures(tmp_path):
    path = tmp_path / "aridity.csv"
    pixels = pd.DataFrame({"row": [1, 2], "col": [0, 0], "longitude": [-5., -5.], "latitude": [40., 39.]})
    ai = pixels.iloc[:1].assign(aridity_index=.5)
    ai.to_csv(path, index=False)
    result = join_aridity(pixels, path)
    assert len(result) == 2 and result.aridity_index.isna().sum() == 1
    ai.longitude = 5
    ai.to_csv(path, index=False)
    with pytest.raises(ValueError, match="coordinates"):
        join_aridity(pixels, path)


def test_spatial_inference_not_inflated_by_duplicate_pixels_and_correct_populations():
    frame = population_frame()
    args = dict(sizes=[100.], has_aridity=False, bootstrap=39, permutations=99, min_blocks=8, seed=7)
    tests, blocks = trend_tests(frame, **args)
    repeated, _ = trend_tests(pd.concat([frame]*5, ignore_index=True), **args)
    np.testing.assert_allclose(tests.rho, repeated.rho, equal_nan=True)
    np.testing.assert_allclose(tests.p_permutation, repeated.p_permutation, equal_nan=True)
    np.testing.assert_allclose(tests.ci_low, repeated.ci_low, equal_nan=True)
    qc = tests.loc[tests.metric.eq("qc_pass")].iloc[0]
    assert qc.rho < -.8 and qc.n_blocks == 24 and qc.p_permutation == .01
    assert np.isnan(qc.p_bh)
    primary = tests.loc[tests.metric.eq("abs_precipitation") & tests.population.eq("primary_qc")].iloc[0]
    stable = tests.loc[tests.metric.eq("abs_precipitation") & tests.population.eq("stable_all")].iloc[0]
    assert primary.n_blocks == 12 and stable.n_blocks == 24
    assert primary.p_bh >= primary.p_permutation
    qc_blocks = blocks.loc[blocks.metric.eq("qc_pass")]
    assert qc_blocks.outcome.sum() == 12


def test_partial_rank_test_removes_a_known_common_gradient():
    rng = np.random.default_rng(31)
    z = rng.uniform(size=160)
    x = z + rng.normal(0, .06, len(z))
    y = z + rng.normal(0, .06, len(z))
    args = dict(bootstrap=39, permutations=199, min_blocks=10)
    unadjusted = association_test(x, y, np.empty((len(x), 0)), rng=np.random.default_rng(1), **args)
    adjusted = association_test(x, y, z[:, None], rng=np.random.default_rng(1), **args)
    assert unadjusted["rho"] > .8 and unadjusted["p_permutation"] == .005
    assert abs(adjusted["rho"]) < .25 and adjusted["p_permutation"] > .05
    constant = association_test(np.ones(len(x)), y, z[:, None], rng=np.random.default_rng(1), **args)
    assert np.isnan(constant["p_permutation"]) and constant["status"] != "ok"


def test_adjusted_and_matched_tests_use_same_valid_pixels_and_block_means():
    frame = population_frame()
    frame.loc[:3, "aridity_index"] = np.nan
    tests, blocks = trend_tests(frame, [100.], True, 19, 19, 8, 12)
    selected = tests.loc[tests.metric.eq("qc_pass")].set_index("adjustment")
    assert selected.loc["unadjusted", "n_pixels"] == 24
    assert selected.loc["adjusted", "n_pixels"] == selected.loc["aridity_matched_unadjusted", "n_pixels"] == 20
    assert selected.loc["adjusted", "controls"] == "aridity_index"
    paired = pd.concat([frame, frame.assign(irrigation_fraction=0, qc_pass=0)], ignore_index=True)
    _, blocks = trend_tests(paired, [100.], False, 19, 19, 8, 12)
    qc = blocks.loc[blocks.metric.eq("qc_pass")]
    np.testing.assert_allclose(qc.irrigation_fraction, frame.irrigation_fraction.to_numpy()[np.argsort([f"{i}:0" for i in range(24)])]/2)
    assert set(qc.outcome) == {0, .5}


def test_bin_denominators_and_too_few_blocks():
    frame = population_frame()
    frame.loc[23, "whiteness_rejected"] = np.nan
    result = bin_summaries(frame, 100, 19, 5, 2)
    high = result.loc[result.bin.eq("High")].set_index("metric")
    assert high.loc["qc_pass", "n_pixels"] == 16
    assert high.loc["whiteness_rejected", "n_pixels"] == 15
    tiny = result.loc[result.bin.eq("Little/no") & result.metric.eq("qc_pass")].iloc[0]
    assert tiny.status == "too_few_blocks" and np.isnan(tiny.ci_low)


def test_gmia_archive_extraction_never_uses_member_path(tmp_path, monkeypatch):
    archive = tmp_path / "gmia_v5_aei_pct_asc.zip"
    # A tiny invalid grid checks safe extraction and global grid validation.
    text = "ncols 1\nnrows 1\nxllcorner -180\nyllcorner -90\ncellsize 0.083333333333\nNODATA_value -9\n-9\n"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../../escape.asc", text)
    monkeypatch.setattr("confoundry.irrigation_covariates.download", lambda *a: None)
    with pytest.raises(ValueError, match="global"):
        download_gmia(tmp_path)
    assert (tmp_path / "gmia_v5_aei_pct.asc").exists()
    assert not (tmp_path.parent / "escape.asc").exists()


def test_download_cache_detects_corruption(tmp_path, monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"valid bytes"

    def get(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr("confoundry.irrigation_covariates.requests.get", get)
    path = tmp_path / "tile.tif"
    download("https://example.org/tile", path)
    download("https://example.org/tile", path)
    assert len(calls) == 1
    path.write_bytes(b"corrupted")
    download("https://example.org/tile", path)
    assert len(calls) == 2 and path.read_bytes() == b"valid bytes"


def test_worldcover_selects_only_tiles_intersecting_original_footprints(tmp_path, monkeypatch):
    from shapely.geometry import box, mapping
    ref = tmp_path / "ref.tif"
    raster(ref, [[0]], from_origin(-8.5, 41, .05, .05))
    requested = []

    def fake_download(url, path):
        requested.append(url)
        if path.suffix == ".geojson":
            features = [dict(type="Feature", properties={"ll_tile": tile}, geometry=mapping(bounds))
                        for tile, bounds in [("N39W009", box(-9, 39, -6, 42)),
                                             ("N39E003", box(3, 39, 6, 42))]]
            path.write_text(json.dumps(dict(type="FeatureCollection", features=features)))
        else:
            path.write_bytes(b"fixture")

    monkeypatch.setattr("confoundry.irrigation_covariates.download", fake_download)
    result = download_worldcover(pd.DataFrame({"row": [0], "col": [0]}), ref, tmp_path)
    assert [p.name for p in result] == ["ESA_WorldCover_10m_2021_v200_N39W009_Map.tif"]
    assert len(requested) == 2 and "/v200/2021/map/" in requested[1]


@pytest.mark.parametrize("use_db", [False, True])
def test_cli_end_to_end_outputs_provenance_all_fitted_and_original_reference_lookup(tmp_path, use_db):
    ref, gmia, crop = [tmp_path / n for n in ["ref.tif", "gmia.tif", "crop.tif"]]
    transform = from_origin(-9, 44, .6, .6)
    raster(ref, np.zeros((12, 1)), transform)
    raster(gmia, np.linspace(0, 60, 12)[:, None], transform, nodata=-9)
    raster(crop, np.full((12, 1), 40), transform, nodata=0)
    qc, effects, ai, source_db = [tmp_path / n for n in ["qc.csv", "effects.csv", "ai.csv", "source.duckdb"]]
    diagnostic_frame().to_csv(qc, index=False)
    pd.DataFrame({"row": np.arange(12), "col": 0, "aridity_index": np.random.default_rng(17).uniform(.2, 1, 12)}).to_csv(ai, index=False)
    rows = []
    for i in range(12):
        for j, name in enumerate(SOURCES):
            rows.append(dict(row=i, col=0, source=name, target="ndvi_resid", horizon=12,
                             scaled_cumulative_total_effect=(i-3)*(j+1)/20))
    effects_frame = pd.DataFrame(rows)
    effects_frame.to_csv(effects, index=False)
    db = tmp_path / "effects.duckdb"
    with duckdb.connect(str(db)) as connection:
        connection.execute("CREATE TABLE pixel_varlingam_effects AS SELECT * FROM effects_frame")
    with duckdb.connect(str(source_db)) as connection:
        connection.execute("CREATE TABLE geotiff_catalog(variable_name VARCHAR, root_dir VARCHAR, file_name VARCHAR, year INTEGER, month INTEGER)")
        connection.execute("INSERT INTO geotiff_catalog VALUES ('modis_ndvi', ?, 'ref.tif', 2005, 1)", [str(tmp_path)])
    out = tmp_path / "out"
    args = ["--effects-db", str(db)] if use_db else ["--effects-csv", str(effects)]
    args += ["--qc-csv", str(qc), "--source-db", str(source_db), "--irrigation-raster", str(gmia),
             "--cropland-raster", str(crop), "--aridity-pixels", str(ai), "--output-dir", str(out),
             "--sensitivity-block-km", "", "--bootstrap-samples", "19", "--permutations", "19", "--min-blocks", "4"]
    if use_db:
        args += ["--no-plots"]
    result = CliRunner().invoke(varlingam_irrigation_stratification, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    saved = pd.read_csv(out / "pixel_irrigation_diagnostics.csv")
    assert len(saved) == 12 and saved.qc_pass.sum() == 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["stable_effect_eligible"] == 12
    assert manifest["reference_grid"]["path"] == str(ref)
    assert str(gmia) in manifest["input_sha256"]
    assert (out / "irrigation_trends_adjusted.tex").exists()
    assert (out / "irrigation_diagnostics.tex").exists()
    assert (out / "references.bib").exists()
    if not use_db:
        for stem in ["irrigation_diagnostics", "irrigation_effects", "irrigation_associations", "irrigation_coverage"]:
            assert (out / (stem + ".pdf")).stat().st_size > 1000
            assert (out / (stem + ".png")).stat().st_size > 1000
    before = (out / "manifest.json").read_bytes()
    repeat = CliRunner().invoke(varlingam_irrigation_stratification, args)
    assert repeat.exit_code != 0 and "never overwritten" in repeat.output
    assert (out / "manifest.json").read_bytes() == before
