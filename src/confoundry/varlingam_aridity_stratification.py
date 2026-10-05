"""Stratify saved VAR-LiNGAM responses by an external aridity climatology.

Run ``python -m confoundry.varlingam_aridity_stratification --help``.
No models are refitted. See docs/varlingam_aridity_stratification.md for
estimands, assumptions, input conventions, and interpretation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import click
import duckdb
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform, transform_bounds
from rasterio.windows import Window, bounds, from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


FIGSHARE_API = "https://api.figshare.com/v2/articles/7504448/versions/5"
DATASET_DOI = "10.6084/m9.figshare.7504448.v5"
PAPER_DOI = "10.1038/s41597-022-01493-1"
BIBTEX = """@article{zomer2022aridity,
  author = {Zomer, Robert J. and Xu, Jianchu and Trabucco, Antonio},
  title = {Version 3 of the Global Aridity Index and Potential Evapotranspiration Database},
  journal = {Scientific Data},
  volume = {9},
  pages = {409},
  year = {2022},
  doi = {10.1038/s41597-022-01493-1}
}

@misc{trabucco2022globalai,
  author = {Trabucco, Antonio and Zomer, Robert},
  title = {Global Aridity Index and Potential Evapotranspiration (ET0) Database: Version 3},
  publisher = {figshare},
  year = {2022},
  doi = {10.6084/m9.figshare.7504448.v5},
  note = {Dataset version 5; climatological baseline 1970--2000}
}
"""
SOURCES = {
    "precipitation_resid": "precipitation",
    "soil_moisture_7_to_28_cm_resid": "shallow_soil_moisture",
    "soil_moisture_28_to_100_cm_resid": "deep_soil_moisture",
    "temperature_resid": "temperature",
    "evaporation_resid": "potential_evaporation",
}
LABELS = {
    "precipitation": "Precipitation",
    "shallow_soil_moisture": "Soil moisture (7–28 cm)",
    "deep_soil_moisture": "Soil moisture (28–100 cm)",
    "temperature": "2 m air temperature",
    "potential_evaporation": "Potential evaporation",
    "water_sum": "Water response: sum of magnitudes",
    "atmospheric_sum": "Atmospheric response: sum of magnitudes",
    "water_mean": "Mean water-response magnitude",
    "atmospheric_mean": "Mean atmospheric-response magnitude",
    "water_atmospheric_ratio": "Water / atmospheric magnitude (W/A)",
    "water_share": "Water-response share (group means)",
}
SIGNED = list(SOURCES.values())
METRICS = list(LABELS)
STYLE = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
         "axes.spines.top": False, "axes.spines.right": False}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def annual_ai_name(name: str) -> bool:
    """Select annual AI, never a monthly stack or ET0-only layer."""
    name = Path(name).name.lower()
    return (bool(re.search(r"(^|[^a-z])ai([^a-z]|$)", name))
            and bool(re.search(r"annual|year|(^|[_\W])yr([_\W]|$)", name))
            and "month" not in name)


def download_aridity(cache: Path, archive_name: str | None = None) -> tuple[Path, dict]:
    """Download only the annual AI file from a pinned Figshare release.

    Metadata and checksums are cached. Extraction copies one selected member
    to a fixed path and cannot follow paths contained in a ZIP archive.
    """
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "figshare_7504448_v5.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        response = requests.get(FIGSHARE_API, timeout=(15, 90))
        response.raise_for_status()
        metadata = response.json()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if metadata.get("version") != 5:
        raise ValueError("Figshare metadata is not the requested version 5.")
    files = metadata.get("files", [])
    candidates = [f for f in files if
                  (f["name"] == archive_name if archive_name else annual_ai_name(f["name"]))
                  and Path(f["name"]).suffix.lower() in {".zip", ".tif", ".tiff"}]
    if len(candidates) != 1:
        names = ", ".join(f["name"] for f in files)
        raise ValueError("Cannot uniquely select the annual AI download. "
                         "Use --aridity-archive-name with its exact filename. "
                         f"Available files: {names}")
    entry = candidates[0]
    expected = entry.get("computed_md5") or entry.get("supplied_md5")
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{32}", expected):
        raise ValueError("The selected Figshare file has no valid MD5 checksum.")
    local = cache / Path(entry["name"]).name
    if not local.exists() or digest(local, "md5") != expected.lower():
        click.echo(f"Downloading {entry['name']} ({entry.get('size', 0):,} bytes)")
        partial = local.with_suffix(local.suffix + ".part")
        with requests.get(entry["download_url"], stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if digest(partial, "md5") != expected.lower():
            raise ValueError(f"Download checksum failed: {partial}")
        partial.replace(local)
    raster = local
    member = None
    if local.suffix.lower() == ".zip":
        with zipfile.ZipFile(local) as archive:
            tiffs = [n for n in archive.namelist()
                     if Path(n).suffix.lower() in {".tif", ".tiff"}
                     and not n.startswith("__MACOSX/")]
            selected = [n for n in tiffs if annual_ai_name(n)]
            if len(selected) != 1:
                # The archive was explicitly selected as annual AI. Some
                # releases call the sole raster simply "ai.tif".
                selected = tiffs if len(tiffs) == 1 else selected
            if len(selected) != 1:
                raise ValueError(f"Ambiguous annual AI raster in archive: {tiffs}")
            member = selected[0]
            raster = cache / "annual_ai_v3.tif"
            partial = raster.with_suffix(".tif.part")
            with archive.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(raster)
    return raster, {"dataset_doi": DATASET_DOI, "paper_doi": PAPER_DOI,
                    "metadata_url": FIGSHARE_API, "file": entry,
                    "archive_member": member, "license": metadata.get("license"),
                    "baseline": "1970–2000", "scale": 0.0001}


def validate_coordinates(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("row", "col"):
        if column not in frame:
            raise ValueError(f"{name} is missing {column}.")
        value = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(value) | (value < 0) | (value != np.floor(value))).any():
            raise ValueError(f"{name} must have finite, nonnegative integer {column} values.")
        frame[column] = value.astype("int64")
    return frame


def load_qc(path: Path, column: str) -> pd.DataFrame:
    qc = validate_coordinates(pd.read_csv(path), "QC table")
    if column not in qc:
        raise ValueError(f"QC table is missing {column}.")
    if qc.duplicated(["row", "col"]).any():
        raise ValueError("QC table has duplicate pixel coordinates.")
    value = qc[column].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False,
         "1.0": True, "0.0": False, "yes": True, "no": False})
    if value.isna().any():
        raise ValueError(f"QC column {column} contains missing or unrecognized booleans.")
    qc["qc_pass"] = value.astype(bool)
    return qc[["row", "col", "qc_pass"]].sort_values(["row", "col"]).reset_index(drop=True)


def load_effects(db: Path | None, csv: Path | None, table: str,
                 target: str, horizon: int, evaporation_multiplier: int) -> pd.DataFrame:
    if db is not None:
        quoted = '"' + table.replace('"', '""') + '"'
        with duckdb.connect(str(db), read_only=True) as connection:
            effects = connection.execute(
                f"SELECT * FROM {quoted} WHERE target = ? AND horizon = ? "
                "AND source IN (?, ?, ?, ?, ?)",
                [target, horizon, *SOURCES]).fetchdf()
    else:
        effects = pd.read_csv(csv)
    required = {"row", "col", "source", "target", "horizon", "scaled_cumulative_total_effect"}
    missing = required - set(effects)
    if missing:
        raise ValueError(f"Effects table is missing columns: {sorted(missing)}")
    effects = effects.loc[(effects.target == target) & (effects.horizon == horizon)
                          & effects.source.isin(SOURCES)].copy()
    if "error" in effects:
        effects = effects.loc[effects.error.isna() | effects.error.eq("")]
    if "model_type" in effects and not effects.model_type.eq("varlingam").all():
        raise ValueError("Selected effects contain models other than VAR-LiNGAM.")
    effects = validate_coordinates(effects, "Effects table")
    if effects.empty:
        raise ValueError("No matching cumulative effects were found.")
    if effects.duplicated(["row", "col", "source"]).any():
        raise ValueError("Duplicate effects for a pixel/source; select a table with one point model.")
    effects["metric"] = effects.source.map(SOURCES)
    value = pd.to_numeric(effects.scaled_cumulative_total_effect, errors="raise")
    effects["value"] = value.where(np.isfinite(value))
    effects.loc[effects.source.eq("evaporation_resid"), "value"] *= evaporation_multiplier
    wide = effects.pivot(index=["row", "col"], columns="metric", values="value")
    wide = wide.reindex(columns=SIGNED)
    # Preserve scale factors and raw derivatives for interpretation audits.
    for column in ("cumulative_total_effect", "source_q_delta", "target_q_delta"):
        if column in effects:
            extra = effects.pivot(index=["row", "col"], columns="metric", values=column)
            wide = wide.join(extra.add_prefix(column + "__"))
    return wide.reset_index().rename_axis(columns=None)


def align_aridity(qc: pd.DataFrame, reference_path: Path, aridity_path: Path,
                  scale: float, band: int, min_coverage: float,
                  output_path: Path) -> pd.DataFrame:
    """Average valid AI within reference cells; read only the study window."""
    with rasterio.open(reference_path) as reference, rasterio.open(aridity_path) as source:
        if reference.crs is None or source.crs is None:
            raise ValueError("Both reference and aridity rasters require a CRS.")
        if (reference.transform.b != 0 or reference.transform.d != 0
                or source.transform.b != 0 or source.transform.d != 0):
            raise ValueError("Use north-up, unrotated rasters for footprint averaging.")
        if (qc.row >= reference.height).any() or (qc.col >= reference.width).any():
            raise ValueError("QC pixel indices fall outside the reference raster.")
        if not 1 <= band <= source.count:
            raise ValueError("Aridity band is outside the source raster.")
        r0, c0 = int(qc.row.min()), int(qc.col.min())
        height, width = int(qc.row.max()) - r0 + 1, int(qc.col.max()) - c0 + 1
        window = Window(c0, r0, width, height)
        dst_transform = reference.window_transform(window)
        roi = transform_bounds(reference.crs, source.crs,
                               *bounds(window, reference.transform), densify_pts=21)
        src_window = from_bounds(*roi, transform=source.transform)
        left, top = np.floor([src_window.col_off, src_window.row_off]).astype(int) - 1
        right = int(np.ceil(src_window.col_off + src_window.width)) + 1
        bottom = int(np.ceil(src_window.row_off + src_window.height)) + 1
        src_window = Window(left, top, right - left, bottom - top)
        if src_window.width * src_window.height > 100_000_000 or width * height > 20_000_000:
            raise ValueError("Study window is unexpectedly large; check the reference grid.")
        array = source.read(band, window=src_window, boundless=True, masked=True,
                            out_dtype="float64").filled(np.nan)
        array *= scale  # Explicit raw-value scale; never silently apply TIFF metadata twice.
        valid = np.isfinite(array) & (array >= 0)
        array[~valid] = np.nan
        aligned = np.full((height, width), np.nan, dtype="float64")
        coverage = np.zeros((height, width), dtype="float64")
        arguments = dict(src_transform=source.window_transform(src_window),
                         src_crs=source.crs, dst_transform=dst_transform,
                         dst_crs=reference.crs, resampling=Resampling.average)
        reproject(array, aligned, src_nodata=np.nan, dst_nodata=np.nan, **arguments)
        reproject(valid.astype("float64"), coverage, dst_nodata=0, **arguments)
        aligned[coverage < min_coverage] = np.nan
        with rasterio.open(output_path, "w", driver="GTiff", width=width, height=height,
                           count=2, dtype="float64", crs=reference.crs,
                           transform=dst_transform, nodata=np.nan, compress="deflate") as dest:
            dest.write(aligned, 1)
            dest.write(coverage, 2)
            dest.set_band_description(1, "Annual AI; larger values are wetter")
            dest.set_band_description(2, "Fraction of footprint with valid AI")
            dest.update_tags(reference_row_offset=r0, reference_col_offset=c0,
                             raw_aridity_scale=scale, aggregation="GDAL average")
        result = qc.copy()
        rr, cc = qc.row.to_numpy() - r0, qc.col.to_numpy() - c0
        result["aridity_index"] = aligned[rr, cc]
        result["aridity_coverage"] = coverage[rr, cc]
        x, y = rasterio.transform.xy(reference.transform, qc.row, qc.col)
        result["longitude"], result["latitude"] = transform(reference.crs, "EPSG:4326", x, y)
        result["easting"], result["northing"] = transform(reference.crs, "EPSG:3035", x, y)
    return result


def assign_bins(pixels: pd.DataFrame, n_quantiles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fix quantile boundaries using ALL fitted pixels with valid AI, before QC."""
    pixels = pixels.copy()
    values = pixels.aridity_index.dropna()
    if values.empty:
        raise ValueError("No fitted pixels overlap valid aridity data.")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_quantiles + 1)))
    if len(edges) == 1:
        cuts = np.array([-np.inf, np.inf])
    else:
        cuts = np.r_[-np.inf, edges[1:-1], np.inf]
    # Right-closed quantiles avoid splitting ties. Constant AI produces one bin.
    labels = [f"Q{i + 1}" for i in range(len(cuts) - 1)]
    pixels["quantile"] = pd.cut(pixels.aridity_index, cuts, labels=labels, right=True)
    definitions = [dict(scheme="quantile", bin=label, order=i,
                        lower=cuts[i], upper=cuts[i + 1], closure="right")
                   for i, label in enumerate(labels)]
    # Follow the classification printed in the chosen dataset's data descriptor
    # (hyperarid cutoff 0.03; other conventions sometimes use 0.05).
    thresholds = [0, 0.03, 0.20, 0.50, 0.65, np.inf]
    names = ["Hyperarid", "Arid", "Semiarid", "Dry subhumid", "Humid"]
    pixels["climate_class"] = pd.cut(pixels.aridity_index, thresholds, labels=names, right=False)
    definitions.extend(dict(scheme="climate_class", bin=label, order=i,
                            lower=thresholds[i], upper=thresholds[i + 1], closure="left")
                       for i, label in enumerate(names))
    return pixels, pd.DataFrame(definitions)


def add_response_indices(pixels: pd.DataFrame) -> pd.DataFrame:
    pixels = pixels.copy()
    complete = np.isfinite(pixels[SIGNED]).all(axis=1)
    pixels["effects_complete"] = complete
    water = pixels[SIGNED[:3]].abs().sum(axis=1, min_count=3).where(complete)
    atmosphere = pixels[SIGNED[3:]].abs().sum(axis=1, min_count=2).where(complete)
    pixels["water_sum"], pixels["atmospheric_sum"] = water, atmosphere
    pixels["water_mean"], pixels["atmospheric_mean"] = water / 3, atmosphere / 2
    # No epsilon and no replacement of absent effects by zero.
    pixels["water_atmospheric_ratio"] = (water / atmosphere.where(atmosphere > 0)).replace(
        [np.inf, -np.inf], np.nan)
    total = pixels.water_mean + pixels.atmospheric_mean
    pixels["water_share"] = pixels.water_mean / total.where(total > 0)
    pixels["ratio_undefined"] = complete & pixels.water_atmospheric_ratio.isna()
    pixels["share_undefined"] = complete & pixels.water_share.isna()
    pixels["analysis_eligible"] = pixels.qc_pass & pixels.aridity_index.notna() & complete
    return pixels


def block_ids(pixels: pd.DataFrame, size_km: float) -> np.ndarray:
    positions = np.floor(pixels[["easting", "northing"]].to_numpy() / (size_km * 1000)).astype("int64")
    return np.array([f"{x}:{y}" for x, y in positions])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Arithmetic in a ratio can create machine-precision variation in an
    # otherwise constant index. Do not turn this rounding noise into ranks.
    for values in (x, y):
        values = np.asarray(values, dtype=float)
        if len(values) < 3 or np.ptp(values) <= 16 * np.finfo(float).eps * np.max(np.abs(values)):
            return np.nan
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fdr_bh(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg within the stated family; undefined tests remain NA."""
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return output


def trend_tests(pixels: pd.DataFrame, sizes: list[float], primary_size: float,
                bootstrap: int, permutations: int, min_blocks: int,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test between-block association using equally weighted block medians.

    Permutations require exchangeable blocks under the null. Adjacent blocks
    are not guaranteed independent; sizes are sensitivity analyses, not a
    data-driven search for the most significant result.
    """
    rows, block_tables = [], []
    metrics = ["water_share", *SIGNED, "water_atmospheric_ratio"]
    for size in sizes:
        data = pixels.copy()
        data["block"] = block_ids(data, size)
        for metric in metrics:
            finite = data.loc[np.isfinite(data[metric])]
            groups = finite.groupby("block", sort=True)
            block = groups[["aridity_index", metric]].median()
            block["n_pixels"] = groups.size()
            block = block.rename(columns={metric: "response"}).reset_index()
            block["metric"], block["block_km"] = metric, size
            block_tables.append(block)
            x, y = block.aridity_index.to_numpy(), block.response.to_numpy()
            observed = spearman(x, y)
            result = dict(metric=metric, block_km=size, primary_block_size=size == primary_size,
                          family="primary" if metric == "water_share" else
                          "signed_effects" if metric in SIGNED else "exploratory_ratio",
                          n_pixels=len(finite), n_blocks=len(block), rho=observed,
                          pixel_rho_descriptive=spearman(finite.aridity_index, finite[metric]),
                          ci_low=np.nan, ci_high=np.nan, p_permutation=np.nan,
                          bootstrap_valid=0, status="too_few_blocks")
            # Stable stream for each size/metric, unaffected by loop ordering.
            stream = int.from_bytes(hashlib.sha256(f"{size:g}:{metric}".encode()).digest()[:4], "little")
            rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
            if len(block) >= min_blocks and np.isfinite(observed):
                xr = pd.Series(x).rank().to_numpy()
                yr = pd.Series(y).rank().to_numpy()
                xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
                yr = (yr - yr.mean()) / np.linalg.norm(yr - yr.mean())
                exceed = sum(abs(float(xr @ rng.permutation(yr))) >= abs(observed) - 1e-12
                             for _ in range(permutations))
                result["p_permutation"] = (exceed + 1) / (permutations + 1)
                estimates = []
                for _ in range(bootstrap):
                    indices = rng.integers(0, len(block), len(block))
                    estimates.append(spearman(x[indices], y[indices]))
                estimates = np.asarray(estimates)
                estimates = estimates[np.isfinite(estimates)]
                result["bootstrap_valid"] = len(estimates)
                result["status"] = "ok" if len(estimates) >= 0.95 * bootstrap else "insufficient_valid_bootstrap"
                if result["status"] == "ok":
                    result["ci_low"], result["ci_high"] = np.quantile(estimates, [0.025, 0.975])
            elif len(block) >= min_blocks:
                result["status"] = "constant_block_values"
            rows.append(result)
    tests = pd.DataFrame(rows)
    tests["p_bh_signed_family"] = np.nan
    for size in sizes:
        mask = tests.block_km.eq(size) & tests.family.eq("signed_effects")
        tests.loc[mask, "p_bh_signed_family"] = fdr_bh(tests.loc[mask, "p_permutation"].to_numpy())
    return tests, pd.concat(block_tables, ignore_index=True)


def bin_summaries(pixels: pd.DataFrame, definitions: pd.DataFrame, block_km: float,
                  bootstrap: int, min_blocks: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage, summaries = [], []
    pixels = pixels.copy()
    pixels["block"] = block_ids(pixels, block_km)
    for definition in definitions.itertuples(index=False):
        group = pixels.loc[pixels[definition.scheme] == definition.bin]
        eligible = group.loc[group.analysis_eligible]
        coverage.append(dict(scheme=definition.scheme, bin=definition.bin, order=definition.order,
                             n_fitted=len(group), n_qc_pass=int(group.qc_pass.sum()),
                             n_analyzed=len(eligible),
                             qc_pass_fraction=group.qc_pass.mean(),
                             ai_min=group.aridity_index.min(), ai_max=group.aridity_index.max(),
                             ai_median=group.aridity_index.median()))
        for metric in METRICS:
            finite = eligible.loc[np.isfinite(eligible[metric])]
            values = finite[metric].to_numpy()
            blocks = [g[metric].to_numpy() for _, g in finite.groupby("block", sort=True)]
            row = dict(scheme=definition.scheme, bin=definition.bin, order=definition.order,
                       metric=metric, n_pixels=len(values), n_blocks=len(blocks),
                       median=np.nan, q05=np.nan, q25=np.nan, q75=np.nan, q95=np.nan,
                       positive_fraction=np.nan, zero_fraction=np.nan,
                       ci_low=np.nan, ci_high=np.nan, ci_status="too_few_blocks")
            if len(values):
                row.update(zip(["q05", "q25", "median", "q75", "q95"],
                               np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])))
                row["positive_fraction"] = np.mean(values > 0)
                row["zero_fraction"] = np.mean(values == 0)
            if len(blocks) >= min_blocks:
                # Same block draws across metrics when their valid sets match.
                stream = int.from_bytes(hashlib.sha256(
                    f"{definition.scheme}:{definition.bin}".encode()).digest()[:4], "little")
                rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
                estimates = [np.median(np.concatenate([blocks[j] for j in
                             rng.integers(0, len(blocks), len(blocks))])) for _ in range(bootstrap)]
                row["ci_low"], row["ci_high"] = np.quantile(estimates, [0.025, 0.975])
                row["ci_status"] = "ok"
            summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(coverage)


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)


def make_figures(pixels: pd.DataFrame, summaries: pd.DataFrame, coverage: pd.DataFrame,
                 blocks: pd.DataFrame, output: Path, block_km: float, horizon: int) -> None:
    def ratio_axis(ax):
        ax.set_yscale("symlog", linthresh=1)
        upper = max(1, ax.get_ylim()[1]) * 1.1
        ax.set_ylim(0, upper)
        if upper <= 3:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))

    def series(ax, subset, metric, color="#276a9f", label=None):
        group = subset.loc[subset.metric == metric].sort_values("order")
        x = group.order.to_numpy()
        ax.plot(x, group["median"], "o-", color=color, label=label, markersize=4)
        ax.vlines(x, group.ci_low, group.ci_high, color=color, linewidth=1.5)
        ax.set_xticks(x, group.bin, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.18)

    with plt.rc_context(STYLE):
        for scheme in ("quantile", "climate_class"):
            subset = summaries.loc[summaries.scheme == scheme]
            cov = coverage.loc[coverage.scheme == scheme].sort_values("order")
            fig, axes = plt.subplots(2, 3, figsize=(10, 6.1), layout="constrained")
            for letter, ax, metric in zip("abcde", axes.flat, SIGNED):
                series(ax, subset, metric)
                ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
                ax.set_title(f"({letter}) {LABELS[metric]}", loc="left")
                ax.set_ylabel(f"Scaled cumulative NDVI response\n(months 0–{horizon})")
            ax = axes.flat[-1]
            ax.bar(cov.order, cov.n_fitted, color="0.82", label="All fitted; valid AI")
            ax.bar(cov.order, cov.n_analyzed, color="#276a9f", label="Included in analysis")
            ax.set_xticks(cov.order, cov.bin, rotation=20, ha="right")
            ax.set_title("(f) Analysis coverage", loc="left")
            ax.set_ylabel("Pixels")
            ax.set_ylim(0, max(1, cov.n_fitted.max()) * 1.3)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(frameon=False)
            fig.supxlabel("Annual aridity index classes (drier → wetter)")
            save_figure(fig, output / f"signed_effects_{scheme}")

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
            series(axes[0], subset, "water_mean", label="Water (3 sources)")
            series(axes[0], subset, "atmospheric_mean", color="#b05a30", label="Atmospheric (2 sources)")
            axes[0].set_ylabel("Mean absolute cumulative response")
            axes[0].legend(frameon=False)
            series(axes[1], subset, "water_share")
            axes[1].set_ylim(-0.03, 1.03)
            axes[1].axhline(0.5, color="0.4", linestyle="--", linewidth=0.7)
            axes[1].set_ylabel("Water-response share (group means)")
            series(axes[2], subset, "water_atmospheric_ratio")
            ratio_axis(axes[2])
            axes[2].set_ylabel("W/A (sum of magnitudes; symlog scale)")
            for letter, ax in zip("abc", axes):
                ax.set_title(f"({letter})", loc="left")
            fig.supxlabel("Annual aridity index classes (drier → wetter)")
            save_figure(fig, output / f"response_regime_{scheme}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
        for ax, metric in zip(axes, ["water_share", "water_atmospheric_ratio"]):
            group = blocks.loc[blocks.block_km.eq(block_km) & blocks.metric.eq(metric)]
            ax.scatter(group.aridity_index, group.response, s=18, alpha=0.65,
                       color="#276a9f", edgecolors="none", rasterized=True)
            ax.set_xlabel("Block median annual AI (drier → wetter)")
            ax.set_ylabel(LABELS[metric])
        axes[0].set_ylim(-0.03, 1.03)
        ratio_axis(axes[1])
        save_figure(fig, output / "block_aridity_relationship")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        valid = pixels.loc[pixels.aridity_index.notna()]
        image = axes[0].scatter(valid.easting / 1000, valid.northing / 1000,
                               c=valid.aridity_index, cmap="BrBG", s=3,
                               marker="s", linewidths=0, rasterized=True)
        fig.colorbar(image, ax=axes[0], label="Annual AI (larger = wetter)", shrink=0.8)
        axes[0].set_title("(a) External aridity climatology", loc="left")
        axes[1].scatter(pixels.easting / 1000, pixels.northing / 1000, s=3,
                        color="0.8", marker="s", linewidths=0, rasterized=True,
                        label="Other fitted pixels")
        selected = pixels.loc[pixels.analysis_eligible]
        axes[1].scatter(selected.easting / 1000, selected.northing / 1000, s=3,
                        color="#276a9f", marker="s", linewidths=0, rasterized=True,
                        label="Included in analysis")
        axes[1].set_title("(b) Analysis coverage", loc="left")
        axes[1].legend(frameon=False, markerscale=3)
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlabel("Easting (km; ETRS89 / LAEA Europe)")
            ax.set_ylabel("Northing (km)")
        save_figure(fig, output / "aridity_coverage_map")


@click.command(context_settings={"show_default": True})
@click.option("--effects-db", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--effects-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--effects-table", default="pixel_varlingam_effects")
@click.option("--qc-csv", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="All fitted pixels, including QC failures, from prepare_varlingam_paper_outputs.")
@click.option("--qc-column", default="primary_eligible")
@click.option("--reference-raster", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Exact raster grid whose zero-based row/col indices were used by gather.")
@click.option("--aridity-raster", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--aridity-scale", type=click.FloatRange(min=0, min_open=True),
              help="Multiplier for raw AI pixels: 0.0001 for Global-AI v3; 1 for physical AI.")
@click.option("--aridity-description", default="User-supplied external annual aridity climatology",
              help="Source, baseline, version and DOI for a local raster; recorded in provenance.")
@click.option("--aridity-band", default=1, type=click.IntRange(min=1))
@click.option("--download-aridity", is_flag=True, help="Fetch annual Global-AI v3, pinned Figshare version 5.")
@click.option("--aridity-archive-name", help="Exact annual AI filename if automatic selection is ambiguous.")
@click.option("--cache-dir", default="data/aridity_climatology", type=click.Path(path_type=Path))
@click.option("--min-aridity-coverage", default=0.5, type=click.FloatRange(0, 1, min_open=True))
@click.option("--target", default="ndvi_resid")
@click.option("--horizon", default=12, type=click.IntRange(min=0), help="Use the saved cumulative response at this horizon.")
@click.option("--evaporation-multiplier", type=click.Choice(["1", "-1"]), default="1",
              help="Explicit sign conversion for signed evaporation effects; magnitudes are unchanged.")
@click.option("--quantiles", default=5, type=click.IntRange(2, 10))
@click.option("--block-km", default=100.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--sensitivity-block-km", default="50,150", help="Comma-separated additional block sizes; empty disables.")
@click.option("--min-blocks", default=10, type=click.IntRange(min=3))
@click.option("--bootstrap", default=999, type=click.IntRange(min=99))
@click.option("--permutations", default=4999, type=click.IntRange(min=99))
@click.option("--seed", default=20261005, type=click.IntRange(min=0))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path), help="New or empty output directory.")
@click.option("--plots/--no-plots", default=True)
def varlingam_aridity_stratification(**options) -> None:
    """Compare saved NDVI responses along an independently mapped aridity gradient.

    Writes pixel data, QC coverage, bin summaries, spatial trend tests,
    publication PNG/PDF figures, LaTeX tables and provenance. Input databases
    are opened read-only. Larger P/ET0 means WETTER, not more arid.
    """
    try:
        run_experiment(**options)
    except (ValueError, OSError, requests.RequestException, duckdb.Error, zipfile.BadZipFile) as exc:
        raise click.ClickException(str(exc)) from exc


def run_experiment(**options) -> None:
    o = options
    if (o["effects_db"] is None) == (o["effects_csv"] is None):
        raise ValueError("Supply exactly one of --effects-db and --effects-csv.")
    if o["download_aridity"] == (o["aridity_raster"] is not None):
        raise ValueError("Supply exactly one of --aridity-raster and --download-aridity.")
    if not o["download_aridity"] and o["aridity_scale"] is None:
        raise ValueError("Set --aridity-scale explicitly for a local raster (0.0001 for Global-AI v3).")
    if o["download_aridity"] and o["aridity_scale"] not in (None, 0.0001):
        raise ValueError("Downloaded Global-AI v3 requires --aridity-scale 0.0001.")
    if o["aridity_scale"] is not None and not np.isfinite(o["aridity_scale"]):
        raise ValueError("Aridity scale must be finite.")
    if not np.isfinite(o["min_aridity_coverage"]):
        raise ValueError("Minimum aridity coverage must be finite.")
    sizes = sorted(set([o["block_km"], *[float(s) for s in o["sensitivity_block_km"].split(",") if s.strip()]]))
    if not all(np.isfinite(s) and s > 0 for s in sizes):
        raise ValueError("All spatial block sizes must be finite and positive.")
    output = o["output_dir"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory to preserve prior results.")
    qc = load_qc(o["qc_csv"], o["qc_column"])
    if qc.empty or not qc.qc_pass.any():
        raise ValueError("QC input has no passing pixels.")
    effects = load_effects(o["effects_db"], o["effects_csv"], o["effects_table"],
                           o["target"], o["horizon"], int(o["evaporation_multiplier"]))
    keys = pd.MultiIndex.from_frame(qc[["row", "col"]])
    if not pd.MultiIndex.from_frame(effects[["row", "col"]]).isin(keys).all():
        raise ValueError("Some effects pixels are absent from the supplied all-fitted QC table.")
    output.mkdir(parents=True, exist_ok=True)
    if o["download_aridity"]:
        raster, source = download_aridity(o["cache_dir"], o["aridity_archive_name"])
        scale = 0.0001
    else:
        raster, scale = o["aridity_raster"], o["aridity_scale"]
        source = {"description": o["aridity_description"], "scale": scale}
    click.echo("Aligning the external climatology and assigning aridity classes...")
    pixels = align_aridity(qc, o["reference_raster"], raster, scale, o["aridity_band"],
                           o["min_aridity_coverage"], output / "aligned_aridity.tif")
    pixels = pixels.merge(effects, how="left", on=["row", "col"], validate="one_to_one")
    pixels = add_response_indices(pixels)
    pixels, definitions = assign_bins(pixels, o["quantiles"])
    selected = pixels.loc[pixels.analysis_eligible].copy()
    if selected.empty:
        raise ValueError("No pixels pass QC and have both valid AI and all five finite effects.")
    pixels["block"] = block_ids(pixels, o["block_km"])
    click.echo(f"Analyzing {len(selected):,} of {len(pixels):,} fitted pixels; "
               f"{len(set(block_ids(selected, o['block_km'])))} primary spatial blocks.")
    summaries, coverage = bin_summaries(pixels, definitions, o["block_km"],
                                       o["bootstrap"], o["min_blocks"], o["seed"])
    click.echo("Testing between-block trends and block-size sensitivity...")
    tests, blocks = trend_tests(selected, sizes, o["block_km"], o["bootstrap"],
                                o["permutations"], o["min_blocks"], o["seed"])
    unavailable = int(tests.status.ne("ok").sum())
    if unavailable:
        click.echo(f"{unavailable} trend results have insufficient blocks, constant values, "
                   "or insufficient bootstrap support; see status in trend_tests.csv.")
    for name, frame in {"pixel_aridity_effects": pixels, "bin_definitions": definitions,
                        "bin_summaries": summaries, "qc_coverage": coverage,
                        "trend_tests": tests, "block_summaries": blocks}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    counts = {"all_fitted": len(pixels), "qc_pass": int(pixels.qc_pass.sum()),
              "missing_aridity_all_fitted": int(pixels.aridity_index.isna().sum()),
              "qc_pass_missing_aridity": int((pixels.qc_pass & pixels.aridity_index.isna()).sum()),
              "qc_pass_incomplete_effects": int((pixels.qc_pass & ~pixels.effects_complete).sum()),
              "analyzed": len(selected), "analyzed_undefined_ratio": int(selected.ratio_undefined.sum()),
              "analyzed_undefined_share": int(selected.share_undefined.sum())}
    if o["plots"]:
        click.echo("Rendering publication figures...")
        make_figures(pixels, summaries, coverage, blocks, output, o["block_km"], o["horizon"])
    write_publication_text(output, counts, summaries, tests, o)
    input_paths = {k: o[k] for k in ("effects_db", "effects_csv", "qc_csv", "reference_raster") if o[k] is not None}
    input_paths["aridity_raster"] = raster
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "options": {k: str(v) if isinstance(v, Path) else v for k, v in o.items()},
                "aridity_source": source, "counts": counts,
                "inputs": {k: {"path": str(v.resolve()), "sha256": digest(v)} for k, v in input_paths.items()},
                "module_sha256": digest(Path(__file__)),
                "versions": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "duckdb", "rasterio", "matplotlib"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Results written to {output.resolve()}")


def write_publication_text(output: Path, counts: dict, summaries: pd.DataFrame,
                           tests: pd.DataFrame, options: dict) -> None:
    def number(value, precision=3):
        return f"{value:.{precision}f}" if np.isfinite(value) else "--"

    primary = tests.loc[tests.primary_block_size]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Response & Blocks & Spearman $\rho$ & $p_{\mathrm{perm}}$ & $p_{\mathrm{BH}}$ \\", r"\midrule"]
    for row in primary.itertuples():
        lines.append(f"{LABELS[row.metric]} & {row.n_blocks} & {number(row.rho)} & "
                     f"{number(row.p_permutation, 4)} & {number(row.p_bh_signed_family, 4)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  "% BH correction applies to the five signed source effects only."])
    (output / "trend_tests.tex").write_text("\n".join(lines) + "\n")
    for scheme in ("quantile", "climate_class"):
        subset = summaries.loc[summaries.scheme.eq(scheme) & summaries.metric.isin(SIGNED)]
        table = subset.pivot(index="bin", columns="metric", values="median")
        ordered = subset[["bin", "order"]].drop_duplicates().sort_values("order").bin
        table = table.reindex(index=ordered, columns=SIGNED)
        # Write the small fragment directly; pandas.to_latex requires the
        # optional Jinja2 package, which is not a Confoundry dependency.
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 "Aridity class & " + " & ".join(LABELS[m].replace("–", "--") for m in SIGNED) + r" \\",
                 r"\midrule"]
        lines.extend(str(label) + " & " + " & ".join(number(v) for v in row) + r" \\"
                     for label, row in table.iterrows())
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (output / f"signed_effects_{scheme}.tex").write_text("\n".join(lines) + "\n")
    horizon = options["horizon"]
    report = f"""# Aridity stratification

Analyzed **{counts['analyzed']:,}** pixels of **{counts['all_fitted']:,}** fitted pixels.
QC passed: {counts['qc_pass']:,}. Among QC-passing pixels, missing AI:
{counts['qc_pass_missing_aridity']:,}; incomplete effects: {counts['qc_pass_incomplete_effects']:,}.
These exclusions can overlap. Undefined W/A among analyzed pixels:
{counts['analyzed_undefined_ratio']:,}; undefined water share: {counts['analyzed_undefined_share']:,}.

## Reading the results

- AI is annual P/ET0: increasing AI means wetter conditions. It is an external
  long-term climate classification, not contemporaneous drought or soil moisture.
- Effects sum months 0–{horizon} inclusive after a one-time source pulse. Use the
  saved locally scaled cumulative effect, not the horizon-specific response.
  Signed source effects are retained, including zero effects from pruning.
- W = |CE_P| + |CE_shallow| + |CE_deep|; A = |CE_T| + |CE_E|. The primary
  response-regime summary is (W/3)/(W/3 + A/2), balancing the numbers of sources.
  W/A is an exploratory secondary summary; zero denominators are missing, never
  stabilized with an arbitrary epsilon. Small positive denominators can still
  produce very large ratios. These are magnitudes of net cumulative responses;
  positive and negative responses over time can cancel before taking absolute values.
- These indices do not partition explained variance or independent causal
  pathways. Total effects can share mediators; potential evaporation includes
  both water and energy controls. The groups are operational summaries.
- A spatial difference can reflect response size, duration, source variability,
  or the local NDVI range. It is not necessarily a difference in response to
  an equal physical perturbation. Exported raw effects and scale factors, when
  available, support this audit. Raw derivatives retain their original signs.
- Evaporation multiplier applied to signed scaled effects: {options['evaporation_multiplier']}.
  Verify the input convention before interpreting a positive pulse as greater
  evaporative demand: native ERA5 evaporation commonly has the opposite sign.

## Statistical scope

The primary test relates block-median water share to block-median AI using
Spearman correlation. Blocks are {options['block_km']:g} km squares in EPSG:3035,
anchored at projection coordinate zero. All blocks receive equal weight in
trend tests. The pixel-level rank correlation is descriptive only. The other
block sizes are sensitivity analyses, not a rule for choosing a favorable p-value.
Two-sided permutation p-values use a plus-one correction. Paired resampling
of block medians gives percentile 95% intervals for the correlation.
The five signed-source tests have a separate Benjamini–Hochberg correction
within each block size. The ratio is exploratory. No correction across
block-size sensitivities is implied.

Permutation validity requires exchangeability of blocks under the null;
block size alone cannot guarantee it. Spatial dependence beyond the chosen
size and broad geographic covariation can remain. Climate, vegetation, soils,
irrigation and land use covary; association with AI does not isolate an aridity
effect or establish causal identification. Compare the sensitivity results.

Bin medians weight pixels equally; their intervals resample whole blocks
within each bin, retaining all pixels and unequal block sizes. These are
pointwise intervals, conditional on saved effects and fixed bin boundaries,
not graph-bootstrap uncertainty or simultaneous bands. At least
{options['min_blocks']} contributing blocks are required for inferential output.
Insufficient/constant cases are labeled in CSV; missing intervals are not zero.
Quintile boundaries (or the requested number of quantiles) use all fitted
pixels with valid AI, before QC. Ties are not split, so bins may be unequal or
fewer than requested. Climate classes follow Zomer et al. (2022), with cutoffs
0.03, 0.20, 0.50 and 0.65 and the wetter class including its lower boundary.
Coverage summaries retain all fitted pixels; missing AI is counted above.

## Figure captions

**signed_effects_*.pdf:** Median locally scaled cumulative NDVI response
through month {horizon} by external aridity class. Vertical lines show pointwise
95% spatial-block bootstrap intervals. Panel (f) shows fitted and analyzed
pixel counts. Increasing class order indicates wetter conditions. Medians
are shown without intervals where too few blocks contribute.

**response_regime_*.pdf:** (a) Median per-source magnitude within each response
group; (b) median water-response share; (c) median W/A, shown on a symmetric
logarithmic scale (linear within [-1, 1]). Indices are calculated per pixel
before taking medians. Vertical lines are spatial-block bootstrap intervals.
Undefined ratios are omitted and counted above. The reference share 0.5
indicates equal mean response magnitude per source in the two groups.

**block_aridity_relationship.pdf:** Each point is a geographic block, using
the median AI and median response over the same valid pixels within that block.
These block summaries, not independent pixels, enter the trend tests.

**aridity_coverage_map.pdf:** External AI averaged over analysis-grid footprints
and the final experiment coverage in a European equal-area projection. Gray
pixels lack QC eligibility, aridity coverage, or complete effects. These maps
describe retained coverage, not absence of environmental sensitivity.

## Sources and provenance

Recommended climatology: Zomer, Xu and Trabucco (2022),
https://doi.org/{PAPER_DOI}; archived dataset https://doi.org/{DATASET_DOI}.
The automatic download is pinned to that release. Local raster provenance is
recorded exactly as supplied in manifest.json; an external file is not assumed
to be Global-AI. Raw AI is averaged by GDAL overlap weights over valid cells;
this is an average of local ratios, not a ratio of aggregated P and ET0.
The aligned GeoTIFF records the original row/column offsets and coverage.
The manifest records parameters, input checksums, software versions and counts.
"""
    (output / "report.md").write_text(report)
    (output / "references.bib").write_text(BIBTEX)


if __name__ == "__main__":
    varlingam_aridity_stratification()
