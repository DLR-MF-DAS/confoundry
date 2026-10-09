"""Test whether irrigation exposure is associated with QC and saved responses.

Run ``python -m confoundry.varlingam_irrigation_stratification --help``.
See docs/varlingam_irrigation_stratification.md for the inferential scope.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import rankdata

from confoundry.irrigation_covariates import (
    GMIA_PAGE, GMIA_URL, align_fraction, download_gmia, download_worldcover, study_grid,
)
from confoundry.varlingam_aridity_stratification import (
    BIBTEX as ARIDITY_BIBTEX, SIGNED, STYLE, block_ids, digest, fdr_bh,
    load_effects, load_qc, save_figure, validate_coordinates,
)
import matplotlib.pyplot as plt  # noqa: E402; imported after selecting the Agg backend

# Label, block/bin aggregation, family, expected direction. Tests are two-sided.
ENDPOINTS = {
    "qc_pass": ("QC pass fraction", "mean", "primary", "negative"),
    "whiteness_rejected": ("Whiteness rejection fraction", "mean", "diagnostics", "positive"),
    "structural_correlation": ("Structural-error correlation", "median", "diagnostics", "positive"),
    "lag1_correlation": ("Lag-1 innovation autocorrelation", "median", "diagnostics", "positive"),
    "crosslag_correlation": ("Cross-lag innovation correlation", "median", "diagnostics", "positive"),
    "precipitation": ("Precipitation: signed effect", "median", "effects", "unspecified"),
    "shallow_soil_moisture": ("Soil moisture (7–28 cm): signed effect", "median", "effects", "unspecified"),
    "deep_soil_moisture": ("Soil moisture (28–100 cm): signed effect", "median", "effects", "unspecified"),
    "abs_precipitation": ("Precipitation: effect magnitude", "median", "effects", "negative"),
    "abs_shallow_soil_moisture": ("Soil moisture (7–28 cm): effect magnitude", "median", "effects", "unspecified"),
    "abs_deep_soil_moisture": ("Soil moisture (28–100 cm): effect magnitude", "median", "effects", "unspecified"),
}
DESCRIPTIVE = {
    "whiteness_valid": ("Valid whiteness calibration fraction", "mean", "availability", "unspecified"),
    "whiteness_p": ("Bootstrap whiteness p-value", "median", "descriptive", "unspecified"),
}
ALL_ENDPOINTS = ENDPOINTS | DESCRIPTIVE
DIAGNOSTIC_FIELDS = {
    "residual_max_abs_corr": "structural_correlation",
    "residual_lag1_max_median_abs_autocorr": "lag1_correlation",
    "residual_crosslag_max_abs_corr": "crosslag_correlation",
}
BIBTEX = r"""@misc{siebert2013gmia,
  author = {Siebert, Stefan and Henrich, Verena and Frenken, Karen and Burke, Jacob},
  title = {Global Map of Irrigation Areas version 5},
  year = {2013},
  publisher = {University of Bonn and Food and Agriculture Organization of the United Nations},
  url = {https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/},
  note = {Area equipped for irrigation around 2005; percentage of total grid-cell area}
}

@misc{zanaga2022worldcover,
  author = {Zanaga, Daniele and Van De Kerchove, Ruben and Daems, Dirk and others},
  title = {{ESA WorldCover} 10 m 2021 v200},
  publisher = {Zenodo},
  year = {2022},
  doi = {10.5281/zenodo.7254221}
}

@article{freedman1983nonstochastic,
  author = {Freedman, David and Lane, David},
  title = {A Nonstochastic Interpretation of Reported Significance Levels},
  journal = {Journal of Business \& Economic Statistics},
  year = {1983}, volume = {1}, number = {4}, pages = {292--298},
  doi = {10.1080/07350015.1983.10509354}
}

@article{benjamini1995controlling,
  author = {Benjamini, Yoav and Hochberg, Yosef},
  title = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  journal = {Journal of the Royal Statistical Society: Series B},
  year = {1995}, volume = {57}, number = {1}, pages = {289--300},
  doi = {10.1111/j.2517-6161.1995.tb02031.x}
}
"""


def rng_for(seed: int, *keys) -> np.random.Generator:
    token = json.dumps([seed, *keys], sort_keys=True).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def numeric(frame: pd.DataFrame, column: str, lower: float, upper: float) -> pd.Series:
    value = pd.to_numeric(frame[column], errors="raise")
    if (value.notna() & (~np.isfinite(value) | (value < lower) | (value > upper))).any():
        raise ValueError(f"Invalid values in {column}; expected [{lower}, {upper}] or missing.")
    return value


def load_diagnostics(path: Path, qc_column: str) -> pd.DataFrame:
    qc = load_qc(path, qc_column)
    source = validate_coordinates(pd.read_csv(path), "QC table")
    required = {*DIAGNOSTIC_FIELDS, "residual_whiteness_bootstrap_status",
                "residual_whiteness_bootstrap_p", "var_stable"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"Use the full pixel_qc.csv from prepare_varlingam_paper_outputs; missing {sorted(missing)}")
    source = source.merge(qc, on=["row", "col"], validate="one_to_one")
    stability = load_qc(path, "var_stable").rename(columns={"qc_pass": "point_stable"})
    source = source.merge(stability, on=["row", "col"], validate="one_to_one")
    for original, new in DIAGNOSTIC_FIELDS.items():
        source[new] = numeric(source, original, 0, 1)
    p = numeric(source, "residual_whiteness_bootstrap_p", 0, 1)
    valid = source.residual_whiteness_bootstrap_status.eq("ok") & p.notna()
    source["whiteness_valid"] = valid.astype(float)
    source["whiteness_p"] = p.where(valid)
    source["whiteness_rejected"] = (p < .05).astype(float).where(valid)
    source["qc_pass"] = source.qc_pass.astype(float)
    # A passed production mask must not include an unavailable primary test.
    if (source.qc_pass.eq(1) & (~valid | p.lt(.05) | ~source.point_stable)).any():
        raise ValueError("QC-passing pixels include invalid/rejected whiteness or unstable models; check QC provenance.")
    return source.sort_values(["row", "col"]).reset_index(drop=True)


def join_aridity(pixels: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    result = pixels.copy()
    if path is None:
        result["aridity_index"] = np.nan
        return result
    ai = validate_coordinates(pd.read_csv(path), "Aridity pixel table")
    if "aridity_index" not in ai or ai.duplicated(["row", "col"]).any():
        raise ValueError("Aridity table needs unique row/col and aridity_index; use pixel_aridity_effects.csv.")
    ai["aridity_index"] = numeric(ai, "aridity_index", 0, np.inf)
    coordinates = [x for x in ["longitude", "latitude"] if x in ai]
    merged = result.merge(ai[["row", "col", "aridity_index", *coordinates]],
                          on=["row", "col"], how="left", validate="one_to_one", suffixes=("", "_ai"))
    for coordinate in coordinates:
        finite = merged[coordinate + "_ai"].notna()
        if not np.allclose(merged.loc[finite, coordinate], merged.loc[finite, coordinate + "_ai"], atol=1e-5, rtol=0):
            raise ValueError("Aridity pixel coordinates do not match the reference raster.")
        merged = merged.drop(columns=coordinate + "_ai")
    return merged


def prepare_population(pixels: pd.DataFrame, effects: pd.DataFrame, minimum_crop: float,
                       cutpoints: tuple[float, float]) -> pd.DataFrame:
    pixels = pixels.merge(effects, on=["row", "col"], how="left", validate="one_to_one")
    pixels["cropland_eligible"] = pixels.cropland_fraction.ge(minimum_crop)
    pixels["irrigation_eligible"] = pixels.cropland_eligible & pixels.irrigation_fraction.notna()
    pixels["effects_complete"] = np.isfinite(pixels[SIGNED[:3]]).all(axis=1)
    pixels["primary_effect_eligible"] = pixels.irrigation_eligible & pixels.qc_pass.eq(1) & pixels.effects_complete
    pixels["stable_effect_eligible"] = pixels.irrigation_eligible & pixels.point_stable & pixels.effects_complete
    for metric in SIGNED[:3]:
        pixels["abs_" + metric] = pixels[metric].abs()
    pixels["irrigation_bin"] = pd.cut(pixels.irrigation_fraction,
        [-np.inf, *cutpoints, np.inf], labels=["Little/no", "Moderate", "High"], right=False)
    return pixels


def endpoint_population(pixels: pd.DataFrame, metric: str, population: str) -> pd.DataFrame:
    if ALL_ENDPOINTS[metric][2] == "effects":
        eligible = pixels["primary_effect_eligible" if population == "primary_qc" else "stable_effect_eligible"]
    else:
        eligible = pixels.irrigation_eligible
    return pixels.loc[eligible & pixels[metric].notna()].copy()


def aggregate(values: np.ndarray, method: str) -> float:
    return float(np.mean(values) if method == "mean" else np.median(values))


def bin_summaries(pixels: pd.DataFrame, block_km: float, bootstrap: int,
                  min_blocks: int, seed: int) -> pd.DataFrame:
    records = []
    for metric, (_, method, family, _) in ALL_ENDPOINTS.items():
        populations = ["primary_qc", "stable_all"] if family == "effects" else ["all_fitted"]
        for population in populations:
            data = endpoint_population(pixels, metric, population)
            data["block"] = block_ids(data, block_km)
            for label in ["Little/no", "Moderate", "High"]:
                group = data.loc[data.irrigation_bin.eq(label)]
                clusters = [g[metric].to_numpy(float) for _, g in group.groupby("block", sort=True)]
                record = dict(metric=metric, population=population, bin=label, estimator=method,
                              n_pixels=len(group), n_blocks=len(clusters), estimate=np.nan,
                              ci_low=np.nan, ci_high=np.nan, spatial_p05=np.nan, spatial_p95=np.nan,
                              bootstrap_valid=0, status="empty")
                if not len(group):
                    records.append(record)
                    continue
                values = group[metric].to_numpy(float)
                record.update(estimate=aggregate(values, method), spatial_p05=np.quantile(values, .05),
                              spatial_p95=np.quantile(values, .95), status="too_few_blocks")
                if len(clusters) >= min_blocks:
                    rng = rng_for(seed, "bins", metric, population, label, block_km)
                    sampled = np.empty(bootstrap)
                    for r in range(bootstrap):
                        chosen = rng.integers(0, len(clusters), len(clusters))
                        sampled[r] = aggregate(np.concatenate([clusters[j] for j in chosen]), method)
                    low, high = np.quantile(sampled, [.025, .975])
                    record.update(ci_low=low, ci_high=high, bootstrap_valid=bootstrap, status="ok")
                records.append(record)
    return pd.DataFrame(records)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    norms = np.linalg.norm(x), np.linalg.norm(y)
    if min(norms) < 1e-10:
        return np.nan
    return float(np.clip(np.dot(x, y) / (norms[0] * norms[1]), -1, 1))


def rank_residuals(x: np.ndarray, y: np.ndarray, controls: np.ndarray):
    n = len(x)
    design = np.column_stack([np.ones(n), *[rankdata(controls[:, k]) for k in range(controls.shape[1])]])
    if np.linalg.matrix_rank(design) < design.shape[1] or n <= design.shape[1] + 2:
        return None
    basis, _ = np.linalg.qr(design, mode="reduced")
    xr, yr = rankdata(x), rankdata(y)
    return xr - basis @ (basis.T @ xr), yr - basis @ (basis.T @ yr), basis


def association_test(x: np.ndarray, y: np.ndarray, controls: np.ndarray,
                     bootstrap: int, permutations: int, min_blocks: int,
                     rng: np.random.Generator) -> dict:
    """Spearman/partial Spearman on equal-weight blocks; residual permutations.

    With controls, reduced-model rank residuals are permuted (Freedman--Lane)
    and residualized again. Paired bootstrap samples rerank all variables.
    Residual exchangeability is an assumption, not guaranteed by blocking.
    """
    result = dict(rho=np.nan, ci_low=np.nan, ci_high=np.nan, p_permutation=np.nan,
                  bootstrap_valid=0, permutation_valid=0, status="too_few_blocks")
    n = len(x)
    if n < min_blocks:
        return result
    fit = rank_residuals(x, y, controls)
    if fit is None:
        return result | {"status": "singular_controls"}
    xr, yr, basis = fit
    observed = correlation(xr, yr)
    if not np.isfinite(observed):
        return result | {"status": "constant_or_explained_by_controls"}
    exceed, valid = 0, 0
    for _ in range(permutations):
        shuffled = rng.permutation(yr)
        value = correlation(xr, shuffled - basis @ (basis.T @ shuffled))
        if np.isfinite(value):
            valid += 1
            exceed += abs(value) >= abs(observed) - 1e-12
    samples = []
    for _ in range(bootstrap):
        chosen = rng.integers(0, n, n)
        fit = rank_residuals(x[chosen], y[chosen], controls[chosen])
        if fit is not None:
            value = correlation(fit[0], fit[1])
            if np.isfinite(value):
                samples.append(value)
    result.update(rho=observed, bootstrap_valid=len(samples), permutation_valid=valid,
                  status="ok", p_permutation=(1 + exceed) / (permutations + 1) if valid == permutations else np.nan)
    if valid != permutations:
        result["status"] = "invalid_permutations"
    if len(samples) >= .95 * bootstrap:
        result["ci_low"], result["ci_high"] = np.quantile(samples, [.025, .975])
    elif result["status"] == "ok":
        result["status"] = "insufficient_valid_bootstrap"
    return result


def trend_tests(pixels: pd.DataFrame, sizes: list[float], has_aridity: bool,
                bootstrap: int, permutations: int, min_blocks: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, block_records = [], []
    for size in sizes:
        click.echo(f"Block associations: {size:g} km")
        for metric, (_, method, family, expected) in ENDPOINTS.items():
            populations = ["primary_qc", "stable_all"] if family == "effects" else ["all_fitted"]
            for population in populations:
                data = endpoint_population(pixels, metric, population)
                modes = ["unadjusted", "aridity_matched_unadjusted", "adjusted"] if has_aridity else ["unadjusted"]
                for mode in modes:
                    subset = data if mode == "unadjusted" else data.loc[data.aridity_index.notna()]
                    subset = subset.assign(block=block_ids(subset, size))
                    blocks = subset.groupby("block", sort=True).agg(
                        irrigation_fraction=("irrigation_fraction", "mean"), outcome=(metric, method),
                        aridity_index=("aridity_index", "median"), cropland_fraction=("cropland_fraction", "mean"),
                        n_pixels=(metric, "size")).reset_index()
                    # Constant nuisance variables add no information and are explicitly logged.
                    controls = [c for c in ["aridity_index", "cropland_fraction"]
                                if mode == "adjusted" and blocks[c].nunique() > 1]
                    ident = dict(block_km=size, metric=metric, population=population, adjustment=mode,
                                 family=family, expected_direction=expected, controls=",".join(controls))
                    test = association_test(blocks.irrigation_fraction.to_numpy(), blocks.outcome.to_numpy(),
                        blocks[controls].to_numpy().reshape(len(blocks), len(controls)), bootstrap,
                        permutations, min_blocks, rng_for(seed, "trend", size, metric, population, mode))
                    records.append(ident | dict(n_blocks=len(blocks), n_pixels=len(subset)) | test)
                    block_records.append(blocks.assign(**ident))
    tests = pd.DataFrame(records)
    tests["p_bh"] = np.nan
    for _, group in tests.groupby(["block_km", "population", "adjustment", "family"]):
        if group.family.iloc[0] in {"diagnostics", "effects"}:
            tests.loc[group.index, "p_bh"] = fdr_bh(group.p_permutation.to_numpy())
    return tests, pd.concat(block_records, ignore_index=True)


def figures(pixels: pd.DataFrame, summaries: pd.DataFrame, tests: pd.DataFrame,
            output: Path, block_km: float, horizon: int, cutpoints: tuple[float, float]) -> None:
    plt.rcParams.update(STYLE)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7})
    bins = ["Little/no", "Moderate", "High"]
    a, b = [v * 100 for v in cutpoints]
    tick_labels = [f"Little/no\n<{a:g}%", f"Moderate\n{a:g}–<{b:g}%", f"High\n≥{b:g}%"]
    diagnostic = ["qc_pass", "whiteness_rejected", "structural_correlation", "lag1_correlation", "crosslag_correlation"]
    for kind, metrics in [("diagnostics", diagnostic), ("effects", list(ENDPOINTS)[5:])]:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), layout="constrained")
        for i, metric in enumerate(metrics):
            ax = axes.flat[i]
            rows = summaries.loc[summaries.metric.eq(metric)]
            for population, color, shift, marker, label in [
                ("all_fitted", "#276e8e", 0, "o", "All fitted cropland"),
                ("primary_qc", "#276e8e", -.07, "o", "Primary QC"),
                ("stable_all", "#b56b26", .07, "s", "All stable (sensitivity)")]:
                group = rows.loc[rows.population.eq(population)].set_index("bin").reindex(bins)
                if group.estimate.isna().all():
                    continue
                scale = 100 if metric in {"qc_pass", "whiteness_rejected"} else 1
                x = np.arange(3) + shift
                ax.plot(x, group.estimate * scale, marker=marker, color=color, label=label, lw=1)
                finite = group.ci_low.notna() & group.ci_high.notna()
                # vlines remain valid if percentile intervals do not contain the estimate.
                ax.vlines(x[finite], group.loc[finite, "ci_low"] * scale,
                          group.loc[finite, "ci_high"] * scale, color=color, lw=1.4)
            title = ALL_ENDPOINTS[metric][0].replace("fraction", "(%)").replace(": ", ":\n")
            title = title.replace("Structural-error correlation", "Structural-error\ncorrelation")
            title = title.replace("Lag-1 innovation autocorrelation", "Lag-1 innovation\nautocorrelation")
            title = title.replace("Cross-lag innovation correlation", "Cross-lag innovation\ncorrelation")
            ax.set_title(f"({chr(97+i)}) {title}", fontsize=8)
            ax.set_xticks(range(3), tick_labels)
            ax.set_xlabel("Equipped area (% of total)")
            if kind == "effects":
                ax.axhline(0, color=".55", lw=.6)
                ax.set_ylabel(f"Scaled response (0–{horizon})")
            elif metric in {"qc_pass", "whiteness_rejected"}:
                ax.set_ylim(-2, 102)
        if kind == "diagnostics":
            ax = axes.flat[5]
            cohort = pixels.loc[pixels.irrigation_eligible]
            for shift, flag, label, color in [(-.15, None, "Fitted cropland", ".65"),
                                             (.15, "qc_pass", "QC pass", "#276e8e")]:
                subset = cohort if flag is None else cohort.loc[cohort[flag].eq(1)]
                counts = subset.groupby("irrigation_bin", observed=False).size().reindex(bins, fill_value=0)
                ax.bar(np.arange(3) + shift, counts, width=.3, label=label, color=color)
            ax.set_xticks(range(3), tick_labels)
            ax.set_ylabel("Pixels")
            ax.set_title("(f) Coverage", fontsize=8)
            ax.legend(frameon=False)
        else:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
        save_figure(fig, output / ("irrigation_" + kind))
    selected = tests.loc[tests.block_km.eq(block_km) & ~tests.population.eq("stable_all")
                         & ~tests.adjustment.eq("aridity_matched_unadjusted")]
    fig, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
    for adjustment, color, offset, label in [("unadjusted", "#276e8e", -.12, "Unadjusted"),
                                             ("adjusted", "#b56b26", .12, "Adjusted for aridity and crop fraction")]:
        group = selected.loc[selected.adjustment.eq(adjustment)].set_index("metric").reindex(ENDPOINTS)
        if group.rho.isna().all():
            continue
        y = np.arange(len(ENDPOINTS)) + offset
        ax.scatter(group.rho, y, color=color, s=20, label=label)
        ax.hlines(y, group.ci_low, group.ci_high, color=color)
    ax.set_yticks(np.arange(len(ENDPOINTS)), [v[0] for v in ENDPOINTS.values()])
    ax.invert_yaxis()
    ax.axvline(0, color=".5", lw=.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Block rank association (95% interval)")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=7)
    save_figure(fig, output / "irrigation_associations")
    cohort = pixels.loc[pixels.irrigation_eligible]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4), layout="constrained")
    for ax in axes:
        ax.scatter(pixels.easting/1000, pixels.northing/1000, c=".88", s=2, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    scatter = axes[0].scatter(cohort.easting/1000, cohort.northing/1000,
        c=cohort.irrigation_fraction * 100, cmap="viridis", vmin=0, vmax=100, s=3, rasterized=True)
    fig.colorbar(scatter, ax=axes[0], label="Land equipped for irrigation (%)", shrink=.7)
    axes[0].set_title("(a) Cropland irrigation exposure")
    for passed, color, label in [(False, "#b56b26", "QC failed"), (True, "#276e8e", "QC passed")]:
        group = cohort.loc[cohort.qc_pass.eq(float(passed))]
        axes[1].scatter(group.easting/1000, group.northing/1000, c=color, s=3, label=label, rasterized=True)
    axes[1].set_title("(b) Cropland QC outcomes")
    axes[1].legend(frameon=False, markerscale=3)
    save_figure(fig, output / "irrigation_coverage")


def write_tables(tests: pd.DataFrame, summaries: pd.DataFrame, output: Path, block_km: float) -> None:
    def number(x, precision=3):
        return "--" if pd.isna(x) else f"{x:.{precision}f}"
    for mode in ["unadjusted", "adjusted"]:
        chosen = tests.loc[tests.block_km.eq(block_km) & tests.adjustment.eq(mode)
                           & ~tests.population.eq("stable_all")]
        if chosen.empty:
            continue
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Response & Blocks & $\rho$ & 95\% interval & $p_{\mathrm{perm}}$ & $p_{\mathrm{BH}}$ \\", r"\midrule"]
        for row in chosen.itertuples():
            interval = (f"[{number(row.ci_low)}, {number(row.ci_high)}]" if np.isfinite(row.ci_low) else "--")
            label = ENDPOINTS[row.metric][0].replace("–", "--")
            lines.append(f"{label} & {row.n_blocks} & {number(row.rho)} & {interval} & "
                         f"{number(row.p_permutation, 4)} & {number(row.p_bh, 4)} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (output / f"irrigation_trends_{mode}.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Diagnostic & Irrigation class & Pixels & Estimate & 95\% interval \\", r"\midrule"]
    for row in summaries.loc[summaries.population.eq("all_fitted")].itertuples():
        label = ALL_ENDPOINTS[row.metric][0]
        lines.append(f"{label} & {row.bin} & {row.n_pixels} & {number(row.estimate)} & "
                     f"[{number(row.ci_low)}, {number(row.ci_high)}] " + r"\\")
    (output / "irrigation_diagnostics.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_report(output: Path, counts: dict, args: dict, tests: pd.DataFrame, crop_description: str) -> None:
    primary = tests.loc[tests.block_km.eq(args["block_km"]) & tests.metric.eq("qc_pass")]
    rows = []
    for row in primary.itertuples():
        rows.append(f"- {row.adjustment}: {row.n_blocks} blocks, rho={row.rho:.3f}, "
                    f"95% interval [{row.ci_low:.3f}, {row.ci_high:.3f}], "
                    f"two-sided p={row.p_permutation:.4g}; status={row.status}.")
    a, b = args["irrigation_cutpoints"]
    text = f"""# Irrigation exposure, model diagnostics, and saved vegetation responses

## Population and primary result

{json.dumps(counts, indent=2)}

The primary endpoint is the association of equipped irrigation fraction with
QC pass fraction across geographic blocks. Diagnostic analyses include all
fitted cropland pixels, including failures. Invalid whiteness calibrations are
reported separately and are not counted as whiteness rejections.

{chr(10).join(rows)}

## Methods

Irrigation input: {args['irrigation_description']}.
Cropland input: {crop_description}.
Native categorical classes are converted to a binary mask before footprint
averaging. Both external rasters require at least {args['min_coverage']:.0%}
valid footprint coverage. Cropland requires fraction >= {args['min_cropland_fraction']:.0%}.
The three equipped-area classes are [0,{a:g}), [{a:g},{b:g}), and [{b:g},1].
They are operational study choices, not FAO classification thresholds.
Irrigation is a fraction of total grid-cell area, not of cropland area.

Spatial blocks use {args['block_km']:g} km squares in {args['block_crs']}, anchored
at coordinate zero. In each endpoint's valid population, block exposure is
mean irrigation fraction; binary outcomes use block fractions and continuous
outcomes use block medians. Each block has equal weight in trend inference.
Two-sided tests use {args['permutations']} permutations and a plus-one correction;
{args['bootstrap_samples']} paired block bootstraps give percentile 95% intervals.
Adjusted tests, when external AI is supplied, correlate rank residuals after
adjusting for block median AI and mean crop fraction, using Freedman--Lane
reduced-model residual permutations. Constant controls are omitted and logged.
The aridity-matched unadjusted test uses exactly the adjusted test's pixels.
Reranking and adjustment are repeated in each bootstrap sample.

Benjamini--Hochberg adjustment applies separately to four secondary diagnostic
tests and six effect tests, within each block size, population and adjustment.
The QC pass test is the single primary endpoint. Alternative block sizes,
adjusted analyses, and stable-model effects are sensitivity analyses; there is
no claimed multiplicity control across all these analyses. Inspect their
consistency rather than selecting the smallest p-value. Fewer than
{args['min_blocks']} blocks or degenerate data suppress inference. Intervals
require at least 95% valid bootstrap replicates.

Bin estimates remain pixel weighted: block resampling within each fixed
irrigation class recomputes the pixel fraction/median. Intervals are pointwise.
Diagnostic estimates are conditional on the fitted records; effect intervals
do not propagate graph-estimation uncertainty. Effects are saved signed sums
over months 0--{args['horizon']} after a one-time pulse, scaled by local source
and NDVI 10th--90th percentile ranges. Absolute cumulative effects assess
coupling magnitude; they are not sums of absolute horizon-specific responses.
Effect comparisons require finite precipitation and both soil-moisture effects.
Pruned zeros are retained. No significance filter is applied. Primary effects
use QC-passing models. Stable-all effects are a selection sensitivity only:
failed-QC effects do not acquire a reliable causal interpretation.

## Interpretation

A negative irrigation/QC association together with more whiteness rejections
and larger unexplained dependence would be consistent with omitted irrigation
inputs affecting model adequacy. It would not identify irrigation as the cause
of failure. Whiteness and correlation screens are components of the same QC
mask, not independent confirmations. Smaller absolute precipitation effects
would be consistent with weaker fitted rainfall coupling; a declining signed
effect alone can instead mean a stronger negative response. No direction is
prescribed for soil-moisture changes. Locally scaled effects do not compare
equal physical perturbations, and cumulative cancellation can reduce magnitude.

GMIA records irrigation infrastructure around 2005, not actual watering during
the observed record. The 5-arcminute layer has coarser support than the NDVI
grid; alignment does not create finer irrigation information. GMIA's documented
-9 cells mean no irrigation and are recoded to zero only in GMIA mode. In other
products, nodata remains missing. If WorldCover class 40 is used, results apply
mainly to annual herbaceous cropland; perennial woody crops such as orchards
and olive groves are not fully represented. Its 2021 snapshot is temporally
different from GMIA and the full observation period.

Aridity adjustment helps assess climatic confounding but cannot account for
all crop, soil, management, elevation, or observation-quality differences.
The effect population is selected by QC, so its gradients can reflect selection.
Spatial blocking and block-size sensitivity do not guarantee independence or
exchangeability. These are geographical associations relevant to diagnostic
plausibility, not proof of causal correctness or diagnostic specificity.

## Figure captions

**irrigation_diagnostics**: QC pass fraction, whiteness rejection among valid
calibrations, and median structural-error and innovation correlations across
irrigation classes in fitted cropland pixels. Vertical intervals are pointwise
95% spatial-block bootstrap intervals. The final panel shows fitted and
QC-passing pixel counts. Calibration availability is exported separately.

**irrigation_effects**: Median signed (top row) and absolute (bottom row)
cumulative NDVI responses to precipitation and two soil-moisture sources, by
irrigation class. Blue represents the primary QC population; orange represents
all stable models as a selection sensitivity. Intervals resample spatial blocks
conditional on the saved fitted effects. Responses use local quantile scaling.

**irrigation_associations**: Block rank associations with mean equipped
irrigation fraction, with 95% paired-block bootstrap intervals. Adjusted values
control rank-transformed aridity and crop fraction where available. Diagnostics
include all fitted cropland; effect estimates use primary QC. Sample counts,
test status and multiplicity-adjusted p-values are in trend_tests.csv.

**irrigation_coverage**: Equipped irrigation fraction and QC status within the
cropland analysis population. Gray background represents other fitted pixels,
including insufficient crop fraction or missing external data. Irrigation
resolution remains that of the external product.
"""
    (output / "report.md").write_text(text)


@click.command(context_settings={"show_default": True})
@click.option("--effects-db", type=click.Path(exists=True, path_type=Path))
@click.option("--effects-csv", type=click.Path(exists=True, path_type=Path))
@click.option("--effects-table", default="pixel_varlingam_effects")
@click.option("--qc-csv", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--qc-column", default="primary_eligible")
@click.option("--reference-raster", type=click.Path(exists=True, path_type=Path))
@click.option("--source-db", type=click.Path(exists=True, path_type=Path), help="Alternative: find an original NDVI reference raster in geotiff_catalog.")
@click.option("--reference-variable", default="modis_ndvi", help="Source-catalog variable defining original row/col indices.")
@click.option("--irrigation-raster", type=click.Path(exists=True, path_type=Path))
@click.option("--download-irrigation", is_flag=True, help="Download the FAO GMIA v5 equipped-area percentage grid.")
@click.option("--irrigation-kind", type=click.Choice(["gmia", "fraction"]), default="gmia")
@click.option("--irrigation-scale", type=click.FloatRange(min=0, min_open=True), default=1., help="Raw multiplier for fraction mode only; use .01 for percentages.")
@click.option("--irrigation-description", default="FAO GMIA v5; equipped area circa 2005")
@click.option("--cropland-raster", multiple=True, type=click.Path(exists=True, path_type=Path), help="Repeat for disjoint native tiles; or give one VRT/mosaic.")
@click.option("--download-worldcover", is_flag=True, help="Download intersecting WorldCover 2021 tiles; can require gigabytes.")
@click.option("--cropland-kind", type=click.Choice(["categorical", "fraction"]), default="categorical")
@click.option("--cropland-classes", default="40", help="Comma-separated crop codes; WorldCover annual cropland is 40.")
@click.option("--cropland-scale", type=click.FloatRange(min=0, min_open=True), default=1.)
@click.option("--cropland-description", default="User-supplied cropland raster; specify product/year for publication")
@click.option("--min-cropland-fraction", type=click.FloatRange(min=0, max=1, min_open=True), default=.5)
@click.option("--min-coverage", type=click.FloatRange(min=0, max=1, min_open=True), default=.5)
@click.option("--irrigation-cutpoints", default="0.01,0.20")
@click.option("--aridity-pixels", type=click.Path(exists=True, path_type=Path), help="pixel_aridity_effects.csv from the aridity experiment, including QC failures.")
@click.option("--target", default="ndvi_resid")
@click.option("--horizon", type=click.IntRange(min=0), default=12)
@click.option("--block-km", type=click.FloatRange(min=0, min_open=True), default=100.)
@click.option("--block-crs", default="EPSG:3035", help="Metric equal-area CRS; default is for Europe.")
@click.option("--sensitivity-block-km", default="50,150", help="Comma-separated additional sizes; empty disables.")
@click.option("--bootstrap-samples", type=click.IntRange(min=19), default=999)
@click.option("--permutations", type=click.IntRange(min=19), default=4999)
@click.option("--min-blocks", type=click.IntRange(min=4), default=10)
@click.option("--seed", type=click.IntRange(min=0), default=20260505)
@click.option("--raster-chunk-cells", type=click.IntRange(min=1, max=64), default=4)
@click.option("--cache-dir", type=click.Path(path_type=Path), default="data/external/irrigation")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--no-plots", is_flag=True)
def varlingam_irrigation_stratification(**args):
    """Associate independent irrigation exposure with QC and cumulative NDVI effects."""
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


def run(args: dict) -> None:
    if bool(args["effects_db"]) == bool(args["effects_csv"]):
        raise ValueError("Specify exactly one of --effects-db or --effects-csv.")
    if bool(args["irrigation_raster"]) == args["download_irrigation"]:
        raise ValueError("Specify --irrigation-raster or --download-irrigation, exclusively.")
    if bool(args["cropland_raster"]) == args["download_worldcover"]:
        raise ValueError("Specify --cropland-raster or --download-worldcover, exclusively.")
    if bool(args["reference_raster"]) == bool(args["source_db"]):
        raise ValueError("Specify exactly one of --reference-raster or --source-db.")
    if args["source_db"]:
        with duckdb.connect(str(args["source_db"]), read_only=True) as connection:
            entries = connection.execute("SELECT root_dir, file_name FROM geotiff_catalog WHERE variable_name = ? ORDER BY year, month, file_name",
                                         [args["reference_variable"]]).fetchall()
        paths = [Path(root) / name for root, name in entries]
        args["reference_raster"] = next((p for p in paths if p.is_file()), None)
        if args["reference_raster"] is None:
            raise ValueError("No accessible reference raster in source catalog; provide --reference-raster on the original grid.")
        click.echo(f"Original reference raster: {args['reference_raster']}")
    if args["download_irrigation"] and args["irrigation_kind"] != "gmia":
        raise ValueError("--download-irrigation requires --irrigation-kind gmia.")
    if args["irrigation_kind"] == "gmia" and args["irrigation_scale"] != 1:
        raise ValueError("GMIA mode handles percent conversion; omit --irrigation-scale.")
    if args["download_worldcover"] and (args["cropland_kind"] != "categorical" or args["cropland_classes"] != "40"):
        raise ValueError("WorldCover download uses categorical annual cropland class 40.")
    if args["cropland_kind"] == "categorical" and args["cropland_scale"] != 1:
        raise ValueError("Categorical cropland codes must not be scaled.")
    cuts = tuple(float(v) for v in args["irrigation_cutpoints"].split(","))
    if len(cuts) != 2 or not 0 < cuts[0] < cuts[1] < 1:
        raise ValueError("Two increasing --irrigation-cutpoints must lie strictly between 0 and 1.")
    args["irrigation_cutpoints"] = cuts
    sizes = list(dict.fromkeys([args["block_km"], *[float(v) for v in args["sensitivity_block_km"].split(",") if v.strip()]]))
    if any(not np.isfinite(v) or v <= 0 for v in sizes):
        raise ValueError("Block sizes must be finite and positive.")
    classes = tuple(int(v) for v in args["cropland_classes"].split(","))
    crs = rasterio.crs.CRS.from_user_input(args["block_crs"])
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("--block-crs must be projected in metres; choose a suitable equal-area CRS.")
    out = args["output_dir"]
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty; existing results are never overwritten.")
    # Validate tables before downloads or expensive raster processing.
    pixels = load_diagnostics(args["qc_csv"], args["qc_column"])
    effects = load_effects(args["effects_db"], args["effects_csv"], args["effects_table"], args["target"], args["horizon"], 1)
    pixels, grid = study_grid(pixels, args["reference_raster"], args["block_crs"])
    pixels = join_aridity(pixels, args["aridity_pixels"])
    out.mkdir(parents=True, exist_ok=True)
    irrigation = download_gmia(args["cache_dir"]) if args["download_irrigation"] else args["irrigation_raster"]
    crop_paths = download_worldcover(pixels, args["reference_raster"], args["cache_dir"] / "worldcover") if args["download_worldcover"] else list(args["cropland_raster"])
    crop_description = ("ESA WorldCover 2021 v200; class 40 annual herbaceous cropland; doi:10.5281/zenodo.7254221"
                        if args["download_worldcover"] else args["cropland_description"])
    click.echo("Aligning irrigation to original pixel footprints")
    pixels = align_fraction(pixels, grid, [irrigation], out / "aligned_irrigation.tif", "irrigation_fraction",
        args["irrigation_kind"], args["irrigation_scale"], min_coverage=args["min_coverage"], chunk_cells=args["raster_chunk_cells"])
    click.echo("Aggregating native cropland coverage (large 10 m inputs may take several minutes)")
    pixels = align_fraction(pixels, grid, crop_paths, out / "aligned_cropland.tif", "cropland_fraction",
        args["cropland_kind"], args["cropland_scale"], classes, args["min_coverage"], args["raster_chunk_cells"])
    pixels = prepare_population(pixels, effects, args["min_cropland_fraction"], cuts)
    if not pixels.irrigation_eligible.any():
        raise ValueError("No fitted pixels satisfy cropland and irrigation coverage requirements.")
    counts = {"fitted_pixels": len(pixels), "valid_irrigation": int(pixels.irrigation_fraction.notna().sum()),
              "valid_cropland": int(pixels.cropland_fraction.notna().sum()),
              **{key: int(pixels[key].sum()) for key in ["cropland_eligible", "irrigation_eligible", "primary_effect_eligible", "stable_effect_eligible"]}}
    cohort = pixels.loc[pixels.irrigation_eligible]
    counts.update(cropland_qc_pass=int(cohort.qc_pass.sum()),
                  cropland_whiteness_valid=int(cohort.whiteness_valid.sum()),
                  cropland_missing_aridity=int(cohort.aridity_index.isna().sum()),
                  stable_cropland_missing_effects=int((cohort.point_stable & ~cohort.effects_complete).sum()))
    if counts["stable_cropland_missing_effects"]:
        click.echo("WARNING: some stable cropland models lack complete effects; see manifest counts. Use the all-fitted effects table, not a primary-only export.", err=True)
    pixels.to_csv(out / "pixel_irrigation_diagnostics.csv", index=False)
    coverage = cohort.groupby("irrigation_bin", observed=False).agg(
        fitted_pixels=("row", "size"), qc_pass=("qc_pass", "sum"),
        valid_whiteness=("whiteness_valid", "sum"), primary_effects=("primary_effect_eligible", "sum"),
        stable_effects=("stable_effect_eligible", "sum"), valid_aridity=("aridity_index", "count"),
        median_aridity=("aridity_index", "median"), mean_crop_fraction=("cropland_fraction", "mean"),
        mean_irrigation=("irrigation_fraction", "mean"),
        min_irrigation=("irrigation_fraction", "min"), max_irrigation=("irrigation_fraction", "max"))
    coverage.to_csv(out / "qc_coverage.csv")
    pd.DataFrame({"bin": ["Little/no", "Moderate", "High"], "lower_inclusive": [0, *cuts],
                  "upper": [*cuts, 1], "upper_inclusive": [False, False, True]}).to_csv(out / "bin_definitions.csv", index=False)
    click.echo("Summarizing irrigation classes with spatial-block intervals")
    summaries = bin_summaries(pixels, args["block_km"], args["bootstrap_samples"], args["min_blocks"], args["seed"])
    tests, blocks = trend_tests(pixels, sizes, args["aridity_pixels"] is not None, args["bootstrap_samples"],
                                args["permutations"], args["min_blocks"], args["seed"])
    summaries.to_csv(out / "bin_summaries.csv", index=False)
    tests.to_csv(out / "trend_tests.csv", index=False)
    blocks.to_csv(out / "block_summaries.csv", index=False)
    write_tables(tests, summaries, out, args["block_km"])
    write_report(out, counts, args, tests, crop_description)
    (out / "references.bib").write_text(BIBTEX + (ARIDITY_BIBTEX if args["aridity_pixels"] else ""))
    if not args["no_plots"]:
        figures(pixels, summaries, tests, out, args["block_km"], args["horizon"], cuts)
    inputs = [args[k] for k in ["qc_csv", "effects_db", "effects_csv", "aridity_pixels", "source_db"] if args[k]]
    inputs += [irrigation, *crop_paths]
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    with rasterio.open(args["reference_raster"]) as ref:
        reference = dict(path=str(args["reference_raster"].resolve()), crs=str(ref.crs),
                         transform=list(ref.transform), width=ref.width, height=ref.height)
    from confoundry import irrigation_covariates, varlingam_aridity_stratification
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), parameters=args, counts=counts,
                    input_sha256=hashes, reference_grid=reference, crop_description=crop_description,
                    gmia_documentation=GMIA_PAGE, gmia_download_url=GMIA_URL if args["download_irrigation"] else None,
                    software={x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scipy", "rasterio", "duckdb", "matplotlib"]},
                    code_sha256={str(p): digest(p) for p in [Path(__file__), Path(irrigation_covariates.__file__),
                                                          Path(varlingam_aridity_stratification.__file__)]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    click.echo(f"Saved irrigation experiment to {out}; {len(cohort):,} eligible fitted cropland pixels.")


if __name__ == "__main__":
    varlingam_irrigation_stratification()
