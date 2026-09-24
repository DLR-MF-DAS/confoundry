"""Validate single-pixel VAR--LiNGAM recovery with synthetic time series.

The stored graph for one pixel is treated as the data-generating model.  The
command resamples that pixel's estimated structural errors, simulates a stable
structural VAR process, adds the stored monthly seasonal cycle and linear
trend, and then reruns residualization and VAR--LiNGAM.  Repeating this process
measures how accurately a 240-month record recovers a known graph.

This is a parameter-recovery (simulation) check.  It can detect implementation
errors and finite-sample limitations, but it cannot establish that the graph
estimated from the real observations is the true environmental causal graph.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import lingam
import matplotlib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from confoundry.analysis_helpers import ensure_identifier, require_files
from confoundry.per_pixel_graph_diagnostics import (
    fit_reduced_form_var,
    structural_residuals_for_graph,
)
from confoundry.per_pixel_graph_discovery import make_prior_knowledge
from confoundry.residualize_timeseries import (
    design_columns,
    design_matrix,
    residualize_group,
)
from confoundry.varlingam_postprocess import (
    dynamic_effect_matrices,
    reduced_form_matrices,
    stability_radius,
)


NOISE_MODES = ("empirical-independent", "empirical-joint", "gaussian-independent")


@dataclass(frozen=True)
class SourceModel:
    """Parameters and observations needed for the simulation experiment."""

    row: int
    col: int
    labels: tuple[str, ...]
    raw_variables: tuple[str, ...]
    dates: pd.DataFrame
    original_residuals: np.ndarray
    contemporaneous: np.ndarray
    lagged: np.ndarray
    innovation_pool: np.ndarray
    residualization_models: pd.DataFrame
    true_baseline: np.ndarray
    global_start_month_index: int
    suffix: str
    expected_suffix: str
    seasonal_model: str
    include_trend: bool
    fit_end_year: int | None


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and minimally validate a generated residual-series config."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict) or "name" not in config or "columns" not in config:
        raise click.ClickException("Config must contain 'name' and 'columns'.")
    return config


def resolve_path(base_dir: Path, value: str | Path | None, default: Path) -> Path:
    """Resolve a config or command-line path in the same way as production."""
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    try:
        cwd_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return base_dir / path
    return cwd_path


def parse_matrix(value: Any, name: str, ndim: int) -> np.ndarray:
    """Parse and validate a JSON matrix or tensor."""
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=float)
    if array.ndim != ndim:
        raise click.ClickException(
            f"{name} must have {ndim} dimensions, got shape {array.shape}."
        )
    return array


def one_graph_row(
    db_path: Path,
    table: str,
    row: int,
    col: int,
) -> dict[str, Any]:
    """Load exactly one stored graph row."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        frame = con.execute(
            f"SELECT * FROM {ensure_identifier(table)} WHERE row = ? AND col = ?",
            [row, col],
        ).fetchdf()
    finally:
        con.close()
    if frame.empty:
        raise click.ClickException(
            f"Pixel (row={row}, col={col}) is absent from {db_path}::{table}."
        )
    if len(frame) != 1:
        raise click.ClickException(
            f"Expected one graph for pixel ({row}, {col}), found {len(frame)}."
        )
    result = frame.iloc[0].to_dict()
    if str(result.get("model_type", "")).lower() != "varlingam":
        raise click.ClickException("The selected graph is not a VAR-LiNGAM graph.")
    return result


def monthly_calendar(start_year: int, start_month: int, n_samples: int) -> pd.DataFrame:
    """Create an uninterrupted monthly calendar."""
    start = int(start_year) * 12 + int(start_month) - 1
    absolute = start + np.arange(int(n_samples))
    months = absolute % 12 + 1
    angle = 2.0 * np.pi * (months - 1) / 12.0
    return pd.DataFrame(
        {
            "row": np.zeros(n_samples, dtype=int),
            "col": np.zeros(n_samples, dtype=int),
            "year": absolute // 12,
            "month": months,
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
        }
    )


def _consistent_value(values: pd.Series, name: str) -> Any:
    non_missing = values.dropna().unique()
    if len(non_missing) != 1:
        raise click.ClickException(
            f"Residualization field {name!r} is not consistent across variables: "
            f"{non_missing.tolist()}."
        )
    return non_missing[0]


def stored_baseline(
    calendar: pd.DataFrame,
    model_rows: pd.DataFrame,
    labels: Sequence[str],
    global_start_month_index: int,
) -> np.ndarray:
    """Evaluate the stored deterministic baseline for each residual variable."""
    frame = calendar.copy()
    absolute = frame["year"].astype(int) * 12 + frame["month"].astype(int) - 1
    frame["_residual_time_index"] = absolute - int(global_start_month_index)
    model_lookup = model_rows.set_index("residual_column", drop=False)
    values: list[np.ndarray] = []
    for label in labels:
        record = model_lookup.loc[label]
        coefficients = json.loads(str(record["coefficients_json"]))
        seasonal_model = str(record["seasonal_model"])
        include_trend = bool(record["include_trend"])
        names = ["intercept", *design_columns(include_trend, seasonal_model)]
        missing = [name for name in names if name not in coefficients]
        if missing:
            raise click.ClickException(
                f"Stored residualization coefficients for {label!r} lack {missing}."
            )
        matrix = design_matrix(
            frame,
            seasonal_model=seasonal_model,
            include_trend=include_trend,
            time_center=float(record["time_center_month_index"]),
        )
        beta = np.asarray([coefficients[name] for name in names], dtype=float)
        values.append(matrix @ beta)
    return np.column_stack(values)


def load_source_model(
    *,
    input_db: Path,
    input_table: str,
    model_table: str,
    graph_db: Path,
    graph_table: str,
    row: int,
    col: int,
    config: Mapping[str, Any],
    n_samples: int,
) -> SourceModel:
    """Load one fitted graph, its source series, and deterministic models."""
    graph = one_graph_row(graph_db, graph_table, row, col)
    labels = tuple(json.loads(str(graph["variable_names_json"])))
    contemporaneous = parse_matrix(graph["adjacency_raw_json"], "adjacency_raw_json", 2)
    lagged = parse_matrix(
        graph["adjacency_lagged_raw_json"],
        "adjacency_lagged_raw_json",
        3,
    )
    if (
        contemporaneous.shape != (len(labels), len(labels))
        or lagged.shape[1:] != contemporaneous.shape
    ):
        raise click.ClickException("Stored graph matrices do not match variable_names_json.")

    selected = ", ".join(
        ["year", "month", *[ensure_identifier(label) for label in labels]]
    )
    con = duckdb.connect(input_db, read_only=True)
    try:
        source = con.execute(
            f"SELECT {selected} FROM {ensure_identifier(input_table)} "
            "WHERE row = ? AND col = ? ORDER BY year, month",
            [row, col],
        ).fetchdf()
        global_start = con.execute(
            f"SELECT min(year * 12 + month - 1) FROM {ensure_identifier(input_table)}"
        ).fetchone()[0]
        models = con.execute(
            f"SELECT * FROM {ensure_identifier(model_table)} "
            "WHERE row = ? AND col = ?",
            [row, col],
        ).fetchdf()
    finally:
        con.close()

    source = source.dropna(subset=list(labels)).reset_index(drop=True)
    if len(source) <= lagged.shape[0]:
        raise click.ClickException("The selected pixel has too few complete observations.")
    absolute = source["year"].astype(int) * 12 + source["month"].astype(int)
    if len(absolute) > 1 and not np.all(np.diff(absolute) == 1):
        raise click.ClickException("The selected pixel does not have consecutive months.")

    models = models[models["residual_column"].isin(labels)].copy()
    missing_models = sorted(set(labels) - set(models["residual_column"]))
    if missing_models:
        raise click.ClickException(
            "Residualization models are missing for: " + ", ".join(missing_models)
        )
    if models["residual_column"].duplicated().any():
        raise click.ClickException("Residualization model rows are not unique by variable.")
    bad_status = models.loc[models["status"] != "fit", ["residual_column", "status"]]
    if not bad_status.empty:
        raise click.ClickException(
            "Selected pixel has unsuccessful residualization models: "
            + bad_status.to_dict(orient="records").__repr__()
        )

    model_lookup = models.set_index("residual_column")
    raw_variables = tuple(str(model_lookup.loc[label, "variable"]) for label in labels)
    residualization = config.get("residualization") or {}
    suffix = str(residualization.get("suffix", "_resid"))
    expected_suffix = str(residualization.get("expected_suffix", "_seasonal_trend"))
    if tuple(f"{variable}{suffix}" for variable in raw_variables) != labels:
        raise click.ClickException(
            "Residual labels cannot be reproduced from the stored raw-variable names "
            f"and suffix {suffix!r}."
        )

    source_values = source[list(labels)].to_numpy(dtype=float)
    structural_errors, _ = structural_residuals_for_graph(
        source_values,
        contemporaneous,
        graph,
    )
    calendar = monthly_calendar(
        int(source["year"].iloc[0]),
        int(source["month"].iloc[0]),
        n_samples,
    )
    calendar["row"] = row
    calendar["col"] = col
    baseline = stored_baseline(calendar, models, labels, int(global_start))

    fit_end_values = models["fit_end_year"].dropna().unique()
    if len(fit_end_values) > 1:
        raise click.ClickException("fit_end_year differs across residualization models.")
    fit_end_year = int(fit_end_values[0]) if len(fit_end_values) else None
    return SourceModel(
        row=row,
        col=col,
        labels=labels,
        raw_variables=raw_variables,
        dates=calendar,
        original_residuals=source_values,
        contemporaneous=contemporaneous,
        lagged=lagged,
        innovation_pool=structural_errors,
        residualization_models=models,
        true_baseline=baseline,
        global_start_month_index=int(global_start),
        suffix=suffix,
        expected_suffix=expected_suffix,
        seasonal_model=str(_consistent_value(models["seasonal_model"], "seasonal_model")),
        include_trend=bool(_consistent_value(models["include_trend"], "include_trend")),
        fit_end_year=fit_end_year,
    )


def draw_innovations(
    pool: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """Draw centered structural errors under an explicit simulation assumption."""
    values = np.asarray(pool, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    n_pool, n_variables = centered.shape
    if mode == "empirical-independent":
        indices = rng.integers(0, n_pool, size=(n_draws, n_variables))
        return centered[indices, np.arange(n_variables)]
    if mode == "empirical-joint":
        return centered[rng.integers(0, n_pool, size=n_draws)]
    if mode == "gaussian-independent":
        scales = np.std(centered, axis=0, ddof=1)
        if np.any(scales <= 0.0):
            raise ValueError("Cannot draw Gaussian shocks from a constant innovation.")
        return rng.normal(size=(n_draws, n_variables)) * scales
    raise ValueError(f"Unknown innovation mode: {mode!r}")


def innovation_moments(pool: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Describe the empirical structural-error marginals used for simulation."""
    values = np.asarray(pool, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        column = values[:, index]
        centered = column - np.mean(column)
        standard_deviation = float(np.std(centered, ddof=1))
        if standard_deviation > 0.0:
            standardized = centered / standard_deviation
            skewness = float(np.mean(standardized**3))
            excess_kurtosis = float(np.mean(standardized**4) - 3.0)
        else:
            skewness = math.nan
            excess_kurtosis = math.nan
        rows.append(
            {
                "variable": label,
                "n": len(column),
                "mean": float(np.mean(column)),
                "standard_deviation": standard_deviation,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
            }
        )
    return pd.DataFrame(rows)


def simulate_structural_var(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    innovation_pool: np.ndarray,
    *,
    n_samples: int,
    burnin: int,
    rng: np.random.Generator,
    noise_mode: str,
) -> np.ndarray:
    """Simulate ``x_t = B0 x_t + sum(B_l x_{t-l}) + e_t``."""
    b0 = np.asarray(contemporaneous, dtype=float)
    blags = np.asarray(lagged, dtype=float)
    total = int(n_samples) + int(burnin)
    errors = draw_innovations(innovation_pool, total, rng, noise_mode)
    values = np.zeros_like(errors)
    multiplier = np.linalg.inv(np.eye(b0.shape[0]) - b0)
    for time_index in range(total):
        lagged_part = np.zeros(b0.shape[0], dtype=float)
        for lag_index, matrix in enumerate(blags, start=1):
            if time_index >= lag_index:
                lagged_part += matrix @ values[time_index - lag_index]
        values[time_index] = multiplier @ (lagged_part + errors[time_index])
    return values[burnin:]


def residualize_synthetic_frame(
    raw_frame: pd.DataFrame,
    source: SourceModel,
    *,
    min_fit_samples: int,
    min_month_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the production residualization fit to one synthetic pixel."""
    result = raw_frame.sort_values(["year", "month"]).reset_index(drop=True).copy()
    absolute = result["year"].astype(int) * 12 + result["month"].astype(int) - 1
    result["_residual_time_index"] = absolute - source.global_start_month_index
    for variable in source.raw_variables:
        result[f"{variable}{source.suffix}"] = np.nan
        result[f"{variable}{source.expected_suffix}"] = np.nan
    records = residualize_group(
        result=result,
        row_index=result.index,
        variables=source.raw_variables,
        fit_end_year=source.fit_end_year,
        min_fit_samples=min_fit_samples,
        min_month_samples=min_month_samples,
        suffix=source.suffix,
        expected_suffix=source.expected_suffix,
        include_trend=source.include_trend,
        seasonal_model=source.seasonal_model,
    )
    models = pd.DataFrame(records)
    bad = models.loc[models["status"] != "fit", ["variable", "status"]]
    if not bad.empty:
        raise RuntimeError(f"Synthetic residualization failed: {bad.to_dict('records')}")
    return result, models


def fit_varlingam_point(
    values: np.ndarray,
    *,
    labels: Sequence[str],
    n_lags: int,
    criterion: str | None,
    prune: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the same point-estimate VAR--LiNGAM model used in production."""
    prior = make_prior_knowledge(labels, {label: 0 for label in labels})
    instantaneous = lingam.DirectLiNGAM(prior_knowledge=prior, random_state=0)
    model = lingam.VARLiNGAM(
        lags=n_lags,
        criterion=criterion,
        prune=prune,
        lingam_model=instantaneous,
        random_state=0,
    )
    model.fit(np.asarray(values, dtype=float))
    matrices = np.asarray(model.adjacency_matrices_, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (len(labels), len(labels)):
        raise RuntimeError(f"Unexpected fitted adjacency shape: {matrices.shape}")
    if len(matrices) != n_lags + 1:
        raise RuntimeError(
            f"VAR-LiNGAM returned {len(matrices) - 1} lags; expected {n_lags}."
        )
    return matrices[0], matrices[1:]


def correlation_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when either input is constant."""
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def support_metrics(true: np.ndarray, estimated: np.ndarray, threshold: float) -> dict[str, float]:
    """Compare known and recovered coefficient support and signs."""
    truth = np.abs(true) >= threshold
    found = np.abs(estimated) >= threshold
    tp = int(np.sum(truth & found))
    fp = int(np.sum(~truth & found))
    fn = int(np.sum(truth & ~found))
    tn = int(np.sum(~truth & ~found))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else math.nan
    both = truth & found
    sign_agreement = (
        float(np.mean(np.sign(true[both]) == np.sign(estimated[both])))
        if np.any(both)
        else math.nan
    )
    return {
        "true_edges": int(np.sum(truth)),
        "recovered_edges": int(np.sum(found)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sign_agreement": sign_agreement,
        "structural_hamming_distance": fp + fn,
    }


def temporal_correlation_metrics(innovations: np.ndarray, max_lag: int = 12) -> dict[str, float]:
    """Summarize remaining within- and cross-variable innovation correlations."""
    values = np.asarray(innovations, dtype=float)
    max_auto = 0.0
    max_cross = 0.0
    for lag in range(1, min(max_lag, len(values) - 1) + 1):
        left = values[lag:]
        right = values[:-lag]
        for child in range(values.shape[1]):
            for parent in range(values.shape[1]):
                correlation = correlation_or_nan(left[:, child], right[:, parent])
                if not np.isfinite(correlation):
                    continue
                if child == parent:
                    max_auto = max(max_auto, abs(correlation))
                else:
                    max_cross = max(max_cross, abs(correlation))
    return {
        "innovation_max_abs_autocorrelation_lags_1_to_12": max_auto,
        "innovation_max_abs_crosslag_correlation_lags_1_to_12": max_cross,
    }


def coefficient_records(
    replicate: int,
    labels: Sequence[str],
    truth_b0: np.ndarray,
    truth_lagged: np.ndarray,
    fitted_b0: np.ndarray,
    fitted_lagged: np.ndarray,
    reference_std: np.ndarray,
    edge_threshold: float,
) -> pd.DataFrame:
    """Return one long-form row per estimable structural coefficient."""
    truth = np.concatenate([truth_b0[None, :, :], truth_lagged], axis=0)
    fitted = np.concatenate([fitted_b0[None, :, :], fitted_lagged], axis=0)
    records: list[dict[str, Any]] = []
    for lag in range(len(truth)):
        for child, target in enumerate(labels):
            for parent, source in enumerate(labels):
                if lag == 0 and child == parent:
                    continue
                scale = reference_std[parent] / reference_std[child]
                true_value = float(truth[lag, child, parent])
                fitted_value = float(fitted[lag, child, parent])
                records.append(
                    {
                        "replicate": replicate,
                        "lag": lag,
                        "matrix": "contemporaneous" if lag == 0 else f"lag_{lag}",
                        "target": target,
                        "source": source,
                        "true_coefficient": true_value,
                        "estimated_coefficient": fitted_value,
                        "error": fitted_value - true_value,
                        "true_standardized_coefficient": true_value * scale,
                        "estimated_standardized_coefficient": fitted_value * scale,
                        "standardized_error": (fitted_value - true_value) * scale,
                        "true_edge": abs(true_value) >= edge_threshold,
                        "estimated_edge": abs(fitted_value) >= edge_threshold,
                    }
                )
    return pd.DataFrame(records)


def metric_rows(
    replicate: int,
    coefficients: pd.DataFrame,
    *,
    true_radius: float,
    recovered_radius: float,
    innovation_metrics: Mapping[str, float],
    edge_threshold: float,
) -> pd.DataFrame:
    """Calculate coefficient and graph-recovery metrics by matrix scope."""
    scopes: list[tuple[str, pd.DataFrame]] = [("all", coefficients)]
    scopes.extend((name, group) for name, group in coefficients.groupby("matrix", sort=False))
    records: list[dict[str, Any]] = []
    for scope, frame in scopes:
        true = frame["true_coefficient"].to_numpy(dtype=float)
        estimated = frame["estimated_coefficient"].to_numpy(dtype=float)
        standardized_error = frame["standardized_error"].to_numpy(dtype=float)
        true_edge_mask = np.abs(true) >= edge_threshold
        true_edge_error = (estimated - true)[true_edge_mask]
        true_edge_standardized_error = standardized_error[true_edge_mask]
        record: dict[str, Any] = {
            "replicate": replicate,
            "scope": scope,
            "n_coefficients": len(frame),
            "coefficient_bias": float(np.mean(estimated - true)),
            "coefficient_mae": float(np.mean(np.abs(estimated - true))),
            "coefficient_rmse": float(np.sqrt(np.mean((estimated - true) ** 2))),
            "standardized_coefficient_mae": float(np.mean(np.abs(standardized_error))),
            "standardized_coefficient_rmse": float(np.sqrt(np.mean(standardized_error**2))),
            "coefficient_correlation": correlation_or_nan(true, estimated),
            "true_edge_coefficient_mae": (
                float(np.mean(np.abs(true_edge_error)))
                if len(true_edge_error)
                else math.nan
            ),
            "true_edge_coefficient_rmse": (
                float(np.sqrt(np.mean(true_edge_error**2)))
                if len(true_edge_error)
                else math.nan
            ),
            "true_edge_standardized_coefficient_mae": (
                float(np.mean(np.abs(true_edge_standardized_error)))
                if len(true_edge_standardized_error)
                else math.nan
            ),
            "true_edge_standardized_coefficient_rmse": (
                float(np.sqrt(np.mean(true_edge_standardized_error**2)))
                if len(true_edge_standardized_error)
                else math.nan
            ),
            "true_stability_radius": true_radius,
            "recovered_stability_radius": recovered_radius,
            "recovered_dynamically_stable": recovered_radius < 1.0,
        }
        record.update(support_metrics(true, estimated, edge_threshold))
        record.update(innovation_metrics)
        records.append(record)
    return pd.DataFrame(records)


def residualization_records(
    replicate: int,
    source: SourceModel,
    fitted_models: pd.DataFrame,
    fitted_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare stored and re-estimated deterministic components."""
    true_lookup = source.residualization_models.set_index("variable")
    fitted_lookup = fitted_models.set_index("variable")
    coefficient_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for index, (label, variable) in enumerate(
        zip(source.labels, source.raw_variables, strict=True)
    ):
        true_coefficients = json.loads(str(true_lookup.loc[variable, "coefficients_json"]))
        fitted_coefficients = json.loads(str(fitted_lookup.loc[variable, "coefficients_json"]))
        for term in true_coefficients:
            true_value = float(true_coefficients[term])
            fitted_value = float(fitted_coefficients[term])
            coefficient_rows.append(
                {
                    "replicate": replicate,
                    "variable": variable,
                    "residual_variable": label,
                    "term": term,
                    "true_coefficient": true_value,
                    "estimated_coefficient": fitted_value,
                    "error": fitted_value - true_value,
                }
            )
        recovered = fitted_frame[f"{variable}{source.expected_suffix}"].to_numpy(dtype=float)
        truth = source.true_baseline[:, index]
        scale = float(np.std(source.original_residuals[:, index], ddof=1))
        rmse = float(np.sqrt(np.mean((recovered - truth) ** 2)))
        baseline_rows.append(
            {
                "replicate": replicate,
                "variable": variable,
                "residual_variable": label,
                "baseline_bias": float(np.mean(recovered - truth)),
                "baseline_mae": float(np.mean(np.abs(recovered - truth))),
                "baseline_rmse": rmse,
                "baseline_rmse_in_residual_sd": rmse / scale if scale > 0 else math.nan,
            }
        )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(baseline_rows)


def dynamic_records(
    replicate: int,
    labels: Sequence[str],
    target: str,
    truth_b0: np.ndarray,
    truth_lagged: np.ndarray,
    fitted_b0: np.ndarray,
    fitted_lagged: np.ndarray,
    horizon: int,
    reference_values: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> pd.DataFrame:
    """Compare known and recovered impulse and cumulative effects."""
    truth, truth_cumulative_matrices, _ = dynamic_effect_matrices(
        truth_b0, truth_lagged, horizon
    )
    estimated, estimated_cumulative_matrices, _ = dynamic_effect_matrices(
        fitted_b0, fitted_lagged, horizon
    )
    target_index = list(labels).index(target)
    target_delta = float(
        np.quantile(reference_values[:, target_index], high_quantile)
        - np.quantile(reference_values[:, target_index], low_quantile)
    )
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(labels):
        if source == target:
            continue
        source_delta = float(
            np.quantile(reference_values[:, source_index], high_quantile)
            - np.quantile(reference_values[:, source_index], low_quantile)
        )
        scale = source_delta / target_delta if target_delta != 0.0 else math.nan
        for step in range(horizon + 1):
            true_effect = float(truth[step, target_index, source_index])
            estimated_effect = float(estimated[step, target_index, source_index])
            true_cumulative_value = float(
                truth_cumulative_matrices[step, target_index, source_index]
            )
            estimated_cumulative_value = float(
                estimated_cumulative_matrices[step, target_index, source_index]
            )
            records.append(
                {
                    "replicate": replicate,
                    "source": source,
                    "target": target,
                    "horizon": step,
                    "true_effect": true_effect,
                    "estimated_effect": estimated_effect,
                    "effect_error": estimated_effect - true_effect,
                    "true_cumulative_effect": true_cumulative_value,
                    "estimated_cumulative_effect": estimated_cumulative_value,
                    "cumulative_effect_error": (
                        estimated_cumulative_value - true_cumulative_value
                    ),
                    "source_quantile_delta": source_delta,
                    "target_quantile_delta": target_delta,
                    "quantile_scale": scale,
                    "true_scaled_effect": true_effect * scale,
                    "estimated_scaled_effect": estimated_effect * scale,
                    "scaled_effect_error": (estimated_effect - true_effect) * scale,
                    "true_scaled_cumulative_effect": (
                        true_cumulative_value * scale
                    ),
                    "estimated_scaled_cumulative_effect": (
                        estimated_cumulative_value * scale
                    ),
                    "scaled_cumulative_effect_error": (
                        estimated_cumulative_value - true_cumulative_value
                    )
                    * scale,
                }
            )
    return pd.DataFrame(records)


def run_replicate(
    replicate: int,
    seed: int,
    source: SourceModel,
    *,
    n_samples: int,
    burnin: int,
    noise_mode: str,
    min_fit_samples: int,
    min_month_samples: int,
    n_lags: int,
    criterion: str | None,
    prune: bool,
    edge_threshold: float,
    horizon: int,
    target: str,
    true_radius: float,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, Any]:
    """Simulate, preprocess, refit, and score one Monte Carlo replicate."""
    try:
        rng = np.random.default_rng(seed)
        latent = simulate_structural_var(
            source.contemporaneous,
            source.lagged,
            source.innovation_pool,
            n_samples=n_samples,
            burnin=burnin,
            rng=rng,
            noise_mode=noise_mode,
        )
        raw = source.dates.copy()
        for index, variable in enumerate(source.raw_variables):
            raw[variable] = source.true_baseline[:, index] + latent[:, index]
        fitted_frame, fitted_models = residualize_synthetic_frame(
            raw,
            source,
            min_fit_samples=min_fit_samples,
            min_month_samples=min_month_samples,
        )
        recovered_values = fitted_frame[list(source.labels)].to_numpy(dtype=float)
        fitted_b0, fitted_lagged = fit_varlingam_point(
            recovered_values,
            labels=source.labels,
            n_lags=n_lags,
            criterion=criterion,
            prune=prune,
        )
        _, reduced_lagged = reduced_form_matrices(fitted_b0, fitted_lagged)
        recovered_radius = stability_radius(reduced_lagged)
        _, reduced_innovations = fit_reduced_form_var(recovered_values, n_lags)
        innovation_metrics = temporal_correlation_metrics(reduced_innovations)
        reference_std = np.std(source.original_residuals, axis=0, ddof=1)
        coefficients = coefficient_records(
            replicate,
            source.labels,
            source.contemporaneous,
            source.lagged,
            fitted_b0,
            fitted_lagged,
            reference_std,
            edge_threshold,
        )
        metrics = metric_rows(
            replicate,
            coefficients,
            true_radius=true_radius,
            recovered_radius=recovered_radius,
            innovation_metrics=innovation_metrics,
            edge_threshold=edge_threshold,
        )
        residualization, baseline = residualization_records(
            replicate, source, fitted_models, fitted_frame
        )
        dynamics = dynamic_records(
            replicate,
            source.labels,
            target,
            source.contemporaneous,
            source.lagged,
            fitted_b0,
            fitted_lagged,
            horizon,
            source.original_residuals,
            low_quantile,
            high_quantile,
        )
        example = None
        if replicate == 0:
            example = raw[["row", "col", "year", "month"]].copy()
            for index, (label, variable) in enumerate(
                zip(source.labels, source.raw_variables, strict=True)
            ):
                example[f"{label}__simulated_before_residualization"] = latent[:, index]
                example[f"{label}__recovered_residual"] = recovered_values[:, index]
                example[f"{variable}__true_baseline"] = source.true_baseline[:, index]
                example[f"{variable}__recovered_baseline"] = fitted_frame[
                    f"{variable}{source.expected_suffix}"
                ].to_numpy(dtype=float)
                example[f"{variable}__synthetic_raw"] = raw[variable]
        return {
            "status": {
                "replicate": replicate,
                "seed": seed,
                "status": "fit",
                "error": None,
            },
            "coefficients": coefficients,
            "metrics": metrics,
            "residualization": residualization,
            "baseline": baseline,
            "dynamics": dynamics,
            "example": example,
        }
    except Exception as exc:  # retain failed replicate details in the audit output
        return {
            "status": {
                "replicate": replicate,
                "seed": seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "coefficients": None,
            "metrics": None,
            "residualization": None,
            "baseline": None,
            "dynamics": None,
            "example": None,
        }


def concatenate_results(results: Sequence[Mapping[str, Any]], key: str) -> pd.DataFrame:
    """Concatenate non-empty data frames from successful replicates."""
    frames = [result[key] for result in results if result.get(key) is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo metrics with robust quantiles."""
    value_columns = [
        "coefficient_bias",
        "coefficient_mae",
        "coefficient_rmse",
        "standardized_coefficient_mae",
        "standardized_coefficient_rmse",
        "coefficient_correlation",
        "true_edge_coefficient_mae",
        "true_edge_coefficient_rmse",
        "true_edge_standardized_coefficient_mae",
        "true_edge_standardized_coefficient_rmse",
        "precision",
        "recall",
        "f1",
        "sign_agreement",
        "structural_hamming_distance",
        "recovered_stability_radius",
        "recovered_dynamically_stable",
        "innovation_max_abs_autocorrelation_lags_1_to_12",
        "innovation_max_abs_crosslag_correlation_lags_1_to_12",
    ]
    rows: list[dict[str, Any]] = []
    for scope, frame in metrics.groupby("scope", sort=False):
        for metric in value_columns:
            values = (
                pd.to_numeric(frame[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            if not len(values):
                continue
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    """Write publication and preview versions of a figure."""
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_coefficient_recovery(coefficients: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot known coefficients against median recovered coefficients."""
    grouped = (
        coefficients.groupby(["matrix", "target", "source"], sort=False)
        .agg(
            truth=("true_standardized_coefficient", "first"),
            median=("estimated_standardized_coefficient", "median"),
            q05=("estimated_standardized_coefficient", lambda x: x.quantile(0.05)),
            q95=("estimated_standardized_coefficient", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    colors = plt.get_cmap("tab10")
    for color_index, (matrix, frame) in enumerate(grouped.groupby("matrix", sort=False)):
        axis.vlines(
            frame["truth"],
            frame["q05"],
            frame["q95"],
            color=colors(color_index),
            alpha=0.25,
            linewidth=0.8,
        )
        axis.scatter(
            frame["truth"],
            frame["median"],
            s=22,
            alpha=0.75,
            label=matrix,
            color=colors(color_index),
        )
    limits = np.asarray(
        [
            grouped[["truth", "q05", "q95"]].min().min(),
            grouped[["truth", "q05", "q95"]].max().max(),
        ]
    )
    padding = max(0.05, 0.05 * float(np.ptp(limits)))
    axis.plot(
        limits + [-padding, padding],
        limits + [-padding, padding],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="perfect recovery",
    )
    axis.set_xlim(limits[0] - padding, limits[1] + padding)
    axis.set_ylim(limits[0] - padding, limits[1] + padding)
    axis.axhline(0.0, color="0.8", linewidth=0.7)
    axis.axvline(0.0, color="0.8", linewidth=0.7)
    axis.set_xlabel("Known standardized coefficient")
    axis.set_ylabel("Median recovered standardized coefficient")
    axis.set_title("Single-pixel synthetic coefficient recovery")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.15)
    save_figure(figure, output_dir, "coefficient_recovery", dpi)


def plot_recovery_metrics(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot overall graph-support recovery distributions."""
    overall = metrics[metrics["scope"] == "all"]
    names = ["precision", "recall", "f1", "sign_agreement"]
    values = [pd.to_numeric(overall[name], errors="coerce").dropna() for name in names]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.boxplot(
        values,
        tick_labels=["Precision", "Recall", "F1", "Sign agreement"],
        showfliers=False,
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Fraction")
    axis.set_title("Known-edge recovery across synthetic replicates")
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, output_dir, "graph_support_recovery", dpi)


def plot_dynamic_recovery(
    dynamics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    *,
    cumulative: bool = False,
) -> None:
    """Plot known and recovered target responses for every source variable."""
    true_column = (
        "true_scaled_cumulative_effect" if cumulative else "true_scaled_effect"
    )
    estimated_column = (
        "estimated_scaled_cumulative_effect"
        if cumulative
        else "estimated_scaled_effect"
    )
    sources = list(dict.fromkeys(dynamics["source"].astype(str)))
    n_columns = 2
    n_rows = math.ceil(len(sources) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(11.0, 3.2 * n_rows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, source in zip(axes_array, sources, strict=False):
        frame = dynamics[dynamics["source"] == source]
        summary = frame.groupby("horizon").agg(
            truth=(true_column, "first"),
            median=(estimated_column, "median"),
            q05=(estimated_column, lambda x: x.quantile(0.05)),
            q95=(estimated_column, lambda x: x.quantile(0.95)),
        )
        horizon = summary.index.to_numpy(dtype=int)
        axis.fill_between(
            horizon,
            summary["q05"],
            summary["q95"],
            color="#4C78A8",
            alpha=0.22,
            label="recovered 5--95%",
        )
        axis.plot(
            horizon,
            summary["median"],
            color="#4C78A8",
            marker="o",
            markersize=3,
            label="recovered median",
        )
        axis.plot(
            horizon,
            summary["truth"],
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label="known effect",
        )
        axis.axhline(0.0, color="0.55", linewidth=0.7)
        axis.set_title(source)
        axis.set_ylabel(
            "IQR-scaled cumulative effect"
            if cumulative
            else "IQR-scaled dynamic effect"
        )
        axis.grid(alpha=0.15)
    for axis in axes_array[-n_columns:]:
        axis.set_xlabel("Horizon (months)")
    for axis in axes_array[len(sources):]:
        axis.set_visible(False)
    if sources:
        axes_array[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Recovery of target cumulative effects"
        if cumulative
        else "Recovery of target dynamic effects",
        y=1.01,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        "cumulative_effect_recovery" if cumulative else "dynamic_effect_recovery",
        dpi,
    )


def aggregate_dynamic_effects(dynamics: pd.DataFrame) -> pd.DataFrame:
    """Summarize pointwise recovery of dynamic and cumulative effects."""
    rows: list[dict[str, Any]] = []
    for keys, frame in dynamics.groupby(
        ["source", "target", "horizon"], sort=False
    ):
        source, target, horizon = keys
        effect_error = frame["effect_error"].to_numpy(dtype=float)
        cumulative_error = frame["cumulative_effect_error"].to_numpy(dtype=float)
        scaled_effect_error = frame["scaled_effect_error"].to_numpy(dtype=float)
        scaled_cumulative_error = frame[
            "scaled_cumulative_effect_error"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "source": source,
                "target": target,
                "horizon": horizon,
                "true_effect": float(frame["true_effect"].iloc[0]),
                "estimated_effect_median": float(frame["estimated_effect"].median()),
                "estimated_effect_q05": float(frame["estimated_effect"].quantile(0.05)),
                "estimated_effect_q95": float(frame["estimated_effect"].quantile(0.95)),
                "effect_bias": float(np.mean(effect_error)),
                "effect_rmse": float(np.sqrt(np.mean(effect_error**2))),
                "true_cumulative_effect": float(
                    frame["true_cumulative_effect"].iloc[0]
                ),
                "estimated_cumulative_effect_median": float(
                    frame["estimated_cumulative_effect"].median()
                ),
                "estimated_cumulative_effect_q05": float(
                    frame["estimated_cumulative_effect"].quantile(0.05)
                ),
                "estimated_cumulative_effect_q95": float(
                    frame["estimated_cumulative_effect"].quantile(0.95)
                ),
                "cumulative_effect_bias": float(np.mean(cumulative_error)),
                "cumulative_effect_rmse": float(
                    np.sqrt(np.mean(cumulative_error**2))
                ),
                "true_scaled_effect": float(frame["true_scaled_effect"].iloc[0]),
                "estimated_scaled_effect_median": float(
                    frame["estimated_scaled_effect"].median()
                ),
                "estimated_scaled_effect_q05": float(
                    frame["estimated_scaled_effect"].quantile(0.05)
                ),
                "estimated_scaled_effect_q95": float(
                    frame["estimated_scaled_effect"].quantile(0.95)
                ),
                "scaled_effect_bias": float(np.mean(scaled_effect_error)),
                "scaled_effect_rmse": float(
                    np.sqrt(np.mean(scaled_effect_error**2))
                ),
                "true_scaled_cumulative_effect": float(
                    frame["true_scaled_cumulative_effect"].iloc[0]
                ),
                "estimated_scaled_cumulative_effect_median": float(
                    frame["estimated_scaled_cumulative_effect"].median()
                ),
                "estimated_scaled_cumulative_effect_q05": float(
                    frame["estimated_scaled_cumulative_effect"].quantile(0.05)
                ),
                "estimated_scaled_cumulative_effect_q95": float(
                    frame["estimated_scaled_cumulative_effect"].quantile(0.95)
                ),
                "scaled_cumulative_effect_bias": float(
                    np.mean(scaled_cumulative_error)
                ),
                "scaled_cumulative_effect_rmse": float(
                    np.sqrt(np.mean(scaled_cumulative_error**2))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_baseline_recovery(baseline: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot residualization-baseline error by variable."""
    order = list(dict.fromkeys(baseline["residual_variable"].astype(str)))
    values = [
        baseline.loc[
            baseline["residual_variable"] == label,
            "baseline_rmse_in_residual_sd",
        ].dropna()
        for label in order
    ]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.boxplot(values, tick_labels=order, showfliers=False)
    axis.set_ylabel("Baseline RMSE / original residual SD")
    axis.set_title("Recovery of monthly seasonality and linear trend")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output_dir, "residualization_recovery", dpi)


def write_report(
    path: Path,
    *,
    source: SourceModel,
    statuses: pd.DataFrame,
    summary: pd.DataFrame,
    n_samples: int,
    noise_mode: str,
    edge_threshold: float,
) -> None:
    """Write a concise, self-contained interpretation guide."""
    successful = int((statuses["status"] == "fit").sum())
    failed = int((statuses["status"] != "fit").sum())
    overall = summary[summary["scope"] == "all"].set_index("metric")

    def median(name: str) -> str:
        return f"{float(overall.loc[name, 'median']):.3f}" if name in overall.index else "NA"

    text = f"""# Single-pixel synthetic VAR--LiNGAM validation

Pixel: row `{source.row}`, column `{source.col}`  
Synthetic record length: `{n_samples}` months  
Successful fits: `{successful}`; failed fits: `{failed}`  
Innovation generator: `{noise_mode}`  
Edge threshold used for support recovery: `{edge_threshold:g}`

## What was tested

The stored pixel graph was treated as known truth.  Each replicate generated a
new structural VAR series, added the pixel's stored monthly seasonal pattern
and linear trend, re-estimated and removed that deterministic baseline, and
then refitted the same VAR--LiNGAM point model.  The comparison therefore tests
the complete residualization-plus-discovery workflow at the available record
length, rather than fitting directly to idealized residuals.

The main all-matrix median results were: coefficient correlation
`{median('coefficient_correlation')}`, standardized coefficient RMSE
`{median('standardized_coefficient_rmse')}`, standardized RMSE over known
nonzero edges `{median('true_edge_standardized_coefficient_rmse')}`, edge precision
`{median('precision')}`, edge recall `{median('recall')}`, edge F1
`{median('f1')}`, and sign agreement `{median('sign_agreement')}`.

Interpret coefficient recovery using both the scatter plot and the numerical
tables.  Precision measures how often recovered edges were truly present;
recall measures how many known edges were found.  Dynamic-effect recovery is a
separate and scientifically important check because different coefficient
errors can partly cancel or accumulate along causal paths.

## Important limitation

This is an internal parameter-recovery experiment, not independent evidence
that the real-data graph is causally correct.  Its ground truth is the graph
previously estimated at this pixel.  The default independently resamples each
structural-error component, preserving its empirical marginal distribution
while enforcing the independence assumed by VAR--LiNGAM.  Run the joint-error
and Gaussian modes as sensitivity or negative-control experiments.

## Main files

- `coefficient_recovery.csv`: every known and recovered coefficient.
- `replicate_metrics.csv` and `summary_metrics.csv`: graph and coefficient metrics.
- `dynamic_effect_recovery.csv`: horizon-specific and cumulative target effects.
- `dynamic_effect_summary.csv`: median, 5--95% range, bias, and RMSE by horizon.
- `residualization_recovery.csv`: deterministic-model coefficient recovery.
- `residualization_baseline_recovery.csv`: baseline curve errors.
- `source_causal_coefficients.csv` and `source_residualization_parameters.csv`:
  the exact data-generating parameters.
- `source_innovation_moments.csv`: marginal skewness and excess kurtosis of the
  empirical structural-error pools.
- `replicate_status.csv`: failures are retained rather than silently dropped.
- `synthetic_series_example.csv`: the first complete synthetic replicate.
"""
    path.write_text(text, encoding="utf-8")


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option("--residualization-model-table", default=None)
@click.option(
    "--graphs-db",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--graphs-table", default="pixel_graphs", show_default=True)
@click.option("--row", required=True, type=int, help="Reference-grid row of the source pixel.")
@click.option("--col", required=True, type=int, help="Reference-grid column of the source pixel.")
@click.option(
    "--target",
    default=None,
    help="Residual target label; defaults to config reference_var.",
)
@click.option("--n-samples", default=240, show_default=True, type=click.IntRange(min=24))
@click.option("--replicates", default=200, show_default=True, type=click.IntRange(min=1))
@click.option("--burnin", default=200, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--noise-mode",
    type=click.Choice(NOISE_MODES),
    default="empirical-independent",
    show_default=True,
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--var-lags", default=None, type=click.IntRange(min=1))
@click.option(
    "--var-criterion",
    default=None,
    type=click.Choice(["source", "none", "aic", "fpe", "hqic", "bic"]),
    show_default=True,
)
@click.option("--var-prune/--no-var-prune", default=None)
@click.option("--edge-threshold", default=0.01, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--horizon", default=12, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--low-quantile",
    default=0.10,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
)
@click.option(
    "--high-quantile",
    default=0.90,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
)
@click.option("--min-fit-samples", default=60, show_default=True, type=click.IntRange(min=4))
@click.option("--min-month-samples", default=3, show_default=True, type=click.IntRange(min=1))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--dpi", default=300, show_default=True, type=click.IntRange(min=72))
def validate_varlingam_synthetic(
    config_path: Path,
    input_db: Path | None,
    input_table: str | None,
    residualization_model_table: str | None,
    graphs_db: Path,
    graphs_table: str,
    row: int,
    col: int,
    target: str | None,
    n_samples: int,
    replicates: int,
    burnin: int,
    noise_mode: str,
    seed: int,
    workers: int,
    var_lags: int | None,
    var_criterion: str | None,
    var_prune: bool | None,
    edge_threshold: float,
    horizon: int,
    low_quantile: float,
    high_quantile: float,
    min_fit_samples: int,
    min_month_samples: int,
    output_dir: Path,
    dpi: int,
) -> None:
    """Run a repeated, full-pipeline synthetic recovery experiment for one pixel."""
    if high_quantile <= low_quantile:
        raise click.BadParameter("high-quantile must be greater than low-quantile.")
    if n_samples < min_fit_samples:
        raise click.BadParameter(
            "n-samples must be at least min-fit-samples for residualization."
        )
    config = read_config(config_path)
    base_dir = config_path.parent
    graph_config = config.get("graph_discovery") or {}
    input_db = resolve_path(
        base_dir,
        input_db or graph_config.get("input_db") or config.get("timeseries_db"),
        base_dir / f"{config['name']}_ard.duckdb",
    )
    input_table = str(
        input_table
        or graph_config.get("input_table")
        or config.get("timeseries_table")
        or config["name"]
    )
    residualization_model_table = str(
        residualization_model_table or f"{input_table}_residualization_models"
    )
    require_files([input_db, graphs_db])
    graph = one_graph_row(graphs_db, graphs_table, row, col)
    source_lags = int(graph["var_lags"])
    n_lags = int(var_lags if var_lags is not None else source_lags)
    if n_lags != source_lags:
        raise click.BadParameter(
            "The fitted lag order must equal the source graph lag order for direct "
            f"coefficient recovery ({source_lags})."
        )
    stored_criterion = graph.get("var_criterion")
    if isinstance(stored_criterion, float) and np.isnan(stored_criterion):
        stored_criterion = None
    if var_criterion in {None, "source"}:
        criterion = str(stored_criterion) if stored_criterion else None
    else:
        criterion = None if var_criterion == "none" else var_criterion
    prune = bool(graph.get("var_prune", True)) if var_prune is None else var_prune

    source = load_source_model(
        input_db=input_db,
        input_table=input_table,
        model_table=residualization_model_table,
        graph_db=graphs_db,
        graph_table=graphs_table,
        row=row,
        col=col,
        config=config,
        n_samples=n_samples,
    )
    target = target or str(config.get("reference_var", ""))
    if target not in source.labels:
        residualization = config.get("residualization") or {}
        suffix = str(residualization.get("suffix", "_resid"))
        if f"{target}{suffix}" in source.labels:
            target = f"{target}{suffix}"
        else:
            raise click.BadParameter(
                f"Target {target!r} is not among graph variables {list(source.labels)}."
            )

    _, true_reduced_lagged = reduced_form_matrices(
        source.contemporaneous, source.lagged
    )
    true_radius = stability_radius(true_reduced_lagged)
    if not np.isfinite(true_radius) or true_radius >= 1.0:
        raise click.ClickException(
            "The selected source graph is not dynamically stable "
            f"(spectral radius={true_radius:.6g}); simulation would not be valid."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(seed)
    replicate_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(replicates)
    ]
    click.echo(
        f"Simulating {replicates} records of {n_samples} months for pixel "
        f"(row={row}, col={col}) with {workers} worker(s)..."
    )
    jobs = (
        delayed(run_replicate)(
            index,
            replicate_seeds[index],
            source,
            n_samples=n_samples,
            burnin=burnin,
            noise_mode=noise_mode,
            min_fit_samples=min_fit_samples,
            min_month_samples=min_month_samples,
            n_lags=n_lags,
            criterion=criterion,
            prune=prune,
            edge_threshold=edge_threshold,
            horizon=horizon,
            target=target,
            true_radius=true_radius,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
        )
        for index in range(replicates)
    )
    results = Parallel(n_jobs=workers, verbose=5)(jobs)
    statuses = pd.DataFrame([result["status"] for result in results])
    coefficients = concatenate_results(results, "coefficients")
    metrics = concatenate_results(results, "metrics")
    residualization = concatenate_results(results, "residualization")
    baseline = concatenate_results(results, "baseline")
    dynamics = concatenate_results(results, "dynamics")
    example_frames = [result["example"] for result in results if result.get("example") is not None]
    if coefficients.empty:
        statuses.to_csv(output_dir / "replicate_status.csv", index=False)
        raise click.ClickException(
            "Every synthetic fit failed. See replicate_status.csv for errors."
        )
    summary = aggregate_metrics(metrics)
    dynamic_summary = aggregate_dynamic_effects(dynamics)

    statuses.to_csv(output_dir / "replicate_status.csv", index=False)
    coefficients.to_csv(output_dir / "coefficient_recovery.csv", index=False)
    metrics.to_csv(output_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    residualization.to_csv(output_dir / "residualization_recovery.csv", index=False)
    baseline.to_csv(output_dir / "residualization_baseline_recovery.csv", index=False)
    dynamics.to_csv(output_dir / "dynamic_effect_recovery.csv", index=False)
    dynamic_summary.to_csv(output_dir / "dynamic_effect_summary.csv", index=False)
    source.residualization_models.to_csv(
        output_dir / "source_residualization_parameters.csv", index=False
    )
    reference_std = np.std(source.original_residuals, axis=0, ddof=1)
    source_coefficients = coefficient_records(
        -1,
        source.labels,
        source.contemporaneous,
        source.lagged,
        source.contemporaneous,
        source.lagged,
        reference_std,
        edge_threshold,
    )
    source_coefficients[
        [
            "lag",
            "matrix",
            "target",
            "source",
            "true_coefficient",
            "true_standardized_coefficient",
            "true_edge",
        ]
    ].to_csv(output_dir / "source_causal_coefficients.csv", index=False)
    pd.DataFrame(source.innovation_pool, columns=source.labels).to_csv(
        output_dir / "source_structural_innovations.csv", index=False
    )
    innovation_moments(source.innovation_pool, source.labels).to_csv(
        output_dir / "source_innovation_moments.csv", index=False
    )
    if example_frames:
        example_frames[0].to_csv(output_dir / "synthetic_series_example.csv", index=False)

    pool_centered = source.innovation_pool - np.mean(source.innovation_pool, axis=0)
    pool_correlation = np.corrcoef(pool_centered, rowvar=False)
    off_diagonal = pool_correlation[~np.eye(len(source.labels), dtype=bool)]
    parameters = {
        "config_path": str(config_path),
        "input_db": str(input_db),
        "input_table": input_table,
        "residualization_model_table": residualization_model_table,
        "graphs_db": str(graphs_db),
        "graphs_table": graphs_table,
        "row": row,
        "col": col,
        "labels": list(source.labels),
        "raw_variables": list(source.raw_variables),
        "target": target,
        "n_samples": n_samples,
        "replicates": replicates,
        "burnin": burnin,
        "noise_mode": noise_mode,
        "seed": seed,
        "var_lags": n_lags,
        "var_criterion": criterion,
        "var_prune": prune,
        "edge_threshold": edge_threshold,
        "horizon": horizon,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "seasonal_model": source.seasonal_model,
        "include_trend": source.include_trend,
        "fit_end_year": source.fit_end_year,
        "true_stability_radius": true_radius,
        "innovation_pool_size": len(source.innovation_pool),
        "source_innovation_max_abs_correlation": float(np.nanmax(np.abs(off_diagonal))),
        "successful_replicates": int((statuses["status"] == "fit").sum()),
        "failed_replicates": int((statuses["status"] == "failed").sum()),
        "contemporaneous_matrix": source.contemporaneous.tolist(),
        "lagged_matrices": source.lagged.tolist(),
    }
    (output_dir / "simulation_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    plot_coefficient_recovery(coefficients, output_dir, dpi)
    plot_recovery_metrics(metrics, output_dir, dpi)
    plot_dynamic_recovery(dynamics, output_dir, dpi)
    plot_dynamic_recovery(dynamics, output_dir, dpi, cumulative=True)
    plot_baseline_recovery(baseline, output_dir, dpi)
    write_report(
        output_dir / "validation_report.md",
        source=source,
        statuses=statuses,
        summary=summary,
        n_samples=n_samples,
        noise_mode=noise_mode,
        edge_threshold=edge_threshold,
    )

    overall = summary[summary["scope"] == "all"].set_index("metric")
    click.echo("")
    click.echo("Synthetic validation complete.")
    click.echo(
        f"Successful replicates: {(statuses['status'] == 'fit').sum()}/{replicates}"
    )
    for metric_name in [
        "coefficient_correlation",
        "standardized_coefficient_rmse",
        "precision",
        "recall",
        "f1",
        "sign_agreement",
    ]:
        if metric_name in overall.index:
            click.echo(
                f"Median {metric_name}: {float(overall.loc[metric_name, 'median']):.3f}"
            )
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Report: {output_dir / 'validation_report.md'}")


if __name__ == "__main__":
    validate_varlingam_synthetic()
