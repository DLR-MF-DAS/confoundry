"""Compute post-hoc statistics and LiNGAM diagnostics for saved graphs.

This command reads per-pixel graph-discovery output and reconstructs the same
data matrices from the configured time-series DuckDB input. DirectLiNGAM errors
use the contemporaneous matrix. For VAR-LiNGAM, contemporaneous diagnostics use
the saved post-pruning structural matrices, while temporal diagnostics refit
the reduced-form VAR at the saved lag order.

No LiNGAM models are refit in this script.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2
from tqdm.contrib.concurrent import process_map

PixelKey = tuple[int, int]


def get_pixel_window_group(
    pixel_key: PixelKey,
    group_lookup: Mapping[PixelKey, pd.DataFrame],
    window_size: int,
) -> pd.DataFrame | None:
    """Collect pixel groups in a square neighborhood around a center pixel."""
    if window_size < 0:
        raise ValueError("window_size must be >= 0")

    row, col = pixel_key
    groups: list[pd.DataFrame] = []

    for r in range(row - window_size, row + window_size + 1):
        for c in range(col - window_size, col + window_size + 1):
            group = group_lookup.get((r, c))
            if group is not None:
                groups.append(group)

    if not groups:
        return None

    return pd.concat(groups, ignore_index=True)


def parse_columns(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    column_specs: Sequence[Mapping[str, Any]],
    apply_shifts: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Select configured columns and optionally apply their temporal shifts."""
    shifted_df = df.sort_values(list(group_cols) + list(order_cols)).copy()
    labels: list[str] = []
    label_lags: dict[str, int] = {}

    for spec in column_specs:
        label = str(spec["name"])
        lag = int(spec["shift"])

        if label in labels:
            raise click.BadParameter(f"Duplicate derived column: {label}")
        if label not in shifted_df.columns:
            raise click.BadParameter(f"Missing data column: {label}")

        if apply_shifts:
            shifted_df[label] = shifted_df.groupby(list(group_cols))[label].shift(lag)
        labels.append(label)
        label_lags[label] = lag if apply_shifts else 0

    return shifted_df, labels, label_lags


def safe_float(value: float | np.floating | None) -> float | None:
    """Return a JSON/CSV-friendly float or ``None`` for non-finite values."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def off_diagonal_mask(n: int) -> np.ndarray:
    """Return a boolean mask selecting off-diagonal entries in an ``n x n`` matrix."""
    return ~np.eye(n, dtype=bool)


def pairwise_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute a correlation matrix while tolerating constant columns."""
    n_vars = X.shape[1]
    corr = np.full((n_vars, n_vars), np.nan, dtype=float)

    for i in range(n_vars):
        xi = X[:, i]
        xi_std = np.nanstd(xi)
        corr[i, i] = 1.0 if xi_std > 0 else np.nan

        for j in range(i + 1, n_vars):
            xj = X[:, j]
            xj_std = np.nanstd(xj)
            if xi_std <= 0 or xj_std <= 0:
                value = np.nan
            else:
                value = float(np.corrcoef(xi, xj)[0, 1])
            corr[i, j] = value
            corr[j, i] = value

    return corr


def top_matrix_pairs(
    matrix: np.ndarray,
    labels: Sequence[str],
    top_n: int,
    *,
    absolute: bool = True,
    min_abs_value: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the strongest off-diagonal pairs in a square matrix."""
    pairs: list[dict[str, Any]] = []

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            score = abs(float(value)) if absolute else float(value)
            if abs(float(value)) < min_abs_value:
                continue
            pairs.append(
                {
                    "var1": str(labels[i]),
                    "var2": str(labels[j]),
                    "value": float(value),
                    "abs_value": abs(float(value)),
                    "score": score,
                }
            )

    pairs.sort(key=lambda item: item["score"], reverse=True)
    for pair in pairs:
        pair.pop("score", None)
    return pairs[:top_n]


def residual_moment_diagnostics(
    residuals: np.ndarray,
    labels: Sequence[str],
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute cheap residual non-Gaussianity diagnostics."""
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    abs_skew: list[float] = []
    abs_kurtosis: list[float] = []

    for idx, label in enumerate(labels):
        x = residuals[:, idx].astype(float)
        x = x[np.isfinite(x)]

        if len(x) < 8:
            row = {
                "variable": str(label),
                "skew": None,
                "excess_kurtosis": None,
                "jarque_bera_stat": None,
                "jarque_bera_p": None,
                "nongaussian_at_alpha": None,
            }
            rows.append(row)
            continue

        xc = x - np.mean(x)
        m2 = float(np.mean(xc**2))
        if m2 <= 0:
            skew = np.nan
            excess_kurtosis = np.nan
            jb = np.nan
            p = np.nan
        else:
            m3 = float(np.mean(xc**3))
            m4 = float(np.mean(xc**4))
            skew = m3 / (m2 ** 1.5)
            excess_kurtosis = m4 / (m2**2) - 3.0
            jb = (len(x) / 6.0) * (skew**2 + 0.25 * excess_kurtosis**2)
            p = float(np.exp(-0.5 * jb))

        row = {
            "variable": str(label),
            "skew": safe_float(skew),
            "excess_kurtosis": safe_float(excess_kurtosis),
            "jarque_bera_stat": safe_float(jb),
            "jarque_bera_p": safe_float(p),
            "nongaussian_at_alpha": bool(np.isfinite(p) and p < alpha),
        }
        rows.append(row)

        if np.isfinite(p):
            p_values.append(float(p))
        if np.isfinite(skew):
            abs_skew.append(abs(float(skew)))
        if np.isfinite(excess_kurtosis):
            abs_kurtosis.append(abs(float(excess_kurtosis)))

    summary = {
        "residual_jb_min_p": safe_float(np.min(p_values)) if p_values else None,
        "residual_jb_median_p": safe_float(np.median(p_values)) if p_values else None,
        "residual_nongaussian_fraction": safe_float(np.mean(np.asarray(p_values) < alpha))
        if p_values
        else None,
        "residual_max_abs_skew": safe_float(np.max(abs_skew)) if abs_skew else None,
        "residual_max_abs_excess_kurtosis": safe_float(np.max(abs_kurtosis))
        if abs_kurtosis
        else None,
    }
    return summary, rows


def residual_dependence_diagnostics(
    residuals: np.ndarray,
    labels: Sequence[str],
    residual_corr_threshold: float,
    top_n: int,
) -> dict[str, Any]:
    """Compute cheap residual-dependence diagnostics."""
    corr = pairwise_correlation_matrix(residuals)
    mask = off_diagonal_mask(corr.shape[0])
    abs_values = np.abs(corr[mask])
    abs_values = abs_values[np.isfinite(abs_values)]

    return {
        "residual_max_abs_corr": safe_float(np.max(abs_values)) if len(abs_values) else None,
        "residual_median_abs_corr": safe_float(np.median(abs_values)) if len(abs_values) else None,
        "residual_corr_pairs_ge_threshold": int(np.sum(abs_values >= residual_corr_threshold))
        if len(abs_values)
        else 0,
        "residual_corr_top_pairs_json": json.dumps(
            top_matrix_pairs(
                corr,
                labels,
                top_n,
                absolute=True,
                min_abs_value=residual_corr_threshold,
            )
        ),
    }


def lag1_autocorrelation_summary(
    values: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[str],
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    threshold: float,
    top_n: int,
) -> dict[str, Any]:
    """Summarize lag-1 autocorrelation within each pixel/group."""
    if len(values) != len(metadata):
        raise ValueError("values and metadata must contain the same number of rows")

    value_df = pd.DataFrame(values, columns=list(labels), index=metadata.index)
    work = pd.concat([metadata[list(group_cols) + list(order_cols)], value_df], axis=1)
    work = work.sort_values(list(group_cols) + list(order_cols))

    records: list[dict[str, Any]] = []

    for label in labels:
        coeffs: list[float] = []
        for _, group in work.groupby(list(group_cols), sort=False):
            x = group[str(label)].dropna().to_numpy(dtype=float)
            if len(x) < 4:
                continue
            x0 = x[:-1]
            x1 = x[1:]
            if np.std(x0) <= 0 or np.std(x1) <= 0:
                continue
            r = float(np.corrcoef(x0, x1)[0, 1])
            if np.isfinite(r):
                coeffs.append(r)

        if coeffs:
            median_r = float(np.median(coeffs))
            median_abs_r = float(np.median(np.abs(coeffs)))
            max_abs_r = float(np.max(np.abs(coeffs)))
            n_groups = len(coeffs)
        else:
            median_r = np.nan
            median_abs_r = np.nan
            max_abs_r = np.nan
            n_groups = 0

        records.append(
            {
                "variable": str(label),
                "median_lag1_autocorr": safe_float(median_r),
                "median_abs_lag1_autocorr": safe_float(median_abs_r),
                "max_abs_lag1_autocorr": safe_float(max_abs_r),
                "n_groups": int(n_groups),
            }
        )

    finite_abs = [r["median_abs_lag1_autocorr"] for r in records if r["median_abs_lag1_autocorr"] is not None]
    top_records = sorted(
        [r for r in records if r["median_abs_lag1_autocorr"] is not None],
        key=lambda item: item["median_abs_lag1_autocorr"],
        reverse=True,
    )[:top_n]

    return {
        "residual_lag1_median_abs_autocorr": safe_float(np.median(finite_abs))
        if finite_abs
        else None,
        "residual_lag1_max_median_abs_autocorr": safe_float(np.max(finite_abs))
        if finite_abs
        else None,
        "residual_lag1_variables_ge_threshold": int(np.sum(np.asarray(finite_abs) >= threshold))
        if finite_abs
        else 0,
        "residual_lag1_top_variables_json": json.dumps(top_records),
    }


def residual_crosslag_diagnostics(
    values: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[str],
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    max_lag: int,
    threshold: float,
    top_n: int,
) -> dict[str, Any]:
    """Summarize all innovation cross-correlations over temporal lags."""
    if len(values) != len(metadata):
        raise ValueError(
            "values and metadata must contain the same number of rows"
        )
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")

    value_df = pd.DataFrame(
        values,
        columns=list(labels),
        index=metadata.index,
    )
    work = pd.concat(
        [metadata[list(group_cols) + list(order_cols)], value_df],
        axis=1,
    ).sort_values(list(group_cols) + list(order_cols))

    coefficients: dict[tuple[int, int, int], list[float]] = {}
    maximum_lag_evaluated = 0
    for _, group in work.groupby(list(group_cols), sort=False):
        array = group[list(labels)].to_numpy(dtype=float)
        for lag in range(1, min(max_lag, len(array) - 4) + 1):
            current = array[lag:]
            previous = array[:-lag]
            current_centered = current - np.mean(current, axis=0)
            previous_centered = previous - np.mean(previous, axis=0)
            numerator = current_centered.T @ previous_centered
            denominator = np.sqrt(
                np.sum(current_centered**2, axis=0)[:, np.newaxis]
                * np.sum(previous_centered**2, axis=0)[np.newaxis, :]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = numerator / denominator
            maximum_lag_evaluated = max(maximum_lag_evaluated, lag)
            for target_index in range(len(labels)):
                for source_index in range(len(labels)):
                    value = correlation[target_index, source_index]
                    if np.isfinite(value):
                        coefficients.setdefault(
                            (lag, target_index, source_index),
                            [],
                        ).append(float(value))

    records: list[dict[str, Any]] = []
    for (lag, target_index, source_index), values_at_lag in coefficients.items():
        array = np.asarray(values_at_lag, dtype=float)
        records.append(
            {
                "source": str(labels[source_index]),
                "target": str(labels[target_index]),
                "lag": int(lag),
                "median_correlation": safe_float(np.median(array)),
                "median_abs_correlation": safe_float(
                    np.median(np.abs(array))
                ),
                "max_abs_correlation": safe_float(
                    np.max(np.abs(array))
                ),
                "n_groups": int(len(array)),
            }
        )

    finite_abs = np.asarray(
        [
            record["median_abs_correlation"]
            for record in records
            if record["median_abs_correlation"] is not None
        ],
        dtype=float,
    )
    top_records = sorted(
        records,
        key=lambda record: (
            record["median_abs_correlation"]
            if record["median_abs_correlation"] is not None
            else -np.inf
        ),
        reverse=True,
    )[:top_n]
    return {
        "residual_crosslag_max_abs_corr": safe_float(
            np.max(finite_abs)
        )
        if len(finite_abs)
        else None,
        "residual_crosslag_median_abs_corr": safe_float(
            np.median(finite_abs)
        )
        if len(finite_abs)
        else None,
        "residual_crosslag_pairs_ge_threshold": int(
            np.sum(finite_abs >= threshold)
        )
        if len(finite_abs)
        else 0,
        "residual_crosslag_lags_requested": int(max_lag),
        "residual_crosslag_lags_evaluated": int(maximum_lag_evaluated),
        "residual_crosslag_top_pairs_json": json.dumps(top_records),
    }


def multivariate_whiteness_diagnostics(
    values: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[str],
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    max_lag: int,
    model_lags: int,
    alpha: float,
) -> dict[str, Any]:
    """Run an adjusted multivariate portmanteau innovation-whiteness test."""
    if len(values) != len(metadata):
        raise ValueError(
            "values and metadata must contain the same number of rows"
        )
    if max_lag <= model_lags:
        raise ValueError(
            "whiteness max_lag must be larger than the fitted VAR lag count"
        )

    value_df = pd.DataFrame(
        values,
        columns=list(labels),
        index=metadata.index,
    )
    work = pd.concat(
        [metadata[list(group_cols) + list(order_cols)], value_df],
        axis=1,
    ).sort_values(list(group_cols) + list(order_cols))

    records: list[dict[str, Any]] = []
    n_variables = len(labels)
    for key, group in work.groupby(list(group_cols), sort=False):
        array = group[list(labels)].to_numpy(dtype=float)
        array = array[np.all(np.isfinite(array), axis=1)]
        lags_evaluated = min(max_lag, len(array) - 1)
        if lags_evaluated <= model_lags or len(array) <= n_variables:
            continue
        centered = array - np.mean(array, axis=0)
        covariance_zero = centered.T @ centered / len(centered)
        try:
            covariance_inverse = np.linalg.inv(covariance_zero)
        except np.linalg.LinAlgError:
            continue

        statistic_sum = 0.0
        for lag in range(1, lags_evaluated + 1):
            covariance_lag = (
                centered[lag:].T @ centered[:-lag] / len(centered)
            )
            contribution = np.trace(
                covariance_lag.T
                @ covariance_inverse
                @ covariance_lag
                @ covariance_inverse
            )
            statistic_sum += float(contribution) / (len(centered) - lag)
        statistic = max(
            0.0,
            float(len(centered) ** 2 * statistic_sum),
        )
        degrees_of_freedom = int(
            n_variables**2 * (lags_evaluated - model_lags)
        )
        p_value = float(chi2.sf(statistic, degrees_of_freedom))
        normalized_key = key if isinstance(key, tuple) else (key,)
        records.append(
            {
                "group_json": json.dumps(
                    list(normalized_key),
                    default=str,
                ),
                "statistic": safe_float(statistic),
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": safe_float(p_value),
                "rejected": bool(p_value < alpha),
                "lags_evaluated": int(lags_evaluated),
                "n_samples": int(len(centered)),
            }
        )

    p_values = np.asarray(
        [
            record["p_value"]
            for record in records
            if record["p_value"] is not None
        ],
        dtype=float,
    )
    statistics = np.asarray(
        [
            record["statistic"]
            for record in records
            if record["statistic"] is not None
        ],
        dtype=float,
    )
    rejected = np.asarray(
        [record["rejected"] for record in records],
        dtype=bool,
    )
    return {
        "residual_whiteness_stat": safe_float(np.max(statistics))
        if len(statistics)
        else None,
        "residual_whiteness_p": safe_float(np.min(p_values))
        if len(p_values)
        else None,
        "residual_whiteness_rejected": bool(np.any(rejected))
        if len(rejected)
        else None,
        "residual_whiteness_reject_fraction": safe_float(
            np.mean(rejected)
        )
        if len(rejected)
        else None,
        "residual_whiteness_n_groups": int(len(records)),
        "residual_whiteness_lags_requested": int(max_lag),
        "residual_whiteness_lags_evaluated": int(
            max(
                (record["lags_evaluated"] for record in records),
                default=0,
            )
        ),
        "residual_whiteness_adjusted": True,
        "residual_whiteness_results_json": json.dumps(records),
    }


def basic_data_diagnostics(X: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    """Compute cheap numerical diagnostics for one pixel/window matrix."""
    n_samples, n_vars = X.shape
    stds = np.nanstd(X, axis=0)
    near_constant = [str(label) for label, std in zip(labels, stds, strict=True) if std <= 1e-12]

    try:
        rank = int(np.linalg.matrix_rank(X))
    except np.linalg.LinAlgError:
        rank = -1

    try:
        condition_number = safe_float(np.linalg.cond(X)) if n_samples >= n_vars else None
    except np.linalg.LinAlgError:
        condition_number = None

    x_corr = pairwise_correlation_matrix(X)
    mask = off_diagonal_mask(n_vars)
    abs_corr = np.abs(x_corr[mask])
    abs_corr = abs_corr[np.isfinite(abs_corr)]

    return {
        "n_variables": int(n_vars),
        "sample_to_variable_ratio": safe_float(n_samples / n_vars) if n_vars else None,
        "matrix_rank": rank,
        "condition_number": condition_number,
        "near_constant_variable_count": int(len(near_constant)),
        "near_constant_variables_json": json.dumps(near_constant),
        "x_max_abs_corr": safe_float(np.max(abs_corr)) if len(abs_corr) else None,
        "x_median_abs_corr": safe_float(np.median(abs_corr)) if len(abs_corr) else None,
    }


def bootstrap_probability_diagnostics(
    probabilities: np.ndarray,
    raw_adjacency: np.ndarray,
    consensus_adjacency: np.ndarray,
    labels: Sequence[str],
    min_prob: float,
    min_abs_effect: float,
    probability_band: float,
    top_n: int,
) -> dict[str, Any]:
    """Summarize already-computed bootstrap edge probabilities."""
    n_vars = len(labels)
    mask = off_diagonal_mask(n_vars)
    p = probabilities[mask]
    p = p[np.isfinite(p)]

    entropy = None
    if len(p):
        clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        entropy = -clipped * np.log2(clipped) - (1.0 - clipped) * np.log2(1.0 - clipped)

    edges: list[dict[str, Any]] = []
    bidirectional: list[dict[str, Any]] = []
    lower = max(0.0, min_prob - probability_band)
    upper = min(1.0, min_prob + probability_band)

    for child_idx, child_name in enumerate(labels):
        for parent_idx, parent_name in enumerate(labels):
            if child_idx == parent_idx:
                continue
            prob = float(probabilities[child_idx, parent_idx])
            coef = float(raw_adjacency[child_idx, parent_idx])
            if not np.isfinite(prob):
                continue
            edges.append(
                {
                    "parent": str(parent_name),
                    "child": str(child_name),
                    "probability": prob,
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "in_consensus": bool(consensus_adjacency[child_idx, parent_idx] != 0.0),
                }
            )

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pij = float(probabilities[i, j])
            pji = float(probabilities[j, i])
            if not np.isfinite(pij) or not np.isfinite(pji):
                continue
            conflict = min(pij, pji)
            if conflict > 0:
                bidirectional.append(
                    {
                        "var1": str(labels[i]),
                        "var2": str(labels[j]),
                        "prob_var1_to_var2": pji,
                        "prob_var2_to_var1": pij,
                        "bidirectional_instability": conflict,
                    }
                )

    edges.sort(key=lambda item: (item["probability"], item["abs_coefficient"]), reverse=True)
    bidirectional.sort(key=lambda item: item["bidirectional_instability"], reverse=True)

    return {
        "raw_edge_count": int(np.sum((np.abs(raw_adjacency) >= min_abs_effect) & mask)),
        "consensus_edge_count": int(np.sum(consensus_adjacency != 0.0)),
        "bootstrap_edges_ge_min_prob": int(np.sum((probabilities >= min_prob) & mask)),
        "bootstrap_edges_near_threshold": int(np.sum((probabilities >= lower) & (probabilities <= upper) & mask)),
        "bootstrap_probability_max": safe_float(np.max(p)) if len(p) else None,
        "bootstrap_probability_mean": safe_float(np.mean(p)) if len(p) else None,
        "bootstrap_probability_entropy_mean": safe_float(np.mean(entropy)) if entropy is not None else None,
        "bootstrap_top_edges_json": json.dumps(edges[:top_n]),
        "bootstrap_bidirectional_instability_max": safe_float(
            bidirectional[0]["bidirectional_instability"] if bidirectional else 0.0
        ),
        "bootstrap_bidirectional_top_pairs_json": json.dumps(bidirectional[:top_n]),
    }


def lagged_bootstrap_probability_diagnostics(
    probabilities: np.ndarray,
    raw_adjacency: np.ndarray,
    consensus_adjacency: np.ndarray,
    labels: Sequence[str],
    min_prob: float,
    min_abs_effect: float,
    probability_band: float,
    top_n: int,
) -> dict[str, Any]:
    """Summarize bootstrap support for all lagged VAR edges."""
    if probabilities.shape != raw_adjacency.shape:
        raise click.ClickException(
            "Lagged probability and raw-adjacency shapes do not match: "
            f"{probabilities.shape} versus {raw_adjacency.shape}."
        )
    if consensus_adjacency.shape != raw_adjacency.shape:
        raise click.ClickException(
            "Lagged consensus and raw-adjacency shapes do not match: "
            f"{consensus_adjacency.shape} versus {raw_adjacency.shape}."
        )

    finite_probabilities = probabilities[np.isfinite(probabilities)]
    entropy = None
    if len(finite_probabilities):
        clipped = np.clip(
            finite_probabilities,
            1e-12,
            1.0 - 1e-12,
        )
        entropy = (
            -clipped * np.log2(clipped)
            - (1.0 - clipped) * np.log2(1.0 - clipped)
        )

    lower = max(0.0, min_prob - probability_band)
    upper = min(1.0, min_prob + probability_band)
    edges: list[dict[str, Any]] = []
    by_lag: list[dict[str, Any]] = []
    for lag_index in range(probabilities.shape[0]):
        lag_probabilities = probabilities[lag_index]
        lag_raw = raw_adjacency[lag_index]
        lag_consensus = consensus_adjacency[lag_index]
        finite_lag = lag_probabilities[np.isfinite(lag_probabilities)]
        lag_entropy = None
        if len(finite_lag):
            clipped = np.clip(finite_lag, 1e-12, 1.0 - 1e-12)
            lag_entropy = (
                -clipped * np.log2(clipped)
                - (1.0 - clipped) * np.log2(1.0 - clipped)
            )
        by_lag.append(
            {
                "lag": int(lag_index + 1),
                "raw_edge_count": int(
                    np.sum(np.abs(lag_raw) >= min_abs_effect)
                ),
                "consensus_edge_count": int(
                    np.sum(lag_consensus != 0.0)
                ),
                "edges_ge_min_prob": int(
                    np.sum(lag_probabilities >= min_prob)
                ),
                "edges_near_threshold": int(
                    np.sum(
                        (lag_probabilities >= lower)
                        & (lag_probabilities <= upper)
                    )
                ),
                "probability_entropy_mean": safe_float(
                    np.mean(lag_entropy)
                )
                if lag_entropy is not None
                else None,
            }
        )
        for child_index, child in enumerate(labels):
            for parent_index, parent in enumerate(labels):
                probability = lag_probabilities[
                    child_index,
                    parent_index,
                ]
                if not np.isfinite(probability):
                    continue
                coefficient = float(
                    lag_raw[child_index, parent_index]
                )
                edges.append(
                    {
                        "lag": int(lag_index + 1),
                        "parent": str(parent),
                        "child": str(child),
                        "autoregressive": bool(
                            child_index == parent_index
                        ),
                        "probability": float(probability),
                        "coefficient": coefficient,
                        "abs_coefficient": abs(coefficient),
                        "in_consensus": bool(
                            lag_consensus[
                                child_index,
                                parent_index,
                            ]
                            != 0.0
                        ),
                    }
                )

    edges.sort(
        key=lambda item: (
            item["probability"],
            item["abs_coefficient"],
        ),
        reverse=True,
    )
    return {
        "lagged_raw_edge_count": int(
            np.sum(np.abs(raw_adjacency) >= min_abs_effect)
        ),
        "lagged_consensus_edge_count": int(
            np.sum(consensus_adjacency != 0.0)
        ),
        "lagged_bootstrap_edges_ge_min_prob": int(
            np.sum(probabilities >= min_prob)
        ),
        "lagged_bootstrap_edges_near_threshold": int(
            np.sum(
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        ),
        "lagged_bootstrap_probability_max": safe_float(
            np.max(finite_probabilities)
        )
        if len(finite_probabilities)
        else None,
        "lagged_bootstrap_probability_mean": safe_float(
            np.mean(finite_probabilities)
        )
        if len(finite_probabilities)
        else None,
        "lagged_bootstrap_probability_entropy_mean": safe_float(
            np.mean(entropy)
        )
        if entropy is not None
        else None,
        "lagged_bootstrap_top_edges_json": json.dumps(edges[:top_n]),
        "lagged_bootstrap_by_lag_json": json.dumps(by_lag),
    }


def reduced_form_var_lags(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
) -> np.ndarray:
    """Convert structural VAR lag matrices to reduced-form matrices."""
    multiplier = np.linalg.inv(
        np.eye(contemporaneous.shape[0], dtype=float)
        - contemporaneous
    )
    return np.einsum("ij,pjk->pik", multiplier, lagged)


def reduced_form_stability_radius(reduced_lagged: np.ndarray) -> float:
    """Return the reduced-form VAR companion spectral radius."""
    n_lags, n_variables, _ = reduced_lagged.shape
    companion = np.zeros(
        (n_lags * n_variables, n_lags * n_variables),
        dtype=float,
    )
    companion[:n_variables, :] = np.concatenate(
        list(reduced_lagged),
        axis=1,
    )
    if n_lags > 1:
        companion[n_variables:, :-n_variables] = np.eye(
            (n_lags - 1) * n_variables,
            dtype=float,
        )
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues)))


def var_stability_diagnostics(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    bootstrap_contemporaneous: np.ndarray,
    bootstrap_lagged: np.ndarray,
    stability_threshold: float,
    bootstrap_limit: int,
) -> dict[str, Any]:
    """Calculate point and paired-bootstrap reduced-form VAR stability."""
    point_radius = None
    point_stable = None
    try:
        reduced_lagged = reduced_form_var_lags(
            contemporaneous,
            lagged,
        )
        point_radius = reduced_form_stability_radius(reduced_lagged)
        point_stable = bool(point_radius < stability_threshold)
    except (np.linalg.LinAlgError, ValueError):
        pass

    if bootstrap_limit > 0:
        bootstrap_contemporaneous = bootstrap_contemporaneous[
            :bootstrap_limit
        ]
        bootstrap_lagged = bootstrap_lagged[:bootstrap_limit]
    radii: list[float] = []
    for bootstrap_b0, bootstrap_blags in zip(
        bootstrap_contemporaneous,
        bootstrap_lagged,
        strict=True,
    ):
        try:
            reduced_lagged = reduced_form_var_lags(
                bootstrap_b0,
                bootstrap_blags,
            )
            radius = reduced_form_stability_radius(reduced_lagged)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(radius):
            radii.append(float(radius))

    radii_array = np.asarray(radii, dtype=float)
    stable = radii_array < stability_threshold
    return {
        "var_stability_radius": safe_float(point_radius),
        "var_stable": point_stable,
        "var_stability_threshold": float(stability_threshold),
        "var_bootstrap_stability_n_total": int(
            len(bootstrap_contemporaneous)
        ),
        "var_bootstrap_stability_n_valid": int(len(radii_array)),
        "var_bootstrap_stable_fraction": safe_float(np.mean(stable))
        if len(stable)
        else None,
        "var_bootstrap_stability_radius_median": safe_float(
            np.median(radii_array)
        )
        if len(radii_array)
        else None,
        "var_bootstrap_stability_radius_q95": safe_float(
            np.quantile(radii_array, 0.95)
        )
        if len(radii_array)
        else None,
        "var_bootstrap_stability_radius_max": safe_float(
            np.max(radii_array)
        )
        if len(radii_array)
        else None,
    }


def make_diagnostics_row(
    pixel_key: PixelKey,
    complete_g: pd.DataFrame,
    X: np.ndarray,
    residuals: np.ndarray,
    temporal_residuals: np.ndarray,
    temporal_residual_basis: str,
    raw_adjacency: np.ndarray,
    probabilities: np.ndarray,
    consensus_adjacency: np.ndarray,
    labels: Sequence[str],
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    min_prob: float,
    min_abs_effect: float,
    diagnostic_alpha: float,
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    whiteness_lags: int,
    model_lags: int,
    probability_band: float,
    diagnostic_top_n: int,
) -> dict[str, Any]:
    """Build one compact sidecar diagnostics row for a fitted pixel/window."""
    if len(temporal_residuals) != len(complete_g):
        raise ValueError(
            "temporal_residuals and complete_g must contain the same number "
            "of rows"
        )
    serialized_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)
    row = dict(zip(group_cols, serialized_pixel_key, strict=False))
    row["n_samples"] = int(len(X))
    row["residual_temporal_basis"] = str(temporal_residual_basis)

    moment_summary, moment_rows = residual_moment_diagnostics(
        residuals=residuals,
        labels=labels,
        alpha=diagnostic_alpha,
    )

    row.update(basic_data_diagnostics(X, labels))
    row.update(moment_summary)
    row["residual_moments_json"] = json.dumps(moment_rows)
    row.update(
        residual_dependence_diagnostics(
            residuals=residuals,
            labels=labels,
            residual_corr_threshold=residual_corr_threshold,
            top_n=diagnostic_top_n,
        )
    )
    row.update(
        lag1_autocorrelation_summary(
            values=temporal_residuals,
            metadata=complete_g,
            labels=labels,
            group_cols=group_cols,
            order_cols=order_cols,
            threshold=autocorr_threshold,
            top_n=diagnostic_top_n,
        )
    )
    row.update(
        residual_crosslag_diagnostics(
            values=temporal_residuals,
            metadata=complete_g,
            labels=labels,
            group_cols=group_cols,
            order_cols=order_cols,
            max_lag=whiteness_lags,
            threshold=residual_crosslag_corr_threshold,
            top_n=diagnostic_top_n,
        )
    )
    row.update(
        multivariate_whiteness_diagnostics(
            values=temporal_residuals,
            metadata=complete_g,
            labels=labels,
            group_cols=group_cols,
            order_cols=order_cols,
            max_lag=whiteness_lags,
            model_lags=model_lags,
            alpha=diagnostic_alpha,
        )
    )
    row.update(
        bootstrap_probability_diagnostics(
            probabilities=probabilities,
            raw_adjacency=raw_adjacency,
            consensus_adjacency=consensus_adjacency,
            labels=labels,
            min_prob=min_prob,
            min_abs_effect=min_abs_effect,
            probability_band=probability_band,
            top_n=diagnostic_top_n,
        )
    )

    residual_corr = row.get("residual_max_abs_corr")
    autocorr = row.get("residual_lag1_max_median_abs_autocorr")
    crosslag_corr = row.get("residual_crosslag_max_abs_corr")
    whiteness_rejected = row.get("residual_whiteness_rejected")
    warning = bool(
        (residual_corr is not None and residual_corr >= residual_corr_threshold)
        or (autocorr is not None and autocorr >= autocorr_threshold)
        or (
            crosslag_corr is not None
            and crosslag_corr >= residual_crosslag_corr_threshold
        )
        or bool(whiteness_rejected)
        or (row.get("near_constant_variable_count", 0) > 0)
    )
    row["lingam_assumption_warning"] = warning
    row["directlingam_assumption_warning"] = warning

    return row


def parse_json_array(value: Any, field_name: str) -> np.ndarray:
    """Parse a graph-table JSON matrix field into a numpy array."""
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    arr = np.asarray(parsed, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise click.ClickException(f"{field_name} must be a square matrix, got shape {arr.shape}.")
    return arr


def parse_json_tensor(
    value: Any,
    field_name: str,
    *,
    ndim: int,
) -> np.ndarray:
    """Parse a numeric JSON tensor and validate its dimensionality."""
    if value is None or (
        isinstance(value, float) and np.isnan(value)
    ):
        raise click.ClickException(f"{field_name} is missing.")
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=float)
    if array.ndim != ndim:
        raise click.ClickException(
            f"{field_name} must be {ndim}-dimensional, got "
            f"shape {array.shape}."
        )
    return array


def graph_model_type(graph_row: Mapping[str, Any]) -> str:
    """Return the stored model type, defaulting older graph rows to DirectLiNGAM."""
    value = graph_row.get("model_type")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "directlingam"
    return str(value).lower()


def structural_residuals_for_graph(
    X: np.ndarray,
    raw_adjacency: np.ndarray,
    graph_row: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Calculate structural errors for DirectLiNGAM or VAR-LiNGAM."""
    model_type = graph_model_type(graph_row)
    if model_type == "directlingam":
        return X - X @ raw_adjacency.T, 0
    if model_type != "varlingam":
        raise click.ClickException(f"Unsupported graph model type: {model_type!r}")

    value = graph_row.get("adjacency_lagged_raw_json")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise click.ClickException(
            "VAR-LiNGAM graph row is missing adjacency_lagged_raw_json."
        )
    parsed = json.loads(value) if isinstance(value, str) else value
    lagged_adjacency = np.asarray(parsed, dtype=float)
    expected_tail = raw_adjacency.shape
    if lagged_adjacency.ndim != 3 or lagged_adjacency.shape[1:] != expected_tail:
        raise click.ClickException(
            "adjacency_lagged_raw_json must have shape "
            f"(lags, {expected_tail[0]}, {expected_tail[1]}), got "
            f"{lagged_adjacency.shape}."
        )

    n_lags = int(lagged_adjacency.shape[0])
    stored_lags = graph_row.get("var_lags")
    if stored_lags is not None and not (
        isinstance(stored_lags, float) and np.isnan(stored_lags)
    ):
        if int(stored_lags) != n_lags:
            raise click.ClickException(
                f"Stored var_lags={stored_lags} does not match {n_lags} "
                "lagged adjacency matrices."
            )
    if n_lags < 1 or len(X) <= n_lags:
        raise click.ClickException(
            f"VAR-LiNGAM needs at least {n_lags + 1} ordered observations."
        )

    current = X[n_lags:]
    fitted = current @ raw_adjacency.T
    for lag, adjacency in enumerate(lagged_adjacency, start=1):
        fitted += X[n_lags - lag : len(X) - lag] @ adjacency.T
    return current - fitted, n_lags


def reduced_form_var_innovations(
    X: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    """Refit an intercept-free reduced-form VAR and return its innovations.

    ``VARLiNGAM`` first fits this ordinary reduced-form VAR with
    ``trend="n"`` and then estimates the contemporaneous LiNGAM model from
    its residuals.  When pruning is enabled, the final structural adjacency
    matrices are re-estimated and no longer necessarily reproduce those
    original VAR residuals.  Temporal whiteness and lag-order adequacy must
    therefore be checked on the reduced-form innovations rather than on
    residuals reconstructed from the post-pruning structural matrices.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if len(values) <= n_lags:
        raise ValueError(
            f"VAR({n_lags}) requires more than {n_lags} observations"
        )

    current = values[n_lags:]
    lagged_design = np.concatenate(
        [
            values[n_lags - lag : len(values) - lag]
            for lag in range(1, n_lags + 1)
        ],
        axis=1,
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        lagged_design,
        current,
        rcond=None,
    )
    return current - lagged_design @ coefficients


def compute_statistics_for_graph(
    graph_row: Mapping[str, Any],
    pixel_key: PixelKey,
    window_group: pd.DataFrame,
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    min_prob: float,
    min_abs_effect: float,
    diagnostic_alpha: float,
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    whiteness_lags: int,
    probability_band: float,
    diagnostic_top_n: int,
    stability_threshold: float,
    stability_bootstrap_limit: int,
) -> dict[str, Any] | None:
    """Compute diagnostics/statistics for one saved graph row."""
    labels_value = graph_row["variable_names_json"]
    labels = json.loads(labels_value) if isinstance(labels_value, str) else list(labels_value)

    complete_g = window_group.dropna(subset=list(labels)).copy()
    X = complete_g[list(labels)].to_numpy()
    if len(X) == 0:
        return None

    raw_adjacency = parse_json_array(graph_row["adjacency_raw_json"], "adjacency_raw_json")
    probabilities = parse_json_array(graph_row["edge_probability_json"], "edge_probability_json")
    consensus_adjacency = parse_json_array(
        graph_row["adjacency_consensus_json"],
        "adjacency_consensus_json",
    )

    expected_shape = (len(labels), len(labels))
    for field_name, arr in {
        "adjacency_raw_json": raw_adjacency,
        "edge_probability_json": probabilities,
        "adjacency_consensus_json": consensus_adjacency,
    }.items():
        if arr.shape != expected_shape:
            raise click.ClickException(
                f"{field_name} for pixel {pixel_key} has shape {arr.shape}, expected {expected_shape}."
            )

    model_type = graph_model_type(graph_row)
    residuals, time_offset = structural_residuals_for_graph(
        X,
        raw_adjacency,
        graph_row,
    )
    effective_X = X[time_offset:]
    effective_metadata = complete_g.iloc[time_offset:].copy()
    if model_type == "varlingam":
        temporal_residuals = reduced_form_var_innovations(
            X,
            time_offset,
        )
        temporal_residual_basis = "refitted_reduced_form_var_innovations"
    else:
        temporal_residuals = residuals
        temporal_residual_basis = "structural_errors"

    row = make_diagnostics_row(
        pixel_key=pixel_key,
        complete_g=effective_metadata,
        X=effective_X,
        residuals=residuals,
        temporal_residuals=temporal_residuals,
        temporal_residual_basis=temporal_residual_basis,
        raw_adjacency=raw_adjacency,
        probabilities=probabilities,
        consensus_adjacency=consensus_adjacency,
        labels=labels,
        group_cols=group_cols,
        order_cols=order_cols,
        min_prob=min_prob,
        min_abs_effect=min_abs_effect,
        diagnostic_alpha=diagnostic_alpha,
        residual_corr_threshold=residual_corr_threshold,
        residual_crosslag_corr_threshold=(
            residual_crosslag_corr_threshold
        ),
        autocorr_threshold=autocorr_threshold,
        whiteness_lags=whiteness_lags,
        model_lags=time_offset,
        probability_band=probability_band,
        diagnostic_top_n=diagnostic_top_n,
    )
    row["model_type"] = model_type
    row["var_lags"] = int(time_offset)

    if model_type == "varlingam":
        lagged_raw = parse_json_tensor(
            graph_row.get("adjacency_lagged_raw_json"),
            "adjacency_lagged_raw_json",
            ndim=3,
        )
        lagged_probabilities = parse_json_tensor(
            graph_row.get("edge_probability_lagged_json"),
            "edge_probability_lagged_json",
            ndim=3,
        )
        lagged_consensus = parse_json_tensor(
            graph_row.get("adjacency_lagged_consensus_json"),
            "adjacency_lagged_consensus_json",
            ndim=3,
        )
        expected_lagged_shape = (
            time_offset,
            len(labels),
            len(labels),
        )
        for field_name, array in {
            "adjacency_lagged_raw_json": lagged_raw,
            "edge_probability_lagged_json": lagged_probabilities,
            "adjacency_lagged_consensus_json": lagged_consensus,
        }.items():
            if array.shape != expected_lagged_shape:
                raise click.ClickException(
                    f"{field_name} for pixel {pixel_key} has shape "
                    f"{array.shape}, expected {expected_lagged_shape}."
                )

        lagged_summary = lagged_bootstrap_probability_diagnostics(
            probabilities=lagged_probabilities,
            raw_adjacency=lagged_raw,
            consensus_adjacency=lagged_consensus,
            labels=labels,
            min_prob=min_prob,
            min_abs_effect=min_abs_effect,
            probability_band=probability_band,
            top_n=diagnostic_top_n,
        )
        row.update(lagged_summary)

        bootstrap_b0 = parse_json_tensor(
            graph_row.get("adjacency_bootstrap_json"),
            "adjacency_bootstrap_json",
            ndim=3,
        )
        bootstrap_lagged = parse_json_tensor(
            graph_row.get("adjacency_bootstrap_lagged_json"),
            "adjacency_bootstrap_lagged_json",
            ndim=4,
        )
        expected_b0_tail = (len(labels), len(labels))
        if bootstrap_b0.shape[1:] != expected_b0_tail:
            raise click.ClickException(
                "adjacency_bootstrap_json has shape "
                f"{bootstrap_b0.shape}, expected "
                f"(bootstrap, {len(labels)}, {len(labels)})."
            )
        if (
            bootstrap_lagged.shape[0] != bootstrap_b0.shape[0]
            or bootstrap_lagged.shape[1:] != expected_lagged_shape
        ):
            raise click.ClickException(
                "adjacency_bootstrap_lagged_json has shape "
                f"{bootstrap_lagged.shape}, expected "
                f"({len(bootstrap_b0)}, {time_offset}, "
                f"{len(labels)}, {len(labels)})."
            )
        row.update(
            var_stability_diagnostics(
                contemporaneous=raw_adjacency,
                lagged=lagged_raw,
                bootstrap_contemporaneous=bootstrap_b0,
                bootstrap_lagged=bootstrap_lagged,
                stability_threshold=stability_threshold,
                bootstrap_limit=stability_bootstrap_limit,
            )
        )

        row["contemporaneous_bootstrap_probability_entropy_mean"] = (
            row.get("bootstrap_probability_entropy_mean")
        )
        row["contemporaneous_bootstrap_edges_near_threshold"] = (
            row.get("bootstrap_edges_near_threshold")
        )
        contemporaneous_probabilities = probabilities[
            off_diagonal_mask(len(labels))
        ]
        combined_probabilities = np.concatenate(
            [
                contemporaneous_probabilities.ravel(),
                lagged_probabilities.ravel(),
            ]
        )
        combined_probabilities = combined_probabilities[
            np.isfinite(combined_probabilities)
        ]
        if len(combined_probabilities):
            clipped = np.clip(
                combined_probabilities,
                1e-12,
                1.0 - 1e-12,
            )
            combined_entropy = (
                -clipped * np.log2(clipped)
                - (1.0 - clipped) * np.log2(1.0 - clipped)
            )
            row["var_bootstrap_probability_entropy_mean"] = safe_float(
                np.mean(combined_entropy)
            )
        else:
            row["var_bootstrap_probability_entropy_mean"] = None

        if row.get("var_stable") is False:
            row["lingam_assumption_warning"] = True
            row["directlingam_assumption_warning"] = True

    stored_n_samples = graph_row.get("n_effective_samples")
    if stored_n_samples is None or (
        isinstance(stored_n_samples, float) and np.isnan(stored_n_samples)
    ):
        stored_n_samples = graph_row.get("n_samples")
    if stored_n_samples is not None and int(stored_n_samples) != row["n_samples"]:
        row["n_samples_mismatch_warning"] = True
        row["graph_table_n_samples"] = int(stored_n_samples)
    else:
        row["n_samples_mismatch_warning"] = False
        row["graph_table_n_samples"] = int(row["n_samples"])

    return row


def compute_statistics_task(args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Unpack a multiprocessing task tuple and compute one diagnostics row."""
    (
        graph_row,
        pixel_key,
        window_group,
        group_cols,
        order_cols,
        min_prob,
        min_abs_effect,
        diagnostic_alpha,
        residual_corr_threshold,
        residual_crosslag_corr_threshold,
        autocorr_threshold,
        whiteness_lags,
        probability_band,
        diagnostic_top_n,
        stability_threshold,
        stability_bootstrap_limit,
    ) = args

    return compute_statistics_for_graph(
        graph_row=graph_row,
        pixel_key=pixel_key,
        window_group=window_group,
        group_cols=group_cols,
        order_cols=order_cols,
        min_prob=min_prob,
        min_abs_effect=min_abs_effect,
        diagnostic_alpha=diagnostic_alpha,
        residual_corr_threshold=residual_corr_threshold,
        residual_crosslag_corr_threshold=(
            residual_crosslag_corr_threshold
        ),
        autocorr_threshold=autocorr_threshold,
        whiteness_lags=whiteness_lags,
        probability_band=probability_band,
        diagnostic_top_n=diagnostic_top_n,
        stability_threshold=stability_threshold,
        stability_bootstrap_limit=stability_bootstrap_limit,
    )


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted DuckDB identifier for simple table/column names."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise click.BadParameter(
            f"Invalid DuckDB identifier: {identifier!r}. Use letters, numbers, and underscores."
        )
    return f'"{identifier}"'


def write_dataframe_table(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, table_name: str) -> None:
    """Create or replace a DuckDB table from a pandas data frame."""
    quoted_table = quote_identifier(table_name)
    con.register("_write_df", df)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {quoted_table} AS SELECT * FROM _write_df")
    finally:
        con.unregister("_write_df")


def resolve_path(base_dir: Path, value: str | Path | None, default: Path) -> Path:
    """Resolve a possibly relative config/CLI path."""
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


def graph_config_value(
    config_data: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Read graph-discovery settings from a nested or top-level config key."""
    graph_config = config_data.get("graph_discovery") or {}
    if not isinstance(graph_config, Mapping):
        raise click.BadParameter("config['graph_discovery'] must be a mapping.")
    return graph_config.get(key, config_data.get(key, default))


def default_diagnostics_path(graphs_db_path: Path) -> Path:
    """Derive a non-colliding diagnostics path from the graph database name."""
    stem = graphs_db_path.stem
    if stem.endswith("_graphs"):
        stem = f"{stem[:-len('_graphs')]}_graph_diagnostics"
    else:
        stem = f"{stem}_diagnostics"
    return graphs_db_path.with_name(f"{stem}{graphs_db_path.suffix}")


def default_varlingam_graph_path(graphs_db_path: Path) -> Path:
    """Mirror graph discovery's non-colliding VAR-LiNGAM output name."""
    stem = graphs_db_path.stem
    if stem.endswith("_graphs"):
        stem = f"{stem[:-len('_graphs')]}_varlingam_graphs"
    else:
        stem = f"{stem}_varlingam"
    return graphs_db_path.with_name(f"{stem}{graphs_db_path.suffix}")


def write_diagnostics_to_duckdb(
    diagnostics_df: pd.DataFrame,
    diagnostics_db: Path,
    diagnostics_table: str,
    metadata: Mapping[str, Any],
    metadata_table: str = "graph_statistics_run_metadata",
) -> None:
    """Write diagnostics/statistics and run metadata to DuckDB tables."""
    diagnostics_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(diagnostics_db)
    try:
        write_dataframe_table(con, diagnostics_df, diagnostics_table)
        metadata_df = pd.DataFrame([dict(metadata)])
        write_dataframe_table(con, metadata_df, metadata_table)
    finally:
        con.close()


@click.command()
@click.option("-c", "--config-path", help="Path to the YAML config file with experiment parameters", required=True)
@click.option(
    "--input-db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Time-series DuckDB. Defaults to the graph-discovery input configured in YAML.",
)
@click.option(
    "--input-table",
    default=None,
    help="Time-series table. Defaults to the graph-discovery input configured in YAML.",
)
@click.option(
    "--graphs-db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="DuckDB file with graph-discovery output. Defaults to the output configured in YAML.",
)
@click.option(
    "--graphs-table",
    default="pixel_graphs",
    show_default=True,
    help="DuckDB table containing saved graph-discovery rows.",
)
@click.option(
    "--diagnostics-db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="DuckDB file for diagnostics/statistics. Defaults alongside the graph database.",
)
@click.option(
    "--diagnostics-table",
    default="pixel_graph_diagnostics",
    show_default=True,
    help="DuckDB table name for per-pixel LiNGAM diagnostics/statistics.",
)
@click.option("--window-size", default=0, show_default=True, type=int, help="Must match graph-discovery window size.")
@click.option("--min-edge-prob", default=0.7, show_default=True, type=float)
@click.option("--min-abs-effect", default=0.01, show_default=True, type=float)
@click.option("--diagnostic-alpha", default=0.05, show_default=True, type=float)
@click.option("--residual-corr-threshold", default=0.2, show_default=True, type=float)
@click.option(
    "--residual-crosslag-corr-threshold",
    default=0.3,
    show_default=True,
    type=float,
)
@click.option("--autocorr-threshold", default=0.3, show_default=True, type=float)
@click.option(
    "--whiteness-lags",
    default=12,
    show_default=True,
    type=click.IntRange(1, None),
    help="Maximum innovation lag for cross-correlation and whiteness tests.",
)
@click.option("--probability-band", default=0.1, show_default=True, type=float)
@click.option("--diagnostic-top-n", default=5, show_default=True, type=int)
@click.option(
    "--stability-threshold",
    default=1.0,
    show_default=True,
    type=click.FloatRange(0.0, None, min_open=True),
)
@click.option(
    "--stability-bootstrap-limit",
    default=0,
    show_default=True,
    type=click.IntRange(0, None),
    help="Maximum paired VAR bootstrap matrices for stability; 0 uses all.",
)
@click.option("-w", "--workers", default=1, show_default=True, type=int)
def graph_statistics(
    config_path: str,
    input_db: Path | None,
    input_table: str | None,
    graphs_db_path: Path | None,
    graphs_table: str,
    diagnostics_db_path: Path | None,
    diagnostics_table: str,
    window_size: int,
    min_edge_prob: float,
    min_abs_effect: float,
    diagnostic_alpha: float,
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    whiteness_lags: int,
    probability_band: float,
    diagnostic_top_n: int,
    stability_threshold: float,
    stability_bootstrap_limit: int,
    workers: int,
) -> None:
    """Compute diagnostics/statistics from saved pixel graph-discovery output."""
    if window_size < 0:
        raise click.BadParameter("window-size must be >= 0")
    if not 0.0 < diagnostic_alpha < 1.0:
        raise click.BadParameter("diagnostic-alpha must be between 0 and 1")
    for name, value in {
        "residual-corr-threshold": residual_corr_threshold,
        "residual-crosslag-corr-threshold": (
            residual_crosslag_corr_threshold
        ),
        "autocorr-threshold": autocorr_threshold,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise click.BadParameter(f"{name} must be between 0 and 1")

    row_col_cols = ["row", "col"]
    order_cols = ["year", "month"]
    config_path_obj = Path(config_path)

    with config_path_obj.open("r") as fd:
        config_data = yaml.safe_load(fd) or {}
    if not isinstance(config_data, Mapping):
        raise click.BadParameter("YAML config must contain a mapping at top level.")

    experiment_dir = config_path_obj.parent
    location_nickname = str(config_data["name"])
    input_db = resolve_path(
        experiment_dir,
        input_db
        or graph_config_value(config_data, "input_db")
        or graph_config_value(config_data, "timeseries_db"),
        experiment_dir / f"{location_nickname}_ard.duckdb",
    )
    input_table = str(
        input_table
        or graph_config_value(config_data, "input_table")
        or graph_config_value(config_data, "timeseries_table")
        or location_nickname
    )
    configured_model = str(
        graph_config_value(config_data, "model", "directlingam")
    ).lower()
    configured_var_graph_db = graph_config_value(config_data, "var_output_db")
    explicit_graphs_db_path = graphs_db_path
    configured_graph_db = (
        configured_var_graph_db
        or graph_config_value(config_data, "output_db")
        or graph_config_value(config_data, "graph_db")
    )
    graphs_db_path = resolve_path(
        experiment_dir,
        explicit_graphs_db_path or configured_graph_db,
        experiment_dir / f"{location_nickname}_graphs.duckdb",
    )
    if (
        configured_model == "varlingam"
        and explicit_graphs_db_path is None
        and configured_var_graph_db is None
    ):
        graphs_db_path = default_varlingam_graph_path(graphs_db_path)
    diagnostics_db_path = resolve_path(
        experiment_dir,
        diagnostics_db_path,
        default_diagnostics_path(graphs_db_path),
    )
    columns = config_data["columns"]

    con = duckdb.connect(input_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if input_table not in tables:
            raise click.BadParameter(
                f"{input_table} not found in {input_db}. Available: {sorted(tables)}"
            )
        df = con.execute(f"SELECT * FROM {quote_identifier(input_table)}").fetchdf()
    finally:
        con.close()

    missing_required = [col for col in row_col_cols + order_cols if col not in df.columns]
    if missing_required:
        raise click.BadParameter(f"Missing required columns: {missing_required}")

    con = duckdb.connect(graphs_db_path, read_only=True)
    try:
        graph_tables = set(con.sql("SHOW TABLES").df()["name"])
        if graphs_table not in graph_tables:
            raise click.BadParameter(
                f"{graphs_table} not found in {graphs_db_path}. Available: {sorted(graph_tables)}"
            )
        graph_df = con.execute(f"SELECT * FROM {quote_identifier(graphs_table)}").fetchdf()
    finally:
        con.close()

    required_graph_cols = [
        *row_col_cols,
        "n_samples",
        "variable_names_json",
        "adjacency_raw_json",
        "edge_probability_json",
        "adjacency_consensus_json",
    ]
    missing_graph_cols = [col for col in required_graph_cols if col not in graph_df.columns]
    if missing_graph_cols:
        raise click.BadParameter(f"Missing graph table columns: {missing_graph_cols}")

    model_types = {
        graph_model_type(row)
        for row in graph_df.to_dict(orient="records")
    }
    if len(model_types) != 1:
        raise click.ClickException(
            "Graph table mixes model types; diagnostics require one model type "
            f"per run. Found: {sorted(model_types)}"
        )
    graph_model = next(iter(model_types))
    if graph_model == "varlingam" and window_size != 0:
        raise click.BadParameter(
            "VAR-LiNGAM diagnostics require --window-size 0."
        )
    if graph_model == "varlingam":
        required_var_columns = {
            "var_lags",
            "adjacency_lagged_raw_json",
            "edge_probability_lagged_json",
            "adjacency_lagged_consensus_json",
            "adjacency_bootstrap_json",
            "adjacency_bootstrap_lagged_json",
        }
        missing_var_columns = sorted(
            required_var_columns - set(graph_df.columns)
        )
        if missing_var_columns:
            raise click.BadParameter(
                "VAR graph table is missing required diagnostic columns: "
                f"{missing_var_columns}"
            )
        maximum_fitted_lags = int(
            pd.to_numeric(
                graph_df["var_lags"],
                errors="coerce",
            ).max()
        )
        if whiteness_lags <= maximum_fitted_lags:
            raise click.BadParameter(
                "whiteness-lags must exceed every fitted VAR lag count; "
                f"maximum fitted lag count is {maximum_fitted_lags}."
            )
    apply_configured_shifts = graph_model == "directlingam"
    df, labels, label_lags = parse_columns(
        df,
        row_col_cols,
        order_cols,
        columns,
        apply_shifts=apply_configured_shifts,
    )
    df = df.dropna(subset=labels + row_col_cols + order_cols)

    groups = list(df.groupby(row_col_cols, sort=True))
    group_lookup = {
        pixel_key if isinstance(pixel_key, tuple) else (pixel_key,): group
        for pixel_key, group in groups
    }

    tasks = []
    for _, row in graph_df.iterrows():
        graph_row = row.to_dict()
        pixel_key = tuple(int(graph_row[col]) for col in row_col_cols)

        if window_size == 0:
            window_group = group_lookup.get(pixel_key)
        else:
            window_group = get_pixel_window_group(
                pixel_key=pixel_key,
                group_lookup=group_lookup,
                window_size=window_size,
            )

        if window_group is None:
            continue

        tasks.append(
            (
                graph_row,
                pixel_key,
                window_group,
                row_col_cols,
                order_cols,
                min_edge_prob,
                min_abs_effect,
                diagnostic_alpha,
                residual_corr_threshold,
                residual_crosslag_corr_threshold,
                autocorr_threshold,
                whiteness_lags,
                probability_band,
                diagnostic_top_n,
                stability_threshold,
                stability_bootstrap_limit,
            )
        )

    results = process_map(
        compute_statistics_task,
        tasks,
        max_workers=workers,
        chunksize=1,
        desc="Graph statistics",
    )
    diagnostics_rows = [result for result in results if result is not None]

    if not diagnostics_rows:
        raise click.ClickException("No graph rows could be matched to input pixel/window data.")

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path_obj),
        "input_db": str(input_db),
        "graph_output_db": str(graphs_db_path),
        "graphs_table": graphs_table,
        "diagnostics_db": str(diagnostics_db_path),
        "diagnostics_table": diagnostics_table,
        "input_table": input_table,
        "n_graph_rows": int(len(graph_df)),
        "n_diagnostics_rows": int(len(diagnostics_df)),
        "min_edge_prob": float(min_edge_prob),
        "min_abs_effect": float(min_abs_effect),
        "window_size": int(window_size),
        "diagnostic_alpha": float(diagnostic_alpha),
        "residual_corr_threshold": float(residual_corr_threshold),
        "residual_crosslag_corr_threshold": float(
            residual_crosslag_corr_threshold
        ),
        "autocorr_threshold": float(autocorr_threshold),
        "whiteness_lags": int(whiteness_lags),
        "probability_band": float(probability_band),
        "diagnostic_top_n": int(diagnostic_top_n),
        "stability_threshold": float(stability_threshold),
        "stability_bootstrap_limit": int(stability_bootstrap_limit),
        "model_type": graph_model,
        "configured_shifts_applied": bool(apply_configured_shifts),
        "variable_names_json": json.dumps(list(labels)),
        "label_lags_json": json.dumps({str(k): int(v) for k, v in label_lags.items()}),
    }
    write_diagnostics_to_duckdb(
        diagnostics_df=diagnostics_df,
        diagnostics_db=diagnostics_db_path,
        diagnostics_table=diagnostics_table,
        metadata=metadata,
    )
    click.echo(
        f"Wrote diagnostics: {diagnostics_db_path}::{diagnostics_table}"
    )
    click.echo(
        "Model: "
        f"{graph_model}; pixels/windows: {len(diagnostics_df)}; "
        f"whiteness lags: {whiteness_lags}"
    )


if __name__ == "__main__":
    graph_statistics()
