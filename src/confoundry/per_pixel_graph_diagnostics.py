"""Compute post-hoc statistics and DirectLiNGAM diagnostics for saved graphs.

This command reads graph-discovery output produced by ``graph_discovery.py`` and
reconstructs the same pixel/window data matrices from the original ARD DuckDB
input. It then computes compact diagnostics/statistics from the saved raw
adjacency, bootstrap probabilities, and consensus adjacency matrices.

No DirectLiNGAM models are refit in this script.
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
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Apply configured temporal shifts to columns."""
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

        shifted_df[label] = shifted_df.groupby(list(group_cols))[label].shift(lag)
        labels.append(label)
        label_lags[label] = lag

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
            bidirectional[0]["bidirectional_instability"] if bidirectional else None
        ),
        "bootstrap_bidirectional_top_pairs_json": json.dumps(bidirectional[:top_n]),
    }


def make_diagnostics_row(
    pixel_key: PixelKey,
    complete_g: pd.DataFrame,
    X: np.ndarray,
    residuals: np.ndarray,
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
    autocorr_threshold: float,
    probability_band: float,
    diagnostic_top_n: int,
) -> dict[str, Any]:
    """Build one compact sidecar diagnostics row for a fitted pixel/window."""
    serialized_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)
    row = dict(zip(group_cols, serialized_pixel_key, strict=False))
    row["n_samples"] = int(len(X))

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
            values=residuals,
            metadata=complete_g,
            labels=labels,
            group_cols=group_cols,
            order_cols=order_cols,
            threshold=autocorr_threshold,
            top_n=diagnostic_top_n,
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
    row["directlingam_assumption_warning"] = bool(
        (residual_corr is not None and residual_corr >= residual_corr_threshold)
        or (autocorr is not None and autocorr >= autocorr_threshold)
        or (row.get("near_constant_variable_count", 0) > 0)
    )

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
    autocorr_threshold: float,
    probability_band: float,
    diagnostic_top_n: int,
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

    residuals = X - X @ raw_adjacency.T

    row = make_diagnostics_row(
        pixel_key=pixel_key,
        complete_g=complete_g,
        X=X,
        residuals=residuals,
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
        autocorr_threshold=autocorr_threshold,
        probability_band=probability_band,
        diagnostic_top_n=diagnostic_top_n,
    )

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
        autocorr_threshold,
        probability_band,
        diagnostic_top_n,
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
        autocorr_threshold=autocorr_threshold,
        probability_band=probability_band,
        diagnostic_top_n=diagnostic_top_n,
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
    "--graphs-db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="DuckDB file with graph-discovery output. Defaults to <name>_graphs.duckdb.",
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
    help="DuckDB file for diagnostics/statistics. Defaults to <name>_graph_diagnostics.duckdb.",
)
@click.option(
    "--diagnostics-table",
    default="pixel_graph_diagnostics",
    show_default=True,
    help="DuckDB table name for per-pixel DirectLiNGAM diagnostics/statistics.",
)
@click.option("--window-size", default=0, show_default=True, type=int, help="Must match graph-discovery window size.")
@click.option("--min-edge-prob", default=0.7, show_default=True, type=float)
@click.option("--min-abs-effect", default=0.01, show_default=True, type=float)
@click.option("--diagnostic-alpha", default=0.05, show_default=True, type=float)
@click.option("--residual-corr-threshold", default=0.2, show_default=True, type=float)
@click.option("--autocorr-threshold", default=0.3, show_default=True, type=float)
@click.option("--probability-band", default=0.1, show_default=True, type=float)
@click.option("--diagnostic-top-n", default=5, show_default=True, type=int)
@click.option("-w", "--workers", default=1, show_default=True, type=int)
def graph_statistics(
    config_path: str,
    graphs_db_path: Path | None,
    graphs_table: str,
    diagnostics_db_path: Path | None,
    diagnostics_table: str,
    window_size: int,
    min_edge_prob: float,
    min_abs_effect: float,
    diagnostic_alpha: float,
    residual_corr_threshold: float,
    autocorr_threshold: float,
    probability_band: float,
    diagnostic_top_n: int,
    workers: int,
) -> None:
    """Compute diagnostics/statistics from saved pixel graph-discovery output."""
    if window_size < 0:
        raise click.BadParameter("window-size must be >= 0")

    row_col_cols = ["row", "col"]
    order_cols = ["year", "month"]
    config_path_obj = Path(config_path)

    with config_path_obj.open("r") as fd:
        config_data = yaml.safe_load(fd)

    experiment_dir = config_path_obj.parent
    location_nickname = config_data["name"]
    input_db = experiment_dir / f"{location_nickname}_ard.duckdb"
    if graphs_db_path is None:
        graphs_db_path = experiment_dir / f"{location_nickname}_graphs.duckdb"
    if diagnostics_db_path is None:
        diagnostics_db_path = experiment_dir / f"{location_nickname}_graph_diagnostics.duckdb"
    input_table = location_nickname
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

    df, labels, label_lags = parse_columns(df, row_col_cols, order_cols, columns)
    df = df.dropna(subset=labels + row_col_cols + order_cols)

    groups = list(df.groupby(row_col_cols, sort=True))
    group_lookup = {
        pixel_key if isinstance(pixel_key, tuple) else (pixel_key,): group
        for pixel_key, group in groups
    }

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
                autocorr_threshold,
                probability_band,
                diagnostic_top_n,
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
        "autocorr_threshold": float(autocorr_threshold),
        "probability_band": float(probability_band),
        "diagnostic_top_n": int(diagnostic_top_n),
        "variable_names_json": json.dumps(list(labels)),
        "label_lags_json": json.dumps({str(k): int(v) for k, v in label_lags.items()}),
    }
    write_diagnostics_to_duckdb(
        diagnostics_df=diagnostics_df,
        diagnostics_db=diagnostics_db_path,
        diagnostics_table=diagnostics_table,
        metadata=metadata,
    )


if __name__ == "__main__":
    graph_statistics()
