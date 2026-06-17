#!/usr/bin/env python3
"""Analyze per-pixel DirectLiNGAM effects using saved bootstrap adjacencies.

This is a DoWhy-free counterpart to ``per_pixel_analysis.py`` for the
Confoundry pipeline.  It reads the shifted ARD time-series data and the
``pixel_graphs`` table produced by ``per_pixel_graph_discovery.py``.  Unlike an
analysis based only on ``edge_probability_json``, this script expects the graph
output to contain the full list of bootstrapped DirectLiNGAM adjacency matrices
(e.g. ``adjacency_bootstrap_json``).  Those matrices are used to propagate
uncertainty into direct, total, and robust quantile-scaled effects.

For each source X and target Y, the main comparable effect is

    scaled_total_effect = TE[X -> Y] * (Q_hi(X) - Q_lo(X)) /
                          (Q_hi(Y) - Q_lo(Y))

where TE is computed from the DirectLiNGAM adjacency matrix as
``(I - B)^(-1) - I``.  The script reports point estimates from a selected point
matrix (``consensus`` by default) and bootstrap summaries from all saved
bootstrap matrices.

It also writes a dominance table and categorical dominance map showing which
source has the largest absolute scaled total effect on the configured target at each
pixel, including bootstrap support for that dominance decision.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import json
import os
import re

import click
import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

try:
    # Reuse the same column-shift and DuckDB helper conventions as graph discovery.
    from confoundry.per_pixel_graph_discovery import (
        parse_columns,
        quote_identifier,
        write_dataframe_table,
    )
except ModuleNotFoundError:  # pragma: no cover - useful when run from src/confoundry directly
    from per_pixel_graph_discovery import (  # type: ignore
        parse_columns,
        quote_identifier,
        write_dataframe_table,
    )

PixelKey = tuple[Any, ...]
_VALID_POINT_MATRIX_CHOICES = {"raw", "consensus", "bootstrap_mean"}
_SEASONAL_NAMES = {"month_sin", "month_cos"}
_BOOTSTRAP_MATRIX_FIELDS = (
    "adjacency_bootstrap_json",
    "bootstrap_adjacency_json",
    "bootstrap_adjacencies_json",
    "adjacency_matrices_json",
    "bootstrap_adjacency_matrices_json",
)


@dataclass(frozen=True)
class Config:
    experiment_dir: Path
    location_name: str
    columns: list[dict[str, Any]]
    timeseries_db: Path
    graph_db: Path
    effects_db: Path
    dominance_db: Path
    effects_csv: Path
    dominance_csv: Path
    plot_dir: Path
    timeseries_table: str
    graph_table: str
    effects_table: str
    dominance_table: str
    row_col_cols: list[str]
    order_cols: list[str]
    target_col: str
    source_cols: list[str] | None
    low_quantile: float
    high_quantile: float
    min_samples: int
    point_matrix: str
    include_seasonality_as_source: bool
    min_path_abs_effect: float
    path_top_n: int
    max_paths_per_pair: int


@dataclass(frozen=True)
class PixelBundle:
    key: PixelKey
    coords: dict[str, Any]
    time_series: pd.DataFrame
    graph_row: dict[str, Any]


def _normalize_key(key: Any) -> PixelKey:
    return key if isinstance(key, tuple) else (key,)


def _as_path(base_dir: Path, value: str | Path | None, default_name: str | Path) -> Path:
    path = Path(value) if value is not None else Path(default_name)
    return path.expanduser() if path.is_absolute() else base_dir / path


def _maybe_load_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r") as fd:
        config_data = yaml.safe_load(fd) or {}
    if not isinstance(config_data, dict):
        raise click.BadParameter("YAML config must contain a mapping at top level.")
    return config_data


def _get_analysis_value(config_data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    analysis_config = config_data.get("analysis") or {}
    if not isinstance(analysis_config, Mapping):
        raise click.BadParameter("config['analysis'] must be a mapping when present.")
    return analysis_config.get(key, config_data.get(key, default))


def _parse_csv_option(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def load_config(
    config_path: str | Path,
    target_override: str | None = None,
    outcome_override: str | None = None,
    sources_override: str | None = None,
    point_matrix_override: str | None = None,
    effects_db_override: Path | None = None,
    effects_csv_override: Path | None = None,
    dominance_db_override: Path | None = None,
    dominance_csv_override: Path | None = None,
    plot_dir_override: Path | None = None,
) -> Config:
    config_path = Path(config_path).expanduser().resolve()
    config_data = _read_yaml(config_path)
    experiment_dir = config_path.parent

    try:
        location_name = str(config_data["name"])
        columns_raw = config_data["columns"]
    except KeyError as exc:
        raise click.BadParameter(f"Missing required config key: {exc.args[0]}") from exc

    if not isinstance(columns_raw, list):
        raise click.BadParameter("config['columns'] must be a list of column specs.")

    columns = [dict(spec) for spec in columns_raw]
    configured_columns = [str(spec["name"]) for spec in columns]

    target_col = (
        target_override
        or outcome_override
        or _get_analysis_value(config_data, "target")
        or _get_analysis_value(config_data, "outcome")
        or config_data.get("reference_var")
        or ("ndvi" if "ndvi" in configured_columns else None)
    )
    if target_col is None:
        raise click.BadParameter(
            "Could not infer target variable. Set analysis.target, analysis.outcome, "
            "reference_var, or pass --target/--outcome."
        )

    source_cols = _parse_csv_option(sources_override)
    if source_cols is None:
        source_cols = _parse_csv_option(_get_analysis_value(config_data, "sources"))

    point_matrix = str(
        point_matrix_override
        or _get_analysis_value(config_data, "directlingam_point_matrix", "consensus")
    )
    if point_matrix not in _VALID_POINT_MATRIX_CHOICES:
        raise click.BadParameter(
            f"point matrix must be one of {sorted(_VALID_POINT_MATRIX_CHOICES)}, got {point_matrix!r}"
        )

    return Config(
        experiment_dir=experiment_dir,
        location_name=location_name,
        columns=columns,
        timeseries_db=_as_path(
            experiment_dir,
            _get_analysis_value(config_data, "timeseries_db"),
            f"{location_name}_ard.duckdb",
        ),
        graph_db=_as_path(
            experiment_dir,
            _get_analysis_value(config_data, "graph_db"),
            f"{location_name}_graphs.duckdb",
        ),
        effects_db=_as_path(
            experiment_dir,
            effects_db_override or _get_analysis_value(config_data, "directlingam_effects_db"),
            f"{location_name}_directlingam_effects.duckdb",
        ),
        dominance_db=_as_path(
            experiment_dir,
            dominance_db_override or _get_analysis_value(config_data, "directlingam_dominance_db"),
            f"{location_name}_directlingam_dominance.duckdb",
        ),
        effects_csv=_as_path(
            experiment_dir,
            effects_csv_override or _get_analysis_value(config_data, "directlingam_effects_csv"),
            f"{location_name}_directlingam_effects.csv",
        ),
        dominance_csv=_as_path(
            experiment_dir,
            dominance_csv_override or _get_analysis_value(config_data, "directlingam_dominance_csv"),
            f"{location_name}_directlingam_dominance.csv",
        ),
        plot_dir=_as_path(
            experiment_dir,
            plot_dir_override or _get_analysis_value(config_data, "directlingam_plot_dir"),
            f"{location_name}_directlingam_plots",
        ),
        timeseries_table=str(_get_analysis_value(config_data, "timeseries_table", location_name)),
        graph_table=str(_get_analysis_value(config_data, "graph_table", "pixel_graphs")),
        effects_table=str(
            _get_analysis_value(config_data, "directlingam_effects_table", "pixel_directlingam_effects")
        ),
        dominance_table=str(
            _get_analysis_value(config_data, "directlingam_dominance_table", "pixel_directlingam_dominance")
        ),
        row_col_cols=list(_get_analysis_value(config_data, "row_col_cols", ["row", "col"])),
        order_cols=list(_get_analysis_value(config_data, "order_cols", ["year", "month"])),
        target_col=str(target_col),
        source_cols=source_cols,
        low_quantile=float(_get_analysis_value(config_data, "low_quantile", 0.10)),
        high_quantile=float(_get_analysis_value(config_data, "high_quantile", 0.90)),
        min_samples=int(_get_analysis_value(config_data, "analysis_min_samples", 5)),
        point_matrix=point_matrix,
        include_seasonality_as_source=bool(
            _get_analysis_value(config_data, "include_seasonality_as_source", False)
        ),
        min_path_abs_effect=float(_get_analysis_value(config_data, "min_path_abs_effect", 0.0)),
        path_top_n=int(_get_analysis_value(config_data, "path_top_n", 5)),
        max_paths_per_pair=int(_get_analysis_value(config_data, "max_paths_per_pair", 5000)),
    )


def _read_table(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {quote_identifier(table_name)}").fetchdf()


def _ensure_table_exists(con: duckdb.DuckDBPyConnection, table_name: str, db_path: Path) -> None:
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if table_name not in tables:
        raise click.BadParameter(
            f"{table_name!r} not found in {db_path}. Available tables: {sorted(tables)}"
        )


def load_shifted_timeseries_and_graphs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ts_con = duckdb.connect(str(cfg.timeseries_db), read_only=True)
    graph_con = duckdb.connect(str(cfg.graph_db), read_only=True)
    try:
        _ensure_table_exists(ts_con, cfg.timeseries_table, cfg.timeseries_db)
        _ensure_table_exists(graph_con, cfg.graph_table, cfg.graph_db)
        ts_df = _read_table(ts_con, cfg.timeseries_table)
        graph_df = _read_table(graph_con, cfg.graph_table)
    finally:
        ts_con.close()
        graph_con.close()

    missing_ts = [c for c in cfg.row_col_cols + cfg.order_cols if c not in ts_df.columns]
    if missing_ts:
        raise click.BadParameter(f"Missing required columns in time series table: {missing_ts}")

    required_graph_cols = [
        *cfg.row_col_cols,
        "variable_names_json",
        "adjacency_raw_json",
        "edge_probability_json",
        "adjacency_consensus_json",
    ]
    missing_graph = [c for c in required_graph_cols if c not in graph_df.columns]
    if missing_graph:
        raise click.BadParameter(f"Missing required columns in graph table: {missing_graph}")

    if not any(field in graph_df.columns for field in _BOOTSTRAP_MATRIX_FIELDS):
        raise click.BadParameter(
            "Graph table does not contain saved bootstrap adjacency matrices. "
            "Add a JSON column such as 'adjacency_bootstrap_json' to graph discovery output. "
            "The existing edge_probability_json is not enough to estimate effect uncertainty."
        )

    shifted_ts_df, labels, _ = parse_columns(
        ts_df,
        group_cols=cfg.row_col_cols,
        order_cols=cfg.order_cols,
        column_specs=cfg.columns,
    )
    shifted_ts_df = shifted_ts_df.dropna(
        subset=list(dict.fromkeys(labels + cfg.row_col_cols + cfg.order_cols))
    )

    dup_graphs = graph_df.duplicated(subset=cfg.row_col_cols, keep=False)
    if dup_graphs.any():
        bad_keys = (
            graph_df.loc[dup_graphs, cfg.row_col_cols]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise click.BadParameter(
            f"Graph table contains duplicate pixel keys for {cfg.row_col_cols}: {bad_keys}"
        )

    return shifted_ts_df, graph_df, labels


def decode_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    for raw_key, parsed_key in [
        ("variable_names_json", "variable_names"),
        ("variable_index_json", "variable_index"),
        ("causal_order_json", "causal_order"),
        ("adjacency_raw_json", "adjacency_raw"),
        ("edge_probability_json", "edge_probability"),
        ("adjacency_consensus_json", "adjacency_consensus"),
    ]:
        if raw_key in parsed:
            parsed[parsed_key] = _maybe_load_json(parsed[raw_key])

    for field in _BOOTSTRAP_MATRIX_FIELDS:
        if field in parsed and parsed[field] is not None and not (isinstance(parsed[field], float) and pd.isna(parsed[field])):
            parsed["adjacency_bootstrap"] = _maybe_load_json(parsed[field])
            parsed["adjacency_bootstrap_field"] = field
            break
    else:
        parsed["adjacency_bootstrap"] = None
        parsed["adjacency_bootstrap_field"] = None

    return parsed


def iter_pixel_groups(cfg: Config, timeseries_df: pd.DataFrame, graph_df: pd.DataFrame) -> Iterator[PixelBundle]:
    graph_keys = graph_df[cfg.row_col_cols].drop_duplicates()
    ts_df = timeseries_df.merge(graph_keys, on=cfg.row_col_cols, how="inner")
    ts_df = ts_df.sort_values(cfg.row_col_cols + cfg.order_cols).reset_index(drop=True)
    graph_df = graph_df.set_index(cfg.row_col_cols, drop=False)

    for key, group in ts_df.groupby(cfg.row_col_cols, sort=True):
        key = _normalize_key(key)
        graph_row = decode_graph_row(graph_df.loc[key].to_dict())
        yield PixelBundle(
            key=key,
            coords=dict(zip(cfg.row_col_cols, key, strict=False)),
            time_series=group.reset_index(drop=True),
            graph_row=graph_row,
        )


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _finite_quantile(values: Sequence[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _quantile_contrast(series: pd.Series, low_q: float, high_q: float) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    q_low = _finite_quantile(values, low_q)
    q_high = _finite_quantile(values, high_q)
    return {
        "q_low": q_low,
        "q_high": q_high,
        "delta": q_high - q_low if np.isfinite(q_low) and np.isfinite(q_high) else float("nan"),
    }


def _point_matrix_from_row(graph_row: Mapping[str, Any], point_matrix: str) -> np.ndarray:
    if point_matrix == "raw":
        arr = np.asarray(graph_row["adjacency_raw"], dtype=float)
    elif point_matrix == "consensus":
        arr = np.asarray(graph_row["adjacency_consensus"], dtype=float)
    elif point_matrix == "bootstrap_mean":
        arr = np.nanmean(_bootstrap_matrices_from_row(graph_row), axis=0)
    else:
        raise ValueError(f"Unknown point matrix kind: {point_matrix}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"point adjacency matrix must be square, got shape {arr.shape}")
    return arr


def _bootstrap_matrices_from_row(graph_row: Mapping[str, Any]) -> np.ndarray:
    value = graph_row.get("adjacency_bootstrap")
    if value is None:
        raise ValueError(
            "Missing adjacency_bootstrap matrices. Save boot.adjacency_matrices_ in graph discovery."
        )
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"bootstrap adjacency matrices must have shape (n_boot, d, d), got {arr.shape}")
    return arr


def _probability_matrix_from_graph_row(graph_row: Mapping[str, Any]) -> np.ndarray | None:
    value = graph_row.get("edge_probability")
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def _total_effect_matrix(B: np.ndarray) -> np.ndarray:
    eye = np.eye(B.shape[0], dtype=float)
    return np.linalg.inv(eye - B) - eye


def _bootstrap_total_effect_matrices(mats: np.ndarray) -> tuple[np.ndarray, int]:
    totals: list[np.ndarray] = []
    failed = 0
    for B in mats:
        try:
            total = _total_effect_matrix(np.asarray(B, dtype=float))
            if np.all(np.isfinite(total)):
                totals.append(total)
            else:
                failed += 1
        except Exception:
            failed += 1
    if not totals:
        return np.empty((0, mats.shape[1], mats.shape[2]), dtype=float), failed
    return np.stack(totals, axis=0), failed


def _summary(values: Sequence[float] | np.ndarray, ci: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "boot_mean": np.nan,
            "boot_median": np.nan,
            "boot_sd": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_ci_width": np.nan,
            "boot_prob_gt_zero": np.nan,
            "boot_prob_lt_zero": np.nan,
            "boot_prob_excludes_zero": False,
            "n_bootstrap_successful": 0,
        }
    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(arr, [alpha, 1.0 - alpha])
    return {
        "boot_mean": float(np.mean(arr)),
        "boot_median": float(np.median(arr)),
        "boot_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "boot_ci_low": float(ci_low),
        "boot_ci_high": float(ci_high),
        "boot_ci_width": float(ci_high - ci_low),
        "boot_prob_gt_zero": float(np.mean(arr > 0.0)),
        "boot_prob_lt_zero": float(np.mean(arr < 0.0)),
        "boot_prob_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
        "n_bootstrap_successful": int(len(arr)),
    }


def _default_sources(labels: Sequence[str], target_col: str, include_seasonality: bool) -> list[str]:
    out = []
    for label in labels:
        if label == target_col:
            continue
        if not include_seasonality and label in _SEASONAL_NAMES:
            continue
        out.append(label)
    return out


def _graph_from_adjacency(B: np.ndarray, labels: Sequence[str], min_abs_effect: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    for child_idx, child_name in enumerate(labels):
        for parent_idx, parent_name in enumerate(labels):
            if child_idx == parent_idx:
                continue
            coef = float(B[child_idx, parent_idx])
            if np.isfinite(coef) and abs(coef) > min_abs_effect:
                graph.add_edge(parent_name, child_name, weight=coef)
    return graph


def _path_product(B: np.ndarray, labels: Sequence[str], path: Sequence[str]) -> float:
    idx = {name: i for i, name in enumerate(labels)}
    coef = 1.0
    for parent, child in zip(path[:-1], path[1:], strict=False):
        coef *= float(B[idx[child], idx[parent]])
    return float(coef)


def _top_paths_from_point_matrix(
    B: np.ndarray,
    labels: Sequence[str],
    source: str,
    outcome: str,
    delta_source: float,
    delta_outcome: float,
    min_path_abs_effect: float,
    top_n: int,
    max_paths: int,
) -> list[dict[str, Any]]:
    graph = _graph_from_adjacency(B, labels, min_abs_effect=min_path_abs_effect)
    if source not in graph.nodes or outcome not in graph.nodes:
        return []
    rows: list[dict[str, Any]] = []
    try:
        for path_idx, path in enumerate(nx.all_simple_paths(graph, source=source, target=outcome)):
            if path_idx >= max_paths:
                break
            coef = _path_product(B, labels, path)
            scaled = coef * delta_source / delta_outcome if np.isfinite(delta_outcome) and delta_outcome != 0 else np.nan
            rows.append(
                {
                    "path": " -> ".join(path),
                    "nodes": list(path),
                    "length": len(path) - 1,
                    "path_effect": _safe_float(coef),
                    "scaled_path_effect": _safe_float(scaled),
                    "abs_scaled_path_effect": _safe_float(abs(scaled)),
                }
            )
    except nx.NetworkXNoPath:
        return []
    rows.sort(key=lambda r: r["abs_scaled_path_effect"] if np.isfinite(r["abs_scaled_path_effect"]) else -np.inf, reverse=True)
    return rows[:top_n]


def _bootstrap_path_summaries_for_fixed_paths(
    boot_mats: np.ndarray,
    labels: Sequence[str],
    paths: Sequence[Mapping[str, Any]],
    delta_source: float,
    delta_outcome: float,
    ci: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path_row in paths:
        nodes = [str(x) for x in path_row.get("nodes", [])]
        if len(nodes) < 2:
            continue
        raw_values: list[float] = []
        scaled_values: list[float] = []
        for B in boot_mats:
            value = _path_product(B, labels, nodes)
            raw_values.append(value)
            scaled = value * delta_source / delta_outcome if np.isfinite(delta_outcome) and delta_outcome != 0 else np.nan
            scaled_values.append(scaled)
        raw_summary = _summary(raw_values, ci=ci)
        scaled_summary = _summary(scaled_values, ci=ci)
        out.append(
            {
                "path": path_row["path"],
                "length": path_row["length"],
                "point_path_effect": path_row["path_effect"],
                "point_scaled_path_effect": path_row["scaled_path_effect"],
                "path_effect_boot_mean": raw_summary["boot_mean"],
                "path_effect_boot_sd": raw_summary["boot_sd"],
                "path_effect_boot_ci_low": raw_summary["boot_ci_low"],
                "path_effect_boot_ci_high": raw_summary["boot_ci_high"],
                "scaled_path_effect_boot_mean": scaled_summary["boot_mean"],
                "scaled_path_effect_boot_sd": scaled_summary["boot_sd"],
                "scaled_path_effect_boot_ci_low": scaled_summary["boot_ci_low"],
                "scaled_path_effect_boot_ci_high": scaled_summary["boot_ci_high"],
            }
        )
    return out


def analyze_pixel(
    bundle: PixelBundle,
    target_col: str,
    source_cols: list[str] | None,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    point_matrix: str,
    include_seasonality_as_source: bool,
    min_path_abs_effect: float,
    path_top_n: int,
    max_paths_per_pair: int,
    ci: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [str(x) for x in bundle.graph_row["variable_names"]]
    # Keep the historical output column name "outcome" for compatibility,
    # but treat it as the user-configured target variable throughout.
    outcome_col = target_col
    base_error = {**bundle.coords, "target": target_col, "outcome": target_col, "error": None}

    if target_col not in labels:
        err = f"target {target_col!r} not in graph labels"
        return ([{**base_error, "source": None, "error": err}], {**base_error, "error": err})

    selected_sources = source_cols or _default_sources(labels, outcome_col, include_seasonality_as_source)
    bad_sources = [src for src in selected_sources if src not in labels]
    if bad_sources:
        err = f"sources not in graph labels: {bad_sources}"
        return ([{**base_error, "source": ",".join(bad_sources), "error": err}], {**base_error, "error": err})

    data = bundle.time_series.dropna(subset=list(dict.fromkeys(labels))).reset_index(drop=True)
    if len(data) < min_samples:
        err = f"too few samples: {len(data)} < {min_samples}"
        return ([{**base_error, "source": None, "error": err}], {**base_error, "error": err})

    try:
        point_B = _point_matrix_from_row(bundle.graph_row, point_matrix=point_matrix)
        boot_B = _bootstrap_matrices_from_row(bundle.graph_row)
        probs = _probability_matrix_from_graph_row(bundle.graph_row)
        if point_B.shape != (len(labels), len(labels)):
            raise ValueError(f"point adjacency shape {point_B.shape} does not match {len(labels)} labels")
        if boot_B.shape[1:] != (len(labels), len(labels)):
            raise ValueError(f"bootstrap adjacency shape {boot_B.shape} does not match {len(labels)} labels")
        point_total = _total_effect_matrix(point_B)
        boot_total, n_total_failed = _bootstrap_total_effect_matrices(boot_B)
        if len(boot_total) == 0:
            raise ValueError("No bootstrap adjacency matrix produced a finite total-effect matrix")
    except Exception as exc:
        err = repr(exc)
        return ([{**base_error, "source": None, "error": err}], {**base_error, "error": err})

    index = {name: idx for idx, name in enumerate(labels)}
    outcome_idx = index[outcome_col]
    outcome_q = _quantile_contrast(data[outcome_col], low_quantile, high_quantile)
    delta_outcome = outcome_q["delta"]

    n_edges_point = int(np.sum(np.isfinite(point_B) & (np.abs(point_B) > 0.0)) - np.sum(np.abs(np.diag(point_B)) > 0.0))
    n_bootstrap_total = int(len(boot_B))
    n_bootstrap_effect_successful = int(len(boot_total))

    effect_rows: list[dict[str, Any]] = []
    dominance_scores_point: dict[str, float] = {}
    dominance_scores_boot: dict[str, np.ndarray] = {}

    for source_col in selected_sources:
        source_idx = index[source_col]
        source_q = _quantile_contrast(data[source_col], low_quantile, high_quantile)
        delta_source = source_q["delta"]

        direct_point = float(point_B[outcome_idx, source_idx])
        total_point = float(point_total[outcome_idx, source_idx])
        scaled_direct_point = (
            direct_point * delta_source / delta_outcome
            if np.isfinite(delta_outcome) and delta_outcome != 0 else np.nan
        )
        scaled_total_point = (
            total_point * delta_source / delta_outcome
            if np.isfinite(delta_outcome) and delta_outcome != 0 else np.nan
        )

        direct_boot = boot_B[:, outcome_idx, source_idx]
        total_boot = boot_total[:, outcome_idx, source_idx]
        scaled_direct_boot = (
            direct_boot * delta_source / delta_outcome
            if np.isfinite(delta_outcome) and delta_outcome != 0 else np.full_like(direct_boot, np.nan, dtype=float)
        )
        scaled_total_boot = (
            total_boot * delta_source / delta_outcome
            if np.isfinite(delta_outcome) and delta_outcome != 0 else np.full_like(total_boot, np.nan, dtype=float)
        )
        abs_scaled_total_boot = np.abs(scaled_total_boot)

        direct_summary = _summary(direct_boot, ci=ci)
        total_summary = _summary(total_boot, ci=ci)
        scaled_direct_summary = _summary(scaled_direct_boot, ci=ci)
        scaled_total_summary = _summary(scaled_total_boot, ci=ci)
        abs_scaled_total_summary = _summary(abs_scaled_total_boot, ci=ci)

        direct_prob = float(probs[outcome_idx, source_idx]) if probs is not None else np.nan

        top_paths = _top_paths_from_point_matrix(
            B=point_B,
            labels=labels,
            source=source_col,
            outcome=outcome_col,
            delta_source=delta_source,
            delta_outcome=delta_outcome,
            min_path_abs_effect=min_path_abs_effect,
            top_n=path_top_n,
            max_paths=max_paths_per_pair,
        )
        boot_path_summaries = _bootstrap_path_summaries_for_fixed_paths(
            boot_mats=boot_B,
            labels=labels,
            paths=top_paths,
            delta_source=delta_source,
            delta_outcome=delta_outcome,
            ci=ci,
        )

        dominance_scores_point[source_col] = abs(scaled_total_point) if np.isfinite(scaled_total_point) else np.nan
        dominance_scores_boot[source_col] = abs_scaled_total_boot

        effect_rows.append(
            {
                **bundle.coords,
                "source": source_col,
                "target": target_col,
                "outcome": outcome_col,
                "point_matrix": point_matrix,
                "n_samples": int(len(data)),
                "n_edges_point": n_edges_point,
                "source_q_low": _safe_float(source_q["q_low"]),
                "source_q_high": _safe_float(source_q["q_high"]),
                "source_delta_qhi_qlo": _safe_float(delta_source),
                "target_q_low": _safe_float(outcome_q["q_low"]),
                "target_q_high": _safe_float(outcome_q["q_high"]),
                "target_delta_qhi_qlo": _safe_float(delta_outcome),
                # Backward-compatible aliases for older plotting/tests.
                "outcome_q_low": _safe_float(outcome_q["q_low"]),
                "outcome_q_high": _safe_float(outcome_q["q_high"]),
                "outcome_delta_qhi_qlo": _safe_float(delta_outcome),
                "direct_effect": _safe_float(direct_point),
                "total_effect": _safe_float(total_point),
                "scaled_direct_effect": _safe_float(scaled_direct_point),
                "scaled_total_effect": _safe_float(scaled_total_point),
                "abs_scaled_total_effect": _safe_float(abs(scaled_total_point)),
                "direct_edge_probability": _safe_float(direct_prob),
                "direct_effect_boot_mean": direct_summary["boot_mean"],
                "direct_effect_boot_sd": direct_summary["boot_sd"],
                "direct_effect_boot_ci_low": direct_summary["boot_ci_low"],
                "direct_effect_boot_ci_high": direct_summary["boot_ci_high"],
                "total_effect_boot_mean": total_summary["boot_mean"],
                "total_effect_boot_sd": total_summary["boot_sd"],
                "total_effect_boot_ci_low": total_summary["boot_ci_low"],
                "total_effect_boot_ci_high": total_summary["boot_ci_high"],
                "scaled_direct_effect_boot_mean": scaled_direct_summary["boot_mean"],
                "scaled_direct_effect_boot_sd": scaled_direct_summary["boot_sd"],
                "scaled_direct_effect_boot_ci_low": scaled_direct_summary["boot_ci_low"],
                "scaled_direct_effect_boot_ci_high": scaled_direct_summary["boot_ci_high"],
                "scaled_total_effect_boot_mean": scaled_total_summary["boot_mean"],
                "scaled_total_effect_boot_median": scaled_total_summary["boot_median"],
                "scaled_total_effect_boot_sd": scaled_total_summary["boot_sd"],
                "scaled_total_effect_boot_ci_low": scaled_total_summary["boot_ci_low"],
                "scaled_total_effect_boot_ci_high": scaled_total_summary["boot_ci_high"],
                "scaled_total_effect_boot_ci_width": scaled_total_summary["boot_ci_width"],
                "scaled_total_effect_boot_prob_gt_zero": scaled_total_summary["boot_prob_gt_zero"],
                "scaled_total_effect_boot_prob_lt_zero": scaled_total_summary["boot_prob_lt_zero"],
                "scaled_total_effect_boot_ci_excludes_zero": scaled_total_summary["boot_prob_excludes_zero"],
                "abs_scaled_total_effect_boot_mean": abs_scaled_total_summary["boot_mean"],
                "abs_scaled_total_effect_boot_sd": abs_scaled_total_summary["boot_sd"],
                "abs_scaled_total_effect_boot_ci_low": abs_scaled_total_summary["boot_ci_low"],
                "abs_scaled_total_effect_boot_ci_high": abs_scaled_total_summary["boot_ci_high"],
                "n_bootstrap_total": n_bootstrap_total,
                "n_bootstrap_effect_successful": n_bootstrap_effect_successful,
                "n_bootstrap_total_effect_failed": int(n_total_failed),
                "top_paths_json": json.dumps(top_paths),
                "top_paths_bootstrap_json": json.dumps(boot_path_summaries),
                "error": None,
            }
        )

    # Dominant driver: variable with largest absolute scaled total effect to the target.
    point_items = {src: val for src, val in dominance_scores_point.items() if np.isfinite(val)}
    if point_items:
        dominant_point_source = max(point_items, key=point_items.get)
        dominant_point_abs_effect = float(point_items[dominant_point_source])
    else:
        dominant_point_source = None
        dominant_point_abs_effect = np.nan

    boot_source_names = list(dominance_scores_boot.keys())
    boot_matrix = np.column_stack([dominance_scores_boot[src] for src in boot_source_names]) if boot_source_names else np.empty((0, 0))
    valid_boot_rows = np.all(np.isfinite(boot_matrix), axis=1) if boot_matrix.size else np.array([], dtype=bool)
    if boot_matrix.size and np.any(valid_boot_rows):
        valid_scores = boot_matrix[valid_boot_rows]
        winners_idx = np.argmax(valid_scores, axis=1)
        winner_names = np.asarray(boot_source_names, dtype=object)[winners_idx]
        winner_counts = pd.Series(winner_names).value_counts().to_dict()
        winner_probs = {str(src): int(count) / int(np.sum(valid_boot_rows)) for src, count in winner_counts.items()}
        dominant_boot_source = max(winner_probs, key=winner_probs.get)
        dominant_boot_probability = float(winner_probs[dominant_boot_source])
        point_winner_probability = float(winner_probs.get(str(dominant_point_source), 0.0)) if dominant_point_source else np.nan
    else:
        winner_probs = {}
        dominant_boot_source = None
        dominant_boot_probability = np.nan
        point_winner_probability = np.nan

    dominance_row = {
        **bundle.coords,
        "target": target_col,
        "outcome": outcome_col,
        "dominant_source_point": dominant_point_source,
        "dominant_abs_scaled_total_effect_point": _safe_float(dominant_point_abs_effect),
        "dominant_source_boot_mode": dominant_boot_source,
        "dominant_source_boot_probability": _safe_float(dominant_boot_probability),
        "dominant_source_point_boot_probability": _safe_float(point_winner_probability),
        "dominant_source_probabilities_json": json.dumps(winner_probs),
        "n_bootstrap_total": n_bootstrap_total,
        "n_bootstrap_dominance_successful": int(np.sum(valid_boot_rows)) if boot_matrix.size else 0,
        "error": None,
    }

    return effect_rows, dominance_row


def _analyze_pixel_task(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return analyze_pixel(*args)


def _grid_from_results(df: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> pd.DataFrame:
    grid = df.pivot(index=row_col, columns=col_col, values=value_col).sort_index(ascending=True)
    return grid.reindex(sorted(grid.columns), axis=1)


def _finite_vlim(values: np.ndarray, q: float = 0.98, symmetric: bool = False) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None
    if symmetric:
        vmax = float(np.quantile(np.abs(arr), q))
        if vmax == 0:
            return None, None
        return -vmax, vmax
    vmax = float(np.quantile(arr, q))
    if vmax == 0:
        return 0.0, None
    return 0.0, vmax


def _first_available_value(df: pd.DataFrame, columns: Sequence[str], default: str = "target") -> str:
    """Return the first non-null value from any candidate column."""
    for column in columns:
        if column in df.columns:
            values = df[column].dropna()
            if not values.empty:
                return str(values.iloc[0])
    return default


def plot_effect_maps(
    effects_df: pd.DataFrame,
    row_col_cols: Sequence[str],
    output_dir: Path,
    show: bool,
    sources: Sequence[str] | None = None,
    target_col: str | None = None,
) -> list[Path]:
    """Write per-source maps for scaled total effect, bootstrap SD, and CI width."""
    if len(row_col_cols) < 2:
        return []
    row_col, col_col = list(row_col_cols)[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if sources is None:
        sources = sorted(str(x) for x in effects_df["source"].dropna().unique())

    for source in sources:
        sub = effects_df[(effects_df["source"] == source) & effects_df["error"].isna()].copy()
        if sub.empty:
            continue
        effect_grid = _grid_from_results(sub, row_col, col_col, "scaled_total_effect")
        sd_grid = _grid_from_results(sub, row_col, col_col, "scaled_total_effect_boot_sd")
        ci_grid = _grid_from_results(sub, row_col, col_col, "scaled_total_effect_boot_ci_width")
        effect_vmin, effect_vmax = _finite_vlim(effect_grid.values, symmetric=True)
        _, sd_vmax = _finite_vlim(sd_grid.values, symmetric=False)
        _, ci_vmax = _finite_vlim(ci_grid.values, symmetric=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(effect_grid.values, origin="upper", cmap="coolwarm", vmin=effect_vmin, vmax=effect_vmax)
        target_label = target_col or _first_available_value(sub, ["target", "outcome"])
        axes[0].set_title(f"{source} → {target_label}\nscaled total effect")
        plt.colorbar(im0, ax=axes[0], shrink=0.7)

        im1 = axes[1].imshow(sd_grid.values, origin="upper", cmap="viridis", vmin=0, vmax=sd_vmax)
        axes[1].set_title("Bootstrap SD")
        plt.colorbar(im1, ax=axes[1], shrink=0.7)

        im2 = axes[2].imshow(ci_grid.values, origin="upper", cmap="magma", vmin=0, vmax=ci_vmax)
        axes[2].set_title("Bootstrap CI width")
        plt.colorbar(im2, ax=axes[2], shrink=0.7)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        out_path = output_dir / f"scaled_total_effect_{_safe_filename(source)}_to_{_safe_filename(target_label)}.png"
        fig.savefig(out_path, dpi=200)
        written.append(out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return written


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def plot_dominance_map(
    dominance_df: pd.DataFrame,
    row_col_cols: Sequence[str],
    output_path: Path,
    source_order: Sequence[str] | None = None,
    show: bool = False,
    target_col: str | None = None,
) -> Path | None:
    """Plot categorical map of the source with largest absolute scaled total effect."""
    if len(row_col_cols) < 2 or dominance_df.empty:
        return None
    row_col, col_col = list(row_col_cols)[:2]
    source_col = "dominant_source_point"
    work = dominance_df[dominance_df["error"].isna()].copy()
    if work.empty:
        return None

    if source_order is None:
        source_order = sorted(str(x) for x in work[source_col].dropna().unique())
    source_to_code = {src: idx for idx, src in enumerate(source_order)}
    work["dominant_code"] = work[source_col].map(source_to_code).astype(float)
    grid = _grid_from_results(work, row_col, col_col, "dominant_code")

    # Let matplotlib supply default categorical-like colors without hard-coding scientific meaning.
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(max(1, len(source_order)))]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(grid.values, origin="upper", cmap=cmap, vmin=-0.5, vmax=len(source_order) - 0.5)
    target_label = target_col or _first_available_value(work, ["target", "outcome"])
    ax.set_title(f"Dominant causal driver of {target_label}")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Patch(facecolor=colors[i], label=src) for i, src in enumerate(source_order)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the YAML experiment config.",
)
@click.option("--target", "target", default=None, help="Override target/reference variable, e.g. ndvi. Alias: --outcome.")
@click.option("--outcome", "outcome_alias", default=None, help="Deprecated alias for --target.")
@click.option(
    "--sources",
    default=None,
    help="Comma-separated source variables. Defaults to all graph variables except target and month terms.",
)
@click.option(
    "--point-matrix",
    default=None,
    type=click.Choice(sorted(_VALID_POINT_MATRIX_CHOICES)),
    help="Point-estimate matrix. Bootstrap uncertainty always uses all saved bootstrap matrices.",
)
@click.option("--low-quantile", default=None, type=float, help="Override low quantile, default config or 0.10.")
@click.option("--high-quantile", default=None, type=float, help="Override high quantile, default config or 0.90.")
@click.option("--min-samples", default=None, type=int, help="Override analysis_min_samples.")
@click.option("--ci", default=0.95, show_default=True, type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True))
@click.option("--include-seasonality-as-source", is_flag=True, help="Include month_sin/month_cos as source variables.")
@click.option("--min-path-abs-effect", default=None, type=float, help="Ignore path edges below this absolute coefficient in path enumeration.")
@click.option("--path-top-n", default=None, type=int, help="Number of strongest point-estimate paths to serialize per source/target.")
@click.option("--max-paths-per-pair", default=None, type=int, help="Safety cap for simple path enumeration.")
@click.option("--effects-db", default=None, type=click.Path(path_type=Path), help="Override effects DuckDB path.")
@click.option("--effects-csv", default=None, type=click.Path(path_type=Path), help="Override effects CSV path.")
@click.option("--dominance-db", default=None, type=click.Path(path_type=Path), help="Override dominance DuckDB path.")
@click.option("--dominance-csv", default=None, type=click.Path(path_type=Path), help="Override dominance CSV path.")
@click.option("--plot-dir", default=None, type=click.Path(path_type=Path), help="Directory for generated PNG maps.")
@click.option("--no-plots", is_flag=True, help="Skip generating effect and dominance maps.")
@click.option("--show", is_flag=True, help="Show plots interactively as they are generated.")
@click.option("--no-progress", is_flag=True, help="Disable progress bars.")
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=int,
    help="Number of parallel worker processes.",
)
@click.option("--chunksize", default=1, show_default=True, type=int)
def per_pixel_directlingam_bootstrap_analysis(
    config_path: Path,
    target: str | None,
    outcome_alias: str | None,
    sources: str | None,
    point_matrix: str | None,
    low_quantile: float | None,
    high_quantile: float | None,
    min_samples: int | None,
    ci: float,
    include_seasonality_as_source: bool,
    min_path_abs_effect: float | None,
    path_top_n: int | None,
    max_paths_per_pair: int | None,
    effects_db: Path | None,
    effects_csv: Path | None,
    dominance_db: Path | None,
    dominance_csv: Path | None,
    plot_dir: Path | None,
    no_plots: bool,
    show: bool,
    no_progress: bool,
    jobs: int,
    chunksize: int,
) -> None:
    """Run per-pixel DirectLiNGAM effect analysis using all bootstrap adjacencies."""
    cfg = load_config(
        config_path=config_path,
        target_override=target,
        outcome_override=outcome_alias,
        sources_override=sources,
        point_matrix_override=point_matrix,
        effects_db_override=effects_db,
        effects_csv_override=effects_csv,
        dominance_db_override=dominance_db,
        dominance_csv_override=dominance_csv,
        plot_dir_override=plot_dir,
    )

    low_q = cfg.low_quantile if low_quantile is None else float(low_quantile)
    high_q = cfg.high_quantile if high_quantile is None else float(high_quantile)
    if not (0.0 <= low_q < high_q <= 1.0):
        raise click.BadParameter("Require 0 <= low_quantile < high_quantile <= 1.")

    effective_min_samples = cfg.min_samples if min_samples is None else int(min_samples)
    effective_include_seasonality = cfg.include_seasonality_as_source or include_seasonality_as_source
    effective_min_path_abs_effect = cfg.min_path_abs_effect if min_path_abs_effect is None else float(min_path_abs_effect)
    effective_path_top_n = cfg.path_top_n if path_top_n is None else int(path_top_n)
    effective_max_paths = cfg.max_paths_per_pair if max_paths_per_pair is None else int(max_paths_per_pair)

    ts_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)
    bundles = list(iter_pixel_groups(cfg, timeseries_df=ts_df, graph_df=graph_df))
    tasks = [
        (
            bundle,
            cfg.target_col,
            cfg.source_cols,
            low_q,
            high_q,
            effective_min_samples,
            cfg.point_matrix,
            effective_include_seasonality,
            effective_min_path_abs_effect,
            effective_path_top_n,
            effective_max_paths,
            ci,
        )
        for bundle in bundles
    ]

    progress_disabled = no_progress or len(tasks) == 0
    if jobs == 1:
        nested = [
            _analyze_pixel_task(task)
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc="Processing pixels",
                unit="pixel",
                disable=progress_disabled,
            )
        ]
    else:
        nested = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_analyze_pixel_task, task) for task in tasks]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing pixels using {jobs} workers",
                unit="pixel",
                disable=progress_disabled,
            )
            for future in iterator:
                nested.append(future.result())

    effect_rows = [row for rows, _ in nested for row in rows]
    dominance_rows = [row for _, row in nested]
    if not effect_rows:
        raise click.ClickException("No DirectLiNGAM effect rows were produced.")

    effects_df = pd.DataFrame(effect_rows)
    dominance_df = pd.DataFrame(dominance_rows)

    cfg.effects_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.dominance_csv.parent.mkdir(parents=True, exist_ok=True)
    effects_df.to_csv(cfg.effects_csv, index=False)
    dominance_df.to_csv(cfg.dominance_csv, index=False)

    cfg.effects_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.effects_db))
    try:
        write_dataframe_table(con, effects_df, cfg.effects_table)
    finally:
        con.close()

    cfg.dominance_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.dominance_db))
    try:
        write_dataframe_table(con, dominance_df, cfg.dominance_table)
    finally:
        con.close()

    written_plots: list[Path] = []
    if not no_plots:
        source_order = cfg.source_cols or sorted(str(x) for x in effects_df["source"].dropna().unique())
        written_plots.extend(plot_effect_maps(effects_df, cfg.row_col_cols, cfg.plot_dir, show=show, sources=source_order, target_col=cfg.target_col))
        dominance_path = cfg.plot_dir / f"dominant_{_safe_filename(cfg.target_col)}_driver.png"
        maybe_path = plot_dominance_map(
            dominance_df,
            cfg.row_col_cols,
            output_path=dominance_path,
            source_order=source_order,
            show=show,
            target_col=cfg.target_col,
        )
        if maybe_path is not None:
            written_plots.append(maybe_path)

    n_effect_failed = int(effects_df["error"].notna().sum()) if "error" in effects_df.columns else 0
    n_dom_failed = int(dominance_df["error"].notna().sum()) if "error" in dominance_df.columns else 0

    print(effects_df.head())
    print(f"\nInput ARD DB: {cfg.timeseries_db}")
    print(f"Input graph DB: {cfg.graph_db}")
    print(f"Effects CSV: {cfg.effects_csv}")
    print(f"Dominance CSV: {cfg.dominance_csv}")
    print(f"Effects DuckDB: {cfg.effects_db}::{cfg.effects_table}")
    print(f"Dominance DuckDB: {cfg.dominance_db}::{cfg.dominance_table}")
    print(f"Target: {cfg.target_col}")
    print(f"Sources: {cfg.source_cols or 'all non-target variables'}")
    print(f"Point matrix: {cfg.point_matrix}")
    print(f"Quantile contrast: Q{high_q:.2f} - Q{low_q:.2f}")
    print(f"CI level: {ci:.2f}")
    print(f"Failed effect rows: {n_effect_failed} / {len(effects_df)}")
    print(f"Failed dominance rows: {n_dom_failed} / {len(dominance_df)}")
    if written_plots:
        print("Plots:")
        for path in written_plots:
            print(f"  {path}")


if __name__ == "__main__":
    per_pixel_directlingam_bootstrap_analysis()
