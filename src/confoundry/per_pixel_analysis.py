#!/usr/bin/env python3
"""Estimate per-pixel causal effects from config-driven Confoundry outputs.

This version follows the newer Confoundry pipeline conventions used by
``gather.py`` and ``per_pixel_graph_discovery.py``:

- read one experiment YAML via ``--config-path``;
- derive ``<name>_ard.duckdb`` and ``<name>_graphs.duckdb`` from the config;
- use the table named by ``config["name"]`` for the ARD time-series table;
- apply ``config["columns"][].shift`` using the same in-place lag semantics as
  graph discovery, so analysis uses exactly the same temporally shifted variables;
- read the graph table ``pixel_graphs`` produced by graph discovery;
- write per-pixel effects to CSV and to a DuckDB table.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
import hashlib
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
from dowhy import CausalModel
from tqdm import tqdm

PixelKey = tuple[Any, ...]
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AnalysisConfig:
    experiment_dir: Path
    location_name: str
    columns: list[dict[str, Any]]
    treatment_col: str
    outcome_col: str
    timeseries_db: Path
    graph_db: Path
    output_db: Path
    timeseries_table: str
    graph_table: str
    output_table: str
    output_csv: Path
    plot_output: Path
    row_col_cols: list[str]
    order_cols: list[str]
    estimator_method: str
    control_value: float | None
    treatment_value: float | None
    control_quantile: float
    treatment_quantile: float
    min_samples: int


@dataclass
class PixelBundle:
    key: PixelKey
    coords: dict[str, Any]
    time_series: pd.DataFrame
    graph_row: dict[str, Any]
    graph: nx.DiGraph


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted DuckDB identifier for simple table/column names."""
    if not _VALID_IDENTIFIER.fullmatch(identifier):
        raise click.BadParameter(
            f"Invalid DuckDB identifier: {identifier!r}. "
            "Use letters, numbers, and underscores."
        )
    return f'"{identifier}"'


def _normalize_key(key: Any) -> PixelKey:
    return key if isinstance(key, tuple) else (key,)


def _as_path(base_dir: Path, value: str | Path | None, default_name: str | Path) -> Path:
    """Resolve paths relative to the directory containing the YAML config.

    Pipeline convention: databases and analysis outputs live next to the
    experiment config unless an absolute path is explicitly provided.
    """
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


def _column_names(columns: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(spec["name"]) for spec in columns]


def _get_analysis_value(
    config_data: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    analysis_config = config_data.get("analysis") or {}
    if not isinstance(analysis_config, Mapping):
        raise click.BadParameter("config['analysis'] must be a mapping when present.")
    return analysis_config.get(key, config_data.get(key, default))


def load_analysis_config(
    config_path: str | Path,
    treatment_override: str | None = None,
    outcome_override: str | None = None,
) -> AnalysisConfig:
    """Load experiment config and derive all analysis paths/settings."""
    # Resolve the config path once. From here on, every relative database/output
    # path is interpreted relative to the directory containing this YAML file.
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
    columns: list[dict[str, Any]] = [dict(spec) for spec in columns_raw]
    configured_columns = _column_names(columns)

    # Prefer explicit analysis settings, but keep a useful fallback for the
    # existing drought example: precipitation -> shifted ndvi.
    treatment_col = (
        treatment_override
        or _get_analysis_value(config_data, "treatment")
        or ("precipitation" if "precipitation" in configured_columns else None)
    )
    outcome_col = (
        outcome_override
        or _get_analysis_value(config_data, "outcome")
        or ("ndvi" if "ndvi" in configured_columns else None)
    )
    if treatment_col is None or outcome_col is None:
        raise click.BadParameter(
            "Set analysis.treatment and analysis.outcome in the YAML config, "
            "or pass --treatment and --outcome. With the new lag system, use "
            "the configured column name after shifting, e.g. outcome: ndvi, "
            "not ndvi_lag-1."
        )

    timeseries_db = _as_path(
        experiment_dir,
        _get_analysis_value(config_data, "timeseries_db"),
        f"{location_name}_ard.duckdb",
    )
    graph_db = _as_path(
        experiment_dir,
        _get_analysis_value(config_data, "graph_db"),
        f"{location_name}_graphs.duckdb",
    )
    output_db = _as_path(
        experiment_dir,
        _get_analysis_value(config_data, "output_db"),
        f"{location_name}_effects.duckdb",
    )
    output_csv = _as_path(
        experiment_dir,
        _get_analysis_value(config_data, "output_csv"),
        f"{location_name}_causal_effects.csv",
    )
    plot_output = _as_path(
        experiment_dir,
        _get_analysis_value(config_data, "plot_output"),
        f"{location_name}_causal_effect_map.png",
    )

    return AnalysisConfig(
        experiment_dir=experiment_dir,
        location_name=location_name,
        columns=columns,
        treatment_col=str(treatment_col),
        outcome_col=str(outcome_col),
        timeseries_db=timeseries_db,
        graph_db=graph_db,
        output_db=output_db,
        timeseries_table=str(_get_analysis_value(config_data, "timeseries_table", location_name)),
        graph_table=str(_get_analysis_value(config_data, "graph_table", "pixel_graphs")),
        output_table=str(_get_analysis_value(config_data, "output_table", "pixel_effects")),
        output_csv=output_csv,
        plot_output=plot_output,
        row_col_cols=list(_get_analysis_value(config_data, "row_col_cols", ["row", "col"])),
        order_cols=list(_get_analysis_value(config_data, "order_cols", ["year", "month"])),
        estimator_method=str(
            _get_analysis_value(config_data, "estimator_method", "backdoor.linear_regression")
        ),
        control_value=_get_analysis_value(config_data, "control_value"),
        treatment_value=_get_analysis_value(config_data, "treatment_value"),
        control_quantile=float(_get_analysis_value(config_data, "control_quantile", 0.75)),
        treatment_quantile=float(_get_analysis_value(config_data, "treatment_quantile", 0.05)),
        min_samples=int(_get_analysis_value(config_data, "analysis_min_samples", 5)),
    )


def parse_columns(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    column_specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Apply the same configured temporal shifts as graph discovery.

    Important: this shifts configured variables in place. A spec such as
    ``{name: ndvi, shift: -1}`` keeps the column name ``ndvi`` but replaces its
    values with the per-pixel shifted series. This mirrors the current
    ``per_pixel_graph_discovery.py`` behavior.
    """
    shifted_df = df.sort_values(list(group_cols) + list(order_cols)).copy()
    labels: list[str] = []
    label_lags: dict[str, int] = {}

    for spec in column_specs:
        if "name" not in spec or "shift" not in spec:
            raise click.BadParameter(
                "Every config['columns'] entry must contain 'name' and 'shift'."
            )
        label = str(spec["name"])
        lag = int(spec["shift"])
        if label in labels:
            raise click.BadParameter(f"Duplicate configured column: {label}")
        if label not in shifted_df.columns:
            raise click.BadParameter(f"Missing data column: {label}")
        shifted_df[label] = shifted_df.groupby(list(group_cols))[label].shift(lag)
        labels.append(label)
        label_lags[label] = lag

    return shifted_df, labels, label_lags


def decode_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON/GML fields from one row of the pixel_graphs table."""
    parsed = dict(row)
    json_fields = [
        ("variable_names_json", "variable_names"),
        ("variable_index_json", "variable_index"),
        ("causal_order_json", "causal_order"),
        ("adjacency_raw_json", "adjacency_raw"),
        ("edge_probability_json", "edge_probability"),
        ("adjacency_consensus_json", "adjacency_consensus"),
    ]
    for raw_key, parsed_key in json_fields:
        if raw_key in parsed:
            parsed[parsed_key] = _maybe_load_json(parsed[raw_key])

    gml_text = parsed.get("gml_graph")
    parsed["nx_graph"] = nx.parse_gml(gml_text.splitlines()) if gml_text else nx.DiGraph()
    return parsed


def _read_table(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {quote_identifier(table_name)}").fetchdf()


def _ensure_table_exists(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    db_path: Path,
) -> None:
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if table_name not in tables:
        raise click.BadParameter(
            f"{table_name!r} not found in {db_path}. Available tables: {sorted(tables)}"
        )


def load_shifted_timeseries_and_graphs(cfg: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read DuckDB inputs and apply config-defined temporal shifts to ARD data."""
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

    missing_graph = [c for c in cfg.row_col_cols if c not in graph_df.columns]
    if missing_graph:
        raise click.BadParameter(f"Missing required columns in graph table: {missing_graph}")

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


def iter_pixel_groups(
    cfg: AnalysisConfig,
    timeseries_df: pd.DataFrame | None = None,
    graph_df: pd.DataFrame | None = None,
) -> Iterator[PixelBundle]:
    """Yield one PixelBundle per pixel key present in both shifted data and graphs."""
    if timeseries_df is None or graph_df is None:
        timeseries_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)

    graph_keys = graph_df[cfg.row_col_cols].drop_duplicates()
    ts_df = timeseries_df.merge(graph_keys, on=cfg.row_col_cols, how="inner")
    ts_df = ts_df.sort_values(cfg.row_col_cols + cfg.order_cols).reset_index(drop=True)
    graph_df = graph_df.set_index(cfg.row_col_cols, drop=False)

    for key, group in ts_df.groupby(cfg.row_col_cols, sort=True):
        key = _normalize_key(key)
        graph_row_raw = graph_df.loc[key].to_dict()
        graph_row = decode_graph_row(graph_row_raw)
        graph = graph_row["nx_graph"]
        yield PixelBundle(
            key=key,
            coords=dict(zip(cfg.row_col_cols, key, strict=False)),
            time_series=group.reset_index(drop=True),
            graph_row=graph_row,
            graph=graph,
        )


def map_pixel_groups(
    cfg: AnalysisConfig,
    func: Callable[[PixelBundle], Any],
    jobs: int = 1,
    chunksize: int = 1,
    show_progress: bool = True,
) -> list[Any]:
    """Apply ``func`` to every pixel bundle and return results."""
    ts_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)
    bundles = iter_pixel_groups(cfg, timeseries_df=ts_df, graph_df=graph_df)

    if jobs == 1:
        iterator = tqdm(
            bundles,
            desc="Processing pixels",
            unit="pixel",
            disable=not show_progress,
        )
        return [func(bundle) for bundle in iterator]

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        iterator = executor.map(func, bundles, chunksize=chunksize)
        return list(
            tqdm(
                iterator,
                desc=f"Processing pixels using {jobs} workers",
                unit="pixel",
                disable=not show_progress,
            )
        )


def _grid_from_results(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> pd.DataFrame:
    grid = df.pivot(index=row_col, columns=col_col, values=value_col).sort_index(ascending=True)
    return grid.reindex(sorted(grid.columns), axis=1)


def _finite_quantile(values: np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.quantile(arr, q))


def plot_effect_and_uncertainty_maps(
    results: Sequence[dict[str, Any]],
    row_col_cols: Sequence[str] = ("row", "col"),
    effect_col: str = "effect",
    se_col: str = "effect_se",
    ci_width_col: str = "effect_ci_width",
    ci_low_col: str = "effect_ci_low",
    ci_high_col: str = "effect_ci_high",
    output_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Plot causal effect, bootstrap SE, and CI width maps."""
    row_col, col_col = list(row_col_cols)[:2]
    df = pd.DataFrame(results)
    if df.empty:
        raise click.ClickException("No results to plot.")

    effect_grid = _grid_from_results(df, row_col, col_col, effect_col)
    se_grid = _grid_from_results(df, row_col, col_col, se_col)
    ci_width_grid = _grid_from_results(df, row_col, col_col, ci_width_col)

    max_abs_effect = _finite_quantile(np.abs(effect_grid.values), 0.98)
    if max_abs_effect is None or max_abs_effect == 0:
        effect_vmin, effect_vmax = None, None
    else:
        effect_vmin, effect_vmax = -max_abs_effect, max_abs_effect

    se_vmax = _finite_quantile(se_grid.values, 0.98)
    ci_width_vmax = _finite_quantile(ci_width_grid.values, 0.98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(
        effect_grid.values,
        origin="upper",
        cmap="coolwarm",
        vmin=effect_vmin,
        vmax=effect_vmax,
    )
    axes[0].set_title("Per-pixel causal effect")
    plt.colorbar(im0, ax=axes[0], shrink=0.6, label=effect_col)

    if ci_low_col in df.columns and ci_high_col in df.columns:
        sig_df = df.copy()
        sig_df["ci_excludes_zero_numeric"] = (
            (sig_df[ci_low_col] > 0) | (sig_df[ci_high_col] < 0)
        ).astype(float)
        sig_grid = _grid_from_results(sig_df, row_col, col_col, "ci_excludes_zero_numeric")
        sig_values = np.nan_to_num(sig_grid.values, nan=0.0)
        if np.nanmax(sig_values) > 0:
            axes[0].contour(sig_values, levels=[0.5], colors="black", linewidths=0.6)

    im1 = axes[1].imshow(
        se_grid.values,
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=se_vmax,
    )
    axes[1].set_title("Bootstrap standard error")
    plt.colorbar(im1, ax=axes[1], shrink=0.6, label=se_col)

    im2 = axes[2].imshow(
        ci_width_grid.values,
        origin="upper",
        cmap="magma",
        vmin=0,
        vmax=ci_width_vmax,
    )
    axes[2].set_title("Bootstrap CI width")
    plt.colorbar(im2, ax=axes[2], shrink=0.6, label=ci_width_col)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df


def _stable_pixel_seed(base_seed: int | None, key: PixelKey) -> int | None:
    """Create a deterministic seed per pixel for reproducible multiprocessing."""
    if base_seed is None:
        return None
    payload = json.dumps(
        {"base_seed": base_seed, "key": [str(x) for x in key]},
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _moving_block_bootstrap_indices(
    n: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    """Bootstrap indices for time series; block_size=1 gives row bootstrap."""
    if n <= 0:
        raise ValueError("Cannot bootstrap empty data.")
    block_size = max(1, min(int(block_size), n))
    if block_size == 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    indices = np.concatenate([np.arange(start, start + block_size) for start in starts])
    return indices[:n]


def _intervention_values(
    data: pd.DataFrame,
    treatment_col: str,
    control_value: float | None,
    treatment_value: float | None,
    control_quantile: float,
    treatment_quantile: float,
) -> tuple[float, float]:
    treatment_series = data[treatment_col].dropna()
    if treatment_series.empty:
        raise ValueError(f"Treatment column {treatment_col!r} has no finite values.")
    resolved_control = (
        float(control_value)
        if control_value is not None
        else float(treatment_series.quantile(control_quantile))
    )
    resolved_treatment = (
        float(treatment_value)
        if treatment_value is not None
        else float(treatment_series.quantile(treatment_quantile))
    )
    return resolved_control, resolved_treatment


def _estimate_effect_for_dataframe(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    treatment_col: str,
    outcome_col: str,
    control_value: float,
    treatment_value: float,
    estimator_method: str,
) -> float:
    """Fit the DoWhy model and return one causal effect estimate."""
    model = CausalModel(
        data=data,
        treatment=treatment_col,
        outcome=outcome_col,
        graph=graph,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=estimator_method,
        control_value=control_value,
        treatment_value=treatment_value,
        test_significance=False,
    )
    return float(estimate.value)


def _nan_summary_from_bootstrap(
    bootstrap_effects: Sequence[float],
    ci: float,
) -> dict[str, Any]:
    """Summarize bootstrap effects into SE and percentile confidence interval."""
    arr = np.asarray(bootstrap_effects, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "effect_bootstrap_mean": np.nan,
            "effect_se": np.nan,
            "effect_ci_low": np.nan,
            "effect_ci_high": np.nan,
            "effect_ci_width": np.nan,
            "ci_excludes_zero": False,
            "n_bootstrap_successful": 0,
        }

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(arr, [alpha, 1.0 - alpha])
    effect_se = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
    return {
        "effect_bootstrap_mean": float(np.mean(arr)),
        "effect_se": effect_se,
        "effect_ci_low": float(ci_low),
        "effect_ci_high": float(ci_high),
        "effect_ci_width": float(ci_high - ci_low),
        "ci_excludes_zero": bool((ci_low > 0) or (ci_high < 0)),
        "n_bootstrap_successful": int(len(arr)),
    }


def causal_effect(
    bundle: PixelBundle,
    treatment_col: str,
    outcome_col: str,
    estimator_method: str = "backdoor.linear_regression",
    control_value: float | None = None,
    treatment_value: float | None = None,
    control_quantile: float = 0.75,
    treatment_quantile: float = 0.05,
    min_samples: int = 5,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    bootstrap_block_size: int = 1,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """Estimate one configured treatment/outcome effect for a pixel bundle."""
    base_result: dict[str, Any] = {
        **bundle.coords,
        "treatment": treatment_col,
        "outcome": outcome_col,
        "effect": np.nan,
        "effect_bootstrap_mean": np.nan,
        "effect_se": np.nan,
        "effect_ci_low": np.nan,
        "effect_ci_high": np.nan,
        "effect_ci_width": np.nan,
        "ci_excludes_zero": False,
        "n_samples": len(bundle.time_series),
        "n_edges": bundle.graph.number_of_edges(),
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_successful": 0,
        "n_bootstrap_failed": 0,
        "ci_level": ci,
        "bootstrap_block_size": bootstrap_block_size,
        "control_value": np.nan,
        "treatment_value": np.nan,
        "estimator_method": estimator_method,
        "error": None,
    }

    try:
        if not 0 < ci < 1:
            raise ValueError(f"ci must be between 0 and 1, got {ci}")

        missing_cols = [
            c for c in [treatment_col, outcome_col] if c not in bundle.time_series.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        missing_graph_nodes = [n for n in [treatment_col, outcome_col] if n not in bundle.graph.nodes]
        if missing_graph_nodes:
            raise ValueError(
                "Treatment/outcome missing from graph nodes: "
                f"{missing_graph_nodes}. Check analysis.treatment/outcome against config.columns."
            )

        graph_cols = list(bundle.graph.nodes)
        missing_graph_cols = [c for c in graph_cols if c not in bundle.time_series.columns]
        if missing_graph_cols:
            raise ValueError(f"Graph nodes missing from shifted time-series data: {missing_graph_cols}")

        required_cols = list(dict.fromkeys([treatment_col, outcome_col] + graph_cols))
        data = bundle.time_series.dropna(subset=required_cols).reset_index(drop=True)
        if len(data) < min_samples:
            raise ValueError(
                f"Too few usable samples after dropping NaNs: {len(data)} < {min_samples}"
            )

        resolved_control, resolved_treatment = _intervention_values(
            data=data,
            treatment_col=treatment_col,
            control_value=control_value,
            treatment_value=treatment_value,
            control_quantile=control_quantile,
            treatment_quantile=treatment_quantile,
        )
        base_result["control_value"] = resolved_control
        base_result["treatment_value"] = resolved_treatment

        effect = _estimate_effect_for_dataframe(
            data=data,
            graph=bundle.graph,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            control_value=resolved_control,
            treatment_value=resolved_treatment,
            estimator_method=estimator_method,
        )
        base_result["effect"] = effect
        base_result["n_samples"] = len(data)

        if n_bootstrap <= 0:
            return base_result

        pixel_seed = _stable_pixel_seed(random_seed, bundle.key)
        rng = np.random.default_rng(pixel_seed)
        bootstrap_effects: list[float] = []
        n_bootstrap_failed = 0

        for _ in range(n_bootstrap):
            indices = _moving_block_bootstrap_indices(
                n=len(data),
                rng=rng,
                block_size=bootstrap_block_size,
            )
            boot_data = data.iloc[indices].reset_index(drop=True)
            try:
                boot_effect = _estimate_effect_for_dataframe(
                    data=boot_data,
                    graph=bundle.graph,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    control_value=resolved_control,
                    treatment_value=resolved_treatment,
                    estimator_method=estimator_method,
                )
                if np.isfinite(boot_effect):
                    bootstrap_effects.append(float(boot_effect))
                else:
                    n_bootstrap_failed += 1
            except Exception:
                n_bootstrap_failed += 1

        base_result.update(_nan_summary_from_bootstrap(bootstrap_effects, ci=ci))
        base_result["n_bootstrap_failed"] = n_bootstrap_failed

    except Exception as exc:
        base_result["error"] = repr(exc)

    return base_result


def write_dataframe_table(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Create or replace a DuckDB table from a pandas data frame."""
    quoted_table = quote_identifier(table_name)
    con.register("_write_df", df)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {quoted_table} AS SELECT * FROM _write_df")
    finally:
        con.unregister("_write_df")


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the YAML experiment config.",
)
@click.option(
    "--treatment",
    default=None,
    help="Override analysis.treatment from the config.",
)
@click.option(
    "--outcome",
    default=None,
    help="Override analysis.outcome from the config.",
)
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=int,
    help="Number of parallel worker processes.",
)
@click.option("--chunksize", default=1, show_default=True, type=int)
@click.option(
    "--n-bootstrap",
    default=200,
    show_default=True,
    type=int,
    help="Number of bootstrap replicates per pixel. Use 0 to disable uncertainty.",
)
@click.option(
    "--bootstrap-block-size",
    default=1,
    show_default=True,
    type=int,
    help="Block size for time-series bootstrap. Use 1 for ordinary row bootstrap.",
)
@click.option(
    "--ci",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    help="Bootstrap confidence interval level.",
)
@click.option(
    "--random-seed",
    default=42,
    show_default=True,
    type=int,
    help="Base random seed for reproducible per-pixel bootstraps.",
)
@click.option("--no-show", is_flag=True, help="Save the plot but do not open a window.")
def per_pixel_analysis(
    config_path: Path,
    treatment: str | None,
    outcome: str | None,
    jobs: int,
    chunksize: int,
    n_bootstrap: int,
    bootstrap_block_size: int,
    ci: float,
    random_seed: int,
    no_show: bool,
) -> None:
    """Run config-driven per-pixel causal effect analysis."""
    cfg = load_analysis_config(
        config_path=config_path,
        treatment_override=treatment,
        outcome_override=outcome,
    )

    effect_func = partial(
        causal_effect,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        estimator_method=cfg.estimator_method,
        control_value=cfg.control_value,
        treatment_value=cfg.treatment_value,
        control_quantile=cfg.control_quantile,
        treatment_quantile=cfg.treatment_quantile,
        min_samples=cfg.min_samples,
        n_bootstrap=n_bootstrap,
        ci=ci,
        bootstrap_block_size=bootstrap_block_size,
        random_seed=random_seed,
    )

    results = map_pixel_groups(
        cfg=cfg,
        func=effect_func,
        jobs=jobs,
        chunksize=chunksize,
        show_progress=True,
    )

    results_df = plot_effect_and_uncertainty_maps(
        results,
        row_col_cols=cfg.row_col_cols,
        output_path=cfg.plot_output,
        show=not no_show,
    )

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(cfg.output_csv, index=False)

    cfg.output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.output_db))
    try:
        write_dataframe_table(con, results_df, cfg.output_table)
    finally:
        con.close()

    n_failed = int(results_df["error"].notna().sum())
    n_uncertainty_failed = int(results_df["n_bootstrap_successful"].eq(0).sum())

    print(results_df.head())
    print(f"\nInput ARD DB: {cfg.timeseries_db}")
    print(f"Input graph DB: {cfg.graph_db}")
    print(f"Treatment -> outcome: {cfg.treatment_col} -> {cfg.outcome_col}")
    print(f"Saved CSV results to: {cfg.output_csv}")
    print(f"Saved DuckDB results to: {cfg.output_db}::{cfg.output_table}")
    print(f"Saved plot to: {cfg.plot_output}")
    print(f"Failed pixels: {n_failed} / {len(results_df)}")
    print(
        "Pixels without usable bootstrap uncertainty: "
        f"{n_uncertainty_failed} / {len(results_df)}"
    )


if __name__ == "__main__":
    per_pixel_analysis()
