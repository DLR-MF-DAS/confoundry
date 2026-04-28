#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
import hashlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import duckdb
import networkx as nx
import pandas as pd
import click
from dowhy import gcm, CausalModel


_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAGGED_VAR = re.compile(r"^(?P<base>.+)_lag(?P<offset>[+-]?\d+)$")


def _grid_from_results(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> pd.DataFrame:
    grid = (
        df.pivot(
            index=row_col,
            columns=col_col,
            values=value_col,
        )
        .sort_index(ascending=True)
    )

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
    """
    Plot causal effect, bootstrap standard error, and CI width maps.

    Black contour on the effect map marks pixels whose bootstrap CI excludes zero.
    """
    row_col_cols = list(row_col_cols)
    row_col = row_col_cols[0]
    col_col = row_col_cols[1]

    df = pd.DataFrame(results)

    if df.empty:
        raise ValueError("No results to plot.")

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

        sig_grid = _grid_from_results(
            sig_df,
            row_col,
            col_col,
            "ci_excludes_zero_numeric",
        )

        sig_values = np.nan_to_num(sig_grid.values, nan=0.0)

        if np.nanmax(sig_values) > 0:
            axes[0].contour(
                sig_values,
                levels=[0.5],
                colors="black",
                linewidths=0.6,
            )

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

    #for ax in axes:
    #    ax.set_xlabel(col_col)
    #    ax.set_ylabel(row_col)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return df


def add_lagged_columns(
    df: pd.DataFrame,
    variable_names: Sequence[str],
    drop_edge_nans: bool = True,
) -> pd.DataFrame:
    """
    Add columns like `ndvi_lag-1` when the graph contains them but the data only
    contains the base column `ndvi`.

    Convention:
      - x_lag-1 -> x shifted one step backwards in time, i.e. next row/month
      - x_lag1  -> x shifted one step forwards in time, i.e. previous row/month

    Assumes df is already sorted by time.
    """
    out = df.copy()
    generated_cols: list[str] = []

    for name in variable_names:
        if name in out.columns:
            continue

        match = _LAGGED_VAR.match(str(name))
        if not match:
            continue

        base = match.group("base")
        offset = int(match.group("offset"))

        if base not in out.columns:
            raise ValueError(
                f"Graph refers to lagged variable {name!r}, "
                f"but base column {base!r} is not present in the time-series data."
            )

        out[name] = out[base].shift(offset)
        generated_cols.append(name)

    if drop_edge_nans and generated_cols:
        out = out.dropna(subset=generated_cols).reset_index(drop=True)

    return out


@dataclass
class PixelBundle:
    key: tuple[Any, ...]
    coords: dict[str, Any]
    time_series: pd.DataFrame
    graph_row: dict[str, Any]
    graph: nx.DiGraph


def _quote_ident(name: str) -> str:
    """Restrict table names to simple SQL identifiers."""
    if not _VALID_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def _normalize_key(key: Any) -> tuple[Any, ...]:
    return key if isinstance(key, tuple) else (key,)


def _maybe_load_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def decode_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Decode JSON/GML fields from one row of the pixel_graphs table.
    Adds parsed objects alongside the original raw columns.
    """
    parsed = dict(row)

    if "variable_names_json" in parsed:
        parsed["variable_names"] = _maybe_load_json(parsed["variable_names_json"])
    if "variable_index_json" in parsed:
        parsed["variable_index"] = _maybe_load_json(parsed["variable_index_json"])
    if "causal_order_json" in parsed:
        parsed["causal_order"] = _maybe_load_json(parsed["causal_order_json"])
    if "adjacency_raw_json" in parsed:
        parsed["adjacency_raw"] = _maybe_load_json(parsed["adjacency_raw_json"])
    if "edge_probability_json" in parsed:
        parsed["edge_probability"] = _maybe_load_json(parsed["edge_probability_json"])
    if "adjacency_consensus_json" in parsed:
        parsed["adjacency_consensus"] = _maybe_load_json(parsed["adjacency_consensus_json"])

    gml_text = parsed.get("gml_graph")
    if gml_text:
        parsed["nx_graph"] = nx.parse_gml(gml_text.splitlines())
    else:
        parsed["nx_graph"] = nx.DiGraph()

    return parsed


def iter_pixel_groups(
    timeseries_db: str | Path,
    timeseries_table: str,
    graph_db: str | Path,
    graph_table: str = "pixel_graphs",
    row_col_cols: Sequence[str] = ("row", "col"),
    order_cols: Sequence[str] = ("year", "month"),
) -> Iterator[PixelBundle]:
    """
    Yield one PixelBundle per pixel key present in both databases.

    Each yielded bundle contains:
      - coords: {"row": ..., "col": ...}
      - time_series: DataFrame for that pixel, sorted by order_cols
      - graph_row: decoded row from graph table
      - graph: networkx DiGraph parsed from gml_graph
    """
    row_col_cols = list(row_col_cols)
    order_cols = list(order_cols)

    ts_con = duckdb.connect(str(timeseries_db), read_only=True)
    graph_con = duckdb.connect(str(graph_db), read_only=True)

    try:
        ts_table_sql = _quote_ident(timeseries_table)
        graph_table_sql = _quote_ident(graph_table)

        ts_df = ts_con.execute(f"SELECT * FROM {ts_table_sql}").fetchdf()
        graph_df = graph_con.execute(f"SELECT * FROM {graph_table_sql}").fetchdf()

        missing_ts = [c for c in row_col_cols + order_cols if c not in ts_df.columns]
        if missing_ts:
            raise ValueError(f"Missing required columns in time series table: {missing_ts}")

        missing_graph = [c for c in row_col_cols if c not in graph_df.columns]
        if missing_graph:
            raise ValueError(f"Missing required columns in graph table: {missing_graph}")

        dup_graphs = graph_df.duplicated(subset=row_col_cols, keep=False)
        if dup_graphs.any():
            bad_keys = (
                graph_df.loc[dup_graphs, row_col_cols]
                .drop_duplicates()
                .to_dict(orient="records")
            )
            raise ValueError(
                f"Graph table contains duplicate pixel keys for {row_col_cols}: {bad_keys}"
            )

        # Keep only pixels that exist in the graph table
        graph_keys = graph_df[row_col_cols].drop_duplicates()
        ts_df = ts_df.merge(graph_keys, on=row_col_cols, how="inner")
        ts_df = ts_df.sort_values(row_col_cols + order_cols).reset_index(drop=True)

        graph_df = graph_df.set_index(row_col_cols, drop=False)

        for key, group in ts_df.groupby(row_col_cols, sort=True):
            key = _normalize_key(key)
            graph_row_raw = graph_df.loc[key].to_dict()
            graph_row = decode_graph_row(graph_row_raw)
            graph = graph_row["nx_graph"]

            pixel_ts = group.reset_index(drop=True)

            graph_variables = set(graph.nodes)
            if graph_row.get("variable_names"):
                graph_variables.update(graph_row["variable_names"])

            pixel_ts = add_lagged_columns(
                pixel_ts,
                variable_names=graph_variables,
                drop_edge_nans=True,
            )

            yield PixelBundle(
                key=key,
                coords=dict(zip(row_col_cols, key)),
                time_series=pixel_ts,
                graph_row=graph_row,
                graph=graph,
            )

    finally:
        ts_con.close()
        graph_con.close()


def map_pixel_groups(
    timeseries_db: str | Path,
    timeseries_table: str,
    graph_db: str | Path,
    func: Callable[[PixelBundle], Any],
    graph_table: str = "pixel_graphs",
    row_col_cols: Sequence[str] = ("row", "col"),
    order_cols: Sequence[str] = ("year", "month"),
    jobs: int = 1,
    chunksize: int = 1,
    show_progress: bool = True,
) -> list[Any]:
    """
    Apply `func` to each pixel bundle and return a list of results.

    If jobs > 1, uses multiprocessing.
    """
    bundles = iter_pixel_groups(
        timeseries_db=timeseries_db,
        timeseries_table=timeseries_table,
        graph_db=graph_db,
        graph_table=graph_table,
        row_col_cols=row_col_cols,
        order_cols=order_cols,
    )

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


def _stable_pixel_seed(base_seed: int | None, key: tuple[Any, ...]) -> int | None:
    """
    Create a deterministic seed per pixel so multiprocessing remains reproducible.
    """
    if base_seed is None:
        return None

    payload = json.dumps(
        {
            "base_seed": base_seed,
            "key": [str(x) for x in key],
        },
        sort_keys=True,
    ).encode("utf-8")

    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _moving_block_bootstrap_indices(
    n: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    """
    Bootstrap indices for time series.

    block_size=1 gives ordinary row bootstrap.
    block_size>1 preserves short-range temporal structure better.
    """
    if n <= 0:
        raise ValueError("Cannot bootstrap empty data.")

    block_size = max(1, min(int(block_size), n))

    if block_size == 1:
        return rng.integers(0, n, size=n)

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    starts = rng.integers(0, max_start + 1, size=n_blocks)

    indices = np.concatenate(
        [np.arange(start, start + block_size) for start in starts]
    )

    return indices[:n]


def _estimate_effect_for_dataframe(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    treatment_col: str,
    outcome_col: str,
    control_value: float,
    treatment_value: float,
) -> float:
    """
    Fit the DoWhy model and return one causal effect estimate.
    """
    model = CausalModel(
        data=data,
        treatment=treatment_col,
        outcome=outcome_col,
        graph=graph,
    )

    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True,
    )

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        control_value=control_value,
        treatment_value=treatment_value,
        test_significance=False,
    )

    return float(estimate.value)


def _nan_summary_from_bootstrap(
    bootstrap_effects: list[float],
    ci: float,
) -> dict[str, Any]:
    """
    Summarize bootstrap effects into SE and percentile confidence interval.
    """
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
    n_bootstrap: int = 200,
    ci: float = 0.95,
    bootstrap_block_size: int = 1,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    treatment_col = "precipitation"
    outcome_col = "ndvi_lag-1"

    base_result = {
        **bundle.coords,
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
        "error": None,
    }

    try:
        if not 0 < ci < 1:
            raise ValueError(f"ci must be between 0 and 1, got {ci}")

        missing_cols = [
            c for c in [treatment_col, outcome_col]
            if c not in bundle.time_series.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        graph_cols = [c for c in bundle.graph.nodes if c in bundle.time_series.columns]
        required_cols = list(dict.fromkeys([treatment_col, outcome_col] + graph_cols))

        data = (
            bundle.time_series
            .dropna(subset=required_cols)
            .reset_index(drop=True)
        )

        if len(data) < 5:
            raise ValueError(
                f"Too few usable samples after dropping NaNs: {len(data)}"
            )

        precip = data[treatment_col].dropna()

        control_value = float(precip.quantile(0.75))
        treatment_value = float(precip.quantile(0.05))

        effect = _estimate_effect_for_dataframe(
            data=data,
            graph=bundle.graph,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            control_value=control_value,
            treatment_value=treatment_value,
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
                    control_value=control_value,
                    treatment_value=treatment_value,
                )

                if np.isfinite(boot_effect):
                    bootstrap_effects.append(float(boot_effect))
                else:
                    n_bootstrap_failed += 1

            except Exception:
                n_bootstrap_failed += 1

        base_result.update(
            _nan_summary_from_bootstrap(
                bootstrap_effects=bootstrap_effects,
                ci=ci,
            )
        )

        base_result["n_bootstrap_failed"] = n_bootstrap_failed

    except Exception as exc:
        base_result["error"] = repr(exc)

    return base_result


@click.command()
@click.option("-d", "--raster-database", required=True)
@click.option("-g", "--graph-database", required=True)
@click.option("--raster-table", required=True)
@click.option("--graph-table", default="pixel_graphs")
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    help="Number of parallel worker processes.",
)
@click.option(
    "--chunksize",
    default=1,
    show_default=True,
    help="Chunksize for multiprocessing.",
)
@click.option(
    "--output-csv",
    default="causal_effects.csv",
    show_default=True,
    help="Where to save the per-pixel results.",
)
@click.option(
    "--plot-output",
    default="causal_effect_map.png",
    show_default=True,
    help="Where to save the plotted effect map.",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Save the plot but do not open an interactive window.",
)
@click.option(
    "--n-bootstrap",
    default=200,
    show_default=True,
    help="Number of bootstrap replicates per pixel. Use 0 to disable uncertainty estimation.",
)
@click.option(
    "--bootstrap-block-size",
    default=1,
    show_default=True,
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
    help="Base random seed for reproducible per-pixel bootstraps.",
)
def per_pixel_analysis(
    raster_database,
    graph_database,
    raster_table,
    graph_table,
    jobs,
    chunksize,
    output_csv,
    plot_output,
    no_show,
    n_bootstrap,
    bootstrap_block_size,
    ci,
    random_seed,
):
    effect_func = partial(
        causal_effect,
        n_bootstrap=n_bootstrap,
        ci=ci,
        bootstrap_block_size=bootstrap_block_size,
        random_seed=random_seed,
    )

    results = map_pixel_groups(
        timeseries_db=raster_database,
        timeseries_table=raster_table,
        graph_db=graph_database,
        graph_table=graph_table,
        func=effect_func,
        jobs=jobs,
        chunksize=chunksize,
        show_progress=True,
    )

    results_df = plot_effect_and_uncertainty_maps(
        results,
        output_path=plot_output,
        show=not no_show,
    )

    results_df.to_csv(output_csv, index=False)

    n_failed = results_df["error"].notna().sum()
    n_uncertainty_failed = results_df["n_bootstrap_successful"].eq(0).sum()

    print(results_df.head())
    print(f"\nSaved results to: {output_csv}")
    print(f"Saved plot to: {plot_output}")
    print(f"Failed pixels: {n_failed} / {len(results_df)}")
    print(
        f"Pixels without usable bootstrap uncertainty: "
        f"{n_uncertainty_failed} / {len(results_df)}"
    )


if __name__ == '__main__':
    per_pixel_analysis()
