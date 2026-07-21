"""Discover causal graphs for individual pixels or pixel neighborhoods.

This command reads pixel-wise time-series data from a DuckDB database, applies
configured temporal shifts to selected variables, fits a DirectLiNGAM model for
each pixel or pixel-centered spatial window, and writes causal matrices plus GML
graph representations to a DuckDB output database.

Statistics and DirectLiNGAM suitability diagnostics are intentionally handled by
``graph_statistics.py`` so graph discovery and post-hoc evaluation can be run as
separate steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import lingam
import networkx as nx
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


def make_prior_knowledge(labels: Sequence[str], label_lags: Mapping[str, int]) -> np.ndarray:
    """Construct a DirectLiNGAM prior-knowledge matrix from variable lags."""
    prior_knowledge = -np.ones((len(labels), len(labels)), dtype=int)

    for parent_idx, parent_name in enumerate(labels):
        for child_idx, child_name in enumerate(labels):
            if parent_idx != child_idx and label_lags[parent_name] < label_lags[child_name]:
                prior_knowledge[child_idx, parent_idx] = 0
            if child_name in {"month_sin", "month_cos"}:
                prior_knowledge[child_idx, parent_idx] = 0

    return prior_knowledge


def to_graph(B: np.ndarray, labels: Sequence[str], min_abs_effect: float) -> nx.DiGraph:
    """Convert a LiNGAM adjacency matrix to a directed NetworkX graph."""
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)

    for child_idx, child_name in enumerate(labels):
        for parent_idx, parent_name in enumerate(labels):
            coefficient = B[child_idx, parent_idx]
            if child_idx != parent_idx and abs(coefficient) >= min_abs_effect:
                graph.add_edge(parent_name, child_name, weight=float(coefficient))

    return graph


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
    return path if path.is_absolute() else base_dir / path


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


def fit_pixel(
    pixel_key: PixelKey,
    g: pd.DataFrame,
    labels: Sequence[str],
    pk: np.ndarray,
    bootstrap_samples: int,
    min_samples: int,
    min_prob: float,
    min_abs_effect: float,
    group_cols: Sequence[str],
) -> dict[str, Any] | None:
    """Fit a consensus causal graph for one pixel/window."""
    complete_g = g.dropna(subset=list(labels)).copy()
    X = complete_g[list(labels)].to_numpy()

    if len(X) < min_samples:
        return None

    model = lingam.DirectLiNGAM(
        prior_knowledge=pk,
        random_state=0,
    )
    model.fit(X)

    boot = model.bootstrap(X, n_sampling=bootstrap_samples)
    probabilities = np.asarray(
        boot.get_probabilities(min_causal_effect=min_abs_effect),
        dtype=float,
    )
    bootstrap_adjacencies = np.asarray(boot.adjacency_matrices_, dtype=float)
    raw_adjacency = np.asarray(model.adjacency_matrix_, dtype=float)
    consensus_adjacency = np.where(probabilities >= min_prob, raw_adjacency, 0.0)
    consensus_adjacency = np.where(
        np.abs(consensus_adjacency) >= min_abs_effect,
        consensus_adjacency,
        0.0,
    )

    graph = to_graph(consensus_adjacency, labels, min_abs_effect)
    serialized_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)
    graph_row = dict(zip(group_cols, serialized_pixel_key, strict=False))
    graph_row.update(
        n_samples=int(len(X)),
        variable_names_json=json.dumps(list(labels)),
        variable_index_json=json.dumps({name: idx for idx, name in enumerate(labels)}),
        causal_order_json=json.dumps([int(idx) for idx in model.causal_order_]),
        adjacency_raw_json=json.dumps(raw_adjacency.tolist()),
        edge_probability_json=json.dumps(probabilities.tolist()),
        adjacency_consensus_json=json.dumps(consensus_adjacency.tolist()),
        adjacency_bootstrap_json=json.dumps(bootstrap_adjacencies.tolist()),
        gml_graph="\n".join(nx.generate_gml(graph)),
    )

    return graph_row


def fit_pixel_task(args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Unpack a multiprocessing task tuple and fit one pixel graph."""
    (
        pixel_key,
        g,
        labels,
        pk,
        bootstrap_samples,
        min_samples,
        min_edge_prob,
        min_abs_effect,
        row_col_cols,
    ) = args

    return fit_pixel(
        pixel_key=pixel_key,
        g=g,
        labels=labels,
        pk=pk,
        bootstrap_samples=bootstrap_samples,
        min_samples=min_samples,
        min_prob=min_edge_prob,
        min_abs_effect=min_abs_effect,
        group_cols=row_col_cols,
    )


@click.command()
@click.option("-c", "--config-path", help="Path to the YAML config file with experiment parameters", required=True)
@click.option("-b", "--bootstrap-samples", default=200, show_default=True, type=int)
@click.option("--min-samples", default=50, show_default=True, type=int)
@click.option("--min-edge-prob", default=0.7, show_default=True, type=float)
@click.option("--min-abs-effect", default=0.01, show_default=True, type=float)
@click.option("--window-size", default=0, show_default=True, type=int)
@click.option("-w", "--workers", default=1, show_default=True, type=int)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--min-year", default=None, type=int)
@click.option("--max-year", default=None, type=int)
def graph_discovery(
    config_path: str,
    bootstrap_samples: int,
    min_samples: int,
    min_edge_prob: float,
    min_abs_effect: float,
    window_size: int,
    workers: int,
    input_db: Path | None,
    input_table: str | None,
    output_db: Path | None,
    min_year: int | None,
    max_year: int | None,
) -> None:
    """Run pixel-wise causal graph discovery from the command line.

    The YAML configuration is expected to live in an experiment directory that
    also contains an input DuckDB database named ``<name>_ard.duckdb``. The input
    table must be named ``<name>`` and must contain ``row``, ``col``, ``year``,
    and ``month`` columns, plus all configured variable columns. Graph results are
    written to ``<name>_graphs.duckdb`` in a table named ``pixel_graphs``.

    Run ``graph_statistics.py`` afterwards to compute diagnostics/statistics from
    the saved graph table.
    """
    if window_size < 0:
        raise click.BadParameter("window-size must be >= 0")

    row_col_cols = ["row", "col"]
    order_cols = ["year", "month"]
    config_path_obj = Path(config_path)

    with config_path_obj.open("r") as fd:
        config_data = yaml.safe_load(fd)

    experiment_dir = config_path_obj.parent
    location_nickname = config_data["name"]
    input_db = resolve_path(
        experiment_dir,
        input_db or graph_config_value(config_data, "input_db")
        or graph_config_value(config_data, "timeseries_db"),
        experiment_dir / f"{location_nickname}_ard.duckdb",
    )
    output_db = resolve_path(
        experiment_dir,
        output_db or graph_config_value(config_data, "output_db")
        or graph_config_value(config_data, "graph_db"),
        experiment_dir / f"{location_nickname}_graphs.duckdb",
    )
    input_table = (
        input_table
        or graph_config_value(config_data, "input_table")
        or graph_config_value(config_data, "timeseries_table")
        or location_nickname
    )
    min_year = min_year if min_year is not None else graph_config_value(
        config_data,
        "min_year",
    )
    max_year = max_year if max_year is not None else graph_config_value(
        config_data,
        "max_year",
    )
    columns = config_data["columns"]

    con = duckdb.connect(input_db, read_only=True)
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if input_table not in tables:
        con.close()
        raise click.BadParameter(
            f"{input_table} not found in {input_db}. Available: {sorted(tables)}"
        )

    df = con.execute(f"SELECT * FROM {quote_identifier(input_table)}").fetchdf()
    con.close()

    missing_required = [col for col in row_col_cols + order_cols if col not in df.columns]
    if missing_required:
        raise click.BadParameter(f"Missing required columns: {missing_required}")
    if min_year is not None:
        df = df[df["year"].astype(int) >= int(min_year)].copy()
    if max_year is not None:
        df = df[df["year"].astype(int) <= int(max_year)].copy()
    if df.empty:
        raise click.ClickException("No rows remain after year filtering.")

    df, labels, label_lags = parse_columns(df, row_col_cols, order_cols, columns)
    df = df.dropna(subset=labels + row_col_cols + order_cols)
    prior_knowledge = make_prior_knowledge(labels, label_lags)

    groups = list(df.groupby(row_col_cols, sort=True))
    group_lookup = {
        pixel_key if isinstance(pixel_key, tuple) else (pixel_key,): group
        for pixel_key, group in groups
    }

    tasks = []
    for pixel_key, _ in groups:
        center_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)

        if window_size == 0:
            window_group = group_lookup[center_pixel_key]
        else:
            window_group = get_pixel_window_group(
                pixel_key=center_pixel_key,
                group_lookup=group_lookup,
                window_size=window_size,
            )

        if window_group is None:
            continue

        tasks.append(
            (
                center_pixel_key,
                window_group,
                labels,
                prior_knowledge,
                bootstrap_samples,
                min_samples,
                min_edge_prob,
                min_abs_effect,
                row_col_cols,
            )
        )

    results = process_map(
        fit_pixel_task,
        tasks,
        max_workers=workers,
        chunksize=1,
        desc="Pixels",
    )
    graph_rows = [result for result in results if result is not None]

    if not graph_rows:
        raise click.ClickException("No pixel had enough samples after lagging/dropna.")

    result_df = pd.DataFrame(graph_rows)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, result_df, "pixel_graphs")
    finally:
        con.close()


if __name__ == "__main__":
    graph_discovery()
