"""Discover causal graphs for individual pixels or pixel neighborhoods.

This module reads pixel-wise time-series data from a DuckDB database, applies
configured temporal shifts to selected variables, fits a DirectLiNGAM model for
each pixel or pixel-centered spatial window, and writes the resulting causal
matrices and graph representations to a DuckDB output database.

The command-line interface is exposed through :func:`graph_discovery`.
"""

from __future__ import annotations

import json
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
    """Collect pixel groups in a square neighborhood around a center pixel.

    The ``window_size`` parameter is interpreted as a pixel radius around the
    center pixel. A value of ``0`` returns only the center pixel, ``1`` returns
    the available pixels in a ``3 x 3`` square, ``2`` returns the available
    pixels in a ``5 x 5`` square, and so on. Missing neighboring pixels are
    ignored, which allows the function to work at image boundaries or with
    sparse pixel grids.

    Parameters
    ----------
    pixel_key : tuple of int
        Center pixel as ``(row, col)``.
    group_lookup : mapping of tuple of int to pandas.DataFrame
        Mapping from pixel keys to data frames containing the observations for
        those pixels.
    window_size : int
        Radius of the square window around ``pixel_key``. Must be non-negative.

    Returns
    -------
    pandas.DataFrame or None
        Concatenated data for all available pixels in the square window. Returns
        ``None`` when no pixel groups are available in the requested window.

    Raises
    ------
    ValueError
        If ``window_size`` is negative.
    """
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
    """Apply configured temporal shifts to columns.

    Rows are first sorted by ``group_cols`` and ``order_cols``. Each column named
    by ``column_specs`` is then shifted within each group using
    :meth:`pandas.core.groupby.SeriesGroupBy.shift`. Positive shifts therefore
    turn a value into a lagged predictor relative to later observations in the
    same pixel or group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing grouping, ordering, and data columns.
    group_cols : sequence of str
        Columns that identify independent time series, typically ``["row",
        "col"]``.
    order_cols : sequence of str
        Columns used to order observations within each group, typically
        ``["year", "month"]``.
    column_specs : sequence of mapping
        Column configuration entries. Each entry must contain ``"name"`` and
        ``"shift"`` keys.

    Returns
    -------
    shifted_df : pandas.DataFrame
        Sorted copy of ``df`` with the requested shifts applied in-place to the
        configured columns.
    labels : list of str
        Names of the configured variables in their input order.
    label_lags : dict of str to int
        Mapping from variable name to integer shift value.

    Raises
    ------
    click.BadParameter
        If a configured variable appears more than once or if a configured
        variable is missing from ``df``.
    KeyError
        If one of ``group_cols`` or ``order_cols`` is missing from ``df``.
    """
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
    """Construct a DirectLiNGAM prior-knowledge matrix from variable lags.

    DirectLiNGAM expects a matrix where ``prior_knowledge[child, parent] == 0``
    means that the parent variable is forbidden from causing the child variable.
    This function forbids less-delayed variables from causing more-delayed
    variables, which helps enforce temporal consistency. It also forbids all
    incoming causal edges into calendar-season variables named ``"month_sin"``
    or ``"month_cos"``.

    Parameters
    ----------
    labels : sequence of str
        Variable names in matrix order.
    label_lags : mapping of str to int
        Mapping from variable name to integer shift value.

    Returns
    -------
    numpy.ndarray
        Integer prior-knowledge matrix with shape ``(n_variables, n_variables)``.
        Entries are ``0`` for forbidden edges and ``-1`` for unknown edges.

    Raises
    ------
    KeyError
        If a label in ``labels`` is not present in ``label_lags``.
    """
    prior_knowledge = -np.ones((len(labels), len(labels)), dtype=int)

    for parent_idx, parent_name in enumerate(labels):
        for child_idx, child_name in enumerate(labels):
            if parent_idx != child_idx and label_lags[parent_name] < label_lags[child_name]:
                prior_knowledge[child_idx, parent_idx] = 0
            if child_name in {"month_sin", "month_cos"}:
                prior_knowledge[child_idx, parent_idx] = 0

    return prior_knowledge


def to_graph(B: np.ndarray, labels: Sequence[str], min_abs_effect: float) -> nx.DiGraph:
    """Convert a LiNGAM adjacency matrix to a directed NetworkX graph.

    LiNGAM uses the convention ``B[child, parent]``. This function converts each
    sufficiently large non-self coefficient into a directed edge from parent to
    child and stores the coefficient as the edge ``weight``.

    Parameters
    ----------
    B : numpy.ndarray
        Square adjacency matrix with shape ``(n_variables, n_variables)``.
    labels : sequence of str
        Variable names corresponding to the rows and columns of ``B``.
    min_abs_effect : float
        Minimum absolute coefficient magnitude required to include an edge.

    Returns
    -------
    networkx.DiGraph
        Directed graph containing all labels as nodes and thresholded causal
        effects as weighted edges.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)

    for child_idx, child_name in enumerate(labels):
        for parent_idx, parent_name in enumerate(labels):
            coefficient = B[child_idx, parent_idx]
            if child_idx != parent_idx and abs(coefficient) >= min_abs_effect:
                graph.add_edge(parent_name, child_name, weight=float(coefficient))

    return graph


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
    """Fit a consensus causal graph for one pixel-centered data group.

    Missing values in the selected variables are dropped before fitting. If the
    remaining sample count is below ``min_samples``, no model is fitted and
    ``None`` is returned. Otherwise, DirectLiNGAM is fitted, bootstrap edge
    probabilities are computed, and a consensus adjacency matrix is produced by
    thresholding both edge probability and absolute effect size.

    Parameters
    ----------
    pixel_key : tuple of int
        Pixel key under which results should be stored. When a spatial window is
        used, this is still the center pixel.
    g : pandas.DataFrame
        Observations used for fitting. This may contain a single pixel or a
        concatenated pixel window.
    labels : sequence of str
        Variable columns used as model inputs.
    pk : numpy.ndarray
        DirectLiNGAM prior-knowledge matrix.
    bootstrap_samples : int
        Number of bootstrap samples passed to ``model.bootstrap``.
    min_samples : int
        Minimum number of complete rows required to fit a model.
    min_prob : float
        Minimum bootstrap edge probability required for consensus edges.
    min_abs_effect : float
        Minimum absolute effect size required for consensus edges.
    group_cols : sequence of str
        Names used to serialize the components of ``pixel_key`` into the output
        row, typically ``["row", "col"]``.

    Returns
    -------
    dict or None
        Result row containing sample count, variable metadata, raw and consensus
        adjacency matrices, bootstrap edge probabilities, causal order, and a GML
        graph. Returns ``None`` if there are fewer than ``min_samples`` complete
        observations.
    """
    X = g[list(labels)].dropna().to_numpy()

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
    raw_adjacency = np.asarray(model.adjacency_matrix_, dtype=float)
    consensus_adjacency = np.where(probabilities >= min_prob, raw_adjacency, 0.0)
    consensus_adjacency = np.where(
        np.abs(consensus_adjacency) >= min_abs_effect,
        consensus_adjacency,
        0.0,
    )

    graph = to_graph(consensus_adjacency, labels, min_abs_effect)
    serialized_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)
    row = dict(zip(group_cols, serialized_pixel_key, strict=False))
    row.update(
        n_samples=int(len(X)),
        variable_names_json=json.dumps(list(labels)),
        variable_index_json=json.dumps({name: idx for idx, name in enumerate(labels)}),
        causal_order_json=json.dumps([int(idx) for idx in model.causal_order_]),
        adjacency_raw_json=json.dumps(raw_adjacency.tolist()),
        edge_probability_json=json.dumps(probabilities.tolist()),
        adjacency_consensus_json=json.dumps(consensus_adjacency.tolist()),
        gml_graph="\n".join(nx.generate_gml(graph)),
    )
    return row


def fit_pixel_task(args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Unpack a multiprocessing task tuple and fit one pixel graph.

    This wrapper exists because :func:`tqdm.contrib.concurrent.process_map`
    expects a single positional argument for each task.

    Parameters
    ----------
    args : tuple
        Task tuple containing ``pixel_key``, data frame, labels,
        prior-knowledge matrix, bootstrap settings, thresholds, and output group
        column names.

    Returns
    -------
    dict or None
        Output of :func:`fit_pixel`.
    """
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
def graph_discovery(
    config_path: str,
    bootstrap_samples: int,
    min_samples: int,
    min_edge_prob: float,
    min_abs_effect: float,
    window_size: int,
    workers: int,
) -> None:
    """Run pixel-wise causal graph discovery from the command line.

    The YAML configuration is expected to live in an experiment directory that
    also contains an input DuckDB database named ``<name>_ard.duckdb``. The input
    table must be named ``<name>`` and must contain ``row``, ``col``, ``year``,
    and ``month`` columns, plus all configured variable columns. Results are
    written to ``<name>_graphs.duckdb`` in a table named ``pixel_graphs``.

    Parameters
    ----------
    config_path : str
        Path to a YAML file containing at least ``name`` and ``columns`` keys.
    bootstrap_samples : int
        Number of bootstrap samples used for edge-probability estimation.
    min_samples : int
        Minimum number of complete observations required per pixel or window.
    min_edge_prob : float
        Minimum bootstrap probability required to keep an edge in the consensus
        graph.
    min_abs_effect : float
        Minimum absolute causal effect required to keep an edge in the consensus
        graph.
    window_size : int
        Pixel-radius of the square fitting window. ``0`` fits only the center
        pixel, ``1`` fits a ``3 x 3`` window, ``2`` fits a ``5 x 5`` window, and
        so on.
    workers : int
        Number of worker processes passed to ``process_map``.

    Raises
    ------
    click.BadParameter
        If the input configuration, input database, required columns, or window
        size are invalid.
    click.ClickException
        If no pixel has enough complete samples to fit a graph.
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
    input_db = experiment_dir / f"{location_nickname}_ard.duckdb"
    output_db = experiment_dir / f"{location_nickname}_graphs.duckdb"
    input_table = location_nickname
    columns = config_data["columns"]

    con = duckdb.connect(input_db, read_only=True)
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if input_table not in tables:
        con.close()
        raise click.BadParameter(
            f"{input_table} not found in {input_db}. Available: {sorted(tables)}"
        )

    df = con.execute(f"SELECT * FROM {input_table}").fetchdf()
    con.close()

    missing_required = [col for col in row_col_cols + order_cols if col not in df.columns]
    if missing_required:
        raise click.BadParameter(f"Missing required columns: {missing_required}")

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

    rows = process_map(
        fit_pixel_task,
        tasks,
        max_workers=workers,
        chunksize=1,
        desc="Pixels",
    )
    rows = [row for row in rows if row is not None]

    if not rows:
        raise click.ClickException("No pixel had enough samples after lagging/dropna.")

    result_df = pd.DataFrame(rows)
    con = duckdb.connect(output_db)
    con.register("result_df", result_df)
    con.execute("CREATE OR REPLACE TABLE pixel_graphs AS SELECT * FROM result_df")
    con.close()


if __name__ == "__main__":
    graph_discovery()
