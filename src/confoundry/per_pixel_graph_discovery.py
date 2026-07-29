"""Discover causal graphs for individual pixels or pixel neighborhoods.

This command reads pixel-wise time-series data from a DuckDB database and fits
either DirectLiNGAM or, as an opt-in time-series alternative, VAR-LiNGAM.
DirectLiNGAM retains the existing configured-shift and spatial-window workflow.
VAR-LiNGAM keeps variables aligned at the same month and estimates lagged plus
contemporaneous effects for individual pixels.

Statistics and LiNGAM suitability diagnostics are intentionally handled by
``per_pixel_graph_diagnostics.py`` so graph discovery and post-hoc evaluation
can be run as separate steps.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
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


def varlingam_output_path(output_db: Path) -> Path:
    """Return a non-colliding VAR-LiNGAM output path."""
    stem = output_db.stem
    if stem.endswith("_varlingam_graphs"):
        return output_db
    if stem.endswith("_graphs"):
        stem = f"{stem[:-len('_graphs')]}_varlingam_graphs"
    else:
        stem = f"{stem}_varlingam"
    return output_db.with_name(f"{stem}{output_db.suffix}")


def has_consecutive_months(
    frame: pd.DataFrame,
    order_cols: Sequence[str],
) -> bool:
    """Return whether rows form one unique, uninterrupted monthly sequence."""
    if list(order_cols) != ["year", "month"]:
        raise ValueError("VAR-LiNGAM currently requires order columns ['year', 'month'].")
    month_index = (
        frame["year"].astype(int).to_numpy() * 12
        + frame["month"].astype(int).to_numpy()
    )
    return bool(
        len(month_index) > 0
        and len(np.unique(month_index)) == len(month_index)
        and (len(month_index) == 1 or np.all(np.diff(month_index) == 1))
    )


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


def threshold_adjacency(
    raw_adjacency: np.ndarray,
    probabilities: np.ndarray,
    min_prob: float,
    min_abs_effect: float,
) -> np.ndarray:
    """Threshold one adjacency matrix by bootstrap support and effect size."""
    consensus = np.where(probabilities >= min_prob, raw_adjacency, 0.0)
    return np.where(np.abs(consensus) >= min_abs_effect, consensus, 0.0)


def direct_bootstrap_adjacencies(boot: Any, n_features: int) -> np.ndarray:
    """Return DirectLiNGAM bootstrap matrices when supplied by the package."""
    value = getattr(boot, "adjacency_matrices_", None)
    if value is None:
        return np.empty((0, n_features, n_features), dtype=float)
    matrices = np.asarray(value, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (n_features, n_features):
        raise click.ClickException(
            "Unexpected DirectLiNGAM bootstrap adjacency shape: "
            f"{matrices.shape}."
        )
    return matrices


def var_bootstrap_adjacencies(
    boot: Any,
    n_features: int,
    n_adjacency_matrices: int,
) -> np.ndarray:
    """Return VAR-LiNGAM bootstrap matrices as bootstrap × lag × child × parent."""
    value = getattr(boot, "adjacency_matrices_", None)
    if value is None:
        return np.empty(
            (0, n_adjacency_matrices, n_features, n_features),
            dtype=float,
        )

    matrices = np.asarray(value, dtype=float)
    if matrices.ndim == 3 and matrices.shape[1:] == (
        n_features,
        n_features * n_adjacency_matrices,
    ):
        return matrices.reshape(
            matrices.shape[0],
            n_features,
            n_adjacency_matrices,
            n_features,
        ).transpose(0, 2, 1, 3)
    if matrices.ndim == 4 and matrices.shape[1:] == (
        n_adjacency_matrices,
        n_features,
        n_features,
    ):
        return matrices
    raise click.ClickException(
        f"Unexpected VAR-LiNGAM bootstrap adjacency shape: {matrices.shape}."
    )


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
    order_cols: Sequence[str] = ("year", "month"),
    model_type: str = "directlingam",
    var_lags: int = 1,
    var_criterion: str | None = "bic",
    var_prune: bool = True,
) -> dict[str, Any] | None:
    """Fit one DirectLiNGAM or VAR-LiNGAM graph."""
    model_type = str(model_type).lower()
    complete_g = g.dropna(subset=list(labels)).copy()
    if set(order_cols).issubset(complete_g.columns):
        complete_g = complete_g.sort_values(list(order_cols))
    elif model_type == "varlingam":
        raise click.ClickException(
            "VAR-LiNGAM input is missing temporal ordering columns: "
            + ", ".join(order_cols)
        )
    X = complete_g[list(labels)].to_numpy()

    required_samples = min_samples + (var_lags if model_type == "varlingam" else 0)
    if len(X) < required_samples:
        return None

    if model_type == "directlingam":
        model = lingam.DirectLiNGAM(
            prior_knowledge=pk,
            random_state=0,
        )
        model.fit(X)
        causal_order = [int(idx) for idx in model.causal_order_]
        boot = model.bootstrap(X, n_sampling=bootstrap_samples)
        probabilities = np.asarray(
            boot.get_probabilities(min_causal_effect=min_abs_effect),
            dtype=float,
        )
        raw_adjacency = np.asarray(model.adjacency_matrix_, dtype=float)
        bootstrap_adjacencies = direct_bootstrap_adjacencies(
            boot,
            n_features=len(labels),
        )
        selected_lags = 0
        lagged_raw = np.empty((0, len(labels), len(labels)), dtype=float)
        lagged_probabilities = lagged_raw.copy()
        lagged_consensus = lagged_raw.copy()
        lagged_bootstrap = np.empty(
            (len(bootstrap_adjacencies), 0, len(labels), len(labels)),
            dtype=float,
        )
        effective_samples = len(X)
    elif model_type == "varlingam":
        if not has_consecutive_months(complete_g, order_cols):
            return None
        instantaneous_model = lingam.DirectLiNGAM(
            prior_knowledge=pk,
            random_state=0,
        )
        model = lingam.VARLiNGAM(
            lags=var_lags,
            criterion=var_criterion,
            prune=var_prune,
            lingam_model=instantaneous_model,
            random_state=0,
        )
        model.fit(X)
        causal_order = [int(idx) for idx in model.causal_order_]
        adjacency_matrices = np.asarray(model.adjacency_matrices_, dtype=float)
        if adjacency_matrices.ndim != 3 or adjacency_matrices.shape[1:] != (
            len(labels),
            len(labels),
        ):
            raise click.ClickException(
                "Unexpected VAR-LiNGAM adjacency shape: "
                f"{adjacency_matrices.shape}."
            )
        selected_lags = int(adjacency_matrices.shape[0] - 1)
        if selected_lags < 1:
            raise click.ClickException(
                "VAR-LiNGAM selected no autoregressive lag; use DirectLiNGAM "
                "for an instantaneous-only model."
            )
        boot = model.bootstrap(X, n_sampling=bootstrap_samples)
        probability_matrices = np.asarray(
            boot.get_probabilities(min_causal_effect=min_abs_effect),
            dtype=float,
        )
        if probability_matrices.shape != adjacency_matrices.shape:
            raise click.ClickException(
                "VAR-LiNGAM bootstrap probabilities do not match adjacency "
                f"matrices: {probability_matrices.shape} versus "
                f"{adjacency_matrices.shape}."
            )
        all_bootstrap = var_bootstrap_adjacencies(
            boot,
            n_features=len(labels),
            n_adjacency_matrices=len(adjacency_matrices),
        )
        raw_adjacency = adjacency_matrices[0]
        probabilities = probability_matrices[0]
        bootstrap_adjacencies = all_bootstrap[:, 0]
        lagged_raw = adjacency_matrices[1:]
        lagged_probabilities = probability_matrices[1:]
        lagged_consensus = np.asarray(
            [
                threshold_adjacency(raw, probability, min_prob, min_abs_effect)
                for raw, probability in zip(
                    lagged_raw,
                    lagged_probabilities,
                    strict=True,
                )
            ]
        )
        lagged_bootstrap = all_bootstrap[:, 1:]
        effective_samples = len(X) - selected_lags
    else:
        raise click.BadParameter(f"Unsupported model: {model_type!r}")

    consensus_adjacency = threshold_adjacency(
        raw_adjacency,
        probabilities,
        min_prob,
        min_abs_effect,
    )

    graph = to_graph(consensus_adjacency, labels, min_abs_effect)
    serialized_pixel_key = pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)
    graph_row = dict(zip(group_cols, serialized_pixel_key, strict=False))
    graph_row.update(
        model_type=model_type,
        n_samples=int(len(X)),
        n_effective_samples=int(effective_samples),
        variable_names_json=json.dumps(list(labels)),
        variable_index_json=json.dumps({name: idx for idx, name in enumerate(labels)}),
        causal_order_json=json.dumps(causal_order),
        adjacency_raw_json=json.dumps(raw_adjacency.tolist()),
        edge_probability_json=json.dumps(probabilities.tolist()),
        adjacency_consensus_json=json.dumps(consensus_adjacency.tolist()),
        adjacency_bootstrap_json=json.dumps(bootstrap_adjacencies.tolist()),
        var_lags=int(selected_lags),
        var_lags_requested=int(var_lags) if model_type == "varlingam" else 0,
        var_criterion=var_criterion if model_type == "varlingam" else None,
        var_prune=bool(var_prune) if model_type == "varlingam" else None,
        adjacency_lagged_raw_json=json.dumps(lagged_raw.tolist()),
        edge_probability_lagged_json=json.dumps(lagged_probabilities.tolist()),
        adjacency_lagged_consensus_json=json.dumps(lagged_consensus.tolist()),
        adjacency_bootstrap_lagged_json=json.dumps(lagged_bootstrap.tolist()),
        gml_graph="\n".join(nx.generate_gml(graph)),
    )

    return graph_row


def fit_pixel_task(args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Unpack a multiprocessing task tuple and fit one pixel graph."""
    if len(args) == 9:
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
    else:
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
            order_cols,
            model_type,
            var_lags,
            var_criterion,
            var_prune,
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
        order_cols=order_cols,
        model_type=model_type,
        var_lags=var_lags,
        var_criterion=var_criterion,
        var_prune=var_prune,
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
@click.option(
    "--model",
    "model_type",
    default=None,
    type=click.Choice(["directlingam", "varlingam"], case_sensitive=False),
    help="Causal-discovery model. Defaults to graph_discovery.model or directlingam.",
)
@click.option(
    "--var-lags",
    default=None,
    type=click.IntRange(min=1),
    help="Maximum VAR lag order. Used only with --model varlingam.",
)
@click.option(
    "--var-criterion",
    default=None,
    type=click.Choice(
        ["none", "aic", "fpe", "hqic", "bic"],
        case_sensitive=False,
    ),
    help="Lag-order criterion; 'none' forces exactly --var-lags.",
)
@click.option(
    "--var-prune/--no-var-prune",
    default=None,
    help="Enable or disable VAR-LiNGAM coefficient pruning.",
)
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
    model_type: str | None,
    var_lags: int | None,
    var_criterion: str | None,
    var_prune: bool | None,
) -> None:
    """Run pixel-wise causal graph discovery from the command line.

    The input table must contain ``row``, ``col``, ``year``, and ``month``
    columns plus all configured variables. DirectLiNGAM writes to the existing
    graph path. VAR-LiNGAM defaults to a separate ``*_varlingam_graphs.duckdb``
    path and stores lagged matrices alongside the compatible contemporaneous
    fields.

    Run ``per_pixel_graph_diagnostics.py`` afterwards to compute diagnostics
    and statistics from the saved graph table.
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
    model_type = str(
        model_type
        or graph_config_value(config_data, "model")
        or "directlingam"
    ).lower()
    if model_type not in {"directlingam", "varlingam"}:
        raise click.BadParameter(
            "graph_discovery.model must be 'directlingam' or 'varlingam'."
        )
    var_lags = int(
        var_lags
        if var_lags is not None
        else graph_config_value(config_data, "var_lags", 1)
    )
    if var_lags < 1:
        raise click.BadParameter("var-lags must be >= 1")
    configured_criterion = (
        var_criterion
        if var_criterion is not None
        else graph_config_value(config_data, "var_criterion", "bic")
    )
    criterion_name = str(configured_criterion).lower()
    if criterion_name not in {"none", "aic", "fpe", "hqic", "bic"}:
        raise click.BadParameter(
            "var-criterion must be one of: none, aic, fpe, hqic, bic."
        )
    var_criterion_value = None if criterion_name == "none" else criterion_name
    if var_prune is None:
        var_prune = bool(graph_config_value(config_data, "var_prune", True))
    if model_type == "varlingam" and window_size != 0:
        raise click.BadParameter(
            "VAR-LiNGAM currently requires --window-size 0 so separate pixel "
            "time series are not concatenated into false temporal transitions."
        )

    input_db = resolve_path(
        experiment_dir,
        input_db or graph_config_value(config_data, "input_db")
        or graph_config_value(config_data, "timeseries_db"),
        experiment_dir / f"{location_nickname}_ard.duckdb",
    )
    explicit_output_db = output_db
    configured_output_db = (
        graph_config_value(config_data, "var_output_db")
        if model_type == "varlingam"
        else None
    )
    configured_output_db = (
        configured_output_db
        or graph_config_value(config_data, "output_db")
        or graph_config_value(config_data, "graph_db")
    )
    output_db = resolve_path(
        experiment_dir,
        explicit_output_db or configured_output_db,
        experiment_dir / f"{location_nickname}_graphs.duckdb",
    )
    if (
        model_type == "varlingam"
        and explicit_output_db is None
        and graph_config_value(config_data, "var_output_db") is None
    ):
        output_db = varlingam_output_path(output_db)
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

    configured_nonzero_shifts = [
        str(spec["name"])
        for spec in columns
        if int(spec.get("shift", 0)) != 0
    ]
    apply_configured_shifts = model_type == "directlingam"
    if model_type == "varlingam" and configured_nonzero_shifts:
        click.echo(
            "VAR-LiNGAM keeps variables aligned at shift 0 and ignores configured "
            "manual shifts for: "
            + ", ".join(configured_nonzero_shifts)
        )
    df, labels, label_lags = parse_columns(
        df,
        row_col_cols,
        order_cols,
        columns,
        apply_shifts=apply_configured_shifts,
    )
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
                order_cols,
                model_type,
                var_lags,
                var_criterion_value,
                var_prune,
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
        metadata = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_path": str(config_path_obj),
                    "input_db": str(input_db),
                    "input_table": str(input_table),
                    "output_db": str(output_db),
                    "model_type": model_type,
                    "configured_shifts_applied": bool(apply_configured_shifts),
                    "var_lags_requested": int(var_lags)
                    if model_type == "varlingam"
                    else 0,
                    "var_criterion": criterion_name
                    if model_type == "varlingam"
                    else None,
                    "var_prune": bool(var_prune)
                    if model_type == "varlingam"
                    else None,
                    "bootstrap_samples": int(bootstrap_samples),
                    "min_samples": int(min_samples),
                    "min_edge_prob": float(min_edge_prob),
                    "min_abs_effect": float(min_abs_effect),
                    "window_size": int(window_size),
                    "min_year": min_year,
                    "max_year": max_year,
                    "n_graph_rows": int(len(result_df)),
                    "n_skipped_pixel_tasks": int(len(tasks) - len(result_df)),
                    "variable_names_json": json.dumps(list(labels)),
                }
            ]
        )
        write_dataframe_table(
            con,
            metadata,
            "graph_discovery_run_metadata",
        )
    finally:
        con.close()

    click.echo(f"Model: {model_type}")
    if model_type == "varlingam":
        selected_counts = result_df["var_lags"].value_counts().sort_index()
        click.echo(
            "Selected VAR lag counts: "
            + ", ".join(
                f"{int(lag)}={int(count)}"
                for lag, count in selected_counts.items()
            )
        )
    skipped = len(tasks) - len(result_df)
    if skipped:
        click.echo(
            f"Skipped pixel tasks: {skipped} "
            "(insufficient complete samples or non-consecutive months)"
        )
    click.echo(f"Wrote graphs: {output_db}::pixel_graphs")


if __name__ == "__main__":
    graph_discovery()
