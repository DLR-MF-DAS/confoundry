"""Shared loading and dynamic-effect helpers for per-pixel VAR-LiNGAM output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import click
import duckdb
import numpy as np
import pandas as pd
import yaml

from confoundry.per_pixel_graph_discovery import (
    graph_config_value,
    has_consecutive_months,
    parse_columns,
    quote_identifier,
    resolve_path,
    varlingam_output_path,
)


@dataclass(frozen=True)
class VARPostprocessConfig:
    """Resolved inputs shared by VAR-LiNGAM post-processing commands."""

    config_path: Path
    experiment_dir: Path
    location_name: str
    columns: list[dict[str, Any]]
    input_db: Path
    input_table: str
    graphs_db: Path
    graphs_table: str
    row_col_cols: list[str]
    order_cols: list[str]
    min_year: int | None
    max_year: int | None
    target: str | None


@dataclass(frozen=True)
class VARPixelBundle:
    """One fitted graph row and its aligned per-pixel time series."""

    key: tuple[Any, ...]
    coords: dict[str, Any]
    time_series: pd.DataFrame
    graph_row: dict[str, Any]


@dataclass(frozen=True)
class VARMatrixSet:
    """Point and paired bootstrap structural VAR matrices."""

    labels: tuple[str, ...]
    contemporaneous: np.ndarray
    lagged: np.ndarray
    bootstrap_contemporaneous: np.ndarray
    bootstrap_lagged: np.ndarray


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise click.BadParameter("YAML config must contain a mapping at top level.")
    return config


def _analysis_value(
    config: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    section = config.get("analysis") or {}
    if not isinstance(section, Mapping):
        raise click.BadParameter("config['analysis'] must be a mapping when present.")
    return section.get(key, config.get(key, default))


def load_var_postprocess_config(
    config_path: Path,
    *,
    input_db: Path | None = None,
    input_table: str | None = None,
    graphs_db: Path | None = None,
    graphs_table: str | None = None,
) -> VARPostprocessConfig:
    """Resolve the residualized time-series and VAR graph inputs."""
    config_path = config_path.expanduser().resolve()
    config = _read_yaml(config_path)
    experiment_dir = config_path.parent
    try:
        location_name = str(config["name"])
        columns = [dict(spec) for spec in config["columns"]]
    except KeyError as exc:
        raise click.BadParameter(f"Missing required config key: {exc.args[0]}") from exc

    resolved_input_db = resolve_path(
        experiment_dir,
        input_db
        or graph_config_value(config, "input_db")
        or graph_config_value(config, "timeseries_db"),
        experiment_dir / f"{location_name}_ard.duckdb",
    )
    resolved_input_table = str(
        input_table
        or graph_config_value(config, "input_table")
        or graph_config_value(config, "timeseries_table")
        or location_name
    )

    if graphs_db is not None:
        resolved_graphs_db = resolve_path(
            experiment_dir,
            graphs_db,
            experiment_dir / f"{location_name}_varlingam_graphs.duckdb",
        )
    else:
        configured_var_db = graph_config_value(config, "var_output_db")
        configured_graph_db = (
            configured_var_db
            or graph_config_value(config, "output_db")
            or graph_config_value(config, "graph_db")
        )
        resolved_graphs_db = resolve_path(
            experiment_dir,
            configured_graph_db,
            experiment_dir / f"{location_name}_graphs.duckdb",
        )
        if configured_var_db is None:
            resolved_graphs_db = varlingam_output_path(resolved_graphs_db)

    row_col_cols = list(_analysis_value(config, "row_col_cols", ["row", "col"]))
    order_cols = list(_analysis_value(config, "order_cols", ["year", "month"]))
    if order_cols != ["year", "month"]:
        raise click.BadParameter(
            "VAR-LiNGAM post-processing currently requires order_cols "
            "['year', 'month']."
        )

    target = (
        _analysis_value(config, "target")
        or _analysis_value(config, "outcome")
        or config.get("reference_var")
    )
    return VARPostprocessConfig(
        config_path=config_path,
        experiment_dir=experiment_dir,
        location_name=location_name,
        columns=columns,
        input_db=resolved_input_db,
        input_table=resolved_input_table,
        graphs_db=resolved_graphs_db,
        graphs_table=str(
            graphs_table
            or graph_config_value(config, "output_table")
            or _analysis_value(config, "graph_table", "pixel_graphs")
        ),
        row_col_cols=row_col_cols,
        order_cols=order_cols,
        min_year=graph_config_value(config, "min_year"),
        max_year=graph_config_value(config, "max_year"),
        target=str(target) if target is not None else None,
    )


def _read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    if not db_path.exists():
        raise click.ClickException(f"Required DuckDB file does not exist: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table_name not in tables:
            raise click.ClickException(
                f"{table_name!r} not found in {db_path}. "
                f"Available tables: {sorted(tables)}"
            )
        return con.execute(
            f"SELECT * FROM {quote_identifier(table_name)}"
        ).fetchdf()
    finally:
        con.close()


def load_var_timeseries_and_graphs(
    config: VARPostprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load aligned shift-zero time series and validated VAR graph rows."""
    graph_df = _read_table(config.graphs_db, config.graphs_table)
    required_graph_columns = {
        *config.row_col_cols,
        "model_type",
        "variable_names_json",
        "adjacency_raw_json",
        "adjacency_consensus_json",
        "adjacency_lagged_raw_json",
        "adjacency_lagged_consensus_json",
        "adjacency_bootstrap_json",
        "adjacency_bootstrap_lagged_json",
    }
    missing_graph = sorted(required_graph_columns - set(graph_df.columns))
    if missing_graph:
        raise click.ClickException(
            f"VAR graph table is missing required columns: {missing_graph}"
        )
    model_types = {
        str(value).lower()
        for value in graph_df["model_type"].dropna().unique()
    }
    if model_types != {"varlingam"}:
        raise click.ClickException(
            "Expected a VAR-LiNGAM graph table, found model types: "
            f"{sorted(model_types)}"
        )
    if graph_df.duplicated(config.row_col_cols).any():
        raise click.ClickException(
            "VAR graph table contains duplicate pixel keys."
        )

    time_series = _read_table(config.input_db, config.input_table)
    missing_time = [
        column
        for column in config.row_col_cols + config.order_cols
        if column not in time_series.columns
    ]
    if missing_time:
        raise click.ClickException(
            f"Time-series table is missing required columns: {missing_time}"
        )
    if config.min_year is not None:
        time_series = time_series[
            time_series["year"].astype(int) >= int(config.min_year)
        ].copy()
    if config.max_year is not None:
        time_series = time_series[
            time_series["year"].astype(int) <= int(config.max_year)
        ].copy()

    aligned, labels, _ = parse_columns(
        time_series,
        group_cols=config.row_col_cols,
        order_cols=config.order_cols,
        column_specs=config.columns,
        apply_shifts=False,
    )
    aligned = aligned.dropna(
        subset=list(
            dict.fromkeys(
                labels + config.row_col_cols + config.order_cols
            )
        )
    )
    return aligned, graph_df, labels


def iter_var_pixel_bundles(
    config: VARPostprocessConfig,
    time_series: pd.DataFrame,
    graph_df: pd.DataFrame,
) -> Iterator[VARPixelBundle]:
    """Yield graph/time-series pairs in stable pixel order."""
    graph_index = graph_df.set_index(config.row_col_cols, drop=False)
    graph_keys = graph_df[config.row_col_cols].drop_duplicates()
    selected = time_series.merge(
        graph_keys,
        on=config.row_col_cols,
        how="inner",
    )
    selected = selected.sort_values(
        config.row_col_cols + config.order_cols
    )
    for key, group in selected.groupby(config.row_col_cols, sort=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        graph_row = graph_index.loc[normalized_key].to_dict()
        yield VARPixelBundle(
            key=normalized_key,
            coords=dict(
                zip(config.row_col_cols, normalized_key, strict=False)
            ),
            time_series=group.reset_index(drop=True),
            graph_row=graph_row,
        )


def validated_pixel_observations(
    bundle: VARPixelBundle,
    labels: Sequence[str],
    *,
    min_samples: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a complete, uninterrupted monthly pixel series."""
    complete = bundle.time_series.dropna(subset=list(labels)).copy()
    if len(complete) < min_samples:
        raise ValueError(
            f"Only {len(complete)} complete observations; "
            f"minimum is {min_samples}"
        )
    if not has_consecutive_months(complete, ["year", "month"]):
        raise ValueError(
            "Complete observations do not form a unique, uninterrupted "
            "monthly sequence"
        )
    return complete, complete[list(labels)].to_numpy(dtype=float)


def parse_json_array(value: Any, field_name: str) -> np.ndarray:
    """Parse a JSON array field with a useful user-facing error."""
    if value is None or (
        isinstance(value, float) and pd.isna(value)
    ):
        raise ValueError(f"{field_name} is missing")
    parsed = json.loads(value) if isinstance(value, str) else value
    try:
        return np.asarray(parsed, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not a numeric JSON array") from exc


def _validated_matrix_shapes(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    *,
    n_features: int,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    if contemporaneous.shape != (n_features, n_features):
        raise ValueError(
            f"{prefix} contemporaneous matrix has shape "
            f"{contemporaneous.shape}, expected {(n_features, n_features)}"
        )
    if (
        lagged.ndim != 3
        or lagged.shape[0] < 1
        or lagged.shape[1:] != (n_features, n_features)
    ):
        raise ValueError(
            f"{prefix} lagged matrices have shape {lagged.shape}, expected "
            f"(lags, {n_features}, {n_features})"
        )
    return contemporaneous, lagged


def matrices_from_graph_row(
    graph_row: Mapping[str, Any],
    *,
    point_matrix: str = "raw",
    bootstrap_limit: int = 0,
) -> VARMatrixSet:
    """Decode point and paired bootstrap VAR matrices from one graph row."""
    labels_value = graph_row["variable_names_json"]
    labels = tuple(
        str(value)
        for value in (
            json.loads(labels_value)
            if isinstance(labels_value, str)
            else labels_value
        )
    )
    n_features = len(labels)

    bootstrap_b0 = parse_json_array(
        graph_row["adjacency_bootstrap_json"],
        "adjacency_bootstrap_json",
    )
    bootstrap_lagged = parse_json_array(
        graph_row["adjacency_bootstrap_lagged_json"],
        "adjacency_bootstrap_lagged_json",
    )
    if (
        bootstrap_b0.ndim != 3
        or bootstrap_b0.shape[1:] != (n_features, n_features)
    ):
        raise ValueError(
            "adjacency_bootstrap_json must have shape "
            f"(bootstrap, {n_features}, {n_features}), got "
            f"{bootstrap_b0.shape}"
        )
    if (
        bootstrap_lagged.ndim != 4
        or bootstrap_lagged.shape[0] != bootstrap_b0.shape[0]
        or bootstrap_lagged.shape[2:] != (n_features, n_features)
        or bootstrap_lagged.shape[1] < 1
    ):
        raise ValueError(
            "adjacency_bootstrap_lagged_json must have shape "
            f"(bootstrap, lags, {n_features}, {n_features}), got "
            f"{bootstrap_lagged.shape}"
        )
    if bootstrap_limit > 0:
        bootstrap_b0 = bootstrap_b0[:bootstrap_limit]
        bootstrap_lagged = bootstrap_lagged[:bootstrap_limit]

    if point_matrix == "raw":
        point_b0 = parse_json_array(
            graph_row["adjacency_raw_json"],
            "adjacency_raw_json",
        )
        point_lagged = parse_json_array(
            graph_row["adjacency_lagged_raw_json"],
            "adjacency_lagged_raw_json",
        )
    elif point_matrix == "consensus":
        point_b0 = parse_json_array(
            graph_row["adjacency_consensus_json"],
            "adjacency_consensus_json",
        )
        point_lagged = parse_json_array(
            graph_row["adjacency_lagged_consensus_json"],
            "adjacency_lagged_consensus_json",
        )
    elif point_matrix == "bootstrap_mean":
        if len(bootstrap_b0) == 0:
            raise ValueError(
                "bootstrap_mean requires saved bootstrap matrices"
            )
        point_b0 = np.nanmean(bootstrap_b0, axis=0)
        point_lagged = np.nanmean(bootstrap_lagged, axis=0)
    else:
        raise ValueError(f"Unknown point matrix: {point_matrix!r}")

    point_b0, point_lagged = _validated_matrix_shapes(
        point_b0,
        point_lagged,
        n_features=n_features,
        prefix=point_matrix,
    )
    if point_lagged.shape[0] != bootstrap_lagged.shape[1]:
        raise ValueError(
            "Point and bootstrap VAR lag counts do not match: "
            f"{point_lagged.shape[0]} versus {bootstrap_lagged.shape[1]}"
        )
    return VARMatrixSet(
        labels=labels,
        contemporaneous=point_b0,
        lagged=point_lagged,
        bootstrap_contemporaneous=bootstrap_b0,
        bootstrap_lagged=bootstrap_lagged,
    )


def reduced_form_matrices(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert structural matrices B0..Bp to C and reduced-form A1..Ap."""
    n_features = contemporaneous.shape[0]
    contemporaneous_multiplier = np.linalg.inv(
        np.eye(n_features, dtype=float) - contemporaneous
    )
    reduced_lagged = np.einsum(
        "ij,pjk->pik",
        contemporaneous_multiplier,
        lagged,
    )
    return contemporaneous_multiplier, reduced_lagged


def companion_matrix(reduced_lagged: np.ndarray) -> np.ndarray:
    """Build the first-order companion matrix of a reduced-form VAR(p)."""
    n_lags, n_features, _ = reduced_lagged.shape
    companion = np.zeros(
        (n_lags * n_features, n_lags * n_features),
        dtype=float,
    )
    companion[:n_features, :] = np.concatenate(
        list(reduced_lagged),
        axis=1,
    )
    if n_lags > 1:
        companion[n_features:, :-n_features] = np.eye(
            (n_lags - 1) * n_features,
            dtype=float,
        )
    return companion


def stability_radius(reduced_lagged: np.ndarray) -> float:
    """Return the maximum companion-matrix eigenvalue magnitude."""
    eigenvalues = np.linalg.eigvals(companion_matrix(reduced_lagged))
    return float(np.max(np.abs(eigenvalues)))


def dynamic_effect_matrices(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return impulse/total effects, cumulative effects, and stability radius."""
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    multiplier, reduced_lagged = reduced_form_matrices(
        contemporaneous,
        lagged,
    )
    effects = np.zeros(
        (horizon + 1, *contemporaneous.shape),
        dtype=float,
    )
    effects[0] = multiplier
    for step in range(1, horizon + 1):
        for lag, matrix in enumerate(reduced_lagged, start=1):
            if step - lag >= 0:
                effects[step] += matrix @ effects[step - lag]
    cumulative = np.cumsum(effects, axis=0)
    return effects, cumulative, stability_radius(reduced_lagged)


def bootstrap_dynamic_effect_matrices(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize dynamic effects over paired bootstrap VAR matrices.

    Returns effects, cumulative effects, stability radii, and a validity mask.
    Singular or non-finite replicates are marked invalid.
    """
    n_bootstrap = contemporaneous.shape[0]
    n_features = contemporaneous.shape[1]
    effects = np.full(
        (n_bootstrap, horizon + 1, n_features, n_features),
        np.nan,
        dtype=float,
    )
    cumulative = np.full_like(effects, np.nan)
    radii = np.full(n_bootstrap, np.nan, dtype=float)
    valid = np.zeros(n_bootstrap, dtype=bool)

    for index in range(n_bootstrap):
        try:
            result, result_cumulative, radius = dynamic_effect_matrices(
                contemporaneous[index],
                lagged[index],
                horizon,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        if (
            np.all(np.isfinite(result))
            and np.all(np.isfinite(result_cumulative))
            and np.isfinite(radius)
        ):
            effects[index] = result
            cumulative[index] = result_cumulative
            radii[index] = radius
            valid[index] = True
    return effects, cumulative, radii, valid


def summarize_bootstrap(
    values: Sequence[float] | np.ndarray,
    *,
    ci: float,
) -> dict[str, Any]:
    """Summarize a one-dimensional bootstrap distribution."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "boot_mean": np.nan,
            "boot_median": np.nan,
            "boot_sd": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_ci_width": np.nan,
            "boot_prob_gt_zero": np.nan,
            "boot_prob_lt_zero": np.nan,
            "boot_ci_excludes_zero": False,
            "n_bootstrap_successful": 0,
        }
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    return {
        "boot_mean": float(np.mean(array)),
        "boot_median": float(np.median(array)),
        "boot_sd": float(np.std(array, ddof=1))
        if len(array) > 1
        else np.nan,
        "boot_ci_low": float(lower),
        "boot_ci_high": float(upper),
        "boot_ci_width": float(upper - lower),
        "boot_prob_gt_zero": float(np.mean(array > 0.0)),
        "boot_prob_lt_zero": float(np.mean(array < 0.0)),
        "boot_ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "n_bootstrap_successful": int(len(array)),
    }


def quantile_contrast(
    values: Sequence[float] | np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float, float, float]:
    """Return low, high, and high-minus-low finite quantiles."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    low, high = np.quantile(array, [low_quantile, high_quantile])
    return float(low), float(high), float(high - low)


def infer_structural_innovations(
    observations: np.ndarray,
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
) -> np.ndarray:
    """Infer structural innovations e_t from one factual trajectory."""
    n_lags = lagged.shape[0]
    if len(observations) <= n_lags:
        raise ValueError(
            f"Need more than {n_lags} observations to infer innovations"
        )
    innovations = np.full_like(observations, np.nan, dtype=float)
    structural = np.eye(observations.shape[1]) - contemporaneous
    for time in range(n_lags, len(observations)):
        value = structural @ observations[time]
        for lag, matrix in enumerate(lagged, start=1):
            value -= matrix @ observations[time - lag]
        innovations[time] = value
    return innovations


def solve_structural_step(
    contemporaneous: np.ndarray,
    lagged_term: np.ndarray,
    innovation: np.ndarray,
    interventions: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Solve one structural VAR step, optionally replacing selected equations."""
    interventions = dict(interventions or {})
    n_features = contemporaneous.shape[0]
    if not interventions:
        return np.linalg.solve(
            np.eye(n_features) - contemporaneous,
            lagged_term + innovation,
        )

    intervention_indices = sorted(interventions)
    if any(index < 0 or index >= n_features for index in intervention_indices):
        raise ValueError("Intervention variable index is out of range")
    remaining = [
        index
        for index in range(n_features)
        if index not in interventions
    ]
    result = np.empty(n_features, dtype=float)
    result[intervention_indices] = [
        interventions[index] for index in intervention_indices
    ]
    if remaining:
        b_rr = contemporaneous[np.ix_(remaining, remaining)]
        b_rj = contemporaneous[np.ix_(remaining, intervention_indices)]
        rhs = (
            lagged_term[remaining]
            + innovation[remaining]
            + b_rj @ result[intervention_indices]
        )
        result[remaining] = np.linalg.solve(
            np.eye(len(remaining)) - b_rr,
            rhs,
        )
    return result


def simulate_structural_var(
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    initial_history: np.ndarray,
    innovations: np.ndarray,
    interventions: Sequence[Mapping[int, float]] | None = None,
) -> np.ndarray:
    """Simulate a structural VAR path for a supplied innovation sequence."""
    n_lags, n_features, _ = lagged.shape
    history = np.asarray(initial_history, dtype=float)
    if history.shape != (n_lags, n_features):
        raise ValueError(
            f"initial_history must have shape {(n_lags, n_features)}, "
            f"got {history.shape}"
        )
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim != 2 or innovations.shape[1] != n_features:
        raise ValueError(
            f"innovations must have shape (steps, {n_features})"
        )
    if interventions is None:
        interventions = [{} for _ in range(len(innovations))]
    if len(interventions) != len(innovations):
        raise ValueError("interventions and innovations must have equal length")

    past = [row.copy() for row in history]
    output = np.empty_like(innovations, dtype=float)
    for step, innovation in enumerate(innovations):
        lagged_term = np.zeros(n_features, dtype=float)
        for lag, matrix in enumerate(lagged, start=1):
            lagged_term += matrix @ past[-lag]
        current = solve_structural_step(
            contemporaneous,
            lagged_term,
            innovation,
            interventions[step],
        )
        output[step] = current
        past.append(current)
    return output
