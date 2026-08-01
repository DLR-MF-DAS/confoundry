"""Residualize ARD time series before causal graph discovery.

This command removes deterministic seasonality and optional long-term trend
from configured environmental variables. The default model uses calendar-month
fixed effects, which allow an arbitrary repeating annual shape instead of
assuming that every variable follows a single sinusoid. A legacy annual-
harmonic model remains available for reproducibility.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from confoundry.analysis_helpers import (
    ensure_identifier,
    require_files,
    write_dataframe_table,
)


SEASONAL_COLUMNS = {"month_sin", "month_cos"}
SEASONAL_MODELS = ("monthly-fixed-effects", "annual-harmonic")
MONTH_EFFECT_NAMES = tuple(f"month_{month:02d}" for month in range(2, 13))


def read_config(config_path: Path) -> dict[str, Any]:
    """Read an experiment YAML file."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise click.ClickException("Experiment YAML must contain a mapping.")
    if "name" not in config or "columns" not in config:
        raise click.ClickException("Config must contain 'name' and 'columns'.")
    return config


def resolve_path(base_dir: Path, value: str | Path | None, default: Path) -> Path:
    """Resolve a possibly relative path against the experiment directory."""
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


def path_for_config(config_dir: Path, path: Path) -> str:
    """Return a stable path string for writing into a generated config."""
    try:
        return str(path.resolve().relative_to(config_dir.resolve()))
    except ValueError:
        return str(path)


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    """Read a table from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table not in tables:
            raise click.ClickException(
                f"{table!r} not found in {db_path}. Available: {sorted(tables)}"
            )
        return con.execute(f"SELECT * FROM {ensure_identifier(table)}").fetchdf()
    finally:
        con.close()


def configured_variables(
    config: Mapping[str, Any],
    requested_variables: Sequence[str],
) -> list[str]:
    """Return variables to residualize."""
    configured = [
        str(spec["name"])
        for spec in config["columns"]
        if str(spec["name"]) not in SEASONAL_COLUMNS
    ]
    if requested_variables:
        requested = [str(variable) for variable in requested_variables]
        missing = [variable for variable in requested if variable not in configured]
        if missing:
            raise click.ClickException(
                "Requested residualization variables are not configured: "
                + ", ".join(missing)
            )
        return requested
    return configured


def design_columns(
    include_trend: bool,
    seasonal_model: str = "monthly-fixed-effects",
) -> list[str]:
    """Return named residualization terms, excluding the intercept."""
    if seasonal_model == "monthly-fixed-effects":
        columns = list(MONTH_EFFECT_NAMES)
    elif seasonal_model == "annual-harmonic":
        columns = ["month_sin", "month_cos"]
    else:
        raise ValueError(f"Unsupported seasonal model: {seasonal_model!r}")
    if include_trend:
        columns.append("time_trend_per_year")
    return columns


def design_matrix(
    frame: pd.DataFrame,
    seasonal_model: str,
    include_trend: bool,
    time_center: float,
) -> np.ndarray:
    """Build a stable intercept-plus-seasonality/trend design matrix."""
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    if seasonal_model == "monthly-fixed-effects":
        months = frame["month"].astype(int).to_numpy()
        columns.extend(
            (months == month).astype(float)
            for month in range(2, 13)
        )
    elif seasonal_model == "annual-harmonic":
        columns.extend(
            [
                frame["month_sin"].astype(float).to_numpy(),
                frame["month_cos"].astype(float).to_numpy(),
            ]
        )
    else:
        raise ValueError(f"Unsupported seasonal model: {seasonal_model!r}")

    if include_trend:
        time_years = (
            frame["_residual_time_index"].astype(float).to_numpy()
            - float(time_center)
        ) / 12.0
        columns.append(time_years)
    return np.column_stack(columns)


def residualize_group(
    result: pd.DataFrame,
    row_index: pd.Index,
    variables: Sequence[str],
    fit_end_year: int | None,
    min_fit_samples: int,
    min_month_samples: int,
    suffix: str,
    expected_suffix: str,
    include_trend: bool,
    seasonal_model: str,
) -> list[dict[str, Any]]:
    """Residualize variables for one pixel group in-place."""
    records: list[dict[str, Any]] = []
    group = result.loc[row_index]
    row_value = int(group["row"].iloc[0])
    col_value = int(group["col"].iloc[0])
    base_fit_mask = pd.Series(True, index=group.index)
    if fit_end_year is not None:
        base_fit_mask &= group["year"].astype(int) <= int(fit_end_year)

    for variable in variables:
        residual_col = f"{variable}{suffix}"
        expected_col = f"{variable}{expected_suffix}"
        needed = [variable, "year", "month", "_residual_time_index"]
        if seasonal_model == "annual-harmonic":
            needed.extend(["month_sin", "month_cos"])
        fit_mask = base_fit_mask & group[needed].notna().all(axis=1)
        prediction_columns = ["year", "month", "_residual_time_index"]
        if seasonal_model == "annual-harmonic":
            prediction_columns.extend(["month_sin", "month_cos"])
        predict_mask = group[prediction_columns].notna().all(axis=1)

        month_counts = (
            group.loc[fit_mask, "month"]
            .astype(int)
            .value_counts()
            .reindex(range(1, 13), fill_value=0)
            .sort_index()
        )
        minimum_month_count = int(month_counts.min())

        record: dict[str, Any] = {
            "row": row_value,
            "col": col_value,
            "variable": variable,
            "residual_column": residual_col,
            "expected_column": expected_col,
            "n_fit": int(fit_mask.sum()),
            "fit_end_year": fit_end_year,
            "seasonal_model": seasonal_model,
            "include_trend": bool(include_trend),
            "n_parameters": 0,
            "design_rank": 0,
            "design_condition_number": np.nan,
            "time_center_month_index": np.nan,
            "min_month_fit_samples": minimum_month_count,
            "month_fit_counts_json": json.dumps(
                {str(month): int(count) for month, count in month_counts.items()}
            ),
            "status": "fit",
            "intercept": np.nan,
            "month_sin": np.nan,
            "month_cos": np.nan,
            "time_trend": np.nan,
            "fit_r2": np.nan,
            "fit_residual_std": np.nan,
            "coefficients_json": None,
        }

        if int(fit_mask.sum()) < min_fit_samples:
            record["status"] = "too_few_fit_samples"
            records.append(record)
            continue

        if (
            seasonal_model == "monthly-fixed-effects"
            and minimum_month_count < min_month_samples
        ):
            record["status"] = "too_few_samples_in_calendar_month"
            records.append(record)
            continue

        fit_frame = group.loc[fit_mask]
        time_center = float(
            fit_frame["_residual_time_index"].astype(float).mean()
        )
        x_fit = design_matrix(
            fit_frame,
            seasonal_model=seasonal_model,
            include_trend=include_trend,
            time_center=time_center,
        )
        y_fit = fit_frame[variable].astype(float).to_numpy()
        design_rank = int(np.linalg.matrix_rank(x_fit))
        n_parameters = int(x_fit.shape[1])
        record["n_parameters"] = n_parameters
        record["design_rank"] = design_rank
        record["design_condition_number"] = float(np.linalg.cond(x_fit))
        record["time_center_month_index"] = time_center
        if design_rank < n_parameters:
            record["status"] = "design_rank_deficient"
            records.append(record)
            continue
        try:
            coefficients, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
        except np.linalg.LinAlgError:
            record["status"] = "linear_solve_failed"
            records.append(record)
            continue

        predict_frame = group.loc[predict_mask]
        x_predict = design_matrix(
            predict_frame,
            seasonal_model=seasonal_model,
            include_trend=include_trend,
            time_center=time_center,
        )
        expected = x_predict @ coefficients
        expected_series = pd.Series(expected, index=predict_frame.index)
        predict_index = predict_frame.index
        residual = group.loc[predict_index, variable].astype(float) - expected_series

        result.loc[predict_index, expected_col] = expected_series
        result.loc[predict_index, residual_col] = residual

        record["intercept"] = float(coefficients[0])
        fitted = x_fit @ coefficients
        centered_sum_squares = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        error_sum_squares = float(np.sum((y_fit - fitted) ** 2))
        record["fit_r2"] = (
            1.0 - error_sum_squares / centered_sum_squares
            if centered_sum_squares > 0.0
            else np.nan
        )
        record["fit_residual_std"] = float(
            np.std(y_fit - fitted, ddof=min(1, len(y_fit) - 1))
        )
        coefficient_names = [
            "intercept",
            *design_columns(include_trend, seasonal_model),
        ]
        coefficient_by_name = {
            predictor: float(coefficient)
            for predictor, coefficient in zip(
                coefficient_names,
                coefficients,
                strict=True,
            )
        }
        record["coefficients_json"] = json.dumps(coefficient_by_name)
        record["month_sin"] = coefficient_by_name.get("month_sin", np.nan)
        record["month_cos"] = coefficient_by_name.get("month_cos", np.nan)
        record["time_trend"] = coefficient_by_name.get(
            "time_trend_per_year",
            np.nan,
        )
        records.append(record)

    return records


def residualize_dataframe(
    df: pd.DataFrame,
    variables: Sequence[str],
    fit_end_year: int | None,
    min_fit_samples: int,
    suffix: str,
    expected_suffix: str,
    include_trend: bool,
    seasonal_model: str = "monthly-fixed-effects",
    min_month_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add residual and expected-value columns to an ARD data frame."""
    if seasonal_model not in SEASONAL_MODELS:
        raise click.ClickException(
            f"Unsupported seasonal model {seasonal_model!r}; choose from "
            + ", ".join(SEASONAL_MODELS)
        )
    required = {"row", "col", "year", "month"}
    if seasonal_model == "annual-harmonic":
        required.update({"month_sin", "month_cos"})
    missing = sorted(required - set(df.columns))
    if missing:
        raise click.ClickException(f"Input table is missing columns: {missing}")
    missing_variables = [variable for variable in variables if variable not in df.columns]
    if missing_variables:
        raise click.ClickException(
            "Input table is missing variables to residualize: "
            + ", ".join(missing_variables)
        )

    result = df.sort_values(["row", "col", "year", "month"]).copy()
    invalid_months = sorted(
        set(result["month"].dropna().astype(int)) - set(range(1, 13))
    )
    if invalid_months:
        raise click.ClickException(
            f"Month values must be in 1..12; found {invalid_months}"
        )
    month_index = result["year"].astype(int) * 12 + result["month"].astype(int) - 1
    result["_residual_time_index"] = (
        month_index - int(month_index.min())
    ).astype(float)
    for variable in variables:
        result[f"{variable}{suffix}"] = np.nan
        result[f"{variable}{expected_suffix}"] = np.nan

    model_records: list[dict[str, Any]] = []
    groups = result.groupby(["row", "col"], sort=True).groups
    for row_index in tqdm(groups.values(), total=len(groups), desc="Residualizing pixels"):
        model_records.extend(
            residualize_group(
                result=result,
                row_index=row_index,
                variables=variables,
                fit_end_year=fit_end_year,
                min_fit_samples=min_fit_samples,
                min_month_samples=min_month_samples,
                suffix=suffix,
                expected_suffix=expected_suffix,
                include_trend=include_trend,
                seasonal_model=seasonal_model,
            )
        )

    model_df = pd.DataFrame(
        model_records,
        columns=[
            "row",
            "col",
            "variable",
            "residual_column",
            "expected_column",
            "n_fit",
            "fit_end_year",
            "seasonal_model",
            "include_trend",
            "n_parameters",
            "design_rank",
            "design_condition_number",
            "time_center_month_index",
            "min_month_fit_samples",
            "month_fit_counts_json",
            "status",
            "intercept",
            "month_sin",
            "month_cos",
            "time_trend",
            "fit_r2",
            "fit_residual_std",
            "coefficients_json",
        ],
    )
    return result, model_df


def write_residual_config(
    config_path: Path,
    config: Mapping[str, Any],
    output_config: Path,
    output_db: Path,
    output_table: str,
    graph_db: Path,
    variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
    fit_end_year: int | None,
    include_trend: bool,
    seasonal_model: str = "monthly-fixed-effects",
    min_month_samples: int = 3,
) -> None:
    """Write a config that points graph discovery to residualized variables."""
    generated = copy.deepcopy(dict(config))
    original_specs = {
        str(spec["name"]): dict(spec)
        for spec in config["columns"]
    }
    generated["columns"] = [
        {
            **{
                key: value
                for key, value in original_specs[variable].items()
                if key != "name"
            },
            "name": f"{variable}{suffix}",
        }
        for variable in variables
    ]

    reference_var = str(config.get("reference_var", ""))
    if reference_var in variables:
        generated["reference_var"] = f"{reference_var}{suffix}"

    config_dir = output_config.parent
    generated["timeseries_db"] = path_for_config(config_dir, output_db)
    generated["timeseries_table"] = output_table
    generated["graph_db"] = path_for_config(config_dir, graph_db)
    generated["graph_discovery"] = {
        **dict(generated.get("graph_discovery") or {}),
        "input_db": path_for_config(config_dir, output_db),
        "input_table": output_table,
        "output_db": path_for_config(config_dir, graph_db),
        "max_year": fit_end_year,
    }
    generated["residualization"] = {
        "source_config": path_for_config(config_dir, config_path),
        "variables": list(variables),
        "suffix": suffix,
        "expected_suffix": expected_suffix,
        "fit_end_year": fit_end_year,
        "include_trend": bool(include_trend),
        "seasonal_model": seasonal_model,
        "min_month_samples": int(min_month_samples),
        "controls": (
            ["calendar_month_fixed_effects"]
            if seasonal_model == "monthly-fixed-effects"
            else ["month_sin", "month_cos"]
        )
        + (["linear_time_trend"] if include_trend else []),
        "note": (
            "Graph discovery should use the residual columns and should not "
            "include deterministic seasonal controls as endogenous variables."
        ),
    }

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as fd:
        yaml.safe_dump(generated, fd, sort_keys=False)


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Experiment YAML configuration.",
)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--output-table", default=None)
@click.option("--output-config", default=None, type=click.Path(path_type=Path))
@click.option("--graph-db", default=None, type=click.Path(path_type=Path))
@click.option("--fit-end-year", default=None, type=int)
@click.option("--variable", "variables", multiple=True)
@click.option("--suffix", default="_resid", show_default=True)
@click.option("--expected-suffix", default="_seasonal_trend", show_default=True)
@click.option("--min-fit-samples", default=60, show_default=True, type=click.IntRange(min=4))
@click.option(
    "--seasonal-model",
    type=click.Choice(SEASONAL_MODELS, case_sensitive=False),
    default="monthly-fixed-effects",
    show_default=True,
    help=(
        "Seasonal baseline. Monthly fixed effects allow an arbitrary annual "
        "shape; annual harmonic preserves the legacy sine/cosine model."
    ),
)
@click.option(
    "--min-month-samples",
    default=3,
    show_default=True,
    type=click.IntRange(min=1),
    help="Minimum fitting observations required in every calendar month.",
)
@click.option("--trend/--no-trend", default=True, show_default=True)
def residualize_timeseries(
    config_path: Path,
    input_db: Path | None,
    input_table: str | None,
    output_db: Path | None,
    output_table: str | None,
    output_config: Path | None,
    graph_db: Path | None,
    fit_end_year: int | None,
    variables: tuple[str, ...],
    suffix: str,
    expected_suffix: str,
    min_fit_samples: int,
    seasonal_model: str,
    min_month_samples: int,
    trend: bool,
) -> None:
    """Remove seasonality/trend from ARD variables and write residual columns."""
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])

    input_db = resolve_path(
        experiment_dir,
        input_db,
        experiment_dir / f"{experiment_name}_ard.duckdb",
    )
    output_db = resolve_path(experiment_dir, output_db, input_db)
    input_table = input_table or str(config.get("timeseries_table", experiment_name))
    output_table = output_table or f"{experiment_name}_residualized"
    output_config = resolve_path(
        experiment_dir,
        output_config,
        experiment_dir / f"{experiment_name}_residualized.yaml",
    )
    graph_db = resolve_path(
        experiment_dir,
        graph_db,
        experiment_dir / f"{experiment_name}_residualized_graphs.duckdb",
    )

    variables_to_residualize = configured_variables(config, variables)
    require_files([input_db])
    click.echo(f"Reading {input_db}::{input_table}...")
    df = read_table(input_db, input_table)
    click.echo(
        "Residualizing variables: "
        + ", ".join(variables_to_residualize)
    )
    click.echo(
        f"Seasonal model: {seasonal_model}; linear trend: {trend}; "
        f"minimum observations per calendar month: {min_month_samples}"
    )
    residualized, models = residualize_dataframe(
        df=df,
        variables=variables_to_residualize,
        fit_end_year=fit_end_year,
        min_fit_samples=min_fit_samples,
        suffix=suffix,
        expected_suffix=expected_suffix,
        include_trend=trend,
        seasonal_model=seasonal_model,
        min_month_samples=min_month_samples,
    )

    output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, residualized, output_table)
        write_dataframe_table(
            con,
            models,
            f"{output_table}_residualization_models",
        )
    finally:
        con.close()

    write_residual_config(
        config_path=config_path,
        config=config,
        output_config=output_config,
        output_db=output_db,
        output_table=output_table,
        graph_db=graph_db,
        variables=variables_to_residualize,
        suffix=suffix,
        expected_suffix=expected_suffix,
        fit_end_year=fit_end_year,
        include_trend=trend,
        seasonal_model=seasonal_model,
        min_month_samples=min_month_samples,
    )

    status_counts = models["status"].value_counts().to_dict()
    click.echo("")
    click.echo("Residualization complete.")
    click.echo(f"Output table: {output_db}::{output_table}")
    click.echo(
        "Model status counts: "
        + ", ".join(f"{status}={count}" for status, count in status_counts.items())
    )
    click.echo(f"Generated config: {output_config}")
    click.echo(f"Configured residual graph DB: {graph_db}")


if __name__ == "__main__":
    residualize_timeseries()
