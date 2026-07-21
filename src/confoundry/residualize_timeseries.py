"""Residualize ARD time series before causal graph discovery.

This command removes deterministic seasonality and optional long-term trend
from configured environmental variables.  The residual columns can then be used
for DirectLiNGAM graph discovery without treating ``month_sin`` and
``month_cos`` as endogenous variables.
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
    return path if path.is_absolute() else base_dir / path


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


def design_columns(include_trend: bool) -> list[str]:
    """Return residualization covariate names."""
    columns = ["month_sin", "month_cos"]
    if include_trend:
        columns.append("_residual_time_index")
    return columns


def design_matrix(frame: pd.DataFrame, predictors: Sequence[str]) -> np.ndarray:
    """Build an intercept-plus-covariates design matrix."""
    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            *[
                frame[predictor].astype(float).to_numpy()
                for predictor in predictors
            ],
        ]
    )


def residualize_group(
    result: pd.DataFrame,
    row_index: pd.Index,
    variables: Sequence[str],
    predictors: Sequence[str],
    fit_end_year: int | None,
    min_fit_samples: int,
    suffix: str,
    expected_suffix: str,
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
        needed = [variable, *predictors]
        fit_mask = base_fit_mask & group[needed].notna().all(axis=1)
        predict_mask = group[predictors].notna().all(axis=1)

        record: dict[str, Any] = {
            "row": row_value,
            "col": col_value,
            "variable": variable,
            "residual_column": residual_col,
            "expected_column": expected_col,
            "n_fit": int(fit_mask.sum()),
            "fit_end_year": fit_end_year,
            "status": "fit",
            "intercept": np.nan,
            "month_sin": np.nan,
            "month_cos": np.nan,
            "time_trend": np.nan,
        }

        if int(fit_mask.sum()) < min_fit_samples:
            record["status"] = "too_few_fit_samples"
            records.append(record)
            continue

        fit_frame = group.loc[fit_mask]
        x_fit = design_matrix(fit_frame, predictors)
        y_fit = fit_frame[variable].astype(float).to_numpy()
        try:
            coefficients, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
        except np.linalg.LinAlgError:
            record["status"] = "linear_solve_failed"
            records.append(record)
            continue

        predict_frame = group.loc[predict_mask]
        x_predict = design_matrix(predict_frame, predictors)
        expected = x_predict @ coefficients
        expected_series = pd.Series(expected, index=predict_frame.index)
        predict_index = predict_frame.index
        residual = group.loc[predict_index, variable].astype(float) - expected_series

        result.loc[predict_index, expected_col] = expected_series
        result.loc[predict_index, residual_col] = residual

        record["intercept"] = float(coefficients[0])
        coefficient_by_name = {
            predictor: float(coefficient)
            for predictor, coefficient in zip(predictors, coefficients[1:], strict=True)
        }
        record["month_sin"] = coefficient_by_name.get("month_sin", np.nan)
        record["month_cos"] = coefficient_by_name.get("month_cos", np.nan)
        record["time_trend"] = coefficient_by_name.get("_residual_time_index", np.nan)
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add residual and expected-value columns to an ARD data frame."""
    required = {"row", "col", "year", "month", "month_sin", "month_cos"}
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
    month_index = result["year"].astype(int) * 12 + result["month"].astype(int) - 1
    result["_residual_time_index"] = (
        month_index - int(month_index.min())
    ).astype(float)
    predictors = design_columns(include_trend)

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
                predictors=predictors,
                fit_end_year=fit_end_year,
                min_fit_samples=min_fit_samples,
                suffix=suffix,
                expected_suffix=expected_suffix,
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
            "status",
            "intercept",
            "month_sin",
            "month_cos",
            "time_trend",
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
        "controls": ["month_sin", "month_cos"]
        + (["_residual_time_index"] if include_trend else []),
        "note": (
            "Graph discovery should use the residual columns and should not "
            "include month_sin or month_cos as endogenous variables."
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
@click.option("--min-fit-samples", default=24, show_default=True, type=click.IntRange(min=4))
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
    residualized, models = residualize_dataframe(
        df=df,
        variables=variables_to_residualize,
        fit_end_year=fit_end_year,
        min_fit_samples=min_fit_samples,
        suffix=suffix,
        expected_suffix=expected_suffix,
        include_trend=trend,
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
