#!/usr/bin/env python3
"""Run a safe residualized 2025 causal holdout validation.

This script deliberately avoids ``confoundry.gather`` because that command
replaces the default ARD table.  Instead it builds a staged ARD database,
residualizes that staged data, writes a staged residualized config, and runs
the holdout validator against the existing pre-2025 graph database.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import pandas as pd
import yaml
from tqdm.contrib.concurrent import process_map

from confoundry.analysis_helpers import write_dataframe_table
from confoundry.causal_holdout_validation import shift_year_month, target_shift
from confoundry.future_ndvi_prediction import (
    ensure_validation_sources,
    target_source_names,
)
from confoundry.gather import (
    assemble_data_frame,
    assemble_timeseries_paths_from_db,
)
from confoundry.residualize_timeseries import (
    configured_variables,
    residualize_dataframe,
    write_residual_config,
)


SEASONAL_COLUMNS = {"month_sin", "month_cos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely stage raw ARD, rebuild residualized data, and run 2025 "
            "causal holdout validation without overwriting the working DBs."
        )
    )
    parser.add_argument("-c", "--config", default="experiment.yaml")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--training-end-year", type=int, default=2024)
    parser.add_argument(
        "--month",
        dest="months",
        action="append",
        type=int,
        help="Observed target month to try. Defaults to all months 1..12.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--stage-dir",
        default=None,
        help="Directory for staged DBs/config. Defaults to holdout_stage_<year>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Validation output directory. Defaults to causal_holdout_<year>_residualized.",
    )
    parser.add_argument(
        "--graph-db",
        default=None,
        help=(
            "Existing residualized graph DB trained through training-end-year. "
            "Defaults to <name>_residualized_graphs.duckdb."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not try to download missing 2025 source rasters.",
    )
    parser.add_argument(
        "--allow-january",
        action="store_true",
        help=(
            "Allow January even if target shifting maps it onto the training "
            "year. By default it is skipped for strict holdout validation."
        ),
    )
    parser.add_argument(
        "--min-fit-samples",
        type=int,
        default=24,
        help="Minimum samples for each residualization fit.",
    )
    parser.add_argument(
        "--validation-min-train-samples",
        type=int,
        default=None,
        help=(
            "Minimum samples for holdout validation. Defaults to the number "
            "of historical years, capped at the validator default of 30."
        ),
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{path} must contain a YAML mapping.")
    for key in ("name", "columns", "name_map", "reference_var"):
        if key not in config:
            raise SystemExit(f"{path} is missing required key {key!r}.")
    return config


def configured_data_variables(config: Mapping[str, Any]) -> list[str]:
    return [
        str(spec["name"])
        for spec in config["columns"]
        if str(spec["name"]) not in SEASONAL_COLUMNS
    ]


def source_db_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    return config_path.parent / f"{config['name']}_source_db.duckdb"


def resolve_cli_path(value: str | None, default: Path) -> Path:
    """Resolve a CLI path relative to the current shell, not the config dir."""
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def first_graph_variables(graph_db: Path) -> list[str]:
    con = duckdb.connect(graph_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if "pixel_graphs" not in tables:
            raise SystemExit(
                f"{graph_db} has no pixel_graphs table. Available: {sorted(tables)}"
            )
        row = con.execute(
            """
            SELECT variable_names_json
            FROM pixel_graphs
            WHERE variable_names_json IS NOT NULL
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise SystemExit(f"{graph_db} has no graph variable names.")
    return [str(value) for value in json.loads(row[0])]


def print_validation_diagnostics(output_dir: Path) -> None:
    diagnostics_path = output_dir / "causal_holdout_diagnostics.csv"
    if not diagnostics_path.exists():
        print(f"No diagnostics file found at {diagnostics_path}", file=sys.stderr)
        return
    diagnostics = pd.read_csv(diagnostics_path)
    if "status" in diagnostics:
        print("", file=sys.stderr)
        print("Validation diagnostic statuses:", file=sys.stderr)
        print(diagnostics["status"].value_counts(dropna=False).to_string(), file=sys.stderr)
    if "missing_evaluation_values" in diagnostics:
        values = diagnostics["missing_evaluation_values"].dropna()
        if not values.empty:
            exploded = values.str.split(",").explode()
            print("", file=sys.stderr)
            print("Missing evaluation values:", file=sys.stderr)
            print(exploded.value_counts().to_string(), file=sys.stderr)
    columns = [
        column
        for column in [
            "status",
            "n_parents",
            "n_train",
            "n_complete",
            "n_monthly_train",
            "missing_columns",
            "missing_evaluation_values",
        ]
        if column in diagnostics.columns
    ]
    if columns:
        print("", file=sys.stderr)
        print("Diagnostic sample:", file=sys.stderr)
        print(diagnostics[columns].head(20).to_string(index=False), file=sys.stderr)


def first_catalog_file_failures(
    source_db: Path,
    years: Sequence[int],
    source_variables: set[str],
    limit: int = 20,
) -> list[str]:
    con = duckdb.connect(source_db, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT year, month, variable_name, root_dir, file_name
            FROM geotiff_catalog
            WHERE year IN (
            """
            + ", ".join(["?"] * len(years))
            + """
            )
            ORDER BY year, month, variable_name
            """,
            list(years),
        ).fetchall()
    finally:
        con.close()

    failures: list[str] = []
    for year, month, variable, root_dir, file_name in rows:
        if str(variable) not in source_variables:
            continue
        path = Path(str(root_dir)) / str(file_name)
        if not path.exists():
            failures.append(f"{year}-{int(month):02d} {variable}: {path}")
            if len(failures) >= limit:
                break
    return failures


def complete_months(
    datasets: Mapping[tuple[int, int], Mapping[str, Path]],
    required_variables: set[str],
) -> set[tuple[int, int]]:
    return {
        key
        for key, month_datasets in datasets.items()
        if required_variables.issubset(set(month_datasets))
    }


def choose_validation_months(
    *,
    requested_months: Sequence[int],
    datasets: Mapping[tuple[int, int], Mapping[str, Path]],
    required_variables: set[str],
    target_variable: str,
    year: int,
    training_end_year: int,
    raw_target_shift: int,
    allow_january: bool,
) -> list[int]:
    complete = complete_months(datasets, required_variables)
    usable: list[int] = []
    skipped: list[str] = []

    for month in sorted(set(int(month) for month in requested_months)):
        model_year, model_month = shift_year_month(year, month, raw_target_shift)
        if not allow_january and model_year <= training_end_year:
            skipped.append(
                f"{month:02d}: shifted model row is {model_year}-{model_month:02d}"
            )
            continue
        observed_has_target = target_variable in datasets.get((year, month), {})
        model_complete = (model_year, model_month) in complete
        if observed_has_target and model_complete:
            usable.append(month)
            continue
        reason = []
        if not observed_has_target:
            reason.append("missing observed target raster")
        if not model_complete:
            reason.append(f"missing complete model row {model_year}-{model_month:02d}")
        skipped.append(f"{month:02d}: " + ", ".join(reason))

    print("Validation months:", ", ".join(str(month) for month in usable) or "none")
    if skipped:
        print("Skipped months:")
        for line in skipped:
            print(f"  - {line}")
    if not usable:
        raise SystemExit("No usable validation months remain.")
    return usable


def build_staged_ard(
    *,
    datasets: Mapping[tuple[int, int], Mapping[str, Path]],
    required_months: set[tuple[int, int]],
    reference_var: str,
    output_db: Path,
    output_table: str,
    workers: int,
) -> pd.DataFrame:
    tasks = [
        (year, month, reference_var, dict(datasets[(year, month)]))
        for year, month in sorted(required_months)
    ]
    print(f"Building staged ARD from {len(tasks)} monthly raster groups...")
    frames = process_map(
        assemble_data_frame,
        tasks,
        max_workers=workers,
        ascii=True,
    )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("No ARD frames were assembled.")
    df = pd.concat(frames, ignore_index=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, df, output_table)
    finally:
        con.close()
    print(f"Staged raw ARD: {output_db}::{output_table}")
    return df


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])
    year = int(args.year)
    training_end_year = int(args.training_end_year)
    requested_months = args.months or list(range(1, 13))
    target_variable = str(config["reference_var"])
    variables = configured_data_variables(config)
    required_variables = set(variables)
    raw_target_shift = target_shift(config, target_variable)

    stage_dir = (
        Path(args.stage_dir)
        if args.stage_dir is not None
        else experiment_dir / f"holdout_stage_{year}"
    )
    if not stage_dir.is_absolute():
        stage_dir = experiment_dir / stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)

    source_db = source_db_path(config_path, config)
    if not source_db.exists():
        raise SystemExit(f"Missing source DB: {source_db}")

    graph_db = resolve_cli_path(
        args.graph_db,
        experiment_dir / f"{experiment_name}_residualized_graphs.duckdb",
    )
    if not graph_db.exists():
        raise SystemExit(f"Missing graph DB: {graph_db}")

    source_variables = {
        source_name
        for source_name, normalized in dict(config["name_map"]).items()
        if str(normalized) in required_variables
    }
    if not args.no_download:
        try:
            ensure_validation_sources(
                config_path=config_path,
                config=config,
                source_db=source_db,
                target_variable=target_variable,
                evaluation_year=year,
                target_months=requested_months,
                download_if_missing=True,
            )
        except click.ClickException as exc:
            print(f"Download/source warning: {exc}", file=sys.stderr)

    missing_examples = first_catalog_file_failures(
        source_db=source_db,
        years=list(range(2005, training_end_year + 1)),
        source_variables=source_variables,
    )
    if missing_examples:
        print("Historical source catalog rows point to missing files.", file=sys.stderr)
        print("First missing examples:", file=sys.stderr)
        for line in missing_examples:
            print(f"  - {line}", file=sys.stderr)
        raise SystemExit(
            "Restore or relink the historical rasters, then rerun this script."
        )

    datasets = assemble_timeseries_paths_from_db(source_db, config["name_map"])
    usable_months = choose_validation_months(
        requested_months=requested_months,
        datasets=datasets,
        required_variables=required_variables,
        target_variable=target_variable,
        year=year,
        training_end_year=training_end_year,
        raw_target_shift=raw_target_shift,
        allow_january=bool(args.allow_january),
    )

    complete = complete_months(datasets, required_variables)
    historical_months = {
        key
        for key in complete
        if int(key[0]) <= training_end_year
    }
    if not historical_months:
        raise SystemExit("No complete historical months are available for residualization.")
    historical_year_count = len({int(year_value) for year_value, _month in historical_months})
    validation_min_train_samples = (
        int(args.validation_min_train_samples)
        if args.validation_min_train_samples is not None
        else min(30, historical_year_count)
    )

    evaluation_model_months = {
        shift_year_month(year, month, raw_target_shift)
        for month in usable_months
    }
    observed_target_months = {
        (year, month)
        for month in usable_months
        if target_variable in datasets.get((year, month), {})
    }
    required_months = historical_months | evaluation_model_months | observed_target_months

    staged_ard_db = stage_dir / f"{experiment_name}_staged_ard.duckdb"
    staged_ard_table = experiment_name
    raw_df = build_staged_ard(
        datasets=datasets,
        required_months=required_months,
        reference_var=target_variable,
        output_db=staged_ard_db,
        output_table=staged_ard_table,
        workers=int(args.workers),
    )

    residualized_db = stage_dir / f"{experiment_name}_staged_residualized.duckdb"
    residualized_table = f"{experiment_name}_residualized"
    residualized_config = stage_dir / f"{experiment_name}_residualized_holdout_{year}.yaml"
    variables_to_residualize = configured_variables(config, ())

    print(f"Residualizing staged ARD with fit_end_year={training_end_year}...")
    residualized, models = residualize_dataframe(
        df=raw_df,
        variables=variables_to_residualize,
        fit_end_year=training_end_year,
        min_fit_samples=int(args.min_fit_samples),
        suffix="_resid",
        expected_suffix="_seasonal_trend",
        include_trend=True,
    )
    con = duckdb.connect(residualized_db)
    try:
        write_dataframe_table(con, residualized, residualized_table)
        write_dataframe_table(
            con,
            models,
            f"{residualized_table}_residualization_models",
        )
    finally:
        con.close()

    status_counts = models["status"].value_counts().to_dict()
    print(
        "Residualization statuses: "
        + ", ".join(f"{status}={count}" for status, count in status_counts.items())
    )
    if int((models["status"] == "fit").sum()) == 0:
        raise SystemExit("Residualization produced no fitted models.")

    write_residual_config(
        config_path=config_path,
        config=config,
        output_config=residualized_config,
        output_db=residualized_db,
        output_table=residualized_table,
        graph_db=graph_db,
        variables=variables_to_residualize,
        suffix="_resid",
        expected_suffix="_seasonal_trend",
        fit_end_year=training_end_year,
        include_trend=True,
    )

    residual_target = f"{target_variable}_resid"
    graph_variables = first_graph_variables(graph_db)
    if residual_target not in graph_variables:
        raise SystemExit(
            f"Graph DB target mismatch. Expected {residual_target!r} in "
            f"{graph_db}::pixel_graphs variable_names_json, but first graph has: "
            + ", ".join(graph_variables)
        )

    output_dir = resolve_cli_path(
        args.output_dir,
        experiment_dir / f"causal_holdout_{year}_residualized",
    )

    command = [
        sys.executable,
        "-m",
        "confoundry.causal_holdout_validation",
        "-c",
        str(residualized_config),
        "--evaluation-year",
        str(year),
        "--training-end-year",
        str(training_end_year),
        "--graph-training-max-year",
        str(training_end_year),
        "--min-train-samples",
        str(validation_min_train_samples),
        "-o",
        str(output_dir),
    ]
    for month in usable_months:
        command.extend(["--observed-target-month", str(month)])

    print("Running validation:")
    print(" ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print_validation_diagnostics(output_dir)
        raise

    metrics_path = output_dir / "causal_holdout_metrics.csv"
    if metrics_path.exists():
        print("")
        print(pd.read_csv(metrics_path).to_string(index=False))
    print("")
    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
