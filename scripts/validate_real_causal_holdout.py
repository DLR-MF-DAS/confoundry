#!/usr/bin/env python3
"""Validate a residualized causal graph against held-out raw observations.

This is intentionally separate from the older holdout runner.  It implements
the validation contract directly:

1. assemble a non-destructive raw ARD table from the source catalog, or read a
   supplied raw ARD database;
2. fit the residualization transform using only training-target years;
3. apply that fixed transform to held-out data;
4. predict held-out target residuals from the frozen causal graph;
5. reconstruct raw target predictions and score them against raw observations.

No default experiment database or graph database is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from confoundry.causal_holdout_validation import (
    graph_target_effect_coefficients,
    shift_year_month,
    target_shift,
)
from confoundry.gather import assemble_data_frame, assemble_timeseries_paths_from_db
from confoundry.landcover_helpers import load_graph_rows
from confoundry.per_pixel_graph_discovery import get_pixel_window_group, parse_columns
from confoundry.residualize_timeseries import configured_variables, residualize_dataframe


SEASONAL_COLUMNS = {"month_sin", "month_cos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a frozen residualized causal graph on real held-out data."
    )
    parser.add_argument("-c", "--config", required=True, help="Raw experiment YAML.")
    parser.add_argument("--evaluation-year", type=int, required=True)
    parser.add_argument("--training-end-year", type=int, required=True)
    parser.add_argument(
        "--month",
        dest="months",
        action="append",
        type=int,
        help="Observed target month. Repeat. Defaults to all available months.",
    )
    parser.add_argument("--source-db", default=None)
    parser.add_argument("--raw-ard-db", default=None)
    parser.add_argument("--raw-table", default=None)
    parser.add_argument("--graph-db", default=None)
    parser.add_argument("--graph-table", default="pixel_graphs")
    parser.add_argument("--graph-window-size", type=int, default=0)
    parser.add_argument("--effect-mode", choices=["direct", "total"], default="direct")
    parser.add_argument("--min-monthly-train-samples", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("-o", "--output-dir", required=True)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{path} must contain a YAML mapping.")
    missing = [key for key in ["name", "columns", "name_map", "reference_var"] if key not in config]
    if missing:
        raise SystemExit(f"{path} is missing required keys: {missing}")
    return config


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def quote_identifier(name: str) -> str:
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise SystemExit(f"Unsafe DuckDB identifier: {name!r}")
    return f'"{name}"'


def raw_variables(config: Mapping[str, Any]) -> list[str]:
    return [
        str(spec["name"])
        for spec in config["columns"]
        if str(spec["name"]) not in SEASONAL_COLUMNS
    ]


def residual_specs(config: Mapping[str, Any], suffix: str = "_resid") -> list[dict[str, Any]]:
    specs = []
    for spec in config["columns"]:
        name = str(spec["name"])
        if name in SEASONAL_COLUMNS:
            continue
        item = {key: value for key, value in dict(spec).items() if key != "name"}
        item["name"] = f"{name}{suffix}"
        specs.append(item)
    return specs


def month_index(year: int, month: int) -> int:
    return year * 12 + month - 1


def year_month_from_index(index: int) -> tuple[int, int]:
    return index // 12, index % 12 + 1


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    start_idx = month_index(*start)
    end_idx = month_index(*end)
    return [year_month_from_index(idx) for idx in range(start_idx, end_idx + 1)]


def read_raw_ard(db_path: Path, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table not in tables:
            raise SystemExit(f"{table!r} not found in {db_path}. Available: {sorted(tables)}")
        return con.execute(f"SELECT * FROM {quote_identifier(table)}").fetchdf()
    finally:
        con.close()


def complete_source_months(
    source_db: Path,
    config: Mapping[str, Any],
    required_variables: set[str],
) -> tuple[dict[tuple[int, int], dict[str, Path]], set[tuple[int, int]]]:
    datasets = assemble_timeseries_paths_from_db(source_db, config["name_map"])
    complete = {
        key
        for key, month_data in datasets.items()
        if required_variables.issubset(set(month_data))
    }
    return datasets, complete


def choose_months(
    *,
    requested_months: Sequence[int] | None,
    evaluation_year: int,
    target: str,
    target_lag: int,
    datasets: Mapping[tuple[int, int], Mapping[str, Path]],
    complete: set[tuple[int, int]],
) -> list[int]:
    candidates = requested_months or list(range(1, 13))
    chosen: list[int] = []
    skipped: list[str] = []
    for month in sorted(set(int(m) for m in candidates)):
        model_key = shift_year_month(evaluation_year, month, target_lag)
        observed_key = (evaluation_year, month)
        reasons = []
        if target not in datasets.get(observed_key, {}):
            reasons.append("missing observed target")
        if model_key not in complete:
            reasons.append(f"missing complete model row {model_key[0]}-{model_key[1]:02d}")
        if reasons:
            skipped.append(f"{month:02d}: " + ", ".join(reasons))
        else:
            chosen.append(month)
    print("Validation months:", ", ".join(str(month) for month in chosen) or "none")
    if skipped:
        print("Skipped months:")
        for line in skipped:
            print(f"  - {line}")
    if not chosen:
        raise SystemExit("No usable held-out months are available.")
    return chosen


def build_raw_ard_from_source(
    *,
    source_db: Path,
    config: Mapping[str, Any],
    required_months: Sequence[tuple[int, int]],
    output_db: Path,
    output_table: str,
    workers: int,
) -> pd.DataFrame:
    datasets, complete = complete_source_months(source_db, config, set(raw_variables(config)))
    missing = [key for key in required_months if key not in complete]
    if missing:
        preview = ", ".join(f"{year}-{month:02d}" for year, month in missing[:20])
        raise SystemExit(f"Missing complete source months needed for validation: {preview}")

    tasks = [
        (year, month, str(config["reference_var"]), dict(datasets[(year, month)]))
        for year, month in required_months
    ]
    print(f"Assembling non-destructive raw ARD from {len(tasks)} monthly groups...")
    frames = process_map(assemble_data_frame, tasks, max_workers=workers, ascii=True)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("No raw ARD rows were assembled.")
    df = pd.concat(frames, ignore_index=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(output_db)
    try:
        con.register("_raw_ard", df)
        con.execute(f"CREATE OR REPLACE TABLE {quote_identifier(output_table)} AS SELECT * FROM _raw_ard")
        con.unregister("_raw_ard")
    finally:
        con.close()
    print(f"Staged raw ARD: {output_db}::{output_table}")
    return df


def add_shifted_target_columns(
    df: pd.DataFrame,
    target: str,
    expected_col: str,
    target_lag: int,
) -> pd.DataFrame:
    result = df.sort_values(["row", "col", "year", "month"]).copy()
    grouped = result.groupby(["row", "col"], sort=False)
    result["_target_raw"] = grouped[target].shift(target_lag)
    result["_target_expected"] = grouped[expected_col].shift(target_lag)
    result["_target_observed_year"] = [
        shift_year_month(int(year), int(month), -target_lag)[0]
        for year, month in zip(result["year"], result["month"], strict=True)
    ]
    result["_target_observed_month"] = [
        shift_year_month(int(year), int(month), -target_lag)[1]
        for year, month in zip(result["year"], result["month"], strict=True)
    ]
    return result


def prepare_shifted_residual_frame(
    raw_df: pd.DataFrame,
    config: Mapping[str, Any],
    training_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    target = str(config["reference_var"])
    target_lag = target_shift(config, target)
    residualized, models = residualize_dataframe(
        df=raw_df,
        variables=configured_variables(config, ()),
        fit_end_year=training_end_year,
        min_fit_samples=10,
        suffix="_resid",
        expected_suffix="_expected",
        include_trend=True,
    )
    expected_col = f"{target}_expected"
    residualized = add_shifted_target_columns(residualized, target, expected_col, target_lag)
    shifted, labels, _lags = parse_columns(
        df=residualized,
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        column_specs=residual_specs(config),
    )
    return shifted, models, f"{target}_resid"


def finite_frame(frame: pd.DataFrame, observed_col: str, predicted_col: str) -> pd.DataFrame:
    subset = frame.dropna(subset=[observed_col, predicted_col]).copy()
    if subset.empty:
        return subset
    observed = subset[observed_col].astype(float)
    predicted = subset[predicted_col].astype(float)
    return subset[np.isfinite(observed) & np.isfinite(predicted)].copy()


def metric_row(frame: pd.DataFrame, group: str, target: str, model: str, observed_col: str, predicted_col: str) -> dict[str, Any] | None:
    subset = finite_frame(frame, observed_col, predicted_col)
    if len(subset) < 2:
        return None
    observed = subset[observed_col].astype(float)
    predicted = subset[predicted_col].astype(float)
    variance = float(observed.var())
    return {
        "group": group,
        "metric_target": target,
        "model": model,
        "n": int(len(subset)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)) if variance > 1e-12 else np.nan,
        "bias": float((predicted - observed).mean()),
        "pearson_r": float(observed.corr(predicted)) if float(predicted.std()) > 1e-12 else np.nan,
    }


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("all", predictions)]
    groups.extend((f"target_month_{int(month):02d}", group) for month, group in predictions.groupby("observed_target_month"))
    specs = [
        ("raw", "causal_graph_raw", "observed_raw", "predicted_raw"),
        ("raw", "seasonal_trend_baseline", "observed_raw", "expected_raw"),
        ("raw", "graph_parent_ridge_raw", "observed_raw", "ridge_raw"),
        ("residual", "causal_graph_residual", "observed_residual", "predicted_residual"),
        ("residual", "zero_residual_baseline", "observed_residual", "zero_residual"),
        ("residual", "graph_parent_ridge_residual", "observed_residual", "ridge_residual"),
    ]
    for group_name, group in groups:
        for metric_target, model, observed_col, predicted_col in specs:
            row = metric_row(group, group_name, metric_target, model, observed_col, predicted_col)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def rmse(values: pd.Series, predictions: pd.Series) -> float:
    return float(math.sqrt(mean_squared_error(values.astype(float), predictions.astype(float))))


def skill_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("raw", "causal_graph_raw", "seasonal_trend_baseline", "observed_raw", "predicted_raw", "expected_raw"),
        ("raw", "graph_parent_ridge_raw", "seasonal_trend_baseline", "observed_raw", "ridge_raw", "expected_raw"),
        ("residual", "causal_graph_residual", "zero_residual_baseline", "observed_residual", "predicted_residual", "zero_residual"),
        ("residual", "graph_parent_ridge_residual", "zero_residual_baseline", "observed_residual", "ridge_residual", "zero_residual"),
    ]
    for metric_target, model, baseline, observed_col, model_col, baseline_col in comparisons:
        subset = finite_frame(predictions, observed_col, model_col)
        subset = finite_frame(subset, observed_col, baseline_col)
        if len(subset) < 2:
            continue
        model_rmse = rmse(subset[observed_col], subset[model_col])
        baseline_rmse = rmse(subset[observed_col], subset[baseline_col])
        rows.append(
            {
                "metric_target": metric_target,
                "model": model,
                "baseline": baseline,
                "n": int(len(subset)),
                "model_rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "rmse_skill": np.nan if baseline_rmse == 0 else 1.0 - model_rmse / baseline_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_group_lookup(frame: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (int(row), int(col)): group.copy()
        for (row, col), group in frame.groupby(["row", "col"], sort=True)
    }


def predict_one_graph(
    graph_row: Any,
    group_lookup: Mapping[tuple[int, int], pd.DataFrame],
    target: str,
    evaluation_year: int,
    months: Sequence[int],
    training_end_year: int,
    graph_window_size: int,
    effect_mode: str,
    min_monthly_train_samples: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = int(graph_row.row)
    col = int(graph_row.col)
    coefficients = graph_target_effect_coefficients(graph_row, target, effect_mode)
    parents = list(coefficients)
    diagnostic: dict[str, Any] = {
        "row": row,
        "col": col,
        "n_parents": len(parents),
        "parents": ",".join(parents),
        "status": "started",
        "n_predictions": 0,
    }
    if not parents:
        diagnostic["status"] = "no_target_parents"
        return [], diagnostic
    window = (
        group_lookup.get((row, col))
        if graph_window_size == 0
        else get_pixel_window_group((row, col), group_lookup, graph_window_size)
    )
    if window is None or window.empty:
        diagnostic["status"] = "no_window_data"
        return [], diagnostic
    needed = [target, *parents, "_target_raw", "_target_expected", "_target_observed_year", "_target_observed_month", "year", "month", "row", "col", "x", "y"]
    missing = [column for column in needed if column not in window.columns]
    if missing:
        diagnostic["status"] = "missing_columns"
        diagnostic["missing_columns"] = ",".join(missing)
        return [], diagnostic
    complete = window.dropna(subset=[target, *parents, "_target_raw", "_target_expected"]).copy()
    train = complete[
        (complete["year"].astype(int) <= training_end_year)
        & (complete["_target_observed_year"].astype(int) <= training_end_year)
    ].copy()
    diagnostic["n_train"] = int(len(train))
    diagnostic["n_complete"] = int(len(complete))
    if len(train) < min_monthly_train_samples:
        diagnostic["status"] = "too_few_train_samples"
        return [], diagnostic

    center = complete[(complete["row"] == row) & (complete["col"] == col)].copy()
    predictions: list[dict[str, Any]] = []
    for month in months:
        model_year, model_month = shift_year_month(evaluation_year, int(month), target_shift_from_residual_target(target))
        eval_rows = center[
            (center["year"].astype(int) == model_year)
            & (center["month"].astype(int) == model_month)
            & (center["_target_observed_year"].astype(int) == evaluation_year)
            & (center["_target_observed_month"].astype(int) == int(month))
        ].dropna(subset=[target, *parents, "_target_raw", "_target_expected"])
        if eval_rows.empty:
            diagnostic["status"] = "missing_evaluation_row"
            continue
        train_month = train[train["month"].astype(int) == model_month].dropna(subset=[target, *parents])
        if len(train_month) < min_monthly_train_samples:
            diagnostic["status"] = "too_few_monthly_train_samples"
            diagnostic["n_monthly_train"] = int(len(train_month))
            continue
        eval_row = eval_rows.iloc[0]
        parent_means = train_month[parents].mean()
        target_mean = float(train_month[target].mean())
        predicted_residual = target_mean + sum(
            float(coefficients[parent]) * (float(eval_row[parent]) - float(parent_means[parent]))
            for parent in parents
        )
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(
            train_month[parents].astype(float).sub(parent_means),
            train_month[target].astype(float) - target_mean,
        )
        ridge_residual = target_mean + float(
            ridge.predict(
                pd.DataFrame(
                    [{parent: float(eval_row[parent]) - float(parent_means[parent]) for parent in parents}],
                    columns=parents,
                )
            )[0]
        )
        expected_raw = float(eval_row["_target_expected"])
        observed_raw = float(eval_row["_target_raw"])
        observed_residual = float(eval_row[target])
        predictions.append(
            {
                "row": row,
                "col": col,
                "longitude": float(eval_row["x"]),
                "latitude": float(eval_row["y"]),
                "model_year": int(model_year),
                "model_month": int(model_month),
                "observed_target_year": int(evaluation_year),
                "observed_target_month": int(month),
                "target": target,
                "observed_raw": observed_raw,
                "expected_raw": expected_raw,
                "predicted_raw": expected_raw + float(predicted_residual),
                "ridge_raw": expected_raw + float(ridge_residual),
                "observed_residual": observed_residual,
                "predicted_residual": float(predicted_residual),
                "zero_residual": 0.0,
                "ridge_residual": float(ridge_residual),
                "raw_residual": observed_raw - (expected_raw + float(predicted_residual)),
                "baseline_raw_residual": observed_raw - expected_raw,
                "n_monthly_train": int(len(train_month)),
                "parents": ",".join(parents),
            }
        )
    if predictions:
        diagnostic["status"] = "predicted"
        diagnostic["n_predictions"] = len(predictions)
    return predictions, diagnostic


def target_shift_from_residual_target(_target: str) -> int:
    # The caller passes months already based on the raw config's target shift.
    # This value is replaced in main by monkey-patching the global below to keep
    # multiprocessing-free row prediction simple.
    return TARGET_SHIFT


TARGET_SHIFT = 0


def plot_outputs(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty:
        return
    for observed_col, predicted_col, filename, label in [
        ("observed_raw", "predicted_raw", "observed_vs_predicted_raw.png", "Raw target"),
        ("observed_residual", "predicted_residual", "observed_vs_predicted_residual.png", "Target residual"),
    ]:
        subset = finite_frame(predictions, observed_col, predicted_col)
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(subset[observed_col], subset[predicted_col], s=3, alpha=0.35)
        lo = min(float(subset[observed_col].min()), float(subset[predicted_col].min()))
        hi = max(float(subset[observed_col].max()), float(subset[predicted_col].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"Observed vs predicted {label}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    for column, filename, title in [
        ("raw_residual", "raw_prediction_residual_map.png", "Observed - predicted raw target"),
        ("baseline_raw_residual", "raw_baseline_residual_map.png", "Observed - seasonal/trend baseline"),
    ]:
        values = predictions[column].astype(float)
        vmax = float(np.nanpercentile(np.abs(values), 98))
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(predictions["longitude"], predictions["latitude"], c=values, s=4, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax, shrink=0.75)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, Path(args.config))
    config = read_config(config_path)
    experiment_dir = config_path.parent
    name = str(config["name"])
    target = str(config["reference_var"])
    global TARGET_SHIFT
    TARGET_SHIFT = target_shift(config, target)

    output_dir = resolve_path(args.output_dir, Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = resolve_path(args.stage_dir, experiment_dir / f"real_holdout_stage_{args.evaluation_year}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    graph_db = resolve_path(args.graph_db, experiment_dir / f"{name}_residualized_graphs.duckdb")
    if not graph_db.exists():
        raise SystemExit(f"Missing graph DB: {graph_db}")

    if args.raw_ard_db:
        raw_db = resolve_path(args.raw_ard_db, Path(args.raw_ard_db))
        raw_table = args.raw_table or name
        raw_df = read_raw_ard(raw_db, raw_table)
    else:
        source_db = resolve_path(args.source_db, experiment_dir / f"{name}_source_db.duckdb")
        datasets, complete = complete_source_months(source_db, config, set(raw_variables(config)))
        months = choose_months(
            requested_months=args.months,
            evaluation_year=args.evaluation_year,
            target=target,
            target_lag=TARGET_SHIFT,
            datasets=datasets,
            complete=complete,
        )
        first_month = (2005, 1)
        last_needed = max(
            [(args.evaluation_year, month) for month in months]
            + [shift_year_month(args.evaluation_year, month, TARGET_SHIFT) for month in months],
            key=lambda item: month_index(*item),
        )
        required_months = month_range(first_month, last_needed)
        raw_df = build_raw_ard_from_source(
            source_db=source_db,
            config=config,
            required_months=required_months,
            output_db=stage_dir / f"{name}_real_holdout_raw.duckdb",
            output_table=name,
            workers=args.workers,
        )

    if args.months:
        months = sorted(set(int(month) for month in args.months))
    else:
        months = sorted(
            int(month)
            for month in raw_df.loc[raw_df["year"].astype(int) == args.evaluation_year, "month"].dropna().unique()
        )

    shifted, residual_models, residual_target = prepare_shifted_residual_frame(
        raw_df=raw_df,
        config=config,
        training_end_year=args.training_end_year,
    )
    residual_models.to_csv(output_dir / "residualization_models.csv", index=False)

    graph_rows = load_graph_rows(graph_db, args.graph_table)
    group_lookup = build_group_lookup(shifted)
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for graph_row in tqdm(graph_rows.itertuples(index=False), total=len(graph_rows), desc="Validating graph pixels"):
        rows, diagnostic = predict_one_graph(
            graph_row=graph_row,
            group_lookup=group_lookup,
            target=residual_target,
            evaluation_year=args.evaluation_year,
            months=months,
            training_end_year=args.training_end_year,
            graph_window_size=args.graph_window_size,
            effect_mode=args.effect_mode,
            min_monthly_train_samples=args.min_monthly_train_samples,
            ridge_alpha=args.ridge_alpha,
        )
        predictions.extend(rows)
        diagnostics.append(diagnostic)

    predictions_df = pd.DataFrame(predictions)
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(output_dir / "diagnostics.csv", index=False)
    if predictions_df.empty:
        print("No predictions produced. Diagnostic status counts:", file=sys.stderr)
        print(diagnostics_df["status"].value_counts(dropna=False).to_string(), file=sys.stderr)
        raise SystemExit(1)

    metrics_df = metric_rows(predictions_df)
    skills_df = skill_rows(predictions_df)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    skills_df.to_csv(output_dir / "skill_scores.csv", index=False)
    plot_outputs(predictions_df, output_dir)

    report = [
        "# Real-Data Causal Holdout Validation",
        "",
        f"- Raw config: `{config_path}`",
        f"- Graph DB: `{graph_db}`",
        f"- Evaluation year: `{args.evaluation_year}`",
        f"- Training target cutoff year: `{args.training_end_year}`",
        f"- Target: `{target}`",
        f"- Residual target: `{residual_target}`",
        f"- Target shift: `{TARGET_SHIFT}`",
        f"- Months evaluated: `{', '.join(str(month) for month in months)}`",
        f"- Predictions: `{len(predictions_df)}`",
        "",
        "## Skill Scores",
        "",
        "```text",
        skills_df.to_string(index=False),
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        metrics_df.to_string(index=False),
        "```",
        "",
        "## Diagnostics",
        "",
        "```text",
        diagnostics_df["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
        .to_string(index=False),
        "```",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")

    print("")
    print(skills_df.to_string(index=False))
    print("")
    print(metrics_df[metrics_df["group"] == "all"].to_string(index=False))
    print("")
    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
