"""Validate historical causal graph models against held-out observations.

This command performs a falsifiable temporal holdout test:

1. Load the existing graph database produced from historical data.
2. Load the ARD time series, applying the same configured temporal shifts used
   during graph discovery.
3. For each graph pixel, use the learned graph parents of the target variable
   as a structural equation.
4. Fit that equation on years before the evaluation year.
5. Predict raw observed target values in the evaluation year.
6. Compare predictions against the actual held-out observations and against a
   historical-climatology baseline.

The graph database is never rebuilt or modified by this command.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

from confoundry.analysis_helpers import ensure_identifier, require_files, write_dataframe_table
from confoundry.landcover_helpers import load_graph_rows
from confoundry.per_pixel_graph_discovery import get_pixel_window_group, parse_columns


PixelKey = tuple[int, int]


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and minimally validate an experiment config."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise click.ClickException("Experiment YAML must contain a mapping.")
    missing = [key for key in ["name", "columns"] if key not in config]
    if missing:
        raise click.ClickException(
            f"Configuration is missing required keys: {missing}"
        )
    return config


def target_shift(config: Mapping[str, Any], target: str) -> int:
    """Return configured temporal shift for a target variable."""
    for spec in config["columns"]:
        if str(spec["name"]) == target:
            return int(spec.get("shift", 0))
    raise click.ClickException(
        f"Target {target!r} is not present in config['columns']."
    )


def shift_year_month(year: int, month: int, delta_months: int) -> tuple[int, int]:
    """Shift a year/month pair by a number of months."""
    absolute = year * 12 + (month - 1) + delta_months
    return absolute // 12, absolute % 12 + 1


def evaluation_model_months(
    evaluation_year: int,
    observed_target_months: Sequence[int],
    configured_target_shift: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Map model row months to the held-out observed target months they predict."""
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for observed_month in observed_target_months:
        if observed_month < 1 or observed_month > 12:
            raise click.ClickException("Observed target months must lie in 1..12.")
        model_year, model_month = shift_year_month(
            evaluation_year,
            int(observed_month),
            configured_target_shift,
        )
        mapping[(model_year, model_month)] = (
            evaluation_year,
            int(observed_month),
        )
    return mapping


def load_shifted_ard(
    ard_db: Path,
    table: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Load ARD data and apply graph-discovery temporal shifts."""
    con = duckdb.connect(ard_db, read_only=True)
    try:
        df = con.execute(
            f"SELECT * FROM {ensure_identifier(table)}"
        ).fetchdf()
    finally:
        con.close()

    for required in ["row", "col", "year", "month", "x", "y"]:
        if required not in df.columns:
            raise click.ClickException(
                f"ARD table {table!r} is missing required column {required!r}."
            )

    shifted, labels, _label_lags = parse_columns(
        df=df,
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        column_specs=config["columns"],
    )
    return shifted, labels


def parse_consensus_matrix(graph_row: Any) -> np.ndarray:
    """Parse one graph-row consensus adjacency matrix."""
    return np.asarray(json.loads(graph_row.adjacency_consensus_json), dtype=float)


def graph_target_parents(graph_row: Any, target: str) -> list[str]:
    """Return parent variable names with nonzero consensus edges into target."""
    variables = list(json.loads(graph_row.variable_names_json))
    if target not in variables:
        return []
    target_idx = variables.index(target)
    matrix = parse_consensus_matrix(graph_row)
    parents = [
        variable
        for source_idx, variable in enumerate(variables)
        if source_idx != target_idx and matrix[target_idx, source_idx] != 0.0
    ]
    return parents


def build_group_lookup(shifted: pd.DataFrame) -> dict[PixelKey, pd.DataFrame]:
    """Group shifted ARD rows by pixel key."""
    groups = shifted.groupby(["row", "col"], sort=True)
    return {
        (int(row), int(col)): group.copy()
        for (row, col), group in groups
    }


def fit_predict_one_graph(
    graph_row: Any,
    group_lookup: dict[PixelKey, pd.DataFrame],
    target: str,
    graph_window_size: int,
    evaluation_rows: dict[tuple[int, int], tuple[int, int]],
    training_end_year: int,
    min_train_samples: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fit one local structural equation and predict held-out observations."""
    row = int(graph_row.row)
    col = int(graph_row.col)
    parents = graph_target_parents(graph_row, target)
    diagnostic: dict[str, Any] = {
        "row": row,
        "col": col,
        "n_parents": int(len(parents)),
        "parents": ",".join(parents),
        "status": "started",
        "n_train": 0,
        "n_predictions": 0,
    }
    if not parents:
        diagnostic["status"] = "no_target_parents"
        return [], [], diagnostic

    if graph_window_size == 0:
        window = group_lookup.get((row, col))
    else:
        window = get_pixel_window_group(
            pixel_key=(row, col),
            group_lookup=group_lookup,
            window_size=graph_window_size,
        )
    if window is None or window.empty:
        diagnostic["status"] = "no_window_data"
        return [], [], diagnostic

    needed = [target, *parents, "year", "month", "row", "col", "x", "y"]
    missing = [column for column in needed if column not in window.columns]
    if missing:
        diagnostic["status"] = "missing_columns"
        diagnostic["missing_columns"] = ",".join(missing)
        return [], [], diagnostic

    complete = window.dropna(subset=[target, *parents]).copy()
    train = complete[complete["year"] <= training_end_year].copy()
    train = train.dropna(subset=[target, *parents])
    diagnostic["n_complete"] = int(len(complete))
    diagnostic["n_train"] = int(len(train))
    if len(train) < min_train_samples:
        diagnostic["status"] = "too_few_train_samples"
        return [], [], diagnostic

    model = Ridge(alpha=ridge_alpha)
    model.fit(train[parents], train[target])

    predictions: list[dict[str, Any]] = []
    coefficient_rows = [
        {
            "row": row,
            "col": col,
            "target": target,
            "parent": parent,
            "coefficient": float(coefficient),
            "n_train": int(len(train)),
        }
        for parent, coefficient in zip(parents, model.coef_, strict=True)
    ]

    train_by_month = train.groupby("month")[target].mean()
    train_mean = float(train[target].mean())

    center_all_rows = window[
        (window["row"] == row) & (window["col"] == col)
    ].copy()
    center_rows = complete[
        (complete["row"] == row) & (complete["col"] == col)
    ].copy()
    for (model_year, model_month), (observed_year, observed_month) in evaluation_rows.items():
        raw_eval_rows = center_all_rows[
            (center_all_rows["year"] == model_year)
            & (center_all_rows["month"] == model_month)
        ]
        eval_rows = center_rows[
            (center_rows["year"] == model_year)
            & (center_rows["month"] == model_month)
        ].dropna(subset=[target, *parents])
        if eval_rows.empty:
            if raw_eval_rows.empty:
                diagnostic["status"] = "evaluation_row_absent"
            else:
                missing_values = [
                    column
                    for column in [target, *parents]
                    if raw_eval_rows[column].isna().all()
                ]
                diagnostic["status"] = "evaluation_values_missing"
                diagnostic["missing_evaluation_values"] = ",".join(
                    missing_values
                )
            continue
        eval_row = eval_rows.iloc[0]
        predicted = float(model.predict(eval_row[parents].to_frame().T)[0])
        climatology = float(train_by_month.get(model_month, train_mean))
        observed = float(eval_row[target])
        predictions.append(
            {
                "row": row,
                "col": col,
                "longitude": float(eval_row["x"]),
                "latitude": float(eval_row["y"]),
                "model_year": int(model_year),
                "model_month": int(model_month),
                "observed_target_year": int(observed_year),
                "observed_target_month": int(observed_month),
                "target": target,
                "observed": observed,
                "predicted": predicted,
                "climatology": climatology,
                "residual": observed - predicted,
                "climatology_residual": observed - climatology,
                "n_train": int(len(train)),
                "parents": ",".join(parents),
            }
        )

    diagnostic["n_predictions"] = int(len(predictions))
    if predictions:
        diagnostic["status"] = "predicted"
    elif diagnostic["status"] == "started":
        diagnostic["status"] = "missing_evaluation_rows"
    return predictions, coefficient_rows, diagnostic


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction metrics for causal model and climatology baseline."""
    rows: list[dict[str, Any]] = []
    group_specs = [("all", predictions)]
    group_specs.extend(
        (
            f"target_month_{int(month):02d}",
            group,
        )
        for month, group in predictions.groupby("observed_target_month")
    )

    for group_name, group in group_specs:
        if len(group) < 2:
            continue
        observed = group["observed"].astype(float)
        for model_name, column in [
            ("causal_graph_sem", "predicted"),
            ("historical_climatology", "climatology"),
        ]:
            predicted = group[column].astype(float)
            rmse = math.sqrt(mean_squared_error(observed, predicted))
            rows.append(
                {
                    "group": group_name,
                    "model": model_name,
                    "n": int(len(group)),
                    "mae": float(mean_absolute_error(observed, predicted)),
                    "rmse": float(rmse),
                    "r2": float(r2_score(observed, predicted)),
                    "bias": float((predicted - observed).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_observed_vs_predicted(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot observed held-out target against model predictions."""
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(
        predictions["observed"],
        predictions["climatology"],
        s=5,
        alpha=0.35,
        label="Historical climatology",
    )
    axis.scatter(
        predictions["observed"],
        predictions["predicted"],
        s=5,
        alpha=0.35,
        label="Causal graph SEM",
    )
    values = pd.concat(
        [
            predictions["observed"],
            predictions["predicted"],
            predictions["climatology"],
        ],
        ignore_index=True,
    ).astype(float)
    lower = float(values.min())
    upper = float(values.max())
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.set_xlabel("Observed held-out target")
    axis.set_ylabel("Predicted target")
    axis.set_title("Held-out causal prediction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_residual_map(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot spatial residuals for held-out causal predictions."""
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    vmax = float(np.nanpercentile(np.abs(predictions["residual"]), 98))
    scatter = axis.scatter(
        predictions["longitude"],
        predictions["latitude"],
        c=predictions["residual"],
        s=5,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.8,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Causal prediction residuals")
    figure.colorbar(scatter, ax=axis, label="Observed - predicted")
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot overall error metrics for causal model and climatology baseline."""
    subset = metrics[metrics["group"] == "all"].copy()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    positions = np.arange(len(subset))
    axis.bar(positions, subset["rmse"])
    axis.set_xticks(positions)
    axis.set_xticklabels(subset["model"], rotation=15, ha="right")
    axis.set_ylabel("RMSE")
    axis.set_title("Held-out raw-target prediction error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_r2_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot overall held-out R2 for causal model and baseline."""
    subset = metrics[metrics["group"] == "all"].copy()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    positions = np.arange(len(subset))
    axis.bar(positions, subset["r2"])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(subset["model"], rotation=15, ha="right")
    axis.set_ylabel("Held-out R2")
    axis.set_title("Unseen-year generalization")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--evaluation-year", required=True, type=int)
@click.option(
    "--observed-target-month",
    "observed_target_months",
    multiple=True,
    type=click.IntRange(min=1, max=12),
    default=(6, 7),
    show_default=True,
    help="Observed held-out target month. Repeat as needed.",
)
@click.option("--training-end-year", default=None, type=int)
@click.option("--target-variable", default=None)
@click.option("--graph-table", default="pixel_graphs", show_default=True)
@click.option("--graph-window-size", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--min-train-samples", default=30, show_default=True, type=click.IntRange(min=2))
@click.option("--ridge-alpha", default=1.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def validate_causal_holdout(
    config_path: Path,
    evaluation_year: int,
    observed_target_months: tuple[int, ...],
    training_end_year: int | None,
    target_variable: str | None,
    graph_table: str,
    graph_window_size: int,
    min_train_samples: int,
    ridge_alpha: float,
    output_dir: Path | None,
) -> None:
    """Validate graph-constrained structural equations on held-out observations."""
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])
    target = str(target_variable or config.get("reference_var") or "ndvi")
    configured_shift = target_shift(config, target)
    training_end_year = (
        evaluation_year - 1 if training_end_year is None else training_end_year
    )
    output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir / f"causal_holdout_{target}_{evaluation_year}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ard_db = experiment_dir / f"{experiment_name}_ard.duckdb"
    graph_db = experiment_dir / f"{experiment_name}_graphs.duckdb"
    output_db = output_dir / f"{experiment_name}_causal_holdout.duckdb"
    require_files([ard_db, graph_db])

    click.echo("Loading and shifting ARD time series...")
    shifted, labels = load_shifted_ard(ard_db, experiment_name, config)
    if target not in labels:
        raise click.ClickException(
            f"Target {target!r} is not in graph-discovery variables."
        )

    click.echo("Loading existing historical graph database...")
    graph_rows = load_graph_rows(graph_db, graph_table)
    group_lookup = build_group_lookup(shifted)
    eval_rows = evaluation_model_months(
        evaluation_year=evaluation_year,
        observed_target_months=observed_target_months,
        configured_target_shift=configured_shift,
    )

    all_predictions: list[dict[str, Any]] = []
    all_coefficients: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for graph_row in tqdm(
        graph_rows.itertuples(index=False),
        total=len(graph_rows),
        desc="Fitting causal holdout models",
    ):
        predictions, coefficients, diagnostic = fit_predict_one_graph(
            graph_row=graph_row,
            group_lookup=group_lookup,
            target=target,
            graph_window_size=graph_window_size,
            evaluation_rows=eval_rows,
            training_end_year=training_end_year,
            min_train_samples=min_train_samples,
            ridge_alpha=ridge_alpha,
        )
        all_predictions.extend(predictions)
        all_coefficients.extend(coefficients)
        diagnostics.append(diagnostic)

    predictions_df = pd.DataFrame(all_predictions)
    coefficients_df = pd.DataFrame(all_coefficients)
    diagnostics_df = pd.DataFrame(diagnostics)
    if predictions_df.empty:
        diagnostics_df.to_csv(
            output_dir / "causal_holdout_diagnostics.csv",
            index=False,
        )
        raise click.ClickException(
            "No held-out predictions were produced. Check graph parents, "
            "target months, training years, and available evaluation data."
        )
    metrics_df = metric_rows(predictions_df)

    predictions_df.to_csv(output_dir / "causal_holdout_predictions.csv", index=False)
    coefficients_df.to_csv(output_dir / "causal_holdout_coefficients.csv", index=False)
    metrics_df.to_csv(output_dir / "causal_holdout_metrics.csv", index=False)
    diagnostics_df.to_csv(output_dir / "causal_holdout_diagnostics.csv", index=False)
    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, predictions_df, "causal_holdout_predictions")
        write_dataframe_table(con, coefficients_df, "causal_holdout_coefficients")
        write_dataframe_table(con, metrics_df, "causal_holdout_metrics")
        write_dataframe_table(con, diagnostics_df, "causal_holdout_diagnostics")
    finally:
        con.close()

    plot_observed_vs_predicted(
        predictions_df,
        output_dir / "observed_vs_predicted.png",
    )
    plot_residual_map(predictions_df, output_dir / "causal_residual_map.png")
    plot_metric_comparison(metrics_df, output_dir / "holdout_rmse.png")
    plot_r2_comparison(metrics_df, output_dir / "holdout_r2.png")

    click.echo("")
    click.echo("Causal holdout validation complete.")
    status_counts = diagnostics_df["status"].value_counts()
    click.echo(
        "Graph-pixel status counts: "
        + ", ".join(
            f"{status}={count}" for status, count in status_counts.items()
        )
    )
    click.echo(f"Predictions: {len(predictions_df):,}")
    overall_metrics = metrics_df[metrics_df["group"] == "all"].copy()
    if not overall_metrics.empty:
        click.echo(
            "Overall held-out R2: "
            + ", ".join(
                f"{row.model}={row.r2:.3f}"
                for row in overall_metrics.itertuples(index=False)
            )
        )
    click.echo(f"Target shift: {configured_shift}")
    click.echo(
        "Observed target months: "
        + ", ".join(str(month) for month in observed_target_months)
    )
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Output database: {output_db}")


if __name__ == "__main__":
    validate_causal_holdout()
