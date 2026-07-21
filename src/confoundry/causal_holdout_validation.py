"""Validate historical causal graph models against held-out observations.

This command performs a falsifiable temporal holdout test.  The default mode is
``response``:

1. Load the existing graph database produced from historical data.
2. Load the ARD time series, applying the same configured temporal shifts used
   during graph discovery.
3. For each graph pixel, use the learned graph effects into the target variable.
4. Estimate historical same-month baseline values from years before the
   evaluation year.
5. Propagate held-out driver departures through the learned graph effects.
6. Compare predicted target responses against observed held-out responses.

The ``level`` mode instead predicts raw held-out target values.  The graph
database is never rebuilt or modified by this command.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

from confoundry.analysis_helpers import ensure_identifier, require_files, write_dataframe_table
from confoundry.landcover_helpers import load_graph_rows
from confoundry.per_pixel_graph_discovery import get_pixel_window_group, parse_columns


PixelKey = tuple[int, int]
COLORBAR_KWARGS = {
    "shrink": 0.72,
    "fraction": 0.035,
    "pad": 0.025,
}


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


def total_effect_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Compute linear total effects implied by an adjacency matrix."""
    identity = np.eye(adjacency.shape[0], dtype=float)
    try:
        return np.linalg.solve(identity - adjacency, identity) - identity
    except np.linalg.LinAlgError:
        return np.full_like(adjacency, np.nan, dtype=float)


def graph_target_parents(graph_row: Any, target: str) -> list[str]:
    """Return parent variable names with nonzero consensus edges into target."""
    return list(graph_target_parent_coefficients(graph_row, target))


def graph_target_parent_coefficients(graph_row: Any, target: str) -> dict[str, float]:
    """Return consensus-graph parent coefficients into target."""
    return graph_target_effect_coefficients(graph_row, target, "direct")


def graph_target_effect_coefficients(
    graph_row: Any,
    target: str,
    effect_mode: str,
) -> dict[str, float]:
    """Return graph coefficients/effects from source variables into target."""
    variables = list(json.loads(graph_row.variable_names_json))
    if target not in variables:
        return {}
    target_idx = variables.index(target)
    matrix = parse_consensus_matrix(graph_row)
    if effect_mode == "direct":
        effects = matrix
    elif effect_mode == "total":
        effects = total_effect_matrix(matrix)
    else:
        raise ValueError(f"Unknown effect_mode: {effect_mode!r}")
    return {
        variable: float(effects[target_idx, source_idx])
        for source_idx, variable in enumerate(variables)
        if (
            source_idx != target_idx
            and np.isfinite(effects[target_idx, source_idx])
            and effects[target_idx, source_idx] != 0.0
        )
    }


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
    prediction_mode: str,
    effect_mode: str,
    fit_mode: str,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fit one local structural equation and predict held-out observations."""
    row = int(graph_row.row)
    col = int(graph_row.col)
    parent_coefficients = graph_target_effect_coefficients(
        graph_row,
        target,
        "direct" if fit_mode == "ridge" else effect_mode,
    )
    parents = list(parent_coefficients)
    diagnostic: dict[str, Any] = {
        "row": row,
        "col": col,
        "n_parents": int(len(parents)),
        "parents": ",".join(parents),
        "status": "started",
        "n_train": 0,
        "n_predictions": 0,
        "prediction_mode": prediction_mode,
        "effect_mode": effect_mode,
        "fit_mode": fit_mode,
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

    if prediction_mode == "response" and fit_mode != "adjacency":
        diagnostic["status"] = "response_requires_adjacency_fit"
        return [], [], diagnostic

    if fit_mode == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=ridge_alpha)
        model.fit(train[parents], train[target])
        coefficients = {
            parent: float(coefficient)
            for parent, coefficient in zip(parents, model.coef_, strict=True)
        }
        intercept = float(model.intercept_)
    elif fit_mode == "adjacency":
        coefficients = dict(parent_coefficients)
        parent_means = train[parents].mean()
        intercept = float(
            train[target].mean()
            - sum(coefficients[parent] * parent_means[parent] for parent in parents)
        )
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode!r}")

    predictions: list[dict[str, Any]] = []
    coefficient_rows = [
        {
            "row": row,
            "col": col,
            "target": target,
            "parent": parent,
            "coefficient": float(coefficients[parent]),
            "fit_mode": fit_mode,
            "prediction_mode": prediction_mode,
            "effect_mode": effect_mode,
            "intercept": intercept,
            "n_train": int(len(train)),
        }
        for parent in parents
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
        climatology = float(train_by_month.get(model_month, train_mean))
        observed = float(eval_row[target])
        if prediction_mode == "level":
            predicted_response = np.nan
            observed_response = observed - climatology
            predicted = float(
                intercept
                + sum(
                    coefficients[parent] * float(eval_row[parent])
                    for parent in parents
                )
            )
        elif prediction_mode == "response":
            train_month = train[train["month"] == model_month].copy()
            if len(train_month) < min_train_samples:
                diagnostic["status"] = "too_few_monthly_train_samples"
                diagnostic["n_monthly_train"] = int(len(train_month))
                continue
            parent_means = train_month[parents].mean()
            target_mean = float(train_month[target].mean())
            predicted_response = float(
                sum(
                    coefficients[parent]
                    * (float(eval_row[parent]) - float(parent_means[parent]))
                    for parent in parents
                )
            )
            observed_response = float(observed - target_mean)
            climatology = target_mean
            predicted = float(target_mean + predicted_response)
        else:
            raise ValueError(f"Unknown prediction_mode: {prediction_mode!r}")
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
                "prediction_minus_climatology": predicted - climatology,
                "observed_response": observed_response,
                "predicted_response": predicted_response,
                "residual": observed - predicted,
                "climatology_residual": observed - climatology,
                "n_train": int(len(train)),
                "parents": ",".join(parents),
                "fit_mode": fit_mode,
                "prediction_mode": prediction_mode,
                "effect_mode": effect_mode,
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
                    "metric_target": "level",
                    "model": model_name,
                    "n": int(len(group)),
                    "mae": float(mean_absolute_error(observed, predicted)),
                    "rmse": float(rmse),
                    "r2": float(r2_score(observed, predicted)),
                    "bias": float((predicted - observed).mean()),
                }
            )
        if {
            "observed_response",
            "predicted_response",
        }.issubset(group.columns):
            response_group = group.dropna(
                subset=["observed_response", "predicted_response"]
            )
            if len(response_group) >= 2:
                observed_response = response_group["observed_response"].astype(float)
                for model_name, values in [
                    (
                        "causal_graph_response",
                        response_group["predicted_response"].astype(float),
                    ),
                    (
                        "zero_response",
                        pd.Series(0.0, index=response_group.index),
                    ),
                ]:
                    rmse = math.sqrt(
                        mean_squared_error(observed_response, values)
                    )
                    rows.append(
                        {
                            "group": group_name,
                            "metric_target": "response",
                            "model": model_name,
                            "n": int(len(response_group)),
                            "mae": float(
                                mean_absolute_error(observed_response, values)
                            ),
                            "rmse": float(rmse),
                            "r2": float(r2_score(observed_response, values)),
                            "bias": float((values - observed_response).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def pixel_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute held-out metrics separately for each graph pixel."""
    rows: list[dict[str, Any]] = []
    for (row, col), group in predictions.groupby(["row", "col"], sort=True):
        longitude = float(group["longitude"].mean())
        latitude = float(group["latitude"].mean())
        observed = group["observed"].astype(float)
        if len(group) >= 2:
            for model_name, column in [
                ("causal_graph_sem", "predicted"),
                ("historical_climatology", "climatology"),
            ]:
                predicted = group[column].astype(float)
                rmse = math.sqrt(mean_squared_error(observed, predicted))
                rows.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "longitude": longitude,
                        "latitude": latitude,
                        "metric_target": "level",
                        "model": model_name,
                        "n": int(len(group)),
                        "mae": float(mean_absolute_error(observed, predicted)),
                        "rmse": float(rmse),
                        "r2": float(r2_score(observed, predicted)),
                        "bias": float((predicted - observed).mean()),
                    }
                )

        if {
            "observed_response",
            "predicted_response",
        }.issubset(group.columns):
            response_group = group.dropna(
                subset=["observed_response", "predicted_response"]
            )
            if len(response_group) >= 2:
                observed_response = response_group["observed_response"].astype(float)
                for model_name, values in [
                    (
                        "causal_graph_response",
                        response_group["predicted_response"].astype(float),
                    ),
                    (
                        "zero_response",
                        pd.Series(0.0, index=response_group.index),
                    ),
                ]:
                    rmse = math.sqrt(
                        mean_squared_error(observed_response, values)
                    )
                    rows.append(
                        {
                            "row": int(row),
                            "col": int(col),
                            "longitude": longitude,
                            "latitude": latitude,
                            "metric_target": "response",
                            "model": model_name,
                            "n": int(len(response_group)),
                            "mae": float(
                                mean_absolute_error(observed_response, values)
                            ),
                            "rmse": float(rmse),
                            "r2": float(r2_score(observed_response, values)),
                            "bias": float((values - observed_response).mean()),
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


def plot_observed_vs_predicted_response(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed held-out response against graph-predicted response."""
    subset = predictions.dropna(
        subset=["observed_response", "predicted_response"]
    ).copy()
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(
        subset["observed_response"],
        subset["predicted_response"],
        s=5,
        alpha=0.35,
        label="Graph-predicted response",
    )
    values = pd.concat(
        [subset["observed_response"], subset["predicted_response"]],
        ignore_index=True,
    ).astype(float)
    lower = float(values.min())
    upper = float(values.max())
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axis.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Observed held-out NDVI response")
    axis.set_ylabel("Predicted held-out NDVI response")
    axis.set_title("Held-out causal response prediction")
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
    axis.set_title("Observed - predicted held-out NDVI")
    figure.colorbar(
        scatter,
        ax=axis,
        label="Observed - predicted NDVI",
        **COLORBAR_KWARGS,
    )
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_monthly_residual_maps(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Plot observed-minus-predicted maps separately for each target month."""
    for month, subset in predictions.groupby("observed_target_month", sort=True):
        if subset.empty:
            continue
        plot_residual_map(
            subset,
            output_dir / f"observed_minus_predicted_ndvi_month_{int(month):02d}.png",
        )


def plot_climatology_residual_map(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed minus historical-climatology baseline."""
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    vmax = float(np.nanpercentile(np.abs(predictions["climatology_residual"]), 98))
    scatter = axis.scatter(
        predictions["longitude"],
        predictions["latitude"],
        c=predictions["climatology_residual"],
        s=5,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.8,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Observed - historical climatology NDVI")
    figure.colorbar(
        scatter,
        ax=axis,
        label="Observed - climatology NDVI",
        **COLORBAR_KWARGS,
    )
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_climatology_difference_map(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot graph prediction minus historical climatology."""
    values = predictions["prediction_minus_climatology"].astype(float)
    vmax = float(np.nanpercentile(np.abs(values), 98))
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    scatter = axis.scatter(
        predictions["longitude"],
        predictions["latitude"],
        c=values,
        s=5,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.8,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Graph prediction - historical climatology NDVI")
    figure.colorbar(
        scatter,
        ax=axis,
        label="Predicted - climatology NDVI",
        **COLORBAR_KWARGS,
    )
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_monthly_prediction_climatology_difference_maps(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot prediction-minus-climatology maps separately for each target month."""
    for month, subset in predictions.groupby("observed_target_month", sort=True):
        if subset.empty:
            continue
        plot_prediction_climatology_difference_map(
            subset,
            output_dir / f"prediction_minus_climatology_ndvi_month_{int(month):02d}.png",
        )


def plot_monthly_climatology_residual_maps(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot observed-minus-climatology maps separately for each target month."""
    for month, subset in predictions.groupby("observed_target_month", sort=True):
        if subset.empty:
            continue
        plot_climatology_residual_map(
            subset,
            output_dir / f"observed_minus_climatology_ndvi_month_{int(month):02d}.png",
        )


def plot_large_difference_map(
    predictions: pd.DataFrame,
    output_path: Path,
    residual_column: str,
    title: str,
    colorbar_label: str,
    percentile: float,
) -> None:
    """Plot only locations with large absolute residuals."""
    residuals = predictions[residual_column].astype(float)
    threshold = float(np.nanpercentile(np.abs(residuals), percentile))
    subset = predictions[np.abs(residuals) >= threshold].copy()
    if subset.empty:
        return

    vmax = float(np.nanpercentile(np.abs(residuals), 98))
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    axis.scatter(
        predictions["longitude"],
        predictions["latitude"],
        s=2,
        color="lightgrey",
        alpha=0.35,
        label="All predictions",
    )
    scatter = axis.scatter(
        subset["longitude"],
        subset["latitude"],
        c=subset[residual_column],
        s=10,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.95,
        label=f"Top {100.0 - percentile:.0f}% absolute differences",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(f"{title}\n|difference| >= {threshold:.3f}")
    axis.legend(loc="best", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    figure.colorbar(
        scatter,
        ax=axis,
        label=colorbar_label,
        **COLORBAR_KWARGS,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_monthly_large_difference_maps(
    predictions: pd.DataFrame,
    output_dir: Path,
    residual_column: str,
    filename_prefix: str,
    title: str,
    colorbar_label: str,
    percentile: float,
) -> None:
    """Plot large-difference maps separately for each target month."""
    for month, subset in predictions.groupby("observed_target_month", sort=True):
        if subset.empty:
            continue
        plot_large_difference_map(
            subset,
            output_dir / f"{filename_prefix}_month_{int(month):02d}.png",
            residual_column=residual_column,
            title=title,
            colorbar_label=colorbar_label,
            percentile=percentile,
        )


def plot_observed_predicted_response_maps(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed and graph-predicted held-out responses side by side."""
    subset = predictions.dropna(
        subset=["observed_response", "predicted_response"]
    ).copy()
    if subset.empty:
        return
    values = pd.concat(
        [subset["observed_response"], subset["predicted_response"]],
        ignore_index=True,
    ).astype(float)
    vmax = float(np.nanpercentile(np.abs(values), 98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(values)))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, column, title in [
        (axes[0], "observed_response", "Observed NDVI response"),
        (axes[1], "predicted_response", "Graph-predicted NDVI response"),
    ]:
        scatter = axis.scatter(
            subset["longitude"],
            subset["latitude"],
            c=subset[column],
            s=5,
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.8,
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")

    figure.colorbar(
        scatter,
        ax=axes,
        label="NDVI response",
        **COLORBAR_KWARGS,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_observed_predicted_maps(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed and predicted held-out target values side by side."""
    values = pd.concat(
        [predictions["observed"], predictions["predicted"]],
        ignore_index=True,
    ).astype(float)
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, column, title in [
        (axes[0], "observed", "Observed held-out NDVI"),
        (axes[1], "predicted", "Predicted held-out NDVI"),
    ]:
        scatter = axis.scatter(
            predictions["longitude"],
            predictions["latitude"],
            c=predictions[column],
            s=5,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")

    figure.colorbar(
        scatter,
        ax=axes,
        label="NDVI",
        **COLORBAR_KWARGS,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
    metric_target: str = "level",
) -> None:
    """Plot overall error metrics for causal model and climatology baseline."""
    subset = metrics[metrics["group"] == "all"].copy()
    if "metric_target" in subset.columns:
        subset = subset[subset["metric_target"] == metric_target].copy()
    if subset.empty:
        return
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


def plot_r2_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
    metric_target: str = "level",
) -> None:
    """Plot overall held-out R2 for causal model and baseline."""
    subset = metrics[metrics["group"] == "all"].copy()
    if "metric_target" in subset.columns:
        subset = subset[subset["metric_target"] == metric_target].copy()
    if subset.empty:
        return
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


def plot_pixel_metric_map(
    pixel_metrics: pd.DataFrame,
    output_path: Path,
    metric: str,
    model: str,
    metric_target: str = "level",
) -> None:
    """Plot one per-pixel metric for one model."""
    subset = pixel_metrics[
        (pixel_metrics["model"] == model)
        & (pixel_metrics["metric_target"] == metric_target)
    ].dropna(subset=[metric])
    if subset.empty:
        return

    values = subset[metric].astype(float)
    if metric == "r2":
        vmin = float(np.nanpercentile(values, 2))
        vmax = float(np.nanpercentile(values, 98))
        cmap = "viridis"
    else:
        max_abs = float(np.nanpercentile(np.abs(values), 98))
        vmin = -max_abs if metric == "bias" else float(np.nanpercentile(values, 2))
        vmax = max_abs if metric == "bias" else float(np.nanpercentile(values, 98))
        cmap = "RdBu" if metric == "bias" else "magma"

    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    scatter = axis.scatter(
        subset["longitude"],
        subset["latitude"],
        c=values,
        s=5,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(f"Per-pixel {metric.upper()}: {model}")
    axis.set_aspect("equal", adjustable="box")
    figure.colorbar(
        scatter,
        ax=axis,
        label=metric,
        **COLORBAR_KWARGS,
    )
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
@click.option(
    "--prediction-mode",
    type=click.Choice(["level", "response"]),
    default="response",
    show_default=True,
    help=(
        "level predicts raw held-out NDVI; response predicts the held-out "
        "departure from historical same-month NDVI by propagating held-out "
        "driver departures through the graph."
    ),
)
@click.option(
    "--effect-mode",
    type=click.Choice(["direct", "total"]),
    default="total",
    show_default=True,
    help=(
        "Graph effects used for adjacency predictions. direct uses only direct "
        "edges into the target; total also propagates indirect paths."
    ),
)
@click.option(
    "--fit-mode",
    type=click.Choice(["adjacency", "ridge"]),
    default="adjacency",
    show_default=True,
    help=(
        "adjacency uses the saved consensus adjacency coefficients and only "
        "calibrates an intercept from historical means; ridge refits slopes "
        "using the graph-selected parents."
    ),
)
@click.option("--ridge-alpha", default=1.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option(
    "--large-difference-percentile",
    default=90.0,
    show_default=True,
    type=click.FloatRange(min=0.0, max=100.0),
    help="Absolute residual percentile used to highlight large-difference areas.",
)
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
    prediction_mode: str,
    effect_mode: str,
    fit_mode: str,
    ridge_alpha: float,
    large_difference_percentile: float,
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
    if prediction_mode == "response" and fit_mode != "adjacency":
        raise click.ClickException(
            "--prediction-mode response requires --fit-mode adjacency, because "
            "the response experiment propagates saved graph coefficients rather "
            "than refitting slopes."
        )

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
        desc="Evaluating causal holdout models",
    ):
        predictions, coefficients, diagnostic = fit_predict_one_graph(
            graph_row=graph_row,
            group_lookup=group_lookup,
            target=target,
            graph_window_size=graph_window_size,
            evaluation_rows=eval_rows,
            training_end_year=training_end_year,
            min_train_samples=min_train_samples,
            prediction_mode=prediction_mode,
            effect_mode=effect_mode,
            fit_mode=fit_mode,
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
    pixel_metrics_df = pixel_metric_rows(predictions_df)
    if pixel_metrics_df.empty:
        pixel_metrics_df = pd.DataFrame(
            columns=[
                "row",
                "col",
                "longitude",
                "latitude",
                "metric_target",
                "model",
                "n",
                "mae",
                "rmse",
                "r2",
                "bias",
            ]
        )

    predictions_df.to_csv(output_dir / "causal_holdout_predictions.csv", index=False)
    coefficients_df.to_csv(output_dir / "causal_holdout_coefficients.csv", index=False)
    metrics_df.to_csv(output_dir / "causal_holdout_metrics.csv", index=False)
    pixel_metrics_df.to_csv(
        output_dir / "causal_holdout_pixel_metrics.csv",
        index=False,
    )
    diagnostics_df.to_csv(output_dir / "causal_holdout_diagnostics.csv", index=False)
    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, predictions_df, "causal_holdout_predictions")
        write_dataframe_table(con, coefficients_df, "causal_holdout_coefficients")
        write_dataframe_table(con, metrics_df, "causal_holdout_metrics")
        if not pixel_metrics_df.empty:
            write_dataframe_table(
                con,
                pixel_metrics_df,
                "causal_holdout_pixel_metrics",
            )
        write_dataframe_table(con, diagnostics_df, "causal_holdout_diagnostics")
    finally:
        con.close()

    plot_observed_vs_predicted(
        predictions_df,
        output_dir / "observed_vs_predicted.png",
    )
    plot_observed_predicted_maps(
        predictions_df,
        output_dir / "observed_predicted_ndvi_maps.png",
    )
    plot_observed_vs_predicted_response(
        predictions_df,
        output_dir / "observed_vs_predicted_response.png",
    )
    plot_observed_predicted_response_maps(
        predictions_df,
        output_dir / "observed_predicted_response_maps.png",
    )
    plot_residual_map(predictions_df, output_dir / "causal_residual_map.png")
    plot_residual_map(
        predictions_df,
        output_dir / "observed_minus_predicted_ndvi_map.png",
    )
    plot_monthly_residual_maps(predictions_df, output_dir)
    plot_climatology_residual_map(
        predictions_df,
        output_dir / "observed_minus_climatology_ndvi_map.png",
    )
    plot_monthly_climatology_residual_maps(predictions_df, output_dir)
    plot_prediction_climatology_difference_map(
        predictions_df,
        output_dir / "prediction_minus_climatology_ndvi_map.png",
    )
    plot_monthly_prediction_climatology_difference_maps(predictions_df, output_dir)
    plot_large_difference_map(
        predictions_df,
        output_dir / "large_observed_minus_predicted_ndvi_map.png",
        residual_column="residual",
        title="Large graph-model NDVI differences",
        colorbar_label="Observed - predicted NDVI",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="residual",
        filename_prefix="large_observed_minus_predicted_ndvi",
        title="Large graph-model NDVI differences",
        colorbar_label="Observed - predicted NDVI",
        percentile=large_difference_percentile,
    )
    plot_large_difference_map(
        predictions_df,
        output_dir / "large_observed_minus_climatology_ndvi_map.png",
        residual_column="climatology_residual",
        title="Large climatology-baseline NDVI differences",
        colorbar_label="Observed - climatology NDVI",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="climatology_residual",
        filename_prefix="large_observed_minus_climatology_ndvi",
        title="Large climatology-baseline NDVI differences",
        colorbar_label="Observed - climatology NDVI",
        percentile=large_difference_percentile,
    )
    plot_large_difference_map(
        predictions_df,
        output_dir / "large_prediction_minus_climatology_ndvi_map.png",
        residual_column="prediction_minus_climatology",
        title="Large graph-vs-climatology NDVI differences",
        colorbar_label="Predicted - climatology NDVI",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="prediction_minus_climatology",
        filename_prefix="large_prediction_minus_climatology_ndvi",
        title="Large graph-vs-climatology NDVI differences",
        colorbar_label="Predicted - climatology NDVI",
        percentile=large_difference_percentile,
    )
    plot_metric_comparison(metrics_df, output_dir / "holdout_rmse.png")
    plot_r2_comparison(metrics_df, output_dir / "holdout_r2.png")
    plot_metric_comparison(
        metrics_df,
        output_dir / "holdout_response_rmse.png",
        metric_target="response",
    )
    plot_r2_comparison(
        metrics_df,
        output_dir / "holdout_response_r2.png",
        metric_target="response",
    )
    plot_pixel_metric_map(
        pixel_metrics_df,
        output_dir / "per_pixel_r2_map.png",
        metric="r2",
        model="causal_graph_sem",
        metric_target="level",
    )
    plot_pixel_metric_map(
        pixel_metrics_df,
        output_dir / "per_pixel_rmse_map.png",
        metric="rmse",
        model="causal_graph_sem",
        metric_target="level",
    )
    plot_pixel_metric_map(
        pixel_metrics_df,
        output_dir / "per_pixel_response_r2_map.png",
        metric="r2",
        model="causal_graph_response",
        metric_target="response",
    )

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
    click.echo(f"Per-pixel metric rows: {len(pixel_metrics_df):,}")
    if pixel_metrics_df.empty:
        click.echo(
            "Per-pixel R2 was not computed because each pixel needs at least "
            "two held-out predictions. Repeat --observed-target-month for "
            "multiple available months."
        )
    click.echo(f"Prediction mode: {prediction_mode}")
    click.echo(f"Effect mode: {effect_mode}")
    click.echo(f"Fit mode: {fit_mode}")
    overall_metrics = metrics_df[metrics_df["group"] == "all"].copy()
    if not overall_metrics.empty:
        click.echo(
            "Overall held-out R2: "
            + ", ".join(
                f"{row.metric_target}:{row.model}={row.r2:.3f}"
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
