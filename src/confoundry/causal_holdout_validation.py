"""Publication-oriented validation of causal graph models on held-out years.

This command performs a falsifiable temporal hindcast.  It keeps a previously
learned graph database fixed, applies the same configured temporal shifts used
during graph discovery, and evaluates held-out target observations without
refitting the causal graph.  The default ``response`` mode tests the most
causally meaningful claim: whether departures in learned driver variables
explain departures in the held-out target relative to same-month historical
conditions.

To make the result suitable for a paper rather than only exploratory analysis,
the command writes:

* held-out predictions for one or more evaluation years;
* strong baselines: historical climatology, persistence, and a ridge response
  model using the same graph-selected parents;
* paired spatial-block bootstrap skill scores with confidence intervals;
* diagnostic maps, per-pixel metrics, and a Markdown validation report.

The graph database is never rebuilt or modified by this command.  For a strict
publication-grade test, use a graph database trained only on years before the
first evaluation year.
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


def target_label(predictions: pd.DataFrame) -> str:
    """Return a readable target label for plot text."""
    if "target" in predictions.columns and not predictions["target"].dropna().empty:
        return str(predictions["target"].dropna().iloc[0])
    return "target"


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


def config_value(
    config: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Read a config value from holdout/analysis/graph-discovery sections."""
    for section_name in ["causal_holdout", "analysis", "graph_discovery"]:
        section = config.get(section_name) or {}
        if not isinstance(section, Mapping):
            raise click.ClickException(
                f"config[{section_name!r}] must be a mapping when present."
            )
        if key in section:
            return section[key]
    return config.get(key, default)


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


def evaluation_model_months_for_years(
    evaluation_years: Sequence[int],
    observed_target_months: Sequence[int],
    configured_target_shift: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Map model row months to held-out target months for all evaluation years."""
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for year in evaluation_years:
        year_mapping = evaluation_model_months(
            evaluation_year=int(year),
            observed_target_months=observed_target_months,
            configured_target_shift=configured_target_shift,
        )
        overlaps = sorted(set(mapping) & set(year_mapping))
        if overlaps:
            raise click.ClickException(
                "Evaluation-year/month choices map to duplicate shifted model "
                f"rows: {overlaps}."
            )
        mapping.update(year_mapping)
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
            ridge_predicted_response = np.nan
            ridge_prediction = np.nan
            persistence_predicted_response = np.nan
            persistence_prediction = np.nan
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
            ridge_predicted_response = np.nan
            ridge_prediction = np.nan
            if len(train_month) >= min_train_samples:
                from sklearn.linear_model import Ridge

                ridge_response_model = Ridge(alpha=ridge_alpha)
                ridge_response_model.fit(
                    train_month[parents].astype(float).sub(parent_means),
                    train_month[target].astype(float) - target_mean,
                )
                ridge_predicted_response = float(
                    ridge_response_model.predict(
                        pd.DataFrame(
                            [
                                {
                                    parent: float(eval_row[parent])
                                    - float(parent_means[parent])
                                    for parent in parents
                                }
                            ],
                            columns=parents,
                        )
                    )[0]
                )
                ridge_prediction = float(target_mean + ridge_predicted_response)

            previous_rows = center_rows[
                (center_rows["year"] == int(model_year) - 1)
                & (center_rows["month"] == model_month)
            ].dropna(subset=[target])
            if previous_rows.empty:
                persistence_predicted_response = np.nan
                persistence_prediction = np.nan
            else:
                persistence_predicted_response = float(
                    previous_rows.iloc[0][target] - target_mean
                )
                persistence_prediction = float(
                    target_mean + persistence_predicted_response
                )
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
                "ridge_prediction": ridge_prediction,
                "persistence_prediction": persistence_prediction,
                "prediction_minus_climatology": predicted - climatology,
                "observed_response": observed_response,
                "predicted_response": predicted_response,
                "zero_predicted_response": 0.0,
                "ridge_predicted_response": ridge_predicted_response,
                "persistence_predicted_response": persistence_predicted_response,
                "residual": observed - predicted,
                "climatology_residual": observed - climatology,
                "ridge_residual": observed - ridge_prediction,
                "persistence_residual": observed - persistence_prediction,
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


def prediction_model_specs(metric_target: str) -> list[tuple[str, str]]:
    """Return model-name and prediction-column pairs for a metric target."""
    if metric_target == "level":
        return [
            ("causal_graph_sem", "predicted"),
            ("historical_climatology", "climatology"),
            ("graph_parent_ridge_level", "ridge_prediction"),
            ("persistence_level", "persistence_prediction"),
        ]
    if metric_target == "response":
        return [
            ("causal_graph_response", "predicted_response"),
            ("zero_response", "zero_predicted_response"),
            ("graph_parent_ridge_response", "ridge_predicted_response"),
            ("persistence_response", "persistence_predicted_response"),
        ]
    raise ValueError(f"Unknown metric target: {metric_target!r}")


def finite_prediction_frame(
    frame: pd.DataFrame,
    observed_col: str,
    predicted_col: str,
) -> pd.DataFrame:
    """Return rows with finite observed and predicted values."""
    if observed_col not in frame.columns or predicted_col not in frame.columns:
        return pd.DataFrame(columns=list(frame.columns))
    subset = frame.dropna(subset=[observed_col, predicted_col]).copy()
    if subset.empty:
        return subset
    observed = subset[observed_col].astype(float)
    predicted = subset[predicted_col].astype(float)
    return subset[np.isfinite(observed) & np.isfinite(predicted)].copy()


def safe_r2(observed: pd.Series, predicted: pd.Series) -> float:
    """Compute R2, returning NaN when undefined."""
    variance = float(observed.var())
    if len(observed) < 2 or not np.isfinite(variance) or variance <= 1e-12:
        return np.nan
    return float(r2_score(observed, predicted))


def safe_pearson(observed: pd.Series, predicted: pd.Series) -> float:
    """Compute Pearson correlation, returning NaN when undefined."""
    if len(observed) < 2:
        return np.nan
    observed_std = float(observed.std())
    predicted_std = float(predicted.std())
    if (
        not np.isfinite(observed_std)
        or not np.isfinite(predicted_std)
        or observed_std <= 1e-12
        or predicted_std <= 1e-12
    ):
        return np.nan
    return float(observed.corr(predicted))


def metric_record(
    frame: pd.DataFrame,
    group_name: str,
    metric_target: str,
    model_name: str,
    observed_col: str,
    predicted_col: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute one metric row for one model and group."""
    subset = finite_prediction_frame(frame, observed_col, predicted_col)
    if len(subset) < 2:
        return None

    observed = subset[observed_col].astype(float)
    predicted = subset[predicted_col].astype(float)
    rmse = math.sqrt(mean_squared_error(observed, predicted))
    row: dict[str, Any] = {
        "group": group_name,
        "metric_target": metric_target,
        "model": model_name,
        "n": int(len(subset)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(rmse),
        "r2": safe_r2(observed, predicted),
        "bias": float((predicted - observed).mean()),
        "pearson_r": safe_pearson(observed, predicted),
    }
    if extra:
        row.update(extra)
    return row


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute held-out metrics for causal predictions and all baselines."""
    rows: list[dict[str, Any]] = []
    group_specs = [("all", predictions)]
    group_specs.extend(
        (
            f"target_year_{int(year)}",
            group,
        )
        for year, group in predictions.groupby("observed_target_year")
    )
    group_specs.extend(
        (
            f"target_month_{int(month):02d}",
            group,
        )
        for month, group in predictions.groupby("observed_target_month")
    )

    for group_name, group in group_specs:
        for model_name, column in prediction_model_specs("level"):
            row = metric_record(
                group,
                group_name,
                "level",
                model_name,
                "observed",
                column,
            )
            if row is not None:
                rows.append(row)
        for model_name, column in prediction_model_specs("response"):
            row = metric_record(
                group,
                group_name,
                "response",
                model_name,
                "observed_response",
                column,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def pixel_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute held-out metrics separately for each graph pixel."""
    rows: list[dict[str, Any]] = []
    for (row, col), group in predictions.groupby(["row", "col"], sort=True):
        extra = {
            "row": int(row),
            "col": int(col),
            "longitude": float(group["longitude"].mean()),
            "latitude": float(group["latitude"].mean()),
        }
        for model_name, column in prediction_model_specs("level"):
            metric = metric_record(
                group,
                "pixel",
                "level",
                model_name,
                "observed",
                column,
                extra=extra,
            )
            if metric is not None:
                rows.append(metric)
        for model_name, column in prediction_model_specs("response"):
            metric = metric_record(
                group,
                "pixel",
                "response",
                model_name,
                "observed_response",
                column,
                extra=extra,
            )
            if metric is not None:
                rows.append(metric)
    return pd.DataFrame(rows)


def rmse_array(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error for NumPy arrays."""
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def mae_array(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error for NumPy arrays."""
    return float(np.mean(np.abs(predicted - observed)))


def bootstrap_summary(values: Sequence[float], ci: float) -> dict[str, float]:
    """Summarize bootstrap values with a central confidence interval."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "boot_mean": np.nan,
            "boot_sd": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
        }
    alpha = 1.0 - ci
    return {
        "boot_mean": float(np.mean(array)),
        "boot_sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "boot_ci_low": float(np.quantile(array, alpha / 2.0)),
        "boot_ci_high": float(np.quantile(array, 1.0 - alpha / 2.0)),
    }


def spatial_block_arrays(
    frame: pd.DataFrame,
    observed_col: str,
    model_col: str,
    baseline_col: str,
    block_size: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split paired predictions into spatial row/col blocks."""
    subset = finite_prediction_frame(frame, observed_col, model_col)
    subset = finite_prediction_frame(subset, observed_col, baseline_col)
    if subset.empty:
        return []

    if block_size < 1:
        raise click.ClickException("--spatial-block-size must be >= 1.")

    work = subset.copy()
    work["_validation_block_row"] = np.floor(
        work["row"].astype(float) / float(block_size)
    ).astype(int)
    work["_validation_block_col"] = np.floor(
        work["col"].astype(float) / float(block_size)
    ).astype(int)

    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _block, group in work.groupby(
        ["_validation_block_row", "_validation_block_col"],
        sort=True,
    ):
        arrays.append(
            (
                group[observed_col].astype(float).to_numpy(),
                group[model_col].astype(float).to_numpy(),
                group[baseline_col].astype(float).to_numpy(),
            )
        )
    return arrays


def paired_skill_record(
    predictions: pd.DataFrame,
    metric_target: str,
    model_name: str,
    model_col: str,
    baseline_name: str,
    baseline_col: str,
    *,
    n_bootstrap: int,
    ci: float,
    block_size: int,
    random_seed: int,
) -> dict[str, Any] | None:
    """Compute paired model-vs-baseline skill with spatial-block bootstrap CIs."""
    observed_col = "observed" if metric_target == "level" else "observed_response"
    arrays = spatial_block_arrays(
        predictions,
        observed_col=observed_col,
        model_col=model_col,
        baseline_col=baseline_col,
        block_size=block_size,
    )
    if not arrays:
        return None

    observed = np.concatenate([item[0] for item in arrays])
    model = np.concatenate([item[1] for item in arrays])
    baseline = np.concatenate([item[2] for item in arrays])
    if len(observed) < 2:
        return None

    model_rmse = rmse_array(observed, model)
    baseline_rmse = rmse_array(observed, baseline)
    model_mae = mae_array(observed, model)
    baseline_mae = mae_array(observed, baseline)
    if baseline_rmse == 0.0 or baseline_mae == 0.0:
        return None

    rng = np.random.default_rng(random_seed)
    rmse_skill_samples: list[float] = []
    mae_skill_samples: list[float] = []
    delta_rmse_samples: list[float] = []
    delta_mae_samples: list[float] = []
    block_count = len(arrays)
    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, block_count, size=block_count)
        boot_observed = np.concatenate([arrays[index][0] for index in sample_indices])
        boot_model = np.concatenate([arrays[index][1] for index in sample_indices])
        boot_baseline = np.concatenate([arrays[index][2] for index in sample_indices])
        boot_model_rmse = rmse_array(boot_observed, boot_model)
        boot_baseline_rmse = rmse_array(boot_observed, boot_baseline)
        boot_model_mae = mae_array(boot_observed, boot_model)
        boot_baseline_mae = mae_array(boot_observed, boot_baseline)
        if boot_baseline_rmse != 0.0:
            rmse_skill_samples.append(1.0 - boot_model_rmse / boot_baseline_rmse)
            delta_rmse_samples.append(boot_model_rmse - boot_baseline_rmse)
        if boot_baseline_mae != 0.0:
            mae_skill_samples.append(1.0 - boot_model_mae / boot_baseline_mae)
            delta_mae_samples.append(boot_model_mae - boot_baseline_mae)

    rmse_skill = 1.0 - model_rmse / baseline_rmse
    mae_skill = 1.0 - model_mae / baseline_mae
    delta_rmse = model_rmse - baseline_rmse
    delta_mae = model_mae - baseline_mae
    delta_rmse_array = np.asarray(delta_rmse_samples, dtype=float)
    delta_rmse_array = delta_rmse_array[np.isfinite(delta_rmse_array)]

    row: dict[str, Any] = {
        "metric_target": metric_target,
        "model": model_name,
        "baseline": baseline_name,
        "n": int(len(observed)),
        "n_spatial_blocks": int(block_count),
        "spatial_block_size": int(block_size),
        "n_bootstrap": int(n_bootstrap),
        "model_rmse": model_rmse,
        "baseline_rmse": baseline_rmse,
        "rmse_skill": rmse_skill,
        "delta_rmse": delta_rmse,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_skill": mae_skill,
        "delta_mae": delta_mae,
        "bootstrap_p_delta_rmse_lt_zero": (
            float(np.mean(delta_rmse_array < 0.0))
            if delta_rmse_array.size
            else np.nan
        ),
    }
    for prefix, values in [
        ("rmse_skill", rmse_skill_samples),
        ("mae_skill", mae_skill_samples),
        ("delta_rmse", delta_rmse_samples),
        ("delta_mae", delta_mae_samples),
    ]:
        summary = bootstrap_summary(values, ci)
        row.update({f"{prefix}_{key}": value for key, value in summary.items()})
    return row


def paired_skill_rows(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    ci: float,
    block_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Compute causal-model skill against all publishable baselines."""
    comparisons = [
        (
            "level",
            "causal_graph_sem",
            "predicted",
            "historical_climatology",
            "climatology",
        ),
        (
            "level",
            "causal_graph_sem",
            "predicted",
            "persistence_level",
            "persistence_prediction",
        ),
        (
            "level",
            "causal_graph_sem",
            "predicted",
            "graph_parent_ridge_level",
            "ridge_prediction",
        ),
        (
            "response",
            "causal_graph_response",
            "predicted_response",
            "zero_response",
            "zero_predicted_response",
        ),
        (
            "response",
            "causal_graph_response",
            "predicted_response",
            "persistence_response",
            "persistence_predicted_response",
        ),
        (
            "response",
            "causal_graph_response",
            "predicted_response",
            "graph_parent_ridge_response",
            "ridge_predicted_response",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for offset, comparison in enumerate(comparisons):
        row = paired_skill_record(
            predictions,
            *comparison,
            n_bootstrap=n_bootstrap,
            ci=ci,
            block_size=block_size,
            random_seed=random_seed + offset,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_observed_vs_predicted(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot observed held-out target against model predictions."""
    label = target_label(predictions)
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
    axis.set_xlabel(f"Observed held-out {label}")
    axis.set_ylabel(f"Predicted {label}")
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
    label = target_label(subset)
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
    axis.set_xlabel(f"Observed held-out {label} response")
    axis.set_ylabel(f"Predicted held-out {label} response")
    axis.set_title("Held-out causal response prediction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_residual_map(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot spatial residuals for held-out causal predictions."""
    label = target_label(predictions)
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
    axis.set_title(f"Observed - predicted held-out {label}")
    figure.colorbar(
        scatter,
        ax=axis,
        label=f"Observed - predicted {label}",
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
    label = target_label(predictions)
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
    axis.set_title(f"Observed - historical climatology {label}")
    figure.colorbar(
        scatter,
        ax=axis,
        label=f"Observed - climatology {label}",
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
    label = target_label(predictions)
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
    axis.set_title(f"Graph prediction - historical climatology {label}")
    figure.colorbar(
        scatter,
        ax=axis,
        label=f"Predicted - climatology {label}",
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
    label = target_label(subset)
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
        (axes[0], "observed_response", f"Observed {label} response"),
        (axes[1], "predicted_response", f"Graph-predicted {label} response"),
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
        label=f"{label} response",
        **COLORBAR_KWARGS,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_observed_predicted_maps(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed and predicted held-out target values side by side."""
    label = target_label(predictions)
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
        (axes[0], "observed", f"Observed held-out {label}"),
        (axes[1], "predicted", f"Predicted held-out {label}"),
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
        label=label,
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


def plot_skill_scores(skill_scores: pd.DataFrame, output_path: Path) -> None:
    """Plot paired RMSE skill scores with bootstrap confidence intervals."""
    if skill_scores.empty or "rmse_skill" not in skill_scores.columns:
        return
    subset = skill_scores.copy()
    subset["label"] = subset.apply(
        lambda row: f"{row['metric_target']}: {row['model']} vs {row['baseline']}",
        axis=1,
    )
    subset = subset.sort_values(["metric_target", "baseline"]).reset_index(drop=True)
    positions = np.arange(len(subset))
    lower = subset["rmse_skill"] - subset["rmse_skill_boot_ci_low"]
    upper = subset["rmse_skill_boot_ci_high"] - subset["rmse_skill"]

    figure, axis = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(subset) + 1.5)))
    axis.barh(positions, subset["rmse_skill"], color="#4c78a8")
    axis.errorbar(
        subset["rmse_skill"],
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(subset["label"])
    axis.set_xlabel("RMSE skill score (positive favors causal graph)")
    axis.set_title("Paired spatial-block bootstrap validation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render a small Markdown table for the validation report."""
    if frame.empty:
        return "_No rows available._"
    display = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
            )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_validation_report(
    output_path: Path,
    *,
    config_path: Path,
    ard_db: Path,
    graph_db: Path,
    graph_table: str,
    target: str,
    evaluation_years: Sequence[int],
    observed_target_months: Sequence[int],
    training_end_year: int,
    prediction_mode: str,
    effect_mode: str,
    fit_mode: str,
    metrics: pd.DataFrame,
    skill_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    spatial_block_size: int,
    bootstrap_resamples: int,
    graph_training_max_year: int | None,
    graph_training_verified: bool,
) -> None:
    """Write a concise Markdown report describing design and headline results."""
    status_counts = diagnostics["status"].value_counts() if "status" in diagnostics else pd.Series(dtype=int)
    overall_metrics = metrics[metrics["group"] == "all"].copy() if "group" in metrics else pd.DataFrame()
    primary = skill_scores[
        (skill_scores["metric_target"] == "response")
        & (skill_scores["model"] == "causal_graph_response")
    ].copy() if not skill_scores.empty else pd.DataFrame()
    if primary.empty and not skill_scores.empty:
        primary = skill_scores.copy()

    lines = [
        "# Causal Holdout Validation Report",
        "",
        "## Design",
        "",
        "This validation keeps the learned causal graph fixed and evaluates "
        "held-out target observations out of sample. The primary estimand is "
        "response prediction: same-month target departures predicted from "
        "held-out driver departures propagated through graph coefficients.",
        "",
        f"- Config: `{config_path}`",
        f"- Time-series DB: `{ard_db}`",
        f"- Graph DB: `{graph_db}`",
        f"- Graph table: `{graph_table}`",
        f"- Target: `{target}`",
        f"- Evaluation years: `{', '.join(str(year) for year in evaluation_years)}`",
        f"- Observed target months: `{', '.join(str(month) for month in observed_target_months)}`",
        f"- Training data used for intercepts/baselines ends in: `{training_end_year}`",
        f"- Prediction mode: `{prediction_mode}`",
        f"- Effect mode: `{effect_mode}`",
        f"- Fit mode: `{fit_mode}`",
        f"- Graph training max year: `{graph_training_max_year}`",
        f"- Graph training verified pre-holdout: `{graph_training_verified}`",
        f"- Spatial bootstrap block size: `{spatial_block_size}` grid cells",
        f"- Bootstrap resamples: `{bootstrap_resamples}`",
        "",
        "For a publication claim, the graph database itself must have been "
        "learned only from years no later than the reported training end year.",
        "",
        "## Headline Paired Skill Scores",
        "",
        "Positive RMSE skill means the causal graph has lower RMSE than the "
        "baseline on the same held-out rows. Confidence intervals are spatial-"
        "block bootstrap intervals.",
        "",
        markdown_table(
            primary,
            [
                "metric_target",
                "model",
                "baseline",
                "n",
                "n_spatial_blocks",
                "rmse_skill",
                "rmse_skill_boot_ci_low",
                "rmse_skill_boot_ci_high",
                "delta_rmse",
                "bootstrap_p_delta_rmse_lt_zero",
            ],
        ),
        "",
        "## Overall Metrics",
        "",
        markdown_table(
            overall_metrics,
            [
                "metric_target",
                "model",
                "n",
                "mae",
                "rmse",
                "r2",
                "bias",
                "pearson_r",
            ],
        ),
        "",
        "## Pixel Inclusion Diagnostics",
        "",
        markdown_table(
            status_counts.rename_axis("status").reset_index(name="count"),
            ["status", "count"],
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--evaluation-year",
    "evaluation_years",
    required=True,
    multiple=True,
    type=int,
    help="Held-out evaluation year. Repeat for a multi-year hindcast.",
)
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
@click.option(
    "--graph-db",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override graph database path, e.g. a graph DB trained only on pre-holdout years.",
)
@click.option("--graph-table", default="pixel_graphs", show_default=True)
@click.option(
    "--graph-training-max-year",
    default=None,
    type=int,
    help="Explicit maximum year used to train the supplied graph DB.",
)
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
    "--bootstrap-resamples",
    default=1000,
    show_default=True,
    type=click.IntRange(min=1),
    help="Spatial-block bootstrap resamples for paired skill intervals.",
)
@click.option(
    "--spatial-block-size",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Row/column grid-cell width of bootstrap spatial blocks.",
)
@click.option(
    "--random-seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for bootstrap resampling.",
)
@click.option(
    "--allow-unverified-graph-training",
    is_flag=True,
    help=(
        "Allow exploratory runs when the config does not record graph_discovery.max_year. "
        "Known leakage is still rejected."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def validate_causal_holdout(
    config_path: Path,
    evaluation_years: tuple[int, ...],
    observed_target_months: tuple[int, ...],
    training_end_year: int | None,
    target_variable: str | None,
    graph_db: Path | None,
    graph_table: str,
    graph_training_max_year: int | None,
    graph_window_size: int,
    min_train_samples: int,
    prediction_mode: str,
    effect_mode: str,
    fit_mode: str,
    ridge_alpha: float,
    large_difference_percentile: float,
    bootstrap_resamples: int,
    spatial_block_size: int,
    random_seed: int,
    allow_unverified_graph_training: bool,
    output_dir: Path | None,
) -> None:
    """Validate graph-constrained structural equations on held-out observations."""
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])
    target = str(target_variable or config.get("reference_var") or "ndvi")
    configured_shift = target_shift(config, target)
    evaluation_years = tuple(sorted(set(int(year) for year in evaluation_years)))
    if not evaluation_years:
        raise click.ClickException("At least one --evaluation-year is required.")
    first_evaluation_year = min(evaluation_years)
    training_end_year = (
        first_evaluation_year - 1 if training_end_year is None else training_end_year
    )
    if training_end_year >= first_evaluation_year:
        raise click.ClickException(
            "--training-end-year must be before the first held-out "
            f"evaluation year ({first_evaluation_year}) for an out-of-sample test."
        )
    output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir
        / (
            f"causal_holdout_{target}_{evaluation_years[0]}"
            if len(evaluation_years) == 1
            else f"causal_holdout_{target}_{evaluation_years[0]}_{evaluation_years[-1]}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ard_db = resolve_path(
        experiment_dir,
        config_value(config, "timeseries_db")
        or config_value(config, "input_db"),
        experiment_dir / f"{experiment_name}_ard.duckdb",
    )
    timeseries_table = str(
        config_value(config, "timeseries_table")
        or config_value(config, "input_table")
        or experiment_name
    )
    graph_db = resolve_path(
        experiment_dir,
        graph_db
        or config_value(config, "graph_db")
        or config_value(config, "output_db"),
        experiment_dir / f"{experiment_name}_graphs.duckdb",
    )
    graph_training_max_year = (
        graph_training_max_year
        if graph_training_max_year is not None
        else config_value(config, "max_year")
    )
    if graph_training_max_year is not None and int(graph_training_max_year) > training_end_year:
        raise click.ClickException(
            "Configured graph_discovery.max_year exceeds the validation "
            f"training end year: {graph_training_max_year} > {training_end_year}. "
            "Use a graph database trained only on pre-holdout years."
        )
    graph_training_verified = graph_training_max_year is not None
    output_db = output_dir / f"{experiment_name}_causal_holdout.duckdb"
    require_files([ard_db, graph_db])
    if graph_training_max_year is None and not allow_unverified_graph_training:
        raise click.ClickException(
            "Cannot verify an out-of-sample graph validation because "
            "graph_discovery.max_year is not set in the config. Re-run graph "
            "discovery with --max-year equal to the validation training end "
            "year, or pass --allow-unverified-graph-training for an exploratory "
            "diagnostic that is not publishable as strict holdout validation."
        )
    if graph_training_max_year is None:
        click.echo(
            "Warning: graph_discovery.max_year is not set; graph-training "
            "cutoff is unverified. Treat outputs as exploratory diagnostics."
        )
    if prediction_mode == "response" and fit_mode != "adjacency":
        raise click.ClickException(
            "--prediction-mode response requires --fit-mode adjacency, because "
            "the response experiment propagates saved graph coefficients rather "
            "than refitting slopes."
        )

    click.echo("Loading and shifting ARD time series...")
    shifted, labels = load_shifted_ard(ard_db, timeseries_table, config)
    if target not in labels:
        raise click.ClickException(
            f"Target {target!r} is not in graph-discovery variables."
        )

    click.echo("Loading existing historical graph database...")
    graph_rows = load_graph_rows(graph_db, graph_table)
    group_lookup = build_group_lookup(shifted)
    eval_rows = evaluation_model_months_for_years(
        evaluation_years=evaluation_years,
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
    plot_target_label = target
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
    skill_scores_df = paired_skill_rows(
        predictions_df,
        n_bootstrap=bootstrap_resamples,
        ci=0.95,
        block_size=spatial_block_size,
        random_seed=random_seed,
    )
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
                "pearson_r",
            ]
        )

    predictions_df.to_csv(output_dir / "causal_holdout_predictions.csv", index=False)
    coefficients_df.to_csv(output_dir / "causal_holdout_coefficients.csv", index=False)
    metrics_df.to_csv(output_dir / "causal_holdout_metrics.csv", index=False)
    skill_scores_df.to_csv(
        output_dir / "causal_holdout_paired_skill_scores.csv",
        index=False,
    )
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
        if not skill_scores_df.empty:
            write_dataframe_table(
                con,
                skill_scores_df,
                "causal_holdout_paired_skill_scores",
            )
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
        title=f"Large graph-model {plot_target_label} differences",
        colorbar_label=f"Observed - predicted {plot_target_label}",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="residual",
        filename_prefix="large_observed_minus_predicted_ndvi",
        title=f"Large graph-model {plot_target_label} differences",
        colorbar_label=f"Observed - predicted {plot_target_label}",
        percentile=large_difference_percentile,
    )
    plot_large_difference_map(
        predictions_df,
        output_dir / "large_observed_minus_climatology_ndvi_map.png",
        residual_column="climatology_residual",
        title=f"Large climatology-baseline {plot_target_label} differences",
        colorbar_label=f"Observed - climatology {plot_target_label}",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="climatology_residual",
        filename_prefix="large_observed_minus_climatology_ndvi",
        title=f"Large climatology-baseline {plot_target_label} differences",
        colorbar_label=f"Observed - climatology {plot_target_label}",
        percentile=large_difference_percentile,
    )
    plot_large_difference_map(
        predictions_df,
        output_dir / "large_prediction_minus_climatology_ndvi_map.png",
        residual_column="prediction_minus_climatology",
        title=f"Large graph-vs-climatology {plot_target_label} differences",
        colorbar_label=f"Predicted - climatology {plot_target_label}",
        percentile=large_difference_percentile,
    )
    plot_monthly_large_difference_maps(
        predictions_df,
        output_dir,
        residual_column="prediction_minus_climatology",
        filename_prefix="large_prediction_minus_climatology_ndvi",
        title=f"Large graph-vs-climatology {plot_target_label} differences",
        colorbar_label=f"Predicted - climatology {plot_target_label}",
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
    plot_skill_scores(
        skill_scores_df,
        output_dir / "paired_spatial_block_skill_scores.png",
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
    write_validation_report(
        output_dir / "validation_report.md",
        config_path=config_path,
        ard_db=ard_db,
        graph_db=graph_db,
        graph_table=graph_table,
        target=target,
        evaluation_years=evaluation_years,
        observed_target_months=observed_target_months,
        training_end_year=training_end_year,
        prediction_mode=prediction_mode,
        effect_mode=effect_mode,
        fit_mode=fit_mode,
        metrics=metrics_df,
        skill_scores=skill_scores_df,
        diagnostics=diagnostics_df,
        spatial_block_size=spatial_block_size,
        bootstrap_resamples=bootstrap_resamples,
        graph_training_max_year=(
            int(graph_training_max_year)
            if graph_training_max_year is not None
            else None
        ),
        graph_training_verified=graph_training_verified,
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
    if not skill_scores_df.empty:
        headline = skill_scores_df[
            (skill_scores_df["metric_target"] == "response")
            & (skill_scores_df["baseline"] == "zero_response")
        ].copy()
        if headline.empty:
            headline = skill_scores_df.head(1)
        click.echo(
            "Primary paired RMSE skill: "
            + ", ".join(
                f"{row.model} vs {row.baseline}={row.rmse_skill:.3f} "
                f"[{row.rmse_skill_boot_ci_low:.3f}, "
                f"{row.rmse_skill_boot_ci_high:.3f}]"
                for row in headline.itertuples(index=False)
            )
        )
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
        "Evaluation years: "
        + ", ".join(str(year) for year in evaluation_years)
    )
    click.echo(
        "Observed target months: "
        + ", ".join(str(month) for month in observed_target_months)
    )
    click.echo(f"Training end year: {training_end_year}")
    click.echo(
        "Graph training max year: "
        + (
            str(graph_training_max_year)
            if graph_training_max_year is not None
            else "unverified"
        )
    )
    click.echo(f"Spatial bootstrap block size: {spatial_block_size}")
    click.echo(f"Bootstrap resamples: {bootstrap_resamples}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Output database: {output_db}")
    click.echo(f"Validation report: {output_dir / 'validation_report.md'}")


if __name__ == "__main__":
    validate_causal_holdout()
