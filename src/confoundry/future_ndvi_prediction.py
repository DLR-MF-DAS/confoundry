"""Predict held-out-year NDVI anomaly classes from historical graph features.

This command tests whether graphs learned from a historical period contain
information about a later vegetation state that was not used during graph
discovery. It expects the standard Confoundry outputs next to the experiment
configuration:

* ``<name>_ard.duckdb`` with the long-form pixel time series.
* ``<name>_graphs.duckdb`` with existing historical graph-discovery output.

The target is computed from the ARD table by comparing evaluation-year NDVI
against a pixel-wise historical monthly climatology. The classifier receives
graph features only from the already discovered graphs, so the evaluation year
must be absent from graph discovery for the experiment to be temporally held
out. This command never rebuilds the graph database; it can only rebuild the ARD
table from an already-populated source catalog when validation-year rows are
missing and ``--regather-if-missing`` is passed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier

from confoundry.analysis_helpers import (
    ensure_identifier,
    require_files,
    write_dataframe_table,
)
from confoundry.landcover_graph_validation import (
    add_spatial_blocks,
    build_graph_features,
    choose_number_of_folds,
    evaluate_model,
    fit_final_model_and_importance,
    make_classifier,
    plot_confusion,
    plot_feature_importance,
    plot_metrics,
)
from confoundry.landcover_helpers import load_graph_rows


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and minimally validate a Confoundry experiment config."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise click.ClickException("Experiment YAML must contain a mapping.")
    required = ["name", "columns"]
    missing = [key for key in required if key not in config]
    if missing:
        raise click.ClickException(
            f"Configuration is missing required keys: {missing}"
        )
    return config


def configured_column_names(config: Mapping[str, Any]) -> list[str]:
    """Return configured variable names from the experiment config."""
    columns = config.get("columns")
    if not isinstance(columns, list):
        raise click.ClickException("config['columns'] must be a list.")
    return [str(spec["name"]) for spec in columns]


def table_columns(db_path: Path, table: str) -> set[str]:
    """Return DuckDB table columns."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table not in tables:
            raise click.ClickException(
                f"{table!r} not found in {db_path}. "
                f"Available tables: {sorted(tables)}"
            )
        return set(
            con.execute(
                f"DESCRIBE {ensure_identifier(table)}"
            ).fetchdf()["column_name"]
        )
    finally:
        con.close()


def year_is_available(
    ard_db: Path,
    table: str,
    target_variable: str,
    evaluation_year: int,
) -> bool:
    """Return whether ARD rows exist for the target variable in a year."""
    con = duckdb.connect(ard_db, read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {ensure_identifier(table)}
            WHERE year = ?
              AND {ensure_identifier(target_variable)} IS NOT NULL
            """,
            [evaluation_year],
        ).fetchone()[0]
        return int(rows) > 0
    finally:
        con.close()


def source_catalog_has_year(
    source_db: Path,
    source_names: set[str],
    evaluation_year: int,
) -> bool:
    """Return whether source GeoTIFF catalog has target-year source rasters."""
    if not source_db.exists():
        return False
    con = duckdb.connect(source_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if "geotiff_catalog" not in tables:
            return False
        rows = con.execute(
            """
            SELECT COUNT(*) AS n
            FROM geotiff_catalog
            WHERE year = ?
              AND variable_name IN (
            """
            + ", ".join(["?"] * len(source_names))
            + ")",
            [evaluation_year, *sorted(source_names)],
        ).fetchone()[0]
        return int(rows) > 0
    finally:
        con.close()


def ensure_evaluation_year(
    config_path: Path,
    config: Mapping[str, Any],
    ard_db: Path,
    source_db: Path,
    table: str,
    target_variable: str,
    evaluation_year: int,
    regather_if_missing: bool,
) -> None:
    """Ensure evaluation-year ARD rows exist, optionally rebuilding ARD."""
    if year_is_available(ard_db, table, target_variable, evaluation_year):
        return

    if not regather_if_missing:
        raise click.ClickException(
            f"No non-null {target_variable!r} ARD rows found for "
            f"{evaluation_year}. Download/source the evaluation year and rerun "
            "gather.py, or pass --regather-if-missing if the source DB already "
            "contains the rasters."
        )

    name_map = config.get("name_map") or {}
    source_names = {
        str(source_name)
        for source_name, normalized_name in dict(name_map).items()
        if str(normalized_name) == target_variable
    }
    source_names.add(target_variable)
    if not source_catalog_has_year(source_db, source_names, evaluation_year):
        raise click.ClickException(
            f"The source catalog does not contain {target_variable!r} rasters "
            f"for {evaluation_year}. Extend the source download through the "
            "evaluation year first, then rerun this command."
        )

    click.echo(
        "Evaluation-year rows are absent from the ARD table; rebuilding ARD "
        "from the source catalog with confoundry.gather..."
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "confoundry.gather",
            "-c",
            str(config_path),
        ],
        check=True,
    )
    if not year_is_available(ard_db, table, target_variable, evaluation_year):
        raise click.ClickException(
            f"ARD rebuild completed, but {target_variable!r} rows for "
            f"{evaluation_year} are still missing."
        )


def load_ndvi_anomaly_targets(
    ard_db: Path,
    table: str,
    target_variable: str,
    evaluation_year: int,
    baseline_start_year: int | None,
    baseline_end_year: int | None,
    months: Sequence[int],
) -> pd.DataFrame:
    """Compute held-out-year NDVI anomaly and z-score per graph pixel."""
    month_values = [int(month) for month in months]
    if not month_values:
        raise click.ClickException("At least one target month is required.")
    if any(month < 1 or month > 12 for month in month_values):
        raise click.ClickException("Target months must lie in 1..12.")

    where_baseline = ["year <> ?"]
    params: list[Any] = [evaluation_year]
    if baseline_start_year is not None:
        where_baseline.append("year >= ?")
        params.append(int(baseline_start_year))
    if baseline_end_year is not None:
        where_baseline.append("year <= ?")
        params.append(int(baseline_end_year))

    month_sql = ", ".join(["?"] * len(month_values))
    target_sql = ensure_identifier(target_variable)
    table_sql = ensure_identifier(table)

    con = duckdb.connect(ard_db, read_only=True)
    try:
        query = f"""
        WITH baseline_monthly AS (
            SELECT
                row,
                col,
                month,
                AVG({target_sql}) AS climatology_mean,
                STDDEV_POP({target_sql}) AS climatology_sd,
                COUNT({target_sql}) AS climatology_n
            FROM {table_sql}
            WHERE {" AND ".join(where_baseline)}
              AND month IN ({month_sql})
              AND {target_sql} IS NOT NULL
            GROUP BY row, col, month
        ),
        evaluation_monthly AS (
            SELECT
                row,
                col,
                month,
                AVG({target_sql}) AS evaluation_value
            FROM {table_sql}
            WHERE year = ?
              AND month IN ({month_sql})
              AND {target_sql} IS NOT NULL
            GROUP BY row, col, month
        ),
        joined AS (
            SELECT
                e.row,
                e.col,
                e.month,
                e.evaluation_value,
                b.climatology_mean,
                b.climatology_sd,
                b.climatology_n,
                e.evaluation_value - b.climatology_mean AS anomaly,
                CASE
                    WHEN b.climatology_sd > 0
                    THEN (e.evaluation_value - b.climatology_mean)
                         / b.climatology_sd
                    ELSE NULL
                END AS anomaly_z
            FROM evaluation_monthly AS e
            JOIN baseline_monthly AS b
              ON e.row = b.row
             AND e.col = b.col
             AND e.month = b.month
        )
        SELECT
            row,
            col,
            AVG(evaluation_value) AS evaluation_value,
            AVG(climatology_mean) AS climatology_mean,
            AVG(climatology_sd) AS climatology_sd,
            AVG(climatology_n) AS climatology_n,
            AVG(anomaly) AS ndvi_anomaly,
            AVG(anomaly_z) AS ndvi_anomaly_z,
            COUNT(*) AS n_target_months
        FROM joined
        GROUP BY row, col
        ORDER BY row, col
        """
        all_params = [
            *params,
            *month_values,
            evaluation_year,
            *month_values,
        ]
        targets = con.execute(query, all_params).fetchdf()
    finally:
        con.close()

    if targets.empty:
        raise click.ClickException(
            "No held-out NDVI anomaly targets could be computed. Check the "
            "evaluation year, target months, and baseline period."
        )
    return targets


def assign_target_classes(
    targets: pd.DataFrame,
    class_mode: str,
    n_quantile_classes: int,
    z_threshold: float,
) -> pd.DataFrame:
    """Add a categorical prediction target to anomaly rows."""
    result = targets.copy()
    if class_mode == "quantile":
        if n_quantile_classes < 2:
            raise click.ClickException("--n-quantile-classes must be >= 2.")
        codes = pd.qcut(
            result["ndvi_anomaly"],
            q=n_quantile_classes,
            labels=False,
            duplicates="drop",
        )
        n_classes = int(codes.max()) + 1 if not codes.dropna().empty else 0
        labels = [
            f"q{idx + 1}_lowest" if idx == 0 else
            f"q{idx + 1}_highest" if idx == n_classes - 1 else
            f"q{idx + 1}"
            for idx in range(n_classes)
        ]
        result["target_class"] = codes.map(
            {idx: label for idx, label in enumerate(labels)}
        )
    elif class_mode == "zscore":
        result["target_class"] = np.select(
            [
                result["ndvi_anomaly_z"] <= -float(z_threshold),
                result["ndvi_anomaly_z"] >= float(z_threshold),
            ],
            [
                "negative_anomaly",
                "positive_anomaly",
            ],
            default="near_normal",
        )
    else:
        raise ValueError(f"Unknown class mode: {class_mode}")

    result = result.dropna(subset=["target_class"]).copy()
    result["target_class"] = result["target_class"].astype(str)
    if result["target_class"].nunique() < 2:
        raise click.ClickException(
            "Fewer than two target classes were produced. Try quantile mode, "
            "more target months, or a less strict z-score threshold."
        )
    return result


def compute_historical_raw_features(
    ard_db: Path,
    table: str,
    graph_pixels: pd.DataFrame,
    variables: Sequence[str],
    graph_window_size: int,
    evaluation_year: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute raw summary features using only years before evaluation."""
    if not variables:
        return graph_pixels[["row", "col"]].copy(), []

    con = duckdb.connect(ard_db, read_only=True)
    try:
        available_columns = table_columns(ard_db, table)
        usable_variables = [
            variable for variable in variables if variable in available_columns
        ]
        if not usable_variables:
            return graph_pixels[["row", "col"]].copy(), []

        con.register(
            "_graph_pixels",
            graph_pixels[["row", "col"]].drop_duplicates(),
        )
        aggregates: list[str] = []
        raw_columns: list[str] = []
        for variable in usable_variables:
            variable_sql = ensure_identifier(variable)
            mean_name = f"raw_historical_mean_{variable}"
            std_name = f"raw_historical_std_{variable}"
            aggregates.extend(
                [
                    f"AVG(a.{variable_sql}) AS {ensure_identifier(mean_name)}",
                    f"STDDEV_POP(a.{variable_sql}) AS {ensure_identifier(std_name)}",
                ]
            )
            raw_columns.extend([mean_name, std_name])

        query = f"""
            SELECT
                gp.row,
                gp.col,
                {", ".join(aggregates)}
            FROM _graph_pixels AS gp
            JOIN {ensure_identifier(table)} AS a
              ON a.row BETWEEN gp.row - {int(graph_window_size)}
                           AND gp.row + {int(graph_window_size)}
             AND a.col BETWEEN gp.col - {int(graph_window_size)}
                           AND gp.col + {int(graph_window_size)}
             AND a.year < {int(evaluation_year)}
            GROUP BY gp.row, gp.col
            ORDER BY gp.row, gp.col
        """
        raw_df = con.execute(query).fetchdf()
    finally:
        try:
            con.unregister("_graph_pixels")
        except Exception:
            pass
        con.close()
    return raw_df, raw_columns


def plot_anomaly_map(samples: pd.DataFrame, output_path: Path) -> None:
    """Plot held-out-year NDVI anomaly by pixel coordinate."""
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    scatter = axis.scatter(
        samples["col"],
        samples["row"],
        c=samples["ndvi_anomaly"],
        s=4,
        cmap="RdYlGn",
        alpha=0.8,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_title("Held-out-year NDVI anomaly")
    figure.colorbar(scatter, ax=axis, label="NDVI anomaly")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Experiment YAML configuration.",
)
@click.option(
    "--evaluation-year",
    required=True,
    type=int,
    help="Held-out year used only for target construction.",
)
@click.option(
    "--target-variable",
    default=None,
    help="Vegetation variable to predict. Defaults to reference_var or ndvi.",
)
@click.option(
    "--target-month",
    "target_months",
    multiple=True,
    type=click.IntRange(min=1, max=12),
    default=(6, 7, 8, 9),
    show_default=True,
    help="Month included in the held-out target. Repeat as needed.",
)
@click.option("--baseline-start-year", default=None, type=int)
@click.option("--baseline-end-year", default=None, type=int)
@click.option(
    "--graph-table",
    default="pixel_graphs",
    show_default=True,
    help="Table containing graph-discovery output.",
)
@click.option(
    "--graph-window-size",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Neighborhood radius used during graph discovery.",
)
@click.option(
    "--feature-set",
    type=click.Choice(
        ["consensus", "raw", "probability", "total_effect", "combined"]
    ),
    default="combined",
    show_default=True,
)
@click.option(
    "--exclude-variable",
    "excluded_variables",
    multiple=True,
    default=("month_sin", "month_cos"),
    show_default=True,
    help="Graph/raw variable to omit from features. Repeat as needed.",
)
@click.option(
    "--class-mode",
    type=click.Choice(["quantile", "zscore"]),
    default="quantile",
    show_default=True,
    help="How to convert continuous anomaly values into classes.",
)
@click.option(
    "--n-quantile-classes",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
)
@click.option(
    "--z-threshold",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Absolute z-score threshold for zscore class mode.",
)
@click.option(
    "--min-class-samples",
    default=20,
    show_default=True,
    type=click.IntRange(min=2),
)
@click.option(
    "--block-size-km",
    default=100.0,
    show_default=True,
    type=click.FloatRange(min=1.0),
)
@click.option("--folds", default=5, show_default=True, type=click.IntRange(min=2))
@click.option(
    "--classifier",
    "classifier_name",
    type=click.Choice(["random_forest", "logistic"]),
    default="random_forest",
    show_default=True,
)
@click.option("--trees", default=500, show_default=True, type=click.IntRange(min=10))
@click.option("--workers", default=-1, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--raw-baseline/--no-raw-baseline",
    default=True,
    show_default=True,
    help="Evaluate historical raw-summary and combined baselines.",
)
@click.option(
    "--regather-if-missing",
    is_flag=True,
    help=(
        "Rebuild/extend the ARD table from the source DB if the evaluation "
        "year is absent. This never changes the graph DB."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Defaults inside the experiment directory.",
)
@click.option("--top-features", default=30, show_default=True, type=click.IntRange(min=1))
def predict_future_ndvi(
    config_path: Path,
    evaluation_year: int,
    target_variable: str | None,
    target_months: tuple[int, ...],
    baseline_start_year: int | None,
    baseline_end_year: int | None,
    graph_table: str,
    graph_window_size: int,
    feature_set: str,
    excluded_variables: tuple[str, ...],
    class_mode: str,
    n_quantile_classes: int,
    z_threshold: float,
    min_class_samples: int,
    block_size_km: float,
    folds: int,
    classifier_name: str,
    trees: int,
    workers: int,
    seed: int,
    raw_baseline: bool,
    regather_if_missing: bool,
    output_dir: Path | None,
    top_features: int,
) -> None:
    """Predict held-out-year NDVI anomaly classes from historical graphs."""
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])
    target_variable = str(
        target_variable
        or config.get("reference_var")
        or ("ndvi" if "ndvi" in configured_column_names(config) else "")
    )
    if not target_variable:
        raise click.ClickException(
            "Could not infer target variable. Pass --target-variable."
        )

    ard_db = experiment_dir / f"{experiment_name}_ard.duckdb"
    source_db = experiment_dir / f"{experiment_name}_source_db.duckdb"
    graph_db = experiment_dir / f"{experiment_name}_graphs.duckdb"
    output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir / f"future_{target_variable}_prediction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / f"{experiment_name}_future_{target_variable}.duckdb"

    require_files([ard_db, graph_db])
    available = table_columns(ard_db, experiment_name)
    required_cols = {"row", "col", "year", "month", target_variable}
    missing_cols = required_cols - available
    if missing_cols:
        raise click.ClickException(
            f"ARD table is missing required columns: {sorted(missing_cols)}"
        )

    ensure_evaluation_year(
        config_path=config_path,
        config=config,
        ard_db=ard_db,
        source_db=source_db,
        table=experiment_name,
        target_variable=target_variable,
        evaluation_year=evaluation_year,
        regather_if_missing=regather_if_missing,
    )

    click.echo(
        "Using existing graph DB. Ensure it was produced without the "
        f"evaluation year {evaluation_year}."
    )

    click.echo("Loading historical graph rows...")
    graph_rows = load_graph_rows(graph_db, graph_table)
    graph_features, graph_columns, graph_variables = build_graph_features(
        graph_rows=graph_rows,
        feature_set=feature_set,
        excluded_variables=set(excluded_variables),
    )

    click.echo("Computing held-out-year NDVI anomaly target...")
    targets = load_ndvi_anomaly_targets(
        ard_db=ard_db,
        table=experiment_name,
        target_variable=target_variable,
        evaluation_year=evaluation_year,
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
        months=target_months,
    )
    targets = assign_target_classes(
        targets,
        class_mode=class_mode,
        n_quantile_classes=n_quantile_classes,
        z_threshold=z_threshold,
    )

    samples = graph_features.merge(
        targets,
        on=["row", "col"],
        how="inner",
        validate="one_to_one",
    )
    class_counts = samples["target_class"].value_counts()
    retained_classes = class_counts[class_counts >= min_class_samples].index
    samples = samples[samples["target_class"].isin(retained_classes)].copy()
    if samples["target_class"].nunique() < 2:
        raise click.ClickException(
            "Fewer than two target classes remain after --min-class-samples."
        )

    samples = add_spatial_blocks(samples, block_size_km)
    samples = samples.sort_values(["row", "col"]).reset_index(drop=True)
    samples.insert(0, "sample_id", np.arange(1, len(samples) + 1))

    raw_columns: list[str] = []
    if raw_baseline:
        raw_variables = [
            name
            for name in configured_column_names(config)
            if name not in set(excluded_variables)
        ]
        raw_features, raw_columns = compute_historical_raw_features(
            ard_db=ard_db,
            table=experiment_name,
            graph_pixels=samples[["row", "col"]],
            variables=raw_variables,
            graph_window_size=graph_window_size,
            evaluation_year=evaluation_year,
        )
        samples = samples.merge(
            raw_features,
            on=["row", "col"],
            how="left",
            validate="one_to_one",
        )

    labels = samples["target_class"].astype(str)
    groups = samples["spatial_block"].astype(str)
    feasible_folds = choose_number_of_folds(labels, groups, folds)
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    try:
        splitter = StratifiedGroupKFold(
            n_splits=feasible_folds,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(samples[graph_columns], labels, groups))
    except ValueError:
        click.echo("Falling back to GroupKFold.")
        splitter = GroupKFold(n_splits=feasible_folds)
        splits = list(splitter.split(samples[graph_columns], labels, groups))

    classifier = make_classifier(classifier_name, seed, trees, workers)
    model_specs: list[tuple[str, Any, list[str]]] = [
        ("majority", DummyClassifier(strategy="most_frequent"), graph_columns),
        ("graph", classifier, graph_columns),
    ]
    if raw_columns:
        model_specs.extend(
            [
                ("raw", classifier, raw_columns),
                ("graph_plus_raw", classifier, graph_columns + raw_columns),
            ]
        )

    metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for model_name, model, feature_columns in model_specs:
        click.echo(f"Evaluating {model_name}...")
        model_metrics, model_predictions = evaluate_model(
            model_name=model_name,
            model=model,
            features=samples,
            feature_columns=feature_columns,
            labels=labels,
            groups=groups,
            splits=splits,
        )
        metrics_parts.append(model_metrics)
        prediction_parts.append(model_predictions)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_metadata = samples[
        [
            "row",
            "col",
            "target_class",
            "ndvi_anomaly",
            "ndvi_anomaly_z",
            "evaluation_value",
            "climatology_mean",
            "spatial_block",
        ]
    ].reset_index(names="sample_index")
    predictions = predictions.merge(
        prediction_metadata.rename(columns={"target_class": "label"}),
        left_on=["sample_index", "true_class"],
        right_on=["sample_index", "label"],
        how="left",
        validate="many_to_one",
    )

    final_model, graph_importance = fit_final_model_and_importance(
        model=classifier,
        features=samples,
        feature_columns=graph_columns,
        labels=labels,
    )

    class_summary = (
        samples.groupby("target_class", as_index=False)
        .agg(
            n_samples=("sample_id", "size"),
            n_spatial_blocks=("spatial_block", "nunique"),
            mean_anomaly=("ndvi_anomaly", "mean"),
            mean_anomaly_z=("ndvi_anomaly_z", "mean"),
        )
        .sort_values("target_class")
    )

    con = duckdb.connect(output_db)
    try:
        write_dataframe_table(con, samples, "future_prediction_samples")
        write_dataframe_table(con, metrics, "cv_metrics")
        write_dataframe_table(con, predictions, "cv_predictions")
        write_dataframe_table(con, graph_importance, "graph_feature_importance")
        write_dataframe_table(con, class_summary, "class_summary")
    finally:
        con.close()

    samples.to_csv(output_dir / "future_prediction_samples.csv", index=False)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    graph_importance.to_csv(output_dir / "graph_feature_importance.csv", index=False)
    class_summary.to_csv(output_dir / "class_summary.csv", index=False)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": graph_columns,
            "class_names": sorted(labels.unique()),
            "graph_variables": graph_variables,
            "metadata": {
                "evaluation_year": evaluation_year,
                "target_variable": target_variable,
                "target_months": list(target_months),
                "class_mode": class_mode,
                "feature_set": feature_set,
                "graph_window_size": graph_window_size,
            },
        },
        output_dir / "graph_future_ndvi_classifier.joblib",
    )

    plot_metrics(metrics, output_dir / "cv_metrics_future_ndvi.png")
    plot_feature_importance(
        graph_importance,
        output_dir / "feature_importance_graph.png",
        top_n=top_features,
    )
    class_names = sorted(labels.unique())
    for model_name, _, _ in model_specs:
        plot_confusion(
            predictions=predictions,
            class_names=class_names,
            model_name=model_name,
            output_path=output_dir / f"confusion_matrix_{model_name}.png",
        )
    plot_anomaly_map(samples, output_dir / "future_ndvi_anomaly_map.png")

    click.echo("")
    click.echo("Future NDVI prediction experiment complete.")
    click.echo(f"Samples: {len(samples):,}")
    click.echo(f"Classes: {', '.join(sorted(labels.unique()))}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Output database: {output_db}")


if __name__ == "__main__":
    predict_future_ndvi()
