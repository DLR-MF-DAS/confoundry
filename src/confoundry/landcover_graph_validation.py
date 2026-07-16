"""Validate pixel-wise causal graphs against independent land-cover classes.

This command performs an end-to-end external validation experiment:

1. Read per-pixel DirectLiNGAM graphs from ``<name>_graphs.duckdb``.
2. Locate the experiment reference raster from ``<name>_source_db.duckdb``.
3. Download ESA WorldCover 2021 tiles covering the graph domain.
4. Assign a dominant land-cover class and purity to every graph footprint.
5. Convert graph matrices into fixed-length tabular features.
6. Optionally compute non-causal raw time-series summary features.
7. Evaluate majority, graph-only, raw-only, and combined classifiers using
   spatially blocked cross-validation.
8. Save samples, predictions, metrics, feature importances, plots, a trained
   final model, and a DuckDB validation database.

The graph footprint is controlled independently through
``--graph-window-size`` and should match the value used during graph discovery.
For example, ``--graph-window-size 1`` labels the full 3x3 reference-pixel
footprint represented by each graph.

ESA WorldCover 2021 is used because it provides a globally consistent 10 m
land-cover product with a compact 11-class legend. The default ``all`` class
set keeps the full WorldCover legend. Use ``--class-set vegetation`` or
``--class-set terrestrial`` for narrower experiments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import click
import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from confoundry.analysis_helpers import (
    ensure_identifier,
    require_files,
    write_dataframe_table,
)
from confoundry.landcover_helpers import (
    CLASS_SETS,
    WORLD_COVER_VERSION,
    WORLD_COVER_YEAR,
    derive_landcover_paths as derive_paths,
    download_worldcover,
    graph_domain_bounds_wgs84,
    label_graph_footprints,
    load_graph_rows,
    locate_reference_raster,
    read_landcover_config as read_config,
    required_worldcover_tiles,
)


def parse_matrix(value: Any, expected_size: int) -> np.ndarray:
    """Parse one JSON matrix and verify its dimensions."""
    matrix = np.asarray(json.loads(value), dtype=float)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Expected {(expected_size, expected_size)}, got {matrix.shape}"
        )
    return matrix


def total_effect_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Compute all linear total effects from an adjacency matrix."""
    identity = np.eye(adjacency.shape[0], dtype=float)
    try:
        return np.linalg.solve(identity - adjacency, identity) - identity
    except np.linalg.LinAlgError:
        return np.full_like(adjacency, np.nan, dtype=float)


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute binary entropy while handling probabilities equal to zero or one."""
    p = np.clip(probabilities, 0.0, 1.0)
    result = np.zeros_like(p, dtype=float)
    interior = (p > 0.0) & (p < 1.0)
    result[interior] = (
        -p[interior] * np.log2(p[interior])
        - (1.0 - p[interior]) * np.log2(1.0 - p[interior])
    )
    return result


def graph_feature_columns(
    variables: Sequence[str],
    feature_set: str,
) -> list[str]:
    """Return deterministic graph-feature names."""
    pairs = [
        (source, target)
        for source in variables
        for target in variables
        if source != target
    ]
    prefixes: list[str]
    if feature_set == "consensus":
        prefixes = ["B"]
    elif feature_set == "raw":
        prefixes = ["RAW"]
    elif feature_set == "probability":
        prefixes = ["P"]
    elif feature_set == "total_effect":
        prefixes = ["TE"]
    elif feature_set == "combined":
        prefixes = ["B", "P", "TE"]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    columns = [
        f"{prefix}::{source}->{target}"
        for prefix in prefixes
        for source, target in pairs
    ]
    columns.extend(
        [
            "graph::n_samples",
            "graph::consensus_edge_count",
            "graph::mean_abs_consensus_effect",
            "graph::mean_edge_probability",
            "graph::mean_edge_entropy",
        ]
    )
    return columns


def build_graph_features(
    graph_rows: pd.DataFrame,
    feature_set: str,
    excluded_variables: set[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Flatten graph matrices into one fixed-length feature row per pixel."""
    if graph_rows.empty:
        raise click.ClickException("The graph table is empty.")

    first_variables = list(json.loads(graph_rows.iloc[0]["variable_names_json"]))
    included_variables = [
        variable
        for variable in first_variables
        if variable not in excluded_variables
    ]
    if len(included_variables) < 2:
        raise click.ClickException(
            "Fewer than two graph variables remain after exclusions."
        )

    included_indices = [
        first_variables.index(variable)
        for variable in included_variables
    ]
    pairs = [
        (source_idx, target_idx)
        for source_idx in included_indices
        for target_idx in included_indices
        if source_idx != target_idx
    ]

    records: list[dict[str, Any]] = []
    for row in tqdm(
        graph_rows.itertuples(index=False),
        total=len(graph_rows),
        desc="Extracting graph features",
    ):
        variables = list(json.loads(row.variable_names_json))
        if variables != first_variables:
            raise click.ClickException(
                "Graph rows do not all use the same variable ordering."
            )

        raw = parse_matrix(row.adjacency_raw_json, len(variables))
        probability = parse_matrix(
            row.edge_probability_json,
            len(variables),
        )
        consensus = parse_matrix(
            row.adjacency_consensus_json,
            len(variables),
        )
        total = total_effect_matrix(consensus)

        record: dict[str, Any] = {
            "row": int(row.row),
            "col": int(row.col),
        }

        matrices: list[tuple[str, np.ndarray]]
        if feature_set == "consensus":
            matrices = [("B", consensus)]
        elif feature_set == "raw":
            matrices = [("RAW", raw)]
        elif feature_set == "probability":
            matrices = [("P", probability)]
        elif feature_set == "total_effect":
            matrices = [("TE", total)]
        else:
            matrices = [
                ("B", consensus),
                ("P", probability),
                ("TE", total),
            ]

        for prefix, matrix in matrices:
            for source_idx, target_idx in pairs:
                source = variables[source_idx]
                target = variables[target_idx]
                record[f"{prefix}::{source}->{target}"] = float(
                    matrix[target_idx, source_idx]
                )

        off_diagonal = ~np.eye(len(variables), dtype=bool)
        included_mask = np.zeros_like(off_diagonal)
        for source_idx, target_idx in pairs:
            included_mask[target_idx, source_idx] = True

        consensus_values = consensus[included_mask]
        probability_values = probability[included_mask]
        record["graph::n_samples"] = int(row.n_samples)
        record["graph::consensus_edge_count"] = int(
            np.count_nonzero(consensus_values)
        )
        record["graph::mean_abs_consensus_effect"] = float(
            np.nanmean(np.abs(consensus_values))
        )
        record["graph::mean_edge_probability"] = float(
            np.nanmean(probability_values)
        )
        record["graph::mean_edge_entropy"] = float(
            np.nanmean(binary_entropy(probability_values))
        )
        records.append(record)

    feature_df = pd.DataFrame(records)
    feature_columns = [
        column
        for column in graph_feature_columns(
            included_variables,
            feature_set,
        )
        if column in feature_df.columns
    ]
    return feature_df, feature_columns, included_variables


def compute_raw_summary_features(
    ard_db: Path,
    table: str,
    graph_pixels: pd.DataFrame,
    variables: Sequence[str],
    graph_window_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate raw-variable means and standard deviations over each graph window."""
    if not variables:
        return graph_pixels[["row", "col"]].copy(), []

    con = duckdb.connect(ard_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table not in tables:
            raise click.ClickException(
                f"{table!r} not found in {ard_db}. "
                f"Available tables: {sorted(tables)}"
            )
        available_columns = set(
            con.execute(
                f"DESCRIBE {ensure_identifier(table)}"
            ).fetchdf()["column_name"]
        )
        usable_variables = [
            variable
            for variable in variables
            if variable in available_columns
        ]
        missing_variables = sorted(set(variables) - set(usable_variables))
        if missing_variables:
            click.echo(
                "Skipping raw baseline variables absent from the ARD table: "
                + ", ".join(missing_variables)
            )
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
            mean_name = f"raw::mean::{variable}"
            std_name = f"raw::std::{variable}"
            aggregates.extend(
                [
                    f"AVG(a.{variable_sql}) AS {ensure_identifier(mean_name.replace(':', '_'))}",
                    f"STDDEV_POP(a.{variable_sql}) AS {ensure_identifier(std_name.replace(':', '_'))}",
                ]
            )
            raw_columns.extend(
                [
                    mean_name.replace(":", "_"),
                    std_name.replace(":", "_"),
                ]
            )

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


def add_spatial_blocks(
    samples: pd.DataFrame,
    block_size_km: float,
) -> pd.DataFrame:
    """Assign equal-area European spatial block identifiers."""
    x, y = transform_coordinates(
        "EPSG:4326",
        "EPSG:3035",
        samples["longitude"].astype(float).tolist(),
        samples["latitude"].astype(float).tolist(),
    )
    result = samples.copy()
    result["block_x_m"] = np.asarray(x, dtype=float)
    result["block_y_m"] = np.asarray(y, dtype=float)
    block_size_m = block_size_km * 1000.0
    result["spatial_block"] = [
        f"{math.floor(xx / block_size_m)}_{math.floor(yy / block_size_m)}"
        for xx, yy in zip(
            result["block_x_m"],
            result["block_y_m"],
            strict=True,
        )
    ]
    return result


def choose_number_of_folds(
    labels: pd.Series,
    groups: pd.Series,
    requested_folds: int,
) -> int:
    """Choose a feasible spatial-fold count from class-wise block support."""
    grouped = pd.DataFrame({"label": labels, "group": groups})
    groups_per_class = grouped.groupby("label")["group"].nunique()
    maximum = int(groups_per_class.min())
    folds = min(requested_folds, maximum)
    if folds < 2:
        raise click.ClickException(
            "At least one class occurs in fewer than two spatial blocks. "
            "Increase the number of samples, reduce block size, lower purity, "
            "or remove rare classes."
        )
    if folds < requested_folds:
        click.echo(
            f"Reducing spatial folds from {requested_folds} to {folds} "
            "because of class-wise block support."
        )
    return folds


def make_classifier(
    classifier_name: str,
    seed: int,
    trees: int,
    workers: int,
) -> Pipeline:
    """Construct the requested classifier pipeline."""
    if classifier_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=trees,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=workers,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", estimator),
            ]
        )

    if classifier_name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            solver="lbfgs",
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")


def evaluate_model(
    model_name: str,
    model: Pipeline,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    labels: pd.Series,
    groups: pd.Series,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model on precomputed spatial splits."""
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    matrix = features[list(feature_columns)]

    for fold, (train_indices, test_indices) in enumerate(splits):
        fold_model = clone(model)
        fold_model.fit(
            matrix.iloc[train_indices],
            labels.iloc[train_indices],
        )
        predicted = fold_model.predict(matrix.iloc[test_indices])

        y_true = labels.iloc[test_indices]
        metrics.append(
            {
                "model": model_name,
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, predicted)
                ),
                "macro_f1": float(
                    f1_score(
                        y_true,
                        predicted,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        y_true,
                        predicted,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )

        for sample_index, true_value, predicted_value in zip(
            test_indices,
            y_true,
            predicted,
            strict=True,
        ):
            predictions.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "spatial_block": str(groups.iloc[sample_index]),
                    "true_class": str(true_value),
                    "predicted_class": str(predicted_value),
                }
            )

    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def fit_final_model_and_importance(
    model: Pipeline,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    labels: pd.Series,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit on all samples and derive model-native feature importance."""
    final_model = clone(model)
    final_model.fit(features[list(feature_columns)], labels)

    estimator = final_model.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.mean(np.abs(np.asarray(estimator.coef_)), axis=0)
    else:
        importance = np.full(len(feature_columns), np.nan)

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    return final_model, importance_df


def plot_confusion(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Plot a row-normalized confusion matrix from out-of-fold predictions."""
    subset = predictions[predictions["model"] == model_name]
    matrix = confusion_matrix(
        subset["true_class"],
        subset["predicted_class"],
        labels=list(class_names),
        normalize="true",
    )
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    display.plot(
        ax=axis,
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True,
    )
    axis.set_title(f"Spatial cross-validation: {model_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot mean cross-validation metrics with fold standard deviations."""
    summary = (
        metrics.groupby("model")[["balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    models = summary.index.tolist()
    positions = np.arange(len(models))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(
        positions - width / 2,
        summary[("balanced_accuracy", "mean")],
        width=width,
        yerr=summary[("balanced_accuracy", "std")],
        label="Balanced accuracy",
        capsize=3,
    )
    axis.bar(
        positions + width / 2,
        summary[("macro_f1", "mean")],
        width=width,
        yerr=summary[("macro_f1", "std")],
        label="Macro F1",
        capsize=3,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Spatial cross-validation score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(
    importance: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    """Plot the most important final-model features."""
    subset = importance.head(top_n).sort_values("importance")
    figure, axis = plt.subplots(
        figsize=(10.0, max(5.0, 0.28 * len(subset)))
    )
    axis.barh(subset["feature"], subset["importance"])
    axis.set_xlabel("Model-native feature importance")
    axis.set_title(f"Top {len(subset)} graph-validation features")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_map(samples: pd.DataFrame, output_path: Path) -> None:
    """Plot retained land-cover labels in geographic coordinates."""
    classes = sorted(samples["landcover_class"].unique())
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    for class_name in classes:
        subset = samples[samples["landcover_class"] == class_name]
        axis.scatter(
            subset["longitude"],
            subset["latitude"],
            s=4,
            label=class_name,
            alpha=0.7,
        )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Land-cover labels retained for graph validation")
    axis.legend(markerscale=3, fontsize=8)
    axis.set_aspect("equal", adjustable="box")
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
    "--graph-table",
    default="pixel_graphs",
    show_default=True,
    help="Table containing the graph-discovery output.",
)
@click.option(
    "--graph-window-size",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "Neighborhood radius used during graph discovery. "
        "0=1x1, 1=3x3, 2=5x5."
    ),
)
@click.option(
    "--feature-set",
    type=click.Choice(
        [
            "consensus",
            "raw",
            "probability",
            "total_effect",
            "combined",
        ]
    ),
    default="combined",
    show_default=True,
    help="Graph representation supplied to the classifier.",
)
@click.option(
    "--exclude-variable",
    "excluded_variables",
    multiple=True,
    default=("month_sin", "month_cos"),
    show_default=True,
    help="Graph variable to omit from features. Repeat as needed.",
)
@click.option(
    "--class-set",
    type=click.Choice(["vegetation", "terrestrial", "all"]),
    default="all",
    show_default=True,
    help="WorldCover classes retained as classification targets.",
)
@click.option(
    "--min-purity",
    default=0.70,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Minimum dominant-class fraction within a graph footprint.",
)
@click.option(
    "--min-valid-landcover-fraction",
    default=0.80,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Minimum fraction of footprint samples covered by WorldCover.",
)
@click.option(
    "--landcover-samples-per-axis",
    default=11,
    show_default=True,
    type=click.IntRange(min=1),
    help="Regular WorldCover samples along each footprint axis.",
)
@click.option(
    "--min-class-samples",
    default=2,
    show_default=True,
    type=click.IntRange(min=2),
    help="Discard target classes represented by fewer graph samples.",
)
@click.option(
    "--block-size-km",
    default=100.0,
    show_default=True,
    type=click.FloatRange(min=1.0),
    help="Spatial cross-validation block width in kilometres.",
)
@click.option(
    "--folds",
    default=5,
    show_default=True,
    type=click.IntRange(min=2),
    help="Requested number of spatial cross-validation folds.",
)
@click.option(
    "--classifier",
    "classifier_name",
    type=click.Choice(["random_forest", "logistic"]),
    default="random_forest",
    show_default=True,
)
@click.option(
    "--trees",
    default=500,
    show_default=True,
    type=click.IntRange(min=10),
    help="Number of trees for the random-forest classifier.",
)
@click.option(
    "--workers",
    default=-1,
    show_default=True,
    type=int,
    help="Parallel workers used by the random forest.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Random seed for folds and classifiers.",
)
@click.option(
    "--raw-baseline/--no-raw-baseline",
    default=True,
    show_default=True,
    help="Evaluate raw time-series summary and combined baselines.",
)
@click.option(
    "--download/--no-download",
    default=True,
    show_default=True,
    help="Download missing WorldCover tiles automatically.",
)
@click.option(
    "--overwrite-worldcover",
    is_flag=True,
    help="Redownload WorldCover tiles that already exist.",
)
@click.option(
    "--reuse-labels",
    is_flag=True,
    help="Reuse saved land-cover labels from the output database.",
)
@click.option(
    "--reference-raster",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional explicit reference raster, overriding catalog discovery.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Defaults inside the experiment directory.",
)
@click.option(
    "--request-timeout",
    default=120.0,
    show_default=True,
    type=click.FloatRange(min=1.0),
    help="HTTP timeout in seconds.",
)
@click.option(
    "--top-features",
    default=30,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of final-model feature importances to plot.",
)
def validate_graphs_with_landcover(
    config_path: Path,
    graph_table: str,
    graph_window_size: int,
    feature_set: str,
    excluded_variables: tuple[str, ...],
    class_set: str,
    min_purity: float,
    min_valid_landcover_fraction: float,
    landcover_samples_per_axis: int,
    min_class_samples: int,
    block_size_km: float,
    folds: int,
    classifier_name: str,
    trees: int,
    workers: int,
    seed: int,
    raw_baseline: bool,
    download: bool,
    overwrite_worldcover: bool,
    reuse_labels: bool,
    reference_raster: Path | None,
    output_dir: Path | None,
    request_timeout: float,
    top_features: int,
) -> None:
    """Run spatially blocked graph-to-land-cover validation."""
    config = read_config(config_path)
    paths = derive_paths(
        config_path,
        str(config["name"]),
        output_dir,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    required = [paths.graph_db, paths.ard_db]
    if reference_raster is None:
        required.append(paths.source_db)
    require_files(required)

    click.echo("Loading graph database...")
    graph_rows = load_graph_rows(paths.graph_db, graph_table)
    click.echo(f"Loaded {len(graph_rows):,} graph rows.")

    click.echo("Extracting graph features...")
    graph_features, graph_columns, graph_variables = build_graph_features(
        graph_rows=graph_rows,
        feature_set=feature_set,
        excluded_variables=set(excluded_variables),
    )

    if reference_raster is None:
        reference_raster = locate_reference_raster(
            source_db=paths.source_db,
            reference_var=str(config["reference_var"]),
            name_map=config["name_map"],
        )
    click.echo(f"Reference raster: {reference_raster}")

    output_con = duckdb.connect(paths.output_db)
    try:
        existing_tables = set(output_con.sql("SHOW TABLES").df()["name"])
        if reuse_labels and "landcover_labels" in existing_tables:
            click.echo("Reusing saved land-cover labels...")
            landcover_labels = output_con.execute(
                "SELECT * FROM landcover_labels ORDER BY row, col"
            ).fetchdf()
        else:
            with rasterio.open(reference_raster) as reference:
                domain_bounds = graph_domain_bounds_wgs84(
                    graph_rows,
                    reference,
                    graph_window_size,
                )
                click.echo(
                    "Graph-domain WGS84 bounds: "
                    + ", ".join(f"{value:.5f}" for value in domain_bounds)
                )
                tiles = required_worldcover_tiles(
                    domain_bounds,
                    timeout=request_timeout,
                )
                click.echo(
                    f"WorldCover tiles required: {len(tiles)} "
                    f"({', '.join(tiles)})"
                )

                if download:
                    worldcover_paths = download_worldcover(
                        tiles=tiles,
                        output_dir=paths.worldcover_dir,
                        overwrite=overwrite_worldcover,
                        timeout=request_timeout,
                    )
                else:
                    worldcover_paths = [
                        paths.worldcover_dir
                        / (
                            f"ESA_WorldCover_10m_{WORLD_COVER_YEAR}_"
                            f"{WORLD_COVER_VERSION}_{tile}_Map.tif"
                        )
                        for tile in tiles
                    ]
                    require_files(worldcover_paths)

                landcover_labels = label_graph_footprints(
                    graph_rows=graph_rows,
                    reference=reference,
                    worldcover_paths=worldcover_paths,
                    graph_window_size=graph_window_size,
                    samples_per_axis=landcover_samples_per_axis,
                )
            write_dataframe_table(
                output_con,
                landcover_labels,
                "landcover_labels",
            )

        samples = graph_features.merge(
            landcover_labels,
            on=["row", "col"],
            how="inner",
            validate="one_to_one",
        )

        allowed_codes = CLASS_SETS[class_set]
        click.echo(
            f"Class set {class_set!r} allows {len(allowed_codes)} "
            "WorldCover classes."
        )
        samples = samples[
            samples["landcover_code"].isin(allowed_codes)
            & (samples["landcover_purity"] >= min_purity)
            & (
                samples["landcover_valid_fraction"]
                >= min_valid_landcover_fraction
            )
        ].copy()

        class_counts = samples["landcover_class"].value_counts()
        click.echo(
            "Classes after land-cover filters: "
            + ", ".join(
                f"{class_name}={count}"
                for class_name, count in class_counts.sort_index().items()
            )
        )
        retained_classes = class_counts[
            class_counts >= min_class_samples
        ].index
        dropped_classes = class_counts[
            class_counts < min_class_samples
        ]
        if not dropped_classes.empty:
            click.echo(
                "Dropping classes below --min-class-samples: "
                + ", ".join(
                    f"{class_name}={count}"
                    for class_name, count in dropped_classes.sort_index().items()
                )
            )
        samples = samples[
            samples["landcover_class"].isin(retained_classes)
        ].copy()

        if samples["landcover_class"].nunique() < 2:
            raise click.ClickException(
                "Fewer than two land-cover classes remain after filtering."
            )

        samples = add_spatial_blocks(samples, block_size_km)
        samples = samples.sort_values(["row", "col"]).reset_index(drop=True)

        raw_columns: list[str] = []
        if raw_baseline:
            click.echo("Computing raw environmental summary baseline...")
            configured_variables = [
                str(spec["name"])
                for spec in config["columns"]
                if str(spec["name"]) not in set(excluded_variables)
            ]
            raw_features, raw_columns = compute_raw_summary_features(
                ard_db=paths.ard_db,
                table=paths.experiment_name,
                graph_pixels=samples[["row", "col"]],
                variables=configured_variables,
                graph_window_size=graph_window_size,
            )
            samples = samples.merge(
                raw_features,
                on=["row", "col"],
                how="left",
                validate="one_to_one",
            )

        samples.insert(
            0,
            "sample_id",
            np.arange(1, len(samples) + 1, dtype=np.int64),
        )

        write_dataframe_table(
            output_con,
            samples,
            "validation_samples",
        )

        samples_csv = paths.output_dir / "validation_samples.csv"
        samples.to_csv(samples_csv, index=False)

        labels = samples["landcover_class"].astype(str)
        groups = samples["spatial_block"].astype(str)
        feasible_folds = choose_number_of_folds(
            labels,
            groups,
            requested_folds=folds,
        )

        try:
            splitter = StratifiedGroupKFold(
                n_splits=feasible_folds,
                shuffle=True,
                random_state=seed,
            )
            splits = list(
                splitter.split(
                    samples[graph_columns],
                    labels,
                    groups,
                )
            )
        except ValueError:
            click.echo(
                "Falling back to GroupKFold because stratified spatial "
                "fold construction was not feasible."
            )
            splitter = GroupKFold(n_splits=feasible_folds)
            splits = list(
                splitter.split(
                    samples[graph_columns],
                    labels,
                    groups,
                )
            )

        classifier = make_classifier(
            classifier_name=classifier_name,
            seed=seed,
            trees=trees,
            workers=workers,
        )
        dummy = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        )

        model_specs: list[tuple[str, Pipeline, list[str]]] = [
            ("majority", dummy, graph_columns),
            ("graph", classifier, graph_columns),
        ]
        if raw_columns:
            model_specs.extend(
                [
                    ("raw_summary", classifier, raw_columns),
                    (
                        "graph_plus_raw",
                        classifier,
                        graph_columns + raw_columns,
                    ),
                ]
            )

        all_metrics: list[pd.DataFrame] = []
        all_predictions: list[pd.DataFrame] = []
        for model_name, model, columns in model_specs:
            click.echo(f"Evaluating {model_name}...")
            metric_df, prediction_df = evaluate_model(
                model_name=model_name,
                model=model,
                features=samples,
                feature_columns=columns,
                labels=labels,
                groups=groups,
                splits=splits,
            )
            all_metrics.append(metric_df)
            all_predictions.append(prediction_df)

        metrics = pd.concat(all_metrics, ignore_index=True)
        predictions = pd.concat(all_predictions, ignore_index=True)
        prediction_metadata = samples[
            [
                "sample_id",
                "row",
                "col",
                "longitude",
                "latitude",
                "landcover_purity",
            ]
        ].reset_index(drop=True)
        predictions = predictions.merge(
            prediction_metadata.reset_index().rename(
                columns={"index": "sample_index"}
            ),
            on="sample_index",
            how="left",
            validate="many_to_one",
        )

        write_dataframe_table(output_con, metrics, "cv_metrics")
        write_dataframe_table(
            output_con,
            predictions,
            "cv_predictions",
        )
        metrics.to_csv(
            paths.output_dir / "cv_metrics.csv",
            index=False,
        )
        predictions.to_csv(
            paths.output_dir / "cv_predictions.csv",
            index=False,
        )

        final_model_specs: list[tuple[str, list[str]]] = [
            ("graph", graph_columns),
        ]
        if raw_columns:
            final_model_specs.append(
                ("graph_plus_raw", graph_columns + raw_columns)
            )

        importance_tables: list[pd.DataFrame] = []
        graph_importance: pd.DataFrame | None = None

        for final_name, final_columns in final_model_specs:
            final_model, importance = fit_final_model_and_importance(
                model=classifier,
                features=samples,
                feature_columns=final_columns,
                labels=labels,
            )
            joblib.dump(
                {
                    "model": final_model,
                    "feature_columns": final_columns,
                    "class_names": sorted(labels.unique()),
                    "graph_variables": graph_variables,
                    "metadata": {
                        "model_name": final_name,
                        "feature_set": feature_set,
                        "graph_window_size": graph_window_size,
                        "min_purity": min_purity,
                        "class_set": class_set,
                        "block_size_km": block_size_km,
                        "reference_raster": str(reference_raster),
                        "worldcover_year": WORLD_COVER_YEAR,
                        "worldcover_version": WORLD_COVER_VERSION,
                    },
                },
                paths.output_dir / f"{final_name}_classifier.joblib",
            )

            importance = importance.copy()
            importance.insert(0, "model", final_name)
            importance_tables.append(importance)
            importance.to_csv(
                paths.output_dir
                / f"feature_importance_{final_name}.csv",
                index=False,
            )
            plot_feature_importance(
                importance=importance,
                output_path=(
                    paths.output_dir
                    / f"feature_importance_{final_name}.png"
                ),
                top_n=top_features,
            )
            if final_name == "graph":
                graph_importance = importance

        all_importance = pd.concat(
            importance_tables,
            ignore_index=True,
        )
        write_dataframe_table(
            output_con,
            all_importance,
            "feature_importance",
        )
        all_importance.to_csv(
            paths.output_dir / "feature_importance.csv",
            index=False,
        )

        class_summary = (
            samples.groupby(
                ["landcover_code", "landcover_class"],
                as_index=False,
            )
            .agg(
                n_samples=("sample_id", "size"),
                n_spatial_blocks=("spatial_block", "nunique"),
                mean_purity=("landcover_purity", "mean"),
            )
            .sort_values("landcover_code")
        )
        write_dataframe_table(
            output_con,
            class_summary,
            "class_summary",
        )
        class_summary.to_csv(
            paths.output_dir / "class_summary.csv",
            index=False,
        )

        plot_metrics(
            metrics,
            paths.output_dir / "cv_metrics.png",
        )
        class_names = sorted(labels.unique())
        for model_name, _, _ in model_specs:
            plot_confusion(
                predictions=predictions,
                class_names=class_names,
                model_name=model_name,
                output_path=(
                    paths.output_dir
                    / f"confusion_matrix_{model_name}.png"
                ),
            )
        plot_class_map(
            samples=samples,
            output_path=paths.output_dir / "landcover_class_map.png",
        )

        metric_summary = (
            metrics.groupby("model")[
                [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                ]
            ]
            .agg(["mean", "std"])
        )
        summary = {
            "experiment": paths.experiment_name,
            "n_graphs_loaded": int(len(graph_rows)),
            "n_validation_samples": int(len(samples)),
            "classes": class_summary.to_dict(orient="records"),
            "feature_set": feature_set,
            "graph_feature_count": int(len(graph_columns)),
            "raw_feature_count": int(len(raw_columns)),
            "classifier": classifier_name,
            "spatial_folds": int(feasible_folds),
            "block_size_km": float(block_size_km),
            "graph_window_size": int(graph_window_size),
            "landcover_samples_per_axis": int(
                landcover_samples_per_axis
            ),
            "min_purity": float(min_purity),
            "class_set": class_set,
            "metrics": {
                model: {
                    metric: {
                        statistic: float(
                            metric_summary.loc[
                                model,
                                (metric, statistic),
                            ]
                        )
                        for statistic in ["mean", "std"]
                    }
                    for metric in [
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "weighted_f1",
                    ]
                }
                for model in metric_summary.index
            },
        }
        (paths.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    finally:
        output_con.close()

    click.echo("")
    click.echo("Validation complete.")
    click.echo(f"Samples: {len(samples):,}")
    click.echo(f"Classes: {', '.join(sorted(labels.unique()))}")
    click.echo(f"Output directory: {paths.output_dir}")
    click.echo(f"Validation database: {paths.output_db}")


if __name__ == "__main__":
    validate_graphs_with_landcover()
