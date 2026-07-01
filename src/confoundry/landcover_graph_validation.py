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
land-cover product with a compact 11-class legend. The default ``vegetation``
class set keeps tree cover, shrubland, grassland, and cropland. Use
``--class-set terrestrial`` or ``--class-set all`` for broader experiments.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import click
import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from rasterio.windows import Window, bounds as window_bounds
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

from confoundry.per_pixel_graph_discovery import quote_identifier


WORLD_COVER_YEAR = 2021
WORLD_COVER_VERSION = "v200"
WORLD_COVER_PREFIX = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
)
WORLD_COVER_GRID_URL = (
    f"{WORLD_COVER_PREFIX}/v100/2020/"
    "esa_worldcover_2020_grid.geojson"
)

WORLD_COVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare or sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

CLASS_SETS: dict[str, set[int]] = {
    "vegetation": {10, 20, 30, 40},
    "terrestrial": {10, 20, 30, 40, 50, 60},
    "all": set(WORLD_COVER_CLASSES),
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths derived from a Confoundry experiment configuration."""

    experiment_dir: Path
    experiment_name: str
    source_db: Path
    ard_db: Path
    graph_db: Path
    output_db: Path
    output_dir: Path
    worldcover_dir: Path


@dataclass
class OpenRaster:
    """One open WorldCover raster and its geographic bounds."""

    path: Path
    dataset: rasterio.io.DatasetReader
    left: float
    bottom: float
    right: float
    top: float


def ensure_identifier(identifier: str) -> str:
    """Validate and quote a DuckDB table or column identifier."""
    return quote_identifier(identifier)


def write_dataframe_table(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Create or replace a DuckDB table from a pandas data frame."""
    table_sql = ensure_identifier(table_name)
    con.register("_validation_df", df)
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE {table_sql} "
            "AS SELECT * FROM _validation_df"
        )
    finally:
        con.unregister("_validation_df")


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and minimally validate the experiment YAML file."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd)

    required = ["name", "reference_var", "name_map", "columns"]
    missing = [key for key in required if key not in config]
    if missing:
        raise click.ClickException(
            f"Configuration is missing required keys: {missing}"
        )
    return config


def derive_paths(
    config_path: Path,
    experiment_name: str,
    output_dir: Path | None,
) -> ExperimentPaths:
    """Construct conventional Confoundry input and output paths."""
    experiment_dir = config_path.parent
    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir / "landcover_graph_validation"
    )
    return ExperimentPaths(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        source_db=experiment_dir / f"{experiment_name}_source_db.duckdb",
        ard_db=experiment_dir / f"{experiment_name}_ard.duckdb",
        graph_db=experiment_dir / f"{experiment_name}_graphs.duckdb",
        output_db=resolved_output_dir / f"{experiment_name}_landcover_validation.duckdb",
        output_dir=resolved_output_dir,
        worldcover_dir=resolved_output_dir / "esa_worldcover_2021",
    )


def require_files(paths: Iterable[Path]) -> None:
    """Raise a user-facing error when any required input file is absent."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise click.ClickException(
            "Required input files are missing:\n" + formatted
        )


def locate_reference_raster(
    source_db: Path,
    reference_var: str,
    name_map: Mapping[str, str],
) -> Path:
    """Find an existing GeoTIFF corresponding to the configured reference variable."""
    reverse_names = {
        source_name
        for source_name, normalized_name in name_map.items()
        if normalized_name == reference_var
    }
    reverse_names.add(reference_var)

    con = duckdb.connect(source_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if "geotiff_catalog" not in tables:
            raise click.ClickException(
                f"'geotiff_catalog' is absent from {source_db}."
            )
        catalog = con.execute(
            """
            SELECT variable_name, root_dir, file_name, year, month
            FROM geotiff_catalog
            ORDER BY year, month
            """
        ).fetchdf()
    finally:
        con.close()

    candidates: list[Path] = []
    for row in catalog.itertuples(index=False):
        if str(row.variable_name) not in reverse_names:
            continue
        candidate = Path(str(row.root_dir)) / str(row.file_name)
        if candidate.exists():
            candidates.append(candidate)

    if not candidates:
        raise click.ClickException(
            "Could not locate an existing reference raster for "
            f"{reference_var!r}. Source names considered: "
            f"{sorted(reverse_names)}"
        )
    return candidates[0]


def load_graph_rows(graph_db: Path, table: str) -> pd.DataFrame:
    """Load graph matrices and metadata from the graph-discovery database."""
    con = duckdb.connect(graph_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table not in tables:
            raise click.ClickException(
                f"{table!r} not found in {graph_db}. "
                f"Available tables: {sorted(tables)}"
            )

        columns = set(
            con.execute(
                f"DESCRIBE {ensure_identifier(table)}"
            ).fetchdf()["column_name"]
        )
        required = {
            "row",
            "col",
            "n_samples",
            "variable_names_json",
            "adjacency_raw_json",
            "edge_probability_json",
            "adjacency_consensus_json",
        }
        missing = required - columns
        if missing:
            raise click.ClickException(
                f"Graph table {table!r} is missing columns: {sorted(missing)}"
            )

        return con.execute(
            f"""
            SELECT
                row,
                col,
                n_samples,
                variable_names_json,
                adjacency_raw_json,
                edge_probability_json,
                adjacency_consensus_json
            FROM {ensure_identifier(table)}
            ORDER BY row, col
            """
        ).fetchdf()
    finally:
        con.close()


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


def geometry_coordinate_bounds(geometry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Compute a bounding box for a GeoJSON geometry without GeoPandas."""
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        raise ValueError("GeoJSON geometry does not contain coordinates.")

    xs: list[float] = []
    ys: list[float] = []

    def visit(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and isinstance(node[0], (int, float))
            and isinstance(node[1], (int, float))
        ):
            xs.append(float(node[0]))
            ys.append(float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(coordinates)
    if not xs:
        raise ValueError("Could not extract coordinates from GeoJSON geometry.")
    return min(xs), min(ys), max(xs), max(ys)


def boxes_intersect(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two axis-aligned geographic bounding boxes intersect."""
    left_a, bottom_a, right_a, top_a = first
    left_b, bottom_b, right_b, top_b = second
    return not (
        right_a < left_b
        or right_b < left_a
        or top_a < bottom_b
        or top_b < bottom_a
    )


def worldcover_tile_url(tile: str) -> str:
    """Construct the official WorldCover 2021 tile URL."""
    filename = (
        f"ESA_WorldCover_10m_{WORLD_COVER_YEAR}_"
        f"{WORLD_COVER_VERSION}_{tile}_Map.tif"
    )
    return (
        f"{WORLD_COVER_PREFIX}/{WORLD_COVER_VERSION}/"
        f"{WORLD_COVER_YEAR}/map/{filename}"
    )


def required_worldcover_tiles(
    geographic_bounds: tuple[float, float, float, float],
    timeout: float,
) -> list[str]:
    """Find official WorldCover grid tiles intersecting a geographic bounding box."""
    response = requests.get(WORLD_COVER_GRID_URL, timeout=timeout)
    response.raise_for_status()
    grid = response.json()

    tiles: list[str] = []
    for feature in grid.get("features", []):
        properties = feature.get("properties", {})
        tile = properties.get("ll_tile")
        geometry = feature.get("geometry")
        if tile is None or geometry is None:
            continue
        tile_bounds = geometry_coordinate_bounds(geometry)
        if boxes_intersect(tile_bounds, geographic_bounds):
            tiles.append(str(tile))
    if not tiles:
        raise click.ClickException(
            "No ESA WorldCover tiles intersect the graph domain."
        )
    return sorted(set(tiles))


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool,
    timeout: float,
) -> None:
    """Download one file with streaming progress and atomic replacement."""
    if output_path.exists() and not overwrite:
        click.echo(f"Using existing WorldCover tile: {output_path.name}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(
        url,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary_path.open("wb") as fd, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fd.write(chunk)
                progress.update(len(chunk))

    temporary_path.replace(output_path)


def download_worldcover(
    tiles: Sequence[str],
    output_dir: Path,
    overwrite: bool,
    timeout: float,
) -> list[Path]:
    """Download all required WorldCover tiles and return local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for tile in tiles:
        filename = (
            f"ESA_WorldCover_10m_{WORLD_COVER_YEAR}_"
            f"{WORLD_COVER_VERSION}_{tile}_Map.tif"
        )
        path = output_dir / filename
        download_file(
            worldcover_tile_url(tile),
            path,
            overwrite=overwrite,
            timeout=timeout,
        )
        paths.append(path)
    return paths


class WorldCoverSampler:
    """Sample categorical values from multiple local WorldCover tiles."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.rasters: list[OpenRaster] = []
        for path in paths:
            dataset = rasterio.open(path)
            if dataset.crs is None:
                dataset.close()
                raise click.ClickException(
                    f"WorldCover tile has no CRS: {path}"
                )
            bounds = dataset.bounds
            self.rasters.append(
                OpenRaster(
                    path=path,
                    dataset=dataset,
                    left=float(bounds.left),
                    bottom=float(bounds.bottom),
                    right=float(bounds.right),
                    top=float(bounds.top),
                )
            )

    def close(self) -> None:
        """Close all open raster datasets."""
        for raster in self.rasters:
            raster.dataset.close()

    def __enter__(self) -> "WorldCoverSampler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def sample(
        self,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
    ) -> np.ndarray:
        """Sample class codes for arrays of WGS84 coordinates."""
        result = np.zeros(len(longitudes), dtype=np.int16)
        assigned = np.zeros(len(longitudes), dtype=bool)

        for raster in self.rasters:
            mask = (
                (~assigned)
                & (longitudes >= raster.left)
                & (longitudes <= raster.right)
                & (latitudes >= raster.bottom)
                & (latitudes <= raster.top)
            )
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue

            coords = [
                (float(longitudes[index]), float(latitudes[index]))
                for index in indices
            ]
            values = np.fromiter(
                (
                    int(sample[0])
                    for sample in raster.dataset.sample(coords, indexes=1)
                ),
                dtype=np.int16,
                count=len(indices),
            )
            result[indices] = values
            assigned[indices] = True
        return result


def clipped_graph_window(
    row: int,
    col: int,
    window_size: int,
    height: int,
    width: int,
) -> Window | None:
    """Construct a graph footprint clipped to reference-raster dimensions."""
    row_start = max(0, row - window_size)
    row_stop = min(height, row + window_size + 1)
    col_start = max(0, col - window_size)
    col_stop = min(width, col + window_size + 1)

    if row_start >= row_stop or col_start >= col_stop:
        return None
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def sample_grid_in_window(
    window: Window,
    transform: rasterio.Affine,
    samples_per_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate evenly spaced map coordinates inside a reference-raster window."""
    left, bottom, right, top = window_bounds(window, transform)
    x_step = (right - left) / samples_per_axis
    y_step = (top - bottom) / samples_per_axis
    xs = left + x_step * (0.5 + np.arange(samples_per_axis))
    ys = bottom + y_step * (0.5 + np.arange(samples_per_axis))
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    return mesh_x.ravel(), mesh_y.ravel()


def graph_domain_bounds_wgs84(
    graph_rows: pd.DataFrame,
    reference: rasterio.io.DatasetReader,
    graph_window_size: int,
) -> tuple[float, float, float, float]:
    """Compute WGS84 bounds covering all graph footprints."""
    row_min = max(
        0,
        int(graph_rows["row"].min()) - graph_window_size,
    )
    row_max = min(
        reference.height,
        int(graph_rows["row"].max()) + graph_window_size + 1,
    )
    col_min = max(
        0,
        int(graph_rows["col"].min()) - graph_window_size,
    )
    col_max = min(
        reference.width,
        int(graph_rows["col"].max()) + graph_window_size + 1,
    )
    window = Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min,
        height=row_max - row_min,
    )
    left, bottom, right, top = window_bounds(
        window,
        reference.transform,
    )
    xs = [left, left, right, right]
    ys = [bottom, top, bottom, top]
    longitudes, latitudes = transform_coordinates(
        reference.crs,
        "EPSG:4326",
        xs,
        ys,
    )
    return (
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
    )


def label_graph_footprints(
    graph_rows: pd.DataFrame,
    reference: rasterio.io.DatasetReader,
    worldcover_paths: Sequence[Path],
    graph_window_size: int,
    samples_per_axis: int,
) -> pd.DataFrame:
    """Assign dominant WorldCover classes to graph footprints."""
    if reference.crs is None:
        raise click.ClickException("Reference raster does not define a CRS.")

    records: list[dict[str, Any]] = []
    with WorldCoverSampler(worldcover_paths) as sampler:
        for row in tqdm(
            graph_rows.itertuples(index=False),
            total=len(graph_rows),
            desc="Labelling graph footprints",
        ):
            pixel_row = int(row.row)
            pixel_col = int(row.col)
            window = clipped_graph_window(
                row=pixel_row,
                col=pixel_col,
                window_size=graph_window_size,
                height=reference.height,
                width=reference.width,
            )
            if window is None:
                continue

            sample_x, sample_y = sample_grid_in_window(
                window,
                reference.transform,
                samples_per_axis,
            )
            longitudes, latitudes = transform_coordinates(
                reference.crs,
                "EPSG:4326",
                sample_x.tolist(),
                sample_y.tolist(),
            )
            lon_array = np.asarray(longitudes, dtype=float)
            lat_array = np.asarray(latitudes, dtype=float)
            codes = sampler.sample(lon_array, lat_array)
            valid_codes = codes[np.isin(codes, list(WORLD_COVER_CLASSES))]

            center_x, center_y = rasterio.transform.xy(
                reference.transform,
                pixel_row,
                pixel_col,
                offset="center",
            )
            center_lon, center_lat = transform_coordinates(
                reference.crs,
                "EPSG:4326",
                [center_x],
                [center_y],
            )

            if len(valid_codes) == 0:
                dominant_code = 0
                purity = np.nan
                valid_fraction = 0.0
            else:
                counts = Counter(int(code) for code in valid_codes)
                dominant_code, dominant_count = counts.most_common(1)[0]
                purity = dominant_count / len(valid_codes)
                valid_fraction = len(valid_codes) / len(codes)

            records.append(
                {
                    "row": pixel_row,
                    "col": pixel_col,
                    "longitude": float(center_lon[0]),
                    "latitude": float(center_lat[0]),
                    "landcover_code": int(dominant_code),
                    "landcover_class": WORLD_COVER_CLASSES.get(
                        int(dominant_code),
                        "Unknown",
                    ),
                    "landcover_purity": float(purity),
                    "landcover_valid_fraction": float(valid_fraction),
                    "landcover_sample_count": int(len(valid_codes)),
                }
            )
    return pd.DataFrame(records)


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
    default="vegetation",
    show_default=True,
    help="WorldCover classes retained as classification targets.",
)
@click.option(
    "--min-purity",
    default=0.80,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Minimum dominant-class fraction within a graph footprint.",
)
@click.option(
    "--min-valid-landcover-fraction",
    default=0.90,
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
    default=100,
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
        samples = samples[
            samples["landcover_code"].isin(allowed_codes)
            & (samples["landcover_purity"] >= min_purity)
            & (
                samples["landcover_valid_fraction"]
                >= min_valid_landcover_fraction
            )
        ].copy()

        class_counts = samples["landcover_class"].value_counts()
        retained_classes = class_counts[
            class_counts >= min_class_samples
        ].index
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
