"""Shared ESA WorldCover helpers for Confoundry analysis commands."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import transform as transform_coordinates
from tqdm.auto import tqdm

from confoundry.analysis_helpers import ensure_identifier


WORLD_COVER_YEAR = 2021
WORLD_COVER_VERSION = "v200"
WORLD_COVER_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
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
class LandcoverExperimentPaths:
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


def read_landcover_config(config_path: Path) -> dict[str, Any]:
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


def derive_landcover_paths(
    config_path: Path,
    experiment_name: str,
    output_dir: Path | None,
) -> LandcoverExperimentPaths:
    """Construct conventional land-cover validation input and output paths."""
    experiment_dir = config_path.parent
    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir / "landcover_graph_validation"
    )
    return LandcoverExperimentPaths(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        source_db=experiment_dir / f"{experiment_name}_source_db.duckdb",
        ard_db=experiment_dir / f"{experiment_name}_ard.duckdb",
        graph_db=experiment_dir / f"{experiment_name}_graphs.duckdb",
        output_db=resolved_output_dir / f"{experiment_name}_landcover_validation.duckdb",
        output_dir=resolved_output_dir,
        worldcover_dir=resolved_output_dir / "esa_worldcover_2021",
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
