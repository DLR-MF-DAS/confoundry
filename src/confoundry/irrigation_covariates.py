"""External irrigation/cropland inputs for the irrigation stratification CLI.

Categorical land cover is classified at its native resolution before averaging.
Target windows bound memory even for 10 m WorldCover tiles. No model is refitted.
"""
from __future__ import annotations

import json
import math
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform, transform_bounds
from rasterio.windows import Window, from_bounds

from confoundry.varlingam_aridity_stratification import digest

GMIA_PAGE = "https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/"
GMIA_URL = ("https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "GIS%2Fgmia_v5_aei_pct_asc.zip?alt=media&token=e448ce53-296f-4756-90c1-75c87f74e569")
WORLDCOVER_ROOT = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"


def download(url: str, path: Path) -> None:
    """Cache complete downloads with a local integrity hash, not a publisher hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = path.with_suffix(path.suffix + ".download.json")
    if path.exists() and record.exists():
        meta = json.loads(record.read_text())
        if meta.get("url") == url and meta.get("sha256") == digest(path):
            return
    click.echo(f"Downloading {path.name}")
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    partial.replace(path)
    record.write_text(json.dumps({"url": url, "sha256": digest(path)}, indent=2) + "\n")


def download_gmia(cache: Path) -> Path:
    archive_path = cache / "gmia_v5_aei_pct_asc.zip"
    download(GMIA_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".asc")
                   and not n.startswith("__MACOSX/")]
        if len(members) != 1:
            raise ValueError(f"Expected one GMIA percentage ASCII grid, found {members}")
        # Copy to a fixed filename: archive paths cannot escape the cache.
        result = cache / "gmia_v5_aei_pct.asc"
        with archive.open(members[0]) as source, result.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    with rasterio.open(result) as source:
        if (source.width, source.height) != (4320, 2160) or not np.allclose(
                tuple(source.bounds), [-180, -90, 180, 90], atol=1e-4):
            raise ValueError("Downloaded GMIA grid does not match the documented global 5-minute grid.")
    return result


def study_grid(pixels: pd.DataFrame, reference_path: Path, block_crs: str) -> tuple[pd.DataFrame, dict]:
    with rasterio.open(reference_path) as ref:
        if not ref.crs or ref.transform.b or ref.transform.d or ref.transform.a <= 0 or ref.transform.e >= 0:
            raise ValueError("The original reference grid must have a CRS and be north-up.")
        if pixels.empty or pixels.row.max() >= ref.height or pixels.col.max() >= ref.width:
            raise ValueError("QC coordinates lie outside the original reference grid, or are empty.")
        result = pixels.copy()
        x, y = rasterio.transform.xy(ref.transform, pixels.row, pixels.col)
        result["longitude"], result["latitude"] = transform(ref.crs, "EPSG:4326", x, y)
        result["easting"], result["northing"] = transform(ref.crs, block_crs, x, y)
        if not np.isfinite(result[["longitude", "latitude", "easting", "northing"]]).all().all():
            raise ValueError("Coordinate transformation failed; check the reference and block CRS.")
        r0, c0 = int(pixels.row.min()), int(pixels.col.min())
        height, width = int(pixels.row.max()) - r0 + 1, int(pixels.col.max()) - c0 + 1
        if height * width > 20_000_000:
            raise ValueError("Study extent exceeds 20 million reference cells; check row/column provenance.")
        grid = dict(height=height, width=width, transform=ref.window_transform(Window(c0, r0, width, height)),
                    crs=ref.crs, row_offset=r0, col_offset=c0)
    return result, grid


def download_worldcover(pixels: pd.DataFrame, reference_path: Path, cache: Path) -> list[Path]:
    """Use the official 3-degree tile grid; download only tiles touching fitted cells."""
    grid_file = cache / "esa_worldcover_2020_grid.geojson"
    # ESA's v200 manual uses the same tile grid published with v100.
    download(WORLDCOVER_ROOT + "/v100/2020/esa_worldcover_2020_grid.geojson", grid_file)
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    with rasterio.open(reference_path) as ref:
        # Group adjacent cells into small windows, avoiding one geometry per pixel.
        cells = pixels.assign(r=pixels.row // 16, c=pixels.col // 16)
        regions = []
        for _, group in cells.groupby(["r", "c"]):
            window = Window(int(group.col.min()), int(group.row.min()),
                            int(group.col.max() - group.col.min() + 1),
                            int(group.row.max() - group.row.min() + 1))
            regions.append(box(*transform_bounds(ref.crs, "EPSG:4326", *ref.window_bounds(window), densify_pts=21)))
    region = unary_union(regions)
    paths = []
    for feature in json.loads(grid_file.read_text())["features"]:
        if not shape(feature["geometry"]).intersects(region):
            continue
        tile = feature["properties"]["ll_tile"]
        name = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        path = cache / name
        download(f"{WORLDCOVER_ROOT}/v200/2021/map/{name}", path)
        paths.append(path)
    if not paths:
        raise ValueError("No WorldCover tiles intersect the fitted pixels.")
    return paths


def align_fraction(pixels: pd.DataFrame, grid: dict, paths: list[Path], output: Path,
                   name: str, kind: str, scale: float = 1, classes: tuple[int, ...] = (),
                   min_coverage: float = .5, chunk_cells: int = 4) -> pd.DataFrame:
    """Average fractions/native binary masks over reference footprints.

    `gmia` interprets documented -9 values INSIDE the source grid as zero, uses
    percentage units, and supplies EPSG:4326 if the ASCII grid lacks a CRS.
    Other inputs honour their nodata masks. Outside raster bounds is missing.
    Tile footprints must not overlap. Coverage is relative to the full
    reference footprint, including missing parts; fractions use its valid part.
    GDAL overlap weights are in the raster coordinate system, as in AI alignment.
    """
    if kind not in {"gmia", "fraction", "categorical"} or (kind == "categorical" and not classes):
        raise ValueError("Select gmia, fraction, or categorical with explicit cropland classes.")
    height, width = grid["height"], grid["width"]
    values = np.full((height, width), np.nan, dtype="float64")
    coverage = np.zeros_like(values)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        for source in sources:
            if not source.crs and kind != "gmia":
                raise ValueError(f"Missing CRS in {source.name}")
            if source.transform.b or source.transform.d or source.transform.a <= 0 or source.transform.e >= 0:
                raise ValueError("Covariate rasters must be north-up.")
            if kind == "gmia" and source.crs and source.crs != rasterio.crs.CRS.from_epsg(4326):
                raise ValueError("GMIA mode expects the original EPSG:4326 grid.")
        for i, a in enumerate(sources):
            for b in sources[i + 1:]:
                if a.crs != b.crs:
                    raise ValueError("Multiple tiles must share a CRS; otherwise supply a single mosaic.")
                overlap_x = min(a.bounds.right, b.bounds.right) - max(a.bounds.left, b.bounds.left)
                overlap_y = min(a.bounds.top, b.bounds.top) - max(a.bounds.bottom, b.bounds.bottom)
                tolerance = min(a.transform.a, b.transform.a, -a.transform.e, -b.transform.e) * 1e-7
                if overlap_x > tolerance and overlap_y > tolerance:
                    raise ValueError("Covariate tile footprints overlap; supply disjoint tiles or a single mosaic.")
        occupied = set(zip((pixels.row - grid["row_offset"]) // chunk_cells,
                           (pixels.col - grid["col_offset"]) // chunk_cells))
        for ir, ic in sorted(occupied):
            r, c = int(ir) * chunk_cells, int(ic) * chunk_cells
            h, w = min(chunk_cells, height - r), min(chunk_cells, width - c)
            window = Window(c, r, w, h)
            dst_transform = rasterio.windows.transform(window, grid["transform"])
            dst_bounds = rasterio.windows.bounds(window, grid["transform"])
            numerator, weight = np.zeros((h, w)), np.zeros((h, w))
            for source in sources:
                crs = source.crs or "EPSG:4326"
                ext = transform_bounds(grid["crs"], crs, *dst_bounds, densify_pts=21)
                if ext[0] >= source.bounds.right or ext[2] <= source.bounds.left or ext[1] >= source.bounds.top or ext[3] <= source.bounds.bottom:
                    continue
                native = from_bounds(*ext, transform=source.transform)
                left, top = math.floor(native.col_off) - 1, math.floor(native.row_off) - 1
                right = math.ceil(native.col_off + native.width) + 1
                bottom = math.ceil(native.row_off + native.height) + 1
                if (bottom - top) * (right - left) > 20_000_000:
                    raise ValueError("A covariate read exceeds 20 million cells; reduce --raster-chunk-cells.")
                expanded = Window(left, top, right - left, bottom - top)
                intersection = expanded.intersection(Window(0, 0, source.width, source.height))
                raw = source.read(1, window=intersection, masked=True, out_dtype="float64")
                data = raw.filled(np.nan)
                if kind == "gmia":
                    data[raw.data == -9] = 0  # FAO: no irrigation, NOT missing information.
                    data *= .01
                elif kind == "categorical":
                    data = np.where(np.isfinite(data), np.isin(data, classes).astype(float), np.nan)
                else:
                    data *= scale
                finite = np.isfinite(data)
                if np.any((data[finite] < 0) | (data[finite] > 1 + 1e-7)):
                    raise ValueError(f"{name} contains values outside [0,1]; check product and scale.")
                data = np.clip(data, 0, 1)
                # Padding supplies zeros to coverage, including beyond tile borders.
                padded = np.full((bottom - top, right - left), np.nan)
                rr, cc = int(intersection.row_off) - top, int(intersection.col_off) - left
                padded[rr:rr + data.shape[0], cc:cc + data.shape[1]] = data
                valid = np.isfinite(padded)
                kwargs = dict(src_transform=source.window_transform(expanded), src_crs=crs,
                              dst_transform=dst_transform, dst_crs=grid["crs"], resampling=Resampling.average)
                mean, fraction = np.full((h, w), np.nan), np.zeros((h, w))
                reproject(padded, mean, src_nodata=np.nan, dst_nodata=np.nan, **kwargs)
                reproject(valid.astype(float), fraction, **kwargs)
                numerator += np.nan_to_num(mean) * fraction
                weight += fraction
            if (weight > 1 + 1e-6).any():
                raise ValueError("Covariate tiles overlap; supply disjoint tiles or a single mosaic.")
            weight = np.minimum(weight, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = numerator / weight
            mean[(weight < min_coverage) | (weight == 0)] = np.nan
            values[r:r+h, c:c+w], coverage[r:r+h, c:c+w] = mean, weight
    with rasterio.open(output, "w", driver="GTiff", height=height, width=width, count=2,
                       dtype="float64", crs=grid["crs"], transform=grid["transform"],
                       nodata=np.nan, compress="deflate") as dest:
        dest.write(values, 1)
        dest.write(coverage, 2)
        dest.set_band_description(1, name)
        dest.set_band_description(2, "Valid fraction of full reference footprint")
        dest.update_tags(reference_row_offset=grid["row_offset"], reference_col_offset=grid["col_offset"],
                         aggregation="GDAL average; fraction within valid area", input_kind=kind)
    result = pixels.copy()
    rr = pixels.row.to_numpy() - grid["row_offset"]
    cc = pixels.col.to_numpy() - grid["col_offset"]
    result[name], result[name + "_coverage"] = values[rr, cc], coverage[rr, cc]
    return result
