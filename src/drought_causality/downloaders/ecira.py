from __future__ import annotations

import logging
import zipfile
import datetime
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional

import rasterio
import rioxarray
import xarray as xr
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class ECIRADownloader(BaseDownloader):
    """
    ECIRA / ECIRAv2 downloader (annual, 1 km, Europe) using **curl** for robust download.

    Interface matches your code:
      - download(polygon, year)   
      - save_geotiff(output_dir, basename)
      - check_geotiff_exists_and_validate(output_dir, basename)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "ecira_cache",
        record_id: str = "15569388",
        zip_name: str = "Total_IR.zip",
        crop_code: Optional[str] = None,
    ) -> None:
        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.record_id = record_id
        self.zip_name = zip_name
        self.crop_code = crop_code

    @property
    def frequency(self) -> str:
        return "yearly"

    def download(self, 
                polygon: dict, 
                time_frame: tuple[datetime.datetime, datetime.datetime], 
                output_dir: Path,
                show_progress: bool = True
                 ) -> list[ItemDownloadReport]:
        """
        Download and clip YEARLY irrigation data to a GeoJSON geometry.
        """
        # Extract start and end year from time_frame
        start_year = time_frame[0].year
        final_year = time_frame[1].year

        # Loop over all month-years and download
        download_report_list = []
        iterator = tqdm(range(start_year, final_year + 1), desc="ECIRA", unit="year", disable=not show_progress)
        for year in iterator:
            try:
                # Download and clip ECIRA data for current year
                data = self._download_single_file(polygon, year)

                # Save to GeoTIFF and validate that the file loads
                self._save_geotiff(
                    data=data, 
                    output_dir=output_dir, 
                    basename=f"ECIRA_{year}"
                    )
                self._validate_geotiff(
                    output_dir=output_dir, 
                    basename=f"ECIRA_{year}"
                    )

                # Create successful download report and append to list
                current_report = ItemDownloadReport(
                    data_source="Uni Goettingen",
                    variable_name="ecira",
                    acquisition_time=datetime.datetime(year, 1, 1),
                    path=output_dir / f"ECIRA_{year}.tif",
                    download_successful=True,
                    error=None,
                    metadata=None
                )
                download_report_list.append(current_report)

            except Exception as e:
                # Log error and create failed download report
                logging.error(f"Error downloading ECIRA for {year}: {e}")
                current_report = ItemDownloadReport(
                    data_source="Uni Goettingen",
                    variable_name="ecira",
                    acquisition_time=datetime.datetime(year, 1, 1),
                    path=output_dir / f"ECIRA_{year}.tif",
                    download_successful=False,
                    error=str(e),
                    metadata=None
                )
                download_report_list.append(current_report)
        return download_report_list

    def _zip_url(self) -> str:
        return f"https://zenodo.org/records/{self.record_id}/files/{self.zip_name}?download=1"

    def _local_zip_path(self) -> Path:
        return self.cache_dir / self.zip_name

    def _extract_dir(self) -> Path:
        return self.cache_dir / self.zip_name.replace(".zip", "")

    def _ensure_downloaded(self) -> Path:
        """
        Download the zip via curl (resume + retries).
        """
        local = self._local_zip_path()
        if local.exists() and local.stat().st_size > 0:
            # quick sanity check
            try:
                with zipfile.ZipFile(local, "r"):
                    return local
            except zipfile.BadZipFile:
                local.unlink(missing_ok=True)

        url = self._zip_url()
        logging.info(f"Downloading ECIRA via curl: {url}")

        cmd = [
            "curl",
            "-L",
            "-C", "-",                    # resume
            "--retry", "20",
            "--retry-delay", "2",
            "--retry-all-errors",
            "-o", str(local),
            url,
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"curl download failed (code {res.returncode})\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

        # validate zip
        try:
            with zipfile.ZipFile(local, "r"):
                pass
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Downloaded file is not a valid ZIP: {local}\n{e}")

        return local

    def _ensure_extracted(self) -> Path:
        out_dir = self._extract_dir()
        if out_dir.exists() and any(out_dir.rglob("*")):
            return out_dir

        zpath = self._ensure_downloaded()
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Extracting {zpath} -> {out_dir}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir)

        return out_dir

    def _find_tif_for_year(self, year: int) -> Path:
        root = self._ensure_extracted()
        year_s = str(year)

        tifs = list(root.rglob(f"*{year_s}*.tif")) + list(root.rglob(f"*{year_s}*.tiff"))
        if not tifs:
            all_tifs = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
            tifs = [p for p in all_tifs if year_s in str(p)]

        if not tifs:
            raise RuntimeError(
                f"No GeoTIFF found for year={year} inside {root}. "
                f"Check year is within ECIRA coverage and zip_name is correct."
            )

        if self.crop_code:
            cc = self.crop_code.lower()
            tifs_cc = [p for p in tifs if cc in p.name.lower() or cc in str(p).lower()]
            if tifs_cc:
                tifs = tifs_cc

        tifs = sorted(tifs, key=lambda p: (len(p.parts), len(p.name)))
        return tifs[0]

    # -------------------------
    # Public API (your interface)
    # -------------------------
    def _download_single_file(self, polygon: dict, year: int) -> xr.DataArray:
        """
        ECIRA is annual -> month ignored.
        Clips polygon after reprojecting it to the raster CRS if needed.
        """
        if not isinstance(polygon, dict) or polygon.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("polygon must be a GeoJSON geometry dict (Polygon/MultiPolygon) in EPSG:4326.")
    
        tif_path = self._find_tif_for_year(year)
        logging.info(f"Using ECIRA GeoTIFF: {tif_path}")
    
        with rasterio.open(tif_path) as src:
            src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
    
            # --- Reproject polygon from EPSG:4326 into raster CRS if needed ---
            geom = shape(polygon)
    
            if src_crs.to_epsg() != 4326:
                try:
                    from shapely.ops import transform as shp_transform
                    from pyproj import Transformer
    
                    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
                    geom = shp_transform(transformer.transform, geom)
                    shapes = [geom.__geo_interface__]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reproject polygon from EPSG:4326 to raster CRS {src_crs}. "
                        f"Install pyproj if missing. Error: {e}"
                    )
            else:
                shapes = [polygon]
    
            # --- Clip raster ---
            out_img, out_transform = rio_mask(
                src,
                shapes=shapes,
                crop=True,
                all_touched=True,
                filled=False,
            )
            arr = out_img[0]
    
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
    
            height, width = arr.shape
            xs = out_transform.c + (np.arange(width) + 0.5) * out_transform.a
            ys = out_transform.f + (np.arange(height) + 0.5) * out_transform.e
    
            # NOTE: coords are in src_crs units (meters if EPSG:3035), not lat/lon
            da = xr.DataArray(
                arr,
                dims=("y", "x"),
                coords={"x": xs, "y": ys},
                name="ecira",
                attrs={
                    "source_file": str(tif_path),
                    "year": year,
                    "zip_name": self.zip_name,
                    "record_id": self.record_id,
                    "crop_code": self.crop_code or "",
                    "crs": str(src_crs),
                },
            )
    
            data = (
                da.rio.write_crs(src_crs, inplace=False)
                .rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            )
        return data

    def _save_geotiff(self, data: xr.DataArray, output_dir: Path, basename: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        data.rio.to_raster(geotiff_path)
        return {"ecira": geotiff_path}
