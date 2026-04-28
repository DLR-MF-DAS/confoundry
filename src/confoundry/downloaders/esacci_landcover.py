from __future__ import annotations


import logging
import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Union, Dict, Any

import requests
from http.client import IncompleteRead
from requests.exceptions import ChunkedEncodingError, ConnectionError

import rasterio
import rioxarray
import numpy as np
import xarray as xr
from rasterio.mask import mask
from rasterio.crs import CRS

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class ESACCILandCoverDownloader(BaseDownloader):
    """
    ESA CCI Land Cover v2.0.7 annual maps (1992–2015) from CEDA, no account.

    IMPORTANT:
    - Files live directly in the v2.0.7 directory (no year subfolders).
    - Annual NetCDF files are named ...v2.0.7b.nc (note the 'b').
    - Annual GeoTIFF files are named ...v2.0.7.tif
      (GeoTIFF is much smaller than NetCDF).
    """

    BASE_DIR = "https://dap.ceda.ac.uk/neodc/esacci/land_cover/data/land_cover_maps/v2.0.7"

    def __init__(self, cache_dir: Union[str, Path] = "esa_cci_cache") -> None:
        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        Download and clip YEARLY landcover data to a GeoJSON geometry.
        """
        # Extract start and end year/month from time_frame
        start_year = time_frame[0].year
        final_year = time_frame[1].year


        # Loop over all month-years and download
        download_report_list = []
        download_years = list(range(start_year, final_year + 1))
        iterator = tqdm(download_years, desc="ESACCI Land Cover", unit="year", disable=not show_progress)
        for year in iterator:
            try:
                # Download and clip ESACCI Land Cover data for current year
                data = self._download_single_file(polygon, year)

                # Save to GeoTIFF and validate that the file loads
                self._save_geotiff(
                    data=data, 
                    output_dir=output_dir, 
                    basename=f"ESACCI_LC_{year}"
                    )
                self._validate_geotiff(
                    output_dir=output_dir, 
                    basename=f"ESACCI_LC_{year}"
                    )

                # Create successful download report and append to list
                current_report = ItemDownloadReport(
                    data_source="ESA CCI",
                    variable_name="landcover",
                    acquisition_time=datetime.datetime(year, 1, 1),
                    path=output_dir / f"ESACCI_LC_{year}.tif",
                    download_successful=True,
                    error=None,
                    metadata=None
                )
                download_report_list.append(current_report)

            except Exception as e:
                # Log error and create failed download report
                logging.error(f"Error downloading ESA CCI Land Cover for {year}: {e}")
                current_report = ItemDownloadReport(
                    data_source="ESA CCI",
                    variable_name="landcover",
                    acquisition_time=datetime.datetime(year, 1, 1),
                    path=output_dir / f"ESACCI_LC_{year}.tif",
                    download_successful=False,
                    error=str(e),
                    metadata=None
                )
                download_report_list.append(current_report)
        return download_report_list

    @staticmethod
    def _assert_geometry(polygon: Dict[str, Any]) -> None:
        if not isinstance(polygon, dict) or polygon.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                "polygon must be a GeoJSON *geometry* dict (Polygon/MultiPolygon) in EPSG:4326 "
                "(same convention as your other downloaders)."
            )

    def _remote_url(self, year: int) -> str:
        # Example from CEDA listing:
        # ESACCI-LC-L4-LCCS-Map-300m-P1Y-2010-v2.0.7.tif is the GeoTIFF for 2010
        return f"{self.BASE_DIR}/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7.tif"

    def _local_path(self, year: int) -> Path:
        return self.cache_dir / f"ESACCI_LC_{year}_v2.0.7.tif"

    def _ensure_downloaded(self, year: int) -> Path:
        local = self._local_path(year)
        if local.exists() and local.stat().st_size > 0:
            return local

        url = self._remote_url(year)
        logging.info(f"Downloading ESA CCI LC {year} GeoTIFF from {url}")
        
        try:
            with requests.get(url, stream=True, timeout=(20, 600)) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return local
        except (IncompleteRead, ChunkedEncodingError, ConnectionError) as e:
            if local.exists():
                local.unlink()
            raise RuntimeError(f"Error downloading MODIS NDVI data: {e}") from e

    def _download_single_file(self, polygon: dict, year: int) -> xr.DataArray:
        """
        Download annual GeoTIFF and clip to polygon.
        """
        self._assert_geometry(polygon)
        tif_path = self._ensure_downloaded(year)

        # Read + clip with rasterio (memory-efficient)
        with rasterio.open(tif_path) as src:
            out_img, out_transform = mask(
                src,
                shapes=[polygon],
                crop=True,
                all_touched=True,
                filled=False,
            )
            out = out_img[0]  # (1,y,x)->(y,x)

            height, width = out.shape
            xs = out_transform.c + (np.arange(width) + 0.5) * out_transform.a
            ys = out_transform.f + (np.arange(height) + 0.5) * out_transform.e

            da = xr.DataArray(
                out,
                dims=("lat", "lon"),
                coords={"lon": xs, "lat": ys},
                name="lccs_class",
            )

            crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
            da = (
                da.rio.write_crs(crs, inplace=False)
                .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            )
        return da

    def _save_geotiff(self, data: xr.DataArray, output_dir: Union[str, Path], basename: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        data.rio.to_raster(geotiff_path)
        return {"landcover": geotiff_path}
