from __future__ import annotations


import logging
import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Union

import requests
from http.client import IncompleteRead
from requests.exceptions import ChunkedEncodingError, ConnectionError

import rioxarray
import xarray as xr
from rasterio.crs import CRS

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class MODISNDVIDownloader(BaseDownloader):
    """
    Download & clip monthly MODIS NDVI (MOD13C2, 0.05° global CMG).

    Uses ICDC Hamburg preprocessed NetCDF files, 1 file per month, e.g.
    https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/modis_terra_vegetationindex/DATA/2017/MODIS-C061_MOD13C2_NDVI__LPDAAC__0.05deg__MONTHLY__UHAM-ICDC__201706__fv0.01.nc
    """

    def __init__(
        self,
        base_url: str = (
            "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
            "ftpthredds/modis_terra_vegetationindex/DATA/"
            "{year}/MODIS-C061_MOD13C2_NDVI__LPDAAC__0.05deg__MONTHLY__"
            "UHAM-ICDC__{year}{month:02d}__fv0.01.nc"
        ),
        cache_dir: Union[str, Path] = "modis_ndvi_cache",
    ):
        self.base_url = base_url

        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def frequency(self) -> str:
        return "monthly"

    def download(
        self,
        polygon: dict,
        time_frame: tuple[datetime.datetime, datetime.datetime],
        output_dir: Path,
        show_progress: bool = True,
    ) -> list[ItemDownloadReport]:
        """
        Download and clip MONTHLY MODIS NDVI data to a GeoJSON geometry.
        """
        start_year, start_month = time_frame[0].year, time_frame[0].month
        final_year, final_month = time_frame[1].year, time_frame[1].month

        # Build list of (year, month) tuples to download
        download_months = []
        for year in range(start_year, final_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = final_month if year == final_year else 12
            for month in range(month_start, month_end + 1):
                download_months.append((year, month))

        # Loop over all month-years and download
        download_report_list = []
        iterator = tqdm(download_months, desc="MODIS NDVI", unit="month", disable=not show_progress)
        for year, month in iterator:
            basename = f"modis_ndvi_{year}{month:02d}"
            try:
                # Download and clip modis ndvi data for current month-year
                data = self._download_single_file(polygon, year, month)

                # Save to GeoTIFF and validate that the file loads
                self._save_geotiff(
                    data=data,
                    output_dir=output_dir,
                    basename=basename,
                )
                self._validate_geotiff(
                    output_dir=output_dir,
                    basename=basename,
                )

                # Create successful download report and append to list
                current_report = ItemDownloadReport(
                    data_source="terra",
                    variable_name="modis_ndvi",
                    acquisition_time=datetime.datetime(year, month, 1),
                    path=output_dir / f"{basename}.tif",
                    download_successful=True,
                    error=None,
                    metadata=None,
                )
                download_report_list.append(current_report)

            except Exception as e:
                logging.error(f"Error downloading MODIS NDVI for {year}-{month:02d}: {e}")
                current_report = ItemDownloadReport(
                    data_source="terra",
                    variable_name="modis_ndvi",
                    acquisition_time=datetime.datetime(year, month, 1),
                    path=output_dir / f"{basename}.tif",
                    download_successful=False,
                    error=str(e),
                    metadata=None,
                )
                download_report_list.append(current_report)
        return download_report_list

    def _ensure_downloaded(self, year: int, month: int) -> Path:
        """
        Ensure the NetCDF for (year, month) exists locally; download if not.
        Returns the local Path.
        """
        local_path = self.cache_dir / f"MODIS_NDVI_{year}{month:02d}.nc"
        if local_path.exists():
            return local_path

        url = self.base_url.format(year=year, month=month)
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.get(url, headers=headers, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return local_path
        except (IncompleteRead, ChunkedEncodingError, ConnectionError) as e:
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Error downloading MODIS NDVI data: {e}") from e

    def _download_single_file(self, polygon: dict, year: int, month: int) -> xr.DataArray:
        """
        Clip monthly MODIS NDVI to a GeoJSON geometry.

        Parameters
        ----------
        polygon : dict
            GeoJSON *geometry* dict (not Feature!) in EPSG:4326.
        year : int
            Year of the MOD13C2 monthly product (>= 2000).
        month : int
            Month (1–12).

        Returns
        -------
        None. Sets self.data to the clipped NDVI (time, lat, lon) – time has length 1.
        """
        nc_path = self._ensure_downloaded(year, month)

        # Open NetCDF; CF metadata will decode scale_factor & _FillValue
        ds = xr.open_dataset(nc_path)

        # Find lat/lon coordinate names (usually 'lat' / 'lon')
        lat_name = [c for c in ds.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in ds.coords if c.lower().startswith("lon")][0]

        # NDVI variable is named 'ndvi' in the ICDC files
        ndvi_da = ds["ndvi"]

        # Register spatial dims + CRS for rioxarray
        ndvi_da = (
            ndvi_da
            .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
            .rio.write_crs("EPSG:4326", inplace=False)
        )

        # rioxarray expects an iterable of geometries → wrap in list
        geometries = [polygon]

        ndvi_clipped = ndvi_da.rio.clip(
            geometries=geometries,
            crs=CRS.from_epsg(4326),
            drop=True,
            all_touched=True,
        )
        data = ndvi_clipped
        return data
    
    def _save_geotiff(self, data: xr.DataArray, output_dir: Path, basename: str):
        """
        Save the clipped MODIS NDVI DataArray to GeoTIFF.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        data.isel(time=0).rio.to_raster(geotiff_path)
        return {"modis_ndvi": geotiff_path}
    