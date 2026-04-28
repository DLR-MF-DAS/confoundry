from __future__ import annotations


import os
import logging
import datetime
import calendar
import rioxarray
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from typing import Union

import requests
from http.client import IncompleteRead
from requests.exceptions import ChunkedEncodingError, ConnectionError

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class SPEIDownloader(BaseDownloader):
    """Class for downloading SPEI data from the Copernicus Climate Change Service (C3S)"""
    def __init__(
            self, 
            cache_dir: Union[str, Path] = "spei_cache") -> None:
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def frequency(self) -> str:
        return "monthly"

    def download(self, 
                polygon: dict, 
                time_frame: tuple[datetime.datetime, datetime.datetime], 
                output_dir: Path,
                show_progress: bool = True
                 ) -> list[ItemDownloadReport]:
        """
        Download and clip MONTHLY SPEI data to a GeoJSON geometry.
        """
        # Extract start and end year/month from time_frame
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
        iterator = tqdm(download_months, desc="SPEI", unit="month", disable=not show_progress)
        for year, month in iterator:
            try:
                # Download and clip SPEI data for current month-year
                data = self._download_single_file(polygon, year, month)

                # Save to GeoTIFF and validate that the file loads
                self._save_geotiff(
                    data=data, 
                    output_dir=output_dir, 
                    basename=f"SPEI_{year}{month:02d}"
                    )
                self._validate_geotiff(
                    output_dir=output_dir, 
                    basename=f"SPEI_{year}{month:02d}"
                    )

                # Create successful download report and append to list
                current_report = ItemDownloadReport(
                    data_source="CSIC",
                    variable_name="spei",
                    acquisition_time=datetime.datetime(year, month, 1),
                    path=output_dir / f"SPEI_{year}{month:02d}.tif",
                    download_successful=True,
                    error=None,
                    metadata=None
                )
                download_report_list.append(current_report)

            except Exception as e:
                # Log error and create failed download report
                logging.error(f"Error downloading SPEI for {year}-{month:02d}: {e}")
                current_report = ItemDownloadReport(
                    data_source="CSIC",
                    variable_name="spei",
                    acquisition_time=datetime.datetime(year, month, 1),
                    path=output_dir / f"SPEI_{year}{month:02d}.tif",
                    download_successful=False,
                    error=str(e),
                    metadata=None
                )
                download_report_list.append(current_report)
        return download_report_list
    
    def _ensure_downloaded(self) -> Path:
        spei_url = "https://digital.csic.es/bitstream/10261/364137/1/spei01.nc"
        out_nc = f"{self.cache_dir}/spei01.nc"
        headers = {"User-Agent": "Mozilla/5.0"}
        logging.info("Downloading SPEIbase file...")
        if not os.path.exists(out_nc):
            try:
                with requests.get(spei_url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    with open(out_nc, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                logging.info(f"Saved: {out_nc}")
            except (IncompleteRead, ChunkedEncodingError, ConnectionError) as e:
                if os.path.exists(out_nc):
                    os.remove(out_nc)
                raise RuntimeError(f"Error downloading SPEI data: {e}") from e
        return out_nc

    def _download_single_file(self, polygon: dict, year: int, month: int) -> Path:
        """
        Download and clip monthly SPEI data to a GeoJSON geometry.
        """
        out_nc = self._ensure_downloaded()
        ds = xr.open_dataset(out_nc)
        lat_name = [c for c in ds.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in ds.coords if c.lower().startswith("lon")][0]
        logging.info(f"Lat range: {float(ds[lat_name].min())} to {float(ds[lat_name].max())}")
        logging.info(f"Lon range: {float(ds[lon_name].min())} to {float(ds[lon_name].max())}")
        spei_da = ds["spei"]
        spei_da = (
            spei_da
            .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
            .rio.write_crs("EPSG:4326", inplace=False)
        )
        spei_da = spei_da.rio.write_crs("EPSG:4326", inplace=False)
        spei_clipped = spei_da.rio.clip(
            [polygon],
            crs=4326,
        )
        last_day = calendar.monthrange(year, month)[1]
        spei_clipped = spei_clipped.sel(time=slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"))
        single_month = spei_clipped.isel(time=0)
        data = single_month
        return data
    
    def _save_geotiff(self, data: xr.DataArray, output_dir: Path, basename: str):
        """
        Save a clipped ndvi DataArray to GeoTIFF.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        data.rio.to_raster(geotiff_path)
        return {"spei": geotiff_path}
    