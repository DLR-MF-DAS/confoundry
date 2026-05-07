from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException
from tqdm import tqdm

import rioxarray
import xarray as xr
from rasterio.crs import CRS

from confoundry.downloaders.downloader import BaseDownloader, ItemDownloadReport


class MODISNDVIDownloader(BaseDownloader):
    """
    Download & clip monthly MODIS NDVI MOD13C2, 0.05° global CMG.
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
        max_workers: int | None = None,
        chunk_size: int = 1024 * 1024,
        timeout: tuple[int, int] = (10, 120),
    ):
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers if max_workers is not None else 6
        self.chunk_size = chunk_size
        self.timeout = timeout

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
        Download and clip monthly MODIS NDVI data to a GeoJSON geometry.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        months = list(self._iter_months(time_frame[0], time_frame[1]))
        reports: list[ItemDownloadReport] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_month, polygon, output_dir, year, month): (year, month)
                for year, month in months
            }

            iterator = as_completed(futures)
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="MODIS NDVI",
                unit="month",
                disable=not show_progress,
            )

            for future in iterator:
                reports.append(future.result())

        return sorted(reports, key=lambda r: r.acquisition_time)

    @staticmethod
    def _iter_months(
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> list[tuple[int, int]]:
        """
        Return inclusive list of (year, month) tuples between start and end.
        """
        months = []
        year, month = start.year, start.month

        while (year, month) <= (end.year, end.month):
            months.append((year, month))

            month += 1
            if month == 13:
                month = 1
                year += 1

        return months

    def _process_month(
        self,
        polygon: dict,
        output_dir: Path,
        year: int,
        month: int,
    ) -> ItemDownloadReport:
        """
        Download, clip, save and validate one MODIS NDVI month.
        """
        basename = f"modis_ndvi_{year}{month:02d}"
        output_path = output_dir / f"{basename}.tif"

        try:
            data = self._download_single_file(polygon, year, month)

            self._save_geotiff(
                data=data,
                output_dir=output_dir,
                basename=basename,
            )

            self._validate_geotiff(
                output_dir=output_dir,
                basename=basename,
            )

            return ItemDownloadReport(
                data_source="terra",
                variable_name="modis_ndvi",
                acquisition_time=datetime.datetime(year, month, 1),
                path=output_path,
                download_successful=True,
                error=None,
                metadata=None,
            )

        except Exception as e:
            logging.exception("Error downloading MODIS NDVI for %04d-%02d", year, month)

            return ItemDownloadReport(
                data_source="terra",
                variable_name="modis_ndvi",
                acquisition_time=datetime.datetime(year, month, 1),
                path=output_path,
                download_successful=False,
                error=str(e),
                metadata=None,
            )

    def _ensure_downloaded(self, year: int, month: int) -> Path:
        """
        Ensure the NetCDF for (year, month) exists locally; download if not.
        Uses an atomic .part file to avoid leaving corrupt cache files.
        """
        local_path = self.cache_dir / f"MODIS_NDVI_{year}{month:02d}.nc"

        if local_path.exists():
            return local_path

        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        url = self.base_url.format(year=year, month=month)

        try:
            with requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()

                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)

            tmp_path.replace(local_path)
            return local_path

        except (
            IncompleteRead,
            ChunkedEncodingError,
            ConnectionError,
            RequestException,
        ) as e:
            tmp_path.unlink(missing_ok=True)
            local_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Error downloading MODIS NDVI data for {year}-{month:02d}: {e}"
            ) from e

    def _download_single_file(
        self,
        polygon: dict,
        year: int,
        month: int,
    ) -> xr.DataArray:
        """
        Download if necessary, open NetCDF, and clip monthly MODIS NDVI
        to a GeoJSON geometry in EPSG:4326.
        """
        nc_path = self._ensure_downloaded(year, month)

        with xr.open_dataset(nc_path) as ds:
            lat_name = next(c for c in ds.coords if c.lower().startswith("lat"))
            lon_name = next(c for c in ds.coords if c.lower().startswith("lon"))

            ndvi_da = (
                ds["ndvi"]
                .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
                .rio.write_crs("EPSG:4326", inplace=False)
            )

            clipped = ndvi_da.rio.clip(
                geometries=[polygon],
                crs=CRS.from_epsg(4326),
                drop=True,
                all_touched=True,
            )

            # Load before leaving the context manager so the NetCDF file can close cleanly.
            return clipped.load()

    def _save_geotiff(
        self,
        data: xr.DataArray,
        output_dir: Path,
        basename: str,
    ):
        """
        Save the clipped MODIS NDVI DataArray to GeoTIFF.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"

        data.isel(time=0).rio.to_raster(geotiff_path)

        return {"modis_ndvi": geotiff_path}
