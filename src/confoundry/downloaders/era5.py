from __future__ import annotations

import os
import logging
import datetime
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import rioxarray
import xarray as xr
from tqdm import tqdm
from rasterio.crs import CRS
from shapely.geometry import shape

from confoundry.downloaders.downloader import BaseDownloader, ItemDownloadReport

import threading

_NETCDF_LOCK = threading.Lock()
_GDAL_LOCK = threading.Lock()


class ERA5Downloader(BaseDownloader):
    """
    Generic downloader for ERA5-Land monthly data via CDS API.

    Downloads, clips, caches and saves ERA5 variables as GeoTIFFs.
    """

    def __init__(
        self,
        engine: str = "netcdf4",
        cache_dir: Union[str, Path] = "era5_cache",
        max_workers: int | None = None,
        quiet_cds: bool = True,
        **kwargs,
    ):
        self.engine = engine
        self.cds_retry_max = int(
            kwargs.pop("cds_retry_max", os.environ.get("CONFOUNDRY_CDS_RETRY_MAX", 5))
        )
        self.cds_sleep_max = int(
            kwargs.pop("cds_sleep_max", os.environ.get("CONFOUNDRY_CDS_SLEEP_MAX", 30))
        )
        self.cds_timeout = int(
            kwargs.pop("cds_timeout", os.environ.get("CONFOUNDRY_CDS_TIMEOUT", 300))
        )

        self.product_type = "monthly_averaged_reanalysis"
        self.dataset = "reanalysis-era5-land-monthly-means"
        self.time_key = "time"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # CDS can throttle aggressively, so keep the default conservative.
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)

        self.quiet_cds = quiet_cds
        if quiet_cds:
            self._silence_cds_logging()
        try:
            self.variables = kwargs['variables']
        except KeyError:
            raise RuntimeError("You must specify at least one ERA5 variable")

    @property
    def frequency(self) -> str:
        return "monthly"

    def download(self, polygon, time_frame, output_dir, show_progress):
        """
        Download and clip monthly ERA5 data to a GeoJSON geometry.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        months = list(self._iter_months(time_frame[0], time_frame[1]))
        reports = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_month, polygon, output_dir, year, month): (year, month)
                for year, month in months
            }

            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="ERA5",
                unit="month",
                disable=not show_progress,
            )

            for future in iterator:
                reports.extend(future.result())

        return sorted(
            reports,
            key=lambda r: (r.acquisition_time, r.variable_name),
        )

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

    @staticmethod
    def _silence_cds_logging() -> None:
        """
        Suppress noisy CDS / urllib3 info messages.
        """
        for logger_name in ("cdsapi", "ecmwfapi", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def _new_client(self) -> cdsapi.Client:
        """
        Create a fresh CDS client.

        Do not share one cdsapi.Client across worker threads.
        """
        try:
            return cdsapi.Client(
                quiet=self.quiet_cds,
                progress=False,
                retry_max=self.cds_retry_max,
                sleep_max=self.cds_sleep_max,
                timeout=self.cds_timeout,
            )
        except TypeError:
            logging.warning(
                "Installed cdsapi.Client does not accept retry_max/sleep_max/timeout; "
                "falling back to default CDS retry behavior."
            )
            return cdsapi.Client(
                quiet=self.quiet_cds,
                progress=False,
            )

    def _process_month(self, polygon, output_dir, year, month):
        """
        Download, clip, save and validate one ERA5 month.
        """
        basename = f"era5_{year}{month:02d}"

        try:
            data = self._download_single_file(polygon, year, month)

            missing_vars = [
                var['short_name'] for var in self.variables
                if var['short_name'] not in data.data_vars
            ]

            if missing_vars:
                logging.warning(
                    "Missing ERA5 variables for %04d-%02d: %s",
                    year,
                    month,
                    missing_vars,
                )

            save_paths = self._save_geotiff(data, output_dir, basename)
            validate_paths = self._validate_geotiff(output_dir, basename)

            reports = []
            for var in self.variables:
                path = save_paths.get(var['full_name'], output_dir / f"{basename}_{var['full_name']}.tif")
                valid = validate_paths.get(path, False)

                error = None
                if var in missing_vars:
                    valid = False
                    error = f"Variable '{var['full_name']}' missing in downloaded file."
                elif not valid:
                    error = "Validation failed"

                reports.append(
                    ItemDownloadReport(
                        data_source="era5",
                        variable_name=var['full_name'],
                        acquisition_time=datetime.datetime(year, month, 1),
                        path=path,
                        download_successful=valid,
                        error=error,
                        metadata=None,
                    )
                )

            return reports

        except Exception as e:
            logging.exception("Error downloading ERA5 for %04d-%02d: %s", year, month, e)

            return [
                ItemDownloadReport(
                    data_source="era5",
                    variable_name=var['full_name'],
                    acquisition_time=datetime.datetime(year, month, 1),
                    path=output_dir / f"{basename}_{var['full_name']}.tif",
                    download_successful=False,
                    error=str(e),
                    metadata=None,
                )
                for var in self.variables
            ]

    def _target_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"ERA5_{year}{month:02d}.nc"

    def _tmp_path(self, year: int, month: int) -> Path:
        """
        Unique temporary path for a month.

        Useful if a previous interrupted run left temp files behind.
        """
        return self.cache_dir / f"_tmp_ERA5_{year}{month:02d}_{os.getpid()}.download"

    def _ensure_downloaded(self, polygon: dict, year: int, month: int) -> Path:
        """
        Ensure the NetCDF for (year, month) exists locally; download if missing.
        """
        target = self._target_path(year, month)

        if target.exists() and target.stat().st_size > 0:
            return target

        tmp_path = self._tmp_path(year, month)
        tmp_path.unlink(missing_ok=True)

        request = self._build_request(polygon, year, month)

        max_attempts = 5
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                tmp_path.unlink(missing_ok=True)

                client = self._new_client()
                client.retrieve(
                    self.dataset,
                    request,
                    str(tmp_path),
                )

                final_path = self._normalize_cds_download(tmp_path, target)
                return final_path

            except Exception as exc:
                last_error = exc
                tmp_path.unlink(missing_ok=True)
                target.unlink(missing_ok=True)

                if attempt < max_attempts:
                    wait_seconds = min(120, 5 * 2 ** (attempt - 1))
                    logging.warning(
                        "Download failed for ERA5 %04d-%02d on attempt %d/%d: %s. "
                        "Retrying in %d seconds.",
                        year,
                        month,
                        attempt,
                        max_attempts,
                        exc,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"Error downloading ERA5 data for {year:04d}-{month:02d} after "
            f"{max_attempts} attempts: {last_error}"
        ) from last_error

    def _build_request(self, polygon: dict, year: int, month: int) -> dict:
        """
        Build CDS API request for one month and polygon bounds.
        """
        geom = shape(polygon)
        minx, miny, maxx, maxy = geom.bounds

        # CDS area order is North, West, South, East.
        area = [maxy, minx, miny, maxx]

        return {
            "format": "netcdf",
            "product_type": self.product_type,
            "variable": list([var['full_name'] for var in self.variables]),
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": "00:00",
            "area": area,
        }

    def _normalize_cds_download(self, tmp_path: Path, target: Path) -> Path:
        """
        Convert CDS output into a cached NetCDF file.

        CDS may return either a NetCDF file directly or a ZIP containing one.
        """
        with tmp_path.open("rb") as f:
            header = f.read(4)

        if header.startswith(b"PK\x03\x04"):
            return self._extract_netcdf_from_zip(tmp_path, target)

        if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
            os.replace(tmp_path, target)
            return target

        with tmp_path.open("rb") as f:
            first_bytes = f.read(200)

        raise RuntimeError(
            "CDS returned unexpected file type for ERA5 request.\n"
            f"Header bytes: {header!r}\n"
            f"First 200 bytes: {first_bytes!r}"
        )

    def _extract_netcdf_from_zip(self, zip_path: Path, target: Path) -> Path:
        """
        Extract the first NetCDF from a CDS ZIP archive and move it to target.
        """
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            with zipfile.ZipFile(zip_path, "r") as z:
                nc_files = [
                    name for name in z.namelist()
                    if name.lower().endswith(".nc")
                ]

                if not nc_files:
                    raise RuntimeError("No NetCDF file found in ERA5 ZIP archive.")

                extracted_path = Path(z.extract(nc_files[0], tmp_dir_path))

            os.replace(extracted_path, target)

        zip_path.unlink(missing_ok=True)
        return target

    def _download_single_file(
        self,
        polygon: dict,
        year: int,
        month: int,
    ) -> xr.Dataset:
        """
        Download if necessary, open NetCDF, rename variables, and clip to polygon.
        """
        nc_path = self._ensure_downloaded(polygon, year, month)

        with _NETCDF_LOCK:
            with xr.open_dataset(nc_path, engine=self.engine) as ds:
                if "valid_time" in ds.dims and self.time_key == "time":
                    ds = ds.rename({"valid_time": "time"})
                ds = ds.load()

        lat_name = next(c for c in ds.coords if c.lower().startswith("lat"))
        lon_name = next(c for c in ds.coords if c.lower().startswith("lon"))

        clipped_vars = {}

        for var_name in ds.data_vars:
            da = ds[var_name]

            da = (
                da
                .rio.set_spatial_dims(
                    x_dim=lon_name,
                    y_dim=lat_name,
                    inplace=False,
                )
                .rio.write_crs("EPSG:4326", inplace=False)
            )

            clipped_vars[var_name] = da.rio.clip(
                [polygon],
                crs=CRS.from_epsg(4326),
                drop=True,
                all_touched=True,
            )

        # Load before leaving the context manager so the NetCDF file closes cleanly.
        return xr.Dataset(clipped_vars).load()

    def _save_geotiff(self, data, output_dir, basename):
        """
        Save each requested ERA5 variable to GeoTIFF.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        for var in self.variables:
            if var['short_name'] not in data:
                continue

            path = output_dir / f"{basename}_{var['full_name']}.tif"

            da = data[var['short_name']]

            if "time" in da.dims:
                da = da.isel(time=0)
            with _GDAL_LOCK:
                da.rio.to_raster(path)
            paths[var['full_name']] = path

        return paths

    def _get_filepaths(self, output_dir: Path, basename: str) -> List[Path]:
        return [
            output_dir / f"{basename}_{var['full_name']}.tif"
            for var in self.variables
        ]
