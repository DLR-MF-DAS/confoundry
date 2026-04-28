from __future__ import annotations


import os
import logging
import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Union, List

import cdsapi
import rioxarray
import xarray as xr
from rasterio.crs import CRS
from shapely.geometry import shape

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class ERA5Downloader(BaseDownloader):
    """
    Generic downloader for ERA5(-Land) monthly data via CDS API.
    Allows specification of variables, product type, and request parameters via config_dict.
    Handles download, clipping, and saving of any ERA5 variable(s) as GeoTIFFs.
    """
    def __init__(
        self,
        variables_dict: dict = {"t2m": "2m_temperature", 
                                "ssrd": "surface_solar_radiation_downwards", 
                                "tp": "total_precipitation", 
                                "swvl1": "volumetric_soil_water_layer_1"},
        engine: str = "netcdf4",
        cache_dir: Union[str, Path] = "era5_cache",
    ):
        # Initialize ERA5Downloader with specified parameters.
        self.engine = engine
        self.variables_dict = variables_dict
        self.product_type = "monthly_averaged_reanalysis"
        self.dataset = "reanalysis-era5-land-monthly-means"
        self.time_key = "time"
        self.client = cdsapi.Client()

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
        iterator = tqdm(download_months, desc="ERA5", unit="month", disable=not show_progress)
        for year, month in iterator:
            basename = f"era5_{year}{month:02d}"
            try:
                # Download and clip ERA5 data for current month-year
                data = self._download_single_file(polygon, year, month)

                # Check for missing variables
                missing_vars = [var for var in self.variables_dict.keys() if var not in data]
                if missing_vars:
                    logging.warning(f"Missing variables for {year}-{month:02d}: {missing_vars}")
                    
                # Save to GeoTIFF and validate that the file loads
                save_paths = self._save_geotiff(data, output_dir, basename)
                validate_paths = self._validate_geotiff(output_dir, basename)

                # Create download reports for each variable and append to list
                for var in self.variables_dict.keys():
                    path = save_paths.get(var, output_dir / f"{basename}_{var}.tif")
                    valid = validate_paths.get(path, False)
                    error = None
                    if var in missing_vars:
                        valid = False
                        error = f"Variable '{var}' missing in downloaded file."
                    elif not valid:
                        error = "Validation failed"
                    current_report = ItemDownloadReport(
                        data_source="era5",
                        variable_name=var,
                        acquisition_time=datetime.datetime(year, month, 1),
                        path=path,
                        download_successful=valid,
                        error=error,
                        metadata=None,
                    )
                    download_report_list.append(current_report)

            except Exception as e:
                logging.error(f"Error downloading ERA5 for {year}-{month:02d}: {e}")
                # Create failed download reports for all variables
                for var in self.variables_dict.keys():
                    fail_path = output_dir / f"{basename}_{var}.tif"
                    current_report = ItemDownloadReport(
                        data_source="era5",
                        variable_name=var,
                        acquisition_time=datetime.datetime(year, month, 1),
                        path=fail_path,
                        download_successful=False,
                        error=str(e),
                        metadata=None,
                    )
                    download_report_list.append(current_report)
        return download_report_list

    def _target_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"ERA5_{year}{month:02d}.nc"

    def _tmp_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"_tmp_ERA5_{year}{month:02d}"

    def _ensure_downloaded(self, polygon: dict, year: int, month: int) -> Path:
        target = self._target_path(year, month)
        if target.exists() and target.stat().st_size > 0:
            return target

        geom = shape(polygon)
        minx, miny, maxx, maxy = geom.bounds
        area = [maxy, minx, miny, maxx]

        cds_vars = list(self.variables_dict.values())
        request = {
            "format": "netcdf",
            "product_type": self.product_type,
            "variable": cds_vars,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": "00:00",
            "area": area,
        }

        tmp_path = self._tmp_path(year, month)
        self.client.retrieve(
            self.dataset,
            request,
            str(tmp_path)
        )

        # Detect file type (ZIP or NetCDF)
        with open(tmp_path, "rb") as f:
            header = f.read(4)
        if header.startswith(b"PK\x03\x04"):
            import zipfile
            with zipfile.ZipFile(tmp_path, "r") as z:
                nc_files = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_files:
                    raise RuntimeError("No NetCDF file found in ERA5 ZIP archive.")
                z.extract(nc_files[0], self.cache_dir)
                os.rename(self.cache_dir / nc_files[0], target)
            os.remove(tmp_path)
            return target
        elif header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
            os.rename(tmp_path, target)
            return target
        else:
            with open(tmp_path, "rb") as f:
                first_bytes = f.read(200)
            raise RuntimeError(
                f"CDS returned unexpected file type for ERA5 request.\n"
                f"Header bytes: {header}\n"
                f"First 200 bytes: {first_bytes}"
            )

    def _download_single_file(self, polygon: dict, year: int, month: int) -> xr.Dataset:
        nc_path = self._ensure_downloaded(polygon, year, month)
        ds = xr.open_dataset(nc_path, engine=self.engine)

        # Optionally rename variables if needed
        if self.variables_dict:
            rename_map = {v: k for k, v in self.variables_dict.items() if v in ds.data_vars}
            ds = ds.rename(rename_map)

        # Optionally fix time dimension
        if "valid_time" in ds.dims and self.time_key == "time":
            ds = ds.rename({"valid_time": "time"})

        # Find spatial coordinate names
        lat_name = [c for c in ds.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in ds.coords if c.lower().startswith("lon")][0]
        clipped_vars = {}
        for v in ds.data_vars:
            da = ds[v]
            da = (
                da
                .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
                .rio.write_crs("EPSG:4326", inplace=False)
            )
            da_clipped = da.rio.clip(
                [polygon],
                crs=CRS.from_epsg(4326),
                drop=True,
                all_touched=True,
            )
            clipped_vars[v] = da_clipped
        data = xr.Dataset(clipped_vars)
        return data

    def _save_geotiff(self, data, output_dir: Path, basename: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for var in self.variables_dict.keys():
            if var in data:
                path = output_dir / f"{basename}_{var}.tif"
                data[var].isel(time=0).rio.to_raster(path)
                paths[var] = path
        return paths

    def _get_filepaths(self, output_dir: Path, basename: str) -> List[Path]:
        return [output_dir / f"{basename}_{var}.tif" for var in self.variables_dict.keys()]
