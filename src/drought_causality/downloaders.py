from __future__ import annotations

import os
import logging
import zipfile
import calendar
import requests
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

import cdsapi
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.mask import mask
from shapely.geometry import shape
from rasterio.warp import reproject
from rasterio.enums import Resampling


Number = Union[int, float]


class SPEIDownloader:
    """Class for downloading SPEI data from the Copernicus Climate Change Service (C3S)"""
    def __init__(
            self, 
            cache_dir: Union[str, Path] = "spei_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_downloaded(self) -> Path:
        spei_url = "https://digital.csic.es/bitstream/10261/364137/1/spei01.nc"
        out_nc = f"{self.cache_dir}/spei01.nc"
        logging.info("Downloading SPEIbase file...")
        if not os.path.exists(out_nc):
            with requests.get(spei_url, stream=True) as r:
                r.raise_for_status()
                with open(out_nc, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("Saved:", out_nc)
        return out_nc

    def download(self, polygon: dict, year: int, month: int) -> Path:
        out_nc = self._ensure_downloaded()
        ds = xr.open_dataset(out_nc)
        lat_name = [c for c in ds.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in ds.coords if c.lower().startswith("lon")][0]
        logging.info("Lat range:", float(ds[lat_name].min()), "to", float(ds[lat_name].max()))
        logging.info("Lon range:", float(ds[lon_name].min()), "to", float(ds[lon_name].max()))
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
        self.data = single_month
    
    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the clipped SPEI DataArray to GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        self.data.rio.to_raster(geotiff_path)
        return [geotiff_path]
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False

class MODISNDVIDownloader:
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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_downloaded(self, year: int, month: int) -> Path:
        """
        Ensure the NetCDF for (year, month) exists locally; download if not.
        Returns the local Path.
        """
        local_path = self.cache_dir / f"MODIS_NDVI_{year}{month:02d}.nc"
        if local_path.exists():
            return local_path

        url = self.base_url.format(year=year, month=month)
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return local_path

    def download(self, polygon: dict, year: int, month: int) -> xr.DataArray:
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
        xarray.DataArray
            Clipped NDVI (time, lat, lon) – time has length 1.
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
        self.data = ndvi_clipped

    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the clipped MODIS NDVI DataArray to GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        self.data.isel(time=0).rio.to_raster(geotiff_path)
        return [geotiff_path]
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False


class ERA5Downloader:
    """
    Downloads ERA5-Land monthly mean 2m temperature (t2m)
    and surface solar radiation downwards (ssrd) via CDS API,
    then clips to a polygon.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "era5_cache",
        engine: str = "netcdf4",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.client = cdsapi.Client()

    def _target_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"ERA5_{year}{month:02d}.nc"

    def _ensure_downloaded(self, polygon: dict, year: int, month: int) -> Path:
        target = self._target_path(year, month)
        if target.exists() and target.stat().st_size > 0:
            return target

        geom = shape(polygon)
        minx, miny, maxx, maxy = geom.bounds
        area = [maxy, minx, miny, maxx]

        request = {
            "format": "netcdf",
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "surface_solar_radiation_downwards",
            ],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": "00:00",
            "area": area,
        }

        tmp_path = self.cache_dir / f"_tmp_ERA5_{year}{month:02d}"
        self.client.retrieve(
            "reanalysis-era5-land-monthly-means",
            request,
            str(tmp_path)
        )
        
        # ---- Detect if file is ZIP or real NetCDF ----
        header = open(tmp_path, "rb").read(4)
        # ZIP magic number = 50 4B 03 04 (PK\003\004)
        if header.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(tmp_path, "r") as z:
                # Find inside .nc file
                nc_files = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_files:
                    raise RuntimeError(
                        f"CDS returned ZIP but no .nc inside! Contents: {z.namelist()}"
                    )

                nc_name = nc_files[0]
                with z.open(nc_name) as zf, open(target, "wb") as out:
                    out.write(zf.read())

            tmp_path.unlink()
            return target

        # Not ZIP: assume it is actually NetCDF
        # NetCDF magic header is 'CDF\001' or '\x89HDF'
        if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
            tmp_path.rename(target)
            return target

        # Otherwise: unknown / HTML / XML error
        first_bytes = open(tmp_path, "rb").read(200)
        raise RuntimeError(
            f"CDS returned unexpected file type for ERA5 request.\n"
            f"Header bytes: {header}\n"
            f"First 200 bytes: {first_bytes}"
        )

    def download(self, polygon: dict, year: int, month: int) -> xr.Dataset:
        """
        Download ERA5-Land t2m & ssrd and clip to GeoJSON polygon.
        """
        nc_path = self._ensure_downloaded(polygon, year, month)

        ds = xr.open_dataset(nc_path)
        # --- FIX 1: Rename valid_time -> time ---
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

            # --- FIX 2: ensure proper datetime ---
        if "time" in ds.coords:
            if not np.issubdtype(ds["time"].dtype, np.datetime64):
                # try CF decode
                try:
                    ds = xr.decode_cf(ds)
                except Exception:
                    pass
        vars_to_keep = [v for v in ("t2m", "ssrd") if v in ds.data_vars]
        ds = ds[vars_to_keep]

        # Spatial coord names (ERA5 always uses lat/lon)
        lat_name = [c for c in ds.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in ds.coords if c.lower().startswith("lon")][0]

        clipped_vars = {}

        for v in vars_to_keep:
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
        self.data = xr.Dataset(clipped_vars)
    
    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the clipped era5 DataArrays to GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        if "t2m" in self.data:
            t2mtiff_path = output_dir / f"{basename}_t2m.tif"
            self.data["t2m"].isel(time=0).rio.to_raster(t2mtiff_path)
            paths.append(t2mtiff_path)
        if "ssrd" in self.data:
            ssrdtiff_path = output_dir / f"{basename}_ssrd.tif"
            self.data["ssrd"].isel(time=0).rio.to_raster(ssrdtiff_path)
            paths.append(ssrdtiff_path)
        return paths
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFFs exist and are valid (not corrupt).
        Returns True if all are valid, False otherwise.
        """
        for var in ("t2m", "ssrd"):
            geotiff_path = output_dir / f"{basename}_{var}.tif"
            if not geotiff_path.exists():
                return False
            try:
                with rasterio.open(geotiff_path) as src:
                    _ = src.read(1, window=((0, 1), (0, 1)))
            except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
                return False
        return True


class ERA5PrecipDownloader:
    """
    Downloads ERA5-Land monthly total precipitation ("tp") using CDS API,
    extracts the NetCDF (CDS returns ZIP!), normalizes the time dimension,
    and clips to a GeoJSON polygon.
    """

    def __init__(self, cache_dir="era5_precip_cache", engine="netcdf4"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.engine = engine
        self.client = cdsapi.Client()

    def _target_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"ERA5_precip_{year}{month:02d}.nc"

    def _tmp_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"_tmp_ERA5_precip_{year}{month:02d}"

    def _ensure_downloaded(self, polygon: dict, year: int, month: int) -> Path:
        """
        Downloads ERA5-Land precipitation if needed, extracts NetCDF, returns path.
        """
        target = self._target_path(year, month)
        if target.exists() and target.stat().st_size > 0:
            return target

        # Spatial subset from polygon bounding box
        geom = shape(polygon)
        minx, miny, maxx, maxy = geom.bounds
        area = [maxy, minx, miny, maxx]

        request = {
            "format": "netcdf",
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["total_precipitation"],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": "00:00",
            "area": area,
        }

        tmp = self._tmp_path(year, month)

        logging.info(f"Downloading ERA5 precipitation {year}-{month:02d} from CDS...")
        self.client.retrieve(
            "reanalysis-era5-land-monthly-means",
            request,
            str(tmp)
        )

        # ---- Detect ZIP vs NetCDF ----
        header = open(tmp, "rb").read(4)

        if header.startswith(b"PK\x03\x04"):
            # ZIP – extract the .nc inside
            with zipfile.ZipFile(tmp, "r") as z:
                nc_files = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_files:
                    raise RuntimeError(
                        f"ZIP returned by CDS contains no .nc files! Contents: {z.namelist()}"
                    )
                with z.open(nc_files[0]) as src, open(target, "wb") as out:
                    out.write(src.read())

            tmp.unlink()
            return target

        # Already a NetCDF? (rare)
        if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
            tmp.rename(target)
            return target

        # Otherwise: HTML/XML error
        content = open(tmp, "rb").read(200)
        raise RuntimeError(
            "CDS returned unexpected file type for ERA5 precipitation.\n"
            f"Header={header}\nFirst bytes={content}"
        )

    def download(self, polygon: dict, year: int, month: int) -> xr.Dataset:
        """
        Returns clipped ERA5 precipitation dataset with correct time dimension.
        """
        path = self._ensure_downloaded(polygon, year, month)

        ds = xr.open_dataset(path, engine=self.engine)

        # ---- Normalize time: valid_time -> time ----
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        if "time" in ds.coords:
            if not np.issubdtype(ds["time"].dtype, np.datetime64):
                try:
                    ds = xr.decode_cf(ds)
                except Exception:
                    pass

        # ---- Extract variable 'tp' ----
        var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]

        # ---- Setup for clipping ----
        lat_name = [c for c in da.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in da.coords if c.lower().startswith("lon")][0]

        da = (
            da
            .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
            .rio.write_crs("EPSG:4326", inplace=False)
        )

        da_clip = da.rio.clip([polygon], CRS.from_epsg(4326), drop=True, all_touched=True)
        self.data = xr.Dataset({"tp": da_clip})

    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the clipped ERA5 precipitation DataArray to GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        if "tp" in self.data:
            geotiff_path = output_dir / f"{basename}.tif"
            self.data["tp"].isel(time=0).rio.to_raster(geotiff_path)
            paths.append(geotiff_path)
        return paths
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False


class ERA5SoilMoistureDownloader:
    """
    Downloads ERA5-Land monthly top-layer soil moisture (swvl1) via CDS API,
    extracts the NetCDF (CDS returns ZIP), normalizes the time dimension,
    and clips to a GeoJSON polygon.

    Soil moisture here is the top layer "volumetric_soil_water_layer_1" (swvl1),
    which is a natural mediator between meteorological drought and vegetation response.
    """

    def __init__(self, cache_dir: str | Path = "era5_soilmoist_cache", engine: str = "netcdf4"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.client = cdsapi.Client()

    def _target_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"ERA5_soilmoist_{year}{month:02d}.nc"

    def _tmp_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"_tmp_ERA5_soilmoist_{year}{month:02d}"

    def _ensure_downloaded(self, polygon: dict, year: int, month: int) -> Path:
        """
        Download ERA5-Land soil moisture if needed, extract NetCDF, return path.
        """
        target = self._target_path(year, month)
        if target.exists() and target.stat().st_size > 0:
            return target

        # Spatial subset from polygon bounding box
        geom = shape(polygon)
        minx, miny, maxx, maxy = geom.bounds
        area = [maxy, minx, miny, maxx]  # [N, W, S, E]

        request = {
            "format": "netcdf",
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["volumetric_soil_water_layer_1"],  # swvl1
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": "00:00",
            "area": area,
        }

        tmp = self._tmp_path(year, month)

        logging.info(f"Downloading ERA5 soil moisture {year}-{month:02d} from CDS...")
        self.client.retrieve(
            "reanalysis-era5-land-monthly-means",
            request,
            str(tmp),
        )

        # ---- Detect ZIP vs NetCDF ----
        with open(tmp, "rb") as f:
            header = f.read(4)

        # ZIP magic
        if header.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(tmp, "r") as z:
                nc_files = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_files:
                    raise RuntimeError(
                        f"ZIP from CDS contains no .nc files! Contents: {z.namelist()}"
                    )
                nc_name = nc_files[0]
                with z.open(nc_name) as src, open(target, "wb") as out:
                    out.write(src.read())
            tmp.unlink()
            return target

        # Already NetCDF (rare)
        if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
            tmp.rename(target)
            return target

        # Otherwise: HTML/XML/error
        with open(tmp, "rb") as f:
            content = f.read(200)
        raise RuntimeError(
            "CDS returned unexpected file type for ERA5 soil moisture.\n"
            f"Header={header}\nFirst bytes={content}"
        )

    def download(self, polygon: dict, year: int, month: int) -> xr.Dataset:
        """
        Download and clip ERA5-Land soil moisture for given month and polygon.

        Returns
        -------
        xr.Dataset
            Dataset with one variable:
            - 'swvl1' : volumetric soil water content, top layer
              dims: time, lat, lon
        """
        path = self._ensure_downloaded(polygon, year, month)

        ds = xr.open_dataset(path, engine=self.engine)

        # ---- Normalize time dimension ----
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
            try:
                ds = xr.decode_cf(ds)
            except Exception:
                pass

        # variable name is usually 'swvl1'
        var = "swvl1" if "swvl1" in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]

        # ---- Spatial metadata for rioxarray ----
        lat_name = [c for c in da.coords if c.lower().startswith("lat")][0]
        lon_name = [c for c in da.coords if c.lower().startswith("lon")][0]

        da = (
            da
            .rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
            .rio.write_crs("EPSG:4326", inplace=False)
        )

        da_clip = da.rio.clip(
            [polygon],
            CRS.from_epsg(4326),
            drop=True,
            all_touched=True,
        )

        self.data = xr.Dataset({"swvl1": da_clip})

    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the clipped ERA5 soil moisture DataArray to GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        if "swvl1" in self.data:
            geotiff_path = output_dir / f"{basename}_swvl1.tif"
            self.data["swvl1"].isel(time=0).rio.to_raster(geotiff_path)
            paths.append(geotiff_path)
        return paths
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}_swvl1.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False


class ESAWorldCoverDownloader:
    """
    Download and clip ESA WorldCover 10 m land cover (2020 or 2021)
    to a GeoJSON polygon.

    Data source:
    - ESA WorldCover S3 bucket (no auth required) [COG GeoTIFFs]
      https://registry.opendata.aws/esa-worldcover/
    - Tiles: 3 x 3 degree COGs in EPSG:4326.
    - We use the 2020 grid GeoJSON to find intersecting tiles,
      then download the corresponding Map tiles for the selected year.
    """

    S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

    def __init__(
        self,
        year: int = 2021,
        cache_dir: Union[str, Path] = "worldcover_cache",
    ):
        """
        Parameters
        ----------
        year : int
            2020 (v100) or 2021 (v200) are supported.
        cache_dir : str or Path
            Directory where downloaded tiles will be cached.
        """
        if year not in (2020, 2021):
            raise ValueError("ESAWorldCover only supports year=2020 or 2021.")

        self.year = year
        self.version = "v100" if year == 2020 else "v200"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _grid_url(self) -> str:
        # Official grid GeoJSON (2020 grid used also for 2021 tiles)
        # cf. WorldCover PUM example code.
        return f"{self.S3_PREFIX}/v100/2020/esa_worldcover_2020_grid.geojson"

    def _load_grid(self) -> List[dict]:
        """
        Download and parse the WorldCover tiling grid (GeoJSON).
        Returns a list of GeoJSON features.
        """
        resp = requests.get(self._grid_url)
        resp.raise_for_status()
        grid = resp.json()
        return grid["features"]

    def _find_tiles_for_polygon(self, polygon: dict) -> List[str]:
        """
        Find all 3x3 degree tiles whose polygons intersect the input polygon.

        Parameters
        ----------
        polygon : dict
            GeoJSON geometry dict (EPSG:4326).

        Returns
        -------
        list of str
            Tile IDs (e.g. 'S48E036') to download.
        """
        aoi_geom = shape(polygon)
        features = self._load_grid()

        tiles = []
        for feat in features:
            tile_geom = shape(feat["geometry"])
            if tile_geom.intersects(aoi_geom):
                props = feat.get("properties", {})
                tile_id = props.get("ll_tile") or props.get("tile_id")
                if tile_id is None:
                    # Fallback: try 'name' or something similar
                    tile_id = props.get("name")
                if tile_id is None:
                    continue
                tiles.append(tile_id)

        if not tiles:
            raise RuntimeError("No ESA WorldCover tiles intersect the given polygon.")

        return tiles

    def _tile_url(self, tile_id: str) -> str:
        """
        Build S3 HTTPS URL for a given tile and year/version.

        Example:
        https://esa-worldcover.s3.eu-central-1.amazonaws.com/
            v200/2021/map/ESA_WorldCover_10m_2021_v200_S48E036_Map.tif
        """
        return (
            f"{self.S3_PREFIX}/"
            f"{self.version}/{self.year}/map/"
            f"ESA_WorldCover_10m_{self.year}_{self.version}_{tile_id}_Map.tif"
        )

    def _local_tile_path(self, tile_id: str) -> Path:
        return (
            self.cache_dir
            / f"ESA_WorldCover_10m_{self.year}_{self.version}_{tile_id}_Map.tif"
        )

    def _download_tile(self, tile_id: str) -> Path:
        """
        Download a single 3x3 degree COG tile if not cached.
        """
        local = self._local_tile_path(tile_id)
        if local.exists() and local.stat().st_size > 0:
            return local

        url = self._tile_url(tile_id)
        logging.info(f"Downloading ESA WorldCover {self.year} tile {tile_id} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local
    
    def download(self, polygon: dict, target_res_deg: float = 0.1) -> xr.DataArray:
        """
        Download ESA WorldCover tiles intersecting the polygon, and aggregate them
        onto a coarse lat/lon grid (e.g. 0.1°) using majority (mode) resampling.

        This keeps memory usage tiny, because the output grid has orders of
        magnitude fewer cells than the native 10 m WorldCover grid.

        Parameters
        ----------
        polygon : dict
            GeoJSON geometry dict in EPSG:4326.
        target_res_deg : float, optional
            Target resolution in degrees (e.g. 0.1 for ~10 km); used for both
            lat and lon. Default 0.1.

        Returns
        -------
        xarray.DataArray
            Land-cover classes on a coarse lat/lon grid, dims: (lat, lon),
            values are integer land-cover codes (majority class per cell).
        """
        # ---- 1. AOI bounds & coarse grid definition ----
        aoi = shape(polygon)
        minx, miny, maxx, maxy = aoi.bounds

        # Expand slightly to make sure we cover edge pixels
        pad = target_res_deg * 0.5
        minx -= pad
        maxx += pad
        miny -= pad
        maxy += pad

        # Compute coarse grid size
        width = int(np.ceil((maxx - minx) / target_res_deg))
        height = int(np.ceil((maxy - miny) / target_res_deg))

        # Destination transform (lon increasing to the right, lat decreasing downward)
        dst_transform = rasterio.transform.from_origin(
            minx,  # west
            maxy,  # north
            target_res_deg,  # xres
            target_res_deg,  # yres
        )
        dst_crs = "EPSG:4326"

        # Prepare an empty mosaic (uint8, nodata=0)
        mosaic = np.zeros((height, width), dtype=np.uint8)
        mosaic_nodata = 0

        # ---- 2. Loop over intersecting tiles and accumulate into coarse grid ----
        tiles = self._find_tiles_for_polygon(polygon)

        for tile in tiles:
            path = self._download_tile(tile)

            with rasterio.open(path) as src:
                src_crs = src.crs
                src_transform = src.transform
                nodata = src.nodata
                if nodata is None:
                    nodata = 0

                # Destination for this tile
                dst_tile = np.full((height, width), fill_value=nodata, dtype=src.dtypes[0])

                # Reproject with categorical mode resampling onto the coarse grid
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_tile,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.mode,
                )

                # Merge into mosaic: where dst_tile != nodata, overwrite
                mask = dst_tile != nodata
                mosaic[mask] = dst_tile[mask]

        # ---- 3. Build xarray DataArray over (lat, lon) ----
        # lon from minx + 0.5*res to ...
        lons = minx + (np.arange(width) + 0.5) * target_res_deg
        # lat from maxy - 0.5*res downward
        lats = maxy - (np.arange(height) + 0.5) * target_res_deg

        da = xr.DataArray(
            mosaic,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
            name="landcover",
        )

        # Restrict exactly to AOI bounds if you want a tighter crop
        da = da.sel(
            lon=slice(aoi.bounds[0], aoi.bounds[2]),
            lat=slice(aoi.bounds[3], aoi.bounds[1]),  # lat is descending
        )

        # Attach CRS for rioxarray (optional, if you want rio.* on it)
        da = (
            da
            .rio.write_crs(dst_crs, inplace=False)
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        )
        self.data = da
    
    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the Worldcover raster as a GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        self.data.rio.to_raster(geotiff_path)
        return [geotiff_path]
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False


class IrrigationMapDownloader:
    """
    Downloader/aggregator for a global irrigation map (GMIA v5).

    Uses the FAO Global Map of Irrigation Areas v5 (GMIA):
    - grid: 5 arc-min (~0.083333°), EPSG:4326
    - value: percentage of each cell equipped for irrigation (0–100)

    This class:
      * ensures the ASCII grid is available locally (unzipping if needed),
      * clips/aggregates it onto a coarse lat/lon grid (e.g. 0.1°) over the AOI,
      * returns an xarray.DataArray with dims (lat, lon), values in %.

    Notes
    -----
    - You need to download the file `gmia_v5_aei_pct_asc.zip` manually from the
      GMIA v5 distribution (FAO / Stars4Water / Aquastat) and place it in
      `cache_dir`, or pass an explicit `ascii_zip_path`.
    """

    def __init__(
        self,
        cache_dir: str | Path = "gmia_cache",
        ascii_zip_path: str | Path | None = None,
        target_res_deg: float = 0.1,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_res_deg = target_res_deg

        # Expected filenames inside the cache
        self.zip_path = (
            Path(ascii_zip_path)
            if ascii_zip_path is not None
            else self.cache_dir / "gmia_v5_aei_pct_asc.zip"
        )
        # The ASCII grid file name inside the ZIP can be changed here if needed
        self.asc_path = self.cache_dir / "gmia_v5_aei_pct.asc"

    GMIA_HA_URL = (
        "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/"
        "o/GIS%2Fgmia_v5_aei_ha_asc.zip"
        "?alt=media&token=416b27f5-fcb5-4178-ab49-1658d5c2c3ad"
    )

    def _ensure_local_asc(self) -> Path:
        """
        Make sure the ASCII grid exists locally; if only ZIP exists, unzip it.
        If neither exists, download the GMIA v5 'hectares per cell' ZIP from FAO.
        """
        # Already extracted?
        if self.asc_path.exists():
            return self.asc_path

        # ZIP missing: download it automatically
        if not self.zip_path.exists():
            logging.info(f"Downloading GMIA v5 irrigation map from FAO to {self.zip_path} ...")
            resp = requests.get(self.GMIA_HA_URL, stream=True)
            resp.raise_for_status()
            with open(self.zip_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        # Unzip the ASCII grid from the archive
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            asc_members = [m for m in zf.namelist() if m.lower().endswith(".asc")]
            if not asc_members:
                raise RuntimeError(
                    f"No .asc file found inside ZIP {self.zip_path}. "
                    "Check contents or adjust IrrigationMapDownloader._ensure_local_asc."
                )
            member = asc_members[0]
            zf.extract(member, path=self.cache_dir)
            extracted = self.cache_dir / member
            # Normalize file name / path
            if extracted != self.asc_path:
                extracted.rename(self.asc_path)

        return self.asc_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download(self, polygon: dict) -> xr.DataArray:
        """
        Clip/aggregate GMIA irrigation map to the given polygon on a coarse grid.

        Parameters
        ----------
        polygon : dict
            GeoJSON geometry dict in EPSG:4326.

        Returns
        -------
        xarray.DataArray
            Irrigation fraction (%) on a coarse lat/lon grid over the AOI,
            dims: (lat, lon), coords: lat, lon, name: 'irrigation_pct'.
        """
        asc_path = self._ensure_local_asc()

        # ---- 1. AOI bounds & coarse target grid definition ----
        aoi = shape(polygon)
        minx, miny, maxx, maxy = aoi.bounds

        pad = self.target_res_deg * 0.5
        minx -= pad
        maxx += pad
        miny -= pad
        maxy += pad

        width = int(np.ceil((maxx - minx) / self.target_res_deg))
        height = int(np.ceil((maxy - miny) / self.target_res_deg))

        dst_transform = rasterio.transform.from_origin(
            minx,  # west
            maxy,  # north
            self.target_res_deg,  # xres
            self.target_res_deg,  # yres
        )
        dst_crs = "EPSG:4326"

        # Destination array for GMIA (percentage irrigated, 0–100)
        dst = np.zeros((height, width), dtype=np.float32)

        # ---- 2. Reproject GMIA 5' grid onto coarse AOI grid ----
        with rasterio.open(asc_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            nodata = src.nodata
            if nodata is None:
                # GMIA docs: cells without irrigation are NODATA = -9
                # but we still respect src.nodata if present.
                nodata = -9.0

            dst.fill(nodata)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,  # average of % values
            )

        # Mask nodata
        dst = np.where(dst == nodata, np.nan, dst)

        # ---- 3. Wrap in xarray with (lat, lon) ----
        lons = minx + (np.arange(width) + 0.5) * self.target_res_deg
        lats = maxy - (np.arange(height) + 0.5) * self.target_res_deg

        da = xr.DataArray(
            dst,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
            name="irrigation_pct",
            attrs={
                "units": "%",
                "description": "GMIA v5 area equipped for irrigation "
                               "(% of cell area, aggregated to coarse grid)",
            },
        )

        # Tight crop to exact AOI bounds
        da = da.sel(
            lon=slice(aoi.bounds[0], aoi.bounds[2]),
            lat=slice(aoi.bounds[3], aoi.bounds[1]),  # lat descending
        )

        # Register CRS and spatial dims for rioxarray
        da = (
            da
            .rio.write_crs(dst_crs, inplace=False)
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        )
        self.data = da
    
    def save_geotiff(self, output_dir: Path, basename: str):
        """
        Save the Irrigation map raster as a GeoTIFF.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_path = output_dir / f"{basename}.tif"
        self.data.rio.to_raster(geotiff_path)
        return [geotiff_path]
    
    def check_geotiff_exists_and_validate(self, output_dir: Path, basename: str) -> bool:
        """
        Check if the expected GeoTIFF exists and is valid (not corrupt).
        Returns True if valid, False otherwise.
        """
        geotiff_path = output_dir / f"{basename}.tif"
        if not geotiff_path.exists():
            return False
        try:
            with rasterio.open(geotiff_path) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False


class ESACCILandCoverDownloader:
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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_geometry(polygon: Dict[str, Any]) -> None:
        if not isinstance(polygon, dict) or polygon.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                "polygon must be a GeoJSON *geometry* dict (Polygon/MultiPolygon) in EPSG:4326 "
                "(same convention as your other downloaders)."
            )

    def _remote_url(self, year: int) -> str:
        # Example from CEDA listing:
        # ESACCI-LC-L4-LCCS-Map-300m-P1Y-2010-v2.0.7.tif  :contentReference[oaicite:2]{index=2}
        return f"{self.BASE_DIR}/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7.tif"

    def _local_path(self, year: int) -> Path:
        return self.cache_dir / f"ESACCI_LC_{year}_v2.0.7.tif"

    def _ensure_downloaded(self, year: int) -> Path:
        local = self._local_path(year)
        if local.exists() and local.stat().st_size > 0:
            return local

        url = self._remote_url(year)
        logging.info(f"Downloading ESA CCI LC {year} GeoTIFF from {url}")

        with requests.get(url, stream=True, timeout=(20, 600)) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return local

    def download(self, polygon: dict, year: int) -> xr.DataArray:
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

        self.data = da
        return da

    def save_geotiff(self, output_dir: Union[str, Path], basename: str):
        if not hasattr(self, "data"):
            raise RuntimeError("No data found. Call download() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{basename}.tif"
        self.data.rio.to_raster(out)
        return [out]

    def check_geotiff_exists_and_validate(self, output_dir: Union[str, Path], basename: str) -> bool:
        out = Path(output_dir) / f"{basename}.tif"
        if not out.exists():
            return False
        try:
            with rasterio.open(out) as src:
                _ = src.read(1, window=((0, 1), (0, 1)))
            return True
        except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
            return False



