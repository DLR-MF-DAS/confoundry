

# class IrrigationMapDownloader(BaseDownloader):
#     """
#     Downloader/aggregator for a global irrigation map (GMIA v5).

#     Uses the FAO Global Map of Irrigation Areas v5 (GMIA):
#     - grid: 5 arc-min (~0.083333°), EPSG:4326
#     - value: percentage of each cell equipped for irrigation (0–100)

#     This class:
#       * ensures the ASCII grid is available locally (unzipping if needed),
#       * clips/aggregates it onto a coarse lat/lon grid (e.g. 0.1°) over the AOI,
#       * returns an xarray.DataArray with dims (lat, lon), values in %.

#     Notes
#     -----
#     - You need to download the file `gmia_v5_aei_pct_asc.zip` manually from the
#       GMIA v5 distribution (FAO / Stars4Water / Aquastat) and place it in
#       `cache_dir`, or pass an explicit `ascii_zip_path`.
#     """

#     def __init__(
#         self,
#         cache_dir: str | Path = "gmia_cache",
#         ascii_zip_path: str | Path | None = None,
#     ):
#         super().__init__(cache_dir)

#         # Expected filenames inside the cache
#         self.zip_path = (
#             Path(ascii_zip_path)
#             if ascii_zip_path is not None
#             else self.cache_dir / "gmia_v5_aei_pct_asc.zip"
#         )
#         # The ASCII grid file name inside the ZIP can be changed here if needed
#         self.asc_path = self.cache_dir / "gmia_v5_aei_pct.asc"

#     GMIA_HA_URL = (
#         "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/"
#         "o/GIS%2Fgmia_v5_aei_ha_asc.zip"
#         "?alt=media&token=416b27f5-fcb5-4178-ab49-1658d5c2c3ad"
#     )

#     def _ensure_local_asc(self) -> Path:
#         """
#         Make sure the ASCII grid exists locally; if only ZIP exists, unzip it.
#         If neither exists, download the GMIA v5 'hectares per cell' ZIP from FAO.
#         """
#         # Already extracted?
#         if self.asc_path.exists():
#             return self.asc_path

#         # ZIP missing: download it automatically
#         if not self.zip_path.exists():
#             logging.info(f"Downloading GMIA v5 irrigation map from FAO to {self.zip_path} ...")
#             resp = requests.get(self.GMIA_HA_URL, stream=True)
#             resp.raise_for_status()
#             with open(self.zip_path, "wb") as f:
#                 for chunk in resp.iter_content(8192):
#                     if chunk:
#                         f.write(chunk)

#         # Unzip the ASCII grid from the archive
#         with zipfile.ZipFile(self.zip_path, "r") as zf:
#             asc_members = [m for m in zf.namelist() if m.lower().endswith(".asc")]
#             if not asc_members:
#                 raise RuntimeError(
#                     f"No .asc file found inside ZIP {self.zip_path}. "
#                     "Check contents or adjust IrrigationMapDownloader._ensure_local_asc."
#                 )
#             member = asc_members[0]
#             zf.extract(member, path=self.cache_dir)
#             extracted = self.cache_dir / member
#             # Normalize file name / path
#             if extracted != self.asc_path:
#                 extracted.rename(self.asc_path)

#         return self.asc_path

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------
#     def download(self, polygon: dict, target_res_deg: float = 0.1) -> xr.DataArray:
#         """
#         Clip/aggregate GMIA irrigation map to the given polygon on a coarse grid.

#         Parameters
#         ----------
#         polygon : dict
#             GeoJSON geometry dict in EPSG:4326.

#         Returns
#         -------
#         xarray.DataArray
#             Irrigation fraction (%) on a coarse lat/lon grid over the AOI,
#             dims: (lat, lon), coords: lat, lon, name: 'irrigation_pct'.
#         """
#         asc_path = self._ensure_local_asc()

#         # ---- 1. AOI bounds & coarse target grid definition ----
#         aoi = shape(polygon)
#         minx, miny, maxx, maxy = aoi.bounds

#         pad = target_res_deg * 0.5
#         minx -= pad
#         maxx += pad
#         miny -= pad
#         maxy += pad

#         width = int(np.ceil((maxx - minx) / target_res_deg))
#         height = int(np.ceil((maxy - miny) / target_res_deg))

#         dst_transform = rasterio.transform.from_origin(
#             minx,  # west
#             maxy,  # north
#             target_res_deg,  # xres
#             target_res_deg,  # yres
#         )
#         dst_crs = "EPSG:4326"

#         # Destination array for GMIA (percentage irrigated, 0–100)
#         dst = np.zeros((height, width), dtype=np.float32)

#         # ---- 2. Reproject GMIA 5' grid onto coarse AOI grid ----
#         with rasterio.open(asc_path) as src:
#             src_crs = src.crs
#             src_transform = src.transform
#             nodata = src.nodata
#             if nodata is None:
#                 # GMIA docs: cells without irrigation are NODATA = -9
#                 # but we still respect src.nodata if present.
#                 nodata = -9.0

#             dst.fill(nodata)

#             reproject(
#                 source=rasterio.band(src, 1),
#                 destination=dst,
#                 src_transform=src_transform,
#                 src_crs=src_crs,
#                 dst_transform=dst_transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.average,  # average of % values
#             )

#         # Mask nodata
#         dst = np.where(dst == nodata, np.nan, dst)

#         # ---- 3. Wrap in xarray with (lat, lon) ----
#         lons = minx + (np.arange(width) + 0.5) * target_res_deg
#         lats = maxy - (np.arange(height) + 0.5) * target_res_deg

#         da = xr.DataArray(
#             dst,
#             dims=("lat", "lon"),
#             coords={"lat": lats, "lon": lons},
#             name="irrigation_pct",
#             attrs={
#                 "units": "%",
#                 "description": "GMIA v5 area equipped for irrigation "
#                                "(% of cell area, aggregated to coarse grid)",
#             },
#         )

#         # Tight crop to exact AOI bounds
#         da = da.sel(
#             lon=slice(aoi.bounds[0], aoi.bounds[2]),
#             lat=slice(aoi.bounds[3], aoi.bounds[1]),  # lat descending
#         )

#         # Register CRS and spatial dims for rioxarray
#         da = (
#             da
#             .rio.write_crs(dst_crs, inplace=False)
#             .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
#         )
#         self.data = da
#         return self.data
    
#     def save_geotiff(self, output_dir: Path, basename: str) -> list[Path]:
#         """
#         Save the clipped irrigation map DataArray to GeoTIFF.
#         Returns a list with a single Path.
#         """
#         if not hasattr(self, "data"):
#             raise RuntimeError("No data to save. Run download() first.")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         geotiff_path = output_dir / f"{basename}.tif"
#         self.data.rio.to_raster(geotiff_path)
#         return [geotiff_path]