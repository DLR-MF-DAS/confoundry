

# class ESAWorldCoverDownloader(BaseDownloader):
#     """
#     Download and clip ESA WorldCover 10 m land cover (2020 or 2021)
#     to a GeoJSON polygon.

#     Data source:
#     - ESA WorldCover S3 bucket (no auth required) [COG GeoTIFFs]
#       https://registry.opendata.aws/esa-worldcover/
#     - Tiles: 3 x 3 degree COGs in EPSG:4326.
#     - We use the 2020 grid GeoJSON to find intersecting tiles,
#       then download the corresponding Map tiles for the selected year.
#     """

#     S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

#     def __init__(
#         self,
#         year: int = 2021,
#         cache_dir: Union[str, Path] = "worldcover_cache",
#     ):
#         """
#         Parameters
#         ----------
#         year : int
#             2020 (v100) or 2021 (v200) are supported.
#         cache_dir : str or Path
#             Directory where downloaded tiles will be cached.
#         """
#         self.year = None
#         self.version = None
#         super().__init__(cache_dir)

#     @property
#     def _grid_url(self) -> str:
#         # Official grid GeoJSON (2020 grid used also for 2021 tiles)
#         # cf. WorldCover PUM example code.
#         return f"{self.S3_PREFIX}/v100/2020/esa_worldcover_2020_grid.geojson"

#     def _load_grid(self) -> List[dict]:
#         """
#         Download and parse the WorldCover tiling grid (GeoJSON).
#         Returns a list of GeoJSON features.
#         """
#         resp = requests.get(self._grid_url)
#         resp.raise_for_status()
#         grid = resp.json()
#         return grid["features"]

#     def _find_tiles_for_polygon(self, polygon: dict) -> List[str]:
#         """
#         Find all 3x3 degree tiles whose polygons intersect the input polygon.

#         Parameters
#         ----------
#         polygon : dict
#             GeoJSON geometry dict (EPSG:4326).

#         Returns
#         -------
#         list of str
#             Tile IDs (e.g. 'S48E036') to download.
#         """
#         aoi_geom = shape(polygon)
#         features = self._load_grid()

#         tiles = []
#         for feat in features:
#             tile_geom = shape(feat["geometry"])
#             if tile_geom.intersects(aoi_geom):
#                 props = feat.get("properties", {})
#                 tile_id = props.get("ll_tile") or props.get("tile_id")
#                 if tile_id is None:
#                     # Fallback: try 'name' or something similar
#                     tile_id = props.get("name")
#                 if tile_id is None:
#                     continue
#                 tiles.append(tile_id)

#         if not tiles:
#             raise RuntimeError("No ESA WorldCover tiles intersect the given polygon.")

#         return tiles

#     def _tile_url(self, tile_id: str) -> str:
#         """
#         Build S3 HTTPS URL for a given tile and year/version.

#         Example:
#         https://esa-worldcover.s3.eu-central-1.amazonaws.com/
#             v200/2021/map/ESA_WorldCover_10m_2021_v200_S48E036_Map.tif
#         """
#         return (
#             f"{self.S3_PREFIX}/"
#             f"{self.version}/{self.year}/map/"
#             f"ESA_WorldCover_10m_{self.year}_{self.version}_{tile_id}_Map.tif"
#         )

#     def _local_tile_path(self, tile_id: str) -> Path:
#         return (
#             self.cache_dir
#             / f"ESA_WorldCover_10m_{self.year}_{self.version}_{tile_id}_Map.tif"
#         )

#     def _download_tile(self, tile_id: str) -> Path:
#         """
#         Download a single 3x3 degree COG tile if not cached.
#         """
#         local = self._local_tile_path(tile_id)
#         if local.exists() and local.stat().st_size > 0:
#             return local

#         url = self._tile_url(tile_id)
#         logging.info(f"Downloading ESA WorldCover {self.year} tile {tile_id} ...")
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status()
#             with open(local, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
#         return local
    
#     def download(self, polygon: dict, year: int = 2021, target_res_deg: float = 0.1) -> xr.DataArray:
#         """
#         Download ESA WorldCover tiles intersecting the polygon, and aggregate them
#         onto a coarse lat/lon grid (e.g. 0.1°) using majority (mode) resampling.

#         This keeps memory usage tiny, because the output grid has orders of
#         magnitude fewer cells than the native 10 m WorldCover grid.

#         Parameters
#         ----------
#         polygon : dict
#             GeoJSON geometry dict in EPSG:4326.
#         target_res_deg : float, optional
#             Target resolution in degrees (e.g. 0.1 for ~10 km); used for both
#             lat and lon. Default 0.1.

#         Returns
#         -------
#         xarray.DataArray
#             Land-cover classes on a coarse lat/lon grid, dims: (lat, lon),
#             values are integer land-cover codes (majority class per cell).
#         """
#         if year not in (2020, 2021):
#             raise ValueError("ESAWorldCover only supports year=2020 or year=2021.")
#         self.year = year
#         self.version = "v100" if year == 2020 else "v200"

#         # ---- 1. AOI bounds & coarse grid definition ----
#         aoi = shape(polygon)
#         minx, miny, maxx, maxy = aoi.bounds

#         # Expand slightly to make sure we cover edge pixels
#         pad = target_res_deg * 0.5
#         minx -= pad
#         maxx += pad
#         miny -= pad
#         maxy += pad

#         # Compute coarse grid size
#         width = int(np.ceil((maxx - minx) / target_res_deg))
#         height = int(np.ceil((maxy - miny) / target_res_deg))

#         # Destination transform (lon increasing to the right, lat decreasing downward)
#         dst_transform = rasterio.transform.from_origin(
#             minx,  # west
#             maxy,  # north
#             target_res_deg,  # xres
#             target_res_deg,  # yres
#         )
#         dst_crs = "EPSG:4326"

#         # Prepare an empty mosaic (uint8, nodata=0)
#         mosaic = np.zeros((height, width), dtype=np.uint8)
#         mosaic_nodata = 0

#         # ---- 2. Loop over intersecting tiles and accumulate into coarse grid ----
#         tiles = self._find_tiles_for_polygon(polygon)

#         for tile in tiles:
#             path = self._download_tile(tile)

#             with rasterio.open(path) as src:
#                 src_crs = src.crs
#                 src_transform = src.transform
#                 nodata = src.nodata
#                 if nodata is None:
#                     nodata = 0

#                 # Destination for this tile
#                 dst_tile = np.full((height, width), fill_value=nodata, dtype=src.dtypes[0])

#                 # Reproject with categorical mode resampling onto the coarse grid
#                 reproject(
#                     source=rasterio.band(src, 1),
#                     destination=dst_tile,
#                     src_transform=src_transform,
#                     src_crs=src_crs,
#                     dst_transform=dst_transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.mode,
#                 )

#                 # Merge into mosaic: where dst_tile != nodata, overwrite
#                 mask = dst_tile != nodata
#                 mosaic[mask] = dst_tile[mask]

#         # ---- 3. Build xarray DataArray over (lat, lon) ----
#         # lon from minx + 0.5*res to ...
#         lons = minx + (np.arange(width) + 0.5) * target_res_deg
#         # lat from maxy - 0.5*res downward
#         lats = maxy - (np.arange(height) + 0.5) * target_res_deg

#         da = xr.DataArray(
#             mosaic,
#             dims=("lat", "lon"),
#             coords={"lat": lats, "lon": lons},
#             name="landcover",
#         )

#         # Restrict exactly to AOI bounds if you want a tighter crop
#         da = da.sel(
#             lon=slice(aoi.bounds[0], aoi.bounds[2]),
#             lat=slice(aoi.bounds[3], aoi.bounds[1]),  # lat is descending
#         )

#         # Attach CRS for rioxarray (optional, if you want rio.* on it)
#         da = (
#             da
#             .rio.write_crs(dst_crs, inplace=False)
#             .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
#         )
#         self.data = da
#         return self.data
    
#     def save_geotiff(self, output_dir: Path, basename: str) -> list[Path]:
#         """
#         Save the clipped ESA WorldCover DataArray to GeoTIFF.
#         Returns a list with a single Path.
#         """
#         if not hasattr(self, "data"):
#             raise RuntimeError("No data to save. Run download() first.")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         geotiff_path = output_dir / f"{basename}.tif"
#         self.data.rio.to_raster(geotiff_path)
#         return [geotiff_path]