import logging
import requests
import datetime
from tqdm import tqdm
from typing import List
from pathlib import Path
import concurrent.futures
from pystac_client import Client

import rasterio
from rasterio.enums import Resampling

from drought_causality.downloaders.downloader import BaseDownloader, ItemDownloadReport


class S2AWSDownloader(BaseDownloader):
    def __init__(self): 
        super().__init__()
        self.stac_url = "https://earth-search.aws.element84.com/v1"
        self.catalog = Client.open(self.stac_url)

        # Maps standard Sentinel-2 band names to Element84 STAC asset keys
        self.S2_BAND_MAP = {
            'B1': 'coastal',    'B2': 'blue',       'B3': 'green',
            'B4': 'red',        'B5': 'rededge1',   'B6': 'rededge2',
            'B7': 'rededge3',   'B8': 'nir',        'B8A': 'nir08',
            'B9': 'nir09',      'B11': 'swir16',    'B12': 'swir22',
            'AOT': 'aot',       'WVP': 'wvp',       'SCL': 'scl',
            'VISUAL': 'visual'
        }

    @property
    def frequency(self) -> str:
        return "daily"

    def download(
        self,
        polygon: dict,
        time_frame: tuple[datetime.datetime, datetime.datetime],
        output_dir: Path,
        show_progress: bool = True,
        data_type: str = "l2a",
        bands: List[str] = ['visual'], 
        **kwargs,
    ) -> List[ItemDownloadReport]:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract datetime (STAC expects YYYY-MM-DD/YYYY-MM-DD)
        start_dt, end_dt = time_frame
        time_range = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

        # Map GEE collection name to STAC collection name
        collection = f"sentinel-2-{data_type}"

        # Search the Catalog for all images within the polygon and time frame.
        search = self.catalog.search(
            collections=[collection],
            intersects=polygon,
            datetime=time_range,
            # Optional: Add a cloud cover filter here if you want to skip completely cloudy tiles
            # query={"eo:cloud_cover": {"lt": 100}}
        )
        
        items = list(search.items())
        if items:
            logging.info(f"Found {len(items)} images on AWS for the specified parameters.")
        else:
            logging.warning("No images found for the specified parameters.")
            return []        
        
        # 3. Multithreaded Download Execution
        # We use a ThreadPoolExecutor to download multiple images at once.
        reports = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all items to the thread pool
            images4dl = {
                executor.submit(self._download_single_item, item, output_dir, collection, bands): item 
                for item in items
            }
            
            # Wrap the 'as_completed' iterator in tqdm for the progress bar
            iterator = tqdm(
                concurrent.futures.as_completed(images4dl), 
                total=len(images4dl), 
                desc="S2_AWS", 
                unit="image", 
                disable=not show_progress
            )
            
            # Run download
            for image4dl in iterator:
                reports.append(image4dl.result())

        return reports

    def _download_single_item(self, item, output_dir: Path, collection: str, bands: List[str]) -> ItemDownloadReport:
        acq_time_str = item.properties["datetime"].replace('Z', '+00:00')
        acqusition_dt = datetime.datetime.fromisoformat(acq_time_str)
        metadata = item.properties
        
        # Get MGRS tile from metadata to construct a meaningful filename
        mgrs_tile = metadata.get("s2:mgrs_tile")
        if not mgrs_tile:
            grid_code = metadata.get("grid:code", "UNKNOWN_GRID")
            # If it's the AWS format 'MGRS-31TFM', strip the prefix
            mgrs_tile = grid_code.replace("MGRS-", "") if grid_code.startswith("MGRS-") else grid_code
        basename = f"S2_{mgrs_tile}_{acqusition_dt.strftime('%Y%m%dT%H%M%S')}"
        out_path = output_dir / f"{basename}.tif"
        
        temp_files = []
        try:
            # 1. Download each band as a temporary file
            for band in bands:
                # Resolve GEE name to STAC name (e.g. 'B4' -> 'red')
                mapped_key = self.S2_BAND_MAP.get(band.upper(), band.lower())
                
                if mapped_key not in item.assets:
                    raise ValueError(f"Band {band} (mapped to '{mapped_key}') not found in STAC assets.")
                
                url = item.assets[mapped_key].href
                temp_path = output_dir / f"{basename}_temp_{mapped_key}.tif"
                
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                temp_files.append(temp_path)
            
            # 2. Stack the temporary files into a single multi-band GeoTIFF
            # Use the first requested band to define the shape and metadata footprint
            with rasterio.open(temp_files[0]) as src0:
                profile = src0.profile.copy()
                profile.update(count=len(bands))
                master_shape = (src0.height, src0.width)
                
            with rasterio.open(out_path, 'w', **profile) as dst:
                for idx, temp_path in enumerate(temp_files):
                    with rasterio.open(temp_path) as src:
                        # Auto-resample if a band has a different native resolution (e.g. 20m vs 10m)
                        if (src.height, src.width) != master_shape:
                            arr = src.read(
                                1, 
                                out_shape=master_shape,
                                resampling=Resampling.bilinear
                            )
                        else:
                            arr = src.read(1)
                        
                        dst.write(arr, idx + 1)
                        dst.set_band_description(idx + 1, bands[idx])
            
            # 3. Clean up the temporary single-band files
            for temp_path in temp_files:
                temp_path.unlink(missing_ok=True)
                    
            return ItemDownloadReport(
                data_source="S2",
                variable_name=collection,
                acquisition_time=acqusition_dt,
                polygon=item.geometry,
                bbox=item.bbox,
                path=out_path,
                download_successful=True,
                metadata=metadata
            )
            
        except Exception as e:
            logging.error(f"Failed to download image {item.id}: {e}")
            
            # Clean up temp files even if it fails partway through
            for temp_path in temp_files:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                    
            return ItemDownloadReport(
                data_source="S2",
                variable_name=collection,
                acquisition_time=acqusition_dt,
                polygon=item.geometry,
                bbox=item.bbox,
                path=out_path, 
                download_successful=False,
                error=str(e),
                metadata=metadata
            )
        
    def _save_geotiff(self, data, output_dir: Path, basename: str) -> dict[str, Path]:
        return {"s2": output_dir / f"{basename}.tif"}