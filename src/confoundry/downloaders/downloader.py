from __future__ import annotations


import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass

import rasterio


@dataclass
class ItemDownloadReport:
    data_source: str
    variable_name: str
    acquisition_time: datetime.datetime
    path: Path
    download_successful: bool
    error: Optional[str] = None
    metadata: Optional[dict] = None


class BaseDownloader(ABC):
    def __init__(self):
        """
        Base class for all data downloaders.
        """
        pass

    @abstractmethod
    def download(self, 
                 polygon: dict, 
                 time_frame: tuple[datetime.datetime, datetime.datetime],
                 output_dir: Path,
                 show_progress: bool = True,
                 **kwargs,
                 ) -> list[ItemDownloadReport]:
        """
        Download all relevant files within the specified time frame for the given polygon.
        Returns a list of all of the attempted downloads with their download report.
        """
        pass

    @property
    @abstractmethod
    def frequency(self) -> str:
        """Returns the frequency of the data (e.g. 'daily', 'monthly', 'hourly')."""
        pass

    @abstractmethod
    def _save_geotiff(self, data, output_dir: Path, basename: str) -> dict[str, Path]:
        """Saves the result and returns a dictionary of variable names and their created file paths."""
        pass

    def _get_filepaths(self, output_dir: Path, basename: str) -> List[Path]:
        """Returns expected file paths for the given output directory and basename."""
        geotiff_path = output_dir / f"{basename}.tif"
        return [geotiff_path]

    def _validate_geotiff(self, output_dir: Path, basename: str) -> dict:
        """
        Returns a dict mapping expected filepaths to True/False (valid/corrupt/missing).
        Example:
            { 'output_dir/ERA5_xxx_t2m.tif': True, 'output_dir/ERA5_xxx_ssrd.tif': False }
        """
        is_valid_dict = {}
        for geotiff_path in self._get_filepaths(output_dir, basename):
            # Default setting to False
            is_valid_dict[geotiff_path] = False

            # Simple check if geotiff exists and is readable
            if geotiff_path.exists():
                try:
                    with rasterio.open(geotiff_path) as src:
                        _ = src.read(1, window=((0, 1), (0, 1)))
                    is_valid_dict[geotiff_path] = True
                except (FileNotFoundError, rasterio.errors.RasterioIOError, OSError, ValueError, PermissionError):
                    continue
        return is_valid_dict