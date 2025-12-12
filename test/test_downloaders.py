import os
import json
from pathlib import Path
from drought_causality.downloaders import (
    SPEIDownloader,
    MODISNDVIDownloader,
    ERA5Downloader,
    ERA5PrecipDownloader,
    ERA5SoilMoistureDownloader,
    ESAWorldCoverDownloader,
    IrrigationMapDownloader,
)


DOWNLOAD_YEAR = 2021
DOWNLOAD_MONTH = 7
TEST_GEOJSON_PATH = 'data/california.json'
TEST_CACHE_DIR = Path(os.getcwd()) / "test/cache"
TEST_DOWNLOAD_DIR = Path(os.getcwd()) / "data/california_test"

with open(TEST_GEOJSON_PATH, 'r') as fd:
        geojson = json.load(fd)
POLYGON = geojson['features'][0]['geometry']


def test_spei_downloader():
    
    downloader = SPEIDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        year=DOWNLOAD_YEAR, 
        month=DOWNLOAD_MONTH
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"), 
        basename=f"spei_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )


def test_modis_ndvi_downloader():
    downloader = MODISNDVIDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        year=DOWNLOAD_YEAR, 
        month=DOWNLOAD_MONTH
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"), 
        basename=f"modis_ndvi_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )


def test_era5_downloader():
    downloader = ERA5Downloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        year=DOWNLOAD_YEAR, 
        month=DOWNLOAD_MONTH
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"),
        basename=f"era5_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )


def test_era5precip_downloader():
    downloader = ERA5PrecipDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        year=DOWNLOAD_YEAR, 
        month=DOWNLOAD_MONTH
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"), 
        basename=f"era5_precip_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )


def test_era5_soil_moisture_downloader():
    downloader = ERA5SoilMoistureDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        year=DOWNLOAD_YEAR, 
        month=DOWNLOAD_MONTH
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"), 
        basename=f"era5_soil_moisture_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )


def test_esa_world_cover_downloader():
    downloader = ESAWorldCoverDownloader(year=2021, cache_dir=TEST_CACHE_DIR)
    downloader.download(
        polygon=POLYGON, 
        target_res_deg=0.1
        )
    downloader.save_geotiff(
        output_dir=Path(f"{TEST_DOWNLOAD_DIR}/{DOWNLOAD_YEAR}/{DOWNLOAD_MONTH}"), 
        basename=f"esa_world_cover_test_{DOWNLOAD_YEAR}_{DOWNLOAD_MONTH:02d}"
        )

def test_irrigation_map_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = IrrigationMapDownloader(target_res_deg=0.1, cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")
