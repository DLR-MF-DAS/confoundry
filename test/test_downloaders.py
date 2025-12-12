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


TEST_CACHE_DIR = Path(os.getcwd()) / "test/cache"


def test_spei_downloader():
    with open('data/california.json', 'r') as fd:
        geojson = json.load(fd)
    polygon = geojson['features'][0]['geometry']
    downloader = SPEIDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, year=2021, month=7)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_modis_ndvi_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = MODISNDVIDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, year=2021, month=7)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_era5_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = ERA5Downloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, year=2021, month=7)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_era5precip_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = ERA5PrecipDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, year=2021, month=7)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_era5_soil_moisture_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = ERA5SoilMoistureDownloader(cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, year=2021, month=7)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_esa_world_cover_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = ESAWorldCoverDownloader(year=2021, cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon, target_res_deg=0.1)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")

def test_irrigation_map_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = IrrigationMapDownloader(target_res_deg=0.1, cache_dir=TEST_CACHE_DIR)
    downloader.download(polygon)
    downloader.save_geotiff(output_dir=Path("data/california_test/2021/7"), basename="california_2021_07")
