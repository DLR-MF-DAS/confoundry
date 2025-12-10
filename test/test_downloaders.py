import json
from drought_causality.downloaders import (
    SPEIDownloader,
    MODISNDVIDownloader,
    ERA5Downloader,
    ERA5PrecipDownloader,
    ERA5SoilMoistureDownloader,
    ESAWorldCoverDownloader,
    IrrigationMapDownloader,
)

def test_spei_downloader():
    with open('data/california.json', 'r') as fd:
        geojson = json.load(fd)
    polygon = geojson['features'][0]['geometry']
    downloader = SPEIDownloader()
    spei_da = downloader.download(polygon, year=2021, month=7)
    spei_da.rio.to_raster("spei01_clipped_aoi_2021-07.tif")

def test_modis_ndvi_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    downloader = MODISNDVIDownloader()
    ndvi_da = downloader.download(polygon, year=2021, month=7)
    ndvi_da.isel(time=0).rio.to_raster("ndvi_2021_07_california.tif")

def test_era5_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    era5 = ERA5Downloader()
    ds_era5 = era5.download(polygon, year=2021, month=7)

    ds_era5["t2m"].isel(time=0).rio.to_raster("era5_t2m_2021_07_california.tif")
    ds_era5["ssrd"].isel(time=0).rio.to_raster("era5_ssrd_2021_07_california.tif")

def test_era5precip_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    era5 = ERA5PrecipDownloader()
    ds_era5 = era5.download(polygon, year=2021, month=7)

    ds_era5["tp"].isel(time=0).rio.to_raster("era5_precip_2021_07_california.tif")

def test_era5_soil_moisture_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    sm = ERA5SoilMoistureDownloader()
    ds_sm = sm.download(polygon, year=2021, month=7)

    assert "swvl1" in ds_sm.data_vars
    ds_sm["swvl1"].isel(time=0).rio.to_raster("era5_swvl1_2021_07_california.tif")

def test_esa_world_cover_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    wc = ESAWorldCoverDownloader(year=2021)
    da_lc = wc.download(polygon, target_res_deg=0.1)

    # Write a GeoTIFF
    da_lc.rio.to_raster("worldcover_2021_california_0p1deg.tif")

def test_irrigation_map_downloader():
    with open("data/california.json") as fd:
        geojson = json.load(fd)
    polygon = geojson["features"][0]["geometry"]

    irr = IrrigationMapDownloader(target_res_deg=0.1)
    da_irr = irr.download(polygon)

    # e.g. write to GeoTIFF
    da_irr.rio.to_raster("gmia_irrigation_0p1deg_california.tif")
