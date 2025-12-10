import os
import json
from pathlib import Path

from downloaders import (ERA5Downloader, 
                         ERA5PrecipDownloader, 
                         ERA5SoilMoistureDownloader,
                         ESAWorldCoverDownloader, 
                         IrrigationMapDownloader, 
                         MODISNDVIDownloader,
                         SPEIDownloader)


def download_timeseries_data(
        polygon: dict,
        location_nickname: str,
        start_year: int,
        start_month: int,
        final_year: int,
        final_month: int,
        world_cover_year: int = 2021,
        target_res_deg: float = 0.1,
        ):  

    # Assert that final date is not before start date
    assert (start_year < final_year) or (start_year == final_year and start_month <= final_month)  
    assert world_cover_year in [2020, 2021], "World Cover year must be 2020 or 2021."

    # Create a cache directory for temporary files
    cache_dir = Path(os.getcwd()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialise downloaders
    wc_downloader = ESAWorldCoverDownloader(year=world_cover_year, cache_dir=cache_dir)
    spei_downloader = SPEIDownloader(cache_dir=cache_dir)
    era5_downloader = ERA5Downloader(cache_dir=cache_dir)
    era5p_downloader = ERA5PrecipDownloader(cache_dir=cache_dir)
    sm_downloader = ERA5SoilMoistureDownloader(cache_dir=cache_dir)
    irr_downloader = IrrigationMapDownloader(target_res_deg=target_res_deg, cache_dir=cache_dir)
    ndvi_downloader = MODISNDVIDownloader(cache_dir=cache_dir)

    # World Cover
    outdir_wc = Path(os.getcwd()) / f"data/{location_nickname}/ESA_WorldCover/{world_cover_year}"
    outdir_wc.mkdir(parents=True, exist_ok=True)
    da_lc = wc_downloader.download(polygon=polygon, target_res_deg=target_res_deg)
    da_lc.rio.to_raster(str(outdir_wc / f"worldcover_{location_nickname}_{world_cover_year}_{target_res_deg}deg.tif"))

    # Loop through each year and month in the specified range
    for year in range(start_year, final_year + 1):
        for month in range(1, 13):
            # Skip months outside the specified range
            if (year == start_year and month < start_month) or (year == final_year and month > final_month):
                continue

            print(f"Downloading data for {year}-{month:02d}...")
            outdir = Path(os.getcwd()) / f"data/{location_nickname}/{year}/{month}"
            outdir.mkdir(parents=True, exist_ok=True)

            # SPEI
            spei_da = spei_downloader.download(polygon=polygon, year=year, month=month)
            spei_da.rio.to_raster(str(outdir / f"spei_{location_nickname}_{year}_{month:02d}.tif"))

            # ERA5
            ds_era5 = era5_downloader.download(polygon=polygon, year=year, month=month)
            if "t2m" in ds_era5:
                ds_era5["t2m"].isel(time=0).rio.to_raster(str(outdir / f"era5_t2m_{location_nickname}_{year}_{month:02d}.tif"))
            if "ssrd" in ds_era5:
                ds_era5["ssrd"].isel(time=0).rio.to_raster(str(outdir / f"era5_ssrd_{location_nickname}_{year}_{month:02d}.tif"))

            # ERA5 Precip
            ds_era5p = era5p_downloader.download(polygon=polygon, year=year, month=month)
            if "tp" in ds_era5p:
                ds_era5p["tp"].isel(time=0).rio.to_raster(str(outdir / f"era5_precip_{location_nickname}_{year}_{month:02d}.tif"))

            # ERA5 Soil Moisture
            ds_sm = sm_downloader.download(polygon=polygon, year=year, month=month)
            if "swvl1" in ds_sm:
                ds_sm["swvl1"].isel(time=0).rio.to_raster(str(outdir / f"era5_swvl1_{location_nickname}_{year}_{month:02d}.tif"))

            # Irrigation Map
            da_irr = irr_downloader.download(polygon=polygon)
            da_irr.rio.to_raster(str(outdir / f"gmia_irrigation_{location_nickname}_{target_res_deg}deg.tif"))

            # MODIS NDVI
            ndvi_da = ndvi_downloader.download(polygon=polygon, year=year, month=month)
            ndvi_da.isel(time=0).rio.to_raster(str(outdir / f"ndvi_{location_nickname}_{year}_{month:02d}.tif"))
    

def main ():
    # Path to the JSON file
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'california.json')
    with open(json_path, 'r') as f:
        california_geojson = json.load(f)
    print('Loaded california.json:', california_geojson)
    
    download_timeseries_data(
        polygon=california_geojson['features'][0]['geometry'],
        location_nickname="california",
        start_year=2018,
        start_month=4,
        final_year=2018,
        final_month=5
    )

if __name__ == "__main__":
    main()
    