import os
import json
import click
import logging
from pathlib import Path

from downloaders import (ERA5Downloader, 
                         ERA5PrecipDownloader, 
                         ERA5SoilMoistureDownloader,
                         ESAWorldCoverDownloader, 
                         IrrigationMapDownloader, 
                         MODISNDVIDownloader,
                         SPEIDownloader)


def download_timeseries_data(
        location_geojson: dict,
        location_nickname: str ,
        start_year: int = 2009,
        start_month: int = 1,
        final_year: int = 2019,
        final_month: int = 12,
        world_cover_year: int = 2021,
        target_res_deg: float = 0.1,
        ):  
    """
    Download geospatial time series datasets for a given location and time range.

    Parameters
    ----------
    location_geojson : dict
        GeoJSON dictionary defining the location polygon.
    location_nickname : str
        Custom name for location, used for output directory and filenames.
    start_year : int, optional
        First year of data to download (default is 2009).
    start_month : int, optional
        First month of data to download (default is 1).
    final_year : int, optional
        Final year of data to download (default is 2019).
    final_month : int, optional
        Final month of data to download (default is 12).
    world_cover_year : int, optional
        Year of ESA World Cover data to download (must be 2020 or 2021, default is 2021).
    target_res_deg : float, optional
        Target resolution in degrees for World Cover and Irrigation Map data (default is 0.1).

    Returns
    -------
    None
        All output files are written to disk in the appropriate directories.

    Notes
    -----
    - Downloads ESA World Cover data once for the specified year.
    - Downloads time series data for each month in the specified range.
    - Output files are saved as GeoTIFFs in organized directories.
    """

    # Assert that final date is not before start date
    assert (start_year < final_year) or (start_year == final_year and start_month <= final_month)  
    
    # World cover only available for 2020 and 2021
    assert world_cover_year in [2020, 2021], "World Cover year must be 2020 or 2021."

    # Get location polygon and nickname
    polygon = location_geojson['features'][0]['geometry']

    # Create a cache directory for temporary files
    cache_dir = Path(os.getcwd()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download ESA World Cover data (only once, as it is static)
    logging.INFO("Downloading ESA World Cover data...")
    wc_downloader = ESAWorldCoverDownloader(year=world_cover_year, cache_dir=cache_dir)
    outdir_wc = Path(os.getcwd()) / f"data/{location_nickname}/ESA_WorldCover/{world_cover_year}"
    outdir_wc.mkdir(parents=True, exist_ok=True)
    da_lc = wc_downloader.download(polygon=polygon, target_res_deg=target_res_deg)
    da_lc.rio.to_raster(str(outdir_wc / f"worldcover_{location_nickname}_{world_cover_year}_{target_res_deg}deg.tif"))

    # Initialize other downloaders for time series data
    logging.INFO("Initialising downloaders for timeseries...")
    spei_downloader = SPEIDownloader(cache_dir=cache_dir)
    era5_downloader = ERA5Downloader(cache_dir=cache_dir)
    era5p_downloader = ERA5PrecipDownloader(cache_dir=cache_dir)
    sm_downloader = ERA5SoilMoistureDownloader(cache_dir=cache_dir)
    irr_downloader = IrrigationMapDownloader(target_res_deg=target_res_deg, cache_dir=cache_dir)
    ndvi_downloader = MODISNDVIDownloader(cache_dir=cache_dir)

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
    logging.INFO("Time-series dataset download is complete.")


@click.command()
@click.option(
    '--geojson_path', 
    help='Path to GeoJSON file defining the location polygon.', 
    required=True
)
@click.option(
    '--location_nickname', 
    default=None, 
    help='Custom name to call location for data storage purposes.'
)
@click.option(
    '--start_year', 
    default=2018, 
    help='First year of data to download.'
)
@click.option(
    '--start_month', 
    default=4, 
    help='First month of data to download.'
)
@click.option(
    '--final_year', 
    default=2018, 
    help='Final year of data to download.'
)
@click.option(
    '--final_month', 
    default=5, 
    help='Final month of data to download.'
)
@click.option(
    '--world_cover_year', 
    default=2021, 
    help='Year of ESA World Cover data to download (only 2020 or 2021).'
)
@click.option(
    '--target_res_deg', 
    default=0.1, 
    help='Target resolution in degrees for World Cover and Irrigation Map data.'
)
def main(
    geojson_path: str, 
    location_nickname: str, 
    start_year: int, 
    start_month: int, 
    final_year: int, 
    final_month: int,
    world_cover_year: int,
    target_res_deg: float,
):
    """
    Main CLI entrypoint for downloading geospatial time series datasets.

    Parameters
    ----------
    geojson_path : str
        Path to GeoJSON file defining the location polygon.
    location_nickname : str or None
        Custom name for location; if None, uses GeoJSON filename stem.
    start_year : int
        First year of data to download.
    start_month : int
        First month of data to download.
    final_year : int
        Final year of data to download.
    final_month : int
        Final month of data to download.
    world_cover_year : int
        Year of ESA World Cover data to download (only 2020 or 2021).
    target_res_deg : float
        Target resolution in degrees for World Cover and Irrigation Map data.
    """
    # Load GeoJSON file
    json_path = Path(geojson_path)
    with open(json_path, 'r') as f:
        geojson_dict = json.load(f)
    
    # If no nickname provided, use the geojson filename (without extension)
    if not location_nickname:
        location_nickname = json_path.stem
    logging.INFO(f'Loaded {json_path}')
    
    # Execute the download
    download_timeseries_data(
        location_geojson=geojson_dict,
        location_nickname=location_nickname,
        start_year=start_year,
        start_month=start_month,
        final_year=final_year,
        final_month=final_month,
        world_cover_year=world_cover_year,
        target_res_deg=target_res_deg,
    )


if __name__ == "__main__":
    main()
