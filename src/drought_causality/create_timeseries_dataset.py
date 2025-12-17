import os
import json
import click
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from drought_causality.downloaders import (ERA5Downloader, 
                         ERA5PrecipDownloader, 
                         ERA5SoilMoistureDownloader,
                         ESAWorldCoverDownloader, 
                         IrrigationMapDownloader, 
                         MODISNDVIDownloader,
                         SPEIDownloader)


# Dictionary mapping downloader names to their classes
DOWNLOADERS_MAP = {
    "spei": SPEIDownloader,
    "era5": ERA5Downloader,
    "era5_precip": ERA5PrecipDownloader,
    "era5_soil_moisture": ERA5SoilMoistureDownloader,
    "esa_world_cover": ESAWorldCoverDownloader,
    "irrigation_map": IrrigationMapDownloader,
    "modis_ndvi": MODISNDVIDownloader,
}


def add_report_entry(
        download_report_list: list,
        downloader_name: str, 
        year: int = None,
        month: int = None, 
        error: str = None
        ):
    download_report_list.append({
        "time": datetime.now(),
        "downloader": downloader_name,
        "year": year,
        "month": month,
        "status": "failed" if error else "success",
        "error": error if error else None
    })


def download_timeseries_data(
        location_geojson: dict,
        location_nickname: str ,
        downloaders: list[str] = None,
        start_year: int = 2009,
        start_month: int = 1,
        final_year: int = 2019,
        final_month: int = 12,
        world_cover_year: int = 2021,
        target_res_deg: float = 0.1,
        output_folder: str = "data",
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

    # Validate requested downloaders (defaults to all if no downloaders specified)
    if downloaders is None:
        downloaders = list(DOWNLOADERS_MAP.keys())
    # If downloaders is a string, convert to a single-element list
    if isinstance(downloaders, str):
        downloaders = [downloaders.split(',')]
    invalid = [d for d in downloaders if d not in DOWNLOADERS_MAP]
    if invalid:
        raise ValueError(f"Unrecognized downloaders: {invalid}")
    logging.info(f"Downloaders to be used: {downloaders}")

    # Get location polygon and nickname
    polygon = location_geojson['features'][0]['geometry']

    # Create a cache directory for the temporary/reusable files
    cache_dir = Path(os.getcwd()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # First, count total tasks
    total_tasks = 0
    for downloader_name in downloaders:
        if downloader_name in ["esa_world_cover", "irrigation_map"]:
            total_tasks += 1
        else:
            # Count months in range
            for year in range(start_year, final_year + 1):
                for month in range(1, 13):
                    if (year == start_year and month < start_month) or (year == final_year and month > final_month):
                        continue
                    total_tasks += 1

    download_report_list = []
    with tqdm(total=total_tasks, desc="Downloading datasets") as pbar:
        for downloader_name in downloaders:
            DownloaderClass = DOWNLOADERS_MAP[downloader_name]

            if downloader_name == "esa_world_cover":
                try:
                    pbar.set_description(f"{downloader_name} static {world_cover_year}")
                    downloader = DownloaderClass(year=world_cover_year, cache_dir=cache_dir)
                    outdir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/static"
                    outdir.mkdir(parents=True, exist_ok=True)
                    downloader.download(polygon=polygon, target_res_deg=target_res_deg)
                    downloader.save_geotiff(output_dir=outdir, basename=f"worldcover_{location_nickname}_{world_cover_year}_{target_res_deg}deg")
                    add_report_entry(
                        download_report_list=download_report_list,
                        downloader_name=downloader_name,
                        year=world_cover_year
                    )
                except Exception as e:
                    add_report_entry(
                        download_report_list=download_report_list,
                        downloader_name=downloader_name,
                        year=world_cover_year,
                        error=str(e)
                    )
                    logging.error(f"ESA World Cover download failed for {world_cover_year}: {e}")
                pbar.update(1)

            elif downloader_name == "irrigation_map":
                try:
                    pbar.set_description(f"{downloader_name} static")
                    downloader = DownloaderClass(target_res_deg=target_res_deg, cache_dir=cache_dir)
                    outdir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/static"
                    outdir.mkdir(parents=True, exist_ok=True)
                    downloader.download(polygon=polygon)
                    downloader.save_geotiff(output_dir=outdir, basename=f"gmia_irrigation_{location_nickname}_{target_res_deg}deg")
                    add_report_entry(
                        download_report_list=download_report_list,
                        downloader_name=downloader_name,
                    )
                except Exception as e:
                    add_report_entry(
                        download_report_list=download_report_list,
                        downloader_name=downloader_name,
                        error=str(e)
                    )
                    logging.error(f"Irrigation Map download failed: {e}")
                pbar.update(1)

            else:
                downloader = DownloaderClass(cache_dir=cache_dir)
                for year in range(start_year, final_year + 1):
                    for month in range(1, 13):
                        if (year == start_year and month < start_month) or (year == final_year and month > final_month):
                            continue
                        try:
                            pbar.set_description(f"{downloader_name} {year}-{month:02d}")
                            logging.info(f"Downloading {downloader_name} for {year}-{month:02d}...")
                            downloader.download(polygon=polygon, year=year, month=month)
                            outdir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/{year}/{month}"
                            outdir.mkdir(parents=True, exist_ok=True)
                            downloader.save_geotiff(output_dir=outdir, basename=f"{downloader_name}_{location_nickname}_{year}_{month:02d}")
                            logging.info(f"{downloader_name} downloaded successfully for {year}-{month:02d}.")
                            add_report_entry(
                                download_report_list=download_report_list,
                                downloader_name=downloader_name,
                                year=year,
                                month=month)
                        except Exception as e:
                            logging.error(f"{downloader_name} failed for {year}-{month:02d}: {e}")
                            add_report_entry(
                                download_report_list=download_report_list,
                                downloader_name=downloader_name,
                                year=year,
                                month=month,
                                error=str(e)
                            )
                        pbar.update(1)
                        
    # Save download report as CSV
    report_df = pd.DataFrame(download_report_list)
    report_csv_path  = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/download_report.csv"
    report_df.to_csv(report_csv_path , index=False)
    logging.info("Time-series dataset download is complete.")


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
    '--downloaders', 
     help='List of downloaders to use (e.g. --downloaders spei --downloaders era5). If not specified, all downloaders are used.',
    default=None,
    multiple=True,
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
    downloaders: list,
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
    if not downloaders:
        downloaders = None
    # Load GeoJSON file
    json_path = Path(geojson_path)
    with open(json_path, 'r') as f:
        geojson_dict = json.load(f)
    
    # If no nickname provided, use the geojson filename (without extension)
    if not location_nickname:
        location_nickname = json_path.stem
    logging.info(f'Loaded {json_path}')
    
    # Execute the download
    download_timeseries_data(
        location_geojson=geojson_dict,
        location_nickname=location_nickname,
        downloaders=downloaders,
        start_year=start_year,
        start_month=start_month,
        final_year=final_year,
        final_month=final_month,
        world_cover_year=world_cover_year,
        target_res_deg=target_res_deg,
    )


if __name__ == "__main__":
    main()
