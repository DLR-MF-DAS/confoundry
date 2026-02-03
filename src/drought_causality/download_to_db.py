import os
import re
import json
import tqdm
import uuid
import click
import duckdb
import inspect
import logging
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime

from drought_causality.duckdb_helpers import connect_to_db, initialise_tables, upsert_file, upsert_location

from drought_causality.downloaders.spei import SPEIDownloader
from drought_causality.downloaders.era5 import ERA5Downloader
from drought_causality.downloaders.ecira import ECIRADownloader
from drought_causality.downloaders.modis_ndvi import MODISNDVIDownloader 
from drought_causality.downloaders.esacci_landcover import ESACCILandCoverDownloader  


# Dictionary mapping downloader names to their classes
DOWNLOADERS_MAP = {
    "spei": SPEIDownloader,
    "era5": ERA5Downloader,
    "ecira": ECIRADownloader,
    "modis_ndvi": MODISNDVIDownloader,
    "esacci_landcover": ESACCILandCoverDownloader,
}


def parse_and_validate_inputs(
        geojson_path: str,
        location_nickname: str,
        downloaders: tuple,
        start_date: str, 
        end_date: str,
        output_folder: str
    ):
    """
    Parse and validate input parameters.
    """
    # Convert start_date and end_date to datetime for comparison and downstream use
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    assert start_date_dt <= end_date_dt

    # If no downloaders specified, set to all
    if not downloaders:
        downloaders = list(DOWNLOADERS_MAP.keys())

    # Ensure downloaders is a list of strings
    if isinstance(downloaders, tuple):
        downloaders = list(downloaders)
    elif isinstance(downloaders, str):
        downloaders = [downloaders]
    invalid = [d for d in downloaders if d not in DOWNLOADERS_MAP]
    if invalid:
        raise ValueError(f"Unrecognised downloaders: {invalid}. Should be from {list(DOWNLOADERS_MAP.keys())}.")
    logging.info(f"Downloaders to be used: {downloaders}")

    # Load GeoJSON file
    json_path = Path(geojson_path)
    with open(json_path, 'r') as f:
        geojson_dict = json.load(f)
    polygon = geojson_dict['features'][0]['geometry']
    
    # If no nickname provided, use the geojson filename (without extension)
    if not location_nickname:
        location_nickname = json_path.stem
    logging.info(f'Loaded {json_path}')

    # Create a cache directory for the temporary/reusable files
    cache_dir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return start_date_dt, end_date_dt, downloaders, geojson_dict, polygon, location_nickname, cache_dir


def setup_database(db_path: str, location_nickname: str, geojson_dict: dict):
    # Initialise database and register location (if new)
    database_connection = connect_to_db(db_path)
    initialise_tables(database_connection)
    location_id = upsert_location(database_connection, location_nickname, geojson_dict)
    return database_connection, location_id


def run_downloading_pipeline(
    downloaders: list,
    polygon: dict,
    start_date_dt: datetime,
    end_date_dt: datetime,
    location_id: str,
    location_nickname: str,
    database_connection: duckdb.DuckDBPyConnection,
    cache_dir: Path,
    output_folder: str
):
    # Core download loop with progress bar
    for downloader_name in downloaders:
        # Initialise current downloader
        DownloaderClass = DOWNLOADERS_MAP[downloader_name]
        downloader = DownloaderClass(cache_dir=cache_dir)

        # Download all data within selected time frame
        logging.info(f"Starting downloads for {downloader_name}...")
        download_report_list = downloader.download(
            polygon=polygon,
            time_frame=(start_date_dt, end_date_dt),
            output_dir=Path(os.getcwd()) / f"{output_folder}/{location_nickname}",
            show_progress=True,
        )
        logging.info(f"Completed downloads for {downloader_name}.")

        # Add each downloaded file to the database
        logging.info(f"Adding files to database for {downloader_name}...")
        for report in tqdm.tqdm(
            download_report_list,
            desc=f"Adding {downloader_name} downloads to database",
            unit="file"
        ):
            year = report.acquisition_time.year
            month = report.acquisition_time.month if hasattr(report.acquisition_time, 'month') else None

            upsert_file(
                db_connection=database_connection,
                location_id=location_id,
                location_nickname=location_nickname,
                data_source=report.data_source,
                variable_name=report.variable_name,
                year=year,
                month=month,
                root_dir=str(report.path.parent),
                file_name=report.path.name,
                file_size_bytes=os.path.getsize(report.path) if report.path.exists() else None,
                download_status="success" if report.download_successful else "failed",
                error_message=report.error
            )
        logging.info(f"Added files to database for {downloader_name}.")


@click.command()
@click.option(
    '--geojson_path', 
    help='Path to GeoJSON file defining the location polygon.', 
    required=True
)
@click.option(
    '--db_path', 
    default='data.duckdb',
    help='Path to the DuckDB database file to create or use.', 
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
    '--start_date', 
    default='2014-01-01', 
    help='First YYYY-MM-DD date of data to download.'
)
@click.option(
    '--end_date', 
    default='2014-03-31', 
    help='Final YYYY-MM-DD date of data to download.'
)
def main(
    geojson_path: str,
    db_path: str,
    location_nickname: str,
    downloaders: tuple,
    start_date: str,
    end_date: str,
    output_folder: str = "data",
):
    """
    Main CLI entrypoint for downloading geospatial time series datasets.

    Parameters
    ----------
    geojson_path : str
        Path to GeoJSON file defining the location polygon.
    db_path : str
        Path to the DuckDB database file to create or use.
    location_nickname : str or None
        Custom name for location; if None, uses GeoJSON filename stem.
    downloaders : tuple
        List of downloaders to use (e.g. ('spei', 'era5')).
    start_date : str
        First YYYY-MM-DD date of data to download.
    end_date : str
        Final YYYY-MM-DD date of data to download.
    output_folder : str
        Output folder for downloaded data.
    """
    
    # Check the inputs are good to go
    start_date_dt, end_date_dt, downloaders, geojson_dict, polygon, location_nickname, cache_dir = parse_and_validate_inputs(
        geojson_path=geojson_path,
        location_nickname=location_nickname,
        downloaders=downloaders,
        start_date=start_date, 
        end_date=end_date,
        output_folder=output_folder
    )

    # Setup database connection and make an entry for this location (if new)
    database_connection, location_id = setup_database(
        db_path=db_path, 
        location_nickname=location_nickname, 
        geojson_dict=geojson_dict
    )

    # Run all the downloaders and add the files to the database
    run_downloading_pipeline(
        downloaders=downloaders,
        polygon=polygon,
        start_date_dt=start_date_dt,
        end_date_dt=end_date_dt,
        location_id=location_id,
        location_nickname=location_nickname,
        database_connection=database_connection,
        cache_dir=cache_dir,
        output_folder=output_folder
    )
        

if __name__ == "__main__":
    main()