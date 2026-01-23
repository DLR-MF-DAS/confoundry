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

from downloaders import (
    ERA5Downloader, 
    ERA5PrecipDownloader, 
    ERA5SoilMoistureDownloader,
    ESAWorldCoverDownloader, 
    IrrigationMapDownloader, 
    MODISNDVIDownloader,
    SPEIDownloader
    )


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


class ImageRegistry:
    def __init__(self, db_path: str = "drought_data.duckdb"):
        # Connect to database
        self.db_connection = duckdb.connect(db_path)

        # Initialise database tables if they don't exist
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                location_id TEXT PRIMARY KEY,
                location_nickname TEXT,
                geojson JSON,
                first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS geotiff_catalog (
                catalog_id TEXT PRIMARY KEY,
                location_id TEXT,
                location_nickname TEXT,
                data_source TEXT,
                year INT,
                month INT,
                target_res_deg FLOAT,
                root_dir TEXT,
                file_name TEXT,
                file_size_bytes INT,
                download_status TEXT,
                error_message TEXT,
                first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT geotiff_unique UNIQUE (location_id, data_source, year, month, file_name)
            )
        """)

    def register_file(
            self, 
            location_id, 
            location_nickname, 
            data_source, 
            year, 
            month, 
            target_res_deg,
            root_dir, 
            file_name, 
            file_size_bytes, 
            download_status, 
            error
            ):
        new_catalog_id = str(uuid.uuid4())
        self.db_connection.execute("""
            INSERT INTO geotiff_catalog (
                catalog_id,
                location_id, 
                location_nickname, 
                data_source, 
                year, 
                month, 
                target_res_deg,
                root_dir, 
                file_name, 
                file_size_bytes, 
                download_status, 
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(location_id, data_source, year, month, file_name) DO UPDATE SET
                catalog_id=excluded.catalog_id,
                location_id=excluded.location_id,
                location_nickname=excluded.location_nickname,
                last_updated=now()
        """, 
        [new_catalog_id,
         location_id, 
         location_nickname, 
         data_source, 
         year,
         month, 
         target_res_deg,
         root_dir, 
         file_name, 
         file_size_bytes, 
         download_status, 
         error])

    def register_location(self, location_nickname, geojson):
        # Check if location_nickname exists
        row = self.db_connection.execute(
            "SELECT location_id, geojson FROM locations WHERE location_nickname = ?",
            [location_nickname]
        ).fetchone()
        geojson_str = json.dumps(geojson)
        if row:
            existing_id, existing_geojson = row
            if existing_geojson == geojson_str:
                # Same geojson, skip insert
                return existing_id
            else:
                # Different geojson for same nickname, raise error
                raise ValueError(f"Location nickname '{location_nickname}' already exists with different geojson.")
        # Insert new location
        new_location_id = str(uuid.uuid4())
        self.db_connection.execute(
            "INSERT INTO locations (location_id, location_nickname, geojson) VALUES (?, ?, ?)",
            [new_location_id, location_nickname, geojson_str]
        )
        return new_location_id
    
    def test(self):
        print(self.db_connection.execute("SELECT * FROM locations").df())
        print(self.db_connection.execute("SELECT * FROM geotiff_catalog").df().head(25))


# Helper to dynamically call class __init__ and download method

def dynamic_init(cls, available_args):
    sig = inspect.signature(cls.__init__)
    call_args = {k: available_args[k] for k in sig.parameters if k != 'self' and k in available_args}
    return cls(**call_args)

def dynamic_download(downloader, available_args):
    sig = inspect.signature(downloader.download)
    call_args = {k: available_args[k] for k in sig.parameters if k in available_args}
    return downloader.download(**call_args)

def process_download(
        database, 
        downloader, 
        downloader_name, 
        polygon, 
        location_id, 
        location_nickname, 
        year, 
        month, 
        target_res_deg, 
        output_folder):
    # Create output directory for this location
    output_dir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use month in basename only if not None
    if month is not None:
        basename = f"{downloader_name}_{location_nickname}_{year}_{month:02d}"
    else:
        basename = f"{downloader_name}_{location_nickname}_{year}"

    # Check if files already exist and are valid
    is_valid_dict = downloader.validate_geotiff(output_dir, basename)

    # Check database for existing successful entries
    if month is not None:
        db_files = database.db_connection.execute(
            """
            SELECT file_name, download_status FROM geotiff_catalog
            WHERE location_id = ? AND data_source = ? AND year = ? AND month = ?
            """,
            [location_id, downloader_name, year, month]
        ).fetchall()
    else:
        db_files = database.db_connection.execute(
            """
            SELECT file_name, download_status FROM geotiff_catalog
            WHERE location_id = ? AND data_source = ? AND year = ? AND month IS NULL
            """,
            [location_id, downloader_name, year]
        ).fetchall()
    db_files_dict = {row[0]: row[1] for row in db_files}

    # If all files are valid on disk and in DB, skip
    if all(is_valid_dict.values()) and all(db_files_dict.get(file_name) == "success" for file_name in is_valid_dict.keys()):
        if month is not None:
            logging.info(f"Validated in DB and on disk: All expected files for {downloader_name} {year}-{month:02d} for {location_nickname} exist.")
        else:
            logging.info(f"Validated in DB and on disk: All expected files for {downloader_name} {year} for {location_nickname} exist.")
        return

    # If files are valid and on disk but not the in DB, add to DB
    elif all(is_valid_dict.values()) and len(db_files_dict) == 0:
        for file_name, valid in is_valid_dict.items():
            file_path = output_dir / file_name
            database.register_file(
                location_id=location_id, 
                location_nickname=location_nickname, 
                data_source=downloader_name, 
                year=year, 
                month=month,
                target_res_deg=target_res_deg,
                root_dir=str(output_dir), 
                file_name=file_name.name, 
                file_size_bytes=os.path.getsize(file_path), 
                download_status="success", 
                error=None
            )
        if month is not None:
            logging.info(f"Validated on disk: All expected files for {downloader_name} {year}-{month:02d} for {location_nickname} exist.")
        else:
            logging.info(f"Validated on disk: All expected files for {downloader_name} {year} for {location_nickname} exist.")
        return

    # Proceed to download if files are missing
    try:
        # Prepare available args for download
        download_args = {
            "polygon": polygon,
            "year": year,
            "target_res_deg": target_res_deg,
        }
        if month is not None:
            download_args["month"] = month

        # Perform download, save, and validate
        dynamic_download(downloader, download_args)
        paths = downloader.save_geotiff(output_dir, basename)
        is_valid_dict = downloader.validate_geotiff(output_dir, basename)

        # Loop over saved files and register in DB
        for file_path in paths:
            try:
                # Register each file in the database
                valid = is_valid_dict.get(file_path, False)
                if valid:
                    download_status = "success"
                    error = None
                    file_size = os.path.getsize(file_path)
                else:
                    download_status = "failed"
                    error = "File validation failed"
                    file_size = None

                database.register_file(
                    location_id=location_id, 
                    location_nickname=location_nickname, 
                    data_source=downloader_name, 
                    year=year,
                    month=month,
                    target_res_deg=target_res_deg,
                    root_dir=str(output_dir), 
                    file_name=file_path.name,
                    file_size_bytes=file_size, 
                    download_status=download_status, 
                    error=error
                )

            # If registering a specific file fails, log failure for that file
            except Exception as e:
                database.register_file(
                    location_id=location_id, 
                    location_nickname=location_nickname, 
                    data_source=downloader_name, 
                    year=year, 
                    month=month,
                    target_res_deg=target_res_deg,
                    root_dir=str(output_dir), 
                    file_name=file_path.name, 
                    file_size_bytes=None,
                    download_status="failed",
                    error=str(e), 
                )

    # If download or save fails, log failure for all expected files
    except Exception as e:
        expected_filepaths = downloader.get_filepaths(output_dir, basename)
        for file_path in expected_filepaths:
            database.register_file(
                location_id=location_id, 
                location_nickname=location_nickname, 
                data_source=downloader_name, 
                year=year, 
                month=month,
                target_res_deg=target_res_deg,
                root_dir=str(output_dir), 
                file_name=file_path.name,
                file_size_bytes=None,
                download_status="failed",
                error=str(e), 
            )


@click.command()
@click.option(
    '--geojson_path', 
    help='Path to GeoJSON file defining the location polygon.', 
    required=True
)
@click.option(
    '--drought_db_path', 
    default='drought_data.duckdb',
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
    drought_db_path: str,
    location_nickname: str, 
    downloaders: list,
    start_year: int, 
    start_month: int, 
    final_year: int, 
    final_month: int,
    world_cover_year: int,
    target_res_deg: float,
    output_folder: str = "data",
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
    # If no downloaders specified, set to None
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


    # Initialise database and register location (if new)
    database = ImageRegistry(drought_db_path)
    location_id = database.register_location(location_nickname, geojson_dict)

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
        raise ValueError(f"Unrecognised downloaders: {invalid}")
    logging.info(f"Downloaders to be used: {downloaders}")

    # Get location polygon and nickname
    polygon = geojson_dict['features'][0]['geometry']

    # Create a cache directory for the temporary/reusable files
    cache_dir = Path(os.getcwd()) / f"{output_folder}/{location_nickname}/cache"
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


    # Core download loop with progress bar
    with tqdm.tqdm(total=total_tasks, desc="Downloading datasets") as pbar:
        for downloader_name in downloaders:
            # Initialise current downloader
            DownloaderClass = DOWNLOADERS_MAP[downloader_name]
            downloader = DownloaderClass(cache_dir=cache_dir)

            download_sig = inspect.signature(downloader.download)
            has_month = "month" in download_sig.parameters

            if has_month:
                for year in range(start_year, final_year + 1):
                    for month in range(1, 13):
                        # Skip months outside the specified range
                        if (year == start_year and month < start_month) or (year == final_year and month > final_month):
                            continue

                        # Create output directory for this location
                        process_download(
                            database=database, 
                            downloader=downloader, 
                            downloader_name=downloader_name, 
                            polygon=polygon, 
                            location_id=location_id, 
                            location_nickname=location_nickname, 
                            year=year, 
                            month=month, 
                            target_res_deg=target_res_deg, 
                            output_folder=output_folder
                        )
                        pbar.update(1)
            else:
                for year in range(start_year, final_year + 1):
                    # Create output directory for this location
                        process_download(
                            database=database, 
                            downloader=downloader, 
                            downloader_name=downloader_name, 
                            polygon=polygon, 
                            location_id=location_id, 
                            location_nickname=location_nickname, 
                            year=year, 
                            month=None, 
                            target_res_deg=target_res_deg, 
                            output_folder=output_folder
                        )
                        pbar.update(1)


if __name__ == "__main__":
    main()
    # database = ImageRegistry("drought_data.duckdb")
    # import pandas as pd
    # #pd.set_option('display.max_colwidth', None)
    # print(database.db_connection.execute("SELECT download_status, data_source, year, month, catalog_id, file_name FROM geotiff_catalog").df().head(50))