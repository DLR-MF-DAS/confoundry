import os
import json
import yaml
import click
import duckdb
import logging

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from confoundry.db_helpers import (
    connect_to_db,
    initialise_tables,
    fetch_or_create_location_id,
    upsert_file,
)

from confoundry.downloaders.spei import SPEIDownloader
from confoundry.downloaders.era5 import ERA5Downloader
from confoundry.downloaders.ecira import ECIRADownloader
from confoundry.downloaders.modis_ndvi import MODISNDVIDownloader
from confoundry.downloaders.esacci_landcover import ESACCILandCoverDownloader


DOWNLOADERS_MAP = {
    "spei": SPEIDownloader,
    "era5": ERA5Downloader,
    "ecira": ECIRADownloader,
    "modis_ndvi": MODISNDVIDownloader,
    "esacci_landcover": ESACCILandCoverDownloader,
}


def parse_and_validate_inputs(geojson_path, location_nickname,
                              downloaders, start_date, end_date,
                              output_folder):
    """
    Parse and validate input parameters.
    """
    output_folder = Path(output_folder)
    geojson_path = Path(geojson_path)

    start_date_dt = datetime.strptime(str(start_date), "%Y-%m-%d")
    end_date_dt = datetime.strptime(str(end_date), "%Y-%m-%d")

    if start_date_dt > end_date_dt:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    invalid_downloaders = [
        name for name in downloaders.keys()
        if name not in DOWNLOADERS_MAP
    ]

    if invalid_downloaders:
        raise ValueError(
            f"Unrecognised downloaders: {invalid_downloaders}. "
            f"Available downloaders are: {list(DOWNLOADERS_MAP.keys())}."
        )

    with geojson_path.open("r") as f:
        geojson_dict = json.load(f)

    polygon = geojson_dict["features"][0]["geometry"]

    if not location_nickname:
        location_nickname = geojson_path.stem

    cache_dir = output_folder / location_nickname / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loaded GeoJSON from %s", geojson_path)
    logging.info("Downloaders to be used: %s", downloaders)

    return (
        start_date_dt,
        end_date_dt,
        downloaders,
        geojson_dict,
        polygon,
        location_nickname,
        cache_dir,
    )


def setup_database(
    db_path: str | Path,
    location_nickname: str,
    geojson_dict: dict,
):
    """
    Initialise the database and register the location if needed.
    """
    database_connection = connect_to_db(db_path)
    initialise_tables(database_connection)

    location_id = fetch_or_create_location_id(
        database_connection,
        location_nickname,
        geojson_dict,
    )

    return database_connection, location_id


def add_reports_to_database(
    reports: list,
    frequency: str,
    location_id: str,
    location_nickname: str,
    database_connection: duckdb.DuckDBPyConnection,
):
    """
    Add downloader reports to the database.
    """
    for report in tqdm(reports, desc="Adding files to database", unit="file"):
        acquisition_time = report.acquisition_time

        year = acquisition_time.year
        month = getattr(acquisition_time, "month", None)

        upsert_file(
            db_connection=database_connection,
            location_id=location_id,
            location_nickname=location_nickname,
            data_source=report.data_source,
            variable_name=report.variable_name,
            frequency=frequency,
            year=year,
            month=month,
            root_dir=str(report.path.parent),
            file_name=report.path.name,
            file_size_bytes=os.path.getsize(report.path)
            if report.path.exists()
            else None,
            download_status="success" if report.download_successful else "failed",
            error_message=report.error,
            metadata=json.dumps(report.metadata)
            if report.metadata is not None
            else None,
        )


def run_downloading_pipeline(downloaders, polygon, start_date_dt, end_date_dt,
                             location_id, location_nickname, database_connection,
                             cache_dir, output_folder,):
    """
    Run each downloader sequentially and add its results to the database.
    """
    output_dir = Path(output_folder) / location_nickname
    output_dir.mkdir(parents=True, exist_ok=True)

    for downloader_name in tqdm(downloaders, desc="Running downloaders", unit="downloader"):
        DownloaderClass = DOWNLOADERS_MAP[downloader_name]

        downloader = DownloaderClass(
            cache_dir=cache_dir / downloader_name,
            **downloaders[downloader_name],
        )

        reports = downloader.download(
            polygon=polygon,
            time_frame=(start_date_dt, end_date_dt),
            output_dir=output_dir,
            show_progress=True,
        )

        add_reports_to_database(
            reports=reports,
            frequency=downloader.frequency,
            location_id=location_id,
            location_nickname=location_nickname,
            database_connection=database_connection,
        )


@click.command()
@click.option(
    "-c",
    "--config-path",
    help="Path to the YAML config file with experiment parameters.",
    required=True,
)
def main(config_path):
    """
    CLI entrypoint for downloading geospatial time series datasets.
    """
    config_path = Path(config_path)

    with config_path.open("r") as fd:
        config_data = yaml.safe_load(fd)

    experiment_dir = config_path.parent

    output_folder = experiment_dir / "data"
    geojson_path = experiment_dir / config_data["geojson"]
    location_nickname = config_data["name"]

    downloaders = config_data["downloaders"]["classes"]

    (
        start_date_dt,
        end_date_dt,
        downloaders,
        geojson_dict,
        polygon,
        location_nickname,
        cache_dir,
    ) = parse_and_validate_inputs(
        geojson_path=geojson_path,
        location_nickname=location_nickname,
        downloaders=downloaders,
        start_date=str(config_data["downloaders"]["start-date"]),
        end_date=str(config_data["downloaders"]["end-date"]),
        output_folder=output_folder,
    )

    db_path = experiment_dir / f"{location_nickname}_source_db.duckdb"

    database_connection, location_id = setup_database(
        db_path=db_path,
        location_nickname=location_nickname,
        geojson_dict=geojson_dict,
    )

    try:
        run_downloading_pipeline(
            downloaders=downloaders,
            polygon=polygon,
            start_date_dt=start_date_dt,
            end_date_dt=end_date_dt,
            location_id=location_id,
            location_nickname=location_nickname,
            database_connection=database_connection,
            cache_dir=cache_dir,
            output_folder=output_folder,
        )
    finally:
        database_connection.close()


if __name__ == "__main__":
    main()
