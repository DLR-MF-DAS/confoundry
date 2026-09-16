import os
import yaml
import json
import click
import dotenv
import logging
import datetime
from collections import Counter
from functools import wraps
from pathlib import Path

from fetcheo.loader import FetchEOLoader


# Set up basic logging config for CLI
logging.basicConfig(level=logging.INFO)

# Map available downloaders (for validation/help)
from fetcheo.loader import DOWNLOADER_DICT
AVAILABLE_DOWNLOADERS = list(DOWNLOADER_DICT.keys())


def load_config(config_path: str = "config.yaml") -> dict:
    # Load environment variables from .env file (if exists)
    dotenv.load_dotenv()

    # Check if config file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate config structure
    if config_dict:
        return config_dict
    else:
        raise ValueError(f"Configuration file '{config_path}' is empty or invalid.")


def validate_downloaders(downloaders):
    if not downloaders:
        return AVAILABLE_DOWNLOADERS
    invalid = [d for d in downloaders if d not in AVAILABLE_DOWNLOADERS]
    if invalid:
        raise click.ClickException(f"Unrecognised downloaders: {invalid}. Should be from {AVAILABLE_DOWNLOADERS}.")
    return list(downloaders)


# Helper function to ensure we get a datetime object
def ensure_datetime(date_input):
    if isinstance(date_input, str):
        return datetime.datetime.strptime(date_input, "%Y-%m-%d")
    elif isinstance(date_input, datetime.date) and not isinstance(date_input, datetime.datetime):
        # Convert date to datetime at midnight
        return datetime.datetime.combine(date_input, datetime.datetime.min.time())
    return date_input # Return as-is if it's already a datetime


def parse_and_validate_inputs(config_dict: dict):
    """
    Parse and validate input parameters.
    """
    # Convert start_date and end_date to datetime for comparison and downstream use
    start_date_dt = ensure_datetime(config_dict["start_date"])
    end_date_dt = ensure_datetime(config_dict["end_date"])
    if start_date_dt > end_date_dt:
        raise click.ClickException("start_date must be on or before end_date.")

    # Load GeoJSON file
    json_path = Path(config_dict["geojson_path"])
    with open(json_path, 'r') as f:
        geojson_dict = json.load(f)
    polygon = geojson_dict['features'][0]['geometry']
    
    # If no nickname provided, use the geojson filename (without extension)
    location_nickname = config_dict.get("name", None)
    if not location_nickname:
        location_nickname = json_path.stem
    logging.info(f'Loaded {json_path}')

    #
    db_path = Path(config_dict["db_path"])

    # Create a cache directory for the temporary/reusable files
    cache_dir = Path(config_dict["output_folder"]) / location_nickname / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get downloaders section from config and validate its keys
    downloaders_section = config_dict.get("downloaders") or {}
    validated_downloaders = validate_downloaders(list(downloaders_section.keys()))

    # Derive downloader_config and downloader_kwargs from validated downloaders
    downloader_config = {name: downloaders_section.get(name, {}).get("enabled", False) for name in validated_downloaders}
    downloader_kwargs = {
        name: {k: v for k, v in downloaders_section.get(name, {}).items() if k != "enabled"}
        for name in validated_downloaders
    }
    # logging.info("Downloaders configured: %s", list(downloader_config))
    return start_date_dt, end_date_dt, geojson_dict, polygon, location_nickname, db_path, cache_dir, downloader_config, downloader_kwargs


def attach_report_metadata(loader) -> None:
    """Complete the metadata on reports returned by FetchEO downloaders.

    FetchEO 0.0.5 stores ``report.frequency`` in its source catalogue, but its
    ``ItemDownloadReport`` does not define that field.  The downloader itself
    is the authoritative source for the value.  Its ERA5 downloader also
    reports short NetCDF names even when the configured mapping supplies
    canonical CDS variable names.  Decorating the enabled downloaders here
    keeps the upstream loader API intact and makes its catalogue directly
    consumable by :mod:`confoundry.gather`.
    """
    for downloader in loader.downloaders.values():
        original_fetch = downloader.fetch
        frequency = downloader.frequency
        variable_names = dict(getattr(downloader, "variables_dict", {}))

        @wraps(original_fetch)
        def fetch_with_metadata(
            *args,
            _original_fetch=original_fetch,
            _frequency=frequency,
            _variable_names=variable_names,
            **kwargs,
        ):
            reports = _original_fetch(*args, **kwargs)
            for report in reports:
                if getattr(report, "frequency", None) is None:
                    report.frequency = _frequency
                variable_name = getattr(report, "variable_name", None)
                if variable_name in _variable_names:
                    report.variable_name = _variable_names[variable_name]
            return reports

        downloader.fetch = fetch_with_metadata


def print_download_summary(reports, db_path: Path) -> None:
    """Print compact per-variable report counts for pipeline checkpoints."""
    counts = Counter(
        (
            str(getattr(report, "variable_name", "unknown")),
            str(getattr(report, "frequency", "unknown")),
            "success"
            if bool(getattr(report, "download_successful", False))
            else "failed",
        )
        for report in reports
    )
    click.echo("Download reports:")
    for variable_name, frequency in sorted(
        {(variable, frequency) for variable, frequency, _status in counts}
    ):
        successful = counts[(variable_name, frequency, "success")]
        failed = counts[(variable_name, frequency, "failed")]
        click.echo(
            f"  {variable_name} [{frequency}]: "
            f"success={successful}, failed={failed}"
        )
    click.echo(f"Source catalog: {db_path}::geotiff_catalog")


@click.command()
@click.option("--config-path", "-c", default="config.yaml", show_default=True, help="Path to YAML config file.")
def main(config_path):
    """Run FetchEOLoader from the command line."""
    # Load config from yaml
    config_dict = load_config(config_path)  

    #
    (start_dt, 
     end_dt,  
     _,
     polygon, 
     location_nickname, 
     db_path,
     _,
     downloader_config, 
     downloader_kwargs) = parse_and_validate_inputs(config_dict=config_dict)

    # Set up loader with enabled downloaders (default kwargs for now)
    loader = FetchEOLoader(
        downloader_config=downloader_config,
        downloader_kwargs=downloader_kwargs,
        db_path=Path(db_path)
    )
    attach_report_metadata(loader)

    # Place output in a subfolder under the location nickname
    data_output_dir = str(Path(config_dict.get("output_folder")) / location_nickname)
    show_progress = config_dict.get("show_progress", True)

    # Download data and add to DB
    reports = loader.fetch(
        polygon=polygon,
        time_frame=(start_dt, end_dt),
        location_nickname=location_nickname,
        output_dir=data_output_dir,
        show_progress=show_progress,
    )
    print_download_summary(reports, db_path)


if __name__ == "__main__":
    main()
