import os
import yaml
import json
import click
import logging
from pathlib import Path
from datetime import datetime

from fetcheo.loader import FetchEOLoader


# Set up basic logging config for CLI
logging.basicConfig(level=logging.INFO)

# Map available downloaders (for validation/help)
from fetcheo.loader import DOWNLOADER_DICT
AVAILABLE_DOWNLOADERS = list(DOWNLOADER_DICT.keys())


def load_config(config_path: str = "config.yaml") -> dict:
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


def parse_and_validate_inputs(config_dict: dict):
    """
    Parse and validate input parameters.
    """
    # Convert start_date and end_date to datetime for comparison and downstream use
    start_date_dt = datetime.strptime(config_dict["start-date"], "%Y-%m-%d")
    end_date_dt = datetime.strptime(config_dict["end-date"], "%Y-%m-%d")
    if start_date_dt > end_date_dt:
        raise ValueError("start_date must be on or before end_date.")

    # Load GeoJSON file
    json_path = Path(config_dict["geojson"])
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
    cache_dir = Path(os.getcwd()) / f"{config_dict['output_folder']}/{location_nickname}/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get downloaders from config and validate against available options
    downloader_config = {name: attrs.get("enabled", False) for name, attrs in config_dict.get("downloaders", {}).items()}
    validate_downloaders(downloader_config)

    # Extract kwargs for each downloader (excluding the "enabled" flag)
    downloader_kwargs = {
        name: {k: v for k, v in attrs.items() if k != "enabled"}
        for name, attrs in config_dict.get("downloaders", {}).items()
    }
	#logging.info(f"Downloaders to be used: {downloaders}")
    return start_date_dt, end_date_dt, geojson_dict, polygon, location_nickname, db_path, cache_dir, downloader_config, downloader_kwargs


@click.command()
@click.option('--config-path', '-c', required=True, default="config.yaml", help=f"Path to YAML config file(s) (overrides other options)")
def main(config_path):
    """Run FetchEOLoader from the command line."""
    # Load config from yaml
    config_dict = load_config(config_path)  

    #
    (start_dt, 
     end_dt,  
     polygon, 
     location_nickname, 
     db_path,
     downloader_config, 
     downloader_kwargs) = parse_and_validate_inputs(config_dict=config_dict)

    # Set up loader with enabled downloaders (default kwargs for now)
    loader = FetchEOLoader(
        downloader_config=downloader_config,
        downloader_kwargs=downloader_kwargs,
        db_path=Path(db_path)
    )

    # Place output in a subfolder under the location nickname
    data_output_dir = str(Path(config_dict.get("output_folder"), "data") / location_nickname)
    show_progress = config_dict.get("show_progress", True)

    # Download data and add to DB
    loader.fetch(
        polygon=polygon,
        time_frame=(start_dt, end_dt),
        location_nickname=location_nickname,
        output_dir=data_output_dir,
        show_progress=show_progress,
    )


if __name__ == '__main__':
	main()

