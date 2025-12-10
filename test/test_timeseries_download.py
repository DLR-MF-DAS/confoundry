import json
from pathlib import Path

from drought_causality.create_timeseries_dataset import download_timeseries_data


def test_download_timeseries_data():
    with open('data/california.json', 'r') as fd:
        geojson = json.load(fd)
    polygon = geojson['features'][0]['geometry']
    location_nickname = "california_test"
    start_year = 2021
    start_month = 7
    final_year = 2021
    final_month = 8  # Now testing two months: July and August

    # Run the downloader for timeseries
    download_timeseries_data(
        polygon=polygon,
        location_nickname=location_nickname,
        start_year=start_year,
        start_month=start_month,
        final_year=final_year,
        final_month=final_month,
        clear_cache=True
    )

    # Check that all expected files exist in the output directories
    outdirs = [Path(f"data/{location_nickname}/{start_year}/{month:02d}") for month in range(start_month, final_month + 1)]
    expected_patterns = [
        f"spei_{location_nickname}_*.tif",
        f"era5_t2m_{location_nickname}_*.tif",
        f"era5_ssrd_{location_nickname}_*.tif",
        f"era5_precip_{location_nickname}_*.tif",
        f"era5_swvl1_{location_nickname}_*.tif",
        f"gmia_irrigation_{location_nickname}_*.tif",
        f"ndvi_{location_nickname}_*.tif"
    ]
    for outdir in outdirs:
        for pattern in expected_patterns:
            matches = list(outdir.glob(pattern))
            assert matches, f"Missing output file matching pattern: {pattern} in {outdir}"