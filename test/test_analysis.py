import pytest
import rasterio
from rasterio.transform import xy
import numpy as np
from drought_causality.analysis import assemble_data_frame, assemble_timeseries_paths
from dowhy import CausalModel
from pathlib import Path

graph = """
digraph {
    soil_moisture -> ndvi;
    drought_severity -> soil_moisture;
    precipitation -> drought_severity;
    temperature -> {drought_severity ndvi};
    solar_radiation -> {drought_severity ndvi};
    world_cover -> {soil_moisture, irrigation ndvi};
    irrigation -> soil_moisture;
}
"""

def test_assemble_data_frame():
    precipitation_file = 'data/california_example/era5_precip_2021_07_california.tif'
    ndvi_file = 'data/california_example/ndvi_2021_07_california.tif'
    temperature_file = 'data/california_example/era5_t2m_2021_07_california.tif'
    solar_radiation_file = 'data/california_example/era5_ssrd_2021_07_california.tif'
    soil_moisture_file = 'data/california_example/era5_swvl1_2021_07_california.tif'
    world_cover_file = 'data/california_example/worldcover_2021_california_0p1deg.tif'
    irrigation_file = 'data/california_example/gmia_irrigation_0p1deg_california.tif'
    drought_severity_file = 'data/california_example/spei01_clipped_aoi_2021-07.tif'
    files = {
        'ndvi': ndvi_file,
        'precipitation': precipitation_file,
        'temperature': temperature_file,
        'solar_radiation': solar_radiation_file,
        'soil_moisture': soil_moisture_file,
        'world_cover': world_cover_file,
        'irrigation': irrigation_file,
        'drought_severity': drought_severity_file,
    }
    df = assemble_data_frame('ndvi', files)
    assert(df.shape == (29516, 12))
    df = df.dropna()
    model = CausalModel(
        data=df,
        treatment='drought_severity',
        outcome='ndvi',
        graph=graph
    )
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

def test_assemble_timeseries_paths_happy_path(tmp_path):
    # Directory layout:
    # root/
    #   2023/
    #     01/
    #     02/
    #   misc/
    root = tmp_path

    (root / "2023" / "01").mkdir(parents=True)
    (root / "2023" / "02").mkdir()
    (root / "misc").mkdir()         # should be ignored
    (root / "2023" / "xx").mkdir()  # should be ignored

    # Create some files
    (root / "2023" / "01" / "sst.nc").write_text("dummy")
    (root / "2023" / "01" / "ssh.nc").write_text("dummy")
    (root / "2023" / "02" / "sst.nc").write_text("dummy")
    (root / "2023" / "02" / "ssh.nc").write_text("dummy")

    dataset_files = {"sst": "sst.nc", "ssh": "ssh.nc"}

    result = assemble_timeseries_paths(root, dataset_files=dataset_files)

    assert len(result) == 2

    expected_01 = {
        "sst": str(root / "2023" / "01" / "sst.nc"),
        "ssh": str(root / "2023" / "01" / "ssh.nc"),
    }
    expected_02 = {
        "sst": str(root / "2023" / "02" / "sst.nc"),
        "ssh": str(root / "2023" / "02" / "ssh.nc"),
    }

    assert result[0] == expected_01
    assert result[1] == expected_02


def test_assemble_timeseries_paths_no_matching_dirs(tmp_path):
    # Only non-numeric dirs: should return empty list
    (tmp_path / "foo").mkdir()
    (tmp_path / "bar").mkdir()

    dataset_files = {"sst": "sst.nc"}

    result = assemble_timeseries_paths(tmp_path, dataset_files=dataset_files)
    assert result == []


def test_assemble_timeseries_paths_ignores_missing_files(tmp_path):
    # The function should construct paths even if the files don't exist
    (tmp_path / "2024" / "01").mkdir(parents=True)

    dataset_files = {"sst": "sst.nc", "ssh": "ssh.nc"}

    result = assemble_timeseries_paths(tmp_path, dataset_files=dataset_files)

    assert len(result) == 1
    paths = result[0]
    assert paths["sst"].endswith("2024/01/sst.nc") or paths["sst"].endswith("2024\\01\\sst.nc")
    assert paths["ssh"].endswith("2024/01/ssh.nc") or paths["ssh"].endswith("2024\\01\\ssh.nc")
