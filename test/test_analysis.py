import pytest
import rasterio
from rasterio.transform import xy, from_origin
import numpy as np
from drought_causality.analysis import (
    timeseries_causal_analysis,
)
from drought_causality.gather import (
    assemble_data_frame,
    assemble_timeseries_paths,
    assemble_timeseries,
)
from dowhy import CausalModel
from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock


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


def assert_affine_close(t1, t2, atol=1e-12, rtol=0):
    """
    Assert two rasterio Affine transforms are numerically close.
    """
    a1 = np.array([t1.a, t1.b, t1.c, t1.d, t1.e, t1.f], dtype=float)
    a2 = np.array([t2.a, t2.b, t2.c, t2.d, t2.e, t2.f], dtype=float)
    assert np.allclose(a1, a2, atol=atol, rtol=rtol), (t1, t2)


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


def test_assemble_timeseries_calls_helpers_in_order(monkeypatch):
    """
    assemble_timeseries should:
      * call assemble_timeseries_paths once with (root, dataset_files)
      * call assemble_data_frame once for each path dict it returns
      * concatenate the returned DataFrames in order.
    """

    root = "/fake/root"
    ref = "ndvi"
    dataset_files = {"ndvi": "ndvi.tif", "spei": "spei.tif"}

    # Fake list of per-timestep path dictionaries
    fake_paths = [
        {"ndvi": "/fake/root/2020/01/ndvi.tif", "spei": "/fake/root/2020/01/spei.tif"},
        {"ndvi": "/fake/root/2020/02/ndvi.tif", "spei": "/fake/root/2020/02/spei.tif"},
    ]

    # Track calls for verification
    calls = {"paths": [], "dfs": []}

    def fake_assemble_timeseries_paths(root_arg, dataset_files_arg):
        calls["paths"].append((root_arg, dataset_files_arg))
        return fake_paths

    def fake_assemble_data_frame(ref_arg, path_dict_arg):
        # Record arguments
        calls["dfs"].append((ref_arg, path_dict_arg))

        # Return a tiny DF that encodes which path dict we saw
        month_label = "2020-01" if "01" in path_dict_arg["ndvi"] else "2020-02"
        return pd.DataFrame(
            {
                "month": [month_label],
                "value": [1 if month_label == "2020-01" else 2],
            }
        )

    monkeypatch.setattr(
        "drought_causality.analysis.assemble_timeseries_paths",
        fake_assemble_timeseries_paths,
    )
    monkeypatch.setattr(
        "drought_causality.analysis.assemble_data_frame",
        fake_assemble_data_frame,
    )

    result = assemble_timeseries(root, ref, dataset_files)

    # --- Check helper calls ---
    # assemble_timeseries_paths is called exactly once with our arguments
    assert calls["paths"] == [(root, dataset_files)]

    # assemble_data_frame is called once per path dict, in the same order
    assert len(calls["dfs"]) == 2
    assert calls["dfs"][0][0] == ref  # first call, ref forwarded
    assert calls["dfs"][1][0] == ref  # second call, ref forwarded

    # Path dicts are passed through correctly, and in order
    assert calls["dfs"][0][1] is fake_paths[0]
    assert calls["dfs"][1][1] is fake_paths[1]

    # --- Check concatenation semantics ---
    # We expect two rows: one from "2020-01" and one from "2020-02"
    assert list(result["month"]) == ["2020-01", "2020-02"]
    assert list(result["value"]) == [1, 2]


def test_assemble_timeseries_propagates_concat_error_when_no_paths(monkeypatch):
    """
    When assemble_timeseries_paths returns an empty list, pd.concat([]) raises
    a ValueError. We assert that this behavior is preserved so callers can
    detect the missing data situation.
    """

    def fake_assemble_timeseries_paths(root_arg, dataset_files_arg):
        return []

    monkeypatch.setattr(
        "drought_causality.analysis.assemble_timeseries_paths",
        fake_assemble_timeseries_paths,
    )

    # We don't expect assemble_data_frame to be called at all here, so
    # no need to monkeypatch it.

    with pytest.raises(ValueError):
        assemble_timeseries("/fake/root", "ndvi", {"ndvi": "ndvi.tif"})

class FakeEstimate:
    def __init__(self, value):
        self.value = value


class FakeCausalModel:
    def __init__(self, data, treatment, outcome, graph):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome

    def identify_effect(self):
        return "estimand"

    def estimate_effect(self, estimand, method_name):
        return FakeEstimate(
            self.data[self.outcome].mean()
            - self.data[self.treatment].mean()
        )


def test_output_shape_matches_grid():
    df = pd.DataFrame(
        {
            "row": [0, 1, 2],
            "col": [0, 1, 2],
            "T": [1.0, 2.0, 3.0],
            "Y": [2.0, 4.0, 6.0],
        }
    )

    result = timeseries_causal_analysis(
        df,
        graph="digraph {}",
        treatment="T",
        outcome="Y",
        model_cls=FakeCausalModel,
    )

    assert result.shape == (3, 3)


def test_missing_cells_are_nan():
    df = pd.DataFrame(
        {
            "row": [0, 2],
            "col": [0, 2],
            "T": [1.0, 2.0],
            "Y": [3.0, 5.0],
        }
    )

    result = timeseries_causal_analysis(
        df,
        graph="digraph {}",
        treatment="T",
        outcome="Y",
        model_cls=FakeCausalModel,
    )

    assert np.isnan(result[1, 1])


def test_correct_cell_value():
    df = pd.DataFrame(
        {
            "row": [1, 1],
            "col": [2, 2],
            "T": [1.0, 3.0],
            "Y": [4.0, 6.0],
        }
    )

    result = timeseries_causal_analysis(
        df,
        graph="digraph {}",
        treatment="T",
        outcome="Y",
        model_cls=FakeCausalModel,
    )

    assert result[1, 2] == 3.0

