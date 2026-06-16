import pytest
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.crs import CRS
from confoundry.gather import map_pixel_to_all, assemble_data_frame
import confoundry.gather


class DummyDataset:
    def __init__(self, transform, width, height, crs=None):
        self.transform = transform
        self.width = width
        self.height = height
        self.crs = crs


@pytest.fixture
def base_transform():
    # pixel size 1x1, origin at (0, 10)
    return Affine.translation(0, 10) * Affine.scale(1, -1)


@pytest.fixture
def datasets(base_transform):
    return {
        "ref": DummyDataset(
            transform=base_transform,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
        "same": DummyDataset(
            transform=base_transform,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
    }


def test_missing_reference_returns_empty_dict(datasets):
    result = map_pixel_to_all(
        row=1,
        col=1,
        ref="does_not_exist",
        datasets=datasets,
    )

    assert result == {}


def test_maps_reference_pixel_to_same_grid(datasets):
    result = map_pixel_to_all(
        row=3,
        col=4,
        ref="ref",
        datasets=datasets,
    )

    assert result["ref"] == (3, 4)
    assert result["same"] == (3, 4)


def test_out_of_bounds_returns_none_when_enabled(base_transform):
    datasets = {
        "ref": DummyDataset(
            transform=base_transform,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
        # shifted far away
        "shifted": DummyDataset(
            transform=Affine.translation(100, 100) * Affine.scale(1, -1),
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
    }

    result = map_pixel_to_all(
        row=2,
        col=2,
        ref="ref",
        datasets=datasets,
        bounds_check=True,
    )

    assert result["shifted"] is None


def test_out_of_bounds_returns_indices_when_disabled(base_transform):
    datasets = {
        "ref": DummyDataset(
            transform=base_transform,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
        "shifted": DummyDataset(
            transform=Affine.translation(100, 100) * Affine.scale(1, -1),
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
        ),
    }

    result = map_pixel_to_all(
        row=2,
        col=2,
        ref="ref",
        datasets=datasets,
        bounds_check=False,
    )

    assert result["shifted"] is not None
    assert isinstance(result["shifted"], tuple)
    assert len(result["shifted"]) == 2


def test_handles_crs_reprojection():
    datasets = {
        "ref": DummyDataset(
            transform=Affine.translation(-180, 90) * Affine.scale(0.1, -0.1),
            width=3600,
            height=1800,
            crs=CRS.from_epsg(4326),
        ),
        "mercator": DummyDataset(
            transform=Affine.translation(
                -20037508.34,
                20037508.34,
            ) * Affine.scale(1000, -1000),
            width=40000,
            height=40000,
            crs=CRS.from_epsg(3857),
        ),
    }

    result = map_pixel_to_all(
        row=100,
        col=100,
        ref="ref",
        datasets=datasets,
    )

    assert result["mercator"] is not None
    r, c = result["mercator"]

    assert isinstance(r, np.int32)
    assert isinstance(c, np.int32)


def test_transform_failure_returns_none(monkeypatch, datasets):
    import confoundry.gather
    
    def broken_rowcol(*args, **kwargs):
        raise ValueError("forced failure")

    monkeypatch.setattr(confoundry.gather, "rowcol", broken_rowcol)

    result = map_pixel_to_all(
        row=1,
        col=1,
        ref="ref",
        datasets=datasets,
    )

    assert result["ref"] is None
    assert result["same"] is None

class DummyDataset2:
    def __init__(
        self,
        array,
        transform,
        crs=None,
        nodata=None,
        scales=None,
        offsets=None,
    ):
        self.array = array
        self.transform = transform
        self.crs = crs
        self.nodata = nodata
        self.scales = scales or [1.0]
        self.offsets = offsets or [0.0]
        self.profile = {
            "height": array.shape[0],
            "width": array.shape[1],
        }
        self.height = array.shape[0]
        self.width = array.shape[1]
        self.closed = False

    def read(self, band):
        assert band == 1
        return self.array.copy()

    def close(self):
        self.closed = True


@pytest.fixture
def transform():
    return Affine.translation(0, 10) * Affine.scale(1, -1)


@pytest.fixture
def raster_data():
    return np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.float32,
    )


def test_returns_dataframe(monkeypatch, transform, raster_data):
    import confoundry.gather

    datasets = {
        "a": DummyDataset2(
            raster_data,
            transform,
            CRS.from_epsg(4326),
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    task = (2024, 6, "a", {"a": "a"})

    df = assemble_data_frame(task)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4


def test_dataframe_contains_expected_columns(
    monkeypatch,
    transform,
    raster_data,
):
    import confoundry.gather

    datasets = {
        "a": DummyDataset2(
            raster_data,
            transform,
            CRS.from_epsg(4326),
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame((2024, 1, "a", {"a": "a"}))

    expected = {
        "year",
        "month",
        "row",
        "col",
        "x",
        "y",
        "month_sin",
        "month_cos",
        "a",
    }

    assert expected.issubset(df.columns)


def test_applies_scale_and_offset(monkeypatch, transform):
    import confoundry.gather

    arr = np.array([[10]], dtype=np.float32)

    datasets = {
        "a": DummyDataset2(
            arr,
            transform,
            scales=[2.0],
            offsets=[5.0],
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame((2024, 1, "a", {"a": "a"}))

    # (10 * 2) + 5
    assert df.iloc[0]["a"] == 25.0


def test_replaces_nodata_with_nan(monkeypatch, transform):
    import confoundry.gather

    arr = np.array([[9999]], dtype=np.float32)

    datasets = {
        "a": DummyDataset2(
            arr,
            transform,
            nodata=9999,
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame((2024, 1, "a", {"a": "a"}))

    assert np.isnan(df.iloc[0]["a"])


def test_missing_reference_returns_empty_dataframe(
    monkeypatch,
    transform,
    raster_data,
):
    import confoundry.gather

    datasets = {
        "a": DummyDataset2(
            raster_data,
            transform,
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame(
        (2024, 1, "missing", {"a": "a"})
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_out_of_bounds_pixel_becomes_nan(monkeypatch, transform):
    import confoundry.gather
    
    arr1 = np.array([[1]], dtype=np.float32)
    arr2 = np.array([[2]], dtype=np.float32)

    datasets = {
        "ref": DummyDataset2(arr1, transform),
        "other": DummyDataset2(
            arr2,
            Affine.translation(100, 100) * Affine.scale(1, -1),
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame(
        (
            2024,
            1,
            "ref",
            {
                "ref": "ref",
                "other": "other",
            },
        )
    )

    assert np.isnan(df.iloc[0]["other"])


def test_closes_all_datasets(monkeypatch, transform, raster_data):
    import confoundry.gather

    datasets = {
        "a": DummyDataset2(raster_data, transform),
        "b": DummyDataset2(raster_data, transform),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    assemble_data_frame(
        (
            2024,
            1,
            "a",
            {
                "a": "a",
                "b": "b",
            },
        )
    )

    assert datasets["a"].closed is True
    assert datasets["b"].closed is True


def test_month_features_are_correct(monkeypatch, transform):
    import confoundry.gather

    datasets = {
        "a": DummyDataset2(
            np.array([[1]], dtype=np.float32),
            transform,
        ),
    }

    monkeypatch.setattr(
        confoundry.gather.rasterio,
        "open",
        lambda path, mode="r": datasets[path],
    )

    df = assemble_data_frame((2024, 3, "a", {"a": "a"}))

    expected_sin = np.sin(2 * np.pi * (3 / 12.0))
    expected_cos = np.cos(2 * np.pi * (3 / 12.0))

    assert np.isclose(df.iloc[0]["month_sin"], expected_sin)
    assert np.isclose(df.iloc[0]["month_cos"], expected_cos)


def test_assemble_timeseries_builds_tasks_and_concatenates(monkeypatch):
    path_dict = {
        (2020, 1): {
            "temperature": "/data/2020/01/temp.tif",
            "precipitation": "/data/2020/01/precip.tif",
        },
        (2020, 2): {
            "temperature": "/data/2020/02/temp.tif",
            "precipitation": "/data/2020/02/precip.tif",
        },
    }

    database = "/tmp/test.duckdb"
    name_map = {"temperature": "tas", "precipitation": "pr"}
    ref = "temperature"

    captured = {}

    def fake_assemble_timeseries_paths_from_db(db, nm):
        captured["database"] = db
        captured["name_map"] = nm
        return path_dict

    def fake_process_map(func, tasks, max_workers, ascii):
        captured["func"] = func
        captured["tasks"] = tasks
        captured["max_workers"] = max_workers
        captured["ascii"] = ascii

        return [
            pd.DataFrame(
                {
                    "year": [2020],
                    "month": [1],
                    "value": [10.0],
                }
            ),
            pd.DataFrame(
                {
                    "year": [2020],
                    "month": [2],
                    "value": [20.0],
                }
            ),
        ]

    monkeypatch.setattr(
        confoundry.gather,
        "assemble_timeseries_paths_from_db",
        fake_assemble_timeseries_paths_from_db,
    )
    monkeypatch.setattr(confoundry.gather, "process_map", fake_process_map)

    result = confoundry.gather.assemble_timeseries(
        database=database,
        name_map=name_map,
        ref=ref,
        max_workers=4,
    )

    expected_tasks = [
        (
            2020,
            1,
            ref,
            {
                "temperature": "/data/2020/01/temp.tif",
                "precipitation": "/data/2020/01/precip.tif",
            },
        ),
        (
            2020,
            2,
            ref,
            {
                "temperature": "/data/2020/02/temp.tif",
                "precipitation": "/data/2020/02/precip.tif",
            },
        ),
    ]

    assert captured["database"] == database
    assert captured["name_map"] == name_map
    assert captured["func"] is confoundry.gather.assemble_data_frame
    assert captured["tasks"] == expected_tasks
    assert captured["max_workers"] == 4
    assert captured["ascii"] is True

    expected = pd.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 2],
            "value": [10.0, 20.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_assemble_timeseries_uses_default_max_workers(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        confoundry.gather,
        "assemble_timeseries_paths_from_db",
        lambda database, name_map: {
            (2021, 6): {"temperature": "/data/2021/06/temp.tif"}
        },
    )

    def fake_process_map(func, tasks, max_workers, ascii):
        captured["max_workers"] = max_workers
        return [
            pd.DataFrame(
                {
                    "year": [2021],
                    "month": [6],
                    "value": [42.0],
                }
            )
        ]

    monkeypatch.setattr(confoundry.gather, "process_map", fake_process_map)

    result = confoundry.gather.assemble_timeseries(
        database="/tmp/test.duckdb",
        name_map={"temperature": "tas"},
        ref="temperature",
    )

    assert captured["max_workers"] == 1
    assert len(result) == 1
    assert result.loc[0, "value"] == 42.0


def test_assemble_timeseries_empty_path_dict_raises_value_error(monkeypatch):
    monkeypatch.setattr(
        confoundry.gather,
        "assemble_timeseries_paths_from_db",
        lambda database, name_map: {},
    )

    monkeypatch.setattr(
        confoundry.gather,
        "process_map",
        lambda func, tasks, max_workers, ascii: [],
    )

    with pytest.raises(ValueError, match="No objects to concatenate"):
        confoundry.gather.assemble_timeseries(
            database="/tmp/test.duckdb",
            name_map={},
            ref="temperature",
        )


def test_assemble_timeseries_preserves_process_map_output_order(monkeypatch):
    monkeypatch.setattr(
        confoundry.gather,
        "assemble_timeseries_paths_from_db",
        lambda database, name_map: {
            (2022, 12): {"a": "/a.tif"},
            (2022, 1): {"a": "/b.tif"},
        },
    )

    def fake_process_map(func, tasks, max_workers, ascii):
        return [
            pd.DataFrame({"month": [12], "value": ["first"]}),
            pd.DataFrame({"month": [1], "value": ["second"]}),
        ]

    monkeypatch.setattr(confoundry.gather, "process_map", fake_process_map)

    result = confoundry.gather.assemble_timeseries(
        database="/tmp/test.duckdb",
        name_map={"a": "a"},
        ref="a",
    )

    assert result["month"].tolist() == [12, 1]
    assert result["value"].tolist() == ["first", "second"]


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows):
        self._rows = rows
        self.executed_sql = None

    def execute(self, sql):
        self.executed_sql = sql
        return FakeCursor(self._rows)


def patch_duckdb_connect(monkeypatch, rows):
    connection = FakeConnection(rows)

    def fake_connect(database):
        connection.database = database
        return connection

    monkeypatch.setattr(confoundry.gather.duckdb, "connect", fake_connect)
    return connection


def test_assemble_timeseries_paths_from_db_adds_monthly_files(
    tmp_path,
    monkeypatch,
):
    raster = tmp_path / "temperature_2020_01.tif"
    raster.touch()

    rows = [
        (
            "tas_raw",
            "monthly",
            str(tmp_path),
            "temperature_2020_01.tif",
            2020,
            1,
        )
    ]

    connection = patch_duckdb_connect(monkeypatch, rows)

    result = confoundry.gather.assemble_timeseries_paths_from_db(
        database="/tmp/catalog.duckdb",
        name_map={"tas_raw": "temperature"},
    )

    assert connection.database == "/tmp/catalog.duckdb"
    assert "FROM geotiff_catalog" in connection.executed_sql

    assert result == {
        (2020, 1): {
            "temperature": raster,
        }
    }


def test_assemble_timeseries_paths_from_db_expands_yearly_files_to_all_months(
    tmp_path,
    monkeypatch,
):
    raster = tmp_path / "elevation_2020.tif"
    raster.touch()

    rows = [
        (
            "dem_raw",
            "yearly",
            str(tmp_path),
            "elevation_2020.tif",
            2020,
            None,
        )
    ]

    patch_duckdb_connect(monkeypatch, rows)

    result = confoundry.gather.assemble_timeseries_paths_from_db(
        database="/tmp/catalog.duckdb",
        name_map={"dem_raw": "elevation"},
    )

    assert set(result) == {(2020, month) for month in range(1, 13)}

    for month in range(1, 13):
        assert result[(2020, month)] == {
            "elevation": raster,
        }


def test_assemble_timeseries_paths_from_db_combines_monthly_and_yearly_files(
    tmp_path,
    monkeypatch,
):
    monthly_raster = tmp_path / "temperature_2020_03.tif"
    yearly_raster = tmp_path / "elevation_2020.tif"
    monthly_raster.touch()
    yearly_raster.touch()

    rows = [
        (
            "tas_raw",
            "monthly",
            str(tmp_path),
            "temperature_2020_03.tif",
            2020,
            3,
        ),
        (
            "dem_raw",
            "yearly",
            str(tmp_path),
            "elevation_2020.tif",
            2020,
            None,
        ),
    ]

    patch_duckdb_connect(monkeypatch, rows)

    result = confoundry.gather.assemble_timeseries_paths_from_db(
        database="/tmp/catalog.duckdb",
        name_map={
            "tas_raw": "temperature",
            "dem_raw": "elevation",
        },
    )

    assert result[(2020, 3)] == {
        "temperature": monthly_raster,
        "elevation": yearly_raster,
    }

    assert result[(2020, 1)] == {
        "elevation": yearly_raster,
    }


def test_assemble_timeseries_paths_from_db_skips_missing_files(
    tmp_path,
    monkeypatch,
):
    existing_raster = tmp_path / "temperature_2020_01.tif"
    existing_raster.touch()

    rows = [
        (
            "tas_raw",
            "monthly",
            str(tmp_path),
            "temperature_2020_01.tif",
            2020,
            1,
        ),
        (
            "pr_raw",
            "monthly",
            str(tmp_path),
            "missing_precipitation_2020_01.tif",
            2020,
            1,
        ),
    ]

    patch_duckdb_connect(monkeypatch, rows)

    result = confoundry.gather.assemble_timeseries_paths_from_db(
        database="/tmp/catalog.duckdb",
        name_map={
            "tas_raw": "temperature",
            "pr_raw": "precipitation",
        },
    )

    assert result == {
        (2020, 1): {
            "temperature": existing_raster,
        }
    }


def test_assemble_timeseries_paths_from_db_raises_for_unknown_frequency(
    tmp_path,
    monkeypatch,
):
    raster = tmp_path / "temperature_2020.tif"
    raster.touch()

    rows = [
        (
            "tas_raw",
            "daily",
            str(tmp_path),
            "temperature_2020.tif",
            2020,
            1,
        )
    ]

    patch_duckdb_connect(monkeypatch, rows)

    with pytest.raises(RuntimeError, match="Unknown frequency: daily"):
        confoundry.gather.assemble_timeseries_paths_from_db(
            database="/tmp/catalog.duckdb",
            name_map={"tas_raw": "temperature"},
        )


def test_assemble_timeseries_paths_from_db_raises_for_unmapped_variable(
    tmp_path,
    monkeypatch,
):
    raster = tmp_path / "temperature_2020_01.tif"
    raster.touch()

    rows = [
        (
            "tas_raw",
            "monthly",
            str(tmp_path),
            "temperature_2020_01.tif",
            2020,
            1,
        )
    ]

    patch_duckdb_connect(monkeypatch, rows)

    with pytest.raises(KeyError, match="tas_raw"):
        confoundry.gather.assemble_timeseries_paths_from_db(
            database="/tmp/catalog.duckdb",
            name_map={},
        )


def test_assemble_timeseries_paths_from_db_returns_empty_dict_when_no_rows(
    monkeypatch,
):
    patch_duckdb_connect(monkeypatch, rows=[])

    result = confoundry.gather.assemble_timeseries_paths_from_db(
        database="/tmp/catalog.duckdb",
        name_map={},
    )

    assert result == {}
