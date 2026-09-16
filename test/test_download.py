import json
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import duckdb
import yaml
from click.testing import CliRunner

import confoundry.download_to_db as d2db


TEST_START_DATE = "2014-01-01"
TEST_END_DATE = "2014-03-30"
TEST_DUMMY_DOWNLOADERS = ["era5", "modis_ndvi"]


def make_test_geojson(tmp_path: Path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-9.25, 38.80],
                        [-9.25, 38.65],
                        [-9.05, 38.65],
                        [-9.05, 38.80],
                        [-9.25, 38.80],
                    ]],
                },
            }
        ],
    }
    geojson_path = tmp_path / "testloc.json"
    with open(geojson_path, "w") as file_handle:
        json.dump(geojson, file_handle)
    return geojson, geojson_path


def make_config(tmp_path: Path, geojson_path: Path, name=None, show_progress=True):
    return {
        "geojson_path": str(geojson_path),
        "name": name,
        "downloaders": {
            downloader: {"enabled": True}
            for downloader in TEST_DUMMY_DOWNLOADERS
        },
        "start_date": TEST_START_DATE,
        "end_date": TEST_END_DATE,
        "output_folder": str(tmp_path / "output"),
        "db_path": str(tmp_path / "testdb.duckdb"),
        "show_progress": show_progress,
    }


def test_parse_and_validate_inputs(tmp_path):
    geojson, geojson_path = make_test_geojson(tmp_path)
    config_dict = make_config(tmp_path, geojson_path)

    result = d2db.parse_and_validate_inputs(config_dict)

    (
        start_date_dt,
        end_date_dt,
        geojson_dict,
        polygon,
        location_nickname,
        db_path_out,
        cache_dir,
        downloader_config,
        downloader_kwargs,
    ) = result

    assert start_date_dt.strftime("%Y-%m-%d") == TEST_START_DATE
    assert end_date_dt.strftime("%Y-%m-%d") == TEST_END_DATE
    assert geojson_dict == geojson
    assert polygon == geojson["features"][0]["geometry"]
    assert location_nickname == "testloc"
    assert cache_dir == Path(config_dict["output_folder"]) / location_nickname / "cache"
    assert cache_dir.exists()
    assert db_path_out == Path(config_dict["db_path"])
    assert downloader_config == {name: True for name in TEST_DUMMY_DOWNLOADERS}
    assert downloader_kwargs == {name: {} for name in TEST_DUMMY_DOWNLOADERS}


def test_main_wires_loader_without_running_fetcheo(tmp_path):
    geojson, geojson_path = make_test_geojson(tmp_path)
    config_dict = make_config(
        tmp_path,
        geojson_path,
        name="coastal_test",
        show_progress=False,
    )
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as file_handle:
        yaml.safe_dump(config_dict, file_handle)

    runner = CliRunner()
    loader_instance = MagicMock()
    loader_instance.downloaders = {}
    loader_instance.fetch.return_value = [
        SimpleNamespace(
            variable_name="temperature",
            frequency="monthly",
            download_successful=True,
        )
    ]

    with patch.object(d2db, "FetchEOLoader", return_value=loader_instance) as mock_loader_class:
        result = runner.invoke(d2db.main, ["--config-path", str(config_path)])

    assert result.exit_code == 0, result.output
    mock_loader_class.assert_called_once_with(
        downloader_config={name: True for name in TEST_DUMMY_DOWNLOADERS},
        downloader_kwargs={name: {} for name in TEST_DUMMY_DOWNLOADERS},
        db_path=Path(config_dict["db_path"]),
    )
    loader_instance.fetch.assert_called_once_with(
        polygon=geojson["features"][0]["geometry"],
        time_frame=(
            d2db.ensure_datetime(TEST_START_DATE),
            d2db.ensure_datetime(TEST_END_DATE),
        ),
        location_nickname="coastal_test",
        output_dir=str(Path(config_dict["output_folder"]) / "coastal_test"),
        show_progress=False,
    )
    assert "temperature [monthly]: success=1, failed=0" in result.output
    assert "testdb.duckdb::geotiff_catalog" in result.output


def test_attach_report_metadata_uses_downloader_metadata():
    class FakeDownloader:
        frequency = "monthly"
        variables_dict = {"t2m": "2m_temperature"}

        def fetch(self):
            return [
                SimpleNamespace(variable_name="t2m"),
                SimpleNamespace(variable_name="ndvi", frequency="monthly"),
            ]

    downloader = FakeDownloader()
    loader = SimpleNamespace(downloaders={"fake": downloader})

    d2db.attach_report_metadata(loader)
    reports = downloader.fetch()

    assert [report.frequency for report in reports] == ["monthly", "monthly"]
    assert [report.variable_name for report in reports] == [
        "2m_temperature",
        "ndvi",
    ]


def test_attached_metadata_is_written_to_fetcheo_catalog(tmp_path):
    raster_path = tmp_path / "era5_202001_t2m.tif"
    raster_path.write_bytes(b"test raster placeholder")

    class FakeDownloader:
        frequency = "monthly"
        variables_dict = {"t2m": "2m_temperature"}

        def fetch(self, polygon, time_frame, output_dir, **kwargs):
            return [
                SimpleNamespace(
                    data_source="era5",
                    variable_name="t2m",
                    acquisition_time=datetime.datetime(2020, 1, 1),
                    path=raster_path,
                    download_successful=True,
                    error=None,
                    metadata=None,
                )
            ]

    database_path = tmp_path / "source.duckdb"
    loader = d2db.FetchEOLoader(
        downloader_config={},
        db_path=database_path,
    )
    loader.downloaders = {"fake": FakeDownloader()}
    d2db.attach_report_metadata(loader)
    loader.fetch(
        polygon={
            "type": "Polygon",
            "coordinates": [[
                [-9.25, 38.80],
                [-9.25, 38.65],
                [-9.05, 38.65],
                [-9.05, 38.80],
                [-9.25, 38.80],
            ]],
        },
        time_frame=(
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 1),
        ),
        location_nickname="test",
        output_dir=tmp_path / "output",
        show_progress=False,
    )

    con = duckdb.connect(str(database_path), read_only=True)
    row = con.execute(
        "SELECT variable_name, frequency, download_status "
        "FROM geotiff_catalog"
    ).fetchone()
    con.close()
    assert row == ("2m_temperature", "monthly", "success")


def test_iberian_download_config_matches_gather_name_map():
    repository_root = Path(__file__).resolve().parents[1]
    experiment_dir = repository_root / "data" / "iberian_drought_experiment"
    with (experiment_dir / "download.yaml").open() as handle:
        download_config = yaml.safe_load(handle)
    with (experiment_dir / "experiment.yaml").open() as handle:
        experiment_config = yaml.safe_load(handle)

    assert download_config["name"] == experiment_config["name"]
    assert download_config["start_date"].isoformat() == "2005-01-01"
    assert download_config["end_date"].isoformat() == "2024-12-31"
    assert download_config["downloaders"]["modis_ndvi"]["enabled"]

    era5 = download_config["downloaders"]["era5"]
    assert era5["enabled"]
    downloaded_names = set(era5["variables_dict"].values())
    downloaded_names.add("modis_ndvi")
    assert downloaded_names == set(experiment_config["name_map"])
