import os
import tempfile
import uuid
import json
import pytest
from pathlib import Path
from datetime import datetime

import drought_causality.download_to_db as d2db
from drought_causality.downloaders.downloader import ItemDownloadReport


TEST_START_DATE = "2014-01-01"
TEST_END_DATE = "2014-03-30"
TEST_REAL_DOWNLOADERS = ["spei", "era5"]
TEST_DUMMY_DOWNLOADERS = ["downloaderA", "downloaderB", "downloaderC"]


class GoodDummyDownloader:
		def __init__(self, cache_dir=None):
			pass
		def download(self, polygon, time_frame, output_dir, show_progress=True):
			return [
				ItemDownloadReport(
					data_source="dummygood",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=True,
					error=None,
					metadata={"test": True}
				),
				ItemDownloadReport(
					data_source="dummygood",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=True,
					error=None,
					metadata={"test": True}
				),
					ItemDownloadReport(
					data_source="dummygood",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=True,
					error=None,
					metadata={"test": True}
				)
			]
		
class BadDummyDownloader:
		def __init__(self, cache_dir=None):
			pass
		def download(self, polygon, time_frame, output_dir, show_progress=True):
			return [
				ItemDownloadReport(
					data_source="dummygood",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=True,
					error=None,
					metadata={"test": True}
				),
				ItemDownloadReport(
					data_source="dummybad",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=False,
					error="A failure occurred here.",
					metadata={"test": False}
				),
				ItemDownloadReport(
					data_source="dummygood",
					variable_name="dummy_var",
					acquisition_time=datetime(2014, 1, 1),
					path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
					download_successful=True,
					error=None,
					metadata={"test": True}
				),
			]
		
DUMMY_DOWNLOADERS_MAP = {
	"downloaderA": GoodDummyDownloader,
	"downloaderB": BadDummyDownloader,
	"downloaderC": GoodDummyDownloader
}
		

def make_test_geojson(tmp_path):
	geojson = {
		"type": "FeatureCollection",
		"features": [
			{
				"type": "Feature",
				"properties": {},
				"geometry": {
					"type": "Polygon",
					"coordinates": [[
						[-9.5, 36.0],
						[3.3, 36.0],
						[3.3, 43.8],
						[-9.5, 43.8],
						[-9.5, 36.0]
					]]
				}
			}
		]
	}
	geojson_path = tmp_path / "testloc.json"
	with open(geojson_path, "w") as f:
		json.dump(geojson, f)
	return geojson, geojson_path

def test_parse_and_validate_inputs(tmp_path):
	geojson, geojson_path = make_test_geojson(tmp_path)
	output_folder = str(tmp_path / "output")
	result = d2db.parse_and_validate_inputs(
		geojson_path=str(geojson_path),
		location_nickname=None,
		downloaders=TEST_REAL_DOWNLOADERS,
		start_date=TEST_START_DATE,
		end_date=TEST_END_DATE,
		output_folder=output_folder
	)
	start_date_dt, end_date_dt, downloaders_out, geojson_dict, polygon, location_nickname, cache_dir = result
	assert start_date_dt.strftime("%Y-%m-%d") == TEST_START_DATE
	assert end_date_dt.strftime("%Y-%m-%d") == TEST_END_DATE
	assert isinstance(geojson_dict, dict)
	assert polygon["type"] == "Polygon"
	assert location_nickname == "testloc"
	assert cache_dir.exists()

def test_setup_database(tmp_path):
	# Get dummy geojson and initialise database
	geojson, geojson_path = make_test_geojson(tmp_path)

	db_path = tmp_path / "testdb.duckdb"
	location_nickname = "testloc"
	db_connection, location_id = d2db.setup_database(str(db_path), location_nickname, geojson)

	# Check database is connected and location id is string
	assert db_connection is not None
	assert isinstance(location_id, str)

	# Check that core tables exist
	tables = set(row[0] for row in db_connection.execute("SHOW TABLES").fetchall())
	assert "locations" in tables
	assert "geotiff_catalog" in tables


def test_downloading_pipeline(tmp_path, monkeypatch):
	# Get dummy geojson and initialise database
	geojson, geojson_path = make_test_geojson(tmp_path)
	db_path = tmp_path / "testdb.duckdb"
	output_folder = tmp_path / "output"
	location_nickname = "testloc"
	start_date_dt = datetime.strptime(TEST_START_DATE, "%Y-%m-%d")
	end_date_dt = datetime.strptime(TEST_END_DATE, "%Y-%m-%d")
	cache_dir = output_folder / location_nickname / "cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	polygon = geojson["features"][0]["geometry"]

	db_connection, location_id = d2db.setup_database(str(db_path), location_nickname, geojson)

	# Patch DOWNLOADERS_MAP to use dummy downloaders
	monkeypatch.setattr(d2db, "DOWNLOADERS_MAP", DUMMY_DOWNLOADERS_MAP)
	d2db.run_downloading_pipeline(
		downloaders=TEST_DUMMY_DOWNLOADERS,
		polygon=polygon,
		start_date_dt=start_date_dt,
		end_date_dt=end_date_dt,
		location_id=location_id,
		location_nickname=location_nickname,
		database_connection=db_connection,
		cache_dir=cache_dir,
		output_folder=str(output_folder)
	)
	# Check that DB has at least one entry for this location
	rows = db_connection.execute("SELECT download_status, data_source FROM geotiff_catalog WHERE location_id=?", [location_id]).fetchall()
	assert rows
	assert len(rows) == 9

	# Check expected statuses: downloaderA and downloaderC are all success, downloaderB has one failed
	statuses = [row[0] for row in rows]
	sources = [row[1] for row in rows]

	# downloaderA: 3 success, downloaderB: 2 success, 1 failed, downloaderC: 3 success
	assert statuses.count("success") == 8
	assert statuses.count("failed") == 1
	
	# The failed one should be from data_source 'dummybad'
	failed_indices = [i for i, s in enumerate(statuses) if s == "failed"]
	assert len(failed_indices) == 1
	assert sources[failed_indices[0]] == "dummybad"
