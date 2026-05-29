import uuid
import json
import pytest
from pathlib import Path
from datetime import datetime

import confoundry.download_to_db as d2db
from fetcheo.loader import FetchEOLoader

TEST_START_DATE = "2014-01-01"
TEST_END_DATE = "2014-03-30"
TEST_DUMMY_DOWNLOADERS = ["downloaderA", "downloaderB"]
class GoodDummyDownloader:
	def __init__(self, cache_dir=None):
		pass
	@property
	def frequency(self):
		return "monthly"
	def download(self, polygon, time_frame, output_dir, show_progress=True):
		from fetcheo.loader import ItemDownloadReport
		return [
			ItemDownloadReport(
				data_source="dummygood",
				variable_name="dummy_var1",
				acquisition_time=datetime(2014, 1, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=True,
				error=None,
				metadata={"test": True}
			),
			ItemDownloadReport(
				data_source="dummygood",
				variable_name="dummy_var2",
				acquisition_time=datetime(2014, 2, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=True,
				error=None,
				metadata={"test": True}
			),
			ItemDownloadReport(
				data_source="dummygood",
				variable_name="dummy_var3",
				acquisition_time=datetime(2014, 3, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=True,
				error=None,
				metadata={"test": True}
			)
	]

class BadDummyDownloader:
    def __init__(self, cache_dir=None):
		pass
	@property
	def frequency(self):
		return "monthly"
	def download(self, polygon, time_frame, output_dir, show_progress=True):
		from fetcheo.loader import ItemDownloadReport
		return [
			ItemDownloadReport(
				data_source="dummybad",
				variable_name="dummy_var1",
				acquisition_time=datetime(2014, 1, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=True,
				error=None,
				metadata={"test": True}
			),
			ItemDownloadReport(
				data_source="dummybad",
				variable_name="dummy_var2",
				acquisition_time=datetime(2014, 2, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=False,
				error="A failure occurred here.",
				metadata={"test": False}
			),
			ItemDownloadReport(
				data_source="dummybad",
				variable_name="dummy_var3",
				acquisition_time=datetime(2014, 3, 1),
				path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
				download_successful=True,
				error=None,
				metadata={"test": True}
			),
		]

DUMMY_DOWNLOADERS_MAP = {
	"downloaderA": GoodDummyDownloader,
	"downloaderB": BadDummyDownloader
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
	db_path = str(tmp_path / "testdb.duckdb")
	config_dict = {
		"geojson": str(geojson_path),
	"name": None,
	"downloaders": {d: {"enabled": True} for d in TEST_DUMMY_DOWNLOADERS},
	"start-date": TEST_START_DATE,
	"end-date": TEST_END_DATE,
	"output_folder": output_folder,
	"db_path": db_path
	}
	result = d2db.parse_and_validate_inputs(config_dict)
	start_date_dt, end_date_dt, geojson_dict, polygon, location_nickname, db_path_out, cache_dir, downloader_config, downloader_kwargs = result
	assert start_date_dt.strftime("%Y-%m-%d") == TEST_START_DATE
	assert end_date_dt.strftime("%Y-%m-%d") == TEST_END_DATE
	assert isinstance(geojson_dict, dict)
	assert polygon["type"] == "Polygon"
	assert location_nickname == "testloc"
	assert cache_dir.exists()
	assert db_path_out == Path(db_path)
	assert set(downloader_config.keys()) == set(TEST_DUMMY_DOWNLOADERS)

def test_fetcheo_loader_with_dummy_downloaders(tmp_path, monkeypatch):
	geojson, geojson_path = make_test_geojson(tmp_path)
	output_folder = tmp_path / "output"
	db_path = tmp_path / "testdb.duckdb"
	location_nickname = "testloc"
	config_dict = {
		"geojson": str(geojson_path),
		"name": location_nickname,
	"downloaders": {d: {"enabled": True} for d in TEST_DUMMY_DOWNLOADERS},
	"start-date": TEST_START_DATE,
	"end-date": TEST_END_DATE,
		"output_folder": str(output_folder),
		"db_path": str(db_path)
	}
	# Patch fetcheo.loader.DOWNLOADER_DICT to use dummy downloaders
	monkeypatch.setattr("fetcheo.loader.DOWNLOADER_DICT", DUMMY_DOWNLOADERS_MAP)

	# Parse config and create loader
	(
	start_dt,
	end_dt,
	geojson_dict,
	polygon,
	location_nickname,
	db_path_out,
	cache_dir,
	downloader_config,
	downloader_kwargs
	) = d2db.parse_and_validate_inputs(config_dict)

	loader = FetchEOLoader(
		downloader_config=downloader_config,
		downloader_kwargs=downloader_kwargs,
		db_path=db_path_out
	)

	data_output_dir = output_folder / "data" / location_nickname
	data_output_dir.mkdir(parents=True, exist_ok=True)

	loader.fetch(
		polygon=polygon,
		time_frame=(start_dt, end_dt),
		location_nickname=location_nickname,
		output_dir=str(data_output_dir),
		show_progress=False,
	)

	# Check that DB has at least one entry for this location
	import duckdb
	con = duckdb.connect(str(db_path_out))
	rows = con.execute("SELECT download_status, data_source FROM geotiff_catalog").fetchall()
	assert rows
	assert len(rows) == 6
	statuses = [row[0] for row in rows]
	sources = [row[1] for row in rows]
	assert statuses.count("success") == 5
	assert statuses.count("failed") == 1
	failed_indices = [i for i, s in enumerate(statuses) if s == "failed"]
	assert len(failed_indices) == 1
	assert sources[failed_indices[0]] == "dummybad"
import uuid
import json
import pytest
from pathlib import Path
from datetime import datetime

import confoundry.download_to_db as d2db
from confoundry.downloaders.downloader import ItemDownloadReport

				"properties": {},
	"downloaderB": BadDummyDownloader,
		json.dump(geojson, f)
	assert db_path_out == Path(db_path)

		import uuid
		import json
		import pytest
		from pathlib import Path
		from datetime import datetime

		import confoundry.download_to_db as d2db
		from fetcheo.loader import FetchEOLoader

		TEST_START_DATE = "2014-01-01"
		TEST_END_DATE = "2014-03-30"
		TEST_DUMMY_DOWNLOADERS = ["downloaderA", "downloaderB"]

		class GoodDummyDownloader:
			def __init__(self, cache_dir=None):
				pass
			@property
			def frequency(self):
				return "monthly"
			def download(self, polygon, time_frame, output_dir, show_progress=True):
				from fetcheo.loader import ItemDownloadReport
			with open(geojson_path, "w") as f:
			assert location_nickname == "testloc"
				start_dt,

				import uuid
				import json
				import pytest
				from pathlib import Path
				from datetime import datetime

				import confoundry.download_to_db as d2db
				from fetcheo.loader import FetchEOLoader

				TEST_START_DATE = "2014-01-01"
				TEST_END_DATE = "2014-03-30"
				TEST_DUMMY_DOWNLOADERS = ["downloaderA", "downloaderB"]

				class GoodDummyDownloader:
					def __init__(self, cache_dir=None):
						pass
					@property
					def frequency(self):
						return "monthly"
					def download(self, polygon, time_frame, output_dir, show_progress=True):
						from fetcheo.loader import ItemDownloadReport
						return [
							ItemDownloadReport(
								data_source="dummygood",
								variable_name="dummy_var1",
								acquisition_time=datetime(2014, 1, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=True,
								error=None,
								metadata={"test": True}
							),
							ItemDownloadReport(
								data_source="dummygood",
								variable_name="dummy_var2",
								acquisition_time=datetime(2014, 2, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=True,
								error=None,
								metadata={"test": True}
							),
							ItemDownloadReport(
								data_source="dummygood",
								variable_name="dummy_var3",
								acquisition_time=datetime(2014, 3, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=True,
								error=None,
								metadata={"test": True}
							)
						]

				class BadDummyDownloader:
					def __init__(self, cache_dir=None):
						pass
					@property
					def frequency(self):
						return "monthly"
					def download(self, polygon, time_frame, output_dir, show_progress=True):
						from fetcheo.loader import ItemDownloadReport
						return [
							ItemDownloadReport(
								data_source="dummybad",
								variable_name="dummy_var1",
								acquisition_time=datetime(2014, 1, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=True,
								error=None,
								metadata={"test": True}
							),
							ItemDownloadReport(
								data_source="dummybad",
								variable_name="dummy_var2",
								acquisition_time=datetime(2014, 2, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=False,
								error="A failure occurred here.",
								metadata={"test": False}
							),
							ItemDownloadReport(
								data_source="dummybad",
								variable_name="dummy_var3",
								acquisition_time=datetime(2014, 3, 1),
								path=Path(f"/tmp/dummy_{uuid.uuid4()}.tif"),
								download_successful=True,
								error=None,
								metadata={"test": True}
							),
						]

				DUMMY_DOWNLOADERS_MAP = {
					"downloaderA": GoodDummyDownloader,
					"downloaderB": BadDummyDownloader
				}

		downloader_kwargs
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

	) = d2db.parse_and_validate_inputs(config_dict)
					geojson, geojson_path = make_test_geojson(tmp_path)
					output_folder = str(tmp_path / "output")
					db_path = str(tmp_path / "testdb.duckdb")
					config_dict = {
						"geojson": str(geojson_path),
						"name": None,
						"downloaders": {d: {"enabled": True} for d in TEST_DUMMY_DOWNLOADERS},
						"start-date": TEST_START_DATE,
						"end-date": TEST_END_DATE,
						"output_folder": output_folder,
						"db_path": db_path
					}
					result = d2db.parse_and_validate_inputs(config_dict)
					start_date_dt, end_date_dt, geojson_dict, polygon, location_nickname, db_path_out, cache_dir, downloader_config, downloader_kwargs = result
					assert start_date_dt.strftime("%Y-%m-%d") == TEST_START_DATE
					assert end_date_dt.strftime("%Y-%m-%d") == TEST_END_DATE
					assert isinstance(geojson_dict, dict)
					assert polygon["type"] == "Polygon"
					assert location_nickname == "testloc"
					assert cache_dir.exists()
					assert db_path_out == Path(db_path)
					assert set(downloader_config.keys()) == set(TEST_DUMMY_DOWNLOADERS)


					geojson, geojson_path = make_test_geojson(tmp_path)
					output_folder = tmp_path / "output"
					db_path = tmp_path / "testdb.duckdb"
					location_nickname = "testloc"
					config_dict = {
						"geojson": str(geojson_path),
						"name": location_nickname,
						"downloaders": {d: {"enabled": True} for d in TEST_DUMMY_DOWNLOADERS},
						"start-date": TEST_START_DATE,
						"end-date": TEST_END_DATE,
						"output_folder": str(output_folder),
						"db_path": str(db_path)
					}
					# Patch fetcheo.loader.DOWNLOADER_DICT to use dummy downloaders
					monkeypatch.setattr("fetcheo.loader.DOWNLOADER_DICT", DUMMY_DOWNLOADERS_MAP)

					# Parse config and create loader
					(
						start_dt,
						end_dt,
						geojson_dict,
						polygon,
						location_nickname,
						db_path_out,
						cache_dir,
						downloader_config,
						downloader_kwargs
					) = d2db.parse_and_validate_inputs(config_dict)

					loader = FetchEOLoader(
						downloader_config=downloader_config,
						downloader_kwargs=downloader_kwargs,
						db_path=db_path_out
					)

					data_output_dir = output_folder / "data" / location_nickname
					data_output_dir.mkdir(parents=True, exist_ok=True)

					loader.fetch(
						polygon=polygon,
						time_frame=(start_dt, end_dt),
						location_nickname=location_nickname,
						output_dir=str(data_output_dir),
						show_progress=False,
					)

					# Check that DB has at least one entry for this location
					import duckdb
					con = duckdb.connect(str(db_path_out))
					rows = con.execute("SELECT download_status, data_source FROM geotiff_catalog").fetchall()
					assert rows
					assert len(rows) == 6
					statuses = [row[0] for row in rows]
					sources = [row[1] for row in rows]
					assert statuses.count("success") == 5
					assert statuses.count("failed") == 1
					failed_indices = [i for i, s in enumerate(statuses) if s == "failed"]
					assert len(failed_indices) == 1
					assert sources[failed_indices[0]] == "dummybad"
	loader = FetchEOLoader(
		downloader_config=downloader_config,
		downloader_kwargs=downloader_kwargs,
		db_path=db_path_out
	)

	data_output_dir = output_folder / "data" / location_nickname
	data_output_dir.mkdir(parents=True, exist_ok=True)

	loader.fetch(
		polygon=polygon,
		time_frame=(start_dt, end_dt),
		location_nickname=location_nickname,
		output_dir=str(data_output_dir),
		show_progress=False,
	)

	# Check that DB has at least one entry for this location
	import duckdb
	con = duckdb.connect(str(db_path_out))
	rows = con.execute("SELECT download_status, data_source FROM geotiff_catalog").fetchall()
	assert rows
	assert len(rows) == 6
	statuses = [row[0] for row in rows]
	sources = [row[1] for row in rows]
	assert statuses.count("success") == 5
	assert statuses.count("failed") == 1
	failed_indices = [i for i, s in enumerate(statuses) if s == "failed"]
	assert len(failed_indices) == 1
	assert sources[failed_indices[0]] == "dummybad"
