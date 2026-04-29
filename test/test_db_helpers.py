import json
import pytest
import numpy as np

from confoundry.db_helpers import (
    connect_to_db, initialise_tables, upsert_file, fetch_or_create_location_id
)


def test_initialise_tables_creates_tables(tmp_path):
    db_path = tmp_path / "test_db.duckdb"
    db_connection = connect_to_db(str(db_path))
    initialise_tables(db_connection)
    tables = set(row[0] for row in db_connection.execute("SHOW TABLES").fetchall())
    assert 'locations' in tables
    assert 'geotiff_catalog' in tables


def test_upsert_location_inserts_and_returns_id(tmp_path):
    db_path = tmp_path / "test_db.duckdb"
    db_connection = connect_to_db(str(db_path))
    initialise_tables(db_connection)

    # Insert dummy location and get id
    geojson = {"type": "Point", "coordinates": [0, 0]}
    loc_id1 = fetch_or_create_location_id(db_connection, "testloc", geojson)
    assert isinstance(loc_id1, str)

    # Check that using the same nickname and geojson returns same id
    loc_id2 = fetch_or_create_location_id(db_connection, "testloc", geojson)
    assert loc_id1 == loc_id2


def test_upsert_location_different_geojson_raises(tmp_path):
    db_path = tmp_path / "test_db.duckdb"
    db_connection = connect_to_db(str(db_path))
    initialise_tables(db_connection)

    # Insert dummy location and get id
    geojson1 = {"type": "Point", "coordinates": [0, 0]}
    fetch_or_create_location_id(db_connection, "testloc", geojson1)

    # Check that using same nickname with different geojson raises error
    geojson2 = {"type": "Point", "coordinates": [1, 1]}
    fetch_or_create_location_id(db_connection, "testloc", geojson1)
    with pytest.raises(ValueError):
        fetch_or_create_location_id(db_connection, "testloc", geojson2)


def test_upsert_file_inserts_and_updates(tmp_path):
    db_path = tmp_path / "test_db.duckdb"
    db_connection = connect_to_db(str(db_path))
    initialise_tables(db_connection)

    # Insert dummy location
    geojson = {"type": "Point", "coordinates": [0, 0]}
    loc_id = fetch_or_create_location_id(db_connection, "testloc", geojson)

    # Insert dummy file record
    upsert_file(
        db_connection=db_connection,
        location_id=loc_id,
        location_nickname="testloc",
        data_source="testsrc",
        variable_name="var1",
        frequency="monthly",
        year=2020,
        month=1,
        root_dir="/tmp",
        file_name="file1.tif",
        file_size_bytes=123,
        download_status="failed",
        error_message="uh-oh",
        metadata=json.dumps({"foo": "bar"})
    )

    # Upsert (should not duplicate)
    upsert_file(
        db_connection=db_connection,
        location_id=loc_id,
        location_nickname="testloc",
        data_source="testsrc",
        variable_name="var1",
        frequency="monthly",
        year=2020,
        month=1,
        root_dir="/tmp",
        file_name="file1.tif",
        file_size_bytes=456,
        download_status="success",
        error_message=None,
        metadata=json.dumps({"foo": "baz"})
    )
    df = db_connection.execute("SELECT * FROM geotiff_catalog WHERE location_id=?", [loc_id]).df()
    assert len(df) == 1
    assert df.iloc[0]["file_size_bytes"] == np.int32(456)
    assert df.iloc[0]["metadata"] == json.dumps({"foo": "baz"})


def test_upsert_file_metadata_and_error(tmp_path):
    db_path = tmp_path / "test_db.duckdb"
    db_connection = connect_to_db(str(db_path))
    initialise_tables(db_connection)

    # Insert dummy location
    geojson = {"type": "Point", "coordinates": [0, 0]}
    loc_id = fetch_or_create_location_id(db_connection, "testloc", geojson)

    # Insert dummy file record with error and metadata
    upsert_file(
        db_connection=db_connection,
        location_id=loc_id,
        location_nickname="testloc",
        data_source="testsrc",
        variable_name="var1",
        frequency="monthly",
        year=2020,
        month=1,
        root_dir="/tmp",
        file_name="file1.tif",
        file_size_bytes=123,
        download_status="failed",
        error_message="Some error",
        metadata=json.dumps({"foo": "bar"})
    )
    df = db_connection.execute("SELECT * FROM geotiff_catalog WHERE location_id=?", [loc_id]).df()
    assert df.iloc[0]["download_status"] == "failed"
    assert df.iloc[0]["error_message"] == "Some error"
    assert json.loads(df.iloc[0]["metadata"])['foo'] == "bar"
