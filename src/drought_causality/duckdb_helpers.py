import json
import uuid
import duckdb


def connect_to_db(db_path: str = "db.duckdb"):
    """Connect to DuckDB database and return the connection object."""
    return duckdb.connect(db_path)


def initialise_tables(db_connection):
    """Initialise database tables if they don't exist."""
    db_connection.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id TEXT PRIMARY KEY,
            location_nickname TEXT,
            geojson JSON,
            first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db_connection.execute("""
        CREATE TABLE IF NOT EXISTS geotiff_catalog (
            catalog_id TEXT PRIMARY KEY,
            location_id TEXT,
            location_nickname TEXT,
            data_source TEXT,
            variable_name TEXT,  
            year INT,
            month INT,
            root_dir TEXT,
            file_name TEXT,
            file_size_bytes INT,
            download_status TEXT,
            error_message TEXT,
            first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT geotiff_unique UNIQUE (location_id, data_source, year, month, file_name)
        )
    """)


def upsert_file(
        db_connection,
        location_id,
        location_nickname,
        data_source,
        year,
        month,
        root_dir,
        file_name,
        file_size_bytes,
        download_status,
        error
    ):
    new_catalog_id = str(uuid.uuid4())
    db_connection.execute("""
        INSERT INTO geotiff_catalog (
            catalog_id,
            location_id,
            location_nickname,
            data_source,
            variable_name,  
            year,
            month,
            root_dir,
            file_name,
            file_size_bytes,
            download_status,
            error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(location_id, data_source, year, month, file_name) DO UPDATE SET
            catalog_id=excluded.catalog_id,
            location_id=excluded.location_id,
            location_nickname=excluded.location_nickname,
            last_updated=now()
    """,
    [new_catalog_id,
     location_id,
     location_nickname,
     data_source,
     year,
     month,
     root_dir,
     file_name,
     file_size_bytes,
     download_status,
     error])


def upsert_location(db_connection, location_nickname, geojson):
    """Insert or check a location by nickname and geojson."""
    row = db_connection.execute(
        "SELECT location_id, geojson FROM locations WHERE location_nickname = ?",
        [location_nickname]
    ).fetchone()
    geojson_str = json.dumps(geojson)
    if row:
        existing_id, existing_geojson = row
        if existing_geojson == geojson_str:
            # Same geojson, skip insert
            return existing_id
        else:
            # Different geojson for same nickname, raise error
            raise ValueError(f"Location nickname '{location_nickname}' already exists with different geojson.")
    # Insert new location
    new_location_id = str(uuid.uuid4())
    db_connection.execute(
        "INSERT INTO locations (location_id, location_nickname, geojson) VALUES (?, ?, ?)",
        [new_location_id, location_nickname, geojson_str]
    )
    return new_location_id


def test_db(db_connection):
    print(db_connection.execute("SELECT * FROM locations").df())
    print(db_connection.execute("SELECT * FROM geotiff_catalog").df().head(25))
