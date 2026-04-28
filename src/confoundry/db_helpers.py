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
            first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location_nickname TEXT UNIQUE,
            geojson JSON
        )
    """)
    db_connection.execute("""
        CREATE TABLE IF NOT EXISTS geotiff_catalog (
            catalog_id TEXT PRIMARY KEY,
            first_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location_id TEXT,
            location_nickname TEXT,
            data_source TEXT,
            variable_name TEXT,  
            frequency TEXT,
            year INT,
            month INT,
            root_dir TEXT,
            file_name TEXT,
            file_size_bytes INT,
            download_status TEXT,
            error_message TEXT,
            metadata JSON,
            CONSTRAINT geotiff_unique UNIQUE (location_id, data_source, variable_name, frequency, year, month)
        )
    """)


def fetch_or_create_location_id(db_connection, location_nickname, geojson):
    """
    Fetch a location ID or insert if new.
    This allows users to reuse geojsons with different location names for multiple experiments.
    """
    # Search for existing location in database
    geojson_str = json.dumps(geojson)
    row = db_connection.execute(
        "SELECT location_id, geojson FROM locations WHERE location_nickname = ?",
        [location_nickname]
    ).fetchone()

    # If found, verify geojson matches
    if row:
        existing_id, existing_geojson = row
        if existing_geojson == geojson_str:
            return existing_id
        else:
            raise ValueError(f"Location nickname '{location_nickname}' already exists with a different geojson.")
    
    # If no record, insert new location with the unique nickname
    else:
        new_location_id = str(uuid.uuid4())
        db_connection.execute(
            "INSERT INTO locations (location_id, location_nickname, geojson) VALUES (?, ?, ?)",
            [new_location_id, location_nickname, geojson_str]
        )
        return new_location_id


def upsert_file(
        db_connection,
        location_id,
        location_nickname,
        data_source,
        variable_name,
        frequency,
        year,
        month,
        root_dir,
        file_name,
        file_size_bytes,
        download_status,
        error_message,
        metadata=None
    ):
    new_catalog_id = str(uuid.uuid4())
    db_connection.execute("""
        INSERT INTO geotiff_catalog (
            catalog_id,
            location_id,
            location_nickname,
            data_source,
            variable_name,  
            frequency,
            year,
            month,
            root_dir,
            file_name,
            file_size_bytes,
            download_status,
            error_message,
            metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(location_id, data_source, variable_name, frequency, year, month) DO UPDATE SET
            root_dir=excluded.root_dir,
            file_name=excluded.file_name,
            file_size_bytes=excluded.file_size_bytes,
            download_status=excluded.download_status,
            error_message=excluded.error_message,
            metadata=excluded.metadata,
            last_updated=now()
    """,
    [new_catalog_id,
     location_id,
     location_nickname,
     data_source,
     variable_name,
     frequency,
     year,
     month,
     root_dir,
     file_name,
     file_size_bytes,
     download_status,
     error_message,
     metadata])
