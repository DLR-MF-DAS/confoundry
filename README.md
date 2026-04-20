# Confoundry

A Causal Inference framework for Earth Observation applications.

# How to Download the Data
To download geospatial time series datasets for a specific region, use the provided command-line interface (CLI):

1. **Prepare a GeoJSON file**  
	Create or obtain a GeoJSON file that defines the polygon of your region of interest (e.g., `data/california.json`).

2. **Activate your Python environment**  
	Make sure your environment is activated:
	```bash
	source .venv/bin/activate
	```

3. **Run the download command**  
	Use the following command to download data and populate the database:
	```bash
	python -m drought_causality.download_to_db \
	  --geojson_path data/california.json \
	  --db_path confoundry_db.duckdb \
	  --start_date 2014-01-01 \
	  --end_date 2014-03-31
	```
	- `--geojson_path`: Path to your GeoJSON file.
	- `--db_path`: Path to the DuckDB database file (will be created if it doesn't exist).
	- `--start_date` / `--end_date`: Date range for data download (format: YYYY-MM-DD).
	- `--location_nickname`: (Optional) Custom name for the location. Otherwise, the filename of the geojson is used by default.
	- `--downloaders`: (Optional, repeatable) Specify which data sources to download (e.g., `spei`, `era5`). If omitted, all available downloaders are used.

	Example with specific downloaders:
	```bash
	python -m drought_causality.download_to_db \
	  --geojson_path data/california.json \
	  --downloaders spei --downloaders era5
	```

4. **Output**  
	Downloaded files will be saved under `data/<location_nickname>/`, and all metadata will be recorded in the DuckDB database.


# Cite this Work (TBC)