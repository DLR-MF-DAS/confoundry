import click
import json
import duckdb
import rasterio
import os
from rasterio.transform import xy, rowcol, from_origin
from rasterio.warp import transform as crs_transform
import pandas as pd
import numpy as np
from pathlib import Path
from dowhy import CausalModel
import glob
from tqdm.contrib.concurrent import process_map
import warnings


def map_pixel_to_all(row, col, ref, datasets, bounds_check=True):
    """
    Given (row, col) in ref_ds, return corresponding (row, col) in all datasets.

    Parameters
    ----------
    row, col : int
        Pixel indices in the reference dataset.
    ref : str
        Name of the reference dataset.
    datasets : dict of rasterio.io.DatasetReader
        Datasets covering (ideally) the same area and CRS.
    bounds_check : bool
        If True, returns None for rasters where the point is outside.

    Returns
    -------
    dict
        {dataset: (row, col) or None} for each dataset in [ref_ds] + other_datasets
        Keys can be paths or indices, depending on how you call it.
    """
    try:
        ref_ds = datasets[ref]
    except KeyError:
        return {}
    # 1) Reference pixel -> map coordinates
    x, y = xy(ref_ds.transform, row, col)  # center of pixel

    results = {}

    # Helper to do bounds checking
    def _safe_rowcol(ds, x, y):
        try:
            # reproject coordinate if needed
            if ds.crs is not None and ref_ds.crs is not None and ds.crs != ref_ds.crs:
                x2, y2 = crs_transform(ref_ds.crs, ds.crs, [x], [y])
                x_use, y_use = x2[0], y2[0]
            else:
                x_use, y_use = x, y
            r, c = rowcol(ds.transform, x_use, y_use)
        except Exception:
            return None

        if not bounds_check:
            return (r, c)

        # check within raster bounds
        if (0 <= r < ds.height) and (0 <= c < ds.width):
            return (r, c)
        else:
            return None

    # Include the reference raster as well (it should map back to the same or neighboring pixel)
    results[ref] = _safe_rowcol(ref_ds, x, y)
    for key in datasets:
        results[key] = _safe_rowcol(datasets[key], x, y)
    return results

def assemble_data_frame(task):
    """Construct a DataFrame from the separate rasterio data sources.

    Parameters
    ----------
    ref: str
        Name of the dataset to use as reference.
    dataset_files: dict of filenames
        A dictionary of named filenames.

    Returns
    -------
    pd.DataFrame
        An aggregate dataframe.
    """
    ref, dataset_files = task
    data = {}
    profiles = {}
    sources = {}
    for variable in dataset_files:
        src = rasterio.open(dataset_files[variable], 'r')
        data[variable] = src.read(1).astype("float64")
        if src.nodata is not None:
            data[variable][data[variable] == src.nodata] = np.nan
        profiles[variable] = dict(src.profile)
        sources[variable] = src
    res = map_pixel_to_all(0, 0, ref, sources)
    all_data = []
    try:
        for row in range(data[ref].shape[0]):
            for col in range(data[ref].shape[1]):
                indices = map_pixel_to_all(row, col, ref, sources)
                res_row = {}
                res_row['row'] = row
                res_row['col'] = col
                lat, lon = xy(sources[ref].transform, row, col)
                res_row['lat'] = lat
                res_row['lon'] = lon
                for s in sources:
                    if indices[s] is None:
                        res_row[s] = np.nan
                    else:
                        r, c = indices[s]
                        res_row[s] = data[s][r][c]
                all_data.append(res_row)
    except KeyError:
        pass
    for src in sources.values():
        src.close()
    return pd.DataFrame(all_data)

def assemble_timeseries(database, name_map, ref, max_workers=1):
    """
    Assemble a long-form time series table from a directory tree of rasters.

    This is a convenience wrapper around :func:`assemble_timeseries_paths`
    and :func:`assemble_data_frame`. It first builds, for each time step
    (e.g. each year/month directory), a dictionary that maps variable names
    to the corresponding raster file paths. For every such dictionary it
    then calls :func:`assemble_data_frame` and finally concatenates all
    per-timestep DataFrames row-wise.

    Parameters
    ----------
    database : str or pathlib.Path
        DuckDB database file.
    ref : str
        Name of the reference dataset passed through to
        :func:`assemble_data_frame`. This identifies which raster defines
        the pixel grid / coordinate system for the aggregation.

    Returns
    -------
    pandas.DataFrame
        A single DataFrame obtained by concatenating the per-timestep
        DataFrames returned by :func:`assemble_data_frame` for each
        year/month combination.

    Notes
    -----
    This function does not add an explicit time column. If time information
    is required, it either needs to be encoded inside the per-timestep
    DataFrames returned by :func:`assemble_data_frame` or added in a
    post-processing step based on the directory structure.
    """
    path_dict = assemble_timeseries_paths_from_db(database, name_map)
    result = []
    tasks = [(ref, path_dict[k]) for k in path_dict]
    result = process_map(assemble_data_frame, tasks, max_workers=max_workers, ascii=True)
    result = pd.concat(result)
    return result

def assemble_timeseries_paths_from_db(database, name_map):
    """
    Assemble per-month dataset file paths from a DuckDB database.

    Parameters
    ----------
    database : str or pathlib.Path
        Path to a DuckDB database.

    Returns
    -------
    list of dict
        A list of dictionaries. Each dictionary represents one (year, month)
        timestep and maps variable names to file paths (as strings).
    """
    conn = duckdb.connect(database)
    datasets = {}
    for row in conn.execute("SELECT variable_name, frequency, root_dir, file_name, year, month FROM geotiff_catalog").fetchall():
        variable_name, frequency, root_dir, file_name, year, month = row
        variable_name = name_map[variable_name]
        file_path = Path(os.path.join(root_dir, file_name))
        if not file_path.exists():
            continue
        if frequency == 'monthly':
            try:
                datasets[(year, month)][variable_name] = file_path
            except KeyError:
                datasets[(year, month)] = {variable_name: file_path}
        elif frequency == 'yearly':
            for m in range(1, 13):
                try:
                    datasets[(year, m)][variable_name] = file_path
                except KeyError:
                    datasets[(year, m)] = {variable_name: file_path}
        else:
            raise RuntimeError(f"Unknown frequency: {frequency}") 
    return datasets


@click.command()
@click.option('-i', '--input-db', help='Database')
@click.option('-n', '--name-map', help='Name map file')
@click.option('-o', '--output-file', help='Output filename')
@click.option('-t', '--table-name', help='Name of the table to create')
@click.option('-w', '--max-workers', help='Number of parallel processes', default=1)
def main(input_db, name_map, output_file, table_name, max_workers):
    with open(name_map, 'r') as fd:
        name_map = json.load(fd)
    df = assemble_timeseries(input_db, name_map, "ndvi", max_workers=max_workers)
    conn = duckdb.connect(output_file)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    conn.close()

if __name__ == '__main__':
    main()
