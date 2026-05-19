import click
import json
import yaml
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
    Map a pixel index from a reference raster to corresponding pixel indices
    in multiple raster datasets.

    The input pixel ``(row, col)`` from the reference dataset is converted
    to map coordinates using the center of the pixel, then transformed into
    the coordinate reference system (CRS) of each target dataset if needed.
    The corresponding pixel indices are computed for every dataset.

    Parameters
    ----------
    row : int
        Row index in the reference dataset.
    col : int
        Column index in the reference dataset.
    ref : hashable
        Key identifying the reference dataset in ``datasets``.
    datasets : dict
        Mapping of dataset keys to ``rasterio.io.DatasetReader``-like objects.
        Each dataset must provide at least:

        - ``transform``
        - ``crs``
        - ``height``
        - ``width``

    bounds_check : bool, default=True
        If ``True``, return ``None`` for datasets where the mapped pixel lies
        outside raster bounds. If ``False``, return computed indices even if
        they fall outside the raster extent.

    Returns
    -------
    dict
        Dictionary mapping each dataset key to either:

        - ``(row, col)`` tuple of mapped pixel indices
        - ``None`` if the coordinate cannot be transformed or falls outside
          raster bounds when ``bounds_check=True``

        If ``ref`` is not present in ``datasets``, an empty dictionary is
        returned.

    Notes
    -----
    - Pixel coordinates are computed using pixel centers via
      ``rasterio.transform.xy``.
    - CRS reprojection is performed with ``rasterio.warp.transform`` when
      source and destination CRS differ.
    - The reference dataset itself is included in the output mapping.
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
    """
    Assemble a tabular dataset from multiple raster datasets aligned to a
    reference raster grid.

    For each pixel in the reference raster, the corresponding pixel location
    is determined in every other raster using ``map_pixel_to_all``. Raster
    values are extracted and combined into a single row of a pandas
    ``DataFrame``.

    Raster values are converted to ``float64`` and adjusted using dataset
    scale and offset metadata when available. Pixels equal to the dataset
    ``nodata`` value are replaced with ``NaN``.

    Parameters
    ----------
    task : tuple
        Tuple containing:

        - year : int
            Year associated with the raster data.
        - month : int
            Month associated with the raster data.
        - ref : hashable
            Key of the reference raster dataset in ``dataset_files``.
        - dataset_files : dict
            Mapping of dataset names to raster file paths.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per pixel in the reference raster.

        The output includes:

        - temporal metadata (``year``, ``month``)
        - pixel indices (``row``, ``col``)
        - spatial coordinates (``x``, ``y``)
        - cyclic month encoding (``month_sin``, ``month_cos``)
        - one column per raster dataset containing extracted values

        If the reference dataset is missing, an empty DataFrame is returned.

    Notes
    -----
    - Raster coordinates are computed using pixel centers via
      ``rasterio.transform.xy``.
    - All raster datasets are closed before returning, even if an exception
      occurs during processing.
    - Pixels outside the extent of a target raster are assigned ``NaN``.
    - This function currently iterates pixel-by-pixel in Python and may be
      slow for large rasters.
    """
    year, month, ref, dataset_files = task
    data = {}
    profiles = {}
    sources = {}

    for variable in dataset_files:
        src = rasterio.open(dataset_files[variable], "r")
        sources[variable] = src
        profiles[variable] = dict(src.profile)
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        scale = 1.0
        offset = 0.0
        if getattr(src, "scales", None) is not None and len(src.scales) >= 1:
            if src.scales[0] is not None:
                scale = src.scales[0]
        if getattr(src, "offsets", None) is not None and len(src.offsets) >= 1:
            if src.offsets[0] is not None:
                offset = src.offsets[0]
        arr = arr * scale + offset
        data[variable] = arr
    all_data = []
    try:
        for row in range(data[ref].shape[0]):
            for col in range(data[ref].shape[1]):
                indices = map_pixel_to_all(row, col, ref, sources)
                res_row = {
                    "year": year,
                    "month": month,
                    "row": row,
                    "col": col,
                    "month_sin": np.sin(2 * np.pi * (month / 12.)),
                    "month_cos": np.cos(2 * np.pi * (month / 12.)),
                }
                x, y = xy(sources[ref].transform, row, col)
                res_row["x"] = x
                res_row["y"] = y
                for s in sources:
                    if indices[s] is None:
                        res_row[s] = np.nan
                    else:
                        r, c = indices[s]
                        res_row[s] = data[s][r, c]
                all_data.append(res_row)
    except KeyError:
        pass
    finally:
        for src in sources.values():
            src.close()
    return pd.DataFrame(all_data)


def assemble_timeseries(database, name_map, ref, max_workers=1):
    """
    Assemble a long-form time series table from raster paths stored in a database.

    This function builds a time-indexed collection of raster file groups using
    :func:`assemble_timeseries_paths_from_db`. Each group corresponds to one
    ``(year, month)`` time step and maps dataset or variable names to raster
    file paths. For each time step, the function calls
    :func:`assemble_data_frame` with a task tuple of the form
    ``(year, month, ref, dataset_files)``. The resulting per-timestep
    DataFrames are concatenated row-wise into a single long-form table.

    Parameters
    ----------
    database : str or pathlib.Path
        Path to the DuckDB database file containing or indexing the raster
        paths.
    name_map : mapping
        Mapping used by :func:`assemble_timeseries_paths_from_db` to translate
        source dataset names, variable names, or database entries into the
        names expected by downstream processing.
    ref : str
        Name of the reference dataset. This is passed through to
        :func:`assemble_data_frame` and identifies the raster used as the
        reference grid, coordinate system, or spatial support.
    max_workers : int, default=1
        Maximum number of worker processes used by :func:`process_map`.

    Returns
    -------
    pandas.DataFrame
        A single DataFrame created by concatenating the per-timestep
        DataFrames returned by :func:`assemble_data_frame`.

    Raises
    ------
    ValueError
        Raised by :func:`pandas.concat` if no per-timestep DataFrames are
        produced.

    See Also
    --------
    assemble_timeseries_paths_from_db :
        Builds the mapping from ``(year, month)`` to raster file dictionaries.
    assemble_data_frame :
        Converts one time-step raster file dictionary into a DataFrame.

    Notes
    -----
    The order of rows in the returned DataFrame follows the iteration order of
    the dictionary returned by :func:`assemble_timeseries_paths_from_db`.

    This function does not add an explicit time column unless
    :func:`assemble_data_frame` does so. If time information is required in the
    final table, ensure that ``assemble_data_frame`` includes it or add it in a
    post-processing step.
    """
    path_dict = assemble_timeseries_paths_from_db(database, name_map)

    tasks = [
        (year, month, ref, dataset_files)
        for (year, month), dataset_files in path_dict.items()
    ]
    result = process_map(
        assemble_data_frame,
        tasks,
        max_workers=max_workers,
        ascii=True
    )
    result = pd.concat(result, ignore_index=True)
    return result


def assemble_timeseries_paths_from_db(database, name_map):
    """
    Assemble per-month raster file paths from a DuckDB GeoTIFF catalog.

    This function reads raster metadata from the ``geotiff_catalog`` table in a
    DuckDB database and builds a dictionary indexed by ``(year, month)``. Each
    value is itself a dictionary mapping normalized variable names to existing
    GeoTIFF file paths.

    Variable names from the database are translated through ``name_map`` before
    being added to the output. Monthly datasets are assigned only to their
    recorded month, while yearly datasets are expanded to all twelve months of
    the corresponding year.

    Parameters
    ----------
    database : str or pathlib.Path
        Path to the DuckDB database containing a ``geotiff_catalog`` table.
        The table must contain the columns ``variable_name``, ``frequency``,
        ``root_dir``, ``file_name``, ``year``, and ``month``.
    name_map : mapping
        Mapping from variable names stored in the database to the variable names
        used in the returned dictionaries.

    Returns
    -------
    dict of tuple of int to dict
        Dictionary mapping ``(year, month)`` tuples to dictionaries of raster
        paths. Each nested dictionary maps normalized variable names to
        :class:`pathlib.Path` objects.

        For example::

            {
                (2020, 1): {
                    "temperature": Path("/data/2020/01/temp.tif"),
                    "elevation": Path("/data/static/elevation.tif"),
                }
            }

    Raises
    ------
    KeyError
        If a database variable name is not present in ``name_map``.
    RuntimeError
        If a row contains an unsupported ``frequency`` value.

    Notes
    -----
    Rows whose resolved file paths do not exist on disk are skipped silently.

    The supported frequency values are ``"monthly"`` and ``"yearly"``. Yearly
    files are duplicated across months 1 through 12 in the returned mapping.
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
@click.option('-c', '--config-path', help='Path to the YAML config file with experiment parameters')
@click.option('-w', '--max-workers', help='Number of parallel processes', default=1)
def main(config_path, max_workers):
    config_path = Path(config_path)
    with config_path.open('r') as fd:
        config_data = yaml.safe_load(fd)
    experiment_dir = config_path.parent
    output_folder = experiment_dir / 'data'
    geojson_path = experiment_dir / config_data['geojson']
    location_nickname = config_data['name']
    input_db = experiment_dir / f"{location_nickname}_source_db.duckdb"
    output_file = experiment_dir / f"{location_nickname}_ard.duckdb"
    name_map = config_data['name_map']
    reference_var = config_data['reference_var']
    df = assemble_timeseries(input_db, name_map, reference_var, max_workers=max_workers)
    conn = duckdb.connect(output_file)
    conn.execute(f"CREATE OR REPLACE TABLE {location_nickname} AS SELECT * FROM df")
    conn.close()

if __name__ == '__main__':
    main()
