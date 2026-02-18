import click
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

def assemble_data_frame(ref, dataset_files):
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

def assemble_timeseries(database, name_map, ref):
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
    for k in path_dict:
        print("Processing:", path_dict[k])
        result.append(assemble_data_frame(ref, path_dict[k]))
    result = pd.concat(result)
    return result

def assemble_timeseries_paths(root, dataset_files):
    """
    Assemble per-month dataset file paths from a directory tree.

    The directory layout is assumed to be:

        root/
            YYYY/
                MM/
                    <files>

    where:
      * YYYY is a 4-digit year directory name (e.g. "2023")
      * MM is a 2-digit month directory name (e.g. "01")

    For each (year, month) directory, this function creates a dictionary
    mapping variable names to the corresponding file paths, using the
    ``dataset_files`` mapping of variable -> filename. The result is a list
    of such dictionaries, ordered by year and month.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory containing the yearly subdirectories.
    dataset_files : dict
        Mapping from variable name to file name (no path). Each file name
        is joined to the corresponding year/month directory.

    Returns
    -------
    list of dict
        A list of dictionaries. Each dictionary represents one (year, month)
        timestep and maps variable names to file paths (as strings).

    Notes
    -----
    This function does not verify that the constructed file paths actually
    exist; it only builds paths based on the directory structure and
    ``dataset_files``.
    """
    root = Path(root)
    all_datasets = []

    # Year dirs: root/2023, root/2024, ...
    year_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4
    )

    for year_dir in year_dirs:
        # Month dirs: root/2023/01, root/2023/02, ...
        month_dirs = sorted(
            d for d in year_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) <= 2
        )

        for month_dir in month_dirs:
            full_paths = {}
            for variable, filename in dataset_files.items():
                filenames = list(glob.glob(str(month_dir / filename)))
                try:
                    full_paths[variable] = filenames[0]
                except IndexError:
                    continue
            all_datasets.append(full_paths)

    return all_datasets

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

def process_group(task):
    row_col, group, params = task
    group = group.dropna()
    row, col = row_col
    if len(group) < 10:
        return row, col, params["fill_value"]
    model = params["model_cls"](
        data=group,
        treatment=params["treatment"],
        outcome=params["outcome"],
        graph=params["graph"],
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(
        estimand,
        method_name=params["method_name"],
        control_value=params["control_value"],
        treatment_value=params["treatment_value"]
    )
    value = float(getattr(estimate, "value", estimate))
    return row, col, value

def timeseries_causal_analysis(
    df,
    graph,
    treatment,
    outcome,
    method_name="backdoor.linear_regression",
    control_value=-1,
    treatment_value=1,
    fill_value=np.nan,
    model_cls=None,
):
    """
    Estimate a causal effect per (row, col) cell for a gridded outcome.

    IMPORTANT INVARIANT
    -------------------
    The returned array has the SAME spatial shape as the outcome grid.
    Each (row, col) pair is interpreted as a direct array index.
    No remapping, compaction, or resizing is performed.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - integer 'row' and 'col' grid indices
          - treatment and outcome columns
        Missing grid cells are allowed and will be filled with `fill_value`.
    graph : str
        DoWhy causal graph specification.
    treatment : str
        Name of the treatment column.
    outcome : str
        Name of the outcome column.
    method_name : str
        DoWhy estimator method.
    fill_value : float
        Value for cells with insufficient data or failed estimation.
    model_cls : class, optional
        DoWhy-compatible CausalModel class (for testing or injection).

    Returns
    -------
    numpy.ndarray
        2D array of causal effect estimates with shape:
        (max_row + 1, max_col + 1)
    """
    if model_cls is None:
        if CausalModel is None:
            raise ImportError("dowhy is not available; install dowhy or pass model_cls")
        model_cls = CausalModel
    for colname in ("row", "col", treatment, outcome):
        if colname not in df.columns:
            raise KeyError("df is missing required column: " + colname)
    if not np.issubdtype(df["row"].dtype, np.integer):
        raise ValueError("'row' must be integer grid indices")
    if not np.issubdtype(df["col"].dtype, np.integer):
        raise ValueError("'col' must be integer grid indices")
    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    if max_row < 0 or max_col < 0:
        raise ValueError("row/col indices must be non-negative")
    result = np.full((max_row + 1, max_col + 1), fill_value)
    warnings.filterwarnings("ignore")
    def append_args(gen):
        for item in gen:
            yield item + (
                {
                    "fill_value": fill_value,
                    "model_cls": model_cls,
                    "treatment": treatment,
                    "outcome": outcome,
                    "graph": graph,
                    "method_name": method_name,
                    "control_value": control_value,
                    "treatment_value": treatment_value,
                },)
    groups = list(append_args(df.groupby(["row", "col"], sort=False)))
    for row, col, value in process_map(process_group, groups, max_workers=4, ascii=True):
        result[row][col] = value
    # for (row, col), group in tqdm.tqdm(df.groupby(["row", "col"], sort=False)):
    #     group = group.dropna()
    #     if len(group) < 10:
    #         continue
    #     model = model_cls(
    #         data=group,
    #         treatment=treatment,
    #         outcome=outcome,
    #         graph=graph,
    #     )
    #     estimand = model.identify_effect()
    #     estimate = model.estimate_effect(
    #         estimand,
    #         method_name=method_name,
    #         control_value=control_value,
    #         treatment_value=treatment_value
    #     )
    #     value = float(getattr(estimate, "value", estimate))
    #     result[row, col] = value
    warnings.resetwarnings()
    return result


def save_array_as_geotiff(array, reference_geotiff, output_path,
                          nodata=None, compress="deflate", overwrite=True):
    """
    Save a NumPy array as a GeoTIFF using the spatial metadata and profile
    from a reference GeoTIFF.

    The input array must have the same spatial dimensions as the reference.
    A 2D array is written as a single-band GeoTIFF, while a 3D array is
    interpreted as (bands, height, width).

    Parameters
    ----------
    array : np.ndarray
        Data to write (H, W) or (bands, H, W).
    reference_geotiff : str or Path
        Path to an existing GeoTIFF whose profile (CRS, transform, etc.)
        will be reused.
    output_path : str or Path
        Path where the output GeoTIFF will be written.
    nodata : optional
        Nodata value to set in the output file. If None, the reference
        nodata value is kept.
    compress : str or None
        GeoTIFF compression method (e.g. "deflate", "lzw").
    overwrite : bool
        Whether to overwrite an existing output file.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    if array.ndim == 2:
        data = array[np.newaxis, ...]
    elif array.ndim == 3:
        data = array
    else:
        raise ValueError("array must be 2D or 3D")
    with rasterio.open(reference_geotiff) as ref:
        if data.shape[1] != ref.height or data.shape[2] != ref.width:
            raise ValueError("array shape does not match reference geotiff")
        profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
    )
    if compress:
        profile["compress"] = compress
    if nodata is not None:
        profile["nodata"] = nodata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

@click.command()
@click.option('-i', '--input-db', help='A DuckDB database with the input table')
@click.option('-n', '--input-table', help='Input table name')
@click.option('-g', '--graph-file', help='A causal graph file in graphviz format')
@click.option('-t', '--treatment', help='Name of the treatment column')
@click.option('-c', '--outcome', help='Name of the outcome column')
@click.option('-o', '--output-file', help='Name of the output file')
@click.option('-r', '--reference', help='Reference GeoTIFF when saving the result')
def analyse_dataframe(input_db, input_table, graph_file, treatment, outcome, output_file, reference):
    conn = duckdb.connect(input_db)
    print(conn.sql("SHOW TABLES").df())
    df = conn.execute(f"SELECT * FROM {input_table}").fetchdf()
    with open(graph_file, 'rt') as fd:
        graph = fd.read()
    result = timeseries_causal_analysis(df, graph, treatment, outcome)
    save_array_as_geotiff(result, reference, output_file)

if __name__ == '__main__':
    analyse_dataframe()
