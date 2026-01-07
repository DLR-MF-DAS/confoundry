import rasterio
from rasterio.transform import xy, rowcol, from_origin
import pandas as pd
import numpy as np
from pathlib import Path
from dowhy import CausalModel
import glob

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
    ref_ds = datasets[ref]
    # 1) Reference pixel -> map coordinates
    x, y = xy(ref_ds.transform, row, col)  # center of pixel

    results = {}

    # Helper to do bounds checking
    def _safe_rowcol(ds, x, y):
        try:
            r, c = rowcol(ds.transform, x, y)
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
        with rasterio.open(dataset_files[variable], 'r') as src:
            data[variable] = src.read(1).astype(float)
            data[variable][data[variable] == src.profile['nodata']] = np.nan
            profiles[variable] = src.profile
            sources[variable] = src
    res = map_pixel_to_all(0, 0, ref, sources)
    all_data = []
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
                try:
                    r, c = indices[s]
                    res_row[s] = data[s][r][c]
                except TypeError:
                    res_row[s] = np.nan
            all_data.append(res_row)
    return pd.DataFrame(all_data)

def assemble_timeseries(root, ref, dataset_files):
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
    root : str or pathlib.Path
        Root directory containing the YYYY/MM directory structure. The
        Year/Month layout and the construction of per-timestep dictionaries
        are handled by :func:`assemble_timeseries_paths`.
    ref : str
        Name of the reference dataset passed through to
        :func:`assemble_data_frame`. This identifies which raster defines
        the pixel grid / coordinate system for the aggregation.
    dataset_files : dict
        Mapping from dataset name to file name (without any path). For each
        year/month directory, these file names are joined with that directory
        to obtain full paths.

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
    paths = assemble_timeseries_paths(root, dataset_files)
    result = pd.concat([assemble_data_frame(ref, path_dict) for path_dict in paths])
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
            if d.is_dir() and d.name.isdigit() and len(d.name) == 2
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

def timeseries_causal_analysis(
    df,
    graph,
    treatment,
    outcome,
    method_name="backdoor.linear_regression",
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
    for (row, col), group in df.groupby(["row", "col"], sort=False):
        group = group.dropna(subset=[treatment, outcome])
        if group.empty:
            continue
        try:
            model = model_cls(
                data=group,
                treatment=treatment,
                outcome=outcome,
                graph=graph,
            )
            estimand = model.identify_effect()
            estimate = model.estimate_effect(
                estimand,
                method_name=method_name,
            )
            value = float(getattr(estimate, "value", estimate))
        except Exception:
            continue
        result[row, col] = value
    return result

def save_array_as_geotiff_from_df(
    arr,
    df,
    out_path,
    crs="EPSG:4326",
    nodata=None,
):
    """
    Save a 2D numpy array as a GeoTIFF using a dataframe that maps (row, col) to (lat, lon).

    Assumptions (common for gridded EO products):
    - df has columns: 'row', 'col', 'lat', 'lon'
    - row/col are 0-based indices that align with arr[row, col]
    - the grid is regular (constant spacing in lat and lon)
    - lat/lon in df refer to pixel centers (not corners)

    Writes a north-up GeoTIFF with an affine geotransform.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be a 2D array")
    required = {"row", "col", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError("df missing columns: " + ", ".join(sorted(missing)))
    height, width = arr.shape
    d = df[(df["row"] >= 0) & (df["row"] < height) & (df["col"] >= 0) & (df["col"] < width)].copy()
    if d.empty:
        raise ValueError("No df entries fall inside the array bounds")
    row_lat = d.groupby("row")["lat"].median()
    col_lon = d.groupby("col")["lon"].median()
    if row_lat.size < 2 or col_lon.size < 2:
        raise ValueError("Need at least 2 distinct rows and cols with lat/lon to infer resolution")
    row_lat = row_lat.sort_index()
    col_lon = col_lon.sort_index()
    dlat = np.median(np.diff(row_lat.values))
    dlon = np.median(np.diff(col_lon.values))
    if not np.isfinite(dlat) or not np.isfinite(dlon) or dlat == 0 or dlon == 0:
        raise ValueError("Could not infer non-zero grid resolution from df")
    arr_to_write = arr
    if dlat > 0:
        arr_to_write = np.flipud(arr_to_write)
        dlat = -dlat
    xres = abs(dlon)
    yres = abs(dlat)
    left = float(col_lon.min() - 0.5 * xres)
    top = float(row_lat.max() + 0.5 * yres)
    transform = from_origin(left, top, xres, yres)
    profile = {
        "driver": "GTiff",
        "height": arr_to_write.shape[0],
        "width": arr_to_write.shape[1],
        "count": 1,
        "dtype": arr_to_write.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2 if np.issubdtype(arr_to_write.dtype, np.floating) else 1,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_to_write, 1)
