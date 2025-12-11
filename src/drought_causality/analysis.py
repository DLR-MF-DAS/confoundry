import rasterio
from rasterio.transform import xy, rowcol
import pandas as pd
import numpy as np
from pathlib import Path


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
    res = map_pixel_to_all(0, 0, 'ndvi', sources)
    all_data = []
    for row in range(data['ndvi'].shape[0]):
        for col in range(data['ndvi'].shape[1]):
            indices = map_pixel_to_all(row, col, 'ndvi', sources)
            res_row = {}
            res_row['row'] = row
            res_row['col'] = col
            lat, lon = xy(sources['ndvi'].transform, row, col)
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
                full_paths[variable] = str(month_dir / filename)
            all_datasets.append(full_paths)

    return all_datasets
