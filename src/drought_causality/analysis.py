import rasterio
from rasterio.transform import xy, rowcol
import pandas as pd
import numpy as np

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

def assemble_timeseries_paths(root, ref, dataset_files):
    all_datasets = []
    for year in [f for f in glob.glob("*/") if f[:-1].isdigit() and len(f[:-1]) == 4]:
        for month in [f for f in glob.glob(f"{year}/*/") if f[:-1].isdigit() and len(f[:-1]) == 2]:
            full_path = {}
            for variable in dataset_files:
                full_path[variable] = os.path.join(month, dataset_files[variable])
            all_datasets.append(full_path)
    return all_datasets
