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
    tables = list(conn.sql("SHOW TABLES").df()['name'])
    if input_table not in tables:
        raise click.BadParameter(f"The table {input_table} not in the {input_db} database. The tables available tables are: {tables}")
    df = conn.execute(f"SELECT * FROM {input_table}").fetchdf()
    with open(graph_file, 'rt') as fd:
        graph = fd.read()
    result = timeseries_causal_analysis(df, graph, treatment, outcome)
    save_array_as_geotiff(result, reference, output_file)

if __name__ == '__main__':
    analyse_dataframe()
