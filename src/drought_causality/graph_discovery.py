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
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import kci

@click.command()
@click.option('-i', '--input-db', help='A DuckDB database with the input table')
@click.option('-n', '--input-table', help='Input table name')
@click.option('-c', '--columns', help='Column names to be used', multiple=True)
@click.option('-o', '--output', help='Output file name for the DOT graph')
def graph_discovery(input_db, input_table, columns, output):
    if output is not None:
        output_path = Path(output)
    conn = duckdb.connect(input_db)
    tables = list(conn.sql("SHOW TABLES").df()['name'])
    if input_table not in tables:
        raise click.BadParameter(f"The table {input_table} not in the {input_db} database. The tables available tables are: {tables}")
    df = conn.execute(f"SELECT * FROM {input_table}").fetchdf()
    df = df.dropna()
    df = df.sample(2000)
    if len(columns) < 1:
        raise click.BadParameter(f"Please specify one or more columns from {list(df.columns.tolist())}")
    df = df[list(columns)]
    labels = list(df.columns)
    cg, edges = fci(df.values, depth=-1, alpha=0.01, independence_test_method=kci, labels=labels)
    pdy = GraphUtils.to_pydot(cg, labels=labels)
    pdy.write_png(output_path)

if __name__ == '__main__':
    graph_discovery()
