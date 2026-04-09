import click
import duckdb
import pandas as pd
import numpy as np
import os
from pathlib import Path
from itertools import repeat
from tqdm.contrib.concurrent import process_map
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci


def bootstrap_once(seed, df, columns, samples):
    analysis_df = df.sample(n=samples, replace=True, random_state=seed)[list(columns)]
    labels = analysis_df.columns.tolist()
    cg, edges = fci(
        analysis_df.values,
        depth=-1,
        alpha=0.01,
        independence_test_method=kci,
        labels=labels,
    )
    return {"labels": labels, "graph": list(cg.graph)}


@click.command()
@click.option('-i', '--input-db', help='A DuckDB database with the input table')
@click.option('-n', '--input-table', help='Input table name')
@click.option('-c', '--columns', help='Column names to be used', multiple=True)
@click.option('-o', '--output', help='Output file name for the DuckDB file')
@click.option('-s', '--samples', help='How many rows to sample with replacement', default=2000)
@click.option('-b', '--bootstrap-samples', help='How many samples for bootstrapping', default=2000)
@click.option('-p', '--processes', help='Number of processes', default=1)
def graph_discovery(input_db, input_table, columns, output, samples, bootstrap_samples, processes):
    conn = duckdb.connect(input_db)
    tables = list(conn.sql("SHOW TABLES").df()['name'])
    if input_table not in tables:
        raise click.BadParameter(
            f"The table {input_table} not in the {input_db} database. "
            f"The available tables are: {tables}"
        )
    df = conn.execute(f"SELECT * FROM {input_table}").fetchdf().dropna()
    conn.close()
    if len(columns) < 1:
        raise click.BadParameter(
            f"Please specify one or more columns from {df.columns.tolist()}"
        )
    result = process_map(
        bootstrap_once,
        range(bootstrap_samples),
        repeat(df),
        repeat(columns),
        repeat(samples),
        max_workers=processes,
        chunksize=1,
    )
    result_df = pd.DataFrame(result)
    conn = duckdb.connect(output)
    conn.sql("CREATE OR REPLACE TABLE graphs AS SELECT * FROM result_df")
    conn.close()


if __name__ == '__main__':
    graph_discovery()

