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
    colnames = ['year', 'month', 'row', 'col', 'x', 'y'] + [c.split(',')[0] for c in columns]
    colnames_str = ",".join(colnames)
    df = conn.execute(f"SELECT {colnames_str} FROM {input_table}").fetchdf()

    if "year" not in df.columns or "month" not in df.columns:
        raise click.BadParameter("Input table must contain 'year' and 'month' columns.")

    # Use row/col as spatial identity if present; otherwise fall back to x/y.
    if {"row", "col"}.issubset(df.columns):
        group_keys = ["row", "col"]
    elif {"x", "y"}.issubset(df.columns):
        group_keys = ["x", "y"]
    else:
        raise click.BadParameter(
            "Input table must contain either ('row','col') or ('x','y') "
            "so shifts can be applied within each spatial location."
        )

    # Create a proper monthly timestamp for alignment
    df = df.copy()
    df["_month_index"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )

    columns_ = []

    for spec in columns:
        parts = [p.strip() for p in spec.split(",")]

        if len(parts) == 1:
            col = parts[0]
            if col not in df.columns:
                raise click.BadParameter(f"Column '{col}' not found in table.")
            columns_.append(col)

        elif len(parts) == 2:
            col = parts[0]
            month_shift = int(parts[1])

            if col not in df.columns:
                raise click.BadParameter(f"Column '{col}' not found in table.")

            # Build a shifted lookup table for this variable
            shifted = df[group_keys + ["_month_index", col]].copy()
            shifted["_month_index"] = shifted["_month_index"] - pd.DateOffset(months=month_shift)

            shifted_name = col
            shifted = shifted.rename(columns={col: shifted_name})

            # Merge shifted values back onto the original rows
            df = df.drop(columns=[shifted_name], errors="ignore").merge(
                shifted,
                on=group_keys + ["_month_index"],
                how="left",
            )

            columns_.append(shifted_name)

        else:
            raise click.BadParameter(
                f"Invalid column specification '{spec}'. "
                "Use either 'column' or 'column,shift_months'."
            )

    columns = columns_
    df = df.dropna(subset=columns)
    df = df.drop(columns=["_month_index"])

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

