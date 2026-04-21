import json
import click
import duckdb
import numpy as np
import pandas as pd
import networkx as nx
import lingam
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def parse_columns(df, group_cols, order_cols, column_specs):
    df = df.sort_values(group_cols + order_cols).copy()
    labels = []
    label_lags = {}

    for spec in column_specs:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) == 1:
            base = parts[0]
            label = base
            lag = 0
        elif len(parts) == 2:
            base = parts[0]
            lag = int(parts[1])
            label = f"{base}_lag{lag}"
            df[label] = df.groupby(group_cols)[base].shift(lag)
        else:
            raise click.BadParameter(f"Invalid column spec: {spec}")

        if label in labels:
            raise click.BadParameter(f"Duplicate derived column: {label}")

        labels.append(label)
        label_lags[label] = lag

    return df, labels, label_lags


def make_prior_knowledge(labels, label_lags):
    # pk[i, j] = 0 means xi cannot cause xj
    # less-delayed variables cannot cause more-delayed variables
    pk = -np.ones((len(labels), len(labels)), dtype=int)
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i != j and label_lags[src] < label_lags[dst]:
                pk[i, j] = 0
    return pk


def to_graph(B, labels, min_abs_effect):
    # LiNGAM adjacency convention: B[child, parent]
    G = nx.DiGraph()
    G.add_nodes_from(labels)
    for child, child_name in enumerate(labels):
        for parent, parent_name in enumerate(labels):
            if child != parent and abs(B[child, parent]) >= min_abs_effect:
                G.add_edge(parent_name, child_name, weight=float(B[child, parent]))
    return G


def fit_pixel(pixel_key, g, labels, pk, bootstrap_samples, min_samples, min_prob, min_abs_effect, group_cols):
    X = g[labels].dropna().to_numpy()

    if len(X) < min_samples:
        return None

    model = lingam.DirectLiNGAM(
        prior_knowledge=pk,
        apply_prior_knowledge_softly=False,
        random_state=0,
    )
    model.fit(X)

    boot = model.bootstrap(X, n_sampling=bootstrap_samples)
    prob = np.asarray(boot.get_probabilities(min_causal_effect=min_abs_effect), dtype=float)
    B_raw = np.asarray(model.adjacency_matrix_, dtype=float)
    B_cons = np.where(prob >= min_prob, B_raw, 0.0)
    B_cons = np.where(np.abs(B_cons) >= min_abs_effect, B_cons, 0.0)

    G = to_graph(B_cons, labels, min_abs_effect)
    row = dict(zip(group_cols, pixel_key if isinstance(pixel_key, tuple) else (pixel_key,)))
    row.update(
        n_samples=int(len(X)),
        variable_names_json=json.dumps(labels),
        variable_index_json=json.dumps({name: i for i, name in enumerate(labels)}),
        causal_order_json=json.dumps([int(i) for i in model.causal_order_]),
        adjacency_raw_json=json.dumps(B_raw.tolist()),
        edge_probability_json=json.dumps(prob.tolist()),
        adjacency_consensus_json=json.dumps(B_cons.tolist()),
        gml_graph="\n".join(nx.generate_gml(G)),
    )
    return row

def fit_pixel_task(args):
    pixel_key, g, labels, pk, bootstrap_samples, min_samples, min_edge_prob, min_abs_effect, row_col_cols = args
    return fit_pixel(
        pixel_key=pixel_key,
        g=g,
        labels=labels,
        pk=pk,
        bootstrap_samples=bootstrap_samples,
        min_samples=min_samples,
        min_prob=min_edge_prob,
        min_abs_effect=min_abs_effect,
        group_cols=row_col_cols,
    )


@click.command()
@click.option("-i", "--input-db", required=True)
@click.option("-n", "--input-table", required=True)
@click.option("-o", "--output-db", required=True)
@click.option("--row-col-cols", multiple=True, default=("row", "col"), show_default=True)
@click.option("--order-cols", multiple=True, default=("year", "month"), show_default=True)
@click.option(
    "-c", "--columns", multiple=True, required=True,
    help="Columns in order, optionally with lag like var,1"
)
@click.option("-b", "--bootstrap-samples", default=200, show_default=True, type=int)
@click.option("--min-samples", default=50, show_default=True, type=int)
@click.option("--min-edge-prob", default=0.7, show_default=True, type=float)
@click.option("--min-abs-effect", default=0.01, show_default=True, type=float)
@click.option("-w", "--workers", default=1, show_default=True)
def graph_discovery(
    input_db,
    input_table,
    output_db,
    row_col_cols,
    order_cols,
    columns,
    bootstrap_samples,
    min_samples,
    min_edge_prob,
    min_abs_effect,
    workers,
):
    row_col_cols = list(row_col_cols)
    order_cols = list(order_cols)

    con = duckdb.connect(input_db, read_only=True)
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if input_table not in tables:
        raise click.BadParameter(f"{input_table} not found in {input_db}. Available: {sorted(tables)}")

    df = con.execute(f"SELECT * FROM {input_table}").fetchdf()
    con.close()

    missing = [c for c in row_col_cols + order_cols if c not in df.columns]
    if missing:
        raise click.BadParameter(f"Missing required columns: {missing}")

    df, labels, label_lags = parse_columns(df, row_col_cols, order_cols, columns)
    missing = [c.split(",")[0].strip() for c in columns if c.split(",")[0].strip() not in df.columns]
    if missing:
        raise click.BadParameter(f"Missing data columns: {missing}")

    df = df.dropna(subset=labels + row_col_cols + order_cols)
    pk = make_prior_knowledge(labels, label_lags)

    groups = df.groupby(row_col_cols, sort=True)
    total = df.groupby(row_col_cols, sort=True).ngroups

    groups = list(df.groupby(row_col_cols, sort=True))
    tasks = [
        (pixel_key, g, labels, pk, bootstrap_samples, min_samples, min_edge_prob, min_abs_effect, row_col_cols)
        for pixel_key, g in groups
    ]
    rows = process_map(
        fit_pixel_task,
        tasks,
        max_workers=workers,
        chunksize=1,
        desc="Pixels",
    )

    rows = [row for row in rows if row is not None]

    if not rows:
        raise click.ClickException("No pixel had enough samples after lagging/dropna.")

    result_df = pd.DataFrame(rows)
    con = duckdb.connect(output_db)
    con.register("result_df", result_df)
    con.execute("CREATE OR REPLACE TABLE pixel_graphs AS SELECT * FROM result_df")
    con.close()


if __name__ == "__main__":
    graph_discovery()
