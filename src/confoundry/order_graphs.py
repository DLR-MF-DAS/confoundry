import json
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_matrix(x):
    return np.asarray(json.loads(x), dtype=float)


def vectorize_matrices(mats, mode="signed", drop_diag=True):
    n = mats[0].shape[0]
    mask = np.ones((n, n), dtype=bool)
    if drop_diag:
        np.fill_diagonal(mask, False)
    X = []
    for B in mats:
        if mode == "signed":
            M = B
        elif mode == "abs":
            M = np.abs(B)
        elif mode == "binary":
            M = (np.abs(B) > 0).astype(float)
        else:
            raise ValueError(mode)
        X.append(M[mask])
    return np.vstack(X)


def similarity_order(X):
    z = PCA(n_components=1, svd_solver="randomized", random_state=0).fit_transform(X).ravel()
    return np.argsort(z)


def add_similarity_color(df, order):
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))

    df["similarity_rank"] = pos
    df["similarity_order"] = pos / max(len(order) - 1, 1)
    return df


def plot_map(df, outpath):
    pivot = df.pivot(index="row", columns="col", values="similarity_order")

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        pivot.values,
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label="graph similarity order", shrink=0.6)
    plt.title("Spatial map of causal graphs ordered by similarity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_edge_signature_by_color(mats, color_values, variable_names, outpath,
                                 n_bins=8, top_k=40, threshold=1e-12):
    mats = np.asarray(mats)
    color_values = np.asarray(color_values)

    n = mats.shape[1]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    rows, cols = np.where(mask)
    edge_labels = np.array([
        f"{variable_names[c]} → {variable_names[r]}"
        for r, c in zip(rows, cols)
    ])

    present = (np.abs(mats[:, mask]) > threshold).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = []
    P = []

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        idx = (color_values >= lo) & (
            color_values < hi if b < n_bins - 1 else color_values <= hi
        )

        bin_labels.append(f"{lo:.2f}–{hi:.2f}")
        P.append(present[idx].mean(axis=0) if idx.any() else np.zeros(present.shape[1]))

    P = np.vstack(P).T  # edges x bins

    # Keep only edges whose presence changes most across the color ordering.
    score = P.max(axis=1) - P.min(axis=1)
    keep = np.argsort(score)[-min(top_k, len(score)):]

    # Sort edges by where they are most present.
    peak_bin = np.argmax(P[keep], axis=1)
    source_i = cols[keep]
    target_i = rows[keep]

    row_order = np.lexsort((target_i, source_i, peak_bin))
    keep = keep[row_order]

    M = P[keep]

    pd.DataFrame(M, index=edge_labels[keep], columns=bin_labels).to_csv(
        Path(outpath).with_suffix(".csv")
    )

    plt.figure(figsize=(9, max(5, 0.25 * len(keep))))
    im = plt.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="gray_r")

    plt.colorbar(im, label="edge presence frequency")
    plt.xticks(range(n_bins), bin_labels, rotation=45, ha="right")
    plt.yticks(range(len(keep)), edge_labels[keep], fontsize=8)

    plt.xlabel("similarity-color range")
    plt.ylabel("edge")
    plt.title("Edges that distinguish graph-color regions")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


@click.command()
@click.option("-d", "--db", required=True)
@click.option("-t", "--table", default="pixel_graphs", show_default=True)
@click.option("-o", "--output-dir", default="graph_similarity_order", show_default=True)
@click.option("--adjacency-col", default="adjacency_consensus_json", show_default=True)
@click.option("--mode", type=click.Choice(["signed", "abs", "binary"]), default="signed")
@click.option("--drop-diag/--keep-diag", default=True)
@click.option("--write-table", default="pixel_graph_similarity_order", show_default=True)
def order_graphs(db, table, output_dir, adjacency_col, mode, drop_diag, write_table):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()

    if df.empty:
        raise click.ClickException(f"No rows found in {table}")

    required = {"row", "col", adjacency_col}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Missing required columns: {sorted(missing)}")

    with click.progressbar(df[adjacency_col], label="Parsing matrices") as xs:
        mats = [parse_matrix(x) for x in xs]
        
    if len({m.shape for m in mats}) != 1:
        raise click.ClickException("Adjacency matrices have inconsistent shapes")

    click.echo("Vectorizing graphs...")
    X = vectorize_matrices(mats, mode=mode, drop_diag=drop_diag)

    click.echo("Scaling features...")
    X = StandardScaler().fit_transform(X)

    click.echo("Ordering graphs by 1D PCA...")
    order = similarity_order(X)
    df = add_similarity_color(df, order)

    click.echo("Plotting...")
    plot_map(df, output_dir / "similarity_order_map.png")

    variable_names = json.loads(df["variable_names_json"].iloc[0])


    plot_edge_signature_by_color(
        mats,
        df["similarity_order"].to_numpy(),
        variable_names,
        output_dir / "edge_signature_by_color.png",
    )

    con.register("ordered_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {write_table} AS SELECT * FROM ordered_df")
    con.close()

    print(f"Wrote ordered table: {write_table}")
    print(f"Wrote plots to: {output_dir}")

    
if __name__ == "__main__":
    order_graphs()
