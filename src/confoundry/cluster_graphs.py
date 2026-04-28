import json
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics.pairwise import cosine_similarity

import click
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



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
            raise ValueError(f"Unknown mode: {mode}")

        X.append(M[mask])

    return np.vstack(X), mask

def compute_cluster_similarity_colormap_values(X, clusters):
    """
    Assign each cluster a scalar color value in [0, 1] based on similarity
    between cluster centroids.

    Similar clusters should receive nearby color values.
    """
    unique_clusters = np.array(sorted(np.unique(clusters)))

    if len(unique_clusters) == 1:
        return {int(unique_clusters[0]): 0.5}, unique_clusters

    centroids = np.vstack([
        X[clusters == c].mean(axis=0)
        for c in unique_clusters
    ])

    # Hierarchical ordering of cluster centroids.
    # Clusters close in this order get similar colors.
    Z = linkage(centroids, method="ward")
    order = leaves_list(Z)

    ordered_clusters = unique_clusters[order]
    color_values = np.linspace(0, 1, len(ordered_clusters))

    cluster_to_color_value = {
        int(c): float(v)
        for c, v in zip(ordered_clusters, color_values)
    }

    return cluster_to_color_value, ordered_clusters


def plot_cluster_map(df, cluster_to_color_value, ordered_clusters, outpath):
    plot_df = df.copy()
    plot_df["cluster_similarity_color"] = plot_df["cluster"].map(cluster_to_color_value)

    pivot = plot_df.pivot(
        index="row",
        columns="col",
        values="cluster_similarity_color",
    )

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        pivot.values,
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(im)
    cbar.set_label("cluster similarity ordering")

    if len(ordered_clusters) <= 20:
        ticks = [cluster_to_color_value[int(c)] for c in ordered_clusters]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(c)) for c in ordered_clusters])

    plt.title("Spatial clusters of causal graph adjacency matrices")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pca(X, labels, cluster_to_color_value, outpath):
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    colors = np.array([
        cluster_to_color_value[int(c)]
        for c in labels
    ])

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=colors,
        s=12,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("cluster similarity ordering")

    # Optional: annotate cluster centroids in PCA space
    for c in sorted(np.unique(labels)):
        idx = labels == c
        plt.text(
            Z[idx, 0].mean(),
            Z[idx, 1].mean(),
            str(int(c)),
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Adjacency matrix clusters in PCA space")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_cluster_similarity_matrix(X, clusters, outpath):
    unique_clusters = np.array(sorted(np.unique(clusters)))

    centroids = np.vstack([
        X[clusters == c].mean(axis=0)
        for c in unique_clusters
    ])

    sim = cosine_similarity(centroids)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        sim,
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(im, label="cosine similarity")
    plt.xticks(range(len(unique_clusters)), unique_clusters)
    plt.yticks(range(len(unique_clusters)), unique_clusters)
    plt.xlabel("cluster")
    plt.ylabel("cluster")
    plt.title("Similarity between cluster mean adjacency patterns")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    plt.close()


def plot_cluster_mean_adjacencies(mats, clusters, variable_names, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    for c in sorted(np.unique(clusters)):
        idx = np.where(clusters == c)[0]
        mean_B = np.mean([mats[i] for i in idx], axis=0)

        vmax = np.max(np.abs(mean_B))
        if vmax == 0:
            vmax = 1.0

        plt.figure(figsize=(9, 8))
        plt.imshow(mean_B, vmin=-vmax, vmax=vmax)
        plt.colorbar(label="mean edge weight")
        plt.xticks(range(len(variable_names)), variable_names, rotation=90)
        plt.yticks(range(len(variable_names)), variable_names)
        plt.xlabel("parent / source")
        plt.ylabel("child / target")
        plt.title(f"Cluster {c}: mean adjacency, n={len(idx)}")
        plt.tight_layout()
        plt.savefig(outdir / f"cluster_{c}_mean_adjacency.png", dpi=200)
        plt.close()


@click.command()
@click.option("-d", "--db", required=True, help="DuckDB database produced by graph_discovery.")
@click.option("-t", "--table", default="pixel_graphs", show_default=True)
@click.option("-o", "--output-dir", default="graph_clusters", show_default=True)
@click.option("--adjacency-col", default="adjacency_consensus_json", show_default=True)
@click.option("--n-clusters", default=6, show_default=True, type=int)
@click.option(
    "--mode",
    type=click.Choice(["signed", "abs", "binary"]),
    default="signed",
    show_default=True,
    help="How adjacency matrices are converted into feature vectors.",
)
@click.option("--drop-diag/--keep-diag", default=True, show_default=True)
@click.option("--write-table", default="pixel_graph_clusters", show_default=True)
def cluster_graphs(
    db,
    table,
    output_dir,
    adjacency_col,
    n_clusters,
    mode,
    drop_diag,
    write_table,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()

    if df.empty:
        raise click.ClickException(f"No rows found in {table}")

    required = {"row", "col", adjacency_col, "variable_names_json"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Missing required columns: {sorted(missing)}")

    variable_names = json.loads(df["variable_names_json"].iloc[0])
    mats = [parse_matrix(x) for x in df[adjacency_col]]

    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise click.ClickException(f"Adjacency matrices have inconsistent shapes: {shapes}")

    X, _ = vectorize_matrices(mats, mode=mode, drop_diag=drop_diag)

    # Standardize features so large-weight edges do not dominate solely by scale.
    X_scaled = StandardScaler().fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clusters = model.fit_predict(X_scaled)

    df["cluster"] = clusters.astype(int)

    if len(np.unique(clusters)) > 1 and len(df) > n_clusters:
        sil = silhouette_score(X_scaled, clusters)
    else:
        sil = np.nan

    summary = (
        df.groupby("cluster")
        .size()
        .reset_index(name="n_pixels")
        .sort_values("cluster")
    )

    df.to_csv(output_dir / "pixel_graph_clusters.csv", index=False)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)


    cluster_to_color_value, ordered_clusters = compute_cluster_similarity_colormap_values(
        X_scaled,
        clusters,
    )
    
    plot_cluster_map(
        df,
        cluster_to_color_value,
        ordered_clusters,
        output_dir / "cluster_map_similarity_colored.png",
    )

    plot_pca(
        X_scaled,
        clusters,
        cluster_to_color_value,
        output_dir / "cluster_pca_similarity_colored.png",
    )

    plot_cluster_similarity_matrix(
        X_scaled,
        clusters,
        output_dir / "cluster_similarity_matrix.png",
    )
    
    plot_cluster_mean_adjacencies(
        mats=mats,
        clusters=clusters,
        variable_names=variable_names,
        outdir=output_dir / "mean_adjacencies",
    )

    con.register("clustered_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {write_table} AS SELECT * FROM clustered_df")
    con.close()

    print(f"Wrote clustered table: {write_table}")
    print(f"Wrote plots to: {output_dir}")
    print(f"Silhouette score: {sil:.4f}" if not np.isnan(sil) else "Silhouette score: n/a")
    print(summary)


if __name__ == "__main__":
    cluster_graphs()
