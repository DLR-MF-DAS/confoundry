import json
import yaml
from pathlib import Path
from typing import Any, Mapping

import click
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from confoundry.per_pixel_graph_discovery import quote_identifier, resolve_path


def parse_matrix(x):
    return np.asarray(json.loads(x), dtype=float)


def graph_config_section(config_data: Mapping[str, Any]) -> Mapping[str, Any]:
    graph_config = config_data.get("graph_discovery") or {}
    if not isinstance(graph_config, Mapping):
        raise click.BadParameter("config['graph_discovery'] must be a mapping.")
    return graph_config


def load_config(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("r", encoding="utf-8") as fd:
        config_data = yaml.safe_load(fd) or {}
    if not isinstance(config_data, Mapping):
        raise click.BadParameter("YAML config must contain a mapping at top level.")
    return config_data


def graph_db_path(config_data: Mapping[str, Any], experiment_dir: Path) -> Path:
    location_nickname = str(config_data["name"])
    graph_config = graph_config_section(config_data)
    graph_db = (
        graph_config.get("output_db")
        or graph_config.get("graph_db")
        or config_data.get("graph_db")
    )
    return resolve_path(
        experiment_dir,
        graph_db,
        experiment_dir / f"{location_nickname}_graphs.duckdb",
    )


def graph_table_name(config_data: Mapping[str, Any]) -> str:
    graph_config = graph_config_section(config_data)
    return str(
        config_data.get("graph_table")
        or graph_config.get("output_table")
        or "pixel_graphs"
    )


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


def plot_map(df, outpath, fig_width=8, fig_height=8, point_size=8):
    plot_df = df[["row", "col", "similarity_order"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sc = ax.scatter(
        plot_df["col"],
        plot_df["row"],
        c=plot_df["similarity_order"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=point_size,
        marker="s",
        linewidths=0,
    )

    fig.colorbar(sc, ax=ax, label="graph similarity order", shrink=0.8)

    ax.set_title("Spatial map of causal graphs ordered by similarity")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Important: preserve geometry, do not stretch.
    ax.set_aspect("equal", adjustable="box")

    # Match image-style row orientation: row 0 at the top.
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_edge_signature_by_color(
    mats,
    color_values,
    variable_names,
    outpath,
    n_bins=8,
    top_k=40,
    threshold=1e-12,
    omit_variables=None,
):
    mats = np.asarray(mats)
    color_values = np.asarray(color_values)
    omit_variables = set(omit_variables or [])

    unknown = sorted(omit_variables - set(variable_names))
    if unknown:
        raise click.ClickException(
            f"Unknown variables in --omit-heatmap-variable: {unknown}. "
            f"Known variables are: {variable_names}"
        )

    omit_idx = {variable_names.index(v) for v in omit_variables}

    n = mats.shape[1]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    rows_all, cols_all = np.where(mask)

    keep_edge_mask = np.array(
        [
            r not in omit_idx and c not in omit_idx
            for r, c in zip(rows_all, cols_all)
        ],
        dtype=bool,
    )

    if not keep_edge_mask.any():
        raise click.ClickException(
            "No edges left for heatmap after omitting variables."
        )

    rows = rows_all[keep_edge_mask]
    cols = cols_all[keep_edge_mask]

    edge_labels = np.array([
        f"{variable_names[c]} → {variable_names[r]}"
        for r, c in zip(rows, cols)
    ])

    present_all = (np.abs(mats[:, mask]) > threshold).astype(float)
    present = present_all[:, keep_edge_mask]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = []
    P = []

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        idx = (color_values >= lo) & (
            color_values < hi if b < n_bins - 1 else color_values <= hi
        )

        bin_labels.append(f"{lo:.2f}–{hi:.2f}")
        P.append(
            present[idx].mean(axis=0)
            if idx.any()
            else np.zeros(present.shape[1])
        )

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

    # Use the same color map as the original spatial ordering plot.
    cmap = plt.get_cmap("viridis")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    color_strip = cmap(bin_centers)[np.newaxis, :, :]

    fig_height = max(5, 0.25 * len(keep) + 1.0)

    # Wider figure so long edge labels have room.
    fig = plt.figure(figsize=(13, fig_height))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[20, 1],
        width_ratios=[30, 1],
        hspace=0.08,
        wspace=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    strip_ax = fig.add_subplot(gs[1, 0])
    
    im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="gray_r")

    fig.colorbar(im, cax=cax, label="edge presence frequency")

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([])
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(edge_labels[keep], fontsize=8)

    ax.set_ylabel("edge")
    ax.set_title("Edges that distinguish graph-color regions")

    strip_ax.imshow(color_strip, aspect="auto")
    strip_ax.set_yticks([])
    strip_ax.set_xticks(range(n_bins))
    strip_ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    strip_ax.set_xlabel("similarity-color range")

    strip_ax.set_xlim(ax.get_xlim())

    # Make extra room on the left for edge descriptions.
    # Increase left further if your variable names are very long.
    fig.subplots_adjust(
        left=0.42,
        right=0.92,
        top=0.93,
        bottom=0.18,
    )

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


@click.command()
@click.option("-c", "--config-path", help="Path to the YAML config file with experiment parameters")
@click.option("--mode", type=click.Choice(["signed", "abs", "binary"]), default="signed")
@click.option("--drop-diag/--keep-diag", default=True)
@click.option(
    "--omit-heatmap-variable",
    multiple=True,
    help="Variable to omit from the edge-signature heatmap. Can be used multiple times.",
)
def order_graphs(config_path, mode, drop_diag, omit_heatmap_variable):
    config_path = Path(config_path)
    config_data = load_config(config_path)
    experiment_dir = config_path.parent
    db = graph_db_path(config_data, experiment_dir)
    table = graph_table_name(config_data)
    output_dir = "graph_similarity_order"
    adjacency_col = "adjacency_consensus_json"
    write_table = "pixel_graph_similarity_order"
    output_dir = Path(experiment_dir) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db)
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if table not in tables:
        con.close()
        raise click.ClickException(
            f"{table!r} not found in {db}. Available tables: {sorted(tables)}"
        )
    df = con.execute(f"SELECT * FROM {quote_identifier(table)}").fetchdf()

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
    plot_map(
        df,
        output_dir / "similarity_order_map.png",
        fig_width=8,
        fig_height=8,
        point_size=8,
    )

    variable_names = json.loads(df["variable_names_json"].iloc[0])


    plot_edge_signature_by_color(
        mats,
        df["similarity_order"].to_numpy(),
        variable_names,
        output_dir / "edge_signature_by_color.png",
        omit_variables=omit_heatmap_variable,
    )

    con.register("ordered_df", df)
    con.execute(
        f"CREATE OR REPLACE TABLE {quote_identifier(write_table)} AS SELECT * FROM ordered_df"
    )
    con.close()

    print(f"Wrote ordered table: {write_table}")
    print(f"Wrote plots to: {output_dir}")

    
if __name__ == "__main__":
    order_graphs()
