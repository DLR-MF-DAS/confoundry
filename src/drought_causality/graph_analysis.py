import json
import hashlib
import click
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def graph_signature(row, mode="topology", weight_decimals=3):
    """
    mode:
      topology -> same directed edges, ignoring weights
      signed    -> same directed edges and edge signs
      weighted  -> same directed edges and rounded edge weights
    """
    labels = json.loads(row["variable_names_json"])
    B = np.asarray(json.loads(row["adjacency_consensus_json"]), dtype=float)

    edges = []

    # Your convention: B[child, parent]
    for child_idx, child_name in enumerate(labels):
        for parent_idx, parent_name in enumerate(labels):
            if child_idx == parent_idx:
                continue

            weight = B[child_idx, parent_idx]

            if weight == 0:
                continue

            if mode == "topology":
                edge = (parent_name, child_name)
            elif mode == "signed":
                edge = (parent_name, child_name, int(np.sign(weight)))
            elif mode == "weighted":
                edge = (parent_name, child_name, round(float(weight), weight_decimals))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            edges.append(edge)

    edges = sorted(edges)

    payload = {
        "mode": mode,
        "nodes": labels,
        "edges": edges,
    }

    canonical = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    return digest, canonical, len(edges)


@click.command()
@click.option("-d", "--db", required=True, help="DuckDB database containing pixel_graphs.")
@click.option("-t", "--table", default="pixel_graphs", show_default=True)
@click.option("--row-col-cols", multiple=True, default=("row", "col"), show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["topology", "signed", "weighted"]),
    default="topology",
    show_default=True,
    help="Definition of a unique graph.",
)
@click.option("--weight-decimals", default=3, show_default=True)
@click.option("--out-prefix", default="graph_groups", show_default=True)
@click.option("--top-n-legend", default=20, show_default=True)
def group_pixel_graphs(
    db,
    table,
    row_col_cols,
    mode,
    weight_decimals,
    out_prefix,
    top_n_legend,
):
    row_col_cols = list(row_col_cols)

    con = duckdb.connect(db)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()

    if df.empty:
        raise click.ClickException(f"Table {table} is empty.")

    missing = [
        c for c in row_col_cols + ["variable_names_json", "adjacency_consensus_json"]
        if c not in df.columns
    ]
    if missing:
        raise click.ClickException(f"Missing required columns: {missing}")

    sigs = df.apply(
        lambda row: graph_signature(row, mode=mode, weight_decimals=weight_decimals),
        axis=1,
    )

    df["graph_signature"] = [x[0] for x in sigs]
    df["graph_canonical_json"] = [x[1] for x in sigs]
    df["n_edges"] = [x[2] for x in sigs]

    counts = (
        df.groupby(["graph_signature", "graph_canonical_json", "n_edges"])
        .size()
        .reset_index(name="n_pixels")
        .sort_values(["n_pixels", "n_edges"], ascending=[False, True])
        .reset_index(drop=True)
    )

    counts["graph_group"] = np.arange(len(counts), dtype=int)

    df = df.merge(
        counts[["graph_signature", "graph_group", "n_pixels"]],
        on="graph_signature",
        how="left",
    )

    df = df.sort_values(row_col_cols).reset_index(drop=True)
    counts = counts[
        ["graph_group", "n_pixels", "n_edges", "graph_signature", "graph_canonical_json"]
    ]

    print()
    print(f"Total fitted pixels: {len(df)}")
    print(f"Unique graph groups: {len(counts)}")
    print()
    print("Largest graph groups:")
    print(counts.head(20)[["graph_group", "n_pixels", "n_edges"]].to_string(index=False))

    con.register("pixel_group_df", df)
    con.register("graph_count_df", counts)

    con.execute("""
        CREATE OR REPLACE TABLE pixel_graph_groups AS
        SELECT * FROM pixel_group_df
    """)

    con.execute("""
        CREATE OR REPLACE TABLE graph_group_counts AS
        SELECT * FROM graph_count_df
    """)

    con.close()

    df.to_csv(f"{out_prefix}_pixels.csv", index=False)
    counts.to_csv(f"{out_prefix}_counts.csv", index=False)

    # Plot pixels colored by graph group
    row_col = row_col_cols[0]
    col_col = row_col_cols[1]

    rows = np.sort(df[row_col].unique())
    cols = np.sort(df[col_col].unique())

    row_to_i = {v: i for i, v in enumerate(rows)}
    col_to_j = {v: j for j, v in enumerate(cols)}

    img = np.full((len(rows), len(cols)), np.nan)

    for _, r in df.iterrows():
        img[row_to_i[r[row_col]], col_to_j[r[col_col]]] = r["graph_group"]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(img, interpolation="nearest", origin="upper")
    plt.title(f"Pixel graph groups ({mode}); {len(counts)} unique graphs")
    plt.xlabel(col_col)
    plt.ylabel(row_col)
    cbar = plt.colorbar(im)
    cbar.set_label("Graph group ID")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_map.png", dpi=200)
    plt.close()

    # Optional second plot: group sizes
    plt.figure(figsize=(10, 5))
    plt.bar(counts["graph_group"], counts["n_pixels"])
    plt.title("Pixels per graph group")
    plt.xlabel("Graph group ID")
    plt.ylabel("Number of pixels")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_counts.png", dpi=200)
    plt.close()

    print()
    print(f"Wrote:")
    print(f"  DuckDB table: pixel_graph_groups")
    print(f"  DuckDB table: graph_group_counts")
    print(f"  CSV: {out_prefix}_pixels.csv")
    print(f"  CSV: {out_prefix}_counts.csv")
    print(f"  PNG: {out_prefix}_map.png")
    print(f"  PNG: {out_prefix}_counts.png")


if __name__ == "__main__":
    group_pixel_graphs()
