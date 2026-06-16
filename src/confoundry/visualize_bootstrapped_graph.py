import duckdb
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Optional SQL filter, e.g. "split = 'test'"
WHERE = None

# causal-learn / PyWhy endpoint labels
EDGE_LABELS = {
    -1: "tail",
     0: "null",
     1: "arrow",
     2: "circle",
     3: "star",
     4: "tail+arrow",
     5: "arrow+arrow",
     6: "tail+tail",
}

# Stable display order
DISPLAY_ORDER = [0, -1, 1, 2, 3, 4, 5, 6]

# Colors chosen to read well on slides
EDGE_COLORS = {
    0: "#d9d9d9",   # null
   -1: "#4e79a7",   # tail
    1: "#e15759",   # arrow
    2: "#59a14f",   # circle
    3: "#9c755f",   # star
    4: "#f28e2b",   # tail+arrow
    5: "#b07aa1",   # arrow+arrow
    6: "#76b7b2",   # tail+tail
}


def load_matrices(db_path, table, column, where=None):
    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"SELECT {column} FROM {table}"
        if where:
            query += f" WHERE {where}"

        rows = con.execute(query).fetchall()
    finally:
        con.close()

    if not rows:
        raise ValueError("Query returned no rows.")
    breakpoint()
    mats = [np.asarray(row[0], dtype=int) for row in rows]

    # all matrices must have same shape
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"Found multiple shapes: {shapes}")

    n, m = mats[0].shape
    if n != m:
        raise ValueError(f"Matrices must be square, got shape {(n, m)}")

    return np.stack(mats, axis=0)   # shape: (n_graphs, n, n)


def load_node_names(db_path, table, labels_column="labels"):
    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(f"SELECT {labels_column} FROM {table} LIMIT 1").fetchone()
    finally:
        con.close()

    if row is None or row[0] is None:
        return None

    return list(row[0])


def plot_endpoint_hist_grid(mats, node_names=None, outfile="fci_endpoint_hist_grid.png"):
    if mats.ndim != 3:
        raise ValueError(f"Expected mats to have shape (n_graphs, n, n), got {mats.shape}")

    n_graphs, n, m = mats.shape
    if n != m:
        raise ValueError(f"Expected square matrices, got {(n, m)}")

    if node_names is None:
        node_names = [f"X{i+1}" for i in range(n)]
    elif len(node_names) != n:
        raise ValueError(
            f"node_names has length {len(node_names)}, but matrices have size {n}x{n}"
        )

    observed = set(np.unique(mats).tolist())
    codes = [c for c in DISPLAY_ORDER if c in observed]
    if not codes:
        raise ValueError("No endpoint codes found.")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Scale figure size with number of variables
    panel_size = 2.1
    fig_w = max(6, min(28, n * panel_size))
    fig_h = max(6, min(28, n * panel_size))

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,   # important for n=1
    )

    x = np.arange(len(codes))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, "—",
                    ha="center", va="center",
                    fontsize=18, color="0.65",
                    transform=ax.transAxes
                )
                continue

            vals = mats[:, i, j]
            counts = np.array([(vals == code).sum() for code in codes], dtype=float)
            total = counts.sum()
            freqs = counts / total if total > 0 else counts

            ax.bar(
                x,
                freqs,
                width=0.82,
                color=[EDGE_COLORS.get(c, "#cccccc") for c in codes],
                edgecolor="white",
                linewidth=0.5
            )

            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            ax.set_axisbelow(True)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.set_title(node_names[j], pad=8)

            if j == 0:
                ax.text(
                    -0.38, 0.5, node_names[i],
                    rotation=90,
                    va="center", ha="center",
                    transform=ax.transAxes,
                    fontsize=12
                )

            ax.set_xticks(x)
            if i == n - 1:
                ax.set_xticklabels([EDGE_LABELS.get(c, str(c)) for c in codes], rotation=90)
            else:
                ax.set_xticklabels([])

            if j != 0:
                ax.set_yticklabels([])

    handles = [
        Patch(
            facecolor=EDGE_COLORS.get(c, "#cccccc"),
            edgecolor="none",
            label=f"{c}: {EDGE_LABELS.get(c, str(c))}"
        )
        for c in codes
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(handles), 4),
        frameon=False
    )

    fig.suptitle(
        f"FCI endpoint distribution by matrix cell (n = {n_graphs} graphs, p = {n} variables)",
        fontsize=18,
        y=1.03
    )

    #fig.supxlabel("column j", fontsize=12)
    #fig.supylabel("row i", fontsize=12)

    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


@click.command()
@click.option("-i", "--input-database", required=True, help="Input DuckDB file")
@click.option("-t", "--table", required=True, help="Table name with adjacency matrices")
@click.option("-c", "--column", required=True, help="Column name with adjacency matrices")
@click.option("--labels-column", default="labels", show_default=True, help="Column with node labels")
@click.option("-o", "--outfile", default="fci_endpoint_hist_grid.png", show_default=True, help="Output image")
def visualize_bootstrapped_graph(input_database, table, column, labels_column, outfile):
    mats = load_matrices(input_database, table, column, WHERE)

    try:
        node_names = load_node_names(input_database, table, labels_column)
    except Exception:
        node_names = None

    plot_endpoint_hist_grid(mats, node_names=node_names, outfile=outfile)


if __name__ == "__main__":
    visualize_bootstrapped_graph()
