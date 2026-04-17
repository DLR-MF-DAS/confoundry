import duckdb
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Optional SQL filter, e.g. "split = 'test'"
WHERE = None

# Optional names for the 8 variables
NODE_NAMES = [f"X{i+1}" for i in range(8)]

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

# A fixed display order makes panels comparable
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
    query = f"SELECT {column} FROM {table}"
    if where:
        query += f" WHERE {where}"

    rows = con.execute(query).fetchall()
    con.close()

    if not rows:
        raise ValueError("Query returned no rows.")

    mats = [np.asarray(row[0], dtype=int) for row in rows]

    # sanity checks
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"Found multiple shapes: {shapes}")
    if mats[0].shape != (8, 8):
        raise ValueError(f"Expected (8, 8), got {mats[0].shape}")

    return np.stack(mats, axis=0)   # shape: (n_rows, 8, 8)

def plot_endpoint_hist_grid(mats, node_names=None, outfile="fci_endpoint_hist_grid.png"):
    n_graphs, n, m = mats.shape
    if (n, m) != (8, 8):
        raise ValueError(f"Expected mats.shape[1:] == (8, 8), got {(n, m)}")

    if node_names is None:
        node_names = [f"X{i+1}" for i in range(n)]

    # Only show codes that actually occur, but keep a stable order
    observed = set(np.unique(mats).tolist())
    codes = [c for c in DISPLAY_ORDER if c in observed]
    if not codes:
        raise ValueError("No endpoint codes found.")

    # Typography tuned for presentation output
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(
        8, 8,
        figsize=(18, 18),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    x = np.arange(len(codes))

    for i in range(8):
        for j in range(8):
            ax = axes[i, j]

            # Hide diagonal
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
            freqs = counts / counts.sum() if counts.sum() > 0 else counts

            ax.bar(
                x,
                freqs,
                width=0.82,
                color=[EDGE_COLORS[c] for c in codes],
                edgecolor="white",
                linewidth=0.5
            )

            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            ax.set_axisbelow(True)

            # Clean spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Column labels on top row
            if i == 0:
                ax.set_title(node_names[j], pad=8)

            # Row labels on first column
            if j == 0:
                ax.text(
                    -0.38, 0.5, node_names[i],
                    rotation=90,
                    va="center", ha="center",
                    transform=ax.transAxes,
                    fontsize=12
                )

            # Only bottom row gets x tick labels
            ax.set_xticks(x)
            if i == 7:
                ax.set_xticklabels([EDGE_LABELS[c] for c in codes], rotation=90)
            else:
                ax.set_xticklabels([])

            # Only first column gets y ticks
            if j != 0:
                ax.set_yticklabels([])

    handles = [
        Patch(facecolor=EDGE_COLORS[c], edgecolor="none", label=f"{c}: {EDGE_LABELS[c]}")
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
        f"FCI endpoint distribution by matrix cell (n = {n_graphs} graphs)",
        fontsize=18,
        y=1.03
    )

    fig.supxlabel("column j", fontsize=12)
    fig.supylabel("row i", fontsize=12)

    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

@click.command()
@click.option('-i', '--input-database', help='Input DuckDB file')
@click.option('-t', '--table', help='Table name with adjacency matrices')
@click.option('-c', '--column', help='Column name with adjacency matrices')
def visualize_bootstrapped_graph(input_database, table, column):
    # -----------------------------
    # Run
    # -----------------------------
    con = duckdb.connect(input_database, read_only=True)
    NODE_NAMES = list(con.execute(f"SELECT labels FROM {table} LIMIT 1").fetchone()[0])
    breakpoint()
    con.close()
    mats = load_matrices(input_database, table, column, None)
    plot_endpoint_hist_grid(mats, node_names=NODE_NAMES, outfile="fci_endpoint_hist_grid.png")

if __name__ == '__main__':
    visualize_bootstrapped_graph()
