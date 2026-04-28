import json
import click
import duckdb
import numpy as np
import matplotlib.pyplot as plt


@click.command()
@click.option("-i", "--input-db", required=True, help="Output DB from the graph script")
@click.option("-t", "--table", default="pixel_graphs", show_default=True)
@click.option("--row-col-cols", nargs=2, default=("row", "col"), show_default=True)
@click.option(
    "--label",
    "labels",
    multiple=True,
    required=True,
    help="Variable label to include. Repeat for multiple labels, e.g. --label A --label B --label C",
)
@click.option("-o", "--output", required=True, help="Output PNG file")
@click.option(
    "--figscale",
    default=3.0,
    show_default=True,
    type=float,
    help="Size multiplier per subplot",
)
def arrow_probability_matrix(input_db, table, row_col_cols, labels, output, figscale):
    """
    Build an N x N matrix of rasters, where subplot (i, j) shows
    P(labels[i] -> labels[j]) across the spatial grid.
    """
    labels = list(labels)
    row_col_cols = list(row_col_cols)
    row_name, col_name = row_col_cols
    row_col_sql = ", ".join(row_col_cols)

    if len(labels) < 2:
        raise click.ClickException("Please provide at least two --label values.")

    con = duckdb.connect(input_db, read_only=True)
    df = con.execute(
        f"""
        SELECT {row_col_sql}, variable_index_json, edge_probability_json
        FROM {table}
        """
    ).fetchdf()
    con.close()

    if df.empty:
        raise click.ClickException("Input table is empty.")

    all_rows = sorted(df[row_name].unique())
    all_cols = sorted(df[col_name].unique())
    row_to_idx = {v: i for i, v in enumerate(all_rows)}
    col_to_idx = {v: i for i, v in enumerate(all_cols)}

    n_labels = len(labels)
    n_rows = len(all_rows)
    n_cols = len(all_cols)

    rasters = np.full((n_labels, n_labels, n_rows, n_cols), np.nan, dtype=float)

    for _, r in df.iterrows():
        index_map = json.loads(r["variable_index_json"])
        prob_mat = np.asarray(json.loads(r["edge_probability_json"]), dtype=float)

        missing = [lab for lab in labels if lab not in index_map]
        if missing:
            raise click.ClickException(
                f"Labels not found in variable_index_json for row "
                f"({row_name}={r[row_name]}, {col_name}={r[col_name]}): {missing}"
            )

        rr = row_to_idx[r[row_name]]
        cc = col_to_idx[r[col_name]]
        indices = [index_map[lab] for lab in labels]

        for i, src_idx in enumerate(indices):
            for j, dst_idx in enumerate(indices):
                if i == j:
                    continue
                rasters[i, j, rr, cc] = float(prob_mat[dst_idx, src_idx])

    fig, axes = plt.subplots(
        n_labels,
        n_labels,
        figsize=(figscale * n_labels + 1.0, figscale * n_labels),
        squeeze=False,
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    last_im = None

    for i, src_label in enumerate(labels):
        for j, dst_label in enumerate(labels):
            ax = axes[i, j]

            if i == j:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=16,
                    transform=ax.transAxes,
                )
                continue

            arr = rasters[i, j]
            masked = np.ma.masked_invalid(arr)

            last_im = ax.imshow(
                masked,
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                cmap=cmap,
                aspect="auto",
            )

            ax.set_title(f"{src_label} → {dst_label}", fontsize=10)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    fig.suptitle("Causal arrow probability matrix", fontsize=14)

    # Leave room on the right for a full-height colorbar.
    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.06,
        top=0.92,
        wspace=0.15,
        hspace=0.25,
    )

    if last_im is not None:
        # [left, bottom, width, height] in figure coordinates
        cax = fig.add_axes([0.90, 0.06, 0.02, 0.86])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Arrow probability")

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    arrow_probability_matrix()
