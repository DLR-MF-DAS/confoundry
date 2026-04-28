import click
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_table_name(conn, table_name=None):
    if table_name is not None:
        return table_name

    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()

    if len(tables) == 0:
        raise RuntimeError("No tables found in database.")
    if len(tables) > 1:
        raise RuntimeError(
            f"Multiple tables found: {tables}. Please provide --table-name explicitly."
        )
    return tables[0]


def shift_year_month(year, month, delta_months):
    """
    Shift (year, month) by delta_months.
    delta_months = -1 -> previous month
    delta_months = +1 -> next month
    """
    absolute = year * 12 + (month - 1) + delta_months
    new_year = absolute // 12
    new_month = absolute % 12 + 1
    return new_year, new_month


def load_month_data(database, year, month, table_name=None, allow_empty=False):
    conn = duckdb.connect(database)
    table_name = get_table_name(conn, table_name)

    df = conn.execute(
        f"""
        SELECT *
        FROM {table_name}
        WHERE year = ? AND month = ?
        ORDER BY row, col
        """,
        [year, month],
    ).fetchdf()

    conn.close()

    if df.empty and not allow_empty:
        raise RuntimeError(
            f"No data found for year={year}, month={month} in table '{table_name}'."
        )

    return df, table_name


def find_raster_columns(df):
    meta_cols = {"year", "month", "row", "col", "x", "y"}
    raster_cols = [c for c in df.columns if c not in meta_cols]
    if not raster_cols:
        raise RuntimeError("No raster value columns found in table.")
    return raster_cols


def parse_column_specs(columns):
    """
    Parse repeated --columns entries.

    Accepted forms:
      --columns ndvi
      --columns ndvi,1
      --columns lst,-1

    Semantics:
      t = 1   -> previous month
      t = 0   -> current month
      t = -1  -> next month
    """
    if not columns:
        return None

    specs = []
    seen = set()

    for item in columns:
        item = item.strip()
        if not item:
            continue

        parts = [p.strip() for p in item.split(",")]

        if len(parts) == 1:
            variable = parts[0]
            t = 0
        elif len(parts) == 2:
            variable = parts[0]
            try:
                t = int(parts[1])
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid lag in column spec '{item}'. Use variable,t with integer t."
                ) from e
        else:
            raise RuntimeError(
                f"Invalid column spec '{item}'. Use either 'variable' or 'variable,t'."
            )

        key = (variable, t)
        if key not in seen:
            specs.append(key)
            seen.add(key)

    return specs


def validate_column_specs(available_raster_cols, specs):
    if specs is None:
        return [(col, 0) for col in available_raster_cols]

    missing = [var for var, _ in specs if var not in available_raster_cols]
    if missing:
        raise RuntimeError(
            f"Requested columns not found: {missing}. "
            f"Available raster columns: {available_raster_cols}"
        )

    return specs


def dataframe_column_to_raster(df, value_col, shape):
    nrows, ncols = shape
    arr = np.full((nrows, ncols), np.nan, dtype=float)

    if df is None or df.empty:
        return arr

    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    arr[rows, cols] = vals
    return arr


def robust_limits(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return None, None

    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)

    if np.isclose(vmin, vmax):
        vmin = np.min(valid)
        vmax = np.max(valid)

    if np.isclose(vmin, vmax):
        return None, None

    return vmin, vmax


def lag_label(t):
    if t == 0:
        return "t=0"
    return f"t={t}"


def build_plot_items(database, table_name, year, month, column_specs):
    """
    Returns a list of dicts, one per panel:
      {
        "name": variable name,
        "lag": t,
        "source_year": source year,
        "source_month": source month,
        "array": 2D numpy array
      }
    """
    base_df, resolved_table = load_month_data(
        database, year, month, table_name=table_name, allow_empty=False
    )

    available_raster_cols = find_raster_columns(base_df)
    column_specs = validate_column_specs(available_raster_cols, column_specs)

    nrows = int(base_df["row"].max()) + 1
    ncols = int(base_df["col"].max()) + 1
    shape = (nrows, ncols)

    month_cache = {(year, month): base_df}
    items = []

    for variable, t in column_specs:
        # user semantics: t=1 means previous month, t=-1 means next month
        source_year, source_month = shift_year_month(year, month, -t)

        if (source_year, source_month) not in month_cache:
            df_src, _ = load_month_data(
                database,
                source_year,
                source_month,
                table_name=resolved_table,
                allow_empty=True,
            )
            month_cache[(source_year, source_month)] = df_src

        df_src = month_cache[(source_year, source_month)]
        arr = dataframe_column_to_raster(df_src, variable, shape=shape)

        items.append(
            {
                "name": variable,
                "lag": t,
                "source_year": source_year,
                "source_month": source_month,
                "array": arr,
                "has_data": not df_src.empty,
            }
        )

    return items, resolved_table, len(base_df)


def plot_rasters_and_histograms(items, year, month, bins=50):
    n = len(items)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(4.8 * n, 6.4),
        squeeze=False,
        layout="compressed",
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )

    fig.suptitle(
        f"Database rasters for requested month {year}-{month:02d}",
        fontsize=14,
    )

    for i, item in enumerate(items):
        arr = item["array"]
        col = item["name"]
        t = item["lag"]
        src_y = item["source_year"]
        src_m = item["source_month"]

        ax_img = axes[0, i]
        ax_hist = axes[1, i]

        vmin, vmax = robust_limits(arr)

        im = ax_img.imshow(arr, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax_img.set_title(
            f"{col} [{lag_label(t)}]  {src_y}-{src_m:02d}",
            fontsize=10,
            pad=2,
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # full-height, narrow colorbar
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.02, shrink=0.6)

        if not item["has_data"]:
            ax_img.text(
                0.5,
                0.03,
                "No data for source month",
                ha="center",
                va="bottom",
                transform=ax_img.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            ax_hist.text(0.5, 0.5, "No valid data", ha="center", va="center")
        else:
            ax_hist.hist(valid.ravel(), bins=bins)

        ax_hist.set_title(f"{col} histogram", fontsize=9, pad=2)
        ax_hist.set_xlabel("Value", fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)

    plt.show()


@click.command()
@click.option("-i", "--input-db", required=True, help="DuckDB database file")
@click.option("-y", "--year", required=True, type=int, help="Year")
@click.option("-m", "--month", required=True, type=int, help="Month (1-12)")
@click.option("-t", "--table-name", default=None, help="Table name in DuckDB")
@click.option(
    "-c",
    "--columns",
    multiple=True,
    help=(
        "Column specs to show. Use once per column. "
        "Accepted forms: --columns ndvi or --columns ndvi,1 or --columns ndvi,-1. "
        "Here t=1 means previous month and t=-1 means next month."
    ),
)
@click.option("--bins", default=50, show_default=True, type=int, help="Histogram bins")
def main(input_db, year, month, table_name, columns, bins):
    if month < 1 or month > 12:
        raise click.BadParameter("month must be between 1 and 12")

    column_specs = parse_column_specs(columns)

    items, resolved_table, row_count = build_plot_items(
        database=input_db,
        table_name=table_name,
        year=year,
        month=month,
        column_specs=column_specs,
    )

    shown = [f"{item['name']}[{lag_label(item['lag'])}]" for item in items]

    print(f"Using table: {resolved_table}")
    print(f"Base month rows loaded: {row_count}")
    print(f"Showing: {shown}")

    plot_rasters_and_histograms(items, year, month, bins=bins)


if __name__ == "__main__":
    main()
