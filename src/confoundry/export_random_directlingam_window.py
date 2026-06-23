"""Export the exact per-pixel/window data matrix used by DirectLiNGAM.

The command applies the same configured temporal shifts and complete-case
filtering as ``per_pixel_graph_discovery.py``, randomly selects one center
pixel whose complete square neighborhood is available, and writes the pooled
pixel/window observations to one CSV file.

Run repeatedly with different ``--seed`` values to export different points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import numpy as np
import pandas as pd
import yaml

from confoundry.per_pixel_graph_discovery import (
    get_pixel_window_group,
    parse_columns,
    quote_identifier,
)

PixelKey = tuple[int, int]


def expected_window_keys(pixel_key: PixelKey, window_size: int) -> list[PixelKey]:
    """Return all pixel keys in the square neighborhood around ``pixel_key``."""
    row, col = pixel_key
    return [
        (r, c)
        for r in range(row - window_size, row + window_size + 1)
        for c in range(col - window_size, col + window_size + 1)
    ]


def has_complete_window(
    pixel_key: PixelKey,
    group_lookup: Mapping[PixelKey, pd.DataFrame],
    window_size: int,
    labels: Sequence[str],
    min_samples: int,
    require_equal_time_index: bool,
    order_cols: Sequence[str],
) -> bool:
    """Check whether every pixel and required value in a window is available."""
    keys = expected_window_keys(pixel_key, window_size)

    if any(key not in group_lookup for key in keys):
        return False

    groups = [group_lookup[key] for key in keys]
    if any(group[list(labels)].isna().any().any() for group in groups):
        return False

    if require_equal_time_index:
        reference = set(
            map(tuple, groups[0][list(order_cols)].drop_duplicates().to_numpy())
        )
        for group in groups[1:]:
            current = set(
                map(tuple, group[list(order_cols)].drop_duplicates().to_numpy())
            )
            if current != reference:
                return False

    pooled_samples = sum(len(group) for group in groups)
    return pooled_samples >= min_samples


def select_random_pixel(
    candidates: Sequence[PixelKey],
    seed: int | None,
) -> PixelKey:
    """Select one candidate pixel reproducibly when a seed is supplied."""
    if not candidates:
        raise click.ClickException(
            "No center pixel has a complete window satisfying the requested "
            "availability and sample constraints."
        )

    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


@click.command()
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Experiment YAML configuration used by graph discovery.",
)
@click.option(
    "--window-size",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "Neighborhood radius, matching graph discovery. A value of 1 gives "
        "a 3x3 window; 2 gives a 5x5 window."
    ),
)
@click.option(
    "--min-samples",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help="Minimum pooled sample count, matching graph discovery.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed. Change it to select another reproducible point.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Output CSV path. By default, a filename containing the selected "
        "row, column, window size, and seed is written beside the config."
    ),
)
@click.option(
    "--require-equal-time-index/--allow-different-time-index",
    default=True,
    show_default=True,
    help=(
        "Require every pixel in the window to contain the same year/month "
        "observations after shifting and complete-case filtering."
    ),
)
@click.option(
    "--include-metadata/--variables-only",
    default=True,
    show_default=True,
    help="Include row, col, year, and month alongside the model variables.",
)
def export_random_directlingam_window(
    config_path: Path,
    window_size: int,
    min_samples: int,
    seed: int | None,
    output_path: Path | None,
    require_equal_time_index: bool,
    include_metadata: bool,
) -> None:
    """Write one random complete DirectLiNGAM pixel/window matrix to CSV."""
    row_col_cols = ["row", "col"]
    order_cols = ["year", "month"]

    with config_path.open("r", encoding="utf-8") as fd:
        config_data: dict[str, Any] = yaml.safe_load(fd)

    experiment_dir = config_path.parent
    experiment_name = str(config_data["name"])
    input_db = experiment_dir / f"{experiment_name}_ard.duckdb"
    input_table = experiment_name
    column_specs = config_data["columns"]

    if not input_db.exists():
        raise click.ClickException(f"Input database does not exist: {input_db}")

    con = duckdb.connect(input_db, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if input_table not in tables:
            raise click.ClickException(
                f"{input_table!r} not found in {input_db}. "
                f"Available tables: {sorted(tables)}"
            )
        df = con.execute(
            f"SELECT * FROM {quote_identifier(input_table)}"
        ).fetchdf()
    finally:
        con.close()

    required = row_col_cols + order_cols
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise click.ClickException(
            f"Input table is missing required columns: {missing_required}"
        )

    shifted_df, labels, label_lags = parse_columns(
        df=df,
        group_cols=row_col_cols,
        order_cols=order_cols,
        column_specs=column_specs,
    )
    shifted_df = shifted_df.dropna(
        subset=list(labels) + row_col_cols + order_cols
    )

    groups = list(shifted_df.groupby(row_col_cols, sort=True))
    group_lookup: dict[PixelKey, pd.DataFrame] = {
        (int(pixel_key[0]), int(pixel_key[1])): group.copy()
        for pixel_key, group in groups
    }

    candidates = [
        pixel_key
        for pixel_key in group_lookup
        if has_complete_window(
            pixel_key=pixel_key,
            group_lookup=group_lookup,
            window_size=window_size,
            labels=labels,
            min_samples=min_samples,
            require_equal_time_index=require_equal_time_index,
            order_cols=order_cols,
        )
    ]

    selected_key = select_random_pixel(candidates, seed=seed)

    window_group = get_pixel_window_group(
        pixel_key=selected_key,
        group_lookup=group_lookup,
        window_size=window_size,
    )
    if window_group is None:
        raise click.ClickException(
            f"Unexpectedly failed to construct window for {selected_key}."
        )

    complete_group = window_group.dropna(subset=list(labels)).copy()
    complete_group = complete_group.sort_values(
        row_col_cols + order_cols
    ).reset_index(drop=True)

    if len(complete_group) < min_samples:
        raise click.ClickException(
            f"Selected window has only {len(complete_group)} samples; "
            f"{min_samples} are required."
        )

    export_columns = (
        row_col_cols + order_cols + list(labels)
        if include_metadata
        else list(labels)
    )
    export_df = complete_group[export_columns].copy()
    export_df.insert(0, "row_id", np.arange(1, len(export_df) + 1, dtype=np.int64))

    row, col = selected_key
    seed_token = "random" if seed is None else str(seed)
    if output_path is None:
        output_path = experiment_dir / (
            f"{experiment_name}_directlingam_input_"
            f"row{row}_col{col}_window{window_size}_seed{seed_token}.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "experiment": experiment_name,
        "config_path": str(config_path),
        "input_db": str(input_db),
        "input_table": input_table,
        "selected_center_row": row,
        "selected_center_col": col,
        "window_size": window_size,
        "window_shape": [2 * window_size + 1, 2 * window_size + 1],
        "window_pixel_count": (2 * window_size + 1) ** 2,
        "n_exported_rows": int(len(export_df)),
        "row_id_column": "row_id",
        "row_id_definition": "Sequential 1-based identifier after final sorting and filtering",
        "model_variables": list(labels),
        "configured_shifts": {
            str(name): int(lag) for name, lag in label_lags.items()
        },
        "min_samples": min_samples,
        "seed": seed,
        "require_equal_time_index": require_equal_time_index,
        "includes_metadata_columns": include_metadata,
        "csv_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    click.echo(f"Selected center pixel: row={row}, col={col}")
    click.echo(
        f"Window: {(2 * window_size + 1)}x"
        f"{(2 * window_size + 1)} pixels"
    )
    click.echo(f"Exported observations: {len(export_df)}")
    click.echo(f"CSV: {output_path}")
    click.echo(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    export_random_directlingam_window()
