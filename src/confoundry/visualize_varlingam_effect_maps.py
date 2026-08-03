"""Create publication maps from per-pixel VAR-LiNGAM effect estimates.

The command consumes the pixel-level tables written by
``per_pixel_varlingam_analysis``.  It does not refit graphs or effects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from confoundry.analysis_helpers import ensure_identifier
from confoundry.visualize_directlingam_diagnostics import (
    PUBLICATION_STYLE,
    parse_label_overrides,
    publication_variable_label,
    save_publication_figure,
)


DEFAULT_SOURCE_ORDER = (
    "temperature_resid",
    "precipitation_resid",
    "evaporation_resid",
    "soil_moisture_7_to_28_cm_resid",
    "soil_moisture_28_to_100_cm_resid",
)

EFFECT_COLUMNS = (
    "scaled_total_effect",
    "scaled_total_effect_boot_ci_low",
    "scaled_total_effect_boot_ci_high",
    "scaled_total_effect_boot_ci_excludes_zero",
    "scaled_cumulative_total_effect",
    "scaled_cumulative_total_effect_boot_ci_low",
    "scaled_cumulative_total_effect_boot_ci_high",
    "scaled_cumulative_total_effect_boot_ci_excludes_zero",
)


def parse_csv_values(value: str | None) -> list[str] | None:
    """Parse a comma-separated list while preserving its order."""
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(values)) or None


def parse_horizons(value: str) -> list[int]:
    """Parse a unique, comma-separated list of non-negative horizons."""
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            horizon = int(item)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid horizon {item!r}; expected comma-separated integers.",
                param_hint="--supplement-horizons",
            ) from exc
        if horizon < 0:
            raise click.BadParameter(
                "Horizons must be non-negative.",
                param_hint="--supplement-horizons",
            )
        if horizon not in result:
            result.append(horizon)
    if not result:
        raise click.BadParameter(
            "At least one supplement horizon is required.",
            param_hint="--supplement-horizons",
        )
    return result


def ordered_sources(values: Sequence[Any]) -> list[str]:
    """Return known drought drivers first and other sources alphabetically."""
    available = list(dict.fromkeys(str(value) for value in values))
    preferred = [value for value in DEFAULT_SOURCE_ORDER if value in available]
    remaining = sorted(value for value in available if value not in preferred)
    return preferred + remaining


def coerce_boolean(series: pd.Series) -> pd.Series:
    """Coerce common CSV boolean representations to nullable booleans."""
    if isinstance(series.dtype, pd.BooleanDtype) or series.dtype == bool:
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series):
        return series.map({1: True, 0: False}).astype("boolean")
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).astype("boolean")


def coordinate_domain(
    effects: pd.DataFrame,
    qc: pd.DataFrame | None,
    row_column: str,
    col_column: str,
) -> tuple[list[Any], list[Any]]:
    """Return the complete sorted map coordinate domain."""
    source = qc if qc is not None else effects
    rows = sorted(source[row_column].dropna().unique().tolist())
    cols = sorted(source[col_column].dropna().unique().tolist())
    if not rows or not cols:
        raise click.ClickException("No finite map coordinates were found.")
    return rows, cols


def grid_from_frame(
    frame: pd.DataFrame,
    *,
    row_column: str,
    col_column: str,
    value_column: str,
    rows: Sequence[Any],
    cols: Sequence[Any],
) -> np.ndarray:
    """Build a dense map grid on an explicitly supplied coordinate domain."""
    if frame.duplicated([row_column, col_column]).any():
        raise click.ClickException(
            f"Duplicate ({row_column}, {col_column}) rows cannot be mapped."
        )
    pivot = frame.pivot(
        index=row_column,
        columns=col_column,
        values=value_column,
    )
    return pivot.reindex(index=rows, columns=cols).to_numpy(dtype=float)


def symmetric_color_limit(values: Sequence[float], quantile: float) -> float:
    """Return a robust, positive, symmetric color limit."""
    array = np.asarray(values, dtype=float).ravel()
    array = np.abs(array[np.isfinite(array)])
    if not len(array):
        raise click.ClickException("No finite effect estimates were found.")
    limit = float(np.quantile(array, quantile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(array))
    return limit if limit > 0.0 else 1.0


def _validate_columns(
    available: set[str],
    required: Sequence[str],
    source: str,
) -> None:
    missing = sorted(set(required) - available)
    if missing:
        raise click.ClickException(
            f"{source} is missing required columns: {missing}"
        )


def load_effects(
    db_path: Path,
    table: str,
    *,
    row_column: str,
    col_column: str,
    target: str | None,
    sources: Sequence[str] | None,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Read only effect columns needed for the requested maps."""
    table_sql = ensure_identifier(table)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        if table not in tables:
            raise click.ClickException(
                f"Table {table!r} was not found in {db_path}. "
                f"Available tables: {sorted(tables)}"
            )
        available = set(
            con.execute(f"DESCRIBE {table_sql}").fetchdf()["column_name"]
        )
        required = [
            row_column,
            col_column,
            "source",
            "target",
            "horizon",
            *EFFECT_COLUMNS,
        ]
        _validate_columns(available, required, f"{db_path}::{table}")

        selected = required.copy()
        selected_sql = ", ".join(ensure_identifier(col) for col in selected)
        conditions = [
            "horizon IN (" + ", ".join("?" for _ in horizons) + ")"
        ]
        parameters: list[Any] = [int(value) for value in horizons]
        if target is not None:
            conditions.append("target = ?")
            parameters.append(target)
        if sources:
            conditions.append(
                "source IN (" + ", ".join("?" for _ in sources) + ")"
            )
            parameters.extend(sources)
        if "error" in available:
            conditions.append("error IS NULL")
        query = (
            f"SELECT {selected_sql} FROM {table_sql} WHERE "
            + " AND ".join(conditions)
        )
        effects = con.execute(query, parameters).fetchdf()
    finally:
        con.close()

    if effects.empty:
        raise click.ClickException(
            "No per-pixel effects matched the requested target, sources, "
            "and horizons."
        )
    duplicate_cols = [row_column, col_column, "source", "target", "horizon"]
    if effects.duplicated(duplicate_cols).any():
        raise click.ClickException(
            "The effect table has duplicate pixel/source/target/horizon rows."
        )
    return effects


def load_qc(
    path: Path | None,
    *,
    row_column: str,
    col_column: str,
    primary_column: str,
) -> pd.DataFrame | None:
    """Load the optional full-footprint quality-control table."""
    if path is None:
        return None
    qc = pd.read_csv(path)
    _validate_columns(
        set(qc.columns),
        [row_column, col_column, primary_column],
        str(path),
    )
    if qc.duplicated([row_column, col_column]).any():
        raise click.ClickException(
            f"{path} has duplicate ({row_column}, {col_column}) rows."
        )
    qc = qc.copy()
    qc[primary_column] = coerce_boolean(qc[primary_column])
    return qc


def _footprint_grid(
    effects: pd.DataFrame,
    qc: pd.DataFrame | None,
    *,
    row_column: str,
    col_column: str,
    rows: Sequence[Any],
    cols: Sequence[Any],
) -> np.ndarray:
    footprint = (
        qc[[row_column, col_column]].copy()
        if qc is not None
        else effects[[row_column, col_column]].drop_duplicates().copy()
    )
    footprint["_footprint"] = 1.0
    return grid_from_frame(
        footprint,
        row_column=row_column,
        col_column=col_column,
        value_column="_footprint",
        rows=rows,
        cols=cols,
    )


def _plot_effect_panel(
    axis: plt.Axes,
    subset: pd.DataFrame,
    *,
    value_column: str,
    support_column: str,
    row_column: str,
    col_column: str,
    rows: Sequence[Any],
    cols: Sequence[Any],
    footprint_grid: np.ndarray,
    norm: TwoSlopeNorm,
    stipple_stride: int,
) -> Any:
    excluded_cmap = ListedColormap(["#D3D3D3"]).with_extremes(
        bad=(1.0, 1.0, 1.0, 0.0)
    )
    axis.imshow(
        np.ma.masked_invalid(footprint_grid),
        origin="upper",
        interpolation="nearest",
        cmap=excluded_cmap,
        vmin=0.0,
        vmax=1.0,
    )

    value_grid = grid_from_frame(
        subset,
        row_column=row_column,
        col_column=col_column,
        value_column=value_column,
        rows=rows,
        cols=cols,
    )
    effect_cmap = plt.get_cmap("RdBu_r").with_extremes(
        bad=(1.0, 1.0, 1.0, 0.0)
    )
    image = axis.imshow(
        np.ma.masked_invalid(value_grid),
        origin="upper",
        interpolation="nearest",
        cmap=effect_cmap,
        norm=norm,
    )

    support_frame = subset[[row_column, col_column, support_column]].copy()
    support_frame["_supported"] = coerce_boolean(
        support_frame[support_column]
    ).fillna(False).astype(float)
    support_grid = grid_from_frame(
        support_frame,
        row_column=row_column,
        col_column=col_column,
        value_column="_supported",
        rows=rows,
        cols=cols,
    )
    support_rows, support_cols = np.where(support_grid == 1.0)
    keep = (
        (support_rows % stipple_stride == 0)
        & (support_cols % stipple_stride == 0)
    )
    if np.any(keep):
        axis.scatter(
            support_cols[keep],
            support_rows[keep],
            marker=".",
            s=3.0,
            linewidths=0.0,
            color="black",
            alpha=0.60,
            rasterized=True,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    return image


def plot_effect_small_multiples(
    effects: pd.DataFrame,
    *,
    sources: Sequence[str],
    target: str,
    horizon: int,
    value_column: str,
    support_column: str,
    row_column: str,
    col_column: str,
    qc: pd.DataFrame | None,
    output_path: Path,
    color_limit: float,
    stipple_stride: int,
    label_overrides: Mapping[str, str] | None,
    cumulative: bool,
) -> list[Path]:
    """Write one common-scale, multi-source spatial effect figure."""
    subset = effects[
        (effects["target"] == target)
        & (effects["horizon"] == horizon)
        & effects["source"].isin(sources)
    ].copy()
    if subset.empty:
        return []
    rows, cols = coordinate_domain(
        subset,
        qc,
        row_column,
        col_column,
    )
    footprint_grid = _footprint_grid(
        subset,
        qc,
        row_column=row_column,
        col_column=col_column,
        rows=rows,
        cols=cols,
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    ncols = min(3, len(sources))
    nrows = int(math.ceil(len(sources) / ncols))

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.1 * ncols, 3.6 * nrows),
            squeeze=False,
        )
        image = None
        for panel_index, source in enumerate(sources):
            axis = axes.flat[panel_index]
            source_subset = subset[subset["source"] == source]
            image = _plot_effect_panel(
                axis,
                source_subset,
                value_column=value_column,
                support_column=support_column,
                row_column=row_column,
                col_column=col_column,
                rows=rows,
                cols=cols,
                footprint_grid=footprint_grid,
                norm=norm,
                stipple_stride=stipple_stride,
            )
            panel_letter = chr(ord("a") + panel_index)
            source_label = publication_variable_label(
                source,
                label_overrides,
            )
            axis.set_title(f"({panel_letter}) {source_label}")
        for axis in axes.flat[len(sources) :]:
            axis.set_visible(False)

        target_label = publication_variable_label(target, label_overrides)
        if cumulative:
            title = (
                f"Cumulative effect on {target_label}, months 0–{horizon}"
            )
        else:
            title = f"Effect on {target_label}, month {horizon}"
        fig.suptitle(title, y=0.99)
        if image is not None:
            colorbar_axis = fig.add_axes([0.91, 0.23, 0.018, 0.54])
            colorbar = fig.colorbar(
                image,
                cax=colorbar_axis,
                extend="both",
            )
            colorbar.set_label("Target IQR per source IQR")
        legend_handles = [
            Patch(facecolor="#D3D3D3", label="Excluded by primary QC"),
            Line2D(
                [],
                [],
                color="black",
                marker=".",
                linestyle="None",
                markersize=5,
                label="95% bootstrap interval excludes zero",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
        )
        fig.subplots_adjust(
            left=0.02,
            right=0.87,
            bottom=0.10,
            top=0.87,
            wspace=0.06,
            hspace=0.18,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_publication_figure(fig, output_path)
        plt.close(fig)
    return [output_path, output_path.with_suffix(".pdf")]


def plot_qc_coverage(
    qc: pd.DataFrame,
    *,
    row_column: str,
    col_column: str,
    primary_column: str,
    output_path: Path,
) -> list[Path]:
    """Write a two-category map of the final paper-analysis population."""
    rows, cols = coordinate_domain(qc, qc, row_column, col_column)
    work = qc[[row_column, col_column, primary_column]].copy()
    work["_coverage"] = coerce_boolean(work[primary_column]).astype(float)
    grid = grid_from_frame(
        work,
        row_column=row_column,
        col_column=col_column,
        value_column="_coverage",
        rows=rows,
        cols=cols,
    )
    eligible = int(work["_coverage"].eq(1.0).sum())
    total = int(work["_coverage"].notna().sum())
    excluded = total - eligible
    percent = 100.0 * eligible / total if total else float("nan")
    cmap = ListedColormap(["#D3D3D3", "#2C7FB8"]).with_extremes(
        bad=(1.0, 1.0, 1.0, 0.0)
    )

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axis = plt.subplots(figsize=(6.8, 5.8))
        axis.imshow(
            np.ma.masked_invalid(grid),
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=1.5,
        )
        axis.set_title(
            f"Primary analysis coverage: {eligible:,}/{total:,} "
            f"pixels ({percent:.2f}%)"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        axis.legend(
            handles=[
                Patch(facecolor="#2C7FB8", label=f"Retained ({eligible:,})"),
                Patch(facecolor="#D3D3D3", label=f"Excluded ({excluded:,})"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_publication_figure(fig, output_path)
        plt.close(fig)
    return [output_path, output_path.with_suffix(".pdf")]


def summarize_main_map(
    effects: pd.DataFrame,
    *,
    target: str,
    sources: Sequence[str],
    horizon: int,
) -> pd.DataFrame:
    """Return numerical source summaries matching the main map."""
    selected = effects[
        (effects["target"] == target)
        & (effects["horizon"] == horizon)
        & effects["source"].isin(sources)
    ].copy()
    rows: list[dict[str, Any]] = []
    for source in sources:
        group = selected[selected["source"] == source]
        values = pd.to_numeric(
            group["scaled_cumulative_total_effect"],
            errors="coerce",
        ).dropna()
        supported = coerce_boolean(
            group[
                "scaled_cumulative_total_effect_boot_ci_excludes_zero"
            ]
        ).dropna()
        rows.append(
            {
                "source": source,
                "target": target,
                "horizon": int(horizon),
                "n_pixels": int(len(values)),
                "effect_median": float(values.median()),
                "effect_q05": float(values.quantile(0.05)),
                "effect_q95": float(values.quantile(0.95)),
                "fraction_positive": float((values > 0.0).mean()),
                "fraction_bootstrap_supported": float(supported.mean()),
            }
        )
    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--effects-db",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--effects-table",
    default="pixel_varlingam_effects_primary",
    show_default=True,
)
@click.option(
    "--qc-csv",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Full-footprint pixel QC CSV used to draw excluded pixels.",
)
@click.option(
    "--qc-primary-column",
    default="primary_eligible",
    show_default=True,
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option("--target", default=None)
@click.option(
    "--sources",
    default=None,
    help="Comma-separated source order; defaults to all available sources.",
)
@click.option("--main-horizon", default=12, show_default=True, type=int)
@click.option(
    "--supplement-horizons",
    default="0,1,3,6,12",
    show_default=True,
)
@click.option(
    "--color-quantile",
    default=0.98,
    show_default=True,
    type=click.FloatRange(0.5, 1.0),
    help="Absolute-effect quantile used for symmetric color limits.",
)
@click.option(
    "--stipple-stride",
    default=2,
    show_default=True,
    type=click.IntRange(1, None),
)
@click.option("--row-column", default="row", show_default=True)
@click.option("--col-column", default="col", show_default=True)
@click.option(
    "--variable-label",
    "variable_labels",
    multiple=True,
    metavar="RAW=DISPLAY",
)
@click.option("--no-supplement", is_flag=True)
def visualize_varlingam_effect_maps(
    effects_db: Path,
    effects_table: str,
    qc_csv: Path | None,
    qc_primary_column: str,
    output_dir: Path,
    target: str | None,
    sources: str | None,
    main_horizon: int,
    supplement_horizons: str,
    color_quantile: float,
    stipple_stride: int,
    row_column: str,
    col_column: str,
    variable_labels: tuple[str, ...],
    no_supplement: bool,
) -> None:
    """Plot paper-ready spatial maps of per-pixel VAR-LiNGAM effects."""
    if main_horizon < 0:
        raise click.BadParameter(
            "Main horizon must be non-negative.",
            param_hint="--main-horizon",
        )
    requested_sources = parse_csv_values(sources)
    supplement = parse_horizons(supplement_horizons)
    horizons = list(dict.fromkeys([main_horizon, *supplement]))
    effects = load_effects(
        effects_db,
        effects_table,
        row_column=row_column,
        col_column=col_column,
        target=target,
        sources=requested_sources,
        horizons=horizons,
    )
    available_targets = sorted(effects["target"].dropna().astype(str).unique())
    if target is None:
        if len(available_targets) != 1:
            raise click.ClickException(
                "The effect table contains multiple targets; pass --target."
            )
        target = available_targets[0]
    available_sources = ordered_sources(
        effects.loc[effects["target"] == target, "source"].dropna().unique()
    )
    selected_sources = requested_sources or available_sources
    missing_sources = sorted(set(selected_sources) - set(available_sources))
    if missing_sources:
        raise click.ClickException(
            f"Requested sources are absent from the selected effects: "
            f"{missing_sources}"
        )
    qc = load_qc(
        qc_csv,
        row_column=row_column,
        col_column=col_column,
        primary_column=qc_primary_column,
    )
    label_overrides = parse_label_overrides(variable_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    main = effects[
        (effects["target"] == target)
        & (effects["horizon"] == main_horizon)
        & effects["source"].isin(selected_sources)
    ].copy()
    main_limit = symmetric_color_limit(
        main["scaled_cumulative_total_effect"],
        color_quantile,
    )
    written = plot_effect_small_multiples(
        effects,
        sources=selected_sources,
        target=target,
        horizon=main_horizon,
        value_column="scaled_cumulative_total_effect",
        support_column=(
            "scaled_cumulative_total_effect_boot_ci_excludes_zero"
        ),
        row_column=row_column,
        col_column=col_column,
        qc=qc,
        output_path=figures_dir / "cumulative_effects_main.png",
        color_limit=main_limit,
        stipple_stride=stipple_stride,
        label_overrides=label_overrides,
        cumulative=True,
    )

    main_export_columns = [
        row_column,
        col_column,
        "source",
        "target",
        "horizon",
        "scaled_cumulative_total_effect",
        "scaled_cumulative_total_effect_boot_ci_low",
        "scaled_cumulative_total_effect_boot_ci_high",
        "scaled_cumulative_total_effect_boot_ci_excludes_zero",
    ]
    main[main_export_columns].to_csv(
        tables_dir / "cumulative_effects_main_data.csv",
        index=False,
    )
    summary = summarize_main_map(
        effects,
        target=target,
        sources=selected_sources,
        horizon=main_horizon,
    )
    summary.to_csv(tables_dir / "cumulative_effects_main_summary.csv", index=False)

    metadata_rows = [
        {
            "figure": "cumulative_effects_main",
            "effect": "scaled_cumulative_total_effect",
            "horizon": int(main_horizon),
            "color_quantile": float(color_quantile),
            "symmetric_color_limit": float(main_limit),
            "stipple_stride": int(stipple_stride),
            "n_pixels": int(
                main[[row_column, col_column]].drop_duplicates().shape[0]
            ),
        }
    ]

    if qc is not None:
        written.extend(
            plot_qc_coverage(
                qc,
                row_column=row_column,
                col_column=col_column,
                primary_column=qc_primary_column,
                output_path=figures_dir / "primary_qc_coverage.png",
            )
        )

    if not no_supplement:
        supplement_data = effects[
            (effects["target"] == target)
            & effects["horizon"].isin(supplement)
            & effects["source"].isin(selected_sources)
        ].copy()
        supplement_limit = symmetric_color_limit(
            supplement_data["scaled_total_effect"],
            color_quantile,
        )
        for horizon in supplement:
            written.extend(
                plot_effect_small_multiples(
                    effects,
                    sources=selected_sources,
                    target=target,
                    horizon=horizon,
                    value_column="scaled_total_effect",
                    support_column=(
                        "scaled_total_effect_boot_ci_excludes_zero"
                    ),
                    row_column=row_column,
                    col_column=col_column,
                    qc=qc,
                    output_path=(
                        figures_dir
                        / f"horizon_effects_month_{horizon:02d}.png"
                    ),
                    color_limit=supplement_limit,
                    stipple_stride=stipple_stride,
                    label_overrides=label_overrides,
                    cumulative=False,
                )
            )
            metadata_rows.append(
                {
                    "figure": f"horizon_effects_month_{horizon:02d}",
                    "effect": "scaled_total_effect",
                    "horizon": int(horizon),
                    "color_quantile": float(color_quantile),
                    "symmetric_color_limit": float(supplement_limit),
                    "stipple_stride": int(stipple_stride),
                    "n_pixels": int(
                        supplement_data.loc[
                            supplement_data["horizon"] == horizon,
                            [row_column, col_column],
                        ].drop_duplicates().shape[0]
                    ),
                }
            )
        supplement_export_columns = [
            row_column,
            col_column,
            "source",
            "target",
            "horizon",
            "scaled_total_effect",
            "scaled_total_effect_boot_ci_low",
            "scaled_total_effect_boot_ci_high",
            "scaled_total_effect_boot_ci_excludes_zero",
        ]
        supplement_data[supplement_export_columns].to_csv(
            tables_dir / "horizon_effects_supplement_data.csv",
            index=False,
        )

    pd.DataFrame(metadata_rows).to_csv(
        tables_dir / "effect_map_figure_metadata.csv",
        index=False,
    )
    click.echo(f"Wrote main per-pixel effect maps: {figures_dir}")
    click.echo(f"Wrote matching figure-data tables: {tables_dir}")
    click.echo(
        f"Target: {target}; sources: {len(selected_sources)}; "
        f"main horizon: {main_horizon}; mapped pixels: "
        f"{main[[row_column, col_column]].drop_duplicates().shape[0]}"
    )
    click.echo(f"Figure files written: {len(written)}")


if __name__ == "__main__":
    visualize_varlingam_effect_maps()
