"""Compare per-pixel DirectLiNGAM analysis outputs with land-cover classes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from confoundry.landcover_graph_validation import (
    CLASS_SETS,
    WORLD_COVER_VERSION,
    WORLD_COVER_YEAR,
    download_worldcover,
    graph_domain_bounds_wgs84,
    label_graph_footprints,
    load_graph_rows,
    locate_reference_raster,
    read_config,
    required_worldcover_tiles,
    require_files,
    write_dataframe_table,
)
from confoundry.per_pixel_directlingam_analysis import (
    Config as DirectLiNGAMConfig,
    load_config as load_directlingam_config,
)


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    output_db: Path
    samples_csv: Path
    class_summary_csv: Path
    correlations_csv: Path
    summary_json: Path


def _default_output_paths(cfg: DirectLiNGAMConfig, output_dir: Path | None) -> OutputPaths:
    resolved_dir = (
        output_dir
        if output_dir is not None
        else cfg.experiment_dir / "landcover_directlingam_analysis"
    )
    return OutputPaths(
        output_dir=resolved_dir,
        output_db=resolved_dir / f"{cfg.location_name}_landcover_directlingam_analysis.duckdb",
        samples_csv=resolved_dir / "effect_landcover_samples.csv",
        class_summary_csv=resolved_dir / "effect_landcover_class_summary.csv",
        correlations_csv=resolved_dir / "effect_landcover_correlations.csv",
        summary_json=resolved_dir / "summary.json",
    )


def _read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table_name not in tables:
            raise click.ClickException(
                f"{table_name!r} not found in {db_path}. Available tables: {sorted(tables)}"
            )
        return con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
    finally:
        con.close()


def _load_effects(
    cfg: DirectLiNGAMConfig,
    effects_csv: Path | None,
    effects_db: Path | None,
    effects_table: str | None,
) -> pd.DataFrame:
    csv_path = effects_csv or cfg.effects_csv
    db_path = effects_db or cfg.effects_db
    table_name = effects_table or cfg.effects_table
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if db_path.exists():
        return _read_table(db_path, table_name)
    raise click.ClickException(
        "No DirectLiNGAM effects output found. Expected either "
        f"{csv_path} or {db_path}::{table_name}."
    )


def _existing_validation_db(config_path: Path, location_name: str) -> Path:
    return config_path.parent / "landcover_graph_validation" / f"{location_name}_landcover_validation.duckdb"


def _load_landcover_labels_from_sources(
    *,
    labels_csv: Path | None,
    labels_db: Path | None,
    labels_table: str,
    default_validation_db: Path,
) -> pd.DataFrame | None:
    if labels_csv is not None:
        if not labels_csv.exists():
            raise click.ClickException(f"Land-cover labels CSV not found: {labels_csv}")
        return pd.read_csv(labels_csv)
    candidate_db = labels_db or default_validation_db
    if candidate_db.exists():
        con = duckdb.connect(str(candidate_db), read_only=True)
        try:
            tables = set(con.sql("SHOW TABLES").df()["name"])
            if labels_table in tables:
                return con.execute(f'SELECT * FROM "{labels_table}" ORDER BY row, col').fetchdf()
            if labels_db is not None:
                raise click.ClickException(
                    f"{labels_table!r} not found in {candidate_db}. "
                    f"Available tables: {sorted(tables)}"
                )
        finally:
            con.close()
    elif labels_db is not None:
        raise click.ClickException(f"Land-cover labels DuckDB not found: {labels_db}")
    return None


def _worldcover_paths_for_tiles(worldcover_dir: Path, tiles: Sequence[str]) -> list[Path]:
    return [
        worldcover_dir
        / f"ESA_WorldCover_10m_{WORLD_COVER_YEAR}_{WORLD_COVER_VERSION}_{tile}_Map.tif"
        for tile in tiles
    ]


def _build_landcover_labels(
    *,
    config_path: Path,
    directlingam_cfg: DirectLiNGAMConfig,
    graph_table: str,
    graph_window_size: int,
    landcover_samples_per_axis: int,
    reference_raster: Path | None,
    worldcover_dir: Path,
    download: bool,
    overwrite_worldcover: bool,
    request_timeout: float,
) -> pd.DataFrame:
    experiment_config = read_config(config_path)
    graph_rows = load_graph_rows(directlingam_cfg.graph_db, graph_table)
    if reference_raster is None:
        reference_raster = locate_reference_raster(
            source_db=directlingam_cfg.experiment_dir / f"{directlingam_cfg.location_name}_source_db.duckdb",
            reference_var=str(experiment_config["reference_var"]),
            name_map=experiment_config["name_map"],
        )
    click.echo(f"Reference raster: {reference_raster}")
    with rasterio.open(reference_raster) as reference:
        domain_bounds = graph_domain_bounds_wgs84(graph_rows, reference, graph_window_size)
        click.echo(
            "Graph-domain WGS84 bounds: "
            + ", ".join(f"{value:.5f}" for value in domain_bounds)
        )
        tiles = required_worldcover_tiles(domain_bounds, timeout=request_timeout)
        click.echo(f"WorldCover tiles required: {len(tiles)} ({', '.join(tiles)})")
        if download:
            worldcover_paths = download_worldcover(
                tiles=tiles,
                output_dir=worldcover_dir,
                overwrite=overwrite_worldcover,
                timeout=request_timeout,
            )
        else:
            worldcover_paths = _worldcover_paths_for_tiles(worldcover_dir, tiles)
            require_files(worldcover_paths)
        return label_graph_footprints(
            graph_rows=graph_rows,
            reference=reference,
            worldcover_paths=worldcover_paths,
            graph_window_size=graph_window_size,
            samples_per_axis=landcover_samples_per_axis,
        )


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _pearson_indicator(values: pd.Series, indicator: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    y = indicator.astype(float).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _filter_samples(
    samples: pd.DataFrame,
    *,
    row_col_cols: Sequence[str],
    metrics: Sequence[str],
    sources: Sequence[str] | None,
    target: str | None,
    class_set: str,
    min_purity: float,
    min_valid_landcover_fraction: float,
    min_class_samples: int,
) -> pd.DataFrame:
    required = [*row_col_cols, "source", "landcover_code", "landcover_class", *metrics]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise click.ClickException(f"Joined analysis/land-cover samples are missing columns: {missing}")
    work = samples.copy()
    if "error" in work.columns:
        work = work[work["error"].isna()].copy()
    if sources is not None:
        work = work[work["source"].astype(str).isin(set(sources))].copy()
    if target is not None:
        target_column = "target" if "target" in work.columns else "outcome"
        if target_column in work.columns:
            work = work[work[target_column].astype(str) == target].copy()
    allowed_codes = CLASS_SETS[class_set]
    work = work[
        work["landcover_code"].isin(allowed_codes)
        & (pd.to_numeric(work["landcover_purity"], errors="coerce") >= min_purity)
        & (
            pd.to_numeric(work["landcover_valid_fraction"], errors="coerce")
            >= min_valid_landcover_fraction
        )
    ].copy()
    class_counts = work["landcover_class"].value_counts()
    retained_classes = set(class_counts[class_counts >= min_class_samples].index)
    work = work[work["landcover_class"].isin(retained_classes)].copy()
    if work.empty:
        raise click.ClickException("No samples remain after land-cover/effect filtering.")
    return work.sort_values([*row_col_cols, "source"]).reset_index(drop=True)


def summarize_by_class(samples: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    group_cols = ["source", "landcover_code", "landcover_class"]
    if "target" in samples.columns:
        group_cols.insert(1, "target")
    elif "outcome" in samples.columns:
        group_cols.insert(1, "outcome")
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        grouped = samples.groupby(group_cols, dropna=False)[metric]
        summary = grouped.agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        summary.insert(0, "metric", metric)
        rows.extend(summary.to_dict(orient="records"))
    return pd.DataFrame(rows)


def compute_class_correlations(samples: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    base_group_cols = ["source"]
    if "target" in samples.columns:
        base_group_cols.append("target")
    elif "outcome" in samples.columns:
        base_group_cols.append("outcome")
    rows: list[dict[str, Any]] = []
    for group_key, group in samples.groupby(base_group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_record = dict(zip(base_group_cols, key_values, strict=False))
        classes = sorted(group["landcover_class"].dropna().astype(str).unique())
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            overall_mean = float(numeric.mean())
            for class_name in classes:
                in_class = group["landcover_class"].astype(str) == class_name
                class_values = numeric[in_class]
                rows.append(
                    {
                        **key_record,
                        "metric": metric,
                        "landcover_class": class_name,
                        "n_total": int(numeric.notna().sum()),
                        "n_class": int(class_values.notna().sum()),
                        "class_fraction": float(in_class.mean()),
                        "indicator_correlation": _pearson_indicator(numeric, in_class),
                        "class_mean": float(class_values.mean()),
                        "other_mean": float(numeric[~in_class].mean()),
                        "overall_mean": overall_mean,
                        "class_minus_other_mean": float(class_values.mean() - numeric[~in_class].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_indicator_correlation"] = out["indicator_correlation"].abs()
        out = out.sort_values(
            ["metric", "abs_indicator_correlation", "source", "landcover_class"],
            ascending=[True, False, True, True],
        )
    return out


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "value"


def _display_label(value: Any) -> str:
    """Format code-style variable names for figure text."""
    text = str(value).strip()
    if not text:
        return text
    text = text.replace("_", " ").replace("-", " ")
    acronym_tokens = {
        "ci": "CI",
        "db": "DB",
        "lst": "LST",
        "ndvi": "NDVI",
        "sd": "SD",
        "spei": "SPEI",
        "vpd": "VPD",
    }
    unit_tokens = {
        "cm": "cm",
        "m": "m",
        "mm": "mm",
    }
    word_tokens = {
        "abs": "Absolute",
        "boot": "Bootstrap",
        "corr": "Correlation",
        "gt": "Greater Than",
        "lt": "Less Than",
        "prob": "Probability",
    }
    lowercase_tokens = {"and", "as", "by", "for", "from", "in", "of", "on", "or", "to", "with"}
    words: list[str] = []
    for idx, raw_word in enumerate(text.split()):
        word = raw_word.strip()
        lower = word.lower()
        if lower in acronym_tokens:
            words.append(acronym_tokens[lower])
        elif lower in unit_tokens:
            words.append(unit_tokens[lower])
        elif lower in word_tokens:
            words.extend(word_tokens[lower].split())
        elif idx > 0 and lower in lowercase_tokens:
            words.append(lower)
        elif word.replace(".", "", 1).isdigit():
            words.append(word)
        else:
            words.append(lower.capitalize())
    return " ".join(words)


def _pivot_for_heatmap(
    frame: pd.DataFrame,
    *,
    metric: str,
    value_col: str,
    top_sources: int,
) -> pd.DataFrame:
    subset = frame[frame["metric"] == metric].copy()
    if subset.empty:
        return pd.DataFrame()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset[np.isfinite(subset[value_col])]
    if subset.empty:
        return pd.DataFrame()
    source_order = (
        subset.groupby("source")[value_col]
        .apply(lambda s: float(np.max(np.abs(s))))
        .sort_values(ascending=False)
        .head(top_sources)
        .index
    )
    subset = subset[subset["source"].isin(source_order)]
    pivot = subset.pivot_table(
        index="source",
        columns="landcover_class",
        values=value_col,
        aggfunc="mean",
    )
    return pivot.loc[list(source_order)]


def plot_correlation_heatmap(
    correlations: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    top_sources: int,
    show: bool,
) -> Path | None:
    pivot = _pivot_for_heatmap(
        correlations,
        metric=metric,
        value_col="indicator_correlation",
        top_sources=top_sources,
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(
        figsize=(
            max(8.0, 1.25 * len(pivot.columns)),
            max(5.5, 0.65 * len(pivot)),
        )
    )
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_display_label(column) for column in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_display_label(index) for index in pivot.index])
    ax.set_title(
        f"{_display_label(metric)}: Correlation with Land-Cover Class Indicators"
    )
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_class_mean_heatmap(
    summary: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    top_sources: int,
    show: bool,
) -> Path | None:
    pivot = _pivot_for_heatmap(
        summary,
        metric=metric,
        value_col="mean",
        top_sources=top_sources,
    )
    if pivot.empty:
        return None
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = float(np.nanquantile(np.abs(finite), 0.98)) if finite.size else 1.0
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(
        figsize=(
            max(8.0, 1.25 * len(pivot.columns)),
            max(5.5, 0.65 * len(pivot)),
        )
    )
    image = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_display_label(column) for column in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_display_label(index) for index in pivot.index])
    ax.set_title(f"{_display_label(metric)}: Mean by Land-Cover Class")
    fig.colorbar(image, ax=ax, label=f"Mean {_display_label(metric)}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_metric_boxplots(
    samples: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    metric: str,
    output_dir: Path,
    top_sources: int,
    show: bool,
) -> list[Path]:
    top = (
        correlations[correlations["metric"] == metric]
        .groupby("source")["abs_indicator_correlation"]
        .max()
        .sort_values(ascending=False)
        .head(top_sources)
        .index
    )
    if len(top) == 0:
        return []
    work = samples[samples["source"].isin(top)].copy()
    classes = sorted(work["landcover_class"].dropna().astype(str).unique())
    if not classes:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in top:
        source_values = []
        for class_name in classes:
            values = pd.to_numeric(
                work[(work["source"] == source) & (work["landcover_class"] == class_name)][metric],
                errors="coerce",
            ).dropna()
            source_values.append(values.to_numpy(dtype=float))
        fig, axis = plt.subplots(
            figsize=(max(9.0, 1.35 * len(classes)), 5.5)
        )
        axis.boxplot(
            source_values,
            labels=[_display_label(class_name) for class_name in classes],
            showfliers=False,
        )
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_title(
            f"{_display_label(metric)} by Land-Cover Class\n"
            f"{_display_label(source)}"
        )
        axis.set_ylabel(_display_label(metric))
        axis.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        output_path = (
            output_dir
            / f"{_safe_filename(metric)}__{_safe_filename(str(source))}__landcover_boxplot.png"
        )
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        written.append(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return written


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Experiment YAML configuration.",
)
@click.option("--target", default=None, help="Optional target/outcome filter.")
@click.option("--sources", default=None, help="Comma-separated source variables to include.")
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    default=("scaled_total_effect",),
    help="Numeric DirectLiNGAM effects column to compare. Repeatable.",
)
@click.option("--effects-csv", default=None, type=click.Path(path_type=Path), help="Override effects CSV path.")
@click.option("--effects-db", default=None, type=click.Path(path_type=Path), help="Override effects DuckDB path.")
@click.option("--effects-table", default=None, help="Override effects DuckDB table name.")
@click.option("--labels-csv", default=None, type=click.Path(path_type=Path), help="Use an existing land-cover labels CSV.")
@click.option("--labels-db", default=None, type=click.Path(path_type=Path), help="Use an existing DuckDB containing land-cover labels.")
@click.option("--labels-table", default="landcover_labels", show_default=True, help="Land-cover label table name.")
@click.option("--graph-table", default=None, help="Graph table used when labels must be generated.")
@click.option("--graph-window-size", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--landcover-samples-per-axis", default=5, show_default=True, type=click.IntRange(min=1))
@click.option("--class-set", type=click.Choice(sorted(CLASS_SETS)), default="vegetation", show_default=True)
@click.option("--min-purity", default=0.8, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option("--min-valid-landcover-fraction", default=0.8, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option("--min-class-samples", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--reference-raster", default=None, type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--worldcover-dir", default=None, type=click.Path(path_type=Path, file_okay=False))
@click.option("--download/--no-download", default=True, show_default=True, help="Download missing ESA WorldCover tiles.")
@click.option("--overwrite-worldcover", is_flag=True, help="Redownload WorldCover tiles even when present.")
@click.option("--request-timeout", default=60.0, show_default=True, type=click.FloatRange(min=1.0))
@click.option("--output-dir", default=None, type=click.Path(path_type=Path, file_okay=False))
@click.option("--top-sources", default=12, show_default=True, type=click.IntRange(min=1))
@click.option("--show", is_flag=True, help="Display plots interactively.")
def compare_directlingam_effects_with_landcover(
    config_path: Path,
    target: str | None,
    sources: str | None,
    metrics: tuple[str, ...],
    effects_csv: Path | None,
    effects_db: Path | None,
    effects_table: str | None,
    labels_csv: Path | None,
    labels_db: Path | None,
    labels_table: str,
    graph_table: str | None,
    graph_window_size: int,
    landcover_samples_per_axis: int,
    class_set: str,
    min_purity: float,
    min_valid_landcover_fraction: float,
    min_class_samples: int,
    reference_raster: Path | None,
    worldcover_dir: Path | None,
    download: bool,
    overwrite_worldcover: bool,
    request_timeout: float,
    output_dir: Path | None,
    top_sources: int,
    show: bool,
) -> None:
    """Join DirectLiNGAM effect rows to land cover and visualize associations."""
    cfg = load_directlingam_config(config_path, target_override=target)
    paths = _default_output_paths(cfg, output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    effects = _load_effects(cfg, effects_csv, effects_db, effects_table)
    default_validation_db = _existing_validation_db(config_path, cfg.location_name)
    labels = _load_landcover_labels_from_sources(
        labels_csv=labels_csv,
        labels_db=labels_db,
        labels_table=labels_table,
        default_validation_db=default_validation_db,
    )
    if labels is None:
        click.echo("No saved land-cover labels found; generating labels from WorldCover.")
        labels = _build_landcover_labels(
            config_path=config_path,
            directlingam_cfg=cfg,
            graph_table=graph_table or cfg.graph_table,
            graph_window_size=graph_window_size,
            landcover_samples_per_axis=landcover_samples_per_axis,
            reference_raster=reference_raster,
            worldcover_dir=worldcover_dir or paths.output_dir / "esa_worldcover_2021",
            download=download,
            overwrite_worldcover=overwrite_worldcover,
            request_timeout=request_timeout,
        )
    else:
        click.echo(f"Loaded {len(labels):,} saved land-cover labels.")

    join_cols = list(cfg.row_col_cols[:2])
    missing_label_cols = [column for column in join_cols if column not in labels.columns]
    if missing_label_cols:
        raise click.ClickException(
            "Land-cover labels are missing pixel coordinate columns "
            f"{missing_label_cols}. The existing WorldCover labelling helpers expect row/col pixels."
        )

    samples = effects.merge(
        labels,
        on=join_cols,
        how="inner",
        validate="many_to_one",
    )
    samples = _filter_samples(
        samples,
        row_col_cols=cfg.row_col_cols[:2],
        metrics=metrics,
        sources=_parse_csv(sources),
        target=target,
        class_set=class_set,
        min_purity=min_purity,
        min_valid_landcover_fraction=min_valid_landcover_fraction,
        min_class_samples=min_class_samples,
    )

    class_summary = summarize_by_class(samples, metrics)
    correlations = compute_class_correlations(samples, metrics)

    samples.to_csv(paths.samples_csv, index=False)
    class_summary.to_csv(paths.class_summary_csv, index=False)
    correlations.to_csv(paths.correlations_csv, index=False)

    con = duckdb.connect(str(paths.output_db))
    try:
        write_dataframe_table(con, labels, "landcover_labels")
        write_dataframe_table(con, samples, "effect_landcover_samples")
        write_dataframe_table(con, class_summary, "effect_landcover_class_summary")
        write_dataframe_table(con, correlations, "effect_landcover_correlations")
    finally:
        con.close()

    written_plots: list[Path] = []
    for metric in metrics:
        maybe_plots = [
            plot_correlation_heatmap(
                correlations,
                metric=metric,
                output_path=paths.output_dir / f"{_safe_filename(metric)}_landcover_indicator_correlations.png",
                top_sources=top_sources,
                show=show,
            ),
            plot_class_mean_heatmap(
                class_summary,
                metric=metric,
                output_path=paths.output_dir / f"{_safe_filename(metric)}_landcover_class_means.png",
                top_sources=top_sources,
                show=show,
            ),
        ]
        for path in maybe_plots:
            if path is not None:
                written_plots.append(path)
        written_plots.extend(
            plot_metric_boxplots(
                samples,
                correlations,
                metric=metric,
                output_dir=paths.output_dir,
                top_sources=min(top_sources, 6),
                show=show,
            )
        )

    summary = {
        "experiment": cfg.location_name,
        "target": target or cfg.target_col,
        "metrics": list(metrics),
        "n_effect_rows_loaded": int(len(effects)),
        "n_landcover_labels": int(len(labels)),
        "n_joined_samples": int(len(samples)),
        "class_set": class_set,
        "min_purity": float(min_purity),
        "min_valid_landcover_fraction": float(min_valid_landcover_fraction),
        "outputs": {
            "samples_csv": str(paths.samples_csv),
            "class_summary_csv": str(paths.class_summary_csv),
            "correlations_csv": str(paths.correlations_csv),
            "duckdb": str(paths.output_db),
            "plots": [str(path) for path in written_plots],
        },
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    click.echo(f"Effect-land-cover samples: {paths.samples_csv}")
    click.echo(f"Class summary: {paths.class_summary_csv}")
    click.echo(f"Class-indicator correlations: {paths.correlations_csv}")
    click.echo(f"DuckDB: {paths.output_db}")
    if written_plots:
        click.echo("Plots:")
        for path in written_plots:
            click.echo(f"  {path}")


if __name__ == "__main__":
    compare_directlingam_effects_with_landcover()
