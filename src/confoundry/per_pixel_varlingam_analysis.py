#!/usr/bin/env python3
"""Compute horizon-specific causal effects from saved per-pixel VAR-LiNGAM graphs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import matplotlib
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from confoundry.analysis_helpers import safe_filename, write_dataframe_table
from confoundry.varlingam_postprocess import (
    VARPixelBundle,
    bootstrap_dynamic_effect_matrices,
    dynamic_effect_matrices,
    iter_var_pixel_bundles,
    load_var_postprocess_config,
    load_var_timeseries_and_graphs,
    matrices_from_graph_row,
    quantile_contrast,
    summarize_bootstrap,
    validated_pixel_observations,
)
from confoundry.visualize_directlingam_diagnostics import (
    PUBLICATION_STYLE,
    parse_label_overrides,
    publication_variable_label,
)


POINT_MATRIX_CHOICES = ("raw", "consensus", "bootstrap_mean")


def _parse_sources(value: str | None) -> list[str] | None:
    if value is None:
        return None
    sources = [item.strip() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(sources)) or None


def _prefixed_summary(
    prefix: str,
    values: Sequence[float] | np.ndarray,
    ci: float,
) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in summarize_bootstrap(values, ci=ci).items()
    }


def analyze_var_pixel(
    bundle: VARPixelBundle,
    *,
    target: str,
    sources: Sequence[str] | None,
    horizon: int,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    stability_threshold: float,
    bootstrap_limit: int,
) -> list[dict[str, Any]]:
    """Compute direct, dynamic-total, and cumulative effects for one pixel."""
    try:
        matrices = matrices_from_graph_row(
            bundle.graph_row,
            point_matrix=point_matrix,
            bootstrap_limit=bootstrap_limit,
        )
        labels = list(matrices.labels)
        if target not in labels:
            raise ValueError(f"Target {target!r} is not in graph variables")
        selected_sources = (
            list(sources)
            if sources is not None
            else [label for label in labels if label != target]
        )
        unknown_sources = sorted(set(selected_sources) - set(labels))
        if unknown_sources:
            raise ValueError(
                f"Sources are not in graph variables: {unknown_sources}"
            )

        _, observations = validated_pixel_observations(
            bundle,
            labels,
            min_samples=min_samples,
        )

        effects, cumulative, point_radius = dynamic_effect_matrices(
            matrices.contemporaneous,
            matrices.lagged,
            horizon,
        )
        point_stable = bool(point_radius < stability_threshold)

        (
            bootstrap_effects,
            bootstrap_cumulative,
            bootstrap_radii,
            bootstrap_valid,
        ) = bootstrap_dynamic_effect_matrices(
            matrices.bootstrap_contemporaneous,
            matrices.bootstrap_lagged,
            horizon,
        )
        bootstrap_stable = (
            bootstrap_valid
            & np.isfinite(bootstrap_radii)
            & (bootstrap_radii < stability_threshold)
        )

        target_index = labels.index(target)
        target_low, target_high, target_delta = quantile_contrast(
            observations[:, target_index],
            low_quantile,
            high_quantile,
        )
        lag_count = matrices.lagged.shape[0]
        contemporaneous_multiplier = effects[0]

        rows: list[dict[str, Any]] = []
        for source in selected_sources:
            if source == target:
                continue
            source_index = labels.index(source)
            source_low, source_high, source_delta = quantile_contrast(
                observations[:, source_index],
                low_quantile,
                high_quantile,
            )
            scale = (
                source_delta / target_delta
                if np.isfinite(source_delta)
                and np.isfinite(target_delta)
                and target_delta != 0.0
                else np.nan
            )

            for step in range(horizon + 1):
                if step == 0:
                    direct_effect = float(
                        matrices.contemporaneous[target_index, source_index]
                    )
                    lag_slice_total = float(
                        contemporaneous_multiplier[target_index, source_index]
                    )
                    bootstrap_direct = matrices.bootstrap_contemporaneous[
                        :, target_index, source_index
                    ]
                elif step <= lag_count:
                    direct_effect = float(
                        matrices.lagged[
                            step - 1,
                            target_index,
                            source_index,
                        ]
                    )
                    lag_slice_total = float(
                        (
                            contemporaneous_multiplier
                            @ matrices.lagged[step - 1]
                        )[target_index, source_index]
                    )
                    bootstrap_direct = matrices.bootstrap_lagged[
                        :,
                        step - 1,
                        target_index,
                        source_index,
                    ]
                else:
                    direct_effect = 0.0
                    lag_slice_total = np.nan
                    bootstrap_direct = np.zeros(
                        len(matrices.bootstrap_contemporaneous),
                        dtype=float,
                    )

                total_effect = float(effects[step, target_index, source_index])
                cumulative_effect = float(
                    cumulative[step, target_index, source_index]
                )
                boot_total = bootstrap_effects[
                    bootstrap_stable,
                    step,
                    target_index,
                    source_index,
                ]
                boot_cumulative = bootstrap_cumulative[
                    bootstrap_stable,
                    step,
                    target_index,
                    source_index,
                ]
                boot_direct_stable = bootstrap_direct[bootstrap_stable]

                row = {
                    **bundle.coords,
                    "model_type": "varlingam",
                    "source": source,
                    "target": target,
                    "horizon": int(step),
                    "var_lags": int(lag_count),
                    "point_matrix": point_matrix,
                    "n_samples": int(len(observations)),
                    "point_stability_radius": float(point_radius),
                    "point_stable": point_stable,
                    "direct_effect": direct_effect,
                    "lag_slice_total_effect": lag_slice_total,
                    "total_effect": total_effect,
                    "cumulative_total_effect": cumulative_effect,
                    "scaled_total_effect": float(total_effect * scale),
                    "scaled_cumulative_total_effect": float(
                        cumulative_effect * scale
                    ),
                    "source_q_low": source_low,
                    "source_q_high": source_high,
                    "source_q_delta": source_delta,
                    "target_q_low": target_low,
                    "target_q_high": target_high,
                    "target_q_delta": target_delta,
                    "n_bootstrap_total": int(
                        len(matrices.bootstrap_contemporaneous)
                    ),
                    "n_bootstrap_matrix_valid": int(
                        np.sum(bootstrap_valid)
                    ),
                    "n_bootstrap_stable": int(
                        np.sum(bootstrap_stable)
                    ),
                    "n_bootstrap_unstable": int(
                        np.sum(bootstrap_valid & ~bootstrap_stable)
                    ),
                    "error": None,
                }
                row.update(
                    _prefixed_summary(
                        "direct_effect",
                        boot_direct_stable,
                        ci,
                    )
                )
                row.update(
                    _prefixed_summary(
                        "total_effect",
                        boot_total,
                        ci,
                    )
                )
                row.update(
                    _prefixed_summary(
                        "cumulative_total_effect",
                        boot_cumulative,
                        ci,
                    )
                )
                row.update(
                    _prefixed_summary(
                        "scaled_total_effect",
                        boot_total * scale,
                        ci,
                    )
                )
                row.update(
                    _prefixed_summary(
                        "scaled_cumulative_total_effect",
                        boot_cumulative * scale,
                        ci,
                    )
                )
                rows.append(row)
        return rows
    except Exception as exc:
        return [
            {
                **bundle.coords,
                "model_type": "varlingam",
                "source": None,
                "target": target,
                "horizon": None,
                "point_matrix": point_matrix,
                "point_stable": False,
                "error": str(exc),
            }
        ]


def _analyze_var_pixel_task(
    args: tuple[Any, ...],
) -> list[dict[str, Any]]:
    (
        bundle,
        target,
        sources,
        horizon,
        point_matrix,
        low_quantile,
        high_quantile,
        min_samples,
        ci,
        stability_threshold,
        bootstrap_limit,
    ) = args
    return analyze_var_pixel(
        bundle,
        target=target,
        sources=sources,
        horizon=horizon,
        point_matrix=point_matrix,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        min_samples=min_samples,
        ci=ci,
        stability_threshold=stability_threshold,
        bootstrap_limit=bootstrap_limit,
    )


def aggregate_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """Aggregate horizon effects across pixels for paper-level summaries."""
    summary_columns = [
        "source",
        "target",
        "horizon",
        "n_pixels",
        "scaled_total_effect_mean",
        "scaled_total_effect_median",
        "scaled_total_effect_q05",
        "scaled_total_effect_q95",
        "scaled_total_effect_fraction_positive",
        "scaled_cumulative_effect_median",
        "scaled_cumulative_effect_q05",
        "scaled_cumulative_effect_q95",
        "pixel_fraction_bootstrap_ci_excludes_zero",
        "median_bootstrap_sd",
        "median_bootstrap_stable_fraction",
    ]
    valid = effects[
        effects["error"].isna()
        & effects["point_stable"].astype(bool)
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=summary_columns)

    def summarize(group: pd.DataFrame) -> pd.Series:
        values = group["scaled_total_effect"].to_numpy(dtype=float)
        cumulative = group[
            "scaled_cumulative_total_effect"
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        cumulative = cumulative[np.isfinite(cumulative)]
        return pd.Series(
            {
                "n_pixels": int(len(group)),
                "scaled_total_effect_mean": float(np.mean(values))
                if len(values)
                else np.nan,
                "scaled_total_effect_median": float(np.median(values))
                if len(values)
                else np.nan,
                "scaled_total_effect_q05": float(np.quantile(values, 0.05))
                if len(values)
                else np.nan,
                "scaled_total_effect_q95": float(np.quantile(values, 0.95))
                if len(values)
                else np.nan,
                "scaled_total_effect_fraction_positive": float(
                    np.mean(values > 0.0)
                )
                if len(values)
                else np.nan,
                "scaled_cumulative_effect_median": float(
                    np.median(cumulative)
                )
                if len(cumulative)
                else np.nan,
                "scaled_cumulative_effect_q05": float(
                    np.quantile(cumulative, 0.05)
                )
                if len(cumulative)
                else np.nan,
                "scaled_cumulative_effect_q95": float(
                    np.quantile(cumulative, 0.95)
                )
                if len(cumulative)
                else np.nan,
                "pixel_fraction_bootstrap_ci_excludes_zero": float(
                    np.mean(
                        group[
                            "scaled_total_effect_boot_ci_excludes_zero"
                        ].astype(bool)
                    )
                ),
                "median_bootstrap_sd": float(
                    np.nanmedian(
                        group["scaled_total_effect_boot_sd"].to_numpy(
                            dtype=float
                        )
                    )
                ),
                "median_bootstrap_stable_fraction": float(
                    np.nanmedian(
                        group["n_bootstrap_stable"].to_numpy(dtype=float)
                        / np.maximum(
                            group["n_bootstrap_total"].to_numpy(dtype=float),
                            1.0,
                        )
                    )
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["source", "target", "horizon"]
    for key, group in valid.groupby(group_columns, sort=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, normalized_key, strict=True))
        row.update(summarize(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows, columns=summary_columns)


def plot_effect_trajectories(
    summary: pd.DataFrame,
    output_dir: Path,
    label_overrides: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write publication-friendly horizon response plots."""
    if summary.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (source, target), group in summary.groupby(
        ["source", "target"],
        sort=True,
    ):
        group = group.sort_values("horizon")
        horizon = group["horizon"].to_numpy(dtype=int)
        source_label = publication_variable_label(
            source,
            label_overrides,
        )
        target_label = publication_variable_label(
            target,
            label_overrides,
        )

        with plt.rc_context(PUBLICATION_STYLE):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(10, 4),
                sharex=True,
            )
            panels = [
                (
                    axes[0],
                    "scaled_total_effect_median",
                    "scaled_total_effect_q05",
                    "scaled_total_effect_q95",
                    "Horizon-specific effect",
                ),
                (
                    axes[1],
                    "scaled_cumulative_effect_median",
                    "scaled_cumulative_effect_q05",
                    "scaled_cumulative_effect_q95",
                    "Cumulative effect",
                ),
            ]
            for axis, median, lower, upper, title in panels:
                axis.fill_between(
                    horizon,
                    group[lower].to_numpy(dtype=float),
                    group[upper].to_numpy(dtype=float),
                    color="#4C78A8",
                    alpha=0.22,
                    label="Spatial 5–95% range",
                )
                axis.plot(
                    horizon,
                    group[median].to_numpy(dtype=float),
                    color="#1F4E79",
                    marker="o",
                    linewidth=1.8,
                    label="Spatial median",
                )
                axis.axhline(0.0, color="0.35", linewidth=0.8)
                axis.set_title(title)
                axis.set_xlabel("Months after intervention")
                axis.set_ylabel("IQR-scaled causal effect")
                axis.grid(alpha=0.2)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{source_label} → {target_label}")
            fig.tight_layout()

            stem = (
                f"varlingam_effect_{safe_filename(str(source))}_to_"
                f"{safe_filename(str(target))}"
            )
            png = output_dir / f"{stem}.png"
            pdf = output_dir / f"{stem}.pdf"
            fig.savefig(
                png,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                pdf,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
        written.extend([png, pdf])
    return written


def _resolve_output_path(
    experiment_dir: Path,
    override: Path | None,
    default_name: str,
) -> Path:
    if override is None:
        return experiment_dir / default_name
    return (
        override.expanduser()
        if override.is_absolute()
        else experiment_dir / override
    )


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--target",
    default=None,
    help="Target variable; defaults to the config target/reference variable.",
)
@click.option(
    "--sources",
    default=None,
    help=(
        "Comma-separated source variables; defaults to every non-target "
        "variable."
    ),
)
@click.option("--horizon", default=12, show_default=True, type=click.IntRange(0, None))
@click.option(
    "--point-matrix",
    type=click.Choice(POINT_MATRIX_CHOICES),
    default="raw",
    show_default=True,
)
@click.option("--low-quantile", default=0.10, show_default=True, type=float)
@click.option("--high-quantile", default=0.90, show_default=True, type=float)
@click.option("--min-samples", default=50, show_default=True, type=click.IntRange(2, None))
@click.option(
    "--ci",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
)
@click.option(
    "--stability-threshold",
    default=1.0,
    show_default=True,
    type=click.FloatRange(0.0, None, min_open=True),
    help="Maximum companion-matrix spectral radius; stable models are strictly below this value.",
)
@click.option(
    "--bootstrap-limit",
    default=0,
    show_default=True,
    type=click.IntRange(0, None),
    help="Use at most this many saved bootstrap replicates per pixel; 0 uses all.",
)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option("--graphs-db", default=None, type=click.Path(path_type=Path))
@click.option("--graphs-table", default=None)
@click.option("--output-csv", default=None, type=click.Path(path_type=Path))
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--effects-table", default="pixel_varlingam_effects", show_default=True)
@click.option("--summary-table", default="varlingam_effect_summary", show_default=True)
@click.option("--plot-dir", default=None, type=click.Path(path_type=Path))
@click.option(
    "--variable-label",
    "variable_labels",
    multiple=True,
    metavar="RAW=DISPLAY",
    help="Override a variable name in figure text; repeat as needed.",
)
@click.option("--no-plots", is_flag=True)
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=click.IntRange(1, None),
)
def per_pixel_varlingam_analysis(
    config_path: Path,
    target: str | None,
    sources: str | None,
    horizon: int,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    stability_threshold: float,
    bootstrap_limit: int,
    input_db: Path | None,
    input_table: str | None,
    graphs_db: Path | None,
    graphs_table: str | None,
    output_csv: Path | None,
    output_db: Path | None,
    effects_table: str,
    summary_table: str,
    plot_dir: Path | None,
    variable_labels: tuple[str, ...],
    no_plots: bool,
    jobs: int,
) -> None:
    """Compute per-pixel VAR-LiNGAM dynamic and cumulative causal effects."""
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise click.BadParameter(
            "Require 0 <= low-quantile < high-quantile <= 1."
        )
    label_overrides = parse_label_overrides(variable_labels)
    config = load_var_postprocess_config(
        config_path,
        input_db=input_db,
        input_table=input_table,
        graphs_db=graphs_db,
        graphs_table=graphs_table,
    )
    selected_target = target or config.target
    if selected_target is None:
        raise click.BadParameter(
            "Could not infer a target; pass --target or set reference_var."
        )
    selected_sources = _parse_sources(sources)
    time_series, graph_df, config_labels = load_var_timeseries_and_graphs(
        config
    )
    if selected_target not in config_labels:
        raise click.BadParameter(
            f"Target {selected_target!r} is not in configured variables."
        )
    if selected_sources is not None:
        unknown = sorted(set(selected_sources) - set(config_labels))
        if unknown:
            raise click.BadParameter(
                f"Unknown source variables: {unknown}"
            )

    bundles = list(
        iter_var_pixel_bundles(config, time_series, graph_df)
    )
    tasks = [
        (
            bundle,
            selected_target,
            selected_sources,
            horizon,
            point_matrix,
            low_quantile,
            high_quantile,
            min_samples,
            ci,
            stability_threshold,
            bootstrap_limit,
        )
        for bundle in bundles
    ]
    if jobs == 1:
        nested = [_analyze_var_pixel_task(task) for task in tasks]
    else:
        nested = process_map(
            _analyze_var_pixel_task,
            tasks,
            max_workers=jobs,
            chunksize=1,
            desc="VAR-LiNGAM effects",
        )
    rows = [row for result in nested for row in result]
    if not rows:
        raise click.ClickException("No VAR-LiNGAM effect rows were produced.")
    effects = pd.DataFrame(rows)
    summary = aggregate_effects(effects)

    output_csv_path = _resolve_output_path(
        config.experiment_dir,
        output_csv,
        f"{config.location_name}_varlingam_effects.csv",
    )
    output_db_path = _resolve_output_path(
        config.experiment_dir,
        output_db,
        f"{config.location_name}_varlingam_effects.duckdb",
    )
    plot_dir_path = _resolve_output_path(
        config.experiment_dir,
        plot_dir,
        f"{config.location_name}_varlingam_effect_plots",
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_csv_path, index=False)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_db_path))
    try:
        write_dataframe_table(con, effects, effects_table)
        write_dataframe_table(con, summary, summary_table)
        metadata = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_path": str(config.config_path),
                    "input_db": str(config.input_db),
                    "graphs_db": str(config.graphs_db),
                    "target": selected_target,
                    "sources_json": json.dumps(selected_sources),
                    "horizon": int(horizon),
                    "point_matrix": point_matrix,
                    "low_quantile": float(low_quantile),
                    "high_quantile": float(high_quantile),
                    "ci": float(ci),
                    "stability_threshold": float(stability_threshold),
                    "bootstrap_limit": int(bootstrap_limit),
                    "n_graph_rows": int(len(graph_df)),
                    "n_effect_rows": int(len(effects)),
                }
            ]
        )
        write_dataframe_table(
            con,
            metadata,
            "varlingam_analysis_run_metadata",
        )
    finally:
        con.close()

    written_plots = (
        []
        if no_plots
        else plot_effect_trajectories(
            summary,
            plot_dir_path,
            label_overrides,
        )
    )
    failed = int(effects["error"].notna().sum())
    unstable = int(
        effects.loc[effects["error"].isna(), "point_stable"]
        .eq(False)
        .sum()
    )
    click.echo(f"Wrote effects: {output_csv_path}")
    click.echo(
        f"Wrote effects database: {output_db_path}::{effects_table}"
    )
    click.echo(f"Wrote aggregate table: {output_db_path}::{summary_table}")
    click.echo(f"Failed rows: {failed}; unstable point-effect rows: {unstable}")
    if written_plots:
        click.echo(f"Wrote plots: {plot_dir_path}")


if __name__ == "__main__":
    per_pixel_varlingam_analysis()
