#!/usr/bin/env python3
"""Create a two-source DirectLiNGAM dominance map without changing the full analysis CLI.

This companion command is intentionally narrow: for exactly two candidate
source variables and one target variable, it computes quantile-scaled
DirectLiNGAM effects per pixel and writes one categorical map showing whether

* neither source has enough influence on the target;
* source A dominates;
* both sources are roughly equal; or
* source B dominates.

The implementation reuses the existing Confoundry DirectLiNGAM bootstrap
analysis helpers for config loading, DuckDB input loading, shifted columns,
bootstrap decoding, total-effect computation, and grid plotting utilities.  It
therefore assumes that ``per_pixel_directlingam_bootstrap_analysis.py`` remains
available next to this file or as ``confoundry.per_pixel_directlingam_bootstrap_analysis``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import os

import click
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from confoundry.per_pixel_directlingam_analysis import (
        Config,
        PixelBundle,
        _as_path,
        _bootstrap_matrices_from_row,
        _bootstrap_total_effect_matrices,
        _finite_vlim,
        _get_analysis_value,
        _grid_from_results,
        _point_matrix_from_row,
        _probability_matrix_from_graph_row,
        _quantile_contrast,
        _read_yaml,
        _safe_filename,
        _safe_float,
        _summary,
        _total_effect_matrix,
        iter_pixel_groups,
        load_config,
        load_shifted_timeseries_and_graphs,
        progress_bar,
    )
except ModuleNotFoundError:  # pragma: no cover - convenient when run from src/confoundry directly
    from per_pixel_directlingam_analysis import (  # type: ignore
        Config,
        PixelBundle,
        _as_path,
        _bootstrap_matrices_from_row,
        _bootstrap_total_effect_matrices,
        _finite_vlim,
        _get_analysis_value,
        _grid_from_results,
        _point_matrix_from_row,
        _probability_matrix_from_graph_row,
        _quantile_contrast,
        _read_yaml,
        _safe_filename,
        _safe_float,
        _summary,
        _total_effect_matrix,
        iter_pixel_groups,
        load_config,
        load_shifted_timeseries_and_graphs,
        progress_bar,
    )

try:
    from confoundry.per_pixel_graph_discovery import write_dataframe_table
except ModuleNotFoundError:  # pragma: no cover
    from per_pixel_graph_discovery import write_dataframe_table  # type: ignore


_EFFECT_MODES = ("direct", "total")
_VALID_POINT_MATRIX_CHOICES = ("raw", "consensus", "bootstrap_mean")
_CATEGORY_ORDER = ("neither", "source_a", "roughly_equal", "source_b")
_CATEGORY_TO_CODE = {name: idx for idx, name in enumerate(_CATEGORY_ORDER)}


def _parse_csv_sources(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        return parts or None
    return [str(item) for item in value]


def _require_two_sources(cfg: Config) -> tuple[str, str]:
    sources = list(cfg.source_cols or [])
    if len(sources) != 2:
        raise click.BadParameter(
            "This companion map requires exactly two sources. "
            "Pass --sources source_a,source_b or set analysis.sources to a two-item list."
        )
    if sources[0] == sources[1]:
        raise click.BadParameter("The two sources must be different variables.")
    return sources[0], sources[1]


def _resolve_companion_outputs(
    config_path: Path,
    cfg: Config,
    *,
    effect_mode: str,
    source_a: str,
    source_b: str,
    output_csv_override: Path | None,
    output_db_override: Path | None,
    output_table_override: str | None,
    output_map_override: Path | None,
) -> tuple[Path, Path, str, Path]:
    """Resolve output paths without adding new fields to the reused Config."""
    config_data = _read_yaml(config_path)
    src_a_slug = _safe_filename(source_a)
    src_b_slug = _safe_filename(source_b)
    target_slug = _safe_filename(cfg.target_col)
    mode_slug = _safe_filename(effect_mode)
    stem = f"{cfg.location_name}_directlingam_two_source_{mode_slug}_{src_a_slug}_vs_{src_b_slug}_to_{target_slug}"

    output_csv = _as_path(
        cfg.experiment_dir,
        output_csv_override or _get_analysis_value(config_data, "directlingam_two_source_csv"),
        f"{stem}.csv",
    )
    output_db = _as_path(
        cfg.experiment_dir,
        output_db_override or _get_analysis_value(config_data, "directlingam_two_source_db"),
        f"{stem}.duckdb",
    )
    output_table = str(
        output_table_override
        or _get_analysis_value(config_data, "directlingam_two_source_table", "pixel_directlingam_two_source")
    )
    output_map = _as_path(
        cfg.experiment_dir,
        output_map_override or _get_analysis_value(config_data, "directlingam_two_source_map"),
        cfg.plot_dir / f"two_source_{mode_slug}_{src_a_slug}_vs_{src_b_slug}_to_{target_slug}.png",
    )
    return output_csv, output_db, output_table, output_map


def _classify_two_source_effects(
    effect_a: float,
    effect_b: float,
    *,
    min_abs_effect: float,
    equal_ratio: float,
) -> str | None:
    """Return one of _CATEGORY_ORDER, or None when effects are non-finite."""
    if not np.isfinite(effect_a) or not np.isfinite(effect_b):
        return None

    abs_a = abs(float(effect_a))
    abs_b = abs(float(effect_b))
    max_abs = max(abs_a, abs_b)
    min_abs = min(abs_a, abs_b)

    if max_abs < min_abs_effect:
        return "neither"
    if min_abs >= min_abs_effect and max_abs > 0 and (min_abs / max_abs) >= equal_ratio:
        return "roughly_equal"
    return "source_a" if abs_a > abs_b else "source_b"


def _classify_bootstrap_pair(
    scaled_a: np.ndarray,
    scaled_b: np.ndarray,
    *,
    min_abs_effect: float,
    equal_ratio: float,
) -> tuple[dict[str, float], int]:
    counts = {category: 0 for category in _CATEGORY_ORDER}
    successful = 0
    for effect_a, effect_b in zip(scaled_a, scaled_b, strict=False):
        category = _classify_two_source_effects(
            float(effect_a),
            float(effect_b),
            min_abs_effect=min_abs_effect,
            equal_ratio=equal_ratio,
        )
        if category is None:
            continue
        counts[category] += 1
        successful += 1
    if successful == 0:
        return {}, 0
    return {category: count / successful for category, count in counts.items() if count > 0}, successful


def _effect_matrix_for_mode(point_B: np.ndarray, effect_mode: str) -> np.ndarray:
    if effect_mode == "direct":
        return point_B
    if effect_mode == "total":
        return _total_effect_matrix(point_B)
    raise ValueError(f"Unsupported effect_mode: {effect_mode!r}")


def _bootstrap_effect_matrices_for_mode(boot_B: np.ndarray, effect_mode: str) -> tuple[np.ndarray, int]:
    if effect_mode == "direct":
        return boot_B, 0
    if effect_mode == "total":
        return _bootstrap_total_effect_matrices(boot_B)
    raise ValueError(f"Unsupported effect_mode: {effect_mode!r}")


def _scaled_effects_for_source(
    effect_mats: np.ndarray,
    *,
    target_idx: int,
    source_idx: int,
    delta_source: float,
    delta_target: float,
) -> np.ndarray:
    raw = effect_mats[:, target_idx, source_idx]
    if not np.isfinite(delta_target) or delta_target == 0:
        return np.full_like(raw, np.nan, dtype=float)
    return raw * delta_source / delta_target


def analyze_two_source_pixel(
    bundle: PixelBundle,
    target_col: str,
    source_a: str,
    source_b: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    point_matrix: str,
    effect_mode: str,
    dominance_min_abs_effect: float,
    dominance_equal_ratio: float,
    ci: float,
) -> dict[str, Any]:
    """Compute one two-source categorical dominance row for a pixel."""
    base_row: dict[str, Any] = {
        **bundle.coords,
        "target": target_col,
        "outcome": target_col,
        "source_a": source_a,
        "source_b": source_b,
        "point_matrix": point_matrix,
        "effect_mode": effect_mode,
        "dominance_min_abs_effect": dominance_min_abs_effect,
        "dominance_equal_ratio": dominance_equal_ratio,
        "category": None,
        "category_code": np.nan,
        "category_label": None,
        "n_samples": 0,
        "n_bootstrap_total": 0,
        "n_bootstrap_effect_successful": 0,
        "n_bootstrap_effect_failed": 0,
        "bootstrap_category_probabilities_json": json.dumps({}),
        "bootstrap_category_mode": None,
        "bootstrap_category_mode_probability": np.nan,
        "point_category_boot_probability": np.nan,
        "error": None,
    }

    try:
        if effect_mode not in _EFFECT_MODES:
            raise ValueError(f"effect_mode must be one of {_EFFECT_MODES}, got {effect_mode!r}")
        if not 0.0 <= dominance_min_abs_effect:
            raise ValueError("dominance_min_abs_effect must be >= 0")
        if not 0.0 <= dominance_equal_ratio <= 1.0:
            raise ValueError("dominance_equal_ratio must be between 0 and 1")

        labels = [str(x) for x in bundle.graph_row["variable_names"]]
        index = {name: idx for idx, name in enumerate(labels)}
        missing = [name for name in [target_col, source_a, source_b] if name not in index]
        if missing:
            raise ValueError(f"variables missing from graph labels: {missing}")

        data = bundle.time_series.dropna(subset=list(dict.fromkeys(labels))).reset_index(drop=True)
        if len(data) < min_samples:
            raise ValueError(f"too few samples: {len(data)} < {min_samples}")
        base_row["n_samples"] = int(len(data))

        target_idx = index[target_col]
        source_a_idx = index[source_a]
        source_b_idx = index[source_b]

        point_B = _point_matrix_from_row(bundle.graph_row, point_matrix=point_matrix)
        boot_B = _bootstrap_matrices_from_row(bundle.graph_row)
        probs = _probability_matrix_from_graph_row(bundle.graph_row)

        if point_B.shape != (len(labels), len(labels)):
            raise ValueError(f"point adjacency shape {point_B.shape} does not match {len(labels)} labels")
        if boot_B.shape[1:] != (len(labels), len(labels)):
            raise ValueError(f"bootstrap adjacency shape {boot_B.shape} does not match {len(labels)} labels")

        target_q = _quantile_contrast(data[target_col], low_quantile, high_quantile)
        source_a_q = _quantile_contrast(data[source_a], low_quantile, high_quantile)
        source_b_q = _quantile_contrast(data[source_b], low_quantile, high_quantile)
        delta_target = target_q["delta"]
        delta_a = source_a_q["delta"]
        delta_b = source_b_q["delta"]

        point_effect = _effect_matrix_for_mode(point_B, effect_mode=effect_mode)
        boot_effect, n_effect_failed = _bootstrap_effect_matrices_for_mode(boot_B, effect_mode=effect_mode)
        if len(boot_effect) == 0:
            raise ValueError(f"No bootstrap matrix produced finite {effect_mode} effects")

        raw_a = float(point_effect[target_idx, source_a_idx])
        raw_b = float(point_effect[target_idx, source_b_idx])
        scaled_a = raw_a * delta_a / delta_target if np.isfinite(delta_target) and delta_target != 0 else np.nan
        scaled_b = raw_b * delta_b / delta_target if np.isfinite(delta_target) and delta_target != 0 else np.nan

        boot_scaled_a = _scaled_effects_for_source(
            boot_effect,
            target_idx=target_idx,
            source_idx=source_a_idx,
            delta_source=delta_a,
            delta_target=delta_target,
        )
        boot_scaled_b = _scaled_effects_for_source(
            boot_effect,
            target_idx=target_idx,
            source_idx=source_b_idx,
            delta_source=delta_b,
            delta_target=delta_target,
        )

        summary_a = _summary(boot_scaled_a, ci=ci)
        summary_b = _summary(boot_scaled_b, ci=ci)
        abs_summary_a = _summary(np.abs(boot_scaled_a), ci=ci)
        abs_summary_b = _summary(np.abs(boot_scaled_b), ci=ci)

        category = _classify_two_source_effects(
            scaled_a,
            scaled_b,
            min_abs_effect=dominance_min_abs_effect,
            equal_ratio=dominance_equal_ratio,
        )
        if category is None:
            raise ValueError("point scaled effects are non-finite; cannot classify pixel")

        boot_probs, n_boot_successful = _classify_bootstrap_pair(
            boot_scaled_a,
            boot_scaled_b,
            min_abs_effect=dominance_min_abs_effect,
            equal_ratio=dominance_equal_ratio,
        )
        if boot_probs:
            boot_mode = max(boot_probs, key=boot_probs.get)
            boot_mode_probability = float(boot_probs[boot_mode])
            point_boot_probability = float(boot_probs.get(category, 0.0))
        else:
            boot_mode = None
            boot_mode_probability = np.nan
            point_boot_probability = np.nan

        category_labels = {
            "neither": "neither above threshold",
            "source_a": f"{source_a} dominates",
            "roughly_equal": f"{source_a} ≈ {source_b}",
            "source_b": f"{source_b} dominates",
        }

        direct_prob_a = float(probs[target_idx, source_a_idx]) if probs is not None else np.nan
        direct_prob_b = float(probs[target_idx, source_b_idx]) if probs is not None else np.nan

        base_row.update(
            {
                "category": category,
                "category_code": float(_CATEGORY_TO_CODE[category]),
                "category_label": category_labels[category],
                "source_a_q_low": _safe_float(source_a_q["q_low"]),
                "source_a_q_high": _safe_float(source_a_q["q_high"]),
                "source_a_delta_qhi_qlo": _safe_float(delta_a),
                "source_b_q_low": _safe_float(source_b_q["q_low"]),
                "source_b_q_high": _safe_float(source_b_q["q_high"]),
                "source_b_delta_qhi_qlo": _safe_float(delta_b),
                "target_q_low": _safe_float(target_q["q_low"]),
                "target_q_high": _safe_float(target_q["q_high"]),
                "target_delta_qhi_qlo": _safe_float(delta_target),
                "source_a_effect": _safe_float(raw_a),
                "source_b_effect": _safe_float(raw_b),
                "source_a_scaled_effect": _safe_float(scaled_a),
                "source_b_scaled_effect": _safe_float(scaled_b),
                "source_a_abs_scaled_effect": _safe_float(abs(scaled_a)),
                "source_b_abs_scaled_effect": _safe_float(abs(scaled_b)),
                "source_a_direct_edge_probability": _safe_float(direct_prob_a),
                "source_b_direct_edge_probability": _safe_float(direct_prob_b),
                "source_a_scaled_effect_boot_mean": summary_a["boot_mean"],
                "source_a_scaled_effect_boot_median": summary_a["boot_median"],
                "source_a_scaled_effect_boot_sd": summary_a["boot_sd"],
                "source_a_scaled_effect_boot_ci_low": summary_a["boot_ci_low"],
                "source_a_scaled_effect_boot_ci_high": summary_a["boot_ci_high"],
                "source_a_scaled_effect_boot_prob_gt_zero": summary_a["boot_prob_gt_zero"],
                "source_a_scaled_effect_boot_prob_lt_zero": summary_a["boot_prob_lt_zero"],
                "source_a_abs_scaled_effect_boot_mean": abs_summary_a["boot_mean"],
                "source_a_abs_scaled_effect_boot_sd": abs_summary_a["boot_sd"],
                "source_b_scaled_effect_boot_mean": summary_b["boot_mean"],
                "source_b_scaled_effect_boot_median": summary_b["boot_median"],
                "source_b_scaled_effect_boot_sd": summary_b["boot_sd"],
                "source_b_scaled_effect_boot_ci_low": summary_b["boot_ci_low"],
                "source_b_scaled_effect_boot_ci_high": summary_b["boot_ci_high"],
                "source_b_scaled_effect_boot_prob_gt_zero": summary_b["boot_prob_gt_zero"],
                "source_b_scaled_effect_boot_prob_lt_zero": summary_b["boot_prob_lt_zero"],
                "source_b_abs_scaled_effect_boot_mean": abs_summary_b["boot_mean"],
                "source_b_abs_scaled_effect_boot_sd": abs_summary_b["boot_sd"],
                "n_bootstrap_total": int(len(boot_B)),
                "n_bootstrap_effect_successful": int(n_boot_successful),
                "n_bootstrap_effect_failed": int(n_effect_failed),
                "bootstrap_category_probabilities_json": json.dumps(boot_probs),
                "bootstrap_category_mode": boot_mode,
                "bootstrap_category_mode_probability": _safe_float(boot_mode_probability),
                "point_category_boot_probability": _safe_float(point_boot_probability),
            }
        )
        return base_row
    except Exception as exc:
        base_row["error"] = repr(exc)
        return base_row


def _analyze_two_source_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return analyze_two_source_pixel(*args)


def plot_two_source_category_map(
    results_df: pd.DataFrame,
    row_col_cols: Sequence[str],
    output_path: Path,
    *,
    source_a: str,
    source_b: str,
    target_col: str,
    effect_mode: str,
    show: bool = False,
) -> Path | None:
    """Save the requested single categorical map."""
    if len(row_col_cols) < 2 or results_df.empty:
        return None

    row_col, col_col = list(row_col_cols)[:2]
    work = results_df[results_df["error"].isna()].copy()
    if work.empty:
        return None

    work["category_code"] = work["category"].map(_CATEGORY_TO_CODE).astype(float)
    grid = _grid_from_results(work, row_col, col_col, "category_code")

    base = plt.get_cmap("tab10")
    colors = [base(i) for i in range(len(_CATEGORY_ORDER))]
    cmap = ListedColormap(colors)

    labels = {
        "neither": "neither above threshold",
        "source_a": f"{source_a} dominates",
        "roughly_equal": f"{source_a} ≈ {source_b}",
        "source_b": f"{source_b} dominates",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.imshow(
        grid.values,
        origin="upper",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(_CATEGORY_ORDER) - 0.5,
    )
    ax.set_title(f"Two-source {effect_mode} effect map for {target_col}")
    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        Patch(facecolor=colors[_CATEGORY_TO_CODE[category]], label=labels[category])
        for category in _CATEGORY_ORDER
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_two_source_effect_maps(
    results_df: pd.DataFrame,
    row_col_cols: Sequence[str],
    output_dir: Path,
    *,
    source_a: str,
    source_b: str,
    target_col: str,
    effect_mode: str,
    show: bool = False,
) -> list[Path]:
    """Optional diagnostic maps for the two signed scaled effects."""
    if len(row_col_cols) < 2 or results_df.empty:
        return []
    row_col, col_col = list(row_col_cols)[:2]
    work = results_df[results_df["error"].isna()].copy()
    if work.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source_label, value_col in [
        (source_a, "source_a_scaled_effect"),
        (source_b, "source_b_scaled_effect"),
    ]:
        grid = _grid_from_results(work, row_col, col_col, value_col)
        vmin, vmax = _finite_vlim(grid.values, symmetric=True)
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        im = ax.imshow(grid.values, origin="upper", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"{source_label} → {target_col}\nscaled {effect_mode} effect")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.75)
        plt.tight_layout()
        output_path = output_dir / (
            f"two_source_{_safe_filename(effect_mode)}_effect_"
            f"{_safe_filename(source_label)}_to_{_safe_filename(target_col)}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    help="Path to the YAML experiment config.",
)
@click.option("--target", "target", default=None, help="Override target/reference variable, e.g. ndvi. Alias: --outcome.")
@click.option("--outcome", "outcome_alias", default=None, help="Deprecated alias for --target.")
@click.option(
    "--sources",
    required=False,
    default=None,
    help="Exactly two comma-separated source variables, e.g. precipitation,temperature.",
)
@click.option(
    "--effect-mode",
    default="direct",
    show_default=True,
    type=click.Choice(_EFFECT_MODES),
    help="Use direct adjacency coefficients or total path effects for classification.",
)
@click.option(
    "--point-matrix",
    default=None,
    type=click.Choice(_VALID_POINT_MATRIX_CHOICES),
    help="Point-estimate matrix used for the point classification.",
)
@click.option("--low-quantile", default=None, type=float, help="Override low quantile, default config or 0.10.")
@click.option("--high-quantile", default=None, type=float, help="Override high quantile, default config or 0.90.")
@click.option("--min-samples", default=None, type=int, help="Override analysis_min_samples.")
@click.option("--ci", default=0.95, show_default=True, type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True))
@click.option(
    "--dominance-min-abs-effect",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, None),
    help="Minimum absolute scaled effect for a source to count as influential.",
)
@click.option(
    "--dominance-equal-ratio",
    default=0.8,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="If both sources are above threshold and min(abs)/max(abs) is at least this value, classify as roughly equal.",
)
@click.option("--output-csv", default=None, type=click.Path(path_type=Path), help="Override output CSV path.")
@click.option("--output-db", default=None, type=click.Path(path_type=Path), help="Override output DuckDB path.")
@click.option("--output-table", default=None, help="Override output DuckDB table name.")
@click.option("--output-map", default=None, type=click.Path(path_type=Path), help="Override categorical PNG map path.")
@click.option("--plot-dir", default=None, type=click.Path(path_type=Path), help="Override plot directory inherited by load_config.")
@click.option("--effect-diagnostic-maps", is_flag=True, help="Also write signed scaled-effect maps for each of the two sources.")
@click.option("--no-map", is_flag=True, help="Skip writing the categorical PNG map.")
@click.option("--show", is_flag=True, help="Show plots interactively as they are generated.")
@click.option("--no-progress", is_flag=True, help="Disable progress bars.")
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=int,
    help="Number of parallel worker processes.",
)
@click.option("--chunksize", default=1, show_default=True, type=int)
def per_pixel_directlingam_two_source_map(
    config_path: Path,
    target: str | None,
    outcome_alias: str | None,
    sources: str | None,
    effect_mode: str,
    point_matrix: str | None,
    low_quantile: float | None,
    high_quantile: float | None,
    min_samples: int | None,
    ci: float,
    dominance_min_abs_effect: float,
    dominance_equal_ratio: float,
    output_csv: Path | None,
    output_db: Path | None,
    output_table: str | None,
    output_map: Path | None,
    plot_dir: Path | None,
    effect_diagnostic_maps: bool,
    no_map: bool,
    show: bool,
    no_progress: bool,
    jobs: int,
    chunksize: int,
) -> None:
    """Run the narrow two-source DirectLiNGAM dominance-map analysis."""
    cfg = load_config(
        config_path=config_path,
        target_override=target,
        outcome_override=outcome_alias,
        sources_override=sources,
        point_matrix_override=point_matrix,
        plot_dir_override=plot_dir,
    )

    source_a, source_b = _require_two_sources(cfg)
    low_q = cfg.low_quantile if low_quantile is None else float(low_quantile)
    high_q = cfg.high_quantile if high_quantile is None else float(high_quantile)
    if not (0.0 <= low_q < high_q <= 1.0):
        raise click.BadParameter("Require 0 <= low_quantile < high_quantile <= 1.")

    effective_min_samples = cfg.min_samples if min_samples is None else int(min_samples)
    output_csv_path, output_db_path, output_table_name, output_map_path = _resolve_companion_outputs(
        config_path=config_path,
        cfg=cfg,
        effect_mode=effect_mode,
        source_a=source_a,
        source_b=source_b,
        output_csv_override=output_csv,
        output_db_override=output_db,
        output_table_override=output_table,
        output_map_override=output_map,
    )

    progress_disabled = no_progress
    if not progress_disabled:
        click.echo("Loading shifted time series and graph tables...")
    ts_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)
    if not progress_disabled:
        click.echo(f"Loaded {len(ts_df):,} time-series rows and {len(graph_df):,} graph rows.")

    bundles = list(
        progress_bar(
            iter_pixel_groups(cfg, timeseries_df=ts_df, graph_df=graph_df),
            total=len(graph_df),
            desc="Preparing two-source pixel tasks",
            unit="pixel",
            disabled=progress_disabled or len(graph_df) == 0,
        )
    )

    tasks = [
        (
            bundle,
            cfg.target_col,
            source_a,
            source_b,
            low_q,
            high_q,
            effective_min_samples,
            cfg.point_matrix,
            effect_mode,
            dominance_min_abs_effect,
            dominance_equal_ratio,
            ci,
        )
        for bundle in bundles
    ]

    if jobs == 1:
        rows = [
            _analyze_two_source_task(task)
            for task in progress_bar(
                tasks,
                total=len(tasks),
                desc="Classifying pixels",
                unit="pixel",
                disabled=progress_disabled or len(tasks) == 0,
            )
        ]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_analyze_two_source_task, task) for task in tasks]
            iterator = progress_bar(
                as_completed(futures),
                total=len(futures),
                desc=f"Classifying pixels using {jobs} workers",
                unit="pixel",
                disabled=progress_disabled or len(futures) == 0,
            )
            for future in iterator:
                rows.append(future.result())

    if not rows:
        raise click.ClickException("No two-source DirectLiNGAM rows were produced.")

    results_df = pd.DataFrame(rows)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_db_path))
    try:
        write_dataframe_table(con, results_df, output_table_name)
    finally:
        con.close()

    written_plots: list[Path] = []
    if not no_map:
        maybe_map = plot_two_source_category_map(
            results_df,
            cfg.row_col_cols,
            output_path=output_map_path,
            source_a=source_a,
            source_b=source_b,
            target_col=cfg.target_col,
            effect_mode=effect_mode,
            show=show,
        )
        if maybe_map is not None:
            written_plots.append(maybe_map)

    if effect_diagnostic_maps:
        written_plots.extend(
            plot_two_source_effect_maps(
                results_df,
                cfg.row_col_cols,
                output_dir=output_map_path.parent,
                source_a=source_a,
                source_b=source_b,
                target_col=cfg.target_col,
                effect_mode=effect_mode,
                show=show,
            )
        )

    n_failed = int(results_df["error"].notna().sum()) if "error" in results_df.columns else 0
    category_counts = results_df.loc[results_df["error"].isna(), "category_label"].value_counts(dropna=False)

    print(results_df.head())
    print(f"\nInput ARD DB: {cfg.timeseries_db}")
    print(f"Input graph DB: {cfg.graph_db}")
    print(f"Target: {cfg.target_col}")
    print(f"Sources: {source_a}, {source_b}")
    print(f"Effect mode: {effect_mode}")
    print(f"Point matrix: {cfg.point_matrix}")
    print(f"Quantile contrast: Q{high_q:.2f} - Q{low_q:.2f}")
    print(f"Dominance threshold: abs(scaled effect) >= {dominance_min_abs_effect:g}")
    print(f"Roughly equal ratio: {dominance_equal_ratio:g}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Output DuckDB: {output_db_path}::{output_table_name}")
    if written_plots:
        print("Plots:")
        for path in written_plots:
            print(f"  {path}")
    print(f"Failed rows: {n_failed} / {len(results_df)}")
    if not category_counts.empty:
        print("Category counts:")
        for label, count in category_counts.items():
            print(f"  {label}: {count}")


if __name__ == "__main__":
    per_pixel_directlingam_two_source_map()
