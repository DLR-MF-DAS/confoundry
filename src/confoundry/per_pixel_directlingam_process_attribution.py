#!/usr/bin/env python3
"""Generic grouped process attribution from saved per-pixel DirectLiNGAM graphs.

This is a post-processing companion to ``per_pixel_graph_discovery.py``.  It
uses the same experiment configuration and loading helpers as
``per_pixel_directlingam_analysis.py``, but all analysis-method parameters are
command-line options rather than a new YAML section.

The script provides two related analyses:

1. Grouped attribution into one target
   Two source groups are compared using dimensionless quantile-scaled direct
   or total effects.  Each group can be represented either by the joint fitted
   contribution of its members or by an aggregation of the members' individual
   scaled effects.

2. Optional upstream--mediator--target decomposition
   An upstream-to-mediator coupling score, mediator-to-target sensitivity
   score, and explicit two-edge mediated-path score are calculated.  Coupling
   and sensitivity can be classified into four generic high/low quadrants.

Every score and class is recomputed for each saved bootstrap adjacency matrix.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from confoundry.analysis_helpers import (
        safe_filename as _safe_filename,
        safe_float as _safe_float,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from src/confoundry
    from analysis_helpers import (  # type: ignore
        safe_filename as _safe_filename,
        safe_float as _safe_float,
    )

try:
    from confoundry.per_pixel_directlingam_analysis import (
        PixelBundle,
        _bootstrap_matrices_from_row,
        _grid_from_results,
        _point_matrix_from_row,
        _quantile_contrast,
        _summary,
        _total_effect_matrix,
        iter_pixel_groups,
        load_config,
        load_shifted_timeseries_and_graphs,
        progress_bar,
    )
    from confoundry.per_pixel_graph_discovery import write_dataframe_table
except ModuleNotFoundError:  # pragma: no cover - direct execution from src/confoundry
    from per_pixel_directlingam_analysis import (  # type: ignore
        PixelBundle,
        _bootstrap_matrices_from_row,
        _grid_from_results,
        _point_matrix_from_row,
        _quantile_contrast,
        _summary,
        _total_effect_matrix,
        iter_pixel_groups,
        load_config,
        load_shifted_timeseries_and_graphs,
        progress_bar,
    )
    from per_pixel_graph_discovery import write_dataframe_table  # type: ignore


_EFFECT_MODES = ("direct", "total")
_GROUP_METHODS = ("joint", "max_abs", "rms_abs", "mean_abs", "sum_abs")
_AGGREGATIONS = ("max_abs", "rms_abs", "mean_abs", "sum_abs")
_POINT_MATRICES = ("raw", "consensus", "bootstrap_mean")


def _parse_csv(value: str | None, option_name: str, *, required: bool) -> list[str]:
    values = [] if value is None else [item.strip() for item in value.split(",") if item.strip()]
    values = list(dict.fromkeys(values))
    if required and not values:
        raise click.BadParameter("must contain at least one comma-separated variable", param_hint=option_name)
    return values


def _resolve_path(base_dir: Path, override: Path | None, default_name: str) -> Path:
    path = override if override is not None else Path(default_name)
    path = path.expanduser()
    return path if path.is_absolute() else base_dir / path


def _aggregate_abs(values: Sequence[float] | np.ndarray, method: str) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    absolute = np.abs(arr)
    if method == "max_abs":
        return float(np.max(absolute))
    if method == "rms_abs":
        return float(np.sqrt(np.mean(np.square(absolute))))
    if method == "mean_abs":
        return float(np.mean(absolute))
    if method == "sum_abs":
        return float(np.sum(absolute))
    raise ValueError(f"Unknown aggregation: {method!r}")


def _effect_matrices(
    point_adjacency: np.ndarray,
    bootstrap_adjacencies: np.ndarray,
    effect_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return point effects, bootstrap effects, aligned raw bootstraps, failures."""
    if effect_mode == "direct":
        return point_adjacency, bootstrap_adjacencies, bootstrap_adjacencies, 0
    if effect_mode != "total":
        raise ValueError(f"Unknown effect mode: {effect_mode!r}")

    point_effects = _total_effect_matrix(point_adjacency)
    effect_matrices: list[np.ndarray] = []
    raw_matrices: list[np.ndarray] = []
    failed = 0
    for adjacency in bootstrap_adjacencies:
        try:
            total = _total_effect_matrix(np.asarray(adjacency, dtype=float))
            if not np.all(np.isfinite(total)):
                failed += 1
                continue
            effect_matrices.append(total)
            raw_matrices.append(np.asarray(adjacency, dtype=float))
        except Exception:
            failed += 1
    if not effect_matrices:
        raise ValueError("No bootstrap adjacency matrix produced finite total effects")
    return point_effects, np.stack(effect_matrices), np.stack(raw_matrices), failed


def _scaled_effect(
    effects: np.ndarray,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    source: str,
    target: str,
) -> float:
    denominator = float(deltas[target])
    numerator = float(deltas[source])
    if not np.isfinite(denominator) or denominator == 0.0 or not np.isfinite(numerator):
        return float("nan")
    return float(effects[index[target], index[source]] * numerator / denominator)


def _joint_group_score(
    effects: np.ndarray,
    data: pd.DataFrame,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    sources: Sequence[str],
    target: str,
    low_quantile: float,
    high_quantile: float,
) -> float:
    target_delta = float(deltas[target])
    if not np.isfinite(target_delta) or target_delta == 0.0:
        return float("nan")
    coefficients = np.asarray([effects[index[target], index[source]] for source in sources], dtype=float)
    contribution = data[list(sources)].to_numpy(dtype=float) @ coefficients
    contribution = contribution[np.isfinite(contribution)]
    if len(contribution) == 0:
        return float("nan")
    q_low, q_high = np.quantile(contribution, [low_quantile, high_quantile])
    return float((q_high - q_low) / target_delta)


def _group_score(
    effects: np.ndarray,
    data: pd.DataFrame,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    sources: Sequence[str],
    target: str,
    method: str,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float, dict[str, float]]:
    individual = {
        source: _scaled_effect(effects, index, deltas, source, target)
        for source in sources
    }
    if method == "joint":
        score = _joint_group_score(
            effects,
            data,
            index,
            deltas,
            sources,
            target,
            low_quantile,
            high_quantile,
        )
    else:
        score = _aggregate_abs(list(individual.values()), method)
    return score, individual


def _coupling_score(
    effects: np.ndarray,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    upstream_sources: Sequence[str],
    mediators: Sequence[str],
    aggregation: str,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for source in upstream_sources:
        for mediator in mediators:
            components[f"{source}->{mediator}"] = _scaled_effect(
                effects, index, deltas, source, mediator
            )
    return _aggregate_abs(list(components.values()), aggregation), components


def _sensitivity_score(
    effects: np.ndarray,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    mediators: Sequence[str],
    target: str,
    aggregation: str,
) -> tuple[float, dict[str, float]]:
    components = {
        f"{mediator}->{target}": _scaled_effect(effects, index, deltas, mediator, target)
        for mediator in mediators
    }
    return _aggregate_abs(list(components.values()), aggregation), components


def _mediated_path_score(
    adjacency: np.ndarray,
    index: Mapping[str, int],
    deltas: Mapping[str, float],
    upstream_sources: Sequence[str],
    mediators: Sequence[str],
    target: str,
    aggregation: str,
) -> tuple[float, dict[str, float]]:
    target_delta = float(deltas[target])
    components: dict[str, float] = {}
    for source in upstream_sources:
        source_delta = float(deltas[source])
        for mediator in mediators:
            key = f"{source}->{mediator}->{target}"
            if (
                not np.isfinite(target_delta)
                or target_delta == 0.0
                or not np.isfinite(source_delta)
            ):
                components[key] = float("nan")
                continue
            raw_product = (
                float(adjacency[index[mediator], index[source]])
                * float(adjacency[index[target], index[mediator]])
            )
            components[key] = float(raw_product * source_delta / target_delta)
    return _aggregate_abs(list(components.values()), aggregation), components


def _classify_dominance(
    score_a: float,
    score_b: float,
    *,
    label_a: str,
    label_b: str,
    weak_label: str,
    mixed_label: str,
    min_abs_effect: float,
    equal_ratio: float,
) -> str | None:
    if not np.isfinite(score_a) or not np.isfinite(score_b):
        return None
    a = abs(float(score_a))
    b = abs(float(score_b))
    maximum = max(a, b)
    minimum = min(a, b)
    # <= deliberately makes two exact zero scores weak even when the threshold is zero.
    if maximum <= min_abs_effect:
        return weak_label
    if maximum > 0.0 and minimum / maximum >= equal_ratio:
        return mixed_label
    return label_a if a > b else label_b


def _classify_quadrant(
    coupling: float,
    sensitivity: float,
    *,
    coupling_threshold: float,
    sensitivity_threshold: float,
    low_low_label: str,
    high_low_label: str,
    low_high_label: str,
    high_high_label: str,
) -> str | None:
    if not np.isfinite(coupling) or not np.isfinite(sensitivity):
        return None
    high_x = abs(float(coupling)) > coupling_threshold
    high_y = abs(float(sensitivity)) > sensitivity_threshold
    if high_x and high_y:
        return high_high_label
    if high_x:
        return high_low_label
    if high_y:
        return low_high_label
    return low_low_label


def _class_summary(
    point_class: str | None,
    bootstrap_classes: Sequence[str | None],
    *,
    min_class_support: float,
    uncertain_label: str,
) -> dict[str, Any]:
    valid = [value for value in bootstrap_classes if value is not None]
    if not valid:
        return {
            "point_class": point_class,
            "class": uncertain_label if point_class is not None and min_class_support > 0.0 else point_class,
            "boot_mode": None,
            "boot_mode_probability": np.nan,
            "point_class_boot_probability": np.nan,
            "bootstrap_probabilities_json": json.dumps({}),
            "n_bootstrap_classified": 0,
        }
    counts = pd.Series(valid, dtype="object").value_counts()
    probabilities = {str(label): float(count / len(valid)) for label, count in counts.items()}
    mode = str(counts.index[0])
    mode_probability = probabilities[mode]
    point_probability = probabilities.get(str(point_class), 0.0) if point_class is not None else np.nan
    final_class = point_class
    if point_class is not None and np.isfinite(point_probability) and point_probability < min_class_support:
        final_class = uncertain_label
    return {
        "point_class": point_class,
        "class": final_class,
        "boot_mode": mode,
        "boot_mode_probability": mode_probability,
        "point_class_boot_probability": point_probability,
        "bootstrap_probabilities_json": json.dumps(probabilities, sort_keys=True),
        "n_bootstrap_classified": len(valid),
    }


def _prefix_summary(prefix: str, values: Sequence[float], ci: float) -> dict[str, Any]:
    summary = _summary(values, ci=ci)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def analyze_pixel(
    bundle: PixelBundle,
    target: str,
    group_a_sources: list[str],
    group_b_sources: list[str],
    group_a_label: str,
    group_b_label: str,
    group_a_method: str,
    group_b_method: str,
    upstream_sources: list[str],
    mediators: list[str],
    effect_mode: str,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    attribution_min_abs_effect: float,
    attribution_equal_ratio: float,
    coupling_aggregation: str,
    sensitivity_aggregation: str,
    mediated_aggregation: str,
    coupling_threshold: float,
    sensitivity_threshold: float,
    low_low_label: str,
    high_low_label: str,
    low_high_label: str,
    high_high_label: str,
    weak_label: str,
    mixed_label: str,
    uncertain_label: str,
    min_class_support: float,
) -> dict[str, Any]:
    base = {
        **bundle.coords,
        "target": target,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "group_a_sources": ",".join(group_a_sources),
        "group_b_sources": ",".join(group_b_sources),
        "group_a_method": group_a_method,
        "group_b_method": group_b_method,
        "upstream_sources": ",".join(upstream_sources),
        "mediators": ",".join(mediators),
        "effect_mode": effect_mode,
        "point_matrix": point_matrix,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "error": None,
    }

    labels = [str(value) for value in bundle.graph_row["variable_names"]]
    required = list(
        dict.fromkeys(
            [target]
            + group_a_sources
            + group_b_sources
            + upstream_sources
            + mediators
        )
    )
    missing = [name for name in required if name not in labels]
    if missing:
        return {**base, "error": f"variables not present in graph: {missing}"}

    # Match the existing DirectLiNGAM analysis: use rows complete for all graph variables.
    data = bundle.time_series.dropna(subset=labels).reset_index(drop=True)
    if len(data) < min_samples:
        return {**base, "n_samples": int(len(data)), "error": f"too few samples: {len(data)} < {min_samples}"}

    try:
        point_adjacency = _point_matrix_from_row(bundle.graph_row, point_matrix=point_matrix)
        bootstrap_adjacencies = _bootstrap_matrices_from_row(bundle.graph_row)
        if point_adjacency.shape != (len(labels), len(labels)):
            raise ValueError(
                f"point adjacency shape {point_adjacency.shape} does not match {len(labels)} labels"
            )
        if bootstrap_adjacencies.shape[1:] != (len(labels), len(labels)):
            raise ValueError(
                "bootstrap adjacency shape "
                f"{bootstrap_adjacencies.shape} does not match {len(labels)} labels"
            )
        point_effects, bootstrap_effects, aligned_bootstrap_adjacencies, n_failed = _effect_matrices(
            point_adjacency, bootstrap_adjacencies, effect_mode
        )
    except Exception as exc:
        return {**base, "n_samples": int(len(data)), "error": repr(exc)}

    index = {name: position for position, name in enumerate(labels)}
    deltas = {
        name: float(_quantile_contrast(data[name], low_quantile, high_quantile)["delta"])
        for name in required
    }
    invalid_deltas = [name for name, delta in deltas.items() if not np.isfinite(delta) or delta == 0.0]
    if invalid_deltas:
        return {
            **base,
            "n_samples": int(len(data)),
            "error": f"zero or invalid quantile range for variables: {invalid_deltas}",
        }

    group_a_score, group_a_components = _group_score(
        point_effects,
        data,
        index,
        deltas,
        group_a_sources,
        target,
        group_a_method,
        low_quantile,
        high_quantile,
    )
    group_b_score, group_b_components = _group_score(
        point_effects,
        data,
        index,
        deltas,
        group_b_sources,
        target,
        group_b_method,
        low_quantile,
        high_quantile,
    )

    group_a_boot: list[float] = []
    group_b_boot: list[float] = []
    attribution_boot_classes: list[str | None] = []
    for effects in bootstrap_effects:
        score_a, _ = _group_score(
            effects,
            data,
            index,
            deltas,
            group_a_sources,
            target,
            group_a_method,
            low_quantile,
            high_quantile,
        )
        score_b, _ = _group_score(
            effects,
            data,
            index,
            deltas,
            group_b_sources,
            target,
            group_b_method,
            low_quantile,
            high_quantile,
        )
        group_a_boot.append(score_a)
        group_b_boot.append(score_b)
        attribution_boot_classes.append(
            _classify_dominance(
                score_a,
                score_b,
                label_a=group_a_label,
                label_b=group_b_label,
                weak_label=weak_label,
                mixed_label=mixed_label,
                min_abs_effect=attribution_min_abs_effect,
                equal_ratio=attribution_equal_ratio,
            )
        )

    attribution_point = _classify_dominance(
        group_a_score,
        group_b_score,
        label_a=group_a_label,
        label_b=group_b_label,
        weak_label=weak_label,
        mixed_label=mixed_label,
        min_abs_effect=attribution_min_abs_effect,
        equal_ratio=attribution_equal_ratio,
    )
    attribution_summary = _class_summary(
        attribution_point,
        attribution_boot_classes,
        min_class_support=min_class_support,
        uncertain_label=uncertain_label,
    )

    row: dict[str, Any] = {
        **base,
        "n_samples": int(len(data)),
        "n_bootstrap_total": int(len(bootstrap_adjacencies)),
        "n_bootstrap_effect_successful": int(len(bootstrap_effects)),
        "n_bootstrap_effect_failed": int(n_failed),
        "group_a_score": _safe_float(group_a_score),
        "group_b_score": _safe_float(group_b_score),
        "group_a_components_json": json.dumps(group_a_components, sort_keys=True),
        "group_b_components_json": json.dumps(group_b_components, sort_keys=True),
        **_prefix_summary("group_a_score", group_a_boot, ci),
        **_prefix_summary("group_b_score", group_b_boot, ci),
        "attribution_point_class": attribution_summary["point_class"],
        "attribution_class": attribution_summary["class"],
        "attribution_boot_mode": attribution_summary["boot_mode"],
        "attribution_boot_mode_probability": attribution_summary["boot_mode_probability"],
        "attribution_point_class_boot_probability": attribution_summary[
            "point_class_boot_probability"
        ],
        "attribution_bootstrap_probabilities_json": attribution_summary[
            "bootstrap_probabilities_json"
        ],
        "n_bootstrap_attribution_classified": attribution_summary[
            "n_bootstrap_classified"
        ],
    }

    if not upstream_sources and not mediators:
        return row

    coupling_score, coupling_components = _coupling_score(
        point_effects,
        index,
        deltas,
        upstream_sources,
        mediators,
        coupling_aggregation,
    )
    sensitivity_score, sensitivity_components = _sensitivity_score(
        point_effects,
        index,
        deltas,
        mediators,
        target,
        sensitivity_aggregation,
    )
    mediated_score, mediated_components = _mediated_path_score(
        point_adjacency,
        index,
        deltas,
        upstream_sources,
        mediators,
        target,
        mediated_aggregation,
    )

    coupling_boot: list[float] = []
    sensitivity_boot: list[float] = []
    mediated_boot: list[float] = []
    quadrant_boot_classes: list[str | None] = []
    for effects, adjacency in zip(
        bootstrap_effects, aligned_bootstrap_adjacencies, strict=True
    ):
        boot_coupling, _ = _coupling_score(
            effects,
            index,
            deltas,
            upstream_sources,
            mediators,
            coupling_aggregation,
        )
        boot_sensitivity, _ = _sensitivity_score(
            effects,
            index,
            deltas,
            mediators,
            target,
            sensitivity_aggregation,
        )
        boot_mediated, _ = _mediated_path_score(
            adjacency,
            index,
            deltas,
            upstream_sources,
            mediators,
            target,
            mediated_aggregation,
        )
        coupling_boot.append(boot_coupling)
        sensitivity_boot.append(boot_sensitivity)
        mediated_boot.append(boot_mediated)
        quadrant_boot_classes.append(
            _classify_quadrant(
                boot_coupling,
                boot_sensitivity,
                coupling_threshold=coupling_threshold,
                sensitivity_threshold=sensitivity_threshold,
                low_low_label=low_low_label,
                high_low_label=high_low_label,
                low_high_label=low_high_label,
                high_high_label=high_high_label,
            )
        )

    quadrant_point = _classify_quadrant(
        coupling_score,
        sensitivity_score,
        coupling_threshold=coupling_threshold,
        sensitivity_threshold=sensitivity_threshold,
        low_low_label=low_low_label,
        high_low_label=high_low_label,
        low_high_label=low_high_label,
        high_high_label=high_high_label,
    )
    quadrant_summary = _class_summary(
        quadrant_point,
        quadrant_boot_classes,
        min_class_support=min_class_support,
        uncertain_label=uncertain_label,
    )

    row.update(
        {
            "coupling_score": _safe_float(coupling_score),
            "sensitivity_score": _safe_float(sensitivity_score),
            "mediated_path_score": _safe_float(mediated_score),
            "coupling_components_json": json.dumps(coupling_components, sort_keys=True),
            "sensitivity_components_json": json.dumps(sensitivity_components, sort_keys=True),
            "mediated_path_components_json": json.dumps(mediated_components, sort_keys=True),
            **_prefix_summary("coupling_score", coupling_boot, ci),
            **_prefix_summary("sensitivity_score", sensitivity_boot, ci),
            **_prefix_summary("mediated_path_score", mediated_boot, ci),
            "coupling_sensitivity_point_class": quadrant_summary["point_class"],
            "coupling_sensitivity_class": quadrant_summary["class"],
            "coupling_sensitivity_boot_mode": quadrant_summary["boot_mode"],
            "coupling_sensitivity_boot_mode_probability": quadrant_summary[
                "boot_mode_probability"
            ],
            "coupling_sensitivity_point_class_boot_probability": quadrant_summary[
                "point_class_boot_probability"
            ],
            "coupling_sensitivity_bootstrap_probabilities_json": quadrant_summary[
                "bootstrap_probabilities_json"
            ],
            "n_bootstrap_coupling_sensitivity_classified": quadrant_summary[
                "n_bootstrap_classified"
            ],
        }
    )
    return row


def _analyze_pixel_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return analyze_pixel(*args)


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return df.copy()
    return df[df["error"].isna()].copy()


def _plot_categorical_map(
    df: pd.DataFrame,
    row_col_cols: Sequence[str],
    value_col: str,
    output_path: Path,
    *,
    title: str,
    class_order: Sequence[str],
    figure_width: float,
    figure_height: float,
    dpi: int,
    title_font_size: float,
    legend_font_size: float,
    show_title: bool,
    show: bool,
) -> Path | None:
    if len(row_col_cols) < 2 or value_col not in df.columns:
        return None
    work = _successful_rows(df)
    work = work[work[value_col].notna()].copy()
    if work.empty:
        return None
    present = [label for label in class_order if label in set(work[value_col].astype(str))]
    extras = sorted(set(work[value_col].astype(str)) - set(present))
    labels = present + extras
    if not labels:
        return None
    codes = {label: idx for idx, label in enumerate(labels)}
    work["_class_code"] = work[value_col].astype(str).map(codes).astype(float)
    grid = _grid_from_results(work, row_col_cols[0], row_col_cols[1], "_class_code")
    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(idx % 10) for idx in range(len(labels))]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.imshow(
        grid.values,
        origin="upper",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        interpolation="nearest",
        aspect="equal",
    )
    if show_title:
        ax.set_title(title, fontsize=title_font_size, pad=6)
    ax.set_axis_off()
    handles = [Patch(facecolor=colors[idx], label=label) for idx, label in enumerate(labels)]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=legend_font_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _finite_symmetric_limit(values: np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None
    maximum = float(np.quantile(np.abs(arr), 0.98))
    if maximum == 0.0:
        return None, None
    return -maximum, maximum


def _plot_diagnostic_maps(
    df: pd.DataFrame,
    row_col_cols: Sequence[str],
    plot_dir: Path,
    *,
    figure_width: float,
    figure_height: float,
    dpi: int,
    title_font_size: float,
    show_title: bool,
    show: bool,
) -> list[Path]:
    if len(row_col_cols) < 2:
        return []
    work = _successful_rows(df)
    columns = [
        ("group_a_score", "Group A score"),
        ("group_b_score", "Group B score"),
        ("coupling_score", "Upstream-to-mediator coupling"),
        ("sensitivity_score", "Mediator-to-target sensitivity"),
        ("mediated_path_score", "Mediated path score"),
    ]
    written: list[Path] = []
    for column, title in columns:
        if column not in work.columns or work[column].notna().sum() == 0:
            continue
        grid = _grid_from_results(work, row_col_cols[0], row_col_cols[1], column)
        vmin, vmax = _finite_symmetric_limit(grid.values)
        if np.nanmin(grid.values) >= 0.0:
            vmin = 0.0
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        image = ax.imshow(
            grid.values,
            origin="upper",
            cmap="viridis" if vmin == 0.0 else "coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        if show_title:
            ax.set_title(title, fontsize=title_font_size, pad=6)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
        output = plot_dir / f"{_safe_filename(column)}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        written.append(output)
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
    help="Path to the existing YAML experiment config.",
)
@click.option("--target", default=None, help="Target/reference variable. Alias: --outcome.")
@click.option("--outcome", "outcome_alias", default=None, help="Deprecated alias for --target.")
@click.option(
    "--group-a-sources",
    required=True,
    help="Comma-separated sources in comparison group A.",
)
@click.option(
    "--group-b-sources",
    required=True,
    help="Comma-separated sources in comparison group B.",
)
@click.option("--group-a-label", default="group_a", show_default=True)
@click.option("--group-b-label", default="group_b", show_default=True)
@click.option(
    "--group-a-method",
    type=click.Choice(_GROUP_METHODS),
    default="joint",
    show_default=True,
    help="How group A effects into the target are combined.",
)
@click.option(
    "--group-b-method",
    type=click.Choice(_GROUP_METHODS),
    default="rms_abs",
    show_default=True,
    help="How group B effects into the target are combined.",
)
@click.option(
    "--upstream-sources",
    default=None,
    help="Optional comma-separated upstream variables for the mediator analysis.",
)
@click.option(
    "--mediators",
    default=None,
    help="Optional comma-separated mediators. Must be supplied with --upstream-sources.",
)
@click.option(
    "--effect-mode",
    type=click.Choice(_EFFECT_MODES),
    default="direct",
    show_default=True,
    help="Use direct adjacency coefficients or total path effects for group/coupling scores.",
)
@click.option(
    "--point-matrix",
    type=click.Choice(_POINT_MATRICES),
    default="consensus",
    show_default=True,
    help="Point-estimate adjacency matrix.",
)
@click.option("--low-quantile", default=0.10, show_default=True, type=float)
@click.option("--high-quantile", default=0.90, show_default=True, type=float)
@click.option("--min-samples", default=5, show_default=True, type=click.IntRange(1, None))
@click.option(
    "--ci",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
)
@click.option(
    "--attribution-min-abs-effect",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, None),
    help="Minimum group score; scores at or below it are assigned to --weak-label.",
)
@click.option(
    "--attribution-equal-ratio",
    default=0.8,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Classify as mixed when min(group scores)/max(group scores) reaches this value.",
)
@click.option("--weak-label", default="weakly_coupled", show_default=True)
@click.option("--mixed-label", default="mixed", show_default=True)
@click.option("--uncertain-label", default="uncertain", show_default=True)
@click.option(
    "--min-class-support",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Relabel a point class as uncertain when its bootstrap support is below this value.",
)
@click.option(
    "--coupling-aggregation",
    type=click.Choice(_AGGREGATIONS),
    default="rms_abs",
    show_default=True,
)
@click.option(
    "--sensitivity-aggregation",
    type=click.Choice(_AGGREGATIONS),
    default="rms_abs",
    show_default=True,
)
@click.option(
    "--mediated-aggregation",
    type=click.Choice(_AGGREGATIONS),
    default="rms_abs",
    show_default=True,
)
@click.option(
    "--coupling-threshold",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option(
    "--sensitivity-threshold",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option("--low-low-label", default="low_coupling_low_sensitivity", show_default=True)
@click.option("--high-low-label", default="high_coupling_low_sensitivity", show_default=True)
@click.option("--low-high-label", default="low_coupling_high_sensitivity", show_default=True)
@click.option("--high-high-label", default="high_coupling_high_sensitivity", show_default=True)
@click.option("--output-csv", default=None, type=click.Path(path_type=Path))
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--output-table", default="pixel_directlingam_process_attribution", show_default=True)
@click.option("--plot-dir", default=None, type=click.Path(path_type=Path))
@click.option("--attribution-map", default=None, type=click.Path(path_type=Path))
@click.option("--coupling-sensitivity-map", default=None, type=click.Path(path_type=Path))
@click.option("--no-plots", is_flag=True, help="Skip categorical and diagnostic maps.")
@click.option(
    "--plots-only",
    is_flag=True,
    help="Rebuild plots from an existing output CSV without recomputing scores.",
)
@click.option("--diagnostic-maps", is_flag=True, help="Also write continuous score maps.")
@click.option("--figure-width", default=8.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--figure-height", default=8.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--plot-dpi", default=600, show_default=True, type=click.IntRange(72, None))
@click.option("--title-font-size", default=10.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--legend-font-size", default=8.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--title/--no-title", "show_title", default=True, show_default=True)
@click.option("--show", is_flag=True, help="Show plots interactively.")
@click.option("--no-progress", is_flag=True, help="Disable progress bars.")
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=click.IntRange(1, None),
)
@click.option("--chunksize", default=1, show_default=True, type=click.IntRange(1, None))
def per_pixel_directlingam_process_attribution(
    config_path: Path,
    target: str | None,
    outcome_alias: str | None,
    group_a_sources: str,
    group_b_sources: str,
    group_a_label: str,
    group_b_label: str,
    group_a_method: str,
    group_b_method: str,
    upstream_sources: str | None,
    mediators: str | None,
    effect_mode: str,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    attribution_min_abs_effect: float,
    attribution_equal_ratio: float,
    weak_label: str,
    mixed_label: str,
    uncertain_label: str,
    min_class_support: float,
    coupling_aggregation: str,
    sensitivity_aggregation: str,
    mediated_aggregation: str,
    coupling_threshold: float,
    sensitivity_threshold: float,
    low_low_label: str,
    high_low_label: str,
    low_high_label: str,
    high_high_label: str,
    output_csv: Path | None,
    output_db: Path | None,
    output_table: str,
    plot_dir: Path | None,
    attribution_map: Path | None,
    coupling_sensitivity_map: Path | None,
    no_plots: bool,
    plots_only: bool,
    diagnostic_maps: bool,
    figure_width: float,
    figure_height: float,
    plot_dpi: int,
    title_font_size: float,
    legend_font_size: float,
    show_title: bool,
    show: bool,
    no_progress: bool,
    jobs: int,
    chunksize: int,
) -> None:
    """Compare source groups and optionally decompose upstream--mediator coupling."""
    del chunksize  # Kept for CLI compatibility with the other per-pixel scripts.

    if target and outcome_alias and target != outcome_alias:
        raise click.UsageError("--target and --outcome specify different variables")
    if not (0.0 <= low_quantile < high_quantile <= 1.0):
        raise click.BadParameter("Require 0 <= low_quantile < high_quantile <= 1.")
    if plots_only and no_plots:
        raise click.UsageError("--plots-only cannot be combined with --no-plots")

    group_a = _parse_csv(group_a_sources, "--group-a-sources", required=True)
    group_b = _parse_csv(group_b_sources, "--group-b-sources", required=True)
    overlap = sorted(set(group_a) & set(group_b))
    if overlap:
        raise click.BadParameter(
            f"source groups must be disjoint; overlap: {overlap}",
            param_hint="--group-a-sources/--group-b-sources",
        )
    upstream = _parse_csv(upstream_sources, "--upstream-sources", required=False)
    mediator_list = _parse_csv(mediators, "--mediators", required=False)
    if bool(upstream) != bool(mediator_list):
        raise click.UsageError("--upstream-sources and --mediators must be supplied together")

    cfg = load_config(
        config_path=config_path,
        target_override=target,
        outcome_override=outcome_alias,
        point_matrix_override=point_matrix,
        plot_dir_override=plot_dir,
    )
    experiment_dir = cfg.experiment_dir
    location = cfg.location_name
    output_csv_path = _resolve_path(
        experiment_dir,
        output_csv,
        f"{location}_directlingam_process_attribution.csv",
    )
    output_db_path = _resolve_path(
        experiment_dir,
        output_db,
        f"{location}_directlingam_process_attribution.duckdb",
    )
    plot_dir_path = _resolve_path(
        experiment_dir,
        plot_dir,
        f"{location}_directlingam_process_attribution_plots",
    )
    attribution_map_path = _resolve_path(
        plot_dir_path,
        attribution_map,
        "group_attribution.png",
    )
    coupling_map_path = _resolve_path(
        plot_dir_path,
        coupling_sensitivity_map,
        "coupling_sensitivity.png",
    )

    if plots_only:
        if not output_csv_path.exists():
            raise click.ClickException(f"--plots-only output CSV does not exist: {output_csv_path}")
        results_df = pd.read_csv(output_csv_path)
        click.echo(f"Plot-only mode: loaded {len(results_df):,} rows from {output_csv_path}")
    else:
        if not no_progress:
            click.echo("Loading shifted time series and graph tables...")
        timeseries_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)
        bundles = list(
            progress_bar(
                iter_pixel_groups(cfg, timeseries_df=timeseries_df, graph_df=graph_df),
                total=len(graph_df),
                desc="Preparing process-attribution tasks",
                unit="pixel",
                disabled=no_progress or len(graph_df) == 0,
            )
        )
        tasks = [
            (
                bundle,
                cfg.target_col,
                group_a,
                group_b,
                group_a_label,
                group_b_label,
                group_a_method,
                group_b_method,
                upstream,
                mediator_list,
                effect_mode,
                point_matrix,
                low_quantile,
                high_quantile,
                min_samples,
                ci,
                attribution_min_abs_effect,
                attribution_equal_ratio,
                coupling_aggregation,
                sensitivity_aggregation,
                mediated_aggregation,
                coupling_threshold,
                sensitivity_threshold,
                low_low_label,
                high_low_label,
                low_high_label,
                high_high_label,
                weak_label,
                mixed_label,
                uncertain_label,
                min_class_support,
            )
            for bundle in bundles
        ]
        if jobs == 1:
            rows = [
                _analyze_pixel_task(task)
                for task in progress_bar(
                    tasks,
                    total=len(tasks),
                    desc="Analyzing pixels",
                    unit="pixel",
                    disabled=no_progress or len(tasks) == 0,
                )
            ]
        else:
            rows = []
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_analyze_pixel_task, task) for task in tasks]
                for future in progress_bar(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Analyzing pixels using {jobs} workers",
                    unit="pixel",
                    disabled=no_progress or len(futures) == 0,
                ):
                    rows.append(future.result())
        if not rows:
            raise click.ClickException("No process-attribution rows were produced")
        results_df = pd.DataFrame(rows)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(output_db_path))
        try:
            write_dataframe_table(connection, results_df, output_table)
        finally:
            connection.close()

    written_plots: list[Path] = []
    if not no_plots:
        attribution = _plot_categorical_map(
            results_df,
            cfg.row_col_cols,
            "attribution_class",
            attribution_map_path,
            title=f"Grouped causal attribution of {cfg.target_col}",
            class_order=[weak_label, group_a_label, group_b_label, mixed_label, uncertain_label],
            figure_width=figure_width,
            figure_height=figure_height,
            dpi=plot_dpi,
            title_font_size=title_font_size,
            legend_font_size=legend_font_size,
            show_title=show_title,
            show=show,
        )
        if attribution is not None:
            written_plots.append(attribution)
        if upstream and mediator_list and "coupling_sensitivity_class" in results_df.columns:
            coupling_plot = _plot_categorical_map(
                results_df,
                cfg.row_col_cols,
                "coupling_sensitivity_class",
                coupling_map_path,
                title=f"Coupling--sensitivity classification for {cfg.target_col}",
                class_order=[
                    low_low_label,
                    high_low_label,
                    low_high_label,
                    high_high_label,
                    uncertain_label,
                ],
                figure_width=figure_width,
                figure_height=figure_height,
                dpi=plot_dpi,
                title_font_size=title_font_size,
                legend_font_size=legend_font_size,
                show_title=show_title,
                show=show,
            )
            if coupling_plot is not None:
                written_plots.append(coupling_plot)
        if diagnostic_maps:
            written_plots.extend(
                _plot_diagnostic_maps(
                    results_df,
                    cfg.row_col_cols,
                    plot_dir_path,
                    figure_width=figure_width,
                    figure_height=figure_height,
                    dpi=plot_dpi,
                    title_font_size=title_font_size,
                    show_title=show_title,
                    show=show,
                )
            )

    successful = _successful_rows(results_df)
    failed = len(results_df) - len(successful)
    click.echo(f"Target: {cfg.target_col}")
    click.echo(f"Group A: {group_a_label} = {', '.join(group_a)} ({group_a_method})")
    click.echo(f"Group B: {group_b_label} = {', '.join(group_b)} ({group_b_method})")
    click.echo(f"Effect mode: {effect_mode}")
    click.echo(f"Point matrix: {point_matrix}")
    click.echo(f"Quantile contrast: Q{high_quantile:.2f} - Q{low_quantile:.2f}")
    if upstream:
        click.echo(f"Upstream sources: {', '.join(upstream)}")
        click.echo(f"Mediators: {', '.join(mediator_list)}")
    click.echo(f"Results CSV: {output_csv_path}")
    if not plots_only:
        click.echo(f"Output DuckDB: {output_db_path}::{output_table}")
    click.echo(f"Failed rows: {failed} / {len(results_df)}")
    for path in written_plots:
        click.echo(f"Plot: {path}")


if __name__ == "__main__":
    per_pixel_directlingam_process_attribution()
