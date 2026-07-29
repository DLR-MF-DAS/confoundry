#!/usr/bin/env python3
"""Simulate dynamic interventions in saved per-pixel VAR-LiNGAM models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
    infer_structural_innovations,
    iter_var_pixel_bundles,
    load_var_postprocess_config,
    load_var_timeseries_and_graphs,
    matrices_from_graph_row,
    quantile_contrast,
    simulate_structural_var,
    summarize_bootstrap,
    validated_pixel_observations,
)
from confoundry.visualize_directlingam_diagnostics import (
    PUBLICATION_STYLE,
    parse_label_overrides,
    publication_variable_label,
)


POINT_MATRIX_CHOICES = ("raw", "consensus", "bootstrap_mean")
MODES = ("counterfactual", "interventional_mean")
VALUE_KINDS = ("delta", "fixed", "quantile", "zdelta", "qdelta")


@dataclass(frozen=True)
class InterventionValue:
    """One intervention value rule."""

    kind: str
    value: float


@dataclass(frozen=True)
class Intervention:
    """One variable intervention within a named scenario."""

    variable: str
    value: InterventionValue


@dataclass(frozen=True)
class Scenario:
    """A simultaneous set of interventions."""

    name: str
    interventions: tuple[Intervention, ...]


def parse_intervention_value(raw: str) -> InterventionValue:
    """Parse fixed/delta/quantile/zdelta/qdelta specifications."""
    kind, separator, value_text = raw.partition(":")
    if not separator:
        kind = "fixed"
        value_text = raw
    kind = kind.strip().lower()
    if kind not in VALUE_KINDS:
        raise click.BadParameter(
            f"Unknown intervention value kind {kind!r}; expected one of "
            f"{VALUE_KINDS}."
        )
    try:
        value = float(value_text)
    except ValueError as exc:
        raise click.BadParameter(
            f"Intervention value must be numeric: {raw!r}"
        ) from exc
    if kind == "quantile" and not 0.0 <= value <= 1.0:
        raise click.BadParameter("quantile must be between 0 and 1")
    return InterventionValue(kind=kind, value=value)


def build_scenarios(
    definitions: Sequence[tuple[str, str, str]],
) -> list[Scenario]:
    """Group repeated CLI intervention definitions by scenario name."""
    grouped: dict[str, list[Intervention]] = {}
    for scenario_name, variable, raw_value in definitions:
        grouped.setdefault(str(scenario_name), []).append(
            Intervention(
                variable=str(variable),
                value=parse_intervention_value(raw_value),
            )
        )
    if not grouped:
        raise click.BadParameter(
            "At least one --intervention SCENARIO VARIABLE SPEC is required."
        )
    scenarios: list[Scenario] = []
    for name, interventions in grouped.items():
        variables = [item.variable for item in interventions]
        if len(set(variables)) != len(variables):
            raise click.BadParameter(
                f"Scenario {name!r} intervenes on a variable more than once."
            )
        scenarios.append(
            Scenario(name=name, interventions=tuple(interventions))
        )
    return scenarios


def _parse_targets(values: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for value in values:
        targets.extend(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise click.BadParameter("At least one --target is required.")
    return targets


def evaluate_intervention_value(
    specification: InterventionValue,
    *,
    reference_values: np.ndarray,
    factual_value: float,
    low_quantile: float,
    high_quantile: float,
) -> float:
    """Evaluate one intervention rule for a pixel and factual month."""
    finite = np.asarray(reference_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("No finite reference values for intervention")
    kind = specification.kind
    value = specification.value
    if kind == "fixed":
        return float(value)
    if kind == "delta":
        return float(factual_value + value)
    if kind == "quantile":
        return float(np.quantile(finite, value))
    if kind == "zdelta":
        standard_deviation = float(np.std(finite, ddof=1))
        return float(factual_value + value * standard_deviation)
    if kind == "qdelta":
        _, _, contrast = quantile_contrast(
            finite,
            low_quantile,
            high_quantile,
        )
        return float(factual_value + value * contrast)
    raise ValueError(f"Unknown intervention kind: {kind}")


def _intervention_schedule(
    scenario: Scenario,
    *,
    labels: Sequence[str],
    observations: np.ndarray,
    factual_path: np.ndarray,
    duration: int,
    horizon: int,
    low_quantile: float,
    high_quantile: float,
) -> list[dict[int, float]]:
    index = {label: position for position, label in enumerate(labels)}
    schedule: list[dict[int, float]] = []
    for step in range(horizon + 1):
        values: dict[int, float] = {}
        if step < duration:
            for intervention in scenario.interventions:
                variable_index = index[intervention.variable]
                values[variable_index] = evaluate_intervention_value(
                    intervention.value,
                    reference_values=observations[:, variable_index],
                    factual_value=float(
                        factual_path[step, variable_index]
                    ),
                    low_quantile=low_quantile,
                    high_quantile=high_quantile,
                )
        schedule.append(values)
    return schedule


def _scenario_json(scenario: Scenario) -> str:
    return json.dumps(
        {
            intervention.variable: (
                f"{intervention.value.kind}:"
                f"{intervention.value.value}"
            )
            for intervention in scenario.interventions
        },
        sort_keys=True,
    )


def _counterfactual_context(
    observations: np.ndarray,
    metadata: pd.DataFrame,
    *,
    lag_count: int,
    horizon: int,
    start_year: int,
    start_month: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    matches = np.flatnonzero(
        (metadata["year"].astype(int).to_numpy() == int(start_year))
        & (
            metadata["month"].astype(int).to_numpy()
            == int(start_month)
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {start_year}-{start_month:02d}, "
            f"found {len(matches)}"
        )
    start = int(matches[0])
    if start < lag_count:
        raise ValueError(
            "Intervention month does not have enough preceding lag history"
        )
    if start + horizon >= len(observations):
        raise ValueError(
            "Intervention horizon extends beyond the factual time series"
        )
    history = observations[start - lag_count : start]
    factual = observations[start : start + horizon + 1]
    return start, history, factual


def _simulate_one_model(
    *,
    contemporaneous: np.ndarray,
    lagged: np.ndarray,
    observations: np.ndarray,
    mode: str,
    start_index: int | None,
    factual_path: np.ndarray,
    schedule: Sequence[Mapping[int, float]],
    horizon: int,
) -> np.ndarray:
    lag_count = lagged.shape[0]
    n_features = contemporaneous.shape[0]
    if mode == "counterfactual":
        assert start_index is not None
        innovations_all = infer_structural_innovations(
            observations,
            contemporaneous,
            lagged,
        )
        innovations = innovations_all[
            start_index : start_index + horizon + 1
        ]
        history = observations[
            start_index - lag_count : start_index
        ]
    elif mode == "interventional_mean":
        innovations = np.zeros(
            (horizon + 1, n_features),
            dtype=float,
        )
        history = np.zeros((lag_count, n_features), dtype=float)
    else:
        raise ValueError(f"Unknown intervention mode: {mode!r}")
    return simulate_structural_var(
        contemporaneous,
        lagged,
        history,
        innovations,
        schedule,
    )


def analyze_intervention_pixel(
    bundle: VARPixelBundle,
    *,
    targets: Sequence[str],
    scenarios: Sequence[Scenario],
    mode: str,
    horizon: int,
    duration: int,
    start_year: int | None,
    start_month: int | None,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    stability_threshold: float,
    bootstrap_limit: int,
) -> list[dict[str, Any]]:
    """Simulate all requested temporal interventions for one pixel."""
    try:
        matrices = matrices_from_graph_row(
            bundle.graph_row,
            point_matrix=point_matrix,
            bootstrap_limit=bootstrap_limit,
        )
        labels = list(matrices.labels)
        required = set(targets)
        for scenario in scenarios:
            required.update(
                intervention.variable
                for intervention in scenario.interventions
            )
        unknown = sorted(required - set(labels))
        if unknown:
            raise ValueError(f"Variables are not in graph: {unknown}")

        complete, observations = validated_pixel_observations(
            bundle,
            labels,
            min_samples=min_samples,
        )
        lag_count = matrices.lagged.shape[0]

        if mode == "counterfactual":
            if start_year is None or start_month is None:
                raise ValueError(
                    "counterfactual mode requires --start-year and "
                    "--start-month"
                )
            start_index, _, factual_path = _counterfactual_context(
                observations,
                complete,
                lag_count=lag_count,
                horizon=horizon,
                start_year=start_year,
                start_month=start_month,
            )
        else:
            start_index = None
            factual_path = np.zeros(
                (horizon + 1, len(labels)),
                dtype=float,
            )

        _, _, point_radius = dynamic_effect_matrices(
            matrices.contemporaneous,
            matrices.lagged,
            0,
        )
        point_stable = bool(point_radius < stability_threshold)
        (
            _,
            _,
            bootstrap_radii,
            bootstrap_valid,
        ) = bootstrap_dynamic_effect_matrices(
            matrices.bootstrap_contemporaneous,
            matrices.bootstrap_lagged,
            0,
        )
        bootstrap_stable = (
            bootstrap_valid
            & np.isfinite(bootstrap_radii)
            & (bootstrap_radii < stability_threshold)
        )

        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            schedule = _intervention_schedule(
                scenario,
                labels=labels,
                observations=observations,
                factual_path=factual_path,
                duration=duration,
                horizon=horizon,
                low_quantile=low_quantile,
                high_quantile=high_quantile,
            )
            counterfactual = _simulate_one_model(
                contemporaneous=matrices.contemporaneous,
                lagged=matrices.lagged,
                observations=observations,
                mode=mode,
                start_index=start_index,
                factual_path=factual_path,
                schedule=schedule,
                horizon=horizon,
            )

            bootstrap_paths: list[np.ndarray] = []
            for index in np.flatnonzero(bootstrap_stable):
                try:
                    path = _simulate_one_model(
                        contemporaneous=(
                            matrices.bootstrap_contemporaneous[index]
                        ),
                        lagged=matrices.bootstrap_lagged[index],
                        observations=observations,
                        mode=mode,
                        start_index=start_index,
                        factual_path=factual_path,
                        schedule=schedule,
                        horizon=horizon,
                    )
                except (np.linalg.LinAlgError, ValueError):
                    continue
                if np.all(np.isfinite(path)):
                    bootstrap_paths.append(path)
            if bootstrap_paths:
                bootstrap_array = np.stack(bootstrap_paths)
            else:
                bootstrap_array = np.empty(
                    (0, horizon + 1, len(labels)),
                    dtype=float,
                )

            for target in targets:
                target_index = labels.index(target)
                factual_target = factual_path[:, target_index]
                point_target = counterfactual[:, target_index]
                point_effect = point_target - factual_target
                bootstrap_target = bootstrap_array[:, :, target_index]
                bootstrap_effect = (
                    bootstrap_target - factual_target[np.newaxis, :]
                )
                cumulative_point = np.cumsum(point_effect)
                cumulative_bootstrap = np.cumsum(
                    bootstrap_effect,
                    axis=1,
                )

                for step in range(horizon + 1):
                    effect_summary = summarize_bootstrap(
                        bootstrap_effect[:, step],
                        ci=ci,
                    )
                    counterfactual_summary = summarize_bootstrap(
                        bootstrap_target[:, step],
                        ci=ci,
                    )
                    cumulative_summary = summarize_bootstrap(
                        cumulative_bootstrap[:, step],
                        ci=ci,
                    )
                    row = {
                        **bundle.coords,
                        "model_type": "varlingam",
                        "scenario": scenario.name,
                        "target": target,
                        "horizon": int(step),
                        "mode": mode,
                        "start_year": start_year,
                        "start_month": start_month,
                        "intervention_duration": int(duration),
                        "interventions_json": _scenario_json(scenario),
                        "active_intervention": bool(step < duration),
                        "point_matrix": point_matrix,
                        "point_stability_radius": float(point_radius),
                        "point_stable": point_stable,
                        "factual_value": float(factual_target[step]),
                        "counterfactual_value": float(point_target[step]),
                        "effect": float(point_effect[step]),
                        "cumulative_effect": float(cumulative_point[step]),
                        "n_samples": int(len(observations)),
                        "n_bootstrap_total": int(
                            len(matrices.bootstrap_contemporaneous)
                        ),
                        "n_bootstrap_matrix_valid": int(
                            np.sum(bootstrap_valid)
                        ),
                        "n_bootstrap_stable": int(
                            np.sum(bootstrap_stable)
                        ),
                        "n_bootstrap_simulation_successful": int(
                            len(bootstrap_array)
                        ),
                        "error": None,
                    }
                    row.update(
                        {
                            f"effect_{key}": value
                            for key, value in effect_summary.items()
                        }
                    )
                    row.update(
                        {
                            f"counterfactual_value_{key}": value
                            for key, value in counterfactual_summary.items()
                        }
                    )
                    row.update(
                        {
                            f"cumulative_effect_{key}": value
                            for key, value in cumulative_summary.items()
                        }
                    )
                    rows.append(row)
        return rows
    except Exception as exc:
        return [
            {
                **bundle.coords,
                "model_type": "varlingam",
                "scenario": None,
                "target": None,
                "horizon": None,
                "mode": mode,
                "point_matrix": point_matrix,
                "point_stable": False,
                "error": str(exc),
            }
        ]


def _analyze_intervention_task(
    args: tuple[Any, ...],
) -> list[dict[str, Any]]:
    (
        bundle,
        targets,
        scenarios,
        mode,
        horizon,
        duration,
        start_year,
        start_month,
        point_matrix,
        low_quantile,
        high_quantile,
        min_samples,
        ci,
        stability_threshold,
        bootstrap_limit,
    ) = args
    return analyze_intervention_pixel(
        bundle,
        targets=targets,
        scenarios=scenarios,
        mode=mode,
        horizon=horizon,
        duration=duration,
        start_year=start_year,
        start_month=start_month,
        point_matrix=point_matrix,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        min_samples=min_samples,
        ci=ci,
        stability_threshold=stability_threshold,
        bootstrap_limit=bootstrap_limit,
    )


def aggregate_interventions(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intervention responses across stable pixel models."""
    summary_columns = [
        "scenario",
        "target",
        "horizon",
        "mode",
        "n_pixels",
        "effect_mean",
        "effect_median",
        "effect_q05",
        "effect_q95",
        "effect_fraction_positive",
        "cumulative_effect_median",
        "cumulative_effect_q05",
        "cumulative_effect_q95",
        "pixel_fraction_bootstrap_ci_excludes_zero",
        "median_bootstrap_sd",
    ]
    valid = results[
        results["error"].isna()
        & results["point_stable"].astype(bool)
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=summary_columns)

    def summarize(group: pd.DataFrame) -> pd.Series:
        effects = group["effect"].to_numpy(dtype=float)
        cumulative = group["cumulative_effect"].to_numpy(dtype=float)
        return pd.Series(
            {
                "n_pixels": int(len(group)),
                "effect_mean": float(np.nanmean(effects)),
                "effect_median": float(np.nanmedian(effects)),
                "effect_q05": float(np.nanquantile(effects, 0.05)),
                "effect_q95": float(np.nanquantile(effects, 0.95)),
                "effect_fraction_positive": float(
                    np.nanmean(effects > 0.0)
                ),
                "cumulative_effect_median": float(
                    np.nanmedian(cumulative)
                ),
                "cumulative_effect_q05": float(
                    np.nanquantile(cumulative, 0.05)
                ),
                "cumulative_effect_q95": float(
                    np.nanquantile(cumulative, 0.95)
                ),
                "pixel_fraction_bootstrap_ci_excludes_zero": float(
                    np.mean(
                        group[
                            "effect_boot_ci_excludes_zero"
                        ].astype(bool)
                    )
                ),
                "median_bootstrap_sd": float(
                    np.nanmedian(
                        group["effect_boot_sd"].to_numpy(dtype=float)
                    )
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["scenario", "target", "horizon", "mode"]
    for key, group in valid.groupby(group_columns, sort=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, normalized_key, strict=True))
        row.update(summarize(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows, columns=summary_columns)


def plot_intervention_trajectories(
    summary: pd.DataFrame,
    output_dir: Path,
    label_overrides: Mapping[str, str] | None = None,
) -> list[Path]:
    """Plot spatial median intervention effects and cumulative effects."""
    if summary.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (scenario, target), group in summary.groupby(
        ["scenario", "target"],
        sort=True,
    ):
        group = group.sort_values("horizon")
        horizon = group["horizon"].to_numpy(dtype=int)
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
                    "effect_median",
                    "effect_q05",
                    "effect_q95",
                    "Horizon-specific response",
                ),
                (
                    axes[1],
                    "cumulative_effect_median",
                    "cumulative_effect_q05",
                    "cumulative_effect_q95",
                    "Cumulative response",
                ),
            ]
            for axis, median, lower, upper, title in panels:
                axis.fill_between(
                    horizon,
                    group[lower].to_numpy(dtype=float),
                    group[upper].to_numpy(dtype=float),
                    color="#E45756",
                    alpha=0.22,
                    label="Spatial 5–95% range",
                )
                axis.plot(
                    horizon,
                    group[median].to_numpy(dtype=float),
                    color="#A32322",
                    marker="o",
                    linewidth=1.8,
                    label="Spatial median",
                )
                axis.axhline(0.0, color="0.35", linewidth=0.8)
                axis.set_xlabel("Months after intervention")
                axis.set_ylabel("Target anomaly response")
                axis.set_title(title)
                axis.grid(alpha=0.2)
            axes[0].legend(frameon=False, fontsize=8)
            target_label = publication_variable_label(
                target,
                label_overrides,
            )
            fig.suptitle(f"{scenario}: {target_label}")
            fig.tight_layout()
            stem = (
                f"varlingam_intervention_"
                f"{safe_filename(str(scenario))}_"
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
    "targets_raw",
    multiple=True,
    required=True,
    help="Target variable; repeat or pass comma-separated variables.",
)
@click.option(
    "--intervention",
    type=(str, str, str),
    multiple=True,
    required=True,
    metavar="SCENARIO VARIABLE SPEC",
    help=(
        "Intervention definition, e.g. wetting soil_moisture_resid "
        "delta:0.05. Repeat variables with the same scenario for a joint intervention."
    ),
)
@click.option(
    "--mode",
    type=click.Choice(MODES),
    default="interventional_mean",
    show_default=True,
)
@click.option("--horizon", default=12, show_default=True, type=click.IntRange(0, None))
@click.option(
    "--intervention-duration",
    default=1,
    show_default=True,
    type=click.IntRange(1, None),
    help="Number of consecutive months during which the intervention is imposed.",
)
@click.option("--start-year", default=None, type=int, help="Required for counterfactual mode.")
@click.option(
    "--start-month",
    default=None,
    type=click.IntRange(1, 12),
    help="Required for counterfactual mode.",
)
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
)
@click.option(
    "--bootstrap-limit",
    default=0,
    show_default=True,
    type=click.IntRange(0, None),
    help="Use at most this many bootstrap replicates per pixel; 0 uses all.",
)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option("--graphs-db", default=None, type=click.Path(path_type=Path))
@click.option("--graphs-table", default=None)
@click.option("--output-csv", default=None, type=click.Path(path_type=Path))
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--output-table", default="pixel_varlingam_interventions", show_default=True)
@click.option("--summary-table", default="varlingam_intervention_summary", show_default=True)
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
def per_pixel_varlingam_interventions(
    config_path: Path,
    targets_raw: tuple[str, ...],
    intervention: tuple[tuple[str, str, str], ...],
    mode: str,
    horizon: int,
    intervention_duration: int,
    start_year: int | None,
    start_month: int | None,
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
    output_table: str,
    summary_table: str,
    plot_dir: Path | None,
    variable_labels: tuple[str, ...],
    no_plots: bool,
    jobs: int,
) -> None:
    """Run dynamic VAR-LiNGAM interventions and counterfactual trajectories."""
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise click.BadParameter(
            "Require 0 <= low-quantile < high-quantile <= 1."
        )
    label_overrides = parse_label_overrides(variable_labels)
    if mode == "counterfactual" and (
        start_year is None or start_month is None
    ):
        raise click.BadParameter(
            "counterfactual mode requires --start-year and --start-month"
        )
    targets = _parse_targets(targets_raw)
    scenarios = build_scenarios(intervention)
    config = load_var_postprocess_config(
        config_path,
        input_db=input_db,
        input_table=input_table,
        graphs_db=graphs_db,
        graphs_table=graphs_table,
    )
    time_series, graph_df, labels = load_var_timeseries_and_graphs(config)
    requested = set(targets)
    for scenario in scenarios:
        requested.update(
            item.variable for item in scenario.interventions
        )
    unknown = sorted(requested - set(labels))
    if unknown:
        raise click.BadParameter(f"Unknown variables: {unknown}")

    bundles = list(
        iter_var_pixel_bundles(config, time_series, graph_df)
    )
    tasks = [
        (
            bundle,
            targets,
            scenarios,
            mode,
            horizon,
            intervention_duration,
            start_year,
            start_month,
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
        nested = [_analyze_intervention_task(task) for task in tasks]
    else:
        nested = process_map(
            _analyze_intervention_task,
            tasks,
            max_workers=jobs,
            chunksize=1,
            desc="VAR-LiNGAM interventions",
        )
    rows = [row for result in nested for row in result]
    if not rows:
        raise click.ClickException(
            "No VAR-LiNGAM intervention rows were produced."
        )
    results = pd.DataFrame(rows)
    summary = aggregate_interventions(results)

    output_csv_path = _resolve_output_path(
        config.experiment_dir,
        output_csv,
        f"{config.location_name}_varlingam_interventions.csv",
    )
    output_db_path = _resolve_output_path(
        config.experiment_dir,
        output_db,
        f"{config.location_name}_varlingam_interventions.duckdb",
    )
    plot_dir_path = _resolve_output_path(
        config.experiment_dir,
        plot_dir,
        f"{config.location_name}_varlingam_intervention_plots",
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv_path, index=False)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_db_path))
    try:
        write_dataframe_table(con, results, output_table)
        write_dataframe_table(con, summary, summary_table)
        metadata = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_path": str(config.config_path),
                    "input_db": str(config.input_db),
                    "graphs_db": str(config.graphs_db),
                    "targets_json": json.dumps(targets),
                    "scenarios_json": json.dumps(
                        {
                            scenario.name: json.loads(
                                _scenario_json(scenario)
                            )
                            for scenario in scenarios
                        },
                        sort_keys=True,
                    ),
                    "mode": mode,
                    "horizon": int(horizon),
                    "intervention_duration": int(
                        intervention_duration
                    ),
                    "start_year": start_year,
                    "start_month": start_month,
                    "point_matrix": point_matrix,
                    "ci": float(ci),
                    "stability_threshold": float(stability_threshold),
                    "bootstrap_limit": int(bootstrap_limit),
                    "n_graph_rows": int(len(graph_df)),
                    "n_result_rows": int(len(results)),
                }
            ]
        )
        write_dataframe_table(
            con,
            metadata,
            "varlingam_intervention_run_metadata",
        )
    finally:
        con.close()

    written_plots = (
        []
        if no_plots
        else plot_intervention_trajectories(
            summary,
            plot_dir_path,
            label_overrides,
        )
    )
    failed = int(results["error"].notna().sum())
    unstable = int(
        results.loc[results["error"].isna(), "point_stable"]
        .eq(False)
        .sum()
    )
    click.echo(f"Wrote interventions: {output_csv_path}")
    click.echo(
        f"Wrote intervention database: {output_db_path}::{output_table}"
    )
    click.echo(
        f"Wrote aggregate table: {output_db_path}::{summary_table}"
    )
    click.echo(f"Failed rows: {failed}; unstable point-result rows: {unstable}")
    if written_plots:
        click.echo(f"Wrote plots: {plot_dir_path}")


if __name__ == "__main__":
    per_pixel_varlingam_interventions()
