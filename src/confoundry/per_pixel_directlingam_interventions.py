#!/usr/bin/env python3
"""Per-pixel interventions and counterfactuals for saved DirectLiNGAM models.

The script is a post-processing companion to ``per_pixel_graph_discovery.py``.
It uses the existing Confoundry experiment YAML only to locate and interpret
shifted time-series and graph tables. All causal-analysis choices are supplied
as command-line options.

Supported analyses
------------------
* Population interventions: expected response under ``do(X=x)`` with zero-mean
  exogenous disturbances.
* Observation-specific counterfactuals: abduction--action--prediction using the
  factual observation's inferred disturbances.
* Simultaneous hard interventions on any number of variables.
* Soft/mechanism interventions that set, add to, or scale structural edges.
* Goal seeking: solve jointly for intervention values required to reach one or
  more target goals.
* Bootstrap propagation using every saved adjacency matrix.
* Exact source-contribution decomposition for simultaneous hard interventions.
* Point-estimate path decomposition after cutting incoming edges into hard-
  intervened variables.

The fitted linear SCM is interpreted as ``z = B z + e`` on variables centered
by their per-pixel sample means. This matches the centered linear equations
represented by the DirectLiNGAM adjacency matrix. Counterfactuals preserve the
factual disturbance vector for each point/bootstrap model.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import click
import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from confoundry.per_pixel_directlingam_analysis import (
        PixelBundle,
        _bootstrap_matrices_from_row,
        _grid_from_results,
        _point_matrix_from_row,
        _quantile_contrast,
        _summary,
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
        iter_pixel_groups,
        load_config,
        load_shifted_timeseries_and_graphs,
        progress_bar,
    )
    from per_pixel_graph_discovery import write_dataframe_table  # type: ignore


_POINT_MATRICES = ("raw", "consensus", "bootstrap_mean")
_MODES = ("counterfactual", "interventional_mean")
_AGGREGATIONS = ("none", "mean", "median", "sum")
_MECHANISM_OPERATIONS = ("scale", "set", "add")
_FILTER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|=|!=|<=|>=|<|>)\s*(.*?)\s*$")
_EDGE_RIGHT_RE = re.compile(r"^\s*([^\s<>-]+)\s*->\s*([^\s<>-]+)\s*$")
_EDGE_LEFT_RE = re.compile(r"^\s*([^\s<>-]+)\s*<-\s*([^\s<>-]+)\s*$")


@dataclass(frozen=True)
class FilterSpec:
    column: str
    operator: str
    value: Any


@dataclass(frozen=True)
class ValueSpec:
    kind: str
    value: float | None = None
    reference: "ValueSpec | None" = None


@dataclass(frozen=True)
class HardIntervention:
    variable: str
    value_spec: ValueSpec


@dataclass(frozen=True)
class MechanismIntervention:
    parent: str
    child: str
    operation: str
    value: float


@dataclass(frozen=True)
class GoalSeek:
    variable: str
    target: str
    goal_spec: ValueSpec


@dataclass(frozen=True)
class Scenario:
    name: str
    interventions: tuple[HardIntervention, ...]
    mechanisms: tuple[MechanismIntervention, ...]
    goals: tuple[GoalSeek, ...]


@dataclass(frozen=True)
class ContextResult:
    factual: np.ndarray
    mechanism_only: np.ndarray
    counterfactual: np.ndarray
    do_values_centered: Mapping[str, float]
    hard_contributions: Mapping[str, np.ndarray]
    mechanism_contribution: np.ndarray
    required_values: Mapping[str, float]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def _resolve_path(base_dir: Path, override: Path | None, default_name: str) -> Path:
    path = override if override is not None else Path(default_name)
    path = path.expanduser()
    return path if path.is_absolute() else base_dir / path


def _parse_csv(value: str | None, option_name: str, *, required: bool = False) -> list[str]:
    values = [] if value is None else [part.strip() for part in value.split(",") if part.strip()]
    values = list(dict.fromkeys(values))
    if required and not values:
        raise click.BadParameter("must contain at least one comma-separated value", param_hint=option_name)
    return values


def _flatten_targets(values: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for value in values:
        targets.extend(_parse_csv(value, "--target", required=True))
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise click.BadParameter("at least one target is required", param_hint="--target")
    return targets


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null", "nan"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _parse_filter(raw: str, option_name: str) -> FilterSpec:
    match = _FILTER_RE.match(raw)
    if not match:
        raise click.BadParameter(
            "expected COLUMN=VALUE or COLUMN<OP>VALUE, e.g. year=2022 or month>=6",
            param_hint=option_name,
        )
    column, operator, value = match.groups()
    return FilterSpec(column=column, operator="==" if operator == "=" else operator, value=_parse_scalar(value))


def _apply_filters(df: pd.DataFrame, filters: Sequence[FilterSpec]) -> pd.DataFrame:
    result = df
    for spec in filters:
        if spec.column not in result.columns:
            raise ValueError(f"filter column {spec.column!r} is not present")
        series = result[spec.column]
        value = spec.value
        if spec.operator == "==":
            mask = series.isna() if value is None else series == value
        elif spec.operator == "!=":
            mask = series.notna() if value is None else series != value
        elif spec.operator == "<":
            mask = series < value
        elif spec.operator == "<=":
            mask = series <= value
        elif spec.operator == ">":
            mask = series > value
        elif spec.operator == ">=":
            mask = series >= value
        else:  # pragma: no cover - parser prevents this
            raise ValueError(f"unsupported filter operator: {spec.operator}")
        result = result.loc[mask.fillna(False)]
    return result.copy()


def _parse_value_spec(raw: str, option_name: str) -> ValueSpec:
    text = raw.strip()
    lower = text.lower()
    if lower in {"mean", "median", "climatology_mean", "climatology_median"}:
        return ValueSpec(kind=lower)
    if lower.startswith("fraction_to:"):
        parts = text.split(":", 2)
        if len(parts) != 3:
            raise click.BadParameter(
                "fraction_to requires fraction and reference, e.g. fraction_to:0.5:climatology_median",
                param_hint=option_name,
            )
        try:
            fraction = float(parts[1])
        except ValueError as exc:
            raise click.BadParameter("invalid fraction", param_hint=option_name) from exc
        reference = _parse_value_spec(parts[2], option_name)
        return ValueSpec(kind="fraction_to", value=fraction, reference=reference)
    if ":" not in text:
        try:
            return ValueSpec(kind="fixed", value=float(text))
        except ValueError as exc:
            raise click.BadParameter(
                "unknown value specification; use fixed:VALUE, mean, median, quantile:Q, "
                "climatology_mean, climatology_median, climatology_quantile:Q, delta:D, "
                "scale:F, qdelta:F, zdelta:F, or fraction_to:F:REFERENCE",
                param_hint=option_name,
            ) from exc
    kind, raw_value = text.split(":", 1)
    kind = kind.lower().strip()
    aliases = {"value": "fixed", "set": "fixed", "q": "quantile", "clim_mean": "climatology_mean", "clim_median": "climatology_median", "clim_q": "climatology_quantile"}
    kind = aliases.get(kind, kind)
    allowed = {
        "fixed",
        "quantile",
        "climatology_quantile",
        "delta",
        "scale",
        "qdelta",
        "zdelta",
    }
    if kind not in allowed:
        raise click.BadParameter(f"unknown value-spec kind {kind!r}", param_hint=option_name)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise click.BadParameter(f"{kind} requires a numeric value", param_hint=option_name) from exc
    if kind in {"quantile", "climatology_quantile"} and not 0.0 <= value <= 1.0:
        raise click.BadParameter("quantiles must lie in [0, 1]", param_hint=option_name)
    return ValueSpec(kind=kind, value=value)


def _parse_edge(raw: str, option_name: str) -> tuple[str, str]:
    right = _EDGE_RIGHT_RE.match(raw)
    if right:
        return right.group(1), right.group(2)
    left = _EDGE_LEFT_RE.match(raw)
    if left:
        child, parent = left.groups()
        return parent, child
    raise click.BadParameter(
        "edge must be written as PARENT->CHILD or CHILD<-PARENT",
        param_hint=option_name,
    )


def _parse_mechanism_spec(raw: str, option_name: str) -> tuple[str, float]:
    if ":" not in raw:
        raise click.BadParameter("mechanism spec must be scale:F, set:V, or add:D", param_hint=option_name)
    operation, value_raw = raw.split(":", 1)
    operation = operation.lower().strip()
    if operation not in _MECHANISM_OPERATIONS:
        raise click.BadParameter(
            f"operation must be one of {_MECHANISM_OPERATIONS}", param_hint=option_name
        )
    try:
        value = float(value_raw)
    except ValueError as exc:
        raise click.BadParameter("mechanism value must be numeric", param_hint=option_name) from exc
    return operation, value


def _build_scenarios(
    intervention_rows: Sequence[tuple[str, str, str]],
    mechanism_rows: Sequence[tuple[str, str, str]],
    goal_rows: Sequence[tuple[str, str, str, str]],
) -> list[Scenario]:
    interventions: dict[str, list[HardIntervention]] = defaultdict(list)
    mechanisms: dict[str, list[MechanismIntervention]] = defaultdict(list)
    goals: dict[str, list[GoalSeek]] = defaultdict(list)
    names: list[str] = []

    def remember(name: str) -> None:
        if not name.strip():
            raise click.BadParameter("scenario names cannot be empty")
        if name not in names:
            names.append(name)

    for scenario, variable, raw_spec in intervention_rows:
        remember(scenario)
        interventions[scenario].append(
            HardIntervention(variable=variable, value_spec=_parse_value_spec(raw_spec, "--intervention"))
        )
    for scenario, edge, raw_spec in mechanism_rows:
        remember(scenario)
        parent, child = _parse_edge(edge, "--mechanism")
        operation, value = _parse_mechanism_spec(raw_spec, "--mechanism")
        mechanisms[scenario].append(
            MechanismIntervention(parent=parent, child=child, operation=operation, value=value)
        )
    for scenario, variable, target, raw_goal in goal_rows:
        remember(scenario)
        goals[scenario].append(
            GoalSeek(variable=variable, target=target, goal_spec=_parse_value_spec(raw_goal, "--goal-seek"))
        )
    if not names:
        raise click.UsageError(
            "define at least one scenario with --intervention, --mechanism, or --goal-seek"
        )

    out: list[Scenario] = []
    for name in names:
        hard_vars = [item.variable for item in interventions[name]]
        if len(hard_vars) != len(set(hard_vars)):
            raise click.BadParameter(f"scenario {name!r} intervenes on the same variable more than once")
        goal_vars = [item.variable for item in goals[name]]
        goal_targets = [item.target for item in goals[name]]
        if len(goal_vars) != len(set(goal_vars)):
            raise click.BadParameter(f"scenario {name!r} goal-seeks the same variable more than once")
        if len(goal_targets) != len(set(goal_targets)):
            raise click.BadParameter(f"scenario {name!r} goal targets must be unique")
        overlap = sorted(set(hard_vars) & set(goal_vars))
        if overlap:
            raise click.BadParameter(
                f"scenario {name!r} has variables both fixed and goal-seeked: {overlap}"
            )
        out.append(
            Scenario(
                name=name,
                interventions=tuple(interventions[name]),
                mechanisms=tuple(mechanisms[name]),
                goals=tuple(goals[name]),
            )
        )
    return out


def _reference_subset(
    reference_data: pd.DataFrame,
    context: pd.Series | None,
    climatology_by: Sequence[str],
) -> pd.DataFrame:
    if not climatology_by:
        return reference_data
    if context is None:
        raise ValueError("climatology value specifications require an event context or empty --climatology-by")
    subset = reference_data
    for column in climatology_by:
        if column not in reference_data.columns:
            raise ValueError(f"climatology grouping column {column!r} is not present")
        if column not in context.index:
            raise ValueError(f"event context does not contain climatology column {column!r}")
        subset = subset.loc[subset[column] == context[column]]
    return subset


def _evaluate_value_spec(
    spec: ValueSpec,
    variable: str,
    *,
    context: pd.Series | None,
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    climatology_by: Sequence[str],
    base_value: float,
    quantile_delta: float,
) -> float:
    if variable not in data.columns:
        raise ValueError(f"variable {variable!r} is not present in time series")
    series = data[variable].astype(float)
    reference_series = reference_data[variable].astype(float)
    kind = spec.kind
    if kind == "fixed":
        assert spec.value is not None
        return float(spec.value)
    if kind == "mean":
        return float(reference_series.mean())
    if kind == "median":
        return float(reference_series.median())
    if kind == "quantile":
        assert spec.value is not None
        return float(reference_series.quantile(spec.value))
    if kind.startswith("climatology_"):
        subset = _reference_subset(reference_data, context, climatology_by)
        if subset.empty:
            raise ValueError(
                f"no reference observations for climatology of {variable!r} and current context"
            )
        clim = subset[variable].astype(float)
        if kind == "climatology_mean":
            return float(clim.mean())
        if kind == "climatology_median":
            return float(clim.median())
        if kind == "climatology_quantile":
            assert spec.value is not None
            return float(clim.quantile(spec.value))
    if kind == "delta":
        assert spec.value is not None
        return float(base_value + spec.value)
    if kind == "scale":
        assert spec.value is not None
        return float(base_value * spec.value)
    if kind == "qdelta":
        assert spec.value is not None
        return float(base_value + spec.value * quantile_delta)
    if kind == "zdelta":
        assert spec.value is not None
        sd = float(reference_series.std(ddof=1))
        if not np.isfinite(sd):
            raise ValueError(f"cannot calculate standard deviation for {variable!r}")
        return float(base_value + spec.value * sd)
    if kind == "fraction_to":
        assert spec.value is not None and spec.reference is not None
        reference = _evaluate_value_spec(
            spec.reference,
            variable,
            context=context,
            data=data,
            reference_data=reference_data,
            climatology_by=climatology_by,
            base_value=base_value,
            quantile_delta=quantile_delta,
        )
        return float(base_value + spec.value * (reference - base_value))
    raise ValueError(f"unsupported value specification: {kind!r}")


def _apply_mechanisms(
    adjacency: np.ndarray,
    index: Mapping[str, int],
    mechanisms: Sequence[MechanismIntervention],
    *,
    allow_new_edges: bool,
) -> np.ndarray:
    modified = np.asarray(adjacency, dtype=float).copy()
    for mechanism in mechanisms:
        if mechanism.parent not in index or mechanism.child not in index:
            raise ValueError(
                f"mechanism edge {mechanism.parent}->{mechanism.child} references a missing variable"
            )
        parent_idx = index[mechanism.parent]
        child_idx = index[mechanism.child]
        if parent_idx == child_idx:
            raise ValueError("self-loop mechanism interventions are not allowed")
        old = float(modified[child_idx, parent_idx])
        if mechanism.operation == "scale":
            new = old * mechanism.value
        elif mechanism.operation == "set":
            new = mechanism.value
        elif mechanism.operation == "add":
            new = old + mechanism.value
        else:  # pragma: no cover
            raise ValueError(f"unsupported mechanism operation: {mechanism.operation}")
        if old == 0.0 and new != 0.0 and not allow_new_edges:
            raise ValueError(
                f"mechanism would create new edge {mechanism.parent}->{mechanism.child}; "
                "pass --allow-new-edges to permit this"
            )
        modified[child_idx, parent_idx] = new
    return modified


def _solve_sem(
    adjacency: np.ndarray,
    disturbances: np.ndarray,
    do_values_centered: Mapping[int, float],
) -> np.ndarray:
    """Solve ``z = Bz + e`` after replacing equations for intervened nodes."""
    B = np.asarray(adjacency, dtype=float)
    e = np.asarray(disturbances, dtype=float)
    d = B.shape[0]
    if B.shape != (d, d) or e.shape != (d,):
        raise ValueError("invalid adjacency/disturbance dimensions")
    intervention_indices = sorted(do_values_centered)
    remaining = [idx for idx in range(d) if idx not in do_values_centered]
    z = np.zeros(d, dtype=float)
    for idx in intervention_indices:
        z[idx] = float(do_values_centered[idx])
    if remaining:
        B_rr = B[np.ix_(remaining, remaining)]
        rhs = e[remaining].copy()
        if intervention_indices:
            B_rj = B[np.ix_(remaining, intervention_indices)]
            rhs += B_rj @ z[intervention_indices]
        z[remaining] = np.linalg.solve(np.eye(len(remaining)) - B_rr, rhs)
    if not np.all(np.isfinite(z)):
        raise ValueError("SCM solution contains non-finite values")
    return z


def _hard_response_matrix(
    adjacency: np.ndarray,
    intervention_indices: Sequence[int],
) -> tuple[list[int], np.ndarray]:
    """Map changes in hard-intervened nodes to remaining nodes."""
    d = adjacency.shape[0]
    J = list(intervention_indices)
    R = [idx for idx in range(d) if idx not in J]
    if not J:
        return R, np.empty((len(R), 0), dtype=float)
    if not R:
        return R, np.empty((0, len(J)), dtype=float)
    B_rr = adjacency[np.ix_(R, R)]
    B_rj = adjacency[np.ix_(R, J)]
    response = np.linalg.solve(np.eye(len(R)) - B_rr, B_rj)
    return R, response


def _goal_values(
    *,
    adjacency: np.ndarray,
    disturbances: np.ndarray,
    fixed_do: Mapping[int, float],
    goal_variables: Sequence[int],
    goal_targets: Sequence[int],
    goal_target_values_centered: Sequence[float],
) -> dict[int, float]:
    """Jointly solve hard-intervention values required to meet target goals."""
    if not goal_variables:
        return {}
    if len(goal_variables) != len(goal_targets):
        raise ValueError("goal seeking requires equally many intervention variables and target goals")
    if len(set(goal_variables)) != len(goal_variables):
        raise ValueError("goal intervention variables must be unique")
    if set(goal_variables) & set(fixed_do):
        raise ValueError("goal intervention variables overlap fixed hard interventions")

    zero_do = dict(fixed_do)
    zero_do.update({idx: 0.0 for idx in goal_variables})
    base = _solve_sem(adjacency, disturbances, zero_do)
    K = np.zeros((len(goal_targets), len(goal_variables)), dtype=float)
    for column, variable_idx in enumerate(goal_variables):
        unit_do = dict(zero_do)
        unit_do[variable_idx] = 1.0
        unit = _solve_sem(adjacency, disturbances, unit_do)
        K[:, column] = unit[list(goal_targets)] - base[list(goal_targets)]
    rhs = np.asarray(goal_target_values_centered, dtype=float) - base[list(goal_targets)]
    try:
        solution = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("goal-seeking response matrix is singular") from exc
    if not np.all(np.isfinite(solution)):
        raise ValueError("goal-seeking solution contains non-finite values")
    return {idx: float(value) for idx, value in zip(goal_variables, solution, strict=True)}


def _run_context(
    *,
    adjacency_original: np.ndarray,
    adjacency_scenario: np.ndarray,
    factual_absolute: np.ndarray,
    means: np.ndarray,
    mode: str,
    fixed_values_absolute: Mapping[int, float],
    goal_variables: Sequence[int],
    goal_targets: Sequence[int],
    goal_values_absolute: Sequence[float],
) -> ContextResult:
    z_factual = np.asarray(factual_absolute, dtype=float) - means
    if mode == "counterfactual":
        disturbances = z_factual - adjacency_original @ z_factual
        factual = z_factual
    elif mode == "interventional_mean":
        disturbances = np.zeros_like(means, dtype=float)
        factual = np.zeros_like(means, dtype=float)
    else:  # pragma: no cover
        raise ValueError(f"unknown mode: {mode}")

    fixed_centered = {idx: float(value - means[idx]) for idx, value in fixed_values_absolute.items()}
    goals_centered = [float(value - means[idx]) for value, idx in zip(goal_values_absolute, goal_targets, strict=True)]
    solved_goals = _goal_values(
        adjacency=adjacency_scenario,
        disturbances=disturbances,
        fixed_do=fixed_centered,
        goal_variables=goal_variables,
        goal_targets=goal_targets,
        goal_target_values_centered=goals_centered,
    )
    all_do = dict(fixed_centered)
    all_do.update(solved_goals)

    mechanism_only = _solve_sem(adjacency_scenario, disturbances, {})
    counterfactual = _solve_sem(adjacency_scenario, disturbances, all_do)
    mechanism_contribution = mechanism_only - factual

    hard_contributions: dict[str, np.ndarray] = {}
    if all_do:
        intervention_indices = sorted(all_do)
        remaining, response = _hard_response_matrix(adjacency_scenario, intervention_indices)
        baseline_intervention_values = mechanism_only[intervention_indices]
        changes = np.asarray([all_do[idx] for idx in intervention_indices]) - baseline_intervention_values
        for column, idx in enumerate(intervention_indices):
            vector = np.zeros_like(means, dtype=float)
            vector[idx] = changes[column]
            if remaining:
                vector[remaining] = response[:, column] * changes[column]
            hard_contributions[str(idx)] = vector

    required_values = {str(idx): float(centered + means[idx]) for idx, centered in solved_goals.items()}
    return ContextResult(
        factual=factual + means,
        mechanism_only=mechanism_only + means,
        counterfactual=counterfactual + means,
        do_values_centered={str(idx): value for idx, value in all_do.items()},
        hard_contributions=hard_contributions,
        mechanism_contribution=mechanism_contribution,
        required_values=required_values,
    )


def _aggregate(values: Sequence[float], method: str) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "sum":
        return float(np.sum(arr))
    if method == "none":
        if len(arr) != 1:
            raise ValueError("aggregation='none' requires one value")
        return float(arr[0])
    raise ValueError(f"unknown aggregation: {method}")


def _prefix_summary(prefix: str, values: Sequence[float], ci: float) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in _summary(values, ci=ci).items()}


def _event_id(context: pd.Series | None, order_cols: Sequence[str], ordinal: int) -> str:
    if context is None:
        return "population"
    fields = {column: context[column] for column in order_cols if column in context.index}
    fields["event_ordinal"] = ordinal
    return json.dumps(fields, default=str, sort_keys=True)


def _scenario_metadata(scenario: Scenario) -> dict[str, str]:
    interventions = {
        item.variable: _value_spec_to_string(item.value_spec) for item in scenario.interventions
    }
    mechanisms = {
        f"{item.parent}->{item.child}": f"{item.operation}:{item.value}"
        for item in scenario.mechanisms
    }
    goals = {
        f"{item.variable}->{item.target}": _value_spec_to_string(item.goal_spec)
        for item in scenario.goals
    }
    return {
        "interventions_json": json.dumps(interventions, sort_keys=True),
        "mechanisms_json": json.dumps(mechanisms, sort_keys=True),
        "goals_json": json.dumps(goals, sort_keys=True),
    }


def _value_spec_to_string(spec: ValueSpec) -> str:
    if spec.kind == "fraction_to":
        assert spec.reference is not None
        return f"fraction_to:{spec.value}:{_value_spec_to_string(spec.reference)}"
    if spec.value is None:
        return spec.kind
    return f"{spec.kind}:{spec.value}"


def _required_variables(scenarios: Sequence[Scenario], targets: Sequence[str]) -> list[str]:
    names = list(targets)
    for scenario in scenarios:
        names.extend(item.variable for item in scenario.interventions)
        for item in scenario.mechanisms:
            names.extend([item.parent, item.child])
        for item in scenario.goals:
            names.extend([item.variable, item.target])
    return list(dict.fromkeys(names))


def _context_inputs(
    *,
    scenario: Scenario,
    context: pd.Series | None,
    mode: str,
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    labels: Sequence[str],
    index: Mapping[str, int],
    means: np.ndarray,
    deltas: Mapping[str, float],
    climatology_by: Sequence[str],
) -> tuple[np.ndarray, dict[int, float], list[int], list[int], list[float]]:
    if mode == "counterfactual":
        if context is None:
            raise ValueError("counterfactual mode requires factual event observations")
        factual = context[list(labels)].to_numpy(dtype=float)
    else:
        factual = means.copy()

    fixed_values: dict[int, float] = {}
    for intervention in scenario.interventions:
        variable_idx = index[intervention.variable]
        base = float(factual[variable_idx])
        value = _evaluate_value_spec(
            intervention.value_spec,
            intervention.variable,
            context=context,
            data=data,
            reference_data=reference_data,
            climatology_by=climatology_by,
            base_value=base,
            quantile_delta=float(deltas[intervention.variable]),
        )
        fixed_values[variable_idx] = value

    goal_variables: list[int] = []
    goal_targets: list[int] = []
    goal_values: list[float] = []
    for goal in scenario.goals:
        variable_idx = index[goal.variable]
        target_idx = index[goal.target]
        base_target = float(factual[target_idx])
        value = _evaluate_value_spec(
            goal.goal_spec,
            goal.target,
            context=context,
            data=data,
            reference_data=reference_data,
            climatology_by=climatology_by,
            base_value=base_target,
            quantile_delta=float(deltas[goal.target]),
        )
        goal_variables.append(variable_idx)
        goal_targets.append(target_idx)
        goal_values.append(value)
    return factual, fixed_values, goal_variables, goal_targets, goal_values


def _cut_incoming_edges(adjacency: np.ndarray, intervention_indices: Sequence[int]) -> np.ndarray:
    cut = np.asarray(adjacency, dtype=float).copy()
    for idx in intervention_indices:
        cut[idx, :] = 0.0
    return cut


def _top_intervention_paths(
    *,
    adjacency: np.ndarray,
    labels: Sequence[str],
    source: str,
    target: str,
    source_change: float,
    intervention_names: Sequence[str],
    top_n: int,
    min_abs_coefficient: float,
    max_paths: int,
) -> list[dict[str, Any]]:
    if top_n <= 0 or source == target:
        return []
    index = {name: idx for idx, name in enumerate(labels)}
    cut = _cut_incoming_edges(adjacency, [index[name] for name in intervention_names])
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    for child_idx, child in enumerate(labels):
        for parent_idx, parent in enumerate(labels):
            if child_idx == parent_idx:
                continue
            coefficient = float(cut[child_idx, parent_idx])
            if np.isfinite(coefficient) and abs(coefficient) > min_abs_coefficient:
                graph.add_edge(parent, child, weight=coefficient)
    rows: list[dict[str, Any]] = []
    try:
        for path_index, path in enumerate(nx.all_simple_paths(graph, source=source, target=target)):
            if path_index >= max_paths:
                break
            product = 1.0
            for parent, child in zip(path[:-1], path[1:], strict=True):
                product *= float(cut[index[child], index[parent]])
            contribution = product * source_change
            rows.append(
                {
                    "path": " -> ".join(path),
                    "coefficient_product": float(product),
                    "source_change": float(source_change),
                    "target_contribution": float(contribution),
                    "abs_target_contribution": float(abs(contribution)),
                }
            )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    rows.sort(key=lambda row: row["abs_target_contribution"], reverse=True)
    return rows[:top_n]


def _aggregate_component_dicts(
    component_dicts: Sequence[Mapping[str, float]],
    method: str,
) -> dict[str, float]:
    keys = sorted({key for mapping in component_dicts for key in mapping})
    return {
        key: _aggregate([mapping.get(key, np.nan) for mapping in component_dicts], method)
        for key in keys
    }


def _analyze_pixel(
    bundle: PixelBundle,
    targets: list[str],
    scenarios: list[Scenario],
    mode: str,
    event_filters: list[FilterSpec],
    reference_filters: list[FilterSpec],
    climatology_by: list[str],
    event_aggregation: str,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    allow_new_edges: bool,
    top_paths: int,
    min_path_abs_coefficient: float,
    max_paths_per_pair: int,
    order_cols: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [str(value) for value in bundle.graph_row["variable_names"]]
    index = {name: idx for idx, name in enumerate(labels)}
    required = _required_variables(scenarios, targets)
    missing = [name for name in required if name not in index]
    base_error = {**bundle.coords, "error": None}
    if missing:
        return ([{**base_error, "error": f"variables not present in graph: {missing}"}], [])

    data = bundle.time_series.dropna(subset=labels).reset_index(drop=True)
    if len(data) < min_samples:
        return ([{**base_error, "n_samples": len(data), "error": f"too few samples: {len(data)} < {min_samples}"}], [])
    try:
        event_data = _apply_filters(data, event_filters)
        reference_data = _apply_filters(data, reference_filters)
    except Exception as exc:
        return ([{**base_error, "n_samples": len(data), "error": repr(exc)}], [])
    if reference_data.empty:
        return ([{**base_error, "n_samples": len(data), "error": "reference filters selected no observations"}], [])
    if mode == "counterfactual" and event_data.empty:
        return ([{**base_error, "n_samples": len(data), "error": "event filters selected no factual observations"}], [])

    try:
        point_B = _point_matrix_from_row(bundle.graph_row, point_matrix=point_matrix)
        boot_B = _bootstrap_matrices_from_row(bundle.graph_row)
        if point_B.shape != (len(labels), len(labels)):
            raise ValueError(f"point adjacency shape {point_B.shape} does not match {len(labels)} labels")
        if boot_B.shape[1:] != (len(labels), len(labels)):
            raise ValueError(f"bootstrap adjacency shape {boot_B.shape} does not match {len(labels)} labels")
    except Exception as exc:
        return ([{**base_error, "n_samples": len(data), "error": repr(exc)}], [])

    means = data[labels].mean().to_numpy(dtype=float)
    deltas = {
        name: float(_quantile_contrast(data[name], low_quantile, high_quantile)["delta"])
        for name in required
    }
    invalid = [name for name in targets if not np.isfinite(deltas[name]) or deltas[name] == 0.0]
    if invalid:
        return ([{**base_error, "n_samples": len(data), "error": f"zero or invalid quantile range for variables: {invalid}"}], [])

    if mode == "counterfactual":
        contexts: list[pd.Series | None] = [row for _, row in event_data.iterrows()]
    elif event_filters:
        contexts = [row for _, row in event_data.iterrows()]
        if not contexts:
            return ([{**base_error, "n_samples": len(data), "error": "event filters selected no contexts"}], [])
    else:
        contexts = [None]

    main_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        try:
            point_B_scenario = _apply_mechanisms(
                point_B, index, scenario.mechanisms, allow_new_edges=allow_new_edges
            )
        except Exception as exc:
            main_rows.append(
                {
                    **base_error,
                    "scenario": scenario.name,
                    "n_samples": len(data),
                    "error": repr(exc),
                }
            )
            continue

        point_context_results: list[ContextResult] = []
        context_input_cache: list[tuple[np.ndarray, dict[int, float], list[int], list[int], list[float]]] = []
        point_context_errors: list[str] = []
        for context in contexts:
            try:
                inputs = _context_inputs(
                    scenario=scenario,
                    context=context,
                    mode=mode,
                    data=data,
                    reference_data=reference_data,
                    labels=labels,
                    index=index,
                    means=means,
                    deltas=deltas,
                    climatology_by=climatology_by,
                )
                context_input_cache.append(inputs)
                point_context_results.append(
                    _run_context(
                        adjacency_original=point_B,
                        adjacency_scenario=point_B_scenario,
                        factual_absolute=inputs[0],
                        means=means,
                        mode=mode,
                        fixed_values_absolute=inputs[1],
                        goal_variables=inputs[2],
                        goal_targets=inputs[3],
                        goal_values_absolute=inputs[4],
                    )
                )
            except Exception as exc:
                point_context_errors.append(repr(exc))
        if point_context_errors:
            main_rows.append(
                {
                    **base_error,
                    "scenario": scenario.name,
                    "n_samples": len(data),
                    "n_event_observations": len(contexts),
                    "error": point_context_errors[0],
                }
            )
            continue

        # Each bootstrap matrix is evaluated for every context, then event-aggregated.
        bootstrap_results: list[list[ContextResult]] = []
        bootstrap_failed = 0
        for B_boot in boot_B:
            try:
                B_scenario = _apply_mechanisms(
                    B_boot, index, scenario.mechanisms, allow_new_edges=allow_new_edges
                )
                boot_contexts = [
                    _run_context(
                        adjacency_original=B_boot,
                        adjacency_scenario=B_scenario,
                        factual_absolute=inputs[0],
                        means=means,
                        mode=mode,
                        fixed_values_absolute=inputs[1],
                        goal_variables=inputs[2],
                        goal_targets=inputs[3],
                        goal_values_absolute=inputs[4],
                    )
                    for inputs in context_input_cache
                ]
                bootstrap_results.append(boot_contexts)
            except Exception:
                bootstrap_failed += 1

        units: list[tuple[str, list[int], str]]
        if event_aggregation == "none":
            units = [
                (_event_id(context, order_cols, ordinal), [ordinal], "none")
                for ordinal, context in enumerate(contexts)
            ]
        else:
            units = [(event_aggregation, list(range(len(contexts))), event_aggregation)]

        metadata = _scenario_metadata(scenario)
        hard_names = [item.variable for item in scenario.interventions] + [item.variable for item in scenario.goals]

        for unit_name, positions, aggregation in units:
            for target in targets:
                target_idx = index[target]
                factual_values = [point_context_results[pos].factual[target_idx] for pos in positions]
                counterfactual_values = [point_context_results[pos].counterfactual[target_idx] for pos in positions]
                delta_values = [cf - factual for cf, factual in zip(counterfactual_values, factual_values, strict=True)]
                mechanism_values = [point_context_results[pos].mechanism_contribution[target_idx] for pos in positions]
                point_factual = _aggregate(factual_values, aggregation)
                point_counterfactual = _aggregate(counterfactual_values, aggregation)
                point_delta = _aggregate(delta_values, aggregation)
                point_mechanism = _aggregate(mechanism_values, aggregation)

                hard_component_contexts: list[dict[str, float]] = []
                required_contexts: list[dict[str, float]] = []
                for pos in positions:
                    result = point_context_results[pos]
                    hard_component_contexts.append(
                        {
                            labels[int(idx_text)]: float(vector[target_idx])
                            for idx_text, vector in result.hard_contributions.items()
                        }
                    )
                    required_contexts.append(
                        {
                            labels[int(idx_text)]: float(value)
                            for idx_text, value in result.required_values.items()
                        }
                    )
                hard_components = _aggregate_component_dicts(hard_component_contexts, aggregation)
                required_values = _aggregate_component_dicts(required_contexts, aggregation)

                boot_counterfactual: list[float] = []
                boot_delta: list[float] = []
                boot_scaled_delta: list[float] = []
                boot_mechanism: list[float] = []
                boot_hard_components: dict[str, list[float]] = defaultdict(list)
                boot_required_values: dict[str, list[float]] = defaultdict(list)
                for boot_contexts in bootstrap_results:
                    factual_b = [boot_contexts[pos].factual[target_idx] for pos in positions]
                    cf_b = [boot_contexts[pos].counterfactual[target_idx] for pos in positions]
                    delta_b_values = [cf - factual for cf, factual in zip(cf_b, factual_b, strict=True)]
                    mechanism_b_values = [boot_contexts[pos].mechanism_contribution[target_idx] for pos in positions]
                    delta_b = _aggregate(delta_b_values, aggregation)
                    boot_counterfactual.append(_aggregate(cf_b, aggregation))
                    boot_delta.append(delta_b)
                    boot_scaled_delta.append(delta_b / deltas[target])
                    boot_mechanism.append(_aggregate(mechanism_b_values, aggregation))
                    per_context_hard = []
                    per_context_required = []
                    for pos in positions:
                        result = boot_contexts[pos]
                        per_context_hard.append(
                            {
                                labels[int(idx_text)]: float(vector[target_idx])
                                for idx_text, vector in result.hard_contributions.items()
                            }
                        )
                        per_context_required.append(
                            {
                                labels[int(idx_text)]: float(value)
                                for idx_text, value in result.required_values.items()
                            }
                        )
                    for name, value in _aggregate_component_dicts(per_context_hard, aggregation).items():
                        boot_hard_components[name].append(value)
                    for name, value in _aggregate_component_dicts(per_context_required, aggregation).items():
                        boot_required_values[name].append(value)

                # Point path decomposition. The source change is relative to the
                # mechanism-only state, matching the exact hard contribution.
                aggregated_paths: dict[str, list[float]] = defaultdict(list)
                aggregated_path_source_changes: dict[str, list[float]] = defaultdict(list)
                path_metadata: dict[str, dict[str, Any]] = {}
                for pos in positions:
                    result = point_context_results[pos]
                    for source in hard_names:
                        source_idx = index[source]
                        do_centered = result.do_values_centered.get(str(source_idx))
                        if do_centered is None:
                            continue
                        source_change = float(do_centered - (result.mechanism_only[source_idx] - means[source_idx]))
                        for path in _top_intervention_paths(
                            adjacency=point_B_scenario,
                            labels=labels,
                            source=source,
                            target=target,
                            source_change=source_change,
                            intervention_names=hard_names,
                            top_n=top_paths,
                            min_abs_coefficient=min_path_abs_coefficient,
                            max_paths=max_paths_per_pair,
                        ):
                            key = f"{source}|{path['path']}"
                            aggregated_paths[key].append(float(path["target_contribution"]))
                            aggregated_path_source_changes[key].append(float(path["source_change"]))
                            path_metadata[key] = path
                top_path_rows = []
                for key, values in aggregated_paths.items():
                    row = dict(path_metadata[key])
                    row["target_contribution"] = _aggregate(values, aggregation)
                    row["source_change"] = _aggregate(aggregated_path_source_changes[key], aggregation)
                    row["abs_target_contribution"] = abs(row["target_contribution"])
                    top_path_rows.append(row)
                top_path_rows.sort(key=lambda row: row["abs_target_contribution"], reverse=True)
                top_path_rows = top_path_rows[:top_paths]

                row = {
                    **bundle.coords,
                    "scenario": scenario.name,
                    "mode": mode,
                    "target": target,
                    "event_unit": unit_name,
                    "event_aggregation": event_aggregation,
                    "n_samples": len(data),
                    "n_event_observations": len(positions),
                    "point_matrix": point_matrix,
                    "low_quantile": low_quantile,
                    "high_quantile": high_quantile,
                    "target_delta_qhi_qlo": deltas[target],
                    **metadata,
                    "factual_value": point_factual,
                    "counterfactual_value": point_counterfactual,
                    "target_change": point_delta,
                    "scaled_target_change": point_delta / deltas[target],
                    "mechanism_target_contribution": point_mechanism,
                    "hard_target_contributions_json": json.dumps(hard_components, sort_keys=True),
                    "required_intervention_values_json": json.dumps(required_values, sort_keys=True),
                    "top_paths_json": json.dumps(top_path_rows, sort_keys=True),
                    "n_bootstrap_total": int(len(boot_B)),
                    "n_bootstrap_successful": int(len(bootstrap_results)),
                    "n_bootstrap_failed": int(bootstrap_failed),
                    **_prefix_summary("counterfactual_value", boot_counterfactual, ci),
                    **_prefix_summary("target_change", boot_delta, ci),
                    **_prefix_summary("scaled_target_change", boot_scaled_delta, ci),
                    **_prefix_summary("mechanism_target_contribution", boot_mechanism, ci),
                    "error": None,
                }
                main_rows.append(row)

                for component_name, point_value in hard_components.items():
                    summary = _prefix_summary("value", boot_hard_components.get(component_name, []), ci)
                    component_rows.append(
                        {
                            **bundle.coords,
                            "scenario": scenario.name,
                            "mode": mode,
                            "target": target,
                            "event_unit": unit_name,
                            "component_type": "hard_intervention_target_contribution",
                            "component": component_name,
                            "point_value": point_value,
                            **summary,
                            "error": None,
                        }
                    )
                if scenario.mechanisms:
                    component_rows.append(
                        {
                            **bundle.coords,
                            "scenario": scenario.name,
                            "mode": mode,
                            "target": target,
                            "event_unit": unit_name,
                            "component_type": "mechanism_target_contribution",
                            "component": "all_mechanism_changes",
                            "point_value": point_mechanism,
                            **_prefix_summary("value", boot_mechanism, ci),
                            "error": None,
                        }
                    )
                for variable, point_value in required_values.items():
                    component_rows.append(
                        {
                            **bundle.coords,
                            "scenario": scenario.name,
                            "mode": mode,
                            "target": target,
                            "event_unit": unit_name,
                            "component_type": "goal_required_intervention_value",
                            "component": variable,
                            "point_value": point_value,
                            **_prefix_summary("value", boot_required_values.get(variable, []), ci),
                            "error": None,
                        }
                    )
                for path in top_path_rows:
                    component_rows.append(
                        {
                            **bundle.coords,
                            "scenario": scenario.name,
                            "mode": mode,
                            "target": target,
                            "event_unit": unit_name,
                            "component_type": "point_path_contribution",
                            "component": path["path"],
                            "point_value": path["target_contribution"],
                            "coefficient_product": path["coefficient_product"],
                            "source_change": path["source_change"],
                            "error": None,
                        }
                    )

    return main_rows, component_rows


def _analyze_pixel_task(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _analyze_pixel(*args)


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return df.copy()
    return df[df["error"].isna()].copy()


def _plot_maps(
    df: pd.DataFrame,
    row_col_cols: Sequence[str],
    plot_dir: Path,
    *,
    figure_width: float,
    figure_height: float,
    dpi: int,
    show_title: bool,
    show: bool,
) -> list[Path]:
    if len(row_col_cols) < 2:
        return []
    work = _successful_rows(df)
    if work.empty:
        return []
    if work.duplicated(subset=[*row_col_cols, "scenario", "target"]).any():
        click.echo(
            "Skipping maps because event aggregation produced multiple rows per pixel/scenario/target. "
            "Use --event-aggregation mean, median, or sum to create maps.",
            err=True,
        )
        return []
    written: list[Path] = []
    for (scenario, target), group in work.groupby(["scenario", "target"], sort=True):
        for column, title, sequential in [
            ("target_change", f"{scenario}: change in {target}", False),
            ("scaled_target_change", f"{scenario}: scaled change in {target}", False),
            ("target_change_boot_prob_gt_zero", f"{scenario}: P(change in {target} > 0)", True),
        ]:
            if column not in group.columns or group[column].notna().sum() == 0:
                continue
            grid = _grid_from_results(group, row_col_cols[0], row_col_cols[1], column)
            values = np.asarray(grid.values, dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            if sequential:
                vmin, vmax, cmap = 0.0, 1.0, "viridis"
            else:
                limit = float(np.quantile(np.abs(finite), 0.98))
                vmin, vmax, cmap = (-limit, limit, "coolwarm") if limit > 0 else (None, None, "coolwarm")
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            image = ax.imshow(
                values,
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if show_title:
                ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
            output = plot_dir / f"{_safe_filename(str(scenario))}__{_safe_filename(str(target))}__{column}.png"
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
    help="Path to the existing Confoundry experiment YAML.",
)
@click.option(
    "--target",
    "targets_raw",
    multiple=True,
    required=True,
    help="Target variable. Repeat or pass comma-separated targets.",
)
@click.option(
    "--mode",
    type=click.Choice(_MODES),
    default="counterfactual",
    show_default=True,
    help="Observation-specific counterfactual or population interventional mean.",
)
@click.option(
    "--intervention",
    type=(str, str, str),
    multiple=True,
    metavar="SCENARIO VARIABLE SPEC",
    help="Hard intervention. Repeat for multiple variables/scenarios.",
)
@click.option(
    "--mechanism",
    type=(str, str, str),
    multiple=True,
    metavar="SCENARIO EDGE SPEC",
    help="Mechanism intervention, e.g. buffered 'sm_surface->ndvi' scale:0.5.",
)
@click.option(
    "--goal-seek",
    type=(str, str, str, str),
    multiple=True,
    metavar="SCENARIO VARIABLE TARGET GOAL",
    help="Solve for intervention VARIABLE required to make TARGET reach GOAL.",
)
@click.option(
    "--event-filter",
    multiple=True,
    help="Factual/context filter, e.g. --event-filter year=2022 --event-filter month>=6.",
)
@click.option(
    "--reference-filter",
    multiple=True,
    help="Filter the reference pool used by means, quantiles and climatologies.",
)
@click.option(
    "--climatology-by",
    default="month",
    show_default=True,
    help="Comma-separated grouping columns for climatology_* value specifications; empty disables grouping.",
)
@click.option(
    "--event-aggregation",
    type=click.Choice(_AGGREGATIONS),
    default="mean",
    show_default=True,
    help="Aggregate selected event observations per pixel, or retain each with 'none'.",
)
@click.option(
    "--point-matrix",
    type=click.Choice(_POINT_MATRICES),
    default="consensus",
    show_default=True,
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
    "--allow-new-edges",
    is_flag=True,
    help="Allow mechanism set/add operations to create edges that are zero in a fitted matrix.",
)
@click.option("--top-paths", default=5, show_default=True, type=click.IntRange(0, None))
@click.option(
    "--min-path-abs-coefficient",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option("--max-paths-per-pair", default=5000, show_default=True, type=click.IntRange(1, None))
@click.option("--output-csv", default=None, type=click.Path(path_type=Path))
@click.option("--components-csv", default=None, type=click.Path(path_type=Path))
@click.option("--output-db", default=None, type=click.Path(path_type=Path))
@click.option("--output-table", default="pixel_directlingam_interventions", show_default=True)
@click.option("--components-table", default="pixel_directlingam_intervention_components", show_default=True)
@click.option("--plot-dir", default=None, type=click.Path(path_type=Path))
@click.option("--no-plots", is_flag=True)
@click.option("--plots-only", is_flag=True)
@click.option("--figure-width", default=8.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--figure-height", default=8.0, show_default=True, type=click.FloatRange(1.0, None))
@click.option("--plot-dpi", default=600, show_default=True, type=click.IntRange(72, None))
@click.option("--title/--no-title", "show_title", default=True, show_default=True)
@click.option("--show", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.option(
    "-j",
    "--jobs",
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    type=click.IntRange(1, None),
)
@click.option("--chunksize", default=1, show_default=True, type=click.IntRange(1, None))
def per_pixel_directlingam_interventions(
    config_path: Path,
    targets_raw: tuple[str, ...],
    mode: str,
    intervention: tuple[tuple[str, str, str], ...],
    mechanism: tuple[tuple[str, str, str], ...],
    goal_seek: tuple[tuple[str, str, str, str], ...],
    event_filter: tuple[str, ...],
    reference_filter: tuple[str, ...],
    climatology_by: str,
    event_aggregation: str,
    point_matrix: str,
    low_quantile: float,
    high_quantile: float,
    min_samples: int,
    ci: float,
    allow_new_edges: bool,
    top_paths: int,
    min_path_abs_coefficient: float,
    max_paths_per_pair: int,
    output_csv: Path | None,
    components_csv: Path | None,
    output_db: Path | None,
    output_table: str,
    components_table: str,
    plot_dir: Path | None,
    no_plots: bool,
    plots_only: bool,
    figure_width: float,
    figure_height: float,
    plot_dpi: int,
    show_title: bool,
    show: bool,
    no_progress: bool,
    jobs: int,
    chunksize: int,
) -> None:
    """Run general hard, mechanism, and goal-seeking SCM scenarios per pixel."""
    del chunksize  # CLI compatibility with the other per-pixel scripts.
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise click.BadParameter("require 0 <= low_quantile < high_quantile <= 1")
    if plots_only and no_plots:
        raise click.UsageError("--plots-only cannot be combined with --no-plots")

    targets = _flatten_targets(targets_raw)
    scenarios = _build_scenarios(intervention, mechanism, goal_seek)
    event_filters = [_parse_filter(raw, "--event-filter") for raw in event_filter]
    reference_filters = [_parse_filter(raw, "--reference-filter") for raw in reference_filter]
    climatology_columns = _parse_csv(climatology_by, "--climatology-by", required=False)

    cfg = load_config(
        config_path=config_path,
        target_override=targets[0],
        point_matrix_override=point_matrix,
        plot_dir_override=plot_dir,
    )
    base_dir = cfg.experiment_dir
    location = cfg.location_name
    output_csv_path = _resolve_path(
        base_dir, output_csv, f"{location}_directlingam_interventions.csv"
    )
    components_csv_path = _resolve_path(
        base_dir, components_csv, f"{location}_directlingam_intervention_components.csv"
    )
    output_db_path = _resolve_path(
        base_dir, output_db, f"{location}_directlingam_interventions.duckdb"
    )
    plot_dir_path = _resolve_path(
        base_dir, plot_dir, f"{location}_directlingam_intervention_plots"
    )

    if plots_only:
        if not output_csv_path.exists():
            raise click.ClickException(f"output CSV does not exist: {output_csv_path}")
        results_df = pd.read_csv(output_csv_path)
        components_df = (
            pd.read_csv(components_csv_path) if components_csv_path.exists() else pd.DataFrame()
        )
    else:
        if not no_progress:
            click.echo("Loading shifted time series and graph tables...")
        timeseries_df, graph_df, _ = load_shifted_timeseries_and_graphs(cfg)
        bundles = list(
            progress_bar(
                iter_pixel_groups(cfg, timeseries_df=timeseries_df, graph_df=graph_df),
                total=len(graph_df),
                desc="Preparing intervention tasks",
                unit="pixel",
                disabled=no_progress or len(graph_df) == 0,
            )
        )
        tasks = [
            (
                bundle,
                targets,
                scenarios,
                mode,
                event_filters,
                reference_filters,
                climatology_columns,
                event_aggregation,
                point_matrix,
                low_quantile,
                high_quantile,
                min_samples,
                ci,
                allow_new_edges,
                top_paths,
                min_path_abs_coefficient,
                max_paths_per_pair,
                cfg.order_cols,
            )
            for bundle in bundles
        ]
        outputs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        if jobs == 1:
            for task in progress_bar(
                tasks,
                total=len(tasks),
                desc="Evaluating interventions",
                unit="pixel",
                disabled=no_progress or len(tasks) == 0,
            ):
                outputs.append(_analyze_pixel_task(task))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_analyze_pixel_task, task) for task in tasks]
                for future in progress_bar(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Evaluating interventions using {jobs} workers",
                    unit="pixel",
                    disabled=no_progress or len(futures) == 0,
                ):
                    outputs.append(future.result())
        main_rows = [row for main, _ in outputs for row in main]
        component_rows = [row for _, components in outputs for row in components]
        if not main_rows:
            raise click.ClickException("no intervention rows were produced")
        results_df = pd.DataFrame(main_rows)
        components_df = pd.DataFrame(component_rows)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        components_csv_path.parent.mkdir(parents=True, exist_ok=True)
        components_df.to_csv(components_csv_path, index=False)
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(output_db_path))
        try:
            write_dataframe_table(con, results_df, output_table)
            write_dataframe_table(con, components_df, components_table)
        finally:
            con.close()

    written_plots: list[Path] = []
    if not no_plots:
        written_plots = _plot_maps(
            results_df,
            cfg.row_col_cols,
            plot_dir_path,
            figure_width=figure_width,
            figure_height=figure_height,
            dpi=plot_dpi,
            show_title=show_title,
            show=show,
        )

    successful = _successful_rows(results_df)
    failed = len(results_df) - len(successful)
    click.echo(f"Mode: {mode}")
    click.echo(f"Targets: {', '.join(targets)}")
    click.echo(f"Scenarios: {', '.join(scenario.name for scenario in scenarios)}")
    click.echo(f"Point matrix: {point_matrix}")
    click.echo(f"Results CSV: {output_csv_path}")
    click.echo(f"Components CSV: {components_csv_path}")
    if not plots_only:
        click.echo(f"Output DuckDB: {output_db_path}::{output_table}, {components_table}")
    click.echo(f"Failed rows: {failed} / {len(results_df)}")
    for path in written_plots:
        click.echo(f"Plot: {path}")


if __name__ == "__main__":
    per_pixel_directlingam_interventions()
