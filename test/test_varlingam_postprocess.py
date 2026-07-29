import json

import duckdb
import numpy as np
import pandas as pd
from click.testing import CliRunner

from confoundry.per_pixel_varlingam_analysis import (
    analyze_var_pixel,
    per_pixel_varlingam_analysis,
)
from confoundry.per_pixel_varlingam_interventions import (
    Intervention,
    InterventionValue,
    Scenario,
    analyze_intervention_pixel,
    build_scenarios,
    evaluate_intervention_value,
    per_pixel_varlingam_interventions,
)
from confoundry.varlingam_postprocess import (
    VARPixelBundle,
    dynamic_effect_matrices,
    infer_structural_innovations,
    matrices_from_graph_row,
    simulate_structural_var,
)


def _known_matrices():
    contemporaneous = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ]
    )
    lagged = np.asarray(
        [
            [
                [0.2, 0.0],
                [0.0, 0.3],
            ]
        ]
    )
    return contemporaneous, lagged


def _known_bundle(n_bootstrap=3):
    contemporaneous, lagged = _known_matrices()
    graph_row = {
        "variable_names_json": json.dumps(["source", "target"]),
        "adjacency_raw_json": json.dumps(contemporaneous.tolist()),
        "adjacency_consensus_json": json.dumps(
            contemporaneous.tolist()
        ),
        "adjacency_lagged_raw_json": json.dumps(lagged.tolist()),
        "adjacency_lagged_consensus_json": json.dumps(
            lagged.tolist()
        ),
        "adjacency_bootstrap_json": json.dumps(
            np.repeat(
                contemporaneous[np.newaxis, :, :],
                n_bootstrap,
                axis=0,
            ).tolist()
        ),
        "adjacency_bootstrap_lagged_json": json.dumps(
            np.repeat(
                lagged[np.newaxis, :, :, :],
                n_bootstrap,
                axis=0,
            ).tolist()
        ),
    }
    time_series = pd.DataFrame(
        {
            "row": [4] * 8,
            "col": [9] * 8,
            "year": [2020] * 8,
            "month": list(range(1, 9)),
            "source": np.linspace(-1.0, 1.0, 8),
            "target": np.linspace(-2.0, 2.0, 8),
        }
    )
    return VARPixelBundle(
        key=(4, 9),
        coords={"row": 4, "col": 9},
        time_series=time_series,
        graph_row=graph_row,
    )


def test_dynamic_effects_propagate_contemporaneous_and_lagged_paths():
    contemporaneous, lagged = _known_matrices()

    effects, cumulative, radius = dynamic_effect_matrices(
        contemporaneous,
        lagged,
        horizon=2,
    )

    expected_zero = np.asarray([[1.0, 0.0], [0.5, 1.0]])
    expected_one = np.asarray([[0.2, 0.0], [0.25, 0.3]])
    np.testing.assert_allclose(effects[0], expected_zero)
    np.testing.assert_allclose(effects[1], expected_one)
    np.testing.assert_allclose(
        effects[2],
        np.asarray([[0.04, 0.0], [0.095, 0.09]]),
    )
    np.testing.assert_allclose(cumulative, np.cumsum(effects, axis=0))
    assert np.isclose(radius, 0.3)


def test_saved_var_matrices_remain_paired_when_bootstrap_is_limited():
    matrices = matrices_from_graph_row(
        _known_bundle(n_bootstrap=4).graph_row,
        point_matrix="bootstrap_mean",
        bootstrap_limit=2,
    )

    assert matrices.labels == ("source", "target")
    assert matrices.bootstrap_contemporaneous.shape == (2, 2, 2)
    assert matrices.bootstrap_lagged.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(
        matrices.contemporaneous,
        matrices.bootstrap_contemporaneous[0],
    )
    np.testing.assert_allclose(
        matrices.lagged,
        matrices.bootstrap_lagged[0],
    )


def test_structural_innovations_reconstruct_the_factual_path():
    contemporaneous, lagged = _known_matrices()
    observations = np.asarray(
        [
            [1.0, 0.5],
            [0.4, 1.1],
            [-0.2, 0.3],
            [0.8, -0.1],
        ]
    )
    innovations = infer_structural_innovations(
        observations,
        contemporaneous,
        lagged,
    )

    reconstructed = simulate_structural_var(
        contemporaneous,
        lagged,
        observations[:1],
        innovations[1:],
    )

    np.testing.assert_allclose(reconstructed, observations[1:])


def test_hard_intervention_replaces_only_the_selected_equation():
    contemporaneous, lagged = _known_matrices()
    path = simulate_structural_var(
        contemporaneous,
        lagged,
        initial_history=np.zeros((1, 2)),
        innovations=np.zeros((2, 2)),
        interventions=[{0: 1.0}, {}],
    )

    np.testing.assert_allclose(
        path,
        np.asarray(
            [
                [1.0, 0.5],
                [0.2, 0.25],
            ]
        ),
    )


def test_var_effect_analysis_reports_lag_slice_and_dynamic_effects():
    rows = analyze_var_pixel(
        _known_bundle(),
        target="target",
        sources=["source"],
        horizon=1,
        point_matrix="raw",
        low_quantile=0.1,
        high_quantile=0.9,
        min_samples=2,
        ci=0.95,
        stability_threshold=1.0,
        bootstrap_limit=0,
    )
    by_horizon = {row["horizon"]: row for row in rows}

    assert by_horizon[0]["direct_effect"] == 0.5
    assert by_horizon[0]["total_effect"] == 0.5
    assert by_horizon[1]["direct_effect"] == 0.0
    assert np.isclose(by_horizon[1]["lag_slice_total_effect"], 0.1)
    assert np.isclose(by_horizon[1]["total_effect"], 0.25)
    assert by_horizon[1]["n_bootstrap_stable"] == 3
    assert by_horizon[1]["total_effect_boot_ci_excludes_zero"]


def test_var_intervention_analysis_propagates_a_one_month_do_operation():
    scenario = Scenario(
        name="pulse",
        interventions=(
            Intervention(
                variable="source",
                value=InterventionValue(kind="delta", value=1.0),
            ),
        ),
    )
    rows = analyze_intervention_pixel(
        _known_bundle(),
        targets=["target"],
        scenarios=[scenario],
        mode="interventional_mean",
        horizon=1,
        duration=1,
        start_year=None,
        start_month=None,
        point_matrix="raw",
        low_quantile=0.1,
        high_quantile=0.9,
        min_samples=2,
        ci=0.95,
        stability_threshold=1.0,
        bootstrap_limit=0,
    )
    by_horizon = {row["horizon"]: row for row in rows}

    assert np.isclose(by_horizon[0]["effect"], 0.5)
    assert np.isclose(by_horizon[1]["effect"], 0.25)
    assert by_horizon[0]["active_intervention"]
    assert not by_horizon[1]["active_intervention"]
    assert by_horizon[1]["n_bootstrap_simulation_successful"] == 3


def test_var_counterfactual_reuses_factual_innovations():
    scenario = Scenario(
        name="event_pulse",
        interventions=(
            Intervention(
                variable="source",
                value=InterventionValue(kind="delta", value=1.0),
            ),
        ),
    )
    rows = analyze_intervention_pixel(
        _known_bundle(),
        targets=["target"],
        scenarios=[scenario],
        mode="counterfactual",
        horizon=1,
        duration=1,
        start_year=2020,
        start_month=3,
        point_matrix="raw",
        low_quantile=0.1,
        high_quantile=0.9,
        min_samples=2,
        ci=0.95,
        stability_threshold=1.0,
        bootstrap_limit=0,
    )
    by_horizon = {row["horizon"]: row for row in rows}

    assert np.isclose(by_horizon[0]["effect"], 0.5)
    assert np.isclose(by_horizon[1]["effect"], 0.25)


def test_intervention_parsing_supports_joint_scenarios_and_scaled_values():
    scenarios = build_scenarios(
        [
            ("joint", "source", "qdelta:0.5"),
            ("joint", "target", "quantile:0.75"),
        ]
    )

    assert len(scenarios) == 1
    assert len(scenarios[0].interventions) == 2
    assert np.isclose(
        evaluate_intervention_value(
            InterventionValue("qdelta", 0.5),
            reference_values=np.asarray([0.0, 1.0, 2.0, 3.0]),
            factual_value=10.0,
            low_quantile=0.25,
            high_quantile=0.75,
        ),
        10.75,
    )


def _write_postprocess_inputs(tmp_path):
    bundle = _known_bundle()
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "name: demo\n"
        "reference_var: target\n"
        "columns:\n"
        "  - name: source\n"
        "    shift: -1\n"
        "  - name: target\n"
        "    shift: 0\n",
        encoding="utf-8",
    )
    input_db = tmp_path / "demo_ard.duckdb"
    con = duckdb.connect(str(input_db))
    try:
        con.register("time_series", bundle.time_series)
        con.execute("CREATE TABLE demo AS SELECT * FROM time_series")
    finally:
        con.close()

    graph_db = tmp_path / "demo_varlingam_graphs.duckdb"
    graph_df = pd.DataFrame(
        [
            {
                "row": 4,
                "col": 9,
                "model_type": "varlingam",
                **bundle.graph_row,
            }
        ]
    )
    con = duckdb.connect(str(graph_db))
    try:
        con.register("graphs", graph_df)
        con.execute("CREATE TABLE pixel_graphs AS SELECT * FROM graphs")
    finally:
        con.close()
    return config_path, graph_db


def test_var_postprocess_cli_commands_write_tables_and_ignore_yaml_shifts(
    tmp_path,
):
    config_path, graph_db = _write_postprocess_inputs(tmp_path)

    effect_result = CliRunner().invoke(
        per_pixel_varlingam_analysis,
        [
            "--config-path",
            str(config_path),
            "--graphs-db",
            str(graph_db),
            "--horizon",
            "1",
            "--min-samples",
            "2",
            "--jobs",
            "1",
            "--variable-label",
            "source=Managed source anomaly",
        ],
    )
    assert effect_result.exit_code == 0, effect_result.output
    effect_db = tmp_path / "demo_varlingam_effects.duckdb"
    con = duckdb.connect(str(effect_db), read_only=True)
    try:
        effect = con.execute(
            "SELECT * FROM pixel_varlingam_effects ORDER BY horizon"
        ).fetchdf()
        summary = con.execute(
            "SELECT * FROM varlingam_effect_summary ORDER BY horizon"
        ).fetchdf()
    finally:
        con.close()
    assert effect["n_samples"].unique().tolist() == [8]
    assert effect["horizon"].tolist() == [0, 1]
    assert summary["horizon"].tolist() == [0, 1]
    assert (
        tmp_path
        / "demo_varlingam_effect_plots"
        / "varlingam_effect_source_to_target.png"
    ).exists()
    assert (
        tmp_path
        / "demo_varlingam_effect_plots"
        / "varlingam_effect_source_to_target.pdf"
    ).exists()

    intervention_result = CliRunner().invoke(
        per_pixel_varlingam_interventions,
        [
            "--config-path",
            str(config_path),
            "--graphs-db",
            str(graph_db),
            "--target",
            "target",
            "--intervention",
            "pulse",
            "source",
            "delta:1",
            "--horizon",
            "1",
            "--min-samples",
            "2",
            "--jobs",
            "1",
            "--variable-label",
            "target=Outcome anomaly",
        ],
    )
    assert intervention_result.exit_code == 0, intervention_result.output
    intervention_db = (
        tmp_path / "demo_varlingam_interventions.duckdb"
    )
    con = duckdb.connect(str(intervention_db), read_only=True)
    try:
        intervention = con.execute(
            "SELECT * FROM pixel_varlingam_interventions ORDER BY horizon"
        ).fetchdf()
        summary = con.execute(
            "SELECT * FROM varlingam_intervention_summary ORDER BY horizon"
        ).fetchdf()
    finally:
        con.close()
    np.testing.assert_allclose(intervention["effect"], [0.5, 0.25])
    assert summary["horizon"].tolist() == [0, 1]
    assert (
        tmp_path
        / "demo_varlingam_intervention_plots"
        / "varlingam_intervention_pulse_target.png"
    ).exists()
