import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from click.testing import CliRunner
from statsmodels.tsa.api import VAR

import confoundry.per_pixel_graph_diagnostics as diagnostics_module

from confoundry.per_pixel_graph_diagnostics import (
    adjusted_portmanteau_statistic,
    batch_adjusted_portmanteau_statistics,
    default_diagnostics_path,
    default_varlingam_graph_path,
    fit_reduced_form_var,
    graph_config_value,
    lagged_bootstrap_probability_diagnostics,
    multivariate_whiteness_diagnostics,
    residual_crosslag_diagnostics,
    residual_bootstrap_whiteness_diagnostics,
    reduced_form_var_innovations,
    resolve_path,
    structural_residuals_for_graph,
    var_stability_diagnostics,
)


def test_residualized_config_paths_are_used_for_diagnostics(tmp_path):
    config = {
        "name": "demo",
        "timeseries_db": "demo_ard.duckdb",
        "timeseries_table": "demo_residualized",
        "graph_db": "demo_residualized_graphs.duckdb",
        "graph_discovery": {
            "input_db": "demo_ard.duckdb",
            "input_table": "demo_residualized",
            "output_db": "demo_residualized_graphs.duckdb",
        },
    }

    input_db = resolve_path(
        tmp_path,
        graph_config_value(config, "input_db")
        or graph_config_value(config, "timeseries_db"),
        tmp_path / "demo_ard.duckdb",
    )
    graph_db = resolve_path(
        tmp_path,
        graph_config_value(config, "output_db")
        or graph_config_value(config, "graph_db"),
        tmp_path / "demo_graphs.duckdb",
    )

    assert input_db == tmp_path / "demo_ard.duckdb"
    assert graph_config_value(config, "input_table") == "demo_residualized"
    assert graph_db == tmp_path / "demo_residualized_graphs.duckdb"
    assert default_diagnostics_path(graph_db) == (
        tmp_path / "demo_residualized_graph_diagnostics.duckdb"
    )


def test_default_diagnostics_path_handles_custom_graph_database_name():
    graph_db = Path("/data/custom.duckdb")

    assert default_diagnostics_path(graph_db) == Path(
        "/data/custom_diagnostics.duckdb"
    )


def test_default_varlingam_graph_path_does_not_replace_direct_graphs():
    graph_db = Path("/data/demo_residualized_graphs.duckdb")

    assert default_varlingam_graph_path(graph_db) == Path(
        "/data/demo_residualized_varlingam_graphs.duckdb"
    )


def test_varlingam_diagnostics_use_contemporaneous_and_lagged_effects():
    X = np.asarray(
        [
            [1.0, 8.0],
            [2.0, 4.0],
            [4.0, 2.0],
            [8.0, 1.0],
        ]
    )
    contemporaneous = np.asarray(
        [
            [0.0, 0.25],
            [0.5, 0.0],
        ]
    )
    lagged = np.asarray(
        [
            [
                [0.5, 0.0],
                [0.0, 0.25],
            ]
        ]
    )

    residuals, time_offset = structural_residuals_for_graph(
        X,
        contemporaneous,
        {
            "model_type": "varlingam",
            "var_lags": 1,
            "adjacency_lagged_raw_json": json.dumps(lagged.tolist()),
        },
    )

    expected = X[1:] - X[1:] @ contemporaneous.T - X[:-1] @ lagged[0].T
    np.testing.assert_allclose(residuals, expected)
    assert time_offset == 1


def test_reduced_form_var_innovations_match_varlingam_initial_var_fit():
    rng = np.random.default_rng(104)
    X = rng.normal(size=(180, 3))

    expected = VAR(X).fit(maxlags=2, trend="n").resid
    actual = reduced_form_var_innovations(X, n_lags=2)

    np.testing.assert_allclose(actual, expected)


def test_reduced_form_var_fit_returns_statsmodels_coefficients():
    rng = np.random.default_rng(204)
    X = rng.normal(size=(180, 3))

    expected = VAR(X).fit(maxlags=2, trend="n")
    coefficients, innovations = fit_reduced_form_var(X, n_lags=2)

    np.testing.assert_allclose(coefficients, expected.coefs)
    np.testing.assert_allclose(innovations, expected.resid)


def _monthly_metadata(n_samples):
    month_index = np.arange(n_samples)
    return pd.DataFrame(
        {
            "row": np.zeros(n_samples, dtype=int),
            "col": np.zeros(n_samples, dtype=int),
            "year": 2000 + month_index // 12,
            "month": month_index % 12 + 1,
        }
    )


def test_crosslag_diagnostics_find_directional_lagged_innovation_pair():
    rng = np.random.default_rng(42)
    n_samples = 400
    source = rng.normal(size=n_samples)
    target = rng.normal(scale=0.05, size=n_samples)
    target[1:] += source[:-1]
    innovations = np.column_stack([source, target])

    result = residual_crosslag_diagnostics(
        innovations,
        _monthly_metadata(n_samples),
        labels=["source", "target"],
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        max_lag=3,
        threshold=0.2,
        top_n=5,
    )

    top_pair = json.loads(
        result["residual_crosslag_top_pairs_json"]
    )[0]
    by_lag = json.loads(result["residual_crosslag_by_lag_json"])
    assert result["residual_crosslag_max_abs_corr"] > 0.99
    assert top_pair["source"] == "source"
    assert top_pair["target"] == "target"
    assert top_pair["lag"] == 1
    assert len(by_lag) == 3
    assert by_lag[0]["max_abs_crossvariable_correlation"] > 0.99
    assert by_lag[0]["n_pairs"] == 4


def test_multivariate_portmanteau_rejects_temporally_dependent_innovations():
    rng = np.random.default_rng(7)
    innovations = np.zeros((500, 2), dtype=float)
    noise = rng.normal(size=(500, 2))
    for time in range(1, len(innovations)):
        innovations[time] = 0.85 * innovations[time - 1] + noise[time]

    result = multivariate_whiteness_diagnostics(
        innovations,
        _monthly_metadata(len(innovations)),
        labels=["a", "b"],
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        max_lag=12,
        model_lags=1,
        alpha=0.05,
    )

    assert result["residual_whiteness_p"] < 1e-10
    assert result["residual_whiteness_rejected"]
    assert result["residual_whiteness_lags_evaluated"] == 12
    by_lag = json.loads(result["residual_whiteness_by_lag_json"])
    assert len(by_lag) == 12
    assert np.isclose(
        sum(record["median_statistic_contribution"] for record in by_lag),
        result["residual_whiteness_stat"],
    )
    assert np.isclose(
        sum(record["median_fraction_total_statistic"] for record in by_lag),
        1.0,
    )


def test_multivariate_portmanteau_matches_statsmodels_adjusted_test():
    rng = np.random.default_rng(73)
    observations = rng.normal(size=(300, 3))
    fitted = VAR(observations).fit(maxlags=1, trend="n")
    expected = fitted.test_whiteness(nlags=12, adjusted=True)

    result = multivariate_whiteness_diagnostics(
        fitted.resid,
        _monthly_metadata(len(fitted.resid)),
        labels=["a", "b", "c"],
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        max_lag=12,
        model_lags=1,
        alpha=0.05,
    )

    assert np.isclose(
        result["residual_whiteness_stat"],
        expected.test_statistic,
    )
    assert np.isclose(
        result["residual_whiteness_p"],
        expected.pvalue,
    )
    statistic = adjusted_portmanteau_statistic(fitted.resid, 12)
    batched = batch_adjusted_portmanteau_statistics(
        fitted.resid[np.newaxis],
        12,
    )
    assert np.isclose(statistic, expected.test_statistic)
    assert np.isclose(batched[0], expected.test_statistic)


def test_residual_bootstrap_whiteness_is_reproducible_and_detects_missing_lag():
    rng = np.random.default_rng(411)
    observations = np.zeros((320, 2), dtype=float)
    innovations = rng.standard_t(df=5, size=observations.shape)
    lag1 = np.asarray([[0.2, 0.05], [0.0, 0.15]])
    lag2 = np.asarray([[0.6, 0.0], [0.0, 0.55]])
    for index in range(2, len(observations)):
        observations[index] = (
            lag1 @ observations[index - 1]
            + lag2 @ observations[index - 2]
            + innovations[index]
        )

    first = residual_bootstrap_whiteness_diagnostics(
        observations,
        n_lags=1,
        max_lag=12,
        alpha=0.05,
        n_bootstrap=39,
        burnin=30,
        seed=17,
        batch_size=16,
    )
    second = residual_bootstrap_whiteness_diagnostics(
        observations,
        n_lags=1,
        max_lag=12,
        alpha=0.05,
        n_bootstrap=39,
        burnin=30,
        seed=17,
        batch_size=16,
    )

    assert first == second
    assert first["residual_whiteness_bootstrap_status"] == "ok"
    assert first["residual_whiteness_bootstrap_samples_valid"] == 39
    assert first["residual_whiteness_bootstrap_p"] <= 0.05
    assert first["residual_whiteness_bootstrap_rejected"]


def test_lagged_bootstrap_diagnostics_keep_autoregressive_diagonal():
    raw = np.asarray([[[0.7, 0.0], [0.1, 0.6]]])
    probabilities = np.asarray([[[0.95, 0.1], [0.8, 0.9]]])
    consensus = np.asarray([[[0.7, 0.0], [0.1, 0.6]]])

    result = lagged_bootstrap_probability_diagnostics(
        probabilities,
        raw,
        consensus,
        labels=["a", "b"],
        min_prob=0.7,
        min_abs_effect=0.01,
        probability_band=0.1,
        top_n=4,
    )

    top_edges = json.loads(result["lagged_bootstrap_top_edges_json"])
    assert result["lagged_consensus_edge_count"] == 3
    assert result["lagged_bootstrap_edges_ge_min_prob"] == 3
    assert top_edges[0]["parent"] == "a"
    assert top_edges[0]["child"] == "a"
    assert top_edges[0]["autoregressive"]


def test_var_stability_reports_point_and_paired_bootstrap_fraction():
    contemporaneous = np.zeros((2, 2), dtype=float)
    lagged = np.asarray([np.eye(2) * 0.5])
    bootstrap_contemporaneous = np.zeros((2, 2, 2), dtype=float)
    bootstrap_lagged = np.asarray(
        [
            [np.eye(2) * 0.5],
            [np.eye(2) * 1.1],
        ]
    )

    result = var_stability_diagnostics(
        contemporaneous,
        lagged,
        bootstrap_contemporaneous,
        bootstrap_lagged,
        stability_threshold=1.0,
        bootstrap_limit=0,
    )

    assert np.isclose(result["var_stability_radius"], 0.5)
    assert result["var_stable"]
    assert result["var_bootstrap_stability_n_valid"] == 2
    assert np.isclose(result["var_bootstrap_stable_fraction"], 0.5)


def test_var_diagnostics_cli_writes_complete_var_metrics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        diagnostics_module,
        "process_map",
        lambda function, tasks, **kwargs: [
            function(task) for task in tasks
        ],
    )
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "name: demo\n"
        "columns:\n"
        "  - name: a\n"
        "    shift: 0\n"
        "  - name: b\n"
        "    shift: -1\n"
        "graph_discovery:\n"
        "  model: varlingam\n"
        "  output_db: demo_graphs.duckdb\n",
        encoding="utf-8",
    )

    n_samples = 60
    rng = np.random.default_rng(11)
    month_index = np.arange(n_samples)
    time_series = pd.DataFrame(
        {
            "row": np.zeros(n_samples, dtype=int),
            "col": np.zeros(n_samples, dtype=int),
            "year": 2000 + month_index // 12,
            "month": month_index % 12 + 1,
            "a": rng.normal(size=n_samples),
            "b": rng.normal(size=n_samples),
        }
    )
    input_db = tmp_path / "demo_ard.duckdb"
    con = duckdb.connect(str(input_db))
    try:
        con.register("time_series", time_series)
        con.execute("CREATE TABLE demo AS SELECT * FROM time_series")
    finally:
        con.close()

    b0 = np.zeros((2, 2), dtype=float)
    b1 = np.asarray([[[0.2, 0.0], [0.0, 0.3]]])
    probability_b0 = np.zeros((2, 2), dtype=float)
    probability_b1 = np.asarray([[[0.9, 0.1], [0.2, 0.85]]])
    consensus_b1 = np.asarray([[[0.2, 0.0], [0.0, 0.3]]])
    bootstrap_b0 = np.repeat(b0[np.newaxis], 2, axis=0)
    bootstrap_b1 = np.repeat(b1[np.newaxis], 2, axis=0)
    graph = pd.DataFrame(
        [
            {
                "row": 0,
                "col": 0,
                "model_type": "varlingam",
                "n_samples": n_samples,
                "n_effective_samples": n_samples - 1,
                "var_lags": 1,
                "variable_names_json": json.dumps(["a", "b"]),
                "adjacency_raw_json": json.dumps(b0.tolist()),
                "edge_probability_json": json.dumps(
                    probability_b0.tolist()
                ),
                "adjacency_consensus_json": json.dumps(b0.tolist()),
                "adjacency_lagged_raw_json": json.dumps(b1.tolist()),
                "edge_probability_lagged_json": json.dumps(
                    probability_b1.tolist()
                ),
                "adjacency_lagged_consensus_json": json.dumps(
                    consensus_b1.tolist()
                ),
                "adjacency_bootstrap_json": json.dumps(
                    bootstrap_b0.tolist()
                ),
                "adjacency_bootstrap_lagged_json": json.dumps(
                    bootstrap_b1.tolist()
                ),
            }
        ]
    )
    graph_db = tmp_path / "demo_varlingam_graphs.duckdb"
    con = duckdb.connect(str(graph_db))
    try:
        con.register("graph", graph)
        con.execute("CREATE TABLE pixel_graphs AS SELECT * FROM graph")
    finally:
        con.close()

    default_result = CliRunner().invoke(
        diagnostics_module.graph_statistics,
        [
            "--config-path",
            str(config_path),
            "--whiteness-lags",
            "3",
            "--workers",
            "1",
        ],
    )
    assert default_result.exit_code == 0, default_result.output
    default_output_db = (
        tmp_path / "demo_varlingam_graph_diagnostics.duckdb"
    )
    con = duckdb.connect(str(default_output_db), read_only=True)
    try:
        default_columns = set(
            con.execute(
                "DESCRIBE pixel_graph_diagnostics"
            ).fetchdf()["column_name"]
        )
        default_metadata_columns = set(
            con.execute(
                "DESCRIBE graph_statistics_run_metadata"
            ).fetchdf()["column_name"]
        )
    finally:
        con.close()
    assert "residual_whiteness_bootstrap_p" not in default_columns
    assert (
        "whiteness_bootstrap_samples"
        not in default_metadata_columns
    )

    calibrated_output_db = tmp_path / "demo_calibrated.duckdb"
    result = CliRunner().invoke(
        diagnostics_module.graph_statistics,
        [
            "--config-path",
            str(config_path),
            "--diagnostics-db-path",
            str(calibrated_output_db),
            "--whiteness-lags",
            "3",
            "--whiteness-bootstrap-samples",
            "9",
            "--whiteness-bootstrap-burnin",
            "10",
            "--whiteness-bootstrap-pixel-limit",
            "1",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    con = duckdb.connect(str(calibrated_output_db), read_only=True)
    try:
        row = con.execute(
            "SELECT * FROM pixel_graph_diagnostics"
        ).fetchdf().iloc[0]
        metadata = con.execute(
            "SELECT * FROM graph_statistics_run_metadata"
        ).fetchdf().iloc[0]
    finally:
        con.close()

    assert row["model_type"] == "varlingam"
    assert row["residual_crosslag_lags_evaluated"] == 3
    assert row["residual_whiteness_lags_evaluated"] == 3
    assert row["residual_temporal_basis"] == (
        "refitted_reduced_form_var_innovations"
    )
    fitted_var = VAR(time_series[["a", "b"]].to_numpy()).fit(
        maxlags=1,
        trend="n",
    )
    expected_whiteness = fitted_var.test_whiteness(
        nlags=3,
        adjusted=True,
    )
    assert np.isclose(
        row["residual_whiteness_stat"],
        expected_whiteness.test_statistic,
    )
    assert np.isclose(
        row["residual_whiteness_p"],
        expected_whiteness.pvalue,
    )
    assert row["residual_whiteness_bootstrap_samples_requested"] == 9
    assert row["residual_whiteness_bootstrap_samples_valid"] == 9
    assert row["residual_whiteness_bootstrap_status"] == "ok"
    assert 0.0 < row["residual_whiteness_bootstrap_p"] <= 1.0
    assert np.isclose(row["var_stability_radius"], 0.3)
    assert bool(row["var_stable"])
    assert row["lagged_consensus_edge_count"] == 2
    assert row["var_bootstrap_stable_fraction"] == 1.0
    assert metadata["whiteness_lags"] == 3
    assert metadata["whiteness_bootstrap_samples"] == 9
    assert metadata["whiteness_bootstrap_burnin"] == 10
    assert metadata["whiteness_bootstrap_pixel_limit"] == 1
    assert metadata["whiteness_bootstrap_pixels_selected"] == 1
    assert not bool(metadata["configured_shifts_applied"])
