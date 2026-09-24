import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from click.testing import CliRunner

import confoundry.validate_varlingam_synthetic as validation
from confoundry.residualize_timeseries import residualize_dataframe


def test_independent_empirical_draws_break_cross_variable_pairing():
    pool = np.column_stack([np.arange(100, dtype=float)] * 2)
    draws = validation.draw_innovations(
        pool,
        20_000,
        np.random.default_rng(4),
        "empirical-independent",
    )

    assert abs(np.corrcoef(draws, rowvar=False)[0, 1]) < 0.03
    np.testing.assert_allclose(np.sort(np.unique(draws[:, 0])), np.arange(100) - 49.5)


def test_structural_var_simulator_applies_contemporaneous_and_lagged_paths():
    b0 = np.asarray([[0.0, 0.0], [0.5, 0.0]])
    lagged = np.asarray([[[0.2, 0.0], [0.0, 0.3]]])
    pool = np.asarray([[1.0, -1.0], [-1.0, 1.0]])

    simulated = validation.simulate_structural_var(
        b0,
        lagged,
        pool,
        n_samples=12,
        burnin=5,
        rng=np.random.default_rng(9),
        noise_mode="empirical-joint",
    )

    assert simulated.shape == (12, 2)
    assert np.isfinite(simulated).all()
    assert not np.allclose(simulated, 0.0)


def _fixture_frame(n_samples: int = 120) -> pd.DataFrame:
    index = np.arange(n_samples)
    month = index % 12 + 1
    rng = np.random.default_rng(12)
    residual_x = rng.laplace(size=n_samples)
    residual_y = np.zeros(n_samples)
    for time_index in range(1, n_samples):
        residual_y[time_index] = (
            0.45 * residual_x[time_index]
            + 0.25 * residual_y[time_index - 1]
            + rng.laplace(scale=0.7)
        )
    return pd.DataFrame(
        {
            "row": 2,
            "col": 3,
            "year": 2005 + index // 12,
            "month": month,
            "x": 10.0 + 0.3 * month + 0.05 * index / 12 + residual_x,
            "y": 20.0 - 0.2 * month + 0.08 * index / 12 + residual_y,
        }
    )


def test_cli_writes_auditable_recovery_outputs(tmp_path: Path):
    source_frame = _fixture_frame()
    residualized, models = residualize_dataframe(
        source_frame,
        variables=["x", "y"],
        fit_end_year=None,
        min_fit_samples=60,
        suffix="_resid",
        expected_suffix="_seasonal_trend",
        include_trend=True,
        seasonal_model="monthly-fixed-effects",
        min_month_samples=3,
    )
    input_db = tmp_path / "ard.duckdb"
    con = duckdb.connect(input_db)
    try:
        con.register("residualized", residualized)
        con.execute("CREATE TABLE demo_residualized AS SELECT * FROM residualized")
        con.register("models", models)
        con.execute(
            "CREATE TABLE demo_residualized_residualization_models "
            "AS SELECT * FROM models"
        )
    finally:
        con.close()

    b0 = np.asarray([[0.0, 0.0], [0.45, 0.0]])
    lagged = np.asarray([[[0.0, 0.0], [0.0, 0.25]]])
    graph = pd.DataFrame(
        [
            {
                "row": 2,
                "col": 3,
                "model_type": "varlingam",
                "variable_names_json": json.dumps(["x_resid", "y_resid"]),
                "adjacency_raw_json": json.dumps(b0.tolist()),
                "adjacency_lagged_raw_json": json.dumps(lagged.tolist()),
                "var_lags": 1,
                "var_criterion": None,
                "var_prune": True,
            }
        ]
    )
    graph_db = tmp_path / "graphs.duckdb"
    con = duckdb.connect(graph_db)
    try:
        con.register("graph", graph)
        con.execute("CREATE TABLE pixel_graphs AS SELECT * FROM graph")
    finally:
        con.close()

    config_path = tmp_path / "demo_residualized.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "reference_var": "y_resid",
                "timeseries_db": str(input_db),
                "timeseries_table": "demo_residualized",
                "columns": [
                    {"name": "x_resid", "shift": 0},
                    {"name": "y_resid", "shift": 0},
                ],
                "residualization": {
                    "suffix": "_resid",
                    "expected_suffix": "_seasonal_trend",
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    result = CliRunner().invoke(
        validation.validate_varlingam_synthetic,
        [
            "--config-path",
            str(config_path),
            "--graphs-db",
            str(graph_db),
            "--row",
            "2",
            "--col",
            "3",
            "--n-samples",
            "120",
            "--replicates",
            "2",
            "--burnin",
            "20",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "coefficient_recovery.csv").exists()
    assert (output_dir / "summary_metrics.csv").exists()
    assert (output_dir / "dynamic_effect_recovery.csv").exists()
    assert (output_dir / "coefficient_recovery.pdf").exists()
    assert (output_dir / "cumulative_effect_recovery.pdf").exists()
    assert (output_dir / "source_causal_coefficients.csv").exists()
    assert (output_dir / "source_residualization_parameters.csv").exists()
    assert (output_dir / "source_innovation_moments.csv").exists()
    assert (output_dir / "validation_report.md").exists()
    statuses = pd.read_csv(output_dir / "replicate_status.csv")
    assert statuses["status"].tolist() == ["fit", "fit"]
    parameters = json.loads(
        (output_dir / "simulation_parameters.json").read_text(encoding="utf-8")
    )
    assert parameters["n_samples"] == 120
    assert parameters["target"] == "y_resid"
    effects = pd.read_csv(output_dir / "dynamic_effect_recovery.csv")
    assert "estimated_scaled_cumulative_effect" in effects.columns
