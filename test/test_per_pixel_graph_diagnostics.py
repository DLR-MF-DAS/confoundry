import json
from pathlib import Path

import numpy as np

from confoundry.per_pixel_graph_diagnostics import (
    default_diagnostics_path,
    default_varlingam_graph_path,
    graph_config_value,
    resolve_path,
    structural_residuals_for_graph,
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
