import importlib.util
import json
import sys
import types

import click
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None
HAS_LINGAM = importlib.util.find_spec("lingam") is not None

if not HAS_DUCKDB:
    sys.modules.setdefault(
        "duckdb",
        types.SimpleNamespace(connect=lambda *args, **kwargs: None),
    )

if not HAS_LINGAM:
    sys.modules.setdefault(
        "lingam",
        types.SimpleNamespace(DirectLiNGAM=None),
    )

import confoundry.per_pixel_graph_discovery as gd


class DummyBootstrap:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def get_probabilities(self, min_causal_effect):
        return self.probabilities


class DummyDirectLiNGAM:
    def __init__(self, prior_knowledge=None, random_state=None):
        self.prior_knowledge = prior_knowledge
        self.random_state = random_state
        self.adjacency_matrix_ = np.array(
            [
                [0.0, 0.05],
                [0.2, 0.0],
            ]
        )
        self.causal_order_ = [1, 0]

    def fit(self, X):
        self.X_ = np.asarray(X)
        return self

    def bootstrap(self, X, n_sampling):
        return DummyBootstrap(
            [
                [0.0, 0.8],
                [0.9, 0.0],
            ]
        )


def make_group(row, col, values):
    return pd.DataFrame(
        {
            "row": row,
            "col": col,
            "value": values,
        }
    )


def test_get_pixel_window_group_window_size_zero_returns_center_only():
    group_lookup = {
        (0, 0): make_group(0, 0, [1, 2]),
        (0, 1): make_group(0, 1, [3, 4]),
    }

    result = gd.get_pixel_window_group((0, 0), group_lookup, window_size=0)

    assert result is not None
    assert result["value"].tolist() == [1, 2]
    assert set(zip(result["row"], result["col"])) == {(0, 0)}


def test_get_pixel_window_group_collects_available_neighbors():
    group_lookup = {
        (0, 0): make_group(0, 0, [1]),
        (0, 1): make_group(0, 1, [2]),
        (1, 0): make_group(1, 0, [3]),
        (2, 2): make_group(2, 2, [99]),
    }

    result = gd.get_pixel_window_group((0, 0), group_lookup, window_size=1)

    assert result is not None
    assert set(result["value"]) == {1, 2, 3}
    assert set(zip(result["row"], result["col"])) == {(0, 0), (0, 1), (1, 0)}


def test_get_pixel_window_group_returns_none_when_no_groups_exist():
    result = gd.get_pixel_window_group((10, 10), {}, window_size=1)

    assert result is None


def test_get_pixel_window_group_rejects_negative_window_size():
    with pytest.raises(ValueError, match="window_size must be >= 0"):
        gd.get_pixel_window_group((0, 0), {}, window_size=-1)


def test_parse_columns_sorts_and_shifts_within_each_group():
    df = pd.DataFrame(
        {
            "row": [0, 0, 0, 0],
            "col": [0, 0, 1, 1],
            "year": [2020, 2020, 2020, 2020],
            "month": [2, 1, 2, 1],
            "x": [20.0, 10.0, 200.0, 100.0],
            "y": [2.0, 1.0, 20.0, 10.0],
        }
    )
    column_specs = [
        {"name": "x", "shift": 1},
        {"name": "y", "shift": 0},
    ]

    shifted_df, labels, label_lags = gd.parse_columns(
        df,
        group_cols=["row", "col"],
        order_cols=["year", "month"],
        column_specs=column_specs,
    )

    assert labels == ["x", "y"]
    assert label_lags == {"x": 1, "y": 0}

    pixel_00 = shifted_df[(shifted_df["row"] == 0) & (shifted_df["col"] == 0)]
    assert pixel_00["month"].tolist() == [1, 2]
    assert np.isnan(pixel_00.iloc[0]["x"])
    assert pixel_00.iloc[1]["x"] == 10.0
    assert pixel_00["y"].tolist() == [1.0, 2.0]


def test_parse_columns_rejects_duplicate_labels():
    df = pd.DataFrame(
        {
            "row": [0],
            "col": [0],
            "year": [2020],
            "month": [1],
            "x": [1.0],
        }
    )
    column_specs = [
        {"name": "x", "shift": 0},
        {"name": "x", "shift": 1},
    ]

    with pytest.raises(click.BadParameter, match="Duplicate derived column: x"):
        gd.parse_columns(df, ["row", "col"], ["year", "month"], column_specs)


def test_parse_columns_rejects_missing_data_column():
    df = pd.DataFrame(
        {
            "row": [0],
            "col": [0],
            "year": [2020],
            "month": [1],
            "x": [1.0],
        }
    )

    with pytest.raises(click.BadParameter, match="Missing data column: y"):
        gd.parse_columns(
            df,
            ["row", "col"],
            ["year", "month"],
            [{"name": "y", "shift": 0}],
        )


def test_make_prior_knowledge_blocks_time_inconsistent_edges_and_calendar_causes():
    labels = ["lagged", "current", "month_sin"]
    label_lags = {
        "lagged": 2,
        "current": 0,
        "month_sin": 0,
    }

    prior_knowledge = gd.make_prior_knowledge(labels, label_lags)

    assert prior_knowledge.shape == (3, 3)
    assert prior_knowledge[0, 1] == 0  # current cannot cause more-lagged variable
    assert prior_knowledge[1, 0] == -1
    assert prior_knowledge[2, :].tolist() == [0, 0, 0]


def test_to_graph_uses_lingam_child_parent_convention():
    B = np.array(
        [
            [0.0, 0.01],
            [0.2, 0.0],
        ]
    )

    graph = gd.to_graph(B, labels=["a", "b"], min_abs_effect=0.05)

    assert set(graph.nodes) == {"a", "b"}
    assert list(graph.edges(data=True)) == [("a", "b", {"weight": 0.2})]


def test_fit_pixel_returns_none_when_too_few_complete_samples():
    g = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, 2.0, np.nan],
        }
    )

    result = gd.fit_pixel(
        pixel_key=(3, 4),
        g=g,
        labels=["a", "b"],
        pk=np.full((2, 2), -1),
        bootstrap_samples=10,
        min_samples=2,
        min_prob=0.7,
        min_abs_effect=0.1,
        group_cols=["row", "col"],
    )

    assert result is None


def test_fit_pixel_thresholds_bootstrap_probabilities_and_effect_sizes(monkeypatch):
    monkeypatch.setattr(gd.lingam, "DirectLiNGAM", DummyDirectLiNGAM)
    g = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 3.0, 6.0],
        }
    )

    result = gd.fit_pixel(
        pixel_key=(3, 4),
        g=g,
        labels=["a", "b"],
        pk=np.full((2, 2), -1),
        bootstrap_samples=5,
        min_samples=2,
        min_prob=0.7,
        min_abs_effect=0.1,
        group_cols=["row", "col"],
    )

    assert result is not None
    assert result["row"] == 3
    assert result["col"] == 4
    assert result["n_samples"] == 3
    assert json.loads(result["variable_names_json"]) == ["a", "b"]
    assert json.loads(result["causal_order_json"]) == [1, 0]
    assert json.loads(result["adjacency_raw_json"]) == [[0.0, 0.05], [0.2, 0.0]]
    assert json.loads(result["edge_probability_json"]) == [[0.0, 0.8], [0.9, 0.0]]
    assert json.loads(result["adjacency_consensus_json"]) == [[0.0, 0.0], [0.2, 0.0]]

    graph = nx.parse_gml(result["gml_graph"])
    assert list(graph.edges(data=True)) == [("a", "b", {"weight": 0.2})]


def test_fit_pixel_task_delegates_to_fit_pixel(monkeypatch):
    calls = {}

    def fake_fit_pixel(**kwargs):
        calls.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(gd, "fit_pixel", fake_fit_pixel)

    result = gd.fit_pixel_task(
        (
            (1, 2),
            "group",
            ["a"],
            "pk",
            10,
            5,
            0.8,
            0.2,
            ["row", "col"],
        )
    )

    assert result == {"ok": True}
    assert calls == {
        "pixel_key": (1, 2),
        "g": "group",
        "labels": ["a"],
        "pk": "pk",
        "bootstrap_samples": 10,
        "min_samples": 5,
        "min_prob": 0.8,
        "min_abs_effect": 0.2,
        "group_cols": ["row", "col"],
    }


def test_graph_discovery_rejects_negative_window_size():
    runner = CliRunner()

    result = runner.invoke(
        gd.graph_discovery,
        ["--config-path", "missing.yaml", "--window-size", "-1"],
    )

    assert result.exit_code != 0
    assert "window-size must be >= 0" in result.output


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb is required for the CLI integration test")
def test_graph_discovery_cli_writes_output_duckdb(tmp_path, monkeypatch):
    import duckdb

    monkeypatch.setattr(gd.lingam, "DirectLiNGAM", DummyDirectLiNGAM)
    monkeypatch.setattr(
        gd,
        "process_map",
        lambda func, tasks, max_workers, chunksize, desc: [func(task) for task in tasks],
    )

    config_path = tmp_path / "demo.yaml"
    input_db = tmp_path / "demo_ard.duckdb"
    output_db = tmp_path / "demo_graphs.duckdb"

    config_path.write_text(
        "name: demo\n"
        "columns:\n"
        "  - name: a\n"
        "    shift: 0\n"
        "  - name: b\n"
        "    shift: 0\n"
    )

    records = []
    for row in [0, 1]:
        for col in [0, 1]:
            for month in [1, 2, 3]:
                records.append(
                    {
                        "row": row,
                        "col": col,
                        "year": 2020,
                        "month": month,
                        "a": float(month),
                        "b": float(month * 2),
                    }
                )
    df = pd.DataFrame(records)

    con = duckdb.connect(input_db)
    con.register("df", df)
    con.execute("CREATE TABLE demo AS SELECT * FROM df")
    con.close()

    runner = CliRunner()
    result = runner.invoke(
        gd.graph_discovery,
        [
            "--config-path",
            str(config_path),
            "--bootstrap-samples",
            "5",
            "--min-samples",
            "2",
            "--min-edge-prob",
            "0.7",
            "--min-abs-effect",
            "0.1",
            "--window-size",
            "1",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_db.exists()

    con = duckdb.connect(output_db, read_only=True)
    result_df = con.execute("SELECT * FROM pixel_graphs ORDER BY row, col").fetchdf()
    con.close()

    assert len(result_df) == 4
    assert result_df["n_samples"].tolist() == [12, 12, 12, 12]
    assert json.loads(result_df.iloc[0]["adjacency_consensus_json"]) == [[0.0, 0.0], [0.2, 0.0]]
