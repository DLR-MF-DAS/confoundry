import importlib.util
import json
import sys
import types
from pathlib import Path

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

import confoundry.order_graphs as og


pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb is not installed")


def test_varlingam_config_uses_noncolliding_graph_database():
    config = {
        "name": "demo",
        "graph_discovery": {
            "model": "varlingam",
            "output_db": "demo_residualized_graphs.duckdb",
        },
    }

    assert og.graph_db_path(config, Path("/experiment")) == Path(
        "/experiment/demo_residualized_varlingam_graphs.duckdb"
    )


def test_order_graphs_reads_residualized_graph_db_from_generated_config(
    tmp_path,
    monkeypatch,
):
    import duckdb

    config_path = tmp_path / "demo_residualized.yaml"
    graph_db = tmp_path / "demo_residualized_graphs.duckdb"

    config_path.write_text(
        "name: demo\n"
        "columns:\n"
        "  - name: a_residual\n"
        "    shift: 0\n"
        "  - name: b_residual\n"
        "    shift: 0\n"
        "graph_discovery:\n"
        "  input_db: demo_residualized.duckdb\n"
        "  input_table: demo_residualized\n"
        "  output_db: demo_residualized_graphs.duckdb\n"
        "residualization:\n"
        "  variables: [a, b]\n"
        "  suffix: _residual\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "row": [0, 0, 1],
            "col": [0, 1, 0],
            "adjacency_consensus_json": [
                json.dumps([[0.0, 0.1], [0.0, 0.0]]),
                json.dumps([[0.0, 0.2], [0.0, 0.0]]),
                json.dumps([[0.0, 0.0], [0.3, 0.0]]),
            ],
        }
    )

    con = duckdb.connect(graph_db)
    con.register("df", df)
    con.execute("CREATE TABLE pixel_graphs AS SELECT * FROM df")
    con.close()

    edge_plot_call = {}

    def fake_plot_edge_signature_by_color(
        mats,
        color_values,
        variable_names,
        outpath,
        **kwargs,
    ):
        edge_plot_call["variable_names"] = variable_names

    monkeypatch.setattr(og, "plot_map", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        og,
        "plot_edge_signature_by_color",
        fake_plot_edge_signature_by_color,
    )

    result = CliRunner().invoke(
        og.order_graphs,
        ["--config-path", str(config_path), "--mode", "abs"],
    )

    assert result.exit_code == 0, result.output

    con = duckdb.connect(graph_db, read_only=True)
    try:
        result_df = con.execute(
            "SELECT row, col, similarity_rank, similarity_order "
            "FROM pixel_graph_similarity_order ORDER BY row, col"
        ).fetchdf()
    finally:
        con.close()

    assert len(result_df) == 3
    assert set(result_df["similarity_rank"]) == {0, 1, 2}
    assert result_df["similarity_order"].between(0, 1).all()
    assert edge_plot_call["variable_names"] == ["a_residual", "b_residual"]
