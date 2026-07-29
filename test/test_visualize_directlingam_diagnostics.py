import json

import duckdb
import numpy as np
import pandas as pd
from click.testing import CliRunner

from confoundry.visualize_directlingam_diagnostics import (
    aggregate_bootstrap_edges,
    aggregate_residual_pairs,
    publication_variable_label,
    save_histogram,
    visualize_diagnostics,
)


def test_publication_variable_labels_expand_names_and_residual_suffixes():
    assert publication_variable_label("temperature_resid") == (
        "2 m air temperature anomaly"
    )
    assert publication_variable_label("ndvi_resid") == "NDVI anomaly"
    assert publication_variable_label("soil_moisture_7_to_28_cm_resid") == (
        "Soil moisture (7–28 cm) anomaly"
    )
    assert publication_variable_label("custom_driver") == "Custom driver"


def test_publication_variable_label_override_accepts_raw_or_base_name():
    assert publication_variable_label(
        "temperature_resid",
        {"temperature_resid": "Air-temperature anomaly"},
    ) == "Air-temperature anomaly"
    assert publication_variable_label(
        "temperature_resid",
        {"temperature": "Near-surface air temperature"},
    ) == "Near-surface air temperature anomaly"


def test_aggregate_labels_are_human_readable():
    diagnostics = pd.DataFrame(
        {
            "residual_corr_top_pairs_json": [
                json.dumps(
                    [
                        {
                            "var1": "temperature_resid",
                            "var2": "ndvi_resid",
                            "abs_value": 0.4,
                            "value": -0.4,
                        }
                    ]
                )
            ],
            "bootstrap_top_edges_json": [
                json.dumps(
                    [
                        {
                            "parent": "precipitation_resid",
                            "child": "ndvi_resid",
                            "probability": 0.9,
                            "abs_coefficient": 0.2,
                            "in_consensus": True,
                        }
                    ]
                )
            ],
        }
    )

    pairs = aggregate_residual_pairs(diagnostics)
    edges = aggregate_bootstrap_edges(diagnostics)

    assert pairs.loc[0, "pair"] == "2 m air temperature anomaly ↔ NDVI anomaly"
    assert edges.loc[0, "edge"] == "Total precipitation anomaly → NDVI anomaly"


def test_publication_figure_writes_png_and_pdf(tmp_path):
    output_path = tmp_path / "histogram.png"

    save_histogram(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        "Maximum absolute residual correlation",
        "Maximum absolute residual correlation",
        output_path,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()


def test_report_uses_publication_labels_and_writes_vector_figures(tmp_path):
    diagnostics_db = tmp_path / "diagnostics.duckdb"
    output_dir = tmp_path / "report"
    diagnostics = pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "model_type": ["varlingam"] * 4,
            "residual_max_abs_corr": [0.1, 0.2, 0.3, 0.4],
            "residual_corr_top_pairs_json": [
                json.dumps(
                    [
                        {
                            "var1": "temperature_resid",
                            "var2": "ndvi_resid",
                            "abs_value": value,
                            "value": value,
                        }
                    ]
                )
                for value in [0.1, 0.2, 0.3, 0.4]
            ],
        }
    )
    con = duckdb.connect(diagnostics_db)
    try:
        con.register("diagnostics", diagnostics)
        con.execute(
            "CREATE TABLE pixel_graph_diagnostics AS SELECT * FROM diagnostics"
        )
    finally:
        con.close()

    result = CliRunner().invoke(
        visualize_diagnostics,
        [
            "--diagnostics-db",
            str(diagnostics_db),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        output_dir / "figures" / "heatmap_residual_max_abs_corr.pdf"
    ).exists()
    assert (
        output_dir / "figures" / "bar_residual_correlation_pairs.pdf"
    ).exists()
    report = (output_dir / "diagnostics_report.html").read_text(encoding="utf-8")
    assert "VAR-LiNGAM diagnostics report" in report
    assert "Maximum absolute residual correlation" in report
    assert "2 m air temperature anomaly ↔ NDVI anomaly" in report
