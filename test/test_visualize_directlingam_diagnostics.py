import json

import duckdb
import numpy as np
import pandas as pd
from click.testing import CliRunner

from confoundry.visualize_directlingam_diagnostics import (
    aggregate_bootstrap_edges,
    aggregate_crosslag_pairs,
    aggregate_lagged_bootstrap_edges,
    aggregate_residual_pairs,
    publication_variable_label,
    save_histogram,
    save_minimal_varlingam_diagnostics,
    save_temporal_model_comparison,
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
            "residual_crosslag_top_pairs_json": [
                json.dumps(
                    [
                        {
                            "source": "temperature_resid",
                            "target": "ndvi_resid",
                            "lag": 2,
                            "median_abs_correlation": 0.3,
                            "max_abs_correlation": 0.4,
                        }
                    ]
                )
            ],
            "lagged_bootstrap_top_edges_json": [
                json.dumps(
                    [
                        {
                            "parent": "precipitation_resid",
                            "child": "ndvi_resid",
                            "lag": 1,
                            "autoregressive": False,
                            "probability": 0.85,
                            "abs_coefficient": 0.15,
                            "in_consensus": True,
                        }
                    ]
                )
            ],
        }
    )

    pairs = aggregate_residual_pairs(diagnostics)
    edges = aggregate_bootstrap_edges(diagnostics)
    crosslag = aggregate_crosslag_pairs(diagnostics)
    lagged_edges = aggregate_lagged_bootstrap_edges(diagnostics)

    assert pairs.loc[0, "pair"] == "2 m air temperature anomaly ↔ NDVI anomaly"
    assert edges.loc[0, "edge"] == "Total precipitation anomaly → NDVI anomaly"
    assert crosslag.loc[0, "pair"] == (
        "2 m air temperature anomaly (t−2) → NDVI anomaly (t)"
    )
    assert lagged_edges.loc[0, "edge"] == (
        "Total precipitation anomaly (t−1) → NDVI anomaly (t)"
    )


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


def test_minimal_var_figure_and_temporal_comparison_are_written(tmp_path):
    var = pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "model_type": ["varlingam"] * 4,
            "residual_crosslag_max_abs_corr": [0.1, 0.2, 0.15, 0.25],
            "residual_whiteness_p": [0.4, 0.2, 0.3, 0.1],
            "residual_max_abs_corr": [0.05, 0.1, 0.08, 0.12],
            "residual_nongaussian_fraction": [0.8, 1.0, 0.8, 1.0],
            "var_stability_radius": [0.4, 0.6, 0.5, 0.7],
        }
    )
    direct = pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "model_type": ["directlingam"] * 4,
            "residual_crosslag_max_abs_corr": [0.4, 0.5, 0.35, 0.6],
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "residual_crosslag_corr_threshold": 0.2,
                "residual_corr_threshold": 0.2,
                "diagnostic_alpha": 0.05,
                "stability_threshold": 1.0,
            }
        ]
    )

    minimal_path = tmp_path / "minimal.png"
    save_minimal_varlingam_diagnostics(var, metadata, minimal_path)
    comparison_path = tmp_path / "comparison.png"
    summary, pairs = save_temporal_model_comparison(
        var,
        None,
        direct,
        None,
        comparison_path,
    )

    assert minimal_path.exists()
    assert minimal_path.with_suffix(".pdf").exists()
    assert comparison_path.exists()
    assert set(summary["model"]) == {"VAR-LiNGAM", "DirectLiNGAM"}
    assert len(pairs) == 4
    assert (pairs["primary_minus_comparison"] < 0).all()


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
