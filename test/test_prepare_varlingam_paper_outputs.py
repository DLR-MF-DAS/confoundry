from __future__ import annotations

import duckdb
import pandas as pd
from click.testing import CliRunner

from confoundry.analysis_helpers import write_dataframe_table
from confoundry.per_pixel_varlingam_analysis import aggregate_effects
from confoundry.prepare_varlingam_paper_outputs import (
    build_quality_populations,
    prepare_varlingam_paper_outputs,
)


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": [0, 0, 1],
            "col": [0, 1, 0],
            "var_lags": [3, 3, 3],
            "residual_whiteness_p": [0.20, 0.01, 0.01],
            "residual_whiteness_rejected": [False, True, True],
            "residual_whiteness_bootstrap_p": [0.20, 0.10, 0.01],
            "residual_whiteness_bootstrap_rejected": [False, False, True],
            "residual_whiteness_bootstrap_status": ["ok", "ok", "ok"],
            "residual_whiteness_bootstrap_samples_valid": [499, 499, 499],
            "residual_max_abs_corr": [0.1, 0.1, 0.1],
            "residual_lag1_max_median_abs_autocorr": [0.1, 0.1, 0.1],
            "residual_crosslag_max_abs_corr": [0.1, 0.1, 0.1],
            "residual_nongaussian_fraction": [0.8, 0.8, 0.8],
            "near_constant_variable_count": [0, 0, 0],
            "var_stability_radius": [0.7, 0.7, 0.7],
            "var_stable": [True, True, True],
            "var_bootstrap_stable_fraction": [1.0, 1.0, 1.0],
        }
    )


def _effects() -> pd.DataFrame:
    rows = []
    for row, col in [(0, 0), (0, 1), (1, 0)]:
        for horizon in [0, 1]:
            effect = float(row + col + horizon + 1) / 10.0
            rows.append(
                {
                    "row": row,
                    "col": col,
                    "source": "precipitation_resid",
                    "target": "ndvi_resid",
                    "horizon": horizon,
                    "error": None,
                    "point_stable": True,
                    "scaled_total_effect": effect,
                    "scaled_cumulative_total_effect": effect * 2.0,
                    "scaled_total_effect_boot_ci_excludes_zero": True,
                    "scaled_total_effect_boot_sd": 0.01,
                    "n_bootstrap_stable": 500,
                    "n_bootstrap_total": 500,
                }
            )
    return pd.DataFrame(rows)


def test_quality_populations_are_nested():
    qc = build_quality_populations(
        _diagnostics(),
        row_columns=["row", "col"],
        residual_corr_threshold=0.20,
        residual_crosslag_corr_threshold=0.30,
        autocorr_threshold=0.30,
        bootstrap_stable_fraction_min=0.95,
    )
    assert int(qc["primary_eligible"].sum()) == 2
    assert int(qc["analytical_sensitivity_eligible"].sum()) == 1
    assert not (
        qc["analytical_sensitivity_eligible"]
        & ~qc["primary_eligible"]
    ).any()


def test_cli_materializes_effect_populations_and_summaries(tmp_path):
    diagnostics_db = tmp_path / "diagnostics.duckdb"
    con = duckdb.connect(str(diagnostics_db))
    write_dataframe_table(con, _diagnostics(), "pixel_graph_diagnostics")
    con.close()

    effects_db = tmp_path / "effects.duckdb"
    effects = _effects()
    con = duckdb.connect(str(effects_db))
    write_dataframe_table(con, effects, "pixel_varlingam_effects")
    write_dataframe_table(
        con,
        aggregate_effects(effects),
        "varlingam_effect_summary",
    )
    con.close()

    output_dir = tmp_path / "paper"
    result = CliRunner().invoke(
        prepare_varlingam_paper_outputs,
        [
            "--diagnostics-db",
            str(diagnostics_db),
            "--effects-db",
            str(effects_db),
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "primary=2" in result.output
    assert "analytical sensitivity=1" in result.output

    qc = pd.read_csv(output_dir / "pixel_qc.csv")
    assert int(qc["primary_eligible"].sum()) == 2
    assert int(qc["analytical_sensitivity_eligible"].sum()) == 1

    con = duckdb.connect(str(effects_db), read_only=True)
    assert con.execute(
        "SELECT count(*) FROM pixel_varlingam_effects_primary"
    ).fetchone()[0] == 4
    assert con.execute(
        "SELECT count(*) "
        "FROM pixel_varlingam_effects_analytical_sensitivity"
    ).fetchone()[0] == 2
    metadata = con.execute(
        "SELECT primary_pixels, analytical_sensitivity_pixels "
        "FROM varlingam_population_run_metadata"
    ).fetchone()
    con.close()
    assert metadata == (2, 1)
    assert (output_dir / "effect_summary_primary.csv").exists()
    assert (
        output_dir / "effect_summary_analytical_sensitivity.csv"
    ).exists()
    assert (
        output_dir / "effect_summary_sensitivity_comparison.csv"
    ).exists()
