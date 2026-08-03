from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from confoundry.visualize_varlingam_effect_maps import (
    grid_from_frame,
    parse_horizons,
    symmetric_color_limit,
    visualize_varlingam_effect_maps,
)


def test_parse_horizons_preserves_order_and_removes_duplicates():
    assert parse_horizons("0, 1,3,1,12") == [0, 1, 3, 12]


def test_parse_horizons_rejects_negative_values():
    with pytest.raises(Exception, match="non-negative"):
        parse_horizons("0,-1")


def test_grid_from_frame_uses_explicit_sparse_domain():
    frame = pd.DataFrame(
        {
            "row": [10, 20],
            "col": [5, 15],
            "effect": [0.25, -0.5],
        }
    )
    grid = grid_from_frame(
        frame,
        row_column="row",
        col_column="col",
        value_column="effect",
        rows=[10, 20],
        cols=[5, 15],
    )
    np.testing.assert_allclose(
        grid,
        np.asarray([[0.25, np.nan], [np.nan, -0.5]]),
        equal_nan=True,
    )


def test_symmetric_color_limit_uses_absolute_quantile():
    assert symmetric_color_limit([-4.0, -1.0, 2.0, 3.0], 1.0) == 4.0


def _effect_rows() -> pd.DataFrame:
    rows = []
    sources = [
        "temperature_resid",
        "precipitation_resid",
        "evaporation_resid",
        "soil_moisture_7_to_28_cm_resid",
        "soil_moisture_28_to_100_cm_resid",
    ]
    for row in range(2):
        for col in range(2):
            for source_index, source in enumerate(sources):
                for horizon in [0, 1, 12]:
                    effect = (
                        (1 if source_index == 0 else -1)
                        * (row + col + 1)
                        * (horizon + 1)
                        / 100.0
                    )
                    rows.append(
                        {
                            "row": row,
                            "col": col,
                            "source": source,
                            "target": "ndvi_resid",
                            "horizon": horizon,
                            "scaled_total_effect": effect,
                            "scaled_total_effect_boot_ci_low": effect - 0.01,
                            "scaled_total_effect_boot_ci_high": effect + 0.01,
                            "scaled_total_effect_boot_ci_excludes_zero": abs(effect)
                            > 0.01,
                            "scaled_cumulative_total_effect": effect * 2.0,
                            "scaled_cumulative_total_effect_boot_ci_low": effect
                            * 2.0
                            - 0.01,
                            "scaled_cumulative_total_effect_boot_ci_high": effect
                            * 2.0
                            + 0.01,
                            "scaled_cumulative_total_effect_boot_ci_excludes_zero": abs(
                                effect * 2.0
                            )
                            > 0.01,
                            "error": None,
                        }
                    )
    return pd.DataFrame(rows)


def test_cli_writes_main_supplement_qc_and_figure_data(tmp_path):
    effects_db = tmp_path / "effects.duckdb"
    effects = _effect_rows()
    con = duckdb.connect(str(effects_db))
    con.register("_effects", effects)
    con.execute(
        "CREATE TABLE pixel_varlingam_effects_primary AS "
        "SELECT * FROM _effects"
    )
    con.close()

    qc_csv = tmp_path / "qc.csv"
    pd.DataFrame(
        {
            "row": [0, 0, 1, 1, 2],
            "col": [0, 1, 0, 1, 1],
            "primary_eligible": [True, True, True, True, False],
        }
    ).to_csv(qc_csv, index=False)
    output_dir = tmp_path / "maps"

    result = CliRunner().invoke(
        visualize_varlingam_effect_maps,
        [
            "--effects-db",
            str(effects_db),
            "--qc-csv",
            str(qc_csv),
            "--output-dir",
            str(output_dir),
            "--target",
            "ndvi_resid",
            "--main-horizon",
            "12",
            "--supplement-horizons",
            "0,1",
            "--stipple-stride",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    expected_figures = [
        "cumulative_effects_main",
        "primary_qc_coverage",
        "horizon_effects_month_00",
        "horizon_effects_month_01",
    ]
    for stem in expected_figures:
        assert (output_dir / "figures" / f"{stem}.png").exists()
        assert (output_dir / "figures" / f"{stem}.pdf").exists()

    main_data = pd.read_csv(
        output_dir / "tables" / "cumulative_effects_main_data.csv"
    )
    summary = pd.read_csv(
        output_dir / "tables" / "cumulative_effects_main_summary.csv"
    )
    metadata = pd.read_csv(
        output_dir / "tables" / "effect_map_figure_metadata.csv"
    )
    assert len(main_data) == 20
    assert set(summary["source"]) == {
        "temperature_resid",
        "precipitation_resid",
        "evaporation_resid",
        "soil_moisture_7_to_28_cm_resid",
        "soil_moisture_28_to_100_cm_resid",
    }
    assert set(metadata["horizon"]) == {0, 1, 12}
    assert "mapped pixels: 4" in result.output
