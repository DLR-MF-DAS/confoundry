import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from click.testing import CliRunner

from confoundry.residualize_timeseries import (
    residualize_dataframe,
    residualize_timeseries,
)


def _monthly_frame(n_years: int = 20) -> pd.DataFrame:
    n_samples = n_years * 12
    index = np.arange(n_samples)
    month = index % 12 + 1
    angle = 2.0 * np.pi * (month - 1) / 12.0
    seasonal_shape = np.asarray(
        [0.0, 0.5, 2.0, 4.0, 3.0, 1.0, -0.5, -1.0, -2.5, -3.0, -1.5, -0.2]
    )
    values = 10.0 + seasonal_shape[month - 1] + 0.25 * (index / 12.0)
    return pd.DataFrame(
        {
            "row": np.zeros(n_samples, dtype=int),
            "col": np.zeros(n_samples, dtype=int),
            "year": 2000 + index // 12,
            "month": month,
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
            "signal": values,
        }
    )


def test_monthly_fixed_effects_remove_nonsinusoidal_cycle_and_trend():
    frame = _monthly_frame()

    residualized, models = residualize_dataframe(
        frame,
        variables=["signal"],
        fit_end_year=None,
        min_fit_samples=60,
        suffix="_resid",
        expected_suffix="_expected",
        include_trend=True,
        seasonal_model="monthly-fixed-effects",
        min_month_samples=3,
    )

    np.testing.assert_allclose(
        residualized["signal_expected"],
        frame["signal"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        residualized["signal_resid"],
        0.0,
        atol=1e-10,
    )
    model = models.iloc[0]
    assert model["status"] == "fit"
    assert model["seasonal_model"] == "monthly-fixed-effects"
    assert model["n_parameters"] == 13
    assert model["design_rank"] == 13
    assert model["min_month_fit_samples"] == 20
    assert model["fit_r2"] > 0.999999
    coefficients = json.loads(model["coefficients_json"])
    assert "month_12" in coefficients
    assert "time_trend_per_year" in coefficients


def test_monthly_fixed_effects_require_calendar_month_coverage():
    frame = _monthly_frame().loc[lambda value: value["month"] != 12].copy()

    residualized, models = residualize_dataframe(
        frame,
        variables=["signal"],
        fit_end_year=None,
        min_fit_samples=60,
        suffix="_resid",
        expected_suffix="_expected",
        include_trend=True,
        seasonal_model="monthly-fixed-effects",
        min_month_samples=3,
    )

    assert models.iloc[0]["status"] == "too_few_samples_in_calendar_month"
    assert models.iloc[0]["min_month_fit_samples"] == 0
    assert residualized["signal_resid"].isna().all()


def test_legacy_harmonic_model_remains_available():
    frame = _monthly_frame()

    residualized, models = residualize_dataframe(
        frame,
        variables=["signal"],
        fit_end_year=None,
        min_fit_samples=24,
        suffix="_resid",
        expected_suffix="_expected",
        include_trend=True,
        seasonal_model="annual-harmonic",
    )

    assert models.iloc[0]["status"] == "fit"
    assert models.iloc[0]["seasonal_model"] == "annual-harmonic"
    assert models.iloc[0]["n_parameters"] == 4
    assert residualized["signal_resid"].std() > 0.1


def test_cli_records_strong_residualization_configuration(tmp_path: Path):
    frame = _monthly_frame()
    input_db = tmp_path / "demo.duckdb"
    con = duckdb.connect(input_db)
    try:
        con.register("frame", frame)
        con.execute("CREATE TABLE demo AS SELECT * FROM frame")
    finally:
        con.close()

    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "timeseries_table": "demo",
                "columns": [
                    {"name": "signal", "shift": 0},
                    {"name": "month_sin", "shift": 0},
                    {"name": "month_cos", "shift": 0},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_db = tmp_path / "residualized.duckdb"
    output_config = tmp_path / "residualized.yaml"

    result = CliRunner().invoke(
        residualize_timeseries,
        [
            "--config-path",
            str(config_path),
            "--input-db",
            str(input_db),
            "--output-db",
            str(output_db),
            "--output-config",
            str(output_config),
            "--seasonal-model",
            "monthly-fixed-effects",
            "--min-fit-samples",
            "60",
            "--min-month-samples",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert generated["residualization"]["seasonal_model"] == (
        "monthly-fixed-effects"
    )
    assert generated["residualization"]["controls"] == [
        "calendar_month_fixed_effects",
        "linear_time_trend",
    ]
    con = duckdb.connect(output_db, read_only=True)
    try:
        model = con.execute(
            "SELECT * FROM demo_residualized_residualization_models"
        ).fetchdf().iloc[0]
    finally:
        con.close()
    assert model["status"] == "fit"
    assert model["design_rank"] == 13
