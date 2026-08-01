import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from confoundry.visualize_residualization_sample import (
    autocorrelation_profile,
    autocorrelation_table,
    match_vertical_scale,
    parse_label_overrides,
)


def test_match_vertical_scale_uses_equal_spans_and_centers_residuals_on_zero():
    figure, (ax_original, ax_residual) = plt.subplots(1, 2)
    try:
        ax_original.plot([0, 1], [280.0, 300.0])
        ax_residual.plot([0, 1], [-4.0, 6.0])

        match_vertical_scale(ax_original, ax_residual)

        original_limits = ax_original.get_ylim()
        residual_limits = ax_residual.get_ylim()
        original_span = original_limits[1] - original_limits[0]
        residual_span = residual_limits[1] - residual_limits[0]

        assert original_span == pytest.approx(residual_span)
        assert residual_limits[0] == pytest.approx(-residual_limits[1])
        assert original_limits[0] > residual_limits[1]
    finally:
        plt.close(figure)


def test_autocorrelation_profile_reports_requested_lags():
    rng = np.random.default_rng(17)
    values = np.zeros(500, dtype=float)
    noise = rng.normal(size=len(values))
    for index in range(1, len(values)):
        values[index] = 0.8 * values[index - 1] + noise[index]

    profile = autocorrelation_profile(values, max_lag=12)

    assert profile.shape == (12,)
    assert profile[0] > 0.7
    assert profile[-1] < profile[0]


def test_publication_label_overrides_are_validated():
    assert parse_label_overrides(["ndvi=Vegetation index"]) == {
        "ndvi": "Vegetation index"
    }
    with pytest.raises(Exception, match="RAW=DISPLAY"):
        parse_label_overrides(["invalid"])


def test_autocorrelation_table_exports_before_and_after_profiles():
    frame = {
        "a": np.arange(30, dtype=float),
        "a_resid": np.tile([-1.0, 0.0, 1.0], 10),
    }

    result = autocorrelation_table(
        pd.DataFrame(frame),
        variables=["a"],
        suffix="_resid",
        max_lag=4,
    )

    assert result["lag"].tolist() == [1, 2, 3, 4]
    assert set(result.columns) == {
        "variable",
        "lag",
        "original_autocorrelation",
        "residual_autocorrelation",
        "approximate_white_noise_95_limit",
    }
