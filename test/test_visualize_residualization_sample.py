import matplotlib.pyplot as plt
import pytest

from confoundry.visualize_residualization_sample import match_vertical_scale


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
