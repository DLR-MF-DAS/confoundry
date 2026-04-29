import pytest
import rasterio
from rasterio.transform import xy, from_origin
import numpy as np
from confoundry.analysis import (
    timeseries_causal_analysis,
)
from confoundry.gather import (
    assemble_data_frame,
    assemble_timeseries,
)
from dowhy import CausalModel
from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock


graph = """
digraph {
    soil_moisture -> ndvi;
    drought_severity -> soil_moisture;
    precipitation -> drought_severity;
    temperature -> {drought_severity ndvi};
    solar_radiation -> {drought_severity ndvi};
    world_cover -> {soil_moisture, irrigation ndvi};
    irrigation -> soil_moisture;
}
"""
