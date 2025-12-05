import pytest
import rasterio
from rasterio.transform import xy
import numpy as np
from drought_causality.analysis import assemble_data_frame
from dowhy import CausalModel

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

def test_assemble_data_frame():
    precipitation_file = 'data/california_example/era5_precip_2021_07_california.tif'
    ndvi_file = 'data/california_example/ndvi_2021_07_california.tif'
    temperature_file = 'data/california_example/era5_t2m_2021_07_california.tif'
    solar_radiation_file = 'data/california_example/era5_ssrd_2021_07_california.tif'
    soil_moisture_file = 'data/california_example/era5_swvl1_2021_07_california.tif'
    world_cover_file = 'data/california_example/worldcover_2021_california_0p1deg.tif'
    irrigation_file = 'data/california_example/gmia_irrigation_0p1deg_california.tif'
    drought_severity_file = 'data/california_example/spei01_clipped_aoi_2021-07.tif'
    files = {
        'ndvi': ndvi_file,
        'precipitation': precipitation_file,
        'temperature': temperature_file,
        'solar_radiation': solar_radiation_file,
        'soil_moisture': soil_moisture_file,
        'world_cover': world_cover_file,
        'irrigation': irrigation_file,
        'drought_severity': drought_severity_file,
    }
    df = assemble_data_frame('ndvi', files)
    assert(df.shape == (29516, 12))
    df = df.dropna()
    model = CausalModel(
        data=df,
        treatment='drought_severity',
        outcome='ndvi',
        graph=graph
    )
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
