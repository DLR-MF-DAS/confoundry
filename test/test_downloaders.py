from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from drought_causality.downloaders import (
    SPEIDownloader,
    MODISNDVIDownloader,
    ERA5Downloader,
    ERA5PrecipDownloader,
    ERA5SoilMoistureDownloader,
    ESAWorldCoverDownloader,
    IrrigationMapDownloader,
)


# Global test variables for consistency
TEST_YEAR = 2021
TEST_MONTH = 7
TEST_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-124.0, 32.0],
            [-114.0, 32.0],
            [-114.0, 42.0],
            [-124.0, 42.0],
            [-124.0, 32.0],
        ]
    ],
}


def _dummy_da():
    """Small dummy DataArray with proper x/y spatial dims."""
    data = np.zeros((1, 2, 2), dtype=float)
    coords = {
        "time": [0],
        "y": [0.0, 1.0],
        "x": [0.0, 1.0],
    }
    da = xr.DataArray(data, coords=coords, dims=("time", "y", "x"))

    try:
        import rioxarray  # noqa: F401

        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    except Exception:
        # If rioxarray or rio accessor isn't available, we still get a usable object.
        pass

    return da


def _dummy_era5_ds():
    """Dummy ERA5-like Dataset with t2m and ssrd variables."""
    da = _dummy_da()
    return xr.Dataset({"t2m": da, "ssrd": da})


def _dummy_era5_precip_ds():
    """Dummy ERA5 precip Dataset with tp variable."""
    da = _dummy_da()
    return xr.Dataset({"tp": da})


def _dummy_soil_moisture_ds():
    """Dummy ERA5 soil moisture Dataset with swvl1 variable."""
    da = _dummy_da()
    return xr.Dataset({"swvl1": da})


@patch("drought_causality.downloaders.SPEIDownloader.download")
def test_spei_downloader(mock_download, tmp_path: Path):
    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    # Initialize downloader and perform download
    downloader = SPEIDownloader(cache_dir=tmp_path)
    downloader.download(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    downloader.data = mock_da

    # Save GeoTIFF and check existence
    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"spei_test_{TEST_YEAR}_{TEST_MONTH:02d}"
        )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    for path in save_paths:
        assert Path(path).exists(
        )

    # Check that the saved GeoTIFF validation works
    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"spei_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    ) 
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)
    
@patch("drought_causality.downloaders.MODISNDVIDownloader.download")
def test_modis_ndvi_downloader(mock_download, tmp_path: Path):
    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    downloader = MODISNDVIDownloader(cache_dir=tmp_path)
    downloader.download(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    downloader.data = mock_da

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"modis_ndvi_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
    )
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"modis_ndvi_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)


# ---- ERA5: patch __init__ + download so CDS is never touched ----
@patch("drought_causality.downloaders.ERA5Downloader.download")
@patch("drought_causality.downloaders.ERA5Downloader.__init__", return_value=None)
def test_era5_downloader(mock_init, mock_download, tmp_path: Path):
    mock_ds = _dummy_era5_ds()
    mock_download.return_value = mock_ds

    downloader = ERA5Downloader(cache_dir=tmp_path)  # __init__ is patched to do nothing
    downloader.download(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    downloader.data = mock_ds

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"era5_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
    )
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"era5_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)

@patch("drought_causality.downloaders.ERA5PrecipDownloader.download")
@patch("drought_causality.downloaders.ERA5PrecipDownloader.__init__", return_value=None)
def test_era5precip_downloader(mock_init, mock_download, tmp_path: Path):
    mock_ds = _dummy_era5_precip_ds()
    mock_download.return_value = mock_ds

    downloader = ERA5PrecipDownloader(cache_dir=tmp_path)  # __init__ patched
    downloader.download(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    downloader.data = mock_ds

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"era5_precip_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
    )
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"era5_precip_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)

@patch("drought_causality.downloaders.ERA5SoilMoistureDownloader.download")
@patch("drought_causality.downloaders.ERA5SoilMoistureDownloader.__init__", return_value=None)
def test_era5_soil_moisture_downloader(mock_init, mock_download, tmp_path: Path):
    mock_ds = _dummy_soil_moisture_ds()
    mock_download.return_value = mock_ds

    downloader = ERA5SoilMoistureDownloader(cache_dir=tmp_path)  # __init__ patched
    downloader.download(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
        )
    downloader.data = mock_ds

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"era5_soil_moisture_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        year=TEST_YEAR, 
        month=TEST_MONTH
    )
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"era5_soil_moisture_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)

@patch("drought_causality.downloaders.ESAWorldCoverDownloader.download")
def test_esa_world_cover_downloader(mock_download, tmp_path: Path):
    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    downloader = ESAWorldCoverDownloader(year=TEST_YEAR, cache_dir=tmp_path)
    downloader.download(
        polygon=TEST_POLYGON, 
        target_res_deg=0.1
        )
    downloader.data = mock_da

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"esa_world_cover_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(
        polygon=TEST_POLYGON, 
        target_res_deg=0.1
    )
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"esa_world_cover_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)

@patch("drought_causality.downloaders.IrrigationMapDownloader.download")
def test_irrigation_map_downloader(mock_download, tmp_path: Path):
    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    downloader = IrrigationMapDownloader(target_res_deg=0.1, cache_dir=tmp_path)
    downloader.download(
        polygon=TEST_POLYGON
        )
    downloader.data = mock_da

    save_paths = downloader.save_geotiff(
        output_dir=tmp_path,
        basename=f"irrigation_map_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    mock_download.assert_called_once_with(polygon=TEST_POLYGON)
    for path in save_paths:
        assert Path(path).exists()

    validate_paths = downloader.validate_geotiff(
        output_dir=tmp_path,
        basename=f"irrigation_map_test_{TEST_YEAR}_{TEST_MONTH:02d}"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)
    