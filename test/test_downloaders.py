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


def _example_polygon():
    """Simple polygon; content doesn't matter because download is mocked."""
    return {
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


@patch("drought_causality.downloaders.SPEIDownloader.download")
def test_spei_downloader(mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    downloader = SPEIDownloader()
    spei_da = downloader.download(polygon, year=2021, month=7)

    out_file = tmp_path / "spei01_clipped_aoi_2021-07.tif"
    spei_da.rio.to_raster(out_file)

    mock_download.assert_called_once_with(polygon, year=2021, month=7)
    assert out_file.exists()


@patch("drought_causality.downloaders.MODISNDVIDownloader.download")
def test_modis_ndvi_downloader(mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    downloader = MODISNDVIDownloader()
    ndvi_da = downloader.download(polygon, year=2021, month=7)

    out_file = tmp_path / "ndvi_2021_07_california.tif"
    ndvi_da.isel(time=0).rio.to_raster(out_file)

    mock_download.assert_called_once_with(polygon, year=2021, month=7)
    assert out_file.exists()


# ---- ERA5: patch __init__ + download so CDS is never touched ----

@patch("drought_causality.downloaders.ERA5Downloader.download")
@patch("drought_causality.downloaders.ERA5Downloader.__init__", return_value=None)
def test_era5_downloader(mock_init, mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_ds = _dummy_era5_ds()
    mock_download.return_value = mock_ds

    era5 = ERA5Downloader()  # __init__ is patched to do nothing
    ds_era5 = era5.download(polygon, year=2021, month=7)

    out_t2m = tmp_path / "era5_t2m_2021_07_california.tif"
    out_ssrd = tmp_path / "era5_ssrd_2021_07_california.tif"

    ds_era5["t2m"].isel(time=0).rio.to_raster(out_t2m)
    ds_era5["ssrd"].isel(time=0).rio.to_raster(out_ssrd)

    mock_download.assert_called_once_with(polygon, year=2021, month=7)
    assert out_t2m.exists()
    assert out_ssrd.exists()


@patch("drought_causality.downloaders.ERA5PrecipDownloader.download")
@patch("drought_causality.downloaders.ERA5PrecipDownloader.__init__", return_value=None)
def test_era5precip_downloader(mock_init, mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_ds = _dummy_era5_precip_ds()
    mock_download.return_value = mock_ds

    era5 = ERA5PrecipDownloader()  # __init__ patched
    ds_era5 = era5.download(polygon, year=2021, month=7)

    out_tp = tmp_path / "era5_precip_2021_07_california.tif"
    ds_era5["tp"].isel(time=0).rio.to_raster(out_tp)

    mock_download.assert_called_once_with(polygon, year=2021, month=7)
    assert out_tp.exists()


@patch("drought_causality.downloaders.ERA5SoilMoistureDownloader.download")
@patch("drought_causality.downloaders.ERA5SoilMoistureDownloader.__init__", return_value=None)
def test_era5_soil_moisture_downloader(mock_init, mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_ds = _dummy_soil_moisture_ds()
    mock_download.return_value = mock_ds

    sm = ERA5SoilMoistureDownloader()  # __init__ patched
    ds_sm = sm.download(polygon, year=2021, month=7)

    assert "swvl1" in ds_sm.data_vars

    out_swvl1 = tmp_path / "era5_swvl1_2021_07_california.tif"
    ds_sm["swvl1"].isel(time=0).rio.to_raster(out_swvl1)

    mock_download.assert_called_once_with(polygon, year=2021, month=7)
    assert out_swvl1.exists()


@patch("drought_causality.downloaders.ESAWorldCoverDownloader.download")
def test_esa_world_cover_downloader(mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    wc = ESAWorldCoverDownloader(year=2021)
    da_lc = wc.download(polygon, target_res_deg=0.1)

    out_file = tmp_path / "worldcover_2021_california_0p1deg.tif"
    da_lc.rio.to_raster(out_file)

    mock_download.assert_called_once_with(polygon, target_res_deg=0.1)
    assert out_file.exists()


@patch("drought_causality.downloaders.IrrigationMapDownloader.download")
def test_irrigation_map_downloader(mock_download, tmp_path: Path):
    polygon = _example_polygon()

    mock_da = _dummy_da()
    mock_download.return_value = mock_da

    irr = IrrigationMapDownloader(target_res_deg=0.1)
    da_irr = irr.download(polygon)

    out_file = tmp_path / "gmia_irrigation_0p1deg_california.tif"
    da_irr.rio.to_raster(out_file)

    mock_download.assert_called_once_with(polygon)
    assert out_file.exists()


