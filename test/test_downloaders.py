import pytest
import datetime
import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from drought_causality.downloaders.downloader import ItemDownloadReport
from drought_causality.downloaders.era5 import ERA5Downloader
from drought_causality.downloaders.modis_ndvi import MODISNDVIDownloader
from drought_causality.downloaders.spei import SPEIDownloader
from drought_causality.downloaders.esacci_landcover import ESACCILandCoverDownloader
from drought_causality.downloaders.ecira import ECIRADownloader


# Global test variables for consistency
TEST_START_DATE = datetime.datetime(2021, 1, 1)
TEST_END_DATE = datetime.datetime(2021, 3, 31)

# Test is california bounding box
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
    return xr.Dataset({"t2m": da, "ssrd": da, "tp": da, "swvl1": da})


def _dummy_era5_precip_ds():
    """Dummy ERA5 precip Dataset with tp variable."""
    da = _dummy_da()
    return xr.Dataset({"tp": da})


def _dummy_soil_moisture_ds():
    """Dummy ERA5 soil moisture Dataset with swvl1 variable."""
    da = _dummy_da()
    return xr.Dataset({"swvl1": da})


def test_spei_downloader_full(tmp_path):
    """Custom test for SPEIDownloader: download (mocked), _save_geotiff, _validate_geotiff."""
    # Prepare dummy report for download
    dummy_report = [
        ItemDownloadReport(
            data_source="test_source",
            variable_name="test_variable",
            acquisition_time=datetime.datetime(2020, 1, 1),
            path=tmp_path / "SPEI_202001.tif",
            download_successful=True,
            error=None,
            metadata=None,
        )
    ]

    with patch("drought_causality.downloaders.spei.SPEIDownloader.download") as mock_download:
        mock_download.return_value = dummy_report
        downloader = SPEIDownloader(cache_dir=tmp_path)
        report = downloader.download(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        mock_download.assert_called_once_with(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        assert isinstance(report, list)
        assert all(isinstance(item, ItemDownloadReport) for item in report)
        assert all(item.download_successful for item in report)
        assert downloader.frequency == "monthly"

    # Now test _save_geotiff and _validate_geotiff with dummy data
    da = _dummy_da()
    save_paths = downloader._save_geotiff(
        data=da,
        output_dir=tmp_path,
        basename="spei_test"
    )
    for path in save_paths.values():
        assert Path(path).exists()
    validate_paths = downloader._validate_geotiff(
        output_dir=tmp_path,
        basename="spei_test"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)


def test_modis_ndvi_downloader_full(tmp_path):
    """Custom test for MODISNDVIDownloader: download (mocked), _save_geotiff, _validate_geotiff."""

    # Prepare dummy report for download
    dummy_report = [
        ItemDownloadReport(
            data_source="test_source",
            variable_name="test_variable",
            acquisition_time=datetime.datetime(2020, 1, 1),
            path=tmp_path / "MODIS_NDVI_202001.tif",
            download_successful=True,
            error=None,
            metadata=None,
        )
    ]

    with patch("drought_causality.downloaders.modis_ndvi.MODISNDVIDownloader.download") as mock_download:
        mock_download.return_value = dummy_report
        downloader = MODISNDVIDownloader(cache_dir=tmp_path)
        report = downloader.download(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        mock_download.assert_called_once_with(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        assert isinstance(report, list)
        assert all(isinstance(item, ItemDownloadReport) for item in report)
        assert all(item.download_successful for item in report)
        assert downloader.frequency == "monthly"

    # Now test _save_geotiff and _validate_geotiff with dummy data
    da = _dummy_da()
    save_paths = downloader._save_geotiff(
        data=da,
        output_dir=tmp_path,
        basename="modis_ndvi_test"
    )
    for path in save_paths.values():
        assert Path(path).exists()
    validate_paths = downloader._validate_geotiff(
        output_dir=tmp_path,
        basename="modis_ndvi_test"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)


@patch("drought_causality.downloaders.era5.cdsapi.Client")
def test_era5_downloader_full(mock_client, tmp_path):
    """Custom test for ERA5Downloader: download (mocked), _save_geotiff, _validate_geotiff."""
    # Set up the mock to simulate .retrieve() behavior
    instance = mock_client.return_value
    instance.retrieve.return_value = None

    # Prepare dummy report for download
    dummy_report = [
        ItemDownloadReport(
            data_source="test_source",
            variable_name="test_variable",
            acquisition_time=datetime.datetime(2020, 1, 1),
            path=tmp_path / "ERA5_202001.tif",
            download_successful=True,
            error=None,
            metadata=None,
        )
    ]

    with patch("drought_causality.downloaders.era5.ERA5Downloader.download") as mock_download:
        mock_download.return_value = dummy_report
        downloader = ERA5Downloader(cache_dir=tmp_path)
        report = downloader.download(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        mock_download.assert_called_once_with(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        assert isinstance(report, list)
        assert all(isinstance(item, ItemDownloadReport) for item in report)
        assert all(item.download_successful for item in report)
        assert downloader.frequency == "monthly"

    # Now test _save_geotiff and _validate_geotiff with dummy data
    da = _dummy_era5_ds()
    save_paths = downloader._save_geotiff(
        data=da,
        output_dir=tmp_path,
        basename="era5_test"
    )
    for path in save_paths.values():
        assert Path(path).exists()
    validate_paths = downloader._validate_geotiff(
        output_dir=tmp_path,
        basename="era5_test"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)


def test_esacci_downloader_full(tmp_path):
    """Custom test for ESACCILandCoverDownloader: download (mocked), _save_geotiff, _validate_geotiff."""

    # Prepare dummy report for download
    dummy_report = [
        ItemDownloadReport(
            data_source="test_source",
            variable_name="test_variable",
            acquisition_time=datetime.datetime(2020, 1, 1),
            path=tmp_path / "ESACCI_Landcover_202001.tif",
            download_successful=True,
            error=None,
            metadata=None,
        )
    ]

    with patch("drought_causality.downloaders.esacci_landcover.ESACCILandCoverDownloader.download") as mock_download:
        mock_download.return_value = dummy_report
        downloader = ESACCILandCoverDownloader(cache_dir=tmp_path)
        report = downloader.download(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        mock_download.assert_called_once_with(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        assert isinstance(report, list)
        assert all(isinstance(item, ItemDownloadReport) for item in report)
        assert all(item.download_successful for item in report)
        assert downloader.frequency == "yearly"

    # Now test _save_geotiff and _validate_geotiff with dummy data
    da = _dummy_da()
    save_paths = downloader._save_geotiff(
        data=da,
        output_dir=tmp_path,
        basename="esacci_landcover_test"
    )
    for path in save_paths.values():
        assert Path(path).exists()
    validate_paths = downloader._validate_geotiff(
        output_dir=tmp_path,
        basename="esacci_landcover_test"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)


def test_ecira_downloader_full(tmp_path):
    """Custom test for ECIRADownloader: download (mocked), _save_geotiff, _validate_geotiff."""

    # Prepare dummy report for download
    dummy_report = [
        ItemDownloadReport(
            data_source="test_source",
            variable_name="test_variable",
            acquisition_time=datetime.datetime(2020, 1, 1),
            path=tmp_path / "ECIRA_202001.tif",
            download_successful=True,
            error=None,
            metadata=None,
        )
    ]

    with patch("drought_causality.downloaders.ecira.ECIRADownloader.download") as mock_download:
        mock_download.return_value = dummy_report
        downloader = ECIRADownloader(cache_dir=tmp_path)
        report = downloader.download(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        mock_download.assert_called_once_with(
            polygon=TEST_POLYGON,
            time_frame=(TEST_START_DATE, TEST_END_DATE),
            output_dir=tmp_path,
        )
        assert isinstance(report, list)
        assert all(isinstance(item, ItemDownloadReport) for item in report)
        assert all(item.download_successful for item in report)
        assert downloader.frequency == "yearly"

    # Now test _save_geotiff and _validate_geotiff with dummy data
    da = _dummy_da()
    save_paths = downloader._save_geotiff(
        data=da,
        output_dir=tmp_path,
        basename="ecira_test"
    )
    for path in save_paths.values():
        assert Path(path).exists()
    validate_paths = downloader._validate_geotiff(
        output_dir=tmp_path,
        basename="ecira_test"
    )
    assert all(validate_paths.values())
    assert len(validate_paths) == len(save_paths)

# @patch("drought_causality.downloaders.ESAWorldCoverDownloader.download")
# def test_esa_world_cover_downloader(mock_download, tmp_path: Path):
#     mock_da = _dummy_da()
#     mock_download.return_value = mock_da

#     downloader = ESAWorldCoverDownloader(cache_dir=tmp_path)
#     downloader.download(
#         polygon=TEST_POLYGON,
#         year=TEST_YEAR,  
#         target_res_deg=0.1
#         )
#     downloader.data = mock_da

#     save_paths = downloader.save_geotiff(
#         output_dir=tmp_path,
#         basename=f"esa_world_cover_test_{TEST_YEAR}_{TEST_MONTH:02d}"
#     )
#     mock_download.assert_called_once_with(
#         polygon=TEST_POLYGON, 
#         year=TEST_YEAR,
#         target_res_deg=0.1
#     )
#     for path in save_paths:
#         assert Path(path).exists()

#     validate_paths = downloader.validate_geotiff(
#         output_dir=tmp_path,
#         basename=f"esa_world_cover_test_{TEST_YEAR}_{TEST_MONTH:02d}"
#     )
#     assert all(validate_paths.values())
#     assert len(validate_paths) == len(save_paths)

# @patch("drought_causality.downloaders.IrrigationMapDownloader.download")
# def test_irrigation_map_downloader(mock_download, tmp_path: Path):
#     mock_da = _dummy_da()
#     mock_download.return_value = mock_da

#     downloader = IrrigationMapDownloader(cache_dir=tmp_path)
#     downloader.download(
#         polygon=TEST_POLYGON,
#         target_res_deg=0.1, 
#         )
#     downloader.data = mock_da

#     save_paths = downloader.save_geotiff(
#         output_dir=tmp_path,
#         basename=f"irrigation_map_test_{TEST_YEAR}_{TEST_MONTH:02d}"
#     )
#     mock_download.assert_called_once_with(polygon=TEST_POLYGON, target_res_deg=0.1)
#     for path in save_paths:
#         assert Path(path).exists()

#     validate_paths = downloader.validate_geotiff(
#         output_dir=tmp_path,
#         basename=f"irrigation_map_test_{TEST_YEAR}_{TEST_MONTH:02d}"
#     )
#     assert all(validate_paths.values())
#     assert len(validate_paths) == len(save_paths)
    