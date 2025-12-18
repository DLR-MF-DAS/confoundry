import numpy as np
import xarray as xr
import pandas as pd

import pytest

from drought_causality.create_timeseries_dataset import download_timeseries_data


# Global test variables for consistency
TEST_FIRST_YEAR = 2021
TEST_FIRST_MONTH = 7
TEST_FINAL_YEAR = 2021
TEST_FINAL_MONTH = 9
WORLD_COVER_YEAR = 2021
TARGET_RES_DEG = 0.1

@pytest.fixture
def dummy_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-123.15, 42.00],
                            [-123.15, 34.20],
                            [-113.84, 34.20],
                            [-113.84, 42.00],
                            [-123.15, 42.00]
                        ]
                    ]
                }
            }
        ]
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


def dummy_download(self, *args, **kwargs):
    cls_name = self.__class__.__name__
    if cls_name == "SPEIDownloader":
        self.data = _dummy_da()
    elif cls_name == "MODISNDVIDownloader":
        self.data = _dummy_da()
    elif cls_name == "ERA5Downloader":
        self.data = _dummy_era5_ds()
    elif cls_name == "ERA5PrecipDownloader":
        self.data = _dummy_era5_precip_ds()
    elif cls_name == "ERA5SoilMoistureDownloader":
        self.data = _dummy_soil_moisture_ds()
    elif cls_name == "ESAWorldCoverDownloader":
        self.data = _dummy_da()
    elif cls_name == "IrrigationMapDownloader":
        self.data = _dummy_da()
    else:
        raise RuntimeError(f"Unknown downloader class: {cls_name}")
    dummy_download.called = True


def test_download_timeseries_data_real_save(dummy_geojson, monkeypatch, tmp_path):
    # Patch __init__ and download for all downloaders for simplicity
    from drought_causality import downloaders
    for cls_name in [
        "SPEIDownloader",
        "MODISNDVIDownloader",
        "ERA5Downloader",
        "ERA5PrecipDownloader",
        "ERA5SoilMoistureDownloader",
        "ESAWorldCoverDownloader",
        "IrrigationMapDownloader"
    ]:
        cls = getattr(downloaders, cls_name)
        monkeypatch.setattr(cls, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(cls, "download", dummy_download)
    dummy_download.called = False

    # Run the function
    download_timeseries_data(
        location_geojson=dummy_geojson,
        location_nickname="mock_location",
        start_year=TEST_FIRST_YEAR,
        start_month=TEST_FIRST_MONTH,
        final_year=TEST_FINAL_YEAR,
        final_month=TEST_FINAL_MONTH,
        world_cover_year=WORLD_COVER_YEAR,
        target_res_deg=TARGET_RES_DEG,
        output_folder=str(tmp_path),
    )

    assert dummy_download.called

    # Check that expected files exist in the correct structure
    # Static files
    static_dir = tmp_path / "mock_location" / "static"
    assert (static_dir / f"worldcover_mock_location_{WORLD_COVER_YEAR}_{TARGET_RES_DEG}deg.tif").exists()
    assert (static_dir / f"gmia_irrigation_mock_location_{TARGET_RES_DEG}deg.tif").exists()

    # Time series files for each downloader and month
    for year in range(TEST_FIRST_YEAR, TEST_FINAL_YEAR + 1):
        for month in range(TEST_FIRST_MONTH, TEST_FINAL_MONTH + 1):
            month_dir = tmp_path / "mock_location" / str(year) / str(month)
            # For each month:
            assert (month_dir / f"era5_mock_location_{year}_{month:02d}_t2m.tif").exists()
            assert (month_dir / f"era5_mock_location_{year}_{month:02d}_ssrd.tif").exists()
            assert (month_dir / f"era5_precip_mock_location_{year}_{month:02d}.tif").exists()
            assert (month_dir / f"era5_soil_moisture_mock_location_{year}_{month:02d}_swvl1.tif").exists()
            assert (month_dir / f"spei_mock_location_{year}_{month:02d}.tif").exists()
            assert (month_dir / f"modis_ndvi_mock_location_{year}_{month:02d}.tif").exists()

    # Assert that the report CSV exists and has entries
    report_csv = tmp_path / "mock_location/download_report.csv"
    assert report_csv.exists(), f"Report CSV {report_csv} does not exist"
    
    # Load the report and check contents
    report_df = pd.read_csv(report_csv)
    assert not report_df.empty, "Report CSV is empty, expected at least one entry"

    # All entries should be 'success'
    assert (report_df['status'] == 'success').all(), "Not all report entries are marked as success"
    
    # All expected columns are present
    expected_columns = {'time', 'downloader', 'year', 'month', 'status', 'error'}
    assert expected_columns.issubset(report_df.columns), f"Missing columns in report: {expected_columns - set(report_df.columns)}"

    # No error messages
    assert report_df['error'].isnull().all() or (report_df['error'] == '').all(), "Some report entries have error messages"


def test_download_timeseries_data_with_failures(dummy_geojson, monkeypatch, tmp_path):
    """
    Simulate failures in some downloaders and check that the report CSV records failures.
    """
    from drought_causality import downloaders

    # Patch all downloaders to succeed except MODISNDVIDownloader and ERA5Downloader
    def dummy_download_or_fail(self, *args, **kwargs):
        cls_name = self.__class__.__name__
        if cls_name == "MODISNDVIDownloader":
            raise RuntimeError("Simulated MODIS NDVI failure!")
        elif cls_name == "ERA5Downloader":
            raise RuntimeError("Simulated ERA5 failure!")
        else:
            dummy_download(self, *args, **kwargs)
    
    for cls_name in [
        "SPEIDownloader",
        "MODISNDVIDownloader",
        "ERA5Downloader",
        "ERA5PrecipDownloader",
        "ERA5SoilMoistureDownloader",
        "ESAWorldCoverDownloader",
        "IrrigationMapDownloader"
    ]:
        cls = getattr(downloaders, cls_name)
        monkeypatch.setattr(cls, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(cls, "download", dummy_download_or_fail)
    dummy_download.called = False

    # Run the function
    download_timeseries_data(
        location_geojson=dummy_geojson,
        location_nickname="mock_location_fail",
        start_year=TEST_FIRST_YEAR,
        start_month=TEST_FIRST_MONTH,
        final_year=TEST_FINAL_YEAR,
        final_month=TEST_FINAL_MONTH,
        world_cover_year=WORLD_COVER_YEAR,
        target_res_deg=TARGET_RES_DEG,
        output_folder=str(tmp_path),
    )

    # Check the report CSV for failed entries
    report_csv = tmp_path / "mock_location_fail/download_report.csv"
    assert report_csv.exists(), f"Report CSV {report_csv} does not exist"
    report_df = pd.read_csv(report_csv)
    assert not report_df.empty, "Report CSV is empty, expected at least one entry"

    # MODISNDVIDownloader and ERA5Downloader should have failed entries
    failed = report_df[report_df['status'] == 'failed']
    assert not failed.empty, "Expected at least one failed entry in the report"
    assert any(failed['downloader'] == 'modis_ndvi'), "MODISNDVIDownloader failure not recorded"
    assert any(failed['downloader'] == 'era5'), "ERA5Downloader failure not recorded"
    # Error messages should be present
    assert failed['error'].notnull().all(), "Failed entries should have error messages"

    # All other downloaders should be marked as success
    succeeded = report_df[report_df['status'] == 'success']
    assert all(~succeeded['downloader'].isin(['modis_ndvi', 'era5'])), "Unexpected success for failed downloaders"

def test_redownload_on_corrupt_file(dummy_geojson, monkeypatch, tmp_path):
    """
    If a file exists but is corrupt (validation returns False),
    the pipeline should re-download and overwrite it.
    """
    # We'll test with SPEIDownloader as a representative example
    class DummyDownloader:
        def __init__(self, cache_dir=None):
            self.download_called = 0
            self.save_called = 0
        def download(self, *a, **k):
            self.download_called += 1
            self.data = _dummy_da()
        def save_geotiff(self, output_dir, basename):
            self.save_called += 1
            # Write a valid file
            path = output_dir / f"{basename}.tif"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"VALID")
            return [path]
        # Simulate: first call (corrupt), second call (valid)
        def check_geotiff_exists_and_validate(self, output_dir, basename):
            if not hasattr(self, "validate_calls"):
                self.validate_calls = 0
            self.validate_calls += 1
            # First call: file exists but is corrupt
            if self.validate_calls == 1:
                path = output_dir / f"{basename}.tif"
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"CORRUPT")
                return False
            # Second call: after save, file is valid
            return True
    # Patch SPEIDownloader in the correct DOWNLOADERS_MAP
    import drought_causality.create_timeseries_dataset as ctds
    monkeypatch.setitem(
        ctds.DOWNLOADERS_MAP, "spei", DummyDownloader
    )
    # Run the function for a single month
    download_timeseries_data(
        location_geojson=dummy_geojson,
        location_nickname="corrupt_test",
        downloaders=["spei"],
        start_year=2021,
        start_month=7,
        final_year=2021,
        final_month=7,
        output_folder=str(tmp_path),
    )
    # Check that the file was re-downloaded and overwritten
    outdir = tmp_path / "corrupt_test" / "2021" / "7"
    path = outdir / "spei_corrupt_test_2021_07.tif"
    assert path.exists(), "Output file does not exist after re-download"
    with open(path, "rb") as f:
        content = f.read()
    assert content == b"VALID", "Output file was not overwritten with valid content"
