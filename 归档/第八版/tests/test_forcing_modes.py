import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.data_io import load_training_frame, normalize_task_mode
from lake_pinn.forcing import suppress_spring_lst_warm_spikes


def test_spring_lst_spike_suppression_rejects_removed_forecast_fill_mode():
    lst = np.array([np.nan, 20.0, 40.0, 10.0], dtype=np.float64)
    air = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    doy = np.array([90.0, 91.0, 92.0, 93.0], dtype=np.float64)

    with pytest.raises(ValueError, match="forecast fill mode was removed"):
        suppress_spring_lst_warm_spikes(
            lst_surface_c=lst,
            air_temp_c=air,
            doy=doy,
            data_fill_mode="forecast",
        )


def test_task_mode_aliases_default_to_reconstruction_only():
    assert normalize_task_mode(None) == "analysis"
    assert normalize_task_mode("reconstruction") == "analysis"
    assert normalize_task_mode("reconstruct") == "analysis"
    assert normalize_task_mode("hindcast") == "analysis"
    with pytest.raises(ValueError, match="forecast/nowcast modes were removed"):
        normalize_task_mode("forecast")
    with pytest.raises(ValueError, match="forecast/nowcast modes were removed"):
        normalize_task_mode("strict_forecast")


def test_spring_lst_spike_suppression_reconstruction_can_use_both_sides():
    lst = np.array([np.nan, 20.0, 22.0], dtype=np.float64)
    air = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    doy = np.array([90.0, 91.0, 92.0], dtype=np.float64)

    corrected = suppress_spring_lst_warm_spikes(
        lst_surface_c=lst,
        air_temp_c=air,
        doy=doy,
        data_fill_mode="reconstruction",
        max_above_air_c=100.0,
        max_above_rolling_median_c=100.0,
        max_daily_warming_c=100.0,
    )

    assert pd.notna(corrected[0])
    assert corrected[0] == 20.0


def test_load_training_frame_prefers_shortwave_wm2_over_lake_is_energy():
    dates = pd.date_range("2019-07-01", periods=3, freq="D")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        era5_path = tmpdir / "era5.csv"
        lst_path = tmpdir / "lst.csv"
        pd.DataFrame(
            {
                "Date": dates,
                "t2m_C": [20.0, 21.0, 22.0],
                "wind_norm_m_per_s": [3.0, 3.0, 3.0],
                "ssrd_W_per_m2": [250.0, 260.0, 270.0],
                "Is_J_per_m2": [1000.0, 1000.0, 1000.0],
                "strd_W_per_m2": [360.0, 360.0, 360.0],
                "latent_heat_upward_W_per_m2": [80.0, 80.0, 80.0],
                "sensible_heat_upward_W_per_m2": [20.0, 20.0, 20.0],
                "rh_percent": [70.0, 70.0, 70.0],
                "sp_Pa": [101325.0, 101325.0, 101325.0],
            }
        ).to_csv(era5_path, index=False)
        pd.DataFrame(
            {
                "Date": dates,
                "LST_surface_C": [22.0, 23.0, 24.0],
            }
        ).to_csv(lst_path, index=False)

        frame, _ = load_training_frame(era5_path, lst_path, data_fill_mode="reconstruction")

    assert np.allclose(frame["Solar_W_m2"].to_numpy(), np.array([250.0, 260.0, 270.0]))


def test_load_training_frame_splits_open_water_and_ice_lst():
    dates = pd.date_range("2019-01-01", periods=3, freq="D")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        era5_path = tmpdir / "era5.csv"
        lst_path = tmpdir / "lst.csv"
        pd.DataFrame(
            {
                "Date": dates,
                "t2m_C": [-5.0, 2.0, 3.0],
                "wind_norm_m_per_s": [3.0, 3.0, 3.0],
                "ssrd_W_per_m2": [50.0, 60.0, 70.0],
                "strd_W_per_m2": [260.0, 280.0, 300.0],
                "latent_heat_upward_W_per_m2": [10.0, 10.0, 10.0],
                "sensible_heat_upward_W_per_m2": [5.0, 5.0, 5.0],
                "rh_percent": [80.0, 80.0, 80.0],
                "sp_Pa": [101325.0, 101325.0, 101325.0],
            }
        ).to_csv(era5_path, index=False)
        pd.DataFrame(
            {
                "Date": dates,
                "LST_surface_C": [-8.0, np.nan, 5.0],
                "ice_fraction": [1.0, 0.0, 0.0],
            }
        ).to_csv(lst_path, index=False)

        frame, _ = load_training_frame(era5_path, lst_path, data_fill_mode="reconstruction")

    assert frame.loc[0, "ice_mask"] == 1.0
    assert pd.isna(frame.loc[0, "LSWT_open_water_C"])
    assert frame.loc[0, "IST_snow_ice_C"] == -8.0
    assert frame.loc[1, "ice_mask"] == 0.0
    assert pd.isna(frame.loc[1, "LSWT_open_water_C"])
    assert np.isfinite(frame.loc[1, "LST_surface_C"])
    assert "LST_filled_C" in frame.columns
    assert "ice_thickness_m" in frame.columns
    assert "snow_depth_m" in frame.columns


if __name__ == "__main__":
    test_task_mode_aliases_default_to_reconstruction_only()
    test_spring_lst_spike_suppression_rejects_removed_forecast_fill_mode()
    test_spring_lst_spike_suppression_reconstruction_can_use_both_sides()
    test_load_training_frame_prefers_shortwave_wm2_over_lake_is_energy()
    test_load_training_frame_splits_open_water_and_ice_lst()
    print("forcing mode sanity checks passed")
