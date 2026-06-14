import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.conditional_priors import (
    infer_bottom_temp_prior_c,
    infer_freezing_lst_prior,
    infer_ice_risk_prior,
    infer_thermal_regime,
)
from lake_pinn.data_io import load_training_frame


def _frame(air, lst, ice=None):
    dates = pd.date_range("2020-01-01", periods=len(air), freq="D")
    data = {
        "Date": dates,
        "T_air_C": air,
        "LST_surface_C": lst,
    }
    if ice is not None:
        data["ice_fraction"] = ice
    return pd.DataFrame(data)


def test_cold_ice_prone_bottom_prior_and_freezing_lst_gate():
    frame = _frame(
        air=[-8.0, -5.0, -2.0, 1.0],
        lst=[-3.0, -1.0, 0.5, 2.0],
        ice=[1.0, 0.8, 0.6, 0.2],
    )
    metadata = {"latitude": 46.0, "longitude": -89.0}

    assert infer_thermal_regime(metadata, frame) == "cold_ice_prone"
    assert np.isclose(infer_bottom_temp_prior_c(metadata, frame, max_depth=20.0), 4.0)
    assert infer_ice_risk_prior(metadata, frame).iloc[0] >= 0.5
    assert infer_freezing_lst_prior(metadata, frame).iloc[0]


def test_temperate_inference_uses_temperate_bottom_floor():
    frame = _frame(
        air=[4.0, 12.0, 20.0, 15.0],
        lst=[5.0, 14.0, 23.0, 17.0],
    )
    metadata = {"latitude": 42.0, "longitude": -74.0}

    assert infer_thermal_regime(metadata, frame) == "temperate"
    prior = infer_bottom_temp_prior_c(metadata, frame, max_depth=18.0)
    assert 4.0 <= prior <= 24.0


def test_warm_nonfreezing_does_not_trigger_freezing_rule():
    frame = _frame(
        air=[7.0, 10.0, 15.0, 18.0],
        lst=[12.0, 15.0, 19.0, 22.0],
    )
    metadata = {"latitude": 32.0, "longitude": 35.0}

    assert infer_thermal_regime(metadata, frame) == "warm_nonfreezing"
    assert not infer_freezing_lst_prior(metadata, frame).any()
    assert infer_bottom_temp_prior_c(metadata, frame, max_depth=30.0) > 10.0


def test_tropical_warm_bottom_prior_stays_warm():
    frame = _frame(
        air=[24.0, 25.0, 27.0, 26.0],
        lst=[27.0, 28.0, 30.0, 29.0],
    )
    metadata = {"latitude": 2.0, "longitude": 30.0}

    assert infer_thermal_regime(metadata, frame) == "tropical_warm"
    assert infer_bottom_temp_prior_c(metadata, frame, max_depth=40.0) >= 18.0
    assert not infer_freezing_lst_prior(metadata, frame).any()


def test_metadata_overrides_win():
    frame = _frame(
        air=[-10.0, -8.0, -5.0],
        lst=[-4.0, -2.0, 0.0],
        ice=[1.0, 1.0, 1.0],
    )
    metadata = {
        "thermal_regime": "tropical_warm",
        "bottom_temp_prior_c": 23.5,
        "latitude": 55.0,
        "longitude": 10.0,
    }

    assert infer_thermal_regime(metadata, frame) == "tropical_warm"
    assert infer_bottom_temp_prior_c(metadata, frame, max_depth=40.0) == 23.5
    assert not infer_freezing_lst_prior(metadata, frame).any()


def test_load_training_frame_uses_metadata_override_for_warm_lake_rules():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        era5_path = tmpdir / "era5_for_model.csv"
        lst_path = tmpdir / "lst_night_for_model.csv"
        pd.DataFrame(
            {
                "Date": dates,
                "t2m_C": [-5.0, -4.0, -3.0],
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
                "LST_surface_C": [-2.0, -1.0, 0.0],
            }
        ).to_csv(lst_path, index=False)
        (tmpdir / "metadata.json").write_text(
            '{"thermal_regime": "tropical_warm", "bottom_temp_prior_c": 23.5, '
            '"latitude": 2.0, "longitude": 30.0, "max_depth_m": 40.0}',
            encoding="utf-8",
        )

        frame, metadata = load_training_frame(era5_path, lst_path, data_fill_mode="reconstruction")

    assert metadata["thermal_regime"] == "tropical_warm"
    assert metadata["bottom_temp_prior_c"] == 23.5
    assert np.allclose(frame["BottomTemp_C"].to_numpy(), 23.5)
    assert frame["BottomTemp_imputed_by_4C_rule"].sum() == 0.0
    assert frame["LST_imputed_by_freezing_rule"].sum() == 0.0


if __name__ == "__main__":
    test_cold_ice_prone_bottom_prior_and_freezing_lst_gate()
    test_temperate_inference_uses_temperate_bottom_floor()
    test_warm_nonfreezing_does_not_trigger_freezing_rule()
    test_tropical_warm_bottom_prior_stays_warm()
    test_metadata_overrides_win()
    test_load_training_frame_uses_metadata_override_for_warm_lake_rules()
    print("conditional prior sanity checks passed")
