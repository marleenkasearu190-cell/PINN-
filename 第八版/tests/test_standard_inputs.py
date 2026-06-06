import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.standard_inputs import main as standard_inputs_main
from lake_pinn.state_multilake import train_multilake_state_forecaster


def test_standard_inputs_generates_state_multilake_manifest_with_decimal_depths(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    era5_path = tmp_path / "era5.csv"
    lst_path = tmp_path / "lst.csv"
    profile_path = tmp_path / "profile.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "Date": dates,
            "t2m_C": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
            "wind_norm_m_per_s": [2.0] * len(dates),
            "ssrd_W_per_m2": [120.0] * len(dates),
            "strd_W_per_m2": [320.0] * len(dates),
            "latent_heat_upward_W_per_m2": [20.0] * len(dates),
            "sensible_heat_upward_W_per_m2": [8.0] * len(dates),
            "rh_percent": [70.0] * len(dates),
            "sp_Pa": [101325.0] * len(dates),
            "net_inflow_m3_s": [1.0] * len(dates),
        }
    ).to_csv(era5_path, index=False)
    pd.DataFrame(
        {
            "Date": dates,
            "LST_surface_C": [6.0, 6.1, 6.2, 6.3, 6.4, 6.5],
            "LST_is_filled": [0, 0, 1, 0, 0, 0],
            "LST_qc_good_fraction": [1.0, 0.8, 0.2, 1.0, 1.0, 1.0],
            "ice_fraction": [0.0] * len(dates),
        }
    ).to_csv(lst_path, index=False)
    pd.DataFrame(
        {
            "Date": dates[:3],
            "Temp_0.5m": [6.0, 6.1, 6.2],
            "Temp_1.5m": [5.8, 5.9, 6.0],
        }
    ).to_csv(profile_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "standard_inputs",
            "--lake-id",
            "demo_lake_2020",
            "--era5",
            str(era5_path),
            "--lst",
            str(lst_path),
            "--profile-obs",
            str(profile_path),
            "--out-dir",
            str(out_dir),
            "--max-depth",
            "2.0",
            "--mean-depth",
            "1.0",
            "--area-km2",
            "0.5",
            "--latitude",
            "45.0",
            "--longitude",
            "-90.0",
        ],
    )
    standard_inputs_main()

    manifest_path = out_dir / "demo_lake_2020_state_multilake_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "forcing" not in manifest
    assert "data_fill_mode" not in manifest
    lake = manifest["lakes"][0]
    assert {"era5", "lst", "profile_obs", "metadata", "max_depth"}.issubset(lake)
    assert "forcing" not in lake
    assert "data_fill_mode" not in lake

    profile_long = pd.read_csv(lake["profile_obs"])
    assert set(profile_long["Depth_m"].round(3)) == {0.5, 1.5}
    lst_model = pd.read_csv(lake["lst"])
    assert {"LST_is_filled", "LST_observed_flag", "LST_quality_factor", "ice_fraction"}.issubset(lst_model.columns)

    result = train_multilake_state_forecaster(
        manifest_path,
        out_dir / "train_read",
        epochs=0,
        depth_points=4,
        history_window_days=2,
        segment_rollout_loss_weight=0.05,
        device="cpu",
    )
    assert result["split_summary"].exists()
