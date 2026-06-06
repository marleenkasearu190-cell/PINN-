import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.state_reconstruction import _profile_lookup
from lake_pinn.diagnostics import write_density_stability_summary, write_heat_closure_summaries
from lake_pinn.state_model import ForcingBatch, LakeStateForecaster
from lake_pinn.state_multilake import (
    DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    _heat_content_transition_effective_weight,
    _heat_content_transition_loss,
    _heat_content_transition_loss_vector,
    _forcing_row_batch,
    _normalize_heat_content_transition_depth_factor,
    _parse_heat_content_transition_season_factors,
    _resolve_heat_content_transition_season_factors,
    _long_rollout_sequence_loss,
    _long_rollout_sequence_losses_for_lake,
    _long_rollout_sequence_losses_for_lakes_cross_batch,
    _transition_loss,
    _transition_losses_for_lake,
    _transition_losses_for_lakes_cross_batch,
    evaluate_heldout_free_rolls,
    evaluate_lake_free_roll,
    evaluate_lake_rolling_start_horizons,
    evaluate_lake_rolling_start_horizons_batched,
    prepare_lake_state_data,
    train_multilake_state_forecaster,
)


def _write_lake_inputs(
    root: Path,
    lake_id: str,
    offset: float,
    latitude: float | None = 45.0,
    lake_group: str | None = None,
):
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    era5 = root / f"{lake_id}_era5.csv"
    lst = root / f"{lake_id}_lst.csv"
    profile = root / f"{lake_id}_profile.csv"
    meta = root / f"{lake_id}_metadata.json"
    pd.DataFrame(
        {
            "Date": dates,
            "t2m_C": [5.0 + offset + idx * 0.2 for idx in range(len(dates))],
            "wind_norm_m_per_s": [2.5] * len(dates),
            "ssrd_W_per_m2": [120.0 + 5.0 * idx for idx in range(len(dates))],
            "strd_W_per_m2": [320.0] * len(dates),
            "latent_heat_upward_W_per_m2": [20.0] * len(dates),
            "sensible_heat_upward_W_per_m2": [8.0] * len(dates),
            "rh_percent": [70.0] * len(dates),
            "sp_Pa": [101325.0] * len(dates),
        }
    ).to_csv(era5, index=False)
    pd.DataFrame(
        {
            "Date": dates,
            "LST_surface_C": [6.0 + offset + idx * 0.2 for idx in range(len(dates))],
            "LST_qc_good_fraction": [1.0] * len(dates),
        }
    ).to_csv(lst, index=False)
    profile_rows = []
    for day in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
        for depth in [0.0, 4.0, 8.0]:
            profile_rows.append(
                {
                    "Date": dates[day],
                    "Depth_m": depth,
                    "Temperature_C": 6.0 + offset + day * 0.15 - depth * 0.2,
                }
            )
    pd.DataFrame(profile_rows).to_csv(profile, index=False)
    meta.write_text(
        json.dumps(
            {
                "lake_id": lake_id,
                "max_depth_m": 8.0,
                "mean_depth_m": 4.0,
                "area_km2": 1.0,
                "latitude": latitude,
                "longitude": -90.0,
                "secchi_m": 2.0,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "lake_id": lake_id,
        "era5": str(era5),
        "lst": str(lst),
        "profile_obs": str(profile),
        "metadata": str(meta),
        "max_depth": 8.0,
    }
    if lake_group is not None:
        config["lake_group"] = lake_group
    return config


def _single_lake_and_model(root: Path):
    torch.manual_seed(123)
    lake_config = _write_lake_inputs(root, "lake_a", 0.0)
    lake = prepare_lake_state_data(
        lake_config,
        split_mode="none",
        depth_points=5,
        max_rollout_days=5,
        long_free_roll_max_days=5,
        history_window_days=3,
        device="cpu",
    )
    lake["heat_content_transition_season_factors"] = DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS
    model = LakeStateForecaster(
        lake["depths"],
        lake["area"],
        hidden_dim=16,
        forcing_context_dim=8,
        forcing_history_hidden_dim=8,
        residual_limit_c=0.25,
        turbulent_flux_mode="bulk",
    )
    model.eval()
    return lake, model


def _compatible_lakes_and_model(root: Path):
    torch.manual_seed(123)
    lakes = []
    for lake_id, offset in [("lake_a", 0.0), ("lake_b", 1.0)]:
        lake = prepare_lake_state_data(
            _write_lake_inputs(root, lake_id, offset),
            split_mode="none",
            depth_points=5,
            max_rollout_days=5,
            long_free_roll_max_days=5,
            history_window_days=3,
            device="cpu",
        )
        lake["heat_content_transition_season_factors"] = DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS
        lakes.append(lake)
    model = LakeStateForecaster(
        lakes[0]["depths"],
        lakes[0]["area"],
        hidden_dim=16,
        forcing_context_dim=8,
        forcing_history_hidden_dim=8,
        residual_limit_c=0.25,
        turbulent_flux_mode="bulk",
    )
    model.eval()
    return lakes, model


def test_multilake_state_training_excludes_heldout_lake_from_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output_dir = root / "out"
        result = train_multilake_state_forecaster(
            manifest_path,
            output_dir,
            epochs=1,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            test_lake_id="lake_b",
            residual_regularization_weight=0.02,
            daily_tendency_weight=0.02,
            free_roll_loss_weight=0.05,
            free_roll_supervision_mode="observed",
            long_free_roll_loss_weight=0.05,
            long_free_roll_start_epoch=0,
            long_free_roll_ramp_epochs=1,
            long_free_roll_max_days=5,
            long_free_roll_samples_per_lake=0,
            teacher_forcing_start=0.5,
            teacher_forcing_end=0.0,
            state_noise_weight=0.0,
            residual_time_smooth_weight=0.01,
            checkpoint_every_epochs=1,
            eval_every_epochs=1,
            profile_runtime=True,
            profile_gpu=True,
            export_after_training="on",
            device="cpu",
        )

        assert result["checkpoint_path"].exists()
        assert (output_dir / "global_state_forecaster_epoch0000.pt").exists()
        assert (output_dir / "global_state_forecaster_training_history_partial.csv").exists()
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert "lake_b" not in bundle["train_lake_ids"]
        assert bundle["heldout_lake_ids"] == ["lake_b"]
        assert bundle["test_lake_ids"] == ["lake_b"]
        assert bundle["heldout_lake_groups"] == ["lake_b"]
        assert bundle["excluded_lake_ids"] == ["lake_b"]
        assert bundle["residual_regularization_weight"] == 0.02
        assert bundle["daily_tendency_weight"] == 0.02
        assert bundle["physical_scale_regularization_weight"] == 0.01
        assert bundle["physical_scale_smoothness_weight"] == 0.005
        assert bundle["heat_content_transition_weight"] == DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT
        assert bundle["heat_content_transition_weight_base"] == DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT
        assert bundle["heat_content_full_column_min_coverage"] == DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE
        assert bundle["heat_content_transition_effective_max"] == DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX
        assert bundle["heat_content_transition_season_mode"] == "auto"
        assert bundle["heat_content_transition_season_factors_override"] is None
        assert bundle["heat_content_transition_depth_factor"] == "on"
        assert bundle["heat_content_transition_lake_configs"]["lake_a"]["resolved_mode"] == "northern"
        assert bundle["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["10"] == 2.0
        assert bundle["transition_loss_weight"] == 1.0
        assert bundle["hard_density_stability"] == "auto"
        assert bundle["hard_density_stability_active"] is True
        assert bundle["turbulent_flux_mode"] == "bulk"
        assert bundle["turbulent_flux_blend_alpha"] == 0.3
        assert bundle["checkpoint_every_epochs"] == 1
        assert bundle["eval_every_epochs"] == 1
        assert bundle["light_eval_every_epochs"] == 1
        assert bundle["full_eval_every_epochs"] == 100
        assert bundle["profile_runtime"] is True
        assert bundle["profile_gpu"] is True
        assert bundle["transition_batch_mode"] == "off"
        assert bundle["segment_rollout_batch_mode"] == "off"
        assert bundle["transition_batch_size"] == 0
        assert bundle["segment_rollout_batch_size"] == 0
        assert bundle["rolling_horizon_batch_mode"] == "off"
        assert bundle["rolling_horizon_batch_size"] == 32
        assert bundle["full_free_roll_batch_mode"] == "off"
        assert bundle["full_free_roll_batch_size"] == 16
        assert bundle["step_forcing_mode"] == "auto"
        assert bundle["diagnostic_mode"] == "auto"
        assert bundle["train_diagnostic_mode"] == "loss"
        assert bundle["rollout_batch_step_mode"] == "off"
        assert bundle["export_after_training"] == "on"
        assert bundle["cross_lake_batch_mode"] == "off"
        assert bundle["cross_lake_batch_size"] == 0
        assert bundle["free_roll_loss_weight"] == 0.05
        assert bundle["free_roll_supervision_mode"] == "observed"
        assert bundle["long_free_roll_loss_weight"] == 0.05
        assert bundle["long_free_roll_start_epoch"] == 0
        assert bundle["long_free_roll_max_days"] == 5
        assert bundle["segment_rollout_loss_weight"] == 0.05
        assert bundle["teacher_forcing_start"] == 0.5
        assert bundle["teacher_forcing_end"] == 0.0
        assert bundle["state_noise_weight"] == 0.0
        assert bundle["residual_time_smooth_weight"] == 0.01
        history = pd.read_csv(result["history_csv"])
        assert "heldout_transition_mean_rmse" in history.columns
        assert "transition_loss_weight" in history.columns
        assert "transition_loss_unweighted" in history.columns
        assert "transition_loss_weighted" in history.columns
        assert float(history["transition_loss_weight"].iloc[-1]) == 1.0
        assert "heldout_free_roll_mean_rmse" in history.columns
        assert "heldout_persistence_mean_rmse" in history.columns
        assert "free_roll_loss" in history.columns
        assert "free_roll_supervision_count" in history.columns
        assert "long_free_roll_loss" in history.columns
        assert "long_free_roll_supervision_count" in history.columns
        assert "long_free_roll_sequence_count" in history.columns
        assert "long_free_roll_weight_eff" in history.columns
        assert "long_free_roll_active_days" in history.columns
        assert "segment_rollout_loss" in history.columns
        assert "segment_supervision_count" in history.columns
        assert "teacher_forcing_probability" in history.columns
        assert "physical_scale_reg_loss" in history.columns
        assert "physical_scale_smooth_loss" in history.columns
        assert "heat_content_transition_loss" in history.columns
        assert "heat_content_transition_weight_base" in history.columns
        assert "heat_content_transition_weighted_loss" in history.columns
        assert "heat_content_transition_effective_weight_mean" in history.columns
        assert "heat_content_transition_effective_weight_min" in history.columns
        assert "heat_content_transition_effective_weight_max" in history.columns
        assert "heat_content_full_column_min_coverage" in history.columns
        assert "heat_content_transition_season_factor_10" in history.columns
        assert "heat_content_transition_season_factor_min_10" in history.columns
        assert "heat_content_transition_season_factor_max_10" in history.columns
        assert "long_free_roll_heat_content_transition_weighted_loss" in history.columns
        assert "long_free_roll_heat_content_transition_effective_weight_mean" in history.columns
        assert "shortwave_scale_mean" in history.columns
        assert "cooling_scale_mean" in history.columns
        assert "surface_flux_bias_mean_wm2" in history.columns
        assert "turbulent_flux_mode" in history.columns
        assert "turbulent_flux_blend_alpha" in history.columns
        assert "checkpoint_every_epochs" in history.columns
        assert "eval_every_epochs" in history.columns
        assert "light_eval_every_epochs" in history.columns
        assert "full_eval_every_epochs" in history.columns
        assert "eval_mode" in history.columns
        assert "profile_runtime" in history.columns
        assert "profile_gpu" in history.columns
        assert "transition_batch_mode" in history.columns
        assert "segment_rollout_batch_mode" in history.columns
        assert "transition_batch_size" in history.columns
        assert "segment_rollout_batch_size" in history.columns
        assert "rolling_horizon_batch_mode" in history.columns
        assert "rolling_horizon_batch_size" in history.columns
        assert "full_free_roll_batch_mode" in history.columns
        assert "full_free_roll_batch_size" in history.columns
        assert "step_forcing_mode" in history.columns
        assert "diagnostic_mode" in history.columns
        assert "train_diagnostic_mode" in history.columns
        assert "rollout_batch_step_mode" in history.columns
        assert "export_after_training" in history.columns
        assert "cross_lake_batch_mode" in history.columns
        assert "cross_lake_batch_size" in history.columns
        assert "transition_seconds" in history.columns
        assert "segment_seconds" in history.columns
        assert "evaluation_seconds" in history.columns
        assert "epoch_seconds" in history.columns
        assert "open_water_sensible_heat_mean_wm2" in history.columns
        assert "open_water_latent_heat_mean_wm2" in history.columns
        assert "temperature_floor_heat_injection_mean_wm2" in history.columns
        assert "heldout_transition_rmse_3d" in history.columns
        assert "val_rolling_start_rmse_3d" in history.columns
        assert "val_rolling_start_bias_3d" in history.columns
        assert "val_rolling_start_count_3d" in history.columns
        assert "heldout_free_roll_rmse_3d" in history.columns
        assert "heldout_free_roll_bias_3d" in history.columns
        assert "heldout_free_roll_count_3d" in history.columns
        assert "heldout_initial_free_roll_rmse_3d" in history.columns
        assert "heldout_initial_free_roll_bias_3d" in history.columns
        assert "heldout_initial_free_roll_count_3d" in history.columns
        assert "heldout_rolling_start_rmse_3d" in history.columns
        assert "heldout_rolling_start_bias_3d" in history.columns
        assert "heldout_rolling_start_count_3d" in history.columns
        assert (
            float(history["heldout_free_roll_rmse_3d"].iloc[-1])
            == float(history["heldout_rolling_start_rmse_3d"].iloc[-1])
        )
        assert (
            float(history["heldout_free_roll_count_3d"].fillna(0.0).iloc[-1])
            >= float(history["heldout_initial_free_roll_count_3d"].fillna(0.0).iloc[-1])
        )
        assert float(history["free_roll_supervision_count"].fillna(0.0).max()) > 0.0
        assert float(history["long_free_roll_supervision_count"].fillna(0.0).max()) > 0.0
        assert float(history["long_free_roll_sequence_count"].fillna(0.0).max()) > 0.0
        assert float(history["segment_supervision_count"].fillna(0.0).max()) > 0.0
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["lake_a"]["train_long_rollout_sequences"] > 0
        assert split_summary["_config"]["test_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["heldout_lake_groups"] == ["lake_b"]
        assert split_summary["_config"]["train_lake_ids"] == ["lake_a"]
        assert split_summary["_config"]["heldout_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["excluded_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["transition_loss_weight"] == 1.0
        assert split_summary["_config"]["heat_content_transition_weight"] == DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT
        assert split_summary["_config"]["heat_content_transition_weight_base"] == DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT
        assert (
            split_summary["_config"]["heat_content_full_column_min_coverage"]
            == DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE
        )
        assert split_summary["_config"]["heat_content_transition_effective_max"] == DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX
        assert split_summary["_config"]["heat_content_transition_season_mode"] == "auto"
        assert split_summary["_config"]["heat_content_transition_season_factors_override"] is None
        assert split_summary["_config"]["heat_content_transition_depth_factor"] == "on"
        assert split_summary["_config"]["heat_content_transition_lake_configs"]["lake_a"]["resolved_mode"] == "northern"
        assert split_summary["_config"]["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["9"] == 2.0
        assert split_summary["lake_a"]["heat_content_transition"]["resolved_mode"] == "northern"
        assert split_summary["_config"]["hard_density_stability"] == "auto"
        assert split_summary["_config"]["hard_density_stability_active"] is True
        assert split_summary["_config"]["turbulent_flux_mode"] == "bulk"
        assert split_summary["_config"]["turbulent_flux_blend_alpha"] == 0.3
        assert split_summary["_config"]["checkpoint_every_epochs"] == 1
        assert split_summary["_config"]["eval_every_epochs"] == 1
        assert split_summary["_config"]["light_eval_every_epochs"] == 1
        assert split_summary["_config"]["full_eval_every_epochs"] == 100
        assert split_summary["_config"]["profile_runtime"] is True
        assert split_summary["_config"]["profile_gpu"] is True
        assert split_summary["_config"]["step_forcing_mode"] == "auto"
        assert split_summary["_config"]["diagnostic_mode"] == "auto"
        assert split_summary["_config"]["train_diagnostic_mode"] == "loss"
        assert split_summary["_config"]["rollout_batch_step_mode"] == "off"
        assert split_summary["_config"]["export_after_training"] == "on"
        assert split_summary["_config"]["cross_lake_batch_mode"] == "off"
        assert split_summary["_config"]["cross_lake_batch_size"] == 0
        assert split_summary["_config"]["transition_batch_mode"] == "off"
        assert split_summary["_config"]["segment_rollout_batch_mode"] == "off"
        assert split_summary["_config"]["transition_batch_size"] == 0
        assert split_summary["_config"]["segment_rollout_batch_size"] == 0
        assert split_summary["_config"]["rolling_horizon_batch_mode"] == "off"
        assert split_summary["_config"]["rolling_horizon_batch_size"] == 32
        assert split_summary["_config"]["full_free_roll_batch_mode"] == "off"
        assert split_summary["_config"]["full_free_roll_batch_size"] == 16
        export_info = result["heldout_exports"][0]
        assert export_info["heat_closure_monthly_summary"].exists()
        assert export_info["heat_closure_annual_summary"].exists()
        assert export_info["density_stability_summary"].exists()
        diagnostics = pd.read_csv(export_info["diagnostics_csv"])
        assert "turbulent_flux_mode" in diagnostics.columns
        assert "open_water_sensible_heat_wm2" in diagnostics.columns
        assert "open_water_latent_heat_wm2" in diagnostics.columns
        assert "temperature_floor_heat_injection_wm2" in diagnostics.columns
        assert "density_adjustment_applied" in diagnostics.columns
        assert "density_adjustment_max_delta_c" in diagnostics.columns
        assert "density_adjustment_heat_delta_j_m2" in diagnostics.columns
        runtime_profile = pd.read_csv(result["runtime_profile_csv"])
        assert "epoch_seconds" in runtime_profile.columns
        assert "gpu_util_percent" in runtime_profile.columns


def test_multi_heldout_ids_infer_groups_exclude_group_mates_and_export_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "train_lake_a_2020", 0.0, lake_group="train_lake_a"),
                _write_lake_inputs(root, "train_lake_d_2020", 0.5, lake_group="train_lake_d"),
                _write_lake_inputs(root, "held_b_2020", 1.0, lake_group="held_b"),
                _write_lake_inputs(root, "held_b_2021", 1.5, lake_group="held_b"),
                _write_lake_inputs(root, "held_c_2020", 2.0, lake_group="held_c"),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        train_dir = root / "train"
        result = train_multilake_state_forecaster(
            manifest_path,
            train_dir,
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_ids=["held_b_2021", "held_c_2020"],
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["test_lake_id"] == "held_b_2021"
        assert bundle["test_lake_ids"] == ["held_b_2021", "held_c_2020"]
        assert bundle["heldout_lake_groups"] == ["held_b", "held_c"]
        assert bundle["heldout_lake_ids"] == ["held_b_2021", "held_c_2020"]
        assert set(bundle["train_lake_ids"]) == {"train_lake_a_2020", "train_lake_d_2020"}
        assert set(bundle["excluded_lake_ids"]) == {"held_b_2020", "held_b_2021", "held_c_2020"}

        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["held_b_2020"]["is_excluded_by_heldout_group"] is True
        assert split_summary["held_b_2020"]["is_heldout_test_lake"] is False
        assert split_summary["held_b_2021"]["is_heldout_test_lake"] is True
        assert split_summary["held_c_2020"]["is_heldout_test_lake"] is True
        assert split_summary["_config"]["test_lake_ids"] == ["held_b_2021", "held_c_2020"]
        assert split_summary["_config"]["heldout_lake_groups"] == ["held_b", "held_c"]

        export_dir = root / "export"
        export_result = train_multilake_state_forecaster(
            manifest_path,
            export_dir,
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_ids="held_b_2021,held_c_2020",
            checkpoint_path=result["checkpoint_path"],
            export_only=True,
            device="cpu",
        )
        exported_ids = {info["lake_id"] for info in export_result["heldout_exports"]}
        assert exported_ids == {"held_b_2021", "held_c_2020"}
        assert (export_dir / "held_b_2021_heldout_state_reconstruction_year_heatmap.png").exists()
        assert (export_dir / "held_c_2020_heldout_state_reconstruction_year_heatmap.png").exists()


def test_export_after_training_defaults_to_off():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output_dir = root / "out"
        result = train_multilake_state_forecaster(
            manifest_path,
            output_dir,
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            device="cpu",
        )

        assert result["heldout_exports"] == []
        assert not list(output_dir.glob("*_year_heatmap.png"))
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["export_after_training"] == "off"
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["_config"]["export_after_training"] == "off"


def test_cross_lake_batch_mode_on_trains_compatible_grid_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "cross_lake_batch_mode": "on",
            "transition_batch_mode": "on",
            "segment_rollout_batch_mode": "on",
            "cross_lake_batch_size": 4,
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
                _write_lake_inputs(root, "lake_c", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=1,
            depth_points=5,
            max_rollout_days=3,
            long_free_roll_start_epoch=0,
            long_free_roll_ramp_epochs=1,
            segment_rollout_loss_weight=0.01,
            segment_rollout_max_days=3,
            long_free_roll_samples_per_lake=1,
            teacher_forcing_start=0.0,
            teacher_forcing_end=0.0,
            state_noise_weight=0.0,
            history_window_days=3,
            test_lake_id="lake_c",
            device="cpu",
        )

        history = pd.read_csv(result["history_csv"])
        assert history["cross_lake_batch_mode"].iloc[-1] == "on"
        assert history["cross_lake_batch_size"].iloc[-1] == 4
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["cross_lake_batch_mode"] == "on"
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["_config"]["cross_lake_batch_mode"] == "on"
        assert split_summary["_config"]["cross_lake_batch_size"] == 4


def test_periodic_checkpoint_resume_continues_history():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        first_dir = root / "first"
        train_multilake_state_forecaster(
            manifest_path,
            first_dir,
            epochs=1,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            checkpoint_every_epochs=1,
            eval_every_epochs=1,
            profile_runtime=True,
            device="cpu",
        )
        resume_checkpoint = first_dir / "global_state_forecaster_epoch0000.pt"
        assert resume_checkpoint.exists()
        resumed_dir = root / "resumed"
        resumed = train_multilake_state_forecaster(
            manifest_path,
            resumed_dir,
            epochs=2,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            resume_checkpoint=resume_checkpoint,
            checkpoint_every_epochs=1,
            eval_every_epochs=1,
            profile_runtime=True,
            device="cpu",
        )
        history = pd.read_csv(resumed["history_csv"])
        assert list(history["epoch"].astype(int)) == [0, 1]
        assert (resumed_dir / "global_state_forecaster_epoch0001.pt").exists()
        assert "epoch_seconds" in history.columns
        assert resumed["checkpoint_path"].exists()


def test_mini_light_full_eval_modes_gate_heavy_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=4,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            eval_every_epochs=1,
            light_eval_every_epochs=2,
            full_eval_every_epochs=4,
            rolling_horizon_eval_max_starts=4,
            device="cpu",
        )
        history = pd.read_csv(result["history_csv"])
        assert history["eval_mode"].tolist() == ["mini", "light", "mini", "full"]
        assert np.isfinite(history["heldout_transition_mean_rmse"]).all()
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[0])
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[1])
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[2])
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[3])
        assert np.isnan(history["heldout_transition_rmse_1d"].iloc[0])
        assert np.isfinite(history["heldout_transition_rmse_1d"].iloc[1])
        assert np.isnan(history["heldout_transition_rmse_1d"].iloc[2])
        assert np.isfinite(history["heldout_transition_rmse_1d"].iloc[3])
        assert np.isnan(history["heldout_persistence_mean_rmse"].iloc[0])
        assert np.isfinite(history["heldout_persistence_mean_rmse"].iloc[1])
        assert np.isnan(history["lake_a_train_rmse"].iloc[0])
        assert np.isfinite(history["lake_a_train_rmse"].iloc[1])
        assert np.isnan(history["val_rolling_start_rmse_1d"].iloc[1])
        assert np.isfinite(history["heldout_rolling_start_rmse_1d"].iloc[3])
        assert np.isfinite(history["heldout_free_roll_rmse_1d"].iloc[3])
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["light_eval_every_epochs"] == 2
        assert bundle["full_eval_every_epochs"] == 4


def test_batched_transition_loss_matches_scalar_loop():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        pairs = [pair for pair in lake["pairs"]["train"] if int(pair[3] - pair[2]) == 1][:3]
        assert len(pairs) >= 2
        kwargs = {
            "profile_huber_delta": 2.0,
            "lst_surface_weight": 0.03,
            "energy_balance_weight": 0.001,
            "heat_content_transition_weight": DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
            "heat_content_full_column_min_coverage": DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
            "heat_content_transition_season_factors": DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
            "heat_content_transition_depth_factor": True,
            "heat_content_transition_effective_max": DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
            "residual_regularization_weight": 0.02,
            "daily_tendency_weight": 0.02,
            "free_roll_loss_weight": 0.0,
            "physical_scale_regularization_weight": 0.01,
            "physical_scale_smoothness_weight": 0.005,
            "task_mode": "analysis",
            "hard_density_stability": True,
        }
        scalar_losses = [_transition_loss(model, lake, pair, **kwargs)[0] for pair in pairs]
        batched_losses, batched_details = _transition_losses_for_lake(
            model,
            lake,
            pairs,
            transition_batch_mode="on",
            transition_batch_size=0,
            **kwargs,
        )
        assert len(batched_losses) == len(scalar_losses)
        assert len(batched_details) == len(scalar_losses)
        assert torch.allclose(torch.stack(scalar_losses), torch.stack(batched_losses), atol=1e-5, rtol=1e-5)


def test_batched_segment_rollout_loss_matches_scalar_loop():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        active_max_days = 5
        groups = {}
        for sequence in lake["long_rollout_sequences"]["train"]:
            start, start_idx, targets = sequence
            usable = [
                (target, target_idx)
                for target, target_idx in targets
                if 1 <= int(target_idx - start_idx) <= active_max_days
            ]
            if usable:
                groups.setdefault(max(int(target_idx - start_idx) for _, target_idx in usable), []).append(
                    (start, start_idx, usable)
                )
        sequences = next(group[:3] for group in groups.values() if len(group) >= 2)
        kwargs = {
            "active_max_days": active_max_days,
            "profile_huber_delta": 2.0,
            "task_mode": "analysis",
            "teacher_forcing_probability": 0.0,
            "state_noise_weight": 0.0,
            "residual_regularization_weight": 0.02,
            "daily_tendency_weight": 0.02,
            "residual_time_smooth_weight": 0.01,
            "physical_scale_regularization_weight": 0.01,
            "physical_scale_smoothness_weight": 0.005,
            "heat_content_transition_weight": DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
            "heat_content_full_column_min_coverage": DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
            "heat_content_transition_season_factors": DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
            "heat_content_transition_depth_factor": True,
            "heat_content_transition_effective_max": DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
            "hard_density_stability": True,
        }
        scalar = [_long_rollout_sequence_loss(model, lake, sequence, **kwargs) for sequence in sequences]
        batched = _long_rollout_sequence_losses_for_lake(
            model,
            lake,
            sequences,
            segment_rollout_batch_mode="on",
            segment_rollout_batch_size=0,
            **kwargs,
        )
        assert [count for _, count, _ in batched] == [count for _, count, _ in scalar]
        assert torch.allclose(
            torch.stack([loss for loss, _, _ in scalar]),
            torch.stack([loss for loss, _, _ in batched]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_cross_lake_transition_batch_matches_per_lake_batch():
    with tempfile.TemporaryDirectory() as tmp:
        lakes, model = _compatible_lakes_and_model(Path(tmp))
        train_pairs = []
        for lake in lakes:
            pairs = [pair for pair in lake["pairs"]["train"] if int(pair[3] - pair[2]) == 1][:2]
            assert len(pairs) >= 2
            lake_copy = dict(lake)
            lake_copy["pairs"] = dict(lake["pairs"])
            lake_copy["pairs"]["train"] = pairs
            train_pairs.append(lake_copy)
        kwargs = {
            "profile_huber_delta": 2.0,
            "lst_surface_weight": 0.03,
            "energy_balance_weight": 0.001,
            "heat_content_transition_weight": DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
            "heat_content_full_column_min_coverage": DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
            "heat_content_transition_depth_factor": True,
            "heat_content_transition_effective_max": DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
            "residual_regularization_weight": 0.02,
            "daily_tendency_weight": 0.02,
            "free_roll_loss_weight": 0.0,
            "physical_scale_regularization_weight": 0.01,
            "physical_scale_smoothness_weight": 0.005,
            "task_mode": "analysis",
            "hard_density_stability": True,
        }
        per_lake = [
            _transition_losses_for_lake(
                model,
                lake,
                lake["pairs"]["train"],
                transition_batch_mode="on",
                transition_batch_size=0,
                heat_content_transition_season_factors=DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
                **kwargs,
            )[0]
            for lake in train_pairs
        ]
        cross = _transition_losses_for_lakes_cross_batch(
            model,
            train_pairs,
            transition_batch_mode="on",
            transition_batch_size=0,
            cross_lake_batch_size=0,
            **kwargs,
        )
        for lake_idx in range(len(train_pairs)):
            assert torch.allclose(
                torch.stack(per_lake[lake_idx]),
                torch.stack(cross[lake_idx][0]),
                atol=1e-5,
                rtol=1e-5,
            )


def test_cross_lake_segment_batch_matches_per_lake_batch():
    with tempfile.TemporaryDirectory() as tmp:
        lakes, model = _compatible_lakes_and_model(Path(tmp))
        active_max_days = 5
        selected_by_lake = {}
        for lake_idx, lake in enumerate(lakes):
            groups = {}
            for sequence in lake["long_rollout_sequences"]["train"]:
                start, start_idx, targets = sequence
                usable = [
                    (target, target_idx)
                    for target, target_idx in targets
                    if 1 <= int(target_idx - start_idx) <= active_max_days
                ]
                if usable:
                    groups.setdefault(max(int(target_idx - start_idx) for _, target_idx in usable), []).append(
                        (start, start_idx, usable)
                    )
            selected_by_lake[lake_idx] = next(group[:2] for group in groups.values() if len(group) >= 2)
        kwargs = {
            "active_max_days": active_max_days,
            "profile_huber_delta": 2.0,
            "task_mode": "analysis",
            "teacher_forcing_probability": 0.0,
            "state_noise_weight": 0.0,
            "residual_regularization_weight": 0.02,
            "daily_tendency_weight": 0.02,
            "residual_time_smooth_weight": 0.01,
            "physical_scale_regularization_weight": 0.01,
            "physical_scale_smoothness_weight": 0.005,
            "heat_content_transition_weight": DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
            "heat_content_full_column_min_coverage": DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
            "heat_content_transition_depth_factor": True,
            "heat_content_transition_effective_max": DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
            "hard_density_stability": True,
        }
        per_lake = {
            lake_idx: _long_rollout_sequence_losses_for_lake(
                model,
                lake,
                selected_by_lake[lake_idx],
                segment_rollout_batch_mode="on",
                segment_rollout_batch_size=0,
                heat_content_transition_season_factors=DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
                **kwargs,
            )
            for lake_idx, lake in enumerate(lakes)
        }
        cross = _long_rollout_sequence_losses_for_lakes_cross_batch(
            model,
            lakes,
            selected_by_lake,
            segment_rollout_batch_mode="on",
            segment_rollout_batch_size=0,
            cross_lake_batch_size=0,
            **kwargs,
        )
        for lake_idx in range(len(lakes)):
            assert [count for _, count, _ in per_lake[lake_idx]] == [
                count for _, count, _ in cross[lake_idx]
            ]
            assert torch.allclose(
                torch.stack([loss for loss, _, _ in per_lake[lake_idx]]),
                torch.stack([loss for loss, _, _ in cross[lake_idx]]),
                atol=1e-5,
                rtol=1e-5,
            )


def test_manifest_transition_loss_weight_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "transition_loss_weight": 0.5,
            "turbulent_flux_mode": "blend",
            "turbulent_flux_blend_alpha": 0.25,
            "checkpoint_every_epochs": 3,
            "eval_every_epochs": 2,
            "light_eval_every_epochs": 3,
            "full_eval_every_epochs": 4,
            "profile_runtime": True,
            "profile_gpu": True,
            "transition_batch_mode": "on",
            "segment_rollout_batch_mode": "on",
            "transition_batch_size": 2,
            "segment_rollout_batch_size": 3,
            "rolling_horizon_batch_mode": "on",
            "rolling_horizon_batch_size": 4,
            "full_free_roll_batch_mode": "on",
            "full_free_roll_batch_size": 5,
            "step_forcing_mode": "tensor",
            "diagnostic_mode": "full",
            "train_diagnostic_mode": "loss",
            "rollout_batch_step_mode": "on",
            "export_after_training": "on",
            "cross_lake_batch_mode": "off",
            "cross_lake_batch_size": 6,
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=0,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            test_lake_id="lake_b",
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert bundle["transition_loss_weight"] == 0.5
        assert bundle["turbulent_flux_mode"] == "blend"
        assert bundle["turbulent_flux_blend_alpha"] == 0.25
        assert bundle["checkpoint_every_epochs"] == 3
        assert bundle["eval_every_epochs"] == 2
        assert bundle["light_eval_every_epochs"] == 3
        assert bundle["full_eval_every_epochs"] == 4
        assert bundle["profile_runtime"] is True
        assert bundle["profile_gpu"] is True
        assert bundle["transition_batch_mode"] == "on"
        assert bundle["segment_rollout_batch_mode"] == "on"
        assert bundle["transition_batch_size"] == 2
        assert bundle["segment_rollout_batch_size"] == 3
        assert bundle["rolling_horizon_batch_mode"] == "on"
        assert bundle["rolling_horizon_batch_size"] == 4
        assert bundle["full_free_roll_batch_mode"] == "on"
        assert bundle["full_free_roll_batch_size"] == 5
        assert bundle["step_forcing_mode"] == "tensor"
        assert bundle["diagnostic_mode"] == "full"
        assert bundle["train_diagnostic_mode"] == "loss"
        assert bundle["rollout_batch_step_mode"] == "on"
        assert bundle["export_after_training"] == "on"
        assert bundle["cross_lake_batch_mode"] == "off"
        assert bundle["cross_lake_batch_size"] == 6
        assert split_summary["_config"]["transition_loss_weight"] == 0.5
        assert split_summary["_config"]["turbulent_flux_mode"] == "blend"
        assert split_summary["_config"]["turbulent_flux_blend_alpha"] == 0.25
        assert split_summary["_config"]["checkpoint_every_epochs"] == 3
        assert split_summary["_config"]["eval_every_epochs"] == 2
        assert split_summary["_config"]["light_eval_every_epochs"] == 3
        assert split_summary["_config"]["full_eval_every_epochs"] == 4
        assert split_summary["_config"]["profile_runtime"] is True
        assert split_summary["_config"]["profile_gpu"] is True
        assert split_summary["_config"]["transition_batch_mode"] == "on"
        assert split_summary["_config"]["segment_rollout_batch_mode"] == "on"
        assert split_summary["_config"]["transition_batch_size"] == 2
        assert split_summary["_config"]["segment_rollout_batch_size"] == 3
        assert split_summary["_config"]["rolling_horizon_batch_mode"] == "on"
        assert split_summary["_config"]["rolling_horizon_batch_size"] == 4
        assert split_summary["_config"]["full_free_roll_batch_mode"] == "on"
        assert split_summary["_config"]["full_free_roll_batch_size"] == 5
        assert split_summary["_config"]["step_forcing_mode"] == "tensor"
        assert split_summary["_config"]["diagnostic_mode"] == "full"
        assert split_summary["_config"]["train_diagnostic_mode"] == "loss"
        assert split_summary["_config"]["rollout_batch_step_mode"] == "on"
        assert split_summary["_config"]["export_after_training"] == "on"
        assert split_summary["_config"]["cross_lake_batch_mode"] == "off"
        assert split_summary["_config"]["cross_lake_batch_size"] == 6


def test_zero_transition_loss_weight_with_segment_rollout_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=1,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            test_lake_id="lake_b",
            transition_loss_weight=0.0,
            segment_rollout_loss_weight=0.20,
            segment_rollout_max_days=5,
            long_free_roll_start_epoch=0,
            long_free_roll_ramp_epochs=1,
            long_free_roll_samples_per_lake=0,
            teacher_forcing_start=0.5,
            teacher_forcing_end=0.0,
            state_noise_weight=0.0,
            transition_batch_mode="on",
            segment_rollout_batch_mode="on",
            transition_batch_size=2,
            segment_rollout_batch_size=2,
            device="cpu",
        )

        history = pd.read_csv(result["history_csv"])
        assert float(history["transition_loss_weight"].iloc[-1]) == 0.0
        assert float(history["transition_loss_weighted"].iloc[-1]) == 0.0
        assert history["transition_batch_mode"].iloc[-1] == "on"
        assert history["segment_rollout_batch_mode"].iloc[-1] == "on"
        assert int(history["transition_batch_size"].iloc[-1]) == 2
        assert int(history["segment_rollout_batch_size"].iloc[-1]) == 2
        assert float(history["long_free_roll_supervision_count"].fillna(0.0).max()) > 0.0


def test_heat_content_transition_dynamic_weight_defaults_and_cap():
    factors = _parse_heat_content_transition_season_factors(None)
    assert factors == DEFAULT_HEAT_CONTENT_TRANSITION_SEASON_FACTORS
    assert factors == DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS
    assert _normalize_heat_content_transition_depth_factor("on") is True
    assert _normalize_heat_content_transition_depth_factor("off") is False

    october_weight = _heat_content_transition_effective_weight(
        0.05,
        pd.Timestamp("2020-10-15"),
        25.3,
        factors,
        use_depth_factor=True,
        effective_max=0.10,
    )
    assert october_weight == pytest.approx(0.10)

    zero_weight = _heat_content_transition_effective_weight(
        0.0,
        pd.Timestamp("2020-10-15"),
        25.3,
        factors,
        use_depth_factor=True,
        effective_max=0.10,
    )
    assert zero_weight == 0.0

    no_depth_weight = _heat_content_transition_effective_weight(
        0.05,
        pd.Timestamp("2020-10-15"),
        25.3,
        factors,
        use_depth_factor=False,
        effective_max=0.20,
    )
    assert no_depth_weight == pytest.approx(0.10)


def test_heat_content_transition_latitude_auto_season_factors():
    northern = _resolve_heat_content_transition_season_factors({"latitude": 43.1}, mode="auto")
    assert northern["resolved_mode"] == "northern"
    assert northern["factors"][9] == 2.0
    assert northern["factors"][10] == 2.0
    assert northern["factors"][11] == 2.0

    southern = _resolve_heat_content_transition_season_factors({"latitude": -43.1}, mode="auto")
    assert southern["resolved_mode"] == "southern"
    assert southern["factors"] == DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS
    assert southern["factors"][3] == 2.0
    assert southern["factors"][4] == 2.0
    assert southern["factors"][5] == 2.0

    tropical = _resolve_heat_content_transition_season_factors({"latitude": 2.0}, mode="auto")
    assert tropical["resolved_mode"] == "tropical"
    assert tropical["factors"] == DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS
    assert set(tropical["factors"].values()) == {1.0}

    fallback = _resolve_heat_content_transition_season_factors({}, mode="auto")
    assert fallback["resolved_mode"] == "northern_fallback"
    assert fallback["latitude"] is None
    assert fallback["factors"] == DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS

    manual = _resolve_heat_content_transition_season_factors(
        {"latitude": -43.1},
        mode="auto",
        override={"3": 3.0, "10": 0.25},
    )
    assert manual["requested_mode"] == "auto"
    assert manual["resolved_mode"] == "manual"
    assert manual["factors"][3] == 3.0
    assert manual["factors"][10] == 0.25


def test_heat_content_transition_season_factor_validation():
    with pytest.raises(ValueError, match="months must be in 1..12"):
        _parse_heat_content_transition_season_factors("13:1.0")
    with pytest.raises(ValueError, match="finite and non-negative"):
        _parse_heat_content_transition_season_factors("10:-1.0")
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        _normalize_heat_content_transition_depth_factor("maybe")
    with pytest.raises(ValueError, match="manual requires"):
        _resolve_heat_content_transition_season_factors({"latitude": 43.1}, mode="manual")


def test_manifest_heat_content_dynamic_weight_config_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "heat_content_transition_weight": 0.05,
            "heat_content_transition_season_factors": {"1": 0.25, "10": 3.0},
            "heat_content_transition_depth_factor": "off",
            "heat_content_transition_effective_max": 0.08,
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=0,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            test_lake_id="lake_b",
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert bundle["heat_content_transition_weight_base"] == 0.05
        assert bundle["heat_content_transition_depth_factor"] == "off"
        assert bundle["heat_content_transition_effective_max"] == 0.08
        assert bundle["heat_content_transition_season_mode"] == "auto"
        assert bundle["heat_content_transition_season_factors_override"]["1"] == 0.25
        assert bundle["heat_content_transition_season_factors_override"]["10"] == 3.0
        assert bundle["heat_content_transition_lake_configs"]["lake_a"]["resolved_mode"] == "manual"
        assert bundle["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["1"] == 0.25
        assert bundle["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["10"] == 3.0
        assert split_summary["_config"]["heat_content_transition_depth_factor"] == "off"
        assert split_summary["_config"]["heat_content_transition_effective_max"] == 0.08
        assert split_summary["_config"]["heat_content_transition_season_mode"] == "auto"
        assert split_summary["_config"]["heat_content_transition_season_factors_override"]["1"] == 0.25
        assert split_summary["_config"]["heat_content_transition_season_factors_override"]["10"] == 3.0
        assert split_summary["_config"]["heat_content_transition_lake_configs"]["lake_a"]["resolved_mode"] == "manual"
        assert split_summary["_config"]["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["1"] == 0.25
        assert split_summary["_config"]["heat_content_transition_lake_configs"]["lake_a"]["season_factors"]["10"] == 3.0


def test_latitude_auto_season_mode_records_per_lake_configs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "heat_content_transition_season_mode": "auto",
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "north_lake", 0.0, latitude=43.1),
                _write_lake_inputs(root, "south_lake", 1.0, latitude=-43.1),
                _write_lake_inputs(root, "tropical_lake", 2.0, latitude=2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=0,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        configs = bundle["heat_content_transition_lake_configs"]
        assert configs["north_lake"]["resolved_mode"] == "northern"
        assert configs["north_lake"]["season_factors"]["10"] == 2.0
        assert configs["south_lake"]["resolved_mode"] == "southern"
        assert configs["south_lake"]["season_factors"]["4"] == 2.0
        assert configs["tropical_lake"]["resolved_mode"] == "tropical"
        assert set(configs["tropical_lake"]["season_factors"].values()) == {1.0}

        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["south_lake"]["heat_content_transition"]["resolved_mode"] == "southern"
        assert split_summary["tropical_lake"]["heat_content_transition"]["resolved_mode"] == "tropical"


def test_removed_forecast_manifest_mode_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "forecast",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="forecast/nowcast modes were removed"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=0,
                depth_points=5,
                max_rollout_days=5,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_removed_forecast_start_manifest_field_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "task_mode": "analysis",
            "forecast_start_date": "2020-01-01",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="forecast_start_date was removed"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=0,
                depth_points=5,
                max_rollout_days=5,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_heat_content_transition_loss_prefers_full_column_when_coverage_is_high():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    area = torch.ones_like(depths)
    start = torch.tensor([10.0, 9.5, 9.0, 8.5, 8.0])
    target = torch.tensor([9.8, 9.3, 8.8, 8.3, 7.8])
    prediction = target.clone()
    prediction[-1] += 4.0
    mostly_covered_mask = [True, True, True, True, False]

    full_column_loss = _heat_content_transition_loss(
        start,
        prediction,
        target,
        depths,
        area,
        start_mask=mostly_covered_mask,
        end_mask=mostly_covered_mask,
        delta_seconds=86400.0,
        min_full_column_coverage=0.75,
    )
    fallback_loss = _heat_content_transition_loss(
        start,
        prediction,
        target,
        depths,
        area,
        start_mask=mostly_covered_mask,
        end_mask=mostly_covered_mask,
        delta_seconds=86400.0,
        min_full_column_coverage=0.95,
    )
    equal_profile_loss = _heat_content_transition_loss(
        start,
        target,
        target,
        depths,
        area,
        start_mask=mostly_covered_mask,
        end_mask=mostly_covered_mask,
        delta_seconds=86400.0,
        min_full_column_coverage=0.75,
    )

    assert float(full_column_loss) > 0.0
    assert float(fallback_loss) == 0.0
    assert float(equal_profile_loss) == 0.0


def test_heat_content_transition_loss_handles_empty_common_mask():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0])
    area = torch.ones_like(depths)
    start = torch.tensor([10.0, 9.0, 8.0, 7.0])
    target = torch.tensor([9.5, 8.5, 7.5, 6.5])
    prediction = target + 2.0

    loss = _heat_content_transition_loss(
        start,
        prediction,
        target,
        depths,
        area,
        start_mask=[True, False, False, False],
        end_mask=[False, True, False, False],
        delta_seconds=86400.0,
        min_full_column_coverage=0.75,
    )

    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_heat_content_transition_loss_vector_matches_scalar():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    area = torch.tensor([1.0, 0.9, 0.7, 0.5, 0.3])
    start = torch.tensor([
        [10.0, 9.5, 9.0, 8.5, 8.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
        [12.0, 11.5, 11.0, 10.5, 10.0],
    ])
    target = torch.tensor([
        [9.8, 9.3, 8.8, 8.3, 7.8],
        [4.1, 4.2, 4.3, 4.4, 4.5],
        [11.8, 11.3, 10.8, 10.3, 9.8],
    ])
    prediction = target.clone()
    prediction[0, -1] += 3.0
    prediction[1, 2:] += 2.0
    prediction[2, :2] -= 1.5
    start_mask = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False],
        [True, False, False, False, False],
    ])
    end_mask = torch.tensor([
        [True, True, True, True, False],
        [False, False, True, True, True],
        [False, True, False, False, False],
    ])
    delta_seconds = torch.tensor([86400.0, 2.0 * 86400.0, 3.0 * 86400.0])

    vector_loss = _heat_content_transition_loss_vector(
        start,
        prediction,
        target,
        depths,
        area,
        start_mask=start_mask,
        end_mask=end_mask,
        delta_seconds=delta_seconds,
        min_full_column_coverage=0.75,
    )
    scalar_losses = torch.stack([
        _heat_content_transition_loss(
            start[idx],
            prediction[idx],
            target[idx],
            depths,
            area,
            start_mask=start_mask[idx],
            end_mask=end_mask[idx],
            delta_seconds=float(delta_seconds[idx]),
            min_full_column_coverage=0.75,
        )
        for idx in range(start.shape[0])
    ])

    assert torch.allclose(vector_loss, scalar_losses, atol=1e-5, rtol=1e-5)


def test_heat_content_transition_loss_accepts_cuda_masks_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    area = torch.ones_like(depths)
    start = torch.tensor([[4.0, 4.0, 4.0, 4.0]], device=device)
    target = torch.tensor([[4.5, 4.5, 5.0, 5.0]], device=device)
    prediction = torch.tensor([[4.5, 4.5, 7.0, 7.0]], device=device)
    mask = torch.tensor([True, True, False, False], dtype=torch.bool, device=device)

    loss = _heat_content_transition_loss(
        start,
        prediction,
        target,
        depths,
        area,
        start_mask=mask,
        end_mask=mask,
        delta_seconds=86400.0,
        min_full_column_coverage=0.75,
    )

    assert torch.isfinite(loss)


def test_rollout_diagnostic_summaries_tolerate_missing_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata = {"file_tag": "diag_lake"}
        diagnostics = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "heat_input_wm2": [float("nan"), float("nan"), float("nan")],
                "heat_tendency_wm2": [float("nan"), float("nan"), float("nan")],
                "energy_residual_wm2": [float("nan"), float("nan"), float("nan")],
                "shortwave_to_water_wm2": [float("nan"), float("nan"), float("nan")],
                "ice_shortwave_transmission": [float("nan"), float("nan"), float("nan")],
                "surface_flux_bias_wm2": [float("nan"), float("nan"), float("nan")],
            }
        )
        heat_paths = write_heat_closure_summaries(diagnostics, metadata, root, "heldout")
        density_paths = write_density_stability_summary(
            np.full((4, 3), np.nan, dtype=np.float32),
            np.linspace(0.0, 3.0, 4),
            diagnostics["Date"],
            metadata,
            root,
            "heldout",
        )

        assert heat_paths["heat_closure_monthly_summary"].exists()
        assert heat_paths["heat_closure_annual_summary"].exists()
        assert density_paths["density_stability_summary"].exists()
        annual = pd.read_csv(heat_paths["heat_closure_annual_summary"])
        density = pd.read_csv(density_paths["density_stability_summary"])
        assert "heat_input_wm2_mean" in annual.columns
        assert "temperature_floor_heat_injection_wm2_mean" in annual.columns
        assert "open_water_sensible_heat_wm2_mean" in annual.columns
        assert "density_unstable_days" in density.columns


def test_metadata_override_updates_forcing_defaults():
    from lake_pinn.state_multilake import prepare_lake_state_data

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lake = _write_lake_inputs(root, "lake_meta", 0.0)
        metadata_path = Path(lake["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"light_extinction_kd": 0.9, "effective_fetch_m": 12345.0})
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        loaded = prepare_lake_state_data(
            lake,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            device="cpu",
        )
        assert float(loaded["df"]["light_extinction_kd"].iloc[0]) == 0.9
        assert float(loaded["df"]["effective_fetch"].iloc[0]) == 12345.0


def test_profile_lookup_returns_depth_mask_without_deep_extrapolation():
    depths = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0]).to_numpy(dtype="float32")
    profile = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01"] * 3),
            "Depth_m": [0.0, 2.0, 4.0],
            "Temperature_C": [10.0, 9.0, 8.0],
        }
    )
    lookup, masks = _profile_lookup(profile, depths, return_masks=True)
    date = pd.Timestamp("2020-01-01")
    assert date in lookup
    assert masks[date].tolist() == [True, True, True, False, False]


def test_rolling_start_horizon_reports_all_observation_starts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lake_config = _write_lake_inputs(root, "lake_roll", 0.0)
        from lake_pinn.state_multilake import prepare_lake_state_data
        from lake_pinn.state_model import LakeStateForecaster

        lake = prepare_lake_state_data(
            lake_config,
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            split_mode="time_blocked",
            task_mode="analysis",
            data_fill_mode="reconstruction",
            device="cpu",
        )
        model = LakeStateForecaster(lake["depths_np"], lake["area_np"], hidden_dim=16)
        metrics = evaluate_lake_rolling_start_horizons(
            model,
            lake,
            horizons=(1, 3),
            task_mode="analysis",
        )
        assert metrics["count_1d"] > 0
        assert metrics["count_3d"] > 0


def test_step_tensor_forcing_matches_dict_and_loss_diagnostics():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        profile = next(iter(lake["lookup_tensors"]["all"].values()))
        current = profile.clone()
        dict_row = lake["forcing_rows"][0]
        dict_next = lake["forcing_rows"][1]
        tensor_row = _forcing_row_batch(lake, [0], step_forcing_mode="tensor")
        tensor_next = _forcing_row_batch(lake, [1], step_forcing_mode="tensor")
        assert isinstance(tensor_row, ForcingBatch)
        dict_prediction, dict_diag = model.step(
            current,
            dict_row,
            lake["static_features"],
            next_forcing_row=dict_next,
            task_mode="analysis",
            depths=lake["depths"],
            area_profile=lake["area"],
            hard_density_stability=True,
            return_diagnostics=True,
            diagnostic_mode="full",
        )
        tensor_prediction, loss_diag = model.step(
            current,
            tensor_row,
            lake["static_features"],
            next_forcing_row=tensor_next,
            task_mode="analysis",
            depths=lake["depths"],
            area_profile=lake["area"],
            hard_density_stability=True,
            return_diagnostics=True,
            diagnostic_mode="loss",
        )
        assert torch.allclose(tensor_prediction, dict_prediction, atol=1e-5, rtol=1e-5)
        for key in (
            "residual_abs_mean_c",
            "residual_surface_c",
            "residual_deep_mean_c",
            "residual_profile_c",
            "shortwave_absorption_scale",
            "surface_cooling_scale_raw",
            "surface_cooling_scale",
            "surface_flux_bias_wm2",
            "heat_input_wm2",
            "heat_tendency_wm2",
            "temperature_floor_heat_injection_wm2",
            "density_adjustment_applied",
        ):
            assert key in loss_diag
            assert torch.allclose(loss_diag[key], dict_diag[key], atol=1e-5, rtol=1e-5)
        assert "open_water_sensible_heat_wm2" not in loss_diag


def test_rollout_batch_matches_manual_loop():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        profile = next(iter(lake["lookup_tensors"]["all"].values()))
        initial = profile.expand(2, -1).clone()
        starts = torch.tensor([0, 1], dtype=torch.long)
        forcing_sequence = [
            _forcing_row_batch(lake, starts + offset, step_forcing_mode="tensor")
            for offset in range(3)
        ]
        next_forcing_sequence = [
            _forcing_row_batch(lake, starts + offset + 1, step_forcing_mode="tensor")
            for offset in range(3)
        ]
        batch_states = model.rollout_batch(
            initial,
            forcing_sequence,
            lake["static_features"].expand(2, -1),
            next_forcing_rows=next_forcing_sequence,
            task_mode="analysis",
            depths=lake["depths"],
            area_profile=lake["area"],
            hard_density_stability=True,
        )
        current = initial
        manual_states = []
        for row, next_row in zip(forcing_sequence, next_forcing_sequence):
            current = model.step(
                current,
                row,
                lake["static_features"].expand(2, -1),
                next_forcing_row=next_row,
                task_mode="analysis",
                depths=lake["depths"],
                area_profile=lake["area"],
                hard_density_stability=True,
            )
            manual_states.append(current)
        manual_states = torch.stack(manual_states, dim=0)
        assert torch.allclose(batch_states, manual_states, atol=1e-5, rtol=1e-5)


def test_batched_rolling_start_horizon_matches_scalar_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        scalar = evaluate_lake_rolling_start_horizons(
            model,
            lake,
            horizons=(1, 3, 7),
            task_mode="analysis",
            max_start_profiles=0,
            hard_density_stability=True,
        )
        batched = evaluate_lake_rolling_start_horizons_batched(
            model,
            lake,
            horizons=(1, 3, 7),
            task_mode="analysis",
            max_start_profiles=0,
            hard_density_stability=True,
            batch_size=3,
        )
        rollout_batched = evaluate_lake_rolling_start_horizons_batched(
            model,
            lake,
            horizons=(1, 3, 7),
            task_mode="analysis",
            max_start_profiles=0,
            hard_density_stability=True,
            batch_size=3,
            rollout_batch_step_mode="on",
        )
        for horizon in (1, 3, 7):
            assert batched[f"count_{horizon}d"] == scalar[f"count_{horizon}d"]
            assert batched[f"rmse_{horizon}d"] == pytest.approx(scalar[f"rmse_{horizon}d"], abs=1e-5)
            assert batched[f"bias_{horizon}d"] == pytest.approx(scalar[f"bias_{horizon}d"], abs=1e-5)
            assert rollout_batched[f"count_{horizon}d"] == scalar[f"count_{horizon}d"]
            assert rollout_batched[f"rmse_{horizon}d"] == pytest.approx(scalar[f"rmse_{horizon}d"], abs=1e-5)
            assert rollout_batched[f"bias_{horizon}d"] == pytest.approx(scalar[f"bias_{horizon}d"], abs=1e-5)


def test_batched_full_free_roll_matches_scalar_for_compatible_lakes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lake_a = prepare_lake_state_data(
            _write_lake_inputs(root, "lake_a", 0.0),
            split_mode="none",
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            device="cpu",
        )
        lake_b = prepare_lake_state_data(
            _write_lake_inputs(root, "lake_b", 1.0),
            split_mode="none",
            depth_points=5,
            max_rollout_days=5,
            history_window_days=3,
            device="cpu",
        )
        model = LakeStateForecaster(
            lake_a["depths"],
            lake_a["area"],
            static_dim=lake_a["static_features"].numel(),
            hidden_dim=16,
            forcing_context_dim=8,
            forcing_history_hidden_dim=8,
            residual_limit_c=0.25,
            turbulent_flux_mode="bulk",
        )
        model.eval()
        scalar = {
            lake["lake_id"]: evaluate_lake_free_roll(
                model,
                lake,
                task_mode="analysis",
                horizons=(1, 3, 7),
                hard_density_stability=True,
            )
            for lake in (lake_a, lake_b)
        }
        batched = evaluate_heldout_free_rolls(
            model,
            [lake_a, lake_b],
            task_mode="analysis",
            horizons=(1, 3, 7),
            hard_density_stability=True,
            full_free_roll_batch_mode="on",
            full_free_roll_batch_size=2,
        )
        for lake_id in scalar:
            for key in ("rmse", "mae", "bias", "n_profiles"):
                if key == "n_profiles":
                    assert batched[lake_id][key] == scalar[lake_id][key]
                else:
                    assert batched[lake_id][key] == pytest.approx(scalar[lake_id][key], abs=1e-5)
            for horizon in (1, 3, 7):
                for metric in ("rmse", "bias", "count"):
                    key = f"{metric}_{horizon}d"
                    if metric == "count":
                        assert batched[lake_id]["horizon_metrics"][key] == scalar[lake_id]["horizon_metrics"][key]
                    else:
                        assert batched[lake_id]["horizon_metrics"][key] == pytest.approx(
                            scalar[lake_id]["horizon_metrics"][key],
                            abs=1e-5,
                        )


if __name__ == "__main__":
    test_multilake_state_training_excludes_heldout_lake_from_checkpoint()
    test_manifest_transition_loss_weight_is_recorded()
    test_zero_transition_loss_weight_with_segment_rollout_smoke()
    test_heat_content_transition_dynamic_weight_defaults_and_cap()
    test_heat_content_transition_latitude_auto_season_factors()
    test_heat_content_transition_season_factor_validation()
    test_manifest_heat_content_dynamic_weight_config_is_recorded()
    test_latitude_auto_season_mode_records_per_lake_configs()
    test_removed_forecast_manifest_mode_is_rejected()
    test_removed_forecast_start_manifest_field_is_rejected()
    test_heat_content_transition_loss_prefers_full_column_when_coverage_is_high()
    test_heat_content_transition_loss_handles_empty_common_mask()
    test_rollout_diagnostic_summaries_tolerate_missing_values()
    test_metadata_override_updates_forcing_defaults()
    test_profile_lookup_returns_depth_mask_without_deep_extrapolation()
    test_rolling_start_horizon_reports_all_observation_starts()
    test_batched_rolling_start_horizon_matches_scalar_evaluation()
    test_batched_full_free_roll_matches_scalar_for_compatible_lakes()
    print("multi-lake state forecaster sanity checks passed")
