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
from lake_pinn.data_io import split_profile_observations
from lake_pinn.diagnostics import write_density_stability_summary, write_heat_closure_summaries
from lake_pinn.state_model import ForcingBatch, LakeStateForecaster, STATIC_FEATURE_DIM, STATIC_FEATURE_KEYS
from lake_pinn.state_multilake import (
    DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS,
    DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
    DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    _build_cross_lake_segment_rollout_epoch_batches,
    _build_segment_rollout_epoch_plan,
    _column_mean_temperature_c_vector,
    _depth_limited_export_grid,
    _heat_content_transition_effective_weight,
    _heat_content_transition_loss,
    _heat_content_transition_loss_vector,
    _forcing_row_batch,
    _forcing_tensor_matrix,
    _normalize_heat_content_transition_depth_factor,
    _parse_heat_content_transition_season_factors,
    _resolve_heat_content_transition_season_factors,
    _scheduled_segment_rollout_days,
    _segment_rollout_sequence_loss,
    _segment_rollout_sequence_losses_for_lake,
    _segment_rollout_sequence_losses_for_lakes_cross_batch,
    _segment_rollout_sequences_for_epoch,
    _select_segment_rollout_sequences,
    _transition_loss,
    _transition_losses_for_lake,
    _transition_losses_for_lakes_cross_batch,
    _warm_column_heat_content_loss_details,
    _warm_column_heat_content_loss_vector,
    _warm_column_heat_content_quantiles,
    evaluate_lake_pairs,
    evaluate_heldout_free_rolls,
    evaluate_lakes_rolling_start_horizons,
    evaluate_lake_free_roll,
    evaluate_lake_rolling_start_horizons,
    evaluate_lake_rolling_start_horizons_batched,
    prepare_lake_state_data,
    train_multilake_state_forecaster,
)


def test_profile_split_defaults_to_train_val_only():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        for depth in [0.0, 2.0]:
            rows.append({
                "Date": date,
                "Depth_m": depth,
                "Temperature_C": 5.0 + 0.1 * len(rows),
            })
    profile_obs = pd.DataFrame(rows)

    splits, info = split_profile_observations(profile_obs, split_mode="time_blocked")

    assert set(splits) == {"train", "val"}
    assert set(info["summary"]) == {"train", "val"}
    assert "assim" not in splits
    assert "test" not in splits
    assert info["summary"]["train"]["date_count"] == 8
    assert info["summary"]["val"]["date_count"] == 2


def test_profile_split_depth_and_seasonal_modes_do_not_create_assim_or_test():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    rows = []
    for date in dates:
        for depth in [0.0, 1.0, 2.0, 3.0, 4.0]:
            rows.append({
                "Date": date,
                "Depth_m": depth,
                "Temperature_C": 6.0 - 0.2 * depth,
            })
    profile_obs = pd.DataFrame(rows)

    for split_mode in ["depth_interleaved", "seasonal_blocked"]:
        splits, info = split_profile_observations(profile_obs, split_mode=split_mode)
        assert set(splits) == {"train", "val"}
        assert set(info["summary"]) == {"train", "val"}
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0


def test_segment_rollout_sequence_sampling_spreads_across_available_dates():
    sequences = [
        (
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx),
            idx,
            [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx + 1), idx + 1)],
        )
        for idx in range(24)
    ]

    selected = _select_segment_rollout_sequences(
        sequences,
        active_max_days=30,
        samples_per_lake=6,
        epoch=0,
    )
    starts = [item[1] for item in selected]

    assert len(starts) == 6
    assert starts[0] == 0
    assert starts[-1] == 23
    assert max(np.diff(starts)) <= 5


def test_segment_rollout_sequence_sampling_rotates_within_uniform_buckets():
    sequences = [
        (
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx),
            idx,
            [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx + 1), idx + 1)],
        )
        for idx in range(24)
    ]

    first = _select_segment_rollout_sequences(
        sequences,
        active_max_days=30,
        samples_per_lake=6,
        epoch=0,
    )
    second = _select_segment_rollout_sequences(
        sequences,
        active_max_days=30,
        samples_per_lake=6,
        epoch=1,
    )

    first_starts = [item[1] for item in first]
    second_starts = [item[1] for item in second]
    assert first_starts == [0, 5, 9, 14, 18, 23]
    assert second_starts != first_starts
    assert second_starts[0] <= 2
    assert second_starts[-1] >= 21


def test_segment_rollout_epoch_plan_matches_selector_and_batches():
    sequences = [
        (
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx),
            idx,
            [
                (pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx + gap), idx + gap)
                for gap in (1, 3, 5)
                if idx + gap < 30
            ],
        )
        for idx in range(24)
    ]
    sequences = [sequence for sequence in sequences if sequence[2]]

    plan = _build_segment_rollout_epoch_plan(
        sequences,
        epochs=8,
        start_epoch=2,
        ramp_epochs=3,
        max_days=20,
        samples_per_lake=6,
        segment_rollout_batch_size=2,
    )

    assert len(plan) == 8
    for epoch, entry in enumerate(plan):
        active_days = _scheduled_segment_rollout_days(
            epoch,
            start_epoch=2,
            ramp_epochs=3,
            max_days=20,
        )
        expected = (
            _select_segment_rollout_sequences(
                sequences,
                active_days,
                samples_per_lake=6,
                epoch=epoch,
            )
            if active_days > 0 else []
        )
        assert entry["active_max_days"] == active_days
        assert list(entry["selected_sequences"]) == expected
        flattened = [sequence for chunk in entry["batches"] for sequence in chunk]
        assert flattened == expected
        assert all(len(chunk) <= 2 for chunk in entry["batches"])


def test_segment_rollout_sequences_for_epoch_uses_cached_plan():
    with tempfile.TemporaryDirectory() as tmp:
        lake, _model = _single_lake_and_model(Path(tmp))
        lake.setdefault("segment_rollout_epoch_plans", {})["train"] = (
            _build_segment_rollout_epoch_plan(
                lake["segment_rollout_sequences"]["train"],
                epochs=4,
                start_epoch=0,
                ramp_epochs=1,
                max_days=5,
                samples_per_lake=2,
                segment_rollout_batch_size=1,
            )
        )
        cached = _segment_rollout_sequences_for_epoch(
            lake,
            "train",
            active_max_days=5,
            samples_per_lake=2,
            epoch=1,
        )

        lake["segment_rollout_sequences"]["train"] = []
        assert cached
        assert _segment_rollout_sequences_for_epoch(
            lake,
            "train",
            active_max_days=5,
            samples_per_lake=2,
            epoch=1,
        ) == cached
        assert _segment_rollout_sequences_for_epoch(
            lake,
            "train",
            active_max_days=4,
            samples_per_lake=2,
            epoch=1,
        ) == []


def test_cross_lake_segment_rollout_epoch_batches_cover_cached_sequences():
    with tempfile.TemporaryDirectory() as tmp:
        lakes, _model = _compatible_lakes_and_model(Path(tmp))
        for lake in lakes:
            lake.setdefault("segment_rollout_epoch_plans", {})["train"] = (
                _build_segment_rollout_epoch_plan(
                    lake["segment_rollout_sequences"]["train"],
                    epochs=3,
                    start_epoch=0,
                    ramp_epochs=1,
                    max_days=5,
                    samples_per_lake=2,
                    segment_rollout_batch_size=1,
                )
            )

        batches = _build_cross_lake_segment_rollout_epoch_batches(
            lakes,
            split_key="train",
            epochs=3,
            segment_rollout_batch_size=1,
            cross_lake_batch_size=2,
        )

        assert len(batches) == 3
        for epoch, epoch_batches in enumerate(batches):
            expected = sum(
                len(lake["segment_rollout_epoch_plans"]["train"][epoch]["selected_sequences"])
                for lake in lakes
            )
            flattened = [item for chunk in epoch_batches for item in chunk]
            assert len(flattened) == expected
            assert all(len(chunk) <= 2 for chunk in epoch_batches)


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


def _write_long_rolling_lake_inputs(root: Path, lake_id: str, offset: float):
    dates = pd.date_range("2020-01-01", periods=481, freq="D")
    era5 = root / f"{lake_id}_era5.csv"
    lst = root / f"{lake_id}_lst.csv"
    profile = root / f"{lake_id}_profile.csv"
    meta = root / f"{lake_id}_metadata.json"
    pd.DataFrame(
        {
            "Date": dates,
            "t2m_C": [8.0 + offset + 0.01 * idx for idx in range(len(dates))],
            "wind_norm_m_per_s": [2.5] * len(dates),
            "ssrd_W_per_m2": [150.0] * len(dates),
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
            "LST_surface_C": [9.0 + offset + 0.01 * idx for idx in range(len(dates))],
            "LST_qc_good_fraction": [1.0] * len(dates),
        }
    ).to_csv(lst, index=False)
    profile_rows = []
    for day in range(0, 481, 30):
        for depth in [0.0, 4.0, 8.0]:
            profile_rows.append(
                {
                    "Date": dates[day],
                    "Depth_m": depth,
                    "Temperature_C": 9.0 + offset + day * 0.01 - depth * 0.15,
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
                "latitude": 45.0,
                "longitude": -90.0,
                "secchi_m": 2.0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "lake_id": lake_id,
        "era5": str(era5),
        "lst": str(lst),
        "profile_obs": str(profile),
        "metadata": str(meta),
        "max_depth": 8.0,
    }


def _single_lake_and_model(root: Path):
    torch.manual_seed(123)
    lake_config = _write_lake_inputs(root, "lake_a", 0.0)
    lake = prepare_lake_state_data(
        lake_config,
        split_mode="none",
        depth_points=5,
        max_rollout_days=5,
        segment_rollout_max_days=5,
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
            segment_rollout_max_days=5,
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


def _set_open_water_lst_supervision(lake, *, value=24.0, quality=1.0, observed=1.0, ice=0.0):
    device = lake["depths"].device
    for row in lake["forcing_rows"]:
        row["lswt_open_water"] = torch.tensor([value], dtype=torch.float32, device=device)
        row["lst_quality"] = torch.tensor([quality], dtype=torch.float32, device=device)
        row["lst_observed_flag"] = torch.tensor([observed], dtype=torch.float32, device=device)
        row["ice_mask"] = torch.tensor([ice], dtype=torch.float32, device=device)
    lake["forcing_tensors"] = _forcing_tensor_matrix(lake["forcing_rows"])


def _enable_warm_column_heat_content_for_test(lake, *, low=0.0, high=1.0):
    lake["warm_season_column_heat_content_low_c"] = float(low)
    lake["warm_season_column_heat_content_high_c"] = float(high)
    lake["warm_season_column_heat_content_quantile_low"] = 0.50
    lake["warm_season_column_heat_content_quantile_high"] = 0.75
    lake["warm_season_column_heat_content_profile_count"] = 3


def test_lake_adaptive_temporal_mode_uses_forcing_context():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lake = prepare_lake_state_data(
            _write_lake_inputs(root, "lake_a", 0.0),
            split_mode="none",
            depth_points=5,
            max_rollout_days=5,
            segment_rollout_max_days=5,
            history_window_days=3,
            device="cpu",
        )
        with pytest.raises(ValueError, match="lake_adaptive_temporal_mode requires lake_adaptive_params"):
            LakeStateForecaster(
                lake["depths"],
                lake["area"],
                static_dim=lake["static_features"].numel(),
                hidden_dim=16,
                forcing_context_dim=8,
                forcing_history_hidden_dim=8,
                turbulent_flux_mode="blend",
                lake_adaptive_temporal_mode="seasonal_forcing",
            )

        torch.manual_seed(321)
        model = LakeStateForecaster(
            lake["depths"],
            lake["area"],
            static_dim=lake["static_features"].numel(),
            hidden_dim=16,
            forcing_context_dim=8,
            forcing_history_hidden_dim=8,
            turbulent_flux_mode="blend",
            lake_adaptive_params="all",
            lake_adaptive_temporal_mode="seasonal_forcing",
            lake_adaptive_temporal_init_spread=0.20,
            lake_adaptive_temporal_scale=0.25,
        )
        model.eval()
        static_features = lake["static_features"].reshape(1, -1)

        def values_for_row(row):
            forcing_features = row["features"].reshape(1, -1)
            forcing_history = row["history_features"].unsqueeze(0)
            forcing_context = model._encode_forcing_context(forcing_features, forcing_history)
            values, regularization = model._adaptive_parameter_values(
                static_features,
                forcing_context=forcing_context,
                forcing_features=forcing_features,
            )
            return {
                key: float(value.reshape(-1)[0].detach())
                for key, value in values.items()
            }, float(regularization.reshape(-1)[0].detach())

        first_values, first_reg = values_for_row(lake["forcing_rows"][0])
        last_values, last_reg = values_for_row(lake["forcing_rows"][-1])
        assert np.isfinite(first_reg)
        assert np.isfinite(last_reg)
        assert any(
            abs(first_values[key] - last_values[key]) > 1.0e-6
            for key in first_values
        )


def test_multilake_state_training_excludes_heldout_lake_from_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            segment_rollout_loss_weight=0.05,
            segment_rollout_start_epoch=0,
            segment_rollout_ramp_epochs=1,
            segment_rollout_max_days=5,
            segment_rollout_samples_per_lake=0,
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
        assert bundle["static_feature_dim"] == STATIC_FEATURE_DIM
        assert bundle["static_feature_keys"] == list(STATIC_FEATURE_KEYS)
        assert bundle["lake_metadata_summary"]["lake_a"]["static_feature_dim"] == STATIC_FEATURE_DIM
        assert bundle["lake_metadata_summary"]["lake_a"]["static_feature_keys"] == list(STATIC_FEATURE_KEYS)
        assert "reservoir_indicator" in bundle["lake_metadata_summary"]["lake_a"]
        assert "residence_time_norm" in bundle["lake_metadata_summary"]["lake_a"]
        assert "catchment_area_norm" in bundle["lake_metadata_summary"]["lake_a"]
        assert bundle["profile_supervision_scope"] == "train"
        assert bundle["residual_regularization_weight"] == 0.02
        assert bundle["daily_tendency_weight"] == 0.02
        assert bundle["physical_scale_regularization_weight"] == 0.01
        assert bundle["physical_scale_smoothness_weight"] == 0.005
        assert bundle["kd_prior_regularization_weight"] == 0.001
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
        assert bundle["shape_aware_mixing"] == "on"
        assert bundle["shape_mixing_strength"] == 0.35
        assert bundle["stratification_mixing_cap"] == "on"
        assert bundle["stratification_mixing_cap_strength"] == 1.0
        assert bundle["lake_adaptive_params"] == "off"
        assert bundle["lake_adaptive_hidden_dim"] == 64
        assert bundle["lake_adaptive_init_spread"] == 0.02
        assert bundle["lake_adaptive_temporal_mode"] == "off"
        assert bundle["lake_adaptive_temporal_init_spread"] == 0.005
        assert bundle["lake_adaptive_temporal_scale"] == 0.25
        assert bundle["adaptive_wind_kz_min"] == 0.4
        assert bundle["adaptive_wind_kz_max"] == 3.0
        assert bundle["adaptive_blend_alpha_min"] == 0.0
        assert bundle["adaptive_blend_alpha_max"] == 0.6
        assert bundle["adaptive_kd_multiplier_min"] == 0.4
        assert bundle["adaptive_kd_multiplier_max"] == 2.0
        assert bundle["adaptive_turbulent_exchange_scale_min"] == 0.5
        assert bundle["adaptive_turbulent_exchange_scale_max"] == 1.8
        assert bundle["adaptive_convective_mixing_scale_min"] == 0.3
        assert bundle["adaptive_convective_mixing_scale_max"] == 2.5
        assert bundle["adaptive_ice_shortwave_scale_min"] == 0.4
        assert bundle["adaptive_ice_shortwave_scale_max"] == 1.8
        assert bundle["adaptive_parameter_regularization_weight"] == 0.01
        assert bundle["checkpoint_every_epochs"] == 1
        assert bundle["eval_every_epochs"] == 1
        assert bundle["full_eval_every_epochs"] == 60
        assert bundle["profile_runtime"] is True
        assert bundle["profile_gpu"] is True
        assert bundle["history_diagnostic_every_epochs"] == 0
        assert bundle["torch_tf32"] == "on"
        assert bundle["torch_matmul_precision"] == "high"
        assert bundle["transition_batch_size"] == 0
        assert bundle["segment_rollout_batch_size"] == 0
        assert bundle["rolling_horizon_batch_size"] == 32
        assert bundle["train_diagnostic_mode"] == "loss"
        assert bundle["export_after_training"] == "on"
        assert bundle["cross_lake_batch_mode"] == "off"
        assert bundle["cross_lake_batch_size"] == 0
        assert bundle["segment_rollout_loss_weight"] == 0.05
        assert bundle["segment_rollout_start_epoch"] == 0
        assert bundle["segment_rollout_ramp_epochs"] == 1
        assert bundle["segment_rollout_max_days"] == 5
        assert bundle["segment_rollout_samples_per_lake"] == 0
        assert bundle["segment_rollout_lst_surface_weight"] == 0.01
        assert bundle["lst_feature_dropout_probability"] == 0.20
        assert bundle["advective_heat_source_mode"] == "reservoir_simple"
        assert bundle["teacher_forcing_start"] == 0.5
        assert bundle["teacher_forcing_end"] == 0.0
        assert bundle["state_noise_weight"] == 0.0
        assert bundle["residual_time_smooth_weight"] == 0.01
        history = pd.read_csv(result["history_csv"])
        assert "heldout_transition_mean_rmse" in history.columns
        assert "profile_supervision_scope" in history.columns
        assert "train_supervision_pair_count" in history.columns
        assert "train_supervision_segment_sequence_count" in history.columns
        assert history["profile_supervision_scope"].iloc[-1] == "train"
        assert "transition_loss_weight" in history.columns
        assert "transition_loss_unweighted" in history.columns
        assert "transition_loss_weighted" in history.columns
        assert float(history["transition_loss_weight"].iloc[-1]) == 1.0
        assert "heldout_free_roll_mean_rmse" in history.columns
        assert "heldout_persistence_mean_rmse" in history.columns
        assert "segment_rollout_loss_weight" in history.columns
        assert "segment_rollout_lst_surface_weight" in history.columns
        assert "segment_rollout_loss" in history.columns
        assert "segment_rollout_supervision_count" in history.columns
        assert "segment_rollout_lst_loss" in history.columns
        assert "segment_rollout_lst_supervision_count" in history.columns
        assert "segment_rollout_lst_weight_mean" in history.columns
        assert "segment_rollout_sequence_count" in history.columns
        assert "segment_rollout_weight_eff" in history.columns
        assert "segment_rollout_active_days" in history.columns
        assert "teacher_forcing_probability" in history.columns
        assert "lst_feature_dropout_probability" in history.columns
        assert "lst_feature_dropout_applied_mean" in history.columns
        assert "physical_scale_reg_loss" in history.columns
        assert "physical_scale_smooth_loss" in history.columns
        assert "kd_prior_regularization_loss" in history.columns
        assert "kd_prior_regularization_weighted_loss" in history.columns
        assert "kd_prior_regularization_weight" in history.columns
        assert "heat_content_transition_loss" in history.columns
        assert "heat_content_transition_weight_base" in history.columns
        assert "heat_content_transition_weighted_loss" in history.columns
        assert "heat_content_transition_effective_weight_mean" in history.columns
        assert "heat_content_transition_effective_weight_min" in history.columns
        assert "heat_content_transition_effective_weight_max" in history.columns
        assert "freezing_storage_ice_mean_j_m2" in history.columns
        assert "freezing_storage_surface_fraction_mean" in history.columns
        assert "freezing_storage_deep_fraction_mean" in history.columns
        assert "heat_content_full_column_min_coverage" in history.columns
        assert "heat_content_transition_season_factor_10" in history.columns
        assert "heat_content_transition_season_factor_min_10" in history.columns
        assert "heat_content_transition_season_factor_max_10" in history.columns
        assert "segment_rollout_heat_content_transition_weighted_loss" in history.columns
        assert "segment_rollout_heat_content_transition_effective_weight_mean" in history.columns
        assert "shortwave_scale_mean" in history.columns
        assert "cooling_scale_mean" in history.columns
        assert "surface_flux_bias_mean_wm2" in history.columns
        assert "turbulent_flux_mode" in history.columns
        assert "turbulent_flux_blend_alpha" in history.columns
        assert "shape_aware_mixing" in history.columns
        assert "shape_mixing_strength" in history.columns
        assert "stratification_mixing_cap" in history.columns
        assert "stratification_mixing_cap_strength" in history.columns
        assert "lake_adaptive_params" in history.columns
        assert "lake_adaptive_hidden_dim" in history.columns
        assert "lake_adaptive_init_spread" in history.columns
        assert "lake_adaptive_temporal_mode" in history.columns
        assert "lake_adaptive_temporal_init_spread" in history.columns
        assert "lake_adaptive_temporal_scale" in history.columns
        assert "adaptive_wind_kz_scale_mean" in history.columns
        assert "adaptive_turbulent_flux_blend_alpha_mean" in history.columns
        assert "adaptive_kd_multiplier_mean" in history.columns
        assert "adaptive_turbulent_exchange_scale_mean" in history.columns
        assert "adaptive_convective_mixing_scale_mean" in history.columns
        assert "adaptive_ice_shortwave_scale_mean" in history.columns
        assert "lake_shape_wind_factor_mean" in history.columns
        assert "lake_shape_decay_depth_mean_m" in history.columns
        assert "stratification_mixing_gate_mean" in history.columns
        assert "stratification_mixing_gate_min" in history.columns
        assert "stratification_mixing_gate_deep_mean" in history.columns
        assert "adaptive_parameter_regularization_loss" in history.columns
        assert "adaptive_parameter_regularization_weight" in history.columns
        assert history["lake_adaptive_params"].iloc[-1] == "off"
        assert history["lake_adaptive_temporal_mode"].iloc[-1] == "off"
        assert history["shape_aware_mixing"].iloc[-1] == "on"
        assert float(history["shape_mixing_strength"].iloc[-1]) == pytest.approx(0.35)
        assert history["stratification_mixing_cap"].iloc[-1] == "on"
        assert float(history["stratification_mixing_cap_strength"].iloc[-1]) == pytest.approx(1.0)
        assert int(history["lake_adaptive_hidden_dim"].iloc[-1]) == 64
        assert float(history["lake_adaptive_init_spread"].iloc[-1]) == pytest.approx(0.02)
        assert float(history["lake_adaptive_temporal_init_spread"].iloc[-1]) == pytest.approx(0.005)
        assert float(history["lake_adaptive_temporal_scale"].iloc[-1]) == pytest.approx(0.25)
        adaptive_summary_path = output_dir / "lake_adaptive_parameter_summary.csv"
        assert result["lake_adaptive_parameter_summary_csv"] == adaptive_summary_path
        assert adaptive_summary_path.exists()
        adaptive_summary = pd.read_csv(adaptive_summary_path)
        assert {
            "lake_id",
            "split",
            "is_train_lake",
            "is_heldout_test_lake",
            "adaptive_wind_kz_scale",
            "adaptive_turbulent_flux_blend_alpha",
            "adaptive_kd_multiplier",
            "adaptive_turbulent_exchange_scale",
            "adaptive_convective_mixing_scale",
            "adaptive_ice_shortwave_scale",
            "adaptive_wind_kz_scale_std",
            "lake_adaptive_temporal_mode",
            "adaptive_parameter_regularization_loss",
        }.issubset(set(adaptive_summary.columns))
        assert set(adaptive_summary["split"]) == {"train", "heldout"}
        assert "checkpoint_every_epochs" in history.columns
        assert "eval_every_epochs" in history.columns
        assert "full_eval_every_epochs" in history.columns
        assert "eval_mode" in history.columns
        assert "profile_runtime" in history.columns
        assert "profile_gpu" in history.columns
        assert "history_diagnostic_every_epochs" in history.columns
        assert "history_diagnostic_enabled" in history.columns
        assert "torch_tf32" in history.columns
        assert "torch_matmul_precision" in history.columns
        assert "transition_batch_size" in history.columns
        assert "segment_rollout_batch_size" in history.columns
        assert "rolling_horizon_batch_size" in history.columns
        assert "train_diagnostic_mode" in history.columns
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
        assert "advective_heat_source_c_per_day_mean" in history.columns
        assert "advective_exchange_fraction_per_day" in history.columns
        assert "heldout_transition_rmse_3d" in history.columns
        assert "val_rolling_start_rmse_3d" in history.columns
        assert "val_rolling_start_bias_3d" in history.columns
        assert "val_rolling_start_count_3d" in history.columns
        assert "heldout_free_roll_rmse_3d" in history.columns
        assert "heldout_free_roll_bias_3d" in history.columns
        assert "heldout_free_roll_count_3d" in history.columns
        assert "heldout_observed_point_mean_rmse" in history.columns
        assert "heldout_observed_point_mean_mae" in history.columns
        assert "heldout_observed_point_mean_bias" in history.columns
        assert "heldout_observed_point_total_count" in history.columns
        assert "lake_b_heldout_observed_point_rmse" in history.columns
        assert "lake_b_heldout_observed_point_surface_rmse" in history.columns
        assert "lake_b_heldout_observed_point_summer_bias" in history.columns
        assert "heldout_initial_free_roll_rmse_3d" in history.columns
        assert "heldout_initial_free_roll_bias_3d" in history.columns
        assert "heldout_initial_free_roll_count_3d" in history.columns
        assert "heldout_rolling_start_rmse_3d" in history.columns
        assert "heldout_rolling_start_bias_3d" in history.columns
        assert "heldout_rolling_start_count_3d" in history.columns
        assert history["eval_mode"].iloc[-1] == "none"
        assert np.isnan(float(history["heldout_free_roll_rmse_3d"].iloc[-1]))
        assert np.isnan(float(history["heldout_rolling_start_rmse_3d"].iloc[-1]))
        assert (
            float(history["heldout_free_roll_count_3d"].fillna(0.0).iloc[-1])
            == 0.0
        )
        assert (
            float(history["heldout_initial_free_roll_count_3d"].fillna(0.0).iloc[-1])
            == 0.0
        )
        assert float(history["segment_rollout_supervision_count"].fillna(0.0).max()) > 0.0
        assert float(history["segment_rollout_sequence_count"].fillna(0.0).max()) > 0.0
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["lake_a"]["train_segment_rollout_sequences"] > 0
        assert "test_pairs" not in split_summary["lake_a"]
        assert split_summary["lake_a"]["all_pairs"] >= split_summary["lake_a"]["train_pairs"]
        assert (
            split_summary["lake_a"]["all_segment_rollout_sequences"]
            >= split_summary["lake_a"]["train_segment_rollout_sequences"]
        )
        assert split_summary["lake_a"]["supervision_pairs"] == split_summary["lake_a"]["train_pairs"]
        assert (
            split_summary["lake_a"]["supervision_segment_rollout_sequences"]
            == split_summary["lake_a"]["train_segment_rollout_sequences"]
        )
        assert split_summary["_config"]["test_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["profile_supervision_scope"] == "train"
        assert split_summary["_config"]["heldout_lake_groups"] == ["lake_b"]
        assert split_summary["_config"]["train_lake_ids"] == ["lake_a"]
        assert split_summary["_config"]["heldout_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["excluded_lake_ids"] == ["lake_b"]
        assert split_summary["_config"]["static_feature_dim"] == STATIC_FEATURE_DIM
        assert split_summary["_config"]["static_feature_keys"] == list(STATIC_FEATURE_KEYS)
        assert split_summary["lake_a"]["metadata"]["static_feature_dim"] == STATIC_FEATURE_DIM
        assert split_summary["lake_a"]["metadata"]["static_feature_keys"] == list(STATIC_FEATURE_KEYS)
        assert "reservoir_indicator" in split_summary["lake_a"]["metadata"]
        assert "shoreline_development_norm" in split_summary["lake_a"]["metadata"]
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
        assert split_summary["_config"]["shape_aware_mixing"] == "on"
        assert split_summary["_config"]["shape_mixing_strength"] == 0.35
        assert split_summary["_config"]["stratification_mixing_cap"] == "on"
        assert split_summary["_config"]["stratification_mixing_cap_strength"] == 1.0
        assert split_summary["_config"]["kd_prior_regularization_weight"] == 0.001
        assert split_summary["_config"]["lake_adaptive_params"] == "off"
        assert split_summary["_config"]["lake_adaptive_hidden_dim"] == 64
        assert split_summary["_config"]["lake_adaptive_init_spread"] == 0.02
        assert split_summary["_config"]["lake_adaptive_temporal_mode"] == "off"
        assert split_summary["_config"]["lake_adaptive_temporal_init_spread"] == 0.005
        assert split_summary["_config"]["lake_adaptive_temporal_scale"] == 0.25
        assert split_summary["_config"]["adaptive_wind_kz_min"] == 0.4
        assert split_summary["_config"]["adaptive_wind_kz_max"] == 3.0
        assert split_summary["_config"]["adaptive_blend_alpha_min"] == 0.0
        assert split_summary["_config"]["adaptive_blend_alpha_max"] == 0.6
        assert split_summary["_config"]["adaptive_kd_multiplier_min"] == 0.4
        assert split_summary["_config"]["adaptive_kd_multiplier_max"] == 2.0
        assert split_summary["_config"]["adaptive_turbulent_exchange_scale_min"] == 0.5
        assert split_summary["_config"]["adaptive_turbulent_exchange_scale_max"] == 1.8
        assert split_summary["_config"]["adaptive_convective_mixing_scale_min"] == 0.3
        assert split_summary["_config"]["adaptive_convective_mixing_scale_max"] == 2.5
        assert split_summary["_config"]["adaptive_ice_shortwave_scale_min"] == 0.4
        assert split_summary["_config"]["adaptive_ice_shortwave_scale_max"] == 1.8
        assert split_summary["_config"]["adaptive_parameter_regularization_weight"] == 0.01
        assert split_summary["_config"]["advective_heat_source_mode"] == "reservoir_simple"
        assert split_summary["_config"]["checkpoint_every_epochs"] == 1
        assert split_summary["_config"]["eval_every_epochs"] == 1
        assert split_summary["_config"]["full_eval_every_epochs"] == 60
        assert split_summary["_config"]["profile_runtime"] is True
        assert split_summary["_config"]["profile_gpu"] is True
        assert split_summary["_config"]["history_diagnostic_every_epochs"] == 0
        assert split_summary["_config"]["torch_tf32"] == "on"
        assert split_summary["_config"]["torch_matmul_precision"] == "high"
        assert split_summary["_config"]["train_diagnostic_mode"] == "loss"
        assert split_summary["_config"]["export_after_training"] == "on"
        assert split_summary["_config"]["cross_lake_batch_mode"] == "off"
        assert split_summary["_config"]["cross_lake_batch_size"] == 0
        assert split_summary["_config"]["segment_rollout_loss_weight"] == 0.05
        assert split_summary["_config"]["segment_rollout_start_epoch"] == 0
        assert split_summary["_config"]["segment_rollout_ramp_epochs"] == 1
        assert split_summary["_config"]["segment_rollout_max_days"] == 5
        assert split_summary["_config"]["segment_rollout_samples_per_lake"] == 0
        assert split_summary["_config"]["segment_rollout_lst_surface_weight"] == 0.01
        assert split_summary["_config"]["lst_feature_dropout_probability"] == 0.20
        assert split_summary["_config"]["transition_batch_size"] == 0
        assert split_summary["_config"]["segment_rollout_batch_size"] == 0
        assert split_summary["_config"]["rolling_horizon_batch_size"] == 32
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


def test_profile_supervision_scope_all_uses_all_training_lake_profiles():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            profile_supervision_scope="all",
            segment_rollout_loss_weight=0.05,
            segment_rollout_start_epoch=0,
            segment_rollout_ramp_epochs=1,
            segment_rollout_max_days=5,
            segment_rollout_samples_per_lake=0,
            state_noise_weight=0.0,
            checkpoint_every_epochs=1,
            eval_every_epochs=1,
            export_after_training="off",
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["profile_supervision_scope"] == "all"
        history = pd.read_csv(result["history_csv"])
        assert history["profile_supervision_scope"].iloc[-1] == "all"
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert split_summary["_config"]["profile_supervision_scope"] == "all"
        assert split_summary["lake_a"]["supervision_pairs"] == split_summary["lake_a"]["all_pairs"]
        assert (
            split_summary["lake_a"]["supervision_segment_rollout_sequences"]
            == split_summary["lake_a"]["all_segment_rollout_sequences"]
        )
        assert split_summary["lake_a"]["all_pairs"] >= split_summary["lake_a"]["train_pairs"]
        assert (
            split_summary["lake_a"]["all_segment_rollout_sequences"]
            >= split_summary["lake_a"]["train_segment_rollout_sequences"]
        )
        assert int(history["train_supervision_pair_count"].iloc[-1]) == split_summary["lake_a"]["all_pairs"]
        assert (
            int(history["train_supervision_segment_sequence_count"].iloc[-1])
            == split_summary["lake_a"]["all_segment_rollout_sequences"]
        )


def test_profile_runtime_defaults_on_and_manifest_can_disable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            root / "default_on",
            epochs=1,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            segment_rollout_loss_weight=0.05,
            eval_every_epochs=1,
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        history = pd.read_csv(result["history_csv"])
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert bundle["profile_runtime"] is True
        assert bundle["checkpoint_every_epochs"] == 5
        assert bundle["full_eval_every_epochs"] == 60
        assert split_summary["_config"]["profile_runtime"] is True
        assert split_summary["_config"]["checkpoint_every_epochs"] == 5
        assert split_summary["_config"]["full_eval_every_epochs"] == 60
        assert int(history["checkpoint_every_epochs"].iloc[-1]) == 5
        assert "epoch_seconds" in history.columns

        manifest["profile_runtime"] = "off"
        manifest_off_path = root / "manifest_off.json"
        manifest_off_path.write_text(json.dumps(manifest), encoding="utf-8")
        disabled = train_multilake_state_forecaster(
            manifest_off_path,
            root / "manifest_off",
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            segment_rollout_loss_weight=0.05,
            device="cpu",
        )
        disabled_bundle = torch.load(disabled["checkpoint_path"], map_location="cpu")
        disabled_split_summary = json.loads(disabled["split_summary"].read_text(encoding="utf-8"))
        assert disabled_bundle["profile_runtime"] is False
        assert disabled_split_summary["_config"]["profile_runtime"] is False


def test_multi_heldout_ids_infer_groups_exclude_group_mates_and_export_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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


def test_export_max_depth_limits_prediction_products_without_changing_internal_depth():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_config = _write_lake_inputs(root, "train_lake_2020", 0.0, lake_group="train_lake")
        deep_config = _write_lake_inputs(root, "deep_held_2020", 1.0, lake_group="deep_held")
        metadata_path = Path(deep_config["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["max_depth_m"] = 40.0
        metadata["mean_depth_m"] = 20.0
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        deep_config["max_depth"] = 40.0
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [train_config, deep_config],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        train_result = train_multilake_state_forecaster(
            manifest_path,
            root / "train",
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="deep_held_2020",
            device="cpu",
        )
        train_split = json.loads(train_result["split_summary"].read_text(encoding="utf-8"))
        assert train_split["deep_held_2020"]["metadata"]["max_depth_m"] == pytest.approx(40.0)

        export_result = train_multilake_state_forecaster(
            manifest_path,
            root / "export",
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="deep_held_2020",
            checkpoint_path=train_result["checkpoint_path"],
            export_only=True,
            export_max_depth_m=25.0,
            device="cpu",
        )
        assert export_result["export_max_depth_m"] == pytest.approx(25.0)
        export_split = json.loads(export_result["split_summary"].read_text(encoding="utf-8"))
        assert export_split["_config"]["export_max_depth_m"] == pytest.approx(25.0)
        assert export_split["deep_held_2020"]["metadata"]["max_depth_m"] == pytest.approx(40.0)
        export_info = export_result["heldout_exports"][0]
        prediction = pd.read_csv(export_info["prediction_csv"])
        assert float(prediction["Depth_m"].max()) == pytest.approx(25.0)
        assert 25.0 in set(np.round(prediction["Depth_m"].unique(), 6))
        init_summary = json.loads(Path(export_info["init_summary"]).read_text(encoding="utf-8"))
        assert init_summary["export_max_depth_m"] == pytest.approx(25.0)
        assert init_summary["effective_export_max_depth_m"] == pytest.approx(25.0)
        assert init_summary["internal_max_depth_m"] == pytest.approx(40.0)
        assert init_summary["export_depth_limited"] is True
        assert export_info["effective_export_max_depth_m"] == pytest.approx(25.0)
        assert export_info["export_depth_limited"] is True


def test_export_after_training_defaults_to_off():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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


def test_segment_rollout_loss_weight_defaults_to_mainline_value():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            epochs=1,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            eval_every_epochs=1,
            device="cpu",
        )

        history = pd.read_csv(result["history_csv"])
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert float(history["segment_rollout_loss_weight"].iloc[-1]) == 0.05
        assert bundle["segment_rollout_loss_weight"] == 0.05
        assert split_summary["_config"]["segment_rollout_loss_weight"] == 0.05
        assert float(history["segment_rollout_lst_surface_weight"].iloc[-1]) == 0.01
        assert float(history["warm_season_column_heat_content_weight"].iloc[-1]) == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT
        )
        assert float(history["warm_season_column_heat_content_quantile_low"].iloc[-1]) == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW
        )
        assert float(history["warm_season_column_heat_content_quantile_high"].iloc[-1]) == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH
        )
        assert int(history["warm_season_column_heat_content_min_gap_days"].iloc[-1]) == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS
        )
        assert float(history["lst_feature_dropout_probability"].iloc[-1]) == 0.20
        assert bundle["segment_rollout_lst_surface_weight"] == 0.01
        assert bundle["warm_season_column_heat_content_weight"] == DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT
        assert bundle["warm_season_column_heat_content_min_gap_days"] == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS
        )
        assert bundle["lst_feature_dropout_probability"] == 0.20
        assert split_summary["_config"]["segment_rollout_lst_surface_weight"] == 0.01
        assert split_summary["_config"]["warm_season_column_heat_content_weight"] == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT
        )
        assert split_summary["_config"]["warm_season_column_heat_content_quantile_low"] == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW
        )
        assert split_summary["_config"]["warm_season_column_heat_content_quantile_high"] == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH
        )
        assert split_summary["_config"]["warm_season_column_heat_content_min_gap_days"] == (
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS
        )
        assert "lake_a" in split_summary["_config"]["warm_season_column_heat_content_lake_configs"]
        assert "warm_season_column_heat_content" in split_summary["lake_a"]
        assert split_summary["_config"]["lst_feature_dropout_probability"] == 0.20


def test_zero_segment_rollout_loss_weight_is_rejected_from_function_arg():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="segment_rollout_loss_weight must be > 0.0"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.0,
                device="cpu",
            )


def test_zero_segment_rollout_loss_weight_is_rejected_from_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "segment_rollout_loss_weight": 0.0,
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="segment_rollout_loss_weight must be > 0.0"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_lst_feature_dropout_probability_is_validated_in_training_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="lst_feature_dropout_probability"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                lst_feature_dropout_probability=1.5,
                device="cpu",
            )

        manifest["lst_feature_dropout_probability"] = -0.1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="lst_feature_dropout_probability"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_manifest",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_export_max_depth_m_is_validated_in_training_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="export_max_depth_m"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_arg",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                export_max_depth_m=0.0,
                device="cpu",
            )

        manifest["export_max_depth_m"] = -1.0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="export_max_depth_m"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_manifest",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_negative_segment_rollout_lst_surface_weight_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="segment_rollout_lst_surface_weight"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_lst_surface_weight=-0.1,
                device="cpu",
            )

        manifest["segment_rollout_lst_surface_weight"] = -0.1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="segment_rollout_lst_surface_weight"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_manifest",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )


def test_kd_prior_regularization_weight_can_be_disabled_and_negative_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
            "kd_prior_regularization_weight": 0.0,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=1,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            segment_rollout_loss_weight=0.05,
            eval_every_epochs=1,
            device="cpu",
        )
        history = pd.read_csv(result["history_csv"])
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert float(history["kd_prior_regularization_weight"].iloc[-1]) == 0.0
        assert float(history["kd_prior_regularization_weighted_loss"].fillna(0.0).iloc[-1]) == 0.0
        assert split_summary["_config"]["kd_prior_regularization_weight"] == 0.0

        manifest["kd_prior_regularization_weight"] = -0.1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="kd_prior_regularization_weight"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_negative",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.05,
                device="cpu",
            )


def test_warm_column_heat_content_training_config_is_validated():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="warm_season_column_heat_content_weight"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_negative_weight",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                warm_season_column_heat_content_weight=-0.1,
                device="cpu",
            )

        manifest["warm_season_column_heat_content_quantile_low"] = 0.8
        manifest["warm_season_column_heat_content_quantile_high"] = 0.7
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="warm_season_column_heat_content_quantile_low/high"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_bad_quantile",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )

        manifest["warm_season_column_heat_content_quantile_low"] = 0.5
        manifest["warm_season_column_heat_content_quantile_high"] = 0.75
        manifest["warm_season_column_heat_content_min_gap_days"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="warm_season_column_heat_content_min_gap_days"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out_bad_gap",
                epochs=1,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                device="cpu",
            )

        manifest["warm_season_column_heat_content_min_gap_days"] = 14
        manifest["warm_season_column_heat_content_weight"] = 0.0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out_disabled",
            epochs=0,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            device="cpu",
        )
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["warm_season_column_heat_content_weight"] == 0.0


def test_cross_lake_batch_mode_on_trains_compatible_grid_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "cross_lake_batch_mode": "on",
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
            segment_rollout_start_epoch=0,
            segment_rollout_ramp_epochs=1,
            segment_rollout_loss_weight=0.01,
            segment_rollout_max_days=3,
            segment_rollout_samples_per_lake=1,
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
            segment_rollout_loss_weight=0.05,
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
            segment_rollout_loss_weight=0.05,
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


def test_eval_and_full_eval_modes_gate_heavy_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            segment_rollout_loss_weight=0.05,
            eval_every_epochs=1,
            full_eval_every_epochs=4,
            rolling_horizon_eval_max_starts=4,
            device="cpu",
        )
        history = pd.read_csv(result["history_csv"])
        assert history["eval_mode"].tolist() == ["none", "light", "light", "full"]
        assert np.isnan(history["heldout_transition_mean_rmse"].iloc[0])
        assert np.isfinite(history["heldout_transition_mean_rmse"].iloc[1:]).all()
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[0])
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[1])
        assert np.isnan(history["heldout_free_roll_mean_rmse"].iloc[2])
        assert np.isfinite(history["heldout_free_roll_mean_rmse"].iloc[3])
        assert np.isfinite(history["heldout_observed_point_mean_rmse"].iloc[3])
        assert history["heldout_observed_point_total_count"].iloc[3] > 0
        assert np.isnan(history["heldout_transition_rmse_1d"].iloc[0])
        assert np.isfinite(history["heldout_transition_rmse_1d"].iloc[1:]).all()
        assert np.isnan(history["heldout_persistence_mean_rmse"].iloc[0])
        assert np.isfinite(history["heldout_persistence_mean_rmse"].iloc[1:]).all()
        assert np.isnan(history["lake_a_train_rmse"].iloc[0])
        assert np.isfinite(history["lake_a_train_rmse"].iloc[1:]).all()
        assert np.isnan(history["val_rolling_start_rmse_1d"].iloc[0])
        assert np.isnan(history["val_rolling_start_rmse_1d"].iloc[1])
        assert np.isnan(history["val_rolling_start_rmse_1d"].iloc[2])
        assert np.isfinite(history["heldout_rolling_start_rmse_1d"].iloc[3])
        assert np.isfinite(history["heldout_free_roll_rmse_1d"].iloc[3])
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["full_eval_every_epochs"] == 4


def test_full_eval_zero_disables_training_heavy_rollouts_and_keeps_light_eval():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            epochs=3,
            depth_points=5,
            max_rollout_days=3,
            history_window_days=3,
            test_lake_id="lake_b",
            segment_rollout_loss_weight=0.05,
            eval_every_epochs=1,
            full_eval_every_epochs=0,
            checkpoint_every_epochs=0,
            device="cpu",
        )
        history = pd.read_csv(result["history_csv"])
        assert history["eval_mode"].tolist() == ["none", "light", "light"]
        assert history["history_diagnostic_enabled"].tolist() == [False, True, True]
        assert np.isnan(history["shortwave_scale_mean"].iloc[0])
        assert np.isfinite(history["shortwave_scale_mean"].iloc[1:]).all()
        assert np.isfinite(history["profile_loss"]).all()
        assert np.isfinite(history["heldout_transition_mean_rmse"].iloc[1:]).all()
        assert history["heldout_free_roll_mean_rmse"].isna().all()
        assert history["heldout_free_roll_rmse_30d"].isna().all()
        assert history["heldout_rolling_start_rmse_30d"].isna().all()
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        assert bundle["full_eval_every_epochs"] == 0
        assert split_summary["_config"]["full_eval_every_epochs"] == 0
        assert not (root / "out" / "best_by_val_rolling.pt").exists()
        assert history["best_by_val_rolling_enabled"].tolist() == [False, False, False]


def test_best_by_val_rolling_checkpoint_updates_on_full_eval():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_long_rolling_lake_inputs(root, "lake_a", 0.0),
                _write_long_rolling_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = train_multilake_state_forecaster(
            manifest_path,
            root / "out",
            epochs=2,
            depth_points=5,
            max_rollout_days=60,
            history_window_days=3,
            test_lake_id="lake_b",
            segment_rollout_loss_weight=0.05,
            eval_every_epochs=1,
            full_eval_every_epochs=2,
            rolling_horizon_eval_max_starts=4,
            rolling_horizon_batch_size=4,
            device="cpu",
        )
        best_path = root / "out" / "best_by_val_rolling.pt"
        metrics_path = root / "out" / "best_by_val_rolling_metrics.json"
        assert best_path.exists()
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["epoch"] == 1
        assert np.isfinite(metrics["score"])
        assert metrics["val_rolling_start_count_30d"] > 0
        assert metrics["val_rolling_start_count_60d"] > 0
        history = pd.read_csv(result["history_csv"])
        assert bool(history["best_by_val_rolling_checkpoint_updated"].iloc[-1])
        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        assert bundle["best_by_val_rolling_epoch"] == 1
        assert bundle["best_by_val_rolling_checkpoint_path"] == str(best_path)


def test_full_eval_every_epochs_negative_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 1.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="full_eval_every_epochs must be non-negative"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=0,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.05,
                full_eval_every_epochs=-1,
                device="cpu",
            )

        manifest["full_eval_every_epochs"] = -1
        manifest_negative_path = root / "manifest_negative.json"
        manifest_negative_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="full_eval_every_epochs must be non-negative"):
            train_multilake_state_forecaster(
                manifest_negative_path,
                root / "manifest_negative",
                epochs=0,
                depth_points=5,
                max_rollout_days=3,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.05,
                device="cpu",
            )


def test_evaluate_lake_pairs_uses_cached_lookup_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        pair = lake["pairs"]["all"][0]
        baseline = evaluate_lake_pairs(model, lake, [pair])
        start_date, end_date = pair[0], pair[1]
        lake["lookups"]["all"][start_date][:] = 999.0
        lake["lookups"]["all"][end_date][:] = -999.0

        cached = evaluate_lake_pairs(model, lake, [pair])

        assert cached == pytest.approx(baseline)


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


def test_segment_rollout_open_water_lst_weak_loss_uses_observed_valid_open_water_only():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        sequence = next(
            sequence
            for sequence in lake["segment_rollout_sequences"]["train"]
            if sequence[2]
        )
        kwargs = {
            "active_max_days": 5,
            "profile_huber_delta": 2.0,
            "task_mode": "analysis",
            "teacher_forcing_probability": 0.0,
            "state_noise_weight": 0.0,
            "residual_regularization_weight": 0.0,
            "daily_tendency_weight": 0.0,
            "residual_time_smooth_weight": 0.0,
            "physical_scale_regularization_weight": 0.0,
            "physical_scale_smoothness_weight": 0.0,
            "heat_content_transition_weight": 0.0,
            "segment_rollout_lst_surface_weight": 0.01,
            "hard_density_stability": True,
        }

        _set_open_water_lst_supervision(lake, value=24.0, quality=1.0, observed=1.0, ice=0.0)
        _, count, details = _segment_rollout_sequence_loss(model, lake, sequence, **kwargs)
        assert count > 0
        assert float(details["segment_rollout_lst_loss"]) > 0.0
        assert float(details["segment_rollout_lst_supervision_count"]) > 0.0
        assert float(details["segment_rollout_lst_weight_mean"]) == pytest.approx(1.0)
        assert float(details["segment_rollout_lst_surface_weight"]) == pytest.approx(0.01)

        for update in (
            {"observed": 0.0, "ice": 0.0, "value": 24.0},
            {"observed": 1.0, "ice": 1.0, "value": 24.0},
            {"observed": 1.0, "ice": 0.0, "value": float("nan")},
        ):
            _set_open_water_lst_supervision(lake, quality=1.0, **update)
            _, _, disabled_details = _segment_rollout_sequence_loss(model, lake, sequence, **kwargs)
            assert float(disabled_details["segment_rollout_lst_loss"]) == 0.0
            assert float(disabled_details["segment_rollout_lst_supervision_count"]) == 0.0
            assert float(disabled_details["segment_rollout_lst_weight_mean"]) == 0.0


def test_transition_lst_loss_uses_open_water_observed_lswt_only():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        pair = lake["pairs"]["train"][0]
        for row in lake["forcing_rows"]:
            row["lst_surface"] = torch.tensor([30.0], dtype=torch.float32)
            row["lswt_open_water"] = torch.tensor([float("nan")], dtype=torch.float32)
            row["lst_quality"] = torch.tensor([1.0], dtype=torch.float32)
            row["lst_observed_flag"] = torch.tensor([0.0], dtype=torch.float32)
            row["ice_mask"] = torch.tensor([0.0], dtype=torch.float32)
        _, filled_details = _transition_loss(
            model,
            lake,
            pair,
            lst_surface_weight=0.03,
            lookup_split="train",
        )
        assert float(filled_details["lst_loss"]) == 0.0

        for row in lake["forcing_rows"]:
            row["lswt_open_water"] = torch.tensor([30.0], dtype=torch.float32)
            row["lst_observed_flag"] = torch.tensor([1.0], dtype=torch.float32)
        _, observed_details = _transition_loss(
            model,
            lake,
            pair,
            lst_surface_weight=0.03,
            lookup_split="train",
        )
        assert float(observed_details["lst_loss"]) > 0.0


def test_batched_segment_rollout_loss_matches_scalar_loop():
    with tempfile.TemporaryDirectory() as tmp:
        lake, model = _single_lake_and_model(Path(tmp))
        _enable_warm_column_heat_content_for_test(lake)
        active_max_days = 5
        groups = {}
        for sequence in lake["segment_rollout_sequences"]["train"]:
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
            "warm_season_column_heat_content_weight": DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
            "warm_season_column_heat_content_min_gap_days": 1,
            "hard_density_stability": True,
        }
        scalar = [_segment_rollout_sequence_loss(model, lake, sequence, **kwargs) for sequence in sequences]
        batched = _segment_rollout_sequence_losses_for_lake(
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
        expected_detail_keys = {
            "segment_rollout_loss",
            "segment_rollout_profile_loss",
            "segment_rollout_horizon_weight_mean",
            "segment_rollout_max_target_gap_days",
            "segment_rollout_lst_loss",
            "segment_rollout_lst_supervision_count",
            "segment_rollout_lst_weight_mean",
            "segment_rollout_lst_surface_weight",
            "segment_rollout_residual_smooth_loss",
            "segment_rollout_daily_tendency_loss",
            "segment_rollout_residual_regularization_loss",
            "segment_rollout_physical_scale_regularization_loss",
            "segment_rollout_physical_scale_smoothness_loss",
            "segment_rollout_heat_content_transition_loss",
            "segment_rollout_heat_content_transition_weighted_loss",
            "segment_rollout_warm_column_heat_content_loss",
            "segment_rollout_warm_column_heat_content_weighted_loss",
            "segment_rollout_warm_column_heat_content_supervision_count",
            "segment_rollout_warm_column_heat_content_warm_factor_mean",
            "segment_rollout_warm_column_heat_content_error_c_mean",
            "segment_rollout_warm_column_heat_content_horizon14_count",
            "segment_rollout_warm_column_heat_content_horizon30_count",
            "segment_rollout_warm_column_heat_content_horizon60_count",
        }
        for _, _, details in scalar + batched:
            assert expected_detail_keys.issubset(details)
            assert float(details["segment_rollout_horizon_weight_mean"]) >= 1.0
            assert float(details["segment_rollout_max_target_gap_days"]) > 0.0


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
        for lake in lakes:
            _enable_warm_column_heat_content_for_test(lake)
        active_max_days = 5
        selected_by_lake = {}
        for lake_idx, lake in enumerate(lakes):
            groups = {}
            for sequence in lake["segment_rollout_sequences"]["train"]:
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
            "warm_season_column_heat_content_weight": DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
            "warm_season_column_heat_content_min_gap_days": 1,
            "hard_density_stability": True,
        }
        per_lake = {
            lake_idx: _segment_rollout_sequence_losses_for_lake(
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
        cross = _segment_rollout_sequence_losses_for_lakes_cross_batch(
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
            for _, _, details in cross[lake_idx]:
                assert "segment_rollout_horizon_weight_mean" in details
                assert "segment_rollout_max_target_gap_days" in details
                assert "segment_rollout_lst_loss" in details
                assert "segment_rollout_lst_supervision_count" in details
                assert "segment_rollout_lst_weight_mean" in details
                assert "segment_rollout_warm_column_heat_content_loss" in details
                assert "segment_rollout_warm_column_heat_content_weighted_loss" in details
                assert "segment_rollout_warm_column_heat_content_supervision_count" in details


def test_manifest_transition_loss_weight_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "transition_loss_weight": 0.5,
            "turbulent_flux_mode": "blend",
            "turbulent_flux_blend_alpha": 0.25,
            "shape_aware_mixing": "off",
            "shape_mixing_strength": 0.12,
            "stratification_mixing_cap": "off",
            "stratification_mixing_cap_strength": 0.25,
            "lake_adaptive_params": "both",
            "lake_adaptive_hidden_dim": 48,
            "lake_adaptive_init_spread": 0.03,
            "lake_adaptive_temporal_mode": "seasonal_forcing",
            "lake_adaptive_temporal_init_spread": 0.04,
            "lake_adaptive_temporal_scale": 0.35,
            "adaptive_wind_kz_min": 0.7,
            "adaptive_wind_kz_max": 2.2,
            "adaptive_blend_alpha_min": 0.05,
            "adaptive_blend_alpha_max": 0.4,
            "adaptive_kd_multiplier_min": 0.75,
            "adaptive_kd_multiplier_max": 1.25,
            "adaptive_turbulent_exchange_scale_min": 0.8,
            "adaptive_turbulent_exchange_scale_max": 1.2,
            "adaptive_convective_mixing_scale_min": 0.6,
            "adaptive_convective_mixing_scale_max": 1.8,
            "adaptive_ice_shortwave_scale_min": 0.7,
            "adaptive_ice_shortwave_scale_max": 1.3,
            "adaptive_parameter_regularization_weight": 0.02,
            "checkpoint_every_epochs": 3,
            "eval_every_epochs": 2,
            "full_eval_every_epochs": 4,
            "profile_runtime": True,
            "profile_gpu": True,
            "history_diagnostic_every_epochs": 7,
            "torch_tf32": "off",
            "torch_matmul_precision": "medium",
            "transition_batch_size": 2,
            "segment_rollout_batch_size": 3,
            "rolling_horizon_batch_size": 4,
            "train_diagnostic_mode": "loss",
            "export_after_training": "on",
            "cross_lake_batch_mode": "off",
            "cross_lake_batch_size": 6,
            "segment_rollout_lst_surface_weight": 0.02,
            "lst_feature_dropout_probability": 0.35,
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
        assert bundle["shape_aware_mixing"] == "off"
        assert bundle["shape_mixing_strength"] == 0.12
        assert bundle["stratification_mixing_cap"] == "off"
        assert bundle["stratification_mixing_cap_strength"] == 0.25
        assert bundle["lake_adaptive_params"] == "kz,flux"
        assert bundle["lake_adaptive_hidden_dim"] == 48
        assert bundle["lake_adaptive_init_spread"] == 0.03
        assert bundle["lake_adaptive_temporal_mode"] == "seasonal_forcing"
        assert bundle["lake_adaptive_temporal_init_spread"] == 0.04
        assert bundle["lake_adaptive_temporal_scale"] == 0.35
        assert bundle["adaptive_wind_kz_min"] == 0.7
        assert bundle["adaptive_wind_kz_max"] == 2.2
        assert bundle["adaptive_blend_alpha_min"] == 0.05
        assert bundle["adaptive_blend_alpha_max"] == 0.4
        assert bundle["adaptive_kd_multiplier_min"] == 0.75
        assert bundle["adaptive_kd_multiplier_max"] == 1.25
        assert bundle["adaptive_turbulent_exchange_scale_min"] == 0.8
        assert bundle["adaptive_turbulent_exchange_scale_max"] == 1.2
        assert bundle["adaptive_convective_mixing_scale_min"] == 0.6
        assert bundle["adaptive_convective_mixing_scale_max"] == 1.8
        assert bundle["adaptive_ice_shortwave_scale_min"] == 0.7
        assert bundle["adaptive_ice_shortwave_scale_max"] == 1.3
        assert bundle["adaptive_parameter_regularization_weight"] == 0.02
        assert bundle["checkpoint_every_epochs"] == 3
        assert bundle["eval_every_epochs"] == 2
        assert bundle["full_eval_every_epochs"] == 4
        assert bundle["profile_runtime"] is True
        assert bundle["profile_gpu"] is True
        assert bundle["history_diagnostic_every_epochs"] == 7
        assert bundle["torch_tf32"] == "off"
        assert bundle["torch_matmul_precision"] == "medium"
        assert bundle["transition_batch_size"] == 2
        assert bundle["segment_rollout_batch_size"] == 3
        assert bundle["rolling_horizon_batch_size"] == 4
        assert bundle["train_diagnostic_mode"] == "loss"
        assert bundle["export_after_training"] == "on"
        assert bundle["cross_lake_batch_mode"] == "off"
        assert bundle["cross_lake_batch_size"] == 6
        assert bundle["segment_rollout_lst_surface_weight"] == 0.02
        assert bundle["lst_feature_dropout_probability"] == 0.35
        assert split_summary["_config"]["transition_loss_weight"] == 0.5
        assert split_summary["_config"]["turbulent_flux_mode"] == "blend"
        assert split_summary["_config"]["turbulent_flux_blend_alpha"] == 0.25
        assert split_summary["_config"]["shape_aware_mixing"] == "off"
        assert split_summary["_config"]["shape_mixing_strength"] == 0.12
        assert split_summary["_config"]["stratification_mixing_cap"] == "off"
        assert split_summary["_config"]["stratification_mixing_cap_strength"] == 0.25
        assert split_summary["_config"]["lake_adaptive_params"] == "kz,flux"
        assert split_summary["_config"]["lake_adaptive_hidden_dim"] == 48
        assert split_summary["_config"]["lake_adaptive_init_spread"] == 0.03
        assert split_summary["_config"]["lake_adaptive_temporal_mode"] == "seasonal_forcing"
        assert split_summary["_config"]["lake_adaptive_temporal_init_spread"] == 0.04
        assert split_summary["_config"]["lake_adaptive_temporal_scale"] == 0.35
        assert split_summary["_config"]["adaptive_wind_kz_min"] == 0.7
        assert split_summary["_config"]["adaptive_wind_kz_max"] == 2.2
        assert split_summary["_config"]["adaptive_blend_alpha_min"] == 0.05
        assert split_summary["_config"]["adaptive_blend_alpha_max"] == 0.4
        assert split_summary["_config"]["adaptive_kd_multiplier_min"] == 0.75
        assert split_summary["_config"]["adaptive_kd_multiplier_max"] == 1.25
        assert split_summary["_config"]["adaptive_turbulent_exchange_scale_min"] == 0.8
        assert split_summary["_config"]["adaptive_turbulent_exchange_scale_max"] == 1.2
        assert split_summary["_config"]["adaptive_convective_mixing_scale_min"] == 0.6
        assert split_summary["_config"]["adaptive_convective_mixing_scale_max"] == 1.8
        assert split_summary["_config"]["adaptive_ice_shortwave_scale_min"] == 0.7
        assert split_summary["_config"]["adaptive_ice_shortwave_scale_max"] == 1.3
        assert split_summary["_config"]["adaptive_parameter_regularization_weight"] == 0.02
        assert split_summary["_config"]["checkpoint_every_epochs"] == 3
        assert split_summary["_config"]["eval_every_epochs"] == 2
        assert split_summary["_config"]["full_eval_every_epochs"] == 4
        assert split_summary["_config"]["profile_runtime"] is True
        assert split_summary["_config"]["profile_gpu"] is True
        assert split_summary["_config"]["history_diagnostic_every_epochs"] == 7
        assert split_summary["_config"]["torch_tf32"] == "off"
        assert split_summary["_config"]["torch_matmul_precision"] == "medium"
        assert split_summary["_config"]["transition_batch_size"] == 2
        assert split_summary["_config"]["segment_rollout_batch_size"] == 3
        assert split_summary["_config"]["rolling_horizon_batch_size"] == 4
        assert split_summary["_config"]["train_diagnostic_mode"] == "loss"
        assert split_summary["_config"]["export_after_training"] == "on"
        assert split_summary["_config"]["cross_lake_batch_mode"] == "off"
        assert split_summary["_config"]["cross_lake_batch_size"] == 6
        assert split_summary["_config"]["segment_rollout_lst_surface_weight"] == 0.02
        assert split_summary["_config"]["lst_feature_dropout_probability"] == 0.35


def test_lake_adaptive_flux_requires_blend_in_training_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="requires turbulent_flux_mode='blend'"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=5,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.05,
                lake_adaptive_params="flux",
                turbulent_flux_mode="bulk",
                device="cpu",
            )


def test_lake_adaptive_exchange_rejects_provided_flux_mode():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "split_mode": "time_blocked",
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="requires turbulent_flux_mode='bulk' or 'blend'"):
            train_multilake_state_forecaster(
                manifest_path,
                root / "out",
                epochs=1,
                depth_points=5,
                max_rollout_days=5,
                history_window_days=3,
                test_lake_id="lake_b",
                segment_rollout_loss_weight=0.05,
                lake_adaptive_params="exchange",
                turbulent_flux_mode="provided",
                device="cpu",
            )


@pytest.mark.parametrize(
    ("adaptive_mode", "turbulent_mode", "blend_alpha"),
    [
        ("kz", "bulk", 0.3),
        ("flux", "blend", 0.2),
        ("kd,exchange,convective,ice", "bulk", 0.3),
        ("all", "blend", 0.2),
    ],
)
def test_lake_adaptive_training_smoke_records_config(adaptive_mode, turbulent_mode, blend_alpha):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            segment_rollout_loss_weight=0.05,
            lake_adaptive_params=adaptive_mode,
            adaptive_parameter_regularization_weight=0.02,
            turbulent_flux_mode=turbulent_mode,
            turbulent_flux_blend_alpha=blend_alpha,
            export_after_training="off",
            eval_every_epochs=1,
            device="cpu",
        )

        bundle = torch.load(result["checkpoint_path"], map_location="cpu")
        history = pd.read_csv(result["history_csv"])
        split_summary = json.loads(result["split_summary"].read_text(encoding="utf-8"))
        expected_mode = "kz,flux,kd,exchange,convective,ice" if adaptive_mode == "all" else adaptive_mode
        assert bundle["lake_adaptive_params"] == expected_mode
        assert bundle["adaptive_parameter_regularization_weight"] == 0.02
        assert split_summary["_config"]["lake_adaptive_params"] == expected_mode
        assert split_summary["_config"]["adaptive_parameter_regularization_weight"] == 0.02
        assert history["lake_adaptive_params"].iloc[-1] == expected_mode
        assert np.isfinite(history["adaptive_parameter_regularization_loss"].iloc[-1])
        assert np.isfinite(history["adaptive_wind_kz_scale_mean"].iloc[-1])
        assert np.isfinite(history["adaptive_turbulent_flux_blend_alpha_mean"].iloc[-1])
        assert np.isfinite(history["adaptive_kd_multiplier_mean"].iloc[-1])
        assert np.isfinite(history["adaptive_turbulent_exchange_scale_mean"].iloc[-1])
        assert np.isfinite(history["adaptive_convective_mixing_scale_mean"].iloc[-1])
        assert np.isfinite(history["adaptive_ice_shortwave_scale_mean"].iloc[-1])


def test_zero_transition_loss_weight_with_segment_rollout_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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
            segment_rollout_start_epoch=0,
            segment_rollout_ramp_epochs=1,
            segment_rollout_samples_per_lake=0,
            teacher_forcing_start=0.5,
            teacher_forcing_end=0.0,
            state_noise_weight=0.0,
            transition_batch_size=2,
            segment_rollout_batch_size=2,
            device="cpu",
        )

        history = pd.read_csv(result["history_csv"])
        assert float(history["transition_loss_weight"].iloc[-1]) == 0.0
        assert float(history["transition_loss_weighted"].iloc[-1]) == 0.0
        assert int(history["transition_batch_size"].iloc[-1]) == 2
        assert int(history["segment_rollout_batch_size"].iloc[-1]) == 2
        assert float(history["segment_rollout_supervision_count"].fillna(0.0).max()) > 0.0


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


def test_removed_forecast_start_manifest_field_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diagnostic_mode", "full", "diagnostic_mode was removed"),
        ("light_eval_every_epochs", 2, "light_eval_every_epochs was removed"),
        ("task_mode", "analysis", "task/data mode fields were removed"),
        ("data_fill_mode", "reconstruction", "task/data mode fields were removed"),
        ("long_free_roll_loss_weight", 0.1, "long_free_roll fields were removed"),
        ("long_free_roll_start_epoch", 0, "long_free_roll fields were removed"),
        ("long_free_roll_ramp_epochs", 1, "long_free_roll fields were removed"),
        ("long_free_roll_max_days", 30, "long_free_roll fields were removed"),
        ("long_free_roll_samples_per_lake", 4, "long_free_roll fields were removed"),
        ("transition_batch_mode", "on", "batch/debug mode fields were removed"),
        ("segment_rollout_batch_mode", "on", "batch/debug mode fields were removed"),
        ("rolling_horizon_batch_mode", "on", "batch/debug mode fields were removed"),
        ("rollout_batch_step_mode", "on", "batch/debug mode fields were removed"),
        ("step_forcing_mode", "tensor", "batch/debug mode fields were removed"),
        ("free_roll_loss_weight", 0.1, "short free-roll loss fields were removed"),
        ("free_roll_horizons", "3,7", "short free-roll loss fields were removed"),
        ("free_roll_supervision_mode", "observed", "short free-roll loss fields were removed"),
    ],
)
def test_removed_manifest_fields_are_rejected(field, value, message):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            field: value,
            "lakes": [
                _write_lake_inputs(root, "lake_a", 0.0),
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("long_free_roll_loss_weight", 0.1, "long_free_roll fields were removed"),
        ("long_free_roll_max_days", 30, "long_free_roll fields were removed"),
        ("free_roll_loss_weight", 0.1, "short free-roll loss fields were removed"),
        ("free_roll_horizons", "3,7", "short free-roll loss fields were removed"),
        ("transition_batch_mode", "on", "batch/debug mode fields were removed"),
        ("segment_rollout_batch_mode", "on", "batch/debug mode fields were removed"),
    ],
)
def test_removed_lake_level_manifest_fields_are_rejected(field, value, message):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lake_a = _write_lake_inputs(root, "lake_a", 0.0)
        lake_a[field] = value
        manifest = {
            "lakes": [
                lake_a,
                _write_lake_inputs(root, "lake_b", 2.0),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
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


def test_depth_limited_export_grid_clips_deep_lake_and_interpolates_boundary():
    depths = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    temp_grid = np.stack([
        depths,
        depths + 1.0,
    ], axis=1).astype(np.float32)

    export_grid, export_depths, info = _depth_limited_export_grid(
        temp_grid,
        depths,
        export_max_depth_m=25.0,
    )

    assert export_depths.tolist() == pytest.approx([0.0, 10.0, 20.0, 25.0])
    assert export_grid.shape == (4, 2)
    assert export_grid[-1, 0] == pytest.approx(25.0)
    assert export_grid[-1, 1] == pytest.approx(26.0)
    assert info["export_max_depth_m"] == pytest.approx(25.0)
    assert info["effective_export_max_depth_m"] == pytest.approx(25.0)
    assert info["internal_max_depth_m"] == pytest.approx(40.0)
    assert info["export_depth_limited"] is True


def test_depth_limited_export_grid_does_not_extrapolate_shallow_lake():
    depths = np.asarray([0.0, 2.0, 4.0, 8.0], dtype=np.float32)
    temp_grid = np.stack([
        10.0 - depths,
        11.0 - depths,
    ], axis=1).astype(np.float32)

    export_grid, export_depths, info = _depth_limited_export_grid(
        temp_grid,
        depths,
        export_max_depth_m=25.0,
    )

    assert export_depths.tolist() == pytest.approx(depths.tolist())
    assert np.allclose(export_grid, temp_grid)
    assert info["export_max_depth_m"] == pytest.approx(25.0)
    assert info["effective_export_max_depth_m"] == pytest.approx(8.0)
    assert info["export_depth_limited"] is False


def test_depth_limited_export_grid_default_preserves_full_depth_and_validates_cap():
    depths = np.asarray([0.0, 5.0, 10.0], dtype=np.float32)
    temp_grid = np.ones((3, 2), dtype=np.float32)

    export_grid, export_depths, info = _depth_limited_export_grid(temp_grid, depths)

    assert np.allclose(export_grid, temp_grid)
    assert export_depths.tolist() == pytest.approx(depths.tolist())
    assert info["export_max_depth_m"] is None
    assert info["effective_export_max_depth_m"] == pytest.approx(10.0)
    assert info["export_depth_limited"] is False
    with pytest.raises(ValueError, match="export_max_depth_m"):
        _depth_limited_export_grid(temp_grid, depths, export_max_depth_m=0.0)


def test_warm_column_heat_content_quantiles_use_lake_internal_distribution():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0])
    area = torch.ones_like(depths)
    lookup = {
        pd.Timestamp("2020-01-01"): torch.tensor([4.0, 4.0, 4.0, 4.0]),
        pd.Timestamp("2020-04-01"): torch.tensor([8.0, 8.0, 8.0, 8.0]),
        pd.Timestamp("2020-07-01"): torch.tensor([16.0, 16.0, 16.0, 16.0]),
        pd.Timestamp("2020-10-01"): torch.tensor([20.0, 20.0, 20.0, 20.0]),
    }
    masks = {date: torch.ones(4, dtype=torch.bool) for date in lookup}

    quantiles = _warm_column_heat_content_quantiles(
        lookup,
        masks,
        depths,
        area,
        quantile_low=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
        quantile_high=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    )

    assert quantiles["count"] == 4
    assert quantiles["low_c"] == pytest.approx(12.0)
    assert quantiles["high_c"] == pytest.approx(17.0)

    lake = {
        "depths": depths,
        "area": area,
        "warm_season_column_heat_content_low_c": quantiles["low_c"],
        "warm_season_column_heat_content_high_c": quantiles["high_c"],
    }
    pred = torch.tensor([
        [4.0, 4.0, 4.0, 4.0],
        [14.0, 14.0, 14.0, 14.0],
        [22.0, 22.0, 22.0, 22.0],
    ])
    target = torch.tensor([
        [4.0, 4.0, 4.0, 4.0],
        [16.0, 16.0, 16.0, 16.0],
        [20.0, 20.0, 20.0, 20.0],
    ])
    raw, weighted, warm_factor, _, active = _warm_column_heat_content_loss_vector(
        pred,
        target,
        lake,
        end_mask=torch.ones_like(target, dtype=torch.bool),
        target_gap_days=torch.tensor([14.0, 14.0, 14.0]),
        horizon_weight=torch.ones(3),
        weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
        min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
        profile_huber_delta=2.0,
    )

    assert not bool(active[0])
    assert bool(active[1])
    assert bool(active[2])
    assert float(warm_factor[0]) == 0.0
    assert 0.0 < float(warm_factor[1]) < 1.0
    assert float(warm_factor[2]) == pytest.approx(1.0)
    assert float(raw[1]) > 0.0
    assert float(weighted[2]) > 0.0


def test_warm_column_heat_content_loss_uses_observed_depth_mask_for_shallow_profiles():
    depths = torch.tensor([0.0, 10.0, 20.0, 30.0])
    area = torch.ones_like(depths)
    lake = {
        "depths": depths,
        "area": area,
        "warm_season_column_heat_content_low_c": 8.0,
        "warm_season_column_heat_content_high_c": 10.0,
    }
    target = torch.tensor([[12.0, 10.0, 4.0, 4.0]])
    pred_deep_wrong = torch.tensor([[12.0, 10.0, 30.0, 30.0]])
    pred_observed_wrong = torch.tensor([[16.0, 14.0, 4.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])

    observed_mean, valid = _column_mean_temperature_c_vector(
        target,
        depths,
        area,
        mask=mask,
        min_full_column_coverage=0.95,
    )
    assert bool(valid[0])
    assert float(observed_mean[0]) == pytest.approx((12.0 * 5.0 + 10.0 * 10.0) / 15.0)

    _, weighted_same_observed, _, _, active = _warm_column_heat_content_loss_vector(
        pred_deep_wrong,
        target,
        lake,
        end_mask=mask,
        target_gap_days=torch.tensor([30.0]),
        horizon_weight=torch.tensor([2.0]),
        weight=0.03,
        min_gap_days=14,
        profile_huber_delta=2.0,
        min_full_column_coverage=0.95,
    )
    _, weighted_observed_wrong, _, _, active_wrong = _warm_column_heat_content_loss_vector(
        pred_observed_wrong,
        target,
        lake,
        end_mask=mask,
        target_gap_days=torch.tensor([30.0]),
        horizon_weight=torch.tensor([2.0]),
        weight=0.03,
        min_gap_days=14,
        profile_huber_delta=2.0,
        min_full_column_coverage=0.95,
    )

    assert bool(active[0])
    assert float(weighted_same_observed[0]) == 0.0
    assert bool(active_wrong[0])
    assert float(weighted_observed_wrong[0]) > 0.0


def test_warm_column_heat_content_gap_weight_and_horizon_counts():
    depths = torch.tensor([0.0, 1.0, 2.0])
    area = torch.ones_like(depths)
    lake = {
        "depths": depths,
        "area": area,
        "warm_season_column_heat_content_low_c": 5.0,
        "warm_season_column_heat_content_high_c": 10.0,
    }
    target = torch.full((4, 3), 12.0)
    prediction = target - 3.0
    mask = torch.ones_like(target, dtype=torch.bool)
    gaps = torch.tensor([13.0, 14.0, 30.0, 60.0])
    horizon = torch.tensor([1.0, 1.5, 2.0, 3.0])

    raw_zero, weighted_zero, _, _, active_zero = _warm_column_heat_content_loss_vector(
        prediction,
        target,
        lake,
        end_mask=mask,
        target_gap_days=gaps,
        horizon_weight=horizon,
        weight=0.0,
        min_gap_days=14,
    )
    raw, weighted, warm_factor, error_c, active = _warm_column_heat_content_loss_vector(
        prediction,
        target,
        lake,
        end_mask=mask,
        target_gap_days=gaps,
        horizon_weight=horizon,
        weight=0.03,
        min_gap_days=14,
    )

    assert torch.all(raw_zero == 0.0)
    assert torch.all(weighted_zero == 0.0)
    assert not torch.any(active_zero)
    assert [bool(item) for item in active] == [False, True, True, True]
    details = _warm_column_heat_content_loss_details(
        [raw[idx] for idx in torch.nonzero(active, as_tuple=False).reshape(-1)],
        [weighted[idx] for idx in torch.nonzero(active, as_tuple=False).reshape(-1)],
        [warm_factor[idx] for idx in torch.nonzero(active, as_tuple=False).reshape(-1)],
        [error_c[idx] for idx in torch.nonzero(active, as_tuple=False).reshape(-1)],
        [gaps[idx] for idx in torch.nonzero(active, as_tuple=False).reshape(-1)],
        device=prediction.device,
        prefix="segment_rollout_",
    )[2]

    assert float(details["segment_rollout_warm_column_heat_content_supervision_count"]) == 3.0
    assert float(details["segment_rollout_warm_column_heat_content_horizon14_count"]) == 3.0
    assert float(details["segment_rollout_warm_column_heat_content_horizon30_count"]) == 2.0
    assert float(details["segment_rollout_warm_column_heat_content_horizon60_count"]) == 1.0


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
        metadata.update({
            "light_extinction_kd": 0.9,
            "effective_fetch_m": 12345.0,
            "is_reservoir": True,
            "retention_time_days": 45.0,
            "shore_len_km": 12.0,
            "shoreline_development": 1.8,
            "watershed_area_km2": 250.0,
            "mean_discharge_m3_s": 6.5,
        })
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
        assert loaded["static_features"].numel() == STATIC_FEATURE_DIM
        assert loaded["metadata"]["reservoir_indicator"] == 1.0
        assert loaded["metadata"]["residence_time_days"] == 45.0
        assert loaded["metadata"]["shoreline_length_km"] == 12.0
        assert loaded["metadata"]["shoreline_development"] == 1.8
        assert loaded["metadata"]["catchment_area_km2"] == 250.0
        assert loaded["metadata"]["discharge_m3_s"] == 6.5
        assert loaded["metadata"]["residence_time_norm"] > 0.0
        assert loaded["metadata"]["catchment_area_norm"] > 0.0


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


def test_rolling_start_horizon_evaluates_specified_lake_list_without_fixed_count():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lakes = [
            prepare_lake_state_data(
                _write_lake_inputs(root, f"heldout_{idx}", float(idx)),
                split_mode="none",
                depth_points=5,
                max_rollout_days=5,
                history_window_days=3,
                device="cpu",
            )
            for idx in range(4)
        ]
        model = LakeStateForecaster(
            lakes[0]["depths"],
            lakes[0]["area"],
            static_dim=lakes[0]["static_features"].numel(),
            hidden_dim=16,
            forcing_context_dim=8,
            forcing_history_hidden_dim=8,
            residual_limit_c=0.25,
            turbulent_flux_mode="bulk",
        )
        metrics = evaluate_lakes_rolling_start_horizons(
            model,
            lakes,
            horizons=(1, 3),
            task_mode="analysis",
            max_start_profiles=0,
            hard_density_stability=True,
            batch_size=2,
            use_batched=True,
        )
        assert list(metrics) == [lake["lake_id"] for lake in lakes]
        for lake in lakes:
            scalar = evaluate_lake_rolling_start_horizons(
                model,
                lake,
                horizons=(1, 3),
                task_mode="analysis",
                max_start_profiles=0,
                hard_density_stability=True,
            )
            assert metrics[lake["lake_id"]]["count_1d"] == scalar["count_1d"]
            assert metrics[lake["lake_id"]]["rmse_3d"] == pytest.approx(scalar["rmse_3d"], abs=1e-5)


def test_heldout_free_rolls_match_scalar_for_compatible_lakes():
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
        )
        for lake_id in scalar:
            for key in ("rmse", "mae", "bias", "n_profiles"):
                if key == "n_profiles":
                    assert batched[lake_id][key] == scalar[lake_id][key]
                else:
                    assert batched[lake_id][key] == pytest.approx(scalar[lake_id][key], abs=1e-5)
            assert batched[lake_id]["observed_point_count"] == scalar[lake_id]["observed_point_count"]
            assert batched[lake_id]["observed_point_count"] > 0
            assert batched[lake_id]["observed_point_rmse"] == pytest.approx(
                scalar[lake_id]["observed_point_rmse"],
                abs=1e-5,
            )
            assert batched[lake_id]["observed_point_bias"] == pytest.approx(
                scalar[lake_id]["observed_point_bias"],
                abs=1e-5,
            )
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
    test_removed_forecast_start_manifest_field_is_rejected()
    test_heat_content_transition_loss_prefers_full_column_when_coverage_is_high()
    test_heat_content_transition_loss_handles_empty_common_mask()
    test_rollout_diagnostic_summaries_tolerate_missing_values()
    test_metadata_override_updates_forcing_defaults()
    test_profile_lookup_returns_depth_mask_without_deep_extrapolation()
    test_rolling_start_horizon_reports_all_observation_starts()
    test_batched_rolling_start_horizon_matches_scalar_evaluation()
    test_rolling_start_horizon_evaluates_specified_lake_list_without_fixed_count()
    test_heldout_free_rolls_match_scalar_for_compatible_lakes()
    print("multi-lake state forecaster sanity checks passed")
