import numpy as np
import pandas as pd
import pytest
import torch

from lake_pinn.data_io import add_past_forcing_memory_features
from lake_pinn.state_model import (
    FORCING_FEATURE_COLUMNS,
    LakeStateForecaster,
    MULTITASK_AUXILIARY_STATE_KEYS,
    STATIC_FEATURE_DIM,
    STATIC_FEATURE_KEYS,
    static_feature_array,
)
from lake_pinn.state_reconstruction import (
    _forcing_tensor_rows,
    _zero_profile_init_conditioning_from_inputs,
    fit_zero_profile_eof_pca_basis,
    initialize_rollout_state,
    normalize_lswt_observer_mode,
    normalize_zero_profile_thermal_basis_balance_mode,
    normalize_zero_profile_initializer_mode,
    zero_profile_thermal_basis_tensors_for_depths,
)
from lake_pinn.state_multilake import (
    DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE,
    DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
    DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS,
    DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
    DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT,
    DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
    DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
    DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
    DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT,
    DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT,
    DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS,
    UNLABELED_HEAT_CLOSURE_MODES,
    _build_unlabeled_heat_closure_windows,
    _build_unlabeled_heat_closure_windows_by_horizon,
    _clip_grad_norm_finite,
    _daily_memory_basis_tensors,
    _daily_memory_heat_and_physics_terms,
    _daily_memory_prediction_batch,
    _daily_memory_reconstruction_training_records,
    _model_mainline_config_fields,
    _multitask_auxiliary_loss_vector,
    _multitask_auxiliary_targets,
    _normalize_model_mainline,
    _normalize_prediction_branch,
    _normalize_thermal_state_profile_fusion_mode,
    _normalize_thermal_state_profile_fusion_time_policy,
    _normalize_unlabeled_heat_closure_mode,
    _parse_unlabeled_heat_closure_horizons,
    _resolve_physics_rollout_thermal_state_weight,
    _resolve_zero_profile_thermal_basis,
    _resolve_model_mainline,
    _select_unlabeled_heat_closure_windows,
    _storage_budget_residual_loss,
    _thermal_state_profile_fusion_batch,
    _unlabeled_heat_closure_loss_for_lake,
    _zero_profile_init_conditioning_array,
    _zero_profile_init_net_training_records,
    ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES,
    export_daily_memory_reconstruction,
)


def test_unlabeled_heat_closure_defaults_use_external_budget_mainline():
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT == pytest.approx(0.0)
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE == 4
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS == (1, 7, 30)
    assert _parse_unlabeled_heat_closure_horizons(None) == (1, 7, 30)
    assert _parse_unlabeled_heat_closure_horizons('1, 7;30') == (1, 7, 30)
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_MODE == 'storage_budget_thresholded'
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE == 'spinup_then_window'
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS == 90
    assert DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT == pytest.approx(0.0003)
    assert DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS == 90
    assert DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT == pytest.approx(0.02)
    assert DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT == pytest.approx(0.0)
    assert DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS == 60
    assert DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS == 2
    assert 'solver_residual_diagnostic' not in UNLABELED_HEAT_CLOSURE_MODES


def test_legacy_solver_residual_heat_closure_mode_is_rejected():
    with pytest.raises(ValueError):
        _normalize_unlabeled_heat_closure_mode('solver_residual_diagnostic')
    with pytest.raises(ValueError):
        _normalize_unlabeled_heat_closure_mode('solver_residual')


def test_ensemble_prediction_branch_is_rejected():
    with pytest.raises(ValueError):
        _normalize_prediction_branch('ensemble')


def test_model_mainline_resolves_to_supported_mainlines():
    assert _normalize_model_mainline('init-physics-rollout') == 'init_physics_rollout'
    assert _normalize_model_mainline('daily-memory') == 'daily_memory'
    assert _resolve_model_mainline(
        'auto',
        daily_memory_reconstruction_weight=0.0,
        prediction_branch='physics_rollout',
    ) == 'init_physics_rollout'
    assert _resolve_model_mainline(
        'auto',
        daily_memory_reconstruction_weight=0.25,
        prediction_branch='physics_rollout',
    ) == 'init_physics_rollout'
    assert _resolve_model_mainline(
        'auto',
        daily_memory_reconstruction_weight=0.25,
        prediction_branch='daily_memory',
    ) == 'daily_memory'
    with pytest.raises(ValueError):
        _resolve_model_mainline(
            'auto',
            daily_memory_reconstruction_weight=0.0,
            prediction_branch='daily_memory',
        )
    fields = _model_mainline_config_fields(
        'auto',
        'init_physics_rollout',
        daily_memory_reconstruction_weight=0.25,
        prediction_branch='physics_rollout',
    )
    assert fields['model_mainline_branch_count'] == 2
    assert fields['model_mainline_branch_names'] == 'init_physics_rollout,daily_memory'
    assert fields['model_mainline_physics_primary'] is True
    assert fields['daily_memory_training_role'] == 'auxiliary'
    assert fields['physics_rollout_thermal_state_enabled'] is True
    assert fields['physics_rollout_thermal_state_weight'] == pytest.approx(
        DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT
    )
    assert fields['physics_rollout_thermal_state_weight_eff'] == pytest.approx(
        DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT
    )
    assert fields['profile_observations_role'] == 'training_supervision_only'
    assert fields['era5_lst_role'] == 'inference_forcing'
    fusion_fields = _model_mainline_config_fields(
        'auto',
        'init_physics_rollout',
        daily_memory_reconstruction_weight=0.0,
        prediction_branch='physics_rollout',
        thermal_state_profile_fusion_mode='init_only',
    )
    assert fusion_fields['profile_observations_role'] == (
        'profile_conditioned_low_dimensional_state_fusion'
    )
    daily_fields = _model_mainline_config_fields(
        'auto',
        'daily_memory',
        daily_memory_reconstruction_weight=0.25,
        prediction_branch='daily_memory',
    )
    assert daily_fields['daily_memory_training_role'] == 'prediction'
    assert daily_fields['model_mainline_physics_primary'] is True
    assert daily_fields['physics_rollout_thermal_state_enabled'] is False
    assert daily_fields['physics_rollout_thermal_state_weight_eff'] == pytest.approx(0.0)


def test_physics_rollout_thermal_state_weight_defaults_to_mainline_auxiliary():
    effective, physics_eff = _resolve_physics_rollout_thermal_state_weight(
        'init_physics_rollout',
        requested_weight=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
        multitask_auxiliary_weight=0.0,
    )
    assert effective == pytest.approx(DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT)
    assert physics_eff == pytest.approx(DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT)

    explicit, physics_eff = _resolve_physics_rollout_thermal_state_weight(
        'init_physics_rollout',
        requested_weight=0.01,
        multitask_auxiliary_weight=0.08,
    )
    assert explicit == pytest.approx(0.08)
    assert physics_eff == pytest.approx(0.08)

    daily_effective, daily_physics_eff = _resolve_physics_rollout_thermal_state_weight(
        'daily_memory',
        requested_weight=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
        multitask_auxiliary_weight=0.0,
    )
    assert daily_effective == pytest.approx(0.0)
    assert daily_physics_eff == pytest.approx(0.0)


def test_explicit_model_mainline_rejects_conflicting_branch_config():
    assert _resolve_model_mainline(
        'init_physics_rollout',
        daily_memory_reconstruction_weight=0.1,
        prediction_branch='physics_rollout',
    ) == 'init_physics_rollout'
    with pytest.raises(ValueError):
        _resolve_model_mainline(
            'init_physics_rollout',
            daily_memory_reconstruction_weight=0.0,
            prediction_branch='ensemble',
        )
    with pytest.raises(ValueError):
        _resolve_model_mainline(
            'daily_memory',
            daily_memory_reconstruction_weight=0.0,
            prediction_branch='daily_memory',
        )
    with pytest.raises(ValueError):
        _resolve_model_mainline(
            'daily_memory',
            daily_memory_reconstruction_weight=0.1,
            prediction_branch='physics_rollout',
        )


def test_thermal_state_profile_fusion_normalizes_future_profile_policy():
    assert DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY == 'past_strict'
    assert DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT == 'train'
    assert _normalize_thermal_state_profile_fusion_mode('on') == 'both'
    assert _normalize_thermal_state_profile_fusion_mode('init') == 'init_only'
    assert _normalize_thermal_state_profile_fusion_mode('daily') == 'daily_only'
    assert _normalize_thermal_state_profile_fusion_time_policy(None) == 'past_strict'
    assert _normalize_thermal_state_profile_fusion_time_policy('future_allowed') == 'nearest'
    assert _normalize_thermal_state_profile_fusion_time_policy('nearest_no_same_day') == 'nearest_strict'
    with pytest.raises(ValueError):
        _normalize_thermal_state_profile_fusion_mode('legacy_surface')
    with pytest.raises(ValueError):
        _normalize_thermal_state_profile_fusion_time_policy('future_only')


def test_uniform_lst_debug_init_mode_is_rejected():
    with pytest.raises(ValueError):
        initialize_rollout_state(
            model=None,
            df=pd.DataFrame(),
            depths=np.asarray([0.0, 1.0], dtype=np.float32),
            all_lookup={},
            forcing_rows=[],
            static_features=torch.zeros(STATIC_FEATURE_DIM),
            metadata={},
            device=torch.device('cpu'),
            init_mode='uniform_lst_debug',
        )


def _toy_profiles(depths):
    dates = pd.to_datetime(['2020-01-01', '2020-01-11', '2020-01-21'])
    return {
        pd.Timestamp(date).normalize(): np.linspace(4.0 + idx, 6.0 + idx, len(depths)).astype(np.float32)
        for idx, date in enumerate(dates)
    }


def _toy_basis(depths, profiles, n_components=3):
    return fit_zero_profile_eof_pca_basis(
        [{'lake_id': 'toy', 'depths': depths, 'lookup': profiles, 'masks': {}}],
        n_components=n_components,
        grid_points=12,
    )


def test_zero_profile_thermal_basis_balance_mode_is_transfer_default():
    assert normalize_zero_profile_thermal_basis_balance_mode(None) == 'lake_season_depth_coverage'
    assert normalize_zero_profile_thermal_basis_balance_mode('transferable') == 'lake_season_depth_coverage'
    assert normalize_zero_profile_thermal_basis_balance_mode('off') == 'off'
    with pytest.raises(ValueError):
        normalize_zero_profile_thermal_basis_balance_mode('lake_id_embedding')


def test_zero_profile_thermal_basis_balances_lake_season_depth_coverage():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    warm_profiles = {
        pd.Timestamp('2020-07-01') + pd.Timedelta(days=idx): (
            np.linspace(23.0, 21.0, len(depths)) + 0.02 * idx
        ).astype(np.float32)
        for idx in range(20)
    }
    cold_profiles = {
        pd.Timestamp('2020-01-01') + pd.Timedelta(days=10 * idx): (
            np.linspace(4.0, 5.0, len(depths)) + 0.05 * idx
        ).astype(np.float32)
        for idx in range(2)
    }
    sources = [
        {'lake_id': 'data_rich_warm_lake', 'depths': depths, 'lookup': warm_profiles, 'masks': {}},
        {'lake_id': 'sparse_cold_lake', 'depths': depths, 'lookup': cold_profiles, 'masks': {}},
    ]
    plain = fit_zero_profile_eof_pca_basis(
        sources,
        n_components=2,
        grid_points=12,
        balance_mode='off',
    )
    balanced = fit_zero_profile_eof_pca_basis(
        sources,
        n_components=2,
        grid_points=12,
        balance_mode='lake_season_depth_coverage',
    )
    assert plain is not None
    assert balanced is not None
    assert balanced['basis_balance_mode'] == 'lake_season_depth_coverage'
    assert balanced['profile_weight_max'] > balanced['profile_weight_min']
    assert balanced['season_profile_counts']['summer'] == 20
    assert balanced['season_profile_counts']['winter'] == 2
    warm_surface = float(np.mean([profile[0] for profile in warm_profiles.values()]))
    balanced_surface = float(np.asarray(balanced['mean_profile_c'], dtype=np.float32)[0])
    plain_surface = float(np.asarray(plain['mean_profile_c'], dtype=np.float32)[0])
    assert abs(balanced_surface - warm_surface) > abs(plain_surface - warm_surface)
    assert balanced_surface < plain_surface


def _toy_forcing_frame(days=21):
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    doy = dates.dayofyear.to_numpy(dtype=np.float32)
    return pd.DataFrame({
        'Date': dates,
        'doy_sin': np.sin(2.0 * np.pi * doy / 365.25),
        'doy_cos': np.cos(2.0 * np.pi * doy / 365.25),
        'T_air_C': np.linspace(1.0, 4.0, days),
        'wind_speed_m_per_s': np.full(days, 2.0),
        'Solar_W_m2': np.full(days, 120.0),
        'LST_surface_C': np.linspace(2.0, 5.0, days),
        'LST_is_filled': np.zeros(days),
        'LST_quality_factor': np.ones(days),
        'Longwave_W_m2': np.full(days, 300.0),
        'latent_heat_upward_W_m2': np.full(days, 30.0),
        'sensible_heat_upward_W_m2': np.full(days, 10.0),
        'relative_humidity': np.full(days, 0.75),
        'surface_pressure_Pa': np.full(days, 101325.0),
        'Secchi_m': np.full(days, 3.0),
        'light_extinction_kd': np.full(days, 0.6),
    })


def test_p0_observable_features_feed_forcing_and_static_inputs():
    df = _toy_forcing_frame(days=35)
    df['LSWT_open_water_C'] = df['LST_surface_C']
    df.loc[np.arange(len(df)) % 3 != 0, 'LSWT_open_water_C'] = np.nan
    df['effective_fetch'] = 25000.0
    engineered = add_past_forcing_memory_features(df, include_current_day=False, max_depth_m=12.0)
    new_columns = [
        'net_radiation_7d',
        'net_radiation_30d',
        'wind_energy_7d',
        'wind_energy_30d',
        'wind_mixing_potential_7d',
        'wind_mixing_potential_30d',
        'days_since_last_raw_LSWT',
        'raw_LSWT_valid_count_30d',
        'raw_LSWT_trend_7d',
        'cooling_degree_days_30d',
    ]
    assert set(new_columns).issubset(engineered.columns)
    assert np.isfinite(engineered[new_columns].to_numpy(dtype=np.float32)).all()

    rows = _forcing_tensor_rows(engineered, history_window_days=30)
    assert rows[-1]['features'].numel() == len(FORCING_FEATURE_COLUMNS)
    assert rows[-1]['history_features'].shape == (30, len(FORCING_FEATURE_COLUMNS))

    static = static_feature_array(
        {
            'max_depth_m': 12.0,
            'mean_depth_m': 6.0,
            'area_km2': 10.0,
            'latitude': 45.0,
            'light_extinction_kd': 0.6,
            'light_penetration_ratio': 0.2,
            'hydrology_missing_flag': 1.0,
            'kd_source_type_code': 0.66,
        },
        12.0,
    )
    assert len(STATIC_FEATURE_KEYS) == STATIC_FEATURE_DIM
    assert static.shape == (STATIC_FEATURE_DIM,)
    assert np.isfinite(static).all()


def test_forcing_rows_use_explicit_lst_observed_flag_when_available():
    df = _toy_forcing_frame(days=3)
    df['LST_is_filled'] = 0.0
    df['LST_observed_flag'] = 0.0

    rows = _forcing_tensor_rows(df, history_window_days=3)
    observed_idx = FORCING_FEATURE_COLUMNS.index('LST_observed_flag')

    assert rows[0]['lst_is_filled'].item() == 0.0
    assert rows[0]['lst_observed_flag'].item() == 0.0
    assert rows[0]['features'][observed_idx].item() == 0.0

    fallback_df = df.drop(columns=['LST_observed_flag'])
    fallback_rows = _forcing_tensor_rows(fallback_df, history_window_days=3)
    assert fallback_rows[0]['lst_is_filled'].item() == 0.0
    assert fallback_rows[0]['lst_observed_flag'].item() == 1.0
    assert fallback_rows[0]['features'][observed_idx].item() == 1.0


def test_zero_profile_init_conditioning_matches_train_and_export_builders():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    metadata = {
        'max_depth_m': 12.0,
        'mean_depth_m': 6.0,
        'area_km2': 10.0,
        'latitude': 45.0,
        'elevation_m': 300.0,
        'light_extinction_kd': 0.6,
    }
    df = _toy_forcing_frame(days=35)
    df['LST_observed_flag'] = 1.0
    df['ice_fraction'] = 0.0
    engineered = add_past_forcing_memory_features(
        df,
        include_current_day=False,
        max_depth_m=12.0,
    )
    lake = {
        'df': engineered,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
    }

    train_builder = _zero_profile_init_conditioning_array(lake, 20)
    export_builder = _zero_profile_init_conditioning_from_inputs(
        engineered,
        depths,
        metadata,
        20,
        area_profile=area,
    )
    assert np.allclose(train_builder, export_builder, atol=1.0e-6)

    hyps_mean_idx = ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES.index(
        'hypsometry_mean_area_ratio'
    )
    hyps_bottom_idx = ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES.index(
        'hypsometry_bottom_area_ratio'
    )
    hyps_cv_idx = ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES.index('hypsometry_area_cv')
    assert train_builder[hyps_mean_idx] == pytest.approx(float(np.mean(area / area[0])))
    assert train_builder[hyps_bottom_idx] == pytest.approx(float(area[-1] / area[0]))
    assert train_builder[hyps_cv_idx] > 0.0


def test_unlabeled_heat_closure_windows_exclude_profile_dates():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    df = _toy_forcing_frame(days=8)
    profile_lookup = {
        pd.Timestamp('2020-01-03'): np.linspace(4.0, 6.0, len(depths), dtype=np.float32),
        pd.Timestamp('2020-01-06'): np.linspace(5.0, 7.0, len(depths), dtype=np.float32),
    }

    windows = _build_unlabeled_heat_closure_windows(df, profile_lookup, window_days=1)
    date_to_index = {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }
    profile_indices = {date_to_index[pd.Timestamp(date).normalize()] for date in profile_lookup}

    assert windows
    for start_idx, end_idx in windows:
        assert not any(idx in profile_indices for idx in range(start_idx, end_idx + 1))

    selected = _select_unlabeled_heat_closure_windows(
        {'lake_id': 'toy', 'unlabeled_heat_closure_windows': windows},
        batch_size=2,
        epoch=0,
    )
    selected_again = _select_unlabeled_heat_closure_windows(
        {'lake_id': 'toy', 'unlabeled_heat_closure_windows': windows},
        batch_size=2,
        epoch=0,
    )
    assert len(selected) == 2
    assert selected == selected_again
    assert set(selected).issubset(set(windows))


def test_unlabeled_heat_closure_windows_support_multi_scale_future_horizons():
    df = _toy_forcing_frame(days=45)
    profile_lookup = {
        pd.Timestamp('2020-01-10'): np.asarray([4.0, 5.0], dtype=np.float32),
    }
    by_horizon = _build_unlabeled_heat_closure_windows_by_horizon(
        df,
        profile_lookup,
        horizons=(1, 7, 30),
    )
    assert set(by_horizon) == {1, 7, 30}
    assert all(by_horizon[horizon] for horizon in (1, 7, 30))
    date_to_index = {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }
    profile_idx = date_to_index[pd.Timestamp('2020-01-10')]
    for horizon, windows in by_horizon.items():
        for start_idx, end_idx in windows:
            assert int(end_idx - start_idx) == int(horizon)
            assert profile_idx not in range(start_idx, end_idx + 1)

    selected = _select_unlabeled_heat_closure_windows(
        {
            'lake_id': 'toy',
            'unlabeled_heat_closure_windows_by_horizon': by_horizon,
        },
        batch_size=3,
        epoch=0,
    )
    assert {end_idx - start_idx for start_idx, end_idx in selected} == {1, 7, 30}


def test_unlabeled_heat_closure_loss_uses_no_profile_labels():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    model = LakeStateForecaster(depths, area, static_dim=STATIC_FEATURE_DIM)
    metadata = {
        'max_depth_m': 12.0,
        'mean_depth_m': 6.0,
        'area_km2': 10.0,
        'latitude': 45.0,
        'light_extinction_kd': 0.6,
    }
    df = _toy_forcing_frame(days=6)
    df['LST_quality_factor'] = 1.0
    df['ice_fraction'] = 0.0
    df['ice_mask'] = 0.0
    engineered = add_past_forcing_memory_features(
        df,
        include_current_day=False,
        max_depth_m=12.0,
    )
    lake = {
        'lake_id': 'toy',
        'df': engineered,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(engineered, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
        'lookups': {'all': _toy_profiles(depths)},
        'unlabeled_heat_closure_windows': ((1, 2), (2, 3)),
    }

    loss, detail = _unlabeled_heat_closure_loss_for_lake(
        model,
        lake,
        ((1, 2),),
        weight=0.1,
        window_days=1,
        tau_wm2=50.0,
        open_water_only='on',
        lst_qc_min=0.0,
        reservoir_mode='include',
        mode='storage_budget_thresholded',
        state_source='spinup_then_window',
        state_spinup_days=2,
        solver_guard_weight=0.01,
        solver_guard_tau_wm2=1.0,
        zero_profile_initializer='low_dof',
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert detail['unlabeled_heat_closure_profile_label_count'].item() == 0.0
    assert detail['unlabeled_heat_closure_window_count'].item() == 1.0
    assert detail['unlabeled_heat_closure_step_count'].item() >= 1.0
    assert detail['unlabeled_heat_closure_active_loss_count'].item() == 1.0
    assert torch.isfinite(detail['unlabeled_heat_closure_residual_abs_mean_wm2'])
    assert torch.isfinite(detail['unlabeled_heat_closure_budget_residual_abs_mean_wm2'])
    assert torch.isfinite(detail['unlabeled_heat_closure_solver_residual_abs_mean_wm2'])
    assert torch.isfinite(detail['unlabeled_heat_closure_solver_guard_loss'])
    assert detail['unlabeled_heat_closure_solver_guard_weight'].item() == pytest.approx(0.01)
    assert detail['unlabeled_heat_closure_solver_guard_effective_weight'].item() == pytest.approx(0.01)
    assert detail['unlabeled_heat_closure_solver_guard_tau_wm2'].item() == pytest.approx(1.0)
    assert detail['unlabeled_heat_closure_solver_guard_active_loss_count'].item() >= 1.0
    assert detail['unlabeled_heat_closure_state_source_code'].item() == 1.0
    assert detail['unlabeled_heat_closure_spinup_days_used'].item() > 0.0

    loss.backward()


def test_unlabeled_heat_closure_loss_reports_multi_scale_horizons():
    depths = np.linspace(0.0, 12.0, 6, dtype=np.float32)
    area = np.linspace(1.0, 0.3, len(depths), dtype=np.float32)
    model = LakeStateForecaster(depths, area, static_dim=STATIC_FEATURE_DIM)
    metadata = {
        'max_depth_m': 12.0,
        'mean_depth_m': 6.0,
        'area_km2': 10.0,
        'latitude': 45.0,
        'light_extinction_kd': 0.6,
    }
    df = _toy_forcing_frame(days=40)
    df['LST_quality_factor'] = 1.0
    df['ice_fraction'] = 0.0
    df['ice_mask'] = 0.0
    engineered = add_past_forcing_memory_features(
        df,
        include_current_day=False,
        max_depth_m=12.0,
    )
    lake = {
        'lake_id': 'toy',
        'df': engineered,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(engineered, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
        'lookups': {'all': {}},
        'unlabeled_heat_closure_windows': ((1, 2), (2, 9), (3, 33)),
    }

    loss, detail = _unlabeled_heat_closure_loss_for_lake(
        model,
        lake,
        lake['unlabeled_heat_closure_windows'],
        weight=0.1,
        window_days=30,
        tau_wm2=50.0,
        open_water_only='on',
        lst_qc_min=0.0,
        reservoir_mode='include',
        mode='storage_budget_thresholded',
        state_source='prior_window',
        solver_guard_weight=0.0,
        no_profile_lst_surface_weight=0.0,
        zero_profile_initializer='low_dof',
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert detail['unlabeled_heat_closure_horizon_days_min'].item() == pytest.approx(1.0)
    assert detail['unlabeled_heat_closure_horizon_days_max'].item() == pytest.approx(30.0)
    assert detail['unlabeled_heat_closure_horizon_count'].item() == pytest.approx(3.0)
    assert detail['unlabeled_heat_closure_step_count'].item() >= 38.0


def test_storage_budget_thresholded_loss_has_dead_zone():
    residual = torch.tensor([-75.0, -25.0, 0.0, 40.0, 125.0], dtype=torch.float32)
    thresholded = _storage_budget_residual_loss(
        residual,
        50.0,
        'storage_budget_thresholded',
    )
    smooth = _storage_budget_residual_loss(
        residual,
        50.0,
        'storage_budget_smooth_l1',
    )

    assert torch.allclose(thresholded, torch.tensor([0.5, 0.0, 0.0, 0.0, 1.5]))
    assert smooth[1].item() > 0.0
    assert smooth[2].item() == 0.0


def test_zero_profile_thermal_basis_reuses_checkpoint_unless_refit_requested():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    train_profiles = _toy_profiles(depths)
    checkpoint_basis = {
        'profile_count': 999,
        'source_lake_count': 999,
        'sentinel': 'checkpoint',
    }
    train_lakes = [{
        'lake_id': 'toy',
        'depths_np': depths,
        'lookups': {'train': train_profiles},
        'lookup_masks': {'train': {
            date: np.ones(len(depths), dtype=bool)
            for date in train_profiles
        }},
    }]

    resolved, source = _resolve_zero_profile_thermal_basis(
        checkpoint_thermal_basis=checkpoint_basis,
        train_lakes=train_lakes,
        zero_profile_initializer='eof_pca_init_net',
        zero_profile_init_net_loss_weight=0.1,
        zero_profile_thermal_basis_components=2,
        zero_profile_thermal_basis_grid_points=12,
        refit_zero_profile_thermal_basis='off',
    )
    assert resolved is checkpoint_basis
    assert source == 'checkpoint'

    refit, refit_source = _resolve_zero_profile_thermal_basis(
        checkpoint_thermal_basis=checkpoint_basis,
        train_lakes=train_lakes,
        zero_profile_initializer='eof_pca_init_net',
        zero_profile_init_net_loss_weight=0.1,
        zero_profile_thermal_basis_components=2,
        zero_profile_thermal_basis_grid_points=12,
        refit_zero_profile_thermal_basis='on',
    )
    assert refit is not checkpoint_basis
    assert refit_source == 'train_refit'
    assert refit is not None
    assert refit['profile_count'] == len(train_profiles)


def test_eof_pca_init_net_mode_and_forward_smoke():
    assert normalize_zero_profile_initializer_mode('eof_pca_init_net') == 'eof_pca_init_net'
    assert normalize_zero_profile_initializer_mode('constrained_init_net') == 'eof_pca_init_net'

    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    profiles = _toy_profiles(depths)
    basis = _toy_basis(depths, profiles)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=3,
    )
    base = torch.full((1, len(depths)), 5.0)
    static = torch.zeros(STATIC_FEATURE_DIM)
    forcing_history = torch.zeros(30, len(FORCING_FEATURE_COLUMNS))
    basis_tensors = zero_profile_thermal_basis_tensors_for_depths(basis, depths)
    encoded = model.zero_profile_initial_state_from_basis(
        base,
        forcing_history,
        static,
        basis_tensors['components_on_depth'],
        basis_tensors['coeff_std'],
    )
    assert encoded['initial_profile_c'].shape == base.shape
    assert torch.isfinite(encoded['initial_profile_c']).all()


def test_removed_legacy_zero_profile_modes_are_rejected():
    for mode in ('legacy_prior', 'legacy', 'prior', 'lst_profile_prior'):
        with pytest.raises(ValueError):
            normalize_zero_profile_initializer_mode(mode)

    for mode in ('legacy_surface', 'surface', 'low_rank', 'enkf_low_rank', 'mld_heat_content'):
        with pytest.raises(ValueError):
            normalize_lswt_observer_mode(mode)


def test_multitask_auxiliary_targets_and_loss_backprop():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    profiles = _toy_profiles(depths)
    basis = _toy_basis(depths, profiles, n_components=2)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=2,
        multitask_auxiliary_hidden_dim=16,
    )
    model.zero_profile_thermal_basis = basis
    metadata = {
        'max_depth_m': 12.0,
        'mean_depth_m': 6.0,
        'area_km2': 10.0,
        'latitude': 45.0,
        'light_extinction_kd': 0.6,
    }
    engineered = add_past_forcing_memory_features(
        _toy_forcing_frame(days=21),
        include_current_day=False,
        max_depth_m=12.0,
    )
    lake = {
        'lake_id': 'toy',
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(engineered, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
    }
    target = torch.tensor(profiles[pd.Timestamp('2020-01-11')]).unsqueeze(0)
    target_mask = torch.ones_like(target, dtype=torch.bool)

    aux_targets, aux_mask = _multitask_auxiliary_targets(model, lake, target, target_mask)
    assert aux_targets.shape == (1, len(MULTITASK_AUXILIARY_STATE_KEYS))
    assert aux_mask.shape == aux_targets.shape
    key_index = {key: idx for idx, key in enumerate(MULTITASK_AUXILIARY_STATE_KEYS)}
    assert 'heat_content' not in key_index
    assert aux_mask[0, key_index['column_mean_temperature']]
    assert aux_mask[0, key_index['areal_heat_content_j_m2_normalized']]
    assert aux_mask[0, :6].all()
    assert aux_mask[0, 6:8].all()
    assert not aux_mask[0, 8:].any()
    assert torch.isfinite(aux_targets[aux_mask]).all()

    prediction = target + 0.25
    loss_vec, details = _multitask_auxiliary_loss_vector(
        model,
        lake,
        prediction,
        target,
        target_mask,
        lake['forcing_rows'][10],
        multitask_auxiliary_weight=0.5,
    )
    assert loss_vec.shape == (1,)
    assert torch.isfinite(loss_vec).all()
    assert details[0]['multitask_auxiliary_supervision_count'].item() == 8.0

    loss = 0.5 * loss_vec.mean()
    loss.backward()
    grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.multitask_auxiliary_head.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_zero_profile_init_net_training_records_use_train_lookup_only():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    all_profiles = _toy_profiles(depths)
    train_dates = tuple(sorted(all_profiles)[:2])
    train_profiles = {date: all_profiles[date] for date in train_dates}
    basis = _toy_basis(depths, train_profiles, n_components=2)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=2,
    )
    model.zero_profile_thermal_basis = basis
    df = _toy_forcing_frame(days=21)
    date_to_index = {
        pd.Timestamp(date).normalize(): int((pd.Timestamp(date).normalize() - df['Date'].iloc[0]).days)
        for date in all_profiles
    }
    lake = {
        'lake_id': 'toy',
        'df': df,
        'metadata': {'max_depth_m': 12.0, 'mean_depth_m': 6.0, 'latitude': 45.0},
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'forcing_rows': [
            {'history_features': torch.zeros(30, len(FORCING_FEATURE_COLUMNS))}
            for _ in range(len(df))
        ],
        'static_features': torch.zeros(STATIC_FEATURE_DIM),
        'date_to_index': date_to_index,
        'lookups': {'train': train_profiles, 'all': all_profiles},
        'lookup_tensors': {
            'train': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
            'all': {key: torch.tensor(value).unsqueeze(0) for key, value in all_profiles.items()},
        },
        'lookup_masks': {
            'train': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
            'all': {key: np.ones(len(depths), dtype=bool) for key in all_profiles},
        },
        'lookup_mask_tensors': {
            'train': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
            'all': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in all_profiles},
        },
    }

    losses, details = _zero_profile_init_net_training_records(
        model,
        lake,
        split_key='train',
        epoch=0,
        samples_per_lake=0,
    )
    assert len(losses) == len(train_profiles)
    assert len(details) == len(train_profiles)
    assert 'zero_profile_init_net_band_profile_loss' in details[0]
    assert 'zero_profile_init_net_conditioning_abs_mean' in details[0]
    total = torch.stack(losses).mean()
    total.backward()
    assert torch.isfinite(total)


def test_zero_profile_init_net_training_records_can_supervise_after_spinup():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    all_profiles = _toy_profiles(depths)
    train_dates = tuple(sorted(all_profiles)[:2])
    train_profiles = {date: all_profiles[date] for date in train_dates}
    basis = _toy_basis(depths, train_profiles, n_components=2)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=2,
    )
    model.eval()
    model.zero_profile_thermal_basis = basis
    df = _toy_forcing_frame(days=21)
    metadata = {'max_depth_m': 12.0, 'mean_depth_m': 6.0, 'latitude': 45.0}
    date_to_index = {
        pd.Timestamp(date).normalize(): int((pd.Timestamp(date).normalize() - df['Date'].iloc[0]).days)
        for date in train_profiles
    }
    lake = {
        'lake_id': 'toy',
        'df': df,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(df, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
        'date_to_index': date_to_index,
        'lookups': {'train': train_profiles, 'all': train_profiles},
        'lookup_tensors': {
            'train': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
            'all': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
        },
        'lookup_masks': {
            'train': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
            'all': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
        },
        'lookup_mask_tensors': {
            'train': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
            'all': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
        },
    }

    losses, details = _zero_profile_init_net_training_records(
        model,
        lake,
        split_key='train',
        epoch=1,
        samples_per_lake=1,
        training_spinup_days=5,
        physics_weight=0.01,
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )
    assert len(losses) == 1
    assert len(details) == 1
    assert details[0]['zero_profile_init_net_spinup_days_used'].item() == 5.0
    assert details[0]['zero_profile_init_net_spinup_enabled_count'].item() == 1.0
    assert torch.isfinite(details[0]['zero_profile_init_net_spinup_profile_loss'])
    assert torch.isfinite(details[0]['zero_profile_init_net_band_profile_loss'])
    assert torch.isfinite(details[0]['zero_profile_init_net_physics_loss'])
    assert torch.isfinite(details[0]['zero_profile_init_net_heat_content_constraint_loss'])
    assert torch.isfinite(details[0]['zero_profile_init_net_surface_bottom_constraint_loss'])
    assert torch.isfinite(details[0]['zero_profile_init_net_bounded_state_loss'])

    total = torch.stack(losses).mean()
    total.backward()
    assert torch.isfinite(total)


def test_zero_profile_init_net_joint_rollout_supervises_future_train_profiles():
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    train_profiles = _toy_profiles(depths)
    basis = _toy_basis(depths, train_profiles, n_components=2)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=2,
    )
    model.zero_profile_thermal_basis = basis
    df = _toy_forcing_frame(days=31)
    metadata = {'max_depth_m': 12.0, 'mean_depth_m': 6.0, 'latitude': 45.0}
    date_to_index = {
        pd.Timestamp(date).normalize(): int((pd.Timestamp(date).normalize() - df['Date'].iloc[0]).days)
        for date in train_profiles
    }
    lake = {
        'lake_id': 'toy',
        'df': df,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(df, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
        'date_to_index': date_to_index,
        'lookups': {'train': train_profiles, 'all': train_profiles},
        'lookup_tensors': {
            'train': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
            'all': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
        },
        'lookup_masks': {
            'train': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
            'all': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
        },
        'lookup_mask_tensors': {
            'train': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
            'all': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
        },
    }

    losses, details = _zero_profile_init_net_training_records(
        model,
        lake,
        split_key='train',
        epoch=0,
        samples_per_lake=1,
        training_spinup_days=0,
        rollout_weight=0.5,
        rollout_max_days=25,
        rollout_targets=2,
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )
    assert len(losses) == 1
    assert len(details) == 1
    assert details[0]['zero_profile_init_net_rollout_supervision_count'].item() == 2.0
    assert details[0]['zero_profile_init_net_rollout_enabled_count'].item() == 1.0
    assert details[0]['zero_profile_init_net_rollout_max_gap_days'].item() == 20.0
    assert torch.isfinite(details[0]['zero_profile_init_net_rollout_profile_loss'])
    assert details[0]['zero_profile_init_net_rollout_weighted_loss'].item() > 0.0

    total = torch.stack(losses).mean()
    total.backward()
    grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.zero_profile_init_head.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_zero_profile_init_net_conditioning_features_affect_coefficients():
    torch.manual_seed(123)
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    profiles = _toy_profiles(depths)
    basis = _toy_basis(depths, profiles, n_components=2)
    assert basis is not None

    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        zero_profile_init_net_components=2,
        zero_profile_init_net_init_spread=0.02,
    )
    base = torch.full((1, len(depths)), 5.0)
    static = torch.zeros(STATIC_FEATURE_DIM)
    forcing_history = torch.zeros(30, len(FORCING_FEATURE_COLUMNS))
    basis_tensors = zero_profile_thermal_basis_tensors_for_depths(basis, depths)
    cold_low_inertia = torch.zeros(
        model.zero_profile_init_head.conditioning_dim,
        dtype=torch.float32,
    )
    warm_high_inertia = torch.ones_like(cold_low_inertia)

    encoded_cold = model.zero_profile_initial_state_from_basis(
        base,
        forcing_history,
        static,
        basis_tensors['components_on_depth'],
        basis_tensors['coeff_std'],
        conditioning_features=cold_low_inertia,
    )
    encoded_warm = model.zero_profile_initial_state_from_basis(
        base,
        forcing_history,
        static,
        basis_tensors['components_on_depth'],
        basis_tensors['coeff_std'],
        conditioning_features=warm_high_inertia,
    )
    assert model.zero_profile_init_head.conditioning_dim > 0
    assert not torch.allclose(encoded_cold['coefficients'], encoded_warm['coefficients'])
    assert torch.isfinite(encoded_warm['conditioning_abs_mean'])


def _toy_daily_memory_lake(days=24):
    depths = np.linspace(0.0, 12.0, 8, dtype=np.float32)
    area = np.linspace(1.0, 0.2, len(depths), dtype=np.float32)
    profiles = _toy_profiles(depths)
    train_profiles = {
        date: profile
        for date, profile in profiles.items()
        if int((date - pd.Timestamp('2020-01-01')).days) < days
    }
    basis = _toy_basis(depths, train_profiles, n_components=2)
    metadata = {
        'max_depth_m': 12.0,
        'mean_depth_m': 6.0,
        'area_km2': 10.0,
        'latitude': 45.0,
        'light_extinction_kd': 0.6,
    }
    df = _toy_forcing_frame(days=days)
    df['LST_quality_factor'] = 1.0
    df['ice_fraction'] = 0.0
    df['ice_mask'] = 0.0
    engineered = add_past_forcing_memory_features(
        df,
        include_current_day=False,
        max_depth_m=12.0,
    )
    date_to_index = {
        pd.Timestamp(date).normalize(): int((pd.Timestamp(date).normalize() - engineered['Date'].iloc[0]).days)
        for date in train_profiles
    }
    lake = {
        'lake_id': 'toy',
        'df': engineered,
        'metadata': metadata,
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'area_profile': torch.tensor(area),
        'max_depth': float(depths[-1]),
        'forcing_rows': _forcing_tensor_rows(engineered, history_window_days=30),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
        'date_to_index': date_to_index,
        'lookups': {'train': train_profiles, 'all': train_profiles},
        'lookup_tensors': {
            'train': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
            'all': {key: torch.tensor(value).unsqueeze(0) for key, value in train_profiles.items()},
        },
        'lookup_masks': {
            'train': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
            'all': {key: np.ones(len(depths), dtype=bool) for key in train_profiles},
        },
        'lookup_mask_tensors': {
            'train': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
            'all': {key: torch.ones(1, len(depths), dtype=torch.bool) for key in train_profiles},
        },
        'unlabeled_heat_closure_windows': ((2, 3), (3, 4), (4, 5), (5, 6)),
    }
    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
        daily_memory_components=2,
        daily_memory_init_spread=0.02,
    )
    model.zero_profile_thermal_basis = basis
    return model, lake, basis


def test_daily_memory_head_outputs_low_rank_profile_shape():
    model, lake, _basis = _toy_daily_memory_lake()
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    encoded = _daily_memory_prediction_batch(
        model,
        lake,
        torch.tensor([0, 1], dtype=torch.long),
        basis_tensors,
    )
    assert encoded['daily_profile_c'].shape == (2, len(lake['depths_np']))
    assert encoded['coefficients'].shape == (2, 2)
    assert torch.isfinite(encoded['daily_profile_c']).all()
    assert torch.all(encoded['daily_profile_c'] >= 0.0)
    assert torch.all(encoded['daily_profile_c'] <= 38.0)


def test_daily_memory_uses_separate_forcing_encoder_from_physics_rollout():
    def grad_sum(module):
        return sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    model, lake, _basis = _toy_daily_memory_lake()
    assert model.daily_memory_forcing_encoder is not model.forcing_encoder
    basis_tensors = _daily_memory_basis_tensors(model, lake)

    model.zero_grad(set_to_none=True)
    encoded = _daily_memory_prediction_batch(
        model,
        lake,
        torch.tensor([2, 3], dtype=torch.long),
        basis_tensors,
    )
    encoded['daily_profile_c'].mean().backward()
    assert grad_sum(model.daily_memory_forcing_encoder) > 0.0
    assert grad_sum(model.forcing_encoder) == pytest.approx(0.0)

    model.zero_grad(set_to_none=True)
    row = lake['forcing_rows'][4]
    context = model._encode_forcing_context(
        row['features'].unsqueeze(0),
        row['history_features'].unsqueeze(0),
    )
    context.mean().backward()
    assert grad_sum(model.forcing_encoder) > 0.0
    assert grad_sum(model.daily_memory_forcing_encoder) == pytest.approx(0.0)


def test_thermal_state_profile_fusion_can_use_nearest_future_profile():
    model, lake, _basis = _toy_daily_memory_lake(days=12)
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    fusion = _thermal_state_profile_fusion_batch(
        lake,
        [7],
        basis_tensors,
        mode='daily_only',
        branch='daily',
        lookup_split='all',
        time_policy='nearest',
        max_age_days=45,
        min_depth_fraction=0.25,
        max_weight=0.75,
    )
    assert fusion['enabled'] is True
    assert fusion['valid'].item() is True
    assert fusion['gate'].item() > 0.0
    assert fusion['age_days'].item() == pytest.approx(3.0)
    assert fusion['future_fraction'].item() == pytest.approx(1.0)
    init_fusion = _thermal_state_profile_fusion_batch(
        lake,
        [7],
        basis_tensors,
        mode='init_only',
        branch='init',
        lookup_split='all',
        time_policy='nearest',
        max_age_days=45,
        min_depth_fraction=0.25,
        max_weight=0.75,
    )
    assert init_fusion['valid'].item() is True
    assert init_fusion['future_fraction'].item() == pytest.approx(1.0)


def test_daily_memory_profile_fusion_preserves_feature_fallback_when_no_profile():
    model, lake, _basis = _toy_daily_memory_lake(days=12)
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    base = _daily_memory_prediction_batch(
        model,
        lake,
        [7],
        basis_tensors,
    )
    fused = _daily_memory_prediction_batch(
        model,
        lake,
        [7],
        basis_tensors,
        profile_fusion_mode='daily_only',
        profile_fusion_lookup_split='all',
        profile_fusion_time_policy='nearest',
        profile_fusion_max_age_days=45,
        profile_fusion_min_depth_fraction=0.25,
        profile_fusion_max_weight=0.75,
    )
    no_match = _daily_memory_prediction_batch(
        model,
        lake,
        [7],
        basis_tensors,
        profile_fusion_mode='daily_only',
        profile_fusion_lookup_split='all',
        profile_fusion_time_policy='past_only',
        profile_fusion_max_age_days=2,
        profile_fusion_min_depth_fraction=0.25,
        profile_fusion_max_weight=0.75,
    )
    assert fused['thermal_state_profile_fusion_gate'].item() > 0.0
    assert fused['thermal_state_profile_fusion_future_fraction'].item() == pytest.approx(1.0)
    assert 'daily_profile_feature_c' in fused
    assert not torch.allclose(fused['daily_profile_c'], base['daily_profile_c'])
    assert no_match['thermal_state_profile_fusion_gate'].item() == pytest.approx(0.0)
    assert torch.allclose(no_match['daily_profile_c'], base['daily_profile_c'])


def test_daily_memory_training_records_skip_when_basis_missing():
    model, lake, _basis = _toy_daily_memory_lake()
    model.zero_profile_thermal_basis = None
    losses, details = _daily_memory_reconstruction_training_records(
        model,
        lake,
        split_key='train',
        samples_per_lake=1,
    )
    assert losses == []
    assert details[0]['daily_memory_supervision_count'].item() == 0.0


def test_daily_memory_training_records_use_no_profile_heat_budget_steps():
    model, lake, _basis = _toy_daily_memory_lake()
    losses, details = _daily_memory_reconstruction_training_records(
        model,
        lake,
        split_key='train',
        epoch=0,
        samples_per_lake=2,
        heat_budget_weight=0.1,
        physics_consistency_weight=0.1,
        coefficient_loss_weight=0.5,
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )
    assert losses
    total = torch.stack(losses).mean()
    assert torch.isfinite(total)
    assert sum(float(item['daily_memory_heat_budget_step_count'].item()) for item in details) >= 1.0
    assert sum(float(item['daily_memory_supervision_count'].item()) for item in details) >= 1.0
    assert any('daily_memory_coeff_01_abs_mean' in item for item in details)
    assert sum(float(item['daily_memory_coefficient_supervision_count'].item()) for item in details) >= 1.0
    assert any(
        float(item['daily_memory_coefficient_target_loss'].item()) >= 0.0
        for item in details
    )
    assert any('daily_memory_coefficient_target_clipped_fraction' in item for item in details)
    assert any('daily_memory_coeff_01_target_unit_error_abs_mean' in item for item in details)
    regularization_items = [
        item for item in details
        if float(item['daily_memory_regularization_loss'].item()) > 0.0
    ]
    assert regularization_items
    assert all(
        float(item['daily_memory_regularization_weighted_loss'].item())
        < float(item['daily_memory_regularization_loss'].item())
        for item in regularization_items
    )
    total.backward()
    grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.daily_memory_head.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_daily_memory_physics_consistency_does_not_update_physics_backbone():
    model, lake, _basis = _toy_daily_memory_lake()
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    day_indices = torch.tensor([2, 3], dtype=torch.long)
    encoded = _daily_memory_prediction_batch(model, lake, day_indices, basis_tensors)
    next_encoded = _daily_memory_prediction_batch(model, lake, day_indices + 1, basis_tensors)

    model.zero_grad(set_to_none=True)
    terms = _daily_memory_heat_and_physics_terms(
        model,
        lake,
        encoded['daily_profile_c'],
        next_encoded['daily_profile_c'],
        day_indices,
        task_mode='analysis',
        step_diagnostic_mode='loss',
    )
    loss = terms['physics_consistency_loss'].mean()
    assert torch.isfinite(loss)
    loss.backward()

    physics_grad = sum(
        float(parameter.grad.detach().abs().sum().item())
        for module in (model.param_net, model.physical_scale_head, model.lake_adaptive_head)
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    daily_grad = sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in model.daily_memory_head.parameters()
        if parameter.grad is not None
    )
    assert physics_grad == pytest.approx(0.0)
    assert daily_grad > 0.0


def test_daily_memory_prediction_uses_past_memory_not_future_rows():
    model, lake, _basis = _toy_daily_memory_lake(days=12)
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    base = _daily_memory_prediction_batch(model, lake, [5], basis_tensors)['daily_profile_c']

    mutated = lake.copy()
    mutated_df = lake['df'].copy()
    mutated_df.loc[mutated_df.index > 5, 'T_air_C'] = 99.0
    mutated_df.loc[mutated_df.index > 5, 'Solar_W_m2'] = 900.0
    mutated['df'] = add_past_forcing_memory_features(
        mutated_df,
        include_current_day=False,
        max_depth_m=12.0,
    )
    mutated['forcing_rows'] = _forcing_tensor_rows(mutated['df'], history_window_days=30)
    changed = _daily_memory_prediction_batch(model, mutated, [5], basis_tensors)['daily_profile_c']
    assert torch.allclose(base, changed, atol=1.0e-6)


def test_daily_memory_export_writes_prediction_csv(tmp_path):
    model, lake, _basis = _toy_daily_memory_lake(days=12)
    info = export_daily_memory_reconstruction(
        model,
        lake,
        tmp_path,
        export_max_depth_m=12.0,
    )
    prediction = pd.read_csv(info['prediction_csv'])
    diagnostics = pd.read_csv(info['diagnostics_csv'])
    assert not prediction.empty
    assert not diagnostics.empty
    assert 'prediction_branch' in diagnostics.columns
    assert set(diagnostics['prediction_branch']) == {'daily_memory'}


def test_multitask_profile_summary_sanitizes_nonfinite_and_large_gradients():
    depths = np.array([0.0, 0.001, 2.0, 5.0], dtype=np.float32)
    area = np.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
    )
    profile = torch.tensor([[4.0, float('nan'), float('inf'), -10.0]], dtype=torch.float32)
    summary = model._multitask_auxiliary_profile_summary(
        profile,
        depths=torch.tensor(depths),
        area_profile=torch.tensor(area),
    )
    assert torch.isfinite(summary).all()
    assert torch.all(summary[:, :6] >= 0.0)
    assert torch.all(summary[:, :6] <= 1.5)
    assert torch.all(summary[:, 7:] >= 0.0)
    assert torch.all(summary[:, 7:] <= 1.0)


def test_multitask_profile_summary_constant_profile_has_finite_gradients():
    depths = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    area = np.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
    )
    profile = torch.full((2, len(depths)), 4.0, dtype=torch.float32, requires_grad=True)
    summary = model._multitask_auxiliary_profile_summary(
        profile,
        depths=torch.tensor(depths),
        area_profile=torch.tensor(area),
    )
    loss = summary.sum()
    loss.backward()
    assert torch.isfinite(summary).all()
    assert profile.grad is not None
    assert torch.isfinite(profile.grad).all()


def test_multitask_auxiliary_loss_constant_prediction_has_finite_gradients():
    depths = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    area = np.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        static_dim=STATIC_FEATURE_DIM,
    )
    metadata = {'max_depth_m': 5.0, 'mean_depth_m': 2.5, 'latitude': 45.0}
    forcing_rows = _forcing_tensor_rows(_toy_forcing_frame(days=5), history_window_days=30)
    lake = {
        'lake_id': 'toy',
        'depths_np': depths,
        'depths': torch.tensor(depths),
        'area': torch.tensor(area),
        'static_features': torch.tensor(static_feature_array(metadata, float(depths[-1]))),
    }
    prediction = torch.full((2, len(depths)), 4.0, dtype=torch.float32, requires_grad=True)
    target = torch.stack((
        torch.linspace(4.0, 7.0, len(depths)),
        torch.linspace(5.0, 8.0, len(depths)),
    ))
    target_mask = torch.ones_like(target, dtype=torch.bool)
    loss_vec, details = _multitask_auxiliary_loss_vector(
        model,
        lake,
        prediction,
        target,
        target_mask,
        forcing_rows[0],
        multitask_auxiliary_weight=0.03,
    )
    loss = loss_vec.mean()
    loss.backward()
    assert torch.isfinite(loss_vec).all()
    assert len(details) == 2
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_clip_grad_norm_finite_clears_nonfinite_gradients():
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    param.grad = torch.tensor([float('nan')], dtype=torch.float32)
    norm, finite = _clip_grad_norm_finite([param], 1.0)
    assert not finite
    assert not torch.isfinite(norm)
    assert param.grad is None
