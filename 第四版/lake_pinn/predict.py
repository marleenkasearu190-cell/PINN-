# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import build_depth_grid, build_segment_frame, subset_profile_observations_by_dates
from .forcing import compute_runtime_surface_target
from .kalman import run_profile_kalman_filter
from .lake_metadata import build_contiguous_season_segments
from .model import build_model_inputs
from .online_control import build_online_ppo_rolling_grid
from .ppo import normalize_kalman_scales
from .train import train_model


def apply_smooth_freezing_surface_limit(
    profile,
    depths,
    air_temp_c,
    previous_profile=None,
    target_surface_c=0.0,
    relaxation=0.35,
    max_surface_cooling_c_per_day=1.5,
    max_surface_warming_c_per_day=1.0,
    influence_depth_m=0.8,
):
    """Smoothly nudge the upper skin toward freezing without hard 0 m strips."""
    if air_temp_c is None or not np.isfinite(air_temp_c) or air_temp_c >= 0.0:
        return profile

    out = np.asarray(profile, dtype=np.float64).copy()
    depths = np.asarray(depths, dtype=np.float64)
    freeze_gate = 1.0 / (1.0 + np.exp((float(air_temp_c) + 0.25) / 0.75))
    desired_surface = out[0] + relaxation * freeze_gate * (target_surface_c - out[0])

    if previous_profile is not None:
        prev_surface = float(np.asarray(previous_profile, dtype=np.float64)[0])
        desired_surface = float(
            np.clip(
                desired_surface,
                prev_surface - max_surface_cooling_c_per_day,
                prev_surface + max_surface_warming_c_per_day,
            )
        )

    desired_surface = max(desired_surface, -0.2)
    delta = desired_surface - out[0]
    weights = np.exp(-depths / max(float(influence_depth_m), 1e-6))
    out = out + delta * weights
    out[0] = desired_surface
    return out


def run_seasonal_segmented_pipeline(
    df,
    metadata,
    max_depth,
    depth_points,
    epochs,
    lr,
    collocation_points,
    device,
    train_profile_obs,
    val_profile_obs,
    assim_profile_obs,
    use_kalman,
    use_ppo,
    ppo_control_interval,
    ppo_rollout_steps,
    ppo_max_updates_run,
    ppo_eval_depth_points,
    ppo_use_kalman_reward,
    ppo_tune_kalman,
    ppo_apply_post_physics,
    kalman_prior_std,
    kalman_process_std,
    kalman_obs_std_surface,
    kalman_obs_std_bottom,
    kalman_obs_std_profile,
    kalman_correlation_length,
    kalman_forecast_blend,
    kalman_forecast_spinup_days,
    kalman_forecast_spinup_max_blend,
    shallow_optimized_grid,
    shallow_focus_depth,
    shallow_grid_fraction,
    rolling_prediction_mode,
    rolling_memory_blend,
    rolling_surface_relaxation,
    rolling_surface_decay_depth,
    rolling_deep_inertia,
    shortwave_attenuation_coef,
    shortwave_surface_fraction,
    use_surface_bulk_correction,
    use_bottom_observation,
    initial_condition_mode,
    surface_obs_depth_m,
    time_continuity_weight,
    time_continuity_depth_points,
    stratification_weight,
    stratification_pairs,
    stratification_margin_c,
    smoothness_weight,
    max_vertical_gradient_c_per_m,
    deep_warming_weight,
    deep_anchor_weight,
    deep_anchor_pairs,
    deep_anchor_amplitude_c,
    vertical_exchange_weight,
    entrainment_velocity_scale_m_per_day,
    convective_mixing_weight,
    autumn_overturn_weight,
    heat_budget_weight,
    heat_budget_depth_points,
    train_until_best,
    train_min_epochs,
    train_patience_windows,
    apply_post_physics,
):
    segments = build_contiguous_season_segments(df)
    if not segments:
        raise ValueError('Seasonal segmentation requires at least one time segment.')

    depth_grid = build_depth_grid(
        max_depth=max_depth,
        n_depth_points=depth_points,
        use_shallow_optimized=shallow_optimized_grid,
        shallow_focus_depth=shallow_focus_depth,
        shallow_fraction=shallow_grid_fraction,
    )
    full_temp_grid = np.zeros((len(depth_grid), len(df)), dtype=np.float32)
    full_kalman_grid = np.zeros_like(full_temp_grid) if use_kalman else None
    segment_summaries = []
    ppo_history_frames = []
    ppo_update_frames = []
    final_weights = {}
    kalman_scales = {}

    total_days = max(len(df), 1)
    for segment in segments:
        segment_df, segment_duration_seconds = build_segment_frame(df, segment['start_idx'], segment['end_idx'])
        segment_dates = pd.to_datetime(segment_df['Date']).dt.normalize().tolist()
        segment_train_obs = subset_profile_observations_by_dates(train_profile_obs, segment_dates)
        segment_val_obs = subset_profile_observations_by_dates(val_profile_obs, segment_dates)
        segment_assim_obs = subset_profile_observations_by_dates(assim_profile_obs, segment_dates)

        segment_metadata = dict(metadata)
        segment_metadata['time_scale_seconds'] = segment_duration_seconds
        segment_metadata['start_date'] = segment_df['Date'].iloc[0]
        segment_metadata['file_tag'] = f"{metadata['file_tag']}_{segment['name']}"

        segment_days = len(segment_df)
        segment_epochs = max(1, int(round(epochs * segment_days / total_days)))
        segment_collocation = max(8, int(round(collocation_points * segment_days / total_days)))
        segment_control_interval = max(1, min(ppo_control_interval, segment_epochs))

        model, training_info = train_model(
            df=segment_df,
            metadata=segment_metadata,
            max_depth=max_depth,
            epochs=segment_epochs,
            lr=lr,
            collocation_points=segment_collocation,
            device=device,
            train_profile_obs=segment_train_obs,
            ppo_validation_profile_obs=segment_val_obs,
            use_ppo=use_ppo,
            ppo_control_interval=segment_control_interval,
            ppo_rollout_steps=ppo_rollout_steps,
            ppo_max_updates_run=ppo_max_updates_run,
            ppo_eval_depth_points=min(ppo_eval_depth_points, depth_points),
            ppo_use_kalman_reward=ppo_use_kalman_reward,
            ppo_tune_kalman=ppo_tune_kalman,
            ppo_apply_post_physics=ppo_apply_post_physics,
            base_kalman_process_std=kalman_process_std,
            base_kalman_obs_std_surface=kalman_obs_std_surface,
            base_kalman_obs_std_bottom=kalman_obs_std_bottom,
            base_kalman_obs_std_profile=kalman_obs_std_profile,
            base_kalman_correlation_length=kalman_correlation_length,
            base_kalman_forecast_blend=kalman_forecast_blend,
            base_kalman_forecast_spinup_days=kalman_forecast_spinup_days,
            base_kalman_forecast_spinup_max_blend=kalman_forecast_spinup_max_blend,
            shallow_optimized_grid=shallow_optimized_grid,
            shallow_focus_depth=shallow_focus_depth,
            shallow_grid_fraction=shallow_grid_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            shortwave_attenuation_coef=shortwave_attenuation_coef,
            shortwave_surface_fraction=shortwave_surface_fraction,
            use_surface_bulk_correction=use_surface_bulk_correction,
            use_bottom_observation=use_bottom_observation,
            initial_condition_mode=initial_condition_mode,
            surface_obs_depth_m=surface_obs_depth_m,
            time_continuity_weight=time_continuity_weight,
            time_continuity_depth_points=time_continuity_depth_points,
            stratification_weight=stratification_weight,
            stratification_pairs=stratification_pairs,
            stratification_margin_c=stratification_margin_c,
            smoothness_weight=smoothness_weight,
            max_vertical_gradient_c_per_m=max_vertical_gradient_c_per_m,
            deep_warming_weight=deep_warming_weight,
            deep_anchor_weight=deep_anchor_weight,
            deep_anchor_pairs=deep_anchor_pairs,
            deep_anchor_amplitude_c=deep_anchor_amplitude_c,
            vertical_exchange_weight=vertical_exchange_weight,
            entrainment_velocity_scale_m_per_day=entrainment_velocity_scale_m_per_day,
            convective_mixing_weight=convective_mixing_weight,
            autumn_overturn_weight=autumn_overturn_weight,
            heat_budget_weight=heat_budget_weight,
            heat_budget_depth_points=heat_budget_depth_points,
            train_until_best=train_until_best,
            train_min_epochs=train_min_epochs,
            train_patience_windows=train_patience_windows,
        )
        segment_temp_grid, segment_depths, _ = predict_temperature_grid(
            model,
            df=segment_df,
            metadata=segment_metadata,
            max_depth=max_depth,
            n_depth_points=depth_points,
            device=device,
            apply_post_physics=apply_post_physics,
            use_shallow_optimized=shallow_optimized_grid,
            shallow_focus_depth=shallow_focus_depth,
            shallow_fraction=shallow_grid_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        if not np.allclose(segment_depths, depth_grid):
            raise ValueError('Seasonal segment depth grid mismatch encountered during stitching.')

        full_temp_grid[:, segment['start_idx']:segment['end_idx']] = segment_temp_grid

        segment_kalman_grid = None
        if use_kalman:
            segment_kalman_grid, _ = run_profile_kalman_filter(
                df=segment_df,
                temp_grid=segment_temp_grid,
                depths=segment_depths,
                metadata=segment_metadata,
                max_depth=max_depth,
                profile_obs_data=segment_assim_obs,
                prior_std=kalman_prior_std,
                process_std=kalman_process_std * training_info['kalman_scales']['process'],
                obs_std_surface=kalman_obs_std_surface * training_info['kalman_scales']['obs'],
                obs_std_bottom=kalman_obs_std_bottom * training_info['kalman_scales']['obs'],
                obs_std_profile=kalman_obs_std_profile * training_info['kalman_scales']['obs'],
                correlation_length=training_info['kalman_scales'].get('correlation_length', kalman_correlation_length),
                forecast_blend=training_info['kalman_scales'].get('forecast_blend', kalman_forecast_blend),
                forecast_spinup_days=kalman_forecast_spinup_days,
                forecast_spinup_max_blend=kalman_forecast_spinup_max_blend,
                use_surface_bulk_correction=use_surface_bulk_correction,
                use_bottom_observation=use_bottom_observation,
                surface_obs_depth_m=surface_obs_depth_m,
                autumn_asymmetric_cooling=False,
                autumn_air_temp_threshold=12.0,
            )
            full_kalman_grid[:, segment['start_idx']:segment['end_idx']] = segment_kalman_grid

        final_weights = training_info['final_weights']
        kalman_scales = training_info['kalman_scales']

        if use_ppo and not training_info['ppo_history'].empty:
            history_df = training_info['ppo_history'].copy()
            history_df['segment'] = segment['name']
            ppo_history_frames.append(history_df)
        if use_ppo and not training_info['ppo_update_stats'].empty:
            update_df = training_info['ppo_update_stats'].copy()
            update_df['segment'] = segment['name']
            ppo_update_frames.append(update_df)

        segment_summaries.append(
            {
                'segment': segment['name'],
                'season': segment['season'],
                'start_date': segment['start_date'].date().isoformat(),
                'end_date': segment['end_date'].date().isoformat(),
                'days': int(segment_days),
                'epochs': int(segment_epochs),
                'collocation_points': int(segment_collocation),
                'train_obs_rows': int(len(segment_train_obs)),
                'val_obs_rows': int(len(segment_val_obs)),
                'assim_obs_rows': int(len(segment_assim_obs)),
            }
        )

    return {
        'temp_grid': full_temp_grid,
        'kalman_grid': full_kalman_grid,
        'depths': depth_grid.astype(np.float32),
        'training_info': {
            'final_weights': dict(final_weights),
            'kalman_scales': dict(kalman_scales),
            'ppo_history': pd.concat(ppo_history_frames, ignore_index=True) if ppo_history_frames else pd.DataFrame(),
            'ppo_update_stats': pd.concat(ppo_update_frames, ignore_index=True) if ppo_update_frames else pd.DataFrame(),
            'use_ppo': bool(use_ppo),
            'surface_correction_info': None,
            'seasonal_segmented': True,
            'segment_summaries': pd.DataFrame(segment_summaries),
        },
    }


def predict_temperature_grid(
    model,
    df,
    metadata=None,
    max_depth=25.0,
    n_depth_points=150,
    device='cpu',
    apply_post_physics=False,
    use_shallow_optimized=False,
    shallow_focus_depth=5.0,
    shallow_fraction=0.55,
    rolling_prediction_mode=False,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    ppo_controller=None,
    ppo_policy_bundle=None,
    online_ppo_update=False,
    online_ppo_control_interval=7,
    online_ppo_rollout_steps=4,
    online_ppo_max_updates_run=None,
    validation_profile_obs=None,
    validation_metadata=None,
    validation_max_depth=None,
):
    model.eval()
    depth_grid = build_depth_grid(
        max_depth=max_depth,
        n_depth_points=n_depth_points,
        use_shallow_optimized=use_shallow_optimized,
        shallow_focus_depth=shallow_focus_depth,
        shallow_fraction=shallow_fraction,
    )
    depths = torch.tensor(depth_grid, dtype=torch.float32, device=device).reshape(-1, 1)
    z_norm = depths / max_depth
    profiles = []
    times = torch.tensor(df['time_norm'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)

    with torch.no_grad():
        for day_idx, time_point in enumerate(times):
            t_day = time_point.expand_as(depths)
            pred = model(
                build_model_inputs(
                    model=model,
                    t=t_day,
                    z=depths,
                    max_depth=max_depth,
                    metadata=metadata,
                    doy_sin=float(df['doy_sin'].iloc[day_idx]) if 'doy_sin' in df.columns else None,
                    doy_cos=float(df['doy_cos'].iloc[day_idx]) if 'doy_cos' in df.columns else None,
                    air_temp=float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else None,
                    wind_speed=float(df['wind_speed_m_per_s'].iloc[day_idx]) if 'wind_speed_m_per_s' in df.columns else None,
                    shortwave=float(df['Solar_W_m2'].iloc[day_idx]) if 'Solar_W_m2' in df.columns else None,
                    lst_surface=float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else None,
                    longwave=float(df['Longwave_W_m2'].iloc[day_idx]) if 'Longwave_W_m2' in df.columns else None,
                    latent_heat=float(df['latent_heat_upward_W_m2'].iloc[day_idx]) if 'latent_heat_upward_W_m2' in df.columns else None,
                    sensible_heat=float(df['sensible_heat_upward_W_m2'].iloc[day_idx]) if 'sensible_heat_upward_W_m2' in df.columns else None,
                    secchi=float(df['Secchi_m'].iloc[day_idx]) if 'Secchi_m' in df.columns else None,
                )
            ).cpu().numpy().flatten()
            profiles.append(pred)

    temp_grid = np.array(profiles).T
    depths_np = depths.cpu().numpy().flatten()

    online_ppo_runtime = {
        'diagnostics': pd.DataFrame(),
        'history': pd.DataFrame(),
        'kalman_scales': normalize_kalman_scales(),
    }
    if ppo_controller is not None and ppo_policy_bundle is not None:
        initial_weights = dict(ppo_policy_bundle.get('final_weights', {'pde': 1.0e5, 'bc': 10.0, 'ic': 5.0, 'obs': 1.0}))
        initial_kalman_scales = normalize_kalman_scales(ppo_policy_bundle.get('final_kalman_scales', {}))
        temp_grid, ppo_diagnostics, ppo_history = build_online_ppo_rolling_grid(
            raw_temp_grid=temp_grid,
            df=df,
            depths=depths_np,
            ppo_controller=ppo_controller,
            initial_weights=initial_weights,
            initial_kalman_scales=initial_kalman_scales,
            control_interval=online_ppo_control_interval,
            rollout_steps=online_ppo_rollout_steps,
            update_policy=online_ppo_update,
            max_policy_updates_run=online_ppo_max_updates_run,
            memory_blend=rolling_memory_blend,
            surface_relaxation=rolling_surface_relaxation,
            surface_decay_depth=rolling_surface_decay_depth,
            deep_inertia=rolling_deep_inertia,
            deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            validation_profile_obs=validation_profile_obs,
            validation_metadata=validation_metadata,
            validation_max_depth=validation_max_depth,
        )
        if not ppo_diagnostics.empty:
            online_ppo_runtime['diagnostics'] = ppo_diagnostics
            last_row = ppo_diagnostics.iloc[-1]
            online_ppo_runtime['kalman_scales'] = normalize_kalman_scales(
                {
                    'process': float(last_row['kalman_process_scale']),
                    'obs': float(last_row['kalman_obs_scale']),
                    'correlation_length': float(last_row.get('kalman_correlation_length', initial_kalman_scales['correlation_length'])),
                    'forecast_blend': float(last_row.get('kalman_forecast_blend', initial_kalman_scales['forecast_blend'])),
                }
            )
        if not ppo_history.empty:
            online_ppo_runtime['history'] = ppo_history
    elif rolling_prediction_mode:
        temp_grid = build_rolling_prediction_grid(
            raw_temp_grid=temp_grid,
            df=df,
            depths=depths_np,
            memory_blend=rolling_memory_blend,
            surface_relaxation=rolling_surface_relaxation,
            surface_decay_depth=rolling_surface_decay_depth,
            deep_inertia=rolling_deep_inertia,
            deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
        )

    if apply_post_physics:
        temp_grid = np.clip(temp_grid, 0.0, 30.0)
        air_temp = df['T_air_C'].values
        for day_idx, t_air in enumerate(air_temp):
            if t_air < 0.0:
                previous_profile = temp_grid[:, day_idx - 1] if day_idx > 0 else None
                temp_grid[:, day_idx] = apply_smooth_freezing_surface_limit(
                    temp_grid[:, day_idx],
                    depths_np,
                    t_air,
                    previous_profile=previous_profile,
                )

    return temp_grid, depths_np, online_ppo_runtime


def build_rolling_prediction_grid(
    raw_temp_grid,
    df,
    depths,
    memory_blend=0.85,
    surface_relaxation=0.35,
    surface_decay_depth=4.0,
    deep_inertia=0.65,
    deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    raw_temp_grid = np.asarray(raw_temp_grid, dtype=np.float64)
    rolled_grid = np.zeros_like(raw_temp_grid)
    depths = np.asarray(depths, dtype=np.float64)
    n_depth, n_days = raw_temp_grid.shape

    memory_blend = float(np.clip(memory_blend, 0.0, 1.0))
    surface_relaxation = float(np.clip(surface_relaxation, 0.0, 1.0))
    surface_decay_depth = float(max(surface_decay_depth, 1e-6))
    deep_inertia = float(np.clip(deep_inertia, 0.0, 0.95))
    deep_anchor = float(np.clip(deep_anchor, 0.0, 0.5))
    surface_weights = np.exp(-depths / surface_decay_depth)
    surface_weights[0] = 1.0

    lst_surface = None
    if 'SurfaceBulkTarget_C' in df.columns:
        lst_surface = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    elif 'LST_surface_C' in df.columns:
        lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64)

    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else None
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(n_days, 2.0, dtype=np.float64)

    rolled_grid[:, 0] = raw_temp_grid[:, 0]
    if lst_surface is not None:
        initial_surface_target = compute_runtime_surface_target(
            df=df,
            day_idx=0,
            runtime_skin_cooling_coef=surface_skin_cooling_coef,
            base_surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        initial_surface_error = float(np.clip(initial_surface_target - rolled_grid[0, 0], -6.0, 6.0))
        initial_mld = float(np.clip(mixed_layer_depth[0], 0.5, max(depths[-1] * 0.9, 0.5)))
        mixed_transition0 = 1.0 / (1.0 + np.exp((depths - (initial_mld + 0.75)) / 0.9))
        nudge_weights0 = surface_weights * (mixed_transition0 + 0.08 * (1.0 - mixed_transition0))
        rolled_grid[:, 0] += surface_relaxation * initial_surface_error * nudge_weights0
    if air_temp is not None and air_temp[0] < 0.0:
        rolled_grid[:, 0] = apply_smooth_freezing_surface_limit(
            rolled_grid[:, 0],
            depths,
            air_temp[0],
            previous_profile=None,
        )
    rolled_grid[:, 0] = np.clip(rolled_grid[:, 0], -1.0, 35.0)

    for day_idx in range(1, n_days):
        raw_today = raw_temp_grid[:, day_idx]
        raw_prev = raw_temp_grid[:, day_idx - 1]
        model_increment = raw_today - raw_prev
        day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
        mixed_transition = 1.0 / (1.0 + np.exp((depths - (day_mld + 0.75)) / 0.9))
        increment_scale = 1.0 - deep_inertia * (1.0 - mixed_transition)
        persisted_state = rolled_grid[:, day_idx - 1] + model_increment * increment_scale
        rolled_today = memory_blend * persisted_state + (1.0 - memory_blend) * raw_today

        if lst_surface is not None:
            runtime_surface_target = compute_runtime_surface_target(
                df=df,
                day_idx=day_idx,
                runtime_skin_cooling_coef=surface_skin_cooling_coef,
                base_surface_skin_cooling_coef=surface_skin_cooling_coef,
            )
            surface_error = float(np.clip(runtime_surface_target - rolled_today[0], -6.0, 6.0))
            nudge_weights = surface_weights * (mixed_transition + 0.08 * (1.0 - mixed_transition))
            rolled_today = rolled_today + surface_relaxation * surface_error * nudge_weights

        warm_driver = 0.0
        if air_temp is not None:
            warm_driver = 1.0 / (1.0 + np.exp(-(air_temp[day_idx] - 9.0) / 2.0))
        solar_driver = 1.0
        if 'Solar_W_m2' in df.columns:
            solar_driver = 1.0 / (1.0 + np.exp(-(float(df['Solar_W_m2'].iloc[day_idx]) - 140.0) / 45.0))
        deep_floor = max(day_mld + 2.5, depths[-1] * 0.45)
        anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
        deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
        deep_excess = np.maximum(rolled_today - deep_anchor_profile, 0.0)
        rolled_today = rolled_today - deep_anchor * warm_driver * solar_driver * anchor_mask * deep_excess

        if air_temp is not None and air_temp[day_idx] < 0.0:
            rolled_today = apply_smooth_freezing_surface_limit(
                rolled_today,
                depths,
                air_temp[day_idx],
                previous_profile=rolled_grid[:, day_idx - 1],
            )

        rolled_grid[:, day_idx] = np.clip(rolled_today, -1.0, 35.0)

    return rolled_grid.astype(np.float32)
