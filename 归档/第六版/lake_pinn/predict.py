# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import build_depth_grid, build_segment_frame, subset_profile_observations_by_dates
from .kalman import run_profile_kalman_filter
from .lake_metadata import build_contiguous_season_segments
from .model import build_model_inputs
from .ppo import normalize_kalman_scales
from .train import train_model


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
    surface_skin_cooling_coef,
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
    surface_mixed_layer_uniformity_weight,
    abrupt_surface_cooling_weight,
    bottom_slow_change_weight,
    mid_deep_temporal_smoothness_weight,
    autumn_downward_cooling_weight,
    warm_deep_winter_weak_gradient_weight,
    autumn_overturn_weight,
    heat_budget_weight,
    heat_budget_depth_points,
    profile_grid_physics_weight,
    profile_grid_day_pairs,
    profile_grid_depth_points,
    density_reg_weight,
    train_until_best,
    train_min_epochs,
    train_patience_windows,
    model_architecture='legacy',
    adaptation_mode='full',
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
            surface_mixed_layer_uniformity_weight=surface_mixed_layer_uniformity_weight,
            abrupt_surface_cooling_weight=abrupt_surface_cooling_weight,
            bottom_slow_change_weight=bottom_slow_change_weight,
            mid_deep_temporal_smoothness_weight=mid_deep_temporal_smoothness_weight,
            autumn_downward_cooling_weight=autumn_downward_cooling_weight,
            warm_deep_winter_weak_gradient_weight=warm_deep_winter_weak_gradient_weight,
            autumn_overturn_weight=autumn_overturn_weight,
            heat_budget_weight=heat_budget_weight,
            heat_budget_depth_points=heat_budget_depth_points,
            profile_grid_physics_weight=profile_grid_physics_weight,
            profile_grid_day_pairs=profile_grid_day_pairs,
            profile_grid_depth_points=profile_grid_depth_points,
            density_reg_weight=density_reg_weight,
            train_until_best=train_until_best,
            train_min_epochs=train_min_epochs,
            train_patience_windows=train_patience_windows,
            model_architecture=model_architecture,
            adaptation_mode=adaptation_mode,
        )
        segment_temp_grid, segment_depths, _ = predict_temperature_grid(
            model,
            df=segment_df,
            metadata=segment_metadata,
            max_depth=max_depth,
            n_depth_points=depth_points,
            device=device,
            use_shallow_optimized=shallow_optimized_grid,
            shallow_focus_depth=shallow_focus_depth,
            shallow_fraction=shallow_grid_fraction,
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


def profile_state_memory_values(profile, depths, max_depth):
    profile = np.asarray(profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    if profile.size == 0 or depths.size == 0:
        base = DEFAULT_INITIAL_WATER_TEMPERATURE_C
        return base, base, base
    safe_profile = np.clip(profile, -1.0, 35.0)
    surface = float(np.interp(0.0, depths, safe_profile))
    shallow_mask = depths <= 3.0
    shallow = float(np.mean(safe_profile[shallow_mask])) if np.any(shallow_mask) else surface
    deep_floor = max(0.70 * float(max_depth), float(max_depth) - 5.0)
    deep_mask = depths >= deep_floor
    deep = float(np.mean(safe_profile[deep_mask])) if np.any(deep_mask) else float(np.interp(float(max_depth), depths, safe_profile))
    return surface, shallow, deep


def predict_temperature_grid(
    model,
    df,
    metadata=None,
    max_depth=25.0,
    n_depth_points=150,
    device='cpu',
    use_shallow_optimized=False,
    shallow_focus_depth=5.0,
    shallow_fraction=0.55,
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
    use_state_memory = int(getattr(model, 'input_dim', 2)) >= PINN_STATE_MEMORY_INPUT_DIM

    if 'SurfaceBulkTarget_C' in df.columns:
        initial_surface = float(df['SurfaceBulkTarget_C'].iloc[0])
    elif 'LST_surface_C' in df.columns:
        initial_surface = float(df['LST_surface_C'].iloc[0])
    else:
        initial_surface = DEFAULT_INITIAL_WATER_TEMPERATURE_C
    if not np.isfinite(initial_surface):
        initial_surface = DEFAULT_INITIAL_WATER_TEMPERATURE_C
    if 'BottomTemp_C' in df.columns and np.isfinite(float(df['BottomTemp_C'].iloc[0])):
        initial_deep = float(df['BottomTemp_C'].iloc[0])
    elif initial_surface > 8.0:
        # Warm/deep lakes should not start with a 4 C deep-state memory.
        initial_deep = max(DEFAULT_INITIAL_WATER_TEMPERATURE_C, initial_surface - 2.0)
    else:
        initial_deep = DEFAULT_INITIAL_WATER_TEMPERATURE_C
    prev_surface_temp = initial_surface
    prev_0_3m_mean = initial_surface
    prev_deep_mean = initial_deep

    with torch.no_grad():
        for day_idx, time_point in enumerate(times):
            t_day = time_point.expand_as(depths)
            state_kwargs = {}
            if use_state_memory:
                state_kwargs = {
                    'prev_surface_temp': prev_surface_temp,
                    'prev_0_3m_mean': prev_0_3m_mean,
                    'prev_deep_mean': prev_deep_mean,
                }
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
                    air_temp_mean_7d=float(df['air_temp_mean_7d'].iloc[day_idx]) if 'air_temp_mean_7d' in df.columns else None,
                    air_temp_mean_30d=float(df['air_temp_mean_30d'].iloc[day_idx]) if 'air_temp_mean_30d' in df.columns else None,
                    shortwave_sum_7d=float(df['shortwave_sum_7d'].iloc[day_idx]) if 'shortwave_sum_7d' in df.columns else None,
                    shortwave_sum_30d=float(df['shortwave_sum_30d'].iloc[day_idx]) if 'shortwave_sum_30d' in df.columns else None,
                    wind_mean_7d=float(df['wind_mean_7d'].iloc[day_idx]) if 'wind_mean_7d' in df.columns else None,
                    lst_mean_7d=float(df['lst_mean_7d'].iloc[day_idx]) if 'lst_mean_7d' in df.columns else None,
                    heating_degree_days_30d=float(df['heating_degree_days_30d'].iloc[day_idx]) if 'heating_degree_days_30d' in df.columns else None,
                    water_level_anomaly=float(df['water_level_anomaly'].iloc[day_idx]) if 'water_level_anomaly' in df.columns else None,
                    light_extinction_kd=float(df['light_extinction_kd'].iloc[day_idx]) if 'light_extinction_kd' in df.columns else None,
                    effective_fetch=float(df['effective_fetch'].iloc[day_idx]) if 'effective_fetch' in df.columns else None,
                    net_inflow=float(df['net_inflow'].iloc[day_idx]) if 'net_inflow' in df.columns else None,
                    **state_kwargs,
                )
            ).cpu().numpy().flatten()
            profiles.append(pred)
            if use_state_memory:
                prev_surface_temp, prev_0_3m_mean, prev_deep_mean = profile_state_memory_values(
                    pred,
                    depth_grid,
                    max_depth,
                )

    temp_grid = np.array(profiles).T
    depths_np = depths.cpu().numpy().flatten()

    online_ppo_runtime = {
        'diagnostics': pd.DataFrame(),
        'history': pd.DataFrame(),
        'kalman_scales': normalize_kalman_scales(),
    }
    return temp_grid, depths_np, online_ppo_runtime
