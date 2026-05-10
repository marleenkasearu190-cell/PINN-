# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import has_profile_observations
from .forcing import compute_runtime_surface_target
from .physics import project_temperature_profile_to_stable_density, water_density_numpy
from .ppo import (
    PPOController,
    apply_online_ppo_action,
    build_ppo_controller_from_bundle,
    build_ppo_state,
    compute_ppo_reward,
    derive_online_control_params_from_weights,
    export_ppo_policy_bundle,
    normalize_kalman_scales,
)
from .validation import evaluate_profile_at_date, evaluate_profile_grid, evaluate_surface_band_validation_at_date

def compute_online_proxy_summary(current_profile, previous_profile, day_idx, df, depths, control_params, kalman_scales):
    current_profile = np.asarray(current_profile, dtype=np.float64)
    previous_profile = None if previous_profile is None else np.asarray(previous_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)

    surface_target = compute_runtime_surface_target(
        df=df,
        day_idx=day_idx,
        runtime_skin_cooling_coef=control_params.get('surface_skin_cooling_coef', SURFACE_SKIN_COOLING_COEF),
    )
    signed_surface_error = float(current_profile[0] - surface_target)
    surface_mismatch = abs(signed_surface_error)
    warm_surface_penalty = float(max(signed_surface_error, 0.0))
    increment_mag = 0.0 if previous_profile is None else float(np.mean(np.abs(current_profile - previous_profile)))
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else surface_target
    prev_raw_lst = (
        float(df['LST_surface_C'].iloc[day_idx - 1])
        if ('LST_surface_C' in df.columns and day_idx > 0)
        else raw_lst_surface
    )
    density_profile = water_density_numpy(current_profile)
    instability = float(np.mean(np.clip(-np.diff(density_profile), 0.0, None))) if len(current_profile) > 1 else 0.0

    mixed_layer_depth = float(df['MixedLayerDepth_m'].iloc[day_idx]) if 'MixedLayerDepth_m' in df.columns else 2.0
    wind_speed = float(df['wind_speed_m_per_s'].iloc[day_idx]) if 'wind_speed_m_per_s' in df.columns else 2.0
    solar_flux = float(df['Solar_W_m2'].iloc[day_idx]) if 'Solar_W_m2' in df.columns else 0.0
    deep_floor = max(mixed_layer_depth + 2.5, float(depths[-1]) * 0.45)
    anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
    deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
    deep_excess = np.maximum(current_profile - deep_anchor_profile, 0.0)
    deep_warm_excess = float(np.mean(anchor_mask * deep_excess))
    deep_warm_peak = float(np.max(anchor_mask * deep_excess))
    stratification_gap = float(max(current_profile[0] - current_profile[-1], 0.0))
    doy = int(pd.to_datetime(df['Date'].iloc[day_idx]).dayofyear) if 'Date' in df.columns else int(day_idx + 1)
    air_temp = float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else 10.0
    lst_jump = abs(raw_lst_surface - prev_raw_lst)
    skin_bulk_gap = abs(raw_lst_surface - surface_target)
    lst_spike_indicator = float(
        np.clip((lst_jump - 2.5) / 4.0, 0.0, 1.5) * (0.45 + 0.55 * np.exp(-np.clip(wind_speed, 0.0, 12.0) / 2.5))
        + 0.35 * np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
        + 0.20 * np.clip((solar_flux - 120.0) / 250.0, 0.0, 1.0)
        + 0.15 * np.clip((abs(raw_lst_surface - air_temp) - 4.0) / 8.0, 0.0, 1.0)
    )

    deep_ref_depth = float(min(9.0, max(5.0, float(depths[-1]) * 0.45)))
    deep_ref_temp = float(np.interp(deep_ref_depth, depths, current_profile))
    summer_9m_temp = deep_ref_temp
    summer_bottom_temp = float(current_profile[-1])
    summer_stratification_penalty = 0.0
    summer_thermocline_thickness_penalty = 0.0
    summer_surface_warming_reward = 0.0
    summer_midlayer_temp_reward = 0.0
    if 160 <= doy <= 250:
        summer_gap = float(current_profile[0] - deep_ref_temp)
        summer_stratification_penalty = float(max(6.0 - summer_gap, 0.0) / 6.0)
        summer_surface_warming_reward = float(np.clip((current_profile[0] - 18.0) / 6.0, 0.0, 1.5))
        mid_ref_depth = float(min(6.0, max(4.0, float(depths[-1]) * 0.30)))
        mid_ref_temp = float(np.interp(mid_ref_depth, depths, current_profile))
        summer_midlayer_temp_reward = float(np.clip(1.0 - abs(mid_ref_temp - 13.0) / 5.0, 0.0, 1.0))

    summer_thermocline_depth_norm = 0.0
    if depths.size >= 2:
        depth_mids = 0.5 * (depths[:-1] + depths[1:])
        gradients = np.diff(current_profile) / np.maximum(np.diff(depths), 1e-6)
        thermo_band = (depth_mids >= 1.0) & (depth_mids <= min(12.0, float(depths[-1]) - 0.5))
        if np.any(thermo_band):
            thermo_strength = -gradients[thermo_band]
            if thermo_strength.size > 0:
                thermo_depth = float(depth_mids[thermo_band][int(np.argmax(thermo_strength))])
                summer_thermocline_depth_norm = thermo_depth / max(float(depths[-1]), 1.0)
                positive_strength = np.clip(thermo_strength, 0.0, None)
                peak_strength = float(np.max(positive_strength))
                if peak_strength <= 1.0e-6:
                    summer_thermocline_thickness_penalty = 1.0
                else:
                    norm_weights = positive_strength / np.maximum(np.sum(positive_strength), 1.0e-8)
                    mean_depth = float(np.sum(depth_mids[thermo_band] * norm_weights))
                    std_depth = float(np.sqrt(np.sum(((depth_mids[thermo_band] - mean_depth) ** 2) * norm_weights)))
                    summer_thermocline_thickness_penalty = float(max(std_depth - 1.2, 0.0) / 1.2)

    autumn_overturn_penalty = 0.0
    autumn_surface_cooling_rate = 0.0
    autumn_gap_collapse = 0.0
    autumn_false_overturn_penalty = 0.0
    autumn_cooling_triggered_overturn_reward = 0.0
    if 280 <= doy <= 330 and air_temp <= 15.0:
        autumn_gap = float(abs(current_profile[0] - current_profile[-1]))
        autumn_overturn_penalty = float(max(autumn_gap - 1.2, 0.0))
        if previous_profile is not None:
            prev_gap = float(max(previous_profile[0] - previous_profile[-1], 0.0))
            current_gap = float(max(current_profile[0] - current_profile[-1], 0.0))
            autumn_surface_cooling_rate = float(max(previous_profile[0] - current_profile[0], 0.0))
            cooling_gate = float(np.clip(autumn_surface_cooling_rate / 0.5, 0.0, 1.0))
            autumn_gap_collapse = float(max(prev_gap - current_gap, 0.0) * cooling_gate)
            deep_warming = float(max(current_profile[-1] - previous_profile[-1], 0.0))
            false_collapse = float(max(max(prev_gap - current_gap, 0.0) - 1.25 * autumn_surface_cooling_rate, 0.0))
            false_warming = float(max(deep_warming - 0.08, 0.0))
            autumn_false_overturn_penalty = false_collapse + 1.5 * false_warming
            autumn_cooling_triggered_overturn_reward = float(
                np.clip(autumn_surface_cooling_rate / 0.6, 0.0, 1.5)
                * np.clip(autumn_gap_collapse / 0.8, 0.0, 1.5)
                * np.exp(-2.5 * (false_collapse + false_warming))
            )

    winter_inverse_penalty = 0.0
    winter_bottom_4c_error = 0.0
    if (doy <= 75 or doy >= 335) and air_temp <= 6.0:
        inverse_gap = float(current_profile[-1] - current_profile[0])
        winter_bottom_4c_error = float(abs(current_profile[-1] - 4.0))
        winter_inverse_penalty = float(max(1.5 - inverse_gap, 0.0) / 1.5 + abs(current_profile[-1] - 4.0) / 4.0)

    deep_smoothness_penalty = 0.0
    deep_mask = depths >= min(10.0, float(depths[-1]) * 0.55)
    deep_indices = np.where(deep_mask)[0]
    if deep_indices.size >= 3:
        deep_segment = current_profile[deep_indices]
        deep_smoothness_penalty = float(np.mean(np.abs(np.diff(deep_segment, n=2))))

    proxy_total = (
        surface_mismatch ** 2
        + 0.5 * increment_mag
        + 8.0 * instability
        + 3.0 * deep_warm_excess ** 2
        + 1.5 * deep_warm_peak ** 2
        + 1.5 * warm_surface_penalty ** 2
        + 2.5 * summer_stratification_penalty
        + 3.6 * summer_thermocline_thickness_penalty
        + 0.55 * max(summer_9m_temp - 14.0, 0.0)
        + 0.40 * max(summer_bottom_temp - 7.5, 0.0)
        + 2.2 * autumn_overturn_penalty
        + 1.8 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 2.0 * max(0.50 - autumn_gap_collapse, 0.0)
        + 4.0 * autumn_false_overturn_penalty
        + 1.8 * winter_inverse_penalty
        + 0.8 * winter_bottom_4c_error
        + 1.5 * deep_smoothness_penalty
        + 0.8 * lst_spike_indicator
        - 1.6 * summer_surface_warming_reward
        - 1.2 * summer_midlayer_temp_reward
        - 1.6 * autumn_cooling_triggered_overturn_reward
    )
    return {
        'loss_pde': max(instability, 1e-8),
        'loss_bc': max(surface_mismatch ** 2 + 1.5 * warm_surface_penalty ** 2, 1e-8),
        'loss_ic': max(increment_mag, 1e-8),
        'loss_obs': max(surface_mismatch ** 2 + 2.0 * warm_surface_penalty ** 2, 1e-8),
        'total': max(proxy_total, 1e-8),
        'kappa_mean': float(max(kalman_scales['process'], 1e-8)),
        'ri_mean': float(stratification_gap),
        'surface_rmse': float(surface_mismatch),
        'warm_surface_bias': float(warm_surface_penalty),
        'instability_penalty': float(instability),
        'deep_warm_penalty': float(deep_warm_excess),
        'summer_stratification_penalty': float(summer_stratification_penalty),
        'summer_thermocline_depth_norm': float(summer_thermocline_depth_norm),
        'summer_thermocline_thickness_penalty': float(summer_thermocline_thickness_penalty),
        'summer_surface_warming_reward': float(summer_surface_warming_reward),
        'summer_midlayer_temp_reward': float(summer_midlayer_temp_reward),
        'summer_9m_temp': float(summer_9m_temp),
        'summer_bottom_temp': float(summer_bottom_temp),
        'autumn_overturn_penalty': float(autumn_overturn_penalty),
        'autumn_surface_cooling_rate': float(autumn_surface_cooling_rate),
        'autumn_gap_collapse': float(autumn_gap_collapse),
        'autumn_false_overturn_penalty': float(autumn_false_overturn_penalty),
        'autumn_cooling_triggered_overturn_reward': float(autumn_cooling_triggered_overturn_reward),
        'winter_inverse_penalty': float(winter_inverse_penalty),
        'winter_bottom_4c_error': float(winter_bottom_4c_error),
        'deep_smoothness_penalty': float(deep_smoothness_penalty),
        'lst_spike_indicator': float(lst_spike_indicator),
    }


def compute_online_proxy_validation(
    current_profile,
    day_idx,
    df,
    runtime_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    base_surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    surface_target = compute_runtime_surface_target(
        df=df,
        day_idx=day_idx,
        runtime_skin_cooling_coef=runtime_skin_cooling_coef,
        base_surface_skin_cooling_coef=base_surface_skin_cooling_coef,
    )
    surface_error = float(current_profile[0] - surface_target)
    warm_surface_bias = float(max(surface_error, 0.0))
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else surface_target
    prev_raw_lst = (
        float(df['LST_surface_C'].iloc[day_idx - 1])
        if ('LST_surface_C' in df.columns and day_idx > 0)
        else raw_lst_surface
    )
    wind_speed = float(df['wind_speed_m_per_s'].iloc[day_idx]) if 'wind_speed_m_per_s' in df.columns else 2.0
    solar_flux = float(df['Solar_W_m2'].iloc[day_idx]) if 'Solar_W_m2' in df.columns else 0.0
    air_temp = float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else surface_target
    lst_jump = abs(raw_lst_surface - prev_raw_lst)
    skin_bulk_gap = abs(raw_lst_surface - surface_target)
    lst_spike_indicator = float(
        np.clip((lst_jump - 2.5) / 4.0, 0.0, 1.5) * (0.45 + 0.55 * np.exp(-np.clip(wind_speed, 0.0, 12.0) / 2.5))
        + 0.35 * np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
        + 0.20 * np.clip((solar_flux - 120.0) / 250.0, 0.0, 1.0)
        + 0.15 * np.clip((abs(raw_lst_surface - air_temp) - 4.0) / 8.0, 0.0, 1.0)
    )
    return {
        'rmse': abs(surface_error) + 1.5 * warm_surface_bias,
        'mae': abs(surface_error) + 1.2 * warm_surface_bias,
        'bias': surface_error,
        'surface_rmse': abs(surface_error),
        'warm_surface_bias': warm_surface_bias,
        'lst_spike_indicator': lst_spike_indicator,
    }


def train_pure_forecast_ppo_policy(
    model,
    df,
    metadata,
    max_depth,
    depth_points,
    device,
    validation_profile_obs,
    initial_weights,
    initial_kalman_scales,
    apply_post_physics=False,
    use_shallow_optimized=False,
    shallow_focus_depth=5.0,
    shallow_fraction=0.55,
    rolling_prediction_mode=True,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    ppo_control_interval=7,
    ppo_rollout_steps=4,
    ppo_max_updates_run=None,
    max_episodes=None,
    initial_ppo_policy_bundle=None,
):
    if not has_profile_observations(validation_profile_obs):
        return None

    ppo_controller = PPOController(state_dim=PPO_STATE_DIM, action_dim=PPO_ONLINE_ACTION_DIM, device=device)
    if initial_ppo_policy_bundle is not None and int(initial_ppo_policy_bundle.get('action_dim', PPO_ONLINE_ACTION_DIM)) == PPO_ONLINE_ACTION_DIM:
        restored_controller, _ = build_ppo_controller_from_bundle(initial_ppo_policy_bundle, device=device)
        if restored_controller is not None:
            ppo_controller = restored_controller
    current_kalman_scales = dict(initial_kalman_scales)
    current_kalman_scales = normalize_kalman_scales(current_kalman_scales)
    total_update_cap = None if ppo_max_updates_run is None else int(max(ppo_max_updates_run, 0))
    if max_episodes is None:
        max_episodes = 1 if total_update_cap is None else max(1, min(total_update_cap, 4))

    best_metric = float('inf')
    best_snapshot = None
    best_temp_grid = None
    best_depths = None
    episode_history_frames = []
    episode_diag_frames = []
    total_updates = 0

    for episode_idx in range(int(max_episodes)):
        remaining_updates = None if total_update_cap is None else max(total_update_cap - total_updates, 0)
        if total_update_cap is not None and remaining_updates <= 0:
            break

        policy_bundle = {
            'final_weights': dict(initial_weights),
            'final_kalman_scales': dict(current_kalman_scales),
        }
        from .predict import predict_temperature_grid

        temp_grid, depths, online_runtime = predict_temperature_grid(
            model,
            df=df,
            metadata=metadata,
            max_depth=max_depth,
            n_depth_points=depth_points,
            device=device,
            apply_post_physics=apply_post_physics,
            use_shallow_optimized=use_shallow_optimized,
            shallow_focus_depth=shallow_focus_depth,
            shallow_fraction=shallow_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            ppo_controller=ppo_controller,
            ppo_policy_bundle=policy_bundle,
            online_ppo_update=True,
            online_ppo_control_interval=ppo_control_interval,
            online_ppo_rollout_steps=ppo_rollout_steps,
            online_ppo_max_updates_run=remaining_updates,
            validation_profile_obs=validation_profile_obs,
            validation_metadata=metadata,
            validation_max_depth=max_depth,
        )
        episode_metrics = evaluate_profile_grid(
            df=df,
            metadata=metadata,
            temp_grid=temp_grid,
            depths=depths,
            max_depth=max_depth,
            profile_obs_data=validation_profile_obs,
        )
        episode_selection_metric = None
        if episode_metrics is not None and np.isfinite(episode_metrics.get('objective', np.nan)):
            episode_selection_metric = float(episode_metrics['objective'])
        elif episode_metrics is not None and np.isfinite(episode_metrics.get('rmse', np.nan)):
            episode_selection_metric = float(episode_metrics['rmse'])

        if episode_selection_metric is not None:
            if episode_selection_metric < best_metric:
                best_metric = episode_selection_metric
                best_snapshot = {
                    'ppo_state_dict': copy.deepcopy(ppo_controller.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(ppo_controller.optimizer.state_dict()),
                    'kalman_scales': dict(online_runtime.get('kalman_scales', current_kalman_scales)),
                    'metrics': dict(episode_metrics),
                }
                best_temp_grid = temp_grid.copy()
                best_depths = depths.copy()

        diagnostics_df = online_runtime.get('diagnostics', pd.DataFrame()).copy()
        if not diagnostics_df.empty:
            diagnostics_df['episode'] = episode_idx
            episode_diag_frames.append(diagnostics_df)
            total_updates += int(diagnostics_df['ppo_update_count'].max())
            last_diag = diagnostics_df.iloc[-1]
            current_kalman_scales = normalize_kalman_scales(
                {
                    'process': float(last_diag['kalman_process_scale']),
                    'obs': float(last_diag['kalman_obs_scale']),
                    'correlation_length': float(last_diag.get('kalman_correlation_length', current_kalman_scales['correlation_length'])),
                    'forecast_blend': float(last_diag.get('kalman_forecast_blend', current_kalman_scales['forecast_blend'])),
                }
            )

        history_df = online_runtime.get('history', pd.DataFrame()).copy()
        if not history_df.empty:
            history_df['episode'] = episode_idx
            episode_history_frames.append(history_df)

        if diagnostics_df.empty or int(diagnostics_df['ppo_update_count'].max()) == 0:
            break

    if best_snapshot is not None:
        ppo_controller.model.load_state_dict(best_snapshot['ppo_state_dict'])
        ppo_controller.optimizer.load_state_dict(best_snapshot['optimizer_state_dict'])
        current_kalman_scales = normalize_kalman_scales(best_snapshot['kalman_scales'])
    elif best_temp_grid is None:
        return None

    final_bundle = export_ppo_policy_bundle(
        ppo_controller,
        final_weights=initial_weights,
        final_kalman_scales=current_kalman_scales,
    )
    return {
        'ppo_controller': ppo_controller,
        'ppo_policy_bundle': final_bundle,
        'ppo_history': pd.concat(episode_history_frames, ignore_index=True) if episode_history_frames else pd.DataFrame(),
        'ppo_update_stats': pd.concat(episode_diag_frames, ignore_index=True) if episode_diag_frames else pd.DataFrame(),
        'best_validation_metrics': None if best_snapshot is None else dict(best_snapshot['metrics']),
        'kalman_scales': dict(current_kalman_scales),
        'temp_grid': best_temp_grid,
        'depths': best_depths,
        'ppo_update_count': int(total_updates),
    }


def build_online_ppo_rolling_grid(
    raw_temp_grid,
    df,
    depths,
    ppo_controller,
    initial_weights,
    initial_kalman_scales,
    control_interval=7,
    rollout_steps=4,
    update_policy=False,
    max_policy_updates_run=None,
    memory_blend=0.85,
    surface_relaxation=0.35,
    surface_decay_depth=4.0,
    deep_inertia=0.65,
    deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    validation_profile_obs=None,
    validation_metadata=None,
    validation_max_depth=None,
):
    raw_temp_grid = np.asarray(raw_temp_grid, dtype=np.float64)
    rolled_grid = np.zeros_like(raw_temp_grid)
    depths = np.asarray(depths, dtype=np.float64)
    n_depth, n_days = raw_temp_grid.shape

    control_params = derive_online_control_params_from_weights(
        initial_weights=initial_weights,
        memory_blend=memory_blend,
        surface_relaxation=surface_relaxation,
        surface_decay_depth=surface_decay_depth,
        deep_inertia=deep_inertia,
        deep_anchor=deep_anchor,
        surface_skin_cooling_coef=surface_skin_cooling_coef,
    )
    kalman_scales = normalize_kalman_scales(initial_kalman_scales)
    policy_weights = dict(initial_weights)
    surface_decay_depth = float(max(surface_decay_depth, 1e-6))

    lst_surface = None
    if 'SurfaceBulkTarget_C' in df.columns:
        lst_surface = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    elif 'LST_surface_C' in df.columns:
        lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64)

    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else None
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(n_days, 2.0, dtype=np.float64)

    diagnostics = []
    ppo_context = None
    ppo_history = []
    ppo_update_count = 0

    rolled_grid[:, 0] = raw_temp_grid[:, 0]

    for day_idx in range(n_days):
        if day_idx > 0:
            raw_today = raw_temp_grid[:, day_idx]
            raw_prev = raw_temp_grid[:, day_idx - 1]
            model_increment = raw_today - raw_prev
            day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
            mixed_transition = 1.0 / (1.0 + np.exp((depths - (day_mld + 0.75)) / 0.9))
            increment_scale = 1.0 - control_params['deep_inertia'] * (1.0 - mixed_transition)
            persisted_state = rolled_grid[:, day_idx - 1] + model_increment * increment_scale
            rolled_today = control_params['memory_blend'] * persisted_state + (1.0 - control_params['memory_blend']) * raw_today

            if lst_surface is not None:
                runtime_surface_target = compute_runtime_surface_target(
                    df=df,
                    day_idx=day_idx,
                    runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
                    base_surface_skin_cooling_coef=surface_skin_cooling_coef,
                )
                surface_error = float(np.clip(runtime_surface_target - rolled_today[0], -6.0, 6.0))
                current_surface_decay_depth = float(max(control_params.get('surface_decay_depth', surface_decay_depth), 1.0e-6))
                surface_weights = np.exp(-depths / current_surface_decay_depth)
                surface_weights[0] = 1.0
                nudge_weights = surface_weights * (mixed_transition + 0.08 * (1.0 - mixed_transition))
                rolled_today = rolled_today + control_params['surface_relaxation'] * surface_error * nudge_weights

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
            rolled_today = rolled_today - control_params['deep_anchor'] * warm_driver * solar_driver * anchor_mask * deep_excess

            if air_temp is not None and air_temp[day_idx] < 0.0:
                rolled_today[0] = 0.0

            rolled_today = np.clip(rolled_today, -1.0, 35.0)
            rolled_today, projection_adjustments = project_temperature_profile_to_stable_density(rolled_today)
            rolled_grid[:, day_idx] = rolled_today
        else:
            projection_adjustments = 0
            if lst_surface is not None:
                initial_surface_target = compute_runtime_surface_target(
                    df=df,
                    day_idx=0,
                    runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
                    base_surface_skin_cooling_coef=surface_skin_cooling_coef,
                )
                initial_surface_error = float(np.clip(initial_surface_target - rolled_grid[0, 0], -6.0, 6.0))
                initial_mld = float(np.clip(mixed_layer_depth[0], 0.5, max(depths[-1] * 0.9, 0.5)))
                mixed_transition0 = 1.0 / (1.0 + np.exp((depths - (initial_mld + 0.75)) / 0.9))
                initial_surface_decay_depth = float(max(control_params.get('surface_decay_depth', surface_decay_depth), 1.0e-6))
                surface_weights0 = np.exp(-depths / initial_surface_decay_depth)
                surface_weights0[0] = 1.0
                nudge_weights0 = surface_weights0 * (mixed_transition0 + 0.08 * (1.0 - mixed_transition0))
                rolled_grid[:, 0] += control_params['surface_relaxation'] * initial_surface_error * nudge_weights0
            if air_temp is not None and air_temp[0] < 0.0:
                rolled_grid[0, 0] = 0.0
            rolled_grid[:, 0] = np.clip(rolled_grid[:, 0], -1.0, 35.0)
            rolled_grid[:, 0], projection_adjustments = project_temperature_profile_to_stable_density(rolled_grid[:, 0])

        current_profile = rolled_grid[:, day_idx]
        previous_profile = None if day_idx == 0 else rolled_grid[:, day_idx - 1]
        summary = compute_online_proxy_summary(
            current_profile=current_profile,
            previous_profile=previous_profile,
            day_idx=day_idx,
            df=df,
            depths=depths,
            control_params=control_params,
            kalman_scales=kalman_scales,
        )
        validation_metrics = compute_online_proxy_validation(
            current_profile,
            day_idx,
            df,
            runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
            base_surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        if has_profile_observations(validation_profile_obs):
            date_validation = evaluate_profile_at_date(
                current_date=df['Date'].iloc[day_idx],
                current_profile=current_profile,
                depths=depths,
                profile_obs_data=validation_profile_obs,
            )
            if date_validation is not None:
                validation_metrics.update(date_validation)
            shallow_surface_validation = evaluate_surface_band_validation_at_date(
                current_date=df['Date'].iloc[day_idx],
                current_profile=current_profile,
                depths=depths,
                profile_obs_data=validation_profile_obs,
                previous_profile_3d=None if day_idx < 3 else rolled_grid[:, day_idx - 3],
                shallow_max_depth=3.0,
            )
            if shallow_surface_validation is not None:
                validation_metrics.update(shallow_surface_validation)

        if ((day_idx + 1) % max(control_interval, 1) == 0) or (day_idx == n_days - 1):
            current_state = build_ppo_state(
                summary=summary,
                weights=policy_weights,
                kalman_scales=kalman_scales,
                learning_rate=1.0e-3,
                validation_metrics=validation_metrics,
            )

            if ppo_context is not None:
                reward = compute_ppo_reward(
                    prev_summary=ppo_context['summary'],
                    current_summary=summary,
                    prev_validation_metrics=ppo_context['validation'],
                    current_validation_metrics=validation_metrics,
                )
                done = bool(day_idx == n_days - 1)
                ppo_controller.store_transition(
                    state=ppo_context['state'],
                    action=ppo_context['action'],
                    log_prob=ppo_context['log_prob'],
                    reward=reward,
                    done=done,
                    value=ppo_context['value'],
                )
                if update_policy and (done or len(ppo_controller.buffer['states']) >= max(rollout_steps, 1)):
                    reached_update_cap = (
                        max_policy_updates_run is not None
                        and ppo_update_count >= int(max(max_policy_updates_run, 0))
                    )
                    if reached_update_cap:
                        ppo_controller.reset_buffer()
                    else:
                        update_stats = ppo_controller.update(last_state=current_state, last_done=done)
                        if update_stats is not None:
                            ppo_update_count += 1

            if day_idx != n_days - 1:
                action, log_prob, value = ppo_controller.select_action(current_state)
                control_params, kalman_scales = apply_online_ppo_action(control_params, kalman_scales, action)
                ppo_context = {
                    'state': current_state,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'summary': summary,
                    'validation': validation_metrics,
                }
                ppo_history.append(
                    {
                        'day_idx': day_idx,
                        'date': pd.Timestamp(df['Date'].iloc[day_idx]),
                        'memory_blend': control_params['memory_blend'],
                        'surface_relaxation': control_params['surface_relaxation'],
                        'surface_decay_depth': control_params['surface_decay_depth'],
                        'deep_inertia': control_params['deep_inertia'],
                        'deep_anchor': control_params['deep_anchor'],
                        'surface_skin_cooling_coef': control_params['surface_skin_cooling_coef'],
                        'kalman_process_scale': kalman_scales['process'],
                        'kalman_obs_scale': kalman_scales['obs'],
                        'kalman_correlation_length': kalman_scales['correlation_length'],
                        'kalman_forecast_blend': kalman_scales['forecast_blend'],
                        'surface_rmse_proxy': validation_metrics['rmse'],
                    }
                )

        diagnostics.append(
            {
                'Date': pd.Timestamp(df['Date'].iloc[day_idx]),
                'memory_blend': control_params['memory_blend'],
                'surface_relaxation': control_params['surface_relaxation'],
                'surface_decay_depth': control_params['surface_decay_depth'],
                'deep_inertia': control_params['deep_inertia'],
                'deep_anchor': control_params['deep_anchor'],
                'surface_skin_cooling_coef': control_params['surface_skin_cooling_coef'],
                'kalman_process_scale': kalman_scales['process'],
                'kalman_obs_scale': kalman_scales['obs'],
                'kalman_correlation_length': kalman_scales['correlation_length'],
                'kalman_forecast_blend': kalman_scales['forecast_blend'],
                'projection_adjustments': int(projection_adjustments),
                'surface_rmse_proxy': validation_metrics['rmse'],
                'ppo_update_count': int(ppo_update_count),
            }
        )

    return rolled_grid.astype(np.float32), pd.DataFrame(diagnostics), pd.DataFrame(ppo_history)
