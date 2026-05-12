# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import build_initial_condition_profile, build_observation_dataframe, has_profile_observations
from .losses import build_annealed_loss_weights, compute_losses, summarize_window_losses
from .model import LakePINN, model_temperature
from .physics import compute_surface_flux_terms
from .ppo import (
    PPOController,
    apply_ppo_action,
    build_ppo_state,
    compute_ppo_reward,
    export_ppo_policy_bundle,
    normalize_kalman_scales,
)
from .validation import evaluate_blind_ppo_proxy, evaluate_profile_grid, merge_profile_selection_metrics


def _profile_state_values(obs_day, max_depth):
    obs_day = obs_day.sort_values('Depth_m')
    depths = obs_day['Depth_m'].to_numpy(dtype=np.float64)
    temps = obs_day['Temperature_C'].to_numpy(dtype=np.float64)
    finite = np.isfinite(depths) & np.isfinite(temps)
    depths = depths[finite]
    temps = temps[finite]
    if depths.size == 0:
        return None
    surface = float(np.interp(0.0, depths, temps))
    shallow_mask = depths <= 3.0
    shallow = float(np.mean(temps[shallow_mask])) if np.any(shallow_mask) else surface
    deep_floor = max(0.70 * float(max_depth), float(max_depth) - 5.0)
    deep_mask = depths >= deep_floor
    deep = float(np.mean(temps[deep_mask])) if np.any(deep_mask) else float(np.interp(float(max_depth), depths, temps))
    return surface, shallow, deep


def build_causal_previous_state_memory(df, train_profile_obs, max_depth):
    """Build past-only previous-state features without using val/test truth."""
    n_days = len(df)
    surface_series = (
        df['SurfaceBulkTarget_C']
        if 'SurfaceBulkTarget_C' in df.columns
        else df['LST_surface_C']
    )
    fallback_surface = pd.to_numeric(surface_series, errors='coerce').interpolate(limit_direction='both').bfill().ffill()
    prev_surface = np.zeros(n_days, dtype=np.float32)
    prev_shallow = np.zeros(n_days, dtype=np.float32)
    prev_deep = np.zeros(n_days, dtype=np.float32)

    last_surface = float(fallback_surface.iloc[0]) if n_days else DEFAULT_INITIAL_WATER_TEMPERATURE_C
    last_shallow = last_surface
    last_deep = DEFAULT_INITIAL_WATER_TEMPERATURE_C

    obs_by_date = {}
    if has_profile_observations(train_profile_obs):
        obs = train_profile_obs.copy()
        obs['Date'] = pd.to_datetime(obs['Date']).dt.normalize()
        obs['Depth_m'] = pd.to_numeric(obs['Depth_m'], errors='coerce')
        obs['Temperature_C'] = pd.to_numeric(obs['Temperature_C'], errors='coerce')
        obs = obs.dropna(subset=['Date', 'Depth_m', 'Temperature_C'])
        obs_by_date = {date_value: day_obs for date_value, day_obs in obs.groupby('Date')}

    dates = pd.to_datetime(df['Date']).dt.normalize()
    for day_idx, date_value in enumerate(dates):
        prev_surface[day_idx] = np.float32(last_surface)
        prev_shallow[day_idx] = np.float32(last_shallow)
        prev_deep[day_idx] = np.float32(last_deep)

        state_values = _profile_state_values(obs_by_date[date_value], max_depth) if date_value in obs_by_date else None
        if state_values is not None:
            last_surface, last_shallow, last_deep = state_values
        else:
            # Surface LST is available in real prediction; use it only as a
            # causal fallback for the next day's shallow state, never for deep truth.
            fallback_value = float(fallback_surface.iloc[day_idx])
            if np.isfinite(fallback_value):
                last_surface = fallback_value
                last_shallow = fallback_value

    return {
        'prev_surface_temp': prev_surface,
        'prev_0_3m_mean': prev_shallow,
        'prev_deep_mean': prev_deep,
    }


def train_model(
    df,
    metadata,
    max_depth=25.0,
    epochs=2500,
    lr=1e-3,
    collocation_points=512,
    device='cpu',
    train_profile_obs=None,
    ppo_validation_profile_obs=None,
    use_ppo=False,
    ppo_control_interval=50,
    ppo_rollout_steps=8,
    ppo_max_updates_run=None,
    ppo_eval_depth_points=80,
    ppo_use_kalman_reward=False,
    ppo_tune_kalman=False,
    ppo_apply_post_physics=False,
    base_kalman_process_std=0.3,
    base_kalman_obs_std_surface=0.5,
    base_kalman_obs_std_bottom=0.5,
    base_kalman_obs_std_profile=0.75,
    base_kalman_correlation_length=2.0,
    base_kalman_forecast_blend=0.2,
    base_kalman_forecast_spinup_days=0,
    base_kalman_forecast_spinup_max_blend=0.9,
    shallow_optimized_grid=False,
    shallow_focus_depth=5.0,
    shallow_grid_fraction=0.55,
    rolling_prediction_mode=False,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    shortwave_attenuation_coef=SHORTWAVE_ATTENUATION,
    shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    initial_condition_mode='uniform_4c',
    surface_obs_depth_m=0.35,
    time_continuity_weight=5.0,
    time_continuity_depth_points=64,
    stratification_weight=0.6,
    stratification_pairs=64,
    stratification_margin_c=STRATIFICATION_MARGIN_C,
    smoothness_weight=0.15,
    max_vertical_gradient_c_per_m=MAX_VERTICAL_GRADIENT_C_PER_M,
    deep_warming_weight=0.25,
    deep_anchor_weight=0.7,
    deep_anchor_pairs=64,
    deep_anchor_amplitude_c=2.2,
    vertical_exchange_weight=0.35,
    entrainment_velocity_scale_m_per_day=MAX_ENTRAINMENT_VELOCITY_M_PER_DAY,
    convective_mixing_weight=0.25,
    surface_mixed_layer_uniformity_weight=0.18,
    abrupt_surface_cooling_weight=0.12,
    bottom_slow_change_weight=0.10,
    autumn_overturn_weight=0.22,
    heat_budget_weight=0.30,
    heat_budget_depth_points=24,
    profile_grid_physics_weight=0.0,
    profile_grid_day_pairs=12,
    profile_grid_depth_points=41,
    density_reg_weight=0.1,
    train_until_best=False,
    train_min_epochs=200,
    train_patience_windows=6,
    resume_checkpoint_bundle=None,
    model_input_dim=None,
):
    df = df.copy()
    state_memory = build_causal_previous_state_memory(
        df=df,
        train_profile_obs=train_profile_obs,
        max_depth=max_depth,
    )
    for name, values in state_memory.items():
        df[name] = values

    resume_input_dim = None
    if resume_checkpoint_bundle is not None:
        if 'input_dim' in resume_checkpoint_bundle:
            resume_input_dim = int(resume_checkpoint_bundle['input_dim'])
        else:
            resume_state = resume_checkpoint_bundle.get('model_state_dict', {}) or {}
            first_weight = resume_state.get('net.0.weight')
            if first_weight is not None:
                resume_input_dim = int(first_weight.shape[1])
    model_input_dim = int(resume_input_dim if resume_input_dim is not None else (model_input_dim or PINN_INPUT_DIM))
    model = LakePINN(input_dim=model_input_dim, hidden_dim=128, hidden_layers=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250)

    weights = {
        'pde': 1.0e5,
        'bc': 10.0,
        'ic': 5.0,
        'obs': 1.0,
        'time_continuity': float(time_continuity_weight),
        'stratification': float(stratification_weight),
        'smoothness': float(smoothness_weight),
        'deep_warming': float(deep_warming_weight),
        'deep_anchor': float(deep_anchor_weight),
        'vertical_exchange': float(vertical_exchange_weight),
        'convective_mixing': float(convective_mixing_weight),
        'surface_mixed_layer_uniformity': float(surface_mixed_layer_uniformity_weight),
        'abrupt_surface_cooling': float(abrupt_surface_cooling_weight),
        'bottom_slow_change': float(bottom_slow_change_weight),
        'autumn_overturn': float(autumn_overturn_weight),
        'heat_budget': float(heat_budget_weight),
        'profile_grid_physics': float(profile_grid_physics_weight),
        'density_reg': float(density_reg_weight),
    }

    resume_training_info = {}
    if resume_checkpoint_bundle is not None:
        model_state = resume_checkpoint_bundle.get('model_state_dict')
        if model_state is not None:
            model.load_state_dict(model_state)
        optimizer_state = resume_checkpoint_bundle.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = resume_checkpoint_bundle.get('scheduler_state_dict')
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception:
                pass
        resume_training_info = dict(resume_checkpoint_bundle.get('training_info', {}) or {})
        if use_ppo:
            for key, value in dict(resume_training_info.get('final_weights', {}) or {}).items():
                if key in weights:
                    weights[key] = float(value)

    t_all = torch.tensor(df['time_norm'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    doy_sin_all = torch.tensor(df['doy_sin'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    doy_cos_all = torch.tensor(df['doy_cos'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    solar_flux = torch.tensor(df['Solar_W_m2'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    wind_speed = torch.tensor(df['wind_speed_m_per_s'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    air_temp = torch.tensor(df['T_air_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    lst_surface = torch.tensor(df['LST_surface_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    relative_humidity = torch.tensor(df['relative_humidity'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    surface_pressure = torch.tensor(df['surface_pressure_Pa'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    mixed_layer_depth = torch.tensor(df['MixedLayerDepth_m'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    longwave = torch.tensor(df['Longwave_W_m2'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    latent_heat = torch.tensor(df['latent_heat_upward_W_m2'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    sensible_heat = torch.tensor(df['sensible_heat_upward_W_m2'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    secchi = torch.tensor(df['Secchi_m'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    history_forcing = {
        name: torch.tensor(df[name].values.reshape(-1, 1), dtype=torch.float32, device=device)
        for name in (*HISTORY_FORCING_COLUMNS, *STATE_MEMORY_COLUMNS)
        if name in df.columns
    }

    observations, surface_correction_info = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=train_profile_obs,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    obs_time = torch.tensor(observations['time_norm'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_depth = torch.tensor(observations['Depth_m'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_temperature = torch.tensor(
        observations['Temperature_C'].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    obs_doy_sin = torch.tensor(observations['doy_sin'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_doy_cos = torch.tensor(observations['doy_cos'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_air_temp = torch.tensor(observations['T_air_C'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_wind_speed = torch.tensor(observations['wind_speed_m_per_s'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_shortwave = torch.tensor(observations['Solar_W_m2'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_lst_surface = torch.tensor(observations['LST_surface_C'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_longwave = torch.tensor(observations['Longwave_W_m2'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_latent_heat = torch.tensor(observations['latent_heat_upward_W_m2'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_sensible_heat = torch.tensor(observations['sensible_heat_upward_W_m2'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_secchi = torch.tensor(observations['Secchi_m'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_history_forcing = {
        name: torch.tensor(observations[name].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
        for name in (*HISTORY_FORCING_COLUMNS, *STATE_MEMORY_COLUMNS)
        if name in observations.columns
    }
    obs_weight = torch.tensor(
        observations['obs_weight'].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )

    z_ic_np, t_ic_np = build_initial_condition_profile(
        df,
        max_depth=max_depth,
        n_points=64,
        mode=initial_condition_mode,
    )
    ic_depth = torch.tensor(z_ic_np, dtype=torch.float32, device=device)
    ic_time = torch.zeros_like(ic_depth, device=device)
    ic_temperature = torch.tensor(t_ic_np, dtype=torch.float32, device=device)
    ic_doy_sin = torch.full_like(ic_time, float(df['doy_sin'].iloc[0]))
    ic_doy_cos = torch.full_like(ic_time, float(df['doy_cos'].iloc[0]))
    ic_air_temp = torch.full_like(ic_time, float(df['T_air_C'].iloc[0]))
    ic_wind_speed = torch.full_like(ic_time, float(df['wind_speed_m_per_s'].iloc[0]))
    ic_shortwave = torch.full_like(ic_time, float(df['Solar_W_m2'].iloc[0]))
    ic_lst_surface = torch.full_like(ic_time, float(df['LST_surface_C'].iloc[0]))
    ic_longwave = torch.full_like(ic_time, float(df['Longwave_W_m2'].iloc[0]))
    ic_latent_heat = torch.full_like(ic_time, float(df['latent_heat_upward_W_m2'].iloc[0]))
    ic_sensible_heat = torch.full_like(ic_time, float(df['sensible_heat_upward_W_m2'].iloc[0]))
    ic_secchi = torch.full_like(ic_time, float(df['Secchi_m'].iloc[0]))
    ic_history_forcing = {
        name: torch.full_like(ic_time, float(df[name].iloc[0]))
        for name in (*HISTORY_FORCING_COLUMNS, *STATE_MEMORY_COLUMNS)
        if name in df.columns
    }

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    best_selection_metric = float('inf')
    best_selection_label = 'loss'
    best_snapshot = None
    validation_patience_counter = 0
    n_days = len(df)
    date_ns = pd.to_datetime(df['Date']).astype('int64').to_numpy(dtype=np.int64, copy=False)
    time_step_seconds_np = np.diff(date_ns.astype(np.float64)) / 1.0e9 if n_days > 1 else np.empty((0,), dtype=np.float64)
    seq_delta_seconds_all = torch.tensor(
        time_step_seconds_np.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    ) if n_days > 1 else None
    window_losses = []

    kalman_scales = normalize_kalman_scales(
        resume_training_info.get('kalman_scales', {}),
        default_process=1.0,
        default_obs=1.0,
        default_correlation_length=base_kalman_correlation_length,
        default_forecast_blend=base_kalman_forecast_blend,
    )
    ppo_history = []
    ppo_update_stats = []
    ppo_update_count = 0
    ppo_context = {
        'state': None,
        'action': None,
        'log_prob': None,
        'value': None,
        'summary': None,
        'validation': None,
    }
    ppo_controller = None
    ppo_tune_kalman = bool(ppo_tune_kalman)
    if use_ppo:
        ppo_controller = PPOController(state_dim=PPO_STATE_DIM, action_dim=PPO_TRAIN_ACTION_DIM, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_progress = epoch / max(epochs - 1, 1)
        effective_weights = build_annealed_loss_weights(weights, epoch_progress)

        day_pick = torch.randint(low=0, high=n_days, size=(collocation_points,), device=device)
        z_colloc = torch.rand((collocation_points, 1), device=device) * max_depth
        if n_days > 1:
            seq_batch_size = int(max(8, min(collocation_points, time_continuity_depth_points)))
            seq_pick = torch.randint(low=0, high=n_days - 1, size=(seq_batch_size,), device=device)
            seq_depth = torch.rand((seq_batch_size, 1), device=device) * max_depth
            seq_time_now = t_all[seq_pick]
            seq_time_next = t_all[seq_pick + 1]
            seq_doy_sin_now = doy_sin_all[seq_pick]
            seq_doy_cos_now = doy_cos_all[seq_pick]
            seq_doy_sin_next = doy_sin_all[seq_pick + 1]
            seq_doy_cos_next = doy_cos_all[seq_pick + 1]
            seq_solar_flux_now = solar_flux[seq_pick]
            seq_solar_flux_next = solar_flux[seq_pick + 1]
            seq_wind_now = wind_speed[seq_pick]
            seq_wind_next = wind_speed[seq_pick + 1]
            seq_air_temp_now = air_temp[seq_pick]
            seq_air_temp_next = air_temp[seq_pick + 1]
            seq_lst_now = lst_surface[seq_pick]
            seq_lst_next = lst_surface[seq_pick + 1]
            seq_longwave_now = longwave[seq_pick]
            seq_longwave_next = longwave[seq_pick + 1]
            seq_latent_heat_now = latent_heat[seq_pick]
            seq_latent_heat_next = latent_heat[seq_pick + 1]
            seq_sensible_heat_now = sensible_heat[seq_pick]
            seq_sensible_heat_next = sensible_heat[seq_pick + 1]
            seq_secchi_now = secchi[seq_pick]
            seq_secchi_next = secchi[seq_pick + 1]
            seq_doy_now = torch.tensor(
                pd.to_datetime(df['Date'].iloc[seq_pick.detach().cpu().numpy()]).dt.dayofyear.to_numpy(dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            )
            seq_mld_now = mixed_layer_depth[seq_pick]
            seq_mld_next = mixed_layer_depth[seq_pick + 1]
            seq_deep_floor = torch.clamp(
                torch.maximum(seq_mld_now + 2.0, torch.full_like(seq_mld_now, max_depth * 0.42)),
                min=3.0,
                max=max_depth - 0.5,
            )
            seq_delta_seconds = seq_delta_seconds_all[seq_pick]
            seq_surface_depth = torch.zeros_like(seq_time_now)
        else:
            seq_depth = torch.empty((0, 1), dtype=torch.float32, device=device)
            seq_time_now = None
            seq_time_next = None
            seq_doy_sin_now = None
            seq_doy_cos_now = None
            seq_doy_sin_next = None
            seq_doy_cos_next = None
            seq_solar_flux_now = None
            seq_solar_flux_next = None
            seq_wind_now = None
            seq_wind_next = None
            seq_air_temp_now = None
            seq_air_temp_next = None
            seq_lst_now = None
            seq_lst_next = None
            seq_longwave_now = None
            seq_longwave_next = None
            seq_latent_heat_now = None
            seq_latent_heat_next = None
            seq_sensible_heat_now = None
            seq_sensible_heat_next = None
            seq_secchi_now = None
            seq_secchi_next = None
            seq_doy_now = None
            seq_mld_now = None
            seq_mld_next = None
            seq_deep_floor = None
            seq_delta_seconds = None
            seq_surface_depth = None

        strat_batch_size = int(max(8, min(collocation_points, stratification_pairs)))
        strat_pick = torch.randint(low=0, high=n_days, size=(strat_batch_size,), device=device)
        strat_time = t_all[strat_pick]
        strat_doy_sin = doy_sin_all[strat_pick]
        strat_doy_cos = doy_cos_all[strat_pick]
        strat_air_temp = air_temp[strat_pick]
        strat_wind_speed = wind_speed[strat_pick]
        strat_solar = solar_flux[strat_pick]
        strat_lst_surface = lst_surface[strat_pick]
        strat_longwave = longwave[strat_pick]
        strat_latent_heat = latent_heat[strat_pick]
        strat_sensible_heat = sensible_heat[strat_pick]
        strat_secchi = secchi[strat_pick]
        strat_mld = mixed_layer_depth[strat_pick]
        shallow_cap = torch.clamp(torch.maximum(torch.full_like(strat_mld, 0.8), 0.55 * strat_mld + 0.6), min=0.8, max=max_depth * 0.35)
        strat_shallow_depth = torch.rand((strat_batch_size, 1), device=device) * shallow_cap
        deep_floor = torch.clamp(torch.maximum(strat_mld + 1.5, torch.full_like(strat_mld, max_depth * 0.35)), min=2.5, max=max_depth - 0.5)
        deep_span = torch.clamp(max_depth - deep_floor, min=0.5)
        strat_deep_depth = deep_floor + torch.rand((strat_batch_size, 1), device=device) * deep_span
        warm_indicator = torch.sigmoid((strat_air_temp - 8.0) / 2.0)
        solar_indicator = torch.sigmoid((strat_solar - 120.0) / 40.0)
        mld_indicator = torch.sigmoid((strat_mld - 1.5) / 0.8)
        strat_weight = warm_indicator * solar_indicator * mld_indicator
        strat_margin = torch.full_like(strat_weight, float(stratification_margin_c))

        deep_batch_size = int(max(8, min(collocation_points, deep_anchor_pairs)))
        deep_pick = torch.randint(low=0, high=n_days, size=(deep_batch_size,), device=device)
        deep_time = t_all[deep_pick]
        deep_doy_sin = doy_sin_all[deep_pick]
        deep_doy_cos = doy_cos_all[deep_pick]
        deep_air_temp = air_temp[deep_pick]
        deep_wind_speed = wind_speed[deep_pick]
        deep_solar = solar_flux[deep_pick]
        deep_lst_surface = lst_surface[deep_pick]
        deep_longwave = longwave[deep_pick]
        deep_latent_heat = latent_heat[deep_pick]
        deep_sensible_heat = sensible_heat[deep_pick]
        deep_secchi = secchi[deep_pick]
        deep_mld = mixed_layer_depth[deep_pick]
        deep_floor = torch.clamp(
            torch.maximum(deep_mld + 2.5, torch.full_like(deep_mld, max_depth * 0.45)),
            min=4.0,
            max=max_depth - 0.5,
        )
        deep_span = torch.clamp(max_depth - deep_floor, min=0.5)
        deep_depth = deep_floor + torch.rand((deep_batch_size, 1), device=device) * deep_span
        deep_anchor_scale = torch.exp(-(deep_depth - deep_floor) / 4.0)
        deep_target = torch.full_like(deep_depth, DEFAULT_INITIAL_WATER_TEMPERATURE_C) + float(deep_anchor_amplitude_c) * deep_anchor_scale
        deep_warm_indicator = torch.sigmoid((deep_air_temp - 9.0) / 2.0) * torch.sigmoid((deep_solar - 140.0) / 45.0)
        deep_stability_indicator = torch.sigmoid((deep_mld - 1.5) / 0.8)
        deep_weight = deep_warm_indicator * deep_stability_indicator

        if n_days > 1 and float(profile_grid_physics_weight) > 0.0:
            profile_grid_pair_count = int(
                max(2, min(n_days - 1, max(2, int(profile_grid_day_pairs))))
            )
            profile_grid_pick = torch.randint(
                low=0,
                high=n_days - 1,
                size=(profile_grid_pair_count,),
                device=device,
            )
            profile_grid_depth_line = torch.linspace(
                0.0,
                max_depth,
                int(max(8, profile_grid_depth_points)),
                device=device,
            ).reshape(-1, 1)
            profile_grid_depth_count = int(profile_grid_depth_line.shape[0])
            profile_grid_depth = profile_grid_depth_line.repeat(profile_grid_pair_count, 1)
            profile_grid_time_now = t_all[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_time_next = t_all[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_doy_sin_now = doy_sin_all[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_doy_cos_now = doy_cos_all[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_doy_sin_next = doy_sin_all[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_doy_cos_next = doy_cos_all[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_solar_flux_now = solar_flux[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_solar_flux_next = solar_flux[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_wind_now = wind_speed[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_wind_next = wind_speed[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_air_temp_now = air_temp[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_air_temp_next = air_temp[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_lst_now = lst_surface[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_lst_next = lst_surface[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_longwave_now = longwave[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_longwave_next = longwave[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_latent_heat_now = latent_heat[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_latent_heat_next = latent_heat[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_sensible_heat_now = sensible_heat[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_sensible_heat_next = sensible_heat[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_secchi_now = secchi[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_secchi_next = secchi[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            profile_grid_doy_now_by_day = torch.tensor(
                pd.to_datetime(df['Date'].iloc[profile_grid_pick.detach().cpu().numpy()]).dt.dayofyear.to_numpy(dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            )
            profile_grid_delta_seconds_by_day = seq_delta_seconds_all[profile_grid_pick]
        else:
            profile_grid_pick = None
            profile_grid_depth_line = None
            profile_grid_depth = None
            profile_grid_depth_count = 0
            profile_grid_pair_count = 0
            profile_grid_time_now = None
            profile_grid_time_next = None
            profile_grid_doy_sin_now = None
            profile_grid_doy_cos_now = None
            profile_grid_doy_sin_next = None
            profile_grid_doy_cos_next = None
            profile_grid_solar_flux_now = None
            profile_grid_solar_flux_next = None
            profile_grid_wind_now = None
            profile_grid_wind_next = None
            profile_grid_air_temp_now = None
            profile_grid_air_temp_next = None
            profile_grid_lst_now = None
            profile_grid_lst_next = None
            profile_grid_longwave_now = None
            profile_grid_longwave_next = None
            profile_grid_latent_heat_now = None
            profile_grid_latent_heat_next = None
            profile_grid_sensible_heat_now = None
            profile_grid_sensible_heat_next = None
            profile_grid_secchi_now = None
            profile_grid_secchi_next = None
            profile_grid_doy_now_by_day = None
            profile_grid_delta_seconds_by_day = None

        if n_days > 1:
            budget_pick = seq_pick
            budget_batch_size = int(budget_pick.shape[0])
            depth_line = torch.linspace(0.0, max_depth, int(max(8, heat_budget_depth_points)), device=device).reshape(-1, 1)
            budget_depth = depth_line.repeat(budget_batch_size, 1)
            budget_time_now = t_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_time_next = t_all[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            budget_doy_sin_now = doy_sin_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_doy_cos_now = doy_cos_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_doy_sin_next = doy_sin_all[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            budget_doy_cos_next = doy_cos_all[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            budget_delta_seconds = seq_delta_seconds_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_solar_flux = solar_flux[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_solar_flux_next = solar_flux[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            budget_surface_time = t_all[budget_pick]
            budget_air_temp = air_temp[budget_pick]
            budget_air_temp_next = air_temp[budget_pick + 1]
            budget_wind_speed = wind_speed[budget_pick]
            budget_wind_speed_next = wind_speed[budget_pick + 1]
            budget_lst_surface = lst_surface[budget_pick]
            budget_lst_surface_next = lst_surface[budget_pick + 1]
            budget_longwave = longwave[budget_pick]
            budget_longwave_next = longwave[budget_pick + 1]
            budget_latent_heat = latent_heat[budget_pick]
            budget_latent_heat_next = latent_heat[budget_pick + 1]
            budget_sensible_heat = sensible_heat[budget_pick]
            budget_sensible_heat_next = sensible_heat[budget_pick + 1]
            budget_secchi = secchi[budget_pick]
            budget_secchi_next = secchi[budget_pick + 1]
            budget_relative_humidity = relative_humidity[budget_pick]
            budget_surface_pressure = surface_pressure[budget_pick]
            budget_group_index = torch.arange(budget_batch_size, device=device).repeat_interleave(depth_line.shape[0])
            budget_dz_value = float(max_depth) / float(max(int(depth_line.shape[0]) - 1, 1))
            budget_dz = torch.full_like(budget_depth, budget_dz_value)
            budget_surface_depth = torch.zeros_like(budget_surface_time)
            budget_air_temp_now_expanded = budget_air_temp.repeat_interleave(depth_line.shape[0], dim=0)
            budget_air_temp_next_expanded = budget_air_temp_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_wind_speed_now_expanded = budget_wind_speed.repeat_interleave(depth_line.shape[0], dim=0)
            budget_wind_speed_next_expanded = budget_wind_speed_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_lst_surface_now_expanded = budget_lst_surface.repeat_interleave(depth_line.shape[0], dim=0)
            budget_lst_surface_next_expanded = budget_lst_surface_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_longwave_now_expanded = budget_longwave.repeat_interleave(depth_line.shape[0], dim=0)
            budget_longwave_next_expanded = budget_longwave_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_latent_heat_now_expanded = budget_latent_heat.repeat_interleave(depth_line.shape[0], dim=0)
            budget_latent_heat_next_expanded = budget_latent_heat_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_sensible_heat_now_expanded = budget_sensible_heat.repeat_interleave(depth_line.shape[0], dim=0)
            budget_sensible_heat_next_expanded = budget_sensible_heat_next.repeat_interleave(depth_line.shape[0], dim=0)
            budget_secchi_now_expanded = budget_secchi.repeat_interleave(depth_line.shape[0], dim=0)
            budget_secchi_next_expanded = budget_secchi_next.repeat_interleave(depth_line.shape[0], dim=0)
        else:
            budget_depth = None
            budget_time_now = None
            budget_time_next = None
            budget_doy_sin_now = None
            budget_doy_cos_now = None
            budget_doy_sin_next = None
            budget_doy_cos_next = None
            budget_delta_seconds = None
            budget_solar_flux = None
            budget_solar_flux_next = None
            budget_surface_time = None
            budget_air_temp = None
            budget_air_temp_next = None
            budget_wind_speed = None
            budget_wind_speed_next = None
            budget_lst_surface = None
            budget_lst_surface_next = None
            budget_longwave = None
            budget_longwave_next = None
            budget_latent_heat = None
            budget_latent_heat_next = None
            budget_sensible_heat = None
            budget_sensible_heat_next = None
            budget_secchi = None
            budget_secchi_next = None
            budget_relative_humidity = None
            budget_surface_pressure = None
            budget_group_index = None
            budget_dz = None
            budget_surface_depth = None
            budget_air_temp_now_expanded = None
            budget_air_temp_next_expanded = None
            budget_wind_speed_now_expanded = None
            budget_wind_speed_next_expanded = None
            budget_lst_surface_now_expanded = None
            budget_lst_surface_next_expanded = None
            budget_longwave_now_expanded = None
            budget_longwave_next_expanded = None
            budget_latent_heat_now_expanded = None
            budget_latent_heat_next_expanded = None
            budget_sensible_heat_now_expanded = None
            budget_sensible_heat_next_expanded = None
            budget_secchi_now_expanded = None
            budget_secchi_next_expanded = None

        batch = {
            't_colloc': t_all[day_pick],
            'z_colloc': z_colloc,
            'doy_sin_colloc': doy_sin_all[day_pick],
            'doy_cos_colloc': doy_cos_all[day_pick],
            'solar_flux_colloc': solar_flux[day_pick],
            'wind_colloc': wind_speed[day_pick],
            'air_temp_colloc': air_temp[day_pick],
            'lst_surface_colloc': lst_surface[day_pick],
            'colloc_longwave': longwave[day_pick],
            'colloc_latent_heat': latent_heat[day_pick],
            'colloc_sensible_heat': sensible_heat[day_pick],
            'colloc_secchi': secchi[day_pick],
            'surface_time': t_all,
            'surface_doy_sin': doy_sin_all,
            'surface_doy_cos': doy_cos_all,
            'surface_shortwave': solar_flux,
            'surface_air_temp': air_temp,
            'surface_wind_speed': wind_speed,
            'surface_lst': lst_surface,
            'surface_longwave': longwave,
            'surface_latent_heat': latent_heat,
            'surface_sensible_heat': sensible_heat,
            'surface_secchi': secchi,
            'surface_relative_humidity': relative_humidity,
            'surface_pressure': surface_pressure,
            'ic_time': ic_time,
            'ic_depth': ic_depth,
            'ic_temperature': ic_temperature,
            'ic_doy_sin': ic_doy_sin,
            'ic_doy_cos': ic_doy_cos,
            'ic_air_temp': ic_air_temp,
            'ic_wind_speed': ic_wind_speed,
            'ic_shortwave': ic_shortwave,
            'ic_lst_surface': ic_lst_surface,
            'ic_longwave': ic_longwave,
            'ic_latent_heat': ic_latent_heat,
            'ic_sensible_heat': ic_sensible_heat,
            'ic_secchi': ic_secchi,
            'obs_time': obs_time,
            'obs_depth': obs_depth,
            'obs_temperature': obs_temperature,
            'obs_doy_sin': obs_doy_sin,
            'obs_doy_cos': obs_doy_cos,
            'obs_air_temp': obs_air_temp,
            'obs_wind_speed': obs_wind_speed,
            'obs_shortwave': obs_shortwave,
            'obs_lst_surface': obs_lst_surface,
            'obs_longwave': obs_longwave,
            'obs_latent_heat': obs_latent_heat,
            'obs_sensible_heat': obs_sensible_heat,
            'obs_secchi': obs_secchi,
            'obs_weight': obs_weight,
            'seq_time_now': seq_time_now,
            'seq_time_next': seq_time_next,
            'seq_depth': seq_depth,
            'seq_doy_sin_now': seq_doy_sin_now,
            'seq_doy_cos_now': seq_doy_cos_now,
            'seq_doy_sin_next': seq_doy_sin_next,
            'seq_doy_cos_next': seq_doy_cos_next,
            'seq_solar_flux_now': seq_solar_flux_now,
            'seq_solar_flux_next': seq_solar_flux_next,
            'seq_wind_now': seq_wind_now,
            'seq_wind_next': seq_wind_next,
            'seq_air_temp_now': seq_air_temp_now,
            'seq_air_temp_next': seq_air_temp_next,
            'seq_lst_now': seq_lst_now,
            'seq_lst_next': seq_lst_next,
            'seq_now_longwave': seq_longwave_now,
            'seq_next_longwave': seq_longwave_next,
            'seq_now_latent_heat': seq_latent_heat_now,
            'seq_next_latent_heat': seq_latent_heat_next,
            'seq_now_sensible_heat': seq_sensible_heat_now,
            'seq_next_sensible_heat': seq_sensible_heat_next,
            'seq_now_secchi': seq_secchi_now,
            'seq_next_secchi': seq_secchi_next,
            'seq_doy_now': seq_doy_now,
            'seq_mld_now': seq_mld_now,
            'seq_mld_next': seq_mld_next,
            'seq_deep_floor': seq_deep_floor,
            'seq_delta_seconds': seq_delta_seconds,
            'seq_surface_depth': seq_surface_depth,
            'strat_time': strat_time,
            'strat_doy_sin': strat_doy_sin,
            'strat_doy_cos': strat_doy_cos,
            'strat_air_temp': strat_air_temp,
            'strat_wind_speed': strat_wind_speed,
            'strat_shortwave': strat_solar,
            'strat_lst_surface': strat_lst_surface,
            'strat_longwave': strat_longwave,
            'strat_latent_heat': strat_latent_heat,
            'strat_sensible_heat': strat_sensible_heat,
            'strat_secchi': strat_secchi,
            'strat_shallow_depth': strat_shallow_depth,
            'strat_deep_depth': strat_deep_depth,
            'strat_weight': strat_weight,
            'strat_margin': strat_margin,
            'deep_anchor_time': deep_time,
            'deep_anchor_doy_sin': deep_doy_sin,
            'deep_anchor_doy_cos': deep_doy_cos,
            'deep_anchor_air_temp': deep_air_temp,
            'deep_anchor_wind_speed': deep_wind_speed,
            'deep_anchor_shortwave': deep_solar,
            'deep_anchor_lst_surface': deep_lst_surface,
            'deep_anchor_longwave': deep_longwave,
            'deep_anchor_latent_heat': deep_latent_heat,
            'deep_anchor_sensible_heat': deep_sensible_heat,
            'deep_anchor_secchi': deep_secchi,
            'deep_anchor_depth': deep_depth,
            'deep_anchor_target': deep_target,
            'deep_anchor_weight': deep_weight,
            'budget_depth': budget_depth,
            'budget_time_now': budget_time_now,
            'budget_time_next': budget_time_next,
            'budget_doy_sin_now': budget_doy_sin_now,
            'budget_doy_cos_now': budget_doy_cos_now,
            'budget_doy_sin_next': budget_doy_sin_next,
            'budget_doy_cos_next': budget_doy_cos_next,
            'budget_delta_seconds': budget_delta_seconds,
            'budget_solar_flux': budget_solar_flux,
            'budget_solar_flux_next': budget_solar_flux_next,
            'budget_surface_time': budget_surface_time,
            'budget_air_temp': budget_air_temp,
            'budget_air_temp_next': budget_air_temp_next,
            'budget_air_temp_now_expanded': budget_air_temp_now_expanded,
            'budget_air_temp_next_expanded': budget_air_temp_next_expanded,
            'budget_wind_speed': budget_wind_speed,
            'budget_wind_speed_next': budget_wind_speed_next,
            'budget_wind_speed_now_expanded': budget_wind_speed_now_expanded,
            'budget_wind_speed_next_expanded': budget_wind_speed_next_expanded,
            'budget_lst_surface': budget_lst_surface,
            'budget_lst_surface_next': budget_lst_surface_next,
            'budget_lst_surface_now_expanded': budget_lst_surface_now_expanded,
            'budget_lst_surface_next_expanded': budget_lst_surface_next_expanded,
            'budget_now_expanded_longwave': budget_longwave_now_expanded,
            'budget_next_expanded_longwave': budget_longwave_next_expanded,
            'budget_now_expanded_latent_heat': budget_latent_heat_now_expanded,
            'budget_next_expanded_latent_heat': budget_latent_heat_next_expanded,
            'budget_now_expanded_sensible_heat': budget_sensible_heat_now_expanded,
            'budget_next_expanded_sensible_heat': budget_sensible_heat_next_expanded,
            'budget_now_expanded_secchi': budget_secchi_now_expanded,
            'budget_next_expanded_secchi': budget_secchi_next_expanded,
            'budget_relative_humidity': budget_relative_humidity,
            'budget_surface_pressure': budget_surface_pressure,
            'budget_group_index': budget_group_index,
            'budget_dz': budget_dz,
            'budget_surface_depth': budget_surface_depth,
            'profile_grid_day_count': profile_grid_pair_count,
            'profile_grid_depth_count': profile_grid_depth_count,
            'profile_grid_depth_line': profile_grid_depth_line,
            'profile_grid_depth': profile_grid_depth,
            'profile_grid_time_now': profile_grid_time_now,
            'profile_grid_time_next': profile_grid_time_next,
            'profile_grid_doy_sin_now': profile_grid_doy_sin_now,
            'profile_grid_doy_cos_now': profile_grid_doy_cos_now,
            'profile_grid_doy_sin_next': profile_grid_doy_sin_next,
            'profile_grid_doy_cos_next': profile_grid_doy_cos_next,
            'profile_grid_solar_flux_now': profile_grid_solar_flux_now,
            'profile_grid_solar_flux_next': profile_grid_solar_flux_next,
            'profile_grid_wind_now': profile_grid_wind_now,
            'profile_grid_wind_next': profile_grid_wind_next,
            'profile_grid_air_temp_now': profile_grid_air_temp_now,
            'profile_grid_air_temp_next': profile_grid_air_temp_next,
            'profile_grid_lst_now': profile_grid_lst_now,
            'profile_grid_lst_next': profile_grid_lst_next,
            'profile_grid_now_longwave': profile_grid_longwave_now,
            'profile_grid_next_longwave': profile_grid_longwave_next,
            'profile_grid_now_latent_heat': profile_grid_latent_heat_now,
            'profile_grid_next_latent_heat': profile_grid_latent_heat_next,
            'profile_grid_now_sensible_heat': profile_grid_sensible_heat_now,
            'profile_grid_next_sensible_heat': profile_grid_sensible_heat_next,
            'profile_grid_now_secchi': profile_grid_secchi_now,
            'profile_grid_next_secchi': profile_grid_secchi_next,
            'profile_grid_doy_now_by_day': profile_grid_doy_now_by_day,
            'profile_grid_delta_seconds_by_day': profile_grid_delta_seconds_by_day,
        }
        for name, values in history_forcing.items():
            batch[f'colloc_{name}'] = values[day_pick]
            batch[f'surface_{name}'] = values
            batch[f'ic_{name}'] = ic_history_forcing.get(name)
            batch[f'obs_{name}'] = obs_history_forcing.get(name)
            batch[f'strat_{name}'] = values[strat_pick]
            batch[f'deep_anchor_{name}'] = values[deep_pick]
            if seq_time_now is not None:
                batch[f'seq_now_{name}'] = values[seq_pick]
                batch[f'seq_next_{name}'] = values[seq_pick + 1]
            else:
                batch[f'seq_now_{name}'] = None
                batch[f'seq_next_{name}'] = None
            if budget_surface_time is not None:
                batch[f'budget_now_expanded_{name}'] = values[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
                batch[f'budget_next_expanded_{name}'] = values[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            else:
                batch[f'budget_now_expanded_{name}'] = None
                batch[f'budget_next_expanded_{name}'] = None
            if profile_grid_pick is not None:
                batch[f'profile_grid_now_{name}'] = values[profile_grid_pick].repeat_interleave(profile_grid_depth_count, dim=0)
                batch[f'profile_grid_next_{name}'] = values[profile_grid_pick + 1].repeat_interleave(profile_grid_depth_count, dim=0)
            else:
                batch[f'profile_grid_now_{name}'] = None
                batch[f'profile_grid_next_{name}'] = None

        if budget_surface_time is not None:
            budget_history_kwargs = {
                name: values[budget_pick]
                for name, values in history_forcing.items()
            }
            budget_surface_temp = model_temperature(
                model,
                budget_surface_time,
                budget_surface_depth,
                max_depth,
                metadata=metadata,
                doy_sin=doy_sin_all[budget_pick],
                doy_cos=doy_cos_all[budget_pick],
                air_temp=budget_air_temp,
                wind_speed=budget_wind_speed,
                shortwave=solar_flux[budget_pick],
                lst_surface=budget_lst_surface,
                longwave=budget_longwave,
                latent_heat=budget_latent_heat,
                sensible_heat=budget_sensible_heat,
                secchi=budget_secchi,
                **budget_history_kwargs,
            )
            budget_flux_terms = compute_surface_flux_terms(
                budget_surface_temp,
                {
                    'surface_air_temp': budget_air_temp,
                    'surface_wind_speed': budget_wind_speed,
                    'surface_relative_humidity': budget_relative_humidity,
                    'surface_pressure': budget_surface_pressure,
                    'surface_shortwave': solar_flux[budget_pick],
                },
                shortwave_surface_fraction=shortwave_surface_fraction,
            )
            batch['budget_surface_flux'] = budget_flux_terms['seb_flux']
        else:
            batch['budget_surface_flux'] = None

        losses = compute_losses(
            model=model,
            batch=batch,
            max_depth=max_depth,
            time_scale_seconds=metadata['time_scale_seconds'],
            metadata=metadata,
            weights=effective_weights,
            shortwave_attenuation=shortwave_attenuation_coef,
            shortwave_surface_fraction=shortwave_surface_fraction,
            max_vertical_gradient_c_per_m=max_vertical_gradient_c_per_m,
            entrainment_velocity_scale_m_per_day=entrainment_velocity_scale_m_per_day,
        )
        losses['total'].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(losses['total'].detach())
        if use_ppo:
            window_losses.append(
                {
                    'total': losses['total'].item(),
                    'loss_pde': losses['loss_pde'].item(),
                    'loss_bc': losses['loss_bc'].item(),
                        'loss_ic': losses['loss_ic'].item(),
                        'loss_obs': losses['loss_obs'].item(),
                        'loss_time_continuity': losses['loss_time_continuity'].item(),
                        'loss_temporal_jump': losses['loss_temporal_jump'].item(),
                        'loss_stratification': losses['loss_stratification'].item(),
                        'loss_smoothness': losses['loss_smoothness'].item(),
                        'loss_deep_warming': losses['loss_deep_warming'].item(),
                        'loss_deep_anchor': losses['loss_deep_anchor'].item(),
                        'loss_vertical_exchange': losses['loss_vertical_exchange'].item(),
                        'loss_convective_mixing': losses['loss_convective_mixing'].item(),
                        'loss_surface_mixed_layer_uniformity': losses['loss_surface_mixed_layer_uniformity'].item(),
                        'loss_abrupt_surface_cooling': losses['loss_abrupt_surface_cooling'].item(),
                        'loss_bottom_slow_change': losses['loss_bottom_slow_change'].item(),
                        'loss_autumn_overturn': losses['loss_autumn_overturn'].item(),
                        'loss_heat_budget': losses['loss_heat_budget'].item(),
                        'loss_profile_grid_physics': losses['loss_profile_grid_physics'].item(),
                        'kappa_mean': losses['kappa_mean'].item(),
                    'ri_mean': losses['ri_mean'].item(),
                    }
            )

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | total={losses['total'].item():.4f} | "
                f"pde={losses['loss_pde'].item():.6e} | bc={losses['loss_bc'].item():.6e} | "
                f"ic={losses['loss_ic'].item():.4f} | obs={losses['loss_obs'].item():.4f} | "
                f"tc={losses['loss_time_continuity'].item():.4f} | jump={losses['loss_temporal_jump'].item():.4f} | strat={losses['loss_stratification'].item():.4f} | smooth={losses['loss_smoothness'].item():.4f} | deepwarm={losses['loss_deep_warming'].item():.4f} | deep={losses['loss_deep_anchor'].item():.4f} | "
                f"adv={losses['loss_vertical_exchange'].item():.4f} | conv={losses['loss_convective_mixing'].item():.4f} | mix={losses['loss_surface_mixed_layer_uniformity'].item():.4f} | drop={losses['loss_abrupt_surface_cooling'].item():.4f} | bottomslow={losses['loss_bottom_slow_change'].item():.4f} | overturn={losses['loss_autumn_overturn'].item():.4f} | heat={losses['loss_heat_budget'].item():.4f} | grid={losses['loss_profile_grid_physics'].item():.4f} | "
                f"grid_jump={losses['profile_grid_day_jump_rms'].item():.3f} | grid_surf={losses['profile_grid_surface_jump_rms'].item():.3f} | grid_col={losses['profile_grid_column_jump_rms'].item():.3f} | grid_dens={losses['profile_grid_density_rms'].item():.3e} | "
                f"kappa={losses['kappa_mean'].item():.3e} | Ri={losses['ri_mean'].item():.3f} | "
                f"obs_w={effective_weights['obs']:.3f} | strat_w={effective_weights['stratification']:.3f} | smooth_w={effective_weights['smoothness']:.3f} | deepwarm_w={effective_weights['deep_warming']:.3f} | deep_w={effective_weights['deep_anchor']:.3f} | "
                f"adv_w={effective_weights['vertical_exchange']:.3f} | conv_w={effective_weights['convective_mixing']:.3f} | mix_w={effective_weights['surface_mixed_layer_uniformity']:.3f} | drop_w={effective_weights['abrupt_surface_cooling']:.3f} | bottomslow_w={effective_weights.get('bottom_slow_change', 0.0):.3f} | overturn_w={effective_weights['autumn_overturn']:.3f} | heat_w={effective_weights['heat_budget']:.3f} | grid_w={effective_weights.get('profile_grid_physics', 0.0):.3f}"
            )

        reached_window_end = ((epoch + 1) % max(ppo_control_interval, 1) == 0) or (epoch == epochs - 1)
        if use_ppo and reached_window_end:
            current_summary = summarize_window_losses(window_losses)
            eval_depth_points = int(max(20, min(ppo_eval_depth_points, 160)))
            from .predict import predict_temperature_grid

            eval_grid, eval_depths, _ = predict_temperature_grid(
                model,
                df=df,
                metadata=metadata,
                max_depth=max_depth,
                n_depth_points=eval_depth_points,
                device=device,
                apply_post_physics=ppo_apply_post_physics,
                use_shallow_optimized=shallow_optimized_grid,
                shallow_focus_depth=shallow_focus_depth,
                shallow_fraction=shallow_grid_fraction,
                rolling_prediction_mode=rolling_prediction_mode,
                rolling_memory_blend=rolling_memory_blend,
                rolling_surface_relaxation=rolling_surface_relaxation,
                rolling_surface_decay_depth=rolling_surface_decay_depth,
                surface_skin_cooling_coef=surface_skin_cooling_coef,
            )
            current_validation = evaluate_blind_ppo_proxy(
                df=df,
                temp_grid=eval_grid,
                depths=eval_depths,
            )
            selection_metrics = None
            if has_profile_observations(ppo_validation_profile_obs):
                selection_metrics = evaluate_profile_grid(
                    df=df,
                    metadata=metadata,
                    temp_grid=eval_grid,
                    depths=eval_depths,
                    max_depth=max_depth,
                    profile_obs_data=ppo_validation_profile_obs,
                )
                current_validation = merge_profile_selection_metrics(current_validation, selection_metrics)

            current_state = build_ppo_state(
                summary=current_summary,
                weights=effective_weights,
                kalman_scales=kalman_scales,
                learning_rate=optimizer.param_groups[0]['lr'],
                validation_metrics=current_validation,
            )

            if ppo_context['state'] is not None:
                reward = compute_ppo_reward(
                    prev_summary=ppo_context['summary'],
                    current_summary=current_summary,
                    prev_validation_metrics=ppo_context['validation'],
                    current_validation_metrics=current_validation,
                )
                done = bool(epoch == epochs - 1)
                ppo_controller.store_transition(
                    state=ppo_context['state'],
                    action=ppo_context['action'],
                    log_prob=ppo_context['log_prob'],
                    reward=reward,
                    done=done,
                    value=ppo_context['value'],
                )
                ppo_history.append(
                    {
                        'epoch': epoch,
                        'reward': reward,
                        'lambda_pde': effective_weights['pde'],
                        'lambda_bc': effective_weights['bc'],
                        'lambda_ic': effective_weights['ic'],
                        'lambda_obs': effective_weights['obs'],
                        'lambda_time_continuity': effective_weights['time_continuity'],
                        'lambda_stratification': effective_weights['stratification'],
                        'lambda_smoothness': effective_weights['smoothness'],
                        'lambda_deep_warming': effective_weights['deep_warming'],
                        'lambda_deep_anchor': effective_weights['deep_anchor'],
                        'lambda_surface_mixed_layer_uniformity': effective_weights['surface_mixed_layer_uniformity'],
                        'lambda_abrupt_surface_cooling': effective_weights['abrupt_surface_cooling'],
                        'lambda_bottom_slow_change': effective_weights.get('bottom_slow_change', 0.0),
                        'kalman_process_scale': kalman_scales['process'],
                        'kalman_obs_scale': kalman_scales['obs'],
                        'kalman_correlation_length': kalman_scales['correlation_length'],
                        'kalman_forecast_blend': kalman_scales['forecast_blend'],
                        'window_total': current_summary['total'],
                        'window_obs': current_summary['loss_obs'],
                        'validation_rmse': np.nan if current_validation is None else current_validation['rmse'],
                        'validation_profile_rmse': np.nan if current_validation is None else current_validation.get('profile_rmse', np.nan),
                        'validation_profile_mae': np.nan if current_validation is None else current_validation.get('profile_mae', np.nan),
                        'validation_profile_bias': np.nan if current_validation is None else current_validation.get('profile_bias', np.nan),
                        'validation_profile_objective': np.nan if current_validation is None else current_validation.get('profile_objective', np.nan),
                    }
                )

                if done or len(ppo_controller.buffer['states']) >= max(ppo_rollout_steps, 1):
                    reached_update_cap = (
                        ppo_max_updates_run is not None
                        and ppo_update_count >= int(max(ppo_max_updates_run, 0))
                    )
                    if reached_update_cap:
                        ppo_controller.reset_buffer()
                    else:
                        update_stats = ppo_controller.update(last_state=current_state, last_done=done)
                        if update_stats is not None:
                            update_stats['epoch'] = epoch
                            ppo_update_count += 1
                            update_stats['update_index'] = ppo_update_count
                            ppo_update_stats.append(update_stats)

            if train_until_best:
                selection_metric = None
                selection_label = None
                if selection_metrics is not None and np.isfinite(selection_metrics.get('objective', np.nan)):
                    selection_metric = float(selection_metrics['objective'])
                    selection_label = 'val_profile_objective'
                elif selection_metrics is not None and np.isfinite(selection_metrics.get('rmse', np.nan)):
                    selection_metric = float(selection_metrics['rmse'])
                    selection_label = 'val_rmse'
                elif current_validation is not None and np.isfinite(current_validation.get('rmse', np.nan)):
                    selection_metric = float(current_validation['rmse'])
                    selection_label = 'blind_rmse'
                if selection_metric is not None:
                    if selection_metric < best_selection_metric - 1e-6:
                        best_selection_metric = selection_metric
                        best_selection_label = selection_label
                        validation_patience_counter = 0
                        best_snapshot = {
                            'model_state': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            'weights': dict(weights),
                            'kalman_scales': dict(kalman_scales),
                            'ppo_state_dict': None if ppo_controller is None else copy.deepcopy(ppo_controller.model.state_dict()),
                        }
                    elif epoch + 1 >= int(max(train_min_epochs, 1)):
                        validation_patience_counter += 1

            if epoch != epochs - 1:
                action, log_prob, value = ppo_controller.select_action(current_state)
                weights, kalman_scales = apply_ppo_action(
                    weights,
                    kalman_scales,
                    action,
                    tune_kalman=ppo_tune_kalman,
                )
                ppo_context = {
                    'state': current_state,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'summary': current_summary,
                    'validation': current_validation,
                }

            window_losses = []
            if train_until_best and epoch + 1 >= int(max(train_min_epochs, 1)) and validation_patience_counter >= int(max(train_patience_windows, 1)):
                print(
                    f"Validation early stopping at epoch {epoch} | "
                    f"best {best_selection_label}={best_selection_metric:.4f}"
                )
                break

        total_value = losses['total'].item()
        if total_value < best_loss:
            best_loss = total_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 800:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_snapshot is not None:
        model.load_state_dict(best_snapshot['model_state'])
        weights = dict(best_snapshot['weights'])
        kalman_scales = normalize_kalman_scales(best_snapshot['kalman_scales'])
        if ppo_controller is not None and best_snapshot['ppo_state_dict'] is not None:
            ppo_controller.model.load_state_dict(best_snapshot['ppo_state_dict'])

    training_info = {
        'final_weights': dict(weights),
        'kalman_scales': dict(kalman_scales),
        'ppo_history': pd.DataFrame(ppo_history),
        'ppo_update_stats': pd.DataFrame(ppo_update_stats),
        'use_ppo': bool(use_ppo),
        'ppo_policy_bundle': export_ppo_policy_bundle(ppo_controller if use_ppo else None, weights, kalman_scales),
        'surface_correction_info': surface_correction_info,
        'best_selection_metric': None if not np.isfinite(best_selection_metric) else float(best_selection_metric),
        'best_selection_label': best_selection_label,
        'ppo_update_count': int(ppo_update_count),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'scheduler_state_dict': copy.deepcopy(scheduler.state_dict()),
    }
    return model, training_info
