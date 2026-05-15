# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import has_profile_observations, load_optional_profile_observations
from .losses import summarize_profile_error_metrics
from .physics import water_density_numpy

def evaluate_blind_ppo_proxy(df, temp_grid, depths):
    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    if temp_grid.ndim != 2 or temp_grid.size == 0:
        return None

    if 'SurfaceBulkTarget_C' in df.columns:
        surface_target = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    else:
        surface_target = df['LST_surface_C'].to_numpy(dtype=np.float64)
    raw_lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64) if 'LST_surface_C' in df.columns else surface_target.copy()
    air_temp_series = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else surface_target.copy()
    wind_speed_series = (
        df['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
        if 'wind_speed_m_per_s' in df.columns
        else np.full(temp_grid.shape[1], 2.0, dtype=np.float64)
    )
    solar_series = (
        df['Solar_W_m2'].to_numpy(dtype=np.float64)
        if 'Solar_W_m2' in df.columns
        else np.zeros(temp_grid.shape[1], dtype=np.float64)
    )

    surface_pred = temp_grid[0, :]
    surface_errors = surface_pred - surface_target
    surface_rmse = float(np.sqrt(np.mean(surface_errors ** 2)))
    surface_mae = float(np.mean(np.abs(surface_errors)))
    surface_bias = float(np.mean(surface_errors))
    warm_surface_bias = float(np.mean(np.clip(surface_errors, 0.0, None)))

    lst_jumps = np.abs(np.diff(raw_lst_surface, prepend=raw_lst_surface[0]))
    skin_bulk_gap = np.abs(raw_lst_surface - surface_target)
    lst_spike_signal = np.clip((lst_jumps - 2.5) / 4.0, 0.0, 1.5)
    skin_gap_signal = np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
    low_wind_signal = np.exp(-np.clip(wind_speed_series, 0.0, 12.0) / 2.5)
    high_solar_signal = np.clip((solar_series - 120.0) / 250.0, 0.0, 1.0)
    air_gap_signal = np.clip((np.abs(raw_lst_surface - air_temp_series) - 4.0) / 8.0, 0.0, 1.0)
    lst_spike_indicator = float(
        np.mean(
            np.clip(
                lst_spike_signal * (0.45 + 0.55 * low_wind_signal)
                + 0.35 * skin_gap_signal
                + 0.20 * high_solar_signal
                + 0.15 * air_gap_signal,
                0.0,
                2.0,
            )
        )
    )

    density_grid = water_density_numpy(temp_grid)
    instability = np.clip(-(density_grid[1:, :] - density_grid[:-1, :]), 0.0, None)
    instability_penalty = float(np.mean(instability))

    warm_season_mask = df['T_air_C'].to_numpy(dtype=np.float64) > 9.0 if 'T_air_C' in df.columns else np.ones(temp_grid.shape[1], dtype=bool)
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(temp_grid.shape[1], 2.0, dtype=np.float64)
    deep_warm_penalties = []
    for day_idx in range(temp_grid.shape[1]):
        if not bool(warm_season_mask[day_idx]):
            continue
        day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
        deep_floor = max(day_mld + 2.5, depths[-1] * 0.45)
        anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
        deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
        deep_excess = np.maximum(temp_grid[:, day_idx] - deep_anchor_profile, 0.0)
        deep_warm_penalties.append(float(np.mean(anchor_mask * deep_excess)))
    deep_warm_penalty = float(np.mean(deep_warm_penalties)) if deep_warm_penalties else 0.0
    deep_warm_peak = float(np.max(deep_warm_penalties)) if deep_warm_penalties else 0.0

    dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else None
    if dates is not None:
        doy = dates.dt.dayofyear.to_numpy(dtype=np.int32)
    else:
        doy = np.arange(1, temp_grid.shape[1] + 1, dtype=np.int32)
    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else np.full(temp_grid.shape[1], 10.0, dtype=np.float64)
    max_depth = float(depths[-1]) if depths.size else 1.0
    deep_ref_depth = float(min(9.0, max(5.0, max_depth * 0.45)))
    deep_ref_temp = np.array(
        [np.interp(deep_ref_depth, depths, temp_grid[:, day_idx]) for day_idx in range(temp_grid.shape[1])],
        dtype=np.float64,
    )

    depth_mids = 0.5 * (depths[:-1] + depths[1:]) if depths.size > 1 else np.array([], dtype=np.float64)
    dz = np.diff(depths) if depths.size > 1 else np.array([], dtype=np.float64)
    gradients = np.diff(temp_grid, axis=0) / np.maximum(dz[:, None], 1e-6) if dz.size else np.zeros((0, temp_grid.shape[1]), dtype=np.float64)

    summer_mask = (doy >= 160) & (doy <= 250)
    summer_strat_penalty = 0.0
    summer_thermocline_depth_norm = 0.0
    summer_thermocline_thickness_penalty = 0.0
    summer_surface_warming_reward = 0.0
    summer_midlayer_temp_reward = 0.0
    if np.any(summer_mask):
        summer_gap = surface_pred[summer_mask] - deep_ref_temp[summer_mask]
        summer_strat_penalty = float(np.mean(np.clip(6.0 - summer_gap, 0.0, None) / 6.0))
        summer_surface_warming_reward = float(np.mean(np.clip((surface_pred[summer_mask] - 18.0) / 6.0, 0.0, 1.5)))
        mid_ref_depth = float(min(6.0, max(4.0, max_depth * 0.30)))
        mid_ref_temp = np.array(
            [np.interp(mid_ref_depth, depths, temp_grid[:, day_idx]) for day_idx in range(temp_grid.shape[1])],
            dtype=np.float64,
        )
        mid_temp_error = np.abs(mid_ref_temp[summer_mask] - 13.0)
        summer_midlayer_temp_reward = float(np.mean(np.clip(1.0 - mid_temp_error / 5.0, 0.0, 1.0)))
        thermo_band = (depth_mids >= 1.0) & (depth_mids <= min(12.0, max_depth - 0.5))
        if np.any(thermo_band):
            thermo_strength = np.clip(-gradients[thermo_band][:, summer_mask], 0.0, None)
            thermo_idx = np.argmax(thermo_strength, axis=0)
            thermo_depths = depth_mids[thermo_band][thermo_idx]
            summer_thermocline_depth_norm = float(np.mean(thermo_depths / max(max_depth, 1.0)))
            thermo_band_depths = depth_mids[thermo_band]
            thickness_penalties = []
            for col_idx in range(thermo_strength.shape[1]):
                weights = thermo_strength[:, col_idx]
                peak_strength = float(np.max(weights))
                if peak_strength <= 1.0e-6:
                    thickness_penalties.append(1.0)
                    continue
                norm_weights = weights / np.maximum(np.sum(weights), 1.0e-8)
                mean_depth = float(np.sum(thermo_band_depths * norm_weights))
                std_depth = float(np.sqrt(np.sum(((thermo_band_depths - mean_depth) ** 2) * norm_weights)))
                thickness_penalties.append(max(std_depth - 1.2, 0.0) / 1.2)
            if thickness_penalties:
                summer_thermocline_thickness_penalty = float(np.mean(thickness_penalties))
    summer_9m_temp = float(np.mean(deep_ref_temp[summer_mask])) if np.any(summer_mask) else float(np.mean(deep_ref_temp))
    summer_bottom_temp = (
        float(np.mean(temp_grid[-1, summer_mask])) if np.any(summer_mask) else float(np.mean(temp_grid[-1, :]))
    )

    autumn_mask = (doy >= 280) & (doy <= 330) & (air_temp <= 15.0)
    autumn_overturn_penalty = 0.0
    if np.any(autumn_mask):
        autumn_gap = np.abs(surface_pred[autumn_mask] - temp_grid[-1, autumn_mask])
        autumn_overturn_penalty = float(np.mean(np.clip(autumn_gap - 1.2, 0.0, None)))
    autumn_surface_cooling_rate = 0.0
    autumn_gap_collapse = 0.0
    autumn_false_overturn_penalty = 0.0
    autumn_cooling_triggered_overturn_reward = 0.0
    cooling_window = 7
    if temp_grid.shape[1] > cooling_window:
        gap_series = np.maximum(surface_pred - temp_grid[-1, :], 0.0)
        autumn_window_mask = autumn_mask[:-cooling_window] & autumn_mask[cooling_window:]
        if np.any(autumn_window_mask):
            cooling = np.maximum(surface_pred[:-cooling_window] - surface_pred[cooling_window:], 0.0)
            gap_collapse = np.maximum(gap_series[:-cooling_window] - gap_series[cooling_window:], 0.0)
            deep_warming = np.maximum(temp_grid[-1, cooling_window:] - temp_grid[-1, :-cooling_window], 0.0)
            cooling = cooling[autumn_window_mask]
            gap_collapse = gap_collapse[autumn_window_mask]
            deep_warming = deep_warming[autumn_window_mask]
            if cooling.size:
                autumn_surface_cooling_rate = float(np.mean(cooling))
                cooling_gate = np.clip(cooling / 0.5, 0.0, 1.0)
                autumn_gap_collapse = float(np.mean(gap_collapse * cooling_gate))
                false_collapse = np.maximum(gap_collapse - 1.25 * cooling, 0.0)
                false_warming = np.maximum(deep_warming - 0.08, 0.0)
                autumn_false_overturn_penalty = float(np.mean(false_collapse + 1.5 * false_warming))
                overturn_success = (
                    np.clip(cooling / 0.6, 0.0, 1.5)
                    * np.clip(gap_collapse / 0.8, 0.0, 1.5)
                    * np.exp(-2.5 * (false_collapse + false_warming))
                )
                autumn_cooling_triggered_overturn_reward = float(np.mean(overturn_success))

    winter_mask = ((doy <= 75) | (doy >= 335)) & (air_temp <= 6.0)
    winter_inverse_penalty = 0.0
    winter_bottom_4c_error = 0.0
    if np.any(winter_mask):
        bottom_temp = temp_grid[-1, winter_mask]
        surface_temp = surface_pred[winter_mask]
        inverse_gap = bottom_temp - surface_temp
        winter_bottom_4c_error = float(np.mean(np.abs(bottom_temp - 4.0)))
        winter_inverse_penalty = float(
            np.mean(np.clip(1.5 - inverse_gap, 0.0, None) / 1.5 + np.abs(bottom_temp - 4.0) / 4.0)
        )

    deep_smoothness_penalty = 0.0
    if depths.size >= 3:
        deep_mask = depths >= min(10.0, max_depth * 0.55)
        deep_indices = np.where(deep_mask)[0]
        if deep_indices.size >= 3:
            deep_profiles = temp_grid[deep_indices, :]
            second_derivative = np.diff(deep_profiles, n=2, axis=0)
            deep_smoothness_penalty = float(np.mean(np.abs(second_derivative)))

    proxy_rmse = (
        surface_rmse
        + 12.0 * instability_penalty
        + 4.8 * deep_warm_penalty
        + 2.8 * deep_warm_peak
        + 1.5 * warm_surface_bias
        + 2.5 * summer_strat_penalty
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
    proxy_mae = (
        surface_mae
        + 8.0 * instability_penalty
        + 4.0 * deep_warm_penalty
        + 2.2 * deep_warm_peak
        + 1.2 * warm_surface_bias
        + 2.0 * summer_strat_penalty
        + 2.8 * summer_thermocline_thickness_penalty
        + 0.45 * max(summer_9m_temp - 14.0, 0.0)
        + 0.32 * max(summer_bottom_temp - 7.5, 0.0)
        + 1.8 * autumn_overturn_penalty
        + 1.4 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 1.6 * max(0.50 - autumn_gap_collapse, 0.0)
        + 3.2 * autumn_false_overturn_penalty
        + 1.5 * winter_inverse_penalty
        + 0.6 * winter_bottom_4c_error
        + 1.2 * deep_smoothness_penalty
        + 0.6 * lst_spike_indicator
        - 1.2 * summer_surface_warming_reward
        - 0.9 * summer_midlayer_temp_reward
        - 1.2 * autumn_cooling_triggered_overturn_reward
    )
    proxy_bias = (
        surface_bias
        + 3.2 * deep_warm_penalty
        + 1.2 * warm_surface_bias
        + 2.2 * summer_thermocline_thickness_penalty
        + 0.45 * max(summer_9m_temp - 14.0, 0.0)
        + 0.32 * max(summer_bottom_temp - 7.5, 0.0)
        + 0.8 * autumn_overturn_penalty
        + 0.6 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 0.6 * max(0.50 - autumn_gap_collapse, 0.0)
        + 2.8 * autumn_false_overturn_penalty
        + 0.6 * winter_inverse_penalty
        + 0.5 * winter_bottom_4c_error
        - 0.8 * summer_surface_warming_reward
        - 0.6 * summer_midlayer_temp_reward
        - 0.8 * autumn_cooling_triggered_overturn_reward
    )
    return {
        'rmse': float(proxy_rmse),
        'mae': float(proxy_mae),
        'bias': float(proxy_bias),
        'surface_rmse': float(surface_rmse),
        'warm_surface_bias': float(warm_surface_bias),
        'instability_penalty': float(instability_penalty),
        'deep_warm_penalty': float(deep_warm_penalty),
        'deep_warm_peak': float(deep_warm_peak),
        'summer_stratification_penalty': float(summer_strat_penalty),
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


def evaluate_profile_grid(df, metadata, temp_grid, depths, max_depth, profile_obs_data):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if observations.empty:
        return None

    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    date_to_index = {
        pd.Timestamp(date_value).normalize(): idx
        for idx, date_value in enumerate(pd.to_datetime(df['Date']))
    }

    errors = []
    obs_depth_values = []
    for date_value, obs_day in observations.groupby(observations['Date'].dt.normalize()):
        day_idx = date_to_index.get(pd.Timestamp(date_value).normalize())
        if day_idx is None:
            continue
        pred_profile = temp_grid[:, day_idx]
        pred_interp = np.interp(
            obs_day['Depth_m'].to_numpy(dtype=np.float64),
            depths,
            pred_profile,
        )
        obs_depths = obs_day['Depth_m'].to_numpy(dtype=np.float64)
        errors.extend((pred_interp - obs_day['Temperature_C'].to_numpy(dtype=np.float64)).tolist())
        obs_depth_values.extend(obs_depths.tolist())

    if not errors:
        return None

    return summarize_profile_error_metrics(obs_depth_values, errors, max_depth)


def merge_profile_selection_metrics(validation_metrics, selection_metrics):
    merged = {} if validation_metrics is None else dict(validation_metrics)
    if selection_metrics is None:
        return merged

    for key, value in dict(selection_metrics).items():
        merged[f'profile_{key}'] = value
    return merged


def evaluate_profile_at_date(current_date, current_profile, depths, profile_obs_data):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(current_date).normalize(),
        time_scale_seconds=SECONDS_PER_DAY,
        max_depth=float(np.max(depths)) if len(depths) else 0.0,
    )
    if observations.empty:
        return None

    obs_day = observations[observations['Date'].dt.normalize() == pd.Timestamp(current_date).normalize()]
    if obs_day.empty:
        return None

    current_profile = np.asarray(current_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    pred_interp = np.interp(
        obs_day['Depth_m'].to_numpy(dtype=np.float64),
        depths,
        current_profile,
    )
    errors = pred_interp - obs_day['Temperature_C'].to_numpy(dtype=np.float64)
    return summarize_profile_error_metrics(obs_day['Depth_m'].to_numpy(dtype=np.float64), errors, float(np.max(depths)) if len(depths) else 0.0)


def smooth_time_gate(doy, start_doy, end_doy, width_days=4.0):
    doy = np.asarray(doy, dtype=np.float64)
    width_days = float(max(width_days, 1.0e-6))
    rise = 1.0 / (1.0 + np.exp(-(doy - float(start_doy)) / width_days))
    fall = 1.0 / (1.0 + np.exp(-(float(end_doy) - doy) / width_days))
    return np.clip(rise * fall, 0.0, 1.0)


def evaluate_surface_band_validation_at_date(
    current_date,
    current_profile,
    depths,
    profile_obs_data,
    previous_profile_3d=None,
    shallow_max_depth=3.0,
):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(current_date).normalize(),
        time_scale_seconds=SECONDS_PER_DAY,
        max_depth=float(np.max(depths)) if len(depths) else 0.0,
    )
    if observations.empty:
        return None

    current_date = pd.Timestamp(current_date).normalize()
    obs_day = observations[observations['Date'].dt.normalize() == current_date]
    if obs_day.empty:
        return None

    shallow_obs = obs_day[pd.to_numeric(obs_day['Depth_m'], errors='coerce') <= float(shallow_max_depth)].copy()
    if shallow_obs.empty:
        return {
            'may_surface_warm_penalty': 0.0,
            'may_surface_rate_penalty': 0.0,
            'july_surface_cool_penalty': 0.0,
            'july_surface_warm_reward': 0.0,
            'surface_band_background_rmse': 0.0,
        }

    current_profile = np.asarray(current_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    shallow_depths = shallow_obs['Depth_m'].to_numpy(dtype=np.float64)
    pred_shallow = np.interp(shallow_depths, depths, current_profile)
    val_shallow = shallow_obs['Temperature_C'].to_numpy(dtype=np.float64)
    shallow_error = pred_shallow - val_shallow

    doy = float(current_date.dayofyear)
    may_gate = float(smooth_time_gate(doy, 110.0, 150.0, width_days=4.0))
    july_gate = float(smooth_time_gate(doy, 180.0, 215.0, width_days=4.0))
    background_gate = float(max(0.0, 1.0 - min(may_gate + july_gate, 1.0)))

    may_surface_warm_penalty = may_gate * float(np.mean(np.clip(shallow_error, 0.0, None)))
    july_surface_cool_penalty = july_gate * float(np.mean(np.clip(-shallow_error, 0.0, None)))
    july_surface_warm_reward = july_gate * float(np.mean(np.exp(-(shallow_error ** 2) / (2.0 ** 2))))
    surface_band_background_rmse = background_gate * float(np.sqrt(np.mean(shallow_error ** 2)))

    may_surface_rate_penalty = 0.0
    lag_date = current_date - pd.Timedelta(days=3)
    if previous_profile_3d is not None:
        obs_lag = observations[observations['Date'].dt.normalize() == lag_date]
        shallow_lag_obs = obs_lag[pd.to_numeric(obs_lag['Depth_m'], errors='coerce') <= float(shallow_max_depth)].copy()
        if not shallow_lag_obs.empty:
            previous_profile_3d = np.asarray(previous_profile_3d, dtype=np.float64)
            pred_shallow_lag = np.interp(
                shallow_lag_obs['Depth_m'].to_numpy(dtype=np.float64),
                depths,
                previous_profile_3d,
            )
            val_shallow_lag = shallow_lag_obs['Temperature_C'].to_numpy(dtype=np.float64)
            rate_pred = (float(np.mean(pred_shallow)) - float(np.mean(pred_shallow_lag))) / 3.0
            rate_val = (float(np.mean(val_shallow)) - float(np.mean(val_shallow_lag))) / 3.0
            may_surface_rate_penalty = may_gate * float(max(rate_pred - rate_val, 0.0))

    return {
        'may_surface_warm_penalty': float(may_surface_warm_penalty),
        'may_surface_rate_penalty': float(may_surface_rate_penalty),
        'july_surface_cool_penalty': float(july_surface_cool_penalty),
        'july_surface_warm_reward': float(july_surface_warm_reward),
        'surface_band_background_rmse': float(surface_band_background_rmse),
    }
