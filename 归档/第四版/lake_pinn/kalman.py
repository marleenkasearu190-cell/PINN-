# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .data_io import build_observation_dataframe, depth_dependent_obs_std
from .physics import apply_autumn_cooling_adjustment, project_temperature_profile_to_stable_density

def build_depth_covariance(depths, variance, correlation_length):
    """Build a depth-correlated covariance matrix for the Kalman filter."""
    depths = np.asarray(depths, dtype=np.float64)
    variance = float(max(variance, 0.0))
    if variance == 0.0:
        return np.zeros((len(depths), len(depths)), dtype=np.float64)

    if correlation_length <= 0.0:
        return np.eye(len(depths), dtype=np.float64) * variance

    delta = depths[:, None] - depths[None, :]
    kernel = np.exp(-0.5 * (delta / correlation_length) ** 2)
    return variance * kernel


def effective_forecast_blend(
    base_blend,
    days_since_last_obs,
    spinup_days=0,
    spinup_max_blend=0.9,
):
    base_blend = float(np.clip(base_blend, 0.0, 1.0))
    spinup_max_blend = float(np.clip(spinup_max_blend, base_blend, 1.0))
    spinup_days = int(max(spinup_days, 0))

    if spinup_days <= 0 or days_since_last_obs is None:
        return base_blend

    if days_since_last_obs <= 0:
        return base_blend

    if spinup_days == 1:
        return spinup_max_blend

    progress = min(max((days_since_last_obs - 1) / max(spinup_days - 1, 1), 0.0), 1.0)
    return float(spinup_max_blend - progress * (spinup_max_blend - base_blend))


def build_kalman_observation_frame(
    df,
    metadata,
    max_depth,
    profile_obs_data,
    depths,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
):
    """Map observations to the nearest state-grid depth index for assimilation."""
    observations, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_obs_data,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    observations = observations.copy()
    observations['Date'] = pd.to_datetime(observations['Date']).dt.normalize()
    depth_grid = np.asarray(depths, dtype=np.float64)
    observations['grid_idx'] = observations['Depth_m'].apply(
        lambda depth_value: int(np.argmin(np.abs(depth_grid - float(depth_value))))
    )
    observations['grid_depth_m'] = observations['grid_idx'].apply(lambda idx: float(depth_grid[idx]))
    observations = (
        observations
        .groupby(['Date', 'source', 'grid_idx', 'grid_depth_m'], as_index=False)
        .agg({'Temperature_C': 'mean'})
        .sort_values(['Date', 'grid_idx'])
        .reset_index(drop=True)
    )
    return observations


def run_profile_kalman_filter(
    df,
    temp_grid,
    depths,
    metadata,
    max_depth,
    profile_obs_data=None,
    prior_std=2.0,
    process_std=0.3,
    obs_std_surface=0.5,
    obs_std_bottom=0.5,
    obs_std_profile=0.75,
    correlation_length=2.0,
    forecast_blend=0.2,
    forecast_spinup_days=0,
    forecast_spinup_max_blend=0.9,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
    daily_process_scale=None,
    daily_obs_scale=None,
    daily_correlation_length=None,
    daily_forecast_blend=None,
    autumn_asymmetric_cooling=False,
    autumn_doy_threshold=270.0,
    autumn_surface_cooling_threshold=1.0,
    autumn_air_temp_threshold=12.0,
    autumn_cooling_strength=0.35,
    autumn_cooling_penetration_scale=5.0,
):
    """
    Assimilate profile observations on top of PINN temperature profiles.

    State vector:
        x_t = [T(z_1, t), ..., T(z_n, t)]^T

    Forecast:
        the PINN profile at day t acts as the model forecast, optionally blended
        with the previous filtered state to preserve temporal continuity.
    """
    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    n_depth, n_days = temp_grid.shape
    filtered_grid = np.zeros_like(temp_grid)
    identity = np.eye(n_depth, dtype=np.float64)
    prior_cov = build_depth_covariance(depths, prior_std ** 2, correlation_length)
    covariance = prior_cov.copy()

    observations = build_kalman_observation_frame(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_obs_data,
        depths=depths,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    obs_by_date = {
        date_value: group.reset_index(drop=True)
        for date_value, group in observations.groupby('Date')
    }

    diagnostics = []
    normalized_dates = pd.to_datetime(df['Date']).dt.normalize()
    last_observation_day_idx = None

    for day_idx, date_value in enumerate(normalized_dates):
        process_scale_today = 1.0 if daily_process_scale is None else float(daily_process_scale[min(day_idx, len(daily_process_scale) - 1)])
        obs_scale_today = 1.0 if daily_obs_scale is None else float(daily_obs_scale[min(day_idx, len(daily_obs_scale) - 1)])
        correlation_length_today = float(correlation_length if daily_correlation_length is None else daily_correlation_length[min(day_idx, len(daily_correlation_length) - 1)])
        forecast_blend_today = float(forecast_blend if daily_forecast_blend is None else daily_forecast_blend[min(day_idx, len(daily_forecast_blend) - 1)])
        process_cov = build_depth_covariance(depths, (process_std * process_scale_today) ** 2, correlation_length_today)
        pinn_forecast = temp_grid[:, day_idx].copy()
        if day_idx == 0:
            state_pred = pinn_forecast
        else:
            if last_observation_day_idx is None:
                days_since_last_obs = None
            else:
                days_since_last_obs = day_idx - last_observation_day_idx
            blend_today = effective_forecast_blend(
                base_blend=forecast_blend_today,
                days_since_last_obs=days_since_last_obs,
                spinup_days=forecast_spinup_days,
                spinup_max_blend=forecast_spinup_max_blend,
            )
            state_pred = (
                blend_today * filtered_grid[:, day_idx - 1] +
                (1.0 - blend_today) * pinn_forecast
            )

        cov_pred = covariance + process_cov
        day_obs = obs_by_date.get(date_value)

        if day_obs is None or day_obs.empty:
            state_upd = state_pred
            covariance = cov_pred
            obs_count = 0
            innovation_rms = np.nan
            autumn_cooling_applied = 0.0
        else:
            y = day_obs['Temperature_C'].to_numpy(dtype=np.float64)
            grid_idx = day_obs['grid_idx'].to_numpy(dtype=np.int64)
            H = np.zeros((len(day_obs), n_depth), dtype=np.float64)
            H[np.arange(len(day_obs)), grid_idx] = 1.0

            obs_std = []
            for source, obs_depth in zip(day_obs['source'], day_obs['grid_depth_m']):
                obs_std.append(
                    depth_dependent_obs_std(
                        source=source,
                        depth_m=obs_depth,
                        max_depth=max_depth,
                        base_surface=obs_std_surface * obs_scale_today,
                        base_bottom=obs_std_bottom * obs_scale_today,
                        base_profile=obs_std_profile * obs_scale_today,
                    )
                )
            R = np.diag(np.square(np.asarray(obs_std, dtype=np.float64)))

            innovation = y - H @ state_pred
            innovation_rms = float(np.sqrt(np.mean(innovation ** 2)))
            S = H @ cov_pred @ H.T + R
            kalman_gain = cov_pred @ H.T @ np.linalg.pinv(S)
            state_upd = state_pred + kalman_gain @ innovation

            # Joseph form keeps covariance positive semidefinite more reliably.
            kh = kalman_gain @ H
            covariance = (identity - kh) @ cov_pred @ (identity - kh).T + kalman_gain @ R @ kalman_gain.T
            obs_count = int(len(day_obs))
            last_observation_day_idx = day_idx

            surface_obs_temp = None
            surface_rows = day_obs[day_obs['source'] == 'surface']
            if not surface_rows.empty:
                surface_obs_temp = float(surface_rows['Temperature_C'].iloc[0])
            state_upd, autumn_cooling_applied = apply_autumn_cooling_adjustment(
                state_pred=state_pred,
                state_upd=state_upd,
                depths=depths,
                day_doy=float(df['full_doy'].iloc[day_idx]),
                mixed_layer_depth=float(df['MixedLayerDepth_m'].iloc[day_idx]) if 'MixedLayerDepth_m' in df.columns else 2.0,
                surface_obs_temp=surface_obs_temp,
                air_temp=float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else None,
                enabled=autumn_asymmetric_cooling,
                doy_threshold=autumn_doy_threshold,
                cooling_threshold=autumn_surface_cooling_threshold,
                air_temp_threshold=autumn_air_temp_threshold,
                propagation_strength=autumn_cooling_strength,
                penetration_scale=autumn_cooling_penetration_scale,
            )

        if df['T_air_C'].iloc[day_idx] < 0.0:
            state_upd[0] = 0.0

        state_upd = np.clip(state_upd, -1.0, 35.0)
        state_upd, projection_adjustments = project_temperature_profile_to_stable_density(state_upd)
        filtered_grid[:, day_idx] = state_upd
        covariance = 0.5 * (covariance + covariance.T)

        diagnostics.append(
            {
                'Date': pd.Timestamp(date_value),
                'obs_count': obs_count,
                'innovation_rms': innovation_rms,
                'surface_temperature_C': float(state_upd[0]),
                'bottom_temperature_C': float(state_upd[-1]),
                'projection_adjustments': int(projection_adjustments),
                'autumn_cooling_applied': float(autumn_cooling_applied),
                'days_since_last_obs': np.nan if last_observation_day_idx is None else float(day_idx - last_observation_day_idx),
                'kalman_correlation_length': float(correlation_length_today),
                'kalman_forecast_blend': float(forecast_blend_today),
            }
        )

    diagnostics_df = pd.DataFrame(diagnostics)
    return filtered_grid.astype(np.float32), diagnostics_df
