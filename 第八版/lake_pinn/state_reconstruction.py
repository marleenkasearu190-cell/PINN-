"""State reconstruction utilities shared by multi-lake training and export."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .conditional_priors import (
    infer_bottom_temp_prior_c,
    infer_ice_risk_prior,
    infer_thermal_regime,
)
from .constants import (
    PINN_FETCH_REFERENCE_M,
    PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
    PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
    PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
    PINN_MAX_LONGWAVE_REFERENCE_W_M2,
    PINN_MAX_SECCHI_REFERENCE_M,
    PINN_MAX_SHORTWAVE_REFERENCE_W_M2,
    PINN_MAX_TEMPERATURE_REFERENCE_C,
    PINN_MAX_WIND_REFERENCE_M_PER_S,
    PINN_INFLOW_REFERENCE_M3_S,
    PINN_SHORTWAVE_SUM_30D_REFERENCE,
    PINN_SHORTWAVE_SUM_7D_REFERENCE,
    PINN_WATER_LEVEL_ANOMALY_REFERENCE_M,
)
from .data_io import normalize_task_mode
from .physics import water_density_torch

def _safe_series_value(row, column, default=0.0):
    value = row[column] if column in row.index else default
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    return value if np.isfinite(value) else float(default)


def _diagnostic_scalar(diagnostics, key, default=0.0):
    value = diagnostics.get(key)
    if value is None:
        return float(default)
    return float(value.detach().cpu().reshape(-1)[0])


def _lst_quality_value(row):
    for column in ('LST_quality_factor', 'LST_observation_weight', 'LST_qc_good_fraction', 'LST_valid_pixel_fraction'):
        if column in row.index:
            value = _safe_series_value(row, column, np.nan)
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
    return 0.5


def _forcing_feature_array(row, allow_lst_features=True):
    """Normalize one daily forcing row for the parameter network."""
    pressure_norm = _safe_series_value(row, 'surface_pressure_Pa', 101325.0) / 101325.0 - 1.0
    lst_feature = _safe_series_value(row, 'LST_surface_C') if allow_lst_features else 0.0
    lst_mean_feature = _safe_series_value(row, 'lst_mean_7d') if allow_lst_features else 0.0
    if allow_lst_features:
        lst_quality = _lst_quality_value(row)
        lst_is_filled = float(np.clip(_safe_series_value(row, 'LST_is_filled', 1.0), 0.0, 1.0))
        lst_observed_flag = 1.0 - lst_is_filled
    else:
        lst_quality = 0.0
        lst_is_filled = 1.0
        lst_observed_flag = 0.0
    values = [
        _safe_series_value(row, 'doy_sin'),
        _safe_series_value(row, 'doy_cos'),
        _safe_series_value(row, 'T_air_C') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _safe_series_value(row, 'wind_speed_m_per_s', 1.0) / PINN_MAX_WIND_REFERENCE_M_PER_S,
        _safe_series_value(row, 'Solar_W_m2') / PINN_MAX_SHORTWAVE_REFERENCE_W_M2,
        lst_feature / PINN_MAX_TEMPERATURE_REFERENCE_C,
        lst_quality,
        lst_is_filled,
        lst_observed_flag,
        _safe_series_value(row, 'Longwave_W_m2') / PINN_MAX_LONGWAVE_REFERENCE_W_M2,
        _safe_series_value(row, 'latent_heat_upward_W_m2') / PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
        _safe_series_value(row, 'sensible_heat_upward_W_m2') / PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
        _safe_series_value(row, 'relative_humidity', 0.75),
        pressure_norm,
        _safe_series_value(row, 'Secchi_m') / PINN_MAX_SECCHI_REFERENCE_M,
        _safe_series_value(row, 'light_extinction_kd') / PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
        _safe_series_value(row, 'effective_fetch') / PINN_FETCH_REFERENCE_M,
        _safe_series_value(row, 'air_temp_mean_7d') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _safe_series_value(row, 'air_temp_mean_30d') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _safe_series_value(row, 'shortwave_sum_7d') / PINN_SHORTWAVE_SUM_7D_REFERENCE,
        _safe_series_value(row, 'shortwave_sum_30d') / PINN_SHORTWAVE_SUM_30D_REFERENCE,
        _safe_series_value(row, 'wind_mean_7d') / PINN_MAX_WIND_REFERENCE_M_PER_S,
        lst_mean_feature / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _safe_series_value(row, 'heating_degree_days_30d') / PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
        _safe_series_value(row, 'ice_fraction', 0.0),
        _safe_series_value(row, 'water_level_anomaly') / PINN_WATER_LEVEL_ANOMALY_REFERENCE_M,
        _safe_series_value(row, 'net_inflow') / PINN_INFLOW_REFERENCE_M3_S,
    ]
    return np.asarray(values, dtype=np.float32)


def apply_lst_surface_assimilation(
    profile,
    lst_surface,
    lst_quality,
    depths,
    strength=0.15,
    decay_depth_m=2.0,
    max_increment_c=1.5,
    ice_mask=None,
    ice_fraction=None,
):
    """Lightly nudge the upper water column toward satellite surface temperature."""
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return profile
    lst_surface = lst_surface.to(device=profile.device, dtype=profile.dtype).reshape(-1, 1)
    lst_quality = torch.clamp(
        lst_quality.to(device=profile.device, dtype=profile.dtype).reshape(-1, 1),
        0.0,
        1.0,
    )
    if lst_surface.shape[0] == 1 and profile.shape[0] > 1:
        lst_surface = lst_surface.expand(profile.shape[0], 1)
        lst_quality = lst_quality.expand(profile.shape[0], 1)
    if ice_mask is not None:
        ice_mask = torch.clamp(
            ice_mask.to(device=profile.device, dtype=profile.dtype).reshape(-1, 1),
            0.0,
            1.0,
        )
        if ice_mask.shape[0] == 1 and profile.shape[0] > 1:
            ice_mask = ice_mask.expand(profile.shape[0], 1)
        open_water_weight = 1.0 - ice_mask
    elif ice_fraction is not None:
        ice_fraction = torch.clamp(
            ice_fraction.to(device=profile.device, dtype=profile.dtype).reshape(-1, 1),
            0.0,
            1.0,
        )
        if ice_fraction.shape[0] == 1 and profile.shape[0] > 1:
            ice_fraction = ice_fraction.expand(profile.shape[0], 1)
        open_water_weight = 1.0 - ice_fraction
    else:
        open_water_weight = 1.0
    depths = depths.to(device=profile.device, dtype=profile.dtype).reshape(1, -1)
    gate = torch.exp(-depths / max(float(decay_depth_m), 1.0e-3))
    increment = open_water_weight * strength * lst_quality * gate * (lst_surface - profile[:, :1])
    increment = torch.clamp(increment, min=-float(max_increment_c), max=float(max_increment_c))
    return torch.clamp(profile + increment, 0.0, 40.0)


def _date_to_index_map(df):
    return {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }


def _resolve_rollout_start_idx(df, rollout_start_date=None, default_idx=0):
    dates = [pd.Timestamp(date).normalize() for date in pd.to_datetime(df['Date'])]
    if not dates:
        return 0
    if rollout_start_date is None:
        return int(np.clip(default_idx, 0, len(dates) - 1))
    requested = pd.Timestamp(rollout_start_date).normalize()
    for idx, date_value in enumerate(dates):
        if date_value >= requested:
            return idx
    return len(dates) - 1


def _recent_surface_water_temperature(df, start_idx, window_days=7):
    start_idx = int(np.clip(start_idx, 0, len(df) - 1))
    window_days = max(1, int(window_days))
    window = df.iloc[max(0, start_idx - window_days + 1): start_idx + 1]
    values = pd.to_numeric(window.get('LSWT_open_water_C'), errors='coerce') if 'LSWT_open_water_C' in window else pd.Series(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 0:
        return float(np.nanmedian(values))
    if 'ice_mask' in df.columns and float(pd.to_numeric(df['ice_mask'], errors='coerce').fillna(0.0).iloc[start_idx]) >= 0.5:
        return 0.2
    fallback = pd.to_numeric(window.get('LST_surface_C'), errors='coerce') if 'LST_surface_C' in window else pd.Series(dtype=float)
    fallback = fallback[np.isfinite(fallback)]
    if len(fallback) > 0:
        return float(np.nanmedian(fallback))
    return float(_safe_series_value(df.iloc[start_idx], 'T_air_C', 4.0))


def build_lst_profile_prior(df, depths, metadata, start_idx, window_days=7):
    """Build a cold-start profile from LST plus conditional lake priors."""
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    max_depth = float(max(float(np.nanmax(depths)) if depths.size else 1.0, 1.0))
    row_idx = int(np.clip(start_idx, 0, len(df) - 1))
    row = df.iloc[row_idx]
    doy = float(_safe_series_value(row, 'full_doy', pd.Timestamp(row['Date']).dayofyear))
    month = int(pd.Timestamp(row['Date']).month)
    mean_depth = float(metadata.get('mean_depth_m', metadata.get('mean_depth', 0.45 * max_depth)) or (0.45 * max_depth))

    thermal_regime = infer_thermal_regime(metadata, df)
    bottom_prior_c = float(infer_bottom_temp_prior_c(metadata, df, max_depth=max_depth))
    ice_risk = infer_ice_risk_prior(metadata, df)
    ice_risk_value = 0.0
    if not ice_risk.empty:
        ice_risk_value = float(pd.to_numeric(ice_risk, errors='coerce').fillna(0.0).iloc[row_idx])

    t_surf = float(np.clip(_recent_surface_water_temperature(df, start_idx, window_days=window_days), 0.0, 38.0))
    surface_minus_bottom = t_surf - bottom_prior_c
    cold_state = thermal_regime == 'cold_ice_prone' and ice_risk_value >= 0.5
    stratified_state = (
        max_depth >= 6.0
        and ice_risk_value < 0.5
        and surface_minus_bottom >= 2.0
    )

    if cold_state:
        t_surf = float(np.clip(t_surf, 0.0, min(4.0, bottom_prior_c + 0.5)))
        t_deep = float(np.clip(max(bottom_prior_c, t_surf + 0.5), 0.0, max(4.5, bottom_prior_c + 1.0)))
        thermocline_depth = 0.85 * max_depth
        thickness = max(3.0, 0.35 * max_depth)
    elif stratified_state:
        t_deep = float(np.clip(bottom_prior_c, 0.0, max(0.0, t_surf - 0.3)))
        thermocline_depth = float(np.clip(0.25 * max_depth + 0.15 * mean_depth, 2.0, 0.65 * max_depth))
        thickness = max(0.8, 0.08 * max_depth)
    else:
        t_deep = float(np.clip(0.65 * t_surf + 0.35 * bottom_prior_c, 0.0, 38.0))
        if t_deep > t_surf and ice_risk_value < 0.5:
            t_deep = float(max(0.0, t_surf - 0.2))
        thermocline_depth = 0.70 * max_depth
        thickness = max(2.0, 0.25 * max_depth)

    if max_depth <= 2.0:
        profile = np.full_like(depths, np.clip(0.5 * (t_surf + t_deep), 0.0, 38.0), dtype=np.float32)
    else:
        gate = 1.0 / (1.0 + np.exp((depths - thermocline_depth) / max(thickness, 1.0e-3)))
        profile = t_deep + (t_surf - t_deep) * gate
    profile = np.clip(profile.astype(np.float32), 0.0, 38.0)
    metadata_info = {
        'prior_surface_temp_c': float(t_surf),
        'prior_deep_temp_c': float(t_deep),
        'prior_bottom_temp_c': float(bottom_prior_c),
        'prior_ice_risk': float(ice_risk_value),
        'prior_thermal_regime': str(thermal_regime),
        'prior_thermocline_depth_m': float(thermocline_depth),
        'prior_transition_thickness_m': float(thickness),
        'prior_doy': float(doy),
        'prior_month': int(month),
    }
    return profile, metadata_info


@torch.no_grad()
def initialize_rollout_state(
    *,
    model,
    df,
    depths,
    all_lookup,
    forcing_rows,
    static_features,
    metadata,
    device,
    init_mode='profile',
    rollout_start_date=None,
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    task_mode='analysis',
    area_profile=None,
    hard_density_stability=False,
):
    """Return a reconstruction/rollout start state and optional spinup trajectory.

    profile mode preserves the old behavior when a start profile exists.
    If no profile exists, profile mode falls back to prior_spinup instead of a
    uniform-LST water column.
    """
    init_mode = str(init_mode or 'profile').strip().lower()
    if init_mode not in {'profile', 'lst_profile_prior', 'prior_spinup', 'uniform_lst_debug'}:
        raise ValueError("init_mode must be profile, lst_profile_prior, prior_spinup, or uniform_lst_debug.")
    date_to_index = _date_to_index_map(df)
    profile_dates = sorted(date for date in all_lookup if date in date_to_index)
    default_profile_idx = date_to_index[profile_dates[0]] if profile_dates else 0
    rollout_start_idx = _resolve_rollout_start_idx(df, rollout_start_date, default_idx=default_profile_idx)
    requested_mode = init_mode
    profile_date = None

    if init_mode == 'profile' and profile_dates:
        eligible = [date for date in profile_dates if date_to_index[date] <= rollout_start_idx]
        if eligible:
            profile_date = max(eligible)
            start_idx = date_to_index[profile_date]
            initial_profile = np.asarray(all_lookup[profile_date], dtype=np.float32)
            current = torch.tensor(initial_profile, dtype=torch.float32, device=device).unsqueeze(0)
            freezing_storage = torch.zeros_like(current)
            profiles_by_index = {start_idx: initial_profile.copy()}
            for day_idx in range(start_idx, rollout_start_idx):
                next_row = forcing_rows[day_idx + 1] if day_idx + 1 < len(forcing_rows) else None
                current, freezing_storage = model.step(
                    current,
                    forcing_rows[day_idx],
                    static_features,
                    next_forcing_row=next_row,
                    task_mode=task_mode,
                    depths=torch.as_tensor(depths, dtype=torch.float32, device=device),
                    area_profile=area_profile,
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage,
                    return_freezing_storage=True,
                )
                profiles_by_index[day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
            return {
                'current': current,
                'freezing_storage_j_m2': freezing_storage,
                'initial_profile': initial_profile,
                'start_idx': start_idx,
                'rollout_start_idx': rollout_start_idx,
                'profiles_by_index': profiles_by_index,
                'diagnostics': [],
                'init_mode': 'profile',
                'requested_init_mode': requested_mode,
                'profile_date': profile_date.date().isoformat(),
                'spinup_days_used': int(max(0, rollout_start_idx - start_idx)),
                'prior_info': {},
            }
        init_mode = 'prior_spinup'

    if init_mode == 'prior_spinup':
        if rollout_start_date is None and rollout_start_idx < int(spinup_days):
            rollout_start_idx = min(max(0, int(spinup_days)), len(df) - 1)
        start_idx = max(0, rollout_start_idx - max(0, int(spinup_days)))
    else:
        start_idx = rollout_start_idx

    if init_mode == 'uniform_lst_debug':
        surface = float(_safe_series_value(df.iloc[start_idx], 'LST_surface_C', _safe_series_value(df.iloc[start_idx], 'T_air_C', 4.0)))
        initial_profile = np.full_like(np.asarray(depths, dtype=np.float32), np.clip(surface, 0.0, 38.0), dtype=np.float32)
        prior_info = {'prior_surface_temp_c': float(initial_profile[0]), 'prior_deep_temp_c': float(initial_profile[-1])}
    else:
        initial_profile, prior_info = build_lst_profile_prior(df, depths, metadata, start_idx)

    current = torch.tensor(initial_profile, dtype=torch.float32, device=device).unsqueeze(0)
    freezing_storage = torch.zeros_like(current)
    profiles_by_index = {start_idx: initial_profile.copy()}
    diagnostics = []
    if init_mode == 'prior_spinup':
        depth_tensor = torch.as_tensor(depths, dtype=torch.float32, device=device)
        for day_idx in range(start_idx, rollout_start_idx):
            next_row = forcing_rows[day_idx + 1] if day_idx + 1 < len(forcing_rows) else None
            current, freezing_storage, step_diagnostics = model.step(
                current,
                forcing_rows[day_idx],
                static_features,
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=depth_tensor,
                area_profile=area_profile,
                return_diagnostics=True,
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
            surface_before = float(current[:, 0].detach().cpu().reshape(-1)[0])
            if float(spinup_lst_assimilation_strength) > 0.0 and next_row is not None:
                assimilation_lst = next_row.get('lswt_open_water', next_row.get('lst_surface'))
                if assimilation_lst is not None and bool(torch.any(torch.isfinite(assimilation_lst)).detach().cpu()):
                    current = apply_lst_surface_assimilation(
                        current,
                        assimilation_lst,
                        next_row['lst_quality'],
                        depth_tensor,
                        strength=spinup_lst_assimilation_strength,
                        decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                        max_increment_c=spinup_lst_assimilation_max_increment_c,
                        ice_mask=next_row.get('ice_mask'),
                    )
            profiles_by_index[day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
            diagnostics.append({
                'Date': pd.Timestamp(df['Date'].iloc[day_idx + 1]).date().isoformat(),
                'spinup_phase': True,
                'init_mode': 'prior_spinup',
                'requested_init_mode': requested_mode,
                'spinup_days_used': int(max(0, rollout_start_idx - start_idx)),
                'surface_temp_before_lst_assim_c': surface_before,
                'surface_temp_c': float(current[:, 0].detach().cpu().reshape(-1)[0]),
                'lst_assimilation_increment_c': float(current[:, 0].detach().cpu().reshape(-1)[0] - surface_before),
                'turbulent_flux_mode': getattr(model, 'turbulent_flux_mode', 'provided'),
                'turbulent_flux_blend_alpha': float(getattr(model, 'turbulent_flux_blend_alpha', 1.0)),
                'freezing_energy_mode': getattr(model, 'freezing_energy_mode', 'clamp'),
                'surface_flux_wm2': _diagnostic_scalar(step_diagnostics, 'surface_flux_wm2'),
                'open_water_surface_flux_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_surface_flux_wm2'),
                'open_water_net_radiation_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_net_radiation_wm2'),
                'open_water_sensible_heat_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_sensible_heat_wm2'),
                'open_water_latent_heat_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_latent_heat_wm2'),
                'open_water_sensible_heat_bulk_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_sensible_heat_bulk_wm2'),
                'open_water_latent_heat_bulk_wm2': _diagnostic_scalar(step_diagnostics, 'open_water_latent_heat_bulk_wm2'),
                'heat_input_wm2': _diagnostic_scalar(step_diagnostics, 'heat_input_wm2'),
                'heat_tendency_wm2': _diagnostic_scalar(step_diagnostics, 'heat_tendency_wm2'),
                'sensible_heat_tendency_wm2': _diagnostic_scalar(step_diagnostics, 'sensible_heat_tendency_wm2'),
                'effective_heat_tendency_wm2': _diagnostic_scalar(step_diagnostics, 'effective_heat_tendency_wm2'),
                'freezing_storage_j_m2': _diagnostic_scalar(step_diagnostics, 'freezing_storage_j_m2'),
                'freezing_storage_change_wm2': _diagnostic_scalar(step_diagnostics, 'freezing_storage_change_wm2'),
                'energy_residual_wm2': float((step_diagnostics['heat_tendency_wm2'] - step_diagnostics['heat_input_wm2']).detach().cpu().reshape(-1)[0]),
                'temperature_floor_heat_injection_wm2': _diagnostic_scalar(step_diagnostics, 'temperature_floor_heat_injection_wm2'),
                'temperature_floor_heat_injection_j_m2': _diagnostic_scalar(step_diagnostics, 'temperature_floor_heat_injection_j_m2'),
                'temperature_ceiling_heat_removal_wm2': _diagnostic_scalar(step_diagnostics, 'temperature_ceiling_heat_removal_wm2'),
                'kz_mean': _diagnostic_scalar(step_diagnostics, 'kz_mean'),
                'kz_surface_mean': _diagnostic_scalar(step_diagnostics, 'kz_surface_mean'),
                'kz_mid_mean': _diagnostic_scalar(step_diagnostics, 'kz_mid_mean'),
                'kz_deep_mean': _diagnostic_scalar(step_diagnostics, 'kz_deep_mean'),
                'shortwave_to_water_wm2': _diagnostic_scalar(step_diagnostics, 'shortwave_to_water_wm2'),
                'ice_shortwave_transmission': _diagnostic_scalar(step_diagnostics, 'ice_shortwave_transmission', 1.0),
                'surface_flux_bias_wm2': _diagnostic_scalar(step_diagnostics, 'surface_flux_bias_wm2'),
                'residual_abs_mean_c': _diagnostic_scalar(step_diagnostics, 'residual_abs_mean_c'),
                'density_adjustment_applied': _diagnostic_scalar(step_diagnostics, 'density_adjustment_applied'),
                'density_adjustment_max_delta_c': _diagnostic_scalar(step_diagnostics, 'density_adjustment_max_delta_c'),
                'density_adjustment_heat_delta_j_m2': _diagnostic_scalar(step_diagnostics, 'density_adjustment_heat_delta_j_m2'),
            })

    return {
        'current': current,
        'freezing_storage_j_m2': freezing_storage,
        'initial_profile': initial_profile,
        'start_idx': start_idx,
        'rollout_start_idx': rollout_start_idx,
        'profiles_by_index': profiles_by_index,
        'diagnostics': diagnostics,
        'init_mode': init_mode,
        'requested_init_mode': requested_mode,
        'profile_date': None,
        'spinup_days_used': int(max(0, rollout_start_idx - start_idx)),
        'prior_info': prior_info,
    }


def _forcing_tensor_rows(df, device='cpu', history_window_days=30, task_mode='analysis'):
    normalize_task_mode(task_mode)
    allow_lst_features = True
    feature_matrix = np.stack(
        [_forcing_feature_array(row, allow_lst_features=allow_lst_features) for _, row in df.iterrows()],
        axis=0,
    ).astype(np.float32)
    history_window_days = max(1, int(history_window_days))
    rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        start_idx = max(0, row_idx - history_window_days + 1)
        history = feature_matrix[start_idx: row_idx + 1]
        if history.shape[0] < history_window_days:
            pad = np.repeat(history[:1], history_window_days - history.shape[0], axis=0)
            history = np.concatenate([pad, history], axis=0)
        features = torch.tensor(feature_matrix[row_idx], dtype=torch.float32, device=device)
        lst_is_filled = float(np.clip(_safe_series_value(row, 'LST_is_filled', 1.0), 0.0, 1.0))
        lst_observed_flag = 1.0 - lst_is_filled
        rows.append(
            {
                'features': features,
                'history_features': torch.tensor(history, dtype=torch.float32, device=device),
                'air_temp': torch.tensor([_safe_series_value(row, 'T_air_C')], dtype=torch.float32, device=device),
                'wind_speed': torch.tensor([_safe_series_value(row, 'wind_speed_m_per_s', 1.0)], dtype=torch.float32, device=device),
                'relative_humidity': torch.tensor([_safe_series_value(row, 'relative_humidity', 0.75)], dtype=torch.float32, device=device),
                'surface_pressure': torch.tensor([_safe_series_value(row, 'surface_pressure_Pa', 101325.0)], dtype=torch.float32, device=device),
                'shortwave': torch.tensor([_safe_series_value(row, 'Solar_W_m2')], dtype=torch.float32, device=device),
                'longwave': torch.tensor([_safe_series_value(row, 'Longwave_W_m2')], dtype=torch.float32, device=device),
                'latent_heat': torch.tensor([_safe_series_value(row, 'latent_heat_upward_W_m2')], dtype=torch.float32, device=device),
                'sensible_heat': torch.tensor([_safe_series_value(row, 'sensible_heat_upward_W_m2')], dtype=torch.float32, device=device),
                'lst_surface': torch.tensor([_safe_series_value(row, 'LST_surface_C')], dtype=torch.float32, device=device),
                'lswt_open_water': torch.tensor([_safe_series_value(row, 'LSWT_open_water_C', np.nan)], dtype=torch.float32, device=device),
                'ist_snow_ice': torch.tensor([_safe_series_value(row, 'IST_snow_ice_C', np.nan)], dtype=torch.float32, device=device),
                'lst_quality': torch.tensor([_lst_quality_value(row)], dtype=torch.float32, device=device),
                'lst_is_filled': torch.tensor([lst_is_filled], dtype=torch.float32, device=device),
                'lst_observed_flag': torch.tensor([lst_observed_flag], dtype=torch.float32, device=device),
                'ice_mask': torch.tensor([_safe_series_value(row, 'ice_mask', 0.0)], dtype=torch.float32, device=device),
                'ice_fraction': torch.tensor([_safe_series_value(row, 'ice_fraction', 0.0)], dtype=torch.float32, device=device),
                'snow_depth': torch.tensor([_safe_series_value(row, 'snow_depth_m', 0.0)], dtype=torch.float32, device=device),
                'ice_thickness': torch.tensor([_safe_series_value(row, 'ice_thickness_m', 0.0)], dtype=torch.float32, device=device),
            }
        )
    return rows


def _profile_lookup(profile_obs, depths, return_masks=False):
    lookup = {}
    masks = {}
    if profile_obs is None or profile_obs.empty:
        return (lookup, masks) if return_masks else lookup
    depths = np.asarray(depths, dtype=np.float64)
    for date_value, group in profile_obs.groupby(profile_obs['Date'].dt.normalize()):
        group = group.sort_values('Depth_m')
        z = group['Depth_m'].to_numpy(dtype=np.float64)
        temp = group['Temperature_C'].to_numpy(dtype=np.float64)
        finite = np.isfinite(z) & np.isfinite(temp)
        z = z[finite]
        temp = temp[finite]
        if z.size == 0:
            continue
        if z.size == 1:
            profile = np.full_like(depths, float(temp[0]), dtype=np.float32)
            mask = np.isclose(depths, float(z[0]), atol=max(float(np.nanmedian(np.diff(depths))) if depths.size > 1 else 1.0, 0.25))
        else:
            profile = np.interp(depths, z, temp).astype(np.float32)
            mask = (depths >= float(np.min(z))) & (depths <= float(np.max(z)))
        if not np.any(mask):
            nearest_idx = int(np.argmin(np.abs(depths - float(z[0]))))
            mask = np.zeros_like(depths, dtype=bool)
            mask[nearest_idx] = True
        lookup[pd.Timestamp(date_value).normalize()] = profile
        masks[pd.Timestamp(date_value).normalize()] = mask.astype(bool)
    return (lookup, masks) if return_masks else lookup


def _build_rollout_pairs(df, profile_lookup, max_rollout_days=45):
    date_to_index = {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }
    dates = sorted(date for date in profile_lookup if date in date_to_index)
    pairs = []
    for start, end in zip(dates[:-1], dates[1:]):
        start_idx = date_to_index[start]
        end_idx = date_to_index[end]
        gap = int(end_idx - start_idx)
        if 1 <= gap <= int(max_rollout_days):
            pairs.append((start, end, start_idx, end_idx))
    return pairs


def _huber_profile_loss(pred, target, delta=2.0, mask=None):
    target = target.to(device=pred.device, dtype=pred.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid = valid & torch.as_tensor(mask, dtype=torch.bool, device=pred.device).reshape(1, -1)
    if not torch.any(valid):
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    loss = F.huber_loss(pred, target, delta=float(delta), reduction='none')
    return loss[valid].mean()


def _profile_physics_loss(profile):
    rho = water_density_torch(profile)
    density_inversion = torch.relu(rho[:, :-1] - rho[:, 1:] - 0.02).pow(2).mean()
    range_loss = torch.relu(-profile).pow(2).mean() + torch.relu(profile - 38.0).pow(2).mean()
    gradient_loss = torch.relu(torch.abs(profile[:, 1:] - profile[:, :-1]) - 8.0).pow(2).mean()
    return range_loss + 4.0 * density_inversion + 0.05 * gradient_loss
