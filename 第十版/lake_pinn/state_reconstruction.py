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
    PINN_LSWT_MEMORY_DAYS_REFERENCE,
    PINN_SHORTWAVE_SUM_30D_REFERENCE,
    PINN_SHORTWAVE_SUM_7D_REFERENCE,
    PINN_NET_RADIATION_SUM_30D_REFERENCE,
    PINN_NET_RADIATION_SUM_7D_REFERENCE,
    PINN_WATER_LEVEL_ANOMALY_REFERENCE_M,
    PINN_WIND_ENERGY_REFERENCE,
)
from .data_io import normalize_task_mode
from .physics import water_density_torch
from .state_model import ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM
from .vertical_solver import layer_thicknesses

MAINLINE_LSWT_OBSERVER_MODE_CHOICES = (
    'off',
    'conservative_surface',
    'conservative_mld_shallow',
)
MAINLINE_LSWT_OBSERVER_MODES = set(MAINLINE_LSWT_OBSERVER_MODE_CHOICES)
LSWT_OBSERVER_MODES = MAINLINE_LSWT_OBSERVER_MODES
MAINLINE_ZERO_PROFILE_INITIALIZER_MODE_CHOICES = (
    'low_dof',
    'lswt_climatology_low_dof',
    'eof_pca_low_dof',
    'eof_pca_init_net',
)
MAINLINE_ZERO_PROFILE_INITIALIZER_MODES = set(MAINLINE_ZERO_PROFILE_INITIALIZER_MODE_CHOICES)
ZERO_PROFILE_INITIALIZER_MODES = MAINLINE_ZERO_PROFILE_INITIALIZER_MODES


def normalize_lswt_observer_mode(mode):
    mode = str(mode or 'off').strip().lower().replace('-', '_')
    aliases = {
        'none': 'off',
        'disabled': 'off',
        'shallow_surface': 'conservative_surface',
        'conservative_shallow': 'conservative_surface',
        'shallow': 'conservative_surface',
        'r20_conservative_surface': 'conservative_surface',
        'conservative_mld': 'conservative_mld_shallow',
        'mld_shallow': 'conservative_mld_shallow',
        'shallow_mld': 'conservative_mld_shallow',
        'conservative_mixed_layer': 'conservative_mld_shallow',
        'mixed_layer_shallow': 'conservative_mld_shallow',
    }
    mode = aliases.get(mode, mode)
    if mode not in LSWT_OBSERVER_MODES:
        raise ValueError(
            'lswt observer mode must be one of the tenth-version mainline modes: '
            + ', '.join(sorted(LSWT_OBSERVER_MODES))
        )
    return mode


def normalize_mainline_lswt_observer_mode(mode):
    mode = normalize_lswt_observer_mode(mode)
    return mode


def normalize_zero_profile_initializer_mode(mode):
    mode = str(mode or 'low_dof').strip().lower().replace('-', '_')
    aliases = {
        'zero_profile_low_dof': 'low_dof',
        'lowdof': 'low_dof',
        'lswt_climatology': 'lswt_climatology_low_dof',
        'lswt_climatology_lowdof': 'lswt_climatology_low_dof',
        'seasonal_lswt_low_dof': 'lswt_climatology_low_dof',
        'raw_lswt_low_dof': 'lswt_climatology_low_dof',
        'eof': 'eof_pca_low_dof',
        'pca': 'eof_pca_low_dof',
        'eof_low_dof': 'eof_pca_low_dof',
        'pca_low_dof': 'eof_pca_low_dof',
        'eof_pca': 'eof_pca_low_dof',
        'thermal_state_low_dof': 'eof_pca_low_dof',
        'low_dim_thermal_state': 'eof_pca_low_dof',
        'init_net': 'eof_pca_init_net',
        'eof_init_net': 'eof_pca_init_net',
        'pca_init_net': 'eof_pca_init_net',
        'eof_pca_net': 'eof_pca_init_net',
        'constrained_init_net': 'eof_pca_init_net',
        'network_initial_column': 'eof_pca_init_net',
    }
    mode = aliases.get(mode, mode)
    if mode not in ZERO_PROFILE_INITIALIZER_MODES:
        raise ValueError(
            'zero-profile initializer mode must be one of the tenth-version mainline modes: '
            + ', '.join(sorted(ZERO_PROFILE_INITIALIZER_MODES))
        )
    return mode


def normalize_mainline_zero_profile_initializer_mode(mode):
    mode = normalize_zero_profile_initializer_mode(mode)
    return mode


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


def _lst_observed_flag_value(row, lst_is_filled):
    if 'LST_observed_flag' in row.index:
        observed_flag = _safe_series_value(row, 'LST_observed_flag', np.nan)
        if np.isfinite(observed_flag):
            return float(np.clip(observed_flag, 0.0, 1.0))
    return float(np.clip(1.0 - float(lst_is_filled), 0.0, 1.0))


def _forcing_feature_array(row, allow_lst_features=True):
    """Normalize one daily forcing row for the parameter network."""
    pressure_norm = _safe_series_value(row, 'surface_pressure_Pa', 101325.0) / 101325.0 - 1.0
    lst_feature = _safe_series_value(row, 'LST_surface_C') if allow_lst_features else 0.0
    lst_mean_feature = _safe_series_value(row, 'lst_mean_7d') if allow_lst_features else 0.0
    days_since_raw_lswt = (
        _safe_series_value(row, 'days_since_last_raw_LSWT', PINN_LSWT_MEMORY_DAYS_REFERENCE)
        if allow_lst_features else PINN_LSWT_MEMORY_DAYS_REFERENCE
    )
    raw_lswt_valid_count_30d = (
        _safe_series_value(row, 'raw_LSWT_valid_count_30d', 0.0)
        if allow_lst_features else 0.0
    )
    raw_lswt_trend_7d = (
        _safe_series_value(row, 'raw_LSWT_trend_7d', 0.0)
        if allow_lst_features else 0.0
    )
    if allow_lst_features:
        lst_quality = _lst_quality_value(row)
        lst_is_filled = float(np.clip(_safe_series_value(row, 'LST_is_filled', 1.0), 0.0, 1.0))
        lst_observed_flag = _lst_observed_flag_value(row, lst_is_filled)
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
        _safe_series_value(row, 'net_radiation_7d') / PINN_NET_RADIATION_SUM_7D_REFERENCE,
        _safe_series_value(row, 'net_radiation_30d') / PINN_NET_RADIATION_SUM_30D_REFERENCE,
        _safe_series_value(row, 'wind_mean_7d') / PINN_MAX_WIND_REFERENCE_M_PER_S,
        _safe_series_value(row, 'wind_energy_7d') / PINN_WIND_ENERGY_REFERENCE,
        _safe_series_value(row, 'wind_energy_30d') / PINN_WIND_ENERGY_REFERENCE,
        _safe_series_value(row, 'wind_mixing_potential_7d') / PINN_WIND_ENERGY_REFERENCE,
        _safe_series_value(row, 'wind_mixing_potential_30d') / PINN_WIND_ENERGY_REFERENCE,
        lst_mean_feature / PINN_MAX_TEMPERATURE_REFERENCE_C,
        days_since_raw_lswt / PINN_LSWT_MEMORY_DAYS_REFERENCE,
        raw_lswt_valid_count_30d / 30.0,
        raw_lswt_trend_7d / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _safe_series_value(row, 'heating_degree_days_30d') / PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
        _safe_series_value(row, 'cooling_degree_days_30d') / PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
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


def _recent_numeric_median(df, column, start_idx, window_days, default=np.nan):
    if column not in df.columns or len(df) == 0:
        return float(default)
    start_idx = int(np.clip(start_idx, 0, len(df) - 1))
    window = df.iloc[max(0, start_idx - max(1, int(window_days)) + 1): start_idx + 1]
    values = pd.to_numeric(window[column], errors='coerce')
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(default)
    return float(np.nanmedian(values))


def _recent_raw_open_water_lswt_summary(df, start_idx, window_days, min_quality=0.05):
    if 'LSWT_open_water_C' not in df.columns or len(df) == 0:
        return {
            'median': np.nan,
            'count': 0,
            'observed_fraction': 0.0,
            'quality_mean': np.nan,
        }
    start_idx = int(np.clip(start_idx, 0, len(df) - 1))
    window = df.iloc[max(0, start_idx - max(1, int(window_days)) + 1): start_idx + 1]
    values = pd.to_numeric(window['LSWT_open_water_C'], errors='coerce')
    valid = np.isfinite(values)

    if 'LST_is_filled' in window:
        filled = pd.to_numeric(window['LST_is_filled'], errors='coerce').fillna(1.0)
        valid &= filled <= 0.5
    if 'LST_observed_flag' in window:
        observed = pd.to_numeric(window['LST_observed_flag'], errors='coerce').fillna(0.0)
        valid &= observed > 0.5
    if 'ice_mask' in window:
        ice = pd.to_numeric(window['ice_mask'], errors='coerce').fillna(0.0)
        valid &= ice < 0.5
    if 'ice_fraction' in window:
        ice_fraction = pd.to_numeric(window['ice_fraction'], errors='coerce').fillna(0.0)
        valid &= ice_fraction < 0.50

    quality_values = []
    quality_valid = pd.Series(np.ones(len(window), dtype=bool), index=window.index)
    for column in ('LST_quality_factor', 'LST_observation_weight', 'LST_qc_good_fraction', 'LST_valid_pixel_fraction'):
        if column in window:
            quality = pd.to_numeric(window[column], errors='coerce')
            quality_values.append(quality)
            quality_valid &= quality.fillna(0.0) >= float(min_quality)
    if quality_values:
        valid &= quality_valid
        quality_stack = pd.concat(quality_values, axis=1)
        quality_arr = quality_stack.where(valid).to_numpy(dtype=float)
        quality_mean = float(np.nanmean(quality_arr)) if np.isfinite(quality_arr).any() else np.nan
    else:
        quality_mean = np.nan

    selected = values[valid]
    count = int(selected.count())
    return {
        'median': float(np.nanmedian(selected)) if count else np.nan,
        'count': count,
        'observed_fraction': float(count / max(1, len(window))),
        'quality_mean': quality_mean,
    }


def _area_weighted_delta_c(delta, depths, area_profile=None):
    if delta.ndim == 1:
        delta = delta.unsqueeze(0)
    depths = depths.to(device=delta.device, dtype=delta.dtype).reshape(-1)
    if area_profile is None:
        area = torch.ones_like(depths)
    else:
        area = torch.as_tensor(area_profile, device=delta.device, dtype=delta.dtype).reshape(-1)
        if area.numel() != depths.numel():
            area = torch.ones_like(depths)
    dz = layer_thicknesses(depths)
    weights = torch.clamp(area * dz, min=1.0e-8).reshape(1, -1)
    return (delta * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0e-8)


def _density_inversion_score(profile):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    rho = water_density_torch(profile)
    return torch.relu(rho[:, :-1] - rho[:, 1:] - 0.02).mean(dim=1)


def _metadata_reservoir_indicator(metadata):
    if not metadata:
        return 0.0
    indicator = metadata.get('reservoir_indicator')
    try:
        indicator = float(indicator)
    except (TypeError, ValueError):
        indicator = np.nan
    if np.isfinite(indicator):
        return float(np.clip(indicator, 0.0, 1.0))
    for key in ('lake_type', 'waterbody_type', 'lake_group', 'lake_id'):
        text = str(metadata.get(key, '') or '').strip().lower()
        if 'reservoir' in text or 'impound' in text or 'regulated' in text:
            return 1.0
    return 0.0


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


def build_zero_profile_low_dof_prior(df, depths, metadata, start_idx, window_days=90):
    """Structured metadata/forcing/LST initializer for no-profile inference.

    This is deliberately low-degree-of-freedom: it adjusts the existing
    conditional prior with recent raw LSWT/forcing summaries, then keeps the
    profile smooth and bounded. It does not use profile observations.
    """
    profile, info = build_lst_profile_prior(df, depths, metadata, start_idx, window_days=7)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if depths.size == 0:
        return profile, info

    row_idx = int(np.clip(start_idx, 0, len(df) - 1))
    max_depth = float(max(float(np.nanmax(depths)), 1.0))
    recent_lswt_30 = _recent_numeric_median(df, 'LSWT_open_water_C', row_idx, 30, np.nan)
    recent_lswt_90 = _recent_numeric_median(df, 'LSWT_open_water_C', row_idx, 90, np.nan)
    recent_air_30 = _recent_numeric_median(df, 'air_temp_mean_30d', row_idx, 30, np.nan)
    if not np.isfinite(recent_air_30):
        recent_air_30 = _recent_numeric_median(df, 'T_air_C', row_idx, 30, np.nan)
    recent_sw_30 = _recent_numeric_median(df, 'shortwave_sum_30d', row_idx, 30, np.nan)
    if not np.isfinite(recent_sw_30):
        recent_sw_30 = _recent_numeric_median(df, 'Solar_W_m2', row_idx, 30, np.nan)

    surface_anchor = float(info.get('prior_surface_temp_c', profile[0]))
    if np.isfinite(recent_lswt_30):
        surface_anchor = float(0.65 * surface_anchor + 0.35 * np.clip(recent_lswt_30, 0.0, 38.0))
    elif np.isfinite(recent_lswt_90):
        surface_anchor = float(0.80 * surface_anchor + 0.20 * np.clip(recent_lswt_90, 0.0, 38.0))

    heating_signal = 0.0
    if np.isfinite(recent_air_30):
        heating_signal += 0.12 * float(np.clip((recent_air_30 - 4.0) / 16.0, -1.0, 1.0))
    if np.isfinite(recent_sw_30):
        scale = PINN_SHORTWAVE_SUM_30D_REFERENCE if recent_sw_30 > 1000.0 else PINN_MAX_SHORTWAVE_REFERENCE_W_M2
        heating_signal += 0.08 * float(np.clip((recent_sw_30 / max(scale, 1.0)) - 0.5, -1.0, 1.0))

    mld_depth = float(np.clip(1.5 + 0.18 * max_depth, 1.0, min(12.0, max_depth)))
    top_gate = np.exp(-depths / max(mld_depth, 1.0e-3)).astype(np.float32)
    deep_gate = (1.0 - np.exp(-depths / max(0.35 * max_depth, 1.0))).astype(np.float32)
    profile = profile.astype(np.float32).copy()
    profile += top_gate * float(surface_anchor - profile[0]) * 0.45
    profile += deep_gate * float(heating_signal)
    if profile.size >= 3:
        smooth = profile.copy()
        smooth[1:-1] = 0.25 * profile[:-2] + 0.50 * profile[1:-1] + 0.25 * profile[2:]
        profile = 0.75 * profile + 0.25 * smooth
    profile = np.clip(profile, 0.0, 38.0).astype(np.float32)
    info.update({
        'zero_profile_initializer': 'low_dof',
        'zero_profile_recent_lswt_30d_c': float(recent_lswt_30) if np.isfinite(recent_lswt_30) else np.nan,
        'zero_profile_recent_lswt_90d_c': float(recent_lswt_90) if np.isfinite(recent_lswt_90) else np.nan,
        'zero_profile_recent_air_30d_c': float(recent_air_30) if np.isfinite(recent_air_30) else np.nan,
        'zero_profile_recent_shortwave_30d': float(recent_sw_30) if np.isfinite(recent_sw_30) else np.nan,
        'zero_profile_mld_depth_proxy_m': float(mld_depth),
        'zero_profile_surface_anchor_c': float(surface_anchor),
        'zero_profile_heating_signal_c': float(heating_signal),
    })
    return profile, info


def build_zero_profile_lswt_climatology_low_dof_prior(
    df,
    depths,
    metadata,
    start_idx,
    window_days=180,
    min_quality=0.05,
):
    """Low-dimensional zero-profile prior anchored only by raw open-water LSWT.

    Filled/reconstructed LST remains available to downstream feature encoders but
    is not used here as a strong surface-temperature target. If no raw/open-water
    LSWT passes the gate, the initializer falls back to ERA5/metadata priors.
    """
    prior_df = df.copy()
    if 'LST_surface_C' in prior_df.columns:
        filled_mask = pd.Series(True, index=prior_df.index)
        if 'LST_is_filled' in prior_df.columns:
            filled_mask = pd.to_numeric(prior_df['LST_is_filled'], errors='coerce').fillna(1.0) > 0.5
        if 'LST_observed_flag' in prior_df.columns:
            observed = pd.to_numeric(prior_df['LST_observed_flag'], errors='coerce').fillna(0.0) > 0.5
            filled_mask |= ~observed
        prior_df.loc[filled_mask, 'LST_surface_C'] = np.nan

    profile, info = build_lst_profile_prior(prior_df, depths, metadata, start_idx, window_days=21)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if depths.size == 0:
        return profile, info

    row_idx = int(np.clip(start_idx, 0, len(df) - 1))
    max_depth = float(max(float(np.nanmax(depths)), 1.0))
    mean_depth = float(metadata.get('mean_depth_m', metadata.get('mean_depth', 0.45 * max_depth)) or (0.45 * max_depth))
    bottom_prior_c = float(info.get('prior_bottom_temp_c', infer_bottom_temp_prior_c(metadata, df, max_depth=max_depth)))
    ice_risk_value = float(info.get('prior_ice_risk', 0.0) or 0.0)

    raw_14 = _recent_raw_open_water_lswt_summary(df, row_idx, 14, min_quality=min_quality)
    raw_30 = _recent_raw_open_water_lswt_summary(df, row_idx, 30, min_quality=min_quality)
    raw_90 = _recent_raw_open_water_lswt_summary(df, row_idx, 90, min_quality=min_quality)
    raw_full = _recent_raw_open_water_lswt_summary(df, row_idx, window_days, min_quality=min_quality)
    recent_air_30 = _recent_numeric_median(df, 'air_temp_mean_30d', row_idx, 30, np.nan)
    if not np.isfinite(recent_air_30):
        recent_air_30 = _recent_numeric_median(df, 'T_air_C', row_idx, 30, np.nan)
    recent_wind_30 = _recent_numeric_median(df, 'wind_speed_m_per_s', row_idx, 30, np.nan)
    recent_sw_30 = _recent_numeric_median(df, 'shortwave_sum_30d', row_idx, 30, np.nan)
    if not np.isfinite(recent_sw_30):
        recent_sw_30 = _recent_numeric_median(df, 'Solar_W_m2', row_idx, 30, np.nan)

    base_surface = float(info.get('prior_surface_temp_c', profile[0]))
    surface_anchor = base_surface
    raw_anchor = np.nan
    raw_count = 0
    if np.isfinite(raw_14['median']):
        raw_anchor = raw_14['median']
        raw_count = int(raw_14['count'])
        surface_anchor = 0.40 * base_surface + 0.60 * float(np.clip(raw_anchor, 0.0, 38.0))
    elif np.isfinite(raw_30['median']):
        raw_anchor = raw_30['median']
        raw_count = int(raw_30['count'])
        surface_anchor = 0.45 * base_surface + 0.55 * float(np.clip(raw_anchor, 0.0, 38.0))
    elif np.isfinite(raw_90['median']):
        raw_anchor = raw_90['median']
        raw_count = int(raw_90['count'])
        surface_anchor = 0.65 * base_surface + 0.35 * float(np.clip(raw_anchor, 0.0, 38.0))
    elif np.isfinite(raw_full['median']):
        raw_anchor = raw_full['median']
        raw_count = int(raw_full['count'])
        surface_anchor = 0.80 * base_surface + 0.20 * float(np.clip(raw_anchor, 0.0, 38.0))
    surface_anchor = float(np.clip(surface_anchor, 0.0, 38.0))

    wind_gate = 0.0
    if np.isfinite(recent_wind_30):
        wind_gate = float(np.clip(recent_wind_30 / 6.0, 0.0, 1.0))
    shortwave_gate = 0.0
    if np.isfinite(recent_sw_30):
        scale = PINN_SHORTWAVE_SUM_30D_REFERENCE if recent_sw_30 > 1000.0 else PINN_MAX_SHORTWAVE_REFERENCE_W_M2
        shortwave_gate = float(np.clip(recent_sw_30 / max(scale, 1.0), 0.0, 1.5))
    warm_signal = float(np.clip((surface_anchor - bottom_prior_c) / 9.0, 0.0, 1.0))
    air_surface_gap = 0.0
    if np.isfinite(recent_air_30):
        air_surface_gap = float(np.clip((recent_air_30 - surface_anchor) / 6.0, -1.0, 1.0))

    mld_upper = max(1.0, min(16.0, 0.60 * max_depth))
    mld_depth = 1.0 + 0.08 * max_depth + 0.18 * mean_depth
    mld_depth += 2.5 * wind_gate
    mld_depth += 2.0 * (1.0 - warm_signal)
    mld_depth = float(np.clip(mld_depth, 1.0, mld_upper))
    if ice_risk_value >= 0.5:
        mld_depth = float(np.clip(0.45 * max_depth, 1.0, mld_upper))

    deep_prior = float(info.get('prior_deep_temp_c', bottom_prior_c))
    deep_anchor = 0.70 * deep_prior + 0.30 * bottom_prior_c
    if warm_signal > 0.25 and ice_risk_value < 0.5:
        deep_anchor = min(deep_anchor, surface_anchor - 0.25)
    if ice_risk_value >= 0.5:
        deep_anchor = max(deep_anchor, min(4.5, surface_anchor + 0.4))
    deep_anchor = float(np.clip(deep_anchor, 0.0, 38.0))

    thermocline_depth = mld_depth + 0.12 * max_depth + 0.08 * mean_depth + 1.5 * shortwave_gate
    thermocline_depth = float(np.clip(thermocline_depth, mld_depth + 0.5, max(mld_depth + 0.5, 0.80 * max_depth)))
    transition_thickness = float(np.clip(0.10 * max_depth + 0.40 * wind_gate * max_depth, 0.75, 5.0))
    gate = 1.0 / (1.0 + np.exp((depths - thermocline_depth) / max(transition_thickness, 1.0e-3)))
    candidate = deep_anchor + (surface_anchor - deep_anchor) * gate
    mixed_gate = np.exp(-depths / max(mld_depth, 1.0e-3))
    candidate = candidate + 0.25 * mixed_gate * (surface_anchor - candidate[0])
    if profile.size == candidate.size:
        candidate = 0.70 * candidate + 0.30 * profile
    if candidate.size >= 3:
        smooth = candidate.copy()
        smooth[1:-1] = 0.25 * candidate[:-2] + 0.50 * candidate[1:-1] + 0.25 * candidate[2:]
        candidate = 0.85 * candidate + 0.15 * smooth
    candidate = np.clip(candidate.astype(np.float32), 0.0, 38.0)

    info.update({
        'zero_profile_initializer': 'lswt_climatology_low_dof',
        'zero_profile_raw_lswt_anchor_c': float(raw_anchor) if np.isfinite(raw_anchor) else np.nan,
        'zero_profile_raw_lswt_count': int(raw_count),
        'zero_profile_raw_lswt_14d_c': float(raw_14['median']) if np.isfinite(raw_14['median']) else np.nan,
        'zero_profile_raw_lswt_30d_c': float(raw_30['median']) if np.isfinite(raw_30['median']) else np.nan,
        'zero_profile_raw_lswt_90d_c': float(raw_90['median']) if np.isfinite(raw_90['median']) else np.nan,
        'zero_profile_raw_lswt_window_count': int(raw_full['count']),
        'zero_profile_raw_lswt_observed_fraction': float(raw_full['observed_fraction']),
        'zero_profile_raw_lswt_quality_mean': float(raw_full['quality_mean']) if np.isfinite(raw_full['quality_mean']) else np.nan,
        'zero_profile_recent_air_30d_c': float(recent_air_30) if np.isfinite(recent_air_30) else np.nan,
        'zero_profile_recent_wind_30d_mps': float(recent_wind_30) if np.isfinite(recent_wind_30) else np.nan,
        'zero_profile_recent_shortwave_30d': float(recent_sw_30) if np.isfinite(recent_sw_30) else np.nan,
        'zero_profile_surface_anchor_c': float(surface_anchor),
        'zero_profile_deep_anchor_c': float(deep_anchor),
        'zero_profile_mld_depth_proxy_m': float(mld_depth),
        'zero_profile_thermocline_depth_proxy_m': float(thermocline_depth),
        'zero_profile_transition_thickness_m': float(transition_thickness),
        'zero_profile_warm_signal': float(warm_signal),
        'zero_profile_air_surface_gap_c_scaled': float(air_surface_gap),
        'zero_profile_filled_lst_strong_target_used': False,
    })
    return candidate, info


def _thermal_basis_depth_grid(grid_points=40):
    grid_points = max(4, int(grid_points))
    return np.linspace(0.0, 1.0, grid_points, dtype=np.float32)


def _normalized_depth_fraction(depths):
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if depths.size == 0:
        return depths
    max_depth = float(np.nanmax(depths[np.isfinite(depths)])) if np.isfinite(depths).any() else 1.0
    max_depth = max(max_depth, 1.0e-6)
    return np.clip(depths / max_depth, 0.0, 1.0).astype(np.float32)


def _interp_by_normalized_depth(values, source_depth_fraction, target_depth_fraction):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    source_depth_fraction = np.asarray(source_depth_fraction, dtype=np.float32).reshape(-1)
    target_depth_fraction = np.asarray(target_depth_fraction, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values) & np.isfinite(source_depth_fraction)
    if int(finite.sum()) < 2:
        return np.full_like(target_depth_fraction, np.nan, dtype=np.float32)
    order = np.argsort(source_depth_fraction[finite])
    x = source_depth_fraction[finite][order]
    y = values[finite][order]
    return np.interp(target_depth_fraction, x, y, left=y[0], right=y[-1]).astype(np.float32)


ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODES = (
    'off',
    'lake_season_depth_coverage',
)


def normalize_zero_profile_thermal_basis_balance_mode(mode):
    text = str(mode or 'lake_season_depth_coverage').strip().lower().replace('-', '_')
    aliases = {
        'none': 'off',
        'disabled': 'off',
        'balanced': 'lake_season_depth_coverage',
        'transfer': 'lake_season_depth_coverage',
        'transferable': 'lake_season_depth_coverage',
        'lake_season_depth': 'lake_season_depth_coverage',
    }
    text = aliases.get(text, text)
    if text not in ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODES:
        allowed = ', '.join(ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODES)
        raise ValueError(f'zero_profile_thermal_basis_balance_mode must be one of: {allowed}.')
    return text


def _thermal_basis_season_key(date_key):
    try:
        month = int(pd.Timestamp(date_key).month)
    except (TypeError, ValueError):
        return 'unknown'
    if month in (12, 1, 2):
        return 'winter'
    if month in (3, 4, 5):
        return 'spring'
    if month in (6, 7, 8):
        return 'summer'
    return 'fall'


def _thermal_basis_depth_coverage_key(depth_fraction, valid):
    depth_fraction = np.asarray(depth_fraction, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if depth_fraction.size != valid.size or not np.any(valid):
        return 'unknown'
    valid_depths = depth_fraction[valid & np.isfinite(depth_fraction)]
    if valid_depths.size == 0:
        return 'unknown'
    coverage_fraction = float(valid_depths.size / max(1, depth_fraction.size))
    depth_span = float(np.nanmax(valid_depths) - np.nanmin(valid_depths))
    shallowest = float(np.nanmin(valid_depths))
    deepest = float(np.nanmax(valid_depths))
    if depth_span >= 0.80 and coverage_fraction >= 0.65:
        return 'full_column'
    if deepest <= 0.35 or depth_span < 0.35:
        return 'surface_upper'
    if shallowest >= 0.35:
        return 'deep_partial'
    return 'partial_column'


def _count_records(records, key):
    counts = {}
    for record in records:
        value = str(record.get(key, 'unknown') or 'unknown')
        counts[value] = counts.get(value, 0) + 1
    return counts


def _thermal_basis_sample_weights(records, balance_mode):
    mode = normalize_zero_profile_thermal_basis_balance_mode(balance_mode)
    if not records:
        return np.asarray([], dtype=np.float32), {
            'basis_balance_mode': mode,
            'season_profile_counts': {},
            'depth_coverage_profile_counts': {},
        }
    lake_counts = _count_records(records, 'lake_id')
    season_counts = _count_records(records, 'season')
    coverage_counts = _count_records(records, 'depth_coverage')
    if mode == 'off':
        weights = np.ones((len(records),), dtype=np.float32)
    else:
        total = float(len(records))
        lake_group_count = max(1, len(lake_counts))
        season_group_count = max(1, len(season_counts))
        coverage_group_count = max(1, len(coverage_counts))
        raw = []
        for record in records:
            lake = str(record.get('lake_id', 'unknown') or 'unknown')
            season = str(record.get('season', 'unknown') or 'unknown')
            coverage = str(record.get('depth_coverage', 'unknown') or 'unknown')
            lake_factor = total / (lake_group_count * max(1, lake_counts.get(lake, 1)))
            season_factor = total / (season_group_count * max(1, season_counts.get(season, 1)))
            coverage_factor = total / (coverage_group_count * max(1, coverage_counts.get(coverage, 1)))
            raw.append(float((lake_factor * season_factor * coverage_factor) ** (1.0 / 3.0)))
        weights = np.asarray(raw, dtype=np.float32)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0).astype(np.float32)
        weights = weights / max(float(np.nanmean(weights)), 1.0e-8)
        weights = np.clip(weights, 0.05, 20.0)
        weights = weights / max(float(np.nanmean(weights)), 1.0e-8)
    return weights.astype(np.float32), {
        'basis_balance_mode': mode,
        'season_profile_counts': {key: int(value) for key, value in sorted(season_counts.items())},
        'depth_coverage_profile_counts': {key: int(value) for key, value in sorted(coverage_counts.items())},
        'profile_weight_min': float(np.nanmin(weights)) if weights.size else 0.0,
        'profile_weight_mean': float(np.nanmean(weights)) if weights.size else 0.0,
        'profile_weight_max': float(np.nanmax(weights)) if weights.size else 0.0,
    }


def _normalize_zero_profile_thermal_basis(thermal_basis):
    if not thermal_basis:
        return None
    try:
        depth_fraction = np.asarray(thermal_basis['depth_fraction'], dtype=np.float32).reshape(-1)
        mean_profile = np.asarray(thermal_basis['mean_profile_c'], dtype=np.float32).reshape(-1)
        components = np.asarray(thermal_basis['components'], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if depth_fraction.ndim != 1 or mean_profile.ndim != 1 or components.ndim != 2:
        return None
    if depth_fraction.size != mean_profile.size or components.shape[1] != depth_fraction.size:
        return None
    if components.shape[0] < 1 or depth_fraction.size < 2:
        return None
    coeff_std = np.asarray(
        thermal_basis.get('coeff_std', np.ones((components.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if coeff_std.size != components.shape[0]:
        coeff_std = np.ones((components.shape[0],), dtype=np.float32)
    coeff_std = np.where(np.isfinite(coeff_std) & (coeff_std > 1.0e-6), coeff_std, 1.0).astype(np.float32)
    return {
        **dict(thermal_basis),
        'depth_fraction': depth_fraction,
        'mean_profile_c': mean_profile,
        'components': components,
        'coeff_std': coeff_std,
    }


def fit_zero_profile_eof_pca_basis(
    profile_sources,
    *,
    n_components=4,
    grid_points=40,
    min_valid_fraction=0.55,
    balance_mode='lake_season_depth_coverage',
):
    """Fit a train-only vertical thermal EOF/PCA basis on normalized depth.

    `profile_sources` must be provided by the caller, typically train split
    profile lookups only.  The function does not inspect validation or heldout
    data by itself.
    """
    balance_mode = normalize_zero_profile_thermal_basis_balance_mode(balance_mode)
    basis_depth = _thermal_basis_depth_grid(grid_points)
    records = []
    source_lakes = set()
    for source_idx, source in enumerate(profile_sources or ()):
        depths = np.asarray(source.get('depths'), dtype=np.float32).reshape(-1)
        lookup = source.get('lookup') or {}
        masks = source.get('masks') or {}
        if depths.size < 2 or not lookup:
            continue
        depth_fraction = _normalized_depth_fraction(depths)
        min_valid = max(2, int(np.ceil(float(min_valid_fraction) * depths.size)))
        lake_id = source.get('lake_id')
        lake_key = str(lake_id) if lake_id is not None else f'source_{source_idx}'
        for date_key, profile in lookup.items():
            profile = np.asarray(profile, dtype=np.float32).reshape(-1)
            if profile.size != depths.size:
                continue
            mask = masks.get(date_key)
            if mask is None:
                valid = np.ones_like(profile, dtype=bool)
            else:
                valid = np.asarray(mask, dtype=bool).reshape(-1)
                if valid.size != profile.size:
                    valid = np.ones_like(profile, dtype=bool)
            valid = valid & np.isfinite(profile) & np.isfinite(depth_fraction)
            if int(valid.sum()) < min_valid:
                continue
            interpolated = _interp_by_normalized_depth(profile[valid], depth_fraction[valid], basis_depth)
            if np.isfinite(interpolated).all():
                records.append({
                    'profile': interpolated,
                    'lake_id': lake_key,
                    'season': _thermal_basis_season_key(date_key),
                    'depth_coverage': _thermal_basis_depth_coverage_key(depth_fraction, valid),
                })
                source_lakes.add(lake_key)
    if len(records) < 2:
        return None

    matrix = np.stack([record['profile'] for record in records], axis=0).astype(np.float32)
    sample_weights, balance_info = _thermal_basis_sample_weights(records, balance_mode)
    probability_weights = sample_weights.astype(np.float64)
    probability_weights = probability_weights / max(float(np.sum(probability_weights)), 1.0e-12)
    mean_profile = np.sum(matrix.astype(np.float64) * probability_weights.reshape(-1, 1), axis=0).astype(np.float32)
    centered = matrix - mean_profile.reshape(1, -1)
    weighted_centered = centered.astype(np.float64) * np.sqrt(probability_weights).reshape(-1, 1)
    try:
        _u, singular_values, vt = np.linalg.svd(weighted_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    component_count = int(min(max(1, int(n_components)), vt.shape[0], vt.shape[1]))
    components = vt[:component_count].astype(np.float32)
    for idx in range(component_count):
        pivot = int(np.argmax(np.abs(components[idx])))
        if components[idx, pivot] < 0.0:
            components[idx] *= -1.0
    singular_sq = np.square(singular_values.astype(np.float64))
    total_variance = float(np.sum(singular_sq))
    explained = (
        (singular_sq[:component_count] / total_variance).astype(np.float32)
        if total_variance > 0.0 else np.zeros((component_count,), dtype=np.float32)
    )
    coeffs = centered @ components.T
    coeff_mean = np.sum(coeffs.astype(np.float64) * probability_weights.reshape(-1, 1), axis=0)
    coeff_var = np.sum(
        np.square(coeffs.astype(np.float64) - coeff_mean.reshape(1, -1))
        * probability_weights.reshape(-1, 1),
        axis=0,
    )
    coeff_std = np.sqrt(np.maximum(coeff_var, 0.0)).astype(np.float32)
    coeff_std = np.where(np.isfinite(coeff_std) & (coeff_std > 1.0e-6), coeff_std, 1.0).astype(np.float32)
    return {
        'basis_type': 'train_only_normalized_depth_eof_pca',
        'depth_fraction': basis_depth.tolist(),
        'mean_profile_c': mean_profile.astype(np.float32).tolist(),
        'components': components.astype(np.float32).tolist(),
        'explained_variance_ratio': explained.astype(np.float32).tolist(),
        'coeff_std': coeff_std.tolist(),
        'profile_count': int(matrix.shape[0]),
        'source_lake_count': int(len(source_lakes)),
        'source_split': 'train',
        'grid_points': int(basis_depth.size),
        **balance_info,
    }


def project_profile_to_zero_profile_thermal_basis(
    profile,
    depths,
    thermal_basis,
    *,
    mask=None,
    coeff_clip_sigma=4.0,
):
    """Project a profile-like estimate onto the train-only thermal basis."""
    basis = _normalize_zero_profile_thermal_basis(thermal_basis)
    if basis is None:
        raise ValueError('thermal_basis is missing or invalid.')
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if profile.size != depths.size:
        raise ValueError('profile and depths must have the same length.')
    depth_fraction = _normalized_depth_fraction(depths)
    mean_on_target = _interp_by_normalized_depth(
        basis['mean_profile_c'],
        basis['depth_fraction'],
        depth_fraction,
    )
    components_on_target = np.stack(
        [
            _interp_by_normalized_depth(component, basis['depth_fraction'], depth_fraction)
            for component in basis['components']
        ],
        axis=0,
    ).astype(np.float32)
    valid = np.isfinite(profile) & np.isfinite(mean_on_target) & np.isfinite(depth_fraction)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.size == profile.size:
            valid &= mask_arr
    valid &= np.isfinite(components_on_target).all(axis=0)
    if int(valid.sum()) < max(2, min(profile.size, components_on_target.shape[0])):
        coeffs = np.zeros((components_on_target.shape[0],), dtype=np.float32)
    else:
        design = components_on_target[:, valid].T.astype(np.float64)
        target = (profile[valid] - mean_on_target[valid]).astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        coeffs = coeffs.astype(np.float32)
    coeff_std = basis['coeff_std'].astype(np.float32)
    clip = np.maximum(float(coeff_clip_sigma) * coeff_std, 1.0)
    coeffs = np.clip(coeffs, -clip, clip).astype(np.float32)
    reconstructed = mean_on_target + np.sum(coeffs.reshape(-1, 1) * components_on_target, axis=0)
    reconstructed = np.clip(reconstructed.astype(np.float32), 0.0, 38.0)
    return reconstructed, {
        'thermal_basis_component_count': int(components_on_target.shape[0]),
        'thermal_basis_profile_count': int(basis.get('profile_count', 0) or 0),
        'thermal_basis_source_lake_count': int(basis.get('source_lake_count', 0) or 0),
        'thermal_basis_explained_variance_sum': float(
            np.nansum(np.asarray(basis.get('explained_variance_ratio', []), dtype=np.float32))
        ),
        'thermal_basis_coefficients': [float(value) for value in coeffs],
        'thermal_basis_coeff_abs_mean': float(np.mean(np.abs(coeffs))) if coeffs.size else 0.0,
    }


def zero_profile_thermal_basis_tensors_for_depths(
    thermal_basis,
    depths,
    *,
    device=None,
    dtype=torch.float32,
):
    """Return train-only thermal basis tensors interpolated to a lake depth grid."""
    basis = _normalize_zero_profile_thermal_basis(thermal_basis)
    if basis is None:
        return None
    depths_arr = np.asarray(depths, dtype=np.float32).reshape(-1)
    depth_fraction = _normalized_depth_fraction(depths_arr)
    components_on_target = np.stack(
        [
            _interp_by_normalized_depth(component, basis['depth_fraction'], depth_fraction)
            for component in basis['components']
        ],
        axis=0,
    ).astype(np.float32)
    mean_profile_on_target = _interp_by_normalized_depth(
        basis['mean_profile_c'],
        basis['depth_fraction'],
        depth_fraction,
    ).astype(np.float32)
    if not np.isfinite(components_on_target).all():
        return None
    if not np.isfinite(mean_profile_on_target).all():
        return None
    coeff_std = np.asarray(basis['coeff_std'], dtype=np.float32).reshape(-1)
    coeff_std = np.where(np.isfinite(coeff_std) & (coeff_std > 1.0e-6), coeff_std, 1.0)
    return {
        'components_on_depth': torch.as_tensor(components_on_target, device=device, dtype=dtype),
        'mean_profile_on_depth': torch.as_tensor(mean_profile_on_target, device=device, dtype=dtype),
        'coeff_std': torch.as_tensor(coeff_std, device=device, dtype=dtype),
        'component_count': int(components_on_target.shape[0]),
        'profile_count': int(basis.get('profile_count', 0) or 0),
        'source_lake_count': int(basis.get('source_lake_count', 0) or 0),
        'explained_variance_sum': float(
            np.nansum(np.asarray(basis.get('explained_variance_ratio', []), dtype=np.float32))
        ),
    }


def build_zero_profile_eof_pca_low_dof_prior(
    df,
    depths,
    metadata,
    start_idx,
    *,
    thermal_basis=None,
    min_quality=0.05,
    basis_blend=0.75,
):
    """Low-dimensional train-basis initializer for zero-profile inference.

    The basis is fitted outside this function from train split profiles.  At
    inference time this function only projects an observable-input prior onto
    that basis; it does not read profile observations.
    """
    base_profile, info = build_zero_profile_lswt_climatology_low_dof_prior(
        df,
        depths,
        metadata,
        start_idx,
        min_quality=min_quality,
    )
    info.update({
        'zero_profile_initializer': 'eof_pca_low_dof',
        'zero_profile_base_initializer': 'lswt_climatology_low_dof',
        'zero_profile_profile_derived_input_used': False,
        'zero_profile_filled_lst_strong_target_used': False,
    })
    if _normalize_zero_profile_thermal_basis(thermal_basis) is None:
        info.update({
            'zero_profile_thermal_basis_applied': False,
            'zero_profile_thermal_basis_reason': 'missing_or_invalid_basis',
        })
        return base_profile, info

    projected, projection_info = project_profile_to_zero_profile_thermal_basis(
        base_profile,
        depths,
        thermal_basis,
    )
    basis_blend = float(np.clip(basis_blend, 0.0, 1.0))
    candidate = basis_blend * projected + (1.0 - basis_blend) * base_profile
    depths_arr = np.asarray(depths, dtype=np.float32).reshape(-1)
    if candidate.size >= 3 and depths_arr.size == candidate.size:
        mld_depth = float(info.get('zero_profile_mld_depth_proxy_m', 2.0) or 2.0)
        surface_gate = np.exp(-depths_arr / max(mld_depth, 1.0e-3)).astype(np.float32)
        candidate = candidate + 0.25 * surface_gate * float(base_profile[0] - candidate[0])
        smooth = candidate.copy()
        smooth[1:-1] = 0.25 * candidate[:-2] + 0.50 * candidate[1:-1] + 0.25 * candidate[2:]
        candidate = 0.90 * candidate + 0.10 * smooth
    candidate = np.clip(candidate.astype(np.float32), 0.0, 38.0)
    info.update({
        'zero_profile_thermal_basis_applied': True,
        'zero_profile_thermal_basis_blend': float(basis_blend),
        'zero_profile_thermal_basis_initial_delta_abs_mean_c': float(np.mean(np.abs(candidate - base_profile))),
        'zero_profile_thermal_basis_surface_delta_c': float(candidate[0] - base_profile[0]),
        'zero_profile_thermal_basis_deep_delta_c': float(candidate[-1] - base_profile[-1]),
        **projection_info,
    })
    return candidate, info


def _zero_profile_init_conditioning_from_inputs(df, depths, metadata, start_idx, *, area_profile=None):
    """Observable-only lake/forcing summary for the learned initial-column head."""
    metadata = metadata or {}

    def metadata_float(*keys, default=0.0):
        for key in keys:
            if key in metadata:
                try:
                    value = float(metadata.get(key))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return float(value)
        return float(default)

    def reservoir_flag():
        for key in ('is_reservoir', 'reservoir', 'reservoir_flag'):
            if key in metadata:
                value = metadata.get(key)
                if isinstance(value, str):
                    return 1.0 if value.strip().lower() in {'1', 'true', 'yes', 'y', 'reservoir'} else 0.0
                try:
                    return 1.0 if float(value) > 0.5 else 0.0
                except (TypeError, ValueError):
                    pass
        waterbody_type = str(metadata.get('waterbody_type', metadata.get('lake_type', ''))).lower()
        return 1.0 if 'reservoir' in waterbody_type or 'impound' in waterbody_type else 0.0

    def mean(window, column, default=0.0):
        if column not in window.columns or len(window) == 0:
            return float(default)
        values = pd.to_numeric(window[column], errors='coerce')
        return float(values.mean()) if values.notna().any() else float(default)

    def trend(window, column, default=0.0):
        if column not in window.columns or len(window) < 2:
            return float(default)
        values = pd.to_numeric(window[column], errors='coerce')
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return float(default)
        return float(values.iloc[-1] - values.iloc[0])

    def fraction(window, column, threshold=0.5, default=0.0):
        if column not in window.columns or len(window) == 0:
            return float(default)
        values = pd.to_numeric(window[column], errors='coerce')
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float(default)
        return float((values > float(threshold)).mean())

    depths_arr = np.asarray(depths, dtype=np.float32).reshape(-1)
    max_depth = metadata_float('max_depth_m', 'max_depth', default=float(np.nanmax(depths_arr)) if depths_arr.size else 0.0)
    mean_depth = metadata_float('mean_depth_m', 'mean_depth', default=max_depth * 0.5)
    area_km2 = metadata_float('area_km2', 'surface_area_km2', 'lake_area_km2', default=0.0)
    volume_proxy = max(area_km2, 0.0) * max(mean_depth, 0.0)
    latitude = metadata_float('latitude', 'lat', default=0.0)
    elevation = metadata_float('elevation_m', 'altitude_m', default=0.0)
    if torch.is_tensor(area_profile):
        area_arr = area_profile.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        area_arr = np.asarray([] if area_profile is None else area_profile, dtype=np.float32).reshape(-1)
    if area_arr.size:
        surface_area = float(max(float(area_arr[0]), 1.0e-6))
        area_ratio = np.clip(area_arr / surface_area, 0.0, 10.0)
        hyps_mean = float(np.nanmean(area_ratio))
        hyps_bottom = float(area_ratio[-1])
        hyps_cv = float(np.nanstd(area_ratio) / max(abs(hyps_mean), 1.0e-6))
    else:
        hyps_mean = 1.0
        hyps_bottom = 1.0
        hyps_cv = 0.0
    start_idx = int(max(0, start_idx))
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        idx = int(np.clip(start_idx, 0, len(df) - 1))
        row = df.iloc[idx: idx + 1]
        doy_sin = mean(row, 'doy_sin', default=0.0)
        doy_cos = mean(row, 'doy_cos', default=0.0)
    else:
        idx = 0
        doy_sin = 0.0
        doy_cos = 0.0
    values = [
        max_depth / 50.0,
        mean_depth / 50.0,
        mean_depth / max(max_depth, 1.0e-6),
        np.log1p(max(area_km2, 0.0)) / 8.0,
        np.log1p(volume_proxy) / 10.0,
        hyps_mean,
        hyps_bottom,
        hyps_cv,
        latitude / 90.0,
        elevation / 3000.0,
        reservoir_flag(),
        doy_sin,
        doy_cos,
    ]
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        for days in (30, 90):
            window = df.iloc[max(0, idx - int(days)): max(idx + 1, 1)]
            air_mean = mean(window, 'T_air_C')
            shortwave_mean = mean(window, 'Solar_W_m2')
            longwave_mean = mean(window, 'Longwave_W_m2')
            latent_mean = mean(window, 'latent_heat_upward_W_m2')
            sensible_mean = mean(window, 'sensible_heat_upward_W_m2')
            wind_mean = mean(window, 'wind_speed_m_per_s')
            net_flux_mean = shortwave_mean + longwave_mean - latent_mean - sensible_mean
            values.extend([
                air_mean / 30.0,
                trend(window, 'T_air_C') / 30.0,
                shortwave_mean / 400.0,
                net_flux_mean / 800.0,
                wind_mean / 10.0,
                float(wind_mean ** 3) / 1000.0,
                mean(window, 'LST_surface_C') / 30.0,
                fraction(window, 'LST_observed_flag', threshold=0.5),
                fraction(window, 'LST_is_filled', threshold=0.5, default=1.0),
                mean(window, 'ice_fraction'),
            ])
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM:
        arr = np.pad(arr, (0, ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM - arr.size))
    elif arr.size > ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM:
        arr = arr[:ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM]
    return arr.astype(np.float32)


def build_zero_profile_eof_pca_init_net_prior(
    df,
    depths,
    metadata,
    start_idx,
    *,
    model=None,
    forcing_history=None,
    static_features=None,
    conditioning_features=None,
    thermal_basis=None,
    area_profile=None,
    min_quality=0.05,
):
    """Observable-only zero-profile initializer with a learned low-rank head.

    The network predicts a bounded correction to the EOF/PCA low-DOF prior.
    It receives only past forcing history and static lake features.  Profile
    observations are never read here; training supervision must be handled by
    the caller on train-split profiles only.
    """
    base_profile, info = build_zero_profile_eof_pca_low_dof_prior(
        df,
        depths,
        metadata,
        start_idx,
        thermal_basis=thermal_basis,
        min_quality=min_quality,
    )
    info.update({
        'zero_profile_initializer': 'eof_pca_init_net',
        'zero_profile_base_initializer': 'eof_pca_low_dof',
        'zero_profile_profile_derived_input_used': False,
        'zero_profile_filled_lst_strong_target_used': False,
        'zero_profile_init_net_applied': False,
    })
    if (
        model is None
        or forcing_history is None
        or static_features is None
        or not hasattr(model, 'zero_profile_initial_state_from_basis')
    ):
        info['zero_profile_init_net_reason'] = 'missing_model_or_features'
        return base_profile, info

    device = static_features.device if torch.is_tensor(static_features) else None
    dtype = static_features.dtype if torch.is_tensor(static_features) else torch.float32
    basis_tensors = zero_profile_thermal_basis_tensors_for_depths(
        thermal_basis,
        depths,
        device=device,
        dtype=dtype,
    )
    if basis_tensors is None:
        info['zero_profile_init_net_reason'] = 'missing_or_invalid_basis'
        return base_profile, info

    if conditioning_features is None:
        conditioning_features = _zero_profile_init_conditioning_from_inputs(
            df,
            depths,
            metadata,
            start_idx,
            area_profile=area_profile,
        )
    base_tensor = torch.as_tensor(base_profile, dtype=dtype, device=device).reshape(1, -1)
    encoded = model.zero_profile_initial_state_from_basis(
        base_tensor,
        forcing_history,
        static_features,
        basis_tensors['components_on_depth'],
        basis_tensors['coeff_std'],
        conditioning_features=conditioning_features,
    )
    candidate = encoded['initial_profile_c'].detach().cpu().reshape(-1).numpy().astype(np.float32)
    delta = encoded['initial_delta_c'].detach().cpu().reshape(-1).numpy().astype(np.float32)
    info.update({
        'zero_profile_init_net_applied': True,
        'zero_profile_init_net_component_count': int(basis_tensors['component_count']),
        'zero_profile_init_net_initial_delta_abs_mean_c': float(np.mean(np.abs(delta))),
        'zero_profile_init_net_surface_delta_c': float(delta[0]),
        'zero_profile_init_net_deep_delta_c': float(delta[-1]),
        'zero_profile_init_net_regularization_loss': float(
            encoded['regularization_loss'].detach().cpu().reshape(-1)[0]
        ),
    })
    return np.clip(candidate, 0.0, 38.0).astype(np.float32), info


def _normalize_profile_fusion_time_policy(value):
    text = str(value or 'past_strict').strip().lower().replace('-', '_')
    aliases = {
        'past': 'past_only',
        'previous': 'past_only',
        'history_only': 'past_only',
        'previous_strict': 'past_strict',
        'history_strict': 'past_strict',
        'past_or_future': 'nearest',
        'future_allowed': 'nearest',
        'nearest_past_or_future': 'nearest',
        'nearest_observed': 'nearest',
        'nearest_no_same_day': 'nearest_strict',
        'past_or_future_strict': 'nearest_strict',
        'future_allowed_strict': 'nearest_strict',
    }
    text = aliases.get(text, text)
    if text not in {'past_only', 'past_strict', 'nearest', 'nearest_strict'}:
        raise ValueError(
            'thermal_state_profile_fusion_time_policy must be one of: '
            'past_only, past_strict, nearest, nearest_strict.'
        )
    return text


def _apply_initial_profile_fusion_from_lookup(
    initial_profile,
    *,
    df,
    depths,
    all_lookup,
    start_idx,
    thermal_basis,
    mode='off',
    time_policy='past_strict',
    max_age_days=45,
    min_depth_fraction=0.25,
    max_weight=0.75,
    coeff_limit_sigma=4.0,
):
    mode = str(mode or 'off').strip().lower().replace('-', '_')
    if mode not in {'init_only', 'both', 'on', 'all'}:
        return np.asarray(initial_profile, dtype=np.float32), {
            'thermal_state_profile_fusion_applied': False,
            'thermal_state_profile_fusion_reason': 'disabled',
        }
    if thermal_basis is None or not all_lookup:
        return np.asarray(initial_profile, dtype=np.float32), {
            'thermal_state_profile_fusion_applied': False,
            'thermal_state_profile_fusion_reason': 'missing_basis_or_profiles',
        }
    policy = _normalize_profile_fusion_time_policy(time_policy)
    date_to_index = _date_to_index_map(df)
    candidates = []
    start_idx = int(start_idx)
    max_age_days = int(max_age_days)
    for raw_date, profile in (all_lookup or {}).items():
        date_value = pd.Timestamp(raw_date).normalize()
        if date_value not in date_to_index:
            continue
        profile_idx = int(date_to_index[date_value])
        if policy == 'past_only' and profile_idx > start_idx:
            continue
        if policy == 'past_strict' and profile_idx >= start_idx:
            continue
        if policy == 'nearest_strict' and profile_idx == start_idx:
            continue
        age_days = abs(profile_idx - start_idx)
        if max_age_days >= 0 and age_days > max_age_days:
            continue
        future_flag = 1 if profile_idx > start_idx else 0
        candidates.append((age_days, 1 if future_flag else 0, profile_idx, date_value, profile, future_flag))
    if not candidates:
        return np.asarray(initial_profile, dtype=np.float32), {
            'thermal_state_profile_fusion_applied': False,
            'thermal_state_profile_fusion_reason': 'no_profile_in_time_window',
        }
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    age_days, _tie, profile_idx, date_value, profile, future_flag = candidates[0]
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    depths_arr = np.asarray(depths, dtype=np.float32).reshape(-1)
    initial = np.asarray(initial_profile, dtype=np.float32).reshape(-1)
    if profile.size != initial.size or depths_arr.size != initial.size:
        return initial, {
            'thermal_state_profile_fusion_applied': False,
            'thermal_state_profile_fusion_reason': 'profile_depth_shape_mismatch',
        }
    valid = np.isfinite(profile) & np.isfinite(depths_arr)
    valid_count = int(valid.sum())
    coverage_fraction = float(valid_count / max(1, profile.size))
    if valid_count >= 2:
        depth_span = float(np.nanmax(depths_arr[valid]) - np.nanmin(depths_arr[valid]))
        depth_span_fraction = depth_span / max(float(np.nanmax(depths_arr) - np.nanmin(depths_arr)), 1.0e-6)
    else:
        depth_span_fraction = 0.0
    min_depth_fraction = max(float(min_depth_fraction), 1.0e-6)
    coverage_score = float(np.clip(coverage_fraction / min_depth_fraction, 0.0, 1.0))
    span_score = float(np.clip(depth_span_fraction / min_depth_fraction, 0.0, 1.0))
    if max_age_days > 0:
        age_score = max(0.0, 1.0 - float(age_days) / float(max_age_days))
    elif max_age_days == 0:
        age_score = 1.0 if int(age_days) == 0 else 0.0
    else:
        age_score = 1.0
    gate = float(np.clip(float(max_weight) * coverage_score * span_score * age_score, 0.0, 1.0))
    if gate <= 0.0:
        return initial, {
            'thermal_state_profile_fusion_applied': False,
            'thermal_state_profile_fusion_reason': 'gate_zero',
            'thermal_state_profile_fusion_gate': gate,
            'thermal_state_profile_fusion_age_days': int(age_days),
            'thermal_state_profile_fusion_future_profile_used': bool(future_flag),
            'thermal_state_profile_fusion_coverage_fraction': coverage_fraction,
            'thermal_state_profile_fusion_depth_span_fraction': depth_span_fraction,
        }
    projected, projection_info = project_profile_to_zero_profile_thermal_basis(
        profile,
        depths_arr,
        thermal_basis,
        mask=valid,
        coeff_clip_sigma=coeff_limit_sigma,
    )
    fused = np.clip(initial + gate * (projected - initial), 0.0, 38.0).astype(np.float32)
    return fused, {
        'thermal_state_profile_fusion_applied': True,
        'thermal_state_profile_fusion_mode': mode,
        'thermal_state_profile_fusion_time_policy': policy,
        'thermal_state_profile_fusion_gate': gate,
        'thermal_state_profile_fusion_age_days': int(age_days),
        'thermal_state_profile_fusion_profile_date': date_value.date().isoformat(),
        'thermal_state_profile_fusion_profile_day_idx': int(profile_idx),
        'thermal_state_profile_fusion_future_profile_used': bool(future_flag),
        'thermal_state_profile_fusion_coverage_fraction': coverage_fraction,
        'thermal_state_profile_fusion_depth_span_fraction': depth_span_fraction,
        'thermal_state_profile_fusion_delta_abs_mean_c': float(np.mean(np.abs(fused - initial))),
        **projection_info,
    }


def _lswt_observation_from_row(row, *, min_quality=0.05, strict_observed=True):
    zero = None
    lswt = row.get('lswt_open_water') if row is not None else zero
    if lswt is None:
        return None
    value = torch.as_tensor(lswt).reshape(-1)
    finite = torch.isfinite(value)
    if not bool(torch.any(finite).detach().cpu()):
        return None
    quality = torch.clamp(torch.as_tensor(
        row.get('lst_quality', torch.ones_like(value)),
        device=value.device,
        dtype=value.dtype,
    ).reshape(-1), 0.0, 1.0)
    observed = torch.clamp(torch.as_tensor(
        row.get('lst_observed_flag', torch.ones_like(value)),
        device=value.device,
        dtype=value.dtype,
    ).reshape(-1), 0.0, 1.0)
    filled = torch.clamp(torch.as_tensor(
        row.get('lst_is_filled', torch.zeros_like(value)),
        device=value.device,
        dtype=value.dtype,
    ).reshape(-1), 0.0, 1.0)
    ice_mask = torch.clamp(torch.as_tensor(
        row.get('ice_mask', torch.zeros_like(value)),
        device=value.device,
        dtype=value.dtype,
    ).reshape(-1), 0.0, 1.0)
    ice_fraction = torch.clamp(torch.as_tensor(
        row.get('ice_fraction', torch.zeros_like(value)),
        device=value.device,
        dtype=value.dtype,
    ).reshape(-1), 0.0, 1.0)
    open_water = 1.0 - torch.maximum(ice_mask, ice_fraction)
    gate = finite.to(value.dtype) * (quality >= float(min_quality)).to(value.dtype) * open_water
    if strict_observed:
        gate = gate * (observed >= 0.5).to(value.dtype) * (filled < 0.5).to(value.dtype)
    if not bool(torch.any(gate > 0.0).detach().cpu()):
        return None
    return {
        'value': value,
        'quality': quality,
        'observed': observed,
        'filled': filled,
        'open_water': open_water,
        'gate': gate,
    }


def _empty_lswt_observer_detail(profile, *, mode):
    dtype = profile.dtype
    device = profile.device
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    return {
        'lswt_observer_mode': str(mode),
        'lswt_observer_applied_count': zero,
        'lswt_observer_quality_mean': zero,
        'lswt_observer_open_water_weight_mean': zero,
        'lswt_observer_surface_innovation_c': zero,
        'lswt_observer_mean_abs_delta_c': zero,
        'lswt_observer_max_abs_delta_c': zero,
        'lswt_observer_heat_content_delta_c': zero,
        'lswt_observer_deep_abs_delta_c': zero,
        'lswt_observer_density_guard_scale': torch.tensor(1.0, dtype=dtype, device=device),
        'lswt_observer_filled_lst_used_count': zero,
        'lswt_observer_kalman_gain_surface': zero,
        'lswt_observer_kalman_gain_mean': zero,
        'lswt_observer_observation_error_c': zero,
        'lswt_observer_state_variance_surface': zero,
        'lswt_observer_localization_depth_m': zero,
        'lswt_observer_reservoir_conservative_scale': torch.tensor(1.0, dtype=dtype, device=device),
        'lswt_observer_heat_content_bound_scale': torch.tensor(1.0, dtype=dtype, device=device),
        'lswt_observer_mld_depth_m': zero,
        'lswt_observer_mld_weight_mean': zero,
        'lswt_observer_mld_heat_content_delta_c': zero,
        'lswt_observer_mld_volume_fraction': zero,
        'lswt_observer_mld_surface_to_heat_gain': zero,
    }


def apply_lswt_observer_update(
    profile,
    row,
    depths,
    *,
    mode='off',
    strength=0.08,
    decay_depth_m=2.0,
    max_increment_c=0.5,
    low_rank_deep_update_fraction=0.15,
    heat_content_limit_c=0.35,
    min_quality=0.05,
    area_profile=None,
    metadata=None,
):
    """Apply a bounded raw-open-water LSWT observation update.

    Tenth-version zero-profile diagnostics may select only `off`,
    `conservative_surface`, and `conservative_mld_shallow`. These modes require
    raw open-water LSWT and give filled/reconstructed LST zero strong-update
    gain.
    """
    mode = normalize_lswt_observer_mode(mode)
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    detail = _empty_lswt_observer_detail(profile, mode=mode)
    strength = float(max(0.0, min(1.0, strength)))
    if mode == 'off' or strength <= 0.0 or row is None:
        return profile, detail
    depths = depths.to(device=profile.device, dtype=profile.dtype).reshape(1, -1)

    obs = _lswt_observation_from_row(row, min_quality=min_quality, strict_observed=True)
    if obs is None:
        return profile, detail
    obs_value = obs['value'].to(device=profile.device, dtype=profile.dtype).reshape(-1, 1)
    gate = obs['gate'].to(device=profile.device, dtype=profile.dtype).reshape(-1, 1)
    quality = obs['quality'].to(device=profile.device, dtype=profile.dtype).reshape(-1, 1)
    open_water = obs['open_water'].to(device=profile.device, dtype=profile.dtype).reshape(-1, 1)
    if obs_value.shape[0] == 1 and profile.shape[0] > 1:
        obs_value = obs_value.expand(profile.shape[0], 1)
        gate = gate.expand(profile.shape[0], 1)
        quality = quality.expand(profile.shape[0], 1)
        open_water = open_water.expand(profile.shape[0], 1)
    innovation = obs_value - profile[:, :1]
    kalman_gain_surface = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    kalman_gain_mean = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    observation_error_c = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    state_variance_surface = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    localization_depth_m = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    reservoir_conservative_scale = torch.ones((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    mld_depth_detail = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    mld_weight_mean = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    mld_volume_fraction = torch.zeros((profile.shape[0],), dtype=profile.dtype, device=profile.device)
    mld_surface_to_heat_gain = torch.zeros((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    mld_gate_for_detail = None

    if mode == 'conservative_surface':
        max_depth = torch.clamp(depths.max(), min=1.0)
        reservoir_indicator = torch.as_tensor(
            _metadata_reservoir_indicator(metadata),
            dtype=profile.dtype,
            device=profile.device,
        ).reshape(1, 1)
        reservoir_conservative_scale = 1.0 - 0.65 * reservoir_indicator
        if reservoir_conservative_scale.shape[0] == 1 and profile.shape[0] > 1:
            reservoir_conservative_scale = reservoir_conservative_scale.expand(profile.shape[0], 1)
        if profile.shape[1] > 1:
            left_depths = depths[:, :-1]
            right_depths = depths[:, 1:]
            mid_depths = 0.5 * (left_depths + right_depths)
            dz = torch.clamp(right_depths - left_depths, min=1.0e-3)
            gradients = torch.abs(profile[:, 1:] - profile[:, :-1]) / dz
            thermocline_idx = gradients.argmax(dim=1, keepdim=True)
            thermocline_depth = mid_depths.expand(profile.shape[0], -1).gather(1, thermocline_idx)
            surface_deep_gradient = torch.clamp((profile[:, :1] - profile[:, -1:]).abs(), min=0.0)
            stratification_gate = torch.exp(-surface_deep_gradient / 3.0)
            geometry_cap = torch.clamp(
                0.18 * max_depth * (0.55 + 0.45 * stratification_gate),
                min=0.75,
                max=3.0,
            )
            thermocline_cap = torch.clamp(thermocline_depth - 0.25, min=0.75, max=3.0)
            cap_depth = torch.minimum(geometry_cap, thermocline_cap)
        else:
            cap_depth = torch.ones((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
        decay_limit = torch.full_like(cap_depth, max(float(decay_depth_m), 0.35))
        depth_scale = torch.clamp(torch.minimum(decay_limit, 0.75 * cap_depth), min=0.35, max=2.0)
        vertical_gain = torch.exp(-((depths / depth_scale) ** 2)) * reservoir_conservative_scale
        vertical_gain = torch.where(depths <= cap_depth, vertical_gain, torch.zeros_like(vertical_gain))
        observation_error_c = torch.clamp(
            0.55 + 1.75 * (1.0 - quality) + 0.75 * (1.0 - open_water) + 0.50 * reservoir_indicator,
            min=0.35,
            max=4.5,
        )
        if observation_error_c.shape[0] == 1 and profile.shape[0] > 1:
            observation_error_c = observation_error_c.expand(profile.shape[0], 1)
        state_variance_surface = torch.clamp(vertical_gain[:, :1], min=0.0, max=1.0)
        localization_depth_m = cap_depth
        kalman_gain_surface = vertical_gain[:, :1]
        kalman_gain_mean = vertical_gain.mean(dim=1, keepdim=True)
    elif mode == 'conservative_mld_shallow':
        max_depth = torch.clamp(depths.max(), min=1.0)
        reservoir_indicator = torch.as_tensor(
            _metadata_reservoir_indicator(metadata),
            dtype=profile.dtype,
            device=profile.device,
        ).reshape(1, 1)
        reservoir_conservative_scale = 1.0 - 0.75 * reservoir_indicator
        if reservoir_conservative_scale.shape[0] == 1 and profile.shape[0] > 1:
            reservoir_conservative_scale = reservoir_conservative_scale.expand(profile.shape[0], 1)

        surface_deep_gradient = torch.clamp((profile[:, :1] - profile[:, -1:]).abs(), min=0.0)
        stratification_gate = torch.exp(-surface_deep_gradient / 3.0)
        wind_speed = torch.as_tensor(
            row.get('wind_speed', torch.ones((profile.shape[0],), device=profile.device, dtype=profile.dtype)),
            device=profile.device,
            dtype=profile.dtype,
        ).reshape(-1, 1)
        if wind_speed.shape[0] == 1 and profile.shape[0] > 1:
            wind_speed = wind_speed.expand(profile.shape[0], 1)
        wind_gate = torch.clamp(wind_speed / 6.0, min=0.0, max=1.0)

        temp_diff_from_surface = torch.abs(profile - profile[:, :1])
        mixed_soft = torch.sigmoid((0.65 - temp_diff_from_surface) / 0.20)
        mixed_mask_depths = torch.where(
            mixed_soft >= 0.5,
            depths.expand_as(profile),
            torch.zeros_like(profile),
        )
        profile_mld_depth = mixed_mask_depths.max(dim=1, keepdim=True).values
        mld_upper = torch.clamp(0.45 * max_depth, min=1.0, max=12.0)
        physics_mld_depth = torch.clamp(
            1.0 + 0.10 * max_depth + 0.18 * max_depth * stratification_gate + 1.8 * wind_gate,
            min=1.0,
            max=mld_upper,
        )
        mld_depth = torch.minimum(
            torch.maximum(profile_mld_depth, physics_mld_depth),
            mld_upper,
        )
        edge_width = torch.clamp(0.20 * mld_depth, min=0.75, max=2.5)
        mld_gate = torch.sigmoid((mld_depth - depths) / torch.clamp(edge_width, min=1.0e-3))
        shallow_shape = 0.65 + 0.35 * torch.exp(-depths / torch.clamp(0.85 * mld_depth, min=1.0e-3))
        mld_surface_to_heat_gain = torch.clamp(
            0.35 + 0.25 * stratification_gate + 0.15 * wind_gate,
            min=0.25,
            max=0.75,
        ) * reservoir_conservative_scale
        vertical_gain = torch.clamp(mld_gate * shallow_shape * mld_surface_to_heat_gain, min=0.0, max=1.0)
        shallow_limit = torch.minimum(mld_depth + edge_width, torch.full_like(mld_depth, 25.0))
        near_deep_mask = depths > shallow_limit
        shallow_deep_cap = min(float(low_rank_deep_update_fraction), 0.02)
        vertical_gain = torch.where(
            near_deep_mask,
            torch.minimum(vertical_gain, torch.full_like(vertical_gain, shallow_deep_cap) * reservoir_conservative_scale),
            vertical_gain,
        )
        vertical_gain = torch.where(depths > 25.0, torch.zeros_like(vertical_gain), vertical_gain)
        heat_content_limit_c = min(float(heat_content_limit_c), 0.08) * float(
            torch.clamp(reservoir_conservative_scale.mean(), min=0.25, max=1.0).detach().cpu()
        )
        observation_error_c = torch.clamp(
            0.50 + 1.60 * (1.0 - quality) + 0.75 * (1.0 - open_water) + 0.60 * reservoir_indicator,
            min=0.30,
            max=4.5,
        )
        if observation_error_c.shape[0] == 1 and profile.shape[0] > 1:
            observation_error_c = observation_error_c.expand(profile.shape[0], 1)
        state_variance_surface = torch.clamp(mld_surface_to_heat_gain, min=0.0, max=1.0)
        localization_depth_m = mld_depth
        kalman_gain_surface = vertical_gain[:, :1]
        kalman_gain_mean = vertical_gain.mean(dim=1, keepdim=True)
        mld_depth_detail = mld_depth
        mld_weight_mean = mld_gate.mean(dim=1, keepdim=True)
        mld_gate_for_detail = mld_gate
    else:
        raise ValueError(f"unsupported tenth-version LSWT observer mode: {mode}")
    delta = strength * gate * quality * open_water * vertical_gain * innovation
    delta = torch.clamp(delta, min=-float(max_increment_c), max=float(max_increment_c))
    heat_delta = _area_weighted_delta_c(delta, depths.reshape(-1), area_profile=area_profile)
    heat_content_bound_scale = torch.ones_like(heat_delta)
    if float(heat_content_limit_c) > 0.0:
        scale = torch.minimum(
            torch.ones_like(heat_delta),
            torch.as_tensor(float(heat_content_limit_c), device=profile.device, dtype=profile.dtype)
            / torch.clamp(heat_delta.abs(), min=1.0e-6),
        ).reshape(-1, 1)
        heat_content_bound_scale = scale.reshape(-1)
        delta = delta * scale
        heat_delta = _area_weighted_delta_c(delta, depths.reshape(-1), area_profile=area_profile)
    density_guard_scale = torch.ones((profile.shape[0], 1), dtype=profile.dtype, device=profile.device)
    before_density = _density_inversion_score(profile)
    for _ in range(4):
        candidate = torch.clamp(profile + delta * density_guard_scale, 0.0, 40.0)
        after_density = _density_inversion_score(candidate)
        bad = after_density > torch.maximum(before_density + 0.02, torch.full_like(before_density, 0.05))
        if not bool(torch.any(bad).detach().cpu()):
            break
        density_guard_scale = torch.where(bad.reshape(-1, 1), density_guard_scale * 0.5, density_guard_scale)
    delta = delta * density_guard_scale
    heat_delta = _area_weighted_delta_c(delta, depths.reshape(-1), area_profile=area_profile)
    mld_heat_delta = torch.zeros_like(heat_delta)
    if mld_gate_for_detail is not None:
        depth_vector = depths.reshape(-1)
        if area_profile is None:
            area = torch.ones_like(depth_vector)
        else:
            area = torch.as_tensor(area_profile, device=profile.device, dtype=profile.dtype).reshape(-1)
            if area.numel() != depth_vector.numel():
                area = torch.ones_like(depth_vector)
        dz = layer_thicknesses(depth_vector)
        column_weights = torch.clamp(area * dz, min=1.0e-8).reshape(1, -1)
        mixed_weights = column_weights * torch.clamp(mld_gate_for_detail, min=0.0, max=1.0)
        mixed_weight_sum = torch.clamp(mixed_weights.sum(dim=1), min=1.0e-8)
        mld_heat_delta = (delta * mixed_weights).sum(dim=1) / mixed_weight_sum
        mld_volume_fraction = mixed_weight_sum / torch.clamp(column_weights.sum(dim=1), min=1.0e-8)
    updated = torch.clamp(profile + delta, 0.0, 40.0)
    deep_delta = delta.abs()[:, depths.reshape(-1) > 25.0]
    detail.update({
        'lswt_observer_applied_count': (delta.abs().amax(dim=1) > 0.0).to(profile.dtype).mean().detach(),
        'lswt_observer_quality_mean': quality.mean().detach(),
        'lswt_observer_open_water_weight_mean': open_water.mean().detach(),
        'lswt_observer_surface_innovation_c': innovation.mean().detach(),
        'lswt_observer_mean_abs_delta_c': delta.abs().mean().detach(),
        'lswt_observer_max_abs_delta_c': delta.abs().max().detach(),
        'lswt_observer_heat_content_delta_c': heat_delta.mean().detach(),
        'lswt_observer_deep_abs_delta_c': (
            deep_delta.mean().detach() if deep_delta.numel() else torch.tensor(0.0, dtype=profile.dtype, device=profile.device)
        ),
        'lswt_observer_density_guard_scale': density_guard_scale.mean().detach(),
        'lswt_observer_filled_lst_used_count': torch.zeros((), dtype=profile.dtype, device=profile.device),
        'lswt_observer_kalman_gain_surface': kalman_gain_surface.mean().detach(),
        'lswt_observer_kalman_gain_mean': kalman_gain_mean.mean().detach(),
        'lswt_observer_observation_error_c': observation_error_c.mean().detach(),
        'lswt_observer_state_variance_surface': state_variance_surface.mean().detach(),
        'lswt_observer_localization_depth_m': localization_depth_m.mean().detach(),
        'lswt_observer_reservoir_conservative_scale': reservoir_conservative_scale.mean().detach(),
        'lswt_observer_heat_content_bound_scale': heat_content_bound_scale.mean().detach(),
        'lswt_observer_mld_depth_m': mld_depth_detail.mean().detach(),
        'lswt_observer_mld_weight_mean': mld_weight_mean.mean().detach(),
        'lswt_observer_mld_heat_content_delta_c': mld_heat_delta.mean().detach(),
        'lswt_observer_mld_volume_fraction': mld_volume_fraction.mean().detach(),
        'lswt_observer_mld_surface_to_heat_gain': mld_surface_to_heat_gain.mean().detach(),
    })
    return updated, detail


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
    zero_profile_initializer='low_dof',
    spinup_lswt_observer_mode='off',
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    zero_profile_thermal_basis=None,
    thermal_state_profile_fusion_mode='off',
    thermal_state_profile_fusion_time_policy='past_strict',
    thermal_state_profile_fusion_max_age_days=45,
    thermal_state_profile_fusion_min_depth_fraction=0.25,
    thermal_state_profile_fusion_max_weight=0.75,
    thermal_state_profile_fusion_coeff_limit_sigma=4.0,
    lswt_observer_low_rank_deep_update_fraction=0.15,
    lswt_observer_heat_content_limit_c=0.35,
    lswt_observer_min_quality=0.05,
    task_mode='analysis',
    area_profile=None,
    hard_density_stability=False,
):
    """Return a reconstruction/rollout start state and optional spinup trajectory.

    profile mode uses the latest observed start profile when available.
    If no profile exists, profile mode falls back to tenth-version zero-profile
    prior_spinup instead of a uniform-LST water column.
    """
    init_mode = str(init_mode or 'profile').strip().lower()
    zero_profile_initializer = normalize_zero_profile_initializer_mode(zero_profile_initializer)
    spinup_lswt_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    if zero_profile_thermal_basis is None and model is not None:
        zero_profile_thermal_basis = getattr(model, 'zero_profile_thermal_basis', None)
    if init_mode not in {'profile', 'prior_spinup', 'zero_profile_low_dof'}:
        raise ValueError(
            "init_mode must be profile, prior_spinup, or zero_profile_low_dof."
        )
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
                'zero_profile_initializer': zero_profile_initializer,
                'spinup_lswt_observer_mode': spinup_lswt_observer_mode,
                'prior_info': {},
            }
        init_mode = 'prior_spinup'

    if init_mode == 'zero_profile_low_dof':
        zero_profile_initializer = 'low_dof'
        init_mode = 'prior_spinup'

    if init_mode == 'prior_spinup':
        if rollout_start_date is None and rollout_start_idx < int(spinup_days):
            rollout_start_idx = min(max(0, int(spinup_days)), len(df) - 1)
        start_idx = max(0, rollout_start_idx - max(0, int(spinup_days)))
    else:
        start_idx = rollout_start_idx

    if zero_profile_initializer == 'low_dof':
        initial_profile, prior_info = build_zero_profile_low_dof_prior(df, depths, metadata, start_idx)
    elif zero_profile_initializer == 'lswt_climatology_low_dof':
        initial_profile, prior_info = build_zero_profile_lswt_climatology_low_dof_prior(
            df,
            depths,
            metadata,
            start_idx,
            min_quality=lswt_observer_min_quality,
        )
    elif zero_profile_initializer == 'eof_pca_low_dof':
        initial_profile, prior_info = build_zero_profile_eof_pca_low_dof_prior(
            df,
            depths,
            metadata,
            start_idx,
            thermal_basis=zero_profile_thermal_basis,
            min_quality=lswt_observer_min_quality,
        )
    elif zero_profile_initializer == 'eof_pca_init_net':
        initial_profile, prior_info = build_zero_profile_eof_pca_init_net_prior(
            df,
            depths,
            metadata,
            start_idx,
            model=model,
            forcing_history=forcing_rows[start_idx]['history_features'],
            static_features=static_features,
            thermal_basis=zero_profile_thermal_basis,
            area_profile=area_profile,
            min_quality=lswt_observer_min_quality,
        )
    else:
        raise ValueError(
            f"unsupported tenth-version zero-profile initializer: {zero_profile_initializer}"
        )

    initial_profile, fusion_info = _apply_initial_profile_fusion_from_lookup(
        initial_profile,
        df=df,
        depths=depths,
        all_lookup=all_lookup,
        start_idx=start_idx,
        thermal_basis=zero_profile_thermal_basis,
        mode=thermal_state_profile_fusion_mode,
        time_policy=thermal_state_profile_fusion_time_policy,
        max_age_days=thermal_state_profile_fusion_max_age_days,
        min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
        max_weight=thermal_state_profile_fusion_max_weight,
        coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
    )
    prior_info.update(fusion_info)

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
            observer_detail = _empty_lswt_observer_detail(current, mode=spinup_lswt_observer_mode)
            if float(spinup_lst_assimilation_strength) > 0.0 and next_row is not None:
                current, observer_detail = apply_lswt_observer_update(
                    current,
                    next_row,
                    depth_tensor,
                    mode=spinup_lswt_observer_mode,
                    strength=spinup_lst_assimilation_strength,
                    decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    max_increment_c=spinup_lst_assimilation_max_increment_c,
                    low_rank_deep_update_fraction=lswt_observer_low_rank_deep_update_fraction,
                    heat_content_limit_c=lswt_observer_heat_content_limit_c,
                    min_quality=lswt_observer_min_quality,
                    area_profile=area_profile,
                    metadata=metadata,
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
                'spinup_lswt_observer_mode': spinup_lswt_observer_mode,
                'spinup_lswt_observer_applied_count': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_applied_count',
                ),
                'spinup_lswt_observer_surface_innovation_c': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_surface_innovation_c',
                ),
                'spinup_lswt_observer_mean_abs_delta_c': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_mean_abs_delta_c',
                ),
                'spinup_lswt_observer_max_abs_delta_c': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_max_abs_delta_c',
                ),
                'spinup_lswt_observer_heat_content_delta_c': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_heat_content_delta_c',
                ),
                'spinup_lswt_observer_deep_abs_delta_c': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_deep_abs_delta_c',
                ),
                'spinup_lswt_observer_filled_lst_used_count': _diagnostic_scalar(
                    observer_detail,
                    'lswt_observer_filled_lst_used_count',
                ),
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
        'zero_profile_initializer': zero_profile_initializer,
        'spinup_lswt_observer_mode': spinup_lswt_observer_mode,
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
        lst_observed_flag = _lst_observed_flag_value(row, lst_is_filled)
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


def _profile_lookup(profile_obs, depths, return_masks=False, target_mode='grid_masked'):
    lookup = {}
    masks = {}
    if profile_obs is None or profile_obs.empty:
        return (lookup, masks) if return_masks else lookup
    mode = str(target_mode or 'grid_masked').strip().lower().replace('-', '_')
    if mode not in {'grid_masked', 'observed_point_strict'}:
        raise ValueError(
            "profile loss target mode must be one of: grid_masked, observed_point_strict."
        )
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
        if mode == 'observed_point_strict':
            profile = np.full_like(depths, np.nan, dtype=np.float32)
            mask = np.zeros_like(depths, dtype=bool)
            values_by_idx = {}
            for depth_value, temp_value in zip(z, temp):
                nearest_idx = int(np.argmin(np.abs(depths - float(depth_value))))
                values_by_idx.setdefault(nearest_idx, []).append(float(temp_value))
            for nearest_idx, values in values_by_idx.items():
                profile[nearest_idx] = float(np.mean(values))
                mask[nearest_idx] = True
        elif z.size == 1:
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
