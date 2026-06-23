"""Conditional lake priors controlled by attributes and climate state.

The functions in this module replace fixed seasonal heuristics with broad,
metadata-driven priors. They are intentionally conservative: explicit metadata
overrides win, otherwise the code infers a coarse thermal regime from latitude,
air/LST climatology, and ice evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lake_metadata import geographic_climate_zone


THERMAL_REGIMES = {
    'cold_ice_prone',
    'temperate',
    'warm_nonfreezing',
    'tropical_warm',
}


_THERMAL_REGIME_ALIASES = {
    'cold': 'cold_ice_prone',
    'ice_prone': 'cold_ice_prone',
    'cold_dimictic': 'cold_ice_prone',
    'dimictic': 'cold_ice_prone',
    'freezing': 'cold_ice_prone',
    'temperate_cool': 'temperate',
    'temperate_dimictic': 'temperate',
    'warm': 'warm_nonfreezing',
    'warm_subtropical': 'warm_nonfreezing',
    'subtropical': 'warm_nonfreezing',
    'nonfreezing': 'warm_nonfreezing',
    'tropical': 'tropical_warm',
    'tropical_nonfreezing': 'tropical_warm',
}


def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _metadata_float(metadata: dict | None, *names, default=np.nan):
    metadata = metadata or {}
    for name in names:
        if name in metadata:
            value = _safe_float(metadata.get(name), default=default)
            if np.isfinite(value):
                return value
    return default


def _normalize_thermal_regime(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    normalized = _THERMAL_REGIME_ALIASES.get(normalized, normalized)
    return normalized if normalized in THERMAL_REGIMES else None


def _series(frame: pd.DataFrame | None, candidates, default=np.nan) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=np.float32)
    lower_to_column = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        column = candidate if candidate in frame.columns else lower_to_column.get(str(candidate).lower())
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        if not values.isna().all():
            return values.astype(np.float32)
    return pd.Series(default, index=frame.index, dtype=np.float32)


def _finite_mean(values: pd.Series, default=np.nan) -> float:
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float(default)
    return float(values.mean())


def _finite_median(values: pd.Series, default=np.nan) -> float:
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float(default)
    return float(values.median())


def _finite_min(values: pd.Series, default=np.nan) -> float:
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float(default)
    return float(values.min())


def _surface_reference_temperature_c(metadata: dict | None, frame: pd.DataFrame | None) -> float:
    explicit = _metadata_float(
        metadata,
        'mean_surface_temp_c',
        'mean_lswt_c',
        'surface_temp_mean_c',
        'water_temp_mean_c',
    )
    if np.isfinite(explicit):
        return float(explicit)

    lst = _series(
        frame,
        ['LSWT_open_water_C', 'LST_surface_C', 'LST_filled_C', 'LST_raw_C', 'LST_observed_raw_C'],
    )
    reference = _finite_median(lst, default=np.nan)
    if np.isfinite(reference):
        return reference

    air = _series(frame, ['T_air_C', 'air_temp_C', 't2m_C'])
    reference = _finite_mean(air, default=np.nan)
    if np.isfinite(reference):
        return reference
    return 12.0


def infer_thermal_regime(metadata: dict | None, frame: pd.DataFrame | None = None) -> str:
    """Infer a coarse lake thermal regime.

    Priority: explicit metadata override, automatic climate inference, then
    temperate fallback.
    """
    metadata = metadata or {}
    explicit = _normalize_thermal_regime(metadata.get('thermal_regime'))
    if explicit is not None:
        return explicit

    zone = str(metadata.get('geographic_climate_zone') or geographic_climate_zone(metadata))
    latitude = _metadata_float(metadata, 'latitude', 'lat', 'latitude_deg', default=np.nan)
    abs_lat = abs(latitude) if np.isfinite(latitude) else np.nan

    air = _series(frame, ['T_air_C', 'air_temp_C', 't2m_C'])
    lst = _series(frame, ['LSWT_open_water_C', 'LST_surface_C', 'LST_filled_C', 'LST_raw_C'])
    ice = _series(frame, ['ice_fraction', 'ice_cover_fraction', 'lake_ice_fraction', 'ice_cover'])
    ice = ice.clip(lower=0.0, upper=1.0)

    air_mean = _finite_mean(air, default=np.nan)
    air_min = _finite_min(air, default=np.nan)
    lst_mean = _finite_mean(lst, default=np.nan)
    ice_max = float(ice.max(skipna=True)) if not ice.empty and not ice.isna().all() else 0.0
    freezing_fraction = float((air <= 0.0).mean()) if not air.empty and not air.isna().all() else 0.0

    has_lat = np.isfinite(abs_lat)
    explicit_ice = ice_max > 0.15
    cold_by_climate = zone == 'cold_high_latitude'
    cold_by_series = (
        np.isfinite(air_min)
        and air_min <= 0.0
        and ((has_lat and abs_lat >= 38.0) or freezing_fraction >= 0.10)
    )
    if explicit_ice or cold_by_climate or cold_by_series:
        return 'cold_ice_prone'

    tropical_by_position = zone == 'tropical' or (has_lat and abs_lat < 23.5)
    if (
        tropical_by_position
        and np.isfinite(air_mean)
        and np.isfinite(air_min)
        and air_mean >= 18.0
        and air_min >= 8.0
    ):
        return 'tropical_warm'
    if tropical_by_position and np.isfinite(lst_mean) and lst_mean >= 22.0:
        return 'tropical_warm'

    warm_by_climate = zone in {'tropical', 'warm_subtropical'}
    warm_latitude_context = (not has_lat) or abs_lat < 38.0
    warm_by_series = (
        np.isfinite(air_min)
        and air_min > 3.0
        and warm_latitude_context
        and (
            (np.isfinite(air_mean) and air_mean >= 14.0)
            or (np.isfinite(lst_mean) and lst_mean >= 16.0)
        )
    )
    if warm_by_climate or warm_by_series:
        return 'warm_nonfreezing'

    return 'temperate'


def infer_bottom_temp_prior_c(
    metadata: dict | None,
    frame: pd.DataFrame | None = None,
    max_depth=None,
) -> float:
    """Infer a bottom-temperature prior in deg C.

    This is a prior, not a truth label. Missing bottom observations can use it
    as a neutral fill value; learned residuals remain free to move away from it.
    """
    explicit = _metadata_float(
        metadata,
        'bottom_temp_prior_c',
        'bottom_temperature_prior_c',
        'deep_temp_prior_c',
        default=np.nan,
    )
    if np.isfinite(explicit):
        return float(np.clip(explicit, 0.0, 35.0))

    metadata = metadata or {}
    regime = infer_thermal_regime(metadata, frame)
    depth = _safe_float(max_depth, default=np.nan)
    if not np.isfinite(depth) or depth <= 0.0:
        depth = _metadata_float(metadata, 'runtime_max_depth_m', 'max_depth_m', 'max_depth', default=20.0)
    if not np.isfinite(depth) or depth <= 0.0:
        depth = 20.0

    surface_ref = _surface_reference_temperature_c(metadata, frame)

    if regime == 'cold_ice_prone':
        return 4.0
    if regime == 'tropical_warm':
        gradient = float(np.clip(0.12 * depth, 2.0, 8.0))
        return float(np.clip(surface_ref - gradient, 18.0, 30.0))
    if regime == 'warm_nonfreezing':
        gradient = float(np.clip(0.10 * depth, 2.0, 7.0))
        return float(np.clip(surface_ref - gradient, 10.0, 28.0))

    gradient = float(np.clip(0.18 * depth, 3.0, 10.0))
    return float(np.clip(surface_ref - gradient, 4.0, 24.0))


def infer_ice_risk_prior(metadata: dict | None, frame: pd.DataFrame | None) -> pd.Series:
    """Return a 0-1 daily ice-risk prior.

    Warm and tropical regimes rely almost entirely on explicit ice evidence;
    cold regimes can infer risk from cold air and cold surface retrievals.
    """
    if frame is None or frame.empty:
        return pd.Series(dtype=np.float32)

    regime = infer_thermal_regime(metadata, frame)
    air = _series(frame, ['T_air_C', 'air_temp_C', 't2m_C'])
    lst = _series(frame, ['LST_raw_C', 'LST_observed_raw_C', 'LST_surface_C', 'LST_filled_C'])
    ice_raw = _series(frame, ['ice_fraction', 'ice_cover_fraction', 'lake_ice_fraction', 'ice_cover'], default=np.nan)
    ice_observed = _series(frame, ['ice_fraction_observed', 'ice_cover_observed'], default=np.nan)
    if ice_observed.empty or ice_observed.isna().all():
        ice_observed_mask = ice_raw.notna()
    else:
        ice_observed_mask = ice_observed.fillna(0.0) > 0.5
    ice = ice_raw.fillna(0.0).clip(lower=0.0, upper=1.0)

    risk = ice.astype(np.float64).copy()
    if regime == 'cold_ice_prone':
        air_filled = air.ffill().bfill()
        lst_filled = lst.ffill().bfill()
        rolling_air_7d = air_filled.rolling(window=7, min_periods=1).mean()
        infer_mask = ~ice_observed_mask

        cold_skin = (air_filled <= 0.0) & (lst_filled <= 1.0)
        persistent_cold = (rolling_air_7d <= 0.0) & (lst_filled <= 2.0)
        hard_freeze_air = air_filled <= -2.0
        shoulder_ice = (rolling_air_7d <= 1.0) & (air_filled <= 2.0) & (lst_filled <= 3.0)

        risk = risk.mask(infer_mask & cold_skin, np.maximum(risk, 1.0))
        risk = risk.mask(infer_mask & persistent_cold, np.maximum(risk, 0.80))
        risk = risk.mask(infer_mask & hard_freeze_air, np.maximum(risk, 0.70))
        risk = risk.mask(infer_mask & shoulder_ice, np.maximum(risk, 0.60))

    return risk.fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float32)


def infer_freezing_lst_prior(metadata: dict | None, frame: pd.DataFrame | None) -> pd.Series:
    """Boolean prior for replacing LST with 0 C under likely ice cover."""
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)

    regime = infer_thermal_regime(metadata, frame)
    ice_risk = infer_ice_risk_prior(metadata, frame)
    air = _series(frame, ['T_air_C', 'air_temp_C', 't2m_C'])
    air = air.ffill().bfill().fillna(10.0)
    mask = (regime == 'cold_ice_prone') & (ice_risk >= 0.5) & (air <= 0.5)
    return pd.Series(mask, index=frame.index, dtype=bool)


__all__ = [
    'THERMAL_REGIMES',
    'infer_bottom_temp_prior_c',
    'infer_freezing_lst_prior',
    'infer_ice_risk_prior',
    'infer_thermal_regime',
]
