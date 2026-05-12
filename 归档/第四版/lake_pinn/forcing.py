# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

def estimate_surface_bulk_temperature(
    lst_surface_c: np.ndarray,
    air_temp_c: np.ndarray,
    solar_w_m2: np.ndarray,
    wind_speed_m_per_s: np.ndarray,
    mixed_layer_depth_m: np.ndarray | None = None,
    skin_cooling_coef: float = SURFACE_SKIN_COOLING_COEF,
    air_blend: float = SURFACE_AIR_BLEND,
) -> np.ndarray:
    lst_surface_c = np.asarray(lst_surface_c, dtype=np.float64)
    air_temp_c = np.asarray(air_temp_c, dtype=np.float64)
    solar_w_m2 = np.asarray(solar_w_m2, dtype=np.float64)
    wind_speed_m_per_s = np.asarray(wind_speed_m_per_s, dtype=np.float64)
    if mixed_layer_depth_m is None:
        mixed_layer_depth_m = np.full_like(lst_surface_c, 2.0, dtype=np.float64)
    else:
        mixed_layer_depth_m = np.asarray(mixed_layer_depth_m, dtype=np.float64)

    skin_cooling_coef = float(max(skin_cooling_coef, 0.0))
    air_blend = float(np.clip(air_blend, 0.0, 1.0))
    wind_damping = np.exp(-np.clip(wind_speed_m_per_s, 0.0, 12.0) / 3.0)
    solar_skin_excess = skin_cooling_coef * np.clip(solar_w_m2, 0.0, None) * wind_damping
    solar_skin_excess = np.clip(solar_skin_excess, 0.0, 6.0)
    bulk_temp = lst_surface_c - solar_skin_excess
    stable_surface_excess = np.clip(lst_surface_c - air_temp_c, 0.0, 8.0)
    shallow_mld_factor = 1.0 / (1.0 + np.exp((mixed_layer_depth_m - 2.0) / 0.6))
    stable_stratification_factor = (
        (1.0 / (1.0 + np.exp(-(np.clip(solar_w_m2, 0.0, None) - 120.0) / 35.0)))
        * np.exp(-np.clip(wind_speed_m_per_s, 0.0, 12.0) / 2.5)
        * shallow_mld_factor
    )
    extra_skin_offset = np.clip(0.45 * stable_surface_excess * stable_stratification_factor, 0.0, 2.5)
    bulk_temp = bulk_temp - extra_skin_offset
    air_residual = np.clip(air_temp_c - bulk_temp, -4.0, 4.0)
    bulk_temp = bulk_temp + air_blend * air_residual
    bulk_temp = np.minimum(bulk_temp, lst_surface_c - 0.15 * stable_stratification_factor)
    return np.clip(bulk_temp, -1.0, 35.0)


def suppress_spring_lst_warm_spikes(
    lst_surface_c: np.ndarray,
    air_temp_c: np.ndarray,
    doy: np.ndarray,
    spring_start_doy: float = 75.0,
    spring_end_doy: float = 125.0,
    max_daily_warming_c: float = 0.85,
    max_above_air_c: float = 4.0,
    max_above_rolling_median_c: float = 2.0,
) -> np.ndarray:
    """Limit isolated warm-skin LST spikes during spring ice-out/warming.

    MODIS skin temperature can briefly run much warmer than bulk water on sunny
    spring days. The PINN uses LST as an input and rolling surface target, so
    those spikes need to be treated as unreliable skin excursions, not as a
    full mixed-layer warming event.
    """
    lst_surface_c = np.asarray(lst_surface_c, dtype=np.float64)
    air_temp_c = np.asarray(air_temp_c, dtype=np.float64)
    doy = np.asarray(doy, dtype=np.float64)
    if lst_surface_c.size == 0:
        return lst_surface_c

    filled = (
        pd.Series(lst_surface_c)
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )
    air_filled = (
        pd.Series(air_temp_c)
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )
    rolling_median = (
        pd.Series(filled)
        .rolling(window=7, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )

    corrected = filled.copy()
    for idx in range(corrected.size):
        if not np.isfinite(corrected[idx]):
            continue
        if doy[idx] < spring_start_doy or doy[idx] > spring_end_doy:
            continue

        cap_values = [
            air_filled[idx] + max_above_air_c,
            rolling_median[idx] + max_above_rolling_median_c,
        ]
        if idx > 0 and np.isfinite(corrected[idx - 1]):
            cap_values.append(corrected[idx - 1] + max_daily_warming_c)
        warm_cap = min(cap_values)
        if corrected[idx] > warm_cap:
            corrected[idx] = warm_cap

    return np.clip(corrected, -1.0, 35.0)


def apply_forcing_adjustments(
    df: pd.DataFrame,
    solar_shading_factor: float = 1.0,
    surface_skin_cooling_coef: float = SURFACE_SKIN_COOLING_COEF,
    surface_air_blend: float = SURFACE_AIR_BLEND,
) -> pd.DataFrame:
    adjusted = df.copy()
    solar_shading_factor = float(max(solar_shading_factor, 0.0))
    if solar_shading_factor != 1.0:
        adjusted['Solar_W_m2'] = adjusted['Solar_W_m2'] * solar_shading_factor
        adjusted['Solar_J_m2'] = adjusted['Solar_J_m2'] * solar_shading_factor
    if 'LST_surface_C_raw' not in adjusted.columns:
        adjusted['LST_surface_C_raw'] = adjusted['LST_surface_C']
    doy_values = (
        adjusted['full_doy'].to_numpy(dtype=np.float64)
        if 'full_doy' in adjusted.columns
        else pd.to_datetime(adjusted['Date']).dt.dayofyear.to_numpy(dtype=np.float64)
    )
    corrected_lst = suppress_spring_lst_warm_spikes(
        lst_surface_c=adjusted['LST_surface_C_raw'].to_numpy(dtype=np.float64),
        air_temp_c=adjusted['T_air_C'].to_numpy(dtype=np.float64),
        doy=doy_values,
    )
    adjusted['LST_surface_C'] = corrected_lst
    adjusted['LST_spring_spike_correction_C'] = adjusted['LST_surface_C_raw'] - adjusted['LST_surface_C']
    adjusted['SurfaceBulkTarget_C'] = estimate_surface_bulk_temperature(
        lst_surface_c=adjusted['LST_surface_C'].to_numpy(dtype=np.float64),
        air_temp_c=adjusted['T_air_C'].to_numpy(dtype=np.float64),
        solar_w_m2=adjusted['Solar_W_m2'].to_numpy(dtype=np.float64),
        wind_speed_m_per_s=adjusted['wind_speed_m_per_s'].to_numpy(dtype=np.float64),
        mixed_layer_depth_m=adjusted['MixedLayerDepth_m'].to_numpy(dtype=np.float64),
        skin_cooling_coef=surface_skin_cooling_coef,
        air_blend=surface_air_blend,
    )
    return adjusted


def compute_runtime_surface_target(
    df,
    day_idx,
    runtime_skin_cooling_coef,
    base_surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    if 'SurfaceBulkTarget_C' not in df.columns:
        return float(df['LST_surface_C'].iloc[day_idx])
    base_target = float(df['SurfaceBulkTarget_C'].iloc[day_idx])
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else base_target
    baseline_coef = max(float(base_surface_skin_cooling_coef), 1.0e-6)
    runtime_coef = float(np.clip(runtime_skin_cooling_coef, 0.005, 0.08))
    skin_gap = raw_lst_surface - base_target
    scaled_gap = skin_gap * (runtime_coef / baseline_coef)
    adjusted_target = raw_lst_surface - scaled_gap
    return float(np.clip(adjusted_target, -1.0, 35.0))
