# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .lake_metadata import infer_metadata, is_geographic_warm_deep_lake, metadata_static_features

def pick_numeric_series(frame: pd.DataFrame, candidates, default=np.nan):
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors='coerce')
            if not values.isna().all():
                return values
    return pd.Series(default, index=frame.index, dtype=np.float32)


def normalize_data_fill_mode(mode: str | None) -> str:
    mode = str(mode or 'reconstruction').strip().lower()
    aliases = {
        'analysis': 'reconstruction',
        'hindcast': 'reconstruction',
        'strict_forecast': 'forecast',
        'causal': 'forecast',
    }
    mode = aliases.get(mode, mode)
    if mode not in {'reconstruction', 'forecast'}:
        raise ValueError(f"Unsupported data_fill_mode={mode!r}; expected reconstruction or forecast.")
    return mode


def fill_numeric_series(
    values,
    data_fill_mode: str = 'reconstruction',
    default=np.nan,
    lower=None,
    upper=None,
) -> pd.Series:
    """Fill a numeric daily series with either offline or causal semantics.

    reconstruction mode is allowed to use both sides of a gap for analysis and
    hindcast reconstruction. forecast mode uses only values available up to the
    current day; leading gaps fall back to the provided neutral/default value.
    """
    filled = pd.to_numeric(values, errors='coerce')
    data_fill_mode = normalize_data_fill_mode(data_fill_mode)
    if data_fill_mode == 'forecast':
        filled = filled.ffill()
        if default is not None:
            filled = filled.fillna(float(default))
    else:
        filled = filled.interpolate(method='linear', limit_direction='both').bfill().ffill()
        if default is not None:
            filled = filled.fillna(float(default))
    if lower is not None or upper is not None:
        filled = filled.clip(lower=lower, upper=upper)
    return filled.astype(np.float32)


def first_existing_column(frame: pd.DataFrame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


LST_QUALITY_COLUMNS = [
    'LST_is_filled',
    'LST_gap_length',
    'LST_valid_pixel_fraction',
    'LST_qc_good_fraction',
    'LST_spatial_std_C',
]

LST_QUALITY_ALIASES = {
    'LST_is_filled': [
        'LST_is_filled',
        'lst_is_filled',
        'LST_daily_mean_has_reconstructed_component',
        'lst_daily_mean_has_reconstructed_component',
        'has_reconstructed_component',
        'is_filled',
        'filled',
        'cloud_filled',
        'gap_filled',
        'was_interpolated',
        'interpolated',
        'is_reconstructed',
        'reconstructed',
        'LST_pipeline_filled',
    ],
    'LST_gap_length': [
        'LST_gap_length',
        'lst_gap_length',
        'gap_length',
        'gap_days',
        'cloud_gap_length',
        'interpolation_gap_days',
    ],
    'LST_valid_pixel_fraction': [
        'LST_valid_pixel_fraction',
        'lst_valid_pixel_fraction',
        'LST_valid',
        'lst_valid',
        'valid_pixel_fraction',
        'valid_fraction',
        'good_pixel_fraction',
        'clear_pixel_fraction',
    ],
    'LST_qc_good_fraction': [
        'LST_qc_good_fraction',
        'lst_qc_good_fraction',
        'LST_observation_weight',
        'lst_observation_weight',
        'ContinuousLST_QA',
        'continuous_lst_qa',
        'qc_good_fraction',
        'good_qc_fraction',
        'qc_fraction',
        'MOD11A1_061_QC_Day_good_fraction',
    ],
    'LST_spatial_std_C': [
        'LST_spatial_std_C',
        'lst_spatial_std_c',
        'spatial_std_C',
        'std_C',
        'lst_std_c',
        'LST_std_C',
    ],
}


def first_existing_column_ci(frame: pd.DataFrame, candidates):
    """Find a column by exact name first, then case-insensitive aliases."""
    exact = first_existing_column(frame, candidates)
    if exact is not None:
        return exact
    lower_to_column = {str(column).lower(): column for column in frame.columns}
    for column in candidates:
        match = lower_to_column.get(str(column).lower())
        if match is not None:
            return match
    return None


def pick_numeric_series_ci(frame: pd.DataFrame, candidates, default=np.nan):
    for candidate in candidates:
        column = first_existing_column_ci(frame, [candidate])
        if column is None:
            continue
        raw_values = frame[column]
        values = pd.to_numeric(raw_values, errors='coerce')
        if values.isna().all():
            text_values = raw_values.astype(str).str.strip().str.lower()
            bool_values = text_values.map(
                {
                    'true': 1.0,
                    't': 1.0,
                    'yes': 1.0,
                    'y': 1.0,
                    '1': 1.0,
                    'false': 0.0,
                    'f': 0.0,
                    'no': 0.0,
                    'n': 0.0,
                    '0': 0.0,
                }
            )
            if not bool_values.isna().all():
                values = bool_values
        if not values.isna().all():
            return values
    return pd.Series(default, index=frame.index, dtype=np.float32)


def normalize_quality_fraction(values: pd.Series) -> pd.Series:
    """Normalize common QA encodings to 0-1 where 1 means most reliable."""
    values = pd.to_numeric(values, errors='coerce')
    if values.isna().all():
        return values
    max_value = float(values.max(skipna=True))
    if max_value > 10.0:
        values = values / 100.0
    elif max_value > 1.5:
        # Continuous MODIS QA is commonly encoded as 0..3.
        values = values / 3.0
    return values.clip(0.0, 1.0)


def infer_lst_qc_good_fraction(lst: pd.DataFrame) -> pd.Series:
    """Infer an internal 0-1 LST quality score from available QA metadata."""
    explicit_quality = pick_numeric_series_ci(
        lst,
        [
            'LST_qc_good_fraction',
            'lst_qc_good_fraction',
            'LST_observation_weight',
            'lst_observation_weight',
            'ContinuousLST_QA',
            'continuous_lst_qa',
        ],
    )
    if not explicit_quality.isna().all():
        return normalize_quality_fraction(explicit_quality)

    quality = pd.Series(np.nan, index=lst.index, dtype=np.float32)

    class_column = first_existing_column_ci(lst, ['LST_daily_mean_observation_class'])
    if class_column is not None:
        classes = lst[class_column].astype(str).str.strip().str.lower()
        class_quality = classes.map(
            {
                'day_and_night_observed': 1.0,
                'day_observed_night_reconstructed': 0.65,
                'night_observed_day_reconstructed': 0.55,
                'both_reconstructed': 0.25,
            }
        )
        quality = quality.combine_first(class_quality.astype(np.float32))

    day_available = pick_numeric_series_ci(lst, ['MODIS_original_day_available'])
    night_available = pick_numeric_series_ci(lst, ['MODIS_original_night_available'])
    if not day_available.isna().all() or not night_available.isna().all():
        day_available = day_available.fillna(0.0).clip(0.0, 1.0)
        night_available = night_available.fillna(0.0).clip(0.0, 1.0)
        observed_count = day_available + night_available
        availability_quality = pd.Series(0.25, index=lst.index, dtype=np.float32)
        availability_quality[observed_count >= 1.0] = 0.60
        availability_quality[observed_count >= 2.0] = 1.0
        quality = quality.combine_first(availability_quality)

    fully_observed = pick_numeric_series_ci(lst, ['LST_daily_mean_is_fully_observed'])
    has_any_original = pick_numeric_series_ci(lst, ['LST_daily_mean_has_any_original_observation'])
    has_reconstructed = pick_numeric_series_ci(lst, ['LST_daily_mean_has_reconstructed_component'])
    if (
        not fully_observed.isna().all()
        or not has_any_original.isna().all()
        or not has_reconstructed.isna().all()
    ):
        metadata_quality = pd.Series(np.nan, index=lst.index, dtype=np.float32)
        metadata_quality[has_reconstructed.fillna(0.0) > 0.5] = 0.35
        metadata_quality[has_any_original.fillna(0.0) > 0.5] = 0.65
        metadata_quality[fully_observed.fillna(0.0) > 0.5] = 1.0
        quality = quality.combine_first(metadata_quality)

    return quality.clip(0.0, 1.0)


def prepare_lst_daily_frame(lst: pd.DataFrame) -> pd.DataFrame:
    """Build daily LST plus optional quality metadata.

    Filled/reconstructed LST can be useful forcing, but it should not carry the
    same training or assimilation authority as direct clear-sky retrievals.
    """
    out = lst[['Date']].copy()
    if first_existing_column_ci(lst, ['LST_surface_C', 'LSTcont_daily_mean_C']) is not None:
        lst_c = pick_numeric_series_ci(lst, ['LST_surface_C', 'LSTcont_daily_mean_C'])
        out['LST_surface_K'] = lst_c + 273.15
    else:
        out['LST_surface_K'] = pick_numeric_series_ci(
            lst,
            ['LST_surface_K', 'LSTcont_daily_mean_K', 'MOD11A1_061_LST_Day_1km', 'lst_surface_k'],
        )
    out.loc[out['LST_surface_K'] <= 0, 'LST_surface_K'] = np.nan

    aggregations = {'LST_surface_K': 'mean'}
    inferred_qc_good_fraction = infer_lst_qc_good_fraction(lst)
    for standard_name, aliases in LST_QUALITY_ALIASES.items():
        values = pick_numeric_series_ci(lst, aliases)
        if standard_name == 'LST_qc_good_fraction' and values.isna().all():
            values = inferred_qc_good_fraction
        if values.isna().all():
            continue
        if standard_name == 'LST_is_filled':
            values = (values.fillna(0.0) > 0.5).astype(np.float32)
            aggregations[standard_name] = 'max'
        elif standard_name == 'LST_gap_length':
            values = values.clip(lower=0.0)
            aggregations[standard_name] = 'max'
        elif standard_name in ('LST_valid_pixel_fraction', 'LST_qc_good_fraction'):
            values = normalize_quality_fraction(values)
            aggregations[standard_name] = 'mean'
        else:
            aggregations[standard_name] = 'mean'
        out[standard_name] = values

    return out.groupby('Date', as_index=False).agg(aggregations).sort_values('Date')


def contiguous_true_run_lengths(mask: pd.Series) -> pd.Series:
    """Length of each contiguous True run, zero outside runs."""
    values = np.asarray(mask, dtype=bool)
    lengths = np.zeros(len(values), dtype=np.float32)
    idx = 0
    while idx < len(values):
        if not values[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < len(values) and values[end]:
            end += 1
        lengths[idx:end] = float(end - idx)
        idx = end
    return pd.Series(lengths, index=mask.index, dtype=np.float32)


def finalize_lst_quality_columns(frame: pd.DataFrame, lst_missing_before_fill: pd.Series) -> pd.DataFrame:
    """Fill absent LST quality fields and mark values filled by this pipeline."""
    frame = frame.copy()
    missing_flag = lst_missing_before_fill.astype(float).to_numpy(dtype=np.float32)

    if 'LST_is_filled' not in frame.columns:
        frame['LST_is_filled'] = 0.0
    frame['LST_is_filled'] = pd.to_numeric(frame['LST_is_filled'], errors='coerce').fillna(0.0)
    frame['LST_is_filled'] = np.maximum(frame['LST_is_filled'].to_numpy(dtype=np.float32), missing_flag)

    if 'LST_gap_length' not in frame.columns:
        frame['LST_gap_length'] = 0.0
    frame['LST_gap_length'] = pd.to_numeric(frame['LST_gap_length'], errors='coerce').fillna(0.0).clip(lower=0.0)
    pipeline_gap_lengths = contiguous_true_run_lengths(lst_missing_before_fill)
    frame.loc[lst_missing_before_fill, 'LST_gap_length'] = np.maximum(
        frame.loc[lst_missing_before_fill, 'LST_gap_length'].to_numpy(dtype=np.float32),
        pipeline_gap_lengths.loc[lst_missing_before_fill].to_numpy(dtype=np.float32),
    )

    for column in ('LST_valid_pixel_fraction', 'LST_qc_good_fraction'):
        if column not in frame.columns:
            frame[column] = 1.0
        frame[column] = normalize_quality_fraction(frame[column]).fillna(1.0).clip(0.0, 1.0)

    if 'LST_spatial_std_C' not in frame.columns:
        frame['LST_spatial_std_C'] = 0.0
    frame['LST_spatial_std_C'] = (
        pd.to_numeric(frame['LST_spatial_std_C'], errors='coerce')
        .fillna(0.0)
        .clip(lower=0.0, upper=20.0)
    )
    return frame


def compute_lst_quality_factor(frame: pd.DataFrame) -> pd.Series:
    """Return a 0-1 trust factor for LST-derived surface labels."""
    factor = pd.Series(1.0, index=frame.index, dtype=np.float64)

    if 'LST_imputed_by_freezing_rule' in frame.columns:
        freezing_rule = pd.to_numeric(frame['LST_imputed_by_freezing_rule'], errors='coerce').fillna(0.0)
        # A freezing-air rule is a physical fallback, not a direct skin-temperature
        # retrieval. Keep it useful as a weak anchor but do not let it dominate.
        factor *= np.where(freezing_rule > 0.5, 0.15, 1.0)

    if 'LST_is_filled' in frame.columns:
        is_filled = pd.to_numeric(frame['LST_is_filled'], errors='coerce').fillna(0.0)
        factor *= np.where(is_filled > 0.5, 0.45, 1.0)

    if 'LST_gap_length' in frame.columns:
        gap_days = pd.to_numeric(frame['LST_gap_length'], errors='coerce').fillna(0.0).clip(lower=0.0)
        factor *= np.clip(1.0 / (1.0 + gap_days / 5.0), 0.25, 1.0)

    if 'LST_valid_pixel_fraction' in frame.columns:
        valid_fraction = normalize_quality_fraction(frame['LST_valid_pixel_fraction']).fillna(1.0)
        factor *= valid_fraction.clip(0.15, 1.0)

    if 'LST_qc_good_fraction' in frame.columns:
        qc_fraction = normalize_quality_fraction(frame['LST_qc_good_fraction']).fillna(1.0)
        factor *= (0.25 + 0.75 * qc_fraction.clip(0.0, 1.0)).clip(0.25, 1.0)

    if 'LST_spatial_std_C' in frame.columns:
        spatial_std = pd.to_numeric(frame['LST_spatial_std_C'], errors='coerce').fillna(0.0).clip(lower=0.0)
        std_excess = (spatial_std - 1.5).clip(lower=0.0)
        factor *= np.clip(1.0 / (1.0 + std_excess / 1.5), 0.25, 1.0)

    return factor.clip(0.08, 1.0).astype(np.float32)


def add_past_forcing_memory_features(frame: pd.DataFrame, include_current_day: bool = False) -> pd.DataFrame:
    """Add leakage-safe historical forcing features using past-only windows."""
    out = frame.sort_values('Date').copy()

    def history_source(column: str) -> pd.Series:
        source = out[column] if include_current_day else out[column].shift(1)
        return source.fillna(out[column])

    def past_mean(column: str, window: int) -> pd.Series:
        return history_source(column).rolling(window=window, min_periods=1).mean()

    def past_sum(column: str, window: int) -> pd.Series:
        return history_source(column).rolling(window=window, min_periods=1).sum()

    out['air_temp_mean_7d'] = past_mean('T_air_C', 7)
    out['air_temp_mean_30d'] = past_mean('T_air_C', 30)
    out['shortwave_sum_7d'] = past_sum('Solar_W_m2', 7)
    out['shortwave_sum_30d'] = past_sum('Solar_W_m2', 30)
    out['wind_mean_7d'] = past_mean('wind_speed_m_per_s', 7)
    out['lst_mean_7d'] = past_mean('LST_surface_C', 7)
    heating_degree = (history_source('T_air_C') - 4.0).clip(lower=0.0)
    out['heating_degree_days_30d'] = heating_degree.rolling(window=30, min_periods=1).sum()
    return out


def empty_profile_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'Depth_m', 'Temperature_C', 'time_norm'])


def has_profile_observations(profile_obs_data) -> bool:
    if profile_obs_data is None:
        return False
    if isinstance(profile_obs_data, (str, Path)):
        return True
    return not profile_obs_data.empty


def has_bottom_temperature_observations(df: pd.DataFrame) -> bool:
    return 'BottomTemp_C' in df.columns and not pd.to_numeric(df['BottomTemp_C'], errors='coerce').isna().all()


def load_training_frame(
    era5_path,
    lst_path,
    data_fill_mode: str = 'reconstruction',
    apply_freezing_lst_rule: bool | None = None,
    apply_bottom_4c_rule: bool | None = None,
):
    data_fill_mode = normalize_data_fill_mode(data_fill_mode)
    if apply_freezing_lst_rule is None:
        apply_freezing_lst_rule = data_fill_mode == 'reconstruction'
    if apply_bottom_4c_rule is None:
        apply_bottom_4c_rule = data_fill_mode == 'reconstruction'
    era5_path = Path(era5_path)
    lst_path = Path(lst_path)

    if not era5_path.exists():
        raise FileNotFoundError(f'ERA5 file not found: {era5_path}')
    if not lst_path.exists():
        raise FileNotFoundError(f'LST file not found: {lst_path}')

    era5 = pd.read_csv(era5_path)
    era5['Date'] = pd.to_datetime(era5['Date'])
    era5 = era5.sort_values('Date').copy()

    lst = pd.read_csv(lst_path)
    lst['Date'] = pd.to_datetime(lst['Date'])
    lst = lst.sort_values('Date').copy()

    lst_daily = prepare_lst_daily_frame(lst)

    merged = era5.merge(lst_daily, on='Date', how='left')
    merged = merged.sort_values('Date').copy()
    lst_missing_before_fill = merged['LST_surface_K'].isna()
    merged = finalize_lst_quality_columns(merged, lst_missing_before_fill)
    merged['day_index'] = np.arange(len(merged), dtype=np.float32)
    merged['full_doy'] = merged['Date'].dt.dayofyear.astype(np.float32)
    merged['month'] = merged['Date'].dt.month
    doy_angle = 2.0 * np.pi * ((merged['full_doy'].to_numpy(dtype=np.float32) - 1.0) / 365.25)
    merged['doy_sin'] = np.sin(doy_angle).astype(np.float32)
    merged['doy_cos'] = np.cos(doy_angle).astype(np.float32)
    merged['seconds_since_start'] = (
        (merged['Date'] - merged['Date'].iloc[0]).dt.total_seconds().astype(np.float32)
    )
    total_duration_seconds = float(max(merged['seconds_since_start'].max(), SECONDS_PER_DAY))
    merged['time_norm'] = merged['seconds_since_start'] / total_duration_seconds

    merged['T_air_C'] = pick_numeric_series(merged, ['t2m_C', 'air_temp_C', 'T_air_C'])
    air_for_lst_fallback = fill_numeric_series(merged['T_air_C'], data_fill_mode, default=np.nan)
    merged['LST_surface_K'] = fill_numeric_series(merged['LST_surface_K'], data_fill_mode, default=np.nan)
    merged['LST_surface_K'] = merged['LST_surface_K'].fillna(air_for_lst_fallback + 273.15)
    merged['LST_surface_K'] = merged['LST_surface_K'].fillna(273.15)
    merged['LST_surface_C'] = merged['LST_surface_K'] - 273.15
    merged['LST_quality_factor'] = compute_lst_quality_factor(merged)
    merged['BottomTemp_C'] = pick_numeric_series(merged, ['lblt_C', 'bottom_temp_C', 'BottomTemp_C'])
    merged['Solar_J_m2'] = pick_numeric_series(merged, ['Is_J_per_m2', 'solar_J_m2'])
    merged['Solar_W_m2'] = merged['Solar_J_m2'] / SECONDS_PER_DAY
    if merged['Solar_W_m2'].isna().all():
        merged['Solar_W_m2'] = pick_numeric_series(merged, ['Solar_W_m2', 'ssrd_W_per_m2', 'shortwave_W_m2', 'shortwave'])
        merged['Solar_J_m2'] = merged['Solar_W_m2'] * SECONDS_PER_DAY
    merged['MixedLayerDepth_m'] = pick_numeric_series(merged, ['lmld_m', 'mixed_layer_depth_m', 'MixedLayerDepth_m'])
    wind_u = pick_numeric_series(merged, ['u10_m_per_s', 'u10', 'u10m'])
    wind_v = pick_numeric_series(merged, ['v10_m_per_s', 'v10', 'v10m'])
    if not wind_u.isna().all() and not wind_v.isna().all():
        merged['wind_speed_m_per_s'] = np.sqrt(wind_u ** 2 + wind_v ** 2)
    else:
        merged['wind_speed_m_per_s'] = pick_numeric_series(
            merged,
            ['wind_norm_m_per_s', 'wind_speed_m_per_s', 'wind_speed', 'u10_norm_m_per_s'],
            default=1.0,
        )

    merged['dewpoint_C'] = pick_numeric_series(merged, ['d2m_C', 'dewpoint_C', 'td_C'])
    if merged['dewpoint_C'].isna().all():
        dewpoint_k = pick_numeric_series(merged, ['d2m_K', 'dewpoint_K', 'td_K'])
        if not dewpoint_k.isna().all():
            merged['dewpoint_C'] = dewpoint_k - 273.15

    rh_series = pick_numeric_series(
        merged,
        ['rh', 'rh_frac', 'relative_humidity', 'relative_humidity_frac', 'rh_percent', 'relative_humidity_percent'],
    )
    if not rh_series.isna().all():
        if rh_series.max(skipna=True) > 1.5:
            rh_series = rh_series / 100.0
        merged['relative_humidity'] = rh_series
    elif not merged['dewpoint_C'].isna().all():
        es_air = saturation_vapor_pressure_np(merged['T_air_C'])
        ea_air = saturation_vapor_pressure_np(merged['dewpoint_C'])
        merged['relative_humidity'] = ea_air / es_air.clip(lower=1.0)
    else:
        merged['relative_humidity'] = 0.75

    cloud_fraction = pick_numeric_series(
        merged,
        ['tcc', 'cloud_cover', 'cloud_fraction', 'cloud_fraction_frac', 'cloud_fraction_percent'],
    )
    if not cloud_fraction.isna().all():
        if cloud_fraction.max(skipna=True) > 1.5:
            cloud_fraction = cloud_fraction / 100.0
        merged['cloud_fraction'] = cloud_fraction
    else:
        merged['cloud_fraction'] = 0.5

    pressure_pa = pick_numeric_series(merged, ['sp_Pa', 'surface_pressure_Pa', 'msl_Pa', 'pressure_Pa'])
    if pressure_pa.isna().all():
        pressure_hpa = pick_numeric_series(merged, ['sp_hPa', 'surface_pressure_hPa', 'msl_hPa', 'pressure_hPa'])
        if not pressure_hpa.isna().all():
            pressure_pa = pressure_hpa * 100.0
    merged['surface_pressure_Pa'] = pressure_pa if not pressure_pa.isna().all() else 101325.0
    merged['Longwave_W_m2'] = pick_numeric_series(
        merged,
        ['Longwave_W_m2', 'longwave_W_m2', 'longwave', 'strd_W_per_m2', 'downwelling_longwave_W_m2'],
        default=0.0,
    )
    merged['latent_heat_upward_W_m2'] = pick_numeric_series(
        merged,
        ['latent_heat_upward_W_m2', 'latent_heat_upward_W_per_m2', 'latent_heat_W_m2', 'latent_heat', 'slhf_W_per_m2_raw'],
        default=0.0,
    )
    merged['sensible_heat_upward_W_m2'] = pick_numeric_series(
        merged,
        ['sensible_heat_upward_W_m2', 'sensible_heat_upward_W_per_m2', 'sensible_heat_W_m2', 'sensible_heat', 'sshf_W_per_m2_raw'],
        default=0.0,
    )
    merged['Secchi_m'] = pick_numeric_series(
        merged,
        ['Secchi_m', 'secchi_m', 'secchi_depth_m', 'SecchiDepth_m'],
        default=0.0,
    )
    raw_water_level = pick_numeric_series(
        merged,
        [
            'WaterLevel_m',
            'water_level_m',
            'LakeLevel_m',
            'lake_level_m',
            'water_surface_elevation_m',
            'stage_m',
        ],
    )
    raw_water_level_anomaly = pick_numeric_series(
        merged,
        ['water_level_anomaly', 'WaterLevelAnomaly_m', 'lake_level_anomaly_m', 'stage_anomaly_m'],
    )
    raw_light_extinction_kd = pick_numeric_series(
        merged,
        [
            'LightExtinctionKd_m_inv',
            'light_extinction_kd',
            'Kd_m_inv',
            'kd_m_inv',
            'extinction_coef',
            'shortwave_attenuation_coef',
        ],
    )
    raw_effective_fetch = pick_numeric_series(
        merged,
        ['EffectiveFetch_m', 'effective_fetch_m', 'effective_fetch', 'fetch_m', 'wind_fetch_m'],
    )
    raw_inflow = pick_numeric_series(
        merged,
        ['Inflow_m3_s', 'inflow_m3_s', 'river_inflow_m3_s', 'qin_m3_s'],
    )
    raw_outflow = pick_numeric_series(
        merged,
        ['Outflow_m3_s', 'outflow_m3_s', 'river_outflow_m3_s', 'qout_m3_s'],
    )
    raw_net_inflow = pick_numeric_series(
        merged,
        ['NetInflow_m3_s', 'net_inflow_m3_s', 'net_inflow', 'inflow_minus_outflow_m3_s', 'net_flow_m3_s'],
    )

    if merged['T_air_C'].isna().all():
        merged['T_air_C'] = merged['LST_surface_C']
    else:
        merged['T_air_C'] = fill_numeric_series(merged['T_air_C'], data_fill_mode, default=np.nan)
        merged['T_air_C'] = merged['T_air_C'].fillna(merged['LST_surface_C'])

    merged['wind_speed_m_per_s'] = (
        fill_numeric_series(merged['wind_speed_m_per_s'], data_fill_mode, default=1.0, lower=0.1)
    )

    merged['LST_imputed_by_freezing_rule'] = 0.0
    if apply_freezing_lst_rule:
        is_freezing = merged['T_air_C'] < 0.0
        merged.loc[is_freezing, 'LST_surface_C'] = 0.0
        merged.loc[is_freezing, 'LST_imputed_by_freezing_rule'] = 1.0

    winter_months = merged['month'].isin([12, 1, 2])
    bottom_unreasonable = (merged['BottomTemp_C'] < 0.5) | (merged['BottomTemp_C'] > 10.0)
    cold_winter_context = (
        winter_months
        & (
            (pd.to_numeric(merged['T_air_C'], errors='coerce') <= 5.0)
            | (pd.to_numeric(merged['LST_surface_C'], errors='coerce') <= 6.0)
        )
    )
    merged['BottomTemp_imputed_by_4C_rule'] = 0.0
    if apply_bottom_4c_rule:
        bottom_4c_mask = cold_winter_context & bottom_unreasonable
        merged.loc[bottom_4c_mask, 'BottomTemp_C'] = 4.0
        merged.loc[bottom_4c_mask, 'BottomTemp_imputed_by_4C_rule'] = 1.0
    merged['BottomTemp_C'] = fill_numeric_series(merged['BottomTemp_C'], data_fill_mode, default=4.0)

    merged.loc[merged['MixedLayerDepth_m'] < 0, 'MixedLayerDepth_m'] = 0.0
    merged['MixedLayerDepth_m'] = merged['MixedLayerDepth_m'].fillna(0.0)
    merged['Solar_J_m2'] = fill_numeric_series(merged['Solar_J_m2'], data_fill_mode, default=0.0, lower=0.0)
    merged['Solar_W_m2'] = fill_numeric_series(merged['Solar_W_m2'], data_fill_mode, default=0.0, lower=0.0)
    merged['relative_humidity'] = (
        fill_numeric_series(merged['relative_humidity'], data_fill_mode, default=0.75, lower=0.2, upper=1.0)
    )
    merged['cloud_fraction'] = (
        fill_numeric_series(merged['cloud_fraction'], data_fill_mode, default=0.5, lower=0.0, upper=1.0)
    )
    merged['surface_pressure_Pa'] = (
        fill_numeric_series(merged['surface_pressure_Pa'], data_fill_mode, default=101325.0, lower=80000.0, upper=110000.0)
    )
    merged['Longwave_W_m2'] = (
        fill_numeric_series(merged['Longwave_W_m2'], data_fill_mode, default=0.0, lower=0.0, upper=700.0)
    )
    merged['latent_heat_upward_W_m2'] = (
        fill_numeric_series(merged['latent_heat_upward_W_m2'], data_fill_mode, default=0.0, lower=-500.0, upper=500.0)
    )
    merged['sensible_heat_upward_W_m2'] = (
        fill_numeric_series(merged['sensible_heat_upward_W_m2'], data_fill_mode, default=0.0, lower=-500.0, upper=500.0)
    )
    merged['Secchi_m'] = (
        fill_numeric_series(merged['Secchi_m'], data_fill_mode, default=0.0, lower=0.0, upper=50.0)
    )

    metadata = infer_metadata(merged, lst, era5_path, lst_path)
    default_secchi_m = float(metadata.get('secchi_m', 0.0))
    if np.isfinite(default_secchi_m) and default_secchi_m > 0.0 and merged['Secchi_m'].max(skipna=True) <= 0.0:
        merged['Secchi_m'] = default_secchi_m

    def filled_optional_series(values, default=0.0, lower=None, upper=None):
        values = pd.to_numeric(values, errors='coerce')
        if values.isna().all():
            values = pd.Series(float(default), index=merged.index, dtype=np.float32)
        return fill_numeric_series(values, data_fill_mode, default=default, lower=lower, upper=upper)

    water_level_default = float(
        metadata.get('water_level_m', metadata.get('mean_water_level_m', metadata.get('elevation_m', 0.0)))
    )
    if not raw_water_level_anomaly.isna().all():
        merged['water_level_anomaly'] = filled_optional_series(
            raw_water_level_anomaly,
            default=0.0,
            lower=-50.0,
            upper=50.0,
        )
    elif not raw_water_level.isna().all():
        water_level = filled_optional_series(raw_water_level, default=water_level_default)
        water_level_reference = float(np.nanmedian(water_level.to_numpy(dtype=np.float64)))
        merged['water_level_anomaly'] = (water_level - water_level_reference).clip(-50.0, 50.0).astype(np.float32)
    else:
        merged['water_level_anomaly'] = 0.0

    default_kd = float(metadata.get('light_extinction_kd', np.nan))
    if raw_light_extinction_kd.isna().all():
        if np.isfinite(default_kd) and default_kd > 0.0:
            light_extinction_kd = pd.Series(default_kd, index=merged.index, dtype=np.float32)
        else:
            secchi_for_kd = pd.to_numeric(merged['Secchi_m'], errors='coerce').replace(0.0, np.nan)
            light_extinction_kd = 1.7 / secchi_for_kd
    else:
        light_extinction_kd = raw_light_extinction_kd
    kd_default_for_fill = float(default_kd) if np.isfinite(default_kd) and default_kd > 0.0 else 0.0
    merged['light_extinction_kd'] = filled_optional_series(
        light_extinction_kd,
        default=kd_default_for_fill,
        lower=0.0,
        upper=5.0,
    )

    default_fetch_m = float(metadata.get('fetch_m', metadata.get('effective_fetch_m', 0.0)))
    merged['effective_fetch'] = filled_optional_series(
        raw_effective_fetch,
        default=default_fetch_m if np.isfinite(default_fetch_m) else 0.0,
        lower=0.0,
        upper=200000.0,
    )

    if not raw_net_inflow.isna().all():
        net_inflow = raw_net_inflow
    elif not raw_inflow.isna().all() or not raw_outflow.isna().all():
        net_inflow = raw_inflow.fillna(0.0) - raw_outflow.fillna(0.0)
    else:
        net_inflow = pd.Series(0.0, index=merged.index, dtype=np.float32)
    merged['net_inflow'] = filled_optional_series(net_inflow, default=0.0, lower=-10000.0, upper=10000.0)

    merged['data_fill_mode'] = data_fill_mode
    merged = add_past_forcing_memory_features(
        merged,
        include_current_day=(data_fill_mode == 'reconstruction'),
    )

    required = [
        'LST_surface_C',
        'Solar_J_m2',
        'Solar_W_m2',
        'MixedLayerDepth_m',
        'T_air_C',
        'wind_speed_m_per_s',
        'relative_humidity',
        'cloud_fraction',
        'surface_pressure_Pa',
        'Longwave_W_m2',
        'latent_heat_upward_W_m2',
        'sensible_heat_upward_W_m2',
        'Secchi_m',
        *OPTIONAL_HYDRO_FEATURE_COLUMNS,
        'time_norm',
        *HISTORY_FORCING_COLUMNS,
    ]
    if merged[required].isna().any().any():
        raise ValueError('Input data still contains missing values after preprocessing.')

    metadata['time_scale_seconds'] = total_duration_seconds
    metadata['start_date'] = merged['Date'].iloc[0]
    metadata_depth = metadata.get('max_depth_m', 20.0)
    if not np.isfinite(float(metadata_depth)) or float(metadata_depth) <= 0.0:
        metadata_depth = 20.0
    static_feature_values = metadata_static_features(metadata, max_depth=float(metadata_depth))
    metadata['max_depth_norm'] = static_feature_values['max_depth_norm']
    metadata['mean_depth_norm'] = static_feature_values['mean_depth_norm']
    metadata['log_area'] = static_feature_values['log_area']
    metadata['latitude_norm'] = static_feature_values['latitude']
    metadata['longitude_norm'] = static_feature_values['longitude']
    metadata['volume_norm'] = static_feature_values['volume_norm']
    metadata['elevation_norm'] = static_feature_values['elevation_norm']
    metadata['light_extinction_norm'] = static_feature_values['light_extinction_norm']
    metadata['fetch_norm'] = static_feature_values['fetch_norm']
    metadata['wind_exposure_norm'] = static_feature_values['wind_exposure_norm']
    metadata['basin_shape_norm'] = static_feature_values['basin_shape_norm']
    return merged, metadata


def load_optional_profile_observations(obs_source, start_date, time_scale_seconds, max_depth):
    if obs_source is None:
        return empty_profile_observation_frame()

    if isinstance(obs_source, pd.DataFrame):
        obs = obs_source.copy()
    else:
        obs_path = Path(obs_source)
        if not obs_path.exists():
            raise FileNotFoundError(f'Profile observation file not found: {obs_path}')
        obs = pd.read_csv(obs_path)
    date_col = first_existing_column(obs, ['Date', 'date', 'Datetime', 'datetime', 'Timestamp', 'timestamp'])
    depth_col = first_existing_column(obs, ['Depth_m', 'depth_m', 'Depth', 'depth'])
    temp_col = first_existing_column(obs, ['Temperature_C', 'temperature_C', 'Temp_C', 'temp_c', 'Temperature', 'temp'])

    if date_col and depth_col and temp_col:
        obs = obs[[date_col, depth_col, temp_col]].rename(
            columns={date_col: 'Date', depth_col: 'Depth_m', temp_col: 'Temperature_C'}
        )
    else:
        wide_temp_columns = []
        for column in obs.columns:
            column_str = str(column)
            depth_match = re.fullmatch(r'(?:Temp(?:erature)?_?)?(\d+(?:\.\d+)?)m', column_str, flags=re.IGNORECASE)
            if depth_match:
                wide_temp_columns.append((column, float(depth_match.group(1))))

        if not date_col or not wide_temp_columns:
            raise ValueError(
                'Profile observation CSV must contain either long-format date/depth/temperature columns '
                '(for example Date/Depth_m/Temperature_C) or wide-format columns like Temp_0m, Temp_1m, ... .'
            )

        rename_map = {column: f'__depth_{depth:g}m' for column, depth in wide_temp_columns}
        obs = obs[[date_col] + [column for column, _ in wide_temp_columns]].rename(columns={date_col: 'Date', **rename_map})
        obs = obs.melt(id_vars=['Date'], var_name='DepthLabel', value_name='Temperature_C')
        obs['Depth_m'] = obs['DepthLabel'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        obs = obs.drop(columns=['DepthLabel'])

    obs['Date'] = pd.to_datetime(obs['Date'])
    obs['Depth_m'] = pd.to_numeric(obs['Depth_m'], errors='coerce')
    obs['Temperature_C'] = pd.to_numeric(obs['Temperature_C'], errors='coerce')
    obs = obs.dropna().copy()
    obs['Depth_m'] = obs['Depth_m'].clip(lower=0.0, upper=max_depth)
    obs['time_norm'] = ((obs['Date'] - pd.Timestamp(start_date)).dt.total_seconds() / time_scale_seconds).clip(0.0, 1.0)
    return obs.sort_values(['Date', 'Depth_m']).reset_index(drop=True)


def split_profile_observations(profile_obs, split_mode='time_blocked'):
    split_frames = {role: empty_profile_observation_frame() for role in PROFILE_SPLIT_ROLES}
    split_summary = {
        role: {'rows': 0, 'depth_count': 0, 'date_count': 0}
        for role in PROFILE_SPLIT_ROLES
    }

    if not has_profile_observations(profile_obs):
        return split_frames, {'mode': split_mode, 'summary': split_summary}

    profile_obs = profile_obs.copy()

    def assign_ordered_dates_to_roles(unique_dates):
        unique_dates = pd.Index(sorted(unique_dates))
        n_dates = len(unique_dates)
        if n_dates == 0:
            return {}
        if n_dates < len(PROFILE_SPLIT_ROLES):
            return {pd.Timestamp(date_value): 'train' for date_value in unique_dates}

        boundaries = {}
        start_idx = 0
        for idx, role in enumerate(PROFILE_SPLIT_ROLES):
            if idx == len(PROFILE_SPLIT_ROLES) - 1:
                end_idx = n_dates
            else:
                fraction = TIME_BLOCK_SPLIT_FRACTIONS[role]
                end_idx = start_idx + int(round(n_dates * fraction))
                remaining_roles = len(PROFILE_SPLIT_ROLES) - idx - 1
                max_end = n_dates - remaining_roles
                end_idx = int(np.clip(end_idx, start_idx + 1, max_end))
            boundaries[role] = (start_idx, end_idx)
            start_idx = end_idx

        date_to_role = {}
        for role, (start_idx, end_idx) in boundaries.items():
            for date_value in unique_dates[start_idx:end_idx]:
                date_to_role[pd.Timestamp(date_value)] = role
        return date_to_role

    if split_mode == 'none':
        split_frames['train'] = profile_obs.copy()
    elif split_mode == 'depth_interleaved':
        rounded_depths = np.round(profile_obs['Depth_m'].to_numpy(dtype=np.float64), 6)
        profile_obs['__depth_key'] = rounded_depths
        unique_depths = np.unique(rounded_depths)
        depth_to_role = {
            depth_value: DEFAULT_PROFILE_SPLIT_PATTERN[idx % len(DEFAULT_PROFILE_SPLIT_PATTERN)]
            for idx, depth_value in enumerate(unique_depths)
        }
        profile_obs['split_role'] = profile_obs['__depth_key'].map(depth_to_role)
        for role in PROFILE_SPLIT_ROLES:
            split_frames[role] = (
                profile_obs[profile_obs['split_role'] == role]
                .drop(columns=['__depth_key', 'split_role'])
                .reset_index(drop=True)
            )
    elif split_mode == 'time_blocked':
        normalized_dates = pd.to_datetime(profile_obs['Date']).dt.normalize()
        unique_dates = pd.Index(sorted(normalized_dates.unique()))
        date_to_role = assign_ordered_dates_to_roles(unique_dates)
        profile_obs['split_role'] = normalized_dates.map(lambda value: date_to_role[pd.Timestamp(value)])
        for role in PROFILE_SPLIT_ROLES:
            split_frames[role] = (
                profile_obs[profile_obs['split_role'] == role]
                .drop(columns=['split_role'])
                .reset_index(drop=True)
            )
    elif split_mode == 'seasonal_blocked':
        normalized_dates = pd.to_datetime(profile_obs['Date']).dt.normalize()
        date_frame = pd.DataFrame({'Date': pd.Index(sorted(normalized_dates.unique()))})

        def seasonal_segment(date_value):
            month = pd.Timestamp(date_value).month
            if month in (1, 2):
                return 'winter_early'
            if 3 <= month <= 5:
                return 'spring'
            if 6 <= month <= 8:
                return 'summer'
            if 9 <= month <= 11:
                return 'autumn'
            return 'winter_late'

        date_frame['seasonal_segment'] = date_frame['Date'].map(seasonal_segment)
        date_to_role = {}
        for segment_name in ('winter_early', 'spring', 'summer', 'autumn', 'winter_late'):
            segment_dates = date_frame.loc[date_frame['seasonal_segment'] == segment_name, 'Date']
            date_to_role.update(assign_ordered_dates_to_roles(segment_dates))

        profile_obs['split_role'] = normalized_dates.map(lambda value: date_to_role.get(pd.Timestamp(value), 'train'))
        for role in PROFILE_SPLIT_ROLES:
            split_frames[role] = (
                profile_obs[profile_obs['split_role'] == role]
                .drop(columns=['split_role'])
                .reset_index(drop=True)
            )
    else:
        raise ValueError(f'Unsupported profile split mode: {split_mode}')

    for role, frame in split_frames.items():
        split_summary[role] = {
            'rows': int(len(frame)),
            'depth_count': int(frame['Depth_m'].nunique()) if not frame.empty else 0,
            'date_count': int(pd.to_datetime(frame['Date']).dt.normalize().nunique()) if not frame.empty else 0,
        }

    return split_frames, {'mode': split_mode, 'summary': split_summary}


def summarize_profile_split_frames(split_frames):
    """Return row/depth/date counts for already-split profile observation frames."""
    split_summary = {}
    for role in PROFILE_SPLIT_ROLES:
        frame = split_frames.get(role, empty_profile_observation_frame())
        split_summary[role] = {
            'rows': int(len(frame)),
            'depth_count': int(frame['Depth_m'].nunique()) if not frame.empty else 0,
            'date_count': int(pd.to_datetime(frame['Date']).dt.normalize().nunique()) if not frame.empty else 0,
        }
    return split_summary


def downsample_profile_train_dates(train_profile_obs, date_fraction=1.0, random_seed=20260511):
    """Keep a deterministic subset of train dates only; val/assim/test remain untouched."""
    if not has_profile_observations(train_profile_obs):
        return train_profile_obs.copy(), {
            'enabled': False,
            'date_fraction': 1.0,
            'rows_before': 0,
            'rows_after': 0,
            'dates_before': 0,
            'dates_after': 0,
        }

    date_fraction = float(date_fraction)
    if date_fraction >= 0.999:
        date_count = int(pd.to_datetime(train_profile_obs['Date']).dt.normalize().nunique())
        return train_profile_obs.copy(), {
            'enabled': False,
            'date_fraction': 1.0,
            'rows_before': int(len(train_profile_obs)),
            'rows_after': int(len(train_profile_obs)),
            'dates_before': date_count,
            'dates_after': date_count,
        }
    if date_fraction <= 0.0:
        raise ValueError('--profile-train-date-fraction must be in (0, 1].')

    normalized_dates = pd.to_datetime(train_profile_obs['Date']).dt.normalize()
    unique_dates = pd.Index(sorted(normalized_dates.unique()))
    keep_count = int(np.clip(round(len(unique_dates) * date_fraction), 1, len(unique_dates)))
    rng = np.random.default_rng(int(random_seed))
    keep_positions = np.sort(rng.choice(len(unique_dates), size=keep_count, replace=False))
    keep_dates = {pd.Timestamp(unique_dates[pos]) for pos in keep_positions}
    downsampled = train_profile_obs[normalized_dates.isin(keep_dates)].copy().reset_index(drop=True)
    return downsampled, {
        'enabled': True,
        'date_fraction': date_fraction,
        'rows_before': int(len(train_profile_obs)),
        'rows_after': int(len(downsampled)),
        'dates_before': int(len(unique_dates)),
        'dates_after': int(len(keep_dates)),
    }


def subset_profile_observations_by_dates(profile_obs_data, dates):
    if not has_profile_observations(profile_obs_data):
        return empty_profile_observation_frame()

    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(min(dates)),
        time_scale_seconds=max((pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))).total_seconds(), SECONDS_PER_DAY),
        max_depth=np.inf,
    )
    date_set = {pd.Timestamp(date_value).normalize() for date_value in pd.to_datetime(list(dates))}
    subset = profile_obs[profile_obs['Date'].dt.normalize().isin(date_set)].copy()
    return subset.reset_index(drop=True)


def build_segment_frame(df: pd.DataFrame, start_idx: int, end_idx: int):
    segment_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    segment_df['day_index'] = np.arange(len(segment_df), dtype=np.float32)
    segment_df['seconds_since_start'] = (
        (segment_df['Date'] - segment_df['Date'].iloc[0]).dt.total_seconds().astype(np.float32)
    )
    total_duration_seconds = float(max(segment_df['seconds_since_start'].max(), SECONDS_PER_DAY))
    segment_df['time_norm'] = segment_df['seconds_since_start'] / total_duration_seconds
    return segment_df, total_duration_seconds


def fit_surface_bulk_correction(df, metadata, max_depth, profile_obs_data=None, max_surface_depth=1.0):
    """Fit a shallow-observation-informed mapping from satellite skin temperature to bulk surface temperature."""
    if not has_profile_observations(profile_obs_data):
        return None, None

    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if profile_obs.empty:
        return None, None

    shallow_obs = profile_obs[profile_obs['Depth_m'] <= max_surface_depth].copy()
    if shallow_obs.empty:
        return None, None

    shallow_daily = (
        shallow_obs.groupby('Date', as_index=False)['Temperature_C']
        .mean()
        .rename(columns={'Temperature_C': 'ObservedSurfaceBulk_C'})
    )
    calibration = df[['Date', 'LST_surface_C', 'T_air_C', 'wind_speed_m_per_s']].merge(
        shallow_daily,
        on='Date',
        how='inner',
    )
    calibration = calibration.dropna().copy()
    if len(calibration) < 10:
        return None, None

    x_lst = calibration['LST_surface_C'].to_numpy(dtype=np.float64)
    x_air = calibration['T_air_C'].to_numpy(dtype=np.float64)
    x_wind = calibration['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
    x_delta = x_lst - x_air
    X = np.column_stack([np.ones(len(calibration)), x_lst, x_air, x_wind, x_delta])
    y = calibration['ObservedSurfaceBulk_C'].to_numpy(dtype=np.float64)

    ridge_alpha = 0.5
    eye = np.eye(X.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + ridge_alpha * eye, X.T @ y)

    full_lst = df['LST_surface_C'].to_numpy(dtype=np.float64)
    full_air = df['T_air_C'].to_numpy(dtype=np.float64)
    full_wind = df['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
    full_delta = full_lst - full_air
    X_full = np.column_stack([np.ones(len(df)), full_lst, full_air, full_wind, full_delta])
    corrected = X_full @ coef
    corrected = np.where(df['T_air_C'].to_numpy(dtype=np.float64) < 0.0, 0.0, corrected)
    corrected = np.clip(corrected, full_lst - 1.5, full_lst + 1.5)
    corrected = np.clip(corrected, -1.0, 35.0)

    raw_rmse = float(np.sqrt(np.mean((x_lst - y) ** 2)))
    fit_rmse = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    diagnostics = {
        'n_matches': int(len(calibration)),
        'raw_rmse': raw_rmse,
        'fit_rmse': fit_rmse,
        'coefficients': coef.tolist(),
    }

    if fit_rmse >= raw_rmse:
        return None, diagnostics

    corrected_series = pd.Series(corrected, index=df.index, dtype=np.float32)
    return corrected_series, diagnostics


def build_observation_dataframe(
    df,
    metadata,
    max_depth,
    profile_obs_data=None,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
):
    corrected_surface_obs = None
    surface_correction_info = None
    if use_surface_bulk_correction:
        corrected_surface_obs, surface_correction_info = fit_surface_bulk_correction(
            df=df,
            metadata=metadata,
            max_depth=max_depth,
            profile_obs_data=profile_obs_data,
            max_surface_depth=1.0,
        )

    base_surface = pd.DataFrame(
        {
            'Date': df['Date'],
            'Depth_m': float(np.clip(surface_obs_depth_m, 0.0, max_depth)),
            'Temperature_C': corrected_surface_obs if corrected_surface_obs is not None else df.get('SurfaceBulkTarget_C', df['LST_surface_C']),
            'time_norm': df['time_norm'],
            'source': 'surface',
        }
    )
    obs_frames = [base_surface]
    if use_bottom_observation:
        if not has_bottom_temperature_observations(df):
            raise ValueError('BottomTemp_C observations are required when use_bottom_observation is enabled.')
        base_bottom = pd.DataFrame(
            {
                'Date': df['Date'],
                'Depth_m': max_depth,
                'Temperature_C': df['BottomTemp_C'],
                'time_norm': df['time_norm'],
                'source': 'bottom',
            }
        )
        obs_frames.append(base_bottom)
    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if not profile_obs.empty:
        profile_obs = profile_obs.copy()
        profile_obs['source'] = 'profile'
        obs_frames.append(profile_obs)

    observations = pd.concat(obs_frames, ignore_index=True)
    observations['obs_weight'] = observations.apply(
        lambda row: compute_observation_weight(row['Depth_m'], row['source'], max_depth),
        axis=1,
    )
    observations = apply_geographic_profile_observation_weights(observations, metadata, max_depth)
    observations['obs_std_scale'] = 1.0
    forcing_columns = [
        'Date',
        'doy_sin',
        'doy_cos',
        'T_air_C',
        'wind_speed_m_per_s',
        'Solar_W_m2',
        'LST_surface_C',
        'Longwave_W_m2',
        'latent_heat_upward_W_m2',
        'sensible_heat_upward_W_m2',
        'Secchi_m',
        'LST_spring_spike_correction_C',
        'LST_quality_factor',
        'LST_imputed_by_freezing_rule',
        'BottomTemp_imputed_by_4C_rule',
        *LST_QUALITY_COLUMNS,
        *HISTORY_FORCING_COLUMNS,
        *STATE_MEMORY_COLUMNS,
        *OPTIONAL_HYDRO_FEATURE_COLUMNS,
    ]
    forcing_columns = [column for column in forcing_columns if column in df.columns]
    observations = observations.merge(df[forcing_columns], on='Date', how='left')
    for column in forcing_columns[1:]:
        observations[column] = observations[column].interpolate(method='linear', limit_direction='both').bfill().ffill()

    surface_mask = observations['source'].eq('surface')
    if any(column in observations.columns for column in [*LST_QUALITY_COLUMNS, 'LST_imputed_by_freezing_rule']):
        lst_quality_factor = compute_lst_quality_factor(observations)
    elif 'LST_quality_factor' in observations.columns:
        lst_quality_factor = pd.to_numeric(observations['LST_quality_factor'], errors='coerce').fillna(1.0).clip(0.08, 1.0)
    else:
        lst_quality_factor = pd.Series(1.0, index=observations.index, dtype=np.float32)
    observations['lst_quality_factor'] = lst_quality_factor.astype(np.float32)
    observations.loc[surface_mask, 'obs_weight'] = (
        observations.loc[surface_mask, 'obs_weight'].astype(float)
        * observations.loc[surface_mask, 'lst_quality_factor'].astype(float)
    )
    observations.loc[surface_mask, 'obs_std_scale'] = (
        observations.loc[surface_mask, 'obs_std_scale'].astype(float)
        / np.sqrt(observations.loc[surface_mask, 'lst_quality_factor'].astype(float).clip(lower=0.08))
    ).clip(1.0, 4.0)

    if 'LST_spring_spike_correction_C' in observations.columns:
        # If the forcing preprocessor had to cool a suspicious spring LST spike,
        # keep the corrected target but reduce its leverage as a surface label.
        correction = observations['LST_spring_spike_correction_C'].fillna(0.0).astype(float)
        spike_factor = 1.0 / (1.0 + np.exp((correction - 0.75) / 0.25))
        spike_factor = np.clip(0.25 + 0.75 * spike_factor, 0.25, 1.0)
        observations.loc[surface_mask, 'obs_weight'] = (
            observations.loc[surface_mask, 'obs_weight'].astype(float)
            * spike_factor[surface_mask]
        )
        observations.loc[surface_mask, 'obs_std_scale'] = (
            observations.loc[surface_mask, 'obs_std_scale'].astype(float)
            / np.sqrt(pd.Series(spike_factor, index=observations.index).loc[surface_mask].clip(lower=0.25))
        ).clip(1.0, 4.0)

    bottom_mask = observations['source'].eq('bottom')
    if 'BottomTemp_imputed_by_4C_rule' in observations.columns and bottom_mask.any():
        bottom_imputed = (
            pd.to_numeric(observations['BottomTemp_imputed_by_4C_rule'], errors='coerce')
            .fillna(0.0)
            .astype(float)
        )
        bottom_factor = pd.Series(1.0, index=observations.index, dtype=np.float64)
        bottom_factor.loc[bottom_imputed > 0.5] = 0.10
        observations.loc[bottom_mask, 'obs_weight'] = (
            observations.loc[bottom_mask, 'obs_weight'].astype(float)
            * bottom_factor.loc[bottom_mask]
        )
        observations.loc[bottom_mask, 'obs_std_scale'] = (
            observations.loc[bottom_mask, 'obs_std_scale'].astype(float)
            / np.sqrt(bottom_factor.loc[bottom_mask].clip(lower=0.10))
        ).clip(1.0, 5.0)

    observations = observations.sort_values(['Date', 'Depth_m']).reset_index(drop=True)
    return observations, surface_correction_info


def build_initial_condition_profile(df, max_depth, n_points=64, mode='uniform_4c'):
    z_ic = np.linspace(0.0, max_depth, n_points, dtype=np.float32)
    surf0_series = df['SurfaceBulkTarget_C'] if 'SurfaceBulkTarget_C' in df.columns else df['LST_surface_C']
    surf0 = float(surf0_series.iloc[0])
    if mode == 'uniform_4c':
        temp_ic = np.full_like(z_ic, DEFAULT_INITIAL_WATER_TEMPERATURE_C, dtype=np.float32)
    elif mode == 'surface_to_uniform_4c':
        temp_ic = surf0 + (DEFAULT_INITIAL_WATER_TEMPERATURE_C - surf0) * (z_ic / max(max_depth, 1e-6))
    elif mode == 'linear_to_bottom_obs':
        if not has_bottom_temperature_observations(df):
            raise ValueError('BottomTemp_C observations are required when initial_condition_mode=linear_to_bottom_obs.')
        bottom0 = float(df['BottomTemp_C'].iloc[0])
        temp_ic = surf0 + (bottom0 - surf0) * (z_ic / max(max_depth, 1e-6))
    else:
        raise ValueError(f'Unsupported initial condition mode: {mode}')
    return z_ic.reshape(-1, 1), temp_ic.reshape(-1, 1)


def compute_observation_weight(depth_m, source, max_depth):
    """Moderately emphasize shallow-layer observations during training."""
    depth_m = float(depth_m)
    if source == 'surface':
        return 4.0
    if source == 'bottom':
        return 1.0
    if depth_m <= 1.0:
        return 3.5
    if depth_m <= 3.0:
        return 2.75
    if depth_m <= 5.0:
        return 1.8
    if depth_m <= min(max_depth, 10.0):
        return 1.2
    if depth_m <= min(max_depth, 15.0):
        return 0.95
    return 0.8


def apply_geographic_profile_observation_weights(observations, metadata, max_depth):
    """Add mild season/depth emphasis for warm deep lakes without touching held-out splits."""
    max_depth = float(max_depth)
    is_warm_deep = is_geographic_warm_deep_lake(metadata, max_depth=max_depth)
    if not is_warm_deep or observations.empty or 'source' not in observations.columns:
        return observations

    profile_mask = observations['source'].eq('profile')
    if not profile_mask.any():
        return observations

    out = observations.copy()
    dates = pd.to_datetime(out['Date'], errors='coerce')
    months = dates.dt.month
    depths = pd.to_numeric(out['Depth_m'], errors='coerce')

    # These are observation-loss weights, not prescribed target shapes. They only
    # tell the trainer to pay closer attention where warm deep lakes showed errors.
    summer_deep = (
        profile_mask
        & months.isin([7, 8, 9])
        & depths.between(max(14.0, 0.45 * max_depth), min(max_depth, 0.70 * max_depth))
    )
    autumn_mid = (
        profile_mask
        & months.isin([10, 11])
        & depths.between(max(8.0, 0.22 * max_depth), min(max_depth, 0.58 * max_depth))
    )
    winter_column = (
        profile_mask
        & months.isin([12, 1, 2])
        & depths.between(0.0, max_depth)
    )
    winter_deep = (
        profile_mask
        & months.isin([12, 1, 2])
        & depths.between(max(20.0, 0.50 * max_depth), max_depth)
    )

    multipliers = pd.Series(1.0, index=out.index, dtype=np.float32)
    multipliers.loc[summer_deep] *= 1.15
    multipliers.loc[autumn_mid] *= 1.20
    multipliers.loc[winter_column] *= 1.05
    multipliers.loc[winter_deep] *= 1.35
    out['obs_weight'] = (
        pd.to_numeric(out['obs_weight'], errors='coerce').fillna(1.0)
        * multipliers
    ).clip(lower=0.05, upper=5.0)
    return out


def depth_dependent_obs_std(source, depth_m, max_depth, base_surface, base_bottom, base_profile):
    """Use mildly depth-dependent observation noise for Kalman updates."""
    depth_m = float(depth_m)
    if source == 'surface':
        return max(0.15, base_surface * 0.75)
    if source == 'bottom':
        return max(0.3, base_bottom * 1.0)

    if depth_m <= 1.0:
        scale = 0.55
    elif depth_m <= 3.0:
        scale = 0.65
    elif depth_m <= 5.0:
        scale = 0.8
    elif depth_m <= min(max_depth, 10.0):
        scale = 0.95
    elif depth_m <= min(max_depth, 15.0):
        scale = 1.15
    else:
        scale = 1.3
    return max(0.2, base_profile * scale)


def build_depth_grid(max_depth, n_depth_points, use_shallow_optimized=True, shallow_focus_depth=5.0, shallow_fraction=0.55):
    """Create a prediction/assimilation depth grid with extra resolution in shallow layers."""
    max_depth = float(max_depth)
    n_depth_points = int(max(2, n_depth_points))
    shallow_focus_depth = float(np.clip(shallow_focus_depth, 0.5, max_depth))
    shallow_fraction = float(np.clip(shallow_fraction, 0.2, 0.9))
    meter_anchors = np.arange(0.0, np.floor(max_depth) + 1.0, 1.0, dtype=np.float32)

    if (not use_shallow_optimized) or shallow_focus_depth >= max_depth or n_depth_points < 8:
        base_grid = np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)
        return np.unique(np.concatenate([base_grid, meter_anchors])).astype(np.float32)

    shallow_points = int(round(n_depth_points * shallow_fraction))
    shallow_points = int(np.clip(shallow_points, 4, n_depth_points - 3))
    deep_points = n_depth_points - shallow_points + 1

    shallow_grid = np.linspace(0.0, shallow_focus_depth, shallow_points, dtype=np.float32)
    deep_grid = np.linspace(shallow_focus_depth, max_depth, deep_points, dtype=np.float32)
    full_grid = np.unique(np.concatenate([shallow_grid, deep_grid]))

    if len(full_grid) < n_depth_points:
        fallback_grid = np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)
        full_grid = np.unique(np.concatenate([full_grid, fallback_grid]))

    full_grid = np.unique(np.concatenate([full_grid, meter_anchors]))
    return np.sort(full_grid.astype(np.float32))
