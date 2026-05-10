# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .lake_metadata import infer_metadata, metadata_static_features

def pick_numeric_series(frame: pd.DataFrame, candidates, default=np.nan):
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors='coerce')
            if not values.isna().all():
                return values
    return pd.Series(default, index=frame.index, dtype=np.float32)


def first_existing_column(frame: pd.DataFrame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


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


def load_training_frame(era5_path, lst_path):
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
    lst['LST_surface_K'] = pd.to_numeric(lst['MOD11A1_061_LST_Day_1km'], errors='coerce')
    lst.loc[lst['LST_surface_K'] <= 0, 'LST_surface_K'] = np.nan

    lst_daily = lst.groupby('Date', as_index=False)['LST_surface_K'].mean().sort_values('Date')

    merged = era5.merge(lst_daily, on='Date', how='left')
    merged = merged.sort_values('Date').copy()
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

    merged['LST_surface_K'] = (
        merged['LST_surface_K']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
    )
    merged['LST_surface_C'] = merged['LST_surface_K'] - 273.15
    merged['BottomTemp_C'] = pick_numeric_series(merged, ['lblt_C', 'bottom_temp_C', 'BottomTemp_C'])
    merged['Solar_J_m2'] = pick_numeric_series(merged, ['Is_J_per_m2', 'solar_J_m2'])
    merged['Solar_W_m2'] = merged['Solar_J_m2'] / SECONDS_PER_DAY
    if merged['Solar_W_m2'].isna().all():
        merged['Solar_W_m2'] = pick_numeric_series(merged, ['Solar_W_m2', 'ssrd_W_per_m2', 'shortwave_W_m2', 'shortwave'])
        merged['Solar_J_m2'] = merged['Solar_W_m2'] * SECONDS_PER_DAY
    merged['MixedLayerDepth_m'] = pick_numeric_series(merged, ['lmld_m', 'mixed_layer_depth_m', 'MixedLayerDepth_m'])
    merged['T_air_C'] = pick_numeric_series(merged, ['t2m_C', 'air_temp_C', 'T_air_C'])

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

    if merged['T_air_C'].isna().all():
        merged['T_air_C'] = merged['LST_surface_C']
    else:
        merged['T_air_C'] = merged['T_air_C'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    merged['wind_speed_m_per_s'] = (
        merged['wind_speed_m_per_s']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=0.1)
    )

    is_freezing = merged['T_air_C'] < 0.0
    merged.loc[is_freezing, 'LST_surface_C'] = 0.0

    winter_months = merged['month'].isin([12, 1, 2])
    bottom_unreasonable = (merged['BottomTemp_C'] < 0.5) | (merged['BottomTemp_C'] > 10.0)
    merged.loc[winter_months & bottom_unreasonable, 'BottomTemp_C'] = 4.0
    merged['BottomTemp_C'] = merged['BottomTemp_C'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    merged.loc[merged['MixedLayerDepth_m'] < 0, 'MixedLayerDepth_m'] = 0.0
    merged['MixedLayerDepth_m'] = merged['MixedLayerDepth_m'].fillna(0.0)
    merged['Solar_J_m2'] = merged['Solar_J_m2'].interpolate(method='linear', limit_direction='both').bfill().ffill()
    merged['Solar_W_m2'] = merged['Solar_W_m2'].interpolate(method='linear', limit_direction='both').bfill().ffill()
    merged['relative_humidity'] = (
        merged['relative_humidity']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(0.2, 1.0)
    )
    merged['cloud_fraction'] = (
        merged['cloud_fraction']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(0.0, 1.0)
    )
    merged['surface_pressure_Pa'] = (
        merged['surface_pressure_Pa']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=80000.0, upper=110000.0)
    )
    merged['Longwave_W_m2'] = (
        merged['Longwave_W_m2']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=0.0, upper=700.0)
    )
    merged['latent_heat_upward_W_m2'] = (
        merged['latent_heat_upward_W_m2']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=-500.0, upper=500.0)
    )
    merged['sensible_heat_upward_W_m2'] = (
        merged['sensible_heat_upward_W_m2']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=-500.0, upper=500.0)
    )
    merged['Secchi_m'] = (
        merged['Secchi_m']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=0.0, upper=50.0)
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
        'time_norm',
    ]
    if merged[required].isna().any().any():
        raise ValueError('Input data still contains missing values after preprocessing.')

    metadata = infer_metadata(merged, lst, era5_path, lst_path)
    metadata['time_scale_seconds'] = total_duration_seconds
    metadata['start_date'] = merged['Date'].iloc[0]
    metadata.update(metadata_static_features(metadata, max_depth=20.0))
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
        n_dates = len(unique_dates)
        if n_dates == 0:
            return split_frames, {'mode': split_mode, 'summary': split_summary}

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
            role_dates = unique_dates[start_idx:end_idx]
            for date_value in role_dates:
                date_to_role[pd.Timestamp(date_value)] = role

        profile_obs['split_role'] = normalized_dates.map(lambda value: date_to_role[pd.Timestamp(value)])
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
    ]
    observations = observations.merge(df[forcing_columns], on='Date', how='left')
    for column in forcing_columns[1:]:
        observations[column] = observations[column].interpolate(method='linear', limit_direction='both').bfill().ffill()
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

    if (not use_shallow_optimized) or shallow_focus_depth >= max_depth or n_depth_points < 8:
        return np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)

    shallow_points = int(round(n_depth_points * shallow_fraction))
    shallow_points = int(np.clip(shallow_points, 4, n_depth_points - 3))
    deep_points = n_depth_points - shallow_points + 1

    shallow_grid = np.linspace(0.0, shallow_focus_depth, shallow_points, dtype=np.float32)
    deep_grid = np.linspace(shallow_focus_depth, max_depth, deep_points, dtype=np.float32)
    full_grid = np.unique(np.concatenate([shallow_grid, deep_grid]))

    if len(full_grid) < n_depth_points:
        fallback_grid = np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)
        full_grid = np.unique(np.concatenate([full_grid, fallback_grid]))

    return np.sort(full_grid.astype(np.float32))
