# Lightweight multi-lake global-adapter trainer.
#
# This module intentionally keeps the first G-series experiment simple:
# train the shared trunk + lake adapter on profile/surface observations from
# multiple lakes, with row-level forcing and row-level lake attributes.  The
# heavier PDE/SEB loss stack remains in the single-lake trainer until its
# batches are made lake-aware end-to-end.
from .common import *
import json
import torch.nn.functional as F
from .checkpoint import save_model_checkpoint_bundle
from .constants import (
    HISTORY_FORCING_COLUMNS,
    OPTIONAL_HYDRO_FEATURE_COLUMNS,
    PINN_FETCH_REFERENCE_M,
    PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
    PINN_INFLOW_REFERENCE_M3_S,
    PINN_INPUT_DIM,
    PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
    PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
    PINN_MAX_LONGWAVE_REFERENCE_W_M2,
    PINN_MAX_SECCHI_REFERENCE_M,
    PINN_MAX_SHORTWAVE_REFERENCE_W_M2,
    PINN_MAX_TEMPERATURE_REFERENCE_C,
    PINN_MAX_WIND_REFERENCE_M_PER_S,
    PINN_SHORTWAVE_SUM_30D_REFERENCE,
    PINN_SHORTWAVE_SUM_7D_REFERENCE,
    PINN_WATER_LEVEL_ANOMALY_REFERENCE_M,
    STATE_MEMORY_COLUMNS,
)
from .data_io import (
    build_observation_dataframe,
    load_optional_profile_observations,
    load_training_frame,
    split_profile_observations,
    summarize_profile_split_frames,
)
from .lake_metadata import metadata_static_features
from .model import create_lake_model
from .physics import water_density_torch
from .train import build_causal_previous_state_memory


STATIC_FEATURE_COLUMNS = (
    'static_max_depth_norm',
    'static_mean_depth_norm',
    'static_log_area',
    'static_latitude',
    'static_longitude',
    'static_volume_norm',
    'static_elevation_norm',
    'static_light_extinction_norm',
    'static_fetch_norm',
    'static_wind_exposure_norm',
    'static_basin_shape_norm',
)


DEFAULT_PHYSICS_REG_CONFIG = {
    'enabled': True,
    'depth_points': 36,
    'weight': 0.05,
    'bias_weight': 0.05,
    'temperature_low_c': -1.0,
    'temperature_strong_low_c': -3.0,
    'temperature_extreme_low_c': -5.0,
    'temperature_lower_c': -1.0,
    'temperature_upper_c': 36.0,
    'temperature_extreme_lower_c': -5.0,
    'temperature_extreme_upper_c': 38.0,
    'surface_jump_threshold_c_per_day': 3.0,
    'surface_band_depth_m': 3.0,
    'surface_band_uniformity_threshold_c': 1.0,
    'column_jump_surface_threshold_c_per_day': 2.5,
    'column_jump_deep_threshold_c_per_day': 0.8,
    'deep_slow_depth_fraction': 0.55,
    'deep_slow_jump_threshold_c_per_day': 0.55,
    'vertical_gradient_threshold_c_per_m': 4.0,
    'density_inversion_drop_kgm3': 0.02,
    'density_unstable_layer_fraction': 0.12,
    'component_weights': {
        'temperature_range': 0.50,
        'surface_jump': 0.35,
        'surface_band_uniformity': 0.45,
        'column_jump': 0.45,
        'deep_slow_change': 0.50,
        'vertical_gradient': 0.30,
        'density_stability': 0.50,
    },
}


def _read_manifest(path):
    path = Path(path)
    with path.open('r', encoding='utf-8-sig') as handle:
        manifest = json.load(handle)
    lakes = manifest.get('lakes', [])
    if not lakes:
        raise ValueError(f'Manifest has no lakes: {path}')
    return manifest


def _as_float_series(frame, column, default=0.0):
    if column in frame.columns:
        values = pd.to_numeric(frame[column], errors='coerce')
    else:
        values = pd.Series(float(default), index=frame.index, dtype=np.float32)
    return values.fillna(float(default)).to_numpy(dtype=np.float32).reshape(-1, 1)


def _add_static_features(observations, metadata, max_depth):
    static = metadata_static_features(metadata, max_depth=max_depth)
    enriched = observations.copy()
    enriched['max_depth_m'] = float(max_depth)
    enriched['static_max_depth_norm'] = static['max_depth_norm']
    enriched['static_mean_depth_norm'] = static['mean_depth_norm']
    enriched['static_log_area'] = static['log_area']
    enriched['static_latitude'] = static['latitude']
    enriched['static_longitude'] = static['longitude']
    enriched['static_volume_norm'] = static['volume_norm']
    enriched['static_elevation_norm'] = static['elevation_norm']
    enriched['static_light_extinction_norm'] = static['light_extinction_norm']
    enriched['static_fetch_norm'] = static['fetch_norm']
    enriched['static_wind_exposure_norm'] = static['wind_exposure_norm']
    enriched['static_basin_shape_norm'] = static['basin_shape_norm']
    return enriched


def _profile_only(observations):
    if 'source' not in observations.columns:
        return observations
    return observations[observations['source'].eq('profile')].copy().reset_index(drop=True)


def prepare_lake_rows(lake_config, split_mode='seasonal_blocked', data_fill_mode='forecast'):
    lake_id = str(lake_config.get('lake_id') or lake_config.get('id') or 'lake')
    era5_path = lake_config['era5']
    lst_path = lake_config['lst']
    profile_path = lake_config.get('profile_obs') or lake_config.get('profile')

    df, metadata = load_training_frame(era5_path, lst_path, data_fill_mode=data_fill_mode)
    metadata['lake_id'] = lake_id
    max_depth = float(lake_config.get('max_depth') or metadata.get('max_depth_m') or 20.0)
    metadata['runtime_max_depth_m'] = max_depth

    profile_obs = load_optional_profile_observations(
        profile_path,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    profile_splits, split_info = split_profile_observations(profile_obs, split_mode=split_mode)

    state_memory = build_causal_previous_state_memory(
        df=df,
        train_profile_obs=profile_splits['train'],
        max_depth=max_depth,
    )
    df = df.copy()
    for name, values in state_memory.items():
        df[name] = values

    train_rows, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_splits['train'],
        use_surface_bulk_correction=False,
        use_bottom_observation=False,
    )
    # Validation/test are kept profile-only so the held-out score is not padded
    # by always-available surface LST pseudo-observations.
    val_rows, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_splits['val'],
        use_surface_bulk_correction=False,
        use_bottom_observation=False,
    )
    test_rows, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_splits['test'],
        use_surface_bulk_correction=False,
        use_bottom_observation=False,
    )

    role_rows = {
        'train': train_rows,
        'val': _profile_only(val_rows),
        'test': _profile_only(test_rows),
    }
    prepared = {}
    for role, rows in role_rows.items():
        rows = _add_static_features(rows, metadata, max_depth)
        rows['lake_id'] = lake_id
        rows['role'] = role
        prepared[role] = rows

    return {
        'lake_id': lake_id,
        'metadata': metadata,
        'max_depth': max_depth,
        'rows': prepared,
        'split_info': split_info,
    }


def rows_to_tensors(rows, device='cpu'):
    if rows.empty:
        raise ValueError('Cannot build tensors from an empty observation table.')
    depth_norm = (
        pd.to_numeric(rows['Depth_m'], errors='coerce')
        / pd.to_numeric(rows['max_depth_m'], errors='coerce').clip(lower=1.0e-6)
    ).to_numpy(dtype=np.float32).reshape(-1, 1)
    features = [
        _as_float_series(rows, 'time_norm'),
        depth_norm,
        _as_float_series(rows, 'doy_sin'),
        _as_float_series(rows, 'doy_cos'),
        _as_float_series(rows, 'T_air_C') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'wind_speed_m_per_s') / PINN_MAX_WIND_REFERENCE_M_PER_S,
        _as_float_series(rows, 'Solar_W_m2') / PINN_MAX_SHORTWAVE_REFERENCE_W_M2,
        _as_float_series(rows, 'LST_surface_C') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'Longwave_W_m2') / PINN_MAX_LONGWAVE_REFERENCE_W_M2,
        _as_float_series(rows, 'latent_heat_upward_W_m2') / PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
        _as_float_series(rows, 'sensible_heat_upward_W_m2') / PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
        _as_float_series(rows, 'Secchi_m') / PINN_MAX_SECCHI_REFERENCE_M,
        _as_float_series(rows, 'static_max_depth_norm'),
        _as_float_series(rows, 'static_mean_depth_norm'),
        _as_float_series(rows, 'static_log_area'),
        _as_float_series(rows, 'static_latitude'),
        _as_float_series(rows, 'static_longitude'),
        _as_float_series(rows, 'air_temp_mean_7d') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'air_temp_mean_30d') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'shortwave_sum_7d') / PINN_SHORTWAVE_SUM_7D_REFERENCE,
        _as_float_series(rows, 'shortwave_sum_30d') / PINN_SHORTWAVE_SUM_30D_REFERENCE,
        _as_float_series(rows, 'wind_mean_7d') / PINN_MAX_WIND_REFERENCE_M_PER_S,
        _as_float_series(rows, 'lst_mean_7d') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'heating_degree_days_30d') / PINN_HEATING_DEGREE_DAYS_30D_REFERENCE,
        _as_float_series(rows, 'prev_surface_temp') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'prev_0_3m_mean') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'prev_deep_mean') / PINN_MAX_TEMPERATURE_REFERENCE_C,
        _as_float_series(rows, 'static_volume_norm'),
        _as_float_series(rows, 'static_elevation_norm'),
        _as_float_series(rows, 'static_light_extinction_norm'),
        _as_float_series(rows, 'static_fetch_norm'),
        _as_float_series(rows, 'static_wind_exposure_norm'),
        _as_float_series(rows, 'static_basin_shape_norm'),
        _as_float_series(rows, 'water_level_anomaly') / PINN_WATER_LEVEL_ANOMALY_REFERENCE_M,
        _as_float_series(rows, 'light_extinction_kd') / PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
        _as_float_series(rows, 'effective_fetch') / PINN_FETCH_REFERENCE_M,
        _as_float_series(rows, 'net_inflow') / PINN_INFLOW_REFERENCE_M3_S,
    ]
    x = np.concatenate(features[:PINN_INPUT_DIM], axis=1).astype(np.float32)
    y = _as_float_series(rows, 'Temperature_C')
    weight = _as_float_series(rows, 'obs_weight', default=1.0)
    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(weight, dtype=torch.float32, device=device),
    )


def _merge_physics_reg_config(manifest):
    config = dict(DEFAULT_PHYSICS_REG_CONFIG)
    user_config = manifest.get('physics_regularization', {}) or {}
    component_weights = dict(config['component_weights'])
    component_weights.update(user_config.get('component_weights', {}) or {})
    config.update({key: value for key, value in user_config.items() if key != 'component_weights'})
    config['component_weights'] = component_weights
    return config


def _build_physics_regularizer_rows(train_rows, depth_points):
    regularizer_rows = {}
    for lake_id, lake_rows in train_rows.groupby('lake_id', sort=False):
        base = lake_rows.sort_values('Date').drop_duplicates('Date').copy()
        if base.empty:
            continue
        max_depth = float(pd.to_numeric(base['max_depth_m'], errors='coerce').dropna().iloc[0])
        depths = np.linspace(0.0, max_depth, int(depth_points), dtype=np.float32)
        repeated = base.loc[base.index.repeat(len(depths))].copy().reset_index(drop=True)
        repeated['Depth_m'] = np.tile(depths, len(base)).astype(np.float32)
        repeated['Temperature_C'] = 0.0
        repeated['obs_weight'] = 1.0
        regularizer_rows[str(lake_id)] = {
            'rows': repeated,
            'n_days': int(len(base)),
            'n_depths': int(len(depths)),
            'depths': depths,
            'max_depth': max_depth,
        }
    return regularizer_rows


def _prepare_physics_regularizer_tensors(regularizer_rows, device='cpu'):
    tensors = {}
    for lake_id, info in regularizer_rows.items():
        x, _, _ = rows_to_tensors(info['rows'], device=device)
        tensors[lake_id] = {
            'x': x,
            'n_days': info['n_days'],
            'n_depths': info['n_depths'],
            'depths': torch.tensor(info['depths'], dtype=torch.float32, device=device),
            'max_depth': float(info['max_depth']),
        }
    return tensors


def _lake_balanced_bias_loss(pred, target, weight, lake_codes):
    losses = []
    for code in torch.unique(lake_codes):
        mask = lake_codes.eq(code)
        if int(mask.sum().detach().cpu()) < 2:
            continue
        local_weight = weight[mask]
        local_residual = pred[mask] - target[mask]
        bias = torch.sum(local_weight * local_residual) / torch.clamp(torch.sum(local_weight), min=1.0)
        losses.append(bias.pow(2))
    if not losses:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return torch.stack(losses).mean()


def _soft_grid_physics_regularization(model, regularizer_tensors, config):
    if not regularizer_tensors:
        return (
            torch.zeros((), device=next(model.parameters()).device),
            {
                'temperature_range': 0.0,
                'surface_jump': 0.0,
                'surface_band_uniformity': 0.0,
                'column_jump': 0.0,
                'deep_slow_change': 0.0,
                'vertical_gradient': 0.0,
                'density_stability': 0.0,
            },
        )
    component_weights = config['component_weights']
    lower = float(config['temperature_lower_c'])
    upper = float(config['temperature_upper_c'])
    low_temp = float(config.get('temperature_low_c', lower))
    strong_low_temp = float(config.get('temperature_strong_low_c', min(lower - 2.0, -3.0)))
    extreme_low_temp = float(config.get('temperature_extreme_low_c', min(strong_low_temp - 2.0, -5.0)))
    extreme_lower = float(config.get('temperature_extreme_lower_c', lower - 2.0))
    extreme_upper = float(config.get('temperature_extreme_upper_c', upper + 2.0))
    surface_depth = float(config['surface_band_depth_m'])
    surface_uniformity_threshold = float(config.get('surface_band_uniformity_threshold_c', 1.0))
    surface_jump_threshold = float(config['surface_jump_threshold_c_per_day'])
    column_surface_threshold = float(config['column_jump_surface_threshold_c_per_day'])
    column_deep_threshold = float(config['column_jump_deep_threshold_c_per_day'])
    deep_slow_depth_fraction = float(config.get('deep_slow_depth_fraction', 0.55))
    deep_slow_jump_threshold = float(config.get('deep_slow_jump_threshold_c_per_day', column_deep_threshold))
    gradient_threshold = float(config['vertical_gradient_threshold_c_per_m'])
    density_drop_threshold = float(config.get('density_inversion_drop_kgm3', 0.02))
    density_fraction_threshold = float(config.get('density_unstable_layer_fraction', 0.12))

    totals = {
        'temperature_range': [],
        'surface_jump': [],
        'surface_band_uniformity': [],
        'column_jump': [],
        'deep_slow_change': [],
        'vertical_gradient': [],
        'density_stability': [],
    }
    for info in regularizer_tensors.values():
        pred = model(info['x']).reshape(info['n_days'], info['n_depths'])
        depths = info['depths']
        max_depth = max(float(info['max_depth']), 1.0)
        depth_norm = (depths / max_depth).clamp(0.0, 1.0)

        low_violation = F.relu(low_temp - pred)
        strong_low_violation = F.relu(strong_low_temp - pred)
        extreme_low_violation = F.relu(extreme_low_temp - pred)
        high_violation = F.relu(pred - upper)
        extreme_high_violation = F.relu(pred - extreme_upper)
        legacy_extreme_low_violation = F.relu(extreme_lower - pred)
        range_violation = (
            0.75 * low_violation
            + 2.50 * strong_low_violation
            + 8.00 * extreme_low_violation
            + high_violation
            + 4.0 * extreme_high_violation
            + 4.0 * legacy_extreme_low_violation
        )
        range_loss = (
            torch.mean(range_violation.pow(2))
            + 0.75 * torch.mean(torch.max(range_violation.pow(2), dim=1).values)
        )
        totals['temperature_range'].append(range_loss)

        if info['n_days'] > 1:
            daily_delta = pred[1:, :] - pred[:-1, :]
            depth_threshold = column_surface_threshold - (
                column_surface_threshold - column_deep_threshold
            ) * depth_norm.pow(0.75)
            column_jump_violation = F.relu(torch.abs(daily_delta) - depth_threshold.reshape(1, -1))
            column_jump_loss = (
                torch.mean(column_jump_violation.pow(2))
                + 0.75 * torch.mean(torch.max(column_jump_violation.pow(2), dim=1).values)
                + 1.50 * torch.max(column_jump_violation.pow(2))
            )
            totals['column_jump'].append(column_jump_loss)

            deep_mask = depths >= max(3.0, max_depth * deep_slow_depth_fraction)
            if bool(torch.any(deep_mask).detach().cpu()):
                deep_daily_delta = daily_delta[:, deep_mask]
                deep_slow_violation = F.relu(torch.abs(deep_daily_delta) - deep_slow_jump_threshold)
                deep_slow_loss = (
                    torch.mean(deep_slow_violation.pow(2))
                    + 1.50 * torch.mean(torch.max(deep_slow_violation.pow(2), dim=1).values)
                    + 1.50 * torch.max(deep_slow_violation.pow(2))
                )
                totals['deep_slow_change'].append(deep_slow_loss)

            surface_mask = depths <= min(surface_depth, max_depth)
            if bool(torch.any(surface_mask).detach().cpu()):
                surface_profile = pred[:, surface_mask]
                if surface_profile.shape[1] > 1:
                    surface_deviation = torch.abs(surface_profile - surface_profile.mean(dim=1, keepdim=True))
                    surface_uniformity_violation = F.relu(surface_deviation - surface_uniformity_threshold)
                    surface_uniformity_loss = (
                        torch.mean(surface_uniformity_violation.pow(2))
                        + 1.50 * torch.mean(torch.max(surface_uniformity_violation.pow(2), dim=1).values)
                        + 1.50 * torch.max(surface_uniformity_violation.pow(2))
                    )
                    totals['surface_band_uniformity'].append(surface_uniformity_loss)
                surface_mean = pred[:, surface_mask].mean(dim=1)
                surface_jump = surface_mean[1:] - surface_mean[:-1]
                surface_jump_violation = F.relu(torch.abs(surface_jump) - surface_jump_threshold)
                surface_jump_loss = (
                    torch.mean(surface_jump_violation.pow(2))
                    + 0.75 * torch.max(surface_jump_violation.pow(2))
                )
                totals['surface_jump'].append(surface_jump_loss)

        if info['n_depths'] > 1:
            dz = torch.clamp(depths[1:] - depths[:-1], min=1.0e-3)
            vertical_gradient = (pred[:, 1:] - pred[:, :-1]) / dz.reshape(1, -1)
            gradient_loss = torch.mean(F.relu(torch.abs(vertical_gradient) - gradient_threshold).pow(2))
            totals['vertical_gradient'].append(gradient_loss)

            density = water_density_torch(pred)
            density_drop = density[:, :-1] - density[:, 1:]
            strong_drop = F.relu(density_drop - density_drop_threshold)
            strong_fraction = torch.mean((density_drop > density_drop_threshold).float(), dim=1)
            density_loss = (
                torch.mean(strong_drop.pow(2))
                + 0.75 * torch.mean(torch.max(strong_drop.pow(2), dim=1).values)
                + 0.50 * torch.mean(F.relu(strong_fraction - density_fraction_threshold).pow(2))
            )
            totals['density_stability'].append(density_loss)

    device = next(model.parameters()).device
    combined = torch.zeros((), device=device)
    components = {}
    for name, values in totals.items():
        value = torch.stack(values).mean() if values else torch.zeros((), device=device)
        components[name] = float(value.detach().cpu())
        combined = combined + float(component_weights.get(name, 0.0)) * value
    return combined, components


def _weighted_rmse(model, rows, device='cpu'):
    if rows.empty:
        return np.nan
    model.eval()
    with torch.no_grad():
        x, y, weight = rows_to_tensors(rows, device=device)
        pred = model(x)
        mse = torch.sum(weight * (pred - y) ** 2) / torch.clamp(torch.sum(weight), min=1.0)
        return float(torch.sqrt(mse).detach().cpu())


def train_global_adapter(manifest_path, output_dir, epochs=600, batch_size=1024, lr=8.0e-4, device=None, data_fill_mode=None):
    manifest = _read_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    effective_data_fill_mode = data_fill_mode or manifest.get('data_fill_mode', 'forecast')
    prepared_lakes = [
        prepare_lake_rows(
            config,
            split_mode=manifest.get('split_mode', 'seasonal_blocked'),
            data_fill_mode=effective_data_fill_mode,
        )
        for config in manifest['lakes']
    ]
    train_rows = pd.concat([lake['rows']['train'] for lake in prepared_lakes], ignore_index=True)
    val_rows = pd.concat([lake['rows']['val'] for lake in prepared_lakes], ignore_index=True)
    test_rows = pd.concat([lake['rows']['test'] for lake in prepared_lakes], ignore_index=True)

    lake_counts = train_rows['lake_id'].value_counts().to_dict()
    n_lakes = max(1, len(lake_counts))
    lake_code_map = {lake_id: idx for idx, lake_id in enumerate(sorted(lake_counts))}
    train_rows['lake_balance_weight'] = train_rows['lake_id'].map(
        lambda lake_id: len(train_rows) / (n_lakes * max(1, lake_counts.get(lake_id, 1)))
    ).astype(np.float32)
    train_rows['obs_weight'] = train_rows['obs_weight'].astype(float) * train_rows['lake_balance_weight'].astype(float)

    physics_reg_config = _merge_physics_reg_config(manifest)
    regularizer_tensors = {}
    if bool(physics_reg_config.get('enabled', True)) and float(physics_reg_config.get('weight', 0.0)) > 0.0:
        regularizer_rows = _build_physics_regularizer_rows(
            train_rows,
            depth_points=int(physics_reg_config.get('depth_points', 36)),
        )
        regularizer_tensors = _prepare_physics_regularizer_tensors(regularizer_rows, device=device)

    x_train, y_train, weight_train = rows_to_tensors(train_rows, device=device)
    lake_code_train = torch.tensor(
        train_rows['lake_id'].map(lake_code_map).to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    model = create_lake_model(
        model_architecture='global_adapter',
        input_dim=PINN_INPUT_DIM,
        hidden_dim=int(manifest.get('hidden_dim', 128)),
        hidden_layers=int(manifest.get('hidden_layers', 8)),
        adapter_dim=int(manifest.get('adapter_dim', 32)),
    ).to(device)
    model.set_adaptation_mode('full')
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    best_val = float('inf')
    best_state = None
    history = []
    n_samples = int(x_train.shape[0])
    last_physics_components = {
        'temperature_range': 0.0,
        'surface_jump': 0.0,
        'surface_band_uniformity': 0.0,
        'column_jump': 0.0,
        'deep_slow_change': 0.0,
        'vertical_gradient': 0.0,
        'density_stability': 0.0,
    }
    last_physics_loss = 0.0
    last_bias_loss = 0.0
    for epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        total_weight = 0.0
        total_bias_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            pred = model(x_train[idx])
            weight = weight_train[idx]
            obs_loss = torch.sum(weight * (pred - y_train[idx]) ** 2) / torch.clamp(torch.sum(weight), min=1.0)
            bias_loss = _lake_balanced_bias_loss(pred, y_train[idx], weight, lake_code_train[idx])
            loss = obs_loss + float(physics_reg_config.get('bias_weight', 0.0)) * bias_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(obs_loss.detach().cpu()) * float(torch.sum(weight).detach().cpu())
            total_weight += float(torch.sum(weight).detach().cpu())
            total_bias_loss += float(bias_loss.detach().cpu())
            n_batches += 1

        if regularizer_tensors:
            physics_loss, physics_components = _soft_grid_physics_regularization(
                model,
                regularizer_tensors,
                physics_reg_config,
            )
            weighted_physics_loss = float(physics_reg_config.get('weight', 0.0)) * physics_loss
            optimizer.zero_grad()
            weighted_physics_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            last_physics_loss = float(physics_loss.detach().cpu())
            last_physics_components = physics_components
        last_bias_loss = total_bias_loss / max(n_batches, 1)

        if epoch % max(1, int(manifest.get('eval_interval', 25))) == 0 or epoch == int(epochs) - 1:
            train_rmse = math.sqrt(total_loss / max(total_weight, 1.0))
            val_rmse = _weighted_rmse(model, val_rows, device=device)
            test_rmse = _weighted_rmse(model, test_rows, device=device)
            record = {
                'epoch': int(epoch),
                'train_rmse': float(train_rmse),
                'val_profile_rmse': float(val_rmse) if np.isfinite(val_rmse) else None,
                'test_profile_rmse': float(test_rmse) if np.isfinite(test_rmse) else None,
                'lake_bias_loss': float(last_bias_loss),
                'physics_reg_loss': float(last_physics_loss),
                'physics_temperature_range_loss': float(last_physics_components.get('temperature_range', 0.0)),
                'physics_surface_jump_loss': float(last_physics_components.get('surface_jump', 0.0)),
                'physics_surface_band_uniformity_loss': float(last_physics_components.get('surface_band_uniformity', 0.0)),
                'physics_column_jump_loss': float(last_physics_components.get('column_jump', 0.0)),
                'physics_deep_slow_change_loss': float(last_physics_components.get('deep_slow_change', 0.0)),
                'physics_vertical_gradient_loss': float(last_physics_components.get('vertical_gradient', 0.0)),
                'physics_density_stability_loss': float(last_physics_components.get('density_stability', 0.0)),
            }
            for lake in prepared_lakes:
                lake_id = lake['lake_id']
                record[f'{lake_id}_val_rmse'] = _weighted_rmse(model, lake['rows']['val'], device=device)
                record[f'{lake_id}_test_rmse'] = _weighted_rmse(model, lake['rows']['test'], device=device)
            history.append(record)
            print(
                f"Epoch {epoch:04d} | train_rmse={train_rmse:.3f} | "
                f"val_profile_rmse={val_rmse:.3f} | test_profile_rmse={test_rmse:.3f} | "
                f"phys={last_physics_loss:.4f} | bias={last_bias_loss:.4f}"
            )
            if np.isfinite(val_rmse) and val_rmse < best_val:
                best_val = float(val_rmse)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    training_info = {
        'final_weights': {},
        'kalman_scales': {},
        'best_selection_metric': best_val,
        'best_selection_label': 'multi_lake_val_profile_rmse',
        'surface_correction_info': None,
        'ppo_policy_bundle': None,
        'multi_lake_manifest': str(Path(manifest_path)),
        'lakes': [lake['lake_id'] for lake in prepared_lakes],
    }
    checkpoint_path = output_dir / 'G1_mohonk_mendota_global_adapter_checkpoint.pt'
    save_model_checkpoint_bundle(model, training_info, checkpoint_path)

    summary = {
        'manifest': str(Path(manifest_path)),
        'checkpoint': str(checkpoint_path),
        'device': device,
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'learning_rate': float(lr),
        'physics_regularization': physics_reg_config,
        'input_rows': {
            'train': int(len(train_rows)),
            'val_profile': int(len(val_rows)),
            'test_profile': int(len(test_rows)),
        },
        'lake_splits': {
            lake['lake_id']: {
                'max_depth_m': lake['max_depth'],
                'summary': summarize_profile_split_frames({
                    role: lake['rows'][role]
                    for role in ('train', 'val', 'test')
                }),
            }
            for lake in prepared_lakes
        },
        'history': history,
    }
    summary_path = output_dir / 'G1_training_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a multi-lake GlobalAdaptiveLakePINN.')
    parser.add_argument('--manifest', required=True, help='JSON manifest listing lake forcing/LST/profile/metadata inputs.')
    parser.add_argument('--output-dir', required=True, help='Directory for checkpoint and summary.')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=8.0e-4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--data-fill-mode', choices=['reconstruction', 'forecast'], default=None, help='Defaults to manifest data_fill_mode, then forecast for multi-lake generalization runs.')
    args = parser.parse_args(argv)
    train_global_adapter(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        data_fill_mode=args.data_fill_mode,
    )


if __name__ == '__main__':
    main()
