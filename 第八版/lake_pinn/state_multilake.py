"""Multi-lake trainer for the reconstruction-state LakePINN.

This module trains the state-space forecaster directly:

    T(t + dt, z) = M(T(t, z), forcing history, lake attributes)

Each lake keeps its own depth grid and hypsometry curve.  The shared neural
parts learn global parameterizations for Kz/Kd/model residuals, while static
lake attributes enter through the state forecaster's FiLM adapter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .conditional_priors import (
    infer_bottom_temp_prior_c,
    infer_ice_risk_prior,
    infer_thermal_regime,
)
from .data_io import (
    build_depth_grid,
    load_optional_profile_observations,
    load_training_frame,
    normalize_data_fill_mode,
    normalize_task_mode,
    split_profile_observations,
)
from .hypsometry import fallback_area_profile
from .constants import RHO_CP, SECONDS_PER_DAY
from .diagnostics import write_density_stability_summary, write_heat_closure_summaries
from .state_model import (
    ForcingBatch,
    LakeStateForecaster,
    STATIC_FEATURE_DIM,
    STATIC_FEATURE_KEYS,
    lake_adaptive_param_set,
    normalize_advective_heat_source_mode,
    normalize_lake_adaptive_params,
    normalize_lake_adaptive_temporal_mode,
    normalize_freezing_energy_mode,
    normalize_shape_aware_mixing,
    resolve_hard_density_stability,
    static_feature_array,
)
from .physics import normalize_turbulent_flux_mode
from .state_reconstruction import (
    _build_rollout_pairs,
    _forcing_tensor_rows,
    _profile_lookup,
    _profile_physics_loss,
    initialize_rollout_state,
)
from .vertical_solver import layer_thicknesses
from .export import export_temperature_tables
from .plotting import plot_year_heatmap
from .scorecard_integration import generate_prediction_diagnostic_figures, run_scorecard_report


DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT = 0.05
DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE = 0.75
DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS = {
    1: 0.5,
    2: 0.5,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 2.0,
    10: 2.0,
    11: 2.0,
    12: 0.8,
}
DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS = {
    1: 1.0,
    2: 1.0,
    3: 2.0,
    4: 2.0,
    5: 2.0,
    6: 0.8,
    7: 0.5,
    8: 0.5,
    9: 1.0,
    10: 1.0,
    11: 1.0,
    12: 1.0,
}
DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS = {
    month: 1.0 for month in range(1, 13)
}
DEFAULT_HEAT_CONTENT_TRANSITION_SEASON_FACTORS = DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS
HEAT_CONTENT_TRANSITION_SEASON_MODES = {'auto', 'northern', 'southern', 'tropical', 'manual'}
DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX = 0.10
DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT = 0.03
DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW = 0.50
DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH = 0.75
DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS = 14


def _read_manifest(path):
    path = Path(path)
    with path.open('r', encoding='utf-8-sig') as handle:
        manifest = json.load(handle)
    lakes = manifest.get('lakes', [])
    if not lakes:
        raise ValueError(f'Multi-lake state manifest has no lakes: {path}')
    return manifest


def _removed_manifest_fields(manifest, fields):
    present = set(fields).intersection(manifest)
    for lake_config in manifest.get('lakes', []):
        if isinstance(lake_config, dict):
            present.update(set(fields).intersection(lake_config))
    return sorted(present)


def _parse_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        values = text.split(',')
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    parsed = []
    for item in values:
        text = str(item).strip()
        if text:
            parsed.append(text)
    return parsed


def _unique_preserve_order(values):
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _lake_group_id(lake):
    return str(lake.get('metadata', {}).get('lake_group') or lake.get('lake_id') or '').strip()


def _resolve_heldout_selection(lakes, *, manifest, test_lake_id=None, test_lake_ids=None, heldout_lake_groups=None):
    cli_test_ids = _parse_string_list(test_lake_ids) + _parse_string_list(test_lake_id)
    manifest_test_ids = _parse_string_list(manifest.get('test_lake_ids')) + _parse_string_list(
        manifest.get('test_lake_id')
    )
    resolved_test_ids = _unique_preserve_order(cli_test_ids if cli_test_ids else manifest_test_ids)
    lake_by_id = {lake['lake_id']: lake for lake in lakes}
    missing_test_ids = [lake_id for lake_id in resolved_test_ids if lake_id not in lake_by_id]
    if missing_test_ids:
        raise ValueError(f'test_lake_ids not found in manifest lakes: {missing_test_ids}')

    inferred_groups = _unique_preserve_order(
        _lake_group_id(lake_by_id[lake_id]) for lake_id in resolved_test_ids
    )
    cli_groups = _parse_string_list(heldout_lake_groups)
    manifest_groups = _parse_string_list(manifest.get('heldout_lake_groups'))
    explicit_groups = _unique_preserve_order(cli_groups if cli_groups else manifest_groups)
    if explicit_groups:
        missing_groups = [group for group in inferred_groups if group not in set(explicit_groups)]
        if missing_groups:
            raise ValueError(
                'heldout_lake_groups must include each selected test lake group; '
                f'missing groups for test_lake_ids: {missing_groups}'
            )
        resolved_groups = explicit_groups
    else:
        resolved_groups = inferred_groups

    group_set = set(resolved_groups)
    if resolved_test_ids:
        heldout_lakes = [lake_by_id[lake_id] for lake_id in resolved_test_ids]
    else:
        heldout_lakes = [lake for lake in lakes if _lake_group_id(lake) in group_set]
        resolved_test_ids = [lake['lake_id'] for lake in heldout_lakes]
    excluded_lakes = [lake for lake in lakes if _lake_group_id(lake) in group_set]
    train_lakes = [lake for lake in lakes if _lake_group_id(lake) not in group_set]
    return {
        'test_lake_id': resolved_test_ids[0] if resolved_test_ids else '',
        'test_lake_ids': resolved_test_ids,
        'heldout_lake_groups': resolved_groups,
        'train_lakes': train_lakes,
        'heldout_lakes': heldout_lakes,
        'excluded_lakes': excluded_lakes,
    }


def _write_lake_adaptive_parameter_summary(model, lakes, train_lakes, heldout_lakes, excluded_lakes, output_dir):
    output_path = Path(output_dir) / 'lake_adaptive_parameter_summary.csv'
    train_ids = {lake['lake_id'] for lake in train_lakes}
    heldout_ids = {lake['lake_id'] for lake in heldout_lakes}
    excluded_ids = {lake['lake_id'] for lake in excluded_lakes}
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = torch.device('cpu')
    was_training = model.training
    rows = []
    model.eval()
    with torch.no_grad():
        for lake in lakes:
            lake_id = lake['lake_id']
            if lake_id in heldout_ids:
                split = 'heldout'
            elif lake_id in train_ids:
                split = 'train'
            elif lake_id in excluded_ids:
                split = 'excluded'
            else:
                split = 'other'
            metadata = lake.get('metadata', {}) or {}
            static_features = lake['static_features'].reshape(1, -1).to(param_device)
            if getattr(model, 'lake_adaptive_temporal_head', None) is None:
                values, regularization = model._adaptive_parameter_values(static_features)
                value_summary = {
                    key: {
                        'mean': float(value.reshape(-1)[0].detach().cpu()),
                        'std': 0.0,
                    }
                    for key, value in values.items()
                }
                regularization_mean = float(regularization.reshape(-1)[0].detach().cpu())
            else:
                per_day_values = {key: [] for key in (
                    'wind_kz_scale',
                    'blend_alpha',
                    'kd_multiplier',
                    'turbulent_exchange_scale',
                    'convective_mixing_scale',
                    'ice_shortwave_scale',
                )}
                regularization_values = []
                for forcing_row in lake.get('forcing_rows', []):
                    forcing_features = forcing_row['features'].to(param_device).reshape(1, -1)
                    forcing_history = forcing_row.get('history_features')
                    if forcing_history is not None:
                        forcing_history = forcing_history.to(param_device)
                        if forcing_history.ndim == 2:
                            forcing_history = forcing_history.unsqueeze(0)
                    forcing_context = model._encode_forcing_context(forcing_features, forcing_history)
                    values, regularization = model._adaptive_parameter_values(
                        static_features,
                        forcing_context=forcing_context,
                        forcing_features=forcing_features,
                    )
                    for key in per_day_values:
                        per_day_values[key].append(float(values[key].reshape(-1)[0].detach().cpu()))
                    regularization_values.append(float(regularization.reshape(-1)[0].detach().cpu()))
                value_summary = {}
                for key, values_for_key in per_day_values.items():
                    series = np.asarray(values_for_key, dtype=np.float64)
                    finite = series[np.isfinite(series)]
                    value_summary[key] = {
                        'mean': float(np.mean(finite)) if finite.size else np.nan,
                        'std': float(np.std(finite)) if finite.size else np.nan,
                    }
                reg_series = np.asarray(regularization_values, dtype=np.float64)
                reg_finite = reg_series[np.isfinite(reg_series)]
                regularization_mean = float(np.mean(reg_finite)) if reg_finite.size else np.nan
            row = {
                'lake_id': lake_id,
                'lake_group': metadata.get('lake_group', ''),
                'split': split,
                'is_train_lake': bool(lake_id in train_ids),
                'is_heldout_test_lake': bool(lake_id in heldout_ids),
                'is_excluded_same_group_lake': bool(lake_id in excluded_ids and lake_id not in heldout_ids),
                'latitude': metadata.get('latitude', np.nan),
                'longitude': metadata.get('longitude', np.nan),
                'max_depth_m': metadata.get('max_depth_m', np.nan),
                'surface_area_m2': metadata.get('surface_area_m2', np.nan),
                'lake_adaptive_temporal_mode': getattr(model, 'lake_adaptive_temporal_mode', 'off'),
                'adaptive_parameter_regularization_loss': regularization_mean,
            }
            for key, column in (
                ('wind_kz_scale', 'adaptive_wind_kz_scale'),
                ('blend_alpha', 'adaptive_turbulent_flux_blend_alpha'),
                ('kd_multiplier', 'adaptive_kd_multiplier'),
                ('turbulent_exchange_scale', 'adaptive_turbulent_exchange_scale'),
                ('convective_mixing_scale', 'adaptive_convective_mixing_scale'),
                ('ice_shortwave_scale', 'adaptive_ice_shortwave_scale'),
            ):
                row[column] = value_summary[key]['mean']
                row[f'{column}_std'] = value_summary[key]['std']
            rows.append(row)
    if was_training:
        model.train()
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _nanmean_or_nan(values):
    values = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else np.nan


def _parse_heat_content_transition_season_factors(value=None):
    factors = dict(DEFAULT_HEAT_CONTENT_TRANSITION_SEASON_FACTORS)
    if value is None:
        return factors
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return factors
        if text.startswith('{'):
            value = json.loads(text)
        else:
            parsed = {}
            for part in text.split(','):
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    key, raw_factor = part.split(':', 1)
                elif '=' in part:
                    key, raw_factor = part.split('=', 1)
                else:
                    raise ValueError(
                        "heat_content_transition_season_factors entries must use month:factor."
                    )
                parsed[key.strip()] = raw_factor.strip()
            value = parsed
    if isinstance(value, dict):
        for raw_month, raw_factor in value.items():
            month = int(raw_month)
            if month < 1 or month > 12:
                raise ValueError('heat_content_transition_season_factors months must be in 1..12.')
            factor = float(raw_factor)
            if not np.isfinite(factor) or factor < 0.0:
                raise ValueError('heat_content_transition_season_factors values must be finite and non-negative.')
            factors[month] = factor
    else:
        items = list(value)
        if len(items) != 12:
            raise ValueError('heat_content_transition_season_factors sequence must have 12 values.')
        for month, raw_factor in enumerate(items, start=1):
            factor = float(raw_factor)
            if not np.isfinite(factor) or factor < 0.0:
                raise ValueError('heat_content_transition_season_factors values must be finite and non-negative.')
            factors[month] = factor
    return {month: float(factors[month]) for month in range(1, 13)}


def _has_heat_content_transition_season_override(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _normalize_heat_content_transition_season_mode(value):
    mode = str(value if value is not None else 'auto').strip().lower()
    if mode not in HEAT_CONTENT_TRANSITION_SEASON_MODES:
        allowed = ', '.join(sorted(HEAT_CONTENT_TRANSITION_SEASON_MODES))
        raise ValueError(f"heat_content_transition_season_mode must be one of: {allowed}.")
    return mode


def _metadata_latitude(metadata):
    metadata = metadata or {}
    for key in ('latitude', 'lat', 'latitude_deg'):
        if key in metadata:
            try:
                latitude = float(metadata.get(key))
            except (TypeError, ValueError):
                return np.nan
            return latitude if np.isfinite(latitude) and abs(latitude) <= 90.0 else np.nan
    return np.nan


def _default_heat_content_transition_factors_for_mode(mode):
    mode = _normalize_heat_content_transition_season_mode(mode)
    if mode == 'northern':
        return dict(DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS)
    if mode == 'southern':
        return dict(DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS)
    if mode == 'tropical':
        return dict(DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS)
    raise ValueError(f'No default heat-content transition factors for mode={mode!r}.')


def _resolve_heat_content_transition_season_factors(metadata, mode='auto', override=None):
    requested_mode = _normalize_heat_content_transition_season_mode(mode)
    latitude = _metadata_latitude(metadata)
    has_override = _has_heat_content_transition_season_override(override)
    if has_override:
        return {
            'requested_mode': requested_mode,
            'resolved_mode': 'manual',
            'latitude': float(latitude) if np.isfinite(latitude) else None,
            'factors': _parse_heat_content_transition_season_factors(override),
        }
    if requested_mode == 'manual':
        raise ValueError('heat_content_transition_season_mode=manual requires heat_content_transition_season_factors.')
    if requested_mode == 'auto':
        if not np.isfinite(latitude):
            resolved_mode = 'northern_fallback'
            factors = dict(DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS)
        elif latitude >= 23.5:
            resolved_mode = 'northern'
            factors = dict(DEFAULT_HEAT_CONTENT_TRANSITION_NORTHERN_SEASON_FACTORS)
        elif latitude <= -23.5:
            resolved_mode = 'southern'
            factors = dict(DEFAULT_HEAT_CONTENT_TRANSITION_SOUTHERN_SEASON_FACTORS)
        else:
            resolved_mode = 'tropical'
            factors = dict(DEFAULT_HEAT_CONTENT_TRANSITION_TROPICAL_SEASON_FACTORS)
    else:
        resolved_mode = requested_mode
        factors = _default_heat_content_transition_factors_for_mode(requested_mode)
    return {
        'requested_mode': requested_mode,
        'resolved_mode': resolved_mode,
        'latitude': float(latitude) if np.isfinite(latitude) else None,
        'factors': {month: float(factors[month]) for month in range(1, 13)},
    }


def _heat_content_transition_season_factors_payload(factors):
    return {str(month): float(factors[month]) for month in range(1, 13)}


def _heat_content_transition_lake_config_payload(lake):
    return {
        'requested_mode': lake.get('heat_content_transition_season_mode_requested'),
        'resolved_mode': lake.get('heat_content_transition_season_mode_resolved'),
        'latitude': lake.get('heat_content_transition_latitude'),
        'season_factors': _heat_content_transition_season_factors_payload(
            lake['heat_content_transition_season_factors']
        ),
    }


def _metadata_summary_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return float(numeric) if np.isfinite(numeric) else None


def _lake_metadata_summary_payload(lake):
    metadata = lake.get('metadata') or {}
    keys = (
        'lake_name',
        'lake_id',
        'lake_group',
        'lake_type',
        'geographic_climate_zone',
        'geographic_lake_regime',
        'thermal_regime',
        'bottom_temp_prior_c',
        'max_depth_m',
        'mean_depth_m',
        'area_km2',
        'latitude',
        'longitude',
        'elevation_m',
        'volume_km3',
        'light_extinction_kd',
        'fetch_m',
        'effective_fetch_m',
        'wind_exposure_index',
        'basin_shape_factor',
        'reservoir_indicator',
        'residence_time_days',
        'shoreline_length_km',
        'shoreline_development',
        'catchment_area_km2',
        'discharge_m3_s',
        'max_depth_norm',
        'mean_depth_norm',
        'log_area',
        'latitude_norm',
        'longitude_norm',
        'volume_norm',
        'elevation_norm',
        'light_extinction_norm',
        'fetch_norm',
        'wind_exposure_norm',
        'basin_shape_norm',
        'residence_time_norm',
        'shoreline_length_norm',
        'shoreline_development_norm',
        'catchment_area_norm',
        'discharge_norm',
        'metadata_path',
    )
    payload = {
        key: _metadata_summary_value(metadata.get(key))
        for key in keys
        if key in metadata
    }
    payload['static_feature_dim'] = int(lake['static_features'].reshape(-1).numel())
    payload['static_feature_keys'] = list(STATIC_FEATURE_KEYS)
    payload['heat_content_transition'] = _heat_content_transition_lake_config_payload(lake)
    return payload


def _normalize_heat_content_transition_depth_factor(value):
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else 'on').strip().lower()
    if text in {'on', 'true', '1', 'yes'}:
        return True
    if text in {'off', 'false', '0', 'no'}:
        return False
    raise ValueError("heat_content_transition_depth_factor must be 'on' or 'off'.")


def _heat_content_transition_depth_multiplier(max_depth_m, enabled=True):
    if not enabled:
        return 1.0
    depth = float(max_depth_m)
    if not np.isfinite(depth) or depth <= 0.0:
        return 1.0
    return float(np.clip(depth / 25.0, 0.75, 1.5))


def _heat_content_transition_effective_weight(
    base_weight,
    target_date,
    max_depth_m,
    season_factors=None,
    *,
    use_depth_factor=True,
    effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
):
    base_weight = float(base_weight)
    if not np.isfinite(base_weight) or base_weight < 0.0:
        raise ValueError('heat_content_transition_weight must be finite and non-negative.')
    if base_weight == 0.0:
        return 0.0
    if isinstance(target_date, (int, np.integer)):
        month = int(target_date)
    else:
        month = int(pd.Timestamp(target_date).month)
    if month < 1 or month > 12:
        raise ValueError('heat-content transition target month must be in 1..12.')
    factors = _parse_heat_content_transition_season_factors(season_factors)
    depth_factor = _heat_content_transition_depth_multiplier(max_depth_m, enabled=use_depth_factor)
    cap = float(effective_max)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError('heat_content_transition_effective_max must be finite and non-negative.')
    effective = base_weight * factors[month] * depth_factor
    return float(np.clip(effective, 0.0, cap))


def _profile_rmse(prediction, target, mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.nan
    return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))


def _lookup_mask(lake, split_name, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    return lake.get('lookup_masks', {}).get(split_name, {}).get(date_value)


def _lookup_mask_tensor(lake, split_name, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    return lake.get('lookup_mask_tensors', {}).get(split_name, {}).get(date_value)


def _lookup_profile_tensor(lake, split_name, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    return lake.get('lookup_tensors', {}).get(split_name, {}).get(date_value)


def _target_lookup_and_mask(lake, preferred_split, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    if date_value in lake['lookups'].get(preferred_split, {}):
        return (
            lake['lookups'][preferred_split][date_value],
            _lookup_mask(lake, preferred_split, date_value),
        )
    return (
        lake['lookups']['all'][date_value],
        _lookup_mask(lake, 'all', date_value),
    )


def _target_tensor_and_mask(lake, preferred_split, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    profile = _lookup_profile_tensor(lake, preferred_split, date_value)
    mask = _lookup_mask_tensor(lake, preferred_split, date_value)
    if profile is not None:
        return profile, mask
    profile = _lookup_profile_tensor(lake, 'all', date_value)
    mask = _lookup_mask_tensor(lake, 'all', date_value)
    if profile is None:
        raw_profile, raw_mask = _target_lookup_and_mask(lake, preferred_split, date_value)
        device = lake['depths'].device
        profile = torch.tensor(raw_profile, dtype=torch.float32, device=device).unsqueeze(0)
        mask = (
            torch.as_tensor(raw_mask, dtype=torch.bool, device=device).reshape(1, -1)
            if raw_mask is not None else None
        )
    return profile, mask


def _masked_huber_profile_loss(prediction, target, mask=None, delta=2.0):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.reshape(1, -1)
        elif mask_tensor.ndim != 2:
            mask_tensor = mask_tensor.reshape(mask_tensor.shape[0], -1)
        valid = valid & mask_tensor
    if not torch.any(valid):
        return torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
    loss = torch.nn.functional.huber_loss(
        prediction,
        target,
        delta=float(delta),
        reduction='none',
    )
    return loss[valid].mean()


def _heat_content_layer_weights(depths, area, *, device, dtype):
    depths = torch.as_tensor(depths, dtype=dtype, device=device).reshape(-1)
    area = torch.as_tensor(area, dtype=dtype, device=device).reshape(-1)
    return area * layer_thicknesses(depths)


def _heat_content_coverage_fraction(mask, depths, area):
    if mask is None:
        return 1.0
    device = area.device if isinstance(area, torch.Tensor) else (
        depths.device if isinstance(depths, torch.Tensor) else torch.device('cpu')
    )
    weights = _heat_content_layer_weights(depths, area, device=device, dtype=torch.float32)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(-1)
    if mask_tensor.numel() != weights.numel():
        raise ValueError(
            f'Heat-content mask length {mask_tensor.numel()} does not match depth grid length {weights.numel()}.'
        )
    total_weight = torch.sum(weights)
    if (not torch.isfinite(total_weight).item()) or total_weight.item() <= 0.0:
        return 0.0
    covered_weight = torch.sum(weights * mask_tensor.to(dtype=weights.dtype))
    fraction = torch.clamp(covered_weight / total_weight, 0.0, 1.0)
    return float(fraction.detach().cpu())


def _heat_content_transition_mask(
    start_profile,
    end_prediction,
    depths,
    area,
    start_mask=None,
    end_mask=None,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if start_mask is None and end_mask is None:
        return None
    threshold = float(np.clip(min_full_column_coverage, 0.0, 1.0))
    start_coverage = _heat_content_coverage_fraction(start_mask, depths, area)
    end_coverage = _heat_content_coverage_fraction(end_mask, depths, area)
    if start_coverage >= threshold and end_coverage >= threshold:
        return None

    n_depths = int(end_prediction.shape[-1])
    device = end_prediction.device if isinstance(end_prediction, torch.Tensor) else torch.device('cpu')
    if start_mask is None:
        start_mask = torch.ones(n_depths, dtype=torch.bool, device=device)
    if end_mask is None:
        end_mask = torch.ones(n_depths, dtype=torch.bool, device=device)
    start_mask = torch.as_tensor(start_mask, dtype=torch.bool, device=device).reshape(-1)
    end_mask = torch.as_tensor(end_mask, dtype=torch.bool, device=device).reshape(-1)
    if start_mask.numel() != n_depths or end_mask.numel() != n_depths:
        raise ValueError(
            f'Heat-content masks must match depth grid length {n_depths}: '
            f'got {start_mask.numel()} and {end_mask.numel()}.'
        )
    return start_mask & end_mask


def _masked_heat_content_j_m2(profile, depths, area, mask=None):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    area = torch.as_tensor(area, dtype=profile.dtype, device=profile.device).reshape(-1)
    weights = _heat_content_layer_weights(
        depths,
        area,
        device=profile.device,
        dtype=profile.dtype,
    ).reshape(1, -1)
    if mask is not None:
        weights = weights * torch.as_tensor(mask, dtype=profile.dtype, device=profile.device).reshape(1, -1)
    surface_area = torch.clamp(area[0], min=1.0e-6)
    return RHO_CP * torch.sum(profile * weights / surface_area, dim=1)


def _heat_content_transition_loss(
    start_profile,
    end_prediction,
    end_target,
    depths,
    area,
    start_mask=None,
    end_mask=None,
    delta_seconds=1.0,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if start_profile.ndim == 1:
        start_profile = start_profile.unsqueeze(0)
    if end_prediction.ndim == 1:
        end_prediction = end_prediction.unsqueeze(0)
    if end_target.ndim == 1:
        end_target = end_target.unsqueeze(0)
    content_mask = _heat_content_transition_mask(
        start_profile,
        end_prediction,
        depths,
        area,
        start_mask=start_mask,
        end_mask=end_mask,
        min_full_column_coverage=min_full_column_coverage,
    )
    if content_mask is not None and not torch.any(torch.as_tensor(content_mask, dtype=torch.bool, device=end_prediction.device)):
        return torch.tensor(0.0, dtype=end_prediction.dtype, device=end_prediction.device)
    heat_start = _masked_heat_content_j_m2(start_profile, depths, area, content_mask)
    heat_pred_end = _masked_heat_content_j_m2(end_prediction, depths, area, content_mask)
    heat_obs_end = _masked_heat_content_j_m2(end_target, depths, area, content_mask)
    dt = max(float(delta_seconds), 1.0)
    pred_rate = (heat_pred_end - heat_start) / dt
    obs_rate = (heat_obs_end - heat_start) / dt
    return torch.mean(((pred_rate - obs_rate) / 150.0).pow(2))


def _append_weighted_heat_content_transition_loss(
    heat_content_losses,
    heat_content_weighted_losses,
    heat_content_effective_weights,
    *,
    base_weight,
    target_date,
    lake,
    start_profile,
    end_prediction,
    end_target,
    start_mask=None,
    end_mask=None,
    delta_seconds=1.0,
    season_factors=None,
    use_depth_factor=True,
    effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    effective_weight = _heat_content_transition_effective_weight(
        base_weight,
        target_date,
        lake['max_depth'],
        season_factors,
        use_depth_factor=use_depth_factor,
        effective_max=effective_max,
    )
    if effective_weight <= 0.0:
        return
    loss = _heat_content_transition_loss(
        start_profile,
        end_prediction,
        end_target,
        lake['depths'],
        lake['area'],
        start_mask=start_mask,
        end_mask=end_mask,
        delta_seconds=delta_seconds,
        min_full_column_coverage=min_full_column_coverage,
    )
    heat_content_losses.append(loss)
    heat_content_weighted_losses.append(float(effective_weight) * loss)
    heat_content_effective_weights.append(torch.tensor(
        float(effective_weight),
        dtype=end_prediction.dtype,
        device=end_prediction.device,
    ))


def _heat_content_transition_loss_vector(
    start_profile,
    end_prediction,
    end_target,
    depths,
    area,
    *,
    start_mask,
    end_mask,
    delta_seconds,
    layer_weights=None,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if start_profile.ndim == 1:
        start_profile = start_profile.unsqueeze(0)
    if end_prediction.ndim == 1:
        end_prediction = end_prediction.unsqueeze(0)
    if end_target.ndim == 1:
        end_target = end_target.unsqueeze(0)
    batch_size, n_depths = end_prediction.shape
    device = end_prediction.device
    dtype = end_prediction.dtype
    if start_mask is None:
        start_mask = torch.ones(batch_size, n_depths, dtype=torch.bool, device=device)
    else:
        start_mask = torch.as_tensor(start_mask, dtype=torch.bool, device=device).reshape(batch_size, n_depths)
    if end_mask is None:
        end_mask = torch.ones(batch_size, n_depths, dtype=torch.bool, device=device)
    else:
        end_mask = torch.as_tensor(end_mask, dtype=torch.bool, device=device).reshape(batch_size, n_depths)

    area_tensor = torch.as_tensor(area, dtype=dtype, device=device).reshape(-1)
    if layer_weights is None:
        layer_weights = _heat_content_layer_weights(
            depths,
            area_tensor,
            device=device,
            dtype=dtype,
        )
    else:
        layer_weights = torch.as_tensor(layer_weights, dtype=dtype, device=device).reshape(-1)
    layer_weights = layer_weights.reshape(1, -1)
    surface_area = torch.clamp(area_tensor[0], min=1.0e-6)
    raw_total = torch.clamp(layer_weights.sum(dim=1), min=1.0e-12)
    start_coverage = (layer_weights * start_mask.to(dtype=dtype)).sum(dim=1) / raw_total
    end_coverage = (layer_weights * end_mask.to(dtype=dtype)).sum(dim=1) / raw_total
    threshold = float(np.clip(min_full_column_coverage, 0.0, 1.0))
    full_column = (start_coverage >= threshold) & (end_coverage >= threshold)
    common_mask = start_mask & end_mask
    all_mask = torch.ones_like(common_mask)
    content_mask = torch.where(full_column.reshape(-1, 1), all_mask, common_mask)
    nonempty = torch.any(content_mask, dim=1)
    masked_weights = layer_weights * content_mask.to(dtype=dtype)

    def _batch_heat(profile):
        return RHO_CP * torch.sum(profile * masked_weights / surface_area, dim=1)

    heat_start = _batch_heat(start_profile)
    heat_pred_end = _batch_heat(end_prediction)
    heat_obs_end = _batch_heat(end_target)
    dt = torch.as_tensor(delta_seconds, dtype=dtype, device=device).reshape(-1)
    dt = torch.clamp(dt, min=1.0)
    pred_rate = (heat_pred_end - heat_start) / dt
    obs_rate = (heat_obs_end - heat_start) / dt
    loss = ((pred_rate - obs_rate) / 150.0).pow(2)
    return torch.where(nonempty, loss, torch.zeros_like(loss))


def _append_weighted_heat_content_transition_loss_batch(
    heat_content_losses,
    heat_content_weighted_losses,
    heat_content_effective_weights,
    *,
    sample_indices,
    base_weight,
    target_dates,
    lake,
    start_profile,
    end_prediction,
    end_target,
    start_mask,
    end_mask,
    delta_seconds,
    season_factors=None,
    use_depth_factor=True,
    effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if float(base_weight) <= 0.0 or not sample_indices:
        return
    device = end_prediction.device
    dtype = end_prediction.dtype
    effective_weight_values = [
        _heat_content_transition_effective_weight(
            base_weight,
            target_date,
            lake['max_depth'],
            season_factors,
            use_depth_factor=use_depth_factor,
            effective_max=effective_max,
        )
        for target_date in target_dates
    ]
    effective_weights = torch.as_tensor(
        effective_weight_values,
        dtype=dtype,
        device=device,
    )
    if not any(weight > 0.0 for weight in effective_weight_values):
        return
    loss_vec = _heat_content_transition_loss_vector(
        start_profile,
        end_prediction,
        end_target,
        lake['depths'],
        lake['area'],
        start_mask=start_mask,
        end_mask=end_mask,
        delta_seconds=delta_seconds,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    for pos, sample_idx in enumerate(sample_indices):
        if effective_weight_values[pos] > 0.0:
            loss = loss_vec[pos]
            weight = effective_weights[pos]
            heat_content_losses[sample_idx].append(loss)
            heat_content_weighted_losses[sample_idx].append(weight * loss)
            heat_content_effective_weights[sample_idx].append(weight.detach())


def _heat_content_transition_loss_details(
    heat_content_losses,
    heat_content_weighted_losses,
    heat_content_effective_weights,
    *,
    device,
    prefix='',
):
    zero = torch.tensor(0.0, device=device)
    unweighted = torch.stack(heat_content_losses).mean() if heat_content_losses else zero
    weighted = torch.stack(heat_content_weighted_losses).mean() if heat_content_weighted_losses else zero
    if heat_content_effective_weights:
        weights = torch.stack(heat_content_effective_weights)
        weight_mean = weights.mean()
        weight_min = weights.min()
        weight_max = weights.max()
    else:
        weight_mean = zero
        weight_min = zero
        weight_max = zero
    return unweighted, weighted, {
        f'{prefix}heat_content_transition_loss': unweighted.detach(),
        f'{prefix}heat_content_transition_weighted_loss': weighted.detach(),
        f'{prefix}heat_content_transition_effective_weight_mean': weight_mean.detach(),
        f'{prefix}heat_content_transition_effective_weight_min': weight_min.detach(),
        f'{prefix}heat_content_transition_effective_weight_max': weight_max.detach(),
    }


def _column_mean_temperature_c_vector(
    profile,
    depths,
    area,
    *,
    mask=None,
    layer_weights=None,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    batch_size, n_depths = profile.shape
    device = profile.device
    dtype = profile.dtype
    if mask is None:
        mask_tensor = torch.ones(batch_size, n_depths, dtype=torch.bool, device=device)
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(batch_size, n_depths)

    if layer_weights is None:
        area_tensor = torch.as_tensor(area, dtype=dtype, device=device).reshape(-1)
        layer_weights = _heat_content_layer_weights(
            depths,
            area_tensor,
            device=device,
            dtype=dtype,
        )
    else:
        layer_weights = torch.as_tensor(layer_weights, dtype=dtype, device=device).reshape(-1)
    layer_weights = layer_weights.reshape(1, -1)
    if layer_weights.shape[1] != n_depths:
        raise ValueError(
            f'Column heat-content weights length {layer_weights.shape[1]} does not match profile depth length {n_depths}.'
        )

    raw_total = torch.clamp(layer_weights.sum(dim=1), min=1.0e-12)
    coverage = (layer_weights * mask_tensor.to(dtype=dtype)).sum(dim=1) / raw_total
    threshold = float(np.clip(min_full_column_coverage, 0.0, 1.0))
    full_column = coverage >= threshold
    effective_mask = torch.where(full_column.reshape(-1, 1), torch.ones_like(mask_tensor), mask_tensor)
    finite = torch.isfinite(profile)
    effective_mask = effective_mask & finite
    masked_weights = layer_weights * effective_mask.to(dtype=dtype)
    denominator = masked_weights.sum(dim=1)
    valid = denominator > 1.0e-12
    safe_denominator = torch.clamp(denominator, min=1.0e-12)
    safe_profile = torch.where(finite, profile, torch.zeros_like(profile))
    column_mean = torch.sum(safe_profile * masked_weights, dim=1) / safe_denominator
    return torch.where(valid, column_mean, torch.zeros_like(column_mean)), valid


def _warm_column_heat_content_quantiles(
    lookup,
    masks,
    depths,
    area,
    *,
    quantile_low=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
    quantile_high=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    layer_weights=None,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if not lookup:
        return {'low_c': float('nan'), 'high_c': float('nan'), 'count': 0}
    dates = sorted(lookup)
    profile = torch.stack([
        torch.as_tensor(lookup[date], dtype=torch.float32).reshape(-1)
        for date in dates
    ], dim=0)
    mask = torch.stack([
        torch.as_tensor(
            masks.get(date, np.isfinite(np.asarray(lookup[date], dtype=float))),
            dtype=torch.bool,
        ).reshape(-1)
        for date in dates
    ], dim=0)
    column_mean, valid = _column_mean_temperature_c_vector(
        profile,
        depths,
        area,
        mask=mask,
        layer_weights=layer_weights,
        min_full_column_coverage=min_full_column_coverage,
    )
    values = column_mean[valid & torch.isfinite(column_mean)].detach().cpu()
    if values.numel() == 0:
        return {'low_c': float('nan'), 'high_c': float('nan'), 'count': 0}
    low = torch.quantile(values, float(quantile_low)).item()
    high = torch.quantile(values, float(quantile_high)).item()
    return {'low_c': float(low), 'high_c': float(high), 'count': int(values.numel())}


def _set_warm_column_heat_content_lake_config(
    lake,
    *,
    quantile_low=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
    quantile_high=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    quantiles = _warm_column_heat_content_quantiles(
        lake['lookups'].get('all', {}),
        lake['lookup_masks'].get('all', {}),
        lake['depths'],
        lake['area'],
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    lake['warm_season_column_heat_content_quantile_low'] = float(quantile_low)
    lake['warm_season_column_heat_content_quantile_high'] = float(quantile_high)
    lake['warm_season_column_heat_content_low_c'] = quantiles['low_c']
    lake['warm_season_column_heat_content_high_c'] = quantiles['high_c']
    lake['warm_season_column_heat_content_profile_count'] = quantiles['count']
    return quantiles


def _warm_column_heat_content_lake_config_payload(lake):
    return {
        'quantile_low': float(lake.get(
            'warm_season_column_heat_content_quantile_low',
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
        )),
        'quantile_high': float(lake.get(
            'warm_season_column_heat_content_quantile_high',
            DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
        )),
        'low_c': float(lake.get('warm_season_column_heat_content_low_c', float('nan'))),
        'high_c': float(lake.get('warm_season_column_heat_content_high_c', float('nan'))),
        'profile_count': int(lake.get('warm_season_column_heat_content_profile_count', 0)),
    }


def _warm_column_heat_content_loss_vector(
    end_prediction,
    end_target,
    lake,
    *,
    end_mask,
    target_gap_days,
    horizon_weight,
    weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    profile_huber_delta=2.0,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if end_prediction.ndim == 1:
        end_prediction = end_prediction.unsqueeze(0)
    if end_target.ndim == 1:
        end_target = end_target.unsqueeze(0)
    device = end_prediction.device
    dtype = end_prediction.dtype
    batch_size = end_prediction.shape[0]
    zero = torch.zeros(batch_size, dtype=dtype, device=device)
    if float(weight) <= 0.0:
        return zero, zero, zero, zero, torch.zeros(batch_size, dtype=torch.bool, device=device)

    low = float(lake.get('warm_season_column_heat_content_low_c', float('nan')))
    high = float(lake.get('warm_season_column_heat_content_high_c', float('nan')))
    if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low:
        return zero, zero, zero, zero, torch.zeros(batch_size, dtype=torch.bool, device=device)

    pred_mean, pred_valid = _column_mean_temperature_c_vector(
        end_prediction,
        lake['depths'],
        lake['area'],
        mask=end_mask,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    obs_mean, obs_valid = _column_mean_temperature_c_vector(
        end_target,
        lake['depths'],
        lake['area'],
        mask=end_mask,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    warm_factor = torch.clamp(
        (obs_mean - torch.as_tensor(low, dtype=dtype, device=device))
        / max(high - low, 1.0e-6),
        0.0,
        1.0,
    )
    gaps = torch.as_tensor(target_gap_days, dtype=dtype, device=device).reshape(-1)
    horizons = torch.as_tensor(horizon_weight, dtype=dtype, device=device).reshape(-1)
    if gaps.numel() == 1 and batch_size > 1:
        gaps = gaps.expand(batch_size)
    if horizons.numel() == 1 and batch_size > 1:
        horizons = horizons.expand(batch_size)
    active = pred_valid & obs_valid & (gaps >= float(min_gap_days)) & (warm_factor > 0.0)
    raw_loss = torch.nn.functional.huber_loss(
        pred_mean,
        obs_mean,
        delta=float(profile_huber_delta),
        reduction='none',
    )
    raw_loss = torch.where(active, raw_loss, zero)
    error_c = torch.where(active, torch.abs(pred_mean - obs_mean), zero)
    weighted_loss = torch.where(
        active,
        float(weight) * warm_factor * horizons * raw_loss,
        zero,
    )
    return raw_loss, weighted_loss, warm_factor, error_c, active


def _append_warm_column_heat_content_loss_batch(
    warm_losses,
    warm_weighted_losses,
    warm_factors,
    warm_errors_c,
    warm_gaps,
    *,
    sample_indices,
    target_gap_days,
    horizon_weight,
    lake,
    end_prediction,
    end_target,
    end_mask,
    weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    profile_huber_delta=2.0,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    if float(weight) <= 0.0 or not sample_indices:
        return
    raw_loss, weighted_loss, warm_factor, error_c, active = _warm_column_heat_content_loss_vector(
        end_prediction,
        end_target,
        lake,
        end_mask=end_mask,
        target_gap_days=target_gap_days,
        horizon_weight=horizon_weight,
        weight=weight,
        min_gap_days=min_gap_days,
        profile_huber_delta=profile_huber_delta,
        min_full_column_coverage=min_full_column_coverage,
    )
    for pos, sample_idx in enumerate(sample_indices):
        if bool(active[pos].detach().cpu().item()):
            warm_losses[sample_idx].append(raw_loss[pos])
            warm_weighted_losses[sample_idx].append(weighted_loss[pos])
            warm_factors[sample_idx].append(warm_factor[pos].detach())
            warm_errors_c[sample_idx].append(error_c[pos].detach())
            warm_gaps[sample_idx].append(torch.as_tensor(
                float(torch.as_tensor(target_gap_days, device=weighted_loss.device).reshape(-1)[pos].detach().cpu()),
                dtype=weighted_loss.dtype,
                device=weighted_loss.device,
            ))


def _warm_column_heat_content_loss_details(
    warm_losses,
    warm_weighted_losses,
    warm_factors,
    warm_errors_c,
    warm_gaps,
    *,
    device,
    prefix='segment_rollout_',
):
    zero = torch.tensor(0.0, device=device)
    unweighted = torch.stack(warm_losses).mean() if warm_losses else zero
    weighted = torch.stack(warm_weighted_losses).mean() if warm_weighted_losses else zero
    warm_factor_mean = torch.stack(warm_factors).mean() if warm_factors else zero
    error_mean = torch.stack(warm_errors_c).mean() if warm_errors_c else zero
    supervision_count = torch.as_tensor(float(len(warm_weighted_losses)), device=device)
    if warm_gaps:
        gaps = torch.stack(warm_gaps)
        horizon14 = torch.sum((gaps >= 14.0).to(dtype=zero.dtype))
        horizon30 = torch.sum((gaps >= 30.0).to(dtype=zero.dtype))
        horizon60 = torch.sum((gaps >= 60.0).to(dtype=zero.dtype))
    else:
        horizon14 = zero
        horizon30 = zero
        horizon60 = zero
    return unweighted, weighted, {
        f'{prefix}warm_column_heat_content_loss': unweighted.detach(),
        f'{prefix}warm_column_heat_content_weighted_loss': weighted.detach(),
        f'{prefix}warm_column_heat_content_supervision_count': supervision_count.detach(),
        f'{prefix}warm_column_heat_content_warm_factor_mean': warm_factor_mean.detach(),
        f'{prefix}warm_column_heat_content_error_c_mean': error_mean.detach(),
        f'{prefix}warm_column_heat_content_horizon14_count': horizon14.detach(),
        f'{prefix}warm_column_heat_content_horizon30_count': horizon30.detach(),
        f'{prefix}warm_column_heat_content_horizon60_count': horizon60.detach(),
    }


def _depth_limited_export_grid(temp_grid, depths, export_max_depth_m=None):
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    temp_grid = np.asarray(temp_grid, dtype=np.float32)
    if depths.size == 0:
        raise ValueError('Cannot export temperature grid with an empty depth axis.')
    if temp_grid.shape[0] != depths.size:
        raise ValueError(
            f'Export temperature grid depth axis {temp_grid.shape[0]} does not match depths length {depths.size}.'
        )
    internal_max_depth = float(np.nanmax(depths))
    if export_max_depth_m is None:
        return temp_grid, depths.astype(np.float32), {
            'export_max_depth_m': None,
            'effective_export_max_depth_m': internal_max_depth,
            'internal_max_depth_m': internal_max_depth,
            'export_depth_limited': False,
        }
    requested_max = float(export_max_depth_m)
    if (not np.isfinite(requested_max)) or requested_max <= 0.0:
        raise ValueError('export_max_depth_m must be positive when provided.')
    effective_max = min(requested_max, internal_max_depth)
    keep_mask = depths <= effective_max + 1.0e-6
    export_depths = depths[keep_mask]
    if export_depths.size == 0:
        export_depths = np.asarray([float(depths[0])], dtype=np.float64)
    if not np.any(np.isclose(export_depths, effective_max, rtol=0.0, atol=1.0e-6)):
        export_depths = np.concatenate([export_depths, np.asarray([effective_max], dtype=np.float64)])
    export_depths = np.unique(np.round(export_depths.astype(np.float64), 6))
    export_depths.sort()
    export_grid = np.empty((export_depths.size, temp_grid.shape[1]), dtype=np.float32)
    for day_idx in range(temp_grid.shape[1]):
        column = temp_grid[:, day_idx].astype(np.float64)
        valid = np.isfinite(column) & np.isfinite(depths)
        if np.count_nonzero(valid) == 0:
            export_grid[:, day_idx] = np.nan
        elif np.count_nonzero(valid) == 1:
            export_grid[:, day_idx] = float(column[valid][0])
        else:
            order = np.argsort(depths[valid])
            export_grid[:, day_idx] = np.interp(
                export_depths,
                depths[valid][order],
                column[valid][order],
            ).astype(np.float32)
    return export_grid, export_depths.astype(np.float32), {
        'export_max_depth_m': requested_max,
        'effective_export_max_depth_m': float(effective_max),
        'internal_max_depth_m': internal_max_depth,
        'export_depth_limited': bool(effective_max < internal_max_depth - 1.0e-6),
    }


def _read_metadata_override(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Metadata override not found: {path}')
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def _sync_metadata_override_to_forcing(df, override):
    """Apply sidecar metadata defaults to forcing columns when CSV values lack them."""
    if not override:
        return df
    df = df.copy()
    kd_value = override.get('light_extinction_kd', override.get('kd_m_inv'))
    if kd_value is None and override.get('secchi_m'):
        try:
            kd_value = 1.7 / max(float(override['secchi_m']), 1.0e-6)
        except (TypeError, ValueError):
            kd_value = None
    if kd_value is not None and 'light_extinction_kd' in df.columns:
        kd_value = float(kd_value)
        df['light_extinction_kd'] = kd_value

    fetch_value = override.get('effective_fetch_m', override.get('fetch_m'))
    if fetch_value is not None and 'effective_fetch' in df.columns:
        fetch_value = float(fetch_value)
        df['effective_fetch'] = fetch_value

    for metadata_key, column_name in (
        ('water_level_anomaly', 'water_level_anomaly'),
        ('net_inflow', 'net_inflow'),
        ('ice_fraction', 'ice_fraction'),
        ('snow_depth_m', 'snow_depth_m'),
        ('ice_thickness_m', 'ice_thickness_m'),
    ):
        if metadata_key in override and column_name in df.columns:
            value = float(override[metadata_key])
            current = pd.to_numeric(df[column_name], errors='coerce')
            df[column_name] = current.fillna(value)
    return df


def _refresh_conditional_prior_columns(df, metadata, max_depth):
    """Recompute conditional prior diagnostics after manifest metadata overrides."""
    df = df.copy()
    thermal_regime = infer_thermal_regime(metadata, df)
    bottom_prior_c = infer_bottom_temp_prior_c(metadata, df, max_depth=max_depth)
    metadata['thermal_regime'] = thermal_regime
    metadata['bottom_temp_prior_c'] = float(bottom_prior_c)
    df['thermal_regime'] = thermal_regime
    df['BottomTemp_prior_C'] = float(bottom_prior_c)
    if 'ice_fraction_observed' not in df.columns:
        df['ice_fraction_observed'] = 0.0
    df['ice_risk_prior'] = infer_ice_risk_prior(metadata, df)
    if 'ice_fraction' in df.columns:
        df['ice_fraction'] = np.maximum(
            pd.to_numeric(df['ice_fraction'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df['ice_risk_prior'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32),
        )

    if 'BottomTemp_imputed_by_prior_rule' in df.columns and 'BottomTemp_C' in df.columns:
        prior_mask = pd.to_numeric(df['BottomTemp_imputed_by_prior_rule'], errors='coerce').fillna(0.0) > 0.5
        if 'BottomTemp_imputed_by_4C_rule' in df.columns:
            hard_4c_mask = pd.to_numeric(df['BottomTemp_imputed_by_4C_rule'], errors='coerce').fillna(0.0) > 0.5
            if thermal_regime != 'cold_ice_prone':
                df.loc[hard_4c_mask, 'BottomTemp_imputed_by_4C_rule'] = 0.0
                prior_mask = prior_mask | hard_4c_mask
        df.loc[prior_mask, 'BottomTemp_C'] = float(bottom_prior_c)
    return df


def _daily_tendency_loss(previous, prediction, depths, max_depth):
    """Penalize implausible daily jumps without post-processing predictions."""
    delta = torch.abs(prediction - previous)
    depths = depths.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
    surface_mask = depths <= min(3.0, float(max_depth))
    mid_deep_mask = depths > min(3.0, float(max_depth))

    losses = []
    if torch.any(surface_mask):
        losses.append(torch.relu(delta[:, surface_mask] - 2.5).pow(2).mean())
    if torch.any(mid_deep_mask):
        losses.append(torch.relu(delta[:, mid_deep_mask] - 0.9).pow(2).mean())
    losses.append(0.5 * torch.relu(delta - 4.0).pow(2).mean())
    return torch.stack(losses).mean()


def _residual_regularization_loss(diagnostics):
    """Keep learned daily residual corrections small enough for stable free-roll."""
    residual_abs = diagnostics['residual_abs_mean_c']
    residual_surface = diagnostics['residual_surface_c']
    residual_deep = diagnostics['residual_deep_mean_c']
    return (
        residual_abs.pow(2).mean()
        + 0.25 * residual_surface.pow(2).mean()
        + 0.50 * residual_deep.pow(2).mean()
    )


def _physical_scale_regularization_loss(diagnostics):
    shortwave_scale = diagnostics.get('shortwave_absorption_scale')
    cooling_scale = diagnostics.get('surface_cooling_scale_raw', diagnostics.get('surface_cooling_scale'))
    flux_bias = diagnostics.get('surface_flux_bias_wm2')
    if shortwave_scale is None or cooling_scale is None:
        device = diagnostics['residual_abs_mean_c'].device
        return torch.tensor(0.0, device=device)
    loss = (
        (shortwave_scale - 1.0).pow(2).mean()
        + (cooling_scale - 1.0).pow(2).mean()
    )
    if flux_bias is not None:
        loss = loss + (flux_bias / 30.0).pow(2).mean()
    return loss


def _kd_prior_regularization_loss(diagnostics):
    kd_prior = diagnostics.get('kd_prior_regularization_loss')
    if kd_prior is None:
        device = diagnostics['residual_abs_mean_c'].device
        return torch.tensor(0.0, device=device)
    return kd_prior.reshape(-1).mean()


def _adaptive_parameter_regularization_loss(diagnostics):
    loss = diagnostics.get('adaptive_parameter_regularization_loss')
    if loss is None:
        device = diagnostics['residual_abs_mean_c'].device
        return torch.tensor(0.0, device=device)
    return loss.reshape(-1).mean()


def _physical_scale_smoothness_loss(diagnostics, previous_scales=None):
    shortwave_scale = diagnostics.get('shortwave_absorption_scale')
    cooling_scale = diagnostics.get('surface_cooling_scale_raw', diagnostics.get('surface_cooling_scale'))
    flux_bias = diagnostics.get('surface_flux_bias_wm2')
    if shortwave_scale is None or cooling_scale is None:
        device = diagnostics['residual_abs_mean_c'].device
        return torch.tensor(0.0, device=device)
    if previous_scales is None:
        return torch.tensor(0.0, device=shortwave_scale.device)
    prev_shortwave, prev_cooling, prev_flux_bias = previous_scales
    loss = (
        (shortwave_scale - prev_shortwave).pow(2).mean()
        + (cooling_scale - prev_cooling).pow(2).mean()
    )
    if flux_bias is not None and prev_flux_bias is not None:
        loss = loss + ((flux_bias - prev_flux_bias) / 30.0).pow(2).mean()
    return loss


def _current_physical_scales(diagnostics):
    return (
        diagnostics['shortwave_absorption_scale'].detach(),
        diagnostics.get('surface_cooling_scale_raw', diagnostics['surface_cooling_scale']).detach(),
        diagnostics.get('surface_flux_bias_wm2').detach() if diagnostics.get('surface_flux_bias_wm2') is not None else None,
    )


def _scale_detail_record(diagnostics):
    shortwave_scale = diagnostics['shortwave_absorption_scale']
    cooling_raw = diagnostics.get('surface_cooling_scale_raw', diagnostics['surface_cooling_scale'])
    cooling_effective = diagnostics['surface_cooling_scale']
    zeros = torch.zeros_like(shortwave_scale)
    return {
        'shortwave_scale_mean': shortwave_scale.detach().mean(),
        'shortwave_scale_min': shortwave_scale.detach().min(),
        'shortwave_scale_max': shortwave_scale.detach().max(),
        'cooling_scale_mean': cooling_raw.detach().mean(),
        'cooling_scale_min': cooling_raw.detach().min(),
        'cooling_scale_max': cooling_raw.detach().max(),
        'cooling_scale_effective_mean': cooling_effective.detach().mean(),
        'surface_flux_bias_mean_wm2': diagnostics.get('surface_flux_bias_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'open_water_sensible_heat_mean_wm2': diagnostics.get('open_water_sensible_heat_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'open_water_latent_heat_mean_wm2': diagnostics.get('open_water_latent_heat_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'temperature_floor_heat_injection_mean_wm2': diagnostics.get('temperature_floor_heat_injection_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'freezing_storage_mean_j_m2': diagnostics.get('freezing_storage_j_m2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'freezing_storage_ice_mean_j_m2': diagnostics.get('freezing_storage_ice_j_m2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'freezing_storage_surface_fraction_mean': diagnostics.get('freezing_storage_surface_fraction', torch.zeros_like(shortwave_scale)).detach().mean(),
        'freezing_storage_deep_fraction_mean': diagnostics.get('freezing_storage_deep_fraction', torch.zeros_like(shortwave_scale)).detach().mean(),
        'freezing_storage_change_mean_wm2': diagnostics.get('freezing_storage_change_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'effective_heat_tendency_mean_wm2': diagnostics.get('effective_heat_tendency_wm2', torch.zeros_like(shortwave_scale)).detach().mean(),
        'lst_feature_dropout_applied_mean': diagnostics.get('lst_feature_dropout_applied', zeros).detach().mean(),
        'advective_heat_source_c_per_day_mean': diagnostics.get('advective_heat_source_c_per_day_mean', zeros).detach().mean(),
        'advective_heat_source_c_per_day_max': diagnostics.get('advective_heat_source_c_per_day_max', zeros).detach().mean(),
        'advective_exchange_fraction_per_day': diagnostics.get('advective_exchange_fraction_per_day', zeros).detach().mean(),
        'advective_heat_source_active_mean': diagnostics.get('advective_heat_source_active_mean', zeros).detach().mean(),
        'background_nn_kz_mean': diagnostics.get('background_nn_kz_mean', zeros).detach().mean(),
        'background_nn_kz_deep_mean': diagnostics.get('background_nn_kz_deep_mean', zeros).detach().mean(),
        'turbulent_nn_kz_mean': diagnostics.get('turbulent_nn_kz_mean', zeros).detach().mean(),
        'turbulent_nn_kz_deep_mean': diagnostics.get('turbulent_nn_kz_deep_mean', zeros).detach().mean(),
        'gated_turbulent_nn_kz_mean': diagnostics.get('gated_turbulent_nn_kz_mean', zeros).detach().mean(),
        'gated_turbulent_nn_kz_deep_mean': diagnostics.get('gated_turbulent_nn_kz_deep_mean', zeros).detach().mean(),
        'kd_base_mean': diagnostics.get('kd_base', zeros).detach().mean(),
        'nn_kd_multiplier_mean': diagnostics.get('nn_kd_multiplier', zeros).detach().mean(),
        'kd_prior_regularization_loss_mean': diagnostics.get('kd_prior_regularization_loss', zeros).detach().mean(),
        'adaptive_wind_kz_scale_mean': diagnostics.get('adaptive_wind_kz_scale', torch.zeros_like(shortwave_scale)).detach().mean(),
        'adaptive_turbulent_flux_blend_alpha_mean': diagnostics.get(
            'adaptive_turbulent_flux_blend_alpha',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'adaptive_kd_multiplier_mean': diagnostics.get(
            'adaptive_kd_multiplier',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'adaptive_turbulent_exchange_scale_mean': diagnostics.get(
            'adaptive_turbulent_exchange_scale',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'adaptive_convective_mixing_scale_mean': diagnostics.get(
            'adaptive_convective_mixing_scale',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'adaptive_ice_shortwave_scale_mean': diagnostics.get(
            'adaptive_ice_shortwave_scale',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'adaptive_parameter_regularization_loss': diagnostics.get(
            'adaptive_parameter_regularization_loss',
            torch.zeros_like(shortwave_scale),
        ).detach().mean(),
        'lake_shape_wind_factor_mean': diagnostics.get(
            'lake_shape_wind_factor',
            torch.ones_like(shortwave_scale),
        ).detach().mean(),
        'lake_shape_decay_depth_mean_m': diagnostics.get(
            'lake_shape_decay_depth_m',
            torch.full_like(shortwave_scale, 5.0),
        ).detach().mean(),
        'stratification_mixing_gate_mean': diagnostics.get(
            'stratification_mixing_gate_mean',
            torch.ones_like(shortwave_scale),
        ).detach().mean(),
        'stratification_mixing_gate_min': diagnostics.get(
            'stratification_mixing_gate_min',
            torch.ones_like(shortwave_scale),
        ).detach().mean(),
        'stratification_mixing_gate_deep_mean': diagnostics.get(
            'stratification_mixing_gate_deep_mean',
            torch.ones_like(shortwave_scale),
        ).detach().mean(),
    }


def _date_index_map(df):
    return {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }


def _build_segment_rollout_sequences(df, profile_lookup, max_rollout_days=30):
    """Build split-specific segment rollout supervision sequences.

    A sequence starts from one observed profile and supervises every later
    observed profile within max_rollout_days.  Callers must pass the lookup
    they want to train on; this function intentionally never falls back to
    another split.
    """
    date_to_index = _date_index_map(df)
    dates = sorted(date for date in profile_lookup if date in date_to_index)
    sequences = []
    max_rollout_days = max(1, int(max_rollout_days))
    for start in dates:
        start_idx = date_to_index[start]
        targets = []
        for target in dates:
            target_idx = date_to_index[target]
            gap = int(target_idx - start_idx)
            if 1 <= gap <= max_rollout_days:
                targets.append((target, target_idx))
        if targets:
            sequences.append((start, start_idx, targets))
    return sequences


def _scheduled_weight(epoch, target, start_epoch, ramp_epochs):
    target = float(target)
    if target <= 0.0:
        return 0.0
    start_epoch = int(start_epoch)
    ramp_epochs = max(0, int(ramp_epochs))
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target
    progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / float(ramp_epochs)))
    return target * progress


def _scheduled_teacher_forcing_probability(epoch, start_epoch, ramp_epochs, start_probability, end_probability):
    """Scheduled sampling probability for replacing the next segment state with observation."""
    start_probability = float(start_probability)
    end_probability = float(end_probability)
    if epoch < int(start_epoch):
        return start_probability
    ramp_epochs = max(1, int(ramp_epochs))
    progress = min(1.0, max(0.0, (epoch - int(start_epoch)) / float(ramp_epochs)))
    return start_probability + progress * (end_probability - start_probability)


def _scheduled_segment_rollout_days(epoch, start_epoch, ramp_epochs, max_days):
    """Grow the active segment-rollout horizon after the warmup starts."""
    max_days = max(1, int(max_days))
    if epoch < int(start_epoch):
        return 0
    if max_days <= 14:
        return max_days
    ramp_epochs = max(1, int(ramp_epochs))
    progress = min(1.0, max(0.0, (epoch - int(start_epoch) + 1) / float(ramp_epochs)))
    return int(round(14 + progress * (max_days - 14)))


def _select_segment_rollout_sequences(sequences, active_max_days, samples_per_lake, epoch):
    active = []
    active_max_days = int(active_max_days)
    for sequence in sequences:
        start, start_idx, targets = sequence
        usable_targets = [
            (target, target_idx)
            for target, target_idx in targets
            if 1 <= int(target_idx - start_idx) <= active_max_days
        ]
        if usable_targets:
            active.append((start, start_idx, usable_targets))
    if not active:
        return []
    samples_per_lake = int(samples_per_lake)
    if samples_per_lake <= 0 or len(active) <= samples_per_lake:
        return active
    active = sorted(active, key=lambda item: int(item[1]))
    starts = np.asarray([int(item[1]) for item in active], dtype=np.float64)
    if samples_per_lake == 1:
        return [active[int(epoch) % len(active)]]
    anchors = np.linspace(float(starts[0]), float(starts[-1]), num=samples_per_lake)
    boundaries = np.empty(samples_per_lake + 1, dtype=np.float64)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    if samples_per_lake > 1:
        boundaries[1:-1] = (anchors[:-1] + anchors[1:]) * 0.5
    selected = []
    selected_indices = set()
    for bucket, anchor in enumerate(anchors):
        left = boundaries[bucket]
        right = boundaries[bucket + 1]
        candidates = [
            idx for idx, start_idx in enumerate(starts)
            if idx not in selected_indices and left <= start_idx < right
        ]
        if not candidates:
            candidates = [idx for idx in range(len(active)) if idx not in selected_indices]
        candidates = sorted(
            candidates,
            key=lambda idx: (abs(float(starts[idx]) - float(anchor)), float(starts[idx])),
        )
        chosen = candidates[int(epoch) % len(candidates)]
        selected_indices.add(chosen)
        selected.append(active[chosen])
    return selected


def _segment_rollout_last_gap(sequence):
    _start, start_idx, targets = sequence
    return max((int(target_idx - start_idx) for _target, target_idx in targets), default=0)


def _segment_rollout_grouped_batches(sequences, batch_size):
    grouped = {}
    for sequence in sequences:
        grouped.setdefault(_segment_rollout_last_gap(sequence), []).append(sequence)
    batches = []
    for last_gap in sorted(grouped):
        batches.extend(tuple(chunk) for chunk in _batch_chunks(grouped[last_gap], batch_size))
    return tuple(batches)


def _build_segment_rollout_epoch_plan(
    sequences,
    *,
    epochs,
    start_epoch,
    ramp_epochs,
    max_days,
    samples_per_lake,
    segment_rollout_batch_size=0,
):
    plan = []
    for epoch in range(max(0, int(epochs))):
        active_max_days = _scheduled_segment_rollout_days(
            epoch,
            start_epoch,
            ramp_epochs,
            max_days,
        )
        selected = (
            _select_segment_rollout_sequences(
                sequences,
                active_max_days,
                samples_per_lake,
                epoch,
            )
            if active_max_days > 0 else []
        )
        selected = tuple(selected)
        plan.append({
            'epoch': int(epoch),
            'active_max_days': int(active_max_days),
            'samples_per_lake': int(samples_per_lake),
            'selected_sequences': selected,
            'batches': _segment_rollout_grouped_batches(
                selected,
                segment_rollout_batch_size,
            ),
        })
    return tuple(plan)


def _segment_rollout_plan_entry_for_epoch(lake, split_key, active_max_days, samples_per_lake, epoch):
    plan = lake.get('segment_rollout_epoch_plans', {}).get(split_key)
    if not plan:
        return None
    epoch = int(epoch)
    if epoch < 0 or epoch >= len(plan):
        return None
    entry = plan[epoch]
    if int(entry.get('active_max_days', -1)) != int(active_max_days):
        return None
    if int(entry.get('samples_per_lake', -1)) != int(samples_per_lake):
        return None
    return entry


def _segment_rollout_sequences_for_epoch(lake, split_key, active_max_days, samples_per_lake, epoch):
    entry = _segment_rollout_plan_entry_for_epoch(
        lake,
        split_key,
        active_max_days,
        samples_per_lake,
        epoch,
    )
    if entry is not None:
        return list(entry['selected_sequences'])
    return _select_segment_rollout_sequences(
        lake['segment_rollout_sequences'][split_key],
        active_max_days,
        samples_per_lake,
        epoch,
    )


def _segment_rollout_batches_for_epoch(lake, split_key, active_max_days, samples_per_lake, epoch):
    entry = _segment_rollout_plan_entry_for_epoch(
        lake,
        split_key,
        active_max_days,
        samples_per_lake,
        epoch,
    )
    return None if entry is None else entry['batches']


def _build_cross_lake_segment_rollout_epoch_batches(
    lakes,
    *,
    split_key,
    epochs,
    segment_rollout_batch_size=0,
    cross_lake_batch_size=0,
):
    epoch_batches = []
    batch_size = int(cross_lake_batch_size or segment_rollout_batch_size or 0)
    for epoch in range(max(0, int(epochs))):
        grouped = {}
        for bucket_entries in _cross_lake_bucket_entries(lakes).values():
            for lake_idx, lake in bucket_entries:
                plan = lake.get('segment_rollout_epoch_plans', {}).get(split_key, ())
                if epoch >= len(plan):
                    continue
                for sequence in plan[epoch]['selected_sequences']:
                    last_gap = _segment_rollout_last_gap(sequence)
                    grouped.setdefault((_cross_lake_batch_key(lake), last_gap), []).append((
                        lake_idx,
                        lake,
                        sequence,
                    ))
        chunks = []
        for key in sorted(grouped, key=lambda value: (value[0][0], value[1])):
            chunks.extend(tuple(chunk) for chunk in _batch_chunks(grouped[key], batch_size))
        epoch_batches.append(tuple(chunks))
    return tuple(epoch_batches)


def _detail_values(details, key):
    values = [record[key].detach().reshape(-1).mean().cpu().item() for record in details if key in record]
    return values


def _mean_detail(details, key):
    values = _detail_values(details, key)
    return float(np.mean(values)) if values else np.nan


def _min_detail(details, key):
    values = _detail_values(details, key)
    return float(np.min(values)) if values else np.nan


def _max_detail(details, key):
    values = _detail_values(details, key)
    return float(np.max(values)) if values else np.nan


def _filter_state_forecaster_state_dict_for_load(model, state_dict):
    """Keep grid buffers local and skip legacy adaptive-head tensors by shape."""
    current_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key in {'depths', 'area_profile'}:
            continue
        if (
            key.startswith('lake_adaptive_head.')
            and key in current_state
            and tuple(current_state[key].shape) != tuple(value.shape)
        ):
            continue
        filtered[key] = value
    return filtered


def _lake_season_factor_stat(lakes, month, reducer):
    values = [
        float(lake['heat_content_transition_season_factors'][month])
        for lake in lakes
        if 'heat_content_transition_season_factors' in lake
    ]
    return float(reducer(values)) if values else np.nan


def _mean_numeric_records(records, key):
    values = [record.get(key, np.nan) for record in records]
    return _nanmean_or_nan(values)


def _gpu_profile_snapshot(device):
    record = {
        'cuda_available': bool(torch.cuda.is_available()),
        'gpu_name': '',
        'gpu_util_percent': np.nan,
        'gpu_memory_used_mb': np.nan,
        'gpu_memory_total_mb': np.nan,
        'torch_memory_allocated_mb': np.nan,
        'torch_memory_reserved_mb': np.nan,
    }
    if not torch.cuda.is_available():
        return record
    cuda_device = torch.device(device if device is not None else 'cuda')
    if cuda_device.type != 'cuda':
        return record
    index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    record['gpu_name'] = torch.cuda.get_device_name(index)
    record['torch_memory_allocated_mb'] = float(torch.cuda.memory_allocated(index) / (1024.0 ** 2))
    record['torch_memory_reserved_mb'] = float(torch.cuda.memory_reserved(index) / (1024.0 ** 2))
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={int(index)}',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        parts = [part.strip() for part in result.stdout.strip().split(',')]
        if len(parts) >= 3:
            record['gpu_util_percent'] = float(parts[0])
            record['gpu_memory_used_mb'] = float(parts[1])
            record['gpu_memory_total_mb'] = float(parts[2])
    except Exception:
        pass
    return record


def _tensorize_profile_lookup(lookup, *, device):
    return {
        pd.Timestamp(date).normalize(): torch.tensor(
            profile,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        for date, profile in lookup.items()
    }


def _tensorize_mask_lookup(masks, *, device):
    return {
        pd.Timestamp(date).normalize(): torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=device,
        ).reshape(1, -1)
        for date, mask in masks.items()
    }


def _forcing_tensor_matrix(forcing_rows):
    """Stack forcing row dicts once so training can batch by day index."""
    if not forcing_rows:
        return {}
    matrix = {}
    for key in forcing_rows[0]:
        values = [row[key] for row in forcing_rows if key in row]
        if len(values) != len(forcing_rows):
            continue
        stacked = torch.stack(values, dim=0)
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        matrix[key] = stacked
    return matrix


def _forcing_batch_view(lake, indices):
    device = lake['depths'].device
    if torch.is_tensor(indices):
        index_tensor = indices.to(device=device, dtype=torch.long).reshape(-1)
        if int(index_tensor.numel()) == 0:
            raise ValueError('forcing batch indices must not be empty.')
    else:
        index_values = [int(idx) for idx in indices]
        if not index_values:
            raise ValueError('forcing batch indices must not be empty.')
        index_tensor = torch.as_tensor(index_values, dtype=torch.long, device=device)
    matrix = lake.get('forcing_tensors')
    if not matrix:
        raise ValueError('forcing tensor cache is not available for this lake.')
    return ForcingBatch({key: value.index_select(0, index_tensor) for key, value in matrix.items()})


def _forcing_row_batch(lake, indices, *, step_forcing_mode=None):
    mode = str(step_forcing_mode or lake.get('step_forcing_mode', 'auto')).strip().lower()
    if mode not in {'auto', 'dict', 'tensor'}:
        raise ValueError('step_forcing_mode must be one of: auto, dict, tensor.')
    if mode in {'auto', 'tensor'} and lake.get('forcing_tensors'):
        return _forcing_batch_view(lake, indices)
    if mode == 'tensor':
        raise ValueError('step_forcing_mode=tensor requires forcing tensor cache.')
    if torch.is_tensor(indices):
        index_list = indices.detach().cpu().reshape(-1).tolist()
    else:
        index_list = [int(idx) for idx in indices]
    if not index_list:
        raise ValueError('forcing batch indices must not be empty.')
    rows = [lake['forcing_rows'][idx] for idx in index_list]
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(rows):
            continue
        stacked = torch.stack(values, dim=0)
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        batch[key] = stacked
    return batch


def _stack_forcing_rows(rows):
    if not rows:
        raise ValueError('forcing rows must not be empty.')
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(rows):
            continue
        stacked = torch.stack(values, dim=0)
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        batch[key] = stacked
    return batch


def _target_tensor_and_mask_batch(lake, preferred_split, dates):
    profiles = []
    masks = []
    for date_value in dates:
        profile, mask = _target_tensor_and_mask(lake, preferred_split, date_value)
        profiles.append(profile)
        if mask is None:
            masks.append(torch.ones_like(profile, dtype=torch.bool))
        else:
            masks.append(mask.to(device=profile.device, dtype=torch.bool))
    if not profiles:
        raise ValueError('target batch dates must not be empty.')
    return torch.cat(profiles, dim=0), torch.cat(masks, dim=0)


def _masked_huber_profile_loss_per_sample(prediction, target, mask=None, delta=2.0):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.reshape(1, -1).expand_as(valid)
        valid = valid & mask_tensor.reshape_as(valid)
    loss = torch.nn.functional.huber_loss(
        prediction,
        target,
        delta=float(delta),
        reduction='none',
    )
    valid_float = valid.to(dtype=prediction.dtype)
    counts = valid_float.sum(dim=1)
    summed = (loss * valid_float).sum(dim=1)
    return torch.where(
        counts > 0,
        summed / torch.clamp(counts, min=1.0),
        torch.zeros_like(summed),
    )


def _profile_physics_loss_per_sample(profile):
    return torch.stack([
        _profile_physics_loss(profile[idx: idx + 1])
        for idx in range(profile.shape[0])
    ])


def _daily_tendency_loss_per_sample(previous, prediction, depths, max_depth):
    delta = torch.abs(prediction - previous)
    depths = depths.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
    surface_mask = depths <= min(3.0, float(max_depth))
    mid_deep_mask = depths > min(3.0, float(max_depth))
    losses = []
    if torch.any(surface_mask):
        losses.append(torch.relu(delta[:, surface_mask] - 2.5).pow(2).mean(dim=1))
    if torch.any(mid_deep_mask):
        losses.append(torch.relu(delta[:, mid_deep_mask] - 0.9).pow(2).mean(dim=1))
    losses.append(0.5 * torch.relu(delta - 4.0).pow(2).mean(dim=1))
    return torch.stack(losses, dim=0).mean(dim=0)


def _residual_regularization_loss_per_sample(diagnostics):
    residual_abs = diagnostics['residual_abs_mean_c'].reshape(-1)
    residual_surface = diagnostics['residual_surface_c'].reshape(-1)
    residual_deep = diagnostics['residual_deep_mean_c'].reshape(-1)
    return (
        residual_abs.pow(2)
        + 0.25 * residual_surface.pow(2)
        + 0.50 * residual_deep.pow(2)
    )


def _physical_scale_regularization_loss_per_sample(diagnostics):
    shortwave_scale = diagnostics.get('shortwave_absorption_scale')
    cooling_scale = diagnostics.get('surface_cooling_scale_raw', diagnostics.get('surface_cooling_scale'))
    flux_bias = diagnostics.get('surface_flux_bias_wm2')
    if shortwave_scale is None or cooling_scale is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    loss = (shortwave_scale.reshape(-1) - 1.0).pow(2) + (cooling_scale.reshape(-1) - 1.0).pow(2)
    if flux_bias is not None:
        loss = loss + (flux_bias.reshape(-1) / 30.0).pow(2)
    return loss


def _kd_prior_regularization_loss_per_sample(diagnostics):
    kd_prior = diagnostics.get('kd_prior_regularization_loss')
    if kd_prior is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    return kd_prior.reshape(-1)


def _adaptive_parameter_regularization_loss_per_sample(diagnostics):
    loss = diagnostics.get('adaptive_parameter_regularization_loss')
    if loss is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    return loss.reshape(-1)


def _physical_scale_smoothness_loss_per_sample(diagnostics, previous_scales=None):
    shortwave_scale = diagnostics.get('shortwave_absorption_scale')
    cooling_scale = diagnostics.get('surface_cooling_scale_raw', diagnostics.get('surface_cooling_scale'))
    flux_bias = diagnostics.get('surface_flux_bias_wm2')
    if shortwave_scale is None or cooling_scale is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    if previous_scales is None:
        return torch.zeros_like(shortwave_scale.reshape(-1))
    prev_shortwave, prev_cooling, prev_flux_bias = previous_scales
    loss = (
        (shortwave_scale.reshape(-1) - prev_shortwave.reshape(-1)).pow(2)
        + (cooling_scale.reshape(-1) - prev_cooling.reshape(-1)).pow(2)
    )
    if flux_bias is not None and prev_flux_bias is not None:
        loss = loss + ((flux_bias.reshape(-1) - prev_flux_bias.reshape(-1)) / 30.0).pow(2)
    return loss


def _residual_profile_smoothness_loss_per_sample(diagnostics, previous_residual=None):
    residual = diagnostics.get('residual_profile_c')
    if residual is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    losses = []
    if residual.shape[1] > 1:
        losses.append(torch.diff(residual, dim=1).pow(2).mean(dim=1))
    if previous_residual is not None:
        losses.append((residual - previous_residual).pow(2).mean(dim=1))
    if not losses:
        return torch.zeros(residual.shape[0], device=residual.device, dtype=residual.dtype)
    return torch.stack(losses, dim=0).mean(dim=0)


def _current_physical_scales_detached(diagnostics):
    return (
        diagnostics['shortwave_absorption_scale'].detach(),
        diagnostics.get('surface_cooling_scale_raw', diagnostics['surface_cooling_scale']).detach(),
        diagnostics.get('surface_flux_bias_wm2').detach()
        if diagnostics.get('surface_flux_bias_wm2') is not None else None,
    )


def _scale_detail_record_for_sample(diagnostics, sample_idx):
    shortwave_scale = diagnostics['shortwave_absorption_scale'].reshape(-1)
    cooling_raw = diagnostics.get('surface_cooling_scale_raw', diagnostics['surface_cooling_scale']).reshape(-1)
    cooling_effective = diagnostics['surface_cooling_scale'].reshape(-1)
    zeros = torch.zeros_like(shortwave_scale)
    return {
        'shortwave_scale_mean': shortwave_scale[sample_idx].detach(),
        'shortwave_scale_min': shortwave_scale[sample_idx].detach(),
        'shortwave_scale_max': shortwave_scale[sample_idx].detach(),
        'cooling_scale_mean': cooling_raw[sample_idx].detach(),
        'cooling_scale_min': cooling_raw[sample_idx].detach(),
        'cooling_scale_max': cooling_raw[sample_idx].detach(),
        'cooling_scale_effective_mean': cooling_effective[sample_idx].detach(),
        'surface_flux_bias_mean_wm2': diagnostics.get('surface_flux_bias_wm2', zeros).reshape(-1)[sample_idx].detach(),
        'open_water_sensible_heat_mean_wm2': diagnostics.get('open_water_sensible_heat_wm2', zeros).reshape(-1)[sample_idx].detach(),
        'open_water_latent_heat_mean_wm2': diagnostics.get('open_water_latent_heat_wm2', zeros).reshape(-1)[sample_idx].detach(),
        'temperature_floor_heat_injection_mean_wm2': diagnostics.get(
            'temperature_floor_heat_injection_wm2',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'freezing_storage_mean_j_m2': diagnostics.get(
            'freezing_storage_j_m2',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'freezing_storage_ice_mean_j_m2': diagnostics.get(
            'freezing_storage_ice_j_m2',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'freezing_storage_surface_fraction_mean': diagnostics.get(
            'freezing_storage_surface_fraction',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'freezing_storage_deep_fraction_mean': diagnostics.get(
            'freezing_storage_deep_fraction',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'freezing_storage_change_mean_wm2': diagnostics.get(
            'freezing_storage_change_wm2',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'lst_feature_dropout_applied_mean': diagnostics.get(
            'lst_feature_dropout_applied',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'advective_heat_source_c_per_day_mean': diagnostics.get(
            'advective_heat_source_c_per_day_mean',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'advective_heat_source_c_per_day_max': diagnostics.get(
            'advective_heat_source_c_per_day_max',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'advective_exchange_fraction_per_day': diagnostics.get(
            'advective_exchange_fraction_per_day',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'advective_heat_source_active_mean': diagnostics.get(
            'advective_heat_source_active_mean',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'background_nn_kz_mean': diagnostics.get('background_nn_kz_mean', zeros).reshape(-1)[sample_idx].detach(),
        'background_nn_kz_deep_mean': diagnostics.get('background_nn_kz_deep_mean', zeros).reshape(-1)[sample_idx].detach(),
        'turbulent_nn_kz_mean': diagnostics.get('turbulent_nn_kz_mean', zeros).reshape(-1)[sample_idx].detach(),
        'turbulent_nn_kz_deep_mean': diagnostics.get('turbulent_nn_kz_deep_mean', zeros).reshape(-1)[sample_idx].detach(),
        'gated_turbulent_nn_kz_mean': diagnostics.get('gated_turbulent_nn_kz_mean', zeros).reshape(-1)[sample_idx].detach(),
        'gated_turbulent_nn_kz_deep_mean': diagnostics.get('gated_turbulent_nn_kz_deep_mean', zeros).reshape(-1)[sample_idx].detach(),
        'kd_base_mean': diagnostics.get('kd_base', zeros).reshape(-1)[sample_idx].detach(),
        'nn_kd_multiplier_mean': diagnostics.get('nn_kd_multiplier', zeros).reshape(-1)[sample_idx].detach(),
        'kd_prior_regularization_loss_mean': diagnostics.get(
            'kd_prior_regularization_loss',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_wind_kz_scale_mean': diagnostics.get(
            'adaptive_wind_kz_scale',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_turbulent_flux_blend_alpha_mean': diagnostics.get(
            'adaptive_turbulent_flux_blend_alpha',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_kd_multiplier_mean': diagnostics.get(
            'adaptive_kd_multiplier',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_turbulent_exchange_scale_mean': diagnostics.get(
            'adaptive_turbulent_exchange_scale',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_convective_mixing_scale_mean': diagnostics.get(
            'adaptive_convective_mixing_scale',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_ice_shortwave_scale_mean': diagnostics.get(
            'adaptive_ice_shortwave_scale',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'adaptive_parameter_regularization_loss': diagnostics.get(
            'adaptive_parameter_regularization_loss',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
    }


def _mean_or_zero(values, *, device):
    return torch.stack(values).mean() if values else torch.tensor(0.0, device=device)


def _forcing_value_vector(row, key, batch_size, *, device, dtype, default=0.0):
    value = row.get(key)
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 0:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    if tensor.numel() == 1 and batch_size > 1:
        return tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f'{key} must be scalar or match prediction batch size.')
    return tensor


def _segment_open_water_lst_loss_per_sample(prediction, next_row):
    batch_size = int(prediction.shape[0])
    device = prediction.device
    dtype = prediction.dtype
    target = _forcing_value_vector(
        next_row,
        'lswt_open_water',
        batch_size,
        device=device,
        dtype=dtype,
        default=float('nan'),
    )
    quality = _forcing_value_vector(
        next_row,
        'lst_quality',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    observed_flag = _forcing_value_vector(
        next_row,
        'lst_observed_flag',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    ice_mask = _forcing_value_vector(
        next_row,
        'ice_mask',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    target_valid = torch.isfinite(target)
    target_safe = torch.where(target_valid, target, prediction[:, 0].detach())
    lst_weight = torch.clamp(
        torch.clamp(quality, 0.0, 1.0)
        * torch.clamp(observed_flag, 0.0, 1.0)
        * (1.0 - torch.clamp(ice_mask, 0.0, 1.0))
        * target_valid.to(dtype=dtype),
        0.0,
        1.0,
    )
    loss_vec = lst_weight * torch.nn.functional.huber_loss(
        prediction[:, 0],
        target_safe,
        delta=2.0,
        reduction='none',
    )
    return loss_vec, lst_weight, lst_weight > 0.0


def _batch_chunks(items, batch_size):
    batch_size = int(batch_size or 0)
    if batch_size <= 0:
        yield list(items)
        return
    for start in range(0, len(items), batch_size):
        yield list(items[start: start + batch_size])


def _normalize_on_off(value, *, name):
    text = str(value or 'off').strip().lower()
    if text not in {'off', 'on'}:
        raise ValueError(f'{name} must be one of: off, on.')
    return text


def _normalize_bool_flag(value, *, name, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean or one of: off, on.')


def _normalize_torch_matmul_precision(value):
    text = str(value or 'high').strip().lower()
    if text not in {'highest', 'high', 'medium'}:
        raise ValueError('torch_matmul_precision must be one of: highest, high, medium.')
    return text


def _apply_torch_runtime_config(*, device, torch_tf32, torch_matmul_precision):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return
    if hasattr(torch, 'set_float32_matmul_precision'):
        precision = torch_matmul_precision if torch_tf32 == 'on' else 'highest'
        torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = torch_tf32 == 'on'
    torch.backends.cudnn.allow_tf32 = torch_tf32 == 'on'


def _normalize_profile_supervision_scope(value):
    text = str(value or 'train').strip().lower()
    if text not in {'train', 'all'}:
        raise ValueError("profile_supervision_scope must be one of: train, all.")
    return text


def _state_noise_like(profile, depths, weight):
    """Small training-only perturbation for closed-loop robustness."""
    weight = float(weight)
    if weight <= 0.0:
        return torch.zeros_like(profile)
    depths = depths.to(device=profile.device, dtype=profile.dtype).reshape(1, -1)
    # Surface noise is intentionally larger than deep noise.
    std = 0.04 + 0.12 * torch.exp(-depths / 3.0)
    return torch.randn_like(profile) * std * weight


def _residual_profile_smoothness_loss(diagnostics, previous_residual=None):
    residual = diagnostics.get('residual_profile_c')
    if residual is None:
        return torch.tensor(0.0, device=diagnostics['residual_abs_mean_c'].device)
    losses = []
    if residual.shape[1] > 1:
        losses.append(torch.diff(residual, dim=1).pow(2).mean())
    if previous_residual is not None:
        losses.append((residual - previous_residual).pow(2).mean())
    if not losses:
        return torch.tensor(0.0, device=residual.device)
    return torch.stack(losses).mean()


def _horizon_metric_record(errors_by_horizon, biases_by_horizon=None, prefix='rmse'):
    record = {}
    for horizon, values in errors_by_horizon.items():
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        key = f'{prefix}_{int(horizon)}d'
        record[key] = float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else np.nan
        record[f'count_{int(horizon)}d'] = int(np.sum(finite))
        if biases_by_horizon is not None:
            bias_values = np.asarray(biases_by_horizon.get(horizon, []), dtype=np.float64)
            bias_finite = np.isfinite(bias_values)
            record[f'bias_{int(horizon)}d'] = float(np.mean(bias_values[bias_finite])) if np.any(bias_finite) else np.nan
    return record


def _metric_summary_from_errors(errors):
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'count': 0}
    return {
        'rmse': float(np.sqrt(np.mean(values ** 2))),
        'mae': float(np.mean(np.abs(values))),
        'bias': float(np.mean(values)),
        'count': int(values.size),
    }


def _empty_observed_point_metrics():
    payload = {
        'observed_point_rmse': np.nan,
        'observed_point_mae': np.nan,
        'observed_point_bias': np.nan,
        'observed_point_count': 0,
        'observed_point_profile_count': 0,
    }
    for name in ('surface', 'mid', 'deep', 'winter', 'spring', 'summer', 'fall'):
        payload[f'observed_point_{name}_rmse'] = np.nan
        payload[f'observed_point_{name}_bias'] = np.nan
        payload[f'observed_point_{name}_count'] = 0
    return payload


def _season_name(month):
    month = int(month)
    if month in (12, 1, 2):
        return 'winter'
    if month in (3, 4, 5):
        return 'spring'
    if month in (6, 7, 8):
        return 'summer'
    return 'fall'


def _observed_point_metrics_from_predictions(lake, predictions_by_index):
    """Score daily profile predictions at the original observation depths."""
    profile_obs_source = lake.get('profile_obs_path')
    if not profile_obs_source or not predictions_by_index:
        return _empty_observed_point_metrics()
    try:
        obs = load_optional_profile_observations(
            profile_obs_source,
            start_date=lake['metadata'].get('start_date', lake['df']['Date'].iloc[0]),
            time_scale_seconds=lake['metadata'].get('time_scale_seconds', SECONDS_PER_DAY),
            max_depth=lake['max_depth'],
        )
    except Exception:
        return _empty_observed_point_metrics()
    if obs.empty:
        return _empty_observed_point_metrics()

    date_to_index = _date_index_map(lake['df'])
    model_depths = np.asarray(lake['depths_np'], dtype=np.float64)
    max_depth = float(lake.get('max_depth', np.nan))
    if not np.isfinite(max_depth) or max_depth <= 0.0:
        max_depth = float(np.nanmax(model_depths)) if model_depths.size else 0.0
    surface_limit = min(3.0, max_depth) if max_depth > 0.0 else 3.0
    deep_limit = 0.7 * max_depth if max_depth > 0.0 else np.inf

    errors = []
    profile_dates = set()
    grouped_errors = {
        'surface': [],
        'mid': [],
        'deep': [],
        'winter': [],
        'spring': [],
        'summer': [],
        'fall': [],
    }
    for date_value, obs_day in obs.groupby('Date'):
        date_value = pd.Timestamp(date_value).normalize()
        day_idx = date_to_index.get(date_value)
        if day_idx is None or day_idx not in predictions_by_index:
            continue
        prediction = np.asarray(predictions_by_index[day_idx], dtype=np.float64).reshape(-1)
        valid_prediction = np.isfinite(model_depths) & np.isfinite(prediction)
        if not np.any(valid_prediction):
            continue
        pred_depths = model_depths[valid_prediction]
        pred_values = prediction[valid_prediction]
        obs_depths = pd.to_numeric(obs_day['Depth_m'], errors='coerce').to_numpy(dtype=np.float64)
        obs_values = pd.to_numeric(obs_day['Temperature_C'], errors='coerce').to_numpy(dtype=np.float64)
        valid_obs = np.isfinite(obs_depths) & np.isfinite(obs_values)
        if not np.any(valid_obs):
            continue
        obs_depths = obs_depths[valid_obs]
        obs_values = obs_values[valid_obs]
        if pred_depths.size == 1:
            interpolated = np.full_like(obs_values, pred_values[0], dtype=np.float64)
        else:
            order = np.argsort(pred_depths)
            interpolated = np.interp(obs_depths, pred_depths[order], pred_values[order])
        diff = interpolated - obs_values
        finite = np.isfinite(diff)
        if not np.any(finite):
            continue
        diff = diff[finite]
        obs_depths = obs_depths[finite]
        errors.extend(diff.tolist())
        profile_dates.add(date_value)
        grouped_errors[_season_name(date_value.month)].extend(diff.tolist())
        for error_value, depth_value in zip(diff, obs_depths):
            if depth_value <= surface_limit:
                grouped_errors['surface'].append(float(error_value))
            elif depth_value >= deep_limit:
                grouped_errors['deep'].append(float(error_value))
            else:
                grouped_errors['mid'].append(float(error_value))

    summary = _metric_summary_from_errors(errors)
    payload = {
        'observed_point_rmse': summary['rmse'],
        'observed_point_mae': summary['mae'],
        'observed_point_bias': summary['bias'],
        'observed_point_count': summary['count'],
        'observed_point_profile_count': int(len(profile_dates)),
    }
    for name, values in grouped_errors.items():
        group_summary = _metric_summary_from_errors(values)
        payload[f'observed_point_{name}_rmse'] = group_summary['rmse']
        payload[f'observed_point_{name}_bias'] = group_summary['bias']
        payload[f'observed_point_{name}_count'] = group_summary['count']
    return payload


def prepare_lake_state_data(
    lake_config,
    *,
    split_mode='time_blocked',
    task_mode='analysis',
    data_fill_mode='reconstruction',
    depth_points=40,
    max_rollout_days=45,
    segment_rollout_max_days=None,
    history_window_days=30,
    device='cpu',
):
    """Load one lake and build state-transition training/evaluation pairs."""
    if task_mode is None:
        task_mode = lake_config.get('task_mode')
    task_mode = normalize_task_mode(task_mode)
    data_fill_mode = normalize_data_fill_mode(data_fill_mode)
    lake_id = str(lake_config.get('lake_id') or lake_config.get('id') or 'lake')
    era5_path = lake_config['era5']
    lst_path = lake_config['lst']
    profile_path = lake_config.get('profile_obs') or lake_config.get('profile')

    df, metadata = load_training_frame(era5_path, lst_path, data_fill_mode=data_fill_mode)
    metadata_override = _read_metadata_override(lake_config.get('metadata'))
    metadata.update(metadata_override)
    df = _sync_metadata_override_to_forcing(df, metadata_override)
    metadata['lake_id'] = lake_id
    metadata['task_mode'] = task_mode
    if lake_config.get('year') is not None:
        metadata['year'] = lake_config.get('year')
    if lake_config.get('lake_group') is not None:
        metadata['lake_group'] = lake_config.get('lake_group')
    max_depth = float(lake_config.get('max_depth') or metadata.get('max_depth_m') or 20.0)
    metadata['runtime_max_depth_m'] = max_depth
    metadata['max_depth_m'] = max_depth
    df = _refresh_conditional_prior_columns(df, metadata, max_depth)

    depths = build_depth_grid(max_depth=max_depth, n_depth_points=depth_points, use_shallow_optimized=False)
    area = fallback_area_profile(depths, metadata=metadata)
    profile_obs = load_optional_profile_observations(
        profile_path,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    splits, split_info = split_profile_observations(profile_obs, split_mode=split_mode)
    train_lookup, train_masks = _profile_lookup(splits['train'], depths, return_masks=True)
    val_lookup, val_masks = _profile_lookup(splits['val'], depths, return_masks=True)
    all_lookup, all_masks = _profile_lookup(profile_obs, depths, return_masks=True)
    train_pairs = _build_rollout_pairs(df, train_lookup, max_rollout_days=max_rollout_days)
    val_pairs = _build_rollout_pairs(df, val_lookup, max_rollout_days=max_rollout_days)
    all_pairs = _build_rollout_pairs(df, all_lookup, max_rollout_days=max_rollout_days)
    segment_rollout_max_days = int(segment_rollout_max_days or max_rollout_days)
    train_segment_sequences = _build_segment_rollout_sequences(
        df,
        train_lookup,
        max_rollout_days=segment_rollout_max_days,
    )
    all_segment_sequences = _build_segment_rollout_sequences(
        df,
        all_lookup,
        max_rollout_days=segment_rollout_max_days,
    )
    forcing_rows = _forcing_tensor_rows(
        df,
        device=device,
        history_window_days=history_window_days,
        task_mode=task_mode,
    )
    depths_tensor = torch.tensor(depths, dtype=torch.float32, device=device)
    area_tensor = torch.tensor(area, dtype=torch.float32, device=device)
    heat_content_layer_weights = _heat_content_layer_weights(
        depths_tensor,
        area_tensor,
        device=depths_tensor.device,
        dtype=depths_tensor.dtype,
    )

    return {
        'lake_id': lake_id,
        'df': df,
        'metadata': metadata,
        'max_depth': max_depth,
        'depths_np': depths,
        'area_np': area,
        'depths': depths_tensor,
        'area': area_tensor,
        'heat_content_layer_weights': heat_content_layer_weights,
        'forcing_rows': forcing_rows,
        'forcing_tensors': _forcing_tensor_matrix(forcing_rows),
        'static_features': torch.tensor(static_feature_array(metadata, max_depth), dtype=torch.float32, device=device),
        'lookups': {
            'train': train_lookup,
            'val': val_lookup,
            'all': all_lookup,
        },
        'lookup_tensors': {
            'train': _tensorize_profile_lookup(train_lookup, device=device),
            'val': _tensorize_profile_lookup(val_lookup, device=device),
            'all': _tensorize_profile_lookup(all_lookup, device=device),
        },
        'lookup_masks': {
            'train': train_masks,
            'val': val_masks,
            'all': all_masks,
        },
        'lookup_mask_tensors': {
            'train': _tensorize_mask_lookup(train_masks, device=device),
            'val': _tensorize_mask_lookup(val_masks, device=device),
            'all': _tensorize_mask_lookup(all_masks, device=device),
        },
        'date_to_index': _date_index_map(df),
        'pairs': {
            'train': train_pairs,
            'val': val_pairs,
            'all': all_pairs,
        },
        'segment_rollout_sequences': {
            'train': train_segment_sequences,
            'all': all_segment_sequences,
        },
        'split_info': split_info,
        'profile_obs_path': profile_path,
    }


def _segment_rollout_sequence_loss(
    model,
    lake,
    sequence,
    *,
    active_max_days,
    profile_huber_delta=2.0,
    task_mode='analysis',
    teacher_forcing_probability=0.0,
    state_noise_weight=0.0,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    residual_time_smooth_weight=0.01,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_season_factors=None,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    segment_rollout_lst_surface_weight=0.01,
    warm_season_column_heat_content_weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    warm_season_column_heat_content_min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    """Segment rollout loss with selected-scope observations inside the rollout window.

    The model is initialized once from the start profile, then rolled forward
    daily without automatic reset.  Loss is evaluated at every selected profile
    encountered in the segment.  Scheduled sampling may replace the *next*
    state with an observed profile after computing the prediction loss.
    """
    start, start_idx, targets = sequence
    device = lake['depths'].device
    active_targets = {
        target_idx: target
        for target, target_idx in targets
        if 1 <= int(target_idx - start_idx) <= int(active_max_days)
    }
    if not active_targets:
        zero = torch.tensor(0.0, device=device)
        return zero, 0, {
            'segment_rollout_loss': zero,
            'segment_rollout_profile_loss': zero,
            'segment_rollout_horizon_weight_mean': zero,
            'segment_rollout_max_target_gap_days': zero,
            'segment_rollout_lst_loss': zero,
            'segment_rollout_lst_supervision_count': zero,
            'segment_rollout_lst_weight_mean': zero,
            'segment_rollout_lst_surface_weight': zero,
            'segment_rollout_residual_smooth_loss': zero,
            'segment_rollout_daily_tendency_loss': zero,
            'segment_rollout_residual_regularization_loss': zero,
            'segment_rollout_physical_scale_regularization_loss': zero,
            'segment_rollout_physical_scale_smoothness_loss': zero,
            'segment_rollout_kd_prior_regularization_loss': zero,
            'segment_rollout_kd_prior_regularization_weighted_loss': zero,
            'segment_rollout_adaptive_parameter_regularization_loss': zero,
            'segment_rollout_heat_content_transition_loss': zero,
            'segment_rollout_heat_content_transition_weighted_loss': zero,
            'segment_rollout_heat_content_transition_effective_weight_mean': zero,
            'segment_rollout_heat_content_transition_effective_weight_min': zero,
            'segment_rollout_heat_content_transition_effective_weight_max': zero,
            'segment_rollout_warm_column_heat_content_loss': zero,
            'segment_rollout_warm_column_heat_content_weighted_loss': zero,
            'segment_rollout_warm_column_heat_content_supervision_count': zero,
            'segment_rollout_warm_column_heat_content_warm_factor_mean': zero,
            'segment_rollout_warm_column_heat_content_error_c_mean': zero,
            'segment_rollout_warm_column_heat_content_horizon14_count': zero,
            'segment_rollout_warm_column_heat_content_horizon30_count': zero,
            'segment_rollout_warm_column_heat_content_horizon60_count': zero,
        }
    last_idx = max(active_targets)
    prediction, start_mask = _target_tensor_and_mask(lake, lookup_split, start)
    freezing_storage = torch.zeros_like(prediction)
    profile_losses = []
    profile_horizon_weights = []
    profile_target_gaps = []
    residual_smooth_losses = []
    daily_tendency_losses = []
    residual_regularization_losses = []
    physical_scale_regularization_losses = []
    physical_scale_smoothness_losses = []
    kd_prior_regularization_losses = []
    adaptive_parameter_regularization_losses = []
    previous_residual = None
    previous_scales = None
    heat_content_losses = []
    heat_content_weighted_losses = []
    heat_content_effective_weights = []
    warm_column_losses = [[]]
    warm_column_weighted_losses = [[]]
    warm_column_factors = [[]]
    warm_column_errors_c = [[]]
    warm_column_gaps = [[]]
    lst_losses = []
    lst_weights = []
    lst_supervision_count = 0
    start_profile = prediction
    for day_idx in range(start_idx, last_idx):
        previous = prediction
        step_input = torch.clamp(
            prediction + _state_noise_like(prediction, lake['depths'], state_noise_weight),
            0.0,
            40.0,
        )
        next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
        prediction, freezing_storage, diagnostics = model.step(
            step_input,
            lake['forcing_rows'][day_idx],
            lake['static_features'],
            next_forcing_row=next_row,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            return_diagnostics=True,
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        daily_tendency_losses.append(
            _daily_tendency_loss(previous, prediction, lake['depths'], lake['max_depth'])
        )
        residual_regularization_losses.append(_residual_regularization_loss(diagnostics))
        physical_scale_regularization_losses.append(_physical_scale_regularization_loss(diagnostics))
        physical_scale_smoothness_losses.append(_physical_scale_smoothness_loss(diagnostics, previous_scales))
        kd_prior_regularization_losses.append(_kd_prior_regularization_loss(diagnostics))
        adaptive_parameter_regularization_losses.append(_adaptive_parameter_regularization_loss(diagnostics))
        previous_scales = _current_physical_scales(diagnostics)
        residual_smooth_losses.append(_residual_profile_smoothness_loss(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if next_row is not None and float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            if torch.any(lst_mask):
                lst_losses.append(lst_loss_vec[lst_mask].mean())
                lst_weights.append(lst_weight[lst_mask].mean())
                lst_supervision_count += int(lst_mask.sum().detach().cpu().item())
        prediction_idx = day_idx + 1
        if prediction_idx in active_targets:
            target_date = active_targets[prediction_idx]
            target, target_mask = _target_tensor_and_mask(lake, lookup_split, target_date)
            target_gap_days = int(prediction_idx - start_idx)
            horizon_weight = torch.as_tensor(
                min(1.0 + float(target_gap_days) / 30.0, 3.0),
                device=device,
                dtype=prediction.dtype,
            )
            profile_losses.append(horizon_weight * _masked_huber_profile_loss(
                prediction,
                target,
                mask=target_mask,
                delta=profile_huber_delta,
            ))
            profile_horizon_weights.append(horizon_weight.detach())
            profile_target_gaps.append(torch.as_tensor(float(target_gap_days), device=device, dtype=prediction.dtype))
            if float(heat_content_transition_weight) > 0.0:
                _append_weighted_heat_content_transition_loss(
                    heat_content_losses,
                    heat_content_weighted_losses,
                    heat_content_effective_weights,
                    base_weight=heat_content_transition_weight,
                    target_date=target_date,
                    lake=lake,
                    start_profile=start_profile,
                    end_prediction=prediction,
                    end_target=target,
                    start_mask=start_mask,
                    end_mask=target_mask,
                    delta_seconds=float(prediction_idx - start_idx) * 86400.0,
                    season_factors=heat_content_transition_season_factors,
                    use_depth_factor=heat_content_transition_depth_factor,
                    effective_max=heat_content_transition_effective_max,
                    min_full_column_coverage=heat_content_full_column_min_coverage,
                )
            _append_warm_column_heat_content_loss_batch(
                warm_column_losses,
                warm_column_weighted_losses,
                warm_column_factors,
                warm_column_errors_c,
                warm_column_gaps,
                sample_indices=[0],
                target_gap_days=[target_gap_days],
                horizon_weight=horizon_weight.reshape(1),
                lake=lake,
                end_prediction=prediction,
                end_target=target,
                end_mask=target_mask,
                weight=warm_season_column_heat_content_weight,
                min_gap_days=warm_season_column_heat_content_min_gap_days,
                profile_huber_delta=profile_huber_delta,
                min_full_column_coverage=heat_content_full_column_min_coverage,
            )
            if float(teacher_forcing_probability) > 0.0:
                if torch.rand((), device=device).item() < float(teacher_forcing_probability):
                    prediction = target
                    freezing_storage = torch.zeros_like(prediction)
    if not profile_losses:
        zero = torch.tensor(0.0, device=device)
        return zero, 0, {
            'segment_rollout_loss': zero,
            'segment_rollout_profile_loss': zero,
            'segment_rollout_horizon_weight_mean': zero,
            'segment_rollout_max_target_gap_days': zero,
            'segment_rollout_lst_loss': zero,
            'segment_rollout_lst_supervision_count': zero,
            'segment_rollout_lst_weight_mean': zero,
            'segment_rollout_lst_surface_weight': zero,
            'segment_rollout_residual_smooth_loss': zero,
            'segment_rollout_daily_tendency_loss': zero,
            'segment_rollout_residual_regularization_loss': zero,
            'segment_rollout_physical_scale_regularization_loss': zero,
            'segment_rollout_physical_scale_smoothness_loss': zero,
            'segment_rollout_kd_prior_regularization_loss': zero,
            'segment_rollout_kd_prior_regularization_weighted_loss': zero,
            'segment_rollout_adaptive_parameter_regularization_loss': zero,
            'segment_rollout_heat_content_transition_loss': zero,
            'segment_rollout_heat_content_transition_weighted_loss': zero,
            'segment_rollout_heat_content_transition_effective_weight_mean': zero,
            'segment_rollout_heat_content_transition_effective_weight_min': zero,
            'segment_rollout_heat_content_transition_effective_weight_max': zero,
            'segment_rollout_warm_column_heat_content_loss': zero,
            'segment_rollout_warm_column_heat_content_weighted_loss': zero,
            'segment_rollout_warm_column_heat_content_supervision_count': zero,
            'segment_rollout_warm_column_heat_content_warm_factor_mean': zero,
            'segment_rollout_warm_column_heat_content_error_c_mean': zero,
            'segment_rollout_warm_column_heat_content_horizon14_count': zero,
            'segment_rollout_warm_column_heat_content_horizon30_count': zero,
            'segment_rollout_warm_column_heat_content_horizon60_count': zero,
        }
    profile_loss = torch.stack(profile_losses).mean()
    horizon_weight_mean = torch.stack(profile_horizon_weights).mean() if profile_horizon_weights else torch.tensor(0.0, device=device)
    max_target_gap_days = torch.stack(profile_target_gaps).max() if profile_target_gaps else torch.tensor(0.0, device=device)
    residual_smooth_loss = (
        torch.stack(residual_smooth_losses).mean()
        if residual_smooth_losses else torch.tensor(0.0, device=device)
    )
    daily_tendency_loss = (
        torch.stack(daily_tendency_losses).mean()
        if daily_tendency_losses else torch.tensor(0.0, device=device)
    )
    residual_regularization_loss = (
        torch.stack(residual_regularization_losses).mean()
        if residual_regularization_losses else torch.tensor(0.0, device=device)
    )
    physical_scale_regularization_loss = (
        torch.stack(physical_scale_regularization_losses).mean()
        if physical_scale_regularization_losses else torch.tensor(0.0, device=device)
    )
    physical_scale_smoothness_loss = (
        torch.stack(physical_scale_smoothness_losses).mean()
        if physical_scale_smoothness_losses else torch.tensor(0.0, device=device)
    )
    kd_prior_regularization_loss = (
        torch.stack(kd_prior_regularization_losses).mean()
        if kd_prior_regularization_losses else torch.tensor(0.0, device=device)
    )
    kd_prior_regularization_weighted_loss = (
        float(kd_prior_regularization_weight) * kd_prior_regularization_loss
    )
    adaptive_parameter_regularization_loss = (
        torch.stack(adaptive_parameter_regularization_losses).mean()
        if adaptive_parameter_regularization_losses else torch.tensor(0.0, device=device)
    )
    lst_loss = torch.stack(lst_losses).mean() if lst_losses else torch.tensor(0.0, device=device)
    lst_weight_mean = torch.stack(lst_weights).mean() if lst_weights else torch.tensor(0.0, device=device)
    lst_supervision_count_tensor = torch.as_tensor(
        float(lst_supervision_count),
        device=device,
        dtype=prediction.dtype,
    )
    (
        _,
        heat_content_transition_weighted_loss,
        heat_content_details,
    ) = _heat_content_transition_loss_details(
        heat_content_losses,
        heat_content_weighted_losses,
        heat_content_effective_weights,
        device=device,
        prefix='segment_rollout_',
    )
    (
        _,
        warm_column_weighted_loss,
        warm_column_details,
    ) = _warm_column_heat_content_loss_details(
        warm_column_losses[0],
        warm_column_weighted_losses[0],
        warm_column_factors[0],
        warm_column_errors_c[0],
        warm_column_gaps[0],
        device=device,
        prefix='segment_rollout_',
    )
    total = (
        profile_loss
        + float(residual_time_smooth_weight) * residual_smooth_loss
        + float(daily_tendency_weight) * daily_tendency_loss
        + float(residual_regularization_weight) * residual_regularization_loss
        + float(physical_scale_regularization_weight) * physical_scale_regularization_loss
        + float(physical_scale_smoothness_weight) * physical_scale_smoothness_loss
        + kd_prior_regularization_weighted_loss
        + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_loss
        + heat_content_transition_weighted_loss
        + warm_column_weighted_loss
        + float(segment_rollout_lst_surface_weight) * lst_loss
    )
    return total, len(profile_losses), {
        'segment_rollout_loss': total.detach(),
        'segment_rollout_profile_loss': profile_loss.detach(),
        'segment_rollout_horizon_weight_mean': horizon_weight_mean.detach(),
        'segment_rollout_max_target_gap_days': max_target_gap_days.detach(),
        'segment_rollout_lst_loss': lst_loss.detach(),
        'segment_rollout_lst_supervision_count': lst_supervision_count_tensor.detach(),
        'segment_rollout_lst_weight_mean': lst_weight_mean.detach(),
        'segment_rollout_lst_surface_weight': torch.as_tensor(
            float(segment_rollout_lst_surface_weight),
            device=device,
            dtype=prediction.dtype,
        ),
        'segment_rollout_residual_smooth_loss': residual_smooth_loss.detach(),
        'segment_rollout_daily_tendency_loss': daily_tendency_loss.detach(),
        'segment_rollout_residual_regularization_loss': residual_regularization_loss.detach(),
        'segment_rollout_physical_scale_regularization_loss': physical_scale_regularization_loss.detach(),
        'segment_rollout_physical_scale_smoothness_loss': physical_scale_smoothness_loss.detach(),
        'segment_rollout_kd_prior_regularization_loss': kd_prior_regularization_loss.detach(),
        'segment_rollout_kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_loss.detach(),
        'segment_rollout_adaptive_parameter_regularization_loss': adaptive_parameter_regularization_loss.detach(),
        **heat_content_details,
        **warm_column_details,
    }


def _zero_segment_rollout_detail(device):
    zero = torch.tensor(0.0, device=device)
    return {
        'segment_rollout_loss': zero,
        'segment_rollout_profile_loss': zero,
        'segment_rollout_horizon_weight_mean': zero,
        'segment_rollout_max_target_gap_days': zero,
        'segment_rollout_lst_loss': zero,
        'segment_rollout_lst_supervision_count': zero,
        'segment_rollout_lst_weight_mean': zero,
        'segment_rollout_lst_surface_weight': zero,
        'segment_rollout_residual_smooth_loss': zero,
        'segment_rollout_daily_tendency_loss': zero,
        'segment_rollout_residual_regularization_loss': zero,
        'segment_rollout_physical_scale_regularization_loss': zero,
        'segment_rollout_physical_scale_smoothness_loss': zero,
        'segment_rollout_kd_prior_regularization_loss': zero,
        'segment_rollout_kd_prior_regularization_weighted_loss': zero,
        'segment_rollout_adaptive_parameter_regularization_loss': zero,
        'segment_rollout_heat_content_transition_loss': zero,
        'segment_rollout_heat_content_transition_weighted_loss': zero,
        'segment_rollout_heat_content_transition_effective_weight_mean': zero,
        'segment_rollout_heat_content_transition_effective_weight_min': zero,
        'segment_rollout_heat_content_transition_effective_weight_max': zero,
        'segment_rollout_warm_column_heat_content_loss': zero,
        'segment_rollout_warm_column_heat_content_weighted_loss': zero,
        'segment_rollout_warm_column_heat_content_supervision_count': zero,
        'segment_rollout_warm_column_heat_content_warm_factor_mean': zero,
        'segment_rollout_warm_column_heat_content_error_c_mean': zero,
        'segment_rollout_warm_column_heat_content_horizon14_count': zero,
        'segment_rollout_warm_column_heat_content_horizon30_count': zero,
        'segment_rollout_warm_column_heat_content_horizon60_count': zero,
    }


def _segment_rollout_sequence_loss_batch_chunk(
    model,
    lake,
    sequences,
    *,
    active_max_days,
    profile_huber_delta=2.0,
    task_mode='analysis',
    teacher_forcing_probability=0.0,
    state_noise_weight=0.0,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    residual_time_smooth_weight=0.01,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_season_factors=None,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    segment_rollout_lst_surface_weight=0.01,
    warm_season_column_heat_content_weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    warm_season_column_heat_content_min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    if not sequences:
        return []
    device = lake['depths'].device
    active_targets = []
    last_gaps = []
    for start, start_idx, targets in sequences:
        target_map = {
            int(target_idx): target
            for target, target_idx in targets
            if 1 <= int(target_idx - start_idx) <= int(active_max_days)
        }
        active_targets.append(target_map)
        last_gaps.append(max(target_map) - int(start_idx) if target_map else 0)
    if not any(last_gaps):
        return [(torch.tensor(0.0, device=device), 0, _zero_segment_rollout_detail(device)) for _ in sequences]
    if len(set(last_gaps)) != 1:
        raise ValueError('batched segment chunk must contain one active rollout length.')

    starts = [sequence[0] for sequence in sequences]
    start_indices = [int(sequence[1]) for sequence in sequences]
    batch_size = len(sequences)
    prediction, start_mask = _target_tensor_and_mask_batch(lake, lookup_split, starts)
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction

    profile_losses = [[] for _ in range(batch_size)]
    profile_horizon_weights = [[] for _ in range(batch_size)]
    profile_target_gaps = [[] for _ in range(batch_size)]
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    warm_column_losses = [[] for _ in range(batch_size)]
    warm_column_weighted_losses = [[] for _ in range(batch_size)]
    warm_column_factors = [[] for _ in range(batch_size)]
    warm_column_errors_c = [[] for _ in range(batch_size)]
    warm_column_gaps = [[] for _ in range(batch_size)]
    lst_losses = [[] for _ in range(batch_size)]
    lst_weights = [[] for _ in range(batch_size)]
    lst_supervision_counts = [0 for _ in range(batch_size)]
    residual_smooth_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_residual = None
    previous_scales = None
    last_gap = last_gaps[0]

    for offset in range(last_gap):
        previous = prediction
        step_input = torch.clamp(
            prediction + _state_noise_like(prediction, lake['depths'], state_noise_weight),
            0.0,
            40.0,
        )
        day_indices = [start_idx + offset for start_idx in start_indices]
        next_indices = [day_idx + 1 for day_idx in day_indices]
        current_row = _forcing_row_batch(lake, day_indices)
        next_row = _forcing_row_batch(lake, next_indices)
        prediction, freezing_storage, diagnostics = model.step(
            step_input,
            current_row,
            lake['static_features'],
            next_forcing_row=next_row,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            return_diagnostics=True,
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        daily_tendency_vectors.append(
            _daily_tendency_loss_per_sample(previous, prediction, lake['depths'], lake['max_depth'])
        )
        residual_regularization_vectors.append(_residual_regularization_loss_per_sample(diagnostics))
        physical_scale_regularization_vectors.append(_physical_scale_regularization_loss_per_sample(diagnostics))
        physical_scale_smoothness_vectors.append(
            _physical_scale_smoothness_loss_per_sample(diagnostics, previous_scales)
        )
        kd_prior_regularization_vectors.append(_kd_prior_regularization_loss_per_sample(diagnostics))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)
        residual_smooth_vectors.append(_residual_profile_smoothness_loss_per_sample(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            for sample_idx in range(batch_size):
                if bool(lst_mask[sample_idx].detach().cpu().item()):
                    lst_losses[sample_idx].append(lst_loss_vec[sample_idx])
                    lst_weights[sample_idx].append(lst_weight[sample_idx])
                    lst_supervision_counts[sample_idx] += 1

        active_indices = []
        active_dates = []
        active_prediction_indices = []
        for sample_idx, start_idx in enumerate(start_indices):
            prediction_idx = int(start_idx + offset + 1)
            target_date = active_targets[sample_idx].get(prediction_idx)
            if target_date is not None:
                active_indices.append(sample_idx)
                active_dates.append(target_date)
                active_prediction_indices.append(prediction_idx)
        if active_indices:
            active_index_tensor = torch.as_tensor(active_indices, dtype=torch.long, device=device)
            target, target_mask = _target_tensor_and_mask_batch(lake, lookup_split, active_dates)
            active_prediction = prediction.index_select(0, active_index_tensor)
            profile_loss_vec = _masked_huber_profile_loss_per_sample(
                active_prediction,
                target,
                mask=target_mask,
                delta=profile_huber_delta,
            )
            for pos, sample_idx in enumerate(active_indices):
                target_gap_days = int(active_prediction_indices[pos] - start_indices[sample_idx])
                horizon_weight = torch.as_tensor(
                    min(1.0 + float(target_gap_days) / 30.0, 3.0),
                    device=device,
                    dtype=profile_loss_vec.dtype,
                )
                profile_losses[sample_idx].append(horizon_weight * profile_loss_vec[pos])
                profile_horizon_weights[sample_idx].append(horizon_weight.detach())
                profile_target_gaps[sample_idx].append(
                    torch.as_tensor(float(target_gap_days), device=device, dtype=profile_loss_vec.dtype)
                )
            target_gap_days_vec = [
                int(active_prediction_indices[pos] - start_indices[sample_idx])
                for pos, sample_idx in enumerate(active_indices)
            ]
            horizon_weight_vec = torch.as_tensor(
                [min(1.0 + float(gap) / 30.0, 3.0) for gap in target_gap_days_vec],
                dtype=profile_loss_vec.dtype,
                device=device,
            )
            if float(heat_content_transition_weight) > 0.0:
                delta_seconds = [
                    float(active_prediction_indices[pos] - start_indices[sample_idx]) * 86400.0
                    for pos, sample_idx in enumerate(active_indices)
                ]
                _append_weighted_heat_content_transition_loss_batch(
                    heat_content_losses,
                    heat_content_weighted_losses,
                    heat_content_effective_weights,
                    sample_indices=active_indices,
                    base_weight=heat_content_transition_weight,
                    target_dates=active_dates,
                    lake=lake,
                    start_profile=start_profile.index_select(0, active_index_tensor),
                    end_prediction=active_prediction,
                    end_target=target,
                    start_mask=start_mask.index_select(0, active_index_tensor),
                    end_mask=target_mask,
                    delta_seconds=delta_seconds,
                    season_factors=heat_content_transition_season_factors,
                    use_depth_factor=heat_content_transition_depth_factor,
                    effective_max=heat_content_transition_effective_max,
                    min_full_column_coverage=heat_content_full_column_min_coverage,
                )
            _append_warm_column_heat_content_loss_batch(
                warm_column_losses,
                warm_column_weighted_losses,
                warm_column_factors,
                warm_column_errors_c,
                warm_column_gaps,
                sample_indices=active_indices,
                target_gap_days=target_gap_days_vec,
                horizon_weight=horizon_weight_vec,
                lake=lake,
                end_prediction=active_prediction,
                end_target=target,
                end_mask=target_mask,
                weight=warm_season_column_heat_content_weight,
                min_gap_days=warm_season_column_heat_content_min_gap_days,
                profile_huber_delta=profile_huber_delta,
                min_full_column_coverage=heat_content_full_column_min_coverage,
            )
            if float(teacher_forcing_probability) > 0.0:
                force_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                replacement = prediction.detach().clone()
                random_mask = (
                    torch.rand(len(active_indices), device=device)
                    < float(teacher_forcing_probability)
                )
                if torch.any(random_mask):
                    forced_samples = active_index_tensor.index_select(
                        0,
                        torch.nonzero(random_mask, as_tuple=False).reshape(-1),
                    )
                    forced_targets = target.index_select(
                        0,
                        torch.nonzero(random_mask, as_tuple=False).reshape(-1),
                    )
                    force_mask[forced_samples] = True
                    replacement[forced_samples] = forced_targets
                prediction = torch.where(force_mask.reshape(-1, 1), replacement, prediction)
                freezing_storage = torch.where(
                    force_mask.reshape(-1, 1),
                    torch.zeros_like(freezing_storage),
                    freezing_storage,
                )

    residual_smooth_vec = torch.stack(residual_smooth_vectors, dim=0).mean(dim=0)
    daily_tendency_vec = torch.stack(daily_tendency_vectors, dim=0).mean(dim=0)
    residual_regularization_vec = torch.stack(residual_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_regularization_vec = torch.stack(physical_scale_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_smoothness_vec = torch.stack(physical_scale_smoothness_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_vec = torch.stack(kd_prior_regularization_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_weighted_vec = float(kd_prior_regularization_weight) * kd_prior_regularization_vec
    adaptive_parameter_regularization_vec = torch.stack(adaptive_parameter_regularization_vectors, dim=0).mean(dim=0)

    results = []
    for sample_idx in range(batch_size):
        if not profile_losses[sample_idx]:
            results.append((torch.tensor(0.0, device=device), 0, _zero_segment_rollout_detail(device)))
            continue
        profile_loss = torch.stack(profile_losses[sample_idx]).mean()
        horizon_weight_mean = (
            torch.stack(profile_horizon_weights[sample_idx]).mean()
            if profile_horizon_weights[sample_idx] else torch.tensor(0.0, device=device)
        )
        max_target_gap_days = (
            torch.stack(profile_target_gaps[sample_idx]).max()
            if profile_target_gaps[sample_idx] else torch.tensor(0.0, device=device)
        )
        _, heat_content_weighted_loss, heat_content_details = _heat_content_transition_loss_details(
            heat_content_losses[sample_idx],
            heat_content_weighted_losses[sample_idx],
            heat_content_effective_weights[sample_idx],
            device=device,
            prefix='segment_rollout_',
        )
        _, warm_column_weighted_loss, warm_column_details = _warm_column_heat_content_loss_details(
            warm_column_losses[sample_idx],
            warm_column_weighted_losses[sample_idx],
            warm_column_factors[sample_idx],
            warm_column_errors_c[sample_idx],
            warm_column_gaps[sample_idx],
            device=device,
            prefix='segment_rollout_',
        )
        lst_loss = _mean_or_zero(lst_losses[sample_idx], device=device)
        lst_weight_mean = _mean_or_zero(lst_weights[sample_idx], device=device)
        lst_supervision_count = torch.as_tensor(
            float(lst_supervision_counts[sample_idx]),
            device=device,
            dtype=profile_loss.dtype,
        )
        total = (
            profile_loss
            + float(residual_time_smooth_weight) * residual_smooth_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + warm_column_weighted_loss
            + float(segment_rollout_lst_surface_weight) * lst_loss
        )
        results.append((
            total,
            len(profile_losses[sample_idx]),
            {
                'segment_rollout_loss': total.detach(),
                'segment_rollout_profile_loss': profile_loss.detach(),
                'segment_rollout_horizon_weight_mean': horizon_weight_mean.detach(),
                'segment_rollout_max_target_gap_days': max_target_gap_days.detach(),
                'segment_rollout_lst_loss': lst_loss.detach(),
                'segment_rollout_lst_supervision_count': lst_supervision_count.detach(),
                'segment_rollout_lst_weight_mean': lst_weight_mean.detach(),
                'segment_rollout_lst_surface_weight': torch.as_tensor(
                    float(segment_rollout_lst_surface_weight),
                    device=device,
                    dtype=profile_loss.dtype,
                ),
                'segment_rollout_residual_smooth_loss': residual_smooth_vec[sample_idx].detach(),
                'segment_rollout_daily_tendency_loss': daily_tendency_vec[sample_idx].detach(),
                'segment_rollout_residual_regularization_loss': residual_regularization_vec[sample_idx].detach(),
                'segment_rollout_physical_scale_regularization_loss': physical_scale_regularization_vec[sample_idx].detach(),
                'segment_rollout_physical_scale_smoothness_loss': physical_scale_smoothness_vec[sample_idx].detach(),
                'segment_rollout_kd_prior_regularization_loss': kd_prior_regularization_vec[sample_idx].detach(),
                'segment_rollout_kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_vec[sample_idx].detach(),
                'segment_rollout_adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
                **heat_content_details,
                **warm_column_details,
            },
        ))
    return results


def _segment_rollout_sequence_losses_for_lake(
    model,
    lake,
    sequences,
    *,
    segment_rollout_batch_mode='off',
    segment_rollout_batch_size=0,
    active_max_days,
    cached_batches=None,
    **kwargs,
):
    if segment_rollout_batch_mode != 'on':
        return [
            _segment_rollout_sequence_loss(
                model,
                lake,
                sequence,
                active_max_days=active_max_days,
                **kwargs,
            )
            for sequence in sequences
        ]

    if cached_batches is not None:
        batches = cached_batches
    else:
        grouped = {}
        for sequence in sequences:
            start, start_idx, targets = sequence
            usable_targets = [
                (target, target_idx)
                for target, target_idx in targets
                if 1 <= int(target_idx - start_idx) <= int(active_max_days)
            ]
            if not usable_targets:
                grouped.setdefault(0, []).append((start, start_idx, usable_targets))
                continue
            last_gap = max(int(target_idx - start_idx) for _, target_idx in usable_targets)
            grouped.setdefault(last_gap, []).append((start, start_idx, usable_targets))
        batches = [
            tuple(chunk)
            for last_gap in sorted(grouped)
            for chunk in _batch_chunks(grouped[last_gap], segment_rollout_batch_size)
        ]
    results = []
    for chunk in batches:
        results.extend(_segment_rollout_sequence_loss_batch_chunk(
            model,
            lake,
            chunk,
            active_max_days=active_max_days,
            **kwargs,
        ))
    return results


def _segment_rollout_sequence_loss_cross_lake_batch_chunk(
    model,
    items,
    *,
    active_max_days,
    profile_huber_delta=2.0,
    task_mode='analysis',
    teacher_forcing_probability=0.0,
    state_noise_weight=0.0,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    residual_time_smooth_weight=0.01,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    segment_rollout_lst_surface_weight=0.01,
    warm_season_column_heat_content_weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    warm_season_column_heat_content_min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    if not items:
        return []
    ref_lake = items[0][1]
    device = ref_lake['depths'].device
    active_targets = []
    last_gaps = []
    for _, _, sequence in items:
        start, start_idx, targets = sequence
        target_map = {
            int(target_idx): target
            for target, target_idx in targets
            if 1 <= int(target_idx - start_idx) <= int(active_max_days)
        }
        active_targets.append(target_map)
        last_gaps.append(max(target_map) - int(start_idx) if target_map else 0)
    if not any(last_gaps):
        return [
            (item[0], torch.tensor(0.0, device=device), 0, _zero_segment_rollout_detail(device))
            for item in items
        ]
    if len(set(last_gaps)) != 1:
        raise ValueError('cross-lake segment chunk must contain one active rollout length.')

    start_indices = [int(item[2][1]) for item in items]
    batch_size = len(items)
    prediction, start_mask = _stack_target_batch_for_items(items, lookup_split, lambda item: item[2][0])
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    static_features = _stack_static_features_for_items(items)

    profile_losses = [[] for _ in range(batch_size)]
    profile_horizon_weights = [[] for _ in range(batch_size)]
    profile_target_gaps = [[] for _ in range(batch_size)]
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    warm_column_losses = [[] for _ in range(batch_size)]
    warm_column_weighted_losses = [[] for _ in range(batch_size)]
    warm_column_factors = [[] for _ in range(batch_size)]
    warm_column_errors_c = [[] for _ in range(batch_size)]
    warm_column_gaps = [[] for _ in range(batch_size)]
    lst_losses = [[] for _ in range(batch_size)]
    lst_weights = [[] for _ in range(batch_size)]
    lst_supervision_counts = [0 for _ in range(batch_size)]
    residual_smooth_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_residual = None
    previous_scales = None
    last_gap = last_gaps[0]

    for offset in range(last_gap):
        previous = prediction
        step_input = torch.clamp(
            prediction + _state_noise_like(prediction, ref_lake['depths'], state_noise_weight),
            0.0,
            40.0,
        )
        day_indices = [start_idx + offset for start_idx in start_indices]
        next_indices = [day_idx + 1 for day_idx in day_indices]
        current_row = _stack_forcing_batch_for_items(items, day_indices)
        next_row = _stack_forcing_batch_for_items(items, next_indices)
        prediction, freezing_storage, diagnostics = model.step(
            step_input,
            current_row,
            static_features,
            next_forcing_row=next_row,
            task_mode=task_mode,
            depths=ref_lake['depths'],
            area_profile=ref_lake['area'],
            return_diagnostics=True,
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        daily_tendency_vectors.append(
            _daily_tendency_loss_per_sample(previous, prediction, ref_lake['depths'], ref_lake['max_depth'])
        )
        residual_regularization_vectors.append(_residual_regularization_loss_per_sample(diagnostics))
        physical_scale_regularization_vectors.append(_physical_scale_regularization_loss_per_sample(diagnostics))
        physical_scale_smoothness_vectors.append(
            _physical_scale_smoothness_loss_per_sample(diagnostics, previous_scales)
        )
        kd_prior_regularization_vectors.append(_kd_prior_regularization_loss_per_sample(diagnostics))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)
        residual_smooth_vectors.append(_residual_profile_smoothness_loss_per_sample(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            for sample_idx in range(batch_size):
                if bool(lst_mask[sample_idx].detach().cpu().item()):
                    lst_losses[sample_idx].append(lst_loss_vec[sample_idx])
                    lst_weights[sample_idx].append(lst_weight[sample_idx])
                    lst_supervision_counts[sample_idx] += 1

        active_positions = []
        active_dates = []
        active_prediction_indices = []
        for sample_idx, start_idx in enumerate(start_indices):
            prediction_idx = int(start_idx + offset + 1)
            target_date = active_targets[sample_idx].get(prediction_idx)
            if target_date is not None:
                active_positions.append(sample_idx)
                active_dates.append(target_date)
                active_prediction_indices.append(prediction_idx)
        if active_positions:
            active_index_tensor = torch.as_tensor(active_positions, dtype=torch.long, device=device)
            active_items = [
                (items[sample_idx][0], items[sample_idx][1], active_dates[pos])
                for pos, sample_idx in enumerate(active_positions)
            ]
            target, target_mask = _stack_target_batch_for_items(active_items, lookup_split, lambda item: item[2])
            active_prediction = prediction.index_select(0, active_index_tensor)
            profile_loss_vec = _masked_huber_profile_loss_per_sample(
                active_prediction,
                target,
                mask=target_mask,
                delta=profile_huber_delta,
            )
            for pos, sample_idx in enumerate(active_positions):
                target_gap_days = int(active_prediction_indices[pos] - start_indices[sample_idx])
                horizon_weight = torch.as_tensor(
                    min(1.0 + float(target_gap_days) / 30.0, 3.0),
                    device=device,
                    dtype=profile_loss_vec.dtype,
                )
                profile_losses[sample_idx].append(horizon_weight * profile_loss_vec[pos])
                profile_horizon_weights[sample_idx].append(horizon_weight.detach())
                profile_target_gaps[sample_idx].append(
                    torch.as_tensor(float(target_gap_days), device=device, dtype=profile_loss_vec.dtype)
                )
                sample_lake = items[sample_idx][1]
                _append_warm_column_heat_content_loss_batch(
                    warm_column_losses,
                    warm_column_weighted_losses,
                    warm_column_factors,
                    warm_column_errors_c,
                    warm_column_gaps,
                    sample_indices=[sample_idx],
                    target_gap_days=[target_gap_days],
                    horizon_weight=horizon_weight.reshape(1),
                    lake=sample_lake,
                    end_prediction=active_prediction[pos:pos + 1],
                    end_target=target[pos:pos + 1],
                    end_mask=target_mask[pos:pos + 1],
                    weight=warm_season_column_heat_content_weight,
                    min_gap_days=warm_season_column_heat_content_min_gap_days,
                    profile_huber_delta=profile_huber_delta,
                    min_full_column_coverage=heat_content_full_column_min_coverage,
                )
                if float(heat_content_transition_weight) > 0.0:
                    _append_weighted_heat_content_transition_loss(
                        heat_content_losses[sample_idx],
                        heat_content_weighted_losses[sample_idx],
                        heat_content_effective_weights[sample_idx],
                        base_weight=heat_content_transition_weight,
                        target_date=active_dates[pos],
                        lake=sample_lake,
                        start_profile=start_profile[sample_idx:sample_idx + 1],
                        end_prediction=active_prediction[pos:pos + 1],
                        end_target=target[pos:pos + 1],
                        start_mask=start_mask[sample_idx:sample_idx + 1],
                        end_mask=target_mask[pos:pos + 1],
                        delta_seconds=float(active_prediction_indices[pos] - start_indices[sample_idx]) * 86400.0,
                        season_factors=sample_lake['heat_content_transition_season_factors'],
                        use_depth_factor=heat_content_transition_depth_factor,
                        effective_max=heat_content_transition_effective_max,
                        min_full_column_coverage=heat_content_full_column_min_coverage,
                    )
            if float(teacher_forcing_probability) > 0.0:
                force_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                replacement = prediction.detach().clone()
                random_mask = torch.rand(len(active_positions), device=device) < float(teacher_forcing_probability)
                if torch.any(random_mask):
                    forced_local = torch.nonzero(random_mask, as_tuple=False).reshape(-1)
                    forced_samples = active_index_tensor.index_select(0, forced_local)
                    forced_targets = target.index_select(0, forced_local)
                    force_mask[forced_samples] = True
                    replacement[forced_samples] = forced_targets
                prediction = torch.where(force_mask.reshape(-1, 1), replacement, prediction)
                freezing_storage = torch.where(
                    force_mask.reshape(-1, 1),
                    torch.zeros_like(freezing_storage),
                    freezing_storage,
                )

    residual_smooth_vec = torch.stack(residual_smooth_vectors, dim=0).mean(dim=0)
    daily_tendency_vec = torch.stack(daily_tendency_vectors, dim=0).mean(dim=0)
    residual_regularization_vec = torch.stack(residual_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_regularization_vec = torch.stack(physical_scale_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_smoothness_vec = torch.stack(physical_scale_smoothness_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_vec = torch.stack(kd_prior_regularization_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_weighted_vec = float(kd_prior_regularization_weight) * kd_prior_regularization_vec
    adaptive_parameter_regularization_vec = torch.stack(adaptive_parameter_regularization_vectors, dim=0).mean(dim=0)

    results = []
    for sample_idx, item in enumerate(items):
        if not profile_losses[sample_idx]:
            results.append((item[0], torch.tensor(0.0, device=device), 0, _zero_segment_rollout_detail(device)))
            continue
        profile_loss = torch.stack(profile_losses[sample_idx]).mean()
        horizon_weight_mean = (
            torch.stack(profile_horizon_weights[sample_idx]).mean()
            if profile_horizon_weights[sample_idx] else torch.tensor(0.0, device=device)
        )
        max_target_gap_days = (
            torch.stack(profile_target_gaps[sample_idx]).max()
            if profile_target_gaps[sample_idx] else torch.tensor(0.0, device=device)
        )
        _, heat_content_weighted_loss, heat_content_details = _heat_content_transition_loss_details(
            heat_content_losses[sample_idx],
            heat_content_weighted_losses[sample_idx],
            heat_content_effective_weights[sample_idx],
            device=device,
            prefix='segment_rollout_',
        )
        _, warm_column_weighted_loss, warm_column_details = _warm_column_heat_content_loss_details(
            warm_column_losses[sample_idx],
            warm_column_weighted_losses[sample_idx],
            warm_column_factors[sample_idx],
            warm_column_errors_c[sample_idx],
            warm_column_gaps[sample_idx],
            device=device,
            prefix='segment_rollout_',
        )
        lst_loss = _mean_or_zero(lst_losses[sample_idx], device=device)
        lst_weight_mean = _mean_or_zero(lst_weights[sample_idx], device=device)
        lst_supervision_count = torch.as_tensor(
            float(lst_supervision_counts[sample_idx]),
            device=device,
            dtype=profile_loss.dtype,
        )
        total = (
            profile_loss
            + float(residual_time_smooth_weight) * residual_smooth_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + warm_column_weighted_loss
            + float(segment_rollout_lst_surface_weight) * lst_loss
        )
        results.append((
            item[0],
            total,
            len(profile_losses[sample_idx]),
            {
                'segment_rollout_loss': total.detach(),
                'segment_rollout_profile_loss': profile_loss.detach(),
                'segment_rollout_horizon_weight_mean': horizon_weight_mean.detach(),
                'segment_rollout_max_target_gap_days': max_target_gap_days.detach(),
                'segment_rollout_lst_loss': lst_loss.detach(),
                'segment_rollout_lst_supervision_count': lst_supervision_count.detach(),
                'segment_rollout_lst_weight_mean': lst_weight_mean.detach(),
                'segment_rollout_lst_surface_weight': torch.as_tensor(
                    float(segment_rollout_lst_surface_weight),
                    device=device,
                    dtype=profile_loss.dtype,
                ),
                'segment_rollout_residual_smooth_loss': residual_smooth_vec[sample_idx].detach(),
                'segment_rollout_daily_tendency_loss': daily_tendency_vec[sample_idx].detach(),
                'segment_rollout_residual_regularization_loss': residual_regularization_vec[sample_idx].detach(),
                'segment_rollout_physical_scale_regularization_loss': physical_scale_regularization_vec[sample_idx].detach(),
                'segment_rollout_physical_scale_smoothness_loss': physical_scale_smoothness_vec[sample_idx].detach(),
                'segment_rollout_kd_prior_regularization_loss': kd_prior_regularization_vec[sample_idx].detach(),
                'segment_rollout_kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_vec[sample_idx].detach(),
                'segment_rollout_adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
                **heat_content_details,
                **warm_column_details,
            },
        ))
    return results


def _segment_rollout_sequence_losses_for_lakes_cross_batch(
    model,
    lakes,
    sequences_by_lake,
    *,
    segment_rollout_batch_mode='off',
    segment_rollout_batch_size=0,
    cross_lake_batch_size=0,
    active_max_days,
    cached_batches=None,
    **kwargs,
):
    if segment_rollout_batch_mode != 'on':
        results = {}
        for lake_idx, lake in enumerate(lakes):
            lake_kwargs = dict(kwargs)
            lake_kwargs.setdefault(
                'heat_content_transition_season_factors',
                lake['heat_content_transition_season_factors'],
            )
            results[lake_idx] = _segment_rollout_sequence_losses_for_lake(
                model,
                lake,
                sequences_by_lake.get(lake_idx, []),
                segment_rollout_batch_mode=segment_rollout_batch_mode,
                segment_rollout_batch_size=segment_rollout_batch_size,
                active_max_days=active_max_days,
                **lake_kwargs,
            )
        return results

    results = {lake_idx: [] for lake_idx in range(len(lakes))}
    if cached_batches is not None:
        batches = cached_batches
    else:
        grouped = {}
        for bucket_entries in _cross_lake_bucket_entries(lakes).values():
            for lake_idx, lake in bucket_entries:
                for sequence in sequences_by_lake.get(lake_idx, []):
                    start, start_idx, targets = sequence
                    usable_targets = [
                        (target, target_idx)
                        for target, target_idx in targets
                        if 1 <= int(target_idx - start_idx) <= int(active_max_days)
                    ]
                    last_gap = max((int(target_idx - start_idx) for _, target_idx in usable_targets), default=0)
                    grouped.setdefault((_cross_lake_batch_key(lake), last_gap), []).append((
                        lake_idx,
                        lake,
                        (start, start_idx, usable_targets),
                    ))

        batch_size = int(cross_lake_batch_size or segment_rollout_batch_size or 0)
        batches = [
            tuple(chunk)
            for key in sorted(grouped, key=lambda value: (value[0][0], value[1]))
            for chunk in _batch_chunks(grouped[key], batch_size)
        ]
    for chunk in batches:
        for lake_idx, loss, count, detail in _segment_rollout_sequence_loss_cross_lake_batch_chunk(
            model,
            chunk,
            active_max_days=active_max_days,
            **kwargs,
        ):
            results[lake_idx].append((loss, count, detail))
    return results


def _transition_loss(
    model,
    lake,
    pair,
    *,
    profile_huber_delta=2.0,
    lst_surface_weight=0.03,
    energy_balance_weight=0.001,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_season_factors=None,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    start, end, start_idx, end_idx = pair
    device = lake['depths'].device
    prediction, start_mask = _target_tensor_and_mask(lake, lookup_split, start)
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    target, target_mask = _target_tensor_and_mask(lake, lookup_split, end)

    lst_losses = []
    energy_losses = []
    daily_tendency_losses = []
    residual_regularization_losses = []
    physical_scale_regularization_losses = []
    physical_scale_smoothness_losses = []
    kd_prior_regularization_losses = []
    adaptive_parameter_regularization_losses = []
    heat_content_losses = []
    heat_content_weighted_losses = []
    heat_content_effective_weights = []
    previous_scales = None
    for day_idx in range(start_idx, end_idx):
        previous = prediction
        next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
        prediction, freezing_storage, diagnostics = model.step(
            prediction,
            lake['forcing_rows'][day_idx],
            lake['static_features'],
            next_forcing_row=next_row,
            return_diagnostics=True,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        daily_tendency_losses.append(
            _daily_tendency_loss(previous, prediction, lake['depths'], lake['max_depth'])
        )
        residual_regularization_losses.append(_residual_regularization_loss(diagnostics))
        physical_scale_regularization_losses.append(_physical_scale_regularization_loss(diagnostics))
        physical_scale_smoothness_losses.append(_physical_scale_smoothness_loss(diagnostics, previous_scales))
        kd_prior_regularization_losses.append(_kd_prior_regularization_loss(diagnostics))
        adaptive_parameter_regularization_losses.append(_adaptive_parameter_regularization_loss(diagnostics))
        previous_scales = _current_physical_scales(diagnostics)
        if next_row is not None:
            lst_loss_vec, _, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            if torch.any(lst_mask):
                lst_losses.append(lst_loss_vec[lst_mask].mean())
        energy_residual = (diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']) / 150.0
        energy_losses.append(torch.mean(energy_residual.pow(2)))
    profile_loss = _masked_huber_profile_loss(prediction, target, mask=target_mask, delta=profile_huber_delta)
    physics_loss = _profile_physics_loss(prediction)
    daily_tendency_loss = (
        torch.stack(daily_tendency_losses).mean()
        if daily_tendency_losses else torch.tensor(0.0, device=device)
    )
    residual_regularization_loss = (
        torch.stack(residual_regularization_losses).mean()
        if residual_regularization_losses else torch.tensor(0.0, device=device)
    )
    physical_scale_regularization_loss = (
        torch.stack(physical_scale_regularization_losses).mean()
        if physical_scale_regularization_losses else torch.tensor(0.0, device=device)
    )
    physical_scale_smoothness_loss = (
        torch.stack(physical_scale_smoothness_losses).mean()
        if physical_scale_smoothness_losses else torch.tensor(0.0, device=device)
    )
    kd_prior_regularization_loss = (
        torch.stack(kd_prior_regularization_losses).mean()
        if kd_prior_regularization_losses else torch.tensor(0.0, device=device)
    )
    kd_prior_regularization_weighted_loss = (
        float(kd_prior_regularization_weight) * kd_prior_regularization_loss
    )
    adaptive_parameter_regularization_loss = (
        torch.stack(adaptive_parameter_regularization_losses).mean()
        if adaptive_parameter_regularization_losses else torch.tensor(0.0, device=device)
    )
    lst_loss = torch.stack(lst_losses).mean() if lst_losses else torch.tensor(0.0, device=device)
    energy_loss = torch.stack(energy_losses).mean() if energy_losses else torch.tensor(0.0, device=device)
    if float(heat_content_transition_weight) > 0.0:
        _append_weighted_heat_content_transition_loss(
            heat_content_losses,
            heat_content_weighted_losses,
            heat_content_effective_weights,
            base_weight=heat_content_transition_weight,
            target_date=end,
            lake=lake,
            start_profile=start_profile,
            end_prediction=prediction,
            end_target=target,
            start_mask=start_mask,
            end_mask=target_mask,
            delta_seconds=float(end_idx - start_idx) * 86400.0,
            season_factors=heat_content_transition_season_factors,
            use_depth_factor=heat_content_transition_depth_factor,
            effective_max=heat_content_transition_effective_max,
            min_full_column_coverage=heat_content_full_column_min_coverage,
        )
    (
        _,
        heat_content_transition_weighted_loss,
        heat_content_details,
    ) = _heat_content_transition_loss_details(
        heat_content_losses,
        heat_content_weighted_losses,
        heat_content_effective_weights,
        device=device,
    )
    total = (
        profile_loss
        + physics_loss
        + float(daily_tendency_weight) * daily_tendency_loss
        + float(residual_regularization_weight) * residual_regularization_loss
        + float(physical_scale_regularization_weight) * physical_scale_regularization_loss
        + float(physical_scale_smoothness_weight) * physical_scale_smoothness_loss
        + kd_prior_regularization_weighted_loss
        + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_loss
        + heat_content_transition_weighted_loss
        + float(lst_surface_weight) * lst_loss
        + float(energy_balance_weight) * energy_loss
    )
    return total, {
        'profile_loss': profile_loss.detach(),
        'lst_loss': lst_loss.detach(),
        'energy_loss': energy_loss.detach(),
        'daily_tendency_loss': daily_tendency_loss.detach(),
        'residual_regularization_loss': residual_regularization_loss.detach(),
        'physical_scale_reg_loss': physical_scale_regularization_loss.detach(),
        'physical_scale_smooth_loss': physical_scale_smoothness_loss.detach(),
        'kd_prior_regularization_loss': kd_prior_regularization_loss.detach(),
        'kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_loss.detach(),
        'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_loss.detach(),
        **heat_content_details,
        **_scale_detail_record(diagnostics),
    }


def _transition_loss_batch_chunk(
    model,
    lake,
    pairs,
    *,
    profile_huber_delta=2.0,
    lst_surface_weight=0.03,
    energy_balance_weight=0.001,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_season_factors=None,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    if not pairs:
        return [], []
    gaps = [int(pair[3] - pair[2]) for pair in pairs]
    if len(set(gaps)) != 1:
        raise ValueError('batched transition chunk must contain one rollout gap.')
    starts = [pair[0] for pair in pairs]
    ends = [pair[1] for pair in pairs]
    start_indices = [int(pair[2]) for pair in pairs]
    end_indices = [int(pair[3]) for pair in pairs]
    batch_size = len(pairs)
    device = lake['depths'].device

    prediction, start_mask = _target_tensor_and_mask_batch(lake, lookup_split, starts)
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    target, target_mask = _target_tensor_and_mask_batch(lake, lookup_split, ends)

    lst_losses = [[] for _ in range(batch_size)]
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    energy_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_scales = None
    final_diagnostics = None

    for offset in range(gaps[0]):
        previous = prediction
        day_indices = [start_idx + offset for start_idx in start_indices]
        next_indices = [day_idx + 1 for day_idx in day_indices]
        current_row = _forcing_row_batch(lake, day_indices)
        next_row = _forcing_row_batch(lake, next_indices)
        prediction, freezing_storage, diagnostics = model.step(
            prediction,
            current_row,
            lake['static_features'],
            next_forcing_row=next_row,
            return_diagnostics=True,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        final_diagnostics = diagnostics
        daily_tendency_vectors.append(
            _daily_tendency_loss_per_sample(previous, prediction, lake['depths'], lake['max_depth'])
        )
        residual_regularization_vectors.append(_residual_regularization_loss_per_sample(diagnostics))
        physical_scale_regularization_vectors.append(_physical_scale_regularization_loss_per_sample(diagnostics))
        physical_scale_smoothness_vectors.append(
            _physical_scale_smoothness_loss_per_sample(diagnostics, previous_scales)
        )
        kd_prior_regularization_vectors.append(_kd_prior_regularization_loss_per_sample(diagnostics))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)

        lst_loss_vec, _, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
        for sample_idx in range(batch_size):
            if bool(lst_mask[sample_idx].detach().cpu().item()):
                lst_losses[sample_idx].append(lst_loss_vec[sample_idx])

        energy_residual = (diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']) / 150.0
        energy_vectors.append(energy_residual.reshape(-1).pow(2))

    profile_loss_vec = _masked_huber_profile_loss_per_sample(
        prediction,
        target,
        mask=target_mask,
        delta=profile_huber_delta,
    )
    physics_loss_vec = _profile_physics_loss_per_sample(prediction)
    daily_tendency_vec = torch.stack(daily_tendency_vectors, dim=0).mean(dim=0)
    residual_regularization_vec = torch.stack(residual_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_regularization_vec = torch.stack(physical_scale_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_smoothness_vec = torch.stack(physical_scale_smoothness_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_vec = torch.stack(kd_prior_regularization_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_weighted_vec = float(kd_prior_regularization_weight) * kd_prior_regularization_vec
    adaptive_parameter_regularization_vec = torch.stack(adaptive_parameter_regularization_vectors, dim=0).mean(dim=0)
    energy_loss_vec = torch.stack(energy_vectors, dim=0).mean(dim=0)
    if float(heat_content_transition_weight) > 0.0:
        _append_weighted_heat_content_transition_loss_batch(
            heat_content_losses,
            heat_content_weighted_losses,
            heat_content_effective_weights,
            sample_indices=list(range(batch_size)),
            base_weight=heat_content_transition_weight,
            target_dates=ends,
            lake=lake,
            start_profile=start_profile,
            end_prediction=prediction,
            end_target=target,
            start_mask=start_mask,
            end_mask=target_mask,
            delta_seconds=[
                float(end_indices[sample_idx] - start_indices[sample_idx]) * 86400.0
                for sample_idx in range(batch_size)
            ],
            season_factors=heat_content_transition_season_factors,
            use_depth_factor=heat_content_transition_depth_factor,
            effective_max=heat_content_transition_effective_max,
            min_full_column_coverage=heat_content_full_column_min_coverage,
        )

    losses = []
    details = []
    for sample_idx in range(batch_size):
        _, heat_content_weighted_loss, heat_content_details = _heat_content_transition_loss_details(
            heat_content_losses[sample_idx],
            heat_content_weighted_losses[sample_idx],
            heat_content_effective_weights[sample_idx],
            device=device,
        )
        lst_loss = _mean_or_zero(lst_losses[sample_idx], device=device)
        total = (
            profile_loss_vec[sample_idx]
            + physics_loss_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + float(lst_surface_weight) * lst_loss
            + float(energy_balance_weight) * energy_loss_vec[sample_idx]
        )
        losses.append(total)
        details.append({
            'profile_loss': profile_loss_vec[sample_idx].detach(),
            'lst_loss': lst_loss.detach(),
            'energy_loss': energy_loss_vec[sample_idx].detach(),
            'daily_tendency_loss': daily_tendency_vec[sample_idx].detach(),
            'residual_regularization_loss': residual_regularization_vec[sample_idx].detach(),
            'physical_scale_reg_loss': physical_scale_regularization_vec[sample_idx].detach(),
            'physical_scale_smooth_loss': physical_scale_smoothness_vec[sample_idx].detach(),
            'kd_prior_regularization_loss': kd_prior_regularization_vec[sample_idx].detach(),
            'kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_vec[sample_idx].detach(),
            'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
            **heat_content_details,
            **_scale_detail_record_for_sample(final_diagnostics, sample_idx),
        })
    return losses, details


def _transition_losses_for_lake(
    model,
    lake,
    pairs,
    *,
    transition_batch_mode='off',
    transition_batch_size=0,
    **kwargs,
):
    if transition_batch_mode != 'on':
        losses = []
        details = []
        for pair in pairs:
            loss, record = _transition_loss(model, lake, pair, **kwargs)
            losses.append(loss)
            details.append(record)
        return losses, details

    grouped = {}
    for pair in pairs:
        gap = int(pair[3] - pair[2])
        grouped.setdefault(gap, []).append(pair)
    losses = []
    details = []
    for gap in sorted(grouped):
        for chunk in _batch_chunks(grouped[gap], transition_batch_size):
            chunk_losses, chunk_details = _transition_loss_batch_chunk(
                model,
                lake,
                chunk,
                **kwargs,
            )
            losses.extend(chunk_losses)
            details.extend(chunk_details)
    return losses, details


def _cross_lake_batch_key(lake):
    """Return a conservative compatibility key for exact same-grid batching."""
    depths = lake['depths'].detach().cpu().reshape(-1)
    area = lake['area'].detach().cpu().reshape(-1)
    forcing_keys = tuple(sorted(lake.get('forcing_tensors', {}).keys()))
    static_dim = int(lake['static_features'].reshape(-1).numel())
    return (
        int(depths.numel()),
        tuple(round(float(value), 6) for value in depths.tolist()),
        tuple(round(float(value), 6) for value in area.tolist()),
        forcing_keys,
        static_dim,
    )


def _cross_lake_bucket_entries(lakes):
    buckets = {}
    for lake_idx, lake in enumerate(lakes):
        buckets.setdefault(_cross_lake_batch_key(lake), []).append((lake_idx, lake))
    return buckets


def _stack_static_features_for_items(items):
    return torch.stack([
        item[1]['static_features'].reshape(-1)
        for item in items
    ], dim=0)


def _stack_target_batch_for_items(items, preferred_split, date_getter):
    profiles = []
    masks = []
    for item in items:
        profile, mask = _target_tensor_and_mask(item[1], preferred_split, date_getter(item))
        profiles.append(profile)
        masks.append(
            torch.ones_like(profile, dtype=torch.bool)
            if mask is None else mask.to(device=profile.device, dtype=torch.bool)
        )
    if not profiles:
        raise ValueError('cross-lake target batch must not be empty.')
    return torch.cat(profiles, dim=0), torch.cat(masks, dim=0)


def _stack_forcing_batch_for_items(items, day_indices):
    if len(items) != len(day_indices):
        raise ValueError('forcing item count must match day index count.')
    return _stack_forcing_rows([
        item[1]['forcing_rows'][int(day_idx)]
        for item, day_idx in zip(items, day_indices)
    ])


def _transition_loss_cross_lake_batch_chunk(
    model,
    items,
    *,
    profile_huber_delta=2.0,
    lst_surface_weight=0.03,
    energy_balance_weight=0.001,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_depth_factor=True,
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
):
    if not items:
        return []
    gaps = [int(item[2][3] - item[2][2]) for item in items]
    if len(set(gaps)) != 1:
        raise ValueError('cross-lake transition chunk must contain one rollout gap.')

    ref_lake = items[0][1]
    device = ref_lake['depths'].device
    batch_size = len(items)
    ends = [item[2][1] for item in items]
    start_indices = [int(item[2][2]) for item in items]
    end_indices = [int(item[2][3]) for item in items]

    prediction, start_mask = _stack_target_batch_for_items(items, lookup_split, lambda item: item[2][0])
    target, target_mask = _stack_target_batch_for_items(items, lookup_split, lambda item: item[2][1])
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    static_features = _stack_static_features_for_items(items)

    lst_losses = [[] for _ in range(batch_size)]
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    energy_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_scales = None
    final_diagnostics = None

    for offset in range(gaps[0]):
        previous = prediction
        day_indices = [start_idx + offset for start_idx in start_indices]
        next_indices = [day_idx + 1 for day_idx in day_indices]
        current_row = _stack_forcing_batch_for_items(items, day_indices)
        next_row = _stack_forcing_batch_for_items(items, next_indices)
        prediction, freezing_storage, diagnostics = model.step(
            prediction,
            current_row,
            static_features,
            next_forcing_row=next_row,
            return_diagnostics=True,
            task_mode=task_mode,
            depths=ref_lake['depths'],
            area_profile=ref_lake['area'],
            diagnostic_mode=step_diagnostic_mode,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        final_diagnostics = diagnostics
        daily_tendency_vectors.append(
            _daily_tendency_loss_per_sample(previous, prediction, ref_lake['depths'], ref_lake['max_depth'])
        )
        residual_regularization_vectors.append(_residual_regularization_loss_per_sample(diagnostics))
        physical_scale_regularization_vectors.append(_physical_scale_regularization_loss_per_sample(diagnostics))
        physical_scale_smoothness_vectors.append(
            _physical_scale_smoothness_loss_per_sample(diagnostics, previous_scales)
        )
        kd_prior_regularization_vectors.append(_kd_prior_regularization_loss_per_sample(diagnostics))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)

        lst_loss_vec, _, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
        for sample_idx in range(batch_size):
            if bool(lst_mask[sample_idx].detach().cpu().item()):
                lst_losses[sample_idx].append(lst_loss_vec[sample_idx])

        energy_residual = (diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']) / 150.0
        energy_vectors.append(energy_residual.reshape(-1).pow(2))

    profile_loss_vec = _masked_huber_profile_loss_per_sample(
        prediction,
        target,
        mask=target_mask,
        delta=profile_huber_delta,
    )
    physics_loss_vec = _profile_physics_loss_per_sample(prediction)
    daily_tendency_vec = torch.stack(daily_tendency_vectors, dim=0).mean(dim=0)
    residual_regularization_vec = torch.stack(residual_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_regularization_vec = torch.stack(physical_scale_regularization_vectors, dim=0).mean(dim=0)
    physical_scale_smoothness_vec = torch.stack(physical_scale_smoothness_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_vec = torch.stack(kd_prior_regularization_vectors, dim=0).mean(dim=0)
    kd_prior_regularization_weighted_vec = float(kd_prior_regularization_weight) * kd_prior_regularization_vec
    adaptive_parameter_regularization_vec = torch.stack(adaptive_parameter_regularization_vectors, dim=0).mean(dim=0)
    energy_loss_vec = torch.stack(energy_vectors, dim=0).mean(dim=0)

    if float(heat_content_transition_weight) > 0.0:
        for sample_idx, item in enumerate(items):
            sample_lake = item[1]
            _append_weighted_heat_content_transition_loss(
                heat_content_losses[sample_idx],
                heat_content_weighted_losses[sample_idx],
                heat_content_effective_weights[sample_idx],
                base_weight=heat_content_transition_weight,
                target_date=ends[sample_idx],
                lake=sample_lake,
                start_profile=start_profile[sample_idx:sample_idx + 1],
                end_prediction=prediction[sample_idx:sample_idx + 1],
                end_target=target[sample_idx:sample_idx + 1],
                start_mask=start_mask[sample_idx:sample_idx + 1],
                end_mask=target_mask[sample_idx:sample_idx + 1],
                delta_seconds=float(end_indices[sample_idx] - start_indices[sample_idx]) * 86400.0,
                season_factors=sample_lake['heat_content_transition_season_factors'],
                use_depth_factor=heat_content_transition_depth_factor,
                effective_max=heat_content_transition_effective_max,
                min_full_column_coverage=heat_content_full_column_min_coverage,
            )

    results = []
    for sample_idx, item in enumerate(items):
        _, heat_content_weighted_loss, heat_content_details = _heat_content_transition_loss_details(
            heat_content_losses[sample_idx],
            heat_content_weighted_losses[sample_idx],
            heat_content_effective_weights[sample_idx],
            device=device,
        )
        lst_loss = _mean_or_zero(lst_losses[sample_idx], device=device)
        total = (
            profile_loss_vec[sample_idx]
            + physics_loss_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + float(lst_surface_weight) * lst_loss
            + float(energy_balance_weight) * energy_loss_vec[sample_idx]
        )
        results.append((
            item[0],
            total,
            {
                'profile_loss': profile_loss_vec[sample_idx].detach(),
                'lst_loss': lst_loss.detach(),
                'energy_loss': energy_loss_vec[sample_idx].detach(),
                'daily_tendency_loss': daily_tendency_vec[sample_idx].detach(),
                'residual_regularization_loss': residual_regularization_vec[sample_idx].detach(),
                'physical_scale_reg_loss': physical_scale_regularization_vec[sample_idx].detach(),
                'physical_scale_smooth_loss': physical_scale_smoothness_vec[sample_idx].detach(),
                'kd_prior_regularization_loss': kd_prior_regularization_vec[sample_idx].detach(),
                'kd_prior_regularization_weighted_loss': kd_prior_regularization_weighted_vec[sample_idx].detach(),
                'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
                **heat_content_details,
                **_scale_detail_record_for_sample(final_diagnostics, sample_idx),
            },
        ))
    return results


def _transition_losses_for_lakes_cross_batch(
    model,
    lakes,
    *,
    pair_key='train',
    transition_batch_mode='off',
    transition_batch_size=0,
    cross_lake_batch_size=0,
    **kwargs,
):
    if transition_batch_mode != 'on':
        results = {}
        for lake_idx, lake in enumerate(lakes):
            lake_kwargs = dict(kwargs)
            lake_kwargs.setdefault(
                'heat_content_transition_season_factors',
                lake['heat_content_transition_season_factors'],
            )
            results[lake_idx] = _transition_losses_for_lake(
                model,
                lake,
                lake['pairs'][pair_key],
                transition_batch_mode=transition_batch_mode,
                transition_batch_size=transition_batch_size,
                lookup_split=pair_key,
                **lake_kwargs,
            )
        return results

    grouped = {}
    for bucket_entries in _cross_lake_bucket_entries(lakes).values():
        for lake_idx, lake in bucket_entries:
            for pair in lake['pairs'][pair_key]:
                gap = int(pair[3] - pair[2])
                key = (_cross_lake_batch_key(lake), gap)
                grouped.setdefault(key, []).append((lake_idx, lake, pair))

    results = {lake_idx: ([], []) for lake_idx in range(len(lakes))}
    batch_size = int(cross_lake_batch_size or transition_batch_size or 0)
    for key in sorted(grouped, key=lambda value: (value[0][0], value[1])):
        for chunk in _batch_chunks(grouped[key], batch_size):
            for lake_idx, loss, detail in _transition_loss_cross_lake_batch_chunk(
                model,
                chunk,
                lookup_split=pair_key,
                **kwargs,
            ):
                results[lake_idx][0].append(loss)
                results[lake_idx][1].append(detail)
    return results


@torch.no_grad()
def evaluate_lake_pairs(model, lake, pairs, *, task_mode='analysis', hard_density_stability=False):
    """Return transition-end RMSE over supplied pairs."""
    if not pairs:
        return np.nan
    errors = []
    for start, end, start_idx, end_idx in pairs:
        if start not in lake['lookups']['all'] or end not in lake['lookups']['all']:
            continue
        prediction, _start_mask = _target_tensor_and_mask(lake, 'all', start)
        prediction = prediction.to(device=lake['depths'].device, dtype=torch.float32).clone()
        freezing_storage = torch.zeros_like(prediction)
        target, target_mask = _target_tensor_and_mask(lake, 'all', end)
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        for day_idx in range(start_idx, end_idx):
            next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
            prediction, freezing_storage = model.step(
                prediction,
                lake['forcing_rows'][day_idx],
                lake['static_features'],
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=lake['depths'],
                area_profile=lake['area'],
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
        valid = torch.isfinite(prediction) & torch.isfinite(target)
        if target_mask is not None:
            valid = valid & target_mask.to(device=prediction.device, dtype=torch.bool).reshape(1, -1)
        if torch.any(valid):
            errors.append(torch.mean((prediction[valid] - target[valid]).pow(2)).detach().cpu().item())
    return float(np.sqrt(np.mean(errors))) if errors else np.nan


@torch.no_grad()
def evaluate_lake_pair_horizons(
    model,
    lake,
    pairs,
    *,
    horizons=(1, 3, 7, 14, 30, 60),
    task_mode='analysis',
    hard_density_stability=False,
):
    """Cumulative transition RMSE by rollout horizon."""
    errors_by_horizon = {int(horizon): [] for horizon in horizons}
    if not pairs:
        return _horizon_metric_record(errors_by_horizon)
    horizons = tuple(sorted(int(horizon) for horizon in horizons))
    for start, end, start_idx, end_idx in pairs:
        if start not in lake['lookups']['all'] or end not in lake['lookups']['all']:
            continue
        gap_days = int(end_idx - start_idx)
        prediction, _start_mask = _target_tensor_and_mask(lake, 'all', start)
        prediction = prediction.to(device=lake['depths'].device, dtype=torch.float32).clone()
        freezing_storage = torch.zeros_like(prediction)
        target, target_mask = _target_tensor_and_mask(lake, 'all', end)
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        for day_idx in range(start_idx, end_idx):
            next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
            prediction, freezing_storage = model.step(
                prediction,
                lake['forcing_rows'][day_idx],
                lake['static_features'],
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=lake['depths'],
                area_profile=lake['area'],
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
        valid = torch.isfinite(prediction) & torch.isfinite(target)
        if target_mask is not None:
            valid = valid & target_mask.to(device=prediction.device, dtype=torch.bool).reshape(1, -1)
        if not torch.any(valid):
            continue
        mse = torch.mean((prediction[valid] - target[valid]).pow(2)).detach().cpu().item()
        for horizon in horizons:
            if gap_days <= horizon:
                errors_by_horizon[horizon].append(mse)
    return _horizon_metric_record(errors_by_horizon)


@torch.no_grad()
def evaluate_persistence_pairs(lake, pairs):
    """Return RMSE from a no-change persistence baseline over transition pairs."""
    if not pairs:
        return np.nan
    errors = []
    for start, end, _start_idx, _end_idx in pairs:
        if start not in lake['lookups']['all'] or end not in lake['lookups']['all']:
            continue
        mask = _lookup_mask(lake, 'all', end)
        rmse = _profile_rmse(lake['lookups']['all'][start], lake['lookups']['all'][end], mask=mask)
        if np.isfinite(rmse):
            errors.append(rmse ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else np.nan


@torch.no_grad()
def evaluate_lake_free_roll(
    model,
    lake,
    *,
    task_mode='analysis',
    horizons=(1, 3, 7, 14, 30, 60),
    init_mode='profile',
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_start_date=None,
    hard_density_stability=False,
):
    """Evaluate full free-roll against all available profile dates after initialization."""
    model.eval()
    df = lake['df']
    all_lookup = lake['lookups']['all']
    if not all_lookup:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            **_empty_observed_point_metrics(),
        }

    date_to_index = _date_index_map(df)
    init_state = initialize_rollout_state(
        model=model,
        df=df,
        depths=lake['depths_np'],
        all_lookup=all_lookup,
        forcing_rows=lake['forcing_rows'],
        static_features=lake['static_features'],
        metadata=lake['metadata'],
        device=lake['depths'].device,
        init_mode=init_mode,
        rollout_start_date=rollout_start_date,
        spinup_days=spinup_days,
        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
        task_mode=task_mode,
        area_profile=lake['area'],
        hard_density_stability=hard_density_stability,
    )
    current = init_state['current']
    freezing_storage = init_state.get('freezing_storage_j_m2', torch.zeros_like(current))
    rollout_start_idx = int(init_state['rollout_start_idx'])
    predictions_by_index = {
        int(idx): np.asarray(profile, dtype=np.float32)
        for idx, profile in init_state['profiles_by_index'].items()
    }
    for day_idx in range(rollout_start_idx, len(df) - 1):
        next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
        current, freezing_storage = model.step(
            current,
            lake['forcing_rows'][day_idx],
            lake['static_features'],
            next_forcing_row=next_row,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        predictions_by_index[day_idx + 1] = current.detach().cpu().numpy().reshape(-1)

    errors = []
    biases = []
    post_spinup_errors = []
    post_spinup_biases = []
    horizon_errors = {int(horizon): [] for horizon in horizons}
    horizon_biases = {int(horizon): [] for horizon in horizons}
    for obs_date, target in all_lookup.items():
        obs_idx = date_to_index.get(pd.Timestamp(obs_date).normalize())
        if obs_idx is None or obs_idx <= rollout_start_idx or obs_idx not in predictions_by_index:
            continue
        prediction = predictions_by_index[obs_idx]
        mask = _lookup_mask(lake, 'all', obs_date)
        valid = np.isfinite(prediction) & np.isfinite(target)
        if mask is not None:
            valid = valid & np.asarray(mask, dtype=bool)
        if not np.any(valid):
            continue
        diff = prediction[valid] - np.asarray(target, dtype=np.float64)[valid]
        errors.extend(diff.tolist())
        biases.append(float(np.mean(diff)))
        post_spinup_errors.extend(diff.tolist())
        post_spinup_biases.append(float(np.mean(diff)))
        gap_days = int(obs_idx - rollout_start_idx)
        for horizon in horizon_errors:
            if gap_days <= horizon:
                horizon_errors[horizon].extend((diff ** 2).tolist())
                horizon_biases[horizon].append(float(np.mean(diff)))

    observed_point_metrics = _observed_point_metrics_from_predictions(lake, predictions_by_index)
    if not errors:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            'horizon_metrics': _horizon_metric_record(horizon_errors, horizon_biases),
            'post_spinup_rmse': np.nan,
            'post_spinup_bias': np.nan,
            'init_mode': init_state['init_mode'],
            'spinup_days_used': init_state['spinup_days_used'],
            **observed_point_metrics,
        }
    errors = np.asarray(errors, dtype=np.float64)
    post_spinup_errors = np.asarray(post_spinup_errors, dtype=np.float64)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
        'n_profiles': int(len(biases)),
        'horizon_metrics': _horizon_metric_record(horizon_errors, horizon_biases),
        'post_spinup_rmse': float(np.sqrt(np.mean(post_spinup_errors ** 2))) if post_spinup_errors.size else np.nan,
        'post_spinup_bias': float(np.mean(post_spinup_errors)) if post_spinup_errors.size else np.nan,
        'post_spinup_profiles': int(len(post_spinup_biases)),
        'init_mode': init_state['init_mode'],
        'spinup_days_used': init_state['spinup_days_used'],
        **observed_point_metrics,
    }


def _free_roll_metrics_from_predictions(
    lake,
    init_state,
    predictions_by_index,
    *,
    horizons=(1, 3, 7, 14, 30, 60),
):
    all_lookup = lake['lookups']['all']
    date_to_index = _date_index_map(lake['df'])
    rollout_start_idx = int(init_state['rollout_start_idx'])
    errors = []
    biases = []
    post_spinup_errors = []
    post_spinup_biases = []
    horizon_errors = {int(horizon): [] for horizon in horizons}
    horizon_biases = {int(horizon): [] for horizon in horizons}
    for obs_date, target in all_lookup.items():
        obs_idx = date_to_index.get(pd.Timestamp(obs_date).normalize())
        if obs_idx is None or obs_idx <= rollout_start_idx or obs_idx not in predictions_by_index:
            continue
        prediction = predictions_by_index[obs_idx]
        mask = _lookup_mask(lake, 'all', obs_date)
        valid = np.isfinite(prediction) & np.isfinite(target)
        if mask is not None:
            valid = valid & np.asarray(mask, dtype=bool)
        if not np.any(valid):
            continue
        diff = prediction[valid] - np.asarray(target, dtype=np.float64)[valid]
        errors.extend(diff.tolist())
        biases.append(float(np.mean(diff)))
        post_spinup_errors.extend(diff.tolist())
        post_spinup_biases.append(float(np.mean(diff)))
        gap_days = int(obs_idx - rollout_start_idx)
        for horizon in horizon_errors:
            if gap_days <= horizon:
                horizon_errors[horizon].extend((diff ** 2).tolist())
                horizon_biases[horizon].append(float(np.mean(diff)))

    if not errors:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            'horizon_metrics': _horizon_metric_record(horizon_errors, horizon_biases),
            'post_spinup_rmse': np.nan,
            'post_spinup_bias': np.nan,
            'init_mode': init_state['init_mode'],
            'spinup_days_used': init_state['spinup_days_used'],
        }
    errors = np.asarray(errors, dtype=np.float64)
    post_spinup_errors = np.asarray(post_spinup_errors, dtype=np.float64)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
        'n_profiles': int(len(biases)),
        'horizon_metrics': _horizon_metric_record(horizon_errors, horizon_biases),
        'post_spinup_rmse': float(np.sqrt(np.mean(post_spinup_errors ** 2))) if post_spinup_errors.size else np.nan,
        'post_spinup_bias': float(np.mean(post_spinup_errors)) if post_spinup_errors.size else np.nan,
        'post_spinup_profiles': int(len(post_spinup_biases)),
        'init_mode': init_state['init_mode'],
        'spinup_days_used': init_state['spinup_days_used'],
    }


def _lake_rollout_geometry_key(lake):
    depths = np.asarray(lake.get('depths_np', lake['depths'].detach().cpu().numpy()), dtype=np.float32)
    area = np.asarray(lake.get('area_np', lake['area'].detach().cpu().numpy()), dtype=np.float32)
    return (
        tuple(np.round(depths.reshape(-1), 6).tolist()),
        tuple(np.round(area.reshape(-1), 6).tolist()),
        str(lake['depths'].device),
    )


@torch.no_grad()
def _evaluate_free_roll_compatible_lakes_batched(
    model,
    lakes,
    init_states,
    *,
    task_mode='analysis',
    horizons=(1, 3, 7, 14, 30, 60),
    hard_density_stability=False,
    batch_size=32,
):
    """Roll compatible lake geometries together without assuming a fixed heldout count."""
    if not lakes:
        return {}
    model.eval()
    batch_size = int(batch_size or 0)
    if batch_size <= 0:
        batch_size = len(lakes)
    depths = lakes[0]['depths']
    area = lakes[0]['area']
    currents = [
        init_state['current'].reshape(-1).to(device=depths.device, dtype=torch.float32)
        for init_state in init_states
    ]
    freezing_storages = [
        init_state.get('freezing_storage_j_m2', torch.zeros_like(init_state['current']))
        .reshape(-1)
        .to(device=depths.device, dtype=torch.float32)
        for init_state in init_states
    ]
    rollout_start_indices = [int(init_state['rollout_start_idx']) for init_state in init_states]
    predictions_by_lake = []
    for init_state in init_states:
        predictions_by_lake.append({
            int(idx): np.asarray(profile, dtype=np.float32)
            for idx, profile in init_state['profiles_by_index'].items()
        })

    min_start = min(rollout_start_indices)
    max_len = max(len(lake['df']) for lake in lakes)
    for day_idx in range(min_start, max_len - 1):
        active_indices = [
            sample_idx
            for sample_idx, lake in enumerate(lakes)
            if day_idx >= rollout_start_indices[sample_idx] and day_idx < len(lake['df']) - 1
        ]
        for chunk in _batch_chunks(active_indices, batch_size):
            forcing_rows = [lakes[sample_idx]['forcing_rows'][day_idx] for sample_idx in chunk]
            next_forcing_rows = [lakes[sample_idx]['forcing_rows'][day_idx + 1] for sample_idx in chunk]
            current_batch = torch.stack([currents[sample_idx] for sample_idx in chunk], dim=0)
            storage_batch = torch.stack([freezing_storages[sample_idx] for sample_idx in chunk], dim=0)
            static_batch = torch.stack(
                [
                    lakes[sample_idx]['static_features'].reshape(-1).to(device=depths.device, dtype=current_batch.dtype)
                    for sample_idx in chunk
                ],
                dim=0,
            )
            stepped, stepped_storage = model.step(
                current_batch,
                _stack_forcing_rows(forcing_rows),
                static_batch,
                next_forcing_row=_stack_forcing_rows(next_forcing_rows),
                task_mode=task_mode,
                depths=depths,
                area_profile=area,
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=storage_batch,
                return_freezing_storage=True,
            )
            for local_idx, sample_idx in enumerate(chunk):
                currents[sample_idx] = stepped[local_idx].detach().clone()
                freezing_storages[sample_idx] = stepped_storage[local_idx].detach().clone()
                predictions_by_lake[sample_idx][day_idx + 1] = (
                    currents[sample_idx].detach().cpu().numpy().reshape(-1)
                )

    results = {}
    for lake, init_state, predictions in zip(lakes, init_states, predictions_by_lake):
        metrics = _free_roll_metrics_from_predictions(
            lake,
            init_state,
            predictions,
            horizons=horizons,
        )
        metrics.update(_observed_point_metrics_from_predictions(lake, predictions))
        results[lake['lake_id']] = metrics
    return results


@torch.no_grad()
def evaluate_lake_rolling_start_horizons(
    model,
    lake,
    *,
    horizons=(1, 3, 7, 14, 30, 60),
    task_mode='analysis',
    max_start_profiles=80,
    hard_density_stability=False,
    lookup_split='all',
    batch_size=None,
    rollout_batch_step_mode='off',
):
    """Evaluate horizon skill by reinitializing at every observed profile date."""
    model.eval()
    df = lake['df']
    lookup_split = str(lookup_split or 'all').strip().lower()
    if lookup_split not in lake['lookups']:
        raise ValueError(f"Unknown lookup_split {lookup_split!r}.")
    lookup = lake['lookups'][lookup_split]
    if not lookup:
        return _horizon_metric_record({int(h): [] for h in horizons})
    horizons = tuple(sorted(int(horizon) for horizon in horizons if int(horizon) > 0))
    errors_by_horizon = {int(horizon): [] for horizon in horizons}
    biases_by_horizon = {int(horizon): [] for horizon in horizons}
    date_to_index = _date_index_map(df)
    index_to_date = {
        int(idx): pd.Timestamp(date).normalize()
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }
    start_items = list(lookup.items())
    if max_start_profiles is not None and int(max_start_profiles) > 0 and len(start_items) > int(max_start_profiles):
        selected = np.linspace(0, len(start_items) - 1, int(max_start_profiles), dtype=int)
        start_items = [start_items[int(idx)] for idx in selected]
    for start_date, _start_profile in start_items:
        start_idx = date_to_index.get(pd.Timestamp(start_date).normalize())
        if start_idx is None:
            continue
        valid_targets = {
            int(horizon): index_to_date.get(int(start_idx + horizon))
            for horizon in horizons
            if int(start_idx + horizon) < len(df)
        }
        valid_targets = {
            horizon: date
            for horizon, date in valid_targets.items()
            if date is not None and date in lookup
        }
        if not valid_targets:
            continue
        current, _start_mask = _target_tensor_and_mask(lake, lookup_split, start_date)
        current = current.to(device=lake['depths'].device, dtype=torch.float32).clone()
        freezing_storage = torch.zeros_like(current)
        predictions_by_horizon = {}
        max_horizon = max(valid_targets)
        for day_idx in range(start_idx, start_idx + max_horizon):
            next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
            current, freezing_storage = model.step(
                current,
                lake['forcing_rows'][day_idx],
                lake['static_features'],
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=lake['depths'],
                area_profile=lake['area'],
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
            horizon = int(day_idx + 1 - start_idx)
            if horizon in valid_targets:
                predictions_by_horizon[horizon] = current.detach().cpu().numpy().reshape(-1)
        for horizon, target_date in valid_targets.items():
            prediction = predictions_by_horizon.get(horizon)
            if prediction is None:
                continue
            target, mask = _target_tensor_and_mask(lake, lookup_split, target_date)
            target = target.reshape(-1).to(device=lake['depths'].device, dtype=torch.float32)
            target_np = target.detach().cpu().numpy().astype(np.float64)
            valid = np.isfinite(prediction) & np.isfinite(target_np)
            if mask is not None:
                valid = valid & mask.reshape(-1).detach().cpu().numpy().astype(bool)
            if not np.any(valid):
                continue
            diff = np.asarray(prediction, dtype=np.float64)[valid] - target_np[valid]
            errors_by_horizon[horizon].extend((diff ** 2).tolist())
            biases_by_horizon[horizon].append(float(np.mean(diff)))
    return _horizon_metric_record(errors_by_horizon, biases_by_horizon)


@torch.no_grad()
def evaluate_lake_rolling_start_horizons_batched(
    model,
    lake,
    *,
    horizons=(1, 3, 7, 14, 30, 60),
    task_mode='analysis',
    max_start_profiles=80,
    hard_density_stability=False,
    lookup_split='all',
    batch_size=32,
    rollout_batch_step_mode='off',
):
    """Evaluate rolling-start horizon skill with multiple start profiles per rollout batch."""
    model.eval()
    df = lake['df']
    lookup_split = str(lookup_split or 'all').strip().lower()
    if lookup_split not in lake['lookups']:
        raise ValueError(f"Unknown lookup_split {lookup_split!r}.")
    lookup = lake['lookups'][lookup_split]
    if not lookup:
        return _horizon_metric_record({int(h): [] for h in horizons})
    horizons = tuple(sorted(int(horizon) for horizon in horizons if int(horizon) > 0))
    errors_by_horizon = {int(horizon): [] for horizon in horizons}
    biases_by_horizon = {int(horizon): [] for horizon in horizons}
    date_to_index = _date_index_map(df)
    index_to_date = {
        int(idx): pd.Timestamp(date).normalize()
        for idx, date in enumerate(pd.to_datetime(df['Date']))
    }
    start_items = list(lookup.items())
    if max_start_profiles is not None and int(max_start_profiles) > 0 and len(start_items) > int(max_start_profiles):
        selected = np.linspace(0, len(start_items) - 1, int(max_start_profiles), dtype=int)
        start_items = [start_items[int(idx)] for idx in selected]

    sequences = []
    for start_date, _start_profile in start_items:
        start_idx = date_to_index.get(pd.Timestamp(start_date).normalize())
        if start_idx is None:
            continue
        valid_targets = {
            int(horizon): index_to_date.get(int(start_idx + horizon))
            for horizon in horizons
            if int(start_idx + horizon) < len(df)
        }
        valid_targets = {
            horizon: date
            for horizon, date in valid_targets.items()
            if date is not None and date in lookup
        }
        if valid_targets:
            sequences.append((int(start_idx), pd.Timestamp(start_date).normalize(), valid_targets))
    if not sequences:
        return _horizon_metric_record(errors_by_horizon, biases_by_horizon)

    batch_size = int(batch_size or 0)
    if batch_size <= 0:
        batch_size = len(sequences)
    for chunk in _batch_chunks(sequences, batch_size):
        start_indices = torch.as_tensor(
            [item[0] for item in chunk],
            dtype=torch.long,
            device=lake['depths'].device,
        )
        current, _start_mask = _target_tensor_and_mask_batch(
            lake,
            lookup_split,
            [item[1] for item in chunk],
        )
        current = current.to(device=lake['depths'].device, dtype=torch.float32).clone()
        freezing_storage = torch.zeros_like(current)
        valid_targets_by_sample = [item[2] for item in chunk]
        max_horizon = max(max(targets) for targets in valid_targets_by_sample)
        predictions_by_sample_horizon: list[dict[int, torch.Tensor]] = [
            {} for _ in chunk
        ]
        static_features = lake['static_features'].expand(len(chunk), -1)
        use_rollout_batch = (
            str(rollout_batch_step_mode or 'off').strip().lower() == 'on'
            and bool(torch.all(start_indices + int(max_horizon) < len(lake['forcing_rows'])).detach().cpu())
        )
        if use_rollout_batch:
            forcing_sequence = [
                _forcing_row_batch(lake, start_indices + int(offset))
                for offset in range(max_horizon)
            ]
            next_forcing_sequence = [
                _forcing_row_batch(lake, start_indices + int(offset) + 1)
                for offset in range(max_horizon)
            ]
            states = model.rollout_batch(
                current,
                forcing_sequence,
                static_features,
                next_forcing_rows=next_forcing_sequence,
                task_mode=task_mode,
                depths=lake['depths'],
                area_profile=lake['area'],
                hard_density_stability=hard_density_stability,
            )
            for sample_idx, targets in enumerate(valid_targets_by_sample):
                for horizon in targets:
                    predictions_by_sample_horizon[sample_idx][int(horizon)] = (
                        states[int(horizon) - 1, sample_idx].detach().clone().reshape(-1)
                    )
        else:
            for offset in range(max_horizon):
                active_indices = [
                    sample_idx
                    for sample_idx, targets in enumerate(valid_targets_by_sample)
                    if int(offset) < max(targets)
                ]
                if not active_indices:
                    continue
                active_tensor = torch.as_tensor(active_indices, dtype=torch.long, device=lake['depths'].device)
                active_start_indices = start_indices.index_select(0, active_tensor)
                day_indices = active_start_indices + int(offset)
                next_day_indices = day_indices + 1
                forcing_row = _forcing_row_batch(lake, day_indices)
                next_forcing_row = _forcing_row_batch(lake, next_day_indices)
                stepped, stepped_storage = model.step(
                    current.index_select(0, active_tensor),
                    forcing_row,
                    static_features.index_select(0, active_tensor),
                    next_forcing_row=next_forcing_row,
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=lake['area'],
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage.index_select(0, active_tensor),
                    return_freezing_storage=True,
                )
                current = current.clone()
                freezing_storage = freezing_storage.clone()
                current.index_copy_(0, active_tensor, stepped)
                freezing_storage.index_copy_(0, active_tensor, stepped_storage)
                horizon = int(offset + 1)
                for sample_idx, targets in enumerate(valid_targets_by_sample):
                    if horizon in targets:
                        predictions_by_sample_horizon[sample_idx][horizon] = (
                            current[sample_idx].detach().clone().reshape(-1)
                        )
        for sample_predictions, targets in zip(predictions_by_sample_horizon, valid_targets_by_sample):
            for horizon, target_date in targets.items():
                prediction = sample_predictions.get(horizon)
                if prediction is None:
                    continue
                target, mask = _target_tensor_and_mask(lake, lookup_split, target_date)
                target = target.reshape(-1).to(device=prediction.device, dtype=prediction.dtype)
                valid = torch.isfinite(prediction) & torch.isfinite(target)
                if mask is not None:
                    valid = valid & mask.reshape(-1).to(device=prediction.device, dtype=torch.bool)
                if not torch.any(valid):
                    continue
                diff = prediction[valid] - target[valid]
                errors_by_horizon[horizon].extend((diff.detach().double().cpu().numpy() ** 2).tolist())
                biases_by_horizon[horizon].append(float(diff.mean().detach().cpu()))
    return _horizon_metric_record(errors_by_horizon, biases_by_horizon)


@torch.no_grad()
def evaluate_lakes_rolling_start_horizons(
    model,
    lakes,
    *,
    horizons=(1, 3, 7, 14, 30, 60),
    task_mode='analysis',
    max_start_profiles=80,
    hard_density_stability=False,
    lookup_split='all',
    batch_size=32,
    rollout_batch_step_mode='off',
    use_batched=True,
):
    """Evaluate rolling-start horizons for an arbitrary specified lake list."""
    eval_fn = evaluate_lake_rolling_start_horizons_batched if use_batched else evaluate_lake_rolling_start_horizons
    return {
        lake['lake_id']: eval_fn(
            model,
            lake,
            horizons=horizons,
            task_mode=task_mode,
            max_start_profiles=max_start_profiles,
            hard_density_stability=hard_density_stability,
            lookup_split=lookup_split,
            batch_size=batch_size,
            rollout_batch_step_mode=rollout_batch_step_mode,
        )
        for lake in lakes
    }


@torch.no_grad()
def evaluate_heldout_free_rolls(
    model,
    lakes,
    *,
    task_mode='analysis',
    horizons=(1, 3, 7, 14, 30, 60),
    init_mode='profile',
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_start_date=None,
    hard_density_stability=False,
    batch_size=32,
):
    """Evaluate specified held-out full free-rolls, batching compatible lake geometries."""
    model.eval()
    if not lakes:
        return {}
    scalar_results = {}
    batch_groups = {}
    for lake in lakes:
        if not lake['lookups']['all']:
            scalar_results[lake['lake_id']] = evaluate_lake_free_roll(
                model,
                lake,
                task_mode=task_mode,
                horizons=horizons,
                init_mode=init_mode,
                spinup_days=spinup_days,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_start_date=rollout_start_date,
                hard_density_stability=hard_density_stability,
            )
            continue
        batch_groups.setdefault(_lake_rollout_geometry_key(lake), []).append(lake)

    results = dict(scalar_results)
    for group_lakes in batch_groups.values():
        if len(group_lakes) == 1:
            lake = group_lakes[0]
            results[lake['lake_id']] = evaluate_lake_free_roll(
                model,
                lake,
                task_mode=task_mode,
                horizons=horizons,
                init_mode=init_mode,
                spinup_days=spinup_days,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_start_date=rollout_start_date,
                hard_density_stability=hard_density_stability,
            )
        else:
            group_init_states = [
                initialize_rollout_state(
                    model=model,
                    df=lake['df'],
                    depths=lake['depths_np'],
                    all_lookup=lake['lookups']['all'],
                    forcing_rows=lake['forcing_rows'],
                    static_features=lake['static_features'],
                    metadata=lake['metadata'],
                    device=lake['depths'].device,
                    init_mode=init_mode,
                    rollout_start_date=rollout_start_date,
                    spinup_days=spinup_days,
                    spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                    spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                    task_mode=task_mode,
                    area_profile=lake['area'],
                    hard_density_stability=hard_density_stability,
                )
                for lake in group_lakes
            ]
            results.update(_evaluate_free_roll_compatible_lakes_batched(
                model,
                group_lakes,
                group_init_states,
                task_mode=task_mode,
                horizons=horizons,
                hard_density_stability=hard_density_stability,
                batch_size=batch_size,
            ))
    return {lake['lake_id']: results[lake['lake_id']] for lake in lakes}


@torch.no_grad()
def evaluate_lakes_free_rolls(
    model,
    lakes,
    **kwargs,
):
    """Alias for free-roll evaluation over an arbitrary specified lake list."""
    return evaluate_heldout_free_rolls(model, lakes, **kwargs)


def _write_physical_scale_diagnostics(diagnostics_df, metadata, output_dir, suffix):
    scale_columns = [
        'Date',
        'spinup_phase',
        'shortwave_absorption_scale',
        'surface_flux_bias_wm2',
        'shortwave_to_water_wm2',
        'ice_shortwave_transmission',
        'surface_cooling_scale_raw',
        'surface_cooling_scale_effective',
        'open_water_sensible_heat_wm2',
        'open_water_latent_heat_wm2',
        'temperature_floor_heat_injection_wm2',
        'freezing_storage_j_m2',
        'freezing_storage_change_wm2',
        'sensible_heat_tendency_wm2',
        'effective_heat_tendency_wm2',
        'adaptive_wind_kz_scale',
        'adaptive_turbulent_flux_blend_alpha',
        'adaptive_kd_multiplier',
        'adaptive_turbulent_exchange_scale',
        'adaptive_convective_mixing_scale',
        'adaptive_ice_shortwave_scale',
        'adaptive_parameter_regularization_loss',
        'lake_shape_wind_factor',
        'lake_shape_decay_depth_m',
        'stratification_mixing_gate_mean',
        'stratification_mixing_gate_min',
        'stratification_mixing_gate_deep_mean',
    ]
    available = [column for column in scale_columns if column in diagnostics_df.columns]
    if 'Date' not in available or 'shortwave_absorption_scale' not in available:
        return {}
    scale_df = diagnostics_df[available].copy()
    scale_df['Date'] = pd.to_datetime(scale_df['Date'], errors='coerce')
    scale_df = scale_df.dropna(subset=['Date'])
    scale_df['month'] = scale_df['Date'].dt.month
    timeseries_csv = output_dir / f"{metadata['file_tag']}_{suffix}_physical_scale_timeseries.csv"
    monthly_csv = output_dir / f"{metadata['file_tag']}_{suffix}_physical_scale_monthly_summary.csv"
    plot_path = output_dir / f"{metadata['file_tag']}_{suffix}_physical_scale_diagnostic.png"
    scale_df.to_csv(timeseries_csv, index=False)
    numeric_cols = [
        column for column in (
            'shortwave_absorption_scale',
            'surface_flux_bias_wm2',
            'shortwave_to_water_wm2',
            'ice_shortwave_transmission',
            'surface_cooling_scale_raw',
            'surface_cooling_scale_effective',
            'open_water_sensible_heat_wm2',
            'open_water_latent_heat_wm2',
            'temperature_floor_heat_injection_wm2',
            'freezing_storage_j_m2',
            'freezing_storage_change_wm2',
            'sensible_heat_tendency_wm2',
            'effective_heat_tendency_wm2',
            'adaptive_wind_kz_scale',
            'adaptive_turbulent_flux_blend_alpha',
            'adaptive_kd_multiplier',
            'adaptive_turbulent_exchange_scale',
            'adaptive_convective_mixing_scale',
            'adaptive_ice_shortwave_scale',
            'adaptive_parameter_regularization_loss',
            'lake_shape_wind_factor',
            'lake_shape_decay_depth_m',
            'stratification_mixing_gate_mean',
            'stratification_mixing_gate_min',
            'stratification_mixing_gate_deep_mean',
        )
        if column in scale_df.columns
    ]
    monthly = scale_df.groupby('month', as_index=False)[numeric_cols].agg(['mean', 'min', 'max'])
    monthly.columns = [
        '_'.join(str(part) for part in column if part)
        for column in monthly.columns.to_flat_index()
    ]
    monthly.to_csv(monthly_csv, index=False)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
        ax.plot(scale_df['Date'], scale_df['shortwave_absorption_scale'], label='Shortwave scale', color='#d97706')
        if 'surface_cooling_scale_raw' in scale_df.columns:
            ax.plot(scale_df['Date'], scale_df['surface_cooling_scale_raw'], label='Cooling scale raw', color='#2563eb')
        if 'surface_cooling_scale_effective' in scale_df.columns:
            ax.plot(scale_df['Date'], scale_df['surface_cooling_scale_effective'], label='Cooling scale effective', color='#0f766e', alpha=0.75)
        if 'surface_flux_bias_wm2' in scale_df.columns:
            ax2 = ax.twinx()
            ax2.plot(scale_df['Date'], scale_df['surface_flux_bias_wm2'], label='Flux bias', color='#7c3aed', alpha=0.50)
            ax2.set_ylabel('Flux bias (W/m2)')
        ax.axhline(1.0, color='black', linewidth=1.0, alpha=0.45)
        ax.set_title(f"{metadata.get('lake_name', metadata['file_tag'])} | Learned Physical Scales")
        ax.set_ylabel('Scale')
        ax.set_xlabel('Date')
        ax.grid(alpha=0.25)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
    except Exception:
        plot_path = None
    return {
        'physical_scale_timeseries': timeseries_csv,
        'physical_scale_monthly_summary': monthly_csv,
        'physical_scale_diagnostic': plot_path,
    }


@torch.no_grad()
def export_heldout_state_forecast(
    model,
    lake,
    output_dir,
    *,
    task_mode='analysis',
    init_mode='profile',
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_start_date=None,
    rollout_mode='free',
    rollout_reinit_scope='train',
    export_max_depth_m=None,
    hard_density_stability=False,
    hard_density_stability_mode=None,
):
    """Export a full-year profile reconstruction rollout for one held-out lake.

    This intentionally avoids rolling post-processing.  In analysis mode,
    rollout_mode='profile_reinit' can reset the water-column state at observed
    profile dates for reconstruction diagnostics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_mode = normalize_task_mode(task_mode)
    rollout_mode = str(rollout_mode or 'free').strip().lower()
    rollout_reinit_scope = str(rollout_reinit_scope or 'train').strip().lower()
    hard_density_stability_label = (
        str(hard_density_stability_mode)
        if hard_density_stability_mode is not None
        else ('on' if hard_density_stability else 'off')
    )
    if rollout_mode not in {'free', 'profile_reinit'}:
        raise ValueError("rollout_mode must be 'free' or 'profile_reinit'.")
    if rollout_reinit_scope not in {'train', 'all'}:
        raise ValueError("rollout_reinit_scope must be 'train' or 'all'.")
    if task_mode != 'analysis' and rollout_mode == 'profile_reinit' and rollout_reinit_scope == 'all':
        raise ValueError("profile_reinit with rollout_reinit_scope='all' is only allowed in analysis mode.")

    df = lake['df']
    depths = lake['depths_np']
    all_lookup = lake['lookups']['all']
    reinit_lookup = lake['lookups']['train'] if rollout_reinit_scope == 'train' else all_lookup
    device = lake['depths'].device
    init_state = initialize_rollout_state(
        model=model,
        df=df,
        depths=depths,
        all_lookup=all_lookup,
        forcing_rows=lake['forcing_rows'],
        static_features=lake['static_features'],
        metadata=lake['metadata'],
        device=device,
        init_mode=init_mode,
        rollout_start_date=rollout_start_date,
        spinup_days=spinup_days,
        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
        task_mode=task_mode,
        area_profile=lake['area'],
        hard_density_stability=hard_density_stability,
    )
    start_idx = int(init_state['start_idx'])
    rollout_start_idx = int(init_state['rollout_start_idx'])
    initial_profile = np.asarray(init_state['initial_profile'], dtype=np.float32)
    temp_grid = np.full((len(depths), len(df)), np.nan, dtype=np.float32)
    current = init_state['current']
    freezing_storage = init_state.get('freezing_storage_j_m2', torch.zeros_like(current))
    for idx, profile in init_state['profiles_by_index'].items():
        temp_grid[:, int(idx)] = np.asarray(profile, dtype=np.float32)
    diagnostic_records = list(init_state['diagnostics'])

    def _diag_scalar(diagnostics, key, default=0.0):
        value = diagnostics.get(key)
        if value is None:
            return float(default)
        return float(value.detach().cpu().reshape(-1)[0])

    model.eval()
    for day_idx in range(rollout_start_idx, len(df) - 1):
        current_date = pd.Timestamp(df['Date'].iloc[day_idx]).normalize()
        state_was_reinitialized = False
        if rollout_mode == 'profile_reinit' and current_date in reinit_lookup:
            current = torch.tensor(
                reinit_lookup[current_date],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            freezing_storage = torch.zeros_like(current)
            temp_grid[:, day_idx] = current.detach().cpu().numpy().reshape(-1)
            state_was_reinitialized = True
        next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
        current, freezing_storage, diagnostics = model.step(
            current,
            lake['forcing_rows'][day_idx],
            lake['static_features'],
            next_forcing_row=next_row,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            return_diagnostics=True,
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        temp_grid[:, day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
        diagnostic_records.append({
            'Date': pd.Timestamp(df['Date'].iloc[day_idx + 1]).date().isoformat(),
            'spinup_phase': False,
            'init_mode': init_state['init_mode'],
            'requested_init_mode': init_state['requested_init_mode'],
            'spinup_days_used': init_state['spinup_days_used'],
            'rollout_mode': rollout_mode,
            'rollout_reinit_scope': rollout_reinit_scope,
            'state_was_reinitialized': state_was_reinitialized,
            'turbulent_flux_mode': getattr(model, 'turbulent_flux_mode', 'provided'),
            'turbulent_flux_blend_alpha': float(getattr(model, 'turbulent_flux_blend_alpha', 1.0)),
            'freezing_energy_mode': getattr(model, 'freezing_energy_mode', 'clamp'),
            'surface_flux_wm2': _diag_scalar(diagnostics, 'surface_flux_wm2'),
            'open_water_surface_flux_wm2': _diag_scalar(diagnostics, 'open_water_surface_flux_wm2'),
            'open_water_net_radiation_wm2': _diag_scalar(diagnostics, 'open_water_net_radiation_wm2'),
            'open_water_sensible_heat_wm2': _diag_scalar(diagnostics, 'open_water_sensible_heat_wm2'),
            'open_water_latent_heat_wm2': _diag_scalar(diagnostics, 'open_water_latent_heat_wm2'),
            'open_water_sensible_heat_bulk_wm2': _diag_scalar(diagnostics, 'open_water_sensible_heat_bulk_wm2'),
            'open_water_latent_heat_bulk_wm2': _diag_scalar(diagnostics, 'open_water_latent_heat_bulk_wm2'),
            'open_water_sensible_heat_bulk_unscaled_wm2': _diag_scalar(
                diagnostics,
                'open_water_sensible_heat_bulk_unscaled_wm2',
            ),
            'open_water_latent_heat_bulk_unscaled_wm2': _diag_scalar(
                diagnostics,
                'open_water_latent_heat_bulk_unscaled_wm2',
            ),
            'heat_input_wm2': _diag_scalar(diagnostics, 'heat_input_wm2'),
            'heat_tendency_wm2': _diag_scalar(diagnostics, 'heat_tendency_wm2'),
            'sensible_heat_tendency_wm2': _diag_scalar(diagnostics, 'sensible_heat_tendency_wm2'),
            'effective_heat_tendency_wm2': _diag_scalar(diagnostics, 'effective_heat_tendency_wm2'),
            'freezing_storage_j_m2': _diag_scalar(diagnostics, 'freezing_storage_j_m2'),
            'freezing_storage_change_wm2': _diag_scalar(diagnostics, 'freezing_storage_change_wm2'),
            'energy_residual_wm2': float((diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']).detach().cpu().reshape(-1)[0]),
            'temperature_floor_heat_injection_wm2': _diag_scalar(diagnostics, 'temperature_floor_heat_injection_wm2'),
            'temperature_floor_heat_injection_j_m2': _diag_scalar(diagnostics, 'temperature_floor_heat_injection_j_m2'),
            'temperature_ceiling_heat_removal_wm2': _diag_scalar(diagnostics, 'temperature_ceiling_heat_removal_wm2'),
            'kz_mean': _diag_scalar(diagnostics, 'kz_mean'),
            'kz_surface_mean': _diag_scalar(diagnostics, 'kz_surface_mean'),
            'kz_mid_mean': _diag_scalar(diagnostics, 'kz_mid_mean'),
            'kz_deep_mean': _diag_scalar(diagnostics, 'kz_deep_mean'),
            'shortwave_absorption_scale': _diag_scalar(diagnostics, 'shortwave_absorption_scale'),
            'surface_flux_bias_wm2': _diag_scalar(diagnostics, 'surface_flux_bias_wm2'),
            'shortwave_to_water_wm2': _diag_scalar(diagnostics, 'shortwave_to_water_wm2'),
            'ice_shortwave_transmission': _diag_scalar(diagnostics, 'ice_shortwave_transmission', 1.0),
            'adaptive_wind_kz_scale': _diag_scalar(diagnostics, 'adaptive_wind_kz_scale', 1.0),
            'adaptive_turbulent_flux_blend_alpha': _diag_scalar(
                diagnostics,
                'adaptive_turbulent_flux_blend_alpha',
                getattr(model, 'turbulent_flux_blend_alpha', 0.3),
            ),
            'adaptive_kd_multiplier': _diag_scalar(diagnostics, 'adaptive_kd_multiplier', 1.0),
            'adaptive_turbulent_exchange_scale': _diag_scalar(
                diagnostics,
                'adaptive_turbulent_exchange_scale',
                1.0,
            ),
            'adaptive_convective_mixing_scale': _diag_scalar(
                diagnostics,
                'adaptive_convective_mixing_scale',
                1.0,
            ),
            'adaptive_ice_shortwave_scale': _diag_scalar(diagnostics, 'adaptive_ice_shortwave_scale', 1.0),
            'adaptive_parameter_regularization_loss': _diag_scalar(
                diagnostics,
                'adaptive_parameter_regularization_loss',
            ),
            'lake_shape_wind_factor': _diag_scalar(diagnostics, 'lake_shape_wind_factor', 1.0),
            'lake_shape_decay_depth_m': _diag_scalar(diagnostics, 'lake_shape_decay_depth_m', 5.0),
            'stratification_mixing_gate_mean': _diag_scalar(
                diagnostics,
                'stratification_mixing_gate_mean',
                1.0,
            ),
            'stratification_mixing_gate_min': _diag_scalar(
                diagnostics,
                'stratification_mixing_gate_min',
                1.0,
            ),
            'stratification_mixing_gate_deep_mean': _diag_scalar(
                diagnostics,
                'stratification_mixing_gate_deep_mean',
                1.0,
            ),
            'surface_cooling_scale_raw': _diag_scalar(diagnostics, 'surface_cooling_scale_raw', _diag_scalar(diagnostics, 'surface_cooling_scale')),
            'surface_cooling_scale_effective': _diag_scalar(diagnostics, 'surface_cooling_scale'),
            'residual_abs_mean_c': _diag_scalar(diagnostics, 'residual_abs_mean_c'),
            'density_adjustment_applied': _diag_scalar(diagnostics, 'density_adjustment_applied'),
            'density_adjustment_max_delta_c': _diag_scalar(diagnostics, 'density_adjustment_max_delta_c'),
            'density_adjustment_heat_delta_j_m2': _diag_scalar(diagnostics, 'density_adjustment_heat_delta_j_m2'),
            'surface_temp_c': float(current[:, 0].detach().cpu().reshape(-1)[0]),
            'mean_temp_c': float(current.mean().detach().cpu()),
        })

    if start_idx > 0:
        temp_grid[:, :start_idx] = np.asarray(initial_profile, dtype=np.float32).reshape(-1, 1)

    metadata = dict(lake['metadata'])
    metadata['file_tag'] = str(lake['lake_id'])
    metadata.setdefault('lake_name', str(lake['lake_id']).replace('_', ' ').title())
    suffix = (
        'heldout_state_reconstruction_profile_reinit'
        if rollout_mode == 'profile_reinit'
        else 'heldout_state_reconstruction'
    )
    export_temp_grid, export_depths, export_depth_info = _depth_limited_export_grid(
        temp_grid,
        depths,
        export_max_depth_m=export_max_depth_m,
    )
    prediction_csv = export_temperature_tables(
        df,
        export_temp_grid,
        export_depths,
        output_dir,
        metadata,
        suffix=suffix,
    )
    diagnostics_csv = output_dir / f"{metadata['file_tag']}_{suffix}_initialization_diagnostics.csv"
    diagnostics_df = pd.DataFrame(diagnostic_records)
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    scale_diagnostics = _write_physical_scale_diagnostics(diagnostics_df, metadata, output_dir, suffix)
    heat_closure_diagnostics = write_heat_closure_summaries(diagnostics_df, metadata, output_dir, suffix)
    density_diagnostics = write_density_stability_summary(
        temp_grid,
        depths,
        df['Date'],
        metadata,
        output_dir,
        suffix,
    )
    init_summary = output_dir / f"{metadata['file_tag']}_{suffix}_initialization_summary.json"
    init_summary.write_text(
        json.dumps(
            {
                'init_mode': init_state['init_mode'],
                'requested_init_mode': init_state['requested_init_mode'],
                'spinup_days_used': int(init_state['spinup_days_used']),
                'rollout_start_date': pd.Timestamp(df['Date'].iloc[rollout_start_idx]).date().isoformat(),
                'start_date': pd.Timestamp(df['Date'].iloc[start_idx]).date().isoformat(),
                'rollout_mode': rollout_mode,
                'rollout_reinit_scope': rollout_reinit_scope,
                'hard_density_stability': hard_density_stability_label,
                'hard_density_stability_active': bool(hard_density_stability),
                'turbulent_flux_mode': getattr(model, 'turbulent_flux_mode', 'provided'),
                'turbulent_flux_blend_alpha': float(getattr(model, 'turbulent_flux_blend_alpha', 1.0)),
                'freezing_energy_mode': getattr(model, 'freezing_energy_mode', 'clamp'),
                **export_depth_info,
                'prior_info': init_state.get('prior_info', {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    heatmap_path = output_dir / f"{metadata['file_tag']}_{suffix}_year_heatmap.png"
    plot_year_heatmap(df, export_temp_grid, export_depths, heatmap_path, metadata)

    profile_reinit_label = ' profile reinit' if rollout_mode == 'profile_reinit' else ''
    model_label = (
        f"{metadata.get('lake_name', lake['lake_id'])} "
        f"reconstruction-state{profile_reinit_label} profile reconstruction"
    )
    diagnostic_figures, diagnostic_status = generate_prediction_diagnostic_figures(
        truth_csv_path=lake.get('profile_obs_path'),
        prediction_csv_path=prediction_csv,
        output_dir=output_dir,
        lake_name=metadata.get('lake_name', lake['lake_id']),
        model_label=model_label,
        file_prefix=f"{metadata['file_tag']}_{suffix}",
    )
    scorecard_report, scorecard_status = run_scorecard_report(
        truth_csv_path=lake.get('profile_obs_path'),
        prediction_csv_path=prediction_csv,
        output_dir=output_dir,
        label=model_label,
        report_name=f"{metadata['file_tag']}_{suffix}_scorecard_report.png",
    )
    return {
        'lake_id': lake['lake_id'],
        'prediction_csv': prediction_csv,
        'diagnostics_csv': diagnostics_csv,
        'init_summary': init_summary,
        'heatmap_path': heatmap_path,
        'diagnostic_figures': diagnostic_figures,
        'diagnostic_status': diagnostic_status,
        'scorecard_report': scorecard_report,
        'scorecard_status': scorecard_status,
        **export_depth_info,
        **scale_diagnostics,
        **heat_closure_diagnostics,
        **density_diagnostics,
    }


def train_multilake_state_forecaster(
    manifest_path,
    output_dir,
    *,
    epochs=100,
    lr=3.0e-4,
    depth_points=40,
    max_rollout_days=45,
    history_window_days=30,
    split_mode='time_blocked',
    profile_supervision_scope='train',
    test_lake_id=None,
    test_lake_ids=None,
    heldout_lake_groups=None,
    residual_limit_c=0.50,
    wind_kz_scale=1.0,
    autumn_convective_boost=1.0,
    profile_huber_delta=2.0,
    lst_surface_weight=0.03,
    lst_feature_dropout_probability=0.20,
    energy_balance_weight=0.001,
    residual_regularization_weight=0.02,
    daily_tendency_weight=0.02,
    physical_scale_regularization_weight=0.01,
    physical_scale_smoothness_weight=0.005,
    kd_prior_regularization_weight=0.001,
    adaptive_parameter_regularization_weight=0.01,
    heat_content_transition_weight=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
    heat_content_full_column_min_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    heat_content_transition_season_factors=None,
    heat_content_transition_season_mode='auto',
    heat_content_transition_depth_factor='on',
    heat_content_transition_effective_max=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
    transition_loss_weight=1.0,
    segment_rollout_loss_weight=0.05,
    segment_rollout_start_epoch=30,
    segment_rollout_ramp_epochs=30,
    segment_rollout_max_days=None,
    segment_rollout_samples_per_lake=12,
    segment_rollout_lst_surface_weight=0.01,
    warm_season_column_heat_content_weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    warm_season_column_heat_content_quantile_low=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
    warm_season_column_heat_content_quantile_high=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    warm_season_column_heat_content_min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    teacher_forcing_start=0.7,
    teacher_forcing_end=0.0,
    state_noise_weight=1.0,
    residual_time_smooth_weight=0.01,
    rolling_horizon_eval_max_starts=40,
    init_mode='profile',
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_start_date=None,
    rollout_mode='free',
    rollout_reinit_scope='train',
    checkpoint_path=None,
    resume_checkpoint=None,
    checkpoint_every_epochs=5,
    eval_every_epochs=None,
    full_eval_every_epochs=None,
    profile_runtime=True,
    profile_gpu=False,
    history_diagnostic_every_epochs=0,
    torch_tf32='on',
    torch_matmul_precision='high',
    transition_batch_size=0,
    segment_rollout_batch_size=0,
    rolling_horizon_batch_size=32,
    train_diagnostic_mode='loss',
    export_after_training='off',
    export_max_depth_m=None,
    cross_lake_batch_mode='off',
    cross_lake_batch_size=0,
    export_only=False,
    hard_density_stability='auto',
    turbulent_flux_mode='bulk',
    turbulent_flux_blend_alpha=0.3,
    freezing_energy_mode='latent_reservoir',
    advective_heat_source_mode='reservoir_simple',
    shape_aware_mixing='on',
    shape_mixing_strength=0.35,
    stratification_mixing_cap='on',
    stratification_mixing_cap_strength=1.0,
    lake_adaptive_params='off',
    lake_adaptive_hidden_dim=64,
    lake_adaptive_init_spread=0.02,
    lake_adaptive_temporal_mode='off',
    lake_adaptive_temporal_init_spread=0.005,
    lake_adaptive_temporal_scale=0.25,
    adaptive_wind_kz_min=0.4,
    adaptive_wind_kz_max=3.0,
    adaptive_blend_alpha_min=0.0,
    adaptive_blend_alpha_max=0.6,
    adaptive_kd_multiplier_min=0.4,
    adaptive_kd_multiplier_max=2.0,
    adaptive_turbulent_exchange_scale_min=0.5,
    adaptive_turbulent_exchange_scale_max=1.8,
    adaptive_convective_mixing_scale_min=0.3,
    adaptive_convective_mixing_scale_max=2.5,
    adaptive_ice_shortwave_scale_min=0.4,
    adaptive_ice_shortwave_scale_max=1.8,
    device=None,
):
    manifest = _read_manifest(manifest_path)
    if 'forecast_start_date' in manifest:
        raise ValueError("Manifest field forecast_start_date was removed; use rollout_start_date.")
    if 'diagnostic_mode' in manifest:
        raise ValueError("Manifest field diagnostic_mode was removed; use train_diagnostic_mode.")
    if 'light_eval_every_epochs' in manifest:
        raise ValueError("Manifest field light_eval_every_epochs was removed; use eval_every_epochs.")
    removed_fixed_mode_fields = {'task_mode', 'data_fill_mode'}
    present_removed_fixed_mode_fields = _removed_manifest_fields(manifest, removed_fixed_mode_fields)
    if present_removed_fixed_mode_fields:
        raise ValueError(
            "Manifest task/data mode fields were removed "
            f"({', '.join(present_removed_fixed_mode_fields)}); "
            "multi-lake training is fixed to analysis/reconstruction."
        )
    removed_segment_rollout_fields = {
        'long_free_roll_loss_weight',
        'long_free_roll_start_epoch',
        'long_free_roll_ramp_epochs',
        'long_free_roll_max_days',
        'long_free_roll_samples_per_lake',
    }
    present_removed_segment_rollout_fields = _removed_manifest_fields(manifest, removed_segment_rollout_fields)
    if present_removed_segment_rollout_fields:
        raise ValueError(
            "Manifest long_free_roll fields were removed "
            f"({', '.join(present_removed_segment_rollout_fields)}); use segment_rollout_* fields."
        )
    removed_free_roll_fields = {
        'free_roll_loss_weight',
        'free_roll_horizons',
        'free_roll_supervision_mode',
    }
    present_removed_free_roll_fields = _removed_manifest_fields(manifest, removed_free_roll_fields)
    if present_removed_free_roll_fields:
        raise ValueError(
            "Manifest short free-roll loss fields were removed "
            f"({', '.join(present_removed_free_roll_fields)}); use "
            "segment_rollout_loss_weight, segment_rollout_max_days, and "
            "segment_rollout_samples_per_lake."
        )
    removed_batch_mode_fields = {
        'transition_batch_mode',
        'segment_rollout_batch_mode',
        'rolling_horizon_batch_mode',
        'rollout_batch_step_mode',
        'step_forcing_mode',
    }
    present_removed_batch_mode_fields = _removed_manifest_fields(manifest, removed_batch_mode_fields)
    if present_removed_batch_mode_fields:
        raise ValueError(
            "Manifest batch/debug mode fields were removed "
            f"({', '.join(present_removed_batch_mode_fields)}); batch paths are fixed internally."
        )
    task_mode = 'analysis'
    data_fill_mode = 'reconstruction'
    profile_supervision_scope = _normalize_profile_supervision_scope(manifest.get(
        'profile_supervision_scope',
        profile_supervision_scope,
    ))
    supervision_pair_key = profile_supervision_scope
    supervision_sequence_key = profile_supervision_scope
    manifest_max_rollout_days = int(manifest.get('max_rollout_days', max_rollout_days))
    segment_rollout_loss_weight = float(manifest.get(
        'segment_rollout_loss_weight',
        segment_rollout_loss_weight,
    ))
    if segment_rollout_loss_weight <= 0.0:
        raise ValueError(
            "segment_rollout_loss_weight must be > 0.0 for the eighth-version "
            "segment rollout mainline; use 0.05 by default or 0.10 for stronger "
            "long-horizon supervision."
        )
    if segment_rollout_max_days is None and 'segment_rollout_max_days' in manifest:
        segment_rollout_max_days = manifest.get('segment_rollout_max_days')
    segment_rollout_max_days = int(segment_rollout_max_days or manifest_max_rollout_days)
    segment_rollout_start_epoch = int(manifest.get('segment_rollout_start_epoch', segment_rollout_start_epoch))
    segment_rollout_ramp_epochs = int(manifest.get('segment_rollout_ramp_epochs', segment_rollout_ramp_epochs))
    segment_rollout_samples_per_lake = int(manifest.get(
        'segment_rollout_samples_per_lake',
        segment_rollout_samples_per_lake,
    ))
    segment_rollout_lst_surface_weight = float(manifest.get(
        'segment_rollout_lst_surface_weight',
        segment_rollout_lst_surface_weight,
    ))
    if segment_rollout_lst_surface_weight < 0.0:
        raise ValueError('segment_rollout_lst_surface_weight must be non-negative.')
    warm_season_column_heat_content_weight = float(manifest.get(
        'warm_season_column_heat_content_weight',
        warm_season_column_heat_content_weight,
    ))
    if warm_season_column_heat_content_weight < 0.0:
        raise ValueError('warm_season_column_heat_content_weight must be non-negative.')
    warm_season_column_heat_content_quantile_low = float(manifest.get(
        'warm_season_column_heat_content_quantile_low',
        warm_season_column_heat_content_quantile_low,
    ))
    warm_season_column_heat_content_quantile_high = float(manifest.get(
        'warm_season_column_heat_content_quantile_high',
        warm_season_column_heat_content_quantile_high,
    ))
    if not (
        0.0 <= warm_season_column_heat_content_quantile_low
        < warm_season_column_heat_content_quantile_high
        <= 1.0
    ):
        raise ValueError(
            'warm_season_column_heat_content_quantile_low/high must satisfy 0 <= low < high <= 1.'
        )
    warm_season_column_heat_content_min_gap_days = int(manifest.get(
        'warm_season_column_heat_content_min_gap_days',
        warm_season_column_heat_content_min_gap_days,
    ))
    if warm_season_column_heat_content_min_gap_days < 1:
        raise ValueError('warm_season_column_heat_content_min_gap_days must be positive.')
    lst_feature_dropout_probability = float(manifest.get(
        'lst_feature_dropout_probability',
        lst_feature_dropout_probability,
    ))
    if not (0.0 <= lst_feature_dropout_probability <= 1.0):
        raise ValueError('lst_feature_dropout_probability must be between 0.0 and 1.0.')
    teacher_forcing_start = float(manifest.get('teacher_forcing_start', teacher_forcing_start))
    teacher_forcing_end = float(manifest.get('teacher_forcing_end', teacher_forcing_end))
    state_noise_weight = float(manifest.get('state_noise_weight', state_noise_weight))
    residual_time_smooth_weight = float(manifest.get(
        'residual_time_smooth_weight',
        residual_time_smooth_weight,
    ))
    rolling_horizon_eval_max_starts = int(manifest.get(
        'rolling_horizon_eval_max_starts',
        rolling_horizon_eval_max_starts,
    ))
    wind_kz_scale = float(manifest.get('wind_kz_scale', wind_kz_scale))
    autumn_convective_boost = float(manifest.get('autumn_convective_boost', autumn_convective_boost))
    shape_aware_mixing = normalize_shape_aware_mixing(manifest.get(
        'shape_aware_mixing',
        shape_aware_mixing,
    ))
    shape_mixing_strength = float(manifest.get('shape_mixing_strength', shape_mixing_strength))
    if shape_mixing_strength < 0.0:
        raise ValueError('shape_mixing_strength must be non-negative.')
    stratification_mixing_cap = normalize_shape_aware_mixing(manifest.get(
        'stratification_mixing_cap',
        stratification_mixing_cap,
    ))
    stratification_mixing_cap_strength = float(manifest.get(
        'stratification_mixing_cap_strength',
        stratification_mixing_cap_strength,
    ))
    if stratification_mixing_cap_strength < 0.0:
        raise ValueError('stratification_mixing_cap_strength must be non-negative.')
    physical_scale_regularization_weight = float(manifest.get(
        'physical_scale_regularization_weight',
        physical_scale_regularization_weight,
    ))
    physical_scale_smoothness_weight = float(manifest.get(
        'physical_scale_smoothness_weight',
        physical_scale_smoothness_weight,
    ))
    kd_prior_regularization_weight = float(manifest.get(
        'kd_prior_regularization_weight',
        kd_prior_regularization_weight,
    ))
    if kd_prior_regularization_weight < 0.0:
        raise ValueError('kd_prior_regularization_weight must be non-negative.')
    adaptive_parameter_regularization_weight = float(manifest.get(
        'adaptive_parameter_regularization_weight',
        adaptive_parameter_regularization_weight,
    ))
    if adaptive_parameter_regularization_weight < 0.0:
        raise ValueError('adaptive_parameter_regularization_weight must be non-negative.')
    heat_content_transition_weight = float(manifest.get(
        'heat_content_transition_weight',
        heat_content_transition_weight,
    ))
    if heat_content_transition_weight < 0.0:
        raise ValueError('heat_content_transition_weight must be non-negative.')
    heat_content_full_column_min_coverage = float(manifest.get(
        'heat_content_full_column_min_coverage',
        heat_content_full_column_min_coverage,
    ))
    heat_content_transition_season_factors_override = manifest.get(
        'heat_content_transition_season_factors',
        heat_content_transition_season_factors,
    )
    heat_content_transition_season_mode = _normalize_heat_content_transition_season_mode(
        manifest.get(
            'heat_content_transition_season_mode',
            heat_content_transition_season_mode,
        )
    )
    heat_content_transition_depth_factor = _normalize_heat_content_transition_depth_factor(
        manifest.get(
            'heat_content_transition_depth_factor',
            heat_content_transition_depth_factor,
        )
    )
    heat_content_transition_effective_max = float(manifest.get(
        'heat_content_transition_effective_max',
        heat_content_transition_effective_max,
    ))
    if (not np.isfinite(heat_content_transition_effective_max)) or heat_content_transition_effective_max < 0.0:
        raise ValueError('heat_content_transition_effective_max must be non-negative.')
    transition_loss_weight = float(manifest.get('transition_loss_weight', transition_loss_weight))
    if transition_loss_weight < 0.0:
        raise ValueError('transition_loss_weight must be non-negative.')
    init_mode = str(manifest.get('init_mode', init_mode) or 'profile').strip().lower()
    spinup_days = int(manifest.get('spinup_days', spinup_days))
    spinup_lst_assimilation_strength = float(manifest.get(
        'spinup_lst_assimilation_strength',
        spinup_lst_assimilation_strength,
    ))
    spinup_lst_assimilation_decay_depth_m = float(manifest.get(
        'spinup_lst_assimilation_decay_depth_m',
        spinup_lst_assimilation_decay_depth_m,
    ))
    spinup_lst_assimilation_max_increment_c = float(manifest.get(
        'spinup_lst_assimilation_max_increment_c',
        spinup_lst_assimilation_max_increment_c,
    ))
    rollout_start_date = manifest.get('rollout_start_date', rollout_start_date)
    hard_density_stability_mode = str(manifest.get(
        'hard_density_stability',
        hard_density_stability,
    ) or 'auto').strip().lower()
    hard_density_stability_active = resolve_hard_density_stability(
        hard_density_stability_mode,
        task_mode=task_mode,
        data_fill_mode=data_fill_mode,
    )
    turbulent_flux_mode = normalize_turbulent_flux_mode(manifest.get(
        'turbulent_flux_mode',
        turbulent_flux_mode,
    ))
    turbulent_flux_blend_alpha = float(manifest.get(
        'turbulent_flux_blend_alpha',
        turbulent_flux_blend_alpha,
    ))
    if (not np.isfinite(turbulent_flux_blend_alpha)) or turbulent_flux_blend_alpha < 0.0 or turbulent_flux_blend_alpha > 1.0:
        raise ValueError('turbulent_flux_blend_alpha must be between 0 and 1.')
    lake_adaptive_params = normalize_lake_adaptive_params(manifest.get(
        'lake_adaptive_params',
        lake_adaptive_params,
    ))
    lake_adaptive_hidden_dim = int(manifest.get('lake_adaptive_hidden_dim', lake_adaptive_hidden_dim))
    lake_adaptive_init_spread = float(manifest.get('lake_adaptive_init_spread', lake_adaptive_init_spread))
    lake_adaptive_temporal_mode = normalize_lake_adaptive_temporal_mode(manifest.get(
        'lake_adaptive_temporal_mode',
        lake_adaptive_temporal_mode,
    ))
    lake_adaptive_temporal_init_spread = float(manifest.get(
        'lake_adaptive_temporal_init_spread',
        lake_adaptive_temporal_init_spread,
    ))
    lake_adaptive_temporal_scale = float(manifest.get(
        'lake_adaptive_temporal_scale',
        lake_adaptive_temporal_scale,
    ))
    if lake_adaptive_hidden_dim <= 0:
        raise ValueError('lake_adaptive_hidden_dim must be positive.')
    if lake_adaptive_init_spread < 0.0:
        raise ValueError('lake_adaptive_init_spread must be non-negative.')
    if lake_adaptive_temporal_init_spread < 0.0:
        raise ValueError('lake_adaptive_temporal_init_spread must be non-negative.')
    if lake_adaptive_temporal_scale < 0.0:
        raise ValueError('lake_adaptive_temporal_scale must be non-negative.')
    adaptive_wind_kz_min = float(manifest.get('adaptive_wind_kz_min', adaptive_wind_kz_min))
    adaptive_wind_kz_max = float(manifest.get('adaptive_wind_kz_max', adaptive_wind_kz_max))
    adaptive_blend_alpha_min = float(manifest.get('adaptive_blend_alpha_min', adaptive_blend_alpha_min))
    adaptive_blend_alpha_max = float(manifest.get('adaptive_blend_alpha_max', adaptive_blend_alpha_max))
    adaptive_kd_multiplier_min = float(manifest.get('adaptive_kd_multiplier_min', adaptive_kd_multiplier_min))
    adaptive_kd_multiplier_max = float(manifest.get('adaptive_kd_multiplier_max', adaptive_kd_multiplier_max))
    adaptive_turbulent_exchange_scale_min = float(manifest.get(
        'adaptive_turbulent_exchange_scale_min',
        adaptive_turbulent_exchange_scale_min,
    ))
    adaptive_turbulent_exchange_scale_max = float(manifest.get(
        'adaptive_turbulent_exchange_scale_max',
        adaptive_turbulent_exchange_scale_max,
    ))
    adaptive_convective_mixing_scale_min = float(manifest.get(
        'adaptive_convective_mixing_scale_min',
        adaptive_convective_mixing_scale_min,
    ))
    adaptive_convective_mixing_scale_max = float(manifest.get(
        'adaptive_convective_mixing_scale_max',
        adaptive_convective_mixing_scale_max,
    ))
    adaptive_ice_shortwave_scale_min = float(manifest.get(
        'adaptive_ice_shortwave_scale_min',
        adaptive_ice_shortwave_scale_min,
    ))
    adaptive_ice_shortwave_scale_max = float(manifest.get(
        'adaptive_ice_shortwave_scale_max',
        adaptive_ice_shortwave_scale_max,
    ))
    adaptive_bounds = {
        'adaptive_wind_kz': (adaptive_wind_kz_min, adaptive_wind_kz_max),
        'adaptive_blend_alpha': (adaptive_blend_alpha_min, adaptive_blend_alpha_max),
        'adaptive_kd_multiplier': (adaptive_kd_multiplier_min, adaptive_kd_multiplier_max),
        'adaptive_turbulent_exchange_scale': (
            adaptive_turbulent_exchange_scale_min,
            adaptive_turbulent_exchange_scale_max,
        ),
        'adaptive_convective_mixing_scale': (
            adaptive_convective_mixing_scale_min,
            adaptive_convective_mixing_scale_max,
        ),
        'adaptive_ice_shortwave_scale': (adaptive_ice_shortwave_scale_min, adaptive_ice_shortwave_scale_max),
    }
    for name, (lower, upper) in adaptive_bounds.items():
        if upper <= lower:
            raise ValueError(f'{name}_max must be greater than {name}_min.')
    adaptive_modes = lake_adaptive_param_set(lake_adaptive_params)
    if 'flux' in adaptive_modes and turbulent_flux_mode != 'blend':
        raise ValueError("lake_adaptive_params including flux requires turbulent_flux_mode='blend'.")
    if 'exchange' in adaptive_modes and turbulent_flux_mode == 'provided':
        raise ValueError("lake_adaptive_params including exchange requires turbulent_flux_mode='bulk' or 'blend'.")
    if lake_adaptive_temporal_mode != 'off' and not adaptive_modes:
        raise ValueError("lake_adaptive_temporal_mode requires lake_adaptive_params other than 'off'.")
    freezing_energy_mode = normalize_freezing_energy_mode(manifest.get(
        'freezing_energy_mode',
        freezing_energy_mode,
    ))
    advective_heat_source_mode = normalize_advective_heat_source_mode(manifest.get(
        'advective_heat_source_mode',
        advective_heat_source_mode,
    ))
    checkpoint_every_epochs = int(manifest.get('checkpoint_every_epochs', checkpoint_every_epochs or 0))
    if checkpoint_every_epochs < 0:
        raise ValueError('checkpoint_every_epochs must be non-negative.')
    if eval_every_epochs is None:
        eval_every_epochs = manifest.get('eval_every_epochs', None)
    eval_every_epochs = None if eval_every_epochs is None else int(eval_every_epochs)
    if eval_every_epochs is not None and eval_every_epochs <= 0:
        raise ValueError('eval_every_epochs must be positive when provided.')
    default_eval_interval = 50
    eval_interval = int(eval_every_epochs or default_eval_interval)
    if full_eval_every_epochs is None:
        full_eval_every_epochs = manifest.get('full_eval_every_epochs', 60)
    full_eval_every_epochs = None if full_eval_every_epochs is None else int(full_eval_every_epochs)
    if full_eval_every_epochs is not None and full_eval_every_epochs < 0:
        raise ValueError('full_eval_every_epochs must be non-negative when provided; use 0 to disable full eval.')
    full_eval_interval = int(0 if full_eval_every_epochs is None else full_eval_every_epochs)
    profile_runtime = _normalize_bool_flag(
        manifest.get('profile_runtime', profile_runtime),
        name='profile_runtime',
        default=profile_runtime,
    )
    profile_gpu = _normalize_bool_flag(
        manifest.get('profile_gpu', profile_gpu),
        name='profile_gpu',
        default=profile_gpu,
    )
    history_diagnostic_every_epochs = int(manifest.get(
        'history_diagnostic_every_epochs',
        history_diagnostic_every_epochs,
    ) or 0)
    if history_diagnostic_every_epochs < 0:
        raise ValueError('history_diagnostic_every_epochs must be non-negative.')
    torch_tf32 = _normalize_on_off(
        manifest.get('torch_tf32', torch_tf32),
        name='torch_tf32',
    )
    torch_matmul_precision = _normalize_torch_matmul_precision(
        manifest.get('torch_matmul_precision', torch_matmul_precision)
    )
    transition_batch_mode = 'on'
    segment_rollout_batch_mode = 'on'
    rolling_horizon_batch_mode = 'on'
    step_forcing_mode = 'auto'
    train_diagnostic_mode = str(
        manifest.get('train_diagnostic_mode', train_diagnostic_mode) or 'loss'
    ).strip().lower()
    if train_diagnostic_mode not in {'loss', 'full'}:
        raise ValueError('train_diagnostic_mode must be one of: loss, full.')
    rollout_batch_step_mode = 'on'
    export_after_training = _normalize_on_off(
        manifest.get('export_after_training', export_after_training),
        name='export_after_training',
    )
    export_max_depth_m = manifest.get('export_max_depth_m', export_max_depth_m)
    export_max_depth_m = None if export_max_depth_m is None else float(export_max_depth_m)
    if export_max_depth_m is not None and (
        (not np.isfinite(export_max_depth_m)) or export_max_depth_m <= 0.0
    ):
        raise ValueError('export_max_depth_m must be positive when provided.')
    cross_lake_batch_mode = _normalize_on_off(
        manifest.get('cross_lake_batch_mode', cross_lake_batch_mode),
        name='cross_lake_batch_mode',
    )
    cross_lake_batch_size = int(manifest.get('cross_lake_batch_size', cross_lake_batch_size or 0))
    if cross_lake_batch_size < 0:
        raise ValueError('cross_lake_batch_size must be non-negative.')
    transition_batch_size = int(manifest.get('transition_batch_size', transition_batch_size or 0))
    segment_rollout_batch_size = int(manifest.get('segment_rollout_batch_size', segment_rollout_batch_size or 0))
    rolling_horizon_batch_size = int(manifest.get('rolling_horizon_batch_size', rolling_horizon_batch_size or 32))
    if transition_batch_size < 0:
        raise ValueError('transition_batch_size must be non-negative.')
    if segment_rollout_batch_size < 0:
        raise ValueError('segment_rollout_batch_size must be non-negative.')
    if rolling_horizon_batch_size < 0:
        raise ValueError('rolling_horizon_batch_size must be non-negative.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    _apply_torch_runtime_config(
        device=device,
        torch_tf32=torch_tf32,
        torch_matmul_precision=torch_matmul_precision,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lakes = [
        prepare_lake_state_data(
            lake_config,
            split_mode=manifest.get('split_mode', split_mode),
            task_mode=task_mode,
            data_fill_mode=data_fill_mode,
            depth_points=int(manifest.get('depth_points', depth_points)),
            max_rollout_days=manifest_max_rollout_days,
            segment_rollout_max_days=segment_rollout_max_days,
            history_window_days=int(manifest.get('history_window_days', history_window_days)),
            device=device,
        )
        for lake_config in manifest['lakes']
    ]
    for lake in lakes:
        lake['step_forcing_mode'] = step_forcing_mode
        season_config = _resolve_heat_content_transition_season_factors(
            lake.get('metadata', {}),
            mode=heat_content_transition_season_mode,
            override=heat_content_transition_season_factors_override,
        )
        lake['heat_content_transition_season_mode_requested'] = season_config['requested_mode']
        lake['heat_content_transition_season_mode_resolved'] = season_config['resolved_mode']
        lake['heat_content_transition_latitude'] = season_config['latitude']
        lake['heat_content_transition_season_factors'] = season_config['factors']
        _set_warm_column_heat_content_lake_config(
            lake,
            quantile_low=warm_season_column_heat_content_quantile_low,
            quantile_high=warm_season_column_heat_content_quantile_high,
            min_full_column_coverage=heat_content_full_column_min_coverage,
        )
    heldout_selection = _resolve_heldout_selection(
        lakes,
        manifest=manifest,
        test_lake_id=test_lake_id,
        test_lake_ids=test_lake_ids,
        heldout_lake_groups=heldout_lake_groups,
    )
    test_lake_id = heldout_selection['test_lake_id']
    test_lake_ids = heldout_selection['test_lake_ids']
    heldout_lake_groups = heldout_selection['heldout_lake_groups']
    train_lakes = heldout_selection['train_lakes']
    heldout_lakes = heldout_selection['heldout_lakes']
    excluded_lakes = heldout_selection['excluded_lakes']
    heat_content_transition_season_factors_override_payload = (
        _heat_content_transition_season_factors_payload(
            _parse_heat_content_transition_season_factors(heat_content_transition_season_factors_override)
        )
        if _has_heat_content_transition_season_override(heat_content_transition_season_factors_override)
        else None
    )
    heat_content_transition_lake_configs = {
        lake['lake_id']: _heat_content_transition_lake_config_payload(lake)
        for lake in lakes
    }
    warm_column_heat_content_lake_configs = {
        lake['lake_id']: _warm_column_heat_content_lake_config_payload(lake)
        for lake in lakes
    }
    if not train_lakes:
        raise ValueError('No training lakes remain after applying heldout lake selection.')
    if any(not lake['pairs'][supervision_pair_key] for lake in train_lakes):
        missing = [lake['lake_id'] for lake in train_lakes if not lake['pairs'][supervision_pair_key]]
        raise ValueError(
            f'Training lakes need at least one {profile_supervision_scope} transition pair: {missing}'
        )
    for lake in train_lakes:
        lake.setdefault('segment_rollout_epoch_plans', {})[supervision_sequence_key] = (
            _build_segment_rollout_epoch_plan(
                lake['segment_rollout_sequences'][supervision_sequence_key],
                epochs=epochs,
                start_epoch=segment_rollout_start_epoch,
                ramp_epochs=segment_rollout_ramp_epochs,
                max_days=segment_rollout_max_days,
                samples_per_lake=segment_rollout_samples_per_lake,
                segment_rollout_batch_size=segment_rollout_batch_size,
            )
        )
    cross_lake_segment_rollout_epoch_batches = (
        _build_cross_lake_segment_rollout_epoch_batches(
            train_lakes,
            split_key=supervision_sequence_key,
            epochs=epochs,
            segment_rollout_batch_size=segment_rollout_batch_size,
            cross_lake_batch_size=cross_lake_batch_size,
        )
        if cross_lake_batch_mode == 'on' and segment_rollout_batch_mode == 'on'
        else ()
    )

    seed_lake = train_lakes[0]
    static_dim = int(seed_lake['static_features'].reshape(-1).numel())
    if static_dim != STATIC_FEATURE_DIM:
        raise ValueError(
            f"Static feature vector has {static_dim} values; expected {STATIC_FEATURE_DIM}. "
            "Regenerate lake data after static metadata feature changes."
        )
    model = LakeStateForecaster(
        seed_lake['depths_np'],
        seed_lake['area_np'],
        static_dim=static_dim,
        residual_limit_c=residual_limit_c,
        wind_kz_scale=wind_kz_scale,
        autumn_convective_boost=autumn_convective_boost,
        lst_feature_dropout_probability=lst_feature_dropout_probability,
        turbulent_flux_mode=turbulent_flux_mode,
        turbulent_flux_blend_alpha=turbulent_flux_blend_alpha,
        freezing_energy_mode=freezing_energy_mode,
        advective_heat_source_mode=advective_heat_source_mode,
        shape_aware_mixing=shape_aware_mixing,
        shape_mixing_strength=shape_mixing_strength,
        stratification_mixing_cap=stratification_mixing_cap,
        stratification_mixing_cap_strength=stratification_mixing_cap_strength,
        lake_adaptive_params=lake_adaptive_params,
        lake_adaptive_hidden_dim=lake_adaptive_hidden_dim,
        lake_adaptive_init_spread=lake_adaptive_init_spread,
        lake_adaptive_temporal_mode=lake_adaptive_temporal_mode,
        lake_adaptive_temporal_init_spread=lake_adaptive_temporal_init_spread,
        lake_adaptive_temporal_scale=lake_adaptive_temporal_scale,
        adaptive_wind_kz_min=adaptive_wind_kz_min,
        adaptive_wind_kz_max=adaptive_wind_kz_max,
        adaptive_blend_alpha_min=adaptive_blend_alpha_min,
        adaptive_blend_alpha_max=adaptive_blend_alpha_max,
        adaptive_kd_multiplier_min=adaptive_kd_multiplier_min,
        adaptive_kd_multiplier_max=adaptive_kd_multiplier_max,
        adaptive_turbulent_exchange_scale_min=adaptive_turbulent_exchange_scale_min,
        adaptive_turbulent_exchange_scale_max=adaptive_turbulent_exchange_scale_max,
        adaptive_convective_mixing_scale_min=adaptive_convective_mixing_scale_min,
        adaptive_convective_mixing_scale_max=adaptive_convective_mixing_scale_max,
        adaptive_ice_shortwave_scale_min=adaptive_ice_shortwave_scale_min,
        adaptive_ice_shortwave_scale_max=adaptive_ice_shortwave_scale_max,
    ).to(device)
    loaded_checkpoint_path = None
    if checkpoint_path:
        loaded_checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(loaded_checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Depth and area buffers are lake-grid specific.  Keep the current
        # runtime grid buffers and pass each target lake grid explicitly during
        # rollout, so a training-lake checkpoint cannot overwrite them here.
        state_dict = _filter_state_forecaster_state_dict_for_load(model, state_dict)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            message = str(exc)
            if 'size mismatch' in message:
                raise ValueError(
                    f"Checkpoint is not compatible with the current {static_dim}D "
                    "static metadata feature head. Retrain after static lake "
                    "metadata feature changes; archived 11D checkpoints are not "
                    "compatible with the extended metadata mainline."
                ) from exc
            raise
        print(f"Loaded state forecaster checkpoint: {loaded_checkpoint_path}")

    if export_only:
        if loaded_checkpoint_path is None:
            raise ValueError('--export-only requires --checkpoint-path.')
        if not heldout_lakes:
            raise ValueError('--export-only requires --test-lake-id or --test-lake-ids to select held-out lakes.')
        split_summary = output_dir / 'global_state_forecaster_split_summary.json'
        split_summary_payload = {
            lake['lake_id']: {
                'train_pairs': len(lake['pairs']['train']),
                'val_pairs': len(lake['pairs']['val']),
                'all_pairs': len(lake['pairs']['all']),
                'train_segment_rollout_sequences': len(lake['segment_rollout_sequences']['train']),
                'all_segment_rollout_sequences': len(lake['segment_rollout_sequences']['all']),
                'supervision_pairs': len(lake['pairs'][profile_supervision_scope]),
                'supervision_segment_rollout_sequences': len(
                    lake['segment_rollout_sequences'][profile_supervision_scope]
                ),
                'is_heldout_test_lake': bool(lake['lake_id'] in set(test_lake_ids)),
                'is_excluded_by_heldout_group': bool(lake['lake_id'] in {item['lake_id'] for item in excluded_lakes}),
                'heat_content_transition': _heat_content_transition_lake_config_payload(lake),
                'warm_season_column_heat_content': _warm_column_heat_content_lake_config_payload(lake),
                'metadata': _lake_metadata_summary_payload(lake),
            }
            for lake in lakes
        }
        split_summary_payload['_config'] = {
            'test_lake_id': test_lake_id or None,
            'test_lake_ids': list(test_lake_ids),
            'heldout_lake_groups': list(heldout_lake_groups),
            'lake_ids': [lake['lake_id'] for lake in lakes],
            'train_lake_ids': [lake['lake_id'] for lake in train_lakes],
            'heldout_lake_ids': [lake['lake_id'] for lake in heldout_lakes],
            'excluded_lake_ids': [lake['lake_id'] for lake in excluded_lakes],
            'profile_supervision_scope': profile_supervision_scope,
            'residual_limit_c': float(residual_limit_c),
            'wind_kz_scale': float(wind_kz_scale),
            'autumn_convective_boost': float(autumn_convective_boost),
            'physical_scale_mode': 'learned_lake_season_forcing',
            'physical_scale_regularization_weight': float(physical_scale_regularization_weight),
            'physical_scale_smoothness_weight': float(physical_scale_smoothness_weight),
            'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
            'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
            'heat_content_transition_weight': float(heat_content_transition_weight),
            'heat_content_transition_weight_base': float(heat_content_transition_weight),
            'heat_content_full_column_min_coverage': float(heat_content_full_column_min_coverage),
            'heat_content_transition_season_mode': heat_content_transition_season_mode,
            'heat_content_transition_season_factors_override': heat_content_transition_season_factors_override_payload,
            'heat_content_transition_lake_configs': heat_content_transition_lake_configs,
            'heat_content_transition_depth_factor': 'on' if heat_content_transition_depth_factor else 'off',
            'heat_content_transition_effective_max': float(heat_content_transition_effective_max),
            'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
            'warm_season_column_heat_content_quantile_low': float(
                warm_season_column_heat_content_quantile_low
            ),
            'warm_season_column_heat_content_quantile_high': float(
                warm_season_column_heat_content_quantile_high
            ),
            'warm_season_column_heat_content_min_gap_days': int(
                warm_season_column_heat_content_min_gap_days
            ),
            'warm_season_column_heat_content_lake_configs': warm_column_heat_content_lake_configs,
            'transition_loss_weight': float(transition_loss_weight),
            'task_mode': task_mode,
            'data_fill_mode': data_fill_mode,
            'rollout_mode': rollout_mode,
            'rollout_reinit_scope': rollout_reinit_scope,
            'hard_density_stability': hard_density_stability_mode,
            'hard_density_stability_active': bool(hard_density_stability_active),
            'static_feature_dim': int(static_dim),
            'static_feature_keys': list(STATIC_FEATURE_KEYS),
            'turbulent_flux_mode': turbulent_flux_mode,
            'turbulent_flux_blend_alpha': float(turbulent_flux_blend_alpha),
            'lake_adaptive_params': lake_adaptive_params,
            'shape_aware_mixing': shape_aware_mixing,
            'shape_mixing_strength': float(shape_mixing_strength),
            'stratification_mixing_cap': stratification_mixing_cap,
            'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
            'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
            'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
            'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
            'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
            'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
            'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
            'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
            'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
            'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
            'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
            'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
            'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
            'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
            'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
            'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
            'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
            'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
            'freezing_energy_mode': freezing_energy_mode,
            'advective_heat_source_mode': advective_heat_source_mode,
            'checkpoint_every_epochs': int(checkpoint_every_epochs),
            'eval_every_epochs': int(eval_interval),
            'full_eval_every_epochs': int(full_eval_interval),
            'profile_runtime': bool(profile_runtime),
            'profile_gpu': bool(profile_gpu),
            'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
            'torch_tf32': torch_tf32,
            'torch_matmul_precision': torch_matmul_precision,
            'transition_batch_size': int(transition_batch_size),
            'segment_rollout_batch_size': int(segment_rollout_batch_size),
            'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
            'train_diagnostic_mode': train_diagnostic_mode,
            'export_after_training': export_after_training,
            'export_max_depth_m': export_max_depth_m,
            'cross_lake_batch_mode': cross_lake_batch_mode,
            'cross_lake_batch_size': int(cross_lake_batch_size),
            'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
            'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
            'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
            'segment_rollout_max_days': int(segment_rollout_max_days),
            'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
            'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
            'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
            'export_only': True,
            'checkpoint_path': str(loaded_checkpoint_path),
        }
        split_summary.write_text(
            json.dumps(split_summary_payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        lake_adaptive_summary_csv = _write_lake_adaptive_parameter_summary(
            model,
            lakes,
            train_lakes,
            heldout_lakes,
            excluded_lakes,
            output_dir,
        )
        heldout_exports = []
        for lake in heldout_lakes:
            export_info = export_heldout_state_forecast(
                model,
                lake,
                output_dir,
                task_mode=task_mode,
                init_mode=init_mode,
                spinup_days=spinup_days,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_start_date=rollout_start_date,
                rollout_mode=rollout_mode,
                rollout_reinit_scope=rollout_reinit_scope,
                export_max_depth_m=export_max_depth_m,
                hard_density_stability=hard_density_stability_active,
                hard_density_stability_mode=hard_density_stability_mode,
            )
            heldout_exports.append(export_info)
            print(
                f"Held-out reconstruction export for {lake['lake_id']}: "
                f"{export_info['heatmap_path']} | score={export_info['scorecard_status']}"
            )
        return {
            'model': model,
            'checkpoint_path': loaded_checkpoint_path,
            'history_csv': None,
            'runtime_profile_csv': None,
            'split_summary': split_summary,
            'lake_adaptive_parameter_summary_csv': lake_adaptive_summary_csv,
            'heldout_exports': heldout_exports,
            'lakes': lakes,
            'history': [],
            'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
            'export_max_depth_m': export_max_depth_m,
            'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
            'warm_season_column_heat_content_quantile_low': float(
                warm_season_column_heat_content_quantile_low
            ),
            'warm_season_column_heat_content_quantile_high': float(
                warm_season_column_heat_content_quantile_high
            ),
            'warm_season_column_heat_content_min_gap_days': int(
                warm_season_column_heat_content_min_gap_days
            ),
            'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
            'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
            'advective_heat_source_mode': advective_heat_source_mode,
            'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
            'torch_tf32': torch_tf32,
            'torch_matmul_precision': torch_matmul_precision,
        }

    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    start_epoch = 0
    history = []
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        resume = torch.load(resume_path, map_location=device)
        state_dict = resume.get('model_state_dict', resume)
        state_dict = _filter_state_forecaster_state_dict_for_load(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        if 'optimizer_state_dict' in resume:
            optimizer.load_state_dict(resume['optimizer_state_dict'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        history = list(resume.get('training_history', resume.get('history', [])))
        start_epoch = int(resume.get('epoch', -1)) + 1
        print(f"Resumed training checkpoint: {resume_path} at epoch {start_epoch}")

    partial_history_csv = output_dir / 'global_state_forecaster_training_history_partial.csv'
    runtime_profile_csv = output_dir / 'runtime_profile.csv'
    runtime_profile_records = []
    best_val_rolling_checkpoint_path = output_dir / 'best_by_val_rolling.pt'
    best_val_rolling_metrics_path = output_dir / 'best_by_val_rolling_metrics.json'
    best_val_rolling_score = float('inf')
    best_val_rolling_epoch = None
    best_val_rolling_enabled = full_eval_interval > 0

    def _best_val_rolling_candidate(record):
        if not best_val_rolling_enabled:
            return None, 'disabled: full_eval_every_epochs=0'
        rmse30 = float(record.get('val_rolling_start_rmse_30d', np.nan))
        rmse60 = float(record.get('val_rolling_start_rmse_60d', np.nan))
        count30 = float(record.get('val_rolling_start_count_30d', 0.0) or 0.0)
        count60 = float(record.get('val_rolling_start_count_60d', 0.0) or 0.0)
        if not (np.isfinite(rmse30) and np.isfinite(rmse60)):
            return None, 'skipped: val rolling 30d/60d rmse is not finite'
        if count30 <= 0.0 or count60 <= 0.0:
            return None, 'skipped: val rolling 30d/60d count is zero'
        return 0.5 * rmse30 + 0.5 * rmse60, None

    def _write_best_val_rolling_metrics(record, score):
        payload = {
            'epoch': int(record['epoch']),
            'score': float(score),
            'score_formula': '0.5 * val_rolling_start_rmse_30d + 0.5 * val_rolling_start_rmse_60d',
            'val_rolling_start_rmse_30d': float(record.get('val_rolling_start_rmse_30d', np.nan)),
            'val_rolling_start_rmse_60d': float(record.get('val_rolling_start_rmse_60d', np.nan)),
            'val_rolling_start_count_30d': float(record.get('val_rolling_start_count_30d', 0.0) or 0.0),
            'val_rolling_start_count_60d': float(record.get('val_rolling_start_count_60d', 0.0) or 0.0),
            'checkpoint_path': str(best_val_rolling_checkpoint_path),
        }
        best_val_rolling_metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return payload

    def _save_training_checkpoint(epoch_value, *, suffix=None):
        path = output_dir / (suffix or f'global_state_forecaster_epoch{int(epoch_value):04d}.pt')
        torch.save(
            {
                'architecture': 'MultiLakeStateForecaster',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': int(epoch_value),
                'training_history': history,
                'manifest': str(Path(manifest_path)),
                'task_mode': task_mode,
                'data_fill_mode': data_fill_mode,
                'profile_supervision_scope': profile_supervision_scope,
                'test_lake_id': test_lake_id or None,
                'test_lake_ids': list(test_lake_ids),
                'heldout_lake_groups': list(heldout_lake_groups),
                'train_lake_ids': [lake['lake_id'] for lake in train_lakes],
                'heldout_lake_ids': [lake['lake_id'] for lake in heldout_lakes],
                'excluded_lake_ids': [lake['lake_id'] for lake in excluded_lakes],
                'static_feature_dim': int(static_dim),
                'static_feature_keys': list(STATIC_FEATURE_KEYS),
                'transition_batch_size': int(transition_batch_size),
                'segment_rollout_batch_size': int(segment_rollout_batch_size),
                'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
                'train_diagnostic_mode': train_diagnostic_mode,
                'export_after_training': export_after_training,
                'export_max_depth_m': export_max_depth_m,
                'best_by_val_rolling_enabled': bool(best_val_rolling_enabled),
                'best_by_val_rolling_checkpoint_path': str(best_val_rolling_checkpoint_path),
                'best_by_val_rolling_metrics_path': str(best_val_rolling_metrics_path),
                'best_by_val_rolling_score': (
                    None if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
                ),
                'best_by_val_rolling_epoch': best_val_rolling_epoch,
            'freezing_energy_mode': freezing_energy_mode,
            'lake_adaptive_params': lake_adaptive_params,
            'shape_aware_mixing': shape_aware_mixing,
            'shape_mixing_strength': float(shape_mixing_strength),
            'stratification_mixing_cap': stratification_mixing_cap,
            'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
            'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
            'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
            'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
            'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
            'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
            'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
            'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
                'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
                'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
                'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
                'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
                'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
                'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
                'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
                'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
                'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
                'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
                'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
                'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
                'eval_every_epochs': int(eval_interval),
                'full_eval_every_epochs': int(full_eval_interval),
                'checkpoint_every_epochs': int(checkpoint_every_epochs),
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
                'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
                'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
                'segment_rollout_max_days': int(segment_rollout_max_days),
                'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
                'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
                'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
                'warm_season_column_heat_content_quantile_low': float(
                    warm_season_column_heat_content_quantile_low
                ),
                'warm_season_column_heat_content_quantile_high': float(
                    warm_season_column_heat_content_quantile_high
                ),
                'warm_season_column_heat_content_min_gap_days': int(
                    warm_season_column_heat_content_min_gap_days
                ),
                'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
                'advective_heat_source_mode': advective_heat_source_mode,
            },
            path,
        )
        return path

    for epoch in range(start_epoch, int(epochs)):
        epoch_start_time = time.perf_counter()
        transition_seconds = 0.0
        segment_seconds = 0.0
        evaluation_seconds = 0.0
        step_diagnostic_mode = train_diagnostic_mode
        model.train()
        optimizer.zero_grad()
        lake_losses = []
        transition_lake_losses = []
        transition_weighted_lake_losses = []
        detail_records = []
        segment_detail_records = []
        segment_weight_eff = _scheduled_weight(
            epoch,
            segment_rollout_loss_weight,
            segment_rollout_start_epoch,
            segment_rollout_ramp_epochs,
        )
        active_segment_days = _scheduled_segment_rollout_days(
            epoch,
            segment_rollout_start_epoch,
            segment_rollout_ramp_epochs,
            segment_rollout_max_days,
        )
        teacher_forcing_probability = _scheduled_teacher_forcing_probability(
            epoch,
            segment_rollout_start_epoch,
            segment_rollout_ramp_epochs,
            teacher_forcing_start,
            teacher_forcing_end,
        )
        if cross_lake_batch_mode == 'on':
            transition_start_time = time.perf_counter()
            transition_results = _transition_losses_for_lakes_cross_batch(
                model,
                train_lakes,
                pair_key=supervision_pair_key,
                transition_batch_mode=transition_batch_mode,
                transition_batch_size=transition_batch_size,
                cross_lake_batch_size=cross_lake_batch_size,
                profile_huber_delta=profile_huber_delta,
                lst_surface_weight=lst_surface_weight,
                energy_balance_weight=energy_balance_weight,
                residual_regularization_weight=residual_regularization_weight,
                daily_tendency_weight=daily_tendency_weight,
                physical_scale_regularization_weight=physical_scale_regularization_weight,
                physical_scale_smoothness_weight=physical_scale_smoothness_weight,
                kd_prior_regularization_weight=kd_prior_regularization_weight,
                adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                heat_content_transition_weight=heat_content_transition_weight,
                heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                heat_content_transition_effective_max=heat_content_transition_effective_max,
                task_mode=task_mode,
                hard_density_stability=hard_density_stability_active,
                step_diagnostic_mode=step_diagnostic_mode,
            )
            transition_seconds += time.perf_counter() - transition_start_time

            sequence_results_by_lake = {}
            if segment_weight_eff > 0.0 and active_segment_days > 0:
                selected_sequences_by_lake = {
                    lake_idx: _segment_rollout_sequences_for_epoch(
                        lake,
                        supervision_sequence_key,
                        active_segment_days,
                        segment_rollout_samples_per_lake,
                        epoch,
                    )
                    for lake_idx, lake in enumerate(train_lakes)
                }
                segment_start_time = time.perf_counter()
                sequence_results_by_lake = _segment_rollout_sequence_losses_for_lakes_cross_batch(
                    model,
                    train_lakes,
                    selected_sequences_by_lake,
                    segment_rollout_batch_mode=segment_rollout_batch_mode,
                    segment_rollout_batch_size=segment_rollout_batch_size,
                    cross_lake_batch_size=cross_lake_batch_size,
                    active_max_days=active_segment_days,
                    cached_batches=(
                        cross_lake_segment_rollout_epoch_batches[epoch]
                        if epoch < len(cross_lake_segment_rollout_epoch_batches) else None
                    ),
                    profile_huber_delta=profile_huber_delta,
                    task_mode=task_mode,
                    teacher_forcing_probability=teacher_forcing_probability,
                    state_noise_weight=state_noise_weight,
                    residual_regularization_weight=residual_regularization_weight,
                    daily_tendency_weight=daily_tendency_weight,
                    residual_time_smooth_weight=residual_time_smooth_weight,
                    physical_scale_regularization_weight=physical_scale_regularization_weight,
                    physical_scale_smoothness_weight=physical_scale_smoothness_weight,
                    kd_prior_regularization_weight=kd_prior_regularization_weight,
                    adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                    heat_content_transition_weight=heat_content_transition_weight,
                    heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                    heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                    heat_content_transition_effective_max=heat_content_transition_effective_max,
                    segment_rollout_lst_surface_weight=segment_rollout_lst_surface_weight,
                    warm_season_column_heat_content_weight=warm_season_column_heat_content_weight,
                    warm_season_column_heat_content_min_gap_days=warm_season_column_heat_content_min_gap_days,
                    hard_density_stability=hard_density_stability_active,
                    step_diagnostic_mode=step_diagnostic_mode,
                    lookup_split=supervision_sequence_key,
                )
                segment_seconds += time.perf_counter() - segment_start_time

            for lake_idx, lake in enumerate(train_lakes):
                pair_losses, pair_details = transition_results[lake_idx]
                detail_records.extend(pair_details)
                transition_lake_loss = torch.stack(pair_losses).mean()
                transition_weighted_loss = float(transition_loss_weight) * transition_lake_loss
                lake_loss = transition_weighted_loss
                transition_lake_losses.append(transition_lake_loss.detach())
                transition_weighted_lake_losses.append(transition_weighted_loss.detach())
                if segment_weight_eff > 0.0 and active_segment_days > 0:
                    sequence_losses = []
                    for sequence_loss, count, sequence_details in sequence_results_by_lake.get(lake_idx, []):
                        if count > 0:
                            sequence_losses.append(sequence_loss)
                            segment_detail_records.append({
                                **sequence_details,
                                'segment_rollout_supervision_count': torch.tensor(
                                    float(count),
                                    device=lake['depths'].device,
                                ),
                                'segment_rollout_sequence_count': torch.tensor(
                                    1.0,
                                    device=lake['depths'].device,
                                ),
                            })
                    if sequence_losses:
                        segment_loss = torch.stack(sequence_losses).mean()
                        lake_loss = lake_loss + float(segment_weight_eff) * segment_loss
                    else:
                        segment_detail_records.append({
                            'segment_rollout_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_profile_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_horizon_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_max_target_gap_days': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_surface_weight': torch.tensor(
                                float(segment_rollout_lst_surface_weight),
                                device=lake['depths'].device,
                            ),
                            'segment_rollout_residual_smooth_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_daily_tendency_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_residual_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_physical_scale_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_physical_scale_smoothness_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_kd_prior_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_kd_prior_regularization_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_adaptive_parameter_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_min': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_max': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_warm_factor_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_error_c_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon14_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon30_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon60_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_sequence_count': torch.tensor(0.0, device=lake['depths'].device),
                        })
                lake_losses.append(lake_loss)
        else:
            for lake in train_lakes:
                transition_start_time = time.perf_counter()
                pair_losses, pair_details = _transition_losses_for_lake(
                    model,
                    lake,
                    lake['pairs'][supervision_pair_key],
                    transition_batch_mode=transition_batch_mode,
                    transition_batch_size=transition_batch_size,
                    lookup_split=supervision_pair_key,
                    profile_huber_delta=profile_huber_delta,
                    lst_surface_weight=lst_surface_weight,
                    energy_balance_weight=energy_balance_weight,
                    residual_regularization_weight=residual_regularization_weight,
                    daily_tendency_weight=daily_tendency_weight,
                    physical_scale_regularization_weight=physical_scale_regularization_weight,
                    physical_scale_smoothness_weight=physical_scale_smoothness_weight,
                    kd_prior_regularization_weight=kd_prior_regularization_weight,
                    adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                    heat_content_transition_weight=heat_content_transition_weight,
                    heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                    heat_content_transition_season_factors=lake['heat_content_transition_season_factors'],
                    heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                    heat_content_transition_effective_max=heat_content_transition_effective_max,
                    task_mode=task_mode,
                    hard_density_stability=hard_density_stability_active,
                    step_diagnostic_mode=step_diagnostic_mode,
                )
                detail_records.extend(pair_details)
                transition_seconds += time.perf_counter() - transition_start_time
                transition_lake_loss = torch.stack(pair_losses).mean()
                transition_weighted_loss = float(transition_loss_weight) * transition_lake_loss
                lake_loss = transition_weighted_loss
                transition_lake_losses.append(transition_lake_loss.detach())
                transition_weighted_lake_losses.append(transition_weighted_loss.detach())
                if segment_weight_eff > 0.0 and active_segment_days > 0:
                    segment_start_time = time.perf_counter()
                    selected_sequences = _segment_rollout_sequences_for_epoch(
                        lake,
                        supervision_sequence_key,
                        active_segment_days,
                        segment_rollout_samples_per_lake,
                        epoch,
                    )
                    sequence_results = _segment_rollout_sequence_losses_for_lake(
                        model,
                        lake,
                        selected_sequences,
                        segment_rollout_batch_mode=segment_rollout_batch_mode,
                        segment_rollout_batch_size=segment_rollout_batch_size,
                        cached_batches=_segment_rollout_batches_for_epoch(
                            lake,
                            supervision_sequence_key,
                            active_segment_days,
                            segment_rollout_samples_per_lake,
                            epoch,
                        ),
                        active_max_days=active_segment_days,
                        profile_huber_delta=profile_huber_delta,
                        task_mode=task_mode,
                        teacher_forcing_probability=teacher_forcing_probability,
                        state_noise_weight=state_noise_weight,
                        residual_regularization_weight=residual_regularization_weight,
                        daily_tendency_weight=daily_tendency_weight,
                        residual_time_smooth_weight=residual_time_smooth_weight,
                        physical_scale_regularization_weight=physical_scale_regularization_weight,
                        physical_scale_smoothness_weight=physical_scale_smoothness_weight,
                        kd_prior_regularization_weight=kd_prior_regularization_weight,
                        adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                        heat_content_transition_weight=heat_content_transition_weight,
                        heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                        heat_content_transition_season_factors=lake['heat_content_transition_season_factors'],
                        heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                        heat_content_transition_effective_max=heat_content_transition_effective_max,
                        segment_rollout_lst_surface_weight=segment_rollout_lst_surface_weight,
                        warm_season_column_heat_content_weight=warm_season_column_heat_content_weight,
                        warm_season_column_heat_content_min_gap_days=warm_season_column_heat_content_min_gap_days,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                        lookup_split=supervision_sequence_key,
                    )
                    sequence_losses = []
                    for sequence_loss, count, sequence_details in sequence_results:
                        if count > 0:
                            sequence_losses.append(sequence_loss)
                            segment_detail_records.append({
                                **sequence_details,
                                'segment_rollout_supervision_count': torch.tensor(
                                    float(count),
                                    device=lake['depths'].device,
                                ),
                                'segment_rollout_sequence_count': torch.tensor(
                                    1.0,
                                    device=lake['depths'].device,
                                ),
                            })
                    if sequence_losses:
                        segment_loss = torch.stack(sequence_losses).mean()
                        lake_loss = lake_loss + float(segment_weight_eff) * segment_loss
                    else:
                        segment_detail_records.append({
                            'segment_rollout_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_profile_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_horizon_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_max_target_gap_days': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_lst_surface_weight': torch.tensor(
                                float(segment_rollout_lst_surface_weight),
                                device=lake['depths'].device,
                            ),
                            'segment_rollout_residual_smooth_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_daily_tendency_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_residual_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_physical_scale_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_physical_scale_smoothness_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_kd_prior_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_kd_prior_regularization_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_adaptive_parameter_regularization_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_min': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_heat_content_transition_effective_weight_max': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_weighted_loss': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_warm_factor_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_error_c_mean': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon14_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon30_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_warm_column_heat_content_horizon60_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_supervision_count': torch.tensor(0.0, device=lake['depths'].device),
                            'segment_rollout_sequence_count': torch.tensor(0.0, device=lake['depths'].device),
                        })
                    segment_seconds += time.perf_counter() - segment_start_time
                lake_losses.append(lake_loss)
        total_loss = torch.stack(lake_losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        eval_epoch_allowed = epoch > 0
        should_mini_evaluate = eval_epoch_allowed and (
            (epoch + 1) % max(1, eval_interval) == 0
            or epoch == int(epochs) - 1
        )
        should_full_evaluate = eval_epoch_allowed and full_eval_interval > 0 and (
            (epoch + 1) % full_eval_interval == 0
            or epoch == int(epochs) - 1
        )
        should_light_evaluate = (
            should_full_evaluate
            or should_mini_evaluate
        )
        should_evaluate = should_mini_evaluate or should_light_evaluate or should_full_evaluate
        checkpoint_this_epoch = checkpoint_every_epochs > 0 and (
            (epoch + 1) % checkpoint_every_epochs == 0 or epoch == int(epochs) - 1
        )
        history_diagnostic_enabled = (
            should_evaluate
            or checkpoint_this_epoch
            or epoch == int(epochs) - 1
            or (
                history_diagnostic_every_epochs > 0
                and (epoch + 1) % history_diagnostic_every_epochs == 0
            )
        )
        detail_mean = _mean_detail if history_diagnostic_enabled else (
            lambda _details, _key: np.nan
        )
        eval_mode = 'none'
        if should_evaluate:
            evaluation_start_time = time.perf_counter()
            eval_mode = 'full' if should_full_evaluate else ('light' if should_light_evaluate else 'mini')
            model.eval()
            train_rmse = {
                lake['lake_id']: evaluate_lake_pairs(
                    model,
                    lake,
                    lake['pairs']['train'],
                    task_mode=task_mode,
                    hard_density_stability=hard_density_stability_active,
                )
                for lake in train_lakes
            }
            val_rmse = {
                lake['lake_id']: evaluate_lake_pairs(
                    model,
                    lake,
                    lake['pairs']['val'],
                    task_mode=task_mode,
                    hard_density_stability=hard_density_stability_active,
                )
                for lake in train_lakes
            }
            heldout_rmse = {
                lake['lake_id']: evaluate_lake_pairs(
                    model,
                    lake,
                    lake['pairs']['all'],
                    task_mode=task_mode,
                    hard_density_stability=hard_density_stability_active,
                )
                for lake in heldout_lakes
            }
            if should_light_evaluate:
                train_persistence_rmse = {
                    lake['lake_id']: evaluate_persistence_pairs(lake, lake['pairs']['train'])
                    for lake in train_lakes
                }
                val_persistence_rmse = {
                    lake['lake_id']: evaluate_persistence_pairs(lake, lake['pairs']['val'])
                    for lake in train_lakes
                }
                heldout_persistence_rmse = {
                    lake['lake_id']: evaluate_persistence_pairs(lake, lake['pairs']['all'])
                    for lake in heldout_lakes
                }
                train_horizon = [
                    evaluate_lake_pair_horizons(
                        model,
                        lake,
                        lake['pairs']['train'],
                        horizons=(1, 3, 7, 14, 30, 60),
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                    )
                    for lake in train_lakes
                ]
                heldout_transition_horizon = [
                    evaluate_lake_pair_horizons(
                        model,
                        lake,
                        lake['pairs']['all'],
                        horizons=(1, 3, 7, 14, 30, 60),
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                    )
                    for lake in heldout_lakes
                ]
            else:
                train_persistence_rmse = {}
                val_persistence_rmse = {}
                heldout_persistence_rmse = {}
                train_horizon = []
                heldout_transition_horizon = []
            heldout_free_roll = {}
            if should_full_evaluate:
                heldout_free_roll = evaluate_heldout_free_rolls(
                    model,
                    heldout_lakes,
                    task_mode=task_mode,
                    horizons=(1, 3, 7, 14, 30, 60),
                    init_mode=init_mode,
                    spinup_days=spinup_days,
                    spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                    spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                    rollout_start_date=rollout_start_date,
                    hard_density_stability=hard_density_stability_active,
                    batch_size=rolling_horizon_batch_size,
                )
                heldout_rolling_start_horizon = evaluate_lakes_rolling_start_horizons(
                    model,
                    heldout_lakes,
                    horizons=(1, 3, 7, 14, 30, 60),
                    task_mode=task_mode,
                    max_start_profiles=rolling_horizon_eval_max_starts,
                    hard_density_stability=hard_density_stability_active,
                    batch_size=rolling_horizon_batch_size,
                    rollout_batch_step_mode=rollout_batch_step_mode,
                    use_batched=rolling_horizon_batch_mode == 'on',
                )
                val_rolling_start_horizon = evaluate_lakes_rolling_start_horizons(
                    model,
                    train_lakes,
                    horizons=(1, 3, 7, 14, 30, 60),
                    task_mode=task_mode,
                    max_start_profiles=rolling_horizon_eval_max_starts,
                    hard_density_stability=hard_density_stability_active,
                    lookup_split='val',
                    batch_size=rolling_horizon_batch_size,
                    rollout_batch_step_mode=rollout_batch_step_mode,
                    use_batched=rolling_horizon_batch_mode == 'on',
                )
            else:
                heldout_rolling_start_horizon = {}
                val_rolling_start_horizon = {}
            heldout_rolling_start_records = list(heldout_rolling_start_horizon.values())
            val_rolling_start_records = list(val_rolling_start_horizon.values())
            record = {
                'epoch': epoch,
                'loss': float(total_loss.detach().cpu()),
                'profile_supervision_scope': profile_supervision_scope,
                'train_supervision_pair_count': int(sum(
                    len(lake['pairs'][supervision_pair_key]) for lake in train_lakes
                )),
                'train_supervision_segment_sequence_count': int(sum(
                    len(lake['segment_rollout_sequences'][supervision_sequence_key]) for lake in train_lakes
                )),
                'transition_loss_weight': float(transition_loss_weight),
                'transition_loss_unweighted': float(torch.stack(transition_lake_losses).mean().detach().cpu())
                if transition_lake_losses else np.nan,
                'transition_loss_weighted': float(torch.stack(transition_weighted_lake_losses).mean().detach().cpu())
                if transition_weighted_lake_losses else np.nan,
                'profile_loss': _mean_detail(detail_records, 'profile_loss'),
                'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
                'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
                'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
                'warm_season_column_heat_content_quantile_low': float(warm_season_column_heat_content_quantile_low),
                'warm_season_column_heat_content_quantile_high': float(warm_season_column_heat_content_quantile_high),
                'warm_season_column_heat_content_min_gap_days': int(warm_season_column_heat_content_min_gap_days),
                'segment_rollout_loss': _mean_detail(segment_detail_records, 'segment_rollout_loss'),
                'segment_rollout_profile_loss': _mean_detail(segment_detail_records, 'segment_rollout_profile_loss'),
                'segment_rollout_horizon_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_horizon_weight_mean'),
                'segment_rollout_max_target_gap_days': _mean_detail(segment_detail_records, 'segment_rollout_max_target_gap_days'),
                'segment_rollout_lst_loss': _mean_detail(segment_detail_records, 'segment_rollout_lst_loss'),
                'segment_rollout_lst_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_lst_supervision_count'),
                'segment_rollout_lst_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_lst_weight_mean'),
                'segment_rollout_residual_smooth_loss': _mean_detail(segment_detail_records, 'segment_rollout_residual_smooth_loss'),
                'segment_rollout_daily_tendency_loss': _mean_detail(segment_detail_records, 'segment_rollout_daily_tendency_loss'),
                'segment_rollout_residual_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_residual_regularization_loss'),
                'segment_rollout_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_supervision_count'),
                'segment_rollout_sequence_count': _mean_detail(segment_detail_records, 'segment_rollout_sequence_count'),
                'segment_rollout_weight_eff': float(segment_weight_eff),
                'segment_rollout_active_days': int(active_segment_days),
                'teacher_forcing_probability': float(teacher_forcing_probability),
                'state_noise_weight': float(state_noise_weight),
                'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
                'lst_feature_dropout_applied_mean': detail_mean(detail_records, 'lst_feature_dropout_applied_mean'),
                'residual_time_smooth_weight': float(residual_time_smooth_weight),
                'daily_tendency_loss': _mean_detail(detail_records, 'daily_tendency_loss'),
                'residual_regularization_loss': _mean_detail(detail_records, 'residual_regularization_loss'),
                'physical_scale_reg_loss': _mean_detail(detail_records, 'physical_scale_reg_loss'),
                'physical_scale_smooth_loss': _mean_detail(detail_records, 'physical_scale_smooth_loss'),
                'kd_prior_regularization_loss': _mean_detail(detail_records, 'kd_prior_regularization_loss'),
                'kd_prior_regularization_weighted_loss': _mean_detail(detail_records, 'kd_prior_regularization_weighted_loss'),
                'adaptive_parameter_regularization_loss': _mean_detail(detail_records, 'adaptive_parameter_regularization_loss'),
                'heat_content_transition_loss': _mean_detail(detail_records, 'heat_content_transition_loss'),
                'heat_content_transition_weighted_loss': _mean_detail(detail_records, 'heat_content_transition_weighted_loss'),
                'heat_content_transition_effective_weight_mean': _mean_detail(detail_records, 'heat_content_transition_effective_weight_mean'),
                'heat_content_transition_effective_weight_min': _min_detail(detail_records, 'heat_content_transition_effective_weight_min'),
                'heat_content_transition_effective_weight_max': _max_detail(detail_records, 'heat_content_transition_effective_weight_max'),
                'shortwave_scale_mean': detail_mean(detail_records, 'shortwave_scale_mean'),
                'shortwave_scale_min': detail_mean(detail_records, 'shortwave_scale_min'),
                'shortwave_scale_max': detail_mean(detail_records, 'shortwave_scale_max'),
                'cooling_scale_mean': detail_mean(detail_records, 'cooling_scale_mean'),
                'cooling_scale_min': detail_mean(detail_records, 'cooling_scale_min'),
                'cooling_scale_max': detail_mean(detail_records, 'cooling_scale_max'),
                'cooling_scale_effective_mean': detail_mean(detail_records, 'cooling_scale_effective_mean'),
                'surface_flux_bias_mean_wm2': detail_mean(detail_records, 'surface_flux_bias_mean_wm2'),
                'open_water_sensible_heat_mean_wm2': detail_mean(detail_records, 'open_water_sensible_heat_mean_wm2'),
                'open_water_latent_heat_mean_wm2': detail_mean(detail_records, 'open_water_latent_heat_mean_wm2'),
                'temperature_floor_heat_injection_mean_wm2': detail_mean(detail_records, 'temperature_floor_heat_injection_mean_wm2'),
                'freezing_storage_mean_j_m2': detail_mean(detail_records, 'freezing_storage_mean_j_m2'),
                'freezing_storage_ice_mean_j_m2': detail_mean(detail_records, 'freezing_storage_ice_mean_j_m2'),
                'freezing_storage_surface_fraction_mean': detail_mean(detail_records, 'freezing_storage_surface_fraction_mean'),
                'freezing_storage_deep_fraction_mean': detail_mean(detail_records, 'freezing_storage_deep_fraction_mean'),
                'freezing_storage_change_mean_wm2': detail_mean(detail_records, 'freezing_storage_change_mean_wm2'),
                'effective_heat_tendency_mean_wm2': detail_mean(detail_records, 'effective_heat_tendency_mean_wm2'),
                'advective_heat_source_c_per_day_mean': detail_mean(detail_records, 'advective_heat_source_c_per_day_mean'),
                'advective_heat_source_c_per_day_max': detail_mean(detail_records, 'advective_heat_source_c_per_day_max'),
                'advective_exchange_fraction_per_day': detail_mean(detail_records, 'advective_exchange_fraction_per_day'),
                'advective_heat_source_active_mean': detail_mean(detail_records, 'advective_heat_source_active_mean'),
                'background_nn_kz_mean': detail_mean(detail_records, 'background_nn_kz_mean'),
                'background_nn_kz_deep_mean': detail_mean(detail_records, 'background_nn_kz_deep_mean'),
                'turbulent_nn_kz_mean': detail_mean(detail_records, 'turbulent_nn_kz_mean'),
                'turbulent_nn_kz_deep_mean': detail_mean(detail_records, 'turbulent_nn_kz_deep_mean'),
                'gated_turbulent_nn_kz_mean': detail_mean(detail_records, 'gated_turbulent_nn_kz_mean'),
                'gated_turbulent_nn_kz_deep_mean': detail_mean(detail_records, 'gated_turbulent_nn_kz_deep_mean'),
                'kd_base_mean': detail_mean(detail_records, 'kd_base_mean'),
                'nn_kd_multiplier_mean': detail_mean(detail_records, 'nn_kd_multiplier_mean'),
                'kd_prior_regularization_loss_mean': detail_mean(detail_records, 'kd_prior_regularization_loss_mean'),
                'adaptive_wind_kz_scale_mean': detail_mean(detail_records, 'adaptive_wind_kz_scale_mean'),
                'adaptive_turbulent_flux_blend_alpha_mean': detail_mean(detail_records, 'adaptive_turbulent_flux_blend_alpha_mean'),
                'adaptive_kd_multiplier_mean': detail_mean(detail_records, 'adaptive_kd_multiplier_mean'),
                'adaptive_turbulent_exchange_scale_mean': detail_mean(detail_records, 'adaptive_turbulent_exchange_scale_mean'),
                'adaptive_convective_mixing_scale_mean': detail_mean(detail_records, 'adaptive_convective_mixing_scale_mean'),
                'adaptive_ice_shortwave_scale_mean': detail_mean(detail_records, 'adaptive_ice_shortwave_scale_mean'),
                'lake_shape_wind_factor_mean': detail_mean(detail_records, 'lake_shape_wind_factor_mean'),
                'lake_shape_decay_depth_mean_m': detail_mean(detail_records, 'lake_shape_decay_depth_mean_m'),
                'stratification_mixing_gate_mean': detail_mean(detail_records, 'stratification_mixing_gate_mean'),
                'stratification_mixing_gate_min': detail_mean(detail_records, 'stratification_mixing_gate_min'),
                'stratification_mixing_gate_deep_mean': detail_mean(
                    detail_records,
                    'stratification_mixing_gate_deep_mean',
                ),
                'segment_rollout_physical_scale_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_physical_scale_regularization_loss'),
                'segment_rollout_physical_scale_smoothness_loss': _mean_detail(segment_detail_records, 'segment_rollout_physical_scale_smoothness_loss'),
                'segment_rollout_kd_prior_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_prior_regularization_loss'),
                'segment_rollout_kd_prior_regularization_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_prior_regularization_weighted_loss'),
                'segment_rollout_adaptive_parameter_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_adaptive_parameter_regularization_loss'),
                'segment_rollout_heat_content_transition_loss': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_loss'),
                'segment_rollout_heat_content_transition_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_weighted_loss'),
                'segment_rollout_heat_content_transition_effective_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_mean'),
                'segment_rollout_heat_content_transition_effective_weight_min': _min_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_min'),
                'segment_rollout_heat_content_transition_effective_weight_max': _max_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_max'),
                'segment_rollout_warm_column_heat_content_loss': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_loss'),
                'segment_rollout_warm_column_heat_content_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_weighted_loss'),
                'segment_rollout_warm_column_heat_content_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_supervision_count'),
                'segment_rollout_warm_column_heat_content_warm_factor_mean': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_warm_factor_mean'),
                'segment_rollout_warm_column_heat_content_error_c_mean': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_error_c_mean'),
                'segment_rollout_warm_column_heat_content_horizon14_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon14_count'),
                'segment_rollout_warm_column_heat_content_horizon30_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon30_count'),
                'segment_rollout_warm_column_heat_content_horizon60_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon60_count'),
                'energy_loss': _mean_detail(detail_records, 'energy_loss'),
                'physical_scale_regularization_weight': float(physical_scale_regularization_weight),
                'physical_scale_smoothness_weight': float(physical_scale_smoothness_weight),
                'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
                'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
                'heat_content_transition_weight': float(heat_content_transition_weight),
                'heat_content_transition_weight_base': float(heat_content_transition_weight),
                'heat_content_full_column_min_coverage': float(heat_content_full_column_min_coverage),
                'heat_content_transition_depth_factor': 1.0 if heat_content_transition_depth_factor else 0.0,
                'heat_content_transition_effective_max': float(heat_content_transition_effective_max),
                'hard_density_stability': hard_density_stability_mode,
                'hard_density_stability_active': bool(hard_density_stability_active),
                'turbulent_flux_mode': turbulent_flux_mode,
                'turbulent_flux_blend_alpha': float(turbulent_flux_blend_alpha),
                'lake_adaptive_params': lake_adaptive_params,
                'shape_aware_mixing': shape_aware_mixing,
                'shape_mixing_strength': float(shape_mixing_strength),
                'stratification_mixing_cap': stratification_mixing_cap,
                'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
                'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
                'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
                'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
                'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
                'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
                'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
                'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
                'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
                'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
                'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
                'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
                'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
                'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
                'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
                'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
                'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
                'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
                'freezing_energy_mode': freezing_energy_mode,
                'advective_heat_source_mode': advective_heat_source_mode,
                'checkpoint_every_epochs': int(checkpoint_every_epochs),
                'eval_every_epochs': int(eval_interval),
                'full_eval_every_epochs': int(full_eval_interval),
                'eval_mode': eval_mode,
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'history_diagnostic_enabled': bool(history_diagnostic_enabled),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                'transition_batch_size': int(transition_batch_size),
                'segment_rollout_batch_size': int(segment_rollout_batch_size),
                'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
                'train_diagnostic_mode': train_diagnostic_mode,
                'export_after_training': export_after_training,
                'export_max_depth_m': export_max_depth_m,
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                'rolling_horizon_eval_max_starts': int(rolling_horizon_eval_max_starts),
                'train_mean_rmse': _nanmean_or_nan(train_rmse.values()) if train_rmse else np.nan,
                'val_mean_rmse': _nanmean_or_nan(val_rmse.values()) if val_rmse else np.nan,
                'heldout_mean_rmse': _nanmean_or_nan(heldout_rmse.values()) if heldout_rmse else np.nan,
                'heldout_transition_mean_rmse': _nanmean_or_nan(heldout_rmse.values()) if heldout_rmse else np.nan,
                'train_persistence_mean_rmse': _nanmean_or_nan(train_persistence_rmse.values()) if train_persistence_rmse else np.nan,
                'val_persistence_mean_rmse': _nanmean_or_nan(val_persistence_rmse.values()) if val_persistence_rmse else np.nan,
                'heldout_persistence_mean_rmse': _nanmean_or_nan(heldout_persistence_rmse.values()) if heldout_persistence_rmse else np.nan,
                'heldout_free_roll_mean_rmse': _nanmean_or_nan(
                    value.get('rmse', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_free_roll_mean_bias': _nanmean_or_nan(
                    value.get('bias', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_observed_point_mean_rmse': _nanmean_or_nan(
                    value.get('observed_point_rmse', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_observed_point_mean_mae': _nanmean_or_nan(
                    value.get('observed_point_mae', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_observed_point_mean_bias': _nanmean_or_nan(
                    value.get('observed_point_bias', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_observed_point_total_count': int(sum(
                    value.get('observed_point_count', 0) for value in heldout_free_roll.values()
                )) if heldout_free_roll else 0,
                'heldout_post_spinup_mean_rmse': _nanmean_or_nan(
                    value.get('post_spinup_rmse', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_post_spinup_mean_bias': _nanmean_or_nan(
                    value.get('post_spinup_bias', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
            }
            for horizon in (1, 3, 7, 14, 30, 60):
                rmse_key = f'rmse_{horizon}d'
                count_key = f'count_{horizon}d'
                record[f'train_transition_rmse_{horizon}d'] = _mean_numeric_records(train_horizon, rmse_key)
                record[f'train_transition_count_{horizon}d'] = _mean_numeric_records(train_horizon, count_key)
                record[f'heldout_transition_rmse_{horizon}d'] = _mean_numeric_records(heldout_transition_horizon, rmse_key)
                record[f'heldout_transition_count_{horizon}d'] = _mean_numeric_records(heldout_transition_horizon, count_key)
                record[f'val_rolling_start_rmse_{horizon}d'] = _mean_numeric_records(val_rolling_start_records, rmse_key)
                record[f'val_rolling_start_bias_{horizon}d'] = _mean_numeric_records(val_rolling_start_records, f'bias_{horizon}d')
                record[f'val_rolling_start_count_{horizon}d'] = _mean_numeric_records(val_rolling_start_records, count_key)
                rolling_rmse = _mean_numeric_records(heldout_rolling_start_records, rmse_key)
                rolling_bias = _mean_numeric_records(heldout_rolling_start_records, f'bias_{horizon}d')
                rolling_count = _mean_numeric_records(heldout_rolling_start_records, count_key)
                record[f'heldout_rolling_start_rmse_{horizon}d'] = rolling_rmse
                record[f'heldout_rolling_start_bias_{horizon}d'] = rolling_bias
                record[f'heldout_rolling_start_count_{horizon}d'] = rolling_count
                record[f'heldout_free_roll_rmse_{horizon}d'] = rolling_rmse
                record[f'heldout_free_roll_bias_{horizon}d'] = rolling_bias
                record[f'heldout_free_roll_count_{horizon}d'] = rolling_count
                record[f'heldout_initial_free_roll_rmse_{horizon}d'] = _nanmean_or_nan(
                    value.get('horizon_metrics', {}).get(rmse_key, np.nan)
                    for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan
                record[f'heldout_initial_free_roll_bias_{horizon}d'] = _nanmean_or_nan(
                    value.get('horizon_metrics', {}).get(f'bias_{horizon}d', np.nan)
                    for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan
                record[f'heldout_initial_free_roll_count_{horizon}d'] = _nanmean_or_nan(
                    value.get('horizon_metrics', {}).get(count_key, np.nan)
                    for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan
            for month in range(1, 13):
                record[f'heat_content_transition_season_factor_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.mean,
                )
                record[f'heat_content_transition_season_factor_min_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.min,
                )
                record[f'heat_content_transition_season_factor_max_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.max,
                )
            if should_light_evaluate:
                for key, value in train_rmse.items():
                    record[f'{key}_train_rmse'] = value
                for key, value in val_rmse.items():
                    record[f'{key}_val_rmse'] = value
                    rolling_metrics = val_rolling_start_horizon.get(key, {})
                    for horizon in (1, 3, 7, 14, 30, 60):
                        record[f'{key}_val_rolling_start_rmse_{horizon}d'] = rolling_metrics.get(f'rmse_{horizon}d', np.nan)
                        record[f'{key}_val_rolling_start_bias_{horizon}d'] = rolling_metrics.get(f'bias_{horizon}d', np.nan)
                        record[f'{key}_val_rolling_start_count_{horizon}d'] = rolling_metrics.get(f'count_{horizon}d', np.nan)
                for key, value in heldout_rmse.items():
                    record[f'{key}_heldout_rmse'] = value
                    record[f'{key}_heldout_transition_rmse'] = value
                for key, value in train_persistence_rmse.items():
                    record[f'{key}_train_persistence_rmse'] = value
                for key, value in val_persistence_rmse.items():
                    record[f'{key}_val_persistence_rmse'] = value
                for key, value in heldout_persistence_rmse.items():
                    record[f'{key}_heldout_persistence_rmse'] = value
            for key, value in heldout_free_roll.items():
                record[f'{key}_heldout_free_roll_rmse'] = value.get('rmse', np.nan)
                record[f'{key}_heldout_free_roll_mae'] = value.get('mae', np.nan)
                record[f'{key}_heldout_free_roll_bias'] = value.get('bias', np.nan)
                record[f'{key}_heldout_free_roll_profiles'] = value.get('n_profiles', 0)
                record[f'{key}_heldout_observed_point_rmse'] = value.get('observed_point_rmse', np.nan)
                record[f'{key}_heldout_observed_point_mae'] = value.get('observed_point_mae', np.nan)
                record[f'{key}_heldout_observed_point_bias'] = value.get('observed_point_bias', np.nan)
                record[f'{key}_heldout_observed_point_count'] = value.get('observed_point_count', 0)
                record[f'{key}_heldout_observed_point_profiles'] = value.get('observed_point_profile_count', 0)
                for observed_group in ('surface', 'mid', 'deep', 'winter', 'spring', 'summer', 'fall'):
                    record[f'{key}_heldout_observed_point_{observed_group}_rmse'] = value.get(
                        f'observed_point_{observed_group}_rmse',
                        np.nan,
                    )
                    record[f'{key}_heldout_observed_point_{observed_group}_bias'] = value.get(
                        f'observed_point_{observed_group}_bias',
                        np.nan,
                    )
                    record[f'{key}_heldout_observed_point_{observed_group}_count'] = value.get(
                        f'observed_point_{observed_group}_count',
                        0,
                    )
                record[f'{key}_heldout_post_spinup_rmse'] = value.get('post_spinup_rmse', np.nan)
                record[f'{key}_heldout_post_spinup_bias'] = value.get('post_spinup_bias', np.nan)
                record[f'{key}_heldout_spinup_days_used'] = value.get('spinup_days_used', np.nan)
                rolling_metrics = heldout_rolling_start_horizon.get(key, {})
                for horizon in (1, 3, 7, 14, 30, 60):
                    initial_metrics = value.get('horizon_metrics', {})
                    record[f'{key}_heldout_initial_free_roll_rmse_{horizon}d'] = initial_metrics.get(f'rmse_{horizon}d', np.nan)
                    record[f'{key}_heldout_initial_free_roll_bias_{horizon}d'] = initial_metrics.get(f'bias_{horizon}d', np.nan)
                    record[f'{key}_heldout_initial_free_roll_count_{horizon}d'] = initial_metrics.get(f'count_{horizon}d', np.nan)
                    record[f'{key}_heldout_rolling_start_rmse_{horizon}d'] = rolling_metrics.get(f'rmse_{horizon}d', np.nan)
                    record[f'{key}_heldout_rolling_start_bias_{horizon}d'] = rolling_metrics.get(f'bias_{horizon}d', np.nan)
                    record[f'{key}_heldout_rolling_start_count_{horizon}d'] = rolling_metrics.get(f'count_{horizon}d', np.nan)
                    record[f'{key}_heldout_free_roll_rmse_{horizon}d'] = rolling_metrics.get(f'rmse_{horizon}d', np.nan)
                    record[f'{key}_heldout_free_roll_bias_{horizon}d'] = rolling_metrics.get(f'bias_{horizon}d', np.nan)
                    record[f'{key}_heldout_free_roll_count_{horizon}d'] = rolling_metrics.get(f'count_{horizon}d', np.nan)
            best_candidate_score, best_skip_reason = (
                _best_val_rolling_candidate(record)
                if should_full_evaluate else (None, 'skipped: no full eval this epoch')
            )
            best_checkpoint_updated = False
            if best_candidate_score is not None and best_candidate_score < best_val_rolling_score:
                best_val_rolling_score = float(best_candidate_score)
                best_val_rolling_epoch = int(epoch)
                best_checkpoint_updated = True
                best_skip_reason = None
            record['best_by_val_rolling_enabled'] = bool(best_val_rolling_enabled)
            record['best_by_val_rolling_score'] = (
                np.nan if best_candidate_score is None else float(best_candidate_score)
            )
            record['best_by_val_rolling_best_score'] = (
                np.nan if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
            )
            record['best_by_val_rolling_checkpoint_updated'] = bool(best_checkpoint_updated)
            record['best_by_val_rolling_skipped_reason'] = best_skip_reason or ''
            record['best_by_val_rolling_checkpoint_path'] = (
                str(best_val_rolling_checkpoint_path) if best_val_rolling_checkpoint_path.exists() or best_checkpoint_updated else ''
            )
            history.append(record)
            if best_checkpoint_updated:
                _save_training_checkpoint(epoch, suffix='best_by_val_rolling.pt')
                _write_best_val_rolling_metrics(record, best_val_rolling_score)
                record['best_by_val_rolling_checkpoint_path'] = str(best_val_rolling_checkpoint_path)
            evaluation_seconds = time.perf_counter() - evaluation_start_time
            if profile_runtime:
                record['transition_seconds'] = float(transition_seconds)
                record['segment_seconds'] = float(segment_seconds)
                record['evaluation_seconds'] = float(evaluation_seconds)
                record['epoch_seconds'] = float(time.perf_counter() - epoch_start_time)
            pd.DataFrame(history).to_csv(partial_history_csv, index=False)
            print(
                f"Epoch {epoch:4d} | multi_state_loss={record['loss']:.5f} | "
                f"train_rmse={record['train_mean_rmse']:.3f} | "
                f"val_rmse={record['val_mean_rmse']:.3f} | "
                f"heldout_transition_rmse={record['heldout_mean_rmse']:.3f} | "
                f"heldout_rolling30d_rmse={record['heldout_free_roll_rmse_30d']:.3f} | "
                f"segment_w={record['segment_rollout_weight_eff']:.4f}"
            )
        else:
            record = {
                'epoch': int(epoch),
                'loss': float(total_loss.detach().cpu().item()),
                'profile_supervision_scope': profile_supervision_scope,
                'train_supervision_pair_count': int(sum(
                    len(lake['pairs'][supervision_pair_key]) for lake in train_lakes
                )),
                'train_supervision_segment_sequence_count': int(sum(
                    len(lake['segment_rollout_sequences'][supervision_sequence_key]) for lake in train_lakes
                )),
                'transition_loss_weight': float(transition_loss_weight),
                'transition_loss_unweighted': float(
                    torch.stack(transition_lake_losses).mean().detach().cpu().item()
                ) if transition_lake_losses else np.nan,
                'transition_loss_weighted': float(
                    torch.stack(transition_weighted_lake_losses).mean().detach().cpu().item()
                ) if transition_weighted_lake_losses else np.nan,
                'profile_loss': _mean_detail(detail_records, 'profile_loss'),
                'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
                'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
                'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
                'warm_season_column_heat_content_quantile_low': float(warm_season_column_heat_content_quantile_low),
                'warm_season_column_heat_content_quantile_high': float(warm_season_column_heat_content_quantile_high),
                'warm_season_column_heat_content_min_gap_days': int(warm_season_column_heat_content_min_gap_days),
                'segment_rollout_loss': _mean_detail(segment_detail_records, 'segment_rollout_loss'),
                'segment_rollout_profile_loss': _mean_detail(segment_detail_records, 'segment_rollout_profile_loss'),
                'segment_rollout_horizon_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_horizon_weight_mean'),
                'segment_rollout_max_target_gap_days': _mean_detail(segment_detail_records, 'segment_rollout_max_target_gap_days'),
                'segment_rollout_lst_loss': _mean_detail(segment_detail_records, 'segment_rollout_lst_loss'),
                'segment_rollout_lst_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_lst_supervision_count'),
                'segment_rollout_lst_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_lst_weight_mean'),
                'segment_rollout_residual_smooth_loss': _mean_detail(segment_detail_records, 'segment_rollout_residual_smooth_loss'),
                'segment_rollout_daily_tendency_loss': _mean_detail(segment_detail_records, 'segment_rollout_daily_tendency_loss'),
                'segment_rollout_residual_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_residual_regularization_loss'),
                'segment_rollout_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_supervision_count'),
                'segment_rollout_sequence_count': _mean_detail(segment_detail_records, 'segment_rollout_sequence_count'),
                'segment_rollout_weight_eff': float(segment_weight_eff),
                'segment_rollout_active_days': int(active_segment_days),
                'teacher_forcing_probability': float(teacher_forcing_probability),
                'state_noise_weight': float(state_noise_weight),
                'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
                'lst_feature_dropout_applied_mean': _mean_detail(detail_records, 'lst_feature_dropout_applied_mean'),
                'residual_time_smooth_weight': float(residual_time_smooth_weight),
                'daily_tendency_loss': _mean_detail(detail_records, 'daily_tendency_loss'),
                'residual_regularization_loss': _mean_detail(detail_records, 'residual_regularization_loss'),
                'physical_scale_reg_loss': _mean_detail(detail_records, 'physical_scale_reg_loss'),
                'physical_scale_smooth_loss': _mean_detail(detail_records, 'physical_scale_smooth_loss'),
                'kd_prior_regularization_loss': _mean_detail(detail_records, 'kd_prior_regularization_loss'),
                'kd_prior_regularization_weighted_loss': _mean_detail(detail_records, 'kd_prior_regularization_weighted_loss'),
                'adaptive_parameter_regularization_loss': _mean_detail(detail_records, 'adaptive_parameter_regularization_loss'),
                'heat_content_transition_loss': _mean_detail(detail_records, 'heat_content_transition_loss'),
                'heat_content_transition_weighted_loss': _mean_detail(detail_records, 'heat_content_transition_weighted_loss'),
                'heat_content_transition_effective_weight_mean': _mean_detail(detail_records, 'heat_content_transition_effective_weight_mean'),
                'heat_content_transition_effective_weight_min': _min_detail(detail_records, 'heat_content_transition_effective_weight_min'),
                'heat_content_transition_effective_weight_max': _max_detail(detail_records, 'heat_content_transition_effective_weight_max'),
                'shortwave_scale_mean': detail_mean(detail_records, 'shortwave_scale_mean'),
                'shortwave_scale_min': detail_mean(detail_records, 'shortwave_scale_min'),
                'shortwave_scale_max': detail_mean(detail_records, 'shortwave_scale_max'),
                'cooling_scale_mean': detail_mean(detail_records, 'cooling_scale_mean'),
                'cooling_scale_min': detail_mean(detail_records, 'cooling_scale_min'),
                'cooling_scale_max': detail_mean(detail_records, 'cooling_scale_max'),
                'cooling_scale_effective_mean': detail_mean(detail_records, 'cooling_scale_effective_mean'),
                'surface_flux_bias_mean_wm2': detail_mean(detail_records, 'surface_flux_bias_mean_wm2'),
                'open_water_sensible_heat_mean_wm2': detail_mean(detail_records, 'open_water_sensible_heat_mean_wm2'),
                'open_water_latent_heat_mean_wm2': detail_mean(detail_records, 'open_water_latent_heat_mean_wm2'),
                'temperature_floor_heat_injection_mean_wm2': detail_mean(detail_records, 'temperature_floor_heat_injection_mean_wm2'),
                'freezing_storage_mean_j_m2': detail_mean(detail_records, 'freezing_storage_mean_j_m2'),
                'freezing_storage_ice_mean_j_m2': detail_mean(detail_records, 'freezing_storage_ice_mean_j_m2'),
                'freezing_storage_surface_fraction_mean': detail_mean(detail_records, 'freezing_storage_surface_fraction_mean'),
                'freezing_storage_deep_fraction_mean': detail_mean(detail_records, 'freezing_storage_deep_fraction_mean'),
                'freezing_storage_change_mean_wm2': detail_mean(detail_records, 'freezing_storage_change_mean_wm2'),
                'effective_heat_tendency_mean_wm2': detail_mean(detail_records, 'effective_heat_tendency_mean_wm2'),
                'advective_heat_source_c_per_day_mean': detail_mean(detail_records, 'advective_heat_source_c_per_day_mean'),
                'advective_heat_source_c_per_day_max': detail_mean(detail_records, 'advective_heat_source_c_per_day_max'),
                'advective_exchange_fraction_per_day': detail_mean(detail_records, 'advective_exchange_fraction_per_day'),
                'advective_heat_source_active_mean': detail_mean(detail_records, 'advective_heat_source_active_mean'),
                'background_nn_kz_mean': detail_mean(detail_records, 'background_nn_kz_mean'),
                'background_nn_kz_deep_mean': detail_mean(detail_records, 'background_nn_kz_deep_mean'),
                'turbulent_nn_kz_mean': detail_mean(detail_records, 'turbulent_nn_kz_mean'),
                'turbulent_nn_kz_deep_mean': detail_mean(detail_records, 'turbulent_nn_kz_deep_mean'),
                'gated_turbulent_nn_kz_mean': detail_mean(detail_records, 'gated_turbulent_nn_kz_mean'),
                'gated_turbulent_nn_kz_deep_mean': detail_mean(detail_records, 'gated_turbulent_nn_kz_deep_mean'),
                'kd_base_mean': detail_mean(detail_records, 'kd_base_mean'),
                'nn_kd_multiplier_mean': detail_mean(detail_records, 'nn_kd_multiplier_mean'),
                'kd_prior_regularization_loss_mean': detail_mean(detail_records, 'kd_prior_regularization_loss_mean'),
                'adaptive_wind_kz_scale_mean': detail_mean(detail_records, 'adaptive_wind_kz_scale_mean'),
                'adaptive_turbulent_flux_blend_alpha_mean': detail_mean(detail_records, 'adaptive_turbulent_flux_blend_alpha_mean'),
                'adaptive_kd_multiplier_mean': detail_mean(detail_records, 'adaptive_kd_multiplier_mean'),
                'adaptive_turbulent_exchange_scale_mean': detail_mean(detail_records, 'adaptive_turbulent_exchange_scale_mean'),
                'adaptive_convective_mixing_scale_mean': detail_mean(detail_records, 'adaptive_convective_mixing_scale_mean'),
                'adaptive_ice_shortwave_scale_mean': detail_mean(detail_records, 'adaptive_ice_shortwave_scale_mean'),
                'lake_shape_wind_factor_mean': detail_mean(detail_records, 'lake_shape_wind_factor_mean'),
                'lake_shape_decay_depth_mean_m': detail_mean(detail_records, 'lake_shape_decay_depth_mean_m'),
                'stratification_mixing_gate_mean': detail_mean(detail_records, 'stratification_mixing_gate_mean'),
                'stratification_mixing_gate_min': detail_mean(detail_records, 'stratification_mixing_gate_min'),
                'stratification_mixing_gate_deep_mean': detail_mean(
                    detail_records,
                    'stratification_mixing_gate_deep_mean',
                ),
                'segment_rollout_physical_scale_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_physical_scale_regularization_loss'),
                'segment_rollout_physical_scale_smoothness_loss': _mean_detail(segment_detail_records, 'segment_rollout_physical_scale_smoothness_loss'),
                'segment_rollout_kd_prior_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_prior_regularization_loss'),
                'segment_rollout_kd_prior_regularization_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_prior_regularization_weighted_loss'),
                'segment_rollout_adaptive_parameter_regularization_loss': _mean_detail(segment_detail_records, 'segment_rollout_adaptive_parameter_regularization_loss'),
                'segment_rollout_heat_content_transition_loss': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_loss'),
                'segment_rollout_heat_content_transition_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_weighted_loss'),
                'segment_rollout_heat_content_transition_effective_weight_mean': _mean_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_mean'),
                'segment_rollout_heat_content_transition_effective_weight_min': _min_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_min'),
                'segment_rollout_heat_content_transition_effective_weight_max': _max_detail(segment_detail_records, 'segment_rollout_heat_content_transition_effective_weight_max'),
                'segment_rollout_warm_column_heat_content_loss': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_loss'),
                'segment_rollout_warm_column_heat_content_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_weighted_loss'),
                'segment_rollout_warm_column_heat_content_supervision_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_supervision_count'),
                'segment_rollout_warm_column_heat_content_warm_factor_mean': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_warm_factor_mean'),
                'segment_rollout_warm_column_heat_content_error_c_mean': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_error_c_mean'),
                'segment_rollout_warm_column_heat_content_horizon14_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon14_count'),
                'segment_rollout_warm_column_heat_content_horizon30_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon30_count'),
                'segment_rollout_warm_column_heat_content_horizon60_count': _mean_detail(segment_detail_records, 'segment_rollout_warm_column_heat_content_horizon60_count'),
                'energy_loss': _mean_detail(detail_records, 'energy_loss'),
                'physical_scale_regularization_weight': float(physical_scale_regularization_weight),
                'physical_scale_smoothness_weight': float(physical_scale_smoothness_weight),
                'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
                'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
                'heat_content_transition_weight': float(heat_content_transition_weight),
                'heat_content_transition_weight_base': float(heat_content_transition_weight),
                'heat_content_full_column_min_coverage': float(heat_content_full_column_min_coverage),
                'heat_content_transition_depth_factor': 1.0 if heat_content_transition_depth_factor else 0.0,
                'heat_content_transition_effective_max': float(heat_content_transition_effective_max),
                'hard_density_stability': hard_density_stability_mode,
                'hard_density_stability_active': bool(hard_density_stability_active),
                'turbulent_flux_mode': turbulent_flux_mode,
                'turbulent_flux_blend_alpha': float(turbulent_flux_blend_alpha),
                'lake_adaptive_params': lake_adaptive_params,
                'shape_aware_mixing': shape_aware_mixing,
                'shape_mixing_strength': float(shape_mixing_strength),
                'stratification_mixing_cap': stratification_mixing_cap,
                'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
                'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
                'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
                'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
                'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
                'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
                'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
                'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
                'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
                'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
                'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
                'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
                'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
                'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
                'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
                'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
                'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
                'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
                'freezing_energy_mode': freezing_energy_mode,
                'advective_heat_source_mode': advective_heat_source_mode,
                'checkpoint_every_epochs': int(checkpoint_every_epochs),
                'eval_every_epochs': int(eval_interval),
                'full_eval_every_epochs': int(full_eval_interval),
                'eval_mode': eval_mode,
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'history_diagnostic_enabled': bool(history_diagnostic_enabled),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                'transition_batch_size': int(transition_batch_size),
                'segment_rollout_batch_size': int(segment_rollout_batch_size),
                'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
                'train_diagnostic_mode': train_diagnostic_mode,
                'export_after_training': export_after_training,
                'export_max_depth_m': export_max_depth_m,
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                'rolling_horizon_eval_max_starts': int(rolling_horizon_eval_max_starts),
                'train_mean_rmse': np.nan,
                'val_mean_rmse': np.nan,
                'heldout_mean_rmse': np.nan,
                'heldout_transition_mean_rmse': np.nan,
                'train_persistence_mean_rmse': np.nan,
                'val_persistence_mean_rmse': np.nan,
                'heldout_persistence_mean_rmse': np.nan,
                'heldout_free_roll_mean_rmse': np.nan,
                'heldout_free_roll_mean_bias': np.nan,
                'heldout_observed_point_mean_rmse': np.nan,
                'heldout_observed_point_mean_mae': np.nan,
                'heldout_observed_point_mean_bias': np.nan,
                'heldout_observed_point_total_count': 0,
                'heldout_post_spinup_mean_rmse': np.nan,
                'heldout_post_spinup_mean_bias': np.nan,
            }
            for horizon in (1, 3, 7, 14, 30, 60):
                record[f'train_transition_rmse_{horizon}d'] = np.nan
                record[f'train_transition_count_{horizon}d'] = np.nan
                record[f'heldout_transition_rmse_{horizon}d'] = np.nan
                record[f'heldout_transition_count_{horizon}d'] = np.nan
                record[f'val_rolling_start_rmse_{horizon}d'] = np.nan
                record[f'val_rolling_start_bias_{horizon}d'] = np.nan
                record[f'val_rolling_start_count_{horizon}d'] = np.nan
                record[f'heldout_rolling_start_rmse_{horizon}d'] = np.nan
                record[f'heldout_rolling_start_bias_{horizon}d'] = np.nan
                record[f'heldout_rolling_start_count_{horizon}d'] = np.nan
                record[f'heldout_free_roll_rmse_{horizon}d'] = np.nan
                record[f'heldout_free_roll_bias_{horizon}d'] = np.nan
                record[f'heldout_free_roll_count_{horizon}d'] = np.nan
                record[f'heldout_initial_free_roll_rmse_{horizon}d'] = np.nan
                record[f'heldout_initial_free_roll_bias_{horizon}d'] = np.nan
                record[f'heldout_initial_free_roll_count_{horizon}d'] = np.nan
            for month in range(1, 13):
                record[f'heat_content_transition_season_factor_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.mean,
                )
                record[f'heat_content_transition_season_factor_min_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.min,
                )
                record[f'heat_content_transition_season_factor_max_{month:02d}'] = _lake_season_factor_stat(
                    train_lakes,
                    month,
                    np.max,
                )
            for lake in train_lakes:
                key = lake['lake_id']
                record[f'{key}_train_rmse'] = np.nan
                record[f'{key}_val_rmse'] = np.nan
                record[f'{key}_train_persistence_rmse'] = np.nan
                record[f'{key}_val_persistence_rmse'] = np.nan
                for horizon in (1, 3, 7, 14, 30, 60):
                    record[f'{key}_val_rolling_start_rmse_{horizon}d'] = np.nan
                    record[f'{key}_val_rolling_start_bias_{horizon}d'] = np.nan
                    record[f'{key}_val_rolling_start_count_{horizon}d'] = np.nan
            for lake in heldout_lakes:
                key = lake['lake_id']
                record[f'{key}_heldout_rmse'] = np.nan
                record[f'{key}_heldout_transition_rmse'] = np.nan
                record[f'{key}_heldout_persistence_rmse'] = np.nan
                record[f'{key}_heldout_free_roll_rmse'] = np.nan
                record[f'{key}_heldout_free_roll_mae'] = np.nan
                record[f'{key}_heldout_free_roll_bias'] = np.nan
                record[f'{key}_heldout_free_roll_profiles'] = 0
                record[f'{key}_heldout_observed_point_rmse'] = np.nan
                record[f'{key}_heldout_observed_point_mae'] = np.nan
                record[f'{key}_heldout_observed_point_bias'] = np.nan
                record[f'{key}_heldout_observed_point_count'] = 0
                record[f'{key}_heldout_observed_point_profiles'] = 0
                for observed_group in ('surface', 'mid', 'deep', 'winter', 'spring', 'summer', 'fall'):
                    record[f'{key}_heldout_observed_point_{observed_group}_rmse'] = np.nan
                    record[f'{key}_heldout_observed_point_{observed_group}_bias'] = np.nan
                    record[f'{key}_heldout_observed_point_{observed_group}_count'] = 0
                record[f'{key}_heldout_post_spinup_rmse'] = np.nan
                record[f'{key}_heldout_post_spinup_bias'] = np.nan
                record[f'{key}_heldout_spinup_days_used'] = np.nan
                for horizon in (1, 3, 7, 14, 30, 60):
                    record[f'{key}_heldout_initial_free_roll_rmse_{horizon}d'] = np.nan
                    record[f'{key}_heldout_initial_free_roll_bias_{horizon}d'] = np.nan
                    record[f'{key}_heldout_initial_free_roll_count_{horizon}d'] = np.nan
                    record[f'{key}_heldout_rolling_start_rmse_{horizon}d'] = np.nan
                    record[f'{key}_heldout_rolling_start_bias_{horizon}d'] = np.nan
                    record[f'{key}_heldout_rolling_start_count_{horizon}d'] = np.nan
                    record[f'{key}_heldout_free_roll_rmse_{horizon}d'] = np.nan
                    record[f'{key}_heldout_free_roll_bias_{horizon}d'] = np.nan
                    record[f'{key}_heldout_free_roll_count_{horizon}d'] = np.nan
            record['best_by_val_rolling_enabled'] = bool(best_val_rolling_enabled)
            record['best_by_val_rolling_score'] = np.nan
            record['best_by_val_rolling_best_score'] = (
                np.nan if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
            )
            record['best_by_val_rolling_checkpoint_updated'] = False
            record['best_by_val_rolling_skipped_reason'] = (
                'disabled: full_eval_every_epochs=0'
                if not best_val_rolling_enabled else 'skipped: no full eval this epoch'
            )
            record['best_by_val_rolling_checkpoint_path'] = (
                str(best_val_rolling_checkpoint_path) if best_val_rolling_checkpoint_path.exists() else ''
            )
            history.append(record)
            if profile_runtime:
                record['transition_seconds'] = float(transition_seconds)
                record['segment_seconds'] = float(segment_seconds)
                record['evaluation_seconds'] = 0.0
                record['epoch_seconds'] = float(time.perf_counter() - epoch_start_time)
            pd.DataFrame(history).to_csv(partial_history_csv, index=False)
            print(
                f"Epoch {epoch:4d} | multi_state_loss={record['loss']:.5f} | "
                "eval_mode=none | "
                f"segment_w={record['segment_rollout_weight_eff']:.4f}"
            )
        if profile_gpu:
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start_time
            gpu_record = {
                'epoch': int(epoch),
                'eval_mode': eval_mode,
                'transition_seconds': float(transition_seconds),
                'segment_seconds': float(segment_seconds),
                'evaluation_seconds': float(evaluation_seconds),
                'epoch_seconds': float(epoch_seconds),
            }
            gpu_record.update(_gpu_profile_snapshot(device))
            runtime_profile_records.append(gpu_record)
            pd.DataFrame(runtime_profile_records).to_csv(runtime_profile_csv, index=False)
        if checkpoint_this_epoch:
            _save_training_checkpoint(epoch)

    checkpoint_path = output_dir / 'global_state_forecaster_checkpoint.pt'
    torch.save(
        {
            'architecture': 'MultiLakeStateForecaster',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': int(epochs) - 1,
            'manifest': str(Path(manifest_path)),
            'task_mode': task_mode,
            'data_fill_mode': data_fill_mode,
            'profile_supervision_scope': profile_supervision_scope,
            'test_lake_id': test_lake_id or None,
            'test_lake_ids': list(test_lake_ids),
            'heldout_lake_groups': list(heldout_lake_groups),
            'lake_ids': [lake['lake_id'] for lake in lakes],
            'train_lake_ids': [lake['lake_id'] for lake in train_lakes],
            'heldout_lake_ids': [lake['lake_id'] for lake in heldout_lakes],
            'excluded_lake_ids': [lake['lake_id'] for lake in excluded_lakes],
            'lake_metadata_summary': {
                lake['lake_id']: _lake_metadata_summary_payload(lake)
                for lake in lakes
            },
            'static_feature_dim': int(static_dim),
            'static_feature_keys': list(STATIC_FEATURE_KEYS),
            'history_window_days': int(history_window_days),
            'depth_points': int(depth_points),
            'max_rollout_days': int(max_rollout_days),
            'residual_limit_c': float(residual_limit_c),
            'wind_kz_scale': float(wind_kz_scale),
            'autumn_convective_boost': float(autumn_convective_boost),
            'residual_regularization_weight': float(residual_regularization_weight),
            'daily_tendency_weight': float(daily_tendency_weight),
            'physical_scale_regularization_weight': float(physical_scale_regularization_weight),
            'physical_scale_smoothness_weight': float(physical_scale_smoothness_weight),
            'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
            'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
            'heat_content_transition_weight': float(heat_content_transition_weight),
            'heat_content_transition_weight_base': float(heat_content_transition_weight),
            'heat_content_full_column_min_coverage': float(heat_content_full_column_min_coverage),
            'heat_content_transition_season_mode': heat_content_transition_season_mode,
            'heat_content_transition_season_factors_override': heat_content_transition_season_factors_override_payload,
            'heat_content_transition_lake_configs': heat_content_transition_lake_configs,
            'heat_content_transition_depth_factor': 'on' if heat_content_transition_depth_factor else 'off',
            'heat_content_transition_effective_max': float(heat_content_transition_effective_max),
            'transition_loss_weight': float(transition_loss_weight),
            'hard_density_stability': hard_density_stability_mode,
            'hard_density_stability_active': bool(hard_density_stability_active),
            'turbulent_flux_mode': turbulent_flux_mode,
            'turbulent_flux_blend_alpha': float(turbulent_flux_blend_alpha),
            'lake_adaptive_params': lake_adaptive_params,
            'shape_aware_mixing': shape_aware_mixing,
            'shape_mixing_strength': float(shape_mixing_strength),
            'stratification_mixing_cap': stratification_mixing_cap,
            'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
            'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
            'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
            'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
            'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
            'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
            'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
            'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
            'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
            'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
            'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
            'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
            'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
            'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
            'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
            'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
            'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
            'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
            'freezing_energy_mode': freezing_energy_mode,
            'advective_heat_source_mode': advective_heat_source_mode,
            'checkpoint_every_epochs': int(checkpoint_every_epochs),
            'eval_every_epochs': int(eval_interval),
            'full_eval_every_epochs': int(full_eval_interval),
            'profile_runtime': bool(profile_runtime),
            'profile_gpu': bool(profile_gpu),
            'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
            'torch_tf32': torch_tf32,
            'torch_matmul_precision': torch_matmul_precision,
            'transition_batch_size': int(transition_batch_size),
            'segment_rollout_batch_size': int(segment_rollout_batch_size),
            'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
            'train_diagnostic_mode': train_diagnostic_mode,
            'export_after_training': export_after_training,
            'export_max_depth_m': export_max_depth_m,
            'cross_lake_batch_mode': cross_lake_batch_mode,
            'cross_lake_batch_size': int(cross_lake_batch_size),
            'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
            'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
            'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
            'segment_rollout_max_days': int(segment_rollout_max_days),
            'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
            'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
            'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
            'warm_season_column_heat_content_quantile_low': float(
                warm_season_column_heat_content_quantile_low
            ),
            'warm_season_column_heat_content_quantile_high': float(
                warm_season_column_heat_content_quantile_high
            ),
            'warm_season_column_heat_content_min_gap_days': int(
                warm_season_column_heat_content_min_gap_days
            ),
            'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
            'teacher_forcing_start': float(teacher_forcing_start),
            'teacher_forcing_end': float(teacher_forcing_end),
            'state_noise_weight': float(state_noise_weight),
            'residual_time_smooth_weight': float(residual_time_smooth_weight),
            'rolling_horizon_eval_max_starts': int(rolling_horizon_eval_max_starts),
            'init_mode': init_mode,
            'spinup_days': int(spinup_days),
            'spinup_lst_assimilation_strength': float(spinup_lst_assimilation_strength),
            'spinup_lst_assimilation_decay_depth_m': float(spinup_lst_assimilation_decay_depth_m),
            'spinup_lst_assimilation_max_increment_c': float(spinup_lst_assimilation_max_increment_c),
            'rollout_start_date': rollout_start_date,
            'rollout_mode': rollout_mode,
            'rollout_reinit_scope': rollout_reinit_scope,
            'best_by_val_rolling_enabled': bool(best_val_rolling_enabled),
            'best_by_val_rolling_checkpoint_path': str(best_val_rolling_checkpoint_path),
            'best_by_val_rolling_metrics_path': str(best_val_rolling_metrics_path),
            'best_by_val_rolling_score': (
                None if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
            ),
            'best_by_val_rolling_epoch': best_val_rolling_epoch,
            'training_history': history,
        },
        checkpoint_path,
    )
    history_csv = output_dir / 'global_state_forecaster_training_history.csv'
    pd.DataFrame(history).to_csv(history_csv, index=False)
    split_summary = output_dir / 'global_state_forecaster_split_summary.json'
    split_summary_payload = {
        lake['lake_id']: {
            'train_pairs': len(lake['pairs']['train']),
            'val_pairs': len(lake['pairs']['val']),
            'all_pairs': len(lake['pairs']['all']),
            'train_segment_rollout_sequences': len(lake['segment_rollout_sequences']['train']),
            'all_segment_rollout_sequences': len(lake['segment_rollout_sequences']['all']),
            'supervision_pairs': len(lake['pairs'][profile_supervision_scope]),
            'supervision_segment_rollout_sequences': len(
                lake['segment_rollout_sequences'][profile_supervision_scope]
            ),
            'is_heldout_test_lake': bool(lake['lake_id'] in set(test_lake_ids)),
            'is_excluded_by_heldout_group': bool(lake['lake_id'] in {item['lake_id'] for item in excluded_lakes}),
            'heat_content_transition': _heat_content_transition_lake_config_payload(lake),
            'warm_season_column_heat_content': _warm_column_heat_content_lake_config_payload(lake),
            'metadata': _lake_metadata_summary_payload(lake),
        }
        for lake in lakes
    }
    split_summary_payload['_config'] = {
        'test_lake_id': test_lake_id or None,
        'test_lake_ids': list(test_lake_ids),
        'heldout_lake_groups': list(heldout_lake_groups),
        'lake_ids': [lake['lake_id'] for lake in lakes],
        'train_lake_ids': [lake['lake_id'] for lake in train_lakes],
        'heldout_lake_ids': [lake['lake_id'] for lake in heldout_lakes],
        'excluded_lake_ids': [lake['lake_id'] for lake in excluded_lakes],
        'profile_supervision_scope': profile_supervision_scope,
        'residual_limit_c': float(residual_limit_c),
        'wind_kz_scale': float(wind_kz_scale),
        'autumn_convective_boost': float(autumn_convective_boost),
        'physical_scale_mode': 'learned_lake_season_forcing',
        'physical_scale_regularization_weight': float(physical_scale_regularization_weight),
        'physical_scale_smoothness_weight': float(physical_scale_smoothness_weight),
        'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
        'adaptive_parameter_regularization_weight': float(adaptive_parameter_regularization_weight),
        'heat_content_transition_weight': float(heat_content_transition_weight),
        'heat_content_transition_weight_base': float(heat_content_transition_weight),
        'heat_content_full_column_min_coverage': float(heat_content_full_column_min_coverage),
        'heat_content_transition_season_mode': heat_content_transition_season_mode,
        'heat_content_transition_season_factors_override': heat_content_transition_season_factors_override_payload,
        'heat_content_transition_lake_configs': heat_content_transition_lake_configs,
        'heat_content_transition_depth_factor': 'on' if heat_content_transition_depth_factor else 'off',
        'heat_content_transition_effective_max': float(heat_content_transition_effective_max),
        'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
        'warm_season_column_heat_content_quantile_low': float(
            warm_season_column_heat_content_quantile_low
        ),
        'warm_season_column_heat_content_quantile_high': float(
            warm_season_column_heat_content_quantile_high
        ),
        'warm_season_column_heat_content_min_gap_days': int(
            warm_season_column_heat_content_min_gap_days
        ),
        'warm_season_column_heat_content_lake_configs': warm_column_heat_content_lake_configs,
        'transition_loss_weight': float(transition_loss_weight),
        'hard_density_stability': hard_density_stability_mode,
        'hard_density_stability_active': bool(hard_density_stability_active),
        'turbulent_flux_mode': turbulent_flux_mode,
        'turbulent_flux_blend_alpha': float(turbulent_flux_blend_alpha),
        'lake_adaptive_params': lake_adaptive_params,
        'shape_aware_mixing': shape_aware_mixing,
        'shape_mixing_strength': float(shape_mixing_strength),
        'stratification_mixing_cap': stratification_mixing_cap,
        'stratification_mixing_cap_strength': float(stratification_mixing_cap_strength),
        'lake_adaptive_hidden_dim': int(lake_adaptive_hidden_dim),
        'lake_adaptive_init_spread': float(lake_adaptive_init_spread),
        'lake_adaptive_temporal_mode': lake_adaptive_temporal_mode,
        'lake_adaptive_temporal_init_spread': float(lake_adaptive_temporal_init_spread),
        'lake_adaptive_temporal_scale': float(lake_adaptive_temporal_scale),
        'adaptive_wind_kz_min': float(adaptive_wind_kz_min),
        'adaptive_wind_kz_max': float(adaptive_wind_kz_max),
        'adaptive_blend_alpha_min': float(adaptive_blend_alpha_min),
        'adaptive_blend_alpha_max': float(adaptive_blend_alpha_max),
        'adaptive_kd_multiplier_min': float(adaptive_kd_multiplier_min),
        'adaptive_kd_multiplier_max': float(adaptive_kd_multiplier_max),
        'adaptive_turbulent_exchange_scale_min': float(adaptive_turbulent_exchange_scale_min),
        'adaptive_turbulent_exchange_scale_max': float(adaptive_turbulent_exchange_scale_max),
        'adaptive_convective_mixing_scale_min': float(adaptive_convective_mixing_scale_min),
        'adaptive_convective_mixing_scale_max': float(adaptive_convective_mixing_scale_max),
        'adaptive_ice_shortwave_scale_min': float(adaptive_ice_shortwave_scale_min),
        'adaptive_ice_shortwave_scale_max': float(adaptive_ice_shortwave_scale_max),
        'freezing_energy_mode': freezing_energy_mode,
        'advective_heat_source_mode': advective_heat_source_mode,
        'checkpoint_every_epochs': int(checkpoint_every_epochs),
        'eval_every_epochs': int(eval_interval),
        'full_eval_every_epochs': int(full_eval_interval),
        'best_by_val_rolling_enabled': bool(best_val_rolling_enabled),
        'best_by_val_rolling_checkpoint_path': str(best_val_rolling_checkpoint_path),
        'best_by_val_rolling_metrics_path': str(best_val_rolling_metrics_path),
        'best_by_val_rolling_score': (
            None if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
        ),
        'best_by_val_rolling_epoch': best_val_rolling_epoch,
        'profile_runtime': bool(profile_runtime),
        'profile_gpu': bool(profile_gpu),
        'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
        'torch_tf32': torch_tf32,
        'torch_matmul_precision': torch_matmul_precision,
        'transition_batch_size': int(transition_batch_size),
        'segment_rollout_batch_size': int(segment_rollout_batch_size),
        'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
        'train_diagnostic_mode': train_diagnostic_mode,
        'export_after_training': export_after_training,
        'export_max_depth_m': export_max_depth_m,
        'cross_lake_batch_mode': cross_lake_batch_mode,
        'cross_lake_batch_size': int(cross_lake_batch_size),
        'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
        'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
        'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
        'segment_rollout_max_days': int(segment_rollout_max_days),
        'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
        'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
        'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
        'task_mode': task_mode,
        'data_fill_mode': data_fill_mode,
        'rollout_mode': rollout_mode,
        'rollout_reinit_scope': rollout_reinit_scope,
        'static_feature_dim': int(static_dim),
        'static_feature_keys': list(STATIC_FEATURE_KEYS),
    }
    split_summary.write_text(
        json.dumps(split_summary_payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    lake_adaptive_summary_csv = _write_lake_adaptive_parameter_summary(
        model,
        lakes,
        train_lakes,
        heldout_lakes,
        excluded_lakes,
        output_dir,
    )
    heldout_exports = []
    export_checkpoint_path = checkpoint_path
    if heldout_lakes and export_after_training == 'on':
        if best_val_rolling_checkpoint_path.exists():
            best_checkpoint = torch.load(best_val_rolling_checkpoint_path, map_location=device)
            best_state_dict = best_checkpoint.get('model_state_dict', best_checkpoint)
            best_state_dict = _filter_state_forecaster_state_dict_for_load(model, best_state_dict)
            model.load_state_dict(best_state_dict, strict=False)
            export_checkpoint_path = best_val_rolling_checkpoint_path
        for lake in heldout_lakes:
            export_info = export_heldout_state_forecast(
                model,
                lake,
                output_dir,
                task_mode=task_mode,
                init_mode=init_mode,
                spinup_days=spinup_days,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_start_date=rollout_start_date,
                rollout_mode=rollout_mode,
                rollout_reinit_scope=rollout_reinit_scope,
                export_max_depth_m=export_max_depth_m,
                hard_density_stability=hard_density_stability_active,
                hard_density_stability_mode=hard_density_stability_mode,
            )
            heldout_exports.append(export_info)
            print(
                f"Held-out reconstruction export for {lake['lake_id']}: "
                f"{export_info['heatmap_path']} | score={export_info['scorecard_status']}"
            )
    elif heldout_lakes:
        print(
            "Skipped held-out export after training (export_after_training=off). "
            "Use --export-only with --checkpoint-path to generate CSV/PNG/scorecard outputs."
        )
    return {
        'model': model,
        'checkpoint_path': checkpoint_path,
        'export_checkpoint_path': export_checkpoint_path,
        'best_by_val_rolling_checkpoint_path': best_val_rolling_checkpoint_path,
        'best_by_val_rolling_metrics_path': best_val_rolling_metrics_path,
        'best_by_val_rolling_score': (
            None if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
        ),
        'best_by_val_rolling_epoch': best_val_rolling_epoch,
        'history_csv': history_csv,
        'runtime_profile_csv': runtime_profile_csv if profile_gpu else None,
        'split_summary': split_summary,
        'lake_adaptive_parameter_summary_csv': lake_adaptive_summary_csv,
        'heldout_exports': heldout_exports,
        'lakes': lakes,
        'history': history,
        'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
        'export_max_depth_m': export_max_depth_m,
        'warm_season_column_heat_content_weight': float(warm_season_column_heat_content_weight),
        'warm_season_column_heat_content_quantile_low': float(
            warm_season_column_heat_content_quantile_low
        ),
        'warm_season_column_heat_content_quantile_high': float(
            warm_season_column_heat_content_quantile_high
        ),
        'warm_season_column_heat_content_min_gap_days': int(
            warm_season_column_heat_content_min_gap_days
        ),
        'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
        'kd_prior_regularization_weight': float(kd_prior_regularization_weight),
        'advective_heat_source_mode': advective_heat_source_mode,
        'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
        'torch_tf32': torch_tf32,
        'torch_matmul_precision': torch_matmul_precision,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a multi-lake reconstruction-state LakePINN.')
    parser.add_argument('--manifest', required=True, help='JSON manifest listing lake forcing/LST/profile/metadata inputs.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--depth-points', type=int, default=40)
    parser.add_argument('--max-rollout-days', type=int, default=45)
    parser.add_argument('--history-window-days', type=int, default=30)
    parser.add_argument('--split-mode', choices=['time_blocked', 'seasonal_blocked', 'depth_interleaved'], default='time_blocked')
    parser.add_argument(
        '--profile-supervision-scope',
        choices=['train', 'all'],
        default='train',
        help='Profiles used for transition/segment supervision on training lake-years. all uses train+val dates.',
    )
    parser.add_argument('--test-lake-id', default=None)
    parser.add_argument(
        '--test-lake-ids',
        default=None,
        help='Comma-separated held-out lake IDs to export/evaluate from one shared multi-lake model.',
    )
    parser.add_argument(
        '--heldout-lake-groups',
        default=None,
        help='Comma-separated lake_group values to exclude entirely from training. Defaults to groups of test-lake-ids.',
    )
    parser.add_argument('--residual-limit-c', type=float, default=0.50)
    parser.add_argument(
        '--wind-kz-scale',
        type=float,
        default=1.0,
        help='Multiplier for wind-driven Kz before stability suppression.',
    )
    parser.add_argument(
        '--autumn-convective-boost',
        type=float,
        default=1.0,
        help='Multiplier for unstable-density convective Kz boost.',
    )
    parser.add_argument(
        '--shape-aware-mixing',
        choices=['on', 'off'],
        default='on',
        help='Use lake area/fetch/basin-shape metadata to scale wind-driven Kz and decay depth.',
    )
    parser.add_argument(
        '--shape-mixing-strength',
        type=float,
        default=0.35,
        help='Strength of lake-shape-aware wind mixing modulation. Use 0 for neutral behavior.',
    )
    parser.add_argument(
        '--stratification-mixing-cap',
        choices=['on', 'off'],
        default='on',
        help='Apply an additional strong-stratification gate to wind-driven Kz below the surface.',
    )
    parser.add_argument(
        '--stratification-mixing-cap-strength',
        type=float,
        default=1.0,
        help='Strength of the stratification-dependent wind mixing cap. Use 0 for neutral behavior.',
    )
    parser.add_argument(
        '--residual-regularization-weight',
        type=float,
        default=0.02,
        help='Weight for penalizing daily learned residual corrections during transition training.',
    )
    parser.add_argument(
        '--daily-tendency-weight',
        type=float,
        default=0.02,
        help='Weight for layered daily temperature jump penalties during transition training.',
    )
    parser.add_argument(
        '--physical-scale-regularization-weight',
        type=float,
        default=0.01,
        help='Weight for keeping learned physical scales near 1.0.',
    )
    parser.add_argument(
        '--physical-scale-smoothness-weight',
        type=float,
        default=0.005,
        help='Weight for penalizing day-to-day jumps in learned physical scales.',
    )
    parser.add_argument(
        '--kd-prior-regularization-weight',
        type=float,
        default=0.001,
        help='Explicit weight for keeping neural Kd multiplier near the optical Kd/Secchi prior. 0 disables.',
    )
    parser.add_argument(
        '--heat-content-transition-weight',
        type=float,
        default=DEFAULT_HEAT_CONTENT_TRANSITION_WEIGHT,
        help='Weight for matching observed and predicted heat-content tendency between profile dates.',
    )
    parser.add_argument(
        '--heat-content-full-column-min-coverage',
        type=float,
        default=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
        help='Minimum hypsometry-weighted observed coverage needed to use full-column heat-content loss.',
    )
    parser.add_argument(
        '--heat-content-transition-season-mode',
        choices=['auto', 'northern', 'southern', 'tropical', 'manual'],
        default='auto',
        help='Latitude-aware season factor mode for heat-content transition loss.',
    )
    parser.add_argument(
        '--heat-content-transition-season-factors',
        default=None,
        help='Manual month:factor heat-content multipliers; overrides season mode when provided.',
    )
    parser.add_argument(
        '--heat-content-transition-depth-factor',
        choices=['on', 'off'],
        default='on',
        help='Enable max-depth multiplier for heat-content transition loss.',
    )
    parser.add_argument(
        '--heat-content-transition-effective-max',
        type=float,
        default=DEFAULT_HEAT_CONTENT_TRANSITION_EFFECTIVE_MAX,
        help='Maximum dynamic heat-content transition effective weight.',
    )
    parser.add_argument(
        '--transition-loss-weight',
        type=float,
        default=1.0,
        help='Outer weight for short transition-pair loss before adding segment rollout loss.',
    )
    parser.add_argument(
        '--hard-density-stability',
        choices=['auto', 'on', 'off'],
        default='auto',
        help="Hard convective adjustment switch. auto enables it for analysis/reconstruction only.",
    )
    parser.add_argument(
        '--turbulent-flux-mode',
        choices=['bulk', 'provided', 'blend'],
        default='bulk',
        help='How to compute open-water sensible/latent heat flux. bulk closes the loop on predicted surface temperature.',
    )
    parser.add_argument(
        '--turbulent-flux-blend-alpha',
        type=float,
        default=0.3,
        help='For --turbulent-flux-mode blend, fraction of provided turbulent flux; 0 uses bulk, 1 uses provided.',
    )
    parser.add_argument(
        '--lake-adaptive-params',
        default='off',
        help=(
            'Comma-separated metadata-conditioned physical parameters: off, kz, flux, kd, '
            'exchange, convective, ice, both, all. Default preserves fixed global values.'
        ),
    )
    parser.add_argument(
        '--lake-adaptive-hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension for the metadata-conditioned lake-adaptive parameter head.',
    )
    parser.add_argument(
        '--lake-adaptive-init-spread',
        type=float,
        default=0.02,
        help='Initial final-layer weight std for small metadata-conditioned adaptive differences.',
    )
    parser.add_argument(
        '--lake-adaptive-temporal-mode',
        choices=['off', 'on', 'seasonal', 'forcing', 'seasonal_forcing'],
        default='off',
        help='Let lake-adaptive parameters vary by day using seasonal and forcing context.',
    )
    parser.add_argument(
        '--lake-adaptive-temporal-init-spread',
        type=float,
        default=0.005,
        help='Initial final-layer weight std for temporal adaptive deltas.',
    )
    parser.add_argument(
        '--lake-adaptive-temporal-scale',
        type=float,
        default=0.25,
        help='Maximum tanh-scaled temporal delta size before adaptive bounds are applied.',
    )
    parser.add_argument(
        '--adaptive-wind-kz-min',
        type=float,
        default=0.4,
        help='Lower bound for metadata-conditioned wind_kz_scale.',
    )
    parser.add_argument(
        '--adaptive-wind-kz-max',
        type=float,
        default=3.0,
        help='Upper bound for metadata-conditioned wind_kz_scale.',
    )
    parser.add_argument(
        '--adaptive-blend-alpha-min',
        type=float,
        default=0.0,
        help='Lower bound for metadata-conditioned turbulent flux blend alpha.',
    )
    parser.add_argument(
        '--adaptive-blend-alpha-max',
        type=float,
        default=0.6,
        help='Upper bound for metadata-conditioned turbulent flux blend alpha.',
    )
    parser.add_argument(
        '--adaptive-kd-multiplier-min',
        type=float,
        default=0.4,
        help='Lower bound for metadata-conditioned shortwave/Kd multiplier.',
    )
    parser.add_argument(
        '--adaptive-kd-multiplier-max',
        type=float,
        default=2.0,
        help='Upper bound for metadata-conditioned shortwave/Kd multiplier.',
    )
    parser.add_argument(
        '--adaptive-turbulent-exchange-scale-min',
        type=float,
        default=0.5,
        help='Lower bound for metadata-conditioned bulk turbulent exchange scale.',
    )
    parser.add_argument(
        '--adaptive-turbulent-exchange-scale-max',
        type=float,
        default=1.8,
        help='Upper bound for metadata-conditioned bulk turbulent exchange scale.',
    )
    parser.add_argument(
        '--adaptive-convective-mixing-scale-min',
        type=float,
        default=0.3,
        help='Lower bound for metadata-conditioned convective/overturn mixing scale.',
    )
    parser.add_argument(
        '--adaptive-convective-mixing-scale-max',
        type=float,
        default=2.5,
        help='Upper bound for metadata-conditioned convective/overturn mixing scale.',
    )
    parser.add_argument(
        '--adaptive-ice-shortwave-scale-min',
        type=float,
        default=0.4,
        help='Lower bound for metadata-conditioned ice/snow shortwave attenuation scale.',
    )
    parser.add_argument(
        '--adaptive-ice-shortwave-scale-max',
        type=float,
        default=1.8,
        help='Upper bound for metadata-conditioned ice/snow shortwave attenuation scale.',
    )
    parser.add_argument(
        '--adaptive-parameter-regularization-weight',
        type=float,
        default=0.01,
        help='Penalty weight for keeping metadata-conditioned physical parameters near CLI base values.',
    )
    parser.add_argument(
        '--freezing-energy-mode',
        choices=['latent_reservoir', 'clamp'],
        default='latent_reservoir',
        help='How sub-0C solver states are handled. latent_reservoir conserves cold content; clamp preserves legacy behavior.',
    )
    parser.add_argument(
        '--advective-heat-source-mode',
        choices=['off', 'reservoir_simple'],
        default='reservoir_simple',
        help='Add a simple positive-net-inflow heat source for reservoirs. off disables it.',
    )
    parser.add_argument(
        '--segment-rollout-start-epoch',
        type=int,
        default=30,
        help='Epoch where segment rollout training starts. Earlier epochs optimize transition losses only.',
    )
    parser.add_argument(
        '--segment-rollout-ramp-epochs',
        type=int,
        default=30,
        help='Number of epochs used to ramp the segment rollout loss weight and horizon.',
    )
    parser.add_argument(
        '--segment-rollout-samples-per-lake',
        type=int,
        default=12,
        help='Maximum segment rollout start profiles sampled per lake per epoch. Use 0 to use all.',
    )
    parser.add_argument(
        '--segment-rollout-loss-weight',
        type=float,
        default=0.05,
        help='Target weight for continuous segment rollout training loss. Must be > 0.0.',
    )
    parser.add_argument(
        '--segment-rollout-lst-surface-weight',
        type=float,
        default=0.01,
        help='Weak open-water LSWT surface supervision weight used only inside segment rollout loss.',
    )
    parser.add_argument(
        '--segment-rollout-max-days',
        type=int,
        default=None,
        help='Maximum scheduled segment rollout horizon in days; defaults to max-rollout-days.',
    )
    parser.add_argument(
        '--warm-season-column-heat-content-weight',
        type=float,
        default=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
        help='Segment-rollout soft loss weight for warm-season column-mean heat-content errors. Use 0 to disable.',
    )
    parser.add_argument(
        '--warm-season-column-heat-content-quantile-low',
        type=float,
        default=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
        help='Lake-internal lower quantile used as the warm-column heat-content ramp start.',
    )
    parser.add_argument(
        '--warm-season-column-heat-content-quantile-high',
        type=float,
        default=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
        help='Lake-internal upper quantile where the warm-column heat-content ramp reaches full strength.',
    )
    parser.add_argument(
        '--warm-season-column-heat-content-min-gap-days',
        type=int,
        default=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
        help='Minimum segment target gap in days before warm-column heat-content loss is enabled.',
    )
    parser.add_argument(
        '--lst-feature-dropout-probability',
        type=float,
        default=0.20,
        help='Training-only probability of masking LST forcing features for closed-loop robustness.',
    )
    parser.add_argument(
        '--teacher-forcing-start',
        type=float,
        default=0.7,
        help='Scheduled sampling probability at segment rollout warmup start.',
    )
    parser.add_argument(
        '--teacher-forcing-end',
        type=float,
        default=0.0,
        help='Scheduled sampling probability after segment rollout ramp.',
    )
    parser.add_argument(
        '--state-noise-weight',
        type=float,
        default=1.0,
        help='Multiplier for small training-only state perturbations inside segment rollout.',
    )
    parser.add_argument(
        '--residual-time-smooth-weight',
        type=float,
        default=0.01,
        help='Weight for residual vertical/time smoothness inside segment rollout loss.',
    )
    parser.add_argument(
        '--rolling-horizon-eval-max-starts',
        type=int,
        default=40,
        help='Maximum observed start profiles sampled for rolling-start horizon metrics. Use 0 or negative for all starts.',
    )
    parser.add_argument(
        '--checkpoint-every-epochs',
        type=int,
        default=5,
        help='Save resumable training checkpoint every N epochs. Default: 5; use 0 to disable periodic checkpoints.',
    )
    parser.add_argument(
        '--resume-checkpoint',
        default=None,
        help='Resumable checkpoint produced by --checkpoint-every-epochs.',
    )
    parser.add_argument(
        '--eval-every-epochs',
        type=int,
        default=None,
        help='Run lightweight train/val/heldout transition and rolling metrics every N epochs. Default: 50.',
    )
    parser.add_argument(
        '--full-eval-every-epochs',
        type=int,
        default=None,
        help='Run full held-out free-roll and rolling-start evaluation every N epochs. Default: 60; use 0 to disable.',
    )
    parser.add_argument(
        '--profile-runtime',
        nargs='?',
        const='on',
        default='on',
        choices=['on', 'off'],
        help='Record transition/segment/evaluation/epoch wall-clock seconds in training history. Default: on.',
    )
    parser.add_argument(
        '--profile-gpu',
        action='store_true',
        help='Write runtime_profile.csv with GPU utilization, memory, and epoch timing snapshots.',
    )
    parser.add_argument(
        '--history-diagnostic-every-epochs',
        type=int,
        default=0,
        help='Aggregate full physical/adaptive history diagnostics every N epochs. 0 means eval/checkpoint/final epochs only.',
    )
    parser.add_argument(
        '--torch-tf32',
        choices=['on', 'off'],
        default='on',
        help='Allow CUDA TF32 matmul/cudnn kernels when available. Default: on.',
    )
    parser.add_argument(
        '--torch-matmul-precision',
        choices=['highest', 'high', 'medium'],
        default='high',
        help='torch.set_float32_matmul_precision value used when TF32 is on. Default: high.',
    )
    parser.add_argument(
        '--transition-batch-size',
        type=int,
        default=0,
        help='Maximum transition pairs per batch group. 0 batches each equal-gap group together.',
    )
    parser.add_argument(
        '--segment-rollout-batch-size',
        type=int,
        default=0,
        help='Maximum segment rollout sequences per batch group. 0 batches each equal-length group together.',
    )
    parser.add_argument(
        '--cross-lake-batch-mode',
        choices=['off', 'on'],
        default='off',
        help='Batch compatible same-grid lake-years together during training. Incompatible grids fall back by bucket.',
    )
    parser.add_argument(
        '--cross-lake-batch-size',
        type=int,
        default=0,
        help='Maximum samples per compatible cross-lake training chunk. 0 uses each full compatible bucket.',
    )
    parser.add_argument(
        '--rolling-horizon-batch-size',
        type=int,
        default=32,
        help='Maximum rolling-start profiles per evaluation batch. 0 batches all selected starts together.',
    )
    parser.add_argument(
        '--train-diagnostic-mode',
        choices=['loss', 'full'],
        default=None,
        help='Diagnostics computed inside training step calls. loss is faster and preserves loss semantics; full is for debugging.',
    )
    parser.add_argument(
        '--init-mode',
        choices=['profile', 'lst_profile_prior', 'prior_spinup', 'uniform_lst_debug'],
        default='profile',
        help='Held-out reconstruction initialization. profile falls back to prior_spinup when no profile exists.',
    )
    parser.add_argument('--spinup-days', type=int, default=90)
    parser.add_argument('--spinup-lst-assimilation-strength', type=float, default=0.08)
    parser.add_argument('--spinup-lst-assimilation-decay-depth-m', type=float, default=2.0)
    parser.add_argument('--spinup-lst-assimilation-max-increment-c', type=float, default=0.5)
    parser.add_argument('--rollout-start-date', default=None)
    parser.add_argument(
        '--rollout-mode',
        choices=['free', 'profile_reinit'],
        default='free',
        help='Held-out export rollout mode. profile_reinit is reconstruction-only and resets on observed profile dates.',
    )
    parser.add_argument(
        '--rollout-reinit-scope',
        choices=['train', 'all'],
        default='train',
        help="Profiles available for profile_reinit. Use all only for analysis/reconstruction diagnostics.",
    )
    parser.add_argument(
        '--checkpoint-path',
        default=None,
        help='Existing checkpoint .pt to load before export or continued training.',
    )
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Load --checkpoint-path, build held-out lake data, and export reconstruction outputs without training.',
    )
    parser.add_argument(
        '--export-after-training',
        choices=['off', 'on'],
        default='off',
        help='Generate held-out CSV/PNG/scorecard immediately after training. off saves checkpoint/history only; use --export-only later.',
    )
    parser.add_argument(
        '--export-max-depth-m',
        type=float,
        default=None,
        help='Optional maximum depth for exported prediction CSV/heatmap/scorecard products only. Training remains full-depth.',
    )
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    train_multilake_state_forecaster(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        depth_points=args.depth_points,
        max_rollout_days=args.max_rollout_days,
        history_window_days=args.history_window_days,
        split_mode=args.split_mode,
        profile_supervision_scope=args.profile_supervision_scope,
        test_lake_id=args.test_lake_id,
        test_lake_ids=args.test_lake_ids,
        heldout_lake_groups=args.heldout_lake_groups,
        residual_limit_c=args.residual_limit_c,
        wind_kz_scale=args.wind_kz_scale,
        autumn_convective_boost=args.autumn_convective_boost,
        shape_aware_mixing=args.shape_aware_mixing,
        shape_mixing_strength=args.shape_mixing_strength,
        stratification_mixing_cap=args.stratification_mixing_cap,
        stratification_mixing_cap_strength=args.stratification_mixing_cap_strength,
        residual_regularization_weight=args.residual_regularization_weight,
        daily_tendency_weight=args.daily_tendency_weight,
        physical_scale_regularization_weight=args.physical_scale_regularization_weight,
        physical_scale_smoothness_weight=args.physical_scale_smoothness_weight,
        kd_prior_regularization_weight=args.kd_prior_regularization_weight,
        adaptive_parameter_regularization_weight=args.adaptive_parameter_regularization_weight,
        heat_content_transition_weight=args.heat_content_transition_weight,
        heat_content_full_column_min_coverage=args.heat_content_full_column_min_coverage,
        heat_content_transition_season_factors=args.heat_content_transition_season_factors,
        heat_content_transition_season_mode=args.heat_content_transition_season_mode,
        heat_content_transition_depth_factor=args.heat_content_transition_depth_factor,
        heat_content_transition_effective_max=args.heat_content_transition_effective_max,
        transition_loss_weight=args.transition_loss_weight,
        segment_rollout_start_epoch=args.segment_rollout_start_epoch,
        segment_rollout_ramp_epochs=args.segment_rollout_ramp_epochs,
        segment_rollout_loss_weight=args.segment_rollout_loss_weight,
        segment_rollout_lst_surface_weight=args.segment_rollout_lst_surface_weight,
        segment_rollout_max_days=args.segment_rollout_max_days,
        segment_rollout_samples_per_lake=args.segment_rollout_samples_per_lake,
        warm_season_column_heat_content_weight=args.warm_season_column_heat_content_weight,
        warm_season_column_heat_content_quantile_low=args.warm_season_column_heat_content_quantile_low,
        warm_season_column_heat_content_quantile_high=args.warm_season_column_heat_content_quantile_high,
        warm_season_column_heat_content_min_gap_days=args.warm_season_column_heat_content_min_gap_days,
        lst_feature_dropout_probability=args.lst_feature_dropout_probability,
        teacher_forcing_start=args.teacher_forcing_start,
        teacher_forcing_end=args.teacher_forcing_end,
        state_noise_weight=args.state_noise_weight,
        residual_time_smooth_weight=args.residual_time_smooth_weight,
        rolling_horizon_eval_max_starts=args.rolling_horizon_eval_max_starts,
        init_mode=args.init_mode,
        spinup_days=args.spinup_days,
        spinup_lst_assimilation_strength=args.spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=args.spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=args.spinup_lst_assimilation_max_increment_c,
        rollout_start_date=args.rollout_start_date,
        rollout_mode=args.rollout_mode,
        rollout_reinit_scope=args.rollout_reinit_scope,
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        eval_every_epochs=args.eval_every_epochs,
        full_eval_every_epochs=args.full_eval_every_epochs,
        profile_runtime=args.profile_runtime,
        profile_gpu=args.profile_gpu,
        history_diagnostic_every_epochs=args.history_diagnostic_every_epochs,
        torch_tf32=args.torch_tf32,
        torch_matmul_precision=args.torch_matmul_precision,
        transition_batch_size=args.transition_batch_size,
        segment_rollout_batch_size=args.segment_rollout_batch_size,
        rolling_horizon_batch_size=args.rolling_horizon_batch_size,
        train_diagnostic_mode=args.train_diagnostic_mode,
        export_after_training=args.export_after_training,
        export_max_depth_m=args.export_max_depth_m,
        cross_lake_batch_mode=args.cross_lake_batch_mode,
        cross_lake_batch_size=args.cross_lake_batch_size,
        export_only=args.export_only,
        hard_density_stability=args.hard_density_stability,
        turbulent_flux_mode=args.turbulent_flux_mode,
        turbulent_flux_blend_alpha=args.turbulent_flux_blend_alpha,
        freezing_energy_mode=args.freezing_energy_mode,
        advective_heat_source_mode=args.advective_heat_source_mode,
        lake_adaptive_params=args.lake_adaptive_params,
        lake_adaptive_hidden_dim=args.lake_adaptive_hidden_dim,
        lake_adaptive_init_spread=args.lake_adaptive_init_spread,
        lake_adaptive_temporal_mode=args.lake_adaptive_temporal_mode,
        lake_adaptive_temporal_init_spread=args.lake_adaptive_temporal_init_spread,
        lake_adaptive_temporal_scale=args.lake_adaptive_temporal_scale,
        adaptive_wind_kz_min=args.adaptive_wind_kz_min,
        adaptive_wind_kz_max=args.adaptive_wind_kz_max,
        adaptive_blend_alpha_min=args.adaptive_blend_alpha_min,
        adaptive_blend_alpha_max=args.adaptive_blend_alpha_max,
        adaptive_kd_multiplier_min=args.adaptive_kd_multiplier_min,
        adaptive_kd_multiplier_max=args.adaptive_kd_multiplier_max,
        adaptive_turbulent_exchange_scale_min=args.adaptive_turbulent_exchange_scale_min,
        adaptive_turbulent_exchange_scale_max=args.adaptive_turbulent_exchange_scale_max,
        adaptive_convective_mixing_scale_min=args.adaptive_convective_mixing_scale_min,
        adaptive_convective_mixing_scale_max=args.adaptive_convective_mixing_scale_max,
        adaptive_ice_shortwave_scale_min=args.adaptive_ice_shortwave_scale_min,
        adaptive_ice_shortwave_scale_max=args.adaptive_ice_shortwave_scale_max,
        device=args.device,
    )


if __name__ == '__main__':
    main()
