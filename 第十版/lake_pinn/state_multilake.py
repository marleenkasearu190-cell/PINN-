"""Multi-lake trainer for the reconstruction-state LakePINN.

This module trains the state-space forecaster directly:

    T(t + dt, z) = M(T(t, z), forcing history, lake attributes)

Each lake keeps its own depth grid and hypsometry curve.  The shared neural
parts learn global parameterizations for Kz/Kd/model residuals, while static
lake attributes enter through the state forecaster's FiLM adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import subprocess
import sys
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
from .constants import (
    PINN_MAX_TEMPERATURE_REFERENCE_C,
    RHO_CP,
    SECONDS_PER_DAY,
    STEFAN_BOLTZMANN,
    SURFACE_ALBEDO_ICE,
    SURFACE_ALBEDO_WATER,
    WATER_EMISSIVITY,
)
from .diagnostics import write_density_stability_summary, write_heat_closure_summaries
from .state_model import (
    ForcingBatch,
    ForcingRowSequence,
    LakeStateForecaster,
    MULTITASK_AUXILIARY_EOF_COMPONENTS,
    MULTITASK_AUXILIARY_STATE_KEYS,
    STATIC_FEATURE_DIM,
    STATIC_FEATURE_KEYS,
    ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM,
    lake_adaptive_param_set,
    normalize_advective_heat_source_mode,
    normalize_lake_adaptive_params,
    normalize_lake_adaptive_temporal_mode,
    normalize_freezing_energy_mode,
    normalize_shape_aware_mixing,
    resolve_hard_density_stability,
    static_feature_array,
)
from .physics import normalize_turbulent_flux_mode, water_density_torch
from .state_reconstruction import (
    MAINLINE_LSWT_OBSERVER_MODE_CHOICES,
    MAINLINE_ZERO_PROFILE_INITIALIZER_MODE_CHOICES,
    _build_rollout_pairs,
    _forcing_tensor_rows,
    _profile_lookup,
    _profile_physics_loss,
    _zero_profile_init_conditioning_from_inputs,
    apply_lswt_observer_update,
    build_zero_profile_low_dof_prior,
    build_zero_profile_lswt_climatology_low_dof_prior,
    build_zero_profile_eof_pca_low_dof_prior,
    build_zero_profile_eof_pca_init_net_prior,
    fit_zero_profile_eof_pca_basis,
    initialize_rollout_state,
    normalize_lswt_observer_mode,
    normalize_mainline_lswt_observer_mode,
    normalize_mainline_zero_profile_initializer_mode,
    normalize_zero_profile_thermal_basis_balance_mode,
    normalize_zero_profile_initializer_mode,
    zero_profile_thermal_basis_tensors_for_depths,
    ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODES,
)
from .unlabeled_heat_closure import (
    DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS,
    build_unlabeled_heat_closure_windows,
    build_unlabeled_heat_closure_windows_by_horizon,
    build_unlabeled_heat_closure_windows_for_horizon,
    format_unlabeled_heat_closure_horizons,
    parse_unlabeled_heat_closure_horizons,
)
from .vertical_solver import layer_thicknesses
from .export import export_temperature_tables
from .plotting import plot_year_heatmap
from .scorecard_integration import generate_prediction_diagnostic_figures, run_scorecard_report


DEFAULT_RESIDUAL_LIMIT_C = 0.25
DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT = 0.05
DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT = 0.03
DEFAULT_DAILY_TENDENCY_WEIGHT = 0.03
DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT = 0.02
DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT = 0.01
DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT = 0.003
DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT = 0.02
DEFAULT_ADAPTIVE_WIND_KZ_MIN = 0.6
DEFAULT_ADAPTIVE_WIND_KZ_MAX = 2.0
DEFAULT_ADAPTIVE_BLEND_ALPHA_MIN = 0.0
DEFAULT_ADAPTIVE_BLEND_ALPHA_MAX = 0.4
DEFAULT_ADAPTIVE_KD_MULTIPLIER_MIN = 0.6
DEFAULT_ADAPTIVE_KD_MULTIPLIER_MAX = 1.6
DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MIN = 0.7
DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MAX = 1.4
DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MIN = 0.5
DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MAX = 1.8
DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MIN = 0.6
DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MAX = 1.4

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
DEFAULT_PROFILE_LOSS_TARGET_MODE = 'grid_masked'
PROFILE_LOSS_TARGET_MODES = {'grid_masked', 'observed_point_strict'}
DEFAULT_PROFILE_SAMPLING_MODE = 'time_uniform'
PROFILE_SAMPLING_MODES = {'time_uniform', 'season_balanced'}
DEFAULT_MULTITASK_AUXILIARY_WEIGHT = 0.0
DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT = 1.0
DEFAULT_MULTITASK_AUXILIARY_HIDDEN_DIM = 64
DEFAULT_MULTITASK_AUXILIARY_HUBER_DELTA = 2.0
DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT = 0.03
DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT = 0.0
DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE = 4
DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS = 1
DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2 = 50.0
DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY = 'on'
DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN = 0.05
DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT = 0.0
DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE = 'diagnostic_only'
DEFAULT_UNLABELED_HEAT_CLOSURE_MODE = 'storage_budget_thresholded'
DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE = 'spinup_then_window'
DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS = 90
DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT = 0.0003
DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2 = 20.0
UNLABELED_HEAT_CLOSURE_RESERVOIR_MODES = {'diagnostic_only', 'include', 'exclude'}
UNLABELED_HEAT_CLOSURE_MODES = {
    'storage_budget_smooth_l1',
    'storage_budget_thresholded',
}
UNLABELED_HEAT_CLOSURE_STATE_SOURCES = {'prior_window', 'spinup_then_window'}
DEPTH_STRATIFIED_RMSE_BOUNDARY_M = 25.0
DEPTH_RMSE_BANDS = ('le25m', 'gt25m')
DEFAULT_EXPORT_STYLE_VALIDATION_MODE = 'off'
DEFAULT_EXPORT_STYLE_VALIDATION_MAX_LAKES = 0
DEFAULT_FULL_EVAL_POINT_DIAGNOSTICS_MODE = 'off'
DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MODE = 'off'
DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MAX_LAKES = 0
DEFAULT_ZERO_PROFILE_INITIALIZER_MODE = 'low_dof'
DEFAULT_ZERO_PROFILE_THERMAL_BASIS_COMPONENTS = 4
DEFAULT_ZERO_PROFILE_THERMAL_BASIS_GRID_POINTS = 40
DEFAULT_ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODE = 'lake_season_depth_coverage'
DEFAULT_REFIT_ZERO_PROFILE_THERMAL_BASIS = 'off'
DEFAULT_ZERO_PROFILE_INIT_NET_LOSS_WEIGHT = 0.0
DEFAULT_ZERO_PROFILE_INIT_NET_START_EPOCH = 0
DEFAULT_ZERO_PROFILE_INIT_NET_RAMP_EPOCHS = 0
DEFAULT_ZERO_PROFILE_INIT_NET_SAMPLES_PER_LAKE = 0
DEFAULT_ZERO_PROFILE_INIT_NET_REGULARIZATION_WEIGHT = 0.02
DEFAULT_ZERO_PROFILE_INIT_NET_HIDDEN_DIM = 64
DEFAULT_ZERO_PROFILE_INIT_NET_INIT_SPREAD = 0.0
DEFAULT_ZERO_PROFILE_INIT_NET_COEFF_LIMIT_SIGMA = 2.0
DEFAULT_ZERO_PROFILE_INIT_NET_DELTA_LIMIT_C = 3.0
DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS = 90
DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT = 0.02
DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT = 0.0
DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS = 60
DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS = 2
DEFAULT_DAILY_MEMORY_RECONSTRUCTION_WEIGHT = 0.0
DEFAULT_DAILY_MEMORY_SAMPLES_PER_LAKE = 0
DEFAULT_DAILY_MEMORY_TEMPORAL_SMOOTHNESS_WEIGHT = 0.01
DEFAULT_DAILY_MEMORY_HEAT_BUDGET_WEIGHT = 0.003
DEFAULT_DAILY_MEMORY_PHYSICS_CONSISTENCY_WEIGHT = 0.01
DEFAULT_DAILY_MEMORY_START_EPOCH = 0
DEFAULT_DAILY_MEMORY_RAMP_EPOCHS = 0
DEFAULT_DAILY_MEMORY_HIDDEN_DIM = 64
DEFAULT_DAILY_MEMORY_INIT_SPREAD = 0.0
DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA = 2.0
DEFAULT_DAILY_MEMORY_REGULARIZATION_WEIGHT = 0.02
DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_WEIGHT = 0.0
DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_RIDGE = 1.0e-4
DAILY_MEMORY_HISTORY_COEFFICIENT_COMPONENTS = DEFAULT_ZERO_PROFILE_THERMAL_BASIS_COMPONENTS
DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE = 'off'
THERMAL_STATE_PROFILE_FUSION_MODES = {'off', 'init_only', 'daily_only', 'both'}
DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY = 'past_strict'
THERMAL_STATE_PROFILE_FUSION_TIME_POLICIES = {
    'past_only',
    'past_strict',
    'nearest',
    'nearest_strict',
}
DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT = 'train'
DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS = 45
DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION = 0.25
DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT = 0.75
DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA = 4.0
PREDICTION_BRANCHES = {'physics_rollout', 'daily_memory'}
MODEL_MAINLINE_MODES = {
    'auto',
    'init_physics_rollout',
    'daily_memory',
}
RESOLVED_MODEL_MAINLINES = (
    'init_physics_rollout',
    'daily_memory',
)
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE = 'off'
DEFAULT_ZERO_PROFILE_SPINUP_DAYS_MATRIX = ''
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH = 0.08
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M = 4.0
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C = 0.75
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION = 0.15
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C = 0.35
DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY = 0.05
DEFAULT_KD_SATURATION_THRESHOLD = 1.55
DEFAULT_KD_SATURATION_PENALTY_WEIGHT = 0.002
DEFAULT_GPU_BATCH_AUTOTUNE = 'off'
DEFAULT_GPU_BATCH_AUTOTUNE_TARGET_BATCH_SIZE = 128
DEFAULT_TRAINING_AMP = 'off'
TRAINING_AMP_MODES = {'off', 'bf16', 'fp16'}
DEFAULT_TRAINING_HISTORY_DETAIL_EVERY_EPOCHS = 5

REMOVED_FEWSHOT_MAINLINE_FIELDS = {
    'episodic_fewshot_mode',
    'episodic_fewshot_loss_weight',
    'episodic_fewshot_start_epoch',
    'episodic_fewshot_ramp_epochs',
    'episodic_fewshot_max_query_days',
    'episodic_fewshot_samples_per_lake',
    'episodic_fewshot_support_profile_count',
    'episodic_fewshot_initial_delta_regularization_weight',
    'episodic_fewshot_unobserved_delta_regularization_weight',
    'episodic_fewshot_heat_content_regularization_weight',
    'episodic_fewshot_adapter_regularization_weight',
    'episodic_fewshot_observer_mode',
    'episodic_fewshot_observer_adapter_decay_days',
    'episodic_fewshot_observer_state_gain',
    'episodic_fewshot_observer_post_assimilation_weight',
    'episodic_fewshot_observer_heat_content_weight',
    'episodic_fewshot_support_schedule_strategy',
    'episodic_fewshot_support_min_gap_days',
    'support_persistence_loss_weight',
    'support_persistence_min_days',
    'support_persistence_max_days',
    'support_persistence_horizon_weight',
    'fewshot_hidden_dim',
    'fewshot_init_spread',
    'fewshot_initial_delta_limit_c',
    'fewshot_unobserved_delta_scale',
    'fewshot_adapter_scale',
    'fewshot_adapter_params',
    'fewshot_mainline_disabled',
}
REMOVED_PROFILE_SUPPORT_ROLLOUT_FIELDS = {
    'support_assimilation_strength',
    'support_assimilation_radius_m',
    'support_assimilation_max_increment_c',
    'support_assimilation_unobserved_depth_scale',
    'support_assimilation_heat_content_limit_c',
    'observer_hidden_dim',
    'observer_init_spread',
    'observer_state_delta_limit_c',
    'observer_unobserved_delta_scale',
    'observer_residual_anchor_fraction',
    'rollout_mode',
    'rollout_reinit_scope',
    'sparse_observer_profile_count',
    'sparse_observer_min_gap_days',
    'sparse_observer_support_schedule_strategy',
    'sparse_observer_state_gain',
    'sparse_observer_adapter_decay_days',
}
REMOVED_ZERO_PROFILE_MAINLINE_FIELDS = (
    REMOVED_FEWSHOT_MAINLINE_FIELDS | REMOVED_PROFILE_SUPPORT_ROLLOUT_FIELDS
)
REMOVED_ZERO_PROFILE_MAINLINE_CLI_FLAGS = {
    '--' + field.replace('_', '-')
    for field in REMOVED_ZERO_PROFILE_MAINLINE_FIELDS
    if field != 'fewshot_mainline_disabled'
}
REMOVED_MAINLINE_OUTPUT_EXACT_KEYS = {
    'fewshot_mainline_disabled',
    'episodic_seconds',
    'train_supervision_episodic_fewshot_sequence_count',
    'supervision_episodic_fewshot_sequences',
    'train_episodic_fewshot_sequences',
    'val_episodic_fewshot_sequences',
    'all_episodic_fewshot_sequences',
    'rollout_mode',
    'rollout_reinit_scope',
}
REMOVED_MAINLINE_OUTPUT_PREFIXES = (
    'episodic_fewshot',
    'fewshot_',
    'support_persistence',
    'support_assimilation',
    'segment_rollout_support_assimilation',
    'sparse_observer',
    'scheduled_sparse_observer',
)


def _is_removed_mainline_output_key(key):
    text = str(key)
    if text in REMOVED_MAINLINE_OUTPUT_EXACT_KEYS:
        return True
    if text.endswith('_support_input_at_inference'):
        return True
    if any(text.startswith(prefix) for prefix in REMOVED_MAINLINE_OUTPUT_PREFIXES):
        return True
    return (
        '_episodic_fewshot_' in text
        or '_fewshot_' in text
        or '_support_persistence_' in text
        or '_support_assimilation_' in text
        or '_sparse_observer_' in text
        or '_scheduled_sparse_observer_' in text
    )


def _prune_removed_mainline_output_fields(mapping):
    return {
        key: value
        for key, value in mapping.items()
        if not _is_removed_mainline_output_key(key)
    }


def _prune_removed_mainline_split_summary(payload):
    pruned = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            pruned[key] = _prune_removed_mainline_output_fields(value)
        else:
            pruned[key] = value
    return pruned


def _reject_removed_zero_profile_fields(manifest):
    present = _removed_manifest_fields(manifest, REMOVED_ZERO_PROFILE_MAINLINE_FIELDS)
    if present:
        raise ValueError(
            "Few-shot/support-profile mainline fields were removed for zero-profile RECON "
            f"({', '.join(present)}). Remove these fields from the manifest; use "
            "zero_profile_initializer, zero_profile_lswt_observer_mode, rolling/export-style "
            "validation, and depth-band metrics instead."
        )


def _suppress_removed_cli_flags(parser):
    for action in parser._actions:
        if set(action.option_strings).intersection(REMOVED_ZERO_PROFILE_MAINLINE_CLI_FLAGS):
            action.help = argparse.SUPPRESS


def _reject_removed_cli_flags(parser, argv):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    present = []
    for token in raw_args:
        if not str(token).startswith('--'):
            continue
        flag = str(token).split('=', 1)[0]
        if flag in REMOVED_ZERO_PROFILE_MAINLINE_CLI_FLAGS:
            present.append(flag)
    if present:
        parser.error(
            "few-shot/support-profile CLI flags were removed for zero-profile RECON: "
            + ', '.join(sorted(set(present)))
        )


def _normalize_model_mainline(mode):
    mode = str(mode or 'auto').strip().lower().replace('-', '_')
    if mode not in MODEL_MAINLINE_MODES:
        raise ValueError(
            'model_mainline must be one of: '
            + ', '.join(sorted(MODEL_MAINLINE_MODES))
        )
    return mode


def _resolve_model_mainline(
    mode,
    *,
    daily_memory_reconstruction_weight=0.0,
    prediction_branch='physics_rollout',
):
    mode = _normalize_model_mainline(mode)
    prediction_branch = str(prediction_branch or 'physics_rollout').strip().lower().replace('-', '_')
    daily_enabled = float(daily_memory_reconstruction_weight or 0.0) > 0.0
    if mode == 'auto':
        if prediction_branch == 'daily_memory':
            if not daily_enabled:
                raise ValueError(
                    "prediction_branch='daily_memory' requires "
                    'daily_memory_reconstruction_weight > 0 so the selected branch is trained.'
                )
            return 'daily_memory'
        return 'init_physics_rollout'
    if mode == 'init_physics_rollout':
        if prediction_branch != 'physics_rollout':
            raise ValueError(
                "model_mainline=init_physics_rollout requires prediction_branch='physics_rollout'."
            )
        return mode
    if mode == 'daily_memory':
        if not daily_enabled:
            raise ValueError(
                'model_mainline=daily_memory requires daily_memory_reconstruction_weight > 0.'
            )
        if prediction_branch != 'daily_memory':
            raise ValueError(
                "model_mainline=daily_memory requires prediction_branch='daily_memory'."
            )
        return mode
    raise ValueError(f'Unsupported model_mainline: {mode}')


def _daily_memory_training_role(
    resolved,
    *,
    daily_memory_reconstruction_weight=0.0,
    prediction_branch='physics_rollout',
):
    resolved = str(resolved)
    prediction_branch = _normalize_prediction_branch(prediction_branch)
    daily_enabled = float(daily_memory_reconstruction_weight or 0.0) > 0.0
    if resolved == 'daily_memory' or prediction_branch == 'daily_memory':
        return 'prediction'
    if daily_enabled:
        return 'auxiliary'
    return 'off'


def _model_mainline_config_fields(
    mode,
    resolved,
    *,
    daily_memory_reconstruction_weight=0.0,
    prediction_branch='physics_rollout',
    physics_rollout_thermal_state_weight=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
    physics_rollout_thermal_state_weight_eff=None,
    thermal_state_profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
):
    resolved = str(resolved)
    fusion_mode = _normalize_thermal_state_profile_fusion_mode(thermal_state_profile_fusion_mode)
    role = _daily_memory_training_role(
        resolved,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
    )
    requested = float(physics_rollout_thermal_state_weight or 0.0)
    if physics_rollout_thermal_state_weight_eff is None:
        effective = requested if resolved == 'init_physics_rollout' else 0.0
    else:
        effective = float(physics_rollout_thermal_state_weight_eff or 0.0)
    return {
        'model_mainline': _normalize_model_mainline(mode),
        'model_mainline_resolved': resolved,
        'model_mainline_branch_count': int(len(RESOLVED_MODEL_MAINLINES)),
        'model_mainline_branch_names': ','.join(RESOLVED_MODEL_MAINLINES),
        'model_mainline_physics_primary': True,
        'physics_rollout_thermal_state_weight': requested,
        'physics_rollout_thermal_state_weight_eff': effective,
        'physics_rollout_thermal_state_enabled': (
            resolved == 'init_physics_rollout' and effective > 0.0
        ),
        'daily_memory_training_role': role,
        'profile_observations_role': (
            'training_supervision_only'
            if fusion_mode == 'off' else
            'profile_conditioned_low_dimensional_state_fusion'
        ),
        'era5_lst_role': 'inference_forcing',
    }


def _resolve_physics_rollout_thermal_state_weight(
    resolved,
    *,
    requested_weight=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
):
    requested = float(requested_weight or 0.0)
    multitask = float(multitask_auxiliary_weight or 0.0)
    if requested < 0.0:
        raise ValueError('physics_rollout_thermal_state_weight must be non-negative.')
    if multitask < 0.0:
        raise ValueError('multitask_auxiliary_weight must be non-negative.')
    if str(resolved) != 'init_physics_rollout':
        return multitask, 0.0
    effective = max(multitask, requested)
    return effective, effective


def _parse_zero_profile_spinup_days_matrix(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw_items = text.split(',')
    days = []
    for item in raw_items:
        if item is None or str(item).strip() == '':
            continue
        day = int(float(str(item).strip()))
        if day < 0:
            raise ValueError('zero_profile_spinup_days_matrix values must be non-negative.')
        if day not in days:
            days.append(day)
    return tuple(days)


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


def _lake_reservoir_bucket(lake):
    metadata = lake.get('metadata') or {}
    indicator = metadata.get('reservoir_indicator')
    try:
        indicator = float(indicator)
    except (TypeError, ValueError):
        indicator = np.nan
    if np.isfinite(indicator):
        return 'reservoir' if indicator >= 0.5 else 'natural'
    for key in ('lake_type', 'waterbody_type', 'lake_group', 'lake_id'):
        text = str(metadata.get(key, lake.get(key, '')) or '').strip().lower()
        if 'reservoir' in text or 'impound' in text or 'regulated' in text:
            return 'reservoir'
    return 'natural'


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
        'heat_capacity_areal_j_m2_c',
        'light_penetration_ratio',
        'hydrology_missing_flag',
        'kd_source_type_code',
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
        'heat_capacity_areal_norm',
        'light_penetration_ratio_norm',
        'hydrology_missing_flag_norm',
        'kd_source_type_norm',
        'metadata_path',
    )
    payload = {
        key: _metadata_summary_value(metadata.get(key))
        for key in keys
        if key in metadata
    }
    payload['static_feature_dim'] = int(lake['static_features'].reshape(-1).numel())
    payload['static_feature_keys'] = list(STATIC_FEATURE_KEYS)
    payload['forcing_cache_mode'] = lake.get('forcing_cache_mode', 'row_dict')
    payload['forcing_tensor_keys'] = sorted(lake.get('forcing_tensors', {}).keys())
    payload['resident_tensor_cache'] = bool(lake.get('resident_tensors'))
    target_cache = lake.get('target_tensors_by_day') or {}
    payload['target_matrix_cache'] = bool(target_cache)
    payload['target_matrix_available_counts'] = {
        split: int(entry.get('available_count', 0))
        for split, entry in target_cache.items()
    }
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


def _empty_depth_squared_errors_by_band():
    return {band: [] for band in DEPTH_RMSE_BANDS}


def _empty_depth_errors_by_band():
    return {band: [] for band in DEPTH_RMSE_BANDS}


def _empty_depth_horizon_errors(horizons):
    return {
        band: {int(horizon): [] for horizon in horizons}
        for band in DEPTH_RMSE_BANDS
    }


def _depth_band_masks(depths):
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    finite_depths = np.isfinite(depths)
    return {
        'le25m': finite_depths & (depths <= DEPTH_STRATIFIED_RMSE_BOUNDARY_M),
        'gt25m': finite_depths & (depths > DEPTH_STRATIFIED_RMSE_BOUNDARY_M),
    }


def _extend_depth_squared_errors(errors_by_band, squared_errors, depths):
    squared_errors = np.asarray(squared_errors, dtype=np.float64).reshape(-1)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    if squared_errors.size != depths.size:
        raise ValueError('squared_errors and depths must have the same length.')
    finite_errors = np.isfinite(squared_errors)
    for band, depth_mask in _depth_band_masks(depths).items():
        valid = finite_errors & depth_mask
        if np.any(valid):
            errors_by_band[band].extend(squared_errors[valid].tolist())


def _extend_depth_errors(errors_by_band, errors, depths):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    if errors.size != depths.size:
        raise ValueError('errors and depths must have the same length.')
    finite_errors = np.isfinite(errors)
    for band, depth_mask in _depth_band_masks(depths).items():
        valid = finite_errors & depth_mask
        if np.any(valid):
            errors_by_band[band].extend(errors[valid].tolist())


def _extend_depth_horizon_errors(depth_errors_by_horizon, horizon, squared_errors, depths):
    horizon = int(horizon)
    squared_errors = np.asarray(squared_errors, dtype=np.float64).reshape(-1)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    if squared_errors.size != depths.size:
        raise ValueError('squared_errors and depths must have the same length.')
    finite_errors = np.isfinite(squared_errors)
    for band, depth_mask in _depth_band_masks(depths).items():
        valid = finite_errors & depth_mask
        if np.any(valid):
            depth_errors_by_horizon[band][horizon].extend(squared_errors[valid].tolist())


def _depth_rmse_record_from_squared_errors(errors_by_band, *, prefix='rmse'):
    record = {}
    for band in DEPTH_RMSE_BANDS:
        values = np.asarray(errors_by_band.get(band, []), dtype=np.float64)
        finite = np.isfinite(values)
        record[f'{prefix}_{band}'] = float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else np.nan
        record[f'count_{band}'] = int(np.sum(finite))
    return record


def _depth_bias_record_from_errors(errors_by_band, *, prefix='bias'):
    record = {}
    for band in DEPTH_RMSE_BANDS:
        values = np.asarray(errors_by_band.get(band, []), dtype=np.float64)
        finite = np.isfinite(values)
        record[f'{prefix}_{band}'] = float(np.mean(values[finite])) if np.any(finite) else np.nan
    return record


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


class PackedLakeTensorStore:
    """Per-lake resident tensor access for forcing, targets, geometry, and static features."""

    __slots__ = (
        'forcing_tensors',
        'target_tensors_by_day',
        'depths',
        'area',
        'heat_content_layer_weights',
        'static_features',
        'static_features_2d',
        'date_to_index',
    )

    def __init__(
        self,
        *,
        forcing_tensors,
        target_tensors_by_day,
        depths,
        area,
        heat_content_layer_weights,
        static_features,
        static_features_2d,
        date_to_index,
    ):
        self.forcing_tensors = forcing_tensors or {}
        self.target_tensors_by_day = target_tensors_by_day or {}
        self.depths = depths
        self.area = area
        self.heat_content_layer_weights = heat_content_layer_weights
        self.static_features = static_features
        self.static_features_2d = static_features_2d
        self.date_to_index = date_to_index or {}

    def forcing_batch(self, indices):
        if not self.forcing_tensors:
            raise ValueError('forcing tensor cache is not available for this lake.')
        if torch.is_tensor(indices):
            index_tensor = indices.to(device=self.depths.device, dtype=torch.long).reshape(-1)
        else:
            index_values = [int(idx) for idx in indices]
            index_tensor = torch.as_tensor(index_values, dtype=torch.long, device=self.depths.device)
        if int(index_tensor.numel()) == 0:
            raise ValueError('forcing batch indices must not be empty.')
        return ForcingBatch(self.forcing_tensors, index_tensor)

    def target_entry_for_day(self, split_name, day_idx):
        entry = self.target_tensors_by_day.get(split_name)
        if not entry:
            return None
        day_idx = int(day_idx)
        if day_idx in entry.get('available_indices', ()):
            return entry
        return None

    def target_matrix_for_date(self, preferred_split, date_value):
        date_value = pd.Timestamp(date_value).normalize()
        day_idx = self.date_to_index.get(date_value)
        if day_idx is None:
            return None, None
        entry = self.target_entry_for_day(preferred_split, day_idx)
        if entry is not None:
            return entry, int(day_idx)
        entry = self.target_entry_for_day('all', day_idx)
        if entry is not None:
            return entry, int(day_idx)
        return None, None

    def static_rows(self, count):
        return self.static_features_2d.expand(int(count), -1)


def _target_matrix_entry_for_day(lake, split_name, day_idx):
    store = lake.get('packed_tensor_store')
    if store is not None:
        return store.target_entry_for_day(split_name, day_idx)
    matrix_by_split = lake.get('target_tensors_by_day') or {}
    entry = matrix_by_split.get(split_name)
    if not entry:
        return None
    day_idx = int(day_idx)
    if day_idx in entry.get('available_indices', ()):
        return entry
    return None


def _target_matrix_for_date(lake, preferred_split, date_value):
    store = lake.get('packed_tensor_store')
    if store is not None:
        return store.target_matrix_for_date(preferred_split, date_value)
    date_value = pd.Timestamp(date_value).normalize()
    day_idx = lake.get('date_to_index', {}).get(date_value)
    if day_idx is None:
        return None, None
    entry = _target_matrix_entry_for_day(lake, preferred_split, day_idx)
    if entry is not None:
        return entry, int(day_idx)
    entry = _target_matrix_entry_for_day(lake, 'all', day_idx)
    if entry is not None:
        return entry, int(day_idx)
    return None, None


def _target_tensor_and_mask(lake, preferred_split, date_value):
    date_value = pd.Timestamp(date_value).normalize()
    matrix_entry, day_idx = _target_matrix_for_date(lake, preferred_split, date_value)
    if matrix_entry is not None:
        return (
            matrix_entry['profiles'][int(day_idx):int(day_idx) + 1],
            matrix_entry['masks'][int(day_idx):int(day_idx) + 1],
        )
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


def _masked_rmse_profile_loss(prediction, target, mask=None):
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
    return torch.sqrt(torch.mean((prediction[valid] - target[valid]).pow(2)) + 1.0e-12)


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


def _area_weighted_profile_mean_delta_c(delta, depths, area, mask=None):
    if delta.ndim == 1:
        delta = delta.unsqueeze(0)
    area = torch.as_tensor(area, dtype=delta.dtype, device=delta.device).reshape(-1)
    weights = _heat_content_layer_weights(
        depths,
        area,
        device=delta.device,
        dtype=delta.dtype,
    ).reshape(1, -1)
    if mask is not None:
        weights = weights * torch.as_tensor(mask, dtype=delta.dtype, device=delta.device).reshape(delta.shape[0], -1)
    denom = torch.clamp(weights.sum(dim=1), min=1.0e-6)
    return torch.sum(delta * weights, dim=1) / denom


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


def _areal_heat_content_j_m2_vector(
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
    if layer_weights.shape[1] != n_depths:
        raise ValueError(
            f'Areal heat-content weights length {layer_weights.shape[1]} does not match profile depth length {n_depths}.'
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
    safe_profile = torch.where(finite, profile, torch.zeros_like(profile))
    surface_area = torch.clamp(area_tensor[0], min=1.0e-6)
    heat_content = RHO_CP * torch.sum(safe_profile * masked_weights / surface_area, dim=1)
    return torch.where(valid, heat_content, torch.zeros_like(heat_content)), valid


def _multitask_auxiliary_weight_vector(
    *,
    device,
    dtype,
    heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
):
    values = []
    for key in MULTITASK_AUXILIARY_STATE_KEYS:
        if key in {'column_mean_temperature', 'areal_heat_content_j_m2_normalized'}:
            values.append(float(heat_weight))
        elif key == 'thermocline_depth':
            values.append(float(thermocline_weight))
        elif key == 'mixed_layer_depth':
            values.append(float(mld_weight))
        elif key == 'surface_bottom_diff':
            values.append(float(surface_bottom_weight))
        elif key == 'schmidt_stability':
            values.append(float(stability_weight))
        elif key.startswith('eof_coeff_'):
            values.append(float(eof_weight))
        else:
            values.append(1.0)
    return torch.as_tensor(values, dtype=dtype, device=device).reshape(1, -1)


def _multitask_auxiliary_basis_tensors(model, lake, *, device, dtype):
    basis = getattr(model, 'zero_profile_thermal_basis', None)
    if basis is None:
        return None
    return zero_profile_thermal_basis_tensors_for_depths(
        basis,
        lake['depths_np'],
        device=device,
        dtype=dtype,
    )


def _project_profiles_to_auxiliary_eof_targets(profile, mask, basis_tensors, component_count):
    batch_size = int(profile.shape[0])
    device = profile.device
    dtype = profile.dtype
    coeff_targets = torch.zeros((batch_size, component_count), dtype=dtype, device=device)
    coeff_mask = torch.zeros((batch_size, component_count), dtype=torch.bool, device=device)
    if basis_tensors is None or component_count <= 0:
        return coeff_targets, coeff_mask
    components = basis_tensors.get('components_on_depth')
    mean_profile = basis_tensors.get('mean_profile_on_depth')
    coeff_std = basis_tensors.get('coeff_std')
    if components is None or mean_profile is None or coeff_std is None:
        return coeff_targets, coeff_mask
    components = components.to(device=device, dtype=dtype)
    mean_profile = mean_profile.to(device=device, dtype=dtype).reshape(-1)
    coeff_std = coeff_std.to(device=device, dtype=dtype).reshape(-1)
    available_count = min(component_count, int(components.shape[0]), int(coeff_std.numel()))
    if available_count <= 0:
        return coeff_targets, coeff_mask
    finite = torch.isfinite(profile) & mask.to(dtype=torch.bool)
    for sample_idx in range(batch_size):
        valid = finite[sample_idx]
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if int(valid_indices.numel()) < available_count:
            continue
        design = components[:available_count].index_select(1, valid_indices).transpose(0, 1)
        residual = (
            profile[sample_idx].index_select(0, valid_indices)
            - mean_profile.index_select(0, valid_indices)
        )
        coeff = torch.linalg.pinv(design) @ residual
        coeff_targets[sample_idx, :available_count] = coeff / torch.clamp(
            coeff_std[:available_count],
            min=1.0e-6,
        )
        coeff_mask[sample_idx, :available_count] = True
    return coeff_targets, coeff_mask


def _multitask_auxiliary_targets(model, lake, target, target_mask):
    if target.ndim == 1:
        target = target.unsqueeze(0)
    batch_size, n_depths = target.shape
    device = target.device
    dtype = target.dtype
    depths = lake['depths'].to(device=device, dtype=dtype).reshape(-1)
    max_depth = torch.clamp(depths[-1], min=1.0)
    if target_mask is None:
        mask = torch.ones((batch_size, n_depths), dtype=torch.bool, device=device)
    else:
        mask = torch.as_tensor(target_mask, dtype=torch.bool, device=device).reshape(batch_size, n_depths)
    finite = torch.isfinite(target)
    mask = mask & finite
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    values = torch.zeros(
        (batch_size, len(MULTITASK_AUXILIARY_STATE_KEYS)),
        dtype=dtype,
        device=device,
    )
    value_mask = torch.zeros_like(values, dtype=torch.bool)
    key_index = {key: idx for idx, key in enumerate(MULTITASK_AUXILIARY_STATE_KEYS)}

    column_mean, column_mean_valid = _column_mean_temperature_c_vector(
        safe_target,
        lake['depths'],
        lake['area'],
        mask=mask,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
    )
    if 'column_mean_temperature' in key_index:
        idx = key_index['column_mean_temperature']
        values[:, idx] = column_mean / float(PINN_MAX_TEMPERATURE_REFERENCE_C)
        value_mask[:, idx] = column_mean_valid

    if 'areal_heat_content_j_m2_normalized' in key_index:
        heat_content, heat_content_valid = _areal_heat_content_j_m2_vector(
            safe_target,
            lake['depths'],
            lake['area'],
            mask=mask,
            layer_weights=lake.get('heat_content_layer_weights'),
            min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
        )
        heat_scale = float(RHO_CP) * float(PINN_MAX_TEMPERATURE_REFERENCE_C) * torch.clamp(
            max_depth,
            min=1.0,
        )
        idx = key_index['areal_heat_content_j_m2_normalized']
        values[:, idx] = heat_content / heat_scale
        value_mask[:, idx] = heat_content_valid

    idx = key_index['surface_bottom_diff']
    endpoint_valid = mask[:, 0] & mask[:, -1] if n_depths > 1 else torch.zeros(batch_size, dtype=torch.bool, device=device)
    values[:, idx] = (safe_target[:, 0] - safe_target[:, -1]) / float(PINN_MAX_TEMPERATURE_REFERENCE_C)
    value_mask[:, idx] = endpoint_valid

    observed_count = mask.sum(dim=1)
    if n_depths > 1:
        dz = torch.clamp(depths[1:] - depths[:-1], min=1.0e-4)
        segment_valid = mask[:, :-1] & mask[:, 1:]
        gradient = torch.abs((safe_target[:, 1:] - safe_target[:, :-1]) / dz.reshape(1, -1))
        gradient_rank = torch.where(segment_valid, gradient, torch.full_like(gradient, -1.0))
        max_idx = torch.argmax(gradient_rank, dim=1)
        mid_depths = 0.5 * (depths[:-1] + depths[1:])
        thermocline_valid = torch.any(segment_valid, dim=1) & (observed_count >= 3)
        idx = key_index['thermocline_depth']
        values[:, idx] = mid_depths.index_select(0, max_idx) / max_depth
        value_mask[:, idx] = thermocline_valid

        surface_valid = mask[:, 0]
        mixed = (torch.abs(safe_target - safe_target[:, :1]) <= 0.5) & mask
        mixed_depths = torch.where(mixed, depths.reshape(1, -1), torch.zeros_like(safe_target))
        idx = key_index['mixed_layer_depth']
        values[:, idx] = mixed_depths.max(dim=1).values / max_depth
        value_mask[:, idx] = surface_valid & torch.any(mixed, dim=1)

    stability_valid = observed_count >= 3
    if torch.any(stability_valid):
        area_tensor = torch.as_tensor(lake['area'], dtype=dtype, device=device).reshape(-1)
        layer_weights = torch.clamp(area_tensor * layer_thicknesses(depths), min=1.0e-8).reshape(1, -1)
        observed_weights = layer_weights * mask.to(dtype=dtype)
        denom = torch.clamp(observed_weights.sum(dim=1), min=1.0e-8)
        rho = water_density_torch(safe_target)
        rho_mean = (rho * observed_weights).sum(dim=1) / denom
        z_mean = (depths.reshape(1, -1) * observed_weights).sum(dim=1) / denom
        stability = (
            (rho - rho_mean.reshape(-1, 1))
            * (depths.reshape(1, -1) - z_mean.reshape(-1, 1))
            * observed_weights
        ).sum(dim=1) / denom
        idx = key_index['schmidt_stability']
        values[:, idx] = stability / 100.0
        value_mask[:, idx] = stability_valid

    basis_tensors = _multitask_auxiliary_basis_tensors(model, lake, device=device, dtype=dtype)
    eof_targets, eof_mask = _project_profiles_to_auxiliary_eof_targets(
        safe_target,
        mask,
        basis_tensors,
        MULTITASK_AUXILIARY_EOF_COMPONENTS,
    )
    for component_idx in range(MULTITASK_AUXILIARY_EOF_COMPONENTS):
        key = f'eof_coeff_{component_idx + 1:02d}'
        if key in key_index:
            values[:, key_index[key]] = eof_targets[:, component_idx]
            value_mask[:, key_index[key]] = eof_mask[:, component_idx]
    return values, value_mask


def _multitask_auxiliary_loss_details(loss_vec, key_loss_matrix, target_mask, *, weight):
    records = []
    for sample_idx in range(loss_vec.shape[0]):
        record = {
            'multitask_auxiliary_loss': loss_vec[sample_idx].detach(),
            'multitask_auxiliary_weighted_loss': (float(weight) * loss_vec[sample_idx]).detach(),
            'multitask_auxiliary_supervision_count': target_mask[sample_idx].to(dtype=loss_vec.dtype).sum().detach(),
        }
        for key_idx, key in enumerate(MULTITASK_AUXILIARY_STATE_KEYS):
            record[f'multitask_auxiliary_{key}_loss'] = key_loss_matrix[sample_idx, key_idx].detach()
            record[f'multitask_auxiliary_{key}_enabled'] = (
                target_mask[sample_idx, key_idx].to(dtype=loss_vec.dtype).detach()
            )
        records.append(record)
    return records


def _multitask_auxiliary_loss_vector(
    model,
    lake,
    prediction,
    target,
    target_mask,
    forcing_row,
    *,
    static_features=None,
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    return_details=True,
):
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    device = prediction.device
    dtype = torch.float32
    batch_size = prediction.shape[0]
    zero_loss = torch.zeros(batch_size, dtype=dtype, device=device)
    zero_key = torch.zeros((batch_size, len(MULTITASK_AUXILIARY_STATE_KEYS)), dtype=dtype, device=device)
    zero_mask = torch.zeros_like(zero_key, dtype=torch.bool)
    if float(multitask_auxiliary_weight) <= 0.0:
        if not return_details:
            return zero_loss, None
        return zero_loss, _multitask_auxiliary_loss_details(
            zero_loss,
            zero_key,
            zero_mask,
            weight=multitask_auxiliary_weight,
        )
    static = lake['static_features'] if static_features is None else static_features
    predicted_aux = model.predict_multitask_auxiliary_state(
        prediction,
        forcing_row,
        static,
        depths=lake['depths'],
        area_profile=lake['area'],
    )
    predicted_aux = predicted_aux.to(dtype=dtype)
    target_aux, aux_mask = _multitask_auxiliary_targets(
        model,
        lake,
        target.to(device=device, dtype=dtype),
        target_mask,
    )
    target_aux = target_aux.to(dtype=dtype)
    weights = _multitask_auxiliary_weight_vector(
        device=device,
        dtype=dtype,
        heat_weight=heat_weight,
        thermocline_weight=thermocline_weight,
        mld_weight=mld_weight,
        stability_weight=stability_weight,
        surface_bottom_weight=surface_bottom_weight,
        eof_weight=eof_weight,
    )
    finite_aux = torch.isfinite(predicted_aux) & torch.isfinite(target_aux)
    active = aux_mask & finite_aux & (weights > 0.0)
    safe_predicted_aux = torch.where(
        torch.isfinite(predicted_aux),
        predicted_aux,
        torch.zeros_like(predicted_aux),
    )
    safe_target_aux = torch.where(
        torch.isfinite(target_aux),
        target_aux,
        torch.zeros_like(target_aux),
    )
    diff = safe_predicted_aux - safe_target_aux
    delta = torch.as_tensor(DEFAULT_MULTITASK_AUXILIARY_HUBER_DELTA, dtype=dtype, device=device)
    abs_diff = diff.abs()
    error = torch.where(
        abs_diff <= delta,
        0.5 * diff.pow(2),
        delta * (abs_diff - 0.5 * delta),
    )
    key_loss = torch.where(aux_mask & finite_aux, error, torch.zeros_like(error))
    weighted_error = torch.where(active, error * weights, torch.zeros_like(error))
    denom = torch.clamp((active.to(dtype=dtype) * weights).sum(dim=1), min=1.0e-8)
    loss_vec = weighted_error.sum(dim=1) / denom
    has_target = torch.any(active, dim=1)
    loss_vec = torch.where(has_target, loss_vec, torch.zeros_like(loss_vec))
    if not return_details:
        return loss_vec, None
    return loss_vec, _multitask_auxiliary_loss_details(
        loss_vec,
        key_loss,
        aux_mask,
        weight=multitask_auxiliary_weight,
    )


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
    target_gap_tensor = torch.as_tensor(
        target_gap_days,
        dtype=weighted_loss.dtype,
        device=weighted_loss.device,
    ).reshape(-1)
    active_f = active.to(dtype=weighted_loss.dtype)
    for pos, sample_idx in enumerate(sample_indices):
        warm_losses[sample_idx].append(raw_loss[pos] * active_f[pos])
        warm_weighted_losses[sample_idx].append(weighted_loss[pos] * active_f[pos])
        warm_factors[sample_idx].append(warm_factor[pos].detach() * active_f[pos])
        warm_errors_c[sample_idx].append(error_c[pos].detach() * active_f[pos])
        warm_gaps[sample_idx].append(torch.where(
            active[pos],
            target_gap_tensor[pos],
            torch.full_like(target_gap_tensor[pos], float('nan')),
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
    if warm_gaps:
        gaps = torch.stack(warm_gaps)
        active_mask = torch.isfinite(gaps)
        active_f = active_mask.to(dtype=zero.dtype)
        active_count = torch.clamp(active_f.sum(), min=1.0)
    else:
        gaps = None
        active_mask = None
        active_f = None
        active_count = torch.tensor(1.0, device=device, dtype=zero.dtype)

    def _active_mean(values):
        if not values:
            return zero
        stacked = torch.stack(values)
        if active_f is None or active_f.shape != stacked.shape:
            return stacked.mean()
        return (stacked * active_f).sum() / active_count

    unweighted = _active_mean(warm_losses)
    weighted = _active_mean(warm_weighted_losses)
    warm_factor_mean = _active_mean(warm_factors)
    error_mean = _active_mean(warm_errors_c)
    supervision_count = active_f.sum() if active_f is not None else zero
    if gaps is not None:
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
    residual_surface = diagnostics.get('residual_surface_abs_mean_c')
    if residual_surface is None:
        residual_surface = diagnostics['residual_surface_c'].abs()
    residual_deep = diagnostics.get('residual_deep_abs_mean_c')
    if residual_deep is None:
        residual_deep = diagnostics['residual_deep_mean_c'].abs()
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


def _kd_saturation_penalty_loss(diagnostics, threshold):
    kd_multiplier = diagnostics.get('nn_kd_multiplier')
    if kd_multiplier is None:
        device = diagnostics['residual_abs_mean_c'].device
        return torch.tensor(0.0, device=device)
    threshold = float(threshold)
    return torch.relu(kd_multiplier.reshape(-1) - threshold).pow(2).mean()


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
        'residual_abs_mean_c': diagnostics.get('residual_abs_mean_c', zeros).detach().mean(),
        'residual_surface_abs_mean_c': diagnostics.get(
            'residual_surface_abs_mean_c',
            diagnostics.get('residual_surface_c', zeros).abs(),
        ).detach().mean(),
        'residual_deep_abs_mean_c': diagnostics.get(
            'residual_deep_abs_mean_c',
            diagnostics.get('residual_deep_mean_c', zeros).abs(),
        ).detach().mean(),
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


def _select_segment_rollout_sequences(
    sequences,
    active_max_days,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
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
    if _normalize_profile_sampling_mode(sampling_mode) == 'season_balanced':
        return _season_balanced_sample(
            active,
            samples_per_lake,
            epoch,
            date_getter=lambda item: item[0],
            index_getter=lambda item: int(item[1]),
        )
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
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
    sampling_mode = _normalize_profile_sampling_mode(sampling_mode)
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
                sampling_mode=sampling_mode,
            )
            if active_max_days > 0 else []
        )
        selected = tuple(selected)
        plan.append({
            'epoch': int(epoch),
            'active_max_days': int(active_max_days),
            'samples_per_lake': int(samples_per_lake),
            'profile_sampling_mode': sampling_mode,
            'selected_sequences': selected,
            'batches': _segment_rollout_grouped_batches(
                selected,
                segment_rollout_batch_size,
            ),
        })
    return tuple(plan)


def _segment_rollout_plan_entry_for_epoch(
    lake,
    split_key,
    active_max_days,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
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
    if _normalize_profile_sampling_mode(
        entry.get('profile_sampling_mode', DEFAULT_PROFILE_SAMPLING_MODE)
    ) != _normalize_profile_sampling_mode(sampling_mode):
        return None
    return entry


def _segment_rollout_sequences_for_epoch(
    lake,
    split_key,
    active_max_days,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
    entry = _segment_rollout_plan_entry_for_epoch(
        lake,
        split_key,
        active_max_days,
        samples_per_lake,
        epoch,
        sampling_mode=sampling_mode,
    )
    if entry is not None:
        return list(entry['selected_sequences'])
    return _select_segment_rollout_sequences(
        lake['segment_rollout_sequences'][split_key],
        active_max_days,
        samples_per_lake,
        epoch,
        sampling_mode=sampling_mode,
    )


def _segment_rollout_batches_for_epoch(
    lake,
    split_key,
    active_max_days,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
    entry = _segment_rollout_plan_entry_for_epoch(
        lake,
        split_key,
        active_max_days,
        samples_per_lake,
        epoch,
        sampling_mode=sampling_mode,
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


def _build_cross_lake_transition_batches(
    lakes,
    *,
    pair_key='train',
    transition_batch_size=0,
    cross_lake_batch_size=0,
):
    grouped = {}
    for bucket_entries in _cross_lake_bucket_entries(lakes).values():
        for lake_idx, lake in bucket_entries:
            for pair in lake['pairs'][pair_key]:
                gap = int(pair[3] - pair[2])
                grouped.setdefault((_cross_lake_batch_key(lake), gap), []).append((lake_idx, lake, pair))

    batch_size = int(cross_lake_batch_size or transition_batch_size or 0)
    chunks = []
    for key in sorted(grouped, key=lambda value: (value[0][0], value[1])):
        chunks.extend(tuple(chunk) for chunk in _batch_chunks(grouped[key], batch_size))
    return tuple(chunks)


def _detail_values(details, key):
    tensors = []
    numeric_values = []
    for record in details:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().reshape(-1).mean())
        else:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
    if numeric_values:
        if tensors:
            ref = tensors[0]
            tensors.extend(
                torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
                for value in numeric_values
            )
        else:
            tensors.extend(torch.as_tensor(value) for value in numeric_values)
    if not tensors:
        return []
    return torch.stack(tensors).detach().float().cpu().numpy().tolist()


def _mean_detail(details, key):
    values = _detail_values(details, key)
    return float(np.mean(values)) if values else np.nan


def _sum_detail(details, key):
    values = _detail_values(details, key)
    return float(np.sum(values)) if values else 0.0


def _min_detail(details, key):
    values = _detail_values(details, key)
    return float(np.min(values)) if values else np.nan


def _max_detail(details, key):
    values = _detail_values(details, key)
    return float(np.max(values)) if values else np.nan


def _quantile_detail(details, key, quantile):
    values = _detail_values(details, key)
    return float(np.quantile(values, float(quantile))) if values else np.nan


def _fraction_detail_ge(details, key, threshold):
    values = np.asarray(_detail_values(details, key), dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite] >= float(threshold)))


def _metric_records_by_lake_bucket(metrics_by_lake, lakes, bucket):
    records = []
    for lake in lakes:
        if _lake_reservoir_bucket(lake) != bucket:
            continue
        record = metrics_by_lake.get(lake['lake_id'])
        if record:
            records.append(record)
    return records


def _add_free_roll_bucket_metrics(record, prefix, metrics_by_lake, lakes):
    for bucket in ('natural', 'reservoir'):
        records = _metric_records_by_lake_bucket(metrics_by_lake, lakes, bucket)
        base = f'{prefix}_{bucket}_free_roll'
        record[f'{base}_lake_count'] = len(records)
        record[f'{base}_rmse'] = _mean_numeric_records(records, 'rmse')
        record[f'{base}_bias'] = _mean_numeric_records(records, 'bias')
        record[f'{base}_mae'] = _mean_numeric_records(records, 'mae')
        record[f'{base}_profiles'] = _sum_numeric_records(records, 'n_profiles')
        for band in DEPTH_RMSE_BANDS:
            record[f'{base}_rmse_{band}'] = _mean_numeric_records(records, f'rmse_{band}')
            record[f'{base}_count_{band}'] = _sum_numeric_records(records, f'count_{band}')


def _add_zero_profile_export_metrics(record, prefix, metrics_by_lake, lakes):
    records = [
        metrics_by_lake.get(lake['lake_id'])
        for lake in lakes
        if metrics_by_lake.get(lake['lake_id'])
    ]
    base = f'{prefix}_zero_profile_export'
    record[f'{base}_lake_count'] = len(records)
    init_modes = sorted({
        str(item.get('init_mode'))
        for item in records
        if item and item.get('init_mode') is not None
    })
    observer_modes = sorted({
        str(item.get('rollout_lswt_observer_mode'))
        for item in records
        if item and item.get('rollout_lswt_observer_mode') is not None
    })
    initializer_modes = sorted({
        str(item.get('zero_profile_initializer'))
        for item in records
        if item and item.get('zero_profile_initializer') is not None
    })
    spinup_observer_modes = sorted({
        str(item.get('spinup_lswt_observer_mode'))
        for item in records
        if item and item.get('spinup_lswt_observer_mode') is not None
    })
    record[f'{base}_init_mode'] = ';'.join(init_modes) if init_modes else 'prior_spinup'
    record[f'{base}_rollout_mode'] = 'free'
    record[f'{base}_initializer'] = (
        ';'.join(initializer_modes) if initializer_modes else DEFAULT_ZERO_PROFILE_INITIALIZER_MODE
    )
    record[f'{base}_spinup_lswt_observer_mode'] = (
        ';'.join(spinup_observer_modes)
        if spinup_observer_modes
        else DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE
    )
    record[f'{base}_rollout_lswt_observer_mode'] = (
        ';'.join(observer_modes) if observer_modes else 'off'
    )
    record[f'{base}_profile_input_at_inference'] = False
    record[f'{base}_mean_rmse'] = _mean_numeric_records(records, 'rmse')
    record[f'{base}_mean_bias'] = _mean_numeric_records(records, 'bias')
    record[f'{base}_mean_mae'] = _mean_numeric_records(records, 'mae')
    record[f'{base}_profile_count'] = _sum_numeric_records(records, 'n_profiles')
    record[f'{base}_surface_rmse'] = _mean_numeric_records(records, 'observed_point_surface_rmse')
    record[f'{base}_surface_bias'] = _mean_numeric_records(records, 'observed_point_surface_bias')
    record[f'{base}_surface_count'] = _sum_numeric_records(records, 'observed_point_surface_count')
    record[f'{base}_mean_rmse_le25m'] = _mean_numeric_records(records, 'rmse_le25m')
    record[f'{base}_mean_bias_le25m'] = _mean_numeric_records(records, 'bias_le25m')
    record[f'{base}_count_le25m'] = _sum_numeric_records(records, 'count_le25m')
    record[f'{base}_mean_rmse_gt25m'] = _mean_numeric_records(records, 'rmse_gt25m')
    record[f'{base}_mean_bias_gt25m'] = _mean_numeric_records(records, 'bias_gt25m')
    record[f'{base}_count_gt25m'] = _sum_numeric_records(records, 'count_gt25m')
    record[f'{base}_post_spinup_mean_rmse'] = _mean_numeric_records(records, 'post_spinup_rmse')
    record[f'{base}_post_spinup_mean_bias'] = _mean_numeric_records(records, 'post_spinup_bias')
    record[f'{base}_lswt_observer_update_count'] = _sum_numeric_records(
        records,
        'lswt_observer_update_count',
    )
    record[f'{base}_lswt_observer_quality_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_quality_mean',
    )
    record[f'{base}_lswt_observer_surface_innovation_mean_c'] = _mean_numeric_records(
        records,
        'lswt_observer_surface_innovation_mean_c',
    )
    record[f'{base}_lswt_observer_mean_abs_delta_c'] = _mean_numeric_records(
        records,
        'lswt_observer_mean_abs_delta_c',
    )
    record[f'{base}_lswt_observer_max_abs_delta_c'] = _mean_numeric_records(
        records,
        'lswt_observer_max_abs_delta_c',
    )
    record[f'{base}_lswt_observer_heat_content_delta_mean_c'] = _mean_numeric_records(
        records,
        'lswt_observer_heat_content_delta_mean_c',
    )
    record[f'{base}_lswt_observer_deep_abs_delta_mean_c'] = _mean_numeric_records(
        records,
        'lswt_observer_deep_abs_delta_mean_c',
    )
    record[f'{base}_lswt_observer_density_guard_scale_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_density_guard_scale_mean',
    )
    record[f'{base}_lswt_observer_filled_lst_used_count'] = _sum_numeric_records(
        records,
        'lswt_observer_filled_lst_used_count',
    )
    record[f'{base}_lswt_observer_kalman_gain_surface_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_kalman_gain_surface_mean',
    )
    record[f'{base}_lswt_observer_kalman_gain_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_kalman_gain_mean',
    )
    record[f'{base}_lswt_observer_observation_error_mean_c'] = _mean_numeric_records(
        records,
        'lswt_observer_observation_error_mean_c',
    )
    record[f'{base}_lswt_observer_state_variance_surface_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_state_variance_surface_mean',
    )
    record[f'{base}_lswt_observer_localization_depth_mean_m'] = _mean_numeric_records(
        records,
        'lswt_observer_localization_depth_mean_m',
    )
    record[f'{base}_lswt_observer_reservoir_conservative_scale_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_reservoir_conservative_scale_mean',
    )
    record[f'{base}_lswt_observer_heat_content_bound_scale_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_heat_content_bound_scale_mean',
    )
    record[f'{base}_lswt_observer_mld_depth_mean_m'] = _mean_numeric_records(
        records,
        'lswt_observer_mld_depth_mean_m',
    )
    record[f'{base}_lswt_observer_mld_weight_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_mld_weight_mean',
    )
    record[f'{base}_lswt_observer_mld_heat_content_delta_mean_c'] = _mean_numeric_records(
        records,
        'lswt_observer_mld_heat_content_delta_mean_c',
    )
    record[f'{base}_lswt_observer_mld_volume_fraction_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_mld_volume_fraction_mean',
    )
    record[f'{base}_lswt_observer_mld_surface_to_heat_gain_mean'] = _mean_numeric_records(
        records,
        'lswt_observer_mld_surface_to_heat_gain_mean',
    )
    record[f'{base}_point_diagnostics_count'] = _sum_numeric_records(
        records,
        'free_roll_point_diagnostics_count',
    )
    record[f'{base}_age_summary_count'] = _sum_numeric_records(
        records,
        'free_roll_age_summary_count',
    )
    record[f'{base}_point_diagnostics_csvs'] = _joined_artifact_paths(
        records,
        'free_roll_point_diagnostics_csv',
    )
    record[f'{base}_age_summary_csvs'] = _joined_artifact_paths(
        records,
        'free_roll_age_summary_csv',
    )
    for bucket in ('natural', 'reservoir'):
        bucket_records = _metric_records_by_lake_bucket(metrics_by_lake, lakes, bucket)
        bucket_base = f'{prefix}_{bucket}_zero_profile_export'
        record[f'{bucket_base}_lake_count'] = len(bucket_records)
        record[f'{bucket_base}_rmse'] = _mean_numeric_records(bucket_records, 'rmse')
        record[f'{bucket_base}_bias'] = _mean_numeric_records(bucket_records, 'bias')
        record[f'{bucket_base}_mae'] = _mean_numeric_records(bucket_records, 'mae')
        record[f'{bucket_base}_profiles'] = _sum_numeric_records(bucket_records, 'n_profiles')
        record[f'{bucket_base}_surface_rmse'] = _mean_numeric_records(
            bucket_records,
            'observed_point_surface_rmse',
        )
        record[f'{bucket_base}_surface_bias'] = _mean_numeric_records(
            bucket_records,
            'observed_point_surface_bias',
        )
        record[f'{bucket_base}_surface_count'] = _sum_numeric_records(
            bucket_records,
            'observed_point_surface_count',
        )
        for band in DEPTH_RMSE_BANDS:
            record[f'{bucket_base}_rmse_{band}'] = _mean_numeric_records(
                bucket_records,
                f'rmse_{band}',
            )
            record[f'{bucket_base}_bias_{band}'] = _mean_numeric_records(
                bucket_records,
                f'bias_{band}',
            )
            record[f'{bucket_base}_count_{band}'] = _sum_numeric_records(
                bucket_records,
                f'count_{band}',
            )


def _filter_state_forecaster_state_dict_for_load(model, state_dict):
    """Keep grid buffers local and skip adaptive-head tensors by shape."""
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
        if (
            key.startswith('zero_profile_init_head.')
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


def _sum_numeric_records(records, key):
    values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
    finite = np.isfinite(values)
    return int(np.sum(values[finite])) if np.any(finite) else 0


def _joined_artifact_paths(records, key):
    paths = []
    for record in records:
        value = record.get(key)
        if value:
            paths.append(str(value))
    return ';'.join(paths)


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


def _build_target_tensor_matrix(df, lookup, masks, depths, *, device, date_to_index=None):
    date_to_index = date_to_index or _date_index_map(df)
    n_days = int(len(df))
    n_depths = int(np.asarray(depths).reshape(-1).size)
    profiles = torch.full((n_days, n_depths), float('nan'), dtype=torch.float32, device=device)
    mask_matrix = torch.zeros((n_days, n_depths), dtype=torch.bool, device=device)
    available_indices = []
    for date, profile in lookup.items():
        date_key = pd.Timestamp(date).normalize()
        day_idx = date_to_index.get(date_key)
        if day_idx is None:
            continue
        day_idx = int(day_idx)
        profile_tensor = torch.as_tensor(
            profile,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if int(profile_tensor.numel()) != n_depths:
            raise ValueError('profile lookup depth count does not match target matrix depth count.')
        raw_mask = masks.get(date_key)
        if raw_mask is None:
            mask_tensor = torch.ones(n_depths, dtype=torch.bool, device=device)
        else:
            mask_tensor = torch.as_tensor(raw_mask, dtype=torch.bool, device=device).reshape(-1)
            if int(mask_tensor.numel()) != n_depths:
                raise ValueError('profile mask depth count does not match target matrix depth count.')
        profiles[day_idx].copy_(profile_tensor)
        mask_matrix[day_idx].copy_(mask_tensor)
        available_indices.append(day_idx)
    return {
        'profiles': profiles.contiguous(),
        'masks': mask_matrix.contiguous(),
        'available_indices': frozenset(available_indices),
        'available_count': int(len(set(available_indices))),
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
        stacked = torch.stack(values, dim=0).contiguous()
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        matrix[key] = stacked
    return matrix


def _forcing_rows_from_matrix(forcing_tensors, length):
    if not forcing_tensors:
        return ()
    return ForcingRowSequence(forcing_tensors, length=length)


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
    store = lake.get('packed_tensor_store')
    if store is not None and store.forcing_tensors:
        return store.forcing_batch(index_tensor)
    matrix = lake.get('forcing_tensors')
    if not matrix:
        raise ValueError('forcing tensor cache is not available for this lake.')
    forcing_rows = lake.get('forcing_rows')
    if hasattr(forcing_rows, 'batch'):
        return forcing_rows.batch(index_tensor)
    return ForcingBatch(matrix, index_tensor)


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
    cached_batch = _stack_forcing_rows_from_shared_cache(rows)
    if cached_batch is not None:
        return cached_batch
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(rows):
            continue
        stacked = torch.stack(values, dim=0).contiguous()
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        batch[key] = stacked
    return batch


def _stack_forcing_rows_from_shared_cache(rows):
    if not rows:
        return None
    if not all(isinstance(row, ForcingBatch) and row.index is not None for row in rows):
        return None
    data = rows[0].data
    index_device = rows[0].index.device
    index_tensors = []
    for row in rows:
        if row.data is not data or row.index is None or int(row.index.numel()) != 1:
            return None
        index_tensors.append(row.index.reshape(-1).to(device=index_device))
    return ForcingBatch(data, torch.cat(index_tensors, dim=0), cache_selects=True)


def _stack_forcing_rows(rows):
    if not rows:
        raise ValueError('forcing rows must not be empty.')
    cached_batch = _stack_forcing_rows_from_shared_cache(rows)
    if cached_batch is not None:
        return cached_batch
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(rows):
            continue
        stacked = torch.stack(values, dim=0).contiguous()
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.reshape(stacked.shape[0])
        batch[key] = stacked
    return batch


def _stack_forcing_tensor_batches_for_items(items, day_indices):
    if len(items) != len(day_indices):
        raise ValueError('forcing item count must match day index count.')
    if not items:
        raise ValueError('forcing items must not be empty.')
    first_matrix = items[0][1].get('forcing_tensors')
    if not first_matrix:
        return None
    first_lake = items[0][1]
    if all(item[1] is first_lake for item in items):
        return _forcing_batch_view(first_lake, day_indices)
    keys = tuple(sorted(first_matrix.keys()))
    for item, _day_idx in zip(items, day_indices):
        matrix = item[1].get('forcing_tensors')
        if not matrix or tuple(sorted(matrix.keys())) != keys:
            return None

    grouped = {}
    for pos, (item, day_idx) in enumerate(zip(items, day_indices)):
        lake = item[1]
        entry = grouped.setdefault(id(lake), {'lake': lake, 'positions': [], 'days': []})
        entry['positions'].append(pos)
        entry['days'].append(int(day_idx))
    for group in grouped.values():
        sample_value = next(iter(group['lake']['forcing_tensors'].values()))
        group['day_tensor'] = torch.as_tensor(
            group['days'],
            dtype=torch.long,
            device=sample_value.device,
        )
        group['position_tensor'] = torch.as_tensor(
            group['positions'],
            dtype=torch.long,
            device=sample_value.device,
        )

    batch = {}
    for key in keys:
        output = None
        for group in grouped.values():
            matrix = group['lake']['forcing_tensors']
            value = matrix[key]
            index_tensor = group['day_tensor']
            if index_tensor.device != value.device:
                index_tensor = index_tensor.to(device=value.device)
            selected = value.index_select(0, index_tensor)
            if selected.ndim == 2 and selected.shape[1] == 1:
                selected = selected.reshape(selected.shape[0])
            if output is None:
                output = torch.empty(
                    (len(items), *selected.shape[1:]),
                    device=selected.device,
                    dtype=selected.dtype,
                )
            position_tensor = group['position_tensor']
            if position_tensor.device != output.device:
                position_tensor = position_tensor.to(device=output.device)
            output.index_copy_(0, position_tensor, selected.to(device=output.device, dtype=output.dtype))
        batch[key] = output
    return ForcingBatch(batch)


def _target_tensor_and_mask_batch(lake, preferred_split, dates):
    dates = list(dates)
    selections = []
    for date_value in dates:
        matrix_entry, day_idx = _target_matrix_for_date(lake, preferred_split, date_value)
        if matrix_entry is None:
            selections = []
            break
        selections.append((matrix_entry, int(day_idx)))
    if selections:
        first_entry = selections[0][0]
        if all(entry is first_entry for entry, _day_idx in selections):
            index_tensor = torch.as_tensor(
                [day_idx for _entry, day_idx in selections],
                dtype=torch.long,
                device=first_entry['profiles'].device,
            )
            return (
                first_entry['profiles'].index_select(0, index_tensor),
                first_entry['masks'].index_select(0, index_tensor),
            )
        first_profiles = first_entry['profiles']
        batch_profiles = torch.empty(
            (len(selections), first_profiles.shape[1]),
            device=first_profiles.device,
            dtype=first_profiles.dtype,
        )
        batch_masks = torch.empty(
            (len(selections), first_entry['masks'].shape[1]),
            device=first_entry['masks'].device,
            dtype=torch.bool,
        )
        grouped = {}
        for pos, (entry, day_idx) in enumerate(selections):
            group = grouped.setdefault(id(entry), {'entry': entry, 'positions': [], 'days': []})
            group['positions'].append(pos)
            group['days'].append(day_idx)
        for group in grouped.values():
            entry = group['entry']
            index_tensor = torch.as_tensor(
                group['days'],
                dtype=torch.long,
                device=entry['profiles'].device,
            )
            position_tensor = torch.as_tensor(
                group['positions'],
                dtype=torch.long,
                device=batch_profiles.device,
            )
            selected_profiles = entry['profiles'].index_select(0, index_tensor)
            selected_masks = entry['masks'].index_select(0, index_tensor.to(device=entry['masks'].device))
            batch_profiles.index_copy_(
                0,
                position_tensor,
                selected_profiles.to(device=batch_profiles.device, dtype=batch_profiles.dtype),
            )
            batch_masks.index_copy_(
                0,
                position_tensor.to(device=batch_masks.device),
                selected_masks.to(device=batch_masks.device),
            )
        return batch_profiles, batch_masks

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


def _segment_rollout_forcing_sequence(lake, start_index_tensor, length):
    """Pre-build same-lake forcing batches for a fixed-length rollout chunk."""
    length = int(length)
    if length <= 0:
        return (), ()
    start_index_tensor = torch.as_tensor(
        start_index_tensor,
        dtype=torch.long,
        device=lake['depths'].device,
    ).reshape(-1)
    current_rows = []
    next_rows = []
    for offset in range(length):
        day_indices = start_index_tensor + int(offset)
        current_rows.append(_forcing_row_batch(lake, day_indices))
        next_rows.append(_forcing_row_batch(lake, day_indices + 1))
    return tuple(current_rows), tuple(next_rows)


def _segment_rollout_forcing_sequence_for_items(items, start_indices, length):
    """Pre-build cross-lake forcing batches for a fixed-length rollout chunk."""
    length = int(length)
    if length <= 0:
        return (), ()
    start_indices = [int(value) for value in start_indices]
    current_rows = []
    next_rows = []
    for offset in range(length):
        day_indices = [start_idx + int(offset) for start_idx in start_indices]
        current_rows.append(_stack_forcing_batch_for_items(items, day_indices))
        next_rows.append(_stack_forcing_batch_for_items(
            items,
            [day_idx + 1 for day_idx in day_indices],
        ))
    return tuple(current_rows), tuple(next_rows)


def _segment_rollout_target_plan_for_lake(
    lake,
    lookup_split,
    active_targets,
    start_indices,
    last_gap,
):
    """Pre-fetch target tensors on the fixed rollout time axis."""
    device = lake['depths'].device
    plans = []
    for offset in range(int(last_gap)):
        active_indices = []
        active_dates = []
        active_prediction_indices = []
        target_gap_days = []
        for sample_idx, start_idx in enumerate(start_indices):
            prediction_idx = int(start_idx + offset + 1)
            target_date = active_targets[sample_idx].get(prediction_idx)
            if target_date is None:
                continue
            active_indices.append(sample_idx)
            active_dates.append(target_date)
            active_prediction_indices.append(prediction_idx)
            target_gap_days.append(int(prediction_idx - start_idx))
        if not active_indices:
            plans.append(None)
            continue
        target, target_mask = _target_tensor_and_mask_batch(lake, lookup_split, active_dates)
        horizon_weight = torch.as_tensor(
            [min(1.0 + float(gap) / 30.0, 3.0) for gap in target_gap_days],
            dtype=target.dtype,
            device=device,
        )
        plans.append({
            'active_indices': active_indices,
            'active_index_tensor': torch.as_tensor(active_indices, dtype=torch.long, device=device),
            'active_dates': active_dates,
            'active_prediction_indices': active_prediction_indices,
            'target_gap_days': target_gap_days,
            'horizon_weight': horizon_weight,
            'target': target,
            'target_mask': target_mask,
        })
    return tuple(plans)


def _segment_rollout_target_plan_for_items(
    items,
    lookup_split,
    active_targets,
    start_indices,
    last_gap,
):
    """Pre-fetch cross-lake target tensors on the fixed rollout time axis."""
    ref_lake = items[0][1]
    device = ref_lake['depths'].device
    plans = []
    for offset in range(int(last_gap)):
        active_positions = []
        active_dates = []
        active_prediction_indices = []
        target_gap_days = []
        for sample_idx, start_idx in enumerate(start_indices):
            prediction_idx = int(start_idx + offset + 1)
            target_date = active_targets[sample_idx].get(prediction_idx)
            if target_date is None:
                continue
            active_positions.append(sample_idx)
            active_dates.append(target_date)
            active_prediction_indices.append(prediction_idx)
            target_gap_days.append(int(prediction_idx - start_idx))
        if not active_positions:
            plans.append(None)
            continue
        active_items = [
            (items[sample_idx][0], items[sample_idx][1], active_dates[pos])
            for pos, sample_idx in enumerate(active_positions)
        ]
        target, target_mask = _stack_target_batch_for_items(
            active_items,
            lookup_split,
            lambda item: item[2],
        )
        horizon_weight = torch.as_tensor(
            [min(1.0 + float(gap) / 30.0, 3.0) for gap in target_gap_days],
            dtype=target.dtype,
            device=device,
        )
        plans.append({
            'active_positions': active_positions,
            'active_index_tensor': torch.as_tensor(active_positions, dtype=torch.long, device=device),
            'active_dates': active_dates,
            'active_prediction_indices': active_prediction_indices,
            'target_gap_days': target_gap_days,
            'horizon_weight': horizon_weight,
            'target': target,
            'target_mask': target_mask,
        })
    return tuple(plans)


def _masked_huber_profile_loss_per_sample(prediction, target, mask=None, delta=2.0):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and prediction.shape[0] > 1:
        target = target.expand(prediction.shape[0], -1)
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.reshape(1, -1).expand_as(valid)
        elif mask_tensor.shape[0] == 1 and prediction.shape[0] > 1:
            mask_tensor = mask_tensor.expand_as(valid)
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


def _masked_rmse_profile_loss_per_sample(prediction, target, mask=None):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and prediction.shape[0] > 1:
        target = target.expand(prediction.shape[0], -1)
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.reshape(1, -1).expand_as(valid)
        elif mask_tensor.shape[0] == 1 and prediction.shape[0] > 1:
            mask_tensor = mask_tensor.expand_as(valid)
        valid = valid & mask_tensor.reshape_as(valid)
    valid_float = valid.to(dtype=prediction.dtype)
    counts = valid_float.sum(dim=1)
    squared = (prediction - target).pow(2) * valid_float
    mean_squared = squared.sum(dim=1) / torch.clamp(counts, min=1.0)
    return torch.where(
        counts > 0,
        torch.sqrt(mean_squared + 1.0e-12),
        torch.zeros_like(mean_squared),
    )


def _masked_rmse_profile_loss_per_sample_with_depth_mask(prediction, target, mask, depth_mask):
    depth_mask = torch.as_tensor(depth_mask, dtype=torch.bool, device=prediction.device).reshape(1, -1)
    if mask is None:
        active_mask = depth_mask
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.reshape(1, -1)
        active_mask = mask_tensor & depth_mask
    losses = _masked_rmse_profile_loss_per_sample(prediction, target, active_mask)
    valid = active_mask
    if valid.shape[0] == 1 and prediction.shape[0] > 1:
        valid = valid.expand(prediction.shape[0], -1)
    valid = valid & torch.isfinite(prediction)
    target_tensor = target.to(device=prediction.device, dtype=prediction.dtype)
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(0)
    if target_tensor.shape[0] == 1 and prediction.shape[0] > 1:
        target_tensor = target_tensor.expand(prediction.shape[0], -1)
    valid = valid & torch.isfinite(target_tensor)
    return losses, valid.to(dtype=prediction.dtype).sum(dim=1)


def _zero_profile_init_net_depth_band_losses(prediction, target, mask, depths):
    depths = torch.as_tensor(depths, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    band_specs = (
        ('surface', depths <= 1.0),
        ('upper', (depths > 1.0) & (depths <= 5.0)),
        ('mid', (depths > 5.0) & (depths <= 25.0)),
        ('deep', depths > 25.0),
    )
    losses = {}
    counts = {}
    active_losses = []
    active_counts = []
    for name, depth_mask in band_specs:
        band_loss, band_count = _masked_rmse_profile_loss_per_sample_with_depth_mask(
            prediction,
            target,
            mask,
            depth_mask,
        )
        losses[name] = band_loss
        counts[name] = band_count
        active_losses.append(band_loss)
        active_counts.append(band_count)
    stacked_loss = torch.stack(active_losses, dim=1)
    stacked_count = torch.stack(active_counts, dim=1)
    active = stacked_count > 0
    active_float = active.to(dtype=prediction.dtype)
    band_mean = torch.where(
        active.any(dim=1),
        (stacked_loss * active_float).sum(dim=1) / torch.clamp(active_float.sum(dim=1), min=1.0),
        torch.zeros((prediction.shape[0],), device=prediction.device, dtype=prediction.dtype),
    )
    return band_mean, losses, counts


def _profile_physics_loss_per_sample(profile):
    return torch.stack([
        _profile_physics_loss(profile[idx: idx + 1])
        for idx in range(profile.shape[0])
    ])


def _huber_loss_vector(error, delta):
    zero = torch.zeros_like(error)
    return torch.nn.functional.huber_loss(
        error,
        zero,
        delta=float(delta),
        reduction='none',
    )


def _surface_bottom_diff_c_vector(profile, mask=None):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    batch_size, n_depths = profile.shape
    device = profile.device
    dtype = profile.dtype
    if n_depths <= 1:
        return (
            torch.zeros((batch_size,), device=device, dtype=dtype),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
        )
    if mask is None:
        mask_tensor = torch.ones((batch_size, n_depths), dtype=torch.bool, device=device)
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(batch_size, n_depths)
    finite = torch.isfinite(profile)
    valid = mask_tensor[:, 0] & mask_tensor[:, -1] & finite[:, 0] & finite[:, -1]
    safe_profile = torch.where(finite, profile, torch.zeros_like(profile))
    return safe_profile[:, 0] - safe_profile[:, -1], valid


def _zero_profile_init_state_physics_loss_per_sample(
    profile,
    target,
    target_mask,
    lake,
    *,
    min_full_column_coverage=DEFAULT_HEAT_CONTENT_FULL_COLUMN_MIN_COVERAGE,
):
    """Physics-shaped state constraints for supervised zero-profile init-net states."""
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    target = target.to(device=profile.device, dtype=profile.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    batch_size, n_depths = profile.shape
    if target.shape[0] == 1 and batch_size > 1:
        target = target.expand(batch_size, -1)
    if target_mask is None:
        mask = torch.ones((batch_size, n_depths), dtype=torch.bool, device=profile.device)
    else:
        mask = torch.as_tensor(target_mask, dtype=torch.bool, device=profile.device).reshape(batch_size, n_depths)
    finite_target = torch.isfinite(target)
    mask = mask & finite_target
    safe_target = torch.where(finite_target, target, torch.zeros_like(target))
    area = lake.get('area', lake.get('area_profile'))
    if area is None:
        area = torch.ones_like(lake['depths'])

    pred_column, pred_column_valid = _column_mean_temperature_c_vector(
        profile,
        lake['depths'],
        area,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    target_column, target_column_valid = _column_mean_temperature_c_vector(
        safe_target,
        lake['depths'],
        area,
        mask=mask,
        layer_weights=lake.get('heat_content_layer_weights'),
        min_full_column_coverage=min_full_column_coverage,
    )
    column_valid = pred_column_valid & target_column_valid
    zero = torch.zeros((batch_size,), device=profile.device, dtype=profile.dtype)
    heat_content_loss = torch.where(
        column_valid,
        _huber_loss_vector(pred_column - target_column, delta=2.0) / 4.0,
        zero,
    )

    pred_surface_bottom, _pred_endpoint_valid = _surface_bottom_diff_c_vector(profile)
    target_surface_bottom, target_endpoint_valid = _surface_bottom_diff_c_vector(safe_target, mask=mask)
    surface_bottom_loss = torch.where(
        target_endpoint_valid,
        _huber_loss_vector(pred_surface_bottom - target_surface_bottom, delta=3.0) / 9.0,
        zero,
    )

    column_range_loss = (
        torch.relu(-pred_column).pow(2)
        + torch.relu(pred_column - 32.0).pow(2)
    ) / (32.0 ** 2)
    surface_bottom_range_loss = torch.relu(torch.abs(pred_surface_bottom) - 25.0).pow(2) / (25.0 ** 2)
    bounded_state_loss = column_range_loss + surface_bottom_range_loss
    state_loss = heat_content_loss + 0.5 * surface_bottom_loss + bounded_state_loss
    return state_loss, {
        'heat_content_constraint_loss': heat_content_loss,
        'surface_bottom_constraint_loss': surface_bottom_loss,
        'bounded_state_loss': bounded_state_loss,
    }


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
    residual_surface = diagnostics.get('residual_surface_abs_mean_c')
    if residual_surface is None:
        residual_surface = diagnostics['residual_surface_c'].abs()
    residual_surface = residual_surface.reshape(-1)
    residual_deep = diagnostics.get('residual_deep_abs_mean_c')
    if residual_deep is None:
        residual_deep = diagnostics['residual_deep_mean_c'].abs()
    residual_deep = residual_deep.reshape(-1)
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


def _kd_saturation_penalty_loss_per_sample(diagnostics, threshold):
    kd_multiplier = diagnostics.get('nn_kd_multiplier')
    if kd_multiplier is None:
        return torch.zeros_like(diagnostics['residual_abs_mean_c'].reshape(-1))
    return torch.relu(kd_multiplier.reshape(-1) - float(threshold)).pow(2)


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
        'residual_abs_mean_c': diagnostics.get(
            'residual_abs_mean_c',
            zeros,
        ).reshape(-1)[sample_idx].detach(),
        'residual_surface_abs_mean_c': diagnostics.get(
            'residual_surface_abs_mean_c',
            diagnostics.get('residual_surface_c', zeros).abs(),
        ).reshape(-1)[sample_idx].detach(),
        'residual_deep_abs_mean_c': diagnostics.get(
            'residual_deep_abs_mean_c',
            diagnostics.get('residual_deep_mean_c', zeros).abs(),
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


def _no_profile_lst_surface_loss_per_sample(
    prediction,
    row,
    *,
    lst_qc_min=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
    open_water_only=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
):
    batch_size = int(prediction.shape[0])
    device = prediction.device
    dtype = prediction.dtype
    target = _forcing_value_vector(
        row,
        'lswt_open_water',
        batch_size,
        device=device,
        dtype=dtype,
        default=float('nan'),
    )
    fallback_target = _forcing_value_vector(
        row,
        'lst_surface',
        batch_size,
        device=device,
        dtype=dtype,
        default=float('nan'),
    )
    target = torch.where(torch.isfinite(target), target, fallback_target)
    quality = _forcing_value_vector(
        row,
        'lst_quality',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    observed_flag = _forcing_value_vector(
        row,
        'lst_observed_flag',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    ice_mask = _forcing_value_vector(
        row,
        'ice_mask',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    ice_fraction = _forcing_value_vector(
        row,
        'ice_fraction',
        batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    open_water_only = _normalize_on_off(open_water_only, name='no_profile_lst_surface_open_water_only')
    ice_level = torch.maximum(
        torch.clamp(ice_mask, 0.0, 1.0),
        torch.clamp(ice_fraction, 0.0, 1.0),
    )
    target_valid = torch.isfinite(target)
    valid = (
        target_valid
        & (observed_flag >= 0.5)
        & (quality >= float(lst_qc_min))
    )
    if open_water_only == 'on':
        valid = valid & (ice_level <= 0.05)
    target_safe = torch.where(valid, target, prediction[:, 0].detach())
    quality_weight = torch.where(valid, torch.clamp(quality, 0.0, 1.0), torch.zeros_like(quality))
    loss_vec = quality_weight * torch.nn.functional.huber_loss(
        prediction[:, 0],
        target_safe,
        delta=2.0,
        reduction='none',
    )
    return loss_vec, quality_weight, valid


def _init_masked_vector_stats(batch_size, *, device, dtype):
    zeros = torch.zeros(int(batch_size), device=device, dtype=dtype)
    return zeros.clone(), zeros.clone(), zeros.clone()


def _update_masked_vector_stats(loss_sum, weight_sum, count, loss_vec, weight_vec, mask):
    mask_f = mask.to(device=loss_vec.device, dtype=loss_vec.dtype).reshape(-1)
    return (
        loss_sum + loss_vec.reshape(-1) * mask_f,
        weight_sum + weight_vec.reshape(-1) * mask_f,
        count + mask_f,
    )


def _masked_mean_from_sum(value_sum, count):
    return value_sum / torch.clamp(count, min=1.0)


def _batch_chunks(items, batch_size):
    batch_size = int(batch_size or 0)
    if batch_size <= 0:
        yield list(items)
        return
    for start in range(0, len(items), batch_size):
        yield list(items[start: start + batch_size])


def _normalize_on_off(value, *, name=None, field_name=None):
    name = name or field_name or 'value'
    text = str(value or 'off').strip().lower().replace('-', '_')
    if text not in {'off', 'on'}:
        raise ValueError(f'{name} must be one of: off, on.')
    return text


def _normalize_profile_loss_target_mode(value):
    text = str(value or DEFAULT_PROFILE_LOSS_TARGET_MODE).strip().lower().replace('-', '_')
    if text not in PROFILE_LOSS_TARGET_MODES:
        allowed = ', '.join(sorted(PROFILE_LOSS_TARGET_MODES))
        raise ValueError(f'profile_loss_target_mode must be one of: {allowed}.')
    return text


def _normalize_profile_sampling_mode(value):
    text = str(value or DEFAULT_PROFILE_SAMPLING_MODE).strip().lower().replace('-', '_')
    if text not in PROFILE_SAMPLING_MODES:
        allowed = ', '.join(sorted(PROFILE_SAMPLING_MODES))
        raise ValueError(f'profile_sampling_mode must be one of: {allowed}.')
    return text


def _season_balanced_sample(items, samples_per_lake, epoch, *, date_getter, index_getter):
    items = list(items)
    samples_per_lake = int(samples_per_lake)
    if samples_per_lake <= 0 or len(items) <= samples_per_lake:
        return items
    season_order = ('winter', 'spring', 'summer', 'fall')
    by_season = {season: [] for season in season_order}
    for item in items:
        date_value = pd.Timestamp(date_getter(item)).normalize()
        season = _season_name(date_value.month)
        by_season.setdefault(season, []).append(item)
    for season in by_season:
        by_season[season] = sorted(by_season[season], key=index_getter)
    active_seasons = [season for season in season_order if by_season.get(season)]
    if not active_seasons:
        return []

    selected = []
    selected_ids = set()
    epoch = int(epoch)
    season_offset = epoch % len(active_seasons)
    season_pick_counts = {season: 0 for season in active_seasons}
    while len(selected) < samples_per_lake:
        progressed = False
        for offset in range(len(active_seasons)):
            if len(selected) >= samples_per_lake:
                break
            season = active_seasons[(season_offset + offset) % len(active_seasons)]
            candidates = [
                item for item in by_season[season]
                if id(item) not in selected_ids
            ]
            if not candidates:
                continue
            pick_idx = (epoch + season_pick_counts[season]) % len(candidates)
            chosen = candidates[pick_idx]
            selected.append(chosen)
            selected_ids.add(id(chosen))
            season_pick_counts[season] += 1
            progressed = True
        if not progressed:
            break
    return sorted(selected, key=index_getter)


def _normalize_training_amp(value):
    text = str(value or DEFAULT_TRAINING_AMP).strip().lower().replace('-', '_')
    if text not in TRAINING_AMP_MODES:
        raise ValueError('training_amp must be one of: off, bf16, fp16.')
    return text


def _training_amp_enabled(device, training_amp):
    return (
        _normalize_training_amp(training_amp) != 'off'
        and isinstance(device, torch.device)
        and device.type == 'cuda'
        and torch.cuda.is_available()
    )


def _training_amp_dtype(training_amp):
    mode = _normalize_training_amp(training_amp)
    if mode == 'fp16':
        return torch.float16
    if mode == 'bf16':
        return torch.bfloat16
    return torch.float32


def _training_autocast_context(device, training_amp):
    if not _training_amp_enabled(device, training_amp):
        return contextlib.nullcontext()
    dtype = _training_amp_dtype(training_amp)
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def _make_training_grad_scaler(device, training_amp):
    enabled = _training_amp_enabled(device, training_amp) and _normalize_training_amp(training_amp) == 'fp16'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _clip_grad_norm_finite(parameters, max_norm):
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return torch.tensor(0.0), True
    total_norm = torch.nn.utils.clip_grad_norm_(
        params,
        float(max_norm),
        error_if_nonfinite=False,
    )
    finite = bool(torch.isfinite(total_norm.detach()).cpu().item())
    if not finite:
        for param in params:
            param.grad = None
    return total_norm, finite


def _scalar_tensor_to_float(value, default=np.nan):
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().reshape(-1)
        if int(tensor.numel()) == 0:
            return float(default)
        return float(tensor[0].cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _training_runtime_config_fields(*, training_amp, training_amp_enabled, training_history_detail_every_epochs):
    return {
        'training_amp': str(training_amp),
        'training_amp_enabled': bool(training_amp_enabled),
        'training_history_detail_every_epochs': int(training_history_detail_every_epochs),
    }


def _resolve_gpu_batch_autotune(
    *,
    gpu_batch_autotune,
    gpu_batch_autotune_target_batch_size,
    transition_batch_size,
    segment_rollout_batch_size,
    rolling_horizon_batch_size,
    unlabeled_heat_closure_batch_size,
    cross_lake_batch_mode,
    cross_lake_batch_size,
):
    mode = _normalize_on_off(gpu_batch_autotune, name='gpu_batch_autotune')
    target = max(1, int(gpu_batch_autotune_target_batch_size or 1))
    resolved = {
        'gpu_batch_autotune': mode,
        'gpu_batch_autotune_target_batch_size': int(target),
        'transition_batch_size': int(transition_batch_size),
        'segment_rollout_batch_size': int(segment_rollout_batch_size),
        'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
        'unlabeled_heat_closure_batch_size': int(unlabeled_heat_closure_batch_size),
        'cross_lake_batch_mode': cross_lake_batch_mode,
        'cross_lake_batch_size': int(cross_lake_batch_size),
        'gpu_batch_autotune_applied': False,
    }
    if mode == 'off':
        return resolved

    resolved['cross_lake_batch_mode'] = 'on'
    resolved['cross_lake_batch_size'] = max(int(cross_lake_batch_size), target)
    if int(transition_batch_size) > 0:
        resolved['transition_batch_size'] = max(int(transition_batch_size), target)
    resolved['segment_rollout_batch_size'] = max(int(segment_rollout_batch_size), max(1, target // 2))
    if int(rolling_horizon_batch_size) > 0:
        resolved['rolling_horizon_batch_size'] = max(int(rolling_horizon_batch_size), max(1, target // 2))
    if int(unlabeled_heat_closure_batch_size) > 0:
        resolved['unlabeled_heat_closure_batch_size'] = max(
            int(unlabeled_heat_closure_batch_size),
            max(1, target // 4),
        )
    resolved['gpu_batch_autotune_applied'] = True
    return resolved


def _gpu_batch_autotune_config_fields(
    *,
    gpu_batch_autotune,
    gpu_batch_autotune_target_batch_size,
    gpu_batch_autotune_applied=False,
):
    return {
        'gpu_batch_autotune': str(gpu_batch_autotune),
        'gpu_batch_autotune_target_batch_size': int(gpu_batch_autotune_target_batch_size),
        'gpu_batch_autotune_applied': bool(gpu_batch_autotune_applied),
    }


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


def _horizon_metric_record(errors_by_horizon, biases_by_horizon=None, prefix='rmse', depth_errors_by_horizon=None):
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
    if depth_errors_by_horizon is not None:
        for band in DEPTH_RMSE_BANDS:
            band_errors = depth_errors_by_horizon.get(band, {})
            for horizon in errors_by_horizon:
                values = np.asarray(band_errors.get(horizon, []), dtype=np.float64)
                finite = np.isfinite(values)
                record[f'{prefix}_{band}_{int(horizon)}d'] = (
                    float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else np.nan
                )
                record[f'count_{band}_{int(horizon)}d'] = int(np.sum(finite))
    return record


def _empty_horizon_tensor_stats(horizons, *, device, dtype=torch.float64):
    stats = {}
    for horizon in horizons:
        horizon = int(horizon)
        stats[horizon] = {
            'sse': torch.zeros((), device=device, dtype=dtype),
            'count': torch.zeros((), device=device, dtype=dtype),
            'bias_sum': torch.zeros((), device=device, dtype=dtype),
            'bias_count': torch.zeros((), device=device, dtype=dtype),
            'band_sse': {
                band: torch.zeros((), device=device, dtype=dtype)
                for band in DEPTH_RMSE_BANDS
            },
            'band_count': {
                band: torch.zeros((), device=device, dtype=dtype)
                for band in DEPTH_RMSE_BANDS
            },
        }
    return stats


def _depth_band_masks_tensor(depths, *, device):
    depths = depths.reshape(-1).to(device=device, dtype=torch.float32)
    finite_depths = torch.isfinite(depths)
    return {
        'le25m': finite_depths & (depths <= float(DEPTH_STRATIFIED_RMSE_BOUNDARY_M)),
        'gt25m': finite_depths & (depths > float(DEPTH_STRATIFIED_RMSE_BOUNDARY_M)),
    }


def _update_horizon_tensor_stats(stats, horizon, prediction, target, valid, depth_band_masks):
    horizon = int(horizon)
    if horizon not in stats:
        return
    stat = stats[horizon]
    dtype = stat['sse'].dtype
    valid = valid.reshape(-1).to(device=prediction.device, dtype=torch.bool)
    diff = prediction.reshape(-1).to(dtype=dtype) - target.reshape(-1).to(dtype=dtype)
    valid_f = valid.to(dtype=dtype)
    masked_diff = torch.where(valid, diff, torch.zeros_like(diff))
    masked_squared = masked_diff.pow(2)
    count = valid_f.sum()
    stat['sse'] = stat['sse'] + masked_squared.sum()
    stat['count'] = stat['count'] + count
    has_valid = (count > 0).to(dtype=dtype)
    stat['bias_sum'] = stat['bias_sum'] + masked_diff.sum() / torch.clamp(count, min=1.0) * has_valid
    stat['bias_count'] = stat['bias_count'] + has_valid
    for band in DEPTH_RMSE_BANDS:
        band_mask = depth_band_masks[band].reshape(-1).to(device=prediction.device)
        band_valid = valid & band_mask
        band_f = band_valid.to(dtype=dtype)
        stat['band_sse'][band] = stat['band_sse'][band] + (masked_squared * band_f).sum()
        stat['band_count'][band] = stat['band_count'][band] + band_f.sum()


def _horizon_metric_record_from_tensor_stats(stats, *, prefix='rmse'):
    record = {}
    for horizon in sorted(int(horizon) for horizon in stats):
        stat = stats[horizon]
        count = float(stat['count'].detach().cpu())
        key = f'{prefix}_{int(horizon)}d'
        record[key] = (
            float(np.sqrt(float(stat['sse'].detach().cpu()) / count))
            if count > 0.0 else np.nan
        )
        record[f'count_{int(horizon)}d'] = int(count)
        bias_count = float(stat['bias_count'].detach().cpu())
        record[f'bias_{int(horizon)}d'] = (
            float(stat['bias_sum'].detach().cpu()) / bias_count
            if bias_count > 0.0 else np.nan
        )
    for band in DEPTH_RMSE_BANDS:
        for horizon in sorted(int(horizon) for horizon in stats):
            stat = stats[horizon]
            count = float(stat['band_count'][band].detach().cpu())
            record[f'{prefix}_{band}_{int(horizon)}d'] = (
                float(np.sqrt(float(stat['band_sse'][band].detach().cpu()) / count))
                if count > 0.0 else np.nan
            )
            record[f'count_{band}_{int(horizon)}d'] = int(count)
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


def _depth_band_name(depth_m):
    depth_value = float(depth_m)
    if not np.isfinite(depth_value):
        return None
    if depth_value <= 1.0:
        return 'surface_le1m'
    if depth_value <= DEPTH_STRATIFIED_RMSE_BOUNDARY_M:
        return 'le25m'
    return 'gt25m'


def prepare_lake_state_data(
    lake_config,
    *,
    split_mode='time_blocked',
    task_mode='analysis',
    data_fill_mode='reconstruction',
    profile_loss_target_mode=DEFAULT_PROFILE_LOSS_TARGET_MODE,
    depth_points=40,
    max_rollout_days=90,
    segment_rollout_max_days=None,
    unlabeled_heat_closure_window_days=DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS,
    unlabeled_heat_closure_horizons=DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS,
    history_window_days=30,
    device='cpu',
):
    """Load one lake and build state-transition training/evaluation pairs."""
    if task_mode is None:
        task_mode = lake_config.get('task_mode')
    task_mode = normalize_task_mode(task_mode)
    data_fill_mode = normalize_data_fill_mode(data_fill_mode)
    profile_loss_target_mode = _normalize_profile_loss_target_mode(profile_loss_target_mode)
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
    train_lookup, train_masks = _profile_lookup(
        splits['train'],
        depths,
        return_masks=True,
        target_mode=profile_loss_target_mode,
    )
    val_lookup, val_masks = _profile_lookup(
        splits['val'],
        depths,
        return_masks=True,
        target_mode=profile_loss_target_mode,
    )
    all_lookup, all_masks = _profile_lookup(
        profile_obs,
        depths,
        return_masks=True,
        target_mode=profile_loss_target_mode,
    )
    date_to_index = _date_index_map(df)
    train_pairs = _build_rollout_pairs(df, train_lookup, max_rollout_days=max_rollout_days)
    val_pairs = _build_rollout_pairs(df, val_lookup, max_rollout_days=max_rollout_days)
    all_pairs = _build_rollout_pairs(df, all_lookup, max_rollout_days=max_rollout_days)
    segment_rollout_max_days = int(segment_rollout_max_days or max_rollout_days)
    train_segment_sequences = _build_segment_rollout_sequences(
        df,
        train_lookup,
        max_rollout_days=segment_rollout_max_days,
    )
    val_segment_sequences = _build_segment_rollout_sequences(
        df,
        val_lookup,
        max_rollout_days=segment_rollout_max_days,
    )
    all_segment_sequences = _build_segment_rollout_sequences(
        df,
        all_lookup,
        max_rollout_days=segment_rollout_max_days,
    )
    unlabeled_heat_closure_horizons = _parse_unlabeled_heat_closure_horizons(
        unlabeled_heat_closure_horizons,
        fallback_window_days=unlabeled_heat_closure_window_days,
    )
    unlabeled_heat_closure_windows_by_horizon = _build_unlabeled_heat_closure_windows_by_horizon(
        df,
        all_lookup,
        horizons=unlabeled_heat_closure_horizons,
        window_days=unlabeled_heat_closure_window_days,
    )
    unlabeled_heat_closure_windows = _build_unlabeled_heat_closure_windows(
        df,
        all_lookup,
        window_days=unlabeled_heat_closure_window_days,
        horizons=unlabeled_heat_closure_horizons,
    )
    raw_forcing_rows = _forcing_tensor_rows(
        df,
        device=device,
        history_window_days=history_window_days,
        task_mode=task_mode,
    )
    forcing_tensors = _forcing_tensor_matrix(raw_forcing_rows)
    forcing_rows = _forcing_rows_from_matrix(forcing_tensors, len(raw_forcing_rows)) or raw_forcing_rows
    depths_tensor = torch.tensor(depths, dtype=torch.float32, device=device)
    area_tensor = torch.tensor(area, dtype=torch.float32, device=device)
    heat_content_layer_weights = _heat_content_layer_weights(
        depths_tensor,
        area_tensor,
        device=depths_tensor.device,
        dtype=depths_tensor.dtype,
    )
    static_features = torch.tensor(static_feature_array(metadata, max_depth), dtype=torch.float32, device=device)
    static_features_2d = static_features.reshape(1, -1).contiguous()
    target_tensors_by_day = {
        'train': _build_target_tensor_matrix(
            df,
            train_lookup,
            train_masks,
            depths,
            device=device,
            date_to_index=date_to_index,
        ),
        'val': _build_target_tensor_matrix(
            df,
            val_lookup,
            val_masks,
            depths,
            device=device,
            date_to_index=date_to_index,
        ),
        'all': _build_target_tensor_matrix(
            df,
            all_lookup,
            all_masks,
            depths,
            device=device,
            date_to_index=date_to_index,
        ),
    }
    packed_tensor_store = PackedLakeTensorStore(
        forcing_tensors=forcing_tensors,
        target_tensors_by_day=target_tensors_by_day,
        depths=depths_tensor,
        area=area_tensor,
        heat_content_layer_weights=heat_content_layer_weights,
        static_features=static_features,
        static_features_2d=static_features_2d,
        date_to_index=date_to_index,
    )
    resident_tensors = {
        'forcing_tensors': forcing_tensors,
        'target_tensors_by_day': target_tensors_by_day,
        'depths': depths_tensor,
        'area': area_tensor,
        'heat_content_layer_weights': heat_content_layer_weights,
        'static_features': static_features,
        'static_features_2d': static_features_2d,
        'packed_tensor_store': packed_tensor_store,
    }
    forcing_cache_mode = (
        'resident_matrix_row_sequence'
        if isinstance(forcing_rows, ForcingRowSequence)
        else 'row_dict'
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
        'forcing_tensors': forcing_tensors,
        'static_features': static_features,
        'static_features_2d': static_features_2d,
        'packed_tensor_store': packed_tensor_store,
        'resident_tensors': resident_tensors,
        'forcing_cache_mode': forcing_cache_mode,
        'profile_loss_target_mode': profile_loss_target_mode,
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
        'target_tensors_by_day': target_tensors_by_day,
        'date_to_index': date_to_index,
        'pairs': {
            'train': train_pairs,
            'val': val_pairs,
            'all': all_pairs,
        },
        'segment_rollout_sequences': {
            'train': train_segment_sequences,
            'val': val_segment_sequences,
            'all': all_segment_sequences,
        },
        'unlabeled_heat_closure_windows': unlabeled_heat_closure_windows,
        'unlabeled_heat_closure_windows_by_horizon': unlabeled_heat_closure_windows_by_horizon,
        'unlabeled_heat_closure_horizons': unlabeled_heat_closure_horizons,
        'split_info': split_info,
        'profile_obs_path': profile_path,
    }


def _fit_zero_profile_thermal_basis_from_train_lakes(
    train_lakes,
    *,
    n_components=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_COMPONENTS,
    grid_points=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_GRID_POINTS,
    balance_mode=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODE,
):
    sources = []
    for lake in train_lakes:
        sources.append({
            'lake_id': lake['lake_id'],
            'depths': lake['depths_np'],
            'lookup': lake['lookups'].get('train', {}),
            'masks': lake.get('lookup_masks', {}).get('train', {}),
        })
    return fit_zero_profile_eof_pca_basis(
        sources,
        n_components=int(n_components),
        grid_points=int(grid_points),
        balance_mode=balance_mode,
    )


def _resolve_zero_profile_thermal_basis(
    *,
    checkpoint_thermal_basis,
    train_lakes,
    zero_profile_initializer,
    zero_profile_init_net_loss_weight,
    zero_profile_thermal_basis_components,
    zero_profile_thermal_basis_grid_points,
    zero_profile_thermal_basis_balance_mode=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODE,
    daily_memory_reconstruction_weight=DEFAULT_DAILY_MEMORY_RECONSTRUCTION_WEIGHT,
    prediction_branch='physics_rollout',
    thermal_state_profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    refit_zero_profile_thermal_basis=DEFAULT_REFIT_ZERO_PROFILE_THERMAL_BASIS,
):
    refit_mode = _normalize_on_off(
        refit_zero_profile_thermal_basis,
        name='refit_zero_profile_thermal_basis',
    )
    zero_profile_thermal_basis_balance_mode = normalize_zero_profile_thermal_basis_balance_mode(
        zero_profile_thermal_basis_balance_mode
    )
    needs_basis = (
        zero_profile_initializer in {'eof_pca_low_dof', 'eof_pca_init_net'}
        or float(zero_profile_init_net_loss_weight) > 0.0
        or float(daily_memory_reconstruction_weight) > 0.0
        or _normalize_prediction_branch(prediction_branch) == 'daily_memory'
        or _normalize_thermal_state_profile_fusion_mode(thermal_state_profile_fusion_mode) != 'off'
    )
    if not needs_basis:
        return checkpoint_thermal_basis, 'checkpoint' if checkpoint_thermal_basis is not None else 'none'
    if checkpoint_thermal_basis is not None and refit_mode == 'off':
        return checkpoint_thermal_basis, 'checkpoint'
    basis = _fit_zero_profile_thermal_basis_from_train_lakes(
        train_lakes,
        n_components=zero_profile_thermal_basis_components,
        grid_points=zero_profile_thermal_basis_grid_points,
        balance_mode=zero_profile_thermal_basis_balance_mode,
    )
    return basis, 'train_refit' if refit_mode == 'on' else 'train_fit'


def _normalize_unlabeled_heat_closure_reservoir_mode(value):
    text = str(value or DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE).strip().lower().replace('-', '_')
    if text not in UNLABELED_HEAT_CLOSURE_RESERVOIR_MODES:
        allowed = ', '.join(sorted(UNLABELED_HEAT_CLOSURE_RESERVOIR_MODES))
        raise ValueError(f'unlabeled_heat_closure_reservoir_mode must be one of: {allowed}.')
    return text


def _normalize_unlabeled_heat_closure_mode(value):
    text = str(value or DEFAULT_UNLABELED_HEAT_CLOSURE_MODE).strip().lower().replace('-', '_')
    aliases = {
        'storage': 'storage_budget_smooth_l1',
        'heat_storage': 'storage_budget_smooth_l1',
        'heat_storage_budget': 'storage_budget_smooth_l1',
        'storage_budget': 'storage_budget_smooth_l1',
        'smooth_l1_storage_budget': 'storage_budget_smooth_l1',
        'thresholded_storage_budget': 'storage_budget_thresholded',
        'storage_budget_relu': 'storage_budget_thresholded',
        'storage_budget_deadzone': 'storage_budget_thresholded',
    }
    text = aliases.get(text, text)
    if text not in UNLABELED_HEAT_CLOSURE_MODES:
        allowed = ', '.join(sorted(UNLABELED_HEAT_CLOSURE_MODES))
        raise ValueError(f'unlabeled_heat_closure_mode must be one of: {allowed}.')
    return text


def _normalize_unlabeled_heat_closure_state_source(value):
    text = str(value or DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE).strip().lower().replace('-', '_')
    aliases = {
        'prior': 'prior_window',
        'window': 'prior_window',
        'prior_spinup': 'prior_window',
        'spinup': 'spinup_then_window',
        'spinup_window': 'spinup_then_window',
        'spinup_then_rollout': 'spinup_then_window',
    }
    text = aliases.get(text, text)
    if text not in UNLABELED_HEAT_CLOSURE_STATE_SOURCES:
        allowed = ', '.join(sorted(UNLABELED_HEAT_CLOSURE_STATE_SOURCES))
        raise ValueError(f'unlabeled_heat_closure_state_source must be one of: {allowed}.')
    return text


def _normalize_prediction_branch(value):
    text = str(value or 'physics_rollout').strip().lower().replace('-', '_')
    aliases = {
        'physics': 'physics_rollout',
        'rollout': 'physics_rollout',
        'free_roll': 'physics_rollout',
        'daily': 'daily_memory',
        'memory': 'daily_memory',
        'daily_memory_reconstruction': 'daily_memory',
    }
    text = aliases.get(text, text)
    if text not in PREDICTION_BRANCHES:
        allowed = ', '.join(sorted(PREDICTION_BRANCHES))
        raise ValueError(f'prediction_branch must be one of: {allowed}.')
    return text


def _normalize_thermal_state_profile_fusion_mode(value):
    text = str(value or DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE).strip().lower().replace('-', '_')
    aliases = {
        'init': 'init_only',
        'initial': 'init_only',
        'zero_profile': 'init_only',
        'daily': 'daily_only',
        'memory': 'daily_only',
        'on': 'both',
        'all': 'both',
    }
    text = aliases.get(text, text)
    if text not in THERMAL_STATE_PROFILE_FUSION_MODES:
        allowed = ', '.join(sorted(THERMAL_STATE_PROFILE_FUSION_MODES))
        raise ValueError(f'thermal_state_profile_fusion_mode must be one of: {allowed}.')
    return text


def _normalize_thermal_state_profile_fusion_time_policy(value):
    text = str(
        value or DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY
    ).strip().lower().replace('-', '_')
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
    if text not in THERMAL_STATE_PROFILE_FUSION_TIME_POLICIES:
        allowed = ', '.join(sorted(THERMAL_STATE_PROFILE_FUSION_TIME_POLICIES))
        raise ValueError(f'thermal_state_profile_fusion_time_policy must be one of: {allowed}.')
    return text


def _unlabeled_heat_closure_state_source_code(value):
    source = _normalize_unlabeled_heat_closure_state_source(value)
    return 1.0 if source == 'spinup_then_window' else 0.0


_parse_unlabeled_heat_closure_horizons = parse_unlabeled_heat_closure_horizons
_format_unlabeled_heat_closure_horizons = format_unlabeled_heat_closure_horizons
_build_unlabeled_heat_closure_windows_for_horizon = (
    build_unlabeled_heat_closure_windows_for_horizon
)
_build_unlabeled_heat_closure_windows_by_horizon = (
    build_unlabeled_heat_closure_windows_by_horizon
)
_build_unlabeled_heat_closure_windows = build_unlabeled_heat_closure_windows


def _select_unlabeled_heat_closure_windows(lake, batch_size, epoch):
    by_horizon = lake.get('unlabeled_heat_closure_windows_by_horizon') or {}
    if by_horizon:
        horizon_items = [
            (int(horizon), tuple(windows))
            for horizon, windows in by_horizon.items()
            if tuple(windows)
        ]
        if not horizon_items:
            return ()
        horizon_items = sorted(horizon_items, key=lambda item: item[0])
        batch_size = int(batch_size or 0)
        if batch_size <= 0:
            selected = []
            for _horizon, windows in horizon_items:
                selected.extend(windows)
            return tuple(selected)
        horizon_count = len(horizon_items)
        base = batch_size // horizon_count
        remainder = batch_size % horizon_count
        selected = []
        for horizon_pos, (horizon, windows) in enumerate(horizon_items):
            quota = base + (1 if horizon_pos < remainder else 0)
            if quota <= 0:
                continue
            if len(windows) <= quota:
                selected.extend(windows)
                continue
            rng = random.Random(f"{lake.get('lake_id', '')}:{int(epoch)}:{horizon}:{len(windows)}")
            indices = sorted(rng.sample(range(len(windows)), quota))
            selected.extend(windows[idx] for idx in indices)
        return tuple(selected)

    windows = tuple(lake.get('unlabeled_heat_closure_windows', ()))
    if not windows:
        return ()
    batch_size = int(batch_size or 0)
    if batch_size <= 0 or len(windows) <= batch_size:
        return windows
    rng = random.Random(f"{lake.get('lake_id', '')}:{int(epoch)}:{len(windows)}")
    indices = rng.sample(range(len(windows)), batch_size)
    return tuple(windows[idx] for idx in indices)


def _forcing_row_scalar_float(row, key, *, default=0.0):
    value = row.get(key) if row is not None else None
    if value is None:
        return float(default)
    try:
        array = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(finite.mean())


def _unlabeled_heat_closure_static_step_allowed(row, next_row, *, open_water_only, lst_qc_min):
    quality = min(
        _forcing_row_scalar_float(row, 'lst_quality', default=0.5),
        _forcing_row_scalar_float(next_row, 'lst_quality', default=0.5),
    )
    if quality < float(lst_qc_min):
        return False
    if open_water_only == 'on':
        ice = max(
            _forcing_row_scalar_float(row, 'ice_mask', default=0.0),
            _forcing_row_scalar_float(row, 'ice_fraction', default=0.0),
            _forcing_row_scalar_float(next_row, 'ice_mask', default=0.0),
            _forcing_row_scalar_float(next_row, 'ice_fraction', default=0.0),
        )
        if ice > 0.05:
            return False
    return True


def _unlabeled_heat_closure_gate_cache_key(*, open_water_only, lst_qc_min):
    open_water_only = _normalize_on_off(open_water_only, name='unlabeled_heat_closure_open_water_only')
    return (open_water_only, round(float(lst_qc_min), 8))


def _build_unlabeled_heat_closure_window_step_cache(lake, *, open_water_only, lst_qc_min):
    windows = tuple(lake.get('unlabeled_heat_closure_windows', ()))
    forcing_rows = lake.get('forcing_rows') or ()
    cache = {}
    for start_idx, end_idx in windows:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if end_idx <= start_idx or end_idx >= len(forcing_rows):
            cache[(start_idx, end_idx)] = ()
            continue
        allowed_steps = []
        for day_idx in range(start_idx, end_idx):
            next_row = forcing_rows[day_idx + 1] if day_idx + 1 < len(forcing_rows) else None
            if _unlabeled_heat_closure_static_step_allowed(
                forcing_rows[day_idx],
                next_row,
                open_water_only=open_water_only,
                lst_qc_min=lst_qc_min,
            ):
                allowed_steps.append(int(day_idx))
        cache[(start_idx, end_idx)] = tuple(allowed_steps)
    return cache


def _unlabeled_heat_closure_window_step_cache(lake, *, open_water_only, lst_qc_min):
    key = _unlabeled_heat_closure_gate_cache_key(
        open_water_only=open_water_only,
        lst_qc_min=lst_qc_min,
    )
    normalized_open_water_only = key[0]
    caches = lake.setdefault('unlabeled_heat_closure_window_step_cache', {})
    if key not in caches:
        caches[key] = _build_unlabeled_heat_closure_window_step_cache(
            lake,
            open_water_only=normalized_open_water_only,
            lst_qc_min=lst_qc_min,
        )
    return caches[key]


def _forcing_row_scalar(row, key, *, device, dtype, default=0.0):
    value = row.get(key) if row is not None else None
    if value is None:
        return torch.tensor(float(default), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return torch.tensor(float(default), device=device, dtype=dtype)
    return finite.mean()


def _forcing_row_tensor(row, key, reference, *, default=0.0):
    value = row.get(key) if row is not None else None
    if value is None:
        return torch.full_like(reference, float(default))
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype).reshape(-1)
    if tensor.numel() == 1 and reference.numel() > 1:
        tensor = tensor.expand(reference.numel())
    elif tensor.numel() != reference.numel():
        tensor = tensor.reshape_as(reference)
    default_tensor = torch.full_like(reference, float(default))
    return torch.where(torch.isfinite(tensor), tensor, default_tensor)


def _unlabeled_heat_storage_budget_prior_wm2(row, profile):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    surface_temp = profile[:, 0]
    shortwave = _forcing_row_tensor(row, 'shortwave', surface_temp, default=0.0)
    longwave = _forcing_row_tensor(row, 'longwave', surface_temp, default=0.0)
    latent = _forcing_row_tensor(row, 'latent_heat', surface_temp, default=0.0)
    sensible = _forcing_row_tensor(row, 'sensible_heat', surface_temp, default=0.0)
    ice_fraction = _forcing_row_tensor(row, 'ice_fraction', surface_temp, default=0.0)
    ice_mask = _forcing_row_tensor(row, 'ice_mask', surface_temp, default=0.0)
    ice_gate = torch.clamp(torch.maximum(ice_fraction, ice_mask), 0.0, 1.0)

    lst_surface = _forcing_row_tensor(row, 'lst_surface', surface_temp, default=float('nan'))
    lst_observed = _forcing_row_tensor(row, 'lst_observed_flag', surface_temp, default=0.0)
    lst_quality = _forcing_row_tensor(row, 'lst_quality', surface_temp, default=0.0)
    use_observed_lst = (
        torch.isfinite(lst_surface)
        & (lst_observed >= 0.5)
        & (lst_quality >= DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN)
    )
    budget_surface_temp = torch.where(use_observed_lst, lst_surface, surface_temp.detach())
    surface_temp_k = torch.clamp(budget_surface_temp + 273.15, min=200.0, max=330.0)
    outgoing_longwave = float(WATER_EMISSIVITY) * float(STEFAN_BOLTZMANN) * surface_temp_k ** 4
    albedo = float(SURFACE_ALBEDO_WATER) * (1.0 - ice_gate) + float(SURFACE_ALBEDO_ICE) * ice_gate
    absorbed_shortwave = (1.0 - albedo) * shortwave
    return absorbed_shortwave + longwave - outgoing_longwave - latent - sensible


def _effective_profile_heat_content_j_m2(model, profile, lake, freezing_storage):
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    heat_content = model.heat_content_j_m2(
        profile,
        depths=lake['depths'],
        area_profile=lake['area'],
    )
    if freezing_storage is None:
        storage = torch.zeros_like(heat_content)
    else:
        storage = freezing_storage.to(device=profile.device, dtype=profile.dtype)
        if storage.ndim == 1:
            storage = storage.unsqueeze(0)
        storage = storage.reshape(storage.shape[0], -1).sum(dim=1)
    return heat_content - storage


def _smooth_l1_scaled_residual(residual, scale):
    scale = max(float(scale), 1.0e-6)
    normalized = residual / scale
    abs_normalized = torch.abs(normalized)
    return torch.where(
        abs_normalized < 1.0,
        0.5 * normalized * normalized,
        abs_normalized - 0.5,
    )


def _thresholded_scaled_residual(residual, scale):
    scale = max(float(scale), 1.0e-6)
    return torch.relu(torch.abs(residual) - scale) / scale


def _storage_budget_residual_loss(residual, scale, mode):
    closure_mode = _normalize_unlabeled_heat_closure_mode(mode)
    if closure_mode == 'storage_budget_thresholded':
        return _thresholded_scaled_residual(residual, scale)
    if closure_mode == 'storage_budget_smooth_l1':
        return _smooth_l1_scaled_residual(residual, scale)
    raise ValueError(
        'storage-budget residual loss requires storage_budget_smooth_l1 '
        f'or storage_budget_thresholded, got {closure_mode}.'
    )


def _initialize_unlabeled_heat_closure_state(
    model,
    lake,
    start_idx,
    *,
    state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    state_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    zero_profile_lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    task_mode='analysis',
    hard_density_stability=False,
    dtype=None,
):
    source = _normalize_unlabeled_heat_closure_state_source(state_source)
    if source == 'spinup_then_window':
        effective_spinup_days = max(0, int(state_spinup_days))
        effective_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    else:
        effective_spinup_days = 0
        effective_observer_mode = 'off'
    device = lake['depths'].device
    dtype = lake['depths'].dtype if dtype is None else dtype
    start_date = pd.Timestamp(lake['df'].iloc[int(start_idx)]['Date']).normalize()
    init_state = initialize_rollout_state(
        model=model,
        df=lake['df'],
        depths=lake['depths_np'],
        all_lookup={},
        forcing_rows=lake['forcing_rows'],
        static_features=lake['static_features'],
        metadata=lake['metadata'],
        device=device,
        init_mode='prior_spinup',
        rollout_start_date=start_date,
        spinup_days=effective_spinup_days,
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=effective_observer_mode,
        spinup_lst_assimilation_strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
        spinup_lst_assimilation_decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
        spinup_lst_assimilation_max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
        zero_profile_thermal_basis=getattr(model, 'zero_profile_thermal_basis', None),
        lswt_observer_low_rank_deep_update_fraction=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
        lswt_observer_heat_content_limit_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
        lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
        task_mode=task_mode,
        area_profile=lake['area'],
        hard_density_stability=hard_density_stability,
    )
    current = init_state['current'].to(device=device, dtype=dtype)
    if current.ndim == 1:
        current = current.unsqueeze(0)
    freezing_storage = init_state['freezing_storage_j_m2'].to(device=device, dtype=dtype)
    if freezing_storage.ndim == 1:
        freezing_storage = freezing_storage.unsqueeze(0)
    return current, freezing_storage, {
        'state_source': source,
        'spinup_days': torch.tensor(
            float(init_state.get('spinup_days_used', effective_spinup_days)),
            device=device,
            dtype=dtype,
        ),
    }


@torch.no_grad()
def _initialize_unlabeled_heat_closure_states_batched(
    model,
    lake,
    start_indices,
    *,
    state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    state_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    zero_profile_lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    task_mode='analysis',
    hard_density_stability=False,
    dtype=None,
):
    source = _normalize_unlabeled_heat_closure_state_source(state_source)
    zero_profile_initializer = normalize_zero_profile_initializer_mode(zero_profile_initializer)
    if source == 'spinup_then_window':
        effective_spinup_days = max(0, int(state_spinup_days))
        effective_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    else:
        effective_spinup_days = 0
        effective_observer_mode = 'off'

    device = lake['depths'].device
    dtype = lake['depths'].dtype if dtype is None else dtype
    if torch.is_tensor(start_indices):
        start_values = [int(value) for value in start_indices.detach().cpu().reshape(-1).tolist()]
    else:
        start_values = [int(value) for value in start_indices]
    if not start_values:
        raise ValueError('batched heat-closure initialization requires at least one start index.')

    grouped = {}
    for output_pos, rollout_start_idx in enumerate(start_values):
        rollout_start_idx = max(0, min(int(rollout_start_idx), len(lake['forcing_rows']) - 1))
        initial_idx = max(0, rollout_start_idx - effective_spinup_days)
        spinup_days_used = max(0, int(rollout_start_idx - initial_idx))
        grouped.setdefault(spinup_days_used, []).append((output_pos, initial_idx))

    profile_slots = [None] * len(start_values)
    freezing_slots = [None] * len(start_values)
    spinup_days_values = torch.zeros((len(start_values),), device=device, dtype=dtype)
    depths_np = lake['depths_np']
    metadata = lake['metadata']
    thermal_basis = getattr(model, 'zero_profile_thermal_basis', None)

    def build_initial_profile(initial_idx):
        if zero_profile_initializer == 'low_dof':
            profile, _prior_info = build_zero_profile_low_dof_prior(
                lake['df'],
                depths_np,
                metadata,
                initial_idx,
            )
        elif zero_profile_initializer == 'lswt_climatology_low_dof':
            profile, _prior_info = build_zero_profile_lswt_climatology_low_dof_prior(
                lake['df'],
                depths_np,
                metadata,
                initial_idx,
                min_quality=zero_profile_lswt_observer_min_quality,
            )
        elif zero_profile_initializer == 'eof_pca_low_dof':
            profile, _prior_info = build_zero_profile_eof_pca_low_dof_prior(
                lake['df'],
                depths_np,
                metadata,
                initial_idx,
                thermal_basis=thermal_basis,
                min_quality=zero_profile_lswt_observer_min_quality,
            )
        elif zero_profile_initializer == 'eof_pca_init_net':
            profile, _prior_info = build_zero_profile_eof_pca_init_net_prior(
                lake['df'],
                depths_np,
                metadata,
                initial_idx,
                model=model,
                forcing_history=lake['forcing_rows'][initial_idx]['history_features'],
                static_features=lake['static_features'],
                conditioning_features=_zero_profile_init_conditioning_batch(
                    lake,
                    [initial_idx],
                    device=device,
                    dtype=dtype,
                ),
                thermal_basis=thermal_basis,
                min_quality=zero_profile_lswt_observer_min_quality,
            )
        else:
            raise ValueError(
                f"unsupported tenth-version zero-profile initializer: {zero_profile_initializer}"
            )
        return torch.as_tensor(profile, dtype=dtype, device=device).reshape(1, -1)

    for spinup_days_used in sorted(grouped):
        entries = grouped[spinup_days_used]
        output_positions = [entry[0] for entry in entries]
        initial_indices = [entry[1] for entry in entries]
        initial_index_tensor = torch.as_tensor(initial_indices, dtype=torch.long, device=device)
        current = torch.cat(
            [build_initial_profile(initial_idx) for initial_idx in initial_indices],
            dim=0,
        )
        freezing_storage = torch.zeros_like(current)
        for offset in range(int(spinup_days_used)):
            day_indices = initial_index_tensor + int(offset)
            next_indices = day_indices + 1
            next_row_batch = _forcing_row_batch(lake, next_indices)
            current, freezing_storage = model.step(
                current,
                _forcing_row_batch(lake, day_indices),
                lake['static_features'],
                next_forcing_row=next_row_batch,
                task_mode=task_mode,
                depths=lake['depths'],
                area_profile=lake['area'],
                diagnostic_mode='none',
                hard_density_stability=hard_density_stability,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
            if effective_observer_mode != 'off':
                current, _observer_detail = apply_lswt_observer_update(
                    current,
                    next_row_batch,
                    lake['depths'],
                    mode=effective_observer_mode,
                    strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
                    decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
                    max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
                    low_rank_deep_update_fraction=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
                    heat_content_limit_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
                    min_quality=zero_profile_lswt_observer_min_quality,
                    area_profile=lake['area'],
                    metadata=metadata,
                )
        for batch_pos, output_pos in enumerate(output_positions):
            profile_slots[output_pos] = current[batch_pos: batch_pos + 1].detach()
            freezing_slots[output_pos] = freezing_storage[batch_pos: batch_pos + 1].detach()
            spinup_days_values[output_pos] = float(spinup_days_used)

    return torch.cat(profile_slots, dim=0), torch.cat(freezing_slots, dim=0), {
        'state_source': source,
        'spinup_days': spinup_days_values,
    }


def _zero_unlabeled_heat_closure_detail(
    device,
    *,
    dtype=torch.float32,
    weight=0.0,
    effective_weight=0.0,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    no_profile_lst_surface_effective_weight=0.0,
    solver_guard_weight=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
    solver_guard_effective_weight=0.0,
    solver_guard_tau=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2,
    state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    state_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
):
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return {
        'unlabeled_heat_closure_loss': zero,
        'unlabeled_heat_closure_weighted_loss': zero,
        'unlabeled_heat_closure_weight': torch.tensor(float(weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_effective_weight': torch.tensor(float(effective_weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_window_count': zero,
        'unlabeled_heat_closure_step_count': zero,
        'unlabeled_heat_closure_active_loss_count': zero,
        'unlabeled_heat_closure_residual_abs_mean_wm2': zero,
        'unlabeled_heat_closure_residual_bias_mean_wm2': zero,
        'unlabeled_heat_closure_budget_residual_abs_mean_wm2': zero,
        'unlabeled_heat_closure_budget_residual_bias_mean_wm2': zero,
        'unlabeled_heat_closure_budget_storage_tendency_mean_wm2': zero,
        'unlabeled_heat_closure_budget_target_mean_wm2': zero,
        'unlabeled_heat_closure_solver_residual_abs_mean_wm2': zero,
        'unlabeled_heat_closure_solver_residual_bias_mean_wm2': zero,
        'unlabeled_heat_closure_lst_surface_loss': zero,
        'unlabeled_heat_closure_lst_surface_weighted_loss': zero,
        'unlabeled_heat_closure_lst_surface_weight': torch.tensor(
            float(no_profile_lst_surface_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_effective_weight': torch.tensor(
            float(no_profile_lst_surface_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_supervision_count': zero,
        'unlabeled_heat_closure_lst_surface_weight_mean': zero,
        'unlabeled_heat_closure_solver_guard_loss': zero,
        'unlabeled_heat_closure_solver_guard_weighted_loss': zero,
        'unlabeled_heat_closure_solver_guard_weight': torch.tensor(
            float(solver_guard_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_effective_weight': torch.tensor(
            float(solver_guard_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_tau_wm2': torch.tensor(
            float(solver_guard_tau),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_active_loss_count': zero,
        'unlabeled_heat_closure_tau_wm2': zero,
        'unlabeled_heat_closure_window_days': zero,
        'unlabeled_heat_closure_horizon_days_mean': zero,
        'unlabeled_heat_closure_horizon_days_min': zero,
        'unlabeled_heat_closure_horizon_days_max': zero,
        'unlabeled_heat_closure_horizon_count': zero,
        'unlabeled_heat_closure_profile_label_count': zero,
        'unlabeled_heat_closure_reservoir_diagnostic_only_count': zero,
        'unlabeled_heat_closure_reservoir_excluded_count': zero,
        'unlabeled_heat_closure_state_source_code': torch.tensor(
            _unlabeled_heat_closure_state_source_code(state_source),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_spinup_days_used': zero,
        'unlabeled_heat_closure_spinup_days_config': torch.tensor(
            float(state_spinup_days),
            device=device,
            dtype=dtype,
        ),
    }


def _unlabeled_heat_closure_step_diagnostic_mode(step_diagnostic_mode):
    mode = str(step_diagnostic_mode or 'loss').strip().lower()
    return 'full' if mode == 'full' else 'loss_fast'


def _unlabeled_heat_storage_budget_loss_for_lake_batched(
    model,
    lake,
    windows,
    *,
    weight,
    effective_weight,
    window_days,
    tau,
    is_reservoir,
    reservoir_mode,
    closure_mode,
    state_source,
    state_spinup_days,
    no_profile_lst_surface_weight,
    no_profile_lst_surface_effective_weight,
    solver_guard_weight,
    solver_guard_effective_weight,
    solver_guard_tau,
    zero_profile_initializer,
    spinup_lswt_observer_mode,
    zero_profile_lswt_observer_min_quality,
    task_mode,
    hard_density_stability,
    step_diagnostic_mode,
    window_step_cache,
    open_water_only,
    lst_qc_min,
):
    if window_step_cache is None or not lake.get('forcing_tensors'):
        return None

    device = lake['depths'].device
    dtype = lake['depths'].dtype
    solver_guard_weight = max(0.0, float(solver_guard_weight))
    solver_guard_effective_weight = max(0.0, float(solver_guard_effective_weight))
    no_profile_lst_surface_weight = max(0.0, float(no_profile_lst_surface_weight))
    no_profile_lst_surface_effective_weight = max(
        0.0,
        float(no_profile_lst_surface_effective_weight),
    )
    solver_guard_tau = max(float(solver_guard_tau), 1.0e-6)
    compute_solver_guard = solver_guard_weight > 0.0
    compute_lst_surface = no_profile_lst_surface_weight > 0.0
    closure_step_diagnostic_mode = _unlabeled_heat_closure_step_diagnostic_mode(step_diagnostic_mode)
    grouped = {}
    for start_idx, end_idx in tuple(windows or ()):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        duration = int(end_idx - start_idx)
        if duration <= 0 or end_idx >= len(lake['forcing_rows']):
            continue
        allowed_steps = tuple(
            int(day_idx)
            for day_idx in window_step_cache.get((start_idx, end_idx), ())
            if start_idx <= int(day_idx) < end_idx
        )
        if len(allowed_steps) != duration:
            continue
        grouped.setdefault(duration, []).append((start_idx, end_idx))

    if not grouped:
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            _zero_unlabeled_heat_closure_detail(
                device,
                dtype=dtype,
                weight=weight,
                effective_weight=effective_weight,
                no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                no_profile_lst_surface_effective_weight=no_profile_lst_surface_effective_weight,
                solver_guard_weight=solver_guard_weight,
                solver_guard_effective_weight=solver_guard_effective_weight,
                solver_guard_tau=solver_guard_tau,
                state_source=state_source,
                state_spinup_days=state_spinup_days,
            ),
        )

    horizon_values = [
        float(duration)
        for duration, entries in grouped.items()
        for _entry in entries
    ]
    horizon_tensor = torch.as_tensor(horizon_values, device=device, dtype=dtype)
    loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    residual_abs_sum = torch.tensor(0.0, device=device, dtype=dtype)
    residual_bias_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_guard_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_residual_abs_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_residual_bias_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_weight_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_supervision_count = torch.tensor(0.0, device=device, dtype=dtype)
    storage_tendency_sum = torch.tensor(0.0, device=device, dtype=dtype)
    budget_target_sum = torch.tensor(0.0, device=device, dtype=dtype)
    active_loss_count = torch.tensor(0.0, device=device, dtype=dtype)
    solver_guard_active_loss_count = torch.tensor(0.0, device=device, dtype=dtype)
    spinup_days_sum = torch.tensor(0.0, device=device, dtype=dtype)
    state_window_count = torch.tensor(0.0, device=device, dtype=dtype)
    step_count = 0
    static_features = lake['static_features']

    for duration in sorted(grouped):
        entries = grouped[duration]
        start_index_tensor = torch.as_tensor(
            [int(start_idx) for start_idx, _ in entries],
            dtype=torch.long,
            device=device,
        )
        prediction, freezing_storage, init_detail = _initialize_unlabeled_heat_closure_states_batched(
            model,
            lake,
            start_index_tensor,
            state_source=state_source,
            state_spinup_days=state_spinup_days,
            zero_profile_initializer=zero_profile_initializer,
            spinup_lswt_observer_mode=spinup_lswt_observer_mode,
            zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
            task_mode=task_mode,
            hard_density_stability=hard_density_stability,
            dtype=dtype,
        )
        spinup_days_sum = spinup_days_sum + init_detail['spinup_days'].detach().reshape(-1).sum()
        state_window_count = state_window_count + torch.tensor(float(len(entries)), device=device, dtype=dtype)
        heat_start = _effective_profile_heat_content_j_m2(
            model,
            prediction,
            lake,
            freezing_storage,
        )
        budget_sum = torch.zeros(prediction.shape[0], device=device, dtype=dtype)

        for offset in range(duration):
            day_indices = start_index_tensor + int(offset)
            next_indices = day_indices + 1
            row_batch = _forcing_row_batch(lake, day_indices)
            next_row_batch = _forcing_row_batch(lake, next_indices)
            budget_sum = budget_sum + _unlabeled_heat_storage_budget_prior_wm2(
                row_batch,
                prediction,
            ).reshape(-1)
            if compute_solver_guard:
                prediction, freezing_storage, diagnostics = model.step(
                    prediction,
                    row_batch,
                    static_features,
                    next_forcing_row=next_row_batch,
                    return_diagnostics=True,
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=lake['area'],
                    diagnostic_mode=closure_step_diagnostic_mode,
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage,
                    return_freezing_storage=True,
                )
                solver_residual = (
                    diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']
                ).reshape(-1)
                solver_finite_mask = torch.isfinite(solver_residual)
                solver_finite_f = solver_finite_mask.to(dtype=dtype)
                safe_solver_residual = torch.where(
                    solver_finite_mask,
                    solver_residual,
                    torch.zeros_like(solver_residual),
                )
                solver_guard_loss_sum = solver_guard_loss_sum + (
                    _thresholded_scaled_residual(safe_solver_residual, solver_guard_tau)
                    * solver_finite_f
                ).sum()
                solver_residual_abs_sum = solver_residual_abs_sum + (
                    torch.abs(safe_solver_residual).detach() * solver_finite_f
                ).sum()
                solver_residual_bias_sum = solver_residual_bias_sum + (
                    safe_solver_residual.detach() * solver_finite_f
                ).sum()
                solver_guard_active_loss_count = (
                    solver_guard_active_loss_count + solver_finite_f.sum()
                )
            else:
                prediction, freezing_storage = model.step(
                    prediction,
                    row_batch,
                    static_features,
                    next_forcing_row=next_row_batch,
                    return_diagnostics=False,
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=lake['area'],
                    diagnostic_mode='none',
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage,
                    return_freezing_storage=True,
                )
            if compute_lst_surface:
                lst_loss_vec, lst_weight, lst_mask = _no_profile_lst_surface_loss_per_sample(
                    prediction,
                    next_row_batch,
                    lst_qc_min=lst_qc_min,
                    open_water_only=open_water_only,
                )
                lst_mask_f = lst_mask.to(device=device, dtype=dtype)
                lst_surface_loss_sum = lst_surface_loss_sum + (
                    lst_loss_vec.reshape(-1) * lst_mask_f
                ).sum()
                lst_surface_weight_sum = lst_surface_weight_sum + (
                    lst_weight.reshape(-1) * lst_mask_f
                ).sum()
                lst_surface_supervision_count = lst_surface_supervision_count + lst_mask_f.sum()
            step_count += int(prediction.shape[0])

        heat_end = _effective_profile_heat_content_j_m2(
            model,
            prediction,
            lake,
            freezing_storage,
        )
        storage_tendency = (heat_end - heat_start) / (float(duration) * float(SECONDS_PER_DAY))
        budget_target = budget_sum / float(duration)
        residual = storage_tendency - budget_target
        finite_mask = torch.isfinite(residual)
        finite_f = finite_mask.to(dtype=dtype)
        safe_residual = torch.where(finite_mask, residual, torch.zeros_like(residual))
        safe_storage_tendency = torch.where(finite_mask, storage_tendency, torch.zeros_like(storage_tendency))
        safe_budget_target = torch.where(finite_mask, budget_target, torch.zeros_like(budget_target))
        loss_sum = loss_sum + (
            _storage_budget_residual_loss(safe_residual, tau, closure_mode) * finite_f
        ).sum()
        residual_abs_sum = residual_abs_sum + (torch.abs(safe_residual).detach() * finite_f).sum()
        residual_bias_sum = residual_bias_sum + (safe_residual.detach() * finite_f).sum()
        storage_tendency_sum = storage_tendency_sum + (safe_storage_tendency.detach() * finite_f).sum()
        budget_target_sum = budget_target_sum + (safe_budget_target.detach() * finite_f).sum()
        active_loss_count = active_loss_count + finite_f.sum()

    active_denominator = torch.clamp(active_loss_count, min=1.0)
    solver_guard_denominator = torch.clamp(solver_guard_active_loss_count, min=1.0)
    lst_surface_denominator = torch.clamp(lst_surface_supervision_count, min=1.0)
    loss = loss_sum / active_denominator
    solver_guard_loss = solver_guard_loss_sum / solver_guard_denominator
    solver_guard_weighted_loss = float(solver_guard_effective_weight) * solver_guard_loss
    lst_surface_loss = lst_surface_loss_sum / lst_surface_denominator
    lst_surface_weighted_loss = float(no_profile_lst_surface_effective_weight) * lst_surface_loss
    weighted_loss = (
        float(effective_weight) * loss
        + solver_guard_weighted_loss
        + lst_surface_weighted_loss
    )
    residual_abs_mean = residual_abs_sum / active_denominator
    residual_bias_mean = residual_bias_sum / active_denominator
    solver_residual_abs_mean = solver_residual_abs_sum / solver_guard_denominator
    solver_residual_bias_mean = solver_residual_bias_sum / solver_guard_denominator
    lst_surface_weight_mean = lst_surface_weight_sum / lst_surface_denominator
    storage_tendency_mean = storage_tendency_sum / active_denominator
    budget_target_mean = budget_target_sum / active_denominator
    spinup_days_used = spinup_days_sum / torch.clamp(state_window_count, min=1.0)
    detail = {
        'unlabeled_heat_closure_loss': loss.detach(),
        'unlabeled_heat_closure_weighted_loss': weighted_loss.detach(),
        'unlabeled_heat_closure_weight': torch.tensor(float(weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_effective_weight': torch.tensor(float(effective_weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_window_count': active_loss_count.detach(),
        'unlabeled_heat_closure_step_count': torch.tensor(float(step_count), device=device, dtype=dtype),
        'unlabeled_heat_closure_active_loss_count': active_loss_count.detach(),
        'unlabeled_heat_closure_residual_abs_mean_wm2': residual_abs_mean.detach(),
        'unlabeled_heat_closure_residual_bias_mean_wm2': residual_bias_mean.detach(),
        'unlabeled_heat_closure_budget_residual_abs_mean_wm2': residual_abs_mean.detach(),
        'unlabeled_heat_closure_budget_residual_bias_mean_wm2': residual_bias_mean.detach(),
        'unlabeled_heat_closure_budget_storage_tendency_mean_wm2': storage_tendency_mean.detach(),
        'unlabeled_heat_closure_budget_target_mean_wm2': budget_target_mean.detach(),
        'unlabeled_heat_closure_solver_residual_abs_mean_wm2': solver_residual_abs_mean.detach(),
        'unlabeled_heat_closure_solver_residual_bias_mean_wm2': solver_residual_bias_mean.detach(),
        'unlabeled_heat_closure_lst_surface_loss': lst_surface_loss.detach(),
        'unlabeled_heat_closure_lst_surface_weighted_loss': lst_surface_weighted_loss.detach(),
        'unlabeled_heat_closure_lst_surface_weight': torch.tensor(
            float(no_profile_lst_surface_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_effective_weight': torch.tensor(
            float(no_profile_lst_surface_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_supervision_count': (
            lst_surface_supervision_count.detach()
        ),
        'unlabeled_heat_closure_lst_surface_weight_mean': lst_surface_weight_mean.detach(),
        'unlabeled_heat_closure_solver_guard_loss': solver_guard_loss.detach(),
        'unlabeled_heat_closure_solver_guard_weighted_loss': solver_guard_weighted_loss.detach(),
        'unlabeled_heat_closure_solver_guard_weight': torch.tensor(
            float(solver_guard_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_effective_weight': torch.tensor(
            float(solver_guard_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_tau_wm2': torch.tensor(
            float(solver_guard_tau),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_active_loss_count': solver_guard_active_loss_count.detach(),
        'unlabeled_heat_closure_tau_wm2': torch.tensor(float(tau), device=device, dtype=dtype),
        'unlabeled_heat_closure_window_days': torch.tensor(float(window_days), device=device, dtype=dtype),
        'unlabeled_heat_closure_horizon_days_mean': horizon_tensor.mean().detach(),
        'unlabeled_heat_closure_horizon_days_min': horizon_tensor.min().detach(),
        'unlabeled_heat_closure_horizon_days_max': horizon_tensor.max().detach(),
        'unlabeled_heat_closure_horizon_count': torch.tensor(
            float(len(set(int(value) for value in horizon_values))),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_profile_label_count': torch.tensor(0.0, device=device, dtype=dtype),
        'unlabeled_heat_closure_reservoir_diagnostic_only_count': torch.tensor(
            1.0 if is_reservoir and reservoir_mode == 'diagnostic_only' else 0.0,
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_reservoir_excluded_count': torch.tensor(0.0, device=device, dtype=dtype),
        'unlabeled_heat_closure_state_source_code': torch.tensor(
            _unlabeled_heat_closure_state_source_code(state_source),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_spinup_days_used': spinup_days_used.detach(),
        'unlabeled_heat_closure_spinup_days_config': torch.tensor(
            float(state_spinup_days),
            device=device,
            dtype=dtype,
        ),
    }
    return weighted_loss, detail


def _unlabeled_heat_storage_budget_loss_for_lake_unbatched(
    model,
    lake,
    windows,
    *,
    weight,
    effective_weight,
    window_days,
    tau,
    is_reservoir,
    reservoir_mode,
    closure_mode,
    state_source,
    state_spinup_days,
    no_profile_lst_surface_weight,
    no_profile_lst_surface_effective_weight,
    solver_guard_weight,
    solver_guard_effective_weight,
    solver_guard_tau,
    zero_profile_initializer,
    spinup_lswt_observer_mode,
    zero_profile_lswt_observer_min_quality,
    task_mode,
    hard_density_stability,
    step_diagnostic_mode,
    window_step_cache,
    open_water_only,
    lst_qc_min,
):
    device = lake['depths'].device
    dtype = lake['depths'].dtype
    solver_guard_weight = max(0.0, float(solver_guard_weight))
    solver_guard_effective_weight = max(0.0, float(solver_guard_effective_weight))
    no_profile_lst_surface_weight = max(0.0, float(no_profile_lst_surface_weight))
    no_profile_lst_surface_effective_weight = max(
        0.0,
        float(no_profile_lst_surface_effective_weight),
    )
    solver_guard_tau = max(float(solver_guard_tau), 1.0e-6)
    compute_solver_guard = solver_guard_weight > 0.0
    compute_lst_surface = no_profile_lst_surface_weight > 0.0
    closure_step_diagnostic_mode = _unlabeled_heat_closure_step_diagnostic_mode(step_diagnostic_mode)
    window_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    valid_window_count = torch.tensor(0.0, device=device, dtype=dtype)
    residual_abs_sum = torch.tensor(0.0, device=device, dtype=dtype)
    residual_bias_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_guard_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_residual_abs_sum = torch.tensor(0.0, device=device, dtype=dtype)
    solver_residual_bias_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_weight_sum = torch.tensor(0.0, device=device, dtype=dtype)
    lst_surface_supervision_count = torch.tensor(0.0, device=device, dtype=dtype)
    storage_tendency_sum = torch.tensor(0.0, device=device, dtype=dtype)
    budget_target_sum = torch.tensor(0.0, device=device, dtype=dtype)
    step_count = 0
    active_loss_count = torch.tensor(0.0, device=device, dtype=dtype)
    solver_guard_active_loss_count = torch.tensor(0.0, device=device, dtype=dtype)
    spinup_days_sum = torch.tensor(0.0, device=device, dtype=dtype)
    state_window_count = torch.tensor(0.0, device=device, dtype=dtype)
    horizon_days_sum = torch.tensor(0.0, device=device, dtype=dtype)
    horizon_days_min = torch.tensor(float('inf'), device=device, dtype=dtype)
    horizon_days_max = torch.tensor(0.0, device=device, dtype=dtype)
    horizon_valid_count = torch.tensor(0.0, device=device, dtype=dtype)
    horizon_seen = set()

    for start_idx, end_idx in tuple(windows or ()):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        duration = int(end_idx - start_idx)
        if duration <= 0 or end_idx >= len(lake['forcing_rows']):
            continue
        if window_step_cache is not None:
            allowed_steps = tuple(window_step_cache.get((start_idx, end_idx), ()))
            if len(allowed_steps) != duration:
                continue
        else:
            allowed = True
            for day_idx in range(start_idx, end_idx):
                row = lake['forcing_rows'][day_idx]
                next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
                if not _unlabeled_heat_closure_static_step_allowed(
                    row,
                    next_row,
                    open_water_only=open_water_only,
                    lst_qc_min=lst_qc_min,
                ):
                    allowed = False
                    break
            if not allowed:
                continue
        prediction, freezing_storage, init_detail = _initialize_unlabeled_heat_closure_state(
            model,
            lake,
            start_idx,
            state_source=state_source,
            state_spinup_days=state_spinup_days,
            zero_profile_initializer=zero_profile_initializer,
            spinup_lswt_observer_mode=spinup_lswt_observer_mode,
            zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
            task_mode=task_mode,
            hard_density_stability=hard_density_stability,
            dtype=dtype,
        )
        spinup_days_sum = spinup_days_sum + init_detail['spinup_days'].detach()
        state_window_count = state_window_count + torch.tensor(1.0, device=device, dtype=dtype)
        heat_start = _effective_profile_heat_content_j_m2(
            model,
            prediction,
            lake,
            freezing_storage,
        )
        budget_sum = torch.zeros(prediction.shape[0], device=device, dtype=dtype)
        for day_idx in range(start_idx, end_idx):
            row = lake['forcing_rows'][day_idx]
            next_row = lake['forcing_rows'][day_idx + 1] if day_idx + 1 < len(lake['forcing_rows']) else None
            budget_sum = budget_sum + _unlabeled_heat_storage_budget_prior_wm2(
                row,
                prediction,
            ).reshape(-1)
            if compute_solver_guard:
                prediction, freezing_storage, diagnostics = model.step(
                    prediction,
                    row,
                    lake['static_features'],
                    next_forcing_row=next_row,
                    return_diagnostics=True,
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=lake['area'],
                    diagnostic_mode=closure_step_diagnostic_mode,
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage,
                    return_freezing_storage=True,
                )
                solver_residual = (
                    diagnostics['heat_tendency_wm2'] - diagnostics['heat_input_wm2']
                ).reshape(-1)
                solver_finite_mask = torch.isfinite(solver_residual)
                solver_finite_f = solver_finite_mask.to(dtype=dtype)
                safe_solver_residual = torch.where(
                    solver_finite_mask,
                    solver_residual,
                    torch.zeros_like(solver_residual),
                )
                solver_guard_loss_sum = solver_guard_loss_sum + (
                    _thresholded_scaled_residual(safe_solver_residual, solver_guard_tau)
                    * solver_finite_f
                ).sum()
                solver_residual_abs_sum = solver_residual_abs_sum + (
                    torch.abs(safe_solver_residual).detach() * solver_finite_f
                ).sum()
                solver_residual_bias_sum = solver_residual_bias_sum + (
                    safe_solver_residual.detach() * solver_finite_f
                ).sum()
                solver_guard_active_loss_count = (
                    solver_guard_active_loss_count + solver_finite_f.sum()
                )
            else:
                prediction, freezing_storage = model.step(
                    prediction,
                    row,
                    lake['static_features'],
                    next_forcing_row=next_row,
                    return_diagnostics=False,
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=lake['area'],
                    diagnostic_mode='none',
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=freezing_storage,
                    return_freezing_storage=True,
                )
            if compute_lst_surface and next_row is not None:
                lst_loss_vec, lst_weight, lst_mask = _no_profile_lst_surface_loss_per_sample(
                    prediction,
                    next_row,
                    lst_qc_min=lst_qc_min,
                    open_water_only=open_water_only,
                )
                lst_mask_f = lst_mask.to(device=device, dtype=dtype)
                lst_surface_loss_sum = lst_surface_loss_sum + (
                    lst_loss_vec.reshape(-1) * lst_mask_f
                ).sum()
                lst_surface_weight_sum = lst_surface_weight_sum + (
                    lst_weight.reshape(-1) * lst_mask_f
                ).sum()
                lst_surface_supervision_count = lst_surface_supervision_count + lst_mask_f.sum()
            step_count += int(prediction.shape[0])
        heat_end = _effective_profile_heat_content_j_m2(
            model,
            prediction,
            lake,
            freezing_storage,
        )
        storage_tendency = (heat_end - heat_start) / (float(duration) * float(SECONDS_PER_DAY))
        budget_target = budget_sum / float(duration)
        residual = storage_tendency - budget_target
        finite_mask = torch.isfinite(residual)
        finite_f = finite_mask.to(dtype=dtype)
        finite_count = finite_f.sum()
        safe_residual = torch.where(finite_mask, residual, torch.zeros_like(residual))
        safe_storage_tendency = torch.where(finite_mask, storage_tendency, torch.zeros_like(storage_tendency))
        safe_budget_target = torch.where(finite_mask, budget_target, torch.zeros_like(budget_target))
        window_loss = (
            _storage_budget_residual_loss(safe_residual, tau, closure_mode) * finite_f
        ).sum() / torch.clamp(finite_count, min=1.0)
        valid_window = (finite_count > 0.0).to(dtype=dtype)
        if bool((finite_count > 0.0).detach().cpu().item()):
            horizon_value = torch.tensor(float(duration), device=device, dtype=dtype)
            horizon_days_sum = horizon_days_sum + horizon_value
            horizon_days_min = torch.minimum(horizon_days_min, horizon_value)
            horizon_days_max = torch.maximum(horizon_days_max, horizon_value)
            horizon_valid_count = horizon_valid_count + torch.tensor(1.0, device=device, dtype=dtype)
            horizon_seen.add(int(duration))
        window_loss_sum = window_loss_sum + window_loss * valid_window
        valid_window_count = valid_window_count + valid_window
        residual_abs_sum = residual_abs_sum + (torch.abs(safe_residual).detach() * finite_f).sum()
        residual_bias_sum = residual_bias_sum + (safe_residual.detach() * finite_f).sum()
        storage_tendency_sum = storage_tendency_sum + (safe_storage_tendency.detach() * finite_f).sum()
        budget_target_sum = budget_target_sum + (safe_budget_target.detach() * finite_f).sum()
        active_loss_count = active_loss_count + finite_count

    active_denominator = torch.clamp(active_loss_count, min=1.0)
    solver_guard_denominator = torch.clamp(solver_guard_active_loss_count, min=1.0)
    lst_surface_denominator = torch.clamp(lst_surface_supervision_count, min=1.0)
    loss = window_loss_sum / torch.clamp(valid_window_count, min=1.0)
    solver_guard_loss = solver_guard_loss_sum / solver_guard_denominator
    solver_guard_weighted_loss = float(solver_guard_effective_weight) * solver_guard_loss
    lst_surface_loss = lst_surface_loss_sum / lst_surface_denominator
    lst_surface_weighted_loss = float(no_profile_lst_surface_effective_weight) * lst_surface_loss
    weighted_loss = (
        float(effective_weight) * loss
        + solver_guard_weighted_loss
        + lst_surface_weighted_loss
    )
    residual_abs_mean = residual_abs_sum / active_denominator
    residual_bias_mean = residual_bias_sum / active_denominator
    solver_residual_abs_mean = solver_residual_abs_sum / solver_guard_denominator
    solver_residual_bias_mean = solver_residual_bias_sum / solver_guard_denominator
    lst_surface_weight_mean = lst_surface_weight_sum / lst_surface_denominator
    storage_tendency_mean = storage_tendency_sum / active_denominator
    budget_target_mean = budget_target_sum / active_denominator
    spinup_days_used = spinup_days_sum / torch.clamp(state_window_count, min=1.0)
    horizon_denominator = torch.clamp(horizon_valid_count, min=1.0)
    horizon_days_mean = horizon_days_sum / horizon_denominator
    horizon_days_min = torch.where(
        horizon_valid_count > 0.0,
        horizon_days_min,
        torch.zeros_like(horizon_days_min),
    )
    detail = {
        'unlabeled_heat_closure_loss': loss.detach(),
        'unlabeled_heat_closure_weighted_loss': weighted_loss.detach(),
        'unlabeled_heat_closure_weight': torch.tensor(float(weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_effective_weight': torch.tensor(float(effective_weight), device=device, dtype=dtype),
        'unlabeled_heat_closure_window_count': valid_window_count.detach(),
        'unlabeled_heat_closure_step_count': torch.tensor(float(step_count), device=device, dtype=dtype),
        'unlabeled_heat_closure_active_loss_count': active_loss_count.detach(),
        'unlabeled_heat_closure_residual_abs_mean_wm2': residual_abs_mean.detach(),
        'unlabeled_heat_closure_residual_bias_mean_wm2': residual_bias_mean.detach(),
        'unlabeled_heat_closure_budget_residual_abs_mean_wm2': residual_abs_mean.detach(),
        'unlabeled_heat_closure_budget_residual_bias_mean_wm2': residual_bias_mean.detach(),
        'unlabeled_heat_closure_budget_storage_tendency_mean_wm2': storage_tendency_mean.detach(),
        'unlabeled_heat_closure_budget_target_mean_wm2': budget_target_mean.detach(),
        'unlabeled_heat_closure_solver_residual_abs_mean_wm2': solver_residual_abs_mean.detach(),
        'unlabeled_heat_closure_solver_residual_bias_mean_wm2': solver_residual_bias_mean.detach(),
        'unlabeled_heat_closure_lst_surface_loss': lst_surface_loss.detach(),
        'unlabeled_heat_closure_lst_surface_weighted_loss': lst_surface_weighted_loss.detach(),
        'unlabeled_heat_closure_lst_surface_weight': torch.tensor(
            float(no_profile_lst_surface_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_effective_weight': torch.tensor(
            float(no_profile_lst_surface_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_lst_surface_supervision_count': (
            lst_surface_supervision_count.detach()
        ),
        'unlabeled_heat_closure_lst_surface_weight_mean': lst_surface_weight_mean.detach(),
        'unlabeled_heat_closure_solver_guard_loss': solver_guard_loss.detach(),
        'unlabeled_heat_closure_solver_guard_weighted_loss': solver_guard_weighted_loss.detach(),
        'unlabeled_heat_closure_solver_guard_weight': torch.tensor(
            float(solver_guard_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_effective_weight': torch.tensor(
            float(solver_guard_effective_weight),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_tau_wm2': torch.tensor(
            float(solver_guard_tau),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_solver_guard_active_loss_count': solver_guard_active_loss_count.detach(),
        'unlabeled_heat_closure_tau_wm2': torch.tensor(float(tau), device=device, dtype=dtype),
        'unlabeled_heat_closure_window_days': torch.tensor(float(window_days), device=device, dtype=dtype),
        'unlabeled_heat_closure_horizon_days_mean': horizon_days_mean.detach(),
        'unlabeled_heat_closure_horizon_days_min': horizon_days_min.detach(),
        'unlabeled_heat_closure_horizon_days_max': horizon_days_max.detach(),
        'unlabeled_heat_closure_horizon_count': torch.tensor(
            float(len(horizon_seen)),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_profile_label_count': torch.tensor(0.0, device=device, dtype=dtype),
        'unlabeled_heat_closure_reservoir_diagnostic_only_count': torch.tensor(
            1.0 if is_reservoir and reservoir_mode == 'diagnostic_only' else 0.0,
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_reservoir_excluded_count': torch.tensor(0.0, device=device, dtype=dtype),
        'unlabeled_heat_closure_state_source_code': torch.tensor(
            _unlabeled_heat_closure_state_source_code(state_source),
            device=device,
            dtype=dtype,
        ),
        'unlabeled_heat_closure_spinup_days_used': spinup_days_used.detach(),
        'unlabeled_heat_closure_spinup_days_config': torch.tensor(
            float(state_spinup_days),
            device=device,
            dtype=dtype,
        ),
    }
    return weighted_loss, detail


def _unlabeled_heat_closure_loss_for_lake(
    model,
    lake,
    windows,
    *,
    weight=DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT,
    window_days=DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS,
    tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2,
    open_water_only=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
    lst_qc_min=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
    reservoir_mode=DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE,
    mode=DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
    state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    state_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    solver_guard_weight=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
    solver_guard_tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    zero_profile_lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    window_step_cache=None,
):
    device = lake['depths'].device
    dtype = lake['depths'].dtype
    weight = float(weight)
    reservoir_mode = _normalize_unlabeled_heat_closure_reservoir_mode(reservoir_mode)
    closure_mode = _normalize_unlabeled_heat_closure_mode(mode)
    state_source = _normalize_unlabeled_heat_closure_state_source(state_source)
    state_spinup_days = max(0, int(state_spinup_days))
    no_profile_lst_surface_weight = max(0.0, float(no_profile_lst_surface_weight))
    solver_guard_weight = max(0.0, float(solver_guard_weight))
    solver_guard_tau = max(float(solver_guard_tau_wm2), 1.0e-6)
    open_water_only = _normalize_on_off(open_water_only, name='unlabeled_heat_closure_open_water_only')
    is_reservoir = _lake_reservoir_bucket(lake) == 'reservoir'
    effective_weight = weight
    no_profile_lst_surface_effective_weight = no_profile_lst_surface_weight
    solver_guard_effective_weight = solver_guard_weight
    if is_reservoir and reservoir_mode == 'exclude':
        detail = _zero_unlabeled_heat_closure_detail(
            device,
            dtype=dtype,
            weight=weight,
            effective_weight=0.0,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
            no_profile_lst_surface_effective_weight=0.0,
            solver_guard_weight=solver_guard_weight,
            solver_guard_effective_weight=0.0,
            solver_guard_tau=solver_guard_tau,
            state_source=state_source,
            state_spinup_days=state_spinup_days,
        )
        detail['unlabeled_heat_closure_reservoir_excluded_count'] = torch.tensor(1.0, device=device, dtype=dtype)
        return torch.tensor(0.0, device=device, dtype=dtype), detail
    if is_reservoir and reservoir_mode == 'diagnostic_only':
        effective_weight = 0.0
        no_profile_lst_surface_effective_weight = 0.0
        solver_guard_effective_weight = 0.0

    windows = tuple(windows or ())
    if not windows:
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            _zero_unlabeled_heat_closure_detail(
                device,
                dtype=dtype,
                weight=weight,
                effective_weight=effective_weight,
                no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                no_profile_lst_surface_effective_weight=no_profile_lst_surface_effective_weight,
                solver_guard_weight=solver_guard_weight,
                solver_guard_effective_weight=solver_guard_effective_weight,
                solver_guard_tau=solver_guard_tau,
                state_source=state_source,
                state_spinup_days=state_spinup_days,
            ),
        )

    tau = max(float(tau_wm2), 1.0e-6)
    batched_budget_result = _unlabeled_heat_storage_budget_loss_for_lake_batched(
        model,
        lake,
        windows,
        weight=weight,
        effective_weight=effective_weight,
        window_days=window_days,
        tau=tau,
        is_reservoir=is_reservoir,
        reservoir_mode=reservoir_mode,
        closure_mode=closure_mode,
        state_source=state_source,
        state_spinup_days=state_spinup_days,
        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        no_profile_lst_surface_effective_weight=no_profile_lst_surface_effective_weight,
        solver_guard_weight=solver_guard_weight,
        solver_guard_effective_weight=solver_guard_effective_weight,
        solver_guard_tau=solver_guard_tau,
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=spinup_lswt_observer_mode,
        zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
        task_mode=task_mode,
        hard_density_stability=hard_density_stability,
        step_diagnostic_mode=step_diagnostic_mode,
        window_step_cache=window_step_cache,
        open_water_only=open_water_only,
        lst_qc_min=lst_qc_min,
    )
    if batched_budget_result is not None:
        return batched_budget_result
    return _unlabeled_heat_storage_budget_loss_for_lake_unbatched(
        model,
        lake,
        windows,
        weight=weight,
        effective_weight=effective_weight,
        window_days=window_days,
        tau=tau,
        is_reservoir=is_reservoir,
        reservoir_mode=reservoir_mode,
        closure_mode=closure_mode,
        state_source=state_source,
        state_spinup_days=state_spinup_days,
        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        no_profile_lst_surface_effective_weight=no_profile_lst_surface_effective_weight,
        solver_guard_weight=solver_guard_weight,
        solver_guard_effective_weight=solver_guard_effective_weight,
        solver_guard_tau=solver_guard_tau,
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=spinup_lswt_observer_mode,
        zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
        task_mode=task_mode,
        hard_density_stability=hard_density_stability,
        step_diagnostic_mode=step_diagnostic_mode,
        window_step_cache=window_step_cache,
        open_water_only=open_water_only,
        lst_qc_min=lst_qc_min,
    )


def _unlabeled_heat_closure_training_record(model, lake, *, epoch, batch_size, **kwargs):
    windows = _select_unlabeled_heat_closure_windows(lake, batch_size, epoch)
    window_step_cache = _unlabeled_heat_closure_window_step_cache(
        lake,
        open_water_only=kwargs.get(
            'open_water_only',
            DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
        ),
        lst_qc_min=kwargs.get(
            'lst_qc_min',
            DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
        ),
    )
    return _unlabeled_heat_closure_loss_for_lake(
        model,
        lake,
        windows,
        window_step_cache=window_step_cache,
        **kwargs,
    )


def _select_zero_profile_init_net_dates(
    lake,
    split_key,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
    lookup = lake['lookups'].get(split_key, {})
    dates = sorted(
        pd.Timestamp(date).normalize()
        for date in lookup
        if pd.Timestamp(date).normalize() in lake['date_to_index']
    )
    if not dates:
        return ()
    samples_per_lake = int(samples_per_lake)
    if samples_per_lake <= 0 or len(dates) <= samples_per_lake:
        return tuple(dates)
    if _normalize_profile_sampling_mode(sampling_mode) == 'season_balanced':
        return tuple(_season_balanced_sample(
            dates,
            samples_per_lake,
            epoch,
            date_getter=lambda date: date,
            index_getter=lambda date: int(lake['date_to_index'][date]),
        ))
    indices = np.asarray([int(lake['date_to_index'][date]) for date in dates], dtype=np.float64)
    if samples_per_lake == 1:
        return (dates[int(epoch) % len(dates)],)
    anchors = np.linspace(float(indices[0]), float(indices[-1]), num=samples_per_lake)
    selected = []
    used = set()
    for anchor in anchors:
        candidates = sorted(
            (idx for idx in range(len(dates)) if idx not in used),
            key=lambda idx: (abs(float(indices[idx]) - float(anchor)), float(indices[idx])),
        )
        if not candidates:
            break
        chosen = candidates[int(epoch) % len(candidates)]
        used.add(chosen)
        selected.append(dates[chosen])
    return tuple(selected)


def _zero_profile_init_net_rollout_target_dates(
    lake,
    split_key,
    anchor_date,
    anchor_idx,
    *,
    max_days=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
    max_targets=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
):
    """Future train-profile targets for end-to-end init-state -> physics rollout loss."""
    lookup = lake['lookups'].get(split_key, {})
    if not lookup:
        return ()
    anchor_date = pd.Timestamp(anchor_date).normalize()
    anchor_idx = int(anchor_idx)
    max_days = int(max_days)
    max_targets = int(max_targets)
    if max_days <= 0:
        return ()
    candidates = []
    for date_value in sorted(pd.Timestamp(date).normalize() for date in lookup):
        if date_value == anchor_date or date_value not in lake['date_to_index']:
            continue
        target_idx = int(lake['date_to_index'][date_value])
        gap_days = target_idx - anchor_idx
        if gap_days <= 0 or gap_days > max_days:
            continue
        if not _forcing_rows_have_step_features(lake, anchor_idx, target_idx):
            continue
        candidates.append((date_value, target_idx))
    if not candidates:
        return ()
    if max_targets <= 0 or len(candidates) <= max_targets:
        return tuple(date for date, _idx in candidates)
    anchors = np.linspace(0, len(candidates) - 1, num=max_targets)
    selected = []
    used = set()
    for anchor in anchors:
        order = sorted(
            (idx for idx in range(len(candidates)) if idx not in used),
            key=lambda idx: (abs(float(idx) - float(anchor)), candidates[idx][1]),
        )
        if not order:
            break
        chosen = order[0]
        used.add(chosen)
        selected.append(candidates[chosen][0])
    return tuple(selected)


def _zero_profile_init_net_empty_detail(device):
    zero = torch.tensor(0.0, device=device)
    return {
        'zero_profile_init_net_loss': zero,
        'zero_profile_init_net_profile_loss': zero,
        'zero_profile_init_net_direct_profile_loss': zero,
        'zero_profile_init_net_spinup_profile_loss': zero,
        'zero_profile_init_net_regularization_loss': zero,
        'zero_profile_init_net_physics_loss': zero,
        'zero_profile_init_net_profile_physics_loss': zero,
        'zero_profile_init_net_heat_content_constraint_loss': zero,
        'zero_profile_init_net_surface_bottom_constraint_loss': zero,
        'zero_profile_init_net_bounded_state_loss': zero,
        'zero_profile_init_net_rollout_profile_loss': zero,
        'zero_profile_init_net_rollout_weighted_loss': zero,
        'zero_profile_init_net_rollout_supervision_count': zero,
        'zero_profile_init_net_rollout_enabled_count': zero,
        'zero_profile_init_net_rollout_target_count': zero,
        'zero_profile_init_net_rollout_max_gap_days': zero,
        'zero_profile_init_net_supervision_count': zero,
        'zero_profile_init_net_spinup_enabled_count': zero,
        'zero_profile_init_net_spinup_days_used': zero,
        'zero_profile_init_net_band_profile_loss': zero,
        'zero_profile_init_net_surface_profile_loss': zero,
        'zero_profile_init_net_upper_profile_loss': zero,
        'zero_profile_init_net_mid_profile_loss': zero,
        'zero_profile_init_net_deep_profile_loss': zero,
        'zero_profile_init_net_conditioning_abs_mean': zero,
        'zero_profile_init_net_profile_fusion_gate_mean': zero,
        'zero_profile_init_net_profile_fusion_active_count': zero,
        'zero_profile_init_net_profile_fusion_delta_abs_mean_c': zero,
        'zero_profile_init_net_profile_fusion_age_days_mean': zero,
        'zero_profile_init_net_profile_fusion_coverage_fraction_mean': zero,
        'zero_profile_init_net_profile_fusion_depth_span_fraction_mean': zero,
        'zero_profile_init_net_profile_fusion_future_fraction_mean': zero,
    }


ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES = (
    'max_depth_norm',
    'mean_depth_norm',
    'mean_to_max_depth_ratio',
    'log_area_km2_norm',
    'log_volume_proxy_norm',
    'hypsometry_mean_area_ratio',
    'hypsometry_bottom_area_ratio',
    'hypsometry_area_cv',
    'latitude_norm',
    'elevation_norm',
    'reservoir_flag',
    'start_doy_sin',
    'start_doy_cos',
    'air_temp_mean_30d_norm',
    'air_temp_trend_30d_norm',
    'shortwave_mean_30d_norm',
    'net_flux_mean_30d_norm',
    'wind_mean_30d_norm',
    'wind_energy_mean_30d_norm',
    'lst_mean_30d_norm',
    'lst_observed_fraction_30d',
    'lst_filled_fraction_30d',
    'ice_fraction_mean_30d',
    'air_temp_mean_90d_norm',
    'air_temp_trend_90d_norm',
    'shortwave_mean_90d_norm',
    'net_flux_mean_90d_norm',
    'wind_mean_90d_norm',
    'wind_energy_mean_90d_norm',
    'lst_mean_90d_norm',
    'lst_observed_fraction_90d',
    'lst_filled_fraction_90d',
    'ice_fraction_mean_90d',
)


def _metadata_numeric(metadata, *keys, default=0.0):
    if not isinstance(metadata, dict):
        return float(default)
    for key in keys:
        if key in metadata:
            try:
                value = float(metadata.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return float(value)
    return float(default)


def _metadata_reservoir_flag(metadata):
    if not isinstance(metadata, dict):
        return 0.0
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


def _series_mean(window, column, default=0.0):
    if column not in window.columns or len(window) == 0:
        return float(default)
    series = pd.to_numeric(window[column], errors='coerce')
    value = float(series.mean()) if series.notna().any() else float(default)
    return value if np.isfinite(value) else float(default)


def _series_fraction(window, column, threshold=0.5, default=0.0):
    if column not in window.columns or len(window) == 0:
        return float(default)
    series = pd.to_numeric(window[column], errors='coerce')
    finite = series[np.isfinite(series)]
    if finite.empty:
        return float(default)
    return float((finite > float(threshold)).mean())


def _series_trend(window, column, default=0.0):
    if column not in window.columns or len(window) < 2:
        return float(default)
    series = pd.to_numeric(window[column], errors='coerce')
    finite = series[np.isfinite(series)]
    if len(finite) < 2:
        return float(default)
    return float(finite.iloc[-1] - finite.iloc[0])


def _zero_profile_init_conditioning_array(lake, start_idx):
    """Observable lake/forcing summary for lake-conditioned initial columns."""
    return _zero_profile_init_conditioning_from_inputs(
        lake.get('df'),
        lake.get('depths_np', lake.get('depths', [])),
        lake.get('metadata', {}),
        int(start_idx),
        area_profile=lake.get('area', lake.get('area_profile')),
    )


def _zero_profile_init_conditioning_batch(lake, start_indices, *, device=None, dtype=None):
    if torch.is_tensor(start_indices):
        indices = [int(value) for value in start_indices.detach().cpu().reshape(-1).tolist()]
        device = start_indices.device if device is None else device
    else:
        indices = [int(value) for value in start_indices]
    device = lake['depths'].device if device is None else device
    dtype = lake['depths'].dtype if dtype is None else dtype
    if not indices:
        return torch.zeros((0, ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM), device=device, dtype=dtype)
    matrix = np.stack([
        _zero_profile_init_conditioning_array(lake, idx)
        for idx in indices
    ], axis=0)
    return torch.as_tensor(matrix, device=device, dtype=dtype)


def _forcing_rows_have_step_features(lake, start_idx, end_idx):
    rows = lake.get('forcing_rows', ())
    start_idx = max(0, int(start_idx))
    end_idx = max(start_idx, int(end_idx))
    if end_idx <= start_idx:
        return True
    if not rows:
        return False
    if end_idx > len(rows):
        return False
    for idx in range(start_idx, end_idx):
        row = rows[idx]
        if row is None or not hasattr(row, 'get') or row.get('features') is None:
            return False
    return True


def _daily_memory_empty_detail(device):
    zero = torch.tensor(0.0, device=device)
    detail = {
        'daily_memory_loss': zero,
        'daily_memory_profile_loss': zero,
        'daily_memory_band_profile_loss': zero,
        'daily_memory_surface_profile_loss': zero,
        'daily_memory_upper_profile_loss': zero,
        'daily_memory_mid_profile_loss': zero,
        'daily_memory_deep_profile_loss': zero,
        'daily_memory_temporal_smoothness_loss': zero,
        'daily_memory_heat_budget_loss': zero,
        'daily_memory_heat_budget_residual_abs_mean_wm2': zero,
        'daily_memory_heat_budget_residual_bias_mean_wm2': zero,
        'daily_memory_heat_budget_step_count': zero,
        'daily_memory_no_profile_lst_surface_loss': zero,
        'daily_memory_no_profile_lst_surface_count': zero,
        'daily_memory_no_profile_lst_surface_weight_mean': zero,
        'daily_memory_physics_consistency_loss': zero,
        'daily_memory_physics_consistency_count': zero,
        'daily_memory_profile_physics_loss': zero,
        'daily_memory_state_constraint_loss': zero,
        'daily_memory_regularization_loss': zero,
        'daily_memory_regularization_weighted_loss': zero,
        'daily_memory_coefficient_target_loss': zero,
        'daily_memory_coefficient_target_clipped_fraction': zero,
        'daily_memory_coefficient_supervision_count': zero,
        'daily_memory_supervision_count': zero,
        'daily_memory_valid_depth_count': zero,
        'daily_memory_coefficient_abs_mean': zero,
        'daily_memory_coefficient_std': zero,
        'daily_memory_coefficient_smoothness': zero,
        'daily_memory_component_count': zero,
        'daily_memory_conditioning_abs_mean': zero,
        'daily_memory_profile_fusion_gate_mean': zero,
        'daily_memory_profile_fusion_active_count': zero,
        'daily_memory_profile_fusion_age_days_mean': zero,
        'daily_memory_profile_fusion_coverage_fraction_mean': zero,
        'daily_memory_profile_fusion_depth_span_fraction_mean': zero,
        'daily_memory_profile_fusion_future_fraction_mean': zero,
    }
    for component_idx in range(DAILY_MEMORY_HISTORY_COEFFICIENT_COMPONENTS):
        suffix = f'{component_idx + 1:02d}'
        detail[f'daily_memory_coeff_{suffix}_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_abs_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_unit_abs_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_next_delta_abs_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_target_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_target_error_abs_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_target_unit_mean'] = zero
        detail[f'daily_memory_coeff_{suffix}_target_unit_error_abs_mean'] = zero
    return detail


def _daily_memory_coefficient_detail(
    encoded,
    row_idx,
    *,
    next_encoded=None,
    target_coefficients=None,
    target_units=None,
):
    coeffs = encoded['coefficients']
    coeff_units = encoded.get('coefficient_unit')
    next_coeffs = None if next_encoded is None else next_encoded.get('coefficients')
    zero = torch.zeros((), device=coeffs.device, dtype=coeffs.dtype)
    detail = {}
    for component_idx in range(DAILY_MEMORY_HISTORY_COEFFICIENT_COMPONENTS):
        suffix = f'{component_idx + 1:02d}'
        if component_idx < coeffs.shape[1]:
            coeff = coeffs[row_idx, component_idx].detach()
            unit = (
                coeff_units[row_idx, component_idx].detach()
                if coeff_units is not None and component_idx < coeff_units.shape[1]
                else zero
            )
            if next_coeffs is not None and component_idx < next_coeffs.shape[1]:
                next_delta = (next_coeffs[row_idx, component_idx] - coeffs[row_idx, component_idx]).abs().detach()
            else:
                next_delta = zero
            if target_coefficients is not None and component_idx < target_coefficients.shape[1]:
                target_coeff = target_coefficients[row_idx, component_idx].detach()
                target_error = (coeffs[row_idx, component_idx] - target_coefficients[row_idx, component_idx]).abs().detach()
            else:
                target_coeff = zero
                target_error = zero
            if target_units is not None and component_idx < target_units.shape[1]:
                target_unit = target_units[row_idx, component_idx].detach()
                target_unit_error = (unit - target_units[row_idx, component_idx]).abs().detach()
            else:
                target_unit = zero
                target_unit_error = zero
        else:
            coeff = zero
            unit = zero
            next_delta = zero
            target_coeff = zero
            target_error = zero
            target_unit = zero
            target_unit_error = zero
        detail[f'daily_memory_coeff_{suffix}_mean'] = coeff
        detail[f'daily_memory_coeff_{suffix}_abs_mean'] = coeff.abs()
        detail[f'daily_memory_coeff_{suffix}_unit_abs_mean'] = unit.abs()
        detail[f'daily_memory_coeff_{suffix}_next_delta_abs_mean'] = next_delta
        detail[f'daily_memory_coeff_{suffix}_target_mean'] = target_coeff
        detail[f'daily_memory_coeff_{suffix}_target_error_abs_mean'] = target_error
        detail[f'daily_memory_coeff_{suffix}_target_unit_mean'] = target_unit
        detail[f'daily_memory_coeff_{suffix}_target_unit_error_abs_mean'] = target_unit_error
    return detail


def _select_daily_memory_dates(
    lake,
    split_key,
    samples_per_lake,
    epoch,
    *,
    sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
):
    return _select_zero_profile_init_net_dates(
        lake,
        split_key,
        samples_per_lake,
        epoch,
        sampling_mode=sampling_mode,
    )


def _select_daily_memory_no_profile_indices(lake, samples_per_lake, epoch):
    windows = tuple(lake.get('unlabeled_heat_closure_windows', ()))
    if windows:
        selected = _select_unlabeled_heat_closure_windows(lake, samples_per_lake, epoch)
        return tuple(int(start_idx) for start_idx, _end_idx in selected)
    profile_indices = {
        int(lake['date_to_index'][pd.Timestamp(date).normalize()])
        for date in lake['lookups'].get('all', {})
        if pd.Timestamp(date).normalize() in lake.get('date_to_index', {})
    }
    candidates = [
        idx for idx in range(0, max(0, len(lake.get('forcing_rows', ())) - 1))
        if idx not in profile_indices
    ]
    if not candidates:
        return ()
    samples_per_lake = int(samples_per_lake)
    if samples_per_lake <= 0 or len(candidates) <= samples_per_lake:
        return tuple(candidates)
    anchors = np.linspace(float(candidates[0]), float(candidates[-1]), num=samples_per_lake)
    selected = []
    used = set()
    for anchor in anchors:
        remaining = sorted(
            (idx for idx in candidates if idx not in used),
            key=lambda idx: (abs(float(idx) - float(anchor)), float(idx)),
        )
        if not remaining:
            break
        chosen = remaining[int(epoch) % len(remaining)]
        selected.append(chosen)
        used.add(chosen)
    return tuple(selected)


def _daily_memory_basis_tensors(model, lake):
    basis = getattr(model, 'zero_profile_thermal_basis', None)
    if basis is None:
        return None
    return zero_profile_thermal_basis_tensors_for_depths(
        basis,
        lake['depths_np'],
        device=lake['depths'].device,
        dtype=lake['depths'].dtype,
    )


def _daily_memory_prediction_batch(
    model,
    lake,
    start_indices,
    basis_tensors,
    *,
    profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    profile_fusion_lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
):
    if basis_tensors is None:
        raise ValueError('daily-memory reconstruction requires zero_profile_thermal_basis.')
    if torch.is_tensor(start_indices):
        index_tensor = start_indices.to(device=lake['depths'].device, dtype=torch.long).reshape(-1)
    else:
        index_tensor = torch.as_tensor(
            [int(value) for value in start_indices],
            dtype=torch.long,
            device=lake['depths'].device,
        )
    row_batch = _forcing_row_batch(lake, index_tensor)
    encoded = model.daily_memory_reconstruction_from_basis(
        row_batch['history_features'],
        row_batch['features'],
        lake['static_features'],
        basis_tensors['components_on_depth'],
        basis_tensors['coeff_std'],
        basis_tensors['mean_profile_on_depth'],
        conditioning_features=_zero_profile_init_conditioning_batch(
            lake,
            index_tensor,
            device=lake['depths'].device,
            dtype=lake['depths'].dtype,
        ),
    )
    fusion = _thermal_state_profile_fusion_batch(
        lake,
        index_tensor,
        basis_tensors,
        mode=profile_fusion_mode,
        branch='daily',
        lookup_split=profile_fusion_lookup_split,
        time_policy=profile_fusion_time_policy,
        max_age_days=profile_fusion_max_age_days,
        min_depth_fraction=profile_fusion_min_depth_fraction,
        max_weight=profile_fusion_max_weight,
        coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
    )
    return _apply_profile_fusion_to_daily_encoded(encoded, fusion, basis_tensors)


def _daily_memory_target_coefficients(
    encoded,
    target_batch,
    target_mask_batch,
    basis_tensors,
    *,
    coeff_limit_sigma=DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA,
):
    pred_coeffs = encoded['coefficients']
    pred_units = encoded.get('coefficient_unit')
    device = pred_coeffs.device
    dtype = pred_coeffs.dtype
    components = basis_tensors['components_on_depth'].to(device=device, dtype=dtype)
    mean_profile = basis_tensors['mean_profile_on_depth'].to(device=device, dtype=dtype).reshape(-1)
    coeff_std = basis_tensors['coeff_std'].to(device=device, dtype=dtype).reshape(-1)
    coeff_count = min(int(pred_coeffs.shape[1]), int(components.shape[0]), int(coeff_std.numel()))
    batch_size = int(pred_coeffs.shape[0])
    target_coeffs = torch.zeros((batch_size, coeff_count), device=device, dtype=dtype)
    target_units = torch.zeros((batch_size, coeff_count), device=device, dtype=dtype)
    clipped = torch.zeros((batch_size,), device=device, dtype=dtype)
    losses = torch.zeros((batch_size,), device=device, dtype=dtype)
    valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    if coeff_count <= 0:
        return losses, target_coeffs, target_units, clipped, valid
    eye = torch.eye(coeff_count, device=device, dtype=dtype)
    coeff_scale = torch.clamp(coeff_std[:coeff_count], min=1.0e-6)
    coeff_limit = max(float(coeff_limit_sigma), 1.0e-6)
    for row_idx in range(batch_size):
        mask = target_mask_batch[row_idx].to(device=device, dtype=torch.bool)
        target = target_batch[row_idx].to(device=device, dtype=dtype)
        mask = mask & torch.isfinite(target)
        if int(mask.detach().sum().item()) < 1:
            continue
        basis = components[:coeff_count, mask].transpose(0, 1)
        centered = target[mask] - mean_profile[mask]
        lhs = basis.transpose(0, 1) @ basis
        lhs = lhs + float(DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_RIDGE) * eye
        rhs = basis.transpose(0, 1) @ centered
        coeff_target = torch.linalg.solve(lhs, rhs)
        target_sigma = coeff_target.detach() / coeff_scale
        clipped_sigma = torch.clamp(target_sigma, -coeff_limit, coeff_limit)
        target_coeffs[row_idx] = clipped_sigma * coeff_scale
        target_unit = clipped_sigma / coeff_limit
        target_units[row_idx] = target_unit
        clipped[row_idx] = (target_sigma.abs() > coeff_limit).to(dtype=dtype).mean()
        if pred_units is not None and pred_units.shape[1] >= coeff_count:
            unit_error = pred_units[row_idx, :coeff_count] - target_unit
        else:
            pred_sigma = pred_coeffs[row_idx, :coeff_count] / coeff_scale
            unit_error = pred_sigma / coeff_limit - target_unit
        losses[row_idx] = torch.nn.functional.huber_loss(
            unit_error,
            torch.zeros_like(unit_error),
            reduction='mean',
            delta=1.0,
        )
        valid[row_idx] = True
    return losses, target_coeffs, target_units, clipped, valid


def _thermal_state_profile_fusion_enabled(mode, branch):
    mode = _normalize_thermal_state_profile_fusion_mode(mode)
    branch = str(branch or '').strip().lower()
    if mode == 'off':
        return False
    if mode == 'both':
        return branch in {'init', 'daily'}
    if mode == 'init_only':
        return branch == 'init'
    if mode == 'daily_only':
        return branch == 'daily'
    return False


def _profile_fusion_candidate_dates(lake, lookup_split):
    lookups = lake.get('lookups') or {}
    split_name = str(lookup_split or DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT)
    lookup = lookups.get(split_name)
    masks = (lake.get('lookup_masks') or {}).get(split_name, {})
    if not lookup and split_name != 'all':
        lookup = lookups.get('all')
        masks = (lake.get('lookup_masks') or {}).get('all', {})
    if not lookup:
        return ()
    date_to_index = lake.get('date_to_index') or {}
    candidates = []
    for raw_date, profile in lookup.items():
        date_value = pd.Timestamp(raw_date).normalize()
        if date_value not in date_to_index:
            continue
        candidates.append((
            date_value,
            int(date_to_index[date_value]),
            profile,
            masks.get(date_value),
        ))
    return tuple(sorted(candidates, key=lambda item: item[1]))


def _select_profile_fusion_candidate(
    lake,
    day_idx,
    *,
    lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
):
    policy = _normalize_thermal_state_profile_fusion_time_policy(time_policy)
    day_idx = int(day_idx)
    max_age_days = int(max_age_days)
    candidates = []
    for date_value, profile_idx, profile, mask in _profile_fusion_candidate_dates(lake, lookup_split):
        if policy == 'past_only' and profile_idx > day_idx:
            continue
        if policy == 'past_strict' and profile_idx >= day_idx:
            continue
        if policy == 'nearest_strict' and profile_idx == day_idx:
            continue
        age_days = abs(int(profile_idx) - day_idx)
        if max_age_days >= 0 and age_days > max_age_days:
            continue
        future_flag = 1 if profile_idx > day_idx else 0
        # On ties prefer past observations, then earlier dates.  Future use is
        # still allowed by the nearest policies, but it stays explicit.
        tie_future_penalty = 1 if future_flag else 0
        candidates.append((
            age_days,
            tie_future_penalty,
            int(profile_idx),
            date_value,
            profile,
            mask,
            future_flag,
        ))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    age_days, _tie, profile_idx, date_value, profile, mask, future_flag = candidates[0]
    return {
        'date': date_value,
        'day_idx': int(profile_idx),
        'age_days': int(age_days),
        'profile': profile,
        'mask': mask,
        'future_flag': int(future_flag),
    }


def _thermal_state_profile_fusion_batch(
    lake,
    day_indices,
    basis_tensors,
    *,
    mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    branch='daily',
    lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
):
    if torch.is_tensor(day_indices):
        index_tensor = day_indices.to(device=lake['depths'].device, dtype=torch.long).reshape(-1)
    else:
        index_tensor = torch.as_tensor(
            [int(value) for value in day_indices],
            dtype=torch.long,
            device=lake['depths'].device,
        )
    batch_size = int(index_tensor.numel())
    device = lake['depths'].device
    dtype = lake['depths'].dtype
    components = basis_tensors['components_on_depth'].to(device=device, dtype=dtype)
    mean_profile = basis_tensors['mean_profile_on_depth'].to(device=device, dtype=dtype).reshape(-1)
    coeff_std = basis_tensors['coeff_std'].to(device=device, dtype=dtype).reshape(-1)
    coeff_count = min(int(components.shape[0]), int(coeff_std.numel()))
    depth_count = int(mean_profile.numel())
    zeros_profile = torch.zeros((batch_size, depth_count), device=device, dtype=dtype)
    zeros_coeff = torch.zeros((batch_size, coeff_count), device=device, dtype=dtype)
    zeros = torch.zeros((batch_size,), device=device, dtype=dtype)
    empty = {
        'enabled': False,
        'gate': zeros,
        'valid': torch.zeros((batch_size,), device=device, dtype=torch.bool),
        'projection': zeros_profile,
        'coefficients': zeros_coeff,
        'age_days': zeros,
        'coverage_fraction': zeros,
        'depth_span_fraction': zeros,
        'future_fraction': zeros,
    }
    if batch_size <= 0 or coeff_count <= 0:
        return empty
    if not _thermal_state_profile_fusion_enabled(mode, branch):
        return empty

    max_age_days = int(max_age_days)
    min_depth_fraction = max(float(min_depth_fraction), 1.0e-6)
    max_weight = float(np.clip(float(max_weight), 0.0, 1.0))
    coeff_limit_sigma = max(float(coeff_limit_sigma), 1.0e-6)
    depths = lake['depths'].to(device=device, dtype=dtype).reshape(-1)
    max_depth = torch.clamp(torch.max(depths) - torch.min(depths), min=1.0e-6)
    eye = torch.eye(coeff_count, device=device, dtype=dtype)
    coeff_scale = torch.clamp(coeff_std[:coeff_count], min=1.0e-6)
    coeff_limit = coeff_limit_sigma * coeff_scale

    projection_rows = []
    coeff_rows = []
    gate_rows = []
    valid_rows = []
    age_rows = []
    coverage_rows = []
    span_rows = []
    future_rows = []
    lookup_split = str(lookup_split or DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT)
    for day_idx in index_tensor.detach().cpu().reshape(-1).tolist():
        candidate = _select_profile_fusion_candidate(
            lake,
            int(day_idx),
            lookup_split=lookup_split,
            time_policy=time_policy,
            max_age_days=max_age_days,
        )
        if candidate is None:
            projection_rows.append(torch.zeros_like(mean_profile))
            coeff_rows.append(torch.zeros((coeff_count,), device=device, dtype=dtype))
            gate_rows.append(torch.zeros((), device=device, dtype=dtype))
            valid_rows.append(False)
            age_rows.append(torch.zeros((), device=device, dtype=dtype))
            coverage_rows.append(torch.zeros((), device=device, dtype=dtype))
            span_rows.append(torch.zeros((), device=device, dtype=dtype))
            future_rows.append(torch.zeros((), device=device, dtype=dtype))
            continue
        profile = torch.as_tensor(candidate['profile'], device=device, dtype=dtype).reshape(-1)
        if int(profile.numel()) != depth_count:
            profile = torch.zeros_like(mean_profile)
            mask = torch.zeros((depth_count,), device=device, dtype=torch.bool)
        else:
            raw_mask = candidate.get('mask')
            if raw_mask is None:
                mask = torch.ones((depth_count,), device=device, dtype=torch.bool)
            else:
                mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool).reshape(-1)
                if int(mask.numel()) != depth_count:
                    mask = torch.ones((depth_count,), device=device, dtype=torch.bool)
        valid_mask = mask & torch.isfinite(profile) & torch.isfinite(mean_profile)
        valid_count = int(valid_mask.detach().sum().item())
        if valid_count >= 2:
            valid_depths = depths[valid_mask]
            span_fraction = (
                torch.clamp((torch.max(valid_depths) - torch.min(valid_depths)) / max_depth, 0.0, 1.0)
            )
        else:
            span_fraction = torch.zeros((), device=device, dtype=dtype)
        coverage_fraction = torch.as_tensor(
            float(valid_count) / float(max(depth_count, 1)),
            device=device,
            dtype=dtype,
        )
        if valid_count < 1:
            coeff = torch.zeros((coeff_count,), device=device, dtype=dtype)
            projection = torch.zeros_like(mean_profile)
            gate = torch.zeros((), device=device, dtype=dtype)
            is_valid = False
        else:
            design = components[:coeff_count, valid_mask].transpose(0, 1)
            centered = profile[valid_mask] - mean_profile[valid_mask]
            lhs = design.transpose(0, 1) @ design
            lhs = lhs + float(DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_RIDGE) * eye
            rhs = design.transpose(0, 1) @ centered
            coeff = torch.linalg.solve(lhs, rhs)
            coeff = torch.clamp(coeff, -coeff_limit, coeff_limit)
            projection = torch.clamp(mean_profile + coeff @ components[:coeff_count], 0.0, 38.0)
            coverage_score = torch.clamp(coverage_fraction / min_depth_fraction, 0.0, 1.0)
            span_score = torch.clamp(span_fraction / min_depth_fraction, 0.0, 1.0)
            if max_age_days > 0:
                age_score = max(0.0, 1.0 - float(candidate['age_days']) / float(max_age_days))
            elif max_age_days == 0:
                age_score = 1.0 if int(candidate['age_days']) == 0 else 0.0
            else:
                age_score = 1.0
            gate = (
                torch.as_tensor(max_weight * age_score, device=device, dtype=dtype)
                * coverage_score
                * span_score
            )
            gate = torch.clamp(gate, 0.0, 1.0)
            is_valid = bool(gate.detach().cpu().item() > 0.0)
        projection_rows.append(projection)
        coeff_rows.append(coeff)
        gate_rows.append(gate)
        valid_rows.append(is_valid)
        age_rows.append(torch.as_tensor(float(candidate['age_days']), device=device, dtype=dtype))
        coverage_rows.append(coverage_fraction)
        span_rows.append(span_fraction)
        future_rows.append(torch.as_tensor(float(candidate['future_flag']), device=device, dtype=dtype))

    return {
        'enabled': True,
        'gate': torch.stack(gate_rows).reshape(-1),
        'valid': torch.as_tensor(valid_rows, device=device, dtype=torch.bool),
        'projection': torch.stack(projection_rows, dim=0),
        'coefficients': torch.stack(coeff_rows, dim=0),
        'age_days': torch.stack(age_rows).reshape(-1),
        'coverage_fraction': torch.stack(coverage_rows).reshape(-1),
        'depth_span_fraction': torch.stack(span_rows).reshape(-1),
        'future_fraction': torch.stack(future_rows).reshape(-1),
    }


def _apply_profile_fusion_to_daily_encoded(encoded, fusion, basis_tensors):
    result = dict(encoded)
    gate = fusion['gate'].to(device=encoded['daily_profile_c'].device, dtype=encoded['daily_profile_c'].dtype)
    result['thermal_state_profile_fusion_gate'] = gate
    result['thermal_state_profile_fusion_valid'] = fusion['valid'].to(device=gate.device)
    result['thermal_state_profile_fusion_age_days'] = fusion['age_days'].to(device=gate.device, dtype=gate.dtype)
    result['thermal_state_profile_fusion_coverage_fraction'] = fusion['coverage_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    result['thermal_state_profile_fusion_depth_span_fraction'] = fusion['depth_span_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    result['thermal_state_profile_fusion_future_fraction'] = fusion['future_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    if not bool((gate > 0.0).any().detach().cpu().item()):
        return result
    coeffs = encoded['coefficients']
    profile_coeffs = fusion['coefficients'].to(device=coeffs.device, dtype=coeffs.dtype)
    coeff_count = min(int(coeffs.shape[1]), int(profile_coeffs.shape[1]))
    fused_coeffs = coeffs.clone()
    gate_col = gate.reshape(-1, 1)
    fused_coeffs[:, :coeff_count] = (
        coeffs[:, :coeff_count]
        + gate_col * (profile_coeffs[:, :coeff_count] - coeffs[:, :coeff_count])
    )
    components = basis_tensors['components_on_depth'].to(device=coeffs.device, dtype=coeffs.dtype)
    mean_profile = basis_tensors['mean_profile_on_depth'].to(device=coeffs.device, dtype=coeffs.dtype)
    fused_delta = fused_coeffs[:, :coeff_count] @ components[:coeff_count]
    fused_profile = torch.clamp(mean_profile.reshape(1, -1) + fused_delta, 0.0, 38.0)
    result['daily_profile_feature_c'] = encoded['daily_profile_c']
    result['daily_profile_c'] = fused_profile
    result['daily_delta_c'] = fused_delta
    result['coefficients_feature'] = coeffs
    result['coefficients'] = fused_coeffs
    result['coefficient_abs_mean'] = fused_coeffs.abs().mean(dim=1)
    result['coefficient_std'] = fused_coeffs.std(dim=1, unbiased=False)
    return result


def _apply_profile_fusion_to_init_encoded(encoded, fusion):
    result = dict(encoded)
    profile = encoded['initial_profile_c']
    gate = fusion['gate'].to(device=profile.device, dtype=profile.dtype)
    result['thermal_state_profile_fusion_gate'] = gate
    result['thermal_state_profile_fusion_valid'] = fusion['valid'].to(device=gate.device)
    result['thermal_state_profile_fusion_age_days'] = fusion['age_days'].to(device=gate.device, dtype=gate.dtype)
    result['thermal_state_profile_fusion_coverage_fraction'] = fusion['coverage_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    result['thermal_state_profile_fusion_depth_span_fraction'] = fusion['depth_span_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    result['thermal_state_profile_fusion_future_fraction'] = fusion['future_fraction'].to(
        device=gate.device,
        dtype=gate.dtype,
    )
    if not bool((gate > 0.0).any().detach().cpu().item()):
        result['thermal_state_profile_fusion_delta_abs_mean_c'] = torch.zeros_like(gate)
        return result
    projection = fusion['projection'].to(device=profile.device, dtype=profile.dtype)
    fused = torch.clamp(profile + gate.reshape(-1, 1) * (projection - profile), 0.0, 38.0)
    result['initial_profile_feature_c'] = profile
    result['initial_profile_c'] = fused
    result['thermal_state_profile_fusion_delta_abs_mean_c'] = (fused - profile).abs().mean(dim=1)
    return result


def _daily_memory_heat_and_physics_terms(
    model,
    lake,
    prediction,
    next_prediction,
    day_indices,
    *,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
):
    device = prediction.device
    dtype = prediction.dtype
    next_indices = day_indices + 1
    next_valid = next_indices < len(lake['forcing_rows'])
    zero = torch.zeros((prediction.shape[0],), device=device, dtype=dtype)
    if not bool(next_valid.any().detach().cpu().item()):
        return {
            'heat_budget_loss': zero,
            'heat_budget_residual': zero,
            'heat_budget_valid': torch.zeros_like(zero, dtype=torch.bool),
            'physics_consistency_loss': zero,
            'physics_consistency_valid': torch.zeros_like(zero, dtype=torch.bool),
        }
    safe_next_indices = torch.clamp(next_indices, max=len(lake['forcing_rows']) - 1)
    row_batch = _forcing_row_batch(lake, day_indices)
    next_row_batch = _forcing_row_batch(lake, safe_next_indices)
    heat_before = _effective_profile_heat_content_j_m2(
        model,
        prediction,
        lake,
        torch.zeros_like(prediction),
    )
    heat_after = _effective_profile_heat_content_j_m2(
        model,
        next_prediction,
        lake,
        torch.zeros_like(next_prediction),
    )
    storage_tendency = (heat_after - heat_before) / float(SECONDS_PER_DAY)
    budget_target = _unlabeled_heat_storage_budget_prior_wm2(row_batch, prediction).reshape(-1)
    residual = storage_tendency - budget_target
    finite = torch.isfinite(residual) & next_valid
    safe_residual = torch.where(finite, residual, torch.zeros_like(residual))
    heat_budget_loss = _storage_budget_residual_loss(
        safe_residual,
        DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2,
        DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
    )
    with torch.no_grad():
        physics_next = model.step(
            prediction.detach(),
            row_batch,
            lake['static_features'],
            next_forcing_row=next_row_batch,
            return_diagnostics=False,
            task_mode=task_mode,
            depths=lake['depths'],
            area_profile=lake['area'],
            diagnostic_mode='none' if step_diagnostic_mode == 'none' else 'loss_fast',
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=torch.zeros_like(prediction),
            return_freezing_storage=False,
        )
    physics_consistency_loss = _masked_rmse_profile_loss_per_sample(
        next_prediction,
        physics_next.detach(),
        None,
    )
    return {
        'heat_budget_loss': heat_budget_loss,
        'heat_budget_residual': safe_residual,
        'heat_budget_valid': finite,
        'physics_consistency_loss': physics_consistency_loss,
        'physics_consistency_valid': torch.isfinite(physics_consistency_loss) & next_valid,
    }


def _daily_memory_reconstruction_training_records(
    model,
    lake,
    *,
    split_key='train',
    epoch=0,
    samples_per_lake=0,
    temporal_smoothness_weight=DEFAULT_DAILY_MEMORY_TEMPORAL_SMOOTHNESS_WEIGHT,
    heat_budget_weight=DEFAULT_DAILY_MEMORY_HEAT_BUDGET_WEIGHT,
    physics_consistency_weight=DEFAULT_DAILY_MEMORY_PHYSICS_CONSISTENCY_WEIGHT,
    regularization_weight=DEFAULT_DAILY_MEMORY_REGULARIZATION_WEIGHT,
    coefficient_loss_weight=DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_WEIGHT,
    profile_sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    no_profile_lst_surface_open_water_only=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
    no_profile_lst_surface_lst_qc_min=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
    profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    profile_fusion_lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    collect_details=True,
):
    device = lake['depths'].device
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    if basis_tensors is None:
        return [], [_daily_memory_empty_detail(device)] if collect_details else []
    no_profile_lst_surface_weight = max(0.0, float(no_profile_lst_surface_weight))
    dates = _select_daily_memory_dates(
        lake,
        split_key,
        samples_per_lake,
        epoch,
        sampling_mode=profile_sampling_mode,
    )
    entries = []
    for date_value in dates:
        date_value = pd.Timestamp(date_value).normalize()
        if date_value not in lake['lookups'].get(split_key, {}):
            continue
        entries.append((date_value, int(lake['date_to_index'][date_value])))
    if not entries:
        return [], [_daily_memory_empty_detail(device)] if collect_details else []

    day_indices = torch.as_tensor([day_idx for _date, day_idx in entries], dtype=torch.long, device=device)
    encoded = _daily_memory_prediction_batch(
        model,
        lake,
        day_indices,
        basis_tensors,
        profile_fusion_mode=profile_fusion_mode,
        profile_fusion_time_policy=profile_fusion_time_policy,
        profile_fusion_lookup_split=profile_fusion_lookup_split,
        profile_fusion_max_age_days=profile_fusion_max_age_days,
        profile_fusion_min_depth_fraction=profile_fusion_min_depth_fraction,
        profile_fusion_max_weight=profile_fusion_max_weight,
        profile_fusion_coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
    )
    prediction = encoded['daily_profile_c']
    target_batch, target_mask_batch = _target_tensor_and_mask_batch(
        lake,
        split_key,
        [date_value for date_value, _day_idx in entries],
    )
    target_batch = target_batch.to(device=device, dtype=lake['depths'].dtype)
    target_mask_batch = target_mask_batch.to(device=device, dtype=torch.bool)
    full_profile_losses = _masked_rmse_profile_loss_per_sample(prediction, target_batch, target_mask_batch)
    band_profile_losses, band_losses, _band_counts = _zero_profile_init_net_depth_band_losses(
        prediction,
        target_batch,
        target_mask_batch,
        lake['depths'],
    )
    profile_losses = 0.75 * full_profile_losses + 0.25 * band_profile_losses
    (
        coefficient_target_losses,
        target_coefficients,
        target_units,
        coefficient_target_clipped_fraction,
        coefficient_target_valid,
    ) = (
        _daily_memory_target_coefficients(
            encoded,
            target_batch,
            target_mask_batch,
            basis_tensors,
            coeff_limit_sigma=getattr(
                getattr(model, 'daily_memory_head', None),
                'coeff_limit_sigma',
                DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA,
            ),
        )
    )
    profile_physics_losses = _profile_physics_loss_per_sample(prediction)
    state_constraint_losses, _state_parts = _zero_profile_init_state_physics_loss_per_sample(
        prediction,
        target_batch,
        target_mask_batch,
        lake,
    )
    coeffs = encoded['coefficients']
    next_indices = torch.clamp(day_indices + 1, max=len(lake['forcing_rows']) - 1)
    next_encoded = _daily_memory_prediction_batch(
        model,
        lake,
        next_indices,
        basis_tensors,
        profile_fusion_mode=profile_fusion_mode,
        profile_fusion_time_policy=profile_fusion_time_policy,
        profile_fusion_lookup_split=profile_fusion_lookup_split,
        profile_fusion_max_age_days=profile_fusion_max_age_days,
        profile_fusion_min_depth_fraction=profile_fusion_min_depth_fraction,
        profile_fusion_max_weight=profile_fusion_max_weight,
        profile_fusion_coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
    )
    next_prediction = next_encoded['daily_profile_c']
    next_coeffs = next_encoded['coefficients']
    next_valid = (day_indices + 1) < len(lake['forcing_rows'])
    coeff_smoothness = torch.where(
        next_valid,
        (next_coeffs - coeffs).pow(2).mean(dim=1),
        torch.zeros((len(entries),), device=device, dtype=prediction.dtype),
    )
    budget_terms = _daily_memory_heat_and_physics_terms(
        model,
        lake,
        prediction,
        next_prediction,
        day_indices,
        task_mode=task_mode,
        hard_density_stability=hard_density_stability,
        step_diagnostic_mode=step_diagnostic_mode,
    )
    heat_budget_loss = torch.where(
        budget_terms['heat_budget_valid'],
        budget_terms['heat_budget_loss'],
        torch.zeros_like(budget_terms['heat_budget_loss']),
    )
    physics_consistency_loss = torch.where(
        budget_terms['physics_consistency_valid'],
        budget_terms['physics_consistency_loss'],
        torch.zeros_like(budget_terms['physics_consistency_loss']),
    )
    regularization_losses = encoded['regularization_loss'].reshape(-1)
    regularization_weighted_losses = float(regularization_weight) * regularization_losses
    total_losses = (
        profile_losses
        + regularization_weighted_losses
        + profile_physics_losses
        + state_constraint_losses
        + float(coefficient_loss_weight) * coefficient_target_losses
        + float(temporal_smoothness_weight) * coeff_smoothness
        + float(heat_budget_weight) * heat_budget_loss
        + float(physics_consistency_weight) * physics_consistency_loss
    )
    valid_counts = target_mask_batch.to(dtype=prediction.dtype).reshape(len(entries), -1).sum(dim=1)

    losses = []
    details = []
    supervision_count = torch.as_tensor(1.0, device=device, dtype=prediction.dtype)
    for row_idx, loss in enumerate(total_losses.unbind(dim=0)):
        losses.append(loss)
        if not collect_details:
            continue
        heat_valid = budget_terms['heat_budget_valid'][row_idx]
        physics_valid = budget_terms['physics_consistency_valid'][row_idx]
        heat_count = torch.as_tensor(
            1.0 if bool(heat_valid.detach().cpu().item()) else 0.0,
            device=device,
            dtype=prediction.dtype,
        )
        physics_count = torch.as_tensor(
            1.0 if bool(physics_valid.detach().cpu().item()) else 0.0,
            device=device,
            dtype=prediction.dtype,
        )
        coeff_count = torch.as_tensor(
            1.0 if bool(coefficient_target_valid[row_idx].detach().cpu().item()) else 0.0,
            device=device,
            dtype=prediction.dtype,
        )
        residual = budget_terms['heat_budget_residual'][row_idx]
        heat_valid_bool = bool(heat_valid.detach().cpu().item())
        details.append({
            'daily_memory_loss': loss.detach(),
            'daily_memory_profile_loss': profile_losses[row_idx].detach(),
            'daily_memory_band_profile_loss': band_profile_losses[row_idx].detach(),
            'daily_memory_surface_profile_loss': band_losses['surface'][row_idx].detach(),
            'daily_memory_upper_profile_loss': band_losses['upper'][row_idx].detach(),
            'daily_memory_mid_profile_loss': band_losses['mid'][row_idx].detach(),
            'daily_memory_deep_profile_loss': band_losses['deep'][row_idx].detach(),
            'daily_memory_temporal_smoothness_loss': coeff_smoothness[row_idx].detach(),
            'daily_memory_heat_budget_loss': heat_budget_loss[row_idx].detach(),
            'daily_memory_heat_budget_residual_abs_mean_wm2': (
                torch.abs(residual).detach() if heat_valid_bool else torch.zeros_like(residual)
            ),
            'daily_memory_heat_budget_residual_bias_mean_wm2': (
                residual.detach() if heat_valid_bool else torch.zeros_like(residual)
            ),
            'daily_memory_heat_budget_step_count': heat_count,
            'daily_memory_no_profile_lst_surface_loss': torch.zeros_like(loss.detach()),
            'daily_memory_no_profile_lst_surface_count': torch.zeros_like(loss.detach()),
            'daily_memory_no_profile_lst_surface_weight_mean': torch.zeros_like(loss.detach()),
            'daily_memory_physics_consistency_loss': physics_consistency_loss[row_idx].detach(),
            'daily_memory_physics_consistency_count': physics_count,
            'daily_memory_profile_physics_loss': profile_physics_losses[row_idx].detach(),
            'daily_memory_state_constraint_loss': state_constraint_losses[row_idx].detach(),
            'daily_memory_regularization_loss': regularization_losses[row_idx].detach(),
            'daily_memory_regularization_weighted_loss': regularization_weighted_losses[row_idx].detach(),
            'daily_memory_coefficient_target_loss': coefficient_target_losses[row_idx].detach(),
            'daily_memory_coefficient_target_clipped_fraction': (
                coefficient_target_clipped_fraction[row_idx].detach()
            ),
            'daily_memory_coefficient_supervision_count': coeff_count,
            'daily_memory_supervision_count': supervision_count,
            'daily_memory_valid_depth_count': valid_counts[row_idx].detach(),
            'daily_memory_coefficient_abs_mean': encoded['coefficient_abs_mean'][row_idx].detach(),
            'daily_memory_coefficient_std': encoded['coefficient_std'][row_idx].detach(),
            'daily_memory_coefficient_smoothness': coeff_smoothness[row_idx].detach(),
            'daily_memory_component_count': encoded['component_count'][row_idx].detach(),
            'daily_memory_conditioning_abs_mean': encoded['conditioning_abs_mean'][row_idx].detach(),
            'daily_memory_profile_fusion_gate_mean': (
                encoded['thermal_state_profile_fusion_gate'][row_idx].detach()
            ),
            'daily_memory_profile_fusion_active_count': (
                (encoded['thermal_state_profile_fusion_gate'][row_idx] > 0.0)
                .to(dtype=prediction.dtype)
                .detach()
            ),
            'daily_memory_profile_fusion_age_days_mean': (
                encoded['thermal_state_profile_fusion_age_days'][row_idx].detach()
            ),
            'daily_memory_profile_fusion_coverage_fraction_mean': (
                encoded['thermal_state_profile_fusion_coverage_fraction'][row_idx].detach()
            ),
            'daily_memory_profile_fusion_depth_span_fraction_mean': (
                encoded['thermal_state_profile_fusion_depth_span_fraction'][row_idx].detach()
            ),
            'daily_memory_profile_fusion_future_fraction_mean': (
                encoded['thermal_state_profile_fusion_future_fraction'][row_idx].detach()
            ),
        })
        details[-1].update(_daily_memory_coefficient_detail(
            encoded,
            row_idx,
            next_encoded=next_encoded,
            target_coefficients=target_coefficients,
            target_units=target_units,
        ))
    no_profile_indices = _select_daily_memory_no_profile_indices(lake, samples_per_lake, epoch)
    if no_profile_indices:
        no_profile_day_indices = torch.as_tensor(
            no_profile_indices,
            dtype=torch.long,
            device=device,
        )
        no_profile_encoded = _daily_memory_prediction_batch(
            model,
            lake,
            no_profile_day_indices,
            basis_tensors,
            profile_fusion_mode=profile_fusion_mode,
            profile_fusion_time_policy=profile_fusion_time_policy,
            profile_fusion_lookup_split=profile_fusion_lookup_split,
            profile_fusion_max_age_days=profile_fusion_max_age_days,
            profile_fusion_min_depth_fraction=profile_fusion_min_depth_fraction,
            profile_fusion_max_weight=profile_fusion_max_weight,
            profile_fusion_coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
        )
        no_profile_prediction = no_profile_encoded['daily_profile_c']
        no_profile_next_indices = torch.clamp(
            no_profile_day_indices + 1,
            max=len(lake['forcing_rows']) - 1,
        )
        no_profile_next_encoded = _daily_memory_prediction_batch(
            model,
            lake,
            no_profile_next_indices,
            basis_tensors,
            profile_fusion_mode=profile_fusion_mode,
            profile_fusion_time_policy=profile_fusion_time_policy,
            profile_fusion_lookup_split=profile_fusion_lookup_split,
            profile_fusion_max_age_days=profile_fusion_max_age_days,
            profile_fusion_min_depth_fraction=profile_fusion_min_depth_fraction,
            profile_fusion_max_weight=profile_fusion_max_weight,
            profile_fusion_coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
        )
        no_profile_next_prediction = no_profile_next_encoded['daily_profile_c']
        no_profile_next_valid = (no_profile_day_indices + 1) < len(lake['forcing_rows'])
        no_profile_coeff_smoothness = torch.where(
            no_profile_next_valid,
            (
                no_profile_next_encoded['coefficients']
                - no_profile_encoded['coefficients']
            ).pow(2).mean(dim=1),
            torch.zeros(
                (len(no_profile_indices),),
                device=device,
                dtype=no_profile_prediction.dtype,
            ),
        )
        no_profile_budget_terms = _daily_memory_heat_and_physics_terms(
            model,
            lake,
            no_profile_prediction,
            no_profile_next_prediction,
            no_profile_day_indices,
            task_mode=task_mode,
            hard_density_stability=hard_density_stability,
            step_diagnostic_mode=step_diagnostic_mode,
        )
        no_profile_heat_loss = torch.where(
            no_profile_budget_terms['heat_budget_valid'],
            no_profile_budget_terms['heat_budget_loss'],
            torch.zeros_like(no_profile_budget_terms['heat_budget_loss']),
        )
        no_profile_physics_consistency = torch.where(
            no_profile_budget_terms['physics_consistency_valid'],
            no_profile_budget_terms['physics_consistency_loss'],
            torch.zeros_like(no_profile_budget_terms['physics_consistency_loss']),
        )
        if no_profile_lst_surface_weight > 0.0:
            (
                no_profile_lst_surface_loss,
                no_profile_lst_surface_quality_weight,
                no_profile_lst_surface_mask,
            ) = _no_profile_lst_surface_loss_per_sample(
                no_profile_prediction,
                _forcing_row_batch(lake, no_profile_day_indices),
                lst_qc_min=no_profile_lst_surface_lst_qc_min,
                open_water_only=no_profile_lst_surface_open_water_only,
            )
        else:
            no_profile_lst_surface_loss = torch.zeros(
                (len(no_profile_indices),),
                device=device,
                dtype=no_profile_prediction.dtype,
            )
            no_profile_lst_surface_quality_weight = torch.zeros_like(no_profile_lst_surface_loss)
            no_profile_lst_surface_mask = torch.zeros_like(no_profile_lst_surface_loss, dtype=torch.bool)
        no_profile_profile_physics = _profile_physics_loss_per_sample(no_profile_prediction)
        no_profile_regularization = no_profile_encoded['regularization_loss'].reshape(-1)
        no_profile_regularization_weighted = float(regularization_weight) * no_profile_regularization
        no_profile_total = (
            no_profile_regularization_weighted
            + no_profile_profile_physics
            + float(temporal_smoothness_weight) * no_profile_coeff_smoothness
            + float(heat_budget_weight) * no_profile_heat_loss
            + float(physics_consistency_weight) * no_profile_physics_consistency
            + float(no_profile_lst_surface_weight) * no_profile_lst_surface_loss
        )
        for row_idx, loss in enumerate(no_profile_total.unbind(dim=0)):
            losses.append(loss)
            if not collect_details:
                continue
            heat_valid = no_profile_budget_terms['heat_budget_valid'][row_idx]
            physics_valid = no_profile_budget_terms['physics_consistency_valid'][row_idx]
            heat_valid_bool = bool(heat_valid.detach().cpu().item())
            residual = no_profile_budget_terms['heat_budget_residual'][row_idx]
            details.append({
                'daily_memory_loss': loss.detach(),
                'daily_memory_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_band_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_surface_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_upper_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_mid_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_deep_profile_loss': torch.zeros_like(loss.detach()),
                'daily_memory_temporal_smoothness_loss': no_profile_coeff_smoothness[row_idx].detach(),
                'daily_memory_heat_budget_loss': no_profile_heat_loss[row_idx].detach(),
                'daily_memory_heat_budget_residual_abs_mean_wm2': (
                    torch.abs(residual).detach() if heat_valid_bool else torch.zeros_like(residual)
                ),
                'daily_memory_heat_budget_residual_bias_mean_wm2': (
                    residual.detach() if heat_valid_bool else torch.zeros_like(residual)
                ),
                'daily_memory_heat_budget_step_count': torch.as_tensor(
                    1.0 if heat_valid_bool else 0.0,
                    device=device,
                    dtype=no_profile_prediction.dtype,
                ),
                'daily_memory_no_profile_lst_surface_loss': (
                    no_profile_lst_surface_loss[row_idx].detach()
                ),
                'daily_memory_no_profile_lst_surface_count': (
                    no_profile_lst_surface_mask[row_idx]
                    .to(dtype=no_profile_prediction.dtype)
                    .detach()
                ),
                'daily_memory_no_profile_lst_surface_weight_mean': (
                    no_profile_lst_surface_quality_weight[row_idx].detach()
                ),
                'daily_memory_physics_consistency_loss': no_profile_physics_consistency[row_idx].detach(),
                'daily_memory_physics_consistency_count': torch.as_tensor(
                    1.0 if bool(physics_valid.detach().cpu().item()) else 0.0,
                    device=device,
                    dtype=no_profile_prediction.dtype,
                ),
                'daily_memory_profile_physics_loss': no_profile_profile_physics[row_idx].detach(),
                'daily_memory_state_constraint_loss': torch.zeros_like(loss.detach()),
                'daily_memory_regularization_loss': no_profile_regularization[row_idx].detach(),
                'daily_memory_regularization_weighted_loss': (
                    no_profile_regularization_weighted[row_idx].detach()
                ),
                'daily_memory_coefficient_target_loss': torch.zeros_like(loss.detach()),
                'daily_memory_coefficient_target_clipped_fraction': torch.zeros_like(loss.detach()),
                'daily_memory_coefficient_supervision_count': torch.zeros_like(loss.detach()),
                'daily_memory_supervision_count': torch.zeros_like(loss.detach()),
                'daily_memory_valid_depth_count': torch.zeros_like(loss.detach()),
                'daily_memory_coefficient_abs_mean': (
                    no_profile_encoded['coefficient_abs_mean'][row_idx].detach()
                ),
                'daily_memory_coefficient_std': no_profile_encoded['coefficient_std'][row_idx].detach(),
                'daily_memory_coefficient_smoothness': no_profile_coeff_smoothness[row_idx].detach(),
                'daily_memory_component_count': no_profile_encoded['component_count'][row_idx].detach(),
                'daily_memory_conditioning_abs_mean': (
                    no_profile_encoded['conditioning_abs_mean'][row_idx].detach()
                ),
                'daily_memory_profile_fusion_gate_mean': (
                    no_profile_encoded['thermal_state_profile_fusion_gate'][row_idx].detach()
                ),
                'daily_memory_profile_fusion_active_count': (
                    (no_profile_encoded['thermal_state_profile_fusion_gate'][row_idx] > 0.0)
                    .to(dtype=no_profile_prediction.dtype)
                    .detach()
                ),
                'daily_memory_profile_fusion_age_days_mean': (
                    no_profile_encoded['thermal_state_profile_fusion_age_days'][row_idx].detach()
                ),
                'daily_memory_profile_fusion_coverage_fraction_mean': (
                    no_profile_encoded['thermal_state_profile_fusion_coverage_fraction'][row_idx].detach()
                ),
                'daily_memory_profile_fusion_depth_span_fraction_mean': (
                    no_profile_encoded['thermal_state_profile_fusion_depth_span_fraction'][row_idx].detach()
                ),
                'daily_memory_profile_fusion_future_fraction_mean': (
                    no_profile_encoded['thermal_state_profile_fusion_future_fraction'][row_idx].detach()
                ),
            })
            details[-1].update(_daily_memory_coefficient_detail(
                no_profile_encoded,
                row_idx,
                next_encoded=no_profile_next_encoded,
            ))
    return losses, details


def _daily_memory_config_fields(
    *,
    reconstruction_weight=DEFAULT_DAILY_MEMORY_RECONSTRUCTION_WEIGHT,
    samples_per_lake=DEFAULT_DAILY_MEMORY_SAMPLES_PER_LAKE,
    temporal_smoothness_weight=DEFAULT_DAILY_MEMORY_TEMPORAL_SMOOTHNESS_WEIGHT,
    heat_budget_weight=DEFAULT_DAILY_MEMORY_HEAT_BUDGET_WEIGHT,
    physics_consistency_weight=DEFAULT_DAILY_MEMORY_PHYSICS_CONSISTENCY_WEIGHT,
    regularization_weight=DEFAULT_DAILY_MEMORY_REGULARIZATION_WEIGHT,
    coefficient_loss_weight=DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_WEIGHT,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    start_epoch=DEFAULT_DAILY_MEMORY_START_EPOCH,
    ramp_epochs=DEFAULT_DAILY_MEMORY_RAMP_EPOCHS,
    hidden_dim=DEFAULT_DAILY_MEMORY_HIDDEN_DIM,
    init_spread=DEFAULT_DAILY_MEMORY_INIT_SPREAD,
    coeff_limit_sigma=DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA,
    prediction_branch='physics_rollout',
):
    return {
        'daily_memory_reconstruction_weight': float(reconstruction_weight),
        'daily_memory_samples_per_lake': int(samples_per_lake),
        'daily_memory_temporal_smoothness_weight': float(temporal_smoothness_weight),
        'daily_memory_heat_budget_weight': float(heat_budget_weight),
        'daily_memory_physics_consistency_weight': float(physics_consistency_weight),
        'daily_memory_regularization_weight': float(regularization_weight),
        'daily_memory_coefficient_loss_weight': float(coefficient_loss_weight),
        'no_profile_lst_surface_weight': float(no_profile_lst_surface_weight),
        'daily_memory_start_epoch': int(start_epoch),
        'daily_memory_ramp_epochs': int(ramp_epochs),
        'daily_memory_hidden_dim': int(hidden_dim),
        'daily_memory_init_spread': float(init_spread),
        'daily_memory_coeff_limit_sigma': float(coeff_limit_sigma),
        'daily_memory_conditioning_feature_dim': int(ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM),
        'daily_memory_conditioning_feature_names': ','.join(ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES),
        'daily_memory_forcing_encoder': 'separate_from_physics',
        'prediction_branch': _normalize_prediction_branch(prediction_branch),
    }


def _thermal_state_profile_fusion_config_fields(
    *,
    mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
):
    return {
        'thermal_state_profile_fusion_mode': _normalize_thermal_state_profile_fusion_mode(mode),
        'thermal_state_profile_fusion_time_policy': (
            _normalize_thermal_state_profile_fusion_time_policy(time_policy)
        ),
        'thermal_state_profile_fusion_lookup_split': str(lookup_split),
        'thermal_state_profile_fusion_max_age_days': int(max_age_days),
        'thermal_state_profile_fusion_min_depth_fraction': float(min_depth_fraction),
        'thermal_state_profile_fusion_max_weight': float(max_weight),
        'thermal_state_profile_fusion_coeff_limit_sigma': float(coeff_limit_sigma),
        'thermal_state_profile_fusion_profile_role': (
            'optional_low_dimensional_state_correction'
        ),
    }


def _daily_memory_history_fields(detail_records, *, weight_eff=0.0, **config):
    fields = _daily_memory_config_fields(**config)
    fields['daily_memory_weight_eff'] = float(weight_eff)
    for key in (
        'daily_memory_loss',
        'daily_memory_profile_loss',
        'daily_memory_band_profile_loss',
        'daily_memory_surface_profile_loss',
        'daily_memory_upper_profile_loss',
        'daily_memory_mid_profile_loss',
        'daily_memory_deep_profile_loss',
        'daily_memory_temporal_smoothness_loss',
        'daily_memory_heat_budget_loss',
        'daily_memory_heat_budget_residual_abs_mean_wm2',
        'daily_memory_heat_budget_residual_bias_mean_wm2',
        'daily_memory_no_profile_lst_surface_loss',
        'daily_memory_no_profile_lst_surface_weight_mean',
        'daily_memory_physics_consistency_loss',
        'daily_memory_profile_physics_loss',
        'daily_memory_state_constraint_loss',
        'daily_memory_regularization_loss',
        'daily_memory_regularization_weighted_loss',
        'daily_memory_coefficient_target_loss',
        'daily_memory_coefficient_target_clipped_fraction',
        'daily_memory_valid_depth_count',
        'daily_memory_coefficient_abs_mean',
        'daily_memory_coefficient_std',
        'daily_memory_coefficient_smoothness',
        'daily_memory_component_count',
        'daily_memory_conditioning_abs_mean',
        'daily_memory_profile_fusion_gate_mean',
        'daily_memory_profile_fusion_age_days_mean',
        'daily_memory_profile_fusion_coverage_fraction_mean',
        'daily_memory_profile_fusion_depth_span_fraction_mean',
        'daily_memory_profile_fusion_future_fraction_mean',
    ):
        fields[key] = _mean_detail(detail_records, key)
    for component_idx in range(DAILY_MEMORY_HISTORY_COEFFICIENT_COMPONENTS):
        suffix = f'{component_idx + 1:02d}'
        for tail in ('mean', 'abs_mean', 'unit_abs_mean', 'next_delta_abs_mean'):
            key = f'daily_memory_coeff_{suffix}_{tail}'
            fields[key] = _mean_detail(detail_records, key)
        for tail in ('target_mean', 'target_error_abs_mean'):
            key = f'daily_memory_coeff_{suffix}_{tail}'
            fields[key] = _mean_detail(detail_records, key)
        for tail in ('target_unit_mean', 'target_unit_error_abs_mean'):
            key = f'daily_memory_coeff_{suffix}_{tail}'
            fields[key] = _mean_detail(detail_records, key)
    fields['daily_memory_supervision_count'] = _sum_detail(
        detail_records,
        'daily_memory_supervision_count',
    )
    fields['daily_memory_heat_budget_step_count'] = _sum_detail(
        detail_records,
        'daily_memory_heat_budget_step_count',
    )
    fields['daily_memory_no_profile_lst_surface_count'] = _sum_detail(
        detail_records,
        'daily_memory_no_profile_lst_surface_count',
    )
    fields['daily_memory_physics_consistency_count'] = _sum_detail(
        detail_records,
        'daily_memory_physics_consistency_count',
    )
    fields['daily_memory_coefficient_supervision_count'] = _sum_detail(
        detail_records,
        'daily_memory_coefficient_supervision_count',
    )
    fields['daily_memory_profile_fusion_active_count'] = _sum_detail(
        detail_records,
        'daily_memory_profile_fusion_active_count',
    )
    return fields


def _zero_profile_init_net_training_records(
    model,
    lake,
    *,
    split_key='train',
    epoch=0,
    samples_per_lake=0,
    profile_huber_delta=2.0,
    regularization_weight=DEFAULT_ZERO_PROFILE_INIT_NET_REGULARIZATION_WEIGHT,
    training_spinup_days=DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS,
    physics_weight=DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT,
    rollout_weight=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT,
    rollout_max_days=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
    rollout_targets=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
    profile_sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
    profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    profile_fusion_lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    collect_details=True,
):
    device = lake['depths'].device
    area_profile = lake.get('area', lake.get('area_profile'))
    if area_profile is None:
        area_profile = torch.ones_like(lake['depths'])
    basis = getattr(model, 'zero_profile_thermal_basis', None)
    if basis is None:
        return [], [_zero_profile_init_net_empty_detail(device)] if collect_details else []
    basis_tensors = zero_profile_thermal_basis_tensors_for_depths(
        basis,
        lake['depths_np'],
        device=lake['depths'].device,
        dtype=lake['depths'].dtype,
    )
    if basis_tensors is None:
        return [], [_zero_profile_init_net_empty_detail(device)] if collect_details else []

    training_spinup_days = max(0, int(training_spinup_days))
    physics_weight = max(0.0, float(physics_weight))
    rollout_weight = max(0.0, float(rollout_weight))
    rollout_max_days = max(0, int(rollout_max_days))
    rollout_targets = max(0, int(rollout_targets))
    losses = []
    details = []
    dates = _select_zero_profile_init_net_dates(
        lake,
        split_key,
        samples_per_lake,
        epoch,
        sampling_mode=profile_sampling_mode,
    )
    grouped_entries = {}
    for date_value in dates:
        # Only dates explicitly selected from the train lookup are supervised.
        if date_value not in lake['lookups'].get(split_key, {}):
            continue
        target_idx = int(lake['date_to_index'][date_value])
        start_idx = max(0, target_idx - training_spinup_days)
        spinup_days_used = max(0, int(target_idx - start_idx))
        if spinup_days_used > 0 and not _forcing_rows_have_step_features(lake, start_idx, target_idx):
            start_idx = target_idx
            spinup_days_used = 0
        base_profile, _base_info = build_zero_profile_eof_pca_low_dof_prior(
            lake['df'],
            lake['depths_np'],
            lake['metadata'],
            start_idx,
            thermal_basis=basis,
        )
        base_tensor = torch.as_tensor(
            base_profile,
            dtype=lake['depths'].dtype,
            device=lake['depths'].device,
        ).reshape(1, -1)
        grouped_entries.setdefault(spinup_days_used, []).append({
            'date': date_value,
            'start_idx': int(start_idx),
            'target_idx': int(target_idx),
            'base_tensor': base_tensor,
        })

    for spinup_days_used in sorted(grouped_entries):
        entries = grouped_entries[spinup_days_used]
        if not entries:
            continue
        start_index_tensor = torch.as_tensor(
            [entry['start_idx'] for entry in entries],
            dtype=torch.long,
            device=lake['depths'].device,
        )
        base_batch = torch.cat([entry['base_tensor'] for entry in entries], dim=0)
        target_batch, target_mask_batch = _target_tensor_and_mask_batch(
            lake,
            split_key,
            [entry['date'] for entry in entries],
        )
        target_batch = target_batch.to(device=lake['depths'].device, dtype=lake['depths'].dtype)
        target_mask_batch = target_mask_batch.to(device=lake['depths'].device, dtype=torch.bool)
        encoded = model.zero_profile_initial_state_from_basis(
            base_batch,
            _forcing_row_batch(lake, start_index_tensor)['history_features'],
            lake['static_features'],
            basis_tensors['components_on_depth'],
            basis_tensors['coeff_std'],
            conditioning_features=_zero_profile_init_conditioning_batch(
                lake,
                start_index_tensor,
                device=lake['depths'].device,
                dtype=lake['depths'].dtype,
            ),
        )
        init_fusion = _thermal_state_profile_fusion_batch(
            lake,
            start_index_tensor,
            basis_tensors,
            mode=profile_fusion_mode,
            branch='init',
            lookup_split=profile_fusion_lookup_split,
            time_policy=profile_fusion_time_policy,
            max_age_days=profile_fusion_max_age_days,
            min_depth_fraction=profile_fusion_min_depth_fraction,
            max_weight=profile_fusion_max_weight,
            coeff_limit_sigma=profile_fusion_coeff_limit_sigma,
        )
        encoded = _apply_profile_fusion_to_init_encoded(encoded, init_fusion)
        initial_profile = encoded['initial_profile_c']
        spinup_profile = initial_profile
        spinup_freezing_storage = torch.zeros_like(spinup_profile)
        if spinup_days_used > 0:
            spinup_forcing_rows, spinup_next_forcing_rows = _segment_rollout_forcing_sequence(
                lake,
                start_index_tensor,
                spinup_days_used,
            )
            for offset in range(spinup_days_used):
                spinup_profile, spinup_freezing_storage, _diagnostics = model.step(
                    spinup_profile,
                    spinup_forcing_rows[offset],
                    lake['static_features'],
                    next_forcing_row=spinup_next_forcing_rows[offset],
                    task_mode=task_mode,
                    depths=lake['depths'],
                    area_profile=area_profile,
                    return_diagnostics=True,
                    diagnostic_mode=step_diagnostic_mode,
                    hard_density_stability=hard_density_stability,
                    freezing_storage_j_m2=spinup_freezing_storage,
                    return_freezing_storage=True,
                )
        direct_profile_losses = _masked_rmse_profile_loss_per_sample(
            initial_profile,
            target_batch,
            target_mask_batch,
        )
        direct_band_profile_losses, direct_band_losses, _direct_band_counts = (
            _zero_profile_init_net_depth_band_losses(
                initial_profile,
                target_batch,
                target_mask_batch,
                lake['depths'],
            )
        )
        spinup_profile_losses = _masked_rmse_profile_loss_per_sample(
            spinup_profile,
            target_batch,
            target_mask_batch,
        )
        spinup_band_profile_losses, spinup_band_losses, _spinup_band_counts = (
            _zero_profile_init_net_depth_band_losses(
                spinup_profile,
                target_batch,
                target_mask_batch,
                lake['depths'],
            )
        )
        full_profile_losses = spinup_profile_losses if spinup_days_used > 0 else direct_profile_losses
        band_profile_losses = (
            spinup_band_profile_losses if spinup_days_used > 0 else direct_band_profile_losses
        )
        profile_losses = 0.75 * full_profile_losses + 0.25 * band_profile_losses
        regularization_losses = encoded['regularization_loss'].reshape(-1)
        if regularization_losses.numel() == 1 and len(entries) > 1:
            regularization_losses = regularization_losses.expand(len(entries))
        profile_physics_losses = _profile_physics_loss_per_sample(initial_profile)
        if spinup_days_used > 0:
            profile_physics_losses = 0.5 * (
                profile_physics_losses + _profile_physics_loss_per_sample(spinup_profile)
            )
        state_constraint_profile = spinup_profile if spinup_days_used > 0 else initial_profile
        state_constraint_losses, state_constraint_parts = _zero_profile_init_state_physics_loss_per_sample(
            state_constraint_profile,
            target_batch,
            target_mask_batch,
            lake,
        )
        physics_losses = profile_physics_losses + state_constraint_losses
        rollout_profile_loss_rows = []
        rollout_target_count_rows = []
        rollout_max_gap_rows = []
        rollout_enabled_rows = []
        zero_like_profile_loss = profile_losses.detach() * 0.0
        for row_idx, entry in enumerate(entries):
            target_dates = ()
            if rollout_weight > 0.0 and rollout_max_days > 0:
                target_dates = _zero_profile_init_net_rollout_target_dates(
                    lake,
                    split_key,
                    entry['date'],
                    entry['target_idx'],
                    max_days=rollout_max_days,
                    max_targets=rollout_targets,
                )
            if not target_dates:
                rollout_profile_loss_rows.append(zero_like_profile_loss[row_idx])
                rollout_target_count_rows.append(0.0)
                rollout_max_gap_rows.append(0.0)
                rollout_enabled_rows.append(0.0)
                continue
            current = spinup_profile[row_idx: row_idx + 1]
            freezing_storage = spinup_freezing_storage[row_idx: row_idx + 1]
            current_idx = int(entry['target_idx'])
            target_losses = []
            gap_days_values = []
            for target_date in target_dates:
                target_idx = int(lake['date_to_index'][target_date])
                step_count = int(target_idx - current_idx)
                if step_count <= 0:
                    continue
                step_start = torch.as_tensor(
                    [current_idx],
                    dtype=torch.long,
                    device=lake['depths'].device,
                )
                forcing_rows, next_forcing_rows = _segment_rollout_forcing_sequence(
                    lake,
                    step_start,
                    step_count,
                )
                for offset in range(step_count):
                    current, freezing_storage, _diagnostics = model.step(
                        current,
                        forcing_rows[offset],
                        lake['static_features'],
                        next_forcing_row=next_forcing_rows[offset],
                        task_mode=task_mode,
                        depths=lake['depths'],
                        area_profile=area_profile,
                        return_diagnostics=True,
                        diagnostic_mode=step_diagnostic_mode,
                        hard_density_stability=hard_density_stability,
                        freezing_storage_j_m2=freezing_storage,
                        return_freezing_storage=True,
                    )
                current_idx = target_idx
                rollout_target, rollout_mask = _target_tensor_and_mask_batch(
                    lake,
                    split_key,
                    [target_date],
                )
                rollout_target = rollout_target.to(
                    device=lake['depths'].device,
                    dtype=lake['depths'].dtype,
                )
                rollout_mask = rollout_mask.to(device=lake['depths'].device, dtype=torch.bool)
                rollout_full_loss = _masked_rmse_profile_loss_per_sample(
                    current,
                    rollout_target,
                    rollout_mask,
                )
                rollout_band_loss, _rollout_band_losses, _rollout_band_counts = (
                    _zero_profile_init_net_depth_band_losses(
                        current,
                        rollout_target,
                        rollout_mask,
                        lake['depths'],
                    )
                )
                target_losses.append((0.75 * rollout_full_loss + 0.25 * rollout_band_loss).reshape(()))
                gap_days_values.append(float(target_idx - int(entry['target_idx'])))
            if target_losses:
                rollout_profile_loss_rows.append(torch.stack(target_losses).mean())
                rollout_target_count_rows.append(float(len(target_losses)))
                rollout_max_gap_rows.append(float(max(gap_days_values)) if gap_days_values else 0.0)
                rollout_enabled_rows.append(1.0)
            else:
                rollout_profile_loss_rows.append(zero_like_profile_loss[row_idx])
                rollout_target_count_rows.append(0.0)
                rollout_max_gap_rows.append(0.0)
                rollout_enabled_rows.append(0.0)
        rollout_profile_losses = torch.stack(rollout_profile_loss_rows)
        rollout_weighted_losses = rollout_weight * rollout_profile_losses
        rollout_target_counts = torch.as_tensor(
            rollout_target_count_rows,
            dtype=lake['depths'].dtype,
            device=lake['depths'].device,
        )
        rollout_max_gaps = torch.as_tensor(
            rollout_max_gap_rows,
            dtype=lake['depths'].dtype,
            device=lake['depths'].device,
        )
        rollout_enabled = torch.as_tensor(
            rollout_enabled_rows,
            dtype=lake['depths'].dtype,
            device=lake['depths'].device,
        )
        total_losses = (
            profile_losses
            + float(regularization_weight) * regularization_losses
            + physics_weight * physics_losses
            + rollout_weighted_losses
        )
        valid_counts = target_mask_batch.to(dtype=lake['depths'].dtype).reshape(len(entries), -1).sum(dim=1)
        delta_abs = encoded['initial_delta_abs_mean_c'].reshape(-1)
        delta_surface = encoded['initial_delta_surface_c'].reshape(-1)
        delta_deep = encoded['initial_delta_deep_c'].reshape(-1)
        conditioning_abs = encoded.get(
            'conditioning_abs_mean',
            torch.zeros_like(delta_abs),
        ).reshape(-1)
        if conditioning_abs.numel() == 1 and len(entries) > 1:
            conditioning_abs = conditioning_abs.expand(len(entries))
        component_count = torch.as_tensor(
            float(basis_tensors['component_count']),
            device=lake['depths'].device,
            dtype=lake['depths'].dtype,
        )
        supervision_count = torch.as_tensor(1.0, device=lake['depths'].device, dtype=lake['depths'].dtype)
        spinup_enabled = torch.as_tensor(
            1.0 if spinup_days_used > 0 else 0.0,
            device=lake['depths'].device,
            dtype=lake['depths'].dtype,
        )
        spinup_days_tensor = torch.as_tensor(
            float(spinup_days_used),
            device=lake['depths'].device,
            dtype=lake['depths'].dtype,
        )
        for row_idx, loss in enumerate(total_losses.unbind(dim=0)):
            losses.append(loss)
            if not collect_details:
                continue
            details.append({
                'zero_profile_init_net_loss': loss.detach(),
                'zero_profile_init_net_profile_loss': profile_losses[row_idx].detach(),
                'zero_profile_init_net_direct_profile_loss': direct_profile_losses[row_idx].detach(),
                'zero_profile_init_net_spinup_profile_loss': spinup_profile_losses[row_idx].detach(),
                'zero_profile_init_net_band_profile_loss': band_profile_losses[row_idx].detach(),
                'zero_profile_init_net_surface_profile_loss': (
                    spinup_band_losses['surface'][row_idx]
                    if spinup_days_used > 0 else direct_band_losses['surface'][row_idx]
                ).detach(),
                'zero_profile_init_net_upper_profile_loss': (
                    spinup_band_losses['upper'][row_idx]
                    if spinup_days_used > 0 else direct_band_losses['upper'][row_idx]
                ).detach(),
                'zero_profile_init_net_mid_profile_loss': (
                    spinup_band_losses['mid'][row_idx]
                    if spinup_days_used > 0 else direct_band_losses['mid'][row_idx]
                ).detach(),
                'zero_profile_init_net_deep_profile_loss': (
                    spinup_band_losses['deep'][row_idx]
                    if spinup_days_used > 0 else direct_band_losses['deep'][row_idx]
                ).detach(),
                'zero_profile_init_net_regularization_loss': regularization_losses[row_idx].detach(),
                'zero_profile_init_net_physics_loss': physics_losses[row_idx].detach(),
                'zero_profile_init_net_profile_physics_loss': profile_physics_losses[row_idx].detach(),
                'zero_profile_init_net_heat_content_constraint_loss': (
                    state_constraint_parts['heat_content_constraint_loss'][row_idx].detach()
                ),
                'zero_profile_init_net_surface_bottom_constraint_loss': (
                    state_constraint_parts['surface_bottom_constraint_loss'][row_idx].detach()
                ),
                'zero_profile_init_net_bounded_state_loss': (
                    state_constraint_parts['bounded_state_loss'][row_idx].detach()
                ),
                'zero_profile_init_net_rollout_profile_loss': (
                    rollout_profile_losses[row_idx].detach()
                ),
                'zero_profile_init_net_rollout_weighted_loss': (
                    rollout_weighted_losses[row_idx].detach()
                ),
                'zero_profile_init_net_rollout_supervision_count': (
                    rollout_target_counts[row_idx].detach()
                ),
                'zero_profile_init_net_rollout_enabled_count': (
                    rollout_enabled[row_idx].detach()
                ),
                'zero_profile_init_net_rollout_target_count': (
                    rollout_target_counts[row_idx].detach()
                ),
                'zero_profile_init_net_rollout_max_gap_days': (
                    rollout_max_gaps[row_idx].detach()
                ),
                'zero_profile_init_net_supervision_count': supervision_count,
                'zero_profile_init_net_spinup_enabled_count': spinup_enabled,
                'zero_profile_init_net_spinup_days_used': spinup_days_tensor,
                'zero_profile_init_net_valid_depth_count': valid_counts[row_idx].detach(),
                'zero_profile_init_net_delta_abs_mean_c': delta_abs[row_idx].detach(),
                'zero_profile_init_net_surface_delta_c': delta_surface[row_idx].detach(),
                'zero_profile_init_net_deep_delta_c': delta_deep[row_idx].detach(),
                'zero_profile_init_net_component_count': component_count,
                'zero_profile_init_net_conditioning_abs_mean': conditioning_abs[row_idx].detach(),
                'zero_profile_init_net_profile_fusion_gate_mean': (
                    encoded['thermal_state_profile_fusion_gate'][row_idx].detach()
                ),
                'zero_profile_init_net_profile_fusion_active_count': (
                    (encoded['thermal_state_profile_fusion_gate'][row_idx] > 0.0)
                    .to(dtype=lake['depths'].dtype)
                    .detach()
                ),
                'zero_profile_init_net_profile_fusion_delta_abs_mean_c': (
                    encoded['thermal_state_profile_fusion_delta_abs_mean_c'][row_idx].detach()
                ),
                'zero_profile_init_net_profile_fusion_age_days_mean': (
                    encoded['thermal_state_profile_fusion_age_days'][row_idx].detach()
                ),
                'zero_profile_init_net_profile_fusion_coverage_fraction_mean': (
                    encoded['thermal_state_profile_fusion_coverage_fraction'][row_idx].detach()
                ),
                'zero_profile_init_net_profile_fusion_depth_span_fraction_mean': (
                    encoded['thermal_state_profile_fusion_depth_span_fraction'][row_idx].detach()
                ),
                'zero_profile_init_net_profile_fusion_future_fraction_mean': (
                    encoded['thermal_state_profile_fusion_future_fraction'][row_idx].detach()
                ),
            })
    if collect_details and not losses:
        details.append(_zero_profile_init_net_empty_detail(device))
    return losses, details


def _zero_profile_init_net_history_fields(
    detail_records,
    *,
    loss_weight=0.0,
    weight_eff=0.0,
    start_epoch=0,
    ramp_epochs=0,
    samples_per_lake=0,
    regularization_weight=DEFAULT_ZERO_PROFILE_INIT_NET_REGULARIZATION_WEIGHT,
    training_spinup_days=DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS,
    physics_weight=DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT,
    rollout_weight=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT,
    rollout_max_days=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
    rollout_targets=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
    hidden_dim=DEFAULT_ZERO_PROFILE_INIT_NET_HIDDEN_DIM,
    init_spread=DEFAULT_ZERO_PROFILE_INIT_NET_INIT_SPREAD,
    coeff_limit_sigma=DEFAULT_ZERO_PROFILE_INIT_NET_COEFF_LIMIT_SIGMA,
    delta_limit_c=DEFAULT_ZERO_PROFILE_INIT_NET_DELTA_LIMIT_C,
):
    return {
        'zero_profile_init_net_loss_weight': float(loss_weight),
        'zero_profile_init_net_weight_eff': float(weight_eff),
        'zero_profile_init_net_profile_loss_mode': 'rmse',
        'zero_profile_init_net_start_epoch': int(start_epoch),
        'zero_profile_init_net_ramp_epochs': int(ramp_epochs),
        'zero_profile_init_net_samples_per_lake': int(samples_per_lake),
        'zero_profile_init_net_regularization_weight': float(regularization_weight),
        'zero_profile_init_net_training_spinup_days': int(training_spinup_days),
        'zero_profile_init_net_physics_weight': float(physics_weight),
        'zero_profile_init_net_rollout_weight': float(rollout_weight),
        'zero_profile_init_net_rollout_max_days': int(rollout_max_days),
        'zero_profile_init_net_rollout_targets': int(rollout_targets),
        'zero_profile_init_net_hidden_dim': int(hidden_dim),
        'zero_profile_init_net_init_spread': float(init_spread),
        'zero_profile_init_net_coeff_limit_sigma': float(coeff_limit_sigma),
        'zero_profile_init_net_delta_limit_c': float(delta_limit_c),
        'zero_profile_init_net_conditioning_feature_dim': int(
            ZERO_PROFILE_INIT_CONDITIONING_FEATURE_DIM
        ),
        'zero_profile_init_net_conditioning_feature_names': ','.join(
            ZERO_PROFILE_INIT_CONDITIONING_FEATURE_NAMES
        ),
        'zero_profile_init_net_loss': _mean_detail(detail_records, 'zero_profile_init_net_loss'),
        'zero_profile_init_net_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_loss',
        ),
        'zero_profile_init_net_direct_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_direct_profile_loss',
        ),
        'zero_profile_init_net_spinup_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_spinup_profile_loss',
        ),
        'zero_profile_init_net_band_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_band_profile_loss',
        ),
        'zero_profile_init_net_surface_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_surface_profile_loss',
        ),
        'zero_profile_init_net_upper_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_upper_profile_loss',
        ),
        'zero_profile_init_net_mid_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_mid_profile_loss',
        ),
        'zero_profile_init_net_deep_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_deep_profile_loss',
        ),
        'zero_profile_init_net_regularization_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_regularization_loss',
        ),
        'zero_profile_init_net_physics_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_physics_loss',
        ),
        'zero_profile_init_net_profile_physics_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_physics_loss',
        ),
        'zero_profile_init_net_heat_content_constraint_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_heat_content_constraint_loss',
        ),
        'zero_profile_init_net_surface_bottom_constraint_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_surface_bottom_constraint_loss',
        ),
        'zero_profile_init_net_bounded_state_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_bounded_state_loss',
        ),
        'zero_profile_init_net_rollout_profile_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_rollout_profile_loss',
        ),
        'zero_profile_init_net_rollout_weighted_loss': _mean_detail(
            detail_records,
            'zero_profile_init_net_rollout_weighted_loss',
        ),
        'zero_profile_init_net_rollout_supervision_count': _sum_detail(
            detail_records,
            'zero_profile_init_net_rollout_supervision_count',
        ),
        'zero_profile_init_net_rollout_enabled_count': _sum_detail(
            detail_records,
            'zero_profile_init_net_rollout_enabled_count',
        ),
        'zero_profile_init_net_rollout_target_count_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_rollout_target_count',
        ),
        'zero_profile_init_net_rollout_max_gap_days_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_rollout_max_gap_days',
        ),
        'zero_profile_init_net_supervision_count': _sum_detail(
            detail_records,
            'zero_profile_init_net_supervision_count',
        ),
        'zero_profile_init_net_spinup_enabled_count': _sum_detail(
            detail_records,
            'zero_profile_init_net_spinup_enabled_count',
        ),
        'zero_profile_init_net_spinup_days_used_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_spinup_days_used',
        ),
        'zero_profile_init_net_valid_depth_count_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_valid_depth_count',
        ),
        'zero_profile_init_net_delta_abs_mean_c': _mean_detail(
            detail_records,
            'zero_profile_init_net_delta_abs_mean_c',
        ),
        'zero_profile_init_net_surface_delta_c': _mean_detail(
            detail_records,
            'zero_profile_init_net_surface_delta_c',
        ),
        'zero_profile_init_net_deep_delta_c': _mean_detail(
            detail_records,
            'zero_profile_init_net_deep_delta_c',
        ),
        'zero_profile_init_net_component_count': _mean_detail(
            detail_records,
            'zero_profile_init_net_component_count',
        ),
        'zero_profile_init_net_conditioning_abs_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_conditioning_abs_mean',
        ),
        'zero_profile_init_net_profile_fusion_gate_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_gate_mean',
        ),
        'zero_profile_init_net_profile_fusion_active_count': _sum_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_active_count',
        ),
        'zero_profile_init_net_profile_fusion_delta_abs_mean_c': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_delta_abs_mean_c',
        ),
        'zero_profile_init_net_profile_fusion_age_days_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_age_days_mean',
        ),
        'zero_profile_init_net_profile_fusion_coverage_fraction_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_coverage_fraction_mean',
        ),
        'zero_profile_init_net_profile_fusion_depth_span_fraction_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_depth_span_fraction_mean',
        ),
        'zero_profile_init_net_profile_fusion_future_fraction_mean': _mean_detail(
            detail_records,
            'zero_profile_init_net_profile_fusion_future_fraction_mean',
        ),
    }


def _multitask_auxiliary_config_fields(
    *,
    weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    requested_weight=None,
    heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    hidden_dim=DEFAULT_MULTITASK_AUXILIARY_HIDDEN_DIM,
):
    requested = float(weight if requested_weight is None else requested_weight)
    return {
        'multitask_auxiliary_weight_requested': requested,
        'multitask_auxiliary_weight': float(weight),
        'multitask_auxiliary_heat_weight': float(heat_weight),
        'multitask_auxiliary_thermocline_weight': float(thermocline_weight),
        'multitask_auxiliary_mld_weight': float(mld_weight),
        'multitask_auxiliary_stability_weight': float(stability_weight),
        'multitask_auxiliary_surface_bottom_weight': float(surface_bottom_weight),
        'multitask_auxiliary_eof_weight': float(eof_weight),
        'multitask_auxiliary_hidden_dim': int(hidden_dim),
        'multitask_auxiliary_state_dim': int(len(MULTITASK_AUXILIARY_STATE_KEYS)),
        'multitask_auxiliary_state_keys': ','.join(MULTITASK_AUXILIARY_STATE_KEYS),
    }


def _multitask_auxiliary_history_fields(
    detail_records,
    *,
    weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    requested_weight=None,
    heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    hidden_dim=DEFAULT_MULTITASK_AUXILIARY_HIDDEN_DIM,
):
    fields = _multitask_auxiliary_config_fields(
        weight=weight,
        requested_weight=requested_weight,
        heat_weight=heat_weight,
        thermocline_weight=thermocline_weight,
        mld_weight=mld_weight,
        stability_weight=stability_weight,
        surface_bottom_weight=surface_bottom_weight,
        eof_weight=eof_weight,
        hidden_dim=hidden_dim,
    )
    fields.update({
        'multitask_auxiliary_loss': _mean_detail(detail_records, 'multitask_auxiliary_loss'),
        'multitask_auxiliary_weighted_loss': _mean_detail(
            detail_records,
            'multitask_auxiliary_weighted_loss',
        ),
        'multitask_auxiliary_supervision_count': _sum_detail(
            detail_records,
            'multitask_auxiliary_supervision_count',
        ),
    })
    for key in MULTITASK_AUXILIARY_STATE_KEYS:
        fields[f'multitask_auxiliary_{key}_loss'] = _mean_detail(
            detail_records,
            f'multitask_auxiliary_{key}_loss',
        )
        fields[f'multitask_auxiliary_{key}_enabled_fraction'] = _mean_detail(
            detail_records,
            f'multitask_auxiliary_{key}_enabled',
        )
    return fields


def _unlabeled_heat_closure_config_fields(
    *,
    weight=DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT,
    batch_size=DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE,
    window_days=DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS,
    horizons=DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS,
    tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2,
    open_water_only=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
    lst_qc_min=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
    reservoir_mode=DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE,
    mode=DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
    state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    state_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    solver_guard_weight=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
    solver_guard_tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2,
):
    parsed_horizons = _parse_unlabeled_heat_closure_horizons(
        horizons,
        fallback_window_days=window_days,
    )
    return {
        'unlabeled_heat_closure_weight': float(weight),
        'unlabeled_heat_closure_batch_size': int(batch_size),
        'unlabeled_heat_closure_window_days': int(window_days),
        'unlabeled_heat_closure_horizons': _format_unlabeled_heat_closure_horizons(parsed_horizons),
        'unlabeled_heat_closure_horizon_count': int(len(parsed_horizons)),
        'unlabeled_heat_closure_horizon_max_days': int(max(parsed_horizons)),
        'unlabeled_heat_closure_tau_wm2': float(tau_wm2),
        'unlabeled_heat_closure_open_water_only': str(open_water_only),
        'unlabeled_heat_closure_lst_qc_min': float(lst_qc_min),
        'unlabeled_heat_closure_reservoir_mode': str(reservoir_mode),
        'unlabeled_heat_closure_mode': _normalize_unlabeled_heat_closure_mode(mode),
        'unlabeled_heat_closure_state_source': _normalize_unlabeled_heat_closure_state_source(state_source),
        'unlabeled_heat_closure_spinup_days': int(state_spinup_days),
        'no_profile_lst_surface_weight': float(no_profile_lst_surface_weight),
        'unlabeled_heat_closure_solver_guard_weight': float(solver_guard_weight),
        'unlabeled_heat_closure_solver_guard_tau_wm2': float(solver_guard_tau_wm2),
    }


def _unlabeled_heat_closure_history_fields(
    detail_records,
    **config,
):
    fields = _unlabeled_heat_closure_config_fields(**config)
    fields.update({
        'unlabeled_heat_closure_loss': _mean_detail(detail_records, 'unlabeled_heat_closure_loss'),
        'unlabeled_heat_closure_weighted_loss': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_weighted_loss',
        ),
        'unlabeled_heat_closure_effective_weight_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_effective_weight',
        ),
        'unlabeled_heat_closure_window_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_window_count',
        ),
        'unlabeled_heat_closure_step_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_step_count',
        ),
        'unlabeled_heat_closure_active_loss_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_active_loss_count',
        ),
        'unlabeled_heat_closure_residual_abs_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_residual_abs_mean_wm2',
        ),
        'unlabeled_heat_closure_residual_bias_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_residual_bias_mean_wm2',
        ),
        'unlabeled_heat_closure_budget_residual_abs_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_budget_residual_abs_mean_wm2',
        ),
        'unlabeled_heat_closure_budget_residual_bias_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_budget_residual_bias_mean_wm2',
        ),
        'unlabeled_heat_closure_budget_storage_tendency_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_budget_storage_tendency_mean_wm2',
        ),
        'unlabeled_heat_closure_budget_target_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_budget_target_mean_wm2',
        ),
        'unlabeled_heat_closure_solver_residual_abs_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_solver_residual_abs_mean_wm2',
        ),
        'unlabeled_heat_closure_solver_residual_bias_mean_wm2': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_solver_residual_bias_mean_wm2',
        ),
        'unlabeled_heat_closure_lst_surface_loss': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_lst_surface_loss',
        ),
        'unlabeled_heat_closure_lst_surface_weighted_loss': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_lst_surface_weighted_loss',
        ),
        'unlabeled_heat_closure_lst_surface_effective_weight_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_lst_surface_effective_weight',
        ),
        'unlabeled_heat_closure_lst_surface_supervision_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_lst_surface_supervision_count',
        ),
        'unlabeled_heat_closure_lst_surface_weight_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_lst_surface_weight_mean',
        ),
        'unlabeled_heat_closure_solver_guard_loss': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_solver_guard_loss',
        ),
        'unlabeled_heat_closure_solver_guard_weighted_loss': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_solver_guard_weighted_loss',
        ),
        'unlabeled_heat_closure_solver_guard_effective_weight_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_solver_guard_effective_weight',
        ),
        'unlabeled_heat_closure_solver_guard_active_loss_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_solver_guard_active_loss_count',
        ),
        'unlabeled_heat_closure_state_source_code_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_state_source_code',
        ),
        'unlabeled_heat_closure_spinup_days_used_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_spinup_days_used',
        ),
        'unlabeled_heat_closure_horizon_days_mean': _mean_detail(
            detail_records,
            'unlabeled_heat_closure_horizon_days_mean',
        ),
        'unlabeled_heat_closure_horizon_days_min': _min_detail(
            detail_records,
            'unlabeled_heat_closure_horizon_days_min',
        ),
        'unlabeled_heat_closure_horizon_days_max': _max_detail(
            detail_records,
            'unlabeled_heat_closure_horizon_days_max',
        ),
        'unlabeled_heat_closure_horizon_count_observed': _max_detail(
            detail_records,
            'unlabeled_heat_closure_horizon_count',
        ),
        'unlabeled_heat_closure_profile_label_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_profile_label_count',
        ),
        'unlabeled_heat_closure_reservoir_diagnostic_only_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_reservoir_diagnostic_only_count',
        ),
        'unlabeled_heat_closure_reservoir_excluded_count': _sum_detail(
            detail_records,
            'unlabeled_heat_closure_reservoir_excluded_count',
        ),
    })
    return fields


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
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    residual_time_smooth_weight=DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
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
    collect_details=True,
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
            'segment_rollout_kd_saturation_penalty_loss': zero,
            'segment_rollout_kd_saturation_penalty_weighted_loss': zero,
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
    kd_saturation_penalty_losses = []
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
    lst_loss_sum, lst_weight_sum, lst_supervision_count = _init_masked_vector_stats(
        1,
        device=device,
        dtype=prediction.dtype,
    )
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
        kd_saturation_penalty_losses.append(_kd_saturation_penalty_loss(diagnostics, kd_saturation_threshold))
        adaptive_parameter_regularization_losses.append(_adaptive_parameter_regularization_loss(diagnostics))
        previous_scales = _current_physical_scales(diagnostics)
        residual_smooth_losses.append(_residual_profile_smoothness_loss(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if next_row is not None and float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            lst_loss_sum, lst_weight_sum, lst_supervision_count = _update_masked_vector_stats(
                lst_loss_sum,
                lst_weight_sum,
                lst_supervision_count,
                lst_loss_vec,
                lst_weight,
                lst_mask,
            )
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
                force_mask = (
                    torch.rand((prediction.shape[0], 1), device=device)
                    < float(teacher_forcing_probability)
                )
                prediction = torch.where(force_mask, target, prediction)
                freezing_storage = torch.where(
                    force_mask,
                    torch.zeros_like(freezing_storage),
                    freezing_storage,
                )
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
            'segment_rollout_kd_saturation_penalty_loss': zero,
            'segment_rollout_kd_saturation_penalty_weighted_loss': zero,
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
    kd_saturation_penalty_loss = (
        torch.stack(kd_saturation_penalty_losses).mean()
        if kd_saturation_penalty_losses else torch.tensor(0.0, device=device)
    )
    kd_saturation_penalty_weighted_loss = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_loss
    )
    adaptive_parameter_regularization_loss = (
        torch.stack(adaptive_parameter_regularization_losses).mean()
        if adaptive_parameter_regularization_losses else torch.tensor(0.0, device=device)
    )
    lst_loss = _masked_mean_from_sum(lst_loss_sum, lst_supervision_count).reshape(-1)[0]
    lst_weight_mean = _masked_mean_from_sum(lst_weight_sum, lst_supervision_count).reshape(-1)[0]
    lst_supervision_count_tensor = lst_supervision_count.reshape(-1)[0]
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
        + kd_saturation_penalty_weighted_loss
        + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_loss
        + heat_content_transition_weighted_loss
        + warm_column_weighted_loss
        + float(segment_rollout_lst_surface_weight) * lst_loss
    )
    if not collect_details:
        return total, len(profile_losses), {}
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
        'segment_rollout_kd_saturation_penalty_loss': kd_saturation_penalty_loss.detach(),
        'segment_rollout_kd_saturation_penalty_weighted_loss': kd_saturation_penalty_weighted_loss.detach(),
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
        'segment_rollout_kd_saturation_penalty_loss': zero,
        'segment_rollout_kd_saturation_penalty_weighted_loss': zero,
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
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    residual_time_smooth_weight=DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
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
    collect_details=True,
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
    start_index_tensor = torch.as_tensor(start_indices, dtype=torch.long, device=device)

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
    lst_loss_sum, lst_weight_sum, lst_supervision_counts = _init_masked_vector_stats(
        batch_size,
        device=device,
        dtype=prediction.dtype,
    )
    residual_smooth_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    kd_saturation_penalty_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_residual = None
    previous_scales = None
    last_gap = last_gaps[0]
    target_plans = _segment_rollout_target_plan_for_lake(
        lake,
        lookup_split,
        active_targets,
        start_indices,
        last_gap,
    )
    current_rows, next_rows = _segment_rollout_forcing_sequence(lake, start_index_tensor, last_gap)

    for offset in range(last_gap):
        previous = prediction
        step_input = torch.clamp(
            prediction + _state_noise_like(prediction, lake['depths'], state_noise_weight),
            0.0,
            40.0,
        )
        current_row = current_rows[offset]
        next_row = next_rows[offset]
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
        kd_saturation_penalty_vectors.append(_kd_saturation_penalty_loss_per_sample(
            diagnostics,
            kd_saturation_threshold,
        ))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)
        residual_smooth_vectors.append(_residual_profile_smoothness_loss_per_sample(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            lst_loss_sum, lst_weight_sum, lst_supervision_counts = _update_masked_vector_stats(
                lst_loss_sum,
                lst_weight_sum,
                lst_supervision_counts,
                lst_loss_vec,
                lst_weight,
                lst_mask,
            )

        target_plan = target_plans[offset]
        if target_plan is not None:
            active_indices = target_plan['active_indices']
            active_index_tensor = target_plan['active_index_tensor']
            active_dates = target_plan['active_dates']
            active_prediction_indices = target_plan['active_prediction_indices']
            target_gap_days_vec = target_plan['target_gap_days']
            target = target_plan['target']
            target_mask = target_plan['target_mask']
            active_prediction = prediction.index_select(0, active_index_tensor)
            profile_loss_vec = _masked_huber_profile_loss_per_sample(
                active_prediction,
                target,
                mask=target_mask,
                delta=profile_huber_delta,
            )
            for pos, sample_idx in enumerate(active_indices):
                target_gap_days = target_gap_days_vec[pos]
                horizon_weight = target_plan['horizon_weight'][pos].to(dtype=profile_loss_vec.dtype)
                profile_losses[sample_idx].append(horizon_weight * profile_loss_vec[pos])
                profile_horizon_weights[sample_idx].append(horizon_weight.detach())
                profile_target_gaps[sample_idx].append(
                    torch.as_tensor(float(target_gap_days), device=device, dtype=profile_loss_vec.dtype)
                )
            horizon_weight_vec = target_plan['horizon_weight'].to(dtype=profile_loss_vec.dtype)
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
    kd_saturation_penalty_vec = torch.stack(kd_saturation_penalty_vectors, dim=0).mean(dim=0)
    kd_saturation_penalty_weighted_vec = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_vec
    )
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
        lst_supervision_count = lst_supervision_counts[sample_idx].to(dtype=profile_loss.dtype)
        lst_loss = _masked_mean_from_sum(lst_loss_sum[sample_idx], lst_supervision_count)
        lst_weight_mean = _masked_mean_from_sum(lst_weight_sum[sample_idx], lst_supervision_count)
        total = (
            profile_loss
            + float(residual_time_smooth_weight) * residual_smooth_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + kd_saturation_penalty_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + warm_column_weighted_loss
            + float(segment_rollout_lst_surface_weight) * lst_loss
        )
        if not collect_details:
            results.append((total, len(profile_losses[sample_idx]), {}))
            continue
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
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    residual_time_smooth_weight=DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
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
    collect_details=True,
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
    lst_loss_sum, lst_weight_sum, lst_supervision_counts = _init_masked_vector_stats(
        batch_size,
        device=device,
        dtype=prediction.dtype,
    )
    residual_smooth_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    kd_saturation_penalty_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_residual = None
    previous_scales = None
    last_gap = last_gaps[0]
    target_plans = _segment_rollout_target_plan_for_items(
        items,
        lookup_split,
        active_targets,
        start_indices,
        last_gap,
    )
    current_rows, next_rows = _segment_rollout_forcing_sequence_for_items(
        items,
        start_indices,
        last_gap,
    )

    for offset in range(last_gap):
        previous = prediction
        step_input = torch.clamp(
            prediction + _state_noise_like(prediction, ref_lake['depths'], state_noise_weight),
            0.0,
            40.0,
        )
        current_row = current_rows[offset]
        next_row = next_rows[offset]
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
        kd_saturation_penalty_vectors.append(_kd_saturation_penalty_loss_per_sample(
            diagnostics,
            kd_saturation_threshold,
        ))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)
        residual_smooth_vectors.append(_residual_profile_smoothness_loss_per_sample(diagnostics, previous_residual))
        previous_residual = diagnostics.get('residual_profile_c')
        if float(segment_rollout_lst_surface_weight) > 0.0:
            lst_loss_vec, lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            lst_loss_sum, lst_weight_sum, lst_supervision_counts = _update_masked_vector_stats(
                lst_loss_sum,
                lst_weight_sum,
                lst_supervision_counts,
                lst_loss_vec,
                lst_weight,
                lst_mask,
            )

        target_plan = target_plans[offset]
        if target_plan is not None:
            active_positions = target_plan['active_positions']
            active_index_tensor = target_plan['active_index_tensor']
            active_dates = target_plan['active_dates']
            active_prediction_indices = target_plan['active_prediction_indices']
            target_gap_days_vec = target_plan['target_gap_days']
            target = target_plan['target']
            target_mask = target_plan['target_mask']
            active_prediction = prediction.index_select(0, active_index_tensor)
            profile_loss_vec = _masked_huber_profile_loss_per_sample(
                active_prediction,
                target,
                mask=target_mask,
                delta=profile_huber_delta,
            )
            for pos, sample_idx in enumerate(active_positions):
                target_gap_days = target_gap_days_vec[pos]
                horizon_weight = target_plan['horizon_weight'][pos].to(dtype=profile_loss_vec.dtype)
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
    kd_saturation_penalty_vec = torch.stack(kd_saturation_penalty_vectors, dim=0).mean(dim=0)
    kd_saturation_penalty_weighted_vec = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_vec
    )
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
        lst_supervision_count = lst_supervision_counts[sample_idx].to(dtype=profile_loss.dtype)
        lst_loss = _masked_mean_from_sum(lst_loss_sum[sample_idx], lst_supervision_count)
        lst_weight_mean = _masked_mean_from_sum(lst_weight_sum[sample_idx], lst_supervision_count)
        total = (
            profile_loss
            + float(residual_time_smooth_weight) * residual_smooth_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + kd_saturation_penalty_weighted_vec[sample_idx]
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
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    multitask_auxiliary_heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    multitask_auxiliary_thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    multitask_auxiliary_mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    multitask_auxiliary_stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    multitask_auxiliary_surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    multitask_auxiliary_eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
    collect_details=True,
):
    start, end, start_idx, end_idx = pair
    device = lake['depths'].device
    prediction, start_mask = _target_tensor_and_mask(lake, lookup_split, start)
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    target, target_mask = _target_tensor_and_mask(lake, lookup_split, end)

    lst_loss_sum, _lst_weight_sum, lst_supervision_count = _init_masked_vector_stats(
        int(prediction.shape[0]),
        device=device,
        dtype=prediction.dtype,
    )
    energy_losses = []
    daily_tendency_losses = []
    residual_regularization_losses = []
    physical_scale_regularization_losses = []
    physical_scale_smoothness_losses = []
    kd_prior_regularization_losses = []
    kd_saturation_penalty_losses = []
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
        kd_saturation_penalty_losses.append(_kd_saturation_penalty_loss(diagnostics, kd_saturation_threshold))
        adaptive_parameter_regularization_losses.append(_adaptive_parameter_regularization_loss(diagnostics))
        previous_scales = _current_physical_scales(diagnostics)
        if next_row is not None:
            lst_loss_vec, _lst_weight, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
            lst_loss_sum, _lst_weight_sum, lst_supervision_count = _update_masked_vector_stats(
                lst_loss_sum,
                _lst_weight_sum,
                lst_supervision_count,
                lst_loss_vec,
                lst_mask.to(dtype=lst_loss_vec.dtype),
                lst_mask,
            )
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
    kd_saturation_penalty_loss = (
        torch.stack(kd_saturation_penalty_losses).mean()
        if kd_saturation_penalty_losses else torch.tensor(0.0, device=device)
    )
    kd_saturation_penalty_weighted_loss = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_loss
    )
    adaptive_parameter_regularization_loss = (
        torch.stack(adaptive_parameter_regularization_losses).mean()
        if adaptive_parameter_regularization_losses else torch.tensor(0.0, device=device)
    )
    lst_loss = _masked_mean_from_sum(
        lst_loss_sum.sum(),
        lst_supervision_count.sum(),
    )
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
    final_forcing_row = lake['forcing_rows'][end_idx]
    multitask_auxiliary_loss_vec, multitask_auxiliary_details = _multitask_auxiliary_loss_vector(
        model,
        lake,
        prediction,
        target,
        target_mask,
        final_forcing_row,
        multitask_auxiliary_weight=multitask_auxiliary_weight,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        return_details=collect_details,
    )
    multitask_auxiliary_loss = multitask_auxiliary_loss_vec.mean()
    total = (
        profile_loss
        + physics_loss
        + float(daily_tendency_weight) * daily_tendency_loss
        + float(residual_regularization_weight) * residual_regularization_loss
        + float(physical_scale_regularization_weight) * physical_scale_regularization_loss
        + float(physical_scale_smoothness_weight) * physical_scale_smoothness_loss
        + kd_prior_regularization_weighted_loss
        + kd_saturation_penalty_weighted_loss
        + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_loss
        + heat_content_transition_weighted_loss
        + float(lst_surface_weight) * lst_loss
        + float(energy_balance_weight) * energy_loss
        + float(multitask_auxiliary_weight) * multitask_auxiliary_loss
    )
    if not collect_details:
        return total, {}
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
        'kd_saturation_penalty_loss': kd_saturation_penalty_loss.detach(),
        'kd_saturation_penalty_weighted_loss': kd_saturation_penalty_weighted_loss.detach(),
        'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_loss.detach(),
        **multitask_auxiliary_details[0],
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
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    multitask_auxiliary_heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    multitask_auxiliary_thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    multitask_auxiliary_mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    multitask_auxiliary_stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    multitask_auxiliary_surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    multitask_auxiliary_eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
    collect_details=True,
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
    start_index_tensor = torch.as_tensor(start_indices, dtype=torch.long, device=device)
    end_index_tensor = torch.as_tensor(end_indices, dtype=torch.long, device=device)

    prediction, start_mask = _target_tensor_and_mask_batch(lake, lookup_split, starts)
    freezing_storage = torch.zeros_like(prediction)
    start_profile = prediction
    target, target_mask = _target_tensor_and_mask_batch(lake, lookup_split, ends)

    lst_loss_sum, _lst_weight_sum, lst_supervision_counts = _init_masked_vector_stats(
        batch_size,
        device=device,
        dtype=prediction.dtype,
    )
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    energy_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    kd_saturation_penalty_vectors = []
    adaptive_parameter_regularization_vectors = []
    previous_scales = None
    final_diagnostics = None

    for offset in range(gaps[0]):
        previous = prediction
        day_indices = start_index_tensor + int(offset)
        next_indices = day_indices + 1
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
        kd_saturation_penalty_vectors.append(_kd_saturation_penalty_loss_per_sample(
            diagnostics,
            kd_saturation_threshold,
        ))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)

        lst_loss_vec, _, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
        lst_loss_sum, _lst_weight_sum, lst_supervision_counts = _update_masked_vector_stats(
            lst_loss_sum,
            _lst_weight_sum,
            lst_supervision_counts,
            lst_loss_vec,
            lst_mask.to(dtype=lst_loss_vec.dtype),
            lst_mask,
        )

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
    kd_saturation_penalty_vec = torch.stack(kd_saturation_penalty_vectors, dim=0).mean(dim=0)
    kd_saturation_penalty_weighted_vec = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_vec
    )
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
    final_forcing_row = _forcing_row_batch(lake, end_index_tensor)
    multitask_auxiliary_loss_vec, multitask_auxiliary_details = _multitask_auxiliary_loss_vector(
        model,
        lake,
        prediction,
        target,
        target_mask,
        final_forcing_row,
        multitask_auxiliary_weight=multitask_auxiliary_weight,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        return_details=collect_details,
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
        lst_loss = _masked_mean_from_sum(lst_loss_sum[sample_idx], lst_supervision_counts[sample_idx])
        total = (
            profile_loss_vec[sample_idx]
            + physics_loss_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + kd_saturation_penalty_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + float(lst_surface_weight) * lst_loss
            + float(energy_balance_weight) * energy_loss_vec[sample_idx]
            + float(multitask_auxiliary_weight) * multitask_auxiliary_loss_vec[sample_idx]
        )
        losses.append(total)
        if not collect_details:
            continue
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
            'kd_saturation_penalty_loss': kd_saturation_penalty_vec[sample_idx].detach(),
            'kd_saturation_penalty_weighted_loss': kd_saturation_penalty_weighted_vec[sample_idx].detach(),
            'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
            **multitask_auxiliary_details[sample_idx],
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


def _static_features_2d_for_lake(lake):
    store = lake.get('packed_tensor_store')
    if store is not None:
        return store.static_features_2d
    return lake.get('static_features_2d', lake['static_features'].reshape(1, -1))


def _stack_static_features_for_items(items):
    if not items:
        raise ValueError('static feature items must not be empty.')
    first_row = _static_features_2d_for_lake(items[0][1])
    if all(_static_features_2d_for_lake(item[1]) is first_row for item in items):
        return first_row.expand(len(items), -1)
    rows = [_static_features_2d_for_lake(item[1]) for item in items]
    return torch.cat(rows, dim=0)


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
    tensor_batch = _stack_forcing_tensor_batches_for_items(items, day_indices)
    if tensor_batch is not None:
        return tensor_batch
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
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    multitask_auxiliary_heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    multitask_auxiliary_thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    multitask_auxiliary_mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    multitask_auxiliary_stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    multitask_auxiliary_surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    multitask_auxiliary_eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
    task_mode='analysis',
    hard_density_stability=False,
    step_diagnostic_mode='loss',
    lookup_split='train',
    collect_details=True,
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

    lst_loss_sum, _lst_weight_sum, lst_supervision_counts = _init_masked_vector_stats(
        batch_size,
        device=device,
        dtype=prediction.dtype,
    )
    heat_content_losses = [[] for _ in range(batch_size)]
    heat_content_weighted_losses = [[] for _ in range(batch_size)]
    heat_content_effective_weights = [[] for _ in range(batch_size)]
    energy_vectors = []
    daily_tendency_vectors = []
    residual_regularization_vectors = []
    physical_scale_regularization_vectors = []
    physical_scale_smoothness_vectors = []
    kd_prior_regularization_vectors = []
    kd_saturation_penalty_vectors = []
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
        kd_saturation_penalty_vectors.append(_kd_saturation_penalty_loss_per_sample(
            diagnostics,
            kd_saturation_threshold,
        ))
        adaptive_parameter_regularization_vectors.append(
            _adaptive_parameter_regularization_loss_per_sample(diagnostics)
        )
        previous_scales = _current_physical_scales_detached(diagnostics)

        lst_loss_vec, _, lst_mask = _segment_open_water_lst_loss_per_sample(prediction, next_row)
        lst_loss_sum, _lst_weight_sum, lst_supervision_counts = _update_masked_vector_stats(
            lst_loss_sum,
            _lst_weight_sum,
            lst_supervision_counts,
            lst_loss_vec,
            lst_mask.to(dtype=lst_loss_vec.dtype),
            lst_mask,
        )

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
    kd_saturation_penalty_vec = torch.stack(kd_saturation_penalty_vectors, dim=0).mean(dim=0)
    kd_saturation_penalty_weighted_vec = (
        float(kd_saturation_penalty_weight) * kd_saturation_penalty_vec
    )
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
    final_forcing_row = _stack_forcing_batch_for_items(items, end_indices)
    multitask_auxiliary_loss_vec, multitask_auxiliary_details = _multitask_auxiliary_loss_vector(
        model,
        ref_lake,
        prediction,
        target,
        target_mask,
        final_forcing_row,
        static_features=static_features,
        multitask_auxiliary_weight=multitask_auxiliary_weight,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        return_details=collect_details,
    )

    results = []
    for sample_idx, item in enumerate(items):
        _, heat_content_weighted_loss, heat_content_details = _heat_content_transition_loss_details(
            heat_content_losses[sample_idx],
            heat_content_weighted_losses[sample_idx],
            heat_content_effective_weights[sample_idx],
            device=device,
        )
        lst_loss = _masked_mean_from_sum(lst_loss_sum[sample_idx], lst_supervision_counts[sample_idx])
        total = (
            profile_loss_vec[sample_idx]
            + physics_loss_vec[sample_idx]
            + float(daily_tendency_weight) * daily_tendency_vec[sample_idx]
            + float(residual_regularization_weight) * residual_regularization_vec[sample_idx]
            + float(physical_scale_regularization_weight) * physical_scale_regularization_vec[sample_idx]
            + float(physical_scale_smoothness_weight) * physical_scale_smoothness_vec[sample_idx]
            + kd_prior_regularization_weighted_vec[sample_idx]
            + kd_saturation_penalty_weighted_vec[sample_idx]
            + float(adaptive_parameter_regularization_weight) * adaptive_parameter_regularization_vec[sample_idx]
            + heat_content_weighted_loss
            + float(lst_surface_weight) * lst_loss
            + float(energy_balance_weight) * energy_loss_vec[sample_idx]
            + float(multitask_auxiliary_weight) * multitask_auxiliary_loss_vec[sample_idx]
        )
        if not collect_details:
            results.append((item[0], total, {}))
            continue
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
                'kd_saturation_penalty_loss': kd_saturation_penalty_vec[sample_idx].detach(),
                'kd_saturation_penalty_weighted_loss': kd_saturation_penalty_weighted_vec[sample_idx].detach(),
                'adaptive_parameter_regularization_loss': adaptive_parameter_regularization_vec[sample_idx].detach(),
                **multitask_auxiliary_details[sample_idx],
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
    cached_batches=None,
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

    if cached_batches is not None:
        batches = cached_batches
    else:
        batches = _build_cross_lake_transition_batches(
            lakes,
            pair_key=pair_key,
            transition_batch_size=transition_batch_size,
            cross_lake_batch_size=cross_lake_batch_size,
        )

    results = {lake_idx: ([], []) for lake_idx in range(len(lakes))}
    for chunk in batches:
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
def evaluate_lake_pair_depth_rmse(model, lake, pairs, *, task_mode='analysis', hard_density_stability=False):
    """Return transition-end RMSE split into <=25 m and >25 m depth bands."""
    errors_by_band = _empty_depth_squared_errors_by_band()
    if not pairs:
        return _depth_rmse_record_from_squared_errors(errors_by_band)
    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
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
        if not torch.any(valid):
            continue
        valid_np = valid.reshape(-1).detach().cpu().numpy().astype(bool)
        squared_errors = (
            (prediction.reshape(-1) - target.reshape(-1))
            .pow(2)
            .detach()
            .double()
            .cpu()
            .numpy()
        )
        _extend_depth_squared_errors(
            errors_by_band,
            squared_errors[valid_np],
            depths_np[valid_np],
        )
    return _depth_rmse_record_from_squared_errors(errors_by_band)


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
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    if not pairs:
        return _horizon_metric_record(errors_by_horizon, depth_errors_by_horizon=depth_errors_by_horizon)
    horizons = tuple(sorted(int(horizon) for horizon in horizons))
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
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
        valid_np = valid.reshape(-1).detach().cpu().numpy().astype(bool)
        squared_errors = (
            (prediction.reshape(-1) - target.reshape(-1))
            .pow(2)
            .detach()
            .double()
            .cpu()
            .numpy()
        )
        mse = torch.mean((prediction[valid] - target[valid]).pow(2)).detach().cpu().item()
        for horizon in horizons:
            if gap_days <= horizon:
                errors_by_horizon[horizon].append(mse)
                _extend_depth_horizon_errors(
                    depth_errors_by_horizon,
                    horizon,
                    squared_errors[valid_np],
                    depths_np[valid_np],
                )
    return _horizon_metric_record(errors_by_horizon, depth_errors_by_horizon=depth_errors_by_horizon)


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


def _aggregate_lswt_observer_details(
    details,
    *,
    rollout_lswt_observer_mode='off',
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
):
    numeric = {
        'lswt_observer_applied_count': [],
        'lswt_observer_quality_mean': [],
        'lswt_observer_open_water_weight_mean': [],
        'lswt_observer_surface_innovation_c': [],
        'lswt_observer_mean_abs_delta_c': [],
        'lswt_observer_max_abs_delta_c': [],
        'lswt_observer_heat_content_delta_c': [],
        'lswt_observer_deep_abs_delta_c': [],
        'lswt_observer_density_guard_scale': [],
        'lswt_observer_filled_lst_used_count': [],
        'lswt_observer_kalman_gain_surface': [],
        'lswt_observer_kalman_gain_mean': [],
        'lswt_observer_observation_error_c': [],
        'lswt_observer_state_variance_surface': [],
        'lswt_observer_localization_depth_m': [],
        'lswt_observer_reservoir_conservative_scale': [],
        'lswt_observer_heat_content_bound_scale': [],
        'lswt_observer_mld_depth_m': [],
        'lswt_observer_mld_weight_mean': [],
        'lswt_observer_mld_heat_content_delta_c': [],
        'lswt_observer_mld_volume_fraction': [],
        'lswt_observer_mld_surface_to_heat_gain': [],
    }
    for detail in details or ():
        for key in numeric:
            value = detail.get(key) if isinstance(detail, dict) else None
            if value is None:
                continue
            try:
                value = float(torch.as_tensor(value).detach().cpu().reshape(-1)[0])
            except (TypeError, ValueError, RuntimeError):
                continue
            if np.isfinite(value):
                numeric[key].append(value)
    def mean(key):
        values = numeric.get(key, ())
        return float(np.mean(values)) if values else 0.0
    def max_value(key):
        values = numeric.get(key, ())
        return float(np.max(values)) if values else 0.0
    return {
        'zero_profile_initializer': str(zero_profile_initializer),
        'spinup_lswt_observer_mode': str(spinup_lswt_observer_mode),
        'rollout_lswt_observer_mode': str(rollout_lswt_observer_mode),
        'lswt_observer_update_count': float(np.sum(numeric['lswt_observer_applied_count']))
        if numeric['lswt_observer_applied_count'] else 0.0,
        'lswt_observer_quality_mean': mean('lswt_observer_quality_mean'),
        'lswt_observer_open_water_weight_mean': mean('lswt_observer_open_water_weight_mean'),
        'lswt_observer_surface_innovation_mean_c': mean('lswt_observer_surface_innovation_c'),
        'lswt_observer_mean_abs_delta_c': mean('lswt_observer_mean_abs_delta_c'),
        'lswt_observer_max_abs_delta_c': max_value('lswt_observer_max_abs_delta_c'),
        'lswt_observer_heat_content_delta_mean_c': mean('lswt_observer_heat_content_delta_c'),
        'lswt_observer_deep_abs_delta_mean_c': mean('lswt_observer_deep_abs_delta_c'),
        'lswt_observer_density_guard_scale_mean': mean('lswt_observer_density_guard_scale'),
        'lswt_observer_filled_lst_used_count': float(np.sum(numeric['lswt_observer_filled_lst_used_count']))
        if numeric['lswt_observer_filled_lst_used_count'] else 0.0,
        'lswt_observer_kalman_gain_surface_mean': mean('lswt_observer_kalman_gain_surface'),
        'lswt_observer_kalman_gain_mean': mean('lswt_observer_kalman_gain_mean'),
        'lswt_observer_observation_error_mean_c': mean('lswt_observer_observation_error_c'),
        'lswt_observer_state_variance_surface_mean': mean('lswt_observer_state_variance_surface'),
        'lswt_observer_localization_depth_mean_m': mean('lswt_observer_localization_depth_m'),
        'lswt_observer_reservoir_conservative_scale_mean': mean('lswt_observer_reservoir_conservative_scale'),
        'lswt_observer_heat_content_bound_scale_mean': mean('lswt_observer_heat_content_bound_scale'),
        'lswt_observer_mld_depth_mean_m': mean('lswt_observer_mld_depth_m'),
        'lswt_observer_mld_weight_mean': mean('lswt_observer_mld_weight_mean'),
        'lswt_observer_mld_heat_content_delta_mean_c': mean('lswt_observer_mld_heat_content_delta_c'),
        'lswt_observer_mld_volume_fraction_mean': mean('lswt_observer_mld_volume_fraction'),
        'lswt_observer_mld_surface_to_heat_gain_mean': mean('lswt_observer_mld_surface_to_heat_gain'),
    }


@torch.no_grad()
def evaluate_lake_free_roll(
    model,
    lake,
    *,
    task_mode='analysis',
    horizons=(1, 3, 7, 14, 30, 60),
    init_mode='profile',
    spinup_days=90,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_lswt_observer_mode='off',
    lswt_observer_strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
    lswt_observer_decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
    lswt_observer_max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
    lswt_observer_low_rank_deep_update_fraction=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
    lswt_observer_heat_content_limit_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
    lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    rollout_start_date=None,
    hard_density_stability=False,
    diagnostic_output_dir=None,
    diagnostic_split_label='free_roll',
    diagnostic_epoch=None,
    diagnostic_rollout_mode='free',
):
    """Evaluate full free-roll against all available profile dates after initialization."""
    model.eval()
    zero_profile_initializer = normalize_zero_profile_initializer_mode(zero_profile_initializer)
    spinup_lswt_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    rollout_lswt_observer_mode = normalize_lswt_observer_mode(rollout_lswt_observer_mode)
    df = lake['df']
    all_lookup = lake['lookups']['all']
    if not all_lookup:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            **_empty_observed_point_metrics(),
            **_empty_free_roll_point_diagnostics_result(),
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
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=spinup_lswt_observer_mode,
        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
        lswt_observer_low_rank_deep_update_fraction=lswt_observer_low_rank_deep_update_fraction,
        lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
        lswt_observer_min_quality=lswt_observer_min_quality,
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
    rollout_observer_details = []
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
        observer_detail = None
        if rollout_lswt_observer_mode != 'off' and next_row is not None:
            current, observer_detail = apply_lswt_observer_update(
                current,
                next_row,
                lake['depths'],
                mode=rollout_lswt_observer_mode,
                strength=lswt_observer_strength,
                decay_depth_m=lswt_observer_decay_depth_m,
                max_increment_c=lswt_observer_max_increment_c,
                low_rank_deep_update_fraction=lswt_observer_low_rank_deep_update_fraction,
                heat_content_limit_c=lswt_observer_heat_content_limit_c,
                min_quality=lswt_observer_min_quality,
                area_profile=lake['area'],
                metadata=lake.get('metadata'),
            )
            rollout_observer_details.append(observer_detail)
        predictions_by_index[day_idx + 1] = current.detach().cpu().numpy().reshape(-1)

    errors = []
    depth_errors_by_band = _empty_depth_squared_errors_by_band()
    depth_biases_by_band = _empty_depth_errors_by_band()
    biases = []
    post_spinup_errors = []
    post_spinup_biases = []
    horizon_errors = {int(horizon): [] for horizon in horizons}
    horizon_biases = {int(horizon): [] for horizon in horizons}
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
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
        squared_errors = diff ** 2
        valid_depths = depths_np[valid]
        errors.extend(diff.tolist())
        _extend_depth_squared_errors(depth_errors_by_band, squared_errors, valid_depths)
        _extend_depth_errors(depth_biases_by_band, diff, valid_depths)
        biases.append(float(np.mean(diff)))
        post_spinup_errors.extend(diff.tolist())
        post_spinup_biases.append(float(np.mean(diff)))
        gap_days = int(obs_idx - rollout_start_idx)
        for horizon in horizon_errors:
            if gap_days <= horizon:
                horizon_errors[horizon].extend(squared_errors.tolist())
                _extend_depth_horizon_errors(
                    depth_errors_by_horizon,
                    horizon,
                    squared_errors,
                    valid_depths,
                )
                horizon_biases[horizon].append(float(np.mean(diff)))

    observed_point_metrics = _observed_point_metrics_from_predictions(lake, predictions_by_index)
    point_diagnostics = (
        _write_free_roll_point_diagnostics(
            lake,
            init_state,
            predictions_by_index,
            diagnostic_output_dir,
            split_label=diagnostic_split_label,
            epoch=diagnostic_epoch,
            horizons=horizons,
            diagnostic_rollout_mode=diagnostic_rollout_mode,
        )
        if diagnostic_output_dir is not None
        else _empty_free_roll_point_diagnostics_result()
    )
    observer_metric = _aggregate_lswt_observer_details(
        rollout_observer_details,
        rollout_lswt_observer_mode=rollout_lswt_observer_mode,
        spinup_lswt_observer_mode=spinup_lswt_observer_mode,
        zero_profile_initializer=init_state.get('zero_profile_initializer', zero_profile_initializer),
    )
    depth_metrics = _depth_rmse_record_from_squared_errors(depth_errors_by_band)
    depth_metrics.update(_depth_bias_record_from_errors(depth_biases_by_band))
    if not errors:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            **depth_metrics,
            'horizon_metrics': _horizon_metric_record(
                horizon_errors,
                horizon_biases,
                depth_errors_by_horizon=depth_errors_by_horizon,
            ),
            'post_spinup_rmse': np.nan,
            'post_spinup_bias': np.nan,
            'init_mode': init_state['init_mode'],
            'spinup_days_used': init_state['spinup_days_used'],
            **observer_metric,
            **observed_point_metrics,
            **point_diagnostics,
        }
    errors = np.asarray(errors, dtype=np.float64)
    post_spinup_errors = np.asarray(post_spinup_errors, dtype=np.float64)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
        'n_profiles': int(len(biases)),
        **depth_metrics,
        'horizon_metrics': _horizon_metric_record(
            horizon_errors,
            horizon_biases,
            depth_errors_by_horizon=depth_errors_by_horizon,
        ),
        'post_spinup_rmse': float(np.sqrt(np.mean(post_spinup_errors ** 2))) if post_spinup_errors.size else np.nan,
        'post_spinup_bias': float(np.mean(post_spinup_errors)) if post_spinup_errors.size else np.nan,
        'post_spinup_profiles': int(len(post_spinup_biases)),
        'init_mode': init_state['init_mode'],
        'spinup_days_used': init_state['spinup_days_used'],
        **observer_metric,
        **observed_point_metrics,
        **point_diagnostics,
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
    depth_errors_by_band = _empty_depth_squared_errors_by_band()
    depth_biases_by_band = _empty_depth_errors_by_band()
    biases = []
    post_spinup_errors = []
    post_spinup_biases = []
    horizon_errors = {int(horizon): [] for horizon in horizons}
    horizon_biases = {int(horizon): [] for horizon in horizons}
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
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
        squared_errors = diff ** 2
        valid_depths = depths_np[valid]
        errors.extend(diff.tolist())
        _extend_depth_squared_errors(depth_errors_by_band, squared_errors, valid_depths)
        _extend_depth_errors(depth_biases_by_band, diff, valid_depths)
        biases.append(float(np.mean(diff)))
        post_spinup_errors.extend(diff.tolist())
        post_spinup_biases.append(float(np.mean(diff)))
        gap_days = int(obs_idx - rollout_start_idx)
        for horizon in horizon_errors:
            if gap_days <= horizon:
                horizon_errors[horizon].extend(squared_errors.tolist())
                _extend_depth_horizon_errors(
                    depth_errors_by_horizon,
                    horizon,
                    squared_errors,
                    valid_depths,
                )
                horizon_biases[horizon].append(float(np.mean(diff)))

    depth_metrics = _depth_rmse_record_from_squared_errors(depth_errors_by_band)
    depth_metrics.update(_depth_bias_record_from_errors(depth_biases_by_band))
    if not errors:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'n_profiles': 0,
            **depth_metrics,
            'horizon_metrics': _horizon_metric_record(
                horizon_errors,
                horizon_biases,
                depth_errors_by_horizon=depth_errors_by_horizon,
            ),
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
        **depth_metrics,
        'horizon_metrics': _horizon_metric_record(
            horizon_errors,
            horizon_biases,
            depth_errors_by_horizon=depth_errors_by_horizon,
        ),
        'post_spinup_rmse': float(np.sqrt(np.mean(post_spinup_errors ** 2))) if post_spinup_errors.size else np.nan,
        'post_spinup_bias': float(np.mean(post_spinup_errors)) if post_spinup_errors.size else np.nan,
        'post_spinup_profiles': int(len(post_spinup_biases)),
        'init_mode': init_state['init_mode'],
        'spinup_days_used': init_state['spinup_days_used'],
    }


def _safe_artifact_token(value):
    text = str(value or 'diagnostic').strip()
    token = ''.join(
        char if char.isalnum() or char in {'_', '-'} else '_'
        for char in text
    ).strip('_')
    return token or 'diagnostic'


def _empty_free_roll_point_diagnostics_result():
    return {
        'free_roll_point_diagnostics_csv': None,
        'free_roll_age_summary_csv': None,
        'free_roll_point_diagnostics_count': 0,
        'free_roll_age_summary_count': 0,
    }


def _write_free_roll_point_diagnostics(
    lake,
    init_state,
    predictions_by_index,
    output_dir,
    *,
    split_label='free_roll',
    epoch=None,
    horizons=(1, 3, 7, 14, 30, 60),
    diagnostic_rollout_mode='free',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_to_index = _date_index_map(lake['df'])
    rollout_start_idx = int(init_state['rollout_start_idx'])
    rollout_start_date = pd.Timestamp(lake['df']['Date'].iloc[rollout_start_idx]).date().isoformat()

    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
    records = []
    for obs_date, target in lake['lookups']['all'].items():
        obs_timestamp = pd.Timestamp(obs_date).normalize()
        obs_idx = date_to_index.get(obs_timestamp)
        if obs_idx is None or obs_idx <= rollout_start_idx or obs_idx not in predictions_by_index:
            continue
        prediction = np.asarray(predictions_by_index[obs_idx], dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        mask = _lookup_mask(lake, 'all', obs_date)
        valid = np.isfinite(prediction) & np.isfinite(target_arr)
        if mask is not None:
            valid = valid & np.asarray(mask, dtype=bool)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        for grid_idx in valid_indices:
            error_value = float(prediction[grid_idx] - target_arr[grid_idx])
            depth_value = float(depths_np[grid_idx])
            records.append({
                'lake_id': lake['lake_id'],
                'lake_type': _lake_reservoir_bucket(lake),
                'eval_split': str(split_label),
                'rollout_mode': str(diagnostic_rollout_mode or 'free'),
                'Date': obs_timestamp.date().isoformat(),
                'rollout_start_date': rollout_start_date,
                'days_since_rollout_start': float(obs_idx - rollout_start_idx),
                'depth_m': depth_value,
                'depth_band': _depth_band_name(depth_value),
                'observed_temperature_c': float(target_arr[grid_idx]),
                'predicted_temperature_c': float(prediction[grid_idx]),
                'error_c': error_value,
                'abs_error_c': float(abs(error_value)),
                'squared_error_c': float(error_value ** 2),
            })
    if not records:
        return _empty_free_roll_point_diagnostics_result()

    split_token = _safe_artifact_token(split_label)
    lake_token = _safe_artifact_token(lake['lake_id'])
    epoch_token = f"epoch{int(epoch):04d}" if epoch is not None else 'epoch_na'
    point_path = output_dir / f"{split_token}_{lake_token}_{epoch_token}_free_roll_points.csv"
    point_df = pd.DataFrame(records)
    point_df.to_csv(point_path, index=False)

    rows = []
    axis_specs = [
        ('days_since_rollout_start', tuple(sorted(int(h) for h in horizons if int(h) > 0))),
    ]
    for age_axis, age_horizons in axis_specs:
        if age_axis not in point_df.columns:
            continue
        age_values = pd.to_numeric(point_df[age_axis], errors='coerce')
        if age_values.dropna().empty:
            continue
        start = 0
        for horizon in age_horizons:
            if int(horizon) <= start:
                continue
            in_bin = (age_values > float(start)) & (age_values <= float(horizon))
            bin_points = point_df[in_bin]
            if bin_points.empty:
                start = int(horizon)
                continue
            for band in ('whole', 'surface_le1m', 'le25m', 'gt25m'):
                if band == 'whole':
                    subset = bin_points
                elif band == 'surface_le1m':
                    subset = bin_points[bin_points['depth_m'] <= 1.0]
                elif band == 'le25m':
                    subset = bin_points[bin_points['depth_m'] <= DEPTH_STRATIFIED_RMSE_BOUNDARY_M]
                else:
                    subset = bin_points[bin_points['depth_m'] > DEPTH_STRATIFIED_RMSE_BOUNDARY_M]
                if subset.empty:
                    continue
                errors = subset['error_c'].to_numpy(dtype=np.float64)
                rows.append({
                    'lake_id': lake['lake_id'],
                    'lake_type': _lake_reservoir_bucket(lake),
                    'eval_split': str(split_label),
                    'rollout_mode': str(diagnostic_rollout_mode or 'free'),
                    'age_axis': age_axis,
                    'bin_start_exclusive_days': int(start),
                    'bin_end_days': int(horizon),
                    'depth_band': band,
                    'rmse_c': float(np.sqrt(np.mean(errors ** 2))),
                    'mae_c': float(np.mean(np.abs(errors))),
                    'bias_c': float(np.mean(errors)),
                    'point_count': int(len(subset)),
                    'profile_date_count': int(subset['Date'].nunique()),
                    'age_mean_days': float(pd.to_numeric(subset[age_axis], errors='coerce').mean()),
                })
            start = int(horizon)

    summary_path = None
    if rows:
        summary_path = output_dir / f"{split_token}_{lake_token}_{epoch_token}_free_roll_age_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
    return {
        'free_roll_point_diagnostics_csv': point_path,
        'free_roll_age_summary_csv': summary_path,
        'free_roll_point_diagnostics_count': int(len(point_df)),
        'free_roll_age_summary_count': int(len(rows)),
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
                    lakes[sample_idx]
                    .get('static_features_2d', lakes[sample_idx]['static_features'].reshape(1, -1))
                    .reshape(-1)
                    .to(device=depths.device, dtype=current_batch.dtype)
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
    horizons = tuple(sorted(int(horizon) for horizon in horizons if int(horizon) > 0))
    errors_by_horizon = {int(horizon): [] for horizon in horizons}
    biases_by_horizon = {int(horizon): [] for horizon in horizons}
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    if not lookup:
        return _horizon_metric_record(
            errors_by_horizon,
            biases_by_horizon,
            depth_errors_by_horizon=depth_errors_by_horizon,
        )
    depths_np = np.asarray(lake['depths_np'], dtype=np.float64).reshape(-1)
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
            squared_errors = diff ** 2
            errors_by_horizon[horizon].extend(squared_errors.tolist())
            _extend_depth_horizon_errors(
                depth_errors_by_horizon,
                horizon,
                squared_errors,
                depths_np[valid],
            )
            biases_by_horizon[horizon].append(float(np.mean(diff)))
    return _horizon_metric_record(
        errors_by_horizon,
        biases_by_horizon,
        depth_errors_by_horizon=depth_errors_by_horizon,
    )


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
    horizons = tuple(sorted(int(horizon) for horizon in horizons if int(horizon) > 0))
    errors_by_horizon = {int(horizon): [] for horizon in horizons}
    biases_by_horizon = {int(horizon): [] for horizon in horizons}
    depth_errors_by_horizon = _empty_depth_horizon_errors(horizons)
    device = lake['depths'].device
    tensor_stats = _empty_horizon_tensor_stats(horizons, device=device)
    depth_band_masks = _depth_band_masks_tensor(lake['depths'], device=device)
    if not lookup:
        return _horizon_metric_record(
            errors_by_horizon,
            biases_by_horizon,
            depth_errors_by_horizon=depth_errors_by_horizon,
        )
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
        return _horizon_metric_record(
            errors_by_horizon,
            biases_by_horizon,
            depth_errors_by_horizon=depth_errors_by_horizon,
        )

    batch_size = int(batch_size or 0)
    if batch_size <= 0:
        batch_size = len(sequences)
    for chunk in _batch_chunks(sequences, batch_size):
        start_indices = torch.as_tensor(
            [item[0] for item in chunk],
            dtype=torch.long,
            device=device,
        )
        current, _start_mask = _target_tensor_and_mask_batch(
            lake,
            lookup_split,
            [item[1] for item in chunk],
        )
        current = current.to(device=device, dtype=torch.float32).clone()
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
                _update_horizon_tensor_stats(
                    tensor_stats,
                    horizon,
                    prediction,
                    target,
                    valid,
                    depth_band_masks,
                )
    return _horizon_metric_record_from_tensor_stats(tensor_stats)


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
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_lswt_observer_mode='off',
    lswt_observer_strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
    lswt_observer_decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
    lswt_observer_max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
    lswt_observer_low_rank_deep_update_fraction=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
    lswt_observer_heat_content_limit_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
    lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    rollout_start_date=None,
    hard_density_stability=False,
    batch_size=32,
    diagnostic_output_dir=None,
    diagnostic_split_label='heldout_free_roll',
    diagnostic_epoch=None,
    diagnostic_rollout_mode='free',
):
    """Evaluate specified held-out full free-rolls, batching compatible lake geometries."""
    model.eval()
    if not lakes:
        return {}
    zero_profile_initializer = normalize_zero_profile_initializer_mode(zero_profile_initializer)
    spinup_lswt_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    rollout_lswt_observer_mode = normalize_lswt_observer_mode(rollout_lswt_observer_mode)
    force_scalar_rollout = (
        diagnostic_output_dir is not None
        or rollout_lswt_observer_mode != 'off'
        or str(init_mode).strip().lower() == 'zero_profile_low_dof'
        or zero_profile_initializer != DEFAULT_ZERO_PROFILE_INITIALIZER_MODE
        or spinup_lswt_observer_mode != DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE
    )
    if force_scalar_rollout:
        return {
            lake['lake_id']: evaluate_lake_free_roll(
                model,
                lake,
                task_mode=task_mode,
                horizons=horizons,
                init_mode=init_mode,
                spinup_days=spinup_days,
                zero_profile_initializer=zero_profile_initializer,
                spinup_lswt_observer_mode=spinup_lswt_observer_mode,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_lswt_observer_mode=rollout_lswt_observer_mode,
                lswt_observer_strength=lswt_observer_strength,
                lswt_observer_decay_depth_m=lswt_observer_decay_depth_m,
                lswt_observer_max_increment_c=lswt_observer_max_increment_c,
                lswt_observer_low_rank_deep_update_fraction=(
                    lswt_observer_low_rank_deep_update_fraction
                ),
                lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
                lswt_observer_min_quality=lswt_observer_min_quality,
                rollout_start_date=rollout_start_date,
                hard_density_stability=hard_density_stability,
                diagnostic_output_dir=diagnostic_output_dir,
                diagnostic_split_label=diagnostic_split_label,
                diagnostic_epoch=diagnostic_epoch,
                diagnostic_rollout_mode=diagnostic_rollout_mode,
            )
            for lake in lakes
        }
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
                zero_profile_initializer=zero_profile_initializer,
                spinup_lswt_observer_mode=spinup_lswt_observer_mode,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_lswt_observer_mode=rollout_lswt_observer_mode,
                lswt_observer_strength=lswt_observer_strength,
                lswt_observer_decay_depth_m=lswt_observer_decay_depth_m,
                lswt_observer_max_increment_c=lswt_observer_max_increment_c,
                lswt_observer_low_rank_deep_update_fraction=(
                    lswt_observer_low_rank_deep_update_fraction
                ),
                lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
                lswt_observer_min_quality=lswt_observer_min_quality,
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
                zero_profile_initializer=zero_profile_initializer,
                spinup_lswt_observer_mode=spinup_lswt_observer_mode,
                spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                rollout_lswt_observer_mode=rollout_lswt_observer_mode,
                lswt_observer_strength=lswt_observer_strength,
                lswt_observer_decay_depth_m=lswt_observer_decay_depth_m,
                lswt_observer_max_increment_c=lswt_observer_max_increment_c,
                lswt_observer_low_rank_deep_update_fraction=(
                    lswt_observer_low_rank_deep_update_fraction
                ),
                lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
                lswt_observer_min_quality=lswt_observer_min_quality,
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
                    zero_profile_initializer=zero_profile_initializer,
                    spinup_lswt_observer_mode=spinup_lswt_observer_mode,
                    spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                    spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                    lswt_observer_low_rank_deep_update_fraction=(
                        lswt_observer_low_rank_deep_update_fraction
                    ),
                    lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
                    lswt_observer_min_quality=lswt_observer_min_quality,
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
        'residual_abs_mean_c',
        'residual_surface_abs_mean_c',
        'residual_deep_abs_mean_c',
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
            'residual_abs_mean_c',
            'residual_surface_abs_mean_c',
            'residual_deep_abs_mean_c',
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
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    spinup_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_lswt_observer_mode='off',
    lswt_observer_strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
    lswt_observer_decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
    lswt_observer_max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
    lswt_observer_low_rank_deep_update_fraction=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
    lswt_observer_heat_content_limit_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
    lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    rollout_start_date=None,
    export_max_depth_m=None,
    hard_density_stability=False,
    hard_density_stability_mode=None,
    thermal_state_profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    thermal_state_profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    thermal_state_profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    thermal_state_profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    thermal_state_profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    thermal_state_profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
):
    """Export a full-year profile reconstruction rollout for one held-out lake.

    This intentionally avoids rolling post-processing and observed-profile
    reinitialization so zero-profile export diagnostics stay product-like.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_mode = normalize_task_mode(task_mode)
    zero_profile_initializer = normalize_zero_profile_initializer_mode(zero_profile_initializer)
    spinup_lswt_observer_mode = normalize_lswt_observer_mode(spinup_lswt_observer_mode)
    rollout_lswt_observer_mode = normalize_lswt_observer_mode(rollout_lswt_observer_mode)
    hard_density_stability_label = (
        str(hard_density_stability_mode)
        if hard_density_stability_mode is not None
        else ('on' if hard_density_stability else 'off')
    )

    df = lake['df']
    depths = lake['depths_np']
    all_lookup = lake['lookups']['all']
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
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=spinup_lswt_observer_mode,
        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
        thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
        thermal_state_profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
        thermal_state_profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
        thermal_state_profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
        thermal_state_profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
        thermal_state_profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
        lswt_observer_low_rank_deep_update_fraction=lswt_observer_low_rank_deep_update_fraction,
        lswt_observer_heat_content_limit_c=lswt_observer_heat_content_limit_c,
        lswt_observer_min_quality=lswt_observer_min_quality,
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
        rollout_observer_detail = None
        if rollout_lswt_observer_mode != 'off' and next_row is not None:
            current, rollout_observer_detail = apply_lswt_observer_update(
                current,
                next_row,
                lake['depths'],
                mode=rollout_lswt_observer_mode,
                strength=lswt_observer_strength,
                decay_depth_m=lswt_observer_decay_depth_m,
                max_increment_c=lswt_observer_max_increment_c,
                low_rank_deep_update_fraction=lswt_observer_low_rank_deep_update_fraction,
                heat_content_limit_c=lswt_observer_heat_content_limit_c,
                min_quality=lswt_observer_min_quality,
                area_profile=lake['area'],
                metadata=lake.get('metadata'),
            )
            temp_grid[:, day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
        elif rollout_observer_detail is None:
            rollout_observer_detail = {
                'lswt_observer_applied_count': torch.tensor(0.0, dtype=current.dtype, device=device),
                'lswt_observer_mean_abs_delta_c': torch.tensor(0.0, dtype=current.dtype, device=device),
                'lswt_observer_heat_content_delta_c': torch.tensor(0.0, dtype=current.dtype, device=device),
                'lswt_observer_deep_abs_delta_c': torch.tensor(0.0, dtype=current.dtype, device=device),
                'lswt_observer_filled_lst_used_count': torch.tensor(0.0, dtype=current.dtype, device=device),
            }
        temp_grid[:, day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
        diagnostic_records.append({
            'Date': pd.Timestamp(df['Date'].iloc[day_idx + 1]).date().isoformat(),
            'spinup_phase': False,
            'init_mode': init_state['init_mode'],
            'requested_init_mode': init_state['requested_init_mode'],
            'spinup_days_used': init_state['spinup_days_used'],
            'rollout_mode': 'free',
            'zero_profile_initializer': zero_profile_initializer,
            'spinup_lswt_observer_mode': spinup_lswt_observer_mode,
            'rollout_lswt_observer_mode': rollout_lswt_observer_mode,
            'rollout_lswt_observer_applied_count': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_applied_count',
            ),
            'rollout_lswt_observer_mean_abs_delta_c': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mean_abs_delta_c',
            ),
            'rollout_lswt_observer_heat_content_delta_c': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_heat_content_delta_c',
            ),
            'rollout_lswt_observer_deep_abs_delta_c': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_deep_abs_delta_c',
            ),
            'rollout_lswt_observer_filled_lst_used_count': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_filled_lst_used_count',
            ),
            'rollout_lswt_observer_kalman_gain_surface': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_kalman_gain_surface',
            ),
            'rollout_lswt_observer_kalman_gain_mean': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_kalman_gain_mean',
            ),
            'rollout_lswt_observer_observation_error_c': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_observation_error_c',
            ),
            'rollout_lswt_observer_state_variance_surface': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_state_variance_surface',
            ),
            'rollout_lswt_observer_localization_depth_m': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_localization_depth_m',
            ),
            'rollout_lswt_observer_reservoir_conservative_scale': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_reservoir_conservative_scale',
                1.0,
            ),
            'rollout_lswt_observer_heat_content_bound_scale': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_heat_content_bound_scale',
                1.0,
            ),
            'rollout_lswt_observer_mld_depth_m': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mld_depth_m',
            ),
            'rollout_lswt_observer_mld_weight_mean': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mld_weight_mean',
            ),
            'rollout_lswt_observer_mld_heat_content_delta_c': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mld_heat_content_delta_c',
            ),
            'rollout_lswt_observer_mld_volume_fraction': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mld_volume_fraction',
            ),
            'rollout_lswt_observer_mld_surface_to_heat_gain': _diag_scalar(
                rollout_observer_detail,
                'lswt_observer_mld_surface_to_heat_gain',
            ),
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
            'residual_surface_abs_mean_c': _diag_scalar(diagnostics, 'residual_surface_abs_mean_c'),
            'residual_deep_abs_mean_c': _diag_scalar(diagnostics, 'residual_deep_abs_mean_c'),
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
    suffix = 'heldout_state_reconstruction'
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
                'rollout_mode': 'free',
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

    model_label = (
        f"{metadata.get('lake_name', lake['lake_id'])} "
        "reconstruction-state profile reconstruction"
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


@torch.no_grad()
def export_daily_memory_reconstruction(
    model,
    lake,
    output_dir,
    *,
    export_max_depth_m=None,
    thermal_state_profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    thermal_state_profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    thermal_state_profile_fusion_lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    thermal_state_profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    thermal_state_profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    thermal_state_profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    thermal_state_profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_tensors = _daily_memory_basis_tensors(model, lake)
    if basis_tensors is None:
        raise ValueError('prediction_branch=daily_memory requires zero_profile_thermal_basis.')
    model.eval()
    df = lake['df'].copy()
    if 'full_doy' not in df.columns:
        dates = pd.to_datetime(df['Date'])
        df['full_doy'] = dates.dt.dayofyear.astype(float)
    depths = lake['depths_np']
    device = lake['depths'].device
    n_days = len(df)
    temp_grid = np.full((len(depths), n_days), np.nan, dtype=np.float32)
    diagnostic_records = []
    batch_size = 256
    for start in range(0, n_days, batch_size):
        end = min(n_days, start + batch_size)
        index_tensor = torch.arange(start, end, dtype=torch.long, device=device)
        encoded = _daily_memory_prediction_batch(
            model,
            lake,
            index_tensor,
            basis_tensors,
            profile_fusion_mode=thermal_state_profile_fusion_mode,
            profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
            profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
            profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
            profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
            profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
            profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
        )
        profiles = encoded['daily_profile_c'].detach().cpu().numpy().astype(np.float32)
        temp_grid[:, start:end] = profiles.T
        coeff_abs = encoded['coefficient_abs_mean'].detach().cpu().numpy().reshape(-1)
        coeff_std = encoded['coefficient_std'].detach().cpu().numpy().reshape(-1)
        conditioning_abs = encoded['conditioning_abs_mean'].detach().cpu().numpy().reshape(-1)
        component_count = encoded['component_count'].detach().cpu().numpy().reshape(-1)
        fusion_gate = encoded['thermal_state_profile_fusion_gate'].detach().cpu().numpy().reshape(-1)
        fusion_age = encoded['thermal_state_profile_fusion_age_days'].detach().cpu().numpy().reshape(-1)
        fusion_coverage = encoded[
            'thermal_state_profile_fusion_coverage_fraction'
        ].detach().cpu().numpy().reshape(-1)
        fusion_span = encoded[
            'thermal_state_profile_fusion_depth_span_fraction'
        ].detach().cpu().numpy().reshape(-1)
        fusion_future = encoded[
            'thermal_state_profile_fusion_future_fraction'
        ].detach().cpu().numpy().reshape(-1)
        for local_idx, day_idx in enumerate(range(start, end)):
            row = df.iloc[day_idx]
            diagnostic_records.append({
                'Date': pd.Timestamp(row['Date']).date().isoformat(),
                'prediction_branch': 'daily_memory',
                'daily_memory_coefficient_abs_mean': float(coeff_abs[local_idx]),
                'daily_memory_coefficient_std': float(coeff_std[local_idx]),
                'daily_memory_conditioning_abs_mean': float(conditioning_abs[local_idx]),
                'daily_memory_component_count': float(component_count[local_idx]),
                'thermal_state_profile_fusion_gate': float(fusion_gate[local_idx]),
                'thermal_state_profile_fusion_age_days': float(fusion_age[local_idx]),
                'thermal_state_profile_fusion_coverage_fraction': float(fusion_coverage[local_idx]),
                'thermal_state_profile_fusion_depth_span_fraction': float(fusion_span[local_idx]),
                'thermal_state_profile_fusion_future_fraction': float(fusion_future[local_idx]),
                'surface_temp_c': float(profiles[local_idx, 0]),
                'mean_temp_c': float(np.nanmean(profiles[local_idx])),
            })
    if n_days > 1:
        all_indices = torch.arange(0, n_days - 1, dtype=torch.long, device=device)
        for start in range(0, int(all_indices.numel()), batch_size):
            index_tensor = all_indices[start:start + batch_size]
            encoded = _daily_memory_prediction_batch(
                model,
                lake,
                index_tensor,
                basis_tensors,
                profile_fusion_mode=thermal_state_profile_fusion_mode,
                profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
            )
            next_encoded = _daily_memory_prediction_batch(
                model,
                lake,
                index_tensor + 1,
                basis_tensors,
                profile_fusion_mode=thermal_state_profile_fusion_mode,
                profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
            )
            terms = _daily_memory_heat_and_physics_terms(
                model,
                lake,
                encoded['daily_profile_c'],
                next_encoded['daily_profile_c'],
                index_tensor,
                task_mode='analysis',
                hard_density_stability=False,
                step_diagnostic_mode='loss',
            )
            residual = terms['heat_budget_residual'].detach().cpu().numpy().reshape(-1)
            heat_loss = terms['heat_budget_loss'].detach().cpu().numpy().reshape(-1)
            phys_loss = terms['physics_consistency_loss'].detach().cpu().numpy().reshape(-1)
            for local_idx, day_idx in enumerate(index_tensor.detach().cpu().numpy().astype(int).tolist()):
                diagnostic_records[day_idx]['daily_memory_heat_budget_residual_wm2'] = float(residual[local_idx])
                diagnostic_records[day_idx]['daily_memory_heat_budget_loss'] = float(heat_loss[local_idx])
                diagnostic_records[day_idx]['daily_memory_physics_consistency_loss'] = float(phys_loss[local_idx])
    metadata = dict(lake['metadata'])
    metadata['file_tag'] = str(lake['lake_id'])
    metadata.setdefault('lake_name', str(lake['lake_id']).replace('_', ' ').title())
    suffix = 'heldout_state_reconstruction_daily_memory'
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
                'prediction_branch': 'daily_memory',
                'zero_profile_thermal_basis_component_count': int(
                    (getattr(model, 'zero_profile_thermal_basis', {}) or {}).get(
                        'component_count',
                        basis_tensors['components_on_depth'].shape[0],
                    )
                ),
                **_thermal_state_profile_fusion_config_fields(
                    mode=thermal_state_profile_fusion_mode,
                    time_policy=thermal_state_profile_fusion_time_policy,
                    lookup_split=thermal_state_profile_fusion_lookup_split,
                    max_age_days=thermal_state_profile_fusion_max_age_days,
                    min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                    max_weight=thermal_state_profile_fusion_max_weight,
                    coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
                ),
                **export_depth_info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    heatmap_path = output_dir / f"{metadata['file_tag']}_{suffix}_year_heatmap.png"
    plot_year_heatmap(df, export_temp_grid, export_depths, heatmap_path, metadata)
    model_label = (
        f"{metadata.get('lake_name', lake['lake_id'])} "
        "daily-memory EOF/PCA profile reconstruction"
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
    profile_loss_target_mode=DEFAULT_PROFILE_LOSS_TARGET_MODE,
    profile_sampling_mode=DEFAULT_PROFILE_SAMPLING_MODE,
    test_lake_id=None,
    test_lake_ids=None,
    heldout_lake_groups=None,
    residual_limit_c=DEFAULT_RESIDUAL_LIMIT_C,
    wind_kz_scale=1.0,
    autumn_convective_boost=1.0,
    profile_huber_delta=2.0,
    lst_surface_weight=0.03,
    lst_feature_dropout_probability=0.20,
    energy_balance_weight=0.001,
    residual_regularization_weight=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
    daily_tendency_weight=DEFAULT_DAILY_TENDENCY_WEIGHT,
    physical_scale_regularization_weight=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
    physical_scale_smoothness_weight=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
    kd_prior_regularization_weight=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
    adaptive_parameter_regularization_weight=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
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
    multitask_auxiliary_weight=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
    multitask_auxiliary_heat_weight=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
    multitask_auxiliary_thermocline_weight=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
    multitask_auxiliary_mld_weight=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
    multitask_auxiliary_stability_weight=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
    multitask_auxiliary_surface_bottom_weight=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
    multitask_auxiliary_eof_weight=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
    multitask_auxiliary_hidden_dim=DEFAULT_MULTITASK_AUXILIARY_HIDDEN_DIM,
    physics_rollout_thermal_state_weight=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
    unlabeled_heat_closure_weight=DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT,
    unlabeled_heat_closure_batch_size=DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE,
    unlabeled_heat_closure_window_days=DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS,
    unlabeled_heat_closure_horizons=DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS,
    unlabeled_heat_closure_tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2,
    unlabeled_heat_closure_open_water_only=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
    unlabeled_heat_closure_lst_qc_min=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
    unlabeled_heat_closure_reservoir_mode=DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE,
    unlabeled_heat_closure_mode=DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
    unlabeled_heat_closure_state_source=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
    unlabeled_heat_closure_spinup_days=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
    unlabeled_heat_closure_solver_guard_weight=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
    unlabeled_heat_closure_solver_guard_tau_wm2=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2,
    no_profile_lst_surface_weight=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
    warm_season_column_heat_content_weight=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_WEIGHT,
    warm_season_column_heat_content_quantile_low=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_LOW,
    warm_season_column_heat_content_quantile_high=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_QUANTILE_HIGH,
    warm_season_column_heat_content_min_gap_days=DEFAULT_WARM_SEASON_COLUMN_HEAT_CONTENT_MIN_GAP_DAYS,
    teacher_forcing_start=0.7,
    teacher_forcing_end=0.0,
    state_noise_weight=1.0,
    residual_time_smooth_weight=DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT,
    rolling_horizon_eval_max_starts=40,
    export_style_validation=DEFAULT_EXPORT_STYLE_VALIDATION_MODE,
    export_style_validation_max_lakes=DEFAULT_EXPORT_STYLE_VALIDATION_MAX_LAKES,
    full_eval_point_diagnostics=DEFAULT_FULL_EVAL_POINT_DIAGNOSTICS_MODE,
    zero_profile_export_validation=DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MODE,
    zero_profile_export_validation_max_lakes=DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MAX_LAKES,
    zero_profile_initializer=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
    zero_profile_thermal_basis_components=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_COMPONENTS,
    zero_profile_thermal_basis_grid_points=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_GRID_POINTS,
    zero_profile_thermal_basis_balance_mode=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODE,
    refit_zero_profile_thermal_basis=DEFAULT_REFIT_ZERO_PROFILE_THERMAL_BASIS,
    zero_profile_init_net_loss_weight=DEFAULT_ZERO_PROFILE_INIT_NET_LOSS_WEIGHT,
    zero_profile_init_net_start_epoch=DEFAULT_ZERO_PROFILE_INIT_NET_START_EPOCH,
    zero_profile_init_net_ramp_epochs=DEFAULT_ZERO_PROFILE_INIT_NET_RAMP_EPOCHS,
    zero_profile_init_net_samples_per_lake=DEFAULT_ZERO_PROFILE_INIT_NET_SAMPLES_PER_LAKE,
    zero_profile_init_net_regularization_weight=DEFAULT_ZERO_PROFILE_INIT_NET_REGULARIZATION_WEIGHT,
    zero_profile_init_net_hidden_dim=DEFAULT_ZERO_PROFILE_INIT_NET_HIDDEN_DIM,
    zero_profile_init_net_init_spread=DEFAULT_ZERO_PROFILE_INIT_NET_INIT_SPREAD,
    zero_profile_init_net_coeff_limit_sigma=DEFAULT_ZERO_PROFILE_INIT_NET_COEFF_LIMIT_SIGMA,
    zero_profile_init_net_delta_limit_c=DEFAULT_ZERO_PROFILE_INIT_NET_DELTA_LIMIT_C,
    zero_profile_init_net_training_spinup_days=DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS,
    zero_profile_init_net_physics_weight=DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT,
    zero_profile_init_net_rollout_weight=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT,
    zero_profile_init_net_rollout_max_days=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
    zero_profile_init_net_rollout_targets=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
    daily_memory_reconstruction_weight=DEFAULT_DAILY_MEMORY_RECONSTRUCTION_WEIGHT,
    daily_memory_samples_per_lake=DEFAULT_DAILY_MEMORY_SAMPLES_PER_LAKE,
    daily_memory_temporal_smoothness_weight=DEFAULT_DAILY_MEMORY_TEMPORAL_SMOOTHNESS_WEIGHT,
    daily_memory_heat_budget_weight=DEFAULT_DAILY_MEMORY_HEAT_BUDGET_WEIGHT,
    daily_memory_physics_consistency_weight=DEFAULT_DAILY_MEMORY_PHYSICS_CONSISTENCY_WEIGHT,
    daily_memory_regularization_weight=DEFAULT_DAILY_MEMORY_REGULARIZATION_WEIGHT,
    daily_memory_coefficient_loss_weight=DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_WEIGHT,
    daily_memory_start_epoch=DEFAULT_DAILY_MEMORY_START_EPOCH,
    daily_memory_ramp_epochs=DEFAULT_DAILY_MEMORY_RAMP_EPOCHS,
    daily_memory_hidden_dim=DEFAULT_DAILY_MEMORY_HIDDEN_DIM,
    daily_memory_init_spread=DEFAULT_DAILY_MEMORY_INIT_SPREAD,
    daily_memory_coeff_limit_sigma=DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA,
    thermal_state_profile_fusion_mode=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
    thermal_state_profile_fusion_time_policy=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
    thermal_state_profile_fusion_lookup_split=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
    thermal_state_profile_fusion_max_age_days=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
    thermal_state_profile_fusion_min_depth_fraction=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
    thermal_state_profile_fusion_max_weight=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
    thermal_state_profile_fusion_coeff_limit_sigma=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
    prediction_branch='physics_rollout',
    model_mainline='auto',
    zero_profile_lswt_observer_mode=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
    zero_profile_spinup_days_matrix=DEFAULT_ZERO_PROFILE_SPINUP_DAYS_MATRIX,
    zero_profile_lswt_observer_strength=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
    zero_profile_lswt_observer_decay_depth_m=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
    zero_profile_lswt_observer_max_increment_c=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
    zero_profile_lswt_observer_deep_update_fraction=(
        DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION
    ),
    zero_profile_lswt_observer_heat_content_limit_c=(
        DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C
    ),
    zero_profile_lswt_observer_min_quality=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
    kd_saturation_threshold=DEFAULT_KD_SATURATION_THRESHOLD,
    kd_saturation_penalty_weight=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
    init_mode='profile',
    spinup_days=90,
    spinup_lst_assimilation_strength=0.08,
    spinup_lst_assimilation_decay_depth_m=2.0,
    spinup_lst_assimilation_max_increment_c=0.5,
    rollout_start_date=None,
    checkpoint_path=None,
    resume_checkpoint=None,
    checkpoint_every_epochs=5,
    eval_every_epochs=None,
    full_eval_every_epochs=None,
    full_eval_start_epoch=0,
    profile_runtime=True,
    profile_gpu=False,
    history_diagnostic_every_epochs=0,
    torch_tf32='on',
    torch_matmul_precision='high',
    training_amp=DEFAULT_TRAINING_AMP,
    training_history_detail_every_epochs=DEFAULT_TRAINING_HISTORY_DETAIL_EVERY_EPOCHS,
    transition_batch_size=0,
    segment_rollout_batch_size=0,
    rolling_horizon_batch_size=32,
    train_diagnostic_mode='loss',
    seed=None,
    export_after_training='off',
    export_max_depth_m=None,
    cross_lake_batch_mode='off',
    cross_lake_batch_size=0,
    gpu_batch_autotune=DEFAULT_GPU_BATCH_AUTOTUNE,
    gpu_batch_autotune_target_batch_size=DEFAULT_GPU_BATCH_AUTOTUNE_TARGET_BATCH_SIZE,
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
    adaptive_wind_kz_min=DEFAULT_ADAPTIVE_WIND_KZ_MIN,
    adaptive_wind_kz_max=DEFAULT_ADAPTIVE_WIND_KZ_MAX,
    adaptive_blend_alpha_min=DEFAULT_ADAPTIVE_BLEND_ALPHA_MIN,
    adaptive_blend_alpha_max=DEFAULT_ADAPTIVE_BLEND_ALPHA_MAX,
    adaptive_kd_multiplier_min=DEFAULT_ADAPTIVE_KD_MULTIPLIER_MIN,
    adaptive_kd_multiplier_max=DEFAULT_ADAPTIVE_KD_MULTIPLIER_MAX,
    adaptive_turbulent_exchange_scale_min=DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MIN,
    adaptive_turbulent_exchange_scale_max=DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MAX,
    adaptive_convective_mixing_scale_min=DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MIN,
    adaptive_convective_mixing_scale_max=DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MAX,
    adaptive_ice_shortwave_scale_min=DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MIN,
    adaptive_ice_shortwave_scale_max=DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MAX,
    device=None,
):
    manifest = _read_manifest(manifest_path)
    epochs = int(manifest.get('epochs', epochs))
    if epochs < 0:
        raise ValueError('epochs must be non-negative.')
    if seed is None and manifest.get('seed') is not None:
        seed = int(manifest['seed'])
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
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
    _reject_removed_zero_profile_fields(manifest)
    task_mode = 'analysis'
    data_fill_mode = 'reconstruction'
    profile_supervision_scope = _normalize_profile_supervision_scope(manifest.get(
        'profile_supervision_scope',
        profile_supervision_scope,
    ))
    profile_loss_target_mode = _normalize_profile_loss_target_mode(manifest.get(
        'profile_loss_target_mode',
        profile_loss_target_mode,
    ))
    profile_sampling_mode = _normalize_profile_sampling_mode(manifest.get(
        'profile_sampling_mode',
        profile_sampling_mode,
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
    multitask_auxiliary_weight = float(manifest.get(
        'multitask_auxiliary_weight',
        multitask_auxiliary_weight,
    ))
    multitask_auxiliary_heat_weight = float(manifest.get(
        'multitask_auxiliary_heat_weight',
        multitask_auxiliary_heat_weight,
    ))
    multitask_auxiliary_thermocline_weight = float(manifest.get(
        'multitask_auxiliary_thermocline_weight',
        multitask_auxiliary_thermocline_weight,
    ))
    multitask_auxiliary_mld_weight = float(manifest.get(
        'multitask_auxiliary_mld_weight',
        multitask_auxiliary_mld_weight,
    ))
    multitask_auxiliary_stability_weight = float(manifest.get(
        'multitask_auxiliary_stability_weight',
        multitask_auxiliary_stability_weight,
    ))
    multitask_auxiliary_surface_bottom_weight = float(manifest.get(
        'multitask_auxiliary_surface_bottom_weight',
        multitask_auxiliary_surface_bottom_weight,
    ))
    multitask_auxiliary_eof_weight = float(manifest.get(
        'multitask_auxiliary_eof_weight',
        multitask_auxiliary_eof_weight,
    ))
    multitask_auxiliary_hidden_dim = int(manifest.get(
        'multitask_auxiliary_hidden_dim',
        multitask_auxiliary_hidden_dim,
    ))
    physics_rollout_thermal_state_weight = float(manifest.get(
        'physics_rollout_thermal_state_weight',
        physics_rollout_thermal_state_weight,
    ))
    multitask_values = {
        'multitask_auxiliary_weight': multitask_auxiliary_weight,
        'multitask_auxiliary_heat_weight': multitask_auxiliary_heat_weight,
        'multitask_auxiliary_thermocline_weight': multitask_auxiliary_thermocline_weight,
        'multitask_auxiliary_mld_weight': multitask_auxiliary_mld_weight,
        'multitask_auxiliary_stability_weight': multitask_auxiliary_stability_weight,
        'multitask_auxiliary_surface_bottom_weight': multitask_auxiliary_surface_bottom_weight,
        'multitask_auxiliary_eof_weight': multitask_auxiliary_eof_weight,
        'physics_rollout_thermal_state_weight': physics_rollout_thermal_state_weight,
    }
    for name, value in multitask_values.items():
        if value < 0.0:
            raise ValueError(f'{name} must be non-negative.')
    if multitask_auxiliary_hidden_dim <= 0:
        raise ValueError('multitask_auxiliary_hidden_dim must be positive.')
    unlabeled_heat_closure_weight = float(manifest.get(
        'unlabeled_heat_closure_weight',
        unlabeled_heat_closure_weight,
    ))
    if unlabeled_heat_closure_weight < 0.0:
        raise ValueError('unlabeled_heat_closure_weight must be non-negative.')
    unlabeled_heat_closure_batch_size = int(manifest.get(
        'unlabeled_heat_closure_batch_size',
        unlabeled_heat_closure_batch_size,
    ))
    if unlabeled_heat_closure_batch_size < 0:
        raise ValueError('unlabeled_heat_closure_batch_size must be non-negative.')
    unlabeled_heat_closure_window_days = int(manifest.get(
        'unlabeled_heat_closure_window_days',
        unlabeled_heat_closure_window_days,
    ))
    if unlabeled_heat_closure_window_days < 1:
        raise ValueError('unlabeled_heat_closure_window_days must be at least 1.')
    unlabeled_heat_closure_horizons = _parse_unlabeled_heat_closure_horizons(
        manifest.get(
            'unlabeled_heat_closure_horizons',
            unlabeled_heat_closure_horizons,
        ),
        fallback_window_days=unlabeled_heat_closure_window_days,
    )
    unlabeled_heat_closure_window_days = int(max(unlabeled_heat_closure_horizons))
    unlabeled_heat_closure_tau_wm2 = float(manifest.get(
        'unlabeled_heat_closure_tau_wm2',
        unlabeled_heat_closure_tau_wm2,
    ))
    if unlabeled_heat_closure_tau_wm2 <= 0.0:
        raise ValueError('unlabeled_heat_closure_tau_wm2 must be positive.')
    unlabeled_heat_closure_open_water_only = _normalize_on_off(
        manifest.get('unlabeled_heat_closure_open_water_only', unlabeled_heat_closure_open_water_only),
        field_name='unlabeled_heat_closure_open_water_only',
    )
    unlabeled_heat_closure_lst_qc_min = float(manifest.get(
        'unlabeled_heat_closure_lst_qc_min',
        unlabeled_heat_closure_lst_qc_min,
    ))
    if not (0.0 <= unlabeled_heat_closure_lst_qc_min <= 1.0):
        raise ValueError('unlabeled_heat_closure_lst_qc_min must be between 0 and 1.')
    unlabeled_heat_closure_reservoir_mode = _normalize_unlabeled_heat_closure_reservoir_mode(
        manifest.get('unlabeled_heat_closure_reservoir_mode', unlabeled_heat_closure_reservoir_mode)
    )
    unlabeled_heat_closure_mode = _normalize_unlabeled_heat_closure_mode(
        manifest.get('unlabeled_heat_closure_mode', unlabeled_heat_closure_mode)
    )
    unlabeled_heat_closure_state_source = _normalize_unlabeled_heat_closure_state_source(
        manifest.get(
            'unlabeled_heat_closure_state_source',
            unlabeled_heat_closure_state_source,
        )
    )
    unlabeled_heat_closure_spinup_days = int(manifest.get(
        'unlabeled_heat_closure_spinup_days',
        unlabeled_heat_closure_spinup_days,
    ))
    if unlabeled_heat_closure_spinup_days < 0:
        raise ValueError('unlabeled_heat_closure_spinup_days must be non-negative.')
    unlabeled_heat_closure_solver_guard_weight = float(manifest.get(
        'unlabeled_heat_closure_solver_guard_weight',
        unlabeled_heat_closure_solver_guard_weight,
    ))
    if unlabeled_heat_closure_solver_guard_weight < 0.0:
        raise ValueError('unlabeled_heat_closure_solver_guard_weight must be non-negative.')
    unlabeled_heat_closure_solver_guard_tau_wm2 = float(manifest.get(
        'unlabeled_heat_closure_solver_guard_tau_wm2',
        unlabeled_heat_closure_solver_guard_tau_wm2,
    ))
    if unlabeled_heat_closure_solver_guard_tau_wm2 <= 0.0:
        raise ValueError('unlabeled_heat_closure_solver_guard_tau_wm2 must be positive.')
    no_profile_lst_surface_weight = float(manifest.get(
        'no_profile_lst_surface_weight',
        no_profile_lst_surface_weight,
    ))
    if no_profile_lst_surface_weight < 0.0:
        raise ValueError('no_profile_lst_surface_weight must be non-negative.')
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
    export_style_validation = _normalize_on_off(
        manifest.get('export_style_validation', export_style_validation),
        field_name='export_style_validation',
    )
    export_style_validation_max_lakes = int(manifest.get(
        'export_style_validation_max_lakes',
        export_style_validation_max_lakes,
    ))
    if export_style_validation_max_lakes < 0:
        raise ValueError('export_style_validation_max_lakes must be non-negative.')
    full_eval_point_diagnostics = _normalize_on_off(
        manifest.get('full_eval_point_diagnostics', full_eval_point_diagnostics),
        name='full_eval_point_diagnostics',
    )
    zero_profile_export_validation = _normalize_on_off(
        manifest.get('zero_profile_export_validation', zero_profile_export_validation),
        field_name='zero_profile_export_validation',
    )
    zero_profile_export_validation_max_lakes = int(manifest.get(
        'zero_profile_export_validation_max_lakes',
        zero_profile_export_validation_max_lakes,
    ))
    if zero_profile_export_validation_max_lakes < 0:
        raise ValueError('zero_profile_export_validation_max_lakes must be non-negative.')
    zero_profile_initializer = normalize_mainline_zero_profile_initializer_mode(manifest.get(
        'zero_profile_initializer',
        zero_profile_initializer,
    ))
    zero_profile_thermal_basis_components = int(manifest.get(
        'zero_profile_thermal_basis_components',
        zero_profile_thermal_basis_components,
    ))
    if zero_profile_thermal_basis_components < 1:
        raise ValueError('zero_profile_thermal_basis_components must be positive.')
    zero_profile_thermal_basis_grid_points = int(manifest.get(
        'zero_profile_thermal_basis_grid_points',
        zero_profile_thermal_basis_grid_points,
    ))
    if zero_profile_thermal_basis_grid_points < 4:
        raise ValueError('zero_profile_thermal_basis_grid_points must be at least 4.')
    zero_profile_thermal_basis_balance_mode = normalize_zero_profile_thermal_basis_balance_mode(
        manifest.get(
            'zero_profile_thermal_basis_balance_mode',
            zero_profile_thermal_basis_balance_mode,
        )
    )
    refit_zero_profile_thermal_basis = _normalize_on_off(
        manifest.get('refit_zero_profile_thermal_basis', refit_zero_profile_thermal_basis),
        field_name='refit_zero_profile_thermal_basis',
    )
    zero_profile_init_net_loss_weight = float(manifest.get(
        'zero_profile_init_net_loss_weight',
        zero_profile_init_net_loss_weight,
    ))
    if zero_profile_init_net_loss_weight < 0.0:
        raise ValueError('zero_profile_init_net_loss_weight must be non-negative.')
    zero_profile_init_net_start_epoch = int(manifest.get(
        'zero_profile_init_net_start_epoch',
        zero_profile_init_net_start_epoch,
    ))
    if zero_profile_init_net_start_epoch < 0:
        raise ValueError('zero_profile_init_net_start_epoch must be non-negative.')
    zero_profile_init_net_ramp_epochs = int(manifest.get(
        'zero_profile_init_net_ramp_epochs',
        zero_profile_init_net_ramp_epochs,
    ))
    if zero_profile_init_net_ramp_epochs < 0:
        raise ValueError('zero_profile_init_net_ramp_epochs must be non-negative.')
    zero_profile_init_net_samples_per_lake = int(manifest.get(
        'zero_profile_init_net_samples_per_lake',
        zero_profile_init_net_samples_per_lake,
    ))
    if zero_profile_init_net_samples_per_lake < 0:
        raise ValueError('zero_profile_init_net_samples_per_lake must be non-negative.')
    zero_profile_init_net_regularization_weight = float(manifest.get(
        'zero_profile_init_net_regularization_weight',
        zero_profile_init_net_regularization_weight,
    ))
    if zero_profile_init_net_regularization_weight < 0.0:
        raise ValueError('zero_profile_init_net_regularization_weight must be non-negative.')
    zero_profile_init_net_hidden_dim = int(manifest.get(
        'zero_profile_init_net_hidden_dim',
        zero_profile_init_net_hidden_dim,
    ))
    if zero_profile_init_net_hidden_dim <= 0:
        raise ValueError('zero_profile_init_net_hidden_dim must be positive.')
    zero_profile_init_net_init_spread = float(manifest.get(
        'zero_profile_init_net_init_spread',
        zero_profile_init_net_init_spread,
    ))
    if zero_profile_init_net_init_spread < 0.0:
        raise ValueError('zero_profile_init_net_init_spread must be non-negative.')
    zero_profile_init_net_coeff_limit_sigma = float(manifest.get(
        'zero_profile_init_net_coeff_limit_sigma',
        zero_profile_init_net_coeff_limit_sigma,
    ))
    if zero_profile_init_net_coeff_limit_sigma < 0.0:
        raise ValueError('zero_profile_init_net_coeff_limit_sigma must be non-negative.')
    zero_profile_init_net_delta_limit_c = float(manifest.get(
        'zero_profile_init_net_delta_limit_c',
        zero_profile_init_net_delta_limit_c,
    ))
    if zero_profile_init_net_delta_limit_c < 0.0:
        raise ValueError('zero_profile_init_net_delta_limit_c must be non-negative.')
    zero_profile_init_net_training_spinup_days = int(manifest.get(
        'zero_profile_init_net_training_spinup_days',
        zero_profile_init_net_training_spinup_days,
    ))
    if zero_profile_init_net_training_spinup_days < 0:
        raise ValueError('zero_profile_init_net_training_spinup_days must be non-negative.')
    zero_profile_init_net_physics_weight = float(manifest.get(
        'zero_profile_init_net_physics_weight',
        zero_profile_init_net_physics_weight,
    ))
    if zero_profile_init_net_physics_weight < 0.0:
        raise ValueError('zero_profile_init_net_physics_weight must be non-negative.')
    zero_profile_init_net_rollout_weight = float(manifest.get(
        'zero_profile_init_net_rollout_weight',
        zero_profile_init_net_rollout_weight,
    ))
    if zero_profile_init_net_rollout_weight < 0.0:
        raise ValueError('zero_profile_init_net_rollout_weight must be non-negative.')
    zero_profile_init_net_rollout_max_days = int(manifest.get(
        'zero_profile_init_net_rollout_max_days',
        zero_profile_init_net_rollout_max_days,
    ))
    if zero_profile_init_net_rollout_max_days < 0:
        raise ValueError('zero_profile_init_net_rollout_max_days must be non-negative.')
    zero_profile_init_net_rollout_targets = int(manifest.get(
        'zero_profile_init_net_rollout_targets',
        zero_profile_init_net_rollout_targets,
    ))
    if zero_profile_init_net_rollout_targets < 0:
        raise ValueError('zero_profile_init_net_rollout_targets must be non-negative.')
    daily_memory_reconstruction_weight = float(manifest.get(
        'daily_memory_reconstruction_weight',
        daily_memory_reconstruction_weight,
    ))
    if daily_memory_reconstruction_weight < 0.0:
        raise ValueError('daily_memory_reconstruction_weight must be non-negative.')
    daily_memory_samples_per_lake = int(manifest.get(
        'daily_memory_samples_per_lake',
        daily_memory_samples_per_lake,
    ))
    if daily_memory_samples_per_lake < 0:
        raise ValueError('daily_memory_samples_per_lake must be non-negative.')
    daily_memory_temporal_smoothness_weight = float(manifest.get(
        'daily_memory_temporal_smoothness_weight',
        daily_memory_temporal_smoothness_weight,
    ))
    daily_memory_heat_budget_weight = float(manifest.get(
        'daily_memory_heat_budget_weight',
        daily_memory_heat_budget_weight,
    ))
    daily_memory_physics_consistency_weight = float(manifest.get(
        'daily_memory_physics_consistency_weight',
        daily_memory_physics_consistency_weight,
    ))
    daily_memory_regularization_weight = float(manifest.get(
        'daily_memory_regularization_weight',
        daily_memory_regularization_weight,
    ))
    daily_memory_coefficient_loss_weight = float(manifest.get(
        'daily_memory_coefficient_loss_weight',
        daily_memory_coefficient_loss_weight,
    ))
    for name, value in {
        'daily_memory_temporal_smoothness_weight': daily_memory_temporal_smoothness_weight,
        'daily_memory_heat_budget_weight': daily_memory_heat_budget_weight,
        'daily_memory_physics_consistency_weight': daily_memory_physics_consistency_weight,
        'daily_memory_regularization_weight': daily_memory_regularization_weight,
        'daily_memory_coefficient_loss_weight': daily_memory_coefficient_loss_weight,
    }.items():
        if value < 0.0:
            raise ValueError(f'{name} must be non-negative.')
    daily_memory_start_epoch = int(manifest.get(
        'daily_memory_start_epoch',
        daily_memory_start_epoch,
    ))
    if daily_memory_start_epoch < 0:
        raise ValueError('daily_memory_start_epoch must be non-negative.')
    daily_memory_ramp_epochs = int(manifest.get(
        'daily_memory_ramp_epochs',
        daily_memory_ramp_epochs,
    ))
    if daily_memory_ramp_epochs < 0:
        raise ValueError('daily_memory_ramp_epochs must be non-negative.')
    daily_memory_hidden_dim = int(manifest.get(
        'daily_memory_hidden_dim',
        daily_memory_hidden_dim,
    ))
    if daily_memory_hidden_dim <= 0:
        raise ValueError('daily_memory_hidden_dim must be positive.')
    daily_memory_init_spread = float(manifest.get(
        'daily_memory_init_spread',
        daily_memory_init_spread,
    ))
    if daily_memory_init_spread < 0.0:
        raise ValueError('daily_memory_init_spread must be non-negative.')
    daily_memory_coeff_limit_sigma = float(manifest.get(
        'daily_memory_coeff_limit_sigma',
        daily_memory_coeff_limit_sigma,
    ))
    if daily_memory_coeff_limit_sigma < 0.0:
        raise ValueError('daily_memory_coeff_limit_sigma must be non-negative.')
    thermal_state_profile_fusion_mode = _normalize_thermal_state_profile_fusion_mode(
        manifest.get('thermal_state_profile_fusion_mode', thermal_state_profile_fusion_mode)
    )
    thermal_state_profile_fusion_time_policy = _normalize_thermal_state_profile_fusion_time_policy(
        manifest.get(
            'thermal_state_profile_fusion_time_policy',
            thermal_state_profile_fusion_time_policy,
        )
    )
    thermal_state_profile_fusion_lookup_split = str(manifest.get(
        'thermal_state_profile_fusion_lookup_split',
        thermal_state_profile_fusion_lookup_split,
    ))
    thermal_state_profile_fusion_max_age_days = int(manifest.get(
        'thermal_state_profile_fusion_max_age_days',
        thermal_state_profile_fusion_max_age_days,
    ))
    thermal_state_profile_fusion_min_depth_fraction = float(manifest.get(
        'thermal_state_profile_fusion_min_depth_fraction',
        thermal_state_profile_fusion_min_depth_fraction,
    ))
    if thermal_state_profile_fusion_min_depth_fraction <= 0.0:
        raise ValueError('thermal_state_profile_fusion_min_depth_fraction must be positive.')
    thermal_state_profile_fusion_max_weight = float(manifest.get(
        'thermal_state_profile_fusion_max_weight',
        thermal_state_profile_fusion_max_weight,
    ))
    if not (0.0 <= thermal_state_profile_fusion_max_weight <= 1.0):
        raise ValueError('thermal_state_profile_fusion_max_weight must be between 0 and 1.')
    thermal_state_profile_fusion_coeff_limit_sigma = float(manifest.get(
        'thermal_state_profile_fusion_coeff_limit_sigma',
        thermal_state_profile_fusion_coeff_limit_sigma,
    ))
    if thermal_state_profile_fusion_coeff_limit_sigma <= 0.0:
        raise ValueError('thermal_state_profile_fusion_coeff_limit_sigma must be positive.')
    prediction_branch = _normalize_prediction_branch(
        manifest.get('prediction_branch', prediction_branch)
    )
    model_mainline = _normalize_model_mainline(manifest.get('model_mainline', model_mainline))
    model_mainline_resolved = _resolve_model_mainline(
        model_mainline,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
    )
    multitask_auxiliary_weight_requested = float(multitask_auxiliary_weight)
    (
        multitask_auxiliary_weight,
        physics_rollout_thermal_state_weight_eff,
    ) = _resolve_physics_rollout_thermal_state_weight(
        model_mainline_resolved,
        requested_weight=physics_rollout_thermal_state_weight,
        multitask_auxiliary_weight=multitask_auxiliary_weight,
    )
    thermal_state_profile_fusion_config = _thermal_state_profile_fusion_config_fields(
        mode=thermal_state_profile_fusion_mode,
        time_policy=thermal_state_profile_fusion_time_policy,
        lookup_split=thermal_state_profile_fusion_lookup_split,
        max_age_days=thermal_state_profile_fusion_max_age_days,
        min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
        max_weight=thermal_state_profile_fusion_max_weight,
        coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
    )
    daily_memory_role = _daily_memory_training_role(
        model_mainline_resolved,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
    )
    if daily_memory_role == 'prediction':
        if daily_memory_heat_budget_weight <= 0.0 or daily_memory_physics_consistency_weight <= 0.0:
            raise ValueError(
                "prediction_branch='daily_memory' is a physics-primary non-rolling branch; "
                'daily_memory_heat_budget_weight and '
                'daily_memory_physics_consistency_weight must both be > 0. '
                'Profiles may supervise training, but they must not be the only driver.'
            )
    zero_profile_lswt_observer_mode = normalize_mainline_lswt_observer_mode(manifest.get(
        'zero_profile_lswt_observer_mode',
        zero_profile_lswt_observer_mode,
    ))
    zero_profile_spinup_days_matrix = manifest.get(
        'zero_profile_spinup_days_matrix',
        zero_profile_spinup_days_matrix,
    )
    zero_profile_spinup_days = _parse_zero_profile_spinup_days_matrix(
        zero_profile_spinup_days_matrix
    )
    zero_profile_lswt_observer_strength = float(manifest.get(
        'zero_profile_lswt_observer_strength',
        zero_profile_lswt_observer_strength,
    ))
    if zero_profile_lswt_observer_strength < 0.0:
        raise ValueError('zero_profile_lswt_observer_strength must be non-negative.')
    zero_profile_lswt_observer_decay_depth_m = float(manifest.get(
        'zero_profile_lswt_observer_decay_depth_m',
        zero_profile_lswt_observer_decay_depth_m,
    ))
    if zero_profile_lswt_observer_decay_depth_m <= 0.0:
        raise ValueError('zero_profile_lswt_observer_decay_depth_m must be positive.')
    zero_profile_lswt_observer_max_increment_c = float(manifest.get(
        'zero_profile_lswt_observer_max_increment_c',
        zero_profile_lswt_observer_max_increment_c,
    ))
    if zero_profile_lswt_observer_max_increment_c < 0.0:
        raise ValueError('zero_profile_lswt_observer_max_increment_c must be non-negative.')
    zero_profile_lswt_observer_deep_update_fraction = float(manifest.get(
        'zero_profile_lswt_observer_deep_update_fraction',
        zero_profile_lswt_observer_deep_update_fraction,
    ))
    if not (0.0 <= zero_profile_lswt_observer_deep_update_fraction <= 1.0):
        raise ValueError('zero_profile_lswt_observer_deep_update_fraction must be between 0 and 1.')
    zero_profile_lswt_observer_heat_content_limit_c = float(manifest.get(
        'zero_profile_lswt_observer_heat_content_limit_c',
        zero_profile_lswt_observer_heat_content_limit_c,
    ))
    if zero_profile_lswt_observer_heat_content_limit_c < 0.0:
        raise ValueError('zero_profile_lswt_observer_heat_content_limit_c must be non-negative.')
    zero_profile_lswt_observer_min_quality = float(manifest.get(
        'zero_profile_lswt_observer_min_quality',
        zero_profile_lswt_observer_min_quality,
    ))
    if not (0.0 <= zero_profile_lswt_observer_min_quality <= 1.0):
        raise ValueError('zero_profile_lswt_observer_min_quality must be between 0 and 1.')
    kd_saturation_threshold = float(manifest.get('kd_saturation_threshold', kd_saturation_threshold))
    if kd_saturation_threshold <= 0.0:
        raise ValueError('kd_saturation_threshold must be positive.')
    kd_saturation_penalty_weight = float(manifest.get(
        'kd_saturation_penalty_weight',
        kd_saturation_penalty_weight,
    ))
    if kd_saturation_penalty_weight < 0.0:
        raise ValueError('kd_saturation_penalty_weight must be non-negative.')
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
    if eval_every_epochs is not None and eval_every_epochs < 0:
        raise ValueError('eval_every_epochs must be non-negative when provided; use 0 to disable train-time eval.')
    default_eval_interval = 50
    eval_interval = int(default_eval_interval if eval_every_epochs is None else eval_every_epochs)
    if full_eval_every_epochs is None:
        full_eval_every_epochs = manifest.get('full_eval_every_epochs', 60)
    full_eval_every_epochs = None if full_eval_every_epochs is None else int(full_eval_every_epochs)
    if full_eval_every_epochs is not None and full_eval_every_epochs < 0:
        raise ValueError('full_eval_every_epochs must be non-negative when provided; use 0 to disable full eval.')
    full_eval_interval = int(0 if full_eval_every_epochs is None else full_eval_every_epochs)
    full_eval_start_epoch = int(manifest.get('full_eval_start_epoch', full_eval_start_epoch or 0))
    if full_eval_start_epoch < 0:
        raise ValueError('full_eval_start_epoch must be non-negative.')
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
    training_amp = _normalize_training_amp(manifest.get('training_amp', training_amp))
    training_history_detail_every_epochs = int(manifest.get(
        'training_history_detail_every_epochs',
        training_history_detail_every_epochs,
    ) or DEFAULT_TRAINING_HISTORY_DETAIL_EVERY_EPOCHS)
    if training_history_detail_every_epochs < 1:
        raise ValueError('training_history_detail_every_epochs must be positive.')
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
    gpu_batch_autotune = _normalize_on_off(
        manifest.get('gpu_batch_autotune', gpu_batch_autotune),
        name='gpu_batch_autotune',
    )
    gpu_batch_autotune_target_batch_size = int(manifest.get(
        'gpu_batch_autotune_target_batch_size',
        gpu_batch_autotune_target_batch_size,
    ))
    if gpu_batch_autotune_target_batch_size < 1:
        raise ValueError('gpu_batch_autotune_target_batch_size must be positive.')
    gpu_batch_autotune_resolved = _resolve_gpu_batch_autotune(
        gpu_batch_autotune=gpu_batch_autotune,
        gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
        transition_batch_size=transition_batch_size,
        segment_rollout_batch_size=segment_rollout_batch_size,
        rolling_horizon_batch_size=rolling_horizon_batch_size,
        unlabeled_heat_closure_batch_size=unlabeled_heat_closure_batch_size,
        cross_lake_batch_mode=cross_lake_batch_mode,
        cross_lake_batch_size=cross_lake_batch_size,
    )
    transition_batch_size = gpu_batch_autotune_resolved['transition_batch_size']
    segment_rollout_batch_size = gpu_batch_autotune_resolved['segment_rollout_batch_size']
    rolling_horizon_batch_size = gpu_batch_autotune_resolved['rolling_horizon_batch_size']
    unlabeled_heat_closure_batch_size = gpu_batch_autotune_resolved['unlabeled_heat_closure_batch_size']
    cross_lake_batch_mode = gpu_batch_autotune_resolved['cross_lake_batch_mode']
    cross_lake_batch_size = gpu_batch_autotune_resolved['cross_lake_batch_size']
    gpu_batch_autotune_applied = gpu_batch_autotune_resolved['gpu_batch_autotune_applied']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    _apply_torch_runtime_config(
        device=device,
        torch_tf32=torch_tf32,
        torch_matmul_precision=torch_matmul_precision,
    )
    training_amp_runtime_enabled = _training_amp_enabled(device, training_amp)
    if training_amp != 'off' and not training_amp_runtime_enabled:
        print(
            f"training_amp={training_amp} requested but disabled because the active device is {device}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lakes = [
        prepare_lake_state_data(
            lake_config,
            split_mode=manifest.get('split_mode', split_mode),
            task_mode=task_mode,
            data_fill_mode=data_fill_mode,
            profile_loss_target_mode=profile_loss_target_mode,
            depth_points=int(manifest.get('depth_points', depth_points)),
            max_rollout_days=manifest_max_rollout_days,
            segment_rollout_max_days=segment_rollout_max_days,
            unlabeled_heat_closure_window_days=unlabeled_heat_closure_window_days,
            unlabeled_heat_closure_horizons=unlabeled_heat_closure_horizons,
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
        if unlabeled_heat_closure_weight > 0.0:
            _unlabeled_heat_closure_window_step_cache(
                lake,
                open_water_only=unlabeled_heat_closure_open_water_only,
                lst_qc_min=unlabeled_heat_closure_lst_qc_min,
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
                sampling_mode=profile_sampling_mode,
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
    cross_lake_transition_batches = (
        _build_cross_lake_transition_batches(
            train_lakes,
            pair_key=supervision_pair_key,
            transition_batch_size=transition_batch_size,
            cross_lake_batch_size=cross_lake_batch_size,
        )
        if cross_lake_batch_mode == 'on' and transition_batch_mode == 'on'
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
        zero_profile_init_net_components=zero_profile_thermal_basis_components,
        zero_profile_init_net_hidden_dim=zero_profile_init_net_hidden_dim,
        zero_profile_init_net_init_spread=zero_profile_init_net_init_spread,
        zero_profile_init_net_coeff_limit_sigma=zero_profile_init_net_coeff_limit_sigma,
        zero_profile_init_net_delta_limit_c=zero_profile_init_net_delta_limit_c,
        daily_memory_components=zero_profile_thermal_basis_components,
        daily_memory_hidden_dim=daily_memory_hidden_dim,
        daily_memory_init_spread=daily_memory_init_spread,
        daily_memory_coeff_limit_sigma=daily_memory_coeff_limit_sigma,
        multitask_auxiliary_hidden_dim=multitask_auxiliary_hidden_dim,
    ).to(device)
    if hasattr(model, 'set_training_amp_mode'):
        model.set_training_amp_mode(training_amp if training_amp_runtime_enabled else 'off')
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

    checkpoint_thermal_basis = None
    if checkpoint_path:
        checkpoint_thermal_basis = checkpoint.get('zero_profile_thermal_basis') if isinstance(checkpoint, dict) else None
    zero_profile_thermal_basis, zero_profile_thermal_basis_source = _resolve_zero_profile_thermal_basis(
        checkpoint_thermal_basis=checkpoint_thermal_basis,
        train_lakes=train_lakes,
        zero_profile_initializer=zero_profile_initializer,
        zero_profile_init_net_loss_weight=zero_profile_init_net_loss_weight,
        zero_profile_thermal_basis_components=zero_profile_thermal_basis_components,
        zero_profile_thermal_basis_grid_points=zero_profile_thermal_basis_grid_points,
        zero_profile_thermal_basis_balance_mode=zero_profile_thermal_basis_balance_mode,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
        thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
        refit_zero_profile_thermal_basis=refit_zero_profile_thermal_basis,
    )
    model.zero_profile_thermal_basis = zero_profile_thermal_basis
    zero_profile_thermal_basis_profile_count = int(
        (zero_profile_thermal_basis or {}).get('profile_count', 0) or 0
    )
    zero_profile_thermal_basis_source_lake_count = int(
        (zero_profile_thermal_basis or {}).get('source_lake_count', 0) or 0
    )
    zero_profile_thermal_basis_balance_mode_effective = str(
        (zero_profile_thermal_basis or {}).get(
            'basis_balance_mode',
            zero_profile_thermal_basis_balance_mode,
        )
        or zero_profile_thermal_basis_balance_mode
    )

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
                'val_segment_rollout_sequences': len(lake['segment_rollout_sequences']['val']),
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
            'profile_loss_target_mode': profile_loss_target_mode,
            'profile_sampling_mode': profile_sampling_mode,
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
            'full_eval_start_epoch': int(full_eval_start_epoch),
            'profile_runtime': bool(profile_runtime),
            'profile_gpu': bool(profile_gpu),
            'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
            'torch_tf32': torch_tf32,
            'torch_matmul_precision': torch_matmul_precision,
            **_training_runtime_config_fields(
                training_amp=training_amp,
                training_amp_enabled=training_amp_runtime_enabled,
                training_history_detail_every_epochs=training_history_detail_every_epochs,
            ),
            'transition_batch_size': int(transition_batch_size),
            'segment_rollout_batch_size': int(segment_rollout_batch_size),
            'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
            'train_diagnostic_mode': train_diagnostic_mode,
            'export_after_training': export_after_training,
            'export_max_depth_m': export_max_depth_m,
            'cross_lake_batch_mode': cross_lake_batch_mode,
            'cross_lake_batch_size': int(cross_lake_batch_size),
            **_gpu_batch_autotune_config_fields(
                gpu_batch_autotune=gpu_batch_autotune,
                gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
                gpu_batch_autotune_applied=gpu_batch_autotune_applied,
            ),
            'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
            'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
            'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
            'segment_rollout_max_days': int(segment_rollout_max_days),
            'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
            'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
            'export_style_validation': export_style_validation,
            'export_style_validation_max_lakes': int(export_style_validation_max_lakes),
            'full_eval_point_diagnostics': full_eval_point_diagnostics,
            'zero_profile_export_validation': zero_profile_export_validation,
            'zero_profile_export_validation_max_lakes': int(zero_profile_export_validation_max_lakes),
            'zero_profile_initializer': zero_profile_initializer,
            'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
            'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
            'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
            'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
            'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
            'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
            'zero_profile_thermal_basis_source_lake_count': int(
                zero_profile_thermal_basis_source_lake_count
            ),
            'zero_profile_init_net_loss_weight': float(zero_profile_init_net_loss_weight),
            'zero_profile_init_net_profile_loss_mode': 'rmse',
            'zero_profile_init_net_start_epoch': int(zero_profile_init_net_start_epoch),
            'zero_profile_init_net_ramp_epochs': int(zero_profile_init_net_ramp_epochs),
            'zero_profile_init_net_samples_per_lake': int(zero_profile_init_net_samples_per_lake),
            'zero_profile_init_net_regularization_weight': float(
                zero_profile_init_net_regularization_weight
            ),
            'zero_profile_init_net_hidden_dim': int(zero_profile_init_net_hidden_dim),
            'zero_profile_init_net_init_spread': float(zero_profile_init_net_init_spread),
            'zero_profile_init_net_coeff_limit_sigma': float(
                zero_profile_init_net_coeff_limit_sigma
            ),
            'zero_profile_init_net_delta_limit_c': float(zero_profile_init_net_delta_limit_c),
            'zero_profile_init_net_training_spinup_days': int(
                zero_profile_init_net_training_spinup_days
            ),
            'zero_profile_init_net_physics_weight': float(zero_profile_init_net_physics_weight),
            'zero_profile_init_net_rollout_weight': float(zero_profile_init_net_rollout_weight),
            'zero_profile_init_net_rollout_max_days': int(zero_profile_init_net_rollout_max_days),
            'zero_profile_init_net_rollout_targets': int(zero_profile_init_net_rollout_targets),
            'zero_profile_lswt_observer_mode': zero_profile_lswt_observer_mode,
            'zero_profile_spinup_days_matrix': ','.join(
                str(int(day)) for day in zero_profile_spinup_days
            ),
            'zero_profile_lswt_observer_strength': float(zero_profile_lswt_observer_strength),
            'zero_profile_lswt_observer_decay_depth_m': float(
                zero_profile_lswt_observer_decay_depth_m
            ),
            'zero_profile_lswt_observer_max_increment_c': float(
                zero_profile_lswt_observer_max_increment_c
            ),
            'zero_profile_lswt_observer_deep_update_fraction': float(
                zero_profile_lswt_observer_deep_update_fraction
            ),
            'zero_profile_lswt_observer_heat_content_limit_c': float(
                zero_profile_lswt_observer_heat_content_limit_c
            ),
            'zero_profile_lswt_observer_min_quality': float(
                zero_profile_lswt_observer_min_quality
            ),
            'kd_saturation_threshold': float(kd_saturation_threshold),
            'kd_saturation_penalty_weight': float(kd_saturation_penalty_weight),
            'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
            'export_only': True,
            'checkpoint_path': str(loaded_checkpoint_path),
        }
        split_summary_payload['_config'].update(_multitask_auxiliary_config_fields(
            weight=multitask_auxiliary_weight,
            requested_weight=multitask_auxiliary_weight_requested,
            heat_weight=multitask_auxiliary_heat_weight,
            thermocline_weight=multitask_auxiliary_thermocline_weight,
            mld_weight=multitask_auxiliary_mld_weight,
            stability_weight=multitask_auxiliary_stability_weight,
            surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
            eof_weight=multitask_auxiliary_eof_weight,
            hidden_dim=multitask_auxiliary_hidden_dim,
        ))
        split_summary_payload['_config'].update(_daily_memory_config_fields(
            reconstruction_weight=daily_memory_reconstruction_weight,
            samples_per_lake=daily_memory_samples_per_lake,
            temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
            heat_budget_weight=daily_memory_heat_budget_weight,
            physics_consistency_weight=daily_memory_physics_consistency_weight,
            regularization_weight=daily_memory_regularization_weight,
            coefficient_loss_weight=daily_memory_coefficient_loss_weight,
            start_epoch=daily_memory_start_epoch,
            ramp_epochs=daily_memory_ramp_epochs,
            hidden_dim=daily_memory_hidden_dim,
            init_spread=daily_memory_init_spread,
            coeff_limit_sigma=daily_memory_coeff_limit_sigma,
            prediction_branch=prediction_branch,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        split_summary_payload['_config'].update(_model_mainline_config_fields(
            model_mainline,
            model_mainline_resolved,
            daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
            prediction_branch=prediction_branch,
            physics_rollout_thermal_state_weight=physics_rollout_thermal_state_weight,
            physics_rollout_thermal_state_weight_eff=physics_rollout_thermal_state_weight_eff,
            thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
        ))
        split_summary_payload['_config'].update(thermal_state_profile_fusion_config)
        split_summary_payload['_config'].update(_unlabeled_heat_closure_config_fields(
            weight=unlabeled_heat_closure_weight,
            batch_size=unlabeled_heat_closure_batch_size,
            window_days=unlabeled_heat_closure_window_days,
            horizons=unlabeled_heat_closure_horizons,
            tau_wm2=unlabeled_heat_closure_tau_wm2,
            open_water_only=unlabeled_heat_closure_open_water_only,
            lst_qc_min=unlabeled_heat_closure_lst_qc_min,
            reservoir_mode=unlabeled_heat_closure_reservoir_mode,
            mode=unlabeled_heat_closure_mode,
            state_source=unlabeled_heat_closure_state_source,
            state_spinup_days=unlabeled_heat_closure_spinup_days,
            solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
            solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        split_summary_payload = _prune_removed_mainline_split_summary(split_summary_payload)
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
            if prediction_branch == 'daily_memory':
                export_info = export_daily_memory_reconstruction(
                    model,
                    lake,
                    output_dir,
                    export_max_depth_m=export_max_depth_m,
                    thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
                    thermal_state_profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                    thermal_state_profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                    thermal_state_profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                    thermal_state_profile_fusion_min_depth_fraction=(
                        thermal_state_profile_fusion_min_depth_fraction
                    ),
                    thermal_state_profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                    thermal_state_profile_fusion_coeff_limit_sigma=(
                        thermal_state_profile_fusion_coeff_limit_sigma
                    ),
                )
            else:
                export_info = export_heldout_state_forecast(
                    model,
                    lake,
                    output_dir,
                    task_mode=task_mode,
                    init_mode=init_mode,
                    spinup_days=spinup_days,
                    zero_profile_initializer=zero_profile_initializer,
                    spinup_lswt_observer_mode=(
                        zero_profile_lswt_observer_mode
                        if zero_profile_lswt_observer_mode != 'off'
                        else DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE
                    ),
                    spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                    spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                    rollout_lswt_observer_mode=zero_profile_lswt_observer_mode,
                    lswt_observer_strength=zero_profile_lswt_observer_strength,
                    lswt_observer_decay_depth_m=zero_profile_lswt_observer_decay_depth_m,
                    lswt_observer_max_increment_c=zero_profile_lswt_observer_max_increment_c,
                    lswt_observer_low_rank_deep_update_fraction=(
                        zero_profile_lswt_observer_deep_update_fraction
                    ),
                    lswt_observer_heat_content_limit_c=zero_profile_lswt_observer_heat_content_limit_c,
                    lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                    rollout_start_date=rollout_start_date,
                    export_max_depth_m=export_max_depth_m,
                    hard_density_stability=hard_density_stability_active,
                    hard_density_stability_mode=hard_density_stability_mode,
                    thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
                    thermal_state_profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                    thermal_state_profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                    thermal_state_profile_fusion_min_depth_fraction=(
                        thermal_state_profile_fusion_min_depth_fraction
                    ),
                    thermal_state_profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                    thermal_state_profile_fusion_coeff_limit_sigma=(
                        thermal_state_profile_fusion_coeff_limit_sigma
                    ),
                )
            heldout_exports.append(export_info)
            print(
                f"Held-out reconstruction export for {lake['lake_id']}: "
                f"{export_info['heatmap_path']} | score={export_info['scorecard_status']}"
            )
        result = {
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
            **_training_runtime_config_fields(
                training_amp=training_amp,
                training_amp_enabled=training_amp_runtime_enabled,
                training_history_detail_every_epochs=training_history_detail_every_epochs,
            ),
        }
        result.update(_multitask_auxiliary_config_fields(
            weight=multitask_auxiliary_weight,
            requested_weight=multitask_auxiliary_weight_requested,
            heat_weight=multitask_auxiliary_heat_weight,
            thermocline_weight=multitask_auxiliary_thermocline_weight,
            mld_weight=multitask_auxiliary_mld_weight,
            stability_weight=multitask_auxiliary_stability_weight,
            surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
            eof_weight=multitask_auxiliary_eof_weight,
            hidden_dim=multitask_auxiliary_hidden_dim,
        ))
        result.update(_daily_memory_config_fields(
            reconstruction_weight=daily_memory_reconstruction_weight,
            samples_per_lake=daily_memory_samples_per_lake,
            temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
            heat_budget_weight=daily_memory_heat_budget_weight,
            physics_consistency_weight=daily_memory_physics_consistency_weight,
            regularization_weight=daily_memory_regularization_weight,
            coefficient_loss_weight=daily_memory_coefficient_loss_weight,
            start_epoch=daily_memory_start_epoch,
            ramp_epochs=daily_memory_ramp_epochs,
            hidden_dim=daily_memory_hidden_dim,
            init_spread=daily_memory_init_spread,
            coeff_limit_sigma=daily_memory_coeff_limit_sigma,
            prediction_branch=prediction_branch,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        result.update(_unlabeled_heat_closure_config_fields(
            weight=unlabeled_heat_closure_weight,
            batch_size=unlabeled_heat_closure_batch_size,
            window_days=unlabeled_heat_closure_window_days,
            horizons=unlabeled_heat_closure_horizons,
            tau_wm2=unlabeled_heat_closure_tau_wm2,
            open_water_only=unlabeled_heat_closure_open_water_only,
            lst_qc_min=unlabeled_heat_closure_lst_qc_min,
            reservoir_mode=unlabeled_heat_closure_reservoir_mode,
            mode=unlabeled_heat_closure_mode,
            state_source=unlabeled_heat_closure_state_source,
            state_spinup_days=unlabeled_heat_closure_spinup_days,
            solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
            solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        return _prune_removed_mainline_output_fields(result)

    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    grad_scaler = _make_training_grad_scaler(device, training_amp)
    start_epoch = 0
    history = []
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        resume = torch.load(resume_path, map_location=device)
        state_dict = resume.get('model_state_dict', resume)
        state_dict = _filter_state_forecaster_state_dict_for_load(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        if 'optimizer_state_dict' in resume:
            try:
                optimizer.load_state_dict(resume['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(device)
            except ValueError:
                print(
                    "Skipped optimizer state from resume checkpoint because model "
                    "parameter groups changed; model weights were loaded."
                )
        if grad_scaler.is_enabled() and 'grad_scaler_state_dict' in resume:
            try:
                grad_scaler.load_state_dict(resume['grad_scaler_state_dict'])
            except (RuntimeError, ValueError):
                print("Skipped AMP grad scaler state from resume checkpoint.")
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
        zero_profile_rmse = float(record.get('val_zero_profile_export_mean_rmse', np.nan))
        zero_profile_count = float(record.get('val_zero_profile_export_profile_count', 0.0) or 0.0)
        if np.isfinite(zero_profile_rmse) and zero_profile_count > 0.0:
            return zero_profile_rmse, None
        export_style_rmse = float(record.get('val_export_style_free_roll_mean_rmse', np.nan))
        export_style_count = float(record.get('val_export_style_free_roll_profile_count', 0.0) or 0.0)
        if np.isfinite(export_style_rmse) and export_style_count > 0.0:
            return export_style_rmse, None
        rmse30 = float(record.get('val_rolling_start_rmse_30d', np.nan))
        rmse60 = float(record.get('val_rolling_start_rmse_60d', np.nan))
        count30 = float(record.get('val_rolling_start_count_30d', 0.0) or 0.0)
        count60 = float(record.get('val_rolling_start_count_60d', 0.0) or 0.0)
        if not (np.isfinite(rmse30) and np.isfinite(rmse60)):
            return None, 'skipped: validation zero-profile/export-style and rolling 30d/60d rmse are not finite'
        if count30 <= 0.0 or count60 <= 0.0:
            return None, 'skipped: validation zero-profile/export-style and rolling 30d/60d count is zero'
        return 0.5 * rmse30 + 0.5 * rmse60, None

    def _write_best_val_rolling_metrics(record, score):
        payload = {
            'epoch': int(record['epoch']),
            'score': float(score),
            'score_formula': (
                'preferred val_zero_profile_export_mean_rmse; '
                'fallback val_export_style_free_roll_mean_rmse; '
                'fallback 0.5 * val_rolling_start_rmse_30d + 0.5 * val_rolling_start_rmse_60d'
            ),
            'val_zero_profile_export_mean_rmse': float(record.get('val_zero_profile_export_mean_rmse', np.nan)),
            'val_zero_profile_export_mean_bias': float(record.get('val_zero_profile_export_mean_bias', np.nan)),
            'val_zero_profile_export_profile_count': float(
                record.get('val_zero_profile_export_profile_count', 0.0) or 0.0
            ),
            'val_export_style_free_roll_mean_rmse': float(
                record.get('val_export_style_free_roll_mean_rmse', np.nan)
            ),
            'val_export_style_free_roll_mean_bias': float(
                record.get('val_export_style_free_roll_mean_bias', np.nan)
            ),
            'val_export_style_free_roll_profile_count': float(
                record.get('val_export_style_free_roll_profile_count', 0.0) or 0.0
            ),
            'val_rolling_start_rmse_30d': float(record.get('val_rolling_start_rmse_30d', np.nan)),
            'val_rolling_start_rmse_60d': float(record.get('val_rolling_start_rmse_60d', np.nan)),
            'val_rolling_start_count_30d': float(record.get('val_rolling_start_count_30d', 0.0) or 0.0),
            'val_rolling_start_count_60d': float(record.get('val_rolling_start_count_60d', 0.0) or 0.0),
            'checkpoint_path': str(best_val_rolling_checkpoint_path),
        }
        for split_name in ('val', 'heldout'):
            for family in ('rolling_start',):
                for horizon in (30, 60):
                    for band in DEPTH_RMSE_BANDS:
                        rmse_key = f'{split_name}_{family}_rmse_{band}_{horizon}d'
                        count_key = f'{split_name}_{family}_count_{band}_{horizon}d'
                        payload[rmse_key] = float(record.get(rmse_key, np.nan))
                        payload[count_key] = float(record.get(count_key, 0.0) or 0.0)
        for split_name in ('train', 'val', 'heldout'):
            for band in DEPTH_RMSE_BANDS:
                rmse_key = f'{split_name}_mean_rmse_{band}'
                count_key = f'{split_name}_count_{band}'
                payload[rmse_key] = float(record.get(rmse_key, np.nan))
                payload[count_key] = float(record.get(count_key, 0.0) or 0.0)
        best_val_rolling_metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return payload

    def _save_training_checkpoint(epoch_value, *, suffix=None):
        path = output_dir / (suffix or f'global_state_forecaster_epoch{int(epoch_value):04d}.pt')
        checkpoint_payload = {
                'architecture': 'MultiLakeStateForecaster',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'grad_scaler_state_dict': grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                'epoch': int(epoch_value),
                'zero_profile_thermal_basis': zero_profile_thermal_basis,
                'seed': None if seed is None else int(seed),
                'training_history': history,
                'manifest': str(Path(manifest_path)),
                'task_mode': task_mode,
                'data_fill_mode': data_fill_mode,
                'profile_supervision_scope': profile_supervision_scope,
                'profile_loss_target_mode': profile_loss_target_mode,
                'profile_sampling_mode': profile_sampling_mode,
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
                'full_eval_start_epoch': int(full_eval_start_epoch),
                'full_eval_point_diagnostics': full_eval_point_diagnostics,
                'zero_profile_initializer': zero_profile_initializer,
                'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
                'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
                'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
                'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
                'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
                'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
                'zero_profile_thermal_basis_source_lake_count': int(
                    zero_profile_thermal_basis_source_lake_count
                ),
                'zero_profile_init_net_loss_weight': float(zero_profile_init_net_loss_weight),
                'zero_profile_init_net_profile_loss_mode': 'rmse',
                'zero_profile_init_net_start_epoch': int(zero_profile_init_net_start_epoch),
                'zero_profile_init_net_ramp_epochs': int(zero_profile_init_net_ramp_epochs),
                'zero_profile_init_net_samples_per_lake': int(zero_profile_init_net_samples_per_lake),
                'zero_profile_init_net_regularization_weight': float(
                    zero_profile_init_net_regularization_weight
                ),
                'zero_profile_init_net_hidden_dim': int(zero_profile_init_net_hidden_dim),
                'zero_profile_init_net_init_spread': float(zero_profile_init_net_init_spread),
                'zero_profile_init_net_coeff_limit_sigma': float(
                    zero_profile_init_net_coeff_limit_sigma
                ),
                'zero_profile_init_net_delta_limit_c': float(zero_profile_init_net_delta_limit_c),
                'zero_profile_init_net_training_spinup_days': int(
                    zero_profile_init_net_training_spinup_days
                ),
                'zero_profile_init_net_physics_weight': float(zero_profile_init_net_physics_weight),
                'zero_profile_init_net_rollout_weight': float(zero_profile_init_net_rollout_weight),
                'zero_profile_init_net_rollout_max_days': int(zero_profile_init_net_rollout_max_days),
                'zero_profile_init_net_rollout_targets': int(zero_profile_init_net_rollout_targets),
                'checkpoint_every_epochs': int(checkpoint_every_epochs),
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                **_training_runtime_config_fields(
                    training_amp=training_amp,
                    training_amp_enabled=training_amp_runtime_enabled,
                    training_history_detail_every_epochs=training_history_detail_every_epochs,
                ),
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                **_gpu_batch_autotune_config_fields(
                    gpu_batch_autotune=gpu_batch_autotune,
                    gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
                    gpu_batch_autotune_applied=gpu_batch_autotune_applied,
                ),
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
            }
        checkpoint_payload.update(_multitask_auxiliary_config_fields(
            weight=multitask_auxiliary_weight,
            requested_weight=multitask_auxiliary_weight_requested,
            heat_weight=multitask_auxiliary_heat_weight,
            thermocline_weight=multitask_auxiliary_thermocline_weight,
            mld_weight=multitask_auxiliary_mld_weight,
            stability_weight=multitask_auxiliary_stability_weight,
            surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
            eof_weight=multitask_auxiliary_eof_weight,
            hidden_dim=multitask_auxiliary_hidden_dim,
        ))
        checkpoint_payload.update(_daily_memory_config_fields(
            reconstruction_weight=daily_memory_reconstruction_weight,
            samples_per_lake=daily_memory_samples_per_lake,
            temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
            heat_budget_weight=daily_memory_heat_budget_weight,
            physics_consistency_weight=daily_memory_physics_consistency_weight,
            regularization_weight=daily_memory_regularization_weight,
            coefficient_loss_weight=daily_memory_coefficient_loss_weight,
            start_epoch=daily_memory_start_epoch,
            ramp_epochs=daily_memory_ramp_epochs,
            hidden_dim=daily_memory_hidden_dim,
            init_spread=daily_memory_init_spread,
            coeff_limit_sigma=daily_memory_coeff_limit_sigma,
            prediction_branch=prediction_branch,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        checkpoint_payload.update(_model_mainline_config_fields(
            model_mainline,
            model_mainline_resolved,
            daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
            prediction_branch=prediction_branch,
            physics_rollout_thermal_state_weight=physics_rollout_thermal_state_weight,
            physics_rollout_thermal_state_weight_eff=physics_rollout_thermal_state_weight_eff,
            thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
        ))
        checkpoint_payload.update(thermal_state_profile_fusion_config)
        checkpoint_payload.update(_unlabeled_heat_closure_config_fields(
            weight=unlabeled_heat_closure_weight,
            batch_size=unlabeled_heat_closure_batch_size,
            window_days=unlabeled_heat_closure_window_days,
            horizons=unlabeled_heat_closure_horizons,
            tau_wm2=unlabeled_heat_closure_tau_wm2,
            open_water_only=unlabeled_heat_closure_open_water_only,
            lst_qc_min=unlabeled_heat_closure_lst_qc_min,
            reservoir_mode=unlabeled_heat_closure_reservoir_mode,
            mode=unlabeled_heat_closure_mode,
            state_source=unlabeled_heat_closure_state_source,
            state_spinup_days=unlabeled_heat_closure_spinup_days,
            solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
            solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
            no_profile_lst_surface_weight=no_profile_lst_surface_weight,
        ))
        torch.save(_prune_removed_mainline_output_fields(checkpoint_payload), path)
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
        zero_profile_init_net_detail_records = []
        daily_memory_detail_records = []
        unlabeled_heat_closure_detail_records = []
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
        zero_profile_init_net_target_weight = (
            float(zero_profile_init_net_loss_weight)
            if zero_profile_initializer == 'eof_pca_init_net' else 0.0
        )
        zero_profile_init_net_weight_eff = _scheduled_weight(
            epoch,
            zero_profile_init_net_target_weight,
            zero_profile_init_net_start_epoch,
            zero_profile_init_net_ramp_epochs,
        )
        daily_memory_weight_eff = _scheduled_weight(
            epoch,
            daily_memory_reconstruction_weight,
            daily_memory_start_epoch,
            daily_memory_ramp_epochs,
        )
        eval_epoch_allowed = epoch > 0
        should_mini_evaluate = eval_epoch_allowed and eval_interval > 0 and (
            (epoch + 1) % max(1, eval_interval) == 0
            or epoch == int(epochs) - 1
        )
        full_eval_epoch_allowed = (
            full_eval_start_epoch <= 0
            or (epoch + 1) >= int(full_eval_start_epoch)
        )
        should_full_evaluate = eval_epoch_allowed and full_eval_interval > 0 and full_eval_epoch_allowed and (
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
        training_history_detail_enabled = (
            history_diagnostic_enabled
            or training_history_detail_every_epochs <= 1
            or (epoch + 1) % training_history_detail_every_epochs == 0
        )
        collect_training_details = bool(training_history_detail_enabled)
        if cross_lake_batch_mode == 'on':
            transition_start_time = time.perf_counter()
            transition_results = _transition_losses_for_lakes_cross_batch(
                model,
                train_lakes,
                pair_key=supervision_pair_key,
                transition_batch_mode=transition_batch_mode,
                transition_batch_size=transition_batch_size,
                cross_lake_batch_size=cross_lake_batch_size,
                cached_batches=cross_lake_transition_batches,
                profile_huber_delta=profile_huber_delta,
                lst_surface_weight=lst_surface_weight,
                energy_balance_weight=energy_balance_weight,
                residual_regularization_weight=residual_regularization_weight,
                daily_tendency_weight=daily_tendency_weight,
                physical_scale_regularization_weight=physical_scale_regularization_weight,
                physical_scale_smoothness_weight=physical_scale_smoothness_weight,
                kd_prior_regularization_weight=kd_prior_regularization_weight,
                kd_saturation_threshold=kd_saturation_threshold,
                kd_saturation_penalty_weight=kd_saturation_penalty_weight,
                adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                heat_content_transition_weight=heat_content_transition_weight,
                heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                heat_content_transition_effective_max=heat_content_transition_effective_max,
                multitask_auxiliary_weight=multitask_auxiliary_weight,
                multitask_auxiliary_heat_weight=multitask_auxiliary_heat_weight,
                multitask_auxiliary_thermocline_weight=multitask_auxiliary_thermocline_weight,
                multitask_auxiliary_mld_weight=multitask_auxiliary_mld_weight,
                multitask_auxiliary_stability_weight=multitask_auxiliary_stability_weight,
                multitask_auxiliary_surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
                multitask_auxiliary_eof_weight=multitask_auxiliary_eof_weight,
                task_mode=task_mode,
                hard_density_stability=hard_density_stability_active,
                step_diagnostic_mode=step_diagnostic_mode,
                collect_details=collect_training_details,
            )
            transition_seconds += time.perf_counter() - transition_start_time

            sequence_results_by_lake = {}
            if segment_weight_eff > 0.0 and active_segment_days > 0:
                cached_segment_batches = (
                    cross_lake_segment_rollout_epoch_batches[epoch]
                    if epoch < len(cross_lake_segment_rollout_epoch_batches) else None
                )
                selected_sequences_by_lake = (
                    {}
                    if cached_segment_batches is not None else {
                        lake_idx: _segment_rollout_sequences_for_epoch(
                            lake,
                            supervision_sequence_key,
                            active_segment_days,
                            segment_rollout_samples_per_lake,
                            epoch,
                            sampling_mode=profile_sampling_mode,
                        )
                        for lake_idx, lake in enumerate(train_lakes)
                    }
                )
                segment_start_time = time.perf_counter()
                sequence_results_by_lake = _segment_rollout_sequence_losses_for_lakes_cross_batch(
                    model,
                    train_lakes,
                    selected_sequences_by_lake,
                    segment_rollout_batch_mode=segment_rollout_batch_mode,
                    segment_rollout_batch_size=segment_rollout_batch_size,
                    cross_lake_batch_size=cross_lake_batch_size,
                    active_max_days=active_segment_days,
                    cached_batches=cached_segment_batches,
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
                    kd_saturation_threshold=kd_saturation_threshold,
                    kd_saturation_penalty_weight=kd_saturation_penalty_weight,
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
                    collect_details=collect_training_details,
                )
                segment_seconds += time.perf_counter() - segment_start_time

            for lake_idx, lake in enumerate(train_lakes):
                pair_losses, pair_details = transition_results[lake_idx]
                if collect_training_details:
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
                            if collect_training_details:
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
                        if collect_training_details:
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
                if zero_profile_init_net_weight_eff > 0.0:
                    init_losses, init_details = _zero_profile_init_net_training_records(
                        model,
                        lake,
                        split_key='train',
                        epoch=epoch,
                        samples_per_lake=zero_profile_init_net_samples_per_lake,
                        profile_huber_delta=profile_huber_delta,
                        regularization_weight=zero_profile_init_net_regularization_weight,
                        training_spinup_days=zero_profile_init_net_training_spinup_days,
                        physics_weight=zero_profile_init_net_physics_weight,
                        rollout_weight=zero_profile_init_net_rollout_weight,
                        rollout_max_days=zero_profile_init_net_rollout_max_days,
                        rollout_targets=zero_profile_init_net_rollout_targets,
                        profile_sampling_mode=profile_sampling_mode,
                        profile_fusion_mode=thermal_state_profile_fusion_mode,
                        profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                        profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                        profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                        profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                        profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                        profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                        collect_details=collect_training_details,
                    )
                    if collect_training_details:
                        zero_profile_init_net_detail_records.extend(init_details)
                    if init_losses:
                        init_loss = torch.stack(init_losses).mean()
                        lake_loss = lake_loss + float(zero_profile_init_net_weight_eff) * init_loss
                if daily_memory_weight_eff > 0.0:
                    daily_losses, daily_details = _daily_memory_reconstruction_training_records(
                        model,
                        lake,
                        split_key='train',
                        epoch=epoch,
                        samples_per_lake=daily_memory_samples_per_lake,
                        temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
                        heat_budget_weight=daily_memory_heat_budget_weight,
                        physics_consistency_weight=daily_memory_physics_consistency_weight,
                        regularization_weight=daily_memory_regularization_weight,
                        coefficient_loss_weight=daily_memory_coefficient_loss_weight,
                        profile_sampling_mode=profile_sampling_mode,
                        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                        no_profile_lst_surface_open_water_only=unlabeled_heat_closure_open_water_only,
                        no_profile_lst_surface_lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                        profile_fusion_mode=thermal_state_profile_fusion_mode,
                        profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                        profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                        profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                        profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                        profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                        profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                        collect_details=collect_training_details,
                    )
                    if collect_training_details:
                        daily_memory_detail_records.extend(daily_details)
                    if daily_losses:
                        daily_loss = torch.stack(daily_losses).mean()
                        lake_loss = lake_loss + float(daily_memory_weight_eff) * daily_loss
                if unlabeled_heat_closure_weight > 0.0:
                    unlabeled_loss, unlabeled_detail = _unlabeled_heat_closure_training_record(
                        model,
                        lake,
                        epoch=epoch,
                        batch_size=unlabeled_heat_closure_batch_size,
                        weight=unlabeled_heat_closure_weight,
                        window_days=unlabeled_heat_closure_window_days,
                        tau_wm2=unlabeled_heat_closure_tau_wm2,
                        open_water_only=unlabeled_heat_closure_open_water_only,
                        lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                        reservoir_mode=unlabeled_heat_closure_reservoir_mode,
                        mode=unlabeled_heat_closure_mode,
                        state_source=unlabeled_heat_closure_state_source,
                        state_spinup_days=unlabeled_heat_closure_spinup_days,
                        solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
                        solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
                        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                        zero_profile_initializer=zero_profile_initializer,
                        spinup_lswt_observer_mode=zero_profile_lswt_observer_mode,
                        zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                    )
                    if collect_training_details:
                        unlabeled_heat_closure_detail_records.append(unlabeled_detail)
                    lake_loss = lake_loss + unlabeled_loss
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
                    kd_saturation_threshold=kd_saturation_threshold,
                    kd_saturation_penalty_weight=kd_saturation_penalty_weight,
                    adaptive_parameter_regularization_weight=adaptive_parameter_regularization_weight,
                    heat_content_transition_weight=heat_content_transition_weight,
                    heat_content_full_column_min_coverage=heat_content_full_column_min_coverage,
                    heat_content_transition_season_factors=lake['heat_content_transition_season_factors'],
                    heat_content_transition_depth_factor=heat_content_transition_depth_factor,
                    heat_content_transition_effective_max=heat_content_transition_effective_max,
                    multitask_auxiliary_weight=multitask_auxiliary_weight,
                    multitask_auxiliary_heat_weight=multitask_auxiliary_heat_weight,
                    multitask_auxiliary_thermocline_weight=multitask_auxiliary_thermocline_weight,
                    multitask_auxiliary_mld_weight=multitask_auxiliary_mld_weight,
                    multitask_auxiliary_stability_weight=multitask_auxiliary_stability_weight,
                    multitask_auxiliary_surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
                    multitask_auxiliary_eof_weight=multitask_auxiliary_eof_weight,
                    task_mode=task_mode,
                    hard_density_stability=hard_density_stability_active,
                    step_diagnostic_mode=step_diagnostic_mode,
                    collect_details=collect_training_details,
                )
                if collect_training_details:
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
                        sampling_mode=profile_sampling_mode,
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
                            sampling_mode=profile_sampling_mode,
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
                        kd_saturation_threshold=kd_saturation_threshold,
                        kd_saturation_penalty_weight=kd_saturation_penalty_weight,
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
                            if collect_training_details:
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
                        if collect_training_details:
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
                if zero_profile_init_net_weight_eff > 0.0:
                    init_losses, init_details = _zero_profile_init_net_training_records(
                        model,
                        lake,
                        split_key='train',
                        epoch=epoch,
                        samples_per_lake=zero_profile_init_net_samples_per_lake,
                        profile_huber_delta=profile_huber_delta,
                        regularization_weight=zero_profile_init_net_regularization_weight,
                        training_spinup_days=zero_profile_init_net_training_spinup_days,
                        physics_weight=zero_profile_init_net_physics_weight,
                        rollout_weight=zero_profile_init_net_rollout_weight,
                        rollout_max_days=zero_profile_init_net_rollout_max_days,
                        rollout_targets=zero_profile_init_net_rollout_targets,
                        profile_sampling_mode=profile_sampling_mode,
                        profile_fusion_mode=thermal_state_profile_fusion_mode,
                        profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                        profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                        profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                        profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                        profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                        profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                    )
                    if collect_training_details:
                        zero_profile_init_net_detail_records.extend(init_details)
                    if init_losses:
                        init_loss = torch.stack(init_losses).mean()
                        lake_loss = lake_loss + float(zero_profile_init_net_weight_eff) * init_loss
                if daily_memory_weight_eff > 0.0:
                    daily_losses, daily_details = _daily_memory_reconstruction_training_records(
                        model,
                        lake,
                        split_key='train',
                        epoch=epoch,
                        samples_per_lake=daily_memory_samples_per_lake,
                        temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
                        heat_budget_weight=daily_memory_heat_budget_weight,
                        physics_consistency_weight=daily_memory_physics_consistency_weight,
                        regularization_weight=daily_memory_regularization_weight,
                        coefficient_loss_weight=daily_memory_coefficient_loss_weight,
                        profile_sampling_mode=profile_sampling_mode,
                        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                        no_profile_lst_surface_open_water_only=unlabeled_heat_closure_open_water_only,
                        no_profile_lst_surface_lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                        profile_fusion_mode=thermal_state_profile_fusion_mode,
                        profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                        profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                        profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                        profile_fusion_min_depth_fraction=thermal_state_profile_fusion_min_depth_fraction,
                        profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                        profile_fusion_coeff_limit_sigma=thermal_state_profile_fusion_coeff_limit_sigma,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                        collect_details=collect_training_details,
                    )
                    if collect_training_details:
                        daily_memory_detail_records.extend(daily_details)
                    if daily_losses:
                        daily_loss = torch.stack(daily_losses).mean()
                        lake_loss = lake_loss + float(daily_memory_weight_eff) * daily_loss
                if unlabeled_heat_closure_weight > 0.0:
                    unlabeled_loss, unlabeled_detail = _unlabeled_heat_closure_training_record(
                        model,
                        lake,
                        epoch=epoch,
                        batch_size=unlabeled_heat_closure_batch_size,
                        weight=unlabeled_heat_closure_weight,
                        window_days=unlabeled_heat_closure_window_days,
                        tau_wm2=unlabeled_heat_closure_tau_wm2,
                        open_water_only=unlabeled_heat_closure_open_water_only,
                        lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                        reservoir_mode=unlabeled_heat_closure_reservoir_mode,
                        mode=unlabeled_heat_closure_mode,
                        state_source=unlabeled_heat_closure_state_source,
                        state_spinup_days=unlabeled_heat_closure_spinup_days,
                        solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
                        solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
                        no_profile_lst_surface_weight=no_profile_lst_surface_weight,
                        zero_profile_initializer=zero_profile_initializer,
                        spinup_lswt_observer_mode=zero_profile_lswt_observer_mode,
                        zero_profile_lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                        task_mode=task_mode,
                        hard_density_stability=hard_density_stability_active,
                        step_diagnostic_mode=step_diagnostic_mode,
                    )
                    if collect_training_details:
                        unlabeled_heat_closure_detail_records.append(unlabeled_detail)
                    lake_loss = lake_loss + unlabeled_loss
                lake_losses.append(lake_loss)
        total_loss = torch.stack(lake_losses).mean()
        optimizer_step_skipped_nonfinite = False
        gradient_norm_value = np.nan
        gradient_norm_tensor = None
        if not bool(torch.isfinite(total_loss.detach()).cpu().item()):
            optimizer_step_skipped_nonfinite = True
            optimizer.zero_grad(set_to_none=True)
            print(
                f"Epoch {epoch:4d} | skipped optimizer step: non-finite total_loss",
                flush=True,
            )
        elif grad_scaler.is_enabled():
            grad_scaler.scale(total_loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_norm, gradients_finite = _clip_grad_norm_finite(model.parameters(), 1.0)
            gradient_norm_tensor = grad_norm.detach().float()
            if gradients_finite:
                grad_scaler.step(optimizer)
            else:
                optimizer_step_skipped_nonfinite = True
                optimizer.zero_grad(set_to_none=True)
                gradient_norm_value = _scalar_tensor_to_float(gradient_norm_tensor)
                print(
                    f"Epoch {epoch:4d} | skipped optimizer step: non-finite gradient_norm={gradient_norm_value}",
                    flush=True,
                )
            grad_scaler.update()
        else:
            total_loss.backward()
            grad_norm, gradients_finite = _clip_grad_norm_finite(model.parameters(), 1.0)
            gradient_norm_tensor = grad_norm.detach().float()
            if gradients_finite:
                optimizer.step()
            else:
                optimizer_step_skipped_nonfinite = True
                optimizer.zero_grad(set_to_none=True)
                gradient_norm_value = _scalar_tensor_to_float(gradient_norm_tensor)
                print(
                    f"Epoch {epoch:4d} | skipped optimizer step: non-finite gradient_norm={gradient_norm_value}",
                    flush=True,
                )

        eval_epoch_allowed = epoch > 0
        should_mini_evaluate = eval_epoch_allowed and eval_interval > 0 and (
            (epoch + 1) % max(1, eval_interval) == 0
            or epoch == int(epochs) - 1
        )
        full_eval_epoch_allowed = (
            full_eval_start_epoch <= 0
            or (epoch + 1) >= int(full_eval_start_epoch)
        )
        should_full_evaluate = eval_epoch_allowed and full_eval_interval > 0 and full_eval_epoch_allowed and (
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
        training_history_detail_enabled = (
            history_diagnostic_enabled
            or training_history_detail_every_epochs <= 1
            or (epoch + 1) % training_history_detail_every_epochs == 0
        )
        if not training_history_detail_enabled:
            detail_records = []
            segment_detail_records = []
            zero_profile_init_net_detail_records = []
            daily_memory_detail_records = []
            unlabeled_heat_closure_detail_records = []
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
            train_depth_rmse = {
                lake['lake_id']: evaluate_lake_pair_depth_rmse(
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
            val_depth_rmse = {
                lake['lake_id']: evaluate_lake_pair_depth_rmse(
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
            heldout_depth_rmse = {
                lake['lake_id']: evaluate_lake_pair_depth_rmse(
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
            val_export_style_free_roll = {}
            export_style_lakes = []
            zero_profile_val_free_roll = {}
            zero_profile_heldout_free_roll = {}
            zero_profile_export_lakes = []
            zero_profile_spinup_matrix_free_roll = {}
            if should_full_evaluate:
                full_eval_point_dir = (
                    output_dir / 'full_eval_point_diagnostics'
                    if full_eval_point_diagnostics == 'on'
                    else None
                )
                export_style_lakes = list(train_lakes)
                if export_style_validation_max_lakes > 0:
                    export_style_lakes = export_style_lakes[:int(export_style_validation_max_lakes)]
                if export_style_validation == 'on' and export_style_lakes:
                    val_export_style_free_roll = evaluate_lakes_free_rolls(
                        model,
                        export_style_lakes,
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
                        diagnostic_output_dir=full_eval_point_dir,
                        diagnostic_split_label='val_export_style',
                        diagnostic_epoch=epoch,
                        diagnostic_rollout_mode='free',
                    )
                zero_profile_validation_ids = {
                    str(value)
                    for value in manifest.get('val_lake_ids', ())
                }
                zero_profile_export_lakes = [
                    lake for lake in train_lakes
                    if lake['lake_id'] in zero_profile_validation_ids
                ]
                if not zero_profile_export_lakes:
                    zero_profile_export_lakes = list(train_lakes)
                if zero_profile_export_validation_max_lakes > 0:
                    zero_profile_export_lakes = zero_profile_export_lakes[
                        :int(zero_profile_export_validation_max_lakes)
                    ]
                if zero_profile_export_validation == 'on' and zero_profile_export_lakes:
                    zero_profile_init_mode = (
                        'zero_profile_low_dof'
                        if zero_profile_initializer == 'low_dof'
                        else 'prior_spinup'
                    )
                    zero_profile_spinup_observer_mode = (
                        DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE
                        if zero_profile_lswt_observer_mode == 'off'
                        else zero_profile_lswt_observer_mode
                    )
                    zero_profile_val_free_roll = evaluate_lakes_free_rolls(
                        model,
                        zero_profile_export_lakes,
                        task_mode=task_mode,
                        horizons=(1, 3, 7, 14, 30, 60),
                        init_mode=zero_profile_init_mode,
                        spinup_days=spinup_days,
                        zero_profile_initializer=zero_profile_initializer,
                        spinup_lswt_observer_mode=zero_profile_spinup_observer_mode,
                        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                        rollout_lswt_observer_mode=zero_profile_lswt_observer_mode,
                        lswt_observer_strength=zero_profile_lswt_observer_strength,
                        lswt_observer_decay_depth_m=zero_profile_lswt_observer_decay_depth_m,
                        lswt_observer_max_increment_c=zero_profile_lswt_observer_max_increment_c,
                        lswt_observer_low_rank_deep_update_fraction=(
                            zero_profile_lswt_observer_deep_update_fraction
                        ),
                        lswt_observer_heat_content_limit_c=(
                            zero_profile_lswt_observer_heat_content_limit_c
                        ),
                        lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                        rollout_start_date=rollout_start_date,
                        hard_density_stability=hard_density_stability_active,
                        batch_size=rolling_horizon_batch_size,
                        diagnostic_output_dir=full_eval_point_dir,
                        diagnostic_split_label='val_zero_profile_export',
                        diagnostic_epoch=epoch,
                        diagnostic_rollout_mode='zero_profile_free',
                    )
                    zero_profile_heldout_free_roll = evaluate_heldout_free_rolls(
                        model,
                        heldout_lakes,
                        task_mode=task_mode,
                        horizons=(1, 3, 7, 14, 30, 60),
                        init_mode=zero_profile_init_mode,
                        spinup_days=spinup_days,
                        zero_profile_initializer=zero_profile_initializer,
                        spinup_lswt_observer_mode=zero_profile_spinup_observer_mode,
                        spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                        spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                        spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                        rollout_lswt_observer_mode=zero_profile_lswt_observer_mode,
                        lswt_observer_strength=zero_profile_lswt_observer_strength,
                        lswt_observer_decay_depth_m=zero_profile_lswt_observer_decay_depth_m,
                        lswt_observer_max_increment_c=zero_profile_lswt_observer_max_increment_c,
                        lswt_observer_low_rank_deep_update_fraction=(
                            zero_profile_lswt_observer_deep_update_fraction
                        ),
                        lswt_observer_heat_content_limit_c=(
                            zero_profile_lswt_observer_heat_content_limit_c
                        ),
                        lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                        rollout_start_date=rollout_start_date,
                        hard_density_stability=hard_density_stability_active,
                        batch_size=rolling_horizon_batch_size,
                        diagnostic_output_dir=full_eval_point_dir,
                        diagnostic_split_label='heldout_zero_profile_export_diagnostic_only',
                        diagnostic_epoch=epoch,
                        diagnostic_rollout_mode='zero_profile_free',
                    )
                    for matrix_spinup_days in zero_profile_spinup_days:
                        if int(matrix_spinup_days) == int(spinup_days):
                            continue
                        zero_profile_spinup_matrix_free_roll[int(matrix_spinup_days)] = evaluate_lakes_free_rolls(
                            model,
                            zero_profile_export_lakes,
                            task_mode=task_mode,
                            horizons=(1, 3, 7, 14, 30, 60),
                            init_mode=zero_profile_init_mode,
                            spinup_days=int(matrix_spinup_days),
                            zero_profile_initializer=zero_profile_initializer,
                            spinup_lswt_observer_mode=zero_profile_spinup_observer_mode,
                            spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                            spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                            spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                            rollout_lswt_observer_mode=zero_profile_lswt_observer_mode,
                            lswt_observer_strength=zero_profile_lswt_observer_strength,
                            lswt_observer_decay_depth_m=zero_profile_lswt_observer_decay_depth_m,
                            lswt_observer_max_increment_c=zero_profile_lswt_observer_max_increment_c,
                            lswt_observer_low_rank_deep_update_fraction=(
                                zero_profile_lswt_observer_deep_update_fraction
                            ),
                            lswt_observer_heat_content_limit_c=(
                                zero_profile_lswt_observer_heat_content_limit_c
                            ),
                            lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                            rollout_start_date=rollout_start_date,
                            hard_density_stability=hard_density_stability_active,
                            batch_size=rolling_horizon_batch_size,
                        )
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
                    diagnostic_output_dir=full_eval_point_dir,
                    diagnostic_split_label='heldout_free_roll_diagnostic_only',
                    diagnostic_epoch=epoch,
                    diagnostic_rollout_mode='free',
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
            val_export_style_free_roll_records = list(val_export_style_free_roll.values())
            if gradient_norm_tensor is not None and not optimizer_step_skipped_nonfinite:
                gradient_norm_value = _scalar_tensor_to_float(gradient_norm_tensor)
            record = {
                'epoch': epoch,
                'loss': float(total_loss.detach().cpu()),
                'profile_supervision_scope': profile_supervision_scope,
                'profile_loss_target_mode': profile_loss_target_mode,
                'profile_sampling_mode': profile_sampling_mode,
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
                'residual_abs_mean_c': detail_mean(detail_records, 'residual_abs_mean_c'),
                'residual_surface_abs_mean_c': detail_mean(detail_records, 'residual_surface_abs_mean_c'),
                'residual_deep_abs_mean_c': detail_mean(detail_records, 'residual_deep_abs_mean_c'),
                'physical_scale_reg_loss': _mean_detail(detail_records, 'physical_scale_reg_loss'),
                'physical_scale_smooth_loss': _mean_detail(detail_records, 'physical_scale_smooth_loss'),
                'kd_prior_regularization_loss': _mean_detail(detail_records, 'kd_prior_regularization_loss'),
                'kd_prior_regularization_weighted_loss': _mean_detail(detail_records, 'kd_prior_regularization_weighted_loss'),
                'kd_saturation_penalty_loss': _mean_detail(detail_records, 'kd_saturation_penalty_loss'),
                'kd_saturation_penalty_weighted_loss': _mean_detail(detail_records, 'kd_saturation_penalty_weighted_loss'),
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
                'nn_kd_multiplier_p50': (
                    _quantile_detail(detail_records, 'nn_kd_multiplier_mean', 0.50)
                    if history_diagnostic_enabled else np.nan
                ),
                'nn_kd_multiplier_p95': (
                    _quantile_detail(detail_records, 'nn_kd_multiplier_mean', 0.95)
                    if history_diagnostic_enabled else np.nan
                ),
                'nn_kd_multiplier_saturation_fraction': (
                    _fraction_detail_ge(detail_records, 'nn_kd_multiplier_mean', kd_saturation_threshold)
                    if history_diagnostic_enabled else np.nan
                ),
                'kd_saturation_threshold': float(kd_saturation_threshold),
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
                'segment_rollout_kd_saturation_penalty_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_saturation_penalty_loss'),
                'segment_rollout_kd_saturation_penalty_weighted_loss': _mean_detail(segment_detail_records, 'segment_rollout_kd_saturation_penalty_weighted_loss'),
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
                'kd_saturation_penalty_weight': float(kd_saturation_penalty_weight),
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
                'full_eval_start_epoch': int(full_eval_start_epoch),
                'eval_mode': eval_mode,
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'history_diagnostic_enabled': bool(history_diagnostic_enabled),
                'training_history_detail_enabled': bool(training_history_detail_enabled),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                **_training_runtime_config_fields(
                    training_amp=training_amp,
                    training_amp_enabled=training_amp_runtime_enabled,
                    training_history_detail_every_epochs=training_history_detail_every_epochs,
                ),
                'transition_batch_size': int(transition_batch_size),
                'segment_rollout_batch_size': int(segment_rollout_batch_size),
                'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
                'train_diagnostic_mode': train_diagnostic_mode,
                'export_after_training': export_after_training,
                'export_max_depth_m': export_max_depth_m,
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                **_gpu_batch_autotune_config_fields(
                    gpu_batch_autotune=gpu_batch_autotune,
                    gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
                    gpu_batch_autotune_applied=gpu_batch_autotune_applied,
                ),
                'rolling_horizon_eval_max_starts': int(rolling_horizon_eval_max_starts),
                'export_style_validation': export_style_validation,
                'export_style_validation_max_lakes': int(export_style_validation_max_lakes),
                'full_eval_point_diagnostics': full_eval_point_diagnostics,
                'zero_profile_export_validation': zero_profile_export_validation,
                'zero_profile_export_validation_max_lakes': int(
                    zero_profile_export_validation_max_lakes
                ),
                'zero_profile_initializer': zero_profile_initializer,
                'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
                'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
                'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
                'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
                'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
                'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
                'zero_profile_thermal_basis_source_lake_count': int(
                    zero_profile_thermal_basis_source_lake_count
                ),
                'zero_profile_lswt_observer_mode': zero_profile_lswt_observer_mode,
                'zero_profile_spinup_days_matrix': ','.join(
                    str(int(day)) for day in zero_profile_spinup_days
                ),
                'zero_profile_lswt_observer_strength': float(zero_profile_lswt_observer_strength),
                'zero_profile_lswt_observer_decay_depth_m': float(
                    zero_profile_lswt_observer_decay_depth_m
                ),
                'zero_profile_lswt_observer_max_increment_c': float(
                    zero_profile_lswt_observer_max_increment_c
                ),
                'zero_profile_lswt_observer_deep_update_fraction': float(
                    zero_profile_lswt_observer_deep_update_fraction
                ),
                'zero_profile_lswt_observer_heat_content_limit_c': float(
                    zero_profile_lswt_observer_heat_content_limit_c
                ),
                'zero_profile_lswt_observer_min_quality': float(
                    zero_profile_lswt_observer_min_quality
                ),
                'zero_profile_export_checkpoint_selection_enabled': False,
                'zero_profile_export_heldout_diagnostic_only': True,
                'export_style_checkpoint_selection_enabled': False,
                'val_export_style_free_roll_mean_rmse': _mean_numeric_records(
                    val_export_style_free_roll_records,
                    'rmse',
                ),
                'val_export_style_free_roll_mean_bias': _mean_numeric_records(
                    val_export_style_free_roll_records,
                    'bias',
                ),
                'val_export_style_free_roll_mean_mae': _mean_numeric_records(
                    val_export_style_free_roll_records,
                    'mae',
                ),
                'val_export_style_free_roll_profile_count': _sum_numeric_records(
                    val_export_style_free_roll_records,
                    'n_profiles',
                ),
                'val_export_style_free_roll_mean_rmse_le25m': _mean_numeric_records(
                    val_export_style_free_roll_records,
                    'rmse_le25m',
                ),
                'val_export_style_free_roll_mean_rmse_gt25m': _mean_numeric_records(
                    val_export_style_free_roll_records,
                    'rmse_gt25m',
                ),
                'val_export_style_free_roll_count_le25m': _sum_numeric_records(
                    val_export_style_free_roll_records,
                    'count_le25m',
                ),
                'val_export_style_free_roll_count_gt25m': _sum_numeric_records(
                    val_export_style_free_roll_records,
                    'count_gt25m',
                ),
                'val_export_style_free_roll_point_diagnostics_count': _sum_numeric_records(
                    val_export_style_free_roll_records,
                    'free_roll_point_diagnostics_count',
                ),
                'val_export_style_free_roll_age_summary_count': _sum_numeric_records(
                    val_export_style_free_roll_records,
                    'free_roll_age_summary_count',
                ),
                'val_export_style_free_roll_point_diagnostics_csvs': _joined_artifact_paths(
                    val_export_style_free_roll_records,
                    'free_roll_point_diagnostics_csv',
                ),
                'val_export_style_free_roll_age_summary_csvs': _joined_artifact_paths(
                    val_export_style_free_roll_records,
                    'free_roll_age_summary_csv',
                ),
                'train_mean_rmse': _nanmean_or_nan(train_rmse.values()) if train_rmse else np.nan,
                'train_mean_rmse_overall': _nanmean_or_nan(train_rmse.values()) if train_rmse else np.nan,
                'train_mean_rmse_le25m': _mean_numeric_records(train_depth_rmse.values(), 'rmse_le25m'),
                'train_mean_rmse_gt25m': _mean_numeric_records(train_depth_rmse.values(), 'rmse_gt25m'),
                'train_count_le25m': _sum_numeric_records(train_depth_rmse.values(), 'count_le25m'),
                'train_count_gt25m': _sum_numeric_records(train_depth_rmse.values(), 'count_gt25m'),
                'val_mean_rmse': _nanmean_or_nan(val_rmse.values()) if val_rmse else np.nan,
                'val_mean_rmse_overall': _nanmean_or_nan(val_rmse.values()) if val_rmse else np.nan,
                'val_mean_rmse_le25m': _mean_numeric_records(val_depth_rmse.values(), 'rmse_le25m'),
                'val_mean_rmse_gt25m': _mean_numeric_records(val_depth_rmse.values(), 'rmse_gt25m'),
                'val_count_le25m': _sum_numeric_records(val_depth_rmse.values(), 'count_le25m'),
                'val_count_gt25m': _sum_numeric_records(val_depth_rmse.values(), 'count_gt25m'),
                'heldout_mean_rmse': _nanmean_or_nan(heldout_rmse.values()) if heldout_rmse else np.nan,
                'heldout_mean_rmse_overall': _nanmean_or_nan(heldout_rmse.values()) if heldout_rmse else np.nan,
                'heldout_mean_rmse_le25m': _mean_numeric_records(heldout_depth_rmse.values(), 'rmse_le25m'),
                'heldout_mean_rmse_gt25m': _mean_numeric_records(heldout_depth_rmse.values(), 'rmse_gt25m'),
                'heldout_count_le25m': _sum_numeric_records(heldout_depth_rmse.values(), 'count_le25m'),
                'heldout_count_gt25m': _sum_numeric_records(heldout_depth_rmse.values(), 'count_gt25m'),
                'heldout_transition_mean_rmse': _nanmean_or_nan(heldout_rmse.values()) if heldout_rmse else np.nan,
                'train_persistence_mean_rmse': _nanmean_or_nan(train_persistence_rmse.values()) if train_persistence_rmse else np.nan,
                'val_persistence_mean_rmse': _nanmean_or_nan(val_persistence_rmse.values()) if val_persistence_rmse else np.nan,
                'heldout_persistence_mean_rmse': _nanmean_or_nan(heldout_persistence_rmse.values()) if heldout_persistence_rmse else np.nan,
                'heldout_free_roll_mean_rmse': _nanmean_or_nan(
                    value.get('rmse', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_free_roll_mean_rmse_le25m': _nanmean_or_nan(
                    value.get('rmse_le25m', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_free_roll_mean_rmse_gt25m': _nanmean_or_nan(
                    value.get('rmse_gt25m', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_free_roll_count_le25m': int(sum(
                    value.get('count_le25m', 0) for value in heldout_free_roll.values()
                )) if heldout_free_roll else 0,
                'heldout_free_roll_count_gt25m': int(sum(
                    value.get('count_gt25m', 0) for value in heldout_free_roll.values()
                )) if heldout_free_roll else 0,
                'heldout_free_roll_mean_bias': _nanmean_or_nan(
                    value.get('bias', np.nan) for value in heldout_free_roll.values()
                ) if heldout_free_roll else np.nan,
                'heldout_free_roll_point_diagnostics_count': _sum_numeric_records(
                    heldout_free_roll.values(),
                    'free_roll_point_diagnostics_count',
                ),
                'heldout_free_roll_age_summary_count': _sum_numeric_records(
                    heldout_free_roll.values(),
                    'free_roll_age_summary_count',
                ),
                'heldout_free_roll_point_diagnostics_csvs': _joined_artifact_paths(
                    heldout_free_roll.values(),
                    'free_roll_point_diagnostics_csv',
                ),
                'heldout_free_roll_age_summary_csvs': _joined_artifact_paths(
                    heldout_free_roll.values(),
                    'free_roll_age_summary_csv',
                ),
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
            _add_zero_profile_export_metrics(
                record,
                'val',
                zero_profile_val_free_roll,
                zero_profile_export_lakes,
            )
            _add_zero_profile_export_metrics(
                record,
                'heldout',
                zero_profile_heldout_free_roll,
                heldout_lakes,
            )
            for matrix_spinup_days, matrix_metrics in zero_profile_spinup_matrix_free_roll.items():
                _add_zero_profile_export_metrics(
                    record,
                    f'val_zp_spinup{int(matrix_spinup_days)}d',
                    matrix_metrics,
                    zero_profile_export_lakes,
                )
            record.update(_zero_profile_init_net_history_fields(
                zero_profile_init_net_detail_records,
                loss_weight=zero_profile_init_net_loss_weight,
                weight_eff=zero_profile_init_net_weight_eff,
                start_epoch=zero_profile_init_net_start_epoch,
                ramp_epochs=zero_profile_init_net_ramp_epochs,
                samples_per_lake=zero_profile_init_net_samples_per_lake,
                regularization_weight=zero_profile_init_net_regularization_weight,
                training_spinup_days=zero_profile_init_net_training_spinup_days,
                physics_weight=zero_profile_init_net_physics_weight,
                rollout_weight=zero_profile_init_net_rollout_weight,
                rollout_max_days=zero_profile_init_net_rollout_max_days,
                rollout_targets=zero_profile_init_net_rollout_targets,
                hidden_dim=zero_profile_init_net_hidden_dim,
                init_spread=zero_profile_init_net_init_spread,
                coeff_limit_sigma=zero_profile_init_net_coeff_limit_sigma,
                delta_limit_c=zero_profile_init_net_delta_limit_c,
            ))
            record.update(_daily_memory_history_fields(
                daily_memory_detail_records,
                reconstruction_weight=daily_memory_reconstruction_weight,
                weight_eff=daily_memory_weight_eff,
                samples_per_lake=daily_memory_samples_per_lake,
                temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
                heat_budget_weight=daily_memory_heat_budget_weight,
                physics_consistency_weight=daily_memory_physics_consistency_weight,
                regularization_weight=daily_memory_regularization_weight,
                coefficient_loss_weight=daily_memory_coefficient_loss_weight,
                start_epoch=daily_memory_start_epoch,
                ramp_epochs=daily_memory_ramp_epochs,
                hidden_dim=daily_memory_hidden_dim,
                init_spread=daily_memory_init_spread,
                coeff_limit_sigma=daily_memory_coeff_limit_sigma,
                prediction_branch=prediction_branch,
            ))
            record.update(thermal_state_profile_fusion_config)
            record.update(_multitask_auxiliary_history_fields(
                detail_records,
                weight=multitask_auxiliary_weight,
                requested_weight=multitask_auxiliary_weight_requested,
                heat_weight=multitask_auxiliary_heat_weight,
                thermocline_weight=multitask_auxiliary_thermocline_weight,
                mld_weight=multitask_auxiliary_mld_weight,
                stability_weight=multitask_auxiliary_stability_weight,
                surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
                eof_weight=multitask_auxiliary_eof_weight,
                hidden_dim=multitask_auxiliary_hidden_dim,
            ))
            record.update(_unlabeled_heat_closure_history_fields(
                unlabeled_heat_closure_detail_records,
                weight=unlabeled_heat_closure_weight,
                batch_size=unlabeled_heat_closure_batch_size,
                window_days=unlabeled_heat_closure_window_days,
                horizons=unlabeled_heat_closure_horizons,
                tau_wm2=unlabeled_heat_closure_tau_wm2,
                open_water_only=unlabeled_heat_closure_open_water_only,
                lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                reservoir_mode=unlabeled_heat_closure_reservoir_mode,
                mode=unlabeled_heat_closure_mode,
                state_source=unlabeled_heat_closure_state_source,
                state_spinup_days=unlabeled_heat_closure_spinup_days,
                solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
                solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
            ))
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
                for band in DEPTH_RMSE_BANDS:
                    band_rmse_key = f'rmse_{band}_{horizon}d'
                    band_count_key = f'count_{band}_{horizon}d'
                    record[f'train_transition_rmse_{band}_{horizon}d'] = _mean_numeric_records(
                        train_horizon,
                        band_rmse_key,
                    )
                    record[f'train_transition_count_{band}_{horizon}d'] = _sum_numeric_records(
                        train_horizon,
                        band_count_key,
                    )
                    record[f'heldout_transition_rmse_{band}_{horizon}d'] = _mean_numeric_records(
                        heldout_transition_horizon,
                        band_rmse_key,
                    )
                    record[f'heldout_transition_count_{band}_{horizon}d'] = _sum_numeric_records(
                        heldout_transition_horizon,
                        band_count_key,
                    )
                    record[f'val_rolling_start_rmse_{band}_{horizon}d'] = _mean_numeric_records(
                        val_rolling_start_records,
                        band_rmse_key,
                    )
                    record[f'val_rolling_start_count_{band}_{horizon}d'] = _sum_numeric_records(
                        val_rolling_start_records,
                        band_count_key,
                    )
                rolling_rmse = _mean_numeric_records(heldout_rolling_start_records, rmse_key)
                rolling_bias = _mean_numeric_records(heldout_rolling_start_records, f'bias_{horizon}d')
                rolling_count = _mean_numeric_records(heldout_rolling_start_records, count_key)
                record[f'heldout_rolling_start_rmse_{horizon}d'] = rolling_rmse
                record[f'heldout_rolling_start_bias_{horizon}d'] = rolling_bias
                record[f'heldout_rolling_start_count_{horizon}d'] = rolling_count
                record[f'heldout_free_roll_rmse_{horizon}d'] = rolling_rmse
                record[f'heldout_free_roll_bias_{horizon}d'] = rolling_bias
                record[f'heldout_free_roll_count_{horizon}d'] = rolling_count
                for band in DEPTH_RMSE_BANDS:
                    band_rmse_key = f'rmse_{band}_{horizon}d'
                    band_count_key = f'count_{band}_{horizon}d'
                    band_rolling_rmse = _mean_numeric_records(heldout_rolling_start_records, band_rmse_key)
                    band_rolling_count = _sum_numeric_records(heldout_rolling_start_records, band_count_key)
                    record[f'heldout_rolling_start_rmse_{band}_{horizon}d'] = band_rolling_rmse
                    record[f'heldout_rolling_start_count_{band}_{horizon}d'] = band_rolling_count
                    record[f'heldout_free_roll_rmse_{band}_{horizon}d'] = band_rolling_rmse
                    record[f'heldout_free_roll_count_{band}_{horizon}d'] = band_rolling_count
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
                for band in DEPTH_RMSE_BANDS:
                    band_rmse_key = f'rmse_{band}_{horizon}d'
                    band_count_key = f'count_{band}_{horizon}d'
                    record[f'heldout_initial_free_roll_rmse_{band}_{horizon}d'] = _nanmean_or_nan(
                        value.get('horizon_metrics', {}).get(band_rmse_key, np.nan)
                        for value in heldout_free_roll.values()
                    ) if heldout_free_roll else np.nan
                    record[f'heldout_initial_free_roll_count_{band}_{horizon}d'] = _sum_numeric_records(
                        (value.get('horizon_metrics', {}) for value in heldout_free_roll.values()),
                        band_count_key,
                    ) if heldout_free_roll else 0
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
                    record[f'{key}_train_rmse_overall'] = value
                    depth_metrics = train_depth_rmse.get(key, {})
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_train_rmse_{band}'] = depth_metrics.get(f'rmse_{band}', np.nan)
                        record[f'{key}_train_count_{band}'] = depth_metrics.get(f'count_{band}', 0)
                for key, value in val_rmse.items():
                    record[f'{key}_val_rmse'] = value
                    record[f'{key}_val_rmse_overall'] = value
                    depth_metrics = val_depth_rmse.get(key, {})
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_val_rmse_{band}'] = depth_metrics.get(f'rmse_{band}', np.nan)
                        record[f'{key}_val_count_{band}'] = depth_metrics.get(f'count_{band}', 0)
                    rolling_metrics = val_rolling_start_horizon.get(key, {})
                    for horizon in (1, 3, 7, 14, 30, 60):
                        record[f'{key}_val_rolling_start_rmse_{horizon}d'] = rolling_metrics.get(f'rmse_{horizon}d', np.nan)
                        record[f'{key}_val_rolling_start_bias_{horizon}d'] = rolling_metrics.get(f'bias_{horizon}d', np.nan)
                        record[f'{key}_val_rolling_start_count_{horizon}d'] = rolling_metrics.get(f'count_{horizon}d', np.nan)
                        for band in DEPTH_RMSE_BANDS:
                            record[f'{key}_val_rolling_start_rmse_{band}_{horizon}d'] = rolling_metrics.get(
                                f'rmse_{band}_{horizon}d',
                                np.nan,
                            )
                            record[f'{key}_val_rolling_start_count_{band}_{horizon}d'] = rolling_metrics.get(
                                f'count_{band}_{horizon}d',
                                0,
                            )
                for key, value in heldout_rmse.items():
                    record[f'{key}_heldout_rmse'] = value
                    record[f'{key}_heldout_rmse_overall'] = value
                    record[f'{key}_heldout_transition_rmse'] = value
                    depth_metrics = heldout_depth_rmse.get(key, {})
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_heldout_rmse_{band}'] = depth_metrics.get(f'rmse_{band}', np.nan)
                        record[f'{key}_heldout_count_{band}'] = depth_metrics.get(f'count_{band}', 0)
                        record[f'{key}_heldout_transition_rmse_{band}'] = depth_metrics.get(f'rmse_{band}', np.nan)
                        record[f'{key}_heldout_transition_count_{band}'] = depth_metrics.get(f'count_{band}', 0)
                for key, value in train_persistence_rmse.items():
                    record[f'{key}_train_persistence_rmse'] = value
                for key, value in val_persistence_rmse.items():
                    record[f'{key}_val_persistence_rmse'] = value
                for key, value in heldout_persistence_rmse.items():
                    record[f'{key}_heldout_persistence_rmse'] = value
            for key, value in heldout_free_roll.items():
                record[f'{key}_heldout_free_roll_rmse'] = value.get('rmse', np.nan)
                record[f'{key}_heldout_free_roll_rmse_overall'] = value.get('rmse', np.nan)
                for band in DEPTH_RMSE_BANDS:
                    record[f'{key}_heldout_free_roll_rmse_{band}'] = value.get(f'rmse_{band}', np.nan)
                    record[f'{key}_heldout_free_roll_count_{band}'] = value.get(f'count_{band}', 0)
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
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_heldout_initial_free_roll_rmse_{band}_{horizon}d'] = initial_metrics.get(
                            f'rmse_{band}_{horizon}d',
                            np.nan,
                        )
                        record[f'{key}_heldout_initial_free_roll_count_{band}_{horizon}d'] = initial_metrics.get(
                            f'count_{band}_{horizon}d',
                            0,
                        )
                        record[f'{key}_heldout_rolling_start_rmse_{band}_{horizon}d'] = rolling_metrics.get(
                            f'rmse_{band}_{horizon}d',
                            np.nan,
                        )
                        record[f'{key}_heldout_rolling_start_count_{band}_{horizon}d'] = rolling_metrics.get(
                            f'count_{band}_{horizon}d',
                            0,
                        )
                        record[f'{key}_heldout_free_roll_rmse_{band}_{horizon}d'] = rolling_metrics.get(
                            f'rmse_{band}_{horizon}d',
                            np.nan,
                        )
                        record[f'{key}_heldout_free_roll_count_{band}_{horizon}d'] = rolling_metrics.get(
                            f'count_{band}_{horizon}d',
                            0,
                        )
            _add_free_roll_bucket_metrics(record, 'heldout', heldout_free_roll, heldout_lakes)
            _add_free_roll_bucket_metrics(
                record,
                'val_export_style',
                val_export_style_free_roll,
                export_style_lakes,
            )
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
            record['gradient_norm'] = float(gradient_norm_value)
            record['optimizer_step_skipped_nonfinite'] = bool(optimizer_step_skipped_nonfinite)
            record = _prune_removed_mainline_output_fields(record)
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
                record = _prune_removed_mainline_output_fields(record)
                history[-1] = record
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
            if gradient_norm_tensor is not None and not optimizer_step_skipped_nonfinite:
                gradient_norm_value = _scalar_tensor_to_float(gradient_norm_tensor)
            record = {
                'epoch': int(epoch),
                'loss': float(total_loss.detach().cpu().item()),
                'profile_supervision_scope': profile_supervision_scope,
                'profile_loss_target_mode': profile_loss_target_mode,
                'profile_sampling_mode': profile_sampling_mode,
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
                'residual_abs_mean_c': _mean_detail(detail_records, 'residual_abs_mean_c'),
                'residual_surface_abs_mean_c': _mean_detail(detail_records, 'residual_surface_abs_mean_c'),
                'residual_deep_abs_mean_c': _mean_detail(detail_records, 'residual_deep_abs_mean_c'),
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
                'nn_kd_multiplier_p50': (
                    _quantile_detail(detail_records, 'nn_kd_multiplier_mean', 0.50)
                    if history_diagnostic_enabled else np.nan
                ),
                'nn_kd_multiplier_p95': (
                    _quantile_detail(detail_records, 'nn_kd_multiplier_mean', 0.95)
                    if history_diagnostic_enabled else np.nan
                ),
                'nn_kd_multiplier_saturation_fraction': (
                    _fraction_detail_ge(detail_records, 'nn_kd_multiplier_mean', kd_saturation_threshold)
                    if history_diagnostic_enabled else np.nan
                ),
                'kd_saturation_threshold': float(kd_saturation_threshold),
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
                'full_eval_start_epoch': int(full_eval_start_epoch),
                'eval_mode': eval_mode,
                'profile_runtime': bool(profile_runtime),
                'profile_gpu': bool(profile_gpu),
                'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
                'history_diagnostic_enabled': bool(history_diagnostic_enabled),
                'training_history_detail_enabled': bool(training_history_detail_enabled),
                'torch_tf32': torch_tf32,
                'torch_matmul_precision': torch_matmul_precision,
                **_training_runtime_config_fields(
                    training_amp=training_amp,
                    training_amp_enabled=training_amp_runtime_enabled,
                    training_history_detail_every_epochs=training_history_detail_every_epochs,
                ),
                'transition_batch_size': int(transition_batch_size),
                'segment_rollout_batch_size': int(segment_rollout_batch_size),
                'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
                'train_diagnostic_mode': train_diagnostic_mode,
                'export_after_training': export_after_training,
                'export_max_depth_m': export_max_depth_m,
                'cross_lake_batch_mode': cross_lake_batch_mode,
                'cross_lake_batch_size': int(cross_lake_batch_size),
                **_gpu_batch_autotune_config_fields(
                    gpu_batch_autotune=gpu_batch_autotune,
                    gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
                    gpu_batch_autotune_applied=gpu_batch_autotune_applied,
                ),
                'rolling_horizon_eval_max_starts': int(rolling_horizon_eval_max_starts),
                'export_style_validation': export_style_validation,
                'export_style_validation_max_lakes': int(export_style_validation_max_lakes),
                'full_eval_point_diagnostics': full_eval_point_diagnostics,
                'zero_profile_export_validation': zero_profile_export_validation,
                'zero_profile_export_validation_max_lakes': int(
                    zero_profile_export_validation_max_lakes
                ),
                'zero_profile_initializer': zero_profile_initializer,
                'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
                'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
                'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
                'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
                'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
                'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
                'zero_profile_thermal_basis_source_lake_count': int(
                    zero_profile_thermal_basis_source_lake_count
                ),
                'zero_profile_lswt_observer_mode': zero_profile_lswt_observer_mode,
                'zero_profile_spinup_days_matrix': ','.join(
                    str(int(day)) for day in zero_profile_spinup_days
                ),
                'zero_profile_lswt_observer_strength': float(zero_profile_lswt_observer_strength),
                'zero_profile_lswt_observer_decay_depth_m': float(
                    zero_profile_lswt_observer_decay_depth_m
                ),
                'zero_profile_lswt_observer_max_increment_c': float(
                    zero_profile_lswt_observer_max_increment_c
                ),
                'zero_profile_lswt_observer_deep_update_fraction': float(
                    zero_profile_lswt_observer_deep_update_fraction
                ),
                'zero_profile_lswt_observer_heat_content_limit_c': float(
                    zero_profile_lswt_observer_heat_content_limit_c
                ),
                'zero_profile_lswt_observer_min_quality': float(
                    zero_profile_lswt_observer_min_quality
                ),
                'zero_profile_export_checkpoint_selection_enabled': False,
                'zero_profile_export_heldout_diagnostic_only': True,
                'export_style_checkpoint_selection_enabled': False,
                'val_export_style_free_roll_mean_rmse': np.nan,
                'val_export_style_free_roll_mean_bias': np.nan,
                'val_export_style_free_roll_mean_mae': np.nan,
                'val_export_style_free_roll_profile_count': 0,
                'val_export_style_free_roll_mean_rmse_le25m': np.nan,
                'val_export_style_free_roll_mean_rmse_gt25m': np.nan,
                'val_export_style_free_roll_count_le25m': 0,
                'val_export_style_free_roll_count_gt25m': 0,
                'val_export_style_free_roll_point_diagnostics_count': 0,
                'val_export_style_free_roll_age_summary_count': 0,
                'val_export_style_free_roll_point_diagnostics_csvs': '',
                'val_export_style_free_roll_age_summary_csvs': '',
                'train_mean_rmse': np.nan,
                'train_mean_rmse_overall': np.nan,
                'train_mean_rmse_le25m': np.nan,
                'train_mean_rmse_gt25m': np.nan,
                'train_count_le25m': 0,
                'train_count_gt25m': 0,
                'val_mean_rmse': np.nan,
                'val_mean_rmse_overall': np.nan,
                'val_mean_rmse_le25m': np.nan,
                'val_mean_rmse_gt25m': np.nan,
                'val_count_le25m': 0,
                'val_count_gt25m': 0,
                'heldout_mean_rmse': np.nan,
                'heldout_mean_rmse_overall': np.nan,
                'heldout_mean_rmse_le25m': np.nan,
                'heldout_mean_rmse_gt25m': np.nan,
                'heldout_count_le25m': 0,
                'heldout_count_gt25m': 0,
                'heldout_transition_mean_rmse': np.nan,
                'train_persistence_mean_rmse': np.nan,
                'val_persistence_mean_rmse': np.nan,
                'heldout_persistence_mean_rmse': np.nan,
                'heldout_free_roll_mean_rmse': np.nan,
                'heldout_free_roll_mean_rmse_le25m': np.nan,
                'heldout_free_roll_mean_rmse_gt25m': np.nan,
                'heldout_free_roll_count_le25m': 0,
                'heldout_free_roll_count_gt25m': 0,
                'heldout_free_roll_mean_bias': np.nan,
                'heldout_free_roll_point_diagnostics_count': 0,
                'heldout_free_roll_age_summary_count': 0,
                'heldout_free_roll_point_diagnostics_csvs': '',
                'heldout_free_roll_age_summary_csvs': '',
                'heldout_observed_point_mean_rmse': np.nan,
                'heldout_observed_point_mean_mae': np.nan,
                'heldout_observed_point_mean_bias': np.nan,
                'heldout_observed_point_total_count': 0,
                'heldout_post_spinup_mean_rmse': np.nan,
                'heldout_post_spinup_mean_bias': np.nan,
            }
            record.update(_zero_profile_init_net_history_fields(
                zero_profile_init_net_detail_records,
                loss_weight=zero_profile_init_net_loss_weight,
                weight_eff=zero_profile_init_net_weight_eff,
                start_epoch=zero_profile_init_net_start_epoch,
                ramp_epochs=zero_profile_init_net_ramp_epochs,
                samples_per_lake=zero_profile_init_net_samples_per_lake,
                regularization_weight=zero_profile_init_net_regularization_weight,
                training_spinup_days=zero_profile_init_net_training_spinup_days,
                physics_weight=zero_profile_init_net_physics_weight,
                rollout_weight=zero_profile_init_net_rollout_weight,
                rollout_max_days=zero_profile_init_net_rollout_max_days,
                rollout_targets=zero_profile_init_net_rollout_targets,
                hidden_dim=zero_profile_init_net_hidden_dim,
                init_spread=zero_profile_init_net_init_spread,
                coeff_limit_sigma=zero_profile_init_net_coeff_limit_sigma,
                delta_limit_c=zero_profile_init_net_delta_limit_c,
            ))
            record.update(_daily_memory_history_fields(
                daily_memory_detail_records,
                reconstruction_weight=daily_memory_reconstruction_weight,
                weight_eff=daily_memory_weight_eff,
                samples_per_lake=daily_memory_samples_per_lake,
                temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
                heat_budget_weight=daily_memory_heat_budget_weight,
                physics_consistency_weight=daily_memory_physics_consistency_weight,
                regularization_weight=daily_memory_regularization_weight,
                coefficient_loss_weight=daily_memory_coefficient_loss_weight,
                start_epoch=daily_memory_start_epoch,
                ramp_epochs=daily_memory_ramp_epochs,
                hidden_dim=daily_memory_hidden_dim,
                init_spread=daily_memory_init_spread,
                coeff_limit_sigma=daily_memory_coeff_limit_sigma,
                prediction_branch=prediction_branch,
            ))
            record.update(_multitask_auxiliary_history_fields(
                detail_records,
                weight=multitask_auxiliary_weight,
                requested_weight=multitask_auxiliary_weight_requested,
                heat_weight=multitask_auxiliary_heat_weight,
                thermocline_weight=multitask_auxiliary_thermocline_weight,
                mld_weight=multitask_auxiliary_mld_weight,
                stability_weight=multitask_auxiliary_stability_weight,
                surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
                eof_weight=multitask_auxiliary_eof_weight,
                hidden_dim=multitask_auxiliary_hidden_dim,
            ))
            record.update(_unlabeled_heat_closure_history_fields(
                unlabeled_heat_closure_detail_records,
                weight=unlabeled_heat_closure_weight,
                batch_size=unlabeled_heat_closure_batch_size,
                window_days=unlabeled_heat_closure_window_days,
                horizons=unlabeled_heat_closure_horizons,
                tau_wm2=unlabeled_heat_closure_tau_wm2,
                open_water_only=unlabeled_heat_closure_open_water_only,
                lst_qc_min=unlabeled_heat_closure_lst_qc_min,
                reservoir_mode=unlabeled_heat_closure_reservoir_mode,
                mode=unlabeled_heat_closure_mode,
                state_source=unlabeled_heat_closure_state_source,
                state_spinup_days=unlabeled_heat_closure_spinup_days,
                solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
                solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
            ))
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
                for band in DEPTH_RMSE_BANDS:
                    record[f'train_transition_rmse_{band}_{horizon}d'] = np.nan
                    record[f'train_transition_count_{band}_{horizon}d'] = 0
                    record[f'heldout_transition_rmse_{band}_{horizon}d'] = np.nan
                    record[f'heldout_transition_count_{band}_{horizon}d'] = 0
                    record[f'val_rolling_start_rmse_{band}_{horizon}d'] = np.nan
                    record[f'val_rolling_start_count_{band}_{horizon}d'] = 0
                    record[f'heldout_rolling_start_rmse_{band}_{horizon}d'] = np.nan
                    record[f'heldout_rolling_start_count_{band}_{horizon}d'] = 0
                    record[f'heldout_free_roll_rmse_{band}_{horizon}d'] = np.nan
                    record[f'heldout_free_roll_count_{band}_{horizon}d'] = 0
                    record[f'heldout_initial_free_roll_rmse_{band}_{horizon}d'] = np.nan
                    record[f'heldout_initial_free_roll_count_{band}_{horizon}d'] = 0
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
            _add_zero_profile_export_metrics(record, 'val', {}, [])
            _add_zero_profile_export_metrics(record, 'heldout', {}, heldout_lakes)
            for lake in train_lakes:
                key = lake['lake_id']
                record[f'{key}_train_rmse'] = np.nan
                record[f'{key}_train_rmse_overall'] = np.nan
                for band in DEPTH_RMSE_BANDS:
                    record[f'{key}_train_rmse_{band}'] = np.nan
                    record[f'{key}_train_count_{band}'] = 0
                record[f'{key}_val_rmse'] = np.nan
                record[f'{key}_val_rmse_overall'] = np.nan
                for band in DEPTH_RMSE_BANDS:
                    record[f'{key}_val_rmse_{band}'] = np.nan
                    record[f'{key}_val_count_{band}'] = 0
                record[f'{key}_train_persistence_rmse'] = np.nan
                record[f'{key}_val_persistence_rmse'] = np.nan
                for horizon in (1, 3, 7, 14, 30, 60):
                    record[f'{key}_val_rolling_start_rmse_{horizon}d'] = np.nan
                    record[f'{key}_val_rolling_start_bias_{horizon}d'] = np.nan
                    record[f'{key}_val_rolling_start_count_{horizon}d'] = np.nan
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_val_rolling_start_rmse_{band}_{horizon}d'] = np.nan
                        record[f'{key}_val_rolling_start_count_{band}_{horizon}d'] = 0
            for lake in heldout_lakes:
                key = lake['lake_id']
                record[f'{key}_heldout_rmse'] = np.nan
                record[f'{key}_heldout_rmse_overall'] = np.nan
                record[f'{key}_heldout_transition_rmse'] = np.nan
                for band in DEPTH_RMSE_BANDS:
                    record[f'{key}_heldout_rmse_{band}'] = np.nan
                    record[f'{key}_heldout_count_{band}'] = 0
                    record[f'{key}_heldout_transition_rmse_{band}'] = np.nan
                    record[f'{key}_heldout_transition_count_{band}'] = 0
                record[f'{key}_heldout_persistence_rmse'] = np.nan
                record[f'{key}_heldout_free_roll_rmse'] = np.nan
                record[f'{key}_heldout_free_roll_rmse_overall'] = np.nan
                for band in DEPTH_RMSE_BANDS:
                    record[f'{key}_heldout_free_roll_rmse_{band}'] = np.nan
                    record[f'{key}_heldout_free_roll_count_{band}'] = 0
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
                    for band in DEPTH_RMSE_BANDS:
                        record[f'{key}_heldout_initial_free_roll_rmse_{band}_{horizon}d'] = np.nan
                        record[f'{key}_heldout_initial_free_roll_count_{band}_{horizon}d'] = 0
                        record[f'{key}_heldout_rolling_start_rmse_{band}_{horizon}d'] = np.nan
                        record[f'{key}_heldout_rolling_start_count_{band}_{horizon}d'] = 0
                        record[f'{key}_heldout_free_roll_rmse_{band}_{horizon}d'] = np.nan
                        record[f'{key}_heldout_free_roll_count_{band}_{horizon}d'] = 0
            _add_free_roll_bucket_metrics(record, 'heldout', {}, heldout_lakes)
            _add_free_roll_bucket_metrics(record, 'val_export_style', {}, [])
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
            record['gradient_norm'] = float(gradient_norm_value)
            record['optimizer_step_skipped_nonfinite'] = bool(optimizer_step_skipped_nonfinite)
            record = _prune_removed_mainline_output_fields(record)
            history.append(record)
            if profile_runtime:
                record['transition_seconds'] = float(transition_seconds)
                record['segment_seconds'] = float(segment_seconds)
                record['evaluation_seconds'] = 0.0
                record['epoch_seconds'] = float(time.perf_counter() - epoch_start_time)
                record = _prune_removed_mainline_output_fields(record)
                history[-1] = record
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
    checkpoint_payload = {
            'architecture': 'MultiLakeStateForecaster',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': int(epochs) - 1,
            'zero_profile_thermal_basis': zero_profile_thermal_basis,
            'seed': None if seed is None else int(seed),
            'manifest': str(Path(manifest_path)),
            'task_mode': task_mode,
            'data_fill_mode': data_fill_mode,
            'profile_supervision_scope': profile_supervision_scope,
            'profile_loss_target_mode': profile_loss_target_mode,
            'profile_sampling_mode': profile_sampling_mode,
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
            'full_eval_start_epoch': int(full_eval_start_epoch),
            'profile_runtime': bool(profile_runtime),
            'profile_gpu': bool(profile_gpu),
            'history_diagnostic_every_epochs': int(history_diagnostic_every_epochs),
            'torch_tf32': torch_tf32,
            'torch_matmul_precision': torch_matmul_precision,
            **_training_runtime_config_fields(
                training_amp=training_amp,
                training_amp_enabled=training_amp_runtime_enabled,
                training_history_detail_every_epochs=training_history_detail_every_epochs,
            ),
            'transition_batch_size': int(transition_batch_size),
            'segment_rollout_batch_size': int(segment_rollout_batch_size),
            'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
            'train_diagnostic_mode': train_diagnostic_mode,
            'export_after_training': export_after_training,
            'export_max_depth_m': export_max_depth_m,
            'cross_lake_batch_mode': cross_lake_batch_mode,
            'cross_lake_batch_size': int(cross_lake_batch_size),
            **_gpu_batch_autotune_config_fields(
                gpu_batch_autotune=gpu_batch_autotune,
                gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
                gpu_batch_autotune_applied=gpu_batch_autotune_applied,
            ),
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
            'export_style_validation': export_style_validation,
            'export_style_validation_max_lakes': int(export_style_validation_max_lakes),
            'full_eval_point_diagnostics': full_eval_point_diagnostics,
            'zero_profile_export_validation': zero_profile_export_validation,
            'zero_profile_export_validation_max_lakes': int(zero_profile_export_validation_max_lakes),
            'zero_profile_initializer': zero_profile_initializer,
            'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
            'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
            'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
            'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
            'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
            'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
            'zero_profile_thermal_basis_source_lake_count': int(
                zero_profile_thermal_basis_source_lake_count
            ),
            'zero_profile_init_net_loss_weight': float(zero_profile_init_net_loss_weight),
            'zero_profile_init_net_profile_loss_mode': 'rmse',
            'zero_profile_init_net_start_epoch': int(zero_profile_init_net_start_epoch),
            'zero_profile_init_net_ramp_epochs': int(zero_profile_init_net_ramp_epochs),
            'zero_profile_init_net_samples_per_lake': int(zero_profile_init_net_samples_per_lake),
            'zero_profile_init_net_regularization_weight': float(
                zero_profile_init_net_regularization_weight
            ),
            'zero_profile_init_net_hidden_dim': int(zero_profile_init_net_hidden_dim),
            'zero_profile_init_net_init_spread': float(zero_profile_init_net_init_spread),
            'zero_profile_init_net_coeff_limit_sigma': float(
                zero_profile_init_net_coeff_limit_sigma
            ),
            'zero_profile_init_net_delta_limit_c': float(zero_profile_init_net_delta_limit_c),
            'zero_profile_init_net_training_spinup_days': int(
                zero_profile_init_net_training_spinup_days
            ),
            'zero_profile_init_net_physics_weight': float(zero_profile_init_net_physics_weight),
            'zero_profile_init_net_rollout_weight': float(zero_profile_init_net_rollout_weight),
            'zero_profile_init_net_rollout_max_days': int(zero_profile_init_net_rollout_max_days),
            'zero_profile_init_net_rollout_targets': int(zero_profile_init_net_rollout_targets),
            'zero_profile_lswt_observer_mode': zero_profile_lswt_observer_mode,
            'zero_profile_spinup_days_matrix': ','.join(
                str(int(day)) for day in zero_profile_spinup_days
            ),
            'zero_profile_lswt_observer_strength': float(zero_profile_lswt_observer_strength),
            'zero_profile_lswt_observer_decay_depth_m': float(
                zero_profile_lswt_observer_decay_depth_m
            ),
            'zero_profile_lswt_observer_max_increment_c': float(
                zero_profile_lswt_observer_max_increment_c
            ),
            'zero_profile_lswt_observer_deep_update_fraction': float(
                zero_profile_lswt_observer_deep_update_fraction
            ),
            'zero_profile_lswt_observer_heat_content_limit_c': float(
                zero_profile_lswt_observer_heat_content_limit_c
            ),
            'zero_profile_lswt_observer_min_quality': float(
                zero_profile_lswt_observer_min_quality
            ),
            'kd_saturation_threshold': float(kd_saturation_threshold),
            'init_mode': init_mode,
            'spinup_days': int(spinup_days),
            'spinup_lst_assimilation_strength': float(spinup_lst_assimilation_strength),
            'spinup_lst_assimilation_decay_depth_m': float(spinup_lst_assimilation_decay_depth_m),
            'spinup_lst_assimilation_max_increment_c': float(spinup_lst_assimilation_max_increment_c),
            'rollout_start_date': rollout_start_date,
            'best_by_val_rolling_enabled': bool(best_val_rolling_enabled),
            'best_by_val_rolling_checkpoint_path': str(best_val_rolling_checkpoint_path),
            'best_by_val_rolling_metrics_path': str(best_val_rolling_metrics_path),
            'best_by_val_rolling_score': (
                None if not np.isfinite(best_val_rolling_score) else float(best_val_rolling_score)
            ),
            'best_by_val_rolling_epoch': best_val_rolling_epoch,
            'training_history': history,
        }
    checkpoint_payload.update(_multitask_auxiliary_config_fields(
        weight=multitask_auxiliary_weight,
        requested_weight=multitask_auxiliary_weight_requested,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        hidden_dim=multitask_auxiliary_hidden_dim,
    ))
    checkpoint_payload.update(_daily_memory_config_fields(
        reconstruction_weight=daily_memory_reconstruction_weight,
        samples_per_lake=daily_memory_samples_per_lake,
        temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
        heat_budget_weight=daily_memory_heat_budget_weight,
        physics_consistency_weight=daily_memory_physics_consistency_weight,
        regularization_weight=daily_memory_regularization_weight,
        coefficient_loss_weight=daily_memory_coefficient_loss_weight,
        start_epoch=daily_memory_start_epoch,
        ramp_epochs=daily_memory_ramp_epochs,
        hidden_dim=daily_memory_hidden_dim,
        init_spread=daily_memory_init_spread,
        coeff_limit_sigma=daily_memory_coeff_limit_sigma,
        prediction_branch=prediction_branch,
    ))
    checkpoint_payload.update(_model_mainline_config_fields(
        model_mainline,
        model_mainline_resolved,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
        physics_rollout_thermal_state_weight=physics_rollout_thermal_state_weight,
        physics_rollout_thermal_state_weight_eff=physics_rollout_thermal_state_weight_eff,
        thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
    ))
    checkpoint_payload.update(thermal_state_profile_fusion_config)
    checkpoint_payload.update(_unlabeled_heat_closure_config_fields(
        weight=unlabeled_heat_closure_weight,
        batch_size=unlabeled_heat_closure_batch_size,
        window_days=unlabeled_heat_closure_window_days,
        horizons=unlabeled_heat_closure_horizons,
        tau_wm2=unlabeled_heat_closure_tau_wm2,
        open_water_only=unlabeled_heat_closure_open_water_only,
        lst_qc_min=unlabeled_heat_closure_lst_qc_min,
        reservoir_mode=unlabeled_heat_closure_reservoir_mode,
        mode=unlabeled_heat_closure_mode,
        state_source=unlabeled_heat_closure_state_source,
        state_spinup_days=unlabeled_heat_closure_spinup_days,
        solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
        solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
    ))
    torch.save(_prune_removed_mainline_output_fields(checkpoint_payload), checkpoint_path)
    history_csv = output_dir / 'global_state_forecaster_training_history.csv'
    pd.DataFrame(history).to_csv(history_csv, index=False)
    split_summary = output_dir / 'global_state_forecaster_split_summary.json'
    split_summary_payload = {
        lake['lake_id']: {
            'train_pairs': len(lake['pairs']['train']),
            'val_pairs': len(lake['pairs']['val']),
            'all_pairs': len(lake['pairs']['all']),
            'train_segment_rollout_sequences': len(lake['segment_rollout_sequences']['train']),
            'val_segment_rollout_sequences': len(lake['segment_rollout_sequences']['val']),
            'all_segment_rollout_sequences': len(lake['segment_rollout_sequences']['all']),
            'unlabeled_heat_closure_windows': len(lake.get('unlabeled_heat_closure_windows', ())),
            'unlabeled_heat_closure_windows_by_horizon': {
                str(int(horizon)): len(windows)
                for horizon, windows in sorted(
                    lake.get('unlabeled_heat_closure_windows_by_horizon', {}).items()
                )
            },
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
        'seed': None if seed is None else int(seed),
        'lake_ids': [lake['lake_id'] for lake in lakes],
        'train_lake_ids': [lake['lake_id'] for lake in train_lakes],
        'heldout_lake_ids': [lake['lake_id'] for lake in heldout_lakes],
        'excluded_lake_ids': [lake['lake_id'] for lake in excluded_lakes],
        'profile_supervision_scope': profile_supervision_scope,
        'profile_loss_target_mode': profile_loss_target_mode,
        'profile_sampling_mode': profile_sampling_mode,
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
        'full_eval_start_epoch': int(full_eval_start_epoch),
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
        **_training_runtime_config_fields(
            training_amp=training_amp,
            training_amp_enabled=training_amp_runtime_enabled,
            training_history_detail_every_epochs=training_history_detail_every_epochs,
        ),
        'transition_batch_size': int(transition_batch_size),
        'segment_rollout_batch_size': int(segment_rollout_batch_size),
        'rolling_horizon_batch_size': int(rolling_horizon_batch_size),
        'train_diagnostic_mode': train_diagnostic_mode,
        'export_after_training': export_after_training,
        'export_max_depth_m': export_max_depth_m,
        'cross_lake_batch_mode': cross_lake_batch_mode,
        'cross_lake_batch_size': int(cross_lake_batch_size),
        **_gpu_batch_autotune_config_fields(
            gpu_batch_autotune=gpu_batch_autotune,
            gpu_batch_autotune_target_batch_size=gpu_batch_autotune_target_batch_size,
            gpu_batch_autotune_applied=gpu_batch_autotune_applied,
        ),
        'segment_rollout_loss_weight': float(segment_rollout_loss_weight),
        'segment_rollout_start_epoch': int(segment_rollout_start_epoch),
        'segment_rollout_ramp_epochs': int(segment_rollout_ramp_epochs),
        'segment_rollout_max_days': int(segment_rollout_max_days),
        'segment_rollout_samples_per_lake': int(segment_rollout_samples_per_lake),
        'segment_rollout_lst_surface_weight': float(segment_rollout_lst_surface_weight),
        'lst_feature_dropout_probability': float(lst_feature_dropout_probability),
        'task_mode': task_mode,
        'data_fill_mode': data_fill_mode,
        'export_style_validation': export_style_validation,
        'export_style_validation_max_lakes': int(export_style_validation_max_lakes),
        'full_eval_point_diagnostics': full_eval_point_diagnostics,
        'zero_profile_export_validation': zero_profile_export_validation,
        'zero_profile_export_validation_max_lakes': int(zero_profile_export_validation_max_lakes),
        'zero_profile_initializer': zero_profile_initializer,
        'zero_profile_thermal_basis_components': int(zero_profile_thermal_basis_components),
        'zero_profile_thermal_basis_grid_points': int(zero_profile_thermal_basis_grid_points),
        'zero_profile_thermal_basis_balance_mode': zero_profile_thermal_basis_balance_mode_effective,
        'refit_zero_profile_thermal_basis': refit_zero_profile_thermal_basis,
        'zero_profile_thermal_basis_source': zero_profile_thermal_basis_source,
        'zero_profile_thermal_basis_profile_count': int(zero_profile_thermal_basis_profile_count),
        'zero_profile_thermal_basis_source_lake_count': int(
            zero_profile_thermal_basis_source_lake_count
        ),
        'zero_profile_init_net_loss_weight': float(zero_profile_init_net_loss_weight),
        'zero_profile_init_net_profile_loss_mode': 'rmse',
        'zero_profile_init_net_start_epoch': int(zero_profile_init_net_start_epoch),
        'zero_profile_init_net_ramp_epochs': int(zero_profile_init_net_ramp_epochs),
        'zero_profile_init_net_samples_per_lake': int(zero_profile_init_net_samples_per_lake),
        'zero_profile_init_net_regularization_weight': float(
            zero_profile_init_net_regularization_weight
        ),
        'zero_profile_init_net_hidden_dim': int(zero_profile_init_net_hidden_dim),
        'zero_profile_init_net_init_spread': float(zero_profile_init_net_init_spread),
        'zero_profile_init_net_coeff_limit_sigma': float(
            zero_profile_init_net_coeff_limit_sigma
        ),
        'zero_profile_init_net_delta_limit_c': float(zero_profile_init_net_delta_limit_c),
        'zero_profile_init_net_training_spinup_days': int(
            zero_profile_init_net_training_spinup_days
        ),
        'zero_profile_init_net_physics_weight': float(zero_profile_init_net_physics_weight),
        'zero_profile_init_net_rollout_weight': float(zero_profile_init_net_rollout_weight),
        'zero_profile_init_net_rollout_max_days': int(zero_profile_init_net_rollout_max_days),
        'zero_profile_init_net_rollout_targets': int(zero_profile_init_net_rollout_targets),
        'zero_profile_lswt_observer_mode': zero_profile_lswt_observer_mode,
        'zero_profile_spinup_days_matrix': ','.join(
            str(int(day)) for day in zero_profile_spinup_days
        ),
        'zero_profile_lswt_observer_strength': float(zero_profile_lswt_observer_strength),
        'zero_profile_lswt_observer_decay_depth_m': float(
            zero_profile_lswt_observer_decay_depth_m
        ),
        'zero_profile_lswt_observer_max_increment_c': float(
            zero_profile_lswt_observer_max_increment_c
        ),
        'zero_profile_lswt_observer_deep_update_fraction': float(
            zero_profile_lswt_observer_deep_update_fraction
        ),
        'zero_profile_lswt_observer_heat_content_limit_c': float(
            zero_profile_lswt_observer_heat_content_limit_c
        ),
        'zero_profile_lswt_observer_min_quality': float(
            zero_profile_lswt_observer_min_quality
        ),
        'kd_saturation_threshold': float(kd_saturation_threshold),
        'kd_saturation_penalty_weight': float(kd_saturation_penalty_weight),
        'static_feature_dim': int(static_dim),
        'static_feature_keys': list(STATIC_FEATURE_KEYS),
    }
    split_summary_payload['_config'].update(_multitask_auxiliary_config_fields(
        weight=multitask_auxiliary_weight,
        requested_weight=multitask_auxiliary_weight_requested,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        hidden_dim=multitask_auxiliary_hidden_dim,
    ))
    split_summary_payload['_config'].update(_daily_memory_config_fields(
        reconstruction_weight=daily_memory_reconstruction_weight,
        samples_per_lake=daily_memory_samples_per_lake,
        temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
        heat_budget_weight=daily_memory_heat_budget_weight,
        physics_consistency_weight=daily_memory_physics_consistency_weight,
        regularization_weight=daily_memory_regularization_weight,
        coefficient_loss_weight=daily_memory_coefficient_loss_weight,
        start_epoch=daily_memory_start_epoch,
        ramp_epochs=daily_memory_ramp_epochs,
        hidden_dim=daily_memory_hidden_dim,
        init_spread=daily_memory_init_spread,
        coeff_limit_sigma=daily_memory_coeff_limit_sigma,
        prediction_branch=prediction_branch,
    ))
    split_summary_payload['_config'].update(_model_mainline_config_fields(
        model_mainline,
        model_mainline_resolved,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
        physics_rollout_thermal_state_weight=physics_rollout_thermal_state_weight,
        physics_rollout_thermal_state_weight_eff=physics_rollout_thermal_state_weight_eff,
        thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
    ))
    split_summary_payload['_config'].update(thermal_state_profile_fusion_config)
    split_summary_payload['_config'].update(_unlabeled_heat_closure_config_fields(
        weight=unlabeled_heat_closure_weight,
        batch_size=unlabeled_heat_closure_batch_size,
        window_days=unlabeled_heat_closure_window_days,
        horizons=unlabeled_heat_closure_horizons,
        tau_wm2=unlabeled_heat_closure_tau_wm2,
        open_water_only=unlabeled_heat_closure_open_water_only,
        lst_qc_min=unlabeled_heat_closure_lst_qc_min,
        reservoir_mode=unlabeled_heat_closure_reservoir_mode,
        mode=unlabeled_heat_closure_mode,
        state_source=unlabeled_heat_closure_state_source,
        state_spinup_days=unlabeled_heat_closure_spinup_days,
        solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
        solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
    ))
    split_summary_payload = _prune_removed_mainline_split_summary(split_summary_payload)
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
            if prediction_branch == 'daily_memory':
                export_info = export_daily_memory_reconstruction(
                    model,
                    lake,
                    output_dir,
                    export_max_depth_m=export_max_depth_m,
                    thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
                    thermal_state_profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                    thermal_state_profile_fusion_lookup_split=thermal_state_profile_fusion_lookup_split,
                    thermal_state_profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                    thermal_state_profile_fusion_min_depth_fraction=(
                        thermal_state_profile_fusion_min_depth_fraction
                    ),
                    thermal_state_profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                    thermal_state_profile_fusion_coeff_limit_sigma=(
                        thermal_state_profile_fusion_coeff_limit_sigma
                    ),
                )
            else:
                export_info = export_heldout_state_forecast(
                    model,
                    lake,
                    output_dir,
                    task_mode=task_mode,
                    init_mode=init_mode,
                    spinup_days=spinup_days,
                    zero_profile_initializer=zero_profile_initializer,
                    spinup_lswt_observer_mode=(
                        zero_profile_lswt_observer_mode
                        if zero_profile_lswt_observer_mode != 'off'
                        else DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE
                    ),
                    spinup_lst_assimilation_strength=spinup_lst_assimilation_strength,
                    spinup_lst_assimilation_decay_depth_m=spinup_lst_assimilation_decay_depth_m,
                    spinup_lst_assimilation_max_increment_c=spinup_lst_assimilation_max_increment_c,
                    rollout_lswt_observer_mode=zero_profile_lswt_observer_mode,
                    lswt_observer_strength=zero_profile_lswt_observer_strength,
                    lswt_observer_decay_depth_m=zero_profile_lswt_observer_decay_depth_m,
                    lswt_observer_max_increment_c=zero_profile_lswt_observer_max_increment_c,
                    lswt_observer_low_rank_deep_update_fraction=(
                        zero_profile_lswt_observer_deep_update_fraction
                    ),
                    lswt_observer_heat_content_limit_c=zero_profile_lswt_observer_heat_content_limit_c,
                    lswt_observer_min_quality=zero_profile_lswt_observer_min_quality,
                    rollout_start_date=rollout_start_date,
                    export_max_depth_m=export_max_depth_m,
                    hard_density_stability=hard_density_stability_active,
                    hard_density_stability_mode=hard_density_stability_mode,
                    thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
                    thermal_state_profile_fusion_time_policy=thermal_state_profile_fusion_time_policy,
                    thermal_state_profile_fusion_max_age_days=thermal_state_profile_fusion_max_age_days,
                    thermal_state_profile_fusion_min_depth_fraction=(
                        thermal_state_profile_fusion_min_depth_fraction
                    ),
                    thermal_state_profile_fusion_max_weight=thermal_state_profile_fusion_max_weight,
                    thermal_state_profile_fusion_coeff_limit_sigma=(
                        thermal_state_profile_fusion_coeff_limit_sigma
                    ),
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
    result = {
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
        **_training_runtime_config_fields(
            training_amp=training_amp,
            training_amp_enabled=training_amp_runtime_enabled,
            training_history_detail_every_epochs=training_history_detail_every_epochs,
        ),
    }
    result.update(_multitask_auxiliary_config_fields(
        weight=multitask_auxiliary_weight,
        requested_weight=multitask_auxiliary_weight_requested,
        heat_weight=multitask_auxiliary_heat_weight,
        thermocline_weight=multitask_auxiliary_thermocline_weight,
        mld_weight=multitask_auxiliary_mld_weight,
        stability_weight=multitask_auxiliary_stability_weight,
        surface_bottom_weight=multitask_auxiliary_surface_bottom_weight,
        eof_weight=multitask_auxiliary_eof_weight,
        hidden_dim=multitask_auxiliary_hidden_dim,
    ))
    result.update(_daily_memory_config_fields(
        reconstruction_weight=daily_memory_reconstruction_weight,
        samples_per_lake=daily_memory_samples_per_lake,
        temporal_smoothness_weight=daily_memory_temporal_smoothness_weight,
        heat_budget_weight=daily_memory_heat_budget_weight,
        physics_consistency_weight=daily_memory_physics_consistency_weight,
        regularization_weight=daily_memory_regularization_weight,
        coefficient_loss_weight=daily_memory_coefficient_loss_weight,
        start_epoch=daily_memory_start_epoch,
        ramp_epochs=daily_memory_ramp_epochs,
        hidden_dim=daily_memory_hidden_dim,
        init_spread=daily_memory_init_spread,
        coeff_limit_sigma=daily_memory_coeff_limit_sigma,
        prediction_branch=prediction_branch,
    ))
    result.update(_model_mainline_config_fields(
        model_mainline,
        model_mainline_resolved,
        daily_memory_reconstruction_weight=daily_memory_reconstruction_weight,
        prediction_branch=prediction_branch,
        physics_rollout_thermal_state_weight=physics_rollout_thermal_state_weight,
        physics_rollout_thermal_state_weight_eff=physics_rollout_thermal_state_weight_eff,
        thermal_state_profile_fusion_mode=thermal_state_profile_fusion_mode,
    ))
    result.update(thermal_state_profile_fusion_config)
    result.update(_unlabeled_heat_closure_config_fields(
        weight=unlabeled_heat_closure_weight,
        batch_size=unlabeled_heat_closure_batch_size,
        window_days=unlabeled_heat_closure_window_days,
        horizons=unlabeled_heat_closure_horizons,
        tau_wm2=unlabeled_heat_closure_tau_wm2,
        open_water_only=unlabeled_heat_closure_open_water_only,
        lst_qc_min=unlabeled_heat_closure_lst_qc_min,
        reservoir_mode=unlabeled_heat_closure_reservoir_mode,
        mode=unlabeled_heat_closure_mode,
        state_source=unlabeled_heat_closure_state_source,
        state_spinup_days=unlabeled_heat_closure_spinup_days,
        solver_guard_weight=unlabeled_heat_closure_solver_guard_weight,
        solver_guard_tau_wm2=unlabeled_heat_closure_solver_guard_tau_wm2,
    ))
    return _prune_removed_mainline_output_fields(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a multi-lake reconstruction-state LakePINN.')
    parser.add_argument('--manifest', required=True, help='JSON manifest listing lake forcing/LST/profile/metadata inputs.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducible training runs.')
    parser.add_argument('--depth-points', type=int, default=40)
    parser.add_argument('--max-rollout-days', type=int, default=90)
    parser.add_argument('--history-window-days', type=int, default=30)
    parser.add_argument('--split-mode', choices=['time_blocked', 'seasonal_blocked', 'depth_interleaved'], default='time_blocked')
    parser.add_argument(
        '--profile-supervision-scope',
        choices=['train', 'all'],
        default='train',
        help='Profiles used for transition/segment supervision on training lake-years. all uses train+val dates.',
    )
    parser.add_argument(
        '--profile-loss-target-mode',
        choices=sorted(PROFILE_LOSS_TARGET_MODES),
        default=DEFAULT_PROFILE_LOSS_TARGET_MODE,
        help=(
            'Profile supervision target construction. grid_masked uses the '
            'interpolated depth grid with an observed-range mask; '
            'observed_point_strict only supervises grid cells nearest real '
            'observed profile depths.'
        ),
    )
    parser.add_argument(
        '--profile-sampling-mode',
        choices=sorted(PROFILE_SAMPLING_MODES),
        default=DEFAULT_PROFILE_SAMPLING_MODE,
        help=(
            'Profile-date sampling for init-net, daily-memory, and segment '
            'rollout supervision. season_balanced rotates available seasons '
            'without creating labels for missing seasons.'
        ),
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
    parser.add_argument('--residual-limit-c', type=float, default=DEFAULT_RESIDUAL_LIMIT_C)
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
        default=DEFAULT_RESIDUAL_REGULARIZATION_WEIGHT,
        help='Weight for penalizing daily learned residual corrections during transition training.',
    )
    parser.add_argument(
        '--daily-tendency-weight',
        type=float,
        default=DEFAULT_DAILY_TENDENCY_WEIGHT,
        help='Weight for layered daily temperature jump penalties during transition training.',
    )
    parser.add_argument(
        '--physical-scale-regularization-weight',
        type=float,
        default=DEFAULT_PHYSICAL_SCALE_REGULARIZATION_WEIGHT,
        help='Weight for keeping learned physical scales near 1.0.',
    )
    parser.add_argument(
        '--physical-scale-smoothness-weight',
        type=float,
        default=DEFAULT_PHYSICAL_SCALE_SMOOTHNESS_WEIGHT,
        help='Weight for penalizing day-to-day jumps in learned physical scales.',
    )
    parser.add_argument(
        '--kd-prior-regularization-weight',
        type=float,
        default=DEFAULT_KD_PRIOR_REGULARIZATION_WEIGHT,
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
        default=DEFAULT_ADAPTIVE_WIND_KZ_MIN,
        help='Lower bound for metadata-conditioned wind_kz_scale.',
    )
    parser.add_argument(
        '--adaptive-wind-kz-max',
        type=float,
        default=DEFAULT_ADAPTIVE_WIND_KZ_MAX,
        help='Upper bound for metadata-conditioned wind_kz_scale.',
    )
    parser.add_argument(
        '--adaptive-blend-alpha-min',
        type=float,
        default=DEFAULT_ADAPTIVE_BLEND_ALPHA_MIN,
        help='Lower bound for metadata-conditioned turbulent flux blend alpha.',
    )
    parser.add_argument(
        '--adaptive-blend-alpha-max',
        type=float,
        default=DEFAULT_ADAPTIVE_BLEND_ALPHA_MAX,
        help='Upper bound for metadata-conditioned turbulent flux blend alpha.',
    )
    parser.add_argument(
        '--adaptive-kd-multiplier-min',
        type=float,
        default=DEFAULT_ADAPTIVE_KD_MULTIPLIER_MIN,
        help='Lower bound for metadata-conditioned shortwave/Kd multiplier.',
    )
    parser.add_argument(
        '--adaptive-kd-multiplier-max',
        type=float,
        default=DEFAULT_ADAPTIVE_KD_MULTIPLIER_MAX,
        help='Upper bound for metadata-conditioned shortwave/Kd multiplier.',
    )
    parser.add_argument(
        '--adaptive-turbulent-exchange-scale-min',
        type=float,
        default=DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MIN,
        help='Lower bound for metadata-conditioned bulk turbulent exchange scale.',
    )
    parser.add_argument(
        '--adaptive-turbulent-exchange-scale-max',
        type=float,
        default=DEFAULT_ADAPTIVE_TURBULENT_EXCHANGE_SCALE_MAX,
        help='Upper bound for metadata-conditioned bulk turbulent exchange scale.',
    )
    parser.add_argument(
        '--adaptive-convective-mixing-scale-min',
        type=float,
        default=DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MIN,
        help='Lower bound for metadata-conditioned convective/overturn mixing scale.',
    )
    parser.add_argument(
        '--adaptive-convective-mixing-scale-max',
        type=float,
        default=DEFAULT_ADAPTIVE_CONVECTIVE_MIXING_SCALE_MAX,
        help='Upper bound for metadata-conditioned convective/overturn mixing scale.',
    )
    parser.add_argument(
        '--adaptive-ice-shortwave-scale-min',
        type=float,
        default=DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MIN,
        help='Lower bound for metadata-conditioned ice/snow shortwave attenuation scale.',
    )
    parser.add_argument(
        '--adaptive-ice-shortwave-scale-max',
        type=float,
        default=DEFAULT_ADAPTIVE_ICE_SHORTWAVE_SCALE_MAX,
        help='Upper bound for metadata-conditioned ice/snow shortwave attenuation scale.',
    )
    parser.add_argument(
        '--adaptive-parameter-regularization-weight',
        type=float,
        default=DEFAULT_ADAPTIVE_PARAMETER_REGULARIZATION_WEIGHT,
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
        '--multitask-auxiliary-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_WEIGHT,
        help='Outer transition-pair weight for auxiliary lake thermal-state heads. Default 0 disables.',
    )
    parser.add_argument(
        '--multitask-auxiliary-heat-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_HEAT_WEIGHT,
        help='Relative auxiliary-head weight for area-weighted heat-content/column-mean temperature.',
    )
    parser.add_argument(
        '--multitask-auxiliary-thermocline-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_THERMOCLINE_WEIGHT,
        help='Relative auxiliary-head weight for thermocline-depth supervision.',
    )
    parser.add_argument(
        '--multitask-auxiliary-mld-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_MLD_WEIGHT,
        help='Relative auxiliary-head weight for mixed-layer-depth supervision.',
    )
    parser.add_argument(
        '--multitask-auxiliary-stability-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_STABILITY_WEIGHT,
        help='Relative auxiliary-head weight for Schmidt-stability proxy supervision.',
    )
    parser.add_argument(
        '--multitask-auxiliary-surface-bottom-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_SURFACE_BOTTOM_WEIGHT,
        help='Relative auxiliary-head weight for surface-bottom temperature difference.',
    )
    parser.add_argument(
        '--multitask-auxiliary-eof-weight',
        type=float,
        default=DEFAULT_MULTITASK_AUXILIARY_EOF_WEIGHT,
        help='Relative auxiliary-head weight for train-only EOF/PCA profile coefficient targets.',
    )
    parser.add_argument(
        '--multitask-auxiliary-hidden-dim',
        type=int,
        default=DEFAULT_MULTITASK_AUXILIARY_HIDDEN_DIM,
        help='Hidden dimension for the auxiliary lake thermal-state prediction head.',
    )
    parser.add_argument(
        '--physics-rollout-thermal-state-weight',
        type=float,
        default=DEFAULT_PHYSICS_ROLLOUT_THERMAL_STATE_WEIGHT,
        help=(
            'Default low-dimensional thermal-state supervision weight for the '
            'init_physics_rollout mainline. It reuses the multitask auxiliary '
            'targets and can be set to 0 to disable.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-weight',
        type=float,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_WEIGHT,
        help=(
            'Weight for no-profile-date external storage-budget heat-closure '
            'collocation loss. Set 0 to disable.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-batch-size',
        type=int,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_BATCH_SIZE,
        help='Number of no-profile heat-closure windows sampled per lake per epoch. Use 0 for all.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-window-days',
        type=int,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_WINDOW_DAYS,
        help='Legacy scalar length in days for no-profile heat-closure windows when horizons are not set.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-horizons',
        default=_format_unlabeled_heat_closure_horizons(DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS),
        help=(
            'Comma-separated no-profile heat-closure horizons in days. '
            'Default 1,7,30 adds daily, weekly, and monthly storage-budget constraints.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-tau-wm2',
        type=float,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_TAU_WM2,
        help='Dead-zone threshold in W/m2 before no-profile heat-closure residuals are penalized.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-open-water-only',
        choices=['off', 'on'],
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_OPEN_WATER_ONLY,
        help='Gate no-profile heat-closure steps to open-water, low-ice days.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-lst-qc-min',
        type=float,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_LST_QC_MIN,
        help='Minimum LST quality factor for no-profile heat-closure collocation steps.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-reservoir-mode',
        choices=sorted(UNLABELED_HEAT_CLOSURE_RESERVOIR_MODES),
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_RESERVOIR_MODE,
        help='How reservoirs participate in no-profile heat closure: diagnostic_only, include, or exclude.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-mode',
        choices=sorted(UNLABELED_HEAT_CLOSURE_MODES),
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_MODE,
        help=(
            'No-profile heat closure signal. storage_budget_thresholded trains '
            'predicted heat-storage change against an independent forcing budget; '
            'storage_budget_smooth_l1 is the smooth storage-budget ablation.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-state-source',
        choices=sorted(UNLABELED_HEAT_CLOSURE_STATE_SOURCES),
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_STATE_SOURCE,
        help=(
            'Start-state source for no-profile heat-closure windows: prior_window '
            'uses the instantaneous zero-profile prior; spinup_then_window rolls '
            'from before the window start before applying the closure loss.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-spinup-days',
        type=int,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_SPINUP_DAYS,
        help='Spinup days used when unlabeled heat closure state source is spinup_then_window.',
    )
    parser.add_argument(
        '--unlabeled-heat-closure-solver-guard-weight',
        type=float,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_WEIGHT,
        help=(
            'Small auxiliary weight for thresholded internal solver heat-closure '
            'residual when using storage-budget no-profile loss. Default 0 disables.'
        ),
    )
    parser.add_argument(
        '--unlabeled-heat-closure-solver-guard-tau-wm2',
        type=float,
        default=DEFAULT_UNLABELED_HEAT_CLOSURE_SOLVER_GUARD_TAU_WM2,
        help='Dead-zone threshold in W/m2 for the internal solver heat-closure guard.',
    )
    parser.add_argument(
        '--no-profile-lst-surface-weight',
        type=float,
        default=DEFAULT_NO_PROFILE_LST_SURFACE_WEIGHT,
        help=(
            'Weak Huber surface-temperature loss weight for no-profile dates '
            'with observed open-water LST/LSWT. Set 0 to disable.'
        ),
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
        default=DEFAULT_RESIDUAL_TIME_SMOOTH_WEIGHT,
        help='Weight for residual vertical/time smoothness inside segment rollout loss.',
    )
    parser.add_argument(
        '--rolling-horizon-eval-max-starts',
        type=int,
        default=40,
        help='Maximum observed start profiles sampled for rolling-start horizon metrics. Use 0 or negative for all starts.',
    )
    parser.add_argument(
        '--export-style-validation',
        choices=['off', 'on'],
        default=DEFAULT_EXPORT_STYLE_VALIDATION_MODE,
        help='Record train-lake full free-roll export-style diagnostics during full eval. Diagnostic only.',
    )
    parser.add_argument(
        '--export-style-validation-max-lakes',
        type=int,
        default=DEFAULT_EXPORT_STYLE_VALIDATION_MAX_LAKES,
        help='Maximum train lakes used for export-style validation diagnostics. 0 uses all train lakes.',
    )
    parser.add_argument(
        '--full-eval-point-diagnostics',
        choices=['off', 'on'],
        default=DEFAULT_FULL_EVAL_POINT_DIAGNOSTICS_MODE,
        help='Write diagnostic-only point-level free-roll residual and depth-band CSVs during full eval.',
    )
    parser.add_argument(
        '--zero-profile-export-validation',
        choices=['off', 'on'],
        default=DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MODE,
        help=(
            'Record diagnostic-only no-profile-inference export metrics during full eval '
            'using prior_spinup + free rollout.'
        ),
    )
    parser.add_argument(
        '--zero-profile-export-validation-max-lakes',
        type=int,
        default=DEFAULT_ZERO_PROFILE_EXPORT_VALIDATION_MAX_LAKES,
        help='Maximum train lakes used for zero-profile export validation. 0 uses all train lakes.',
    )
    parser.add_argument(
        '--zero-profile-initializer',
        choices=list(MAINLINE_ZERO_PROFILE_INITIALIZER_MODE_CHOICES),
        default=DEFAULT_ZERO_PROFILE_INITIALIZER_MODE,
        help='Tenth-version mainline initializer used by zero-profile export validation. low_dof is the baseline; lswt_climatology_low_dof uses raw/open-water LSWT only as the strong surface anchor; eof_pca_low_dof projects the zero-profile prior onto a train-only EOF/PCA thermal basis; eof_pca_init_net adds a train-supervised low-rank network correction.',
    )
    parser.add_argument(
        '--zero-profile-thermal-basis-components',
        type=int,
        default=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_COMPONENTS,
        help='Number of train-only EOF/PCA components used by zero_profile_initializer=eof_pca_low_dof.',
    )
    parser.add_argument(
        '--zero-profile-thermal-basis-grid-points',
        type=int,
        default=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_GRID_POINTS,
        help='Normalized-depth grid size used to fit the zero-profile EOF/PCA thermal basis.',
    )
    parser.add_argument(
        '--zero-profile-thermal-basis-balance-mode',
        choices=list(ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODES),
        default=DEFAULT_ZERO_PROFILE_THERMAL_BASIS_BALANCE_MODE,
        help=(
            'How train-split profiles are weighted when fitting the EOF/PCA thermal basis. '
            'lake_season_depth_coverage reduces dominance by data-rich lakes, seasons, '
            'and depth-coverage patterns; off reproduces plain unweighted PCA.'
        ),
    )
    parser.add_argument(
        '--refit-zero-profile-thermal-basis',
        choices=['off', 'on'],
        default=DEFAULT_REFIT_ZERO_PROFILE_THERMAL_BASIS,
        help=(
            'Default off reuses zero_profile_thermal_basis stored in a checkpoint. '
            'Use on only when intentionally refitting the train-split EOF/PCA basis.'
        ),
    )
    parser.add_argument(
        '--zero-profile-init-net-loss-weight',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_LOSS_WEIGHT,
        help='Outer train-split profile supervision weight for zero_profile_initializer=eof_pca_init_net. Default 0 disables.',
    )
    parser.add_argument(
        '--zero-profile-init-net-start-epoch',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_START_EPOCH,
        help='Epoch where train-split zero-profile init-net supervision starts.',
    )
    parser.add_argument(
        '--zero-profile-init-net-ramp-epochs',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_RAMP_EPOCHS,
        help='Ramp epochs for zero-profile init-net supervision weight.',
    )
    parser.add_argument(
        '--zero-profile-init-net-samples-per-lake',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_SAMPLES_PER_LAKE,
        help='Train profiles sampled per lake per epoch for init-net supervision. 0 uses all train profiles.',
    )
    parser.add_argument(
        '--zero-profile-init-net-regularization-weight',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_REGULARIZATION_WEIGHT,
        help='Inner regularization weight on init-net low-rank coefficients and profile delta.',
    )
    parser.add_argument(
        '--zero-profile-init-net-hidden-dim',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_HIDDEN_DIM,
        help='Hidden dimension of the zero-profile init-net low-rank correction head.',
    )
    parser.add_argument(
        '--zero-profile-init-net-init-spread',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_INIT_SPREAD,
        help='Final-layer init std for the zero-profile init-net head. 0 starts exactly at the prior.',
    )
    parser.add_argument(
        '--zero-profile-init-net-coeff-limit-sigma',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_COEFF_LIMIT_SIGMA,
        help='Bound for learned low-rank coefficients in train-basis coefficient standard deviations.',
    )
    parser.add_argument(
        '--zero-profile-init-net-delta-limit-c',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_DELTA_LIMIT_C,
        help='Per-depth tanh cap for the init-net temperature correction in C. 0 disables the cap.',
    )
    parser.add_argument(
        '--zero-profile-init-net-training-spinup-days',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_TRAINING_SPINUP_DAYS,
        help=(
            'Days before each supervised train-profile date to initialize with the '
            'init-net and differentiably roll physics before applying the profile '
            'loss. 0 keeps direct T_init supervision.'
        ),
    )
    parser.add_argument(
        '--zero-profile-init-net-physics-weight',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_PHYSICS_WEIGHT,
        help='Weight for physical/range/density/smoothness regularization on init-net profiles.',
    )
    parser.add_argument(
        '--zero-profile-init-net-rollout-weight',
        type=float,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_WEIGHT,
        help=(
            'Inner weight for end-to-end init-net -> physics rollout supervision '
            'against future train profiles. Default 0 disables.'
        ),
    )
    parser.add_argument(
        '--zero-profile-init-net-rollout-max-days',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_MAX_DAYS,
        help='Maximum future horizon in days for init-net -> physics rollout profile targets.',
    )
    parser.add_argument(
        '--zero-profile-init-net-rollout-targets',
        type=int,
        default=DEFAULT_ZERO_PROFILE_INIT_NET_ROLLOUT_TARGETS,
        help='Maximum future train-profile targets per sampled init-net anchor. 0 uses all within horizon.',
    )
    parser.add_argument(
        '--daily-memory-reconstruction-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_RECONSTRUCTION_WEIGHT,
        help=(
            'Outer weight for the optional EOF/PCA daily-memory auxiliary branch. '
            "With prediction_branch='physics_rollout', profiles train this auxiliary "
            'low-dimensional state head while physics rollout remains the prediction path. '
            'Default 0 disables.'
        ),
    )
    parser.add_argument(
        '--daily-memory-samples-per-lake',
        type=int,
        default=DEFAULT_DAILY_MEMORY_SAMPLES_PER_LAKE,
        help='Profile/no-profile days sampled per lake per epoch for daily-memory training. 0 uses all candidates.',
    )
    parser.add_argument(
        '--daily-memory-temporal-smoothness-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_TEMPORAL_SMOOTHNESS_WEIGHT,
        help='Inner weight for adjacent-day EOF coefficient smoothness in the daily-memory branch.',
    )
    parser.add_argument(
        '--daily-memory-heat-budget-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_HEAT_BUDGET_WEIGHT,
        help='Inner weight for external heat-budget closure on daily-memory no-profile/profile dates.',
    )
    parser.add_argument(
        '--daily-memory-physics-consistency-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_PHYSICS_CONSISTENCY_WEIGHT,
        help='Inner weight for one-step physics consistency from daily_memory(t) to daily_memory(t+1).',
    )
    parser.add_argument(
        '--daily-memory-regularization-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_REGULARIZATION_WEIGHT,
        help=(
            'Inner weight for daily-memory EOF coefficient shrinkage. '
            'Small values avoid collapsing the branch to the train-basis mean profile.'
        ),
    )
    parser.add_argument(
        '--daily-memory-coefficient-loss-weight',
        type=float,
        default=DEFAULT_DAILY_MEMORY_COEFFICIENT_LOSS_WEIGHT,
        help=(
            'Inner weight for supervised EOF/PCA coefficient loss on profile dates. '
            '0 disables direct coefficient supervision.'
        ),
    )
    parser.add_argument(
        '--daily-memory-start-epoch',
        type=int,
        default=DEFAULT_DAILY_MEMORY_START_EPOCH,
        help='Epoch where daily-memory branch training starts.',
    )
    parser.add_argument(
        '--daily-memory-ramp-epochs',
        type=int,
        default=DEFAULT_DAILY_MEMORY_RAMP_EPOCHS,
        help='Ramp epochs for daily-memory branch outer weight.',
    )
    parser.add_argument(
        '--daily-memory-hidden-dim',
        type=int,
        default=DEFAULT_DAILY_MEMORY_HIDDEN_DIM,
        help='Hidden dimension of the daily-memory EOF coefficient head.',
    )
    parser.add_argument(
        '--daily-memory-init-spread',
        type=float,
        default=DEFAULT_DAILY_MEMORY_INIT_SPREAD,
        help='Final-layer init std for the daily-memory head. 0 starts at the EOF/PCA mean profile.',
    )
    parser.add_argument(
        '--daily-memory-coeff-limit-sigma',
        type=float,
        default=DEFAULT_DAILY_MEMORY_COEFF_LIMIT_SIGMA,
        help='Bound for daily-memory EOF coefficients in train-basis coefficient standard deviations.',
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-mode',
        choices=sorted(THERMAL_STATE_PROFILE_FUSION_MODES),
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MODE,
        help=(
            'Optional low-dimensional profile correction for thermal state features. '
            'off preserves pure feature inference; init_only applies to physics-rollout '
            'initial state; daily_only applies to the daily-memory branch; both applies '
            'to both mainlines.'
        ),
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-time-policy',
        choices=sorted(THERMAL_STATE_PROFILE_FUSION_TIME_POLICIES),
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_TIME_POLICY,
        help=(
            'Profile selection policy for thermal-state fusion. nearest allows the '
            'closest past or future observed profile and should be reported as '
            'profile-conditioned reconstruction, not zero-profile forecasting. '
            'Default past_strict avoids same-day/future leakage.'
        ),
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-lookup-split',
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_LOOKUP_SPLIT,
        help=(
            'Profile lookup split for thermal-state fusion. Default train avoids '
            'validation/heldout leakage; all is only for explicit profile-conditioned diagnostics.'
        ),
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-max-age-days',
        type=int,
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_AGE_DAYS,
        help='Maximum absolute day distance from prediction date to a profile used for thermal-state fusion.',
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-min-depth-fraction',
        type=float,
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MIN_DEPTH_FRACTION,
        help='Minimum observed depth coverage fraction before profile-fusion gate reaches full coverage strength.',
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-max-weight',
        type=float,
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_MAX_WEIGHT,
        help='Maximum interpolation weight from feature-predicted thermal state toward the projected profile state.',
    )
    parser.add_argument(
        '--thermal-state-profile-fusion-coeff-limit-sigma',
        type=float,
        default=DEFAULT_THERMAL_STATE_PROFILE_FUSION_COEFF_LIMIT_SIGMA,
        help='Clamp projected profile EOF coefficients in train-basis coefficient standard deviations.',
    )
    parser.add_argument(
        '--prediction-branch',
        choices=sorted(PREDICTION_BRANCHES),
        default='physics_rollout',
        help=(
            'Export/eval prediction branch: physics_rollout or daily_memory. '
            'Use physics_rollout when ERA5/LST-driven physics should remain the product path.'
        ),
    )
    parser.add_argument(
        '--model-mainline',
        choices=sorted(MODEL_MAINLINE_MODES),
        default='auto',
        help=(
            'Explicit tenth-version mainline selector. auto resolves to one of the '
            'two prediction mainlines: init_physics_rollout or daily_memory. '
            'daily-memory can still be trained as an auxiliary head when '
            "prediction_branch='physics_rollout'."
        ),
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-mode',
        choices=list(MAINLINE_LSWT_OBSERVER_MODE_CHOICES),
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MODE,
        help='Tenth-version raw-open-water LSWT observer for zero-profile export diagnostics: off baseline, conservative_surface, or conservative_mld_shallow. Filled LST has zero strong-update gain.',
    )
    parser.add_argument(
        '--zero-profile-spinup-days-matrix',
        default=DEFAULT_ZERO_PROFILE_SPINUP_DAYS_MATRIX,
        help='Comma-separated spin-up day values for optional zero-profile validation matrix, e.g. 0,30,90,180.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-strength',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_STRENGTH,
        help='Gain for zero-profile LSWT observer diagnostics.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-decay-depth-m',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DECAY_DEPTH_M,
        help='Surface-update decay depth for zero-profile LSWT observer diagnostics.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-max-increment-c',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MAX_INCREMENT_C,
        help='Per-depth cap for zero-profile LSWT observer increments.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-deep-update-fraction',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_DEEP_UPDATE_FRACTION,
        help='Maximum low-rank observer vertical gain below the inferred mixed layer.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-heat-content-limit-c',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_HEAT_CONTENT_LIMIT_C,
        help='Area-weighted whole-column equivalent-C cap for each LSWT observer update.',
    )
    parser.add_argument(
        '--zero-profile-lswt-observer-min-quality',
        type=float,
        default=DEFAULT_ZERO_PROFILE_LSWT_OBSERVER_MIN_QUALITY,
        help='Minimum LST/LSWT quality gate for strong raw-open-water LSWT updates.',
    )
    parser.add_argument(
        '--kd-saturation-threshold',
        type=float,
        default=DEFAULT_KD_SATURATION_THRESHOLD,
        help='Threshold used to report nn_kd_multiplier_saturation_fraction.',
    )
    parser.add_argument(
        '--kd-saturation-penalty-weight',
        type=float,
        default=DEFAULT_KD_SATURATION_PENALTY_WEIGHT,
        help='Weak Kd binding guard: penalize nn_kd_multiplier above --kd-saturation-threshold. Default 0 disables.',
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
        help='Run lightweight train/val/heldout transition and rolling metrics every N epochs. Default: 50; use 0 to disable train-time eval.',
    )
    parser.add_argument(
        '--full-eval-every-epochs',
        type=int,
        default=None,
        help='Run full held-out free-roll and rolling-start evaluation every N epochs. Default: 60; use 0 to disable.',
    )
    parser.add_argument(
        '--full-eval-start-epoch',
        type=int,
        default=0,
        help='Do not run full evaluation before this 1-based epoch. Default 0 preserves existing scheduling.',
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
        '--training-amp',
        choices=sorted(TRAINING_AMP_MODES),
        default=DEFAULT_TRAINING_AMP,
        help=(
            'Opt-in mixed precision for neural submodules during training only. '
            'off preserves baseline numerics; bf16 is the preferred CUDA speed path; '
            'fp16 enables GradScaler.'
        ),
    )
    parser.add_argument(
        '--training-history-detail-every-epochs',
        type=int,
        default=DEFAULT_TRAINING_HISTORY_DETAIL_EVERY_EPOCHS,
        help=(
            'Scalarize detailed train-step diagnostics every N epochs outside eval/checkpoint/final epochs. '
            '1 preserves previous per-epoch detail history.'
        ),
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
        '--gpu-batch-autotune',
        choices=['off', 'on'],
        default=DEFAULT_GPU_BATCH_AUTOTUNE,
        help=(
            'Default off. When on, use larger compatible GPU batches without changing '
            'losses, samples, or physics weights.'
        ),
    )
    parser.add_argument(
        '--gpu-batch-autotune-target-batch-size',
        type=int,
        default=DEFAULT_GPU_BATCH_AUTOTUNE_TARGET_BATCH_SIZE,
        help='Target compatible training batch size used when --gpu-batch-autotune on.',
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
        choices=['profile', 'prior_spinup', 'zero_profile_low_dof'],
        default='profile',
        help='Held-out reconstruction initialization. profile falls back to prior_spinup when no profile exists.',
    )
    parser.add_argument('--spinup-days', type=int, default=90)
    parser.add_argument('--spinup-lst-assimilation-strength', type=float, default=0.08)
    parser.add_argument('--spinup-lst-assimilation-decay-depth-m', type=float, default=2.0)
    parser.add_argument('--spinup-lst-assimilation-max-increment-c', type=float, default=0.5)
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
    _suppress_removed_cli_flags(parser)
    _reject_removed_cli_flags(parser, argv)
    args = parser.parse_args(argv)
    train_multilake_state_forecaster(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        depth_points=args.depth_points,
        max_rollout_days=args.max_rollout_days,
        history_window_days=args.history_window_days,
        split_mode=args.split_mode,
        profile_supervision_scope=args.profile_supervision_scope,
        profile_loss_target_mode=args.profile_loss_target_mode,
        profile_sampling_mode=args.profile_sampling_mode,
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
        multitask_auxiliary_weight=args.multitask_auxiliary_weight,
        multitask_auxiliary_heat_weight=args.multitask_auxiliary_heat_weight,
        multitask_auxiliary_thermocline_weight=args.multitask_auxiliary_thermocline_weight,
        multitask_auxiliary_mld_weight=args.multitask_auxiliary_mld_weight,
        multitask_auxiliary_stability_weight=args.multitask_auxiliary_stability_weight,
        multitask_auxiliary_surface_bottom_weight=args.multitask_auxiliary_surface_bottom_weight,
        multitask_auxiliary_eof_weight=args.multitask_auxiliary_eof_weight,
        multitask_auxiliary_hidden_dim=args.multitask_auxiliary_hidden_dim,
        physics_rollout_thermal_state_weight=args.physics_rollout_thermal_state_weight,
        unlabeled_heat_closure_weight=args.unlabeled_heat_closure_weight,
        unlabeled_heat_closure_batch_size=args.unlabeled_heat_closure_batch_size,
        unlabeled_heat_closure_window_days=args.unlabeled_heat_closure_window_days,
        unlabeled_heat_closure_horizons=args.unlabeled_heat_closure_horizons,
        unlabeled_heat_closure_tau_wm2=args.unlabeled_heat_closure_tau_wm2,
        unlabeled_heat_closure_open_water_only=args.unlabeled_heat_closure_open_water_only,
        unlabeled_heat_closure_lst_qc_min=args.unlabeled_heat_closure_lst_qc_min,
        unlabeled_heat_closure_reservoir_mode=args.unlabeled_heat_closure_reservoir_mode,
        unlabeled_heat_closure_mode=args.unlabeled_heat_closure_mode,
        unlabeled_heat_closure_state_source=args.unlabeled_heat_closure_state_source,
        unlabeled_heat_closure_spinup_days=args.unlabeled_heat_closure_spinup_days,
        unlabeled_heat_closure_solver_guard_weight=args.unlabeled_heat_closure_solver_guard_weight,
        unlabeled_heat_closure_solver_guard_tau_wm2=args.unlabeled_heat_closure_solver_guard_tau_wm2,
        no_profile_lst_surface_weight=args.no_profile_lst_surface_weight,
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
        export_style_validation=args.export_style_validation,
        export_style_validation_max_lakes=args.export_style_validation_max_lakes,
        full_eval_point_diagnostics=args.full_eval_point_diagnostics,
        zero_profile_export_validation=args.zero_profile_export_validation,
        zero_profile_export_validation_max_lakes=args.zero_profile_export_validation_max_lakes,
        zero_profile_initializer=args.zero_profile_initializer,
        zero_profile_thermal_basis_components=args.zero_profile_thermal_basis_components,
        zero_profile_thermal_basis_grid_points=args.zero_profile_thermal_basis_grid_points,
        zero_profile_thermal_basis_balance_mode=args.zero_profile_thermal_basis_balance_mode,
        refit_zero_profile_thermal_basis=args.refit_zero_profile_thermal_basis,
        zero_profile_init_net_loss_weight=args.zero_profile_init_net_loss_weight,
        zero_profile_init_net_start_epoch=args.zero_profile_init_net_start_epoch,
        zero_profile_init_net_ramp_epochs=args.zero_profile_init_net_ramp_epochs,
        zero_profile_init_net_samples_per_lake=args.zero_profile_init_net_samples_per_lake,
        zero_profile_init_net_regularization_weight=args.zero_profile_init_net_regularization_weight,
        zero_profile_init_net_hidden_dim=args.zero_profile_init_net_hidden_dim,
        zero_profile_init_net_init_spread=args.zero_profile_init_net_init_spread,
        zero_profile_init_net_coeff_limit_sigma=args.zero_profile_init_net_coeff_limit_sigma,
        zero_profile_init_net_delta_limit_c=args.zero_profile_init_net_delta_limit_c,
        zero_profile_init_net_training_spinup_days=args.zero_profile_init_net_training_spinup_days,
        zero_profile_init_net_physics_weight=args.zero_profile_init_net_physics_weight,
        zero_profile_init_net_rollout_weight=args.zero_profile_init_net_rollout_weight,
        zero_profile_init_net_rollout_max_days=args.zero_profile_init_net_rollout_max_days,
        zero_profile_init_net_rollout_targets=args.zero_profile_init_net_rollout_targets,
        daily_memory_reconstruction_weight=args.daily_memory_reconstruction_weight,
        daily_memory_samples_per_lake=args.daily_memory_samples_per_lake,
        daily_memory_temporal_smoothness_weight=args.daily_memory_temporal_smoothness_weight,
        daily_memory_heat_budget_weight=args.daily_memory_heat_budget_weight,
        daily_memory_physics_consistency_weight=args.daily_memory_physics_consistency_weight,
        daily_memory_regularization_weight=args.daily_memory_regularization_weight,
        daily_memory_coefficient_loss_weight=args.daily_memory_coefficient_loss_weight,
        daily_memory_start_epoch=args.daily_memory_start_epoch,
        daily_memory_ramp_epochs=args.daily_memory_ramp_epochs,
        daily_memory_hidden_dim=args.daily_memory_hidden_dim,
        daily_memory_init_spread=args.daily_memory_init_spread,
        daily_memory_coeff_limit_sigma=args.daily_memory_coeff_limit_sigma,
        prediction_branch=args.prediction_branch,
        model_mainline=args.model_mainline,
        thermal_state_profile_fusion_mode=args.thermal_state_profile_fusion_mode,
        thermal_state_profile_fusion_time_policy=args.thermal_state_profile_fusion_time_policy,
        thermal_state_profile_fusion_lookup_split=args.thermal_state_profile_fusion_lookup_split,
        thermal_state_profile_fusion_max_age_days=args.thermal_state_profile_fusion_max_age_days,
        thermal_state_profile_fusion_min_depth_fraction=(
            args.thermal_state_profile_fusion_min_depth_fraction
        ),
        thermal_state_profile_fusion_max_weight=args.thermal_state_profile_fusion_max_weight,
        thermal_state_profile_fusion_coeff_limit_sigma=(
            args.thermal_state_profile_fusion_coeff_limit_sigma
        ),
        zero_profile_lswt_observer_mode=args.zero_profile_lswt_observer_mode,
        zero_profile_spinup_days_matrix=args.zero_profile_spinup_days_matrix,
        zero_profile_lswt_observer_strength=args.zero_profile_lswt_observer_strength,
        zero_profile_lswt_observer_decay_depth_m=args.zero_profile_lswt_observer_decay_depth_m,
        zero_profile_lswt_observer_max_increment_c=args.zero_profile_lswt_observer_max_increment_c,
        zero_profile_lswt_observer_deep_update_fraction=(
            args.zero_profile_lswt_observer_deep_update_fraction
        ),
        zero_profile_lswt_observer_heat_content_limit_c=(
            args.zero_profile_lswt_observer_heat_content_limit_c
        ),
        zero_profile_lswt_observer_min_quality=args.zero_profile_lswt_observer_min_quality,
        kd_saturation_threshold=args.kd_saturation_threshold,
        kd_saturation_penalty_weight=args.kd_saturation_penalty_weight,
        init_mode=args.init_mode,
        spinup_days=args.spinup_days,
        spinup_lst_assimilation_strength=args.spinup_lst_assimilation_strength,
        spinup_lst_assimilation_decay_depth_m=args.spinup_lst_assimilation_decay_depth_m,
        spinup_lst_assimilation_max_increment_c=args.spinup_lst_assimilation_max_increment_c,
        rollout_start_date=getattr(args, 'rollout_start_date', None),
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        eval_every_epochs=args.eval_every_epochs,
        full_eval_every_epochs=args.full_eval_every_epochs,
        full_eval_start_epoch=args.full_eval_start_epoch,
        profile_runtime=args.profile_runtime,
        profile_gpu=args.profile_gpu,
        history_diagnostic_every_epochs=args.history_diagnostic_every_epochs,
        torch_tf32=args.torch_tf32,
        torch_matmul_precision=args.torch_matmul_precision,
        training_amp=args.training_amp,
        training_history_detail_every_epochs=args.training_history_detail_every_epochs,
        transition_batch_size=args.transition_batch_size,
        segment_rollout_batch_size=args.segment_rollout_batch_size,
        rolling_horizon_batch_size=args.rolling_horizon_batch_size,
        train_diagnostic_mode=args.train_diagnostic_mode,
        export_after_training=args.export_after_training,
        export_max_depth_m=args.export_max_depth_m,
        cross_lake_batch_mode=args.cross_lake_batch_mode,
        cross_lake_batch_size=args.cross_lake_batch_size,
        gpu_batch_autotune=args.gpu_batch_autotune,
        gpu_batch_autotune_target_batch_size=args.gpu_batch_autotune_target_batch_size,
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
