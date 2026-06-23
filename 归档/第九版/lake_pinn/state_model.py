"""Shared LakePINN state-space model core for multi-lake reconstruction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    DIFFUSIVITY_K0,
    DIFFUSIVITY_RI_SENSITIVITY,
    DIFFUSIVITY_UNSTABLE_BOOST,
    DIFFUSIVITY_UNSTABLE_GRADIENT_EPS,
    DIFFUSIVITY_UNSTABLE_GRADIENT_WIDTH,
    DIFFUSIVITY_WIND_DECAY_DEPTH,
    GRAVITY,
    MAX_SHORTWAVE_ATTENUATION,
    MAX_TOTAL_DIFFUSIVITY,
    MIN_SHORTWAVE_ATTENUATION,
    MIN_TOTAL_DIFFUSIVITY,
    MOLECULAR_DIFFUSIVITY,
    PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
    PINN_INFLOW_REFERENCE_M3_S,
    PINN_LOG_AREA_REFERENCE_KM2,
    PINN_MAX_DEPTH_REFERENCE_M,
    PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
    PINN_MAX_MEAN_DEPTH_REFERENCE_M,
    PINN_MAX_SECCHI_REFERENCE_M,
    PINN_MAX_TEMPERATURE_REFERENCE_C,
    PINN_VOLUME_REFERENCE_KM3,
    RHO_CP,
    RI_WIND_SHEAR_FACTOR,
    SECONDS_PER_DAY,
    SHORTWAVE_SURFACE_FRACTION,
    SURFACE_ALBEDO_ICE,
    SURFACE_ALBEDO_WATER,
    WATER_DENSITY,
)
from .data_io import normalize_task_mode
from .lake_metadata import metadata_static_features
from .physics import compute_surface_flux_terms, normalize_turbulent_flux_mode, water_density_torch
from .vertical_solver import implicit_diffusion_step, layer_thicknesses, one_day_heat_sources

FORCING_FEATURE_COLUMNS = (
    'doy_sin',
    'doy_cos',
    'T_air_C',
    'wind_speed_m_per_s',
    'Solar_W_m2',
    'LST_surface_C',
    'LST_quality_factor',
    'LST_is_filled',
    'LST_observed_flag',
    'Longwave_W_m2',
    'latent_heat_upward_W_m2',
    'sensible_heat_upward_W_m2',
    'relative_humidity',
    'surface_pressure_Pa',
    'Secchi_m',
    'light_extinction_kd',
    'effective_fetch',
    'air_temp_mean_7d',
    'air_temp_mean_30d',
    'shortwave_sum_7d',
    'shortwave_sum_30d',
    'wind_mean_7d',
    'lst_mean_7d',
    'heating_degree_days_30d',
    'ice_fraction',
    'water_level_anomaly',
    'net_inflow',
)
FORCING_FEATURE_INDEX = {key: idx for idx, key in enumerate(FORCING_FEATURE_COLUMNS)}

STATIC_FEATURE_KEYS = (
    'max_depth_norm',
    'mean_depth_norm',
    'log_area',
    'latitude',
    'longitude',
    'elevation_norm',
    'volume_norm',
    'light_extinction_norm',
    'fetch_norm',
    'wind_exposure_norm',
    'basin_shape_norm',
    'reservoir_indicator',
    'residence_time_norm',
    'shoreline_length_norm',
    'shoreline_development_norm',
    'catchment_area_norm',
    'discharge_norm',
)
STATIC_FEATURE_DIM = len(STATIC_FEATURE_KEYS)

HARD_DENSITY_STABILITY_MODES = {'auto', 'on', 'off'}
FREEZING_ENERGY_MODES = {'latent_reservoir', 'clamp'}
SHAPE_AWARE_MIXING_MODES = {'on', 'off'}
LAKE_ADAPTIVE_TEMPORAL_MODES = {'off', 'seasonal_forcing'}
ADVECTIVE_HEAT_SOURCE_MODES = {'off', 'reservoir_simple'}
ICE_SHORTWAVE_ATTENUATION_M_INV = 1.50
SNOW_SHORTWAVE_ATTENUATION_M_INV = 20.0
DEFAULT_KD_BASE_M_INV = 0.45


class ForcingBatch:
    """Lightweight mapping wrapper for batched forcing tensor views."""

    __slots__ = ('data',)

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()


def _coerce_forcing_batch(row):
    if isinstance(row, ForcingBatch):
        return row
    return ForcingBatch(row)


def resolve_hard_density_stability(mode='auto', *, task_mode='analysis', data_fill_mode='reconstruction'):
    """Resolve the hard-density switch to a concrete runtime boolean."""
    mode = str(mode or 'auto').strip().lower()
    if mode not in HARD_DENSITY_STABILITY_MODES:
        raise ValueError("hard_density_stability must be one of: auto, on, off.")
    if mode == 'on':
        return True
    if mode == 'off':
        return False
    normalize_task_mode(task_mode)
    return str(data_fill_mode or '').strip().lower() == 'reconstruction'


def normalize_freezing_energy_mode(mode='latent_reservoir'):
    mode = str(mode or 'latent_reservoir').strip().lower().replace('-', '_')
    if mode not in FREEZING_ENERGY_MODES:
        raise ValueError("freezing_energy_mode must be one of: latent_reservoir, clamp.")
    return mode


def normalize_shape_aware_mixing(mode='on'):
    mode = str(mode or 'on').strip().lower()
    if mode not in SHAPE_AWARE_MIXING_MODES:
        raise ValueError("shape_aware_mixing must be one of: on, off.")
    return mode


def normalize_lake_adaptive_temporal_mode(mode='off'):
    mode = str(mode or 'off').strip().lower().replace('-', '_')
    aliases = {
        'false': 'off',
        'no': 'off',
        'none': 'off',
        '0': 'off',
        'on': 'seasonal_forcing',
        'seasonal': 'seasonal_forcing',
        'forcing': 'seasonal_forcing',
        'seasonal_forcing': 'seasonal_forcing',
    }
    mode = aliases.get(mode, mode)
    if mode not in LAKE_ADAPTIVE_TEMPORAL_MODES:
        raise ValueError("lake_adaptive_temporal_mode must be one of: off, seasonal_forcing.")
    return mode


def normalize_advective_heat_source_mode(mode='reservoir_simple'):
    mode = str(mode or 'reservoir_simple').strip().lower().replace('-', '_')
    if mode not in ADVECTIVE_HEAT_SOURCE_MODES:
        raise ValueError("advective_heat_source_mode must be one of: off, reservoir_simple.")
    return mode


def _coerce_freezing_storage(freezing_storage_j_m2, temperature, *, surface_only=False):
    if freezing_storage_j_m2 is None:
        storage = torch.zeros_like(temperature)
        return storage
    storage = torch.as_tensor(
        freezing_storage_j_m2,
        device=temperature.device,
        dtype=temperature.dtype,
    )
    if storage.ndim == 1:
        storage = storage.unsqueeze(0)
    if storage.shape[0] == 1 and temperature.shape[0] > 1:
        storage = storage.expand_as(temperature)
    if storage.shape != temperature.shape:
        raise ValueError(
            "freezing_storage_j_m2 must have shape [depth] or [batch, depth] matching temperature."
        )
    storage = torch.clamp(storage, min=0.0)
    if surface_only:
        surface_storage = torch.zeros_like(storage)
        surface_storage[:, :1] = storage.sum(dim=1, keepdim=True)
        return surface_storage
    return storage


def _apply_latent_reservoir_floor(raw_temperature, freezing_storage_j_m2, layer_heat_capacity_j_m2_c):
    """Project temperature to non-negative values with a surface ice/cold reservoir."""
    capacity = torch.clamp(
        layer_heat_capacity_j_m2_c.to(device=raw_temperature.device, dtype=raw_temperature.dtype),
        min=1.0e-12,
    ).reshape(1, -1)
    positive_energy = torch.clamp(raw_temperature, min=0.0) * capacity
    cold_deficit = torch.clamp(-raw_temperature, min=0.0) * capacity
    surface_storage_total = freezing_storage_j_m2.sum(dim=1) + cold_deficit.sum(dim=1)
    surface_positive_energy = positive_energy[:, 0]
    melt_energy = torch.minimum(surface_positive_energy, surface_storage_total)
    next_storage = torch.zeros_like(freezing_storage_j_m2)
    next_storage[:, 0] = torch.clamp(
        surface_storage_total - melt_energy,
        min=0.0,
    )
    next_temperature = positive_energy / capacity
    next_temperature[:, 0] = (surface_positive_energy - melt_energy) / capacity[:, 0]
    return next_temperature, next_storage


def remove_area_weighted_mean(residual, depths, area_profile):
    """Remove residual heat-content change while preserving profile-shape fixes."""
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)
    depths = depths.flatten().to(device=residual.device, dtype=residual.dtype)
    area = area_profile.flatten().to(device=residual.device, dtype=residual.dtype)
    dz = layer_thicknesses(depths)
    weights = torch.clamp(area.reshape(1, -1) * dz.reshape(1, -1), min=1.0e-8)
    mean = torch.sum(residual * weights, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)
    return residual - mean


def _convective_adjustment_row(temperature_row, weights, density_tolerance_kgm3):
    block_temps = [temperature_row[idx] for idx in range(temperature_row.numel())]
    block_weights = [weights[idx] for idx in range(weights.numel())]
    block_indices = [[idx] for idx in range(temperature_row.numel())]
    idx = 0
    while idx < len(block_temps) - 1:
        upper_rho = water_density_torch(block_temps[idx].reshape(1))[0]
        lower_rho = water_density_torch(block_temps[idx + 1].reshape(1))[0]
        if float((upper_rho - lower_rho).detach().cpu()) > float(density_tolerance_kgm3):
            merged_weight = block_weights[idx] + block_weights[idx + 1]
            merged_temp = (
                block_temps[idx] * block_weights[idx]
                + block_temps[idx + 1] * block_weights[idx + 1]
            ) / torch.clamp(merged_weight, min=1.0e-12)
            block_temps[idx] = merged_temp
            block_weights[idx] = merged_weight
            block_indices[idx] = block_indices[idx] + block_indices[idx + 1]
            del block_temps[idx + 1]
            del block_weights[idx + 1]
            del block_indices[idx + 1]
            idx = max(0, idx - 1)
        else:
            idx += 1
    adjusted_values = [None] * temperature_row.numel()
    for temp, indices in zip(block_temps, block_indices):
        for layer_idx in indices:
            adjusted_values[layer_idx] = temp
    return torch.stack(adjusted_values)


def heat_conserving_convective_adjustment(
    temperature,
    depths,
    area_profile,
    *,
    density_tolerance_kgm3=0.0,
):
    """Mix statically unstable adjacent layers while conserving heat content."""
    input_was_1d = temperature.ndim == 1
    if input_was_1d:
        temperature = temperature.unsqueeze(0)
    depths = depths.flatten().to(device=temperature.device, dtype=temperature.dtype)
    area = area_profile.flatten().to(device=temperature.device, dtype=temperature.dtype)
    dz = layer_thicknesses(depths)
    weights = torch.clamp(area * dz, min=1.0e-8)
    if temperature.shape[1] < 2:
        adjusted = temperature
    else:
        adjusted = torch.stack([
            _convective_adjustment_row(row, weights, density_tolerance_kgm3)
            for row in temperature
        ])
    delta = adjusted - temperature
    surface_area = torch.clamp(area[0], min=1.0e-6)
    heat_delta_j_m2 = (
        RHO_CP
        * (
            torch.sum(adjusted.to(torch.float64) * weights.to(torch.float64).reshape(1, -1), dim=1)
            - torch.sum(temperature.to(torch.float64) * weights.to(torch.float64).reshape(1, -1), dim=1)
        )
        / surface_area.to(torch.float64)
    ).to(dtype=temperature.dtype)
    max_delta_c = torch.amax(torch.abs(delta), dim=1)
    applied = (max_delta_c > 1.0e-6).to(dtype=temperature.dtype)
    diagnostics = {
        'density_adjustment_applied': applied,
        'density_adjustment_max_delta_c': max_delta_c,
        'density_adjustment_heat_delta_j_m2': heat_delta_j_m2,
    }
    if input_was_1d:
        return adjusted.squeeze(0), diagnostics
    return adjusted, diagnostics


def ice_conductive_flux_wm2(
    water_interface_temp_c,
    ice_skin_temp_c,
    snow_depth_m=None,
    ice_thickness_m=None,
    k_snow=0.30,
    k_ice=2.10,
):
    """Approximate ice/snow conductive flux, positive into the water column."""
    water_interface_temp_c = water_interface_temp_c.reshape(-1)
    ice_skin_temp_c = ice_skin_temp_c.to(
        device=water_interface_temp_c.device,
        dtype=water_interface_temp_c.dtype,
    ).reshape(-1)
    if ice_skin_temp_c.numel() == 1 and water_interface_temp_c.numel() > 1:
        ice_skin_temp_c = ice_skin_temp_c.expand_as(water_interface_temp_c)
    if snow_depth_m is None:
        snow_depth_m = torch.zeros_like(water_interface_temp_c)
    else:
        snow_depth_m = snow_depth_m.to(
            device=water_interface_temp_c.device,
            dtype=water_interface_temp_c.dtype,
        ).reshape(-1)
    if ice_thickness_m is None:
        ice_thickness_m = torch.full_like(water_interface_temp_c, 0.10)
    else:
        ice_thickness_m = ice_thickness_m.to(
            device=water_interface_temp_c.device,
            dtype=water_interface_temp_c.dtype,
        ).reshape(-1)
    if snow_depth_m.numel() == 1 and water_interface_temp_c.numel() > 1:
        snow_depth_m = snow_depth_m.expand_as(water_interface_temp_c)
    if ice_thickness_m.numel() == 1 and water_interface_temp_c.numel() > 1:
        ice_thickness_m = ice_thickness_m.expand_as(water_interface_temp_c)
    resistance = torch.clamp(
        torch.clamp(snow_depth_m, min=0.0) / float(k_snow)
        + torch.clamp(ice_thickness_m, min=0.0) / float(k_ice),
        min=0.02,
    )
    # If the ice/snow skin is colder than the water interface, heat leaves the
    # water column, so the flux into water is negative.
    return (ice_skin_temp_c - water_interface_temp_c) / resistance


def static_feature_array(metadata, max_depth):
    static = metadata_static_features(metadata, max_depth=max_depth)
    return np.asarray([float(static[key]) for key in STATIC_FEATURE_KEYS], dtype=np.float32)


def _average_forcing_rows(current_row, next_row=None):
    """Use interval-average forcing for one state update."""
    return _average_forcing_rows_for_task(current_row, next_row=next_row, task_mode='analysis')


def _average_forcing_rows_for_task(current_row, next_row=None, task_mode='analysis'):
    """Use interval-average forcing for one reconstruction state update."""
    current_row = _coerce_forcing_batch(current_row)
    if next_row is None:
        return current_row
    next_row = _coerce_forcing_batch(next_row)
    normalize_task_mode(task_mode)
    averaged = {}
    nan_aware_keys = {'lswt_open_water', 'ist_snow_ice', 'lst_quality', 'ice_fraction'}
    for key, value in current_row.items():
        if key in next_row:
            next_value = next_row[key].to(device=value.device, dtype=value.dtype)
            if key in nan_aware_keys:
                stacked = torch.stack([value, next_value], dim=0)
                finite = torch.isfinite(stacked)
                averaged[key] = torch.where(
                    finite.any(dim=0),
                    torch.nan_to_num(stacked, nan=0.0).sum(dim=0)
                    / finite.sum(dim=0).clamp_min(1),
                    value,
                )
            else:
                averaged[key] = 0.5 * (value + next_value)
        else:
            averaged[key] = value
    return ForcingBatch(averaged)


class ForcingHistoryEncoder(nn.Module):
    """Encode past-only forcing windows with a lightweight GRU."""

    def __init__(self, forcing_dim, hidden_dim=48, output_dim=48):
        super().__init__()
        self.gru = nn.GRU(int(forcing_dim), int(hidden_dim), batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.SiLU(),
        )

    def forward(self, forcing_history):
        if forcing_history.ndim == 2:
            forcing_history = forcing_history.unsqueeze(0)
        _, hidden = self.gru(forcing_history)
        return self.proj(hidden[-1])


class StateParameterNet(nn.Module):
    """Predict Kz, Kd, and a bounded daily residual correction.

    Lake attributes enter through a small FiLM adapter.  This keeps the shared
    trunk global while allowing geometry/optics/exposure to modulate physical
    parameters without directly rewriting the temperature profile.
    """

    def __init__(self, forcing_dim, static_dim, hidden_dim=96, residual_limit_c=0.50):
        super().__init__()
        self.residual_limit_c = float(residual_limit_c)
        input_dim = 1 + forcing_dim + static_dim
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.lake_adapter = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, depth_norm, forcing_context, static_features):
        batch_size = forcing_context.shape[0]
        n_depths = depth_norm.shape[0]
        z = depth_norm.reshape(1, n_depths, 1).expand(batch_size, n_depths, 1)
        forcing = forcing_context.reshape(batch_size, 1, -1).expand(batch_size, n_depths, -1)
        static = static_features.reshape(batch_size, 1, -1).expand(batch_size, n_depths, -1)
        hidden = self.input(torch.cat([z, forcing, static], dim=-1))
        gamma, beta = self.lake_adapter(static_features).chunk(2, dim=-1)
        gamma = 0.20 * torch.tanh(gamma).reshape(batch_size, 1, -1)
        beta = 0.20 * torch.tanh(beta).reshape(batch_size, 1, -1)
        hidden = hidden * (1.0 + gamma) + beta
        hidden = self.hidden(hidden)
        raw = self.head(hidden)
        background_nn_kz = 2.0e-6 * torch.sigmoid(raw[..., 0])
        turbulent_nn_kz = 4.0e-5 * F.softplus(raw[..., 1])
        nn_kd_multiplier = torch.exp(0.5 * torch.tanh(raw[..., 2])).mean(dim=1)
        residual = self.residual_limit_c * torch.tanh(raw[..., 3])
        return background_nn_kz, turbulent_nn_kz, nn_kd_multiplier, residual


class PhysicalScaleHead(nn.Module):
    """Learn bounded lake/season/forcing-conditioned physical coefficients."""

    def __init__(
        self,
        forcing_context_dim,
        static_dim,
        forcing_dim,
        hidden_dim=64,
        shortwave_bounds=(0.85, 1.30),
        cooling_bounds=(0.90, 1.40),
        flux_bias_bounds=(-30.0, 30.0),
    ):
        super().__init__()
        self.shortwave_min, self.shortwave_max = map(float, shortwave_bounds)
        self.cooling_min, self.cooling_max = map(float, cooling_bounds)
        self.flux_bias_min, self.flux_bias_max = map(float, flux_bias_bounds)
        input_dim = int(forcing_context_dim) + int(static_dim) + int(forcing_dim) + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self._initialize_near_unity()

    @staticmethod
    def _bounded(raw, lower, upper):
        return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)

    @staticmethod
    def _logit_for_value(value, lower, upper):
        ratio = (float(value) - float(lower)) / max(float(upper) - float(lower), 1.0e-8)
        ratio = min(max(ratio, 1.0e-4), 1.0 - 1.0e-4)
        return float(np.log(ratio / (1.0 - ratio)))

    def _initialize_near_unity(self):
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        final.bias.data[0] = self._logit_for_value(1.0, self.shortwave_min, self.shortwave_max)
        final.bias.data[1] = self._logit_for_value(1.0, self.cooling_min, self.cooling_max)
        final.bias.data[2] = self._logit_for_value(0.0, self.flux_bias_min, self.flux_bias_max)

    def forward(self, forcing_context, static_features, forcing_features, raw_surface_flux_wm2, ice_fraction):
        if forcing_context.ndim == 1:
            forcing_context = forcing_context.unsqueeze(0)
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        if forcing_features.ndim == 1:
            forcing_features = forcing_features.unsqueeze(0)
        batch_size = forcing_context.shape[0]
        if static_features.shape[0] == 1 and batch_size > 1:
            static_features = static_features.expand(batch_size, -1)
        if forcing_features.shape[0] == 1 and batch_size > 1:
            forcing_features = forcing_features.expand(batch_size, -1)
        raw_surface_flux_wm2 = raw_surface_flux_wm2.reshape(-1, 1).to(
            device=forcing_context.device,
            dtype=forcing_context.dtype,
        )
        if raw_surface_flux_wm2.shape[0] == 1 and batch_size > 1:
            raw_surface_flux_wm2 = raw_surface_flux_wm2.expand(batch_size, 1)
        flux_norm = torch.clamp(
            raw_surface_flux_wm2 / float(PINN_MAX_HEAT_FLUX_REFERENCE_W_M2),
            min=-3.0,
            max=3.0,
        )
        ice_fraction = ice_fraction.reshape(-1, 1).to(
            device=forcing_context.device,
            dtype=forcing_context.dtype,
        )
        if ice_fraction.shape[0] == 1 and batch_size > 1:
            ice_fraction = ice_fraction.expand(batch_size, 1)
        x = torch.cat([forcing_context, static_features, forcing_features, flux_norm, ice_fraction], dim=1)
        raw = self.net(x)
        shortwave_scale = self._bounded(raw[:, 0], self.shortwave_min, self.shortwave_max)
        cooling_scale = self._bounded(raw[:, 1], self.cooling_min, self.cooling_max)
        flux_bias_wm2 = self._bounded(raw[:, 2], self.flux_bias_min, self.flux_bias_max)
        return shortwave_scale, cooling_scale, flux_bias_wm2


LAKE_ADAPTIVE_PARAM_ORDER = ('kz', 'flux', 'kd', 'exchange', 'convective', 'ice')
LAKE_ADAPTIVE_PARAM_MODES = {'off', 'both', 'all', *LAKE_ADAPTIVE_PARAM_ORDER}
LAKE_ADAPTIVE_OUTPUT_KEYS = (
    'wind_kz_scale',
    'blend_alpha',
    'kd_multiplier',
    'turbulent_exchange_scale',
    'convective_mixing_scale',
    'ice_shortwave_scale',
)
LAKE_ADAPTIVE_MODE_TO_OUTPUT = {
    'kz': 'wind_kz_scale',
    'flux': 'blend_alpha',
    'kd': 'kd_multiplier',
    'exchange': 'turbulent_exchange_scale',
    'convective': 'convective_mixing_scale',
    'ice': 'ice_shortwave_scale',
}


def normalize_lake_adaptive_params(mode):
    if isinstance(mode, (list, tuple, set)):
        raw_parts = [str(part).strip().lower() for part in mode]
    else:
        raw_parts = [
            part.strip().lower()
            for part in str(mode or 'off').replace(';', ',').split(',')
        ]
    parts = [part for part in raw_parts if part]
    if not parts or parts == ['off']:
        return 'off'
    expanded = set()
    for part in parts:
        if part == 'off':
            if len(parts) > 1:
                raise ValueError("lake_adaptive_params='off' cannot be combined with other modes.")
            return 'off'
        if part == 'both':
            expanded.update(('kz', 'flux'))
            continue
        if part == 'all':
            expanded.update(LAKE_ADAPTIVE_PARAM_ORDER)
            continue
        if part not in LAKE_ADAPTIVE_PARAM_MODES:
            raise ValueError(
                "lake_adaptive_params must use: off, kz, flux, kd, exchange, "
                "convective, ice, both, all."
            )
        expanded.add(part)
    return ','.join(mode for mode in LAKE_ADAPTIVE_PARAM_ORDER if mode in expanded)


def lake_adaptive_param_set(mode):
    normalized = normalize_lake_adaptive_params(mode)
    if normalized == 'off':
        return set()
    return set(normalized.split(','))


class LakeAdaptiveParameterHead(nn.Module):
    """Map lake metadata features to bounded physical parameter multipliers."""

    def __init__(
        self,
        static_dim,
        hidden_dim=64,
        init_spread=0.02,
        wind_kz_bounds=(0.4, 3.0),
        blend_alpha_bounds=(0.0, 0.6),
        kd_multiplier_bounds=(0.4, 2.0),
        exchange_scale_bounds=(0.5, 1.8),
        convective_scale_bounds=(0.3, 2.5),
        ice_shortwave_scale_bounds=(0.4, 1.8),
        base_wind_kz_scale=1.0,
        base_blend_alpha=0.3,
        base_kd_multiplier=1.0,
        base_exchange_scale=1.0,
        base_convective_scale=1.0,
        base_ice_shortwave_scale=1.0,
    ):
        super().__init__()
        self.bounds = {
            'wind_kz_scale': tuple(map(float, wind_kz_bounds)),
            'blend_alpha': tuple(map(float, blend_alpha_bounds)),
            'kd_multiplier': tuple(map(float, kd_multiplier_bounds)),
            'turbulent_exchange_scale': tuple(map(float, exchange_scale_bounds)),
            'convective_mixing_scale': tuple(map(float, convective_scale_bounds)),
            'ice_shortwave_scale': tuple(map(float, ice_shortwave_scale_bounds)),
        }
        for key, (lower, upper) in self.bounds.items():
            if upper <= lower:
                raise ValueError(f'adaptive {key} max must be greater than min.')
        self.hidden_dim = int(hidden_dim)
        self.init_spread = float(init_spread)
        if self.hidden_dim <= 0:
            raise ValueError('lake adaptive hidden_dim must be positive.')
        if self.init_spread < 0.0:
            raise ValueError('lake adaptive init_spread must be non-negative.')
        base_values = {
            'wind_kz_scale': base_wind_kz_scale,
            'blend_alpha': base_blend_alpha,
            'kd_multiplier': base_kd_multiplier,
            'turbulent_exchange_scale': base_exchange_scale,
            'convective_mixing_scale': base_convective_scale,
            'ice_shortwave_scale': base_ice_shortwave_scale,
        }
        self.base_values = {
            key: float(np.clip(base_values[key], *self.bounds[key]))
            for key in LAKE_ADAPTIVE_OUTPUT_KEYS
        }
        self.net = nn.Sequential(
            nn.Linear(int(static_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(LAKE_ADAPTIVE_OUTPUT_KEYS)),
        )
        self._initialize_near_base()

    @staticmethod
    def _bounded(raw, lower, upper):
        return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)

    @staticmethod
    def _logit_for_value(value, lower, upper):
        ratio = (float(value) - float(lower)) / max(float(upper) - float(lower), 1.0e-8)
        ratio = min(max(ratio, 1.0e-4), 1.0 - 1.0e-4)
        return float(np.log(ratio / (1.0 - ratio)))

    def _initialize_near_base(self):
        final = self.net[-1]
        if self.init_spread > 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=self.init_spread)
        else:
            nn.init.zeros_(final.weight)
        for idx, key in enumerate(LAKE_ADAPTIVE_OUTPUT_KEYS):
            lower, upper = self.bounds[key]
            final.bias.data[idx] = self._logit_for_value(self.base_values[key], lower, upper)

    def forward(self, static_features):
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        raw = self.net(static_features)
        return {
            key: self._bounded(raw[:, idx], *self.bounds[key])
            for idx, key in enumerate(LAKE_ADAPTIVE_OUTPUT_KEYS)
        }


class LakeAdaptiveTemporalDeltaHead(nn.Module):
    """Small context head that lets adaptive physical parameters vary by season and forcing."""

    def __init__(
        self,
        static_dim,
        forcing_context_dim,
        forcing_dim,
        hidden_dim=64,
        init_spread=0.005,
    ):
        super().__init__()
        self.static_dim = int(static_dim)
        self.forcing_context_dim = int(forcing_context_dim)
        self.forcing_dim = int(forcing_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_spread = float(init_spread)
        if self.hidden_dim <= 0:
            raise ValueError('lake adaptive temporal hidden_dim must be positive.')
        if self.init_spread < 0.0:
            raise ValueError('lake adaptive temporal init_spread must be non-negative.')
        input_dim = self.static_dim + self.forcing_context_dim + self.forcing_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(LAKE_ADAPTIVE_OUTPUT_KEYS)),
        )
        self._initialize_near_zero()

    def _initialize_near_zero(self):
        final = self.net[-1]
        if self.init_spread > 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=self.init_spread)
        else:
            nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, static_features, forcing_context, forcing_features):
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        if forcing_context.ndim == 1:
            forcing_context = forcing_context.unsqueeze(0)
        if forcing_features.ndim == 1:
            forcing_features = forcing_features.unsqueeze(0)
        batch_size = static_features.shape[0]
        if forcing_context.shape[0] == 1 and batch_size > 1:
            forcing_context = forcing_context.expand(batch_size, -1)
        if forcing_features.shape[0] == 1 and batch_size > 1:
            forcing_features = forcing_features.expand(batch_size, -1)
        inputs = torch.cat((static_features, forcing_context, forcing_features), dim=1)
        raw = self.net(inputs)
        return {
            key: raw[:, idx]
            for idx, key in enumerate(LAKE_ADAPTIVE_OUTPUT_KEYS)
        }


def _resize_depth_vector(values, target_count):
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError('target depth count must be positive.')
    source_count = int(values.shape[-1])
    if source_count == target_count:
        return values
    original_shape = tuple(values.shape)
    flat = values.reshape(-1, 1, source_count)
    if source_count == 1:
        resized = flat.expand(-1, -1, target_count)
    else:
        resized = F.interpolate(flat, size=target_count, mode='linear', align_corners=True)
    return resized.reshape(*original_shape[:-1], target_count)


def _resize_support_profiles_and_masks(support_profiles, support_masks, target_count):
    target_count = int(target_count)
    if int(support_profiles.shape[-1]) == target_count:
        return support_profiles, support_masks
    if support_masks is None:
        support_masks = torch.ones_like(support_profiles)
    support_masks = support_masks.to(device=support_profiles.device, dtype=support_profiles.dtype)
    valid = torch.isfinite(support_profiles) & (support_masks > 0.5)
    valid_weight = valid.to(dtype=support_profiles.dtype)
    filled = torch.where(valid, support_profiles, torch.zeros_like(support_profiles))
    weighted_values = _resize_depth_vector(filled * valid_weight, target_count)
    weights = _resize_depth_vector(valid_weight, target_count)
    resized_profiles = weighted_values / torch.clamp(weights, min=1.0e-6)
    resized_masks = weights > 0.25
    resized_profiles = torch.where(
        resized_masks,
        resized_profiles,
        torch.zeros_like(resized_profiles),
    )
    return resized_profiles, resized_masks.to(dtype=support_masks.dtype)


class FewShotStateEncoder(nn.Module):
    """Encode support profiles into a bounded initial-state and adapter update."""

    def __init__(
        self,
        depth_count,
        static_dim,
        forcing_context_dim,
        hidden_dim=64,
        init_spread=0.005,
        initial_delta_limit_c=2.0,
        unobserved_delta_scale=1.0,
    ):
        super().__init__()
        self.depth_count = int(depth_count)
        self.static_dim = int(static_dim)
        self.forcing_context_dim = int(forcing_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_spread = float(init_spread)
        self.initial_delta_limit_c = float(initial_delta_limit_c)
        self.unobserved_delta_scale = float(unobserved_delta_scale)
        if self.depth_count <= 0:
            raise ValueError('few-shot depth_count must be positive.')
        if self.hidden_dim <= 0:
            raise ValueError('few-shot hidden_dim must be positive.')
        if self.init_spread < 0.0:
            raise ValueError('few-shot init_spread must be non-negative.')
        if self.initial_delta_limit_c < 0.0:
            raise ValueError('few-shot initial_delta_limit_c must be non-negative.')
        if not (0.0 <= self.unobserved_delta_scale <= 1.0):
            raise ValueError('few-shot unobserved_delta_scale must be between 0.0 and 1.0.')
        input_dim = self.depth_count * 2 + 4 + self.static_dim + self.forcing_context_dim
        output_dim = self.depth_count + len(LAKE_ADAPTIVE_OUTPUT_KEYS)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )
        self._initialize_near_zero()

    def _initialize_near_zero(self):
        final = self.net[-1]
        if self.init_spread > 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=self.init_spread)
        else:
            nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, support_profiles, support_masks, support_ages_days, static_features, forcing_context):
        if support_profiles.ndim == 2:
            support_profiles = support_profiles.unsqueeze(0)
        if support_masks.ndim == 2:
            support_masks = support_masks.unsqueeze(0)
        if support_ages_days.ndim == 1:
            support_ages_days = support_ages_days.unsqueeze(0)
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        if forcing_context.ndim == 1:
            forcing_context = forcing_context.unsqueeze(0)
        batch_size = support_profiles.shape[0]
        support_profiles = support_profiles.to(device=static_features.device, dtype=static_features.dtype)
        support_masks = support_masks.to(device=static_features.device, dtype=static_features.dtype)
        support_ages_days = support_ages_days.to(device=static_features.device, dtype=static_features.dtype)
        if static_features.shape[0] == 1 and batch_size > 1:
            static_features = static_features.expand(batch_size, -1)
        if forcing_context.shape[0] == 1 and batch_size > 1:
            forcing_context = forcing_context.expand(batch_size, -1)

        finite_mask = torch.isfinite(support_profiles) & (support_masks > 0.5)
        valid = finite_mask.to(dtype=static_features.dtype)
        filled_profiles = torch.where(finite_mask, support_profiles, torch.zeros_like(support_profiles))
        recency_weight = torch.exp(-torch.clamp(support_ages_days, min=0.0) / 90.0).reshape(batch_size, -1, 1)
        weighted_valid = valid * recency_weight
        denom = torch.clamp(weighted_valid.sum(dim=1), min=1.0e-6)
        support_mean = (filled_profiles * weighted_valid).sum(dim=1) / denom
        coverage = torch.clamp(valid.mean(dim=1), min=0.0, max=1.0)
        support_count = valid.amax(dim=2).sum(dim=1, keepdim=True)
        age_valid = valid.amax(dim=2)
        age_denom = torch.clamp(age_valid.sum(dim=1, keepdim=True), min=1.0e-6)
        age_mean = (support_ages_days * age_valid).sum(dim=1, keepdim=True) / age_denom
        age_min = torch.where(
            age_valid > 0.5,
            support_ages_days,
            torch.full_like(support_ages_days, 3650.0),
        ).min(dim=1, keepdim=True).values
        age_max = torch.where(
            age_valid > 0.5,
            support_ages_days,
            torch.zeros_like(support_ages_days),
        ).max(dim=1, keepdim=True).values
        support_stats = torch.cat(
            (
                support_count / 10.0,
                age_mean / 365.0,
                age_min / 365.0,
                age_max / 365.0,
            ),
            dim=1,
        )
        encoded = self.net(torch.cat(
            (support_mean, coverage, support_stats, static_features, forcing_context),
            dim=1,
        ))
        profile_raw = encoded[:, :self.depth_count]
        adapter_raw = encoded[:, self.depth_count:]
        profile_delta = torch.tanh(profile_raw) * float(self.initial_delta_limit_c)
        coverage_gate = torch.clamp(
            coverage + float(self.unobserved_delta_scale) * (1.0 - coverage),
            min=0.0,
            max=1.0,
        )
        profile_delta = profile_delta * coverage_gate
        observed_depth = coverage > 0.0
        unobserved_depth = ~observed_depth
        observed_denom = torch.clamp(observed_depth.to(dtype=profile_delta.dtype).sum(dim=1), min=1.0)
        unobserved_denom = torch.clamp(unobserved_depth.to(dtype=profile_delta.dtype).sum(dim=1), min=1.0)
        observed_abs_delta = (
            profile_delta.abs() * observed_depth.to(dtype=profile_delta.dtype)
        ).sum(dim=1) / observed_denom
        unobserved_abs_delta = (
            profile_delta.abs() * unobserved_depth.to(dtype=profile_delta.dtype)
        ).sum(dim=1) / unobserved_denom
        adapter = {
            key: adapter_raw[:, idx]
            for idx, key in enumerate(LAKE_ADAPTIVE_OUTPUT_KEYS)
        }
        regularization = profile_delta.pow(2).mean(dim=1) + adapter_raw.pow(2).mean(dim=1)
        return {
            'initial_profile_delta_c': profile_delta,
            'adapter_raw': adapter,
            'regularization_loss': regularization,
            'support_profile_count': support_count.reshape(-1),
            'support_age_mean_days': age_mean.reshape(-1),
            'support_depth_coverage_mean': coverage.mean(dim=1),
            'support_unobserved_depth_fraction': unobserved_depth.to(dtype=profile_delta.dtype).mean(dim=1),
            'initial_delta_observed_abs_mean_c': observed_abs_delta,
            'initial_delta_unobserved_abs_mean_c': unobserved_abs_delta,
        }


class SparsePersistentStateObserver(nn.Module):
    """Encode sparse support residuals into a persistent state and adapter update."""

    def __init__(
        self,
        depth_count,
        static_dim,
        forcing_context_dim,
        hidden_dim=64,
        init_spread=0.005,
        state_delta_limit_c=2.0,
        unobserved_delta_scale=1.0,
        residual_anchor_fraction=0.0,
    ):
        super().__init__()
        self.depth_count = int(depth_count)
        self.static_dim = int(static_dim)
        self.forcing_context_dim = int(forcing_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_spread = float(init_spread)
        self.state_delta_limit_c = float(state_delta_limit_c)
        self.unobserved_delta_scale = float(unobserved_delta_scale)
        self.residual_anchor_fraction = float(residual_anchor_fraction)
        if self.depth_count <= 0:
            raise ValueError('observer depth_count must be positive.')
        if self.hidden_dim <= 0:
            raise ValueError('observer hidden_dim must be positive.')
        if self.init_spread < 0.0:
            raise ValueError('observer init_spread must be non-negative.')
        if self.state_delta_limit_c < 0.0:
            raise ValueError('observer state_delta_limit_c must be non-negative.')
        if not (0.0 <= self.unobserved_delta_scale <= 1.0):
            raise ValueError('observer unobserved_delta_scale must be between 0.0 and 1.0.')
        if not (0.0 <= self.residual_anchor_fraction <= 1.0):
            raise ValueError('observer residual_anchor_fraction must be between 0.0 and 1.0.')
        input_dim = self.depth_count * 4 + 4 + self.static_dim + self.forcing_context_dim
        output_dim = self.depth_count + len(LAKE_ADAPTIVE_OUTPUT_KEYS)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )
        self._initialize_near_zero()

    def _initialize_near_zero(self):
        final = self.net[-1]
        if self.init_spread > 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=self.init_spread)
        else:
            nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        current_profile,
        support_profiles,
        support_masks,
        support_ages_days,
        static_features,
        forcing_context,
    ):
        if current_profile.ndim == 1:
            current_profile = current_profile.unsqueeze(0)
        if support_profiles.ndim == 2:
            support_profiles = support_profiles.unsqueeze(0)
        if support_masks.ndim == 2:
            support_masks = support_masks.unsqueeze(0)
        if support_ages_days.ndim == 1:
            support_ages_days = support_ages_days.unsqueeze(0)
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        if forcing_context.ndim == 1:
            forcing_context = forcing_context.unsqueeze(0)

        batch_size = support_profiles.shape[0]
        device = static_features.device
        dtype = static_features.dtype
        current_profile = current_profile.to(device=device, dtype=dtype)
        support_profiles = support_profiles.to(device=device, dtype=dtype)
        support_masks = support_masks.to(device=device, dtype=dtype)
        support_ages_days = support_ages_days.to(device=device, dtype=dtype)
        if current_profile.shape[0] == 1 and batch_size > 1:
            current_profile = current_profile.expand(batch_size, -1)
        if static_features.shape[0] == 1 and batch_size > 1:
            static_features = static_features.expand(batch_size, -1)
        if forcing_context.shape[0] == 1 and batch_size > 1:
            forcing_context = forcing_context.expand(batch_size, -1)

        finite_mask = torch.isfinite(support_profiles) & (support_masks > 0.5)
        valid = finite_mask.to(dtype=dtype)
        filled_profiles = torch.where(finite_mask, support_profiles, torch.zeros_like(support_profiles))
        current_expanded = current_profile.unsqueeze(1).expand_as(filled_profiles)
        filled_residual = torch.where(finite_mask, filled_profiles - current_expanded, torch.zeros_like(filled_profiles))
        recency_weight = torch.exp(-torch.clamp(support_ages_days, min=0.0) / 120.0).reshape(batch_size, -1, 1)
        weighted_valid = valid * recency_weight
        denom = torch.clamp(weighted_valid.sum(dim=1), min=1.0e-6)
        support_mean = (filled_profiles * weighted_valid).sum(dim=1) / denom
        residual_mean = (filled_residual * weighted_valid).sum(dim=1) / denom
        coverage = torch.clamp(valid.mean(dim=1), min=0.0, max=1.0)
        support_count = valid.amax(dim=2).sum(dim=1, keepdim=True)
        age_valid = valid.amax(dim=2)
        age_denom = torch.clamp(age_valid.sum(dim=1, keepdim=True), min=1.0e-6)
        age_mean = (support_ages_days * age_valid).sum(dim=1, keepdim=True) / age_denom
        age_min = torch.where(
            age_valid > 0.5,
            support_ages_days,
            torch.full_like(support_ages_days, 3650.0),
        ).min(dim=1, keepdim=True).values
        age_max = torch.where(
            age_valid > 0.5,
            support_ages_days,
            torch.zeros_like(support_ages_days),
        ).max(dim=1, keepdim=True).values
        support_stats = torch.cat(
            (
                support_count / 10.0,
                age_mean / 365.0,
                age_min / 365.0,
                age_max / 365.0,
            ),
            dim=1,
        )
        encoded = self.net(torch.cat(
            (
                current_profile,
                support_mean,
                residual_mean,
                coverage,
                support_stats,
                static_features,
                forcing_context,
            ),
            dim=1,
        ))
        state_raw = encoded[:, :self.depth_count]
        adapter_raw = encoded[:, self.depth_count:]
        state_delta_limit = float(self.state_delta_limit_c)
        state_delta = torch.tanh(state_raw) * state_delta_limit
        observed_depth = coverage > 0.0
        unobserved_depth = ~observed_depth
        observed_denom = torch.clamp(observed_depth.to(dtype=dtype).sum(dim=1, keepdim=True), min=1.0)
        if self.residual_anchor_fraction > 0.0 and state_delta_limit > 0.0:
            observed_residual_mean = (
                residual_mean * observed_depth.to(dtype=dtype)
            ).sum(dim=1, keepdim=True) / observed_denom
            residual_anchor = torch.where(
                observed_depth,
                residual_mean,
                observed_residual_mean.expand_as(residual_mean),
            )
            state_delta = torch.tanh(
                (state_delta + residual_anchor * float(self.residual_anchor_fraction))
                / state_delta_limit
            ) * state_delta_limit
        coverage_gate = torch.clamp(
            coverage + float(self.unobserved_delta_scale) * (1.0 - coverage),
            min=0.0,
            max=1.0,
        )
        state_delta = state_delta * coverage_gate
        observed_denom = torch.clamp(observed_depth.to(dtype=dtype).sum(dim=1), min=1.0)
        unobserved_denom = torch.clamp(unobserved_depth.to(dtype=dtype).sum(dim=1), min=1.0)
        observed_abs_delta = (
            state_delta.abs() * observed_depth.to(dtype=dtype)
        ).sum(dim=1) / observed_denom
        unobserved_abs_delta = (
            state_delta.abs() * unobserved_depth.to(dtype=dtype)
        ).sum(dim=1) / unobserved_denom
        adapter = {
            key: adapter_raw[:, idx]
            for idx, key in enumerate(LAKE_ADAPTIVE_OUTPUT_KEYS)
        }
        regularization = state_delta.pow(2).mean(dim=1) + adapter_raw.pow(2).mean(dim=1)
        residual_observed_abs = (
            residual_mean.abs() * observed_depth.to(dtype=dtype)
        ).sum(dim=1) / observed_denom
        return {
            'observer_state_delta_c': state_delta,
            'adapter_raw': adapter,
            'regularization_loss': regularization,
            'support_profile_count': support_count.reshape(-1),
            'support_age_mean_days': age_mean.reshape(-1),
            'support_depth_coverage_mean': coverage.mean(dim=1),
            'support_unobserved_depth_fraction': unobserved_depth.to(dtype=dtype).mean(dim=1),
            'observer_delta_observed_abs_mean_c': observed_abs_delta,
            'observer_delta_unobserved_abs_mean_c': unobserved_abs_delta,
            'observer_support_residual_observed_abs_mean_c': residual_observed_abs,
        }


class LakeStateForecaster(nn.Module):
    """State-space lake profile forecaster with a differentiable 1D solver."""

    def __init__(
        self,
        depths,
        area_profile,
        forcing_dim=len(FORCING_FEATURE_COLUMNS),
        static_dim=STATIC_FEATURE_DIM,
        hidden_dim=96,
        forcing_context_dim=48,
        forcing_history_hidden_dim=48,
        residual_limit_c=0.50,
        lst_feature_dropout_probability=0.20,
        shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
        wind_kz_scale=1.0,
        autumn_convective_boost=1.0,
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
        fewshot_hidden_dim=64,
        fewshot_init_spread=0.005,
        fewshot_initial_delta_limit_c=2.0,
        fewshot_unobserved_delta_scale=1.0,
        observer_hidden_dim=64,
        observer_init_spread=0.005,
        observer_state_delta_limit_c=2.0,
        observer_unobserved_delta_scale=1.0,
        observer_residual_anchor_fraction=0.0,
        fewshot_adapter_scale=0.25,
        fewshot_adapter_params='kz,kd,exchange,convective,ice',
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
    ):
        super().__init__()
        self.register_buffer('depths', torch.tensor(np.asarray(depths, dtype=np.float32)))
        self.register_buffer('area_profile', torch.tensor(np.asarray(area_profile, dtype=np.float32)))
        self.forcing_encoder = ForcingHistoryEncoder(
            forcing_dim=forcing_dim,
            hidden_dim=forcing_history_hidden_dim,
            output_dim=forcing_context_dim,
        )
        self.param_net = StateParameterNet(
            forcing_context_dim,
            static_dim,
            hidden_dim,
            residual_limit_c=residual_limit_c,
        )
        self.physical_scale_head = PhysicalScaleHead(
            forcing_context_dim,
            static_dim,
            forcing_dim,
            hidden_dim=max(32, hidden_dim // 2),
        )
        self.lst_feature_dropout_probability = float(lst_feature_dropout_probability)
        if not (0.0 <= self.lst_feature_dropout_probability <= 1.0):
            raise ValueError('lst_feature_dropout_probability must be between 0.0 and 1.0.')
        self.shortwave_surface_fraction = float(shortwave_surface_fraction)
        self.wind_kz_scale = float(wind_kz_scale)
        self.autumn_convective_boost = float(autumn_convective_boost)
        self.turbulent_flux_mode = normalize_turbulent_flux_mode(turbulent_flux_mode)
        self.turbulent_flux_blend_alpha = float(np.clip(turbulent_flux_blend_alpha, 0.0, 1.0))
        self.freezing_energy_mode = normalize_freezing_energy_mode(freezing_energy_mode)
        self.advective_heat_source_mode = normalize_advective_heat_source_mode(advective_heat_source_mode)
        self.shape_aware_mixing = normalize_shape_aware_mixing(shape_aware_mixing)
        self.shape_mixing_strength = float(shape_mixing_strength)
        if self.shape_mixing_strength < 0.0:
            raise ValueError('shape_mixing_strength must be non-negative.')
        self.stratification_mixing_cap = normalize_shape_aware_mixing(stratification_mixing_cap)
        self.stratification_mixing_cap_strength = float(stratification_mixing_cap_strength)
        if self.stratification_mixing_cap_strength < 0.0:
            raise ValueError('stratification_mixing_cap_strength must be non-negative.')
        self.lake_adaptive_params = normalize_lake_adaptive_params(lake_adaptive_params)
        adaptive_modes = lake_adaptive_param_set(self.lake_adaptive_params)
        if 'flux' in adaptive_modes and self.turbulent_flux_mode != 'blend':
            raise ValueError("lake_adaptive_params including flux requires turbulent_flux_mode='blend'.")
        if 'exchange' in adaptive_modes and self.turbulent_flux_mode == 'provided':
            raise ValueError("lake_adaptive_params including exchange requires turbulent_flux_mode='bulk' or 'blend'.")
        self.lake_adaptive_hidden_dim = int(lake_adaptive_hidden_dim)
        self.lake_adaptive_init_spread = float(lake_adaptive_init_spread)
        self.lake_adaptive_temporal_mode = normalize_lake_adaptive_temporal_mode(lake_adaptive_temporal_mode)
        self.lake_adaptive_temporal_init_spread = float(lake_adaptive_temporal_init_spread)
        self.lake_adaptive_temporal_scale = float(lake_adaptive_temporal_scale)
        if self.lake_adaptive_temporal_init_spread < 0.0:
            raise ValueError('lake_adaptive_temporal_init_spread must be non-negative.')
        if self.lake_adaptive_temporal_scale < 0.0:
            raise ValueError('lake_adaptive_temporal_scale must be non-negative.')
        if self.lake_adaptive_temporal_mode != 'off' and not adaptive_modes:
            raise ValueError("lake_adaptive_temporal_mode requires lake_adaptive_params other than 'off'.")
        self.adaptive_wind_kz_min = float(adaptive_wind_kz_min)
        self.adaptive_wind_kz_max = float(adaptive_wind_kz_max)
        self.adaptive_blend_alpha_min = float(adaptive_blend_alpha_min)
        self.adaptive_blend_alpha_max = float(adaptive_blend_alpha_max)
        self.adaptive_kd_multiplier_min = float(adaptive_kd_multiplier_min)
        self.adaptive_kd_multiplier_max = float(adaptive_kd_multiplier_max)
        self.adaptive_turbulent_exchange_scale_min = float(adaptive_turbulent_exchange_scale_min)
        self.adaptive_turbulent_exchange_scale_max = float(adaptive_turbulent_exchange_scale_max)
        self.adaptive_convective_mixing_scale_min = float(adaptive_convective_mixing_scale_min)
        self.adaptive_convective_mixing_scale_max = float(adaptive_convective_mixing_scale_max)
        self.adaptive_ice_shortwave_scale_min = float(adaptive_ice_shortwave_scale_min)
        self.adaptive_ice_shortwave_scale_max = float(adaptive_ice_shortwave_scale_max)
        self.lake_adaptive_head = LakeAdaptiveParameterHead(
            static_dim,
            hidden_dim=self.lake_adaptive_hidden_dim,
            init_spread=self.lake_adaptive_init_spread,
            wind_kz_bounds=(self.adaptive_wind_kz_min, self.adaptive_wind_kz_max),
            blend_alpha_bounds=(self.adaptive_blend_alpha_min, self.adaptive_blend_alpha_max),
            kd_multiplier_bounds=(self.adaptive_kd_multiplier_min, self.adaptive_kd_multiplier_max),
            exchange_scale_bounds=(
                self.adaptive_turbulent_exchange_scale_min,
                self.adaptive_turbulent_exchange_scale_max,
            ),
            convective_scale_bounds=(
                self.adaptive_convective_mixing_scale_min,
                self.adaptive_convective_mixing_scale_max,
            ),
            ice_shortwave_scale_bounds=(self.adaptive_ice_shortwave_scale_min, self.adaptive_ice_shortwave_scale_max),
            base_wind_kz_scale=self.wind_kz_scale,
            base_blend_alpha=self.turbulent_flux_blend_alpha,
        )
        self.lake_adaptive_temporal_head = None
        if self.lake_adaptive_temporal_mode != 'off':
            self.lake_adaptive_temporal_head = LakeAdaptiveTemporalDeltaHead(
                static_dim,
                forcing_context_dim,
                forcing_dim,
                hidden_dim=self.lake_adaptive_hidden_dim,
                init_spread=self.lake_adaptive_temporal_init_spread,
            )
        self.fewshot_adapter_params = normalize_lake_adaptive_params('off')
        self.fewshot_adapter_scale = 0.0
        self.fewshot_initial_delta_limit_c = 0.0
        self.fewshot_unobserved_delta_scale = 0.0
        self.observer_state_delta_limit_c = float(observer_state_delta_limit_c)
        self.observer_unobserved_delta_scale = float(observer_unobserved_delta_scale)
        self.observer_residual_anchor_fraction = float(observer_residual_anchor_fraction)
        if self.fewshot_adapter_scale < 0.0:
            raise ValueError('fewshot_adapter_scale must be non-negative.')
        if not (0.0 <= self.fewshot_unobserved_delta_scale <= 1.0):
            raise ValueError('fewshot_unobserved_delta_scale must be between 0.0 and 1.0.')
        if self.observer_state_delta_limit_c < 0.0:
            raise ValueError('observer_state_delta_limit_c must be non-negative.')
        if not (0.0 <= self.observer_unobserved_delta_scale <= 1.0):
            raise ValueError('observer_unobserved_delta_scale must be between 0.0 and 1.0.')
        if not (0.0 <= self.observer_residual_anchor_fraction <= 1.0):
            raise ValueError('observer_residual_anchor_fraction must be between 0.0 and 1.0.')
        self.fewshot_encoder = None
        self.sparse_observer = SparsePersistentStateObserver(
            len(self.depths),
            static_dim,
            forcing_context_dim,
            hidden_dim=int(observer_hidden_dim),
            init_spread=float(observer_init_spread),
            state_delta_limit_c=self.observer_state_delta_limit_c,
            unobserved_delta_scale=self.observer_unobserved_delta_scale,
            residual_anchor_fraction=self.observer_residual_anchor_fraction,
        )

    def _adaptive_parameter_values(
        self,
        static_features,
        forcing_context=None,
        forcing_features=None,
        fewshot_adapter=None,
    ):
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0)
        batch_size = static_features.shape[0]
        device = static_features.device
        dtype = static_features.dtype
        base_wind = torch.full(
            (batch_size,),
            float(self.wind_kz_scale),
            device=device,
            dtype=dtype,
        )
        base_alpha = torch.full(
            (batch_size,),
            float(self.turbulent_flux_blend_alpha),
            device=device,
            dtype=dtype,
        )
        base_values = {
            'wind_kz_scale': base_wind,
            'blend_alpha': base_alpha,
            'kd_multiplier': torch.ones((batch_size,), device=device, dtype=dtype),
            'turbulent_exchange_scale': torch.ones((batch_size,), device=device, dtype=dtype),
            'convective_mixing_scale': torch.ones((batch_size,), device=device, dtype=dtype),
            'ice_shortwave_scale': torch.ones((batch_size,), device=device, dtype=dtype),
        }
        active_modes = lake_adaptive_param_set(self.lake_adaptive_params)
        values = dict(base_values)
        losses = []
        if active_modes:
            learned_values = self.lake_adaptive_head(static_features)
            for mode in active_modes:
                key = LAKE_ADAPTIVE_MODE_TO_OUTPUT[mode]
                values[key] = learned_values[key]
                lower, upper = self.lake_adaptive_head.bounds[key]
                value_range = max(float(upper) - float(lower), 1.0e-8)
                losses.append(((values[key] - base_values[key]) / value_range).pow(2))
        if active_modes and self.lake_adaptive_temporal_head is not None and self.lake_adaptive_temporal_scale > 0.0:
            if forcing_context is None or forcing_features is None:
                raise ValueError('forcing_context and forcing_features are required for adaptive temporal mode.')
            temporal_raw = self.lake_adaptive_temporal_head(static_features, forcing_context, forcing_features)
            temporal_losses = []
            for mode in active_modes:
                key = LAKE_ADAPTIVE_MODE_TO_OUTPUT[mode]
                lower, upper = self.lake_adaptive_head.bounds[key]
                value_range = max(float(upper) - float(lower), 1.0e-8)
                delta = torch.tanh(temporal_raw[key]) * float(self.lake_adaptive_temporal_scale)
                if key == 'blend_alpha':
                    adjusted = values[key] + delta * value_range
                else:
                    adjusted = values[key] * torch.exp(delta)
                lower_tensor = torch.as_tensor(float(lower), device=device, dtype=dtype)
                upper_tensor = torch.as_tensor(float(upper), device=device, dtype=dtype)
                values[key] = torch.clamp(adjusted, min=lower_tensor, max=upper_tensor)
                temporal_losses.append(delta.pow(2))
            if temporal_losses:
                losses.append(torch.stack(temporal_losses, dim=0).mean(dim=0))
        fewshot_modes = lake_adaptive_param_set(self.fewshot_adapter_params)
        if fewshot_adapter is not None and fewshot_modes and self.fewshot_adapter_scale > 0.0:
            fewshot_losses = []
            for mode in fewshot_modes:
                key = LAKE_ADAPTIVE_MODE_TO_OUTPUT[mode]
                raw_delta = fewshot_adapter.get(key)
                if raw_delta is None:
                    continue
                raw_delta = raw_delta.to(device=device, dtype=dtype).reshape(-1)
                if raw_delta.numel() == 1 and batch_size > 1:
                    raw_delta = raw_delta.expand(batch_size)
                lower, upper = self.lake_adaptive_head.bounds[key]
                value_range = max(float(upper) - float(lower), 1.0e-8)
                delta = torch.tanh(raw_delta) * float(self.fewshot_adapter_scale)
                if key == 'blend_alpha':
                    adjusted = values[key] + delta * value_range
                else:
                    adjusted = values[key] * torch.exp(delta)
                values[key] = torch.clamp(
                    adjusted,
                    min=torch.as_tensor(float(lower), device=device, dtype=dtype),
                    max=torch.as_tensor(float(upper), device=device, dtype=dtype),
                )
                fewshot_losses.append(delta.pow(2))
            if fewshot_losses:
                losses.append(torch.stack(fewshot_losses, dim=0).mean(dim=0))
        regularization = torch.stack(losses, dim=0).mean(dim=0) if losses else torch.zeros_like(base_wind)
        return values, regularization

    def encode_fewshot_support(
        self,
        support_profiles,
        support_masks,
        support_ages_days,
        static_features,
        query_forcing_row,
    ):
        raise RuntimeError(
            'few-shot support encoding has been removed from the zero-profile RECON mainline.'
        )

    def encode_sparse_observer_update(
        self,
        current_profile,
        support_profiles,
        support_masks,
        support_ages_days,
        static_features,
        query_forcing_row,
    ):
        target_depth_count = int(current_profile.shape[-1])
        encoder_depth_count = int(self.sparse_observer.depth_count)
        query_features = query_forcing_row['features']
        query_history = query_forcing_row.get('history_features')
        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        query_features = query_features.to(device=static_features.device, dtype=static_features.dtype)
        if query_history is not None:
            query_history = query_history.to(device=static_features.device, dtype=static_features.dtype)
            if query_history.ndim == 2:
                query_history = query_history.unsqueeze(0)
        forcing_context = self._encode_forcing_context(query_features, query_history)
        if target_depth_count != encoder_depth_count:
            current_profile = _resize_depth_vector(current_profile, encoder_depth_count)
            support_profiles, support_masks = _resize_support_profiles_and_masks(
                support_profiles,
                support_masks,
                encoder_depth_count,
            )
        encoded = self.sparse_observer(
            current_profile,
            support_profiles,
            support_masks,
            support_ages_days,
            static_features,
            forcing_context,
        )
        if target_depth_count != encoder_depth_count:
            encoded = dict(encoded)
            encoded['observer_state_delta_c'] = _resize_depth_vector(
                encoded['observer_state_delta_c'],
                target_depth_count,
            )
        return encoded

    def _encode_forcing_context(self, forcing_features, forcing_history=None):
        if forcing_history is None:
            forcing_history = forcing_features.unsqueeze(1)
        forcing_history = forcing_history.to(device=forcing_features.device, dtype=forcing_features.dtype)
        if forcing_history.ndim == 2:
            forcing_history = forcing_history.unsqueeze(0)
        return self.forcing_encoder(forcing_history)

    def _forcing_feature_column(self, forcing_features, name, default=0.0):
        features = forcing_features
        if features.ndim == 1:
            features = features.unsqueeze(0)
        idx = FORCING_FEATURE_INDEX.get(name)
        if idx is None or features.shape[1] <= idx:
            return torch.full(
                (features.shape[0],),
                float(default),
                device=features.device,
                dtype=features.dtype,
            )
        value = features[:, idx]
        return torch.where(
            torch.isfinite(value),
            value,
            torch.full_like(value, float(default)),
        )

    def _static_feature_column(self, static_features, name, default=0.0):
        features = static_features
        if features.ndim == 1:
            features = features.unsqueeze(0)
        try:
            idx = STATIC_FEATURE_KEYS.index(name)
        except ValueError:
            idx = None
        if idx is None or features.shape[1] <= idx:
            return torch.full(
                (features.shape[0],),
                float(default),
                device=features.device,
                dtype=features.dtype,
            )
        value = features[:, idx]
        return torch.where(
            torch.isfinite(value),
            value,
            torch.full_like(value, float(default)),
        )

    def _apply_lst_feature_dropout(self, forcing_features, forcing_history=None):
        batch_size = int(forcing_features.shape[0])
        device = forcing_features.device
        dtype = forcing_features.dtype
        probability = float(self.lst_feature_dropout_probability)
        if (not self.training) or probability <= 0.0:
            mask = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            return forcing_features, forcing_history, mask.to(dtype=dtype)
        if probability >= 1.0:
            mask = torch.ones((batch_size,), device=device, dtype=torch.bool)
        else:
            mask = torch.rand((batch_size,), device=device) < probability

        dropped_features = forcing_features.clone()
        drop_columns = (
            'LST_surface_C',
            'lst_mean_7d',
            'LST_quality_factor',
            'LST_observed_flag',
        )
        fill_columns = ('LST_is_filled',)
        sample_mask = mask.reshape(-1)
        for name in drop_columns:
            idx = FORCING_FEATURE_INDEX.get(name)
            if idx is not None and idx < dropped_features.shape[1]:
                dropped_features[sample_mask, idx] = 0.0
        for name in fill_columns:
            idx = FORCING_FEATURE_INDEX.get(name)
            if idx is not None and idx < dropped_features.shape[1]:
                dropped_features[sample_mask, idx] = 1.0

        dropped_history = forcing_history
        if forcing_history is not None:
            dropped_history = forcing_history.clone()
            for name in drop_columns:
                idx = FORCING_FEATURE_INDEX.get(name)
                if idx is not None and idx < dropped_history.shape[-1]:
                    dropped_history[sample_mask, :, idx] = 0.0
            for name in fill_columns:
                idx = FORCING_FEATURE_INDEX.get(name)
                if idx is not None and idx < dropped_history.shape[-1]:
                    dropped_history[sample_mask, :, idx] = 1.0
        return dropped_features, dropped_history, mask.to(dtype=dtype)

    def _kd_base_from_forcing_features(self, forcing_features):
        light_extinction_norm = self._forcing_feature_column(forcing_features, 'light_extinction_kd', 0.0)
        secchi_norm = self._forcing_feature_column(forcing_features, 'Secchi_m', 0.0)
        light_extinction_kd = light_extinction_norm * float(PINN_LIGHT_EXTINCTION_REFERENCE_M_INV)
        secchi_m = secchi_norm * float(PINN_MAX_SECCHI_REFERENCE_M)
        kd_from_secchi = 1.7 / torch.clamp(secchi_m, min=0.10)
        default = torch.full_like(light_extinction_kd, float(DEFAULT_KD_BASE_M_INV))
        kd_base = torch.where(
            light_extinction_kd > 0.0,
            light_extinction_kd,
            torch.where(secchi_m > 0.10, kd_from_secchi, default),
        )
        return torch.clamp(kd_base, min=MIN_SHORTWAVE_ATTENUATION, max=MAX_SHORTWAVE_ATTENUATION)

    def _advective_heat_source(
        self,
        temperature,
        forcing_features,
        active_forcing,
        static_features,
        depths,
        area_profile,
        dt_seconds,
    ):
        zeros = torch.zeros_like(temperature)
        batch_zeros = torch.zeros(
            temperature.shape[0],
            device=temperature.device,
            dtype=temperature.dtype,
        )
        if self.advective_heat_source_mode == 'off':
            return zeros, {
                'advective_heat_source_c_per_day_mean': batch_zeros,
                'advective_heat_source_c_per_day_max': batch_zeros,
                'advective_exchange_fraction_per_day': batch_zeros,
                'advective_heat_source_active_mean': batch_zeros,
            }

        reservoir_indicator = torch.clamp(
            self._static_feature_column(static_features, 'reservoir_indicator', 0.0),
            0.0,
            1.0,
        )
        net_inflow_m3_s = torch.relu(
            self._forcing_feature_column(forcing_features, 'net_inflow', 0.0)
            * float(PINN_INFLOW_REFERENCE_M3_S)
        )

        volume_from_metadata = (
            torch.clamp(self._static_feature_column(static_features, 'volume_norm', 0.0), min=0.0)
            * float(PINN_VOLUME_REFERENCE_KM3)
            * 1.0e9
        )
        log_area_norm = torch.clamp(
            self._static_feature_column(static_features, 'log_area', 0.0),
            min=0.0,
        )
        area_km2 = torch.expm1(
            torch.clamp(log_area_norm * float(PINN_LOG_AREA_REFERENCE_KM2), min=0.0, max=20.0)
        )
        mean_depth_m = torch.clamp(
            self._static_feature_column(static_features, 'mean_depth_norm', 0.0)
            * float(PINN_MAX_MEAN_DEPTH_REFERENCE_M),
            min=0.0,
        )
        max_depth_m = torch.clamp(
            self._static_feature_column(static_features, 'max_depth_norm', 0.0)
            * float(PINN_MAX_DEPTH_REFERENCE_M),
            min=0.0,
        )
        fallback_mean_depth_m = torch.where(
            mean_depth_m > 0.0,
            mean_depth_m,
            0.45 * max_depth_m,
        )
        estimated_volume_m3 = area_km2 * 1.0e6 * fallback_mean_depth_m
        volume_m3 = torch.where(
            volume_from_metadata > 1.0e3,
            volume_from_metadata,
            estimated_volume_m3,
        )
        valid_volume = volume_m3 > 1.0e3
        active = (
            (reservoir_indicator > 0.0)
            & (net_inflow_m3_s > 0.0)
            & valid_volume
        ).to(dtype=temperature.dtype)
        exchange_fraction = torch.clamp(
            net_inflow_m3_s * float(dt_seconds) / torch.clamp(volume_m3, min=1.0),
            min=0.0,
            max=0.10,
        ) * active

        air_temp_norm_7d = self._forcing_feature_column(forcing_features, 'air_temp_mean_7d', float('nan'))
        forcing_air_temp = active_forcing['air_temp'].to(device=temperature.device, dtype=temperature.dtype).reshape(-1)
        if forcing_air_temp.numel() == 1 and temperature.shape[0] > 1:
            forcing_air_temp = forcing_air_temp.expand(temperature.shape[0])
        inflow_temp_c = torch.where(
            torch.isfinite(air_temp_norm_7d),
            air_temp_norm_7d * float(PINN_MAX_TEMPERATURE_REFERENCE_C),
            forcing_air_temp,
        )

        depths = depths.flatten().to(device=temperature.device, dtype=temperature.dtype)
        area = area_profile.flatten().to(device=temperature.device, dtype=temperature.dtype)
        dz = layer_thicknesses(depths)
        max_depth = torch.clamp(depths[-1], min=1.0)
        center = torch.clamp(0.25 * max_depth, min=0.5, max=max_depth)
        width = torch.clamp(0.15 * max_depth, min=1.0)
        shape = torch.exp(-0.5 * ((depths - center) / width).pow(2))
        volume_weights = area * dz
        shape_mean = torch.sum(shape * volume_weights) / torch.clamp(torch.sum(volume_weights), min=1.0e-6)
        shape = shape / torch.clamp(shape_mean, min=1.0e-6)

        source_c_per_day = (
            exchange_fraction.reshape(-1, 1)
            * (inflow_temp_c.reshape(-1, 1) - temperature)
            * shape.reshape(1, -1)
        )
        return source_c_per_day / float(dt_seconds), {
            'advective_heat_source_c_per_day_mean': source_c_per_day.mean(dim=1),
            'advective_heat_source_c_per_day_max': source_c_per_day.max(dim=1).values,
            'advective_exchange_fraction_per_day': exchange_fraction.reshape(-1),
            'advective_heat_source_active_mean': active.reshape(-1),
        }

    def _density_stability_terms(self, temperature, depths, wind_speed):
        depths = depths.flatten().to(device=temperature.device, dtype=temperature.dtype)
        rho = water_density_torch(temperature)
        if depths.numel() <= 1:
            zeros = torch.zeros_like(temperature)
            return zeros, zeros
        dz_interface = torch.clamp(depths[1:] - depths[:-1], min=1.0e-4)
        drho_dz_interface = (rho[:, 1:] - rho[:, :-1]) / dz_interface.reshape(1, -1)
        drho_dz = torch.empty_like(temperature)
        drho_dz[:, 0] = drho_dz_interface[:, 0]
        drho_dz[:, -1] = drho_dz_interface[:, -1]
        if depths.numel() > 2:
            drho_dz[:, 1:-1] = 0.5 * (drho_dz_interface[:, :-1] + drho_dz_interface[:, 1:])
        n2 = (GRAVITY / WATER_DENSITY) * drho_dz
        depth_scale = torch.clamp(depths[-1], min=1.0)
        shear = RI_WIND_SHEAR_FACTOR * torch.clamp(wind_speed.reshape(-1, 1), min=0.1).pow(2) / depth_scale.pow(2)
        richardson = n2 / torch.clamp(shear, min=1.0e-8)
        return n2, richardson

    def _shape_mixing_terms(self, static_features, depths):
        batch_size = static_features.shape[0]
        device = static_features.device
        dtype = static_features.dtype
        depths = depths.flatten().to(device=device, dtype=dtype)
        ones = torch.ones((batch_size,), device=device, dtype=dtype)
        if self.shape_aware_mixing == 'off' or self.shape_mixing_strength <= 0.0:
            return ones, torch.full_like(ones, float(DIFFUSIVITY_WIND_DECAY_DEPTH))

        def feature(name, default, *, positive_default=False):
            idx = STATIC_FEATURE_KEYS.index(name)
            if static_features.shape[1] <= idx:
                return torch.full_like(ones, float(default))
            value = static_features[:, idx].to(device=device, dtype=dtype)
            valid = torch.isfinite(value)
            if positive_default:
                valid = valid & (value > 0.0)
            return torch.where(valid, value, torch.full_like(value, float(default)))

        max_depth_norm = feature('max_depth_norm', 0.40, positive_default=True)
        mean_depth_norm = feature('mean_depth_norm', 0.14, positive_default=True)
        log_area = feature('log_area', 0.30, positive_default=True)
        fetch_norm = feature('fetch_norm', 0.03, positive_default=True)
        wind_exposure = feature('wind_exposure_norm', 1.0, positive_default=True)
        shoreline_development = feature('shoreline_development_norm', 1.0, positive_default=True)
        reservoir_indicator = torch.clamp(feature('reservoir_indicator', 0.0), 0.0, 1.0)
        depth_ratio = torch.clamp(mean_depth_norm / torch.clamp(max_depth_norm, min=1.0e-4), 0.05, 1.20)
        basin_shape = feature('basin_shape_norm', 0.35, positive_default=True)
        basin_shape = torch.where(basin_shape > 0.0, basin_shape, depth_ratio)

        area_signal = torch.clamp((log_area - 0.30) / 0.40, -1.5, 1.5)
        fetch_signal = torch.clamp(
            torch.log1p(fetch_norm / 0.03) / np.log(2.0) - 1.0,
            -1.5,
            1.5,
        )
        exposure_signal = torch.clamp(wind_exposure - 1.0, -1.0, 1.0)
        basin_signal = torch.clamp((basin_shape - 0.35) / 0.35, -1.0, 1.0)
        depth_signal = torch.clamp((max_depth_norm - 0.40) / 0.60, -1.0, 1.0)
        shoreline_signal = torch.clamp(shoreline_development - 1.0, -1.0, 1.0)

        shape_signal = (
            0.35 * area_signal
            + 0.40 * fetch_signal
            + 0.20 * exposure_signal
            + 0.20 * basin_signal
            - 0.10 * shoreline_signal
            - 0.10 * reservoir_indicator
        )
        penetration_signal = (
            0.25 * area_signal
            + 0.45 * fetch_signal
            + 0.25 * basin_signal
            - 0.15 * depth_signal
            - 0.10 * shoreline_signal
        )
        strength = torch.as_tensor(self.shape_mixing_strength, device=device, dtype=dtype)
        wind_factor = torch.clamp(torch.exp(strength * shape_signal), 0.50, 2.00)
        decay_factor = torch.clamp(torch.exp(strength * penetration_signal), 0.50, 2.00)
        decay_depth_m = torch.clamp(
            float(DIFFUSIVITY_WIND_DECAY_DEPTH) * decay_factor,
            min=1.0,
            max=25.0,
        )
        return wind_factor, decay_depth_m

    def predict_params(
        self,
        forcing_features,
        static_features,
        wind_speed,
        temperature=None,
        forcing_history=None,
        depths=None,
        ice_fraction=None,
        forcing_context=None,
        wind_kz_scale=None,
        kd_multiplier=None,
        convective_mixing_scale=None,
        return_mixing_diagnostics=False,
    ):
        depths = self.depths if depths is None else depths
        depths = depths.flatten().to(device=forcing_features.device, dtype=forcing_features.dtype)
        if forcing_context is None:
            forcing_context = self._encode_forcing_context(forcing_features, forcing_history)
        depth_norm = depths / torch.clamp(depths[-1], min=1.0)
        background_nn_kz, turbulent_nn_kz, nn_kd_multiplier, residual = self.param_net(
            depth_norm,
            forcing_context,
            static_features,
        )
        kd_base = self._kd_base_from_forcing_features(forcing_features)
        kd = kd_base * nn_kd_multiplier
        kd_prior_regularization_loss = torch.log(torch.clamp(nn_kd_multiplier, min=1.0e-6)).pow(2)
        if kd_multiplier is not None:
            kd_multiplier = torch.as_tensor(
                kd_multiplier,
                device=forcing_features.device,
                dtype=forcing_features.dtype,
            ).reshape(-1)
            if kd_multiplier.shape[0] == 1 and kd.shape[0] > 1:
                kd_multiplier = kd_multiplier.expand(kd.shape[0])
            kd = kd * kd_multiplier
        kd = torch.clamp(kd, min=MIN_SHORTWAVE_ATTENUATION, max=MAX_SHORTWAVE_ATTENUATION)
        wind = torch.clamp(wind_speed.reshape(-1, 1), min=0.1)
        if wind_kz_scale is None:
            wind_kz_scale = torch.full(
                (wind.shape[0], 1),
                float(self.wind_kz_scale),
                device=forcing_features.device,
                dtype=forcing_features.dtype,
            )
        else:
            wind_kz_scale = torch.as_tensor(
                wind_kz_scale,
                device=forcing_features.device,
                dtype=forcing_features.dtype,
            ).reshape(-1, 1)
            if wind_kz_scale.shape[0] == 1 and wind.shape[0] > 1:
                wind_kz_scale = wind_kz_scale.expand(wind.shape[0], 1)
        shape_wind_factor, shape_decay_depth_m = self._shape_mixing_terms(static_features, depths)
        if shape_wind_factor.shape[0] == 1 and wind.shape[0] > 1:
            shape_wind_factor = shape_wind_factor.expand(wind.shape[0])
            shape_decay_depth_m = shape_decay_depth_m.expand(wind.shape[0])
        wind_kz = (
            wind_kz_scale
            * shape_wind_factor.reshape(-1, 1)
            * DIFFUSIVITY_K0
            * (0.2 + wind.pow(1.5))
            * torch.exp(-depths.reshape(1, -1) / shape_decay_depth_m.reshape(-1, 1))
        )
        if ice_fraction is not None:
            ice_fraction = torch.clamp(
                ice_fraction.to(device=forcing_features.device, dtype=forcing_features.dtype).reshape(-1, 1),
                0.0,
                1.0,
            )
            if ice_fraction.shape[0] == 1 and wind_kz.shape[0] > 1:
                ice_fraction = ice_fraction.expand(wind_kz.shape[0], 1)
            wind_kz = wind_kz * (1.0 - ice_fraction).pow(2)
        if temperature is None:
            stability_factor = 1.0
            convective_kz = 0.0
            richardson = torch.zeros_like(wind_kz)
            n2 = torch.zeros_like(wind_kz)
            stratification_gate = torch.ones_like(wind_kz)
        else:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(0)
            n2, richardson = self._density_stability_terms(temperature, depths, wind.reshape(-1))
            stable_ri = torch.clamp(richardson, min=0.0, max=50.0)
            stability_factor = (1.0 + DIFFUSIVITY_RI_SENSITIVITY * stable_ri).pow(-1.0)
            if (
                self.stratification_mixing_cap == 'on'
                and self.stratification_mixing_cap_strength > 0.0
            ):
                depth_fraction = depths / torch.clamp(depths[-1], min=1.0)
                depth_gate = torch.sigmoid((depth_fraction - 0.15) / 0.08).reshape(1, -1)
                n2_scaled = torch.clamp(torch.clamp(n2, min=0.0) / 1.0e-4, min=0.0, max=100.0)
                cap_strength = torch.as_tensor(
                    self.stratification_mixing_cap_strength,
                    device=forcing_features.device,
                    dtype=forcing_features.dtype,
                )
                stratification_gate = (1.0 + cap_strength * depth_gate * n2_scaled).pow(-1.0)
            else:
                stratification_gate = torch.ones_like(wind_kz)
            unstable_gate = torch.sigmoid(
                (-n2 - DIFFUSIVITY_UNSTABLE_GRADIENT_EPS)
                / DIFFUSIVITY_UNSTABLE_GRADIENT_WIDTH
            )
            convective_kz = (
                self.autumn_convective_boost
                * unstable_gate
                * (DIFFUSIVITY_UNSTABLE_BOOST - 1.0)
                * MOLECULAR_DIFFUSIVITY
            )
            if convective_mixing_scale is not None:
                convective_mixing_scale = torch.as_tensor(
                    convective_mixing_scale,
                    device=forcing_features.device,
                    dtype=forcing_features.dtype,
                ).reshape(-1, 1)
                if convective_mixing_scale.shape[0] == 1 and convective_kz.shape[0] > 1:
                    convective_mixing_scale = convective_mixing_scale.expand(convective_kz.shape[0], 1)
                convective_kz = convective_kz * convective_mixing_scale
        gated_turbulent_nn_kz = turbulent_nn_kz * stability_factor * stratification_gate
        kz = torch.clamp(
            (
                MOLECULAR_DIFFUSIVITY
                + background_nn_kz
                + (wind_kz + turbulent_nn_kz) * stability_factor * stratification_gate
                + convective_kz
            ),
            min=MIN_TOTAL_DIFFUSIVITY,
            max=MAX_TOTAL_DIFFUSIVITY,
        )
        if return_mixing_diagnostics:
            deep_mask = depths >= (0.7 * torch.clamp(depths[-1], min=1.0))
            if not torch.any(deep_mask):
                deep_mask = torch.ones_like(depths, dtype=torch.bool)
            mixing_diagnostics = {
                'lake_shape_wind_factor': shape_wind_factor.reshape(-1),
                'lake_shape_decay_depth_m': shape_decay_depth_m.reshape(-1),
                'stratification_mixing_gate_mean': stratification_gate.mean(dim=1),
                'stratification_mixing_gate_min': stratification_gate.min(dim=1).values,
                'stratification_mixing_gate_deep_mean': stratification_gate[:, deep_mask].mean(dim=1),
                'background_nn_kz_mean': background_nn_kz.mean(dim=1),
                'background_nn_kz_deep_mean': background_nn_kz[:, deep_mask].mean(dim=1),
                'turbulent_nn_kz_mean': turbulent_nn_kz.mean(dim=1),
                'turbulent_nn_kz_deep_mean': turbulent_nn_kz[:, deep_mask].mean(dim=1),
                'gated_turbulent_nn_kz_mean': gated_turbulent_nn_kz.mean(dim=1),
                'gated_turbulent_nn_kz_deep_mean': gated_turbulent_nn_kz[:, deep_mask].mean(dim=1),
                'kd_base': kd_base.reshape(-1),
                'nn_kd_multiplier': nn_kd_multiplier.reshape(-1),
                'kd_prior_regularization_loss': kd_prior_regularization_loss.reshape(-1),
            }
            return kz, kd, residual, mixing_diagnostics
        return kz, kd, residual

    def _kz_band_means(self, kz, depths):
        depths = depths.flatten().to(device=kz.device, dtype=kz.dtype)

        def masked_mean(mask):
            if torch.any(mask):
                return kz[:, mask].mean(dim=1)
            return kz.mean(dim=1)

        return {
            'kz_surface_mean': masked_mean(depths <= 3.0),
            'kz_mid_mean': masked_mean((depths > 3.0) & (depths <= 10.0)),
            'kz_deep_mean': masked_mean(depths > 10.0),
        }

    def heat_content_j_m2(self, temperature, depths=None, area_profile=None):
        """Area-weighted heat content per unit surface area."""
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(0)
        depths = self.depths if depths is None else depths
        area_profile = self.area_profile if area_profile is None else area_profile
        depths = depths.to(device=temperature.device, dtype=temperature.dtype)
        area = area_profile.to(device=temperature.device, dtype=temperature.dtype)
        dz = layer_thicknesses(depths)
        surface_area = torch.clamp(area[0], min=1.0e-6)
        weights = area.reshape(1, -1) * dz.reshape(1, -1) / surface_area
        return RHO_CP * torch.sum(temperature * weights, dim=1)

    def step(
        self,
        temperature,
        forcing_row,
        static_features,
        dt_seconds=SECONDS_PER_DAY,
        next_forcing_row=None,
        return_diagnostics=False,
        task_mode='analysis',
        depths=None,
        area_profile=None,
        hard_density_stability=False,
        diagnostic_mode='full',
        freezing_storage_j_m2=None,
        return_freezing_storage=False,
        freezing_energy_mode=None,
        fewshot_adapter=None,
    ):
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(0)
        freezing_energy_mode = normalize_freezing_energy_mode(
            freezing_energy_mode or self.freezing_energy_mode
        )
        depths = self.depths if depths is None else torch.as_tensor(depths, device=temperature.device, dtype=temperature.dtype)
        area_profile = self.area_profile if area_profile is None else torch.as_tensor(area_profile, device=temperature.device, dtype=temperature.dtype)
        freezing_storage_before_profile = _coerce_freezing_storage(
            freezing_storage_j_m2,
            temperature,
            surface_only=freezing_energy_mode == 'latent_reservoir',
        )
        active_forcing = _average_forcing_rows_for_task(forcing_row, next_forcing_row, task_mode=task_mode)
        forcing_features = active_forcing['features']
        forcing_history = active_forcing.get('history_features')
        static_features = static_features.to(device=temperature.device, dtype=temperature.dtype)
        if static_features.ndim == 1:
            static_features = static_features.unsqueeze(0).expand(temperature.shape[0], -1)
        forcing_features = forcing_features.to(device=temperature.device, dtype=temperature.dtype)
        if forcing_features.ndim == 1:
            forcing_features = forcing_features.unsqueeze(0).expand(temperature.shape[0], -1)
        if forcing_history is not None:
            forcing_history = forcing_history.to(device=temperature.device, dtype=temperature.dtype)
            if forcing_history.ndim == 2:
                forcing_history = forcing_history.unsqueeze(0).expand(temperature.shape[0], -1, -1)
        forcing_features, forcing_history, lst_feature_dropout_mask = self._apply_lst_feature_dropout(
            forcing_features,
            forcing_history,
        )
        forcing_context = self._encode_forcing_context(forcing_features, forcing_history)
        adaptive_values, adaptive_regularization = self._adaptive_parameter_values(
            static_features,
            forcing_context=forcing_context,
            forcing_features=forcing_features,
            fewshot_adapter=fewshot_adapter,
        )
        effective_wind_kz_scale = adaptive_values['wind_kz_scale']
        effective_blend_alpha = adaptive_values['blend_alpha']
        effective_kd_multiplier = adaptive_values['kd_multiplier']
        effective_turbulent_exchange_scale = adaptive_values['turbulent_exchange_scale']
        effective_convective_mixing_scale = adaptive_values['convective_mixing_scale']
        effective_ice_shortwave_scale = adaptive_values['ice_shortwave_scale']

        wind_speed = active_forcing['wind_speed'].to(device=temperature.device, dtype=temperature.dtype)
        if wind_speed.ndim == 0:
            wind_speed = wind_speed.reshape(1).expand(temperature.shape[0])
        ice_fraction = active_forcing.get('ice_fraction')
        if ice_fraction is not None:
            ice_fraction = ice_fraction.to(device=temperature.device, dtype=temperature.dtype)
        kz, kd, residual, mixing_diagnostics = self.predict_params(
            forcing_features,
            static_features,
            wind_speed,
            temperature=temperature,
            forcing_history=forcing_history,
            depths=depths,
            ice_fraction=ice_fraction,
            forcing_context=forcing_context,
            wind_kz_scale=effective_wind_kz_scale,
            kd_multiplier=effective_kd_multiplier,
            convective_mixing_scale=effective_convective_mixing_scale,
            return_mixing_diagnostics=True,
        )

        surface_temp = temperature[:, 0]
        base_shortwave = active_forcing['shortwave'].to(surface_temp).reshape(-1)
        if base_shortwave.numel() == 1 and temperature.shape[0] > 1:
            base_shortwave = base_shortwave.expand(temperature.shape[0])
        base_open_water_flux = compute_surface_flux_terms(
            surface_temp,
            {
                'surface_air_temp': active_forcing['air_temp'].to(surface_temp),
                'surface_wind_speed': wind_speed,
                'surface_relative_humidity': active_forcing['relative_humidity'].to(surface_temp),
                'surface_pressure': active_forcing['surface_pressure'].to(surface_temp),
                'surface_shortwave': base_shortwave,
                'surface_longwave': active_forcing['longwave'].to(surface_temp),
                'surface_latent_heat': active_forcing['latent_heat'].to(surface_temp),
                'surface_sensible_heat': active_forcing['sensible_heat'].to(surface_temp),
                'surface_ice_fraction': torch.zeros_like(surface_temp),
            },
            shortwave_surface_fraction=self.shortwave_surface_fraction,
            turbulent_flux_mode=self.turbulent_flux_mode,
            turbulent_flux_blend_alpha=effective_blend_alpha,
            turbulent_exchange_scale=effective_turbulent_exchange_scale,
        )
        base_open_water_flux_wm2 = (
            base_open_water_flux['net_radiation']
            - base_open_water_flux['sensible_heat']
            - base_open_water_flux['latent_heat']
        )
        if ice_fraction is not None:
            ice_gate = torch.clamp(ice_fraction.reshape(-1).to(surface_temp), 0.0, 1.0)
        else:
            ice_gate = torch.clamp(
                active_forcing.get('ice_mask', torch.zeros_like(surface_temp)).to(surface_temp).reshape(-1),
                0.0,
                1.0,
            )
        air_temp_for_ice = active_forcing['air_temp'].to(surface_temp).reshape(-1)
        if air_temp_for_ice.numel() == 1 and surface_temp.numel() > 1:
            air_temp_for_ice = air_temp_for_ice.expand_as(surface_temp)
        ice_skin = active_forcing.get('ist_snow_ice')
        if ice_skin is None:
            ice_skin = air_temp_for_ice
        else:
            ice_skin = ice_skin.to(surface_temp).reshape(-1)
            if ice_skin.numel() == 1 and surface_temp.numel() > 1:
                ice_skin = ice_skin.expand_as(surface_temp)
            ice_skin = torch.where(torch.isfinite(ice_skin), ice_skin, air_temp_for_ice)
        ice_flux_wm2 = ice_conductive_flux_wm2(
            torch.clamp(surface_temp, min=0.0, max=0.5),
            ice_skin,
            snow_depth_m=active_forcing.get('snow_depth'),
            ice_thickness_m=active_forcing.get('ice_thickness'),
        )
        base_surface_flux_wm2 = (1.0 - ice_gate) * base_open_water_flux_wm2 + ice_gate * ice_flux_wm2
        shortwave_scale, cooling_scale_raw, surface_flux_bias_wm2 = self.physical_scale_head(
            forcing_context,
            static_features,
            forcing_features,
            base_surface_flux_wm2,
            ice_gate,
        )
        scaled_shortwave = base_shortwave * shortwave_scale
        open_water_ice_indicator = base_open_water_flux['ice_indicator'].reshape_as(surface_temp)
        open_water_albedo = (
            float(SURFACE_ALBEDO_WATER) * (1.0 - open_water_ice_indicator)
            + float(SURFACE_ALBEDO_ICE) * open_water_ice_indicator
        )
        shortwave_net_delta = (
            float(self.shortwave_surface_fraction)
            * (1.0 - open_water_albedo)
            * (scaled_shortwave - base_shortwave)
        )
        open_water_net_radiation_wm2 = base_open_water_flux['net_radiation'] + shortwave_net_delta
        open_water_sensible_heat_wm2 = base_open_water_flux['sensible_heat']
        open_water_latent_heat_wm2 = base_open_water_flux['latent_heat']
        open_water_flux_wm2 = (
            open_water_net_radiation_wm2
            - open_water_sensible_heat_wm2
            - open_water_latent_heat_wm2
        )
        surface_flux_wm2 = (1.0 - ice_gate) * open_water_flux_wm2 + ice_gate * ice_flux_wm2
        cooling_multiplier = torch.where(
            surface_flux_wm2 < 0.0,
            cooling_scale_raw.to(surface_flux_wm2),
            torch.ones_like(surface_flux_wm2),
        )
        surface_flux_wm2 = surface_flux_wm2 * cooling_multiplier + surface_flux_bias_wm2.to(surface_flux_wm2)
        snow_depth = active_forcing.get('snow_depth')
        if snow_depth is None:
            snow_depth = torch.zeros_like(surface_flux_wm2)
        else:
            snow_depth = snow_depth.to(device=surface_flux_wm2.device, dtype=surface_flux_wm2.dtype).reshape(-1)
        ice_thickness = active_forcing.get('ice_thickness')
        if ice_thickness is None:
            ice_thickness = torch.zeros_like(surface_flux_wm2)
        else:
            ice_thickness = ice_thickness.to(device=surface_flux_wm2.device, dtype=surface_flux_wm2.dtype).reshape(-1)
        if snow_depth.numel() == 1 and surface_flux_wm2.numel() > 1:
            snow_depth = snow_depth.expand_as(surface_flux_wm2)
        if ice_thickness.numel() == 1 and surface_flux_wm2.numel() > 1:
            ice_thickness = ice_thickness.expand_as(surface_flux_wm2)
        effective_ice_thickness = torch.where(
            ice_thickness > 0.0,
            ice_thickness,
            (0.05 + 0.45 * ice_gate) * (ice_gate > 0.0).to(surface_flux_wm2),
        )
        ice_shortwave_transmission = torch.exp(
            -effective_ice_shortwave_scale.reshape(-1).to(surface_flux_wm2)
            * (
                float(SNOW_SHORTWAVE_ATTENUATION_M_INV) * torch.clamp(snow_depth, min=0.0)
                + float(ICE_SHORTWAVE_ATTENUATION_M_INV) * torch.clamp(effective_ice_thickness, min=0.0)
            )
        )
        shortwave_to_water = scaled_shortwave * (
            (1.0 - ice_gate) + ice_gate * ice_shortwave_transmission
        )
        source = one_day_heat_sources(
            depths.to(device=temperature.device, dtype=temperature.dtype),
            surface_flux_wm2=surface_flux_wm2,
            shortwave_wm2=shortwave_to_water,
            kd=kd,
            shortwave_surface_fraction=self.shortwave_surface_fraction,
            area_profile=area_profile.to(device=temperature.device, dtype=temperature.dtype),
        )
        advective_source, advective_diagnostics = self._advective_heat_source(
            temperature,
            forcing_features,
            active_forcing,
            static_features,
            depths,
            area_profile,
            dt_seconds,
        )
        source = source + advective_source
        heat_before = self.heat_content_j_m2(temperature, depths=depths, area_profile=area_profile)
        depths = depths.to(device=temperature.device, dtype=temperature.dtype)
        area = area_profile.to(device=temperature.device, dtype=temperature.dtype)
        dz = layer_thicknesses(depths)
        layer_heat_capacity_j_m2_c = (
            RHO_CP * area.reshape(-1) * dz.reshape(-1)
        ) / torch.clamp(area[0], min=1.0e-6)
        freezing_storage_before_profile = freezing_storage_before_profile.to(
            device=temperature.device,
            dtype=temperature.dtype,
        )
        freezing_storage_before_j_m2 = freezing_storage_before_profile.sum(dim=1)
        heat_input_wm2 = torch.sum(
            source * RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1),
            dim=1,
        ) / torch.clamp(area[0], min=1.0e-6)
        next_temperature = implicit_diffusion_step(
            temperature,
            depths,
            area,
            kz,
            source_c_per_s=source,
            dt_seconds=dt_seconds,
        )
        residual_raw = residual
        residual = remove_area_weighted_mean(residual_raw, depths, area)
        unclamped_temperature = next_temperature + residual
        if freezing_energy_mode == 'latent_reservoir':
            floor_clamped_temperature, freezing_storage_after_profile = _apply_latent_reservoir_floor(
                unclamped_temperature,
                freezing_storage_before_profile,
                layer_heat_capacity_j_m2_c,
            )
            temperature_floor_heat_injection_j_m2 = torch.zeros(
                temperature.shape[0],
                device=temperature.device,
                dtype=temperature.dtype,
            )
        else:
            floor_clamped_temperature = torch.clamp(unclamped_temperature, min=0.0)
            floor_delta_c = floor_clamped_temperature - unclamped_temperature
            temperature_floor_heat_injection_j_m2 = torch.sum(
                floor_delta_c * RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1),
                dim=1,
            ) / torch.clamp(area[0], min=1.0e-6)
            freezing_storage_after_profile = torch.zeros_like(freezing_storage_before_profile)
        next_temperature = torch.clamp(floor_clamped_temperature, max=40.0)
        ceiling_delta_c = floor_clamped_temperature - next_temperature
        temperature_ceiling_heat_removal_j_m2 = torch.sum(
            ceiling_delta_c * RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1),
            dim=1,
        ) / torch.clamp(area[0], min=1.0e-6)
        density_adjustment_diagnostics = {
            'density_adjustment_applied': torch.zeros(next_temperature.shape[0], device=next_temperature.device, dtype=next_temperature.dtype),
            'density_adjustment_max_delta_c': torch.zeros(next_temperature.shape[0], device=next_temperature.device, dtype=next_temperature.dtype),
            'density_adjustment_heat_delta_j_m2': torch.zeros(next_temperature.shape[0], device=next_temperature.device, dtype=next_temperature.dtype),
        }
        if bool(hard_density_stability):
            next_temperature, density_adjustment_diagnostics = heat_conserving_convective_adjustment(
                next_temperature,
                depths,
                area,
            )
        diagnostic_mode = str(diagnostic_mode or 'full').strip().lower()
        if diagnostic_mode not in {'none', 'loss', 'full'}:
            raise ValueError("diagnostic_mode must be one of: none, loss, full.")
        if not return_diagnostics:
            if return_freezing_storage:
                return next_temperature, freezing_storage_after_profile
            return next_temperature
        if diagnostic_mode == 'none':
            if return_freezing_storage:
                return next_temperature, freezing_storage_after_profile, {}
            return next_temperature, {}
        heat_after = self.heat_content_j_m2(next_temperature, depths=depths, area_profile=area)
        freezing_storage_after_j_m2 = freezing_storage_after_profile.sum(dim=1)
        freezing_storage_deep_j_m2 = (
            freezing_storage_after_profile[:, 1:].sum(dim=1)
            if freezing_storage_after_profile.shape[1] > 1
            else torch.zeros_like(freezing_storage_after_j_m2)
        )
        storage_fraction_denominator = torch.clamp(freezing_storage_after_j_m2, min=1.0e-12)
        freezing_storage_surface_fraction = torch.where(
            freezing_storage_after_j_m2 > 0.0,
            freezing_storage_after_profile[:, 0] / storage_fraction_denominator,
            torch.zeros_like(freezing_storage_after_j_m2),
        )
        freezing_storage_deep_fraction = torch.where(
            freezing_storage_after_j_m2 > 0.0,
            freezing_storage_deep_j_m2 / storage_fraction_denominator,
            torch.zeros_like(freezing_storage_after_j_m2),
        )
        sensible_heat_tendency_wm2 = (heat_after - heat_before) / float(dt_seconds)
        freezing_storage_change_wm2 = (
            freezing_storage_after_j_m2 - freezing_storage_before_j_m2
        ) / float(dt_seconds)
        effective_heat_tendency_wm2 = sensible_heat_tendency_wm2 - freezing_storage_change_wm2
        deep_mask = depths >= (0.7 * torch.clamp(depths[-1], min=1.0))
        if not torch.any(deep_mask):
            deep_mask = torch.ones_like(depths, dtype=torch.bool)
        diagnostics = {
            'shortwave_absorption_scale': shortwave_scale.reshape(-1),
            'surface_flux_bias_wm2': surface_flux_bias_wm2.reshape(-1),
            'surface_cooling_scale_raw': cooling_scale_raw.reshape(-1),
            'surface_cooling_scale': cooling_multiplier,
            'heat_input_wm2': heat_input_wm2,
            'heat_tendency_wm2': effective_heat_tendency_wm2,
            'sensible_heat_tendency_wm2': sensible_heat_tendency_wm2,
            'effective_heat_tendency_wm2': effective_heat_tendency_wm2,
            'freezing_storage_j_m2': freezing_storage_after_j_m2,
            'freezing_storage_ice_j_m2': freezing_storage_after_j_m2,
            'freezing_storage_before_j_m2': freezing_storage_before_j_m2,
            'freezing_storage_change_wm2': freezing_storage_change_wm2,
            'freezing_storage_profile_j_m2': freezing_storage_after_profile,
            'freezing_storage_surface_fraction': freezing_storage_surface_fraction,
            'freezing_storage_deep_fraction': freezing_storage_deep_fraction,
            'temperature_floor_heat_injection_j_m2': temperature_floor_heat_injection_j_m2,
            'temperature_floor_heat_injection_wm2': temperature_floor_heat_injection_j_m2 / float(dt_seconds),
            'lst_feature_dropout_probability': torch.full_like(
                freezing_storage_after_j_m2,
                float(self.lst_feature_dropout_probability),
            ),
            'lst_feature_dropout_applied': lst_feature_dropout_mask.reshape(-1).to(freezing_storage_after_j_m2),
            'adaptive_wind_kz_scale': effective_wind_kz_scale.reshape(-1),
            'adaptive_turbulent_flux_blend_alpha': effective_blend_alpha.reshape(-1),
            'adaptive_kd_multiplier': effective_kd_multiplier.reshape(-1),
            'adaptive_turbulent_exchange_scale': effective_turbulent_exchange_scale.reshape(-1),
            'adaptive_convective_mixing_scale': effective_convective_mixing_scale.reshape(-1),
            'adaptive_ice_shortwave_scale': effective_ice_shortwave_scale.reshape(-1),
            'adaptive_parameter_regularization_loss': adaptive_regularization.reshape(-1),
            'residual_mean_c': residual.mean(dim=1),
            'residual_abs_mean_c': residual.abs().mean(dim=1),
            'residual_surface_c': residual[:, 0],
            'residual_deep_mean_c': residual[:, deep_mask].mean(dim=1),
            'residual_profile_c': residual,
            **advective_diagnostics,
            **mixing_diagnostics,
            **density_adjustment_diagnostics,
        }
        if diagnostic_mode == 'full':
            kz_band_means = self._kz_band_means(kz, depths)
            diagnostics.update({
                'surface_flux_wm2': surface_flux_wm2,
                'open_water_surface_flux_wm2': open_water_flux_wm2,
                'open_water_net_radiation_wm2': open_water_net_radiation_wm2,
                'open_water_sensible_heat_wm2': open_water_sensible_heat_wm2,
                'open_water_latent_heat_wm2': open_water_latent_heat_wm2,
                'open_water_sensible_heat_bulk_wm2': base_open_water_flux['sensible_heat_bulk'],
                'open_water_latent_heat_bulk_wm2': base_open_water_flux['latent_heat_bulk'],
                'open_water_sensible_heat_bulk_unscaled_wm2': base_open_water_flux['sensible_heat_bulk_unscaled'],
                'open_water_latent_heat_bulk_unscaled_wm2': base_open_water_flux['latent_heat_bulk_unscaled'],
                'turbulent_exchange_scale': base_open_water_flux['turbulent_exchange_scale'],
                'ice_conductive_flux_wm2': ice_flux_wm2,
                'ice_fraction': ice_gate,
                'shortwave_wm2': scaled_shortwave.reshape(-1),
                'shortwave_to_water_wm2': shortwave_to_water.reshape(-1),
                'ice_shortwave_transmission': ice_shortwave_transmission.reshape(-1),
                'turbulent_flux_blend_alpha': effective_blend_alpha.reshape_as(surface_flux_wm2),
                'heat_content_before_j_m2': heat_before,
                'heat_content_after_j_m2': heat_after,
                'effective_heat_content_before_j_m2': heat_before - freezing_storage_before_j_m2,
                'effective_heat_content_after_j_m2': heat_after - freezing_storage_after_j_m2,
                'temperature_ceiling_heat_removal_j_m2': temperature_ceiling_heat_removal_j_m2,
                'temperature_ceiling_heat_removal_wm2': temperature_ceiling_heat_removal_j_m2 / float(dt_seconds),
                'kd': kd,
                'kz_mean': kz.mean(dim=1),
                **kz_band_means,
                'residual_raw_mean_c': residual_raw.mean(dim=1),
            })
        if return_freezing_storage:
            return next_temperature, freezing_storage_after_profile, diagnostics
        return next_temperature, diagnostics

    def rollout(self, initial_profile, forcing_rows, static_features, task_mode='analysis', depths=None, area_profile=None):
        states = []
        current = initial_profile
        if current.ndim == 1:
            current = current.unsqueeze(0)
        freezing_storage = torch.zeros_like(current)
        for row_idx, forcing_row in enumerate(forcing_rows):
            next_row = forcing_rows[row_idx + 1] if row_idx + 1 < len(forcing_rows) else None
            current, freezing_storage = self.step(
                current,
                forcing_row,
                static_features,
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=depths,
                area_profile=area_profile,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
            states.append(current)
        return torch.cat(states, dim=0) if states else current

    def rollout_batch(
        self,
        initial_profile,
        forcing_rows,
        static_features,
        *,
        next_forcing_rows=None,
        task_mode='analysis',
        depths=None,
        area_profile=None,
        hard_density_stability=False,
        return_diagnostics=False,
        diagnostic_mode='none',
    ):
        """Roll a same-lake batch through a pre-batched forcing sequence."""
        states = []
        diagnostics = []
        current = initial_profile
        if current.ndim == 1:
            current = current.unsqueeze(0)
        freezing_storage = torch.zeros_like(current)
        for row_idx, forcing_row in enumerate(forcing_rows):
            if next_forcing_rows is not None:
                next_row = next_forcing_rows[row_idx]
            else:
                next_row = forcing_rows[row_idx + 1] if row_idx + 1 < len(forcing_rows) else None
            step_result = self.step(
                current,
                forcing_row,
                static_features,
                next_forcing_row=next_row,
                task_mode=task_mode,
                depths=depths,
                area_profile=area_profile,
                hard_density_stability=hard_density_stability,
                return_diagnostics=return_diagnostics,
                diagnostic_mode=diagnostic_mode,
                freezing_storage_j_m2=freezing_storage,
                return_freezing_storage=True,
            )
            if return_diagnostics:
                current, freezing_storage, step_diagnostics = step_result
                diagnostics.append(step_diagnostics)
            else:
                current, freezing_storage = step_result
            states.append(current)
        stacked = torch.stack(states, dim=0) if states else current.unsqueeze(0)
        if return_diagnostics:
            return stacked, diagnostics
        return stacked
