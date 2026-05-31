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
    PINN_MAX_DEPTH_REFERENCE_M,
    PINN_MAX_HEAT_FLUX_REFERENCE_W_M2,
    PINN_MAX_MEAN_DEPTH_REFERENCE_M,
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

HARD_DENSITY_STABILITY_MODES = {'auto', 'on', 'off'}
FREEZING_ENERGY_MODES = {'latent_reservoir', 'clamp'}
ICE_SHORTWAVE_ATTENUATION_M_INV = 1.50
SNOW_SHORTWAVE_ATTENUATION_M_INV = 20.0


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


def _coerce_freezing_storage(freezing_storage_j_m2, temperature):
    if freezing_storage_j_m2 is None:
        return torch.zeros_like(temperature)
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
    return torch.clamp(storage, min=0.0)


def _apply_latent_reservoir_floor(raw_temperature, freezing_storage_j_m2, layer_heat_capacity_j_m2_c):
    """Project temperature to non-negative values while conserving latent cold content."""
    capacity = torch.clamp(
        layer_heat_capacity_j_m2_c.to(device=raw_temperature.device, dtype=raw_temperature.dtype),
        min=1.0e-12,
    ).reshape(1, -1)
    positive_energy = torch.clamp(raw_temperature, min=0.0) * capacity
    cold_deficit = torch.clamp(-raw_temperature, min=0.0) * capacity
    melt_energy = torch.minimum(positive_energy, freezing_storage_j_m2)
    next_storage = torch.clamp(freezing_storage_j_m2 + cold_deficit - melt_energy, min=0.0)
    next_temperature = (positive_energy - melt_energy) / capacity
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
    return np.asarray(
        [
            float(max_depth) / PINN_MAX_DEPTH_REFERENCE_M,
            float(metadata.get('mean_depth_m', max_depth * 0.5)) / PINN_MAX_MEAN_DEPTH_REFERENCE_M,
            float(static['log_area']),
            float(static['latitude']),
            float(static['longitude']),
            float(static['elevation_norm']),
            float(metadata.get('volume_km3', 0.0)) / PINN_VOLUME_REFERENCE_KM3,
            float(static['light_extinction_norm']),
            float(static['fetch_norm']),
            float(static['wind_exposure_norm']),
            float(static['basin_shape_norm']),
        ],
        dtype=np.float32,
    )


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
    for key, value in current_row.items():
        if key in next_row:
            averaged[key] = 0.5 * (value + next_row[key].to(device=value.device, dtype=value.dtype))
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
        self.head = nn.Linear(hidden_dim, 3)

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
        nn_kz = 4.0e-5 * F.softplus(raw[..., 0])
        kd_profile = MIN_SHORTWAVE_ATTENUATION + (
            MAX_SHORTWAVE_ATTENUATION - MIN_SHORTWAVE_ATTENUATION
        ) * torch.sigmoid(raw[..., 1])
        residual = self.residual_limit_c * torch.tanh(raw[..., 2])
        return nn_kz, kd_profile.mean(dim=1), residual


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


class LakeStateForecaster(nn.Module):
    """State-space lake profile forecaster with a differentiable 1D solver."""

    def __init__(
        self,
        depths,
        area_profile,
        forcing_dim=len(FORCING_FEATURE_COLUMNS),
        static_dim=11,
        hidden_dim=96,
        forcing_context_dim=48,
        forcing_history_hidden_dim=48,
        residual_limit_c=0.50,
        shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
        wind_kz_scale=1.0,
        autumn_convective_boost=1.0,
        turbulent_flux_mode='bulk',
        turbulent_flux_blend_alpha=0.3,
        freezing_energy_mode='latent_reservoir',
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
        self.shortwave_surface_fraction = float(shortwave_surface_fraction)
        self.wind_kz_scale = float(wind_kz_scale)
        self.autumn_convective_boost = float(autumn_convective_boost)
        self.turbulent_flux_mode = normalize_turbulent_flux_mode(turbulent_flux_mode)
        self.turbulent_flux_blend_alpha = float(np.clip(turbulent_flux_blend_alpha, 0.0, 1.0))
        self.freezing_energy_mode = normalize_freezing_energy_mode(freezing_energy_mode)

    def _encode_forcing_context(self, forcing_features, forcing_history=None):
        if forcing_history is None:
            forcing_history = forcing_features.unsqueeze(1)
        forcing_history = forcing_history.to(device=forcing_features.device, dtype=forcing_features.dtype)
        if forcing_history.ndim == 2:
            forcing_history = forcing_history.unsqueeze(0)
        return self.forcing_encoder(forcing_history)

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
    ):
        depths = self.depths if depths is None else depths
        depths = depths.flatten().to(device=forcing_features.device, dtype=forcing_features.dtype)
        if forcing_context is None:
            forcing_context = self._encode_forcing_context(forcing_features, forcing_history)
        depth_norm = depths / torch.clamp(depths[-1], min=1.0)
        nn_kz, kd, residual = self.param_net(depth_norm, forcing_context, static_features)
        wind = torch.clamp(wind_speed.reshape(-1, 1), min=0.1)
        wind_kz = self.wind_kz_scale * DIFFUSIVITY_K0 * (0.2 + wind.pow(1.5)) * torch.exp(
            -depths.reshape(1, -1) / DIFFUSIVITY_WIND_DECAY_DEPTH
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
        else:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(0)
            n2, richardson = self._density_stability_terms(temperature, depths, wind.reshape(-1))
            stable_ri = torch.clamp(richardson, min=0.0, max=50.0)
            stability_factor = (1.0 + DIFFUSIVITY_RI_SENSITIVITY * stable_ri).pow(-1.0)
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
        kz = torch.clamp(
            MOLECULAR_DIFFUSIVITY + wind_kz * stability_factor + convective_kz + nn_kz,
            min=MIN_TOTAL_DIFFUSIVITY,
            max=MAX_TOTAL_DIFFUSIVITY,
        )
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
    ):
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(0)
        freezing_energy_mode = normalize_freezing_energy_mode(
            freezing_energy_mode or self.freezing_energy_mode
        )
        depths = self.depths if depths is None else torch.as_tensor(depths, device=temperature.device, dtype=temperature.dtype)
        area_profile = self.area_profile if area_profile is None else torch.as_tensor(area_profile, device=temperature.device, dtype=temperature.dtype)
        freezing_storage_before_profile = _coerce_freezing_storage(freezing_storage_j_m2, temperature)
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
        forcing_context = self._encode_forcing_context(forcing_features, forcing_history)

        wind_speed = active_forcing['wind_speed'].to(device=temperature.device, dtype=temperature.dtype)
        if wind_speed.ndim == 0:
            wind_speed = wind_speed.reshape(1).expand(temperature.shape[0])
        ice_fraction = active_forcing.get('ice_fraction')
        if ice_fraction is not None:
            ice_fraction = ice_fraction.to(device=temperature.device, dtype=temperature.dtype)
        kz, kd, residual = self.predict_params(
            forcing_features,
            static_features,
            wind_speed,
            temperature=temperature,
            forcing_history=forcing_history,
            depths=depths,
            ice_fraction=ice_fraction,
            forcing_context=forcing_context,
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
            turbulent_flux_blend_alpha=self.turbulent_flux_blend_alpha,
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
            -float(SNOW_SHORTWAVE_ATTENUATION_M_INV) * torch.clamp(snow_depth, min=0.0)
            -float(ICE_SHORTWAVE_ATTENUATION_M_INV) * torch.clamp(effective_ice_thickness, min=0.0)
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
            'freezing_storage_before_j_m2': freezing_storage_before_j_m2,
            'freezing_storage_change_wm2': freezing_storage_change_wm2,
            'freezing_storage_profile_j_m2': freezing_storage_after_profile,
            'temperature_floor_heat_injection_j_m2': temperature_floor_heat_injection_j_m2,
            'temperature_floor_heat_injection_wm2': temperature_floor_heat_injection_j_m2 / float(dt_seconds),
            'residual_mean_c': residual.mean(dim=1),
            'residual_abs_mean_c': residual.abs().mean(dim=1),
            'residual_surface_c': residual[:, 0],
            'residual_deep_mean_c': residual[:, deep_mask].mean(dim=1),
            'residual_profile_c': residual,
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
                'ice_conductive_flux_wm2': ice_flux_wm2,
                'ice_fraction': ice_gate,
                'shortwave_wm2': scaled_shortwave.reshape(-1),
                'shortwave_to_water_wm2': shortwave_to_water.reshape(-1),
                'ice_shortwave_transmission': ice_shortwave_transmission.reshape(-1),
                'turbulent_flux_blend_alpha': torch.full_like(surface_flux_wm2, self.turbulent_flux_blend_alpha),
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
