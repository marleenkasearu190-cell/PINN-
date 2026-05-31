"""Physical helper functions shared by the reconstruction-state model."""

import numpy as np
import torch

from .constants import (
    AIR_HEAT_CAPACITY,
    ATMOSPHERIC_EMISSIVITY,
    ICE_TRANSITION_EPS,
    LATENT_HEAT_VAPORIZATION,
    RHO_CP,
    SHORTWAVE_SURFACE_FRACTION,
    STEFAN_BOLTZMANN,
    SURFACE_ALBEDO_ICE,
    SURFACE_ALBEDO_WATER,
    TRANSFER_COEFF_HEAT,
    TRANSFER_COEFF_MOISTURE,
    WATER_EMISSIVITY,
)

def saturation_vapor_pressure_np(temp_c):
    return 610.94 * np.exp((17.625 * temp_c) / (temp_c + 243.04))


def saturation_vapor_pressure_torch(temp_c: torch.Tensor) -> torch.Tensor:
    return 610.94 * torch.exp((17.625 * temp_c) / (temp_c + 243.04))


def specific_humidity_from_vapor_pressure_torch(vapor_pressure_pa: torch.Tensor, pressure_pa: torch.Tensor) -> torch.Tensor:
    return 0.622 * vapor_pressure_pa / (pressure_pa - 0.378 * vapor_pressure_pa).clamp(min=1.0)


def water_density_torch(temp_c: torch.Tensor) -> torch.Tensor:
    return 1000.0 * (
        1.0
        - ((temp_c + 288.9414) / (508929.2 * (temp_c + 68.12963)))
        * (temp_c - 3.9863) ** 2
    )


def water_density_numpy(temp_c) -> np.ndarray:
    temp_c = np.asarray(temp_c, dtype=np.float64)
    return 1000.0 * (
        1.0
        - ((temp_c + 288.9414) / (508929.2 * (temp_c + 68.12963)))
        * np.square(temp_c - 3.9863)
    )


def project_temperature_profile_to_stable_density(temp_profile, max_iterations: int = 256):
    temp_profile = np.asarray(temp_profile, dtype=np.float64).copy()
    if temp_profile.size < 2:
        return temp_profile, 0

    adjustments = 0
    for _ in range(max_iterations):
        density_profile = water_density_numpy(temp_profile)
        unstable_idx = np.where(np.diff(density_profile) < -1e-9)[0]
        if unstable_idx.size == 0:
            break
        for idx in unstable_idx:
            mixed_temp = 0.5 * (temp_profile[idx] + temp_profile[idx + 1])
            temp_profile[idx] = mixed_temp
            temp_profile[idx + 1] = mixed_temp
            adjustments += 1
    return temp_profile, adjustments


def apply_autumn_cooling_adjustment(*args, **kwargs):
    state_upd = np.asarray(kwargs.get('state_upd', args[1] if len(args) > 1 else None), dtype=np.float64)
    return state_upd, 0.0


def smooth_ice_indicator(temp_surface_c: torch.Tensor, transition_eps: float = ICE_TRANSITION_EPS) -> torch.Tensor:
    return torch.sigmoid((-temp_surface_c) / transition_eps)


def _use_provided_flux(batch, key, reference):
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=reference.device, dtype=reference.dtype)
    else:
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = torch.full_like(reference, float(tensor))
    elif tensor.shape != reference.shape:
        tensor = tensor.expand_as(reference)
    if bool(torch.any(torch.isfinite(tensor) & (torch.abs(tensor) > 1.0e-6)).detach().cpu()):
        return tensor
    return None


TURBULENT_FLUX_MODES = {'bulk', 'provided', 'blend'}


def normalize_turbulent_flux_mode(mode):
    mode = str(mode or 'bulk').strip().lower()
    if mode not in TURBULENT_FLUX_MODES:
        raise ValueError("turbulent_flux_mode must be one of: bulk, provided, blend.")
    return mode


def compute_surface_flux_terms(
    surface_temp,
    batch,
    shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
    *,
    turbulent_flux_mode='provided',
    turbulent_flux_blend_alpha=1.0,
):
    air_temp = batch['surface_air_temp']
    air_temp_k = air_temp + 273.15
    surface_temp_k = surface_temp + 273.15
    wind_speed = batch['surface_wind_speed'].clamp(min=0.1)
    surface_pressure = batch['surface_pressure'].clamp(min=80000.0, max=110000.0)

    air_density = surface_pressure / (287.05 * air_temp_k.clamp(min=200.0))
    vapor_pressure_air = batch['surface_relative_humidity'].clamp(0.2, 1.0) * saturation_vapor_pressure_torch(air_temp)
    vapor_pressure_surface = saturation_vapor_pressure_torch(surface_temp)
    q_air = specific_humidity_from_vapor_pressure_torch(vapor_pressure_air, surface_pressure)
    q_surface = specific_humidity_from_vapor_pressure_torch(vapor_pressure_surface, surface_pressure)

    sensible_heat_bulk = air_density * AIR_HEAT_CAPACITY * TRANSFER_COEFF_HEAT * wind_speed * (surface_temp - air_temp)
    latent_heat_bulk = air_density * LATENT_HEAT_VAPORIZATION * TRANSFER_COEFF_MOISTURE * wind_speed * (q_surface - q_air)
    sensible_heat_provided = _use_provided_flux(batch, 'surface_sensible_heat', surface_temp)
    latent_heat_provided = _use_provided_flux(batch, 'surface_latent_heat', surface_temp)
    mode = normalize_turbulent_flux_mode(turbulent_flux_mode)
    blend_alpha = float(np.clip(turbulent_flux_blend_alpha, 0.0, 1.0))
    if mode == 'bulk' or sensible_heat_provided is None:
        sensible_heat = sensible_heat_bulk
    elif mode == 'blend':
        sensible_heat = blend_alpha * sensible_heat_provided + (1.0 - blend_alpha) * sensible_heat_bulk
    else:
        sensible_heat = sensible_heat_provided
    if mode == 'bulk' or latent_heat_provided is None:
        latent_heat = latent_heat_bulk
    elif mode == 'blend':
        latent_heat = blend_alpha * latent_heat_provided + (1.0 - blend_alpha) * latent_heat_bulk
    else:
        latent_heat = latent_heat_provided

    explicit_ice = batch.get('surface_ice_fraction')
    if explicit_ice is None:
        explicit_ice = batch.get('surface_ice_mask')
    if explicit_ice is not None:
        ice_indicator = torch.as_tensor(explicit_ice, device=surface_temp.device, dtype=surface_temp.dtype)
        if ice_indicator.ndim == 0:
            ice_indicator = torch.full_like(surface_temp, float(ice_indicator))
        elif ice_indicator.numel() == 1 and surface_temp.numel() > 1:
            ice_indicator = torch.full_like(surface_temp, float(ice_indicator.reshape(-1)[0]))
        else:
            ice_indicator = ice_indicator.reshape_as(surface_temp).clamp(0.0, 1.0)
    else:
        ice_indicator = smooth_ice_indicator(surface_temp)
    surface_albedo = (
        SURFACE_ALBEDO_WATER * (1.0 - ice_indicator) +
        SURFACE_ALBEDO_ICE * ice_indicator
    )
    shortwave_surface_fraction = float(np.clip(shortwave_surface_fraction, 0.0, 1.0))
    absorbed_surface_shortwave = shortwave_surface_fraction * (1.0 - surface_albedo) * batch['surface_shortwave']
    downwelling_longwave = _use_provided_flux(batch, 'surface_longwave', surface_temp)
    if downwelling_longwave is None:
        downwelling_longwave = ATMOSPHERIC_EMISSIVITY * STEFAN_BOLTZMANN * air_temp_k ** 4
    longwave_net = downwelling_longwave - WATER_EMISSIVITY * STEFAN_BOLTZMANN * surface_temp_k ** 4
    net_radiation = absorbed_surface_shortwave + longwave_net
    seb_flux = (net_radiation - sensible_heat - latent_heat) / RHO_CP

    return {
        'seb_flux': seb_flux,
        'ice_indicator': ice_indicator,
        'net_radiation': net_radiation,
        'sensible_heat': sensible_heat,
        'latent_heat': latent_heat,
        'sensible_heat_bulk': sensible_heat_bulk,
        'latent_heat_bulk': latent_heat_bulk,
        'sensible_heat_provided': sensible_heat_provided
        if sensible_heat_provided is not None else torch.full_like(surface_temp, torch.nan),
        'latent_heat_provided': latent_heat_provided
        if latent_heat_provided is not None else torch.full_like(surface_temp, torch.nan),
    }
