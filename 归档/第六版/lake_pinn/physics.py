# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .model import model_temperature

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


def compute_diffusivity(
    model,
    t_col,
    z_col,
    max_depth,
    wind_speed,
    water_depth,
    metadata=None,
    doy_sin=None,
    doy_cos=None,
    air_temp=None,
    shortwave=None,
    lst_surface=None,
    longwave=None,
    latent_heat=None,
    sensible_heat=None,
    secchi=None,
    air_temp_mean_7d=None,
    air_temp_mean_30d=None,
    shortwave_sum_7d=None,
    shortwave_sum_30d=None,
    wind_mean_7d=None,
    lst_mean_7d=None,
    heating_degree_days_30d=None,
    prev_surface_temp=None,
    prev_0_3m_mean=None,
    prev_deep_mean=None,
    water_level_anomaly=None,
    light_extinction_kd=None,
    effective_fetch=None,
    net_inflow=None,
):
    temp_pred = model_temperature(
        model,
        t_col,
        z_col,
        max_depth,
        metadata=metadata,
        doy_sin=doy_sin,
        doy_cos=doy_cos,
        air_temp=air_temp,
        wind_speed=wind_speed,
        shortwave=shortwave,
        lst_surface=lst_surface,
        longwave=longwave,
        latent_heat=latent_heat,
        sensible_heat=sensible_heat,
        secchi=secchi,
        air_temp_mean_7d=air_temp_mean_7d,
        air_temp_mean_30d=air_temp_mean_30d,
        shortwave_sum_7d=shortwave_sum_7d,
        shortwave_sum_30d=shortwave_sum_30d,
        wind_mean_7d=wind_mean_7d,
        lst_mean_7d=lst_mean_7d,
        heating_degree_days_30d=heating_degree_days_30d,
        prev_surface_temp=prev_surface_temp,
        prev_0_3m_mean=prev_0_3m_mean,
        prev_deep_mean=prev_deep_mean,
        water_level_anomaly=water_level_anomaly,
        light_extinction_kd=light_extinction_kd,
        effective_fetch=effective_fetch,
        net_inflow=net_inflow,
    )
    dT_dt_norm = torch.autograd.grad(temp_pred, t_col, grad_outputs=torch.ones_like(temp_pred), create_graph=True)[0]
    dT_dz = torch.autograd.grad(temp_pred, z_col, grad_outputs=torch.ones_like(temp_pred), create_graph=True)[0]

    density = water_density_torch(temp_pred)
    density_gradient = torch.autograd.grad(density, z_col, grad_outputs=torch.ones_like(density), create_graph=True)[0]

    depth_scale = torch.full_like(z_col, max(float(water_depth), 1.0))
    shear_term = RI_WIND_SHEAR_FACTOR * wind_speed.clamp(min=0.1) ** 2 / depth_scale ** 2
    # z is positive downward. Stable stratification has d(rho)/dz > 0, so
    # Richardson number should be positive and suppress eddy diffusivity.
    richardson_number = (GRAVITY / WATER_DENSITY) * density_gradient / shear_term.clamp(min=1e-8)

    stability_factor = (1.0 + DIFFUSIVITY_RI_SENSITIVITY * richardson_number).clamp(min=0.05)
    wind_decay = torch.exp(-z_col / DIFFUSIVITY_WIND_DECAY_DEPTH)
    wind_mixing = (
        DIFFUSIVITY_WIND_COEFF
        * wind_speed.clamp(min=0.0) ** DIFFUSIVITY_WIND_EXPONENT
        * wind_decay
    )
    eddy_diffusivity = DIFFUSIVITY_K0 * wind_mixing * stability_factor.pow(-DIFFUSIVITY_ALPHA)
    eddy_diffusivity = eddy_diffusivity.clamp(min=MIN_EDDY_DIFFUSIVITY)
    base_diffusivity = MOLECULAR_DIFFUSIVITY + eddy_diffusivity

    # If density decreases with depth, the water column is statically unstable.
    # Use a smooth boost so convective mixing emerges through K instead of a
    # hard seasonal correction or a discontinuous if-branch.
    unstable_gate = torch.sigmoid(
        (-density_gradient - DIFFUSIVITY_UNSTABLE_GRADIENT_EPS)
        / DIFFUSIVITY_UNSTABLE_GRADIENT_WIDTH
    )
    unstable_boost = 1.0 + (DIFFUSIVITY_UNSTABLE_BOOST - 1.0) * unstable_gate
    diffusivity = (base_diffusivity * unstable_boost).clamp(
        min=MIN_TOTAL_DIFFUSIVITY,
        max=MAX_TOTAL_DIFFUSIVITY,
    )
    return temp_pred, dT_dt_norm, dT_dz, density_gradient, richardson_number, diffusivity


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


def compute_surface_flux_terms(surface_temp, batch, shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION):
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
    sensible_heat = _use_provided_flux(batch, 'surface_sensible_heat', surface_temp)
    latent_heat = _use_provided_flux(batch, 'surface_latent_heat', surface_temp)
    if sensible_heat is None:
        sensible_heat = sensible_heat_bulk
    if latent_heat is None:
        latent_heat = latent_heat_bulk

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
    }
