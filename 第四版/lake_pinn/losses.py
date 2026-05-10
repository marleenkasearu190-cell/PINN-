# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .model import model_temperature
from .physics import compute_diffusivity, compute_surface_flux_terms

def _extended_forcing_kwargs(batch, prefix: str):
    return {
        'longwave': batch.get(f'{prefix}_longwave'),
        'latent_heat': batch.get(f'{prefix}_latent_heat'),
        'sensible_heat': batch.get(f'{prefix}_sensible_heat'),
        'secchi': batch.get(f'{prefix}_secchi'),
    }


def smooth_anneal(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def build_annealed_loss_weights(base_weights, progress: float):
    shape_weight_sum = (
        base_weights.get('time_continuity', 0.0)
        + base_weights.get('stratification', 0.0)
        + base_weights.get('smoothness', 0.0)
        + base_weights.get('deep_warming', 0.0)
        + base_weights.get('deep_anchor', 0.0)
        + base_weights.get('vertical_exchange', 0.0)
        + base_weights.get('convective_mixing', 0.0)
        + base_weights.get('surface_mixed_layer_uniformity', 0.0)
        + base_weights.get('abrupt_surface_cooling', 0.0)
        + base_weights.get('autumn_overturn', 0.0)
        + base_weights.get('heat_budget', 0.0)
    )
    if shape_weight_sum <= 0.0:
        return dict(base_weights)
    anneal = smooth_anneal(progress)
    weights = dict(base_weights)
    weights['obs'] = base_weights['obs'] * (1.6 - 0.85 * anneal)
    weights['time_continuity'] = base_weights.get('time_continuity', 0.0) * (0.35 + 0.65 * anneal)
    weights['stratification'] = base_weights.get('stratification', 0.0) * (0.1 + 0.9 * anneal)
    weights['smoothness'] = base_weights.get('smoothness', 0.0) * (0.2 + 0.8 * anneal)
    weights['deep_warming'] = base_weights.get('deep_warming', 0.0) * (0.25 + 0.75 * anneal)
    weights['deep_anchor'] = base_weights.get('deep_anchor', 0.0) * (0.05 + 0.55 * anneal)
    weights['vertical_exchange'] = base_weights.get('vertical_exchange', 0.0) * (0.20 + 0.80 * anneal)
    weights['convective_mixing'] = base_weights.get('convective_mixing', 0.0) * (0.15 + 0.85 * anneal)
    weights['surface_mixed_layer_uniformity'] = base_weights.get('surface_mixed_layer_uniformity', 0.0) * (0.20 + 0.80 * anneal)
    weights['abrupt_surface_cooling'] = base_weights.get('abrupt_surface_cooling', 0.0) * (0.30 + 0.70 * anneal)
    weights['autumn_overturn'] = base_weights.get('autumn_overturn', 0.0) * (0.25 + 0.75 * anneal)
    weights['heat_budget'] = base_weights.get('heat_budget', 0.0) * (0.35 + 0.65 * anneal)
    return weights


def compute_losses(
    model,
    batch,
    max_depth,
    time_scale_seconds,
    metadata,
    weights,
    shortwave_attenuation=SHORTWAVE_ATTENUATION,
    shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
    max_vertical_gradient_c_per_m=MAX_VERTICAL_GRADIENT_C_PER_M,
    entrainment_velocity_scale_m_per_day=MAX_ENTRAINMENT_VELOCITY_M_PER_DAY,
):
    shortwave_attenuation = float(np.clip(shortwave_attenuation, MIN_SHORTWAVE_ATTENUATION, MAX_SHORTWAVE_ATTENUATION))
    max_vertical_gradient_c_per_m = float(max(max_vertical_gradient_c_per_m, 0.1))
    t_col = batch['t_colloc'].clone().detach().requires_grad_(True)
    z_col = batch['z_colloc'].clone().detach().requires_grad_(True)

    temp_pred, dT_dt_norm, dT_dz, density_gradient, richardson_number, diffusivity = compute_diffusivity(
        model=model,
        t_col=t_col,
        z_col=z_col,
        max_depth=max_depth,
        wind_speed=batch['wind_colloc'],
        water_depth=max_depth,
        metadata=metadata,
        doy_sin=batch.get('doy_sin_colloc'),
        doy_cos=batch.get('doy_cos_colloc'),
        air_temp=batch.get('air_temp_colloc'),
        shortwave=batch.get('solar_flux_colloc'),
        lst_surface=batch.get('lst_surface_colloc'),
        **_extended_forcing_kwargs(batch, 'colloc'),
    )

    dT_dt_real = dT_dt_norm / time_scale_seconds
    conductive_flux = diffusivity * dT_dz
    dflux_dz = torch.autograd.grad(conductive_flux, z_col, grad_outputs=torch.ones_like(conductive_flux), create_graph=True)[0]

    penetrating_shortwave = (
        (1.0 - shortwave_surface_fraction)
        * batch['solar_flux_colloc']
        * torch.exp(-shortwave_attenuation * z_col)
    )
    dphi_dz = -shortwave_attenuation * penetrating_shortwave
    heating_term = -dphi_dz / RHO_CP

    pde_residual = dT_dt_real - dflux_dz - heating_term
    loss_pde = torch.mean(pde_residual ** 2)

    t_surface = batch['surface_time'].clone().detach().requires_grad_(True)
    z_surface = torch.zeros_like(t_surface, requires_grad=True)
    surface_temp = model_temperature(
        model,
        t_surface,
        z_surface,
        max_depth,
        metadata=metadata,
        doy_sin=batch.get('surface_doy_sin'),
        doy_cos=batch.get('surface_doy_cos'),
        air_temp=batch.get('surface_air_temp'),
        wind_speed=batch.get('surface_wind_speed'),
        shortwave=batch.get('surface_shortwave'),
        lst_surface=batch.get('surface_lst'),
        **_extended_forcing_kwargs(batch, 'surface'),
    )
    dT_dz_surface = torch.autograd.grad(
        surface_temp,
        z_surface,
        grad_outputs=torch.ones_like(surface_temp),
        create_graph=True,
    )[0]
    _, _, _, _, _, diffusivity_surface = compute_diffusivity(
        model=model,
        t_col=t_surface,
        z_col=z_surface,
        max_depth=max_depth,
        wind_speed=batch['surface_wind_speed'],
        water_depth=max_depth,
        metadata=metadata,
        doy_sin=batch.get('surface_doy_sin'),
        doy_cos=batch.get('surface_doy_cos'),
        air_temp=batch.get('surface_air_temp'),
        shortwave=batch.get('surface_shortwave'),
        lst_surface=batch.get('surface_lst'),
        **_extended_forcing_kwargs(batch, 'surface'),
    )
    surface_flux = compute_surface_flux_terms(
        surface_temp,
        batch,
        shortwave_surface_fraction=shortwave_surface_fraction,
    )
    seb_residual = diffusivity_surface * dT_dz_surface - surface_flux['seb_flux']
    ice_residual = surface_temp
    loss_surface_bc = torch.mean(
        (1.0 - surface_flux['ice_indicator']) * seb_residual ** 2
        + surface_flux['ice_indicator'] * ice_residual ** 2
    )

    t_bottom = batch['surface_time'].clone().detach().requires_grad_(True)
    z_bottom = torch.full_like(t_bottom, max_depth, requires_grad=True)
    bottom_temp = model_temperature(
        model,
        t_bottom,
        z_bottom,
        max_depth,
        metadata=metadata,
        doy_sin=batch.get('surface_doy_sin'),
        doy_cos=batch.get('surface_doy_cos'),
        air_temp=batch.get('surface_air_temp'),
        wind_speed=batch.get('surface_wind_speed'),
        shortwave=batch.get('surface_shortwave'),
        lst_surface=batch.get('surface_lst'),
        **_extended_forcing_kwargs(batch, 'surface'),
    )
    dT_dz_bottom = torch.autograd.grad(
        bottom_temp,
        z_bottom,
        grad_outputs=torch.ones_like(bottom_temp),
        create_graph=True,
    )[0]
    loss_bottom_flux = torch.mean(dT_dz_bottom ** 2)
    loss_bc = loss_surface_bc + loss_bottom_flux

    ic_temp_pred = model_temperature(
        model,
        batch['ic_time'],
        batch['ic_depth'],
        max_depth,
        metadata=metadata,
        doy_sin=batch.get('ic_doy_sin'),
        doy_cos=batch.get('ic_doy_cos'),
        air_temp=batch.get('ic_air_temp'),
        wind_speed=batch.get('ic_wind_speed'),
        shortwave=batch.get('ic_shortwave'),
        lst_surface=batch.get('ic_lst_surface'),
        **_extended_forcing_kwargs(batch, 'ic'),
    )
    loss_ic = torch.mean((ic_temp_pred - batch['ic_temperature']) ** 2)

    obs_temp_pred = model_temperature(
        model,
        batch['obs_time'],
        batch['obs_depth'],
        max_depth,
        metadata=metadata,
        doy_sin=batch.get('obs_doy_sin'),
        doy_cos=batch.get('obs_doy_cos'),
        air_temp=batch.get('obs_air_temp'),
        wind_speed=batch.get('obs_wind_speed'),
        shortwave=batch.get('obs_shortwave'),
        lst_surface=batch.get('obs_lst_surface'),
        **_extended_forcing_kwargs(batch, 'obs'),
    )
    obs_residual_sq = (obs_temp_pred - batch['obs_temperature']) ** 2
    loss_obs = torch.mean(batch['obs_weight'] * obs_residual_sq)

    loss_time_continuity = torch.zeros((), dtype=torch.float32, device=t_col.device)
    continuity_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        t_seq_now = batch['seq_time_now'].clone().detach().requires_grad_(True)
        z_seq = batch['seq_depth'].clone().detach().requires_grad_(True)
        temp_seq_now, _, dT_dz_seq, _, _, diffusivity_seq = compute_diffusivity(
            model=model,
            t_col=t_seq_now,
            z_col=z_seq,
            max_depth=max_depth,
            wind_speed=batch['seq_wind_now'],
            water_depth=max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_now'),
            doy_cos=batch.get('seq_doy_cos_now'),
            air_temp=batch.get('seq_air_temp_now'),
            shortwave=batch.get('seq_solar_flux_now'),
            lst_surface=batch.get('seq_lst_now'),
            **_extended_forcing_kwargs(batch, 'seq_now'),
        )
        conductive_flux_seq = diffusivity_seq * dT_dz_seq
        dflux_dz_seq = torch.autograd.grad(
            conductive_flux_seq,
            z_seq,
            grad_outputs=torch.ones_like(conductive_flux_seq),
            create_graph=True,
        )[0]
        penetrating_shortwave_seq = (
            (1.0 - shortwave_surface_fraction)
            * batch['seq_solar_flux_now']
            * torch.exp(-shortwave_attenuation * z_seq)
        )
        dphi_dz_seq = -shortwave_attenuation * penetrating_shortwave_seq
        heating_seq = -dphi_dz_seq / RHO_CP
        temp_seq_next = model_temperature(
            model,
            batch['seq_time_next'],
            z_seq,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_next'),
            doy_cos=batch.get('seq_doy_cos_next'),
            air_temp=batch.get('seq_air_temp_next'),
            wind_speed=batch.get('seq_wind_next'),
            shortwave=batch.get('seq_solar_flux_next'),
            lst_surface=batch.get('seq_lst_next'),
            **_extended_forcing_kwargs(batch, 'seq_next'),
        )
        continuity_residual = temp_seq_next - temp_seq_now - batch['seq_delta_seconds'] * (dflux_dz_seq + heating_seq)
        loss_time_continuity = torch.mean(continuity_residual ** 2)
        continuity_residual_rms = torch.sqrt(torch.mean(continuity_residual ** 2) + 1e-12)

    loss_vertical_exchange = torch.zeros((), dtype=torch.float32, device=t_col.device)
    vertical_exchange_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    surface_seq_now = None
    surface_seq_next = None
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        entrainment_velocity_cap = float(max(entrainment_velocity_scale_m_per_day, 0.05)) / SECONDS_PER_DAY
        mld_deepening = torch.relu(batch['seq_mld_next'] - batch['seq_mld_now'])
        entrainment_velocity = torch.clamp(
            mld_deepening / torch.clamp(batch['seq_delta_seconds'], min=1.0),
            min=0.0,
            max=entrainment_velocity_cap,
        )
        entrainment_center = batch['seq_mld_now'] + 0.35 * mld_deepening + 0.5
        entrainment_gate = torch.exp(-((z_seq - entrainment_center) / 1.25) ** 2)
        advection_tendency = -entrainment_velocity * dT_dz_seq * entrainment_gate
        vertical_exchange_residual = temp_seq_next - temp_seq_now - batch['seq_delta_seconds'] * (
            dflux_dz_seq + heating_seq + advection_tendency
        )
        loss_vertical_exchange = torch.mean((entrainment_gate * vertical_exchange_residual) ** 2)
        vertical_exchange_residual_rms = torch.sqrt(torch.mean((entrainment_gate * vertical_exchange_residual) ** 2) + 1e-12)
        surface_seq_now = model_temperature(
            model,
            batch['seq_time_now'],
            batch['seq_surface_depth'],
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_now'),
            doy_cos=batch.get('seq_doy_cos_now'),
            air_temp=batch.get('seq_air_temp_now'),
            wind_speed=batch.get('seq_wind_now'),
            shortwave=batch.get('seq_solar_flux_now'),
            lst_surface=batch.get('seq_lst_now'),
            **_extended_forcing_kwargs(batch, 'seq_now'),
        )
        surface_seq_next = model_temperature(
            model,
            batch['seq_time_next'],
            batch['seq_surface_depth'],
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_next'),
            doy_cos=batch.get('seq_doy_cos_next'),
            air_temp=batch.get('seq_air_temp_next'),
            wind_speed=batch.get('seq_wind_next'),
            shortwave=batch.get('seq_solar_flux_next'),
            lst_surface=batch.get('seq_lst_next'),
            **_extended_forcing_kwargs(batch, 'seq_next'),
        )

    loss_convective_mixing = torch.zeros((), dtype=torch.float32, device=t_col.device)
    convective_mixing_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if surface_seq_now is not None:
        cooling_indicator = torch.sigmoid((8.0 - batch['seq_air_temp_now']) / 2.0)
        low_solar_indicator = torch.sigmoid((120.0 - batch['seq_solar_flux_now']) / 35.0)
        deepening_indicator = torch.sigmoid((batch['seq_mld_next'] - batch['seq_mld_now'] - 0.05) / 0.15)
        convective_gate = cooling_indicator * low_solar_indicator * deepening_indicator
        within_mixed_layer = torch.sigmoid((batch['seq_mld_now'] + 0.75 - z_seq) / 0.5)
        convective_residual = convective_gate * within_mixed_layer * (temp_seq_now - surface_seq_now)
        loss_convective_mixing = torch.mean(convective_residual ** 2)
        convective_mixing_residual_rms = torch.sqrt(torch.mean(convective_residual ** 2) + 1e-12)

    loss_surface_mixed_layer_uniformity = torch.zeros((), dtype=torch.float32, device=t_col.device)
    surface_mixed_layer_uniformity_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if surface_seq_now is not None:
        mixed_layer_ref_depth = torch.clamp(0.35 * batch['seq_mld_now'], min=0.20, max=1.50)
        mixed_layer_ref_temp = model_temperature(
            model,
            batch['seq_time_now'],
            mixed_layer_ref_depth,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_now'),
            doy_cos=batch.get('seq_doy_cos_now'),
            air_temp=batch.get('seq_air_temp_now'),
            wind_speed=batch.get('seq_wind_now'),
            shortwave=batch.get('seq_solar_flux_now'),
            lst_surface=batch.get('seq_lst_now'),
            **_extended_forcing_kwargs(batch, 'seq_now'),
        )
        wind_mixing_indicator = torch.sigmoid((batch['seq_wind_now'] - 2.2) / 0.9)
        cooling_mixing_indicator = torch.sigmoid((surface_seq_now - surface_seq_next - 0.08) / 0.10)
        active_mld_gate = torch.sigmoid((batch['seq_mld_now'] - 0.75) / 0.20)
        within_surface_mixed_layer = torch.sigmoid((batch['seq_mld_now'] + 0.30 - z_seq) / 0.30)
        mixing_driver = torch.clamp(0.35 + 0.45 * wind_mixing_indicator + 0.35 * cooling_mixing_indicator, 0.0, 1.25)
        mixed_layer_uniformity_residual = (
            active_mld_gate
            * within_surface_mixed_layer
            * mixing_driver
            * (temp_seq_now - mixed_layer_ref_temp)
        )
        loss_surface_mixed_layer_uniformity = torch.mean(mixed_layer_uniformity_residual ** 2)
        surface_mixed_layer_uniformity_residual_rms = torch.sqrt(
            torch.mean(mixed_layer_uniformity_residual ** 2) + 1e-12
        )

    loss_abrupt_surface_cooling = torch.zeros((), dtype=torch.float32, device=t_col.device)
    abrupt_surface_cooling_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if surface_seq_now is not None:
        capped_surface_mld = torch.minimum(batch['seq_mld_now'], torch.full_like(batch['seq_mld_now'], 3.0))
        within_surface_mixed_layer = torch.sigmoid((capped_surface_mld + 0.25 - z_seq) / 0.30)
        warm_surface_indicator = torch.sigmoid((surface_seq_now - 5.0) / 1.5)
        low_solar_indicator = torch.sigmoid((140.0 - batch['seq_solar_flux_now']) / 35.0)
        cool_air_indicator = torch.sigmoid((12.0 - batch['seq_air_temp_now']) / 2.2)
        windy_indicator = torch.sigmoid((batch['seq_wind_now'] - 4.5) / 0.8)
        active_surface_gate = warm_surface_indicator * within_surface_mixed_layer
        allowable_surface_drop = (
            SURFACE_MIXED_LAYER_MAX_COOLING_C_PER_DAY
            + 0.35 * low_solar_indicator
            + 0.30 * cool_air_indicator
            + 0.30 * windy_indicator
        ) * (batch['seq_delta_seconds'] / SECONDS_PER_DAY)
        abrupt_surface_cooling_excess = torch.relu((temp_seq_now - temp_seq_next) - allowable_surface_drop)
        abrupt_surface_cooling_residual = active_surface_gate * abrupt_surface_cooling_excess
        loss_abrupt_surface_cooling = torch.mean(abrupt_surface_cooling_residual ** 2)
        abrupt_surface_cooling_residual_rms = torch.sqrt(
            torch.mean(abrupt_surface_cooling_residual ** 2) + 1e-12
        )

    loss_autumn_overturn = torch.zeros((), dtype=torch.float32, device=t_col.device)
    autumn_overturn_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if surface_seq_now is not None and batch.get('seq_doy_now') is not None:
        autumn_indicator = torch.sigmoid((batch['seq_doy_now'] - 255.0) / 8.0) * torch.sigmoid((335.0 - batch['seq_doy_now']) / 8.0)
        surface_cooling_indicator = torch.sigmoid((surface_seq_now - surface_seq_next - 0.10) / 0.10)
        mld_deepening_indicator = torch.sigmoid((batch['seq_mld_next'] - batch['seq_mld_now'] - 0.05) / 0.12)
        low_solar_indicator = torch.sigmoid((140.0 - batch['seq_solar_flux_now']) / 30.0)
        cool_air_indicator = torch.sigmoid((14.0 - batch['seq_air_temp_now']) / 2.0)
        overturn_gate = autumn_indicator * surface_cooling_indicator * mld_deepening_indicator * low_solar_indicator * cool_air_indicator

        overturn_deep_floor = torch.clamp(
            torch.maximum(batch['seq_mld_now'] + 1.5, torch.full_like(batch['seq_mld_now'], max_depth * 0.35)),
            min=2.5,
            max=max_depth - 0.5,
        )
        overturn_deep_depth = torch.clamp(overturn_deep_floor + 1.0, min=3.0, max=max_depth)
        deep_seq_now = model_temperature(
            model,
            batch['seq_time_now'],
            overturn_deep_depth,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_now'),
            doy_cos=batch.get('seq_doy_cos_now'),
            air_temp=batch.get('seq_air_temp_now'),
            wind_speed=batch.get('seq_wind_now'),
            shortwave=batch.get('seq_solar_flux_now'),
            lst_surface=batch.get('seq_lst_now'),
            **_extended_forcing_kwargs(batch, 'seq_now'),
        )
        deep_seq_next = model_temperature(
            model,
            batch['seq_time_next'],
            overturn_deep_depth,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('seq_doy_sin_next'),
            doy_cos=batch.get('seq_doy_cos_next'),
            air_temp=batch.get('seq_air_temp_next'),
            wind_speed=batch.get('seq_wind_next'),
            shortwave=batch.get('seq_solar_flux_next'),
            lst_surface=batch.get('seq_lst_next'),
            **_extended_forcing_kwargs(batch, 'seq_next'),
        )
        prev_gap = torch.relu(surface_seq_now - deep_seq_now)
        next_gap = torch.relu(surface_seq_next - deep_seq_next)
        gap_collapse = torch.relu(prev_gap - next_gap)
        insufficient_collapse = torch.relu(AUTUMN_OVERTURN_TARGET_GAP_COLLAPSE_C - gap_collapse)
        allowable_deep_warm = AUTUMN_OVERTURN_DEEP_WARM_ALLOWANCE_C_PER_DAY * (batch['seq_delta_seconds'] / SECONDS_PER_DAY)
        fake_overturn_warm = torch.relu((deep_seq_next - deep_seq_now) - allowable_deep_warm)
        overturn_residual = overturn_gate * (insufficient_collapse + 1.5 * fake_overturn_warm)
        loss_autumn_overturn = torch.mean(overturn_residual ** 2)
        autumn_overturn_residual_rms = torch.sqrt(torch.mean(overturn_residual ** 2) + 1e-12)

    loss_stratification = torch.zeros((), dtype=torch.float32, device=t_col.device)
    stratification_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('strat_time') is not None and batch['strat_time'].numel() > 0:
        strat_time = batch['strat_time']
        shallow_temp = model_temperature(
            model,
            strat_time,
            batch['strat_shallow_depth'],
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('strat_doy_sin'),
            doy_cos=batch.get('strat_doy_cos'),
            air_temp=batch.get('strat_air_temp'),
            wind_speed=batch.get('strat_wind_speed'),
            shortwave=batch.get('strat_shortwave'),
            lst_surface=batch.get('strat_lst_surface'),
            **_extended_forcing_kwargs(batch, 'strat'),
        )
        deep_temp = model_temperature(
            model,
            strat_time,
            batch['strat_deep_depth'],
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('strat_doy_sin'),
            doy_cos=batch.get('strat_doy_cos'),
            air_temp=batch.get('strat_air_temp'),
            wind_speed=batch.get('strat_wind_speed'),
            shortwave=batch.get('strat_shortwave'),
            lst_surface=batch.get('strat_lst_surface'),
            **_extended_forcing_kwargs(batch, 'strat'),
        )
        strat_violation = torch.relu(deep_temp - (shallow_temp - batch['strat_margin']))
        loss_stratification = torch.mean(batch['strat_weight'] * strat_violation ** 2)
        stratification_residual_rms = torch.sqrt(torch.mean(strat_violation ** 2) + 1e-12)

    smoothness_excess = torch.relu(torch.abs(dT_dz) - max_vertical_gradient_c_per_m)
    loss_smoothness = torch.mean(smoothness_excess ** 2)
    smoothness_residual_rms = torch.sqrt(torch.mean(smoothness_excess ** 2) + 1e-12)

    loss_deep_warming = torch.zeros((), dtype=torch.float32, device=t_col.device)
    deep_warming_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        deep_gate = torch.sigmoid((z_seq - batch['seq_deep_floor']) / 0.9)
        warm_gate = torch.sigmoid((batch['seq_air_temp_now'] - 8.0) / 2.0) * torch.sigmoid((batch['seq_solar_flux_now'] - 110.0) / 35.0)
        allowable_warming = DEEP_WARMING_ALLOWANCE_C_PER_DAY * (batch['seq_delta_seconds'] / SECONDS_PER_DAY)
        deep_warming_excess = torch.relu((temp_seq_next - temp_seq_now) - allowable_warming)
        deep_warming_residual = deep_gate * warm_gate * deep_warming_excess
        loss_deep_warming = torch.mean(deep_warming_residual ** 2)
        deep_warming_residual_rms = torch.sqrt(torch.mean(deep_warming_residual ** 2) + 1e-12)

    loss_deep_anchor = torch.zeros((), dtype=torch.float32, device=t_col.device)
    deep_anchor_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('deep_anchor_time') is not None and batch['deep_anchor_time'].numel() > 0:
        deep_temp = model_temperature(
            model,
            batch['deep_anchor_time'],
            batch['deep_anchor_depth'],
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('deep_anchor_doy_sin'),
            doy_cos=batch.get('deep_anchor_doy_cos'),
            air_temp=batch.get('deep_anchor_air_temp'),
            wind_speed=batch.get('deep_anchor_wind_speed'),
            shortwave=batch.get('deep_anchor_shortwave'),
            lst_surface=batch.get('deep_anchor_lst_surface'),
            **_extended_forcing_kwargs(batch, 'deep_anchor'),
        )
        deep_excess = torch.relu(deep_temp - batch['deep_anchor_target'])
        loss_deep_anchor = torch.mean(batch['deep_anchor_weight'] * deep_excess ** 2)
        deep_anchor_residual_rms = torch.sqrt(torch.mean(deep_excess ** 2) + 1e-12)

    loss_heat_budget = torch.zeros((), dtype=torch.float32, device=t_col.device)
    heat_budget_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('budget_time_now') is not None and batch['budget_time_now'].numel() > 0:
        t_budget_now = batch['budget_time_now']
        t_budget_next = batch['budget_time_next']
        z_budget = batch['budget_depth'].clone().detach().requires_grad_(True)
        temp_budget_now = model_temperature(
            model,
            t_budget_now,
            z_budget,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('budget_doy_sin_now'),
            doy_cos=batch.get('budget_doy_cos_now'),
            air_temp=batch.get('budget_air_temp_now_expanded'),
            wind_speed=batch.get('budget_wind_speed_now_expanded'),
            shortwave=batch.get('budget_solar_flux'),
            lst_surface=batch.get('budget_lst_surface_now_expanded'),
            **_extended_forcing_kwargs(batch, 'budget_now_expanded'),
        )
        temp_budget_next = model_temperature(
            model,
            t_budget_next,
            z_budget,
            max_depth,
            metadata=metadata,
            doy_sin=batch.get('budget_doy_sin_next'),
            doy_cos=batch.get('budget_doy_cos_next'),
            air_temp=batch.get('budget_air_temp_next_expanded'),
            wind_speed=batch.get('budget_wind_speed_next_expanded'),
            shortwave=batch.get('budget_solar_flux_next'),
            lst_surface=batch.get('budget_lst_surface_next_expanded'),
            **_extended_forcing_kwargs(batch, 'budget_next_expanded'),
        )
        dz_budget = batch['budget_dz']
        heat_content_rate = ((temp_budget_next - temp_budget_now) / torch.clamp(batch['budget_delta_seconds'], min=1.0)) * dz_budget
        budget_group_index = batch['budget_group_index'].reshape(-1, 1)
        integrated_heat_tendency = torch.zeros_like(batch['budget_surface_flux'])
        integrated_heat_tendency.scatter_add_(0, budget_group_index, heat_content_rate)

        penetrating_shortwave_budget = (
            (1.0 - shortwave_surface_fraction)
            * batch['budget_solar_flux']
            * torch.exp(-shortwave_attenuation * z_budget)
        )
        dphi_dz_budget = -shortwave_attenuation * penetrating_shortwave_budget
        internal_heating_rate = (-dphi_dz_budget / RHO_CP) * dz_budget
        integrated_internal_heating = torch.zeros_like(batch['budget_surface_flux'])
        integrated_internal_heating.scatter_add_(0, budget_group_index, internal_heating_rate)

        external_heat_tendency = batch['budget_surface_flux'] + integrated_internal_heating
        heat_budget_residual = integrated_heat_tendency - external_heat_tendency
        loss_heat_budget = torch.mean(heat_budget_residual ** 2)
        heat_budget_residual_rms = torch.sqrt(torch.mean(heat_budget_residual ** 2) + 1e-12)

    density_instability = torch.relu(-density_gradient)
    loss_density_reg = torch.mean(density_instability ** 2)

    loss_total = (
        weights['pde'] * loss_pde +
        weights['bc'] * loss_bc +
        weights['ic'] * loss_ic +
        weights['obs'] * loss_obs +
        weights.get('time_continuity', 0.0) * loss_time_continuity +
        weights.get('stratification', 0.0) * loss_stratification +
        weights.get('smoothness', 0.0) * loss_smoothness +
        weights.get('deep_warming', 0.0) * loss_deep_warming +
        weights.get('deep_anchor', 0.0) * loss_deep_anchor +
        weights.get('vertical_exchange', 0.0) * loss_vertical_exchange +
        weights.get('convective_mixing', 0.0) * loss_convective_mixing +
        weights.get('surface_mixed_layer_uniformity', 0.0) * loss_surface_mixed_layer_uniformity +
        weights.get('abrupt_surface_cooling', 0.0) * loss_abrupt_surface_cooling +
        weights.get('autumn_overturn', 0.0) * loss_autumn_overturn +
        weights.get('heat_budget', 0.0) * loss_heat_budget +
        weights['density_reg'] * loss_density_reg
    )

    return {
        'total': loss_total,
        'loss_pde': loss_pde,
        'loss_bc': loss_bc,
        'loss_surface_bc': loss_surface_bc,
        'loss_bottom_flux': loss_bottom_flux,
        'loss_ic': loss_ic,
        'loss_obs': loss_obs,
        'loss_time_continuity': loss_time_continuity,
        'loss_stratification': loss_stratification,
        'loss_smoothness': loss_smoothness,
        'loss_deep_warming': loss_deep_warming,
        'loss_deep_anchor': loss_deep_anchor,
        'loss_vertical_exchange': loss_vertical_exchange,
        'loss_convective_mixing': loss_convective_mixing,
        'loss_surface_mixed_layer_uniformity': loss_surface_mixed_layer_uniformity,
        'loss_abrupt_surface_cooling': loss_abrupt_surface_cooling,
        'loss_autumn_overturn': loss_autumn_overturn,
        'loss_heat_budget': loss_heat_budget,
        'loss_density_reg': loss_density_reg,
        'seb_residual_rms': torch.sqrt(torch.mean(seb_residual ** 2) + 1e-12),
        'pde_residual_rms': torch.sqrt(torch.mean(pde_residual ** 2) + 1e-12),
        'continuity_residual_rms': continuity_residual_rms,
        'stratification_residual_rms': stratification_residual_rms,
        'smoothness_residual_rms': smoothness_residual_rms,
        'deep_warming_residual_rms': deep_warming_residual_rms,
        'deep_anchor_residual_rms': deep_anchor_residual_rms,
        'vertical_exchange_residual_rms': vertical_exchange_residual_rms,
        'convective_mixing_residual_rms': convective_mixing_residual_rms,
        'surface_mixed_layer_uniformity_residual_rms': surface_mixed_layer_uniformity_residual_rms,
        'abrupt_surface_cooling_residual_rms': abrupt_surface_cooling_residual_rms,
        'autumn_overturn_residual_rms': autumn_overturn_residual_rms,
        'heat_budget_residual_rms': heat_budget_residual_rms,
        'ri_mean': richardson_number.mean().detach(),
        'kappa_mean': diffusivity.mean().detach(),
    }


def summarize_window_losses(window_losses):
    if not window_losses:
        raise ValueError('window_losses must not be empty.')

    keys = window_losses[0].keys()
    summary = {}
    for key in keys:
        values = [float(item[key]) for item in window_losses]
        summary[key] = float(np.mean(values))
    return summary


def summarize_profile_error_metrics(obs_depths, errors, max_depth):
    obs_depths = np.asarray(obs_depths, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return None

    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))

    shallow_limit = float(min(3.0, max_depth))
    thermocline_upper = float(min(8.0, max_depth))

    shallow_mask = obs_depths <= (shallow_limit + 1.0e-6)
    thermocline_mask = (obs_depths > (shallow_limit + 1.0e-6)) & (obs_depths <= (thermocline_upper + 1.0e-6))
    deep_mask = obs_depths > (thermocline_upper + 1.0e-6)

    def band_stats(mask):
        if not np.any(mask):
            return overall_rmse, overall_mae, overall_bias, 0
        band_errors = errors[mask]
        return (
            float(np.sqrt(np.mean(band_errors ** 2))),
            float(np.mean(np.abs(band_errors))),
            float(np.mean(band_errors)),
            int(band_errors.size),
        )

    surface_band_rmse, surface_band_mae, surface_band_bias, surface_band_count = band_stats(shallow_mask)
    thermocline_band_rmse, thermocline_band_mae, thermocline_band_bias, thermocline_band_count = band_stats(thermocline_mask)
    deep_band_rmse, deep_band_mae, deep_band_bias, deep_band_count = band_stats(deep_mask)

    profile_objective = (
        0.45 * overall_rmse
        + 0.25 * surface_band_rmse
        + 0.20 * thermocline_band_rmse
        + 0.10 * deep_band_rmse
        + 0.05 * abs(overall_bias)
    )

    return {
        'matched_rows': int(errors.size),
        'rmse': overall_rmse,
        'mae': overall_mae,
        'bias': overall_bias,
        'surface_band_rmse': float(surface_band_rmse),
        'surface_band_mae': float(surface_band_mae),
        'surface_band_bias': float(surface_band_bias),
        'surface_band_rows': int(surface_band_count),
        'thermocline_band_rmse': float(thermocline_band_rmse),
        'thermocline_band_mae': float(thermocline_band_mae),
        'thermocline_band_bias': float(thermocline_band_bias),
        'thermocline_band_rows': int(thermocline_band_count),
        'deep_band_rmse': float(deep_band_rmse),
        'deep_band_mae': float(deep_band_mae),
        'deep_band_bias': float(deep_band_bias),
        'deep_band_rows': int(deep_band_count),
        'objective': float(profile_objective),
    }
