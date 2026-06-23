import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.constants import (
    PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
    PINN_MAX_SECCHI_REFERENCE_M,
    PINN_MAX_TEMPERATURE_REFERENCE_C,
    RHO_CP,
)
from lake_pinn.hypsometry import fallback_area_profile
from lake_pinn.state_model import (
    FORCING_FEATURE_COLUMNS,
    FORCING_FEATURE_INDEX,
    STATIC_FEATURE_DIM,
    STATIC_FEATURE_KEYS,
    _average_forcing_rows_for_task,
    _coerce_freezing_storage,
    LakeStateForecaster,
    heat_conserving_convective_adjustment,
    ice_conductive_flux_wm2,
    remove_area_weighted_mean,
    resolve_hard_density_stability,
    static_feature_array,
)
from lake_pinn.state_reconstruction import (
    _forcing_tensor_rows,
    _huber_profile_loss,
    apply_lst_surface_assimilation,
    build_lst_profile_prior,
    initialize_rollout_state,
)
import lake_pinn.physics as physics_module
from lake_pinn.physics import compute_surface_flux_terms, water_density_torch
from lake_pinn.vertical_solver import (
    _implicit_diffusion_step_dense_reference,
    area_edges_from_nodes,
    implicit_diffusion_step,
    layer_thicknesses,
    one_day_heat_sources,
)


def _forcing_row():
    return {
        "features": torch.zeros(len(FORCING_FEATURE_COLUMNS), dtype=torch.float32),
        "air_temp": torch.tensor([12.0], dtype=torch.float32),
        "wind_speed": torch.tensor([3.0], dtype=torch.float32),
        "relative_humidity": torch.tensor([0.75], dtype=torch.float32),
        "surface_pressure": torch.tensor([101325.0], dtype=torch.float32),
        "shortwave": torch.tensor([150.0], dtype=torch.float32),
        "longwave": torch.tensor([320.0], dtype=torch.float32),
        "latent_heat": torch.tensor([20.0], dtype=torch.float32),
        "sensible_heat": torch.tensor([10.0], dtype=torch.float32),
        "ice_fraction": torch.tensor([0.0], dtype=torch.float32),
        "ice_mask": torch.tensor([0.0], dtype=torch.float32),
    }


def test_layer_thicknesses_are_positive():
    depths = torch.linspace(0.0, 10.0, 6)
    dz = layer_thicknesses(depths)
    assert dz.shape == depths.shape
    assert torch.all(dz > 0.0)
    assert torch.allclose(dz.sum(), torch.tensor(10.0), atol=1.0e-6)


def test_hypsometry_accepts_numpy_area_depth_arrays():
    depths = np.linspace(0.0, 10.0, 6)
    metadata = {
        "hypsometry_depth_m": np.array([0.0, 5.0, 10.0]),
        "hypsometry_area_fraction": np.array([1.0, 0.5, 0.1]),
    }
    area = fallback_area_profile(depths, metadata=metadata)
    assert area.shape == depths.shape
    assert np.isfinite(area).all()
    assert area[0] >= area[-1]


def test_implicit_diffusion_step_keeps_constant_profile_constant_without_sources():
    depths = torch.linspace(0.0, 10.0, 8)
    area = torch.ones_like(depths)
    kz = torch.full((1, len(depths)), 1.0e-5)
    profile = torch.full((1, len(depths)), 8.0)
    stepped = implicit_diffusion_step(profile, depths, area, kz)
    assert torch.allclose(stepped, profile, atol=1.0e-5)


def test_implicit_diffusion_conserves_area_weighted_heat_without_sources():
    depths = torch.linspace(0.0, 10.0, 8)
    area = torch.linspace(1.0, 0.25, 8)
    dz = layer_thicknesses(depths)
    kz = torch.full((1, len(depths)), 1.0e-5)
    profile = torch.linspace(20.0, 6.0, 8).unsqueeze(0)
    stepped = implicit_diffusion_step(profile, depths, area, kz)
    before = torch.sum(profile * area.reshape(1, -1) * dz.reshape(1, -1))
    after = torch.sum(stepped * area.reshape(1, -1) * dz.reshape(1, -1))
    assert torch.allclose(after, before, atol=1.0e-4)


def test_batched_tridiagonal_solver_matches_dense_reference():
    generator = torch.Generator().manual_seed(123)
    depths = torch.linspace(0.0, 24.0, 25, dtype=torch.float64)
    area = torch.linspace(1.0, 0.12, 25, dtype=torch.float64)
    temperature = 3.0 + 18.0 * torch.rand((7, 25), generator=generator, dtype=torch.float64)
    kz = 5.0e-6 + 8.0e-5 * torch.rand((7, 25), generator=generator, dtype=torch.float64)
    source = 2.0e-6 * torch.randn((7, 25), generator=generator, dtype=torch.float64)
    stepped = implicit_diffusion_step(temperature, depths, area, kz, source_c_per_s=source)
    reference = _implicit_diffusion_step_dense_reference(temperature, depths, area, kz, source_c_per_s=source)
    assert torch.allclose(stepped, reference, atol=1.0e-5, rtol=1.0e-5)


def test_batched_tridiagonal_solver_matches_dense_reference_uniform_area_no_source():
    generator = torch.Generator().manual_seed(456)
    depths = torch.linspace(0.0, 16.0, 17, dtype=torch.float32)
    area = torch.ones_like(depths)
    temperature = 4.0 + 12.0 * torch.rand((5, 17), generator=generator)
    kz = 1.0e-5 + 4.0e-5 * torch.rand((5, 17), generator=generator)
    stepped = implicit_diffusion_step(temperature, depths, area, kz)
    reference = _implicit_diffusion_step_dense_reference(temperature, depths, area, kz)
    assert torch.allclose(stepped, reference, atol=1.0e-5, rtol=1.0e-5)


def test_batched_tridiagonal_solver_expands_single_kz_and_source_rows():
    generator = torch.Generator().manual_seed(789)
    depths = torch.linspace(0.0, 12.0, 13)
    area = torch.linspace(1.0, 0.3, 13)
    temperature = 5.0 + 10.0 * torch.rand((4, 13), generator=generator)
    kz = torch.full((1, 13), 2.5e-5)
    source = torch.zeros((1, 13))
    stepped = implicit_diffusion_step(temperature, depths, area, kz, source_c_per_s=source)
    reference = _implicit_diffusion_step_dense_reference(temperature, depths, area, kz, source_c_per_s=source)
    assert torch.allclose(stepped, reference, atol=1.0e-5, rtol=1.0e-5)


def test_finite_volume_shortwave_source_matches_layer_absorption_energy():
    depths = torch.linspace(0.0, 10.0, 11)
    area = torch.ones_like(depths)
    dz = layer_thicknesses(depths)
    surface_flux = torch.tensor([100.0], dtype=torch.float32)
    shortwave = torch.tensor([200.0], dtype=torch.float32)
    kd = torch.tensor([0.4], dtype=torch.float32)
    source = one_day_heat_sources(
        depths,
        surface_flux_wm2=surface_flux,
        shortwave_wm2=shortwave,
        kd=kd,
        shortwave_surface_fraction=0.45,
        area_profile=area,
    )
    energy_wm2 = torch.sum(source * RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1), dim=1)
    expected = surface_flux + (1.0 - 0.45) * (1.0 - 0.06) * shortwave * (1.0 - torch.exp(-kd * depths[-1]))
    assert torch.allclose(energy_wm2, expected, rtol=1.0e-4, atol=1.0e-4)


def test_hypsometry_shortwave_absorption_uses_edge_areas():
    depths = torch.linspace(0.0, 10.0, 11)
    area = torch.linspace(1.0, 0.2, 11)
    dz = layer_thicknesses(depths)
    from lake_pinn.vertical_solver import layer_edges_from_nodes
    edges = layer_edges_from_nodes(depths)
    # Use the function directly instead of assuming uniform-area absorption.
    edge_area = area_edges_from_nodes(depths, area)
    source = one_day_heat_sources(
        depths,
        surface_flux_wm2=torch.tensor([0.0], dtype=torch.float32),
        shortwave_wm2=torch.tensor([200.0], dtype=torch.float32),
        kd=torch.tensor([0.4], dtype=torch.float32),
        shortwave_surface_fraction=0.45,
        area_profile=area,
    )
    energy_w = torch.sum(source * RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1), dim=1)
    q_pen = (1.0 - 0.45) * (1.0 - 0.06) * 200.0
    expected_power = torch.sum(
        edge_area[:-1] * q_pen * torch.exp(-0.4 * edges[:-1])
        - edge_area[1:] * q_pen * torch.exp(-0.4 * edges[1:])
    )
    assert torch.allclose(energy_w, expected_power.reshape(1), rtol=1.0e-4, atol=1.0e-4)


def test_ice_conductive_flux_is_negative_when_skin_is_colder_than_water():
    flux = ice_conductive_flux_wm2(
        torch.tensor([0.0]),
        torch.tensor([-10.0]),
        snow_depth_m=torch.tensor([0.0]),
        ice_thickness_m=torch.tensor([0.2]),
    )
    assert flux.item() < 0.0


def test_residual_area_weighted_mean_removal_preserves_heat_content():
    depths = torch.linspace(0.0, 10.0, 6)
    area = torch.linspace(1.0, 0.3, 6)
    dz = layer_thicknesses(depths)
    residual = torch.tensor([[1.0, 0.5, -0.2, -0.5, 0.1, 0.3]], dtype=torch.float32)
    adjusted = remove_area_weighted_mean(residual, depths, area)
    heat_delta_like = torch.sum(adjusted * area.reshape(1, -1) * dz.reshape(1, -1), dim=1)
    assert torch.allclose(heat_delta_like, torch.zeros_like(heat_delta_like), atol=1.0e-6)


def test_heat_conserving_convective_adjustment_stabilizes_and_preserves_heat():
    depths = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32)
    area = torch.tensor([1.0, 0.8, 0.5, 0.25], dtype=torch.float32)
    dz = layer_thicknesses(depths)
    profile = torch.tensor([[4.0, 9.0, 16.0, 22.0]], dtype=torch.float32)

    adjusted, diagnostics = heat_conserving_convective_adjustment(profile, depths, area)
    rho = water_density_torch(adjusted)
    heat_before = torch.sum(profile * area.reshape(1, -1) * dz.reshape(1, -1), dim=1)
    heat_after = torch.sum(adjusted * area.reshape(1, -1) * dz.reshape(1, -1), dim=1)

    assert diagnostics["density_adjustment_applied"].item() == 1.0
    assert diagnostics["density_adjustment_max_delta_c"].item() > 0.0
    assert torch.all(rho[:, :-1] - rho[:, 1:] <= 1.0e-4)
    assert torch.allclose(heat_after, heat_before, atol=1.0e-5)
    assert abs(diagnostics["density_adjustment_heat_delta_j_m2"].item()) < 10.0


def test_heat_conserving_convective_adjustment_leaves_stable_profiles_unchanged_in_batch():
    depths = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=torch.float32)
    area = torch.tensor([1.0, 0.9, 0.6, 0.3], dtype=torch.float32)
    profiles = torch.tensor(
        [
            [24.0, 18.0, 12.0, 8.0],
            [4.0, 9.0, 16.0, 22.0],
        ],
        dtype=torch.float32,
    )

    adjusted, diagnostics = heat_conserving_convective_adjustment(profiles, depths, area)

    assert adjusted.shape == profiles.shape
    assert torch.allclose(adjusted[0], profiles[0], atol=1.0e-6)
    assert diagnostics["density_adjustment_applied"][0].item() == 0.0
    assert diagnostics["density_adjustment_applied"][1].item() == 1.0


def test_hard_density_stability_auto_is_reconstruction_default_and_rejects_removed_modes():
    assert resolve_hard_density_stability("auto", task_mode="analysis", data_fill_mode="reconstruction")
    assert not resolve_hard_density_stability("off", task_mode="analysis", data_fill_mode="reconstruction")
    with pytest.raises(ValueError, match="forecast/nowcast modes were removed"):
        resolve_hard_density_stability("auto", task_mode="forecast_no_lst", data_fill_mode="forecast")


def test_state_forecaster_default_static_dim_matches_extended_metadata_features():
    depths = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    forcing_context_dim = next(
        module.out_features
        for module in reversed(model.forcing_encoder.proj)
        if isinstance(module, torch.nn.Linear)
    )

    assert model.param_net.input.in_features == 1 + forcing_context_dim + STATIC_FEATURE_DIM
    assert model.physical_scale_head.net[0].in_features == (
        forcing_context_dim + STATIC_FEATURE_DIM + len(FORCING_FEATURE_COLUMNS) + 2
    )
    assert model.lake_adaptive_head.net[0].in_features == STATIC_FEATURE_DIM


def test_latent_reservoir_stores_subzero_cold_content_without_heat_injection():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    area = torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float32)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="provided",
        freezing_energy_mode="latent_reservoir",
    )
    forcing = _forcing_row()
    forcing["shortwave"] = torch.tensor([0.0])
    forcing["longwave"] = torch.tensor([0.0])
    forcing["latent_heat"] = torch.tensor([900.0])
    forcing["sensible_heat"] = torch.tensor([900.0])

    stepped, storage, diagnostics = model.step(
        torch.full((1, len(depths)), 0.05),
        forcing,
        torch.zeros(STATIC_FEATURE_DIM),
        return_diagnostics=True,
        return_freezing_storage=True,
        diagnostic_mode="full",
    )

    assert torch.all(stepped >= 0.0)
    assert torch.sum(storage).item() > 0.0
    assert torch.allclose(storage[:, 1:], torch.zeros_like(storage[:, 1:]))
    assert diagnostics["freezing_storage_j_m2"].item() > 0.0
    assert diagnostics["freezing_storage_ice_j_m2"].item() == pytest.approx(
        diagnostics["freezing_storage_j_m2"].item(),
        abs=1.0e-6,
    )
    assert diagnostics["freezing_storage_surface_fraction"].item() == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["freezing_storage_deep_fraction"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert abs(diagnostics["temperature_floor_heat_injection_wm2"].item()) < 1.0e-6
    residual = diagnostics["heat_tendency_wm2"] - diagnostics["heat_input_wm2"]
    assert abs(residual.item()) < 1.0e-3


def test_latent_reservoir_melts_before_warming_water():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    area = torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float32)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="provided",
        freezing_energy_mode="latent_reservoir",
    )
    static = torch.zeros(STATIC_FEATURE_DIM)
    cold_forcing = _forcing_row()
    cold_forcing["shortwave"] = torch.tensor([0.0])
    cold_forcing["longwave"] = torch.tensor([0.0])
    cold_forcing["latent_heat"] = torch.tensor([900.0])
    cold_forcing["sensible_heat"] = torch.tensor([900.0])
    warm_forcing = _forcing_row()
    warm_forcing["shortwave"] = torch.tensor([700.0])
    warm_forcing["longwave"] = torch.tensor([360.0])
    warm_forcing["latent_heat"] = torch.tensor([0.0])
    warm_forcing["sensible_heat"] = torch.tensor([0.0])

    cold_profile, storage, _ = model.step(
        torch.full((1, len(depths)), 0.05),
        cold_forcing,
        static,
        return_diagnostics=True,
        return_freezing_storage=True,
    )
    storage_before = storage.sum().item()
    warm_profile, storage_after, diagnostics = model.step(
        cold_profile,
        warm_forcing,
        static,
        freezing_storage_j_m2=storage,
        return_diagnostics=True,
        return_freezing_storage=True,
    )

    assert storage_after.sum().item() < storage_before
    assert torch.allclose(storage_after[:, 1:], torch.zeros_like(storage_after[:, 1:]))
    assert torch.all(warm_profile >= 0.0)
    assert abs(diagnostics["temperature_floor_heat_injection_wm2"].item()) < 1.0e-6


def test_latent_reservoir_legacy_profile_storage_is_summarized_to_surface():
    temperature = torch.full((1, 4), 1.0, dtype=torch.float32)
    legacy_storage = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    storage = _coerce_freezing_storage(legacy_storage, temperature, surface_only=True)

    assert storage[0, 0].item() == pytest.approx(10.0)
    assert torch.allclose(storage[:, 1:], torch.zeros_like(storage[:, 1:]))


def test_freezing_clamp_mode_preserves_legacy_heat_injection():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="provided",
        freezing_energy_mode="clamp",
    )
    forcing = _forcing_row()
    forcing["shortwave"] = torch.tensor([0.0])
    forcing["longwave"] = torch.tensor([0.0])
    forcing["latent_heat"] = torch.tensor([900.0])
    forcing["sensible_heat"] = torch.tensor([900.0])

    stepped, storage, diagnostics = model.step(
        torch.full((1, len(depths)), 0.05),
        forcing,
        torch.zeros(STATIC_FEATURE_DIM),
        return_diagnostics=True,
        return_freezing_storage=True,
    )

    assert torch.all(stepped >= 0.0)
    assert torch.allclose(storage, torch.zeros_like(storage))
    assert diagnostics["temperature_floor_heat_injection_wm2"].item() > 0.0


def test_lake_adaptive_params_off_does_not_change_step_when_head_changes():
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        lake_adaptive_params="off",
        turbulent_flux_mode="bulk",
    )
    model.eval()
    forcing = _forcing_row()
    initial = torch.full((1, len(depths)), 8.0)
    static = torch.zeros(STATIC_FEATURE_DIM)
    first, first_diag = model.step(
        initial,
        forcing,
        static,
        return_diagnostics=True,
        diagnostic_mode="full",
    )
    with torch.no_grad():
        for parameter in model.lake_adaptive_head.parameters():
            parameter.add_(10.0)
    second, second_diag = model.step(
        initial,
        forcing,
        static,
        return_diagnostics=True,
        diagnostic_mode="full",
    )
    assert torch.allclose(first, second, atol=1.0e-6)
    assert torch.allclose(first_diag["adaptive_wind_kz_scale"], torch.tensor([1.0]))
    assert torch.allclose(second_diag["adaptive_turbulent_flux_blend_alpha"], torch.tensor([0.3]))
    assert torch.allclose(second_diag["adaptive_kd_multiplier"], torch.tensor([1.0]))
    assert torch.allclose(second_diag["adaptive_turbulent_exchange_scale"], torch.tensor([1.0]))
    assert torch.allclose(second_diag["adaptive_convective_mixing_scale"], torch.tensor([1.0]))
    assert torch.allclose(second_diag["adaptive_ice_shortwave_scale"], torch.tensor([1.0]))
    assert second_diag["adaptive_parameter_regularization_loss"].item() == 0.0


def test_lake_adaptive_kz_outputs_metadata_conditioned_bounded_values():
    depths = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        wind_kz_scale=1.5,
        lake_adaptive_params="kz",
        turbulent_flux_mode="bulk",
    )
    with torch.no_grad():
        first = model.lake_adaptive_head.net[0]
        middle = model.lake_adaptive_head.net[2]
        final = model.lake_adaptive_head.net[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        middle.weight.zero_()
        middle.bias.zero_()
        middle.weight[0, 0] = 1.0
        final.weight.zero_()
        final.weight[0, 0] = 5.0
    static = torch.zeros((2, STATIC_FEATURE_DIM))
    static[1, 0] = 1.0
    values, reg = model._adaptive_parameter_values(static)
    wind = values["wind_kz_scale"]
    alpha = values["blend_alpha"]
    assert torch.all((wind >= 0.4) & (wind <= 3.0))
    assert wind[0].item() != pytest.approx(wind[1].item())
    assert torch.allclose(alpha, torch.full_like(alpha, 0.3))
    assert torch.all(reg >= 0.0)


def test_lake_adaptive_flux_requires_blend_and_outputs_bounded_alpha():
    depths = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    with pytest.raises(ValueError, match="requires turbulent_flux_mode='blend'"):
        LakeStateForecaster(depths, area, lake_adaptive_params="flux", turbulent_flux_mode="bulk")
    with pytest.raises(ValueError, match="requires turbulent_flux_mode='bulk' or 'blend'"):
        LakeStateForecaster(depths, area, lake_adaptive_params="exchange", turbulent_flux_mode="provided")
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="blend",
        turbulent_flux_blend_alpha=0.2,
        lake_adaptive_params="flux",
    )
    with torch.no_grad():
        first = model.lake_adaptive_head.net[0]
        middle = model.lake_adaptive_head.net[2]
        final = model.lake_adaptive_head.net[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 2] = 1.0
        middle.weight.zero_()
        middle.bias.zero_()
        middle.weight[0, 0] = 1.0
        final.weight.zero_()
        final.weight[1, 0] = 5.0
    static = torch.zeros((2, STATIC_FEATURE_DIM))
    static[1, 2] = 1.0
    values, reg = model._adaptive_parameter_values(static)
    wind = values["wind_kz_scale"]
    alpha = values["blend_alpha"]
    assert torch.allclose(wind, torch.full_like(wind, 1.0))
    assert torch.all((alpha >= 0.0) & (alpha <= 0.6))
    assert alpha[0].item() != pytest.approx(alpha[1].item())
    assert torch.all(reg >= 0.0)


def test_lake_adaptive_all_outputs_are_bounded_and_diagnosed():
    depths = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="blend",
        turbulent_flux_blend_alpha=0.2,
        lake_adaptive_params="all",
    )
    static = torch.zeros((2, STATIC_FEATURE_DIM))
    static[1, 0] = 1.0
    values, reg = model._adaptive_parameter_values(static)
    assert set(values) == {
        "wind_kz_scale",
        "blend_alpha",
        "kd_multiplier",
        "turbulent_exchange_scale",
        "convective_mixing_scale",
        "ice_shortwave_scale",
    }
    assert torch.all((values["wind_kz_scale"] >= 0.4) & (values["wind_kz_scale"] <= 3.0))
    assert torch.all((values["blend_alpha"] >= 0.0) & (values["blend_alpha"] <= 0.6))
    assert torch.all((values["kd_multiplier"] >= 0.4) & (values["kd_multiplier"] <= 2.0))
    assert torch.all((values["turbulent_exchange_scale"] >= 0.5) & (values["turbulent_exchange_scale"] <= 1.8))
    assert torch.all((values["convective_mixing_scale"] >= 0.3) & (values["convective_mixing_scale"] <= 2.5))
    assert torch.all((values["ice_shortwave_scale"] >= 0.4) & (values["ice_shortwave_scale"] <= 1.8))
    assert torch.all(reg >= 0.0)

    _, diagnostics = model.step(
        torch.full((1, len(depths)), 4.0),
        _forcing_row(),
        torch.zeros(STATIC_FEATURE_DIM),
        return_diagnostics=True,
    )
    for key in (
        "adaptive_kd_multiplier",
        "adaptive_turbulent_exchange_scale",
        "adaptive_convective_mixing_scale",
        "adaptive_ice_shortwave_scale",
    ):
        assert key in diagnostics
        assert torch.isfinite(diagnostics[key]).all()


def test_lake_adaptive_head_depth_and_init_spread_create_metadata_variation():
    torch.manual_seed(7)
    depths = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    area = torch.ones_like(depths)
    model = LakeStateForecaster(
        depths,
        area,
        residual_limit_c=0.0,
        turbulent_flux_mode="blend",
        lake_adaptive_params="all",
        lake_adaptive_hidden_dim=64,
        lake_adaptive_init_spread=0.20,
    )
    assert model.lake_adaptive_hidden_dim == 64
    assert model.lake_adaptive_init_spread == pytest.approx(0.20)
    assert model.lake_adaptive_head.hidden_dim == 64
    assert model.lake_adaptive_head.init_spread == pytest.approx(0.20)
    assert len(model.lake_adaptive_head.net) == 5

    static = torch.zeros((2, STATIC_FEATURE_DIM))
    static[1, 0] = 1.0
    static[1, 7] = 1.0
    values, _ = model._adaptive_parameter_values(static)
    assert any(
        not torch.allclose(value[0], value[1], atol=1.0e-6, rtol=1.0e-6)
        for value in values.values()
    )


def test_state_forecaster_one_step_shape_and_range():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    profile = torch.linspace(14.0, 8.0, 10)
    static_features = torch.zeros(STATIC_FEATURE_DIM, dtype=torch.float32)
    prediction = model.step(profile, _forcing_row(), static_features)
    assert prediction.shape == (1, 10)
    assert torch.isfinite(prediction).all()
    assert prediction.min().item() >= 0.0
    assert prediction.max().item() <= 40.0


def test_turbulent_flux_modes_control_provided_flux_use():
    surface_temp = torch.tensor([20.0], dtype=torch.float32)
    batch = {
        "surface_air_temp": torch.tensor([10.0], dtype=torch.float32),
        "surface_wind_speed": torch.tensor([4.0], dtype=torch.float32),
        "surface_relative_humidity": torch.tensor([0.5], dtype=torch.float32),
        "surface_pressure": torch.tensor([101325.0], dtype=torch.float32),
        "surface_shortwave": torch.tensor([100.0], dtype=torch.float32),
        "surface_longwave": torch.tensor([320.0], dtype=torch.float32),
        "surface_sensible_heat": torch.tensor([500.0], dtype=torch.float32),
        "surface_latent_heat": torch.tensor([600.0], dtype=torch.float32),
        "surface_ice_fraction": torch.tensor([0.0], dtype=torch.float32),
    }

    bulk = compute_surface_flux_terms(surface_temp, batch, turbulent_flux_mode="bulk")
    provided = compute_surface_flux_terms(surface_temp, batch, turbulent_flux_mode="provided")
    blended = compute_surface_flux_terms(
        surface_temp,
        batch,
        turbulent_flux_mode="blend",
        turbulent_flux_blend_alpha=0.25,
    )

    assert not torch.allclose(bulk["sensible_heat"], torch.tensor([500.0]))
    assert torch.allclose(provided["sensible_heat"], torch.tensor([500.0]))
    assert torch.allclose(provided["latent_heat"], torch.tensor([600.0]))
    assert torch.allclose(
        blended["sensible_heat"],
        0.25 * provided["sensible_heat"] + 0.75 * bulk["sensible_heat"],
    )
    assert torch.allclose(
        blended["latent_heat"],
        0.25 * provided["latent_heat"] + 0.75 * bulk["latent_heat"],
    )


def test_turbulent_flux_bulk_skips_sensible_latent_provided_lookup(monkeypatch):
    surface_temp = torch.tensor([20.0], dtype=torch.float32)
    batch = {
        "surface_air_temp": torch.tensor([10.0], dtype=torch.float32),
        "surface_wind_speed": torch.tensor([4.0], dtype=torch.float32),
        "surface_relative_humidity": torch.tensor([0.5], dtype=torch.float32),
        "surface_pressure": torch.tensor([101325.0], dtype=torch.float32),
        "surface_shortwave": torch.tensor([100.0], dtype=torch.float32),
        "surface_longwave": torch.tensor([320.0], dtype=torch.float32),
        "surface_sensible_heat": torch.tensor([500.0], dtype=torch.float32),
        "surface_latent_heat": torch.tensor([600.0], dtype=torch.float32),
        "surface_ice_fraction": torch.tensor([0.0], dtype=torch.float32),
    }
    calls = []
    original = physics_module._use_provided_flux

    def spy_use_provided_flux(batch, key, reference):
        calls.append(key)
        return original(batch, key, reference)

    monkeypatch.setattr(physics_module, "_use_provided_flux", spy_use_provided_flux)

    compute_surface_flux_terms(surface_temp, batch, turbulent_flux_mode="bulk")
    assert calls == ["surface_longwave"]

    calls.clear()
    compute_surface_flux_terms(surface_temp, batch, turbulent_flux_mode="provided")
    assert calls == ["surface_sensible_heat", "surface_latent_heat", "surface_longwave"]


def test_learned_physical_scales_are_bounded_and_diagnostic():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    profile = torch.linspace(14.0, 8.0, 10)
    static_features = torch.zeros(STATIC_FEATURE_DIM, dtype=torch.float32)
    _, diagnostics = model.step(profile, _forcing_row(), static_features, return_diagnostics=True)
    shortwave_scale = diagnostics["shortwave_absorption_scale"]
    cooling_raw = diagnostics["surface_cooling_scale_raw"]
    cooling_effective = diagnostics["surface_cooling_scale"]
    assert torch.all(shortwave_scale >= 0.85)
    assert torch.all(shortwave_scale <= 1.30)
    assert torch.all(cooling_raw >= 0.90)
    assert torch.all(cooling_raw <= 1.40)
    if diagnostics["surface_flux_wm2"].item() >= 0.0:
        assert torch.allclose(cooling_effective, torch.ones_like(cooling_effective))
    else:
        assert torch.allclose(cooling_effective, cooling_raw)
    assert torch.all(diagnostics["surface_flux_bias_wm2"] >= -30.0)
    assert torch.all(diagnostics["surface_flux_bias_wm2"] <= 30.0)
    assert "open_water_sensible_heat_wm2" in diagnostics
    assert "open_water_latent_heat_wm2" in diagnostics
    assert "temperature_floor_heat_injection_wm2" in diagnostics
    assert diagnostics["temperature_floor_heat_injection_wm2"].item() >= 0.0


def test_single_lake_profile_loss_uses_depth_mask():
    prediction = torch.tensor([[10.0, 9.0, 8.0, 100.0]], dtype=torch.float32)
    target = torch.tensor([[10.0, 9.0, 8.0, 0.0]], dtype=torch.float32)
    masked_loss = _huber_profile_loss(
        prediction,
        target,
        delta=2.0,
        mask=[True, True, True, False],
    )
    unmasked_loss = _huber_profile_loss(prediction, target, delta=2.0)

    assert float(masked_loss) == 0.0
    assert float(unmasked_loss) > 0.0


def test_ice_shortwave_attenuation_reduces_water_shortwave():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    profile = torch.linspace(2.0, 3.0, 10)
    forcing = _forcing_row()
    forcing["ice_fraction"] = torch.tensor([1.0], dtype=torch.float32)
    forcing["ice_mask"] = torch.tensor([1.0], dtype=torch.float32)
    forcing["snow_depth"] = torch.tensor([0.1], dtype=torch.float32)
    forcing["ice_thickness"] = torch.tensor([0.3], dtype=torch.float32)
    forcing["ist_snow_ice"] = torch.tensor([-8.0], dtype=torch.float32)
    static_features = torch.zeros(STATIC_FEATURE_DIM, dtype=torch.float32)
    _, diagnostics = model.step(profile, forcing, static_features, return_diagnostics=True)
    assert diagnostics["ice_shortwave_transmission"].item() < 1.0
    assert diagnostics["shortwave_to_water_wm2"].item() < diagnostics["shortwave_wm2"].item()


def test_density_aware_kz_suppresses_stable_and_boosts_unstable_profiles():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    forcing_features = torch.zeros((1, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    forcing_history = torch.zeros((1, 3, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    static_features = torch.zeros((1, STATIC_FEATURE_DIM), dtype=torch.float32)
    wind = torch.tensor([3.0], dtype=torch.float32)

    stable_profile = torch.linspace(24.0, 8.0, 10).unsqueeze(0)
    neutral_profile = torch.full((1, 10), 10.0)
    unstable_profile = torch.linspace(4.0, 24.0, 10).unsqueeze(0)

    stable_kz, _, _ = model.predict_params(
        forcing_features,
        static_features,
        wind,
        temperature=stable_profile,
        forcing_history=forcing_history,
    )
    neutral_kz, _, _ = model.predict_params(
        forcing_features,
        static_features,
        wind,
        temperature=neutral_profile,
        forcing_history=forcing_history,
    )
    unstable_kz, _, _ = model.predict_params(
        forcing_features,
        static_features,
        wind,
        temperature=unstable_profile,
        forcing_history=forcing_history,
    )

    assert stable_kz.mean().item() < neutral_kz.mean().item()
    assert unstable_kz.mean().item() > stable_kz.mean().item()


def test_neural_kz_turbulent_term_is_gated_under_stable_stratification():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(
        depths,
        area,
        hidden_dim=16,
        stratification_mixing_cap="on",
        stratification_mixing_cap_strength=2.0,
    )
    forcing_features = torch.zeros((1, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    forcing_history = torch.zeros((1, 3, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    static_features = torch.zeros((1, STATIC_FEATURE_DIM), dtype=torch.float32)
    stable_profile = torch.linspace(25.0, 5.0, 10).unsqueeze(0)

    _, _, _, diagnostics = model.predict_params(
        forcing_features,
        static_features,
        torch.tensor([3.0], dtype=torch.float32),
        temperature=stable_profile,
        forcing_history=forcing_history,
        return_mixing_diagnostics=True,
    )

    assert diagnostics["background_nn_kz_mean"].item() <= 2.0e-6
    assert diagnostics["gated_turbulent_nn_kz_deep_mean"].item() < diagnostics[
        "turbulent_nn_kz_deep_mean"
    ].item()
    assert diagnostics["stratification_mixing_gate_deep_mean"].item() < 1.0


def test_kd_uses_light_extinction_base_and_secchi_fallback():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    with torch.no_grad():
        for parameter in model.param_net.parameters():
            parameter.zero_()
    forcing_features = torch.zeros((1, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    forcing_history = torch.zeros((1, 3, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    static_features = torch.zeros((1, STATIC_FEATURE_DIM), dtype=torch.float32)

    forcing_features[:, FORCING_FEATURE_INDEX["light_extinction_kd"]] = (
        0.6 / PINN_LIGHT_EXTINCTION_REFERENCE_M_INV
    )
    _, kd_from_light, _ = model.predict_params(
        forcing_features,
        static_features,
        torch.tensor([3.0], dtype=torch.float32),
        forcing_history=forcing_history,
    )
    assert kd_from_light.item() == pytest.approx(0.6, abs=1e-6)

    forcing_features[:, FORCING_FEATURE_INDEX["light_extinction_kd"]] = 0.0
    forcing_features[:, FORCING_FEATURE_INDEX["Secchi_m"]] = 4.0 / PINN_MAX_SECCHI_REFERENCE_M
    _, kd_from_secchi, _ = model.predict_params(
        forcing_features,
        static_features,
        torch.tensor([3.0], dtype=torch.float32),
        forcing_history=forcing_history,
    )
    assert kd_from_secchi.item() == pytest.approx(1.7 / 4.0, abs=1e-6)


def test_shape_aware_mixing_uses_lake_metadata_and_stratification_gate():
    depths = torch.linspace(0.0, 12.0, 10).numpy()
    area = torch.linspace(1.0, 0.2, 10).numpy()
    model = LakeStateForecaster(
        depths,
        area,
        hidden_dim=16,
        shape_aware_mixing="on",
        shape_mixing_strength=0.5,
        stratification_mixing_cap="on",
        stratification_mixing_cap_strength=1.0,
    )
    forcing = _forcing_row()
    stable_profile = torch.linspace(24.0, 8.0, 10)
    neutral_profile = torch.full((10,), 10.0)

    sheltered = torch.zeros(STATIC_FEATURE_DIM, dtype=torch.float32)
    exposed = torch.zeros(STATIC_FEATURE_DIM, dtype=torch.float32)
    for features in (sheltered, exposed):
        features[STATIC_FEATURE_KEYS.index("max_depth_norm")] = 0.4
        features[STATIC_FEATURE_KEYS.index("mean_depth_norm")] = 0.14
    sheltered[STATIC_FEATURE_KEYS.index("log_area")] = 0.12
    sheltered[STATIC_FEATURE_KEYS.index("fetch_norm")] = 0.005
    sheltered[STATIC_FEATURE_KEYS.index("wind_exposure_norm")] = 0.5
    sheltered[STATIC_FEATURE_KEYS.index("basin_shape_norm")] = 0.20
    sheltered[STATIC_FEATURE_KEYS.index("shoreline_development_norm")] = 1.6
    exposed[STATIC_FEATURE_KEYS.index("log_area")] = 0.65
    exposed[STATIC_FEATURE_KEYS.index("fetch_norm")] = 0.12
    exposed[STATIC_FEATURE_KEYS.index("wind_exposure_norm")] = 1.4
    exposed[STATIC_FEATURE_KEYS.index("basin_shape_norm")] = 0.60
    exposed[STATIC_FEATURE_KEYS.index("shoreline_development_norm")] = 0.8

    _, sheltered_diag = model.step(
        stable_profile,
        forcing,
        sheltered,
        return_diagnostics=True,
        diagnostic_mode="full",
    )
    _, exposed_diag = model.step(
        stable_profile,
        forcing,
        exposed,
        return_diagnostics=True,
        diagnostic_mode="full",
    )
    _, neutral_diag = model.step(
        neutral_profile,
        forcing,
        exposed,
        return_diagnostics=True,
        diagnostic_mode="full",
    )

    assert exposed_diag["lake_shape_wind_factor"].item() > sheltered_diag["lake_shape_wind_factor"].item()
    assert exposed_diag["lake_shape_decay_depth_m"].item() > sheltered_diag["lake_shape_decay_depth_m"].item()
    assert exposed_diag["stratification_mixing_gate_deep_mean"].item() < neutral_diag[
        "stratification_mixing_gate_deep_mean"
    ].item()
    assert exposed_diag["stratification_mixing_gate_min"].item() <= exposed_diag[
        "stratification_mixing_gate_mean"
    ].item()


def _sample_forcing_frame_for_lsts():
    import pandas as pd

    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "doy_sin": [0.0, 0.1, 0.2, 0.3],
            "doy_cos": [1.0, 0.9, 0.8, 0.7],
            "T_air_C": [1.0, 2.0, 3.0, 4.0],
            "wind_speed_m_per_s": [2.0, 2.0, 2.0, 2.0],
            "Solar_W_m2": [10.0, 20.0, 30.0, 40.0],
            "LST_surface_C": [11.0, 12.0, 13.0, 14.0],
            "LST_quality_factor": [0.9, 0.8, 0.7, 0.6],
            "LST_is_filled": [0.0, 1.0, 0.0, 1.0],
            "Longwave_W_m2": [300.0, 300.0, 300.0, 300.0],
            "latent_heat_upward_W_m2": [0.0, 0.0, 0.0, 0.0],
            "sensible_heat_upward_W_m2": [0.0, 0.0, 0.0, 0.0],
            "relative_humidity": [0.75, 0.75, 0.75, 0.75],
            "surface_pressure_Pa": [101325.0, 101325.0, 101325.0, 101325.0],
            "Secchi_m": [2.0, 2.0, 2.0, 2.0],
            "light_extinction_kd": [0.5, 0.5, 0.5, 0.5],
            "effective_fetch": [100.0, 100.0, 100.0, 100.0],
            "air_temp_mean_7d": [1.0, 1.5, 2.0, 2.5],
            "air_temp_mean_30d": [1.0, 1.5, 2.0, 2.5],
            "shortwave_sum_7d": [10.0, 30.0, 60.0, 100.0],
            "shortwave_sum_30d": [10.0, 30.0, 60.0, 100.0],
            "wind_mean_7d": [2.0, 2.0, 2.0, 2.0],
            "lst_mean_7d": [11.0, 11.5, 12.0, 12.5],
            "heating_degree_days_30d": [0.0, 0.0, 0.0, 0.0],
            "ice_fraction": [0.0, 0.0, 0.0, 0.0],
            "water_level_anomaly": [0.0, 0.0, 0.0, 0.0],
            "net_inflow": [0.0, 0.0, 0.0, 0.0],
            "LSWT_open_water_C": [11.0, 12.0, 13.0, 14.0],
            "IST_snow_ice_C": [np.nan, np.nan, np.nan, np.nan],
        }
    )


def test_forcing_history_encoder_uses_past_window_only_and_keeps_lst_for_reconstruction():
    frame = _sample_forcing_frame_for_lsts()
    rows = _forcing_tensor_rows(frame, history_window_days=2, task_mode="analysis")

    assert torch.allclose(
        rows[2]["features"][FORCING_FEATURE_INDEX["LST_surface_C"]],
        torch.tensor(frame.loc[2, "LST_surface_C"] / PINN_MAX_TEMPERATURE_REFERENCE_C, dtype=torch.float32),
    )
    assert torch.allclose(
        rows[2]["features"][FORCING_FEATURE_INDEX["LST_quality_factor"]],
        torch.tensor(0.7, dtype=torch.float32),
    )
    assert torch.allclose(
        rows[2]["features"][FORCING_FEATURE_INDEX["LST_is_filled"]],
        torch.tensor(0.0, dtype=torch.float32),
    )
    assert torch.allclose(
        rows[2]["features"][FORCING_FEATURE_INDEX["LST_observed_flag"]],
        torch.tensor(1.0, dtype=torch.float32),
    )
    assert torch.allclose(rows[2]["lst_quality"], torch.tensor([0.7], dtype=torch.float32))
    assert torch.allclose(rows[2]["lst_is_filled"], torch.tensor([0.0], dtype=torch.float32))
    assert torch.allclose(rows[2]["lst_observed_flag"], torch.tensor([1.0], dtype=torch.float32))
    # The row-2 history contains rows 1 and 2 only, not future row 3.
    row1_air_norm = frame.loc[1, "T_air_C"] / 30.0
    row2_air_norm = frame.loc[2, "T_air_C"] / 30.0
    row3_air_norm = frame.loc[3, "T_air_C"] / 30.0
    history_air = rows[2]["history_features"][:, 2]
    assert torch.allclose(history_air, torch.tensor([row1_air_norm, row2_air_norm], dtype=torch.float32))
    assert not torch.any(torch.isclose(history_air, torch.tensor(row3_air_norm, dtype=torch.float32)))


def test_lst_feature_dropout_masks_current_and_history_only_in_train_mode():
    depths = torch.linspace(0.0, 6.0, 4).numpy()
    area = torch.ones(4).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16, lst_feature_dropout_probability=1.0)
    features = torch.ones((2, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    history = torch.ones((2, 3, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    features[:, FORCING_FEATURE_INDEX["LST_is_filled"]] = 0.0
    history[:, :, FORCING_FEATURE_INDEX["LST_is_filled"]] = 0.0

    model.train()
    dropped_features, dropped_history, mask = model._apply_lst_feature_dropout(features, history)

    for name in ("LST_surface_C", "lst_mean_7d", "LST_quality_factor", "LST_observed_flag"):
        idx = FORCING_FEATURE_INDEX[name]
        assert torch.allclose(dropped_features[:, idx], torch.zeros(2))
        assert torch.allclose(dropped_history[:, :, idx], torch.zeros((2, 3)))
    filled_idx = FORCING_FEATURE_INDEX["LST_is_filled"]
    assert torch.allclose(dropped_features[:, filled_idx], torch.ones(2))
    assert torch.allclose(dropped_history[:, :, filled_idx], torch.ones((2, 3)))
    assert torch.allclose(mask, torch.ones(2))

    model.eval()
    eval_features, eval_history, eval_mask = model._apply_lst_feature_dropout(features, history)
    assert torch.allclose(eval_features, features)
    assert torch.allclose(eval_history, history)
    assert torch.allclose(eval_mask, torch.zeros(2))


def test_lst_feature_dropout_probability_zero_returns_original_tensors_in_train_mode():
    depths = torch.linspace(0.0, 6.0, 4).numpy()
    area = torch.ones(4).numpy()
    model = LakeStateForecaster(depths, area, hidden_dim=16, lst_feature_dropout_probability=0.0)
    features = torch.ones((2, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)
    history = torch.ones((2, 3, len(FORCING_FEATURE_COLUMNS)), dtype=torch.float32)

    model.train()
    dropped_features, dropped_history, mask = model._apply_lst_feature_dropout(features, history)

    assert dropped_features is features
    assert dropped_history is history
    assert torch.allclose(mask, torch.zeros(2))


def test_lst_feature_dropout_probability_rejects_invalid_values():
    depths = torch.linspace(0.0, 6.0, 4).numpy()
    area = torch.ones(4).numpy()

    with pytest.raises(ValueError, match="lst_feature_dropout_probability"):
        LakeStateForecaster(depths, area, hidden_dim=16, lst_feature_dropout_probability=-0.1)
    with pytest.raises(ValueError, match="lst_feature_dropout_probability"):
        LakeStateForecaster(depths, area, hidden_dim=16, lst_feature_dropout_probability=1.1)


def test_reservoir_simple_advective_heat_source_requires_reservoir_and_positive_inflow():
    depths = torch.linspace(0.0, 20.0, 11)
    metadata = {
        "max_depth_m": 20.0,
        "mean_depth_m": 8.0,
        "area_km2": 2.0,
        "volume_km3": 0.016,
        "reservoir_indicator": 1.0,
    }
    area = fallback_area_profile(depths.numpy(), metadata=metadata)
    model = LakeStateForecaster(depths, area, hidden_dim=16, advective_heat_source_mode="reservoir_simple")
    model.eval()
    forcing = _forcing_row()
    forcing["features"] = forcing["features"].clone()
    forcing["features"][FORCING_FEATURE_INDEX["net_inflow"]] = 1.0
    forcing["features"][FORCING_FEATURE_INDEX["air_temp_mean_7d"]] = 20.0 / PINN_MAX_TEMPERATURE_REFERENCE_C
    forcing["air_temp"] = torch.tensor([20.0], dtype=torch.float32)
    profile = torch.full((1, len(depths)), 6.0)
    static_features = torch.tensor(static_feature_array(metadata, 20.0), dtype=torch.float32)

    _, diagnostics = model.step(
        profile,
        forcing,
        static_features,
        return_diagnostics=True,
        diagnostic_mode="loss",
    )
    assert float(diagnostics["advective_heat_source_active_mean"]) == pytest.approx(1.0)
    assert float(diagnostics["advective_exchange_fraction_per_day"]) > 0.0
    assert float(diagnostics["advective_heat_source_c_per_day_mean"]) > 0.0

    non_reservoir = dict(metadata, reservoir_indicator=0.0)
    _, lake_diag = model.step(
        profile,
        forcing,
        torch.tensor(static_feature_array(non_reservoir, 20.0), dtype=torch.float32),
        return_diagnostics=True,
        diagnostic_mode="loss",
    )
    assert float(lake_diag["advective_heat_source_active_mean"]) == 0.0
    assert float(lake_diag["advective_exchange_fraction_per_day"]) == 0.0

    negative_forcing = dict(forcing)
    negative_forcing["features"] = forcing["features"].clone()
    negative_forcing["features"][FORCING_FEATURE_INDEX["net_inflow"]] = -1.0
    _, negative_diag = model.step(
        profile,
        negative_forcing,
        static_features,
        return_diagnostics=True,
        diagnostic_mode="loss",
    )
    assert float(negative_diag["advective_heat_source_active_mean"]) == 0.0
    assert float(negative_diag["advective_exchange_fraction_per_day"]) == 0.0


def test_average_forcing_rows_keeps_single_sided_observation_values():
    current = {
        "features": torch.tensor([0.0, 2.0], dtype=torch.float32),
        "lswt_open_water": torch.tensor([float("nan")], dtype=torch.float32),
        "ist_snow_ice": torch.tensor([-5.0], dtype=torch.float32),
        "lst_quality": torch.tensor([float("nan")], dtype=torch.float32),
        "ice_fraction": torch.tensor([0.2], dtype=torch.float32),
    }
    next_row = {
        "features": torch.tensor([2.0, 4.0], dtype=torch.float32),
        "lswt_open_water": torch.tensor([12.0], dtype=torch.float32),
        "ist_snow_ice": torch.tensor([float("nan")], dtype=torch.float32),
        "lst_quality": torch.tensor([0.8], dtype=torch.float32),
        "ice_fraction": torch.tensor([0.6], dtype=torch.float32),
    }

    averaged = _average_forcing_rows_for_task(current, next_row, task_mode="analysis")

    assert torch.allclose(averaged["features"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(averaged["lswt_open_water"], torch.tensor([12.0]))
    assert torch.allclose(averaged["ist_snow_ice"], torch.tensor([-5.0]))
    assert torch.allclose(averaged["lst_quality"], torch.tensor([0.8]))
    assert torch.allclose(averaged["ice_fraction"], torch.tensor([0.4]))


def test_removed_task_modes_are_rejected_by_forcing_rows():
    frame = _sample_forcing_frame_for_lsts()
    with pytest.raises(ValueError, match="forecast/nowcast modes were removed"):
        _forcing_tensor_rows(frame, history_window_days=2, task_mode="forecast_with_lst")



def test_lst_assimilation_is_surface_weighted_and_bounded():
    depths = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float32)
    profile = torch.full((1, 4), 10.0, dtype=torch.float32)
    nudged = apply_lst_surface_assimilation(
        profile,
        lst_surface=torch.tensor([20.0]),
        lst_quality=torch.tensor([1.0]),
        depths=depths,
        strength=0.2,
        decay_depth_m=2.0,
        max_increment_c=1.5,
    )
    increments = nudged - profile
    assert increments[0, 0].item() == 1.5
    assert 0.0 < increments[0, 1].item() < increments[0, 0].item()
    assert increments[0, -1].item() < 0.02


def test_lst_assimilation_is_gated_off_for_ice_surface():
    depths = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float32)
    profile = torch.full((1, 4), 2.0, dtype=torch.float32)
    nudged = apply_lst_surface_assimilation(
        profile,
        lst_surface=torch.tensor([-10.0]),
        lst_quality=torch.tensor([1.0]),
        depths=depths,
        strength=0.5,
        ice_mask=torch.tensor([1.0]),
    )
    assert torch.allclose(nudged, profile)


def _summer_forcing_frame():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-07-01", periods=5, freq="D"),
            "full_doy": [183, 184, 185, 186, 187],
            "doy_sin": [0.0] * 5,
            "doy_cos": [-1.0] * 5,
            "T_air_C": [22.0] * 5,
            "wind_speed_m_per_s": [2.0] * 5,
            "Solar_W_m2": [250.0] * 5,
            "LST_surface_C": [24.0] * 5,
            "LST_quality_factor": [1.0] * 5,
            "Longwave_W_m2": [350.0] * 5,
            "latent_heat_upward_W_m2": [40.0] * 5,
            "sensible_heat_upward_W_m2": [10.0] * 5,
            "relative_humidity": [0.75] * 5,
            "surface_pressure_Pa": [101325.0] * 5,
            "Secchi_m": [3.0] * 5,
            "light_extinction_kd": [0.55] * 5,
            "effective_fetch": [1000.0] * 5,
            "air_temp_mean_7d": [22.0] * 5,
            "air_temp_mean_30d": [20.0] * 5,
            "shortwave_sum_7d": [1500.0] * 5,
            "shortwave_sum_30d": [6000.0] * 5,
            "wind_mean_7d": [2.0] * 5,
            "lst_mean_7d": [24.0] * 5,
            "heating_degree_days_30d": [120.0] * 5,
            "ice_fraction": [0.0] * 5,
            "ice_mask": [0.0] * 5,
            "water_level_anomaly": [0.0] * 5,
            "net_inflow": [0.0] * 5,
            "LSWT_open_water_C": [24.0] * 5,
            "IST_snow_ice_C": [np.nan] * 5,
            "snow_depth_m": [0.0] * 5,
            "ice_thickness_m": [0.0] * 5,
        }
    )


def test_lst_profile_prior_is_not_uniform_lst_for_summer_deep_lake():
    depths = np.linspace(0.0, 20.0, 11).astype(np.float32)
    metadata = {"latitude": 43.0, "max_depth_m": 20.0, "mean_depth_m": 10.0}
    prior, info = build_lst_profile_prior(_summer_forcing_frame(), depths, metadata, start_idx=0)
    assert prior.shape == depths.shape
    assert np.isfinite(prior).all()
    assert prior[0] > prior[-1] + 3.0
    assert info["prior_surface_temp_c"] > info["prior_deep_temp_c"]


def test_prior_spinup_initialization_uses_physical_prior_not_uniform_lst():
    depths = np.linspace(0.0, 20.0, 11).astype(np.float32)
    area = np.linspace(1.0, 0.3, 11).astype(np.float32)
    frame = _summer_forcing_frame()
    metadata = {"latitude": 43.0, "max_depth_m": 20.0, "mean_depth_m": 10.0}
    model = LakeStateForecaster(depths, area, hidden_dim=16)
    forcing_rows = _forcing_tensor_rows(frame, history_window_days=2, task_mode="analysis")
    init = initialize_rollout_state(
        model=model,
        df=frame,
        depths=depths,
        all_lookup={},
        forcing_rows=forcing_rows,
        static_features=torch.tensor(static_feature_array(metadata, 20.0), dtype=torch.float32),
        metadata=metadata,
        device=torch.device("cpu"),
        init_mode="prior_spinup",
        spinup_days=2,
        task_mode="analysis",
        area_profile=torch.tensor(area, dtype=torch.float32),
    )
    assert init["init_mode"] == "prior_spinup"
    assert init["spinup_days_used"] == 2
    assert init["rollout_start_idx"] == 2
    assert init["initial_profile"][0] > init["initial_profile"][-1] + 3.0
    assert len(init["diagnostics"]) == 2


if __name__ == "__main__":
    test_layer_thicknesses_are_positive()
    test_hypsometry_accepts_numpy_area_depth_arrays()
    test_implicit_diffusion_step_keeps_constant_profile_constant_without_sources()
    test_implicit_diffusion_conserves_area_weighted_heat_without_sources()
    test_finite_volume_shortwave_source_matches_layer_absorption_energy()
    test_hypsometry_shortwave_absorption_uses_edge_areas()
    test_ice_conductive_flux_is_negative_when_skin_is_colder_than_water()
    test_residual_area_weighted_mean_removal_preserves_heat_content()
    test_heat_conserving_convective_adjustment_stabilizes_and_preserves_heat()
    test_heat_conserving_convective_adjustment_leaves_stable_profiles_unchanged_in_batch()
    test_hard_density_stability_auto_is_reconstruction_default_and_rejects_removed_modes()
    test_state_forecaster_one_step_shape_and_range()
    test_learned_physical_scales_are_bounded_and_diagnostic()
    test_single_lake_profile_loss_uses_depth_mask()
    test_ice_shortwave_attenuation_reduces_water_shortwave()
    test_density_aware_kz_suppresses_stable_and_boosts_unstable_profiles()
    test_forcing_history_encoder_uses_past_window_only_and_keeps_lst_for_reconstruction()
    test_removed_task_modes_are_rejected_by_forcing_rows()
    test_lst_assimilation_is_surface_weighted_and_bounded()
    test_lst_assimilation_is_gated_off_for_ice_surface()
    test_lst_profile_prior_is_not_uniform_lst_for_summer_deep_lake()
    test_prior_spinup_initialization_uses_physical_prior_not_uniform_lst()
    print("state forecaster solver sanity checks passed")
