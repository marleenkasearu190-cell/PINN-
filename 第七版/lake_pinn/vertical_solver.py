"""Differentiable one-dimensional lake temperature update solver.

The solver advances a temperature profile by one time step using an
area-weighted implicit diffusion update plus explicit heat-source terms.
Depth is positive downward and all fluxes are positive into the lake.
"""

from __future__ import annotations

import torch

from .constants import (
    MAX_TOTAL_DIFFUSIVITY,
    MIN_TOTAL_DIFFUSIVITY,
    RHO_CP,
    SECONDS_PER_DAY,
    SHORTWAVE_SURFACE_FRACTION,
    SURFACE_ALBEDO_WATER,
)


def layer_edges_from_nodes(depths: torch.Tensor) -> torch.Tensor:
    """Return finite-volume layer edges for a node grid from surface to bottom.

    The reconstruction-state depth grid is built as nodes that include 0 m and the
    maximum depth.  The physical water column should therefore end exactly at
    those two nodes rather than extending half a cell above/below them.
    """
    depths = depths.flatten()
    if depths.numel() == 1:
        return torch.stack([depths[0], depths[0] + torch.ones_like(depths[0])])
    edges = torch.empty(depths.numel() + 1, dtype=depths.dtype, device=depths.device)
    edges[1:-1] = 0.5 * (depths[:-1] + depths[1:])
    edges[0] = torch.zeros_like(depths[0])
    edges[-1] = depths[-1]
    return edges


def layer_thicknesses(depths: torch.Tensor) -> torch.Tensor:
    depths = depths.flatten()
    if depths.numel() == 1:
        return torch.ones_like(depths)
    edges = layer_edges_from_nodes(depths)
    return torch.clamp(edges[1:] - edges[:-1], min=1.0e-4)


def area_edges_from_nodes(depths: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Interpolate layer-edge areas from node areas for conservative sources."""
    depths = depths.flatten()
    area = torch.clamp(area.flatten().to(device=depths.device, dtype=depths.dtype), min=1.0e-6)
    if depths.numel() == 1:
        return torch.stack([area[0], area[0]])
    edge_area = torch.empty(depths.numel() + 1, dtype=depths.dtype, device=depths.device)
    edge_area[0] = area[0]
    edge_area[-1] = area[-1]
    edge_area[1:-1] = 0.5 * (area[:-1] + area[1:])
    return torch.clamp(edge_area, min=1.0e-6)


def build_area_weighted_diffusion_matrix(
    depths: torch.Tensor,
    area: torch.Tensor,
    kz_profile: torch.Tensor,
) -> torch.Tensor:
    """Build L where dT/dt = L @ T for vertical diffusion."""
    depths = depths.flatten()
    area = torch.clamp(area.flatten(), min=1.0e-4)
    kz_profile = torch.clamp(kz_profile.flatten(), min=MIN_TOTAL_DIFFUSIVITY, max=MAX_TOTAL_DIFFUSIVITY)
    n_depths = int(depths.numel())
    matrix = torch.zeros((n_depths, n_depths), dtype=depths.dtype, device=depths.device)
    if n_depths <= 1:
        return matrix

    dz_cell = layer_thicknesses(depths)
    dz_interface = torch.clamp(depths[1:] - depths[:-1], min=1.0e-4)
    area_interface = 0.5 * (area[:-1] + area[1:])
    kz_interface = 0.5 * (kz_profile[:-1] + kz_profile[1:])
    conductance = area_interface * kz_interface / dz_interface

    for idx in range(n_depths - 1):
        upper_scale = conductance[idx] / (area[idx] * dz_cell[idx])
        lower_scale = conductance[idx] / (area[idx + 1] * dz_cell[idx + 1])
        matrix[idx, idx] = matrix[idx, idx] - upper_scale
        matrix[idx, idx + 1] = matrix[idx, idx + 1] + upper_scale
        matrix[idx + 1, idx] = matrix[idx + 1, idx] + lower_scale
        matrix[idx + 1, idx + 1] = matrix[idx + 1, idx + 1] - lower_scale
    return matrix


def _build_area_weighted_tridiagonal_coefficients(
    depths: torch.Tensor,
    area: torch.Tensor,
    kz: torch.Tensor,
    dt_seconds: float = SECONDS_PER_DAY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lower/diagonal/upper coefficients for the implicit diffusion matrix."""
    depths = depths.flatten()
    area = torch.clamp(area.flatten().to(device=depths.device, dtype=depths.dtype), min=1.0e-4)
    if kz.ndim == 1:
        kz = kz.unsqueeze(0)
    kz = torch.clamp(kz.to(device=depths.device, dtype=depths.dtype), min=MIN_TOTAL_DIFFUSIVITY, max=MAX_TOTAL_DIFFUSIVITY)
    batch = int(kz.shape[0])
    n_depths = int(depths.numel())
    dt = torch.as_tensor(float(dt_seconds), dtype=depths.dtype, device=depths.device)
    diag = torch.ones((batch, n_depths), dtype=depths.dtype, device=depths.device)
    if n_depths <= 1:
        empty = torch.empty((batch, 0), dtype=depths.dtype, device=depths.device)
        return empty, diag, empty

    dz_cell = layer_thicknesses(depths)
    dz_interface = torch.clamp(depths[1:] - depths[:-1], min=1.0e-4)
    area_interface = 0.5 * (area[:-1] + area[1:])
    kz_interface = 0.5 * (kz[:, :-1] + kz[:, 1:])
    conductance = area_interface.reshape(1, -1) * kz_interface / dz_interface.reshape(1, -1)

    upper_scale = conductance / (area[:-1] * dz_cell[:-1]).reshape(1, -1)
    lower_scale = conductance / (area[1:] * dz_cell[1:]).reshape(1, -1)

    upper = -dt * upper_scale
    lower = -dt * lower_scale
    diag[:, :-1] = diag[:, :-1] + dt * upper_scale
    diag[:, 1:] = diag[:, 1:] + dt * lower_scale
    return lower, diag, upper


def _safe_tridiagonal_pivot(pivot: torch.Tensor) -> torch.Tensor:
    eps = torch.as_tensor(torch.finfo(pivot.dtype).eps, dtype=pivot.dtype, device=pivot.device)
    signed_eps = torch.where(pivot < 0.0, -eps, eps)
    return torch.where(torch.abs(pivot) < eps, signed_eps, pivot)


def _solve_batched_tridiagonal(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve a batch of tridiagonal systems with Thomas elimination."""
    if rhs.ndim == 1:
        rhs = rhs.unsqueeze(0)
    n_depths = int(diag.shape[1])
    if n_depths == 1:
        return rhs / _safe_tridiagonal_pivot(diag[:, 0]).reshape(-1, 1)

    c_prime: list[torch.Tensor] = []
    d_prime: list[torch.Tensor] = []
    pivot = _safe_tridiagonal_pivot(diag[:, 0])
    c_prime.append(upper[:, 0] / pivot)
    d_prime.append(rhs[:, 0] / pivot)

    for idx in range(1, n_depths):
        pivot = _safe_tridiagonal_pivot(diag[:, idx] - lower[:, idx - 1] * c_prime[idx - 1])
        if idx < n_depths - 1:
            c_prime.append(upper[:, idx] / pivot)
        d_prime.append((rhs[:, idx] - lower[:, idx - 1] * d_prime[idx - 1]) / pivot)

    solution: list[torch.Tensor] = [torch.empty_like(rhs[:, 0]) for _ in range(n_depths)]
    solution[-1] = d_prime[-1]
    for idx in range(n_depths - 2, -1, -1):
        solution[idx] = d_prime[idx] - c_prime[idx] * solution[idx + 1]
    return torch.stack(solution, dim=1)


def one_day_heat_sources(
    depths: torch.Tensor,
    surface_flux_wm2: torch.Tensor,
    shortwave_wm2: torch.Tensor,
    kd: torch.Tensor,
    shortwave_surface_fraction: float = SHORTWAVE_SURFACE_FRACTION,
    area_profile: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return C/s source terms for surface energy and penetrating shortwave."""
    depths = depths.flatten()
    dz = layer_thicknesses(depths)
    edges = layer_edges_from_nodes(depths)
    area = (
        torch.ones_like(depths)
        if area_profile is None
        else torch.clamp(area_profile.flatten().to(device=depths.device, dtype=depths.dtype), min=1.0e-6)
    )
    surface_area = torch.clamp(area[0], min=1.0e-6)
    batch = surface_flux_wm2.reshape(-1, 1).shape[0]
    source = torch.zeros((batch, depths.numel()), dtype=depths.dtype, device=depths.device)

    surface_fraction = float(max(0.0, min(1.0, shortwave_surface_fraction)))
    source[:, 0] = source[:, 0] + surface_flux_wm2.reshape(-1) * surface_area / (RHO_CP * area[0] * dz[0])

    kd = torch.clamp(kd.reshape(-1, 1), min=0.03, max=3.0)
    z_top = edges[:-1].reshape(1, -1)
    z_bottom = edges[1:].reshape(1, -1)
    edge_area = area_edges_from_nodes(depths, area)
    q_pen = (
        (1.0 - surface_fraction)
        * (1.0 - SURFACE_ALBEDO_WATER)
        * shortwave_wm2.reshape(-1, 1)
    )
    flux_top = q_pen * torch.exp(-kd * z_top)
    flux_bottom = q_pen * torch.exp(-kd * z_bottom)
    absorbed_layer_power_w = (
        edge_area[:-1].reshape(1, -1) * flux_top
        - edge_area[1:].reshape(1, -1) * flux_bottom
    )
    absorbed_layer_power_w = torch.clamp(absorbed_layer_power_w, min=0.0)
    penetrating = absorbed_layer_power_w / (RHO_CP * area.reshape(1, -1) * dz.reshape(1, -1))
    return source + penetrating


def _implicit_diffusion_step_dense_reference(
    temperature: torch.Tensor,
    depths: torch.Tensor,
    area: torch.Tensor,
    kz: torch.Tensor,
    source_c_per_s: torch.Tensor | None = None,
    dt_seconds: float = SECONDS_PER_DAY,
) -> torch.Tensor:
    """Dense reference implementation used to validate the tridiagonal solver."""
    if temperature.ndim == 1:
        temperature = temperature.unsqueeze(0)
    if kz.ndim == 1:
        kz = kz.unsqueeze(0).expand_as(temperature)
    elif kz.shape[0] == 1 and temperature.shape[0] > 1:
        kz = kz.expand_as(temperature)
    if source_c_per_s is None:
        source_c_per_s = torch.zeros_like(temperature)
    elif source_c_per_s.ndim == 1:
        source_c_per_s = source_c_per_s.unsqueeze(0).expand_as(temperature)
    elif source_c_per_s.shape[0] == 1 and temperature.shape[0] > 1:
        source_c_per_s = source_c_per_s.expand_as(temperature)

    depths = depths.flatten().to(device=temperature.device, dtype=temperature.dtype)
    area = area.flatten().to(device=temperature.device, dtype=temperature.dtype)
    kz = kz.to(device=temperature.device, dtype=temperature.dtype)
    source_c_per_s = source_c_per_s.to(device=temperature.device, dtype=temperature.dtype)
    identity = torch.eye(depths.numel(), dtype=temperature.dtype, device=temperature.device)
    dt = torch.as_tensor(float(dt_seconds), dtype=temperature.dtype, device=temperature.device)
    outputs = []
    for row_idx in range(temperature.shape[0]):
        diffusion = build_area_weighted_diffusion_matrix(depths, area, kz[row_idx])
        lhs = identity - dt * diffusion
        rhs = temperature[row_idx] + dt * source_c_per_s[row_idx]
        outputs.append(torch.linalg.solve(lhs, rhs))
    return torch.stack(outputs, dim=0)


def implicit_diffusion_step(
    temperature: torch.Tensor,
    depths: torch.Tensor,
    area: torch.Tensor,
    kz: torch.Tensor,
    source_c_per_s: torch.Tensor | None = None,
    dt_seconds: float = SECONDS_PER_DAY,
) -> torch.Tensor:
    """Advance profiles by one implicit diffusion step."""
    if temperature.ndim == 1:
        temperature = temperature.unsqueeze(0)
    if kz.ndim == 1:
        kz = kz.unsqueeze(0).expand_as(temperature)
    elif kz.shape[0] == 1 and temperature.shape[0] > 1:
        kz = kz.expand_as(temperature)
    if source_c_per_s is None:
        source_c_per_s = torch.zeros_like(temperature)
    elif source_c_per_s.ndim == 1:
        source_c_per_s = source_c_per_s.unsqueeze(0).expand_as(temperature)
    elif source_c_per_s.shape[0] == 1 and temperature.shape[0] > 1:
        source_c_per_s = source_c_per_s.expand_as(temperature)

    depths = depths.flatten().to(device=temperature.device, dtype=temperature.dtype)
    area = area.flatten().to(device=temperature.device, dtype=temperature.dtype)
    kz = kz.to(device=temperature.device, dtype=temperature.dtype)
    source_c_per_s = source_c_per_s.to(device=temperature.device, dtype=temperature.dtype)
    dt = torch.as_tensor(float(dt_seconds), dtype=temperature.dtype, device=temperature.device)
    rhs = temperature + dt * source_c_per_s
    if depths.numel() <= 1:
        return rhs
    lower, diag, upper = _build_area_weighted_tridiagonal_coefficients(depths, area, kz, dt_seconds=dt_seconds)
    return _solve_batched_tridiagonal(lower, diag, upper, rhs)
