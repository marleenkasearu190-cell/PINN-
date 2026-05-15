import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.physics import compute_diffusivity


class LinearDepthTemperature(torch.nn.Module):
    """Small differentiable model for checking density-gradient physics."""

    input_dim = 2

    def __init__(self, intercept_c: float, normalized_depth_slope_c: float):
        super().__init__()
        self.intercept_c = float(intercept_c)
        self.normalized_depth_slope_c = float(normalized_depth_slope_c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[:, 0:1]
        z_norm = inputs[:, 1:2]
        return self.intercept_c + self.normalized_depth_slope_c * z_norm + 0.0 * t


def _diffusivity_for_profile(intercept_c: float, normalized_depth_slope_c: float):
    depth_m = 20.0
    z = torch.linspace(0.0, depth_m, 64, dtype=torch.float32).reshape(-1, 1)
    z = z.clone().detach().requires_grad_(True)
    t = torch.zeros_like(z).requires_grad_(True)
    wind = torch.full_like(z, 3.0)
    model = LinearDepthTemperature(intercept_c, normalized_depth_slope_c)

    _, _, _, density_gradient, richardson_number, diffusivity = compute_diffusivity(
        model=model,
        t_col=t,
        z_col=z,
        max_depth=depth_m,
        wind_speed=wind,
        water_depth=depth_m,
    )
    return density_gradient.detach(), richardson_number.detach(), diffusivity.detach()


def test_richardson_sign_and_diffusivity_response():
    # Around 1-4 C, density increases with temperature. Warming with depth is
    # stable for z-positive-down coordinates and should produce Ri > 0.
    stable_drho_dz, stable_ri, stable_k = _diffusivity_for_profile(1.0, 3.0)
    neutral_drho_dz, neutral_ri, neutral_k = _diffusivity_for_profile(2.5, 0.0)
    unstable_drho_dz, unstable_ri, unstable_k = _diffusivity_for_profile(4.0, -3.0)

    assert stable_drho_dz.mean().item() > 0.0
    assert stable_ri.mean().item() > 0.0
    assert unstable_drho_dz.mean().item() < 0.0
    assert unstable_ri.mean().item() < 0.0

    # Stable stratification should not get more eddy diffusion than neutral
    # water, while density inversion should trigger stronger mixing.
    assert stable_k.mean().item() < neutral_k.mean().item()
    assert unstable_k.mean().item() > neutral_k.mean().item()


if __name__ == "__main__":
    test_richardson_sign_and_diffusivity_response()
    print("physics diffusivity sanity check passed")
