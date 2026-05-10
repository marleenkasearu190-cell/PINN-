# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .lake_metadata import metadata_static_features

class LakePINN(nn.Module):
    """Lake temperature PINN with T(z, t, forcing, lake_attrs) -> temperature in degree Celsius."""

    def __init__(self, input_dim=2, hidden_dim=128, hidden_layers=8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.net(inputs)


def _feature_like(reference: torch.Tensor, value, scale: float = 1.0) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=reference.device, dtype=reference.dtype)
    else:
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        tensor = torch.full_like(reference, float(tensor))
    elif tensor.shape != reference.shape:
        tensor = tensor.expand_as(reference)
    if scale != 1.0:
        tensor = tensor / float(scale)
    return tensor


def build_model_inputs(
    model,
    t,
    z,
    max_depth,
    metadata=None,
    doy_sin=None,
    doy_cos=None,
    air_temp=None,
    wind_speed=None,
    shortwave=None,
    lst_surface=None,
    longwave=None,
    latent_heat=None,
    sensible_heat=None,
    secchi=None,
):
    z_norm = z / max(max_depth, 1.0e-6)
    input_dim = int(getattr(model, 'input_dim', 2))
    if input_dim <= 2:
        return torch.cat([t, z_norm], dim=1)

    static_features = metadata_static_features(metadata, max_depth=max_depth)
    if input_dim >= PINN_EXTENDED_FORCING_INPUT_DIM:
        features = [
            t,
            z_norm,
            _feature_like(t, doy_sin),
            _feature_like(t, doy_cos),
            _feature_like(t, air_temp, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, wind_speed, PINN_MAX_WIND_REFERENCE_M_PER_S),
            _feature_like(t, shortwave, PINN_MAX_SHORTWAVE_REFERENCE_W_M2),
            _feature_like(t, lst_surface, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, longwave, PINN_MAX_LONGWAVE_REFERENCE_W_M2),
            _feature_like(t, latent_heat, PINN_MAX_HEAT_FLUX_REFERENCE_W_M2),
            _feature_like(t, sensible_heat, PINN_MAX_HEAT_FLUX_REFERENCE_W_M2),
            _feature_like(t, secchi, PINN_MAX_SECCHI_REFERENCE_M),
            _feature_like(t, static_features['max_depth_norm']),
            _feature_like(t, static_features['mean_depth_norm']),
            _feature_like(t, static_features['log_area']),
            _feature_like(t, static_features['latitude']),
            _feature_like(t, static_features['longitude']),
        ]
    else:
        features = [
            t,
            z_norm,
            _feature_like(t, doy_sin),
            _feature_like(t, doy_cos),
            _feature_like(t, air_temp, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, wind_speed, PINN_MAX_WIND_REFERENCE_M_PER_S),
            _feature_like(t, shortwave, PINN_MAX_SHORTWAVE_REFERENCE_W_M2),
            _feature_like(t, lst_surface, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, static_features['max_depth_norm']),
            _feature_like(t, static_features['log_area']),
            _feature_like(t, static_features['latitude']),
        ]
    return torch.cat(features[:input_dim], dim=1)


def model_temperature(
    model,
    t,
    z,
    max_depth,
    metadata=None,
    doy_sin=None,
    doy_cos=None,
    air_temp=None,
    wind_speed=None,
    shortwave=None,
    lst_surface=None,
    longwave=None,
    latent_heat=None,
    sensible_heat=None,
    secchi=None,
):
    return model(
        build_model_inputs(
            model=model,
            t=t,
            z=z,
            max_depth=max_depth,
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
        )
    )
