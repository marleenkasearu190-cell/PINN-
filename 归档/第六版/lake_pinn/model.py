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


def _linear_layers(module: nn.Module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            yield child


def _init_linear_stack(module: nn.Module) -> None:
    for layer in _linear_layers(module):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


class GlobalAdaptiveLakePINN(nn.Module):
    """
    Two-level LakePINN for multi-lake transfer.

    The shared trunk learns cross-lake physics/forcing responses.  A small
    lake-attribute FiLM adapter adds low-rank lake-specific corrections, so a
    new lake can be adapted by training only the adapter or final head instead
    of rewriting the global model.
    """

    DEFAULT_LAKE_ATTR_INDICES = (
        11,  # Secchi / transparency when present in the 37D input.
        12, 13, 14, 15, 16,  # max depth, mean depth, area, latitude, longitude.
        27, 28, 29, 30, 31, 32,  # volume, elevation, Kd, fetch, wind exposure, basin shape.
        34, 35,  # dynamic Kd/effective fetch if provided.
    )

    def __init__(
        self,
        input_dim=PINN_INPUT_DIM,
        hidden_dim=128,
        hidden_layers=8,
        adapter_dim=32,
        lake_attr_indices=None,
        adapter_residual_limit_c=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.adapter_dim = int(adapter_dim)
        self.model_class = 'GlobalAdaptiveLakePINN'
        attr_indices = lake_attr_indices or self.DEFAULT_LAKE_ATTR_INDICES
        self.lake_attr_indices = tuple(int(idx) for idx in attr_indices if int(idx) < self.input_dim)
        if not self.lake_attr_indices:
            self.lake_attr_indices = (max(0, self.input_dim - 1),)
        self.lake_attr_dim = len(self.lake_attr_indices)

        trunk_layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.Tanh()]
        for _ in range(max(0, self.hidden_layers - 1)):
            trunk_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()])
        self.shared_trunk = nn.Sequential(*trunk_layers)
        self.global_head = nn.Linear(self.hidden_dim, 1)
        self.adapter_context = nn.Sequential(
            nn.Linear(self.lake_attr_dim, self.adapter_dim),
            nn.Tanh(),
            nn.Linear(self.adapter_dim, 2 * self.hidden_dim),
        )
        self.adapter_delta_head = nn.Linear(self.hidden_dim, 1)
        self.adapter_scale = 0.10
        self.adapter_residual_limit_c = None
        self.set_adapter_residual_limit(adapter_residual_limit_c)

        _init_linear_stack(self.shared_trunk)
        nn.init.xavier_normal_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        _init_linear_stack(self.adapter_context)
        nn.init.zeros_(self.adapter_context[-1].weight)
        nn.init.zeros_(self.adapter_context[-1].bias)
        nn.init.zeros_(self.adapter_delta_head.weight)
        nn.init.zeros_(self.adapter_delta_head.bias)

    def lake_attribute_tensor(self, inputs):
        index = torch.as_tensor(self.lake_attr_indices, device=inputs.device, dtype=torch.long)
        return torch.index_select(inputs, dim=1, index=index)

    def set_adapter_residual_limit(self, limit_c=None):
        if limit_c is None:
            self.adapter_residual_limit_c = None
        else:
            limit_c = float(limit_c)
            self.adapter_residual_limit_c = limit_c if limit_c > 0.0 else None
        return self

    def forward(self, inputs):
        shared = self.shared_trunk(inputs)
        global_temperature = self.global_head(shared)
        adapter_params = self.adapter_context(self.lake_attribute_tensor(inputs))
        gamma, beta = torch.chunk(adapter_params, chunks=2, dim=1)
        adapted = shared * (1.0 + self.adapter_scale * torch.tanh(gamma)) + self.adapter_scale * beta
        adapter_residual = self.adapter_delta_head(torch.tanh(adapted))
        residual_limit = getattr(self, 'adapter_residual_limit_c', None)
        if residual_limit is not None and residual_limit > 0.0:
            adapter_residual = float(residual_limit) * torch.tanh(adapter_residual / float(residual_limit))
        return global_temperature + adapter_residual

    def set_adaptation_mode(self, mode='full'):
        mode = str(mode or 'full').lower()
        for parameter in self.parameters():
            parameter.requires_grad = mode == 'full'
        if mode == 'adapter_only':
            for parameter in self.adapter_context.parameters():
                parameter.requires_grad = True
            for parameter in self.adapter_delta_head.parameters():
                parameter.requires_grad = True
        elif mode == 'adapter_context_only':
            for parameter in self.adapter_context.parameters():
                parameter.requires_grad = True
        elif mode == 'last_layer':
            for parameter in self.global_head.parameters():
                parameter.requires_grad = True
        elif mode == 'adapter_and_head':
            for parameter in self.adapter_context.parameters():
                parameter.requires_grad = True
            for parameter in self.adapter_delta_head.parameters():
                parameter.requires_grad = True
            for parameter in self.global_head.parameters():
                parameter.requires_grad = True
        elif mode != 'full':
            raise ValueError(f'Unknown adaptation mode: {mode}')
        self.adaptation_mode = mode
        return self


def create_lake_model(
    model_architecture='legacy',
    input_dim=PINN_INPUT_DIM,
    hidden_dim=128,
    hidden_layers=8,
    adapter_dim=32,
    lake_attr_indices=None,
    adapter_residual_limit_c=None,
):
    architecture = str(model_architecture or 'legacy').lower()
    if architecture in {'legacy', 'lakepinn', 'lake_pinn'}:
        return LakePINN(input_dim=input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if architecture in {'global_adapter', 'globaladaptive', 'global_adaptive', 'adaptive'}:
        return GlobalAdaptiveLakePINN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            adapter_dim=adapter_dim,
            lake_attr_indices=lake_attr_indices,
            adapter_residual_limit_c=adapter_residual_limit_c,
        )
    raise ValueError(f'Unknown model architecture: {model_architecture}')


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
            _feature_like(t, air_temp_mean_7d, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, air_temp_mean_30d, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, shortwave_sum_7d, PINN_SHORTWAVE_SUM_7D_REFERENCE),
            _feature_like(t, shortwave_sum_30d, PINN_SHORTWAVE_SUM_30D_REFERENCE),
            _feature_like(t, wind_mean_7d, PINN_MAX_WIND_REFERENCE_M_PER_S),
            _feature_like(t, lst_mean_7d, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, heating_degree_days_30d, PINN_HEATING_DEGREE_DAYS_30D_REFERENCE),
            _feature_like(t, prev_surface_temp, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, prev_0_3m_mean, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, prev_deep_mean, PINN_MAX_TEMPERATURE_REFERENCE_C),
            _feature_like(t, static_features['volume_norm']),
            _feature_like(t, static_features['elevation_norm']),
            _feature_like(t, static_features['light_extinction_norm']),
            _feature_like(t, static_features['fetch_norm']),
            _feature_like(t, static_features['wind_exposure_norm']),
            _feature_like(t, static_features['basin_shape_norm']),
            _feature_like(t, water_level_anomaly, PINN_WATER_LEVEL_ANOMALY_REFERENCE_M),
            _feature_like(t, light_extinction_kd, PINN_LIGHT_EXTINCTION_REFERENCE_M_INV),
            _feature_like(t, effective_fetch, PINN_FETCH_REFERENCE_M),
            _feature_like(t, net_inflow, PINN_INFLOW_REFERENCE_M3_S),
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
    )
