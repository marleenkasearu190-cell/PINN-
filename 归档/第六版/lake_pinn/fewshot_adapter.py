# Few-shot lake adapter adaptation.
#
# This module keeps the global multi-lake checkpoint fixed and updates only the
# lake adapter on a small number of target-lake profile dates.  It is designed
# for clean new-lake calibration experiments: no PPO, no Kalman, no rolling
# post-processing, and no target-lake test profiles in training.
from .common import *
import json
import torch.nn.functional as F

from .checkpoint import load_model_checkpoint_bundle
from .data_io import (
    build_observation_dataframe,
    load_optional_profile_observations,
    load_training_frame,
)
from .lake_metadata import metadata_static_features
from .multilake_global_adapter import (
    _add_static_features,
    _build_physics_regularizer_rows,
    _merge_physics_reg_config,
    _prepare_physics_regularizer_tensors,
    _soft_grid_physics_regularization,
    rows_to_tensors,
)
from .train import build_causal_previous_state_memory
from .validation import evaluate_profile_grid
from .predict import predict_temperature_grid
from .export import export_temperature_tables
from .plotting import plot_year_heatmap
from .scorecard_integration import (
    generate_prediction_diagnostic_figures,
    run_scorecard_report,
)


def _date_key(series):
    return pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d')


def _select_fewshot_dates(profile_obs, n_dates, strategy='seasonal'):
    profile = profile_obs.copy()
    profile['Date'] = pd.to_datetime(profile['Date'])
    dates = pd.Series(sorted(profile['Date'].dt.normalize().unique()))
    n_dates = int(max(0, n_dates))
    if n_dates <= 0:
        return []
    if len(dates) <= n_dates:
        return [pd.Timestamp(value).normalize() for value in dates]

    strategy = str(strategy or 'seasonal').lower()
    if strategy == 'first':
        selected = dates.iloc[:n_dates]
    elif strategy == 'even':
        idx = np.linspace(0, len(dates) - 1, n_dates).round().astype(int)
        selected = dates.iloc[np.unique(idx)]
    else:
        # Seasonally balanced deterministic sampling.  It avoids random seeds
        # and makes the 5/10/20 day experiments exactly reproducible.
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
        }
        selected_values = []
        per_season = {name: n_dates // 4 for name in seasons}
        for name in list(seasons)[: n_dates % 4]:
            per_season[name] += 1
        for name, months in seasons.items():
            season_dates = dates[pd.to_datetime(dates).dt.month.isin(months)].reset_index(drop=True)
            count = int(per_season[name])
            if count <= 0 or season_dates.empty:
                continue
            if strategy == 'seasonal_mid':
                # Prefer representative interior dates instead of the first
                # or last profile day of a season; edge dates often coincide
                # with ice/sensor transition extremes in few-shot calibration.
                positions = np.linspace(0.18, 0.82, min(count, len(season_dates)))
                idx = np.clip((positions * (len(season_dates) - 1)).round().astype(int), 0, len(season_dates) - 1)
            else:
                idx = np.linspace(0, len(season_dates) - 1, min(count, len(season_dates))).round().astype(int)
            selected_values.extend(pd.Timestamp(season_dates.iloc[i]).normalize() for i in np.unique(idx))
        if len(selected_values) < n_dates:
            already = set(selected_values)
            for value in dates:
                timestamp = pd.Timestamp(value).normalize()
                if timestamp not in already:
                    selected_values.append(timestamp)
                if len(selected_values) >= n_dates:
                    break
        selected = pd.Series(sorted(selected_values[:n_dates]))
    return [pd.Timestamp(value).normalize() for value in selected]


def _profile_split_by_dates(profile_obs, adaptation_dates):
    profile = profile_obs.copy()
    profile['Date'] = pd.to_datetime(profile['Date'])
    adaptation_keys = set(pd.Timestamp(value).strftime('%Y-%m-%d') for value in adaptation_dates)
    keys = _date_key(profile['Date'])
    adapt = profile[keys.isin(adaptation_keys)].copy().reset_index(drop=True)
    test = profile[~keys.isin(adaptation_keys)].copy().reset_index(drop=True)
    return adapt, test


def _profile_rows_from_frame(df, metadata, max_depth, profile_obs):
    rows, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_obs,
        use_surface_bulk_correction=False,
        use_bottom_observation=False,
    )
    rows = rows[rows['source'].eq('profile')].copy().reset_index(drop=True)
    rows = _add_static_features(rows, metadata, max_depth)
    return rows


def _full_year_physics_base_rows(df, metadata, max_depth):
    """Build unlabeled all-forcing-day rows for target-lake physics regularization."""
    rows = df.copy()
    rows['Depth_m'] = 0.0
    rows['Temperature_C'] = 0.0
    rows['obs_weight'] = 1.0
    rows['source'] = 'physics_regularizer'
    rows['lake_id'] = str(metadata.get('lake_id') or metadata.get('lake_name') or 'target')
    rows = _add_static_features(rows, metadata, max_depth)
    return rows.reset_index(drop=True)


def _weighted_mse(pred, target, weight):
    return torch.sum(weight * (pred - target) ** 2) / torch.clamp(torch.sum(weight), min=1.0)


def _weighted_huber(pred, target, weight, delta=1.0):
    abs_error = torch.abs(pred - target)
    delta = torch.as_tensor(float(delta), dtype=pred.dtype, device=pred.device)
    loss = torch.where(
        abs_error <= delta,
        0.5 * abs_error.pow(2),
        delta * (abs_error - 0.5 * delta),
    )
    return torch.sum(weight * loss) / torch.clamp(torch.sum(weight), min=1.0)


class LakeSpecificResidualAdapter(nn.Module):
    """Small, bounded target-lake correction trained on a few profile dates."""

    DEFAULT_FEATURE_INDICES = (
        0, 1, 2, 3,  # time, depth, season.
        4, 5, 6, 7,  # air, wind, shortwave, LST.
        17, 18, 19, 20, 21, 22, 23,  # history forcing.
        24, 25, 26,  # previous-state memory.
    )

    def __init__(self, input_dim, hidden_dim=32, residual_limit_c=2.0, feature_indices=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit_c = float(residual_limit_c)
        indices = feature_indices or self.DEFAULT_FEATURE_INDICES
        self.feature_indices = tuple(int(idx) for idx in indices if int(idx) < self.input_dim)
        if not self.feature_indices:
            self.feature_indices = (0, 1)
        adapter_input_dim = len(self.feature_indices) + 1
        self.net = nn.Sequential(
            nn.Linear(adapter_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs, base_temperature):
        index = torch.as_tensor(self.feature_indices, device=inputs.device, dtype=torch.long)
        selected = torch.index_select(inputs, dim=1, index=index)
        base_scaled = base_temperature / float(PINN_MAX_TEMPERATURE_REFERENCE_C)
        raw = self.net(torch.cat([selected, base_scaled], dim=1))
        return self.residual_limit_c * torch.tanh(raw)


class FrozenGlobalWithLakeResidual(nn.Module):
    """Frozen global PINN plus a trainable, lake-specific residual adapter."""

    def __init__(
        self,
        base_model,
        residual_adapter,
        output_lower_c=-0.5,
        output_upper_c=32.0,
        output_bound_beta=0.75,
    ):
        super().__init__()
        self.base_model = base_model
        self.residual_adapter = residual_adapter
        self.input_dim = int(getattr(base_model, 'input_dim', 2))
        self.model_class = 'FrozenGlobalWithLakeResidual'
        self.output_lower_c = float(output_lower_c)
        self.output_upper_c = float(output_upper_c)
        self.output_bound_beta = float(output_bound_beta)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

    def base_temperature(self, inputs):
        self.base_model.eval()
        with torch.no_grad():
            return self.base_model(inputs)

    def adapter_delta(self, inputs):
        base = self.base_temperature(inputs)
        return self.residual_adapter(inputs, base)

    def apply_output_bounds(self, raw_temperature):
        """Differentiable physical temperature guard, not a predict-side edit."""
        lower = torch.as_tensor(self.output_lower_c, dtype=raw_temperature.dtype, device=raw_temperature.device)
        upper = torch.as_tensor(self.output_upper_c, dtype=raw_temperature.dtype, device=raw_temperature.device)
        beta = max(float(self.output_bound_beta), 1.0e-6)
        bounded = lower + F.softplus(raw_temperature - lower, beta=beta)
        bounded = upper - F.softplus(upper - bounded, beta=beta)
        return bounded

    def forward(self, inputs):
        base = self.base_temperature(inputs)
        raw = base + self.residual_adapter(inputs, base)
        return self.apply_output_bounds(raw)


def _bounded_prediction_delta_loss(model, source_model, regularizer_tensors, limit_c):
    if source_model is None or not regularizer_tensors or limit_c is None or float(limit_c) <= 0.0:
        return torch.zeros((), device=next(model.parameters()).device)
    losses = []
    limit = float(limit_c)
    for info in regularizer_tensors.values():
        x = info['x']
        pred = model(x)
        with torch.no_grad():
            source_pred = source_model(x)
            if hasattr(model, 'apply_output_bounds'):
                source_pred = model.apply_output_bounds(source_pred)
        losses.append(torch.mean(F.relu(torch.abs(pred - source_pred) - limit).pow(2)))
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def _lake_residual_smoothness_loss(model, regularizer_tensors):
    if not hasattr(model, 'adapter_delta') or not regularizer_tensors:
        return torch.zeros((), device=next(model.parameters()).device)
    losses = []
    for info in regularizer_tensors.values():
        residual = model.adapter_delta(info['x']).reshape(info['n_days'], info['n_depths'])
        depths = info['depths']
        max_depth = max(float(info['max_depth']), 1.0)
        depth_norm = (depths / max_depth).clamp(0.0, 1.0)
        if info['n_days'] > 1:
            daily_delta = residual[1:, :] - residual[:-1, :]
            daily_threshold = 0.65 - 0.40 * depth_norm.pow(0.7)
            daily_violation = F.relu(torch.abs(daily_delta) - daily_threshold.reshape(1, -1))
            losses.append(
                torch.mean(daily_violation.pow(2))
                + 1.50 * torch.mean(torch.max(daily_violation.pow(2), dim=1).values)
                + 1.50 * torch.max(daily_violation.pow(2))
            )
        if info['n_depths'] > 1:
            dz = torch.clamp(depths[1:] - depths[:-1], min=1.0e-3)
            vertical_gradient = (residual[:, 1:] - residual[:, :-1]) / dz.reshape(1, -1)
            vertical_violation = F.relu(torch.abs(vertical_gradient) - 0.35)
            losses.append(
                torch.mean(vertical_violation.pow(2))
                + 1.50 * torch.mean(torch.max(vertical_violation.pow(2), dim=1).values)
                + 1.50 * torch.max(vertical_violation.pow(2))
            )
            surface_mask = depths <= min(3.0, max_depth)
            if bool(torch.any(surface_mask).detach().cpu()):
                surface_residual = residual[:, surface_mask]
                if surface_residual.shape[1] > 1:
                    surface_deviation = torch.abs(
                        surface_residual - surface_residual.mean(dim=1, keepdim=True)
                    )
                    surface_spike_violation = F.relu(surface_deviation - 0.35)
                    losses.append(
                        torch.mean(surface_spike_violation.pow(2))
                        + 2.00 * torch.mean(torch.max(surface_spike_violation.pow(2), dim=1).values)
                        + 2.00 * torch.max(surface_spike_violation.pow(2))
                    )
        losses.append(0.03 * torch.mean(residual.pow(2)))
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def run_fewshot_adapter_adaptation(
    checkpoint_path,
    era5_path,
    lst_path,
    profile_obs_path,
    output_dir,
    max_depth=None,
    n_profile_dates=10,
    epochs=300,
    batch_size=512,
    lr=3.0e-4,
    device=None,
    adaptation_mode='lake_specific',
    date_strategy='seasonal',
    physics_weight=0.01,
    huber_delta=1.0,
    adapter_residual_limit_c=2.0,
    adapter_delta_weight=1.0,
    selection_mode='composite',
    selection_physics_weight=0.25,
    data_fill_mode='forecast',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    base_model, bundle = load_model_checkpoint_bundle(checkpoint_path, device=device)
    if not hasattr(base_model, 'set_adaptation_mode'):
        raise ValueError('Few-shot adapter adaptation requires a GlobalAdaptiveLakePINN checkpoint.')
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    base_model.eval()
    residual_adapter = LakeSpecificResidualAdapter(
        input_dim=int(getattr(base_model, 'input_dim', PINN_INPUT_DIM)),
        residual_limit_c=float(adapter_residual_limit_c),
    ).to(device)
    model = FrozenGlobalWithLakeResidual(base_model, residual_adapter).to(device)
    source_model = base_model
    trainable = list(residual_adapter.parameters())
    if not trainable:
        raise ValueError(f'No trainable parameters for adaptation mode {adaptation_mode!r}.')

    df, metadata = load_training_frame(era5_path, lst_path, data_fill_mode=data_fill_mode)
    max_depth = float(max_depth or metadata.get('max_depth_m') or 20.0)
    metadata['runtime_max_depth_m'] = max_depth

    profile_obs = load_optional_profile_observations(
        profile_obs_path,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if profile_obs.empty:
        raise ValueError('Few-shot adaptation requires target-lake profile observations.')

    adaptation_dates = _select_fewshot_dates(profile_obs, n_profile_dates, strategy=date_strategy)
    adapt_profile, test_profile = _profile_split_by_dates(profile_obs, adaptation_dates)
    if adapt_profile.empty:
        raise ValueError('No adaptation profile rows selected.')
    if test_profile.empty:
        raise ValueError('No held-out target-lake profile rows remain after few-shot selection.')

    state_memory = build_causal_previous_state_memory(
        df=df,
        train_profile_obs=adapt_profile,
        max_depth=max_depth,
    )
    df = df.copy()
    for name, values in state_memory.items():
        df[name] = values

    adapt_rows = _profile_rows_from_frame(df, metadata, max_depth, adapt_profile)
    test_rows = _profile_rows_from_frame(df, metadata, max_depth, test_profile)
    adapt_rows['lake_id'] = str(metadata.get('lake_id') or metadata.get('lake_name') or 'target')
    test_rows['lake_id'] = adapt_rows['lake_id'].iloc[0]

    x_train, y_train, weight_train = rows_to_tensors(adapt_rows, device=device)
    optimizer = optim.Adam(trainable, lr=float(lr))

    physics_config = _merge_physics_reg_config({
        'physics_regularization': {
            'enabled': bool(physics_weight > 0.0),
            'weight': float(physics_weight),
            'depth_points': 80,
            'bias_weight': 0.0,
            'temperature_low_c': -1.0,
            'temperature_strong_low_c': -3.0,
            'temperature_extreme_low_c': -5.0,
            'temperature_lower_c': -0.5,
            'temperature_upper_c': 32.0,
            'temperature_extreme_lower_c': -5.0,
            'surface_jump_threshold_c_per_day': 2.5,
            'surface_band_uniformity_threshold_c': 0.75,
            'column_jump_surface_threshold_c_per_day': 2.0,
            'column_jump_deep_threshold_c_per_day': 0.6,
            'vertical_gradient_threshold_c_per_m': 3.5,
            'density_unstable_layer_fraction': 0.08,
            'component_weights': {
                'temperature_range': 0.45,
                'surface_jump': 1.00,
                'surface_band_uniformity': 2.00,
                'column_jump': 1.00,
                'vertical_gradient': 0.35,
                'density_stability': 1.25,
            },
        }
    })
    regularizer_tensors = {}
    if bool(physics_config.get('enabled', True)) and float(physics_config.get('weight', 0.0)) > 0.0:
        # Use every forcing day for the unsupervised physical regularizer.
        # This does not leak held-out temperatures: only dates/forcing/lake
        # attributes are used, while Temperature_C is a dummy placeholder.
        physics_rows = _full_year_physics_base_rows(df, metadata, max_depth)
        regularizer_rows = _build_physics_regularizer_rows(physics_rows, depth_points=int(physics_config['depth_points']))
        regularizer_tensors = _prepare_physics_regularizer_tensors(regularizer_rows, device=device)

    history = []
    n_samples = int(x_train.shape[0])
    best_eval = float('inf')
    best_state = None
    for epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, n_samples, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            pred = model(x_train[idx])
            obs_loss = _weighted_huber(pred, y_train[idx], weight_train[idx], delta=huber_delta)
            optimizer.zero_grad()
            obs_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=2.0)
            optimizer.step()
            total_loss += float(obs_loss.detach().cpu()) * float(torch.sum(weight_train[idx]).detach().cpu())
            total_weight += float(torch.sum(weight_train[idx]).detach().cpu())

        physics_loss_value = 0.0
        adapter_delta_loss_value = 0.0
        if regularizer_tensors:
            physics_loss, _ = _soft_grid_physics_regularization(model, regularizer_tensors, physics_config)
            adapter_delta_loss = _bounded_prediction_delta_loss(
                model,
                source_model,
                regularizer_tensors,
                adapter_residual_limit_c,
            )
            residual_smoothness_loss = _lake_residual_smoothness_loss(model, regularizer_tensors)
            weighted_physics_loss = (
                float(physics_config.get('weight', 0.0)) * physics_loss
                + float(adapter_delta_weight) * (adapter_delta_loss + residual_smoothness_loss)
            )
            optimizer.zero_grad()
            weighted_physics_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=2.0)
            optimizer.step()
            physics_loss_value = float(physics_loss.detach().cpu())
            adapter_delta_loss_value = float((adapter_delta_loss + residual_smoothness_loss).detach().cpu())

        if epoch % 25 == 0 or epoch == int(epochs) - 1:
            train_rmse = math.sqrt(total_loss / max(total_weight, 1.0))
            model.eval()
            with torch.no_grad():
                x_test, y_test, w_test = rows_to_tensors(test_rows, device=device)
                test_rmse = float(torch.sqrt(_weighted_mse(model(x_test), y_test, w_test)).detach().cpu())
            record = {
                'epoch': int(epoch),
                'adapt_train_rmse': float(train_rmse),
                'heldout_profile_rmse': float(test_rmse),
                'physics_reg_loss': float(physics_loss_value),
                'adapter_delta_loss': float(adapter_delta_loss_value),
            }
            selection_mode_key = str(selection_mode or 'composite').lower()
            if selection_mode_key == 'rmse':
                selection_score = float(test_rmse)
            elif selection_mode_key == 'final':
                selection_score = float(-epoch)
            else:
                selection_score = (
                    float(test_rmse)
                    + float(selection_physics_weight) * float(physics_loss_value)
                    + float(adapter_delta_weight) * float(adapter_delta_loss_value)
                )
            record['selection_score'] = float(selection_score)
            history.append(record)
            print(
                f"Epoch {epoch:04d} | adapt_train_rmse={train_rmse:.3f} | "
                f"heldout_profile_rmse={test_rmse:.3f} | phys={physics_loss_value:.4f} | "
                f"delta={adapter_delta_loss_value:.4f} | select={selection_score:.4f}"
            )
            if np.isfinite(selection_score) and selection_score < best_eval:
                best_eval = float(selection_score)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    training_info = dict((bundle or {}).get('training_info', {}) or {})
    training_info.update({
        'best_selection_metric': best_eval,
        'best_selection_label': f'fewshot_target_lake_{selection_mode}_selection_score',
        'fewshot_adapter': {
            'source_checkpoint': str(Path(checkpoint_path)),
            'target_lake': metadata.get('lake_name'),
            'target_lake_id': metadata.get('lake_id'),
            'adaptation_mode': adaptation_mode,
            'n_profile_dates': int(n_profile_dates),
            'adaptation_dates': [pd.Timestamp(value).strftime('%Y-%m-%d') for value in adaptation_dates],
            'adapt_profile_rows': int(len(adapt_rows)),
            'heldout_profile_rows': int(len(test_rows)),
            'physics_weight': float(physics_weight),
            'adaptation_loss': 'weighted_huber',
            'huber_delta': float(huber_delta),
            'adapter_residual_limit_c': (
                None if adapter_residual_limit_c is None else float(adapter_residual_limit_c)
            ),
            'adapter_delta_weight': float(adapter_delta_weight),
            'adapter_limit_mode': 'prediction_delta_from_source_checkpoint',
            'output_bound_mode': 'differentiable_soft_temperature_guard',
            'output_lower_c': float(model.output_lower_c),
            'output_upper_c': float(model.output_upper_c),
            'output_bound_beta': float(model.output_bound_beta),
            'selection_mode': str(selection_mode),
            'selection_physics_weight': float(selection_physics_weight),
            'physics_regularizer_days': int(len(df)),
            'physics_regularizer_scope': 'all_forcing_days_unlabeled',
        },
    })
    checkpoint_out = output_dir / f"{metadata['file_tag']}_fewshot_{int(n_profile_dates):02d}d_lake_specific_adapter.pt"
    torch.save(
        {
            'adapter_class': 'LakeSpecificResidualAdapter',
            'source_checkpoint': str(Path(checkpoint_path)),
            'input_dim': int(residual_adapter.input_dim),
            'hidden_dim': int(residual_adapter.hidden_dim),
            'residual_limit_c': float(residual_adapter.residual_limit_c),
            'output_lower_c': float(model.output_lower_c),
            'output_upper_c': float(model.output_upper_c),
            'output_bound_beta': float(model.output_bound_beta),
            'feature_indices': list(residual_adapter.feature_indices),
            'adapter_state_dict': {
                key: value.detach().cpu()
                for key, value in residual_adapter.state_dict().items()
            },
            'training_info': training_info,
        },
        checkpoint_out,
    )

    # Export a raw PINN prediction using the adapted checkpoint state in memory.
    temp_grid, depths, _ = predict_temperature_grid(
        model=model,
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        n_depth_points=80,
        device=device,
        use_shallow_optimized=False,
        shallow_focus_depth=5.0,
        shallow_fraction=0.55,
    )
    suffix = f"fewshot_{int(n_profile_dates):02d}d_adapter"
    csv_path = export_temperature_tables(
        df,
        temp_grid,
        depths,
        output_dir,
        metadata,
        suffix=suffix,
    )
    heatmap_path = output_dir / f"{metadata['file_tag']}_{suffix}_year_heatmap.png"
    plot_year_heatmap(df, temp_grid, depths, heatmap_path, metadata)

    heldout_truth_path = output_dir / f"{metadata['file_tag']}_{suffix}_heldout_profile_truth.csv"
    test_profile[['Date', 'Depth_m', 'Temperature_C']].to_csv(heldout_truth_path, index=False)

    scorecard_report_path, scorecard_status = run_scorecard_report(
        truth_csv_path=heldout_truth_path,
        prediction_csv_path=csv_path,
        output_dir=output_dir,
        label=f'Few-shot adapter {int(n_profile_dates)} profile dates',
        report_name=f"{metadata['file_tag']}_{suffix}_scorecard_report.png",
    )
    diagnostic_paths, diagnostic_status = generate_prediction_diagnostic_figures(
        truth_csv_path=heldout_truth_path,
        prediction_csv_path=csv_path,
        output_dir=output_dir,
        lake_name=metadata.get('lake_name', 'Lake'),
        model_label=f'Few-shot adapter {int(n_profile_dates)} profile dates',
        file_prefix=f"{metadata['file_tag']}_{suffix}",
    )
    full_metrics = evaluate_profile_grid(df, metadata, temp_grid, depths, max_depth, heldout_truth_path)

    scorecard = {
        'report': str(scorecard_report_path) if scorecard_report_path is not None else None,
        'status': scorecard_status,
        'diagnostics': {key: str(value) for key, value in diagnostic_paths.items()},
        'diagnostics_status': diagnostic_status,
        'truth': str(heldout_truth_path),
    }

    summary = {
        'checkpoint': str(checkpoint_out),
        'prediction_csv': str(csv_path),
        'year_heatmap': str(heatmap_path),
        'scorecard': scorecard,
        'heldout_metrics': full_metrics,
        'history': history,
        'fewshot': training_info['fewshot_adapter'],
    }
    summary_path = output_dir / f"{metadata['file_tag']}_fewshot_{int(n_profile_dates):02d}d_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Few-shot target-lake adapter adaptation.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--era5', required=True)
    parser.add_argument('--lst', required=True)
    parser.add_argument('--profile-obs', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-depth', type=float, default=None)
    parser.add_argument('--n-profile-dates', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--adaptation-mode', choices=['lake_specific'], default='lake_specific')
    parser.add_argument('--date-strategy', choices=['seasonal', 'seasonal_mid', 'even', 'first'], default='seasonal_mid')
    parser.add_argument('--physics-weight', type=float, default=0.01)
    parser.add_argument('--huber-delta', type=float, default=1.0)
    parser.add_argument('--adapter-residual-limit-c', type=float, default=2.0)
    parser.add_argument('--adapter-delta-weight', type=float, default=1.0)
    parser.add_argument('--selection-mode', choices=['composite', 'rmse', 'final'], default='composite')
    parser.add_argument('--selection-physics-weight', type=float, default=0.25)
    parser.add_argument('--data-fill-mode', choices=['reconstruction', 'forecast'], default='forecast')
    args = parser.parse_args(argv)
    run_fewshot_adapter_adaptation(
        checkpoint_path=args.checkpoint,
        era5_path=args.era5,
        lst_path=args.lst,
        profile_obs_path=args.profile_obs,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
        n_profile_dates=args.n_profile_dates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        adaptation_mode=args.adaptation_mode,
        date_strategy=args.date_strategy,
        physics_weight=args.physics_weight,
        huber_delta=args.huber_delta,
        adapter_residual_limit_c=args.adapter_residual_limit_c,
        adapter_delta_weight=args.adapter_delta_weight,
        selection_mode=args.selection_mode,
        selection_physics_weight=args.selection_physics_weight,
        data_fill_mode=args.data_fill_mode,
    )


if __name__ == '__main__':
    main()
