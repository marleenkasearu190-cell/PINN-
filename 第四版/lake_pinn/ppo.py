# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

def build_ppo_state(summary, weights, kalman_scales, learning_rate, validation_metrics=None):
    def get_metric(name, default=0.0):
        if validation_metrics is not None and name in validation_metrics:
            return float(np.nan_to_num(validation_metrics.get(name, default), nan=default))
        return float(np.nan_to_num(summary.get(name, default), nan=default))

    validation_rmse = get_metric('rmse', 0.0)
    validation_mae = get_metric('mae', 0.0)
    validation_bias = get_metric('bias', 0.0)
    surface_rmse = get_metric('surface_rmse', validation_rmse)
    warm_surface_bias = get_metric('warm_surface_bias', max(validation_bias, 0.0))
    instability_penalty = get_metric('instability_penalty', max(summary.get('loss_pde', 0.0), 0.0))
    deep_warm_penalty = get_metric('deep_warm_penalty', 0.0)
    summer_strat_penalty = get_metric('summer_stratification_penalty', 0.0)
    summer_thermocline_depth_norm = get_metric('summer_thermocline_depth_norm', 0.0)
    summer_thermocline_thickness_penalty = get_metric('summer_thermocline_thickness_penalty', 0.0)
    summer_surface_warming_reward = get_metric('summer_surface_warming_reward', 0.0)
    summer_midlayer_temp_reward = get_metric('summer_midlayer_temp_reward', 0.0)
    summer_9m_temp = get_metric('summer_9m_temp', 0.0)
    summer_bottom_temp = get_metric('summer_bottom_temp', 0.0)
    autumn_overturn_penalty = get_metric('autumn_overturn_penalty', 0.0)
    autumn_surface_cooling_rate = get_metric('autumn_surface_cooling_rate', 0.0)
    autumn_gap_collapse = get_metric('autumn_gap_collapse', 0.0)
    autumn_false_overturn_penalty = get_metric('autumn_false_overturn_penalty', 0.0)
    autumn_cooling_triggered_overturn_reward = get_metric('autumn_cooling_triggered_overturn_reward', 0.0)
    winter_inverse_penalty = get_metric('winter_inverse_penalty', 0.0)
    winter_bottom_4c_error = get_metric('winter_bottom_4c_error', 0.0)
    deep_smoothness_penalty = get_metric('deep_smoothness_penalty', 0.0)
    lst_spike_indicator = get_metric('lst_spike_indicator', 0.0)

    state = np.array(
        [
            np.log10(summary['loss_pde'] + PPO_STATE_EPS),
            np.log10(summary['loss_bc'] + PPO_STATE_EPS),
            np.log10(summary['loss_ic'] + PPO_STATE_EPS),
            np.log10(summary['loss_obs'] + PPO_STATE_EPS),
            np.log10(summary['total'] + PPO_STATE_EPS),
            np.log10(summary['kappa_mean'] + PPO_STATE_EPS),
            float(summary['ri_mean']),
            *[np.log10(float(weights.get(key, 0.0)) + PPO_STATE_EPS) for key in PPO_WEIGHT_STATE_KEYS],
            np.log10(kalman_scales['process'] + PPO_STATE_EPS),
            np.log10(kalman_scales['obs'] + PPO_STATE_EPS),
            np.log10(learning_rate + PPO_STATE_EPS),
            validation_rmse,
            validation_mae,
            validation_bias,
            surface_rmse,
            warm_surface_bias,
            instability_penalty,
            deep_warm_penalty,
            summer_strat_penalty,
            summer_thermocline_depth_norm,
            summer_thermocline_thickness_penalty,
            summer_surface_warming_reward,
            summer_midlayer_temp_reward,
            summer_9m_temp,
            summer_bottom_temp,
            autumn_overturn_penalty,
            autumn_surface_cooling_rate,
            autumn_gap_collapse,
            autumn_false_overturn_penalty,
            autumn_cooling_triggered_overturn_reward,
            winter_inverse_penalty,
            winter_bottom_4c_error,
            deep_smoothness_penalty,
            lst_spike_indicator,
        ],
        dtype=np.float32,
    )
    return state


def update_control_value(current_value, action_value, lower, upper, step_size=0.35):
    base_value = max(float(current_value), float(lower), PPO_STATE_EPS)
    updated = base_value * float(np.exp(step_size * float(np.clip(action_value, -1.0, 1.0))))
    return float(np.clip(updated, lower, upper))


def normalize_kalman_scales(
    kalman_scales=None,
    default_process=1.0,
    default_obs=1.0,
    default_correlation_length=2.0,
    default_forecast_blend=0.2,
):
    merged = {
        'process': float(default_process),
        'obs': float(default_obs),
        'correlation_length': float(default_correlation_length),
        'forecast_blend': float(default_forecast_blend),
    }
    for key, value in dict(kalman_scales or {}).items():
        if key in merged:
            merged[key] = float(value)
    merged['process'] = float(np.clip(merged['process'], 0.5, 2.5))
    merged['obs'] = float(np.clip(merged['obs'], 0.6, 3.5))
    merged['correlation_length'] = float(np.clip(merged['correlation_length'], 0.6, 8.0))
    merged['forecast_blend'] = float(np.clip(merged['forecast_blend'], 0.02, 0.85))
    return merged


def apply_ppo_action(weights, kalman_scales, action, tune_kalman=True):
    action = np.asarray(action, dtype=np.float32)
    updated_weights = dict(weights)
    updated_scales = normalize_kalman_scales(kalman_scales)

    weight_specs = {
        'pde': (5.0e4, 6.0e5, 0.18),
        'bc': (2.0, 25.0, 0.15),
        'ic': (1.0, 20.0, 0.12),
        'obs': (0.4, 8.0, 0.12),
        'time_continuity': (0.05, 20.0, 0.10),
        'stratification': (0.05, 5.0, 0.10),
        'smoothness': (0.01, 1.0, 0.08),
        'deep_warming': (0.02, 1.0, 0.10),
        'deep_anchor': (0.01, 1.0, 0.10),
        'vertical_exchange': (0.05, 1.0, 0.10),
        'convective_mixing': (0.05, 1.0, 0.10),
        'autumn_overturn': (0.05, 1.0, 0.10),
        'heat_budget': (0.05, 1.0, 0.10),
    }

    for idx, key in enumerate(PPO_WEIGHT_STATE_KEYS):
        lower, upper, step = weight_specs[key]
        updated_weights[key] = update_control_value(weights.get(key, 0.0), action[idx], lower, upper, step_size=step)
    if tune_kalman:
        updated_scales['process'] = update_control_value(
            updated_scales['process'],
            action[len(PPO_WEIGHT_STATE_KEYS)],
            0.5,
            2.5,
            step_size=0.10,
        )
        updated_scales['obs'] = update_control_value(
            updated_scales['obs'],
            action[len(PPO_WEIGHT_STATE_KEYS) + 1],
            0.6,
            3.5,
            step_size=0.10,
        )
        updated_scales['correlation_length'] = update_control_value(
            updated_scales['correlation_length'],
            action[len(PPO_WEIGHT_STATE_KEYS) + 2],
            0.6,
            8.0,
            step_size=0.08,
        )
        updated_scales['forecast_blend'] = update_control_value(
            updated_scales['forecast_blend'],
            action[len(PPO_WEIGHT_STATE_KEYS) + 3],
            0.02,
            0.85,
            step_size=0.08,
        )

    return updated_weights, updated_scales


def compute_ppo_reward(prev_summary, current_summary, prev_validation_metrics=None, current_validation_metrics=None):
    def relative_improvement(prev_value, current_value, scale=1.0, floor=1.0):
        denom = max(abs(float(prev_value)), abs(float(current_value)), float(floor), PPO_STATE_EPS)
        delta = (float(prev_value) - float(current_value)) / denom
        return float(scale) * float(np.clip(delta, -2.0, 2.0))

    def relative_gain(prev_value, current_value, scale=1.0, floor=1.0):
        denom = max(abs(float(prev_value)), abs(float(current_value)), float(floor), PPO_STATE_EPS)
        delta = (float(current_value) - float(prev_value)) / denom
        return float(scale) * float(np.clip(delta, -2.0, 2.0))

    reward = 0.0

    total_prev = prev_summary['total'] + PPO_STATE_EPS
    total_curr = current_summary['total'] + PPO_STATE_EPS
    reward += relative_improvement(total_prev, total_curr, scale=0.10, floor=1.0)
    reward -= 0.02 * np.log10(current_summary['loss_bc'] + 1.0)
    reward -= 0.01 * np.log10(current_summary['loss_pde'] * 1.0e12 + 1.0)

    if prev_validation_metrics is not None and current_validation_metrics is not None:
        lst_reliability = float(np.clip(1.0 - current_validation_metrics.get('lst_spike_indicator', 0.0), 0.0, 1.0))
        current_profile_rmse = float(current_validation_metrics.get('profile_rmse', 0.0))
        current_profile_mae = float(current_validation_metrics.get('profile_mae', 0.0))
        current_profile_bias = float(current_validation_metrics.get('profile_bias', 0.0))
        current_profile_objective = float(
            current_validation_metrics.get(
                'profile_objective',
                current_profile_rmse + 0.15 * abs(current_profile_bias),
            )
        )
        current_profile_surface_band_rmse = float(
            current_validation_metrics.get('profile_surface_band_rmse', current_profile_rmse)
        )
        current_profile_thermocline_band_rmse = float(
            current_validation_metrics.get('profile_thermocline_band_rmse', current_profile_rmse)
        )
        current_profile_deep_band_rmse = float(
            current_validation_metrics.get('profile_deep_band_rmse', current_profile_rmse)
        )
        reward += 0.25 * relative_improvement(
            prev_validation_metrics.get('rmse', 0.0),
            current_validation_metrics.get('rmse', 0.0),
            scale=1.0,
            floor=1.0,
        )
        reward += relative_improvement(
            prev_validation_metrics.get(
                'profile_objective',
                prev_validation_metrics.get('profile_rmse', 0.0),
            ),
            current_profile_objective,
            scale=1.1,
            floor=0.5,
        )
        reward += relative_improvement(
            prev_validation_metrics.get(
                'profile_surface_band_rmse',
                prev_validation_metrics.get('profile_rmse', 0.0),
            ),
            current_profile_surface_band_rmse,
            scale=0.45,
            floor=0.35,
        )
        reward += relative_improvement(
            prev_validation_metrics.get(
                'profile_thermocline_band_rmse',
                prev_validation_metrics.get('profile_rmse', 0.0),
            ),
            current_profile_thermocline_band_rmse,
            scale=0.40,
            floor=0.45,
        )
        reward += relative_improvement(
            prev_validation_metrics.get(
                'profile_deep_band_rmse',
                prev_validation_metrics.get('profile_rmse', 0.0),
            ),
            current_profile_deep_band_rmse,
            scale=0.25,
            floor=0.45,
        )
        reward += relative_improvement(
            abs(prev_validation_metrics.get('profile_bias', 0.0)),
            abs(current_validation_metrics.get('profile_bias', 0.0)),
            scale=0.20,
            floor=0.25,
        )
        reward += lst_reliability * relative_improvement(
            prev_validation_metrics.get('surface_rmse', 0.0),
            current_validation_metrics.get('surface_rmse', 0.0),
            scale=0.9,
            floor=0.4,
        )
        reward += relative_improvement(
            abs(prev_validation_metrics.get('bias', 0.0)),
            abs(current_validation_metrics.get('bias', 0.0)),
            scale=0.5,
            floor=0.5,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('deep_warm_penalty', 0.0),
            current_validation_metrics.get('deep_warm_penalty', 0.0),
            scale=2.0,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('summer_stratification_penalty', 0.0),
            current_validation_metrics.get('summer_stratification_penalty', 0.0),
            scale=1.0,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('summer_thermocline_thickness_penalty', 0.0),
            current_validation_metrics.get('summer_thermocline_thickness_penalty', 0.0),
            scale=3.2,
            floor=0.1,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('may_surface_warm_penalty', 0.0),
            current_validation_metrics.get('may_surface_warm_penalty', 0.0),
            scale=1.2,
            floor=0.15,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('may_surface_rate_penalty', 0.0),
            current_validation_metrics.get('may_surface_rate_penalty', 0.0),
            scale=0.8,
            floor=0.08,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('july_surface_cool_penalty', 0.0),
            current_validation_metrics.get('july_surface_cool_penalty', 0.0),
            scale=1.2,
            floor=0.15,
        )
        reward += relative_gain(
            prev_validation_metrics.get('july_surface_warm_reward', 0.0),
            current_validation_metrics.get('july_surface_warm_reward', 0.0),
            scale=0.6,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('surface_band_background_rmse', 0.0),
            current_validation_metrics.get('surface_band_background_rmse', 0.0),
            scale=0.8,
            floor=0.2,
        )
        reward += relative_gain(
            prev_validation_metrics.get('summer_surface_warming_reward', 0.0),
            current_validation_metrics.get('summer_surface_warming_reward', 0.0),
            scale=1.8,
            floor=0.2,
        )
        reward += relative_gain(
            prev_validation_metrics.get('summer_midlayer_temp_reward', 0.0),
            current_validation_metrics.get('summer_midlayer_temp_reward', 0.0),
            scale=1.5,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('summer_9m_temp', 0.0),
            current_validation_metrics.get('summer_9m_temp', 0.0),
            scale=4.2,
            floor=6.0,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('summer_bottom_temp', 0.0),
            current_validation_metrics.get('summer_bottom_temp', 0.0),
            scale=3.4,
            floor=4.0,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('autumn_overturn_penalty', 0.0),
            current_validation_metrics.get('autumn_overturn_penalty', 0.0),
            scale=1.4,
            floor=0.2,
        )
        reward += relative_gain(
            prev_validation_metrics.get('autumn_surface_cooling_rate', 0.0),
            current_validation_metrics.get('autumn_surface_cooling_rate', 0.0),
            scale=1.8,
            floor=0.2,
        )
        reward += relative_gain(
            prev_validation_metrics.get('autumn_gap_collapse', 0.0),
            current_validation_metrics.get('autumn_gap_collapse', 0.0),
            scale=2.4,
            floor=0.2,
        )
        reward += relative_gain(
            prev_validation_metrics.get('autumn_cooling_triggered_overturn_reward', 0.0),
            current_validation_metrics.get('autumn_cooling_triggered_overturn_reward', 0.0),
            scale=2.6,
            floor=0.1,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('autumn_false_overturn_penalty', 0.0),
            current_validation_metrics.get('autumn_false_overturn_penalty', 0.0),
            scale=4.0,
            floor=0.1,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('winter_inverse_penalty', 0.0),
            current_validation_metrics.get('winter_inverse_penalty', 0.0),
            scale=1.6,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('winter_bottom_4c_error', 0.0),
            current_validation_metrics.get('winter_bottom_4c_error', 0.0),
            scale=2.2,
            floor=0.2,
        )
        reward += relative_improvement(
            prev_validation_metrics.get('deep_smoothness_penalty', 0.0),
            current_validation_metrics.get('deep_smoothness_penalty', 0.0),
            scale=1.4,
            floor=0.02,
        )
        reward -= 0.24 * float(current_validation_metrics.get('instability_penalty', 0.0))
        reward -= 0.18 * max(float(current_validation_metrics.get('bias', 0.0)), 0.0)
        reward -= 0.22 * float(current_validation_metrics.get('warm_surface_bias', 0.0)) * lst_reliability
        reward -= 0.08 * current_profile_objective
        reward -= 0.03 * abs(current_profile_bias)
        reward -= 0.80 * float(current_validation_metrics.get('summer_thermocline_thickness_penalty', 0.0))
        reward -= 1.20 * float(current_validation_metrics.get('may_surface_warm_penalty', 0.0))
        reward -= 0.80 * float(current_validation_metrics.get('may_surface_rate_penalty', 0.0))
        reward -= 1.20 * float(current_validation_metrics.get('july_surface_cool_penalty', 0.0))
        reward += 0.60 * float(current_validation_metrics.get('july_surface_warm_reward', 0.0))
        reward -= 0.80 * float(current_validation_metrics.get('surface_band_background_rmse', 0.0))
        reward += 0.60 * float(current_validation_metrics.get('summer_surface_warming_reward', 0.0))
        reward += 0.45 * float(current_validation_metrics.get('summer_midlayer_temp_reward', 0.0))
        reward -= 0.85 * max(float(current_validation_metrics.get('summer_9m_temp', 0.0)) - 14.5, 0.0)
        reward -= 0.60 * max(float(current_validation_metrics.get('summer_bottom_temp', 0.0)) - 7.8, 0.0)
        reward -= 0.55 * max(0.40 - float(current_validation_metrics.get('autumn_surface_cooling_rate', 0.0)), 0.0)
        reward -= 0.70 * max(0.50 - float(current_validation_metrics.get('autumn_gap_collapse', 0.0)), 0.0)
        reward += 0.70 * float(current_validation_metrics.get('autumn_cooling_triggered_overturn_reward', 0.0))
        reward -= 1.40 * float(current_validation_metrics.get('autumn_false_overturn_penalty', 0.0))
        reward -= 0.45 * float(current_validation_metrics.get('winter_bottom_4c_error', 0.0))
        reward -= 0.30 * float(current_validation_metrics.get('lst_spike_indicator', 0.0))

    return float(reward)


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state):
        features = self.body(state)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PPOController:
    def __init__(
        self,
        state_dim,
        action_dim,
        device='cpu',
        lr=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_eps=0.2,
        update_epochs=8,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.model = PPOActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
        }

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, value = self.model(state_tensor)
            std = torch.exp(self.model.log_std).unsqueeze(0)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action_clipped).sum(dim=-1)
        return (
            action_clipped.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.buffer['states'].append(np.asarray(state, dtype=np.float32))
        self.buffer['actions'].append(np.asarray(action, dtype=np.float32))
        self.buffer['log_probs'].append(float(log_prob))
        self.buffer['rewards'].append(float(reward))
        self.buffer['dones'].append(bool(done))
        self.buffer['values'].append(float(value))

    def update(self, last_state=None, last_done=True):
        if not self.buffer['states']:
            return None

        if last_done or last_state is None:
            last_value = 0.0
        else:
            last_state_tensor = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, last_value_tensor = self.model(last_state_tensor)
            last_value = float(last_value_tensor.item())

        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        values = self.buffer['values']
        advantages = []
        gae = 0.0
        next_value = last_value
        for idx in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[idx])
            delta = rewards[idx] + self.gamma * next_value * mask - values[idx]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[idx]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        states = torch.tensor(np.asarray(self.buffer['states']), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.asarray(self.buffer['actions']), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(self.buffer['log_probs'], dtype=torch.float32, device=self.device)

        for _ in range(self.update_epochs):
            mean, value_pred = self.model(states)
            std = torch.exp(self.model.log_std).unsqueeze(0).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratios = torch.exp(log_probs - old_log_probs)

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = torch.mean((returns - value_pred) ** 2)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        stats = {
            'buffer_size': len(rewards),
            'reward_mean': float(np.mean(rewards)),
            'reward_last': float(rewards[-1]),
        }
        self.reset_buffer()
        return stats


def export_ppo_policy_bundle(ppo_controller, final_weights, final_kalman_scales):
    if ppo_controller is None:
        return None
    return {
        'state_dim': int(getattr(ppo_controller.model.body[0], 'in_features', PPO_STATE_DIM)),
        'action_dim': int(getattr(ppo_controller.model.policy_head, 'out_features', PPO_TRAIN_ACTION_DIM)),
        'model_state_dict': {k: v.detach().cpu() for k, v in ppo_controller.model.state_dict().items()},
        'optimizer_state_dict': ppo_controller.optimizer.state_dict(),
        'final_weights': dict(final_weights),
        'final_kalman_scales': dict(final_kalman_scales),
    }


def build_ppo_controller_from_bundle(bundle, device='cpu'):
    if bundle is None:
        return None, None
    controller = PPOController(
        state_dim=int(bundle.get('state_dim', PPO_STATE_DIM)),
        action_dim=int(bundle.get('action_dim', 6)),
        device=device,
    )
    controller.model.load_state_dict(bundle['model_state_dict'])
    optimizer_state = bundle.get('optimizer_state_dict')
    if optimizer_state is not None:
        controller.optimizer.load_state_dict(optimizer_state)
    return controller, bundle


def derive_online_control_params_from_weights(
    initial_weights,
    memory_blend,
    surface_relaxation,
    surface_decay_depth,
    deep_inertia,
    deep_anchor,
    surface_skin_cooling_coef,
):
    """Map trained loss weights into runtime rolling/KF controls for predict mode."""
    weights = dict(initial_weights or {})
    base_controls = {
        'memory_blend': float(np.clip(memory_blend, 0.0, 1.0)),
        'surface_relaxation': float(np.clip(surface_relaxation, 0.0, 1.0)),
        'surface_decay_depth': float(np.clip(surface_decay_depth, 1.5, 6.5)),
        'deep_inertia': float(np.clip(deep_inertia, 0.0, 0.95)),
        'deep_anchor': float(np.clip(deep_anchor, 0.0, 0.5)),
        'surface_skin_cooling_coef': float(np.clip(surface_skin_cooling_coef, 0.005, 0.08)),
    }

    def norm_ratio(key, reference):
        value = max(float(weights.get(key, reference)), 1.0e-6)
        reference = max(float(reference), 1.0e-6)
        return float(np.clip(np.log(value / reference), -2.0, 2.0))

    obs_pull = norm_ratio('obs', 1.0)
    continuity_pull = norm_ratio('time_continuity', 0.5)
    strat_pull = norm_ratio('stratification', 0.8)
    smooth_pull = norm_ratio('smoothness', 0.15)
    deepwarm_pull = norm_ratio('deep_warming', 0.25)
    deepprotection_pull = 0.6 * strat_pull + 0.9 * deepwarm_pull + 0.4 * smooth_pull
    mixing_pull = 0.5 * norm_ratio('vertical_exchange', 0.35) + 0.5 * norm_ratio('convective_mixing', 0.25)
    autumn_pull = norm_ratio('autumn_overturn', 0.22)
    heat_pull = norm_ratio('heat_budget', 0.30)

    memory_shift = (
        0.05 * continuity_pull
        + 0.04 * deepprotection_pull
        - 0.05 * obs_pull
        - 0.04 * mixing_pull
        - 0.02 * autumn_pull
    )
    surface_shift = (
        0.05 * obs_pull
        + 0.02 * heat_pull
        - 0.02 * continuity_pull
        - 0.03 * smooth_pull
    )
    surface_decay_shift = (
        -0.28 * strat_pull
        -0.22 * smooth_pull
        -0.18 * deepwarm_pull
        + 0.20 * mixing_pull
        + 0.10 * obs_pull
    )
    skin_cooling_shift = (
        0.004 * obs_pull
        + 0.003 * heat_pull
        - 0.002 * mixing_pull
    )
    deep_inertia_shift = (
        0.07 * deepprotection_pull
        + 0.03 * continuity_pull
        - 0.05 * mixing_pull
        - 0.02 * autumn_pull
    )
    deep_anchor_shift = (
        0.05 * deepwarm_pull
        + 0.04 * strat_pull
        + 0.02 * smooth_pull
        - 0.03 * mixing_pull
    )

    base_controls['memory_blend'] = float(np.clip(base_controls['memory_blend'] + memory_shift, 0.68, 0.92))
    base_controls['surface_relaxation'] = float(np.clip(base_controls['surface_relaxation'] + surface_shift, 0.06, 0.24))
    base_controls['surface_decay_depth'] = float(
        np.clip(base_controls['surface_decay_depth'] + surface_decay_shift, 2.0, 6.0)
    )
    base_controls['deep_inertia'] = float(np.clip(base_controls['deep_inertia'] + deep_inertia_shift, 0.40, 0.90))
    base_controls['deep_anchor'] = float(np.clip(base_controls['deep_anchor'] + deep_anchor_shift, 0.02, 0.22))
    base_controls['surface_skin_cooling_coef'] = float(
        np.clip(base_controls['surface_skin_cooling_coef'] + skin_cooling_shift, 0.008, 0.045)
    )
    return base_controls


def apply_online_ppo_action(control_params, kalman_scales, action):
    action = np.asarray(action, dtype=np.float32)[:PPO_ONLINE_ACTION_DIM]
    updated_controls = dict(control_params)
    updated_scales = normalize_kalman_scales(kalman_scales)
    action_len = int(len(action))

    updated_controls['memory_blend'] = update_control_value(
        max(control_params['memory_blend'], 0.72),
        action[0],
        0.72,
        0.88,
        step_size=0.06,
    )
    updated_controls['surface_relaxation'] = update_control_value(
        max(control_params['surface_relaxation'], 0.08),
        action[1],
        0.08,
        0.22,
        step_size=0.08,
    )
    updated_controls['deep_inertia'] = update_control_value(
        max(control_params['deep_inertia'], 0.45),
        action[2],
        0.45,
        0.82,
        step_size=0.07,
    )
    updated_controls['deep_anchor'] = update_control_value(
        max(control_params['deep_anchor'], 0.02),
        action[3],
        0.02,
        0.10,
        step_size=0.05,
    )
    updated_controls['surface_skin_cooling_coef'] = update_control_value(
        max(control_params.get('surface_skin_cooling_coef', SURFACE_SKIN_COOLING_COEF), 0.008),
        action[4],
        0.008,
        0.045,
        step_size=0.08,
    )
    if action_len >= 10:
        updated_controls['surface_decay_depth'] = update_control_value(
            max(control_params.get('surface_decay_depth', 4.0), 2.0),
            action[5],
            2.0,
            6.0,
            step_size=0.08,
        )
        process_action = action[6]
        obs_action = action[7]
        correlation_action = action[8]
        forecast_blend_action = action[9]
    elif action_len >= 8:
        updated_controls['surface_decay_depth'] = update_control_value(
            max(control_params.get('surface_decay_depth', 4.0), 2.0),
            action[5],
            2.0,
            6.0,
            step_size=0.08,
        )
        process_action = action[6]
        obs_action = action[7]
        correlation_action = 0.0
        forecast_blend_action = 0.0
    else:
        updated_controls['surface_decay_depth'] = float(np.clip(control_params.get('surface_decay_depth', 4.0), 2.0, 6.0))
        process_action = action[5] if action_len >= 6 else 0.0
        obs_action = action[6] if action_len >= 7 else 0.0
        correlation_action = 0.0
        forecast_blend_action = 0.0

    updated_scales['process'] = update_control_value(updated_scales['process'], process_action, 0.6, 2.0, step_size=0.06)
    updated_scales['obs'] = update_control_value(updated_scales['obs'], obs_action, 0.8, 3.0, step_size=0.06)
    updated_scales['correlation_length'] = update_control_value(
        updated_scales['correlation_length'],
        correlation_action,
        0.6,
        8.0,
        step_size=0.08,
    )
    updated_scales['forecast_blend'] = update_control_value(
        updated_scales['forecast_blend'],
        forecast_blend_action,
        0.02,
        0.85,
        step_size=0.08,
    )

    return updated_controls, updated_scales
