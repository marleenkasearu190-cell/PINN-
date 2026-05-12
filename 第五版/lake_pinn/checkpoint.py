# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .model import LakePINN
from .ppo import build_ppo_controller_from_bundle, normalize_kalman_scales

def save_ppo_policy_bundle(bundle, output_path):
    if bundle is None or output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return output_path


def export_model_checkpoint_bundle(model, training_info):
    if model is None:
        return None
    info = training_info or {}
    return {
        'model_class': 'LakePINN',
        'input_dim': int(getattr(model, 'input_dim', 2)),
        'hidden_dim': int(getattr(model, 'hidden_dim', 128)),
        'hidden_layers': int(getattr(model, 'hidden_layers', 8)),
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': info.get('optimizer_state_dict'),
        'scheduler_state_dict': info.get('scheduler_state_dict'),
        'training_info': {
            'final_weights': dict(info.get('final_weights', {})),
            'kalman_scales': normalize_kalman_scales(info.get('kalman_scales', {})),
            'surface_correction_info': info.get('surface_correction_info'),
            'best_selection_metric': info.get('best_selection_metric'),
            'best_selection_label': info.get('best_selection_label'),
            'ppo_policy_bundle': info.get('ppo_policy_bundle'),
        },
    }


def save_model_checkpoint_bundle(model, training_info, output_path):
    if model is None or output_path is None:
        return None
    bundle = export_model_checkpoint_bundle(model, training_info)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return output_path


def load_ppo_policy_bundle(policy_path, device='cpu'):
    if policy_path is None:
        return None, None
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f'PPO policy file not found: {policy_path}')
    bundle = torch.load(policy_path, map_location=device)
    return build_ppo_controller_from_bundle(bundle, device=device)


def load_model_checkpoint_bundle(checkpoint_path, device='cpu'):
    if checkpoint_path is None:
        return None, None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Model checkpoint file not found: {checkpoint_path}')
    bundle = torch.load(checkpoint_path, map_location=device)
    model_state = bundle['model_state_dict']
    inferred_input_dim = int(bundle.get('input_dim', model_state.get('net.0.weight').shape[1]))
    hidden_dim = int(bundle.get('hidden_dim', 128))
    hidden_layers = int(bundle.get('hidden_layers', 8))
    model = LakePINN(input_dim=inferred_input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, bundle


def checkpoint_has_embedded_ppo_policy(checkpoint_path):
    if checkpoint_path is None:
        return False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Model checkpoint file not found: {checkpoint_path}')
    bundle = torch.load(checkpoint_path, map_location='cpu')
    training_info = dict((bundle or {}).get('training_info', {}) or {})
    return training_info.get('ppo_policy_bundle') is not None
