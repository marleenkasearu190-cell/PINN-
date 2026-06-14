"""Compatibility imports for the old state_forecaster module name.

Multi-lake reconstruction is now the public training path.  Core model classes
live in state_model.py, and reconstruction helpers live in state_reconstruction.py.
"""

from .state_model import (
    FORCING_FEATURE_COLUMNS,
    FORCING_FEATURE_INDEX,
    HARD_DENSITY_STABILITY_MODES,
    STATIC_FEATURE_DIM,
    STATIC_FEATURE_KEYS,
    ForcingHistoryEncoder,
    LakeStateForecaster,
    PhysicalScaleHead,
    StateParameterNet,
    heat_conserving_convective_adjustment,
    ice_conductive_flux_wm2,
    remove_area_weighted_mean,
    resolve_hard_density_stability,
    static_feature_array,
)
from .state_reconstruction import (
    _build_rollout_pairs,
    _forcing_tensor_rows,
    _huber_profile_loss,
    _profile_lookup,
    _profile_physics_loss,
    apply_lst_surface_assimilation,
    build_lst_profile_prior,
    initialize_rollout_state,
)

__all__ = [
    'FORCING_FEATURE_COLUMNS',
    'FORCING_FEATURE_INDEX',
    'HARD_DENSITY_STABILITY_MODES',
    'STATIC_FEATURE_DIM',
    'STATIC_FEATURE_KEYS',
    'ForcingHistoryEncoder',
    'LakeStateForecaster',
    'PhysicalScaleHead',
    'StateParameterNet',
    '_build_rollout_pairs',
    '_forcing_tensor_rows',
    '_huber_profile_loss',
    '_profile_lookup',
    '_profile_physics_loss',
    'apply_lst_surface_assimilation',
    'build_lst_profile_prior',
    'heat_conserving_convective_adjustment',
    'ice_conductive_flux_wm2',
    'initialize_rollout_state',
    'remove_area_weighted_mean',
    'resolve_hard_density_stability',
    'static_feature_array',
]


def main(argv=None):
    raise SystemExit(
        "The single-lake state_forecaster CLI has been removed. "
        "Use `python -m lake_pinn.state_multilake --manifest ... --output-dir ...`."
    )


if __name__ == '__main__':
    main()
