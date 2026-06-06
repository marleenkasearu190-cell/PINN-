"""Public API for the reconstruction-state LakePINN package.

The seventh-version public surface intentionally points at the state-space
forecaster:

    T(t + dt, z) = M(T(t, z), forcing[t:t+dt], lake_attrs)

Legacy direct reconstruction code is archived in earlier versions; this package
exports only the reconstruction-state training, rollout, plotting, and scoring tools.
"""

from .data_io import (
    build_depth_grid,
    load_optional_profile_observations,
    load_training_frame,
    split_profile_observations,
)
from .conditional_priors import (
    infer_bottom_temp_prior_c,
    infer_freezing_lst_prior,
    infer_ice_risk_prior,
    infer_thermal_regime,
)
from .export import export_temperature_tables
from .hypsometry import fallback_area_profile, torch_area_profile
from .lake_metadata import infer_metadata, metadata_static_features
from .physics import (
    compute_surface_flux_terms,
    normalize_turbulent_flux_mode,
    water_density_numpy,
    water_density_torch,
)
from .plotting import plot_year_heatmap
from .scorecard_integration import (
    generate_prediction_diagnostic_figures,
    score_prediction_candidates,
)
from .state_model import (
    ForcingHistoryEncoder,
    LakeStateForecaster,
    StateParameterNet,
    heat_conserving_convective_adjustment,
    ice_conductive_flux_wm2,
    remove_area_weighted_mean,
    resolve_hard_density_stability,
    static_feature_array,
)
from .state_multilake import main, train_multilake_state_forecaster
from .state_reconstruction import (
    apply_lst_surface_assimilation,
    build_lst_profile_prior,
    initialize_rollout_state,
)
from .vertical_solver import (
    build_area_weighted_diffusion_matrix,
    implicit_diffusion_step,
    layer_thicknesses,
    one_day_heat_sources,
)

__all__ = [
    'LakeStateForecaster',
    'ForcingHistoryEncoder',
    'StateParameterNet',
    'apply_lst_surface_assimilation',
    'build_area_weighted_diffusion_matrix',
    'build_depth_grid',
    'build_lst_profile_prior',
    'compute_surface_flux_terms',
    'export_temperature_tables',
    'fallback_area_profile',
    'generate_prediction_diagnostic_figures',
    'heat_conserving_convective_adjustment',
    'implicit_diffusion_step',
    'infer_bottom_temp_prior_c',
    'infer_freezing_lst_prior',
    'infer_ice_risk_prior',
    'infer_metadata',
    'infer_thermal_regime',
    'initialize_rollout_state',
    'ice_conductive_flux_wm2',
    'layer_thicknesses',
    'load_optional_profile_observations',
    'load_training_frame',
    'main',
    'metadata_static_features',
    'normalize_turbulent_flux_mode',
    'one_day_heat_sources',
    'plot_year_heatmap',
    'remove_area_weighted_mean',
    'resolve_hard_density_stability',
    'score_prediction_candidates',
    'split_profile_observations',
    'static_feature_array',
    'torch_area_profile',
    'train_multilake_state_forecaster',
    'water_density_numpy',
    'water_density_torch',
]
