"""Constants for the modular run9 LakePINN implementation."""

import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ERA5_PATH = str(PROJECT_DIR / 'ERA5_mendota_2018_Daily.csv')
DEFAULT_LST_PATH = str(PROJECT_DIR / 'Lake-Mendota-MOD11A1-061-results.csv')

WATER_DENSITY = 1000.0
WATER_HEAT_CAPACITY = 4186.0
RHO_CP = WATER_DENSITY * WATER_HEAT_CAPACITY
MOLECULAR_DIFFUSIVITY = 1.4e-7
MIN_EDDY_DIFFUSIVITY = 1.0e-7
MIN_TOTAL_DIFFUSIVITY = MOLECULAR_DIFFUSIVITY + MIN_EDDY_DIFFUSIVITY
MAX_TOTAL_DIFFUSIVITY = 1.0e-3
GRAVITY = 9.81
SECONDS_PER_DAY = 86400.0

DIFFUSIVITY_K0 = 1.0e-5
DIFFUSIVITY_RI_SENSITIVITY = 5.0
DIFFUSIVITY_ALPHA = 1.0
DIFFUSIVITY_WIND_COEFF = 0.0012
DIFFUSIVITY_WIND_EXPONENT = 1.5
DIFFUSIVITY_WIND_DECAY_DEPTH = 5.0
DIFFUSIVITY_UNSTABLE_BOOST = 50.0
DIFFUSIVITY_UNSTABLE_GRADIENT_EPS = 5.0e-5
DIFFUSIVITY_UNSTABLE_GRADIENT_WIDTH = 1.0e-5
RI_WIND_SHEAR_FACTOR = 0.1

SURFACE_ALBEDO_WATER = 0.06
SURFACE_ALBEDO_ICE = 0.60
ATMOSPHERIC_EMISSIVITY = 0.90
WATER_EMISSIVITY = 0.97
STEFAN_BOLTZMANN = 5.67e-8
AIR_HEAT_CAPACITY = 1005.0
LATENT_HEAT_VAPORIZATION = 2.5e6
TRANSFER_COEFF_HEAT = 1.3e-3
TRANSFER_COEFF_MOISTURE = 1.3e-3
SHORTWAVE_SURFACE_FRACTION = 0.45
SHORTWAVE_ATTENUATION = 0.2
MIN_SHORTWAVE_ATTENUATION = 0.05
MAX_SHORTWAVE_ATTENUATION = 1.5
ICE_TRANSITION_EPS = 0.1
SURFACE_SKIN_COOLING_COEF = 0.012
SURFACE_AIR_BLEND = 0.18
STRATIFICATION_MARGIN_C = 1.0
MAX_VERTICAL_GRADIENT_C_PER_M = 5.0
DEEP_WARMING_ALLOWANCE_C_PER_DAY = 0.08
MAX_ENTRAINMENT_VELOCITY_M_PER_DAY = 1.0
AUTUMN_OVERTURN_TARGET_GAP_COLLAPSE_C = 0.45
AUTUMN_OVERTURN_DEEP_WARM_ALLOWANCE_C_PER_DAY = 0.04
SURFACE_MIXED_LAYER_MAX_COOLING_C_PER_DAY = 1.10
BOTTOM_SLOW_CHANGE_START_FRACTION = 0.78
BOTTOM_SLOW_CHANGE_TRANSITION_WIDTH_M = 0.8
BOTTOM_SLOW_CHANGE_MAX_C_PER_DAY = 0.08
BOTTOM_SLOW_CHANGE_AUTUMN_EXTRA_C_PER_DAY = 0.08
PPO_STATE_EPS = 1e-8
PROFILE_SPLIT_ROLES = ('train', 'val')
DEFAULT_PROFILE_SPLIT_PATTERN = ('train', 'train', 'train', 'train', 'val')
TIME_BLOCK_SPLIT_FRACTIONS = {
    'train': 0.80,
    'val': 0.20,
}
DEFAULT_INITIAL_WATER_TEMPERATURE_C = 4.0
PPO_WEIGHT_STATE_KEYS = (
    'pde',
    'bc',
    'ic',
    'obs',
    'time_continuity',
    'stratification',
    'smoothness',
    'deep_warming',
    'deep_anchor',
    'vertical_exchange',
    'convective_mixing',
    'autumn_overturn',
    'heat_budget',
)
PPO_TRAIN_ACTION_KEYS = PPO_WEIGHT_STATE_KEYS + (
    'kalman_process_scale',
    'kalman_obs_scale',
    'kalman_correlation_length',
    'kalman_forecast_blend',
)
PPO_TRAIN_ACTION_DIM = len(PPO_TRAIN_ACTION_KEYS)
PPO_ONLINE_ACTION_DIM = 10
PPO_STATE_DIM = 46
PINN_LEGACY_INPUT_DIM = 11
PINN_EXTENDED_FORCING_INPUT_DIM = 17
PINN_HISTORY_FORCING_INPUT_DIM = 24
PINN_STATE_MEMORY_INPUT_DIM = 27
PINN_FULL_STATIC_INPUT_DIM = 32
PINN_HYPSOMETRY_HYDRO_INPUT_DIM = 37
PINN_INPUT_DIM = PINN_HYPSOMETRY_HYDRO_INPUT_DIM
PINN_MAX_DEPTH_REFERENCE_M = 50.0
PINN_MAX_MEAN_DEPTH_REFERENCE_M = 50.0
PINN_VOLUME_REFERENCE_KM3 = 100.0
PINN_ELEVATION_REFERENCE_M = 1000.0
PINN_LIGHT_EXTINCTION_REFERENCE_M_INV = 3.0
PINN_FETCH_REFERENCE_M = 50000.0
PINN_RESIDENCE_TIME_REFERENCE_DAYS = 365.0
PINN_SHORELINE_LENGTH_REFERENCE_KM = 500.0
PINN_SHORELINE_DEVELOPMENT_REFERENCE = 10.0
PINN_CATCHMENT_AREA_REFERENCE_KM2 = 10000.0
PINN_WATER_LEVEL_ANOMALY_REFERENCE_M = 10.0
PINN_INFLOW_REFERENCE_M3_S = 1000.0
PINN_MAX_WIND_REFERENCE_M_PER_S = 15.0
PINN_MAX_SHORTWAVE_REFERENCE_W_M2 = 450.0
PINN_MAX_LONGWAVE_REFERENCE_W_M2 = 500.0
PINN_MAX_HEAT_FLUX_REFERENCE_W_M2 = 250.0
PINN_MAX_TEMPERATURE_REFERENCE_C = 30.0
PINN_MAX_SECCHI_REFERENCE_M = 20.0
PINN_SHORTWAVE_SUM_7D_REFERENCE = PINN_MAX_SHORTWAVE_REFERENCE_W_M2 * 7.0
PINN_SHORTWAVE_SUM_30D_REFERENCE = PINN_MAX_SHORTWAVE_REFERENCE_W_M2 * 30.0
PINN_HEATING_DEGREE_DAYS_30D_REFERENCE = 30.0 * 30.0
PINN_LOG_AREA_REFERENCE_KM2 = math.log1p(1000.0)
HISTORY_FORCING_COLUMNS = (
    'air_temp_mean_7d',
    'air_temp_mean_30d',
    'shortwave_sum_7d',
    'shortwave_sum_30d',
    'wind_mean_7d',
    'lst_mean_7d',
    'heating_degree_days_30d',
)
STATE_MEMORY_COLUMNS = (
    'prev_surface_temp',
    'prev_0_3m_mean',
    'prev_deep_mean',
)
OPTIONAL_HYDRO_FEATURE_COLUMNS = (
    'water_level_anomaly',
    'light_extinction_kd',
    'effective_fetch',
    'net_inflow',
)
KNOWN_LAKE_ATTRIBUTES = {
    'mohonk': {
        'area_km2': 0.069,
        'latitude': 41.766,
        'longitude': -74.158,
        'mean_depth_m': 9.7,
        'max_depth_m': 18.5,
        'volume_km3': 0.00066,
        'elevation_m': 379.0,
        'secchi_m': 3.81,
        'light_extinction_kd': 0.446,
        'fetch_m': 800.0,
        'effective_fetch_m': 800.0,
        'basin_shape_factor': 0.517,
        'mean_water_level_m': 379.0,
    },
    'mohonk lake': {
        'area_km2': 0.069,
        'latitude': 41.766,
        'longitude': -74.158,
        'mean_depth_m': 9.7,
        'max_depth_m': 18.5,
        'volume_km3': 0.00066,
        'elevation_m': 379.0,
        'secchi_m': 3.81,
        'light_extinction_kd': 0.446,
        'fetch_m': 800.0,
        'effective_fetch_m': 800.0,
        'basin_shape_factor': 0.517,
        'mean_water_level_m': 379.0,
    },
    'mendota': {'area_km2': 39.40, 'latitude': 43.10, 'max_depth_m': 25.0},
    'lake mendota': {'area_km2': 39.40, 'latitude': 43.10, 'max_depth_m': 25.0},
    'kinneret': {
        'area_km2': 170.0,
        'latitude': 32.82,
        'longitude': 35.59,
        'mean_depth_m': 24.0,
        'max_depth_m': 43.0,
        'volume_km3': 4.0,
        'elevation_m': -210.0,
        'secchi_m': 3.0,
        'light_extinction_kd': 0.57,
        'fetch_m': 22000.0,
        'effective_fetch_m': 22000.0,
        'basin_shape_factor': 0.5472,
        'mean_water_level_m': -210.0,
    },
    'lake kinneret': {
        'area_km2': 170.0,
        'latitude': 32.82,
        'longitude': 35.59,
        'mean_depth_m': 24.0,
        'max_depth_m': 43.0,
        'volume_km3': 4.0,
        'elevation_m': -210.0,
        'secchi_m': 3.0,
        'light_extinction_kd': 0.57,
        'fetch_m': 22000.0,
        'effective_fetch_m': 22000.0,
        'basin_shape_factor': 0.5472,
        'mean_water_level_m': -210.0,
    },
    'sea of galilee': {
        'area_km2': 170.0,
        'latitude': 32.82,
        'longitude': 35.59,
        'mean_depth_m': 24.0,
        'max_depth_m': 43.0,
        'volume_km3': 4.0,
        'elevation_m': -210.0,
        'secchi_m': 3.0,
        'light_extinction_kd': 0.57,
        'fetch_m': 22000.0,
        'effective_fetch_m': 22000.0,
        'basin_shape_factor': 0.5472,
        'mean_water_level_m': -210.0,
    },
}
