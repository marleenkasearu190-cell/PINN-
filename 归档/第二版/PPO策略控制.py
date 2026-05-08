"""
PDF-aligned lake temperature PINN baseline.

This stage focuses on the physics-informed core before adding Kalman filtering
and PPO-based adaptive weighting. The main loss follows the document structure:

    L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * L_ic + lambda_obs * L_obs

Implemented pieces:
1. Heat diffusion PDE with depth-dependent diffusivity.
2. Surface energy balance (SEB) boundary with smooth ice switching.
3. Bottom zero-flux boundary.
4. Initial-condition loss.
5. Observation loss using surface/bottom daily observations and optional profile data.
"""

import argparse
import calendar
import copy
import ctypes
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ERA5_PATH = str(PROJECT_DIR / 'ERA5_mendota_2018_Daily.csv')
DEFAULT_LST_PATH = str(PROJECT_DIR / 'Lake-Mendota-MOD11A1-061-results.csv')

WATER_DENSITY = 1000.0
WATER_HEAT_CAPACITY = 4186.0
RHO_CP = WATER_DENSITY * WATER_HEAT_CAPACITY
MOLECULAR_DIFFUSIVITY = 1.4e-7
MIN_EDDY_DIFFUSIVITY = 1.0e-7
MIN_TOTAL_DIFFUSIVITY = MOLECULAR_DIFFUSIVITY + MIN_EDDY_DIFFUSIVITY
GRAVITY = 9.81
SECONDS_PER_DAY = 86400.0

# PDF-aligned diffusivity parameters
DIFFUSIVITY_K0 = 1.0e-5
DIFFUSIVITY_RI_SENSITIVITY = 5.0
DIFFUSIVITY_ALPHA = 1.0
DIFFUSIVITY_WIND_COEFF = 0.0012
DIFFUSIVITY_WIND_EXPONENT = 1.5
DIFFUSIVITY_WIND_DECAY_DEPTH = 5.0
RI_WIND_SHEAR_FACTOR = 0.1

# Surface energy balance parameters
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
PPO_STATE_EPS = 1e-8
PROFILE_SPLIT_ROLES = ('train', 'val', 'assim', 'test')
DEFAULT_PROFILE_SPLIT_PATTERN = ('train', 'train', 'val', 'assim', 'test')
TIME_BLOCK_SPLIT_FRACTIONS = {
    'train': 0.60,
    'val': 0.15,
    'assim': 0.15,
    'test': 0.10,
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
PPO_TRAIN_ACTION_KEYS = PPO_WEIGHT_STATE_KEYS + ('kalman_process_scale', 'kalman_obs_scale')
PPO_TRAIN_ACTION_DIM = len(PPO_TRAIN_ACTION_KEYS)
PPO_ONLINE_ACTION_DIM = 7
PPO_STATE_DIM = 46


class LakePINN(nn.Module):
    """Lake temperature PINN with T(z, t) -> temperature in degree Celsius."""

    def __init__(self, hidden_dim=128, hidden_layers=8):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
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


def sanitize_name(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def infer_output_dir(era5_path: Path, lst_path: Path) -> Path:
    return PROJECT_DIR


def normalize_input_path(raw_value: str) -> Path:
    cleaned = str(raw_value).strip()
    if cleaned.startswith('& '):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.strip('"').strip("'")
    return Path(cleaned).expanduser()


def enable_high_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def ask_path_in_terminal(label: str, optional: bool = False) -> Path | None:
    while True:
        prompt = f'请输入 {label} 文件路径'
        if optional:
            prompt += '（可直接回车跳过）'
        prompt += ': '
        raw_value = input(prompt).strip()

        if not raw_value:
            if optional:
                return None
            print(f'{label} 文件路径不能为空，请重新输入。')
            continue

        path = normalize_input_path(raw_value)
        if path.exists() and path.is_file():
            return path

        print(f'{label} 文件不存在: {path}')
        print('请检查路径后重新输入。')


def prompt_for_existing_path(label: str, optional: bool = False) -> Path | None:
    try:
        enable_high_dpi()
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.tk.call('tk', 'scaling', 1.0)
        file_path = filedialog.askopenfilename(title=f'请选择 {label} 文件')
        root.destroy()
    except Exception:
        file_path = ''

    if file_path:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f'已选择 {label} 文件: {path}')
            return path

    if optional:
        fallback = input(f'未选择 {label} 文件，是否在终端手动输入路径？[y/N]: ').strip().lower()
        if fallback not in {'y', 'yes'}:
            return None

    print(f'未通过弹窗选择 {label} 文件，切换为终端输入。')
    return ask_path_in_terminal(label, optional=optional)


def prompt_for_output_dir(default_dir: Path) -> Path:
    try:
        enable_high_dpi()
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.tk.call('tk', 'scaling', 1.0)
        selected_dir = filedialog.askdirectory(
            title='请选择输出文件夹',
            initialdir=str(default_dir.resolve()),
            mustexist=False,
        )
        root.destroy()
    except Exception:
        selected_dir = ''

    if selected_dir:
        output_dir = Path(selected_dir).expanduser().resolve()
        print(f'已选择输出文件夹: {output_dir}')
        return output_dir

    fallback = input('未通过弹窗选择输出文件夹，是否在终端手动输入保存目录？[y/N]: ').strip().lower()
    if fallback not in {'y', 'yes'}:
        return default_dir.resolve()

    raw_value = input(f'请输入保存目录，直接回车使用默认目录 [{default_dir}]: ').strip()
    if not raw_value:
        return default_dir.resolve()
    return normalize_input_path(raw_value).resolve()


def prompt_text_value(prompt: str, default_value: str) -> str:
    raw_value = input(f'{prompt} [{default_value}]: ').strip()
    if not raw_value:
        return default_value
    return raw_value


def prompt_choice_value(prompt: str, choices, default_value: str) -> str:
    normalized_choices = [str(choice) for choice in choices]
    lowered_lookup = {choice.lower(): choice for choice in normalized_choices}
    options_text = '/'.join(normalized_choices)

    while True:
        raw_value = input(f'{prompt} ({options_text}) [{default_value}]: ').strip()
        if not raw_value:
            return default_value
        lowered = raw_value.lower()
        if lowered in lowered_lookup:
            return lowered_lookup[lowered]
        print(f'Please enter one of: {options_text}')


def prompt_yes_no_value(prompt: str, default_value: bool) -> bool:
    default_text = 'Y/n' if default_value else 'y/N'
    while True:
        raw_value = input(f'{prompt} [{default_text}]: ').strip().lower()
        if not raw_value:
            return default_value
        if raw_value in {'y', 'yes'}:
            return True
        if raw_value in {'n', 'no'}:
            return False
        print('Please enter y or n.')


def prompt_int_value(prompt: str, default_value: int, minimum: int = 1) -> int:
    while True:
        raw_value = input(f'{prompt} [{default_value}]: ').strip()
        if not raw_value:
            return default_value
        try:
            parsed = int(raw_value)
        except ValueError:
            print(f'Please enter an integer >= {minimum}.')
            continue
        if parsed >= minimum:
            return parsed
        print(f'Please enter an integer >= {minimum}.')


def prompt_float_value(prompt: str, default_value: float, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        raw_value = input(f'{prompt} [{default_value}]: ').strip()
        if not raw_value:
            return default_value
        try:
            parsed = float(raw_value)
        except ValueError:
            print('Please enter a valid number.')
            continue
        if minimum is not None and parsed < minimum:
            print(f'Please enter a value >= {minimum}.')
            continue
        if maximum is not None and parsed > maximum:
            print(f'Please enter a value <= {maximum}.')
            continue
        return parsed


def prompt_path_value(prompt: str, optional: bool = False) -> Path | None:
    return prompt_for_existing_path(prompt, optional=optional)


def apply_practical_prediction_defaults(args: argparse.Namespace, has_profile_obs: bool) -> argparse.Namespace:
    args.practical_prediction_mode = True
    args.profile_split_mode = 'time_blocked' if has_profile_obs else 'none'
    args.seasonal_segmented = False
    args.use_bottom_observation = False
    args.initial_condition_mode = 'uniform_4c'
    args.use_kalman = True
    args.use_ppo = False
    args.kalman_forecast_spinup_days = 14
    args.kalman_forecast_spinup_max_blend = 0.95
    args.kalman_obs_std_surface = 2.5
    args.autumn_asymmetric_cooling = False
    args.autumn_doy_threshold = 270.0
    args.autumn_surface_cooling_threshold = 1.0
    args.autumn_air_temp_threshold = 12.0
    args.autumn_cooling_strength = 0.35
    args.autumn_cooling_penetration_scale = 5.0
    args.ppo_use_kalman_reward = False
    args.online_ppo_update = False
    args.shallow_optimized_grid = False
    args.surface_bulk_correction = False
    args.apply_post_physics = False
    args.rolling_prediction_mode = True
    args.rolling_memory_blend = 0.85
    args.rolling_surface_relaxation = 0.08
    args.rolling_surface_decay_depth = 4.0
    args.solar_shading_factor = 0.65
    args.shortwave_attenuation_coef = 1.5
    args.shortwave_surface_fraction = 0.72
    args.surface_skin_cooling_coef = 0.02
    args.surface_air_blend = 0.25
    args.stratification_weight = 0.8
    args.stratification_pairs = 64
    args.stratification_margin_c = 1.2
    args.smoothness_weight = 0.25
    args.max_vertical_gradient_c_per_m = 5.0
    args.deep_warming_weight = 0.45
    args.rolling_deep_inertia = 0.72
    args.deep_anchor_weight = 0.15
    args.deep_anchor_pairs = 64
    args.deep_anchor_amplitude_c = 2.5
    args.rolling_deep_anchor = 0.08
    args.vertical_exchange_weight = 0.22
    args.entrainment_velocity_scale_m_per_day = 0.80
    args.convective_mixing_weight = 0.18
    args.autumn_overturn_weight = 0.18
    args.heat_budget_weight = 0.20
    args.heat_budget_depth_points = 24
    return args


def apply_train_mode_defaults(args: argparse.Namespace, has_profile_obs: bool) -> argparse.Namespace:
    args.mode = 'train'
    args.practical_prediction_mode = False
    args.profile_split_mode = 'depth_interleaved' if has_profile_obs else 'none'
    args.seasonal_segmented = False
    args.use_bottom_observation = False
    args.initial_condition_mode = 'uniform_4c'
    args.use_kalman = True
    args.use_ppo = bool(has_profile_obs)
    args.kalman_forecast_spinup_days = 0
    args.kalman_forecast_spinup_max_blend = 0.9
    args.kalman_obs_std_surface = 0.5
    args.autumn_asymmetric_cooling = False
    args.autumn_doy_threshold = 270.0
    args.autumn_surface_cooling_threshold = 1.0
    args.autumn_air_temp_threshold = 12.0
    args.autumn_cooling_strength = 0.35
    args.autumn_cooling_penetration_scale = 5.0
    args.ppo_use_kalman_reward = False
    args.shallow_optimized_grid = False
    args.surface_bulk_correction = False
    args.apply_post_physics = False
    args.rolling_prediction_mode = True
    args.rolling_memory_blend = 0.82
    args.rolling_surface_relaxation = 0.12
    args.rolling_surface_decay_depth = 4.0
    args.solar_shading_factor = 0.80
    args.shortwave_attenuation_coef = 1.0
    args.shortwave_surface_fraction = 0.62
    args.surface_skin_cooling_coef = 0.018
    args.surface_air_blend = 0.22
    args.time_continuity_weight = 0.0
    args.stratification_weight = 0.0
    args.stratification_pairs = 64
    args.stratification_margin_c = STRATIFICATION_MARGIN_C
    args.smoothness_weight = 0.08
    args.max_vertical_gradient_c_per_m = MAX_VERTICAL_GRADIENT_C_PER_M
    args.deep_warming_weight = 0.12
    args.rolling_deep_inertia = 0.70
    args.deep_anchor_weight = 0.0
    args.deep_anchor_pairs = 64
    args.deep_anchor_amplitude_c = 2.2
    args.rolling_deep_anchor = 0.0
    args.vertical_exchange_weight = 0.28
    args.entrainment_velocity_scale_m_per_day = 0.75
    args.convective_mixing_weight = 0.20
    args.autumn_overturn_weight = 0.24
    args.heat_budget_weight = 0.25
    args.heat_budget_depth_points = 24
    if args.train_until_best is None:
        args.train_until_best = True
    if args.train_min_epochs is None:
        args.train_min_epochs = 200
    if args.train_patience_windows is None:
        args.train_patience_windows = 6
    return args


def configure_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    print('=' * 72)
    print('Interactive PINN runner')
    print('Terminal quick mode: only ERA5, LST, and optional profile-obs are asked.')
    print('If you provide profile-obs, the script uses train mode.')
    print('If you skip profile-obs, the script uses predict mode.')
    print('=' * 72)

    era5_path = prompt_path_value('ERA5 CSV path')
    lst_path = prompt_path_value('LST CSV path')
    if era5_path is None or lst_path is None:
        raise ValueError('ERA5 and LST files are required.')

    args.era5 = str(era5_path.resolve())
    args.lst = str(lst_path.resolve())

    profile_obs_path = prompt_path_value('Profile observation CSV path (optional; press Enter to skip)', optional=True)
    args.profile_obs = str(profile_obs_path.resolve()) if profile_obs_path else None

    args.output_dir = str(prompt_for_output_dir(infer_output_dir(era5_path.resolve(), lst_path.resolve())).resolve())
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.epochs = 600
    args.collocation_points = 128
    args.depth_points = 80
    args.max_depth = 20.0
    args.ppo_control_interval = 50
    args.ppo_rollout_steps = 4
    args.ppo_eval_depth_points = 80
    if args.profile_obs:
        args.mode = 'train'
        args = apply_train_mode_defaults(args, has_profile_obs=True)
        resume_model_checkpoint_path = prompt_path_value('Resume PINN model checkpoint path (optional; press Enter to start from scratch)', optional=True)
        args.resume_model_checkpoint = str(resume_model_checkpoint_path.resolve()) if resume_model_checkpoint_path else None
        args.model_checkpoint_path = None
        args.ppo_policy_path = None
    else:
        args.mode = 'predict'
        args = apply_practical_prediction_defaults(args, has_profile_obs=False)
        model_checkpoint_path = prompt_path_value('PINN model checkpoint path')
        if model_checkpoint_path is None:
            raise ValueError('A PINN model checkpoint is required for predict mode.')
        args.model_checkpoint_path = str(model_checkpoint_path.resolve())
        args.resume_model_checkpoint = None
        ppo_policy_path = prompt_path_value('PPO policy path (optional; press Enter to skip)', optional=True)
        args.ppo_policy_path = str(ppo_policy_path.resolve()) if ppo_policy_path else None

    print(f"Mode selected: {args.mode}")
    print('Recommended defaults applied:')
    print(f'  output_dir={args.output_dir}')
    print(f'  device={args.device}')
    print('  epochs=600 | collocation_points=128 | depth_points=80 | max_depth=20.0')
    print(f'  profile_split_mode={args.profile_split_mode} | seasonal_segmented={args.seasonal_segmented}')
    print('  use_bottom_observation=False | initial_condition_mode=uniform_4c')
    print(f'  use_kalman={args.use_kalman} | use_ppo={args.use_ppo}')
    print('  kalman_forecast_spinup_days=14 | kalman_forecast_spinup_max_blend=0.95')
    print('  rolling_prediction_mode=True | rolling_memory_blend=0.85 | rolling_surface_relaxation=0.08')
    print('  solar_shading_factor=0.65 | shortwave_attenuation_coef=1.5 | kalman_obs_std_surface=2.5')
    return args


def infer_metadata(merged: pd.DataFrame, lst: pd.DataFrame, era5_path: Path, lst_path: Path):
    year = int(merged['Date'].dt.year.mode().iloc[0])

    lake_name = None
    for col in ['Category', 'ID']:
        if col in lst.columns:
            values = lst[col].dropna().astype(str)
            values = values[values.str.strip() != '']
            if col == 'ID':
                values = values[~values.str.fullmatch(r'\d+')]
            if not values.empty:
                lake_name = values.iloc[0].strip()
                break

    if not lake_name:
        lake_name = lst_path.stem.replace('-', ' ').replace('_', ' ')

    base_tag = sanitize_name(lake_name)
    if not base_tag:
        base_tag = sanitize_name(era5_path.stem.replace('-', ' ').replace('_', ' '))
    file_tag = f'{base_tag}_{year}'

    return {'lake_name': lake_name, 'year': year, 'file_tag': file_tag}


def season_label_for_month(month_value: int) -> str:
    if month_value in {12, 1, 2}:
        return 'winter'
    if month_value in {3, 4, 5}:
        return 'spring'
    if month_value in {6, 7, 8}:
        return 'summer'
    return 'autumn'


def build_contiguous_season_segments(df: pd.DataFrame):
    if df.empty:
        return []

    season_labels = df['month'].apply(lambda month_value: season_label_for_month(int(month_value))).tolist()
    segments = []
    season_counts = {}
    start_idx = 0

    for idx in range(1, len(df) + 1):
        is_boundary = idx == len(df) or season_labels[idx] != season_labels[start_idx]
        if not is_boundary:
            continue

        season_name = season_labels[start_idx]
        season_counts[season_name] = season_counts.get(season_name, 0) + 1
        occurrence = season_counts[season_name]
        segment_name = season_name if occurrence == 1 else f'{season_name}_{occurrence}'
        segments.append(
            {
                'name': segment_name,
                'season': season_name,
                'start_idx': start_idx,
                'end_idx': idx,
                'start_date': pd.Timestamp(df['Date'].iloc[start_idx]),
                'end_date': pd.Timestamp(df['Date'].iloc[idx - 1]),
            }
        )
        start_idx = idx

    return segments


def pick_numeric_series(frame: pd.DataFrame, candidates, default=np.nan):
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors='coerce')
            if not values.isna().all():
                return values
    return pd.Series(default, index=frame.index, dtype=np.float32)


def first_existing_column(frame: pd.DataFrame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def empty_profile_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'Depth_m', 'Temperature_C', 'time_norm'])


def has_profile_observations(profile_obs_data) -> bool:
    if profile_obs_data is None:
        return False
    if isinstance(profile_obs_data, (str, Path)):
        return True
    return not profile_obs_data.empty


def has_bottom_temperature_observations(df: pd.DataFrame) -> bool:
    return 'BottomTemp_C' in df.columns and not pd.to_numeric(df['BottomTemp_C'], errors='coerce').isna().all()


def saturation_vapor_pressure_np(temp_c):
    return 610.94 * np.exp((17.625 * temp_c) / (temp_c + 243.04))


def saturation_vapor_pressure_torch(temp_c: torch.Tensor) -> torch.Tensor:
    return 610.94 * torch.exp((17.625 * temp_c) / (temp_c + 243.04))


def specific_humidity_from_vapor_pressure_torch(vapor_pressure_pa: torch.Tensor, pressure_pa: torch.Tensor) -> torch.Tensor:
    return 0.622 * vapor_pressure_pa / (pressure_pa - 0.378 * vapor_pressure_pa).clamp(min=1.0)


def water_density_torch(temp_c: torch.Tensor) -> torch.Tensor:
    return 1000.0 * (1.0 - 6.63e-6 * (temp_c - 4.0) ** 2)


def water_density_numpy(temp_c) -> np.ndarray:
    temp_c = np.asarray(temp_c, dtype=np.float64)
    return 1000.0 * (1.0 - 6.63e-6 * (temp_c - 4.0) ** 2)


def project_temperature_profile_to_stable_density(temp_profile, max_iterations: int = 256):
    temp_profile = np.asarray(temp_profile, dtype=np.float64).copy()
    if temp_profile.size < 2:
        return temp_profile, 0

    adjustments = 0
    for _ in range(max_iterations):
        density_profile = water_density_numpy(temp_profile)
        unstable_idx = np.where(np.diff(density_profile) < -1e-9)[0]
        if unstable_idx.size == 0:
            break
        for idx in unstable_idx:
            mixed_temp = 0.5 * (temp_profile[idx] + temp_profile[idx + 1])
            temp_profile[idx] = mixed_temp
            temp_profile[idx + 1] = mixed_temp
            adjustments += 1
    return temp_profile, adjustments


def evaluate_blind_ppo_proxy(df, temp_grid, depths):
    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    if temp_grid.ndim != 2 or temp_grid.size == 0:
        return None

    if 'SurfaceBulkTarget_C' in df.columns:
        surface_target = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    else:
        surface_target = df['LST_surface_C'].to_numpy(dtype=np.float64)
    raw_lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64) if 'LST_surface_C' in df.columns else surface_target.copy()
    air_temp_series = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else surface_target.copy()
    wind_speed_series = (
        df['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
        if 'wind_speed_m_per_s' in df.columns
        else np.full(temp_grid.shape[1], 2.0, dtype=np.float64)
    )
    solar_series = (
        df['Solar_W_m2'].to_numpy(dtype=np.float64)
        if 'Solar_W_m2' in df.columns
        else np.zeros(temp_grid.shape[1], dtype=np.float64)
    )

    surface_pred = temp_grid[0, :]
    surface_errors = surface_pred - surface_target
    surface_rmse = float(np.sqrt(np.mean(surface_errors ** 2)))
    surface_mae = float(np.mean(np.abs(surface_errors)))
    surface_bias = float(np.mean(surface_errors))
    warm_surface_bias = float(np.mean(np.clip(surface_errors, 0.0, None)))

    lst_jumps = np.abs(np.diff(raw_lst_surface, prepend=raw_lst_surface[0]))
    skin_bulk_gap = np.abs(raw_lst_surface - surface_target)
    lst_spike_signal = np.clip((lst_jumps - 2.5) / 4.0, 0.0, 1.5)
    skin_gap_signal = np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
    low_wind_signal = np.exp(-np.clip(wind_speed_series, 0.0, 12.0) / 2.5)
    high_solar_signal = np.clip((solar_series - 120.0) / 250.0, 0.0, 1.0)
    air_gap_signal = np.clip((np.abs(raw_lst_surface - air_temp_series) - 4.0) / 8.0, 0.0, 1.0)
    lst_spike_indicator = float(
        np.mean(
            np.clip(
                lst_spike_signal * (0.45 + 0.55 * low_wind_signal)
                + 0.35 * skin_gap_signal
                + 0.20 * high_solar_signal
                + 0.15 * air_gap_signal,
                0.0,
                2.0,
            )
        )
    )

    density_grid = water_density_numpy(temp_grid)
    instability = np.clip(-(density_grid[1:, :] - density_grid[:-1, :]), 0.0, None)
    instability_penalty = float(np.mean(instability))

    warm_season_mask = df['T_air_C'].to_numpy(dtype=np.float64) > 9.0 if 'T_air_C' in df.columns else np.ones(temp_grid.shape[1], dtype=bool)
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(temp_grid.shape[1], 2.0, dtype=np.float64)
    deep_warm_penalties = []
    for day_idx in range(temp_grid.shape[1]):
        if not bool(warm_season_mask[day_idx]):
            continue
        day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
        deep_floor = max(day_mld + 2.5, depths[-1] * 0.45)
        anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
        deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
        deep_excess = np.maximum(temp_grid[:, day_idx] - deep_anchor_profile, 0.0)
        deep_warm_penalties.append(float(np.mean(anchor_mask * deep_excess)))
    deep_warm_penalty = float(np.mean(deep_warm_penalties)) if deep_warm_penalties else 0.0
    deep_warm_peak = float(np.max(deep_warm_penalties)) if deep_warm_penalties else 0.0

    dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else None
    if dates is not None:
        doy = dates.dt.dayofyear.to_numpy(dtype=np.int32)
    else:
        doy = np.arange(1, temp_grid.shape[1] + 1, dtype=np.int32)
    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else np.full(temp_grid.shape[1], 10.0, dtype=np.float64)
    max_depth = float(depths[-1]) if depths.size else 1.0
    deep_ref_depth = float(min(9.0, max(5.0, max_depth * 0.45)))
    deep_ref_temp = np.array(
        [np.interp(deep_ref_depth, depths, temp_grid[:, day_idx]) for day_idx in range(temp_grid.shape[1])],
        dtype=np.float64,
    )

    depth_mids = 0.5 * (depths[:-1] + depths[1:]) if depths.size > 1 else np.array([], dtype=np.float64)
    dz = np.diff(depths) if depths.size > 1 else np.array([], dtype=np.float64)
    gradients = np.diff(temp_grid, axis=0) / np.maximum(dz[:, None], 1e-6) if dz.size else np.zeros((0, temp_grid.shape[1]), dtype=np.float64)

    summer_mask = (doy >= 160) & (doy <= 250)
    summer_strat_penalty = 0.0
    summer_thermocline_depth_norm = 0.0
    summer_thermocline_thickness_penalty = 0.0
    summer_surface_warming_reward = 0.0
    summer_midlayer_temp_reward = 0.0
    if np.any(summer_mask):
        summer_gap = surface_pred[summer_mask] - deep_ref_temp[summer_mask]
        summer_strat_penalty = float(np.mean(np.clip(6.0 - summer_gap, 0.0, None) / 6.0))
        summer_surface_warming_reward = float(np.mean(np.clip((surface_pred[summer_mask] - 18.0) / 6.0, 0.0, 1.5)))
        mid_ref_depth = float(min(6.0, max(4.0, max_depth * 0.30)))
        mid_ref_temp = np.array(
            [np.interp(mid_ref_depth, depths, temp_grid[:, day_idx]) for day_idx in range(temp_grid.shape[1])],
            dtype=np.float64,
        )
        mid_temp_error = np.abs(mid_ref_temp[summer_mask] - 13.0)
        summer_midlayer_temp_reward = float(np.mean(np.clip(1.0 - mid_temp_error / 5.0, 0.0, 1.0)))
        thermo_band = (depth_mids >= 1.0) & (depth_mids <= min(12.0, max_depth - 0.5))
        if np.any(thermo_band):
            thermo_strength = np.clip(-gradients[thermo_band][:, summer_mask], 0.0, None)
            thermo_idx = np.argmax(thermo_strength, axis=0)
            thermo_depths = depth_mids[thermo_band][thermo_idx]
            summer_thermocline_depth_norm = float(np.mean(thermo_depths / max(max_depth, 1.0)))
            thermo_band_depths = depth_mids[thermo_band]
            thickness_penalties = []
            for col_idx in range(thermo_strength.shape[1]):
                weights = thermo_strength[:, col_idx]
                peak_strength = float(np.max(weights))
                if peak_strength <= 1.0e-6:
                    thickness_penalties.append(1.0)
                    continue
                norm_weights = weights / np.maximum(np.sum(weights), 1.0e-8)
                mean_depth = float(np.sum(thermo_band_depths * norm_weights))
                std_depth = float(np.sqrt(np.sum(((thermo_band_depths - mean_depth) ** 2) * norm_weights)))
                thickness_penalties.append(max(std_depth - 1.2, 0.0) / 1.2)
            if thickness_penalties:
                summer_thermocline_thickness_penalty = float(np.mean(thickness_penalties))
    summer_9m_temp = float(np.mean(deep_ref_temp[summer_mask])) if np.any(summer_mask) else float(np.mean(deep_ref_temp))
    summer_bottom_temp = (
        float(np.mean(temp_grid[-1, summer_mask])) if np.any(summer_mask) else float(np.mean(temp_grid[-1, :]))
    )

    autumn_mask = (doy >= 280) & (doy <= 330) & (air_temp <= 15.0)
    autumn_overturn_penalty = 0.0
    if np.any(autumn_mask):
        autumn_gap = np.abs(surface_pred[autumn_mask] - temp_grid[-1, autumn_mask])
        autumn_overturn_penalty = float(np.mean(np.clip(autumn_gap - 1.2, 0.0, None)))
    autumn_surface_cooling_rate = 0.0
    autumn_gap_collapse = 0.0
    autumn_false_overturn_penalty = 0.0
    autumn_cooling_triggered_overturn_reward = 0.0
    cooling_window = 7
    if temp_grid.shape[1] > cooling_window:
        gap_series = np.maximum(surface_pred - temp_grid[-1, :], 0.0)
        autumn_window_mask = autumn_mask[:-cooling_window] & autumn_mask[cooling_window:]
        if np.any(autumn_window_mask):
            cooling = np.maximum(surface_pred[:-cooling_window] - surface_pred[cooling_window:], 0.0)
            gap_collapse = np.maximum(gap_series[:-cooling_window] - gap_series[cooling_window:], 0.0)
            deep_warming = np.maximum(temp_grid[-1, cooling_window:] - temp_grid[-1, :-cooling_window], 0.0)
            cooling = cooling[autumn_window_mask]
            gap_collapse = gap_collapse[autumn_window_mask]
            deep_warming = deep_warming[autumn_window_mask]
            if cooling.size:
                autumn_surface_cooling_rate = float(np.mean(cooling))
                cooling_gate = np.clip(cooling / 0.5, 0.0, 1.0)
                autumn_gap_collapse = float(np.mean(gap_collapse * cooling_gate))
                false_collapse = np.maximum(gap_collapse - 1.25 * cooling, 0.0)
                false_warming = np.maximum(deep_warming - 0.08, 0.0)
                autumn_false_overturn_penalty = float(np.mean(false_collapse + 1.5 * false_warming))
                overturn_success = (
                    np.clip(cooling / 0.6, 0.0, 1.5)
                    * np.clip(gap_collapse / 0.8, 0.0, 1.5)
                    * np.exp(-2.5 * (false_collapse + false_warming))
                )
                autumn_cooling_triggered_overturn_reward = float(np.mean(overturn_success))

    winter_mask = ((doy <= 75) | (doy >= 335)) & (air_temp <= 6.0)
    winter_inverse_penalty = 0.0
    winter_bottom_4c_error = 0.0
    if np.any(winter_mask):
        bottom_temp = temp_grid[-1, winter_mask]
        surface_temp = surface_pred[winter_mask]
        inverse_gap = bottom_temp - surface_temp
        winter_bottom_4c_error = float(np.mean(np.abs(bottom_temp - 4.0)))
        winter_inverse_penalty = float(
            np.mean(np.clip(1.5 - inverse_gap, 0.0, None) / 1.5 + np.abs(bottom_temp - 4.0) / 4.0)
        )

    deep_smoothness_penalty = 0.0
    if depths.size >= 3:
        deep_mask = depths >= min(10.0, max_depth * 0.55)
        deep_indices = np.where(deep_mask)[0]
        if deep_indices.size >= 3:
            deep_profiles = temp_grid[deep_indices, :]
            second_derivative = np.diff(deep_profiles, n=2, axis=0)
            deep_smoothness_penalty = float(np.mean(np.abs(second_derivative)))

    proxy_rmse = (
        surface_rmse
        + 12.0 * instability_penalty
        + 4.8 * deep_warm_penalty
        + 2.8 * deep_warm_peak
        + 1.5 * warm_surface_bias
        + 2.5 * summer_strat_penalty
        + 3.6 * summer_thermocline_thickness_penalty
        + 0.55 * max(summer_9m_temp - 14.0, 0.0)
        + 0.40 * max(summer_bottom_temp - 7.5, 0.0)
        + 2.2 * autumn_overturn_penalty
        + 1.8 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 2.0 * max(0.50 - autumn_gap_collapse, 0.0)
        + 4.0 * autumn_false_overturn_penalty
        + 1.8 * winter_inverse_penalty
        + 0.8 * winter_bottom_4c_error
        + 1.5 * deep_smoothness_penalty
        + 0.8 * lst_spike_indicator
        - 1.6 * summer_surface_warming_reward
        - 1.2 * summer_midlayer_temp_reward
        - 1.6 * autumn_cooling_triggered_overturn_reward
    )
    proxy_mae = (
        surface_mae
        + 8.0 * instability_penalty
        + 4.0 * deep_warm_penalty
        + 2.2 * deep_warm_peak
        + 1.2 * warm_surface_bias
        + 2.0 * summer_strat_penalty
        + 2.8 * summer_thermocline_thickness_penalty
        + 0.45 * max(summer_9m_temp - 14.0, 0.0)
        + 0.32 * max(summer_bottom_temp - 7.5, 0.0)
        + 1.8 * autumn_overturn_penalty
        + 1.4 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 1.6 * max(0.50 - autumn_gap_collapse, 0.0)
        + 3.2 * autumn_false_overturn_penalty
        + 1.5 * winter_inverse_penalty
        + 0.6 * winter_bottom_4c_error
        + 1.2 * deep_smoothness_penalty
        + 0.6 * lst_spike_indicator
        - 1.2 * summer_surface_warming_reward
        - 0.9 * summer_midlayer_temp_reward
        - 1.2 * autumn_cooling_triggered_overturn_reward
    )
    proxy_bias = (
        surface_bias
        + 3.2 * deep_warm_penalty
        + 1.2 * warm_surface_bias
        + 2.2 * summer_thermocline_thickness_penalty
        + 0.45 * max(summer_9m_temp - 14.0, 0.0)
        + 0.32 * max(summer_bottom_temp - 7.5, 0.0)
        + 0.8 * autumn_overturn_penalty
        + 0.6 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 0.6 * max(0.50 - autumn_gap_collapse, 0.0)
        + 2.8 * autumn_false_overturn_penalty
        + 0.6 * winter_inverse_penalty
        + 0.5 * winter_bottom_4c_error
        - 0.8 * summer_surface_warming_reward
        - 0.6 * summer_midlayer_temp_reward
        - 0.8 * autumn_cooling_triggered_overturn_reward
    )
    return {
        'rmse': float(proxy_rmse),
        'mae': float(proxy_mae),
        'bias': float(proxy_bias),
        'surface_rmse': float(surface_rmse),
        'warm_surface_bias': float(warm_surface_bias),
        'instability_penalty': float(instability_penalty),
        'deep_warm_penalty': float(deep_warm_penalty),
        'deep_warm_peak': float(deep_warm_peak),
        'summer_stratification_penalty': float(summer_strat_penalty),
        'summer_thermocline_depth_norm': float(summer_thermocline_depth_norm),
        'summer_thermocline_thickness_penalty': float(summer_thermocline_thickness_penalty),
        'summer_surface_warming_reward': float(summer_surface_warming_reward),
        'summer_midlayer_temp_reward': float(summer_midlayer_temp_reward),
        'summer_9m_temp': float(summer_9m_temp),
        'summer_bottom_temp': float(summer_bottom_temp),
        'autumn_overturn_penalty': float(autumn_overturn_penalty),
        'autumn_surface_cooling_rate': float(autumn_surface_cooling_rate),
        'autumn_gap_collapse': float(autumn_gap_collapse),
        'autumn_false_overturn_penalty': float(autumn_false_overturn_penalty),
        'autumn_cooling_triggered_overturn_reward': float(autumn_cooling_triggered_overturn_reward),
        'winter_inverse_penalty': float(winter_inverse_penalty),
        'winter_bottom_4c_error': float(winter_bottom_4c_error),
        'deep_smoothness_penalty': float(deep_smoothness_penalty),
        'lst_spike_indicator': float(lst_spike_indicator),
    }


def apply_autumn_cooling_adjustment(*args, **kwargs):
    state_upd = np.asarray(kwargs.get('state_upd', args[1] if len(args) > 1 else None), dtype=np.float64)
    return state_upd, 0.0


def smooth_anneal(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def build_annealed_loss_weights(base_weights, progress: float):
    shape_weight_sum = (
        base_weights.get('time_continuity', 0.0)
        + base_weights.get('stratification', 0.0)
        + base_weights.get('smoothness', 0.0)
        + base_weights.get('deep_warming', 0.0)
        + base_weights.get('deep_anchor', 0.0)
        + base_weights.get('vertical_exchange', 0.0)
        + base_weights.get('convective_mixing', 0.0)
        + base_weights.get('autumn_overturn', 0.0)
        + base_weights.get('heat_budget', 0.0)
    )
    if shape_weight_sum <= 0.0:
        return dict(base_weights)
    anneal = smooth_anneal(progress)
    weights = dict(base_weights)
    weights['obs'] = base_weights['obs'] * (1.6 - 0.85 * anneal)
    weights['time_continuity'] = base_weights.get('time_continuity', 0.0) * (0.35 + 0.65 * anneal)
    weights['stratification'] = base_weights.get('stratification', 0.0) * (0.1 + 0.9 * anneal)
    weights['smoothness'] = base_weights.get('smoothness', 0.0) * (0.2 + 0.8 * anneal)
    weights['deep_warming'] = base_weights.get('deep_warming', 0.0) * (0.25 + 0.75 * anneal)
    weights['deep_anchor'] = base_weights.get('deep_anchor', 0.0) * (0.05 + 0.55 * anneal)
    weights['vertical_exchange'] = base_weights.get('vertical_exchange', 0.0) * (0.20 + 0.80 * anneal)
    weights['convective_mixing'] = base_weights.get('convective_mixing', 0.0) * (0.15 + 0.85 * anneal)
    weights['autumn_overturn'] = base_weights.get('autumn_overturn', 0.0) * (0.25 + 0.75 * anneal)
    weights['heat_budget'] = base_weights.get('heat_budget', 0.0) * (0.35 + 0.65 * anneal)
    return weights


def smooth_ice_indicator(temp_surface_c: torch.Tensor, transition_eps: float = ICE_TRANSITION_EPS) -> torch.Tensor:
    return torch.sigmoid((-temp_surface_c) / transition_eps)


def load_training_frame(era5_path, lst_path):
    era5_path = Path(era5_path)
    lst_path = Path(lst_path)

    if not era5_path.exists():
        raise FileNotFoundError(f'ERA5 file not found: {era5_path}')
    if not lst_path.exists():
        raise FileNotFoundError(f'LST file not found: {lst_path}')

    era5 = pd.read_csv(era5_path)
    era5['Date'] = pd.to_datetime(era5['Date'])
    era5 = era5.sort_values('Date').copy()

    lst = pd.read_csv(lst_path)
    lst['Date'] = pd.to_datetime(lst['Date'])
    lst = lst.sort_values('Date').copy()
    lst['LST_surface_K'] = pd.to_numeric(lst['MOD11A1_061_LST_Day_1km'], errors='coerce')
    lst.loc[lst['LST_surface_K'] <= 0, 'LST_surface_K'] = np.nan

    lst_daily = lst.groupby('Date', as_index=False)['LST_surface_K'].mean().sort_values('Date')

    merged = era5.merge(lst_daily, on='Date', how='left')
    merged = merged.sort_values('Date').copy()
    merged['day_index'] = np.arange(len(merged), dtype=np.float32)
    merged['full_doy'] = merged['Date'].dt.dayofyear.astype(np.float32)
    merged['month'] = merged['Date'].dt.month
    merged['seconds_since_start'] = (
        (merged['Date'] - merged['Date'].iloc[0]).dt.total_seconds().astype(np.float32)
    )
    total_duration_seconds = float(max(merged['seconds_since_start'].max(), SECONDS_PER_DAY))
    merged['time_norm'] = merged['seconds_since_start'] / total_duration_seconds

    merged['LST_surface_K'] = (
        merged['LST_surface_K']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
    )
    merged['LST_surface_C'] = merged['LST_surface_K'] - 273.15
    merged['BottomTemp_C'] = pick_numeric_series(merged, ['lblt_C', 'bottom_temp_C', 'BottomTemp_C'])
    merged['Solar_J_m2'] = pick_numeric_series(merged, ['Is_J_per_m2', 'solar_J_m2'])
    merged['Solar_W_m2'] = merged['Solar_J_m2'] / SECONDS_PER_DAY
    merged['MixedLayerDepth_m'] = pick_numeric_series(merged, ['lmld_m', 'mixed_layer_depth_m', 'MixedLayerDepth_m'])
    merged['T_air_C'] = pick_numeric_series(merged, ['t2m_C', 'air_temp_C', 'T_air_C'])

    wind_u = pick_numeric_series(merged, ['u10_m_per_s', 'u10', 'u10m'])
    wind_v = pick_numeric_series(merged, ['v10_m_per_s', 'v10', 'v10m'])
    if not wind_u.isna().all() and not wind_v.isna().all():
        merged['wind_speed_m_per_s'] = np.sqrt(wind_u ** 2 + wind_v ** 2)
    else:
        merged['wind_speed_m_per_s'] = pick_numeric_series(
            merged,
            ['wind_norm_m_per_s', 'wind_speed_m_per_s', 'wind_speed', 'u10_norm_m_per_s'],
            default=1.0,
        )

    merged['dewpoint_C'] = pick_numeric_series(merged, ['d2m_C', 'dewpoint_C', 'td_C'])
    if merged['dewpoint_C'].isna().all():
        dewpoint_k = pick_numeric_series(merged, ['d2m_K', 'dewpoint_K', 'td_K'])
        if not dewpoint_k.isna().all():
            merged['dewpoint_C'] = dewpoint_k - 273.15

    rh_series = pick_numeric_series(
        merged,
        ['rh', 'rh_frac', 'relative_humidity', 'relative_humidity_frac', 'rh_percent', 'relative_humidity_percent'],
    )
    if not rh_series.isna().all():
        if rh_series.max(skipna=True) > 1.5:
            rh_series = rh_series / 100.0
        merged['relative_humidity'] = rh_series
    elif not merged['dewpoint_C'].isna().all():
        es_air = saturation_vapor_pressure_np(merged['T_air_C'])
        ea_air = saturation_vapor_pressure_np(merged['dewpoint_C'])
        merged['relative_humidity'] = ea_air / es_air.clip(lower=1.0)
    else:
        merged['relative_humidity'] = 0.75

    cloud_fraction = pick_numeric_series(
        merged,
        ['tcc', 'cloud_cover', 'cloud_fraction', 'cloud_fraction_frac', 'cloud_fraction_percent'],
    )
    if not cloud_fraction.isna().all():
        if cloud_fraction.max(skipna=True) > 1.5:
            cloud_fraction = cloud_fraction / 100.0
        merged['cloud_fraction'] = cloud_fraction
    else:
        merged['cloud_fraction'] = 0.5

    pressure_pa = pick_numeric_series(merged, ['sp_Pa', 'surface_pressure_Pa', 'msl_Pa', 'pressure_Pa'])
    if pressure_pa.isna().all():
        pressure_hpa = pick_numeric_series(merged, ['sp_hPa', 'surface_pressure_hPa', 'msl_hPa', 'pressure_hPa'])
        if not pressure_hpa.isna().all():
            pressure_pa = pressure_hpa * 100.0
    merged['surface_pressure_Pa'] = pressure_pa if not pressure_pa.isna().all() else 101325.0

    if merged['T_air_C'].isna().all():
        merged['T_air_C'] = merged['LST_surface_C']
    else:
        merged['T_air_C'] = merged['T_air_C'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    merged['wind_speed_m_per_s'] = (
        merged['wind_speed_m_per_s']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=0.1)
    )

    is_freezing = merged['T_air_C'] < 0.0
    merged.loc[is_freezing, 'LST_surface_C'] = 0.0

    winter_months = merged['month'].isin([12, 1, 2])
    bottom_unreasonable = (merged['BottomTemp_C'] < 0.5) | (merged['BottomTemp_C'] > 10.0)
    merged.loc[winter_months & bottom_unreasonable, 'BottomTemp_C'] = 4.0
    merged['BottomTemp_C'] = merged['BottomTemp_C'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    merged.loc[merged['MixedLayerDepth_m'] < 0, 'MixedLayerDepth_m'] = 0.0
    merged['MixedLayerDepth_m'] = merged['MixedLayerDepth_m'].fillna(0.0)
    merged['Solar_J_m2'] = merged['Solar_J_m2'].interpolate(method='linear', limit_direction='both').bfill().ffill()
    merged['Solar_W_m2'] = merged['Solar_W_m2'].interpolate(method='linear', limit_direction='both').bfill().ffill()
    merged['relative_humidity'] = (
        merged['relative_humidity']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(0.2, 1.0)
    )
    merged['cloud_fraction'] = (
        merged['cloud_fraction']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(0.0, 1.0)
    )
    merged['surface_pressure_Pa'] = (
        merged['surface_pressure_Pa']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .clip(lower=80000.0, upper=110000.0)
    )

    required = [
        'LST_surface_C',
        'Solar_J_m2',
        'Solar_W_m2',
        'MixedLayerDepth_m',
        'T_air_C',
        'wind_speed_m_per_s',
        'relative_humidity',
        'cloud_fraction',
        'surface_pressure_Pa',
        'time_norm',
    ]
    if merged[required].isna().any().any():
        raise ValueError('Input data still contains missing values after preprocessing.')

    metadata = infer_metadata(merged, lst, era5_path, lst_path)
    metadata['time_scale_seconds'] = total_duration_seconds
    metadata['start_date'] = merged['Date'].iloc[0]
    return merged, metadata


def estimate_surface_bulk_temperature(
    lst_surface_c: np.ndarray,
    air_temp_c: np.ndarray,
    solar_w_m2: np.ndarray,
    wind_speed_m_per_s: np.ndarray,
    mixed_layer_depth_m: np.ndarray | None = None,
    skin_cooling_coef: float = SURFACE_SKIN_COOLING_COEF,
    air_blend: float = SURFACE_AIR_BLEND,
) -> np.ndarray:
    lst_surface_c = np.asarray(lst_surface_c, dtype=np.float64)
    air_temp_c = np.asarray(air_temp_c, dtype=np.float64)
    solar_w_m2 = np.asarray(solar_w_m2, dtype=np.float64)
    wind_speed_m_per_s = np.asarray(wind_speed_m_per_s, dtype=np.float64)
    if mixed_layer_depth_m is None:
        mixed_layer_depth_m = np.full_like(lst_surface_c, 2.0, dtype=np.float64)
    else:
        mixed_layer_depth_m = np.asarray(mixed_layer_depth_m, dtype=np.float64)

    skin_cooling_coef = float(max(skin_cooling_coef, 0.0))
    air_blend = float(np.clip(air_blend, 0.0, 1.0))
    wind_damping = np.exp(-np.clip(wind_speed_m_per_s, 0.0, 12.0) / 3.0)
    solar_skin_excess = skin_cooling_coef * np.clip(solar_w_m2, 0.0, None) * wind_damping
    solar_skin_excess = np.clip(solar_skin_excess, 0.0, 6.0)
    bulk_temp = lst_surface_c - solar_skin_excess
    stable_surface_excess = np.clip(lst_surface_c - air_temp_c, 0.0, 8.0)
    shallow_mld_factor = 1.0 / (1.0 + np.exp((mixed_layer_depth_m - 2.0) / 0.6))
    stable_stratification_factor = (
        (1.0 / (1.0 + np.exp(-(np.clip(solar_w_m2, 0.0, None) - 120.0) / 35.0)))
        * np.exp(-np.clip(wind_speed_m_per_s, 0.0, 12.0) / 2.5)
        * shallow_mld_factor
    )
    extra_skin_offset = np.clip(0.45 * stable_surface_excess * stable_stratification_factor, 0.0, 2.5)
    bulk_temp = bulk_temp - extra_skin_offset
    air_residual = np.clip(air_temp_c - bulk_temp, -4.0, 4.0)
    bulk_temp = bulk_temp + air_blend * air_residual
    bulk_temp = np.minimum(bulk_temp, lst_surface_c - 0.15 * stable_stratification_factor)
    return np.clip(bulk_temp, -1.0, 35.0)


def apply_forcing_adjustments(
    df: pd.DataFrame,
    solar_shading_factor: float = 1.0,
    surface_skin_cooling_coef: float = SURFACE_SKIN_COOLING_COEF,
    surface_air_blend: float = SURFACE_AIR_BLEND,
) -> pd.DataFrame:
    adjusted = df.copy()
    solar_shading_factor = float(max(solar_shading_factor, 0.0))
    if solar_shading_factor != 1.0:
        adjusted['Solar_W_m2'] = adjusted['Solar_W_m2'] * solar_shading_factor
        adjusted['Solar_J_m2'] = adjusted['Solar_J_m2'] * solar_shading_factor
    adjusted['SurfaceBulkTarget_C'] = estimate_surface_bulk_temperature(
        lst_surface_c=adjusted['LST_surface_C'].to_numpy(dtype=np.float64),
        air_temp_c=adjusted['T_air_C'].to_numpy(dtype=np.float64),
        solar_w_m2=adjusted['Solar_W_m2'].to_numpy(dtype=np.float64),
        wind_speed_m_per_s=adjusted['wind_speed_m_per_s'].to_numpy(dtype=np.float64),
        mixed_layer_depth_m=adjusted['MixedLayerDepth_m'].to_numpy(dtype=np.float64),
        skin_cooling_coef=surface_skin_cooling_coef,
        air_blend=surface_air_blend,
    )
    return adjusted


def load_optional_profile_observations(obs_source, start_date, time_scale_seconds, max_depth):
    if obs_source is None:
        return empty_profile_observation_frame()

    if isinstance(obs_source, pd.DataFrame):
        obs = obs_source.copy()
    else:
        obs_path = Path(obs_source)
        if not obs_path.exists():
            raise FileNotFoundError(f'Profile observation file not found: {obs_path}')
        obs = pd.read_csv(obs_path)
    date_col = first_existing_column(obs, ['Date', 'date', 'Datetime', 'datetime', 'Timestamp', 'timestamp'])
    depth_col = first_existing_column(obs, ['Depth_m', 'depth_m', 'Depth', 'depth'])
    temp_col = first_existing_column(obs, ['Temperature_C', 'temperature_C', 'Temp_C', 'temp_c', 'Temperature', 'temp'])

    if date_col and depth_col and temp_col:
        obs = obs[[date_col, depth_col, temp_col]].rename(
            columns={date_col: 'Date', depth_col: 'Depth_m', temp_col: 'Temperature_C'}
        )
    else:
        wide_temp_columns = []
        for column in obs.columns:
            column_str = str(column)
            depth_match = re.fullmatch(r'(?:Temp(?:erature)?_?)?(\d+(?:\.\d+)?)m', column_str, flags=re.IGNORECASE)
            if depth_match:
                wide_temp_columns.append((column, float(depth_match.group(1))))

        if not date_col or not wide_temp_columns:
            raise ValueError(
                'Profile observation CSV must contain either long-format date/depth/temperature columns '
                '(for example Date/Depth_m/Temperature_C) or wide-format columns like Temp_0m, Temp_1m, ... .'
            )

        rename_map = {column: f'__depth_{depth:g}m' for column, depth in wide_temp_columns}
        obs = obs[[date_col] + [column for column, _ in wide_temp_columns]].rename(columns={date_col: 'Date', **rename_map})
        obs = obs.melt(id_vars=['Date'], var_name='DepthLabel', value_name='Temperature_C')
        obs['Depth_m'] = obs['DepthLabel'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        obs = obs.drop(columns=['DepthLabel'])

    obs['Date'] = pd.to_datetime(obs['Date'])
    obs['Depth_m'] = pd.to_numeric(obs['Depth_m'], errors='coerce')
    obs['Temperature_C'] = pd.to_numeric(obs['Temperature_C'], errors='coerce')
    obs = obs.dropna().copy()
    obs['Depth_m'] = obs['Depth_m'].clip(lower=0.0, upper=max_depth)
    obs['time_norm'] = ((obs['Date'] - pd.Timestamp(start_date)).dt.total_seconds() / time_scale_seconds).clip(0.0, 1.0)
    return obs.sort_values(['Date', 'Depth_m']).reset_index(drop=True)


def split_profile_observations(profile_obs, split_mode='depth_interleaved'):
    split_frames = {role: empty_profile_observation_frame() for role in PROFILE_SPLIT_ROLES}
    split_summary = {
        role: {'rows': 0, 'depth_count': 0, 'date_count': 0}
        for role in PROFILE_SPLIT_ROLES
    }

    if not has_profile_observations(profile_obs):
        return split_frames, {'mode': split_mode, 'summary': split_summary}

    profile_obs = profile_obs.copy()
    if split_mode == 'none':
        split_frames['train'] = profile_obs.copy()
    elif split_mode == 'depth_interleaved':
        rounded_depths = np.round(profile_obs['Depth_m'].to_numpy(dtype=np.float64), 6)
        profile_obs['__depth_key'] = rounded_depths
        unique_depths = np.unique(rounded_depths)
        depth_to_role = {
            depth_value: DEFAULT_PROFILE_SPLIT_PATTERN[idx % len(DEFAULT_PROFILE_SPLIT_PATTERN)]
            for idx, depth_value in enumerate(unique_depths)
        }
        profile_obs['split_role'] = profile_obs['__depth_key'].map(depth_to_role)
        for role in PROFILE_SPLIT_ROLES:
            split_frames[role] = (
                profile_obs[profile_obs['split_role'] == role]
                .drop(columns=['__depth_key', 'split_role'])
                .reset_index(drop=True)
            )
    elif split_mode == 'time_blocked':
        normalized_dates = pd.to_datetime(profile_obs['Date']).dt.normalize()
        unique_dates = pd.Index(sorted(normalized_dates.unique()))
        n_dates = len(unique_dates)
        if n_dates == 0:
            return split_frames, {'mode': split_mode, 'summary': split_summary}

        boundaries = {}
        start_idx = 0
        for idx, role in enumerate(PROFILE_SPLIT_ROLES):
            if idx == len(PROFILE_SPLIT_ROLES) - 1:
                end_idx = n_dates
            else:
                fraction = TIME_BLOCK_SPLIT_FRACTIONS[role]
                end_idx = start_idx + int(round(n_dates * fraction))
                remaining_roles = len(PROFILE_SPLIT_ROLES) - idx - 1
                max_end = n_dates - remaining_roles
                end_idx = int(np.clip(end_idx, start_idx + 1, max_end))
            boundaries[role] = (start_idx, end_idx)
            start_idx = end_idx

        date_to_role = {}
        for role, (start_idx, end_idx) in boundaries.items():
            role_dates = unique_dates[start_idx:end_idx]
            for date_value in role_dates:
                date_to_role[pd.Timestamp(date_value)] = role

        profile_obs['split_role'] = normalized_dates.map(lambda value: date_to_role[pd.Timestamp(value)])
        for role in PROFILE_SPLIT_ROLES:
            split_frames[role] = (
                profile_obs[profile_obs['split_role'] == role]
                .drop(columns=['split_role'])
                .reset_index(drop=True)
            )
    else:
        raise ValueError(f'Unsupported profile split mode: {split_mode}')

    for role, frame in split_frames.items():
        split_summary[role] = {
            'rows': int(len(frame)),
            'depth_count': int(frame['Depth_m'].nunique()) if not frame.empty else 0,
            'date_count': int(pd.to_datetime(frame['Date']).dt.normalize().nunique()) if not frame.empty else 0,
        }

    return split_frames, {'mode': split_mode, 'summary': split_summary}


def subset_profile_observations_by_dates(profile_obs_data, dates):
    if not has_profile_observations(profile_obs_data):
        return empty_profile_observation_frame()

    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(min(dates)),
        time_scale_seconds=max((pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))).total_seconds(), SECONDS_PER_DAY),
        max_depth=np.inf,
    )
    date_set = {pd.Timestamp(date_value).normalize() for date_value in pd.to_datetime(list(dates))}
    subset = profile_obs[profile_obs['Date'].dt.normalize().isin(date_set)].copy()
    return subset.reset_index(drop=True)


def build_segment_frame(df: pd.DataFrame, start_idx: int, end_idx: int):
    segment_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    segment_df['day_index'] = np.arange(len(segment_df), dtype=np.float32)
    segment_df['seconds_since_start'] = (
        (segment_df['Date'] - segment_df['Date'].iloc[0]).dt.total_seconds().astype(np.float32)
    )
    total_duration_seconds = float(max(segment_df['seconds_since_start'].max(), SECONDS_PER_DAY))
    segment_df['time_norm'] = segment_df['seconds_since_start'] / total_duration_seconds
    return segment_df, total_duration_seconds


def fit_surface_bulk_correction(df, metadata, max_depth, profile_obs_data=None, max_surface_depth=1.0):
    """Fit a shallow-observation-informed mapping from satellite skin temperature to bulk surface temperature."""
    if not has_profile_observations(profile_obs_data):
        return None, None

    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if profile_obs.empty:
        return None, None

    shallow_obs = profile_obs[profile_obs['Depth_m'] <= max_surface_depth].copy()
    if shallow_obs.empty:
        return None, None

    shallow_daily = (
        shallow_obs.groupby('Date', as_index=False)['Temperature_C']
        .mean()
        .rename(columns={'Temperature_C': 'ObservedSurfaceBulk_C'})
    )
    calibration = df[['Date', 'LST_surface_C', 'T_air_C', 'wind_speed_m_per_s']].merge(
        shallow_daily,
        on='Date',
        how='inner',
    )
    calibration = calibration.dropna().copy()
    if len(calibration) < 10:
        return None, None

    x_lst = calibration['LST_surface_C'].to_numpy(dtype=np.float64)
    x_air = calibration['T_air_C'].to_numpy(dtype=np.float64)
    x_wind = calibration['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
    x_delta = x_lst - x_air
    X = np.column_stack([np.ones(len(calibration)), x_lst, x_air, x_wind, x_delta])
    y = calibration['ObservedSurfaceBulk_C'].to_numpy(dtype=np.float64)

    ridge_alpha = 0.5
    eye = np.eye(X.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + ridge_alpha * eye, X.T @ y)

    full_lst = df['LST_surface_C'].to_numpy(dtype=np.float64)
    full_air = df['T_air_C'].to_numpy(dtype=np.float64)
    full_wind = df['wind_speed_m_per_s'].to_numpy(dtype=np.float64)
    full_delta = full_lst - full_air
    X_full = np.column_stack([np.ones(len(df)), full_lst, full_air, full_wind, full_delta])
    corrected = X_full @ coef
    corrected = np.where(df['T_air_C'].to_numpy(dtype=np.float64) < 0.0, 0.0, corrected)
    corrected = np.clip(corrected, full_lst - 1.5, full_lst + 1.5)
    corrected = np.clip(corrected, -1.0, 35.0)

    raw_rmse = float(np.sqrt(np.mean((x_lst - y) ** 2)))
    fit_rmse = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    diagnostics = {
        'n_matches': int(len(calibration)),
        'raw_rmse': raw_rmse,
        'fit_rmse': fit_rmse,
        'coefficients': coef.tolist(),
    }

    if fit_rmse >= raw_rmse:
        return None, diagnostics

    corrected_series = pd.Series(corrected, index=df.index, dtype=np.float32)
    return corrected_series, diagnostics


def build_observation_dataframe(
    df,
    metadata,
    max_depth,
    profile_obs_data=None,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
):
    corrected_surface_obs = None
    surface_correction_info = None
    if use_surface_bulk_correction:
        corrected_surface_obs, surface_correction_info = fit_surface_bulk_correction(
            df=df,
            metadata=metadata,
            max_depth=max_depth,
            profile_obs_data=profile_obs_data,
            max_surface_depth=1.0,
        )

    base_surface = pd.DataFrame(
        {
            'Date': df['Date'],
            'Depth_m': float(np.clip(surface_obs_depth_m, 0.0, max_depth)),
            'Temperature_C': corrected_surface_obs if corrected_surface_obs is not None else df.get('SurfaceBulkTarget_C', df['LST_surface_C']),
            'time_norm': df['time_norm'],
            'source': 'surface',
        }
    )
    obs_frames = [base_surface]
    if use_bottom_observation:
        if not has_bottom_temperature_observations(df):
            raise ValueError('BottomTemp_C observations are required when use_bottom_observation is enabled.')
        base_bottom = pd.DataFrame(
            {
                'Date': df['Date'],
                'Depth_m': max_depth,
                'Temperature_C': df['BottomTemp_C'],
                'time_norm': df['time_norm'],
                'source': 'bottom',
            }
        )
        obs_frames.append(base_bottom)
    profile_obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if not profile_obs.empty:
        profile_obs = profile_obs.copy()
        profile_obs['source'] = 'profile'
        obs_frames.append(profile_obs)

    observations = pd.concat(obs_frames, ignore_index=True)
    observations['obs_weight'] = observations.apply(
        lambda row: compute_observation_weight(row['Depth_m'], row['source'], max_depth),
        axis=1,
    )
    observations = observations.sort_values(['Date', 'Depth_m']).reset_index(drop=True)
    return observations, surface_correction_info


def build_initial_condition_profile(df, max_depth, n_points=64, mode='uniform_4c'):
    z_ic = np.linspace(0.0, max_depth, n_points, dtype=np.float32)
    surf0_series = df['SurfaceBulkTarget_C'] if 'SurfaceBulkTarget_C' in df.columns else df['LST_surface_C']
    surf0 = float(surf0_series.iloc[0])
    if mode == 'uniform_4c':
        temp_ic = np.full_like(z_ic, DEFAULT_INITIAL_WATER_TEMPERATURE_C, dtype=np.float32)
    elif mode == 'surface_to_uniform_4c':
        temp_ic = surf0 + (DEFAULT_INITIAL_WATER_TEMPERATURE_C - surf0) * (z_ic / max(max_depth, 1e-6))
    elif mode == 'linear_to_bottom_obs':
        if not has_bottom_temperature_observations(df):
            raise ValueError('BottomTemp_C observations are required when initial_condition_mode=linear_to_bottom_obs.')
        bottom0 = float(df['BottomTemp_C'].iloc[0])
        temp_ic = surf0 + (bottom0 - surf0) * (z_ic / max(max_depth, 1e-6))
    else:
        raise ValueError(f'Unsupported initial condition mode: {mode}')
    return z_ic.reshape(-1, 1), temp_ic.reshape(-1, 1)


def compute_observation_weight(depth_m, source, max_depth):
    """Moderately emphasize shallow-layer observations during training."""
    depth_m = float(depth_m)
    if source == 'surface':
        return 4.0
    if source == 'bottom':
        return 1.0
    if depth_m <= 1.0:
        return 3.5
    if depth_m <= 3.0:
        return 2.75
    if depth_m <= 5.0:
        return 1.8
    if depth_m <= min(max_depth, 10.0):
        return 1.2
    if depth_m <= min(max_depth, 15.0):
        return 0.95
    return 0.8


def depth_dependent_obs_std(source, depth_m, max_depth, base_surface, base_bottom, base_profile):
    """Use mildly depth-dependent observation noise for Kalman updates."""
    depth_m = float(depth_m)
    if source == 'surface':
        return max(0.15, base_surface * 0.75)
    if source == 'bottom':
        return max(0.3, base_bottom * 1.0)

    if depth_m <= 1.0:
        scale = 0.55
    elif depth_m <= 3.0:
        scale = 0.65
    elif depth_m <= 5.0:
        scale = 0.8
    elif depth_m <= min(max_depth, 10.0):
        scale = 0.95
    elif depth_m <= min(max_depth, 15.0):
        scale = 1.15
    else:
        scale = 1.3
    return max(0.2, base_profile * scale)


def build_depth_grid(max_depth, n_depth_points, use_shallow_optimized=True, shallow_focus_depth=5.0, shallow_fraction=0.55):
    """Create a prediction/assimilation depth grid with extra resolution in shallow layers."""
    max_depth = float(max_depth)
    n_depth_points = int(max(2, n_depth_points))
    shallow_focus_depth = float(np.clip(shallow_focus_depth, 0.5, max_depth))
    shallow_fraction = float(np.clip(shallow_fraction, 0.2, 0.9))

    if (not use_shallow_optimized) or shallow_focus_depth >= max_depth or n_depth_points < 8:
        return np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)

    shallow_points = int(round(n_depth_points * shallow_fraction))
    shallow_points = int(np.clip(shallow_points, 4, n_depth_points - 3))
    deep_points = n_depth_points - shallow_points + 1

    shallow_grid = np.linspace(0.0, shallow_focus_depth, shallow_points, dtype=np.float32)
    deep_grid = np.linspace(shallow_focus_depth, max_depth, deep_points, dtype=np.float32)
    full_grid = np.unique(np.concatenate([shallow_grid, deep_grid]))

    if len(full_grid) < n_depth_points:
        fallback_grid = np.linspace(0.0, max_depth, n_depth_points, dtype=np.float32)
        full_grid = np.unique(np.concatenate([full_grid, fallback_grid]))

    return np.sort(full_grid.astype(np.float32))


def model_temperature(model, t, z, max_depth):
    return model(torch.cat([t, z / max_depth], dim=1))


def compute_diffusivity(model, t_col, z_col, max_depth, wind_speed, water_depth):
    temp_pred = model_temperature(model, t_col, z_col, max_depth)
    dT_dt_norm = torch.autograd.grad(temp_pred, t_col, grad_outputs=torch.ones_like(temp_pred), create_graph=True)[0]
    dT_dz = torch.autograd.grad(temp_pred, z_col, grad_outputs=torch.ones_like(temp_pred), create_graph=True)[0]

    density = water_density_torch(temp_pred)
    density_gradient = torch.autograd.grad(density, z_col, grad_outputs=torch.ones_like(density), create_graph=True)[0]

    depth_scale = torch.full_like(z_col, max(float(water_depth), 1.0))
    shear_term = RI_WIND_SHEAR_FACTOR * wind_speed.clamp(min=0.1) ** 2 / depth_scale ** 2
    richardson_number = -(GRAVITY / WATER_DENSITY) * density_gradient / shear_term.clamp(min=1e-8)

    stability_factor = (1.0 + DIFFUSIVITY_RI_SENSITIVITY * richardson_number).clamp(min=0.05)
    wind_decay = torch.exp(-z_col / DIFFUSIVITY_WIND_DECAY_DEPTH)
    wind_mixing = (
        DIFFUSIVITY_WIND_COEFF
        * wind_speed.clamp(min=0.0) ** DIFFUSIVITY_WIND_EXPONENT
        * wind_decay
    )
    eddy_diffusivity = DIFFUSIVITY_K0 * wind_mixing * stability_factor.pow(-DIFFUSIVITY_ALPHA)
    eddy_diffusivity = eddy_diffusivity.clamp(min=MIN_EDDY_DIFFUSIVITY)
    diffusivity = (MOLECULAR_DIFFUSIVITY + eddy_diffusivity).clamp(min=MIN_TOTAL_DIFFUSIVITY)
    return temp_pred, dT_dt_norm, dT_dz, density_gradient, richardson_number, diffusivity


def compute_surface_flux_terms(surface_temp, batch, shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION):
    air_temp = batch['surface_air_temp']
    air_temp_k = air_temp + 273.15
    surface_temp_k = surface_temp + 273.15
    wind_speed = batch['surface_wind_speed'].clamp(min=0.1)
    surface_pressure = batch['surface_pressure'].clamp(min=80000.0, max=110000.0)

    air_density = surface_pressure / (287.05 * air_temp_k.clamp(min=200.0))
    vapor_pressure_air = batch['surface_relative_humidity'].clamp(0.2, 1.0) * saturation_vapor_pressure_torch(air_temp)
    vapor_pressure_surface = saturation_vapor_pressure_torch(surface_temp)
    q_air = specific_humidity_from_vapor_pressure_torch(vapor_pressure_air, surface_pressure)
    q_surface = specific_humidity_from_vapor_pressure_torch(vapor_pressure_surface, surface_pressure)

    sensible_heat = air_density * AIR_HEAT_CAPACITY * TRANSFER_COEFF_HEAT * wind_speed * (surface_temp - air_temp)
    latent_heat = air_density * LATENT_HEAT_VAPORIZATION * TRANSFER_COEFF_MOISTURE * wind_speed * (q_surface - q_air)

    ice_indicator = smooth_ice_indicator(surface_temp)
    surface_albedo = (
        SURFACE_ALBEDO_WATER * (1.0 - ice_indicator) +
        SURFACE_ALBEDO_ICE * ice_indicator
    )
    shortwave_surface_fraction = float(np.clip(shortwave_surface_fraction, 0.0, 1.0))
    absorbed_surface_shortwave = shortwave_surface_fraction * (1.0 - surface_albedo) * batch['surface_shortwave']
    longwave_net = (
        ATMOSPHERIC_EMISSIVITY * STEFAN_BOLTZMANN * air_temp_k ** 4
        - WATER_EMISSIVITY * STEFAN_BOLTZMANN * surface_temp_k ** 4
    )
    net_radiation = absorbed_surface_shortwave + longwave_net
    seb_flux = (net_radiation - sensible_heat - latent_heat) / RHO_CP

    return {
        'seb_flux': seb_flux,
        'ice_indicator': ice_indicator,
        'net_radiation': net_radiation,
        'sensible_heat': sensible_heat,
        'latent_heat': latent_heat,
    }


def compute_losses(
    model,
    batch,
    max_depth,
    time_scale_seconds,
    weights,
    shortwave_attenuation=SHORTWAVE_ATTENUATION,
    shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
    max_vertical_gradient_c_per_m=MAX_VERTICAL_GRADIENT_C_PER_M,
    entrainment_velocity_scale_m_per_day=MAX_ENTRAINMENT_VELOCITY_M_PER_DAY,
):
    shortwave_attenuation = float(np.clip(shortwave_attenuation, MIN_SHORTWAVE_ATTENUATION, MAX_SHORTWAVE_ATTENUATION))
    max_vertical_gradient_c_per_m = float(max(max_vertical_gradient_c_per_m, 0.1))
    t_col = batch['t_colloc'].clone().detach().requires_grad_(True)
    z_col = batch['z_colloc'].clone().detach().requires_grad_(True)

    temp_pred, dT_dt_norm, dT_dz, density_gradient, richardson_number, diffusivity = compute_diffusivity(
        model=model,
        t_col=t_col,
        z_col=z_col,
        max_depth=max_depth,
        wind_speed=batch['wind_colloc'],
        water_depth=max_depth,
    )

    dT_dt_real = dT_dt_norm / time_scale_seconds
    conductive_flux = diffusivity * dT_dz
    dflux_dz = torch.autograd.grad(conductive_flux, z_col, grad_outputs=torch.ones_like(conductive_flux), create_graph=True)[0]

    penetrating_shortwave = (
        (1.0 - shortwave_surface_fraction)
        * batch['solar_flux_colloc']
        * torch.exp(-shortwave_attenuation * z_col)
    )
    dphi_dz = -shortwave_attenuation * penetrating_shortwave
    heating_term = -dphi_dz / RHO_CP

    pde_residual = dT_dt_real - dflux_dz - heating_term
    loss_pde = torch.mean(pde_residual ** 2)

    t_surface = batch['surface_time'].clone().detach().requires_grad_(True)
    z_surface = torch.zeros_like(t_surface, requires_grad=True)
    surface_temp = model_temperature(model, t_surface, z_surface, max_depth)
    dT_dz_surface = torch.autograd.grad(
        surface_temp,
        z_surface,
        grad_outputs=torch.ones_like(surface_temp),
        create_graph=True,
    )[0]
    _, _, _, _, _, diffusivity_surface = compute_diffusivity(
        model=model,
        t_col=t_surface,
        z_col=z_surface,
        max_depth=max_depth,
        wind_speed=batch['surface_wind_speed'],
        water_depth=max_depth,
    )
    surface_flux = compute_surface_flux_terms(
        surface_temp,
        batch,
        shortwave_surface_fraction=shortwave_surface_fraction,
    )
    seb_residual = diffusivity_surface * dT_dz_surface - surface_flux['seb_flux']
    ice_residual = surface_temp
    loss_surface_bc = torch.mean(
        (1.0 - surface_flux['ice_indicator']) * seb_residual ** 2
        + surface_flux['ice_indicator'] * ice_residual ** 2
    )

    t_bottom = batch['surface_time'].clone().detach().requires_grad_(True)
    z_bottom = torch.full_like(t_bottom, max_depth, requires_grad=True)
    bottom_temp = model_temperature(model, t_bottom, z_bottom, max_depth)
    dT_dz_bottom = torch.autograd.grad(
        bottom_temp,
        z_bottom,
        grad_outputs=torch.ones_like(bottom_temp),
        create_graph=True,
    )[0]
    loss_bottom_flux = torch.mean(dT_dz_bottom ** 2)
    loss_bc = loss_surface_bc + loss_bottom_flux

    ic_temp_pred = model_temperature(model, batch['ic_time'], batch['ic_depth'], max_depth)
    loss_ic = torch.mean((ic_temp_pred - batch['ic_temperature']) ** 2)

    obs_temp_pred = model_temperature(model, batch['obs_time'], batch['obs_depth'], max_depth)
    obs_residual_sq = (obs_temp_pred - batch['obs_temperature']) ** 2
    loss_obs = torch.mean(batch['obs_weight'] * obs_residual_sq)

    loss_time_continuity = torch.zeros((), dtype=torch.float32, device=t_col.device)
    continuity_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        t_seq_now = batch['seq_time_now'].clone().detach().requires_grad_(True)
        z_seq = batch['seq_depth'].clone().detach().requires_grad_(True)
        temp_seq_now, _, dT_dz_seq, _, _, diffusivity_seq = compute_diffusivity(
            model=model,
            t_col=t_seq_now,
            z_col=z_seq,
            max_depth=max_depth,
            wind_speed=batch['seq_wind_now'],
            water_depth=max_depth,
        )
        conductive_flux_seq = diffusivity_seq * dT_dz_seq
        dflux_dz_seq = torch.autograd.grad(
            conductive_flux_seq,
            z_seq,
            grad_outputs=torch.ones_like(conductive_flux_seq),
            create_graph=True,
        )[0]
        penetrating_shortwave_seq = (
            (1.0 - shortwave_surface_fraction)
            * batch['seq_solar_flux_now']
            * torch.exp(-shortwave_attenuation * z_seq)
        )
        dphi_dz_seq = -shortwave_attenuation * penetrating_shortwave_seq
        heating_seq = -dphi_dz_seq / RHO_CP
        temp_seq_next = model_temperature(model, batch['seq_time_next'], z_seq, max_depth)
        continuity_residual = temp_seq_next - temp_seq_now - batch['seq_delta_seconds'] * (dflux_dz_seq + heating_seq)
        loss_time_continuity = torch.mean(continuity_residual ** 2)
        continuity_residual_rms = torch.sqrt(torch.mean(continuity_residual ** 2) + 1e-12)

    loss_vertical_exchange = torch.zeros((), dtype=torch.float32, device=t_col.device)
    vertical_exchange_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        entrainment_velocity_cap = float(max(entrainment_velocity_scale_m_per_day, 0.05)) / SECONDS_PER_DAY
        mld_deepening = torch.relu(batch['seq_mld_next'] - batch['seq_mld_now'])
        entrainment_velocity = torch.clamp(
            mld_deepening / torch.clamp(batch['seq_delta_seconds'], min=1.0),
            min=0.0,
            max=entrainment_velocity_cap,
        )
        entrainment_center = batch['seq_mld_now'] + 0.35 * mld_deepening + 0.5
        entrainment_gate = torch.exp(-((z_seq - entrainment_center) / 1.25) ** 2)
        advection_tendency = -entrainment_velocity * dT_dz_seq * entrainment_gate
        vertical_exchange_residual = temp_seq_next - temp_seq_now - batch['seq_delta_seconds'] * (
            dflux_dz_seq + heating_seq + advection_tendency
        )
        loss_vertical_exchange = torch.mean((entrainment_gate * vertical_exchange_residual) ** 2)
        vertical_exchange_residual_rms = torch.sqrt(torch.mean((entrainment_gate * vertical_exchange_residual) ** 2) + 1e-12)

    loss_convective_mixing = torch.zeros((), dtype=torch.float32, device=t_col.device)
    convective_mixing_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        surface_seq_now = model_temperature(model, batch['seq_time_now'], batch['seq_surface_depth'], max_depth)
        cooling_indicator = torch.sigmoid((8.0 - batch['seq_air_temp_now']) / 2.0)
        low_solar_indicator = torch.sigmoid((120.0 - batch['seq_solar_flux_now']) / 35.0)
        deepening_indicator = torch.sigmoid((batch['seq_mld_next'] - batch['seq_mld_now'] - 0.05) / 0.15)
        convective_gate = cooling_indicator * low_solar_indicator * deepening_indicator
        within_mixed_layer = torch.sigmoid((batch['seq_mld_now'] + 0.75 - z_seq) / 0.5)
        convective_residual = convective_gate * within_mixed_layer * (temp_seq_now - surface_seq_now)
        loss_convective_mixing = torch.mean(convective_residual ** 2)
        convective_mixing_residual_rms = torch.sqrt(torch.mean(convective_residual ** 2) + 1e-12)

    loss_autumn_overturn = torch.zeros((), dtype=torch.float32, device=t_col.device)
    autumn_overturn_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0 and batch.get('seq_doy_now') is not None:
        surface_seq_now = model_temperature(model, batch['seq_time_now'], batch['seq_surface_depth'], max_depth)
        surface_seq_next = model_temperature(model, batch['seq_time_next'], batch['seq_surface_depth'], max_depth)
        autumn_indicator = torch.sigmoid((batch['seq_doy_now'] - 255.0) / 8.0) * torch.sigmoid((335.0 - batch['seq_doy_now']) / 8.0)
        surface_cooling_indicator = torch.sigmoid((surface_seq_now - surface_seq_next - 0.10) / 0.10)
        mld_deepening_indicator = torch.sigmoid((batch['seq_mld_next'] - batch['seq_mld_now'] - 0.05) / 0.12)
        low_solar_indicator = torch.sigmoid((140.0 - batch['seq_solar_flux_now']) / 30.0)
        cool_air_indicator = torch.sigmoid((14.0 - batch['seq_air_temp_now']) / 2.0)
        overturn_gate = autumn_indicator * surface_cooling_indicator * mld_deepening_indicator * low_solar_indicator * cool_air_indicator

        overturn_deep_floor = torch.clamp(
            torch.maximum(batch['seq_mld_now'] + 1.5, torch.full_like(batch['seq_mld_now'], max_depth * 0.35)),
            min=2.5,
            max=max_depth - 0.5,
        )
        overturn_deep_depth = torch.clamp(overturn_deep_floor + 1.0, min=3.0, max=max_depth)
        deep_seq_now = model_temperature(model, batch['seq_time_now'], overturn_deep_depth, max_depth)
        deep_seq_next = model_temperature(model, batch['seq_time_next'], overturn_deep_depth, max_depth)
        prev_gap = torch.relu(surface_seq_now - deep_seq_now)
        next_gap = torch.relu(surface_seq_next - deep_seq_next)
        gap_collapse = torch.relu(prev_gap - next_gap)
        insufficient_collapse = torch.relu(AUTUMN_OVERTURN_TARGET_GAP_COLLAPSE_C - gap_collapse)
        allowable_deep_warm = AUTUMN_OVERTURN_DEEP_WARM_ALLOWANCE_C_PER_DAY * (batch['seq_delta_seconds'] / SECONDS_PER_DAY)
        fake_overturn_warm = torch.relu((deep_seq_next - deep_seq_now) - allowable_deep_warm)
        overturn_residual = overturn_gate * (insufficient_collapse + 1.5 * fake_overturn_warm)
        loss_autumn_overturn = torch.mean(overturn_residual ** 2)
        autumn_overturn_residual_rms = torch.sqrt(torch.mean(overturn_residual ** 2) + 1e-12)

    loss_stratification = torch.zeros((), dtype=torch.float32, device=t_col.device)
    stratification_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('strat_time') is not None and batch['strat_time'].numel() > 0:
        strat_time = batch['strat_time']
        shallow_temp = model_temperature(model, strat_time, batch['strat_shallow_depth'], max_depth)
        deep_temp = model_temperature(model, strat_time, batch['strat_deep_depth'], max_depth)
        strat_violation = torch.relu(deep_temp - (shallow_temp - batch['strat_margin']))
        loss_stratification = torch.mean(batch['strat_weight'] * strat_violation ** 2)
        stratification_residual_rms = torch.sqrt(torch.mean(strat_violation ** 2) + 1e-12)

    smoothness_excess = torch.relu(torch.abs(dT_dz) - max_vertical_gradient_c_per_m)
    loss_smoothness = torch.mean(smoothness_excess ** 2)
    smoothness_residual_rms = torch.sqrt(torch.mean(smoothness_excess ** 2) + 1e-12)

    loss_deep_warming = torch.zeros((), dtype=torch.float32, device=t_col.device)
    deep_warming_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('seq_time_now') is not None and batch['seq_time_now'].numel() > 0:
        deep_gate = torch.sigmoid((z_seq - batch['seq_deep_floor']) / 0.9)
        warm_gate = torch.sigmoid((batch['seq_air_temp_now'] - 8.0) / 2.0) * torch.sigmoid((batch['seq_solar_flux_now'] - 110.0) / 35.0)
        allowable_warming = DEEP_WARMING_ALLOWANCE_C_PER_DAY * (batch['seq_delta_seconds'] / SECONDS_PER_DAY)
        deep_warming_excess = torch.relu((temp_seq_next - temp_seq_now) - allowable_warming)
        deep_warming_residual = deep_gate * warm_gate * deep_warming_excess
        loss_deep_warming = torch.mean(deep_warming_residual ** 2)
        deep_warming_residual_rms = torch.sqrt(torch.mean(deep_warming_residual ** 2) + 1e-12)

    loss_deep_anchor = torch.zeros((), dtype=torch.float32, device=t_col.device)
    deep_anchor_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('deep_anchor_time') is not None and batch['deep_anchor_time'].numel() > 0:
        deep_temp = model_temperature(model, batch['deep_anchor_time'], batch['deep_anchor_depth'], max_depth)
        deep_excess = torch.relu(deep_temp - batch['deep_anchor_target'])
        loss_deep_anchor = torch.mean(batch['deep_anchor_weight'] * deep_excess ** 2)
        deep_anchor_residual_rms = torch.sqrt(torch.mean(deep_excess ** 2) + 1e-12)

    loss_heat_budget = torch.zeros((), dtype=torch.float32, device=t_col.device)
    heat_budget_residual_rms = torch.zeros((), dtype=torch.float32, device=t_col.device)
    if batch.get('budget_time_now') is not None and batch['budget_time_now'].numel() > 0:
        t_budget_now = batch['budget_time_now']
        t_budget_next = batch['budget_time_next']
        z_budget = batch['budget_depth'].clone().detach().requires_grad_(True)
        temp_budget_now = model_temperature(model, t_budget_now, z_budget, max_depth)
        temp_budget_next = model_temperature(model, t_budget_next, z_budget, max_depth)
        dz_budget = batch['budget_dz']
        heat_content_rate = ((temp_budget_next - temp_budget_now) / torch.clamp(batch['budget_delta_seconds'], min=1.0)) * dz_budget
        budget_group_index = batch['budget_group_index'].reshape(-1, 1)
        integrated_heat_tendency = torch.zeros_like(batch['budget_surface_flux'])
        integrated_heat_tendency.scatter_add_(0, budget_group_index, heat_content_rate)

        penetrating_shortwave_budget = (
            (1.0 - shortwave_surface_fraction)
            * batch['budget_solar_flux']
            * torch.exp(-shortwave_attenuation * z_budget)
        )
        dphi_dz_budget = -shortwave_attenuation * penetrating_shortwave_budget
        internal_heating_rate = (-dphi_dz_budget / RHO_CP) * dz_budget
        integrated_internal_heating = torch.zeros_like(batch['budget_surface_flux'])
        integrated_internal_heating.scatter_add_(0, budget_group_index, internal_heating_rate)

        external_heat_tendency = batch['budget_surface_flux'] + integrated_internal_heating
        heat_budget_residual = integrated_heat_tendency - external_heat_tendency
        loss_heat_budget = torch.mean(heat_budget_residual ** 2)
        heat_budget_residual_rms = torch.sqrt(torch.mean(heat_budget_residual ** 2) + 1e-12)

    density_instability = torch.relu(-density_gradient)
    loss_density_reg = torch.mean(density_instability ** 2)

    loss_total = (
        weights['pde'] * loss_pde +
        weights['bc'] * loss_bc +
        weights['ic'] * loss_ic +
        weights['obs'] * loss_obs +
        weights.get('time_continuity', 0.0) * loss_time_continuity +
        weights.get('stratification', 0.0) * loss_stratification +
        weights.get('smoothness', 0.0) * loss_smoothness +
        weights.get('deep_warming', 0.0) * loss_deep_warming +
        weights.get('deep_anchor', 0.0) * loss_deep_anchor +
        weights.get('vertical_exchange', 0.0) * loss_vertical_exchange +
        weights.get('convective_mixing', 0.0) * loss_convective_mixing +
        weights.get('autumn_overturn', 0.0) * loss_autumn_overturn +
        weights.get('heat_budget', 0.0) * loss_heat_budget +
        weights['density_reg'] * loss_density_reg
    )

    return {
        'total': loss_total,
        'loss_pde': loss_pde,
        'loss_bc': loss_bc,
        'loss_surface_bc': loss_surface_bc,
        'loss_bottom_flux': loss_bottom_flux,
        'loss_ic': loss_ic,
        'loss_obs': loss_obs,
        'loss_time_continuity': loss_time_continuity,
        'loss_stratification': loss_stratification,
        'loss_smoothness': loss_smoothness,
        'loss_deep_warming': loss_deep_warming,
        'loss_deep_anchor': loss_deep_anchor,
        'loss_vertical_exchange': loss_vertical_exchange,
        'loss_convective_mixing': loss_convective_mixing,
        'loss_autumn_overturn': loss_autumn_overturn,
        'loss_heat_budget': loss_heat_budget,
        'loss_density_reg': loss_density_reg,
        'seb_residual_rms': torch.sqrt(torch.mean(seb_residual ** 2) + 1e-12),
        'pde_residual_rms': torch.sqrt(torch.mean(pde_residual ** 2) + 1e-12),
        'continuity_residual_rms': continuity_residual_rms,
        'stratification_residual_rms': stratification_residual_rms,
        'smoothness_residual_rms': smoothness_residual_rms,
        'deep_warming_residual_rms': deep_warming_residual_rms,
        'deep_anchor_residual_rms': deep_anchor_residual_rms,
        'vertical_exchange_residual_rms': vertical_exchange_residual_rms,
        'convective_mixing_residual_rms': convective_mixing_residual_rms,
        'autumn_overturn_residual_rms': autumn_overturn_residual_rms,
        'heat_budget_residual_rms': heat_budget_residual_rms,
        'ri_mean': richardson_number.mean().detach(),
        'kappa_mean': diffusivity.mean().detach(),
    }


def summarize_window_losses(window_losses):
    if not window_losses:
        raise ValueError('window_losses must not be empty.')

    keys = window_losses[0].keys()
    summary = {}
    for key in keys:
        values = [float(item[key]) for item in window_losses]
        summary[key] = float(np.mean(values))
    return summary


def evaluate_profile_grid(df, metadata, temp_grid, depths, max_depth, profile_obs_data):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=max_depth,
    )
    if observations.empty:
        return None

    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    date_to_index = {
        pd.Timestamp(date_value).normalize(): idx
        for idx, date_value in enumerate(pd.to_datetime(df['Date']))
    }

    errors = []
    for date_value, obs_day in observations.groupby(observations['Date'].dt.normalize()):
        day_idx = date_to_index.get(pd.Timestamp(date_value).normalize())
        if day_idx is None:
            continue
        pred_profile = temp_grid[:, day_idx]
        pred_interp = np.interp(
            obs_day['Depth_m'].to_numpy(dtype=np.float64),
            depths,
            pred_profile,
        )
        errors.extend((pred_interp - obs_day['Temperature_C'].to_numpy(dtype=np.float64)).tolist())

    if not errors:
        return None

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'matched_rows': int(len(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
    }


def evaluate_profile_at_date(current_date, current_profile, depths, profile_obs_data):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(current_date).normalize(),
        time_scale_seconds=SECONDS_PER_DAY,
        max_depth=float(np.max(depths)) if len(depths) else 0.0,
    )
    if observations.empty:
        return None

    obs_day = observations[observations['Date'].dt.normalize() == pd.Timestamp(current_date).normalize()]
    if obs_day.empty:
        return None

    current_profile = np.asarray(current_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    pred_interp = np.interp(
        obs_day['Depth_m'].to_numpy(dtype=np.float64),
        depths,
        current_profile,
    )
    errors = pred_interp - obs_day['Temperature_C'].to_numpy(dtype=np.float64)
    return {
        'matched_rows': int(len(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
    }


def smooth_time_gate(doy, start_doy, end_doy, width_days=4.0):
    doy = np.asarray(doy, dtype=np.float64)
    width_days = float(max(width_days, 1.0e-6))
    rise = 1.0 / (1.0 + np.exp(-(doy - float(start_doy)) / width_days))
    fall = 1.0 / (1.0 + np.exp(-(float(end_doy) - doy) / width_days))
    return np.clip(rise * fall, 0.0, 1.0)


def evaluate_surface_band_validation_at_date(
    current_date,
    current_profile,
    depths,
    profile_obs_data,
    previous_profile_3d=None,
    shallow_max_depth=3.0,
):
    if not has_profile_observations(profile_obs_data):
        return None

    observations = load_optional_profile_observations(
        profile_obs_data,
        start_date=pd.Timestamp(current_date).normalize(),
        time_scale_seconds=SECONDS_PER_DAY,
        max_depth=float(np.max(depths)) if len(depths) else 0.0,
    )
    if observations.empty:
        return None

    current_date = pd.Timestamp(current_date).normalize()
    obs_day = observations[observations['Date'].dt.normalize() == current_date]
    if obs_day.empty:
        return None

    shallow_obs = obs_day[pd.to_numeric(obs_day['Depth_m'], errors='coerce') <= float(shallow_max_depth)].copy()
    if shallow_obs.empty:
        return {
            'may_surface_warm_penalty': 0.0,
            'may_surface_rate_penalty': 0.0,
            'july_surface_cool_penalty': 0.0,
            'july_surface_warm_reward': 0.0,
            'surface_band_background_rmse': 0.0,
        }

    current_profile = np.asarray(current_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    shallow_depths = shallow_obs['Depth_m'].to_numpy(dtype=np.float64)
    pred_shallow = np.interp(shallow_depths, depths, current_profile)
    val_shallow = shallow_obs['Temperature_C'].to_numpy(dtype=np.float64)
    shallow_error = pred_shallow - val_shallow

    doy = float(current_date.dayofyear)
    may_gate = float(smooth_time_gate(doy, 110.0, 150.0, width_days=4.0))
    july_gate = float(smooth_time_gate(doy, 180.0, 215.0, width_days=4.0))
    background_gate = float(max(0.0, 1.0 - min(may_gate + july_gate, 1.0)))

    may_surface_warm_penalty = may_gate * float(np.mean(np.clip(shallow_error, 0.0, None)))
    july_surface_cool_penalty = july_gate * float(np.mean(np.clip(-shallow_error, 0.0, None)))
    july_surface_warm_reward = july_gate * float(np.mean(np.exp(-(shallow_error ** 2) / (2.0 ** 2))))
    surface_band_background_rmse = background_gate * float(np.sqrt(np.mean(shallow_error ** 2)))

    may_surface_rate_penalty = 0.0
    lag_date = current_date - pd.Timedelta(days=3)
    if previous_profile_3d is not None:
        obs_lag = observations[observations['Date'].dt.normalize() == lag_date]
        shallow_lag_obs = obs_lag[pd.to_numeric(obs_lag['Depth_m'], errors='coerce') <= float(shallow_max_depth)].copy()
        if not shallow_lag_obs.empty:
            previous_profile_3d = np.asarray(previous_profile_3d, dtype=np.float64)
            pred_shallow_lag = np.interp(
                shallow_lag_obs['Depth_m'].to_numpy(dtype=np.float64),
                depths,
                previous_profile_3d,
            )
            val_shallow_lag = shallow_lag_obs['Temperature_C'].to_numpy(dtype=np.float64)
            rate_pred = (float(np.mean(pred_shallow)) - float(np.mean(pred_shallow_lag))) / 3.0
            rate_val = (float(np.mean(val_shallow)) - float(np.mean(val_shallow_lag))) / 3.0
            may_surface_rate_penalty = may_gate * float(max(rate_pred - rate_val, 0.0))

    return {
        'may_surface_warm_penalty': float(may_surface_warm_penalty),
        'may_surface_rate_penalty': float(may_surface_rate_penalty),
        'july_surface_cool_penalty': float(july_surface_cool_penalty),
        'july_surface_warm_reward': float(july_surface_warm_reward),
        'surface_band_background_rmse': float(surface_band_background_rmse),
    }


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


def compute_runtime_surface_target(
    df,
    day_idx,
    runtime_skin_cooling_coef,
    base_surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    if 'SurfaceBulkTarget_C' not in df.columns:
        return float(df['LST_surface_C'].iloc[day_idx])
    base_target = float(df['SurfaceBulkTarget_C'].iloc[day_idx])
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else base_target
    baseline_coef = max(float(base_surface_skin_cooling_coef), 1.0e-6)
    runtime_coef = float(np.clip(runtime_skin_cooling_coef, 0.005, 0.08))
    skin_gap = raw_lst_surface - base_target
    scaled_gap = skin_gap * (runtime_coef / baseline_coef)
    adjusted_target = raw_lst_surface - scaled_gap
    return float(np.clip(adjusted_target, -1.0, 35.0))


def apply_ppo_action(weights, kalman_scales, action, tune_kalman=True):
    action = np.asarray(action, dtype=np.float32)
    updated_weights = dict(weights)
    updated_scales = dict(kalman_scales)

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
            kalman_scales['process'],
            action[len(PPO_WEIGHT_STATE_KEYS)],
            0.5,
            2.5,
            step_size=0.10,
        )
        updated_scales['obs'] = update_control_value(
            kalman_scales['obs'],
            action[len(PPO_WEIGHT_STATE_KEYS) + 1],
            0.6,
            3.5,
            step_size=0.10,
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
        reward += 0.25 * relative_improvement(
            prev_validation_metrics.get('rmse', 0.0),
            current_validation_metrics.get('rmse', 0.0),
            scale=1.0,
            floor=1.0,
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
        'hidden_dim': 128,
        'hidden_layers': 8,
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': info.get('optimizer_state_dict'),
        'scheduler_state_dict': info.get('scheduler_state_dict'),
        'training_info': {
            'final_weights': dict(info.get('final_weights', {})),
            'kalman_scales': dict(info.get('kalman_scales', {'process': 1.0, 'obs': 1.0})),
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
    hidden_dim = int(bundle.get('hidden_dim', 128))
    hidden_layers = int(bundle.get('hidden_layers', 8))
    model = LakePINN(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()
    return model, bundle


def derive_online_control_params_from_weights(
    initial_weights,
    memory_blend,
    surface_relaxation,
    deep_inertia,
    deep_anchor,
    surface_skin_cooling_coef,
):
    """Map trained loss weights into runtime rolling/KF controls for predict mode."""
    weights = dict(initial_weights or {})
    base_controls = {
        'memory_blend': float(np.clip(memory_blend, 0.0, 1.0)),
        'surface_relaxation': float(np.clip(surface_relaxation, 0.0, 1.0)),
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
    base_controls['deep_inertia'] = float(np.clip(base_controls['deep_inertia'] + deep_inertia_shift, 0.40, 0.90))
    base_controls['deep_anchor'] = float(np.clip(base_controls['deep_anchor'] + deep_anchor_shift, 0.02, 0.22))
    base_controls['surface_skin_cooling_coef'] = float(
        np.clip(base_controls['surface_skin_cooling_coef'] + skin_cooling_shift, 0.008, 0.045)
    )
    return base_controls


def compute_online_proxy_summary(current_profile, previous_profile, day_idx, df, depths, control_params, kalman_scales):
    current_profile = np.asarray(current_profile, dtype=np.float64)
    previous_profile = None if previous_profile is None else np.asarray(previous_profile, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)

    surface_target = compute_runtime_surface_target(
        df=df,
        day_idx=day_idx,
        runtime_skin_cooling_coef=control_params.get('surface_skin_cooling_coef', SURFACE_SKIN_COOLING_COEF),
    )
    signed_surface_error = float(current_profile[0] - surface_target)
    surface_mismatch = abs(signed_surface_error)
    warm_surface_penalty = float(max(signed_surface_error, 0.0))
    increment_mag = 0.0 if previous_profile is None else float(np.mean(np.abs(current_profile - previous_profile)))
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else surface_target
    prev_raw_lst = (
        float(df['LST_surface_C'].iloc[day_idx - 1])
        if ('LST_surface_C' in df.columns and day_idx > 0)
        else raw_lst_surface
    )
    density_profile = water_density_numpy(current_profile)
    instability = float(np.mean(np.clip(-np.diff(density_profile), 0.0, None))) if len(current_profile) > 1 else 0.0

    mixed_layer_depth = float(df['MixedLayerDepth_m'].iloc[day_idx]) if 'MixedLayerDepth_m' in df.columns else 2.0
    wind_speed = float(df['wind_speed_m_per_s'].iloc[day_idx]) if 'wind_speed_m_per_s' in df.columns else 2.0
    solar_flux = float(df['Solar_W_m2'].iloc[day_idx]) if 'Solar_W_m2' in df.columns else 0.0
    deep_floor = max(mixed_layer_depth + 2.5, float(depths[-1]) * 0.45)
    anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
    deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
    deep_excess = np.maximum(current_profile - deep_anchor_profile, 0.0)
    deep_warm_excess = float(np.mean(anchor_mask * deep_excess))
    deep_warm_peak = float(np.max(anchor_mask * deep_excess))
    stratification_gap = float(max(current_profile[0] - current_profile[-1], 0.0))
    doy = int(pd.to_datetime(df['Date'].iloc[day_idx]).dayofyear) if 'Date' in df.columns else int(day_idx + 1)
    air_temp = float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else 10.0
    lst_jump = abs(raw_lst_surface - prev_raw_lst)
    skin_bulk_gap = abs(raw_lst_surface - surface_target)
    lst_spike_indicator = float(
        np.clip((lst_jump - 2.5) / 4.0, 0.0, 1.5) * (0.45 + 0.55 * np.exp(-np.clip(wind_speed, 0.0, 12.0) / 2.5))
        + 0.35 * np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
        + 0.20 * np.clip((solar_flux - 120.0) / 250.0, 0.0, 1.0)
        + 0.15 * np.clip((abs(raw_lst_surface - air_temp) - 4.0) / 8.0, 0.0, 1.0)
    )

    deep_ref_depth = float(min(9.0, max(5.0, float(depths[-1]) * 0.45)))
    deep_ref_temp = float(np.interp(deep_ref_depth, depths, current_profile))
    summer_9m_temp = deep_ref_temp
    summer_bottom_temp = float(current_profile[-1])
    summer_stratification_penalty = 0.0
    summer_thermocline_thickness_penalty = 0.0
    summer_surface_warming_reward = 0.0
    summer_midlayer_temp_reward = 0.0
    if 160 <= doy <= 250:
        summer_gap = float(current_profile[0] - deep_ref_temp)
        summer_stratification_penalty = float(max(6.0 - summer_gap, 0.0) / 6.0)
        summer_surface_warming_reward = float(np.clip((current_profile[0] - 18.0) / 6.0, 0.0, 1.5))
        mid_ref_depth = float(min(6.0, max(4.0, float(depths[-1]) * 0.30)))
        mid_ref_temp = float(np.interp(mid_ref_depth, depths, current_profile))
        summer_midlayer_temp_reward = float(np.clip(1.0 - abs(mid_ref_temp - 13.0) / 5.0, 0.0, 1.0))

    summer_thermocline_depth_norm = 0.0
    if depths.size >= 2:
        depth_mids = 0.5 * (depths[:-1] + depths[1:])
        gradients = np.diff(current_profile) / np.maximum(np.diff(depths), 1e-6)
        thermo_band = (depth_mids >= 1.0) & (depth_mids <= min(12.0, float(depths[-1]) - 0.5))
        if np.any(thermo_band):
            thermo_strength = -gradients[thermo_band]
            if thermo_strength.size > 0:
                thermo_depth = float(depth_mids[thermo_band][int(np.argmax(thermo_strength))])
                summer_thermocline_depth_norm = thermo_depth / max(float(depths[-1]), 1.0)
                positive_strength = np.clip(thermo_strength, 0.0, None)
                peak_strength = float(np.max(positive_strength))
                if peak_strength <= 1.0e-6:
                    summer_thermocline_thickness_penalty = 1.0
                else:
                    norm_weights = positive_strength / np.maximum(np.sum(positive_strength), 1.0e-8)
                    mean_depth = float(np.sum(depth_mids[thermo_band] * norm_weights))
                    std_depth = float(np.sqrt(np.sum(((depth_mids[thermo_band] - mean_depth) ** 2) * norm_weights)))
                    summer_thermocline_thickness_penalty = float(max(std_depth - 1.2, 0.0) / 1.2)

    autumn_overturn_penalty = 0.0
    autumn_surface_cooling_rate = 0.0
    autumn_gap_collapse = 0.0
    autumn_false_overturn_penalty = 0.0
    autumn_cooling_triggered_overturn_reward = 0.0
    if 280 <= doy <= 330 and air_temp <= 15.0:
        autumn_gap = float(abs(current_profile[0] - current_profile[-1]))
        autumn_overturn_penalty = float(max(autumn_gap - 1.2, 0.0))
        if previous_profile is not None:
            prev_gap = float(max(previous_profile[0] - previous_profile[-1], 0.0))
            current_gap = float(max(current_profile[0] - current_profile[-1], 0.0))
            autumn_surface_cooling_rate = float(max(previous_profile[0] - current_profile[0], 0.0))
            cooling_gate = float(np.clip(autumn_surface_cooling_rate / 0.5, 0.0, 1.0))
            autumn_gap_collapse = float(max(prev_gap - current_gap, 0.0) * cooling_gate)
            deep_warming = float(max(current_profile[-1] - previous_profile[-1], 0.0))
            false_collapse = float(max(max(prev_gap - current_gap, 0.0) - 1.25 * autumn_surface_cooling_rate, 0.0))
            false_warming = float(max(deep_warming - 0.08, 0.0))
            autumn_false_overturn_penalty = false_collapse + 1.5 * false_warming
            autumn_cooling_triggered_overturn_reward = float(
                np.clip(autumn_surface_cooling_rate / 0.6, 0.0, 1.5)
                * np.clip(autumn_gap_collapse / 0.8, 0.0, 1.5)
                * np.exp(-2.5 * (false_collapse + false_warming))
            )

    winter_inverse_penalty = 0.0
    winter_bottom_4c_error = 0.0
    if (doy <= 75 or doy >= 335) and air_temp <= 6.0:
        inverse_gap = float(current_profile[-1] - current_profile[0])
        winter_bottom_4c_error = float(abs(current_profile[-1] - 4.0))
        winter_inverse_penalty = float(max(1.5 - inverse_gap, 0.0) / 1.5 + abs(current_profile[-1] - 4.0) / 4.0)

    deep_smoothness_penalty = 0.0
    deep_mask = depths >= min(10.0, float(depths[-1]) * 0.55)
    deep_indices = np.where(deep_mask)[0]
    if deep_indices.size >= 3:
        deep_segment = current_profile[deep_indices]
        deep_smoothness_penalty = float(np.mean(np.abs(np.diff(deep_segment, n=2))))

    proxy_total = (
        surface_mismatch ** 2
        + 0.5 * increment_mag
        + 8.0 * instability
        + 3.0 * deep_warm_excess ** 2
        + 1.5 * deep_warm_peak ** 2
        + 1.5 * warm_surface_penalty ** 2
        + 2.5 * summer_stratification_penalty
        + 3.6 * summer_thermocline_thickness_penalty
        + 0.55 * max(summer_9m_temp - 14.0, 0.0)
        + 0.40 * max(summer_bottom_temp - 7.5, 0.0)
        + 2.2 * autumn_overturn_penalty
        + 1.8 * max(0.35 - autumn_surface_cooling_rate, 0.0)
        + 2.0 * max(0.50 - autumn_gap_collapse, 0.0)
        + 4.0 * autumn_false_overturn_penalty
        + 1.8 * winter_inverse_penalty
        + 0.8 * winter_bottom_4c_error
        + 1.5 * deep_smoothness_penalty
        + 0.8 * lst_spike_indicator
        - 1.6 * summer_surface_warming_reward
        - 1.2 * summer_midlayer_temp_reward
        - 1.6 * autumn_cooling_triggered_overturn_reward
    )
    return {
        'loss_pde': max(instability, 1e-8),
        'loss_bc': max(surface_mismatch ** 2 + 1.5 * warm_surface_penalty ** 2, 1e-8),
        'loss_ic': max(increment_mag, 1e-8),
        'loss_obs': max(surface_mismatch ** 2 + 2.0 * warm_surface_penalty ** 2, 1e-8),
        'total': max(proxy_total, 1e-8),
        'kappa_mean': float(max(kalman_scales['process'], 1e-8)),
        'ri_mean': float(stratification_gap),
        'surface_rmse': float(surface_mismatch),
        'warm_surface_bias': float(warm_surface_penalty),
        'instability_penalty': float(instability),
        'deep_warm_penalty': float(deep_warm_excess),
        'summer_stratification_penalty': float(summer_stratification_penalty),
        'summer_thermocline_depth_norm': float(summer_thermocline_depth_norm),
        'summer_thermocline_thickness_penalty': float(summer_thermocline_thickness_penalty),
        'summer_surface_warming_reward': float(summer_surface_warming_reward),
        'summer_midlayer_temp_reward': float(summer_midlayer_temp_reward),
        'summer_9m_temp': float(summer_9m_temp),
        'summer_bottom_temp': float(summer_bottom_temp),
        'autumn_overturn_penalty': float(autumn_overturn_penalty),
        'autumn_surface_cooling_rate': float(autumn_surface_cooling_rate),
        'autumn_gap_collapse': float(autumn_gap_collapse),
        'autumn_false_overturn_penalty': float(autumn_false_overturn_penalty),
        'autumn_cooling_triggered_overturn_reward': float(autumn_cooling_triggered_overturn_reward),
        'winter_inverse_penalty': float(winter_inverse_penalty),
        'winter_bottom_4c_error': float(winter_bottom_4c_error),
        'deep_smoothness_penalty': float(deep_smoothness_penalty),
        'lst_spike_indicator': float(lst_spike_indicator),
    }


def compute_online_proxy_validation(
    current_profile,
    day_idx,
    df,
    runtime_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    base_surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    surface_target = compute_runtime_surface_target(
        df=df,
        day_idx=day_idx,
        runtime_skin_cooling_coef=runtime_skin_cooling_coef,
        base_surface_skin_cooling_coef=base_surface_skin_cooling_coef,
    )
    surface_error = float(current_profile[0] - surface_target)
    warm_surface_bias = float(max(surface_error, 0.0))
    raw_lst_surface = float(df['LST_surface_C'].iloc[day_idx]) if 'LST_surface_C' in df.columns else surface_target
    prev_raw_lst = (
        float(df['LST_surface_C'].iloc[day_idx - 1])
        if ('LST_surface_C' in df.columns and day_idx > 0)
        else raw_lst_surface
    )
    wind_speed = float(df['wind_speed_m_per_s'].iloc[day_idx]) if 'wind_speed_m_per_s' in df.columns else 2.0
    solar_flux = float(df['Solar_W_m2'].iloc[day_idx]) if 'Solar_W_m2' in df.columns else 0.0
    air_temp = float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else surface_target
    lst_jump = abs(raw_lst_surface - prev_raw_lst)
    skin_bulk_gap = abs(raw_lst_surface - surface_target)
    lst_spike_indicator = float(
        np.clip((lst_jump - 2.5) / 4.0, 0.0, 1.5) * (0.45 + 0.55 * np.exp(-np.clip(wind_speed, 0.0, 12.0) / 2.5))
        + 0.35 * np.clip((skin_bulk_gap - 0.8) / 3.0, 0.0, 1.5)
        + 0.20 * np.clip((solar_flux - 120.0) / 250.0, 0.0, 1.0)
        + 0.15 * np.clip((abs(raw_lst_surface - air_temp) - 4.0) / 8.0, 0.0, 1.0)
    )
    return {
        'rmse': abs(surface_error) + 1.5 * warm_surface_bias,
        'mae': abs(surface_error) + 1.2 * warm_surface_bias,
        'bias': surface_error,
        'surface_rmse': abs(surface_error),
        'warm_surface_bias': warm_surface_bias,
        'lst_spike_indicator': lst_spike_indicator,
    }


def apply_online_ppo_action(control_params, kalman_scales, action):
    action = np.asarray(action, dtype=np.float32)[:PPO_ONLINE_ACTION_DIM]
    updated_controls = dict(control_params)
    updated_scales = dict(kalman_scales)

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
    updated_scales['process'] = update_control_value(kalman_scales['process'], action[5], 0.6, 2.0, step_size=0.06)
    updated_scales['obs'] = update_control_value(kalman_scales['obs'], action[6], 0.8, 3.0, step_size=0.06)

    return updated_controls, updated_scales


def train_model(
    df,
    metadata,
    max_depth=25.0,
    epochs=2500,
    lr=1e-3,
    collocation_points=512,
    device='cpu',
    train_profile_obs=None,
    ppo_validation_profile_obs=None,
    use_ppo=False,
    ppo_control_interval=50,
    ppo_rollout_steps=8,
    ppo_max_updates_run=None,
    ppo_eval_depth_points=80,
    ppo_use_kalman_reward=False,
    ppo_apply_post_physics=False,
    base_kalman_process_std=0.3,
    base_kalman_obs_std_surface=0.5,
    base_kalman_obs_std_bottom=0.5,
    base_kalman_obs_std_profile=0.75,
    base_kalman_correlation_length=2.0,
    base_kalman_forecast_blend=0.2,
    base_kalman_forecast_spinup_days=0,
    base_kalman_forecast_spinup_max_blend=0.9,
    shallow_optimized_grid=False,
    shallow_focus_depth=5.0,
    shallow_grid_fraction=0.55,
    rolling_prediction_mode=False,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    shortwave_attenuation_coef=SHORTWAVE_ATTENUATION,
    shortwave_surface_fraction=SHORTWAVE_SURFACE_FRACTION,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    initial_condition_mode='uniform_4c',
    surface_obs_depth_m=0.35,
    time_continuity_weight=5.0,
    time_continuity_depth_points=64,
    stratification_weight=0.6,
    stratification_pairs=64,
    stratification_margin_c=STRATIFICATION_MARGIN_C,
    smoothness_weight=0.15,
    max_vertical_gradient_c_per_m=MAX_VERTICAL_GRADIENT_C_PER_M,
    deep_warming_weight=0.25,
    deep_anchor_weight=0.7,
    deep_anchor_pairs=64,
    deep_anchor_amplitude_c=2.2,
    vertical_exchange_weight=0.35,
    entrainment_velocity_scale_m_per_day=MAX_ENTRAINMENT_VELOCITY_M_PER_DAY,
    convective_mixing_weight=0.25,
    autumn_overturn_weight=0.22,
    heat_budget_weight=0.30,
    heat_budget_depth_points=24,
    train_until_best=False,
    train_min_epochs=200,
    train_patience_windows=6,
    resume_checkpoint_bundle=None,
):
    model = LakePINN(hidden_dim=128, hidden_layers=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250)

    weights = {
        'pde': 1.0e5,
        'bc': 10.0,
        'ic': 5.0,
        'obs': 1.0,
        'time_continuity': float(time_continuity_weight),
        'stratification': float(stratification_weight),
        'smoothness': float(smoothness_weight),
        'deep_warming': float(deep_warming_weight),
        'deep_anchor': float(deep_anchor_weight),
        'vertical_exchange': float(vertical_exchange_weight),
        'convective_mixing': float(convective_mixing_weight),
        'autumn_overturn': float(autumn_overturn_weight),
        'heat_budget': float(heat_budget_weight),
        'density_reg': 0.1,
    }

    resume_training_info = {}
    if resume_checkpoint_bundle is not None:
        model_state = resume_checkpoint_bundle.get('model_state_dict')
        if model_state is not None:
            model.load_state_dict(model_state)
        optimizer_state = resume_checkpoint_bundle.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = resume_checkpoint_bundle.get('scheduler_state_dict')
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception:
                pass
        resume_training_info = dict(resume_checkpoint_bundle.get('training_info', {}) or {})
        for key, value in dict(resume_training_info.get('final_weights', {}) or {}).items():
            if key in weights:
                weights[key] = float(value)

    t_all = torch.tensor(df['time_norm'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    solar_flux = torch.tensor(df['Solar_W_m2'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    wind_speed = torch.tensor(df['wind_speed_m_per_s'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    air_temp = torch.tensor(df['T_air_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    relative_humidity = torch.tensor(df['relative_humidity'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    surface_pressure = torch.tensor(df['surface_pressure_Pa'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    mixed_layer_depth = torch.tensor(df['MixedLayerDepth_m'].values.reshape(-1, 1), dtype=torch.float32, device=device)

    observations, surface_correction_info = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=train_profile_obs,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    obs_time = torch.tensor(observations['time_norm'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_depth = torch.tensor(observations['Depth_m'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)
    obs_temperature = torch.tensor(
        observations['Temperature_C'].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    obs_weight = torch.tensor(
        observations['obs_weight'].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )

    z_ic_np, t_ic_np = build_initial_condition_profile(
        df,
        max_depth=max_depth,
        n_points=64,
        mode=initial_condition_mode,
    )
    ic_depth = torch.tensor(z_ic_np, dtype=torch.float32, device=device)
    ic_time = torch.zeros_like(ic_depth, device=device)
    ic_temperature = torch.tensor(t_ic_np, dtype=torch.float32, device=device)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    best_selection_metric = float('inf')
    best_selection_label = 'loss'
    best_snapshot = None
    validation_patience_counter = 0
    n_days = len(df)
    date_ns = pd.to_datetime(df['Date']).astype('int64').to_numpy(dtype=np.int64, copy=False)
    time_step_seconds_np = np.diff(date_ns.astype(np.float64)) / 1.0e9 if n_days > 1 else np.empty((0,), dtype=np.float64)
    seq_delta_seconds_all = torch.tensor(
        time_step_seconds_np.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    ) if n_days > 1 else None
    window_losses = []

    kalman_scales = {'process': 1.0, 'obs': 1.0}
    for key, value in dict(resume_training_info.get('kalman_scales', {}) or {}).items():
        if key in kalman_scales:
            kalman_scales[key] = float(value)
    ppo_history = []
    ppo_update_stats = []
    ppo_update_count = 0
    ppo_context = {
        'state': None,
        'action': None,
        'log_prob': None,
        'value': None,
        'summary': None,
        'validation': None,
    }
    ppo_controller = None
    ppo_tune_kalman = False
    if use_ppo:
        ppo_controller = PPOController(state_dim=PPO_STATE_DIM, action_dim=PPO_TRAIN_ACTION_DIM, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_progress = epoch / max(epochs - 1, 1)
        effective_weights = build_annealed_loss_weights(weights, epoch_progress)

        day_pick = torch.randint(low=0, high=n_days, size=(collocation_points,), device=device)
        z_colloc = torch.rand((collocation_points, 1), device=device) * max_depth
        if n_days > 1:
            seq_batch_size = int(max(8, min(collocation_points, time_continuity_depth_points)))
            seq_pick = torch.randint(low=0, high=n_days - 1, size=(seq_batch_size,), device=device)
            seq_depth = torch.rand((seq_batch_size, 1), device=device) * max_depth
            seq_time_now = t_all[seq_pick]
            seq_time_next = t_all[seq_pick + 1]
            seq_solar_flux_now = solar_flux[seq_pick]
            seq_wind_now = wind_speed[seq_pick]
            seq_air_temp_now = air_temp[seq_pick]
            seq_doy_now = torch.tensor(
                pd.to_datetime(df['Date'].iloc[seq_pick.detach().cpu().numpy()]).dt.dayofyear.to_numpy(dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            )
            seq_mld_now = mixed_layer_depth[seq_pick]
            seq_mld_next = mixed_layer_depth[seq_pick + 1]
            seq_deep_floor = torch.clamp(
                torch.maximum(seq_mld_now + 2.0, torch.full_like(seq_mld_now, max_depth * 0.42)),
                min=3.0,
                max=max_depth - 0.5,
            )
            seq_delta_seconds = seq_delta_seconds_all[seq_pick]
            seq_surface_depth = torch.zeros_like(seq_time_now)
        else:
            seq_depth = torch.empty((0, 1), dtype=torch.float32, device=device)
            seq_time_now = None
            seq_time_next = None
            seq_solar_flux_now = None
            seq_wind_now = None
            seq_air_temp_now = None
            seq_doy_now = None
            seq_mld_now = None
            seq_mld_next = None
            seq_deep_floor = None
            seq_delta_seconds = None
            seq_surface_depth = None

        strat_batch_size = int(max(8, min(collocation_points, stratification_pairs)))
        strat_pick = torch.randint(low=0, high=n_days, size=(strat_batch_size,), device=device)
        strat_time = t_all[strat_pick]
        strat_air_temp = air_temp[strat_pick]
        strat_solar = solar_flux[strat_pick]
        strat_mld = mixed_layer_depth[strat_pick]
        shallow_cap = torch.clamp(torch.maximum(torch.full_like(strat_mld, 0.8), 0.55 * strat_mld + 0.6), min=0.8, max=max_depth * 0.35)
        strat_shallow_depth = torch.rand((strat_batch_size, 1), device=device) * shallow_cap
        deep_floor = torch.clamp(torch.maximum(strat_mld + 1.5, torch.full_like(strat_mld, max_depth * 0.35)), min=2.5, max=max_depth - 0.5)
        deep_span = torch.clamp(max_depth - deep_floor, min=0.5)
        strat_deep_depth = deep_floor + torch.rand((strat_batch_size, 1), device=device) * deep_span
        warm_indicator = torch.sigmoid((strat_air_temp - 8.0) / 2.0)
        solar_indicator = torch.sigmoid((strat_solar - 120.0) / 40.0)
        mld_indicator = torch.sigmoid((strat_mld - 1.5) / 0.8)
        strat_weight = warm_indicator * solar_indicator * mld_indicator
        strat_margin = torch.full_like(strat_weight, float(stratification_margin_c))

        deep_batch_size = int(max(8, min(collocation_points, deep_anchor_pairs)))
        deep_pick = torch.randint(low=0, high=n_days, size=(deep_batch_size,), device=device)
        deep_time = t_all[deep_pick]
        deep_air_temp = air_temp[deep_pick]
        deep_solar = solar_flux[deep_pick]
        deep_mld = mixed_layer_depth[deep_pick]
        deep_floor = torch.clamp(
            torch.maximum(deep_mld + 2.5, torch.full_like(deep_mld, max_depth * 0.45)),
            min=4.0,
            max=max_depth - 0.5,
        )
        deep_span = torch.clamp(max_depth - deep_floor, min=0.5)
        deep_depth = deep_floor + torch.rand((deep_batch_size, 1), device=device) * deep_span
        deep_anchor_scale = torch.exp(-(deep_depth - deep_floor) / 4.0)
        deep_target = torch.full_like(deep_depth, DEFAULT_INITIAL_WATER_TEMPERATURE_C) + float(deep_anchor_amplitude_c) * deep_anchor_scale
        deep_warm_indicator = torch.sigmoid((deep_air_temp - 9.0) / 2.0) * torch.sigmoid((deep_solar - 140.0) / 45.0)
        deep_stability_indicator = torch.sigmoid((deep_mld - 1.5) / 0.8)
        deep_weight = deep_warm_indicator * deep_stability_indicator

        if n_days > 1:
            budget_pick = seq_pick
            budget_batch_size = int(budget_pick.shape[0])
            depth_line = torch.linspace(0.0, max_depth, int(max(8, heat_budget_depth_points)), device=device).reshape(-1, 1)
            budget_depth = depth_line.repeat(budget_batch_size, 1)
            budget_time_now = t_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_time_next = t_all[budget_pick + 1].repeat_interleave(depth_line.shape[0], dim=0)
            budget_delta_seconds = seq_delta_seconds_all[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_solar_flux = solar_flux[budget_pick].repeat_interleave(depth_line.shape[0], dim=0)
            budget_surface_time = t_all[budget_pick]
            budget_air_temp = air_temp[budget_pick]
            budget_wind_speed = wind_speed[budget_pick]
            budget_relative_humidity = relative_humidity[budget_pick]
            budget_surface_pressure = surface_pressure[budget_pick]
            budget_group_index = torch.arange(budget_batch_size, device=device).repeat_interleave(depth_line.shape[0])
            budget_dz_value = float(max_depth) / float(max(int(depth_line.shape[0]) - 1, 1))
            budget_dz = torch.full_like(budget_depth, budget_dz_value)
            budget_surface_depth = torch.zeros_like(budget_surface_time)
        else:
            budget_depth = None
            budget_time_now = None
            budget_time_next = None
            budget_delta_seconds = None
            budget_solar_flux = None
            budget_surface_time = None
            budget_air_temp = None
            budget_wind_speed = None
            budget_relative_humidity = None
            budget_surface_pressure = None
            budget_group_index = None
            budget_dz = None
            budget_surface_depth = None

        batch = {
            't_colloc': t_all[day_pick],
            'z_colloc': z_colloc,
            'solar_flux_colloc': solar_flux[day_pick],
            'wind_colloc': wind_speed[day_pick],
            'surface_time': t_all,
            'surface_shortwave': solar_flux,
            'surface_air_temp': air_temp,
            'surface_wind_speed': wind_speed,
            'surface_relative_humidity': relative_humidity,
            'surface_pressure': surface_pressure,
            'ic_time': ic_time,
            'ic_depth': ic_depth,
            'ic_temperature': ic_temperature,
            'obs_time': obs_time,
            'obs_depth': obs_depth,
            'obs_temperature': obs_temperature,
            'obs_weight': obs_weight,
            'seq_time_now': seq_time_now,
            'seq_time_next': seq_time_next,
            'seq_depth': seq_depth,
            'seq_solar_flux_now': seq_solar_flux_now,
            'seq_wind_now': seq_wind_now,
            'seq_air_temp_now': seq_air_temp_now,
            'seq_doy_now': seq_doy_now,
            'seq_mld_now': seq_mld_now,
            'seq_mld_next': seq_mld_next,
            'seq_deep_floor': seq_deep_floor,
            'seq_delta_seconds': seq_delta_seconds,
            'seq_surface_depth': seq_surface_depth,
            'strat_time': strat_time,
            'strat_shallow_depth': strat_shallow_depth,
            'strat_deep_depth': strat_deep_depth,
            'strat_weight': strat_weight,
            'strat_margin': strat_margin,
            'deep_anchor_time': deep_time,
            'deep_anchor_depth': deep_depth,
            'deep_anchor_target': deep_target,
            'deep_anchor_weight': deep_weight,
            'budget_depth': budget_depth,
            'budget_time_now': budget_time_now,
            'budget_time_next': budget_time_next,
            'budget_delta_seconds': budget_delta_seconds,
            'budget_solar_flux': budget_solar_flux,
            'budget_surface_time': budget_surface_time,
            'budget_air_temp': budget_air_temp,
            'budget_wind_speed': budget_wind_speed,
            'budget_relative_humidity': budget_relative_humidity,
            'budget_surface_pressure': budget_surface_pressure,
            'budget_group_index': budget_group_index,
            'budget_dz': budget_dz,
            'budget_surface_depth': budget_surface_depth,
        }

        if budget_surface_time is not None:
            budget_surface_temp = model_temperature(model, budget_surface_time, budget_surface_depth, max_depth)
            budget_flux_terms = compute_surface_flux_terms(
                budget_surface_temp,
                {
                    'surface_air_temp': budget_air_temp,
                    'surface_wind_speed': budget_wind_speed,
                    'surface_relative_humidity': budget_relative_humidity,
                    'surface_pressure': budget_surface_pressure,
                    'surface_shortwave': solar_flux[budget_pick],
                },
                shortwave_surface_fraction=shortwave_surface_fraction,
            )
            batch['budget_surface_flux'] = budget_flux_terms['seb_flux']
        else:
            batch['budget_surface_flux'] = None

        losses = compute_losses(
            model=model,
            batch=batch,
            max_depth=max_depth,
            time_scale_seconds=metadata['time_scale_seconds'],
            weights=effective_weights,
            shortwave_attenuation=shortwave_attenuation_coef,
            shortwave_surface_fraction=shortwave_surface_fraction,
            max_vertical_gradient_c_per_m=max_vertical_gradient_c_per_m,
            entrainment_velocity_scale_m_per_day=entrainment_velocity_scale_m_per_day,
        )
        losses['total'].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(losses['total'].detach())
        if use_ppo:
            window_losses.append(
                {
                    'total': losses['total'].item(),
                    'loss_pde': losses['loss_pde'].item(),
                    'loss_bc': losses['loss_bc'].item(),
                        'loss_ic': losses['loss_ic'].item(),
                        'loss_obs': losses['loss_obs'].item(),
                        'loss_time_continuity': losses['loss_time_continuity'].item(),
                        'loss_stratification': losses['loss_stratification'].item(),
                        'loss_smoothness': losses['loss_smoothness'].item(),
                        'loss_deep_warming': losses['loss_deep_warming'].item(),
                    'loss_deep_anchor': losses['loss_deep_anchor'].item(),
                    'loss_vertical_exchange': losses['loss_vertical_exchange'].item(),
                    'loss_convective_mixing': losses['loss_convective_mixing'].item(),
                    'loss_autumn_overturn': losses['loss_autumn_overturn'].item(),
                    'loss_heat_budget': losses['loss_heat_budget'].item(),
                    'kappa_mean': losses['kappa_mean'].item(),
                    'ri_mean': losses['ri_mean'].item(),
                    }
            )

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | total={losses['total'].item():.4f} | "
                f"pde={losses['loss_pde'].item():.6e} | bc={losses['loss_bc'].item():.6e} | "
                f"ic={losses['loss_ic'].item():.4f} | obs={losses['loss_obs'].item():.4f} | "
                f"tc={losses['loss_time_continuity'].item():.4f} | strat={losses['loss_stratification'].item():.4f} | smooth={losses['loss_smoothness'].item():.4f} | deepwarm={losses['loss_deep_warming'].item():.4f} | deep={losses['loss_deep_anchor'].item():.4f} | "
                f"adv={losses['loss_vertical_exchange'].item():.4f} | conv={losses['loss_convective_mixing'].item():.4f} | overturn={losses['loss_autumn_overturn'].item():.4f} | heat={losses['loss_heat_budget'].item():.4f} | "
                f"kappa={losses['kappa_mean'].item():.3e} | Ri={losses['ri_mean'].item():.3f} | "
                f"obs_w={effective_weights['obs']:.3f} | strat_w={effective_weights['stratification']:.3f} | smooth_w={effective_weights['smoothness']:.3f} | deepwarm_w={effective_weights['deep_warming']:.3f} | deep_w={effective_weights['deep_anchor']:.3f} | "
                f"adv_w={effective_weights['vertical_exchange']:.3f} | conv_w={effective_weights['convective_mixing']:.3f} | overturn_w={effective_weights['autumn_overturn']:.3f} | heat_w={effective_weights['heat_budget']:.3f}"
            )

        reached_window_end = ((epoch + 1) % max(ppo_control_interval, 1) == 0) or (epoch == epochs - 1)
        if use_ppo and reached_window_end:
            current_summary = summarize_window_losses(window_losses)
            eval_depth_points = int(max(20, min(ppo_eval_depth_points, 160)))
            eval_grid, eval_depths, _ = predict_temperature_grid(
                model,
                df=df,
                max_depth=max_depth,
                n_depth_points=eval_depth_points,
                device=device,
                apply_post_physics=ppo_apply_post_physics,
                use_shallow_optimized=shallow_optimized_grid,
                shallow_focus_depth=shallow_focus_depth,
                shallow_fraction=shallow_grid_fraction,
                rolling_prediction_mode=rolling_prediction_mode,
                rolling_memory_blend=rolling_memory_blend,
                rolling_surface_relaxation=rolling_surface_relaxation,
                rolling_surface_decay_depth=rolling_surface_decay_depth,
                surface_skin_cooling_coef=surface_skin_cooling_coef,
            )
            current_validation = evaluate_blind_ppo_proxy(
                df=df,
                temp_grid=eval_grid,
                depths=eval_depths,
            )
            selection_metrics = None
            if has_profile_observations(ppo_validation_profile_obs):
                selection_metrics = evaluate_profile_grid(
                    df=df,
                    metadata=metadata,
                    temp_grid=eval_grid,
                    depths=eval_depths,
                    max_depth=max_depth,
                    profile_obs_data=ppo_validation_profile_obs,
                )

            current_state = build_ppo_state(
                summary=current_summary,
                weights=effective_weights,
                kalman_scales=kalman_scales,
                learning_rate=optimizer.param_groups[0]['lr'],
                validation_metrics=current_validation,
            )

            if ppo_context['state'] is not None:
                reward = compute_ppo_reward(
                    prev_summary=ppo_context['summary'],
                    current_summary=current_summary,
                    prev_validation_metrics=ppo_context['validation'],
                    current_validation_metrics=current_validation,
                )
                done = bool(epoch == epochs - 1)
                ppo_controller.store_transition(
                    state=ppo_context['state'],
                    action=ppo_context['action'],
                    log_prob=ppo_context['log_prob'],
                    reward=reward,
                    done=done,
                    value=ppo_context['value'],
                )
                ppo_history.append(
                    {
                        'epoch': epoch,
                        'reward': reward,
                        'lambda_pde': effective_weights['pde'],
                        'lambda_bc': effective_weights['bc'],
                        'lambda_ic': effective_weights['ic'],
                        'lambda_obs': effective_weights['obs'],
                        'lambda_time_continuity': effective_weights['time_continuity'],
                        'lambda_stratification': effective_weights['stratification'],
                        'lambda_smoothness': effective_weights['smoothness'],
                        'lambda_deep_warming': effective_weights['deep_warming'],
                        'lambda_deep_anchor': effective_weights['deep_anchor'],
                        'kalman_process_scale': kalman_scales['process'],
                        'kalman_obs_scale': kalman_scales['obs'],
                        'window_total': current_summary['total'],
                        'window_obs': current_summary['loss_obs'],
                        'validation_rmse': np.nan if current_validation is None else current_validation['rmse'],
                    }
                )

                if done or len(ppo_controller.buffer['states']) >= max(ppo_rollout_steps, 1):
                    reached_update_cap = (
                        ppo_max_updates_run is not None
                        and ppo_update_count >= int(max(ppo_max_updates_run, 0))
                    )
                    if reached_update_cap:
                        ppo_controller.reset_buffer()
                    else:
                        update_stats = ppo_controller.update(last_state=current_state, last_done=done)
                        if update_stats is not None:
                            update_stats['epoch'] = epoch
                            ppo_update_count += 1
                            update_stats['update_index'] = ppo_update_count
                            ppo_update_stats.append(update_stats)

            if train_until_best:
                selection_metric = None
                selection_label = None
                if selection_metrics is not None and np.isfinite(selection_metrics.get('rmse', np.nan)):
                    selection_metric = float(selection_metrics['rmse'])
                    selection_label = 'val_rmse'
                elif current_validation is not None and np.isfinite(current_validation.get('rmse', np.nan)):
                    selection_metric = float(current_validation['rmse'])
                    selection_label = 'blind_rmse'
                if selection_metric is not None:
                    if selection_metric < best_selection_metric - 1e-6:
                        best_selection_metric = selection_metric
                        best_selection_label = selection_label
                        validation_patience_counter = 0
                        best_snapshot = {
                            'model_state': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            'weights': dict(weights),
                            'kalman_scales': dict(kalman_scales),
                            'ppo_state_dict': None if ppo_controller is None else copy.deepcopy(ppo_controller.model.state_dict()),
                        }
                    elif epoch + 1 >= int(max(train_min_epochs, 1)):
                        validation_patience_counter += 1

            if epoch != epochs - 1:
                action, log_prob, value = ppo_controller.select_action(current_state)
                weights, kalman_scales = apply_ppo_action(
                    weights,
                    kalman_scales,
                    action,
                    tune_kalman=ppo_tune_kalman,
                )
                ppo_context = {
                    'state': current_state,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'summary': current_summary,
                    'validation': current_validation,
                }

            window_losses = []
            if train_until_best and epoch + 1 >= int(max(train_min_epochs, 1)) and validation_patience_counter >= int(max(train_patience_windows, 1)):
                print(
                    f"Validation early stopping at epoch {epoch} | "
                    f"best {best_selection_label}={best_selection_metric:.4f}"
                )
                break

        total_value = losses['total'].item()
        if total_value < best_loss:
            best_loss = total_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 800:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_snapshot is not None:
        model.load_state_dict(best_snapshot['model_state'])
        weights = dict(best_snapshot['weights'])
        kalman_scales = dict(best_snapshot['kalman_scales'])
        if ppo_controller is not None and best_snapshot['ppo_state_dict'] is not None:
            ppo_controller.model.load_state_dict(best_snapshot['ppo_state_dict'])

    training_info = {
        'final_weights': dict(weights),
        'kalman_scales': dict(kalman_scales),
        'ppo_history': pd.DataFrame(ppo_history),
        'ppo_update_stats': pd.DataFrame(ppo_update_stats),
        'use_ppo': bool(use_ppo),
        'ppo_policy_bundle': export_ppo_policy_bundle(ppo_controller if use_ppo else None, weights, kalman_scales),
        'surface_correction_info': surface_correction_info,
        'best_selection_metric': None if not np.isfinite(best_selection_metric) else float(best_selection_metric),
        'best_selection_label': best_selection_label,
        'ppo_update_count': int(ppo_update_count),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'scheduler_state_dict': copy.deepcopy(scheduler.state_dict()),
    }
    return model, training_info


def train_pure_forecast_ppo_policy(
    model,
    df,
    metadata,
    max_depth,
    depth_points,
    device,
    validation_profile_obs,
    initial_weights,
    initial_kalman_scales,
    apply_post_physics=False,
    use_shallow_optimized=False,
    shallow_focus_depth=5.0,
    shallow_fraction=0.55,
    rolling_prediction_mode=True,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    ppo_control_interval=7,
    ppo_rollout_steps=4,
    ppo_max_updates_run=None,
    max_episodes=None,
    initial_ppo_policy_bundle=None,
):
    if not has_profile_observations(validation_profile_obs):
        return None

    ppo_controller = PPOController(state_dim=PPO_STATE_DIM, action_dim=PPO_ONLINE_ACTION_DIM, device=device)
    if initial_ppo_policy_bundle is not None and int(initial_ppo_policy_bundle.get('action_dim', PPO_ONLINE_ACTION_DIM)) == PPO_ONLINE_ACTION_DIM:
        restored_controller, _ = build_ppo_controller_from_bundle(initial_ppo_policy_bundle, device=device)
        if restored_controller is not None:
            ppo_controller = restored_controller
    current_kalman_scales = dict(initial_kalman_scales)
    total_update_cap = None if ppo_max_updates_run is None else int(max(ppo_max_updates_run, 0))
    if max_episodes is None:
        max_episodes = 1 if total_update_cap is None else max(1, min(total_update_cap, 4))

    best_metric = float('inf')
    best_snapshot = None
    best_temp_grid = None
    best_depths = None
    episode_history_frames = []
    episode_diag_frames = []
    total_updates = 0

    for episode_idx in range(int(max_episodes)):
        remaining_updates = None if total_update_cap is None else max(total_update_cap - total_updates, 0)
        if total_update_cap is not None and remaining_updates <= 0:
            break

        policy_bundle = {
            'final_weights': dict(initial_weights),
            'final_kalman_scales': dict(current_kalman_scales),
        }
        temp_grid, depths, online_runtime = predict_temperature_grid(
            model,
            df=df,
            max_depth=max_depth,
            n_depth_points=depth_points,
            device=device,
            apply_post_physics=apply_post_physics,
            use_shallow_optimized=use_shallow_optimized,
            shallow_focus_depth=shallow_focus_depth,
            shallow_fraction=shallow_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            ppo_controller=ppo_controller,
            ppo_policy_bundle=policy_bundle,
            online_ppo_update=True,
            online_ppo_control_interval=ppo_control_interval,
            online_ppo_rollout_steps=ppo_rollout_steps,
            online_ppo_max_updates_run=remaining_updates,
            validation_profile_obs=validation_profile_obs,
            validation_metadata=metadata,
            validation_max_depth=max_depth,
        )
        episode_metrics = evaluate_profile_grid(
            df=df,
            metadata=metadata,
            temp_grid=temp_grid,
            depths=depths,
            max_depth=max_depth,
            profile_obs_data=validation_profile_obs,
        )
        if episode_metrics is not None and np.isfinite(episode_metrics.get('rmse', np.nan)):
            if float(episode_metrics['rmse']) < best_metric:
                best_metric = float(episode_metrics['rmse'])
                best_snapshot = {
                    'ppo_state_dict': copy.deepcopy(ppo_controller.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(ppo_controller.optimizer.state_dict()),
                    'kalman_scales': dict(online_runtime.get('kalman_scales', current_kalman_scales)),
                    'metrics': dict(episode_metrics),
                }
                best_temp_grid = temp_grid.copy()
                best_depths = depths.copy()

        diagnostics_df = online_runtime.get('diagnostics', pd.DataFrame()).copy()
        if not diagnostics_df.empty:
            diagnostics_df['episode'] = episode_idx
            episode_diag_frames.append(diagnostics_df)
            total_updates += int(diagnostics_df['ppo_update_count'].max())
            last_diag = diagnostics_df.iloc[-1]
            current_kalman_scales = {
                'process': float(last_diag['kalman_process_scale']),
                'obs': float(last_diag['kalman_obs_scale']),
            }

        history_df = online_runtime.get('history', pd.DataFrame()).copy()
        if not history_df.empty:
            history_df['episode'] = episode_idx
            episode_history_frames.append(history_df)

        if diagnostics_df.empty or int(diagnostics_df['ppo_update_count'].max()) == 0:
            break

    if best_snapshot is not None:
        ppo_controller.model.load_state_dict(best_snapshot['ppo_state_dict'])
        ppo_controller.optimizer.load_state_dict(best_snapshot['optimizer_state_dict'])
        current_kalman_scales = dict(best_snapshot['kalman_scales'])
    elif best_temp_grid is None:
        return None

    final_bundle = export_ppo_policy_bundle(
        ppo_controller,
        final_weights=initial_weights,
        final_kalman_scales=current_kalman_scales,
    )
    return {
        'ppo_controller': ppo_controller,
        'ppo_policy_bundle': final_bundle,
        'ppo_history': pd.concat(episode_history_frames, ignore_index=True) if episode_history_frames else pd.DataFrame(),
        'ppo_update_stats': pd.concat(episode_diag_frames, ignore_index=True) if episode_diag_frames else pd.DataFrame(),
        'best_validation_metrics': None if best_snapshot is None else dict(best_snapshot['metrics']),
        'kalman_scales': dict(current_kalman_scales),
        'temp_grid': best_temp_grid,
        'depths': best_depths,
        'ppo_update_count': int(total_updates),
    }


def run_seasonal_segmented_pipeline(
    df,
    metadata,
    max_depth,
    depth_points,
    epochs,
    lr,
    collocation_points,
    device,
    train_profile_obs,
    val_profile_obs,
    assim_profile_obs,
    use_kalman,
    use_ppo,
    ppo_control_interval,
    ppo_rollout_steps,
    ppo_max_updates_run,
    ppo_eval_depth_points,
    ppo_use_kalman_reward,
    ppo_apply_post_physics,
    kalman_prior_std,
    kalman_process_std,
    kalman_obs_std_surface,
    kalman_obs_std_bottom,
    kalman_obs_std_profile,
    kalman_correlation_length,
    kalman_forecast_blend,
    kalman_forecast_spinup_days,
    kalman_forecast_spinup_max_blend,
    shallow_optimized_grid,
    shallow_focus_depth,
    shallow_grid_fraction,
    rolling_prediction_mode,
    rolling_memory_blend,
    rolling_surface_relaxation,
    rolling_surface_decay_depth,
    rolling_deep_inertia,
    shortwave_attenuation_coef,
    shortwave_surface_fraction,
    use_surface_bulk_correction,
    use_bottom_observation,
    initial_condition_mode,
    surface_obs_depth_m,
    time_continuity_weight,
    time_continuity_depth_points,
    stratification_weight,
    stratification_pairs,
    stratification_margin_c,
    smoothness_weight,
    max_vertical_gradient_c_per_m,
    deep_warming_weight,
    deep_anchor_weight,
    deep_anchor_pairs,
    deep_anchor_amplitude_c,
    vertical_exchange_weight,
    entrainment_velocity_scale_m_per_day,
    convective_mixing_weight,
    autumn_overturn_weight,
    heat_budget_weight,
    heat_budget_depth_points,
    train_until_best,
    train_min_epochs,
    train_patience_windows,
    apply_post_physics,
):
    segments = build_contiguous_season_segments(df)
    if not segments:
        raise ValueError('Seasonal segmentation requires at least one time segment.')

    depth_grid = build_depth_grid(
        max_depth=max_depth,
        n_depth_points=depth_points,
        use_shallow_optimized=shallow_optimized_grid,
        shallow_focus_depth=shallow_focus_depth,
        shallow_fraction=shallow_grid_fraction,
    )
    full_temp_grid = np.zeros((len(depth_grid), len(df)), dtype=np.float32)
    full_kalman_grid = np.zeros_like(full_temp_grid) if use_kalman else None
    segment_summaries = []
    ppo_history_frames = []
    ppo_update_frames = []
    final_weights = {}
    kalman_scales = {}

    total_days = max(len(df), 1)
    for segment in segments:
        segment_df, segment_duration_seconds = build_segment_frame(df, segment['start_idx'], segment['end_idx'])
        segment_dates = pd.to_datetime(segment_df['Date']).dt.normalize().tolist()
        segment_train_obs = subset_profile_observations_by_dates(train_profile_obs, segment_dates)
        segment_val_obs = subset_profile_observations_by_dates(val_profile_obs, segment_dates)
        segment_assim_obs = subset_profile_observations_by_dates(assim_profile_obs, segment_dates)

        segment_metadata = dict(metadata)
        segment_metadata['time_scale_seconds'] = segment_duration_seconds
        segment_metadata['start_date'] = segment_df['Date'].iloc[0]
        segment_metadata['file_tag'] = f"{metadata['file_tag']}_{segment['name']}"

        segment_days = len(segment_df)
        segment_epochs = max(1, int(round(epochs * segment_days / total_days)))
        segment_collocation = max(8, int(round(collocation_points * segment_days / total_days)))
        segment_control_interval = max(1, min(ppo_control_interval, segment_epochs))

        model, training_info = train_model(
            df=segment_df,
            metadata=segment_metadata,
            max_depth=max_depth,
            epochs=segment_epochs,
            lr=lr,
            collocation_points=segment_collocation,
            device=device,
            train_profile_obs=segment_train_obs,
            ppo_validation_profile_obs=segment_val_obs,
            use_ppo=use_ppo,
            ppo_control_interval=segment_control_interval,
            ppo_rollout_steps=ppo_rollout_steps,
            ppo_max_updates_run=ppo_max_updates_run,
            ppo_eval_depth_points=min(ppo_eval_depth_points, depth_points),
            ppo_use_kalman_reward=ppo_use_kalman_reward,
            ppo_apply_post_physics=ppo_apply_post_physics,
            base_kalman_process_std=kalman_process_std,
            base_kalman_obs_std_surface=kalman_obs_std_surface,
            base_kalman_obs_std_bottom=kalman_obs_std_bottom,
            base_kalman_obs_std_profile=kalman_obs_std_profile,
            base_kalman_correlation_length=kalman_correlation_length,
            base_kalman_forecast_blend=kalman_forecast_blend,
            base_kalman_forecast_spinup_days=kalman_forecast_spinup_days,
            base_kalman_forecast_spinup_max_blend=kalman_forecast_spinup_max_blend,
            shallow_optimized_grid=shallow_optimized_grid,
            shallow_focus_depth=shallow_focus_depth,
            shallow_grid_fraction=shallow_grid_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            shortwave_attenuation_coef=shortwave_attenuation_coef,
            shortwave_surface_fraction=shortwave_surface_fraction,
            use_surface_bulk_correction=use_surface_bulk_correction,
            use_bottom_observation=use_bottom_observation,
            initial_condition_mode=initial_condition_mode,
            surface_obs_depth_m=surface_obs_depth_m,
            time_continuity_weight=time_continuity_weight,
            time_continuity_depth_points=time_continuity_depth_points,
            stratification_weight=stratification_weight,
            stratification_pairs=stratification_pairs,
            stratification_margin_c=stratification_margin_c,
            smoothness_weight=smoothness_weight,
            max_vertical_gradient_c_per_m=max_vertical_gradient_c_per_m,
            deep_warming_weight=deep_warming_weight,
            deep_anchor_weight=deep_anchor_weight,
            deep_anchor_pairs=deep_anchor_pairs,
            deep_anchor_amplitude_c=deep_anchor_amplitude_c,
            vertical_exchange_weight=vertical_exchange_weight,
            entrainment_velocity_scale_m_per_day=entrainment_velocity_scale_m_per_day,
            convective_mixing_weight=convective_mixing_weight,
            autumn_overturn_weight=autumn_overturn_weight,
            heat_budget_weight=heat_budget_weight,
            heat_budget_depth_points=heat_budget_depth_points,
            train_until_best=train_until_best,
            train_min_epochs=train_min_epochs,
            train_patience_windows=train_patience_windows,
        )
        segment_temp_grid, segment_depths, _ = predict_temperature_grid(
            model,
            df=segment_df,
            max_depth=max_depth,
            n_depth_points=depth_points,
            device=device,
            apply_post_physics=apply_post_physics,
            use_shallow_optimized=shallow_optimized_grid,
            shallow_focus_depth=shallow_focus_depth,
            shallow_fraction=shallow_grid_fraction,
            rolling_prediction_mode=rolling_prediction_mode,
            rolling_memory_blend=rolling_memory_blend,
            rolling_surface_relaxation=rolling_surface_relaxation,
            rolling_surface_decay_depth=rolling_surface_decay_depth,
            rolling_deep_inertia=rolling_deep_inertia,
            rolling_deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        if not np.allclose(segment_depths, depth_grid):
            raise ValueError('Seasonal segment depth grid mismatch encountered during stitching.')

        full_temp_grid[:, segment['start_idx']:segment['end_idx']] = segment_temp_grid

        segment_kalman_grid = None
        if use_kalman:
            segment_kalman_grid, _ = run_profile_kalman_filter(
                df=segment_df,
                temp_grid=segment_temp_grid,
                depths=segment_depths,
                metadata=segment_metadata,
                max_depth=max_depth,
                profile_obs_data=segment_assim_obs,
                prior_std=kalman_prior_std,
                process_std=kalman_process_std * training_info['kalman_scales']['process'],
                obs_std_surface=kalman_obs_std_surface * training_info['kalman_scales']['obs'],
                obs_std_bottom=kalman_obs_std_bottom * training_info['kalman_scales']['obs'],
                obs_std_profile=kalman_obs_std_profile * training_info['kalman_scales']['obs'],
                correlation_length=kalman_correlation_length,
                forecast_blend=kalman_forecast_blend,
                forecast_spinup_days=kalman_forecast_spinup_days,
                forecast_spinup_max_blend=kalman_forecast_spinup_max_blend,
                use_surface_bulk_correction=use_surface_bulk_correction,
                use_bottom_observation=use_bottom_observation,
                surface_obs_depth_m=surface_obs_depth_m,
                autumn_asymmetric_cooling=False,
                autumn_air_temp_threshold=12.0,
            )
            full_kalman_grid[:, segment['start_idx']:segment['end_idx']] = segment_kalman_grid

        final_weights = training_info['final_weights']
        kalman_scales = training_info['kalman_scales']

        if use_ppo and not training_info['ppo_history'].empty:
            history_df = training_info['ppo_history'].copy()
            history_df['segment'] = segment['name']
            ppo_history_frames.append(history_df)
        if use_ppo and not training_info['ppo_update_stats'].empty:
            update_df = training_info['ppo_update_stats'].copy()
            update_df['segment'] = segment['name']
            ppo_update_frames.append(update_df)

        segment_summaries.append(
            {
                'segment': segment['name'],
                'season': segment['season'],
                'start_date': segment['start_date'].date().isoformat(),
                'end_date': segment['end_date'].date().isoformat(),
                'days': int(segment_days),
                'epochs': int(segment_epochs),
                'collocation_points': int(segment_collocation),
                'train_obs_rows': int(len(segment_train_obs)),
                'val_obs_rows': int(len(segment_val_obs)),
                'assim_obs_rows': int(len(segment_assim_obs)),
            }
        )

    return {
        'temp_grid': full_temp_grid,
        'kalman_grid': full_kalman_grid,
        'depths': depth_grid.astype(np.float32),
        'training_info': {
            'final_weights': dict(final_weights),
            'kalman_scales': dict(kalman_scales),
            'ppo_history': pd.concat(ppo_history_frames, ignore_index=True) if ppo_history_frames else pd.DataFrame(),
            'ppo_update_stats': pd.concat(ppo_update_frames, ignore_index=True) if ppo_update_frames else pd.DataFrame(),
            'use_ppo': bool(use_ppo),
            'surface_correction_info': None,
            'seasonal_segmented': True,
            'segment_summaries': pd.DataFrame(segment_summaries),
        },
    }


def predict_temperature_grid(
    model,
    df,
    max_depth=25.0,
    n_depth_points=150,
    device='cpu',
    apply_post_physics=False,
    use_shallow_optimized=False,
    shallow_focus_depth=5.0,
    shallow_fraction=0.55,
    rolling_prediction_mode=False,
    rolling_memory_blend=0.85,
    rolling_surface_relaxation=0.35,
    rolling_surface_decay_depth=4.0,
    rolling_deep_inertia=0.65,
    rolling_deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    ppo_controller=None,
    ppo_policy_bundle=None,
    online_ppo_update=False,
    online_ppo_control_interval=7,
    online_ppo_rollout_steps=4,
    online_ppo_max_updates_run=None,
    validation_profile_obs=None,
    validation_metadata=None,
    validation_max_depth=None,
):
    model.eval()
    depth_grid = build_depth_grid(
        max_depth=max_depth,
        n_depth_points=n_depth_points,
        use_shallow_optimized=use_shallow_optimized,
        shallow_focus_depth=shallow_focus_depth,
        shallow_fraction=shallow_fraction,
    )
    depths = torch.tensor(depth_grid, dtype=torch.float32, device=device).reshape(-1, 1)
    z_norm = depths / max_depth
    profiles = []
    times = torch.tensor(df['time_norm'].to_numpy(dtype=np.float32).reshape(-1, 1), device=device)

    with torch.no_grad():
        for time_point in times:
            t_day = time_point.expand_as(depths)
            pred = model(torch.cat([t_day, z_norm], dim=1)).cpu().numpy().flatten()
            profiles.append(pred)

    temp_grid = np.array(profiles).T
    depths_np = depths.cpu().numpy().flatten()

    online_ppo_runtime = {
        'diagnostics': pd.DataFrame(),
        'history': pd.DataFrame(),
        'kalman_scales': {'process': 1.0, 'obs': 1.0},
    }
    if ppo_controller is not None and ppo_policy_bundle is not None:
        initial_weights = dict(ppo_policy_bundle.get('final_weights', {'pde': 1.0e5, 'bc': 10.0, 'ic': 5.0, 'obs': 1.0}))
        initial_kalman_scales = dict(ppo_policy_bundle.get('final_kalman_scales', {'process': 1.0, 'obs': 1.0}))
        temp_grid, ppo_diagnostics, ppo_history = build_online_ppo_rolling_grid(
            raw_temp_grid=temp_grid,
            df=df,
            depths=depths_np,
            ppo_controller=ppo_controller,
            initial_weights=initial_weights,
            initial_kalman_scales=initial_kalman_scales,
            control_interval=online_ppo_control_interval,
            rollout_steps=online_ppo_rollout_steps,
            update_policy=online_ppo_update,
            max_policy_updates_run=online_ppo_max_updates_run,
            memory_blend=rolling_memory_blend,
            surface_relaxation=rolling_surface_relaxation,
            surface_decay_depth=rolling_surface_decay_depth,
            deep_inertia=rolling_deep_inertia,
            deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
            validation_profile_obs=validation_profile_obs,
            validation_metadata=validation_metadata,
            validation_max_depth=validation_max_depth,
        )
        if not ppo_diagnostics.empty:
            online_ppo_runtime['diagnostics'] = ppo_diagnostics
            last_row = ppo_diagnostics.iloc[-1]
            online_ppo_runtime['kalman_scales'] = {
                'process': float(last_row['kalman_process_scale']),
                'obs': float(last_row['kalman_obs_scale']),
            }
        if not ppo_history.empty:
            online_ppo_runtime['history'] = ppo_history
    elif rolling_prediction_mode:
        temp_grid = build_rolling_prediction_grid(
            raw_temp_grid=temp_grid,
            df=df,
            depths=depths_np,
            memory_blend=rolling_memory_blend,
            surface_relaxation=rolling_surface_relaxation,
            surface_decay_depth=rolling_surface_decay_depth,
            deep_inertia=rolling_deep_inertia,
            deep_anchor=rolling_deep_anchor,
            surface_skin_cooling_coef=surface_skin_cooling_coef,
        )

    if apply_post_physics:
        temp_grid = np.clip(temp_grid, 0.0, 30.0)
        air_temp = df['T_air_C'].values
        for day_idx, t_air in enumerate(air_temp):
            if t_air < 0.0:
                temp_grid[0, day_idx] = 0.0

    return temp_grid, depths_np, online_ppo_runtime


def build_rolling_prediction_grid(
    raw_temp_grid,
    df,
    depths,
    memory_blend=0.85,
    surface_relaxation=0.35,
    surface_decay_depth=4.0,
    deep_inertia=0.65,
    deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
):
    raw_temp_grid = np.asarray(raw_temp_grid, dtype=np.float64)
    rolled_grid = np.zeros_like(raw_temp_grid)
    depths = np.asarray(depths, dtype=np.float64)
    n_depth, n_days = raw_temp_grid.shape

    memory_blend = float(np.clip(memory_blend, 0.0, 1.0))
    surface_relaxation = float(np.clip(surface_relaxation, 0.0, 1.0))
    surface_decay_depth = float(max(surface_decay_depth, 1e-6))
    deep_inertia = float(np.clip(deep_inertia, 0.0, 0.95))
    deep_anchor = float(np.clip(deep_anchor, 0.0, 0.5))
    surface_weights = np.exp(-depths / surface_decay_depth)
    surface_weights[0] = 1.0

    lst_surface = None
    if 'SurfaceBulkTarget_C' in df.columns:
        lst_surface = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    elif 'LST_surface_C' in df.columns:
        lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64)

    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else None
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(n_days, 2.0, dtype=np.float64)

    rolled_grid[:, 0] = raw_temp_grid[:, 0]
    if lst_surface is not None:
        initial_surface_target = compute_runtime_surface_target(
            df=df,
            day_idx=0,
            runtime_skin_cooling_coef=surface_skin_cooling_coef,
            base_surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        initial_surface_error = float(np.clip(initial_surface_target - rolled_grid[0, 0], -6.0, 6.0))
        initial_mld = float(np.clip(mixed_layer_depth[0], 0.5, max(depths[-1] * 0.9, 0.5)))
        mixed_transition0 = 1.0 / (1.0 + np.exp((depths - (initial_mld + 0.75)) / 0.9))
        nudge_weights0 = surface_weights * (mixed_transition0 + 0.08 * (1.0 - mixed_transition0))
        rolled_grid[:, 0] += surface_relaxation * initial_surface_error * nudge_weights0
    if air_temp is not None and air_temp[0] < 0.0:
        rolled_grid[0, 0] = 0.0
    rolled_grid[:, 0] = np.clip(rolled_grid[:, 0], -1.0, 35.0)

    for day_idx in range(1, n_days):
        raw_today = raw_temp_grid[:, day_idx]
        raw_prev = raw_temp_grid[:, day_idx - 1]
        model_increment = raw_today - raw_prev
        day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
        mixed_transition = 1.0 / (1.0 + np.exp((depths - (day_mld + 0.75)) / 0.9))
        increment_scale = 1.0 - deep_inertia * (1.0 - mixed_transition)
        persisted_state = rolled_grid[:, day_idx - 1] + model_increment * increment_scale
        rolled_today = memory_blend * persisted_state + (1.0 - memory_blend) * raw_today

        if lst_surface is not None:
            runtime_surface_target = compute_runtime_surface_target(
                df=df,
                day_idx=day_idx,
                runtime_skin_cooling_coef=surface_skin_cooling_coef,
                base_surface_skin_cooling_coef=surface_skin_cooling_coef,
            )
            surface_error = float(np.clip(runtime_surface_target - rolled_today[0], -6.0, 6.0))
            nudge_weights = surface_weights * (mixed_transition + 0.08 * (1.0 - mixed_transition))
            rolled_today = rolled_today + surface_relaxation * surface_error * nudge_weights

        warm_driver = 0.0
        if air_temp is not None:
            warm_driver = 1.0 / (1.0 + np.exp(-(air_temp[day_idx] - 9.0) / 2.0))
        solar_driver = 1.0
        if 'Solar_W_m2' in df.columns:
            solar_driver = 1.0 / (1.0 + np.exp(-(float(df['Solar_W_m2'].iloc[day_idx]) - 140.0) / 45.0))
        deep_floor = max(day_mld + 2.5, depths[-1] * 0.45)
        anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
        deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
        deep_excess = np.maximum(rolled_today - deep_anchor_profile, 0.0)
        rolled_today = rolled_today - deep_anchor * warm_driver * solar_driver * anchor_mask * deep_excess

        if air_temp is not None and air_temp[day_idx] < 0.0:
            rolled_today[0] = 0.0

        rolled_grid[:, day_idx] = np.clip(rolled_today, -1.0, 35.0)

    return rolled_grid.astype(np.float32)


def build_online_ppo_rolling_grid(
    raw_temp_grid,
    df,
    depths,
    ppo_controller,
    initial_weights,
    initial_kalman_scales,
    control_interval=7,
    rollout_steps=4,
    update_policy=False,
    max_policy_updates_run=None,
    memory_blend=0.85,
    surface_relaxation=0.35,
    surface_decay_depth=4.0,
    deep_inertia=0.65,
    deep_anchor=0.18,
    surface_skin_cooling_coef=SURFACE_SKIN_COOLING_COEF,
    validation_profile_obs=None,
    validation_metadata=None,
    validation_max_depth=None,
):
    raw_temp_grid = np.asarray(raw_temp_grid, dtype=np.float64)
    rolled_grid = np.zeros_like(raw_temp_grid)
    depths = np.asarray(depths, dtype=np.float64)
    n_depth, n_days = raw_temp_grid.shape

    control_params = derive_online_control_params_from_weights(
        initial_weights=initial_weights,
        memory_blend=memory_blend,
        surface_relaxation=surface_relaxation,
        deep_inertia=deep_inertia,
        deep_anchor=deep_anchor,
        surface_skin_cooling_coef=surface_skin_cooling_coef,
    )
    kalman_scales = dict(initial_kalman_scales)
    policy_weights = dict(initial_weights)
    surface_decay_depth = float(max(surface_decay_depth, 1e-6))
    surface_weights = np.exp(-depths / surface_decay_depth)
    surface_weights[0] = 1.0

    lst_surface = None
    if 'SurfaceBulkTarget_C' in df.columns:
        lst_surface = df['SurfaceBulkTarget_C'].to_numpy(dtype=np.float64)
    elif 'LST_surface_C' in df.columns:
        lst_surface = df['LST_surface_C'].to_numpy(dtype=np.float64)

    air_temp = df['T_air_C'].to_numpy(dtype=np.float64) if 'T_air_C' in df.columns else None
    mixed_layer_depth = df['MixedLayerDepth_m'].to_numpy(dtype=np.float64) if 'MixedLayerDepth_m' in df.columns else np.full(n_days, 2.0, dtype=np.float64)

    diagnostics = []
    ppo_context = None
    ppo_history = []
    ppo_update_count = 0

    rolled_grid[:, 0] = raw_temp_grid[:, 0]

    for day_idx in range(n_days):
        if day_idx > 0:
            raw_today = raw_temp_grid[:, day_idx]
            raw_prev = raw_temp_grid[:, day_idx - 1]
            model_increment = raw_today - raw_prev
            day_mld = float(np.clip(mixed_layer_depth[day_idx], 0.5, max(depths[-1] * 0.9, 0.5)))
            mixed_transition = 1.0 / (1.0 + np.exp((depths - (day_mld + 0.75)) / 0.9))
            increment_scale = 1.0 - control_params['deep_inertia'] * (1.0 - mixed_transition)
            persisted_state = rolled_grid[:, day_idx - 1] + model_increment * increment_scale
            rolled_today = control_params['memory_blend'] * persisted_state + (1.0 - control_params['memory_blend']) * raw_today

            if lst_surface is not None:
                runtime_surface_target = compute_runtime_surface_target(
                    df=df,
                    day_idx=day_idx,
                    runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
                    base_surface_skin_cooling_coef=surface_skin_cooling_coef,
                )
                surface_error = float(np.clip(runtime_surface_target - rolled_today[0], -6.0, 6.0))
                nudge_weights = surface_weights * (mixed_transition + 0.08 * (1.0 - mixed_transition))
                rolled_today = rolled_today + control_params['surface_relaxation'] * surface_error * nudge_weights

            warm_driver = 0.0
            if air_temp is not None:
                warm_driver = 1.0 / (1.0 + np.exp(-(air_temp[day_idx] - 9.0) / 2.0))
            solar_driver = 1.0
            if 'Solar_W_m2' in df.columns:
                solar_driver = 1.0 / (1.0 + np.exp(-(float(df['Solar_W_m2'].iloc[day_idx]) - 140.0) / 45.0))
            deep_floor = max(day_mld + 2.5, depths[-1] * 0.45)
            anchor_mask = 1.0 / (1.0 + np.exp(-(depths - deep_floor) / 1.2))
            deep_anchor_profile = DEFAULT_INITIAL_WATER_TEMPERATURE_C + 2.5 * np.exp(-np.clip(depths - deep_floor, 0.0, None) / 4.0)
            deep_excess = np.maximum(rolled_today - deep_anchor_profile, 0.0)
            rolled_today = rolled_today - control_params['deep_anchor'] * warm_driver * solar_driver * anchor_mask * deep_excess

            if air_temp is not None and air_temp[day_idx] < 0.0:
                rolled_today[0] = 0.0

            rolled_today = np.clip(rolled_today, -1.0, 35.0)
            rolled_today, projection_adjustments = project_temperature_profile_to_stable_density(rolled_today)
            rolled_grid[:, day_idx] = rolled_today
        else:
            projection_adjustments = 0
            if lst_surface is not None:
                initial_surface_target = compute_runtime_surface_target(
                    df=df,
                    day_idx=0,
                    runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
                    base_surface_skin_cooling_coef=surface_skin_cooling_coef,
                )
                initial_surface_error = float(np.clip(initial_surface_target - rolled_grid[0, 0], -6.0, 6.0))
                initial_mld = float(np.clip(mixed_layer_depth[0], 0.5, max(depths[-1] * 0.9, 0.5)))
                mixed_transition0 = 1.0 / (1.0 + np.exp((depths - (initial_mld + 0.75)) / 0.9))
                nudge_weights0 = surface_weights * (mixed_transition0 + 0.08 * (1.0 - mixed_transition0))
                rolled_grid[:, 0] += control_params['surface_relaxation'] * initial_surface_error * nudge_weights0
            if air_temp is not None and air_temp[0] < 0.0:
                rolled_grid[0, 0] = 0.0
            rolled_grid[:, 0] = np.clip(rolled_grid[:, 0], -1.0, 35.0)
            rolled_grid[:, 0], projection_adjustments = project_temperature_profile_to_stable_density(rolled_grid[:, 0])

        current_profile = rolled_grid[:, day_idx]
        previous_profile = None if day_idx == 0 else rolled_grid[:, day_idx - 1]
        summary = compute_online_proxy_summary(
            current_profile=current_profile,
            previous_profile=previous_profile,
            day_idx=day_idx,
            df=df,
            depths=depths,
            control_params=control_params,
            kalman_scales=kalman_scales,
        )
        validation_metrics = compute_online_proxy_validation(
            current_profile,
            day_idx,
            df,
            runtime_skin_cooling_coef=control_params['surface_skin_cooling_coef'],
            base_surface_skin_cooling_coef=surface_skin_cooling_coef,
        )
        if has_profile_observations(validation_profile_obs):
            date_validation = evaluate_profile_at_date(
                current_date=df['Date'].iloc[day_idx],
                current_profile=current_profile,
                depths=depths,
                profile_obs_data=validation_profile_obs,
            )
            if date_validation is not None:
                validation_metrics.update(date_validation)
            shallow_surface_validation = evaluate_surface_band_validation_at_date(
                current_date=df['Date'].iloc[day_idx],
                current_profile=current_profile,
                depths=depths,
                profile_obs_data=validation_profile_obs,
                previous_profile_3d=None if day_idx < 3 else rolled_grid[:, day_idx - 3],
                shallow_max_depth=3.0,
            )
            if shallow_surface_validation is not None:
                validation_metrics.update(shallow_surface_validation)

        if ((day_idx + 1) % max(control_interval, 1) == 0) or (day_idx == n_days - 1):
            current_state = build_ppo_state(
                summary=summary,
                weights=policy_weights,
                kalman_scales=kalman_scales,
                learning_rate=1.0e-3,
                validation_metrics=validation_metrics,
            )

            if ppo_context is not None:
                reward = compute_ppo_reward(
                    prev_summary=ppo_context['summary'],
                    current_summary=summary,
                    prev_validation_metrics=ppo_context['validation'],
                    current_validation_metrics=validation_metrics,
                )
                done = bool(day_idx == n_days - 1)
                ppo_controller.store_transition(
                    state=ppo_context['state'],
                    action=ppo_context['action'],
                    log_prob=ppo_context['log_prob'],
                    reward=reward,
                    done=done,
                    value=ppo_context['value'],
                )
                if update_policy and (done or len(ppo_controller.buffer['states']) >= max(rollout_steps, 1)):
                    reached_update_cap = (
                        max_policy_updates_run is not None
                        and ppo_update_count >= int(max(max_policy_updates_run, 0))
                    )
                    if reached_update_cap:
                        ppo_controller.reset_buffer()
                    else:
                        update_stats = ppo_controller.update(last_state=current_state, last_done=done)
                        if update_stats is not None:
                            ppo_update_count += 1

            if day_idx != n_days - 1:
                action, log_prob, value = ppo_controller.select_action(current_state)
                control_params, kalman_scales = apply_online_ppo_action(control_params, kalman_scales, action)
                ppo_context = {
                    'state': current_state,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'summary': summary,
                    'validation': validation_metrics,
                }
                ppo_history.append(
                    {
                        'day_idx': day_idx,
                        'date': pd.Timestamp(df['Date'].iloc[day_idx]),
                        'memory_blend': control_params['memory_blend'],
                        'surface_relaxation': control_params['surface_relaxation'],
                        'deep_inertia': control_params['deep_inertia'],
                        'deep_anchor': control_params['deep_anchor'],
                        'surface_skin_cooling_coef': control_params['surface_skin_cooling_coef'],
                        'kalman_process_scale': kalman_scales['process'],
                        'kalman_obs_scale': kalman_scales['obs'],
                        'surface_rmse_proxy': validation_metrics['rmse'],
                    }
                )

        diagnostics.append(
            {
                'Date': pd.Timestamp(df['Date'].iloc[day_idx]),
                'memory_blend': control_params['memory_blend'],
                'surface_relaxation': control_params['surface_relaxation'],
                'deep_inertia': control_params['deep_inertia'],
                'deep_anchor': control_params['deep_anchor'],
                'surface_skin_cooling_coef': control_params['surface_skin_cooling_coef'],
                'kalman_process_scale': kalman_scales['process'],
                'kalman_obs_scale': kalman_scales['obs'],
                'projection_adjustments': int(projection_adjustments),
                'surface_rmse_proxy': validation_metrics['rmse'],
                'ppo_update_count': int(ppo_update_count),
            }
        )

    return rolled_grid.astype(np.float32), pd.DataFrame(diagnostics), pd.DataFrame(ppo_history)


def build_depth_covariance(depths, variance, correlation_length):
    """Build a depth-correlated covariance matrix for the Kalman filter."""
    depths = np.asarray(depths, dtype=np.float64)
    variance = float(max(variance, 0.0))
    if variance == 0.0:
        return np.zeros((len(depths), len(depths)), dtype=np.float64)

    if correlation_length <= 0.0:
        return np.eye(len(depths), dtype=np.float64) * variance

    delta = depths[:, None] - depths[None, :]
    kernel = np.exp(-0.5 * (delta / correlation_length) ** 2)
    return variance * kernel


def effective_forecast_blend(
    base_blend,
    days_since_last_obs,
    spinup_days=0,
    spinup_max_blend=0.9,
):
    base_blend = float(np.clip(base_blend, 0.0, 1.0))
    spinup_max_blend = float(np.clip(spinup_max_blend, base_blend, 1.0))
    spinup_days = int(max(spinup_days, 0))

    if spinup_days <= 0 or days_since_last_obs is None:
        return base_blend

    if days_since_last_obs <= 0:
        return base_blend

    if spinup_days == 1:
        return spinup_max_blend

    progress = min(max((days_since_last_obs - 1) / max(spinup_days - 1, 1), 0.0), 1.0)
    return float(spinup_max_blend - progress * (spinup_max_blend - base_blend))


def build_kalman_observation_frame(
    df,
    metadata,
    max_depth,
    profile_obs_data,
    depths,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
):
    """Map observations to the nearest state-grid depth index for assimilation."""
    observations, _ = build_observation_dataframe(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_obs_data,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    observations = observations.copy()
    observations['Date'] = pd.to_datetime(observations['Date']).dt.normalize()
    depth_grid = np.asarray(depths, dtype=np.float64)
    observations['grid_idx'] = observations['Depth_m'].apply(
        lambda depth_value: int(np.argmin(np.abs(depth_grid - float(depth_value))))
    )
    observations['grid_depth_m'] = observations['grid_idx'].apply(lambda idx: float(depth_grid[idx]))
    observations = (
        observations
        .groupby(['Date', 'source', 'grid_idx', 'grid_depth_m'], as_index=False)
        .agg({'Temperature_C': 'mean'})
        .sort_values(['Date', 'grid_idx'])
        .reset_index(drop=True)
    )
    return observations


def run_profile_kalman_filter(
    df,
    temp_grid,
    depths,
    metadata,
    max_depth,
    profile_obs_data=None,
    prior_std=2.0,
    process_std=0.3,
    obs_std_surface=0.5,
    obs_std_bottom=0.5,
    obs_std_profile=0.75,
    correlation_length=2.0,
    forecast_blend=0.2,
    forecast_spinup_days=0,
    forecast_spinup_max_blend=0.9,
    use_surface_bulk_correction=False,
    use_bottom_observation=False,
    surface_obs_depth_m=0.35,
    daily_process_scale=None,
    daily_obs_scale=None,
    autumn_asymmetric_cooling=False,
    autumn_doy_threshold=270.0,
    autumn_surface_cooling_threshold=1.0,
    autumn_air_temp_threshold=12.0,
    autumn_cooling_strength=0.35,
    autumn_cooling_penetration_scale=5.0,
):
    """
    Assimilate profile observations on top of PINN temperature profiles.

    State vector:
        x_t = [T(z_1, t), ..., T(z_n, t)]^T

    Forecast:
        the PINN profile at day t acts as the model forecast, optionally blended
        with the previous filtered state to preserve temporal continuity.
    """
    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    n_depth, n_days = temp_grid.shape
    filtered_grid = np.zeros_like(temp_grid)
    identity = np.eye(n_depth, dtype=np.float64)
    prior_cov = build_depth_covariance(depths, prior_std ** 2, correlation_length)
    covariance = prior_cov.copy()

    observations = build_kalman_observation_frame(
        df=df,
        metadata=metadata,
        max_depth=max_depth,
        profile_obs_data=profile_obs_data,
        depths=depths,
        use_surface_bulk_correction=use_surface_bulk_correction,
        use_bottom_observation=use_bottom_observation,
        surface_obs_depth_m=surface_obs_depth_m,
    )
    obs_by_date = {
        date_value: group.reset_index(drop=True)
        for date_value, group in observations.groupby('Date')
    }

    diagnostics = []
    normalized_dates = pd.to_datetime(df['Date']).dt.normalize()
    last_observation_day_idx = None

    for day_idx, date_value in enumerate(normalized_dates):
        process_scale_today = 1.0 if daily_process_scale is None else float(daily_process_scale[min(day_idx, len(daily_process_scale) - 1)])
        obs_scale_today = 1.0 if daily_obs_scale is None else float(daily_obs_scale[min(day_idx, len(daily_obs_scale) - 1)])
        process_cov = build_depth_covariance(depths, (process_std * process_scale_today) ** 2, correlation_length)
        pinn_forecast = temp_grid[:, day_idx].copy()
        if day_idx == 0:
            state_pred = pinn_forecast
        else:
            if last_observation_day_idx is None:
                days_since_last_obs = None
            else:
                days_since_last_obs = day_idx - last_observation_day_idx
            blend_today = effective_forecast_blend(
                base_blend=forecast_blend,
                days_since_last_obs=days_since_last_obs,
                spinup_days=forecast_spinup_days,
                spinup_max_blend=forecast_spinup_max_blend,
            )
            state_pred = (
                blend_today * filtered_grid[:, day_idx - 1] +
                (1.0 - blend_today) * pinn_forecast
            )

        cov_pred = covariance + process_cov
        day_obs = obs_by_date.get(date_value)

        if day_obs is None or day_obs.empty:
            state_upd = state_pred
            covariance = cov_pred
            obs_count = 0
            innovation_rms = np.nan
            autumn_cooling_applied = 0.0
        else:
            y = day_obs['Temperature_C'].to_numpy(dtype=np.float64)
            grid_idx = day_obs['grid_idx'].to_numpy(dtype=np.int64)
            H = np.zeros((len(day_obs), n_depth), dtype=np.float64)
            H[np.arange(len(day_obs)), grid_idx] = 1.0

            obs_std = []
            for source, obs_depth in zip(day_obs['source'], day_obs['grid_depth_m']):
                obs_std.append(
                    depth_dependent_obs_std(
                        source=source,
                        depth_m=obs_depth,
                        max_depth=max_depth,
                        base_surface=obs_std_surface * obs_scale_today,
                        base_bottom=obs_std_bottom * obs_scale_today,
                        base_profile=obs_std_profile * obs_scale_today,
                    )
                )
            R = np.diag(np.square(np.asarray(obs_std, dtype=np.float64)))

            innovation = y - H @ state_pred
            innovation_rms = float(np.sqrt(np.mean(innovation ** 2)))
            S = H @ cov_pred @ H.T + R
            kalman_gain = cov_pred @ H.T @ np.linalg.pinv(S)
            state_upd = state_pred + kalman_gain @ innovation

            # Joseph form keeps covariance positive semidefinite more reliably.
            kh = kalman_gain @ H
            covariance = (identity - kh) @ cov_pred @ (identity - kh).T + kalman_gain @ R @ kalman_gain.T
            obs_count = int(len(day_obs))
            last_observation_day_idx = day_idx

            surface_obs_temp = None
            surface_rows = day_obs[day_obs['source'] == 'surface']
            if not surface_rows.empty:
                surface_obs_temp = float(surface_rows['Temperature_C'].iloc[0])
            state_upd, autumn_cooling_applied = apply_autumn_cooling_adjustment(
                state_pred=state_pred,
                state_upd=state_upd,
                depths=depths,
                day_doy=float(df['full_doy'].iloc[day_idx]),
                mixed_layer_depth=float(df['MixedLayerDepth_m'].iloc[day_idx]) if 'MixedLayerDepth_m' in df.columns else 2.0,
                surface_obs_temp=surface_obs_temp,
                air_temp=float(df['T_air_C'].iloc[day_idx]) if 'T_air_C' in df.columns else None,
                enabled=autumn_asymmetric_cooling,
                doy_threshold=autumn_doy_threshold,
                cooling_threshold=autumn_surface_cooling_threshold,
                air_temp_threshold=autumn_air_temp_threshold,
                propagation_strength=autumn_cooling_strength,
                penetration_scale=autumn_cooling_penetration_scale,
            )

        if df['T_air_C'].iloc[day_idx] < 0.0:
            state_upd[0] = 0.0

        state_upd = np.clip(state_upd, -1.0, 35.0)
        state_upd, projection_adjustments = project_temperature_profile_to_stable_density(state_upd)
        filtered_grid[:, day_idx] = state_upd
        covariance = 0.5 * (covariance + covariance.T)

        diagnostics.append(
            {
                'Date': pd.Timestamp(date_value),
                'obs_count': obs_count,
                'innovation_rms': innovation_rms,
                'surface_temperature_C': float(state_upd[0]),
                'bottom_temperature_C': float(state_upd[-1]),
                'projection_adjustments': int(projection_adjustments),
                'autumn_cooling_applied': float(autumn_cooling_applied),
                'days_since_last_obs': np.nan if last_observation_day_idx is None else float(day_idx - last_observation_day_idx),
            }
        )

    diagnostics_df = pd.DataFrame(diagnostics)
    return filtered_grid.astype(np.float32), diagnostics_df


def export_temperature_tables(df, temp_grid, depths, output_dir, metadata, suffix=''):
    records = []
    day_axis = df['full_doy'].to_numpy()
    dates = pd.to_datetime(df['Date']).to_numpy()

    for day_idx, (date_value, doy_value) in enumerate(zip(dates, day_axis)):
        month_value = pd.Timestamp(date_value).month
        for depth_idx, depth_value in enumerate(depths):
            records.append({
                'Date': pd.Timestamp(date_value).date().isoformat(),
                'Month': month_value,
                'DOY': int(doy_value),
                'Depth_m': float(depth_value),
                'Temperature_C': float(temp_grid[depth_idx, day_idx]),
            })

    temp_df = pd.DataFrame.from_records(records)
    suffix = f"_{suffix}" if suffix else ''
    full_path = output_dir / f"{metadata['file_tag']}{suffix}_temperature_depth_predictions.csv"
    temp_df.to_csv(full_path, index=False)
    return full_path


def evaluate_profile_predictions(prediction_csv_path, profile_obs_data):
    """Evaluate predictions against profile observations using depth interpolation."""
    if not has_profile_observations(profile_obs_data):
        return None

    pred = pd.read_csv(prediction_csv_path)
    pred['Date'] = pd.to_datetime(pred['Date'])
    obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=pred['Date'].min(),
        time_scale_seconds=max((pd.to_datetime(pred['Date']).max() - pd.to_datetime(pred['Date']).min()).total_seconds(), SECONDS_PER_DAY),
        max_depth=float(pred['Depth_m'].max()),
    )

    errors = []
    matched_rows = 0
    for date_value, obs_day in obs.groupby('Date'):
        pred_day = pred[pred['Date'] == date_value].sort_values('Depth_m')
        if pred_day.empty:
            continue
        pred_interp = np.interp(
            obs_day['Depth_m'].to_numpy(),
            pred_day['Depth_m'].to_numpy(),
            pred_day['Temperature_C'].to_numpy(),
        )
        errors.extend((pred_interp - obs_day['Temperature_C'].to_numpy()).tolist())
        matched_rows += len(obs_day)

    if not errors:
        return None

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'matched_rows': int(matched_rows),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
    }


def plot_year_heatmap(df, temp_grid, depths, output_path, metadata):
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    day_axis = df['full_doy'].to_numpy()
    vmin = float(np.nanmin(temp_grid))
    vmax = float(np.nanmax(temp_grid))
    filled_levels = np.linspace(vmin, vmax, 28)
    line_levels = np.arange(np.floor(vmin / 4.0) * 4.0, np.ceil(vmax / 4.0) * 4.0 + 0.1, 4.0)
    if line_levels.size < 2:
        line_levels = np.linspace(vmin, vmax, 6)

    image = ax.contourf(day_axis, depths, temp_grid, levels=filled_levels, cmap='RdYlBu_r', extend='both')
    contour_lines = ax.contour(day_axis, depths, temp_grid, levels=line_levels, colors='black', linewidths=1.1, alpha=0.45)
    ax.clabel(contour_lines, fmt='%d', fontsize=10, inline=True)

    month_midpoints = df.groupby(df['Date'].dt.month)['full_doy'].mean()
    ax.set_xticks(month_midpoints.values)
    ax.set_xticklabels([calendar.month_abbr[month] for month in month_midpoints.index], fontsize=17)
    ax.set_xlabel('Month', fontsize=20, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    ax.set_title(f"Annual Water Temperature Profile of {metadata['lake_name']}", fontsize=26)
    ax.set_ylim(depths[-1], depths[0])
    ax.tick_params(axis='y', labelsize=15)

    max_depth = float(depths[-1])
    ax.text(25, max_depth * 0.10, 'Winter\nInverse\nStratification', color='blue', fontsize=22, fontweight='bold', ha='center')
    ax.text(120, max_depth * 0.88, 'Spring\nWarming', color='green', fontsize=22, fontweight='bold', ha='center')
    ax.text(210, max_depth * 0.78, 'Summer\nStratification\n(Thermocline)', color='red', fontsize=24, fontweight='bold', ha='center')
    ax.text(305, max_depth * 0.52, 'Autumn\nOverturn\n(Homothermal)', color='black', fontsize=22, fontweight='bold', ha='center')

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label('Temperature (掳C)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_depth_curves(df, temp_grid, depths, output_path, metadata):
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    dates = pd.to_datetime(df['Date'])
    day_numbers = (dates - dates.min()).dt.days.values
    temp_min = np.nanmin(temp_grid)
    temp_max = np.nanmax(temp_grid)
    norm = plt.Normalize(vmin=temp_min, vmax=temp_max)

    step = 5
    for i in range(0, len(day_numbers), step):
        day_num = day_numbers[i]
        temp_profile = temp_grid[:, i]
        depth_axis = depths[::-1]
        color = plt.cm.RdYlBu_r(norm(np.mean(temp_profile)))
        linewidth = 1.5 if (i % 30 == 0) else 0.8
        ax.plot(temp_profile, depth_axis, color=color, linewidth=linewidth, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Temperature (掳C)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    month_starts = dates[dates.dt.day == 1].dt.dayofyear.values
    for doy in month_starts:
        if doy < len(day_numbers):
            day_idx = np.abs(day_numbers - doy).argmin()
            ax.axvline(day_numbers[day_idx], color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(
                day_numbers[day_idx],
                depths[-1] * 0.95,
                calendar.month_abbr[dates[day_idx].month],
                ha='center',
                va='top',
                fontsize=14,
                fontweight='bold',
                color='gray',
            )

    max_depth = float(depths[-1])
    ax.text(day_numbers[10], max_depth * 0.10, 'Winter', color='blue', fontsize=16, fontweight='bold', ha='center')
    ax.text(day_numbers[80], max_depth * 0.88, 'Spring', color='green', fontsize=16, fontweight='bold', ha='center')
    ax.text(day_numbers[170], max_depth * 0.78, 'Summer', color='red', fontsize=16, fontweight='bold', ha='center')
    ax.text(day_numbers[260], max_depth * 0.52, 'Autumn', color='black', fontsize=16, fontweight='bold', ha='center')

    ax.set_xlabel('Time (days)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    ax.set_title(f"Time-Depth Temperature Profiles: {metadata['lake_name']} {metadata['year']}", fontsize=22, fontweight='bold')
    ax.set_xlim(0, len(day_numbers))
    ax.set_ylim(depths[-1], depths[0])
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_monthly_heatmaps(df, temp_grid, depths, output_path, metadata):
    dates = pd.to_datetime(df['Date'])
    temp_df = pd.DataFrame(
        {
            'Date': dates,
            'Month': dates.dt.month,
            'Day': dates.dt.day,
        }
    )

    monthly_day_max = temp_df.groupby('Month')['Day'].max().reindex(range(1, 13), fill_value=31)
    vmin = float(np.nanmin(temp_grid))
    vmax = float(np.nanmax(temp_grid))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    pcm = None

    for month_idx, ax in enumerate(axes.flat, start=1):
        month_mask = temp_df['Month'] == month_idx
        month_rows = temp_df.loc[month_mask]
        if month_rows.empty:
            ax.set_title(calendar.month_abbr[month_idx], fontsize=14, fontweight='bold')
            ax.set_xlim(1, monthly_day_max.loc[month_idx])
            ax.set_ylim(depths[-1], depths[0])
            ax.grid(True, alpha=0.35)
            continue

        month_grid = temp_grid[:, month_mask.to_numpy()]
        day_values = month_rows['Day'].to_numpy(dtype=np.float32)
        x_edges = np.arange(0.5, monthly_day_max.loc[month_idx] + 1.5, 1.0, dtype=np.float32)
        if len(depths) > 1:
            depth_step = np.diff(depths)
            top_edge = max(0.0, depths[0] - depth_step[0] / 2.0)
            bottom_edge = depths[-1] + depth_step[-1] / 2.0
            inner_edges = 0.5 * (depths[:-1] + depths[1:])
            y_edges = np.concatenate([[top_edge], inner_edges, [bottom_edge]])
        else:
            y_edges = np.array([0.0, max(float(depths[0]), 1.0)], dtype=np.float32)

        padded_grid = np.full((len(depths), len(x_edges) - 1), np.nan, dtype=np.float32)
        for col_idx, day_value in enumerate(day_values.astype(int)):
            padded_grid[:, day_value - 1] = month_grid[:, col_idx]

        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            padded_grid,
            cmap='turbo',
            vmin=vmin,
            vmax=vmax,
            shading='flat',
        )
        ax.set_title(calendar.month_abbr[month_idx], fontsize=14, fontweight='bold')
        ax.set_xlim(1, monthly_day_max.loc[month_idx])
        ax.set_ylim(depths[-1], depths[0])
        ax.set_xticks(np.arange(5, monthly_day_max.loc[month_idx] + 1, 5))
        ax.grid(True, alpha=0.35, color='white', linewidth=0.8)
        if month_idx in (1, 5, 9):
            ax.set_ylabel('Depth (m)')
        else:
            ax.set_yticklabels([])
        if month_idx >= 9:
            ax.set_xlabel('Day of month')

    fig.suptitle(
        f"{metadata['lake_name']} {metadata['year']} Monthly Temperature-Depth Heatmaps",
        fontsize=18,
    )
    cbar = fig.colorbar(pcm, ax=axes, shrink=0.94, pad=0.02)
    cbar.set_label('Temperature (掳C)')
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='PDF-aligned PINN baseline for lake temperature reconstruction')
    parser.add_argument('--era5', default=None, help='Path to ERA5 forcing data CSV')
    parser.add_argument('--lst', default=None, help='Path to surface LST observation CSV')
    parser.add_argument('--profile-obs', default=None, help='Optional profile observation CSV with Date/Depth_m/Temperature_C')
    parser.add_argument('--mode', choices=['train', 'predict'], default=None, help='train: use profile observations for training/evaluation when available; predict: ignore profile observations and use only LST + ERA5 for inference')
    parser.add_argument('--practical-prediction-mode', action='store_true', help='Use a continuous practical prediction setup: no seasonal segmentation, no bottom observation, Kalman on, PPO off')
    parser.add_argument('--profile-split-mode', choices=['none', 'depth_interleaved', 'time_blocked'], default='depth_interleaved', help='How to split profile observations into train/val/assim/test roles to avoid leakage')
    parser.add_argument('--seasonal-segmented', action='store_true', help='Train separate models for contiguous seasonal blocks and stitch them into a full-year prediction')
    parser.add_argument('--use-bottom-observation', action='store_true', help='Use BottomTemp_C as an observation boundary; disable for strict ERA5+LST prediction')
    parser.add_argument('--initial-condition-mode', choices=['uniform_4c', 'surface_to_uniform_4c', 'linear_to_bottom_obs'], default='uniform_4c', help='How to construct the initial temperature profile')
    parser.add_argument('--max-depth', type=float, default=20.0, help='Maximum lake depth in meters')
    parser.add_argument('--depth-points', type=int, default=150, help='Number of depth samples for exported profiles')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--collocation-points', type=int, default=512, help='Number of PDE collocation points per epoch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--output-dir', default=str(PROJECT_DIR), help='Output directory')
    parser.add_argument('--apply-post-physics', action='store_true', help='Apply light post-processing to the predicted grid')
    parser.add_argument('--rolling-prediction-mode', action='store_true', help='Generate profiles sequentially in time using the previous day state instead of treating each day independently')
    parser.add_argument('--rolling-memory-blend', type=float, default=0.85, help='How strongly rolling prediction keeps the previous day state relative to the raw PINN daily profile')
    parser.add_argument('--rolling-surface-relaxation', type=float, default=0.35, help='How strongly daily surface LST nudges the rolling profile')
    parser.add_argument('--rolling-surface-decay-depth', type=float, default=4.0, help='E-folding depth (m) for propagating daily surface nudges downward in rolling mode')
    parser.add_argument('--rolling-deep-inertia', type=float, default=0.65, help='How strongly mixed-layer-below temperatures resist day-to-day changes in rolling prediction')
    parser.add_argument('--rolling-deep-anchor', type=float, default=0.18, help='How strongly rolling prediction damps excessive warming below the mixed layer toward a cold deep-water anchor')
    parser.add_argument('--solar-shading-factor', type=float, default=1.0, help='Multiply incoming shortwave radiation by this factor to simulate shading and reduce net heating')
    parser.add_argument('--shortwave-attenuation-coef', type=float, default=SHORTWAVE_ATTENUATION, help='Shortwave attenuation coefficient (1/m); larger values trap heating nearer the surface')
    parser.add_argument('--shortwave-surface-fraction', type=float, default=SHORTWAVE_SURFACE_FRACTION, help='Fraction of net shortwave absorbed at the surface boundary instead of penetrating into the water column')
    parser.add_argument('--surface-skin-cooling-coef', type=float, default=SURFACE_SKIN_COOLING_COEF, help='Coefficient used to cool satellite skin temperature toward a near-surface bulk-water target under sunny, low-wind conditions')
    parser.add_argument('--surface-air-blend', type=float, default=SURFACE_AIR_BLEND, help='Blend fraction nudging the bulk-surface target toward air temperature to avoid over-trusting skin-temperature spikes')
    parser.add_argument('--surface-obs-depth-m', type=float, default=0.35, help='Effective bulk-water depth assigned to LST observations instead of exact 0 m skin depth')
    parser.add_argument('--time-continuity-weight', type=float, default=5.0, help='Weight for the sequential time-continuity loss term')
    parser.add_argument('--time-continuity-depth-points', type=int, default=64, help='Sample count per epoch for time-continuity depth pairs')
    parser.add_argument('--stratification-weight', type=float, default=0.6, help='Weight for the warm-season stratification loss that discourages overly warm deep water')
    parser.add_argument('--stratification-pairs', type=int, default=64, help='Sample count per epoch for shallow-vs-deep stratification pairs')
    parser.add_argument('--stratification-margin-c', type=float, default=STRATIFICATION_MARGIN_C, help='Target shallow-minus-deep temperature margin during warm stratified conditions')
    parser.add_argument('--smoothness-weight', type=float, default=0.15, help='Weight for penalizing unrealistically sharp vertical temperature jumps')
    parser.add_argument('--max-vertical-gradient-c-per-m', type=float, default=MAX_VERTICAL_GRADIENT_C_PER_M, help='Maximum allowed vertical temperature gradient before the smoothness penalty activates')
    parser.add_argument('--deep-warming-weight', type=float, default=0.25, help='Weight for penalizing warming below the mixed layer during warm stratified periods')
    parser.add_argument('--deep-anchor-weight', type=float, default=0.7, help='Weight for the deep cold-water anchor loss during warm stratified periods')
    parser.add_argument('--deep-anchor-pairs', type=int, default=64, help='Sample count per epoch for deep anchor training points')
    parser.add_argument('--deep-anchor-amplitude-c', type=float, default=2.2, help='Allowed warming above 4 C near the top of the deep cold-water reservoir')
    parser.add_argument('--vertical-exchange-weight', type=float, default=0.35, help='Weight for explicit vertical entrainment/advection exchange near a deepening mixed-layer base')
    parser.add_argument('--entrainment-velocity-scale-m-per-day', type=float, default=MAX_ENTRAINMENT_VELOCITY_M_PER_DAY, help='Upper cap (m/day) for the explicit entrainment velocity used in the vertical exchange term')
    parser.add_argument('--convective-mixing-weight', type=float, default=0.25, help='Weight for the convective mixing loss that homogenizes the cooling mixed layer')
    parser.add_argument('--autumn-overturn-weight', type=float, default=0.22, help='Weight for explicit autumn overturn loss driven by surface cooling, mixed-layer deepening, and gap collapse without fake deep warming')
    parser.add_argument('--heat-budget-weight', type=float, default=0.30, help='Weight for whole-column heat-content / surface-budget closure')
    parser.add_argument('--heat-budget-depth-points', type=int, default=24, help='Depth quadrature points per sampled day used by the heat-budget closure term')
    parser.add_argument('--train-until-best', action='store_true', default=None, help='In train mode, keep PPO/model training until validation stops improving and restore the best checkpoint')
    parser.add_argument('--train-min-epochs', type=int, default=None, help='Minimum training epochs before validation-based early stopping can trigger')
    parser.add_argument('--train-patience-windows', type=int, default=None, help='Number of validation windows without improvement before early stopping in train mode')
    parser.add_argument('--shallow-optimized-grid', action='store_true', help='Use a denser depth grid in the upper water column')
    parser.add_argument('--shallow-focus-depth', type=float, default=5.0, help='Depth range to emphasize in the nonuniform grid (m)')
    parser.add_argument('--shallow-grid-fraction', type=float, default=0.55, help='Fraction of grid points allocated to the shallow-focus layer')
    parser.add_argument('--surface-bulk-correction', action='store_true', help='Fit a shallow-observation-informed correction from satellite LST to bulk surface temperature')
    parser.add_argument('--use-kalman', action='store_true', help='Apply stage-2 Kalman filtering to the PINN temperature profiles')
    parser.add_argument('--use-ppo', action='store_true', help='Use PPO to dynamically tune loss weights during training')
    parser.add_argument('--resume-model-checkpoint', default=None, help='Optional PINN model checkpoint bundle to resume training from in train mode')
    parser.add_argument('--model-checkpoint-path', default=None, help='Path to a saved PINN model checkpoint bundle used for predict-mode inference')
    parser.add_argument('--save-model-checkpoint', default=None, help='Optional path to save the trained PINN model checkpoint bundle after train mode finishes')
    parser.add_argument('--ppo-policy-path', default=None, help='Path to a saved PPO policy bundle used to drive online dynamic tuning in predict mode')
    parser.add_argument('--save-ppo-policy', default=None, help='Optional path to save the trained PPO policy bundle after train mode finishes')
    parser.add_argument('--online-ppo-update', action='store_true', help='Continue updating the PPO policy online during predict mode using proxy rewards; otherwise predict mode only executes the loaded PPO policy')
    parser.add_argument('--online-ppo-control-interval', type=int, default=7, help='Day interval between PPO control decisions during predict mode')
    parser.add_argument('--online-ppo-rollout-steps', type=int, default=4, help='Rollout length before PPO online updates in predict mode')
    parser.add_argument('--online-ppo-max-updates-run', type=int, default=None, help='Maximum PPO update steps allowed in this predict run when online PPO update is enabled')
    parser.add_argument('--ppo-control-interval', type=int, default=50, help='Epoch interval between PPO control decisions')
    parser.add_argument('--ppo-rollout-steps', type=int, default=8, help='Number of PPO transitions collected before each policy update')
    parser.add_argument('--ppo-max-updates-run', type=int, default=None, help='Maximum PPO update steps allowed in this train run; after reaching it, PPO stops learning and only executes the current policy')
    parser.add_argument('--ppo-eval-depth-points', type=int, default=80, help='Depth points used in PPO validation probes')
    parser.add_argument('--ppo-use-kalman-reward', action='store_true', help='Include Kalman-filtered validation RMSE in the PPO reward when profile observations are available')
    parser.add_argument('--kalman-prior-std', type=float, default=2.0, help='Initial state prior std for the Kalman filter (deg C)')
    parser.add_argument('--kalman-process-std', type=float, default=0.3, help='Process noise std for the Kalman filter (deg C)')
    parser.add_argument('--kalman-obs-std-surface', type=float, default=0.5, help='Surface observation std for the Kalman filter (deg C)')
    parser.add_argument('--kalman-obs-std-bottom', type=float, default=0.5, help='Bottom observation std for the Kalman filter (deg C)')
    parser.add_argument('--kalman-obs-std-profile', type=float, default=0.75, help='Profile observation std for the Kalman filter (deg C)')
    parser.add_argument('--kalman-correlation-length', type=float, default=2.0, help='Depth correlation length scale for Kalman covariances (m)')
    parser.add_argument('--kalman-forecast-blend', type=float, default=0.2, help='Blend weight on previous filtered state in the Kalman forecast step')
    parser.add_argument('--kalman-forecast-spinup-days', type=int, default=0, help='Days to keep a stronger forecast blend after the most recent assimilation update')
    parser.add_argument('--kalman-forecast-spinup-max-blend', type=float, default=0.9, help='Maximum persistence blend used during forecast spin-up')
    parser.add_argument('--autumn-asymmetric-cooling', action='store_true', help='In autumn, if surface observations are colder than the model surface, force part of that cooling to propagate downward during Kalman assimilation')
    parser.add_argument('--autumn-doy-threshold', type=float, default=270.0, help='DOY threshold after which asymmetric autumn cooling is enabled')
    parser.add_argument('--autumn-surface-cooling-threshold', type=float, default=1.0, help='Minimum negative surface innovation (deg C) needed to trigger asymmetric autumn cooling')
    parser.add_argument('--autumn-air-temp-threshold', type=float, default=12.0, help='Only trigger asymmetric autumn cooling when air temperature is below this threshold (deg C)')
    parser.add_argument('--autumn-cooling-strength', type=float, default=0.35, help='Strength of the extra downward autumn cooling propagation')
    parser.add_argument('--autumn-cooling-penetration-scale', type=float, default=5.0, help='Penetration scale (m) for extra autumn cooling below the mixed layer')
    args = parser.parse_args()
    interactive_mode = len(sys.argv) == 1
    if interactive_mode:
        args = configure_interactive_args(args)

    era5_path = normalize_input_path(args.era5).resolve() if args.era5 else prompt_for_existing_path('ERA5')
    lst_path = normalize_input_path(args.lst).resolve() if args.lst else prompt_for_existing_path('LST')
    profile_obs_path = normalize_input_path(args.profile_obs).resolve() if args.profile_obs else None

    if args.mode is None:
        if args.practical_prediction_mode:
            args.mode = 'predict'
        elif profile_obs_path is not None:
            args.mode = 'train'
        else:
            args.mode = 'predict'
    if args.mode == 'predict':
        if profile_obs_path is not None:
            print('Predict mode ignores profile-obs and uses only ERA5 + LST.')
        profile_obs_path = None
        args.profile_obs = None
        args = apply_practical_prediction_defaults(args, has_profile_obs=False)
        if args.ppo_policy_path:
            args.use_ppo = True
        if not args.model_checkpoint_path:
            raise ValueError('Predict mode requires --model-checkpoint-path so inference uses a trained PINN instead of retraining.')
    else:
        args = apply_train_mode_defaults(args, has_profile_obs=profile_obs_path is not None)
        if profile_obs_path is not None and args.profile_split_mode == 'none':
            print('Train mode disallows profile_split_mode=none to avoid data leakage; using depth_interleaved instead.')
            args.profile_split_mode = 'depth_interleaved'
        if False and profile_obs_path is None and not interactive_mode and args.profile_split_mode != 'none':
            profile_obs_path = prompt_for_existing_path('鍓栭潰瑙傛祴', optional=True)
            if profile_obs_path is not None:
                args = apply_train_mode_defaults(args, has_profile_obs=True)
    if False and args.mode != 'predict':
        profile_obs_path = prompt_for_existing_path('鍓栭潰瑙傛祴', optional=True)

    if False and args.practical_prediction_mode and profile_obs_path is None:
        args = apply_practical_prediction_defaults(args, has_profile_obs=False)

    args.shortwave_attenuation_coef = float(
        np.clip(args.shortwave_attenuation_coef, MIN_SHORTWAVE_ATTENUATION, MAX_SHORTWAVE_ATTENUATION)
    )
    args.max_vertical_gradient_c_per_m = float(max(args.max_vertical_gradient_c_per_m, 0.1))

    default_output_dir = infer_output_dir(era5_path, lst_path)
    output_dir = normalize_input_path(args.output_dir).resolve() if args.output_dir else prompt_for_output_dir(default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PDF-aligned PINN lake temperature baseline")
    print("Mode:", args.mode)
    print("Implemented terms: LPDE + LBC(SEB/ice + bottom flux) + LIC + Lobs")
    print("Diffusivity: Richardson-number-dependent eddy diffusivity")
    print("Practical prediction mode enabled:" if args.practical_prediction_mode else "Practical prediction mode disabled:", bool(args.practical_prediction_mode))
    print("Rolling prediction mode enabled:" if args.rolling_prediction_mode else "Rolling prediction mode disabled:", bool(args.rolling_prediction_mode))
    print(f"Solar shading factor: {args.solar_shading_factor:.2f}")
    print(f"Shortwave attenuation coef: {args.shortwave_attenuation_coef:.2f}")
    print(f"Shortwave surface fraction: {args.shortwave_surface_fraction:.2f}")
    print(f"Surface skin cooling coef: {args.surface_skin_cooling_coef:.3f}")
    print(f"Surface air blend: {args.surface_air_blend:.2f}")
    print(f"Rolling deep inertia: {args.rolling_deep_inertia:.2f}")
    print(f"Rolling deep anchor: {args.rolling_deep_anchor:.2f}")
    print(f"Wind mixing decay depth: {DIFFUSIVITY_WIND_DECAY_DEPTH:.2f} m")
    print(f"Stratification weight: {args.stratification_weight:.2f} | margin={args.stratification_margin_c:.2f} C")
    print(f"Smoothness weight: {args.smoothness_weight:.2f} | max |dT/dz|={args.max_vertical_gradient_c_per_m:.2f} C/m")
    print(f"Deep warming weight: {args.deep_warming_weight:.2f} | allowance={DEEP_WARMING_ALLOWANCE_C_PER_DAY:.2f} C/day")
    print(f"Deep anchor weight: {args.deep_anchor_weight:.2f} | amplitude={args.deep_anchor_amplitude_c:.2f} C")
    print(
        f"Vertical exchange weight: {args.vertical_exchange_weight:.2f} | "
        f"entrainment cap={args.entrainment_velocity_scale_m_per_day:.2f} m/day"
    )
    print(f"Convective mixing weight: {args.convective_mixing_weight:.2f}")
    print(f"Autumn overturn weight: {args.autumn_overturn_weight:.2f}")
    print(
        f"Heat budget weight: {args.heat_budget_weight:.2f} | "
        f"budget depth points={int(args.heat_budget_depth_points)}"
    )
    if args.mode == "train":
        print(
            f"Train-until-best: {bool(args.train_until_best)} | "
            f"min_epochs={int(args.train_min_epochs)} | "
            f"patience_windows={int(args.train_patience_windows)}"
        )
    print("Shallow optimization enabled:" if args.shallow_optimized_grid else "Shallow optimization disabled:", bool(args.shallow_optimized_grid))
    print("Surface bulk correction enabled:" if args.surface_bulk_correction else "Surface bulk correction disabled:", bool(args.surface_bulk_correction))
    print("PPO stage enabled:" if args.use_ppo else "PPO stage disabled:", bool(args.use_ppo))
    if args.use_ppo:
        print(f"PPO max updates this run: {args.ppo_max_updates_run if args.ppo_max_updates_run is not None else 'unlimited'}")
    print("Kalman stage enabled:" if args.use_kalman else "Kalman stage disabled:", bool(args.use_kalman))
    print("Bottom observation enabled:" if args.use_bottom_observation else "Bottom observation disabled:", bool(args.use_bottom_observation))
    print("Initial condition mode:", args.initial_condition_mode)
    print("Seasonal segmented enabled:" if args.seasonal_segmented else "Seasonal segmented disabled:", bool(args.seasonal_segmented))
    if args.kalman_forecast_spinup_days > 0:
        print(
            "Kalman forecast spin-up | "
            f"days={args.kalman_forecast_spinup_days} | "
            f"max_blend={args.kalman_forecast_spinup_max_blend:.2f}"
        )
    if args.online_ppo_update:
        print(f"Online PPO max updates this run: {args.online_ppo_max_updates_run if args.online_ppo_max_updates_run is not None else 'unlimited'}")
    print("Training device:", args.device)
    print("=" * 72)

    df, metadata = load_training_frame(era5_path, lst_path)
    df = apply_forcing_adjustments(
        df,
        solar_shading_factor=args.solar_shading_factor,
        surface_skin_cooling_coef=args.surface_skin_cooling_coef,
        surface_air_blend=args.surface_air_blend,
    )
    full_profile_obs = load_optional_profile_observations(
        profile_obs_path,
        start_date=metadata['start_date'],
        time_scale_seconds=metadata['time_scale_seconds'],
        max_depth=args.max_depth,
    )
    profile_splits, profile_split_info = split_profile_observations(
        full_profile_obs,
        split_mode=args.profile_split_mode,
    )
    train_profile_obs = profile_splits['train']
    val_profile_obs = profile_splits['val']
    assim_profile_obs = profile_splits['assim']
    test_profile_obs = profile_splits['test']

    if has_profile_observations(full_profile_obs):
        print(f"Profile split mode: {profile_split_info['mode']}")
        for role in PROFILE_SPLIT_ROLES:
            summary = profile_split_info['summary'][role]
            print(
                f"  {role}: rows={summary['rows']} | "
                f"depths={summary['depth_count']} | "
                f"dates={summary['date_count']}"
            )

    validation_label = 'Profile held-out validation' if args.profile_split_mode != 'none' else 'Profile validation'
    kalman_validation_label = 'Kalman held-out validation' if args.profile_split_mode != 'none' else 'Kalman validation'
    training_use_ppo = bool(args.use_ppo and args.mode == 'train')

    model_checkpoint_bundle = None
    resume_model_checkpoint_bundle = None
    online_ppo_controller = None
    online_ppo_bundle = None
    if args.mode == 'predict' and args.use_ppo and args.ppo_policy_path:
        online_ppo_controller, online_ppo_bundle = load_ppo_policy_bundle(args.ppo_policy_path, device=args.device)
        print(
            "Predict PPO mode: online policy update enabled"
            if args.online_ppo_update
            else "Predict PPO mode: execute loaded PPO policy only (no online updates)"
        )

    model = None
    training_info = {
        'final_weights': {},
        'kalman_scales': {'process': 1.0, 'obs': 1.0},
        'ppo_history': pd.DataFrame(),
        'ppo_update_stats': pd.DataFrame(),
        'use_ppo': False,
        'ppo_policy_bundle': None,
        'surface_correction_info': None,
        'best_selection_metric': None,
        'best_selection_label': None,
        'ppo_update_count': 0,
    }
    online_ppo_runtime = {
        'diagnostics': pd.DataFrame(),
        'history': pd.DataFrame(),
        'kalman_scales': {'process': 1.0, 'obs': 1.0},
    }
    if args.mode == 'predict':
        model, model_checkpoint_bundle = load_model_checkpoint_bundle(args.model_checkpoint_path, device=args.device)
        checkpoint_info = dict((model_checkpoint_bundle or {}).get('training_info', {}) or {})
        training_info.update({
            'final_weights': dict(checkpoint_info.get('final_weights', {})),
            'kalman_scales': dict(checkpoint_info.get('kalman_scales', {'process': 1.0, 'obs': 1.0})),
            'surface_correction_info': checkpoint_info.get('surface_correction_info'),
            'best_selection_metric': checkpoint_info.get('best_selection_metric'),
            'best_selection_label': checkpoint_info.get('best_selection_label'),
            'ppo_policy_bundle': checkpoint_info.get('ppo_policy_bundle'),
        })
        if args.use_ppo and online_ppo_bundle is None and training_info.get('ppo_policy_bundle') is not None:
            online_ppo_controller, online_ppo_bundle = build_ppo_controller_from_bundle(
                training_info['ppo_policy_bundle'],
                device=args.device,
            )
            print(
                "Predict PPO mode: online policy update enabled"
                if args.online_ppo_update
                else "Predict PPO mode: execute checkpoint-embedded PPO policy only (no online updates)"
            )
        temp_grid, depths, online_ppo_runtime = predict_temperature_grid(
            model,
            df=df,
            max_depth=args.max_depth,
            n_depth_points=args.depth_points,
            device=args.device,
            apply_post_physics=args.apply_post_physics,
            use_shallow_optimized=args.shallow_optimized_grid,
            shallow_focus_depth=args.shallow_focus_depth,
            shallow_fraction=args.shallow_grid_fraction,
            rolling_prediction_mode=args.rolling_prediction_mode,
            rolling_memory_blend=args.rolling_memory_blend,
            rolling_surface_relaxation=args.rolling_surface_relaxation,
            rolling_surface_decay_depth=args.rolling_surface_decay_depth,
            rolling_deep_inertia=args.rolling_deep_inertia,
            rolling_deep_anchor=args.rolling_deep_anchor,
            surface_skin_cooling_coef=args.surface_skin_cooling_coef,
            ppo_controller=online_ppo_controller,
            ppo_policy_bundle=online_ppo_bundle,
            online_ppo_update=args.online_ppo_update,
            online_ppo_control_interval=args.online_ppo_control_interval,
            online_ppo_rollout_steps=args.online_ppo_rollout_steps,
            online_ppo_max_updates_run=args.online_ppo_max_updates_run,
        )
        precomputed_kalman_grid = None
    elif args.seasonal_segmented:
        seasonal_outputs = run_seasonal_segmented_pipeline(
            df=df,
            metadata=metadata,
            max_depth=args.max_depth,
            depth_points=args.depth_points,
            epochs=args.epochs,
            lr=args.lr,
            collocation_points=args.collocation_points,
            device=args.device,
            train_profile_obs=train_profile_obs,
            val_profile_obs=val_profile_obs,
            assim_profile_obs=assim_profile_obs,
            use_kalman=args.use_kalman,
            use_ppo=False,
            ppo_control_interval=args.ppo_control_interval,
            ppo_rollout_steps=args.ppo_rollout_steps,
            ppo_max_updates_run=args.ppo_max_updates_run,
            ppo_eval_depth_points=args.ppo_eval_depth_points,
            ppo_use_kalman_reward=(args.ppo_use_kalman_reward and training_use_ppo),
            ppo_apply_post_physics=args.apply_post_physics,
            kalman_prior_std=args.kalman_prior_std,
            kalman_process_std=args.kalman_process_std,
            kalman_obs_std_surface=args.kalman_obs_std_surface,
            kalman_obs_std_bottom=args.kalman_obs_std_bottom,
            kalman_obs_std_profile=args.kalman_obs_std_profile,
            kalman_correlation_length=args.kalman_correlation_length,
            kalman_forecast_blend=args.kalman_forecast_blend,
            kalman_forecast_spinup_days=args.kalman_forecast_spinup_days,
            kalman_forecast_spinup_max_blend=args.kalman_forecast_spinup_max_blend,
            shallow_optimized_grid=args.shallow_optimized_grid,
            shallow_focus_depth=args.shallow_focus_depth,
            shallow_grid_fraction=args.shallow_grid_fraction,
            rolling_prediction_mode=args.rolling_prediction_mode,
            rolling_memory_blend=args.rolling_memory_blend,
            rolling_surface_relaxation=args.rolling_surface_relaxation,
            rolling_surface_decay_depth=args.rolling_surface_decay_depth,
            rolling_deep_inertia=args.rolling_deep_inertia,
            rolling_deep_anchor=args.rolling_deep_anchor,
            surface_skin_cooling_coef=args.surface_skin_cooling_coef,
            shortwave_attenuation_coef=args.shortwave_attenuation_coef,
            shortwave_surface_fraction=args.shortwave_surface_fraction,
            use_surface_bulk_correction=args.surface_bulk_correction,
            use_bottom_observation=args.use_bottom_observation,
            initial_condition_mode=args.initial_condition_mode,
            surface_obs_depth_m=args.surface_obs_depth_m,
            time_continuity_weight=args.time_continuity_weight,
            time_continuity_depth_points=args.time_continuity_depth_points,
            stratification_weight=args.stratification_weight,
            stratification_pairs=args.stratification_pairs,
            stratification_margin_c=args.stratification_margin_c,
            smoothness_weight=args.smoothness_weight,
            max_vertical_gradient_c_per_m=args.max_vertical_gradient_c_per_m,
            deep_warming_weight=args.deep_warming_weight,
            deep_anchor_weight=args.deep_anchor_weight,
            deep_anchor_pairs=args.deep_anchor_pairs,
            deep_anchor_amplitude_c=args.deep_anchor_amplitude_c,
            vertical_exchange_weight=args.vertical_exchange_weight,
            entrainment_velocity_scale_m_per_day=args.entrainment_velocity_scale_m_per_day,
            convective_mixing_weight=args.convective_mixing_weight,
            autumn_overturn_weight=args.autumn_overturn_weight,
            heat_budget_weight=args.heat_budget_weight,
            heat_budget_depth_points=args.heat_budget_depth_points,
            train_until_best=args.train_until_best,
            train_min_epochs=args.train_min_epochs,
            train_patience_windows=args.train_patience_windows,
            apply_post_physics=args.apply_post_physics,
        )
        training_info = seasonal_outputs['training_info']
        temp_grid = seasonal_outputs['temp_grid']
        depths = seasonal_outputs['depths']
        precomputed_kalman_grid = seasonal_outputs['kalman_grid']
        if training_use_ppo:
            print('Pure-forecast PPO training is currently only enabled for the non-seasonal train pipeline; skipping PPO stage.')
    else:
        if args.mode == 'train' and args.resume_model_checkpoint:
            _, resume_model_checkpoint_bundle = load_model_checkpoint_bundle(args.resume_model_checkpoint, device=args.device)
            print(f"Resuming train mode from PINN model checkpoint: {normalize_input_path(args.resume_model_checkpoint).resolve()}")
        model, training_info = train_model(
            df=df,
            metadata=metadata,
            max_depth=args.max_depth,
            epochs=args.epochs,
            lr=args.lr,
            collocation_points=args.collocation_points,
            device=args.device,
            train_profile_obs=train_profile_obs,
            ppo_validation_profile_obs=val_profile_obs,
            use_ppo=False,
            ppo_control_interval=args.ppo_control_interval,
            ppo_rollout_steps=args.ppo_rollout_steps,
            ppo_max_updates_run=args.ppo_max_updates_run,
            ppo_eval_depth_points=args.ppo_eval_depth_points,
            ppo_use_kalman_reward=(args.ppo_use_kalman_reward and training_use_ppo),
            ppo_apply_post_physics=args.apply_post_physics,
            base_kalman_process_std=args.kalman_process_std,
            base_kalman_obs_std_surface=args.kalman_obs_std_surface,
            base_kalman_obs_std_bottom=args.kalman_obs_std_bottom,
            base_kalman_obs_std_profile=args.kalman_obs_std_profile,
            base_kalman_correlation_length=args.kalman_correlation_length,
            base_kalman_forecast_blend=args.kalman_forecast_blend,
            base_kalman_forecast_spinup_days=args.kalman_forecast_spinup_days,
            base_kalman_forecast_spinup_max_blend=args.kalman_forecast_spinup_max_blend,
            shallow_optimized_grid=args.shallow_optimized_grid,
            shallow_focus_depth=args.shallow_focus_depth,
            shallow_grid_fraction=args.shallow_grid_fraction,
            rolling_prediction_mode=args.rolling_prediction_mode,
            rolling_memory_blend=args.rolling_memory_blend,
            rolling_surface_relaxation=args.rolling_surface_relaxation,
            rolling_surface_decay_depth=args.rolling_surface_decay_depth,
            rolling_deep_inertia=args.rolling_deep_inertia,
            rolling_deep_anchor=args.rolling_deep_anchor,
            surface_skin_cooling_coef=args.surface_skin_cooling_coef,
            shortwave_attenuation_coef=args.shortwave_attenuation_coef,
            shortwave_surface_fraction=args.shortwave_surface_fraction,
            use_surface_bulk_correction=args.surface_bulk_correction,
            use_bottom_observation=args.use_bottom_observation,
            initial_condition_mode=args.initial_condition_mode,
            surface_obs_depth_m=args.surface_obs_depth_m,
            time_continuity_weight=args.time_continuity_weight,
            time_continuity_depth_points=args.time_continuity_depth_points,
            stratification_weight=args.stratification_weight,
            stratification_pairs=args.stratification_pairs,
            stratification_margin_c=args.stratification_margin_c,
            smoothness_weight=args.smoothness_weight,
            max_vertical_gradient_c_per_m=args.max_vertical_gradient_c_per_m,
            deep_warming_weight=args.deep_warming_weight,
            deep_anchor_weight=args.deep_anchor_weight,
            deep_anchor_pairs=args.deep_anchor_pairs,
            deep_anchor_amplitude_c=args.deep_anchor_amplitude_c,
            vertical_exchange_weight=args.vertical_exchange_weight,
            entrainment_velocity_scale_m_per_day=args.entrainment_velocity_scale_m_per_day,
            convective_mixing_weight=args.convective_mixing_weight,
            autumn_overturn_weight=args.autumn_overturn_weight,
            heat_budget_weight=args.heat_budget_weight,
            heat_budget_depth_points=args.heat_budget_depth_points,
            train_until_best=args.train_until_best,
            train_min_epochs=args.train_min_epochs,
            train_patience_windows=args.train_patience_windows,
            resume_checkpoint_bundle=resume_model_checkpoint_bundle,
        )
        if training_use_ppo and has_profile_observations(val_profile_obs):
            print('Starting pure-forecast PPO training on validation profiles after PINN training...')
            forecast_ppo_info = train_pure_forecast_ppo_policy(
                model=model,
                df=df,
                metadata=metadata,
                max_depth=args.max_depth,
                depth_points=args.depth_points,
                device=args.device,
                validation_profile_obs=val_profile_obs,
                initial_weights=training_info['final_weights'],
                initial_kalman_scales=training_info['kalman_scales'],
                apply_post_physics=args.apply_post_physics,
                use_shallow_optimized=args.shallow_optimized_grid,
                shallow_focus_depth=args.shallow_focus_depth,
                shallow_fraction=args.shallow_grid_fraction,
                rolling_prediction_mode=args.rolling_prediction_mode,
                rolling_memory_blend=args.rolling_memory_blend,
                rolling_surface_relaxation=args.rolling_surface_relaxation,
                rolling_surface_decay_depth=args.rolling_surface_decay_depth,
                rolling_deep_inertia=args.rolling_deep_inertia,
                rolling_deep_anchor=args.rolling_deep_anchor,
                surface_skin_cooling_coef=args.surface_skin_cooling_coef,
                ppo_control_interval=args.online_ppo_control_interval,
                ppo_rollout_steps=args.online_ppo_rollout_steps,
                ppo_max_updates_run=args.ppo_max_updates_run,
                initial_ppo_policy_bundle=None if resume_model_checkpoint_bundle is None else dict((resume_model_checkpoint_bundle.get('training_info', {}) or {})).get('ppo_policy_bundle'),
            )
            if forecast_ppo_info is not None:
                training_info['ppo_history'] = forecast_ppo_info['ppo_history']
                training_info['ppo_update_stats'] = forecast_ppo_info['ppo_update_stats']
                training_info['ppo_policy_bundle'] = forecast_ppo_info['ppo_policy_bundle']
                training_info['ppo_update_count'] = forecast_ppo_info['ppo_update_count']
                training_info['kalman_scales'] = forecast_ppo_info['kalman_scales']
                training_info['best_selection_metric'] = None if forecast_ppo_info['best_validation_metrics'] is None else float(forecast_ppo_info['best_validation_metrics']['rmse'])
                training_info['best_selection_label'] = 'forecast_val_rmse'
                online_ppo_controller = forecast_ppo_info['ppo_controller']
                online_ppo_bundle = forecast_ppo_info['ppo_policy_bundle']
            else:
                print('Skipping pure-forecast PPO training because validation profile observations are unavailable.')

        temp_grid, depths, online_ppo_runtime = predict_temperature_grid(
            model,
            df=df,
            max_depth=args.max_depth,
            n_depth_points=args.depth_points,
            device=args.device,
            apply_post_physics=args.apply_post_physics,
            use_shallow_optimized=args.shallow_optimized_grid,
            shallow_focus_depth=args.shallow_focus_depth,
            shallow_fraction=args.shallow_grid_fraction,
            rolling_prediction_mode=args.rolling_prediction_mode,
            rolling_memory_blend=args.rolling_memory_blend,
            rolling_surface_relaxation=args.rolling_surface_relaxation,
            rolling_surface_decay_depth=args.rolling_surface_decay_depth,
            rolling_deep_inertia=args.rolling_deep_inertia,
            rolling_deep_anchor=args.rolling_deep_anchor,
            surface_skin_cooling_coef=args.surface_skin_cooling_coef,
            ppo_controller=online_ppo_controller,
            ppo_policy_bundle=online_ppo_bundle,
            online_ppo_update=False if args.mode == 'train' else args.online_ppo_update,
            online_ppo_control_interval=args.online_ppo_control_interval,
            online_ppo_rollout_steps=args.online_ppo_rollout_steps,
            online_ppo_max_updates_run=args.online_ppo_max_updates_run,
        )
        precomputed_kalman_grid = None

    year_path = output_dir / f"{metadata['file_tag']}_year_heatmap.png"
    monthly_heatmap_path = output_dir / f"{metadata['file_tag']}_monthly_heatmaps.png"
    full_csv_path = export_temperature_tables(df, temp_grid, depths, output_dir, metadata, suffix='pinn')
    plot_year_heatmap(df, temp_grid, depths, year_path, metadata)
    plot_monthly_heatmaps(df, temp_grid, depths, monthly_heatmap_path, metadata)
    validation_metrics = evaluate_profile_grid(df, metadata, temp_grid, depths, args.max_depth, test_profile_obs)

    saved_model_checkpoint_path = None
    if args.mode == 'train':
        model_checkpoint_output = args.save_model_checkpoint
        if model_checkpoint_output is None:
            model_checkpoint_output = str(output_dir / f"{metadata['file_tag']}_pinn_model_checkpoint.pt")
        saved_model_checkpoint_path = save_model_checkpoint_bundle(model, training_info, model_checkpoint_output)

    saved_ppo_policy_path = None
    if training_use_ppo and args.save_ppo_policy:
        saved_ppo_policy_path = save_ppo_policy_bundle(training_info.get('ppo_policy_bundle'), args.save_ppo_policy)

    kalman_validation_metrics = None
    learned_kalman_process_scale = training_info['kalman_scales']['process']
    learned_kalman_obs_scale = training_info['kalman_scales']['obs']
    daily_process_scale = None
    daily_obs_scale = None
    if not online_ppo_runtime['diagnostics'].empty:
        daily_process_scale = online_ppo_runtime['diagnostics']['kalman_process_scale'].to_numpy(dtype=np.float64)
        daily_obs_scale = online_ppo_runtime['diagnostics']['kalman_obs_scale'].to_numpy(dtype=np.float64)
    if args.use_kalman:
        if precomputed_kalman_grid is None:
            kalman_grid, kalman_diagnostics = run_profile_kalman_filter(
                df=df,
                temp_grid=temp_grid,
                depths=depths,
                metadata=metadata,
                max_depth=args.max_depth,
                profile_obs_data=assim_profile_obs,
                prior_std=args.kalman_prior_std,
                process_std=args.kalman_process_std * learned_kalman_process_scale,
                obs_std_surface=args.kalman_obs_std_surface * learned_kalman_obs_scale,
                obs_std_bottom=args.kalman_obs_std_bottom * learned_kalman_obs_scale,
                obs_std_profile=args.kalman_obs_std_profile * learned_kalman_obs_scale,
                correlation_length=args.kalman_correlation_length,
                forecast_blend=args.kalman_forecast_blend,
                forecast_spinup_days=args.kalman_forecast_spinup_days,
                forecast_spinup_max_blend=args.kalman_forecast_spinup_max_blend,
                use_surface_bulk_correction=args.surface_bulk_correction,
                use_bottom_observation=args.use_bottom_observation,
                surface_obs_depth_m=args.surface_obs_depth_m,
                daily_process_scale=daily_process_scale,
                daily_obs_scale=daily_obs_scale,
                autumn_asymmetric_cooling=args.autumn_asymmetric_cooling,
                autumn_doy_threshold=args.autumn_doy_threshold,
                autumn_surface_cooling_threshold=args.autumn_surface_cooling_threshold,
                autumn_air_temp_threshold=args.autumn_air_temp_threshold,
                autumn_cooling_strength=args.autumn_cooling_strength,
                autumn_cooling_penetration_scale=args.autumn_cooling_penetration_scale,
            )
        else:
            kalman_grid = precomputed_kalman_grid
            kalman_diagnostics = pd.DataFrame()

        kalman_validation_metrics = evaluate_profile_grid(
            df,
            metadata,
            kalman_grid,
            depths,
            args.max_depth,
            test_profile_obs,
        )

    print(f"\nSaved annual heatmap to: {year_path}")
    print(f"Saved monthly heatmaps to: {monthly_heatmap_path}")
    print(f"Saved full prediction table to: {full_csv_path}")
    if training_info.get('surface_correction_info') is not None:
        info = training_info['surface_correction_info']
        print(
            "Surface correction fit | "
            f"matches={info['n_matches']} | "
            f"raw_rmse={info['raw_rmse']:.3f} | "
            f"fit_rmse={info['fit_rmse']:.3f}"
        )
    if training_use_ppo:
        weight_order = [
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
        ]
        weight_parts = [
            f"lambda_{key}={training_info['final_weights'][key]:.3e}"
            for key in weight_order
            if key in training_info['final_weights']
        ]
        print("PPO tuned weights | " + " | ".join(weight_parts))
        print(
            "PPO tuned Kalman scales | "
            f"process_scale={learned_kalman_process_scale:.3f} | "
            f"obs_scale={learned_kalman_obs_scale:.3f}"
        )
    if saved_ppo_policy_path is not None:
        print(f"Saved PPO policy bundle to: {saved_ppo_policy_path}")
    if saved_model_checkpoint_path is not None:
        print(f"Saved PINN model checkpoint to: {saved_model_checkpoint_path}")
    if args.mode == 'predict' and args.model_checkpoint_path:
        print(f"Loaded PINN model checkpoint from: {normalize_input_path(args.model_checkpoint_path).resolve()}")
    if args.mode == 'predict' and args.ppo_policy_path:
        print(f"Loaded PPO policy bundle from: {normalize_input_path(args.ppo_policy_path).resolve()}")
        if not online_ppo_runtime['diagnostics'].empty:
            first_diag = online_ppo_runtime['diagnostics'].iloc[0]
            print(
                "Predict PPO mapped controls | "
                f"memory_blend={first_diag['memory_blend']:.3f} | "
                f"surface_relaxation={first_diag['surface_relaxation']:.3f} | "
                f"deep_inertia={first_diag['deep_inertia']:.3f} | "
                f"deep_anchor={first_diag['deep_anchor']:.3f} | "
                f"surface_skin_cooling_coef={first_diag['surface_skin_cooling_coef']:.3f}"
            )
        if not online_ppo_runtime['history'].empty:
            last_online = online_ppo_runtime['history'].iloc[-1]
            print(
                "Online PPO controls | "
                f"memory_blend={last_online['memory_blend']:.3f} | "
                f"surface_relaxation={last_online['surface_relaxation']:.3f} | "
                f"deep_inertia={last_online['deep_inertia']:.3f} | "
                f"deep_anchor={last_online['deep_anchor']:.3f} | "
                f"surface_skin_cooling_coef={last_online['surface_skin_cooling_coef']:.3f}"
            )
    if validation_metrics is not None:
        print(
            f"{validation_label} | "
            f"matched={validation_metrics['matched_rows']} | "
            f"RMSE={validation_metrics['rmse']:.3f} | "
            f"MAE={validation_metrics['mae']:.3f} | "
            f"bias={validation_metrics['bias']:.3f}"
        )
    if args.use_kalman:
        if kalman_validation_metrics is not None:
            print(
                f"{kalman_validation_label} | "
                f"matched={kalman_validation_metrics['matched_rows']} | "
                f"RMSE={kalman_validation_metrics['rmse']:.3f} | "
                f"MAE={kalman_validation_metrics['mae']:.3f} | "
                f"bias={kalman_validation_metrics['bias']:.3f}"
            )
    print("=" * 72)


if __name__ == '__main__':
    main()


