# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

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
    if getattr(args, 'profile_split_mode', None) in (None, ''):
        args.profile_split_mode = 'seasonal_blocked' if has_profile_obs else 'none'
    args.seasonal_segmented = False
    args.use_bottom_observation = False
    if args.initial_condition_mode is None:
        args.initial_condition_mode = 'uniform_4c'
    args.use_kalman = bool(args.use_kalman)
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
    args.solar_shading_factor = 0.65
    args.shortwave_attenuation_coef = 1.5
    args.shortwave_surface_fraction = 0.72
    args.surface_skin_cooling_coef = 0.02
    args.surface_air_blend = 0.25
    args.stratification_weight = 0.8
    args.stratification_pairs = 64
    args.stratification_margin_c = 1.2
    args.smoothness_weight = 0.25
    if args.max_vertical_gradient_c_per_m is None:
        args.max_vertical_gradient_c_per_m = 5.0
    if args.deep_warming_weight is None:
        args.deep_warming_weight = 0.45
    args.deep_anchor_weight = 0.15
    args.deep_anchor_pairs = 64
    args.deep_anchor_amplitude_c = 2.5
    args.vertical_exchange_weight = 0.22
    args.entrainment_velocity_scale_m_per_day = 0.80
    args.convective_mixing_weight = 0.18
    args.surface_mixed_layer_uniformity_weight = 0.16
    args.abrupt_surface_cooling_weight = 0.12
    args.bottom_slow_change_weight = 0.10
    if args.mid_deep_temporal_smoothness_weight is None:
        args.mid_deep_temporal_smoothness_weight = 0.0
    if args.autumn_downward_cooling_weight is None:
        args.autumn_downward_cooling_weight = 0.0
    if args.warm_deep_winter_weak_gradient_weight is None:
        args.warm_deep_winter_weak_gradient_weight = 0.0
    if args.thermocline_shape_weight is None:
        args.thermocline_shape_weight = 0.0
    if args.autumn_overturn_weight is None:
        args.autumn_overturn_weight = 0.18
    if args.heat_budget_weight is None:
        args.heat_budget_weight = 0.20
    if args.heat_budget_depth_points is None:
        args.heat_budget_depth_points = 24
    return args


def apply_train_mode_defaults(args: argparse.Namespace, has_profile_obs: bool) -> argparse.Namespace:
    args.mode = 'train'
    args.practical_prediction_mode = False
    if getattr(args, 'profile_split_mode', None) in (None, ''):
        args.profile_split_mode = 'seasonal_blocked' if has_profile_obs else 'none'
    args.seasonal_segmented = False
    args.use_bottom_observation = False
    if args.initial_condition_mode is None:
        args.initial_condition_mode = 'uniform_4c'
    args.use_kalman = bool(args.use_kalman)
    args.use_ppo = bool(args.use_ppo)
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
    args.solar_shading_factor = 0.80
    args.shortwave_attenuation_coef = 1.0
    args.shortwave_surface_fraction = 0.62
    args.surface_skin_cooling_coef = 0.018
    args.surface_air_blend = 0.22
    args.time_continuity_weight = 0.90
    args.stratification_weight = 0.0
    args.stratification_pairs = 64
    args.stratification_margin_c = STRATIFICATION_MARGIN_C
    args.smoothness_weight = 0.30
    if args.max_vertical_gradient_c_per_m is None:
        args.max_vertical_gradient_c_per_m = 3.5
    if args.deep_warming_weight is None:
        args.deep_warming_weight = 0.16
    args.deep_anchor_weight = 0.0
    args.deep_anchor_pairs = 64
    args.deep_anchor_amplitude_c = 2.2
    args.vertical_exchange_weight = 0.26
    args.entrainment_velocity_scale_m_per_day = 0.75
    args.convective_mixing_weight = 0.22
    args.surface_mixed_layer_uniformity_weight = 0.22
    args.abrupt_surface_cooling_weight = 0.20
    args.bottom_slow_change_weight = 0.50
    if args.mid_deep_temporal_smoothness_weight is None:
        args.mid_deep_temporal_smoothness_weight = 0.35
    if args.autumn_downward_cooling_weight is None:
        args.autumn_downward_cooling_weight = 0.0
    if args.warm_deep_winter_weak_gradient_weight is None:
        args.warm_deep_winter_weak_gradient_weight = 0.0
    if args.thermocline_shape_weight is None:
        args.thermocline_shape_weight = 0.0
    if args.autumn_overturn_weight is None:
        args.autumn_overturn_weight = 0.22
    if args.heat_budget_weight is None:
        args.heat_budget_weight = 0.08
    if args.heat_budget_depth_points is None:
        args.heat_budget_depth_points = 24
    if args.profile_grid_physics_weight is None:
        args.profile_grid_physics_weight = 0.85
    if args.profile_grid_day_pairs is None:
        args.profile_grid_day_pairs = 18
    if args.profile_grid_depth_points is None:
        args.profile_grid_depth_points = 51
    if args.density_reg_weight is None:
        args.density_reg_weight = 3.20
    if args.train_until_best is None:
        args.train_until_best = True
    if args.train_min_epochs is None:
        args.train_min_epochs = 200
    if args.train_patience_windows is None:
        args.train_patience_windows = 6
    return args


def apply_train_like_prediction_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Use train-mode forcing/loss defaults while staying in predict mode."""
    model_checkpoint_path = args.model_checkpoint_path
    ppo_policy_path = args.ppo_policy_path
    save_ppo_policy = args.save_ppo_policy
    output_dir = args.output_dir
    predict_defaults_profile = args.predict_defaults_profile

    args = apply_train_mode_defaults(args, has_profile_obs=False)
    args.mode = 'predict'
    args.practical_prediction_mode = True
    args.profile_split_mode = 'none'
    args.use_ppo = bool(ppo_policy_path)
    args.online_ppo_update = False
    args.model_checkpoint_path = model_checkpoint_path
    args.ppo_policy_path = ppo_policy_path
    args.save_ppo_policy = save_ppo_policy
    args.output_dir = output_dir
    args.predict_defaults_profile = predict_defaults_profile
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
    args.max_depth = None
    args.ppo_control_interval = 50
    args.ppo_rollout_steps = 4
    args.ppo_eval_depth_points = 80
    if args.profile_obs:
        args.mode = 'train'
        args = apply_train_mode_defaults(args, has_profile_obs=True)
        args.epochs = prompt_int_value('Training epochs', 600, minimum=1)
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
        args.ppo_policy_path = None
        has_embedded_ppo = checkpoint_has_embedded_ppo_policy(args.model_checkpoint_path)
        args.use_ppo = bool(has_embedded_ppo)
        if has_embedded_ppo:
            print('Predict mode: checkpoint contains an embedded PPO policy and it will be used automatically.')
        else:
            print('Predict mode: checkpoint has no embedded PPO policy; running without PPO control.')

    print(f"Mode selected: {args.mode}")
    print('Recommended defaults applied:')
    print(f'  output_dir={args.output_dir}')
    print(f'  device={args.device}')
    print('  epochs=600 | collocation_points=128 | depth_points=80 | max_depth=metadata max_depth_m')
    print(f'  profile_split_mode={args.profile_split_mode} | seasonal_segmented={args.seasonal_segmented}')
    print('  use_bottom_observation=False | initial_condition_mode=uniform_4c')
    print(f'  use_kalman={args.use_kalman} | use_ppo={args.use_ppo}')
    print('  kalman_forecast_spinup_days=14 | kalman_forecast_spinup_max_blend=0.95')
    print('  raw PINN output is the default; no predict-side physics correction is applied')
    print('  solar_shading_factor=0.65 | shortwave_attenuation_coef=1.5 | kalman_obs_std_surface=2.5')
    return args
