# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *
from .checkpoint import (
    load_model_checkpoint_bundle,
    load_ppo_policy_bundle,
    save_model_checkpoint_bundle,
    save_ppo_policy_bundle,
)
from .cli_config import (
    apply_practical_prediction_defaults,
    apply_train_like_prediction_defaults,
    apply_train_mode_defaults,
    configure_interactive_args,
    infer_output_dir,
    normalize_input_path,
    prompt_for_existing_path,
    prompt_for_output_dir,
)
from .data_io import (
    has_profile_observations,
    load_optional_profile_observations,
    load_training_frame,
    split_profile_observations,
)
from .export import export_temperature_tables
from .forcing import apply_forcing_adjustments
from .kalman import run_profile_kalman_filter
from .online_control import train_pure_forecast_ppo_policy
from .plotting import plot_year_heatmap
from .ppo import build_ppo_controller_from_bundle, normalize_kalman_scales
from .predict import predict_temperature_grid, run_seasonal_segmented_pipeline
from .scorecard_integration import run_scorecard_report
from .train import train_model
from .validation import evaluate_profile_grid

def main():
    parser = argparse.ArgumentParser(description='PDF-aligned PINN baseline for lake temperature reconstruction')
    parser.add_argument('--era5', default=None, help='Path to ERA5 forcing data CSV')
    parser.add_argument('--lst', default=None, help='Path to surface LST observation CSV')
    parser.add_argument('--profile-obs', default=None, help='Optional profile observation CSV with Date/Depth_m/Temperature_C')
    parser.add_argument('--mode', choices=['train', 'predict'], default=None, help='train: use profile observations for training/evaluation when available; predict: ignore profile observations and use only LST + ERA5 for inference')
    parser.add_argument('--practical-prediction-mode', action='store_true', help='Use a continuous practical prediction setup: no seasonal segmentation, no bottom observation, Kalman on, PPO off')
    parser.add_argument('--profile-split-mode', choices=['none', 'depth_interleaved', 'time_blocked'], default='time_blocked', help='How to split profile observations into train/val/assim/test roles to avoid leakage')
    parser.add_argument('--seasonal-segmented', action='store_true', help='Train separate models for contiguous seasonal blocks and stitch them into a full-year prediction')
    parser.add_argument('--use-bottom-observation', action='store_true', help='Use BottomTemp_C as an observation boundary; disable for strict ERA5+LST prediction')
    parser.add_argument('--initial-condition-mode', choices=['uniform_4c', 'surface_to_uniform_4c', 'linear_to_bottom_obs'], default='uniform_4c', help='How to construct the initial temperature profile')
    parser.add_argument('--max-depth', type=float, default=20.0, help='Maximum lake depth in meters')
    parser.add_argument('--depth-points', type=int, default=150, help='Number of depth samples for exported profiles')
    parser.add_argument('--model-input-dim', type=int, default=PINN_INPUT_DIM, help='PINN input dimension: default 17 for extended forcing; use 11 only for archived run9-compatible retraining')
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
    parser.add_argument('--surface-mixed-layer-uniformity-weight', type=float, default=0.18, help='Weight for homogenizing temperatures inside the surface mixed layer around a shallow mixed-layer reference temperature')
    parser.add_argument('--abrupt-surface-cooling-weight', type=float, default=0.12, help='Weight for penalizing unrealistically abrupt cooling within the warm surface mixed layer')
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
    parser.add_argument('--predict-defaults-profile', choices=['practical', 'train_like'], default='practical', help='Default parameter profile used in predict mode: practical forecast defaults or train-like rolling/forcing defaults')
    parser.add_argument('--save-model-checkpoint', default=None, help='Optional path to save the trained PINN model checkpoint bundle after train mode finishes')
    parser.add_argument('--train-export-artifacts', action='store_true', help='In train mode, also export post-training prediction CSV/heatmaps. By default train mode saves only the checkpoint.')
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
    parser.add_argument('--ppo-tune-kalman', action='store_true', help='Allow training-stage PPO to tune Kalman process/observation scales during main PINN training')
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
    predict_score_profile_obs_path = None

    if args.mode is None:
        if args.practical_prediction_mode:
            args.mode = 'predict'
        elif profile_obs_path is not None:
            args.mode = 'train'
        else:
            args.mode = 'predict'
    if args.mode == 'predict':
        if profile_obs_path is not None:
            print('Predict mode uses profile-obs only for output scoring/selection; inference still uses only ERA5 + LST.')
            predict_score_profile_obs_path = profile_obs_path
        profile_obs_path = None
        args.profile_obs = None
        if args.predict_defaults_profile == 'train_like':
            args = apply_train_like_prediction_defaults(args)
        else:
            args = apply_practical_prediction_defaults(args, has_profile_obs=False)
        if args.ppo_policy_path:
            args.use_ppo = True
        if not args.model_checkpoint_path:
            raise ValueError('Predict mode requires --model-checkpoint-path so inference uses a trained PINN instead of retraining.')
    else:
        args = apply_train_mode_defaults(args, has_profile_obs=profile_obs_path is not None)
        if profile_obs_path is not None and args.profile_split_mode == 'none':
            print('Train mode disallows profile_split_mode=none to avoid data leakage; using time_blocked instead.')
            args.profile_split_mode = 'time_blocked'
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
    if args.mode == 'predict':
        print("Predict defaults profile:", args.predict_defaults_profile)
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
    print(f"Surface mixed-layer uniformity weight: {args.surface_mixed_layer_uniformity_weight:.2f}")
    print(
        f"Abrupt surface cooling weight: {args.abrupt_surface_cooling_weight:.2f} | "
        f"allowance={SURFACE_MIXED_LAYER_MAX_COOLING_C_PER_DAY:.2f} C/day"
    )
    print(f"Autumn overturn weight: {args.autumn_overturn_weight:.2f}")
    print(
        f"Heat budget weight: {args.heat_budget_weight:.2f} | "
        f"budget depth points={int(args.heat_budget_depth_points)}"
    )
    training_use_ppo = bool(args.use_ppo and args.mode == 'train')
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
        print(f"PPO tune Kalman in training: {bool(args.ppo_tune_kalman and training_use_ppo)}")
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
    if args.mode == 'predict' and predict_score_profile_obs_path is not None:
        score_profile_obs = load_optional_profile_observations(
            predict_score_profile_obs_path,
            start_date=metadata['start_date'],
            time_scale_seconds=metadata['time_scale_seconds'],
            max_depth=args.max_depth,
        )
        if has_profile_observations(score_profile_obs):
            test_profile_obs = score_profile_obs
            print(
                "Predict scoring profile observations loaded | "
                f"rows={len(score_profile_obs)} | "
                f"dates={score_profile_obs['Date'].nunique()} | "
                f"depths={score_profile_obs['Depth_m'].nunique()}"
            )

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
        'kalman_scales': normalize_kalman_scales({}),
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
        'kalman_scales': normalize_kalman_scales({}),
    }
    if args.mode == 'predict':
        model, model_checkpoint_bundle = load_model_checkpoint_bundle(args.model_checkpoint_path, device=args.device)
        checkpoint_info = dict((model_checkpoint_bundle or {}).get('training_info', {}) or {})
        training_info.update({
            'final_weights': dict(checkpoint_info.get('final_weights', {})),
            'kalman_scales': normalize_kalman_scales(checkpoint_info.get('kalman_scales', {})),
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
            metadata=metadata,
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
            use_ppo=training_use_ppo,
            ppo_control_interval=args.ppo_control_interval,
            ppo_rollout_steps=args.ppo_rollout_steps,
            ppo_max_updates_run=args.ppo_max_updates_run,
            ppo_eval_depth_points=args.ppo_eval_depth_points,
            ppo_use_kalman_reward=(args.ppo_use_kalman_reward and training_use_ppo),
            ppo_tune_kalman=(args.ppo_tune_kalman and training_use_ppo),
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
        surface_mixed_layer_uniformity_weight=args.surface_mixed_layer_uniformity_weight,
        abrupt_surface_cooling_weight=args.abrupt_surface_cooling_weight,
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
            use_ppo=training_use_ppo,
            ppo_control_interval=args.ppo_control_interval,
            ppo_rollout_steps=args.ppo_rollout_steps,
            ppo_max_updates_run=args.ppo_max_updates_run,
            ppo_eval_depth_points=args.ppo_eval_depth_points,
            ppo_use_kalman_reward=(args.ppo_use_kalman_reward and training_use_ppo),
            ppo_tune_kalman=(args.ppo_tune_kalman and training_use_ppo),
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
        surface_mixed_layer_uniformity_weight=args.surface_mixed_layer_uniformity_weight,
        abrupt_surface_cooling_weight=args.abrupt_surface_cooling_weight,
        autumn_overturn_weight=args.autumn_overturn_weight,
        heat_budget_weight=args.heat_budget_weight,
        heat_budget_depth_points=args.heat_budget_depth_points,
            train_until_best=args.train_until_best,
            train_min_epochs=args.train_min_epochs,
            train_patience_windows=args.train_patience_windows,
            resume_checkpoint_bundle=resume_model_checkpoint_bundle,
            model_input_dim=args.model_input_dim,
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
                training_info['best_selection_metric'] = None if forecast_ppo_info['best_validation_metrics'] is None else float(
                    forecast_ppo_info['best_validation_metrics'].get(
                        'objective',
                        forecast_ppo_info['best_validation_metrics'].get('rmse', np.nan),
                    )
                )
                training_info['best_selection_label'] = 'forecast_val_profile_objective'
                online_ppo_controller = forecast_ppo_info['ppo_controller']
                online_ppo_bundle = forecast_ppo_info['ppo_policy_bundle']
            else:
                print('Skipping pure-forecast PPO training because validation profile observations are unavailable.')

    if args.mode == 'train' and not args.train_export_artifacts:
        saved_model_checkpoint_path = None
        if model is not None:
            model_checkpoint_output = args.save_model_checkpoint
            if model_checkpoint_output is None:
                model_checkpoint_output = str(output_dir / f"{metadata['file_tag']}_pinn_model_checkpoint.pt")
            saved_model_checkpoint_path = save_model_checkpoint_bundle(model, training_info, model_checkpoint_output)

        saved_ppo_policy_path = None
        if training_use_ppo and args.save_ppo_policy:
            saved_ppo_policy_path = save_ppo_policy_bundle(training_info.get('ppo_policy_bundle'), args.save_ppo_policy)

        print(
            "\nTrain artifact export disabled: skipped post-training prediction CSV, "
            "heatmaps, Kalman outputs, and manifest."
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
            final_weights = dict(training_info.get('final_weights', {}) or {})
            weight_parts = [
                f"lambda_{key}={final_weights[key]:.3e}"
                for key in weight_order
                if key in final_weights
            ]
            if weight_parts:
                print("PPO tuned weights | " + " | ".join(weight_parts))
            kalman_scales_summary = normalize_kalman_scales(training_info.get('kalman_scales', {}))
            print(
                "PPO tuned Kalman scales | "
                f"process_scale={kalman_scales_summary['process']:.3f} | "
                f"obs_scale={kalman_scales_summary['obs']:.3f} | "
                f"correlation_length={kalman_scales_summary['correlation_length']:.3f} | "
                f"forecast_blend={kalman_scales_summary['forecast_blend']:.3f}"
            )
        if saved_ppo_policy_path is not None:
            print(f"Saved PPO policy bundle to: {saved_ppo_policy_path}")
        if saved_model_checkpoint_path is not None:
            print(f"Saved PINN model checkpoint to: {saved_model_checkpoint_path}")
        elif args.seasonal_segmented:
            print("Seasonal-segmented train mode did not export a single model checkpoint bundle.")
        return

    temp_grid, depths, online_ppo_runtime = predict_temperature_grid(
        model,
        df=df,
        metadata=metadata,
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
    kalman_grid = None
    kalman_csv_path = None
    kalman_year_path = None
    learned_kalman_process_scale = training_info['kalman_scales']['process']
    learned_kalman_obs_scale = training_info['kalman_scales']['obs']
    learned_kalman_correlation_length = training_info['kalman_scales'].get('correlation_length', args.kalman_correlation_length)
    learned_kalman_forecast_blend = training_info['kalman_scales'].get('forecast_blend', args.kalman_forecast_blend)
    daily_process_scale = None
    daily_obs_scale = None
    daily_correlation_length = None
    daily_forecast_blend = None
    if not online_ppo_runtime['diagnostics'].empty:
        daily_process_scale = online_ppo_runtime['diagnostics']['kalman_process_scale'].to_numpy(dtype=np.float64)
        daily_obs_scale = online_ppo_runtime['diagnostics']['kalman_obs_scale'].to_numpy(dtype=np.float64)
        if 'kalman_correlation_length' in online_ppo_runtime['diagnostics'].columns:
            daily_correlation_length = online_ppo_runtime['diagnostics']['kalman_correlation_length'].to_numpy(dtype=np.float64)
        if 'kalman_forecast_blend' in online_ppo_runtime['diagnostics'].columns:
            daily_forecast_blend = online_ppo_runtime['diagnostics']['kalman_forecast_blend'].to_numpy(dtype=np.float64)
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
                correlation_length=learned_kalman_correlation_length,
                forecast_blend=learned_kalman_forecast_blend,
                forecast_spinup_days=args.kalman_forecast_spinup_days,
                forecast_spinup_max_blend=args.kalman_forecast_spinup_max_blend,
                use_surface_bulk_correction=args.surface_bulk_correction,
                use_bottom_observation=args.use_bottom_observation,
                surface_obs_depth_m=args.surface_obs_depth_m,
                daily_process_scale=daily_process_scale,
                daily_obs_scale=daily_obs_scale,
                daily_correlation_length=daily_correlation_length,
                daily_forecast_blend=daily_forecast_blend,
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

    def _selection_rmse(metrics):
        if metrics is None:
            return float('inf')
        rmse_value = metrics.get('rmse', np.nan)
        if not np.isfinite(rmse_value):
            return float('inf')
        return float(rmse_value)

    pinn_rmse = _selection_rmse(validation_metrics)
    kalman_rmse = _selection_rmse(kalman_validation_metrics)
    selected_stage = 'pinn_rolling'
    selected_suffix = 'pinn'
    selected_grid = temp_grid
    selected_metrics = validation_metrics
    selected_note = 'Selected PINN/rolling forecast because Kalman assimilation is disabled or unavailable.'
    if kalman_grid is not None:
        selected_stage = 'kalman_assimilated'
        selected_suffix = 'kalman'
        selected_grid = kalman_grid
        selected_metrics = kalman_validation_metrics
        selected_note = 'Selected Kalman-assimilated forecast by default. Profile observations are used only for reporting, not output selection.'

    selected_csv_path = export_temperature_tables(df, selected_grid, depths, output_dir, metadata, suffix=selected_suffix)
    selected_year_path = output_dir / f"{metadata['file_tag']}_{selected_suffix}_year_heatmap.png"
    plot_year_heatmap(df, selected_grid, depths, selected_year_path, metadata)
    if selected_stage == 'kalman_assimilated':
        kalman_csv_path = selected_csv_path
        kalman_year_path = selected_year_path

    output_manifest_rows = [
        {
            'stage': selected_stage,
            'recommended_for_scoring': True,
            'recommended_for_display': True,
            'csv_path': str(selected_csv_path),
            'year_heatmap_path': str(selected_year_path),
            'pinn_rmse': None if not np.isfinite(pinn_rmse) else pinn_rmse,
            'kalman_rmse': None if not np.isfinite(kalman_rmse) else kalman_rmse,
            'selected_rmse': None if selected_metrics is None else selected_metrics.get('rmse'),
            'note': selected_note,
        }
    ]
    output_manifest_path = output_dir / f"{metadata['file_tag']}_prediction_outputs_manifest.csv"
    pd.DataFrame(output_manifest_rows).to_csv(output_manifest_path, index=False)

    scorecard_report_path = None
    scorecard_status = None
    if args.mode == 'predict' and predict_score_profile_obs_path is not None:
        scorecard_report_path, scorecard_status = run_scorecard_report(
            truth_csv_path=predict_score_profile_obs_path,
            prediction_csv_path=selected_csv_path,
            output_dir=output_dir,
            label=selected_stage,
            report_name=f"{metadata['file_tag']}_scorecard_report.png",
        )

    print(f"\nSelected prediction stage: {selected_stage}")
    print(f"Saved selected annual heatmap to: {selected_year_path}")
    print(f"Saved selected prediction table to: {selected_csv_path}")
    print(f"Saved prediction output manifest to: {output_manifest_path}")
    if scorecard_report_path is not None:
        print(f"Saved scorecard report to: {scorecard_report_path}")
    elif scorecard_status is not None:
        print(f"Scorecard report skipped/failed: {scorecard_status}")
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
            f"obs_scale={learned_kalman_obs_scale:.3f} | "
            f"correlation_length={learned_kalman_correlation_length:.3f} | "
            f"forecast_blend={learned_kalman_forecast_blend:.3f}"
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
                f"surface_decay_depth={first_diag['surface_decay_depth']:.3f} | "
                f"deep_inertia={first_diag['deep_inertia']:.3f} | "
                f"deep_anchor={first_diag['deep_anchor']:.3f} | "
                f"surface_skin_cooling_coef={first_diag['surface_skin_cooling_coef']:.3f} | "
                f"kalman_corr_len={first_diag['kalman_correlation_length']:.3f} | "
                f"kalman_forecast_blend={first_diag['kalman_forecast_blend']:.3f}"
            )
        if not online_ppo_runtime['history'].empty:
            last_online = online_ppo_runtime['history'].iloc[-1]
            print(
                "Online PPO controls | "
                f"memory_blend={last_online['memory_blend']:.3f} | "
                f"surface_relaxation={last_online['surface_relaxation']:.3f} | "
                f"surface_decay_depth={last_online['surface_decay_depth']:.3f} | "
                f"deep_inertia={last_online['deep_inertia']:.3f} | "
                f"deep_anchor={last_online['deep_anchor']:.3f} | "
                f"surface_skin_cooling_coef={last_online['surface_skin_cooling_coef']:.3f} | "
                f"kalman_corr_len={last_online['kalman_correlation_length']:.3f} | "
                f"kalman_forecast_blend={last_online['kalman_forecast_blend']:.3f}"
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
