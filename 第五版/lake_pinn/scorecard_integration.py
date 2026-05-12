# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *


def default_scorecard_script_path() -> Path:
    """Locate the shared scorecard script used by the experiment notebooks."""
    candidates = [
        Path(__file__).resolve().with_name('lake_profile_scorecard.py'),
        PROJECT_DIR.parent / '归档' / '第三版' / 'lake_profile_scorecard.py',
        PROJECT_DIR / 'lake_profile_scorecard.py',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_scorecard_report(
    truth_csv_path,
    prediction_csv_path,
    output_dir,
    label='selected_prediction',
    report_name=None,
    scorecard_script_path=None,
):
    """Generate the layered scorecard PNG in the prediction output directory."""
    truth_csv_path = Path(truth_csv_path) if truth_csv_path is not None else None
    prediction_csv_path = Path(prediction_csv_path) if prediction_csv_path is not None else None
    output_dir = Path(output_dir)
    scorecard_script_path = Path(scorecard_script_path) if scorecard_script_path else default_scorecard_script_path()

    if truth_csv_path is None or not truth_csv_path.exists():
        return None, 'truth_missing'
    if prediction_csv_path is None or not prediction_csv_path.exists():
        return None, 'prediction_missing'
    if not scorecard_script_path.exists():
        return None, 'scorecard_script_missing'

    if report_name is None:
        report_name = f"{prediction_csv_path.stem}_scorecard_report.png"

    command = [
        sys.executable,
        str(scorecard_script_path),
        '--truth',
        str(truth_csv_path),
        '--pred',
        str(prediction_csv_path),
        '--label',
        str(label),
        '--out-dir',
        str(output_dir),
        '--report-name',
        str(report_name),
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f'exit_code_{result.returncode}'
        return None, message

    return output_dir / report_name, result.stdout.strip()


def score_prediction_candidates(
    truth_csv_path,
    candidates,
    output_dir,
    report_name='scorecard_report.png',
):
    """Score multiple prediction CSVs in-process and return ranked rows.

    This is used during predict-time stage selection. It avoids writing the
    detailed scorecard CSV tables, so the prediction output directory only gets
    the compact PNG report plus the final selected prediction artifacts.
    """
    truth_csv_path = Path(truth_csv_path) if truth_csv_path is not None else None
    output_dir = Path(output_dir)
    if truth_csv_path is None or not truth_csv_path.exists():
        return None, 'truth_missing', []
    if not candidates:
        return None, 'candidate_missing', []

    try:
        from . import lake_profile_scorecard as scorecard

        truth = scorecard.load_profile_csv(truth_csv_path)
        thresholds = scorecard.ScoreThresholds()
        score_rows = []
        veto_rows = []
        diagnostics_rows = []

        for candidate in candidates:
            label = str(candidate['label'])
            pred_path = Path(candidate['prediction_csv_path'])
            if not pred_path.exists():
                return None, f'prediction_missing:{pred_path}', []

            pred = scorecard.load_profile_csv(pred_path)
            aligned, depths, dates, eff_depth = scorecard.build_aligned_cube(truth, pred)
            daily = scorecard.compute_daily_features(
                aligned=aligned,
                depths=depths,
                dates=dates,
                eff_depth=eff_depth,
                surface_band_max_m=scorecard.SURFACE_BAND_MAX_M,
                thermocline_delta_min_c=scorecard.THERMO_DELTA_T_MIN_C,
                mld_threshold_c=1.0,
                mix_delta_col_max_c=1.0,
                mix_stdz_max_c=0.8,
                mix_consecutive_days=7,
            )
            tv_metrics = scorecard.compute_tv_metrics(aligned, depths, dates)
            masks = scorecard.make_masks(aligned, daily, eff_depth, scorecard.SURFACE_BAND_MAX_M)
            thermo_mask = scorecard.thermocline_band_mask(aligned, daily)

            score_args = argparse.Namespace(
                winter_inverse_min_c=0.8,
                winter_inverse_frac_min=0.60,
                summer_strat_min_c=4.0,
                summer_strat_frac_min=0.60,
                summer_delta_col_threshold_c=3.0,
                tmix_abs_max_days=15.0,
                monthly_bias_max_c=1.5,
                annual_bias_max_c=0.5,
                thermocline_depth_max_m=2.5,
                thermocline_thickness_max_m=2.0,
                deep_julsep_rmse_max_c=1.5,
                min_physical_temp_c=-0.5,
                max_physical_temp_c=32.0,
                max_surface_jump_c_per_day=4.0,
                max_surface_band_jump_c_per_day=2.5,
                max_column_jump_c_per_day=4.0,
                max_grad_p995_c_per_m=6.0,
                max_grad_extreme_c_per_m=8.0,
                max_april_surface_error_c=3.0,
                max_april_surface_jump_c_per_day=1.5,
                density_inversion_drop_kgm3=0.02,
                surface_band_max_m=scorecard.SURFACE_BAND_MAX_M,
                max_density_unstable_layer_frac=0.30,
                max_density_unstable_days=12,
                seed_score_std=np.nan,
                reload_mae=np.nan,
            )
            vetoes = scorecard.evaluate_vetoes(
                aligned, daily, depths, dates, masks, thermo_mask, score_args
            )
            scores, extra = scorecard.score_run(
                aligned,
                daily,
                tv_metrics,
                eff_depth,
                masks,
                thermo_mask,
                score_args,
                thresholds,
                visual_score=np.nan,
                visual_note='',
            )

            score_rows.append({
                'run': label,
                'stage': candidate.get('stage', label),
                'prediction_csv': str(pred_path),
                'effective_depth_m': eff_depth,
                **scores,
            })
            veto_rows.append({'run': label, 'prediction_csv': str(pred_path), **vetoes})
            diagnostics_rows.append({
                'run': label,
                'prediction_csv': str(pred_path),
                'effective_depth_m': eff_depth,
                **tv_metrics,
                **extra,
            })

        score_df = pd.DataFrame(score_rows)
        veto_df = pd.DataFrame(veto_rows)
        diag_df = pd.DataFrame(diagnostics_rows)
        merged = (
            score_df
            .merge(veto_df, on=['run', 'prediction_csv'], how='left')
            .merge(diag_df, on=['run', 'prediction_csv', 'effective_depth_m'], how='left')
        )
        merged['layer1_physics_pass'] = merged['pass_all_vetoes'].astype(bool)
        merged['scorecard_v2_failed_check_count_for_sort'] = -merged.get(
            'scorecard_v2_failed_check_count',
            pd.Series(999, index=merged.index),
        ).fillna(999).astype(float)
        merged['layer4_stability_score_for_sort'] = merged['layer4_stability_score'].fillna(-1.0)
        merged['layer5_visual_score_for_sort'] = merged['layer5_visual_score'].fillna(-1.0)
        merged = merged.sort_values(
            by=[
                'layer1_physics_pass',
                'scorecard_v2_failed_check_count_for_sort',
                'layer2_key_seasonal_score',
                'layer3_numeric_score',
                'layer4_stability_score_for_sort',
                'layer5_visual_score_for_sort',
            ],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)
        merged['selection_rank'] = np.arange(1, len(merged) + 1, dtype=int)

        report_path = output_dir / report_name
        scorecard.write_scorecard_report(merged, report_path)
        return report_path, 'ok', merged.to_dict('records')
    except Exception as exc:
        return None, f'scorecard_failed:{exc}', []
