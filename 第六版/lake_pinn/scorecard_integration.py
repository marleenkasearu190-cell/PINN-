# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

MONTH_LABELS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
    lake_type='universal',
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
        try:
            discrete_truth = _load_truth_profile_long(truth_csv_path)
        except Exception:
            discrete_truth = None
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
            resolved_lake_type = scorecard.resolve_lake_type(lake_type, truth=truth, eff_depth=eff_depth)
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
                lake_type=resolved_lake_type,
                warm_deep_winter_rmse_max_c=3.0,
                warm_deep_winter_bias_max_c=2.0,
                warm_deep_min_temp_c=6.0,
                warm_deep_thermocline_depth_max_m=5.0,
                warm_deep_thermocline_thickness_max_m=5.0,
                warm_deep_deep_julsep_rmse_max_c=2.5,
                warm_deep_autumn_final_extra_c=4.0,
                warm_deep_autumn_gap_reduction_frac=0.25,
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
            discrete_metrics = {}
            if discrete_truth is not None:
                try:
                    discrete_pred = _load_prediction_profile_long(pred_path)
                    matched_points = _interpolate_prediction_to_observation_points(discrete_truth, discrete_pred)
                    discrete_metrics = _discrete_scorecard_metrics(matched_points, thresholds)
                except Exception as discrete_exc:
                    discrete_metrics = {'discrete_status': f'failed:{discrete_exc}'}
            if discrete_metrics:
                scores.update(discrete_metrics)
                discrete_vetoes = _discrete_point_vetoes(discrete_metrics, thresholds)
                vetoes.update(discrete_vetoes)
                if not bool(discrete_vetoes.get('discrete_point_pass', True)):
                    vetoes['pass_all_vetoes'] = False
                    _append_failed_check(vetoes, 'discrete_point_error')
                discrete_score = scores.get('layer3_discrete_numeric_score', np.nan)
                if np.isfinite(discrete_score):
                    grid_score = float(scores.get('layer3_numeric_score', np.nan))
                    if np.isfinite(grid_score):
                        blended_score = 0.45 * grid_score + 0.55 * float(discrete_score)
                        scores['layer3_grid_numeric_score'] = grid_score
                        scores['layer3_numeric_score'] = blended_score
                        score_delta = blended_score - grid_score
                        scores['layered_selection_score_raw'] = float(scores['layered_selection_score_raw']) + score_delta
                        scores['layered_selection_score_100'] = float(scores['layered_selection_score_raw']) / 75.0 * 100.0

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
        if 'lake_type_x' in merged.columns or 'lake_type_y' in merged.columns:
            merged['lake_type'] = merged.get('lake_type_x', pd.Series(index=merged.index, dtype=object)).combine_first(
                merged.get('lake_type_y', pd.Series(index=merged.index, dtype=object))
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


def _load_prediction_profile_long(prediction_csv_path):
    pred = pd.read_csv(prediction_csv_path)
    pred['Date'] = pd.to_datetime(pred['Date'])
    temp_col = None
    for column in ['Temperature_C', 'Predicted_Temperature_C', 'temperature_c', 'Temp_C']:
        if column in pred.columns:
            temp_col = column
            break
    if temp_col is None:
        raise ValueError(f'No prediction temperature column found: {prediction_csv_path}')
    out = pred[['Date', 'Depth_m', temp_col]].rename(columns={temp_col: 'pred_grid'})
    out['Depth_m'] = pd.to_numeric(out['Depth_m'], errors='coerce')
    out['pred_grid'] = pd.to_numeric(out['pred_grid'], errors='coerce')
    return out.dropna(subset=['Date', 'Depth_m', 'pred_grid']).sort_values(['Date', 'Depth_m']).reset_index(drop=True)


def _load_truth_profile_long(truth_csv_path):
    truth = pd.read_csv(truth_csv_path)
    truth['Date'] = pd.to_datetime(truth['Date'])

    long_temp_col = None
    for column in ['Temperature_C', 'temperature_C', 'temperature_c', 'Temp_C', 'temp_c', 'Temperature', 'temp']:
        if column in truth.columns:
            long_temp_col = column
            break

    if long_temp_col is not None and ('Depth_m' in truth.columns or 'depth_m' in truth.columns):
        depth_col = 'Depth_m' if 'Depth_m' in truth.columns else 'depth_m'
        out = truth[['Date', depth_col, long_temp_col]].rename(columns={depth_col: 'Depth_m', long_temp_col: 'obs'})
    else:
        value_cols = []
        depths = {}
        for column in truth.columns:
            match = re.fullmatch(r'Temp_([0-9]+(?:\.[0-9]+)?)m', str(column))
            if match:
                value_cols.append(column)
                depths[column] = float(match.group(1))
        if not value_cols:
            raise ValueError(f'No profile temperature columns found: {truth_csv_path}')
        out = truth.melt(id_vars=['Date'], value_vars=value_cols, var_name='depth_col', value_name='obs')
        out['Depth_m'] = out['depth_col'].map(depths).astype(float)
        out = out.drop(columns=['depth_col'])

    out['Depth_m'] = pd.to_numeric(out['Depth_m'], errors='coerce')
    out['obs'] = pd.to_numeric(out['obs'], errors='coerce')
    return out.dropna(subset=['Date', 'Depth_m', 'obs']).sort_values(['Date', 'Depth_m']).reset_index(drop=True)


def _interpolate_prediction_to_observation_points(truth, pred):
    pred_by_date = {date_value: group.sort_values('Depth_m') for date_value, group in pred.groupby('Date')}
    rows = []
    for date_value, obs_day in truth.groupby('Date'):
        pred_day = pred_by_date.get(date_value)
        if pred_day is None or pred_day.empty:
            continue
        z_pred = pred_day['Depth_m'].to_numpy(dtype=np.float64)
        t_pred = pred_day['pred_grid'].to_numpy(dtype=np.float64)
        in_range = (obs_day['Depth_m'] >= np.nanmin(z_pred)) & (obs_day['Depth_m'] <= np.nanmax(z_pred))
        if not in_range.any():
            continue
        obs_matched = obs_day.loc[in_range].copy()
        obs_matched['pred'] = np.interp(obs_matched['Depth_m'].to_numpy(dtype=np.float64), z_pred, t_pred)
        rows.append(obs_matched)

    if not rows:
        return pd.DataFrame(columns=['Date', 'Depth_m', 'obs', 'pred', 'err', 'abs_err', 'month', 'season'])

    matched = pd.concat(rows, ignore_index=True)
    matched['err'] = matched['pred'] - matched['obs']
    matched['abs_err'] = matched['err'].abs()
    matched['month'] = matched['Date'].dt.month
    matched['season'] = np.select(
        [
            matched['month'].isin([12, 1, 2]),
            matched['month'].isin([3, 4, 5]),
            matched['month'].isin([6, 7, 8]),
            matched['month'].isin([9, 10, 11]),
        ],
        ['Winter', 'Spring', 'Summer', 'Autumn'],
        default='Other',
    )
    return matched


def _metric_series(frame):
    if frame.empty:
        return {
            'N': 0,
            'RMSE': np.nan,
            'MAE': np.nan,
            'Bias': np.nan,
            'P95Abs': np.nan,
            'MaxAbs': np.nan,
        }
    err = frame['err'].to_numpy(dtype=np.float64)
    abs_err = np.abs(err)
    return {
        'N': int(len(frame)),
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'MAE': float(np.mean(abs_err)),
        'Bias': float(np.mean(err)),
        'P95Abs': float(np.percentile(abs_err, 95)),
        'MaxAbs': float(np.max(abs_err)),
    }


def _grouped_metrics(frame, column):
    rows = []
    for key, group in frame.groupby(column, observed=False):
        rows.append({column: key, **_metric_series(group)})
    return pd.DataFrame(rows)


def _score_down(value, good, bad, weight):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(weight) * float(np.clip((float(bad) - value) / max(float(bad) - float(good), 1e-12), 0.0, 1.0))


def _append_failed_check(vetoes, label):
    existing = str(vetoes.get('scorecard_v2_failed_checks', '') or '').strip()
    if not existing or existing == 'none':
        labels = []
    else:
        labels = [item.strip() for item in existing.split(',') if item.strip()]
    if label not in labels:
        labels.append(label)
    vetoes['scorecard_v2_failed_checks'] = ', '.join(labels) if labels else 'none'
    vetoes['scorecard_v2_failed_check_count'] = int(len(labels))


def _discrete_point_vetoes(metrics, thresholds):
    """Universal pointwise gate based on real observation date/depth matches."""
    status = str(metrics.get('discrete_status', 'missing'))
    if status != 'ok':
        return {
            'discrete_point_pass': False,
            'discrete_point_failed_checks': f'discrete_{status}',
            'discrete_rmse_gate_c': np.nan,
            'discrete_abs_bias_gate_c': np.nan,
            'discrete_p95_abs_gate_c': np.nan,
            'discrete_balanced_rmse_gate_c': np.nan,
        }

    rmse_gate = max(float(thresholds.overall_rmse_bad), 2.5)
    bias_gate = max(float(thresholds.abs_bias_bad) * 2.0, 1.0)
    p95_gate = 4.0
    balanced_gate = max(float(thresholds.thermocline_rmse_bad), 3.0)
    frac_gt2_gate = 0.30
    frac_gt3_gate = 0.15
    frac_gt4_gate = 0.06
    excess_over_2c_gate = 0.60
    failed = []

    rmse = float(metrics.get('discrete_rmse', np.nan))
    bias = float(metrics.get('discrete_bias', np.nan))
    p95_abs = float(metrics.get('discrete_p95_abs', np.nan))
    balanced_rmse = float(metrics.get('discrete_balanced_rmse', np.nan))
    frac_gt2 = float(metrics.get('discrete_abs_gt_2c_frac', np.nan))
    frac_gt3 = float(metrics.get('discrete_abs_gt_3c_frac', np.nan))
    frac_gt4 = float(metrics.get('discrete_abs_gt_4c_frac', np.nan))
    excess_over_2c = float(metrics.get('discrete_mean_excess_over_2c', np.nan))

    if not np.isfinite(rmse) or rmse > rmse_gate:
        failed.append('discrete_rmse')
    if not np.isfinite(bias) or abs(bias) > bias_gate:
        failed.append('discrete_bias')
    if not np.isfinite(p95_abs) or p95_abs > p95_gate:
        failed.append('discrete_p95_abs')
    if not np.isfinite(balanced_rmse) or balanced_rmse > balanced_gate:
        failed.append('discrete_balanced_rmse')
    if not np.isfinite(frac_gt2) or frac_gt2 > frac_gt2_gate:
        failed.append('discrete_gt2_count')
    if not np.isfinite(frac_gt3) or frac_gt3 > frac_gt3_gate:
        failed.append('discrete_gt3_count')
    if not np.isfinite(frac_gt4) or frac_gt4 > frac_gt4_gate:
        failed.append('discrete_gt4_count')
    if not np.isfinite(excess_over_2c) or excess_over_2c > excess_over_2c_gate:
        failed.append('discrete_error_magnitude')

    return {
        'discrete_point_pass': len(failed) == 0,
        'discrete_point_failed_checks': ', '.join(failed) if failed else 'none',
        'discrete_rmse_gate_c': rmse_gate,
        'discrete_abs_bias_gate_c': bias_gate,
        'discrete_p95_abs_gate_c': p95_gate,
        'discrete_balanced_rmse_gate_c': balanced_gate,
        'discrete_abs_gt_2c_frac_gate': frac_gt2_gate,
        'discrete_abs_gt_3c_frac_gate': frac_gt3_gate,
        'discrete_abs_gt_4c_frac_gate': frac_gt4_gate,
        'discrete_mean_excess_over_2c_gate': excess_over_2c_gate,
    }


def _finite_mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def _count_fraction_over(abs_err, threshold):
    mask = abs_err > float(threshold)
    count = int(np.sum(mask))
    frac = float(count / max(len(abs_err), 1))
    return count, frac


def _discrete_scorecard_metrics(matched, thresholds):
    """Score prediction accuracy at real observation dates/depths.

    These metrics are used only for predict-time evaluation and model
    selection. They are not fed back into training, so held-out truth does not
    leak into model parameters.
    """
    if matched is None or matched.empty:
        return {
            'discrete_status': 'no_matched_points',
            'discrete_n': 0,
            'layer3_discrete_numeric_score': np.nan,
        }

    matched = matched.copy()
    abs_err = matched['abs_err'].to_numpy(dtype=np.float64)
    count_gt_1c, frac_gt_1c = _count_fraction_over(abs_err, 1.0)
    count_gt_2c, frac_gt_2c = _count_fraction_over(abs_err, 2.0)
    count_gt_3c, frac_gt_3c = _count_fraction_over(abs_err, 3.0)
    count_gt_4c, frac_gt_4c = _count_fraction_over(abs_err, 4.0)
    excess_over_2c = np.clip(abs_err - 2.0, 0.0, None)
    excess_over_3c = np.clip(abs_err - 3.0, 0.0, None)
    mean_excess_over_2c = float(np.mean(excess_over_2c)) if len(excess_over_2c) else np.nan
    rms_excess_over_3c = float(np.sqrt(np.mean(excess_over_3c ** 2))) if len(excess_over_3c) else np.nan
    max_depth = float(matched['Depth_m'].max())
    deep_cut = 0.7 * max_depth
    overall = _metric_series(matched)
    surface = _metric_series(matched[matched['Depth_m'] <= 3.0])
    mid = _metric_series(matched[(matched['Depth_m'] > 3.0) & (matched['Depth_m'] < deep_cut)])
    deep = _metric_series(matched[matched['Depth_m'] >= deep_cut])

    seasonal = _grouped_metrics(matched, 'season')
    seasonal_rmse = _finite_mean(seasonal['RMSE'].to_numpy(dtype=np.float64)) if not seasonal.empty else np.nan
    depth_rmse = _finite_mean([surface['RMSE'], mid['RMSE'], deep['RMSE']])
    balanced_rmse = _finite_mean([seasonal_rmse, depth_rmse])

    discrete_scores = {
        'score_discrete_overall_rmse': _score_down(
            overall['RMSE'], thresholds.overall_rmse_good, thresholds.overall_rmse_bad, 10.0
        ),
        'score_discrete_overall_mae': _score_down(
            overall['MAE'], thresholds.overall_mae_good, thresholds.overall_mae_bad, 8.0
        ),
        'score_discrete_abs_bias': _score_down(
            abs(overall['Bias']), thresholds.abs_bias_good, thresholds.abs_bias_bad, 4.0
        ),
        'score_discrete_surface_rmse': _score_down(
            surface['RMSE'], thresholds.surface_rmse_good, thresholds.surface_rmse_bad, 4.0
        ),
        'score_discrete_balanced_rmse': _score_down(
            balanced_rmse, thresholds.thermocline_rmse_good, thresholds.thermocline_rmse_bad, 4.0
        ),
        'score_discrete_large_error_count': _score_down(
            frac_gt_2c, 0.05, 0.25, 5.0
        ),
        'score_discrete_error_magnitude': _score_down(
            mean_excess_over_2c, 0.05, 0.60, 5.0
        ),
    }

    return {
        'discrete_status': 'ok',
        'discrete_n': overall['N'],
        'discrete_rmse': overall['RMSE'],
        'discrete_mae': overall['MAE'],
        'discrete_bias': overall['Bias'],
        'discrete_p95_abs': overall['P95Abs'],
        'discrete_abs_gt_1c_count': count_gt_1c,
        'discrete_abs_gt_1c_frac': frac_gt_1c,
        'discrete_abs_gt_2c_count': count_gt_2c,
        'discrete_abs_gt_2c_frac': frac_gt_2c,
        'discrete_abs_gt_3c_count': count_gt_3c,
        'discrete_abs_gt_3c_frac': frac_gt_3c,
        'discrete_abs_gt_4c_count': count_gt_4c,
        'discrete_abs_gt_4c_frac': frac_gt_4c,
        'discrete_mean_excess_over_2c': mean_excess_over_2c,
        'discrete_rms_excess_over_3c': rms_excess_over_3c,
        'discrete_surface_rmse': surface['RMSE'],
        'discrete_mid_rmse': mid['RMSE'],
        'discrete_deep_rmse': deep['RMSE'],
        'discrete_season_balanced_rmse': seasonal_rmse,
        'discrete_depth_balanced_rmse': depth_rmse,
        'discrete_balanced_rmse': balanced_rmse,
        'layer3_discrete_numeric_score': float(sum(discrete_scores.values())),
        **discrete_scores,
    }


def _ensure_chinese_font():
    for font_path in [
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/NotoSansSC-VF.ttf'),
        Path('C:/Windows/Fonts/simhei.ttf'),
    ]:
        if font_path.exists():
            try:
                from matplotlib import font_manager
                font_manager.fontManager.addfont(str(font_path))
                if font_path.name.lower().startswith('msyh'):
                    plt.rcParams['font.family'] = 'Microsoft YaHei'
                elif font_path.name.lower().startswith('simhei'):
                    plt.rcParams['font.family'] = 'SimHei'
                else:
                    plt.rcParams['font.family'] = 'Noto Sans SC'
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception:
                continue


def _plot_discrete_point_evaluation(matched, lake_name, model_label, output_path):
    _ensure_chinese_font()
    max_depth = float(matched['Depth_m'].max()) if not matched.empty else 1.0
    matched = matched.copy()
    matched['depth_band'] = pd.cut(
        matched['Depth_m'],
        [-np.inf, 3.0, 0.7 * max_depth, np.inf],
        labels=['Surface 0-3m', f'Mid 3-{0.7 * max_depth:.1f}m', f'Deep {0.7 * max_depth:.1f}-{max_depth:.1f}m'],
    )
    overall = _metric_series(matched)
    seasonal = _grouped_metrics(matched, 'season').set_index('season')
    bands = _grouped_metrics(matched, 'depth_band').set_index('depth_band')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=180)
    fig.suptitle('Discrete Observation-Point Evaluation | 离散观测点评估', fontsize=18, fontweight='bold')

    ax = axes[0]
    ax.axis('off')
    table_rows = [
        ['Lake 湖泊', lake_name],
        ['Model 模型', model_label],
        ['Matched 匹配点数', f"{len(matched):,}"],
        ['Dates/Depths', f"{matched['Date'].nunique()} / {matched['Depth_m'].nunique()}"],
        ['RMSE', f"{overall['RMSE']:.3f} C"],
        ['MAE', f"{overall['MAE']:.3f} C"],
        ['Bias', f"{overall['Bias']:+.3f} C"],
        ['P95 |err|', f"{overall['P95Abs']:.3f} C"],
        ['Max |err|', f"{overall['MaxAbs']:.3f} C"],
    ]
    table = ax.table(cellText=table_rows, colLabels=['Metric 指标', 'Value 数值'], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    ax = axes[1]
    sample = matched.sample(n=min(len(matched), 7000), random_state=42) if len(matched) > 7000 else matched
    ax.scatter(sample['obs'], sample['pred'], s=5, alpha=0.25, color='#2563eb')
    lo = float(min(matched['obs'].min(), matched['pred'].min()))
    hi = float(max(matched['obs'].max(), matched['pred'].max()))
    ax.plot([lo - 0.5, hi + 0.5], [lo - 0.5, hi + 0.5], '--', color='black', lw=1)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlabel('Observed 观测 (C)')
    ax.set_ylabel('Predicted 预测 (C)')
    ax.set_title('Predicted vs Observed')
    ax.grid(alpha=0.25)

    ax = axes[2]
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_rmse = seasonal.reindex(season_order)['RMSE']
    band_rmse = bands['RMSE']
    x1 = np.arange(len(season_rmse))
    x2 = np.arange(len(band_rmse)) + len(season_rmse) + 1
    ax.bar(x1, season_rmse.values, color='#f59e0b', label='Season 季节')
    ax.bar(x2, band_rmse.values, color='#10b981', label='Depth 深度带')
    ax.set_xticks(list(x1) + list(x2))
    ax.set_xticklabels(season_order + [str(value) for value in band_rmse.index], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (C)')
    ax.set_title('RMSE by Season / Depth')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _month_depth_bias_grid(matched, max_depth, depth_step):
    plot_depths = np.arange(0.0, max_depth + 1e-6, depth_step, dtype=np.float64)
    grid = np.full((len(plot_depths), 12), np.nan, dtype=np.float64)
    count_grid = np.zeros((len(plot_depths), 12), dtype=np.int64)
    if matched.empty:
        return plot_depths, grid, count_grid

    # Strict point-to-point comparison: prediction is first interpolated only to
    # the actual observation depth/time, then binned for display. We do not
    # interpolate bias vertically to fill empty heatmap cells.
    point_bias = matched[['month', 'Depth_m', 'err']].dropna().copy()
    if point_bias.empty:
        return plot_depths, grid, count_grid
    depth_index = np.rint(point_bias['Depth_m'].to_numpy(dtype=np.float64) / depth_step).astype(int)
    depth_index = np.clip(depth_index, 0, len(plot_depths) - 1)
    point_bias['depth_index'] = depth_index
    means = point_bias.groupby(['depth_index', 'month'])['err'].agg(['mean', 'count']).reset_index()
    for _, row in means.iterrows():
        depth_idx = int(row['depth_index'])
        month_idx = int(row['month']) - 1
        if 0 <= depth_idx < grid.shape[0] and 0 <= month_idx < grid.shape[1]:
            grid[depth_idx, month_idx] = float(row['mean'])
            count_grid[depth_idx, month_idx] = int(row['count'])
    return plot_depths, grid, count_grid


def _annotate_bias_grid(ax, grid, vlim, count_grid=None):
    for depth_idx in range(grid.shape[0]):
        for month_idx in range(grid.shape[1]):
            value = grid[depth_idx, month_idx]
            if not np.isfinite(value):
                ax.text(month_idx, depth_idx, 'NA', ha='center', va='center', fontsize=7, fontweight='bold', color='#6b7280')
                continue
            label = f'{value:.2f}'
            if count_grid is not None and count_grid[depth_idx, month_idx] > 0:
                label = f'{label}\nn={int(count_grid[depth_idx, month_idx])}'
            color = 'white' if abs(value) > vlim * 0.55 else '#111111'
            ax.text(month_idx, depth_idx, label, ha='center', va='center', fontsize=8, fontweight='bold', color=color)


def _plot_monthly_depth_bias(matched, lake_name, model_label, output_path):
    _ensure_chinese_font()
    max_depth = float(matched['Depth_m'].max()) if not matched.empty else 1.0
    depth_step = 1.0 if max_depth <= 20.0 else 2.0
    depths, grid, count_grid = _month_depth_bias_grid(matched, max_depth=max_depth, depth_step=depth_step)
    finite = grid[np.isfinite(grid)]
    vlim = max(2.0, min(4.0, float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 3.0))

    fig = plt.figure(figsize=(16, 10), dpi=180)
    ax = fig.add_axes([0.08, 0.12, 0.78, 0.74])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='#d9d9d9')
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-vlim, vmax=vlim, aspect='auto')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTH_LABELS_SHORT, fontsize=13, fontweight='bold')
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([f'{int(depth) if float(depth).is_integer() else depth:g} m' for depth in depths], fontsize=12, fontweight='bold')
    ax.set_xlabel('Month of the Year', fontsize=16, fontweight='bold', labelpad=14)
    ax.set_ylabel('Depth (Meters)', fontsize=16, fontweight='bold', labelpad=16)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    _annotate_bias_grid(ax, grid, vlim, count_grid=count_grid)
    fig.text(0.03, 0.94, 'Point-to-Point Temperature Bias | 逐点温度偏差', fontsize=26, fontweight='bold', ha='left')
    fig.text(
        0.03,
        0.90,
        f'{lake_name} | {model_label} vs. Observed profiles | cell = mean(Pred - Obs), gray = no matched observation',
        fontsize=15,
        color='#555555',
        ha='left',
    )
    cax = fig.add_axes([0.90, 0.20, 0.025, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Temperature Bias (°C)\nPrediction - Observation\nRed = too warm, Blue = too cold', fontsize=13, fontweight='bold', labelpad=16)
    cbar.ax.tick_params(labelsize=11)
    ax.text(
        0.0,
        -0.10,
        'NA/灰色: no observation points in this month-depth bin; colors are based only on matched real observation points.',
        transform=ax.transAxes,
        fontsize=10,
        color='#555555',
        ha='left',
        va='top',
    )
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _pointwise_bias_profile_grid(matched, max_depth, depth_step):
    matched = matched.copy()
    matched['Date'] = pd.to_datetime(matched['Date'])
    dates = pd.date_range(matched['Date'].min(), matched['Date'].max(), freq='D')
    plot_depths = np.arange(0.0, max_depth + 1e-6, depth_step, dtype=np.float64)
    grid = np.full((len(plot_depths), len(dates)), np.nan, dtype=np.float64)
    if matched.empty:
        return dates, plot_depths, grid

    date_to_col = {date: idx for idx, date in enumerate(dates)}
    for date_value, day in matched.groupby('Date'):
        col_idx = date_to_col.get(pd.Timestamp(date_value).normalize())
        if col_idx is None:
            continue
        day = day.sort_values('Depth_m')
        z = day['Depth_m'].to_numpy(dtype=np.float64)
        err = day['err'].to_numpy(dtype=np.float64)
        finite = np.isfinite(z) & np.isfinite(err)
        z = z[finite]
        err = err[finite]
        if z.size == 0:
            continue
        if z.size == 1:
            depth_idx = int(np.clip(round(z[0] / depth_step), 0, len(plot_depths) - 1))
            grid[depth_idx, col_idx] = float(err[0])
            continue
        valid_depth_mask = (plot_depths >= np.nanmin(z)) & (plot_depths <= np.nanmax(z))
        if not np.any(valid_depth_mask):
            continue
        grid[valid_depth_mask, col_idx] = np.interp(plot_depths[valid_depth_mask], z, err)
    return dates, plot_depths, grid


def _gaussian_kernel1d(sigma):
    sigma = float(max(sigma, 1.0e-6))
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _smooth_masked_grid_for_display(grid, sigma_depth=0.20, sigma_time=0.45):
    """Very light display smoothing without filling originally missing cells."""
    valid = np.isfinite(grid)
    if not np.any(valid):
        return grid

    values = np.where(valid, grid, 0.0).astype(np.float64)
    weights = valid.astype(np.float64)

    for axis, sigma in ((0, sigma_depth), (1, sigma_time)):
        kernel = _gaussian_kernel1d(sigma)
        values = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis, values)
        weights = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis, weights)

    smoothed = np.full_like(grid, np.nan, dtype=np.float64)
    smoothed[valid] = values[valid] / np.clip(weights[valid], 1.0e-6, None)
    return smoothed


def _plot_pointwise_bias_profile_heatmap(matched, lake_name, model_label, output_path):
    _ensure_chinese_font()
    max_depth = float(matched['Depth_m'].max()) if not matched.empty else 1.0
    matched = matched.copy()
    matched['Date'] = pd.to_datetime(matched['Date'])
    matched = matched.dropna(subset=['Date', 'Depth_m', 'err'])
    matched['plot_day'] = matched['Date'].dt.dayofyear.astype(float)
    finite = matched['err'].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vlim = max(2.0, min(5.0, float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 3.0))

    fig, ax = plt.subplots(figsize=(17, 8.5), constrained_layout=True, dpi=180)
    ax.set_facecolor('#d9d9d9')
    cmap = plt.get_cmap('RdBu_r')
    image = ax.scatter(
        matched['plot_day'],
        matched['Depth_m'],
        c=matched['err'],
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        marker='s',
        s=18 if len(matched) <= 15000 else 10,
        linewidths=0.0,
        alpha=0.92,
    )

    month_midpoints = []
    month_labels = []
    all_dates = pd.date_range(matched['Date'].min(), matched['Date'].max(), freq='D')
    for month, month_dates in pd.Series(all_dates).groupby(pd.Series(all_dates).dt.month):
        month_midpoints.append(float(month_dates.dt.dayofyear.mean()))
        month_labels.append(calendar.month_abbr[int(month)])
    ax.set_xticks(month_midpoints)
    ax.set_xticklabels(month_labels, fontsize=14, fontweight='bold')
    ax.set_xlim(float(all_dates[0].dayofyear) - 0.5, float(all_dates[-1].dayofyear) + 0.5)
    ax.set_ylim(max_depth, 0.0)
    ax.set_xlabel('Month of the Year | 月份', fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel('Depth (m) | 深度', fontsize=16, fontweight='bold', labelpad=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title('Strict Observation-Point Temperature Bias | 严格逐点温度误差', fontsize=24, fontweight='bold', pad=18)
    ax.text(
        0.0,
        1.01,
        f'{lake_name} | {model_label} vs. observed profiles | each square = one matched observed point',
        transform=ax.transAxes,
        fontsize=13,
        color='#555555',
        ha='left',
        va='bottom',
    )
    ax.text(
        0.0,
        -0.10,
        'Red/红色: prediction too warm; Blue/蓝色: prediction too cold; Gray/灰色: no observation point. No interpolation, no smoothing.',
        transform=ax.transAxes,
        fontsize=10,
        color='#555555',
        ha='left',
        va='top',
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.025)
    cbar.set_label('Temperature Bias (°C)\nPrediction - Observation', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_bias_contour_heatmap(matched, lake_name, model_label, output_path):
    _ensure_chinese_font()
    max_depth = float(matched['Depth_m'].max()) if not matched.empty else 1.0
    dates, depths, grid = _pointwise_bias_profile_grid(
        matched,
        max_depth=max_depth,
        depth_step=0.5 if max_depth <= 20.0 else 1.0,
    )
    masked_grid = np.ma.masked_invalid(grid)
    finite = grid[np.isfinite(grid)]
    vlim = max(2.0, min(5.0, float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 3.0))
    if finite.size == 0:
        return

    fig, ax = plt.subplots(figsize=(15.5, 8.5), constrained_layout=True, dpi=180)
    ax.set_facecolor('#d9d9d9')
    day_axis = np.array([date.dayofyear for date in dates], dtype=np.float64)
    filled_levels = np.linspace(-vlim, vlim, 29)
    contour_levels = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float64)
    contour_levels = contour_levels[(contour_levels >= -vlim) & (contour_levels <= vlim)]
    if contour_levels.size < 3:
        contour_levels = np.linspace(-vlim, vlim, 7)

    image = ax.contourf(
        day_axis,
        depths,
        masked_grid,
        levels=filled_levels,
        cmap='RdBu_r',
        extend='both',
    )
    contour_lines = ax.contour(
        day_axis,
        depths,
        masked_grid,
        levels=contour_levels,
        colors='black',
        linewidths=0.8,
        alpha=0.38,
    )
    ax.clabel(contour_lines, fmt='%.0f', fontsize=8, inline=True)

    month_midpoints = []
    month_labels = []
    for month, month_dates in pd.Series(dates).groupby(pd.Series(dates).dt.month):
        month_midpoints.append(float(month_dates.dt.dayofyear.mean()))
        month_labels.append(calendar.month_abbr[int(month)])
    ax.set_xticks(month_midpoints)
    ax.set_xticklabels(month_labels, fontsize=13, fontweight='bold')
    ax.set_xlim(float(day_axis.min()), float(day_axis.max()))
    ax.set_ylim(max_depth, 0.0)
    ax.set_xlabel('Month | 月份', fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel('Depth (m) | 深度', fontsize=16, fontweight='bold', labelpad=12)
    ax.set_title('Interpolated Bias Contour | 插值误差等值图', fontsize=23, fontweight='bold', pad=16)
    ax.text(
        0.0,
        1.01,
        f'{lake_name} | {model_label} | Bias = Prediction - Observation',
        transform=ax.transAxes,
        fontsize=13,
        color='#555555',
        ha='left',
        va='bottom',
    )
    ax.text(
        0.0,
        -0.10,
        'Visualization only: bias is computed at real observation points, then interpolated vertically within each observed profile; no time interpolation.',
        transform=ax.transAxes,
        fontsize=10,
        color='#555555',
        ha='left',
        va='top',
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.025)
    cbar.set_label('Temperature Bias (°C)\nPrediction - Observation', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_prediction_diagnostic_figures(
    truth_csv_path,
    prediction_csv_path,
    output_dir,
    lake_name='Lake',
    model_label='selected prediction',
    file_prefix='prediction',
):
    """Create compact diagnostic figures after predict-time score selection."""
    truth_csv_path = Path(truth_csv_path) if truth_csv_path is not None else None
    prediction_csv_path = Path(prediction_csv_path) if prediction_csv_path is not None else None
    output_dir = Path(output_dir)
    if truth_csv_path is None or not truth_csv_path.exists():
        return {}, 'truth_missing'
    if prediction_csv_path is None or not prediction_csv_path.exists():
        return {}, 'prediction_missing'
    try:
        truth = _load_truth_profile_long(truth_csv_path)
        pred = _load_prediction_profile_long(prediction_csv_path)
        matched = _interpolate_prediction_to_observation_points(truth, pred)
        if matched.empty:
            return {}, 'no_matched_points'
        safe_prefix = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(file_prefix)).strip('_') or 'prediction'
        discrete_path = output_dir / f'{safe_prefix}_discrete_observation_point_evaluation.png'
        bias_path = output_dir / f'{safe_prefix}_bias_contour_heatmap.png'
        _plot_discrete_point_evaluation(matched, lake_name=lake_name, model_label=model_label, output_path=discrete_path)
        _plot_bias_contour_heatmap(matched, lake_name=lake_name, model_label=model_label, output_path=bias_path)
        return {
            'discrete_point_evaluation': discrete_path,
            'bias_contour_heatmap': bias_path,
        }, 'ok'
    except Exception as exc:
        return {}, f'diagnostic_figures_failed:{exc}'
