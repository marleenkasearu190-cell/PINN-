"""R19 eval-only autopsy for LSWT observer update behavior.

This script replays the zero-profile rollout with the diagnostic
``mld_heat_content`` observer and records every raw-open-water LSWT update
before and after the observer is applied.  It does not train, tune, edit
splits, or change model defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lake_pinn.state_multilake import (  # noqa: E402
    DEPTH_STRATIFIED_RMSE_BOUNDARY_M,
    _lake_group_id,
    _lake_reservoir_bucket,
    _lookup_mask,
    train_multilake_state_forecaster,
)
from lake_pinn.state_reconstruction import (  # noqa: E402
    _area_weighted_delta_c,
    _density_inversion_score,
    apply_lswt_observer_update,
    initialize_rollout_state,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
DEFAULT_MANIFEST = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R16_full_eval_point_diagnostics_20260612"
    / "R16_local_R14_manifest_path_mapped.json"
)
DEFAULT_CHECKPOINT = (
    PIPELINE_ROOT
    / "remote_artifact_backups"
    / "RECON_R14_EXPORT_ALIGNED_STATE_PERSISTENCE_DIAG_v1_20260611_1653_direct_stop"
    / "results"
    / "best_by_val_rolling.pt"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R19_observer_update_autopsy_20260612"
)
DEFAULT_VAL_IDS = "erken_2019,erken_2020,mohonk_2017,carvins_cove_2022"
DEFAULT_HELDOUT_IDS = "lacawac_2016,el_val_2019,el_val_2022,namco_2012"


def _parse_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _scalar(value, default=np.nan) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _diag_scalar(detail: dict, key: str, default=np.nan) -> float:
    return _scalar(detail.get(key), default=default)


def _finite_rmse(prediction, target, mask=None) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(valid):
        return float("nan")
    diff = prediction[valid] - target[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def _finite_bias(prediction, target, mask=None) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(prediction[valid] - target[valid]))


def _thermocline_profile_metrics(profile, depths) -> dict[str, float]:
    profile = np.asarray(profile, dtype=np.float64).reshape(-1)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    valid = np.isfinite(profile) & np.isfinite(depths)
    profile = profile[valid]
    depths = depths[valid]
    if profile.size < 2:
        return {
            "thermocline_depth_pred": float("nan"),
            "stratification_strength": float("nan"),
            "max_vertical_gradient_c_per_m": float("nan"),
        }
    order = np.argsort(depths)
    profile = profile[order]
    depths = depths[order]
    dz = np.diff(depths)
    dtemp = np.diff(profile)
    valid_dz = np.isfinite(dz) & (np.abs(dz) > 1.0e-8)
    if not np.any(valid_dz):
        max_gradient = float("nan")
        thermocline_depth = float("nan")
    else:
        gradients = np.full_like(dz, np.nan, dtype=np.float64)
        gradients[valid_dz] = np.abs(dtemp[valid_dz] / dz[valid_dz])
        idx = int(np.nanargmax(gradients))
        thermocline_depth = float(0.5 * (depths[idx] + depths[idx + 1]))
        max_gradient = float(gradients[idx])
    return {
        "thermocline_depth_pred": thermocline_depth,
        "stratification_strength": float(np.nanmax(profile) - np.nanmin(profile)),
        "max_vertical_gradient_c_per_m": max_gradient,
    }


def _band_masks(depths, profile_mask=None) -> dict[str, np.ndarray]:
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    base = np.isfinite(depths)
    if profile_mask is not None:
        base &= np.asarray(profile_mask, dtype=bool).reshape(-1)
    return {
        "surface": base & (depths <= 1.0),
        "le25m": base & (depths <= DEPTH_STRATIFIED_RMSE_BOUNDARY_M),
        "gt25m": base & (depths > DEPTH_STRATIFIED_RMSE_BOUNDARY_M),
        "whole": base,
    }


def _same_day_profile_metrics(lake, date_value, before_np, after_np) -> dict[str, float]:
    target = lake["lookups"]["all"].get(pd.Timestamp(date_value).normalize())
    if target is None:
        return {
            "profile_observed_same_day": 0,
            "surface_rmse_before_update": float("nan"),
            "surface_rmse_after_update": float("nan"),
            "surface_rmse_delta_after_minus_before": float("nan"),
            "le25m_rmse_before_update": float("nan"),
            "le25m_rmse_after_update": float("nan"),
            "le25m_rmse_delta_after_minus_before": float("nan"),
            "gt25m_rmse_before_update": float("nan"),
            "gt25m_rmse_after_update": float("nan"),
            "gt25m_rmse_delta_after_minus_before": float("nan"),
            "whole_profile_rmse_before_update": float("nan"),
            "whole_profile_rmse_after_update": float("nan"),
            "whole_profile_rmse_delta_after_minus_before": float("nan"),
            "whole_profile_bias_before_update": float("nan"),
            "whole_profile_bias_after_update": float("nan"),
        }
    mask = _lookup_mask(lake, "all", date_value)
    masks = _band_masks(lake["depths_np"], profile_mask=mask)
    out = {"profile_observed_same_day": 1}
    for band, band_mask in masks.items():
        before = _finite_rmse(before_np, target, band_mask)
        after = _finite_rmse(after_np, target, band_mask)
        key = "whole_profile" if band == "whole" else band
        out[f"{key}_rmse_before_update"] = before
        out[f"{key}_rmse_after_update"] = after
        out[f"{key}_rmse_delta_after_minus_before"] = (
            after - before if np.isfinite(before) and np.isfinite(after) else float("nan")
        )
    out["whole_profile_bias_before_update"] = _finite_bias(before_np, target, masks["whole"])
    out["whole_profile_bias_after_update"] = _finite_bias(after_np, target, masks["whole"])
    return out


def _delta_band_summary(delta_np, depths_np) -> dict[str, float]:
    delta_np = np.asarray(delta_np, dtype=np.float64).reshape(-1)
    depths_np = np.asarray(depths_np, dtype=np.float64).reshape(-1)
    masks = _band_masks(depths_np)
    out = {
        "update_delta_surface": float(delta_np[0]) if delta_np.size else float("nan"),
        "update_delta_surface_abs": float(abs(delta_np[0])) if delta_np.size else float("nan"),
    }
    for band in ("le25m", "gt25m"):
        values = delta_np[masks[band]]
        finite = values[np.isfinite(values)]
        out[f"update_delta_{band}"] = float(np.mean(finite)) if finite.size else float("nan")
        out[f"update_delta_{band}_abs"] = float(np.mean(np.abs(finite))) if finite.size else float("nan")
    return out


def _run_lake_autopsy(
    model,
    lake,
    *,
    split_label,
    mode,
    spinup_days,
    strength,
    decay_depth_m,
    max_increment_c,
    low_rank_deep_update_fraction,
    heat_content_limit_c,
    min_quality,
    init_mode,
    zero_profile_initializer,
    hard_density_stability,
) -> list[dict]:
    model.eval()
    df = lake["df"]
    init_state = initialize_rollout_state(
        model=model,
        df=df,
        depths=lake["depths_np"],
        all_lookup=lake["lookups"]["all"],
        forcing_rows=lake["forcing_rows"],
        static_features=lake["static_features"],
        metadata=lake["metadata"],
        device=lake["depths"].device,
        init_mode=init_mode,
        rollout_start_date=None,
        spinup_days=spinup_days,
        zero_profile_initializer=zero_profile_initializer,
        spinup_lswt_observer_mode=mode,
        spinup_lst_assimilation_strength=strength,
        spinup_lst_assimilation_decay_depth_m=decay_depth_m,
        spinup_lst_assimilation_max_increment_c=max_increment_c,
        lswt_observer_low_rank_deep_update_fraction=low_rank_deep_update_fraction,
        lswt_observer_heat_content_limit_c=heat_content_limit_c,
        lswt_observer_min_quality=min_quality,
        task_mode="analysis",
        area_profile=lake["area"],
        hard_density_stability=hard_density_stability,
    )
    current = init_state["current"]
    freezing_storage = init_state.get("freezing_storage_j_m2", torch.zeros_like(current))
    rollout_start_idx = int(init_state["rollout_start_idx"])
    depths_tensor = lake["depths"]
    depths_np = np.asarray(lake["depths_np"], dtype=np.float64).reshape(-1)
    rows: list[dict] = []

    for day_idx in range(rollout_start_idx, len(df) - 1):
        next_row = lake["forcing_rows"][day_idx + 1] if day_idx + 1 < len(lake["forcing_rows"]) else None
        current, freezing_storage = model.step(
            current,
            lake["forcing_rows"][day_idx],
            lake["static_features"],
            next_forcing_row=next_row,
            task_mode="analysis",
            depths=depths_tensor,
            area_profile=lake["area"],
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
        if next_row is None:
            continue
        before = current.detach().clone()
        updated, detail = apply_lswt_observer_update(
            current,
            next_row,
            depths_tensor,
            mode=mode,
            strength=strength,
            decay_depth_m=decay_depth_m,
            max_increment_c=max_increment_c,
            low_rank_deep_update_fraction=low_rank_deep_update_fraction,
            heat_content_limit_c=heat_content_limit_c,
            min_quality=min_quality,
            area_profile=lake["area"],
            metadata=lake.get("metadata"),
        )
        current = updated
        date_value = pd.Timestamp(df["Date"].iloc[day_idx + 1]).normalize()
        before_np = before.detach().cpu().numpy().reshape(-1)
        after_np = current.detach().cpu().numpy().reshape(-1)
        delta_np = after_np - before_np
        lswt_value = _scalar(next_row.get("lswt_open_water"))
        t_surface_before = float(before_np[0]) if before_np.size else float("nan")
        t_surface_after = float(after_np[0]) if after_np.size else float("nan")
        innovation_before = (
            lswt_value - t_surface_before
            if np.isfinite(lswt_value) and np.isfinite(t_surface_before)
            else float("nan")
        )
        innovation_after = (
            lswt_value - t_surface_after
            if np.isfinite(lswt_value) and np.isfinite(t_surface_after)
            else float("nan")
        )
        abs_reduction = (
            abs(innovation_before) - abs(innovation_after)
            if np.isfinite(innovation_before) and np.isfinite(innovation_after)
            else float("nan")
        )
        thermocline = _thermocline_profile_metrics(before_np, depths_np)
        mld_depth = _diag_scalar(detail, "lswt_observer_mld_depth_m")
        density_before = _scalar(_density_inversion_score(before))
        density_after = _scalar(_density_inversion_score(current))
        heat_delta_direct = _scalar(
            _area_weighted_delta_c(
                torch.as_tensor(delta_np, dtype=current.dtype, device=current.device).reshape(1, -1),
                depths_tensor.reshape(-1),
                area_profile=lake["area"],
            )
        )
        row = {
            "split": split_label,
            "lake_id": lake["lake_id"],
            "date": date_value.date().isoformat(),
            "lake_group": _lake_group_id(lake),
            "lake_type": _lake_reservoir_bucket(lake),
            "LSWT_open_water_C": lswt_value,
            "T_surface_before_update": t_surface_before,
            "T_surface_after_update": t_surface_after,
            "innovation_before": innovation_before,
            "innovation_after": innovation_after,
            "abs_innovation_reduction": abs_reduction,
            "surface_innovation_reduced": (
                bool(abs_reduction > 0.0) if np.isfinite(abs_reduction) else False
            ),
            "LST_quality_factor": _scalar(next_row.get("lst_quality")),
            "LST_observed_flag": _scalar(next_row.get("lst_observed_flag")),
            "LST_is_filled": _scalar(next_row.get("lst_is_filled")),
            "ice_fraction": _scalar(next_row.get("ice_fraction"), default=0.0),
            "ice_mask": _scalar(next_row.get("ice_mask"), default=0.0),
            "observer_mode": mode,
            "observer_applied_count": _diag_scalar(detail, "lswt_observer_applied_count", default=0.0),
            "observer_quality_mean": _diag_scalar(detail, "lswt_observer_quality_mean"),
            "observer_open_water_weight_mean": _diag_scalar(detail, "lswt_observer_open_water_weight_mean"),
            "observer_surface_innovation_c": _diag_scalar(detail, "lswt_observer_surface_innovation_c"),
            "observer_mean_abs_delta_c": _diag_scalar(detail, "lswt_observer_mean_abs_delta_c"),
            "observer_max_abs_delta_c": _diag_scalar(detail, "lswt_observer_max_abs_delta_c"),
            "observer_filled_lst_used_count": _diag_scalar(
                detail,
                "lswt_observer_filled_lst_used_count",
                default=0.0,
            ),
            "mld_depth": mld_depth,
            "mld_volume_fraction": _diag_scalar(detail, "lswt_observer_mld_volume_fraction"),
            "mld_weight_mean": _diag_scalar(detail, "lswt_observer_mld_weight_mean"),
            "mld_surface_to_heat_gain": _diag_scalar(detail, "lswt_observer_mld_surface_to_heat_gain"),
            "update_heat_content_delta": _diag_scalar(detail, "lswt_observer_heat_content_delta_c"),
            "update_heat_content_delta_direct": heat_delta_direct,
            "update_mld_heat_content_delta": _diag_scalar(
                detail,
                "lswt_observer_mld_heat_content_delta_c",
            ),
            "deep_abs_delta_c": _diag_scalar(detail, "lswt_observer_deep_abs_delta_c"),
            "density_guard_scale": _diag_scalar(detail, "lswt_observer_density_guard_scale", default=1.0),
            "density_inversion_before": density_before,
            "density_inversion_after": density_after,
            "density_adjustment_after_update": bool(
                _diag_scalar(detail, "lswt_observer_density_guard_scale", default=1.0) < 0.999
            ),
            "kalman_gain_surface": _diag_scalar(detail, "lswt_observer_kalman_gain_surface"),
            "kalman_gain_mean": _diag_scalar(detail, "lswt_observer_kalman_gain_mean"),
            "observation_error_c": _diag_scalar(detail, "lswt_observer_observation_error_c"),
            "state_variance_surface": _diag_scalar(detail, "lswt_observer_state_variance_surface"),
            "localization_depth_m": _diag_scalar(detail, "lswt_observer_localization_depth_m"),
            "reservoir_conservative_scale": _diag_scalar(
                detail,
                "lswt_observer_reservoir_conservative_scale",
                default=1.0,
            ),
            "heat_content_bound_scale": _diag_scalar(
                detail,
                "lswt_observer_heat_content_bound_scale",
                default=1.0,
            ),
            "thermocline_depth_pred": thermocline["thermocline_depth_pred"],
            "mld_depth_pred": mld_depth,
            "mld_minus_thermocline_depth": (
                mld_depth - thermocline["thermocline_depth_pred"]
                if np.isfinite(mld_depth) and np.isfinite(thermocline["thermocline_depth_pred"])
                else float("nan")
            ),
            "stratification_strength": thermocline["stratification_strength"],
            "max_vertical_gradient_c_per_m": thermocline["max_vertical_gradient_c_per_m"],
        }
        row.update(_delta_band_summary(delta_np, depths_np))
        row.update(_same_day_profile_metrics(lake, date_value, before_np, after_np))
        rows.append(row)
    return rows


def _safe_mean(series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if not values.empty else float("nan")


def _safe_sum(series) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(values.sum()) if not values.empty else 0.0


def _safe_rate(mask) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(np.mean(np.asarray(mask, dtype=bool)))


def _summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for group_values, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        applied = group[pd.to_numeric(group["observer_applied_count"], errors="coerce").fillna(0.0) > 0.0]
        profile_rows = applied[pd.to_numeric(applied["profile_observed_same_day"], errors="coerce").fillna(0) > 0]
        record = {col: value for col, value in zip(group_cols, group_values)}
        record.update(
            {
                "row_count": int(len(group)),
                "applied_count": int(len(applied)),
                "filled_lst_strong_update_count": int(
                    len(applied[pd.to_numeric(applied["LST_is_filled"], errors="coerce").fillna(0.0) >= 0.5])
                ),
                "surface_improvement_rate": _safe_rate(
                    pd.to_numeric(applied["abs_innovation_reduction"], errors="coerce") > 0.0
                ),
                "surface_no_reduction_count": int(
                    np.sum(pd.to_numeric(applied["abs_innovation_reduction"], errors="coerce") <= 0.0)
                ),
                "abs_innovation_reduction_mean": _safe_mean(applied["abs_innovation_reduction"]),
                "innovation_before_abs_mean": _safe_mean(
                    pd.to_numeric(applied["innovation_before"], errors="coerce").abs()
                ),
                "innovation_after_abs_mean": _safe_mean(
                    pd.to_numeric(applied["innovation_after"], errors="coerce").abs()
                ),
                "mld_depth_mean": _safe_mean(applied["mld_depth"]),
                "mld_volume_fraction_mean": _safe_mean(applied["mld_volume_fraction"]),
                "mld_volume_fraction_gt_025_rate": _safe_rate(
                    pd.to_numeric(applied["mld_volume_fraction"], errors="coerce") > 0.25
                ),
                "mld_volume_fraction_gt_030_rate": _safe_rate(
                    pd.to_numeric(applied["mld_volume_fraction"], errors="coerce") > 0.30
                ),
                "mld_over_thermocline_rate": _safe_rate(
                    pd.to_numeric(applied["mld_minus_thermocline_depth"], errors="coerce") > 0.0
                ),
                "update_delta_surface_mean": _safe_mean(applied["update_delta_surface"]),
                "update_delta_le25m_mean": _safe_mean(applied["update_delta_le25m"]),
                "update_delta_gt25m_mean": _safe_mean(applied["update_delta_gt25m"]),
                "update_heat_content_delta_mean": _safe_mean(applied["update_heat_content_delta"]),
                "update_mld_heat_content_delta_mean": _safe_mean(applied["update_mld_heat_content_delta"]),
                "density_adjustment_rate": _safe_rate(applied["density_adjustment_after_update"]),
                "same_day_profile_count": int(len(profile_rows)),
                "same_day_surface_rmse_delta_mean": _safe_mean(
                    profile_rows["surface_rmse_delta_after_minus_before"]
                ),
                "same_day_le25m_rmse_delta_mean": _safe_mean(
                    profile_rows["le25m_rmse_delta_after_minus_before"]
                ),
                "same_day_gt25m_rmse_delta_mean": _safe_mean(
                    profile_rows["gt25m_rmse_delta_after_minus_before"]
                ),
                "same_day_whole_profile_rmse_delta_mean": _safe_mean(
                    profile_rows["whole_profile_rmse_delta_after_minus_before"]
                ),
                "same_day_whole_profile_regression_rate": _safe_rate(
                    pd.to_numeric(
                        profile_rows["whole_profile_rmse_delta_after_minus_before"],
                        errors="coerce",
                    )
                    > 0.0
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if not isinstance(value, (str, bytes, bool, type(None))) and pd.isna(value):
        return None
    return value


def _write_report(output_dir: Path, summary: dict, paths: dict[str, Path]) -> Path:
    flags = summary["failure_flags"]
    group_records = summary["group_summary"]
    lines = [
        "# R19 Observer Update Autopsy",
        "",
        "Scope: eval-only / diagnostic-only. No training, no remote task, no split/manifest/_standard_inputs changes.",
        "",
        "Protocol: replay R18 zero-profile low-dof + mld_heat_content raw-open-water LSWT observer and record before/after update diagnostics.",
        "",
        "Key failure flags:",
        f"- filled_lst_strong_update_count: {flags.get('filled_lst_strong_update_count')}",
        f"- surface_innovation_not_reduced_fraction: {flags.get('surface_innovation_not_reduced_fraction')}",
        f"- mld_volume_fraction_gt_025_fraction: {flags.get('mld_volume_fraction_gt_025_fraction')}",
        f"- mld_volume_fraction_gt_030_fraction: {flags.get('mld_volume_fraction_gt_030_fraction')}",
        f"- mld_over_thermocline_fraction: {flags.get('mld_over_thermocline_fraction')}",
        f"- reservoir_update_surface_improvement_rate: {flags.get('reservoir_update_surface_improvement_rate')}",
        f"- reservoir_same_day_whole_profile_regression_mean: {flags.get('reservoir_same_day_whole_profile_regression_mean')}",
        "",
        "Split / lake-type summary:",
    ]
    for record in group_records:
        lines.append(
            "- "
            f"{record.get('split')} / {record.get('lake_type')}: "
            f"applied={record.get('applied_count')}, "
            f"surface_improve_rate={record.get('surface_improvement_rate')}, "
            f"mld_volume_mean={record.get('mld_volume_fraction_mean')}, "
            f"vol>0.30={record.get('mld_volume_fraction_gt_030_rate')}, "
            f"same_day_whole_delta={record.get('same_day_whole_profile_rmse_delta_mean')}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- If abs(innovation_after) does not decrease on applied updates, sign/gain/time alignment or surface-temperature semantics are suspect.",
            "- If surface innovation improves but same-day whole-profile RMSE worsens, the observer is likely spreading a valid surface correction too deeply or across the wrong depth band.",
            "- MLD volume fractions above 0.25-0.30 are treated as high-risk for this diagnostic because they allow a surface LSWT residual to alter too much water-column heat content.",
            "- Heldout rows are diagnostic-only and were not used for checkpoint selection or tuning.",
            "",
            "Artifacts:",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: {path}")
    report_path = output_dir / "R19_observer_update_autopsy_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R19 LSWT observer update autopsy.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-ids", default=DEFAULT_VAL_IDS)
    parser.add_argument("--heldout-ids", default=DEFAULT_HELDOUT_IDS)
    parser.add_argument("--mode", default="mld_heat_content")
    parser.add_argument("--init-mode", default="zero_profile_low_dof")
    parser.add_argument("--zero-profile-initializer", default="low_dof")
    parser.add_argument("--spinup-days", type=int, default=90)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--decay-depth-m", type=float, default=2.0)
    parser.add_argument("--max-increment-c", type=float, default=0.75)
    parser.add_argument("--deep-update-fraction", type=float, default=0.10)
    parser.add_argument("--heat-content-limit-c", type=float, default=0.25)
    parser.add_argument("--min-quality", type=float, default=0.05)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_manifest = output_dir / "R19_loader_manifest_epochs0.json"
    manifest_payload = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    manifest_payload["epochs"] = 0
    manifest_payload["export_after_training"] = "off"
    loader_manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = train_multilake_state_forecaster(
        loader_manifest,
        output_dir / "bootstrap_epochs0_loader",
        epochs=0,
        checkpoint_path=args.checkpoint,
        export_after_training="off",
        profile_runtime=False,
        device=device,
    )
    model = result["model"]
    lakes = {lake["lake_id"]: lake for lake in result["lakes"]}
    val_ids = _parse_ids(args.val_ids)
    heldout_ids = _parse_ids(args.heldout_ids)
    missing = [lake_id for lake_id in val_ids + heldout_ids if lake_id not in lakes]
    if missing:
        raise ValueError(f"Requested lake IDs missing from manifest load: {missing}")

    records = []
    with torch.no_grad():
        for split_label, lake_ids in (("val", val_ids), ("heldout_diagnostic_only", heldout_ids)):
            for lake_id in lake_ids:
                print(f"Autopsy rollout {split_label}: {lake_id}", flush=True)
                records.extend(
                    _run_lake_autopsy(
                        model,
                        lakes[lake_id],
                        split_label=split_label,
                        mode=args.mode,
                        spinup_days=args.spinup_days,
                        strength=args.strength,
                        decay_depth_m=args.decay_depth_m,
                        max_increment_c=args.max_increment_c,
                        low_rank_deep_update_fraction=args.deep_update_fraction,
                        heat_content_limit_c=args.heat_content_limit_c,
                        min_quality=args.min_quality,
                        init_mode=args.init_mode,
                        zero_profile_initializer=args.zero_profile_initializer,
                        hard_density_stability=False,
                    )
                )

    updates = pd.DataFrame.from_records(records)
    updates_path = output_dir / "R19_observer_update_autopsy_updates.csv"
    updates.to_csv(updates_path, index=False, encoding="utf-8-sig")

    group_summary = _summarize_group(updates, ["split", "lake_type"])
    group_summary_path = output_dir / "R19_observer_update_autopsy_summary_by_split_lake_type.csv"
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

    lake_summary = _summarize_group(updates, ["split", "lake_id", "lake_type"])
    lake_summary_path = output_dir / "R19_observer_update_autopsy_lake_summary.csv"
    lake_summary.to_csv(lake_summary_path, index=False, encoding="utf-8-sig")

    applied = updates[pd.to_numeric(updates["observer_applied_count"], errors="coerce").fillna(0.0) > 0.0]
    reservoir = applied[applied["lake_type"] == "reservoir"]
    reservoir_profile = reservoir[
        pd.to_numeric(reservoir["profile_observed_same_day"], errors="coerce").fillna(0) > 0
    ]
    failure_flags = {
        "filled_lst_strong_update_count": int(
            np.sum(pd.to_numeric(applied["LST_is_filled"], errors="coerce").fillna(0.0) >= 0.5)
        ),
        "surface_innovation_not_reduced_fraction": _safe_rate(
            pd.to_numeric(applied["abs_innovation_reduction"], errors="coerce") <= 0.0
        ),
        "mld_volume_fraction_gt_025_fraction": _safe_rate(
            pd.to_numeric(applied["mld_volume_fraction"], errors="coerce") > 0.25
        ),
        "mld_volume_fraction_gt_030_fraction": _safe_rate(
            pd.to_numeric(applied["mld_volume_fraction"], errors="coerce") > 0.30
        ),
        "mld_over_thermocline_fraction": _safe_rate(
            pd.to_numeric(applied["mld_minus_thermocline_depth"], errors="coerce") > 0.0
        ),
        "reservoir_update_count": int(len(reservoir)),
        "reservoir_update_delta_heat": _safe_sum(reservoir["update_heat_content_delta"]),
        "reservoir_update_surface_improvement_rate": _safe_rate(
            pd.to_numeric(reservoir["abs_innovation_reduction"], errors="coerce") > 0.0
        ),
        "reservoir_update_surface_improvement_mean": _safe_mean(reservoir["abs_innovation_reduction"]),
        "reservoir_update_whole_profile_regression_rate": _safe_rate(
            pd.to_numeric(
                reservoir_profile["whole_profile_rmse_delta_after_minus_before"],
                errors="coerce",
            )
            > 0.0
        ),
        "reservoir_same_day_whole_profile_regression_mean": _safe_mean(
            reservoir_profile["whole_profile_rmse_delta_after_minus_before"]
        ),
    }
    summary = {
        "diagnostic_id": "RECON_R19_OBSERVER_UPDATE_AUTOPSY_v1",
        "scope": "eval_only_diagnostic_no_training",
        "manifest": str(Path(args.manifest)),
        "checkpoint": str(Path(args.checkpoint)),
        "observer_mode": args.mode,
        "init_mode": args.init_mode,
        "zero_profile_initializer": args.zero_profile_initializer,
        "spinup_days": int(args.spinup_days),
        "observer_strength": float(args.strength),
        "max_increment_c": float(args.max_increment_c),
        "deep_update_fraction": float(args.deep_update_fraction),
        "heat_content_limit_c": float(args.heat_content_limit_c),
        "min_quality": float(args.min_quality),
        "val_ids": val_ids,
        "heldout_diagnostic_only_ids": heldout_ids,
        "row_count": int(len(updates)),
        "applied_count": int(len(applied)),
        "failure_flags": failure_flags,
        "group_summary": group_summary.replace({np.nan: None}).to_dict(orient="records"),
        "lake_summary": lake_summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    summary_path = output_dir / "R19_observer_update_autopsy_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flags_path = output_dir / "R19_observer_update_autopsy_failure_flags.json"
    flags_path.write_text(
        json.dumps(_json_safe(failure_flags), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = _write_report(
        output_dir,
        summary,
        {
            "updates_csv": updates_path,
            "group_summary_csv": group_summary_path,
            "lake_summary_csv": lake_summary_path,
            "summary_json": summary_path,
            "failure_flags_json": flags_path,
        },
    )
    print(f"Wrote {updates_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
