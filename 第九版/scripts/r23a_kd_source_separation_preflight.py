"""R23A eval-only Kd/source-separation preflight diagnostic.

This script reads existing R21B/R22 artifacts only. It does not train, load a
checkpoint, start a remote job, or change any split/manifest/_standard_inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DIAGNOSTIC_ID = "RECON_R23A_KD_GUARD_SOURCE_SEPARATION_PREFLIGHT_v1"


def _num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.max()) if values.notna().any() else float("nan")


def _sum(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.sum()) if values.notna().any() else 0.0


def _corr_rows(df: pd.DataFrame, subset_name: str) -> list[dict]:
    rows: list[dict] = []
    pairs = [
        ("kd_p95_delta_e23_minus_e7", "whole_rmse_delta_on_e23_minus_e7"),
        ("kd_p95_delta_e23_minus_e7", "surface_rmse_delta_on_e23_minus_e7"),
        ("kd_p95_delta_e23_minus_e7", "whole_observer_gain_e23"),
        ("kd_p95_delta_e23_minus_e7", "surface_observer_gain_e23"),
        ("kd_mean_delta_e23_minus_e7", "whole_rmse_delta_on_e23_minus_e7"),
        ("kd_mean_delta_e23_minus_e7", "surface_rmse_delta_on_e23_minus_e7"),
        ("nn_kd_multiplier_p95_e23", "whole_rmse_e23_observer_on_s020"),
        ("nn_kd_multiplier_p95_e23", "surface_rmse_e23_observer_on_s020"),
        ("nn_kd_multiplier_saturation_fraction_e23", "whole_rmse_e23_observer_on_s020"),
        ("nn_kd_multiplier_saturation_fraction_e23", "whole_bias_e23_observer_on_s020"),
        ("residual_abs_mean_c_delta_e23_minus_e7", "whole_rmse_delta_on_e23_minus_e7"),
        ("kz_mean_delta_e23_minus_e7", "whole_rmse_delta_on_e23_minus_e7"),
    ]
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        n = len(xy)
        pearson = float("nan")
        spearman = float("nan")
        if n >= 3 and xy[x_col].nunique() > 1 and xy[y_col].nunique() > 1:
            pearson = float(np.corrcoef(xy[x_col], xy[y_col])[0, 1])
            ranked = xy.rank(method="average")
            spearman = float(np.corrcoef(ranked[x_col], ranked[y_col])[0, 1])
        rows.append(
            {
                "subset": subset_name,
                "x": x_col,
                "y": y_col,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
                "note": "small_n_diagnostic_not_for_tuning" if n < 8 else "diagnostic_only",
            }
        )
    return rows


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _summarize_lake_groups(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(groups, dropna=False)
        .agg(
            lake_count=("lake_id", "nunique"),
            kd_mean_delta_mean=("kd_mean_delta_e23_minus_e7", _mean),
            kd_p95_delta_mean=("kd_p95_delta_e23_minus_e7", _mean),
            kd_p95_e23_mean=("nn_kd_multiplier_p95_e23", _mean),
            kd_p95_e23_max=("nn_kd_multiplier_p95_e23", _max),
            kd_saturation_e23_mean=("nn_kd_multiplier_saturation_fraction_e23", _mean),
            kd_saturation_e23_max=("nn_kd_multiplier_saturation_fraction_e23", _max),
            whole_rmse_delta_on_mean=("whole_rmse_delta_on_e23_minus_e7", _mean),
            surface_rmse_delta_on_mean=("surface_rmse_delta_on_e23_minus_e7", _mean),
            whole_observer_gain_e23_mean=("whole_observer_gain_e23", _mean),
            surface_observer_gain_e23_mean=("surface_observer_gain_e23", _mean),
            residual_abs_delta_mean=("residual_abs_mean_c_delta_e23_minus_e7", _mean),
            kz_delta_mean=("kz_mean_delta_e23_minus_e7", _mean),
        )
        .reset_index()
    )
    return agg


def build_diagnostics(args: argparse.Namespace) -> dict:
    r22_dir = Path(args.r22_dir)
    r21b_dir = Path(args.r21b_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lake_metrics = pd.read_csv(r22_dir / "R22_lake_metrics.csv")
    kd_lake = pd.read_csv(r22_dir / "R22_kd_diagnostics_by_lake.csv")
    history = pd.read_csv(r21b_dir / "global_state_forecaster_training_history.csv")
    updates = pd.read_csv(r22_dir / "R22_observer_update_autopsy_updates.csv")

    metric_num_cols = [
        "checkpoint_epoch",
        "strength",
        "reservoir_scale",
        "rmse",
        "bias",
        "rmse_le25m",
        "bias_le25m",
        "rmse_gt25m",
        "bias_gt25m",
        "surface_rmse",
        "surface_bias",
        "filled_lst_strong_update_count",
        "observer_deep_abs_delta_mean_c",
        "observer_localization_depth_mean_m",
    ]
    kd_num_cols = [
        "checkpoint_epoch",
        "nn_kd_multiplier_mean",
        "nn_kd_multiplier_p50",
        "nn_kd_multiplier_p95",
        "nn_kd_multiplier_saturation_fraction",
        "kd_base_mean",
        "adaptive_kd_multiplier_mean",
        "residual_abs_mean_c",
        "kz_mean",
    ]
    lake_metrics = _num(lake_metrics, metric_num_cols)
    kd_lake = _num(kd_lake, kd_num_cols)

    keys = ["split", "lake_id", "lake_group", "lake_type"]
    kd7 = kd_lake[kd_lake["checkpoint_epoch"] == 7].copy()
    kd23 = kd_lake[kd_lake["checkpoint_epoch"] == 23].copy()
    kd_wide = kd7.merge(kd23, on=keys, suffixes=("_e7", "_e23"))
    for base in [
        "nn_kd_multiplier_mean",
        "nn_kd_multiplier_p50",
        "nn_kd_multiplier_p95",
        "nn_kd_multiplier_saturation_fraction",
        "kd_base_mean",
        "adaptive_kd_multiplier_mean",
        "residual_abs_mean_c",
        "kz_mean",
    ]:
        e7 = f"{base}_e7"
        e23 = f"{base}_e23"
        if e7 in kd_wide.columns and e23 in kd_wide.columns:
            kd_wide[f"{base.replace('nn_kd_multiplier_', 'kd_')}_delta_e23_minus_e7"] = (
                pd.to_numeric(kd_wide[e23], errors="coerce")
                - pd.to_numeric(kd_wide[e7], errors="coerce")
            )

    onoff = lake_metrics[lake_metrics["family"] == "on_off"].copy()
    metric_cols = [
        "rmse",
        "bias",
        "rmse_le25m",
        "bias_le25m",
        "rmse_gt25m",
        "bias_gt25m",
        "surface_rmse",
        "surface_bias",
        "filled_lst_strong_update_count",
        "observer_deep_abs_delta_mean_c",
        "observer_localization_depth_mean_m",
    ]
    pivot = onoff.pivot_table(
        index=keys,
        columns=["checkpoint_epoch", "config_id"],
        values=metric_cols,
        aggfunc="mean",
    )
    pivot.columns = [
        f"{metric}_e{int(epoch)}_{config}" for metric, epoch, config in pivot.columns
    ]
    pivot = pivot.reset_index()
    lake = kd_wide.merge(pivot, on=keys, how="left")

    def add_delta(name: str, out: str, e23_suffix: str, e7_suffix: str) -> None:
        c23 = f"{name}_{e23_suffix}"
        c7 = f"{name}_{e7_suffix}"
        if c23 in lake.columns and c7 in lake.columns:
            lake[out] = (
                pd.to_numeric(lake[c23], errors="coerce")
                - pd.to_numeric(lake[c7], errors="coerce")
            )

    add_delta("rmse", "whole_rmse_delta_on_e23_minus_e7", "e23_observer_on_s020", "e7_observer_on_s020")
    add_delta(
        "surface_rmse",
        "surface_rmse_delta_on_e23_minus_e7",
        "e23_observer_on_s020",
        "e7_observer_on_s020",
    )
    add_delta(
        "rmse_le25m",
        "le25m_rmse_delta_on_e23_minus_e7",
        "e23_observer_on_s020",
        "e7_observer_on_s020",
    )
    add_delta(
        "rmse_gt25m",
        "gt25m_rmse_delta_on_e23_minus_e7",
        "e23_observer_on_s020",
        "e7_observer_on_s020",
    )

    rename_metrics = {
        "rmse_e7_observer_on_s020": "whole_rmse_e7_observer_on_s020",
        "rmse_e23_observer_on_s020": "whole_rmse_e23_observer_on_s020",
        "bias_e7_observer_on_s020": "whole_bias_e7_observer_on_s020",
        "bias_e23_observer_on_s020": "whole_bias_e23_observer_on_s020",
        "surface_rmse_e7_observer_on_s020": "surface_rmse_e7_observer_on_s020",
        "surface_rmse_e23_observer_on_s020": "surface_rmse_e23_observer_on_s020",
        "surface_bias_e7_observer_on_s020": "surface_bias_e7_observer_on_s020",
        "surface_bias_e23_observer_on_s020": "surface_bias_e23_observer_on_s020",
    }
    lake = lake.rename(columns=rename_metrics)
    for epoch in (7, 23):
        off = f"rmse_e{epoch}_observer_off"
        on = f"whole_rmse_e{epoch}_observer_on_s020"
        if off in lake.columns and on in lake.columns:
            lake[f"whole_observer_gain_e{epoch}"] = (
                pd.to_numeric(lake[off], errors="coerce")
                - pd.to_numeric(lake[on], errors="coerce")
            )
        soff = f"surface_rmse_e{epoch}_observer_off"
        son = f"surface_rmse_e{epoch}_observer_on_s020"
        if soff in lake.columns and son in lake.columns:
            lake[f"surface_observer_gain_e{epoch}"] = (
                pd.to_numeric(lake[soff], errors="coerce")
                - pd.to_numeric(lake[son], errors="coerce")
            )

    lake["kd_risk_e23"] = (
        (pd.to_numeric(lake["nn_kd_multiplier_p95_e23"], errors="coerce") >= 1.23)
        | (pd.to_numeric(lake["nn_kd_multiplier_saturation_fraction_e23"], errors="coerce") >= 0.05)
    )

    lake_path = output_dir / "R23A_kd_drift_metric_gain_by_lake.csv"
    lake.to_csv(lake_path, index=False)

    by_type = _summarize_lake_groups(lake, ["split", "lake_type"])
    by_group = _summarize_lake_groups(lake, ["split", "lake_group", "lake_type"])
    by_type_path = output_dir / "R23A_kd_drift_by_lake_type.csv"
    by_group_path = output_dir / "R23A_kd_drift_by_lake_group.csv"
    by_type.to_csv(by_type_path, index=False)
    by_group.to_csv(by_group_path, index=False)

    corr_rows: list[dict] = []
    corr_rows.extend(_corr_rows(lake, "all_lakes"))
    for split, sub in lake.groupby("split", dropna=False):
        corr_rows.extend(_corr_rows(sub, f"split={split}"))
    for lake_type, sub in lake.groupby("lake_type", dropna=False):
        corr_rows.extend(_corr_rows(sub, f"lake_type={lake_type}"))
    corr = pd.DataFrame(corr_rows)
    corr_path = output_dir / "R23A_kd_metric_gain_correlations.csv"
    corr.to_csv(corr_path, index=False)

    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")
    updates["month"] = updates["date"].dt.month
    updates["season"] = updates["month"].apply(lambda x: _season(int(x)) if pd.notna(x) else "unknown")
    update_num_cols = [
        "checkpoint_epoch",
        "LST_is_filled",
        "observer_filled_lst_used_count",
        "innovation_before",
        "innovation_after",
        "surface_innovation_reduced",
        "deep_abs_delta_c",
        "localization_depth_m",
        "mld_over_thermocline_rate",
        "mld_minus_thermocline_depth",
        "whole_profile_rmse_delta_after_minus_before",
        "surface_rmse_delta_after_minus_before",
        "profile_observed_same_day",
        "reservoir_scale_applied",
        "effective_strength",
    ]
    updates = _num(updates, update_num_cols)
    seasonal = (
        updates.groupby(["checkpoint_epoch", "split", "lake_type", "season"], dropna=False)
        .agg(
            update_count=("observer_applied_count", "count"),
            filled_lst_strong_update_count=("observer_filled_lst_used_count", _sum),
            raw_filled_feature_rows=("LST_is_filled", _sum),
            surface_improvement_rate=("surface_innovation_reduced", _mean),
            abs_innovation_before_mean=("innovation_before", lambda s: _mean(s.abs())),
            abs_innovation_after_mean=("innovation_after", lambda s: _mean(s.abs())),
            same_day_profile_count=("profile_observed_same_day", _sum),
            same_day_surface_rmse_delta_mean=("surface_rmse_delta_after_minus_before", _mean),
            same_day_whole_rmse_delta_mean=("whole_profile_rmse_delta_after_minus_before", _mean),
            deep_abs_delta_mean=("deep_abs_delta_c", _mean),
            localization_depth_mean=("localization_depth_m", _mean),
            thermocline_crossing_proxy_rate=(
                "mld_minus_thermocline_depth",
                lambda s: float((pd.to_numeric(s, errors="coerce") > 0.0).mean()) if len(s) else float("nan"),
            ),
        )
        .reset_index()
    )
    seasonal["kd_by_season_available_in_current_artifacts"] = False
    seasonal_path = output_dir / "R23A_seasonal_observer_update_proxy.csv"
    seasonal.to_csv(seasonal_path, index=False)

    e23_on = onoff[
        (onoff["checkpoint_epoch"] == 23) & (onoff["config_id"] == "observer_on_s020")
    ].copy()
    kd23_small = kd23[
        keys
        + [
            "nn_kd_multiplier_mean",
            "nn_kd_multiplier_p50",
            "nn_kd_multiplier_p95",
            "nn_kd_multiplier_saturation_fraction",
            "residual_abs_mean_c",
            "kz_mean",
        ]
    ].copy()
    bias = e23_on.merge(kd23_small, on=keys, how="left")
    bias["surface_bias_bin"] = pd.cut(
        pd.to_numeric(bias["surface_bias"], errors="coerce"),
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=["cold_gt1c", "cold_025_1c", "near_zero", "warm_025_1c", "warm_gt1c"],
    )
    bias_summary = (
        bias.groupby(["split", "lake_type", "surface_bias_bin"], dropna=False, observed=False)
        .agg(
            lake_count=("lake_id", "nunique"),
            surface_bias_mean=("surface_bias", _mean),
            whole_bias_mean=("bias", _mean),
            whole_rmse_mean=("rmse", _mean),
            surface_rmse_mean=("surface_rmse", _mean),
            kd_p95_mean=("nn_kd_multiplier_p95", _mean),
            kd_saturation_mean=("nn_kd_multiplier_saturation_fraction", _mean),
            residual_abs_mean=("residual_abs_mean_c", _mean),
            kz_mean=("kz_mean", _mean),
        )
        .reset_index()
    )
    bias_path = output_dir / "R23A_surface_bias_kd_bins.csv"
    bias_summary.to_csv(bias_path, index=False)

    full_history = _num(
        history[history.get("eval_mode", "") == "full"].copy(),
        [
            "epoch",
            "nn_kd_multiplier_mean",
            "nn_kd_multiplier_p50",
            "nn_kd_multiplier_p95",
            "nn_kd_multiplier_saturation_fraction",
            "kd_saturation_penalty_loss",
            "kd_saturation_penalty_weighted_loss",
            "val_zero_profile_export_mean_rmse",
            "val_zero_profile_export_surface_rmse",
            "val_rolling_start_rmse_30d",
            "val_rolling_start_rmse_60d",
            "val_fewshot_rmse_30d",
            "val_fewshot_rmse_60d",
            "residual_abs_mean_c",
            "kz_mean",
        ],
    )
    history_path = output_dir / "R23A_history_full_eval_kd_trace.csv"
    full_history.to_csv(history_path, index=False)

    max_val_reservoir_sat = _max(
        lake[
            (lake["split"] == "val")
            & (lake["lake_type"] == "reservoir")
        ]["nn_kd_multiplier_saturation_fraction_e23"]
    )
    max_any_p95 = _max(lake["nn_kd_multiplier_p95_e23"])
    kd_delta_mean = _mean(lake["kd_p95_delta_e23_minus_e7"])
    heldout_whole_mean = _mean(
        lake[lake["split"] == "heldout_diagnostic_only"][
            "whole_rmse_e23_observer_on_s020"
        ]
    )
    val_whole_delta = _mean(
        lake[lake["split"] == "val"]["whole_rmse_delta_on_e23_minus_e7"]
    )
    safety = {
        "filled_lst_strong_update_count": _safe_float(
            seasonal["filled_lst_strong_update_count"].sum()
        ),
        "deep_abs_delta_mean": _safe_float(_mean(seasonal["deep_abs_delta_mean"])),
        "kd_by_season_available_in_current_artifacts": False,
    }
    plain_training_go = False
    kd_guard_ablation_candidate = (
        max_any_p95 is not None
        and max_any_p95 >= 1.23
        and safety["filled_lst_strong_update_count"] == 0.0
    )
    decision = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "scope": "eval_only_existing_artifacts_no_training_no_remote_no_model_or_data_change",
        "plain_r23_training_go": plain_training_go,
        "r23b_kd_guard_short_training_candidate": bool(kd_guard_ablation_candidate),
        "r23b_requires_separate_user_approval": True,
        "heldout_policy": "diagnostic_only_not_for_checkpoint_or_tuning",
        "kd_compensation_risk": bool(max_any_p95 is not None and max_any_p95 >= 1.23),
        "reservoir_kd_risk": bool(max_val_reservoir_sat is not None and max_val_reservoir_sat >= 0.10),
        "observer_safety": safety,
        "key_numbers": {
            "mean_kd_p95_delta_e23_minus_e7": _safe_float(kd_delta_mean),
            "max_kd_p95_epoch23": _safe_float(max_any_p95),
            "max_val_reservoir_kd_saturation_epoch23": _safe_float(max_val_reservoir_sat),
            "heldout_diag_epoch23_observer_on_whole_rmse_mean": _safe_float(heldout_whole_mean),
            "val_observer_on_whole_rmse_delta_e23_minus_e7_mean": _safe_float(val_whole_delta),
        },
        "artifacts": {
            "lake": str(lake_path),
            "by_type": str(by_type_path),
            "by_group": str(by_group_path),
            "correlations": str(corr_path),
            "seasonal_proxy": str(seasonal_path),
            "surface_bias_bins": str(bias_path),
            "history_trace": str(history_path),
        },
    }

    summary_path = output_dir / "R23A_kd_source_separation_preflight_summary.json"
    summary_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# R23A Kd Guard + Source-Separation Preflight",
        "",
        "Scope: eval-only from existing R21B/R22 artifacts. No training, no remote job, no model/loss/bounds/split/manifest/_standard_inputs change.",
        "",
        "Heldout rows are diagnostic-only and were not used for checkpoint, gain, Kd guard, or reservoir-scale selection.",
        "",
        "## Main Finding",
        "",
        "- Conservative surface observer remains locally useful, but plain R23 training is still No-Go.",
        f"- Mean Kd p95 drift from epoch7 to epoch23: {kd_delta_mean:.3f}.",
        f"- Max epoch23 Kd p95 across checked lake rows: {max_any_p95:.3f}.",
        f"- Max validation reservoir Kd saturation at epoch23: {max_val_reservoir_sat:.3f}.",
        f"- Heldout diagnostic-only epoch23 observer-on whole RMSE mean: {heldout_whole_mean:.3f} C.",
        "",
        "## Lake-Type Summary",
        "",
    ]
    for _, row in by_type.sort_values(["split", "lake_type"]).iterrows():
        report_lines.append(
            "- "
            f"{row['split']} {row['lake_type']}: "
            f"kd_p95_delta={row['kd_p95_delta_mean']:.3f}, "
            f"kd_p95_e23_mean/max={row['kd_p95_e23_mean']:.3f}/{row['kd_p95_e23_max']:.3f}, "
            f"sat_e23_mean/max={row['kd_saturation_e23_mean']:.3f}/{row['kd_saturation_e23_max']:.3f}, "
            f"whole_delta_on={row['whole_rmse_delta_on_mean']:.3f} C, "
            f"surface_delta_on={row['surface_rmse_delta_on_mean']:.3f} C"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Plain R23 short training: No-Go.",
            "- R23B Kd-guard short ablation: candidate only, requires separate approval.",
            "- Recommended next action: approve only a Kd-guard ablation brief if the goal is to train; otherwise continue eval-only reservoir source separation.",
            "",
            "## Artifact Paths",
        ]
    )
    for name, path in decision["artifacts"].items():
        report_lines.append(f"- {name}: {path}")
    report_path = output_dir / "R23A_kd_source_separation_preflight_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    decision["artifacts"]["summary_json"] = str(summary_path)
    decision["artifacts"]["report"] = str(report_path)
    summary_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r22-dir",
        default=(
            "<local-pipeline>/reports/failure_diagnosis/"
            "R22_conservative_surface_observer_generalization_autopsy_20260613"
        ),
    )
    parser.add_argument(
        "--r21b-dir",
        default=(
            "<local-pipeline>/remote_artifact_backups/"
            "RECON_R21B_CONSERVATIVE_SURFACE_OBSERVER_LONGDIAG_v2_20260613_completed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "<local-pipeline>/reports/failure_diagnosis/"
            "R23A_kd_source_separation_preflight_20260613"
        ),
    )
    args = parser.parse_args()
    decision = build_diagnostics(args)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
