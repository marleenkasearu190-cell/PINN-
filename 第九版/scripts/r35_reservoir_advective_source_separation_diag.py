"""R35 reservoir advective source-separation diagnostic.

This is an eval-only/data-only diagnostic. It reads an existing R25 result
directory and standard inputs, then asks whether reservoir errors align with
available hydrology / advective heat-source evidence.

No model, loss, split, checkpoint, or standard input file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ID = "RECON_R35_RESERVOIR_ADVECTIVE_SOURCE_SEPARATION_DIAG_v1"
RUN_TAG = "R35_reservoir_advective_source_separation_diag_20260614"
SECONDS_PER_DAY = 86400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R35 reservoir advective source-separation diagnostic.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/root/pinn_r10/ninth/results/RECON_R25_LOCAL62_NONSTRESS_SHORTDIAG_v1"),
    )
    parser.add_argument(
        "--standard-input-root",
        type=Path,
        default=Path("/root/pinn_r10/data/_standard_inputs"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/root/pinn_r10/ninth/experiments/manifests_clean/diagnostics/RECON_R25_LOCAL62_NONSTRESS_SHORTDIAG_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/pinn_r10/ninth/results") / EXPERIMENT_ID,
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def safe_float(value, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def finite_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def parse_lake_group(lake_id: str) -> str:
    match = re.match(r"^(.*)_\d{4}$", str(lake_id))
    return match.group(1) if match else str(lake_id)


def load_manifest_or_split(result_dir: Path, manifest_path: Path) -> dict:
    if manifest_path.exists():
        return read_json(manifest_path)
    launch = result_dir / "launch_command.txt"
    if launch.exists():
        text = launch.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"(/[^ \n\r\t]+\.json)", text)
        for item in matches:
            path = Path(item)
            if path.exists():
                return read_json(path)
    split_summary = result_dir / "global_state_forecaster_split_summary.json"
    summary = read_json(split_summary)
    return {"all_lake_ids": list(summary)}


def all_lake_ids(manifest: dict, split_summary: dict) -> list[str]:
    ids: list[str] = []
    for key in ("train_lake_ids", "val_lake_ids", "heldout_lake_ids", "test_lake_ids", "stress_or_ood_lake_ids"):
        ids.extend(str(x) for x in manifest.get(key, []))
    if not ids:
        ids.extend(str(x) for x in split_summary.keys())
    return list(dict.fromkeys(ids))


def role_from_manifest(manifest: dict, split_summary: dict, lake_id: str) -> str:
    if lake_id in set(manifest.get("train_lake_ids", [])):
        return "train"
    if lake_id in set(manifest.get("val_lake_ids", [])):
        return "validation"
    if lake_id in set(manifest.get("heldout_lake_ids", manifest.get("test_lake_ids", []))):
        return "heldout_diagnostic_only"
    if lake_id in set(manifest.get("stress_or_ood_lake_ids", [])):
        return "stress_ood_diagnostic_only"
    row = split_summary.get(lake_id, {})
    if row.get("is_heldout_test_lake") or row.get("is_excluded_by_heldout_group"):
        return "heldout_diagnostic_only"
    return "train_or_validation_unknown"


def is_reservoir(metadata: dict) -> bool:
    for key in ("reservoir_indicator", "is_reservoir", "metadata_is_reservoir"):
        value = metadata.get(key)
        if value is not None and not pd.isna(value):
            try:
                return float(value) > 0.5
            except (TypeError, ValueError):
                pass
    return "reservoir" in str(metadata.get("lake_type", "")).lower()


def estimate_volume_m3(metadata: dict) -> float:
    volume_km3 = safe_float(metadata.get("volume_km3"))
    if math.isfinite(volume_km3) and volume_km3 > 0.0:
        return volume_km3 * 1.0e9
    area_km2 = safe_float(metadata.get("area_km2", metadata.get("surface_area_km2", np.nan)))
    mean_depth_m = safe_float(metadata.get("mean_depth_m", metadata.get("mean_depth", np.nan)))
    if math.isfinite(area_km2) and area_km2 > 0.0 and math.isfinite(mean_depth_m) and mean_depth_m > 0.0:
        return area_km2 * 1.0e6 * mean_depth_m
    return float("nan")


def hydrology_status(era5: pd.DataFrame | None) -> str:
    if era5 is None or era5.empty:
        return "missing_era5"
    if (finite_series(era5, ["net_inflow", "net_inflow_m3_s", "NetInflow_m3_s"]).abs() > 1.0e-8).any():
        return "net_inflow_available"
    inflow = finite_series(era5, ["inflow_m3_s", "Inflow_m3_s", "river_inflow_m3_s", "qin_m3_s"])
    outflow = finite_series(era5, ["outflow_m3_s", "Outflow_m3_s", "river_outflow_m3_s", "qout_m3_s"])
    if (inflow.abs() > 1.0e-8).any() or (outflow.abs() > 1.0e-8).any():
        return "inflow_outflow_available"
    return "missing_or_zero_filled"


def compute_lake_advective_inputs(lake_id: str, role: str, standard_root: Path) -> dict:
    lake_dir = standard_root / lake_id
    metadata_path = lake_dir / "metadata.json"
    era5_path = lake_dir / "era5_for_model.csv"
    profile_path = lake_dir / "profile_for_model.csv"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    era5 = read_csv(era5_path) if era5_path.exists() else None
    reservoir = is_reservoir(metadata)
    volume = estimate_volume_m3(metadata)
    net_inflow = finite_series(era5, ["net_inflow", "net_inflow_m3_s", "NetInflow_m3_s"]) if era5 is not None else pd.Series(dtype="float64")
    if era5 is not None and not net_inflow.notna().any():
        inflow = finite_series(era5, ["inflow_m3_s", "Inflow_m3_s", "river_inflow_m3_s", "qin_m3_s"]).fillna(0.0)
        outflow = finite_series(era5, ["outflow_m3_s", "Outflow_m3_s", "river_outflow_m3_s", "qout_m3_s"]).fillna(0.0)
        net_inflow = inflow - outflow
    net_inflow = pd.to_numeric(net_inflow, errors="coerce") if len(net_inflow) else pd.Series(dtype="float64")
    positive = net_inflow.clip(lower=0.0)
    exchange = positive * SECONDS_PER_DAY / volume if math.isfinite(volume) and volume > 0.0 else pd.Series(np.nan, index=net_inflow.index)
    exchange_clipped = exchange.clip(lower=0.0, upper=0.10)
    air = finite_series(era5, ["t2m_C", "T_air_C", "air_temp_C"]) if era5 is not None else pd.Series(dtype="float64")
    if era5 is not None and not air.notna().any() and "t2m_K" in era5.columns:
        air = pd.to_numeric(era5["t2m_K"], errors="coerce") - 273.15
    profile_median = float("nan")
    if profile_path.exists():
        profile = read_csv(profile_path)
        temp = finite_series(profile, ["Temperature_C"])
        if not temp.notna().any():
            temp_cols = [c for c in profile.columns if c.startswith("Temp_")]
            vals = []
            for col in temp_cols:
                vals.extend(pd.to_numeric(profile[col], errors="coerce").dropna().tolist())
            profile_median = float(np.nanmedian(vals)) if vals else float("nan")
        else:
            profile_median = float(temp.dropna().median())
    air_minus_profile = air - profile_median if math.isfinite(profile_median) and len(air) else pd.Series(dtype="float64")
    active = (reservoir and len(net_inflow) > 0 and math.isfinite(volume) and volume > 0.0)
    advective_heat_proxy_c_day = exchange_clipped * air_minus_profile if len(air_minus_profile) else pd.Series(np.nan, index=exchange_clipped.index)
    return {
        "lake_id": lake_id,
        "lake_group": parse_lake_group(lake_id),
        "role": role,
        "lake_type": "reservoir" if reservoir else "natural",
        "metadata_present": metadata_path.exists(),
        "era5_present": era5_path.exists(),
        "profile_present": profile_path.exists(),
        "hydrology_status": hydrology_status(era5),
        "volume_m3": volume,
        "net_inflow_days": int(net_inflow.notna().sum()),
        "net_inflow_nonzero_days": int((net_inflow.fillna(0.0).abs() > 1.0e-8).sum()) if len(net_inflow) else 0,
        "positive_net_inflow_days": int((net_inflow.fillna(0.0) > 0.0).sum()) if len(net_inflow) else 0,
        "positive_net_inflow_fraction": float((net_inflow.fillna(0.0) > 0.0).mean()) if len(net_inflow) else np.nan,
        "net_inflow_mean_m3_s": float(net_inflow.mean()) if len(net_inflow) else np.nan,
        "net_inflow_p95_m3_s": float(net_inflow.quantile(0.95)) if len(net_inflow) else np.nan,
        "net_inflow_min_m3_s": float(net_inflow.min()) if len(net_inflow) else np.nan,
        "net_inflow_max_m3_s": float(net_inflow.max()) if len(net_inflow) else np.nan,
        "simple_advective_active_candidate": bool(active and (positive > 0.0).any()),
        "exchange_fraction_mean_per_day": float(exchange_clipped.mean()) if len(exchange_clipped) else np.nan,
        "exchange_fraction_p95_per_day": float(exchange_clipped.quantile(0.95)) if len(exchange_clipped) else np.nan,
        "exchange_fraction_max_per_day": float(exchange_clipped.max()) if len(exchange_clipped) else np.nan,
        "advective_heat_proxy_mean_c_day": float(advective_heat_proxy_c_day.mean()) if len(advective_heat_proxy_c_day) else np.nan,
        "advective_heat_proxy_abs_mean_c_day": float(advective_heat_proxy_c_day.abs().mean()) if len(advective_heat_proxy_c_day) else np.nan,
        "advective_heat_proxy_p95_c_day": float(advective_heat_proxy_c_day.quantile(0.95)) if len(advective_heat_proxy_c_day) else np.nan,
        "profile_median_temperature_c": profile_median,
    }


def value_from_row(row: pd.Series, name: str) -> float:
    if name not in row.index:
        return float("nan")
    return safe_float(row[name])


def best_or_final_rows(history: pd.DataFrame, metrics_path: Path) -> tuple[pd.Series, pd.Series, int, int]:
    history = history.copy()
    history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
    final = history.sort_values("epoch").iloc[-1]
    best_epoch = safe_float(read_json(metrics_path).get("epoch")) if metrics_path.exists() else float("nan")
    if math.isfinite(best_epoch):
        match = history[history["epoch"] == int(best_epoch)]
        if not match.empty:
            best = match.iloc[-1]
        else:
            best = final
    else:
        best = final
    return best, final, int(value_from_row(best, "epoch")), int(value_from_row(final, "epoch"))


def extract_lake_metrics(lake_id: str, role: str, row: pd.Series) -> dict:
    candidates = {
        "point_rmse": [
            f"{lake_id}_val_rmse",
            f"{lake_id}_train_rmse",
            f"{lake_id}_heldout_mean_rmse",
        ],
        "point_rmse_le25m": [
            f"{lake_id}_val_rmse_le25m",
            f"{lake_id}_train_rmse_le25m",
            f"{lake_id}_heldout_mean_rmse_le25m",
        ],
        "point_rmse_gt25m": [
            f"{lake_id}_val_rmse_gt25m",
            f"{lake_id}_train_rmse_gt25m",
            f"{lake_id}_heldout_mean_rmse_gt25m",
        ],
        "free_roll_rmse": [
            f"{lake_id}_heldout_free_roll_rmse",
            f"{lake_id}_val_free_roll_rmse",
            f"{lake_id}_val_rolling_start_rmse_60d",
            f"{lake_id}_val_rolling_start_rmse_30d",
        ],
        "free_roll_bias": [
            f"{lake_id}_heldout_free_roll_bias",
            f"{lake_id}_val_free_roll_bias",
            f"{lake_id}_val_rolling_start_bias_60d",
            f"{lake_id}_val_rolling_start_bias_30d",
        ],
        "free_roll_rmse_le25m": [
            f"{lake_id}_heldout_free_roll_rmse_le25m",
            f"{lake_id}_val_rolling_start_rmse_le25m_60d",
            f"{lake_id}_val_rolling_start_rmse_le25m_30d",
        ],
        "free_roll_rmse_gt25m": [
            f"{lake_id}_heldout_free_roll_rmse_gt25m",
            f"{lake_id}_val_rolling_start_rmse_gt25m_60d",
            f"{lake_id}_val_rolling_start_rmse_gt25m_30d",
        ],
        "rolling30_rmse": [f"{lake_id}_val_rolling_start_rmse_30d", f"{lake_id}_heldout_rolling_start_rmse_30d"],
        "rolling60_rmse": [f"{lake_id}_val_rolling_start_rmse_60d", f"{lake_id}_heldout_rolling_start_rmse_60d"],
    }
    out = {"lake_id": lake_id, "role": role}
    for metric, cols in candidates.items():
        out[metric] = np.nan
        out[f"{metric}_source_col"] = ""
        for col in cols:
            val = value_from_row(row, col)
            if math.isfinite(val):
                out[metric] = val
                out[f"{metric}_source_col"] = col
                break
    return out


def aggregate_history_physics(row: pd.Series) -> dict:
    names = [
        "advective_heat_source_c_per_day_mean",
        "advective_heat_source_c_per_day_max",
        "advective_exchange_fraction_per_day",
        "advective_heat_source_active_mean",
        "nn_kd_multiplier_mean",
        "nn_kd_multiplier_p50",
        "nn_kd_multiplier_p95",
        "nn_kd_multiplier_saturation_fraction",
        "background_nn_kz_mean",
        "background_nn_kz_deep_mean",
        "turbulent_nn_kz_mean",
        "turbulent_nn_kz_deep_mean",
        "gated_turbulent_nn_kz_mean",
        "gated_turbulent_nn_kz_deep_mean",
        "residual_regularization_loss",
        "segment_rollout_residual_regularization_loss",
        "heat_content_transition_loss",
        "heat_content_transition_weighted_loss",
        "effective_heat_tendency_mean_wm2",
        "surface_flux_bias_mean_wm2",
    ]
    return {name: value_from_row(row, name) for name in names}


def summarize_by_group(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, lake_type), group in joined.groupby(["role", "lake_type"], dropna=False):
        rows.append({
            "role": role,
            "lake_type": lake_type,
            "lake_years": int(len(group)),
            "hydrology_nonzero_rate": float((group["net_inflow_nonzero_days"] > 0).mean()),
            "advective_active_rate": float(group["simple_advective_active_candidate"].astype(bool).mean()),
            "exchange_fraction_p95_mean": float(pd.to_numeric(group["exchange_fraction_p95_per_day"], errors="coerce").mean()),
            "advective_heat_proxy_abs_mean_c_day": float(pd.to_numeric(group["advective_heat_proxy_abs_mean_c_day"], errors="coerce").mean()),
            "point_rmse_mean": float(pd.to_numeric(group["point_rmse"], errors="coerce").mean()),
            "free_roll_rmse_mean": float(pd.to_numeric(group["free_roll_rmse"], errors="coerce").mean()),
            "free_roll_bias_mean": float(pd.to_numeric(group["free_roll_bias"], errors="coerce").mean()),
            "free_roll_rmse_gt25m_mean": float(pd.to_numeric(group["free_roll_rmse_gt25m"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values(["role", "lake_type"]).reset_index(drop=True)


def correlation_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = joined[joined["lake_type"] == "reservoir"].copy()
    xs = [
        "positive_net_inflow_fraction",
        "exchange_fraction_p95_per_day",
        "advective_heat_proxy_abs_mean_c_day",
        "net_inflow_p95_m3_s",
    ]
    ys = ["point_rmse", "free_roll_rmse", "free_roll_bias", "free_roll_rmse_gt25m"]
    for x in xs:
        for y in ys:
            valid = subset[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) >= 3 and valid[x].std() > 0.0 and valid[y].std() > 0.0:
                corr = float(valid[x].corr(valid[y]))
            else:
                corr = np.nan
            rows.append({"subset": "reservoir", "x": x, "y": y, "n": int(len(valid)), "pearson_corr": corr})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if not math.isfinite(float(val)) else f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, joined: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame, best_epoch: int, final_epoch: int, physics: dict) -> Path:
    report = output_dir / "R35_reservoir_advective_source_separation_report.md"
    role_counts = Counter(joined["role"])
    reservoir = joined[joined["lake_type"] == "reservoir"].copy()
    reservoir_active = int(reservoir["simple_advective_active_candidate"].astype(bool).sum())
    reservoir_total = int(len(reservoir))
    valid_corr = corr.dropna(subset=["pearson_corr"])
    strongest = valid_corr.iloc[valid_corr["pearson_corr"].abs().argmax()].to_dict() if not valid_corr.empty else {}
    heat_proxy_mean = float(pd.to_numeric(reservoir["advective_heat_proxy_abs_mean_c_day"], errors="coerce").mean())
    exchange_p95 = float(pd.to_numeric(reservoir["exchange_fraction_p95_per_day"], errors="coerce").mean())
    advective_not_enough = (reservoir_active < reservoir_total) or (not math.isfinite(heat_proxy_mean)) or heat_proxy_mean < 0.02
    decision = "proposal_needed_for_better_reservoir_diagnostics_not_model_merge"
    if reservoir_active > 0 and math.isfinite(heat_proxy_mean) and heat_proxy_mean >= 0.02:
        decision = "advective_signal_present_but_eval_only_evidence_needs_targeted_ablation"
    lines = [
        "# R35 reservoir advective source-separation diagnostic",
        "",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        "- status: `completed_remote_eval_only_no_training`",
        "- model/loss/physics/split/_standard_inputs: unchanged",
        "- checkpoint: not loaded; R25 history and standard inputs only",
        "- heldout/stress: diagnostic-only, not used for checkpoint selection or tuning",
        f"- R25 best epoch used for metrics: {best_epoch}; final epoch: {final_epoch}",
        "",
        "## What Was Tested",
        "",
        "Whether reservoir error aligns with hydrology availability, positive net inflow, estimated exchange fraction, and the current `reservoir_simple` advective heat-source proxy.",
        "",
        "## Split Counts",
        "",
        f"- train: {role_counts.get('train', 0)}",
        f"- validation: {role_counts.get('validation', 0)}",
        f"- heldout diagnostic-only: {role_counts.get('heldout_diagnostic_only', 0)}",
        f"- stress/OOD diagnostic-only or unknown: {role_counts.get('stress_ood_diagnostic_only', 0)}",
        "",
        "## Advective Signal",
        "",
        f"- reservoir lake-years audited: {reservoir_total}",
        f"- reservoir simple-advective active candidates: {reservoir_active}/{reservoir_total}",
        f"- reservoir mean p95 exchange fraction per day: {exchange_p95:.4g}",
        f"- reservoir mean abs advective heat proxy: {heat_proxy_mean:.4g} C/day",
        f"- aggregate R25 advective active mean: {physics.get('advective_heat_source_active_mean', np.nan):.4g}",
        f"- aggregate R25 advective heat source mean/max: {physics.get('advective_heat_source_c_per_day_mean', np.nan):.4g} / {physics.get('advective_heat_source_c_per_day_max', np.nan):.4g} C/day",
        "",
        "## Role / Lake-Type Summary",
        "",
        md_table(summary),
        "",
        "## Reservoir Correlations",
        "",
        md_table(corr),
        "",
        "## Interpretation",
        "",
        f"- decision: `{decision}`",
        f"- advective_not_enough_flag: `{str(bool(advective_not_enough)).lower()}`",
        "- If reservoir error is high where hydrology is missing or advective proxy is near zero, the current `reservoir_simple` channel is not enough evidence for a full reservoir-physics merge.",
        "- If correlations are strong, the next step is still an approval-gated ablation, not an automatic model merge.",
        "- R34/R35 together support separating reservoir hydrology diagnostics from the EOF/PCA low-dimensional thermal-state branch.",
    ]
    if strongest:
        lines.extend([
            "",
            "## Strongest Reservoir Association",
            "",
            f"- `{strongest.get('x')}` vs `{strongest.get('y')}`: r={strongest.get('pearson_corr'):.3f}, n={int(strongest.get('n', 0))}",
        ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = read_json(args.result_dir / "global_state_forecaster_split_summary.json")
    manifest = load_manifest_or_split(args.result_dir, args.manifest)
    history = read_csv(args.result_dir / "global_state_forecaster_training_history.csv")
    best, final, best_epoch, final_epoch = best_or_final_rows(history, args.result_dir / "best_by_val_rolling_metrics.json")
    lake_ids = all_lake_ids(manifest, split_summary)
    input_rows = []
    metric_rows = []
    for lake_id in lake_ids:
        role = role_from_manifest(manifest, split_summary, lake_id)
        input_rows.append(compute_lake_advective_inputs(lake_id, role, args.standard_input_root))
        metric_rows.append(extract_lake_metrics(lake_id, role, best))
    inputs = pd.DataFrame(input_rows)
    metrics = pd.DataFrame(metric_rows)
    joined = inputs.merge(metrics.drop(columns=["role"]), on="lake_id", how="left")
    physics = aggregate_history_physics(best)
    summary = summarize_by_group(joined)
    corr = correlation_table(joined)
    inputs.to_csv(args.output_dir / "R35_advective_input_coverage_by_lakeyear.csv", index=False)
    metrics.to_csv(args.output_dir / "R35_r25_metric_extract_by_lakeyear.csv", index=False)
    joined.to_csv(args.output_dir / "R35_advective_metric_join_by_lakeyear.csv", index=False)
    summary.to_csv(args.output_dir / "R35_advective_source_summary_by_role_laketype.csv", index=False)
    corr.to_csv(args.output_dir / "R35_reservoir_advective_error_correlations.csv", index=False)
    (args.output_dir / "R35_r25_best_epoch_physics_diagnostics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "final_epoch": final_epoch, **physics}, indent=2),
        encoding="utf-8",
    )
    report = write_report(args.output_dir, joined, summary, corr, best_epoch, final_epoch, physics)
    print(f"experiment_id={EXPERIMENT_ID}")
    print(f"output_dir={args.output_dir}")
    print(f"report={report}")
    print(f"lake_years={len(joined)}")
    print(f"reservoir_lake_years={(joined['lake_type'] == 'reservoir').sum()}")


if __name__ == "__main__":
    main()
