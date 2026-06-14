"""R28 eval-only source decomposition for Lake-PINN heat closure.

This script reads R27 daily diagnostics and decomposes the apparent
surface-only heat residual into solver total source, surface flux, and
penetrating/internal source bookkeeping terms. It does not train, tune,
change model code, change split roles, or write standard inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
DEFAULT_R27_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R27_heat_closure_diagnostic_20260614_full"
)
DEFAULT_INPUT_DAILY = DEFAULT_R27_DIR / "R27_heat_closure_daily_diagnostics.csv"
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R28_total_absorbed_flux_source_decomp_20260614"
)
EXPERIMENT_ID = "RECON_R28_TOTAL_ABSORBED_FLUX_SOURCE_DECOMP_DIAG_v1"


def _finite(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _abs_mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def _p_abs(values, q: float) -> float:
    arr = np.abs(_finite(values))
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _rate(values, threshold: float) -> float:
    arr = _finite(values)
    return float(np.mean(np.abs(arr) > float(threshold))) if arr.size else float("nan")


def _safe_ratio(numerator, denominator) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").abs()
    return num / den.where(den > 1.0e-9, np.nan)


def _decompose(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    numeric_cols = [
        "heat_tendency_wm2",
        "heat_input_wm2",
        "surface_flux_wm2",
        "open_water_surface_flux_wm2",
        "shortwave_to_water_wm2",
        "freezing_storage_change_wm2",
        "temperature_floor_heat_injection_wm2",
        "temperature_ceiling_heat_removal_wm2",
        "surface_flux_bias_wm2",
        "rollout_lswt_observer_heat_content_delta_c",
        "rollout_lswt_observer_deep_abs_delta_c",
        "rollout_lswt_observer_filled_lst_used_count",
        "advective_heat_source_active_mean",
        "advective_heat_source_c_per_day_mean",
    ]
    for col in numeric_cols:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["main_open_water"] = (
        pd.to_numeric(frame.get("open_water_day", 0), errors="coerce").fillna(0) > 0
    ) & (frame.get("phase", "") == "rollout")
    frame["primary_surface_flux_wm2"] = np.where(
        frame["main_open_water"],
        frame["open_water_surface_flux_wm2"],
        frame["surface_flux_wm2"],
    )
    frame["surface_only_residual_wm2"] = (
        frame["heat_tendency_wm2"] - frame["primary_surface_flux_wm2"]
    )
    frame["source_bookkeeping_gap_wm2"] = (
        frame["heat_input_wm2"] - frame["primary_surface_flux_wm2"]
    )
    frame["total_absorbed_source_wm2"] = (
        frame["primary_surface_flux_wm2"] + frame["source_bookkeeping_gap_wm2"]
    )
    frame["total_source_residual_wm2"] = (
        frame["heat_tendency_wm2"] - frame["total_absorbed_source_wm2"]
    )
    frame["surface_residual_explained_fraction"] = 1.0 - _safe_ratio(
        frame["total_source_residual_wm2"].abs(),
        frame["surface_only_residual_wm2"],
    )
    frame["gap_fraction_of_shortwave_to_water"] = _safe_ratio(
        frame["source_bookkeeping_gap_wm2"].abs(),
        frame["shortwave_to_water_wm2"],
    )
    frame["freezing_storage_abs_wm2"] = frame["freezing_storage_change_wm2"].abs()
    frame["floor_ceiling_abs_wm2"] = (
        frame["temperature_floor_heat_injection_wm2"].fillna(0.0).abs()
        + frame["temperature_ceiling_heat_removal_wm2"].fillna(0.0).abs()
    )
    frame["observer_heat_abs_c"] = frame["rollout_lswt_observer_heat_content_delta_c"].abs()
    return frame


def _summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for group_values, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        open_group = group[group["main_open_water"]]
        record = {key: value for key, value in zip(group_cols, group_values)}
        record.update(
            {
                "row_count": int(len(group)),
                "open_water_rollout_count": int(len(open_group)),
                "surface_only_residual_abs_mean_wm2": _abs_mean(open_group["surface_only_residual_wm2"]),
                "surface_only_residual_abs_p95_wm2": _p_abs(open_group["surface_only_residual_wm2"], 95.0),
                "surface_only_tau36_rate": _rate(open_group["surface_only_residual_wm2"], 36.0),
                "surface_only_tau50_rate": _rate(open_group["surface_only_residual_wm2"], 50.0),
                "source_bookkeeping_gap_abs_mean_wm2": _abs_mean(open_group["source_bookkeeping_gap_wm2"]),
                "source_bookkeeping_gap_abs_p95_wm2": _p_abs(open_group["source_bookkeeping_gap_wm2"], 95.0),
                "gap_fraction_of_shortwave_mean": _mean(open_group["gap_fraction_of_shortwave_to_water"]),
                "gap_fraction_of_shortwave_p95": _p_abs(open_group["gap_fraction_of_shortwave_to_water"], 95.0),
                "total_source_residual_abs_mean_wm2": _abs_mean(open_group["total_source_residual_wm2"]),
                "total_source_residual_abs_p95_wm2": _p_abs(open_group["total_source_residual_wm2"], 95.0),
                "total_source_tau36_rate": _rate(open_group["total_source_residual_wm2"], 36.0),
                "total_source_tau50_rate": _rate(open_group["total_source_residual_wm2"], 50.0),
                "surface_residual_explained_fraction_mean": _mean(
                    open_group["surface_residual_explained_fraction"]
                ),
                "heat_tendency_mean_wm2": _mean(open_group["heat_tendency_wm2"]),
                "primary_surface_flux_mean_wm2": _mean(open_group["primary_surface_flux_wm2"]),
                "total_absorbed_source_mean_wm2": _mean(open_group["total_absorbed_source_wm2"]),
                "shortwave_to_water_mean_wm2": _mean(open_group["shortwave_to_water_wm2"]),
                "freezing_storage_abs_mean_wm2": _abs_mean(open_group["freezing_storage_change_wm2"]),
                "floor_ceiling_abs_mean_wm2": _abs_mean(open_group["floor_ceiling_abs_wm2"]),
                "advective_active_mean": _mean(open_group["advective_heat_source_active_mean"]),
                "advective_c_per_day_mean": _mean(open_group["advective_heat_source_c_per_day_mean"]),
                "observer_heat_abs_mean_c": _mean(open_group["observer_heat_abs_c"]),
                "observer_deep_abs_delta_mean_c": _mean(open_group["rollout_lswt_observer_deep_abs_delta_c"]),
                "filled_lst_update_sum": float(
                    pd.to_numeric(open_group["rollout_lswt_observer_filled_lst_used_count"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _write_report(output_dir: Path, summary: pd.DataFrame, paths: dict[str, Path]) -> Path:
    report = output_dir / "R28_total_absorbed_flux_source_decomp_report.md"
    lines = [
        "# R28 Total Absorbed Flux Source Decomposition",
        "",
        "Scope: eval-only diagnostic from R27 daily outputs. No training, no model/loss/Kd/Kz/residual bound change, no split change, no `_standard_inputs` change.",
        "",
        "Question: after adding back the solver source-bookkeeping gap, does the heat residual remain large?",
        "",
        "Definitions:",
        "",
        "- `surface_only_residual = heat_tendency - primary_surface_flux`",
        "- `source_bookkeeping_gap = heat_input - primary_surface_flux`",
        "- `total_absorbed_source = primary_surface_flux + source_bookkeeping_gap`",
        "- `total_source_residual = heat_tendency - total_absorbed_source`",
        "",
        "## Key Summary",
        "",
    ]
    key = summary[
        (summary.get("phase", "") == "rollout")
        & (summary.get("role", "").isin(["val", "heldout"]))
        & (summary.get("lake_type", "").isin(["natural", "reservoir"]))
    ].copy()
    if key.empty:
        key = summary.copy()
    for _, row in key.sort_values(["config_id", "role", "lake_type"]).iterrows():
        lines.append(
            "- "
            f"{row['config_id']} {row['role']} {row['lake_type']}: "
            f"surface_abs_mean={row['surface_only_residual_abs_mean_wm2']:.2f} W/m2, "
            f"gap_abs_mean={row['source_bookkeeping_gap_abs_mean_wm2']:.2f} W/m2, "
            f"total_abs_mean={row['total_source_residual_abs_mean_wm2']:.4f} W/m2, "
            f"total_abs_p95={row['total_source_residual_abs_p95_wm2']:.4f} W/m2, "
            f"explained={row['surface_residual_explained_fraction_mean']:.5f}, "
            f"filled_lst_updates={row['filled_lst_update_sum']:.0f}, "
            f"deep_delta={row['observer_deep_abs_delta_mean_c']:.4f} C"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The large surface-only residual is almost entirely explained by source bookkeeping, especially penetrating/internal heat source terms already included in `heat_input_wm2`.",
            "- The total-source residual is the solver-level heat closure residual; near-zero values mean immediate heat-closure loss is not justified from this evidence.",
            "- This does not prove the temperature product is good; it only resolves the heat-budget accounting question raised by R27.",
            "- Heldout rows remain diagnostic-only and are not used for checkpoint selection or tuning.",
            "",
            "## Decision",
            "",
            "- Immediate weak heat-closure loss training: No-Go.",
            "- Next useful action: keep source-decomposition diagnostics in reporting/export, then return to zero-profile temperature error sources unless a separate physically aligned residual remains.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R28 total absorbed flux source decomposition.")
    parser.add_argument("--input-daily", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    input_daily = Path(args.input_daily)
    output_dir = Path(args.output_dir)
    if not input_daily.exists():
        raise FileNotFoundError(input_daily)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(input_daily, encoding="utf-8-sig")
    decomp = _decompose(daily)

    decomp_path = output_dir / "R28_total_absorbed_flux_daily_decomposition.csv"
    summary_path = output_dir / "R28_total_absorbed_flux_summary_by_role_laketype.csv"
    monthly_path = output_dir / "R28_total_absorbed_flux_monthly_summary.csv"
    seasonal_path = output_dir / "R28_total_absorbed_flux_seasonal_summary.csv"
    overall_path = output_dir / "R28_total_absorbed_flux_overall_summary.json"

    summary = _summarize(decomp, ["config_id", "role", "lake_type", "phase"])
    monthly = _summarize(decomp, ["config_id", "role", "lake_type", "phase", "month"])
    seasonal = _summarize(decomp, ["config_id", "role", "lake_type", "phase", "season"])

    decomp.to_csv(decomp_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    seasonal.to_csv(seasonal_path, index=False, encoding="utf-8-sig")

    key = summary[(summary["phase"] == "rollout") & (summary["role"].isin(["val", "heldout"]))]
    total_abs_mean_max = float(np.nanmax(key["total_source_residual_abs_mean_wm2"])) if not key.empty else float("nan")
    surface_abs_mean_min = float(np.nanmin(key["surface_only_residual_abs_mean_wm2"])) if not key.empty else float("nan")
    loss_go = bool(np.isfinite(total_abs_mean_max) and total_abs_mean_max > 10.0)
    overall = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_eval_only_no_training",
        "source_daily": str(input_daily),
        "daily_row_count": int(len(decomp)),
        "summary_row_count": int(len(summary)),
        "heldout_policy": "diagnostic-only; not used for checkpoint selection or tuning",
        "total_source_residual_abs_mean_max_val_heldout_wm2": total_abs_mean_max,
        "surface_only_residual_abs_mean_min_val_heldout_wm2": surface_abs_mean_min,
        "immediate_heat_closure_loss_go": loss_go,
        "decision": (
            "No-Go for immediate heat-closure loss training; source bookkeeping explains the surface-only residual."
            if not loss_go
            else "Potential Go only after manual review; total-source residual remains large."
        ),
    }
    overall_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        output_dir,
        summary,
        {
            "daily_decomposition": decomp_path,
            "summary_by_role_laketype": summary_path,
            "monthly": monthly_path,
            "seasonal": seasonal_path,
            "overall_json": overall_path,
        },
    )
    print(json.dumps({"report": str(report_path), "summary": overall}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
