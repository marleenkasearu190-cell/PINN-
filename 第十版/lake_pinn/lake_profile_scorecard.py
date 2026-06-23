"""
Lake profile scorecard v1.

This script scores one or more predicted temperature-profile CSV files against
an observed/reference profile CSV using a layered selection template:

1. Physical bottom line
2. Key seasonal processes
3. Numeric accuracy
4. Stability
5. Visual heatmap impression (manual, optional)

Heatmap-shape metrics are part of the physical bottom line so visually
unphysical strips or spikes cannot win on RMSE alone.

Supported input formats
-----------------------
Prediction CSV (long format):
    Date,Depth_m,Temperature_C

Reference CSV:
    Either the same long format, or wide format like:
    Date,Temp_0m,Temp_1m,...,Temp_13m

Example
-------
python lake_profile_scorecard.py ^
  --truth "<local-data>\\mohonk\\validation\\MohonkLake_temp_2017.csv" ^
  --pred "<local-outputs>\\run7\\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" ^
  --label "run7" ^
  --pred "<local-outputs>\\11d_run1\\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" ^
  --label "11d_run1" ^
  --out-dir "<local-outputs>\\score_outputs"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


SURFACE_BAND_MAX_M = 3.0
THERMO_DELTA_T_MIN_C = 2.0
SUMMER_MONTHS = (7, 8)
AUTUMN_START = (9, 15)
AUTUMN_END = (11, 30)
WINTER_MONTHS = (1, 2)


@dataclass
class ScoreThresholds:
    overall_rmse_good: float = 0.9
    overall_rmse_bad: float = 2.3
    overall_mae_good: float = 0.6
    overall_mae_bad: float = 1.6
    abs_bias_good: float = 0.10
    abs_bias_bad: float = 0.60
    surface_rmse_good: float = 0.8
    surface_rmse_bad: float = 2.5
    thermocline_rmse_good: float = 1.0
    thermocline_rmse_bad: float = 3.0
    may_warming_good: float = 0.5
    may_warming_bad: float = 2.5
    july_surface_good: float = 0.6
    july_surface_bad: float = 2.5
    thermocline_depth_good: float = 0.8
    thermocline_depth_bad: float = 3.0
    thermocline_thickness_good: float = 0.8
    thermocline_thickness_bad: float = 3.0
    autumn_tmix_good_days: float = 5.0
    autumn_tmix_bad_days: float = 20.0
    winter_inverse_good: float = 0.3
    winter_inverse_bad: float = 1.5
    winter_surface_good: float = 0.3
    winter_surface_bad: float = 1.5
    thermocline_sharpness_good: float = 0.15
    thermocline_sharpness_bad: float = 0.60
    heat_distribution_good: float = 0.3
    heat_distribution_bad: float = 1.5
    smoothness_good: float = 0.10
    smoothness_bad: float = 0.60
    stability_sigma_good: float = 1.0
    stability_sigma_bad: float = 6.0
    reload_mae_good: float = 0.02
    reload_mae_bad: float = 0.20
    visual_score_good: float = 4.5
    visual_score_bad: float = 2.0


def s_down(value: float, good: float, bad: float, weight: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if bad <= good:
        return float(weight)
    scaled = (bad - float(value)) / (bad - good)
    return float(weight) * float(np.clip(scaled, 0.0, 1.0))


def normalize_lake_type(lake_type: Optional[str]) -> str:
    """Normalize lake-type aliases used by metadata and command-line calls."""
    if lake_type is None:
        return "universal"
    text = str(lake_type).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "universal", "generic", "general", "cross_lake", "multi_lake"}:
        return "universal"
    if text in {"auto", "unknown", "none"}:
        return "auto"
    if text in {"warm", "warm_deep", "warm_deep_monomictic", "monomictic", "kinneret"}:
        return "warm_deep_monomictic"
    if text in {"cold", "cold_dimictic", "dimictic", "temperate_dimictic", "mohonk"}:
        return "cold_dimictic"
    return text


def infer_lake_type_from_truth(truth: pd.DataFrame, eff_depth: Optional[float] = None) -> str:
    """Infer whether the scorecard should use cold-dimictic or warm-deep rules."""
    max_depth = float(eff_depth) if eff_depth is not None and np.isfinite(eff_depth) else float(truth["Depth_m"].max())
    winter = truth[truth["Month"].isin(WINTER_MONTHS)]
    winter_mean = float(winter["Temperature_C"].mean()) if len(winter) else np.nan
    if max_depth >= 25.0 and np.isfinite(winter_mean) and winter_mean >= 8.0:
        return "warm_deep_monomictic"
    return "cold_dimictic"


def resolve_lake_type(lake_type: Optional[str], truth: Optional[pd.DataFrame] = None, eff_depth: Optional[float] = None) -> str:
    normalized = normalize_lake_type(lake_type)
    if normalized == "universal":
        return "universal"
    if normalized != "auto":
        return normalized
    if truth is not None:
        return infer_lake_type_from_truth(truth, eff_depth=eff_depth)
    return "cold_dimictic"


def arg_value(args: argparse.Namespace, name: str, default):
    return getattr(args, name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score lake temperature profile predictions.")
    parser.add_argument("--truth", required=True, help="Observed/reference profile CSV.")
    parser.add_argument("--pred", action="append", required=True, help="Prediction CSV (repeatable).")
    parser.add_argument("--label", action="append", default=None, help="Label for each prediction CSV.")
    parser.add_argument("--out-dir", default=None, help="Directory to write score outputs.")
    parser.add_argument("--report-name", default="scorecard_report.png", help="PNG report filename written to --out-dir.")
    parser.add_argument("--write-csv", action="store_true", help="Also write detailed scorecard CSV tables for debugging.")
    parser.add_argument("--lake-type", default="universal", choices=["universal", "auto", "cold_dimictic", "warm_deep_monomictic"], help="Physical scorecard family. universal keeps only cross-lake physical vetoes; lake-type modes add seasonal-process vetoes.")
    parser.add_argument("--thermocline-delta-min-c", type=float, default=THERMO_DELTA_T_MIN_C)
    parser.add_argument("--surface-band-max-m", type=float, default=SURFACE_BAND_MAX_M)
    parser.add_argument("--winter-inverse-min-c", type=float, default=0.8)
    parser.add_argument("--winter-inverse-frac-min", type=float, default=0.60)
    parser.add_argument("--summer-strat-min-c", type=float, default=4.0)
    parser.add_argument("--summer-strat-frac-min", type=float, default=0.60)
    parser.add_argument("--summer-delta-col-threshold-c", type=float, default=3.0)
    parser.add_argument("--tmix-abs-max-days", type=float, default=15.0)
    parser.add_argument("--monthly-bias-max-c", type=float, default=1.5)
    parser.add_argument("--annual-bias-max-c", type=float, default=0.5)
    parser.add_argument("--thermocline-depth-max-m", type=float, default=2.5)
    parser.add_argument("--thermocline-thickness-max-m", type=float, default=2.0)
    parser.add_argument("--deep-julsep-rmse-max-c", type=float, default=1.5)
    parser.add_argument("--mld-threshold-c", type=float, default=1.0)
    parser.add_argument("--mix-delta-col-max-c", type=float, default=1.0)
    parser.add_argument("--mix-stdz-max-c", type=float, default=0.8)
    parser.add_argument("--mix-consecutive-days", type=int, default=7)
    parser.add_argument("--min-physical-temp-c", type=float, default=-0.5, help="Legacy single-point minimum threshold kept for reporting compatibility.")
    parser.add_argument("--extreme-low-temp-c", type=float, default=-5.0, help="Absolute hard lower bound for any predicted water temperature.")
    parser.add_argument("--strong-low-temp-c", type=float, default=-3.0, help="Strong low-temperature anomaly threshold used with a fractional tolerance.")
    parser.add_argument("--max-strong-low-temp-frac", type=float, default=0.001, help="Maximum fraction of grid cells allowed below --strong-low-temp-c.")
    parser.add_argument("--low-temp-c", type=float, default=-1.0, help="Mild low-temperature anomaly threshold used with a fractional tolerance.")
    parser.add_argument("--max-low-temp-frac", type=float, default=0.01, help="Maximum fraction of grid cells allowed below --low-temp-c.")
    parser.add_argument("--max-physical-temp-c", type=float, default=32.0)
    parser.add_argument("--max-surface-jump-c-per-day", type=float, default=4.0)
    parser.add_argument("--max-surface-band-jump-c-per-day", type=float, default=2.5)
    parser.add_argument("--max-column-jump-c-per-day", type=float, default=4.0)
    parser.add_argument("--max-grad-p995-c-per-m", type=float, default=6.0)
    parser.add_argument("--max-grad-extreme-c-per-m", type=float, default=8.0)
    parser.add_argument("--max-april-surface-error-c", type=float, default=3.0)
    parser.add_argument("--max-april-surface-jump-c-per-day", type=float, default=1.5)
    parser.add_argument("--density-inversion-drop-kgm3", type=float, default=0.02)
    parser.add_argument("--max-density-unstable-layer-frac", type=float, default=0.30)
    parser.add_argument("--max-density-unstable-days", type=int, default=7)
    parser.add_argument("--warm-deep-winter-rmse-max-c", type=float, default=3.0)
    parser.add_argument("--warm-deep-winter-bias-max-c", type=float, default=2.0)
    parser.add_argument("--warm-deep-min-temp-c", type=float, default=6.0)
    parser.add_argument("--warm-deep-thermocline-depth-max-m", type=float, default=5.0)
    parser.add_argument("--warm-deep-thermocline-thickness-max-m", type=float, default=5.0)
    parser.add_argument("--warm-deep-deep-julsep-rmse-max-c", type=float, default=2.5)
    parser.add_argument("--warm-deep-autumn-final-extra-c", type=float, default=4.0)
    parser.add_argument("--warm-deep-autumn-gap-reduction-frac", type=float, default=0.25)
    parser.add_argument("--seed-score-std", type=float, default=np.nan, help="Optional total-score std over seed reruns for stability scoring.")
    parser.add_argument("--reload-mae", type=float, default=np.nan, help="Optional train-vs-predict MAE for stability scoring.")
    parser.add_argument("--visual-score", action="append", default=None, help="Optional manual heatmap visual score for each prediction (repeatable, recommended 0-5).")
    parser.add_argument("--visual-note", action="append", default=None, help="Optional manual visual note for each prediction (repeatable).")
    return parser.parse_args()


def load_profile_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"])

    if {"Depth_m", "Temperature_C"}.issubset(df.columns):
        out = df[["Date", "Depth_m", "Temperature_C"]].copy()
    else:
        temp_cols = [c for c in df.columns if re.fullmatch(r"Temp_[0-9]+m", c)]
        if not temp_cols:
            raise ValueError(f"Unsupported profile format in {path}")
        out = df.melt(id_vars=["Date"], value_vars=temp_cols, var_name="DepthCol", value_name="Temperature_C")
        out["Depth_m"] = out["DepthCol"].str.extract(r"Temp_([0-9]+)m").astype(float)
        out = out.drop(columns=["DepthCol"])

    out["Depth_m"] = pd.to_numeric(out["Depth_m"], errors="coerce")
    out["Temperature_C"] = pd.to_numeric(out["Temperature_C"], errors="coerce")
    out = out.dropna(subset=["Date", "Depth_m", "Temperature_C"]).copy()
    out["Month"] = out["Date"].dt.month.astype(int)
    return out.sort_values(["Date", "Depth_m"]).reset_index(drop=True)


def build_common_grid(truth: pd.DataFrame, pred: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex, float]:
    common_dates = pd.DatetimeIndex(sorted(set(truth["Date"]).intersection(set(pred["Date"]))))
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between truth and prediction.")
    eff_depth = math.floor(min(float(truth["Depth_m"].max()), float(pred["Depth_m"].max())))
    if eff_depth < 1:
        raise ValueError("Effective common depth is too shallow.")
    common_depths = np.arange(0.0, float(eff_depth) + 1.0, 1.0, dtype=np.float64)
    return common_depths, common_dates, float(eff_depth)


def interpolate_profile(group: pd.DataFrame, target_depths: np.ndarray) -> np.ndarray:
    g = group.sort_values("Depth_m")
    x = g["Depth_m"].to_numpy(dtype=np.float64)
    y = g["Temperature_C"].to_numpy(dtype=np.float64)
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]
    if len(x_unique) == 1:
        return np.full(len(target_depths), y_unique[0], dtype=np.float64)
    return np.interp(target_depths, x_unique, y_unique)


def build_aligned_cube(truth: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, float]:
    depths, dates, eff_depth = build_common_grid(truth, pred)
    truth_by_date = {d: g for d, g in truth.groupby("Date")}
    pred_by_date = {d: g for d, g in pred.groupby("Date")}
    rows: List[pd.DataFrame] = []
    for date in dates:
        truth_vec = interpolate_profile(truth_by_date[date], depths)
        pred_vec = interpolate_profile(pred_by_date[date], depths)
        frame = pd.DataFrame(
            {
                "Date": date,
                "Month": int(date.month),
                "DOY": int(date.dayofyear),
                "Depth_m": depths,
                "Tobs": truth_vec,
                "That": pred_vec,
            }
        )
        frame["diff"] = frame["That"] - frame["Tobs"]
        rows.append(frame)
    aligned = pd.concat(rows, ignore_index=True)
    return aligned, depths, dates, eff_depth


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def mae(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.abs(values)))


def bias(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values))


def subset_metrics(df: pd.DataFrame) -> Dict[str, float]:
    diffs = df["diff"].to_numpy(dtype=np.float64)
    return {"rmse": rmse(diffs), "mae": mae(diffs), "bias": bias(diffs)}


def first_depth_meeting(depths: np.ndarray, values: np.ndarray, threshold: float, fallback: float) -> float:
    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return float(fallback)
    return float(depths[int(idx[0])])


def rolling_first_consecutive(dates: pd.DatetimeIndex, condition: np.ndarray, window: int) -> Optional[pd.Timestamp]:
    if len(condition) < window:
        return None
    cond = np.asarray(condition, dtype=bool)
    for i in range(0, len(cond) - window + 1):
        if np.all(cond[i : i + window]):
            expected = pd.date_range(dates[i], periods=window, freq="D")
            if list(expected) == list(dates[i : i + window]):
                return pd.Timestamp(dates[i])
    return None


def compute_daily_features(
    aligned: pd.DataFrame,
    depths: np.ndarray,
    dates: pd.DatetimeIndex,
    eff_depth: float,
    surface_band_max_m: float,
    thermocline_delta_min_c: float,
    mld_threshold_c: float,
    mix_delta_col_max_c: float,
    mix_stdz_max_c: float,
    mix_consecutive_days: int,
) -> pd.DataFrame:
    surf_mask = depths <= 1.0
    deep_mask = depths >= (0.7 * eff_depth)
    rows = []
    for date in dates:
        day = aligned[aligned["Date"] == date].sort_values("Depth_m")
        tobs = day["Tobs"].to_numpy(dtype=np.float64)
        that = day["That"].to_numpy(dtype=np.float64)
        tsurf_obs = float(np.mean(tobs[surf_mask]))
        tsurf_hat = float(np.mean(that[surf_mask]))
        tdeep_obs = float(np.mean(tobs[deep_mask]))
        tdeep_hat = float(np.mean(that[deep_mask]))
        delta_obs = tsurf_obs - tdeep_obs
        delta_hat = tsurf_hat - tdeep_hat
        mld_obs = first_depth_meeting(depths, tsurf_obs - tobs, mld_threshold_c, eff_depth)
        mld_hat = first_depth_meeting(depths, tsurf_hat - that, mld_threshold_c, eff_depth)

        z10_obs = z90_obs = z10_hat = z90_hat = np.nan
        zth_obs = hth_obs = zth_hat = hth_hat = np.nan
        if delta_obs >= thermocline_delta_min_c:
            z10_obs = first_depth_meeting(depths, tsurf_obs - tobs, 0.1 * delta_obs, eff_depth)
            z90_obs = first_depth_meeting(depths, tsurf_obs - tobs, 0.9 * delta_obs, eff_depth)
            zth_obs = 0.5 * (z10_obs + z90_obs)
            hth_obs = z90_obs - z10_obs
        if delta_hat >= thermocline_delta_min_c:
            z10_hat = first_depth_meeting(depths, tsurf_hat - that, 0.1 * delta_hat, eff_depth)
            z90_hat = first_depth_meeting(depths, tsurf_hat - that, 0.9 * delta_hat, eff_depth)
            zth_hat = 0.5 * (z10_hat + z90_hat)
            hth_hat = z90_hat - z10_hat

        grad_obs = np.abs(np.diff(tobs) / np.diff(depths))
        grad_hat = np.abs(np.diff(that) / np.diff(depths))
        rows.append(
            {
                "Date": date,
                "Month": int(date.month),
                "DOY": int(date.dayofyear),
                "Tsurf_obs": tsurf_obs,
                "Tsurf_hat": tsurf_hat,
                "Tdeep_obs": tdeep_obs,
                "Tdeep_hat": tdeep_hat,
                "DeltaTcol_obs": delta_obs,
                "DeltaTcol_hat": delta_hat,
                "MLD_obs": mld_obs,
                "MLD_hat": mld_hat,
                "z10_obs": z10_obs,
                "z90_obs": z90_obs,
                "zth_obs": zth_obs,
                "hth_obs": hth_obs,
                "z10_hat": z10_hat,
                "z90_hat": z90_hat,
                "zth_hat": zth_hat,
                "hth_hat": hth_hat,
                "stdz_obs": float(np.std(tobs)),
                "stdz_hat": float(np.std(that)),
                "G_obs": float(np.max(grad_obs)) if len(grad_obs) else np.nan,
                "G_hat": float(np.max(grad_hat)) if len(grad_hat) else np.nan,
                "bottom_obs": float(tobs[-1]),
                "bottom_hat": float(that[-1]),
                "mean_obs": float(np.mean(tobs)),
                "mean_hat": float(np.mean(that)),
            }
        )
    daily = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    autumn_mask = (
        ((daily["Month"] == AUTUMN_START[0]) & (daily["Date"].dt.day >= AUTUMN_START[1]))
        | ((daily["Month"] > AUTUMN_START[0]) & (daily["Month"] < AUTUMN_END[0]))
        | ((daily["Month"] == AUTUMN_END[0]) & (daily["Date"].dt.day <= AUTUMN_END[1]))
    )
    autumn = daily.loc[autumn_mask].copy()
    cond_obs = (np.abs(autumn["DeltaTcol_obs"]) < mix_delta_col_max_c) & (autumn["stdz_obs"] < mix_stdz_max_c)
    cond_hat = (np.abs(autumn["DeltaTcol_hat"]) < mix_delta_col_max_c) & (autumn["stdz_hat"] < mix_stdz_max_c)
    tmix_obs = rolling_first_consecutive(pd.DatetimeIndex(autumn["Date"]), cond_obs.to_numpy(), mix_consecutive_days)
    tmix_hat = rolling_first_consecutive(pd.DatetimeIndex(autumn["Date"]), cond_hat.to_numpy(), mix_consecutive_days)
    daily["tmix_obs"] = tmix_obs
    daily["tmix_hat"] = tmix_hat
    return daily


def compute_tv_metrics(aligned: pd.DataFrame, depths: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, float]:
    pivot_obs = aligned.pivot(index="Depth_m", columns="Date", values="Tobs").reindex(index=depths, columns=dates)
    pivot_hat = aligned.pivot(index="Depth_m", columns="Date", values="That").reindex(index=depths, columns=dates)
    obs = pivot_obs.to_numpy(dtype=np.float64)
    hat = pivot_hat.to_numpy(dtype=np.float64)
    tvt_obs = float(np.mean(np.abs(np.diff(obs, axis=1))))
    tvt_hat = float(np.mean(np.abs(np.diff(hat, axis=1))))
    tvz_obs = float(np.mean(np.abs(np.diff(obs, axis=0))))
    tvz_hat = float(np.mean(np.abs(np.diff(hat, axis=0))))
    return {
        "TVt_obs": tvt_obs,
        "TVt_hat": tvt_hat,
        "TVz_obs": tvz_obs,
        "TVz_hat": tvz_hat,
    }


def freshwater_density_kgm3(temp_c: np.ndarray) -> np.ndarray:
    """Freshwater density approximation with maximum density near 3.98 C."""
    temp = np.asarray(temp_c, dtype=np.float64)
    return 1000.0 * (
        1.0
        - ((temp + 288.9414) / (508929.2 * (temp + 68.12963)))
        * np.square(temp - 3.9863)
    )


def _finite_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _safe_nanmin(values, default=np.nan) -> float:
    finite = _finite_array(values)
    return float(np.min(finite)) if finite.size else float(default)


def _safe_nanmax(values, default=np.nan) -> float:
    finite = _finite_array(values)
    return float(np.max(finite)) if finite.size else float(default)


def _safe_nanpercentile(values, percentile, default=np.nan) -> float:
    finite = _finite_array(values)
    return float(np.percentile(finite, percentile)) if finite.size else float(default)


def _safe_nanargmax(values) -> int:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return -1
    return int(np.argmax(np.where(finite, array, -np.inf)))


def _safe_nanmean_axis(values, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, array, 0.0), axis=axis)
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _safe_nanmax_axis(values, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    out = np.max(np.where(finite, array, -np.inf), axis=axis)
    return np.where(np.any(finite, axis=axis), out, np.nan)


def compute_heatmap_physics_diagnostics(
    aligned: pd.DataFrame,
    depths: np.ndarray,
    dates: pd.DatetimeIndex,
    surface_band_max_m: float,
    density_inversion_drop_kgm3: float,
    low_temp_c: float = -1.0,
    strong_low_temp_c: float = -3.0,
    extreme_low_temp_c: float = -5.0,
) -> Dict[str, float]:
    """Diagnostics that catch visually unphysical heatmap artifacts.

    These metrics intentionally focus on morphology rather than pointwise RMSE:
    abrupt surface strips, whole-column day jumps, impossible temperature range,
    extreme vertical gradients, and persistent density inversions.
    """
    pivot_obs = aligned.pivot(index="Date", columns="Depth_m", values="Tobs").reindex(index=dates, columns=depths)
    pivot_hat = aligned.pivot(index="Date", columns="Depth_m", values="That").reindex(index=dates, columns=depths)
    obs = pivot_obs.to_numpy(dtype=np.float64)
    hat = pivot_hat.to_numpy(dtype=np.float64)

    surface_hat = hat[:, 0]
    surface_obs = obs[:, 0]
    surface_band_mask = depths <= surface_band_max_m
    surface_band_hat = (
        _safe_nanmean_axis(hat[:, surface_band_mask], axis=1)
        if np.any(surface_band_mask)
        else surface_hat.copy()
    )

    delta_days = np.diff(pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[D]").astype("int64")).astype(np.float64)
    consecutive_mask = delta_days <= 1.5
    surface_jump_all = np.abs(np.diff(surface_hat))
    surface_band_jump_all = np.abs(np.diff(surface_band_hat))
    column_jump_all = _safe_nanmax_axis(np.abs(np.diff(hat, axis=0)), axis=1)
    surface_jump = surface_jump_all[consecutive_mask]
    surface_band_jump = surface_band_jump_all[consecutive_mask]
    column_jump = column_jump_all[consecutive_mask]

    dz = np.diff(depths)
    if len(dz):
        vertical_grad = np.abs(np.diff(hat, axis=1) / dz.reshape(1, -1))
        grad_p995 = _safe_nanpercentile(vertical_grad, 99.5)
        grad_extreme = _safe_nanmax(vertical_grad)
    else:
        grad_p995 = np.nan
        grad_extreme = np.nan

    rho = freshwater_density_kgm3(hat)
    rho_drop = -np.diff(rho, axis=1)
    weak_unstable = rho_drop > 0.005
    strong_unstable = rho_drop > density_inversion_drop_kgm3
    weak_unstable_frac = (
        np.nanmean(weak_unstable, axis=1)
        if weak_unstable.size
        else np.zeros(len(dates), dtype=np.float64)
    )
    strong_unstable_frac = (
        np.nanmean(strong_unstable, axis=1)
        if strong_unstable.size
        else np.zeros(len(dates), dtype=np.float64)
    )

    april_mask = np.asarray(dates.month == 4)
    if np.any(april_mask):
        april_surface_max_hat = _safe_nanmax(surface_hat[april_mask])
        april_surface_max_obs = _safe_nanmax(surface_obs[april_mask])
        april_surface_error = april_surface_max_hat - april_surface_max_obs
        april_jump_mask = april_mask[:-1] & april_mask[1:] & consecutive_mask
        april_surface_jump = _safe_nanmax(surface_jump_all[april_jump_mask]) if np.any(april_jump_mask) else np.nan
    else:
        april_surface_max_hat = np.nan
        april_surface_max_obs = np.nan
        april_surface_error = np.nan
        april_surface_jump = np.nan

    jump_source_indices = np.where(consecutive_mask)[0]
    max_surface_jump_pos = _safe_nanargmax(surface_jump) if surface_jump.size else -1
    max_column_jump_pos = _safe_nanargmax(column_jump) if column_jump.size else -1
    max_surface_jump_idx = int(jump_source_indices[max_surface_jump_pos]) if max_surface_jump_pos >= 0 else -1
    max_column_jump_idx = int(jump_source_indices[max_column_jump_pos]) if max_column_jump_pos >= 0 else -1

    valid_hat = hat[np.isfinite(hat)]
    if valid_hat.size:
        low_temp_fraction = float(np.mean(valid_hat < float(low_temp_c)))
        strong_low_temp_fraction = float(np.mean(valid_hat < float(strong_low_temp_c)))
        extreme_low_temp_count = int(np.sum(valid_hat < float(extreme_low_temp_c)))
    else:
        low_temp_fraction = np.nan
        strong_low_temp_fraction = np.nan
        extreme_low_temp_count = 0

    return {
        "heatmap_min_temp_c": _safe_nanmin(hat),
        "heatmap_max_temp_c": _safe_nanmax(hat),
        "heatmap_low_temp_threshold_c": float(low_temp_c),
        "heatmap_low_temp_fraction": low_temp_fraction,
        "heatmap_strong_low_temp_threshold_c": float(strong_low_temp_c),
        "heatmap_strong_low_temp_fraction": strong_low_temp_fraction,
        "heatmap_extreme_low_temp_threshold_c": float(extreme_low_temp_c),
        "heatmap_extreme_low_temp_count": extreme_low_temp_count,
        "heatmap_max_surface_jump_c_day": _safe_nanmax(surface_jump) if surface_jump.size else np.nan,
        "heatmap_max_surface_jump_date": dates[max_surface_jump_idx + 1] if max_surface_jump_idx >= 0 else pd.NaT,
        "heatmap_max_surface_band_jump_c_day": _safe_nanmax(surface_band_jump) if surface_band_jump.size else np.nan,
        "heatmap_max_column_jump_c_day": _safe_nanmax(column_jump) if column_jump.size else np.nan,
        "heatmap_max_column_jump_date": dates[max_column_jump_idx + 1] if max_column_jump_idx >= 0 else pd.NaT,
        "heatmap_grad_p995_c_m": grad_p995,
        "heatmap_grad_extreme_c_m": grad_extreme,
        "heatmap_weak_density_unstable_layer_fraction": float(np.nanmean(weak_unstable)) if weak_unstable.size else 0.0,
        "heatmap_weak_density_unstable_days": int(np.nansum(weak_unstable_frac > 0.20)) if weak_unstable_frac.size else 0,
        "heatmap_max_weak_density_unstable_layer_frac": _safe_nanmax(weak_unstable_frac) if weak_unstable_frac.size else np.nan,
        "heatmap_max_density_unstable_layer_frac": _safe_nanmax(strong_unstable_frac) if strong_unstable_frac.size else np.nan,
        "heatmap_density_unstable_days": int(np.nansum(strong_unstable_frac > 0.20)) if strong_unstable_frac.size else 0,
        "heatmap_consecutive_jump_pairs": int(np.nansum(consecutive_mask)),
        "april_surface_max_hat_c": april_surface_max_hat,
        "april_surface_max_obs_c": april_surface_max_obs,
        "april_surface_max_error_c": april_surface_error,
        "april_surface_max_jump_c_day": april_surface_jump,
    }


def make_masks(aligned: pd.DataFrame, daily: pd.DataFrame, eff_depth: float, surface_band_max_m: float) -> Dict[str, pd.Series]:
    surface_mask = aligned["Depth_m"] <= surface_band_max_m
    deep_mask = aligned["Depth_m"] >= (0.7 * eff_depth)
    jul_sep_mask = aligned["Month"].between(7, 9)
    midsummer_mask = aligned["Month"].isin(SUMMER_MONTHS)
    return {
        "surface": surface_mask,
        "deep": deep_mask,
        "jul_sep": jul_sep_mask,
        "midsummer": midsummer_mask,
        "middepth_4_8": aligned["Depth_m"].between(4.0, 8.0),
    }


def thermocline_band_mask(aligned: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    summer_daily = daily[daily["Month"].between(7, 9)][["Date", "z10_obs", "z90_obs"]].copy()
    summer_daily["zlow"] = summer_daily["z10_obs"] - 1.0
    summer_daily["zhigh"] = summer_daily["z90_obs"] + 1.0
    merged = aligned.merge(summer_daily[["Date", "zlow", "zhigh"]], on="Date", how="left")
    return merged["zlow"].notna() & merged["Depth_m"].between(merged["zlow"], merged["zhigh"])


def evaluate_vetoes(
    aligned: pd.DataFrame,
    daily: pd.DataFrame,
    depths: np.ndarray,
    dates: pd.DatetimeIndex,
    masks: Dict[str, pd.Series],
    thermo_mask: pd.Series,
    args: argparse.Namespace,
) -> Dict[str, object]:
    lake_type = resolve_lake_type(arg_value(args, "lake_type", "auto"), truth=aligned.rename(columns={"Tobs": "Temperature_C"})[["Date", "Month", "Depth_m", "Temperature_C"]], eff_depth=float(np.nanmax(depths)))
    is_universal = lake_type == "universal"
    is_warm_deep = lake_type == "warm_deep_monomictic"

    winter = daily[daily["Month"].isin(WINTER_MONTHS)].copy()
    iw_pred = float(np.median(winter["Tdeep_hat"] - winter["Tsurf_hat"])) if len(winter) else np.nan
    finv_pred = float(np.mean((winter["Tdeep_hat"] - winter["Tsurf_hat"]) > 0.5)) if len(winter) else np.nan
    winter_inverse_pass_cold = bool(
        np.isfinite(iw_pred)
        and np.isfinite(finv_pred)
        and iw_pred >= args.winter_inverse_min_c
        and finv_pred >= args.winter_inverse_frac_min
    )
    winter_profile = aligned[aligned["Month"].isin(WINTER_MONTHS)]
    winter_profile_metrics = subset_metrics(winter_profile) if len(winter_profile) else {"rmse": np.nan, "bias": np.nan}
    winter_min_hat = float(winter_profile["That"].min()) if len(winter_profile) else np.nan
    winter_warm_structure_pass = bool(
        np.isfinite(winter_profile_metrics["rmse"])
        and winter_profile_metrics["rmse"] <= arg_value(args, "warm_deep_winter_rmse_max_c", 3.0)
        and abs(winter_profile_metrics["bias"]) <= arg_value(args, "warm_deep_winter_bias_max_c", 2.0)
        and np.isfinite(winter_min_hat)
        and winter_min_hat >= arg_value(args, "warm_deep_min_temp_c", 6.0)
    )
    winter_inverse_pass = winter_warm_structure_pass if is_warm_deep else winter_inverse_pass_cold

    summer = daily[daily["Month"].isin(SUMMER_MONTHS)].copy()
    is_pred = float(np.median(summer["DeltaTcol_hat"])) if len(summer) else np.nan
    fstrat_pred = float(np.mean(summer["DeltaTcol_hat"] > args.summer_delta_col_threshold_c)) if len(summer) else np.nan
    summer_strat_pass = bool(
        np.isfinite(is_pred)
        and np.isfinite(fstrat_pred)
        and is_pred >= args.summer_strat_min_c
        and fstrat_pred >= args.summer_strat_frac_min
    )

    tmix_obs = pd.Timestamp(daily["tmix_obs"].iloc[0]) if pd.notna(daily["tmix_obs"].iloc[0]) else None
    tmix_hat = pd.Timestamp(daily["tmix_hat"].iloc[0]) if pd.notna(daily["tmix_hat"].iloc[0]) else None
    tmix_delta_days = float(abs((tmix_hat - tmix_obs).days)) if tmix_obs is not None and tmix_hat is not None else np.nan
    autumn_tmix_pass = bool(np.isfinite(tmix_delta_days) and tmix_delta_days <= args.tmix_abs_max_days)
    autumn_trend_pass = False
    autumn_trend_delta_obs = np.nan
    autumn_trend_delta_hat = np.nan
    autumn_gap_reduction_frac = np.nan
    autumn_final_gap_excess_c = np.nan
    autumn = daily[daily["Month"].isin((9, 10, 11))].copy()
    if not autumn_tmix_pass and len(autumn) >= 3:
        autumn = autumn.sort_values("Date")
        autumn_trend_delta_obs = float(autumn["DeltaTcol_obs"].iloc[-1] - autumn["DeltaTcol_obs"].iloc[0])
        autumn_trend_delta_hat = float(autumn["DeltaTcol_hat"].iloc[-1] - autumn["DeltaTcol_hat"].iloc[0])
        obs_gap_reduction = autumn_trend_delta_obs <= -1.0
        pred_gap_reduction = autumn_trend_delta_hat <= -0.6
        final_gap_reasonable = float(autumn["DeltaTcol_hat"].iloc[-1]) <= max(float(autumn["DeltaTcol_obs"].iloc[-1]) + 2.0, 3.0)
        autumn_trend_pass = bool(obs_gap_reduction and pred_gap_reduction and final_gap_reasonable)
        autumn_tmix_pass = autumn_trend_pass
        if is_warm_deep:
            obs_reduction = max(float(autumn["DeltaTcol_obs"].iloc[0] - autumn["DeltaTcol_obs"].iloc[-1]), 1e-6)
            pred_reduction = float(autumn["DeltaTcol_hat"].iloc[0] - autumn["DeltaTcol_hat"].iloc[-1])
            autumn_gap_reduction_frac = pred_reduction / obs_reduction
            autumn_final_gap_excess_c = float(autumn["DeltaTcol_hat"].iloc[-1] - autumn["DeltaTcol_obs"].iloc[-1])
            warm_gap_reduction_pass = autumn_gap_reduction_frac >= arg_value(args, "warm_deep_autumn_gap_reduction_frac", 0.25)
            warm_final_gap_pass = autumn_final_gap_excess_c <= arg_value(args, "warm_deep_autumn_final_extra_c", 4.0)
            autumn_trend_pass = bool(obs_gap_reduction and warm_gap_reduction_pass and warm_final_gap_pass)
            autumn_tmix_pass = autumn_trend_pass

    yearly_bias = subset_metrics(aligned)["bias"]
    monthly_bias = aligned.groupby("Month")["diff"].mean().reindex(range(1, 13))
    bad_months = monthly_bias.abs() > args.monthly_bias_max_c
    consecutive_bad = any(bool(bad_months.iloc[i] and bad_months.iloc[i + 1]) for i in range(len(bad_months) - 1))
    drift_pass = bool(abs(yearly_bias) <= args.annual_bias_max_c and not consecutive_bad)

    thermo_daily = daily[daily["Month"].between(7, 9)].copy()
    zth_diff = np.abs(thermo_daily["zth_hat"] - thermo_daily["zth_obs"])
    hth_diff = np.abs(thermo_daily["hth_hat"] - thermo_daily["hth_obs"])
    median_zth_error = float(np.nanmedian(zth_diff)) if len(thermo_daily) else np.nan
    median_hth_error = float(np.nanmedian(hth_diff)) if len(thermo_daily) else np.nan
    deep_rmse_julsep = subset_metrics(aligned[masks["deep"] & masks["jul_sep"]])["rmse"]
    thermocline_depth_max_m = args.thermocline_depth_max_m
    thermocline_thickness_max_m = args.thermocline_thickness_max_m
    deep_julsep_rmse_max_c = args.deep_julsep_rmse_max_c
    if is_warm_deep:
        thermocline_depth_max_m = max(
            thermocline_depth_max_m,
            arg_value(args, "warm_deep_thermocline_depth_max_m", 5.0),
        )
        thermocline_thickness_max_m = max(
            thermocline_thickness_max_m,
            arg_value(args, "warm_deep_thermocline_thickness_max_m", 5.0),
        )
        deep_julsep_rmse_max_c = max(
            deep_julsep_rmse_max_c,
            arg_value(args, "warm_deep_deep_julsep_rmse_max_c", 2.5),
        )
    thermocline_pass = bool(
        np.isfinite(median_zth_error)
        and np.isfinite(median_hth_error)
        and median_zth_error <= thermocline_depth_max_m
        and median_hth_error <= thermocline_thickness_max_m
        and deep_rmse_julsep <= deep_julsep_rmse_max_c
    )

    heatmap_metrics = compute_heatmap_physics_diagnostics(
        aligned,
        depths,
        dates,
        args.surface_band_max_m,
        args.density_inversion_drop_kgm3,
        args.low_temp_c,
        args.strong_low_temp_c,
        args.extreme_low_temp_c,
    )
    temperature_range_pass = bool(
        heatmap_metrics["heatmap_min_temp_c"] >= args.extreme_low_temp_c
        and heatmap_metrics["heatmap_strong_low_temp_fraction"] <= args.max_strong_low_temp_frac
        and heatmap_metrics["heatmap_low_temp_fraction"] <= args.max_low_temp_frac
        and heatmap_metrics["heatmap_max_temp_c"] <= args.max_physical_temp_c
    )
    jump_pairs = int(heatmap_metrics.get("heatmap_consecutive_jump_pairs", 0))
    surface_jump_pass = bool(
        jump_pairs == 0 or heatmap_metrics["heatmap_max_surface_jump_c_day"] <= args.max_surface_jump_c_per_day
    )
    surface_band_jump_pass = bool(
        jump_pairs == 0 or heatmap_metrics["heatmap_max_surface_band_jump_c_day"] <= args.max_surface_band_jump_c_per_day
    )
    column_jump_pass = bool(
        jump_pairs == 0 or heatmap_metrics["heatmap_max_column_jump_c_day"] <= args.max_column_jump_c_per_day
    )
    vertical_gradient_pass = bool(
        heatmap_metrics["heatmap_grad_p995_c_m"] <= args.max_grad_p995_c_per_m
        and heatmap_metrics["heatmap_grad_extreme_c_m"] <= args.max_grad_extreme_c_per_m
    )
    april_spike_pass = bool(
        abs(heatmap_metrics["april_surface_max_error_c"]) <= args.max_april_surface_error_c
        and (
            not np.isfinite(heatmap_metrics["april_surface_max_jump_c_day"])
            or heatmap_metrics["april_surface_max_jump_c_day"] <= args.max_april_surface_jump_c_per_day
        )
    )
    density_stability_pass = bool(
        heatmap_metrics["heatmap_max_density_unstable_layer_frac"] <= args.max_density_unstable_layer_frac
        and heatmap_metrics["heatmap_density_unstable_days"] <= args.max_density_unstable_days
    )
    heatmap_physics_checks = [
        temperature_range_pass,
        surface_jump_pass,
        surface_band_jump_pass,
        column_jump_pass,
        vertical_gradient_pass,
        density_stability_pass,
    ]
    if not is_universal:
        heatmap_physics_checks.append(april_spike_pass)
    heatmap_physics_pass = all(heatmap_physics_checks)

    universal_veto_checks = [
        ("annual_drift", drift_pass),
        ("temperature_range", temperature_range_pass),
        ("surface_jump", surface_jump_pass),
        ("surface_band_jump", surface_band_jump_pass),
        ("column_jump", column_jump_pass),
        ("vertical_gradient", vertical_gradient_pass),
        ("density_stability", density_stability_pass),
    ]
    lake_type_veto_checks = [
        ("winter_warm_structure" if is_warm_deep else "winter_inverse", winter_inverse_pass),
        ("summer_stratification", summer_strat_pass),
        ("autumn_overturn_timing", autumn_tmix_pass),
        ("thermocline_deep_structure", thermocline_pass),
        ("april_surface_spike", april_spike_pass),
    ]
    active_veto_checks = universal_veto_checks if is_universal else lake_type_veto_checks + universal_veto_checks
    all_pass = all(passed for _, passed in active_veto_checks)
    failed_checks = [label for label, passed in active_veto_checks if not passed]
    seasonal_test_dates = {
        season: int(count)
        for season, count in daily.assign(
            __season=daily["Month"].map(
                lambda month: (
                    "winter" if month in WINTER_MONTHS
                    else "spring" if month in (3, 4, 5)
                    else "summer" if month in SUMMER_MONTHS
                    else "autumn"
                )
            )
        ).groupby("__season")["Date"].nunique().to_dict().items()
    }
    seasonal_coverage_pass = all(seasonal_test_dates.get(season, 0) > 0 for season in ("winter", "spring", "summer", "autumn"))

    return {
        "lake_type": lake_type,
        "pass_all_vetoes": all_pass,
        "scorecard_v2_failed_checks": ", ".join(failed_checks) if failed_checks else "none",
        "scorecard_v2_failed_check_count": int(len(failed_checks)),
        "scorecard_v2_seasonal_coverage_pass": seasonal_coverage_pass,
        "scorecard_v2_test_winter_dates": seasonal_test_dates.get("winter", 0),
        "scorecard_v2_test_spring_dates": seasonal_test_dates.get("spring", 0),
        "scorecard_v2_test_summer_dates": seasonal_test_dates.get("summer", 0),
        "scorecard_v2_test_autumn_dates": seasonal_test_dates.get("autumn", 0),
        "winter_inverse_pass": winter_inverse_pass,
        "summer_strat_pass": summer_strat_pass,
        "autumn_tmix_pass": autumn_tmix_pass,
        "autumn_trend_pass": autumn_trend_pass,
        "autumn_trend_delta_obs_c": autumn_trend_delta_obs,
        "autumn_trend_delta_hat_c": autumn_trend_delta_hat,
        "drift_pass": drift_pass,
        "thermocline_pass": thermocline_pass,
        "heatmap_physics_pass": heatmap_physics_pass,
        "temperature_range_pass": temperature_range_pass,
        "surface_jump_pass": surface_jump_pass,
        "surface_band_jump_pass": surface_band_jump_pass,
        "column_jump_pass": column_jump_pass,
        "vertical_gradient_pass": vertical_gradient_pass,
        "april_spike_pass": april_spike_pass,
        "density_stability_pass": density_stability_pass,
        "iw_pred": iw_pred,
        "finv_pred": finv_pred,
        "winter_profile_rmse_c": winter_profile_metrics["rmse"],
        "winter_profile_bias_c": winter_profile_metrics["bias"],
        "winter_min_hat_c": winter_min_hat,
        "is_pred": is_pred,
        "fstrat_pred": fstrat_pred,
        "tmix_obs": tmix_obs,
        "tmix_hat": tmix_hat,
        "tmix_delta_days": tmix_delta_days,
        "autumn_gap_reduction_frac": autumn_gap_reduction_frac,
        "autumn_final_gap_excess_c": autumn_final_gap_excess_c,
        "yearly_bias": yearly_bias,
        "consecutive_bad_month_bias": consecutive_bad,
        "median_zth_error_summer_m": median_zth_error,
        "median_hth_error_summer_m": median_hth_error,
        "deep_rmse_jul_sep_c": deep_rmse_julsep,
        **heatmap_metrics,
    }


def score_run(
    aligned: pd.DataFrame,
    daily: pd.DataFrame,
    tv_metrics: Dict[str, float],
    eff_depth: float,
    masks: Dict[str, pd.Series],
    thermo_mask: pd.Series,
    args: argparse.Namespace,
    thresholds: ScoreThresholds,
    visual_score: float = np.nan,
    visual_note: str = "",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    lake_type = resolve_lake_type(
        arg_value(args, "lake_type", "auto"),
        truth=aligned.rename(columns={"Tobs": "Temperature_C"})[["Date", "Month", "Depth_m", "Temperature_C"]],
        eff_depth=eff_depth,
    )
    is_warm_deep = lake_type == "warm_deep_monomictic"
    overall = subset_metrics(aligned)
    surface = subset_metrics(aligned[masks["surface"]])
    deep = subset_metrics(aligned[masks["deep"]])
    thermo_band = subset_metrics(aligned[thermo_mask]) if thermo_mask.any() else {"rmse": np.nan, "mae": np.nan, "bias": np.nan}

    may = daily[daily["Month"] == 5]
    april = daily[daily["Month"] == 4]
    july = daily[daily["Month"] == 7]
    winter = daily[daily["Month"].isin(WINTER_MONTHS)]
    therm_summer = daily[daily["Month"].isin(SUMMER_MONTHS)]

    e_may = abs(
        (float(may["Tsurf_hat"].mean()) - float(april["Tsurf_hat"].mean()))
        - (float(may["Tsurf_obs"].mean()) - float(april["Tsurf_obs"].mean()))
    )
    e_july = abs(float(july["Tsurf_hat"].mean()) - float(july["Tsurf_obs"].mean()))
    ez = float(np.nanmedian(np.abs(therm_summer["zth_hat"] - therm_summer["zth_obs"])))
    eh = float(np.nanmedian(np.abs(therm_summer["hth_hat"] - therm_summer["hth_obs"])))
    tmix_obs = pd.Timestamp(daily["tmix_obs"].iloc[0]) if pd.notna(daily["tmix_obs"].iloc[0]) else None
    tmix_hat = pd.Timestamp(daily["tmix_hat"].iloc[0]) if pd.notna(daily["tmix_hat"].iloc[0]) else None
    mix_delta = float(abs((tmix_hat - tmix_obs).days)) if tmix_obs is not None and tmix_hat is not None else np.nan
    einv = abs(
        float(np.median(winter["Tdeep_hat"] - winter["Tsurf_hat"]))
        - float(np.median(winter["Tdeep_obs"] - winter["Tsurf_obs"]))
    )
    esurf_w = abs(float(winter["Tsurf_hat"].mean()) - float(winter["Tsurf_obs"].mean()))

    g_obs = float(np.nanmedian(therm_summer["G_obs"]))
    g_hat = float(np.nanmedian(therm_summer["G_hat"]))
    delta_g = abs(g_hat - g_obs) / max(abs(g_obs), 1e-6)
    mid_bias = abs(subset_metrics(aligned[masks["middepth_4_8"] & masks["jul_sep"]])["bias"])
    deep_bias = abs(subset_metrics(aligned[masks["deep"] & masks["jul_sep"]])["bias"])
    e_heat = 0.5 * mid_bias + 0.5 * deep_bias
    delta_smooth = 0.5 * abs(tv_metrics["TVt_hat"] / max(tv_metrics["TVt_obs"], 1e-6) - 1.0) + 0.5 * abs(
        tv_metrics["TVz_hat"] / max(tv_metrics["TVz_obs"], 1e-6) - 1.0
    )

    numeric_scores = {
        "score_overall_rmse": s_down(overall["rmse"], thresholds.overall_rmse_good, thresholds.overall_rmse_bad, 15.0),
        "score_overall_mae": s_down(overall["mae"], thresholds.overall_mae_good, thresholds.overall_mae_bad, 10.0),
        "score_abs_bias": s_down(abs(overall["bias"]), thresholds.abs_bias_good, thresholds.abs_bias_bad, 5.0),
        "score_surface_rmse": s_down(surface["rmse"], thresholds.surface_rmse_good, thresholds.surface_rmse_bad, 5.0),
        "score_thermocline_rmse": s_down(thermo_band["rmse"], thresholds.thermocline_rmse_good, thresholds.thermocline_rmse_bad, 5.0),
    }
    thermocline_depth_good = thresholds.thermocline_depth_good
    thermocline_depth_bad = thresholds.thermocline_depth_bad
    thermocline_thickness_good = thresholds.thermocline_thickness_good
    thermocline_thickness_bad = thresholds.thermocline_thickness_bad
    if is_warm_deep:
        thermocline_depth_good = max(thermocline_depth_good, 1.5)
        thermocline_depth_bad = max(thermocline_depth_bad, arg_value(args, "warm_deep_thermocline_depth_max_m", 5.0))
        thermocline_thickness_good = max(thermocline_thickness_good, 1.5)
        thermocline_thickness_bad = max(thermocline_thickness_bad, arg_value(args, "warm_deep_thermocline_thickness_max_m", 5.0))

    seasonal_scores = {
        "score_may_warming": s_down(e_may, thresholds.may_warming_good, thresholds.may_warming_bad, 8.0),
        "score_july_surface": s_down(e_july, thresholds.july_surface_good, thresholds.july_surface_bad, 8.0),
        "score_thermocline_depth": s_down(ez, thermocline_depth_good, thermocline_depth_bad, 4.0),
        "score_thermocline_thickness": s_down(eh, thermocline_thickness_good, thermocline_thickness_bad, 4.0),
        "score_autumn_tmix": s_down(mix_delta, thresholds.autumn_tmix_good_days, thresholds.autumn_tmix_bad_days, 7.0),
        "score_winter_inverse": s_down(einv, thresholds.winter_inverse_good, thresholds.winter_inverse_bad, 2.0),
        "score_winter_surface": s_down(esurf_w, thresholds.winter_surface_good, thresholds.winter_surface_bad, 2.0),
    }
    shape_scores = {
        "score_thermocline_sharpness": s_down(delta_g, thresholds.thermocline_sharpness_good, thresholds.thermocline_sharpness_bad, 5.0),
        "score_heat_distribution": s_down(e_heat, thresholds.heat_distribution_good, thresholds.heat_distribution_bad, 5.0),
        "score_smoothness": s_down(delta_smooth, thresholds.smoothness_good, thresholds.smoothness_bad, 5.0),
    }

    stability_scores = {
        "score_seed_stability": np.nan,
        "score_reload_consistency": np.nan,
        "stability_available": False,
    }
    stability_subtotal = 0.0
    if np.isfinite(args.seed_score_std) or np.isfinite(args.reload_mae):
        stability_scores["stability_available"] = True
        seed_score = s_down(args.seed_score_std, thresholds.stability_sigma_good, thresholds.stability_sigma_bad, 5.0) if np.isfinite(args.seed_score_std) else np.nan
        reload_score = s_down(args.reload_mae, thresholds.reload_mae_good, thresholds.reload_mae_bad, 5.0) if np.isfinite(args.reload_mae) else np.nan
        stability_scores["score_seed_stability"] = seed_score
        stability_scores["score_reload_consistency"] = reload_score
        stability_subtotal = float(np.nansum([seed_score, reload_score]))

    visual_score_value = float(visual_score) if np.isfinite(visual_score) else np.nan
    visual_subtotal = s_down(5.0 - visual_score_value, 5.0 - thresholds.visual_score_good, 5.0 - thresholds.visual_score_bad, 5.0) if np.isfinite(visual_score_value) else np.nan

    numeric_subtotal = float(sum(numeric_scores.values()))
    seasonal_subtotal = float(sum(seasonal_scores.values()))
    shape_subtotal = float(sum(shape_scores.values()))
    total_raw = numeric_subtotal + seasonal_subtotal + shape_subtotal + stability_subtotal
    available_weight = 100.0 if stability_scores["stability_available"] else 90.0
    total_scaled_100 = total_raw if stability_scores["stability_available"] else (total_raw / available_weight) * 100.0

    layered_total_raw = numeric_subtotal + seasonal_subtotal
    layered_available_weight = 75.0
    if stability_scores["stability_available"]:
        layered_total_raw += stability_subtotal
        layered_available_weight += 10.0
    if np.isfinite(visual_subtotal):
        layered_total_raw += float(visual_subtotal)
        layered_available_weight += 5.0
    layered_total_scaled_100 = (layered_total_raw / layered_available_weight) * 100.0 if layered_available_weight > 0 else np.nan

    diagnostics = {
        "lake_type": lake_type,
        "overall_rmse": overall["rmse"],
        "overall_mae": overall["mae"],
        "overall_bias": overall["bias"],
        "surface_rmse": surface["rmse"],
        "surface_mae": surface["mae"],
        "surface_bias": surface["bias"],
        "thermocline_rmse": thermo_band["rmse"],
        "thermocline_mae": thermo_band["mae"],
        "thermocline_bias": thermo_band["bias"],
        "deep_rmse": deep["rmse"],
        "deep_mae": deep["mae"],
        "deep_bias": deep["bias"],
        "may_surface_warming_error_c": e_may,
        "july_surface_error_c": e_july,
        "summer_zth_error_m": ez,
        "summer_hth_error_m": eh,
        "tmix_delta_days": mix_delta,
        "winter_inverse_error_c": einv,
        "winter_surface_error_c": esurf_w,
        "thermocline_sharpness_rel_error": delta_g,
        "heat_distribution_error_c": e_heat,
        "smoothness_rel_error": delta_smooth,
        "MLD_RMSE": rmse(daily["MLD_hat"] - daily["MLD_obs"]),
        "BottomTemp_MAE": mae(daily["bottom_hat"] - daily["bottom_obs"]),
        "WholeLakeMean_MAE": mae(daily["mean_hat"] - daily["mean_obs"]),
        "ThermoclineShape_Error": float(np.nanmean(0.5 * np.abs(daily["zth_hat"] - daily["zth_obs"]) + 0.5 * np.abs(daily["hth_hat"] - daily["hth_obs"]))),
        "layer2_key_seasonal_score": seasonal_subtotal,
        "layer3_numeric_score": numeric_subtotal,
        "shape_diagnostic_score": shape_subtotal,
        "layer4_stability_score": stability_subtotal if stability_scores["stability_available"] else np.nan,
        "layer5_visual_score": visual_score_value,
        "layer5_visual_score_normalized": visual_subtotal,
        "layered_selection_score_raw": layered_total_raw,
        "layered_selection_score_100": layered_total_scaled_100,
        "legacy_score_shape": shape_subtotal,
        "legacy_score_total_raw": total_raw,
        "legacy_score_total_scaled_100": total_scaled_100,
        "visual_note": visual_note,
    }
    diagnostics.update(numeric_scores)
    diagnostics.update(seasonal_scores)
    diagnostics.update(shape_scores)
    diagnostics.update(stability_scores)
    return diagnostics, {
        "surface_band_rmse": surface["rmse"],
        "thermocline_band_rmse": thermo_band["rmse"],
        "deep_band_rmse": deep["rmse"],
    }


def default_label(path: Path) -> str:
    return path.parent.name + "_" + path.stem


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:.{digits}f}"


def _fmt_bool(value: object) -> str:
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "1", "yes", "pass"}
    return "通过 PASS" if bool(value) else "未过 FAIL"


def _fmt_run(value: object) -> str:
    text = str(value)
    if text.startswith("official_predict_"):
        return "official_predict\n" + text.removeprefix("official_predict_")
    parts = text.split("_")
    if len(text) > 22 and len(parts) > 2:
        midpoint = max(1, len(parts) // 2)
        return "_".join(parts[:midpoint]) + "\n" + "_".join(parts[midpoint:])
    return text


def configure_report_font():
    """Use a CJK-capable font so bilingual report images render correctly."""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_prop
    plt.rcParams["axes.unicode_minus"] = False
    return None


def write_scorecard_report(merged: pd.DataFrame, output_path: Path) -> None:
    """Write a compact bilingual v2 scorecard PNG for model selection."""
    font_prop = configure_report_font()
    report_cols = [
        ("排名\nRank", "selection_rank", lambda v: str(int(v)) if pd.notna(v) else ""),
        ("模型/结果\nRun", "run", _fmt_run),
        ("湖型\nType", "lake_type", lambda v: str(v).replace("_", "\n") if pd.notna(v) else ""),
        ("物理底线\nPhysics", "layer1_physics_pass", _fmt_bool),
        ("热图物理\nHeatmap", "heatmap_physics_pass", _fmt_bool),
        ("逐点底线\nPoint", "discrete_point_pass", _fmt_bool),
        ("季节覆盖\nSeason Test", "scorecard_v2_seasonal_coverage_pass", _fmt_bool),
        ("失败项数\nFails", "scorecard_v2_failed_check_count", lambda v: str(int(v)) if pd.notna(v) else ""),
        ("总分\nScore", "layered_selection_score_100", lambda v: _fmt_num(v, 2)),
        ("离散点\nRMSE", "discrete_rmse", _fmt_num),
        ("离散点\nBias", "discrete_bias", _fmt_num),
        ("|误差|>2℃\nCount", "discrete_abs_gt_2c_count", lambda v: str(int(v)) if pd.notna(v) else ""),
        ("|误差|>2℃\nFrac", "discrete_abs_gt_2c_frac", lambda v: _fmt_num(float(v) * 100.0, 1) + "%" if pd.notna(v) else ""),
        ("超2℃负荷\nExcess", "discrete_mean_excess_over_2c", _fmt_num),
        ("总体\nRMSE", "overall_rmse", _fmt_num),
        ("总体\nMAE", "overall_mae", _fmt_num),
        ("偏差\nBias", "overall_bias", _fmt_num),
        ("表层\nSurf RMSE", "surface_rmse", _fmt_num),
        ("温跃层\nThermo RMSE", "thermocline_rmse", _fmt_num),
        ("深层\nDeep RMSE", "deep_rmse", _fmt_num),
        ("冬季结构\nWinter", "winter_inverse_pass", _fmt_bool),
        ("夏季分层\nSummer", "summer_strat_pass", _fmt_bool),
        ("秋季翻混\nAutumn", "autumn_tmix_pass", _fmt_bool),
        ("全年漂移\nDrift", "drift_pass", _fmt_bool),
    ]

    headers = [item[0] for item in report_cols]
    rows = []
    for _, row in merged.iterrows():
        rows.append([
            formatter(row[column]) if column in row else ""
            for _, column, formatter in report_cols
        ])

    n_rows = max(len(rows), 1)
    fig_width = 22
    fig_height = max(4.8, 1.35 + 0.58 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(
        "湖泊温度剖面评分 v2 / Lake Profile Scorecard v2",
        fontsize=18,
        fontweight="bold",
        loc="left",
        pad=18,
        fontproperties=font_prop,
    )
    ax.text(
        0.0,
        0.96,
        "Selection rule / 选择规则: physics gates first, then seasonal process, numeric accuracy, stability, and heatmap morphology.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
        va="top",
        fontproperties=font_prop,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.14, 1.0, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)

    pass_fail_columns = {
        idx
        for idx, (_, column, _) in enumerate(report_cols)
        if column.endswith("_pass") or column == "layer1_physics_pass"
    }
    score_col = next(idx for idx, (_, column, _) in enumerate(report_cols) if column == "layered_selection_score_100")
    fail_count_col = next(idx for idx, (_, column, _) in enumerate(report_cols) if column == "scorecard_v2_failed_check_count")

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if font_prop is not None:
            cell.get_text().set_fontproperties(font_prop)
        if r == 0:
            cell.set_facecolor("#0F172A")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            continue
        cell.set_facecolor("#F8FAFC" if r % 2 == 0 else "white")

        text = cell.get_text().get_text()
        if c in pass_fail_columns:
            if "PASS" in text:
                cell.set_facecolor("#DCFCE7")
                cell.get_text().set_color("#166534")
                cell.get_text().set_weight("bold")
            elif "FAIL" in text:
                cell.set_facecolor("#FEE2E2")
                cell.get_text().set_color("#991B1B")
                cell.get_text().set_weight("bold")
        elif c == score_col:
            try:
                score_value = float(text)
            except ValueError:
                score_value = np.nan
            if np.isfinite(score_value):
                if score_value >= 75.0:
                    cell.set_facecolor("#DCFCE7")
                elif score_value >= 65.0:
                    cell.set_facecolor("#FEF3C7")
                else:
                    cell.set_facecolor("#FEE2E2")
                cell.get_text().set_weight("bold")
        elif c == fail_count_col:
            try:
                fail_count = int(text)
            except ValueError:
                fail_count = 0
            if fail_count > 0:
                cell.set_facecolor("#FEE2E2")
                cell.get_text().set_color("#991B1B")
                cell.get_text().set_weight("bold")

    failure_lines = []
    for _, row in merged.head(3).iterrows():
        run_name = _fmt_run(row.get("run", "")).replace("\n", " ")
        failed = str(row.get("scorecard_v2_failed_checks", ""))
        if failed and failed != "none":
            failure_lines.append(f"{run_name}: {failed}")
    if failure_lines:
        ax.text(
            0.0,
            0.04,
            "Failed checks / 未通过项: " + " | ".join(failure_lines),
            transform=ax.transAxes,
            fontsize=8.5,
            color="#7F1D1D",
            va="bottom",
            wrap=True,
            fontproperties=font_prop,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pred_paths = [Path(p) for p in args.pred]
    labels = args.label or []
    visual_scores = args.visual_score or []
    visual_notes = args.visual_note or []
    if labels and len(labels) != len(pred_paths):
        raise ValueError("The number of --label values must match the number of --pred values.")
    if visual_scores and len(visual_scores) != len(pred_paths):
        raise ValueError("The number of --visual-score values must match the number of --pred values.")
    if visual_notes and len(visual_notes) != len(pred_paths):
        raise ValueError("The number of --visual-note values must match the number of --pred values.")
    if not labels:
        labels = [default_label(p) for p in pred_paths]
    if not visual_scores:
        visual_scores = [np.nan] * len(pred_paths)
    else:
        visual_scores = [float(v) if v is not None and str(v).strip() != "" else np.nan for v in visual_scores]
    if not visual_notes:
        visual_notes = [""] * len(pred_paths)

    truth = load_profile_csv(Path(args.truth))
    args.lake_type = resolve_lake_type(args.lake_type, truth=truth)
    thresholds = ScoreThresholds()

    score_rows: List[Dict[str, object]] = []
    veto_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    for label, pred_path, visual_score, visual_note in zip(labels, pred_paths, visual_scores, visual_notes):
        pred = load_profile_csv(pred_path)
        aligned, depths, dates, eff_depth = build_aligned_cube(truth, pred)
        daily = compute_daily_features(
            aligned=aligned,
            depths=depths,
            dates=dates,
            eff_depth=eff_depth,
            surface_band_max_m=args.surface_band_max_m,
            thermocline_delta_min_c=args.thermocline_delta_min_c,
            mld_threshold_c=args.mld_threshold_c,
            mix_delta_col_max_c=args.mix_delta_col_max_c,
            mix_stdz_max_c=args.mix_stdz_max_c,
            mix_consecutive_days=args.mix_consecutive_days,
        )
        tv_metrics = compute_tv_metrics(aligned, depths, dates)
        masks = make_masks(aligned, daily, eff_depth, args.surface_band_max_m)
        thermo_mask = thermocline_band_mask(aligned, daily)
        vetoes = evaluate_vetoes(aligned, daily, depths, dates, masks, thermo_mask, args)
        scores, extra = score_run(aligned, daily, tv_metrics, eff_depth, masks, thermo_mask, args, thresholds, visual_score=visual_score, visual_note=visual_note)

        veto_rows.append({"run": label, "prediction_csv": str(pred_path), **vetoes})
        score_rows.append({"run": label, "prediction_csv": str(pred_path), "effective_depth_m": eff_depth, **scores})
        diagnostics_rows.append({"run": label, "prediction_csv": str(pred_path), "effective_depth_m": eff_depth, **tv_metrics, **extra})

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "lake_profile_scorecard_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    veto_df = pd.DataFrame(veto_rows)
    score_df = pd.DataFrame(score_rows)
    diag_df = pd.DataFrame(diagnostics_rows)

    merged = score_df.merge(veto_df, on=["run", "prediction_csv"], how="left").merge(diag_df, on=["run", "prediction_csv", "effective_depth_m"], how="left")
    if "lake_type_x" in merged.columns or "lake_type_y" in merged.columns:
        merged["lake_type"] = merged.get("lake_type_x", pd.Series(index=merged.index, dtype=object)).combine_first(
            merged.get("lake_type_y", pd.Series(index=merged.index, dtype=object))
        )
    merged["layer1_physics_pass"] = merged["pass_all_vetoes"].astype(bool)
    merged["scorecard_v2_failed_check_count_for_sort"] = -merged.get(
        "scorecard_v2_failed_check_count",
        pd.Series(999, index=merged.index),
    ).fillna(999).astype(float)
    merged["layer4_stability_score_for_sort"] = merged["layer4_stability_score"].fillna(-1.0)
    merged["layer5_visual_score_for_sort"] = merged["layer5_visual_score"].fillna(-1.0)
    merged = merged.sort_values(
        by=[
            "layer1_physics_pass",
            "scorecard_v2_failed_check_count_for_sort",
            "layer2_key_seasonal_score",
            "layer3_numeric_score",
            "layer4_stability_score_for_sort",
            "layer5_visual_score_for_sort",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    merged["selection_rank"] = np.arange(1, len(merged) + 1, dtype=int)

    if args.write_csv:
        veto_df.to_csv(out_dir / "scorecard_vetoes.csv", index=False)
        score_df.to_csv(out_dir / "scorecard_scores.csv", index=False)
        diag_df.to_csv(out_dir / "scorecard_diagnostics.csv", index=False)
        merged.to_csv(out_dir / "scorecard_summary.csv", index=False)

    report_path = out_dir / args.report_name
    write_scorecard_report(merged, report_path)

    show_cols = [
        "selection_rank",
        "run",
        "layer1_physics_pass",
        "layer2_key_seasonal_score",
        "layer3_numeric_score",
        "layer4_stability_score",
        "layer5_visual_score",
        "layered_selection_score_100",
        "shape_diagnostic_score",
        "overall_rmse",
        "overall_mae",
        "overall_bias",
    ]
    print(merged[show_cols].to_string(index=False))
    print(f"\nWrote score report image to: {report_path}")
    if args.write_csv:
        print(f"Wrote detailed CSV tables to: {out_dir}")


if __name__ == "__main__":
    main()
