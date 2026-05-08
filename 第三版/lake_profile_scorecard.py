"""
Lake profile scorecard v1.

This script scores one or more predicted temperature-profile CSV files against
an observed/reference profile CSV using a layered selection template:

1. Physical bottom line
2. Key seasonal processes
3. Numeric accuracy
4. Stability
5. Visual heatmap impression (manual, optional)

Physical-shape metrics are still computed, but they are treated as supporting
diagnostics rather than primary ranking layers.

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
  --truth "E:\\pycharm\\PINN\\数据\\mohonk\\验证\\MohonkLake_temp_2017_filled_from_2014_2017.csv" ^
  --pred "E:\\pycharm\\PINN\\策略测试\\七\\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" ^
  --label "run7" ^
  --pred "E:\\pycharm\\PINN\\11维测试\\一\\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" ^
  --label "11d_run1" ^
  --out-dir "E:\\pycharm\\PINN\\第三版\\score_outputs"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score lake temperature profile predictions.")
    parser.add_argument("--truth", required=True, help="Observed/reference profile CSV.")
    parser.add_argument("--pred", action="append", required=True, help="Prediction CSV (repeatable).")
    parser.add_argument("--label", action="append", default=None, help="Label for each prediction CSV.")
    parser.add_argument("--out-dir", default=None, help="Directory to write score tables.")
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
    masks: Dict[str, pd.Series],
    thermo_mask: pd.Series,
    args: argparse.Namespace,
) -> Dict[str, object]:
    winter = daily[daily["Month"].isin(WINTER_MONTHS)].copy()
    iw_pred = float(np.median(winter["Tdeep_hat"] - winter["Tsurf_hat"])) if len(winter) else np.nan
    finv_pred = float(np.mean((winter["Tdeep_hat"] - winter["Tsurf_hat"]) > 0.5)) if len(winter) else np.nan
    winter_inverse_pass = bool(
        np.isfinite(iw_pred)
        and np.isfinite(finv_pred)
        and iw_pred >= args.winter_inverse_min_c
        and finv_pred >= args.winter_inverse_frac_min
    )

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
    thermocline_pass = bool(
        np.isfinite(median_zth_error)
        and np.isfinite(median_hth_error)
        and median_zth_error <= args.thermocline_depth_max_m
        and median_hth_error <= args.thermocline_thickness_max_m
        and deep_rmse_julsep <= args.deep_julsep_rmse_max_c
    )

    all_pass = all(
        [
            winter_inverse_pass,
            summer_strat_pass,
            autumn_tmix_pass,
            drift_pass,
            thermocline_pass,
        ]
    )

    return {
        "pass_all_vetoes": all_pass,
        "winter_inverse_pass": winter_inverse_pass,
        "summer_strat_pass": summer_strat_pass,
        "autumn_tmix_pass": autumn_tmix_pass,
        "drift_pass": drift_pass,
        "thermocline_pass": thermocline_pass,
        "iw_pred": iw_pred,
        "finv_pred": finv_pred,
        "is_pred": is_pred,
        "fstrat_pred": fstrat_pred,
        "tmix_obs": tmix_obs,
        "tmix_hat": tmix_hat,
        "tmix_delta_days": tmix_delta_days,
        "yearly_bias": yearly_bias,
        "consecutive_bad_month_bias": consecutive_bad,
        "median_zth_error_summer_m": median_zth_error,
        "median_hth_error_summer_m": median_hth_error,
        "deep_rmse_jul_sep_c": deep_rmse_julsep,
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
    overall = subset_metrics(aligned)
    surface = subset_metrics(aligned[masks["surface"]])
    deep = subset_metrics(aligned[masks["deep"]])
    thermo_band = subset_metrics(aligned[thermo_mask]) if thermo_mask.any() else {"rmse": np.nan, "mae": np.nan, "bias": np.nan}

    may = daily[daily["Month"] == 5]
    april = daily[daily["Month"] == 4]
    july = daily[daily["Month"] == 7]
    winter = daily[daily["Month"].isin(WINTER_MONTHS)]
    summer = daily[daily["Month"].isin(SUMMER_MONTHS)]
    autumn = daily[daily["Month"].between(10, 11)]
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
    seasonal_scores = {
        "score_may_warming": s_down(e_may, thresholds.may_warming_good, thresholds.may_warming_bad, 8.0),
        "score_july_surface": s_down(e_july, thresholds.july_surface_good, thresholds.july_surface_bad, 8.0),
        "score_thermocline_depth": s_down(ez, thresholds.thermocline_depth_good, thresholds.thermocline_depth_bad, 4.0),
        "score_thermocline_thickness": s_down(eh, thresholds.thermocline_thickness_good, thresholds.thermocline_thickness_bad, 4.0),
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
        vetoes = evaluate_vetoes(aligned, daily, masks, thermo_mask, args)
        scores, extra = score_run(aligned, daily, tv_metrics, eff_depth, masks, thermo_mask, args, thresholds, visual_score=visual_score, visual_note=visual_note)

        veto_rows.append({"run": label, "prediction_csv": str(pred_path), **vetoes})
        score_rows.append({"run": label, "prediction_csv": str(pred_path), "effective_depth_m": eff_depth, **scores})
        diagnostics_rows.append({"run": label, "prediction_csv": str(pred_path), "effective_depth_m": eff_depth, **tv_metrics, **extra})

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "lake_profile_scorecard_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    veto_df = pd.DataFrame(veto_rows)
    score_df = pd.DataFrame(score_rows)
    diag_df = pd.DataFrame(diagnostics_rows)

    veto_df.to_csv(out_dir / "scorecard_vetoes.csv", index=False)
    score_df.to_csv(out_dir / "scorecard_scores.csv", index=False)
    diag_df.to_csv(out_dir / "scorecard_diagnostics.csv", index=False)

    merged = score_df.merge(veto_df, on=["run", "prediction_csv"], how="left").merge(diag_df, on=["run", "prediction_csv", "effective_depth_m"], how="left")
    merged["layer1_physics_pass"] = merged["pass_all_vetoes"].astype(bool)
    merged["layer4_stability_score_for_sort"] = merged["layer4_stability_score"].fillna(-1.0)
    merged["layer5_visual_score_for_sort"] = merged["layer5_visual_score"].fillna(-1.0)
    merged = merged.sort_values(
        by=[
            "layer1_physics_pass",
            "layer2_key_seasonal_score",
            "layer3_numeric_score",
            "layer4_stability_score_for_sort",
            "layer5_visual_score_for_sort",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    merged["selection_rank"] = np.arange(1, len(merged) + 1, dtype=int)
    merged.to_csv(out_dir / "scorecard_summary.csv", index=False)

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
    print(f"\nWrote score tables to: {out_dir}")


if __name__ == "__main__":
    main()
