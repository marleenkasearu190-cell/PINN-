"""R27 eval-only heat-closure diagnostic for LOCAL62 zero-profile RECON.

This script loads an existing checkpoint, replays zero-profile annual rollouts,
and records daily areal heat-closure residuals. It does not train, tune by
heldout, change model code, change split roles, or write standard inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lake_pinn.state_multilake import (  # noqa: E402
    _lake_group_id,
    _lake_reservoir_bucket,
    _lookup_mask,
    train_multilake_state_forecaster,
)
from lake_pinn.state_reconstruction import (  # noqa: E402
    apply_lswt_observer_update,
    initialize_rollout_state,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
DEFAULT_MANIFEST = (
    ROOT
    / "experiments"
    / "manifests_clean"
    / "diagnostics"
    / "RECON_LOCAL62_ZERO_PROFILE_GROUPHELDOUT_V3_active_manifest_20260614.json"
)
DEFAULT_SPLIT = ROOT / "experiments" / "splits" / "LOCAL62_ZERO_PROFILE_GROUPHELDOUT_V3.json"
DEFAULT_CHECKPOINT = (
    PIPELINE_ROOT
    / "remote_artifact_backups"
    / "RECON_R25_LOCAL62_NONSTRESS_SHORTDIAG_v1_20260614_0806"
    / "best_by_val_rolling.pt"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R27_heat_closure_diagnostic_20260614"
)
EXPERIMENT_ID = "RECON_R27_HEAT_CLOSURE_DIAGNOSTIC_v1"


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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
        return None if not np.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _finite(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _finite_mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_abs_mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def _finite_p(values, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _finite_rate(values, predicate) -> float:
    arr = _finite(values)
    return float(np.mean(predicate(arr))) if arr.size else float("nan")


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        arr = value.detach().cpu().reshape(-1)
        if arr.numel() == 0:
            return float(default)
        return float(arr[0])
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            return float(arr[0]) if arr.size else float(default)
        return float(value)
    except Exception:
        return float(default)


def _diag_scalar(diagnostics: dict, key: str, default: float = 0.0) -> float:
    return _as_float(diagnostics.get(key), default=default)


def _row_scalar(row: dict, key: str, default: float = 0.0) -> float:
    return _as_float(row.get(key), default=default)


def _date_to_index(df: pd.DataFrame) -> dict[pd.Timestamp, int]:
    return {
        pd.Timestamp(date).normalize(): int(idx)
        for idx, date in enumerate(pd.to_datetime(df["Date"], errors="coerce"))
        if not pd.isna(date)
    }


def _config_specs() -> list[dict]:
    return [
        {
            "config_id": "observer_off",
            "observer_mode": "off",
            "strength": 0.0,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.0,
            "heat_content_limit_c": 0.0,
        },
        {
            "config_id": "conservative_surface_s020",
            "observer_mode": "conservative_surface",
            "strength": 0.20,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.30,
            "heat_content_limit_c": 0.05,
        },
    ]


def _model_metric_row(lake: dict, predictions_by_index: dict[int, np.ndarray], rollout_start_idx: int) -> dict:
    date_to_index = _date_to_index(lake["df"])
    depths = np.asarray(lake["depths_np"], dtype=np.float64).reshape(-1)
    records: list[dict] = []
    for obs_date, target in lake["lookups"]["all"].items():
        obs_ts = pd.Timestamp(obs_date).normalize()
        obs_idx = date_to_index.get(obs_ts)
        if obs_idx is None or obs_idx <= rollout_start_idx or obs_idx not in predictions_by_index:
            continue
        pred = np.asarray(predictions_by_index[obs_idx], dtype=np.float64).reshape(-1)
        target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        mask = _lookup_mask(lake, "all", obs_date)
        valid = np.isfinite(pred) & np.isfinite(target_arr)
        if mask is not None:
            valid = valid & np.asarray(mask, dtype=bool).reshape(-1)
        if not np.any(valid):
            continue
        diff = pred[valid] - target_arr[valid]
        for depth, error in zip(depths[valid], diff):
            records.append({"depth_m": float(depth), "error_c": float(error)})

    def metric(mask_fn) -> tuple[float, float, int]:
        subset = np.asarray(
            [row["error_c"] for row in records if mask_fn(row["depth_m"])],
            dtype=np.float64,
        )
        subset = subset[np.isfinite(subset)]
        if subset.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.sqrt(np.mean(subset ** 2))), float(np.mean(subset)), int(subset.size)

    whole_rmse, whole_bias, whole_count = metric(lambda _z: True)
    surface_rmse, surface_bias, surface_count = metric(lambda z: z <= 1.0)
    le25_rmse, le25_bias, le25_count = metric(lambda z: z <= 25.0)
    gt25_rmse, gt25_bias, gt25_count = metric(lambda z: z > 25.0)
    return {
        "whole_rmse_c": whole_rmse,
        "whole_bias_c": whole_bias,
        "whole_point_count": whole_count,
        "surface_rmse_c": surface_rmse,
        "surface_bias_c": surface_bias,
        "surface_point_count": surface_count,
        "le25m_rmse_c": le25_rmse,
        "le25m_bias_c": le25_bias,
        "le25m_point_count": le25_count,
        "gt25m_rmse_c": gt25_rmse,
        "gt25m_bias_c": gt25_bias,
        "gt25m_point_count": gt25_count,
    }


def _daily_record_base(
    *,
    role: str,
    config: dict,
    lake: dict,
    date_value,
    phase: str,
    row_dict: dict,
) -> dict:
    ts = pd.Timestamp(date_value)
    ice_fraction = _row_scalar(row_dict, "ice_fraction", 0.0)
    open_water = bool(ice_fraction < 0.10)
    lake_type = _lake_reservoir_bucket(lake)
    return {
        "experiment_id": EXPERIMENT_ID,
        "config_id": config["config_id"],
        "observer_mode": config["observer_mode"],
        "observer_strength": float(config["strength"]),
        "role": role,
        "heldout_policy": "diagnostic_only" if role in {"heldout", "stress_ood"} else "validation_only" if role == "val" else "train",
        "lake_id": lake["lake_id"],
        "lake_group": _lake_group_id(lake),
        "lake_type": lake_type,
        "is_reservoir": int(lake_type == "reservoir"),
        "Date": ts.date().isoformat(),
        "month": int(ts.month),
        "season": _season(int(ts.month)),
        "phase": phase,
        "ice_fraction": float(ice_fraction),
        "open_water_day": int(open_water),
        "ice_transition_day": int(not open_water),
        "lst_observed_flag": _row_scalar(row_dict, "lst_observed_flag", 0.0),
        "lst_is_filled": _row_scalar(row_dict, "lst_is_filled", 1.0),
    }


def _append_diagnostic_values(target: dict, diagnostics: dict, prefix: str = "") -> None:
    for key in (
        "surface_flux_wm2",
        "open_water_surface_flux_wm2",
        "open_water_net_radiation_wm2",
        "open_water_sensible_heat_wm2",
        "open_water_latent_heat_wm2",
        "heat_input_wm2",
        "heat_tendency_wm2",
        "sensible_heat_tendency_wm2",
        "effective_heat_tendency_wm2",
        "freezing_storage_j_m2",
        "freezing_storage_change_wm2",
        "temperature_floor_heat_injection_wm2",
        "temperature_ceiling_heat_removal_wm2",
        "surface_flux_bias_wm2",
        "shortwave_to_water_wm2",
        "kz_mean",
        "kz_surface_mean",
        "kz_mid_mean",
        "kz_deep_mean",
        "adaptive_kd_multiplier",
        "nn_kd_multiplier",
        "residual_abs_mean_c",
        "density_adjustment_applied",
        "density_adjustment_max_delta_c",
        "density_adjustment_heat_delta_j_m2",
        "advective_heat_source_active_mean",
        "advective_heat_source_c_per_day_mean",
    ):
        target[f"{prefix}{key}"] = _diag_scalar(diagnostics, key, float("nan"))
    heat_tendency = target.get(f"{prefix}heat_tendency_wm2", float("nan"))
    heat_input = target.get(f"{prefix}heat_input_wm2", float("nan"))
    surface_flux = target.get(f"{prefix}surface_flux_wm2", float("nan"))
    open_surface_flux = target.get(f"{prefix}open_water_surface_flux_wm2", float("nan"))

    solver_residual = heat_tendency - heat_input
    surface_residual = heat_tendency - surface_flux
    open_surface_residual = heat_tendency - open_surface_flux
    heat_input_surface_gap = heat_input - surface_flux
    heat_input_open_surface_gap = heat_input - open_surface_flux

    # Solver residual is a numerical closure check. Surface residual is the
    # physically relevant storage-vs-surface-flux diagnostic for R27.
    target[f"{prefix}solver_energy_residual_wm2"] = float(solver_residual)
    target[f"{prefix}solver_energy_residual_abs_wm2"] = (
        float(abs(solver_residual)) if np.isfinite(solver_residual) else float("nan")
    )
    target[f"{prefix}surface_energy_residual_wm2"] = float(surface_residual)
    target[f"{prefix}surface_energy_residual_abs_wm2"] = (
        float(abs(surface_residual)) if np.isfinite(surface_residual) else float("nan")
    )
    target[f"{prefix}open_water_surface_energy_residual_wm2"] = float(open_surface_residual)
    target[f"{prefix}open_water_surface_energy_residual_abs_wm2"] = (
        float(abs(open_surface_residual)) if np.isfinite(open_surface_residual) else float("nan")
    )
    target[f"{prefix}heat_input_minus_surface_flux_wm2"] = float(heat_input_surface_gap)
    target[f"{prefix}heat_input_minus_open_water_surface_flux_wm2"] = float(heat_input_open_surface_gap)
    target[f"{prefix}energy_residual_wm2"] = float(surface_residual)
    target[f"{prefix}energy_residual_abs_wm2"] = (
        float(abs(surface_residual)) if np.isfinite(surface_residual) else float("nan")
    )
    target[f"{prefix}threshold36_violation"] = int(np.isfinite(surface_residual) and abs(surface_residual) > 36.0)
    target[f"{prefix}threshold50_violation"] = int(np.isfinite(surface_residual) and abs(surface_residual) > 50.0)


def _append_observer_values(target: dict, detail: dict, prefix: str) -> None:
    for key in (
        "lswt_observer_applied_count",
        "lswt_observer_surface_innovation_c",
        "lswt_observer_mean_abs_delta_c",
        "lswt_observer_max_abs_delta_c",
        "lswt_observer_heat_content_delta_c",
        "lswt_observer_deep_abs_delta_c",
        "lswt_observer_filled_lst_used_count",
        "lswt_observer_localization_depth_m",
        "lswt_observer_reservoir_conservative_scale",
        "lswt_observer_heat_content_bound_scale",
        "lswt_observer_density_guard_scale",
    ):
        target[f"{prefix}{key}"] = _diag_scalar(detail, key, 0.0)


@torch.no_grad()
def _run_lake_heat_closure(model, lake: dict, role: str, config: dict, *, spinup_days: int, min_quality: float) -> tuple[list[dict], dict]:
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
        init_mode="zero_profile_low_dof",
        rollout_start_date=None,
        spinup_days=int(spinup_days),
        zero_profile_initializer="low_dof",
        spinup_lswt_observer_mode=config["observer_mode"],
        spinup_lst_assimilation_strength=float(config["strength"]),
        spinup_lst_assimilation_decay_depth_m=float(config["decay_depth_m"]),
        spinup_lst_assimilation_max_increment_c=float(config["max_increment_c"]),
        lswt_observer_low_rank_deep_update_fraction=0.0,
        lswt_observer_heat_content_limit_c=float(config["heat_content_limit_c"]),
        lswt_observer_min_quality=float(min_quality),
        task_mode="analysis",
        area_profile=lake["area"],
        hard_density_stability=False,
    )
    current = init_state["current"]
    freezing_storage = init_state.get("freezing_storage_j_m2", torch.zeros_like(current))
    rollout_start_idx = int(init_state["rollout_start_idx"])
    records: list[dict] = []
    predictions_by_index = {
        int(idx): np.asarray(profile, dtype=np.float32)
        for idx, profile in init_state["profiles_by_index"].items()
    }
    date_to_idx = _date_to_index(df)

    for diag in init_state.get("diagnostics", []):
        date = pd.Timestamp(diag.get("Date"))
        row_idx = date_to_idx.get(date.normalize())
        row_dict = lake["forcing_rows"][row_idx] if row_idx is not None and row_idx < len(lake["forcing_rows"]) else {}
        record = _daily_record_base(
            role=role,
            config=config,
            lake=lake,
            date_value=date,
            phase="spinup",
            row_dict=row_dict,
        )
        _append_diagnostic_values(record, diag)
        for src_key, out_key in (
            ("spinup_lswt_observer_applied_count", "spinup_lswt_observer_applied_count"),
            ("spinup_lswt_observer_heat_content_delta_c", "spinup_lswt_observer_heat_content_delta_c"),
            ("spinup_lswt_observer_deep_abs_delta_c", "spinup_lswt_observer_deep_abs_delta_c"),
            ("spinup_lswt_observer_filled_lst_used_count", "spinup_lswt_observer_filled_lst_used_count"),
            ("spinup_lswt_observer_mean_abs_delta_c", "spinup_lswt_observer_mean_abs_delta_c"),
        ):
            record[out_key] = float(diag.get(src_key, 0.0))
        records.append(record)

    for day_idx in range(rollout_start_idx, len(df) - 1):
        next_row = lake["forcing_rows"][day_idx + 1] if day_idx + 1 < len(lake["forcing_rows"]) else None
        current, freezing_storage, step_diag = model.step(
            current,
            lake["forcing_rows"][day_idx],
            lake["static_features"],
            next_forcing_row=next_row,
            task_mode="analysis",
            depths=lake["depths"],
            area_profile=lake["area"],
            hard_density_stability=False,
            freezing_storage_j_m2=freezing_storage,
            return_diagnostics=True,
            return_freezing_storage=True,
        )
        observer_detail = {}
        if config["observer_mode"] != "off" and next_row is not None:
            current, observer_detail = apply_lswt_observer_update(
                current,
                next_row,
                lake["depths"],
                mode=config["observer_mode"],
                strength=float(config["strength"]),
                decay_depth_m=float(config["decay_depth_m"]),
                max_increment_c=float(config["max_increment_c"]),
                low_rank_deep_update_fraction=0.0,
                heat_content_limit_c=float(config["heat_content_limit_c"]),
                min_quality=float(min_quality),
                area_profile=lake["area"],
                metadata=lake.get("metadata"),
            )
        predictions_by_index[day_idx + 1] = current.detach().cpu().numpy().reshape(-1)
        date_value = pd.Timestamp(df["Date"].iloc[day_idx + 1])
        record = _daily_record_base(
            role=role,
            config=config,
            lake=lake,
            date_value=date_value,
            phase="rollout",
            row_dict=next_row or lake["forcing_rows"][day_idx],
        )
        _append_diagnostic_values(record, step_diag)
        _append_observer_values(record, observer_detail, "rollout_")
        record["surface_temp_c"] = float(current[:, 0].detach().cpu().reshape(-1)[0])
        record["mean_temp_c"] = float(current.detach().cpu().mean())
        records.append(record)

    metrics = _model_metric_row(lake, predictions_by_index, rollout_start_idx)
    metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_id": config["config_id"],
            "observer_mode": config["observer_mode"],
            "observer_strength": float(config["strength"]),
            "role": role,
            "heldout_policy": "diagnostic_only" if role in {"heldout", "stress_ood"} else "validation_only" if role == "val" else "train",
            "lake_id": lake["lake_id"],
            "lake_group": _lake_group_id(lake),
            "lake_type": _lake_reservoir_bucket(lake),
            "rollout_start_idx": rollout_start_idx,
            "spinup_days_used": int(init_state.get("spinup_days_used", 0)),
        }
    )
    return records, metrics


def _summarize_daily(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["main_open_water"] = (pd.to_numeric(frame["open_water_day"], errors="coerce").fillna(0) > 0) & (
        frame["phase"] == "rollout"
    )
    group_iter = frame.groupby(group_cols, dropna=False, sort=True)
    for group_values, group in group_iter:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = {key: value for key, value in zip(group_cols, group_values)}
        surface_residual = pd.to_numeric(group["surface_energy_residual_wm2"], errors="coerce")
        solver_residual = pd.to_numeric(group["solver_energy_residual_wm2"], errors="coerce")
        surface_gap = pd.to_numeric(group["heat_input_minus_surface_flux_wm2"], errors="coerce")
        open_group = group[group["main_open_water"]]
        open_surface_residual = pd.to_numeric(open_group["open_water_surface_energy_residual_wm2"], errors="coerce")
        open_solver_residual = pd.to_numeric(open_group["solver_energy_residual_wm2"], errors="coerce")
        open_surface_gap = pd.to_numeric(open_group["heat_input_minus_open_water_surface_flux_wm2"], errors="coerce")
        record.update(
            {
                "day_count": int(len(group)),
                "rollout_day_count": int((group["phase"] == "rollout").sum()),
                "open_water_rollout_day_count": int(len(open_group)),
                "energy_residual_mean_wm2": _finite_mean(surface_residual),
                "energy_residual_abs_mean_wm2": _finite_abs_mean(surface_residual),
                "energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(surface_residual)), 95.0),
                "threshold36_violation_fraction": _finite_rate(surface_residual, lambda x: np.abs(x) > 36.0),
                "threshold50_violation_fraction": _finite_rate(surface_residual, lambda x: np.abs(x) > 50.0),
                "surface_energy_residual_mean_wm2": _finite_mean(surface_residual),
                "surface_energy_residual_abs_mean_wm2": _finite_abs_mean(surface_residual),
                "surface_energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(surface_residual)), 95.0),
                "surface_threshold36_violation_fraction": _finite_rate(surface_residual, lambda x: np.abs(x) > 36.0),
                "surface_threshold50_violation_fraction": _finite_rate(surface_residual, lambda x: np.abs(x) > 50.0),
                "open_water_energy_residual_mean_wm2": _finite_mean(open_surface_residual),
                "open_water_energy_residual_abs_mean_wm2": _finite_abs_mean(open_surface_residual),
                "open_water_energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(open_surface_residual)), 95.0),
                "open_water_threshold36_violation_fraction": _finite_rate(open_surface_residual, lambda x: np.abs(x) > 36.0),
                "open_water_threshold50_violation_fraction": _finite_rate(open_surface_residual, lambda x: np.abs(x) > 50.0),
                "open_water_surface_energy_residual_mean_wm2": _finite_mean(open_surface_residual),
                "open_water_surface_energy_residual_abs_mean_wm2": _finite_abs_mean(open_surface_residual),
                "open_water_surface_energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(open_surface_residual)), 95.0),
                "open_water_surface_threshold36_violation_fraction": _finite_rate(open_surface_residual, lambda x: np.abs(x) > 36.0),
                "open_water_surface_threshold50_violation_fraction": _finite_rate(open_surface_residual, lambda x: np.abs(x) > 50.0),
                "solver_energy_residual_abs_mean_wm2": _finite_abs_mean(solver_residual),
                "solver_energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(solver_residual)), 95.0),
                "open_water_solver_energy_residual_abs_mean_wm2": _finite_abs_mean(open_solver_residual),
                "open_water_solver_energy_residual_abs_p95_wm2": _finite_p(np.abs(_finite(open_solver_residual)), 95.0),
                "heat_input_minus_surface_flux_abs_mean_wm2": _finite_abs_mean(surface_gap),
                "heat_input_minus_surface_flux_abs_p95_wm2": _finite_p(np.abs(_finite(surface_gap)), 95.0),
                "open_water_heat_input_minus_surface_flux_abs_mean_wm2": _finite_abs_mean(open_surface_gap),
                "open_water_heat_input_minus_surface_flux_abs_p95_wm2": _finite_p(np.abs(_finite(open_surface_gap)), 95.0),
                "heat_input_mean_wm2": _finite_mean(group.get("heat_input_wm2", [])),
                "heat_tendency_mean_wm2": _finite_mean(group.get("heat_tendency_wm2", [])),
                "surface_flux_bias_mean_wm2": _finite_mean(group.get("surface_flux_bias_wm2", [])),
                "rollout_observer_applied_count_sum": float(pd.to_numeric(group.get("rollout_lswt_observer_applied_count", 0), errors="coerce").fillna(0.0).sum()),
                "rollout_observer_heat_content_delta_abs_mean_c": _finite_abs_mean(group.get("rollout_lswt_observer_heat_content_delta_c", [])),
                "rollout_observer_deep_abs_delta_mean_c": _finite_mean(group.get("rollout_lswt_observer_deep_abs_delta_c", [])),
                "rollout_filled_lst_strong_update_count_sum": float(pd.to_numeric(group.get("rollout_lswt_observer_filled_lst_used_count", 0), errors="coerce").fillna(0.0).sum()),
                "adaptive_kd_multiplier_mean": _finite_mean(group.get("adaptive_kd_multiplier", [])),
                "adaptive_kd_multiplier_p95": _finite_p(group.get("adaptive_kd_multiplier", []), 95.0),
                "residual_abs_mean_c": _finite_mean(group.get("residual_abs_mean_c", [])),
                "density_adjustment_applied_rate": _finite_mean(group.get("density_adjustment_applied", [])),
            }
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _summarize_lake_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for group_values, group in metrics.groupby(["config_id", "role", "lake_type"], dropna=False, sort=True):
        config_id, role, lake_type = group_values
        rows.append(
            {
                "config_id": config_id,
                "role": role,
                "lake_type": lake_type,
                "lake_count": int(len(group)),
                "whole_rmse_mean_c": _finite_mean(group["whole_rmse_c"]),
                "whole_bias_mean_c": _finite_mean(group["whole_bias_c"]),
                "surface_rmse_mean_c": _finite_mean(group["surface_rmse_c"]),
                "surface_bias_mean_c": _finite_mean(group["surface_bias_c"]),
                "le25m_rmse_mean_c": _finite_mean(group["le25m_rmse_c"]),
                "le25m_bias_mean_c": _finite_mean(group["le25m_bias_c"]),
                "gt25m_rmse_mean_c": _finite_mean(group["gt25m_rmse_c"]),
                "gt25m_bias_mean_c": _finite_mean(group["gt25m_bias_c"]),
                "whole_point_count_sum": int(pd.to_numeric(group["whole_point_count"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_report(output_dir: Path, summary: pd.DataFrame, lake_metrics_summary: pd.DataFrame, paths: dict[str, Path]) -> Path:
    report = output_dir / "R27_heat_closure_diagnostic_report.md"
    lines = [
        "# R27 Heat Closure Diagnostic",
        "",
        "Scope: eval-only diagnostic. No training, no model/loss/Kd/Kz/residual bound change, no split change, no `_standard_inputs` change.",
        "",
        "Checkpoint: R25 best_by_val_rolling from local remote-artifact backup.",
        "",
        "Main protocol: zero-profile low-dof free rollout, observer-off and conservative-surface observer-on comparison.",
        "",
        "## Key Heat Closure Summary",
        "",
        "Primary residual: heat_tendency_wm2 - open_water_surface_flux_wm2 on open-water rollout days.",
        "Solver residual is reported separately as a numerical closure check.",
        "",
    ]
    if not summary.empty:
        key = summary[
            (summary["phase"] == "rollout")
            & (summary["role"].isin(["val", "heldout"]))
            & (summary["lake_type"].isin(["natural", "reservoir"]))
        ].copy()
        if key.empty:
            key = summary[summary["phase"] == "rollout"].copy()
        for _, row in key.sort_values(["config_id", "role", "lake_type"]).iterrows():
            lines.append(
                "- "
                f"{row['config_id']} {row['role']} {row['lake_type']}: "
                f"open_surface_abs_mean={row['open_water_surface_energy_residual_abs_mean_wm2']:.2f} W/m2, "
                f"open_surface_abs_p95={row['open_water_surface_energy_residual_abs_p95_wm2']:.2f} W/m2, "
                f"tau36_rate={row['open_water_surface_threshold36_violation_fraction']:.3f}, "
                f"solver_abs_mean={row['open_water_solver_energy_residual_abs_mean_wm2']:.4f} W/m2, "
                f"input_minus_flux_abs={row['open_water_heat_input_minus_surface_flux_abs_mean_wm2']:.2f} W/m2, "
                f"observer_heat_abs={row['rollout_observer_heat_content_delta_abs_mean_c']:.4f} C"
            )
    lines.extend(["", "## Product Metric Context", ""])
    if not lake_metrics_summary.empty:
        for _, row in lake_metrics_summary.sort_values(["config_id", "role", "lake_type"]).iterrows():
            lines.append(
                "- "
                f"{row['config_id']} {row['role']} {row['lake_type']}: "
                f"whole={row['whole_rmse_mean_c']:.3f} C, "
                f"surface={row['surface_rmse_mean_c']:.3f} C, "
                f"<=25m={row['le25m_rmse_mean_c']:.3f} C, "
                f">25m={row['gt25m_rmse_mean_c']:.3f} C"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Heldout metrics are diagnostic-only and are not used for checkpoint selection or tuning.",
            "- Surface heat residual reduction alone is not model success; temperature RMSE, Kd/Kz/residual, density, natural/reservoir split, and observer heat increment must remain clean.",
            "- Near-zero solver residual only means the numerical update is internally consistent; it is not evidence that the lake heat budget matches external surface flux.",
            "- The primary surface residual excludes penetrating shortwave absorbed below the surface layer; do not turn it into a loss until total absorbed flux and source-term bookkeeping are aligned.",
            "- Reservoir heat closure is diagnostic-only because unmodeled inflow/outflow/withdrawal heat can be non-small.",
            "- Open-water rows are the primary heat-closure read; ice-transition rows are diagnostic-only.",
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
    parser = argparse.ArgumentParser(description="R27 heat-closure eval-only diagnostic.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--roles", default="train,val,heldout")
    parser.add_argument("--max-lakes-per-role", type=int, default=0)
    parser.add_argument("--spinup-days", type=int, default=90)
    parser.add_argument("--min-quality", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = Path(args.manifest)
    split_path = Path(args.split)
    checkpoint_path = Path(args.checkpoint)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    loader_manifest = output_dir / "R27_loader_manifest_epochs0.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_payload["epochs"] = 0
    manifest_payload["export_after_training"] = "off"
    loader_manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"R27 loading model on {device} from {checkpoint_path}", flush=True)
    result = train_multilake_state_forecaster(
        loader_manifest,
        output_dir / "bootstrap_epochs0_loader",
        epochs=0,
        checkpoint_path=checkpoint_path,
        export_after_training="off",
        profile_runtime=False,
        device=device,
    )
    model = result["model"]
    lakes = {lake["lake_id"]: lake for lake in result["lakes"]}
    split_payload = json.loads(split_path.read_text(encoding="utf-8-sig"))
    role_to_ids = {
        "train": split_payload.get("train_lake_ids", []),
        "val": split_payload.get("val_lake_ids", []),
        "heldout": split_payload.get("heldout_lake_ids", []),
        "stress_ood": split_payload.get("stress_or_ood_lake_ids", []),
    }
    roles = _parse_list(args.roles)
    daily_rows: list[dict] = []
    metric_rows: list[dict] = []
    run_lake_rows: list[dict] = []
    for config in _config_specs():
        for role in roles:
            ids = [lake_id for lake_id in role_to_ids.get(role, []) if lake_id in lakes]
            if args.max_lakes_per_role > 0:
                ids = ids[: int(args.max_lakes_per_role)]
            for lake_id in ids:
                lake = lakes[lake_id]
                print(f"R27 {config['config_id']} {role}: {lake_id}", flush=True)
                lake_daily, lake_metrics = _run_lake_heat_closure(
                    model,
                    lake,
                    role,
                    config,
                    spinup_days=args.spinup_days,
                    min_quality=args.min_quality,
                )
                daily_rows.extend(lake_daily)
                metric_rows.append(lake_metrics)
                run_lake_rows.append(
                    {
                        "config_id": config["config_id"],
                        "role": role,
                        "lake_id": lake_id,
                        "lake_group": _lake_group_id(lake),
                        "lake_type": _lake_reservoir_bucket(lake),
                        "daily_rows": int(len(lake_daily)),
                    }
                )

    daily = pd.DataFrame.from_records(daily_rows)
    lake_metrics = pd.DataFrame.from_records(metric_rows)
    daily_path = output_dir / "R27_heat_closure_daily_diagnostics.csv"
    lake_metrics_path = output_dir / "R27_heat_closure_lake_temperature_metrics.csv"
    run_lakes_path = output_dir / "R27_heat_closure_run_lakes.csv"
    summary_path = output_dir / "R27_heat_closure_summary_by_role_laketype.csv"
    monthly_path = output_dir / "R27_heat_closure_monthly_summary.csv"
    seasonal_path = output_dir / "R27_heat_closure_seasonal_summary.csv"
    lake_summary_path = output_dir / "R27_heat_closure_lake_temperature_metrics_summary.csv"
    overall_path = output_dir / "R27_heat_closure_overall_summary.json"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    lake_metrics.to_csv(lake_metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(run_lake_rows).to_csv(run_lakes_path, index=False, encoding="utf-8-sig")

    summary = _summarize_daily(daily, ["config_id", "role", "lake_type", "phase"])
    monthly = _summarize_daily(daily, ["config_id", "role", "lake_type", "phase", "month"])
    seasonal = _summarize_daily(daily, ["config_id", "role", "lake_type", "phase", "season"])
    lake_metrics_summary = _summarize_lake_metrics(lake_metrics)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    seasonal.to_csv(seasonal_path, index=False, encoding="utf-8-sig")
    lake_metrics_summary.to_csv(lake_summary_path, index=False, encoding="utf-8-sig")

    overall = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_eval_only_no_training",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "roles": roles,
        "configs": _config_specs(),
        "daily_row_count": int(len(daily)),
        "lake_metric_count": int(len(lake_metrics)),
        "heldout_policy": "diagnostic-only; not used for checkpoint selection or tuning",
        "stress_policy": "not included unless explicitly requested and present in manifest; diagnostic-only",
    }
    if not summary.empty:
        rollout = summary[summary["phase"] == "rollout"]
        overall["rollout_summary_records"] = _json_safe(rollout.to_dict(orient="records"))
    overall_path.write_text(json.dumps(_json_safe(overall), ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = _write_report(
        output_dir,
        summary,
        lake_metrics_summary,
        {
            "daily": daily_path,
            "summary_by_role_laketype": summary_path,
            "monthly": monthly_path,
            "seasonal": seasonal_path,
            "lake_temperature_metrics": lake_metrics_path,
            "lake_temperature_metrics_summary": lake_summary_path,
            "run_lakes": run_lakes_path,
            "overall_json": overall_path,
        },
    )
    print(json.dumps(_json_safe({"report": report_path, "summary": overall}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
