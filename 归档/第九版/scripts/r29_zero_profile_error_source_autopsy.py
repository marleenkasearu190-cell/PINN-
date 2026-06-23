"""R29 eval-only zero-profile error-source autopsy.

This script replays existing zero-profile rollouts from an existing checkpoint
and writes observation-level depth-band errors. It does not train, tune by
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
from r27_heat_closure_diagnostic import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    _config_specs,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R29_zero_profile_error_source_autopsy_20260614"
)
EXPERIMENT_ID = "RECON_R29_ZERO_PROFILE_ERROR_SOURCE_AUTOPSY_v1"


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
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _rmse(values) -> float:
    arr = _finite(values)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _mae(values) -> float:
    arr = _finite(values)
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def _mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _p(values, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


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


def _error_band_records(
    *,
    role: str,
    config: dict,
    lake: dict,
    obs_date,
    obs_idx: int,
    rollout_start_idx: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask,
    row_dict: dict,
    last_observer_idx: int | None,
    cumulative_observer_count: int,
    latest_observer_detail: dict,
) -> list[dict]:
    depths = np.asarray(lake["depths_np"], dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(valid):
        return []
    lake_type = _lake_reservoir_bucket(lake)
    ts = pd.Timestamp(obs_date)
    days_since_observer = np.nan if last_observer_idx is None else int(obs_idx - last_observer_idx)
    base = {
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
        "day_idx": int(obs_idx),
        "rollout_start_idx": int(rollout_start_idx),
        "days_since_rollout_start": int(obs_idx - rollout_start_idx),
        "days_since_observer_update": days_since_observer,
        "cumulative_observer_update_count": int(cumulative_observer_count),
        "ice_fraction": _row_scalar(row_dict, "ice_fraction", 0.0),
        "open_water_day": int(_row_scalar(row_dict, "ice_fraction", 0.0) < 0.10),
        "lst_observed_flag": _row_scalar(row_dict, "lst_observed_flag", 0.0),
        "lst_is_filled": _row_scalar(row_dict, "lst_is_filled", 1.0),
        "lst_quality": max(
            _row_scalar(row_dict, "lst_quality", float("nan")),
            _row_scalar(row_dict, "lst_day_quality", float("nan")),
        ),
        "latest_observer_surface_innovation_c": _diag_scalar(
            latest_observer_detail, "lswt_observer_surface_innovation_c", float("nan")
        ),
        "latest_observer_heat_content_delta_c": _diag_scalar(
            latest_observer_detail, "lswt_observer_heat_content_delta_c", 0.0
        ),
        "latest_observer_deep_abs_delta_c": _diag_scalar(
            latest_observer_detail, "lswt_observer_deep_abs_delta_c", 0.0
        ),
        "latest_observer_filled_lst_used_count": _diag_scalar(
            latest_observer_detail, "lswt_observer_filled_lst_used_count", 0.0
        ),
    }
    bands = {
        "surface": valid & (depths <= 1.0),
        "le25m": valid & (depths <= 25.0),
        "gt25m": valid & (depths > 25.0),
        "whole": valid,
    }
    rows: list[dict] = []
    for band, band_mask in bands.items():
        if not np.any(band_mask):
            continue
        errors = pred[band_mask] - target[band_mask]
        rows.append(
            {
                **base,
                "depth_band": band,
                "point_count": int(errors.size),
                "rmse_c": float(np.sqrt(np.mean(errors**2))),
                "mae_c": float(np.mean(np.abs(errors))),
                "bias_c": float(np.mean(errors)),
                "abs_bias_c": float(abs(np.mean(errors))),
                "min_depth_m": float(np.min(depths[band_mask])),
                "max_depth_m": float(np.max(depths[band_mask])),
            }
        )
    return rows


@torch.no_grad()
def _run_lake(model, lake: dict, role: str, config: dict, *, spinup_days: int, min_quality: float) -> list[dict]:
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
    date_to_idx = _date_to_index(df)
    profile_idx_to_date = {
        date_to_idx[pd.Timestamp(obs_date).normalize()]: obs_date
        for obs_date in lake["lookups"]["all"].keys()
        if pd.Timestamp(obs_date).normalize() in date_to_idx
    }
    rows: list[dict] = []
    last_observer_idx: int | None = None
    cumulative_observer_count = 0
    latest_observer_detail: dict = {}

    for day_idx in range(rollout_start_idx, len(df) - 1):
        next_row = lake["forcing_rows"][day_idx + 1] if day_idx + 1 < len(lake["forcing_rows"]) else None
        current, freezing_storage, _step_diag = model.step(
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
            applied = int(round(_diag_scalar(observer_detail, "lswt_observer_applied_count", 0.0)))
            if applied > 0:
                last_observer_idx = int(day_idx + 1)
                cumulative_observer_count += applied
                latest_observer_detail = dict(observer_detail)

        obs_idx = day_idx + 1
        if obs_idx not in profile_idx_to_date or obs_idx <= rollout_start_idx:
            continue
        obs_date = profile_idx_to_date[obs_idx]
        target = np.asarray(lake["lookups"]["all"][obs_date], dtype=np.float64).reshape(-1)
        pred = current.detach().cpu().numpy().reshape(-1).astype(np.float64)
        mask = _lookup_mask(lake, "all", obs_date)
        row_dict = next_row or lake["forcing_rows"][day_idx]
        rows.extend(
            _error_band_records(
                role=role,
                config=config,
                lake=lake,
                obs_date=obs_date,
                obs_idx=obs_idx,
                rollout_start_idx=rollout_start_idx,
                pred=pred,
                target=target,
                mask=mask,
                row_dict=row_dict,
                last_observer_idx=last_observer_idx,
                cumulative_observer_count=cumulative_observer_count,
                latest_observer_detail=latest_observer_detail,
            )
        )
    return rows


def _add_bins(obs: pd.DataFrame) -> pd.DataFrame:
    frame = obs.copy()
    frame["rollout_age_bin"] = pd.cut(
        pd.to_numeric(frame["days_since_rollout_start"], errors="coerce"),
        bins=[-0.1, 30, 90, 180, 365, 99999],
        labels=["0-30d", "31-90d", "91-180d", "181-365d", ">365d"],
    ).astype(str)
    frame["observer_age_bin"] = pd.cut(
        pd.to_numeric(frame["days_since_observer_update"], errors="coerce"),
        bins=[-0.1, 7, 30, 90, 365, 99999],
        labels=["0-7d", "8-30d", "31-90d", "91-365d", ">365d"],
    ).astype(str)
    frame.loc[pd.to_numeric(frame["days_since_observer_update"], errors="coerce").isna(), "observer_age_bin"] = "no_update"
    frame["lst_status"] = np.select(
        [
            pd.to_numeric(frame["lst_observed_flag"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(frame["lst_is_filled"], errors="coerce").fillna(1) > 0,
        ],
        ["raw_observed", "filled_or_reconstructed"],
        default="unknown",
    )
    return frame


def _summarize(obs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if obs.empty:
        return pd.DataFrame()
    for group_values, group in obs.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        rec = {key: value for key, value in zip(group_cols, group_values)}
        rec.update(
            {
                "observation_count": int(len(group)),
                "lake_count": int(group["lake_id"].nunique()),
                "point_count_sum": int(pd.to_numeric(group["point_count"], errors="coerce").fillna(0).sum()),
                "rmse_mean_c": _mean(group["rmse_c"]),
                "rmse_point_weighted_c": float(
                    np.sqrt(
                        np.average(
                            np.square(pd.to_numeric(group["rmse_c"], errors="coerce").fillna(np.nan)),
                            weights=pd.to_numeric(group["point_count"], errors="coerce").fillna(0.0),
                        )
                    )
                )
                if pd.to_numeric(group["point_count"], errors="coerce").fillna(0.0).sum() > 0
                else float("nan"),
                "mae_mean_c": _mean(group["mae_c"]),
                "bias_mean_c": _mean(group["bias_c"]),
                "abs_bias_mean_c": _mean(group["abs_bias_c"]),
                "rmse_p95_c": _p(group["rmse_c"], 95.0),
                "days_since_rollout_mean": _mean(group["days_since_rollout_start"]),
                "days_since_observer_mean": _mean(group["days_since_observer_update"]),
                "filled_lst_update_sum": float(
                    pd.to_numeric(group["latest_observer_filled_lst_used_count"], errors="coerce").fillna(0.0).sum()
                ),
                "deep_delta_mean_c": _mean(group["latest_observer_deep_abs_delta_c"]),
                "observer_heat_delta_abs_mean_c": _mean(
                    pd.to_numeric(group["latest_observer_heat_content_delta_c"], errors="coerce").abs()
                ),
            }
        )
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def _observer_gain(summary: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["role", "lake_type", "depth_band"]
    base = summary[summary["config_id"] == "observer_off"][key_cols + ["rmse_point_weighted_c", "bias_mean_c"]]
    on = summary[summary["config_id"] == "conservative_surface_s020"][key_cols + ["rmse_point_weighted_c", "bias_mean_c"]]
    merged = base.merge(on, on=key_cols, how="outer", suffixes=("_off", "_on"))
    merged["rmse_gain_c"] = merged["rmse_point_weighted_c_off"] - merged["rmse_point_weighted_c_on"]
    merged["bias_abs_gain_c"] = merged["bias_mean_c_off"].abs() - merged["bias_mean_c_on"].abs()
    return merged


def _diagnostic_decisions(summary: pd.DataFrame, gain: pd.DataFrame, age_summary: pd.DataFrame) -> dict:
    decisions: dict[str, object] = {
        "heat_loss_training": "no_go_from_R28",
        "formal_claim": "no_go_diagnostic_only",
    }
    val = summary[(summary["role"] == "val") & (summary["config_id"] == "conservative_surface_s020")]
    heldout = summary[(summary["role"] == "heldout") & (summary["config_id"] == "conservative_surface_s020")]
    surface_gain = gain[(gain["role"] == "val") & (gain["depth_band"] == "surface")]["rmse_gain_c"]
    whole_gain = gain[(gain["role"] == "val") & (gain["depth_band"] == "whole")]["rmse_gain_c"]
    decisions["surface_observer_gain_mean_c"] = _mean(surface_gain)
    decisions["whole_observer_gain_mean_c"] = _mean(whole_gain)
    decisions["observer_assessment"] = (
        "surface_helpful_but_not_full_solution"
        if _mean(surface_gain) > 0.25 and _mean(whole_gain) < _mean(surface_gain)
        else "observer_gain_weak_or_uniform"
    )
    natural_whole = val[(val["lake_type"] == "natural") & (val["depth_band"] == "whole")]["rmse_point_weighted_c"]
    reservoir_whole = val[(val["lake_type"] == "reservoir") & (val["depth_band"] == "whole")]["rmse_point_weighted_c"]
    decisions["val_natural_whole_rmse_c"] = _mean(natural_whole)
    decisions["val_reservoir_whole_rmse_c"] = _mean(reservoir_whole)
    decisions["reservoir_assessment"] = (
        "reservoir_specific_diagnostic_needed"
        if np.isfinite(_mean(reservoir_whole)) and np.isfinite(_mean(natural_whole)) and _mean(reservoir_whole) > _mean(natural_whole) + 0.4
        else "reservoir_not_primary_on_validation_but_check_heldout"
    )
    age_val = age_summary[
        (age_summary["role"] == "val")
        & (age_summary["config_id"] == "conservative_surface_s020")
        & (age_summary["depth_band"] == "whole")
    ]
    early = age_val[age_val["rollout_age_bin"].isin(["0-30d", "31-90d"])]["rmse_point_weighted_c"]
    late = age_val[age_val["rollout_age_bin"].isin(["181-365d", ">365d"])]["rmse_point_weighted_c"]
    decisions["early_whole_rmse_c"] = _mean(early)
    decisions["late_whole_rmse_c"] = _mean(late)
    if np.isfinite(_mean(early)) and _mean(early) > 2.5:
        decisions["initializer_assessment"] = "initializer_or_spinup_error_present_early"
    else:
        decisions["initializer_assessment"] = "early_error_not_enough_to_call_initializer_primary"
    if np.isfinite(_mean(late)) and np.isfinite(_mean(early)) and _mean(late) > _mean(early) + 0.5:
        decisions["drift_assessment"] = "rollout_error_grows_with_age"
    else:
        decisions["drift_assessment"] = "no_clear_age_growth_from_available_profile_dates"
    decisions["heldout_policy"] = "diagnostic_only_not_for_tuning"
    decisions["next_action"] = (
        "write approval-gated R30 proposal only if changing initializer/observer/reservoir physics; "
        "do not launch training automatically"
    )
    decisions["heldout_rows_evaluated"] = int(len(heldout))
    return decisions


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    gain: pd.DataFrame,
    age_summary: pd.DataFrame,
    decisions: dict,
    paths: dict[str, Path],
) -> Path:
    report = output_dir / "R29_zero_profile_error_source_autopsy_report.md"
    lines = [
        "# R29 Zero-Profile Error Source Autopsy",
        "",
        "Scope: eval-only diagnostic. No training, no model/loss/Kd/Kz/residual bound change, no split change, no `_standard_inputs` change.",
        "",
        "Goal: identify whether remaining zero-profile error is dominated by surface observer limits, early initializer/spinup error, rollout-age drift, depth-band failure, or reservoir-specific behavior.",
        "",
        "## Observer Gain",
        "",
    ]
    if not gain.empty:
        key = gain[gain["role"].isin(["val", "heldout"])].copy()
        for _, row in key.sort_values(["role", "lake_type", "depth_band"]).iterrows():
            lines.append(
                "- "
                f"{row['role']} {row['lake_type']} {row['depth_band']}: "
                f"off={row['rmse_point_weighted_c_off']:.3f} C, "
                f"on={row['rmse_point_weighted_c_on']:.3f} C, "
                f"gain={row['rmse_gain_c']:.3f} C, "
                f"abs_bias_gain={row['bias_abs_gain_c']:.3f} C"
            )
    lines.extend(["", "## Conservative Observer Metrics", ""])
    key_summary = summary[
        (summary["config_id"] == "conservative_surface_s020")
        & (summary["role"].isin(["val", "heldout"]))
        & (summary["depth_band"].isin(["surface", "le25m", "gt25m", "whole"]))
    ]
    for _, row in key_summary.sort_values(["role", "lake_type", "depth_band"]).iterrows():
        lines.append(
            "- "
            f"{row['role']} {row['lake_type']} {row['depth_band']}: "
            f"rmse={row['rmse_point_weighted_c']:.3f} C, "
            f"bias={row['bias_mean_c']:.3f} C, "
            f"obs={int(row['observation_count'])}, lakes={int(row['lake_count'])}"
        )
    lines.extend(["", "## Rollout Age", ""])
    age_key = age_summary[
        (age_summary["config_id"] == "conservative_surface_s020")
        & (age_summary["role"] == "val")
        & (age_summary["depth_band"] == "whole")
    ]
    for _, row in age_key.sort_values(["lake_type", "rollout_age_bin"]).iterrows():
        lines.append(
            "- "
            f"val {row['lake_type']} {row['rollout_age_bin']}: "
            f"rmse={row['rmse_point_weighted_c']:.3f} C, "
            f"bias={row['bias_mean_c']:.3f} C, obs={int(row['observation_count'])}"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- observer_assessment: `{decisions.get('observer_assessment')}`",
            f"- initializer_assessment: `{decisions.get('initializer_assessment')}`",
            f"- drift_assessment: `{decisions.get('drift_assessment')}`",
            f"- reservoir_assessment: `{decisions.get('reservoir_assessment')}`",
            f"- next_action: `{decisions.get('next_action')}`",
            "",
            "## Guardrails",
            "",
            "- Heldout metrics are diagnostic-only and are not used for checkpoint selection or tuning.",
            "- This run cannot support formal L3/L5/L7 claims.",
            "- Any initializer, observer, reservoir, or loss change requires a separate approval packet before implementation or training.",
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
    parser = argparse.ArgumentParser(description="R29 zero-profile error-source autopsy.")
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

    loader_manifest = output_dir / "R29_loader_manifest_epochs0.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_payload["epochs"] = 0
    manifest_payload["export_after_training"] = "off"
    loader_manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"R29 loading model on {device} from {checkpoint_path}", flush=True)
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
    obs_rows: list[dict] = []
    run_lakes: list[dict] = []
    for config in _config_specs():
        for role in roles:
            ids = [lake_id for lake_id in role_to_ids.get(role, []) if lake_id in lakes]
            if args.max_lakes_per_role > 0:
                ids = ids[: int(args.max_lakes_per_role)]
            for lake_id in ids:
                lake = lakes[lake_id]
                print(f"R29 {config['config_id']} {role}: {lake_id}", flush=True)
                rows = _run_lake(
                    model,
                    lake,
                    role,
                    config,
                    spinup_days=args.spinup_days,
                    min_quality=args.min_quality,
                )
                obs_rows.extend(rows)
                run_lakes.append(
                    {
                        "config_id": config["config_id"],
                        "role": role,
                        "lake_id": lake_id,
                        "lake_group": _lake_group_id(lake),
                        "lake_type": _lake_reservoir_bucket(lake),
                        "observation_band_rows": int(len(rows)),
                    }
                )

    observations = _add_bins(pd.DataFrame.from_records(obs_rows))
    obs_path = output_dir / "R29_observation_depth_band_errors.csv"
    run_lakes_path = output_dir / "R29_run_lakes.csv"
    summary_path = output_dir / "R29_error_summary_by_role_laketype_band.csv"
    gain_path = output_dir / "R29_observer_gain_by_role_laketype_band.csv"
    age_path = output_dir / "R29_error_by_rollout_age.csv"
    observer_age_path = output_dir / "R29_error_by_observer_age.csv"
    lst_path = output_dir / "R29_error_by_lst_status.csv"
    decisions_path = output_dir / "R29_error_source_decisions.json"

    observations.to_csv(obs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(run_lakes).to_csv(run_lakes_path, index=False, encoding="utf-8-sig")
    summary = _summarize(observations, ["config_id", "role", "lake_type", "depth_band"])
    age_summary = _summarize(observations, ["config_id", "role", "lake_type", "depth_band", "rollout_age_bin"])
    observer_age_summary = _summarize(
        observations,
        ["config_id", "role", "lake_type", "depth_band", "observer_age_bin"],
    )
    lst_summary = _summarize(observations, ["config_id", "role", "lake_type", "depth_band", "lst_status"])
    gain = _observer_gain(summary)
    decisions = _diagnostic_decisions(summary, gain, age_summary)

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gain.to_csv(gain_path, index=False, encoding="utf-8-sig")
    age_summary.to_csv(age_path, index=False, encoding="utf-8-sig")
    observer_age_summary.to_csv(observer_age_path, index=False, encoding="utf-8-sig")
    lst_summary.to_csv(lst_path, index=False, encoding="utf-8-sig")
    decisions_path.write_text(json.dumps(_json_safe(decisions), ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = _write_report(
        output_dir,
        summary,
        gain,
        age_summary,
        decisions,
        {
            "observation_errors": obs_path,
            "summary_by_role_laketype_band": summary_path,
            "observer_gain": gain_path,
            "rollout_age": age_path,
            "observer_age": observer_age_path,
            "lst_status": lst_path,
            "decisions": decisions_path,
            "run_lakes": run_lakes_path,
        },
    )
    overall = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_eval_only_no_training",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "roles": roles,
        "daily_observation_band_row_count": int(len(observations)),
        "heldout_policy": "diagnostic-only; not used for checkpoint selection or tuning",
        "decisions": decisions,
        "report": str(report_path),
    }
    print(json.dumps(_json_safe(overall), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
