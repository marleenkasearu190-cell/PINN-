"""R30 eval-only zero-profile initializer and shallow observer comparison.

This script loads an existing checkpoint and compares zero-profile initializer,
spin-up length, and raw-open-water LSWT observer variants. It does not train,
tune by heldout, change splits, or write standard inputs.
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
)
from r29_zero_profile_error_source_autopsy import (  # noqa: E402
    _add_bins,
    _as_float,
    _date_to_index,
    _diag_scalar,
    _error_band_records,
    _json_safe,
    _mean,
    _parse_list,
    _row_scalar,
    _summarize,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R30_zero_profile_initializer_observer_eval_20260614"
)
EXPERIMENT_ID = "RECON_R30_ZERO_PROFILE_INITIALIZER_SPINUP_EVAL_v1"


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(max(0, int(float(item))))
    return out or [90]


def _config_specs(initializers: list[str], observer_modes: list[str], spinup_days: list[int]) -> list[dict]:
    configs: list[dict] = []
    for initializer in initializers:
        for observer_mode in observer_modes:
            for days in spinup_days:
                if observer_mode == "off":
                    strength = 0.0
                    max_increment = 0.0
                    heat_limit = 0.0
                elif observer_mode == "conservative_mld_shallow":
                    strength = 0.20
                    max_increment = 0.30
                    heat_limit = 0.05
                else:
                    strength = 0.20
                    max_increment = 0.30
                    heat_limit = 0.05
                configs.append(
                    {
                        "config_id": f"{initializer}_{observer_mode}_spinup{int(days)}d",
                        "zero_profile_initializer": initializer,
                        "observer_mode": observer_mode,
                        "strength": strength,
                        "decay_depth_m": 1.5,
                        "max_increment_c": max_increment,
                        "heat_content_limit_c": heat_limit,
                        "spinup_days": int(days),
                    }
                )
    return configs


@torch.no_grad()
def _run_lake(model, lake: dict, role: str, config: dict, *, min_quality: float) -> list[dict]:
    model.eval()
    df = lake["df"]
    initializer = str(config["zero_profile_initializer"])
    init_mode = "zero_profile_low_dof" if initializer == "low_dof" else "prior_spinup"
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
        spinup_days=int(config["spinup_days"]),
        zero_profile_initializer=initializer,
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
        band_rows = _error_band_records(
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
        for row in band_rows:
            row["experiment_id"] = EXPERIMENT_ID
            row["zero_profile_initializer"] = initializer
            row["spinup_days"] = int(config["spinup_days"])
            row["post_spinup_surface_temp_c"] = _as_float(init_state["current"][:, 0], float("nan"))
            row["prior_raw_lswt_count"] = int(init_state.get("prior_info", {}).get("zero_profile_raw_lswt_count", 0) or 0)
            row["prior_filled_lst_strong_target_used"] = bool(
                init_state.get("prior_info", {}).get("zero_profile_filled_lst_strong_target_used", False)
            )
            row["latest_observer_mld_depth_m"] = _diag_scalar(
                latest_observer_detail,
                "lswt_observer_mld_depth_m",
                0.0,
            )
            row["latest_observer_localization_depth_m"] = _diag_scalar(
                latest_observer_detail,
                "lswt_observer_localization_depth_m",
                0.0,
            )
            row["latest_observer_reservoir_conservative_scale"] = _diag_scalar(
                latest_observer_detail,
                "lswt_observer_reservoir_conservative_scale",
                1.0,
            )
            row["update_day_surface_innovation_c"] = _diag_scalar(
                latest_observer_detail,
                "lswt_observer_surface_innovation_c",
                float("nan"),
            )
            row["lst_surface_c"] = _row_scalar(row_dict, "lst_surface", float("nan"))
            row["lswt_open_water_c"] = _row_scalar(row_dict, "lswt_open_water", float("nan"))
        rows.extend(band_rows)
    return rows


def _compare_to_baseline(summary: pd.DataFrame, baseline_config: str) -> pd.DataFrame:
    key_cols = ["role", "lake_type", "depth_band"]
    baseline = summary[summary["config_id"] == baseline_config][
        key_cols + ["rmse_point_weighted_c", "bias_mean_c"]
    ].rename(
        columns={
            "rmse_point_weighted_c": "baseline_rmse_point_weighted_c",
            "bias_mean_c": "baseline_bias_mean_c",
        }
    )
    comp = summary.merge(baseline, on=key_cols, how="left")
    comp["rmse_gain_vs_baseline_c"] = comp["baseline_rmse_point_weighted_c"] - comp["rmse_point_weighted_c"]
    comp["abs_bias_gain_vs_baseline_c"] = comp["baseline_bias_mean_c"].abs() - comp["bias_mean_c"].abs()
    return comp


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    lst_summary: pd.DataFrame,
    decisions: dict,
    paths: dict[str, Path],
) -> Path:
    report = output_dir / "R30_zero_profile_initializer_observer_eval_report.md"
    lines = [
        "# R30 Zero-Profile Initializer / Observer Eval",
        "",
        "Scope: eval-only diagnostic. No training, no checkpoint selection by heldout, no split or `_standard_inputs` change.",
        "",
        "Goal: compare `lswt_climatology_low_dof` and `conservative_mld_shallow` against low-dof/off baselines for zero-profile profile reconstruction.",
        "",
        "## Key Validation Metrics",
        "",
    ]
    key = comparison[
        (comparison["role"] == "val")
        & (comparison["depth_band"].isin(["surface", "le25m", "gt25m", "whole"]))
    ]
    for _, row in key.sort_values(["depth_band", "lake_type", "config_id"]).iterrows():
        lines.append(
            "- "
            f"{row['config_id']} {row['lake_type']} {row['depth_band']}: "
            f"rmse={row['rmse_point_weighted_c']:.3f} C, "
            f"gain_vs_baseline={row['rmse_gain_vs_baseline_c']:.3f} C, "
            f"bias={row['bias_mean_c']:.3f} C"
        )
    lines.extend(["", "## LST Guardrails", ""])
    if not lst_summary.empty:
        guard = lst_summary[
            (lst_summary["role"].isin(["val", "heldout"]))
            & (lst_summary["depth_band"] == "whole")
        ]
        for _, row in guard.sort_values(["role", "config_id", "lst_status"]).iterrows():
            lines.append(
                "- "
                f"{row['role']} {row['config_id']} {row['lst_status']}: "
                f"rmse={row['rmse_point_weighted_c']:.3f} C, "
                f"filled_update_sum={row['filled_lst_update_sum']:.1f}, "
                f"deep_delta_mean={row['deep_delta_mean_c']:.4f} C"
            )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- best_validation_config_by_whole_rmse: `{decisions.get('best_validation_config_by_whole_rmse')}`",
            f"- validation_whole_rmse_gain_c: `{decisions.get('validation_whole_rmse_gain_c')}`",
            f"- filled_lst_strong_update_sum: `{decisions.get('filled_lst_strong_update_sum')}`",
            f"- max_deep_delta_mean_c: `{decisions.get('max_deep_delta_mean_c')}`",
            f"- next_action: `{decisions.get('next_action')}`",
            "",
            "Heldout remains diagnostic-only and is not used to choose configs.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _decisions(summary: pd.DataFrame, comparison: pd.DataFrame, baseline_config: str) -> dict:
    val_whole = comparison[
        (comparison["role"] == "val")
        & (comparison["depth_band"] == "whole")
        & (comparison["lake_type"] == "all")
    ].copy()
    baseline_rmse = _mean(val_whole[val_whole["config_id"] == baseline_config]["rmse_point_weighted_c"])
    best_config = ""
    best_rmse = float("nan")
    best_gain = float("nan")
    if not val_whole.empty:
        best_idx = pd.to_numeric(val_whole["rmse_point_weighted_c"], errors="coerce").idxmin()
        if pd.notna(best_idx):
            best_row = val_whole.loc[best_idx]
            best_config = str(best_row["config_id"])
            best_rmse = float(best_row["rmse_point_weighted_c"])
            best_gain = float(best_row["rmse_gain_vs_baseline_c"])
    filled_sum = float(
        pd.to_numeric(summary.get("filled_lst_update_sum", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    )
    max_deep_delta = float(
        pd.to_numeric(summary.get("deep_delta_mean_c", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs().max()
    )
    go = np.isfinite(best_gain) and best_gain > 0.10 and filled_sum == 0.0 and max_deep_delta <= 0.02
    return {
        "baseline_validation_whole_rmse_c": baseline_rmse,
        "best_validation_config_by_whole_rmse": best_config,
        "best_validation_whole_rmse_c": best_rmse,
        "validation_whole_rmse_gain_c": best_gain,
        "filled_lst_strong_update_sum": filled_sum,
        "max_deep_delta_mean_c": max_deep_delta,
        "heldout_policy": "diagnostic-only; not used for config selection",
        "next_action": (
            "remote_short_diagnostic_training_can_be_requested"
            if go
            else "no_training_until_eval_matrix_or_safety_issue_is_reviewed"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R30 zero-profile initializer/observer eval-only comparison.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--roles", default="val,heldout")
    parser.add_argument("--initializers", default="low_dof,lswt_climatology_low_dof")
    parser.add_argument("--observer-modes", default="off,conservative_surface,conservative_mld_shallow")
    parser.add_argument("--spinup-days-matrix", default="30,90,180")
    parser.add_argument("--max-lakes-per-role", type=int, default=0)
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

    loader_manifest = output_dir / "R30_loader_manifest_epochs0.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_payload["epochs"] = 0
    manifest_payload["export_after_training"] = "off"
    loader_manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"R30 loading model on {device} from {checkpoint_path}", flush=True)
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
    initializers = _parse_list(args.initializers)
    observer_modes = _parse_list(args.observer_modes)
    spinup_days = _parse_int_list(args.spinup_days_matrix)
    configs = _config_specs(initializers, observer_modes, spinup_days)
    baseline_config = f"{initializers[0]}_off_spinup{spinup_days[0]}d"

    obs_rows: list[dict] = []
    run_lakes: list[dict] = []
    for config in configs:
        for role in roles:
            ids = [lake_id for lake_id in role_to_ids.get(role, []) if lake_id in lakes]
            if args.max_lakes_per_role > 0:
                ids = ids[: int(args.max_lakes_per_role)]
            for lake_id in ids:
                lake = lakes[lake_id]
                print(f"R30 {config['config_id']} {role}: {lake_id}", flush=True)
                rows = _run_lake(model, lake, role, config, min_quality=args.min_quality)
                obs_rows.extend(rows)
                run_lakes.append(
                    {
                        "config_id": config["config_id"],
                        "zero_profile_initializer": config["zero_profile_initializer"],
                        "observer_mode": config["observer_mode"],
                        "spinup_days": int(config["spinup_days"]),
                        "role": role,
                        "lake_id": lake_id,
                        "lake_group": _lake_group_id(lake),
                        "lake_type": _lake_reservoir_bucket(lake),
                        "observation_band_rows": int(len(rows)),
                    }
                )

    observations = _add_bins(pd.DataFrame.from_records(obs_rows))
    obs_path = output_dir / "R30_observation_depth_band_errors.csv"
    run_lakes_path = output_dir / "R30_run_lakes.csv"
    summary_path = output_dir / "R30_error_summary_by_role_laketype_band.csv"
    comparison_path = output_dir / "R30_comparison_vs_baseline.csv"
    age_path = output_dir / "R30_error_by_rollout_age.csv"
    lst_path = output_dir / "R30_error_by_lst_status.csv"
    decisions_path = output_dir / "R30_eval_decisions.json"

    observations.to_csv(obs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(run_lakes).to_csv(run_lakes_path, index=False, encoding="utf-8-sig")
    summary_by_type = _summarize(observations, ["config_id", "role", "lake_type", "depth_band"])
    summary_all = _summarize(observations, ["config_id", "role", "depth_band"])
    if not summary_all.empty:
        summary_all.insert(2, "lake_type", "all")
    summary = pd.concat([summary_by_type, summary_all], ignore_index=True, sort=False)
    age_by_type = _summarize(observations, ["config_id", "role", "lake_type", "depth_band", "rollout_age_bin"])
    age_all = _summarize(observations, ["config_id", "role", "depth_band", "rollout_age_bin"])
    if not age_all.empty:
        age_all.insert(2, "lake_type", "all")
    age_summary = pd.concat([age_by_type, age_all], ignore_index=True, sort=False)
    lst_by_type = _summarize(observations, ["config_id", "role", "lake_type", "depth_band", "lst_status"])
    lst_all = _summarize(observations, ["config_id", "role", "depth_band", "lst_status"])
    if not lst_all.empty:
        lst_all.insert(2, "lake_type", "all")
    lst_summary = pd.concat([lst_by_type, lst_all], ignore_index=True, sort=False)
    comparison = _compare_to_baseline(summary, baseline_config)
    decisions = _decisions(summary, comparison, baseline_config)

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    age_summary.to_csv(age_path, index=False, encoding="utf-8-sig")
    lst_summary.to_csv(lst_path, index=False, encoding="utf-8-sig")
    decisions_path.write_text(json.dumps(_json_safe(decisions), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        output_dir,
        summary,
        comparison,
        lst_summary,
        decisions,
        {
            "observation_errors": obs_path,
            "summary_by_role_laketype_band": summary_path,
            "comparison_vs_baseline": comparison_path,
            "rollout_age": age_path,
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
        "configs": configs,
        "baseline_config": baseline_config,
        "daily_observation_band_row_count": int(len(observations)),
        "heldout_policy": "diagnostic-only; not used for checkpoint selection or tuning",
        "decisions": decisions,
        "report": str(report_path),
    }
    print(json.dumps(_json_safe(overall), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
