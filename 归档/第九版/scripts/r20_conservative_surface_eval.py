"""R20 eval-only matrix for conservative shallow LSWT observer.

The script loads an existing checkpoint with an epochs=0 manifest copy and
evaluates zero-profile free-roll metrics.  It does not train, tune on heldout,
or change data/splits/model defaults.
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
    evaluate_lake_free_roll,
    _lake_group_id,
    _lake_reservoir_bucket,
    train_multilake_state_forecaster,
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
    / "R20_conservative_surface_observer_eval_20260612"
)
DEFAULT_VAL_IDS = "erken_2019,erken_2020,mohonk_2017,carvins_cove_2022"
DEFAULT_HELDOUT_IDS = "lacawac_2016,el_val_2019,el_val_2022,namco_2012"


def _parse_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _finite_mean(values) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_sum(values) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return float(np.sum(arr)) if arr.size else 0.0


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
    return value


def _candidate_configs() -> list[dict]:
    return [
        {
            "config_id": "off_reference",
            "mode": "off",
            "strength": 0.0,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.0,
            "deep_update_fraction": 0.0,
            "heat_content_limit_c": 0.0,
        },
        {
            "config_id": "low_rank_reference_s025",
            "mode": "low_rank",
            "strength": 0.25,
            "decay_depth_m": 2.0,
            "max_increment_c": 0.75,
            "deep_update_fraction": 0.10,
            "heat_content_limit_c": 0.25,
        },
        {
            "config_id": "conservative_surface_s005",
            "mode": "conservative_surface",
            "strength": 0.05,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.30,
            "deep_update_fraction": 0.0,
            "heat_content_limit_c": 0.05,
        },
        {
            "config_id": "conservative_surface_s010",
            "mode": "conservative_surface",
            "strength": 0.10,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.30,
            "deep_update_fraction": 0.0,
            "heat_content_limit_c": 0.05,
        },
        {
            "config_id": "conservative_surface_s020",
            "mode": "conservative_surface",
            "strength": 0.20,
            "decay_depth_m": 1.5,
            "max_increment_c": 0.30,
            "deep_update_fraction": 0.0,
            "heat_content_limit_c": 0.05,
        },
    ]


def _write_report(output_dir: Path, summary: pd.DataFrame, best_val: dict | None, paths: dict[str, Path]) -> Path:
    lines = [
        "# R20 Conservative Surface Observer Eval",
        "",
        "Scope: local eval-only diagnostic. No training, no remote task, no split/manifest/_standard_inputs changes.",
        "",
        "Protocol: zero_profile_low_dof free-roll matrix using existing R14 best checkpoint.",
        "",
        "Validation ranking by whole-profile RMSE:",
    ]
    val_rows = summary[summary["split"] == "val"].sort_values("rmse_mean")
    for _, row in val_rows.iterrows():
        lines.append(
            "- "
            f"{row['config_id']}: rmse={row['rmse_mean']:.4f}, "
            f"surface={row['surface_rmse_mean']:.4f}, "
            f"le25m={row['rmse_le25m_mean']:.4f}, "
            f"reservoir_rmse={row.get('reservoir_rmse_mean', np.nan):.4f}, "
            f"filled_updates={row['filled_lst_strong_update_count']:.0f}"
        )
    if best_val:
        lines.extend(
            [
                "",
                "Best validation config:",
                f"- config_id: {best_val.get('config_id')}",
                f"- mode: {best_val.get('mode')}",
                f"- strength: {best_val.get('strength')}",
                f"- heat_content_limit_c: {best_val.get('heat_content_limit_c')}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- Heldout metrics are diagnostic-only and are not used for selecting a configuration.",
            "- This matrix cannot support formal L3/L5/L7 claims by itself.",
            "- A candidate only remains viable if filled-LST strong update count is 0 and validation surface gains do not create whole-profile or reservoir regression.",
            "",
            "Artifacts:",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: {path}")
    report_path = output_dir / "R20_conservative_surface_observer_eval_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R20 conservative surface observer eval matrix.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-ids", default=DEFAULT_VAL_IDS)
    parser.add_argument("--heldout-ids", default=DEFAULT_HELDOUT_IDS)
    parser.add_argument("--spinup-days", type=int, default=90)
    parser.add_argument("--min-quality", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader_manifest = output_dir / "R20_loader_manifest_epochs0.json"
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

    rows = []
    with torch.no_grad():
        for config in _candidate_configs():
            for split, lake_ids in (("val", val_ids), ("heldout_diagnostic_only", heldout_ids)):
                for lake_id in lake_ids:
                    lake = lakes[lake_id]
                    mode = config["mode"]
                    print(f"R20 eval {config['config_id']} {split}: {lake_id}", flush=True)
                    metrics = evaluate_lake_free_roll(
                        model,
                        lake,
                        task_mode="analysis",
                        init_mode="zero_profile_low_dof",
                        spinup_days=int(args.spinup_days),
                        zero_profile_initializer="low_dof",
                        spinup_lswt_observer_mode=mode,
                        spinup_lst_assimilation_strength=float(config["strength"]),
                        spinup_lst_assimilation_decay_depth_m=float(config["decay_depth_m"]),
                        spinup_lst_assimilation_max_increment_c=float(config["max_increment_c"]),
                        rollout_lswt_observer_mode=mode,
                        lswt_observer_strength=float(config["strength"]),
                        lswt_observer_decay_depth_m=float(config["decay_depth_m"]),
                        lswt_observer_max_increment_c=float(config["max_increment_c"]),
                        lswt_observer_low_rank_deep_update_fraction=float(config["deep_update_fraction"]),
                        lswt_observer_heat_content_limit_c=float(config["heat_content_limit_c"]),
                        lswt_observer_min_quality=float(args.min_quality),
                        hard_density_stability=False,
                    )
                    row = {
                        **config,
                        "split": split,
                        "lake_id": lake_id,
                        "lake_group": _lake_group_id(lake),
                        "lake_type": _lake_reservoir_bucket(lake),
                        "rmse": metrics.get("rmse", np.nan),
                        "mae": metrics.get("mae", np.nan),
                        "bias": metrics.get("bias", np.nan),
                        "rmse_le25m": metrics.get("rmse_le25m", np.nan),
                        "rmse_gt25m": metrics.get("rmse_gt25m", np.nan),
                        "count_le25m": metrics.get("count_le25m", 0),
                        "count_gt25m": metrics.get("count_gt25m", 0),
                        "surface_rmse": metrics.get("observed_point_surface_rmse", np.nan),
                        "surface_bias": metrics.get("observed_point_surface_bias", np.nan),
                        "surface_count": metrics.get("observed_point_surface_count", 0),
                        "observed_point_rmse": metrics.get("observed_point_rmse", np.nan),
                        "observed_point_bias": metrics.get("observed_point_bias", np.nan),
                        "profile_count": metrics.get("n_profiles", 0),
                        "observer_update_count": metrics.get("lswt_observer_update_count", 0.0),
                        "filled_lst_strong_update_count": metrics.get("lswt_observer_filled_lst_used_count", 0.0),
                        "observer_mean_abs_delta_c": metrics.get("lswt_observer_mean_abs_delta_c", np.nan),
                        "observer_heat_content_delta_mean_c": metrics.get("lswt_observer_heat_content_delta_mean_c", np.nan),
                        "observer_deep_abs_delta_mean_c": metrics.get("lswt_observer_deep_abs_delta_mean_c", np.nan),
                        "observer_localization_depth_mean_m": metrics.get("lswt_observer_localization_depth_mean_m", np.nan),
                        "observer_reservoir_scale_mean": metrics.get("lswt_observer_reservoir_conservative_scale_mean", np.nan),
                        "observer_heat_content_bound_scale_mean": metrics.get("lswt_observer_heat_content_bound_scale_mean", np.nan),
                    }
                    rows.append(row)

    lake_metrics = pd.DataFrame.from_records(rows)
    lake_metrics_path = output_dir / "R20_conservative_surface_observer_lake_metrics.csv"
    lake_metrics.to_csv(lake_metrics_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for config_id, group in lake_metrics.groupby("config_id", sort=False):
        for split, split_group in group.groupby("split", sort=False):
            record = {
                "config_id": config_id,
                "split": split,
                "mode": split_group["mode"].iloc[0],
                "strength": float(split_group["strength"].iloc[0]),
                "heat_content_limit_c": float(split_group["heat_content_limit_c"].iloc[0]),
                "lake_count": int(len(split_group)),
                "rmse_mean": _finite_mean(split_group["rmse"]),
                "bias_mean": _finite_mean(split_group["bias"]),
                "rmse_le25m_mean": _finite_mean(split_group["rmse_le25m"]),
                "rmse_gt25m_mean": _finite_mean(split_group["rmse_gt25m"]),
                "surface_rmse_mean": _finite_mean(split_group["surface_rmse"]),
                "surface_bias_mean": _finite_mean(split_group["surface_bias"]),
                "observer_update_count": _finite_sum(split_group["observer_update_count"]),
                "filled_lst_strong_update_count": _finite_sum(split_group["filled_lst_strong_update_count"]),
                "observer_mean_abs_delta_c": _finite_mean(split_group["observer_mean_abs_delta_c"]),
                "observer_heat_content_delta_mean_c": _finite_mean(split_group["observer_heat_content_delta_mean_c"]),
                "observer_deep_abs_delta_mean_c": _finite_mean(split_group["observer_deep_abs_delta_mean_c"]),
                "observer_localization_depth_mean_m": _finite_mean(split_group["observer_localization_depth_mean_m"]),
                "natural_rmse_mean": _finite_mean(split_group.loc[split_group["lake_type"] == "natural", "rmse"]),
                "reservoir_rmse_mean": _finite_mean(split_group.loc[split_group["lake_type"] == "reservoir", "rmse"]),
                "natural_surface_rmse_mean": _finite_mean(
                    split_group.loc[split_group["lake_type"] == "natural", "surface_rmse"]
                ),
                "reservoir_surface_rmse_mean": _finite_mean(
                    split_group.loc[split_group["lake_type"] == "reservoir", "surface_rmse"]
                ),
            }
            summary_rows.append(record)
    summary = pd.DataFrame.from_records(summary_rows)
    summary_path = output_dir / "R20_conservative_surface_observer_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    val_summary = summary[summary["split"] == "val"].copy()
    best_val = None
    if not val_summary.empty:
        val_summary = val_summary.sort_values(["rmse_mean", "surface_rmse_mean"], na_position="last")
        best_val = val_summary.iloc[0].to_dict()

    payload = {
        "diagnostic_id": "RECON_R20_CONSERVATIVE_SURFACE_OBSERVER_EVAL_v1",
        "scope": "eval_only_no_training",
        "manifest": str(Path(args.manifest)),
        "loader_manifest_epochs0": str(loader_manifest),
        "checkpoint": str(Path(args.checkpoint)),
        "val_ids": val_ids,
        "heldout_diagnostic_only_ids": heldout_ids,
        "configs": _candidate_configs(),
        "best_val_by_whole_rmse": best_val,
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    summary_json_path = output_dir / "R20_conservative_surface_observer_summary.json"
    summary_json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = _write_report(
        output_dir,
        summary,
        best_val,
        {
            "lake_metrics_csv": lake_metrics_path,
            "summary_csv": summary_path,
            "summary_json": summary_json_path,
        },
    )
    print(f"Wrote {lake_metrics_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
