"""R22 eval-only autopsy for conservative shallow LSWT observer.

This script compares R21B epoch 7 and epoch 23 checkpoints without training.
It reuses existing zero-profile export/evaluation routines and the R19 update
autopsy helper. Heldout rows are diagnostic-only and are never used to select
gain, checkpoint, or reservoir scale.
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
from lake_pinn.state_reconstruction import (  # noqa: E402
    apply_lswt_observer_update,
    initialize_rollout_state,
)
from scripts.r19_observer_update_autopsy import (  # noqa: E402
    _json_safe,
    _run_lake_autopsy,
    _safe_mean,
    _summarize_group,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
R21B_BACKUP = (
    PIPELINE_ROOT
    / "remote_artifact_backups"
    / "RECON_R21B_CONSERVATIVE_SURFACE_OBSERVER_LONGDIAG_v2_20260613_completed"
)
DEFAULT_MANIFEST = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R16_full_eval_point_diagnostics_20260612"
    / "R16_local_R14_manifest_path_mapped.json"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "reports"
    / "failure_diagnosis"
    / "R22_conservative_surface_observer_generalization_autopsy_20260613"
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


def _finite_p50(values) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 50.0)) if arr.size else float("nan")


def _finite_p95(values) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 95.0)) if arr.size else float("nan")


def _checkpoint_specs() -> list[dict]:
    return [
        {
            "checkpoint_label": "epoch7_best_by_val_rolling",
            "checkpoint_epoch": 7,
            "checkpoint_path": R21B_BACKUP / "global_state_forecaster_epoch0007.pt",
        },
        {
            "checkpoint_label": "epoch23_final",
            "checkpoint_epoch": 23,
            "checkpoint_path": R21B_BACKUP / "global_state_forecaster_epoch0023.pt",
        },
    ]


def _eval_configs() -> list[dict]:
    configs = [
        {
            "config_id": "observer_off",
            "family": "on_off",
            "mode": "off",
            "strength": 0.0,
            "reservoir_scale": 0.0,
        },
        {
            "config_id": "observer_on_s020",
            "family": "on_off",
            "mode": "conservative_surface",
            "strength": 0.20,
            "reservoir_scale": 1.0,
        },
    ]
    for strength in (0.05, 0.10, 0.15, 0.20):
        configs.append(
            {
                "config_id": f"gain_s{int(strength * 1000):03d}",
                "family": "validation_gain_sweep",
                "mode": "conservative_surface",
                "strength": float(strength),
                "reservoir_scale": 1.0,
            }
        )
    for scale in (0.0, 0.25, 0.50, 1.0):
        configs.append(
            {
                "config_id": f"reservoir_scale_{int(scale * 100):03d}",
                "family": "validation_reservoir_scale_sweep",
                "mode": "conservative_surface",
                "strength": 0.20,
                "reservoir_scale": float(scale),
            }
        )
    return configs


def _effective_strength(config: dict, lake_type: str) -> float:
    strength = float(config["strength"])
    if config["family"] == "validation_reservoir_scale_sweep" and lake_type == "reservoir":
        return strength * float(config["reservoir_scale"])
    return strength


def _load_model(checkpoint: Path, manifest: Path, output_dir: Path, device: str):
    loader_manifest = output_dir / f"loader_manifest_{checkpoint.stem}_epochs0.json"
    manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
    manifest_payload["epochs"] = 0
    manifest_payload["export_after_training"] = "off"
    loader_manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = train_multilake_state_forecaster(
        loader_manifest,
        output_dir / f"bootstrap_{checkpoint.stem}_epochs0_loader",
        epochs=0,
        checkpoint_path=checkpoint,
        export_after_training="off",
        profile_runtime=False,
        device=device,
    )
    return result, loader_manifest


def _lake_metric_row(model, lake, split: str, checkpoint: dict, config: dict, *, spinup_days: int, min_quality: float) -> dict:
    lake_type = _lake_reservoir_bucket(lake)
    strength = _effective_strength(config, lake_type)
    mode = str(config["mode"])
    metrics = evaluate_lake_free_roll(
        model,
        lake,
        task_mode="analysis",
        init_mode="zero_profile_low_dof",
        spinup_days=int(spinup_days),
        zero_profile_initializer="low_dof",
        spinup_lswt_observer_mode=mode,
        spinup_lst_assimilation_strength=strength,
        spinup_lst_assimilation_decay_depth_m=1.5,
        spinup_lst_assimilation_max_increment_c=0.30 if mode != "off" else 0.0,
        rollout_lswt_observer_mode=mode,
        lswt_observer_strength=strength,
        lswt_observer_decay_depth_m=1.5,
        lswt_observer_max_increment_c=0.30 if mode != "off" else 0.0,
        lswt_observer_low_rank_deep_update_fraction=0.0,
        lswt_observer_heat_content_limit_c=0.05 if mode != "off" else 0.0,
        lswt_observer_min_quality=float(min_quality),
        hard_density_stability=False,
    )
    return {
        "checkpoint_label": checkpoint["checkpoint_label"],
        "checkpoint_epoch": int(checkpoint["checkpoint_epoch"]),
        "config_id": config["config_id"],
        "family": config["family"],
        "mode": mode,
        "strength": float(config["strength"]),
        "effective_strength": float(strength),
        "reservoir_scale": float(config["reservoir_scale"]),
        "split": split,
        "lake_id": lake["lake_id"],
        "lake_group": _lake_group_id(lake),
        "lake_type": lake_type,
        "rmse": metrics.get("rmse", np.nan),
        "bias": metrics.get("bias", np.nan),
        "mae": metrics.get("mae", np.nan),
        "rmse_le25m": metrics.get("rmse_le25m", np.nan),
        "bias_le25m": metrics.get("bias_le25m", np.nan),
        "count_le25m": metrics.get("count_le25m", 0),
        "rmse_gt25m": metrics.get("rmse_gt25m", np.nan),
        "bias_gt25m": metrics.get("bias_gt25m", np.nan),
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
        "observer_max_abs_delta_c": metrics.get("lswt_observer_max_abs_delta_c", np.nan),
        "observer_heat_content_delta_mean_c": metrics.get("lswt_observer_heat_content_delta_mean_c", np.nan),
        "observer_deep_abs_delta_mean_c": metrics.get("lswt_observer_deep_abs_delta_mean_c", np.nan),
        "observer_density_guard_scale_mean": metrics.get("lswt_observer_density_guard_scale_mean", np.nan),
        "observer_localization_depth_mean_m": metrics.get("lswt_observer_localization_depth_mean_m", np.nan),
        "observer_reservoir_scale_mean": metrics.get("lswt_observer_reservoir_conservative_scale_mean", np.nan),
        "observer_heat_content_bound_scale_mean": metrics.get("lswt_observer_heat_content_bound_scale_mean", np.nan),
    }


def _summarize_metrics(lake_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for group_values, group in lake_metrics.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = {col: value for col, value in zip(group_cols, group_values)}
        record.update(
            {
                "lake_count": int(len(group)),
                "whole_rmse_mean": _finite_mean(group["rmse"]),
                "whole_bias_mean": _finite_mean(group["bias"]),
                "surface_rmse_mean": _finite_mean(group["surface_rmse"]),
                "surface_bias_mean": _finite_mean(group["surface_bias"]),
                "le25m_rmse_mean": _finite_mean(group["rmse_le25m"]),
                "le25m_bias_mean": _finite_mean(group["bias_le25m"]),
                "gt25m_rmse_mean": _finite_mean(group["rmse_gt25m"]),
                "gt25m_bias_mean": _finite_mean(group["bias_gt25m"]),
                "profile_count_sum": _finite_sum(group["profile_count"]),
                "observer_update_count_sum": _finite_sum(group["observer_update_count"]),
                "filled_lst_strong_update_count_sum": _finite_sum(group["filled_lst_strong_update_count"]),
                "deep_abs_delta_mean_c": _finite_mean(group["observer_deep_abs_delta_mean_c"]),
                "mean_abs_delta_c": _finite_mean(group["observer_mean_abs_delta_c"]),
                "localization_depth_mean_m": _finite_mean(group["observer_localization_depth_mean_m"]),
                "density_guard_scale_mean": _finite_mean(group["observer_density_guard_scale_mean"]),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _run_autopsy_rows(model, lakes: dict, lake_ids: list[str], split: str, checkpoint: dict, *, spinup_days: int, min_quality: float) -> list[dict]:
    rows = []
    for lake_id in lake_ids:
        lake = lakes[lake_id]
        lake_type = _lake_reservoir_bucket(lake)
        strength = 0.20
        print(f"R22 update autopsy {checkpoint['checkpoint_label']} {split}: {lake_id}", flush=True)
        lake_rows = _run_lake_autopsy(
            model,
            lake,
            split_label=split,
            mode="conservative_surface",
            spinup_days=spinup_days,
            strength=strength,
            decay_depth_m=1.5,
            max_increment_c=0.30,
            low_rank_deep_update_fraction=0.0,
            heat_content_limit_c=0.05,
            min_quality=min_quality,
            init_mode="zero_profile_low_dof",
            zero_profile_initializer="low_dof",
            hard_density_stability=False,
        )
        for row in lake_rows:
            row["checkpoint_label"] = checkpoint["checkpoint_label"]
            row["checkpoint_epoch"] = int(checkpoint["checkpoint_epoch"])
            row["effective_strength"] = strength
            row["reservoir_scale_applied"] = 1.0 if lake_type == "reservoir" else 1.0
        rows.extend(lake_rows)
    return rows


def _run_kd_diagnostics(
    model,
    lake,
    split: str,
    checkpoint: dict,
    *,
    spinup_days: int,
    min_quality: float,
    kd_saturation_threshold: float,
) -> dict:
    model.eval()
    lake_type = _lake_reservoir_bucket(lake)
    init_state = initialize_rollout_state(
        model=model,
        df=lake["df"],
        depths=lake["depths_np"],
        all_lookup=lake["lookups"]["all"],
        forcing_rows=lake["forcing_rows"],
        static_features=lake["static_features"],
        metadata=lake["metadata"],
        device=lake["depths"].device,
        init_mode="zero_profile_low_dof",
        rollout_start_date=None,
        spinup_days=spinup_days,
        zero_profile_initializer="low_dof",
        spinup_lswt_observer_mode="conservative_surface",
        spinup_lst_assimilation_strength=0.20,
        spinup_lst_assimilation_decay_depth_m=1.5,
        spinup_lst_assimilation_max_increment_c=0.30,
        lswt_observer_low_rank_deep_update_fraction=0.0,
        lswt_observer_heat_content_limit_c=0.05,
        lswt_observer_min_quality=min_quality,
        task_mode="analysis",
        area_profile=lake["area"],
        hard_density_stability=False,
    )
    current = init_state["current"]
    freezing_storage = init_state.get("freezing_storage_j_m2", torch.zeros_like(current))
    rollout_start_idx = int(init_state["rollout_start_idx"])
    kd_values = []
    kd_base_values = []
    adaptive_kd_values = []
    residual_abs_values = []
    kz_mean_values = []
    with torch.no_grad():
        for day_idx in range(rollout_start_idx, len(lake["df"]) - 1):
            next_row = lake["forcing_rows"][day_idx + 1] if day_idx + 1 < len(lake["forcing_rows"]) else None
            current, freezing_storage, diagnostics = model.step(
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
                diagnostic_mode="loss",
            )
            for key, target in (
                ("nn_kd_multiplier", kd_values),
                ("kd_base", kd_base_values),
                ("adaptive_kd_multiplier", adaptive_kd_values),
                ("residual_abs_mean_c", residual_abs_values),
                ("kz_mean", kz_mean_values),
            ):
                value = diagnostics.get(key)
                if value is not None:
                    arr = value.detach().cpu().reshape(-1).numpy().astype(np.float64)
                    target.extend(arr[np.isfinite(arr)].tolist())
            if next_row is not None:
                current, _ = apply_lswt_observer_update(
                    current,
                    next_row,
                    lake["depths"],
                    mode="conservative_surface",
                    strength=0.20,
                    decay_depth_m=1.5,
                    max_increment_c=0.30,
                    low_rank_deep_update_fraction=0.0,
                    heat_content_limit_c=0.05,
                    min_quality=min_quality,
                    area_profile=lake["area"],
                    metadata=lake.get("metadata"),
                )
    kd_arr = np.asarray(kd_values, dtype=np.float64)
    kd_arr = kd_arr[np.isfinite(kd_arr)]
    return {
        "checkpoint_label": checkpoint["checkpoint_label"],
        "checkpoint_epoch": int(checkpoint["checkpoint_epoch"]),
        "split": split,
        "lake_id": lake["lake_id"],
        "lake_group": _lake_group_id(lake),
        "lake_type": lake_type,
        "kd_sample_count": int(kd_arr.size),
        "nn_kd_multiplier_mean": float(np.mean(kd_arr)) if kd_arr.size else float("nan"),
        "nn_kd_multiplier_p50": float(np.percentile(kd_arr, 50.0)) if kd_arr.size else float("nan"),
        "nn_kd_multiplier_p95": float(np.percentile(kd_arr, 95.0)) if kd_arr.size else float("nan"),
        "nn_kd_multiplier_saturation_fraction": (
            float(np.mean(kd_arr >= float(kd_saturation_threshold))) if kd_arr.size else float("nan")
        ),
        "kd_base_mean": _finite_mean(kd_base_values),
        "adaptive_kd_multiplier_mean": _finite_mean(adaptive_kd_values),
        "residual_abs_mean_c": _finite_mean(residual_abs_values),
        "kz_mean": _finite_mean(kz_mean_values),
    }


def _history_kd_summary(history_path: Path) -> pd.DataFrame:
    df = pd.read_csv(history_path)
    cols = [
        "epoch",
        "eval_mode",
        "nn_kd_multiplier_mean",
        "nn_kd_multiplier_p50",
        "nn_kd_multiplier_p95",
        "nn_kd_multiplier_saturation_fraction",
        "kd_saturation_penalty_loss",
        "kd_saturation_penalty_weighted_loss",
    ]
    available = [c for c in cols if c in df.columns]
    out = df.loc[df["epoch"].isin([7, 23]), available].copy()
    return out


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    onoff: pd.DataFrame,
    gain_sweep: pd.DataFrame,
    reservoir_sweep: pd.DataFrame,
    update_summary: pd.DataFrame,
    kd_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    lines = [
        "# R22 Conservative Surface Observer Generalization Autopsy",
        "",
        "Scope: local eval-only diagnostic. No training, no remote task, no split/manifest/_standard_inputs changes.",
        "",
        "Heldout rows are diagnostic-only and were not used for checkpoint, gain, or reservoir-scale selection.",
        "",
        "## Epoch 7 vs Epoch 23 on/off",
    ]
    for _, row in onoff.sort_values(["split", "checkpoint_epoch", "config_id"]).iterrows():
        lines.append(
            "- "
            f"epoch {int(row['checkpoint_epoch'])} {row['split']} {row['config_id']}: "
            f"whole={row['whole_rmse_mean']:.3f} C, "
            f"bias={row['whole_bias_mean']:.3f} C, "
            f"surface={row['surface_rmse_mean']:.3f} C, "
            f"le25={row['le25m_rmse_mean']:.3f} C, "
            f"gt25={row['gt25m_rmse_mean'] if np.isfinite(row['gt25m_rmse_mean']) else np.nan:.3f} C, "
            f"filled={row['filled_lst_strong_update_count_sum']:.0f}, "
            f"deep_delta={row['deep_abs_delta_mean_c']:.4f}"
        )
    lines.extend(["", "## Validation-only gain sweep"])
    for _, row in gain_sweep.sort_values(["checkpoint_epoch", "strength"]).iterrows():
        lines.append(
            "- "
            f"epoch {int(row['checkpoint_epoch'])} gain {row['strength']:.2f}: "
            f"whole={row['whole_rmse_mean']:.3f} C, "
            f"surface={row['surface_rmse_mean']:.3f} C, "
            f"bias={row['whole_bias_mean']:.3f} C"
        )
    lines.extend(["", "## Validation-only reservoir scale sweep"])
    for _, row in reservoir_sweep.sort_values(["checkpoint_epoch", "reservoir_scale"]).iterrows():
        lines.append(
            "- "
            f"epoch {int(row['checkpoint_epoch'])} reservoir_scale {row['reservoir_scale']:.2f}: "
            f"whole={row['whole_rmse_mean']:.3f} C, "
            f"surface={row['surface_rmse_mean']:.3f} C, "
            f"bias={row['whole_bias_mean']:.3f} C"
        )
    lines.extend(["", "## Update autopsy"])
    for _, row in update_summary.sort_values(["split", "lake_type", "checkpoint_epoch"]).iterrows():
        lines.append(
            "- "
            f"epoch {int(row['checkpoint_epoch'])} {row['split']} {row['lake_type']}: "
            f"applied={int(row['applied_count'])}, "
            f"filled={int(row['filled_lst_strong_update_count'])}, "
            f"surface_improve_rate={row['surface_improvement_rate']:.3f}, "
            f"innovation_abs_before/after={row['innovation_before_abs_mean']:.3f}/{row['innovation_after_abs_mean']:.3f}, "
            f"thermocline_cross={row['mld_over_thermocline_rate']:.3f}, "
            f"same_day_whole_delta={row['same_day_whole_profile_rmse_delta_mean']:.4f}"
        )
    lines.extend(["", "## Kd drift"])
    for _, row in kd_summary.sort_values(["checkpoint_epoch", "split", "lake_type"]).iterrows():
        lines.append(
            "- "
            f"epoch {int(row['checkpoint_epoch'])} {row['split']} {row['lake_type']}: "
            f"mean/p50/p95/sat="
            f"{row['nn_kd_multiplier_mean']:.3f}/"
            f"{row['nn_kd_multiplier_p50']:.3f}/"
            f"{row['nn_kd_multiplier_p95']:.3f}/"
            f"{row['nn_kd_multiplier_saturation_fraction']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation guards",
            "",
            "- Do not use heldout diagnostic-only metrics for tuning.",
            "- A reservoir observer scale recommendation may only be based on validation sweep plus safety diagnostics.",
            "- Kd p95 near the configured upper bound is treated as compensation risk even when saturation fraction is still zero.",
            "- This R22 autopsy is not a formal L3/L5/L7 claim.",
            "",
            "## Artifacts",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: {path}")
    report_path = output_dir / "R22_conservative_surface_observer_generalization_autopsy_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R22 conservative surface observer generalization autopsy.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-ids", default=DEFAULT_VAL_IDS)
    parser.add_argument("--heldout-ids", default=DEFAULT_HELDOUT_IDS)
    parser.add_argument("--spinup-days", type=int, default=90)
    parser.add_argument("--min-quality", type=float, default=0.05)
    parser.add_argument("--kd-saturation-threshold", type=float, default=1.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ids = _parse_ids(args.val_ids)
    heldout_ids = _parse_ids(args.heldout_ids)
    lake_metric_rows = []
    autopsy_rows = []
    kd_rows = []
    loader_manifests = []
    configs = _eval_configs()

    for checkpoint in _checkpoint_specs():
        checkpoint_path = Path(checkpoint["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        print(f"R22 loading {checkpoint['checkpoint_label']} from {checkpoint_path}", flush=True)
        result, loader_manifest = _load_model(checkpoint_path, Path(args.manifest), output_dir, device)
        loader_manifests.append(loader_manifest)
        model = result["model"]
        lakes = {lake["lake_id"]: lake for lake in result["lakes"]}
        missing = [lake_id for lake_id in val_ids + heldout_ids if lake_id not in lakes]
        if missing:
            raise ValueError(f"Requested lake IDs missing from manifest load: {missing}")

        with torch.no_grad():
            for config in configs:
                split_items = [("val", val_ids)]
                if config["family"] == "on_off":
                    split_items.append(("heldout_diagnostic_only", heldout_ids))
                for split, lake_ids in split_items:
                    for lake_id in lake_ids:
                        print(
                            f"R22 eval {checkpoint['checkpoint_label']} {config['config_id']} {split}: {lake_id}",
                            flush=True,
                        )
                        lake_metric_rows.append(
                            _lake_metric_row(
                                model,
                                lakes[lake_id],
                                split,
                                checkpoint,
                                config,
                                spinup_days=args.spinup_days,
                                min_quality=args.min_quality,
                            )
                        )

            for split, lake_ids in (("val", val_ids), ("heldout_diagnostic_only", heldout_ids)):
                autopsy_rows.extend(
                    _run_autopsy_rows(
                        model,
                        lakes,
                        lake_ids,
                        split,
                        checkpoint,
                        spinup_days=args.spinup_days,
                        min_quality=args.min_quality,
                    )
                )
                for lake_id in lake_ids:
                    print(f"R22 Kd diagnostic {checkpoint['checkpoint_label']} {split}: {lake_id}", flush=True)
                    kd_rows.append(
                        _run_kd_diagnostics(
                            model,
                            lakes[lake_id],
                            split,
                            checkpoint,
                            spinup_days=args.spinup_days,
                            min_quality=args.min_quality,
                            kd_saturation_threshold=args.kd_saturation_threshold,
                        )
                    )

    lake_metrics = pd.DataFrame.from_records(lake_metric_rows)
    lake_metrics_path = output_dir / "R22_lake_metrics.csv"
    lake_metrics.to_csv(lake_metrics_path, index=False, encoding="utf-8-sig")

    summary = _summarize_metrics(
        lake_metrics,
        ["checkpoint_label", "checkpoint_epoch", "config_id", "family", "mode", "strength", "reservoir_scale", "split"],
    )
    summary_path = output_dir / "R22_summary_by_config_split.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lake_type_summary = _summarize_metrics(
        lake_metrics,
        [
            "checkpoint_label",
            "checkpoint_epoch",
            "config_id",
            "family",
            "mode",
            "strength",
            "reservoir_scale",
            "split",
            "lake_type",
        ],
    )
    lake_type_summary_path = output_dir / "R22_summary_by_lake_type.csv"
    lake_type_summary.to_csv(lake_type_summary_path, index=False, encoding="utf-8-sig")

    lake_group_summary = _summarize_metrics(
        lake_metrics,
        [
            "checkpoint_label",
            "checkpoint_epoch",
            "config_id",
            "family",
            "mode",
            "strength",
            "reservoir_scale",
            "split",
            "lake_group",
        ],
    )
    lake_group_summary_path = output_dir / "R22_summary_by_lake_group.csv"
    lake_group_summary.to_csv(lake_group_summary_path, index=False, encoding="utf-8-sig")

    onoff = summary[summary["family"] == "on_off"].copy()
    onoff_path = output_dir / "R22_epoch7_epoch23_observer_on_off_comparison.csv"
    onoff.to_csv(onoff_path, index=False, encoding="utf-8-sig")

    gain_sweep = summary[
        (summary["family"] == "validation_gain_sweep") & (summary["split"] == "val")
    ].copy()
    gain_sweep_path = output_dir / "R22_validation_only_gain_sweep.csv"
    gain_sweep.to_csv(gain_sweep_path, index=False, encoding="utf-8-sig")

    reservoir_sweep = summary[
        (summary["family"] == "validation_reservoir_scale_sweep") & (summary["split"] == "val")
    ].copy()
    reservoir_sweep_path = output_dir / "R22_validation_only_reservoir_scale_sweep.csv"
    reservoir_sweep.to_csv(reservoir_sweep_path, index=False, encoding="utf-8-sig")

    updates = pd.DataFrame.from_records(autopsy_rows)
    updates_path = output_dir / "R22_observer_update_autopsy_updates.csv"
    updates.to_csv(updates_path, index=False, encoding="utf-8-sig")

    update_summary = _summarize_group(updates, ["checkpoint_label", "checkpoint_epoch", "split", "lake_type"])
    update_summary_path = output_dir / "R22_observer_update_autopsy_by_split_lake_type.csv"
    update_summary.to_csv(update_summary_path, index=False, encoding="utf-8-sig")

    update_lake_summary = _summarize_group(updates, ["checkpoint_label", "checkpoint_epoch", "split", "lake_id", "lake_type"])
    update_lake_summary_path = output_dir / "R22_observer_update_autopsy_by_lake.csv"
    update_lake_summary.to_csv(update_lake_summary_path, index=False, encoding="utf-8-sig")

    kd_lake = pd.DataFrame.from_records(kd_rows)
    kd_lake_path = output_dir / "R22_kd_diagnostics_by_lake.csv"
    kd_lake.to_csv(kd_lake_path, index=False, encoding="utf-8-sig")
    kd_summary = (
        kd_lake.groupby(["checkpoint_label", "checkpoint_epoch", "split", "lake_type"], dropna=False)
        .agg(
            lake_count=("lake_id", "count"),
            nn_kd_multiplier_mean=("nn_kd_multiplier_mean", _finite_mean),
            nn_kd_multiplier_p50=("nn_kd_multiplier_p50", _finite_p50),
            nn_kd_multiplier_p95=("nn_kd_multiplier_p95", _finite_p95),
            nn_kd_multiplier_saturation_fraction=("nn_kd_multiplier_saturation_fraction", _finite_mean),
            kd_base_mean=("kd_base_mean", _finite_mean),
            adaptive_kd_multiplier_mean=("adaptive_kd_multiplier_mean", _finite_mean),
            residual_abs_mean_c=("residual_abs_mean_c", _finite_mean),
            kz_mean=("kz_mean", _finite_mean),
        )
        .reset_index()
    )
    kd_summary_path = output_dir / "R22_kd_diagnostics_by_epoch_split_lake_type.csv"
    kd_summary.to_csv(kd_summary_path, index=False, encoding="utf-8-sig")

    history_kd = _history_kd_summary(R21B_BACKUP / "global_state_forecaster_training_history.csv")
    history_kd_path = output_dir / "R22_history_kd_epoch7_epoch23.csv"
    history_kd.to_csv(history_kd_path, index=False, encoding="utf-8-sig")

    validation_gain_rank = gain_sweep.sort_values(["whole_rmse_mean", "surface_rmse_mean"], na_position="last")
    validation_reservoir_rank = reservoir_sweep.sort_values(["whole_rmse_mean", "surface_rmse_mean"], na_position="last")
    best_gain = validation_gain_rank.iloc[0].to_dict() if not validation_gain_rank.empty else None
    best_reservoir_scale = validation_reservoir_rank.iloc[0].to_dict() if not validation_reservoir_rank.empty else None

    latest_kd = kd_summary[kd_summary["checkpoint_epoch"] == 23].copy()
    kd_risk = bool(
        (pd.to_numeric(latest_kd["nn_kd_multiplier_p95"], errors="coerce") >= 1.23).any()
        or (pd.to_numeric(history_kd.get("nn_kd_multiplier_p95", pd.Series(dtype=float)), errors="coerce") >= 1.23).any()
    )
    filled_count = _finite_sum(updates.get("observer_filled_lst_used_count", pd.Series(dtype=float)))
    deep_delta = _safe_mean(updates.get("deep_abs_delta_c", pd.Series(dtype=float)))
    validation_on = onoff[(onoff["split"] == "val") & (onoff["config_id"] == "observer_on_s020")]
    heldout_on = onoff[
        (onoff["split"] == "heldout_diagnostic_only") & (onoff["config_id"] == "observer_on_s020")
    ]
    r23_go = False
    r23_reason = (
        "No-Go: R22 remains diagnostic-only; heldout zero-profile is unstable, "
        "few-shot/export mismatch remains from R21B, and Kd p95 is close to the upper bound."
    )
    summary_payload = {
        "diagnostic_id": "RECON_R22_CONSERVATIVE_SURFACE_OBSERVER_GENERALIZATION_AUTOPSY_v1",
        "scope": "eval_only_no_training_no_remote_no_split_manifest_standard_inputs_change",
        "manifest": str(Path(args.manifest)),
        "checkpoints": [
            {**spec, "checkpoint_path": str(spec["checkpoint_path"])}
            for spec in _checkpoint_specs()
        ],
        "val_ids": val_ids,
        "heldout_diagnostic_only_ids": heldout_ids,
        "validation_only_best_gain_by_whole_rmse": _json_safe(best_gain),
        "validation_only_best_reservoir_scale_by_whole_rmse": _json_safe(best_reservoir_scale),
        "observer_safety": {
            "filled_lst_strong_update_count": int(filled_count),
            "deep_abs_delta_mean_c": _json_safe(deep_delta),
        },
        "kd_compensation_risk": kd_risk,
        "r23_short_diagnostic_training_go": r23_go,
        "r23_reason": r23_reason,
        "validation_observer_on_rows": validation_on.replace({np.nan: None}).to_dict(orient="records"),
        "heldout_observer_on_rows_diagnostic_only": heldout_on.replace({np.nan: None}).to_dict(orient="records"),
        "artifacts": {
            "lake_metrics": str(lake_metrics_path),
            "summary": str(summary_path),
            "lake_type_summary": str(lake_type_summary_path),
            "lake_group_summary": str(lake_group_summary_path),
            "onoff": str(onoff_path),
            "gain_sweep": str(gain_sweep_path),
            "reservoir_sweep": str(reservoir_sweep_path),
            "updates": str(updates_path),
            "update_summary": str(update_summary_path),
            "kd_lake": str(kd_lake_path),
            "kd_summary": str(kd_summary_path),
            "history_kd": str(history_kd_path),
        },
    }
    summary_json_path = output_dir / "R22_conservative_surface_observer_generalization_autopsy_summary.json"
    summary_json_path.write_text(
        json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = _write_report(
        output_dir,
        summary,
        onoff,
        gain_sweep,
        reservoir_sweep,
        update_summary,
        kd_summary,
        {
            "lake_metrics": lake_metrics_path,
            "summary_by_config_split": summary_path,
            "summary_by_lake_type": lake_type_summary_path,
            "summary_by_lake_group": lake_group_summary_path,
            "on_off_comparison": onoff_path,
            "gain_sweep": gain_sweep_path,
            "reservoir_scale_sweep": reservoir_sweep_path,
            "update_rows": updates_path,
            "update_summary": update_summary_path,
            "kd_summary": kd_summary_path,
            "summary_json": summary_json_path,
        },
    )
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
