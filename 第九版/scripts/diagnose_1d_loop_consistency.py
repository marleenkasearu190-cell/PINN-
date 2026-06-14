"""Diagnose one-day closed-loop consistency for a trained RECON checkpoint.

This is a read-only evaluation step: it loads an existing manifest and
checkpoint, evaluates teacher-forced transition pairs and rolling-start
closed-loop predictions on the same lake data, then writes CSV/JSON/MD
diagnostics.  It does not train, change splits, or select checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from lake_pinn.state_model import LakeStateForecaster, STATIC_FEATURE_DIM
from lake_pinn.state_multilake import (
    _filter_state_forecaster_state_dict_for_load,
    _resolve_heldout_selection,
    evaluate_lake_pair_horizons,
    evaluate_lakes_rolling_start_horizons,
    prepare_lake_state_data,
)


EXPERIMENT_ID = "RECON_DIAG_1D_LOOP_CONSISTENCY_v1"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_int_list(value: str | Iterable[int], *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(item).strip() for item in value]
    parsed = tuple(sorted({int(part) for part in parts if int(part) > 0}))
    if not parsed:
        raise ValueError("At least one positive horizon is required.")
    return parsed


def _parse_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _finite(values: Iterable[float]) -> list[float]:
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def _mean(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else float("nan")


def _sum_counts(values: Iterable[float]) -> int:
    total = 0
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            total += int(number)
    return int(total)


def _ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(float(numerator)) and math.isfinite(float(denominator))):
        return float("nan")
    if abs(float(denominator)) < 1.0e-12:
        return float("nan")
    return float(numerator) / float(denominator)


def classify_consistency(
    transition_rmse_1d: float,
    rolling_rmse_1d: float,
    *,
    pass_ratio: float = 1.35,
    warn_ratio: float = 2.0,
    pass_delta_c: float = 0.35,
    warn_delta_c: float = 1.0,
) -> str:
    """Classify whether the one-day closed-loop path matches pair evaluation."""
    if not (math.isfinite(float(transition_rmse_1d)) and math.isfinite(float(rolling_rmse_1d))):
        return "inconclusive"
    delta = float(rolling_rmse_1d) - float(transition_rmse_1d)
    ratio = _ratio(rolling_rmse_1d, transition_rmse_1d)
    if ratio <= float(pass_ratio) or delta <= float(pass_delta_c):
        return "pass"
    if ratio <= float(warn_ratio) or delta <= float(warn_delta_c):
        return "warn"
    return "fail"


def _device_from_arg(value: str | None) -> torch.device:
    if value and value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_load(path: Path, *, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dict-like payload: {path}")
    return checkpoint


def _config_value(checkpoint: dict, manifest: dict, key: str, default):
    if key in checkpoint and checkpoint[key] is not None:
        return checkpoint[key]
    if key in manifest and manifest[key] is not None:
        return manifest[key]
    return default


def _build_model(
    *,
    checkpoint: dict,
    manifest: dict,
    seed_lake: dict,
    device: torch.device,
) -> LakeStateForecaster:
    static_dim = int(seed_lake["static_features"].reshape(-1).numel())
    if static_dim != STATIC_FEATURE_DIM:
        raise ValueError(
            f"Static feature vector has {static_dim} values; expected {STATIC_FEATURE_DIM}."
        )
    model = LakeStateForecaster(
        seed_lake["depths_np"],
        seed_lake["area_np"],
        static_dim=static_dim,
        residual_limit_c=float(_config_value(checkpoint, manifest, "residual_limit_c", 0.50)),
        wind_kz_scale=float(_config_value(checkpoint, manifest, "wind_kz_scale", 1.0)),
        autumn_convective_boost=float(_config_value(checkpoint, manifest, "autumn_convective_boost", 1.0)),
        lst_feature_dropout_probability=float(
            _config_value(checkpoint, manifest, "lst_feature_dropout_probability", 0.0)
        ),
        turbulent_flux_mode=str(_config_value(checkpoint, manifest, "turbulent_flux_mode", "bulk")),
        turbulent_flux_blend_alpha=float(_config_value(checkpoint, manifest, "turbulent_flux_blend_alpha", 0.3)),
        freezing_energy_mode=str(_config_value(checkpoint, manifest, "freezing_energy_mode", "latent_reservoir")),
        advective_heat_source_mode=str(
            _config_value(checkpoint, manifest, "advective_heat_source_mode", "reservoir_simple")
        ),
        shape_aware_mixing=str(_config_value(checkpoint, manifest, "shape_aware_mixing", "on")),
        shape_mixing_strength=float(_config_value(checkpoint, manifest, "shape_mixing_strength", 0.35)),
        stratification_mixing_cap=str(_config_value(checkpoint, manifest, "stratification_mixing_cap", "on")),
        stratification_mixing_cap_strength=float(
            _config_value(checkpoint, manifest, "stratification_mixing_cap_strength", 1.0)
        ),
        lake_adaptive_params=str(_config_value(checkpoint, manifest, "lake_adaptive_params", "off")),
        lake_adaptive_hidden_dim=int(_config_value(checkpoint, manifest, "lake_adaptive_hidden_dim", 64)),
        lake_adaptive_init_spread=float(_config_value(checkpoint, manifest, "lake_adaptive_init_spread", 0.02)),
        lake_adaptive_temporal_mode=str(_config_value(checkpoint, manifest, "lake_adaptive_temporal_mode", "off")),
        lake_adaptive_temporal_init_spread=float(
            _config_value(checkpoint, manifest, "lake_adaptive_temporal_init_spread", 0.005)
        ),
        lake_adaptive_temporal_scale=float(
            _config_value(checkpoint, manifest, "lake_adaptive_temporal_scale", 0.25)
        ),
        fewshot_hidden_dim=int(_config_value(checkpoint, manifest, "fewshot_hidden_dim", 64)),
        fewshot_init_spread=float(_config_value(checkpoint, manifest, "fewshot_init_spread", 0.005)),
        fewshot_initial_delta_limit_c=float(
            _config_value(checkpoint, manifest, "fewshot_initial_delta_limit_c", 2.0)
        ),
        fewshot_adapter_scale=float(_config_value(checkpoint, manifest, "fewshot_adapter_scale", 0.25)),
        fewshot_adapter_params=str(
            _config_value(checkpoint, manifest, "fewshot_adapter_params", "kz,kd,exchange,convective,ice")
        ),
        adaptive_wind_kz_min=float(_config_value(checkpoint, manifest, "adaptive_wind_kz_min", 0.4)),
        adaptive_wind_kz_max=float(_config_value(checkpoint, manifest, "adaptive_wind_kz_max", 3.0)),
        adaptive_blend_alpha_min=float(_config_value(checkpoint, manifest, "adaptive_blend_alpha_min", 0.0)),
        adaptive_blend_alpha_max=float(_config_value(checkpoint, manifest, "adaptive_blend_alpha_max", 0.6)),
        adaptive_kd_multiplier_min=float(_config_value(checkpoint, manifest, "adaptive_kd_multiplier_min", 0.4)),
        adaptive_kd_multiplier_max=float(_config_value(checkpoint, manifest, "adaptive_kd_multiplier_max", 2.0)),
        adaptive_turbulent_exchange_scale_min=float(
            _config_value(checkpoint, manifest, "adaptive_turbulent_exchange_scale_min", 0.5)
        ),
        adaptive_turbulent_exchange_scale_max=float(
            _config_value(checkpoint, manifest, "adaptive_turbulent_exchange_scale_max", 1.8)
        ),
        adaptive_convective_mixing_scale_min=float(
            _config_value(checkpoint, manifest, "adaptive_convective_mixing_scale_min", 0.3)
        ),
        adaptive_convective_mixing_scale_max=float(
            _config_value(checkpoint, manifest, "adaptive_convective_mixing_scale_max", 2.5)
        ),
        adaptive_ice_shortwave_scale_min=float(
            _config_value(checkpoint, manifest, "adaptive_ice_shortwave_scale_min", 0.4)
        ),
        adaptive_ice_shortwave_scale_max=float(
            _config_value(checkpoint, manifest, "adaptive_ice_shortwave_scale_max", 1.8)
        ),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _filter_state_forecaster_state_dict_for_load(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _prepare_lakes(manifest: dict, *, device: torch.device) -> list[dict]:
    depth_points = int(manifest.get("depth_points", 40))
    max_rollout_days = int(manifest.get("max_rollout_days", manifest.get("segment_rollout_max_days", 45)))
    segment_rollout_max_days = int(manifest.get("segment_rollout_max_days", max_rollout_days))
    history_window_days = int(manifest.get("history_window_days", 30))
    episodic_max_query = int(manifest.get("episodic_fewshot_max_query_days", 120))
    split_mode = str(manifest.get("split_mode", "time_blocked"))
    return [
        prepare_lake_state_data(
            lake_config,
            split_mode=split_mode,
            task_mode="analysis",
            data_fill_mode="reconstruction",
            depth_points=depth_points,
            max_rollout_days=max_rollout_days,
            segment_rollout_max_days=segment_rollout_max_days,
            episodic_fewshot_max_query_days=episodic_max_query,
            history_window_days=history_window_days,
            device=device,
        )
        for lake_config in manifest.get("lakes", [])
    ]


def _ordered_lakes_by_id(lakes: list[dict], lake_ids: Iterable[str]) -> list[dict]:
    by_id = {lake["lake_id"]: lake for lake in lakes}
    return [by_id[lake_id] for lake_id in lake_ids if lake_id in by_id]


def _maybe_limit(lakes: list[dict], limit: int) -> list[dict]:
    if limit and int(limit) > 0:
        return lakes[: int(limit)]
    return lakes


def _evaluate_split(
    *,
    model: LakeStateForecaster,
    split_name: str,
    lakes: list[dict],
    pair_split: str,
    rolling_lookup_split: str,
    horizons: tuple[int, ...],
    max_start_profiles: int,
    hard_density_stability: bool,
    rolling_batch_size: int,
    use_batched_rolling: bool,
) -> list[dict]:
    if not lakes:
        return []
    rolling = evaluate_lakes_rolling_start_horizons(
        model,
        lakes,
        horizons=horizons,
        task_mode="analysis",
        max_start_profiles=max_start_profiles,
        hard_density_stability=hard_density_stability,
        lookup_split=rolling_lookup_split,
        batch_size=rolling_batch_size,
        use_batched=use_batched_rolling,
    )
    rows = []
    for lake in lakes:
        transition = evaluate_lake_pair_horizons(
            model,
            lake,
            lake["pairs"][pair_split],
            horizons=horizons,
            task_mode="analysis",
            hard_density_stability=hard_density_stability,
        )
        row = {
            "split": split_name,
            "lake_id": lake["lake_id"],
            "pair_split": pair_split,
            "rolling_lookup_split": rolling_lookup_split,
            "profile_pair_count": len(lake["pairs"][pair_split]),
        }
        rolling_record = rolling.get(lake["lake_id"], {})
        for horizon in horizons:
            transition_rmse = float(transition.get(f"rmse_{horizon}d", float("nan")))
            rolling_rmse = float(rolling_record.get(f"rmse_{horizon}d", float("nan")))
            row[f"transition_rmse_{horizon}d"] = transition_rmse
            row[f"transition_count_{horizon}d"] = int(transition.get(f"count_{horizon}d", 0) or 0)
            row[f"rolling_rmse_{horizon}d"] = rolling_rmse
            row[f"rolling_count_{horizon}d"] = int(rolling_record.get(f"count_{horizon}d", 0) or 0)
            row[f"rolling_minus_transition_{horizon}d"] = (
                rolling_rmse - transition_rmse
                if math.isfinite(rolling_rmse) and math.isfinite(transition_rmse)
                else float("nan")
            )
            row[f"rolling_to_transition_ratio_{horizon}d"] = _ratio(rolling_rmse, transition_rmse)
        row["status_1d"] = classify_consistency(
            row.get("transition_rmse_1d", float("nan")),
            row.get("rolling_rmse_1d", float("nan")),
        )
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict], *, horizons: tuple[int, ...]) -> list[dict]:
    summaries = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        summary = {
            "split": split,
            "lake_count": len(split_rows),
            "status_1d": "inconclusive",
        }
        for horizon in horizons:
            transition = _mean(row.get(f"transition_rmse_{horizon}d") for row in split_rows)
            rolling = _mean(row.get(f"rolling_rmse_{horizon}d") for row in split_rows)
            summary[f"transition_rmse_{horizon}d"] = transition
            summary[f"rolling_rmse_{horizon}d"] = rolling
            summary[f"rolling_minus_transition_{horizon}d"] = (
                rolling - transition
                if math.isfinite(rolling) and math.isfinite(transition)
                else float("nan")
            )
            summary[f"rolling_to_transition_ratio_{horizon}d"] = _ratio(rolling, transition)
            summary[f"transition_count_{horizon}d"] = _sum_counts(
                row.get(f"transition_count_{horizon}d") for row in split_rows
            )
            summary[f"rolling_count_{horizon}d"] = _sum_counts(
                row.get(f"rolling_count_{horizon}d") for row in split_rows
            )
        summary["status_1d"] = classify_consistency(
            summary.get("transition_rmse_1d", float("nan")),
            summary.get("rolling_rmse_1d", float("nan")),
        )
        summaries.append(summary)
    return summaries


def _overall_status(summary_rows: list[dict]) -> str:
    by_split = {row["split"]: row for row in summary_rows}
    primary = by_split.get("checkpoint_validation") or by_split.get("declared_validation_lakes")
    if primary is None:
        return "inconclusive"
    if primary["status_1d"] == "fail":
        return "fail"
    heldout = by_split.get("heldout_diagnostic")
    if heldout and heldout["status_1d"] == "fail":
        return "warn"
    return primary["status_1d"]


def _write_markdown_report(
    path: Path,
    *,
    payload: dict,
    summary_rows: list[dict],
    csv_path: Path,
    summary_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {payload['experiment_id']} closeout",
        "",
        "## Hypothesis Tested",
        "Check whether 1-day closed-loop rolling-start evaluation is consistent with teacher-forced transition-pair evaluation for the same trained checkpoint.",
        "",
        "## Decision",
        f"- Status: `{payload['status']}`",
        f"- Source experiment: `{payload['source_experiment_id']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        "- This is diagnostic-only; it does not select a checkpoint and does not support a formal L3 transfer claim by itself.",
        "",
        "## Split Summary",
        "| split | lakes | transition 1d RMSE | rolling 1d RMSE | ratio | delta C | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {split} | {lake_count} | {transition:.3f} | {rolling:.3f} | {ratio:.3f} | {delta:.3f} | {status} |".format(
                split=row["split"],
                lake_count=int(row["lake_count"]),
                transition=float(row.get("transition_rmse_1d", float("nan"))),
                rolling=float(row.get("rolling_rmse_1d", float("nan"))),
                ratio=float(row.get("rolling_to_transition_ratio_1d", float("nan"))),
                delta=float(row.get("rolling_minus_transition_1d", float("nan"))),
                status=row.get("status_1d", "inconclusive"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `checkpoint_validation` mirrors the existing training history semantics: non-heldout training lakes evaluated on their time-block validation profiles.",
            "- `declared_validation_lakes` is reported separately because LOCAL34 manifests name validation lake-years, but the trainer's checkpoint validation still uses time-block validation over all non-heldout training lakes.",
            "- `heldout_diagnostic` is reported only as a diagnosis. It must not drive checkpoint selection or tuning.",
            "",
            "## Artifacts",
            f"- By-lake CSV: `{csv_path}`",
            f"- Summary JSON: `{summary_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostic_manifest(
    path: Path,
    *,
    source_manifest: Path,
    checkpoint_path: Path,
    output_dir: Path,
    summary_path: Path,
    report_path: Path | None,
    payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT_ID,
        "level": "diagnostic_eval",
        "hypothesis": "One-day closed-loop rolling-start metrics should be close to one-day teacher-forced transition metrics if the training and inference loops are consistent.",
        "source_manifest": str(source_manifest),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "report": str(report_path) if report_path else "",
        "checkpoint_selection": "none_diagnostic_only",
        "heldout_policy": "heldout metrics are diagnostic only and are not used for checkpoint selection or tuning",
        "primary_metric": "checkpoint_validation rolling_to_transition_ratio_1d",
        "status": payload["status"],
        "created_at": payload["created_at"],
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    source_manifest: Path,
    checkpoint_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...],
    max_start_profiles: int,
    rolling_batch_size: int,
    device_arg: str,
    max_lakes_per_split: int = 0,
    include_train_split: bool = False,
    report_path: Path | None = None,
    diagnostic_manifest_path: Path | None = None,
) -> dict:
    source_manifest = Path(source_manifest)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(source_manifest)
    device = _device_from_arg(device_arg)
    checkpoint = _checkpoint_load(checkpoint_path, device=device)
    lakes = _prepare_lakes(manifest, device=device)
    if not lakes:
        raise ValueError("No lakes were prepared from the source manifest.")
    model = _build_model(
        checkpoint=checkpoint,
        manifest=manifest,
        seed_lake=lakes[0],
        device=device,
    )
    hard_density = bool(_config_value(checkpoint, manifest, "hard_density_stability_active", True))
    selection = _resolve_heldout_selection(lakes, manifest=manifest)
    train_lakes = selection["train_lakes"]
    heldout_lakes = selection["heldout_lakes"]
    declared_val_lakes = _ordered_lakes_by_id(lakes, _parse_string_list(manifest.get("val_lake_ids")))
    declared_train_lakes = _ordered_lakes_by_id(lakes, _parse_string_list(manifest.get("train_lake_ids")))
    if not declared_train_lakes:
        declared_train_lakes = train_lakes

    rows = []
    if include_train_split:
        rows.extend(
            _evaluate_split(
                model=model,
                split_name="train_pairs",
                lakes=_maybe_limit(declared_train_lakes, max_lakes_per_split),
                pair_split="train",
                rolling_lookup_split="train",
                horizons=horizons,
                max_start_profiles=max_start_profiles,
                hard_density_stability=hard_density,
                rolling_batch_size=rolling_batch_size,
                use_batched_rolling=True,
            )
        )
    rows.extend(
        _evaluate_split(
            model=model,
            split_name="checkpoint_validation",
            lakes=_maybe_limit(train_lakes, max_lakes_per_split),
            pair_split="val",
            rolling_lookup_split="val",
            horizons=horizons,
            max_start_profiles=max_start_profiles,
            hard_density_stability=hard_density,
            rolling_batch_size=rolling_batch_size,
            use_batched_rolling=True,
        )
    )
    if declared_val_lakes:
        rows.extend(
            _evaluate_split(
                model=model,
                split_name="declared_validation_lakes",
                lakes=_maybe_limit(declared_val_lakes, max_lakes_per_split),
                pair_split="val",
                rolling_lookup_split="val",
                horizons=horizons,
                max_start_profiles=max_start_profiles,
                hard_density_stability=hard_density,
                rolling_batch_size=rolling_batch_size,
                use_batched_rolling=True,
            )
        )
    rows.extend(
        _evaluate_split(
            model=model,
            split_name="heldout_diagnostic",
            lakes=_maybe_limit(heldout_lakes, max_lakes_per_split),
            pair_split="all",
            rolling_lookup_split="all",
            horizons=horizons,
            max_start_profiles=max_start_profiles,
            hard_density_stability=hard_density,
            rolling_batch_size=rolling_batch_size,
            use_batched_rolling=True,
        )
    )
    summary_rows = summarize_rows(rows, horizons=horizons)
    status = _overall_status(summary_rows)
    by_lake_csv = output_dir / "diagnostic_1d_loop_consistency_by_lake.csv"
    summary_json = output_dir / "diagnostic_1d_loop_consistency_summary.json"
    pd.DataFrame(rows).to_csv(by_lake_csv, index=False)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": manifest.get("experiment_id") or manifest.get("experiment") or "",
        "source_manifest": str(source_manifest),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "horizons": list(horizons),
        "max_start_profiles": int(max_start_profiles),
        "rolling_batch_size": int(rolling_batch_size),
        "status": status,
        "summary": summary_rows,
        "by_lake_csv": str(by_lake_csv),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if report_path is not None:
        _write_markdown_report(
            Path(report_path),
            payload=payload,
            summary_rows=summary_rows,
            csv_path=by_lake_csv,
            summary_path=summary_json,
        )
        payload["report_path"] = str(report_path)
    if diagnostic_manifest_path is not None:
        _write_diagnostic_manifest(
            Path(diagnostic_manifest_path),
            source_manifest=source_manifest,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            summary_path=summary_json,
            report_path=report_path,
            payload=payload,
        )
        payload["diagnostic_manifest_path"] = str(diagnostic_manifest_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RECON one-day loop-consistency diagnostics.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--max-start-profiles", type=int, default=16)
    parser.add_argument("--rolling-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-lakes-per-split", type=int, default=0)
    parser.add_argument("--include-train-split", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--diagnostic-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_diagnostic(
        source_manifest=args.source_manifest,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        horizons=_parse_int_list(args.horizons, default=(1, 3, 7)),
        max_start_profiles=args.max_start_profiles,
        rolling_batch_size=args.rolling_batch_size,
        device_arg=args.device,
        max_lakes_per_split=args.max_lakes_per_split,
        include_train_split=args.include_train_split,
        report_path=args.report_path,
        diagnostic_manifest_path=args.diagnostic_manifest,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
