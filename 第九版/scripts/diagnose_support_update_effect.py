"""Diagnose whether few-shot support updates improve query-start state.

This is a read-only diagnostic. It loads an existing manifest and checkpoint,
inspects prebuilt support/query episodes, and measures whether the few-shot
initial-state delta moves the query-start prior profile toward the observed
query-start profile. It does not train, select checkpoints, change splits, or
export predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from lake_pinn.state_multilake import (
    _resolve_heldout_selection,
    _select_episodic_fewshot_sequences,
    _target_tensor_and_mask,
)
from lake_pinn.state_reconstruction import build_lst_profile_prior
from scripts.diagnose_1d_loop_consistency import (
    _build_model,
    _checkpoint_load,
    _device_from_arg,
    _json_default,
    _ordered_lakes_by_id,
    _parse_string_list,
    _prepare_lakes,
    _read_json,
)


EXPERIMENT_ID = "RECON_DIAG_SUPPORT_UPDATE_EFFECT_v1"


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _finite_sum(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.sum()) if arr.size else 0.0


def _rmse_from_sse(sse: float, count: float) -> float:
    if not math.isfinite(float(sse)) or float(count) <= 0.0:
        return float("nan")
    return float(math.sqrt(float(sse) / float(count)))


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().double().cpu().numpy().reshape(-1)


def _masked_arrays(prediction, target, mask):
    pred = _to_numpy(prediction)
    truth = _to_numpy(target)
    valid = np.isfinite(pred) & np.isfinite(truth)
    if mask is not None:
        valid &= _to_numpy(mask.to(dtype=torch.float32)).astype(bool)
    return pred[valid], truth[valid], valid


def _lake_group(lake: dict) -> str:
    metadata = lake.get("metadata") or {}
    return str(metadata.get("lake_group") or lake.get("lake_group") or lake.get("lake_id", "")).strip()


def _selected_sequences(lake: dict, *, lookup_split: str, active_max_days: int, max_episodes: int) -> list:
    sequences = list(lake.get("episodic_fewshot_sequences", {}).get(lookup_split, ()))
    filtered = _select_episodic_fewshot_sequences(
        sequences,
        active_max_days=active_max_days,
        samples_per_lake=max_episodes,
        epoch=0,
    )
    return list(filtered)


@torch.no_grad()
def _diagnose_lake(
    model,
    lake: dict,
    *,
    split_name: str,
    lookup_split: str,
    active_max_days: int,
    support_profile_count: int,
    max_episodes: int,
) -> list[dict]:
    rows = []
    date_to_index = lake["date_to_index"]
    device = lake["depths"].device
    for episode_index, (query_start, query_start_idx, support_dates, targets) in enumerate(
        _selected_sequences(
            lake,
            lookup_split=lookup_split,
            active_max_days=active_max_days,
            max_episodes=max_episodes,
        )
    ):
        active_targets = tuple(
            (target, target_idx)
            for target, target_idx in targets
            if 1 <= int(target_idx - query_start_idx) <= int(active_max_days)
        )
        selected_support = tuple(
            date for date in sorted(support_dates)[-max(1, int(support_profile_count)) :]
            if date in lake["lookups"].get(lookup_split, {})
        )
        leak_count = sum(1 for date in selected_support if not (pd.Timestamp(date) < pd.Timestamp(query_start)))
        if not selected_support or not active_targets:
            continue

        support_profiles = []
        support_masks = []
        support_ages = []
        for support_date in selected_support:
            profile, mask = _target_tensor_and_mask(lake, lookup_split, support_date)
            support_profiles.append(profile.reshape(-1))
            support_masks.append(
                torch.ones_like(profile, dtype=torch.float32).reshape(-1)
                if mask is None
                else mask.reshape(-1).to(device=device, dtype=torch.float32)
            )
            support_ages.append(float(query_start_idx - date_to_index[support_date]))
        support_profiles = torch.stack(support_profiles, dim=0).unsqueeze(0)
        support_masks = torch.stack(support_masks, dim=0).unsqueeze(0)
        support_ages_tensor = torch.tensor(support_ages, dtype=torch.float32, device=device).unsqueeze(0)

        base_profile, _prior_info = build_lst_profile_prior(
            lake["df"],
            lake["depths_np"],
            lake["metadata"],
            int(query_start_idx),
        )
        base_profile = torch.tensor(base_profile, dtype=torch.float32, device=device).reshape(1, -1)
        encoded = model.encode_fewshot_support(
            support_profiles,
            support_masks,
            support_ages_tensor,
            lake["static_features"],
            lake["forcing_rows"][int(query_start_idx)],
        )
        delta = encoded["initial_profile_delta_c"]
        corrected = torch.clamp(base_profile + delta, 0.0, 40.0)
        query_profile, query_mask = _target_tensor_and_mask(lake, lookup_split, query_start)
        base_valid, truth_valid, _valid_mask = _masked_arrays(base_profile, query_profile, query_mask)
        corrected_valid, _truth_again, _ = _masked_arrays(corrected, query_profile, query_mask)
        delta_valid = corrected_valid - base_valid
        if base_valid.size == 0:
            continue
        base_error = base_valid - truth_valid
        corrected_error = corrected_valid - truth_valid
        direction_hits = (delta_valid * base_error) < 0.0
        abs_improvement = np.abs(base_error) - np.abs(corrected_error)
        row = {
            "experiment_id": EXPERIMENT_ID,
            "source_lake_id": lake["lake_id"],
            "lake_group": _lake_group(lake),
            "split": split_name,
            "lookup_split": lookup_split,
            "episode_index": int(episode_index),
            "query_start": str(pd.Timestamp(query_start).date()),
            "query_start_idx": int(query_start_idx),
            "support_profile_count": int(len(selected_support)),
            "support_age_mean_days": float(np.mean(support_ages)),
            "support_age_max_days": float(np.max(support_ages)),
            "target_count": int(len(active_targets)),
            "target_max_gap_days": int(max(int(idx - query_start_idx) for _target, idx in active_targets)),
            "support_query_leak_count": int(leak_count),
            "point_count": int(base_valid.size),
            "base_query_start_sse": float(np.sum(base_error ** 2)),
            "corrected_query_start_sse": float(np.sum(corrected_error ** 2)),
            "base_query_start_rmse": float(np.sqrt(np.mean(base_error ** 2))),
            "corrected_query_start_rmse": float(np.sqrt(np.mean(corrected_error ** 2))),
            "query_start_rmse_delta_corrected_minus_base": float(
                np.sqrt(np.mean(corrected_error ** 2)) - np.sqrt(np.mean(base_error ** 2))
            ),
            "base_query_start_mae": float(np.mean(np.abs(base_error))),
            "corrected_query_start_mae": float(np.mean(np.abs(corrected_error))),
            "query_start_mae_delta_corrected_minus_base": float(
                np.mean(np.abs(corrected_error)) - np.mean(np.abs(base_error))
            ),
            "base_query_start_bias": float(np.mean(base_error)),
            "corrected_query_start_bias": float(np.mean(corrected_error)),
            "delta_mean_c": float(np.mean(delta_valid)),
            "delta_abs_mean_c": float(np.mean(np.abs(delta_valid))),
            "delta_rms_c": float(np.sqrt(np.mean(delta_valid ** 2))),
            "delta_max_abs_c": float(np.max(np.abs(delta_valid))),
            "direction_hit_fraction": float(np.mean(direction_hits)),
            "abs_error_improvement_mean_c": float(np.mean(abs_improvement)),
            "regularization_loss": float(encoded["regularization_loss"].detach().double().cpu().reshape(-1)[0]),
            "encoded_support_profile_count": float(
                encoded["support_profile_count"].detach().double().cpu().reshape(-1)[0]
            ),
            "encoded_support_age_mean_days": float(
                encoded["support_age_mean_days"].detach().double().cpu().reshape(-1)[0]
            ),
        }
        rows.append(row)
    return rows


def _summarize(rows: list[dict], *, group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    summary = []
    for key, items in sorted(grouped.items()):
        point_count = _finite_sum(item.get("point_count", 0) for item in items)
        base_sse = _finite_sum(item.get("base_query_start_sse", 0.0) for item in items)
        corrected_sse = _finite_sum(item.get("corrected_query_start_sse", 0.0) for item in items)
        leak_count = int(_finite_sum(item.get("support_query_leak_count", 0) for item in items))
        direction_num = _finite_sum(
            float(item.get("direction_hit_fraction", float("nan"))) * float(item.get("point_count", 0))
            for item in items
        )
        row = {
            group_key: key,
            "episode_count": int(len(items)),
            "lake_count": int(len({item.get("source_lake_id") for item in items})),
            "point_count": int(point_count),
            "base_query_start_rmse": _rmse_from_sse(base_sse, point_count),
            "corrected_query_start_rmse": _rmse_from_sse(corrected_sse, point_count),
            "query_start_rmse_delta_corrected_minus_base": (
                _rmse_from_sse(corrected_sse, point_count) - _rmse_from_sse(base_sse, point_count)
            ),
            "base_query_start_mae": _finite_mean(item.get("base_query_start_mae") for item in items),
            "corrected_query_start_mae": _finite_mean(item.get("corrected_query_start_mae") for item in items),
            "query_start_mae_delta_corrected_minus_base": _finite_mean(
                item.get("query_start_mae_delta_corrected_minus_base") for item in items
            ),
            "delta_abs_mean_c": _finite_mean(item.get("delta_abs_mean_c") for item in items),
            "delta_rms_c": _finite_mean(item.get("delta_rms_c") for item in items),
            "direction_hit_fraction_weighted": float(direction_num / point_count) if point_count > 0 else float("nan"),
            "abs_error_improvement_mean_c": _finite_mean(item.get("abs_error_improvement_mean_c") for item in items),
            "support_query_leak_count": leak_count,
            "status": "pass" if leak_count == 0 else "fail_leakage",
        }
        summary.append(row)
    return summary


def _overall_status(split_summary: list[dict]) -> str:
    leaks = sum(int(row.get("support_query_leak_count", 0)) for row in split_summary)
    if leaks:
        return "fail_leakage"
    primary = [row for row in split_summary if row.get("split") in {"checkpoint_validation", "declared_validation_lakes"}]
    if not primary:
        return "inconclusive"
    mean_delta = _finite_mean(row.get("query_start_rmse_delta_corrected_minus_base") for row in primary)
    mean_direction = _finite_mean(row.get("direction_hit_fraction_weighted") for row in primary)
    if math.isfinite(mean_delta) and mean_delta < -0.05 and math.isfinite(mean_direction) and mean_direction >= 0.55:
        return "support_update_helpful_at_query_start"
    if math.isfinite(mean_delta) and mean_delta > 0.05:
        return "support_update_hurts_query_start"
    return "support_update_weak_or_inconclusive"


def _write_report(path: Path, *, payload: dict, split_summary: list[dict], by_lake_csv: Path, by_episode_csv: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {EXPERIMENT_ID} closeout",
        "",
        "## Hypothesis Tested",
        "A support-profile analysis update should move the query-start prior profile toward the observed query-start profile before any long rollout.",
        "",
        "## Decision",
        f"- Status: `{payload['status']}`",
        f"- Source experiment: `{payload['source_experiment_id']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        "- This is diagnostic-only; it does not select a checkpoint and does not justify formal L3/L7 claims.",
        "",
        "## Split Summary",
        "| split | lakes | episodes | base RMSE | corrected RMSE | delta | direction hit | leak count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_summary:
        lines.append(
            "| {split} | {lake_count} | {episode_count} | {base:.3f} | {corr:.3f} | {delta:.3f} | {direction:.3f} | {leak} |".format(
                split=row.get("split", ""),
                lake_count=int(row.get("lake_count", 0)),
                episode_count=int(row.get("episode_count", 0)),
                base=float(row.get("base_query_start_rmse", float("nan"))),
                corr=float(row.get("corrected_query_start_rmse", float("nan"))),
                delta=float(row.get("query_start_rmse_delta_corrected_minus_base", float("nan"))),
                direction=float(row.get("direction_hit_fraction_weighted", float("nan"))),
                leak=int(row.get("support_query_leak_count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "- Negative delta means the support update improves the query-start profile.",
            "- Direction hit fraction above 0.55 means the update usually moves against the prior error.",
            "- If query-start improves but few-shot rollout RMSE remains weak, the likely failure is downstream physical propagation rather than support encoding alone.",
            "- If query-start does not improve, a larger adapter or formal seed run should not be started before fixing support representation.",
            "",
            "## Artifacts",
            f"- By-episode CSV: `{by_episode_csv}`",
            f"- By-lake CSV: `{by_lake_csv}`",
            f"- Summary JSON: `{payload['summary_json']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    source_manifest: Path,
    checkpoint_path: Path,
    output_dir: Path,
    active_max_days: int,
    support_profile_count: int,
    max_episodes_per_lake: int,
    device_arg: str,
    report_path: Path | None = None,
    diagnostic_manifest_path: Path | None = None,
) -> dict:
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
    selection = _resolve_heldout_selection(lakes, manifest=manifest)
    train_lakes = selection["train_lakes"]
    heldout_lakes = selection["heldout_lakes"]
    declared_val_lakes = _ordered_lakes_by_id(lakes, _parse_string_list(manifest.get("val_lake_ids")))

    rows = []
    for lake in train_lakes:
        rows.extend(
            _diagnose_lake(
                model,
                lake,
                split_name="checkpoint_validation",
                lookup_split="val",
                active_max_days=active_max_days,
                support_profile_count=support_profile_count,
                max_episodes=max_episodes_per_lake,
            )
        )
    for lake in declared_val_lakes:
        rows.extend(
            _diagnose_lake(
                model,
                lake,
                split_name="declared_validation_lakes",
                lookup_split="val",
                active_max_days=active_max_days,
                support_profile_count=support_profile_count,
                max_episodes=max_episodes_per_lake,
            )
        )
    for lake in heldout_lakes:
        rows.extend(
            _diagnose_lake(
                model,
                lake,
                split_name="heldout_diagnostic",
                lookup_split="all",
                active_max_days=active_max_days,
                support_profile_count=support_profile_count,
                max_episodes=max_episodes_per_lake,
            )
        )

    by_episode_csv = output_dir / "diagnostic_support_update_effect_by_episode.csv"
    by_lake_csv = output_dir / "diagnostic_support_update_effect_by_lake.csv"
    summary_json = output_dir / "diagnostic_support_update_effect_summary.json"
    pd.DataFrame(rows).to_csv(by_episode_csv, index=False)
    by_lake = _summarize(rows, group_key="source_lake_id")
    split_summary = _summarize(rows, group_key="split")
    pd.DataFrame(by_lake).to_csv(by_lake_csv, index=False)
    status = _overall_status(split_summary)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": manifest.get("experiment_id") or manifest.get("experiment") or "",
        "source_manifest": str(source_manifest),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "active_max_days": int(active_max_days),
        "support_profile_count": int(support_profile_count),
        "max_episodes_per_lake": int(max_episodes_per_lake),
        "status": status,
        "split_summary": split_summary,
        "by_lake_csv": str(by_lake_csv),
        "by_episode_csv": str(by_episode_csv),
        "summary_json": str(summary_json),
        "checkpoint_selection": "none_diagnostic_only",
        "heldout_policy": "heldout metrics are diagnostic only and are not used for checkpoint selection or tuning",
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if report_path is not None:
        _write_report(
            Path(report_path),
            payload=payload,
            split_summary=split_summary,
            by_lake_csv=by_lake_csv,
            by_episode_csv=by_episode_csv,
        )
        payload["report_path"] = str(report_path)
    if diagnostic_manifest_path is not None:
        diagnostic_manifest = {
            "experiment_id": EXPERIMENT_ID,
            "level": "diagnostic_eval",
            "hypothesis": "Support-profile initial-state updates should improve query-start profiles without support/query leakage.",
            "source_manifest": str(source_manifest),
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "summary_json": str(summary_json),
            "report": str(report_path) if report_path else "",
            "checkpoint_selection": "none_diagnostic_only",
            "heldout_policy": payload["heldout_policy"],
            "status": status,
            "created_at": payload["created_at"],
        }
        Path(diagnostic_manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(diagnostic_manifest_path).write_text(
            json.dumps(diagnostic_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["diagnostic_manifest_path"] = str(diagnostic_manifest_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose support-profile update direction and magnitude.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-max-days", type=int, default=60)
    parser.add_argument("--support-profile-count", type=int, default=5)
    parser.add_argument("--max-episodes-per-lake", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--diagnostic-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_diagnostic(
        source_manifest=args.source_manifest,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        active_max_days=args.active_max_days,
        support_profile_count=args.support_profile_count,
        max_episodes_per_lake=args.max_episodes_per_lake,
        device_arg=args.device,
        report_path=args.report_path,
        diagnostic_manifest_path=args.diagnostic_manifest,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
