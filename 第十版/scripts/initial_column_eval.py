"""Evaluate zero-profile initial-column errors by lake and depth band.

This diagnostic loads an existing state forecaster checkpoint, builds the
multi-lake data bundle with epochs=0, and compares the EOF/PCA init-net
initial column against observed profiles.  It does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lake_pinn.state_model import resolve_hard_density_stability
from lake_pinn.state_multilake import (
    _target_tensor_and_mask,
    train_multilake_state_forecaster,
)
from lake_pinn.state_reconstruction import (
    build_zero_profile_eof_pca_low_dof_prior,
    zero_profile_thermal_basis_tensors_for_depths,
)


BANDS = ("whole", "surface", "le25m", "gt25m")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initial-column per-lake/depth-band eval-only diagnostic."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--spinup-days", type=int, default=30)
    parser.add_argument(
        "--max-dates-per-split",
        type=int,
        default=12,
        help="Evenly sample at most this many dates per lake/split. 0 uses all dates.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,all",
        help="Comma-separated profile splits to evaluate for non-heldout lakes.",
    )
    return parser.parse_args()


def _lake_group(lake: dict) -> str:
    return str(lake.get("metadata", {}).get("lake_group") or lake.get("lake_id") or "")


def _lake_role(lake: dict, heldout_ids: set[str], heldout_groups: set[str]) -> str:
    if lake["lake_id"] in heldout_ids or _lake_group(lake) in heldout_groups:
        return "heldout_diagnostic_only"
    return "train_group"


def _selected_dates(lake: dict, split_name: str, max_dates: int) -> list[pd.Timestamp]:
    lookup = lake.get("lookups", {}).get(split_name, {})
    date_to_index = lake.get("date_to_index", {})
    dates = sorted(date for date in lookup if date in date_to_index)
    if max_dates <= 0 or len(dates) <= max_dates:
        return dates
    indices = np.linspace(0, len(dates) - 1, int(max_dates))
    chosen = []
    used = set()
    for value in indices:
        idx = int(round(float(value)))
        idx = min(max(idx, 0), len(dates) - 1)
        if idx not in used:
            used.add(idx)
            chosen.append(dates[idx])
    return chosen


def _init_profile_for_index(model, lake: dict, start_idx: int) -> tuple[torch.Tensor, dict]:
    basis = getattr(model, "zero_profile_thermal_basis", None)
    base_profile, _info = build_zero_profile_eof_pca_low_dof_prior(
        lake["df"],
        lake["depths_np"],
        lake["metadata"],
        int(start_idx),
        thermal_basis=basis,
    )
    base_tensor = torch.as_tensor(
        base_profile,
        dtype=lake["depths"].dtype,
        device=lake["depths"].device,
    ).reshape(1, -1)
    basis_tensors = zero_profile_thermal_basis_tensors_for_depths(
        basis,
        lake["depths_np"],
        device=lake["depths"].device,
        dtype=lake["depths"].dtype,
    )
    if basis_tensors is None:
        raise RuntimeError("zero_profile_thermal_basis is missing or invalid.")
    encoded = model.zero_profile_initial_state_from_basis(
        base_tensor,
        lake["forcing_rows"][int(start_idx)]["history_features"],
        lake["static_features"],
        basis_tensors["components_on_depth"],
        basis_tensors["coeff_std"],
    )
    return encoded["initial_profile_c"], encoded


def _roll_profile_to_index(
    model,
    lake: dict,
    profile: torch.Tensor,
    start_idx: int,
    target_idx: int,
    *,
    hard_density_stability: bool,
) -> torch.Tensor:
    current = profile
    freezing_storage = torch.zeros_like(current)
    for day_idx in range(int(start_idx), int(target_idx)):
        next_row = (
            lake["forcing_rows"][day_idx + 1]
            if day_idx + 1 < len(lake["forcing_rows"])
            else None
        )
        current, freezing_storage, _diagnostics = model.step(
            current,
            lake["forcing_rows"][day_idx],
            lake["static_features"],
            next_forcing_row=next_row,
            task_mode="analysis",
            depths=lake["depths"],
            area_profile=lake["area"],
            return_diagnostics=True,
            diagnostic_mode="loss",
            hard_density_stability=hard_density_stability,
            freezing_storage_j_m2=freezing_storage,
            return_freezing_storage=True,
        )
    return current


def _band_masks(depths: np.ndarray) -> dict[str, np.ndarray]:
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    finite = np.isfinite(depths)
    surface = np.zeros_like(finite, dtype=bool)
    if finite.any():
        surface[int(np.where(finite)[0][0])] = True
    return {
        "whole": finite,
        "surface": surface,
        "le25m": finite & (depths <= 25.0),
        "gt25m": finite & (depths > 25.0),
    }


def _stats_for_prediction(prediction, target, mask, depths) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    obs = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    bands = _band_masks(depths)
    out = {}
    errors = pred - obs
    for band in BANDS:
        band_valid = valid & bands[band]
        count = int(np.sum(band_valid))
        out[f"{band}_count"] = count
        if count <= 0:
            out[f"{band}_rmse"] = math.nan
            out[f"{band}_mae"] = math.nan
            out[f"{band}_bias"] = math.nan
            continue
        e = errors[band_valid]
        out[f"{band}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{band}_mae"] = float(np.mean(np.abs(e)))
        out[f"{band}_bias"] = float(np.mean(e))
    return out


def _empty_accumulator():
    return {
        "sse": defaultdict(float),
        "abs": defaultdict(float),
        "sum": defaultdict(float),
        "count": defaultdict(int),
        "dates": 0,
    }


def _add_stats(acc, prefix: str, stats: dict[str, float]) -> None:
    for band in BANDS:
        count = int(stats.get(f"{band}_count", 0) or 0)
        if count <= 0:
            continue
        rmse = stats.get(f"{band}_rmse", math.nan)
        mae = stats.get(f"{band}_mae", math.nan)
        bias = stats.get(f"{band}_bias", math.nan)
        if np.isfinite(rmse):
            acc["sse"][(prefix, band)] += float(rmse) ** 2 * count
        if np.isfinite(mae):
            acc["abs"][(prefix, band)] += float(mae) * count
        if np.isfinite(bias):
            acc["sum"][(prefix, band)] += float(bias) * count
        acc["count"][(prefix, band)] += count


def _summary_row(key: tuple, acc) -> dict[str, object]:
    row = {
        "lake_id": key[0],
        "lake_group": key[1],
        "role": key[2],
        "split": key[3],
        "evaluated_dates": int(acc["dates"]),
    }
    for prefix in ("same_day", "spinup"):
        for band in BANDS:
            count = int(acc["count"].get((prefix, band), 0))
            row[f"{prefix}_{band}_count"] = count
            if count <= 0:
                row[f"{prefix}_{band}_rmse"] = ""
                row[f"{prefix}_{band}_mae"] = ""
                row[f"{prefix}_{band}_bias"] = ""
                continue
            row[f"{prefix}_{band}_rmse"] = math.sqrt(acc["sse"][(prefix, band)] / count)
            row[f"{prefix}_{band}_mae"] = acc["abs"][(prefix, band)] / count
            row[f"{prefix}_{band}_bias"] = acc["sum"][(prefix, band)] / count
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    load_only_manifest = dict(manifest)
    load_only_manifest.update(
        {
            "epochs": 0,
            "checkpoint_every_epochs": 0,
            "eval_every_epochs": 0,
            "full_eval_every_epochs": 0,
            "export_after_training": "off",
            "initial_column_eval_load_only": True,
        }
    )
    load_only_manifest_path = output_dir / "initial_column_eval_load_only_manifest.json"
    load_only_manifest_path.write_text(
        json.dumps(load_only_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = train_multilake_state_forecaster(
        manifest_path=load_only_manifest_path,
        output_dir=output_dir / "_load_only",
        epochs=0,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every_epochs=0,
        export_after_training="off",
        full_eval_every_epochs=0,
        eval_every_epochs=50,
        export_only=False,
        device=args.device,
    )
    model = result["model"]
    model.eval()
    lakes = result["lakes"]
    heldout_ids = set(manifest.get("heldout_lake_ids") or manifest.get("test_lake_ids") or [])
    heldout_groups = set(manifest.get("heldout_lake_groups") or manifest.get("test_lake_groups") or [])
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    hard_density = resolve_hard_density_stability(
        str(manifest.get("hard_density_stability", "auto")).strip().lower(),
        task_mode="analysis",
        data_fill_mode="reconstruction",
    )
    spinup_days = max(0, int(args.spinup_days))

    detail_rows = []
    accumulators = defaultdict(_empty_accumulator)
    torch.set_grad_enabled(False)
    with torch.no_grad():
        for lake in lakes:
            role = _lake_role(lake, heldout_ids, heldout_groups)
            eval_splits = ["all"] if role == "heldout_diagnostic_only" else requested_splits
            for split_name in eval_splits:
                if split_name not in lake.get("lookups", {}):
                    continue
                dates = _selected_dates(lake, split_name, int(args.max_dates_per_split))
                for date_value in dates:
                    target_idx = int(lake["date_to_index"][date_value])
                    target, target_mask = _target_tensor_and_mask(lake, split_name, date_value)
                    if target is None:
                        continue
                    same_profile, same_encoded = _init_profile_for_index(model, lake, target_idx)
                    same_stats = _stats_for_prediction(
                        same_profile.detach().cpu().numpy().reshape(-1),
                        target.detach().cpu().numpy().reshape(-1),
                        target_mask.detach().cpu().numpy().reshape(-1) if target_mask is not None else None,
                        lake["depths_np"],
                    )
                    start_idx = max(0, target_idx - spinup_days)
                    spin_profile0, spin_encoded = _init_profile_for_index(model, lake, start_idx)
                    spin_profile = _roll_profile_to_index(
                        model,
                        lake,
                        spin_profile0,
                        start_idx,
                        target_idx,
                        hard_density_stability=hard_density,
                    )
                    spin_stats = _stats_for_prediction(
                        spin_profile.detach().cpu().numpy().reshape(-1),
                        target.detach().cpu().numpy().reshape(-1),
                        target_mask.detach().cpu().numpy().reshape(-1) if target_mask is not None else None,
                        lake["depths_np"],
                    )
                    key = (lake["lake_id"], _lake_group(lake), role, split_name)
                    accumulators[key]["dates"] += 1
                    _add_stats(accumulators[key], "same_day", same_stats)
                    _add_stats(accumulators[key], "spinup", spin_stats)
                    row = {
                        "lake_id": lake["lake_id"],
                        "lake_group": _lake_group(lake),
                        "role": role,
                        "split": split_name,
                        "date": str(pd.Timestamp(date_value).date()),
                        "target_idx": target_idx,
                        "spinup_start_idx": start_idx,
                        "spinup_days_used": int(target_idx - start_idx),
                        "same_day_delta_abs_mean_c": float(
                            same_encoded["initial_delta_abs_mean_c"].detach().cpu().reshape(-1)[0]
                        ),
                        "spinup_start_delta_abs_mean_c": float(
                            spin_encoded["initial_delta_abs_mean_c"].detach().cpu().reshape(-1)[0]
                        ),
                    }
                    for name, stats in (("same_day", same_stats), ("spinup", spin_stats)):
                        for metric, value in stats.items():
                            row[f"{name}_{metric}"] = value
                    detail_rows.append(row)

    lake_rows = [_summary_row(key, acc) for key, acc in sorted(accumulators.items())]
    group_acc = defaultdict(_empty_accumulator)
    role_acc = defaultdict(_empty_accumulator)
    overall_acc = _empty_accumulator()
    for key, acc in accumulators.items():
        lake_id, lake_group, role, split = key
        for prefix in ("same_day", "spinup"):
            for band in BANDS:
                count = int(acc["count"].get((prefix, band), 0))
                if count <= 0:
                    continue
                for target_acc in (
                    group_acc[(lake_group, role, split)],
                    role_acc[(role, split)],
                    overall_acc,
                ):
                    target_acc["sse"][(prefix, band)] += acc["sse"][(prefix, band)]
                    target_acc["abs"][(prefix, band)] += acc["abs"][(prefix, band)]
                    target_acc["sum"][(prefix, band)] += acc["sum"][(prefix, band)]
                    target_acc["count"][(prefix, band)] += count
        group_acc[(lake_group, role, split)]["dates"] += acc["dates"]
        role_acc[(role, split)]["dates"] += acc["dates"]
        overall_acc["dates"] += acc["dates"]

    group_rows = []
    for key, acc in sorted(group_acc.items()):
        row = _summary_row(("", key[0], key[1], key[2]), acc)
        row.pop("lake_id", None)
        group_rows.append(row)
    role_rows = []
    for key, acc in sorted(role_acc.items()):
        row = _summary_row(("", "", key[0], key[1]), acc)
        row.pop("lake_id", None)
        row.pop("lake_group", None)
        role_rows.append(row)
    overall_row = _summary_row(("ALL", "ALL", "all", "all"), overall_acc)

    _write_csv(output_dir / "initial_column_eval_profile_rows.csv", detail_rows)
    _write_csv(output_dir / "initial_column_eval_lake_summary.csv", lake_rows)
    _write_csv(output_dir / "initial_column_eval_group_summary.csv", group_rows)
    _write_csv(output_dir / "initial_column_eval_role_summary.csv", role_rows)
    (output_dir / "initial_column_eval_overall_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(Path(args.manifest)),
                "checkpoint_path": str(Path(args.checkpoint_path)),
                "spinup_days": spinup_days,
                "max_dates_per_split": int(args.max_dates_per_split),
                "hard_density_stability": bool(hard_density),
                "detail_rows": len(detail_rows),
                "overall": overall_row,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(detail_rows)} profile rows to {output_dir}")
    print(json.dumps(overall_row, ensure_ascii=False))


if __name__ == "__main__":
    main()
