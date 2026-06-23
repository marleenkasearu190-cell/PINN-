from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[1]
if CANONICAL_REPO_ROOT.exists():
    REPO_ROOT = CANONICAL_REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_r10_experiments import (
    CHECKPOINT_SELECTION,
    DEFAULT_STANDARD_INPUT_ROOT,
    PRIMARY_TRANSFER_METRIC,
    _path_string,
    _train_lake_ids,
    _unique,
    code_hash,
    data_version_from_readiness,
    git_short_hash,
    load_usable_lakes,
    resolve_readiness_csv,
    upsert_registry,
)


PINN_ROOT = REPO_ROOT.parent
DEFAULT_PIPELINE_ROOT = PINN_ROOT / "pipeline"
LOCAL34_SPLIT_ID = "LOCAL34_GROUPHELDOUT_V1"
LOCAL34_DATASET_ID = "LOCAL34_CORE"
L3_MANIFEST_NAME = "M2_LOCAL34_GROUPHELDOUT_V1.json"
L3_EXPERIMENT_BASE = "RECON_L3_LOCAL34_CLEANPHYS_gheldout_v1"
L3_SEEDS = (1, 2, 3)
LOCAL34_STRESS_GROUPS = ("kivu", "lake_maggiore", "suggs", "sunapee", "toolik")
LOCAL34_DOCUMENT_UNUSABLE_IDS = ("kivu_2013", "lough_feeagh_2016", "rimov_2012", "sammamish_2015")
LOCAL34_TRAIN_GROUPS = (
    "barco",
    "beaverdam_reservoir",
    "crystal_bog",
    "falling_creek_reservoir",
    "green_lake_4",
    "kinneret",
    "lake_washington",
    "lough_feeagh",
    "mendota",
    "sparkling",
    "trout_bog",
    "trout_lake",
)
LOCAL34_VAL_GROUPS = ("erken", "mohonk", "carvins_cove")
LOCAL34_TEST_GROUPS = ("lacawac", "el_val", "namco")
ROADMAP_MANIFEST_DIRS = (
    "L1_smoke",
    "L2_single_lake",
    "L3_local_core",
    "L4_physics_ablation",
    "L5_group_transfer",
    "L6_climate_transfer",
    "L7_fewshot",
    "L8_uncertainty",
    "L9_global_product",
)
PIPELINE_DIRS = ("audits", "download_plans", "scoreboards", "reports")
L3_GO_NO_GO_CRITERIA = (
    "median point RMSE below 2.2-2.5 C on transfer-valid LOCAL34 groups",
    "median rolling 30d RMSE below 2.7-3.0 C",
    "median rolling 60d RMSE below 3.2-3.5 C",
    "best checkpoint selected by validation rolling/few-shot metrics only",
    "LOCAL34 train/validation/test lake groups are disjoint",
    "Kz/Kd/residual diagnostics show no physical bypass",
    "heldout groups are diagnostic and not used for hyperparameter tuning",
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _upsert_csv(path: Path, new_rows: list[dict[str, str]], default_fields: list[str], *, key: str = "queue_id") -> None:
    existing_rows, existing_fields = _read_csv(path)
    by_key = {row.get(key, ""): row for row in existing_rows if row.get(key)}
    ordered_keys = [row.get(key, "") for row in existing_rows if row.get(key)]
    for row in new_rows:
        row_key = row[key]
        by_key[row_key] = {**by_key.get(row_key, {}), **row}
        if row_key not in ordered_keys:
            ordered_keys.append(row_key)
    fields = list(dict.fromkeys(existing_fields + default_fields + [field for row in new_rows for field in row]))
    _write_csv(path, [by_key[item] for item in ordered_keys if item in by_key], fields)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_roadmap_directories(*, v9_root: Path = REPO_ROOT, pipeline_root: Path = DEFAULT_PIPELINE_ROOT) -> dict[str, list[str]]:
    manifest_dirs = []
    for name in ROADMAP_MANIFEST_DIRS:
        path = v9_root / "experiments" / "manifests_clean" / name
        path.mkdir(parents=True, exist_ok=True)
        manifest_dirs.append(_path_string(path))
    for path in (v9_root / "experiments" / "splits", v9_root / "results"):
        path.mkdir(parents=True, exist_ok=True)
    pipeline_dirs = []
    for name in PIPELINE_DIRS:
        path = pipeline_root / name
        path.mkdir(parents=True, exist_ok=True)
        pipeline_dirs.append(_path_string(path))
    return {
        "manifest_dirs": manifest_dirs,
        "split_dir": [_path_string(v9_root / "experiments" / "splits")],
        "results_dir": [_path_string(v9_root / "results")],
        "pipeline_dirs": pipeline_dirs,
    }


def _lake_ids_for_groups(
    lakes: list[dict],
    groups: tuple[str, ...],
    *,
    excluded_ids: set[str],
) -> list[str]:
    ids: list[str] = []
    for group in groups:
        group_ids = [
            lake["lake_id"]
            for lake in lakes
            if lake.get("lake_group") == group and lake.get("lake_id") not in excluded_ids
        ]
        if not group_ids:
            raise ValueError(f"LOCAL34 group has no usable lake-years: {group}")
        ids.extend(sorted(group_ids))
    return ids


def _check_disjoint_groups(split: dict) -> None:
    train = set(split["train_lake_groups"])
    val = set(split["val_lake_groups"])
    test = set(split["test_lake_groups"])
    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    leaking = {name: groups for name, groups in overlaps.items() if groups}
    if leaking:
        raise ValueError(f"LOCAL34 lake-group leakage: {leaking}")


def build_local34_split(
    lakes: list[dict],
    *,
    standard_input_root: Path,
    readiness_csv: Path,
    data_version: str,
) -> dict:
    excluded_ids = set(LOCAL34_DOCUMENT_UNUSABLE_IDS)
    train_ids = _lake_ids_for_groups(lakes, LOCAL34_TRAIN_GROUPS, excluded_ids=excluded_ids)
    val_ids = _lake_ids_for_groups(lakes, LOCAL34_VAL_GROUPS, excluded_ids=excluded_ids)
    test_ids = _lake_ids_for_groups(lakes, LOCAL34_TEST_GROUPS, excluded_ids=excluded_ids)
    selected_ids = train_ids + val_ids + test_ids
    by_id = {lake["lake_id"]: lake for lake in lakes}
    selected_groups = {by_id[lake_id]["lake_group"] for lake_id in selected_ids}
    stress_included = sorted(selected_groups & set(LOCAL34_STRESS_GROUPS))
    if stress_included:
        raise ValueError(f"LOCAL34 selected stress groups unexpectedly: {stress_included}")
    selected_unusable = sorted(set(selected_ids) & excluded_ids)
    if selected_unusable:
        raise ValueError(f"LOCAL34 selected document-unusable lake-years unexpectedly: {selected_unusable}")
    split = {
        "split_id": LOCAL34_SPLIT_ID,
        "dataset_id": LOCAL34_DATASET_ID,
        "level": "L3_local_multilake_clean_physics",
        "role": "development_transfer_split",
        "data_version": data_version,
        "standard_input_root": _path_string(standard_input_root),
        "readiness_csv": _path_string(readiness_csv),
        "train_lake_groups": list(LOCAL34_TRAIN_GROUPS),
        "val_lake_groups": list(LOCAL34_VAL_GROUPS),
        "test_lake_groups": list(LOCAL34_TEST_GROUPS),
        "train_lake_ids": train_ids,
        "val_lake_ids": val_ids,
        "test_lake_ids": test_ids,
        "all_lake_ids": selected_ids,
        "excluded_stress_lake_groups": list(LOCAL34_STRESS_GROUPS),
        "excluded_document_unusable_lake_ids": list(LOCAL34_DOCUMENT_UNUSABLE_IDS),
        "group_leakage_check": "passed",
        "counts": {
            "train_groups": len(LOCAL34_TRAIN_GROUPS),
            "val_groups": len(LOCAL34_VAL_GROUPS),
            "test_groups": len(LOCAL34_TEST_GROUPS),
            "train_lake_years": len(train_ids),
            "val_lake_years": len(val_ids),
            "test_lake_years": len(test_ids),
            "total_lake_years": len(selected_ids),
        },
        "notes": [
            "LOCAL34 is a development transfer split, not the final locked global test.",
            "carvins_cove is used as a LOCAL34 validation group by the roadmap contract.",
        ],
    }
    _check_disjoint_groups(split)
    if len(selected_ids) != 34:
        raise ValueError(f"LOCAL34 expected 34 lake-years, found {len(selected_ids)}")
    return split


def build_l3_manifest(*, split: dict, lakes: list[dict], split_path: Path, data_version: str, code_version: str) -> dict:
    by_id = {lake["lake_id"]: lake for lake in lakes}
    selected_lakes = [by_id[lake_id] for lake_id in split["all_lake_ids"]]
    return {
        "experiment_id": L3_EXPERIMENT_BASE,
        "experiment": L3_EXPERIMENT_BASE,
        "level": "L3_local_multilake_clean_physics",
        "hypothesis": (
            "A clean-physics RECON backbone trained on LOCAL34 train groups can improve "
            "whole-lake-group validation/test rolling and few-shot transfer without physical bypass."
        ),
        "code_hash": code_version,
        "data_version": data_version,
        "split_file": _path_string(split_path),
        "split_id": split["split_id"],
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "primary_transfer_metric": "0.5 * val_rolling_start_rmse_30d + 0.5 * val_rolling_start_rmse_60d",
        "secondary_physics_metrics": [
            "val_fewshot_rmse_30d",
            "val_fewshot_rmse_60d",
            "heldout_free_roll_mean_rmse",
            "residual_abs_mean_c",
            "kd_prior_regularization_loss_mean",
            "stratification_mixing_gate_deep_mean",
        ],
        "go_no_go_criteria": list(L3_GO_NO_GO_CRITERIA),
        "split_mode": "time_blocked",
        "split_contract": "LOCAL34 train/validation/test lake groups are disjoint; locked final tests are diagnostic only.",
        "locked_test_policy": "development split; do not use as final locked-test evidence",
        "profile_supervision_scope": "train",
        "depth_points": 40,
        "history_window_days": 30,
        "max_rollout_days": 60,
        "segment_rollout_max_days": 60,
        "recommended_epochs": 200,
        "learning_rate": 3e-4,
        "full_eval_every_epochs": 20,
        "checkpoint_every_epochs": 20,
        "eval_every_epochs": 20,
        "residual_limit_c": 0.15,
        "daily_tendency_weight": 0.0,
        "warm_season_column_heat_content_weight": 0.0,
        "lst_feature_dropout_probability": 0.40,
        "episodic_fewshot_mode": "on",
        "episodic_fewshot_support_profile_count": 3,
        "episodic_fewshot_max_query_days": 120,
        "fewshot_adapter_params": "kz,kd,exchange,convective,ice",
        "support_query_policy": "support profiles must precede query windows; no future query leakage",
        "train_lake_ids": split["train_lake_ids"],
        "val_lake_ids": split["val_lake_ids"],
        "test_lake_ids": split["test_lake_ids"],
        "heldout_lake_ids": split["test_lake_ids"],
        "train_lake_groups": split["train_lake_groups"],
        "val_lake_groups": split["val_lake_groups"],
        "test_lake_groups": split["test_lake_groups"],
        "heldout_lake_groups": split["test_lake_groups"],
        "lakes": selected_lakes,
    }


def _registry_row_from_manifest(
    *,
    experiment_id: str,
    manifest: dict,
    manifest_path: Path,
    split_path: Path,
    seed: int | None,
    output_dir: Path,
    status: str,
) -> dict[str, str]:
    train_lakes = manifest.get("train_lake_ids") or _train_lake_ids(
        manifest["lakes"],
        manifest.get("heldout_lake_groups", []),
    )
    val_lakes = manifest.get("val_lake_ids") or train_lakes
    test_lakes = manifest.get("test_lake_ids", [])
    train_groups = manifest.get("train_lake_groups") or _unique(
        [lake["lake_group"] for lake in manifest["lakes"] if lake["lake_id"] in set(train_lakes)]
    )
    val_groups = manifest.get("val_lake_groups") or train_groups
    test_groups = manifest.get("test_lake_groups") or manifest.get("heldout_lake_groups", [])
    return {
        "experiment_id": experiment_id,
        "level": str(manifest.get("level", "")),
        "hypothesis": str(manifest.get("hypothesis", "")),
        "code_hash": str(manifest.get("code_hash", "")),
        "data_version": str(manifest.get("data_version", "")),
        "manifest": _path_string(manifest_path),
        "split_file": str(manifest.get("split_file") or _path_string(split_path)),
        "split_mode": str(manifest.get("split_mode", "")),
        "train_lakes": ",".join(train_lakes),
        "val_lakes": ",".join(val_lakes),
        "test_lakes": ",".join(test_lakes),
        "train_lake_groups": ",".join(train_groups),
        "val_lake_groups": ",".join(val_groups),
        "test_lake_groups": ",".join(test_groups),
        "heldout_lake_groups": ",".join(manifest.get("heldout_lake_groups", test_groups)),
        "support_query_policy": str(manifest.get("support_query_policy", "")),
        "model_variant": "cleanphys_fewshot",
        "kz_variant": "background_plus_gated_turbulent",
        "kd_variant": "optical_prior_regularized",
        "lst_variant": f"night_quality_dropout_{manifest.get('lst_feature_dropout_probability', 0.0):.2f}",
        "segment_variant": "horizon_weighted_roll60",
        "adapter_variant": "fewshot_support3",
        "process_pretrain": "none",
        "primary_metric": str(manifest.get("primary_transfer_metric", PRIMARY_TRANSFER_METRIC)),
        "secondary_physics_metrics": ";".join(manifest.get("secondary_physics_metrics", [])),
        "seed": "" if seed is None else str(seed),
        "checkpoint_selection": str(manifest.get("checkpoint_selection", "")),
        "status": status,
        "main_result_json": _path_string(output_dir / "global_state_forecaster_split_summary.json"),
        "result_path": _path_string(output_dir),
        "notes": "generated by prepare_recon_roadmap.py; formal GPU training requires approval",
    }


def migrate_existing_r10_manifests(*, v9_root: Path, code_version: str) -> list[dict]:
    manifests: list[dict] = []
    for path in sorted((v9_root / "experiments" / "manifests").glob("R10_CLEANPHYS_FEWSHOT*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        is_smoke = "smoke" in payload.get("experiment_id", payload.get("experiment", ""))
        payload.setdefault("level", "L1_smoke" if is_smoke else "R10_formal_short")
        payload.setdefault(
            "hypothesis",
            "Clean-physics few-shot RECON keeps locked heldout diagnostic and selects checkpoints by validation rolling/few-shot metrics.",
        )
        payload.setdefault("code_hash", code_version)
        payload.setdefault("split_file", "inline:R10_heldout_lake_groups")
        payload.setdefault("split_contract", "R10 heldout lake groups are diagnostic only; checkpoint uses validation rolling/few-shot metrics.")
        payload.setdefault("test_lake_groups", payload.get("heldout_lake_groups", []))
        payload.setdefault("support_query_policy", "episodic support profiles precede query windows; locked heldout is final diagnostic only")
        _write_json(path, payload)
        manifests.append({"path": path, "payload": payload})
    return manifests


def build_r10_registry_rows_from_existing(manifests: list[dict], *, v9_root: Path) -> list[dict[str, str]]:
    rows = []
    for item in manifests:
        payload = item["payload"]
        experiment_id = payload.get("experiment_id") or payload.get("experiment")
        if not experiment_id:
            continue
        status = "queued" if "smoke" in experiment_id else "needs_approval"
        seed = payload.get("seed")
        rows.append(
            _registry_row_from_manifest(
                experiment_id=experiment_id,
                manifest=payload,
                manifest_path=item["path"],
                split_path=item["path"],
                seed=int(seed) if str(seed).strip().isdigit() else None,
                output_dir=v9_root / "experiments" / experiment_id,
                status=status,
            )
        )
    return rows


def build_l3_registry_rows(*, manifest: dict, manifest_path: Path, split_path: Path, v9_root: Path) -> list[dict[str, str]]:
    rows = []
    for seed in L3_SEEDS:
        experiment_id = f"{L3_EXPERIMENT_BASE}_seed{seed:02d}"
        rows.append(
            _registry_row_from_manifest(
                experiment_id=experiment_id,
                manifest=manifest,
                manifest_path=manifest_path,
                split_path=split_path,
                seed=seed,
                output_dir=v9_root / "results" / experiment_id,
                status="needs_approval",
            )
        )
    return rows


def write_pipeline_queue_rows(*, pipeline_root: Path, v9_root: Path) -> dict[str, list[str]]:
    experiment_fields = [
        "queue_id",
        "task_type",
        "experiment_id",
        "manifest",
        "output_dir",
        "status",
        "approval_required",
        "approval_reason",
        "remote_host",
        "remote_port",
        "remote_path",
        "pid",
        "log_path",
        "checkpoint_path",
        "notes",
    ]
    experiment_rows = []
    for seed in L3_SEEDS:
        experiment_id = f"{L3_EXPERIMENT_BASE}_seed{seed:02d}"
        experiment_rows.append(
            {
                "queue_id": f"exp-l3-local34-seed{seed:02d}",
                "task_type": "formal_training",
                "experiment_id": experiment_id,
                "manifest": "experiments/manifests_clean/L3_local_core/M2_LOCAL34_GROUPHELDOUT_V1.json",
                "output_dir": f"results/{experiment_id}",
                "status": "needs_approval",
                "approval_required": "true",
                "approval_reason": "Formal L3 LOCAL34 GPU training waits for R10 smoke go/no-go and explicit approval.",
                "remote_host": "connect.bjb2.seetacloud.com",
                "remote_port": "20448",
                "remote_path": "/root/pinn_r10/ninth",
                "pid": "",
                "log_path": f"results/{experiment_id}/train.log",
                "checkpoint_path": f"results/{experiment_id}/best_by_val_rolling.pt",
                "notes": "Do not start before smoke passes and user approves L3 group-heldout run.",
            }
        )
    _upsert_csv(pipeline_root / "experiment_queue.csv", experiment_rows, experiment_fields)

    download_fields = [
        "queue_id",
        "task_type",
        "lake_id",
        "status",
        "approval_required",
        "source",
        "output_dir",
        "notes",
    ]
    download_rows = [
        {
            "queue_id": "data-l0-era5-missing-001",
            "task_type": "repair_missing_era5",
            "lake_id": ",".join(LOCAL34_DOCUMENT_UNUSABLE_IDS),
            "status": "needs_approval",
            "approval_required": "true",
            "source": "ERA5 standard-input repair plan",
            "output_dir": _path_string(DEFAULT_STANDARD_INPUT_ROOT),
            "notes": "Repair only after approval; current latest audit shows some IDs may already be fixed, so rerun L0 audit first.",
        },
        {
            "queue_id": "data-l0-lst-v8-flags-001",
            "task_type": "overwrite_standard_inputs",
            "lake_id": "all_standard_lake_years",
            "status": "needs_approval",
            "approval_required": "true",
            "source": "satellite LST v8 semantic flag migration",
            "output_dir": _path_string(DEFAULT_STANDARD_INPUT_ROOT),
            "notes": "Add LSWT/IST/quality/fill/observed/ice_fraction fields only after approval; no fake LST labels.",
        },
        {
            "queue_id": "data-l0-reservoir-hydrology-001",
            "task_type": "download_new_data",
            "lake_id": "reservoir_subset",
            "status": "needs_approval",
            "approval_required": "true",
            "source": "reservoir hydrology candidate sources",
            "output_dir": _path_string(DEFAULT_STANDARD_INPUT_ROOT),
            "notes": "Plan inflow/outflow/water-level/withdrawal-depth fields; do not claim reservoir mechanisms until supported.",
        },
    ]
    _upsert_csv(pipeline_root / "download_queue.csv", download_rows, download_fields)
    return {
        "experiment_queue_ids": [row["queue_id"] for row in experiment_rows],
        "download_queue_ids": [row["queue_id"] for row in download_rows],
    }


def prepare_recon_roadmap_assets(
    *,
    v9_root: Path = REPO_ROOT,
    pipeline_root: Path = DEFAULT_PIPELINE_ROOT,
    standard_input_root: Path = DEFAULT_STANDARD_INPUT_ROOT,
    readiness_csv: Path | None = None,
) -> dict:
    standard_input_root = Path(standard_input_root)
    readiness_path = resolve_readiness_csv(standard_input_root, Path(readiness_csv) if readiness_csv else None)
    data_version = data_version_from_readiness(readiness_path)
    code_version = code_hash(v9_root)
    git_hash = git_short_hash(v9_root)
    if git_hash:
        code_version = f"{code_version};git:{git_hash}"
    lakes, skipped_rows = load_usable_lakes(standard_input_root, readiness_path, lst_kind="night")
    directories = ensure_roadmap_directories(v9_root=v9_root, pipeline_root=pipeline_root)
    split = build_local34_split(
        lakes,
        standard_input_root=standard_input_root,
        readiness_csv=readiness_path,
        data_version=data_version,
    )
    split_path = v9_root / "experiments" / "splits" / f"{LOCAL34_SPLIT_ID}.json"
    _write_json(split_path, split)
    manifest_path = v9_root / "experiments" / "manifests_clean" / "L3_local_core" / L3_MANIFEST_NAME
    manifest = build_l3_manifest(
        split=split,
        lakes=lakes,
        split_path=split_path,
        data_version=data_version,
        code_version=code_version,
    )
    _write_json(manifest_path, manifest)

    migrated_r10 = migrate_existing_r10_manifests(v9_root=v9_root, code_version=code_version)
    registry_rows = build_r10_registry_rows_from_existing(migrated_r10, v9_root=v9_root)
    registry_rows.extend(build_l3_registry_rows(manifest=manifest, manifest_path=manifest_path, split_path=split_path, v9_root=v9_root))
    registry_path = v9_root / "experiments" / "registry.csv"
    upsert_registry(registry_path, registry_rows)
    queue_summary = write_pipeline_queue_rows(pipeline_root=pipeline_root, v9_root=v9_root)
    return {
        "v9_root": _path_string(v9_root),
        "pipeline_root": _path_string(pipeline_root),
        "standard_input_root": _path_string(standard_input_root),
        "readiness_csv": _path_string(readiness_path),
        "data_version": data_version,
        "code_hash": code_version,
        "usable_lake_count": len(lakes),
        "skipped_lakes": skipped_rows,
        "local34_lake_count": split["counts"]["total_lake_years"],
        "local34_train_groups": split["train_lake_groups"],
        "local34_val_groups": split["val_lake_groups"],
        "local34_test_groups": split["test_lake_groups"],
        "split_path": _path_string(split_path),
        "l3_manifest_path": _path_string(manifest_path),
        "registry_path": _path_string(registry_path),
        "directories": directories,
        "queues": queue_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Lake-PINN v9 RECON L0-L9 roadmap assets.")
    parser.add_argument("--v9-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--standard-input-root", type=Path, default=DEFAULT_STANDARD_INPUT_ROOT)
    parser.add_argument("--readiness-csv", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = prepare_recon_roadmap_assets(
        v9_root=args.v9_root,
        pipeline_root=args.pipeline_root,
        standard_input_root=args.standard_input_root,
        readiness_csv=args.readiness_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
