from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_STATUSES = {"queued", "running", "completed", "failed", "blocked", "needs_approval"}
APPROVAL_REQUIRED_TASK_TYPES = {
    "download_new_data",
    "overwrite_standard_inputs",
    "modify_locked_split",
    "formal_training",
    "long_gpu_training",
    "model_structure_change",
}
REQUIRED_MANIFEST_FIELDS = (
    "experiment_id",
    "level",
    "hypothesis",
    "code_hash",
    "data_version",
    "split_file",
    "checkpoint_selection",
    "heldout_lake_ids",
    "heldout_lake_groups",
    "primary_transfer_metric",
    "go_no_go_criteria",
)
REQUIRED_REGISTRY_FIELDS = (
    "experiment_id",
    "level",
    "hypothesis",
    "code_hash",
    "data_version",
    "manifest",
    "split_file",
    "checkpoint_selection",
    "status",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return bool(str(value).strip())


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def validate_status_rows(rows: list[dict[str, str]], *, source: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        status = str(row.get("status", "")).strip()
        if status and status not in VALID_STATUSES:
            errors.append(f"{source}:{index} has invalid status '{status}'")
    return errors


def validate_registry_contract(rows: list[dict[str, str]], *, source: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        for field in REQUIRED_REGISTRY_FIELDS:
            if not _non_empty(row.get(field)):
                errors.append(f"{source}:{index} missing required registry field '{field}'")
    return errors


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_contract(manifest: dict, *, source: str) -> list[str]:
    errors: list[str] = []
    level = str(manifest.get("level", "")).strip()
    for field in REQUIRED_MANIFEST_FIELDS:
        if field in {"heldout_lake_ids", "heldout_lake_groups"} and level == "L1_smoke" and field in manifest:
            continue
        if not _non_empty(manifest.get(field)):
            errors.append(f"{source} missing required manifest field '{field}'")
    if manifest.get("checkpoint_selection") != "best_by_val_rolling":
        errors.append(f"{source} checkpoint_selection must be best_by_val_rolling")
    return errors


def _manifest_group_sets(manifest: dict) -> dict[str, list[str]]:
    heldout_groups = _string_list(manifest.get("heldout_lake_groups"))
    train_groups = _string_list(manifest.get("train_lake_groups"))
    if not train_groups:
        heldout_group_set = set(heldout_groups)
        train_groups = _unique(
            [
                str(lake.get("lake_group", "")).strip()
                for lake in manifest.get("lakes", [])
                if str(lake.get("lake_group", "")).strip()
                and str(lake.get("lake_group", "")).strip() not in heldout_group_set
            ]
        )
    val_groups = _string_list(manifest.get("val_lake_groups"))
    test_groups = _string_list(manifest.get("test_lake_groups")) or heldout_groups
    return {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
        "heldout": heldout_groups,
    }


def validate_manifest_group_contract(manifest: dict, *, source: str) -> list[str]:
    groups = _manifest_group_sets(manifest)
    train = set(groups["train"])
    val = set(groups["val"])
    test = set(groups["test"])
    errors: list[str] = []
    train_test_overlap = sorted(train & test)
    if train_test_overlap:
        errors.append(f"{source} train/test lake-group leakage: {','.join(train_test_overlap)}")
    if groups["val"]:
        train_val_overlap = sorted(train & val)
        val_test_overlap = sorted(val & test)
        if train_val_overlap:
            errors.append(f"{source} train/validation lake-group leakage: {','.join(train_val_overlap)}")
        if val_test_overlap:
            errors.append(f"{source} validation/test lake-group leakage: {','.join(val_test_overlap)}")
    return errors


def validate_locked_split(manifest: dict, locked_rows: list[dict[str, str]], *, source: str) -> list[str]:
    group_sets = _manifest_group_sets(manifest)
    train_groups = set(group_sets["train"])
    heldout_groups = set(group_sets["heldout"])
    heldout_ids = set(_string_list(manifest.get("heldout_lake_ids") or manifest.get("test_lake_ids")))
    lake_groups = {str(lake.get("lake_id", "")): str(lake.get("lake_group", "")) for lake in manifest.get("lakes", [])}
    errors: list[str] = []
    for row in locked_rows:
        role = str(row.get("role", "")).strip().lower()
        if role not in {"locked_test", "test", "refresh_validation"}:
            continue
        lake_id = str(row.get("lake_id", "")).strip()
        lake_group = str(row.get("lake_group", "") or lake_groups.get(lake_id, "")).strip()
        if not lake_id and not lake_group:
            continue
        if lake_group and lake_group in train_groups:
            errors.append(f"{source} would train on locked {role} group {lake_group}")
        if role in {"locked_test", "test"} and lake_id in heldout_ids and lake_group and lake_group not in heldout_groups:
            errors.append(f"{source} locked test lake {lake_id} is held out without locked group {lake_group}")
    return errors


def validate_manifest_path(manifest_path: Path | None, locked_rows: list[dict[str, str]], *, source: str) -> list[str]:
    if manifest_path is None:
        return [f"{source} missing manifest path"]
    if not manifest_path.exists():
        return [f"{source} manifest not found: {manifest_path}"]
    manifest = load_manifest(manifest_path)
    return (
        validate_manifest_contract(manifest, source=str(manifest_path))
        + validate_manifest_group_contract(manifest, source=str(manifest_path))
        + validate_locked_split(
            manifest,
            locked_rows,
            source=str(manifest_path),
        )
    )


def validate_registry(registry_path: Path, locked_rows: list[dict[str, str]], *, repo_root: Path) -> list[str]:
    rows = read_csv_rows(registry_path)
    errors = validate_status_rows(rows, source=str(registry_path))
    errors.extend(validate_registry_contract(rows, source=str(registry_path)))
    for row in rows:
        status = str(row.get("status", "")).strip()
        if status not in {"queued", "running", "needs_approval"}:
            continue
        manifest_path = _resolve_path(row.get("manifest"), repo_root)
        source = row.get("experiment_id") or str(registry_path)
        errors.extend(validate_manifest_path(manifest_path, locked_rows, source=source))
    return errors


def _first_by_status(rows: list[dict[str, str]], status: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("status", "")).strip() == status:
            return row
    return None


def _approval_required(row: dict[str, str]) -> bool:
    task_type = str(row.get("task_type", "")).strip()
    return _csv_bool(row.get("approval_required")) or task_type in APPROVAL_REQUIRED_TASK_TYPES


def decide_next_action(*, repo_root: Path, v9_root: Path, pipeline_root: Path, dry_run: bool = True) -> dict:
    experiment_queue = read_csv_rows(pipeline_root / "experiment_queue.csv")
    download_queue = read_csv_rows(pipeline_root / "download_queue.csv")
    locked_rows = read_csv_rows(pipeline_root / "locked_test_lakes.csv")

    errors: list[str] = []
    errors.extend(validate_status_rows(experiment_queue, source=str(pipeline_root / "experiment_queue.csv")))
    errors.extend(validate_status_rows(download_queue, source=str(pipeline_root / "download_queue.csv")))
    errors.extend(validate_registry(v9_root / "experiments" / "registry.csv", locked_rows, repo_root=v9_root))
    for row in experiment_queue:
        if str(row.get("status", "")).strip() in {"queued", "running", "needs_approval"}:
            errors.extend(validate_manifest_path(
                _resolve_path(row.get("manifest"), v9_root),
                locked_rows,
                source=row.get("queue_id") or row.get("experiment_id") or "experiment_queue",
            ))
    if errors:
        return {
            "action": "blocked",
            "dry_run": bool(dry_run),
            "reason": "validation_failed",
            "errors": errors,
        }

    running = _first_by_status(experiment_queue, "running")
    if running:
        return {
            "action": "inspect_running",
            "dry_run": bool(dry_run),
            "queue_id": running.get("queue_id", ""),
            "experiment_id": running.get("experiment_id", ""),
            "remote_host": running.get("remote_host", ""),
            "remote_path": running.get("remote_path", ""),
            "pid": running.get("pid", ""),
            "log_path": running.get("log_path", ""),
        }

    needs_approval = _first_by_status(experiment_queue, "needs_approval")
    if needs_approval:
        return {
            "action": "needs_approval",
            "dry_run": bool(dry_run),
            "queue_id": needs_approval.get("queue_id", ""),
            "experiment_id": needs_approval.get("experiment_id", ""),
            "reason": needs_approval.get("approval_reason") or "task is waiting for explicit approval",
        }

    queued_experiment = _first_by_status(experiment_queue, "queued")
    if queued_experiment:
        if _approval_required(queued_experiment):
            return {
                "action": "needs_approval",
                "dry_run": bool(dry_run),
                "queue_id": queued_experiment.get("queue_id", ""),
                "experiment_id": queued_experiment.get("experiment_id", ""),
                "reason": "queued experiment requires approval before execution",
            }
        return {
            "action": "would_start_experiment" if dry_run else "start_experiment_not_implemented",
            "dry_run": bool(dry_run),
            "queue_id": queued_experiment.get("queue_id", ""),
            "experiment_id": queued_experiment.get("experiment_id", ""),
            "manifest": queued_experiment.get("manifest", ""),
        }

    queued_download = _first_by_status(download_queue, "queued")
    if queued_download:
        if _approval_required(queued_download):
            return {
                "action": "needs_approval",
                "dry_run": bool(dry_run),
                "queue_id": queued_download.get("queue_id", ""),
                "reason": "queued data task requires approval before execution",
            }
        return {
            "action": "would_run_data_audit_or_repair_plan" if dry_run else "data_action_not_implemented",
            "dry_run": bool(dry_run),
            "queue_id": queued_download.get("queue_id", ""),
        }

    return {
        "action": "idle",
        "dry_run": bool(dry_run),
        "reason": "no queued or running work",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Lake-PINN controlled automation controller.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--v9-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2] / "pipeline")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    result = decide_next_action(
        repo_root=args.repo_root,
        v9_root=args.v9_root,
        pipeline_root=args.pipeline_root,
        dry_run=args.dry_run,
    )
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["action"])
        for key, value in result.items():
            if key != "action":
                print(f"{key}: {value}")
    return 0 if result["action"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
