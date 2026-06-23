import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline_controller import decide_next_action


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest(path: Path, *, heldout_groups=None, extra=None) -> None:
    payload = {
        "experiment_id": "R10_TEST",
        "experiment": "R10_TEST",
        "level": "L1_smoke",
        "hypothesis": "test transfer hypothesis",
        "code_hash": "lake_pinn_sha256:test",
        "data_version": "standard_inputs_test",
        "split_file": "inline:test",
        "checkpoint_selection": "best_by_val_rolling",
        "heldout_lake_ids": ["held_2020"],
        "test_lake_ids": ["held_2020"],
        "heldout_lake_groups": heldout_groups if heldout_groups is not None else ["held"],
        "primary_transfer_metric": "0.5 * val_fewshot_rmse_30d + 0.5 * val_fewshot_rmse_60d",
        "go_no_go_criteria": ["history written", "best checkpoint written"],
        "lakes": [
            {"lake_id": "train_2020", "lake_group": "train"},
            {"lake_id": "held_2020", "lake_group": "held"},
        ],
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_pipeline(tmp_path: Path):
    repo_root = tmp_path / "PINN"
    v9_root = repo_root / "第九版"
    pipeline_root = repo_root / "pipeline"
    manifest = v9_root / "experiments" / "manifests" / "R10_TEST.json"
    _manifest(manifest)
    _write_csv(
        pipeline_root / "locked_test_lakes.csv",
        [{"lake_id": "held_2020", "lake_group": "held", "role": "locked_test", "notes": ""}],
        ["lake_id", "lake_group", "role", "notes"],
    )
    _write_csv(
        v9_root / "experiments" / "registry.csv",
        [
            {
                "experiment_id": "R10_TEST",
                "level": "L1_smoke",
                "hypothesis": "test transfer hypothesis",
                "code_hash": "lake_pinn_sha256:test",
                "data_version": "standard_inputs_test",
                "manifest": str(manifest),
                "split_file": "inline:test",
                "checkpoint_selection": "best_by_val_rolling",
                "status": "running",
            }
        ],
        ["experiment_id", "level", "hypothesis", "code_hash", "data_version", "manifest", "split_file", "checkpoint_selection", "status"],
    )
    _write_csv(pipeline_root / "download_queue.csv", [], ["queue_id", "task_type", "status"])
    return repo_root, v9_root, pipeline_root, manifest


def test_pipeline_controller_dry_run_inspects_running_experiment(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [
            {
                "queue_id": "exp-001",
                "task_type": "smoke_training",
                "experiment_id": "R10_TEST",
                "manifest": str(manifest),
                "output_dir": "experiments/R10_TEST",
                "status": "running",
                "approval_required": "false",
                "remote_host": "connect.example",
                "remote_path": "/root/pinn_r10/ninth",
                "pid": "123",
                "log_path": "experiments/R10_TEST/train.log",
            }
        ],
        [
            "queue_id",
            "task_type",
            "experiment_id",
            "manifest",
            "output_dir",
            "status",
            "approval_required",
            "remote_host",
            "remote_path",
            "pid",
            "log_path",
        ],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "inspect_running"
    assert result["experiment_id"] == "R10_TEST"
    assert result["pid"] == "123"


def test_pipeline_controller_blocks_missing_manifest_contract(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("primary_transfer_metric")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [{"queue_id": "exp-001", "task_type": "smoke_training", "experiment_id": "R10_TEST", "manifest": str(manifest), "status": "queued"}],
        ["queue_id", "task_type", "experiment_id", "manifest", "status"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "blocked"
    assert any("primary_transfer_metric" in error for error in result["errors"])


def test_pipeline_controller_allows_l1_smoke_empty_heldout_contract(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _manifest(
        manifest,
        heldout_groups=[],
        extra={
            "heldout_lake_ids": [],
            "test_lake_ids": [],
            "test_lake_groups": [],
            "lakes": [
                {"lake_id": "train_2020", "lake_group": "train"},
                {"lake_id": "val_2020", "lake_group": "val"},
            ],
        },
    )
    _write_csv(pipeline_root / "locked_test_lakes.csv", [], ["lake_id", "lake_group", "role", "notes"])
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [
            {
                "queue_id": "exp-001",
                "task_type": "smoke_training",
                "experiment_id": "R10_TEST",
                "manifest": str(manifest),
                "status": "running",
                "approval_required": "false",
                "pid": "123",
            }
        ],
        ["queue_id", "task_type", "experiment_id", "manifest", "status", "approval_required", "pid"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "inspect_running"


def test_pipeline_controller_blocks_missing_registry_contract(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _write_csv(
        v9_root / "experiments" / "registry.csv",
        [
            {
                "experiment_id": "R10_TEST",
                "level": "",
                "hypothesis": "test transfer hypothesis",
                "code_hash": "lake_pinn_sha256:test",
                "data_version": "standard_inputs_test",
                "manifest": str(manifest),
                "split_file": "inline:test",
                "checkpoint_selection": "best_by_val_rolling",
                "status": "queued",
            }
        ],
        ["experiment_id", "level", "hypothesis", "code_hash", "data_version", "manifest", "split_file", "checkpoint_selection", "status"],
    )
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [{"queue_id": "exp-001", "task_type": "smoke_training", "experiment_id": "R10_TEST", "manifest": str(manifest), "status": "queued"}],
        ["queue_id", "task_type", "experiment_id", "manifest", "status"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "blocked"
    assert any("missing required registry field 'level'" in error for error in result["errors"])


def test_pipeline_controller_blocks_locked_group_leakage(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _manifest(manifest, heldout_groups=[])
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [{"queue_id": "exp-001", "task_type": "smoke_training", "experiment_id": "R10_TEST", "manifest": str(manifest), "status": "queued"}],
        ["queue_id", "task_type", "experiment_id", "manifest", "status"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "blocked"
    assert any("would train on locked locked_test group held" in error for error in result["errors"])


def test_pipeline_controller_blocks_explicit_group_split_leakage(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _manifest(
        manifest,
        extra={
            "train_lake_groups": ["train", "held"],
            "val_lake_groups": ["val"],
            "test_lake_groups": ["held"],
        },
    )
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [{"queue_id": "exp-001", "task_type": "smoke_training", "experiment_id": "R10_TEST", "manifest": str(manifest), "status": "queued"}],
        ["queue_id", "task_type", "experiment_id", "manifest", "status"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "blocked"
    assert any("train/test lake-group leakage: held" in error for error in result["errors"])


def test_pipeline_controller_blocks_invalid_queue_status(tmp_path):
    repo_root, v9_root, pipeline_root, manifest = _base_pipeline(tmp_path)
    _write_csv(
        pipeline_root / "experiment_queue.csv",
        [{"queue_id": "exp-001", "task_type": "smoke_training", "experiment_id": "R10_TEST", "manifest": str(manifest), "status": "manifest_ready"}],
        ["queue_id", "task_type", "experiment_id", "manifest", "status"],
    )

    result = decide_next_action(repo_root=repo_root, v9_root=v9_root, pipeline_root=pipeline_root)

    assert result["action"] == "blocked"
    assert any("invalid status 'manifest_ready'" in error for error in result["errors"])
