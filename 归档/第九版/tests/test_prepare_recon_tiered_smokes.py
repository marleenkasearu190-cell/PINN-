import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_recon_tiered_smokes import (
    MEDIUM_EXCLUDED_STRESS_GROUPS,
    MEDIUM_SMOKE_ID,
    MEDIUM_SMOKE_LAKE_IDS,
    SPEED_SMOKE_ID,
    SPEED_SMOKE_LAKE_IDS,
    prepare_recon_tiered_smokes,
)


def _lake_group(lake_id: str) -> str:
    return lake_id.rsplit("_", 1)[0]


def _write_standard_lake(root: Path, lake_id: str) -> None:
    lake_root = root / lake_id
    lake_root.mkdir(parents=True)
    for name in (
        "era5_for_model.csv",
        "lst_night_for_model.csv",
        "profile_for_model.csv",
    ):
        (lake_root / name).write_text("Date,value\n2020-01-01,1\n", encoding="utf-8")
    (lake_root / "metadata.json").write_text(
        json.dumps({"lake_id": lake_id, "max_depth_m": 10.0}),
        encoding="utf-8",
    )


def _write_readiness(root: Path, lake_ids: list[str]) -> Path:
    readiness = root / "_local_standard_inputs_training_readiness_20260606.csv"
    lines = ["lake_id,lake_group,year,prepare_ok,metadata_max_depth_m,profile_max_depth_m"]
    for lake_id in lake_ids:
        year = lake_id.rsplit("_", 1)[1]
        lines.append(f"{lake_id},{_lake_group(lake_id)},{year},True,10.0,9.5")
    readiness.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readiness


def _standard_root_with_tiered_lakes(tmp_path: Path) -> tuple[Path, Path]:
    standard_root = tmp_path / "data" / "_standard_inputs"
    standard_root.mkdir(parents=True)
    lake_ids = sorted(set(SPEED_SMOKE_LAKE_IDS) | set(MEDIUM_SMOKE_LAKE_IDS))
    for lake_id in lake_ids:
        _write_standard_lake(standard_root, lake_id)
    readiness = _write_readiness(standard_root, lake_ids)
    return standard_root, readiness


def test_prepare_recon_tiered_smokes_dry_run_does_not_write(tmp_path):
    standard_root, readiness = _standard_root_with_tiered_lakes(tmp_path)
    v9_root = tmp_path / "ninth"
    pipeline_root = tmp_path / "pipeline"

    summary = prepare_recon_tiered_smokes(
        v9_root=v9_root,
        pipeline_root=pipeline_root,
        standard_input_root=standard_root,
        readiness_csv=readiness,
        base_manifest=None,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert [item["lake_count"] for item in summary["experiments"]] == [12, 24]
    assert not (v9_root / "experiments" / "registry.csv").exists()
    assert not (pipeline_root / "experiment_queue.csv").exists()
    assert not (v9_root / "experiments" / "manifests_clean" / "L1_smoke").exists()


def test_prepare_recon_tiered_smokes_writes_manifests_registry_and_queue(tmp_path):
    standard_root, readiness = _standard_root_with_tiered_lakes(tmp_path)
    v9_root = tmp_path / "ninth"
    pipeline_root = tmp_path / "pipeline"

    summary = prepare_recon_tiered_smokes(
        v9_root=v9_root,
        pipeline_root=pipeline_root,
        standard_input_root=standard_root,
        readiness_csv=readiness,
        base_manifest=None,
    )

    assert summary["dry_run"] is False
    speed_manifest = json.loads(
        (v9_root / "experiments" / "manifests_clean" / "L1_smoke" / f"{SPEED_SMOKE_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    medium_manifest = json.loads(
        (v9_root / "experiments" / "manifests_clean" / "L1_smoke" / f"{MEDIUM_SMOKE_ID}.json").read_text(
            encoding="utf-8"
        )
    )

    assert [lake["lake_id"] for lake in speed_manifest["lakes"]] == list(SPEED_SMOKE_LAKE_IDS)
    assert speed_manifest["epochs"] == 5
    assert speed_manifest["depth_points"] == 24
    assert speed_manifest["full_eval_every_epochs"] == 5
    assert speed_manifest["export_after_training"] == "off"
    assert speed_manifest["heldout_lake_ids"] == ["lacawac_2016", "carvins_cove_2022", "lake_maggiore_2024"]

    assert [lake["lake_id"] for lake in medium_manifest["lakes"]] == list(MEDIUM_SMOKE_LAKE_IDS)
    assert medium_manifest["epochs"] == 10
    assert medium_manifest["depth_points"] == 30
    assert medium_manifest["full_eval_every_epochs"] == 10
    assert medium_manifest["heldout_lake_ids"] == []
    assert not ({lake["lake_group"] for lake in medium_manifest["lakes"]} & MEDIUM_EXCLUDED_STRESS_GROUPS)

    with (v9_root / "experiments" / "registry.csv").open("r", encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in registry_rows] == [SPEED_SMOKE_ID, MEDIUM_SMOKE_ID]
    assert [row["status"] for row in registry_rows] == [
        "queued_after_current_smoke",
        "queued_after_speed_smoke_pass",
    ]
    assert registry_rows[0]["checkpoint_selection"] == "best_by_val_rolling"
    assert registry_rows[1]["level"] == "L1_smoke"

    with (pipeline_root / "experiment_queue.csv").open("r", encoding="utf-8", newline="") as handle:
        queue_rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in queue_rows] == [SPEED_SMOKE_ID, MEDIUM_SMOKE_ID]
    assert [row["task_type"] for row in queue_rows] == ["smoke_training", "smoke_training"]
    assert [row["status"] for row in queue_rows] == [
        "queued_after_current_smoke",
        "queued_after_speed_smoke_pass",
    ]
    assert all(row["approval_required"] == "false" for row in queue_rows)
    assert all(not row["pid"] for row in queue_rows)
