import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_r10_experiments import build_experiment_assets


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


def test_prepare_r10_assets_filters_usable_lakes_and_writes_registry(tmp_path):
    standard_root = tmp_path / "data" / "_standard_inputs"
    standard_root.mkdir(parents=True)
    _write_standard_lake(standard_root, "train_lake_2020")
    _write_standard_lake(standard_root, "held_lake_2021")
    readiness = standard_root / "_local_standard_inputs_training_readiness_20260606.csv"
    readiness.write_text(
        "\n".join(
            [
                "lake_id,lake_group,year,usable_basic,warnings,metadata_max_depth_m,profile_max_depth_m",
                "train_lake_2020,train_lake,2020,True,,10.0,9.5",
                "held_lake_2021,held_lake,2021,True,,11.0,10.5",
                "missing_lake_2022,missing_lake,2022,False,missing_file,12.0,11.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "experiments"
    summary = build_experiment_assets(
        standard_input_root=standard_root,
        readiness_csv=readiness,
        output_root=output_root,
        heldout_ids=("held_lake_2021",),
        seeds=(1, 2),
        write_launch=False,
    )

    assert summary["usable_lake_count"] == 2
    assert summary["heldout_lake_groups"] == ["held_lake"]
    assert {item["lake_id"] for item in summary["skipped_lakes"]} == {"missing_lake_2022"}
    assert len(summary["manifest_paths"]) == 3

    smoke_manifest = json.loads(
        (output_root / "manifests" / "R10_CLEANPHYS_FEWSHOT_smoke.json").read_text(encoding="utf-8")
    )
    assert smoke_manifest["full_eval_every_epochs"] == 10
    assert smoke_manifest["experiment_id"] == "R10_CLEANPHYS_FEWSHOT_smoke"
    assert smoke_manifest["level"] == "L1_smoke"
    assert smoke_manifest["hypothesis"]
    assert smoke_manifest["split_file"] == "inline:R10_heldout_lake_groups"
    assert smoke_manifest["checkpoint_selection"] == "best_by_val_rolling"
    assert smoke_manifest["heldout_lake_ids"] == ["held_lake_2021"]
    assert smoke_manifest["primary_transfer_metric"]
    assert smoke_manifest["go_no_go_criteria"]
    assert smoke_manifest["code_hash"].startswith("lake_pinn_sha256:")
    assert smoke_manifest["residual_limit_c"] == 0.15
    assert smoke_manifest["warm_season_column_heat_content_weight"] == 0.0
    assert smoke_manifest["test_lake_ids"] == ["held_lake_2021"]
    assert smoke_manifest["heldout_lake_groups"] == ["held_lake"]
    train_lake = {lake["lake_id"]: lake for lake in smoke_manifest["lakes"]}["train_lake_2020"]
    assert train_lake["era5"].endswith("/train_lake_2020/era5_for_model.csv")
    assert "鏁版嵁" not in train_lake["era5"]

    seed_manifest = json.loads(
        (output_root / "manifests" / "R10_CLEANPHYS_FEWSHOT_seed2.json").read_text(encoding="utf-8")
    )
    assert seed_manifest["seed"] == 2
    assert seed_manifest["full_eval_every_epochs"] == 20
    assert seed_manifest["segment_rollout_samples_per_lake"] == 12

    with (output_root / "registry.csv").open("r", encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in registry_rows] == [
        "R10_CLEANPHYS_FEWSHOT_smoke",
        "R10_CLEANPHYS_FEWSHOT_seed1",
        "R10_CLEANPHYS_FEWSHOT_seed2",
    ]
    assert registry_rows[0]["checkpoint_selection"] == "best_by_val_rolling"
    assert registry_rows[0]["level"] == "L1_smoke"
    assert registry_rows[0]["split_file"] == "inline:R10_heldout_lake_groups"
    assert registry_rows[0]["train_lake_groups"] == "train_lake"
    assert registry_rows[0]["test_lake_groups"] == "held_lake"
    assert registry_rows[0]["status"] == "queued"
    assert registry_rows[1]["status"] == "needs_approval"
    assert registry_rows[1]["seed"] == "1"
