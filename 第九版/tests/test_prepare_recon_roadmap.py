import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_recon_roadmap import (
    L3_EXPERIMENT_BASE,
    LOCAL34_STRESS_GROUPS,
    LOCAL34_TEST_GROUPS,
    LOCAL34_TRAIN_GROUPS,
    LOCAL34_VAL_GROUPS,
    prepare_recon_roadmap_assets,
)


LOCAL34_LAKES = [
    ("barco_2020", "barco", 2020),
    ("beaverdam_reservoir_2021", "beaverdam_reservoir", 2021),
    ("beaverdam_reservoir_2022", "beaverdam_reservoir", 2022),
    ("beaverdam_reservoir_2023", "beaverdam_reservoir", 2023),
    ("crystal_bog_2017", "crystal_bog", 2017),
    ("crystal_bog_2018", "crystal_bog", 2018),
    ("crystal_bog_2019", "crystal_bog", 2019),
    ("crystal_bog_2020", "crystal_bog", 2020),
    ("falling_creek_reservoir_2019", "falling_creek_reservoir", 2019),
    ("falling_creek_reservoir_2021", "falling_creek_reservoir", 2021),
    ("falling_creek_reservoir_2022", "falling_creek_reservoir", 2022),
    ("falling_creek_reservoir_2023", "falling_creek_reservoir", 2023),
    ("green_lake_4_2019", "green_lake_4", 2019),
    ("kinneret_2006", "kinneret", 2006),
    ("lake_washington_2009", "lake_washington", 2009),
    ("lough_feeagh_2004", "lough_feeagh", 2004),
    ("lough_feeagh_2005", "lough_feeagh", 2005),
    ("mendota_2018", "mendota", 2018),
    ("mendota_2019", "mendota", 2019),
    ("mendota_2020", "mendota", 2020),
    ("sparkling_2002", "sparkling", 2002),
    ("sparkling_2003", "sparkling", 2003),
    ("sparkling_2004", "sparkling", 2004),
    ("sparkling_2005", "sparkling", 2005),
    ("trout_bog_2011", "trout_bog", 2011),
    ("trout_lake_2005", "trout_lake", 2005),
    ("erken_2019", "erken", 2019),
    ("erken_2020", "erken", 2020),
    ("mohonk_2017", "mohonk", 2017),
    ("carvins_cove_2022", "carvins_cove", 2022),
    ("lacawac_2016", "lacawac", 2016),
    ("el_val_2019", "el_val", 2019),
    ("el_val_2022", "el_val", 2022),
    ("namco_2012", "namco", 2012),
    ("lough_feeagh_2016", "lough_feeagh", 2016),
    ("kivu_2002", "kivu", 2002),
    ("lake_maggiore_2024", "lake_maggiore", 2024),
    ("suggs_2022", "suggs", 2022),
    ("sunapee_2022", "sunapee", 2022),
    ("toolik_2009", "toolik", 2009),
]


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


def test_prepare_recon_roadmap_writes_local34_split_manifest_registry_and_queues(tmp_path):
    standard_root = tmp_path / "data" / "_standard_inputs"
    standard_root.mkdir(parents=True)
    for lake_id, _, _ in LOCAL34_LAKES:
        _write_standard_lake(standard_root, lake_id)
    readiness = standard_root / "_local_standard_inputs_training_readiness_20260606_151147.csv"
    lines = [
        "lake_id,lake_group,year,prepare_ok,metadata_max_depth_m,profile_max_depth_m",
        *[
            f"{lake_id},{group},{year},True,10.0,9.5"
            for lake_id, group, year in LOCAL34_LAKES
        ],
    ]
    readiness.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v9_root = tmp_path / "ninth"
    pipeline_root = tmp_path / "pipeline"
    summary = prepare_recon_roadmap_assets(
        v9_root=v9_root,
        pipeline_root=pipeline_root,
        standard_input_root=standard_root,
        readiness_csv=readiness,
    )

    assert summary["local34_lake_count"] == 34
    split = json.loads((v9_root / "experiments" / "splits" / "LOCAL34_GROUPHELDOUT_V1.json").read_text(encoding="utf-8"))
    assert split["train_lake_groups"] == list(LOCAL34_TRAIN_GROUPS)
    assert split["val_lake_groups"] == list(LOCAL34_VAL_GROUPS)
    assert split["test_lake_groups"] == list(LOCAL34_TEST_GROUPS)
    assert set(split["excluded_stress_lake_groups"]) == set(LOCAL34_STRESS_GROUPS)
    assert not (set(split["all_lake_ids"]) & {"lough_feeagh_2016", "kivu_2002", "lake_maggiore_2024"})

    manifest = json.loads(
        (v9_root / "experiments" / "manifests_clean" / "L3_local_core" / "M2_LOCAL34_GROUPHELDOUT_V1.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["experiment_id"] == L3_EXPERIMENT_BASE
    assert manifest["level"] == "L3_local_multilake_clean_physics"
    assert manifest["checkpoint_selection"] == "best_by_val_rolling"
    assert manifest["train_lake_groups"] == list(LOCAL34_TRAIN_GROUPS)
    assert manifest["val_lake_groups"] == list(LOCAL34_VAL_GROUPS)
    assert manifest["test_lake_groups"] == list(LOCAL34_TEST_GROUPS)

    with (v9_root / "experiments" / "registry.csv").open("r", encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in registry_rows] == [
        f"{L3_EXPERIMENT_BASE}_seed01",
        f"{L3_EXPERIMENT_BASE}_seed02",
        f"{L3_EXPERIMENT_BASE}_seed03",
    ]
    assert registry_rows[0]["status"] == "needs_approval"
    assert registry_rows[0]["split_file"].endswith("/LOCAL34_GROUPHELDOUT_V1.json")
    assert registry_rows[0]["train_lake_groups"] == ",".join(LOCAL34_TRAIN_GROUPS)

    with (pipeline_root / "experiment_queue.csv").open("r", encoding="utf-8", newline="") as handle:
        experiment_queue = list(csv.DictReader(handle))
    assert [row["status"] for row in experiment_queue] == ["needs_approval", "needs_approval", "needs_approval"]

    with (pipeline_root / "download_queue.csv").open("r", encoding="utf-8", newline="") as handle:
        download_queue = list(csv.DictReader(handle))
    assert {row["queue_id"] for row in download_queue} == {
        "data-l0-era5-missing-001",
        "data-l0-lst-v8-flags-001",
        "data-l0-reservoir-hydrology-001",
    }
