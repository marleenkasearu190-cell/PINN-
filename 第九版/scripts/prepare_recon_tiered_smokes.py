from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[1]
if CANONICAL_REPO_ROOT.exists():
    REPO_ROOT = CANONICAL_REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_r10_experiments import (
    DEFAULT_HELDOUT_IDS,
    DEFAULT_STANDARD_INPUT_ROOT,
    GO_NO_GO_CRITERIA,
    ExperimentAsset,
    _base_manifest,
    _path_string,
    _registry_row,
    code_hash,
    data_version_from_readiness,
    git_short_hash,
    load_usable_lakes,
    resolve_readiness_csv,
    upsert_registry,
)


DEFAULT_PIPELINE_ROOT = REPO_ROOT.parent / "pipeline"
DEFAULT_BASE_MANIFEST = REPO_ROOT / "experiments" / "manifests" / "R10_CLEANPHYS_FEWSHOT_smoke.json"
REMOTE_HOST = "connect.bjb2.seetacloud.com"
REMOTE_PORT = "20448"
REMOTE_PATH = "/root/pinn_r10/ninth"
SPEED_SMOKE_ID = "RECON_L1_SPEED_SMOKE_v1"
MEDIUM_SMOKE_ID = "RECON_L1_MEDIUM_SMOKE_v1"

SPEED_SMOKE_LAKE_IDS = (
    "barco_2020",
    "beaverdam_reservoir_2022",
    "crystal_bog_2019",
    "falling_creek_reservoir_2022",
    "erken_2020",
    "kinneret_2006",
    "mendota_2019",
    "sparkling_2003",
    "trout_lake_2005",
    "lacawac_2016",
    "carvins_cove_2022",
    "lake_maggiore_2024",
)

MEDIUM_SMOKE_LAKE_IDS = (
    "barco_2020",
    "beaverdam_reservoir_2021",
    "beaverdam_reservoir_2022",
    "crystal_bog_2018",
    "crystal_bog_2019",
    "erken_2019",
    "erken_2020",
    "falling_creek_reservoir_2019",
    "falling_creek_reservoir_2021",
    "falling_creek_reservoir_2022",
    "green_lake_4_2019",
    "kinneret_2006",
    "lake_washington_2009",
    "lough_feeagh_2004",
    "lough_feeagh_2005",
    "mendota_2018",
    "mendota_2019",
    "mendota_2020",
    "mohonk_2017",
    "sparkling_2002",
    "sparkling_2003",
    "sparkling_2004",
    "trout_bog_2011",
    "trout_lake_2005",
)

MEDIUM_EXCLUDED_STRESS_GROUPS = {
    "kivu",
    "lake_maggiore",
    "suggs",
    "sunapee",
    "toolik",
}


@dataclass(frozen=True)
class TieredSmokeSpec:
    experiment_id: str
    lake_ids: tuple[str, ...]
    epochs: int
    depth_points: int
    history_window_days: int
    segment_rollout_max_days: int
    segment_rollout_samples_per_lake: int
    episodic_fewshot_samples_per_lake: int
    full_eval_every_epochs: int
    eval_every_epochs: int
    rolling_horizon_eval_max_starts: int
    status: str
    queue_id: str
    heldout_ids: tuple[str, ...]
    hypothesis: str
    notes: str


TIERED_SMOKE_SPECS = (
    TieredSmokeSpec(
        experiment_id=SPEED_SMOKE_ID,
        lake_ids=SPEED_SMOKE_LAKE_IDS,
        epochs=5,
        depth_points=24,
        history_window_days=14,
        segment_rollout_max_days=30,
        segment_rollout_samples_per_lake=2,
        episodic_fewshot_samples_per_lake=2,
        full_eval_every_epochs=5,
        eval_every_epochs=5,
        rolling_horizon_eval_max_starts=4,
        status="queued_after_current_smoke",
        queue_id="exp-l1-speed-smoke-v1",
        heldout_ids=DEFAULT_HELDOUT_IDS,
        hypothesis=(
            "A 12 lake-year L1 speed smoke can validate code, runtime, and best-by-validation "
            "rolling checkpoint closure without changing the RECON transfer claim."
        ),
        notes="L1 speed/debug smoke only; do not use as transfer evidence.",
    ),
    TieredSmokeSpec(
        experiment_id=MEDIUM_SMOKE_ID,
        lake_ids=MEDIUM_SMOKE_LAKE_IDS,
        epochs=10,
        depth_points=30,
        history_window_days=14,
        segment_rollout_max_days=45,
        segment_rollout_samples_per_lake=3,
        episodic_fewshot_samples_per_lake=3,
        full_eval_every_epochs=10,
        eval_every_epochs=5,
        rolling_horizon_eval_max_starts=6,
        status="queued_after_speed_smoke_pass",
        queue_id="exp-l1-medium-smoke-v1",
        heldout_ids=(),
        hypothesis=(
            "A 24 lake-year non-stress L1 medium smoke can validate runtime scaling and "
            "rolling/few-shot artifact closure before any formal LOCAL34 or R10 seed run."
        ),
        notes="L1 medium smoke only; no locked heldout and no science claim.",
    ),
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


def _upsert_queue(path: Path, new_rows: list[dict[str, str]], default_fields: list[str]) -> None:
    existing_rows, existing_fields = _read_csv(path)
    by_id = {row.get("queue_id", ""): row for row in existing_rows if row.get("queue_id")}
    ordered_ids = [row.get("queue_id", "") for row in existing_rows if row.get("queue_id")]
    for row in new_rows:
        queue_id = row["queue_id"]
        old_row = by_id.get(queue_id, {})
        merged = {**old_row, **row}
        if old_row.get("status"):
            merged["status"] = old_row["status"]
        by_id[queue_id] = merged
        if queue_id not in ordered_ids:
            ordered_ids.append(queue_id)
    fields = list(dict.fromkeys(existing_fields + default_fields + [field for row in new_rows for field in row]))
    _write_csv(path, [by_id[item] for item in ordered_ids if item in by_id], fields)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_lakes(lakes: list[dict], lake_ids: tuple[str, ...]) -> list[dict]:
    by_id = {lake["lake_id"]: lake for lake in lakes}
    missing = [lake_id for lake_id in lake_ids if lake_id not in by_id]
    if missing:
        raise ValueError(f"tiered smoke lake IDs missing from usable lakes: {', '.join(missing)}")
    return [by_id[lake_id] for lake_id in lake_ids]


def _validate_medium_non_stress(lakes: list[dict]) -> None:
    stress = sorted(
        {
            lake.get("lake_group", "")
            for lake in lakes
            if lake.get("lake_group", "") in MEDIUM_EXCLUDED_STRESS_GROUPS
        }
    )
    if stress:
        raise ValueError(f"medium smoke includes excluded stress groups: {', '.join(stress)}")


def _build_manifest(
    *,
    spec: TieredSmokeSpec,
    lakes: list[dict],
    data_version: str,
    code_version: str,
) -> dict:
    manifest = _base_manifest(
        experiment_id=spec.experiment_id,
        lakes=lakes,
        heldout_ids=spec.heldout_ids,
        stage="smoke",
        seed=1,
        data_version=data_version,
        code_version=code_version,
    )
    manifest.update(
        {
            "experiment_id": spec.experiment_id,
            "experiment": spec.experiment_id,
            "level": "L1_smoke",
            "hypothesis": spec.hypothesis,
            "experiment_stage": "l1_tiered_smoke",
            "tiered_smoke_role": "speed_debug" if spec.experiment_id == SPEED_SMOKE_ID else "medium_runtime",
            "split_file": f"inline:{spec.experiment_id}_selected_lake_years",
            "split_contract": (
                "L1 smoke validates runtime and artifact closure only; L3/L5 group-heldout "
                "experiments are required for transfer claims."
            ),
            "epochs": spec.epochs,
            "depth_points": spec.depth_points,
            "history_window_days": spec.history_window_days,
            "max_rollout_days": spec.segment_rollout_max_days,
            "segment_rollout_max_days": spec.segment_rollout_max_days,
            "segment_rollout_start_epoch": 1 if spec.epochs <= 5 else 2,
            "segment_rollout_ramp_epochs": 3 if spec.epochs <= 5 else 5,
            "segment_rollout_samples_per_lake": spec.segment_rollout_samples_per_lake,
            "episodic_fewshot_start_epoch": 1 if spec.epochs <= 5 else 2,
            "episodic_fewshot_ramp_epochs": 3 if spec.epochs <= 5 else 5,
            "episodic_fewshot_samples_per_lake": spec.episodic_fewshot_samples_per_lake,
            "rolling_horizon_eval_max_starts": spec.rolling_horizon_eval_max_starts,
            "checkpoint_every_epochs": spec.full_eval_every_epochs,
            "eval_every_epochs": spec.eval_every_epochs,
            "full_eval_every_epochs": spec.full_eval_every_epochs,
            "export_after_training": "off",
            "go_no_go_criteria": [
                "no runtime crash or NaN",
                "global_state_forecaster_training_history.csv is written",
                "best_by_val_rolling.pt and best_by_val_rolling_metrics.json are written",
                "history includes val_rolling_start_rmse_30d/60d and val_fewshot_rmse_30d/60d",
                "L1 smoke results are not used as formal transfer evidence",
                *list(GO_NO_GO_CRITERIA[-3:]),
            ],
        }
    )
    return manifest


def _registry_rows(
    *,
    specs_and_manifests: list[tuple[TieredSmokeSpec, dict, Path]],
    code_version: str,
    data_version: str,
    output_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec, manifest, manifest_path in specs_and_manifests:
        asset = ExperimentAsset(
            experiment_id=spec.experiment_id,
            manifest_path=manifest_path,
            output_dir=output_root / spec.experiment_id,
            epochs=spec.epochs,
            seed=manifest.get("seed"),
            stage="smoke",
        )
        row = _registry_row(
            asset=asset,
            manifest=manifest,
            manifest_path=manifest_path,
            code_version=code_version,
            data_version=data_version,
            output_root=output_root,
        )
        row["status"] = spec.status
        row["notes"] = f"{spec.notes}; epochs={spec.epochs}; full_eval_every_epochs={spec.full_eval_every_epochs}"
        rows.append(row)
    return rows


def _queue_rows(*, specs: tuple[TieredSmokeSpec, ...]) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        rows.append(
            {
                "queue_id": spec.queue_id,
                "task_type": "smoke_training",
                "experiment_id": spec.experiment_id,
                "manifest": f"experiments/manifests_clean/L1_smoke/{spec.experiment_id}.json",
                "output_dir": f"results/{spec.experiment_id}",
                "status": spec.status,
                "approval_required": "false",
                "approval_reason": "",
                "remote_host": REMOTE_HOST,
                "remote_port": REMOTE_PORT,
                "remote_path": REMOTE_PATH,
                "pid": "",
                "log_path": f"results/{spec.experiment_id}/train.log",
                "checkpoint_path": f"results/{spec.experiment_id}/best_by_val_rolling.pt",
                "notes": (
                    f"{spec.notes} Wait for the configured prerequisite smoke gate; "
                    "do not start formal seeds from this queue row."
                ),
            }
        )
    return rows


def prepare_recon_tiered_smokes(
    *,
    v9_root: Path = REPO_ROOT,
    pipeline_root: Path = DEFAULT_PIPELINE_ROOT,
    standard_input_root: Path = DEFAULT_STANDARD_INPUT_ROOT,
    readiness_csv: Path | None = None,
    base_manifest: Path | None = DEFAULT_BASE_MANIFEST,
    dry_run: bool = False,
) -> dict:
    v9_root = Path(v9_root)
    pipeline_root = Path(pipeline_root)
    standard_input_root = Path(standard_input_root)
    base_manifest_path = Path(base_manifest) if base_manifest else None
    readiness_path: Path | None = None
    data_version: str
    if readiness_csv is not None or list(standard_input_root.glob("*training_readiness*.csv")):
        readiness_path = resolve_readiness_csv(standard_input_root, Path(readiness_csv) if readiness_csv else None)
        data_version = data_version_from_readiness(readiness_path)
        usable_lakes, skipped_lakes = load_usable_lakes(standard_input_root, readiness_path, lst_kind="night")
    elif base_manifest_path is not None and base_manifest_path.exists():
        base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        data_version = str(base.get("data_version") or "standard_inputs_from_base_manifest")
        usable_lakes = list(base.get("lakes") or [])
        skipped_lakes = []
    else:
        readiness_path = resolve_readiness_csv(standard_input_root, Path(readiness_csv) if readiness_csv else None)
        data_version = data_version_from_readiness(readiness_path)
        usable_lakes, skipped_lakes = load_usable_lakes(standard_input_root, readiness_path, lst_kind="night")
    code_version = code_hash(v9_root)
    git_hash = git_short_hash(v9_root)
    if git_hash:
        code_version = f"{code_version};git:{git_hash}"

    manifest_dir = v9_root / "experiments" / "manifests_clean" / "L1_smoke"
    output_root = v9_root / "results"
    specs_and_manifests: list[tuple[TieredSmokeSpec, dict, Path]] = []
    for spec in TIERED_SMOKE_SPECS:
        selected_lakes = _select_lakes(usable_lakes, spec.lake_ids)
        if spec.experiment_id == MEDIUM_SMOKE_ID:
            _validate_medium_non_stress(selected_lakes)
        manifest = _build_manifest(
            spec=spec,
            lakes=selected_lakes,
            data_version=data_version,
            code_version=code_version,
        )
        specs_and_manifests.append((spec, manifest, manifest_dir / f"{spec.experiment_id}.json"))

    registry_rows = _registry_rows(
        specs_and_manifests=specs_and_manifests,
        code_version=code_version,
        data_version=data_version,
        output_root=output_root,
    )
    queue_rows = _queue_rows(specs=TIERED_SMOKE_SPECS)

    registry_path = v9_root / "experiments" / "registry.csv"
    queue_path = pipeline_root / "experiment_queue.csv"
    if not dry_run:
        for _, manifest, manifest_path in specs_and_manifests:
            _write_json(manifest_path, manifest)
        upsert_registry(registry_path, registry_rows)
        _upsert_queue(
            queue_path,
            queue_rows,
            [
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
            ],
        )

    return {
        "dry_run": dry_run,
        "v9_root": _path_string(v9_root),
        "pipeline_root": _path_string(pipeline_root),
        "standard_input_root": _path_string(standard_input_root),
        "readiness_csv": _path_string(readiness_path) if readiness_path is not None else "",
        "base_manifest": _path_string(base_manifest_path) if base_manifest_path is not None else "",
        "data_version": data_version,
        "code_hash": code_version,
        "usable_lake_count": len(usable_lakes),
        "skipped_lake_count": len(skipped_lakes),
        "manifest_paths": [_path_string(path) for _, _, path in specs_and_manifests],
        "registry_path": _path_string(registry_path),
        "queue_path": _path_string(queue_path),
        "experiments": [
            {
                "experiment_id": spec.experiment_id,
                "lake_count": len(manifest["lakes"]),
                "epochs": spec.epochs,
                "full_eval_every_epochs": spec.full_eval_every_epochs,
                "status": spec.status,
                "heldout_lake_ids": manifest.get("heldout_lake_ids", []),
            }
            for spec, manifest, _ in specs_and_manifests
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Lake-PINN RECON L1 tiered smoke manifests and queues.")
    parser.add_argument("--v9-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--standard-input-root", type=Path, default=DEFAULT_STANDARD_INPUT_ROOT)
    parser.add_argument("--readiness-csv", type=Path, default=None)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare_recon_tiered_smokes(
        v9_root=args.v9_root,
        pipeline_root=args.pipeline_root,
        standard_input_root=args.standard_input_root,
        readiness_csv=args.readiness_csv,
        base_manifest=args.base_manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
