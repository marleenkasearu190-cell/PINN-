from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[1]
if CANONICAL_REPO_ROOT.exists():
    REPO_ROOT = CANONICAL_REPO_ROOT
DEFAULT_READINESS_FILENAME = "_local_standard_inputs_training_readiness_20260606.csv"
DEFAULT_STANDARD_INPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "_standard_inputs"
DEFAULT_HELDOUT_IDS = ("lacawac_2016", "carvins_cove_2022", "lake_maggiore_2024")
DEFAULT_SEEDS = (1, 2, 3)
CHECKPOINT_SELECTION = "best_by_val_rolling"
PRIMARY_TRANSFER_METRIC = "0.5 * val_fewshot_rmse_30d + 0.5 * val_fewshot_rmse_60d"
GO_NO_GO_CRITERIA = (
    "smoke reaches full_eval_every_epochs without runtime failure",
    "global_state_forecaster_training_history.csv is written",
    "best_by_val_rolling.pt and best_by_val_rolling_metrics.json are written",
    "history includes val_rolling_start_rmse_30d/60d and val_fewshot_rmse_30d/60d",
    "heldout lakes are diagnostic only and are not used for checkpoint selection",
    "R10/R11 target median point RMSE is below 2.2-2.5 C on transfer-valid splits",
    "R10/R11 target median rolling 30d RMSE is below 2.7-3.0 C",
    "R10/R11 target median rolling 60d RMSE is below 3.2-3.5 C",
    "Kz/Kd/residual diagnostics show no physical bypass",
    "split has no lake-group leakage",
)

REGISTRY_FIELDS = (
    "experiment_id",
    "level",
    "hypothesis",
    "code_hash",
    "data_version",
    "manifest",
    "split_file",
    "split_mode",
    "train_lakes",
    "val_lakes",
    "test_lakes",
    "train_lake_groups",
    "val_lake_groups",
    "test_lake_groups",
    "heldout_lake_groups",
    "support_query_policy",
    "model_variant",
    "kz_variant",
    "kd_variant",
    "lst_variant",
    "segment_variant",
    "adapter_variant",
    "process_pretrain",
    "primary_metric",
    "secondary_physics_metrics",
    "seed",
    "checkpoint_selection",
    "status",
    "main_result_json",
    "result_path",
    "notes",
)


@dataclass(frozen=True)
class ExperimentAsset:
    experiment_id: str
    manifest_path: Path
    output_dir: Path
    epochs: int
    seed: int | None
    stage: str


def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _csv_ready(row: dict) -> bool:
    if "prepare_ok" in row:
        return _csv_bool(row.get("prepare_ok"))
    return _csv_bool(row.get("usable_basic"))


def _row_lake_group(row: dict, lake_id: str) -> str:
    return str(row.get("metadata_lake_group") or row.get("lake_group") or lake_id).strip()


def _path_string(path: Path) -> str:
    return path.resolve().as_posix()


def resolve_readiness_csv(standard_input_root: Path, readiness_csv: Path | None = None) -> Path:
    if readiness_csv is not None:
        path = readiness_csv
    else:
        matches = sorted(
            standard_input_root.glob("*training_readiness*.csv"),
            key=lambda item: item.stat().st_mtime,
        )
        path = matches[-1] if matches else standard_input_root / DEFAULT_READINESS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"training readiness CSV not found: {path}")
    return path


def data_version_from_readiness(readiness_csv: Path) -> str:
    stem = readiness_csv.stem
    for part in stem.split("_"):
        if len(part) == 8 and part.isdigit():
            return f"standard_inputs_{part}"
    return stem


def code_hash(root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for folder_name in ("lake_pinn", "scripts"):
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return f"lake_pinn_sha256:{digest.hexdigest()[:12]}"


def git_short_hash(root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _required_lake_paths(standard_input_root: Path, lake_id: str, *, lst_kind: str) -> dict[str, Path]:
    lst_name = "lst_night_for_model.csv" if lst_kind == "night" else "lst_day_for_model.csv"
    lake_root = standard_input_root / lake_id
    return {
        "era5": lake_root / "era5_for_model.csv",
        "lst": lake_root / lst_name,
        "profile_obs": lake_root / "profile_for_model.csv",
        "metadata": lake_root / "metadata.json",
    }


def load_usable_lakes(
    standard_input_root: Path,
    readiness_csv: Path,
    *,
    lst_kind: str = "night",
) -> tuple[list[dict], list[dict]]:
    usable_lakes: list[dict] = []
    skipped_rows: list[dict] = []
    with readiness_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            lake_id = str(row.get("lake_id", "")).strip()
            if not lake_id:
                continue
            if not _csv_ready(row):
                skipped_rows.append(
                    {
                        "lake_id": lake_id,
                        "reason": (
                            row.get("blockers")
                            or row.get("warnings")
                            or row.get("error")
                            or "prepare_ok/usable_basic is false"
                        ),
                    }
                )
                continue
            paths = _required_lake_paths(standard_input_root, lake_id, lst_kind=lst_kind)
            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                skipped_rows.append(
                    {
                        "lake_id": lake_id,
                        "reason": f"missing required files: {', '.join(missing)}",
                    }
                )
                continue
            max_depth = (
                _float_or_none(row.get("metadata_max_depth_m"))
                or _float_or_none(row.get("profile_max_depth_m"))
                or 1.0
            )
            year = _int_or_none(row.get("year"))
            lake_config = {
                "lake_id": lake_id,
                "lake_group": _row_lake_group(row, lake_id),
                "year": year,
                "max_depth": max_depth,
                "era5": _path_string(paths["era5"]),
                "lst": _path_string(paths["lst"]),
                "profile_obs": _path_string(paths["profile_obs"]),
                "metadata": _path_string(paths["metadata"]),
            }
            usable_lakes.append(lake_config)
    usable_lakes.sort(key=lambda item: item["lake_id"])
    return usable_lakes, skipped_rows


def _heldout_groups(lakes: list[dict], heldout_ids: tuple[str, ...] | list[str]) -> list[str]:
    group_by_id = {lake["lake_id"]: lake["lake_group"] for lake in lakes}
    missing = [lake_id for lake_id in heldout_ids if lake_id not in group_by_id]
    if missing:
        raise ValueError(f"heldout lake IDs not present in usable lakes: {', '.join(missing)}")
    return _unique([group_by_id[lake_id] for lake_id in heldout_ids])


def _train_lake_ids(lakes: list[dict], heldout_groups: list[str]) -> list[str]:
    heldout_group_set = set(heldout_groups)
    return [lake["lake_id"] for lake in lakes if lake["lake_group"] not in heldout_group_set]


def _base_manifest(
    *,
    experiment_id: str,
    lakes: list[dict],
    heldout_ids: tuple[str, ...] | list[str],
    stage: str,
    seed: int | None,
    data_version: str,
    code_version: str,
) -> dict:
    heldout_groups = _heldout_groups(lakes, heldout_ids)
    common = {
        "experiment_id": experiment_id,
        "experiment": experiment_id,
        "level": "L1_smoke" if stage == "smoke" else "R10_formal_short",
        "hypothesis": (
            "Clean-physics few-shot RECON smoke can reach validation rolling/few-shot eval "
            "without using locked heldout lakes for checkpoint selection."
            if stage == "smoke"
            else "Clean-physics few-shot RECON formal seed can improve transfer diagnostics "
            "after the smoke go/no-go gate passes."
        ),
        "experiment_stage": stage,
        "code_hash": code_version,
        "data_version": data_version,
        "split_file": "inline:R10_heldout_lake_groups",
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "primary_transfer_metric": PRIMARY_TRANSFER_METRIC,
        "secondary_physics_metrics": [
            "val_rolling_start_rmse_30d",
            "val_rolling_start_rmse_60d",
            "heldout_free_roll_mean_rmse",
            "residual_abs_mean_c",
            "kd_prior_regularization_loss_mean",
            "stratification_mixing_gate_deep_mean",
        ],
        "go_no_go_criteria": list(GO_NO_GO_CRITERIA),
        "split_mode": "time_blocked",
        "split_contract": "R10 heldout lake groups are diagnostic only; checkpoint uses validation rolling/few-shot metrics.",
        "profile_supervision_scope": "train",
        "depth_points": 30,
        "history_window_days": 30,
        "max_rollout_days": 60,
        "heldout_lake_ids": list(heldout_ids),
        "test_lake_ids": list(heldout_ids),
        "heldout_lake_groups": heldout_groups,
        "test_lake_groups": heldout_groups,
        "support_query_policy": "episodic support profiles precede query windows; locked heldout is final diagnostic only",
        "residual_limit_c": 0.15,
        "residual_regularization_weight": 0.01,
        "daily_tendency_weight": 0.0,
        "physical_scale_regularization_weight": 0.001,
        "physical_scale_smoothness_weight": 0.0,
        "kd_prior_regularization_weight": 0.001,
        "adaptive_parameter_regularization_weight": 0.005,
        "heat_content_transition_weight": 0.0,
        "warm_season_column_heat_content_weight": 0.0,
        "wind_kz_scale": 1.5,
        "shape_aware_mixing": "on",
        "stratification_mixing_cap": "on",
        "turbulent_flux_mode": "bulk",
        "freezing_energy_mode": "latent_reservoir",
        "advective_heat_source_mode": "off",
        "lake_adaptive_params": "off",
        "lake_adaptive_temporal_mode": "off",
        "segment_rollout_loss_weight": 0.05,
        "segment_rollout_lst_surface_weight": 0.003,
        "segment_rollout_max_days": 60,
        "lst_feature_dropout_probability": 0.40,
        "teacher_forcing_start": 0.5,
        "teacher_forcing_end": 0.0,
        "state_noise_weight": 1.0,
        "residual_time_smooth_weight": 0.01,
        "episodic_fewshot_mode": "on",
        "episodic_fewshot_loss_weight": 0.10,
        "episodic_fewshot_max_query_days": 120,
        "episodic_fewshot_support_profile_count": 3,
        "fewshot_adapter_params": "kz,kd,exchange,convective,ice",
        "cross_lake_batch_mode": "on",
        "transition_batch_size": 0,
        "segment_rollout_batch_size": 0,
        "rolling_horizon_batch_size": 32,
        "torch_tf32": "on",
        "torch_matmul_precision": "high",
        "profile_runtime": True,
        "profile_gpu": False,
        "train_diagnostic_mode": "loss",
        "export_after_training": "off",
        "export_max_depth_m": 25.0,
        "lakes": lakes,
    }
    if seed is not None:
        common["seed"] = int(seed)
    return common


def build_smoke_manifest(
    *,
    lakes: list[dict],
    heldout_ids: tuple[str, ...] | list[str],
    data_version: str,
    code_version: str,
) -> tuple[dict, int]:
    manifest = _base_manifest(
        experiment_id="R10_CLEANPHYS_FEWSHOT_smoke",
        lakes=lakes,
        heldout_ids=heldout_ids,
        stage="smoke",
        seed=1,
        data_version=data_version,
        code_version=code_version,
    )
    manifest.update(
        {
            "segment_rollout_start_epoch": 2,
            "segment_rollout_ramp_epochs": 8,
            "segment_rollout_samples_per_lake": 4,
            "episodic_fewshot_start_epoch": 2,
            "episodic_fewshot_ramp_epochs": 8,
            "episodic_fewshot_samples_per_lake": 4,
            "rolling_horizon_eval_max_starts": 10,
            "checkpoint_every_epochs": 5,
            "eval_every_epochs": 5,
            "full_eval_every_epochs": 10,
        }
    )
    return manifest, 20


def build_formal_manifest(
    *,
    seed: int,
    lakes: list[dict],
    heldout_ids: tuple[str, ...] | list[str],
    data_version: str,
    code_version: str,
) -> tuple[dict, int]:
    experiment_id = f"R10_CLEANPHYS_FEWSHOT_seed{int(seed)}"
    manifest = _base_manifest(
        experiment_id=experiment_id,
        lakes=lakes,
        heldout_ids=heldout_ids,
        stage="formal_short",
        seed=int(seed),
        data_version=data_version,
        code_version=code_version,
    )
    manifest.update(
        {
            "segment_rollout_start_epoch": 20,
            "segment_rollout_ramp_epochs": 40,
            "segment_rollout_samples_per_lake": 12,
            "episodic_fewshot_start_epoch": 20,
            "episodic_fewshot_ramp_epochs": 40,
            "episodic_fewshot_samples_per_lake": 4,
            "rolling_horizon_eval_max_starts": 20,
            "checkpoint_every_epochs": 20,
            "eval_every_epochs": 20,
            "full_eval_every_epochs": 20,
        }
    )
    return manifest, 60


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _registry_row(
    *,
    asset: ExperimentAsset,
    manifest: dict,
    manifest_path: Path,
    code_version: str,
    data_version: str,
    output_root: Path,
) -> dict[str, str]:
    heldout_groups = manifest["heldout_lake_groups"]
    train_lakes = _train_lake_ids(manifest["lakes"], heldout_groups)
    train_lake_set = set(train_lakes)
    train_groups = _unique(
        [lake["lake_group"] for lake in manifest["lakes"] if lake["lake_id"] in train_lake_set]
    )
    main_result = asset.output_dir / "global_state_forecaster_split_summary.json"
    status = "queued" if asset.stage == "smoke" else "needs_approval"
    return {
        "experiment_id": asset.experiment_id,
        "level": str(manifest.get("level", "")),
        "hypothesis": str(manifest.get("hypothesis", "")),
        "code_hash": code_version,
        "data_version": data_version,
        "manifest": _path_string(manifest_path),
        "split_file": str(manifest.get("split_file", "")),
        "split_mode": manifest["split_mode"],
        "train_lakes": ",".join(train_lakes),
        "val_lakes": ",".join(train_lakes),
        "test_lakes": ",".join(manifest["test_lake_ids"]),
        "train_lake_groups": ",".join(train_groups),
        "val_lake_groups": ",".join(train_groups),
        "test_lake_groups": ",".join(heldout_groups),
        "heldout_lake_groups": ",".join(heldout_groups),
        "support_query_policy": str(manifest.get("support_query_policy", "")),
        "model_variant": "cleanphys_fewshot",
        "kz_variant": "background_plus_gated_turbulent",
        "kd_variant": "optical_prior_regularized",
        "lst_variant": f"night_quality_dropout_{manifest['lst_feature_dropout_probability']:.2f}",
        "segment_variant": "horizon_weighted_roll60",
        "adapter_variant": "fewshot_support3_metadata_off",
        "process_pretrain": "none",
        "primary_metric": PRIMARY_TRANSFER_METRIC,
        "secondary_physics_metrics": ";".join(manifest.get("secondary_physics_metrics", [])),
        "seed": "" if asset.seed is None else str(asset.seed),
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "status": status,
        "main_result_json": _path_string(main_result),
        "result_path": _path_string(asset.output_dir),
        "notes": (
            f"{asset.stage}; epochs={asset.epochs}; output_root={_path_string(output_root)}; "
            f"heldout_groups={','.join(heldout_groups)}"
        ),
    }


def _read_existing_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})


def upsert_registry(path: Path, new_rows: list[dict[str, str]]) -> None:
    existing = _read_existing_registry(path)
    by_id = {row.get("experiment_id", ""): row for row in existing if row.get("experiment_id")}
    for row in new_rows:
        old_row = by_id.get(row["experiment_id"], {})
        merged = {**old_row, **row}
        for field in ("status", "notes"):
            if old_row.get(field):
                merged[field] = old_row[field]
        by_id[row["experiment_id"]] = merged
    ordered_ids = [row.get("experiment_id", "") for row in existing if row.get("experiment_id")]
    for row in new_rows:
        if row["experiment_id"] not in ordered_ids:
            ordered_ids.append(row["experiment_id"])
    _write_registry(path, [by_id[item] for item in ordered_ids if item in by_id])


def _command_for(asset: ExperimentAsset) -> str:
    return (
        "python -m lake_pinn "
        f'--manifest "{asset.manifest_path}" '
        f'--output-dir "{asset.output_dir}" '
        f"--epochs {asset.epochs} "
        f"--seed {asset.seed if asset.seed is not None else 0} "
        "--device cuda"
    )


def _write_launch_commands(path: Path, assets: list[ExperimentAsset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by scripts/prepare_r10_experiments.py",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for asset in assets:
        lines.append(f"# {asset.experiment_id}")
        lines.append(f"New-Item -ItemType Directory -Force -Path \"{asset.output_dir}\" | Out-Null")
        lines.append(_command_for(asset))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_experiment_assets(
    *,
    standard_input_root: Path = DEFAULT_STANDARD_INPUT_ROOT,
    readiness_csv: Path | None = None,
    output_root: Path | None = None,
    heldout_ids: tuple[str, ...] | list[str] = DEFAULT_HELDOUT_IDS,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
    lst_kind: str = "night",
    write_launch: bool = True,
) -> dict:
    standard_input_root = Path(standard_input_root)
    output_root = Path(output_root) if output_root is not None else REPO_ROOT / "experiments"
    readiness_path = resolve_readiness_csv(standard_input_root, Path(readiness_csv) if readiness_csv else None)
    data_version = data_version_from_readiness(readiness_path)
    lakes, skipped_rows = load_usable_lakes(
        standard_input_root,
        readiness_path,
        lst_kind=lst_kind,
    )
    heldout_ids = tuple(heldout_ids)
    heldout_groups = _heldout_groups(lakes, heldout_ids)
    manifests_dir = output_root / "manifests"
    assets: list[ExperimentAsset] = []
    registry_rows: list[dict[str, str]] = []
    code_version = code_hash(REPO_ROOT)
    git_hash = git_short_hash(REPO_ROOT)
    if git_hash:
        code_version = f"{code_version};git:{git_hash}"

    smoke_manifest, smoke_epochs = build_smoke_manifest(
        lakes=lakes,
        heldout_ids=heldout_ids,
        data_version=data_version,
        code_version=code_version,
    )
    smoke_path = manifests_dir / f"{smoke_manifest['experiment']}.json"
    _write_json(smoke_path, smoke_manifest)
    smoke_asset = ExperimentAsset(
        experiment_id=smoke_manifest["experiment"],
        manifest_path=smoke_path,
        output_dir=output_root / smoke_manifest["experiment"],
        epochs=smoke_epochs,
        seed=smoke_manifest.get("seed"),
        stage="smoke",
    )
    assets.append(smoke_asset)
    registry_rows.append(
        _registry_row(
            asset=smoke_asset,
            manifest=smoke_manifest,
            manifest_path=smoke_path,
            code_version=code_version,
            data_version=data_version,
            output_root=output_root,
        )
    )

    for seed in seeds:
        formal_manifest, formal_epochs = build_formal_manifest(
            seed=int(seed),
            lakes=lakes,
            heldout_ids=heldout_ids,
            data_version=data_version,
            code_version=code_version,
        )
        formal_path = manifests_dir / f"{formal_manifest['experiment']}.json"
        _write_json(formal_path, formal_manifest)
        formal_asset = ExperimentAsset(
            experiment_id=formal_manifest["experiment"],
            manifest_path=formal_path,
            output_dir=output_root / formal_manifest["experiment"],
            epochs=formal_epochs,
            seed=int(seed),
            stage="formal_short",
        )
        assets.append(formal_asset)
        registry_rows.append(
            _registry_row(
                asset=formal_asset,
                manifest=formal_manifest,
                manifest_path=formal_path,
                code_version=code_version,
                data_version=data_version,
                output_root=output_root,
            )
        )

    registry_path = output_root / "registry.csv"
    upsert_registry(registry_path, registry_rows)
    launch_path = output_root / "launch_R10_CLEANPHYS_FEWSHOT.ps1"
    if write_launch:
        _write_launch_commands(launch_path, assets)

    return {
        "standard_input_root": _path_string(standard_input_root),
        "readiness_csv": _path_string(readiness_path),
        "data_version": data_version,
        "code_hash": code_version,
        "usable_lake_count": len(lakes),
        "skipped_lakes": skipped_rows,
        "heldout_lake_ids": list(heldout_ids),
        "heldout_lake_groups": heldout_groups,
        "manifest_paths": [_path_string(asset.manifest_path) for asset in assets],
        "registry_path": _path_string(registry_path),
        "launch_commands_path": _path_string(launch_path) if write_launch else None,
    }


def _parse_csv_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare LakePINN v9 R10 experiment manifests and registry.")
    parser.add_argument("--standard-input-root", type=Path, default=DEFAULT_STANDARD_INPUT_ROOT)
    parser.add_argument("--readiness-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--heldout-ids", default=",".join(DEFAULT_HELDOUT_IDS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--lst-kind", choices=["night", "day"], default="night")
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args(argv)

    heldout_ids = _parse_csv_list(args.heldout_ids) or DEFAULT_HELDOUT_IDS
    seeds = tuple(int(item) for item in _parse_csv_list(args.seeds)) or DEFAULT_SEEDS
    summary = build_experiment_assets(
        standard_input_root=args.standard_input_root,
        readiness_csv=args.readiness_csv,
        output_root=args.output_root,
        heldout_ids=heldout_ids,
        seeds=seeds,
        lst_kind=args.lst_kind,
        write_launch=not args.no_launch,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
