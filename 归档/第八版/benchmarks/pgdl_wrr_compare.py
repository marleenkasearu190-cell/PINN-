from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = REPO_ROOT / "external" / "pgdl_wrr_2019"
OUTPUT_DIR = REPO_ROOT / "experiments" / "PGDL_WRR_COMPARE_Mendota_20260528"
LAKEPINN_INPUT_DIR = EXTERNAL_DIR / "lakepinn_inputs"

ZENODO_RECORD_API = "https://zenodo.org/api/records/3497495"
PGDL_SOFTWARE_CITATION = "USGS-CIDA/ms-pgdl-wrr v1.0.0, Zenodo record 3497495"
PGDL_DATA_CITATION = "USGS data release DOI 10.5066/P9AQPIVD"

SCIENCEBASE_ITEMS = {
    "inputs": {
        "item_id": "5d98e0c4e4b0c4f70d1186f1",
        "title": "Process-guided deep learning water temperature predictions: 3a Lake Mendota inputs",
        "files": ["mendota_meteo.csv"],
    },
    "training": {
        "item_id": "5d8a837fe4b0c4f70d0ae8ac",
        "title": "Process-guided deep learning water temperature predictions: 4a Lake Mendota detailed training data",
        "files": ["me_year_training.csv"],
    },
    "prediction": {
        "item_id": "5d915cb2e4b0c4f70d0ce523",
        "title": "Process-guided deep learning water temperature predictions: 5a Lake Mendota detailed prediction data",
        "files": [
            "me_year_predict_pb.csv",
            "me_year_predict_dl.csv",
            "me_year_predict_pgdl.csv",
        ],
    },
    "evaluation": {
        "item_id": "5d925066e4b0c4f70d0d0599",
        "title": "Process-guided deep learning water temperature predictions: 6a Lake Mendota detailed evaluation data",
        "files": ["me_test.csv", "me_RMSE.csv"],
    },
    "config": {
        "item_id": "5d8a2257e4b0c4f70d0ae513",
        "title": "Process-guided deep learning water temperature predictions: 2 Model configurations",
        "files": ["glm_config.json"],
    },
}

PRIMARY_EXPER_ID = "year_500"
PRIMARY_EXPER_N = 1
METADATA = {
    "lake_id": "mendota_pgdl",
    "name": "Lake Mendota",
    "source": PGDL_DATA_CITATION,
    "latitude": 43.09885,
    "longitude": -89.40573,
    "area_km2": 39.4,
    "max_depth_m": 24.5,
    "mean_depth_m": 12.8,
    "volume_km3": 0.505,
    "secchi_m": 2.0,
    "light_extinction_kd": 0.85,
    "effective_fetch_m": 9000.0,
    "thermal_regime": "cold_ice_prone",
}


def ensure_dirs() -> None:
    for path in (EXTERNAL_DIR, OUTPUT_DIR, LAKEPINN_INPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_url(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "LakePINN benchmark downloader"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def download_file(url: str, output_path: Path, *, retries: int = 4, overwrite: bool = False) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
        return {
            "path": str(output_path),
            "downloaded": False,
            "bytes": int(output_path.stat().st_size),
            "sha256": sha256_file(output_path),
        }

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LakePINN benchmark downloader"})
            with urllib.request.urlopen(req, timeout=180) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(output_path)
            return {
                "path": str(output_path),
                "downloaded": True,
                "bytes": int(output_path.stat().st_size),
                "sha256": sha256_file(output_path),
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def sciencebase_file_index(item_id: str) -> dict[str, dict]:
    payload = read_json_url(f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json")
    files = {}
    for file_info in payload.get("files", []):
        name = file_info.get("name")
        if name:
            files[name] = file_info
    return files


def download_pgdl_assets(*, overwrite: bool = False) -> dict:
    ensure_dirs()
    manifest: dict[str, object] = {
        "created_at_unix": time.time(),
        "repo_root": str(REPO_ROOT),
        "external_dir": str(EXTERNAL_DIR),
        "pgdl_software_citation": PGDL_SOFTWARE_CITATION,
        "pgdl_data_citation": PGDL_DATA_CITATION,
        "zenodo_record_api": ZENODO_RECORD_API,
        "sciencebase_items": {},
        "files": {},
    }

    zenodo = read_json_url(ZENODO_RECORD_API)
    software_files = zenodo.get("files", [])
    if not software_files:
        raise RuntimeError("Zenodo record has no downloadable files.")
    software_file = software_files[0]
    software_name = software_file["key"].split("/")[-1]
    software_url = software_file["links"]["self"]
    software_path = EXTERNAL_DIR / "software" / software_name
    manifest["files"][software_name] = download_file(software_url, software_path, overwrite=overwrite)
    if software_path.suffix.lower() == ".zip":
        extract_dir = EXTERNAL_DIR / "software" / software_path.stem
        if overwrite or not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(software_path, "r") as archive:
                archive.extractall(extract_dir)
        manifest["files"][software_name]["extracted_to"] = str(extract_dir)

    for group, spec in SCIENCEBASE_ITEMS.items():
        item_id = spec["item_id"]
        file_index = sciencebase_file_index(item_id)
        manifest["sciencebase_items"][group] = {
            "item_id": item_id,
            "title": spec["title"],
            "requested_files": list(spec["files"]),
        }
        for name in spec["files"]:
            if name not in file_index:
                raise RuntimeError(f"ScienceBase item {item_id} did not contain {name}")
            file_info = file_index[name]
            url = file_info.get("downloadUri") or file_info.get("url")
            if not url:
                raise RuntimeError(f"ScienceBase file {name} has no download URL")
            output_path = EXTERNAL_DIR / "data" / group / name
            record = download_file(url, output_path, overwrite=overwrite)
            record.update(
                {
                    "sciencebase_item_id": item_id,
                    "sciencebase_title": spec["title"],
                    "source_url": url,
                    "reported_bytes": file_info.get("size"),
                }
            )
            manifest["files"][name] = record

    manifest_path = OUTPUT_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def data_path(group: str, name: str) -> Path:
    return EXTERNAL_DIR / "data" / group / name


def prediction_to_long(path: Path, model_name: str, exper_id: str | None = None, exper_n: int | None = None) -> pd.DataFrame:
    wide = pd.read_csv(path)
    if exper_id is not None and "exper_id" in wide.columns:
        wide = wide[wide["exper_id"].astype(str) == exper_id].copy()
    if exper_n is not None and "exper_n" in wide.columns:
        wide = wide[pd.to_numeric(wide["exper_n"], errors="coerce") == int(exper_n)].copy()
    temp_cols = [c for c in wide.columns if c.startswith("temp_")]
    long = wide.melt(
        id_vars=[c for c in ["date", "exper_n", "exper_id"] if c in wide.columns],
        value_vars=temp_cols,
        var_name="depth_label",
        value_name=f"{model_name}_C",
    )
    long["Date"] = pd.to_datetime(long["date"])
    long["Depth_m"] = long["depth_label"].str.replace("temp_", "", regex=False).astype(float)
    keep = ["Date", "Depth_m", f"{model_name}_C"]
    if "exper_n" in long.columns:
        long["exper_n"] = pd.to_numeric(long["exper_n"], errors="coerce").astype("Int64")
        keep.append("exper_n")
    if "exper_id" in long.columns:
        keep.append("exper_id")
    return long[keep].dropna(subset=[f"{model_name}_C"])


def observed_to_long(path: Path, exper_type: str = "year", exper_n: int | None = None) -> pd.DataFrame:
    obs = pd.read_csv(path)
    if "exper_type" in obs.columns:
        obs = obs[obs["exper_type"].astype(str) == exper_type].copy()
    if exper_n is not None and "exper_n" in obs.columns:
        obs = obs[pd.to_numeric(obs["exper_n"], errors="coerce") == int(exper_n)].copy()
    obs = obs.rename(columns={"date": "Date", "depth": "Depth_m", "temp": "Observed_C"})
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["Depth_m"] = pd.to_numeric(obs["Depth_m"], errors="coerce")
    obs["Observed_C"] = pd.to_numeric(obs["Observed_C"], errors="coerce")
    keep = ["Date", "Depth_m", "Observed_C"]
    if "exper_n" in obs.columns:
        obs["exper_n"] = pd.to_numeric(obs["exper_n"], errors="coerce").astype("Int64")
        keep.append("exper_n")
    return obs[keep].dropna(subset=["Date", "Depth_m", "Observed_C"])


def metric_record(frame: pd.DataFrame, model_col: str, model_name: str, group_name: str = "overall") -> dict:
    diff = pd.to_numeric(frame[model_col], errors="coerce") - pd.to_numeric(frame["Observed_C"], errors="coerce")
    finite = np.isfinite(diff.to_numpy(dtype=float))
    if not np.any(finite):
        return {
            "model": model_name,
            "group": group_name,
            "matched_points": 0,
            "matched_dates": 0,
            "rmse_C": np.nan,
            "mae_C": np.nan,
            "bias_C": np.nan,
            "p95_abs_error_C": np.nan,
        }
    diff = diff[finite]
    subset = frame.loc[diff.index]
    return {
        "model": model_name,
        "group": group_name,
        "matched_points": int(len(diff)),
        "matched_dates": int(pd.to_datetime(subset["Date"]).dt.normalize().nunique()),
        "rmse_C": float(np.sqrt(np.mean(np.square(diff)))),
        "mae_C": float(np.mean(np.abs(diff))),
        "bias_C": float(np.mean(diff)),
        "p95_abs_error_C": float(np.percentile(np.abs(diff), 95)),
    }


def compute_grouped_metrics(matched: pd.DataFrame, model_cols: dict[str, str]) -> pd.DataFrame:
    records = []
    groups: list[tuple[str, pd.Series]] = [
        ("overall", pd.Series(True, index=matched.index)),
        ("surface_0_3m", matched["Depth_m"] <= 3.0),
        ("mid_3_12m", (matched["Depth_m"] > 3.0) & (matched["Depth_m"] <= 12.0)),
        ("deep_gt12m", matched["Depth_m"] > 12.0),
    ]
    for month in sorted(pd.to_datetime(matched["Date"]).dt.month.dropna().unique()):
        groups.append((f"month_{int(month):02d}", pd.to_datetime(matched["Date"]).dt.month == month))
    for model_name, column in model_cols.items():
        for group_name, mask in groups:
            subset = matched[mask & matched[column].notna()].copy()
            records.append(metric_record(subset, column, model_name, group_name))
    return pd.DataFrame.from_records(records)


def build_official_pgdl_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rmse = pd.read_csv(data_path("evaluation", "me_RMSE.csv"))
    rmse.to_csv(OUTPUT_DIR / "pgdl_official_metrics.csv", index=False)

    observed = observed_to_long(
        data_path("evaluation", "me_test.csv"),
        exper_type="year",
        exper_n=PRIMARY_EXPER_N,
    )
    matched = observed.copy()
    for model_name in ("pb", "dl", "pgdl"):
        pred = prediction_to_long(
            data_path("prediction", f"me_year_predict_{model_name}.csv"),
            model_name=model_name.upper(),
            exper_id=PRIMARY_EXPER_ID,
            exper_n=PRIMARY_EXPER_N,
        )
        matched = matched.merge(pred[["Date", "Depth_m", f"{model_name.upper()}_C"]], on=["Date", "Depth_m"], how="left")

    matched.to_csv(OUTPUT_DIR / "matched_point_predictions.csv", index=False)
    metrics = compute_grouped_metrics(
        matched,
        {
            "PB_official": "PB_C",
            "DL_official": "DL_C",
            "PGDL_official": "PGDL_C",
        },
    )
    metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)
    return rmse, matched, metrics


def convert_meteo_to_lakepinn_inputs() -> tuple[Path, Path, Path, Path, Path]:
    input_root = LAKEPINN_INPUT_DIR / "mendota_pgdl"
    train_dir = input_root / "mendota_pgdl_train"
    test_dir = input_root / "mendota_pgdl_test"
    for directory in (train_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    meteo = pd.read_csv(data_path("inputs", "mendota_meteo.csv"))
    meteo["Date"] = pd.to_datetime(meteo["date"])
    era5 = pd.DataFrame(
        {
            "Date": meteo["Date"],
            "T_air_C": pd.to_numeric(meteo["AirTemp"], errors="coerce"),
            "t2m_C": pd.to_numeric(meteo["AirTemp"], errors="coerce"),
            "wind_speed_m_per_s": pd.to_numeric(meteo["WindSpeed"], errors="coerce"),
            "wind_norm_m_per_s": pd.to_numeric(meteo["WindSpeed"], errors="coerce"),
            "Solar_W_m2": pd.to_numeric(meteo["ShortWave"], errors="coerce"),
            "ssrd_W_per_m2": pd.to_numeric(meteo["ShortWave"], errors="coerce"),
            "Longwave_W_m2": pd.to_numeric(meteo["LongWave"], errors="coerce"),
            "relative_humidity": pd.to_numeric(meteo["RelHum"], errors="coerce") / 100.0,
            "rh_percent": pd.to_numeric(meteo["RelHum"], errors="coerce"),
            "surface_pressure_Pa": 101325.0,
            "latent_heat_upward_W_m2": 0.0,
            "sensible_heat_upward_W_m2": 0.0,
            "Secchi_m": METADATA["secchi_m"],
            "light_extinction_kd": METADATA["light_extinction_kd"],
            "effective_fetch": METADATA["effective_fetch_m"],
            "net_inflow": 0.0,
        }
    )
    lst = pd.DataFrame({"Date": era5["Date"], "LST_surface_C": np.nan, "LST_quality_factor": 0.0})

    train_obs = pd.read_csv(data_path("training", "me_year_training.csv"))
    train_obs = train_obs[train_obs["exper_id"].astype(str) == PRIMARY_EXPER_ID].copy()
    train_obs["Date"] = pd.to_datetime(train_obs["date"])
    train_profile = train_obs.rename(columns={"depth": "Depth_m", "temp": "Temperature_C"})[
        ["Date", "Depth_m", "Temperature_C"]
    ]

    test_profile = observed_to_long(
        data_path("evaluation", "me_test.csv"),
        exper_type="year",
        exper_n=PRIMARY_EXPER_N,
    ).rename(columns={"Observed_C": "Temperature_C"})[["Date", "Depth_m", "Temperature_C"]]

    metadata_train = dict(METADATA, lake_id="mendota_pgdl_train", benchmark_role="pgdl_train")
    metadata_test = dict(METADATA, lake_id="mendota_pgdl_test", benchmark_role="pgdl_test")
    for directory, profile, metadata in (
        (train_dir, train_profile, metadata_train),
        (test_dir, test_profile, metadata_test),
    ):
        era5.to_csv(directory / "era5_for_model.csv", index=False)
        lst.to_csv(directory / "lst_no_observation_for_model.csv", index=False)
        profile.to_csv(directory / "profile_for_model.csv", index=False)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest = {
        "task_mode": "analysis",
        "data_fill_mode": "reconstruction",
        "split_mode": "none",
        "test_lake_id": "mendota_pgdl_test",
        "export_output_set": "core",
        "max_rollout_days": 45,
        "depth_points": 50,
        "history_window_days": 30,
        "lst_surface_weight": 0.0,
        "lakes": [
            {
                "lake_id": "mendota_pgdl_train",
                "lake_group": "mendota_pgdl",
                "era5": str(train_dir / "era5_for_model.csv"),
                "lst": str(train_dir / "lst_no_observation_for_model.csv"),
                "profile": str(train_dir / "profile_for_model.csv"),
                "metadata": str(train_dir / "metadata.json"),
                "max_depth": METADATA["max_depth_m"],
            },
            {
                "lake_id": "mendota_pgdl_test",
                "lake_group": "mendota_pgdl",
                "era5": str(test_dir / "era5_for_model.csv"),
                "lst": str(test_dir / "lst_no_observation_for_model.csv"),
                "profile": str(test_dir / "profile_for_model.csv"),
                "metadata": str(test_dir / "metadata.json"),
                "max_depth": METADATA["max_depth_m"],
            },
        ],
    }
    manifest_path = OUTPUT_DIR / "lakepinn_pgdl_mendota_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        train_dir / "era5_for_model.csv",
        train_dir / "lst_no_observation_for_model.csv",
        train_dir / "profile_for_model.csv",
        test_dir / "profile_for_model.csv",
        manifest_path,
    )


def run_lakepinn(
    manifest_path: Path,
    *,
    epochs: int,
    output_name: str,
    smoke: bool = False,
) -> dict:
    output_dir = OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    api_code = f"""
from pathlib import Path
from lake_pinn.state_multilake import train_multilake_state_forecaster

train_multilake_state_forecaster(
    manifest_path={str(manifest_path)!r},
    output_dir={str(output_dir)!r},
    epochs={int(epochs)!r},
    test_lake_id='mendota_pgdl_test',
    lst_surface_weight=0.0,
    depth_points=50,
    max_rollout_days=45,
    history_window_days=30,
    transition_loss_weight=0.5,
    segment_rollout_loss_weight={0.0 if smoke else 0.05},
    eval_every_epochs={max(1, int(epochs))!r},
    rolling_horizon_eval_max_starts={10 if smoke else 40},
    transition_batch_mode='on',
    transition_batch_size=0,
    rolling_horizon_batch_mode='on',
    rolling_horizon_batch_size=32,
    step_forcing_mode='tensor',
    diagnostic_mode='auto',
    export_output_set='core',
    device='cpu',
)
"""
    cmd = [sys.executable, "-c", api_code]
    env = None
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60 * 60 * 8,
        env=env,
    )
    elapsed = time.time() - start
    (output_dir / "lakepinn_stdout.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
    (output_dir / "lakepinn_stderr.log").write_text(proc.stderr, encoding="utf-8", errors="ignore")
    return {
        "command": cmd,
        "output_dir": str(output_dir),
        "returncode": int(proc.returncode),
        "seconds": elapsed,
        "stdout_log": str(output_dir / "lakepinn_stdout.log"),
        "stderr_log": str(output_dir / "lakepinn_stderr.log"),
    }


def find_lakepinn_prediction_csv(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*heldout_state_reconstruction_temperature_depth_predictions.csv"))
    return candidates[0] if candidates else None


def add_lakepinn_to_comparison(lakepinn_csv: Path, label: str) -> pd.DataFrame:
    matched_path = OUTPUT_DIR / "matched_point_predictions.csv"
    if not matched_path.exists():
        build_official_pgdl_outputs()
    matched = pd.read_csv(matched_path)
    matched["Date"] = pd.to_datetime(matched["Date"])
    pred = pd.read_csv(lakepinn_csv)
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred_col = f"{label}_C"
    pred = pred.rename(columns={"Temperature_C": pred_col})[["Date", "Depth_m", pred_col]]

    rows = []
    for date, group in matched.groupby("Date", sort=False):
        pred_group = pred[pred["Date"].dt.normalize() == pd.Timestamp(date).normalize()].sort_values("Depth_m")
        if pred_group.empty:
            continue
        z = pred_group["Depth_m"].to_numpy(dtype=float)
        t = pred_group[pred_col].to_numpy(dtype=float)
        finite = np.isfinite(z) & np.isfinite(t)
        if finite.sum() < 2:
            continue
        values = np.interp(group["Depth_m"].to_numpy(dtype=float), z[finite], t[finite])
        tmp = group.copy()
        tmp[pred_col] = values
        rows.append(tmp)
    if not rows:
        raise RuntimeError(f"No LakePINN predictions could be matched from {lakepinn_csv}")
    updated = pd.concat(rows, ignore_index=True)
    updated.to_csv(matched_path, index=False)
    model_cols = {
        "PB_official": "PB_C",
        "DL_official": "DL_C",
        "PGDL_official": "PGDL_C",
        label: pred_col,
    }
    metrics = compute_grouped_metrics(updated, model_cols)
    metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)
    lakepinn_metrics = metrics[metrics["model"] == label].copy()
    lakepinn_metrics.to_csv(OUTPUT_DIR / "lakepinn_metrics.csv", index=False)
    return metrics


def plot_outputs() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = OUTPUT_DIR / "model_comparison_metrics.csv"
    matched_path = OUTPUT_DIR / "matched_point_predictions.csv"
    if not metrics_path.exists() or not matched_path.exists():
        build_official_pgdl_outputs()
    metrics = pd.read_csv(metrics_path)
    matched = pd.read_csv(matched_path)
    matched["Date"] = pd.to_datetime(matched["Date"])
    paths = []

    overall = metrics[metrics["group"] == "overall"].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(overall["model"], overall["rmse_C"], color=["#777777", "#4C78A8", "#59A14F", "#E15759"][: len(overall)])
    plt.ylabel("RMSE (C)")
    plt.title("Mendota matched observation-point RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = OUTPUT_DIR / "model_comparison_rmse.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    month_metrics = metrics[metrics["group"].str.startswith("month_")].copy()
    if not month_metrics.empty:
        month_metrics["month"] = month_metrics["group"].str.replace("month_", "", regex=False).astype(int)
        plt.figure(figsize=(9, 4.8))
        for model, group in month_metrics.groupby("model"):
            plt.plot(group["month"], group["rmse_C"], marker="o", label=model)
        plt.ylabel("RMSE (C)")
        plt.xlabel("Month")
        plt.title("Monthly RMSE on matched observation points")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = OUTPUT_DIR / "monthly_rmse.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    depth_metrics = metrics[metrics["group"].isin(["surface_0_3m", "mid_3_12m", "deep_gt12m"])].copy()
    if not depth_metrics.empty:
        pivot = depth_metrics.pivot(index="group", columns="model", values="rmse_C").loc[
            ["surface_0_3m", "mid_3_12m", "deep_gt12m"]
        ]
        pivot.plot(kind="bar", figsize=(9, 4.8))
        plt.ylabel("RMSE (C)")
        plt.title("Depth-band RMSE")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = OUTPUT_DIR / "depthband_rmse.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    pred_columns = [c for c in matched.columns if c.endswith("_C") and c != "Observed_C"]
    if pred_columns:
        ncols = min(2, len(pred_columns))
        nrows = int(math.ceil(len(pred_columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows), squeeze=False)
        obs = matched["Observed_C"].to_numpy(dtype=float)
        lo = float(np.nanmin(obs))
        hi = float(np.nanmax(obs))
        for ax, column in zip(axes.ravel(), pred_columns):
            ax.scatter(obs, matched[column], s=8, alpha=0.35)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_title(column.replace("_C", ""))
            ax.set_xlabel("Observed (C)")
            ax.set_ylabel("Predicted (C)")
        for ax in axes.ravel()[len(pred_columns):]:
            ax.axis("off")
        fig.suptitle("Predicted vs observed at matched points")
        fig.tight_layout()
        path = OUTPUT_DIR / "predicted_vs_observed.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    daily_records = []
    for column in pred_columns:
        model = column.replace("_C", "")
        for date, group in matched.dropna(subset=[column]).groupby("Date"):
            diff = group[column] - group["Observed_C"]
            daily_records.append(
                {
                    "Date": date,
                    "model": model,
                    "rmse_C": float(np.sqrt(np.mean(np.square(diff)))),
                    "n": int(len(diff)),
                }
            )
    if daily_records:
        daily = pd.DataFrame(daily_records)
        daily.to_csv(OUTPUT_DIR / "daily_rmse.csv", index=False)
        plt.figure(figsize=(10, 4.8))
        for model, group in daily.groupby("model"):
            group = group.sort_values("Date")
            plt.plot(group["Date"], group["rmse_C"], label=model, alpha=0.85)
        plt.ylabel("Daily profile RMSE (C)")
        plt.title("Daily profile RMSE on matched observation dates")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = OUTPUT_DIR / "daily_rmse.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))
    return paths


def write_run_notes(extra: dict | None = None) -> dict:
    notes = {
        "benchmark": "PGDL-WRR 2019 Mendota comparison",
        "repo_root": str(REPO_ROOT),
        "external_dir": str(EXTERNAL_DIR),
        "output_dir": str(OUTPUT_DIR),
        "primary_experiment": {"exper_id": PRIMARY_EXPER_ID, "exper_n": PRIMARY_EXPER_N},
        "data_citation": PGDL_DATA_CITATION,
        "software_citation": PGDL_SOFTWARE_CITATION,
        "main_comparison_policy": "Main LakePINN manifest sets lst_surface_weight=0 and uses a dummy all-missing LST file filled internally from air temperature.",
        "fairness_note": "LakePINN core still expects an LST feature column; in the no-LST benchmark it is not a satellite observation and carries no LST loss authority.",
        "historical_note": "Older seventh-version Hostetler and LakePINN outputs are not used as primary inputs for this benchmark.",
    }
    if extra:
        notes.update(extra)
    (OUTPUT_DIR / "run_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return notes


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dirs()
    extra_notes: dict[str, object] = {}
    if not args.skip_download:
        download_pgdl_assets(overwrite=args.overwrite)
    rmse, matched, metrics = build_official_pgdl_outputs()
    _, _, _, _, manifest_path = convert_meteo_to_lakepinn_inputs()
    extra_notes["lakepinn_manifest"] = str(manifest_path)
    extra_notes["pgdl_official_overall_metrics"] = str(OUTPUT_DIR / "pgdl_official_metrics.csv")
    extra_notes["model_comparison_metrics"] = str(OUTPUT_DIR / "model_comparison_metrics.csv")

    if args.run_lakepinn_smoke:
        result = run_lakepinn(manifest_path, epochs=args.smoke_epochs, output_name="lakepinn_no_lst_smoke", smoke=True)
        extra_notes["lakepinn_smoke"] = result
        prediction_csv = find_lakepinn_prediction_csv(Path(result["output_dir"]))
        if prediction_csv is not None and result["returncode"] == 0:
            add_lakepinn_to_comparison(prediction_csv, "LakePINN_no_LST_smoke")

    if args.run_lakepinn:
        result = run_lakepinn(manifest_path, epochs=args.epochs, output_name="lakepinn_no_lst_full", smoke=False)
        extra_notes["lakepinn_full"] = result
        prediction_csv = find_lakepinn_prediction_csv(Path(result["output_dir"]))
        if prediction_csv is not None and result["returncode"] == 0:
            add_lakepinn_to_comparison(prediction_csv, "LakePINN_no_LST")

    plot_paths = plot_outputs()
    extra_notes["figures"] = plot_paths
    extra_notes["official_matched_points"] = int(len(matched))
    write_run_notes(extra_notes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PGDL-WRR 2019 benchmark comparison under LakePINN v8.")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded official assets.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if they already exist.")
    parser.add_argument("--run-lakepinn-smoke", action="store_true", help="Run a short LakePINN no-LST smoke training.")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--run-lakepinn", action="store_true", help="Run full LakePINN no-LST training for this benchmark.")
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
