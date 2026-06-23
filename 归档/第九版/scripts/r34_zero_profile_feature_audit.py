"""R34 zero-profile feature audit.

This diagnostic reads the active LOCAL62 manifest/split and standard inputs,
then reports which zero-profile observable features and profile-derived
auxiliary targets are already available or constructible.

It does not train, evaluate checkpoints, modify model code, or edit standard
inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ID = "RECON_R34_ZERO_PROFILE_FEATURE_AUDIT_v1"
RUN_TAG = "R34_zero_profile_feature_audit_20260614"


@dataclass(frozen=True)
class Roots:
    v9_root: Path
    pin_root: Path
    data_root: Path
    pipeline_root: Path


def default_roots() -> Roots:
    v9_root = Path(__file__).resolve().parents[1]
    pin_root = v9_root.parent
    data_root = pin_root / "\u6570\u636e" / "_standard_inputs"
    pipeline_root = pin_root / "pipeline"
    return Roots(
        v9_root=v9_root,
        pin_root=pin_root,
        data_root=data_root,
        pipeline_root=pipeline_root,
    )


def parse_args() -> argparse.Namespace:
    roots = default_roots()
    return _parse_args(roots)


def _parse_args(roots: Roots) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R34 zero-profile feature audit.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=roots.v9_root
        / "experiments"
        / "manifests_clean"
        / "diagnostics"
        / "RECON_LOCAL62_ZERO_PROFILE_GROUPHELDOUT_V3_active_manifest_20260614.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=roots.v9_root
        / "experiments"
        / "splits"
        / "LOCAL62_ZERO_PROFILE_GROUPHELDOUT_V3.json",
    )
    parser.add_argument("--standard-input-root", type=Path, default=roots.data_root)
    parser.add_argument(
        "--roles-csv",
        type=Path,
        default=roots.pipeline_root
        / "reports"
        / "data_readiness"
        / "A6_LOCAL62_ZERO_PROFILE_V3_candidate_roles_20260614.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=roots.pipeline_root / "reports" / "feature_audit" / RUN_TAG,
    )
    parser.add_argument(
        "--closeout",
        type=Path,
        default=roots.pipeline_root / "cycle_closeouts" / f"{RUN_TAG}_closeout.md",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def finite_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def first_existing(columns: list[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def nonempty_numeric(frame: pd.DataFrame, columns: list[str]) -> bool:
    series = finite_series(frame, columns)
    return bool(series.notna().any())


def nonzero_numeric(frame: pd.DataFrame, columns: list[str], tol: float = 1.0e-8) -> bool:
    series = finite_series(frame, columns)
    if not series.notna().any():
        return False
    return bool((series.fillna(0.0).abs() > tol).any())


def parse_lake_year(lake_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_(\d{4})$", str(lake_id))
    if not match:
        return str(lake_id), None
    return match.group(1), int(match.group(2))


def role_from_manifest(manifest: dict, lake_id: str) -> str:
    if lake_id in set(manifest.get("train_lake_ids", [])):
        return "train"
    if lake_id in set(manifest.get("val_lake_ids", [])):
        return "validation"
    if lake_id in set(manifest.get("heldout_lake_ids", manifest.get("test_lake_ids", []))):
        return "heldout_diagnostic_only"
    if lake_id in set(manifest.get("stress_or_ood_lake_ids", [])):
        return "stress_ood_diagnostic_only"
    if lake_id in set(manifest.get("blocked_lake_ids", [])):
        return "blocked"
    return "unknown"


def role_normalized(candidate_role: str, manifest_role: str) -> str:
    value = str(candidate_role or "").lower()
    if "train" in value:
        return "train"
    if "val" in value:
        return "validation"
    if "heldout" in value:
        return "heldout_diagnostic_only"
    if "stress" in value or "ood" in value:
        return "stress_ood_diagnostic_only"
    return manifest_role


def is_reservoir(metadata: dict, role_row: dict | None = None) -> bool:
    role_value = None if role_row is None else role_row.get("metadata_is_reservoir")
    if role_value is not None and not pd.isna(role_value):
        try:
            return float(role_value) > 0.5
        except (TypeError, ValueError):
            pass
    for key in ("reservoir_indicator", "is_reservoir"):
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value) > 0.5
            except (TypeError, ValueError):
                pass
    lake_type = str(metadata.get("lake_type", "")).lower()
    return "reservoir" in lake_type


def safe_float(value, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def mean_depth(metadata: dict) -> float:
    return safe_float(metadata.get("mean_depth_m", metadata.get("mean_depth", np.nan)))


def max_depth(metadata: dict) -> float:
    return safe_float(metadata.get("max_depth_m", metadata.get("max_depth", np.nan)))


def area_km2(metadata: dict) -> float:
    return safe_float(metadata.get("area_km2", metadata.get("surface_area_km2", np.nan)))


def kd_value(metadata: dict, era5: pd.DataFrame | None = None) -> float:
    candidates = [
        metadata.get("light_extinction_kd"),
        metadata.get("kd_m_inv"),
        metadata.get("Kd_m_inv"),
    ]
    if era5 is not None:
        series = finite_series(era5, ["light_extinction_kd", "Kd_m_inv", "kd_m_inv"])
        if series.notna().any():
            candidates.append(float(series.dropna().median()))
    for value in candidates:
        numeric = safe_float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            return numeric
    secchi = safe_float(metadata.get("secchi_m", metadata.get("Secchi_m", np.nan)))
    if math.isfinite(secchi) and secchi > 0.0:
        return 1.7 / secchi
    return float("nan")


def secchi_value(metadata: dict, era5: pd.DataFrame | None = None) -> float:
    candidates = [metadata.get("secchi_m"), metadata.get("Secchi_m")]
    if era5 is not None:
        series = finite_series(era5, ["Secchi_m", "secchi_m"])
        if series.notna().any():
            candidates.append(float(series.dropna().median()))
    for value in candidates:
        numeric = safe_float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            return numeric
    return float("nan")


def kd_source_type(metadata: dict, era5: pd.DataFrame | None = None) -> str:
    if era5 is not None and nonempty_numeric(era5, ["light_extinction_kd", "Kd_m_inv", "kd_m_inv"]):
        return "daily_or_forcing_kd"
    completion = metadata.get("metadata_completion_v20260612_optics")
    if isinstance(completion, dict):
        sources = " ".join(str(x) for x in completion.get("sources_or_methods", []))
        status = str(completion.get("measurement_status", ""))
        if "estimated" in sources.lower() or "estimated" in status.lower():
            return "metadata_estimated_or_default"
        if "derived" in sources.lower() or "derived" in status.lower():
            return "metadata_derived"
    if safe_float(metadata.get("light_extinction_kd")) > 0.0:
        return "metadata_kd"
    if safe_float(metadata.get("secchi_m")) > 0.0:
        return "metadata_secchi_derived"
    return "missing"


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return read_csv(path)
    except Exception:
        return None


def lswt_raw_mask(lst: pd.DataFrame) -> pd.Series:
    lswt = finite_series(lst, ["LSWT_open_water_C"])
    observed = finite_series(lst, ["LST_observed_flag"]).fillna(0.0) > 0.5
    filled = finite_series(lst, ["LST_is_filled"]).fillna(0.0) > 0.5
    ice = finite_series(lst, ["ice_fraction"]).fillna(0.0)
    quality = finite_series(lst, ["LST_quality_factor"]).fillna(0.0)
    return lswt.notna() | (observed & (~filled) & (ice <= 0.0) & (quality > 0.0))


def lswt_series(lst: pd.DataFrame) -> pd.Series:
    lswt = finite_series(lst, ["LSWT_open_water_C"])
    if lswt.notna().any():
        return lswt
    surface = finite_series(lst, ["LST_surface_C", "LST_day_C"])
    return surface.where(lswt_raw_mask(lst))


def rolling_count(values: pd.Series, window: int) -> pd.Series:
    return values.notna().astype(float).rolling(window, min_periods=1).sum()


def days_since_raw(mask: pd.Series) -> pd.Series:
    result = []
    last = None
    for idx, raw in enumerate(mask.astype(bool).tolist()):
        if raw:
            last = idx
            result.append(0.0)
        elif last is None:
            result.append(np.nan)
        else:
            result.append(float(idx - last))
    return pd.Series(result, index=mask.index, dtype="float64")


def trend_availability(values: pd.Series, window: int = 7) -> pd.Series:
    return rolling_count(values, window) >= 2.0


def audit_lswt_memory(lake_id: str, role: str, lake_type: str, lst: pd.DataFrame | None) -> dict:
    row = {
        "lake_id": lake_id,
        "role": role,
        "lake_type": lake_type,
        "lst_file_present": lst is not None,
        "days_total": 0,
        "raw_open_water_lswt_days": 0,
        "filled_or_reconstructed_days": 0,
        "missing_lswt_days": 0,
        "raw_open_water_fraction": np.nan,
        "filled_fraction": np.nan,
        "days_since_last_raw_LSWT_status": "missing",
        "days_since_last_raw_LSWT_p50": np.nan,
        "days_since_last_raw_LSWT_p95": np.nan,
        "days_since_last_raw_LSWT_max": np.nan,
        "raw_LSWT_valid_count_30d_status": "missing",
        "raw_LSWT_valid_count_30d_mean": np.nan,
        "raw_LSWT_valid_count_30d_min": np.nan,
        "raw_LSWT_trend_7d_status": "missing",
        "raw_LSWT_trend_7d_available_fraction": np.nan,
        "filled_LST_strong_update_allowed": False,
        "filled_LST_use_policy": "feature_only_forbidden_as_strong_truth",
    }
    if lst is None or lst.empty:
        return row
    row["days_total"] = int(len(lst))
    raw_mask = lswt_raw_mask(lst)
    filled = finite_series(lst, ["LST_is_filled"]).fillna(0.0) > 0.5
    values = lswt_series(lst).where(raw_mask)
    raw_count = int(raw_mask.sum())
    row["raw_open_water_lswt_days"] = raw_count
    row["filled_or_reconstructed_days"] = int(filled.sum())
    row["missing_lswt_days"] = int(max(len(lst) - raw_count - int(filled.sum()), 0))
    row["raw_open_water_fraction"] = raw_count / max(len(lst), 1)
    row["filled_fraction"] = int(filled.sum()) / max(len(lst), 1)
    if raw_count > 0:
        age = days_since_raw(raw_mask)
        valid_age = age.dropna()
        row["days_since_last_raw_LSWT_status"] = "constructible_derived"
        row["days_since_last_raw_LSWT_p50"] = float(valid_age.quantile(0.50)) if not valid_age.empty else np.nan
        row["days_since_last_raw_LSWT_p95"] = float(valid_age.quantile(0.95)) if not valid_age.empty else np.nan
        row["days_since_last_raw_LSWT_max"] = float(valid_age.max()) if not valid_age.empty else np.nan
        count_30d = rolling_count(values, 30)
        row["raw_LSWT_valid_count_30d_status"] = "constructible_derived"
        row["raw_LSWT_valid_count_30d_mean"] = float(count_30d.mean())
        row["raw_LSWT_valid_count_30d_min"] = float(count_30d.min())
        trend_ok = trend_availability(values, 7)
        row["raw_LSWT_trend_7d_status"] = "constructible_derived" if bool(trend_ok.any()) else "insufficient_raw_lswt"
        row["raw_LSWT_trend_7d_available_fraction"] = float(trend_ok.mean())
    return row


def profile_to_long(profile: pd.DataFrame) -> pd.DataFrame:
    if {"Date", "Depth_m", "Temperature_C"}.issubset(profile.columns):
        out = profile[["Date", "Depth_m", "Temperature_C"]].copy()
        out["Depth_m"] = pd.to_numeric(out["Depth_m"], errors="coerce")
        out["Temperature_C"] = pd.to_numeric(out["Temperature_C"], errors="coerce")
        return out.dropna(subset=["Date", "Depth_m", "Temperature_C"])
    temp_columns = [c for c in profile.columns if re.fullmatch(r"Temp_[0-9]+(?:\.[0-9]+)?m", c)]
    rows = []
    for column in temp_columns:
        depth = float(column.removeprefix("Temp_").removesuffix("m"))
        partial = profile[["Date", column]].copy().rename(columns={column: "Temperature_C"})
        partial["Depth_m"] = depth
        rows.append(partial)
    if not rows:
        return pd.DataFrame(columns=["Date", "Depth_m", "Temperature_C"])
    out = pd.concat(rows, ignore_index=True)
    out["Temperature_C"] = pd.to_numeric(out["Temperature_C"], errors="coerce")
    return out.dropna(subset=["Date", "Depth_m", "Temperature_C"])


def audit_auxiliary_targets(lake_id: str, role: str, lake_type: str, profile: pd.DataFrame | None) -> dict:
    row = {
        "lake_id": lake_id,
        "role": role,
        "lake_type": lake_type,
        "profile_file_present": profile is not None,
        "profile_dates": 0,
        "profile_rows": 0,
        "median_layers_per_profile": np.nan,
        "median_observed_depth_range_m": np.nan,
        "surface_temperature_aux_status": "missing",
        "bottom_temperature_aux_status": "missing",
        "surface_deep_delta_aux_status": "missing",
        "observed_heat_content_0_obsmax_status": "missing",
        "observed_heat_content_0_25m_status": "missing",
        "thermocline_depth_obs_status": "missing",
        "MLD_obs_status": "missing",
        "stratification_strength_obs_status": "missing",
        "EOF_coefficients_obs_status": "missing",
        "dates_with_ge3_layers": 0,
        "dates_with_ge4_layers": 0,
        "dates_with_heat_content_0_25m": 0,
        "dates_with_thermocline_candidate": 0,
        "dates_with_eof_projection_candidate": 0,
        "profile_derived_as_zero_profile_input_policy": "forbidden_as_input",
    }
    if profile is None or profile.empty:
        return row
    long_profile = profile_to_long(profile)
    if long_profile.empty:
        return row
    row["profile_rows"] = int(len(long_profile))
    layer_counts = []
    depth_ranges = []
    ge3 = ge4 = heat25 = thermo = eof = 0
    for _, group in long_profile.groupby("Date"):
        depths = pd.to_numeric(group["Depth_m"], errors="coerce").to_numpy(dtype=float)
        temps = pd.to_numeric(group["Temperature_C"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(depths) & np.isfinite(temps)
        depths = depths[valid]
        temps = temps[valid]
        if depths.size == 0:
            continue
        order = np.argsort(depths)
        depths = depths[order]
        temps = temps[order]
        unique_depths = np.unique(np.round(depths, 3))
        layers = int(unique_depths.size)
        depth_range = float(np.nanmax(depths) - np.nanmin(depths)) if layers else np.nan
        layer_counts.append(layers)
        depth_ranges.append(depth_range)
        if layers >= 3:
            ge3 += 1
            eof += 1
        if layers >= 4:
            ge4 += 1
        if layers >= 2 and np.nanmin(depths) <= 25.0:
            heat25 += 1
        if layers >= 4 and np.isfinite(depth_range) and depth_range >= 2.0:
            gradients = np.abs(np.diff(temps) / np.clip(np.diff(depths), 1.0e-6, None))
            if gradients.size and np.nanmax(gradients) >= 0.05:
                thermo += 1
    date_count = len(layer_counts)
    row["profile_dates"] = int(date_count)
    row["median_layers_per_profile"] = float(np.nanmedian(layer_counts)) if layer_counts else np.nan
    row["median_observed_depth_range_m"] = float(np.nanmedian(depth_ranges)) if depth_ranges else np.nan
    row["dates_with_ge3_layers"] = int(ge3)
    row["dates_with_ge4_layers"] = int(ge4)
    row["dates_with_heat_content_0_25m"] = int(heat25)
    row["dates_with_thermocline_candidate"] = int(thermo)
    row["dates_with_eof_projection_candidate"] = int(eof)
    if date_count > 0:
        row["surface_temperature_aux_status"] = "profile_only_auxiliary"
        row["bottom_temperature_aux_status"] = "profile_only_auxiliary"
        row["surface_deep_delta_aux_status"] = "profile_only_auxiliary" if ge3 > 0 else "insufficient_depth_layers"
        row["stratification_strength_obs_status"] = "profile_only_auxiliary" if ge3 > 0 else "insufficient_depth_layers"
    if heat25 > 0:
        row["observed_heat_content_0_25m_status"] = "profile_only_auxiliary"
    if ge3 > 0:
        row["observed_heat_content_0_obsmax_status"] = "profile_only_auxiliary"
        row["MLD_obs_status"] = "profile_only_auxiliary"
        row["EOF_coefficients_obs_status"] = "profile_only_auxiliary"
    if thermo > 0:
        row["thermocline_depth_obs_status"] = "profile_only_auxiliary"
    return row


def feature_status(condition: bool, true_status: str = "constructible_derived") -> str:
    return true_status if condition else "missing"


def audit_lake_year(
    lake_id: str,
    manifest: dict,
    standard_root: Path,
    role_row: dict | None,
) -> tuple[dict, dict, dict]:
    lake_group, year = parse_lake_year(lake_id)
    manifest_role = role_from_manifest(manifest, lake_id)
    candidate_role = "" if role_row is None else str(role_row.get("candidate_role", ""))
    role = role_normalized(candidate_role, manifest_role)
    lake_dir = standard_root / lake_id
    metadata_path = lake_dir / "metadata.json"
    era5_path = lake_dir / "era5_for_model.csv"
    lst_path = lake_dir / "lst_day_for_model.csv"
    profile_path = lake_dir / "profile_for_model.csv"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    era5 = load_optional_csv(era5_path)
    lst = load_optional_csv(lst_path)
    profile = load_optional_csv(profile_path)
    reservoir = is_reservoir(metadata, role_row)
    lake_type = "reservoir" if reservoir else "natural"
    md = mean_depth(metadata)
    xd = max_depth(metadata)
    area = area_km2(metadata)
    kd = kd_value(metadata, era5)
    secchi = secchi_value(metadata, era5)
    fetch = safe_float(metadata.get("effective_fetch", metadata.get("effective_fetch_m", np.nan)))
    if era5 is not None and not math.isfinite(fetch):
        fetch_series = finite_series(era5, ["effective_fetch", "effective_fetch_m", "fetch_m", "wind_fetch_m"])
        if fetch_series.notna().any():
            fetch = float(fetch_series.dropna().median())
    wind_available = era5 is not None and (
        nonempty_numeric(era5, ["wind_speed_m_per_s", "wind_norm_m_per_s", "wind_speed"])
        or (
            nonempty_numeric(era5, ["u10_m_per_s", "u10", "u10m"])
            and nonempty_numeric(era5, ["v10_m_per_s", "v10", "v10m"])
        )
    )
    shortwave_available = era5 is not None and nonempty_numeric(era5, ["Solar_W_m2", "ssrd_W_per_m2", "ssrd_J_per_m2"])
    longwave_available = era5 is not None and nonempty_numeric(era5, ["Longwave_W_m2", "strd_W_per_m2", "strd_J_per_m2"])
    latent_available = era5 is not None and nonempty_numeric(era5, ["latent_heat_upward_W_m2", "latent_heat_upward_W_per_m2", "slhf_W_per_m2_raw"])
    sensible_available = era5 is not None and nonempty_numeric(era5, ["sensible_heat_upward_W_m2", "sensible_heat_upward_W_per_m2", "sshf_W_per_m2_raw"])
    water_level_available = era5 is not None and nonzero_numeric(era5, ["water_level_anomaly", "WaterLevelAnomaly_m", "lake_level_anomaly_m"])
    inflow_available = era5 is not None and nonzero_numeric(era5, ["net_inflow", "net_inflow_m3_s", "NetInflow_m3_s", "inflow_m3_s", "outflow_m3_s"])
    hydrology_available = bool(water_level_available or inflow_available or safe_float(metadata.get("residence_time_days")) > 0.0)
    lswt_row = audit_lswt_memory(lake_id, role, lake_type, lst)
    aux_row = audit_auxiliary_targets(lake_id, role, lake_type, profile)
    heat_capacity_ready = math.isfinite(md) and md > 0.0
    light_ready = math.isfinite(xd) and xd > 0.0 and (math.isfinite(kd) or math.isfinite(secchi))
    wind_direct_ready = wind_available and math.isfinite(fetch) and fetch > 0.0 and math.isfinite(xd) and xd > 0.0
    wind_proxy_ready = wind_available and math.isfinite(area) and area > 0.0 and math.isfinite(xd) and xd > 0.0
    net_radiation_ready = bool(shortwave_available and longwave_available)
    forcing_history_ready = era5 is not None and len(era5) >= 30
    feature_row = {
        "lake_id": lake_id,
        "lake_group": lake_group,
        "year": year,
        "role": role,
        "heldout_policy": "diagnostic_only" if "heldout" in role or "stress" in role else "validation_or_train_only",
        "lake_type": lake_type,
        "standard_input_dir": str(lake_dir),
        "metadata_present": metadata_path.exists(),
        "era5_present": era5 is not None,
        "lst_day_present": lst is not None,
        "profile_present": profile is not None,
        "metadata_mean_depth_m": md,
        "metadata_max_depth_m": xd,
        "metadata_area_km2": area,
        "metadata_kd_m_inv": kd,
        "metadata_secchi_m": secchi,
        "Kd_source_type": kd_source_type(metadata, era5),
        "ERA5_forcing_status": "existing_input" if era5 is not None else "missing",
        "metadata_status": "existing_input" if metadata_path.exists() else "missing",
        "LST_LSWT_status": "existing_input" if lst is not None else "missing",
        "LST_quality_fill_flags_status": "existing_input"
        if lst is not None and {"LST_quality_factor", "LST_is_filled", "LST_observed_flag"}.intersection(lst.columns)
        else "missing",
        "reservoir_indicator_status": "existing_input" if metadata_path.exists() else "missing",
        "hydrology_input_status": "existing_input" if hydrology_available else "missing_or_zero_filled",
        "hydrology_missing_flag_status": "constructible_derived",
        "heat_capacity_areal_status": feature_status(heat_capacity_ready),
        "light_penetration_ratio_status": feature_status(light_ready),
        "wind_mixing_potential_7d_status": feature_status(wind_direct_ready or wind_proxy_ready),
        "wind_mixing_potential_30d_status": feature_status(wind_direct_ready or wind_proxy_ready),
        "wind_mixing_potential_source": "direct_fetch"
        if wind_direct_ready
        else ("area_proxy" if wind_proxy_ready else "missing"),
        "net_radiation_7d_status": feature_status(net_radiation_ready),
        "net_radiation_30d_status": feature_status(net_radiation_ready),
        "net_heat_flux_components_status": "constructible_derived"
        if (shortwave_available and longwave_available and latent_available and sensible_available)
        else "partial_or_missing",
        "forcing_cumulative_7d_30d_status": feature_status(forcing_history_ready),
        "days_since_last_raw_LSWT_status": lswt_row["days_since_last_raw_LSWT_status"],
        "raw_LSWT_valid_count_30d_status": lswt_row["raw_LSWT_valid_count_30d_status"],
        "raw_LSWT_trend_7d_status": lswt_row["raw_LSWT_trend_7d_status"],
        "profile_derived_auxiliary_status": "profile_only_auxiliary" if aux_row["profile_dates"] > 0 else "missing",
        "profile_derived_as_zero_profile_input_policy": "forbidden_as_input",
        "filled_LST_strong_update_allowed": False,
    }
    return feature_row, lswt_row, aux_row


def build_role_lookup(roles_csv: Path) -> dict[str, dict]:
    if not roles_csv.exists():
        return {}
    roles = read_csv(roles_csv)
    if "lake_id" not in roles.columns:
        return {}
    return {
        str(row["lake_id"]): row.to_dict()
        for _, row in roles.iterrows()
    }


def all_lake_ids(manifest: dict) -> list[str]:
    ids: list[str] = []
    for key in ("train_lake_ids", "val_lake_ids", "heldout_lake_ids", "stress_or_ood_lake_ids", "blocked_lake_ids"):
        ids.extend(str(x) for x in manifest.get(key, []))
    if not ids and "all_standard_lake_ids" in manifest:
        ids.extend(str(x) for x in manifest.get("all_standard_lake_ids", []))
    return list(dict.fromkeys(ids))


def status_rate(frame: pd.DataFrame, column: str, ok_values: set[str]) -> float:
    if column not in frame.columns or frame.empty:
        return np.nan
    return float(frame[column].isin(ok_values).mean())


def summarize_gaps(feature_df: pd.DataFrame, lswt_df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    merged = feature_df.merge(
        lswt_df[["lake_id", "raw_open_water_fraction", "days_since_last_raw_LSWT_p95", "raw_LSWT_valid_count_30d_mean"]],
        on="lake_id",
        how="left",
    ).merge(
        aux_df[["lake_id", "profile_dates", "dates_with_eof_projection_candidate", "dates_with_thermocline_candidate"]],
        on="lake_id",
        how="left",
    )
    rows = []
    for (role, lake_type), group in merged.groupby(["role", "lake_type"], dropna=False):
        rows.append(
            {
                "role": role,
                "lake_type": lake_type,
                "lake_years": int(len(group)),
                "era5_input_rate": status_rate(group, "ERA5_forcing_status", {"existing_input"}),
                "lst_input_rate": status_rate(group, "LST_LSWT_status", {"existing_input"}),
                "heat_capacity_areal_ready_rate": status_rate(group, "heat_capacity_areal_status", {"constructible_derived"}),
                "light_penetration_ratio_ready_rate": status_rate(group, "light_penetration_ratio_status", {"constructible_derived"}),
                "wind_mixing_potential_ready_rate": status_rate(group, "wind_mixing_potential_7d_status", {"constructible_derived"}),
                "net_radiation_ready_rate": status_rate(group, "net_radiation_7d_status", {"constructible_derived"}),
                "lswt_memory_ready_rate": status_rate(group, "days_since_last_raw_LSWT_status", {"constructible_derived"}),
                "hydrology_available_rate": status_rate(group, "hydrology_input_status", {"existing_input"}),
                "aux_profile_available_rate": status_rate(group, "profile_derived_auxiliary_status", {"profile_only_auxiliary"}),
                "mean_raw_open_water_fraction": float(group["raw_open_water_fraction"].mean()),
                "median_days_since_raw_p95": float(group["days_since_last_raw_LSWT_p95"].median()),
                "mean_raw_LSWT_valid_count_30d": float(group["raw_LSWT_valid_count_30d_mean"].mean()),
                "total_profile_dates": int(pd.to_numeric(group["profile_dates"], errors="coerce").fillna(0).sum()),
                "total_eof_projection_candidate_dates": int(
                    pd.to_numeric(group["dates_with_eof_projection_candidate"], errors="coerce").fillna(0).sum()
                ),
                "total_thermocline_candidate_dates": int(
                    pd.to_numeric(group["dates_with_thermocline_candidate"], errors="coerce").fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "lake_type"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    manifest_path: Path,
    split_path: Path,
    standard_root: Path,
    feature_df: pd.DataFrame,
    lswt_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    report_path = output_dir / "R34_zero_profile_feature_audit_report.md"
    proposal_path = output_dir / "R34_next_model_change_proposal.md"
    role_counts = Counter(feature_df["role"])
    p0_features = [
        "heat_capacity_areal",
        "light_penetration_ratio",
        "wind_mixing_potential_7d/30d",
        "net_radiation_7d/30d",
        "days_since_last_raw_LSWT",
        "raw_LSWT_valid_count_30d",
        "raw_LSWT_trend_7d",
        "hydrology_missing_flag",
    ]
    train_aux = aux_df[aux_df["role"] == "train"]
    train_eof_dates = int(pd.to_numeric(train_aux["dates_with_eof_projection_candidate"], errors="coerce").fillna(0).sum())
    train_thermo_dates = int(pd.to_numeric(train_aux["dates_with_thermocline_candidate"], errors="coerce").fillna(0).sum())
    train_profile_dates = int(pd.to_numeric(train_aux["profile_dates"], errors="coerce").fillna(0).sum())
    train_lakes_with_eof = int((pd.to_numeric(train_aux["dates_with_eof_projection_candidate"], errors="coerce").fillna(0) > 0).sum())
    eof_go = train_eof_dates >= 500 and train_lakes_with_eof >= 10
    heat_loss_go = False
    feature_cols = [
        "heat_capacity_areal_status",
        "light_penetration_ratio_status",
        "wind_mixing_potential_7d_status",
        "net_radiation_7d_status",
        "days_since_last_raw_LSWT_status",
        "raw_LSWT_valid_count_30d_status",
        "raw_LSWT_trend_7d_status",
        "hydrology_missing_flag_status",
    ]
    ready_lines = []
    for col in feature_cols:
        ready = int(feature_df[col].isin({"constructible_derived"}).sum())
        ready_lines.append(f"- `{col}`: {ready}/{len(feature_df)} ready")
    summary_table = dataframe_to_markdown(summary_df)
    report = [
        "# R34 zero-profile feature audit",
        "",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        "- status: `completed_local_audit_no_training`",
        "- autonomy_level: `A1_local_diagnostic` / `A6_data_split_steward_readonly`",
        "- model/loss/physics/split/_standard_inputs: unchanged",
        "- heldout and stress/OOD: diagnostic-only, not used for checkpoint selection or tuning",
        f"- manifest: `{manifest_path}`",
        f"- split: `{split_path}`",
        f"- standard inputs: `{standard_root}`",
        "",
        "## Split coverage",
        "",
        f"- total lake-years audited: {len(feature_df)}",
        f"- train: {role_counts.get('train', 0)}",
        f"- validation: {role_counts.get('validation', 0)}",
        f"- heldout diagnostic-only: {role_counts.get('heldout_diagnostic_only', 0)}",
        f"- stress/OOD diagnostic-only: {role_counts.get('stress_ood_diagnostic_only', 0)}",
        "",
        "## P0 feature readiness",
        "",
        *ready_lines,
        "",
        "Interpretation: most P0 features are audit-safe because they use only metadata, ERA5, satellite LSWT/LST, or missingness flags. Profile-derived quantities remain auxiliary labels only.",
        "",
        "## LSWT memory",
        "",
        f"- mean raw-open-water LSWT fraction: {lswt_df['raw_open_water_fraction'].mean():.3f}",
        f"- median p95 days-since-raw-LSWT: {lswt_df['days_since_last_raw_LSWT_p95'].median():.1f} days",
        "- filled/reconstructed LST strong update allowed: `False` for every audited lake-year",
        "",
        "## Profile-derived auxiliary targets",
        "",
        f"- train profile dates: {train_profile_dates}",
        f"- train EOF/PCA projection candidate dates: {train_eof_dates}",
        f"- train thermocline candidate dates: {train_thermo_dates}",
        f"- train lake-years with EOF projection candidates: {train_lakes_with_eof}",
        f"- R35 EOF/PCA low-dimensional thermal state branch readiness: `{'Go for proposal' if eof_go else 'No-Go until more auxiliary coverage'}`",
        "",
        "## Heat-closure decision",
        "",
        "- R27/R28 already showed that surface-only heat residual is a source-bookkeeping issue.",
        "- Immediate weak heat-closure loss training remains `No-Go` in R34.",
        f"- heat-closure loss go flag: `{str(heat_loss_go).lower()}`",
        "",
        "## Role/lake-type summary",
        "",
        summary_table,
        "",
        "## Next action",
        "",
        "- Write/approve an R35 model-change proposal for a train-only EOF/PCA low-dimensional thermal-state auxiliary branch.",
        "- Add P0 observable feature engineering only after approval because it changes model input dimensionality.",
        "- Do not start R35/R36 training from this audit alone.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    proposal = [
        "# R35 low-dimensional thermal state proposal",
        "",
        f"Source diagnostic: `{EXPERIMENT_ID}`",
        "",
        "## Decision",
        "",
        "`needs_approval`: R34 supports writing a concrete R35 implementation plan, but does not approve model code changes or training.",
        "",
        "## Proposed model direction",
        "",
        "- Fit EOF/PCA vertical thermal basis on train profiles only; do not fit on validation, heldout, or stress/OOD profiles.",
        "- Project sparse profiles onto the frozen train basis with masked least squares to create auxiliary coefficient labels.",
        "- Add an auxiliary head/loss for EOF coefficients, observed heat content, MLD, thermocline depth, and stratification strength.",
        "- Keep zero-profile inference inputs restricted to metadata + ERA5 + satellite LSWT/LST + quality/missingness/hydrology flags.",
        "- Keep profile-derived labels forbidden as zero-profile inputs.",
        "",
        "## P0 observable features to consider with approval",
        "",
        *[f"- `{feature}`" for feature in p0_features],
        "",
        "## Explicit No-Go",
        "",
        "- No immediate heat-closure loss training.",
        "- No process-model pretraining before the EOF/PCA auxiliary branch is designed and approved.",
        "- No heldout/stress tuning or checkpoint selection.",
        "- No formal seed or long GPU run from R34 alone.",
    ]
    proposal_path.write_text("\n".join(proposal) + "\n", encoding="utf-8")
    return report_path, proposal_path


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_closeout(
    closeout_path: Path,
    output_dir: Path,
    report_path: Path,
    proposal_path: Path,
    feature_df: pd.DataFrame,
    lswt_df: pd.DataFrame,
    aux_df: pd.DataFrame,
) -> None:
    role_counts = Counter(feature_df["role"])
    train_aux = aux_df[aux_df["role"] == "train"]
    train_eof_dates = int(pd.to_numeric(train_aux["dates_with_eof_projection_candidate"], errors="coerce").fillna(0).sum())
    text = [
        "# R34 zero-profile feature audit closeout",
        "",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        "- status: completed local audit; no training",
        "- model-change statement: no model/loss/physics/data/split/_standard_inputs change",
        "- checkpoint: not used",
        "- heldout/stress: diagnostic-only, not used for checkpoint selection or tuning",
        "",
        "## What was executed",
        "",
        "- Added and ran a local feature-audit diagnostic script.",
        "- Audited zero-profile observable inputs, derived P0 feature readiness, LSWT memory coverage, and profile-derived auxiliary target availability.",
        "- Did not implement EOF/PCA, heat-closure loss, process pretraining, reservoir advection, or any training run.",
        "",
        "## Split counts",
        "",
        f"- train: {role_counts.get('train', 0)} lake-years",
        f"- validation: {role_counts.get('validation', 0)} lake-years",
        f"- heldout diagnostic-only: {role_counts.get('heldout_diagnostic_only', 0)} lake-years",
        f"- stress/OOD diagnostic-only: {role_counts.get('stress_ood_diagnostic_only', 0)} lake-years",
        "",
        "## Main findings",
        "",
        f"- LSWT raw-open-water mean coverage: {lswt_df['raw_open_water_fraction'].mean():.3f}",
        f"- train EOF/PCA projection candidate dates: {train_eof_dates}",
        "- filled/reconstructed LST strong-update policy remains forbidden.",
        "- profile-derived MLD/thermocline/heat-content/EOF labels are auxiliary-only and forbidden as zero-profile inputs.",
        "- immediate heat-closure loss remains No-Go because R28 source decomposition already explained the surface-only residual.",
        "",
        "## Next boundary",
        "",
        "- Next action class: model-change proposal only.",
        "- R35 EOF/PCA low-dimensional thermal state branch requires explicit approval before implementation or training.",
        "",
        "## Artifacts",
        "",
        f"- report_dir: `{output_dir}`",
        f"- report: `{report_path}`",
        f"- proposal: `{proposal_path}`",
        f"- feature availability: `{output_dir / 'R34_feature_availability_by_lakeyear.csv'}`",
        f"- LSWT memory: `{output_dir / 'R34_lstm_lswt_memory_coverage.csv'}`",
        f"- auxiliary targets: `{output_dir / 'R34_auxiliary_target_availability.csv'}`",
        f"- gap summary: `{output_dir / 'R34_feature_gap_summary_by_role_laketype.csv'}`",
    ]
    closeout_path.parent.mkdir(parents=True, exist_ok=True)
    closeout_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    lake_ids = all_lake_ids(manifest)
    role_lookup = build_role_lookup(args.roles_csv)
    if not lake_ids:
        raise ValueError(f"No lake ids found in manifest: {args.manifest}")
    feature_rows = []
    lswt_rows = []
    aux_rows = []
    for lake_id in lake_ids:
        feature_row, lswt_row, aux_row = audit_lake_year(
            lake_id=lake_id,
            manifest=manifest,
            standard_root=args.standard_input_root,
            role_row=role_lookup.get(lake_id),
        )
        feature_rows.append(feature_row)
        lswt_rows.append(lswt_row)
        aux_rows.append(aux_row)
    feature_df = pd.DataFrame(feature_rows)
    lswt_df = pd.DataFrame(lswt_rows)
    aux_df = pd.DataFrame(aux_rows)
    summary_df = summarize_gaps(feature_df, lswt_df, aux_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "R34_feature_availability_by_lakeyear.csv"
    lswt_path = args.output_dir / "R34_lstm_lswt_memory_coverage.csv"
    aux_path = args.output_dir / "R34_auxiliary_target_availability.csv"
    summary_path = args.output_dir / "R34_feature_gap_summary_by_role_laketype.csv"
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    lswt_df.to_csv(lswt_path, index=False, encoding="utf-8-sig")
    aux_df.to_csv(aux_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path, proposal_path = write_report(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        split_path=args.split,
        standard_root=args.standard_input_root,
        feature_df=feature_df,
        lswt_df=lswt_df,
        aux_df=aux_df,
        summary_df=summary_df,
    )
    write_closeout(
        closeout_path=args.closeout,
        output_dir=args.output_dir,
        report_path=report_path,
        proposal_path=proposal_path,
        feature_df=feature_df,
        lswt_df=lswt_df,
        aux_df=aux_df,
    )
    print(f"experiment_id={EXPERIMENT_ID}")
    print(f"audited_lake_years={len(feature_df)}")
    print(f"report={report_path}")
    print(f"proposal={proposal_path}")
    print(f"closeout={args.closeout}")


if __name__ == "__main__":
    main()
