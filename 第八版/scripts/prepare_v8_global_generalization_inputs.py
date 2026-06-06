"""Prepare v8 LST semantics and manifest files for global generalization work.

This script is intentionally non-destructive:

* raw ``lst_day_for_model.csv`` and ``lst_night_for_model.csv`` are not edited;
* v8 LST files are written as sidecars under an experiment output directory;
* manifests reference the sidecars while keeping the existing five-file
  standard-input contract intact.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CORE_FILES = (
    "profile_for_model.csv",
    "metadata.json",
    "era5_for_model.csv",
    "lst_day_for_model.csv",
    "lst_night_for_model.csv",
)

HYDROLOGY_COLUMNS = (
    "net_inflow_m3_s",
    "inflow_m3_s",
    "outflow_m3_s",
    "water_level_anomaly_m",
    "withdrawal_depth_m",
    "inflow_temperature_C",
)

ERA_KEY_COLUMNS = ("lmld_m", "lblt_C", "t2m_C", "ssrd_W_per_m2", "wind_norm_m_per_s")

SATELLITE_OBSERVED_PATTERNS = (
    "vnp21",
    "vj121",
    "mod11",
    "myd11",
    "mcd11",
    "modis",
    "viirs",
    "landsat",
    "ecostress",
    "sentinel",
    "slstr",
    "goes",
    "lswt",
)

SATELLITE_FILLED_PATTERNS = (
    "gadmlst",
    "trims",
    "tpdc",
    "continuous",
    "lstcont",
    "gapless",
    "gap-filled",
    "gap_filled",
    "all-weather",
    "all_weather",
    "reconstructed",
    "cloud_filled",
)

BANNED_LST_PATTERNS = ("era5", "in_situ", "proxy")

RESERVOIR_HINTS = (
    "reservoir",
    "carvins_cove",
    "falling_creek",
    "beaverdam",
    "rimov",
    "el_val",
    "alqueva",
    "caia",
    "odeleite",
    "santaclara",
    "santa_clara",
)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result
    except (TypeError, ValueError):
        return default


def _numeric_bool(series: pd.Series, default: float = np.nan) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        text = series.astype(str).str.strip().str.lower()
        values = text.map(
            {
                "true": 1.0,
                "t": 1.0,
                "yes": 1.0,
                "y": 1.0,
                "1": 1.0,
                "false": 0.0,
                "f": 0.0,
                "no": 0.0,
                "n": 0.0,
                "0": 0.0,
            }
        )
    return values.fillna(default)


def _first_numeric(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], default: float = np.nan) -> pd.Series:
    lower = {str(col).lower(): col for col in frame.columns}
    for candidate in columns:
        column = lower.get(str(candidate).lower())
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.isna().all():
            return values
    return pd.Series(default, index=frame.index, dtype="float64")


def _first_text(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.Series:
    parts = []
    lower = {str(col).lower(): col for col in frame.columns}
    for candidate in columns:
        column = lower.get(str(candidate).lower())
        if column is not None:
            parts.append(frame[column].astype(str))
    if not parts:
        return pd.Series("", index=frame.index, dtype="object")
    combined = parts[0].copy()
    for part in parts[1:]:
        combined = combined.str.cat(part, sep=" | ")
    return combined.fillna("")


def _source_contains(source_text: pd.Series, patterns: tuple[str, ...]) -> pd.Series:
    lowered = source_text.astype(str).str.lower()
    mask = pd.Series(False, index=source_text.index)
    for pattern in patterns:
        mask = mask | lowered.str.contains(re.escape(pattern), na=False)
    return mask


def _expected_days(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def _lake_year_from_id(lake_year: str) -> int | None:
    match = re.search(r"_(\d{4})$", lake_year)
    if not match:
        return None
    return int(match.group(1))


def _lake_group(lake_year: str) -> str:
    return re.sub(r"_\d{4}$", "", lake_year)


def _date_range(year: int) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")})


def _derive_lst_v8(frame: pd.DataFrame | None, year: int, mode: str) -> pd.DataFrame:
    base = _date_range(year)
    if frame is None or frame.empty or "Date" not in frame.columns:
        out = base.copy()
        out["LST_surface_C"] = np.nan
        out["LSWT_open_water_C"] = np.nan
        out["IST_snow_ice_C"] = np.nan
        out["LST_quality_factor"] = 0.0
        out["LST_is_filled"] = 0
        out["LST_observed_flag"] = 0
        out["ice_fraction"] = 0.0
        out["LST_source"] = ""
        out["LST_notes"] = "missing_lst_file"
        out["LST_day_or_night"] = mode
        return out

    raw = frame.copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    raw["Date"] = raw["Date"].dt.normalize()
    raw = base.merge(raw, on="Date", how="left")

    if "LST_surface_C" in raw.columns:
        lst_c = pd.to_numeric(raw["LST_surface_C"], errors="coerce")
    elif "LST_surface_K" in raw.columns:
        lst_c = pd.to_numeric(raw["LST_surface_K"], errors="coerce") - 273.15
    else:
        lst_c = pd.Series(np.nan, index=raw.index)

    source_text = _first_text(
        raw,
        [
            "LST_source",
            "LST_notes",
            "ContinuousLST_QA",
            "LST_daily_mean_observation_class",
        ],
    )
    source_lower = source_text.astype(str).str.lower()
    value_present = lst_c.notna()
    banned = _source_contains(source_text, BANNED_LST_PATTERNS)
    filled_source = _source_contains(source_text, SATELLITE_FILLED_PATTERNS)
    observed_source = _source_contains(source_text, SATELLITE_OBSERVED_PATTERNS)

    filled_raw = _first_numeric(raw, ["LST_is_filled", "is_filled", "filled"], default=np.nan)
    filled_flag = (
        ((filled_raw.fillna(0.0) > 0.5) | filled_source)
        & value_present
        & ~banned
    )

    original_day = _first_numeric(raw, ["MODIS_original_day_available"], default=np.nan)
    original_night = _first_numeric(raw, ["MODIS_original_night_available"], default=np.nan)
    if mode == "day":
        original_available = original_day.fillna(0.0) > 0.5
    elif mode == "night":
        original_available = original_night.fillna(0.0) > 0.5
    else:
        original_available = (original_day.fillna(0.0) > 0.5) | (original_night.fillna(0.0) > 0.5)

    explicit_observed = _first_numeric(raw, ["LST_observed_flag", "observed_flag"], default=np.nan)
    observed_flag = (
        (
            (explicit_observed.fillna(0.0) > 0.5)
            | original_available
            | (observed_source & ~filled_source)
            | (value_present & ~filled_flag & source_lower.str.strip().eq(""))
        )
        & value_present
        & ~filled_flag
        & ~banned
    )

    quality = _first_numeric(
        raw,
        [
            "LST_quality_factor",
            "LST_observation_weight",
            "LST_qc_good_fraction",
            "LST_valid_pixel_fraction",
            "valid_pixel_fraction",
        ],
        default=np.nan,
    )
    if quality.max(skipna=True) > 10.0:
        quality = quality / 100.0
    elif quality.max(skipna=True) > 1.5:
        quality = quality / 3.0
    quality = quality.clip(0.0, 1.0)
    default_quality = pd.Series(
        np.where(observed_flag, 1.0, np.where(filled_flag, 0.45, 0.0)),
        index=raw.index,
        dtype="float64",
    )
    quality = quality.fillna(default_quality)
    quality = pd.Series(quality, index=raw.index).astype(float)
    quality.loc[filled_flag] = np.minimum(quality.loc[filled_flag], 0.55)
    quality.loc[observed_flag] = np.maximum(quality.loc[observed_flag], 0.80)
    quality.loc[~value_present | banned] = 0.0

    ice_fraction = _first_numeric(
        raw,
        ["ice_fraction", "ice_cover_fraction", "lake_ice_fraction", "ice_cover", "ice_concentration"],
        default=0.0,
    )
    if ice_fraction.max(skipna=True) > 1.5:
        ice_fraction = ice_fraction / 100.0
    ice_fraction = ice_fraction.fillna(0.0).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "Date": raw["Date"].dt.strftime("%Y-%m-%d"),
            "LST_surface_C": lst_c,
            "LSWT_open_water_C": lst_c.where(observed_flag & (ice_fraction <= 0.05)),
            "IST_snow_ice_C": np.nan,
            "LST_quality_factor": quality,
            "LST_is_filled": filled_flag.astype(int),
            "LST_observed_flag": observed_flag.astype(int),
            "ice_fraction": ice_fraction,
            "LST_source": raw["LST_source"] if "LST_source" in raw.columns else "",
            "LST_notes": raw["LST_notes"] if "LST_notes" in raw.columns else "",
            "LST_day_or_night": mode,
        }
    )
    return out


def _profile_summary(profile: pd.DataFrame | None, year: int | None) -> dict[str, Any]:
    if profile is None or profile.empty or "Date" not in profile.columns:
        return {
            "profile_days": 0,
            "profile_rows": 0,
            "profile_min_depth_m": np.nan,
            "profile_max_depth_m": np.nan,
            "profile_median_layers_per_date": 0.0,
            "profile_is_partial": True,
            "profile_temperature_min_C": np.nan,
            "profile_temperature_max_C": np.nan,
        }
    df = profile.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if {"Depth_m", "Temperature_C"}.issubset(df.columns):
        depth = pd.to_numeric(df["Depth_m"], errors="coerce")
        temp = pd.to_numeric(df["Temperature_C"], errors="coerce")
        layers = df.assign(_depth=depth).dropna(subset=["_depth"]).groupby(df["Date"].dt.normalize())["_depth"].nunique()
    else:
        temp_cols = [c for c in df.columns if re.fullmatch(r"Temp_[0-9]+(?:\.[0-9]+)?m", str(c))]
        depths = [float(re.fullmatch(r"Temp_([0-9]+(?:\.[0-9]+)?)m", str(c)).group(1)) for c in temp_cols]
        depth = pd.Series(depths, dtype=float)
        temp = df[temp_cols].apply(pd.to_numeric, errors="coerce").stack() if temp_cols else pd.Series(dtype=float)
        layers = df[temp_cols].notna().sum(axis=1) if temp_cols else pd.Series(dtype=float)
        if "Date" in df.columns and temp_cols:
            layers.index = df["Date"].dt.normalize()
    days = int(df["Date"].dt.normalize().nunique())
    expected = _expected_days(year) if year else days
    return {
        "profile_days": days,
        "profile_rows": int(len(df)),
        "profile_min_depth_m": float(depth.min(skipna=True)) if len(depth) else np.nan,
        "profile_max_depth_m": float(depth.max(skipna=True)) if len(depth) else np.nan,
        "profile_median_layers_per_date": float(layers.median(skipna=True)) if len(layers) else 0.0,
        "profile_is_partial": bool(days < expected),
        "profile_temperature_min_C": float(temp.min(skipna=True)) if len(temp) else np.nan,
        "profile_temperature_max_C": float(temp.max(skipna=True)) if len(temp) else np.nan,
    }


def _era_summary(era: pd.DataFrame | None, year: int | None) -> dict[str, Any]:
    if era is None or era.empty or "Date" not in era.columns:
        return {
            "era_rows": 0,
            "era_unique_days": 0,
            "era_expected_days": _expected_days(year) if year else 0,
            "era_complete": False,
            "era_key_columns_complete": False,
        }
    df = era.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    days = int(df["Date"].dt.normalize().nunique())
    expected = _expected_days(year) if year else days
    key_ok = True
    for column in ERA_KEY_COLUMNS:
        if column not in df.columns or pd.to_numeric(df[column], errors="coerce").isna().any():
            key_ok = False
            break
    return {
        "era_rows": int(len(df)),
        "era_unique_days": days,
        "era_expected_days": expected,
        "era_complete": bool(days == expected and len(df) == expected),
        "era_key_columns_complete": bool(key_ok),
    }


def _metadata_value(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
    return None


def _climate_zone(lat: float, elevation_m: float | None) -> str:
    abs_lat = abs(lat)
    if elevation_m is not None and np.isfinite(elevation_m) and elevation_m >= 1500:
        return "alpine_or_high_altitude"
    if abs_lat < 23.5:
        return "tropical"
    if abs_lat >= 60:
        return "boreal_or_arctic"
    if abs_lat < 35:
        return "subtropical_or_warm_temperate"
    return "temperate"


def _is_reservoir(lake_year: str, meta: dict[str, Any]) -> bool:
    text = " ".join(
        [
            lake_year,
            str(_metadata_value(meta, "lake_name", "name", "lake_id") or ""),
            str(_metadata_value(meta, "lake_type", "lake_type_class", "reservoir_indicator") or ""),
        ]
    ).lower()
    if any(hint in text for hint in RESERVOIR_HINTS):
        return True
    indicator = _to_float(_metadata_value(meta, "reservoir_indicator"), default=0.0)
    return indicator > 0.5


def _hydrology_available(era: pd.DataFrame | None) -> bool:
    if era is None or era.empty:
        return False
    for column in HYDROLOGY_COLUMNS:
        if column in era.columns and pd.to_numeric(era[column], errors="coerce").notna().any():
            return True
    # Current model code also accepts shorter aliases.
    aliases = ("net_inflow", "inflow", "outflow", "water_level_anomaly")
    return any(column in era.columns and pd.to_numeric(era[column], errors="coerce").notna().any() for column in aliases)


def _lst_counts(v8: pd.DataFrame) -> dict[str, int]:
    surface = pd.to_numeric(v8["LST_surface_C"], errors="coerce")
    return {
        "observed_days": int(pd.to_numeric(v8["LST_observed_flag"], errors="coerce").fillna(0).gt(0.5).sum()),
        "filled_days": int(pd.to_numeric(v8["LST_is_filled"], errors="coerce").fillna(0).gt(0.5).sum()),
        "missing_days": int(surface.isna().sum()),
    }


def _score_row(row: dict[str, Any]) -> float:
    expected = max(int(row.get("profile_expected_days") or 1), 1)
    profile_score = min(1.0, float(row["profile_days"]) / expected)
    era_score = 1.0 if row["era_complete"] and row["era_key_columns_complete"] else 0.0
    lst_score = 0.5 * (1.0 - row["lst_day_missing_days"] / expected) + 0.5 * (
        1.0 - row["lst_night_missing_days"] / expected
    )
    meta_score = 1.0 if row["metadata_has_required_geometry"] else 0.5
    return round(100.0 * (0.60 * profile_score + 0.20 * era_score + 0.15 * lst_score + 0.05 * meta_score), 3)


def _recommended_split(row: dict[str, Any], rank: int, total: int) -> str:
    if row["profile_days"] < 10:
        return "stress_test"
    if row["profile_days"] < 50:
        return "stress_test"
    climate = row["climate_zone"]
    if climate in {"tropical", "boreal_or_arctic", "alpine_or_high_altitude"}:
        return "validation_climate_gap" if rank % 2 else "locked_climate_heldout_test"
    if rank >= int(total * 0.80):
        return "locked_lake_heldout_test"
    if rank >= int(total * 0.60):
        return "validation_lake_heldout"
    return "train_expansion"


def _manifest_lake(row: dict[str, Any], standard_root: Path, v8_dir: Path) -> dict[str, Any]:
    lake_year = row["lake_year"]
    year = _lake_year_from_id(lake_year)
    day_v8 = v8_dir / f"{lake_year}_lst_day_v8.csv"
    night_v8 = v8_dir / f"{lake_year}_lst_night_v8.csv"
    lake_dir = standard_root / lake_year
    return {
        "lake_id": lake_year,
        "lake_group": _lake_group(lake_year),
        "year": year,
        "era5": str((lake_dir / "era5_for_model.csv").as_posix()),
        "lst": str(night_v8.as_posix()),
        "lst_day": str(day_v8.as_posix()),
        "lst_night": str(night_v8.as_posix()),
        "profile_obs": str((lake_dir / "profile_for_model.csv").as_posix()),
        "metadata": str((lake_dir / "metadata.json").as_posix()),
        "max_depth": row["max_depth_m"],
        "lst_semantics_version": "v8",
        "has_v8_lst_flags": True,
        "hydrology_available": bool(row["hydrology_available"]),
        "reservoir_without_hydrology": bool(row["is_reservoir"] and not row["hydrology_available"]),
        "recommended_split": row["recommended_split"],
        "profile_is_partial": bool(row["profile_is_partial"]),
        "climate_zone": row["climate_zone"],
        "lst_day_observed_days": row["lst_day_observed_days"],
        "lst_night_observed_days": row["lst_night_observed_days"],
        "lst_day_filled_days": row["lst_day_filled_days"],
        "lst_night_filled_days": row["lst_night_filled_days"],
    }


def _build_candidate_lakes() -> pd.DataFrame:
    rows = [
        ("lake_hayes", "LAWA/NZ lake profile candidate", "New Zealand", "Oceania", -45.00, 168.78, "southern_temperate", 0, 0, 0, 0, 0, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_rotorua", "Waikato/BOP regional monitoring candidate", "New Zealand", "Oceania", -38.08, 176.27, "southern_temperate", 0, 0, 0, 0, 0, False, 79.8, 45.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_taupo", "NZ regional monitoring candidate", "New Zealand", "Oceania", -38.80, 175.90, "southern_temperate", 0, 0, 0, 0, 0, False, 616.0, 186.0, 97.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_alexandrina", "Australian water quality profile candidate", "Australia", "Oceania", -35.45, 139.10, "southern_arid_to_temperate", 0, 0, 1, 0, 0, False, 649.0, 6.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_titicaca", "global high-altitude profile candidate", "Peru/Bolivia", "South America", -15.80, -69.40, "tropical_high_altitude", 0, 1, 0, 0, 1, False, 8372.0, 281.0, 107.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_atitlan", "tropical highland profile candidate", "Guatemala", "North America", 14.69, -91.20, "tropical_high_altitude", 0, 1, 0, 0, 1, False, 130.0, 340.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_tana", "tropical highland profile candidate", "Ethiopia", "Africa", 12.02, 37.30, "tropical_highland", 0, 1, 0, 0, 1, False, 3200.0, 15.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_tanganyika", "African great lake profile candidate", "Tanzania/DRC/Burundi/Zambia", "Africa", -6.50, 29.60, "tropical_deep", 1, 1, 0, 0, 0, False, 32900.0, 1470.0, 570.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_victoria", "African great lake profile candidate", "Kenya/Tanzania/Uganda", "Africa", -1.00, 33.00, "tropical_large_shallow", 1, 1, 0, 0, 0, False, 68800.0, 84.0, 40.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_kivu", "African rift lake profile candidate", "Rwanda/DRC", "Africa", -1.70, 29.20, "tropical_deep", 1, 1, 0, 0, 1, False, 2370.0, 485.0, 240.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("toolik_lake", "Arctic LTER profile candidate", "USA", "North America", 68.63, -149.60, "arctic", 0, 0, 0, 1, 0, False, 1.5, 25.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_tornetrask", "subarctic profile candidate", "Sweden", "Europe", 68.42, 19.00, "subarctic", 0, 0, 0, 1, 0, False, 330.0, 168.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_kilpisjarvi", "subarctic alpine profile candidate", "Finland", "Europe", 69.05, 20.80, "subarctic_alpine", 0, 0, 0, 1, 1, False, 37.0, 57.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_thingvallavatn", "north Atlantic high-latitude candidate", "Iceland", "Europe", 64.18, -21.15, "subarctic_oceanic", 0, 0, 0, 1, 0, False, 84.0, 114.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_tahoe", "mountain oligotrophic profile candidate", "USA", "North America", 39.10, -120.03, "alpine_or_high_altitude", 0, 0, 0, 0, 1, False, 496.0, 501.0, 305.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_geneva", "deep alpine foreland profile candidate", "Switzerland/France", "Europe", 46.45, 6.53, "alpine_foreland", 0, 0, 0, 0, 1, False, 580.0, 310.0, 154.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_lunz", "alpine lake profile candidate", "Austria", "Europe", 47.85, 15.05, "alpine_or_high_altitude", 0, 0, 0, 0, 1, False, 0.68, 34.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_mead", "arid reservoir profile candidate", "USA", "North America", 36.25, -114.39, "arid_reservoir", 0, 0, 1, 0, 0, True, 640.0, 162.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, True, "candidate_profile_first_pending", "stress_test"),
        ("lake_powell", "arid reservoir profile candidate", "USA", "North America", 37.05, -111.30, "arid_reservoir", 0, 0, 1, 0, 0, True, 653.0, 170.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, True, "candidate_profile_first_pending", "stress_test"),
        ("lake_kinneret", "semi-arid lake profile candidate", "Israel", "Asia", 32.82, 35.59, "semi_arid_subtropical", 0, 0, 1, 0, 0, False, 166.0, 43.0, 25.6, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("alqueva_reservoir", "SNIRH Portuguese reservoir candidate", "Portugal", "Europe", 38.20, -7.50, "mediterranean_reservoir", 0, 0, 1, 0, 0, True, 250.0, 90.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, True, "blocked_by_snirh_export_access", "stress_test"),
        ("caia_reservoir", "SNIRH Portuguese reservoir candidate", "Portugal", "Europe", 39.00, -7.10, "mediterranean_reservoir", 0, 0, 1, 0, 0, True, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, True, "blocked_by_snirh_export_access", "stress_test"),
        ("odeleite_reservoir", "SNIRH Portuguese reservoir candidate", "Portugal", "Europe", 37.33, -7.53, "mediterranean_reservoir", 0, 0, 1, 0, 0, True, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, True, "blocked_by_snirh_export_access", "stress_test"),
        ("lake_washington", "US temperate urban lake profile candidate", "USA", "North America", 47.62, -122.25, "temperate", 0, 0, 0, 0, 0, False, 87.6, 65.2, 32.9, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_maggiore", "deep alpine foreland profile candidate", "Italy/Switzerland", "Europe", 45.83, 8.62, "alpine_foreland", 0, 0, 0, 0, 1, False, 212.0, 372.0, 177.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_lake_heldout_test"),
        ("lake_balaton", "large shallow temperate profile candidate", "Hungary", "Europe", 46.83, 17.73, "temperate_shallow", 0, 0, 0, 0, 0, False, 592.0, 12.2, 3.2, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
        ("lake_biwa", "humid subtropical monomictic lake candidate", "Japan", "Asia", 35.25, 136.05, "humid_subtropical", 0, 0, 0, 0, 0, False, 670.0, 104.0, 41.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "validation_climate_gap"),
        ("lake_qinghai", "high plateau saline lake candidate", "China", "Asia", 36.90, 100.20, "high_plateau_semi_arid", 0, 0, 1, 0, 1, False, 4317.0, 32.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "stress_test"),
        ("lake_baikal", "deep cold continental lake candidate", "Russia", "Asia", 53.50, 108.00, "boreal_or_arctic", 0, 0, 0, 1, 0, False, 31722.0, 1642.0, 744.0, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "locked_climate_heldout_test"),
        ("lake_vansjo", "boreal lowland lake candidate", "Norway", "Europe", 59.40, 10.75, "boreal_temperate", 0, 0, 0, 0, 0, False, 36.0, 41.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, "candidate_profile_first_pending", "train_expansion"),
    ]
    columns = [
        "lake_id",
        "source",
        "country",
        "continent",
        "lat",
        "lon",
        "climate_zone",
        "is_southern_hemisphere",
        "is_tropical",
        "is_arid",
        "is_boreal_or_arctic",
        "is_alpine_or_high_altitude",
        "is_reservoir",
        "area_km2",
        "max_depth_m",
        "mean_depth_m",
        "volume_m3",
        "secchi_m",
        "light_extinction_kd",
        "profile_dates",
        "profile_depth_max_m",
        "profile_depth_coverage_ratio",
        "lst_observed_fraction",
        "forcing_available",
        "hydrology_available",
        "qc_status",
        "recommended_split",
    ]
    normalized_rows = []
    for row in rows:
        if len(row) == len(columns) - 2:
            row = row[:21] + (np.nan, np.nan) + row[21:]
        if len(row) != len(columns):
            raise ValueError(f"Candidate row for {row[0]!r} has {len(row)} values; expected {len(columns)}")
        normalized_rows.append(row)
    return pd.DataFrame(normalized_rows, columns=columns)


def prepare(standard_root: Path, output_dir: Path, date_tag: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v8_dir = output_dir / "v8_lst"
    v8_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for lake_dir in sorted(path for path in standard_root.iterdir() if path.is_dir()):
        lake_year = lake_dir.name
        year = _lake_year_from_id(lake_year)
        if year is None:
            continue
        core_exists = {name: (lake_dir / name).exists() and (lake_dir / name).stat().st_size > 0 for name in CORE_FILES}

        day = _read_csv(lake_dir / "lst_day_for_model.csv")
        night = _read_csv(lake_dir / "lst_night_for_model.csv")
        era = _read_csv(lake_dir / "era5_for_model.csv")
        profile = _read_csv(lake_dir / "profile_for_model.csv")
        meta = _read_metadata(lake_dir / "metadata.json")

        day_v8 = _derive_lst_v8(day, year, "day")
        night_v8 = _derive_lst_v8(night, year, "night")
        day_v8_path = v8_dir / f"{lake_year}_lst_day_v8.csv"
        night_v8_path = v8_dir / f"{lake_year}_lst_night_v8.csv"
        day_v8.to_csv(day_v8_path, index=False, encoding="utf-8-sig")
        night_v8.to_csv(night_v8_path, index=False, encoding="utf-8-sig")

        for mode, frame in (("day", day_v8), ("night", night_v8)):
            source_text = (frame["LST_source"].astype(str) + " | " + frame["LST_notes"].astype(str)).str.lower()
            bad_mask = _source_contains(source_text, BANNED_LST_PATTERNS)
            if bad_mask.any():
                violations.append(
                    {
                        "lake_year": lake_year,
                        "mode": mode,
                        "bad_rows": int(bad_mask.sum()),
                        "bad_sources": "; ".join(sorted(frame.loc[bad_mask, "LST_source"].dropna().astype(str).unique())[:10]),
                    }
                )

        profile_stats = _profile_summary(profile, year)
        era_stats = _era_summary(era, year)
        day_counts = _lst_counts(day_v8)
        night_counts = _lst_counts(night_v8)

        lat = _to_float(_metadata_value(meta, "latitude", "lat"), default=np.nan)
        lon = _to_float(_metadata_value(meta, "longitude", "lon"), default=np.nan)
        max_depth = _to_float(_metadata_value(meta, "max_depth_m", "max_depth"), default=np.nan)
        mean_depth = _to_float(_metadata_value(meta, "mean_depth_m", "mean_depth"), default=np.nan)
        area = _to_float(_metadata_value(meta, "area_km2", "surface_area_km2"), default=np.nan)
        elevation = _to_float(_metadata_value(meta, "elevation_m", "elevation"), default=np.nan)
        metadata_has_required_geometry = all(np.isfinite(v) for v in (lat, lon, max_depth, mean_depth, area))
        hydrology_available = _hydrology_available(era)
        is_reservoir = _is_reservoir(lake_year, meta)
        climate = _climate_zone(lat if np.isfinite(lat) else 0.0, elevation)

        row: dict[str, Any] = {
            "lake_year": lake_year,
            "year": year,
            "lake_group": _lake_group(lake_year),
            "core_files_present": all(core_exists.values()),
            **{f"has_{name}": exists for name, exists in core_exists.items()},
            **profile_stats,
            "profile_expected_days": _expected_days(year),
            **era_stats,
            "lst_day_observed_days": day_counts["observed_days"],
            "lst_day_filled_days": day_counts["filled_days"],
            "lst_day_missing_days": day_counts["missing_days"],
            "lst_night_observed_days": night_counts["observed_days"],
            "lst_night_filled_days": night_counts["filled_days"],
            "lst_night_missing_days": night_counts["missing_days"],
            "lst_day_v8_path": str(day_v8_path),
            "lst_night_v8_path": str(night_v8_path),
            "lst_semantics_version": "v8",
            "has_v8_lst_flags": True,
            "hydrology_available": hydrology_available,
            "hydrology_available_columns": ",".join([c for c in HYDROLOGY_COLUMNS if era is not None and c in era.columns]),
            "is_reservoir": is_reservoir,
            "reservoir_without_hydrology": bool(is_reservoir and not hydrology_available),
            "latitude": lat,
            "longitude": lon,
            "area_km2": area,
            "max_depth_m": max_depth,
            "mean_depth_m": mean_depth,
            "elevation_m": elevation,
            "metadata_has_required_geometry": metadata_has_required_geometry,
            "climate_zone": climate,
        }
        row["v8_input_score_100"] = _score_row(row)
        rows.append(row)

    audit = pd.DataFrame(rows).sort_values(["v8_input_score_100", "lake_year"], ascending=[False, True])
    if not audit.empty:
        total = len(audit)
        audit["recommended_split"] = [
            _recommended_split(row._asdict(), idx, total)
            for idx, row in enumerate(audit.itertuples(index=False), start=0)
        ]
        usable = audit[
            (audit["core_files_present"])
            & (audit["era_rows"] > 0)
            & (audit["profile_days"] >= 50)
            & (audit["lst_day_missing_days"] < audit["profile_expected_days"])
            & (audit["lst_night_missing_days"] < audit["profile_expected_days"])
        ].copy()
    else:
        usable = audit.copy()

    clean = usable[
        (usable["era_complete"])
        & (usable["era_key_columns_complete"])
        & (usable["profile_days"] >= 100)
        & (usable["metadata_has_required_geometry"])
    ].copy()
    clean = clean.head(34)
    all_usable = usable.head(39)

    audit_path = output_dir / f"standard_inputs_v8_audit_{date_tag}.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    hydrology_path = output_dir / f"hydrology_schema_summary_{date_tag}.csv"
    audit[
        [
            "lake_year",
            "is_reservoir",
            "hydrology_available",
            "reservoir_without_hydrology",
            "hydrology_available_columns",
            "recommended_split",
        ]
    ].to_csv(hydrology_path, index=False, encoding="utf-8-sig")

    violations_df = pd.DataFrame(violations, columns=["lake_year", "mode", "bad_rows", "bad_sources"])
    violations_path = output_dir / f"lst_source_violations_{date_tag}.csv"
    violations_df.to_csv(violations_path, index=False, encoding="utf-8-sig")

    candidate_path = output_dir / "candidate_lakes_global.csv"
    _build_candidate_lakes().to_csv(candidate_path, index=False, encoding="utf-8-sig")

    def write_manifest(frame: pd.DataFrame, manifest_id: str, filename: str) -> Path:
        lakes = [_manifest_lake(row._asdict(), standard_root, v8_dir) for row in frame.itertuples(index=False)]
        manifest = {
            "manifest_id": manifest_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "standard_root": str(standard_root),
            "split_mode": "time_blocked",
            "profile_supervision_scope": "all",
            "lst_semantics_version": "v8",
            "has_v8_lst_flags": True,
            "hydrology_schema_version": "v1",
            "hydrology_columns": list(HYDROLOGY_COLUMNS),
            "selection_rule": (
                "profile-first; core files present; LST v8 sidecars generated; "
                "clean uses complete ERA5, non-empty day/night LST, geometry metadata, profile_days>=100; "
                "all uses top usable lake-years by v8_input_score_100."
            ),
            "lakes": lakes,
        }
        path = output_dir / filename
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    clean_manifest = write_manifest(clean, "M2_local_34core_clean", "M2_local_34core_clean.json")
    all_manifest = write_manifest(all_usable, "M2_local_39usable_all", "M2_local_39usable_all.json")

    summary = {
        "audit_path": str(audit_path),
        "hydrology_path": str(hydrology_path),
        "violations_path": str(violations_path),
        "candidate_path": str(candidate_path),
        "clean_manifest": str(clean_manifest),
        "all_manifest": str(all_manifest),
        "lake_years_scanned": int(len(audit)),
        "clean_manifest_lakes": int(len(clean)),
        "all_manifest_lakes": int(len(all_usable)),
        "lst_source_violation_lake_modes": int(len(violations_df)),
        "era_incomplete_lake_years": audit.loc[~audit["era_complete"], "lake_year"].tolist() if not audit.empty else [],
    }
    summary_path = output_dir / f"v8_preparation_summary_{date_tag}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", default=str(Path("data") / "_standard_inputs"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / f"M2_v8_global_generalization_{args.date_tag}"
    summary = prepare(Path(args.standard_root), output_dir, args.date_tag)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
