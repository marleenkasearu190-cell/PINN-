"""Build a LakePINN lake-year data completeness and type inventory.

The inventory separates forcing completeness, LST completeness, profile
supervision density, metadata completeness, and physical lake type. It is meant
for selecting small-sample transfer experiments without mixing together very
different data-quality cases.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ERA_GROUPS = {
    "air_temperature": ("t2m_C", "t2m_K"),
    "wind": ("wind_norm_m_per_s", "u10_m_per_s", "v10_m_per_s"),
    "humidity": ("rh_percent", "specific_humidity_kg_per_kg", "d2m_C", "d2m_K"),
    "pressure": ("sp_Pa", "sp_hPa"),
    "shortwave": ("ssrd_W_per_m2", "ssrd_J_per_m2"),
    "longwave": ("strd_W_per_m2", "strd_J_per_m2"),
}

ERA_OPTIONAL_GROUPS = {
    "latent_heat": ("latent_heat_upward_W_per_m2", "latent_heat_upward_J_per_m2"),
    "sensible_heat": ("sensible_heat_upward_W_per_m2", "sensible_heat_upward_J_per_m2"),
    "boundary_layer": ("lmld_m", "lblt_C", "lblt_K"),
}

LST_TEMP_COLUMNS = (
    "LST_surface_C",
    "LST_night_C",
    "LST_day_C",
    "LST_surface_K",
    "LSWT_open_water_C",
    "IST_snow_ice_C",
    "LST_filled_C",
    "LST_observed_raw_C",
)

LST_OBSERVED_FLAG_COLUMNS = (
    "LST_daily_mean_has_any_original_observation",
    "LST_daily_mean_is_fully_observed",
    "MODIS_original_night_available",
    "MODIS_original_day_available",
)

STANDARD_INPUT_FILES = (
    "era5_for_model.csv",
    "lst_night_for_model.csv",
    "lst_day_for_model.csv",
    "profile_for_model.csv",
    "metadata.json",
)

METADATA_REQUIRED_FIELDS = ("latitude", "longitude", "area_km2", "max_depth_m", "mean_depth_m")
METADATA_EXTENDED_FIELDS = (
    "elevation_m",
    "fetch_m",
    "effective_fetch_m",
    "basin_shape_factor",
    "shoreline_development",
    "reservoir_indicator",
    "light_extinction_kd",
    "secchi_m",
)

COMPLETENESS_WEIGHTS = {
    "profile": 0.60,
    "era": 0.20,
    "lst": 0.15,
    "metadata": 0.05,
}

SUMMARY_LABELS = {
    "Q5_complete_supervised": "Q5 完整监督数据：ERA/LST/metadata 完整，剖面也完整，可用于正式训练和 heldout 评估",
    "Q4_good_supervised": "Q4 良好监督数据：总体可训练，但某一项完整度略弱，需要在结果解释中说明",
    "Q3_few_shot_profile": "Q3 少样本剖面数据：ERA/LST/metadata 可用，但剖面少，适合 few-shot 迁移",
    "Q2_reconstruction_ready_no_profile": "Q2 可重建但无剖面：可做无剖面重建外推，不能计算 profile RMSE",
    "Q1_partial_inputs": "Q1 输入部分缺失：有关键输入不完整，进入主实验前应先修数据",
    "Q0_core_incomplete": "Q0 核心输入缺失：ERA/LST/metadata 等核心数据缺失，暂不用于正式训练",
    "formal_training_and_heldout_evaluation": "正式训练和 heldout 评估",
    "supervised_training_or_validation_with_quality_note": "可用于监督训练或验证，但需要注明数据质量限制",
    "few_shot_adaptation_or_sparse_supervision": "少样本适应或稀疏剖面监督",
    "reconstruction_only_no_profile_rmse": "仅用于无剖面重建展示，不能计算剖面 RMSE",
    "repair_inputs_before_main_training": "先修复输入数据，再进入主训练集",
    "exclude_until_core_inputs_fixed": "核心输入修复前排除",
    "A_full_supervised": "旧版 A：完整监督训练数据",
    "B_few_shot_profile": "旧版 B：少样本剖面数据",
    "C_no_profile_reconstruction_candidate": "旧版 C：无剖面重建候选",
    "D_partial_or_sparse_inputs": "旧版 D：输入部分缺失或稀疏",
    "D_incomplete_core_inputs": "旧版 D：核心输入缺失",
    "complete": "完整",
    "partial": "部分完整",
    "sparse": "稀疏",
    "weak": "较弱",
    "missing": "缺失",
    "dense": "密集",
    "moderate": "中等",
    "very_sparse": "极少",
    "temperate": "温带",
    "cold_high_latitude": "高纬寒冷区",
    "warm_subtropical": "暖温带/亚热带",
    "tropical": "热带",
    "unknown": "未知",
    "shallow": "浅水",
    "medium_depth": "中等深度",
    "deep": "深水",
    "small": "小面积",
    "medium_area": "中等面积",
    "large_area": "大面积",
    "natural_lake": "天然湖",
    "reservoir": "水库",
}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clip01(value: Any) -> float:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return 0.0
    return float(min(1.0, max(0.0, numeric)))


def _date_stats(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or "Date" not in frame.columns or frame.empty:
        return {
            "start_date": "",
            "end_date": "",
            "unique_days": 0,
            "span_days": 0,
            "date_coverage_fraction": 0.0,
        }
    dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
    if dates.empty:
        return {
            "start_date": "",
            "end_date": "",
            "unique_days": 0,
            "span_days": 0,
            "date_coverage_fraction": 0.0,
        }
    normalized = dates.dt.normalize()
    unique_days = int(normalized.nunique())
    start = normalized.min()
    end = normalized.max()
    span_days = int((end - start).days) + 1
    coverage = unique_days / span_days if span_days > 0 else 0.0
    return {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "unique_days": unique_days,
        "span_days": span_days,
        "date_coverage_fraction": coverage,
    }


def _non_null_fraction(frame: pd.DataFrame, columns: list[str]) -> float:
    present = [col for col in columns if col in frame.columns]
    if not present or frame.empty:
        return 0.0
    return float(frame[present].notna().any(axis=1).mean())


def _group_presence(frame: pd.DataFrame | None, groups: dict[str, tuple[str, ...]]) -> dict[str, bool]:
    if frame is None:
        return {name: False for name in groups}
    columns = set(frame.columns)
    return {name: any(column in columns for column in options) for name, options in groups.items()}


def _era_summary(lake_dir: Path) -> dict[str, Any]:
    frame = _read_csv(lake_dir / "era5_for_model.csv")
    stats = _date_stats(frame)
    row_count = 0 if frame is None else int(len(frame))
    required_presence = _group_presence(frame, ERA_GROUPS)
    optional_presence = _group_presence(frame, ERA_OPTIONAL_GROUPS)
    required_present_count = sum(required_presence.values())
    optional_present_count = sum(optional_presence.values())
    required_fraction = required_present_count / len(ERA_GROUPS)
    daily_complete = 0.0
    if frame is not None and not frame.empty:
        group_valid = []
        for options in ERA_GROUPS.values():
            present = [column for column in options if column in frame.columns]
            if present:
                group_valid.append(frame[present].notna().any(axis=1))
            else:
                group_valid.append(pd.Series(False, index=frame.index))
        daily_complete = float(pd.concat(group_valid, axis=1).all(axis=1).mean())
    if frame is None:
        grade = "missing"
    elif stats["unique_days"] >= 300 and required_fraction >= 0.95 and daily_complete >= 0.90:
        grade = "complete"
    elif stats["unique_days"] >= 180 and required_fraction >= 0.80 and daily_complete >= 0.70:
        grade = "partial"
    elif stats["unique_days"] >= 30 and required_fraction >= 0.50:
        grade = "sparse"
    else:
        grade = "weak"
    result: dict[str, Any] = {
        "era_file_exists": frame is not None,
        "era_rows": row_count,
        "era_start_date": stats["start_date"],
        "era_end_date": stats["end_date"],
        "era_unique_days": stats["unique_days"],
        "era_span_days": stats["span_days"],
        "era_date_coverage_fraction": stats["date_coverage_fraction"],
        "era_required_group_fraction": required_fraction,
        "era_required_daily_complete_fraction": daily_complete,
        "era_optional_group_fraction": optional_present_count / len(ERA_OPTIONAL_GROUPS),
        "era_grade": grade,
    }
    for name, present in required_presence.items():
        result[f"era_has_{name}"] = present
    for name, present in optional_presence.items():
        result[f"era_has_optional_{name}"] = present
    return result


def _lst_summary(lake_dir: Path) -> dict[str, Any]:
    night = _read_csv(lake_dir / "lst_night_for_model.csv")
    day = _read_csv(lake_dir / "lst_day_for_model.csv")
    frame = night if night is not None else day
    source = "night" if night is not None else ("day" if day is not None else "")
    stats = _date_stats(frame)
    row_count = 0 if frame is None else int(len(frame))
    temp_fraction = 0.0
    observed_fraction = 0.0
    reconstructed_fraction = 0.0
    if frame is not None and not frame.empty:
        temp_fraction = _non_null_fraction(frame, list(LST_TEMP_COLUMNS))
        flag_present = [column for column in LST_OBSERVED_FLAG_COLUMNS if column in frame.columns]
        if flag_present:
            observed_fraction = float(frame[flag_present].fillna(False).astype(bool).any(axis=1).mean())
        elif "LST_is_filled" in frame.columns:
            observed_fraction = float((~frame["LST_is_filled"].fillna(True).astype(bool)).mean())
        else:
            observed_fraction = temp_fraction
        if "LST_is_filled" in frame.columns:
            reconstructed_fraction = float(frame["LST_is_filled"].fillna(False).astype(bool).mean())
        elif "LST_daily_mean_has_reconstructed_component" in frame.columns:
            reconstructed_fraction = float(
                frame["LST_daily_mean_has_reconstructed_component"].fillna(False).astype(bool).mean()
            )
    if frame is None:
        grade = "missing"
    elif stats["unique_days"] >= 300 and temp_fraction >= 0.90:
        grade = "complete"
    elif stats["unique_days"] >= 180 and temp_fraction >= 0.60:
        grade = "partial"
    elif stats["unique_days"] >= 30 and temp_fraction > 0.0:
        grade = "sparse"
    else:
        grade = "weak"
    if grade == "missing":
        support = "missing"
    elif observed_fraction >= 0.70:
        support = "observed_dominant"
    elif temp_fraction >= 0.70:
        support = "reconstructed_dominant"
    else:
        support = "limited"
    return {
        "lst_file_source": source,
        "lst_file_exists": frame is not None,
        "lst_rows": row_count,
        "lst_start_date": stats["start_date"],
        "lst_end_date": stats["end_date"],
        "lst_unique_days": stats["unique_days"],
        "lst_span_days": stats["span_days"],
        "lst_date_coverage_fraction": stats["date_coverage_fraction"],
        "lst_temperature_valid_fraction": temp_fraction,
        "lst_original_observation_fraction": observed_fraction,
        "lst_reconstructed_fraction": reconstructed_fraction,
        "lst_grade": grade,
        "lst_support_type": support,
    }


def _profile_summary(lake_dir: Path) -> dict[str, Any]:
    frame = _read_csv(lake_dir / "profile_for_model.csv")
    stats = _date_stats(frame)
    row_count = 0 if frame is None else int(len(frame))
    valid = pd.DataFrame()
    if frame is not None and {"Date", "Depth_m", "Temperature_C"}.issubset(frame.columns):
        valid = frame.dropna(subset=["Date", "Depth_m", "Temperature_C"]).copy()
    temperature_valid_fraction = len(valid) / row_count if row_count > 0 else 0.0
    profile_count = int(valid["Date"].nunique()) if not valid.empty else 0
    unique_depth_count = int(valid["Depth_m"].nunique()) if not valid.empty else 0
    min_depth = float(valid["Depth_m"].min()) if not valid.empty else float("nan")
    max_observed_depth = float(valid["Depth_m"].max()) if not valid.empty else float("nan")
    layers_per_profile = valid.groupby("Date")["Depth_m"].nunique() if not valid.empty else pd.Series(dtype=float)
    median_layers = float(layers_per_profile.median()) if not layers_per_profile.empty else 0.0
    mean_layers = float(layers_per_profile.mean()) if not layers_per_profile.empty else 0.0
    if profile_count >= 40 and median_layers >= 6:
        grade = "dense"
    elif profile_count >= 20 and median_layers >= 4:
        grade = "moderate"
    elif profile_count >= 5:
        grade = "sparse"
    elif profile_count >= 1:
        grade = "very_sparse"
    else:
        grade = "missing"
    if grade in {"dense", "moderate"}:
        supervision = "strong_profile_supervision"
    elif grade in {"sparse", "very_sparse"}:
        supervision = "few_shot_profile_supervision"
    else:
        supervision = "no_profile_supervision"
    return {
        "profile_file_exists": frame is not None,
        "profile_rows": row_count,
        "profile_start_date": stats["start_date"],
        "profile_end_date": stats["end_date"],
        "profile_unique_days_raw": stats["unique_days"],
        "profile_count": profile_count,
        "profile_unique_depth_count": unique_depth_count,
        "profile_min_depth_m": min_depth,
        "profile_max_observed_depth_m": max_observed_depth,
        "profile_median_layers_per_date": median_layers,
        "profile_mean_layers_per_date": mean_layers,
        "profile_temperature_valid_fraction": temperature_valid_fraction,
        "profile_grade": grade,
        "profile_supervision_type": supervision,
    }


def _metadata_summary(lake_dir: Path) -> dict[str, Any]:
    metadata = _read_metadata(lake_dir / "metadata.json")
    required_present = {field: pd.notna(_to_float(metadata.get(field))) for field in METADATA_REQUIRED_FIELDS}
    extended_present = {field: metadata.get(field) is not None for field in METADATA_EXTENDED_FIELDS}
    required_fraction = sum(required_present.values()) / len(METADATA_REQUIRED_FIELDS)
    extended_fraction = sum(extended_present.values()) / len(METADATA_EXTENDED_FIELDS)
    if not metadata:
        grade = "missing"
    elif required_fraction >= 1.0 and extended_fraction >= 0.60:
        grade = "complete"
    elif required_fraction >= 0.80:
        grade = "usable"
    elif required_fraction >= 0.40:
        grade = "weak"
    else:
        grade = "missing"
    latitude = _to_float(metadata.get("latitude"))
    max_depth = _to_float(metadata.get("max_depth_m"))
    mean_depth = _to_float(metadata.get("mean_depth_m"))
    area = _to_float(metadata.get("area_km2"))
    elevation = _to_float(metadata.get("elevation_m"))
    kd = _to_float(metadata.get("light_extinction_kd"))
    secchi = _to_float(metadata.get("secchi_m"))
    fetch = _to_float(metadata.get("effective_fetch_m", metadata.get("fetch_m")))
    basin_shape = _to_float(metadata.get("basin_shape_factor"))
    depth_ratio = mean_depth / max_depth if max_depth > 0 and np.isfinite(mean_depth) else float("nan")
    return {
        "metadata_file_exists": bool(metadata),
        "metadata_grade": grade,
        "metadata_required_fraction": required_fraction,
        "metadata_extended_fraction": extended_fraction,
        "lake_name": metadata.get("lake_name", ""),
        "lake_group": metadata.get("lake_id", ""),
        "lake_type_raw": metadata.get("lake_type", ""),
        "latitude": latitude,
        "longitude": _to_float(metadata.get("longitude")),
        "area_km2": area,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
        "depth_ratio_mean_to_max": depth_ratio,
        "elevation_m": elevation,
        "light_extinction_kd": kd,
        "secchi_m": secchi,
        "effective_fetch_m": fetch,
        "basin_shape_factor": basin_shape,
        "shoreline_development": _to_float(metadata.get("shoreline_development")),
        "reservoir_indicator": _to_float(metadata.get("reservoir_indicator")),
    }


def _physical_classes(row: dict[str, Any]) -> dict[str, str]:
    lat = row.get("latitude", float("nan"))
    max_depth = row.get("max_depth_m", float("nan"))
    area = row.get("area_km2", float("nan"))
    elevation = row.get("elevation_m", float("nan"))
    depth_ratio = row.get("depth_ratio_mean_to_max", float("nan"))
    kd = row.get("light_extinction_kd", float("nan"))
    secchi = row.get("secchi_m", float("nan"))
    reservoir = row.get("reservoir_indicator", float("nan"))

    abs_lat = abs(lat) if np.isfinite(lat) else float("nan")
    if not np.isfinite(abs_lat):
        climate_zone = "unknown"
    elif abs_lat < 23.5:
        climate_zone = "tropical"
    elif abs_lat < 35.0:
        climate_zone = "warm_subtropical"
    elif abs_lat < 50.0:
        climate_zone = "temperate"
    else:
        climate_zone = "cold_high_latitude"

    if not np.isfinite(max_depth):
        depth_class = "unknown"
    elif max_depth < 10.0:
        depth_class = "shallow"
    elif max_depth < 30.0:
        depth_class = "medium_depth"
    else:
        depth_class = "deep"

    if not np.isfinite(area):
        area_class = "unknown"
    elif area < 1.0:
        area_class = "small"
    elif area < 10.0:
        area_class = "medium_area"
    else:
        area_class = "large_area"

    if not np.isfinite(elevation):
        elevation_class = "unknown"
    elif elevation < 500.0:
        elevation_class = "lowland"
    elif elevation < 1500.0:
        elevation_class = "upland"
    else:
        elevation_class = "highland"

    if not np.isfinite(depth_ratio):
        morphology_class = "unknown"
    elif depth_ratio >= 0.55:
        morphology_class = "broad_shallow_basin"
    elif depth_ratio >= 0.25:
        morphology_class = "bowl_shaped"
    else:
        morphology_class = "deep_basin"

    if np.isfinite(kd):
        if kd < 0.5:
            transparency_class = "clear"
        elif kd < 1.5:
            transparency_class = "moderate"
        else:
            transparency_class = "turbid"
    elif np.isfinite(secchi):
        if secchi > 4.0:
            transparency_class = "clear"
        elif secchi >= 1.5:
            transparency_class = "moderate"
        else:
            transparency_class = "turbid"
    else:
        transparency_class = "unknown"

    if not np.isfinite(reservoir):
        lake_type_class = "unknown"
    elif reservoir >= 0.5:
        lake_type_class = "reservoir"
    else:
        lake_type_class = "natural_lake"

    return {
        "climate_zone": climate_zone,
        "depth_class": depth_class,
        "area_class": area_class,
        "elevation_class": elevation_class,
        "morphology_class": morphology_class,
        "transparency_class": transparency_class,
        "lake_type_class": lake_type_class,
    }


def _usable_grade(row: dict[str, Any]) -> str:
    era = row["era_grade"]
    lst = row["lst_grade"]
    profile = row["profile_grade"]
    metadata = row["metadata_grade"]
    forcing_ok = era in {"complete", "partial"}
    lst_ok = lst in {"complete", "partial"}
    metadata_ok = metadata in {"complete", "usable"}
    if forcing_ok and lst_ok and metadata_ok and profile in {"dense", "moderate"}:
        return "A_full_supervised"
    if forcing_ok and lst_ok and metadata_ok and profile in {"sparse", "very_sparse"}:
        return "B_few_shot_profile"
    if forcing_ok and lst_ok and metadata_ok and profile == "missing":
        return "C_no_profile_reconstruction_candidate"
    if era in {"missing", "weak"} or metadata in {"missing", "weak"}:
        return "D_incomplete_core_inputs"
    return "D_partial_or_sparse_inputs"


def _era_completeness_score(row: dict[str, Any]) -> float:
    score = (
        0.45 * _clip01(row.get("era_date_coverage_fraction"))
        + 0.40 * _clip01(row.get("era_required_daily_complete_fraction"))
        + 0.15 * _clip01(row.get("era_required_group_fraction"))
    )
    return round(100.0 * score, 3)


def _lst_completeness_score(row: dict[str, Any]) -> float:
    score = (
        0.50 * _clip01(row.get("lst_temperature_valid_fraction"))
        + 0.30 * _clip01(row.get("lst_date_coverage_fraction"))
        + 0.20 * _clip01(row.get("lst_original_observation_fraction"))
    )
    return round(100.0 * score, 3)


def _profile_completeness_score(row: dict[str, Any]) -> float:
    max_observed_depth = _to_float(row.get("profile_max_observed_depth_m"))
    max_depth = _to_float(row.get("max_depth_m"))
    if np.isfinite(max_depth) and max_depth > 0.0 and np.isfinite(max_observed_depth):
        depth_coverage = max_observed_depth / max_depth
    else:
        depth_coverage = 0.0
    score = (
        0.40 * _clip01(_to_float(row.get("profile_count")) / 40.0)
        + 0.25 * _clip01(_to_float(row.get("profile_median_layers_per_date")) / 8.0)
        + 0.25 * _clip01(depth_coverage)
        + 0.10 * _clip01(row.get("profile_temperature_valid_fraction"))
    )
    return round(100.0 * score, 3)


def _metadata_completeness_score(row: dict[str, Any]) -> float:
    score = 0.75 * _clip01(row.get("metadata_required_fraction")) + 0.25 * _clip01(
        row.get("metadata_extended_fraction")
    )
    return round(100.0 * score, 3)


def _data_completeness_class(row: dict[str, Any]) -> str:
    era = row["era_grade"]
    lst = row["lst_grade"]
    profile = row["profile_grade"]
    metadata = row["metadata_grade"]
    score = _to_float(row["data_completeness_score_100"])
    core_ready = era in {"complete", "partial"} and lst in {"complete", "partial"} and metadata in {
        "complete",
        "usable",
    }
    if era in {"missing", "weak"} or lst in {"missing", "weak"} or metadata in {"missing", "weak"}:
        return "Q0_core_incomplete"
    if not core_ready:
        return "Q1_partial_inputs"
    if profile == "missing":
        return "Q2_reconstruction_ready_no_profile"
    if profile in {"sparse", "very_sparse"}:
        return "Q3_few_shot_profile"
    if (
        score >= 90.0
        and era == "complete"
        and lst == "complete"
        and metadata == "complete"
        and profile == "dense"
    ):
        return "Q5_complete_supervised"
    return "Q4_good_supervised"


def _recommended_use_case(data_class: str) -> str:
    return {
        "Q5_complete_supervised": "formal_training_and_heldout_evaluation",
        "Q4_good_supervised": "supervised_training_or_validation_with_quality_note",
        "Q3_few_shot_profile": "few_shot_adaptation_or_sparse_supervision",
        "Q2_reconstruction_ready_no_profile": "reconstruction_only_no_profile_rmse",
        "Q1_partial_inputs": "repair_inputs_before_main_training",
        "Q0_core_incomplete": "exclude_until_core_inputs_fixed",
    }[data_class]


def _inventory_row(lake_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"lake_year_id": lake_dir.name, "lake_year_dir": str(lake_dir)}
    row.update(_era_summary(lake_dir))
    row.update(_lst_summary(lake_dir))
    row.update(_profile_summary(lake_dir))
    row.update(_metadata_summary(lake_dir))
    row.update(_physical_classes(row))
    row["era_completeness_score"] = _era_completeness_score(row)
    row["lst_completeness_score"] = _lst_completeness_score(row)
    row["profile_completeness_score"] = _profile_completeness_score(row)
    row["metadata_completeness_score"] = _metadata_completeness_score(row)
    row["data_completeness_score_100"] = round(
        COMPLETENESS_WEIGHTS["profile"] * row["profile_completeness_score"]
        + COMPLETENESS_WEIGHTS["era"] * row["era_completeness_score"]
        + COMPLETENESS_WEIGHTS["lst"] * row["lst_completeness_score"]
        + COMPLETENESS_WEIGHTS["metadata"] * row["metadata_completeness_score"],
        3,
    )
    row["data_completeness_class"] = _data_completeness_class(row)
    row["recommended_use_case"] = _recommended_use_case(row["data_completeness_class"])
    row["usable_grade"] = _usable_grade(row)
    row["data_completeness_signature"] = (
        f"CLASS={row['data_completeness_class']};SCORE={row['data_completeness_score_100']:.1f};"
        f"ERA={row['era_grade']};LST={row['lst_grade']};PROFILE={row['profile_grade']};"
        f"METADATA={row['metadata_grade']}"
    )
    return row


def build_inventory(data_root: Path, *, include_empty_dirs: bool = False) -> pd.DataFrame:
    rows = []
    for lake_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if lake_dir.name.startswith(".") or lake_dir.name == "all_heatmaps":
            continue
        if not include_empty_dirs and not any((lake_dir / filename).exists() for filename in STANDARD_INPUT_FILES):
            continue
        rows.append(_inventory_row(lake_dir))
    return pd.DataFrame(rows)


def _write_summary(inventory: pd.DataFrame, output_dir: Path, date_tag: str) -> Path:
    summary_path = output_dir / f"lake_year_inventory_summary_{date_tag}.md"

    def label(value: Any) -> str:
        text = str(value)
        return SUMMARY_LABELS.get(text, text)

    def value_counts_block(series: pd.Series) -> str:
        counts = series.value_counts(dropna=False)
        return "\n".join(f"- {label(index)}：{int(value)}" for index, value in counts.items())

    physical_counts = (
        inventory[["climate_zone", "depth_class", "area_class", "lake_type_class"]]
        .value_counts(dropna=False)
        .reset_index(name="count")
    )
    physical_lines = [
        "- "
        + ", ".join(
            [
                f"气候={label(row.climate_zone)}",
                f"深度={label(row.depth_class)}",
                f"面积={label(row.area_class)}",
                f"类型={label(row.lake_type_class)}",
                f"数量={int(row.count)}",
            ]
        )
        for row in physical_counts.itertuples(index=False)
    ]
    lines = [
        "# LakePINN 数据完整度清单摘要",
        "",
        f"生成日期标签：{date_tag}",
        f"lake-year 数量：{len(inventory)}",
        "",
        "## 数据完整度主分类",
        value_counts_block(inventory["data_completeness_class"]),
        "",
        "## 推荐用途",
        value_counts_block(inventory["recommended_use_case"]),
        "",
        "## 数据完整度总分",
        f"- 平均值：{inventory['data_completeness_score_100'].mean():.2f}",
        f"- 最小值：{inventory['data_completeness_score_100'].min():.2f}",
        f"- 最大值：{inventory['data_completeness_score_100'].max():.2f}",
        "",
        "## 旧版可用等级",
        value_counts_block(inventory["usable_grade"]),
        "",
        "## ERA 完整度",
        value_counts_block(inventory["era_grade"]),
        "",
        "## LST 完整度",
        value_counts_block(inventory["lst_grade"]),
        "",
        "## 剖面完整度",
        value_counts_block(inventory["profile_grade"]),
        "",
        "## 物理类型覆盖",
        "\n".join(physical_lines),
        "",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing standard lake-year inputs, e.g. data/_standard_inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/data_inventory",
        help="Directory for inventory CSV and summary outputs.",
    )
    parser.add_argument(
        "--date-tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Suffix for output filenames.",
    )
    parser.add_argument(
        "--include-empty-dirs",
        action="store_true",
        help="Include directories that have none of the standard input files.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(data_root, include_empty_dirs=args.include_empty_dirs)
    inventory_path = output_dir / f"lake_year_inventory_{args.date_tag}.csv"
    coverage_path = output_dir / f"lake_year_coverage_matrix_{args.date_tag}.csv"
    inventory.to_csv(inventory_path, index=False)

    coverage_columns = [
        "lake_year_id",
        "data_completeness_class",
        "data_completeness_score_100",
        "recommended_use_case",
        "profile_completeness_score",
        "era_completeness_score",
        "lst_completeness_score",
        "metadata_completeness_score",
        "usable_grade",
        "data_completeness_signature",
        "era_grade",
        "era_unique_days",
        "era_required_daily_complete_fraction",
        "lst_grade",
        "lst_unique_days",
        "lst_temperature_valid_fraction",
        "lst_original_observation_fraction",
        "profile_grade",
        "profile_count",
        "profile_median_layers_per_date",
        "profile_max_observed_depth_m",
        "profile_temperature_valid_fraction",
        "metadata_grade",
        "climate_zone",
        "depth_class",
        "area_class",
        "elevation_class",
        "morphology_class",
        "transparency_class",
        "lake_type_class",
    ]
    inventory[coverage_columns].to_csv(coverage_path, index=False)
    summary_path = _write_summary(inventory, output_dir, args.date_tag)

    print(f"Wrote inventory: {inventory_path}")
    print(f"Wrote coverage matrix: {coverage_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
