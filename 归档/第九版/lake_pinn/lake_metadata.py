import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    KNOWN_LAKE_ATTRIBUTES,
    PINN_CATCHMENT_AREA_REFERENCE_KM2,
    PINN_ELEVATION_REFERENCE_M,
    PINN_FETCH_REFERENCE_M,
    PINN_INFLOW_REFERENCE_M3_S,
    PINN_LIGHT_EXTINCTION_REFERENCE_M_INV,
    PINN_LOG_AREA_REFERENCE_KM2,
    PINN_MAX_DEPTH_REFERENCE_M,
    PINN_MAX_MEAN_DEPTH_REFERENCE_M,
    PINN_RESIDENCE_TIME_REFERENCE_DAYS,
    PINN_SHORELINE_DEVELOPMENT_REFERENCE,
    PINN_SHORELINE_LENGTH_REFERENCE_KM,
    PINN_VOLUME_REFERENCE_KM3,
)


def sanitize_name(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _metadata_value(metadata: dict, *names, default=np.nan):
    if not metadata:
        return default
    for name in names:
        if name in metadata:
            return _safe_float(metadata.get(name), default=default)
    lower_map = {str(key).lower(): value for key, value in metadata.items()}
    for name in names:
        key = str(name).lower()
        if key in lower_map:
            return _safe_float(lower_map[key], default=default)
    return default


def _metadata_text(metadata: dict, *names, default=''):
    if not metadata:
        return default
    for name in names:
        if name in metadata and metadata.get(name) is not None:
            text = str(metadata.get(name)).strip()
            if text:
                return text
    lower_map = {str(key).lower(): value for key, value in metadata.items()}
    for name in names:
        key = str(name).lower()
        if key in lower_map and lower_map[key] is not None:
            text = str(lower_map[key]).strip()
            if text:
                return text
    return default


def _reservoir_indicator_from_value(value):
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = _safe_float(value, default=np.nan)
    if np.isfinite(numeric):
        return 1.0 if float(numeric) > 0.5 else 0.0
    text = str(value).strip().lower()
    if not text:
        return np.nan
    positive_tokens = ('reservoir', 'impoundment', 'dam', 'regulated')
    negative_tokens = ('lake', 'natural', 'pond')
    if any(token in text for token in positive_tokens):
        return 1.0
    if any(token in text for token in negative_tokens):
        return 0.0
    if text in {'true', 'yes', 'y'}:
        return 1.0
    if text in {'false', 'no', 'n'}:
        return 0.0
    return np.nan


def _metadata_reservoir_indicator(metadata: dict | None):
    metadata = metadata or {}
    lower_map = {str(key).lower(): value for key, value in metadata.items()}
    for name in (
        'reservoir_indicator',
        'is_reservoir',
        'reservoir',
        'lake_type',
        'waterbody_type',
        'water_body_type',
        'type',
    ):
        value = metadata.get(name)
        if value is None:
            value = lower_map.get(name.lower())
        indicator = _reservoir_indicator_from_value(value)
        if np.isfinite(indicator):
            return float(indicator)
    return np.nan


def geographic_climate_zone(metadata: dict | None) -> str:
    """Classify broad thermal climate from coordinates, not a hand-written lake type."""
    metadata = metadata or {}
    latitude = _safe_float(metadata.get('latitude'), default=np.nan)
    longitude = _safe_float(metadata.get('longitude'), default=np.nan)
    if not np.isfinite(latitude) or abs(latitude) > 90.0:
        return 'unknown'
    # Longitude is kept in the check so missing/invalid coordinates do not
    # silently masquerade as a confident geographic classification.
    if not np.isfinite(longitude) or abs(longitude) > 180.0:
        return 'unknown'
    abs_lat = abs(float(latitude))
    if abs_lat < 23.5:
        return 'tropical'
    if abs_lat < 38.0:
        return 'warm_subtropical'
    if abs_lat < 50.0:
        return 'temperate_cool'
    return 'cold_high_latitude'


def is_geographic_warm_deep_lake(metadata: dict | None, max_depth=None) -> bool:
    """Return True for low-latitude deep lakes such as Kinneret.

    This intentionally ignores metadata['lake_type']; the gate is based on
    latitude/longitude plus depth so model behavior is not decided by an
    arbitrary label in the JSON.
    """
    metadata = metadata or {}
    zone = geographic_climate_zone(metadata)
    depth_value = max_depth
    if depth_value is None:
        depth_value = metadata.get('runtime_max_depth_m', metadata.get('max_depth_m', np.nan))
    depth_value = _safe_float(depth_value, default=np.nan)
    if not np.isfinite(depth_value):
        return False
    return zone in {'tropical', 'warm_subtropical'} and float(depth_value) >= 25.0


def scorecard_lake_type_from_geography(metadata: dict | None, max_depth=None) -> str:
    """Map geographic regime to the existing scorecard family names."""
    return 'warm_deep_monomictic' if is_geographic_warm_deep_lake(metadata, max_depth=max_depth) else 'cold_dimictic'


def _load_sidecar_metadata(era5_path: Path, lst_path: Path):
    """Load optional lake metadata JSON placed next to the forcing/LST inputs."""
    candidate_dirs = []
    for path in (lst_path, era5_path):
        parent = Path(path).parent
        if parent not in candidate_dirs:
            candidate_dirs.append(parent)

    explicit_names = [
        "lake_metadata.json",
        "metadata.json",
        "kinneret_metadata.json",
        f"{sanitize_name(lst_path.stem)}_metadata.json",
        f"{sanitize_name(era5_path.stem)}_metadata.json",
    ]
    candidates = []
    for directory in candidate_dirs:
        for name in explicit_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob("*metadata*.json")))

    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            loaded["_metadata_path"] = str(path)
            return loaded
    return {}

def infer_metadata(merged: pd.DataFrame, lst: pd.DataFrame, era5_path: Path, lst_path: Path):
    year = int(merged['Date'].dt.year.mode().iloc[0])

    lake_name = None
    for col in ['Category', 'ID']:
        if col in lst.columns:
            values = lst[col].dropna().astype(str)
            values = values[values.str.strip() != '']
            if col == 'ID':
                values = values[~values.str.fullmatch(r'\d+')]
            if not values.empty:
                lake_name = values.iloc[0].strip()
                break

    if not lake_name:
        lake_name = lst_path.stem.replace('-', ' ').replace('_', ' ')

    sidecar_metadata = _load_sidecar_metadata(era5_path, lst_path)

    sidecar_lake_name = sidecar_metadata.get('lake_name') or sidecar_metadata.get('name')
    if sidecar_lake_name is not None and str(sidecar_lake_name).strip():
        lake_name = str(sidecar_lake_name).strip()
    elif sidecar_metadata.get('lake_id') is not None and str(sidecar_metadata.get('lake_id')).strip():
        lake_name = str(sidecar_metadata.get('lake_id')).strip().replace('_', ' ').title()

    base_tag = sanitize_name(lake_name)
    if not base_tag:
        base_tag = sanitize_name(era5_path.stem.replace('-', ' ').replace('_', ' '))
    file_tag = f'{base_tag}_{year}'
    def first_existing_local(frame: pd.DataFrame, candidates):
        for column in candidates:
            if column in frame.columns:
                return column
        return None
    area_km2 = np.nan
    latitude = np.nan
    longitude = np.nan
    mean_depth_m = np.nan
    max_depth_m = np.nan
    for frame in (lst, merged):
        area_column = first_existing_local(
            frame,
            ['Area_km2', 'area_km2', 'LakeArea_km2', 'lake_area_km2', 'Area_km^2'],
        )
        if area_column is not None:
            area_series = pd.to_numeric(frame[area_column], errors='coerce').dropna()
            if not area_series.empty:
                area_km2 = float(area_series.iloc[0])
                break
    for frame in (lst, merged):
        latitude_column = first_existing_local(
            frame,
            ['Latitude', 'latitude', 'Lat', 'lat', 'latitude_deg'],
        )
        if latitude_column is not None:
            latitude_series = pd.to_numeric(frame[latitude_column], errors='coerce').dropna()
            if not latitude_series.empty:
                latitude = float(latitude_series.iloc[0])
                break
    for frame in (lst, merged):
        longitude_column = first_existing_local(
            frame,
            ['Longitude', 'longitude', 'Lon', 'lon', 'longitude_deg'],
        )
        if longitude_column is not None:
            longitude_series = pd.to_numeric(frame[longitude_column], errors='coerce').dropna()
            if not longitude_series.empty:
                longitude = float(longitude_series.iloc[0])
                break
    for frame in (lst, merged):
        mean_depth_column = first_existing_local(
            frame,
            ['MeanDepth_m', 'mean_depth_m', 'mean_depth', 'MeanDepth'],
        )
        if mean_depth_column is not None:
            mean_depth_series = pd.to_numeric(frame[mean_depth_column], errors='coerce').dropna()
            if not mean_depth_series.empty:
                mean_depth_m = float(mean_depth_series.iloc[0])
                break
    for frame in (lst, merged):
        max_depth_column = first_existing_local(
            frame,
            ['MaxDepth_m', 'max_depth_m', 'max_depth', 'MaxDepth'],
        )
        if max_depth_column is not None:
            max_depth_series = pd.to_numeric(frame[max_depth_column], errors='coerce').dropna()
            if not max_depth_series.empty:
                max_depth_m = float(max_depth_series.iloc[0])
                break

    if sidecar_metadata:
        if not np.isfinite(area_km2):
            area_km2 = _metadata_value(sidecar_metadata, 'area_km2', 'area', 'surface_area_km2')
        if not np.isfinite(latitude):
            latitude = _metadata_value(sidecar_metadata, 'latitude', 'lat', 'latitude_deg')
        if not np.isfinite(longitude):
            longitude = _metadata_value(sidecar_metadata, 'longitude', 'lon', 'longitude_deg')
        if not np.isfinite(mean_depth_m):
            mean_depth_m = _metadata_value(sidecar_metadata, 'mean_depth_m', 'mean_depth')
        if not np.isfinite(max_depth_m):
            max_depth_m = _metadata_value(sidecar_metadata, 'max_depth_m', 'max_depth')

    lake_lookup = KNOWN_LAKE_ATTRIBUTES.get(base_tag) or KNOWN_LAKE_ATTRIBUTES.get(sanitize_name(lake_name))
    if lake_lookup is None:
        for known_name, attrs in KNOWN_LAKE_ATTRIBUTES.items():
            known_tag = sanitize_name(known_name)
            if known_tag and (known_tag in base_tag or known_tag in sanitize_name(lake_name)):
                lake_lookup = attrs
                break
    if lake_lookup:
        if not np.isfinite(area_km2):
            area_km2 = float(lake_lookup.get('area_km2', np.nan))
        if not np.isfinite(latitude):
            latitude = float(lake_lookup.get('latitude', np.nan))
        if not np.isfinite(longitude):
            longitude = float(lake_lookup.get('longitude', np.nan))
        if not np.isfinite(mean_depth_m):
            mean_depth_m = float(lake_lookup.get('mean_depth_m', np.nan))
        if not np.isfinite(max_depth_m):
            max_depth_m = float(lake_lookup.get('max_depth_m', np.nan))

    if not np.isfinite(area_km2):
        area_km2 = 1.0
    if not np.isfinite(latitude):
        latitude = 45.0
    if not np.isfinite(longitude):
        longitude = 0.0
    if not np.isfinite(mean_depth_m):
        mean_depth_m = 0.0
    if not np.isfinite(max_depth_m):
        max_depth_m = 0.0

    metadata = {
        'lake_name': lake_name,
        'year': year,
        'file_tag': file_tag,
        'area_km2': float(area_km2),
        'latitude': float(latitude),
        'longitude': float(longitude),
        'mean_depth_m': float(mean_depth_m),
        'max_depth_m': float(max_depth_m),
    }
    metadata['geographic_climate_zone'] = geographic_climate_zone(metadata)
    metadata['geographic_lake_regime'] = scorecard_lake_type_from_geography(metadata, max_depth=max_depth_m)
    for key in ['lake_id', 'lake_type', 'thermal_regime']:
        value = _metadata_text(sidecar_metadata, key)
        if (value is None or value == '') and lake_lookup:
            value = _metadata_text(lake_lookup, key)
        if value is not None and value != '':
            metadata[key] = str(value)
    reservoir_indicator = _metadata_reservoir_indicator(sidecar_metadata)
    if not np.isfinite(reservoir_indicator) and lake_lookup:
        reservoir_indicator = _metadata_reservoir_indicator(lake_lookup)
    if not np.isfinite(reservoir_indicator):
        reservoir_indicator = _metadata_reservoir_indicator(metadata)
    if np.isfinite(reservoir_indicator):
        metadata['reservoir_indicator'] = float(reservoir_indicator)
    for key in [
        'volume_km3',
        'elevation_m',
        'secchi_m',
        'light_extinction_kd',
        'fetch_m',
        'effective_fetch_m',
        'wind_exposure_index',
        'basin_shape_factor',
        'water_level_m',
        'mean_water_level_m',
        'bottom_temp_prior_c',
        'bottom_temperature_prior_c',
        'deep_temp_prior_c',
    ]:
        value = _metadata_value(sidecar_metadata, key)
        if not np.isfinite(value) and lake_lookup:
            value = _metadata_value(lake_lookup, key)
        if np.isfinite(value):
            metadata[key] = float(value)
    extended_numeric_aliases = {
        'residence_time_days': (
            'residence_time_days',
            'residence_time',
            'retention_time_days',
            'retention_time',
            'res_time_days',
            'res_time',
            'residence_time_d',
            'retention_time_d',
        ),
        'shoreline_length_km': (
            'shoreline_length_km',
            'shore_len_km',
            'shoreline_length',
            'shore_len',
            'shore_length_km',
            'shore_length',
        ),
        'shoreline_development': (
            'shoreline_development',
            'shoreline_dev',
            'shore_dev',
            'shoreline_development_index',
        ),
        'catchment_area_km2': (
            'catchment_area_km2',
            'catchment_area',
            'watershed_area_km2',
            'watershed_area',
            'drainage_area_km2',
            'drainage_area',
        ),
        'discharge_m3_s': (
            'discharge_m3_s',
            'mean_discharge_m3_s',
            'discharge',
            'mean_discharge',
            'outflow_m3_s',
            'inflow_m3_s',
        ),
    }
    for key, aliases in extended_numeric_aliases.items():
        value = _metadata_value(sidecar_metadata, *aliases)
        if not np.isfinite(value) and lake_lookup:
            value = _metadata_value(lake_lookup, *aliases)
        if np.isfinite(value):
            metadata[key] = float(value)
    if sidecar_metadata.get('_metadata_path'):
        metadata['metadata_path'] = sidecar_metadata['_metadata_path']
    return metadata


def normalize_max_depth_feature(max_depth: float) -> float:
    return float(np.clip(float(max_depth) / PINN_MAX_DEPTH_REFERENCE_M, 0.0, 2.0))


def normalize_mean_depth_feature(mean_depth: float) -> float:
    return float(np.clip(float(mean_depth) / PINN_MAX_MEAN_DEPTH_REFERENCE_M, 0.0, 2.0))


def normalize_log_area_feature(area_km2: float) -> float:
    area_km2 = max(float(area_km2), 1.0e-6)
    return float(np.log1p(area_km2) / PINN_LOG_AREA_REFERENCE_KM2)


def normalize_latitude_feature(latitude: float) -> float:
    if np.isfinite(latitude) and abs(float(latitude)) <= 1.0:
        return float(np.clip(float(latitude), -1.0, 1.0))
    return float(np.clip(float(latitude) / 90.0, -1.0, 1.0))


def normalize_longitude_feature(longitude: float) -> float:
    if np.isfinite(longitude) and abs(float(longitude)) <= 1.0:
        return float(np.clip(float(longitude), -1.0, 1.0))
    return float(np.clip(float(longitude) / 180.0, -1.0, 1.0))


def normalize_signed_reference(value: float, reference: float, default: float = 0.0) -> float:
    if not np.isfinite(float(value)):
        value = default
    return float(np.clip(float(value) / float(reference), -2.0, 2.0))


def normalize_positive_reference(value: float, reference: float, default: float = 0.0) -> float:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        value = default
    return float(np.clip(float(value) / float(reference), 0.0, 2.0))


def normalize_unit_interval(value: float, default: float = 0.0) -> float:
    if not np.isfinite(float(value)):
        value = default
    return float(np.clip(float(value), 0.0, 1.0))


def metadata_static_features(metadata: dict | None, max_depth: float) -> dict:
    metadata = metadata or {}
    area_km2 = float(metadata.get('area_km2', 1.0))
    latitude = float(metadata.get('latitude', 45.0))
    longitude = float(metadata.get('longitude', 0.0))
    mean_depth_m = float(metadata.get('mean_depth_m', 0.0))
    if not np.isfinite(mean_depth_m) or mean_depth_m <= 0.0:
        mean_depth_m = 0.0
    volume_km3 = float(metadata.get('volume_km3', 0.0))
    elevation_m = float(metadata.get('elevation_m', 0.0))
    light_extinction_kd = float(metadata.get('light_extinction_kd', 0.0))
    fetch_m = float(metadata.get('fetch_m', metadata.get('effective_fetch_m', 0.0)))
    wind_exposure_index = float(metadata.get('wind_exposure_index', 0.0))
    reservoir_indicator = _metadata_reservoir_indicator(metadata)
    if not np.isfinite(reservoir_indicator):
        reservoir_indicator = 0.0
    residence_time_days = _metadata_value(
        metadata,
        'residence_time_days',
        'residence_time',
        'retention_time_days',
        'retention_time',
        'res_time_days',
        'res_time',
        default=0.0,
    )
    shoreline_length_km = _metadata_value(
        metadata,
        'shoreline_length_km',
        'shore_len_km',
        'shoreline_length',
        'shore_len',
        'shore_length_km',
        'shore_length',
        default=0.0,
    )
    catchment_area_km2 = _metadata_value(
        metadata,
        'catchment_area_km2',
        'catchment_area',
        'watershed_area_km2',
        'watershed_area',
        'drainage_area_km2',
        'drainage_area',
        default=0.0,
    )
    discharge_m3_s = _metadata_value(
        metadata,
        'discharge_m3_s',
        'mean_discharge_m3_s',
        'discharge',
        'mean_discharge',
        'outflow_m3_s',
        'inflow_m3_s',
        default=0.0,
    )
    max_depth_m = float(max_depth)
    basin_shape_factor = float(metadata.get('basin_shape_factor', np.nan))
    if not np.isfinite(basin_shape_factor):
        if (
            np.isfinite(volume_km3)
            and volume_km3 > 0.0
            and np.isfinite(area_km2)
            and area_km2 > 0.0
            and np.isfinite(max_depth_m)
            and max_depth_m > 0.0
        ):
            # volume_km3 / (area_km2 * depth_m * 0.001) is a compact
            # hypsometry surrogate: 1.0 is box-like, smaller is bowl-like.
            basin_shape_factor = volume_km3 / (area_km2 * max_depth_m * 0.001)
        elif np.isfinite(mean_depth_m) and mean_depth_m > 0.0 and max_depth_m > 0.0:
            basin_shape_factor = mean_depth_m / max_depth_m
        else:
            basin_shape_factor = 0.0
    shoreline_development = _metadata_value(
        metadata,
        'shoreline_development',
        'shoreline_dev',
        'shore_dev',
        'shoreline_development_index',
        default=np.nan,
    )
    if not np.isfinite(shoreline_development):
        if (
            np.isfinite(shoreline_length_km)
            and shoreline_length_km > 0.0
            and np.isfinite(area_km2)
            and area_km2 > 0.0
        ):
            shoreline_development = shoreline_length_km / (2.0 * np.sqrt(np.pi * area_km2))
        else:
            shoreline_development = 0.0
    return {
        'max_depth_norm': normalize_max_depth_feature(max_depth),
        'mean_depth_norm': normalize_mean_depth_feature(mean_depth_m),
        'log_area': normalize_log_area_feature(area_km2),
        'latitude': normalize_latitude_feature(latitude),
        'longitude': normalize_longitude_feature(longitude),
        'volume_norm': normalize_positive_reference(volume_km3, PINN_VOLUME_REFERENCE_KM3),
        'elevation_norm': normalize_signed_reference(elevation_m, PINN_ELEVATION_REFERENCE_M),
        'light_extinction_norm': normalize_positive_reference(light_extinction_kd, PINN_LIGHT_EXTINCTION_REFERENCE_M_INV),
        'fetch_norm': normalize_positive_reference(fetch_m, PINN_FETCH_REFERENCE_M),
        'wind_exposure_norm': normalize_positive_reference(wind_exposure_index, 1.0),
        'basin_shape_norm': float(np.clip(basin_shape_factor, 0.0, 2.0)),
        'reservoir_indicator': normalize_unit_interval(reservoir_indicator),
        'residence_time_norm': normalize_positive_reference(
            residence_time_days,
            PINN_RESIDENCE_TIME_REFERENCE_DAYS,
        ),
        'shoreline_length_norm': normalize_positive_reference(
            shoreline_length_km,
            PINN_SHORELINE_LENGTH_REFERENCE_KM,
        ),
        'shoreline_development_norm': normalize_positive_reference(
            shoreline_development,
            PINN_SHORELINE_DEVELOPMENT_REFERENCE,
        ),
        'catchment_area_norm': normalize_positive_reference(
            catchment_area_km2,
            PINN_CATCHMENT_AREA_REFERENCE_KM2,
        ),
        'discharge_norm': normalize_positive_reference(
            discharge_m3_s,
            PINN_INFLOW_REFERENCE_M3_S,
        ),
    }


def season_label_for_month(month_value: int) -> str:
    if month_value in {12, 1, 2}:
        return 'winter'
    if month_value in {3, 4, 5}:
        return 'spring'
    if month_value in {6, 7, 8}:
        return 'summer'
    return 'autumn'


def build_contiguous_season_segments(df: pd.DataFrame):
    if df.empty:
        return []

    season_labels = df['month'].apply(lambda month_value: season_label_for_month(int(month_value))).tolist()
    segments = []
    season_counts = {}
    start_idx = 0

    for idx in range(1, len(df) + 1):
        is_boundary = idx == len(df) or season_labels[idx] != season_labels[start_idx]
        if not is_boundary:
            continue

        season_name = season_labels[start_idx]
        season_counts[season_name] = season_counts.get(season_name, 0) + 1
        occurrence = season_counts[season_name]
        segment_name = season_name if occurrence == 1 else f'{season_name}_{occurrence}'
        segments.append(
            {
                'name': segment_name,
                'season': season_name,
                'start_idx': start_idx,
                'end_idx': idx,
                'start_date': pd.Timestamp(df['Date'].iloc[start_idx]),
                'end_date': pd.Timestamp(df['Date'].iloc[idx - 1]),
            }
        )
        start_idx = idx

    return segments
