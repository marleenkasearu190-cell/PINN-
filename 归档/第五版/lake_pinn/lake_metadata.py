# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

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

    return {
        'lake_name': lake_name,
        'year': year,
        'file_tag': file_tag,
        'area_km2': float(area_km2),
        'latitude': float(latitude),
        'longitude': float(longitude),
        'mean_depth_m': float(mean_depth_m),
        'max_depth_m': float(max_depth_m),
    }


def normalize_max_depth_feature(max_depth: float) -> float:
    return float(np.clip(float(max_depth) / PINN_MAX_DEPTH_REFERENCE_M, 0.0, 2.0))


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


def metadata_static_features(metadata: dict | None, max_depth: float) -> dict:
    metadata = metadata or {}
    area_km2 = float(metadata.get('area_km2', 1.0))
    latitude = float(metadata.get('latitude', 45.0))
    longitude = float(metadata.get('longitude', 0.0))
    mean_depth_m = float(metadata.get('mean_depth_m', 0.0))
    if not np.isfinite(mean_depth_m) or mean_depth_m <= 0.0:
        mean_depth_m = 0.0
    return {
        'max_depth_norm': normalize_max_depth_feature(max_depth),
        'mean_depth_norm': normalize_max_depth_feature(mean_depth_m),
        'log_area': normalize_log_area_feature(area_km2),
        'latitude': normalize_latitude_feature(latitude),
        'longitude': normalize_longitude_feature(longitude),
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
