"""Build standard wide-input files for multi-lake LakePINN experiments.

The builder does not modify raw ERA5/LST/profile files. It writes three
derived files that future multi-lake or 17D forcing branches can consume:

1. forcing wide table: one row per lake/date
2. profile observation wide table: one row per lake/date with Temp_*m columns
3. metadata JSON: one record of static lake attributes
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_FORCING_COLUMNS = [
    "lake_id",
    "Date",
    "LST_surface_C",
    "T_air_C",
    "wind_speed_m_per_s",
    "Solar_W_m2",
    "Longwave_W_m2",
    "relative_humidity",
    "surface_pressure_Pa",
    "latent_heat_upward_W_m2",
    "sensible_heat_upward_W_m2",
    "Secchi_m",
    "water_level_anomaly",
    "light_extinction_kd",
    "effective_fetch",
    "net_inflow",
]


def sanitize_lake_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return cleaned or "lake"


def read_csv_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def pick_numeric_series(df: pd.DataFrame, candidates, default=np.nan) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def fill_standard_series(series: pd.Series, mode: str, default=np.nan) -> pd.Series:
    mode = str(mode or "reconstruction").lower()
    values = pd.to_numeric(series, errors="coerce")
    if mode == "reconstruction":
        values = values.interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(default)
    else:
        raise ValueError(f"Unsupported data fill mode: {mode!r}")
    return values


def lst_surface_c_from_frame(lst: pd.DataFrame) -> pd.DataFrame:
    out = lst[["Date"]].copy()
    if "LST_surface_C" in lst.columns:
        out["LST_surface_C"] = pd.to_numeric(lst["LST_surface_C"], errors="coerce")
    elif "LST_surface_K" in lst.columns:
        out["LST_surface_C"] = pd.to_numeric(lst["LST_surface_K"], errors="coerce") - 273.15
    elif "MOD11A1_061_LST_Day_1km" in lst.columns:
        lst_k = pd.to_numeric(lst["MOD11A1_061_LST_Day_1km"], errors="coerce")
        lst_k = lst_k.where(lst_k > 0)
        out["LST_surface_C"] = lst_k - 273.15
    else:
        raise ValueError("LST file must contain LST_surface_C, LST_surface_K, or MOD11A1_061_LST_Day_1km.")
    return out.groupby("Date", as_index=False)["LST_surface_C"].mean().sort_values("Date")


def build_standard_forcing(
    lake_id: str,
    era5_path: Path,
    lst_path: Path | None,
    secchi_path: Path | None,
    data_fill_mode: str = "reconstruction",
) -> pd.DataFrame:
    era5 = read_csv_with_date(era5_path)
    forcing = pd.DataFrame({"Date": era5["Date"]})

    if lst_path is not None:
        lst = lst_surface_c_from_frame(read_csv_with_date(lst_path))
        forcing = forcing.merge(lst, on="Date", how="left")
    elif "LST_surface_C" in era5.columns:
        forcing["LST_surface_C"] = pd.to_numeric(era5["LST_surface_C"], errors="coerce")
    else:
        forcing["LST_surface_C"] = np.nan

    wind_u = pick_numeric_series(era5, ["u10_m_per_s", "u10", "u10m"])
    wind_v = pick_numeric_series(era5, ["v10_m_per_s", "v10", "v10m"])
    wind_norm = pick_numeric_series(era5, ["wind_norm_m_per_s", "wind_speed_m_per_s", "wind_speed"])
    if not wind_u.isna().all() and not wind_v.isna().all():
        forcing["wind_speed_m_per_s"] = np.sqrt(np.square(wind_u) + np.square(wind_v))
    else:
        forcing["wind_speed_m_per_s"] = wind_norm

    forcing["T_air_C"] = pick_numeric_series(era5, ["t2m_C", "T_air_C", "air_temp_C"])
    if forcing["T_air_C"].isna().all() and "t2m_K" in era5.columns:
        forcing["T_air_C"] = pd.to_numeric(era5["t2m_K"], errors="coerce") - 273.15

    forcing["Solar_W_m2"] = pick_numeric_series(era5, ["Solar_W_m2", "ssrd_W_per_m2", "shortwave_W_m2", "shortwave"])
    if forcing["Solar_W_m2"].isna().all() and "ssrd_J_per_m2" in era5.columns:
        forcing["Solar_W_m2"] = pd.to_numeric(era5["ssrd_J_per_m2"], errors="coerce") / 86400.0
    if forcing["Solar_W_m2"].isna().all() and "Is_J_per_m2" in era5.columns:
        forcing["Solar_W_m2"] = pd.to_numeric(era5["Is_J_per_m2"], errors="coerce") / 86400.0

    forcing["Longwave_W_m2"] = pick_numeric_series(era5, ["Longwave_W_m2", "strd_W_per_m2", "longwave_W_m2", "longwave"])
    forcing["latent_heat_upward_W_m2"] = pick_numeric_series(
        era5,
        ["latent_heat_upward_W_m2", "latent_heat_upward_W_per_m2", "latent_heat_W_m2", "slhf_W_per_m2_raw", "latent_heat"],
        default=0.0,
    )
    forcing["sensible_heat_upward_W_m2"] = pick_numeric_series(
        era5,
        ["sensible_heat_upward_W_m2", "sensible_heat_upward_W_per_m2", "sensible_heat_W_m2", "sshf_W_per_m2_raw", "sensible_heat"],
        default=0.0,
    )

    rh = pick_numeric_series(era5, ["relative_humidity", "rh", "rh_percent", "relative_humidity_percent"])
    if not rh.isna().all() and rh.max(skipna=True) > 1.5:
        rh = rh / 100.0
    forcing["relative_humidity"] = rh
    pressure = pick_numeric_series(era5, ["surface_pressure_Pa", "sp_Pa", "pressure_Pa"])
    if pressure.isna().all():
        pressure_hpa = pick_numeric_series(era5, ["sp_hPa", "surface_pressure_hPa", "pressure_hPa"])
        if not pressure_hpa.isna().all():
            pressure = pressure_hpa * 100.0
    forcing["surface_pressure_Pa"] = pressure

    forcing["Secchi_m"] = np.nan
    if secchi_path is not None and secchi_path.exists():
        secchi = read_csv_with_date(secchi_path)
        secchi_col = "Secchi_m" if "Secchi_m" in secchi.columns else None
        if secchi_col is not None:
            secchi_daily = secchi[["Date", secchi_col]].copy().rename(columns={secchi_col: "Secchi_m"})
            forcing = forcing.drop(columns=["Secchi_m"]).merge(secchi_daily, on="Date", how="left")

    forcing["water_level_anomaly"] = pick_numeric_series(
        era5,
        ["water_level_anomaly", "WaterLevelAnomaly_m", "lake_level_anomaly_m", "stage_anomaly_m"],
        default=0.0,
    )
    forcing["light_extinction_kd"] = pick_numeric_series(
        era5,
        ["light_extinction_kd", "LightExtinctionKd_m_inv", "Kd_m_inv", "kd_m_inv", "shortwave_attenuation_coef"],
        default=np.nan,
    )
    forcing["effective_fetch"] = pick_numeric_series(
        era5,
        ["effective_fetch", "effective_fetch_m", "EffectiveFetch_m", "fetch_m", "wind_fetch_m"],
        default=np.nan,
    )
    if "net_inflow" in era5.columns or "net_inflow_m3_s" in era5.columns or "NetInflow_m3_s" in era5.columns:
        forcing["net_inflow"] = pick_numeric_series(era5, ["net_inflow", "net_inflow_m3_s", "NetInflow_m3_s"], default=0.0)
    else:
        inflow = pick_numeric_series(era5, ["inflow_m3_s", "Inflow_m3_s"], default=np.nan)
        outflow = pick_numeric_series(era5, ["outflow_m3_s", "Outflow_m3_s"], default=np.nan)
        forcing["net_inflow"] = (inflow.fillna(0.0) - outflow.fillna(0.0)) if (not inflow.isna().all() or not outflow.isna().all()) else 0.0

    forcing = forcing.sort_values("Date").reset_index(drop=True)
    default_by_column = {
        "LST_surface_C": forcing["T_air_C"] if "T_air_C" in forcing.columns else 0.0,
        "T_air_C": forcing["LST_surface_C"] if "LST_surface_C" in forcing.columns else 0.0,
        "wind_speed_m_per_s": 1.0,
        "Solar_W_m2": 0.0,
        "Longwave_W_m2": 0.0,
        "relative_humidity": 0.75,
        "surface_pressure_Pa": 101325.0,
        "latent_heat_upward_W_m2": 0.0,
        "sensible_heat_upward_W_m2": 0.0,
        "Secchi_m": 0.0,
        "water_level_anomaly": 0.0,
        "light_extinction_kd": 0.0,
        "effective_fetch": 0.0,
        "net_inflow": 0.0,
    }
    fill_columns = [c for c in STANDARD_FORCING_COLUMNS if c not in {"lake_id", "Date"}]
    for column in fill_columns:
        if column not in forcing.columns:
            forcing[column] = np.nan
        forcing[column] = pd.to_numeric(forcing[column], errors="coerce")
        forcing[column] = fill_standard_series(
            forcing[column],
            mode=data_fill_mode,
            default=default_by_column.get(column, 0.0),
        )

    forcing["lake_id"] = sanitize_lake_id(lake_id)
    return forcing[STANDARD_FORCING_COLUMNS]


def build_standard_profile_obs(lake_id: str, profile_path: Path) -> pd.DataFrame:
    profile = read_csv_with_date(profile_path)
    temp_columns = [c for c in profile.columns if re.fullmatch(r"Temp_[0-9]+m", c)]
    if temp_columns:
        out = profile[["Date", *temp_columns]].copy()
    elif {"Depth_m", "Temperature_C"}.issubset(profile.columns):
        long_profile = profile[["Date", "Depth_m", "Temperature_C"]].copy()
        long_profile["Depth_m"] = pd.to_numeric(long_profile["Depth_m"], errors="coerce")
        long_profile["Temperature_C"] = pd.to_numeric(long_profile["Temperature_C"], errors="coerce")
        long_profile = long_profile.dropna(subset=["Date", "Depth_m", "Temperature_C"])
        long_profile["depth_key"] = long_profile["Depth_m"].round(3)
        wide = long_profile.pivot_table(
            index="Date",
            columns="depth_key",
            values="Temperature_C",
            aggfunc="mean",
        ).sort_index()
        wide.columns = [f"Temp_{float(depth):g}m" for depth in wide.columns]
        out = wide.reset_index()
    else:
        raise ValueError(
            "Profile observation file must be wide Temp_*m columns or long Date/Depth_m/Temperature_C columns."
        )
    out.insert(0, "lake_id", sanitize_lake_id(lake_id))
    return out


def write_metadata(
    output_path: Path,
    lake_id: str,
    max_depth: float | None,
    mean_depth: float | None,
    area_km2: float | None,
    latitude: float | None,
    longitude: float | None,
) -> dict:
    metadata = {
        "lake_id": sanitize_lake_id(lake_id),
        "max_depth": max_depth,
        "mean_depth": mean_depth,
        "area_km2": area_km2,
        "latitude": latitude,
        "longitude": longitude,
    }
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standard wide LakePINN inputs for multi-lake experiments.")
    parser.add_argument("--lake-id", required=True)
    parser.add_argument("--era5", required=True)
    parser.add_argument("--lst", default=None)
    parser.add_argument("--profile-obs", default=None)
    parser.add_argument("--secchi", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--mean-depth", type=float, default=None)
    parser.add_argument("--area-km2", type=float, default=None)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--data-fill-mode", choices=["reconstruction"], default="reconstruction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lake_id = sanitize_lake_id(args.lake_id)

    forcing = build_standard_forcing(
        lake_id=lake_id,
        era5_path=Path(args.era5),
        lst_path=Path(args.lst) if args.lst else None,
        secchi_path=Path(args.secchi) if args.secchi else None,
        data_fill_mode=args.data_fill_mode,
    )
    forcing_path = out_dir / f"{lake_id}_forcing_wide_standard.csv"
    forcing.to_csv(forcing_path, index=False, encoding="utf-8-sig")

    profile_path = None
    if args.profile_obs:
        profile = build_standard_profile_obs(lake_id=lake_id, profile_path=Path(args.profile_obs))
        profile_path = out_dir / f"{lake_id}_profile_obs_wide_standard.csv"
        profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    metadata_path = out_dir / f"{lake_id}_metadata.json"
    metadata = write_metadata(
        metadata_path,
        lake_id=lake_id,
        max_depth=args.max_depth,
        mean_depth=args.mean_depth,
        area_km2=args.area_km2,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    manifest = {
        "forcing": str(forcing_path),
        "profile_obs": str(profile_path) if profile_path is not None else None,
        "metadata": str(metadata_path),
        "forcing_rows": int(len(forcing)),
        "metadata_values": metadata,
        "columns": STANDARD_FORCING_COLUMNS,
        "data_fill_mode": args.data_fill_mode,
    }
    manifest_path = out_dir / f"{lake_id}_standard_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
