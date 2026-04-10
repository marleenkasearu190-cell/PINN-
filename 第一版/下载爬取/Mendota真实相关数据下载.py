import argparse
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


PACKAGES = {
    "water_temp": ("knb-lter-ntl", "130", "31"),
    "met": ("knb-lter-ntl", "129", "35"),
    "under_ice": ("knb-lter-ntl", "390", "3"),
    "sonde": ("knb-lter-ntl", "400", "4"),
}

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "数据" / "处理后" / "PINN输入"


def fetch_eml(scope, identifier, revision):
    url = f"https://pasta.lternet.edu/package/metadata/eml/{scope}/{identifier}/{revision}"
    last_error = None
    for _ in range(4):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, timeout=60, context=ctx) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise last_error


def get_object_urls(scope, identifier, revision):
    xml_text = fetch_eml(scope, identifier, revision)
    root = ET.fromstring(xml_text)
    urls = {}
    for data_table in root.findall(".//dataTable"):
        object_name = data_table.find("./physical/objectName")
        online_url = data_table.find("./physical/distribution/online/url")
        if object_name is not None and online_url is not None:
            urls[object_name.text] = online_url.text
    return urls


def download_csv(url):
    last_error = None
    for _ in range(4):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, timeout=120, context=ctx) as response:
                return pd.read_csv(response, low_memory=False)
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise last_error


def load_mendota_water_temperature(year):
    scope, identifier, revision = PACKAGES["water_temp"]
    urls = get_object_urls(scope, identifier, revision)
    candidates = [
        f"ntl130_{year}_v2.csv",
        f"ntl130_{year}_v1.csv",
    ]
    for name in candidates:
        if name in urls:
            df = download_csv(urls[name])
            df["datetime"] = pd.to_datetime(df["sampledate"] + " " + df["sampletime"], errors="coerce")
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
            df["wtemp"] = pd.to_numeric(df["wtemp"], errors="coerce")
            df = df.dropna(subset=["datetime", "depth", "wtemp"]).copy()
            df["source"] = "open_water_buoy"
            return df
    raise FileNotFoundError(f"No Mendota buoy water-temperature file found for year {year}.")


def load_mendota_meteorology(year):
    scope, identifier, revision = PACKAGES["met"]
    urls = get_object_urls(scope, identifier, revision)
    name = "ntl129_3_v11.csv"
    if name not in urls:
        raise FileNotFoundError(f"Expected meteorology file {name} not found in package metadata.")

    df = download_csv(urls[name])
    df["datetime"] = pd.to_datetime(df["sampledate"] + " " + df["sampletime"], errors="coerce")
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["par"] = pd.to_numeric(df["par"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df = df[df["datetime"].dt.year == year].copy()
    return df


def load_mendota_under_ice(year):
    scope, identifier, revision = PACKAGES["under_ice"]
    urls = get_object_urls(scope, identifier, revision)
    name = "winter_lake_temp.csv"
    if name not in urls:
        raise FileNotFoundError(f"Expected under-ice file {name} not found in package metadata.")

    df = download_csv(urls[name])
    df["datetime"] = pd.to_datetime(df["Sampledate"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["wtemp"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["lake"] = df["lake"].astype(str).str.strip()
    df = df[(df["lake"] == "Lake Mendota") & (df["datetime"].dt.year == year)].copy()
    df = df.dropna(subset=["datetime", "depth", "wtemp"]).copy()
    df["source"] = "under_ice_buoy"
    return df


def load_mendota_sonde(year):
    scope, identifier, revision = PACKAGES["sonde"]
    urls = get_object_urls(scope, identifier, revision)
    name = "ntl400_v4.csv"
    if name not in urls:
        raise FileNotFoundError(f"Expected sonde file {name} not found in package metadata.")

    df = download_csv(urls[name])
    df["datetime"] = pd.to_datetime(df["sampledate"] + " " + df["sampletime"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["wtemp"] = pd.to_numeric(df["wtemp"], errors="coerce")
    df = df[(df["lakeid"] == "ME") & (df["datetime"].dt.year == year)].copy()
    df = df.dropna(subset=["datetime", "depth", "wtemp"]).copy()
    return df


def build_temperature_table(year):
    open_water = load_mendota_water_temperature(year)
    under_ice = load_mendota_under_ice(year)
    all_temp = pd.concat([open_water, under_ice], ignore_index=True)
    all_temp["Date"] = all_temp["datetime"].dt.date.astype(str)
    all_temp["DOY"] = all_temp["datetime"].dt.dayofyear
    all_temp["Month"] = all_temp["datetime"].dt.month
    all_temp = all_temp.rename(columns={"depth": "Depth_m", "wtemp": "Temperature_C"})
    all_temp = all_temp[["datetime", "Date", "Month", "DOY", "Depth_m", "Temperature_C", "source"]]
    all_temp = all_temp.sort_values(["datetime", "Depth_m"]).reset_index(drop=True)
    return all_temp


def build_daily_forcing_table(year, met_df):
    met_df = met_df.copy()
    met_df["Date"] = met_df["datetime"].dt.date.astype(str)
    met_df["DOY"] = met_df["datetime"].dt.dayofyear
    daily = (
        met_df.groupby("Date", as_index=False)
        .agg(
            DOY=("DOY", "first"),
            air_temp_C=("air_temp", "mean"),
            wind_speed_mps=("wind_speed", "mean"),
            par_mean=("par", "mean"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    daily["Year"] = year
    return daily


def build_validation_table(year, sonde_df):
    sonde_df = sonde_df.copy()
    sonde_df["Date"] = sonde_df["datetime"].dt.date.astype(str)
    sonde_df["DOY"] = sonde_df["datetime"].dt.dayofyear
    sonde_df["Month"] = sonde_df["datetime"].dt.month
    sonde_df = sonde_df.rename(columns={"depth": "Depth_m", "wtemp": "Temperature_C"})
    sonde_df = sonde_df[["datetime", "Date", "Month", "DOY", "Depth_m", "Temperature_C"]]
    sonde_df = sonde_df.sort_values(["datetime", "Depth_m"]).reset_index(drop=True)
    return sonde_df


def main():
    parser = argparse.ArgumentParser(
        description="Download Lake Mendota EDI datasets and prepare PINN-ready input tables."
    )
    parser.add_argument("--year", type=int, default=2018, help="Target year. Best supported: 2018 or 2020.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output CSV files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_table = build_temperature_table(args.year)
    met_raw = load_mendota_meteorology(args.year)
    forcing_table = build_daily_forcing_table(args.year, met_raw)
    validation_table = build_validation_table(args.year, load_mendota_sonde(args.year))

    temp_path = output_dir / f"门多塔湖_{args.year}_温度剖面.csv"
    forcing_path = output_dir / f"门多塔湖_{args.year}_日强迫.csv"
    validation_path = output_dir / f"门多塔湖_{args.year}_浮标验证.csv"

    temp_table.to_csv(temp_path, index=False)
    forcing_table.to_csv(forcing_path, index=False)
    validation_table.to_csv(validation_path, index=False)

    print(f"Saved temperature profiles to: {temp_path}")
    print(f"Saved daily forcing to: {forcing_path}")
    print(f"Saved sonde validation profiles to: {validation_path}")
    print("Note: PAR is not the same unit as shortwave radiation energy. If your PDE needs Is_J_per_m2, supplement with ERA5 or convert PAR carefully.")


if __name__ == "__main__":
    main()
