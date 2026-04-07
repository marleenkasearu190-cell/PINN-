import argparse
import calendar
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


DATASET = 'reanalysis-era5-single-levels'
OUTPUT_DIR = Path(__file__).resolve().parent
VARIABLES = [
    'lake_mix_layer_depth',
    'lake_bottom_temperature',
    'surface_solar_radiation_downwards',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
]
REQUIRED_SHORT_NAMES = {'lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m'}
VARIABLE_LIST = ['lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m']
COLUMN_ORDER = [
    'Date', 'lmld_m', 'lblt_K', 'lblt_C', 'Is_J_per_m2',
    'u10_m_per_s', 'v10_m_per_s', 'wind_norm_m_per_s',
    't2m_K', 't2m_C'
]
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 90
LAKE_PRESETS = {
    'mendota': {
        'display_name': 'Mendota',
        'bbox': [43.25, -89.50, 43.00, -89.25],
    },
    'acton': {
        'display_name': 'Acton',
        'bbox': [39.59, -84.00, 39.52, -83.90],
    },
}


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def build_client():
    return cdsapi.Client()


def build_paths(lake_slug: str, year: str):
    monthly_template = f'ERA5_{lake_slug}_{{year}}_{{month}}.nc'
    merged_file = OUTPUT_DIR / f'ERA5_{lake_slug}_{year}_full.nc'
    hourly_file = OUTPUT_DIR / f'ERA5_{lake_slug}_{year}_Hourly.csv'
    daily_file = OUTPUT_DIR / f'ERA5_{lake_slug}_{year}_Daily.csv'
    return monthly_template, merged_file, hourly_file, daily_file


def month_file(month: int, monthly_template: str, year: str) -> Path:
    return OUTPUT_DIR / monthly_template.format(year=year, month=f'{month:02d}')


def archive_contains_required_vars(archive_path: Path) -> bool:
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(archive_path, 'r') as archive:
                archive.extractall(temp_dir)

            extracted_files = list(Path(temp_dir).glob('*.nc'))
            if not extracted_files:
                return False

            found = set()
            for file in extracted_files:
                ds = xr.open_dataset(file, engine='netcdf4')
                try:
                    found.update(ds.data_vars)
                finally:
                    ds.close()

        return REQUIRED_SHORT_NAMES.issubset(found)
    except Exception:
        return False


def download_year_data(client, lake_name: str, bbox, year: str, monthly_template: str):
    for month in range(1, 13):
        month_str = f'{month:02d}'
        days_in_month = calendar.monthrange(int(year), month)[1]
        output_file = month_file(month, monthly_template, year)

        if archive_contains_required_vars(output_file):
            print(f'跳过 {output_file.name}，文件已包含所需变量')
            continue

        print(f'开始下载 {lake_name} 湖 {year} 年 {month_str} 月数据 -> {output_file.name}')
        request = {
            'product_type': ['reanalysis'],
            'variable': VARIABLES,
            'year': [year],
            'month': [month_str],
            'day': [f'{day:02d}' for day in range(1, days_in_month + 1)],
            'time': [f'{hour:02d}:00' for hour in range(24)],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            'area': bbox,
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if output_file.exists() and not archive_contains_required_vars(output_file):
                    output_file.unlink()

                print(f'尝试 {attempt}/{MAX_RETRIES}')
                client.retrieve(DATASET, request, str(output_file))

                if not archive_contains_required_vars(output_file):
                    raise RuntimeError(f'{output_file.name} 下载后变量仍不完整')

                print(f'{output_file.name} 下载完成')
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f'{output_file.name} 下载失败: {exc}')
                if attempt < MAX_RETRIES:
                    print(f'{RETRY_WAIT_SECONDS} 秒后重试...')
                    time.sleep(RETRY_WAIT_SECONDS)

        if last_error is not None:
            raise last_error


def check_missing_months(monthly_template: str, year: str):
    missing = []
    incomplete = []
    for month in range(1, 13):
        file_path = month_file(month, monthly_template, year)
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing.append(f'{month:02d}')
        elif not archive_contains_required_vars(file_path):
            incomplete.append(f'{month:02d}')

    if missing:
        print(f'缺失月份: {", ".join(missing)}')
    if incomplete:
        print(f'变量不完整的月份: {", ".join(incomplete)}')
    if not missing and not incomplete:
        print('12 个月份数据齐全，且包含全部所需变量')

    return missing + incomplete


def load_month_dataset(month_path: Path) -> xr.Dataset:
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(month_path, 'r') as archive:
            archive.extractall(temp_dir)

        extracted_files = sorted(Path(temp_dir).glob('*.nc'))
        if not extracted_files:
            raise FileNotFoundError(f'压缩包内未找到 NetCDF 文件: {month_path.name}')

        opened = [xr.open_dataset(file, engine='netcdf4') for file in extracted_files]
        try:
            month_ds = xr.merge(opened, compat='override', combine_attrs='drop_conflicts')
            month_ds = month_ds.load()
        finally:
            for ds in opened:
                ds.close()

    if 'valid_time' in month_ds.dims or 'valid_time' in month_ds.coords:
        month_ds = month_ds.rename({'valid_time': 'time'})

    return month_ds


def merge_year_data(monthly_template: str, merged_file: Path, year: str):
    missing = check_missing_months(monthly_template, year)
    if missing:
        raise RuntimeError('存在缺失或不完整月份，无法合并全年数据。')

    monthly_files = [month_file(month, monthly_template, year) for month in range(1, 13)]
    print(f'开始合并全年数据 -> {merged_file.name}')

    datasets = [load_month_dataset(path) for path in monthly_files]
    try:
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')
        temp_merged_file = Path(r'C:\Users\A\Documents\Playground') / merged_file.name
        if temp_merged_file.exists():
            temp_merged_file.unlink()
        merged.to_netcdf(temp_merged_file, engine='scipy')
        if merged_file.exists():
            merged_file.unlink()
        shutil.copy2(temp_merged_file, merged_file)
        temp_merged_file.unlink()
    finally:
        for ds in datasets:
            ds.close()

    print(f'全年合并完成 -> {merged_file.name}')
    return merged_file


def open_dataset_with_fallback(input_file: Path):
    try:
        return xr.open_dataset(input_file, engine='netcdf4'), None
    except Exception as first_error:
        temp_dir = Path(tempfile.mkdtemp(prefix='lake_nc_'))
        temp_file = temp_dir / input_file.name
        shutil.copy2(input_file, temp_file)

        last_error = first_error
        for engine in ['netcdf4', 'scipy', 'h5netcdf']:
            try:
                ds = xr.open_dataset(temp_file, engine=engine)
                print(f'   使用临时英文路径打开成功，engine={engine}')
                return ds, temp_dir
            except Exception as exc:
                last_error = exc

        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f'无法打开 NetCDF 文件: {input_file}') from last_error


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    if 'valid_time' in ds.coords or 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    for dim in ['latitude', 'longitude']:
        if dim in ds.dims and ds.sizes.get(dim, 0) == 1:
            ds = ds.isel({dim: 0})

    if 'expver' in ds.dims:
        ds = ds.max(dim='expver', skipna=True)

    return ds


def build_dataframe(ds_avg: xr.Dataset) -> pd.DataFrame:
    df = ds_avg.to_dataframe().reset_index()
    df = df.rename(
        columns={
            'time': 'Date',
            'lmld': 'lmld_m',
            'lblt': 'lblt_K',
            'ssrd': 'Is_J_per_m2',
            'u10': 'u10_m_per_s',
            'v10': 'v10_m_per_s',
            't2m': 't2m_K',
        }
    )
    df['Date'] = pd.to_datetime(df['Date'])
    df['lblt_C'] = df['lblt_K'] - 273.15
    df['t2m_C'] = df['t2m_K'] - 273.15
    df['wind_norm_m_per_s'] = np.sqrt(df['u10_m_per_s'] ** 2 + df['v10_m_per_s'] ** 2)
    return df


def save_csv_with_fallback(df: pd.DataFrame, target: Path) -> Path:
    try:
        df.to_csv(target, index=False, encoding='utf-8-sig')
        return target
    except PermissionError:
        fallback = target.with_name(f'{target.stem}_new{target.suffix}')
        df.to_csv(fallback, index=False, encoding='utf-8-sig')
        return fallback


def extract_csv_files(merged_file: Path, hourly_file: Path, daily_file: Path):
    print(f'开始提取 CSV -> {merged_file.name}')
    ds, temp_dir = open_dataset_with_fallback(merged_file)

    try:
        ds = normalize_dataset(ds)
        missing_vars = [var for var in VARIABLE_LIST if var not in ds.data_vars]
        if missing_vars:
            raise ValueError('全年文件缺少以下变量，无法提取标准 CSV: ' + ', '.join(missing_vars))

        spatial_dims = [dim for dim in ['latitude', 'longitude'] if dim in ds['lmld'].dims]
        if spatial_dims:
            ds_spatial_mean = ds[VARIABLE_LIST].mean(dim=spatial_dims)
        else:
            ds_spatial_mean = ds[VARIABLE_LIST]

        hourly_df = build_dataframe(ds_spatial_mean)
        hourly_df = hourly_df[COLUMN_ORDER]
        saved_hourly = save_csv_with_fallback(hourly_df, hourly_file)

        ds_daily = ds_spatial_mean.resample(time='1D').mean()
        daily_df = build_dataframe(ds_daily)
        daily_df = daily_df[COLUMN_ORDER]
        saved_daily = save_csv_with_fallback(daily_df, daily_file)
    finally:
        ds.close()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f'小时尺度文件: {saved_hourly}')
    print(f'日尺度文件: {saved_daily}')
    return saved_hourly, saved_daily


def parse_args():
    parser = argparse.ArgumentParser(description='Download ERA5 monthly files for a lake, merge them, and extract cleaned CSV files in one script.')
    parser.add_argument('--lake', default='mendota', help='Preset lake name, for example mendota or acton')
    parser.add_argument('--year', default='2018', help='Target year, for example 2018')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help='Override bbox as north west south east')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading and reuse existing monthly files')
    parser.add_argument('--skip-merge', action='store_true', help='Skip merging and reuse existing merged file')
    parser.add_argument('--merged-file', help='Optional path to an existing merged nc file')
    parser.add_argument('--output-dir', help='Optional directory for merged nc and extracted csv outputs')
    return parser.parse_args()


def resolve_lake_config(args):
    lake_key = slugify(args.lake)
    preset = LAKE_PRESETS.get(lake_key)

    if preset is None and args.bbox is None:
        known = ', '.join(sorted(LAKE_PRESETS))
        raise ValueError(f'未知湖泊预设: {args.lake}。请使用已知预设 ({known})，或通过 --bbox 指定范围。')

    display_name = preset['display_name'] if preset else args.lake
    bbox = args.bbox if args.bbox is not None else preset['bbox']
    lake_slug = lake_key or slugify(display_name)
    return display_name, lake_slug, bbox


def main():
    args = parse_args()
    display_name, lake_slug, bbox = resolve_lake_config(args)
    monthly_template, default_merged_file, default_hourly_file, default_daily_file = build_paths(lake_slug, args.year)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_file = output_dir / default_merged_file.name
        hourly_file = output_dir / default_hourly_file.name
        daily_file = output_dir / default_daily_file.name
    else:
        merged_file = default_merged_file
        hourly_file = default_hourly_file
        daily_file = default_daily_file

    if args.merged_file:
        merged_file = Path(args.merged_file).expanduser().resolve()

    print(f'开始处理 {display_name} 湖 {args.year} ERA5 数据')
    print(f'使用范围: {bbox}')

    if not args.skip_download and not args.merged_file:
        client = build_client()
        download_year_data(client, display_name, bbox, args.year, monthly_template)
    elif args.skip_download:
        print('已跳过下载步骤，直接使用现有月文件或全年文件')

    if args.skip_merge or args.merged_file:
        if not merged_file.exists():
            raise FileNotFoundError(f'未找到全年文件: {merged_file}')
        print(f'已跳过合并步骤，直接使用: {merged_file.name}')
    else:
        merge_year_data(monthly_template, merged_file, args.year)

    extract_csv_files(merged_file, hourly_file, daily_file)
    print('全部任务完成')


if __name__ == '__main__':
    main()
