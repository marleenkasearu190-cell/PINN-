import argparse
import calendar
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr


DATASET = 'reanalysis-era5-single-levels'
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ERA5_DIR = BASE_DIR / '数据' / '原始' / 'ERA5数据'
EXTRACT_SCRIPT = Path(__file__).resolve().parent / '提取.py'
VARIABLES = [
    'lake_mix_layer_depth',
    'lake_bottom_temperature',
    'surface_solar_radiation_downwards',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
]
REQUIRED_SHORT_NAMES = {'lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m'}
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
    merged_file = RAW_ERA5_DIR / f'ERA5_{lake_slug}_{year}_full.nc'
    return monthly_template, merged_file


def month_file(month: int, monthly_template: str, year: str) -> Path:
    month_str = f'{month:02d}'
    return RAW_ERA5_DIR / monthly_template.format(year=year, month=month_str)


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


def download_year_data(client, lake_name: str, lake_slug: str, bbox, year: str, monthly_template: str):
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
        print('存在缺失或不完整月份，已跳过全年合并')
        return None

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


def extract_merged_data(merged_file: Path, output_dir: Path | None = None):
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f'未找到提取脚本: {EXTRACT_SCRIPT}')

    command = [sys.executable, str(EXTRACT_SCRIPT), '--input', str(merged_file)]
    if output_dir is not None:
        command.extend(['--output-dir', str(output_dir)])

    print(f'开始自动提取 CSV -> {merged_file.name}')
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Download ERA5 monthly files for a lake, merge them, and optionally extract CSV files.')
    parser.add_argument('--lake', default='mendota', help='Preset lake name, for example mendota or acton')
    parser.add_argument('--year', default='2018', help='Target year, for example 2018')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help='Override bbox as north west south east')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading and reuse existing monthly files')
    parser.add_argument('--skip-extract', action='store_true', help='Skip CSV extraction after merge')
    parser.add_argument('--extract-output-dir', help='Optional directory for extracted CSV files')
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
    monthly_template, merged_file = build_paths(lake_slug, args.year)

    return display_name, lake_slug, bbox, monthly_template, merged_file


def main():
    args = parse_args()
    display_name, lake_slug, bbox, monthly_template, merged_file = resolve_lake_config(args)
    RAW_ERA5_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client() if not args.skip_download else None

    print(f'开始处理 {display_name} 湖 {args.year} 全年 ERA5 数据')
    print(f'使用范围: {bbox}')

    if not args.skip_download:
        download_year_data(client, display_name, lake_slug, bbox, args.year, monthly_template)
    else:
        print('已跳过下载步骤，直接使用现有月文件')

    check_missing_months(monthly_template, args.year)
    merged_result = merge_year_data(monthly_template, merged_file, args.year)

    if merged_result is not None and not args.skip_extract:
        extract_output_dir = Path(args.extract_output_dir).expanduser().resolve() if args.extract_output_dir else None
        if extract_output_dir is not None:
            extract_output_dir.mkdir(parents=True, exist_ok=True)
        extract_merged_data(merged_result, extract_output_dir)
    elif args.skip_extract:
        print('已跳过自动提取步骤')

    print('全部任务完成')


if __name__ == '__main__':
    main()
