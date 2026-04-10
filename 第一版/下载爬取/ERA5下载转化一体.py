import argparse
import base64
import calendar
import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cdsapi
import numpy as np
import pandas as pd
from requests.exceptions import SSLError
import xarray as xr


DATASET = 'reanalysis-era5-single-levels'
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_ERA5_DIR = BASE_DIR / '数据' / '原始' / 'ERA5数据'
PROCESSED_ERA5_DIR = BASE_DIR / '数据' / '处理后' / 'ERA5数据'
RAW_LST_DIR = BASE_DIR / '数据' / '原始' / 'LST数据'
PROCESSED_LST_DIR = BASE_DIR / '数据' / '处理后' / 'LST数据'

REQUEST_GROUPS = {
    'instant': [
        'lake_mix_layer_depth',
        'lake_bottom_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
    ],
    'accum': [
        'surface_solar_radiation_downwards',
    ],
}
REQUIRED_SHORT_NAMES = {'lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m'}
VARIABLE_LIST = ['lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m']
COLUMN_ORDER = [
    'Date', 'lmld_m', 'lblt_K', 'lblt_C', 'Is_J_per_m2',
    'u10_m_per_s', 'v10_m_per_s', 'wind_norm_m_per_s',
    't2m_K', 't2m_C',
]
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 90

APPEEARS_API = 'https://appeears.earthdatacloud.nasa.gov/api/'
MODIS_PRODUCT = 'MOD11A1.061'
MODIS_LAYERS = ['LST_Day_1km', 'QC_Day']

LAKE_PRESETS = {
    'mendota': {
        'display_name': 'Mendota',
        'bbox': [43.25, -89.50, 43.00, -89.25],
        'lat': 43.0997,
        'lon': -89.4127,
    },
    'acton': {
        'display_name': 'Acton',
        'bbox': [39.59, -84.00, 39.52, -83.90],
        'lat': 39.57,
        'lon': -84.74,
    },
    'mohonk': {
        'display_name': 'Mohonk',
        'bbox': [41.90, -74.30, 41.65, -74.00],
        'lat': 41.77196,
        'lon': -74.15366,
    },
}


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def build_client():
    return cdsapi.Client()


def build_paths(lake_slug: str, year: str):
    monthly_template = f'ERA5_{lake_slug}_{{year}}_{{month}}'
    merged_file = RAW_ERA5_DIR / f'ERA5_{lake_slug}_{year}_full.nc'
    hourly_file = PROCESSED_ERA5_DIR / f'ERA5_{lake_slug}_{year}_Hourly.csv'
    daily_file = PROCESSED_ERA5_DIR / f'ERA5_{lake_slug}_{year}_Daily.csv'
    lst_raw_dir = RAW_LST_DIR / lake_slug
    lst_processed_file = PROCESSED_LST_DIR / f'{lake_slug}-LST-{year}-MOD11A1-061-results.csv'
    return monthly_template, merged_file, hourly_file, daily_file, lst_raw_dir, lst_processed_file


def month_file(month: int, monthly_template: str, year: str, group: str | None = None) -> Path:
    stem = monthly_template.format(year=year, month=f'{month:02d}')
    if group:
        stem = f'{stem}_{group}'
    return RAW_ERA5_DIR / f'{stem}.nc'


def file_contains_vars(file_path: Path, required_vars) -> bool:
    required_vars = set(required_vars)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False

    try:
        ds = load_month_dataset(file_path)
        try:
            return required_vars.issubset(set(ds.data_vars))
        finally:
            ds.close()
    except Exception:
        return False


def archive_contains_required_vars(archive_path: Path) -> bool:
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        return False

    try:
        ds = load_month_dataset(archive_path)
        try:
            return REQUIRED_SHORT_NAMES.issubset(set(ds.data_vars))
        finally:
            ds.close()
    except Exception:
        return False


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, SSLError) or any(
        token in message
        for token in [
            'ssl',
            'unexpected eof',
            'httpsconnectionpool',
            'max retries exceeded',
            'proxyerror',
            'remotedisconnected',
            'temporarily unavailable',
            'connection aborted',
            'connection reset',
        ]
    )


def download_year_data(lake_name: str, bbox, year: str, monthly_template: str, max_retries: int, retry_wait_seconds: int):
    for month in range(1, 13):
        month_str = f'{month:02d}'
        days_in_month = calendar.monthrange(int(year), month)[1]
        for group_name, variables in REQUEST_GROUPS.items():
            output_file = month_file(month, monthly_template, year, group_name)
            expected_short_names = {'ssrd'} if group_name == 'accum' else {'lmld', 'lblt', 'u10', 'v10', 't2m'}

            if file_contains_vars(output_file, expected_short_names):
                print(f'跳过 {output_file.name}，文件已包含所需变量')
                continue

            print(f'开始下载 {lake_name} 湖 {year} 年 {month_str} 月 {group_name} 数据 -> {output_file.name}')
            request = {
                'product_type': ['reanalysis'],
                'variable': variables,
                'year': [year],
                'month': [month_str],
                'day': [f'{day:02d}' for day in range(1, days_in_month + 1)],
                'time': [f'{hour:02d}:00' for hour in range(24)],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'area': bbox,
            }

            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    if output_file.exists() and not file_contains_vars(output_file, expected_short_names):
                        output_file.unlink()

                    print(f'尝试 {attempt}/{max_retries}')
                    client = build_client()
                    client.retrieve(DATASET, request, str(output_file))

                    if not file_contains_vars(output_file, expected_short_names):
                        raise RuntimeError(f'{output_file.name} 下载后变量仍不完整')

                    print(f'{output_file.name} 下载完成')
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f'{output_file.name} 下载失败: {exc}')
                    if output_file.exists() and not file_contains_vars(output_file, expected_short_names):
                        try:
                            output_file.unlink()
                        except Exception:
                            pass
                    if attempt < max_retries:
                        wait_seconds = retry_wait_seconds * attempt if is_retryable_error(exc) else retry_wait_seconds
                        print(f'{wait_seconds} 秒后重试...')
                        time.sleep(wait_seconds)

            if last_error is not None:
                raise last_error


def check_missing_months(monthly_template: str, year: str):
    missing = []
    incomplete = []
    for month in range(1, 13):
        month_missing = False
        month_incomplete = False
        for group_name in REQUEST_GROUPS:
            file_path = month_file(month, monthly_template, year, group_name)
            expected_short_names = {'ssrd'} if group_name == 'accum' else {'lmld', 'lblt', 'u10', 'v10', 't2m'}
            if not file_path.exists() or file_path.stat().st_size == 0:
                month_missing = True
            elif not file_contains_vars(file_path, expected_short_names):
                month_incomplete = True
        if month_missing:
            missing.append(f'{month:02d}')
        elif month_incomplete:
            incomplete.append(f'{month:02d}')

    if missing:
        print(f'缺失月份: {", ".join(missing)}')
    if incomplete:
        print(f'变量不完整的月份: {", ".join(incomplete)}')
    if not missing and not incomplete:
        print('12 个月份数据齐全，且包含全部所需变量')

    return missing + incomplete


def load_month_dataset(month_path: Path) -> xr.Dataset:
    try:
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
    except zipfile.BadZipFile:
        ds, temp_dir = open_dataset_with_fallback(month_path)
        try:
            month_ds = normalize_dataset(ds).load()
        finally:
            ds.close()
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    if 'valid_time' in month_ds.dims or 'valid_time' in month_ds.coords:
        month_ds = month_ds.rename({'valid_time': 'time'})

    return month_ds


def merge_year_data(monthly_template: str, merged_file: Path, year: str):
    missing = check_missing_months(monthly_template, year)
    if missing:
        raise RuntimeError('存在缺失或不完整月份，无法合并全年数据。')

    print(f'开始合并全年数据 -> {merged_file.name}')
    datasets = []
    for month in range(1, 13):
        month_parts = [load_month_dataset(month_file(month, monthly_template, year, group_name)) for group_name in REQUEST_GROUPS]
        try:
            month_ds = xr.merge(month_parts, compat='override', combine_attrs='drop_conflicts')
            month_ds = month_ds.load()
        finally:
            for ds in month_parts:
                ds.close()
        datasets.append(month_ds)
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
                print(f'使用临时英文路径打开成功，engine={engine}')
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


def http_json(
    url: str,
    method: str = 'GET',
    payload: dict | None = None,
    headers: dict | None = None,
    auth: tuple[str, str] | None = None,
):
    body = None
    req_headers = {'User-Agent': 'Mozilla/5.0'}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode('utf-8')
        req_headers['Content-Type'] = 'application/json'
    if auth is not None:
        raw = f'{auth[0]}:{auth[1]}'.encode('utf-8')
        req_headers['Authorization'] = 'Basic ' + base64.b64encode(raw).decode('ascii')
    request = Request(url, data=body, headers=req_headers, method=method)
    with urlopen(request, timeout=120) as response:
        text = response.read().decode('utf-8', errors='replace')
    return json.loads(text)


def download_binary(url: str, target: Path, headers: dict | None = None) -> None:
    req_headers = {'User-Agent': 'Mozilla/5.0'}
    if headers:
        req_headers.update(headers)
    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=300) as response, target.open('wb') as fh:
        fh.write(response.read())


def get_earthdata_credentials(args) -> tuple[str | None, str | None]:
    username = args.earthdata_username or args.earthdata_user
    password = args.earthdata_password or args.earthdata_pass
    return username, password


def download_lst_data(
    args,
    display_name: str,
    lake_slug: str,
    year: str,
    lat: float,
    lon: float,
    lst_raw_dir: Path,
    lst_processed_file: Path,
):
    username, password = get_earthdata_credentials(args)
    if not username or not password:
        print('未检测到 Earthdata 账号，已跳过 LST 下载。')
        print('请设置 EARTHDATA_USERNAME 和 EARTHDATA_PASSWORD，或传入 --earthdata-user 和 --earthdata-pass。')
        return []

    lst_raw_dir.mkdir(parents=True, exist_ok=True)
    token_response = http_json(f'{APPEEARS_API}login', method='POST', auth=(username, password))
    token = token_response.get('token')
    if not token:
        raise RuntimeError(f'AppEEARS 登录未返回 token: {token_response}')
    bearer_headers = {'Authorization': f'Bearer {token}'}

    task_name = f'{lake_slug}_mod11a1_{year}'
    payload = {
        'task_type': 'point',
        'task_name': task_name,
        'params': {
            'dates': [{'startDate': f'01-01-{year}', 'endDate': f'12-31-{year}'}],
            'layers': [{'product': MODIS_PRODUCT, 'layer': layer} for layer in MODIS_LAYERS],
            'output': {'format': {'type': 'csv'}},
            'geo': {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [lon, lat],
                        },
                        'properties': {'id': lake_slug},
                    }
                ],
            },
        },
    }

    task_response = http_json(f'{APPEEARS_API}task', method='POST', payload=payload, headers=bearer_headers)
    task_id = task_response.get('task_id') or task_response.get('taskid') or task_response.get('id')
    if not task_id:
        raise RuntimeError(f'AppEEARS 任务提交返回异常: {task_response}')
    print(f'已提交 {display_name} 的 LST 任务: {task_id}')

    status = None
    for _ in range(180):
        task_status = http_json(f'{APPEEARS_API}task/{task_id}', headers=bearer_headers)
        status = (task_status.get('status') or task_status.get('state') or '').lower()
        print(f'AppEEARS 状态: {status}')
        if status in {'done', 'complete', 'completed', 'success', 'successful'}:
            break
        if status in {'failed', 'error'}:
            raise RuntimeError(f'AppEEARS 任务失败: {task_status}')
        time.sleep(20)
    else:
        raise TimeoutError(f'AppEEARS 任务 {task_id} 超时未完成。')

    bundle_response = http_json(f'{APPEEARS_API}bundle/{task_id}', headers=bearer_headers)
    if isinstance(bundle_response, dict) and 'files' in bundle_response:
        files = bundle_response['files']
    elif isinstance(bundle_response, list):
        files = bundle_response
    else:
        files = bundle_response.get('bundle', []) if isinstance(bundle_response, dict) else []

    downloaded = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        file_name = (
            entry.get('file_name')
            or entry.get('fileName')
            or entry.get('name')
            or Path(urlparse(entry.get('url', '')).path).name
        )
        if not file_name:
            continue
        download_url = entry.get('url') or f'{APPEEARS_API}bundle/{task_id}/{file_name}'
        target = lst_raw_dir / file_name
        print(f'下载 AppEEARS 文件: {file_name}')
        download_binary(download_url, target, headers=bearer_headers)
        downloaded.append(target)

    processed = []
    for file_path in downloaded:
        if file_path.suffix.lower() != '.csv':
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if 'MOD11A1_061_LST_Day_1km' in df.columns:
            saved = save_csv_with_fallback(df, lst_processed_file)
            processed.append(saved)

    if processed:
        print('LST 处理后文件:')
        for path in processed:
            print(path)
    else:
        print('没有识别到标准 MOD11A1 LST CSV，但原始 AppEEARS 文件已下载完成。')

    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description='下载湖泊 ERA5 月数据，合并提取为标准 CSV，并可选下载 MODIS LST。'
    )
    parser.add_argument('--lake', default=None, help='预设湖泊名，例如 mendota、acton、mohonk')
    parser.add_argument('--display-name', help='可选显示名称')
    parser.add_argument('--year', default=None, help='目标年份，例如 2018')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help='自定义 ERA5 范围')
    parser.add_argument('--lat', type=float, help='LST 点位纬度')
    parser.add_argument('--lon', type=float, help='LST 点位经度')
    parser.add_argument('--skip-download', action='store_true', help='跳过 ERA5 月数据下载')
    parser.add_argument('--skip-merge', action='store_true', help='跳过 ERA5 合并')
    parser.add_argument('--download-only', action='store_true', help='只运行下载步骤，跳过 ERA5 合并和提取')
    parser.add_argument('--skip-era5', action='store_true', help='跳过 ERA5 流程')
    parser.add_argument('--skip-lst', action='store_true', help='跳过 MODIS LST 下载')
    parser.add_argument('--merged-file', help='指定已有的全年 nc 文件')
    parser.add_argument('--output-dir', help='指定 merged nc、CSV 和处理后 LST 的输出目录')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help='每个月份分组下载的最大重试次数')
    parser.add_argument('--retry-wait-seconds', type=int, default=RETRY_WAIT_SECONDS, help='下载失败后的基础等待秒数')
    parser.add_argument('--earthdata-user', dest='earthdata_user', default=None, help='NASA Earthdata 用户名')
    parser.add_argument('--earthdata-pass', dest='earthdata_pass', default=None, help='NASA Earthdata 密码')
    parser.add_argument('--earthdata-username', default=None, help='NASA Earthdata 用户名别名')
    parser.add_argument('--earthdata-password', default=None, help='NASA Earthdata 密码别名')
    return parser.parse_args()


def prompt_text(label: str, default: str | None = None) -> str:
    while True:
        suffix = f' [{default}]' if default else ''
        value = input(f'{label}{suffix}: ').strip()
        if value:
            return value
        if default is not None:
            return default
        print(f'{label} 不能为空，请重新输入。')


def prompt_float(label: str, default: float | None = None) -> float:
    while True:
        suffix = f' [{default}]' if default is not None else ''
        raw = input(f'{label}{suffix}: ').strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print(f'{label} 需要输入数字，请重新输入。')


def prompt_bbox(default_bbox=None):
    while True:
        default_text = ''
        if default_bbox is not None:
            default_text = ' [' + ' '.join(str(value) for value in default_bbox) + ']'
        raw = input(f'请输入 bbox (NORTH WEST SOUTH EAST){default_text}: ').strip()
        if not raw and default_bbox is not None:
            return [float(value) for value in default_bbox]
        parts = raw.replace(',', ' ').split()
        if len(parts) != 4:
            print('bbox 需要输入 4 个数字，例如: 41.90 -74.30 41.65 -74.00')
            continue
        try:
            return [float(value) for value in parts]
        except ValueError:
            print('bbox 中包含非数字，请重新输入。')


def prompt_lake_config(args):
    known = ', '.join(sorted(LAKE_PRESETS))
    if not args.lake:
        args.lake = prompt_text(f'请输入湖泊名称（预设: {known}；自定义也可直接输入）')
    if not args.year:
        args.year = prompt_text('请输入年份', '2018')

    lake_key = slugify(args.lake)
    preset = LAKE_PRESETS.get(lake_key)

    if preset is None and args.display_name is None:
        args.display_name = prompt_text('请输入湖泊显示名称', args.lake)

    if preset is None and args.bbox is None:
        args.bbox = prompt_bbox()

    if not args.skip_lst and preset is None:
        if args.lat is None:
            args.lat = prompt_float('请输入湖泊纬度 lat')
        if args.lon is None:
            args.lon = prompt_float('请输入湖泊经度 lon')

    return args


def resolve_lake_config(args):
    lake_key = slugify(args.lake)
    preset = LAKE_PRESETS.get(lake_key)

    if preset is None and args.bbox is None:
        known = ', '.join(sorted(LAKE_PRESETS))
        raise ValueError(f'未知湖泊预设: {args.lake}。请使用已知预设 ({known})，或通过 --bbox 指定范围。')

    display_name = args.display_name or (preset['display_name'] if preset else args.lake)
    bbox = args.bbox if args.bbox is not None else preset['bbox']
    lat = args.lat if args.lat is not None else (preset.get('lat') if preset else None)
    lon = args.lon if args.lon is not None else (preset.get('lon') if preset else None)

    if not args.skip_lst and (lat is None or lon is None):
        raise ValueError('LST 下载需要点位坐标，请提供 --lat 和 --lon，或使用带坐标的湖泊预设。')

    lake_slug = lake_key or slugify(display_name)
    return display_name, lake_slug, bbox, lat, lon


def main():
    args = parse_args()
    args = prompt_lake_config(args)
    display_name, lake_slug, bbox, lat, lon = resolve_lake_config(args)

    RAW_ERA5_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ERA5_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_LST_DIR.mkdir(parents=True, exist_ok=True)

    monthly_template, default_merged_file, default_hourly_file, default_daily_file, lst_raw_dir, default_lst_file = build_paths(lake_slug, args.year)
    lst_raw_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_file = output_dir / default_merged_file.name
        hourly_file = output_dir / default_hourly_file.name
        daily_file = output_dir / default_daily_file.name
        lst_file = output_dir / default_lst_file.name
    else:
        merged_file = default_merged_file
        hourly_file = default_hourly_file
        daily_file = default_daily_file
        lst_file = default_lst_file

    if args.merged_file:
        merged_file = Path(args.merged_file).expanduser().resolve()

    print(f'开始处理 {display_name} 湖 {args.year} 数据')
    print(f'ERA5 范围: {bbox}')
    if not args.skip_lst:
        print(f'LST 点位: ({lat}, {lon})')

    if not args.skip_era5:
        if not args.skip_download and not args.merged_file:
            download_year_data(display_name, bbox, args.year, monthly_template, args.max_retries, args.retry_wait_seconds)
        elif args.skip_download:
            print('已跳过 ERA5 下载步骤，直接使用现有月文件或全年文件')
    else:
        print('已按参数跳过 ERA5 流程')

    if not args.skip_lst:
        download_lst_data(args, display_name, lake_slug, args.year, lat, lon, lst_raw_dir, lst_file)
    else:
        print('已按参数跳过 LST 下载')

    if args.download_only:
        print('已完成下载步骤，按参数设置跳过 ERA5 合并和提取。')
        return

    if not args.skip_era5:
        if args.skip_merge or args.merged_file:
            if not merged_file.exists():
                raise FileNotFoundError(f'未找到全年文件: {merged_file}')
            print(f'已跳过 ERA5 合并步骤，直接使用: {merged_file.name}')
        else:
            merge_year_data(monthly_template, merged_file, args.year)

        extract_csv_files(merged_file, hourly_file, daily_file)

    print('全部任务完成')


if __name__ == '__main__':
    main()
