import argparse
import base64
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

LAKE_PRESETS = {
    'mohonk': {
        'display_name': 'Mohonk Lake',
        'lat': 41.77196,
        'lon': -74.15366,
    },
    'acton': {
        'display_name': 'Acton Lake',
        'lat': 39.57,
        'lon': -84.74,
    },
    'mendota': {
        'display_name': 'Lake Mendota',
        'lat': 43.0997,
        'lon': -89.4127,
    },
}

APPEEARS_API = 'https://appeears.earthdatacloud.nasa.gov/api/'
MODIS_PRODUCT = 'MOD11A1.061'
MODIS_LAYERS = ['LST_Day_1km', 'QC_Day']


def sanitize_name(text: str) -> str:
    return ''.join(char.lower() if char.isalnum() else '_' for char in text).strip('_')


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


def prompt_lake_config(args):
    known = ', '.join(sorted(LAKE_PRESETS))
    if not args.lake:
        args.lake = prompt_text(f'请输入湖泊名称（预设: {known}；自定义也可直接输入）')
    if not args.year:
        args.year = prompt_text('请输入年份', '2017')

    preset = LAKE_PRESETS.get(args.lake.lower())
    if args.display_name is None:
        args.display_name = f'{args.lake}_{args.year}_lst'
    if preset is None:
        if args.lat is None:
            args.lat = prompt_float('请输入湖泊纬度 lat')
        if args.lon is None:
            args.lon = prompt_float('请输入湖泊经度 lon')
    return args


def build_config(args):
    preset = LAKE_PRESETS.get(args.lake.lower()) if args.lake else None
    default_display_name = f'{args.lake}_{args.year}_lst' if args.lake and args.year else None
    display_name = args.display_name or default_display_name or (preset['display_name'] if preset else args.lake)
    if not display_name:
        raise ValueError('请提供湖泊名称或显示名称。')

    lat = args.lat if args.lat is not None else (preset['lat'] if preset else None)
    lon = args.lon if args.lon is not None else (preset['lon'] if preset else None)
    if lat is None or lon is None:
        raise ValueError('自定义湖泊请提供 --lat 和 --lon。')

    lake_slug = sanitize_name(args.lake or display_name)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (PROJECT_DIR / lake_slug)
    return {
        'lake_slug': lake_slug,
        'display_name': display_name,
        'year': str(args.year),
        'lat': float(lat),
        'lon': float(lon),
        'lst_processed_dir': output_root,
    }


def ensure_dirs(config):
    config['lst_processed_dir'].mkdir(parents=True, exist_ok=True)


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


def download_lst(args, config) -> list[Path]:
    username, password = get_earthdata_credentials(args)
    if not username or not password:
        print('未检测到 Earthdata 账号，已跳过 LST 下载。')
        print('请设置 EARTHDATA_USERNAME 和 EARTHDATA_PASSWORD，或传入 --earthdata-user 和 --earthdata-pass。')
        return []

    token_response = http_json(f'{APPEEARS_API}login', method='POST', auth=(username, password))
    token = token_response.get('token')
    if not token:
        raise RuntimeError(f'AppEEARS 登录未返回 token: {token_response}')
    bearer_headers = {'Authorization': f'Bearer {token}'}

    task_name = f"{config['lake_slug']}_mod11a1_{config['year']}"
    payload = {
        'task_type': 'point',
        'task_name': task_name,
        'params': {
            'dates': [{'startDate': f"01-01-{config['year']}", 'endDate': f"12-31-{config['year']}"}],
            'layers': [{'product': MODIS_PRODUCT, 'layer': layer} for layer in MODIS_LAYERS],
            'output': {'format': {'type': 'csv'}},
            'geo': {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [config['lon'], config['lat']],
                        },
                        'properties': {'id': config['lake_slug']},
                    }
                ],
            },
        },
    }

    task_response = http_json(f'{APPEEARS_API}task', method='POST', payload=payload, headers=bearer_headers)
    task_id = task_response.get('task_id') or task_response.get('taskid') or task_response.get('id')
    if not task_id:
        raise RuntimeError(f'AppEEARS 任务提交返回异常: {task_response}')
    print(f'AppEEARS 任务已提交: {task_id}')

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

    processed = []
    with tempfile.TemporaryDirectory(prefix='lst_bundle_') as temp_dir:
        temp_dir_path = Path(temp_dir)
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
            target = temp_dir_path / file_name
            print(f'下载 AppEEARS 文件: {file_name}')
            download_binary(download_url, target, headers=bearer_headers)

            if target.suffix.lower() != '.csv':
                continue
            try:
                df = pd.read_csv(target)
            except Exception:
                continue
            if 'MOD11A1_061_LST_Day_1km' in df.columns:
                output_path = config['lst_processed_dir'] / f"{config['lake_slug']}-LST-{config['year']}-MOD11A1-061-results.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                processed.append(output_path)

    if processed:
        print('处理后的 LST 文件:')
        for path in processed:
            print(path)
    else:
        print('没有识别到标准 MOD11A1 LST CSV。')

    return processed


def main():
    parser = argparse.ArgumentParser(description='下载指定湖泊的 MODIS LST 数据。')
    parser.add_argument('--lake', default=None, help='预设湖泊名或自定义湖泊名。')
    parser.add_argument('--display-name', default=None, help='显示名称。')
    parser.add_argument('--year', default=None, help='目标年份。')
    parser.add_argument('--lat', type=float, default=None, help='自定义湖泊纬度。')
    parser.add_argument('--lon', type=float, default=None, help='自定义湖泊经度。')
    parser.add_argument('--output-root', default=None, help='处理后 CSV 的保存目录，默认是脚本目录/<lake>。')
    parser.add_argument('--earthdata-user', dest='earthdata_user', default=None, help='NASA Earthdata 用户名。')
    parser.add_argument('--earthdata-pass', dest='earthdata_pass', default=None, help='NASA Earthdata 密码。')
    parser.add_argument('--earthdata-username', default=None, help='NASA Earthdata 用户名别名。')
    parser.add_argument('--earthdata-password', default=None, help='NASA Earthdata 密码别名。')
    args = parser.parse_args()

    args = prompt_lake_config(args)
    config = build_config(args)
    ensure_dirs(config)
    print(f"Lake: {config['display_name']}")
    print(f"Year: {config['year']}")
    print(f"Point: ({config['lat']}, {config['lon']})")

    download_lst(args, config)


if __name__ == '__main__':
    main()
