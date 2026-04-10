import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = ['lmld', 'lblt', 'ssrd', 'u10', 'v10', 't2m']
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / '数据' / '处理后' / 'ERA5数据'
COLUMN_ORDER = [
    'Date', 'lmld_m', 'lblt_K', 'lblt_C', 'Is_J_per_m2',
    'u10_m_per_s', 'v10_m_per_s', 'wind_norm_m_per_s',
    't2m_K', 't2m_C'
]


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


def infer_prefix(input_file: Path) -> str:
    stem = input_file.stem
    for suffix in ['_full', '_merged', '_hourly', '_daily']:
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def main():
    parser = argparse.ArgumentParser(description='Extract any lake ERA5 NetCDF into cleaned hourly/daily CSV files.')
    parser.add_argument('--input', required=True, help='Path to the lake NetCDF file')
    parser.add_argument('--output-dir', help='Directory to save CSV files; default is next to input file')
    parser.add_argument('--prefix', help='Output filename prefix; default is inferred from input filename')
    args = parser.parse_args()

    input_file = Path(args.input).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f'输入文件不存在: {input_file}')

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or infer_prefix(input_file)
    output_daily_file = output_dir / f'{prefix}_Daily.csv'
    output_hourly_file = output_dir / f'{prefix}_Hourly.csv'

    print(f'1. 读取全年文件: {input_file}')
    ds, temp_dir = open_dataset_with_fallback(input_file)

    try:
        ds = normalize_dataset(ds)
        missing_vars = [var for var in VARIABLES if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(
                '全年文件缺少以下变量，无法提取标准 CSV: ' + ', '.join(missing_vars)
            )

        print('2. 对湖区网格做空间平均...')
        spatial_dims = [dim for dim in ['latitude', 'longitude'] if dim in ds[VARIABLES[0]].dims]
        if spatial_dims:
            ds_spatial_mean = ds[VARIABLES].mean(dim=spatial_dims)
        else:
            ds_spatial_mean = ds[VARIABLES]

        print('3. 导出小时尺度 CSV...')
        hourly_df = build_dataframe(ds_spatial_mean)
        hourly_df = hourly_df[COLUMN_ORDER]
        saved_hourly = save_csv_with_fallback(hourly_df, output_hourly_file)

        print('4. 导出日尺度 CSV...')
        ds_daily = ds_spatial_mean.resample(time='1D').mean()
        daily_df = build_dataframe(ds_daily)
        daily_df = daily_df[COLUMN_ORDER]
        saved_daily = save_csv_with_fallback(daily_df, output_daily_file)

        print('\n日尺度前 3 行预览:')
        print(daily_df.head(3))
        print(f'\n小时尺度文件: {saved_hourly}')
        print(f'日尺度文件: {saved_daily}')
    finally:
        ds.close()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
