# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *

def export_temperature_tables(df, temp_grid, depths, output_dir, metadata, suffix=''):
    records = []
    day_axis = df['full_doy'].to_numpy()
    dates = pd.to_datetime(df['Date']).to_numpy()

    for day_idx, (date_value, doy_value) in enumerate(zip(dates, day_axis)):
        month_value = pd.Timestamp(date_value).month
        for depth_idx, depth_value in enumerate(depths):
            records.append({
                'Date': pd.Timestamp(date_value).date().isoformat(),
                'Month': month_value,
                'DOY': int(doy_value),
                'Depth_m': float(depth_value),
                'Temperature_C': float(temp_grid[depth_idx, day_idx]),
            })

    temp_df = pd.DataFrame.from_records(records)
    suffix = f"_{suffix}" if suffix else ''
    full_path = output_dir / f"{metadata['file_tag']}{suffix}_temperature_depth_predictions.csv"
    temp_df.to_csv(full_path, index=False)
    return full_path


def evaluate_profile_predictions(prediction_csv_path, profile_obs_data):
    """Evaluate predictions against profile observations using depth interpolation."""
    if not has_profile_observations(profile_obs_data):
        return None

    pred = pd.read_csv(prediction_csv_path)
    pred['Date'] = pd.to_datetime(pred['Date'])
    obs = load_optional_profile_observations(
        profile_obs_data,
        start_date=pred['Date'].min(),
        time_scale_seconds=max((pd.to_datetime(pred['Date']).max() - pd.to_datetime(pred['Date']).min()).total_seconds(), SECONDS_PER_DAY),
        max_depth=float(pred['Depth_m'].max()),
    )

    errors = []
    matched_rows = 0
    for date_value, obs_day in obs.groupby('Date'):
        pred_day = pred[pred['Date'] == date_value].sort_values('Depth_m')
        if pred_day.empty:
            continue
        pred_interp = np.interp(
            obs_day['Depth_m'].to_numpy(),
            pred_day['Depth_m'].to_numpy(),
            pred_day['Temperature_C'].to_numpy(),
        )
        errors.extend((pred_interp - obs_day['Temperature_C'].to_numpy()).tolist())
        matched_rows += len(obs_day)

    if not errors:
        return None

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'matched_rows': int(matched_rows),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'bias': float(np.mean(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
    }
