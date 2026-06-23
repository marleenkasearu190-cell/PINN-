"""Diagnostic summary writers for LakePINN rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .physics import water_density_numpy


HEAT_CLOSURE_COLUMNS = (
    'surface_flux_wm2',
    'open_water_surface_flux_wm2',
    'open_water_net_radiation_wm2',
    'open_water_sensible_heat_wm2',
    'open_water_latent_heat_wm2',
    'open_water_sensible_heat_bulk_wm2',
    'open_water_latent_heat_bulk_wm2',
    'shortwave_wm2',
    'shortwave_to_water_wm2',
    'ice_shortwave_transmission',
    'surface_flux_bias_wm2',
    'heat_input_wm2',
    'heat_tendency_wm2',
    'sensible_heat_tendency_wm2',
    'effective_heat_tendency_wm2',
    'energy_residual_wm2',
    'freezing_storage_j_m2',
    'freezing_storage_change_wm2',
    'temperature_floor_heat_injection_wm2',
    'temperature_ceiling_heat_removal_wm2',
)


def _finite_stats(values):
    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'sum': np.nan,
        }
    finite_values = values[finite]
    return {
        'mean': float(np.mean(finite_values)),
        'min': float(np.min(finite_values)),
        'max': float(np.max(finite_values)),
        'sum': float(np.sum(finite_values)),
    }


def _heat_summary_record(frame, period, month=None):
    record = {
        'period': period,
        'month': month,
        'n_days': int(len(frame)),
        'date_start': pd.Timestamp(frame['Date'].min()).date().isoformat() if len(frame) else '',
        'date_end': pd.Timestamp(frame['Date'].max()).date().isoformat() if len(frame) else '',
    }
    for column in HEAT_CLOSURE_COLUMNS:
        stats = _finite_stats(frame[column] if column in frame.columns else [])
        for stat_name, value in stats.items():
            record[f'{column}_{stat_name}'] = value
    return record


def write_heat_closure_summaries(diagnostics_df, metadata, output_dir, suffix):
    """Write monthly and annual heat-closure summaries without failing on sparse data."""
    output_dir = Path(output_dir)
    file_tag = str(metadata.get('file_tag') or metadata.get('lake_id') or 'lake')
    monthly_csv = output_dir / f'{file_tag}_{suffix}_heat_closure_monthly_summary.csv'
    annual_csv = output_dir / f'{file_tag}_{suffix}_heat_closure_annual_summary.csv'

    frame = pd.DataFrame(diagnostics_df).copy()
    if 'Date' not in frame.columns:
        pd.DataFrame().to_csv(monthly_csv, index=False)
        pd.DataFrame([_heat_summary_record(pd.DataFrame({'Date': []}), 'annual')]).to_csv(annual_csv, index=False)
        return {
            'heat_closure_monthly_summary': monthly_csv,
            'heat_closure_annual_summary': annual_csv,
        }

    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame = frame.dropna(subset=['Date'])
    for column in HEAT_CLOSURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors='coerce')

    if frame.empty:
        monthly = pd.DataFrame()
        annual = pd.DataFrame([_heat_summary_record(pd.DataFrame({'Date': []}), 'annual')])
    else:
        frame['month'] = frame['Date'].dt.month
        monthly = pd.DataFrame(
            _heat_summary_record(group, f'month_{int(month):02d}', int(month))
            for month, group in frame.groupby('month', sort=True)
        )
        annual = pd.DataFrame([_heat_summary_record(frame, 'annual')])

    monthly.to_csv(monthly_csv, index=False)
    annual.to_csv(annual_csv, index=False)
    return {
        'heat_closure_monthly_summary': monthly_csv,
        'heat_closure_annual_summary': annual_csv,
    }


def _density_fraction_by_day(rho_drop, threshold):
    if rho_drop.size == 0:
        return np.zeros(0, dtype=np.float64)
    valid = np.isfinite(rho_drop)
    unstable = rho_drop > float(threshold)
    fractions = np.full(rho_drop.shape[0], np.nan, dtype=np.float64)
    for idx in range(rho_drop.shape[0]):
        if np.any(valid[idx]):
            fractions[idx] = float(np.mean(unstable[idx, valid[idx]]))
    return fractions


def _density_summary_record(period, month, dates, strong_frac, weak_frac):
    strong_valid = strong_frac[np.isfinite(strong_frac)]
    weak_valid = weak_frac[np.isfinite(weak_frac)]
    return {
        'period': period,
        'month': month,
        'n_days': int(len(dates)),
        'valid_profile_days': int(strong_valid.size),
        'date_start': pd.Timestamp(dates.min()).date().isoformat() if len(dates) else '',
        'date_end': pd.Timestamp(dates.max()).date().isoformat() if len(dates) else '',
        'density_unstable_days': int(np.sum(strong_valid > 0.20)) if strong_valid.size else 0,
        'max_density_unstable_layer_frac': float(np.max(strong_valid)) if strong_valid.size else np.nan,
        'mean_density_unstable_layer_frac': float(np.mean(strong_valid)) if strong_valid.size else np.nan,
        'weak_density_unstable_days': int(np.sum(weak_valid > 0.20)) if weak_valid.size else 0,
        'max_weak_density_unstable_layer_frac': float(np.max(weak_valid)) if weak_valid.size else np.nan,
        'mean_weak_density_unstable_layer_frac': float(np.mean(weak_valid)) if weak_valid.size else np.nan,
    }


def write_density_stability_summary(
    temp_grid,
    depths,
    dates,
    metadata,
    output_dir,
    suffix,
    *,
    density_inversion_drop_kgm3=0.02,
):
    """Write annual and monthly density-stability metrics for a depth-time grid."""
    output_dir = Path(output_dir)
    file_tag = str(metadata.get('file_tag') or metadata.get('lake_id') or 'lake')
    summary_csv = output_dir / f'{file_tag}_{suffix}_density_stability_summary.csv'

    temp_grid = np.asarray(temp_grid, dtype=np.float64)
    if temp_grid.ndim != 2 or temp_grid.shape[0] < 2 or temp_grid.shape[1] == 0:
        pd.DataFrame([
            _density_summary_record('annual', None, pd.DatetimeIndex([]), np.array([]), np.array([])),
        ]).to_csv(summary_csv, index=False)
        return {'density_stability_summary': summary_csv}

    dates = pd.to_datetime(pd.Series(dates), errors='coerce')
    if len(dates) != temp_grid.shape[1]:
        dates = pd.date_range('1970-01-01', periods=temp_grid.shape[1], freq='D')
    else:
        dates = pd.DatetimeIndex(dates)
    valid_date_mask = ~pd.isna(dates)
    dates = pd.DatetimeIndex(dates[valid_date_mask])
    temp_by_day = temp_grid[:, np.asarray(valid_date_mask, dtype=bool)].T
    if temp_by_day.size == 0:
        pd.DataFrame([
            _density_summary_record('annual', None, pd.DatetimeIndex([]), np.array([]), np.array([])),
        ]).to_csv(summary_csv, index=False)
        return {'density_stability_summary': summary_csv}

    rho = water_density_numpy(temp_by_day)
    rho_drop = -np.diff(rho, axis=1)
    strong_frac = _density_fraction_by_day(rho_drop, density_inversion_drop_kgm3)
    weak_frac = _density_fraction_by_day(rho_drop, 0.005)

    records = [_density_summary_record('annual', None, dates, strong_frac, weak_frac)]
    months = pd.Series(dates.month)
    for month in sorted(months.dropna().unique()):
        mask = months.to_numpy() == month
        records.append(
            _density_summary_record(
                f'month_{int(month):02d}',
                int(month),
                dates[mask],
                strong_frac[mask],
                weak_frac[mask],
            )
        )

    pd.DataFrame(records).to_csv(summary_csv, index=False)
    return {'density_stability_summary': summary_csv}
