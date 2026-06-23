"""Utilities for no-profile heat-closure training windows.

This module keeps the pure scheduling logic out of ``state_multilake`` so the
multi-lake trainer can focus on tensor training and evaluation paths.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS = (1, 7, 30)


def parse_unlabeled_heat_closure_horizons(value=None, *, fallback_window_days=None) -> tuple[int, ...]:
    """Parse multi-scale heat-closure horizons from CLI/manifest values."""

    if value is None:
        raw_values = DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raw_values = (
                DEFAULT_UNLABELED_HEAT_CLOSURE_HORIZONS
                if fallback_window_days is None
                else (fallback_window_days,)
            )
        else:
            raw_values = [
                item
                for item in text.replace(';', ',').replace('|', ',').replace(' ', ',').split(',')
                if item.strip()
            ]
    elif isinstance(value, (int, np.integer, float, np.floating)):
        raw_values = (value,)
    elif isinstance(value, Iterable):
        raw_values = tuple(value)
    else:
        raw_values = (value,)

    horizons: list[int] = []
    for raw in raw_values:
        try:
            horizon = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'unlabeled_heat_closure_horizons entries must be positive integers.'
            ) from exc
        if horizon < 1:
            raise ValueError('unlabeled_heat_closure_horizons entries must be at least 1.')
        if horizon not in horizons:
            horizons.append(horizon)
    if not horizons:
        fallback = 1 if fallback_window_days is None else int(fallback_window_days)
        horizons = [max(1, fallback)]
    return tuple(sorted(horizons))


def format_unlabeled_heat_closure_horizons(horizons) -> str:
    """Format heat-closure horizons for config/history output."""

    return ','.join(str(int(value)) for value in parse_unlabeled_heat_closure_horizons(horizons))


def _date_index_map(df: pd.DataFrame) -> dict[pd.Timestamp, int]:
    date_column = 'Date' if 'Date' in df.columns else 'date'
    return {
        pd.Timestamp(date).normalize(): int(idx)
        for idx, date in enumerate(pd.to_datetime(df[date_column]))
    }


def build_unlabeled_heat_closure_windows_for_horizon(
    df: pd.DataFrame,
    profile_lookup,
    *,
    window_days: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Build no-profile windows for a single heat-budget horizon."""

    window_days = max(1, int(window_days or 1))
    date_to_index = _date_index_map(df)
    profile_indices = {
        int(date_to_index[pd.Timestamp(date).normalize()])
        for date in profile_lookup or {}
        if pd.Timestamp(date).normalize() in date_to_index
    }
    windows: list[tuple[int, int]] = []
    max_start_idx = len(df) - window_days - 1
    for start_idx in range(max(0, max_start_idx + 1)):
        end_idx = start_idx + window_days
        if any(idx in profile_indices for idx in range(start_idx, end_idx + 1)):
            continue
        windows.append((int(start_idx), int(end_idx)))
    return tuple(windows)


def build_unlabeled_heat_closure_windows_by_horizon(
    df: pd.DataFrame,
    profile_lookup,
    *,
    horizons=None,
    window_days: int = 1,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Build no-profile heat-closure windows keyed by horizon length."""

    parsed_horizons = parse_unlabeled_heat_closure_horizons(
        horizons,
        fallback_window_days=window_days,
    )
    return {
        int(horizon): build_unlabeled_heat_closure_windows_for_horizon(
            df,
            profile_lookup,
            window_days=int(horizon),
        )
        for horizon in parsed_horizons
    }


def build_unlabeled_heat_closure_windows(
    df: pd.DataFrame,
    profile_lookup,
    *,
    window_days: int = 1,
    horizons=None,
) -> tuple[tuple[int, int], ...]:
    """Build flattened no-profile windows, preserving single-horizon compatibility."""

    if horizons is None:
        return build_unlabeled_heat_closure_windows_for_horizon(
            df,
            profile_lookup,
            window_days=window_days,
        )
    by_horizon = build_unlabeled_heat_closure_windows_by_horizon(
        df,
        profile_lookup,
        horizons=horizons,
        window_days=window_days,
    )
    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for horizon in sorted(by_horizon):
        for window in by_horizon[horizon]:
            if window in seen:
                continue
            seen.add(window)
            windows.append(window)
    return tuple(windows)
