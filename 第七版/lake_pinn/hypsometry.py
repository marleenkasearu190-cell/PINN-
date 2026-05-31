"""Hypsometry helpers for area-weighted one-dimensional lake solvers."""

from __future__ import annotations

import numpy as np
import torch


def fallback_area_profile(depths, metadata=None, min_area_fraction=0.05):
    """Return a normalized area-depth curve A(z) for solver weighting.

    The first reconstruction-state MVP should not block on missing bathymetry.  When
    a real area-depth curve is absent, this function builds a smooth monotone
    fallback from maximum and mean depth metadata.  Values are normalized by
    surface area, so only relative layer volumes matter.
    """
    metadata = metadata or {}
    depths = np.asarray(depths, dtype=np.float64)
    if depths.size == 0:
        return np.asarray([], dtype=np.float32)
    max_depth = float(np.nanmax(depths))
    if not np.isfinite(max_depth) or max_depth <= 0.0:
        return np.ones_like(depths, dtype=np.float32)

    curve_depths = metadata.get('hypsometry_depth_m')
    if curve_depths is None:
        curve_depths = metadata.get('area_depth_depth_m')
    curve_areas = metadata.get('hypsometry_area_fraction')
    if curve_areas is None:
        curve_areas = metadata.get('area_depth_area_fraction')
    if curve_depths is not None and curve_areas is not None:
        try:
            curve_depths = np.asarray(curve_depths, dtype=np.float64)
            curve_areas = np.asarray(curve_areas, dtype=np.float64)
            finite = np.isfinite(curve_depths) & np.isfinite(curve_areas)
            curve_depths = curve_depths[finite]
            curve_areas = curve_areas[finite]
            if curve_depths.size >= 2 and curve_areas.size == curve_depths.size:
                order = np.argsort(curve_depths)
                area = np.interp(depths, curve_depths[order], curve_areas[order])
                area = area / max(float(np.nanmax(area)), 1.0e-6)
                return np.clip(area, min_area_fraction, 1.0).astype(np.float32)
        except (TypeError, ValueError):
            pass

    mean_depth = metadata.get('mean_depth_m', metadata.get('mean_depth', np.nan))
    try:
        mean_depth = float(mean_depth)
    except (TypeError, ValueError):
        mean_depth = np.nan
    mean_fraction = mean_depth / max_depth if np.isfinite(mean_depth) and mean_depth > 0.0 else 0.45
    # A cone has mean_depth/max_depth ~= 1/3; a box has ~= 1.  Use a power-law
    # family A(z)=(1-z/H)^p and infer a gentle shape exponent from mean depth.
    exponent = max(0.15, min(3.0, (1.0 / max(mean_fraction, 0.08)) - 1.0))
    z_norm = np.clip(depths / max_depth, 0.0, 1.0)
    area = np.maximum(1.0 - z_norm, 0.0) ** exponent
    return np.clip(area, min_area_fraction, 1.0).astype(np.float32)


def torch_area_profile(depths, metadata=None, device=None, dtype=torch.float32):
    area = fallback_area_profile(depths, metadata=metadata)
    return torch.tensor(area, dtype=dtype, device=device)
