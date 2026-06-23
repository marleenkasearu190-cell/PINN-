import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.lake_metadata import (
    geographic_climate_zone,
    infer_metadata,
    is_geographic_warm_deep_lake,
    metadata_static_features,
)
from lake_pinn.state_model import STATIC_FEATURE_DIM, STATIC_FEATURE_KEYS, static_feature_array


def test_geographic_lake_regime_and_static_features():
    kinneret_like = {
        "latitude": 32.8,
        "longitude": 35.6,
        "max_depth_m": 43.0,
        "mean_depth_m": 24.0,
        "area_km2": 166.0,
        "volume_km3": 4.0,
        "elevation_m": -210.0,
        "light_extinction_kd": 0.5,
        "fetch_m": 20000.0,
        "lake_type": "reservoir",
        "residence_time_days": 365.0,
        "shoreline_length_km": 50.0,
        "catchment_area_km2": 1660.0,
        "discharge_m3_s": 100.0,
    }
    mohonk_like = {"latitude": 41.8, "longitude": -74.1, "max_depth_m": 18.5}

    assert geographic_climate_zone(kinneret_like) == "warm_subtropical"
    assert is_geographic_warm_deep_lake(kinneret_like)
    assert not is_geographic_warm_deep_lake(mohonk_like)

    features = metadata_static_features(kinneret_like, max_depth=43.0)
    assert set(features) >= set(STATIC_FEATURE_KEYS)
    assert features["reservoir_indicator"] == 1.0
    assert features["residence_time_norm"] > 0.0
    assert features["shoreline_length_norm"] > 0.0
    assert features["shoreline_development_norm"] > 0.0
    assert features["catchment_area_norm"] > 0.0
    assert features["discharge_norm"] > 0.0
    assert all(np.isfinite(float(value)) for value in features.values())
    static = static_feature_array(kinneret_like, max_depth=43.0)
    assert len(STATIC_FEATURE_KEYS) == STATIC_FEATURE_DIM
    assert static.shape == (STATIC_FEATURE_DIM,)
    assert static[STATIC_FEATURE_KEYS.index("reservoir_indicator")] == 1.0


def test_metadata_static_features_support_extended_aliases_and_defaults():
    alias_metadata = {
        "lake_type": "reservoir",
        "retention_time_days": 730.0,
        "shore_len_km": 250.0,
        "shoreline_development": 3.0,
        "watershed_area_km2": 5000.0,
        "mean_discharge_m3_s": 250.0,
    }
    features = metadata_static_features(alias_metadata, max_depth=20.0)

    assert features["reservoir_indicator"] == 1.0
    assert features["residence_time_norm"] == 2.0
    assert features["shoreline_length_norm"] == 0.5
    assert features["shoreline_development_norm"] == 0.3
    assert features["catchment_area_norm"] == 0.5
    assert features["discharge_norm"] == 0.25

    missing = metadata_static_features({}, max_depth=20.0)
    for key in (
        "reservoir_indicator",
        "residence_time_norm",
        "shoreline_length_norm",
        "shoreline_development_norm",
        "catchment_area_norm",
        "discharge_norm",
    ):
        assert missing[key] == 0.0


def test_infer_metadata_reads_sidecar_extended_aliases(tmp_path):
    era5_path = tmp_path / "forcing.csv"
    lst_path = tmp_path / "lst.csv"
    (tmp_path / "lake_metadata.json").write_text(
        json.dumps(
            {
                "lake_id": "alias_lake",
                "lake_type": "reservoir",
                "retention_time_days": 90.0,
                "shore_len_km": 40.0,
                "shoreline_development": 1.5,
                "watershed_area_km2": 300.0,
                "mean_discharge_m3_s": 12.5,
            }
        ),
        encoding="utf-8",
    )
    merged = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    lst = pd.DataFrame({"Category": ["Alias Lake"], "Date": pd.to_datetime(["2020-01-01"])})

    metadata = infer_metadata(merged, lst, era5_path, lst_path)

    assert metadata["lake_id"] == "alias_lake"
    assert metadata["lake_type"] == "reservoir"
    assert metadata["reservoir_indicator"] == 1.0
    assert metadata["residence_time_days"] == 90.0
    assert metadata["shoreline_length_km"] == 40.0
    assert metadata["shoreline_development"] == 1.5
    assert metadata["catchment_area_km2"] == 300.0
    assert metadata["discharge_m3_s"] == 12.5


if __name__ == "__main__":
    test_geographic_lake_regime_and_static_features()
    test_metadata_static_features_support_extended_aliases_and_defaults()
    print("metadata sanity checks passed")
