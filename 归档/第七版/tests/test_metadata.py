import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lake_pinn.lake_metadata import (
    geographic_climate_zone,
    is_geographic_warm_deep_lake,
    metadata_static_features,
)
from lake_pinn.state_model import static_feature_array


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
    }
    mohonk_like = {"latitude": 41.8, "longitude": -74.1, "max_depth_m": 18.5}

    assert geographic_climate_zone(kinneret_like) == "warm_subtropical"
    assert is_geographic_warm_deep_lake(kinneret_like)
    assert not is_geographic_warm_deep_lake(mohonk_like)

    features = metadata_static_features(kinneret_like, max_depth=43.0)
    assert set(features) >= {"max_depth_norm", "mean_depth_norm", "log_area", "latitude", "longitude", "elevation_norm"}
    assert all(np.isfinite(float(value)) for value in features.values())
    assert static_feature_array(kinneret_like, max_depth=43.0).shape == (11,)


if __name__ == "__main__":
    test_geographic_lake_regime_and_static_features()
    print("metadata sanity checks passed")
