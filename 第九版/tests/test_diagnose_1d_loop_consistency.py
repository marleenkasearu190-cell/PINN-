import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_1d_loop_consistency import (
    _overall_status,
    _parse_int_list,
    classify_consistency,
    summarize_rows,
)


def test_classify_consistency_uses_ratio_and_delta_guards():
    assert classify_consistency(1.0, 1.2) == "pass"
    assert classify_consistency(1.0, 1.8) == "warn"
    assert classify_consistency(1.0, 2.5) == "fail"
    assert classify_consistency(float("nan"), 1.0) == "inconclusive"


def test_summarize_rows_aggregates_split_metrics():
    rows = [
        {
            "split": "checkpoint_validation",
            "transition_rmse_1d": 1.0,
            "rolling_rmse_1d": 1.2,
            "transition_count_1d": 2,
            "rolling_count_1d": 3,
        },
        {
            "split": "checkpoint_validation",
            "transition_rmse_1d": 2.0,
            "rolling_rmse_1d": 2.4,
            "transition_count_1d": 5,
            "rolling_count_1d": 7,
        },
    ]
    summary = summarize_rows(rows, horizons=(1,))
    assert len(summary) == 1
    assert summary[0]["transition_rmse_1d"] == 1.5
    assert math.isclose(summary[0]["rolling_rmse_1d"], 1.8)
    assert summary[0]["transition_count_1d"] == 7
    assert summary[0]["rolling_count_1d"] == 10
    assert summary[0]["status_1d"] == "pass"


def test_overall_status_warns_when_heldout_fails_but_validation_passes():
    status = _overall_status(
        [
            {"split": "checkpoint_validation", "status_1d": "pass"},
            {"split": "heldout_diagnostic", "status_1d": "fail"},
        ]
    )
    assert status == "warn"


def test_parse_int_list_rejects_empty_positive_horizons():
    assert _parse_int_list("7,1,3,3", default=(1,)) == (1, 3, 7)
    try:
        _parse_int_list("0,-2", default=(1,))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive horizons")


def test_summarize_rows_handles_missing_values_as_inconclusive():
    summary = summarize_rows(
        [
            {
                "split": "checkpoint_validation",
                "transition_rmse_1d": math.nan,
                "rolling_rmse_1d": 1.0,
                "transition_count_1d": 0,
                "rolling_count_1d": 1,
            }
        ],
        horizons=(1,),
    )
    assert summary[0]["status_1d"] == "inconclusive"
