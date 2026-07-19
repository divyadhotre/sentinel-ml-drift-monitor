"""
test_rolling.py

Tests for rolling-window drift monitoring. Verifies:
    - Partial/truncated boundary periods are correctly excluded
    - A gradually drifting series shows increasing PSI over time
    - overall_drift_timeline correctly collapses to one row per period
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from sentinel.rolling import split_into_periods, rolling_drift_report, overall_drift_timeline


@pytest.fixture
def gradually_drifting_data():
    rng = np.random.default_rng(6)
    dates = pd.date_range("2021-01-01", "2021-03-31", freq="D")
    rows = []
    for d in dates:
        days_elapsed = (d - dates[0]).days
        drift_amount = days_elapsed / len(dates) * 4.0
        for v in rng.normal(loc=10 + drift_amount, scale=1.5, size=40):
            rows.append({"date": d, "feature_x": v})
    return pd.DataFrame(rows)


class TestSplitIntoPeriods:
    def test_excludes_truncated_boundary_periods(self, gradually_drifting_data):
        periods = split_into_periods(gradually_drifting_data, "date", freq="W", require_full_period=True)
        # every remaining period should span (close to) a full 7-day week
        for period_str, df in periods.items():
            days_present = pd.to_datetime(df["date"]).dt.date.nunique()
            assert days_present >= 5  # allows a little slack, rejects clearly-truncated weeks

    def test_includes_all_periods_when_not_requiring_full(self, gradually_drifting_data):
        periods_strict = split_into_periods(gradually_drifting_data, "date", freq="W", require_full_period=True)
        periods_loose = split_into_periods(gradually_drifting_data, "date", freq="W", require_full_period=False)
        assert len(periods_loose) >= len(periods_strict)


class TestRollingDriftReport:
    def test_psi_increases_over_time_for_gradual_drift(self, gradually_drifting_data):
        report = rolling_drift_report(gradually_drifting_data, "date", ["feature_x"], freq="W")
        timeline = overall_drift_timeline(report)
        psi_values = timeline["max_psi"].tolist()
        # later periods should generally show more drift than earlier ones
        assert psi_values[-1] > psi_values[0]

    def test_overall_timeline_one_row_per_period(self, gradually_drifting_data):
        report = rolling_drift_report(gradually_drifting_data, "date", ["feature_x"], freq="W")
        timeline = overall_drift_timeline(report)
        assert len(timeline) == report["period"].nunique()
