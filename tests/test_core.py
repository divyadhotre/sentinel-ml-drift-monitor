"""
test_core.py

Tests for SentinelMonitor, the main public API. Verifies:
    - Basic run() with no target/model returns correct status and drift report
    - Providing a target column enables the concept drift check
    - Providing a trained model enables the performance check
    - A mismatched feature column raises a clear error, rather than failing silently
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from sentinel.core import SentinelMonitor, SentinelResult


@pytest.fixture
def basic_dataframes():
    rng = np.random.default_rng(10)
    ref = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 1000),
        "feature_b": rng.normal(10, 2, 1000),
        "target": rng.normal(5, 1, 1000),
    })
    cur = pd.DataFrame({
        "feature_a": rng.normal(3, 1, 1000),  # shifted
        "feature_b": rng.normal(10, 2, 1000),  # stable
        "target": rng.normal(5, 1, 1000),
    })
    return ref, cur


class TestSentinelMonitorBasic:
    def test_run_returns_sentinel_result(self, basic_dataframes):
        ref, cur = basic_dataframes
        result = SentinelMonitor(ref, cur, feature_columns=["feature_a", "feature_b"]).run()
        assert isinstance(result, SentinelResult)
        assert result.status in ("OK", "WATCH", "ALERT")

    def test_shifted_feature_is_detected(self, basic_dataframes):
        ref, cur = basic_dataframes
        result = SentinelMonitor(ref, cur, feature_columns=["feature_a", "feature_b"]).run()
        top_feature = result.drift_report.sort_values("psi", ascending=False).iloc[0]["feature"]
        assert top_feature == "feature_a"

    def test_no_target_means_no_concept_drift_check(self, basic_dataframes):
        ref, cur = basic_dataframes
        result = SentinelMonitor(ref, cur, feature_columns=["feature_a", "feature_b"]).run()
        assert result.concept_drift is None
        assert result.performance is None


class TestSentinelMonitorWithTarget:
    def test_target_column_enables_concept_drift_check(self, basic_dataframes):
        ref, cur = basic_dataframes
        result = SentinelMonitor(
            ref, cur, feature_columns=["feature_a", "feature_b"], target_column="target"
        ).run()
        assert result.concept_drift is not None
        assert result.concept_drift["verdict"] in ("NO CONCEPT DRIFT", "CONCEPT DRIFT DETECTED")


class TestSentinelMonitorWithModel:
    def test_model_enables_performance_check(self, basic_dataframes):
        ref, cur = basic_dataframes
        model = RandomForestRegressor(n_estimators=20, random_state=0)
        model.fit(ref[["feature_a", "feature_b"]], ref["target"])

        result = SentinelMonitor(
            ref, cur, feature_columns=["feature_a", "feature_b"],
            target_column="target", model=model,
        ).run()
        assert result.performance is not None
        assert "mae" in result.performance
        assert "r2" in result.performance


class TestSentinelMonitorValidation:
    def test_missing_feature_column_raises_clear_error(self, basic_dataframes):
        ref, cur = basic_dataframes
        with pytest.raises(ValueError, match="not found"):
            SentinelMonitor(ref, cur, feature_columns=["feature_a", "does_not_exist"])
