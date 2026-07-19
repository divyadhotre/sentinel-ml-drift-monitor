"""
test_concept_drift.py

Tests for relationship-level (concept) drift detection. Verifies:
    - A stable relationship (same data-generating process) is NOT flagged
    - A genuinely changed relationship (target depends on different
      features between eras) IS flagged, via both the correlation-shift
      and feature-importance-shift checks
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from sentinel.concept_drift import (
    compare_correlation_shift,
    compare_feature_importance,
    train_model_on_era,
    generate_concept_drift_report,
)


@pytest.fixture
def stable_relationship_data():
    """Same relationship in both eras: target = 3*x + noise, in both."""
    rng = np.random.default_rng(1)
    x_ref = rng.normal(0, 1, 2000)
    y_ref = 3 * x_ref + rng.normal(0, 0.5, 2000)
    ref_df = pd.DataFrame({"x": x_ref, "other": rng.normal(0, 1, 2000), "y": y_ref})

    x_cur = rng.normal(0, 1, 2000)
    y_cur = 3 * x_cur + rng.normal(0, 0.5, 2000)
    cur_df = pd.DataFrame({"x": x_cur, "other": rng.normal(0, 1, 2000), "y": y_cur})
    return ref_df, cur_df


@pytest.fixture
def changed_relationship_data():
    """Relationship flips: target depends on 'x' in reference, on 'other' in current."""
    rng = np.random.default_rng(2)
    x_ref = rng.normal(0, 1, 2000)
    other_ref = rng.normal(0, 1, 2000)
    y_ref = 5 * x_ref + rng.normal(0, 0.3, 2000)
    ref_df = pd.DataFrame({"x": x_ref, "other": other_ref, "y": y_ref})

    x_cur = rng.normal(0, 1, 2000)
    other_cur = rng.normal(0, 1, 2000)
    y_cur = 5 * other_cur + rng.normal(0, 0.3, 2000)  # relationship moved to 'other'
    cur_df = pd.DataFrame({"x": x_cur, "other": other_cur, "y": y_cur})
    return ref_df, cur_df


class TestCorrelationShift:
    def test_stable_relationship_shows_low_correlation_shift(self, stable_relationship_data):
        ref_df, cur_df = stable_relationship_data
        report = compare_correlation_shift(ref_df, cur_df, ["x", "other"], "y")
        assert not report["concept_drift_flag"].any()

    def test_changed_relationship_flags_correlation_shift(self, changed_relationship_data):
        ref_df, cur_df = changed_relationship_data
        report = compare_correlation_shift(ref_df, cur_df, ["x", "other"], "y")
        assert report["concept_drift_flag"].any()
        flagged = set(report.loc[report["concept_drift_flag"], "feature"])
        assert "x" in flagged or "other" in flagged


class TestFeatureImportanceShift:
    def test_changed_relationship_flags_importance_shift(self, changed_relationship_data):
        ref_df, cur_df = changed_relationship_data
        model_ref = train_model_on_era(ref_df, ["x", "other"], "y")
        model_cur = train_model_on_era(cur_df, ["x", "other"], "y")
        report = compare_feature_importance(model_ref, model_cur, ["x", "other"])
        assert report["concept_drift_flag"].any()


class TestFullReport:
    def test_stable_data_gives_no_concept_drift_verdict(self, stable_relationship_data):
        ref_df, cur_df = stable_relationship_data
        result = generate_concept_drift_report(ref_df, cur_df, ["x", "other"], "y")
        assert result["verdict"] == "NO CONCEPT DRIFT"

    def test_changed_data_gives_concept_drift_detected_verdict(self, changed_relationship_data):
        ref_df, cur_df = changed_relationship_data
        result = generate_concept_drift_report(ref_df, cur_df, ["x", "other"], "y")
        assert result["verdict"] == "CONCEPT DRIFT DETECTED"
        assert len(result["flagged_features"]) > 0
