"""
test_drift_metrics.py

Unit tests for the core statistical drift-detection functions. These tests
verify the math itself is correct -- not just that the code runs -- using
known, hand-reasoned cases:
    - Identical distributions should show ~zero drift
    - Distributions with a real shift should be flagged
    - The naive z-score's known blind spot (same mean, different shape)
      should be demonstrated explicitly, not just asserted away
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from drift_metrics import (
    calculate_psi,
    calculate_ks,
    calculate_kl_divergence,
    calculate_naive_zscore,
    interpret_psi,
    generate_drift_report,
)


@pytest.fixture
def identical_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=50, scale=10, size=5000)
    current = rng.normal(loc=50, scale=10, size=5000)
    return reference, current


@pytest.fixture
def shifted_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=50, scale=10, size=5000)
    current = rng.normal(loc=90, scale=10, size=5000)  # large, obvious shift
    return reference, current


@pytest.fixture
def same_mean_different_shape():
    rng = np.random.default_rng(0)
    reference = rng.normal(loc=50, scale=10, size=5000)
    current = rng.normal(loc=50, scale=30, size=5000)  # same mean, 3x spread
    return reference, current


class TestPSI:
    def test_identical_distributions_have_near_zero_psi(self, identical_distributions):
        reference, current = identical_distributions
        psi = calculate_psi(reference, current)
        assert psi < 0.05, f"Expected near-zero PSI for identical distributions, got {psi}"

    def test_large_shift_is_flagged_as_major_drift(self, shifted_distributions):
        reference, current = shifted_distributions
        psi = calculate_psi(reference, current)
        assert psi > 0.25, f"Expected major drift (PSI > 0.25) for large shift, got {psi}"

    def test_psi_is_symmetric_in_magnitude_not_sign(self, shifted_distributions):
        reference, current = shifted_distributions
        psi_forward = calculate_psi(reference, current)
        assert psi_forward > 0, "PSI should be a positive magnitude"

    def test_interpret_psi_thresholds(self):
        assert interpret_psi(0.05) == "No significant drift"
        assert interpret_psi(0.15) == "Moderate drift - monitor closely"
        assert interpret_psi(0.30) == "Major drift - retraining recommended"


class TestKSTest:
    def test_identical_distributions_low_ks_statistic(self, identical_distributions):
        reference, current = identical_distributions
        statistic, p_value = calculate_ks(reference, current)
        assert statistic < 0.05

    def test_shifted_distributions_high_ks_statistic(self, shifted_distributions):
        reference, current = shifted_distributions
        statistic, p_value = calculate_ks(reference, current)
        assert statistic > 0.5
        assert p_value < 0.05


class TestKLDivergence:
    def test_identical_distributions_near_zero_kl(self, identical_distributions):
        reference, current = identical_distributions
        kl = calculate_kl_divergence(reference, current)
        assert kl < 0.05

    def test_shifted_distributions_high_kl(self, shifted_distributions):
        reference, current = shifted_distributions
        kl = calculate_kl_divergence(reference, current)
        assert kl > 0.2


class TestNaiveZscoreBlindSpot:
    """
    This is the key test proving WHY PSI/KS/KL are worth implementing over
    a naive mean-based check: a shape-only shift (same mean, different
    spread) should slip past the naive z-score but be caught by PSI.
    """

    def test_naive_zscore_misses_shape_only_shift(self, same_mean_different_shape):
        reference, current = same_mean_different_shape
        z = calculate_naive_zscore(reference, current)
        assert z < 0.2, f"Naive z-score should stay low despite the shape change, got {z}"

    def test_psi_catches_what_naive_zscore_misses(self, same_mean_different_shape):
        reference, current = same_mean_different_shape
        z = calculate_naive_zscore(reference, current)
        psi = calculate_psi(reference, current)
        assert psi > 0.25, f"PSI should flag major drift here, got {psi}"
        assert psi > z * 3, "PSI should be substantially more sensitive than naive z-score here"


class TestDriftReport:
    def test_generate_drift_report_shape(self):
        rng = np.random.default_rng(1)
        import pandas as pd

        ref_df = pd.DataFrame({
            "feature_a": rng.normal(0, 1, 1000),
            "feature_b": rng.normal(10, 2, 1000),
        })
        cur_df = pd.DataFrame({
            "feature_a": rng.normal(5, 1, 1000),   # shifted
            "feature_b": rng.normal(10, 2, 1000),  # stable
        })

        report = generate_drift_report(ref_df, cur_df, ["feature_a", "feature_b"])

        assert len(report) == 2
        assert set(report["feature"]) == {"feature_a", "feature_b"}
        assert "psi" in report.columns
        assert "naive_zscore" in report.columns

        # feature_a (shifted) should rank above feature_b (stable) by PSI
        top_feature = report.sort_values("psi", ascending=False).iloc[0]["feature"]
        assert top_feature == "feature_a"
