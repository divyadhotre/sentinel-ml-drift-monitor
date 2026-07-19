"""
test_performance.py

Tests for label-free performance estimation. Verifies:
    - The loss estimator can be trained and produces per-row estimates
    - validate_estimator's estimated MAE is reasonably close to the real
      MAE when ground truth happens to be available (the validation case)
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from sentinel.performance import train_loss_estimator, estimate_performance, validate_estimator


@pytest.fixture
def trained_model_and_data():
    rng = np.random.default_rng(3)
    n = 3000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise_scale = 1 + np.abs(x1)  # error grows with |x1| -- a learnable pattern
    y = 2 * x1 + 3 * x2 + rng.normal(0, noise_scale)
    ref_df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
    model.fit(ref_df[["x1", "x2"]], ref_df["y"])
    return model, ref_df


class TestLossEstimator:
    def test_train_loss_estimator_returns_fitted_model(self, trained_model_and_data):
        model, ref_df = trained_model_and_data
        loss_estimator = train_loss_estimator(ref_df, ["x1", "x2"], "y", model)
        assert hasattr(loss_estimator, "predict")

    def test_estimate_performance_returns_expected_keys(self, trained_model_and_data):
        model, ref_df = trained_model_and_data
        loss_estimator = train_loss_estimator(ref_df, ["x1", "x2"], "y", model)
        result = estimate_performance(ref_df, ["x1", "x2"], model, loss_estimator)
        assert "estimated_mae" in result
        assert result["estimated_mae"] >= 0
        assert len(result["per_row_estimated_error"]) == len(ref_df)


class TestValidateEstimator:
    def test_estimate_is_reasonably_close_to_real_mae_on_similar_data(self, trained_model_and_data):
        model, ref_df = trained_model_and_data
        loss_estimator = train_loss_estimator(ref_df, ["x1", "x2"], "y", model)

        # "current" data drawn from the SAME distribution -- estimate should track reality closely
        rng = np.random.default_rng(4)
        n = 1000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        noise_scale = 1 + np.abs(x1)
        y = 2 * x1 + 3 * x2 + rng.normal(0, noise_scale)
        cur_df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        result = validate_estimator(cur_df, ["x1", "x2"], "y", model, loss_estimator)
        assert result["estimate_error_pct"] < 50  # loose bound -- proves it's in the right ballpark, not exact
