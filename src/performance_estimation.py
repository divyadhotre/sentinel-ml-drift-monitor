"""
performance_estimation.py

LABEL-FREE PERFORMANCE ESTIMATION -- inspired by NannyML's Direct Loss
Estimation (DLE) approach, one of the hardest and most valuable problems
in real MLOps.

THE PROBLEM: in production, you usually don't have the true target value
right away. A delivery-time model's "true" answer only exists once the
delivery is actually complete; a loan-default model's "true" answer only
exists months later. So monitor.py's current approach (comparing predictions
to REAL labels) only works retrospectively -- it can't tell you TODAY
whether TODAY's predictions are trustworthy.

THE SOLUTION: train a second model -- a "loss estimator" -- whose job is
to predict how wrong the primary model's prediction is likely to be, using
only the input features and the primary model's own prediction (never the
true label). Once trained on a reference period where labels ARE available,
this loss estimator can be applied to brand-new data where labels are NOT
available yet, giving an estimated error/performance metric in real time.

This module also includes a validation step: on a period where we DO have
true labels (like our current_era data), we compare the ESTIMATED
performance against the ACTUAL performance, to prove the estimator itself
is trustworthy before relying on it in a truly label-free setting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_loss_estimator(reference_df, feature_columns, target_column, primary_model, random_state=42):
    """
    Trains the secondary "loss estimator" model on the reference era, where
    true labels are available.

    Steps:
      1. Get the primary model's predictions on the reference data.
      2. Compute the actual absolute error per row (this requires true labels
         -- which is why this training step can ONLY happen on a labeled
         reference period, not on live, label-free data).
      3. Train a Random Forest to predict that absolute error from the
         input features PLUS the primary model's own prediction (the
         prediction itself is informative -- e.g. very long predicted
         trips tend to have larger absolute errors).
    """
    X = reference_df[feature_columns].copy()
    y_true = reference_df[target_column]

    y_pred = primary_model.predict(X)
    absolute_error = np.abs(y_true - y_pred)

    # The loss estimator's own input space: original features + the primary
    # model's prediction as an extra signal.
    X_loss = X.copy()
    X_loss["primary_prediction"] = y_pred

    loss_estimator = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state)
    loss_estimator.fit(X_loss, absolute_error)

    return loss_estimator


def estimate_performance(current_df, feature_columns, primary_model, loss_estimator):
    """
    Applies the loss estimator to NEW data to estimate expected absolute
    error -- WITHOUT using any true target values from current_df, even if
    they happen to be present in the dataframe. This simulates the real
    production scenario where labels are not yet available.

    Returns the estimated MAE and per-row estimated errors.
    """
    X = current_df[feature_columns].copy()
    primary_predictions = primary_model.predict(X)

    X_loss = X.copy()
    X_loss["primary_prediction"] = primary_predictions

    estimated_errors = loss_estimator.predict(X_loss)
    estimated_mae = float(np.mean(estimated_errors))

    return {
        "estimated_mae": estimated_mae,
        "per_row_estimated_error": estimated_errors,
        "primary_predictions": primary_predictions,
    }


def validate_estimator(current_df, feature_columns, target_column, primary_model, loss_estimator):
    """
    VALIDATION ONLY: for a period where true labels ARE available (like our
    current_era data), compares the estimated MAE against the real MAE.
    This is how you'd build trust in the estimator before deploying it on
    truly label-free live data -- exactly what a real MLOps team would do
    before trusting DLE-style monitoring in production.
    """
    estimate_result = estimate_performance(current_df, feature_columns, primary_model, loss_estimator)
    estimated_mae = estimate_result["estimated_mae"]

    true_y = current_df[target_column]
    real_mae = mean_absolute_error(true_y, estimate_result["primary_predictions"])

    error_of_the_estimate = abs(estimated_mae - real_mae)
    pct_error = 100 * error_of_the_estimate / real_mae if real_mae else float("nan")

    return {
        "estimated_mae": round(estimated_mae, 3),
        "real_mae": round(real_mae, 3),
        "estimate_error": round(error_of_the_estimate, 3),
        "estimate_error_pct": round(pct_error, 1),
    }


if __name__ == "__main__":
    import joblib
    from model import FEATURE_COLUMNS, TARGET_COLUMN

    train_df = pd.read_csv("data/processed/taxi_training_era.csv")
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")
    primary_model = joblib.load("data/processed/baseline_model.pkl")

    print("Training loss estimator on reference (Jan 2020) data...")
    loss_estimator = train_loss_estimator(train_df, FEATURE_COLUMNS, TARGET_COLUMN, primary_model)

    print("\nValidating: does the label-free estimate match reality on Apr 2020 data?")
    print("(In real production, current_df would have NO target column at all --")
    print(" we only use it here to prove the estimator works before trusting it live.)\n")

    result = validate_estimator(current_df, FEATURE_COLUMNS, TARGET_COLUMN, primary_model, loss_estimator)

    print(f"  Estimated MAE (label-free): {result['estimated_mae']} min")
    print(f"  Real MAE (using true labels): {result['real_mae']} min")
    print(f"  Estimate error: {result['estimate_error']} min ({result['estimate_error_pct']}% off)")
