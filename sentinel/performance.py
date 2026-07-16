"""
sentinel/performance.py

Label-free performance estimation: trains a secondary "loss estimator"
model on a labeled reference period, then applies it to new data to
estimate expected error WITHOUT needing true labels on the new data.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_loss_estimator(reference_df, feature_columns, target_column, primary_model, random_state=42):
    X = reference_df[feature_columns].copy()
    y_true = reference_df[target_column]
    y_pred = primary_model.predict(X)
    absolute_error = np.abs(y_true - y_pred)

    X_loss = X.copy()
    X_loss["primary_prediction"] = y_pred

    loss_estimator = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state)
    loss_estimator.fit(X_loss, absolute_error)
    return loss_estimator


def estimate_performance(current_df, feature_columns, primary_model, loss_estimator):
    X = current_df[feature_columns].copy()
    primary_predictions = primary_model.predict(X)

    X_loss = X.copy()
    X_loss["primary_prediction"] = primary_predictions

    estimated_errors = loss_estimator.predict(X_loss)
    return {
        "estimated_mae": float(np.mean(estimated_errors)),
        "per_row_estimated_error": estimated_errors,
        "primary_predictions": primary_predictions,
    }


def validate_estimator(current_df, feature_columns, target_column, primary_model, loss_estimator):
    """Validation-only helper: compares the estimate against real labels, when available."""
    result = estimate_performance(current_df, feature_columns, primary_model, loss_estimator)
    real_mae = mean_absolute_error(current_df[target_column], result["primary_predictions"])
    error = abs(result["estimated_mae"] - real_mae)
    pct = 100 * error / real_mae if real_mae else float("nan")
    return {
        "estimated_mae": round(result["estimated_mae"], 3),
        "real_mae": round(real_mae, 3),
        "estimate_error": round(error, 3),
        "estimate_error_pct": round(pct, 1),
    }
