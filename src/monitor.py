"""
monitor.py

The "watchdog" -- ties together drift_metrics.py and model.py into a single
function that answers the practical question a real ML team cares about:

    "Should we be worried about our production model right now, and why?"

Given a reference (training) dataset, a current dataset, and the trained
model, this produces:
  - a per-feature drift report (from drift_metrics.py)
  - an overall drift health status (OK / WATCH / ALERT)
  - the model's real performance on current data (MAE, R^2)
  - a plain-English summary naming the top drifted feature(s)

This is the function the Streamlit dashboard calls directly.
"""

import joblib
import pandas as pd

from drift_metrics import generate_drift_report
from model import FEATURE_COLUMNS, TARGET_COLUMN, evaluate_on_new_data

# Thresholds for overall system status, based on the WORST single feature's PSI.
PSI_WATCH_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25


def overall_status(drift_report):
    """
    Looks at the highest PSI value across all features and maps it to an
    overall system health status.
    """
    max_psi = drift_report["psi"].max()

    if max_psi >= PSI_ALERT_THRESHOLD:
        status = "ALERT"
    elif max_psi >= PSI_WATCH_THRESHOLD:
        status = "WATCH"
    else:
        status = "OK"

    return status, float(max_psi)


def build_summary(drift_report, status, max_psi, performance_metrics=None):
    """
    Builds a short, plain-English message -- the kind of line you'd want
    to see in a Slack alert, not a raw table of numbers.
    """
    top_drifted = drift_report[drift_report["psi"] >= PSI_WATCH_THRESHOLD]

    if status == "OK":
        msg = "No meaningful drift detected. Model inputs still resemble training data."
    else:
        feature_names = ", ".join(top_drifted["feature"].tolist())
        msg = (
            f"{status}: Significant drift detected in [{feature_names}] "
            f"(highest PSI = {max_psi:.2f}). "
        )
        if status == "ALERT":
            msg += "Retraining is recommended before this model is trusted further."
        else:
            msg += "Monitor closely; consider scheduling a retrain soon."

    if performance_metrics is not None:
        msg += (
            f" Current-data performance: MAE = {performance_metrics['mae']:.2f} min, "
            f"R^2 = {performance_metrics['r2']:.3f}."
        )

    return msg


def run_monitor(reference_df, current_df, model=None, feature_columns=None):
    """
    Main entry point. Returns a dict with everything the dashboard needs:
      - drift_report (DataFrame, one row per feature)
      - status ("OK" / "WATCH" / "ALERT")
      - max_psi
      - performance (dict with mae/r2, or None if model not supplied)
      - summary (plain-English string)
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    drift_report = generate_drift_report(reference_df, current_df, feature_columns)
    status, max_psi = overall_status(drift_report)

    performance = None
    if model is not None and TARGET_COLUMN in current_df.columns:
        performance = evaluate_on_new_data(model, current_df, label="current data")

    summary = build_summary(drift_report, status, max_psi, performance)

    return {
        "drift_report": drift_report,
        "status": status,
        "max_psi": max_psi,
        "performance": performance,
        "summary": summary,
    }


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/taxi_training_era.csv")
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")
    model = joblib.load("data/processed/baseline_model.pkl")

    result = run_monitor(train_df, current_df, model=model)

    print("=" * 70)
    print(f"OVERALL STATUS: {result['status']}  (max PSI = {result['max_psi']:.4f})")
    print("=" * 70)
    print(result["drift_report"].to_string(index=False))
    print("-" * 70)
    print(result["summary"])