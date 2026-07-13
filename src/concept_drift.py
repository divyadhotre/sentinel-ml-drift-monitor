"""
concept_drift.py

DATA drift (drift_metrics.py) asks: "did the INPUTS change?"
CONCEPT drift asks a different, harder question: "did the RELATIONSHIP between
inputs and the target change?" A model can see totally normal-looking inputs
and still be wrong, if the real-world relationship it learned no longer holds
(e.g. a fraud model where fraudsters changed tactics, not just transaction
volume increasing).

This module detects concept drift two ways:

1. FEATURE IMPORTANCE SHIFT
   Train two separate models -- one on the reference era, one on the current
   era -- and compare which features each model leans on. If a feature that
   used to matter a lot suddenly matters much less (or vice versa), the
   underlying relationship has changed.

2. CORRELATION SHIFT
   For each feature, compare its raw correlation with the target in the
   reference era vs the current era. A feature whose correlation with the
   target flips sign or changes substantially is a strong, simple signal
   that the relationship itself has drifted -- independent of whether the
   feature's own distribution (PSI) moved at all.

Used together with drift_metrics.py, this lets us distinguish:
  - Data drift only         -> inputs shifted, relationship intact
  - Concept drift only      -> relationship changed, inputs look normal
  - Both                    -> the hardest, most dangerous case
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CORR_ALERT_THRESHOLD = 0.15   # absolute change in correlation considered "concept drift"
IMPORTANCE_ALERT_THRESHOLD = 0.08  # absolute change in feature importance considered notable


def train_model_on_era(df, feature_columns, target_column, random_state=42):
    """Trains a fresh Random Forest on a given era's data (no train/test split --
    we want the model that best represents THAT era's relationship for comparison)."""
    X = df[feature_columns]
    y = df[target_column]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state)
    model.fit(X, y)
    return model


def compare_feature_importance(model_ref, model_cur, feature_columns):
    """
    Compares which features each era's model relies on.
    A large positive/negative diff means the model's "reasoning" changed
    between eras -- a sign of concept drift.
    """
    imp_ref = model_ref.feature_importances_
    imp_cur = model_cur.feature_importances_

    df = pd.DataFrame({
        "feature": feature_columns,
        "importance_ref": imp_ref,
        "importance_cur": imp_cur,
    })
    df["abs_diff"] = (df["importance_cur"] - df["importance_ref"]).abs()
    df["concept_drift_flag"] = df["abs_diff"] >= IMPORTANCE_ALERT_THRESHOLD
    return df.sort_values("abs_diff", ascending=False).reset_index(drop=True)


def compare_correlation_shift(reference_df, current_df, feature_columns, target_column):
    """
    Compares each feature's raw correlation with the target across eras.
    This is a simpler, model-free signal of concept drift: if trip_distance
    used to correlate strongly with trip_duration but now barely does,
    something about the underlying relationship changed.
    """
    rows = []
    for col in feature_columns:
        corr_ref = reference_df[col].corr(reference_df[target_column])
        corr_cur = current_df[col].corr(current_df[target_column])
        abs_diff = abs(corr_cur - corr_ref)
        rows.append({
            "feature": col,
            "corr_ref": round(corr_ref, 4),
            "corr_cur": round(corr_cur, 4),
            "abs_diff": round(abs_diff, 4),
            "concept_drift_flag": abs_diff >= CORR_ALERT_THRESHOLD,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("abs_diff", ascending=False).reset_index(drop=True)


def generate_concept_drift_report(reference_df, current_df, feature_columns, target_column):
    """
    Main entry point. Trains a model per era, compares feature importances,
    and compares raw correlations. Returns both reports plus an overall
    concept-drift verdict.
    """
    model_ref = train_model_on_era(reference_df, feature_columns, target_column, random_state=42)
    model_cur = train_model_on_era(current_df, feature_columns, target_column, random_state=42)

    importance_report = compare_feature_importance(model_ref, model_cur, feature_columns)
    correlation_report = compare_correlation_shift(reference_df, current_df, feature_columns, target_column)

    any_importance_flag = importance_report["concept_drift_flag"].any()
    any_correlation_flag = correlation_report["concept_drift_flag"].any()

    if any_importance_flag or any_correlation_flag:
        verdict = "CONCEPT DRIFT DETECTED"
    else:
        verdict = "NO CONCEPT DRIFT"

    flagged_features = sorted(set(
        importance_report.loc[importance_report["concept_drift_flag"], "feature"].tolist()
        + correlation_report.loc[correlation_report["concept_drift_flag"], "feature"].tolist()
    ))

    return {
        "verdict": verdict,
        "flagged_features": flagged_features,
        "importance_report": importance_report,
        "correlation_report": correlation_report,
    }


if __name__ == "__main__":
    from model import FEATURE_COLUMNS, TARGET_COLUMN

    train_df = pd.read_csv("data/processed/taxi_training_era.csv")
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")

    result = generate_concept_drift_report(train_df, current_df, FEATURE_COLUMNS, TARGET_COLUMN)

    print("=" * 70)
    print(f"CONCEPT DRIFT VERDICT: {result['verdict']}")
    print(f"Flagged features: {result['flagged_features'] or 'none'}")
    print("=" * 70)
    print("\nFeature Importance Comparison (per-era trained models):")
    print(result["importance_report"].to_string(index=False))
    print("\nCorrelation Shift (feature vs target, per era):")
    print(result["correlation_report"].to_string(index=False))
