"""
sentinel/concept_drift.py

Detects whether the RELATIONSHIP between features and target changed
between two eras, not just the inputs themselves. Works on any tabular
data with a numeric target.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CORR_ALERT_THRESHOLD = 0.15
IMPORTANCE_ALERT_THRESHOLD = 0.08


def train_model_on_era(df, feature_columns, target_column, random_state=42):
    X = df[feature_columns]
    y = df[target_column]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state)
    model.fit(X, y)
    return model


def compare_feature_importance(model_ref, model_cur, feature_columns):
    df = pd.DataFrame({
        "feature": feature_columns,
        "importance_ref": model_ref.feature_importances_,
        "importance_cur": model_cur.feature_importances_,
    })
    df["abs_diff"] = (df["importance_cur"] - df["importance_ref"]).abs()
    df["concept_drift_flag"] = df["abs_diff"] >= IMPORTANCE_ALERT_THRESHOLD
    return df.sort_values("abs_diff", ascending=False).reset_index(drop=True)


def compare_correlation_shift(reference_df, current_df, feature_columns, target_column):
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
    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)


def generate_concept_drift_report(reference_df, current_df, feature_columns, target_column):
    model_ref = train_model_on_era(reference_df, feature_columns, target_column, random_state=42)
    model_cur = train_model_on_era(current_df, feature_columns, target_column, random_state=42)

    importance_report = compare_feature_importance(model_ref, model_cur, feature_columns)
    correlation_report = compare_correlation_shift(reference_df, current_df, feature_columns, target_column)

    any_flag = importance_report["concept_drift_flag"].any() or correlation_report["concept_drift_flag"].any()
    verdict = "CONCEPT DRIFT DETECTED" if any_flag else "NO CONCEPT DRIFT"

    flagged = sorted(set(
        importance_report.loc[importance_report["concept_drift_flag"], "feature"].tolist()
        + correlation_report.loc[correlation_report["concept_drift_flag"], "feature"].tolist()
    ))

    return {
        "verdict": verdict,
        "flagged_features": flagged,
        "importance_report": importance_report,
        "correlation_report": correlation_report,
    }
