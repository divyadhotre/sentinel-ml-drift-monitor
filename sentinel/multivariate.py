"""
sentinel/multivariate.py

MULTIVARIATE DRIFT DETECTION -- catches a real, documented blind spot in
every per-feature check (PSI, KS-test, KL-divergence): two features can
each look completely unchanged in isolation, while the RELATIONSHIP
between them shifts (e.g. two variables that used to move together now
move oppositely). Univariate checks are structurally incapable of seeing
this, because they only ever look at one column at a time.

THE TECHNIQUE: a "domain classifier" -- the same real approach used by
NannyML and Evidently AI for this exact problem.

  1. Label every reference-era row 0, every current-era row 1.
  2. Train a classifier to predict that label FROM THE FEATURES ONLY
     (never using the label as an input, obviously).
  3. Use cross-validation to get an honest, out-of-fold AUC score for
     how well the classifier can tell the two eras apart.
  4. AUC near 0.5 -> the classifier can't distinguish them; the joint
     distribution of features is genuinely similar.
     AUC well above 0.5 -> something about the overall feature space
     changed, even if no single column's own distribution moved much.

The classifier's feature importances double as an interpretability tool:
which features contributed most to distinguishing the two eras.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AUC_STABLE_THRESHOLD = 0.60
AUC_MAJOR_THRESHOLD = 0.75


def _interpret_auc(auc):
    if auc < AUC_STABLE_THRESHOLD:
        return "No significant multivariate drift"
    elif auc < AUC_MAJOR_THRESHOLD:
        return "Moderate multivariate drift - relationships between features may have shifted"
    else:
        return "Major multivariate drift - the joint feature distribution has changed substantially"


def detect_multivariate_drift(reference_df, current_df, feature_columns, n_splits=5, random_state=42):
    """
    Trains a domain classifier to distinguish reference_df rows from
    current_df rows, using only feature_columns. Returns a dict with:
      - domain_classifier_auc: cross-validated AUC (0.5 = indistinguishable, 1.0 = perfectly separable)
      - verdict: plain-language interpretation
      - feature_importance: which features drove the separation, sorted descending
      - n_reference, n_current: row counts used

    A minimum of ~30 rows per era and 2 feature columns is recommended for
    a stable result; very small datasets will produce a noisy AUC estimate.
    """
    ref = reference_df[feature_columns].copy()
    cur = current_df[feature_columns].copy()

    ref = ref.dropna()
    cur = cur.dropna()

    combined = pd.concat([ref, cur], ignore_index=True)
    labels = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])

    if len(ref) < 10 or len(cur) < 10:
        return {"error": f"Not enough clean rows to run a multivariate check (reference={len(ref)}, current={len(cur)})."}

    n_splits_used = min(n_splits, min(len(ref), len(cur)))
    if n_splits_used < 2:
        return {"error": "Not enough rows per era to perform cross-validation for the multivariate check."}

    classifier = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=random_state)

    cv = StratifiedKFold(n_splits=n_splits_used, shuffle=True, random_state=random_state)
    oof_probs = cross_val_predict(classifier, combined, labels, cv=cv, method="predict_proba")[:, 1]

    auc = float(roc_auc_score(labels, oof_probs))

    # Fit once on all data purely to extract feature importances for interpretability.
    classifier.fit(combined, labels)
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "domain_classifier_auc": round(auc, 4),
        "verdict": _interpret_auc(auc),
        "feature_importance": importance_df,
        "n_reference": len(ref),
        "n_current": len(cur),
    }
