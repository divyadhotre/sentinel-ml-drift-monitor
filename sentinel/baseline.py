"""
sentinel/baseline.py

QUICK BASELINE MODEL -- explicitly NOT "AutoML". A single Random Forest,
default settings, no hyperparameter tuning, no model comparison. Built to
give a fast sanity-check of drift's real performance impact on ANY
uploaded dataset, with safety features an honest implementation needs:

1. AUTOMATIC ID-COLUMN DETECTION
   Columns where almost every value is unique AND the name matches a
   common ID pattern (or is a high-cardinality integer column) are
   flagged and excluded by default -- see notebooks/02_citibike_validation.ipynb
   for the real example (bikeid) that motivated this.

2. A HOLDOUT SPLIT FOR AN HONEST BASELINE NUMBER
   Instead of training on all reference data and only reporting
   performance on current data, the reference data is split into a
   train/holdout portion first (stratified for classification targets).
   This gives an honest "how good is this model on same-era, unseen
   data" baseline BEFORE any drift is even considered -- so a drop in
   performance on current data can be compared against a real baseline,
   not just reported in isolation.

3. A DATA-QUALITY GATE
   Refuses to train on too little clean data (default: fewer than 200
   rows after dropping missing values).

4. AUTOMATIC SAMPLING FOR LARGE INPUTS
   Caps training data to a maximum row count (default 20,000).

Every result from this module should be displayed with a permanent,
visible "Quick baseline -- not a validated model" label -- a UI concern,
not enforced here, but callers must not omit it.
"""

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

ID_COLUMN_UNIQUENESS_THRESHOLD = 0.95
MIN_ROWS_TO_TRAIN = 200
MAX_ROWS_TO_TRAIN = 20_000
MAX_CLASSES_FOR_CLASSIFICATION = 20
HOLDOUT_FRACTION = 0.2

_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|id$|identifier|uuid|guid)", re.IGNORECASE)


def detect_id_like_columns(df, columns, threshold=ID_COLUMN_UNIQUENESS_THRESHOLD):
    """
    Flags columns likely to be identifiers, using two signals:
      1. The column NAME matches a common ID pattern (contains "id",
         "identifier", "uuid", "guid" -- e.g. "bikeid", "start station id").
         This alone is enough to flag a column, regardless of dtype.
      2. The column is INTEGER-typed AND has near-total uniqueness relative
         to row count (a classic auto-increment ID signature). Uniqueness
         alone is NOT sufficient -- continuous float measurements (like
         trip duration or GPS coordinates) are also nearly all-unique, but
         are real measurements, not identifiers, so float columns are
         never flagged by cardinality alone.
    """
    id_like = []
    n_rows = len(df)
    if n_rows == 0:
        return id_like

    for col in columns:
        if col not in df.columns:
            continue

        normalized_name = col.strip().lower().replace(" ", "_")
        if _ID_NAME_PATTERN.search(normalized_name):
            id_like.append(col)
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            n_unique = df[col].nunique(dropna=True)
            if (n_unique / n_rows) >= threshold:
                id_like.append(col)

    return id_like


def detect_problem_type(target_series, max_classes=MAX_CLASSES_FOR_CLASSIFICATION):
    """Guesses regression vs classification from the target column."""
    n_unique = target_series.nunique(dropna=True)
    if n_unique <= max_classes:
        return "classification"
    return "regression"


def _compute_metrics(problem_type, y_true, y_pred, n_rows):
    if problem_type == "classification":
        return {
            "problem_type": problem_type,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted")),
            "n_rows_evaluated": n_rows,
        }
    return {
        "problem_type": problem_type,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_rows_evaluated": n_rows,
    }


def train_quick_baseline(
    reference_df,
    feature_columns,
    target_column,
    min_rows=MIN_ROWS_TO_TRAIN,
    max_rows=MAX_ROWS_TO_TRAIN,
    id_threshold=ID_COLUMN_UNIQUENESS_THRESHOLD,
    holdout_fraction=HOLDOUT_FRACTION,
    random_state=42,
):
    """
    Trains a single Random Forest (classifier or regressor, auto-detected)
    on a TRAIN split of reference_df, holding out `holdout_fraction` of it
    (stratified, for classification targets) to report an honest baseline
    performance number -- before any drift is considered.

    Returns a dict with: model, problem_type, used_features,
    excluded_id_columns, n_rows_used, n_rows_dropped_missing, warnings,
    baseline_holdout_metrics -- or {"error": "..."} if training was refused.
    """
    warnings = []

    id_like = detect_id_like_columns(reference_df, feature_columns, threshold=id_threshold)
    used_features = [c for c in feature_columns if c not in id_like]
    if id_like:
        warnings.append(f"Excluded likely ID column(s) from training: {id_like}")

    if not used_features:
        return {"error": "No usable feature columns remain after excluding ID-like columns."}

    df = reference_df[used_features + [target_column]].copy()
    n_before = len(df)
    df = df.dropna()
    n_after_dropna = len(df)
    n_dropped_missing = n_before - n_after_dropna

    if n_dropped_missing > 0:
        pct = 100 * n_dropped_missing / n_before if n_before else 0
        warnings.append(f"Dropped {n_dropped_missing:,} rows ({pct:.1f}%) with missing values before training.")

    if len(df) < min_rows:
        return {"error": f"Not enough clean data for a reliable baseline (need {min_rows}+ rows, had {len(df)})."}

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
        warnings.append(f"Sampled down to {max_rows:,} rows for training speed.")

    problem_type = detect_problem_type(df[target_column])

    X = df[used_features]
    y = df[target_column]

    # Holdout split: train on one part, get an honest same-era baseline on the rest.
    stratify_col = y if problem_type == "classification" and y.nunique() > 1 else None
    try:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=holdout_fraction, random_state=random_state, stratify=stratify_col
        )
    except ValueError:
        # Stratification can fail if a class has too few members -- fall back to unstratified.
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=holdout_fraction, random_state=random_state
        )
        warnings.append("Stratified split not possible (a class had too few examples) -- used a plain random split instead.")

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)

    model.fit(X_train, y_train)

    holdout_preds = model.predict(X_holdout)
    baseline_holdout_metrics = _compute_metrics(problem_type, y_holdout, holdout_preds, len(X_holdout))

    if len(X_holdout) < 50:
        warnings.append(f"Small holdout set ({len(X_holdout)} rows) — baseline metric may be noisy.")

    return {
        "model": model,
        "problem_type": problem_type,
        "used_features": used_features,
        "excluded_id_columns": id_like,
        "n_rows_used": len(df),
        "n_rows_train": len(X_train),
        "n_rows_holdout": len(X_holdout),
        "n_rows_dropped_missing": n_dropped_missing,
        "warnings": warnings,
        "baseline_holdout_metrics": baseline_holdout_metrics,
    }


def evaluate_quick_baseline(baseline_result, current_df, target_column):
    """
    Evaluates a trained quick-baseline model (from train_quick_baseline)
    on current_df -- the "did performance change on genuinely new data"
    comparison point against baseline_holdout_metrics.
    """
    if "error" in baseline_result:
        return {"error": baseline_result["error"]}

    model = baseline_result["model"]
    used_features = baseline_result["used_features"]
    problem_type = baseline_result["problem_type"]

    df = current_df[used_features + [target_column]].copy().dropna()
    if len(df) < 10:
        return {"error": f"Not enough clean current-era data to evaluate (had {len(df)} usable rows)."}

    X = df[used_features]
    y_true = df[target_column]
    y_pred = model.predict(X)

    return _compute_metrics(problem_type, y_true, y_pred, len(df))
