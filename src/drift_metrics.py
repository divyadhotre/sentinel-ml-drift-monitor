"""
drift_metrics.py

Implements three industry-standard statistical drift-detection metrics,
built from their actual formulas (not just calling a black-box library),
so we can measure -- per feature -- how much the current-era data has
shifted away from the training-era data.

1. PSI  (Population Stability Index) -- the metric literally used by banks
   and credit-risk teams to monitor model input drift. Rule of thumb:
     PSI < 0.1  -> no significant change
     0.1 - 0.25 -> moderate shift, watch closely
     > 0.25     -> major shift, retrain likely needed

2. KS-test (Kolmogorov-Smirnov) -- a classic statistical test that measures
   the maximum distance between two cumulative distributions. Returns a
   statistic (0 = identical, 1 = completely different) and a p-value.

3. KL-divergence (Kullback-Leibler) -- measures how much "extra information"
   is needed to describe the current distribution using the training
   distribution as a reference. 0 = identical, higher = more different.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _get_bin_edges(reference, n_bins=10):
    """
    Creates shared bin edges from the REFERENCE (training) data only.
    Both distributions are later binned using these same edges so the
    comparison is apples-to-apples.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # fall back to equal-width bins if the feature has very few unique values
        edges = np.linspace(reference.min(), reference.max(), n_bins + 1)
    return edges


def calculate_psi(reference, current, n_bins=10, epsilon=1e-4):
    """
    Population Stability Index between reference (training) and current data
    for a single numeric feature.

    PSI = sum( (current_pct - ref_pct) * ln(current_pct / ref_pct) )
    over each bin.
    """
    edges = _get_bin_edges(reference, n_bins)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # convert to percentages, add epsilon to avoid divide-by-zero / log(0)
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_ks(reference, current):
    """
    Kolmogorov-Smirnov two-sample test.
    Returns (statistic, p_value).
    Small p-value (<0.05) => distributions are statistically significantly different.
    """
    statistic, p_value = ks_2samp(reference, current)
    return float(statistic), float(p_value)


def calculate_kl_divergence(reference, current, n_bins=10, epsilon=1e-4):
    """
    KL-divergence D(current || reference), computed over histogram bins
    built from the reference distribution.
    """
    edges = _get_bin_edges(reference, n_bins)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    kl = np.sum(cur_pct * np.log(cur_pct / ref_pct))
    return float(kl)


def calculate_naive_zscore(reference, current):
    """
    The "obvious" naive drift check most teams reach for first: how many
    reference-era standard deviations away is the current mean?

        z = |mean(current) - mean(reference)| / std(reference)

    This is fast and intuitive, but it only looks at the MEAN. It is blind
    to shape changes (variance, skew, multi-modality) that leave the mean
    roughly unchanged -- exactly the kind of shift PSI/KS-test/KL-divergence
    are designed to catch. See notebooks/05_robustness_analysis.ipynb for a
    concrete case where this method fails and PSI succeeds.
    """
    ref_std = np.std(reference)
    if ref_std == 0:
        return 0.0
    z = abs(np.mean(current) - np.mean(reference)) / ref_std
    return float(z)


def interpret_psi(psi_value):
    if psi_value < 0.1:
        return "No significant drift"
    elif psi_value < 0.25:
        return "Moderate drift - monitor closely"
    else:
        return "Major drift - retraining recommended"


def generate_drift_report(reference_df, current_df, feature_columns, n_bins=10):
    """
    Runs PSI, KS-test, and KL-divergence for every feature in feature_columns,
    comparing reference_df (training era) vs current_df (current era).

    Returns a tidy DataFrame -- one row per feature -- ready to display in
    the Streamlit dashboard or save as a report.
    """
    rows = []
    for col in feature_columns:
        ref_vals = reference_df[col].values
        cur_vals = current_df[col].values

        psi = calculate_psi(ref_vals, cur_vals, n_bins=n_bins)
        ks_stat, ks_p = calculate_ks(ref_vals, cur_vals)
        kl = calculate_kl_divergence(ref_vals, cur_vals, n_bins=n_bins)
        naive_z = calculate_naive_zscore(ref_vals, cur_vals)

        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "psi_verdict": interpret_psi(psi),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 6),
            "kl_divergence": round(kl, 4),
            "naive_zscore": round(naive_z, 4),
            "ref_mean": round(np.mean(ref_vals), 3),
            "cur_mean": round(np.mean(cur_vals), 3),
        })

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/taxi_training_era.csv")
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")

    feature_columns = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "hour_of_day",
        "is_weekend",
    ]

    report = generate_drift_report(train_df, current_df, feature_columns)
    pd.set_option("display.width", 120)
    print(report.to_string(index=False))

    report.to_csv("reports/drift_report.csv", index=False)
    print("\nSaved drift report to reports/drift_report.csv")
