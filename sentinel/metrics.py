"""
sentinel/metrics.py

Statistical drift-detection functions: PSI, KS-test, KL-divergence, and a
naive z-score baseline for comparison. Works on any numeric pandas Series
or numpy array -- no assumptions about column names or domain.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _get_bin_edges(reference, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max(), n_bins + 1)
    return edges


def calculate_psi(reference, current, n_bins=10, epsilon=1e-4):
    """Population Stability Index between two numeric arrays."""
    edges = _get_bin_edges(np.asarray(reference), n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_ks(reference, current):
    """Kolmogorov-Smirnov two-sample test. Returns (statistic, p_value)."""
    statistic, p_value = ks_2samp(reference, current)
    return float(statistic), float(p_value)


def calculate_kl_divergence(reference, current, n_bins=10, epsilon=1e-4):
    """KL-divergence D(current || reference) over histogram bins."""
    edges = _get_bin_edges(np.asarray(reference), n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon
    return float(np.sum(cur_pct * np.log(cur_pct / ref_pct)))


def calculate_naive_zscore(reference, current):
    """The 'obvious' naive baseline: how many reference std-devs away is the current mean?"""
    ref_std = np.std(reference)
    if ref_std == 0:
        return 0.0
    return float(abs(np.mean(current) - np.mean(reference)) / ref_std)


def interpret_psi(psi_value):
    if psi_value < 0.1:
        return "No significant drift"
    elif psi_value < 0.25:
        return "Moderate drift - monitor closely"
    else:
        return "Major drift - retraining recommended"


def generate_drift_report(reference_df, current_df, feature_columns, n_bins=10):
    """
    Runs PSI, KS-test, KL-divergence, and naive z-score for every feature,
    comparing reference_df vs current_df. Returns a tidy DataFrame, one row
    per feature, sorted by PSI descending.
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
            "ref_mean": round(float(np.mean(ref_vals)), 3),
            "cur_mean": round(float(np.mean(cur_vals)), 3),
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
