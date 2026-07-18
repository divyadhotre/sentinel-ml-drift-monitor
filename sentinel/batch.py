"""
sentinel/batch.py

Runs SentinelMonitor across multiple jobs (each a separate model/dataset
pair) defined in a config file, and produces one consolidated summary --
a lightweight step toward monitoring many models at once instead of only
one at a time.
"""

import pandas as pd

from .core import SentinelMonitor
from .config import load_config


def run_batch(config_path):
    """
    Loads a config file and runs SentinelMonitor for every job in it.
    Returns (summary_df, results_dict) where summary_df has one row per
    job (name, status, max_psi, n_drifted_features) and results_dict maps
    job name -> full SentinelResult for anyone who wants the details.
    """
    jobs = load_config(config_path)

    summary_rows = []
    results = {}

    for job in jobs:
        try:
            reference_df = pd.read_csv(job["reference"])
            current_df = pd.read_csv(job["current"])

            monitor = SentinelMonitor(
                reference_df=reference_df,
                current_df=current_df,
                feature_columns=job["features"],
                target_column=job["target"],
            )
            result = monitor.run()
            results[job["name"]] = result

            n_drifted = int((result.drift_report["psi"] >= 0.10).sum())
            summary_rows.append({
                "job": job["name"],
                "status": result.status,
                "max_psi": round(result.max_psi, 4),
                "n_features_drifted": n_drifted,
                "n_features_total": len(job["features"]),
                "error": None,
            })

        except Exception as e:
            summary_rows.append({
                "job": job["name"],
                "status": "ERROR",
                "max_psi": None,
                "n_features_drifted": None,
                "n_features_total": len(job["features"]),
                "error": str(e),
            })
            results[job["name"]] = None

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, results
