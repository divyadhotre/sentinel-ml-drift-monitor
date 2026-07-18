"""
sentinel/cli.py

Command-line interface for Sentinel. Lets anyone run drift detection on
their own data without writing any Python code.

Usage:
    sentinel monitor --reference old.csv --current new.csv --features distance,hour,price
    sentinel monitor --reference old.csv --current new.csv --features distance,hour --target duration
"""

import argparse
import sys
import pandas as pd

from .core import SentinelMonitor
from .batch import run_batch


def _print_report(result):
    print("=" * 70)
    print(f"SENTINEL STATUS: {result.status}   (max PSI = {result.max_psi:.4f})")
    print("=" * 70)
    print(result.drift_report.to_string(index=False))
    print("-" * 70)
    print(result.summary)

    if result.concept_drift is not None:
        print("\nConcept drift check:", result.concept_drift["verdict"])
        if result.concept_drift["flagged_features"]:
            print("Flagged features:", result.concept_drift["flagged_features"])


def monitor_command(args):
    reference_df = pd.read_csv(args.reference)
    current_df = pd.read_csv(args.current)
    feature_columns = [f.strip() for f in args.features.split(",")]

    monitor = SentinelMonitor(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
        target_column=args.target,
    )
    result = monitor.run()
    _print_report(result)

    if args.output:
        result.drift_report.to_csv(args.output, index=False)
        print(f"\nDrift report saved to {args.output}")

    sys.exit(0 if result.status != "ALERT" else 1)


def monitor_all_command(args):
    summary_df, results = run_batch(args.config)

    print("=" * 70)
    print("SENTINEL BATCH REPORT")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("-" * 70)

    n_alert = (summary_df["status"] == "ALERT").sum()
    n_error = (summary_df["status"] == "ERROR").sum()
    print(f"\n{len(summary_df)} job(s) checked: {n_alert} ALERT, {n_error} ERROR")

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"Batch summary saved to {args.output}")

    sys.exit(1 if (n_alert > 0 or n_error > 0) else 0)


def main():
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel: statistical ML model drift detection on any tabular dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor_parser = subparsers.add_parser("monitor", help="Compare two datasets for drift.")
    monitor_parser.add_argument("--reference", required=True, help="Path to reference (old) CSV")
    monitor_parser.add_argument("--current", required=True, help="Path to current (new) CSV")
    monitor_parser.add_argument("--features", required=True, help="Comma-separated feature column names")
    monitor_parser.add_argument("--target", default=None, help="Optional target column name (enables concept drift check)")
    monitor_parser.add_argument("--output", default=None, help="Optional path to save drift report CSV")
    monitor_parser.set_defaults(func=monitor_command)

    batch_parser = subparsers.add_parser("monitor-all", help="Run drift checks for multiple models from a config file.")
    batch_parser.add_argument("--config", required=True, help="Path to YAML config file listing jobs")
    batch_parser.add_argument("--output", default=None, help="Optional path to save batch summary CSV")
    batch_parser.set_defaults(func=monitor_all_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
