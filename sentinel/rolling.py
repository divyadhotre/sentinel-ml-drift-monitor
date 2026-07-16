"""
sentinel/rolling.py

ROLLING-WINDOW MONITORING -- turns Sentinel from a single "before vs after"
snapshot comparison into genuine monitoring over time.

Real production drift monitoring doesn't get two neat CSVs labeled "old"
and "new" -- it gets a continuous stream of data, and has to answer:
"as of THIS week, how much has drift accumulated so far?"

This module splits a time-stamped dataset into periods (e.g. weekly), and
computes drift for each period against a FIXED reference period, producing
a drift-score-over-time series -- the same shape of output a real
monitoring dashboard would show.
"""

import pandas as pd

from .metrics import generate_drift_report


def split_into_periods(df, date_column, freq="W", require_full_period=True):
    """
    Splits a dataframe into time periods based on a date column.
    freq follows pandas offset aliases: 'D' (daily), 'W' (weekly),
    'M' (monthly), etc.

    If require_full_period is True, periods that don't span a full window
    (e.g. a truncated week at the start/end of available data, containing
    only 3-4 days instead of 7) are dropped. Partial periods can have
    skewed weekday/weekend composition and produce unstable, misleading
    drift scores when compared against full periods.

    Returns a dict of {period_label: sub_dataframe}.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["_period"] = df[date_column].dt.to_period(freq)

    periods = {}
    for period, group in df.groupby("_period"):
        if require_full_period:
            days_present = group[date_column].dt.date.nunique()
            expected_days = (period.end_time - period.start_time).days + 1
            if days_present < expected_days * 0.85:  # allow small gaps, reject clearly-truncated periods
                continue
        periods[str(period)] = group.drop(columns="_period")
    return periods


def rolling_drift_report(df, date_column, feature_columns, reference_period=None, freq="W", n_bins=10, min_rows=30):
    """
    Computes drift for each time period in df against a fixed reference
    period, producing a long-format DataFrame suitable for a time-series
    chart: columns = [period, feature, psi, psi_verdict, ...].

    reference_period: label of the period to use as reference (e.g. the
    output of split_into_periods' first key). If None, uses the earliest
    period automatically.

    Periods with fewer than `min_rows` rows are skipped, since small or
    partial (e.g. truncated boundary) periods can produce unstable,
    misleadingly large PSI values due to non-representative sampling.
    """
    periods = split_into_periods(df, date_column, freq=freq)
    period_labels = sorted(periods.keys())

    if not period_labels:
        raise ValueError("No periods found -- check date_column and freq.")

    reference_period = reference_period or period_labels[0]
    if reference_period not in periods:
        raise ValueError(f"reference_period '{reference_period}' not found among periods: {period_labels}")

    reference_df = periods[reference_period]

    rows = []
    for label in period_labels:
        current_df = periods[label]
        if len(current_df) < min_rows:
            continue  # skip periods with too little/partial data to be statistically reliable

        report = generate_drift_report(reference_df, current_df, feature_columns, n_bins=n_bins)
        report.insert(0, "period", label)
        report.insert(1, "n_rows", len(current_df))
        rows.append(report)

    return pd.concat(rows, ignore_index=True)


def overall_drift_timeline(rolling_report):
    """
    Collapses the per-feature rolling report into one row per period,
    showing the MAX PSI across all features for that period -- the
    single number a time-series chart of 'overall drift over time' needs.
    """
    timeline = (
        rolling_report.groupby(["period", "n_rows"])["psi"]
        .max()
        .reset_index()
        .rename(columns={"psi": "max_psi"})
        .sort_values("period")
        .reset_index(drop=True)
    )
    return timeline
