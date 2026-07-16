"""
run_rolling_analysis.py

Reconstructs a real date column from the Jan/Apr 2020 taxi data (using
day_of_month + the known month of each file), combines both into one
continuous timeline, and runs Sentinel's rolling-window drift detection
across it -- producing a week-by-week drift trend instead of a single
before/after snapshot.

Run from the project root:
    python run_rolling_analysis.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import matplotlib.pyplot as plt

from sentinel.rolling import rolling_drift_report, overall_drift_timeline
from model import FEATURE_COLUMNS

train_df = pd.read_csv("data/processed/taxi_training_era.csv")
current_df = pd.read_csv("data/processed/taxi_current_era.csv")

# Reconstruct real dates: training file = January 2020, current file = April 2020
train_df = train_df.copy()
train_df["date"] = pd.to_datetime("2020-01-" + train_df["day_of_month"].astype(str).str.zfill(2))

current_df = current_df.copy()
current_df = current_df[current_df["day_of_month"] <= 30]  # April has only 30 days
current_df["date"] = pd.to_datetime("2020-04-" + current_df["day_of_month"].astype(str).str.zfill(2))

combined = pd.concat([train_df, current_df], ignore_index=True)

print(f"Combined dataset: {len(combined):,} rows spanning {combined['date'].min().date()} to {combined['date'].max().date()}")

report = rolling_drift_report(
    combined,
    date_column="date",
    feature_columns=FEATURE_COLUMNS,
    freq="W",
)
timeline = overall_drift_timeline(report)

print("\nWeek-by-week max PSI:")
print(timeline.to_string(index=False))

timeline.to_csv("reports/rolling_drift_timeline.csv", index=False)

# Chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(timeline)), timeline["max_psi"], marker="o", color="#0092ff", linewidth=2)
ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="Moderate drift threshold")
ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="Major drift threshold")

# Mark the Feb/Mar data gap honestly, since only Jan and Apr data exist
period_labels = timeline["period"].tolist()
gap_index = None
for i in range(len(period_labels) - 1):
    if "2020-01" in period_labels[i] and "2020-04" in period_labels[i + 1]:
        gap_index = i
        break
if gap_index is not None:
    ax.axvspan(gap_index + 0.5, gap_index + 0.5, color="gray", alpha=0.3)
    ax.axvline(gap_index + 0.5, color="gray", linestyle=":", linewidth=2, label="No data (Feb-Mar gap)")

ax.set_xticks(range(len(timeline)))
ax.set_xticklabels(timeline["period"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Max PSI across all features")
ax.set_title("Drift Accumulation Over Time: Jan 2020 -> Apr 2020 (Weekly, Full Weeks Only)")
ax.legend()
plt.tight_layout()
plt.savefig("reports/figures/07_rolling_drift_timeline.png", dpi=150)
plt.show()

print("\nSaved chart to reports/figures/07_rolling_drift_timeline.png")
print("Saved data to reports/rolling_drift_timeline.csv")
