"""
sentinel/core.py

SentinelMonitor -- the single, generic entry point tying together data
drift, concept drift, and label-free performance estimation. Works on any
two pandas DataFrames the caller supplies: no hardcoded column names, no
domain assumptions.

Example:
    from sentinel import SentinelMonitor

    result = SentinelMonitor(
        reference_df=old_data,
        current_df=new_data,
        feature_columns=["distance", "hour", "price"],
        target_column="duration",      # optional
        model=my_trained_sklearn_model,  # optional
    ).run()

    print(result.status)          # "OK" / "WATCH" / "ALERT"
    print(result.drift_report)    # PSI/KS/KL table
    print(result.summary)         # plain-English verdict
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import pandas as pd

from .metrics import generate_drift_report
from .concept_drift import generate_concept_drift_report

PSI_WATCH_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25


@dataclass
class SentinelResult:
    status: str
    max_psi: float
    drift_report: pd.DataFrame
    summary: str
    performance: Optional[dict] = None
    concept_drift: Optional[dict] = None


class SentinelMonitor:
    """
    Generic ML drift monitor. Point it at any two datasets with matching
    feature columns and it will report data drift (always), concept drift
    (if a target_column is supplied), and model performance (if both a
    target_column and a trained model are supplied).
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        model: Optional[Any] = None,
        n_bins: int = 10,
    ):
        missing_ref = [c for c in feature_columns if c not in reference_df.columns]
        missing_cur = [c for c in feature_columns if c not in current_df.columns]
        if missing_ref or missing_cur:
            raise ValueError(
                f"feature_columns not found -- reference missing: {missing_ref}, "
                f"current missing: {missing_cur}"
            )

        self.reference_df = reference_df
        self.current_df = current_df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model = model
        self.n_bins = n_bins

    def _overall_status(self, drift_report):
        max_psi = float(drift_report["psi"].max())
        if max_psi >= PSI_ALERT_THRESHOLD:
            return "ALERT", max_psi
        elif max_psi >= PSI_WATCH_THRESHOLD:
            return "WATCH", max_psi
        return "OK", max_psi

    def _build_summary(self, drift_report, status, max_psi, performance):
        drifted = drift_report[drift_report["psi"] >= PSI_WATCH_THRESHOLD]
        if status == "OK":
            msg = "No meaningful drift detected. Model inputs still resemble the reference distribution."
        else:
            names = ", ".join(drifted["feature"].tolist())
            msg = f"{status}: drift detected in [{names}] (highest PSI = {max_psi:.2f})."
            msg += " Retraining recommended." if status == "ALERT" else " Monitor closely."
        if performance is not None:
            msg += f" Current MAE: {performance['mae']:.3f}, R^2: {performance['r2']:.3f}."
        return msg

    def run(self) -> SentinelResult:
        drift_report = generate_drift_report(
            self.reference_df, self.current_df, self.feature_columns, n_bins=self.n_bins
        )
        status, max_psi = self._overall_status(drift_report)

        performance = None
        if self.model is not None and self.target_column is not None and self.target_column in self.current_df.columns:
            from sklearn.metrics import mean_absolute_error, r2_score
            X = self.current_df[self.feature_columns]
            y = self.current_df[self.target_column]
            preds = self.model.predict(X)
            performance = {"mae": float(mean_absolute_error(y, preds)), "r2": float(r2_score(y, preds))}

        concept_drift = None
        if self.target_column is not None and self.target_column in self.reference_df.columns and self.target_column in self.current_df.columns:
            concept_drift = generate_concept_drift_report(
                self.reference_df, self.current_df, self.feature_columns, self.target_column
            )

        summary = self._build_summary(drift_report, status, max_psi, performance)

        return SentinelResult(
            status=status,
            max_psi=max_psi,
            drift_report=drift_report,
            summary=summary,
            performance=performance,
            concept_drift=concept_drift,
        )
