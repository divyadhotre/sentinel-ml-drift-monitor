"""
Sentinel: statistical ML model drift detection for any tabular dataset.

    from sentinel import SentinelMonitor

    result = SentinelMonitor(reference_df, current_df, feature_columns=[...]).run()
    print(result.status, result.summary)
"""

from .core import SentinelMonitor, SentinelResult
from . import metrics
from . import concept_drift
from . import performance
from . import alerting

__all__ = ["SentinelMonitor", "SentinelResult", "metrics", "concept_drift", "performance", "alerting"]
__version__ = "0.1.0"
