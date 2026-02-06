"""
driftdetect - One-line drift detection for ML APIs.

"Like Pydantic or a rate limiter, but for data drift."
"""

from .decorators import (
    check_drift,
    DriftMonitor,
    DriftCallback,
    DriftCheckResult,
    DriftSeverity,
    FeatureDriftResult,
)

__version__ = "1.0.0"

__all__ = [
    "check_drift",
    "DriftMonitor",
    "DriftCallback",
    "DriftCheckResult",
    "DriftSeverity",
    "FeatureDriftResult",
]
