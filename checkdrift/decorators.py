"""
Drift detection decorator for FastAPI endpoints.
"""

import functools
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

logger = logging.getLogger("driftdetect")


class DriftSeverity(Enum):
    """Drift severity levels."""
    OK = "ok"
    WARNING = "warning"  # Moderate drift detected
    ALARM = "alarm"      # Significant drift detected


@dataclass
class FeatureDriftResult:
    """Drift detection result for a single feature."""
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    wasserstein: float
    severity: DriftSeverity

    @property
    def is_drifted(self) -> bool:
        return self.severity != DriftSeverity.OK


@dataclass
class DriftCheckResult:
    """Complete drift check result passed to callback."""
    request_count: int
    features: dict[str, FeatureDriftResult] = field(default_factory=dict)

    @property
    def severity(self) -> DriftSeverity:
        """Overall severity (worst across all features)."""
        if any(f.severity == DriftSeverity.ALARM for f in self.features.values()):
            return DriftSeverity.ALARM
        if any(f.severity == DriftSeverity.WARNING for f in self.features.values()):
            return DriftSeverity.WARNING
        return DriftSeverity.OK

    @property
    def drifted_features(self) -> list[str]:
        return [name for name, f in self.features.items() if f.is_drifted]


# Type alias for callback
DriftCallback = Callable[[DriftCheckResult], None]


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    """
    _, bin_edges = np.histogram(reference, bins=bins)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions, avoid division by zero
    ref_pct = (ref_counts + 1) / (len(reference) + bins)
    curr_pct = (curr_counts + 1) / (len(current) + bins)

    # PSI formula: sum((curr - ref) * ln(curr / ref))
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


class DriftMonitor:
    """
    Monitors feature distributions for drift against a baseline.

    Maintains a sliding window of recent observations and periodically
    checks for drift using PSI, KS test, and Wasserstein distance.

    Thresholds (banking industry standard):
    - PSI > 0.2: Significant drift
    - PSI > 0.1: Moderate drift
    - KS p-value < 0.05: Statistically significant shift
    """

    PSI_WARNING = 0.1
    PSI_ALARM = 0.2
    KS_PVALUE_THRESHOLD = 0.05

    def __init__(
        self,
        baseline_path: str | Path,
        window_size: int = 100,
        check_interval: int = 50,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        callback: DriftCallback | None = None,
    ):
        self.baseline_path = Path(baseline_path)
        self.window_size = window_size
        self.check_interval = check_interval
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.callback = callback

        # Load baseline distributions
        with open(self.baseline_path) as f:
            self.baseline = json.load(f)

        self.features = list(self.baseline["distributions"].keys())

        # Sliding window per feature
        self.windows: dict[str, deque] = {
            feat: deque(maxlen=window_size) for feat in self.features
        }
        self.request_count = 0
        self.last_check_results: DriftCheckResult | None = None

    def push(self, values: dict[str, float]) -> None:
        """Add observation to sliding windows."""
        for feat in self.features:
            if feat in values:
                self.windows[feat].append(values[feat])
        self.request_count += 1

        if self.request_count % self.check_interval == 0:
            self._check_drift()

    def _determine_severity(self, psi: float, ks_pvalue: float) -> DriftSeverity:
        """Determine drift severity combining PSI and KS test."""
        if psi > self.PSI_ALARM:
            return DriftSeverity.ALARM
        if psi > self.PSI_WARNING and ks_pvalue < self.ks_pvalue_threshold:
            return DriftSeverity.ALARM
        if psi > self.PSI_WARNING or ks_pvalue < self.ks_pvalue_threshold:
            return DriftSeverity.WARNING
        return DriftSeverity.OK

    def _check_drift(self) -> None:
        """Run drift detection on current windows."""
        result = DriftCheckResult(request_count=self.request_count)

        for feat in self.features:
            if len(self.windows[feat]) < self.check_interval:
                continue

            current = np.array(self.windows[feat])
            reference = np.array(self.baseline["distributions"][feat])

            psi = compute_psi(reference, current)
            ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
            wasserstein = stats.wasserstein_distance(reference, current)

            severity = self._determine_severity(psi, ks_pvalue)

            result.features[feat] = FeatureDriftResult(
                feature=feat,
                psi=round(psi, 4),
                ks_stat=round(ks_stat, 4),
                ks_pvalue=round(ks_pvalue, 4),
                wasserstein=round(wasserstein, 4),
                severity=severity,
            )

        self.last_check_results = result

        if result.severity != DriftSeverity.OK:
            self._handle_drift(result)

    def _handle_drift(self, result: DriftCheckResult) -> None:
        """Log drift or invoke callback."""
        if self.callback:
            try:
                self.callback(result)
            except Exception as e:
                logger.error(f"[DriftDetect] Callback error: {e}")
        else:
            drifted = result.drifted_features
            severity = result.severity.value.upper()

            summary = {
                feat: {
                    "psi": r.psi,
                    "ks_p": r.ks_pvalue,
                    "wasserstein": r.wasserstein,
                }
                for feat, r in result.features.items()
                if r.is_drifted
            }

            log_msg = (
                f"Drift {severity} in {drifted} "
                f"after {result.request_count} requests: {summary}"
            )

            if result.severity == DriftSeverity.ALARM:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)


# Global registry of monitors
_monitors: dict[str, DriftMonitor] = {}


def check_drift(
    baseline: str | Path,
    window_size: int = 100,
    check_interval: int = 50,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
    on_drift: DriftCallback | None = None,
):
    """
    Decorator for drift detection on FastAPI endpoints.

    Monitors incoming request distributions against a baseline using
    industry-standard metrics:
    - PSI (Population Stability Index): >0.2 = significant, >0.1 = moderate
    - KS test p-value: <0.05 = statistically significant shift
    - Wasserstein distance: magnitude of distribution shift

    Args:
        baseline: Path to baseline JSON file with reference distributions
        window_size: Number of recent requests to keep in sliding window
        check_interval: Check drift every N requests
        psi_threshold: PSI threshold for drift alert (default 0.2)
        ks_pvalue_threshold: KS test p-value threshold (default 0.05)
        on_drift: Callback invoked when drift is detected

    Example:
        @app.post("/predict")
        @check_drift(baseline="baseline.json")
        async def predict(application: LoanApplication):
            return model.predict(application)
    """
    def decorator(func: Callable) -> Callable:
        endpoint_name = func.__name__

        if endpoint_name not in _monitors:
            _monitors[endpoint_name] = DriftMonitor(
                baseline_path=baseline,
                window_size=window_size,
                check_interval=check_interval,
                psi_threshold=psi_threshold,
                ks_pvalue_threshold=ks_pvalue_threshold,
                callback=on_drift,
            )

        monitor = _monitors[endpoint_name]

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for arg in kwargs.values():
                if hasattr(arg, "model_dump"):  # Pydantic v2
                    data = arg.model_dump()
                    features = {
                        k: float(v) for k, v in data.items()
                        if isinstance(v, (int, float)) and k in monitor.features
                    }
                    monitor.push(features)
                    break

            return await func(*args, **kwargs)

        wrapper.drift_monitor = monitor
        return wrapper

    return decorator
