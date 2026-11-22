"""Climate anomaly detection module using Isolation Forest.

Provides VPD calculation, feature scaling, isolation forest scoring,
and detection of unusual environmental features.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Union
import numpy as np
import pandas as pd


class AnomalyResult(tuple):
    """Result structure supporting tuple unpacking, dict lookup, and attribute access."""
    __slots__ = ()

    def __new__(
        cls,
        status: str,
        severity: str,
        score: float,
        threshold: float,
        vpd: float,
        unusual_features: List[str]
    ) -> AnomalyResult:
        return super().__new__(cls, (status, severity, score, threshold, vpd, unusual_features))

    @property
    def status(self) -> str:
        return self[0]

    @property
    def severity(self) -> str:
        return self[1]

    @property
    def score(self) -> float:
        return self[2]

    @property
    def threshold(self) -> float:
        return self[3]

    @property
    def vpd(self) -> float:
        return self[4]

    @property
    def unusual_features(self) -> List[str]:
        return self[5]

    def __getitem__(self, key: Union[int, str]) -> Any:
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "severity": self.severity,
            "score": self.score,
            "threshold": self.threshold,
            "vpd": self.vpd,
            "unusual_features": self.unusual_features,
        }

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(status='{self.status}', severity='{self.severity}', "
            f"score={self.score:.4f}, threshold={self.threshold:.4f}, "
            f"vpd={self.vpd:.4f}, unusual_features={self.unusual_features})"
        )


def calculate_vpd(temperature: float, relative_humidity: float) -> float:
    """Calculate Vapor Pressure Deficit (VPD) in kPa from temperature and RH.

    Formula: VPD = 0.6108 * exp(17.27 * T / (T + 237.3)) * (1 - RH / 100)
    """
    svp = 0.6108 * math.exp(17.27 * temperature / (temperature + 237.3))
    vpd = svp * (1.0 - relative_humidity / 100.0)
    return float(vpd)


def score_reading(
    T: float,
    RH: float,
    bundle: Dict[str, Any]
) -> AnomalyResult:
    """Score a single environmental reading using the Isolation Forest inference bundle.

    Parameters
    ----------
    T : float
        Temperature in °C.
    RH : float
        Relative humidity in %.
    bundle : dict
        Inference bundle dictionary with keys:
        'model', 'imputer', 'scaler', 'features', 'threshold',
        'high_severity_threshold', 'vpd_formula'.

    Returns
    -------
    AnomalyResult (NamedTuple)
        Tuple of (status, severity, score, threshold, vpd, unusual_features)
        which also allows dict-like and attribute access.
    """
    vpd = calculate_vpd(float(T), float(RH))
    features = list(bundle.get("features", ["T", "RH", "VPD"]))

    # Prepare input 2D array for imputer and scaler
    raw_dict = {"T": float(T), "RH": float(RH), "VPD": float(vpd)}
    x_raw = np.array([[raw_dict.get(f, 0.0) for f in features]], dtype=float)

    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    # Transform through imputer and scaler
    x_imputed = imputer.transform(x_raw)
    x_scaled = scaler.transform(x_imputed)

    # Negative score_samples: higher value indicates higher anomaly
    score = float(-model.score_samples(x_scaled)[0])
    threshold = float(bundle.get("threshold", 0.6315))
    high_threshold = float(bundle.get("high_severity_threshold", threshold))

    is_anomaly = score > threshold
    if is_anomaly:
        status = "Anomaly"
        severity = "HIGH" if score > high_threshold else "MEDIUM"
    else:
        status = "Normal"
        severity = "NORMAL"

    # Identify the 2 features with largest absolute scaled value
    scaled_values = x_scaled[0]
    feature_abs_pairs = sorted(
        zip(features, [abs(float(v)) for v in scaled_values]),
        key=lambda pair: pair[1],
        reverse=True
    )
    unusual_features = [pair[0] for pair in feature_abs_pairs[:2]]

    return AnomalyResult(
        status=status,
        severity=severity,
        score=score,
        threshold=threshold,
        vpd=vpd,
        unusual_features=unusual_features,
    )
