"""Anomaly detection: Isolation Forest (primary) and a rolling z-score baseline.

The deployable artifact is a static `AnomalyDetector`: an IsolationForest plus score
normalization plus a single calibrated threshold — exactly what the serving API loads
in Phase 3. Scores are the negated decision function, standardized on training-window
scores. The threshold is calibrated on a labeled calibration segment at the tail of the
initial training region (best-F1), with a score-percentile fallback for unlabeled data.

Evaluation (Phase 2) runs the STATIC detector over daily folds of the walk-forward
evaluation region. This mirrors serving reality: a registered champion stays fixed
between promotions; adaptation to drift is Phase 4's explicit retraining flow, not a
silent refit inside evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyDetector:
    model: IsolationForest
    score_mean: float
    score_std: float
    threshold: float

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_scores(X)
        return (raw - self.score_mean) / (self.score_std + 1e-12)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) >= self.threshold


def calibrate_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing F1 on a labeled calibration segment.

    Scans every distinct score as a candidate boundary (pred = score >= t) using
    suffix sums — the exact optimum over thresholds, in O(n log n).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float(scores.min())

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n = len(scores)

    unique_scores, start_idx = np.unique(sorted_scores, return_index=True)
    tp = np.cumsum(sorted_labels[::-1])[::-1][start_idx]  # positives at score >= t
    pred_count = n - start_idx
    precision = tp / np.maximum(pred_count, 1)
    recall = tp / n_pos
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return float(unique_scores[int(np.argmax(f1))])


def percentile_threshold(scores: np.ndarray, quantile: float) -> float:
    """Unsupervised fallback: flag the top (1 - quantile) fraction of scores."""
    return float(np.quantile(np.asarray(scores, dtype=float), quantile))


def fit_anomaly_detector(
    X_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    *,
    n_estimators: int,
    contamination: float,
    seed: int,
    calibration: str = "best_f1",
    percentile_fallback: float = 0.99,
    n_jobs: int = -1,
) -> tuple[AnomalyDetector, str]:
    """Fit the detector; returns (detector, threshold_method_note)."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train)
    detector = AnomalyDetector(model=model, score_mean=0.0, score_std=1.0, threshold=0.0)
    raw_train = detector.raw_scores(X_train)
    detector.score_mean = float(np.mean(raw_train))
    detector.score_std = float(np.std(raw_train) + 1e-12)

    y_calibration = np.asarray(y_calibration, dtype=bool)
    calibration_scores = detector.score(X_calibration)
    if calibration == "percentile" or not y_calibration.any():
        detector.threshold = percentile_threshold(calibration_scores, percentile_fallback)
        note = f"percentile_{percentile_fallback}"
        if not y_calibration.any():
            note += "_fallback_no_positive_labels"
    else:
        detector.threshold = calibrate_best_f1(calibration_scores, y_calibration)
        note = "best_f1"
    return detector, note


class RollingZScoreDetector:
    """Baseline detector: |v_t - trailing mean| / trailing std, window excludes v_t."""

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def score(self, values: pd.Series) -> pd.Series:
        v = values.astype(float)
        past = v.shift(1).rolling(self.window, min_periods=self.window)
        return ((v - past.mean()) / (past.std(ddof=0) + 1e-9)).abs()
