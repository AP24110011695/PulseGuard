"""MLflow pyfunc wrappers — self-contained, deployable PulseGuard artifacts.

These wrap the in-memory Phase 2 objects (AnomalyDetector / QuantileForecaster) so the
registered champion carries everything serving needs: the underlying models, score
normalization, the calibrated threshold, horizons/quantiles, and the expected feature
columns. Serving builds features with the shared ml.features code and hands the frame
to `predict` — no train/serve skew by construction.
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ml.anomaly import AnomalyDetector
from ml.forecast import QuantileForecaster


def _as_feature_frame(model_input, feature_names: list[str]) -> pd.DataFrame:
    if not isinstance(model_input, pd.DataFrame):
        raise ValueError("model_input must be a pandas DataFrame with the expected feature columns")
    missing = [c for c in feature_names if c not in model_input.columns]
    if missing:
        raise ValueError(f"model_input is missing feature columns: {missing}")
    return model_input


class AnomalyDetectorWrapper(mlflow.pyfunc.PythonModel):
    """Input: feature DataFrame. Output: DataFrame [score, is_anomaly]."""

    def __init__(self, detector: AnomalyDetector, feature_names: list[str]) -> None:
        self.if_model = detector.model
        self.score_mean = detector.score_mean
        self.score_std = detector.score_std
        self.threshold = detector.threshold
        self.feature_names = feature_names

    def predict(self, context, model_input, params=None):
        frame = _as_feature_frame(model_input, self.feature_names)
        raw = -self.if_model.decision_function(frame[self.feature_names].to_numpy())
        scores = (raw - self.score_mean) / (self.score_std + 1e-12)
        return pd.DataFrame({"score": scores, "is_anomaly": scores >= self.threshold})


class QuantileForecasterWrapper(mlflow.pyfunc.PythonModel):
    """Input: feature DataFrame plus an integer `horizon` column. Output: one row per
    input row with columns [horizon, q<tau>...], e.g. q0.05 / q0.5 / q0.95."""

    def __init__(self, forecaster: QuantileForecaster, feature_names: list[str]) -> None:
        self.models = forecaster.models
        self.horizons = tuple(forecaster.horizons)
        self.quantiles = tuple(forecaster.quantiles)
        self.feature_names = feature_names

    def predict(self, context, model_input, params=None):
        frame = _as_feature_frame(model_input, [*self.feature_names, "horizon"])
        outputs = []
        for horizon, group in frame.groupby("horizon", sort=True):
            horizon = int(horizon)
            if horizon not in self.horizons:
                raise ValueError(f"unsupported horizon {horizon}; model supports {self.horizons}")
            X = group[self.feature_names].to_numpy()
            row = {"horizon": np.full(len(group), horizon)}
            for tau in self.quantiles:
                row[f"q{tau}"] = self.models[(horizon, tau)].predict(X)
            outputs.append(pd.DataFrame(row))
        return pd.concat(outputs, ignore_index=True)
