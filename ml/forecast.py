"""Quantile forecasting with LightGBM plus naive baselines.

One LightGBM model is trained per (horizon, quantile) with the quantile objective;
together they form a single deployable `QuantileForecaster` artifact (Phase 3 wraps it
as one MLflow model that returns the full band). Per-fold training uses a time-ordered
validation tail for early stopping — never the evaluation region, never the golden set.

Quantile models are trained independently, so the band can occasionally cross
(p05 > p50); crossing rates are measured and reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np


@dataclass
class QuantileForecaster:
    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]
    models: dict = field(default_factory=dict)  # (horizon, quantile) -> LGBMRegressor

    def add(self, horizon: int, quantile: float, model: lgb.LGBMRegressor) -> None:
        self.models[(horizon, quantile)] = model

    def predict(self, X: np.ndarray, horizon: int) -> dict[float, np.ndarray]:
        return {q: self.models[(horizon, q)].predict(X) for q in self.quantiles}


def fit_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: tuple[float, ...],
    params: dict,
    seed: int,
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM quantile regressor per quantile with early stopping."""
    base = dict(params)
    early_stopping_rounds = int(base.pop("early_stopping_rounds", 50))
    base.update(
        objective="quantile",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    models: dict[float, lgb.LGBMRegressor] = {}
    for tau in quantiles:
        model = lgb.LGBMRegressor(**base, alpha=tau)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        models[tau] = model
    return models


def naive_forecast(values: np.ndarray) -> np.ndarray:
    """Prediction for v_{t+h} made at row t: the last observed value v_t (row-aligned)."""
    return np.asarray(values, dtype=float)


def seasonal_naive_forecast(values: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Prediction for v_{t+h}: v_{t+h-season} (e.g. the value one season earlier)."""
    v = np.asarray(values, dtype=float)
    pred = np.full(len(v), np.nan)
    lag = season - horizon
    if lag < 0:
        raise ValueError("horizon must not exceed the season length")
    if lag < len(v):
        pred[lag:] = v[: len(v) - lag]
    return pred


def band_crossing_rate(preds_by_quantile: dict[float, np.ndarray]) -> float:
    """Fraction of rows where the ordered-quantile assumption (low <= median <= high) breaks."""
    quantiles = sorted(preds_by_quantile)
    if len(quantiles) < 3:
        return 0.0
    low = preds_by_quantile[quantiles[0]]
    median = preds_by_quantile[min(quantiles, key=lambda q: abs(q - 0.5))]
    high = preds_by_quantile[quantiles[-1]]
    crossings = np.sum((low > median) | (median > high))
    return float(crossings / len(median))
