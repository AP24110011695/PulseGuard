import numpy as np
import pandas as pd
import pytest

from ml.anomaly import (
    RollingZScoreDetector,
    fit_anomaly_detector,
    percentile_threshold,
)
from ml.features import FeatureConfig, build_features
from ml.forecast import (
    QuantileForecaster,
    band_crossing_rate,
    fit_quantile_models,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_anomaly_detector_flags_outliers_with_calibrated_threshold():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(400, 2))
    # sparse scatter, not a tight cluster: Isolation Forest isolates individually
    # isolated points, so clustered "outliers" would legitimately be hard to rank
    outliers = rng.uniform(6.0, 10.0, size=(40, 2))
    X = np.vstack([normal, outliers])
    labels = np.array([False] * 400 + [True] * 40)

    detector, note = fit_anomaly_detector(
        X, X, labels, n_estimators=200, contamination=0.02, seed=7, calibration="best_f1"
    )
    assert note == "best_f1"
    pred = detector.predict_labels(X)
    metrics_f1 = 2 * (pred & labels).sum() / (pred.sum() + labels.sum())
    assert metrics_f1 > 0.8
    # scores are standardized: threshold lives on the normalized scale
    scores = detector.score(X)
    assert (scores >= detector.threshold).sum() == pred.sum()


def test_percentile_threshold_fallback_when_no_positive_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(300, 2))
    detector, note = fit_anomaly_detector(
        X, X, np.zeros(len(X), dtype=bool),
        n_estimators=50, contamination=0.01, seed=7, calibration="best_f1",
    )
    assert "fallback" in note
    expected = percentile_threshold(detector.score(X), 0.99)
    assert detector.threshold == pytest.approx(expected)


def test_rolling_zscore_baseline_detects_spike():
    values = pd.Series(np.zeros(200))
    values.iloc[150] = 10.0  # spike against a flat context
    z = RollingZScoreDetector(window=30).score(values)
    assert z.iloc[150] > 3.0
    assert np.isnan(z.iloc[:30]).all()  # warmup


def _tiny_forecast_setup():
    rng = np.random.default_rng(2)
    n = 2500
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    wave = 50.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / 1440.0)
    values = pd.Series(wave + rng.normal(0, 0.5, n))
    cfg = FeatureConfig(lags=(1, 2, 3), rolling_windows=(), roc_offsets=(), calendar=False)
    feats = build_features(values, cfg, ts)
    return values.to_numpy(), feats, cfg


def test_quantile_forecaster_band_is_mostly_ordered():
    values, feats, cfg = _tiny_forecast_setup()
    rows = feats.index.to_numpy()
    usable = rows + 1 < len(values)
    rows = rows[usable]
    X = feats.to_numpy()[usable]
    # target aligned to feature rows: the row labeled L predicts values[L + 1]
    y_x = values[rows + 1]
    fit_pos = np.flatnonzero(rows < 2000)[:-150]
    val_pos = np.flatnonzero(rows < 2000)[-150:]
    pred_pos = np.flatnonzero((rows >= 2000) & (rows < 2400))

    forecaster = QuantileForecaster(horizons=(1,), quantiles=(0.05, 0.5, 0.95))
    models = fit_quantile_models(
        X[fit_pos], y_x[fit_pos],
        X[val_pos], y_x[val_pos],
        quantiles=(0.05, 0.5, 0.95),
        params={"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15,
                "min_data_in_leaf": 20, "early_stopping_rounds": 10},
        seed=7,
    )
    for tau, model in models.items():
        forecaster.add(1, tau, model)
    preds = forecaster.predict(X[pred_pos], 1)

    # independent quantile models cross occasionally; this 40-tree toy sits around
    # 10% — the real evaluation reports the measured rate rather than assuming order
    assert band_crossing_rate(preds) < 0.20
    y_true = y_x[pred_pos]
    assert all(np.isfinite(arr).all() for arr in preds.values())
    assert np.mean(np.abs(preds[0.5] - y_true)) < 1.5  # a sine wave is learnable


def test_naive_forecast_is_last_value():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(naive_forecast(values), values)


def test_seasonal_naive_forecast_alignment():
    values = np.arange(10, dtype=float)
    pred = seasonal_naive_forecast(values, horizon=1, season=4)
    assert np.isnan(pred[:3]).all()
    assert np.array_equal(pred[3:], values[:7])  # pred[t] = values[t + 1 - 4]
