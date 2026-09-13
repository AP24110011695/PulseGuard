"""MLflow registration + champion alias flow, exercised on a sqlite tracking store."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from ml import tracking
from ml.anomaly import fit_anomaly_detector
from ml.features import FeatureConfig, build_features, feature_names
from ml.forecast import QuantileForecaster, fit_quantile_models
from ml.models import AnomalyDetectorWrapper, QuantileForecasterWrapper


def _series(n=900):
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    wave = 50.0 + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 240.0)
    values = pd.Series(wave + rng.normal(0, 0.8, n))
    values.iloc[400:410] += 25.0  # obvious anomalies
    labels = np.zeros(n, dtype=bool)
    labels[400:410] = True
    return values, ts, labels


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory):
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module")
def wrappers(tracking_uri):
    values, ts, labels = _series()
    cfg = FeatureConfig()
    feats = build_features(values, cfg, ts)
    rows = feats.index.to_numpy()
    # drop the last feature row: its h=1 target lies beyond the series end
    usable = rows + 1 < len(values)
    rows = rows[usable]
    X = feats.to_numpy()[usable]

    detector, _ = fit_anomaly_detector(
        X, X, labels[rows],
        n_estimators=60, contamination=0.02, seed=7,
    )
    anomaly_wrapper = AnomalyDetectorWrapper(detector, feature_names(cfg))

    # target aligned to feature rows: the row labeled L predicts values[L + 1]
    y_x = values.to_numpy()[rows + 1]
    fit_pos, val_pos = np.arange(len(X) - 150), np.arange(len(X) - 150, len(X))
    models = fit_quantile_models(
        X[fit_pos], y_x[fit_pos],
        X[val_pos], y_x[val_pos],
        quantiles=(0.05, 0.5, 0.95),
        params={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15,
                "min_data_in_leaf": 10, "early_stopping_rounds": 10},
        seed=7,
    )
    forecaster = QuantileForecaster(horizons=(1,), quantiles=(0.05, 0.5, 0.95))
    for tau, model in models.items():
        forecaster.add(1, tau, model)
    forecast_wrapper = QuantileForecasterWrapper(forecaster, feature_names(cfg))
    return {"anomaly": anomaly_wrapper, "forecast": forecast_wrapper}, feats, values, labels


def test_register_and_resolve_champion(tracking_uri, wrappers):
    wrapper_dict, *_ = wrappers
    name, version, run_id = tracking.register_champion(
        "forecast", wrapper_dict["forecast"], seed=7, config_sha256="a" * 64,
        feature_config={"calendar": False}, horizons=(1,),
        quantiles=(0.05, 0.5, 0.95),
    )
    assert name == "pulseguard-forecaster"
    assert version >= 1 and run_id

    resolved_name, resolved_version = tracking.get_champion("forecast")
    assert (resolved_name, resolved_version) == (name, version)


def test_alias_move_bumps_champion(tracking_uri, wrappers):
    wrapper_dict, *_ = wrappers
    _, first_version, _ = tracking.register_champion(
        "forecast", wrapper_dict["forecast"], seed=7, config_sha256="b" * 64, feature_config={},
        horizons=(1,), quantiles=(0.05, 0.5, 0.95),
    )
    name, second_version, _ = tracking.register_champion(
        "forecast", wrapper_dict["forecast"], seed=7, config_sha256="c" * 64, feature_config={},
        horizons=(1,), quantiles=(0.05, 0.5, 0.95),
    )
    assert second_version == first_version + 1
    assert tracking.get_champion("forecast")[1] == second_version


def test_loaded_champion_runs_predictions(tracking_uri, wrappers):
    wrapper_dict, feats, values, _ = wrappers
    tracking.register_champion(
        "forecast", wrapper_dict["forecast"], seed=7, config_sha256="d" * 64, feature_config={},
        horizons=(1,), quantiles=(0.05, 0.5, 0.95),
    )
    _, _, model = tracking.load_champion("forecast")

    sample = feats.iloc[[-1]].copy()
    sample["horizon"] = 1
    prediction = model.predict(sample)
    assert {"horizon", "q0.05", "q0.5", "q0.95"} <= set(prediction.columns)
    actual = values.iloc[-1]  # the row labeled n-1 predicts the value at n-1 + 1
    assert abs(prediction.iloc[0]["q0.5"] - actual) < 5.0  # the wave is learnable


def test_anomaly_wrapper_output_shape(tracking_uri, wrappers):
    wrapper_dict, feats, _, labels = wrappers
    detector_wrapper = wrapper_dict["anomaly"]
    tracking.register_champion(
        "anomaly", detector_wrapper, seed=7, config_sha256="e" * 64, feature_config={},
        threshold=detector_wrapper.threshold,
    )
    _, _, model = tracking.load_champion("anomaly")
    prediction = model.predict(feats)
    assert {"score", "is_anomaly"} <= set(prediction.columns)
    assert (prediction["is_anomaly"] == (prediction["score"] >= detector_wrapper.threshold)).all()
    # the injected spike block scores above the normal bulk
    scores = prediction["score"].to_numpy()
    assert np.median(scores[labels[feats.index]]) > np.median(scores)


def test_unknown_horizon_rejected(tracking_uri, wrappers):
    wrapper_dict, feats, *_ = wrappers
    sample = feats.iloc[[-1]].copy()
    sample["horizon"] = 99
    with pytest.raises(ValueError, match="unsupported horizon"):
        wrapper_dict["forecast"].predict(None, sample)
