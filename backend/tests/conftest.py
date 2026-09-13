import uuid

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

import backend.app.models  # noqa: F401  (registers tables on the shared metadata)
from backend.app.core.config import settings
from backend.app.core.database import Base, get_db
from backend.app.main import app

TEST_DB_NAME = "pulseguard_test"


@pytest.fixture(scope="session")
def test_engine():
    url = make_url(settings.database_url)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB_NAME}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin.dispose()

    engine = create_engine(url.set(database=TEST_DB_NAME))
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="session")
def mlflow_env(tmp_path_factory):
    """A sqlite-backed MLflow store, wired into settings AND the environment (the ml
package's tracking helpers read the env var) before the app starts."""
    import os

    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    old = settings.mlflow_tracking_uri
    old_env = os.environ.get("MLFLOW_TRACKING_URI")
    settings.mlflow_tracking_uri = uri
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    yield uri
    settings.mlflow_tracking_uri = old
    if old_env is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = old_env
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="session")
def champion_models(mlflow_env):
    """Register tiny champion models (default FeatureConfig) so the API can serve."""
    from ml import tracking
    from ml.anomaly import fit_anomaly_detector
    from ml.features import FeatureConfig, build_features, feature_names
    from ml.forecast import QuantileForecaster, fit_quantile_models
    from ml.models import AnomalyDetectorWrapper, QuantileForecasterWrapper

    n = 900
    rng = np.random.default_rng(5)
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    wave = 50.0 + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 240.0)
    values = pd.Series(wave + rng.normal(0, 0.8, n))
    labels = np.zeros(n, dtype=bool)
    labels[400:410] = True

    cfg = FeatureConfig()
    feats = build_features(values, cfg, ts)
    rows = feats.index.to_numpy()
    usable = rows + 1 < n  # drop the last row: its h=1 target is beyond the series
    rows = rows[usable]
    X = feats.to_numpy()[usable]
    feature_config = {
        "lags": list(cfg.lags),
        "rolling_windows": list(cfg.rolling_windows),
        "rolling_stats": list(cfg.rolling_stats),
        "roc_offsets": list(cfg.roc_offsets),
        "calendar": cfg.calendar,
    }

    detector, _ = fit_anomaly_detector(
        X, X, labels[rows],
        n_estimators=60, contamination=0.02, seed=7,
    )
    _, anomaly_version, _ = tracking.register_champion(
        "anomaly", AnomalyDetectorWrapper(detector, feature_names(cfg)),
        seed=7, config_sha256="0" * 64, feature_config=feature_config,
        threshold=detector.threshold, run_name="test-anomaly-champion",
    )

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
    _, forecast_version, _ = tracking.register_champion(
        "forecast", QuantileForecasterWrapper(forecaster, feature_names(cfg)),
        seed=7, config_sha256="0" * 64, feature_config=feature_config,
        horizons=(1,), quantiles=(0.05, 0.5, 0.95), run_name="test-forecast-champion",
    )
    state = {
        "uri": mlflow_env,
        "anomaly_version": anomaly_version,
        "forecast_version": forecast_version,
        "config_sha256": "0" * 64,
        "feature_config": feature_config,
    }

    def register(reference):
        return _register_with_ref(state, forecaster, reference)

    state["register"] = register
    yield state


def _register_with_ref(state, forecaster, reference):
    """Register a new champion version (optionally carrying a drift reference) and keep
    the recorded champion version current for later assertions."""
    from ml import tracking
    from ml.features import FeatureConfig, feature_names
    from ml.models import QuantileForecasterWrapper

    name, version, _ = tracking.register_champion(
        "forecast",
        QuantileForecasterWrapper(
            forecaster, feature_names(FeatureConfig(**state["feature_config"]))
        ),
        seed=7,
        config_sha256=state["config_sha256"],
        feature_config=state["feature_config"],
        horizons=(1,),
        quantiles=(0.05, 0.5, 0.95),
        reference={"series": reference} if reference is not None else None,
        run_name="test-forecast-champion",
    )
    state["forecast_version"] = version
    return version


@pytest.fixture(scope="session")
def client(test_engine, champion_models):
    TestSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.state.session_factory = TestSession  # background tasks use the test DB too
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def unique_name() -> str:
    return f"test_series_{uuid.uuid4().hex[:10]}"
