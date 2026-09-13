"""Drift + lifecycle API: on-demand PSI/residual checks, status, events, promotion
history, and the controlled retraining trigger (background task)."""

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from ml.drift import feature_reference
from ml.features import FeatureConfig, build_features, feature_names

T0 = datetime(2026, 8, 16, tzinfo=UTC)


def create_series_with_points(client, unique_name, n_points=400):
    resp = client.post(
        "/api/v1/series",
        json={"name": unique_name, "unit": "ms", "source": "simulator"},
    )
    assert resp.status_code == 201
    series_id = resp.json()["id"]
    points = [
        {
            "ts": (T0 + timedelta(minutes=i)).isoformat(),
            "value": 50.0 + 8.0 * math.sin((i % 240) / 240.0 * 2 * math.pi),
        }
        for i in range(n_points)
    ]
    resp = client.post(f"/api/v1/series/{series_id}/points", json={"points": points})
    assert resp.status_code == 201
    return series_id, points


def test_drift_check_status_and_events(client, champion_models, unique_name):
    series_id, points = create_series_with_points(client, unique_name)

    # register a champion version carrying a reference for THIS series
    frame = pd.DataFrame(
        {
            "value": [p["value"] for p in points],
            "ts": pd.to_datetime([p["ts"] for p in points], utc=True),
        }
    )
    cfg = FeatureConfig()
    feats = build_features(frame["value"], cfg, frame["ts"])
    reference = {unique_name: feature_reference(feats.to_numpy(), feature_names(cfg))}
    champion_models["register"](reference)
    client.post("/api/v1/models/reload")

    resp = client.post(
        "/api/v1/drift/check",
        json={"series_id": series_id, "task": "forecast", "window_points": 200},
    )
    assert resp.status_code == 200
    body = resp.json()
    psi_checks = [c for c in body["checks"] if c["kind"] == "psi"]
    assert psi_checks[0]["status"] in ("ok", "warn", "drift")
    assert body["events_recorded"] >= 1

    status = client.get("/api/v1/drift/status", params={"series_id": series_id}).json()
    assert status["series"][0]["series_id"] == series_id
    assert status["series"][0]["psi"]["status"] in ("ok", "warn", "drift")
    assert status["series"][0]["overall"] in ("ok", "warn", "drift")

    events = client.get("/api/v1/drift/events", params={"series_id": series_id}).json()
    assert events["count"] >= 1
    assert events["events"][0]["kind"] in ("psi", "residual")


def test_drift_check_requires_reference(client, champion_models, unique_name):
    """A champion without a stored reference reports a skipped check, not a crash."""
    series_id, _ = create_series_with_points(client, unique_name)
    resp = client.post(
        "/api/v1/drift/check",
        json={"series_id": series_id, "task": "forecast", "window_points": 200},
    )
    assert resp.status_code == 200
    psi_checks = [c for c in resp.json()["checks"] if c["kind"] == "psi"]
    assert psi_checks[0]["status"] == "skipped"


def test_retraining_trigger_records_decision(client, champion_models, unique_name, tmp_path):
    series_id, _ = create_series_with_points(client, unique_name)

    # a tiny retraining config so the background job finishes quickly
    sim_config = tmp_path / "sim.yaml"
    sim_config.write_text(
        """
seed: 21
start: "2026-08-16T00:00:00Z"
days: 6
series:
  - name: trigger_metric
    unit: x
    base: 50.0
    daily_amplitude: 4.0
    noise_sigma: 0.8
    spikes: { count: 3, magnitude: [6.0, 10.0], width_minutes: [1, 2] }
    level_shifts: { count: 1, magnitude: 10.0, transition_minutes: 10, at_day: 4.5 }
    variance_changes: { count: 0, sigma_multiplier: 1.0, duration_minutes: 1 }
    contextual: { count: 0, magnitude_fraction: [0.0, 0.0] }
""",
        encoding="utf-8",
    )
    train_config = tmp_path / "train_trigger.yaml"
    train_config.write_text(
        f"""
seed: 21
dataset:
  source: simulator
  simulator_config: {sim_config.as_posix()}
  points_per_day: 1440
splits:
  mode: days
  train_days: 3
  eval_days: 2
  golden_days: 1
  refit_every_days: 1
features:
  lags: [1, 2, 3, 5]
  rolling_windows: [5, 10]
  rolling_stats: [mean, std]
  roc_offsets: [1]
  calendar: true
anomaly:
  n_estimators: 50
  contamination: 0.01
  calibration: best_f1
  percentile_fallback: 0.99
  zscore_window: 10
forecast:
  horizons: [1, 5]
  quantiles: [0.05, 0.5, 0.95]
  val_fraction: 0.2
  params:
    learning_rate: 0.1
    num_leaves: 15
    n_estimators: 30
    min_data_in_leaf: 10
    early_stopping_rounds: 10
gate:
  min_relative_improvement: 0.02
  max_regression: 0.01
  band_nominal_coverage: 0.90
  coverage_tolerance: 0.50
  auc_tolerance: 0.005
  f1_tolerance: 0.02
  min_train_rows: 100
mape_epsilon: 1.0
report_dir: {tmp_path.as_posix()}
""",
        encoding="utf-8",
    )

    resp = client.post(
        "/api/v1/retraining/trigger",
        json={"task": "forecast", "config": str(train_config), "challenger_window": "expanding"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"

    # TestClient executes the background task before returning; the decision is recorded
    promotions = client.get("/api/v1/promotions").json()
    assert promotions["count"] >= 1
    decision = promotions["decisions"][0]
    assert decision["task"] == "forecast"
    assert decision["decision"] in ("promoted", "rejected")
    assert "champion_golden" in decision["metrics"]
    assert "rule_trace" in decision["metrics"]


def test_retraining_trigger_validates_task(client):
    resp = client.post(
        "/api/v1/retraining/trigger", json={"task": "not_a_task"}
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_task"
