"""Alert feed: anomaly-hit alerts, forecast band breaches, drift alerts, ack flow."""

from datetime import UTC, datetime, timedelta

T0 = datetime(2026, 8, 16, tzinfo=UTC)


def create_series_with_points(client, unique_name, n_points=400):
    import math

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


def test_anomaly_scoring_creates_alerts_and_ack_works(client, champion_models, unique_name):
    series_id, _ = create_series_with_points(client, unique_name)
    resp = client.post(f"/api/v1/series/{series_id}/anomalies/score", json={"limit": 400})
    assert resp.status_code == 200

    alerts = client.get(
        "/api/v1/alerts", params={"series_id": series_id, "limit": 1000}
    ).json()
    anomaly_alerts = [a for a in alerts["alerts"] if a["kind"] == "anomaly"]
    scored_flagged = sum(1 for r in resp.json()["results"] if r["is_anomaly"])
    assert len(anomaly_alerts) == scored_flagged
    assert all(a["acknowledged"] is False for a in anomaly_alerts)

    # re-scoring the same window is deduped: no new alerts
    client.post(f"/api/v1/series/{series_id}/anomalies/score", json={"limit": 400})
    alerts_again = client.get(
        "/api/v1/alerts", params={"series_id": series_id, "limit": 1000}
    ).json()
    assert len([a for a in alerts_again["alerts"] if a["kind"] == "anomaly"]) == len(anomaly_alerts)

    # acknowledge the first alert
    target = anomaly_alerts[0]["id"]
    ack = client.post(f"/api/v1/alerts/{target}/ack")
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] is True
    acked_list = client.get(
        "/api/v1/alerts", params={"series_id": series_id, "acknowledged": True}
    ).json()
    assert [a["id"] for a in acked_list["alerts"]] == [target]


def test_ack_unknown_alert_404(client):
    resp = client.post("/api/v1/alerts/999999/ack")
    assert resp.status_code == 404
    assert resp.json()["code"] == "alert_not_found"


def test_drift_alert_created_on_psi_drift(client, champion_models, unique_name):
    import pandas as pd

    from ml.drift import feature_reference
    from ml.features import FeatureConfig, build_features, feature_names

    series_id, points = create_series_with_points(client, unique_name)
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

    # shift the recent values far away from the reference → PSI drift → alert
    shifted = [
        {"ts": p["ts"], "value": p["value"] + 200.0} for p in points[-200:]
    ]
    resp = client.post(f"/api/v1/series/{series_id}/points", json={"points": shifted})
    assert resp.status_code == 201

    check = client.post(
        "/api/v1/drift/check",
        json={"series_id": series_id, "task": "forecast", "window_points": 200},
    )
    assert check.status_code == 200
    alerts = client.get("/api/v1/alerts", params={"series_id": series_id, "kind": "drift"}).json()
    assert alerts["count"] >= 1
    drift_alert = alerts["alerts"][0]
    assert drift_alert["severity"] == "critical"
    assert "PSI drift" in drift_alert["message"]


def test_forecast_breach_alert(client, champion_models, unique_name):
    """An actual outside the stored band raises a forecast_breach alert on drift check."""
    series_id, points = create_series_with_points(client, unique_name)
    # store a forecast whose band is far from the actual at a target INSIDE the series
    # (the actual at T0+200min is 50±8, far outside the seeded [10, 14] band)
    target_ts = T0 + timedelta(minutes=200)

    # the drift check scans StoredForecast rows; seed one with a custom band directly
    # through the session factory the app uses
    from backend.app.models.prediction import StoredForecast
    from backend.app.services.inference import model_manager

    champion = model_manager.get("forecast")
    row = StoredForecast(
        series_id=series_id,
        horizon_minutes=5,
        target_ts=target_ts,
        quantiles={"0.05": 10.0, "0.5": 12.0, "0.95": 14.0},
        model_name=champion.name,
        model_version=champion.version,
    )
    from backend.app.main import app

    db = app.state.session_factory()  # the app's (test-overridden) session factory
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    finally:
        db.close()

    # the actual at target_ts is 50.0±8 — far outside [10, 14]
    check = client.post(
        "/api/v1/drift/check",
        json={"series_id": series_id, "task": "forecast", "window_points": 200},
    )
    assert check.status_code == 200
    breaches = [c for c in check.json()["checks"] if c["kind"] == "forecast_breach"]
    assert breaches and breaches[0]["breaches"] >= 1

    alerts = client.get(
        "/api/v1/alerts", params={"series_id": series_id, "kind": "forecast_breach"}
    ).json()
    assert alerts["count"] >= 1
    assert alerts["alerts"][0]["payload"]["target_ts"].startswith("2026-08-16T03:20")
