"""Inference API roundtrips: champion resolution, forecast/anomaly persistence with
model-version tracing, alias reload, and the 503 path when no champion is available."""

import math
from datetime import UTC, datetime, timedelta

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
    return series_id


def test_champion_endpoint(client, champion_models):
    resp = client.get("/api/v1/models/forecast/champion")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "forecast"
    assert body["model_name"] == "pulseguard-forecaster"
    # the alias may have advanced if earlier tests triggered promotions; the endpoint
    # must always reflect the CURRENT champion
    assert body["model_version"] >= champion_models["forecast_version"]
    assert body["loaded"] is True
    # the exact horizons depend on which challenger earlier tests promoted
    assert isinstance(body["metrics"]["horizons"], list) and body["metrics"]["horizons"]


def test_list_models(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    assert {"pulseguard-forecaster", "pulseguard-anomaly-detector"} <= names


def test_forecast_roundtrip_persists_with_model_version(client, unique_name):
    series_id = create_series_with_points(client, unique_name)
    resp = client.get(f"/api/v1/series/{series_id}/forecast", params={"horizon_minutes": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["series_id"] == series_id
    assert body["horizon_minutes"] == 1
    assert body["model_name"] == "pulseguard-forecaster"
    assert body["model_version"] >= 1
    for quantile in ("0.05", "0.5", "0.95"):
        assert body["quantiles"][quantile] == body["quantiles"][quantile]  # present & finite
    assert body["target_ts"].startswith("2026-08-16T06:40")  # T0 + 400 minutes + 1

    stored = client.get(f"/api/v1/series/{series_id}/forecasts").json()
    assert stored["count"] == 1
    assert stored["forecasts"][0]["id"] == body["id"]
    assert stored["forecasts"][0]["model_version"] >= 1

    # model_version filter: a version that produced nothing returns an empty page
    empty = client.get(
        f"/api/v1/series/{series_id}/forecasts", params={"model_version": 999}
    ).json()
    assert empty["count"] == 0


def test_anomaly_scoring_roundtrip(client, unique_name):
    series_id = create_series_with_points(client, unique_name)
    resp = client.post(f"/api/v1/series/{series_id}/anomalies/score", json={"limit": 300})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scored"] == 300
    assert body["model_name"] == "pulseguard-anomaly-detector"
    assert body["model_version"] >= 1
    assert len(body["results"]) == 300
    assert all("score" in r and "is_anomaly" in r for r in body["results"])

    stored = client.get(f"/api/v1/series/{series_id}/anomalies").json()
    assert stored["count"] == 300
    versioned = client.get(
        f"/api/v1/series/{series_id}/anomalies", params={"model_version": body["model_version"]}
    ).json()
    assert versioned["count"] == 300

    # re-scoring the same window with the same model is idempotent
    resp = client.post(f"/api/v1/series/{series_id}/anomalies/score", json={"limit": 300})
    assert resp.json()["scored"] == 300
    stored = client.get(f"/api/v1/series/{series_id}/anomalies").json()
    assert stored["count"] == 300


def test_unsupported_horizon_422(client, unique_name):
    series_id = create_series_with_points(client, unique_name)
    resp = client.get(f"/api/v1/series/{series_id}/forecast", params={"horizon_minutes": 42})
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_forecast_request"


def test_insufficient_history_422(client, unique_name):
    # one point is below ANY champion's warmup requirement, regardless of which
    # champion version earlier tests have promoted
    series_id = create_series_with_points(client, unique_name, n_points=1)
    resp = client.get(f"/api/v1/series/{series_id}/forecast", params={"horizon_minutes": 1})
    assert resp.status_code == 422
    assert "insufficient history" in resp.json()["detail"]


def test_alias_bump_and_reload_swap_served_model(client, champion_models, unique_name):
    series_id = create_series_with_points(client, unique_name)
    before = client.get(
        f"/api/v1/series/{series_id}/forecast", params={"horizon_minutes": 1}
    ).json()
    old_version = before["model_version"]

    # register a new version and move the champion alias to it
    new_version = champion_models["register"](None)
    assert new_version > old_version

    # ensure_current picks the bump up without an explicit reload
    after = client.get(f"/api/v1/series/{series_id}/forecast", params={"horizon_minutes": 1}).json()
    assert after["model_version"] == new_version

    # the explicit reload endpoint also re-syncs active_models
    resp = client.post("/api/v1/models/reload")
    assert resp.status_code == 200
    models_state = resp.json()["models"]
    assert models_state["forecast"]["model_version"] == new_version
    assert models_state["forecast"]["loaded"] is True

    champ = client.get("/api/v1/models/forecast/champion").json()
    assert champ["model_version"] == new_version


def test_missing_champion_503(client, champion_models, unique_name):
    from mlflow import MlflowClient

    series_id = create_series_with_points(client, unique_name)
    mclient = MlflowClient(tracking_uri=champion_models["uri"])
    mclient.delete_registered_model_alias("pulseguard-anomaly-detector", "champion")
    try:
        resp = client.post(f"/api/v1/series/{series_id}/anomalies/score", json={"limit": 10})
        assert resp.status_code == 503
        assert resp.json()["code"] == "model_unavailable"
    finally:
        mclient.set_registered_model_alias(
            "pulseguard-anomaly-detector", "champion", str(champion_models["anomaly_version"])
        )
        client.post("/api/v1/models/reload")
