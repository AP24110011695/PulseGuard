from datetime import UTC, datetime, timedelta

T0 = datetime(2026, 8, 16, tzinfo=UTC)


def make_points(n: int, start: datetime = T0, label: bool | None = False):
    return [
        {
            "ts": (start + timedelta(minutes=i)).isoformat(),
            "value": 10.0 + i * 0.5,
            "source_label": label,
        }
        for i in range(n)
    ]


def create_series(client, name: str) -> int:
    resp = client.post("/api/v1/series", json={"name": name, "source": "simulator"})
    assert resp.status_code == 201
    return resp.json()["id"]


def test_ingest_roundtrip_and_aggregates(client, unique_name):
    series_id = create_series(client, unique_name)
    resp = client.post(f"/api/v1/series/{series_id}/points", json={"points": make_points(120)})
    assert resp.status_code == 201
    body = resp.json()
    assert body == {
        "series_id": series_id,
        "received": 120,
        "inserted": 120,
        "duplicates": 0,
    }

    detail = client.get(f"/api/v1/series/{series_id}").json()
    assert detail["point_count"] == 120
    assert detail["first_ts"].startswith("2026-08-16T00:00:00")
    assert detail["last_ts"].startswith("2026-08-16T01:59:00")


def test_ingest_is_idempotent(client, unique_name):
    series_id = create_series(client, unique_name)
    payload = {"points": make_points(60)}
    assert client.post(f"/api/v1/series/{series_id}/points", json=payload).json()["inserted"] == 60
    resp = client.post(f"/api/v1/series/{series_id}/points", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["inserted"] == 0
    assert body["duplicates"] == 60
    assert client.get(f"/api/v1/series/{series_id}").json()["point_count"] == 60


def test_query_all_points_ordered(client, unique_name):
    series_id = create_series(client, unique_name)
    client.post(f"/api/v1/series/{series_id}/points", json={"points": make_points(100)})
    resp = client.get(f"/api/v1/series/{series_id}/points")
    assert resp.status_code == 200
    body = resp.json()
    assert body["downsampled"] is False
    assert body["count"] == 100
    ts_list = [p["ts"] for p in body["points"]]
    assert ts_list == sorted(ts_list)
    assert body["points"][0]["value"] == 10.0


def test_query_window_is_half_open(client, unique_name):
    series_id = create_series(client, unique_name)
    client.post(f"/api/v1/series/{series_id}/points", json={"points": make_points(120)})
    start = (T0 + timedelta(minutes=60)).isoformat()
    end = (T0 + timedelta(minutes=90)).isoformat()
    body = client.get(
        f"/api/v1/series/{series_id}/points", params={"start": start, "end": end}
    ).json()
    assert body["count"] == 30
    assert body["downsampled"] is False
    assert body["points"][0]["value"] == 10.0 + 60 * 0.5


def test_query_downsamples_to_max_points(client, unique_name):
    series_id = create_series(client, unique_name)
    client.post(f"/api/v1/series/{series_id}/points", json={"points": make_points(1200)})
    body = client.get(f"/api/v1/series/{series_id}/points", params={"max_points": 50}).json()
    assert body["downsampled"] is True
    assert body["bucket_seconds"] is not None
    assert 1 <= body["count"] <= 50
    ts_list = [p["ts"] for p in body["points"]]
    assert ts_list == sorted(ts_list)


def test_ingest_rejects_non_finite_value(client, unique_name):
    series_id = create_series(client, unique_name)
    # httpx refuses to serialize Infinity into JSON, so send the raw payload.
    raw = f'{{"points": [{{"ts": "{T0.isoformat()}", "value": Infinity}}]}}'
    resp = client.post(
        f"/api/v1/series/{series_id}/points",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_points_endpoints_404_for_unknown_series(client):
    resp = client.post("/api/v1/series/999999/points", json={"points": make_points(1)})
    assert resp.status_code == 404
    assert resp.json()["code"] == "series_not_found"
    resp = client.get("/api/v1/series/999999/points")
    assert resp.status_code == 404
