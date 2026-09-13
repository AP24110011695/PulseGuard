from backend.app.schemas.metric import SeriesCreate


def test_create_series(client, unique_name):
    resp = client.post(
        "/api/v1/series",
        json={"name": unique_name, "unit": "ms", "source": "simulator", "tags": {"env": "test"}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == unique_name
    assert body["unit"] == "ms"
    assert body["source"] == "simulator"
    assert body["tags"] == {"env": "test"}
    assert body["id"] > 0


def test_create_duplicate_name_conflict(client, unique_name):
    payload = {"name": unique_name}
    assert client.post("/api/v1/series", json=payload).status_code == 201
    resp = client.post("/api/v1/series", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "series_exists"


def test_list_and_filter_by_name(client, unique_name):
    client.post("/api/v1/series", json={"name": unique_name, "source": "nab"})
    listed = client.get("/api/v1/series", params={"name": unique_name})
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["source"] == "nab"


def test_series_detail_empty_aggregates(client, unique_name):
    created = client.post("/api/v1/series", json={"name": unique_name}).json()
    detail = client.get(f"/api/v1/series/{created['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["point_count"] == 0
    assert body["first_ts"] is None
    assert body["last_ts"] is None


def test_unknown_series_404_shape(client):
    resp = client.get("/api/v1/series/999999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "series_not_found"
    assert "999999" in body["detail"]


def test_series_create_validation(client):
    resp = client.post("/api/v1/series", json={"name": ""})
    assert resp.status_code == 422
    # Schema-level check: defaults applied cleanly for valid payloads.
    payload = SeriesCreate(name="whatever").model_dump()
    assert payload["source"] == "api"
