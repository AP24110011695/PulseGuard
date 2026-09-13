"""HTTP client used by ML-side CLIs to push generated/loaded data into the PulseGuard API."""

from __future__ import annotations

import time

import httpx
import pandas as pd

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def ensure_series(
    api_url: str,
    *,
    name: str,
    unit: str | None = None,
    description: str | None = None,
    source: str = "simulator",
    tags: dict[str, str] | None = None,
) -> int:
    """Return the id of the named series, creating it if it does not exist yet."""
    with httpx.Client(base_url=api_url, timeout=120.0) as client:
        existing = client.get("/api/v1/series", params={"name": name})
        existing.raise_for_status()
        found = existing.json()
        if found:
            return int(found[0]["id"])
        created = client.post(
            "/api/v1/series",
            json={
                "name": name,
                "unit": unit,
                "description": description,
                "source": source,
                "tags": tags,
            },
        )
        created.raise_for_status()
        return int(created.json()["id"])


def post_points(
    api_url: str, series_id: int, records: list[dict], chunk_size: int = 10_000
) -> dict:
    inserted = 0
    duplicates = 0
    elapsed = 0.0
    with httpx.Client(base_url=api_url, timeout=300.0) as client:
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            t0 = time.perf_counter()
            resp = client.post(f"/api/v1/series/{series_id}/points", json={"points": chunk})
            resp.raise_for_status()
            elapsed += time.perf_counter() - t0
            body = resp.json()
            inserted += body["inserted"]
            duplicates += body["duplicates"]
    return {
        "series_id": series_id,
        "received": len(records),
        "inserted": inserted,
        "duplicates": duplicates,
        "elapsed_seconds": round(elapsed, 3),
    }


def ingest_dataframe(
    api_url: str,
    df: pd.DataFrame,
    series_meta: dict[str, dict] | None = None,
    chunk_size: int = 10_000,
) -> dict:
    """Ingest a standardized frame (series_name, ts, value, is_anomaly) series by series.

    Re-ingesting the same data is a no-op thanks to the API's idempotent upsert.
    """
    series_meta = series_meta or {}
    totals: dict = {"series": {}, "received": 0, "inserted": 0, "duplicates": 0}
    for name, group in df.groupby("series_name", sort=True):
        name = str(name)
        meta = series_meta.get(name, {})
        series_id = ensure_series(
            api_url,
            name=name,
            unit=meta.get("unit"),
            description=meta.get("description"),
            source=meta.get("source", "simulator"),
        )
        records = [
            {"ts": ts.strftime(_TS_FORMAT), "value": float(v), "source_label": bool(a)}
            for ts, v, a in zip(group["ts"], group["value"], group["is_anomaly"], strict=True)
        ]
        result = post_points(api_url, series_id, records, chunk_size=chunk_size)
        totals["series"][name] = result
        totals["received"] += result["received"]
        totals["inserted"] += result["inserted"]
        totals["duplicates"] += result["duplicates"]
    return totals
