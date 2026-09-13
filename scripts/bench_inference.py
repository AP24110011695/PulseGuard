"""Inference benchmark (Phase 5).

    python scripts/bench_inference.py [--api-url ...] [--series-id N] [--runs 20]

Measures warm latency (p50/p95) of the forecast and anomaly-scoring endpoints against
the current champions. Cold model loading happens only on champion changes and is
reported separately by the API logs, not here. Requires registered champions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx


def measure(client: httpx.Client, method: str, url: str, json_body=None, runs: int = 20) -> dict:
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        resp = client.request(method, url, json=json_body)
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        latencies.append(elapsed)
    return {
        "runs": runs,
        "latency_s": {
            "p50": round(statistics.median(latencies), 4),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
            "mean": round(statistics.mean(latencies), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--series-id", type=int, default=None)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    with httpx.Client(base_url=args.api_url, timeout=120.0) as client:
        series = client.get("/api/v1/series").json()
        target = next(
            (s for s in series if s["id"] == args.series_id),
            max(series, key=lambda s: s["id"]) if series else None,
        )
        if target is None:
            print(json.dumps({"error": "no series available"}))
            return 1
        series_id = target["id"]

        result: dict = {"series_id": series_id, "series_name": target["name"]}
        result["forecast_h15"] = measure(
            client, "GET", f"/api/v1/series/{series_id}/forecast?horizon_minutes=15", runs=args.runs
        )
        result["forecast_h1"] = measure(
            client, "GET", f"/api/v1/series/{series_id}/forecast?horizon_minutes=1", runs=args.runs
        )
        result["anomaly_score_400"] = measure(
            client,
            "POST",
            f"/api/v1/series/{series_id}/anomalies/score",
            json_body={"limit": 400},
            runs=max(10, args.runs // 2),
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
