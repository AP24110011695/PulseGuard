"""Points-query benchmark (Phase 5).

    python scripts/bench_query.py [--api-url ...] [--series-id N] [--runs 20]

Queries the largest available series (or a given one) with max_points=2000 over the
full stored range and reports p50/p95 latencies over `runs` warm requests.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--series-id", type=int, default=None)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    with httpx.Client(base_url=args.api_url, timeout=120.0) as client:
        series = client.get("/api/v1/series").json()
        if not series:
            print(json.dumps({"error": "no series available; ingest data first"}))
            return 1
        if args.series_id is not None:
            target = next((s for s in series if s["id"] == args.series_id), None)
        else:
            target = max(series, key=lambda s: s["id"])  # benchmark series land last
        series_id, name = target["id"], target["name"]

        latencies: list[float] = []
        counts: list[int] = []
        detail = None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            resp = client.get(f"/api/v1/series/{series_id}/points", params={"max_points": 2000})
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            body = resp.json()
            latencies.append(elapsed)
            counts.append(body["count"])
            detail = {"downsampled": body["downsampled"], "bucket_seconds": body["bucket_seconds"]}

    result = {
        "series_id": series_id,
        "series_name": name,
        "runs": args.runs,
        "max_points": 2000,
        "latency_s": {
            "p50": round(statistics.median(latencies), 4),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
            "mean": round(statistics.mean(latencies), 4),
        },
        "returned_rows": counts[0],
        "query": detail,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
