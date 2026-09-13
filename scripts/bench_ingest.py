"""Bulk ingestion benchmark (Phase 5).

    python scripts/bench_ingest.py [--api-url http://127.0.0.1:8000] [--total 20000] [--chunk 5000]

Creates a dedicated benchmark series, ingests `total` points in `chunk`-sized bulk
requests, and reports per-request latency and overall throughput. Results are printed;
record them in docs/BENCHMARKS.md manually (values are machine-dependent).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--total", type=int, default=20_000)
    parser.add_argument("--chunk", type=int, default=5_000)
    args = parser.parse_args()

    series_name = f"bench_ingest_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    with httpx.Client(base_url=args.api_url, timeout=300.0) as client:
        resp = client.post(
            "/api/v1/series", json={"name": series_name, "unit": "pt", "source": "benchmark"}
        )
        resp.raise_for_status()
        series_id = resp.json()["id"]

        start = datetime(2026, 6, 1, tzinfo=UTC)
        latencies: list[float] = []
        t0 = time.perf_counter()
        sent = 0
        for offset in range(0, args.total, args.chunk):
            n = min(args.chunk, args.total - offset)
            points = [
                {
                    "ts": (start + timedelta(minutes=offset + i)).isoformat(),
                    "value": 50.0 + 10.0 * math.sin((offset + i) / 240.0),
                }
                for i in range(n)
            ]
            t_req = time.perf_counter()
            resp = client.post(f"/api/v1/series/{series_id}/points", json={"points": points})
            resp.raise_for_status()
            latencies.append(time.perf_counter() - t_req)
            sent += n
        elapsed = time.perf_counter() - t0

    def pct(values: list[float], p: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    result = {
        "series_id": series_id,
        "points_total": sent,
        "chunk_size": args.chunk,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_points_per_s": round(sent / elapsed, 1),
        "request_latency_s": {
            "p50": round(pct(latencies, 0.5), 3),
            "p95": round(pct(latencies, 0.95), 3),
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
