# PulseGuard Benchmarks (Phase 5)

All numbers below are **measured on the reference development machine** and are
reproducible with the benchmark scripts in `scripts/`. They describe a single-node
local Docker deployment — they are **not** production performance claims.

- **Date:** 2026-09-13
- **Host:** Windows 11, Docker Desktop (WSL2 backend), all services in containers
  (`db` postgres:16-alpine, `api` python:3.12-slim + uvicorn, 1 uvicorn worker)
- **Client:** Python 3.12.10 + httpx on the host, warm client, loopback
- **Data:** the seeded synthetic dataset (3 series × 28 days @ 1 min = 40,320 points
  each; level shifts at days 1.1 / 15.7 / 16.3) unless noted

Reproduce:

```
python scripts/bench_ingest.py                 # 20,000 fresh points in 5k chunks
python scripts/bench_query.py --series-id 4 --runs 20
python scripts/bench_inference.py --series-id 4 --runs 20
```

## Bulk ingestion (`bench_ingest.py`)

20,000 new points posted in 5,000-point bulk requests (4 requests):

| metric | value |
|---|---|
| total elapsed | 7.01 s |
| throughput | **2,854 points/s** |
| request latency p50 / p95 | 1.76 s / 2.11 s per 5k-point request |

Ingestion is an idempotent upsert (`INSERT … ON CONFLICT DO NOTHING`) — re-running the
benchmark with the same series name adds 0 duplicates.

## Points query (`bench_query.py`, series 4 — 40,320 points)

Downsampled read (max_points=2000, bucketed via `date_bin`), 20 warm requests:

| metric | value |
|---|---|
| latency p50 | **44 ms** |
| latency p95 | 1.95 s (see note) |
| latency mean | 157 ms |
| rows returned | 1,998 |

Note: the p95 is dominated by one cold outlier (first request after a container
restart pays for the new DB connection pool + the 40k-row aggregate scan); the p50 and
mean describe steady state. The same benchmark on a fresh 20k-point series shows
p50 = 41 ms / p95 = 181 ms / mean = 52 ms.

## Inference (`bench_inference.py`, series 4, warm champions)

| endpoint | runs | p50 | p95 | mean |
|---|---|---|---|---|
| `GET /series/4/forecast?horizon_minutes=1` | 20 | **54 ms** | 82 ms | 55 ms |
| `GET /series/4/forecast?horizon_minutes=15` | 20 | **55 ms** | 1.11 s (see note) | 111 ms |
| `POST /series/4/anomalies/score` (400 rows) | 10 | **199 ms** | 394 ms | 252 ms |

Notes:
- Warm latency includes pulling the point history from PostgreSQL, building the
  feature row with the shared `ml.features` code, and persisting the prediction —
  it is an end-to-end serving measurement, not model-only.
- The p95 outliers (~1 s) coincide with occasional slow DB round-trips under local
  Docker Desktop; the distribution is otherwise tight around the p50.
- Cold model loading (champion alias change → `mlflow.pyfunc.load_model`) happens
  outside these numbers and takes on the order of seconds; the API returns 503 for a
  task whose champion is not yet loaded rather than blocking.

## What this does and does not demonstrate

It demonstrates that the implemented serving path meets local interactive-latency
expectations (sub-100 ms warm forecasts, ~0.2 s for a 400-point anomaly scoring pass,
~2.9k pts/s idempotent ingestion) on a laptop-class single-node deployment. It does
**not** measure horizontal scalability, multi-user concurrency, or production
infrastructure behavior — no such claims are made.
