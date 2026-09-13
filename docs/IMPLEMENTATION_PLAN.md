# PulseGuard — Implementation Plan

> **Companion to [`PULSEGUARD.md`](../PULSEGUARD.md).** That file defines what PulseGuard
> is and the rules; this file defines exactly how it gets built, phase by phase.
>
> - **Status:** Phase 5 complete (2026-09-13). Phase 6 — Final Demo, Docker & Documentation — is next.
> - **Last updated:** 2026-09-13

---

## 0. How to use this plan (agent workflow rules)

1. Read `PULSEGUARD.md` first in every session. It overrides habits; this plan overrides
   improvisation.
2. Execute phases **strictly in order: 1 → 6**. Within a phase, finish its completion
   criteria before moving on. Do not implement ahead of the current phase.
3. If reality contradicts the plan (a decision turns out wrong), **update this plan first
   with a short rationale**, then implement. Do not silently drift from the documented
   architecture.
4. Never invent measured values. Numbers appear in docs only after the producing command
   has been run, with the command recorded next to the number.
5. Each phase lists **verification commands** — run them and show real output before
   claiming a phase is done. Then tick the phase's completion checkboxes in §19.
6. Keep changes small and reviewable; no giant refactors without a documented reason.

---

## 1. Overall architecture

```
                        ┌────────────────────────────────────────────────┐
                        │                  docker compose                │
                        │                                                │
  ml/simulator.py ──────┼──▶ PostgreSQL 16 ◀──── FastAPI (backend.app)   │
  ml/loaders.py (NAB)   │      (metrics,        │        │               │
                        │       predictions,    │        │  ▲            │
  ml/train.py ──────────┼──▶ MLflow 2.x server  │◀───────┘  │            │
  (tracking, registry)  │      (runs, registry, │  champion alias resolve   │
                        │       artifacts)      │                           │
  ml/retrain.py ────────┼──▶ promotion gate ────┤ (alias move on promote)   │
  ml/drift.py (PSI)     │                       │                           │
                        │  React + Vite (frontend) ──▶ GET /api/v1/*       │
                        └────────────────────────────────────────────────┘
```

- **One codebase, one Python environment, one backend Docker image** containing both the
  `backend.app` (serving) and `ml` (offline training/tooling) packages. Training jobs run
  locally (`python -m ml.train`) or via `docker compose run --rm api python -m ml.train ...`.
- **PostgreSQL is the single operational datastore**: raw metrics, stored predictions,
  model-assignment metadata, drift events, promotion decisions, alerts. MLflow uses the
  same Postgres instance (separate `mlflow` database) for runs/registry, with artifacts on
  a shared local volume.
- **No streaming bus, no scheduler daemon.** Ingestion is request-driven (bulk POST);
  training, drift checks, and retraining are explicit CLI/script actions. This keeps the
  lifecycle honest and explainable.
- Data path: simulator/loader → ingestion API → Postgres → feature builder → models →
  MLflow registry → inference API (champion by alias) → persisted predictions → dashboard.

## 2. Repository structure

See `PULSEGUARD.md §8` for the annotated target tree. Current bootstrap state: directory
skeleton + `PULSEGUARD.md` + `docs/IMPLEMENTATION_PLAN.md` + `.gitignore` +
`.env.example` + README stub exist; **all code, configs, and Dockerfiles are created in
their phases** (file lists in §19).

Key structural decisions:

- `backend/` and `ml/` are both packages of the **single root `pyproject.toml`**
  (`[tool.setuptools.packages.find] include = ["backend*", "ml*"]`), installed with
  `pip install -e .`. Uvicorn target: `backend.app.main:app`. Training: `python -m ml.train`.
- Tests live next to their package: `backend/tests/` (API/integration, needs the compose
  Postgres) and `ml/tests/` (pure unit + leakage + small integration tests).
- `ml/configs/` holds all YAML configuration (simulator profiles, training/backtest setup,
  promotion gates, demo scenarios). Behavior is config-driven; magic numbers in code are
  a defect.
- `data/` and `ml/reports/` are gitignored output locations (downloaded datasets,
  generated series, evaluation JSON). Measured numbers get transcribed into docs by hand,
  with the producing command.

## 3. Database & data-model strategy

Postgres 16, `timestamptz` (UTC) everywhere, integer/bigint surrogate keys, one Alembic
migration per phase that changes schema. Data volume target: ~3–5 series at 1-minute
frequency over ~28 days ≈ 120k–200k points — trivially within Postgres' comfort zone; no
time-series extension needed (TimescaleDB = postponed stretch).

| Table | Phase | Key columns | Notes |
|---|---|---|---|
| `metric_series` | P1 | id, name (unique), unit, description, source (`simulator`\|`nab`\|`api`), tags jsonb, created_at | one row per metric stream |
| `metric_points` | P1 | id, series_id FK, ts timestamptz, value double precision, source_label bool null | `UNIQUE(series_id, ts)`; index `(series_id, ts DESC)`; `source_label` carries ground truth from simulator/public data, kept separate from model output |
| `anomaly_results` | P3 | id, series_id, ts, score real, is_anomaly bool, threshold real, model_name, model_version, created_at | model output only; index `(series_id, ts)` |
| `forecasts` | P3 | id, series_id, created_at, horizon_minutes, target_ts, quantiles jsonb (`{"0.05":…,"0.5":…,"0.95":…}`), model_name, model_version | jsonb keeps quantile set flexible; index `(series_id, target_ts)` |
| `active_models` | P3 | id, task (`anomaly`\|`forecast`), alias (`champion`), mlflow_model_name, model_version, metrics jsonb, promoted_at | one row per (task, alias); mirrors registry for API traceability |
| `drift_references` | P4 | id, series_id, model_name, model_version, kind (`psi`\|`residual`), payload jsonb (bin edges/frequencies or residual baseline), created_at | reference distributions captured at champion training |
| `drift_events` | P4 | id, series_id, detected_at, kind (`psi`\|`residual`), feature, psi_value, threshold, status (`warn`\|`drift`), details jsonb | append-only event log |
| `promotion_decisions` | P4 | id, decided_at, task, champion_name/version, challenger_name/version, decision (`promoted`\|`rejected`), reason, metrics jsonb | full audit trail incl. rule trace |
| `alerts` | P5 | id, created_at, series_id, kind (`anomaly`\|`forecast_breach`\|`drift`), severity, message, payload jsonb, acknowledged bool | derived feed for the dashboard |

Conventions: ingestion is idempotent upsert (`INSERT … ON CONFLICT (series_id, ts) DO
NOTHING`), chunked (~5k rows/statement, ≤50k points per request); queries always bound by
`series_id` + ts range; chart reads use bucketed downsampling (`date_bin`) with a
`max_points` cap. Retention policies: not implemented (optional stretch).

## 4. API strategy

- FastAPI, all routes under **`/api/v1`**, Pydantic request/response models, OpenAPI docs
  at `/docs`. JSON only; timestamps are UTC ISO-8601.
- Errors: consistent shape `{"detail": <human message>, "code": <machine code>}` with
  proper status codes (404 unknown series, 422 validation, 503 when a champion model is
  required but not loaded).
- **No authentication/multi-tenancy** — deliberate scope decision for a local demo tool;
  documented in README limitations.
- Bulk limits: ≤50,000 points per ingest request; queries capped by `max_points`
  (default 2,000) with bucket-averaged downsampling for chart ranges.
- CORS enabled for the frontend origin only.

Endpoint map (added per phase):

| Method | Path | Phase |
|---|---|---|
| GET | `/api/v1/health` | P1 |
| POST/GET | `/api/v1/series`, GET `/api/v1/series/{id}` | P1 |
| POST | `/api/v1/series/{id}/points` (bulk ingest) | P1 |
| GET | `/api/v1/series/{id}/points?start&end&max_points` | P1 |
| GET | `/api/v1/models`, `/api/v1/models/{task}/champion` | P3 |
| POST | `/api/v1/models/reload` (re-resolve aliases) | P3 |
| GET | `/api/v1/series/{id}/forecast?horizon_minutes=` (compute now + persist) | P3 |
| GET | `/api/v1/series/{id}/forecasts?start&end&model_version` (stored) | P3 |
| POST | `/api/v1/series/{id}/anomalies/score` (score window + persist) | P3 |
| GET | `/api/v1/series/{id}/anomalies?start&end&model_version` (stored) | P3 |
| GET | `/api/v1/drift/status`, `/api/v1/drift/events`, POST `/api/v1/drift/check` | P4 |
| GET | `/api/v1/promotions` | P4 |
| POST | `/api/v1/retraining/trigger` (optional; BackgroundTasks) | P4 |
| GET | `/api/v1/alerts`, POST `/api/v1/alerts/{id}/ack` | P5 |

## 5. Frontend strategy

- Vite + React 18 + TypeScript, Recharts for charts. Plain fetch wrapper
  (`src/api/client.ts`, typed) + small data hooks; **no state library** — add TanStack
  Query only if state complexity genuinely demands it (justify in the plan first).
- Target structure (scaffolded P1, filled out P5):

```
frontend/src/
├── api/client.ts
├── components/  Chart.tsx, MetricsTable.tsx, StatusBadge.tsx,
│                EmptyState.tsx, ErrorState.tsx, LoadingSkeleton.tsx
├── pages/       Overview.tsx, SeriesDetail.tsx, Models.tsx, Drift.tsx, Alerts.tsx
├── hooks/       useSeries, usePoints, useForecast, ...
└── App.tsx, main.tsx
```

- Forecast bands rendered as a Recharts `Area` with `[low, high]` data keys; anomaly
  markers as scatter dots colored by score.
- Every view ships with loading / empty / error states and a responsive grid (P1 pages
  get minimal versions; P5 hardens them). **Mock data is forbidden** — the UI renders only
  what the API returns.

## 6. ML pipeline

Files and responsibilities (details in phase sections):

| Stage | File (P2 unless noted) | Content |
|---|---|---|
| Data | `ml/simulator.py` (P1), `ml/loaders.py` (P1) | labeled synthetic streams; NAB loader → standard schema (`ts,value,source_label`) |
| Features | `ml/features.py` | single source of truth for train **and** serve |
| Splits | `ml/backtest.py` | golden split + walk-forward folds with embargo |
| Models | `ml/anomaly.py`, `ml/forecast.py` | IsolationForest + z-score baseline; LGBM quantile + naive baselines |
| Metrics | `ml/evaluate.py` | all metric math, one code path |
| Orchestration | `ml/train.py` | CLI: config → features → folds → metrics JSON report |
| Tracking/Registry | `ml/tracking.py`, `ml/models.py` (P3) | MLflow runs; pyfunc wrappers; alias management |
| Drift/Retrain | `ml/drift.py`, `ml/retrain.py` (P4) | PSI, residual monitor; challenger + gate |

**Feature spec (defaults, all config-tunable):** lags at t−{1,2,3,5,15,30,60} min;
rolling {5,15,60}-min mean/std/min/max (trailing); rate-of-change `Δ1`, `Δ60`, and
z-score vs the 60-min rolling mean/std; calendar features (hour-of-day sin/cos,
day-of-week). Warmup rows (`≥ max window`) dropped. Forecast targets `v_{t+h}`, h ∈
{1,5,15}; anomaly is unsupervised (labels used only for evaluation/threshold calibration).

**Golden set spec:** the final segment of the timeline (default: last 5 days of the
synthetic window; last 10% for public data), defined by config + seed + ts-range and
hashed. Never used for training, early stopping, or threshold calibration. Champion
golden metrics are stored at promotion time (`active_models.metrics` + MLflow artifacts);
challengers are evaluated on the identical set through the identical evaluation code path.

**Determinism:** fixed seeds end-to-end; config normalized → SHA-256 recorded in every
report and MLflow run tag, so any reported number can be regenerated.

## 7. Experiment tracking strategy (MLflow, P3)

- Compose service `mlflow`: `mlflow server --backend-store-uri postgresql://…/mlflow
  --default-artifact-root /mlartifacts` (shared volume), tracking URI via env.
- One MLflow run per training execution. Tags: task, model type, config SHA-256, dataset
  spec. Params: full model + feature + split config. Metrics: per-fold and aggregate.
  Artifacts: metrics JSON, feature list, model (pyfunc), and (P4) drift references.
- Tracking helpers live in `ml/tracking.py`; tests use a local file-based tracking store
  fixture so unit tests don't need the server.

## 8. Model registry & lifecycle strategy

- Two registered models: `pulseguard-forecaster` and `pulseguard-anomaly-detector`.
- **Aliases, not stages:** `champion` and `challenger` aliases (MLflow 2.x client
  aliases). `active_models` mirrors the current champion per task for fast API reads and
  an in-database audit trail.
- P3: scripted/manual promotion after a good evaluation (`ml/train.py --register` →
  register version → set `champion` alias). P4 replaces the human judgment with the
  measured promotion gate (§11) and records every decision.
- Champion golden metrics are recomputed and stored at promotion; challenger comparisons
  always recompute both sides on the frozen golden set. Versions are never deleted.

## 9. Inference architecture (P3)

- `app/services/inference.py` model manager: on startup resolves the `champion` alias per
  task via the MLflow client, loads the pyfunc model, caches it. A cheap alias-version
  check runs per request; `POST /api/v1/models/reload` forces re-resolution.
- Forecast flow: pull last `lookback` points for the series from Postgres → build features
  with **`ml.features` (same code as training)** → predict quantiles for requested
  horizons → persist to `forecasts` → return band.
- Anomaly flow: score a window of recent points → persist to `anomaly_results` (score,
  threshold, label, model version).
- Every persisted prediction row carries `model_name` + `model_version` → full
  provenance; stored-result endpoints accept a `model_version` filter.
- If no champion is registered, model-requiring endpoints return **503** with a clear
  message (never a fabricated prediction).

## 10. Drift strategy (P4)

- **PSI:** reference = champion's training-time feature distributions (10 quantile bins,
  edges + frequencies stored in `drift_references` and as a model artifact); current =
  recent window. `PSI = Σ (a%−e%)·ln(a%/e%)` with ε-smoothing for empty bins.
  Thresholds: `<0.1` stable, `0.1–0.2` warn, `>0.2` drift.
- **Residual monitor:** rolling mean absolute error of the stored median forecasts vs
  actuals, compared to the golden-set residual baseline; drift if `> 1.5× baseline` for
  ≥3 consecutive windows (config).
- Checks run on demand (`python -m ml.drift`, `POST /api/v1/drift/check`) — **no daemon**.
  Outcomes append to `drift_events`; `GET /api/v1/drift/status` summarizes latest state
  per series/task.

## 11. Retraining & promotion strategy (P4)

Triggered, controlled flow (CLI `python -m ml.retrain`, demo script, optional API
trigger): challenger trained on a window that includes post-drift data → **both** champion
and challenger evaluated on the frozen golden set → gate → promote or reject.

Promotion gate (defaults in config; final numbers tuned once P2 baselines exist):

| Criterion | Forecast model | Anomaly model |
|---|---|---|
| Primary | mean pinball improvement ≥ 2% relative | golden PR-AUC ≥ champion − 0.005 |
| Secondary | MAE regression ≤ 1% | F1 (calibrated threshold) ≥ champion − 0.02 |
| Sanity | empirical 90% band coverage within ±5pp of nominal | — |

- Promote: register challenger version, move `champion` alias, update `active_models`,
  append `promotion_decisions(promoted, metrics, rule trace)`.
- Reject: append `promotion_decisions(rejected, …)`; champion untouched. **Rejection is a
  first-class outcome** — the demo must show at least one scripted rejection (e.g.,
  challenger trained on insufficient/pre-drift data) alongside a promotion.
- Minimum data requirement before retraining is allowed (config, e.g. ≥ N post-drift
  points).

## 12. Testing strategy

- **Unit (ml/tests):** feature builder (values, warmup, trailing-only), leakage test
  (perturbing future rows must not change past features), pinball/PR metrics vs
  hand-computed & sklearn references, PSI on known distributions, fold ordering +
  embargo, simulator determinism/labels, promotion-gate branches (promote, reject,
  tolerance boundaries).
- **API/integration (backend/tests):** pytest + FastAPI TestClient against a dedicated
  `pulseguard_test` database on the compose Postgres (fixtures: create/drop schema per
  session). Ingest idempotency, query windows/downsampling, inference roundtrip (fixture
  model via a local-file MLflow store), persistence with model_version, drift/promotion
  endpoints.
- **E2E smoke:** one scripted pass per major capability (P2 evaluation run; P3
  train→register→serve; P4 drift demo) — small configs, minutes not hours.
- **Frontend:** `npm run build` + lint must pass; component tests (vitest) optional.
- **Benchmarks are scripts, not tests** (P5): `scripts/bench_ingest.py`,
  `bench_query.py`, `bench_inference.py` printing measured p50/p95/throughput.
- Never report coverage percentages or performance numbers that weren't actually measured.

## 13. Docker & environment strategy

| Service | Image | Port | Notes |
|---|---|---|---|
| `db` | postgres:16 | 5432 | volume `pgdata`; init script creates `pulseguard` + `mlflow` databases |
| `api` | build `backend/Dockerfile` (python:3.12-slim) | 8000 | entrypoint: `alembic upgrade head` then uvicorn; contains `backend` + `ml` packages so it can also run training jobs |
| `mlflow` | same `api` image, different command (P3) | 5000 | Postgres backend store, `mlartifacts` volume |
| `frontend` | node:20 dev (P1) / multi-stage build → static serve (P6) | host 5174 → container 5173 (see Phase log §P1-3) / 80 | `VITE_API_BASE_URL` wiring |

- Volumes: `pgdata`, `mlartifacts`. Env via `.env` (from `.env.example`); dev-only
  placeholder credentials, no secrets.
- Local dev (Windows + Git Bash): Python 3.12 venv + `pip install -e .`; Postgres/MLflow
  via compose; Node ≥ 20 for the frontend. Alembic config lives at
  `backend/alembic.ini`; run `alembic -c backend/alembic.ini upgrade head` from the repo
  root (or rely on the api entrypoint).

## 14. Documentation strategy

Maintained: `PULSEGUARD.md` (source of truth), `docs/IMPLEMENTATION_PLAN.md` (this file —
checkboxes updated as phases complete). Created later, only once their content exists:
`README.md` (full version, P6), `docs/EVALUATION_REPORT.md` and `docs/BENCHMARKS.md`
(P6/P5 respectively — measured numbers + repro commands only). Nothing else without a
demonstrated need; no doc sprawl.

## 15. Major technical decisions (log)

1. **Monorepo, single Python env:** `backend.app` + `ml` packages in one root
   `pyproject.toml`; one backend image serves the API and runs training jobs. Prevents
   train/serve skew and packaging gymnastics.
2. **Shared feature code:** `ml/features.py` is imported by both training and the
   inference service — the structural guarantee against train/serve skew.
3. **PostgreSQL for everything operational; no TSDB.** 1-min metrics at this volume fit
   comfortably; TimescaleDB postponed as stretch.
4. **Direct multi-horizon quantile forecasting:** one LightGBM model per (horizon,
   quantile), wrapped as a single MLflow pyfunc artifact returning the full band.
5. **Isolation Forest with explicit threshold calibration** on labeled synthetic
   validation (percentile fallback for unlabeled data); PR-AUC as the primary metric.
6. **Walk-forward backtesting with embargo gaps** and a frozen golden holdout as the only
   champion/challenger battleground. No random splits anywhere, enforced by tests.
7. **MLflow 2.x with Postgres backend + alias-based registry** (`champion`/`challenger`
   aliases, mirrored in `active_models`); artifact store on a shared volume.
8. **Inference loads the champion by alias** with lightweight alias-version reload checks;
   every prediction persisted with model name+version.
9. **Controlled, triggered retraining** (CLI/script) — no background retraining daemon;
   claims match behavior.
10. **Drift = PSI + residual monitoring** with references captured at champion training
    time and persisted for on-demand checks.
11. **No auth, no multi-tenancy, no streaming bus** — documented scope decisions for a
    local demo tool.
12. **Config over constants:** all sim/training/gate numbers live in YAML configs whose
    normalized SHA-256 tags every report and MLflow run.
13. **Alembic from day one** — schema evolves across four phases; migrations keep
    environments reproducible.
14. **NAB as the default public benchmark** (labeled, small, easily downloadable), Yahoo
    S5 as fallback.

## 16. Scope control

**Required (the six phases as specified):** scaffold + ingestion + simulator + minimal
dashboard; features + IF + quantile LGBM + walk-forward evaluation with baselines;
MLflow tracking/registry + inference service with persistence/tracing; PSI + residual
drift + challenger retraining + measured promotion gate + reproducible drift-recovery
demo; full dashboard + UX states + benchmarks + tests; one-command Docker demo + final
docs.

**Optional — only with demonstrated justification:** Redis (ingest buffering/caching),
Pandera (ingest schema validation), Prometheus (`/metrics` endpoint), vitest component
tests, tiny hyperparameter sweep, SHAP-style model explanations, additional public
datasets.

**Explicitly NOT built (resume-keyword traps):** Kafka/Redpanda or any streaming bus;
Grafana; ONNX; LSTM/TFT/deep learning; distributed stream processing; Kubernetes;
complex CI/CD pipelines (a basic lint+test workflow may be added after core completion);
Airflow/Prefect orchestrators; feature stores; auth/multi-tenancy; real-time
"self-learning" claims; more datasets than needed to make a point.

**Postponed until core is stable:** TimescaleDB, alerting integrations (email/Slack),
model explainability dashboards, additional forecast horizons/quantiles, multi-series
global models.

## 17. Project risks & mitigations

| Risk | Mitigation |
|---|---|
| Subtle time-series leakage (centered windows, future lags, split boundary bleed) | Leakage rules in `PULSEGUARD.md §3.3`; dedicated perturbation test; embargo gaps; single evaluation code path |
| Windows dev-environment friction (paths, venv, native deps) | Docker for Postgres/MLflow; pure-Python deps (psycopg binary, LightGBM wheels); venv via `python -m venv`; Git Bash for commands |
| NAB download/link rot | Loader reads local CSVs under `data/raw/` with documented manual download; Yahoo S5 fallback; feature degrades gracefully — sanity check only |
| MLflow/registry API churn | Pin MLflow 2.x in `pyproject.toml`; use client aliases (stable since 2.0); registry interactions isolated in `ml/tracking.py` |
| Invented-metrics temptation under deadline pressure | Rule 3 + every number requires a command; plan checkboxes demand real output |
| Scope creep / keyword-chasing | §16 lists are binding; new tech requires a plan update with justification first |
| Frontend chart perf on long ranges | `max_points` downsampling at the API; no client-side raw dumps |
| MAPE nonsense near zero values | Conditional MAPE (`|y| ≥ 1.0` default) with exclusions documented per report |
| PSI instability on sparse bins | Quantile-bin reference, ε-smoothing, minimum-bin-count guard |
| Gate thresholds set arbitrarily | Defaults are explicitly provisional; tuned once P2 measured baselines exist; always recorded in config, never hardcoded |

## 18. Definition of done (project)

- All six phases meet their completion criteria (checkboxes in §19 ticked with real
  verification output).
- `docker compose up --build` + seed script yields a working demo from a fresh clone.
- Evaluation report and benchmarks contain **only measured numbers**, each with its
  producing command; baselines compared honestly (including any losses).
- Champion/challenger promotion and rejection both demonstrated and auditable in the
  database; one reproducible induced drift-and-recovery cycle recorded.
- Test suites green; ruff clean; frontend build clean; no secrets or generated artifacts
  in git; docs truthful to the implemented state; the whole system explainable in an
  interview.

---

## 19. The six phases

### Phase 1 — Foundation, Data & Ingestion

**Objective.** A standing skeleton: Docker Compose brings up Postgres + FastAPI + React;
the seeded simulator generates labeled synthetic metrics; ingestion and query APIs work;
the dashboard charts a series. No models yet.

**Tasks.**
1. Root scaffold: `pyproject.toml` (runtime: fastapi, uvicorn[standard], sqlalchemy≥2,
   alembic, psycopg[binary], pydantic-settings, pandas, numpy, scikit-learn, lightgbm,
   mlflow, pyyaml; dev: pytest, httpx, ruff; package discovery `backend*`+`ml*`);
   `pip install -e .`.
2. Settings & DB: `app/core/config.py` (pydantic-settings reading `.env`),
   `app/core/database.py` (engine/session/Base); Alembic init; migration **001** —
   `metric_series`, `metric_points`.
3. ORM models + Pydantic schemas for series/points.
4. Ingestion service: validated bulk upsert, chunked inserts, idempotent re-ingestion.
5. API routers (P1 rows of the §4 table) incl. `date_bin` downsampling.
6. Simulator: `ml/simulator.py` + `ml/configs/simulator_default.yaml` + CLI
   (`python -m ml.simulator --config … [--out CSV] [--ingest --api-url …]`); components
   and labeling per `PULSEGUARD.md §4`; deterministic per seed; pure core function for
   testability. Default profile: 28 days @ 1-min, 3 series (smooth seasonal, spiky,
   shift-prone).
7. `ml/loaders.py`: NAB CSV → standard schema (documented manual download into
   `data/raw/nab/`); graceful error when absent.
8. Frontend scaffold (Vite React TS) in `frontend/`; typed API client; `Overview` +
   `SeriesDetail` pages (Recharts line); minimal loading/empty/error states.
9. Docker: `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` (db, api,
   frontend; volumes; healthchecks; env wiring). MLflow service deferred to P3.
10. Tests: simulator determinism + label sanity; ingest idempotency; query window/
    downsampling correctness; health.

**Expected files.** `pyproject.toml`; `backend/app/main.py`, `core/{config,database}.py`,
`models/{metric_series,metric_point}.py`, `schemas/*`, `api/{health,series,points}.py`,
`services/ingestion.py`; `backend/alembic.ini`, `backend/alembic/versions/001_*.py`;
`ml/simulator.py`, `ml/loaders.py`, `ml/configs/simulator_default.yaml`;
`frontend/**` (Vite scaffold); `backend/Dockerfile`, `frontend/Dockerfile`,
`docker-compose.yml`; tests in `backend/tests/`, `ml/tests/`.

**Dependencies.** External: Docker Desktop, Python 3.12, Node ≥ 20. Internal: none.

**Data flow.** simulator (seed) → CSV / `POST /points` → Postgres → `GET /points` →
React chart.

**API requirements.** P1 rows in §4; bulk ingest ≤50k points/request; idempotent upsert.

**ML requirements.** Simulator correctness only (labels, bounded anomaly rates,
reproducibility). No models.

**Testing.** As in task 10; test DB `pulseguard_test` on the compose Postgres.

**Verification commands.**
```
docker compose up -d --build
alembic -c backend/alembic.ini upgrade head
python -m ml.simulator --config ml/configs/simulator_default.yaml --ingest --api-url http://localhost:8000
curl http://localhost:8000/api/v1/series
curl "http://localhost:8000/api/v1/series/1/points?max_points=500"
pytest backend/tests ml/tests
```
Dashboard (`npm run dev`) shows the ingested series.

**Completion criteria.**
- [x] `docker compose up -d --build` starts db, api, frontend without manual fixes
- [x] Migration 001 applies; re-run is a no-op
- [x] Same seed → byte-identical simulator output; labels present at configured rates
- [x] Bulk ingest of a full 28-day series measured (record real elapsed time in the phase log)
- [x] Points query honors start/end/max_points (test-verified)
- [x] Dashboard lists series and renders values from the API (no mock data)
- [x] `pytest backend/tests ml/tests` green; `ruff check .` clean

**Expected working result.** Metrics can be generated, ingested, stored, queried, and
visualized end-to-end.

---

### Phase 2 — Feature Engineering & Core ML

**Objective.** A reproducible offline ML evaluation producing real measured metrics for
Isolation Forest anomaly detection and LightGBM quantile forecasting against baselines,
using walk-forward backtesting. No MLflow, no serving yet.

**Tasks.**
1. `ml/features.py` per the §6 feature spec; trailing-only; warmup dropped; shared later
   with serving.
2. `ml/backtest.py`: golden-split utility + walk-forward fold iterator (expanding window,
   optional max train window, embargo gap = max(horizon, max_lag)).
3. `ml/anomaly.py`: IsolationForest detector (fit on train folds; score =
   normalized −decision_function; threshold calibrated best-F1 on labeled validation
   window; percentile fallback) + rolling z-score baseline detector.
4. `ml/forecast.py`: quantile LightGBM per (h, τ) with time-ordered early-stopping tail;
   `QuantileForecaster` returning `target_ts, p05, p50, p95`; naive + seasonal-naive
   baselines.
5. `ml/evaluate.py`: precision/recall/F1/PR-AUC; MAE/RMSE/conditional MAPE; per-τ and
   mean pinball; per-fold + aggregated reporting with baseline comparison.
6. `ml/train.py` CLI (`--task anomaly|forecast --config ml/configs/train_synthetic.yaml`):
   full pipeline → `ml/reports/{task}_eval.json` (config SHA, seeds, per-fold + aggregate
   metrics, baselines) + printed table.
7. `ml/configs/train_synthetic.yaml`: dataset/split/feature/model/gate defaults
   (quantiles 0.05/0.5/0.95; horizons 1/5/15; splits 14d train / 9d walk-forward eval
   with daily refits / 5d golden).
8. Public sanity run (recommended): same pipeline on one labeled NAB series, reported
   separately as a sanity check.
9. Tests: feature leakage (perturb future → past features unchanged), warmup/trailing
   correctness, pinball vs hand-computed, PR metrics vs sklearn reference, splitter
   ordering + embargo, p05 ≤ p50 ≤ p95 sanity, simulator label-rate sanity.

**Expected files.** `ml/{features,backtest,anomaly,forecast,evaluate,train}.py`;
`ml/configs/train_synthetic.yaml`; `ml/tests/test_{features,backtest,metrics,models}.py`.

**Dependencies.** Phase 1 complete (data + simulator available).

**Data flow.** DB/CSV → features → walk-forward folds → fit/score → metrics JSON.

**API requirements.** None (offline phase).

**ML requirements.** `PULSEGUARD.md §3` in full: leakage rules, PR-AUC primary for
anomalies, pinball primary for forecasts, baselines mandatory, golden holdout untouched.

**Testing.** Unit + integration per task 9; one small end-to-end eval run in CI-style
smoke (tiny config).

**Verification commands.**
```
python -m ml.train --task anomaly  --config ml/configs/train_synthetic.yaml
python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml
pytest ml/tests
```

**Completion criteria.**
- [x] One command reproduces the full evaluation (fixed config + seeds; config SHA in report)
- [x] Walk-forward only — no random split anywhere (enforced + tested)
- [x] Anomaly report shows measured precision, recall, F1, PR-AUC
- [x] Forecast report shows measured MAE, RMSE, conditional MAPE, mean pinball per horizon
- [x] Both baselines compared honestly (beaten or the gap is explained in the report)
- [x] Leakage tests pass; reports written to `ml/reports/` with config hash + seed

**Expected working result.** A reproducible ML evaluation workflow producing real
measured metrics.

---

### Phase 3 — MLflow, Model Registry & Serving

**Objective.** Training runs tracked; models registered and versioned; champion alias
workflow; FastAPI inference loads the registered champion, computes features from stored
data, and persists predictions with model-version tracing.

**Tasks.**
1. Compose `mlflow` service (Postgres backend store, `mlartifacts` volume); tracking URI
   via env; §13 table updated.
2. `ml/models.py`: MLflow pyfunc wrappers — `AnomalyDetectorWrapper` (features →
   score/label) and `QuantileForecasterWrapper` (features → per-horizon quantiles) — so
   registered artifacts are self-contained.
3. `ml/tracking.py`: run creation with tags/params/metrics/artifacts; model logging;
   registration of `pulseguard-forecaster` / `pulseguard-anomaly-detector`; alias
   helpers (`champion`).
4. `ml/train.py --register`: after evaluation, log model + register version + scripted
   promotion to champion (gate automation arrives in P4).
5. Migration **002**: `anomaly_results`, `forecasts`, `active_models`.
6. `app/services/inference.py`: model manager (resolve alias at startup, cache, per-
   request alias-version check, `POST /api/v1/models/reload`); feature building from DB
   via `ml.features`.
7. Endpoints (P3 rows in §4): model info + reload; `GET …/forecast` (compute + persist);
   `POST …/anomalies/score`; stored-results reads with `model_version` filter.
8. Tests: registry alias resolution with a local-file tracking store fixture; inference
   roundtrip (seed points → score/forecast → persisted rows carry model name+version);
   reload logic; 503 when no champion.

**Expected files.** `ml/{models,tracking}.py`; `ml/train.py` (extended);
`backend/app/services/inference.py`; `backend/app/api/{models,inference}.py`; migration
`002_*.py`; new/updated tests.

**Dependencies.** Phases 1–2 complete; MLflow service added to compose.

**Data flow.** training → MLflow (run + registry + artifacts) → API loads champion by
alias → features from Postgres → predictions → back to Postgres with provenance.

**API requirements.** P3 rows in §4; 503 when model required but absent.

**ML requirements.** Registered artifacts must reproduce evaluation behavior (same
wrapper, same features); champion golden metrics stored at promotion.

**Testing.** As task 8; alias-change + reload demonstrated by test or scripted step.

**Verification commands.**
```
docker compose up -d mlflow            # UI at http://localhost:5000
python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml --register
curl http://localhost:8000/api/v1/models/forecast/champion
curl "http://localhost:8000/api/v1/series/1/forecast?horizon_minutes=15"
docker compose exec db psql -U pulseguard -d pulseguard -c "select target_ts, quantiles, model_version from forecasts limit 5;"
pytest backend/tests ml/tests
```

**Completion criteria.**
- [x] MLflow UI shows params/metrics/artifacts for evaluation runs
- [x] Model registered with ≥1 version; `champion` alias resolvable via API
- [x] API serves forecasts from the registered champion (not a local pickle)
- [x] Predictions persisted with model_version; `model_version` filter works
- [x] Alias change + reload swaps the served model (verified)
- [x] pytest green

**Expected working result.** A trained model can be registered and loaded by the
inference API.

---

### Phase 4 — Drift, Retraining & Promotion

**Objective.** The complete demonstrated lifecycle: PSI + residual drift detection,
controlled challenger retraining, measured champion comparison on the frozen golden set,
auditable promotion/rejection, and a reproducible induced drift-and-recovery cycle.

**Tasks.**
1. `ml/drift.py`: PSI implementation (§10 spec), per-feature PSI vs stored reference,
   residual monitor with sustained-breach rule; reference capture at champion training
   (artifact + `drift_references` row); CLI `python -m ml.drift`.
2. Migration **003**: `drift_references`, `drift_events`, `promotion_decisions`.
3. Endpoints (P4 rows in §4): drift status/events/check; promotions history; optional
   retraining trigger (BackgroundTasks).
4. `ml/retrain.py`: challenger pipeline (train on window incl. post-drift data) →
   evaluate champion AND challenger on the frozen golden set via the shared eval path →
   gate per §11 → promote (alias move + `active_models` update) or reject →
   `promotion_decisions` row with both metric sets + rule trace.
5. Promotion-gate config (`ml/configs/gates.yaml` or section in train config): primary/
   secondary/sanity criteria + minimum-data rule; final numbers tuned here once P2
   baselines are measured.
6. `scripts/drift_demo.py` + `ml/configs/drift_demo.yaml`: seeded scenario — healthy
   stream, champion serving, inject level shift, drift check fires (PSI event), retrain,
   gate decision (expected promotion), post-promotion recovery visible; writes
   `ml/reports/drift_demo_report.json`; must run twice with identical results.
7. Scripted rejection demonstration (e.g., challenger trained on insufficient/pre-drift
   data → gate rejects → champion unchanged).
8. Tests: PSI on known distributions; residual rule; gate branches (promote, reject,
   tolerance boundaries); golden-set immutability (hash); demo smoke test (tiny config).

**Expected files.** `ml/{drift,retrain}.py`; `scripts/drift_demo.py`;
`ml/configs/{drift_demo,gates}.yaml`; migration `003_*.py`; `backend/app/api/drift.py`,
`backend/app/services/{drift,promotion}.py`; tests.

**Dependencies.** Phase 3 complete (registry, serving, stored predictions).

**Data flow.** stored predictions/points → PSI/residual checks vs references →
`drift_events` → retrain → golden-set evaluation → gate → alias/`active_models`/
`promotion_decisions`.

**API requirements.** P4 rows in §4.

**ML requirements.** §10–§11 in full; golden set remains untouched by retraining;
rejection path first-class.

**Testing.** Unit + gate branches + demo smoke per task 8.

**Verification commands.**
```
python -m ml.drift --series-id 1                 # after injecting a shift
python -m ml.retrain --config ml/configs/drift_demo.yaml --task forecast
python scripts/drift_demo.py --config ml/configs/drift_demo.yaml
curl http://localhost:8000/api/v1/drift/status
curl http://localhost:8000/api/v1/promotions
pytest ml/tests backend/tests
```

**Completion criteria.**
- [x] PSI + residual monitoring implemented with stored references
- [x] Champion vs challenger compared on the frozen golden set; both metric sets recorded
- [x] Promotion AND rejection paths demonstrated (tests + ≥1 scripted rejection)
- [x] One seeded drift → retrain → decision → recovery cycle runs end-to-end, reproducibly
- [x] Drift status and promotion history visible via API (and dashboard-ready)
- [x] pytest green

**Expected working result.** A complete demonstrated ML lifecycle.

---

### Phase 5 — Dashboard, Evaluation & Hardening

**Objective.** A polished, interview-ready product: full dashboard over real data,
hardened UX, measured latency/throughput benchmarks, comprehensive tests.

**Tasks.**
1. Dashboard pages (§5 structure): SeriesDetail with zoomable chart, forecast band
   overlay, anomaly markers, model-provenance badge; Models page (champion per task,
   version list, evaluation metrics, champion-vs-challenger comparison); Drift page
   (status cards, per-feature PSI bars, events); Alerts feed; Overview cards.
2. Shared components (§5) with loading skeleton / empty / error states everywhere;
   responsive grid; accessible labels.
3. Migration **004**: `alerts`; alert creation on anomaly hits, forecast band breaches
   (once actuals arrive), and drift events; feed + acknowledge endpoints.
4. Backend hardening: validation edge cases, consistent error shape, CORS, request
   logging, graceful 503s.
5. Benchmarks: `scripts/bench_ingest.py` (bulk ingest throughput), `bench_query.py`
   (points query p50/p95), `bench_inference.py` (forecast + score endpoints, warm p50/p95;
   cold model-load noted separately). Results → `docs/BENCHMARKS.md` (measured only, with
   methodology + hardware context).
6. Frontend build + lint clean; optional vitest for pure utils; expanded API test
   coverage (error paths, validation).
7. Optional (justify first): Prometheus `/metrics`; Pandera ingestion validation.

**Expected files.** `frontend/src/**` (pages/components/hooks filled out); migration
`004_*.py`; `backend/app/api/alerts.py`; `scripts/bench_*.py`; `docs/BENCHMARKS.md`.

**Dependencies.** Phases 1–4 complete.

**Data flow.** Postgres (metrics, forecasts, anomalies, drift, promotions) → API →
dashboard; alerts derived from stored results.

**API requirements.** P5 rows in §4.

**ML requirements.** None new; dashboards surface measured evaluation metrics only.

**Testing.** Expanded backend tests; build/lint gates; manual UI walkthrough checklist
(append results to the phase log).

**Verification commands.**
```
cd frontend && npm run build && npm run lint
pytest backend/tests ml/tests
python scripts/bench_ingest.py
python scripts/bench_query.py
python scripts/bench_inference.py
```

**Completion criteria.**
- [x] All dashboard views render real backend data (zero mock data in the frontend)
- [x] Loading/empty/error states on every view; responsive at 1280px and 768px widths
- [x] Alerts feed wired to real events; acknowledge works
- [x] Benchmarks run; measured results recorded in `docs/BENCHMARKS.md` with methodology
- [x] Test suites green; frontend build clean

**Expected working result.** A polished, interview-ready product.

---

### Phase 6 — Final Demo, Docker & Documentation

**Objective.** A reproducible, recruiter-ready project: one-command startup, seeded demo
mode, final documentation with only measured numbers, fresh-machine verification.

**Tasks.**
1. Final compose polish: healthchecks, ordered `depends_on`, complete `.env.example`,
   documented `docker compose down -v` reset; frontend served as a static build.
2. `scripts/seed_demo.py`: idempotent fresh-clone seed — simulator series (one with an
   induced drift episode), ingest, train/register champion models (small config,
   documented runtime), one forecast + scoring pass so the dashboard is alive.
3. Full `README.md`: overview, architecture diagram (mermaid), quickstart tested
   verbatim, demo workflow, repo map, limitations.
4. `docs/EVALUATION_REPORT.md`: methodology summary, measured metric tables (from
   `ml/reports/`, with producing commands), baseline comparison, drift-demo outcome,
   limitations & threats to validity (synthetic-dominant evaluation, single node,
   threshold sensitivity, MAPE exclusions).
5. Final verification from a clean state: `docker compose down -v && docker compose up
   --build` → seed → walkthrough (visualize → forecast → drift demo → promotion) → all
   tests green → docs numbers match reports.
6. Cleanup: dead code, TODO resolution, `.gitignore` audit, no large/generated files in
   git history going forward.

**Expected files.** `scripts/seed_demo.py`; final `README.md`; `docs/EVALUATION_REPORT.md`;
polished `docker-compose.yml`; `docs/BENCHMARKS.md` (from P5).

**Dependencies.** Phases 1–5 complete.

**Data flow.** seed script → simulator → ingest → train/register → API/dashboard;
everything reproducible from the seed.

**API requirements.** No new endpoints; stability of existing ones.

**ML requirements.** Demo models trained by the same CLI paths as the real workflow —
no special demo shortcuts.

**Testing.** Full suites from a clean state; verbatim quickstart test; walkthrough
checklist.

**Verification commands.**
```
docker compose down -v && docker compose up --build
python scripts/seed_demo.py
pytest backend/tests ml/tests
cd frontend && npm run build
```

**Completion criteria.**
- [ ] Fresh-machine runbook works: clone → compose up → seed → demo
- [ ] README quickstart accurate when executed verbatim
- [ ] Evaluation report contains only measured numbers with repro commands
- [ ] All six phases' completion criteria checked
- [ ] Repo clean: no generated artifacts, no secrets, docs truthful

**Expected working result.** A reproducible, recruiter-ready project.

---

## Phase log (measured results & deviations)

### Phase 1 — Foundation, Data & Ingestion (completed 2026-09-13)

Verification results (all commands from §19 Phase 1 run on Windows 11, Docker Desktop
with WSL2 backend, Python 3.12.10):

- **Migration 001:** applied; second `upgrade head` ran as a no-op; `alembic current`
  → `0001 (head)`.
- **Simulator:** 3 series × 28 days @ 1 min = 120,960 points; anomaly rates measured
  0.61% / 0.72% / 0.94% per series; all four anomaly types present in every series.
  Determinism verified by SHA-256: two runs of the same config produced byte-identical
  CSVs (`d94ce795…45883`).
- **Bulk ingest (client-measured, local venv → compose API → Postgres):** 40,320-point
  series ingested in 7.9–9.7 s per series; all 3 series (120,960 points) in 28.1 s total.
- **Tests:** `pytest backend/tests ml/tests` → 26 passed. `ruff check .` → clean.
- **Frontend:** `npm run build` (tsc + vite) succeeds; bundle ~532 kB minified
  (Recharts-dominated; code-splitting is a Phase 5 option). Dashboard verified in a
  browser: series list renders from the API, series detail renders a bucket-averaged
  chart (1,998 points of 40,320, bucket = 1,211 s) with visible daily seasonality,
  spikes, and the permanent level shift.

Deviations and fixes made during the phase (all reflected in the code):

1. **Bulk-insert counting:** `CursorResult.rowcount` returns −1 for bulk
   `ON CONFLICT DO NOTHING` inserts on psycopg3; ingestion now counts via `RETURNING id`.
2. **Validation-error hardening:** pydantic 422 payloads can contain non-finite floats
   and exception objects that the (strict) JSON encoder rejects; a
   `RequestValidationError` handler sanitizes them and returns the standard
   `{"detail", "code": "validation_error"}` shape.
3. **Frontend port:** the compose frontend is published on **host port 5174** (container
   still 5173) because another Docker/WSL2-relayed stack on this development machine
   already owns `[::1]:5173` (wslrelay), which made `localhost:5173` resolve
   nondeterministically. CORS allows 5173 and 5174 on both `localhost` and `127.0.0.1`.
4. **API base URL pinned to `127.0.0.1`:** the same WSL2 relay also owns
   `[::1]:8000` for the other stack, so browser fetches to `localhost:8000` could hit
   the wrong server; IPv4 loopback is unambiguous. Default in `client.ts`, compose env,
   and `.env.example`.
5. **`useFetch` dependency fix:** the fetcher closure is excluded from the effect deps
   (callers pass a fresh closure each render; including it caused an infinite refetch
   loop that kept views in the loading state).
6. **Frontend dev bind mount:** compose mounts `frontend/src` (and config files) into
   the dev container so host edits hot-reload instead of requiring an image rebuild.

Implementation notes for Phase 2: the API test suite requires the compose Postgres
running (`docker compose up -d db`); tests use a dedicated `pulseguard_test` database
created automatically by the fixtures.

### Phase 2 — Feature Engineering & Core ML (completed 2026-09-13)

Reproduce with:
```
python -m ml.train --task anomaly  --config ml/configs/train_synthetic.yaml
python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml
python -m ml.train --task anomaly  --config ml/configs/train_public.yaml  --report anomaly_public_eval.json
python -m ml.train --task forecast --config ml/configs/train_public.yaml --report forecast_public_eval.json
```
Reports (with config SHA-256, seed, golden-set hash, per-fold metrics): `ml/reports/`.
Setup: Windows 11, Python 3.12.10, seed 42; splits 14d train / 9d walk-forward eval
(daily refits, embargo = max horizon = 15 rows) / 5d frozen golden; 28 features
(value, 7 lags, 3×4 rolling, 2 roc, z-score, 4 calendar).

> **Corrected 2026-09-13 (Phase 3):** the numbers below were re-measured after a
> warmup-alignment bug was discovered during Phase 3 (feature rows were paired with
> targets 60 rows stale — effectively predicting the recent past, which inflated
> forecast quality). The aligned results are the honest ones and supersede the
> originally logged tables. The bug and fix are described in the Phase 3 log.

**Forecast — synthetic (mean pinball / MAE / conditional MAPE on median; Δ vs best
baseline on mean pinball):**

| series | h=1 | h=5 | h=15 |
|---|---|---|---|
| api_latency_p95_ms | 0.970 / 3.69 / 2.36% / −57.9% | 1.018 / 3.71 / 2.37% / −57.2% | 0.989 / 3.73 / 2.38% / −58.8% |
| cpu_usage_pct | 0.650 / 2.22 / 8.42% / −44.7% | 0.666 / 2.21 / 8.32% / −43.4% | 0.660 / 2.32 / 8.75% / −45.5% |
| request_rate_rps | 10.309 / 39.80 / 4.81% / −0.2% | 10.790 / 41.55 / 5.05% / **+0.1%** | 11.254 / 40.41 / 4.98% / −3.4% |

The quantile model beats the best baseline on 8 of 9 series-horizon pairs; the
exception is request_rate h=5, where the naive baseline wins by 0.09% (recorded in the
report notes — the request-rate level shift inside the evaluation region hurts the
model there). Quantile crossing is measured per fold, not hidden. Elapsed 350.0 s.

**Anomaly — synthetic (pooled over 9 daily folds, per-fold refit + per-fold best-F1
threshold on training-window labels):**

| series | precision | recall | F1 | PR-AUC pooled | z-score PR-AUC |
|---|---|---|---|---|---|
| api_latency_p95_ms | 0.074±0.210 | 0.044±0.126 | 0.056 | 0.495 | 0.852 |
| cpu_usage_pct | 0.192±0.322 | 0.185±0.343 | 0.122 | 0.100 | 0.038 |
| request_rate_rps | 0.128±0.309 | 0.512±0.464 | 0.131 | 0.156 | 0.271 |

Honest findings, recorded in the report's notes: (1) on cpu_usage, fold 0 (stationary)
reaches PR-AUC 0.984 with 143 TP / 2 FP — the detector works when training matches the
evaluated regime; (2) cpu_usage and request_rate have a permanent level shift INSIDE the
evaluation region — the post-shift regime is unlabeled, inflates false positives until
walk-forward refits absorb the new level, and the instantly-adapting rolling z-score
wins pooled PR-AUC there; that is precisely the drift effect Phase 4 retrains on;
(3) api_latency has only 18 eval anomalies — per-fold P/R are high-variance there.
Elapsed 53.2 s.

**Public sanity check — NAB `ec2_cpu_utilization_53ea38` (4,032 pts @ 5 min, 2 label
windows ≈ 10% of rows, fractions split 0.6/0.3/0.1, 11 folds):**
anomaly: PR-AUC pooled 0.430 (Isolation Forest) vs 0.204 (z-score) — on this real
series the IF wins; flagged 228 of 1,209 eval rows (201 labeled). Forecast: h=1 mean
pinball 0.0155 (MAE 0.057, MAPE 3.00%, −49.0% vs seasonal-naive); h=6 mean pinball
0.0119 (−57.5% vs naive). Sanity check only — not proof of real-world validity.

**Tests:** `pytest backend/tests ml/tests` → 60 passed (46 ml incl. the
future-perturbation leakage test, fold/embargo tests, hand-computed metric tests,
detector/forecaster behavior tests, and an end-to-end config→report smoke test).
`ruff check .` clean.

**Deviations and fixes made during the phase:**
1. Anomaly evaluation moved from a static per-series detector to true per-fold
   walk-forward refits (per `PULSEGUARD.md §3.3`): the static run exposed the
   level-shift regime-flagging collapse; per-fold refits with per-fold best-F1
   thresholds (historical labels only; percentile fallback when a training window has
   no positives) keep the evaluation honest while showing the drift-then-adapt arc.
2. `calibrate_best_f1` implemented as an exact O(n log n) scan over sorted unique
   scores (replaces a 500-point quantile grid).
3. Fraction splits accumulate per-region row counts (float drift gave 899 instead of
   900 rows); `forecast_metrics` exposes clean `mape`/`mape_n_used`/`mape_n_total` keys.
4. NAB file name corrected (`ec2_cpu_utilization_24ae8bf` does not exist in the repo;
   the actual series used is `ec2_cpu_utilization_53ea38`), downloaded into
   `data/raw/nab/` and standardized with `ml.loaders`.
5. Newer lightgbm emits an `eval_set` deprecation warning; kept the compatible API
   (revisit at the next dependency bump).

### Phase 3 — MLflow, Model Registry & Serving (completed 2026-09-13)

**What was delivered.** MLflow 3.16 server on the compose network (Postgres backend
store `mlflow` DB, shared `mlartifacts` volume); `ml/models.py` pyfunc wrappers
(`AnomalyDetectorWrapper`, `QuantileForecasterWrapper` — self-contained artifacts
carrying models, normalization, threshold, horizons/quantiles and feature names);
`ml/tracking.py` (run logging, registration, `champion` alias helpers); `ml/train.py
--register` (trains a pooled, series-agnostic champion on the full initial training
region and promotes it); migration 002 (`forecasts`, `anomaly_results`,
`active_models`); `backend/app/services/inference.py` (ModelManager with per-request
alias-version checks) + `/api/v1/models*` and inference endpoints.

**Registered champions** (via `docker compose run --rm api python -m ml.train …
--register`): `pulseguard-forecaster` and `pulseguard-anomaly-detector`, both with the
`champion` alias; runs carry tags (`pulseguard.task`, `pulseguard.config_sha256`),
params (feature config JSON, horizons, quantiles, threshold, seed), 27/12 flattened
eval metrics, and the eval report artifact. MLflow UI verified in a browser
(experiment + runs present).

**Verified end-to-end (measured):**
- `GET /api/v1/series/4/forecast?horizon_minutes=15` → band {p05 127.29, p50 134.68,
  p95 141.70} for target 2026-09-13T00:14Z, persisted with `model_version=1`;
  `?model_version=` filter verified against the stored rows.
- `POST /api/v1/series/5/anomalies/score` → 200 points scored, persisted rows with
  score/is_anomaly/threshold/model_version; idempotent re-scoring verified.
- Alias bump (new version + alias move) picked up by the API **without restart**
  (next request served the new version); `POST /api/v1/models/reload` re-syncs
  `active_models`.
- Champion alias deleted → inference returns **503** `model_unavailable`; alias
  restored → reload → 200.

**Deviations and fixes made during the phase:**
1. **Warmup-alignment bug (critical, found via Phase 3 tests):** feature frames keep
   original-position labels (60..n−1) but training/eval code used those labels as
   positional indices into the feature matrix — pairing every feature row with targets
   60 rows stale (i.e., effectively predicting the recent past). Fixed everywhere by
   selecting X rows with boolean masks / true positions; a label↔position invariant
   test now pins the convention. All Phase 2 reports were re-measured with the fixed
   code (see the corrected Phase 2 log above).
2. **mlflow 3.16 API drift:** file-based tracking stores are rejected (tests use a
   sqlite store via the `MLFLOW_TRACKING_URI` env var); alias management is
   `set/delete_registered_model_alias`; `search_registered_models(filter_string=…)`;
   `create_model_version` requires an existing registered model; `log_model(name=…)`
   replaces `artifact_path`.
3. **MLflow server security middleware** (3.16 default: localhost-only) rejected the
   api container's `Host: mlflow:5000` header with 403 — compose command now sets
   `--allowed-hosts '*'` (closed local network; documented).
4. **LightGBM in slim images** needs `libgomp1` — added to `backend/Dockerfile`.
5. Registered champions are pooled (series-agnostic) models — the plan's postponed
   list item "multi-series global models" is superseded by this deliberate choice
   (single champion per task keeps the lifecycle story simple); per-series
   specialization remains postponed.
6. Host-side tracking defaults use `127.0.0.1:5000` (dual-stack localhost trap, same
   as the API base URL in Phase 1).

### Phase 4 — Drift, Retraining & Promotion (completed 2026-09-13)

**What was delivered.** `ml/drift.py` (PSI with quantile-bin references, per-series
reference-set builder, rolling residual monitor, promotion gate); migration 003
(`drift_references`, `drift_events`, `promotion_decisions`); `ml/retrain.py` (champion
golden-set evaluation → challenger training on a configured window → gate →
promote/reject with audited decisions); reference capture in `--register` (artifact +
`drift_references` cache); backend drift service + endpoints (`GET /drift/status`,
`GET /drift/events`, `POST /drift/check`, `GET /promotions`,
`POST /retraining/trigger` with a BackgroundTasks worker); `scripts/drift_demo.py`
+ `ml/configs/drift_demo.yaml`; simulator `at_day` option for fixed level-shift
placement (used by tests).

**Demonstrated cycle (measured, reproducible — two demo passes produce identical
reports after stripping versions/timestamps/elapsed):**
- PSI drift fired on all three series vs the pre-drift references (worst features:
  cpu `roll_60_mean` = 3.21, request_rate `roll_15_max` = 1.51, latency
  `roll_15_min` = 0.55 — the seeded level shifts at days 15.7/16.3 sit between the
  reference window and the recent window).
- Residual monitor (pre-promotion): request_rate `drift`, latency `warn`, cpu `ok`.
- Forecast: expanding challenger **PROMOTED** — mean pinball improved **71.05%** on
  the frozen golden set; pre-drift challenger **REJECTED** (primary, secondary and
  sanity rules all failed). Decisions recorded in `promotion_decisions` with both
  metric sets and the rule trace.
- Anomaly: expanding challenger **REJECTED** — golden PR-AUC did not improve
  (honestly measured; the golden region contains very few labeled anomalies, so the
  metric is noisy). The gate held: no promotion without measured improvement.
- Recovery: after the forecast promotion the residual monitor reads `ok` on all
  series against the new champion; the new champion's reference covers the current
  regime by construction.
- Drift status and promotion history visible via the API (159 drift events, 33
  promotion decisions across the demo/test runs at verification time).

**Reproducibility:** run 1 vs run 2 reports (`ml/reports/drift_demo_report_run{1,2}.json`)
are identical after stripping version numbers, timestamps and elapsed time. Demo
command (Git Bash):
```
MSYS_NO_PATHCONV=1 docker compose run --rm   -v "$(pwd -W)/scripts:/code/scripts" -v "$(pwd -W)/ml/configs:/code/ml/configs"   -v "$(pwd -W)/ml/reports:/code/ml/reports" api   python scripts/drift_demo.py --config ml/configs/drift_demo.yaml   --api-url http://api:8000 --report ml/reports/drift_demo_report_run1.json
```

**Deviations and decisions made during the phase:**
1. Reference distributions capture the champion's **trailing 7-day training window**
   (its recent regime) rather than all training rows, and calendar encodings
   (`hour_*`, `dow_*`) are excluded from PSI — a shorter check window vs a 7-day
   reference otherwise fires spuriously on day-of-week composition. PSI check windows
   are matched to the reference span (7 days) in the demo config.
2. Residual baseline = the champion's median-forecast MAE over the **final 2 days of
   its training window** (a healthy period), not the golden set — the seeded shifts
   sit inside the golden region and would contaminate a golden-based baseline.
   Baselines are persisted at promotion time (`drift_references`, kind `residual`).
3. The anomaly challenger's rejection is kept as the recorded outcome (no
   metric-chasing); the anomaly recovery narrative therefore rests on the gate
   holding and the PSI status remaining consistent with the un-promoted champion.
4. The champion endpoint resolves the alias live on every read (`ensure_current` +
   sync) — promotions performed outside the API process are reflected immediately.
5. LightGBM/IsolationForest thread counts are configurable (`n_jobs`) and capped in
   the demo config after a run container was OOM-killed; Docker DNS flakiness killed
   two demo containers mid-run (infrastructure; retried cleanly).

**Tests:** 99 passed — 21 new (PSI math incl. a hand-computed case, known-shift
detection, clipping, residual monitor branches, gate promote/reject/boundary/
insufficient-data, golden-set identity, retraining promote/reject smoke on a tiny
seeded config, and the drift API endpoints incl. the retraining trigger recording a
decision). `ruff check .` clean.

### Phase 5 — Dashboard, Evaluation & Hardening (completed 2026-09-13)

**What was delivered.**
- **Alerts (backend):** migration 004 (`alerts` table with `(series_id, kind,
  dedupe_key)` dedupe identity); alerts raised on anomaly hits (per flagged point),
  forecast band breaches (actuals outside the stored p05–p95 band, checked during
  drift checks), and PSI/residual drift events; `GET /api/v1/alerts` (+ series and
  acknowledged filters) and `POST /api/v1/alerts/{id}/ack`.
- **Dashboard (frontend):** navigation across five pages — Overview (system summary
  cards: champions + loaded state, drift status, open alerts), SeriesDetail (value
  line + shaded p05–p95 forecast band + dashed median + anomaly markers + Recharts
  Brush zoom + model-provenance lines), Models (champion cards with horizons/
  quantiles/threshold, registered versions with aliases, promotion-decision table),
  Drift (per-series status cards, per-feature PSI bar chart for the selected series,
  recent events table), Alerts (feed with severity/kind badges, acknowledge action,
  unacknowledged/all filter). Every view renders live API data — zero mock data —
  with loading/empty/error states; responsive grids verified at 1280px and 768px in
  a browser.
- **ESLint** (flat config, react-hooks + typescript-eslint) added; `npm run lint`
  clean; `npm run build` (tsc + vite) clean.
- **Benchmarks:** `scripts/bench_ingest.py`, `bench_query.py`, `bench_inference.py`;
  measured results in `docs/BENCHMARKS.md` (methodology + hardware context).

**Measured benchmark results** (local Docker Desktop, WSL2; details and caveats in
`docs/BENCHMARKS.md`): bulk ingest 20,000 pts in 5k chunks → **2,854 pts/s** (5k-batch
p50 1.76 s); downsampled 2,000-row points query over 40,320 pts → **p50 44 ms**
(p95 1.95 s — one cold outlier after restart); warm forecast serving → **p50 54 ms**
(h=1) / 55 ms (h=15), anomaly scoring of 400 rows → p50 199 ms.

**Live verification:** a drift check on cpu_usage created 6 forecast-breach alerts +
a drift alert; the feed lists them with payloads; acknowledge persists (1 acked).
SeriesDetail shows the p05–p95 band overlay from 287 stored forecasts with the
champion provenance badge (pulseguard-forecaster v24).

**Deviations and decisions made during the phase:**
1. Stored-forecast reads are newest-first (charts surface recent predictions; the
   Phase 3 assertion was updated accordingly).
2. The champion endpoint now resolves the alias live on every read (stale-mirror
   fix carried over from the Phase 4 demo).
3. Optional items (Prometheus, Pandera, vitest component tests) were **not** added —
   no concrete need demonstrated yet; they remain in the optional list.
4. ESLint was not present after Phase 1 (build-only setup); Phase 5 adds the
   standard flat-config toolchain the completion criteria require.
5. Benchmark scripts run on the host against the live compose stack and write
   nothing automatically — numbers are transcribed into `docs/BENCHMARKS.md` with
   methodology and explicit non-claims (no scalability/concurrency measurement).

**Tests:** 103 passed — 4 new (anomaly alerts + dedupe + ack flow, drift alerts on
PSI drift with a shifted window, forecast-breach alerts from a seeded out-of-band
forecast, ack of an unknown alert → 404).
