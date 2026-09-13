# PULSEGUARD.md — PulseGuard Source of Truth

> **Every session starts here.** This file defines what PulseGuard is, what it must never
> become, and the rules all work must follow. The detailed phase-by-phase execution plan
> lives in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md). If code and this
> file disagree, fix the code — or update this file deliberately, never silently.
>
> - **Status:** Phase 4 complete (2026-09-13). Phase 5 — Dashboard, Evaluation & Hardening — is next.
> - **Last updated:** 2026-09-13

---

## 1. Purpose

PulseGuard is a **real-time metric anomaly detection & probabilistic forecasting platform** —
a production-oriented ML system that:

1. ingests service/infrastructure time-series metrics,
2. performs preprocessing and feature engineering,
3. detects anomalous behavior (unsupervised scoring + thresholded classification),
4. forecasts expected future metric ranges with uncertainty bands,
5. evaluates models quantitatively (walk-forward backtesting, never random splits),
6. tracks experiments and registers/version-controls models (MLflow),
7. serves models through a FastAPI inference API with prediction persistence and
   model-version tracing,
8. monitors data/model drift (PSI + residual behavior) and triggers **controlled** retraining,
9. trains challenger models, compares them against the current champion on a frozen golden
   set, and promotes or rejects them based on measured results,
10. visualizes all of it in a React dashboard.

The full demonstrated lifecycle:

```
data → preprocessing → feature engineering → training → validation → evaluation
     → registration → inference → monitoring → retraining → promotion / rejection
```

**Relationship to CodeGraphV2:** CodeGraphV2 covers code/graph/LLM analysis. PulseGuard is
strictly **classical ML over numeric time-series**. There is no LLM, RAG, semantic search,
embedding, or code-analysis component here. The only conceptual future tie-in: CodeGraphV2
could emit operational metrics that PulseGuard ingests (not planned, not built).

## 2. What PulseGuard must NOT become

- Not a RAG app, chatbot, semantic-search system, generic AI assistant, or code analyzer.
- Not a résumé-keyword showcase: no technology without implemented, verifiable work behind it.
- Not a claim of autonomous self-learning production behavior: retraining is explicitly
  triggered and demonstrated, and we say so.
- No fake anything: no mock data in the UI, no invented metrics, no performance claims
  without a reproducible measurement.

## 3. ML methodology

### 3.1 Anomaly detection — Isolation Forest (primary)

- Unsupervised `sklearn.ensemble.IsolationForest` over trailing-window features.
- Output is a continuous anomaly score; a calibrated threshold turns scores into binary
  labels. Calibration: best-F1 threshold on a labeled synthetic validation window
  (score-percentile fallback for unlabeled/public data).
- Metrics: Precision, Recall, F1, **PR-AUC (primary — anomalies are rare)**.
- Mandatory baseline: rolling z-score detector, compared honestly.

### 3.2 Forecasting — LightGBM quantile regression (primary)

- `lightgbm` with `objective="quantile"`, one model per (horizon, quantile), wrapped and
  deployed as a single artifact that returns the full band.
- Default quantiles **τ ∈ {0.05, 0.50, 0.95}** → 90% prediction band (lower / median /
  upper). Default horizons **{1, 5, 15} minutes**. All values live in config, not code.
- Training & evaluation loss: **pinball loss** (mean across quantiles), plus MAE, RMSE,
  and MAPE computed only where `|actual| ≥ 1.0` (exclusions documented in every report).
- Mandatory baselines: naive last-value and seasonal-naive (daily season for 1-min data).

### 3.3 Validation — leakage rules (non-negotiable)

1. **No random splits.** Strict temporal ordering everywhere.
2. **Walk-forward backtesting:** expanding-window refits at each fold, with a gap
   (embargo) of `max(horizon, max_lag)` at every fold boundary.
3. Features at time *t* use only data ≤ *t*: trailing windows only, no centered rolling,
   no negatively-shifted lags.
4. Targets are future values; a prediction made at *t* targets *t+h*.
5. A final **golden holdout** segment is frozen and never used for training, threshold
   calibration, or tuning. It is the only comparison ground for champion vs challenger
   (Phase 4).
6. Every split, seed, and config is recorded (config SHA-256 in every report and MLflow
   run). All metrics come from one shared evaluation code path.

## 4. Data strategy

### A. Controlled synthetic data (primary)

`ml/simulator.py` — config-driven, seeded metric simulator generating: trend, daily/weekly
seasonality, Gaussian noise, spikes, level shifts, variance changes, and contextual
anomalies, with **ground-truth labels** (`is_anomaly`, `anomaly_type`).

Labeling convention: permanent **level shifts** label only their transition window — the
new level becomes the new normal afterwards. Temporary regimes (spikes, variance-change
windows, contextual anomalies) label the affected points, since they revert instead of
becoming a new normal.

### B. Public benchmark data (secondary sanity check)

NAB (default choice — `realAWSCloudwatch` labeled CSVs; loader: `ml/loaders.py`), with
Yahoo S5 as a fallback option. Used to sanity-check the pipeline on real data only. Never
presented as proof of real-world validity. Dataset statistics are measured and reported,
never invented.

## 5. Model lifecycle

```
experiment (MLflow run) → candidate → evaluation (walk-forward + golden set)
  → registered version → challenger → promotion gate
  → champion alias (promoted)   |   rejected (recorded)
```

- A challenger **never** automatically replaces the champion. Promotion requires measured
  improvement on the frozen golden set (gate rules in `IMPLEMENTATION_PLAN.md §11`,
  threshold values in config).
- Every promotion/rejection is persisted in `promotion_decisions` with both metric sets
  and the rule trace.
- The API serves the champion resolved via MLflow alias; every stored prediction records
  model name + version (traceability).

## 6. Drift & retraining

- **PSI** (Population Stability Index) on input/feature distributions vs the champion's
  stored training reference. Thresholds: `< 0.1` stable, `0.1–0.2` moderate, `> 0.2`
  significant drift.
- **Residual monitoring:** rolling forecast error vs the golden-set residual baseline;
  sustained elevation = drift signal.
- Demonstrated flow: `drift detected → retraining triggered (controlled: CLI/scripted,
  NOT a hidden daemon) → challenger trained → evaluated on golden set → compared to
  champion → promoted OR rejected`.
- At least one **reproducible induced drift-and-recovery cycle** must run end-to-end
  (seeded script, Phase 4). We do not claim continuous real-world self-learning.

## 7. Technology stack

| Layer | Required |
|---|---|
| Backend / ML | Python 3.12+, FastAPI, Uvicorn, SQLAlchemy 2, Alembic, PostgreSQL 16, pandas, NumPy, scikit-learn, LightGBM, MLflow 2.x, pytest |
| Frontend | React 18, TypeScript, Vite, Recharts |
| Infra | Docker Compose |

**Optional — only when genuinely justified (justify in the plan before adding):**
Redis, Pandera, Prometheus.

**Explicitly out of scope until the core system is done** (stretch work only):
Kafka/Redpanda, Grafana, ONNX, LSTM/TFT or any deep learning, distributed stream
processing, Kubernetes, complex CI/CD.

## 8. Repository layout (target)

```
PulseGuard/
├── PULSEGUARD.md                  # this file
├── README.md                      # full README delivered in Phase 6
├── docs/
│   └── IMPLEMENTATION_PLAN.md     # detailed 6-phase execution plan
├── backend/                       # FastAPI application (package `backend.app`)
│   ├── app/
│   │   ├── api/                   # routers: series, points, inference, models, drift, alerts
│   │   ├── core/                  # settings (pydantic-settings), database session
│   │   ├── models/                # SQLAlchemy ORM models
│   │   ├── schemas/               # Pydantic request/response schemas
│   │   └── services/              # ingestion, inference, drift, promotion logic
│   ├── alembic/                   # migrations (001 core → 004 alerts)
│   ├── tests/                     # API contract & integration tests (pytest)
│   └── Dockerfile
├── ml/                            # offline ML: simulator, features, training, drift (package `ml`)
│   ├── simulator.py               # seeded, config-driven labeled metric generator (P1)
│   ├── loaders.py                 # public dataset loader (NAB), standardized schema (P1)
│   ├── features.py                # trailing-window feature builder — SHARED train+serve (P2)
│   ├── backtest.py                # golden split + walk-forward folds with embargo (P2)
│   ├── anomaly.py                 # IsolationForest detector + z-score baseline (P2)
│   ├── forecast.py                # LightGBM quantile forecaster + naive baselines (P2)
│   ├── evaluate.py                # precision/recall/F1/PR-AUC, MAE/RMSE/MAPE/pinball (P2)
│   ├── models.py                  # MLflow pyfunc wrappers (detector, forecaster) (P3)
│   ├── tracking.py                # MLflow run/param/metric/artifact helpers (P3)
│   ├── train.py                   # CLI: run full evaluation, optionally register (P2/P3)
│   ├── drift.py                   # PSI + residual monitoring, drift events (P4)
│   ├── retrain.py                 # challenger training + promotion gate CLI (P4)
│   ├── configs/                   # YAML configs: simulator, training, gates, demo
│   ├── reports/                   # gitignored evaluation outputs (measured metrics JSON)
│   └── tests/                     # unit + leakage + integration tests (pytest)
├── frontend/                      # React 18 + TypeScript + Vite (scaffolded in P1)
├── data/
│   ├── raw/                       # gitignored: downloaded public datasets
│   └── generated/                 # gitignored: simulator output
├── scripts/                       # seed_demo.py, drift_demo.py, bench_*.py (P4–P6)
├── docker-compose.yml             # db + api + frontend (P1); + mlflow (P3)
├── .env.example                   # dev-only placeholder settings
└── .gitignore
```

**Single Python environment:** `backend*` and `ml*` are packages of one root
`pyproject.toml`. Training jobs and the API share one codebase (and one Docker image);
`ml/features.py` is the single source of feature logic for both training and serving —
no train/serve skew.

## 9. Development & quality rules

1. **Phases in order.** Exactly six phases (`docs/IMPLEMENTATION_PLAN.md`). Never start a
   phase before the previous one meets its completion criteria. No Phase-1 implementation
   until explicitly instructed ("Read PULSEGUARD.md and execute Phase 1.").
2. **Session protocol:** read this file first → work only on the current phase → run that
   phase's verification commands → update its completion checkboxes in the plan.
3. **Accuracy over impressive claims.** Real data over fake UI data. Real measured metrics
   only. Never invent: model performance, dataset size, latency, throughput, accuracy, F1,
   PR-AUC, MAE, RMSE, pinball loss, improvement percentages.
4. Every résumé keyword must correspond to actual, implemented work.
5. No secrets in git (`.env` is ignored; `.env.example` holds dev-only placeholders).
   No generated artifacts committed (`data/`, `mlruns/`, `mlartifacts/`, `ml/reports/`).
6. Type-hinted Python; ruff-clean; pytest coverage of all core pure logic; no notebooks as
   source of truth.
7. Keep dependencies minimal; every addition is justified in the plan first.
8. Keep the system explainable in an interview: simple, standard patterns over cleverness.
   No giant refactors without a documented reason.
9. Documentation stays lightweight and truthful: `PULSEGUARD.md` (what/why/rules),
   `docs/IMPLEMENTATION_PLAN.md` (how/status), plus README, `docs/EVALUATION_REPORT.md`
   and `docs/BENCHMARKS.md` only once real measured numbers exist. No doc sprawl.
