# PulseGuard

Real-time metric anomaly detection & probabilistic forecasting platform: ingest service
metrics, detect anomalies (Isolation Forest), forecast expected ranges with uncertainty
bands (LightGBM quantile regression), track and register models (MLflow), monitor drift,
and run controlled champion/challenger retraining with measured promotion decisions.

> **Status: bootstrap phase.** Project foundation and implementation plan are in place;
> implementation has not started yet.

- [`PULSEGUARD.md`](PULSEGUARD.md) — project source of truth: purpose, architecture, ML
  methodology, data strategy, lifecycle rules, development rules.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — detailed six-phase
  execution plan (objectives, tasks, verification, completion criteria per phase).

The full README (quickstart, architecture diagram, demo workflow) is delivered in Phase 6,
once the system it describes actually exists.
