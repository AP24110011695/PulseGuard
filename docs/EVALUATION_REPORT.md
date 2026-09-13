# PulseGuard Evaluation Report (Phase 6)

All numbers below are **measured** by code in this repository with the reproducing
command recorded. Machine context: Windows 11, Python 3.12.10, Docker Desktop (WSL2),
seed 42, single node. Serving/benchmarks: `docs/BENCHMARKS.md`.

---

## 1. Methodology

### Data

- **Primary — synthetic (config-driven, seeded):** `ml/simulator.py` generates
  1-minute metric streams (28 days × 3 series = 120,960 points) with trend, daily/weekly
  seasonality, noise, spikes, permanent level shifts, variance changes, and contextual
  anomalies, with ground-truth labels. Labeling convention: permanent level shifts label
  only their transition window; temporary regimes label the affected points. Config:
  `ml/configs/simulator_default.yaml`.
- **Secondary — public sanity check:** one labeled NAB series
  (`ec2_cpu_utilization_53ea38`, 4,032 points @ 5 min, 2 label windows ≈ 10% of rows),
  standardized by `ml/loaders.py`. Sanity check only — never proof of real-world
  validity.

### Features

Trailing-only, shared by training and serving (`ml/features.py`): the current value, 7
lags (1–60 rows), rolling mean/std/min/max over 5/15/60-row windows, two rates of
change, a z-score against a past-exclusive window, and calendar encodings (28 columns).
Warmup rows dropped. A regression test pins the label↔position convention and a
perturbation test proves no future information leaks into past features.

### Validation — walk-forward only

No random splits anywhere (grep-verifiable; enforced by tests). Timeline:
**14 days training / 9 days walk-forward evaluation (daily refits, embargo = max
horizon = 15 rows) / 5 days frozen golden holdout**. The golden set is never used for
training, early stopping, or threshold calibration; its identity is hashed
(`golden_sha256` in every report) and it is the only comparison ground for
champion-vs-challenger. Feature rows use only data ≤ t.

### Metrics (one code path: `ml/evaluate.py`)

- Anomaly: precision / recall / F1 at the calibrated threshold + **PR-AUC**
  (primary — anomalies are rare).
- Forecast: MAE, RMSE, conditional MAPE (|actual| ≥ 1.0, exclusions counted), and
  **pinball loss** per quantile plus the mean across quantiles (primary); band
  coverage and quantile-crossing rate are measured, not assumed.
- Baselines are mandatory: naive and seasonal-naive for forecasting; a rolling
  z-score detector for anomalies.

---

## 2. Forecast — synthetic walk-forward (corrected implementation)

Command:
```
python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml
```
Report: `ml/reports/forecast_eval.json` (config SHA-256, seed, golden hash, 9 daily
folds). Value in each cell: **mean pinball / MAE / conditional MAPE / Δ mean-pinball vs
the best baseline**.

| series | h=1 | h=5 | h=15 |
|---|---|---|---|
| api_latency_p95_ms | 0.970 / 3.69 / 2.36% / −57.9% | 1.018 / 3.71 / 2.37% / −57.2% | 0.989 / 3.73 / 2.38% / −58.8% |
| cpu_usage_pct | 0.650 / 2.22 / 8.42% / −44.7% | 0.666 / 2.21 / 8.32% / −43.4% | 0.660 / 2.32 / 8.75% / −45.5% |
| request_rate_rps | 10.309 / 39.80 / 4.81% / −0.2% | 10.790 / 41.55 / 5.05% / **+0.1%** | 11.254 / 40.41 / 4.98% / −3.4% |

The quantile model beats the best baseline on 8 of 9 series-horizon pairs; the
exception (request_rate h=5, naive wins by 0.09%) is recorded in the report notes —
the seeded request-rate level shift inside the evaluation region hurts the model there.

> **Correction note:** these numbers supersede the originally logged Phase 2 tables.
> During Phase 3 a warmup-alignment bug was found (feature rows paired with targets 60
> rows stale — effectively predicting the recent past, inflating quality); the fixed
> implementation was re-measured and a label↔position invariant test now guards the
> convention.

## 3. Anomaly — synthetic walk-forward

Command:
```
python -m ml.train --task anomaly --config ml/configs/train_synthetic.yaml
```
Per-fold refits (expanding window) with per-fold best-F1 thresholds calibrated on
training-window labels only. Pooled over 9 daily folds:

| series | precision | recall | F1 | PR-AUC pooled | z-score PR-AUC |
|---|---|---|---|---|---|
| api_latency_p95_ms | 0.074±0.210 | 0.044±0.126 | 0.056 | 0.495 | 0.852 |
| cpu_usage_pct | 0.192±0.322 | 0.185±0.343 | 0.122 | 0.100 | 0.038 |
| request_rate_rps | 0.128±0.309 | 0.512±0.464 | 0.131 | 0.156 | 0.271 |

Findings, recorded in the report notes:
- In stationary regimes the detector is strong (cpu fold 0: PR-AUC 0.984, 143 TP /
  2 FP).
- Two series have a **permanent level shift inside the evaluation region**; the
  post-shift regime is unlabeled and inflates false positives until walk-forward refits
  absorb the new level — the drift effect Phase 4 retrains on. The instantly-adapting
  z-score baseline wins pooled PR-AUC there, which the report explains.
- Latency has only 18 labeled evaluation anomalies — per-fold P/R are high-variance.

## 4. Anomaly — public benchmark sanity check (NAB)

Command:
```
python -m ml.loaders --csv data/raw/nab/ec2_cpu_utilization_53ea38.csv \
    --labels data/raw/nab/labels/combined_windows.json \
    --out data/generated/nab_ec2_cpu_utilization_53ea38.csv
python -m ml.train --task anomaly --config ml/configs/train_public.yaml --report anomaly_public_eval.json
python -m ml.train --task forecast --config ml/configs/train_public.yaml --report forecast_public_eval.json
```
NAB `ec2_cpu_utilization_53ea38` (4,032 points @ 5 min, fractions split 0.6/0.3/0.1,
11 folds): anomaly **PR-AUC pooled 0.430 (Isolation Forest) vs 0.204 (z-score)** — on
this real series the IF wins; flagged 228 of 1,209 evaluation rows (201 labeled).
Forecast: h=1 mean pinball 0.0155 (MAE 0.057, MAPE 3.00%, −49.0% vs seasonal-naive);
h=6 mean pinball 0.0119 (−57.5% vs naive). Sanity check only.

## 5. Drift-and-recovery demonstration

Command (see `docs/IMPLEMENTATION_PLAN.md` Phase 4 for the full Git Bash invocation):
```
python scripts/drift_demo.py --config ml/configs/drift_demo.yaml ...
```
Two passes produce **identical reports** after stripping versions/timestamps
(`ml/reports/drift_demo_report_run{1,2}.json`).

Measured outcome:
- **PSI drift fired on all three series** vs the pre-drift references (worst features:
  cpu `roll_60_mean` = 3.21, request_rate `roll_15_max` = 1.51, latency
  `roll_15_min` = 0.55) — the seeded level shifts (days 15.7 / 16.3) sit between the
  reference window and the recent window.
- Residual monitor (pre-promotion): request_rate `drift`, latency `warn`, cpu `ok`.
- **Forecast challenger PROMOTED**: mean pinball improved **71.05%** on the frozen
  golden set (primary/secondary/sanity rules all passed; both metric sets + rule trace
  recorded in `promotion_decisions`).
- **Pre-drift challenger REJECTED** (primary, secondary, sanity all failed) — a
  challenger that did not learn the new regime cannot replace the champion.
- **Anomaly challenger REJECTED**: golden PR-AUC did not improve (the golden region
  contains very few labeled anomalies; the pre-drift champion's regime-flagging ranks
  the true anomalies comparably). The gate held — recorded as measured.
- **Recovery**: after the forecast promotion the residual monitor reads `ok` on all
  series against the new champion, and the new champion's reference covers the current
  regime.

## 6. Threats to validity & limitations

1. **Synthetic dominance.** The primary evaluation uses generated data with known
   anomaly labels; real-world anomaly structure is richer. One NAB series provides a
   sanity check only.
2. **Few positives in the golden region.** 5 golden days contain very few labeled
   anomalies per series — champion-vs-challenger PR-AUC comparisons there are noisy
   (visible in the anomaly challenger's rejection).
3. **Single node.** All measurements are from a laptop-class Docker deployment; no
   horizontal-scaling or concurrency claims are made (see `docs/BENCHMARKS.md`).
4. **Threshold sensitivity.** Anomaly F1 depends on the calibrated threshold;
   PR-AUC (threshold-free) is reported alongside everywhere.
5. **MAPE exclusions.** MAPE is computed only where |actual| ≥ 1.0; the excluded-row
   count is reported with every MAPE value.
6. **Static-vs-refit evaluation semantics.** Forecast models are refit per
   walk-forward fold; anomaly detectors are refit per fold too, but the *serving*
   champion between promotions is static — the Phase 4 demo shows exactly what that
   implies under drift.

## 7. Reproduction index

| claim | command |
|---|---|
| synthetic forecast metrics | `python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml` |
| synthetic anomaly metrics | `python -m ml.train --task anomaly --config ml/configs/train_synthetic.yaml` |
| NAB sanity runs | `python -m ml.train --task anomaly\|forecast --config ml/configs/train_public.yaml` |
| drift-and-recovery demo | `python scripts/drift_demo.py --config ml/configs/drift_demo.yaml …` (full invocation in `docs/IMPLEMENTATION_PLAN.md` Phase 4) |
| serving benchmarks | `python scripts/bench_query.py --series-id 4 --runs 20` etc. |
| tests | `pytest backend/tests ml/tests` |
