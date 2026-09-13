"""End-to-end smoke test: the full evaluation pipeline runs from config to report."""

import json

import pytest

from ml.train import main

SIM_CONFIG = """
seed: 7
start: "2026-08-16T00:00:00Z"
days: 6
series:
  - name: smoke_metric
    unit: x
    base: 50.0
    daily_amplitude: 5.0
    daily_phase_hours: 9.0
    noise_sigma: 1.0
    spikes: { count: 6, magnitude: [10.0, 20.0], width_minutes: [1, 3] }
    level_shifts: { count: 1, magnitude: 5.0, transition_minutes: 5 }
    variance_changes: { count: 1, sigma_multiplier: 3.0, duration_minutes: 30 }
    contextual: { count: 3, magnitude_fraction: [0.6, 0.9] }
"""

TRAIN_CONFIG_TEMPLATE = """
seed: 7
dataset:
  source: simulator
  simulator_config: {sim_config}
  points_per_day: 1440
splits:
  mode: days
  train_days: 3
  eval_days: 2
  golden_days: 1
  refit_every_days: 1
features:
  lags: [1, 2, 3, 5]
  rolling_windows: [5, 10]
  rolling_stats: [mean, std]
  roc_offsets: [1]
  calendar: true
anomaly:
  n_estimators: 50
  contamination: 0.01
  calibration_days: 1
  calibration: best_f1
  percentile_fallback: 0.99
  zscore_window: 10
forecast:
  horizons: [1, 5]
  quantiles: [0.05, 0.5, 0.95]
  val_fraction: 0.2
  params:
    learning_rate: 0.1
    num_leaves: 15
    n_estimators: 30
    min_data_in_leaf: 10
    early_stopping_rounds: 10
mape_epsilon: 1.0
report_dir: {report_dir}
"""


@pytest.fixture(scope="module")
def run_outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train_smoke")
    sim_config = tmp / "sim_smoke.yaml"
    sim_config.write_text(SIM_CONFIG, encoding="utf-8")
    report_dir = tmp / "reports"
    train_config = tmp / "train_smoke.yaml"
    train_config.write_text(
        TRAIN_CONFIG_TEMPLATE.format(
            sim_config=sim_config.as_posix(), report_dir=report_dir.as_posix()
        ),
        encoding="utf-8",
    )
    results = {}
    for task in ("anomaly", "forecast"):
        assert main(["--task", task, "--config", str(train_config)]) == 0
        results[task] = json.loads((report_dir / f"{task}_eval.json").read_text(encoding="utf-8"))
    return results


def test_reports_carry_provenance(run_outputs):
    for task, report in run_outputs.items():
        assert report["task"] == task
        assert len(report["config_sha256"]) == 64
        assert report["seed"] == 7
        assert report["split"]["golden_start"] == 5 * 1440
        assert len(report["split"]["golden_sha256"]) == 64
        assert report["split"]["folds"] == 2
        assert report["elapsed_seconds"] >= 0


def test_anomaly_report_has_measured_metrics(run_outputs):
    entry = run_outputs["anomaly"]["per_series"]["smoke_metric"]
    agg = entry["aggregate"]
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= agg[key]["mean"] <= 1.0
        assert 0.0 <= agg["f1"]["std"] <= 1.0
    assert 0.0 <= agg["pr_auc_pooled"] <= 1.0
    assert 0.0 <= agg["zscore_pr_auc_pooled"] <= 1.0
    assert len(entry["per_fold"]) == 2
    assert entry["pooled_counts"]["n"] == 2 * 1440
    # one command ran the whole thing; every fold's threshold came from somewhere explicit
    assert all(fold["threshold_method"] for fold in entry["per_fold"])


def test_forecast_report_has_measured_metrics(run_outputs):
    entry = run_outputs["forecast"]["per_series"]["smoke_metric"]
    for horizon in ("h1", "h5"):
        agg = entry["aggregate"][horizon]
        assert agg["pinball_mean"]["mean"] >= 0.0
        assert agg["mae"]["mean"] >= 0.0
        assert agg["rmse"]["mean"] >= agg["mae"]["mean"] - 1e-9  # RMSE >= MAE always
        assert 0.0 <= agg["band_crossing_rate"]["mean"] < 0.5
        comp = entry["comparison"][horizon.lstrip("h")]
        assert comp["best_baseline"] in ("naive", "seasonal_naive")
        assert isinstance(comp["model_wins"], bool)
    assert entry["baselines"]["naive"]["1"]["mae"] >= 0.0


def test_reports_are_valid_json_without_nan(run_outputs):
    for report in run_outputs.values():
        text = json.dumps(report, allow_nan=False)  # raises if NaN slipped in
        assert "NaN" not in text
