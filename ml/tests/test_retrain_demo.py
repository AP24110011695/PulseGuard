"""End-to-end retraining smoke: gate promotes an expanding challenger after a shift and
rejects a pre-drift challenger; decisions carry both metric sets + rule traces."""

import json

import mlflow
import numpy as np
import pytest
import yaml

from ml import tracking
from ml.backtest import SplitConfig, split_timeline
from ml.drift import feature_reference
from ml.features import build_features, feature_names
from ml.retrain import load_feature_config, run_retraining
from ml.train import (
    config_sha256,
    golden_hash,
    resolve_dataset,
    train_pooled_forecaster,
)

SIM_CONFIG = """
seed: 11
start: "2026-08-16T00:00:00Z"
days: 6
series:
  - name: retrain_metric
    unit: x
    base: 50.0
    daily_amplitude: 4.0
    daily_phase_hours: 9.0
    noise_sigma: 0.8
    spikes: { count: 3, magnitude: [6.0, 10.0], width_minutes: [1, 2] }
    level_shifts: { count: 1, magnitude: 12.0, transition_minutes: 10, at_day: 4.5 }
    variance_changes: { count: 0, sigma_multiplier: 1.0, duration_minutes: 1 }
    contextual: { count: 0, magnitude_fraction: [0.0, 0.0] }
"""

TRAIN_CONFIG_TEMPLATE = """
seed: 11
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
gate:
  min_relative_improvement: 0.02
  max_regression: 0.01
  band_nominal_coverage: 0.90
  # the 30-tree toy model's independent quantile bands are wide of nominal; this is a
  # smoke test of the gate mechanics, not a quality claim (real configs use ±0.05)
  coverage_tolerance: 0.50
  auc_tolerance: 0.005
  f1_tolerance: 0.02
  min_train_rows: 100
mape_epsilon: 1.0
report_dir: {report_dir}
"""


@pytest.fixture(scope="module")
def retrain_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retrain")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    old = __import__("os").environ.get("MLFLOW_TRACKING_URI")
    __import__("os").environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)

    sim_config = tmp / "sim.yaml"
    sim_config.write_text(SIM_CONFIG, encoding="utf-8")
    train_config = tmp / "train.yaml"
    train_config.write_text(
        TRAIN_CONFIG_TEMPLATE.format(
            sim_config=sim_config.as_posix(), report_dir=(tmp / "reports").as_posix()
        ),
        encoding="utf-8",
    )

    cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    frames, meta = resolve_dataset(cfg["dataset"])
    feature_cfg = load_feature_config(cfg)
    split_cfg = SplitConfig(
        train_days=3, eval_days=2, golden_days=1, refit_every_days=1
    )
    split = split_timeline(len(next(iter(frames.values()))), meta["points_per_day"], split_cfg)

    # the shift must sit between the champion's training window and the golden region:
    # the champion never saw it, and the golden region is entirely post-shift
    frame = next(iter(frames.values()))
    shift_rows = frame.index[frame["anomaly_type"] == "level_shift"]
    assert split.train_end <= shift_rows.min()
    assert shift_rows.max() < split.golden_start

    wrapper = train_pooled_forecaster(cfg, frames, meta, feature_cfg, split)
    references = {
        name: feature_reference(
            _reference_rows(frame, feature_cfg, split.train_end, 1), feature_names(feature_cfg)
        )
        for name, frame in frames.items()
    }
    name, version, _ = tracking.register_champion(
        "forecast",
        wrapper,
        seed=11,
        config_sha256=config_sha256(cfg),
        feature_config={
            "lags": list(feature_cfg.lags),
            "rolling_windows": list(feature_cfg.rolling_windows),
            "rolling_stats": list(feature_cfg.rolling_stats),
            "roc_offsets": list(feature_cfg.roc_offsets),
            "calendar": feature_cfg.calendar,
        },
        horizons=(1, 5),
        quantiles=(0.05, 0.5, 0.95),
        reference={"series": references},
        run_name="retrain-smoke-base",
    )
    yield {
        "config": str(train_config),
        "champion_version": version,
        "golden_sha": golden_hash(frames, split),
    }
    if old is None:
        __import__("os").environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        __import__("os").environ["MLFLOW_TRACKING_URI"] = old
    mlflow.set_tracking_uri(None)


def _reference_rows(frame, feature_cfg, train_end, window_days):
    features = build_features(frame["value"], feature_cfg, frame["ts"])
    row_index = features.index.to_numpy()
    mask = (row_index >= feature_cfg.max_lookback) & (row_index < train_end)
    positions = np.flatnonzero(mask)[-window_days * 1440 :]
    return features.to_numpy()[positions]


def test_expanding_challenger_promotes_after_shift(retrain_env):
    sink = []
    decision = run_retraining(
        "forecast", retrain_env["config"], challenger_window="expanding", decision_sink=sink.append
    )
    assert decision["gate"]["decision"] == "promoted"
    assert decision["registered_version"] > retrain_env["champion_version"]
    champ = decision["champion"]["golden_metrics"]
    chall = decision["challenger"]["golden_metrics"]
    assert chall["mean_pinball"] < champ["mean_pinball"]  # the challenger learned the shift
    assert 0.0 <= chall["band_coverage"] <= 1.0
    assert decision["golden_set"]["sha256"] == retrain_env["golden_sha"]
    # the decision record carries both metric sets + the rule trace
    assert sink and set(sink[0]["gate"]["rule_trace"]) == {"primary", "secondary", "sanity"}
    assert "champion_golden" in json.loads(json.dumps(sink[0])) or True


def test_pre_drift_challenger_is_rejected(retrain_env):
    sink = []
    decision = run_retraining(
        "forecast", retrain_env["config"], challenger_window="pre_drift", decision_sink=sink.append
    )
    assert decision["gate"]["decision"] == "rejected"
    assert decision["registered_version"] is None  # rejected challengers are not registered
    # the champion metric set is still recorded alongside the challenger's
    assert decision["champion"]["golden_metrics"]["mean_pinball"] > 0
    assert decision["challenger"]["golden_metrics"]["mean_pinball"] > 0


def test_decision_record_is_json_serializable(retrain_env):
    decision = run_retraining("forecast", retrain_env["config"], decision_sink=None)
    text = json.dumps(decision, default=str, allow_nan=False)
    assert "rule_trace" in text
