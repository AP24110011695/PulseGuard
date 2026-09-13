"""Controlled retraining: challenger training, frozen-golden-set evaluation, measured
promotion gate (plan §11).

Flow: load the current champion (MLflow) → evaluate champion AND challenger on the
frozen golden region through the same code path → apply the gate → promote (register +
move the champion alias) or reject → record the decision (both metric sets + rule trace)
via the decision sink. Nothing runs automatically: callers are the CLI, the demo script,
or the API trigger endpoint.

Challenger training windows:
- "expanding": everything before the series end (minus the target embargo) — includes
  post-drift data, the standard recovery path;
- "pre_drift": the champion's original training window — the rejection demonstration
  (a challenger that did not learn the new regime cannot beat the champion).

Feature-row convention (PULSEGUARD.md §3.3): feature frames keep original-position
labels; X selections use boolean masks / true positions, targets are label-indexed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

import ml.evaluate as ev
from ml import tracking
from ml.backtest import SplitConfig, split_timeline
from ml.drift import build_reference_set, evaluate_gate
from ml.features import FeatureConfig, build_features, feature_names
from ml.forecast import band_crossing_rate
from ml.train import (
    config_sha256,
    golden_hash,
    resolve_dataset,
    train_pooled_detector,
    train_pooled_forecaster,
)


def load_feature_config(cfg: dict) -> FeatureConfig:
    f = cfg["features"]
    return FeatureConfig(
        lags=tuple(f["lags"]),
        rolling_windows=tuple(f["rolling_windows"]),
        rolling_stats=tuple(f["rolling_stats"]),
        roc_offsets=tuple(f["roc_offsets"]),
        calendar=bool(f.get("calendar", True)),
    )


def feature_config_dict(cfg: FeatureConfig) -> dict:
    return {
        "lags": list(cfg.lags),
        "rolling_windows": list(cfg.rolling_windows),
        "rolling_stats": list(cfg.rolling_stats),
        "roc_offsets": list(cfg.roc_offsets),
        "calendar": cfg.calendar,
    }


def load_champion(task: str) -> dict:
    """Resolve + load the current champion; returns name/version/predict/meta."""
    name, version = tracking.get_champion(task)
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    mv = tracking.client().get_model_version(name, str(version))
    run_params = {}
    if mv.run_id:
        run_params = tracking.client().get_run(mv.run_id).data.params
    feature_config = (
        json.loads(run_params["feature_config"]) if run_params.get("feature_config") else {}
    )
    horizons = (
        tuple(int(h) for h in json.loads(run_params["horizons"]))
        if run_params.get("horizons")
        else ()
    )
    quantiles = (
        tuple(float(q) for q in json.loads(run_params["quantiles"]))
        if run_params.get("quantiles")
        else ()
    )
    threshold = float(run_params["threshold"]) if run_params.get("threshold") else None
    return {
        "name": name,
        "version": version,
        "model": model,
        "feature_config": feature_config,
        "horizons": horizons,
        "quantiles": quantiles,
        "threshold": threshold,
    }


def evaluate_forecaster_on_golden(
    predict,
    frames: dict[str, pd.DataFrame],
    feature_cfg: FeatureConfig,
    golden_start: int,
    horizons: tuple[int, ...],
    quantiles: tuple[float, ...],
    epsilon: float,
) -> dict:
    """Evaluate a forecaster on the frozen golden region (static model, no refits).

    Features at row t use data <= t (legitimate serving knowledge); targets are
    v[t+h] with both row and target inside the golden region. Returns pooled
    per-horizon metrics plus the headline scalars used by the gate.
    """
    per_series: dict[str, dict] = {}
    all_pinball: dict[int, list[float]] = {h: [] for h in horizons}
    all_mae: dict[int, list[float]] = {h: [] for h in horizons}
    coverage_values: list[float] = []

    for name, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        values = frame["value"].to_numpy(dtype=float)
        eval_positions = np.flatnonzero(row_index >= golden_start)
        series_metrics: dict[str, dict] = {}
        for horizon in horizons:
            y_all = np.full(len(values), np.nan)
            y_all[:-horizon] = values[horizon:]
            y_true = y_all[row_index[eval_positions]]
            valid = np.isfinite(y_true) & (row_index[eval_positions] + horizon < len(values))
            y_true = y_true[valid]
            rows = features.iloc[eval_positions][valid]
            raw = predict(rows, horizon)
            preds = {q: raw[q] for q in quantiles}
            metrics = ev.forecast_metrics(y_true, preds, epsilon)
            low, high = preds[min(quantiles)], preds[max(quantiles)]
            metrics["band_coverage"] = float(np.mean((y_true >= low) & (y_true <= high)))
            metrics["band_crossing_rate"] = band_crossing_rate({q: preds[q] for q in quantiles})
            series_metrics[str(horizon)] = metrics
            all_pinball[horizon].append(metrics["pinball_mean"])
            all_mae[horizon].append(metrics["mae"])
            coverage_values.append(metrics["band_coverage"])
        per_series[name] = series_metrics

    per_horizon = {
        str(h): {
            "pinball_mean": float(np.mean(all_pinball[h])),
            "mae": float(np.mean(all_mae[h])),
        }
        for h in horizons
    }
    return {
        "per_series": per_series,
        "per_horizon": per_horizon,
        "mean_pinball": float(np.mean([per_horizon[str(h)]["pinball_mean"] for h in horizons])),
        "mae": float(np.mean([per_horizon[str(h)]["mae"] for h in horizons])),
        "band_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
    }


def evaluate_detector_on_golden(
    predict,
    frames: dict[str, pd.DataFrame],
    feature_cfg: FeatureConfig,
    golden_start: int,
) -> dict:
    """Score the frozen golden region with a static detector (its own threshold)."""
    pooled_labels, pooled_scores, pooled_preds = [], [], []
    for _, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        eval_positions = np.flatnonzero(row_index >= golden_start)
        X = features.to_numpy()[eval_positions]
        labels = frame["is_anomaly"].to_numpy(dtype=bool)[row_index][eval_positions]
        prediction = predict(X)
        pooled_labels.append(labels)
        pooled_scores.append(prediction["score"].to_numpy())
        pooled_preds.append(prediction["is_anomaly"].to_numpy())
    labels = np.concatenate(pooled_labels)
    scores = np.concatenate(pooled_scores)
    preds = np.concatenate(pooled_preds)
    metrics = ev.precision_recall_f1(labels, preds)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": ev.pr_auc(labels, scores) if labels.any() else 0.0,
        "n": int(len(labels)),
        "anomalies": int(labels.sum()),
    }


def train_challenger(
    task: str,
    cfg: dict,
    frames: dict,
    meta: dict,
    feature_cfg: FeatureConfig,
    split,
    window: str,
) -> tuple[object, int]:
    """Train the pooled challenger on the configured window; returns (wrapper, rows)."""
    n = max(len(f) for f in frames.values())
    if window == "expanding":
        train_end = n
    elif window == "pre_drift":
        train_end = split.train_end
    else:
        raise ValueError(f"unsupported challenger window {window!r}")
    warmup = feature_cfg.max_lookback
    row_count = 0
    for frame in frames.values():
        row_index = build_features(frame["value"], feature_cfg, frame["ts"]).index.to_numpy()
        row_count += int(((row_index >= warmup) & (row_index < train_end)).sum())
    if task == "forecast":
        wrapper = train_pooled_forecaster(
            cfg, frames, meta, feature_cfg, split, train_end_override=train_end
        )
    else:
        wrapper = train_pooled_detector(
            cfg, frames, meta, feature_cfg, split, train_end_override=train_end
        )
    return wrapper, row_count


def run_retraining(
    task: str,
    config_path: str | Path,
    challenger_window: str = "expanding",
    decision_sink=None,
) -> dict:
    """Full controlled-retraining cycle for one task; returns the decision record."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sha = config_sha256(cfg)
    frames, meta = resolve_dataset(cfg["dataset"])
    feature_cfg = load_feature_config(cfg)
    s = cfg["splits"]
    split_cfg = SplitConfig(
        train_days=int(s.get("train_days", 14)),
        eval_days=int(s.get("eval_days", 9)),
        golden_days=int(s.get("golden_days", 5)),
        refit_every_days=int(s.get("refit_every_days", 1)),
        fractions=tuple(s["fractions"]) if s.get("fractions") else None,
        n_folds=int(s.get("n_folds", 10)),
    )
    reference_frame = next(iter(frames))
    split = split_timeline(len(frames[reference_frame]), meta["points_per_day"], split_cfg)
    gate_cfg = cfg.get("gate", {})
    epsilon = float(cfg.get("mape_epsilon", 1.0))

    champion = load_champion(task)
    champ_feature_cfg = FeatureConfig(**champion["feature_config"])

    if task == "forecast":
        quantiles = champion["quantiles"]

        def champ_predict(rows: pd.DataFrame, horizon: int) -> dict[float, np.ndarray]:
            frame = rows.copy()
            frame["horizon"] = horizon
            raw = champion["model"].predict(frame)
            return {q: raw[f"q{q}"].to_numpy() for q in quantiles}

        champ_metrics = evaluate_forecaster_on_golden(
            champ_predict,
            frames,
            champ_feature_cfg,
            split.golden_start,
            champion["horizons"],
            champion["quantiles"],
            epsilon,
        )
        challenger, challenger_rows = train_challenger(
            task, cfg, frames, meta, feature_cfg, split, challenger_window
        )
        horizons = tuple(int(h) for h in cfg["forecast"]["horizons"])
        chall_quantiles = tuple(float(q) for q in cfg["forecast"]["quantiles"])

        def chall_predict(rows: pd.DataFrame, horizon: int) -> dict[float, np.ndarray]:
            frame = rows.copy()
            frame["horizon"] = horizon
            raw = challenger.predict(None, frame)
            return {q: raw[f"q{q}"].to_numpy() for q in chall_quantiles}

        chall_metrics = evaluate_forecaster_on_golden(
            chall_predict,
            frames,
            feature_cfg,
            split.golden_start,
            horizons,
            chall_quantiles,
            epsilon,
        )
    else:
        champ_names = feature_names(FeatureConfig(**champion["feature_config"]))

        def champ_predict(X: np.ndarray) -> dict:
            return champion["model"].predict(pd.DataFrame(X, columns=champ_names))

        champ_metrics = evaluate_detector_on_golden(
            champ_predict, frames, champ_feature_cfg, split.golden_start
        )
        challenger, challenger_rows = train_challenger(
            task, cfg, frames, meta, feature_cfg, split, challenger_window
        )
        chall_names = feature_names(feature_cfg)

        def chall_predict(X: np.ndarray) -> dict:
            return challenger.predict(None, pd.DataFrame(X, columns=chall_names))

        chall_metrics = evaluate_detector_on_golden(
            chall_predict, frames, feature_cfg, split.golden_start
        )

    gate = evaluate_gate(task, champ_metrics, chall_metrics, gate_cfg, challenger_rows)
    decision = {
        "task": task,
        "champion": {
            "name": champion["name"],
            "version": champion["version"],
            "golden_metrics": champ_metrics,
        },
        "challenger": {
            "golden_metrics": chall_metrics,
            "train_rows": challenger_rows,
            "window": challenger_window,
        },
        "golden_set": {
            "start_row": split.golden_start,
            "n": split.n,
            "sha256": golden_hash(frames, split),
        },
        "config_sha256": sha,
        "gate": {"decision": gate.decision, "reason": gate.reason, "rule_trace": gate.rule_trace},
        "registered_version": None,
    }

    if gate.decision == "promoted":
        horizons = (
            tuple(int(h) for h in cfg["forecast"]["horizons"]) if task == "forecast" else ()
        )
        quantiles = (
            tuple(float(q) for q in cfg["forecast"]["quantiles"]) if task == "forecast" else ()
        )
        n = max(len(f) for f in frames.values())
        references = build_reference_set(
            frames,
            feature_cfg,
            meta["points_per_day"],
            window_end=(n if challenger_window == "expanding" else split.train_end),
            window_days=int(cfg.get("drift", {}).get("reference_window_days", 7)),
        )
        name, version, _ = tracking.register_champion(
            task,
            challenger,
            seed=int(cfg["seed"]),
            config_sha256=sha,
            feature_config=feature_config_dict(feature_cfg),
            horizons=horizons,
            quantiles=quantiles,
            threshold=float(challenger.threshold) if task == "anomaly" else None,
            reference={"series": references},
            run_name=f"{task}-challenger-promoted",
        )
        decision["registered_version"] = version
        decision["challenger"]["name"] = name

    if decision_sink is not None:
        decision_sink(decision)
    return decision


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ml.retrain", description=__doc__)
    parser.add_argument("--task", required=True, choices=["forecast", "anomaly"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--challenger-window",
        default="expanding",
        choices=["expanding", "pre_drift"],
        help="expanding = include all recent data; pre_drift = rejection demonstration",
    )
    args = parser.parse_args(argv)
    decision = run_retraining(args.task, args.config, challenger_window=args.challenger_window)
    print(json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
