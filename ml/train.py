"""Reproducible offline evaluation entry point (Phase 2).

    python -m ml.train --task anomaly  --config ml/configs/train_synthetic.yaml
    python -m ml.train --task forecast --config ml/configs/train_synthetic.yaml

Pipeline: resolve dataset (deterministic simulator regeneration or standardized CSV)
-> shared trailing features -> timeline split (train / walk-forward eval / frozen golden)
-> per-series evaluation -> metrics JSON in ml/reports/ (config SHA-256, seed, per-fold
and aggregate metrics, baselines, honest model-vs-baseline comparison).

Every number in the report is measured by ml/evaluate.py on the walk-forward evaluation
region. The golden region is never touched: forecast targets that would reach into it
are excluded from the metrics, and it is only hashed (for Phase 4 immutability checks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import ml.evaluate as ev
from ml import tracking
from ml.anomaly import (
    RollingZScoreDetector,
    calibrate_best_f1,
    fit_anomaly_detector,
    percentile_threshold,
)
from ml.backtest import SplitConfig, TimelineSplit, split_timeline, walk_forward_folds
from ml.features import FeatureConfig, build_features, feature_names
from ml.forecast import (
    QuantileForecaster,
    band_crossing_rate,
    fit_quantile_models,
    naive_forecast,
    seasonal_naive_forecast,
)
from ml.models import AnomalyDetectorWrapper, QuantileForecasterWrapper


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def config_sha256(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def golden_hash(frames: dict[str, pd.DataFrame], split: TimelineSplit) -> str:
    digest = hashlib.sha256()
    for name in sorted(frames):
        digest.update(name.encode("utf-8"))
        golden = frames[name]["value"].to_numpy()[split.golden_start :]
        digest.update(np.ascontiguousarray(golden, dtype=np.float64).tobytes())
    return digest.hexdigest()


def resolve_dataset(dataset_cfg: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return {series_name: frame[ts, value, is_anomaly]} plus dataset metadata."""
    if dataset_cfg.get("source") == "simulator":
        from ml.simulator import generate_dataset, load_config

        sim = load_config(dataset_cfg["simulator_config"])
        raw = generate_dataset(sim)
        columns = ["ts", "value", "is_anomaly", "anomaly_type"]
        frames = {
            str(name): group[columns].reset_index(drop=True)
            for name, group in raw.groupby("series_name", sort=True)
        }
        meta = {
            "source": f"simulator:{dataset_cfg['simulator_config']}",
            "points_per_day": int(dataset_cfg.get("points_per_day", 1440)),
            "series": sorted(frames),
            "points": {name: int(len(f)) for name, f in frames.items()},
        }
        return frames, meta
    if dataset_cfg.get("source") == "csv":
        path = Path(dataset_cfg["csv"])
        if not path.exists():
            raise FileNotFoundError(
                f"dataset csv not found at {path}; standardize it first, e.g. "
                "`python -m ml.loaders --csv ... --labels ... --out <path>`"
            )
        raw = pd.read_csv(path)
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        frames = {
            str(name): group[["ts", "value", "is_anomaly", "anomaly_type"]].reset_index(drop=True)
            for name, group in raw.groupby("series_name", sort=True)
        }
        meta = {
            "source": f"csv:{path}",
            "points_per_day": int(dataset_cfg["points_per_day"]),
            "series": sorted(frames),
            "points": {name: int(len(f)) for name, f in frames.items()},
        }
        return frames, meta
    raise ValueError(f"unsupported dataset source: {dataset_cfg!r}")


def _shifted_target(values: np.ndarray, horizon: int) -> np.ndarray:
    y = np.full(len(values), np.nan)
    if horizon < len(values):
        y[:-horizon] = values[horizon:]
    return y


def run_anomaly(
    cfg: dict, frames: dict, meta: dict, feature_cfg: FeatureConfig, split_cfg: SplitConfig
) -> dict:
    """Walk-forward anomaly evaluation: at each fold the detector is refit on all
    (feature) rows before the fold and its threshold is recalibrated by best F1 on the
    fold's training-window labels — historical labels only, the evaluation region is
    never touched. Standardized scores keep thresholds comparable across refits."""
    anomaly_cfg = cfg["anomaly"]
    points_per_day = meta["points_per_day"]
    notes: list[str] = []

    per_series: dict[str, dict] = {}
    for name, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        warmup = feature_cfg.max_lookback
        split = split_timeline(len(frame), points_per_day, split_cfg)
        folds = walk_forward_folds(split, embargo=0, cfg=split_cfg, points_per_day=points_per_day)

        labels = frame["is_anomaly"].to_numpy(dtype=bool)[row_index]
        anomaly_types = frame["anomaly_type"].to_numpy()[row_index]
        X = features.to_numpy()

        zscore = RollingZScoreDetector(window=int(anomaly_cfg.get("zscore_window", 60)))
        # align the full-series z-scores to the feature-row positions (warmup dropped)
        z_aligned = zscore.score(frame["value"]).to_numpy()[row_index]
        z_cal = z_aligned[(row_index >= warmup) & (row_index < split.train_end)]
        z_cal_valid = ~np.isnan(z_cal)
        z_cal_labels = labels[(row_index >= warmup) & (row_index < split.train_end)]
        if z_cal_labels.any() and z_cal_valid.any():
            z_threshold = calibrate_best_f1(z_cal[z_cal_valid], z_cal_labels[z_cal_valid])
        else:
            z_threshold = percentile_threshold(
                z_cal[z_cal_valid], float(anomaly_cfg.get("percentile_fallback", 0.99))
            )

        per_fold = []
        pooled_scores, pooled_labels, pooled_preds, pooled_types, pooled_z = [], [], [], [], []
        for fold in folds:
            fit_mask = (row_index >= warmup) & (row_index < fold.train_end)
            detector, threshold_note = fit_anomaly_detector(
                X[fit_mask],
                X[fit_mask],
                labels[fit_mask],
                n_estimators=int(anomaly_cfg["n_estimators"]),
                contamination=float(anomaly_cfg["contamination"]),
                seed=int(cfg["seed"]),
                calibration="best_f1",
                percentile_fallback=float(anomaly_cfg.get("percentile_fallback", 0.99)),
                n_jobs=int(anomaly_cfg.get("n_jobs", -1)),
            )
            if "fallback" in threshold_note:
                notes.append(
                    f"{name} fold {fold.fold_id}: no positive labels in the training window; "
                    "used percentile threshold fallback"
                )

            eval_mask = (row_index >= fold.eval_start) & (row_index < fold.eval_end)
            y = labels[eval_mask]
            scores = detector.score(X[eval_mask])
            preds = detector.predict_labels(X[eval_mask])
            z = z_aligned[eval_mask]
            z_valid = ~np.isnan(z)
            fold_metrics = ev.precision_recall_f1(y, preds)
            per_fold.append(
                {
                    "fold": fold.fold_id,
                    "eval_rows": [int(fold.eval_start), int(fold.eval_end)],
                    "n": int(eval_mask.sum()),
                    "anomalies": int(y.sum()),
                    "threshold": float(detector.threshold),
                    "threshold_method": threshold_note,
                    "train_rows": int(fit_mask.sum()),
                    "pr_auc": ev.pr_auc(y, scores) if y.any() else None,
                    "zscore": (
                        {
                            **ev.precision_recall_f1(y[z_valid], z[z_valid] >= z_threshold),
                            "pr_auc": ev.pr_auc(y[z_valid], z[z_valid]) if y.any() else None,
                        }
                        if z_valid.any()
                        else {}
                    ),
                    **{
                        k: fold_metrics[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")
                    },
                }
            )
            pooled_scores.append(scores)
            pooled_labels.append(y)
            pooled_preds.append(preds)
            pooled_types.append(anomaly_types[eval_mask])
            pooled_z.append(z)

        pooled_scores = np.concatenate(pooled_scores)
        pooled_labels = np.concatenate(pooled_labels)
        pooled_preds = np.concatenate(pooled_preds)
        pooled_types = np.concatenate(pooled_types)
        pooled_z = np.concatenate(pooled_z)
        z_valid = ~np.isnan(pooled_z)

        recall_by_type = {}
        for anomaly_type in sorted(set(pooled_types[pooled_labels]) - {""}):
            in_type = pooled_types == anomaly_type
            recall_by_type[anomaly_type] = float(
                (pooled_preds & in_type & pooled_labels).sum() / (in_type & pooled_labels).sum()
            )

        model_auc_pooled = (
            ev.pr_auc(pooled_labels, pooled_scores) if pooled_labels.any() else None
        )
        z_pooled = ev.precision_recall_f1(
            pooled_labels[z_valid], pooled_z[z_valid] >= z_threshold
        )
        z_auc_pooled = (
            ev.pr_auc(pooled_labels[z_valid], pooled_z[z_valid]) if pooled_labels.any() else None
        )

        shift_rows = np.flatnonzero(frame["anomaly_type"].to_numpy() == "level_shift")
        if (
            shift_rows.size
            and split.eval_start <= shift_rows.mean() < split.eval_end
            and z_auc_pooled is not None
            and model_auc_pooled is not None
            and z_auc_pooled > model_auc_pooled
        ):
            notes.append(
                f"{name}: the level-shift transition (day "
                f"{shift_rows.mean() / points_per_day:.2f}) lies inside the evaluation region; "
                "post-shift regime points are unlabeled and inflate false positives until "
                "walk-forward refits absorb the new level (see per-fold fp). The rolling "
                "z-score baseline adapts within its window, which explains its pooled PR-AUC "
                "advantage — the drift effect Phase 4 retrains on."
            )
        if pooled_labels.sum() < 30:
            notes.append(
                f"{name}: only {int(pooled_labels.sum())} labeled anomalies in the evaluation "
                "region — per-fold precision/recall are high-variance; PR-AUC is the more "
                "stable indicator"
            )

        per_series[name] = {
            "aggregate": {
                **{
                    k: ev.aggregate_folds([f[k] for f in per_fold])
                    for k in ("precision", "recall", "f1")
                },
                "pr_auc_mean": float(
                    np.mean([f["pr_auc"] for f in per_fold if f["pr_auc"] is not None])
                ),
                "pr_auc_pooled": model_auc_pooled,
                "zscore_pr_auc_pooled": z_auc_pooled,
                "zscore_f1_pooled": z_pooled["f1"] if z_valid.any() else None,
            },
            "pooled_counts": {
                "n": int(len(pooled_labels)),
                "anomalies": int(pooled_labels.sum()),
                "flagged": int(pooled_preds.sum()),
                "recall_by_type": recall_by_type,
            },
            "per_fold": per_fold,
        }

    return {"task": "anomaly", "per_series": per_series, "notes": notes}


def run_forecast(
    cfg: dict, frames: dict, meta: dict, feature_cfg: FeatureConfig, split_cfg: SplitConfig
) -> dict:
    forecast_cfg = cfg["forecast"]
    horizons = tuple(int(h) for h in forecast_cfg["horizons"])
    quantiles = tuple(float(q) for q in forecast_cfg["quantiles"])
    params = dict(forecast_cfg["params"])
    val_fraction = float(forecast_cfg.get("val_fraction", 0.2))
    epsilon = float(cfg.get("mape_epsilon", 1.0))
    season = int(meta["points_per_day"])
    embargo = max(horizons)
    notes: list[str] = []

    per_series: dict[str, dict] = {}
    for name, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        # row_index[p] = the row's ORIGINAL position in the series (warmup offset kept);
        # X rows are positional, so all X selections use positions (flatnonzero of masks)
        # while values/targets are indexed by original position (labels).
        row_index = features.index.to_numpy()
        warmup = feature_cfg.max_lookback
        values = frame["value"].to_numpy(dtype=float)
        split = split_timeline(len(frame), season, split_cfg)
        folds = walk_forward_folds(split, embargo=embargo, cfg=split_cfg, points_per_day=season)
        X_all = features.to_numpy()
        y_aligned = {h: _shifted_target(values, h)[row_index] for h in horizons}

        per_fold = []
        for fold in folds:
            train_positions = np.flatnonzero(
                (row_index >= warmup) & (row_index < fold.train_end)
            )
            val_count = max(1, int(len(train_positions) * val_fraction))
            val_pos, fit_pos = train_positions[-val_count:], train_positions[:-val_count]

            forecaster = QuantileForecaster(horizons=horizons, quantiles=quantiles)
            fold_predictions: dict[int, dict[float, np.ndarray]] = {}
            pred_positions = np.flatnonzero(
                (row_index >= fold.eval_start) & (row_index < fold.eval_end)
            )
            for horizon in horizons:
                models = fit_quantile_models(
                    X_all[fit_pos], y_aligned[horizon][fit_pos],
                    X_all[val_pos], y_aligned[horizon][val_pos],
                    quantiles=quantiles, params=params, seed=int(cfg["seed"]),
                )
                for tau, model in models.items():
                    forecaster.add(horizon, tau, model)
                fold_predictions[horizon] = forecaster.predict(X_all[pred_positions], horizon)

            fold_entry = {"fold": fold.fold_id, "n_best_iterations": {}}
            for horizon in horizons:
                # keep the golden region untouched: targets must end before eval_end
                valid = (row_index[pred_positions] + horizon) < split.eval_end
                y_true = y_aligned[horizon][pred_positions][valid]
                best_iters = {
                    tau: int(forecaster.models[(horizon, tau)].best_iteration_ or 0)
                    for tau in quantiles
                }
                fold_entry["n_best_iterations"][str(horizon)] = best_iters
                preds = {tau: arr[valid] for tau, arr in fold_predictions[horizon].items()}
                fold_entry[f"h{horizon}"] = ev.forecast_metrics(y_true, preds, epsilon)
                fold_entry[f"h{horizon}"]["band_crossing_rate"] = band_crossing_rate(preds)
            per_fold.append(fold_entry)

        # Baselines over the whole evaluation region (no fitting required); aligned to
        # the eval feature rows via their original positions.
        eval_positions = np.flatnonzero(
            (row_index >= split.eval_start) & (row_index < split.eval_end)
        )
        eval_labels = row_index[eval_positions]
        baselines: dict[str, dict] = {"naive": {}, "seasonal_naive": {}}
        for horizon in horizons:
            valid = ((eval_labels + horizon) < split.eval_end) & (
                eval_labels + horizon < len(values)
            )
            y_true = values[eval_labels + horizon][valid]
            naive = naive_forecast(values)[eval_labels][valid]
            seasonal = seasonal_naive_forecast(values, horizon, season)[eval_labels][valid]
            s_valid = ~np.isnan(seasonal)
            baselines["naive"][str(horizon)] = ev.forecast_metrics(
                y_true, {q: naive for q in quantiles}, epsilon
            )
            baselines["seasonal_naive"][str(horizon)] = (
                ev.forecast_metrics(
                    y_true[s_valid], {q: seasonal[s_valid] for q in quantiles}, epsilon
                )
                if s_valid.any()
                else {}
            )

        def mean_pinball(source: dict, horizon: int) -> float:
            entry = source.get(str(horizon), {})
            return float(entry.get("pinball_mean") or math.nan)

        comparison = {}
        for horizon in horizons:
            model_value = float(np.mean([f[f"h{horizon}"]["pinball_mean"] for f in per_fold]))
            naive_value = mean_pinball(baselines["naive"], horizon)
            seasonal_value = mean_pinball(baselines["seasonal_naive"], horizon)
            best_name, best_value = min(
                [("naive", naive_value), ("seasonal_naive", seasonal_value)],
                key=lambda kv: kv[1],
            )
            delta_pct = (model_value - best_value) / best_value * 100.0
            comparison[str(horizon)] = {
                "model_mean_pinball": model_value,
                "best_baseline": best_name,
                "best_baseline_mean_pinball": best_value,
                "model_vs_baseline_pct": delta_pct,
                "model_wins": model_value < best_value,
            }
            if model_value >= best_value:
                notes.append(
                    f"{name} h={horizon}: the {best_name} baseline matches or beats the model on "
                    f"mean pinball ({best_value:.4f} vs {model_value:.4f}) — reported as measured"
                )

        per_series[name] = {
            "aggregate": {
                f"h{horizon}": ev.aggregate_folds([f[f"h{horizon}"] for f in per_fold])
                for horizon in horizons
            },
            "comparison": comparison,
            "baselines": baselines,
            "per_fold": per_fold,
        }

    return {"task": "forecast", "per_series": per_series, "notes": notes}


def train_pooled_forecaster(
    cfg: dict,
    frames: dict,
    meta: dict,
    feature_cfg: FeatureConfig,
    split: TimelineSplit,
    train_end_override: int | None = None,
) -> QuantileForecasterWrapper:
    """Train the pooled (series-agnostic) forecaster on the full initial training
    region of every series — the deployable champion candidate. `train_end_override`
    bounds the training window (e.g. the full timeline for a post-drift challenger)."""
    forecast_cfg = cfg["forecast"]
    horizons = tuple(int(h) for h in forecast_cfg["horizons"])
    quantiles = tuple(float(q) for q in forecast_cfg["quantiles"])
    params = dict(forecast_cfg["params"])
    val_fraction = float(forecast_cfg.get("val_fraction", 0.2))
    embargo = max(horizons)
    warmup = feature_cfg.max_lookback
    seed = int(cfg["seed"])
    effective_end = train_end_override if train_end_override is not None else split.train_end

    fit_X: list[np.ndarray] = []
    val_X: list[np.ndarray] = []
    fit_y = {h: [] for h in horizons}
    val_y = {h: [] for h in horizons}
    for frame in frames.values():
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        values = frame["value"].to_numpy(dtype=float)
        train_mask = (row_index >= warmup) & (row_index < effective_end - embargo)
        train_positions = np.flatnonzero(train_mask)
        val_count = max(1, int(len(train_positions) * val_fraction))
        val_pos, fit_pos = train_positions[-val_count:], train_positions[:-val_count]
        fit_X.append(features.to_numpy()[fit_pos])
        val_X.append(features.to_numpy()[val_pos])
        y_all = {h: _shifted_target(values, h)[row_index] for h in horizons}
        for horizon in horizons:
            fit_y[horizon].append(y_all[horizon][fit_pos])
            val_y[horizon].append(y_all[horizon][val_pos])

    X_fit, X_val = np.vstack(fit_X), np.vstack(val_X)
    forecaster = QuantileForecaster(horizons=horizons, quantiles=quantiles)
    for horizon in horizons:
        models = fit_quantile_models(
            X_fit,
            np.concatenate(fit_y[horizon]),
            X_val,
            np.concatenate(val_y[horizon]),
            quantiles=quantiles,
            params=params,
            seed=seed,
        )
        for tau, model in models.items():
            forecaster.add(horizon, tau, model)
    return QuantileForecasterWrapper(forecaster, feature_names(feature_cfg))


def train_pooled_detector(
    cfg: dict,
    frames: dict,
    meta: dict,
    feature_cfg: FeatureConfig,
    split: TimelineSplit,
    train_end_override: int | None = None,
) -> AnomalyDetectorWrapper:
    """Train the pooled (series-agnostic) Isolation Forest detector with a threshold
    calibrated by best F1 on the pooled training-window labels. `train_end_override`
    bounds the training window (e.g. the full timeline for a post-drift challenger)."""
    anomaly_cfg = cfg["anomaly"]
    warmup = feature_cfg.max_lookback
    effective_end = train_end_override if train_end_override is not None else split.train_end
    X_parts, label_parts = [], []
    for frame in frames.values():
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        mask = (row_index >= warmup) & (row_index < effective_end)
        X_parts.append(features.to_numpy()[mask])
        label_parts.append(frame["is_anomaly"].to_numpy(dtype=bool)[row_index][mask])
    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    detector, _ = fit_anomaly_detector(
        X,
        X,
        labels,
        n_estimators=int(anomaly_cfg["n_estimators"]),
        contamination=float(anomaly_cfg["contamination"]),
        seed=int(cfg["seed"]),
        calibration="best_f1",
        percentile_fallback=float(anomaly_cfg.get("percentile_fallback", 0.99)),
        n_jobs=int(anomaly_cfg.get("n_jobs", -1)),
    )
    return AnomalyDetectorWrapper(detector, feature_names(feature_cfg))


def register_final_model(
    task: str,
    cfg: dict,
    sha: str,
    frames: dict,
    meta: dict,
    feature_cfg: FeatureConfig,
    split: TimelineSplit,
    eval_report: dict | None,
) -> dict:
    """Train the pooled champion candidate and register it under the champion alias."""
    started = time.perf_counter()
    if eval_report and eval_report.get("per_series"):
        metrics = tracking.flatten_metrics(eval_report["per_series"], task)
    else:
        metrics = {}
    if task == "forecast":
        wrapper = train_pooled_forecaster(cfg, frames, meta, feature_cfg, split)
        horizons = tuple(int(h) for h in cfg["forecast"]["horizons"])
        quantiles = tuple(float(q) for q in cfg["forecast"]["quantiles"])
        threshold = None
    else:
        wrapper = train_pooled_detector(cfg, frames, meta, feature_cfg, split)
        horizons, quantiles = (), ()
        threshold = float(wrapper.threshold)

    # drift references: per-series feature distributions over the trailing
    # reference window of the champion's training rows (plan §10); travels with
    # the model as an artifact
    from ml.drift import build_reference_set

    references = build_reference_set(
        frames,
        feature_cfg,
        meta["points_per_day"],
        window_end=split.train_end,
        window_days=int(cfg.get("drift", {}).get("reference_window_days", 7)),
    )

    feature_config = {
        "lags": list(feature_cfg.lags),
        "rolling_windows": list(feature_cfg.rolling_windows),
        "rolling_stats": list(feature_cfg.rolling_stats),
        "roc_offsets": list(feature_cfg.roc_offsets),
        "calendar": feature_cfg.calendar,
    }
    name, version, run_id = tracking.register_champion(
        task,
        wrapper,
        seed=int(cfg["seed"]),
        config_sha256=sha,
        feature_config=feature_config,
        horizons=horizons,
        quantiles=quantiles,
        threshold=threshold,
        metrics=metrics,
        report=eval_report,
        reference=references,
    )
    return {
        "model_name": name,
        "model_version": version,
        "run_id": run_id,
        "alias": tracking.CHAMPION_ALIAS,
        "eval_metrics_logged": len(metrics),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train", description=__doc__)
    parser.add_argument("--task", required=True, choices=["anomaly", "forecast"])
    parser.add_argument("--config", required=True, help="path to the training YAML config")
    parser.add_argument("--report", default=None, help="override the report file name")
    parser.add_argument(
        "--register",
        action="store_true",
        help="after evaluation, train the pooled final model and register it as champion",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    raw_path = Path(args.config)
    cfg = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    sha = config_sha256(cfg)
    feature_cfg = FeatureConfig(
        lags=tuple(cfg["features"]["lags"]),
        rolling_windows=tuple(cfg["features"]["rolling_windows"]),
        rolling_stats=tuple(cfg["features"]["rolling_stats"]),
        roc_offsets=tuple(cfg["features"]["roc_offsets"]),
        calendar=bool(cfg["features"].get("calendar", True)),
    )
    s = cfg["splits"]
    split_cfg = SplitConfig(
        train_days=int(s.get("train_days", 14)),
        eval_days=int(s.get("eval_days", 9)),
        golden_days=int(s.get("golden_days", 5)),
        refit_every_days=int(s.get("refit_every_days", 1)),
        fractions=tuple(s["fractions"]) if s.get("fractions") else None,
        n_folds=int(s.get("n_folds", 10)),
    )

    frames, meta = resolve_dataset(cfg["dataset"])
    reference = next(iter(frames))
    split = split_timeline(len(frames[reference]), meta["points_per_day"], split_cfg)
    embargo = (
        max(int(h) for h in cfg["forecast"]["horizons"]) if args.task == "forecast" else 0
    )
    fold_count = len(
        walk_forward_folds(
            split, embargo=embargo, cfg=split_cfg, points_per_day=meta["points_per_day"]
        )
    )

    print(f"=== PulseGuard offline evaluation — task: {args.task} ===")
    print(f"config sha256: {sha}  seed: {cfg['seed']}")
    print(
        f"dataset: {meta['source']} | {len(frames)} series | "
        f"points {meta['points']} | {meta['points_per_day']}/day"
    )
    print(
        f"split: train [0, {split.train_end})  eval [{split.eval_start}, {split.eval_end})  "
        f"golden [{split.golden_start}, {split.n})  "
        f"golden sha256 {golden_hash(frames, split)[:16]}…"
    )

    if args.task == "anomaly":
        result = run_anomaly(cfg, frames, meta, feature_cfg, split_cfg)
    else:
        result = run_forecast(cfg, frames, meta, feature_cfg, split_cfg)

    report = {
        "task": args.task,
        "config": str(raw_path),
        "config_sha256": sha,
        "seed": int(cfg["seed"]),
        "dataset": meta,
        "split": {
            "mode": split_cfg.mode,
            "train_end": split.train_end,
            "eval_start": split.eval_start,
            "eval_end": split.eval_end,
            "golden_start": split.golden_start,
            "n": split.n,
            "golden_sha256": golden_hash(frames, split),
            "embargo_rows": embargo,
            "folds": fold_count,
        },
        "feature_config": {
            "lags": list(feature_cfg.lags),
            "rolling_windows": list(feature_cfg.rolling_windows),
            "rolling_stats": list(feature_cfg.rolling_stats),
            "roc_offsets": list(feature_cfg.roc_offsets),
            "calendar": feature_cfg.calendar,
            "warmup_rows": feature_cfg.max_lookback,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "per_series": _jsonify(result["per_series"]),
        "notes": result["notes"],
    }

    if args.register:
        print("\nregistering pooled champion candidate...")
        report["registration"] = register_final_model(
            args.task, cfg, sha, frames, meta, feature_cfg, split, report
        )
        reg = report["registration"]
        print(
            f"registered {reg['model_name']} version {reg['model_version']} "
            f"with alias '{reg['alias']}' (run {reg['run_id'][:8]}…, "
            f"{reg['eval_metrics_logged']} eval metrics logged, {reg['elapsed_seconds']}s)"
        )

    report_dir = Path(cfg.get("report_dir", "ml/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (args.report or f"{args.task}_eval.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for name, entry in report["per_series"].items():
        print(f"\nseries {name}")
        if args.task == "anomaly":
            agg = entry["aggregate"]
            z_auc = agg["zscore_pr_auc_pooled"]
            pooled = entry["pooled_counts"]
            print(
                f"  precision {agg['precision']['mean']:.3f}±{agg['precision']['std']:.3f} | "
                f"recall {agg['recall']['mean']:.3f}±{agg['recall']['std']:.3f} | "
                f"f1 {agg['f1']['mean']:.3f}±{agg['f1']['std']:.3f} | "
                f"PR-AUC pooled {agg['pr_auc_pooled']:.3f}"
                + (f" (z-score baseline {z_auc:.3f})" if z_auc is not None else "")
            )
            print(
                f"  flagged {pooled['flagged']} of {pooled['n']} points "
                f"({pooled['anomalies']} labeled anomalies) | recall by type: "
                + ", ".join(f"{t}={r:.2f}" for t, r in pooled["recall_by_type"].items())
            )
        else:
            for horizon, agg in entry["aggregate"].items():
                mape = agg.get("mape", {}).get("mean")
                mape_txt = f"{mape * 100:.2f}%" if mape is not None else "n/a"
                comp = entry["comparison"][horizon.lstrip("h")]
                standing = "model wins" if comp["model_wins"] else "MODEL LOSES"
                print(
                    f"  {horizon}: mean pinball {agg['pinball_mean']['mean']:.4f} | "
                    f"mae {agg['mae']['mean']:.4f} | rmse {agg['rmse']['mean']:.4f} | "
                    f"mape {mape_txt} | vs {comp['best_baseline']}: "
                    f"{comp['model_vs_baseline_pct']:+.2f}% ({standing})"
                )
    for note in report["notes"]:
        print(f"note: {note}")
    print(f"\nreport written: {report_path} (elapsed {report['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
