"""Reproducible drift-and-recovery demonstration (Phase 4).

Run inside the compose network so the DB, MLflow and the API are reachable:

    docker compose run --rm -v "$(pwd)/scripts:/code/scripts" api \
        python scripts/drift_demo.py --config ml/configs/drift_demo.yaml \
        --api-url http://api:8000

Scenario per task (forecast + anomaly), fully seeded:
1. BASE champion registered from the pre-drift training window (deterministic reset,
   so repeated runs produce identical decisions).
2. PSI drift check of the recent window vs the base champion's stored reference —
   fires on the series whose seeded level shift lies inside the evaluation region.
3. Historical serving simulation (stored forecasts at past cutoffs) + residual
   monitor — fires on the shifted series.
4. Challenger retrained on the expanding window (includes post-shift data) →
   champion vs challenger evaluated on the frozen golden set → gate → PROMOTED.
5. Recovery: PSI re-check + residual re-check against the new champion → ok.
6. Rejection demonstration: a challenger trained on the pre-drift window cannot
   improve on the golden set → gate REJECTS, champion unchanged.

Writes ml/reports/drift_demo_report.json. Re-running produces identical decisions and
metrics (version numbers, timestamps and elapsed times are volatile by nature).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml import tracking
from ml.backtest import SplitConfig, split_timeline
from ml.drift import build_reference_set
from ml.features import FeatureConfig, build_features, feature_names
from ml.retrain import feature_config_dict, load_feature_config, run_retraining
from ml.train import (
    config_sha256,
    golden_hash,
    resolve_dataset,
    train_pooled_detector,
    train_pooled_forecaster,
)


def _quantile_cols(quantiles: tuple[float, ...], raw: pd.DataFrame) -> dict[str, float]:
    return {str(q): float(raw[f"q{q}"].iloc[0]) for q in quantiles}


def main(argv: list[str] | None = None) -> int:
    from backend.app.core.database import SessionLocal
    from backend.app.services.drift import (
        record_promotion_decision,
        run_psi_check,
        run_residual_check,
        store_residual_baseline,
    )

    parser = argparse.ArgumentParser(prog="scripts/drift_demo.py", description=__doc__)
    parser.add_argument("--config", default="ml/configs/drift_demo.yaml")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", default="ml/reports/drift_demo_report.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
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
    drift_cfg = cfg.get("drift", {})
    psi_bins = int(drift_cfg.get("psi_bins", 10))
    reference_window_days = int(drift_cfg.get("reference_window_days", 7))
    residual_cfg = drift_cfg.get("residual", {})

    print("=== PulseGuard drift-and-recovery demo ===")
    print(f"config sha256: {sha} | golden rows [{split.golden_start}, {split.n}) | "
          f"golden sha256 {golden_hash(frames, split)[:16]}…")

    # ---- step 0: ingest the dataset through the API (idempotent) -----------------
    from ml.api_client import ingest_dataframe

    meta_info = {
        name: {"unit": "demo", "description": f"drift-demo series {name}", "source": "simulator"}
        for name in frames
    }
    ingest = ingest_dataframe(args.api_url, df=_to_ingest_frame(frames), series_meta=meta_info)
    series_ids = {name: entry["series_id"] for name, entry in ingest["series"].items()}
    print(f"ingested/verified {ingest['inserted']} new + {ingest['duplicates']} duplicate points")

    report: dict = {
        "config_sha256": sha,
        "seed": int(cfg["seed"]),
        "golden_set": {
            "start_row": split.golden_start,
            "n": split.n,
            "sha256": golden_hash(frames, split),
        },
        "series_ids": series_ids,
        "tasks": {},
    }

    for task in ("forecast", "anomaly"):
        print(f"\n--- task: {task} ---")
        task_report: dict = {"base": {}, "drift_checks": [], "recovery_checks": [], "decisions": []}

        # ---- step 1: deterministic pre-drift base champion -----------------------
        base_wrapper = (
            train_pooled_forecaster(cfg, frames, meta, feature_cfg, split)
            if task == "forecast"
            else train_pooled_detector(cfg, frames, meta, feature_cfg, split)
        )
        references = build_reference_set(
            frames,
            feature_cfg,
            meta["points_per_day"],
            window_end=split.train_end,
            window_days=reference_window_days,
            n_bins=psi_bins,
        )
        horizons = (
            tuple(int(h) for h in cfg["forecast"]["horizons"]) if task == "forecast" else ()
        )
        quantiles = (
            tuple(float(q) for q in cfg["forecast"]["quantiles"]) if task == "forecast" else ()
        )
        base_name, base_version, _ = tracking.register_champion(
            task,
            base_wrapper,
            seed=int(cfg["seed"]),
            config_sha256=sha,
            feature_config=feature_config_dict(feature_cfg),
            horizons=horizons,
            quantiles=quantiles,
            threshold=float(base_wrapper.threshold) if task == "anomaly" else None,
            reference={"series": references},
            run_name=f"{task}-demo-base",
        )
        task_report["base"] = {"name": base_name, "version": base_version}
        print(f"base champion: {base_name} v{base_version} (pre-drift window)")

        db = SessionLocal()
        try:
            # ---- step 2: PSI drift check against the base reference --------------
            window_points = int(drift_cfg.get("window_points", 2880))
            for series_id in sorted(series_ids.values()):
                check = run_psi_check(db, series_id, task, window_points)
                task_report["drift_checks"].append(check)
                print(
                    f"psi check series {series_id}: {check['status']} "
                    f"(worst {check['worst_feature']}={check['worst_psi']:.3f})"
                )

            # ---- step 3+4: historical serving + residual check (forecast only) ---
            if task == "forecast":
                step = int(residual_cfg.get("serving_step_minutes", 360))
                horizon = int(horizons[0])
                cutoffs = list(
                    range(split.train_end + 1440, split.n - horizon, step)
                )
                for series_name, series_id in sorted(series_ids.items()):
                    store_residual_baseline(
                        db,
                        series_id,
                        task,
                        base_name,
                        base_version,
                        _training_tail_baseline(
                            base_wrapper, frames[series_name], feature_cfg, split, cfg
                        ),
                    )
                    _persist_simulated_forecasts(
                        db, series_id, frames[series_name], base_wrapper, feature_cfg,
                        quantiles, cutoffs, horizon, base_version, base_name,
                    )
                    residual = run_residual_check(
                        db, series_id, task,
                        window=int(residual_cfg.get("window", 12)),
                        multiplier=float(residual_cfg.get("multiplier", 1.5)),
                        sustained=int(residual_cfg.get("sustained", 3)),
                    )
                    task_report["drift_checks"].append(residual)
                    print(f"residual check series {series_id} (base): {residual['status']}")

            # ---- step 5: challenger retraining → gate → promote ------------------
            def promoted_sink(decision: dict, _db=db) -> None:
                record_promotion_decision(_db, decision)

            promotion = run_retraining(
                task, args.config, challenger_window="expanding", decision_sink=promoted_sink
            )
            task_report["decisions"].append(promotion)
            print(
                f"expanding challenger → {promotion['gate']['decision'].upper()}: "
                f"{promotion['gate']['reason']}"
            )

            # ---- step 6: recovery evidence against the new champion --------------
            for series_id in sorted(series_ids.values()):
                check = run_psi_check(db, series_id, task, window_points)
                task_report["recovery_checks"].append(check)
                print(f"psi recovery series {series_id}: {check['status']}")
            if task == "forecast":
                new_version = promotion.get("registered_version")
                new_wrapper = train_pooled_forecaster(
                    cfg, frames, meta, feature_cfg, split, train_end_override=split.n
                )
                for series_name, series_id in sorted(series_ids.items()):
                    store_residual_baseline(
                        db, series_id, task, base_name, new_version,
                        _training_tail_baseline(
                            new_wrapper,
                            frames[series_name],
                            feature_cfg,
                            split,
                            cfg,
                            tail_of=split.n,
                        ),
                    )
                    _persist_simulated_forecasts(
                        db, series_id, frames[series_name], new_wrapper, feature_cfg,
                        quantiles, cutoffs, horizon, new_version, base_name,
                    )
                    residual = run_residual_check(
                        db, series_id, task,
                        window=int(residual_cfg.get("window", 12)),
                        multiplier=float(residual_cfg.get("multiplier", 1.5)),
                        sustained=int(residual_cfg.get("sustained", 3)),
                    )
                    task_report["recovery_checks"].append(residual)
                    print(
                        f"residual recovery series {series_id} (new champion): {residual['status']}"
                    )

            # ---- step 7: rejection demonstration ---------------------------------
            rejection = run_retraining(
                task, args.config, challenger_window="pre_drift", decision_sink=promoted_sink
            )
            task_report["decisions"].append(rejection)
            print(
                f"pre-drift challenger → {rejection['gate']['decision'].upper()}: "
                f"{rejection['gate']['reason']}"
            )
        finally:
            db.close()
        import gc

        gc.collect()
        report["tasks"][task] = task_report

    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nreport written: {report_path} (elapsed {report['elapsed_seconds']}s)")
    return 0


def _to_ingest_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        part = frame[["ts", "value", "is_anomaly"]].copy()
        part.insert(0, "series_name", name)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _training_tail_baseline(
    wrapper,
    frame: pd.DataFrame,
    feature_cfg: FeatureConfig,
    split,
    cfg: dict,
    tail_of: int | None = None,
    horizon: int = 1,
) -> float:
    """Median-forecast MAE on the final `baseline_days` of the model's training window
    (a healthy period for that model) — the residual monitor's baseline."""
    residual_cfg = cfg.get("drift", {}).get("residual", {})
    days = int(residual_cfg.get("baseline_days", 2))
    end = tail_of if tail_of is not None else split.train_end
    features = build_features(frame["value"], feature_cfg, frame["ts"])
    row_index = features.index.to_numpy()
    values = frame["value"].to_numpy(dtype=float)
    tail_positions = np.flatnonzero(
        (row_index >= end - days * 1440) & (row_index + horizon < end)
    )
    X = features.to_numpy()[tail_positions]
    frame_in = pd.DataFrame(X, columns=feature_names(feature_cfg))
    frame_in["horizon"] = horizon
    raw = wrapper.predict(None, frame_in)
    medians = raw[f"q{0.5}"].to_numpy()
    actuals = values[row_index[tail_positions] + horizon]
    return float(np.mean(np.abs(actuals - medians)))


def _persist_simulated_forecasts(
    db,
    series_id: int,
    frame: pd.DataFrame,
    wrapper,
    feature_cfg: FeatureConfig,
    quantiles: tuple[float, ...],
    cutoffs: list[int],
    horizon: int,
    model_version: int,
    model_name: str,
) -> int:
    """Persist StoredForecast rows at historical cutoffs (real schema, real model)."""
    from backend.app.models.prediction import StoredForecast

    features = build_features(frame["value"], feature_cfg, frame["ts"])
    ts_values = pd.to_datetime(frame["ts"])
    count = 0
    for cutoff in cutoffs:
        positions = np.flatnonzero(features.index.to_numpy() == cutoff)
        if len(positions) == 0:
            continue
        row = features.iloc[[positions[0]]].copy()
        row["horizon"] = horizon
        raw = wrapper.predict(None, row)
        quantiles_map = _quantile_cols(quantiles, raw)
        target_ts = ts_values.iloc[cutoff + horizon].to_pydatetime()
        db.add(
            StoredForecast(
                series_id=series_id,
                created_at=datetime.now(UTC),
                horizon_minutes=horizon,
                target_ts=target_ts,
                quantiles=quantiles_map,
                model_name=model_name,
                model_version=model_version,
            )
        )
        count += 1
    db.commit()
    return count


if __name__ == "__main__":
    raise SystemExit(main())
