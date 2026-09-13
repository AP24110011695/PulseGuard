"""Idempotent fresh-clone demo seed (Phase 6).

Run inside the compose network after `docker compose up -d --build`:

    docker compose run --rm api python scripts/seed_demo.py

What it does (safe to re-run):
1. ingests the seeded synthetic dataset through the ingestion API (duplicates no-op);
2. registers pooled champion models (forecast + anomaly) with drift references if the
   registry has none — training on the full pre-drift window, ~1-2 min on first run;
3. serves one forecast per horizon and one anomaly-scoring pass through the real
   serving path, plus a trailing strip of stored forecasts so the dashboard chart
   shows a prediction band;
4. runs one drift check (records PSI/residual events; drift alerts where they fire).

The result is a dashboard that is immediately alive against measured data.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml import tracking
from ml.backtest import SplitConfig, split_timeline
from ml.drift import feature_reference
from ml.features import FeatureConfig, build_features, feature_names
from ml.train import config_sha256, resolve_dataset, train_pooled_detector, train_pooled_forecaster


def _feature_config_dict(cfg: FeatureConfig) -> dict:
    return {
        "lags": list(cfg.lags),
        "rolling_windows": list(cfg.rolling_windows),
        "rolling_stats": list(cfg.rolling_stats),
        "roc_offsets": list(cfg.roc_offsets),
        "calendar": cfg.calendar,
    }


def _register_if_missing(
    task: str, cfg: dict, sha: str, frames: dict, meta: dict, feature_cfg: FeatureConfig, split
) -> tuple[str, int, bool]:
    """Register a pooled champion with references unless one already exists."""
    from sqlalchemy import select  # noqa: PLC0415

    from backend.app.core.database import SessionLocal  # noqa: PLC0415
    from backend.app.models.prediction import ActiveModel  # noqa: PLC0415

    try:
        # the registry is authoritative: an existing champion alias means we are done
        name, version = tracking.get_champion(task)
        return name, version, False
    except Exception:
        pass

    db = SessionLocal()
    try:
        existing = db.execute(
            select(ActiveModel).where(ActiveModel.task == task, ActiveModel.alias == "champion")
        ).scalars().first()
    finally:
        db.close()
    if existing is not None:
        return existing.mlflow_model_name, existing.model_version, False

    reference_window_days = int(cfg.get("drift", {}).get("reference_window_days", 7))
    names = [
        n
        for n in feature_names(feature_cfg)
        if not (n.startswith("hour_") or n.startswith("dow_"))
    ]
    references: dict = {}
    for name, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        mask = (row_index >= feature_cfg.max_lookback) & (row_index < split.train_end)
        positions = np.flatnonzero(mask)[-reference_window_days * meta["points_per_day"] :]
        references[name] = feature_reference(features.to_numpy()[positions], names, n_bins=10)

    wrapper = (
        train_pooled_forecaster(cfg, frames, meta, feature_cfg, split)
        if task == "forecast"
        else train_pooled_detector(cfg, frames, meta, feature_cfg, split)
    )
    horizons = tuple(int(h) for h in cfg["forecast"]["horizons"]) if task == "forecast" else ()
    quantiles = tuple(float(q) for q in cfg["forecast"]["quantiles"]) if task == "forecast" else ()
    name, version, _ = tracking.register_champion(
        task,
        wrapper,
        seed=int(cfg["seed"]),
        config_sha256=sha,
        feature_config=_feature_config_dict(feature_cfg),
        horizons=horizons,
        quantiles=quantiles,
        threshold=float(wrapper.threshold) if task == "anomaly" else None,
        reference={"series": references},
        run_name=f"{task}-seed",
    )
    return name, version, True


def _persist_forecast_strip(
    db,
    series_id: int,
    frame: pd.DataFrame,
    champion,
    feature_cfg: FeatureConfig,
    horizon: int,
    count: int,
) -> int:
    """Persist forecasts over the last `count` minutes (5-minute spacing) so the chart
    shows a prediction band against actuals."""
    from backend.app.models.prediction import StoredForecast

    features = build_features(frame["value"], feature_cfg, frame["ts"])
    ts_values = pd.to_datetime(frame["ts"])
    n = len(frame)
    created = 0
    for cutoff in range(max(features.index.min(), n - count), n - horizon, 5):
        # feature labels are ORIGINAL series positions (post-alignment-fix
        # convention) — select by label, not by position
        row = features.loc[[cutoff]].copy()
        row["horizon"] = horizon
        raw = champion.model.predict(row)
        db.add(
            StoredForecast(
                series_id=series_id,
                horizon_minutes=horizon,
                target_ts=ts_values.iloc[cutoff + horizon].to_pydatetime(),
                quantiles={str(q): float(raw[f"q{q}"].iloc[0]) for q in champion.quantiles},
                model_name=champion.name,
                model_version=champion.version,
            )
        )
        created += 1
    db.commit()
    return created


def main(argv: list[str] | None = None) -> int:
    from backend.app.core.database import SessionLocal
    from backend.app.services.alerts import check_forecast_breaches
    from backend.app.services.drift import run_psi_check, run_residual_check
    from backend.app.services.inference import (
        forecast_series,
        model_manager,
        score_series_anomalies,
    )

    parser = argparse.ArgumentParser(prog="scripts/seed_demo.py", description=__doc__)
    parser.add_argument("--config", default="ml/configs/train_synthetic.yaml")
    parser.add_argument(
        "--api-url",
        default="http://api:8000",
        help="in-compose default; use http://127.0.0.1:8000 when running on the host",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sha = config_sha256(cfg)
    frames, meta = resolve_dataset(cfg["dataset"])
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
    reference_frame = next(iter(frames))
    split = split_timeline(len(frames[reference_frame]), meta["points_per_day"], split_cfg)

    # ---- 1. ingest through the API (idempotent) ---------------------------------
    from ml.api_client import ingest_dataframe

    series_meta = {
        name: {"unit": "seed", "description": f"Seeded demo series {name}", "source": "simulator"}
        for name in frames
    }
    ingest = ingest_dataframe(
        args.api_url, df=_ingest_frame(frames), series_meta=series_meta, chunk_size=10_000
    )
    series_ids = {name: entry["series_id"] for name, entry in ingest["series"].items()}
    print(
        f"[seed] ingested {ingest['inserted']} new points "
        f"({ingest['duplicates']} duplicates skipped) across {len(series_ids)} series"
    )

    # ---- 2. champions (registered only if the registry has none) ----------------
    champions: dict[str, tuple[str, int, bool]] = {}
    for task in ("forecast", "anomaly"):
        name, version, created = _register_if_missing(
            task, cfg, sha, frames, meta, feature_cfg, split
        )
        champions[task] = (name, version, created)
        model_manager.reload(task)  # the API process picks the alias up per request
        action = "registered" if created else "already present"
        print(f"[seed] {task} champion: {name} v{version} ({action})")

    # ---- 3. live serving pass + forecast strip ----------------------------------
    db = SessionLocal()
    try:
        first_series = sorted(series_ids.values())[0]
        forecast = forecast_series(db, first_series, horizon_minutes=15)
        scored = score_series_anomalies(db, first_series, None, None, limit=400)
        print(
            f"[seed] forecast series {first_series} h=15 → band "
            f"{forecast.quantiles['0.05']:.1f}–{forecast.quantiles['0.95']:.1f} "
            f"(model v{forecast.model_version}); scored {len(scored)} points "
            f"({sum(1 for r in scored if r.is_anomaly)} flagged)"
        )

        champion = model_manager.get("forecast")
        for series_id in sorted(series_ids.values()):
            strip = _persist_forecast_strip(
                db,
                series_id,
                frames[_series_name(db, series_id)],
                champion,
                feature_cfg,
                horizon=5,
                count=1440,
            )
            print(f"[seed] stored {strip} forecasts for series {series_id}")

        # ---- 4. one drift check (events + alerts where they fire) ---------------
        for series_id in sorted(series_ids.values()):
            psi = run_psi_check(db, series_id, "forecast", window_points=2880)
            run_residual_check(db, series_id, "forecast", window=12, multiplier=1.5, sustained=3)
            check_forecast_breaches(db, series_id)
            print(f"[seed] drift check series {series_id}: psi={psi['status']}")
    finally:
        db.close()

    print(f"[seed] done in {time.perf_counter() - started:.1f}s — dashboard is ready")
    return 0


def _series_name(db, series_id: int) -> str:
    from sqlalchemy import select

    from backend.app.models.metric import MetricSeries

    return db.scalar(select(MetricSeries.name).where(MetricSeries.id == series_id))


def _ingest_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        part = frame[["ts", "value", "is_anomaly"]].copy()
        part.insert(0, "series_name", name)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
