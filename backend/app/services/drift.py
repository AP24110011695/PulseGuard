"""Drift-check DB wiring: references, PSI checks, residual checks, events, status.

The pure math lives in ml/drift.py; this service loads the champion's stored reference
(from its MLflow artifact, cached into drift_references on first use), pulls the recent
window from the metric store, computes PSI + residual status, and appends drift_events.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow.artifacts
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.drift import DriftEvent, DriftReference
from backend.app.models.metric import MetricPoint
from backend.app.models.prediction import StoredForecast
from backend.app.services.inference import (
    LOOKBACK_BUFFER,
    model_manager,
)
from ml.drift import compute_feature_psi, psi_status, residual_drift
from ml.features import FeatureConfig, build_features


def _champion_reference(db: Session, task: str, champion) -> dict | None:
    """Load the champion's drift reference (per-series feature distributions) from its
    run artifact, caching it into drift_references on first use."""
    cached = db.execute(
        select(DriftReference).where(
            DriftReference.task == task,
            DriftReference.model_name == champion.name,
            DriftReference.model_version == champion.version,
            DriftReference.kind == "psi",
        )
    ).scalars().first()
    if cached is not None:
        return cached.payload

    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=_run_id(db, task, champion),
            artifact_path="drift_reference.json",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            dst_path=tempfile.mkdtemp(),
        )
    except Exception as exc:
        import logging

        logging.getLogger("pulseguard").warning("reference download failed: %s", exc)
        return None
    payload = json.loads(Path(local).read_text(encoding="utf-8"))
    # cache the multi-series payload against an arbitrary series id; lookups key on
    # (task, model_name, model_version) and read the per-series entry from the payload
    db.add(
        DriftReference(
            series_id=_any_series_id(db),
            task=task,
            model_name=champion.name,
            model_version=champion.version,
            kind="psi",
            payload=payload,
        )
    )
    db.commit()
    return payload


def _run_id(db: Session, task: str, champion) -> str:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mv = client.get_model_version(champion.name, str(champion.version))
    return mv.run_id


def _any_series_id(db: Session) -> int:
    from backend.app.models.metric import MetricSeries

    return db.scalar(select(MetricSeries.id).order_by(MetricSeries.id)) or 0


def _series_name(db: Session, series_id: int) -> str:
    from backend.app.models.metric import MetricSeries

    return db.get(MetricSeries, series_id).name


def run_psi_check(
    db: Session,
    series_id: int,
    task: str,
    window_points: int,
) -> dict:
    """PSI of the recent window against the champion's stored reference."""
    champion = model_manager.ensure_current(task)
    feature_cfg = FeatureConfig(**champion.feature_config)
    reference = _champion_reference(db, task, champion)
    if reference is None:
        raise ValueError(
            f"champion {champion.name} v{champion.version} has no drift reference artifact"
        )

    series_name = _series_name(db, series_id)
    series_ref = (reference.get("series") or {}).get(series_name)
    if series_ref is None:
        raise ValueError(f"reference has no entry for series '{series_name}'")

    lookback = window_points + feature_cfg.max_lookback + LOOKBACK_BUFFER
    points = db.execute(
        select(MetricPoint)
        .where(MetricPoint.series_id == series_id)
        .order_by(MetricPoint.ts.desc())
        .limit(lookback)
    ).scalars().all()
    if len(points) < feature_cfg.max_lookback + 1:
        raise ValueError("insufficient history for a drift check")

    points = list(reversed(points))  # restore chronological order for trailing features
    frame = pd.DataFrame(
        {
            "value": [p.value for p in points],
            "ts": pd.to_datetime([p.ts for p in points], utc=True),
        }
    )
    features = build_features(frame["value"], feature_cfg, frame["ts"])
    recent = features.tail(window_points)
    from ml.features import feature_names as _names

    psis = compute_feature_psi(series_ref, recent.to_numpy(), _names(feature_cfg))
    worst_feature = max(psis, key=psis.get)
    worst_value = psis[worst_feature]
    status = psi_status(worst_value)

    event = DriftEvent(
        series_id=series_id,
        kind="psi",
        status=status,
        feature=worst_feature,
        psi_value=worst_value,
        threshold=0.20,
        details={
            "task": task,
            "model_name": champion.name,
            "model_version": champion.version,
            "window_points": window_points,
            "feature_psis": psis,
        },
    )
    db.add(event)
    db.commit()
    return {
        "series_id": series_id,
        "series_name": series_name,
        "kind": "psi",
        "status": status,
        "worst_feature": worst_feature,
        "worst_psi": worst_value,
        "feature_psis": psis,
        "event_id": event.id,
        "model_version": champion.version,
    }


def run_residual_check(
    db: Session,
    series_id: int,
    task: str,
    window: int,
    multiplier: float,
    sustained: int,
) -> dict:
    """Residual monitor over stored median forecasts vs actuals (plan §10)."""
    champion = model_manager.ensure_current(task)
    now = datetime.now(UTC)
    forecasts = db.execute(
        select(StoredForecast)
        .where(
            StoredForecast.series_id == series_id,
            StoredForecast.model_name == champion.name,
            StoredForecast.model_version == champion.version,
            StoredForecast.target_ts <= now,
        )
        .order_by(StoredForecast.target_ts.asc())
    ).scalars().all()
    if len(forecasts) < window:
        return {
            "series_id": series_id,
            "kind": "residual",
            "status": "unknown",
            "details": {"reason": f"fewer than {window} stored forecasts with actuals"},
        }

    actuals, medians = [], []
    for f in forecasts:
        median = f.quantiles.get("0.5")
        if median is None:
            continue
        actual = db.scalar(
            select(MetricPoint.value).where(
                MetricPoint.series_id == series_id, MetricPoint.ts == f.target_ts
            )
        )
        if actual is not None:
            actuals.append(float(actual))
            medians.append(float(median))

    baseline_row = db.execute(
        select(DriftReference).where(
            DriftReference.task == task,
            DriftReference.model_name == champion.name,
            DriftReference.model_version == champion.version,
            DriftReference.series_id == series_id,
            DriftReference.kind == "residual",
        )
    ).scalars().first()
    baseline = baseline_row.payload.get("baseline_mae") if baseline_row else None

    result = residual_drift(actuals, medians, baseline, window, multiplier, sustained)
    event = DriftEvent(
        series_id=series_id,
        kind="residual",
        status=result["status"],
        feature=None,
        psi_value=None,
        threshold=multiplier,
        details={
            "task": task,
            "model_name": champion.name,
            "model_version": champion.version,
            "window_maes": result["window_maes"],
            "baseline_mae": result["baseline_mae"],
            "n_forecasts": len(actuals),
        },
    )
    db.add(event)
    db.commit()
    return {"series_id": series_id, "kind": "residual", **result, "event_id": event.id}


def store_residual_baseline(
    db: Session,
    series_id: int,
    task: str,
    model_name: str,
    model_version: int,
    baseline_mae: float,
) -> None:
    """Persist the champion's healthy-period residual baseline (written at promotion
    time by the retraining flow / demo)."""
    db.add(
        DriftReference(
            series_id=series_id,
            task=task,
            model_name=model_name,
            model_version=model_version,
            kind="residual",
            payload={"baseline_mae": baseline_mae},
        )
    )
    db.commit()


def record_promotion_decision(db: Session, decision: dict) -> None:
    """Persist a promotion-gate decision (both metric sets + rule trace)."""
    from backend.app.models.drift import PromotionDecision

    db.add(
        PromotionDecision(
            task=decision["task"],
            champion_name=decision["champion"]["name"],
            champion_version=decision["champion"]["version"],
            challenger_name=decision["challenger"].get("name", "pulseguard-challenger"),
            challenger_version=decision.get("registered_version"),
            decision=decision["gate"]["decision"],
            reason=decision["gate"]["reason"][:500],
            metrics={
                "champion_golden": decision["champion"]["golden_metrics"],
                "challenger_golden": decision["challenger"]["golden_metrics"],
                "rule_trace": decision["gate"]["rule_trace"],
                "golden_set": decision["golden_set"],
                "challenger_window": decision["challenger"]["window"],
            },
        )
    )
    db.commit()


def drift_status(db: Session, series_id: int | None = None, task: str = "forecast") -> list[dict]:
    """Latest PSI + residual event summary per series (read-only)."""
    from backend.app.models.metric import MetricSeries

    query = select(MetricSeries).order_by(MetricSeries.id)
    if series_id is not None:
        query = query.where(MetricSeries.id == series_id)
    out = []
    for series in db.execute(query).scalars().all():
        entry = {"series_id": series.id, "series_name": series.name, "psi": None, "residual": None}
        for kind in ("psi", "residual"):
            event = db.execute(
                select(DriftEvent)
                .where(DriftEvent.series_id == series.id, DriftEvent.kind == kind)
                .order_by(DriftEvent.detected_at.desc(), DriftEvent.id.desc())
                .limit(1)
            ).scalars().first()
            if event:
                entry[kind] = {
                    "status": event.status,
                    "detected_at": event.detected_at,
                    "feature": event.feature,
                    "psi_value": event.psi_value,
                    "details": event.details,
                }
        psi_status_value = (entry["psi"] or {}).get("status")
        residual_status_value = (entry["residual"] or {}).get("status")
        statuses = [s for s in (psi_status_value, residual_status_value) if s and s != "unknown"]
        entry["overall"] = (
            "drift" if "drift" in statuses else ("warn" if "warn" in statuses else "ok")
        )
        out.append(entry)
    return out
