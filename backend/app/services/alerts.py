"""Alert creation and feed (plan §5).

Alerts are raised on anomaly hits, forecast band breaches (actuals outside the stored
p05–p95 band once actuals exist), and drift events. The (series_id, kind, dedupe_key)
unique constraint makes repeated checks/scoring idempotent.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.models.alert import Alert
from backend.app.models.metric import MetricPoint
from backend.app.models.prediction import StoredForecast

SEVERITY = {"anomaly": "warning", "forecast_breach": "warning", "drift": "critical"}


def raise_alert(
    db: Session,
    series_id: int,
    kind: str,
    message: str,
    payload: dict,
    dedupe_key: str,
) -> bool:
    """Insert an alert if its dedupe identity is new; returns True when created."""
    stmt = pg_insert(Alert).values(
        series_id=series_id,
        kind=kind,
        severity=SEVERITY.get(kind, "info"),
        message=message[:300],
        payload=payload,
        dedupe_key=dedupe_key[:80],
    )
    stmt = stmt.on_conflict_do_nothing(constraint="uq_alerts_dedupe")
    created = db.execute(stmt).rowcount
    db.commit()
    return bool(created)


def raise_anomaly_alerts(db: Session, series_id: int, scored_rows) -> int:
    """One alert per flagged anomaly row (deduped by point timestamp)."""
    created = 0
    for row in scored_rows:
        if not row.is_anomaly:
            continue
        payload = {
            "score": row.score,
            "threshold": row.threshold,
            "model_version": row.model_version,
        }
        if raise_alert(
            db,
            series_id,
            "anomaly",
            f"anomaly detected at {row.ts} (score {row.score:.3f})",
            payload,
            dedupe_key=str(row.ts),
        ):
            created += 1
    return created


def check_forecast_breaches(db: Session, series_id: int, limit: int = 200) -> int:
    """Alert on stored forecasts whose actual arrived outside the p05–p95 band."""
    from backend.app.services.inference import model_manager

    champion = model_manager.get("forecast")
    forecasts = db.execute(
        select(StoredForecast)
        .where(
            StoredForecast.series_id == series_id,
            StoredForecast.model_name == champion.name,
            StoredForecast.model_version == champion.version,
        )
        .order_by(StoredForecast.target_ts.asc())
        .limit(limit)
    ).scalars().all()

    created = 0
    for f in forecasts:
        band = f.quantiles or {}
        low, high = band.get("0.05"), band.get("0.95")
        if low is None or high is None:
            continue
        actual = db.scalar(
            select(MetricPoint.value).where(
                MetricPoint.series_id == series_id, MetricPoint.ts == f.target_ts
            )
        )
        if actual is None:
            continue
        if low <= actual <= high:
            continue
        payload = {
            "target_ts": f.target_ts.isoformat(),
            "actual": actual,
            "p05": low,
            "p95": high,
            "model_version": champion.version,
        }
        if raise_alert(
            db,
            series_id,
            "forecast_breach",
            f"actual {actual:.2f} outside band [{low:.2f}, {high:.2f}] at {f.target_ts}",
            payload,
            dedupe_key=str(f.target_ts),
        ):
            created += 1
    return created


def list_alerts(db: Session, series_id: int | None, acknowledged: bool | None, limit: int):
    stmt = select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc()).limit(limit)
    if series_id is not None:
        stmt = stmt.where(Alert.series_id == series_id)
    if acknowledged is not None:
        stmt = stmt.where(Alert.acknowledged == acknowledged)
    return list(db.execute(stmt).scalars().all())


def acknowledge_alert(db: Session, alert_id: int) -> Alert | None:
    alert = db.get(Alert, alert_id)
    if alert is None:
        return None
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return alert
