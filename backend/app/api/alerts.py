"""Alert feed endpoints (Phase 5)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_series_or_404
from backend.app.core.errors import ApiError
from backend.app.schemas.alert import AckResponse, AlertsPage
from backend.app.services.alerts import acknowledge_alert, list_alerts

router = APIRouter(tags=["alerts"])


@router.get("/alerts", response_model=AlertsPage)
def get_alerts(
    series_id: int | None = None,
    acknowledged: bool | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> AlertsPage:
    if series_id is not None:
        get_series_or_404(series_id, db)
    rows = list_alerts(db, series_id, acknowledged, limit)
    unack = sum(1 for a in rows if not a.acknowledged)
    return AlertsPage(count=len(rows), unacknowledged=unack, alerts=rows)


@router.post("/alerts/{alert_id}/ack", response_model=AckResponse)
def ack_alert(alert_id: int, db: Session = Depends(get_db)) -> AckResponse:
    alert = acknowledge_alert(db, alert_id)
    if alert is None:
        raise ApiError(404, "alert_not_found", f"alert {alert_id} not found")
    return AckResponse(id=alert.id, acknowledged=alert.acknowledged)
