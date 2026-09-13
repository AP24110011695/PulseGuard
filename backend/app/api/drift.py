"""Drift + lifecycle endpoints: status, events, on-demand checks, promotion history,
and the controlled retraining trigger."""

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.core.errors import ApiError
from backend.app.models.drift import DriftEvent, PromotionDecision
from backend.app.schemas.drift import (
    DriftCheckRequest,
    DriftCheckResponse,
    DriftEventsPage,
    DriftStatusResponse,
    PromotionsPage,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
)
from backend.app.services.alerts import check_forecast_breaches
from backend.app.services.drift import (
    drift_status,
    record_promotion_decision,
    run_psi_check,
    run_residual_check,
)
from backend.app.services.inference import ModelUnavailable

router = APIRouter(tags=["drift"])

TASKS = ("forecast", "anomaly")


@router.get("/drift/status", response_model=DriftStatusResponse)
def get_drift_status(
    series_id: int | None = None,
    db: Session = Depends(get_db),
) -> DriftStatusResponse:
    return DriftStatusResponse(series=drift_status(db, series_id))


@router.get("/drift/events", response_model=DriftEventsPage)
def get_drift_events(
    series_id: int | None = None,
    kind: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> DriftEventsPage:
    stmt = select(DriftEvent)
    if series_id is not None:
        stmt = stmt.where(DriftEvent.series_id == series_id)
    if kind is not None:
        stmt = stmt.where(DriftEvent.kind == kind)
    rows = db.execute(
        stmt.order_by(DriftEvent.detected_at.desc(), DriftEvent.id.desc()).limit(limit)
    ).scalars().all()
    return DriftEventsPage(count=len(rows), events=list(rows))


@router.post("/drift/check", response_model=DriftCheckResponse)
def run_drift_check_endpoint(
    payload: DriftCheckRequest,
    db: Session = Depends(get_db),
) -> DriftCheckResponse:
    if payload.task not in TASKS:
        raise ApiError(422, "invalid_task", f"task must be one of {list(TASKS)}")
    from backend.app.models.metric import MetricSeries

    if payload.series_id is not None:
        series_ids = [payload.series_id]
    else:
        series_ids = list(db.execute(select(MetricSeries.id).order_by(MetricSeries.id)).scalars())
    if not series_ids:
        raise ApiError(404, "no_series", "no metric series exist")

    checks: list[dict] = []
    events_recorded = 0
    for series_id in series_ids:
        try:
            psi_check = run_psi_check(db, series_id, payload.task, payload.window_points)
            checks.append(psi_check)
            events_recorded += 1
        except (ValueError, ModelUnavailable) as exc:
            checks.append(
                {"series_id": series_id, "kind": "psi", "status": "skipped", "error": str(exc)}
            )
        try:
            residual = run_residual_check(
                db, series_id, payload.task, window=12, multiplier=1.5, sustained=3
            )
            checks.append(residual)
            if residual["status"] != "unknown":
                events_recorded += 1
        except ModelUnavailable as exc:
            checks.append(
                {"series_id": series_id, "kind": "residual", "status": "skipped", "error": str(exc)}
            )
        if payload.task == "forecast":
            try:
                breaches = check_forecast_breaches(db, series_id)
                checks.append(
                    {
                        "series_id": series_id,
                        "kind": "forecast_breach",
                        "status": "ok",
                        "breaches": breaches,
                    }
                )
            except ModelUnavailable as exc:
                checks.append(
                    {
                        "series_id": series_id,
                        "kind": "forecast_breach",
                        "status": "skipped",
                        "error": str(exc),
                    }
                )
    return DriftCheckResponse(checks=checks, events_recorded=events_recorded)


@router.get("/promotions", response_model=PromotionsPage)
def get_promotions(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> PromotionsPage:
    rows = db.execute(
        select(PromotionDecision)
        .order_by(PromotionDecision.decided_at.desc(), PromotionDecision.id.desc())
        .limit(limit)
    ).scalars().all()
    return PromotionsPage(count=len(rows), decisions=list(rows))


@router.post("/retraining/trigger", response_model=RetrainingTriggerResponse, status_code=202)
def trigger_retraining(
    payload: RetrainingTriggerRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> RetrainingTriggerResponse:
    if payload.task not in TASKS:
        raise ApiError(422, "invalid_task", f"task must be one of {list(TASKS)}")
    if payload.challenger_window not in ("expanding", "pre_drift"):
        raise ApiError(422, "invalid_window", "challenger_window must be expanding or pre_drift")

    def _run() -> None:
        from ml.retrain import run_retraining

        session_factory = request.app.state.session_factory

        def sink(decision: dict) -> None:
            db = session_factory()
            try:
                record_promotion_decision(db, decision)
            finally:
                db.close()

        run_retraining(
            payload.task,
            payload.config,
            challenger_window=payload.challenger_window,
            decision_sink=sink,
        )

    background_tasks.add_task(_run)
    return RetrainingTriggerResponse(
        status="queued",
        task=payload.task,
        config=payload.config,
        challenger_window=payload.challenger_window,
    )
