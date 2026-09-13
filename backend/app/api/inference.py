"""Inference endpoints: forecast band computation and anomaly scoring, plus the
stored-prediction reads that expose model-version tracing."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_series_or_404
from backend.app.core.errors import ApiError
from backend.app.models.prediction import AnomalyResult, StoredForecast
from backend.app.schemas.inference import (
    AnomalyResultsPage,
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    StoredForecastOut,
    StoredForecastsPage,
)
from backend.app.services.inference import (
    ModelUnavailable,
    forecast_series,
    model_manager,
    score_series_anomalies,
    sync_active_model,
)

router = APIRouter(tags=["inference"])


def _map_unavailable(exc: Exception) -> ApiError:
    return ApiError(503, "model_unavailable", str(exc))


@router.get("/series/{series_id}/forecast", response_model=StoredForecastOut)
def compute_forecast(
    horizon_minutes: int = Query(gt=0),
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> StoredForecast:
    try:
        forecast = forecast_series(db, series.id, horizon_minutes)
    except ModelUnavailable as exc:
        raise _map_unavailable(exc) from exc
    except ValueError as exc:
        raise ApiError(422, "invalid_forecast_request", str(exc)) from exc
    sync_active_model(db, model_manager.get("forecast"))
    return forecast


@router.get("/series/{series_id}/forecasts", response_model=StoredForecastsPage)
def list_forecasts(
    start: datetime | None = None,
    end: datetime | None = None,
    model_version: int | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> StoredForecastsPage:
    stmt = select(StoredForecast).where(StoredForecast.series_id == series.id)
    if start is not None:
        stmt = stmt.where(StoredForecast.target_ts >= start)
    if end is not None:
        stmt = stmt.where(StoredForecast.target_ts < end)
    if model_version is not None:
        stmt = stmt.where(StoredForecast.model_version == model_version)
    # newest first: charts and feeds surface the most recent predictions
    rows = db.execute(stmt.order_by(StoredForecast.target_ts.desc()).limit(limit)).scalars().all()
    return StoredForecastsPage(series_id=series.id, count=len(rows), forecasts=list(rows))


@router.post("/series/{series_id}/anomalies/score", response_model=AnomalyScoreResponse)
def score_anomalies(
    payload: AnomalyScoreRequest,
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> AnomalyScoreResponse:
    try:
        results = score_series_anomalies(db, series.id, payload.start, payload.end, payload.limit)
    except ModelUnavailable as exc:
        raise _map_unavailable(exc) from exc
    except ValueError as exc:
        raise ApiError(422, "invalid_scoring_request", str(exc)) from exc
    sync_active_model(db, model_manager.get("anomaly"))
    return AnomalyScoreResponse(
        scored=len(results),
        flagged=sum(1 for r in results if r.is_anomaly),
        model_name=results[0].model_name,
        model_version=results[0].model_version,
        results=results,
    )


@router.get("/series/{series_id}/anomalies", response_model=AnomalyResultsPage)
def list_anomalies(
    start: datetime | None = None,
    end: datetime | None = None,
    model_version: int | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> AnomalyResultsPage:
    stmt = select(AnomalyResult).where(AnomalyResult.series_id == series.id)
    if start is not None:
        stmt = stmt.where(AnomalyResult.ts >= start)
    if end is not None:
        stmt = stmt.where(AnomalyResult.ts < end)
    if model_version is not None:
        stmt = stmt.where(AnomalyResult.model_version == model_version)
    rows = db.execute(stmt.order_by(AnomalyResult.ts.asc()).limit(limit)).scalars().all()
    return AnomalyResultsPage(series_id=series.id, count=len(rows), results=list(rows))
