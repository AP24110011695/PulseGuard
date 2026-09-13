from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_series_or_404
from backend.app.core.database import get_db
from backend.app.core.errors import ApiError
from backend.app.models.metric import MetricPoint, MetricSeries
from backend.app.schemas.metric import SeriesCreate, SeriesDetailOut, SeriesOut

router = APIRouter(tags=["series"])


@router.post("/series", response_model=SeriesOut, status_code=201)
def create_series(payload: SeriesCreate, db: Session = Depends(get_db)) -> MetricSeries:
    exists = db.scalar(select(MetricSeries).where(MetricSeries.name == payload.name))
    if exists is not None:
        raise ApiError(409, "series_exists", f"metric series name '{payload.name}' already exists")
    series = MetricSeries(**payload.model_dump())
    db.add(series)
    db.commit()
    db.refresh(series)
    return series


@router.get("/series", response_model=list[SeriesOut])
def list_series(
    source: str | None = None,
    name: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[MetricSeries]:
    stmt = select(MetricSeries).order_by(MetricSeries.id)
    if source is not None:
        stmt = stmt.where(MetricSeries.source == source)
    if name is not None:
        stmt = stmt.where(MetricSeries.name == name)
    return list(db.scalars(stmt.limit(limit).offset(offset)).all())


@router.get("/series/{series_id}", response_model=SeriesDetailOut)
def get_series(series=Depends(get_series_or_404), db: Session = Depends(get_db)) -> SeriesDetailOut:
    stats = db.execute(
        select(
            func.count(MetricPoint.id),
            func.min(MetricPoint.ts),
            func.max(MetricPoint.ts),
        ).where(MetricPoint.series_id == series.id)
    ).one()
    return SeriesDetailOut(
        id=series.id,
        name=series.name,
        unit=series.unit,
        description=series.description,
        source=series.source,
        tags=series.tags,
        created_at=series.created_at,
        point_count=stats[0] or 0,
        first_ts=stats[1],
        last_ts=stats[2],
    )
