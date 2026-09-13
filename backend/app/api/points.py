from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.app.api.deps import get_series_or_404
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.metric import MetricPoint
from backend.app.schemas.metric import IngestResult, PointOut, PointsIngest, PointsPage
from backend.app.services.ingestion import ingest_points

router = APIRouter(tags=["points"])


@router.post("/series/{series_id}/points", response_model=IngestResult, status_code=201)
def ingest_points_endpoint(
    payload: PointsIngest,
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> IngestResult:
    return ingest_points(db, series.id, payload.points)


@router.get("/series/{series_id}/points", response_model=PointsPage)
def query_points(
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: int = Query(
        default=settings.query_default_max_points, ge=1, le=settings.query_max_points_ceiling
    ),
    series=Depends(get_series_or_404),
    db: Session = Depends(get_db),
) -> PointsPage:
    filters = [MetricPoint.series_id == series.id]
    if start is not None:
        filters.append(MetricPoint.ts >= start)  # half-open window: start <= ts < end
    if end is not None:
        filters.append(MetricPoint.ts < end)

    total = db.scalar(select(func.count()).select_from(MetricPoint).where(*filters)) or 0
    page = PointsPage(
        series_id=series.id,
        start=start,
        end=end,
        max_points=max_points,
        downsampled=False,
        bucket_seconds=None,
        count=0,
        points=[],
    )
    if total == 0:
        return page

    if total <= max_points:
        rows = db.execute(
            select(MetricPoint.ts, MetricPoint.value, MetricPoint.source_label)
            .where(*filters)
            .order_by(MetricPoint.ts)
        ).all()
        page.points = [PointOut(ts=r[0], value=r[1], source_label=r[2]) for r in rows]
        page.count = len(page.points)
        return page

    first_ts, last_ts = db.execute(
        select(func.min(MetricPoint.ts), func.max(MetricPoint.ts)).where(*filters)
    ).one()
    span_seconds = (last_ts - first_ts).total_seconds() or 1.0

    if max_points == 1:
        row = db.execute(
            select(
                func.min(MetricPoint.ts),
                func.avg(MetricPoint.value),
                func.bool_or(MetricPoint.source_label),
            ).where(*filters)
        ).one()
        page.downsampled = True
        page.count = 1
        page.points = [PointOut(ts=row[0], value=float(row[1]), source_label=row[2])]
        return page

    # Bins anchored at first_ts: bucket = ceil(span / (max_points - 1)) guarantees
    # ceil(span / bucket) + 1 boundary buckets <= max_points.
    bucket_seconds = max(1, int(-(-span_seconds // (max_points - 1))))
    origin = text(f"TIMESTAMPTZ '{first_ts.isoformat()}'")
    bucket_col = func.date_bin(
        text(f"INTERVAL '{bucket_seconds} seconds'"), MetricPoint.ts, origin
    ).label("bucket")
    rows = db.execute(
        select(
            bucket_col,
            func.avg(MetricPoint.value).label("value"),
            func.bool_or(MetricPoint.source_label).label("source_label"),
        )
        .where(*filters)
        .group_by(bucket_col)
        .order_by(bucket_col)
    ).all()
    page.downsampled = True
    page.bucket_seconds = bucket_seconds
    page.points = [PointOut(ts=r[0], value=float(r[1]), source_label=r[2]) for r in rows]
    page.count = len(page.points)
    return page
