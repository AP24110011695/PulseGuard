"""Bulk metric ingestion: validated, chunked, idempotent upserts (plan §3)."""

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.metric import MetricPoint
from backend.app.schemas.metric import IngestResult, PointIn


def ingest_points(db: Session, series_id: int, points: list[PointIn]) -> IngestResult:
    rows = [
        {
            "series_id": series_id,
            "ts": p.ts,
            "value": p.value,
            "source_label": p.source_label,
        }
        for p in points
    ]
    inserted = 0
    for start in range(0, len(rows), settings.ingest_chunk_size):
        chunk = rows[start : start + settings.ingest_chunk_size]
        stmt = pg_insert(MetricPoint).values(chunk)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_metric_points_series_ts")
        # rowcount is unreliable (-1) for bulk ON CONFLICT inserts on psycopg3;
        # RETURNING gives the exact number of rows actually inserted.
        inserted += len(db.execute(stmt.returning(MetricPoint.id)).all())
    db.commit()
    return IngestResult(
        series_id=series_id,
        received=len(rows),
        inserted=inserted,
        duplicates=len(rows) - inserted,
    )
