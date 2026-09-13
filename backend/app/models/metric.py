from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class MetricSeries(Base):
    __tablename__ = "metric_series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    unit: Mapped[str | None] = mapped_column(String(32))
    description: Mapped[str | None] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(32), nullable=False, server_default="api")
    tags: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MetricPoint(Base):
    __tablename__ = "metric_points"
    __table_args__ = (
        UniqueConstraint("series_id", "ts", name="uq_metric_points_series_ts"),
        Index("ix_metric_points_series_ts", "series_id", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Double, nullable=False)
    # Ground-truth anomaly label when the data source provides one (simulator, NAB).
    # Kept separate from model output so raw data stays clean (plan §3).
    source_label: Mapped[bool | None] = mapped_column(Boolean)
