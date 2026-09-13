from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("series_id", "kind", "dedupe_key", name="uq_alerts_dedupe"),
        Index("ix_alerts_series_time", "series_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # "anomaly" | "forecast_breach" | "drift"
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False)  # info | warning | critical
    message: Mapped[str] = mapped_column(String(300), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # stable identity for dedupe (e.g. point ts, forecast target ts, drift feature+day)
    dedupe_key: Mapped[str] = mapped_column(String(80), nullable=False)
