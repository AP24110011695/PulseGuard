from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class DriftReference(Base):
    __tablename__ = "drift_references"
    __table_args__ = (
        Index(
            "ix_drift_references_model",
            "task",
            "model_name",
            "model_version",
            "series_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    task: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # "psi" (per-feature bin edges + frequencies) | "residual" (healthy-period MAE)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DriftEvent(Base):
    __tablename__ = "drift_events"
    __table_args__ = (Index("ix_drift_events_series_time", "series_id", "detected_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # "psi" | "residual"
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(8), nullable=False)  # ok | warn | drift
    # psi: worst feature + per-feature values; residual: window MAEs + baseline
    feature: Mapped[str | None] = mapped_column(String(120))
    psi_value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False)


class PromotionDecision(Base):
    __tablename__ = "promotion_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    task: Mapped[str] = mapped_column(String(16), nullable=False)
    champion_name: Mapped[str] = mapped_column(String(120), nullable=False)
    champion_version: Mapped[int] = mapped_column(Integer, nullable=False)
    challenger_name: Mapped[str] = mapped_column(String(120), nullable=False)
    challenger_version: Mapped[int | None] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(String(8), nullable=False)  # promoted | rejected
    reason: Mapped[str] = mapped_column(String(500), nullable=False)
    # both metric sets + the gate rule trace + golden-set identity
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
