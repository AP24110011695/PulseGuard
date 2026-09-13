from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
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


class StoredForecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_forecasts_series_target_ts", "series_id", "target_ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # {"0.05": ..., "0.5": ..., "0.95": ...} — keyed by quantile
    quantiles: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"
    __table_args__ = (
        UniqueConstraint(
            "series_id", "ts", "model_name", "model_version", name="uq_anomaly_results_key"
        ),
        Index("ix_anomaly_results_series_ts", "series_id", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("metric_series.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ActiveModel(Base):
    __tablename__ = "active_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task: Mapped[str] = mapped_column(String(16), nullable=False)
    alias: Mapped[str] = mapped_column(String(32), nullable=False, server_default="champion")
    mlflow_model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    promoted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    __table_args__ = (UniqueConstraint("task", "alias", name="uq_active_models_task_alias"),)
