"""Revision ID: 0002 — predictions and registry mirror tables."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecasts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("target_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantiles", postgresql.JSONB(), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecasts_series_target_ts", "forecasts", ["series_id", "target_ts"])

    op.create_table(
        "anomaly_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("series_id", "ts", "model_name", "model_version", name="uq_anomaly_results_key"),
    )
    op.create_index("ix_anomaly_results_series_ts", "anomaly_results", ["series_id", "ts"])

    op.create_table(
        "active_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("alias", sa.String(length=32), server_default="champion", nullable=False),
        sa.Column("mlflow_model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task", "alias", name="uq_active_models_task_alias"),
    )


def downgrade() -> None:
    op.drop_table("active_models")
    op.drop_index("ix_anomaly_results_series_ts", table_name="anomaly_results")
    op.drop_table("anomaly_results")
    op.drop_index("ix_forecasts_series_target_ts", table_name="forecasts")
    op.drop_table("forecasts")
