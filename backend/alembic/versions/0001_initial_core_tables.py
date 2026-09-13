"""core tables: metric_series, metric_points

Revision ID: 0001
Revises:
Create Date: 2026-09-13

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "metric_series",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=True),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("source", sa.String(length=32), server_default="api", nullable=False),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_metric_series_name"),
    )
    op.create_table(
        "metric_points",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Double(), nullable=False),
        sa.Column("source_label", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("series_id", "ts", name="uq_metric_points_series_ts"),
    )
    op.create_index(
        "ix_metric_points_series_ts", "metric_points", ["series_id", "ts"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_metric_points_series_ts", table_name="metric_points")
    op.drop_table("metric_points")
    op.drop_table("metric_series")
