"""Revision ID: 0004 — alerts table (anomaly hits, forecast band breaches, drift)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("message", sa.String(length=300), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("dedupe_key", sa.String(length=80), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("series_id", "kind", "dedupe_key", name="uq_alerts_dedupe"),
    )
    op.create_index("ix_alerts_series_time", "alerts", ["series_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_alerts_series_time", table_name="alerts")
    op.drop_table("alerts")
