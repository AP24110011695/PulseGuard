"""Revision ID: 0003 — drift references, drift events, promotion decisions."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drift_references",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_drift_references_model",
        "drift_references",
        ["task", "model_name", "model_version", "series_id"],
    )

    op.create_table(
        "drift_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("feature", sa.String(length=120), nullable=True),
        sa.Column("psi_value", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["metric_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_events_series_time", "drift_events", ["series_id", "detected_at"])

    op.create_table(
        "promotion_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("champion_name", sa.String(length=120), nullable=False),
        sa.Column("champion_version", sa.Integer(), nullable=False),
        sa.Column("challenger_name", sa.String(length=120), nullable=False),
        sa.Column("challenger_version", sa.Integer(), nullable=True),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("promotion_decisions")
    op.drop_index("ix_drift_events_series_time", table_name="drift_events")
    op.drop_table("drift_events")
    op.drop_index("ix_drift_references_model", table_name="drift_references")
    op.drop_table("drift_references")
