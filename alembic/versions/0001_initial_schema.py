"""Initial Aurelius database schema

Revision ID: 0001
Revises:
Create Date: 2026-04-26

Creates tables for agents, sessions, activity, notifications,
memory entries, logs, and configuration store.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("state", sa.String(32), nullable=False, server_default="idle"),
        sa.Column("role", sa.String(128), nullable=False, server_default=""),
        sa.Column("metrics_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "activity",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("command", sa.String(512), nullable=False),
        sa.Column("success", sa.Boolean, nullable=False, server_default=sa.text("1")),
        sa.Column("output", sa.Text, nullable=False, server_default=""),
        sa.Column("agent_id", sa.String(64), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("idx_activity_timestamp", "activity", ["timestamp"])
    op.create_index("idx_activity_command", "activity", ["command"])

    op.create_table(
        "notifications",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("channel", sa.String(32), nullable=False, server_default="system"),
        sa.Column("priority", sa.String(16), nullable=False, server_default="medium"),
        sa.Column("category", sa.String(32), nullable=False, server_default="info"),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("body", sa.Text, nullable=False, server_default=""),
        sa.Column("read", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("delivered", sa.Boolean, nullable=False, server_default=sa.text("0")),
    )
    op.create_index("idx_notifications_timestamp", "notifications", ["timestamp"])
    op.create_index("idx_notifications_category", "notifications", ["category"])

    op.create_table(
        "memory_entries",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("layer", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.String(32), nullable=False),
        sa.Column("access_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("importance_score", sa.Float, nullable=False, server_default=sa.text("0.5")),
    )
    op.create_index("idx_memory_layer", "memory_entries", ["layer"])

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="user"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("last_activity", sa.DateTime, server_default=sa.func.now()),
        sa.Column("metadata_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("ip_address", sa.String(45), nullable=False, server_default=""),
        sa.Column("user_agent", sa.String(512), nullable=False, server_default=""),
    )
    op.create_index("idx_sessions_user", "sessions", ["user_id"])
    op.create_index("idx_sessions_expires", "sessions", ["expires_at"])

    op.create_table(
        "logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.String(32), nullable=False),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("logger", sa.String(64), nullable=False, server_default="system"),
        sa.Column("message", sa.Text, nullable=False),
    )
    op.create_index("idx_logs_level", "logs", ["level"])
    op.create_index("idx_logs_timestamp", "logs", ["timestamp"])

    op.create_table(
        "config",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("value", sa.Text, nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="user"),
        sa.Column("scopes", sa.Text, nullable=False, server_default="read"),
        sa.Column("rate_limit_rps", sa.Float, nullable=False, server_default=sa.text("60.0")),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=True),
        sa.Column("last_used_at", sa.DateTime, nullable=True),
    )
    op.create_index("idx_api_keys_user", "api_keys", ["user_id"])


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("config")
    op.drop_table("logs")
    op.drop_table("sessions")
    op.drop_table("memory_entries")
    op.drop_table("notifications")
    op.drop_table("activity")
    op.drop_table("agents")
