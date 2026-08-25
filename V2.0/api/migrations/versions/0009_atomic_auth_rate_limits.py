"""Add atomic fixed-window authentication rate-limit buckets.

Revision ID: 0009_atomic_auth_rate_limits
Revises: 0008_phase5b3_storage_inventory
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_atomic_auth_rate_limits"
down_revision: str | Sequence[str] | None = "0008_phase5b3_storage_inventory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auth_rate_limit_buckets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("identity_key", sa.String(320), nullable=False),
        sa.Column("window_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "scope IN ('client', 'email')",
            name=op.f("ck_auth_rate_limit_buckets_scope"),
        ),
        sa.CheckConstraint(
            "scope <> 'client' OR identity_key ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_auth_rate_limit_buckets_client_identity_lower_hex"),
        ),
        sa.CheckConstraint(
            "scope <> 'email' OR identity_key = lower(identity_key)",
            name=op.f("ck_auth_rate_limit_buckets_email_identity_normalized"),
        ),
        sa.CheckConstraint(
            "request_count >= 0",
            name=op.f("ck_auth_rate_limit_buckets_request_count_nonnegative"),
        ),
        sa.CheckConstraint(
            "expires_at > window_started_at",
            name=op.f("ck_auth_rate_limit_buckets_window_order"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_auth_rate_limit_buckets"),
        sa.UniqueConstraint(
            "scope",
            "identity_key",
            "window_started_at",
            name="uq_auth_rate_limit_bucket_window",
        ),
    )
    op.create_index(
        "ix_auth_rate_limit_buckets_expires",
        "auth_rate_limit_buckets",
        ["expires_at"],
    )

    # Preserve the current default one-hour email window and any client observations already
    # pseudonymized by 6B-1. Legacy raw client addresses are deliberately not copied.
    op.execute(
        sa.text(
            """
            WITH bounds AS (
                SELECT
                    to_timestamp(
                        floor(extract(epoch FROM statement_timestamp()) / 3600) * 3600
                    ) AS window_started_at
            ), observations AS (
                SELECT 'client'::text AS scope, client_key AS identity_key, requested_at
                FROM auth_requests
                WHERE client_key ~ '^[0-9a-f]{64}$'
                UNION ALL
                SELECT 'email'::text AS scope, lower(email) AS identity_key, requested_at
                FROM auth_requests
            )
            INSERT INTO auth_rate_limit_buckets (
                id, scope, identity_key, window_started_at,
                expires_at, request_count, updated_at
            )
            SELECT
                md5(
                    observations.scope || ':' || observations.identity_key || ':' ||
                    extract(epoch FROM bounds.window_started_at)::text
                )::uuid,
                observations.scope,
                observations.identity_key,
                bounds.window_started_at,
                bounds.window_started_at + interval '3600 seconds',
                count(*)::integer,
                statement_timestamp()
            FROM observations
            CROSS JOIN bounds
            WHERE observations.requested_at >= bounds.window_started_at
              AND observations.requested_at < bounds.window_started_at + interval '3600 seconds'
            GROUP BY observations.scope, observations.identity_key, bounds.window_started_at
            """
        )
    )
    op.alter_column("auth_rate_limit_buckets", "request_count", server_default=None)


def downgrade() -> None:
    connection = op.get_bind()
    retained = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM auth_rate_limit_buckets)")
    ).scalar_one()
    if retained:
        raise RuntimeError(
            "Authentication rate-limit buckets exist; wait for expiry or explicitly clear "
            "disposable limiter state before downgrading to 0008."
        )
    op.drop_index(
        "ix_auth_rate_limit_buckets_expires",
        table_name="auth_rate_limit_buckets",
    )
    op.drop_table("auth_rate_limit_buckets")
