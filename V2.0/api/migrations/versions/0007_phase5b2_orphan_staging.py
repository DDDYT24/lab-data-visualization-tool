"""Persist two-pass orphan staging inventory candidates."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_phase5b2_orphan_staging"
down_revision: str | Sequence[str] | None = "0006_worker_leases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "orphan_staging_candidates",
        sa.Column("staging_key", sa.String(1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_error_code", sa.String(128), nullable=True),
        sa.Column("last_error_message", sa.String(1024), nullable=True),
        sa.Column("quarantined_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "observation_count >= 1",
            name=op.f("ck_orphan_staging_candidates_observation_count_positive"),
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name=op.f("ck_orphan_staging_candidates_size_bytes_nonnegative"),
        ),
        sa.CheckConstraint(
            "retry_count >= 0",
            name=op.f("ck_orphan_staging_candidates_retry_count_nonnegative"),
        ),
        sa.PrimaryKeyConstraint("staging_key", name="pk_orphan_staging_candidates"),
    )
    op.create_index(
        "ix_orphan_staging_candidates_cleanup",
        "orphan_staging_candidates",
        ["quarantined_at", "next_attempt_at", "first_seen_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_orphan_staging_candidates_cleanup",
        table_name="orphan_staging_candidates",
    )
    op.drop_table("orphan_staging_candidates")
