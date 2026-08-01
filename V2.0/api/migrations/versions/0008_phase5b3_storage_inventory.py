"""Add provider-scoped resumable staging inventory checkpoints."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_phase5b3_storage_inventory"
down_revision: str | Sequence[str] | None = "0007_phase5b2_orphan_staging"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("uq_stored_objects_object_key", "stored_objects", type_="unique")
    op.drop_constraint("uq_stored_objects_staging_key", "stored_objects", type_="unique")
    op.create_unique_constraint(
        "uq_stored_objects_backend_object_key",
        "stored_objects",
        ["storage_backend", "object_key"],
    )
    op.create_unique_constraint(
        "uq_stored_objects_backend_staging_key",
        "stored_objects",
        ["storage_backend", "staging_key"],
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column("backend_name", sa.String(64), nullable=False, server_default="local"),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column(
            "inventory_scope",
            sa.String(1024),
            nullable=False,
            server_default="legacy-local-staging-v1",
        ),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column(
            "provider_last_modified",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column("provider_etag", sa.String(512), nullable=True),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column("seen_generation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column("deletion_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "orphan_staging_candidates",
        sa.Column("deletion_completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_orphan_staging_candidates_deletion_completion",
        "orphan_staging_candidates",
        "deletion_completed_at IS NULL OR deletion_started_at IS NOT NULL",
    )
    op.drop_index(
        "ix_orphan_staging_candidates_cleanup",
        table_name="orphan_staging_candidates",
    )
    op.drop_constraint(
        "ck_orphan_staging_candidates_observation_count_positive",
        "orphan_staging_candidates",
        type_="check",
    )
    op.drop_constraint(
        "pk_orphan_staging_candidates",
        "orphan_staging_candidates",
        type_="primary",
    )
    op.create_primary_key(
        "pk_orphan_staging_candidates",
        "orphan_staging_candidates",
        ["backend_name", "inventory_scope", "staging_key"],
    )
    op.create_check_constraint(
        "ck_orphan_staging_candidates_observation_count_nonnegative",
        "orphan_staging_candidates",
        "observation_count >= 0",
    )
    op.alter_column(
        "orphan_staging_candidates",
        "observation_count",
        server_default="0",
    )
    op.create_index(
        "ix_orphan_staging_candidates_cleanup",
        "orphan_staging_candidates",
        ["backend_name", "inventory_scope", "quarantined_at", "next_attempt_at", "first_seen_at"],
    )
    op.alter_column("orphan_staging_candidates", "backend_name", server_default=None)
    op.alter_column("orphan_staging_candidates", "inventory_scope", server_default=None)
    op.alter_column("orphan_staging_candidates", "provider_last_modified", server_default=None)

    op.create_table(
        "storage_inventory_checkpoints",
        sa.Column("backend_name", sa.String(64), nullable=False),
        sa.Column("inventory_scope", sa.String(1024), nullable=False),
        sa.Column("generation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("cursor", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_checkpoint_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(255), nullable=False),
        sa.Column("task_fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error_code", sa.String(128), nullable=True),
        sa.Column("last_error_message", sa.String(1024), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_storage_inventory_checkpoints_status",
        ),
        sa.CheckConstraint(
            "page_count >= 0",
            name="ck_storage_inventory_checkpoints_page_count_nonnegative",
        ),
        sa.CheckConstraint(
            "item_count >= 0",
            name="ck_storage_inventory_checkpoints_item_count_nonnegative",
        ),
        sa.CheckConstraint(
            "task_fencing_token >= 0",
            name="ck_storage_inventory_checkpoints_task_fencing_token_nonnegative",
        ),
        sa.CheckConstraint(
            "(status = 'completed' AND completed_at IS NOT NULL AND cursor IS NULL) OR "
            "(status <> 'completed' AND completed_at IS NULL)",
            name="ck_storage_inventory_checkpoints_completion_state",
        ),
        sa.PrimaryKeyConstraint(
            "backend_name",
            "inventory_scope",
            name="pk_storage_inventory_checkpoints",
        ),
        sa.UniqueConstraint(
            "generation_id",
            name="uq_storage_inventory_checkpoints_generation",
        ),
    )


def downgrade() -> None:
    connection = op.get_bind()
    dangerous = connection.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM storage_inventory_checkpoints) OR "
            "EXISTS (SELECT 1 FROM orphan_staging_candidates)"
        )
    ).scalar_one()
    if dangerous:
        raise RuntimeError(
            "Phase 5B-3 inventory metadata exists; complete or explicitly clear the "
            "inventory manifest before downgrading to 0007."
        )
    duplicate_object_keys = connection.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM stored_objects GROUP BY object_key HAVING count(*) > 1)"
        )
    ).scalar_one()
    duplicate_staging_keys = connection.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM stored_objects WHERE staging_key IS NOT NULL "
            "GROUP BY staging_key HAVING count(*) > 1)"
        )
    ).scalar_one()
    if duplicate_object_keys or duplicate_staging_keys:
        raise RuntimeError(
            "Phase 5B-3 downgrade refused because provider-scoped object keys collide; "
            "reconcile or explicitly remove the duplicate backend rows first."
        )
    op.drop_table("storage_inventory_checkpoints")
    op.drop_index(
        "ix_orphan_staging_candidates_cleanup",
        table_name="orphan_staging_candidates",
    )
    op.drop_constraint(
        "ck_orphan_staging_candidates_observation_count_nonnegative",
        "orphan_staging_candidates",
        type_="check",
    )
    op.drop_constraint(
        "pk_orphan_staging_candidates",
        "orphan_staging_candidates",
        type_="primary",
    )
    op.create_primary_key(
        "pk_orphan_staging_candidates",
        "orphan_staging_candidates",
        ["staging_key"],
    )
    op.create_check_constraint(
        "ck_orphan_staging_candidates_observation_count_positive",
        "orphan_staging_candidates",
        "observation_count >= 1",
    )
    op.alter_column(
        "orphan_staging_candidates",
        "observation_count",
        server_default="1",
    )
    op.create_index(
        "ix_orphan_staging_candidates_cleanup",
        "orphan_staging_candidates",
        ["quarantined_at", "next_attempt_at", "first_seen_at"],
    )
    op.drop_constraint(
        "ck_orphan_staging_candidates_deletion_completion",
        "orphan_staging_candidates",
        type_="check",
    )
    op.drop_column("orphan_staging_candidates", "deletion_completed_at")
    op.drop_column("orphan_staging_candidates", "deletion_started_at")
    op.drop_column("orphan_staging_candidates", "seen_generation_id")
    op.drop_column("orphan_staging_candidates", "provider_etag")
    op.drop_column("orphan_staging_candidates", "provider_last_modified")
    op.drop_column("orphan_staging_candidates", "inventory_scope")
    op.drop_column("orphan_staging_candidates", "backend_name")
    op.drop_constraint(
        "uq_stored_objects_backend_staging_key",
        "stored_objects",
        type_="unique",
    )
    op.drop_constraint(
        "uq_stored_objects_backend_object_key",
        "stored_objects",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_stored_objects_staging_key",
        "stored_objects",
        ["staging_key"],
    )
    op.create_unique_constraint(
        "uq_stored_objects_object_key",
        "stored_objects",
        ["object_key"],
    )
