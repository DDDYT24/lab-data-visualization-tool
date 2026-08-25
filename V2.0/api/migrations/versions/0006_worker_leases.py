"""Add Phase 5B-1 task and work-item lease infrastructure."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_worker_leases"
down_revision: str | Sequence[str] | None = "0005_share_publication_exports"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TASKS = (
    "pending-reconciliation",
    "project-lifecycle",
    "stored-object-gc",
    "orphan-staging-inventory",
    "metadata-cleanup",
)


def _add_work_item_columns(
    table_name: str,
    *,
    index_name: str,
    index_columns: list[str],
) -> None:
    op.add_column(table_name, sa.Column("lease_owner", sa.String(255), nullable=True))
    op.add_column(
        table_name,
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        table_name,
        sa.Column(
            "fencing_token",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        table_name,
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        table_name,
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        table_name,
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(table_name, sa.Column("last_error_code", sa.String(128), nullable=True))
    op.add_column(table_name, sa.Column("last_error_message", sa.String(1024), nullable=True))
    op.add_column(
        table_name,
        sa.Column("quarantined_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        op.f(f"ck_{table_name}_worker_lease_pair"),
        table_name,
        "(lease_owner IS NULL) = (lease_until IS NULL)",
    )
    op.create_check_constraint(
        op.f(f"ck_{table_name}_worker_fencing_token_nonnegative"),
        table_name,
        "fencing_token >= 0",
    )
    op.create_check_constraint(
        op.f(f"ck_{table_name}_worker_retry_count_nonnegative"),
        table_name,
        "retry_count >= 0",
    )
    op.create_check_constraint(
        op.f(f"ck_{table_name}_worker_quarantine_has_no_lease"),
        table_name,
        "quarantined_at IS NULL OR (lease_owner IS NULL AND lease_until IS NULL)",
    )
    op.create_index(index_name, table_name, index_columns)


def _drop_work_item_columns(table_name: str, *, index_name: str) -> None:
    op.drop_index(index_name, table_name=table_name)
    for suffix in (
        "worker_quarantine_has_no_lease",
        "worker_retry_count_nonnegative",
        "worker_fencing_token_nonnegative",
        "worker_lease_pair",
    ):
        op.drop_constraint(op.f(f"ck_{table_name}_{suffix}"), table_name, type_="check")
    for column_name in (
        "quarantined_at",
        "last_error_message",
        "last_error_code",
        "retry_count",
        "last_attempt_at",
        "next_attempt_at",
        "fencing_token",
        "lease_until",
        "lease_owner",
    ):
        op.drop_column(table_name, column_name)


def upgrade() -> None:
    op.create_table(
        "worker_leases",
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("lease_owner", sa.String(255), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "fencing_token",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "task IN ('pending-reconciliation', 'project-lifecycle', 'stored-object-gc', "
            "'orphan-staging-inventory', 'metadata-cleanup')",
            name=op.f("ck_worker_leases_task"),
        ),
        sa.CheckConstraint(
            "(lease_owner IS NULL) = (lease_until IS NULL)",
            name=op.f("ck_worker_leases_lease_pair"),
        ),
        sa.CheckConstraint(
            "fencing_token >= 0",
            name=op.f("ck_worker_leases_fencing_token_nonnegative"),
        ),
        sa.PrimaryKeyConstraint("task", name="pk_worker_leases"),
    )
    op.create_index("ix_worker_leases_until", "worker_leases", ["lease_until"])
    task_values = ", ".join(
        f"('{task}', 0, clock_timestamp(), clock_timestamp())" for task in TASKS
    )
    op.execute(
        "INSERT INTO worker_leases "
        "(task, fencing_token, created_at, updated_at) VALUES " + task_values
    )

    _add_work_item_columns(
        "projects",
        index_name="ix_projects_worker_claim",
        index_columns=[
            "quarantined_at",
            "next_attempt_at",
            "lease_until",
            "expires_at",
            "purge_after",
        ],
    )
    _add_work_item_columns(
        "stored_objects",
        index_name="ix_stored_objects_worker_claim",
        index_columns=[
            "status",
            "quarantined_at",
            "next_attempt_at",
            "lease_until",
            "gc_candidate_at",
        ],
    )
    _add_work_item_columns(
        "stored_object_write_intents",
        index_name="ix_stored_object_write_intents_worker_claim",
        index_columns=["status", "quarantined_at", "next_attempt_at", "lease_until"],
    )

    # 0005 intentionally guarded immutable intent identity and its one-way status
    # transition. 0006 keeps those rules while permitting fenced lease metadata
    # updates on pending intents.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION labviz_guard_write_intent_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.project_id IS DISTINCT FROM OLD.project_id
               OR NEW.export_job_id IS DISTINCT FROM OLD.export_job_id
               OR NEW.stored_object_id IS DISTINCT FROM OLD.stored_object_id
               OR NEW.operation IS DISTINCT FROM OLD.operation
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR OLD.status = 'completed'
               OR (OLD.status = 'pending' AND NEW.status NOT IN ('pending', 'completed'))
            THEN
                RAISE EXCEPTION 'invalid StoredObjectWriteIntent transition';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )

    op.execute(
        """
        CREATE FUNCTION labviz_guard_fencing_token() RETURNS trigger AS $$
        BEGIN
            IF NEW.fencing_token < OLD.fencing_token THEN
                RAISE EXCEPTION 'fencing_token cannot decrease';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    for table_name in (
        "worker_leases",
        "projects",
        "stored_objects",
        "stored_object_write_intents",
    ):
        op.execute(
            f"CREATE TRIGGER trg_{table_name}_fencing_monotonic "
            f"BEFORE UPDATE ON {table_name} FOR EACH ROW "
            "EXECUTE FUNCTION labviz_guard_fencing_token()"
        )


def downgrade() -> None:
    for table_name in (
        "stored_object_write_intents",
        "stored_objects",
        "projects",
        "worker_leases",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_fencing_monotonic ON {table_name}")
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_fencing_token()")

    op.execute(
        """
        CREATE OR REPLACE FUNCTION labviz_guard_write_intent_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.project_id IS DISTINCT FROM OLD.project_id
               OR NEW.export_job_id IS DISTINCT FROM OLD.export_job_id
               OR NEW.stored_object_id IS DISTINCT FROM OLD.stored_object_id
               OR NEW.operation IS DISTINCT FROM OLD.operation
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR OLD.status = 'completed'
               OR (OLD.status = 'pending' AND NEW.status <> 'completed')
            THEN
                RAISE EXCEPTION 'invalid StoredObjectWriteIntent transition';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )

    _drop_work_item_columns(
        "stored_object_write_intents",
        index_name="ix_stored_object_write_intents_worker_claim",
    )
    _drop_work_item_columns(
        "stored_objects",
        index_name="ix_stored_objects_worker_claim",
    )
    _drop_work_item_columns("projects", index_name="ix_projects_worker_claim")
    op.drop_index("ix_worker_leases_until", table_name="worker_leases")
    op.drop_table("worker_leases")
