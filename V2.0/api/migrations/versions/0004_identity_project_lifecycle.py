"""Phase 4 identity, ownership, project lifecycle, and object reuse.

Revision ID: 0004_identity_project_lifecycle
Revises: 0003_quality_cleaning_lineage
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0004_identity_project_lifecycle"
down_revision: str | Sequence[str] | None = "0003_quality_cleaning_lineage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PHASE4_IMMUTABLE_TABLES = ("idempotency_records",)


def upgrade() -> None:
    op.create_table(
        "guest_sessions",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("token_digest", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "token_digest ~ '^[0-9a-f]{64}$'", name="ck_guest_sessions_token_digest_lower_hex"
        ),
        sa.CheckConstraint("status IN ('active', 'revoked')", name="ck_guest_sessions_status"),
        sa.CheckConstraint(
            "(status = 'revoked' AND revoked_at IS NOT NULL) OR "
            "(status = 'active' AND revoked_at IS NULL)",
            name="ck_guest_sessions_revoked_status_time",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_guest_sessions"),
        sa.UniqueConstraint("token_digest", name="uq_guest_sessions_token_digest"),
    )
    op.create_index("ix_guest_sessions_status_expires", "guest_sessions", ["status", "expires_at"])

    op.add_column("projects", sa.Column("guest_session_id", UUID(as_uuid=True), nullable=True))
    op.add_column("projects", sa.Column("saved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "projects", sa.Column("lock_version", sa.Integer(), nullable=False, server_default="1")
    )
    op.create_foreign_key(
        "fk_projects_guest_session_id_guest_sessions",
        "projects",
        "guest_sessions",
        ["guest_session_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.execute(
        """
        INSERT INTO guest_sessions (
            id, token_digest, status, expires_at, last_seen_at, revoked_at, created_at
        )
        SELECT MIN(id::text)::uuid, guest_token_digest, 'active',
               MAX(COALESCE(expires_at, updated_at + INTERVAL '7 days')),
               MAX(updated_at), NULL, MIN(created_at)
        FROM projects
        WHERE storage_mode = 'temporary-cloud' AND guest_token_digest IS NOT NULL
        GROUP BY guest_token_digest
        """
    )
    op.execute(
        """
        INSERT INTO guest_sessions (
            id, token_digest, status, expires_at, last_seen_at, revoked_at, created_at
        )
        SELECT id,
               md5(id::text) || md5('legacy-guest:' || id::text),
               'active', COALESCE(expires_at, updated_at + INTERVAL '7 days'),
               updated_at, NULL, created_at
        FROM projects
        WHERE storage_mode = 'temporary-cloud' AND guest_token_digest IS NULL
        """
    )
    op.execute(
        """
        UPDATE projects AS project
        SET guest_session_id = guest.id
        FROM guest_sessions AS guest
        WHERE project.storage_mode = 'temporary-cloud'
          AND (
            project.guest_token_digest = guest.token_digest
            OR (project.guest_token_digest IS NULL AND project.id = guest.id)
          )
        """
    )
    op.drop_constraint("ck_projects_saved_project_owner", "projects", type_="check")
    op.drop_constraint("ck_projects_temporary_project_expiry", "projects", type_="check")
    op.create_check_constraint(
        "ownership_by_storage_mode",
        "projects",
        "(storage_mode = 'temporary-cloud' AND expires_at IS NOT NULL AND "
        "((owner_user_id IS NULL) <> (guest_session_id IS NULL))) OR "
        "(storage_mode = 'saved-cloud' AND owner_user_id IS NOT NULL AND "
        "guest_session_id IS NULL AND expires_at IS NULL) OR storage_mode = 'local'",
    )
    op.alter_column("projects", "lock_version", server_default=None)
    op.drop_column("projects", "guest_token_digest")

    op.create_table(
        "auth_challenges",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("salt", sa.String(64), nullable=False),
        sa.Column("code_digest", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resend_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failed_attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("email = lower(email)", name="ck_auth_challenges_email_normalized"),
        sa.CheckConstraint(
            "failed_attempts >= 0", name="ck_auth_challenges_failed_attempts_nonnegative"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_auth_challenges"),
    )
    op.create_index("ix_auth_challenges_email_created", "auth_challenges", ["email", "created_at"])
    op.create_index("ix_auth_challenges_expires", "auth_challenges", ["expires_at"])

    op.create_table(
        "auth_sessions",
        sa.Column("token_digest", sa.String(64), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "token_digest ~ '^[0-9a-f]{64}$'", name="ck_auth_sessions_token_digest_lower_hex"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_auth_sessions_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("token_digest", name="pk_auth_sessions"),
    )
    op.create_index("ix_auth_sessions_user_expires", "auth_sessions", ["user_id", "expires_at"])
    op.create_index("ix_auth_sessions_expires", "auth_sessions", ["expires_at"])

    op.create_table(
        "auth_requests",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("client_key", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_auth_requests"),
    )
    op.create_index("ix_auth_requests_client_time", "auth_requests", ["client_key", "requested_at"])
    op.create_index("ix_auth_requests_email_time", "auth_requests", ["email", "requested_at"])

    op.add_column("stored_objects", sa.Column("dedup_scope", sa.String(255), nullable=True))
    op.add_column(
        "stored_objects", sa.Column("format_contract_version", sa.String(128), nullable=True)
    )
    op.add_column(
        "stored_objects", sa.Column("gc_candidate_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute("UPDATE stored_objects SET dedup_scope = 'legacy:' || id::text")
    op.execute(
        """
        UPDATE stored_objects AS stored
        SET dedup_scope = CASE
            WHEN project.storage_mode = 'saved-cloud' AND project.owner_user_id IS NOT NULL
                THEN 'user:' || replace(project.owner_user_id::text, '-', '')
            WHEN project.guest_session_id IS NOT NULL
                THEN 'guest:' || replace(project.guest_session_id::text, '-', '')
            ELSE stored.dedup_scope
        END
        FROM dataset_versions AS version
        JOIN projects AS project ON project.id = version.project_id
        WHERE version.stored_object_id = stored.id
        """
    )
    op.execute(
        """
        UPDATE stored_objects AS stored
        SET dedup_scope = CASE
            WHEN project.storage_mode = 'saved-cloud' AND project.owner_user_id IS NOT NULL
                THEN 'user:' || replace(project.owner_user_id::text, '-', '')
            WHEN project.guest_session_id IS NOT NULL
                THEN 'guest:' || replace(project.guest_session_id::text, '-', '')
            ELSE stored.dedup_scope
        END
        FROM source_files AS source
        JOIN projects AS project ON project.id = source.project_id
        WHERE source.stored_object_id = stored.id
        """
    )
    op.execute(
        "UPDATE stored_objects SET format_contract_version = "
        "CASE WHEN purpose = 'dataset' THEN 'parquet-v1' ELSE 'legacy-v1' END"
    )
    op.alter_column("stored_objects", "dedup_scope", nullable=False)
    op.alter_column("stored_objects", "format_contract_version", nullable=False)
    op.create_check_constraint("dedup_scope_nonempty", "stored_objects", "length(dedup_scope) > 0")
    op.create_check_constraint(
        "format_contract_version_nonempty",
        "stored_objects",
        "length(format_contract_version) > 0",
    )
    op.create_index("ix_stored_objects_gc_candidate", "stored_objects", ["gc_candidate_at"])
    op.create_index(
        "ix_stored_objects_dedup_lookup",
        "stored_objects",
        [
            "storage_backend",
            "dedup_scope",
            "purpose",
            "media_type",
            "format_contract_version",
            "sha256",
            "size_bytes",
            "encryption_key_id",
        ],
    )
    op.drop_constraint("uq_dataset_versions_stored_object_id", "dataset_versions", type_="unique")

    op.add_column(
        "processing_runs",
        sa.Column("execution_mode", sa.String(24), nullable=False, server_default="executed"),
    )
    op.add_column(
        "processing_runs", sa.Column("origin_processing_run_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "processing_runs", sa.Column("origin_run_uuid_snapshot", UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        "fk_processing_runs_origin_processing_run_id_processing_runs",
        "processing_runs",
        "processing_runs",
        ["origin_processing_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "execution_mode",
        "processing_runs",
        "execution_mode IN ('executed', 'reused-result')",
    )
    op.create_check_constraint(
        "reused_result_origin",
        "processing_runs",
        "execution_mode <> 'reused-result' OR origin_run_uuid_snapshot IS NOT NULL",
    )
    op.alter_column("processing_runs", "execution_mode", server_default=None)

    op.create_table(
        "project_claims",
        sa.Column("project_id", UUID(as_uuid=True), nullable=False),
        sa.Column("guest_session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("guest_session_uuid_snapshot", UUID(as_uuid=True), nullable=False),
        sa.Column("user_uuid_snapshot", UUID(as_uuid=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["guest_session_id"],
            ["guest_sessions.id"],
            name="fk_project_claims_guest_session_id_guest_sessions",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_project_claims_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_project_claims_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("project_id", name="pk_project_claims"),
    )

    op.create_table(
        "project_origins",
        sa.Column("target_project_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source_project_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source_revision_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source_project_uuid_snapshot", UUID(as_uuid=True), nullable=False),
        sa.Column("source_revision_uuid_snapshot", UUID(as_uuid=True), nullable=False),
        sa.Column("origin_kind", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "origin_kind IN ('duplicate', 'local-import')", name="ck_project_origins_origin_kind"
        ),
        sa.ForeignKeyConstraint(
            ["source_project_id"],
            ["projects.id"],
            name="fk_project_origins_source_project_id_projects",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_revision_id"],
            ["project_revisions.id"],
            name="fk_project_origins_source_revision_id_project_revisions",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["target_project_id"],
            ["projects.id"],
            name="fk_project_origins_target_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("target_project_id", name="pk_project_origins"),
    )

    op.create_table(
        "project_lifecycle_events",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", UUID(as_uuid=True), nullable=True),
        sa.Column("project_uuid_snapshot", UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("details", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('claim', 'save', 'duplicate', 'delete', 'restore', "
            "'revision-restore', 'purge')",
            name="ck_project_lifecycle_events_event_type",
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name="fk_project_lifecycle_events_actor_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_project_lifecycle_events_project_id_projects",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_project_lifecycle_events"),
    )
    op.create_index(
        "ix_project_lifecycle_events_project_created",
        "project_lifecycle_events",
        ["project_uuid_snapshot", "created_at"],
    )

    op.create_table(
        "idempotency_records",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("resource_id", UUID(as_uuid=True), nullable=False),
        sa.Column("response_document", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 255", name="ck_idempotency_records_key_length"
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name="fk_idempotency_records_actor_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_idempotency_records"),
        sa.UniqueConstraint(
            "actor_user_id",
            "operation",
            "idempotency_key",
            name="uq_idempotency_actor_operation",
        ),
    )
    op.create_index("ix_idempotency_records_created", "idempotency_records", ["created_at"])

    op.execute(
        """
        CREATE FUNCTION labviz_guard_project_claim_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.project_id IS DISTINCT FROM OLD.project_id
               OR NEW.guest_session_uuid_snapshot IS DISTINCT FROM OLD.guest_session_uuid_snapshot
               OR NEW.user_uuid_snapshot IS DISTINCT FROM OLD.user_uuid_snapshot
               OR NEW.claimed_at IS DISTINCT FROM OLD.claimed_at
               OR (NEW.guest_session_id IS DISTINCT FROM OLD.guest_session_id
                   AND NEW.guest_session_id IS NOT NULL)
               OR (NEW.user_id IS DISTINCT FROM OLD.user_id AND NEW.user_id IS NOT NULL)
            THEN
                RAISE EXCEPTION 'project_claims is immutable except for FK nullification';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_project_claims_guard BEFORE UPDATE ON project_claims "
        "FOR EACH ROW EXECUTE FUNCTION labviz_guard_project_claim_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_guard_project_origin_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.target_project_id IS DISTINCT FROM OLD.target_project_id
               OR NEW.source_project_uuid_snapshot IS DISTINCT FROM OLD.source_project_uuid_snapshot
               OR NEW.source_revision_uuid_snapshot
                  IS DISTINCT FROM OLD.source_revision_uuid_snapshot
               OR NEW.origin_kind IS DISTINCT FROM OLD.origin_kind
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR (NEW.source_project_id IS DISTINCT FROM OLD.source_project_id
                   AND NEW.source_project_id IS NOT NULL)
               OR (NEW.source_revision_id IS DISTINCT FROM OLD.source_revision_id
                   AND NEW.source_revision_id IS NOT NULL)
            THEN
                RAISE EXCEPTION 'project_origins is immutable except for FK nullification';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_project_origins_guard BEFORE UPDATE ON project_origins "
        "FOR EACH ROW EXECUTE FUNCTION labviz_guard_project_origin_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_guard_lifecycle_event_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.project_uuid_snapshot IS DISTINCT FROM OLD.project_uuid_snapshot
               OR NEW.event_type IS DISTINCT FROM OLD.event_type
               OR NEW.details IS DISTINCT FROM OLD.details
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR (NEW.project_id IS DISTINCT FROM OLD.project_id AND NEW.project_id IS NOT NULL)
               OR (NEW.actor_user_id IS DISTINCT FROM OLD.actor_user_id
                   AND NEW.actor_user_id IS NOT NULL)
            THEN
                RAISE EXCEPTION
                    'project_lifecycle_events is immutable except for FK nullification';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_project_lifecycle_events_guard "
        "BEFORE UPDATE ON project_lifecycle_events FOR EACH ROW "
        "EXECUTE FUNCTION labviz_guard_lifecycle_event_update()"
    )

    for table_name in PHASE4_IMMUTABLE_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table_name}_immutable "
            f"BEFORE UPDATE ON {table_name} FOR EACH ROW "
            "EXECUTE FUNCTION labviz_reject_immutable_update()"
        )


def downgrade() -> None:
    for table_name in PHASE4_IMMUTABLE_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_immutable ON {table_name}")

    op.execute(
        "DROP TRIGGER IF EXISTS trg_project_lifecycle_events_guard " "ON project_lifecycle_events"
    )
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_lifecycle_event_update()")
    op.execute("DROP TRIGGER IF EXISTS trg_project_origins_guard ON project_origins")
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_project_origin_update()")
    op.execute("DROP TRIGGER IF EXISTS trg_project_claims_guard ON project_claims")
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_project_claim_update()")

    op.drop_index("ix_idempotency_records_created", table_name="idempotency_records")
    op.drop_table("idempotency_records")
    op.drop_index(
        "ix_project_lifecycle_events_project_created", table_name="project_lifecycle_events"
    )
    op.drop_table("project_lifecycle_events")
    op.drop_table("project_origins")
    op.drop_table("project_claims")

    op.drop_constraint(
        op.f("ck_processing_runs_reused_result_origin"),
        "processing_runs",
        type_="check",
    )
    op.drop_constraint(op.f("ck_processing_runs_execution_mode"), "processing_runs", type_="check")
    op.drop_constraint(
        "fk_processing_runs_origin_processing_run_id_processing_runs",
        "processing_runs",
        type_="foreignkey",
    )
    op.drop_column("processing_runs", "origin_run_uuid_snapshot")
    op.drop_column("processing_runs", "origin_processing_run_id")
    op.drop_column("processing_runs", "execution_mode")

    op.create_unique_constraint(
        "uq_dataset_versions_stored_object_id", "dataset_versions", ["stored_object_id"]
    )
    op.drop_index("ix_stored_objects_dedup_lookup", table_name="stored_objects")
    op.drop_index("ix_stored_objects_gc_candidate", table_name="stored_objects")
    op.drop_constraint(
        op.f("ck_stored_objects_format_contract_version_nonempty"),
        "stored_objects",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_stored_objects_dedup_scope_nonempty"),
        "stored_objects",
        type_="check",
    )
    op.drop_column("stored_objects", "gc_candidate_at")
    op.drop_column("stored_objects", "format_contract_version")
    op.drop_column("stored_objects", "dedup_scope")

    op.drop_index("ix_auth_requests_email_time", table_name="auth_requests")
    op.drop_index("ix_auth_requests_client_time", table_name="auth_requests")
    op.drop_table("auth_requests")
    op.drop_index("ix_auth_sessions_expires", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_expires", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("ix_auth_challenges_expires", table_name="auth_challenges")
    op.drop_index("ix_auth_challenges_email_created", table_name="auth_challenges")
    op.drop_table("auth_challenges")

    op.add_column("projects", sa.Column("guest_token_digest", sa.String(64), nullable=True))
    op.execute(
        """
        UPDATE projects AS project
        SET guest_token_digest = guest.token_digest
        FROM guest_sessions AS guest
        WHERE project.guest_session_id = guest.id
        """
    )
    op.drop_constraint(op.f("ck_projects_ownership_by_storage_mode"), "projects", type_="check")
    op.create_check_constraint(
        "saved_project_owner",
        "projects",
        "storage_mode <> 'saved-cloud' OR owner_user_id IS NOT NULL",
    )
    op.create_check_constraint(
        "temporary_project_expiry",
        "projects",
        "storage_mode <> 'temporary-cloud' OR expires_at IS NOT NULL",
    )
    op.drop_constraint(
        "fk_projects_guest_session_id_guest_sessions", "projects", type_="foreignkey"
    )
    op.drop_column("projects", "lock_version")
    op.drop_column("projects", "saved_at")
    op.drop_column("projects", "guest_session_id")
    op.drop_index("ix_guest_sessions_status_expires", table_name="guest_sessions")
    op.drop_table("guest_sessions")
