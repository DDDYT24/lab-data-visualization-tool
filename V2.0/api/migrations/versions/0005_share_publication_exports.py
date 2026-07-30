"""Add revision-pinned sharing and permanent publication exports."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_share_publication_exports"
down_revision: str | Sequence[str] | None = "0004_identity_project_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSON_DOCUMENT = sa.JSON().with_variant(postgresql.JSONB(none_as_null=True), "postgresql")


def upgrade() -> None:
    op.add_column("idempotency_records", sa.Column("guest_session_id", UUID, nullable=True))
    op.add_column("idempotency_records", sa.Column("request_sha256", sa.String(64), nullable=True))
    op.add_column(
        "idempotency_records",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.alter_column("idempotency_records", "actor_user_id", nullable=True)
    op.create_foreign_key(
        "fk_idempotency_records_guest_session_id_guest_sessions",
        "idempotency_records",
        "guest_sessions",
        ["guest_session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_idempotency_guest_operation",
        "idempotency_records",
        ["guest_session_id", "operation", "idempotency_key"],
    )
    op.create_check_constraint(
        "ck_idempotency_records_exactly_one_actor",
        "idempotency_records",
        "((actor_user_id IS NOT NULL)::int + (guest_session_id IS NOT NULL)::int) = 1",
    )
    op.create_check_constraint(
        "ck_idempotency_records_request_sha256_lower_hex",
        "idempotency_records",
        "request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_index("ix_idempotency_records_expires", "idempotency_records", ["expires_at"])

    op.create_table(
        "share_links",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("project_revision_id", UUID, nullable=False),
        sa.Column("created_by_user_id", UUID, nullable=True),
        sa.Column("revoked_by_user_id", UUID, nullable=True),
        sa.Column("token_digest", sa.String(64), nullable=False),
        sa.Column("token_key_version", sa.Integer(), nullable=False),
        sa.Column("downloads_enabled", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > created_at",
            name="ck_share_links_expiry_after_create",
        ),
        sa.CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL AND revoked_by_user_id IS NULL) OR "
            "(status = 'revoked' AND revoked_at IS NOT NULL)",
            name="ck_share_links_revocation_state",
        ),
        sa.CheckConstraint("status IN ('active', 'revoked')", name="ck_share_links_status"),
        sa.CheckConstraint(
            "token_digest ~ '^[0-9a-f]{64}$'",
            name="ck_share_links_token_digest_lower_hex",
        ),
        sa.CheckConstraint(
            "token_key_version >= 1", name="ck_share_links_token_key_version_positive"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_share_links_created_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_share_links_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_revision_id", "project_id"],
            ["project_revisions.id", "project_revisions.project_id"],
            name="fk_share_links_revision_same_project",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["revoked_by_user_id"],
            ["users.id"],
            name="fk_share_links_revoked_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_share_links"),
        sa.UniqueConstraint(
            "id",
            "project_id",
            "project_revision_id",
            name="uq_share_links_identity_scope",
        ),
        sa.UniqueConstraint("token_digest", name="uq_share_links_token_digest"),
    )
    op.create_index("ix_share_links_project_created", "share_links", ["project_id", "created_at"])
    op.create_index("ix_share_links_status_expires", "share_links", ["status", "expires_at"])

    op.create_table(
        "share_link_events",
        sa.Column("id", UUID, nullable=False),
        sa.Column("share_link_id", UUID, nullable=True),
        sa.Column("share_uuid_snapshot", UUID, nullable=False),
        sa.Column("project_uuid_snapshot", UUID, nullable=False),
        sa.Column("project_revision_uuid_snapshot", UUID, nullable=False),
        sa.Column("actor_user_id", UUID, nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("details", JSON_DOCUMENT, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('create', 'downloads-update', 'revoke')",
            name="ck_share_link_events_event_type",
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name="fk_share_link_events_actor_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["share_link_id"],
            ["share_links.id"],
            name="fk_share_link_events_share_link_id_share_links",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_share_link_events"),
    )
    op.create_index(
        "ix_share_link_events_share_created",
        "share_link_events",
        ["share_uuid_snapshot", "created_at"],
    )

    op.create_table(
        "export_jobs",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("project_revision_id", UUID, nullable=False),
        sa.Column("requested_by_user_id", UUID, nullable=True),
        sa.Column("guest_session_id", UUID, nullable=True),
        sa.Column("current_processing_run_id", UUID, nullable=True),
        sa.Column("pending_stored_object_id", UUID, nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("attempt_count >= 1", name="ck_export_jobs_attempt_count_positive"),
        sa.CheckConstraint(
            "((requested_by_user_id IS NOT NULL)::int + (guest_session_id IS NOT NULL)::int) = 1",
            name="ck_export_jobs_exactly_one_actor",
        ),
        sa.CheckConstraint("format IN ('png', 'svg', 'pdf')", name="ck_export_jobs_format"),
        sa.CheckConstraint(
            "status <> 'ready' OR pending_stored_object_id IS NULL",
            name="ck_export_jobs_ready_has_no_pending_object",
        ),
        sa.CheckConstraint(
            "request_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_export_jobs_request_sha256_lower_hex",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'rendering', 'ready', 'failed')",
            name="ck_export_jobs_status",
        ),
        sa.ForeignKeyConstraint(
            ["current_processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_export_jobs_run_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["guest_session_id"],
            ["guest_sessions.id"],
            name="fk_export_jobs_guest_session_id_guest_sessions",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["pending_stored_object_id"],
            ["stored_objects.id"],
            name="fk_export_jobs_pending_stored_object_id_stored_objects",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_export_jobs_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_revision_id", "project_id"],
            ["project_revisions.id", "project_revisions.project_id"],
            name="fk_export_jobs_revision_same_project",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"],
            ["users.id"],
            name="fk_export_jobs_requested_by_user_id_users",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_export_jobs"),
        sa.UniqueConstraint("id", "project_id", name="uq_export_jobs_id_project"),
        sa.UniqueConstraint(
            "id",
            "project_id",
            "project_revision_id",
            "format",
            name="uq_export_jobs_id_project_revision_format",
        ),
    )
    op.create_index("ix_export_jobs_project_created", "export_jobs", ["project_id", "created_at"])
    op.create_index("ix_export_jobs_status_created", "export_jobs", ["status", "created_at"])

    op.create_table(
        "publication_exports",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("project_revision_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("cleaning_decision_set_id", UUID, nullable=True),
        sa.Column("chart_spec_revision_id", UUID, nullable=False),
        sa.Column("processing_run_id", UUID, nullable=False),
        sa.Column("stored_object_id", UUID, nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column("renderer_name", sa.String(128), nullable=False),
        sa.Column("renderer_version", sa.String(128), nullable=False),
        sa.Column("render_contract_version", sa.String(128), nullable=False),
        sa.Column("render_spec_document", JSON_DOCUMENT, nullable=False),
        sa.Column("size_preset", sa.String(32), nullable=False),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
        sa.Column("unit", sa.String(8), nullable=False),
        sa.Column("dpi", sa.Integer(), nullable=False),
        sa.Column("output_sha256", sa.String(64), nullable=False),
        sa.Column("output_size_bytes", sa.Integer(), nullable=False),
        sa.Column("validation_document", JSON_DOCUMENT, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("dpi > 0", name="ck_publication_exports_dpi_positive"),
        sa.CheckConstraint("format IN ('png', 'svg', 'pdf')", name="ck_publication_exports_format"),
        sa.CheckConstraint(
            "output_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_publication_exports_output_sha256_lower_hex",
        ),
        sa.CheckConstraint(
            "output_size_bytes >= 0", name="ck_publication_exports_output_size_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["id", "project_id", "project_revision_id", "format"],
            [
                "export_jobs.id",
                "export_jobs.project_id",
                "export_jobs.project_revision_id",
                "export_jobs.format",
            ],
            name="fk_publication_exports_export_job_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chart_spec_revision_id", "project_id"],
            ["chart_spec_revisions.id", "chart_spec_revisions.project_id"],
            name="fk_publication_exports_chart_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["cleaning_decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_publication_exports_decision_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_publication_exports_dataset_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_publication_exports_run_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_publication_exports_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.id"],
            name="fk_publication_exports_stored_object_id_stored_objects",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_publication_exports"),
        sa.UniqueConstraint(
            "id",
            "project_id",
            "project_revision_id",
            "format",
            name="uq_publication_exports_binding_scope",
        ),
    )
    op.create_index(
        "ix_publication_exports_revision_format",
        "publication_exports",
        ["project_revision_id", "format"],
    )

    op.create_table(
        "stored_object_write_intents",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("export_job_id", UUID, nullable=False),
        sa.Column("stored_object_id", UUID, nullable=False),
        sa.Column("operation", sa.String(24), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL)",
            name="ck_stored_object_write_intents_completion_state",
        ),
        sa.CheckConstraint("operation = 'export'", name="ck_stored_object_write_intents_operation"),
        sa.CheckConstraint(
            "status IN ('pending', 'completed')",
            name="ck_stored_object_write_intents_status",
        ),
        sa.ForeignKeyConstraint(
            ["export_job_id", "project_id"],
            ["export_jobs.id", "export_jobs.project_id"],
            name="fk_stored_object_write_intents_export_job_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.id"],
            name="fk_stored_object_write_intents_stored_object_id_stored_objects",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_stored_object_write_intents"),
        sa.UniqueConstraint("export_job_id", name="uq_stored_object_write_intents_export_job_id"),
    )
    op.create_index(
        "ix_stored_object_write_intents_status_created",
        "stored_object_write_intents",
        ["status", "created_at"],
    )

    op.create_table(
        "share_export_bindings",
        sa.Column("share_link_id", UUID, nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("project_revision_id", UUID, nullable=False),
        sa.Column("publication_export_id", UUID, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "format IN ('png', 'svg', 'pdf')", name="ck_share_export_bindings_format"
        ),
        sa.ForeignKeyConstraint(
            ["publication_export_id", "project_id", "project_revision_id", "format"],
            [
                "publication_exports.id",
                "publication_exports.project_id",
                "publication_exports.project_revision_id",
                "publication_exports.format",
            ],
            name="fk_share_export_bindings_export_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["share_link_id", "project_id", "project_revision_id"],
            ["share_links.id", "share_links.project_id", "share_links.project_revision_id"],
            name="fk_share_export_bindings_share_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("share_link_id", "format", name="pk_share_export_bindings"),
    )
    op.create_index(
        "ix_share_export_bindings_export",
        "share_export_bindings",
        ["publication_export_id"],
    )

    op.execute(
        "CREATE TRIGGER trg_publication_exports_immutable "
        "BEFORE UPDATE ON publication_exports FOR EACH ROW "
        "EXECUTE FUNCTION labviz_reject_immutable_update()"
    )
    op.execute(
        "CREATE TRIGGER trg_share_export_bindings_immutable "
        "BEFORE UPDATE ON share_export_bindings FOR EACH ROW "
        "EXECUTE FUNCTION labviz_reject_immutable_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_guard_share_link_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.project_id IS DISTINCT FROM OLD.project_id
               OR NEW.project_revision_id IS DISTINCT FROM OLD.project_revision_id
               OR NEW.token_digest IS DISTINCT FROM OLD.token_digest
               OR NEW.token_key_version IS DISTINCT FROM OLD.token_key_version
               OR NEW.expires_at IS DISTINCT FROM OLD.expires_at
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR (OLD.status = 'revoked' AND NEW.status <> 'revoked')
               OR (NEW.created_by_user_id IS DISTINCT FROM OLD.created_by_user_id
                   AND NEW.created_by_user_id IS NOT NULL)
               OR (NEW.revoked_by_user_id IS DISTINCT FROM OLD.revoked_by_user_id
                   AND NEW.revoked_by_user_id IS NOT NULL
                   AND OLD.revoked_by_user_id IS NOT NULL)
            THEN
                RAISE EXCEPTION 'immutable ShareLink identity or revoked state changed';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_share_links_guard BEFORE UPDATE ON share_links "
        "FOR EACH ROW EXECUTE FUNCTION labviz_guard_share_link_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_guard_share_event_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.share_uuid_snapshot IS DISTINCT FROM OLD.share_uuid_snapshot
               OR NEW.project_uuid_snapshot IS DISTINCT FROM OLD.project_uuid_snapshot
               OR NEW.project_revision_uuid_snapshot
                  IS DISTINCT FROM OLD.project_revision_uuid_snapshot
               OR NEW.event_type IS DISTINCT FROM OLD.event_type
               OR NEW.details IS DISTINCT FROM OLD.details
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR (NEW.share_link_id IS DISTINCT FROM OLD.share_link_id
                   AND NEW.share_link_id IS NOT NULL)
               OR (NEW.actor_user_id IS DISTINCT FROM OLD.actor_user_id
                   AND NEW.actor_user_id IS NOT NULL)
            THEN
                RAISE EXCEPTION 'share_link_events is immutable except for FK nullification';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_share_link_events_guard BEFORE UPDATE ON share_link_events "
        "FOR EACH ROW EXECUTE FUNCTION labviz_guard_share_event_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_guard_write_intent_update() RETURNS trigger AS $$
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
    op.execute(
        "CREATE TRIGGER trg_stored_object_write_intents_guard "
        "BEFORE UPDATE ON stored_object_write_intents FOR EACH ROW "
        "EXECUTE FUNCTION labviz_guard_write_intent_update()"
    )
    op.execute(
        """
        CREATE FUNCTION labviz_validate_publication_export() RETURNS trigger AS $$
        DECLARE
            revision project_revisions%ROWTYPE;
            stored stored_objects%ROWTYPE;
        BEGIN
            SELECT * INTO revision FROM project_revisions WHERE id = NEW.project_revision_id;
            IF revision.active_dataset_version_id IS DISTINCT FROM NEW.dataset_version_id
               OR revision.cleaning_decision_set_id
                  IS DISTINCT FROM NEW.cleaning_decision_set_id
               OR revision.chart_spec_revision_id IS DISTINCT FROM NEW.chart_spec_revision_id
            THEN
                RAISE EXCEPTION 'PublicationExport lineage differs from ProjectRevision';
            END IF;
            SELECT * INTO stored FROM stored_objects WHERE id = NEW.stored_object_id;
            IF stored.status <> 'available'
               OR stored.sha256 IS DISTINCT FROM NEW.output_sha256
               OR stored.size_bytes IS DISTINCT FROM NEW.output_size_bytes
            THEN
                RAISE EXCEPTION 'PublicationExport StoredObject is not verified and available';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_publication_exports_validate "
        "BEFORE INSERT ON publication_exports FOR EACH ROW "
        "EXECUTE FUNCTION labviz_validate_publication_export()"
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM publication_exports)
               OR EXISTS (
                   SELECT 1 FROM stored_object_write_intents WHERE status <> 'completed'
               )
               OR EXISTS (
                   SELECT 1 FROM export_jobs
                   WHERE pending_stored_object_id IS NOT NULL
                      OR status IN ('queued', 'rendering')
               )
               OR EXISTS (
                   SELECT 1 FROM stored_objects
                   WHERE purpose = 'export' AND status <> 'deleted'
               )
               OR EXISTS (
                   SELECT 1 FROM idempotency_records
                   WHERE operation = 'publication-export'
               )
            THEN
                RAISE EXCEPTION
                    'Phase 5A downgrade refused: publication exports or tracked export writes exist'
                    USING HINT =
                        'Inventory publication artifacts and explicitly complete or clean their '
                        'database and object-storage lifecycle before downgrading to 0004.';
            END IF;
        END;
        $$
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_publication_exports_validate ON publication_exports")
    op.execute("DROP FUNCTION IF EXISTS labviz_validate_publication_export()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_stored_object_write_intents_guard "
        "ON stored_object_write_intents"
    )
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_write_intent_update()")
    op.execute("DROP TRIGGER IF EXISTS trg_share_link_events_guard ON share_link_events")
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_share_event_update()")
    op.execute("DROP TRIGGER IF EXISTS trg_share_links_guard ON share_links")
    op.execute("DROP FUNCTION IF EXISTS labviz_guard_share_link_update()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_share_export_bindings_immutable ON share_export_bindings"
    )
    op.execute("DROP TRIGGER IF EXISTS trg_publication_exports_immutable ON publication_exports")

    op.drop_index("ix_share_export_bindings_export", table_name="share_export_bindings")
    op.drop_table("share_export_bindings")
    op.drop_index(
        "ix_stored_object_write_intents_status_created",
        table_name="stored_object_write_intents",
    )
    op.drop_table("stored_object_write_intents")
    op.drop_index("ix_publication_exports_revision_format", table_name="publication_exports")
    op.drop_table("publication_exports")
    op.drop_index("ix_export_jobs_status_created", table_name="export_jobs")
    op.drop_index("ix_export_jobs_project_created", table_name="export_jobs")
    op.drop_table("export_jobs")
    op.drop_index("ix_share_link_events_share_created", table_name="share_link_events")
    op.drop_table("share_link_events")
    op.drop_index("ix_share_links_status_expires", table_name="share_links")
    op.drop_index("ix_share_links_project_created", table_name="share_links")
    op.drop_table("share_links")

    op.drop_index("ix_idempotency_records_expires", table_name="idempotency_records")
    op.drop_constraint(
        "ck_idempotency_records_request_sha256_lower_hex",
        "idempotency_records",
        type_="check",
    )
    op.drop_constraint(
        "ck_idempotency_records_exactly_one_actor",
        "idempotency_records",
        type_="check",
    )
    op.drop_constraint("uq_idempotency_guest_operation", "idempotency_records", type_="unique")
    op.drop_constraint(
        "fk_idempotency_records_guest_session_id_guest_sessions",
        "idempotency_records",
        type_="foreignkey",
    )
    op.alter_column("idempotency_records", "actor_user_id", nullable=False)
    op.drop_column("idempotency_records", "expires_at")
    op.drop_column("idempotency_records", "request_sha256")
    op.drop_column("idempotency_records", "guest_session_id")
