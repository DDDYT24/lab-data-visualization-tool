"""Create the first production metadata and lineage tables.

Revision ID: 0001_core_foundation
Revises:
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_core_foundation"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = sa.Uuid()
TIMESTAMP = sa.DateTime(timezone=True)
JSONB = postgresql.JSONB(none_as_null=True)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint("email = lower(email)", name=op.f("ck_users_email_normalized")),
        sa.CheckConstraint("length(email) BETWEEN 3 AND 320", name=op.f("ck_users_email_length")),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "stored_objects",
        sa.Column("id", UUID, nullable=False),
        sa.Column("storage_backend", sa.String(length=64), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("encryption_key_id", sa.String(length=512), nullable=True),
        sa.Column("expires_at", TIMESTAMP, nullable=True),
        sa.Column("deleted_at", TIMESTAMP, nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "purpose IN ('source-upload', 'dataset', 'export')",
            name=op.f("ck_stored_objects_purpose"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'available', 'deleting', 'deleted')",
            name=op.f("ck_stored_objects_status"),
        ),
        sa.CheckConstraint("size_bytes >= 0", name=op.f("ck_stored_objects_size_nonnegative")),
        sa.CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_stored_objects_sha256_lower_hex")
        ),
        sa.CheckConstraint(
            "length(object_key) > 0", name=op.f("ck_stored_objects_object_key_nonempty")
        ),
        sa.CheckConstraint(
            "(status = 'deleted' AND deleted_at IS NOT NULL) OR "
            "(status <> 'deleted' AND deleted_at IS NULL)",
            name=op.f("ck_stored_objects_deleted_status_time"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_stored_objects"),
        sa.UniqueConstraint("object_key", name="uq_stored_objects_object_key"),
    )
    op.create_index(
        "ix_stored_objects_status_expires",
        "stored_objects",
        ["status", "expires_at"],
    )

    op.create_table(
        "projects",
        sa.Column("id", UUID, nullable=False),
        sa.Column("owner_user_id", UUID, nullable=True),
        sa.Column("current_revision_id", UUID, nullable=True),
        sa.Column("storage_mode", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("last_activity_at", TIMESTAMP, nullable=False),
        sa.Column("expires_at", TIMESTAMP, nullable=True),
        sa.Column("deleted_at", TIMESTAMP, nullable=True),
        sa.Column("purge_after", TIMESTAMP, nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "storage_mode IN ('temporary-cloud', 'saved-cloud', 'local')",
            name=op.f("ck_projects_storage_mode"),
        ),
        sa.CheckConstraint(
            "storage_mode <> 'saved-cloud' OR owner_user_id IS NOT NULL",
            name=op.f("ck_projects_saved_project_owner"),
        ),
        sa.CheckConstraint(
            "storage_mode <> 'temporary-cloud' OR expires_at IS NOT NULL",
            name=op.f("ck_projects_temporary_project_expiry"),
        ),
        sa.CheckConstraint(
            "length(title) BETWEEN 1 AND 200", name=op.f("ck_projects_title_length")
        ),
        sa.CheckConstraint(
            "(deleted_at IS NULL AND purge_after IS NULL) OR "
            "(storage_mode = 'saved-cloud' AND deleted_at IS NOT NULL AND "
            "purge_after = deleted_at + INTERVAL '24 hours')",
            name=op.f("ck_projects_deletion_window"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.id"],
            name="fk_projects_owner_user_id_users",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_projects"),
    )
    op.create_index("ix_projects_owner_updated", "projects", ["owner_user_id", "updated_at"])
    op.create_index("ix_projects_purge_after", "projects", ["purge_after"])

    op.create_table(
        "source_files",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("stored_object_id", UUID, nullable=True),
        sa.Column("original_name", sa.String(length=512), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("sheet_name", sa.String(length=255), nullable=True),
        sa.Column("header_row", sa.Integer(), nullable=True),
        sa.Column("parser_name", sa.String(length=128), nullable=False),
        sa.Column("parser_version", sa.String(length=128), nullable=False),
        sa.Column("binary_deleted_at", TIMESTAMP, nullable=True),
        sa.Column("parsed_at", TIMESTAMP, nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint("size_bytes >= 0", name=op.f("ck_source_files_size_nonnegative")),
        sa.CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_source_files_sha256_lower_hex")
        ),
        sa.CheckConstraint(
            "header_row IS NULL OR header_row >= 1",
            name=op.f("ck_source_files_header_row_positive"),
        ),
        sa.CheckConstraint(
            "length(original_name) > 0", name=op.f("ck_source_files_original_name_nonempty")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_source_files_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.id"],
            name="fk_source_files_stored_object_id_stored_objects",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_files"),
        sa.UniqueConstraint("id", "project_id", name="uq_source_files_id_project"),
        sa.UniqueConstraint("stored_object_id", name="uq_source_files_stored_object_id"),
    )
    op.create_index("ix_source_files_project_created", "source_files", ["project_id", "created_at"])

    op.create_table(
        "datasets",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("source_file_id", UUID, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("sheet_name", sa.String(length=255), nullable=True),
        sa.Column("header_row", sa.Integer(), nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint("length(name) BETWEEN 1 AND 200", name=op.f("ck_datasets_name_length")),
        sa.CheckConstraint(
            "header_row IS NULL OR header_row >= 1", name=op.f("ck_datasets_header_row_positive")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_datasets_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_file_id", "project_id"],
            ["source_files.id", "source_files.project_id"],
            name="fk_datasets_source_same_project",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
        sa.UniqueConstraint("id", "project_id", name="uq_datasets_id_project"),
    )
    op.create_index("ix_datasets_project_created", "datasets", ["project_id", "created_at"])

    op.create_table(
        "dataset_versions",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("parent_version_id", UUID, nullable=True),
        sa.Column("stored_object_id", UUID, nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("schema_document", JSONB, nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("column_count", sa.Integer(), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "version_number >= 1", name=op.f("ck_dataset_versions_version_positive")
        ),
        sa.CheckConstraint(
            "kind IN ('parsed', 'cleaned', 'derived')", name=op.f("ck_dataset_versions_kind")
        ),
        sa.CheckConstraint(
            "row_count >= 0", name=op.f("ck_dataset_versions_row_count_nonnegative")
        ),
        sa.CheckConstraint(
            "column_count >= 1", name=op.f("ck_dataset_versions_column_count_positive")
        ),
        sa.CheckConstraint(
            "parent_version_id IS NULL OR parent_version_id <> id",
            name=op.f("ck_dataset_versions_parent_not_self"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_dataset_versions_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id", "project_id"],
            ["datasets.id", "datasets.project_id"],
            name="fk_dataset_versions_dataset_same_project",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_dataset_versions_parent_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["stored_object_id"],
            ["stored_objects.id"],
            name="fk_dataset_versions_stored_object_id_stored_objects",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_dataset_versions"),
        sa.UniqueConstraint("id", "project_id", name="uq_dataset_versions_id_project"),
        sa.UniqueConstraint(
            "dataset_id", "version_number", name="uq_dataset_versions_dataset_version"
        ),
        sa.UniqueConstraint("stored_object_id", name="uq_dataset_versions_stored_object_id"),
    )
    op.create_index(
        "ix_dataset_versions_project_created",
        "dataset_versions",
        ["project_id", "created_at"],
    )

    op.create_table(
        "processing_runs",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("input_dataset_version_id", UUID, nullable=True),
        sa.Column("output_dataset_version_id", UUID, nullable=True),
        sa.Column("operation", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("parameters", JSONB, nullable=False),
        sa.Column("algorithm_version", sa.String(length=128), nullable=False),
        sa.Column("code_version", sa.String(length=128), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", TIMESTAMP, nullable=True),
        sa.Column("finished_at", TIMESTAMP, nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "operation IN ('parse', 'profile', 'clean', 'analyze', 'export')",
            name=op.f("ck_processing_runs_operation"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_processing_runs_status"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)",
            name=op.f("ck_processing_runs_finish_after_start"),
        ),
        sa.CheckConstraint(
            "output_dataset_version_id IS NULL OR status = 'succeeded'",
            name=op.f("ck_processing_runs_output_succeeded"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_processing_runs_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["input_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_processing_runs_input_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["output_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_processing_runs_output_same_project",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_processing_runs"),
        sa.UniqueConstraint("output_dataset_version_id", name="uq_processing_runs_output_version"),
    )
    op.create_index(
        "ix_processing_runs_project_created", "processing_runs", ["project_id", "created_at"]
    )
    op.create_index(
        "ix_processing_runs_status_created", "processing_runs", ["status", "created_at"]
    )

    op.create_table(
        "chart_spec_revisions",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("created_by_user_id", UUID, nullable=True),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("decision_set_revision", sa.Integer(), nullable=True),
        sa.Column("spec_document", JSONB, nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "revision_number >= 1", name=op.f("ck_chart_spec_revisions_revision_positive")
        ),
        sa.CheckConstraint(
            "schema_version >= 1", name=op.f("ck_chart_spec_revisions_schema_version_positive")
        ),
        sa.CheckConstraint(
            "decision_set_revision IS NULL OR decision_set_revision >= 1",
            name=op.f("ck_chart_spec_revisions_decision_revision_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_chart_spec_revisions_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_chart_spec_revisions_dataset_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_chart_spec_revisions_created_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chart_spec_revisions"),
        sa.UniqueConstraint("id", "project_id", name="uq_chart_spec_revisions_id_project"),
        sa.UniqueConstraint(
            "project_id", "revision_number", name="uq_chart_spec_revisions_project_revision"
        ),
    )
    op.create_index(
        "ix_chart_spec_revisions_project_created",
        "chart_spec_revisions",
        ["project_id", "created_at"],
    )

    op.create_table(
        "project_revisions",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("active_dataset_version_id", UUID, nullable=False),
        sa.Column("chart_spec_revision_id", UUID, nullable=False),
        sa.Column("created_by_user_id", UUID, nullable=True),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("spec_schema_version", sa.Integer(), nullable=False),
        sa.Column("spec_document", JSONB, nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "revision_number >= 1", name=op.f("ck_project_revisions_revision_positive")
        ),
        sa.CheckConstraint(
            "spec_schema_version >= 1", name=op.f("ck_project_revisions_schema_version_positive")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_project_revisions_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["active_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_project_revisions_dataset_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["chart_spec_revision_id", "project_id"],
            ["chart_spec_revisions.id", "chart_spec_revisions.project_id"],
            name="fk_project_revisions_chart_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_project_revisions_created_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_project_revisions"),
        sa.UniqueConstraint("id", "project_id", name="uq_project_revisions_id_project"),
        sa.UniqueConstraint(
            "project_id", "revision_number", name="uq_project_revisions_project_revision"
        ),
    )
    op.create_index(
        "ix_project_revisions_project_created",
        "project_revisions",
        ["project_id", "created_at"],
    )

    op.create_foreign_key(
        "fk_projects_current_revision_same_project",
        "projects",
        "project_revisions",
        ["current_revision_id", "id"],
        ["id", "project_id"],
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    op.drop_constraint("fk_projects_current_revision_same_project", "projects", type_="foreignkey")
    op.drop_index("ix_project_revisions_project_created", table_name="project_revisions")
    op.drop_table("project_revisions")
    op.drop_index("ix_chart_spec_revisions_project_created", table_name="chart_spec_revisions")
    op.drop_table("chart_spec_revisions")
    op.drop_index("ix_processing_runs_status_created", table_name="processing_runs")
    op.drop_index("ix_processing_runs_project_created", table_name="processing_runs")
    op.drop_table("processing_runs")
    op.drop_index("ix_dataset_versions_project_created", table_name="dataset_versions")
    op.drop_table("dataset_versions")
    op.drop_index("ix_datasets_project_created", table_name="datasets")
    op.drop_table("datasets")
    op.drop_index("ix_source_files_project_created", table_name="source_files")
    op.drop_table("source_files")
    op.drop_index("ix_projects_purge_after", table_name="projects")
    op.drop_index("ix_projects_owner_updated", table_name="projects")
    op.drop_table("projects")
    op.drop_index("ix_stored_objects_status_expires", table_name="stored_objects")
    op.drop_table("stored_objects")
    op.drop_table("users")
