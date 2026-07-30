"""Add fields required by the Phase 2 project persistence slice.

Revision ID: 0002_phase2_persistence
Revises: 0001_core_foundation
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_phase2_persistence"
down_revision: str | Sequence[str] | None = "0001_core_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(none_as_null=True)


def upgrade() -> None:
    op.add_column("projects", sa.Column("guest_token_digest", sa.String(64), nullable=True))
    op.add_column("stored_objects", sa.Column("staging_key", sa.String(1024), nullable=True))
    op.create_unique_constraint("uq_stored_objects_staging_key", "stored_objects", ["staging_key"])
    op.create_check_constraint(
        "pending_staging_key",
        "stored_objects",
        "(status = 'pending' AND staging_key IS NOT NULL) OR "
        "(status <> 'pending' AND staging_key IS NULL)",
    )

    op.add_column("source_files", sa.Column("available_sheets", JSONB, nullable=True))
    op.execute("UPDATE source_files SET available_sheets = '[]'::jsonb")
    op.alter_column("source_files", "available_sheets", nullable=False)

    op.add_column("dataset_versions", sa.Column("preview_document", JSONB, nullable=True))
    op.add_column("dataset_versions", sa.Column("quality_document", JSONB, nullable=True))
    op.add_column(
        "dataset_versions", sa.Column("parquet_schema_version", sa.Integer(), nullable=True)
    )
    op.add_column("dataset_versions", sa.Column("content_sha256", sa.String(64), nullable=True))
    op.execute("UPDATE dataset_versions SET preview_document = '{}'::jsonb")
    op.execute("UPDATE dataset_versions SET quality_document = '{}'::jsonb")
    op.execute("UPDATE dataset_versions SET parquet_schema_version = 1")
    op.execute("UPDATE dataset_versions SET content_sha256 = repeat('0', 64)")
    for column in (
        "preview_document",
        "quality_document",
        "parquet_schema_version",
        "content_sha256",
    ):
        op.alter_column("dataset_versions", column, nullable=False)
    op.create_check_constraint(
        "parquet_schema_v1",
        "dataset_versions",
        "parquet_schema_version = 1",
    )
    op.create_check_constraint(
        "content_sha256_lower_hex",
        "dataset_versions",
        "content_sha256 ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS "
        "ck_dataset_versions_ck_dataset_versions_content_sha256_lower_hex"
    )
    op.execute(
        "ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS "
        "ck_dataset_versions_content_sha256_lower_hex"
    )
    op.execute(
        "ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS "
        "ck_dataset_versions_ck_dataset_versions_parquet_schema_v1"
    )
    op.execute(
        "ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS "
        "ck_dataset_versions_parquet_schema_v1"
    )
    op.drop_column("dataset_versions", "content_sha256")
    op.drop_column("dataset_versions", "parquet_schema_version")
    op.drop_column("dataset_versions", "quality_document")
    op.drop_column("dataset_versions", "preview_document")
    op.drop_column("source_files", "available_sheets")
    op.execute(
        "ALTER TABLE stored_objects DROP CONSTRAINT IF EXISTS "
        "ck_stored_objects_ck_stored_objects_pending_staging_key"
    )
    op.execute(
        "ALTER TABLE stored_objects DROP CONSTRAINT IF EXISTS ck_stored_objects_pending_staging_key"
    )
    op.drop_constraint("uq_stored_objects_staging_key", "stored_objects", type_="unique")
    op.drop_column("stored_objects", "staging_key")
    op.drop_column("projects", "guest_token_digest")
