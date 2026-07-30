"""First production metadata and lineage entities from DATABASE_DESIGN.md."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

JSON_DOCUMENT = JSON().with_variant(JSONB(none_as_null=True), "postgresql")


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-created rows."""

    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="email_normalized"),
        CheckConstraint("length(email) BETWEEN 3 AND 320", name="email_length"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    projects: Mapped[list[Project]] = relationship(back_populates="owner")
    created_project_revisions: Mapped[list[ProjectRevision]] = relationship(
        back_populates="created_by",
        foreign_keys="ProjectRevision.created_by_user_id",
    )
    created_chart_revisions: Mapped[list[ChartSpecRevision]] = relationship(
        back_populates="created_by",
        foreign_keys="ChartSpecRevision.created_by_user_id",
    )


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (
        ForeignKeyConstraint(
            ["current_revision_id", "id"],
            ["project_revisions.id", "project_revisions.project_id"],
            name="fk_projects_current_revision_same_project",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint(
            "storage_mode IN ('temporary-cloud', 'saved-cloud', 'local')",
            name="storage_mode",
        ),
        CheckConstraint(
            "storage_mode <> 'saved-cloud' OR owner_user_id IS NOT NULL",
            name="saved_project_owner",
        ),
        CheckConstraint(
            "storage_mode <> 'temporary-cloud' OR expires_at IS NOT NULL",
            name="temporary_project_expiry",
        ),
        CheckConstraint("length(title) BETWEEN 1 AND 200", name="title_length"),
        CheckConstraint(
            "(deleted_at IS NULL AND purge_after IS NULL) OR "
            "(storage_mode = 'saved-cloud' AND deleted_at IS NOT NULL AND "
            "purge_after = deleted_at + INTERVAL '24 hours')",
            name="deletion_window",
        ),
        Index("ix_projects_owner_updated", "owner_user_id", "updated_at"),
        Index("ix_projects_purge_after", "purge_after"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    current_revision_id: Mapped[UUID | None] = mapped_column(nullable=True)
    guest_token_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purge_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    owner: Mapped[User | None] = relationship(back_populates="projects")
    current_revision: Mapped[ProjectRevision | None] = relationship(
        foreign_keys=[current_revision_id],
        post_update=True,
    )
    revisions: Mapped[list[ProjectRevision]] = relationship(
        back_populates="project",
        foreign_keys="ProjectRevision.project_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    source_files: Mapped[list[SourceFile]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    dataset_versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    processing_runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    chart_revisions: Mapped[list[ChartSpecRevision]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class StoredObject(Base):
    __tablename__ = "stored_objects"
    __table_args__ = (
        CheckConstraint("purpose IN ('source-upload', 'dataset', 'export')", name="purpose"),
        CheckConstraint("status IN ('pending', 'available', 'deleting', 'deleted')", name="status"),
        CheckConstraint("size_bytes >= 0", name="size_nonnegative"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_lower_hex"),
        CheckConstraint("length(object_key) > 0", name="object_key_nonempty"),
        CheckConstraint(
            "(status = 'deleted' AND deleted_at IS NOT NULL) OR "
            "(status <> 'deleted' AND deleted_at IS NULL)",
            name="deleted_status_time",
        ),
        CheckConstraint(
            "(status = 'pending' AND staging_key IS NOT NULL) OR "
            "(status <> 'pending' AND staging_key IS NULL)",
            name="pending_staging_key",
        ),
        Index("ix_stored_objects_status_expires", "status", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    staging_key: Mapped[str | None] = mapped_column(String(1024), nullable=True, unique=True)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    encryption_key_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    source_file: Mapped[SourceFile | None] = relationship(back_populates="stored_object")
    dataset_version: Mapped[DatasetVersion | None] = relationship(back_populates="stored_object")


class SourceFile(Base):
    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint("id", "project_id", name="uq_source_files_id_project"),
        CheckConstraint("size_bytes >= 0", name="size_nonnegative"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_lower_hex"),
        CheckConstraint("header_row IS NULL OR header_row >= 1", name="header_row_positive"),
        CheckConstraint("length(original_name) > 0", name="original_name_nonempty"),
        Index("ix_source_files_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    stored_object_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    original_name: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    sheet_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    available_sheets: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False, default=list)
    header_row: Mapped[int | None] = mapped_column(nullable=True)
    parser_name: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    binary_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="source_files")
    stored_object: Mapped[StoredObject | None] = relationship(back_populates="source_file")
    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="source_file",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Dataset.source_file_id",
    )


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_file_id", "project_id"],
            ["source_files.id", "source_files.project_id"],
            name="fk_datasets_source_same_project",
            ondelete="CASCADE",
        ),
        UniqueConstraint("id", "project_id", name="uq_datasets_id_project"),
        CheckConstraint("length(name) BETWEEN 1 AND 200", name="name_length"),
        CheckConstraint("header_row IS NULL OR header_row >= 1", name="header_row_positive"),
        Index("ix_datasets_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_file_id: Mapped[UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    sheet_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    header_row: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="datasets")
    source_file: Mapped[SourceFile] = relationship(
        back_populates="datasets", foreign_keys=[source_file_id]
    )
    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="DatasetVersion.dataset_id",
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id", "project_id"],
            ["datasets.id", "datasets.project_id"],
            name="fk_dataset_versions_dataset_same_project",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["parent_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_dataset_versions_parent_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_dataset_versions_id_project"),
        UniqueConstraint(
            "dataset_id", "version_number", name="uq_dataset_versions_dataset_version"
        ),
        CheckConstraint("version_number >= 1", name="version_positive"),
        CheckConstraint("kind IN ('parsed', 'cleaned', 'derived')", name="kind"),
        CheckConstraint("row_count >= 0", name="row_count_nonnegative"),
        CheckConstraint("column_count >= 1", name="column_count_positive"),
        CheckConstraint(
            "parent_version_id IS NULL OR parent_version_id <> id", name="parent_not_self"
        ),
        CheckConstraint("parquet_schema_version = 1", name="parquet_schema_v1"),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'", name="content_sha256_lower_hex"),
        Index("ix_dataset_versions_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[UUID] = mapped_column(nullable=False)
    parent_version_id: Mapped[UUID | None] = mapped_column(nullable=True)
    stored_object_id: Mapped[UUID] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    version_number: Mapped[int] = mapped_column(nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    schema_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    preview_document: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=dict
    )
    quality_document: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=dict
    )
    parquet_schema_version: Mapped[int] = mapped_column(nullable=False, default=1)
    content_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, default=lambda: "0" * 64
    )
    row_count: Mapped[int] = mapped_column(nullable=False)
    column_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    dataset: Mapped[Dataset] = relationship(back_populates="versions", foreign_keys=[dataset_id])
    project: Mapped[Project] = relationship(back_populates="dataset_versions")
    parent_version: Mapped[DatasetVersion | None] = relationship(
        remote_side=[id], foreign_keys=[parent_version_id]
    )
    stored_object: Mapped[StoredObject] = relationship(back_populates="dataset_version")
    input_to_runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="input_dataset_version",
        foreign_keys="ProcessingRun.input_dataset_version_id",
    )
    output_of_run: Mapped[ProcessingRun | None] = relationship(
        back_populates="output_dataset_version",
        foreign_keys="ProcessingRun.output_dataset_version_id",
    )


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["input_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_processing_runs_input_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["output_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_processing_runs_output_same_project",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "operation IN ('parse', 'profile', 'clean', 'analyze', 'export')", name="operation"
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')", name="status"
        ),
        CheckConstraint(
            "finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)",
            name="finish_after_start",
        ),
        CheckConstraint(
            "output_dataset_version_id IS NULL OR status = 'succeeded'", name="output_succeeded"
        ),
        UniqueConstraint("output_dataset_version_id", name="uq_processing_runs_output_version"),
        Index("ix_processing_runs_project_created", "project_id", "created_at"),
        Index("ix_processing_runs_status_created", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    input_dataset_version_id: Mapped[UUID | None] = mapped_column(nullable=True)
    output_dataset_version_id: Mapped[UUID | None] = mapped_column(nullable=True)
    operation: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    algorithm_version: Mapped[str] = mapped_column(String(128), nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="processing_runs")
    input_dataset_version: Mapped[DatasetVersion | None] = relationship(
        back_populates="input_to_runs", foreign_keys=[input_dataset_version_id]
    )
    output_dataset_version: Mapped[DatasetVersion | None] = relationship(
        back_populates="output_of_run", foreign_keys=[output_dataset_version_id]
    )


class ChartSpecRevision(Base):
    __tablename__ = "chart_spec_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_chart_spec_revisions_dataset_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_chart_spec_revisions_id_project"),
        UniqueConstraint(
            "project_id", "revision_number", name="uq_chart_spec_revisions_project_revision"
        ),
        CheckConstraint("revision_number >= 1", name="revision_positive"),
        CheckConstraint("schema_version >= 1", name="schema_version_positive"),
        CheckConstraint(
            "decision_set_revision IS NULL OR decision_set_revision >= 1",
            name="decision_revision_positive",
        ),
        Index("ix_chart_spec_revisions_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version_id: Mapped[UUID] = mapped_column(nullable=False)
    created_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    revision_number: Mapped[int] = mapped_column(nullable=False)
    schema_version: Mapped[int] = mapped_column(nullable=False)
    decision_set_revision: Mapped[int | None] = mapped_column(nullable=True)
    spec_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="chart_revisions")
    dataset_version: Mapped[DatasetVersion] = relationship(foreign_keys=[dataset_version_id])
    created_by: Mapped[User | None] = relationship(
        back_populates="created_chart_revisions", foreign_keys=[created_by_user_id]
    )


class ProjectRevision(Base):
    __tablename__ = "project_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["active_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_project_revisions_dataset_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["chart_spec_revision_id", "project_id"],
            ["chart_spec_revisions.id", "chart_spec_revisions.project_id"],
            name="fk_project_revisions_chart_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_project_revisions_id_project"),
        UniqueConstraint(
            "project_id", "revision_number", name="uq_project_revisions_project_revision"
        ),
        CheckConstraint("revision_number >= 1", name="revision_positive"),
        CheckConstraint("spec_schema_version >= 1", name="schema_version_positive"),
        Index("ix_project_revisions_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    active_dataset_version_id: Mapped[UUID] = mapped_column(nullable=False)
    chart_spec_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    created_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    revision_number: Mapped[int] = mapped_column(nullable=False)
    spec_schema_version: Mapped[int] = mapped_column(nullable=False)
    spec_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="revisions", foreign_keys=[project_id])
    active_dataset_version: Mapped[DatasetVersion] = relationship(
        foreign_keys=[active_dataset_version_id]
    )
    chart_spec_revision: Mapped[ChartSpecRevision] = relationship(
        foreign_keys=[chart_spec_revision_id]
    )
    created_by: Mapped[User | None] = relationship(
        back_populates="created_project_revisions", foreign_keys=[created_by_user_id]
    )
