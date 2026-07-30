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
    created_cleaning_decision_sets: Mapped[list[CleaningDecisionSet]] = relationship(
        back_populates="created_by",
        foreign_keys="CleaningDecisionSet.created_by_user_id",
    )
    auth_sessions: Mapped[list[AuthSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class GuestSession(Base):
    """Server-side anonymous browser identity; raw cookie tokens are never stored."""

    __tablename__ = "guest_sessions"
    __table_args__ = (
        CheckConstraint("token_digest ~ '^[0-9a-f]{64}$'", name="token_digest_lower_hex"),
        CheckConstraint("status IN ('active', 'revoked')", name="status"),
        CheckConstraint(
            "(status = 'revoked' AND revoked_at IS NOT NULL) OR "
            "(status = 'active' AND revoked_at IS NULL)",
            name="revoked_status_time",
        ),
        Index("ix_guest_sessions_status_expires", "status", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    token_digest: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    projects: Mapped[list[Project]] = relationship(back_populates="guest_session")


class AuthChallenge(Base):
    __tablename__ = "auth_challenges"
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="email_normalized"),
        CheckConstraint("failed_attempts >= 0", name="failed_attempts_nonnegative"),
        Index("ix_auth_challenges_email_created", "email", "created_at"),
        Index("ix_auth_challenges_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    code_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resend_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    failed_attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (
        CheckConstraint("token_digest ~ '^[0-9a-f]{64}$'", name="token_digest_lower_hex"),
        Index("ix_auth_sessions_user_expires", "user_id", "expires_at"),
        Index("ix_auth_sessions_expires", "expires_at"),
    )

    token_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    user: Mapped[User] = relationship(back_populates="auth_sessions")


class AuthRequest(Base):
    __tablename__ = "auth_requests"
    __table_args__ = (
        Index("ix_auth_requests_client_time", "client_key", "requested_at"),
        Index("ix_auth_requests_email_time", "email", "requested_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_key: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
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
            "(storage_mode = 'temporary-cloud' AND expires_at IS NOT NULL AND "
            "((owner_user_id IS NULL) <> (guest_session_id IS NULL))) OR "
            "(storage_mode = 'saved-cloud' AND owner_user_id IS NOT NULL AND "
            "guest_session_id IS NULL AND expires_at IS NULL) OR "
            "storage_mode = 'local'",
            name="ownership_by_storage_mode",
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
    guest_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="RESTRICT"), nullable=True
    )
    current_revision_id: Mapped[UUID | None] = mapped_column(nullable=True)
    storage_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purge_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    saved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lock_version: Mapped[int] = mapped_column(nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    owner: Mapped[User | None] = relationship(back_populates="projects")
    guest_session: Mapped[GuestSession | None] = relationship(back_populates="projects")
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
    quality_reports: Mapped[list[QualityReportRecord]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    cleaning_decision_sets: Mapped[list[CleaningDecisionSet]] = relationship(
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
        CheckConstraint("length(dedup_scope) > 0", name="dedup_scope_nonempty"),
        CheckConstraint(
            "length(format_contract_version) > 0", name="format_contract_version_nonempty"
        ),
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
        Index("ix_stored_objects_gc_candidate", "gc_candidate_at"),
        Index(
            "ix_stored_objects_dedup_lookup",
            "storage_backend",
            "dedup_scope",
            "purpose",
            "media_type",
            "format_contract_version",
            "sha256",
            "size_bytes",
            "encryption_key_id",
        ),
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
    dedup_scope: Mapped[str] = mapped_column(String(255), nullable=False, default="legacy")
    format_contract_version: Mapped[str] = mapped_column(
        String(128), nullable=False, default="unknown"
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    gc_candidate_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    source_file: Mapped[SourceFile | None] = relationship(back_populates="stored_object")
    dataset_versions: Mapped[list[DatasetVersion]] = relationship(back_populates="stored_object")


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
        ForeignKeyConstraint(
            ["cleaning_decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_dataset_versions_decision_set_same_project",
            ondelete="RESTRICT",
            use_alter=True,
        ),
        UniqueConstraint(
            "cleaning_decision_set_id", name="uq_dataset_versions_cleaning_decision_set"
        ),
        Index("ix_dataset_versions_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[UUID] = mapped_column(nullable=False)
    parent_version_id: Mapped[UUID | None] = mapped_column(nullable=True)
    stored_object_id: Mapped[UUID] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="RESTRICT"), nullable=False
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
    cleaning_decision_set_id: Mapped[UUID | None] = mapped_column(nullable=True)
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
    stored_object: Mapped[StoredObject] = relationship(back_populates="dataset_versions")
    input_to_runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="input_dataset_version",
        foreign_keys="ProcessingRun.input_dataset_version_id",
    )
    output_of_run: Mapped[ProcessingRun | None] = relationship(
        back_populates="output_dataset_version",
        foreign_keys="ProcessingRun.output_dataset_version_id",
    )
    quality_reports: Mapped[list[QualityReportRecord]] = relationship(
        back_populates="dataset_version",
        foreign_keys="QualityReportRecord.dataset_version_id",
    )
    cleaning_decision_set: Mapped[CleaningDecisionSet | None] = relationship(
        back_populates="output_dataset_version",
        foreign_keys=[cleaning_decision_set_id],
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
        CheckConstraint("execution_mode IN ('executed', 'reused-result')", name="execution_mode"),
        CheckConstraint(
            "execution_mode <> 'reused-result' OR origin_run_uuid_snapshot IS NOT NULL",
            name="reused_result_origin",
        ),
        CheckConstraint(
            "finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)",
            name="finish_after_start",
        ),
        CheckConstraint(
            "output_dataset_version_id IS NULL OR status = 'succeeded'", name="output_succeeded"
        ),
        UniqueConstraint("id", "project_id", name="uq_processing_runs_id_project"),
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
    execution_mode: Mapped[str] = mapped_column(String(24), nullable=False, default="executed")
    origin_processing_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="SET NULL"), nullable=True
    )
    origin_run_uuid_snapshot: Mapped[UUID | None] = mapped_column(nullable=True)
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
    quality_report: Mapped[QualityReportRecord | None] = relationship(
        back_populates="processing_run", overlaps="quality_reports"
    )


class QualityReportRecord(Base):
    """Immutable profiler result bound to one immutable DatasetVersion and run."""

    __tablename__ = "quality_reports"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_quality_reports_dataset_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_quality_reports_run_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_quality_reports_id_project"),
        UniqueConstraint(
            "dataset_version_id",
            "revision_number",
            name="uq_quality_reports_dataset_revision",
        ),
        UniqueConstraint("processing_run_id", name="uq_quality_reports_processing_run"),
        CheckConstraint("revision_number >= 1", name="revision_positive"),
        CheckConstraint("status IN ('completed', 'failed')", name="status"),
        CheckConstraint("length(profiler_name) > 0", name="profiler_name_nonempty"),
        Index("ix_quality_reports_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version_id: Mapped[UUID] = mapped_column(nullable=False)
    processing_run_id: Mapped[UUID] = mapped_column(nullable=False)
    revision_number: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    profiler_name: Mapped[str] = mapped_column(String(128), nullable=False)
    profiler_version: Mapped[str] = mapped_column(String(128), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(128), nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    report_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(
        back_populates="quality_reports", overlaps="quality_report"
    )
    dataset_version: Mapped[DatasetVersion] = relationship(
        back_populates="quality_reports", foreign_keys=[dataset_version_id]
    )
    processing_run: Mapped[ProcessingRun] = relationship(
        back_populates="quality_report", overlaps="project,quality_reports"
    )
    findings: Mapped[list[QualityFindingRecord]] = relationship(
        back_populates="quality_report",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    decision_sets: Mapped[list[CleaningDecisionSet]] = relationship(
        back_populates="quality_report", overlaps="cleaning_decision_sets"
    )


class QualityFindingRecord(Base):
    """Stable finding identity plus bounded evidence tied to an immutable report."""

    __tablename__ = "quality_findings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["quality_report_id", "project_id"],
            ["quality_reports.id", "quality_reports.project_id"],
            name="fk_quality_findings_report_same_project",
            ondelete="CASCADE",
        ),
        UniqueConstraint("id", "project_id", name="uq_quality_findings_id_project"),
        UniqueConstraint(
            "quality_report_id", "external_id", name="uq_quality_findings_report_external"
        ),
        CheckConstraint("affected_count >= 0", name="affected_count_nonnegative"),
        CheckConstraint(
            "kind IN ('missing', 'duplicate', 'type-conflict', 'extreme-value', "
            "'sudden-change', 'outside-range', 'trend-inconsistent')",
            name="kind",
        ),
        CheckConstraint("severity IN ('info', 'warning', 'error')", name="severity"),
        Index("ix_quality_findings_report_kind", "quality_report_id", "kind"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    quality_report_id: Mapped[UUID] = mapped_column(nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    column_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    column_identity: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT, nullable=True)
    source_record_refs: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=list
    )
    affected_count: Mapped[int] = mapped_column(nullable=False)
    evidence_document: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=dict
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    quality_report: Mapped[QualityReportRecord] = relationship(back_populates="findings")
    decisions: Mapped[list[CleaningDecisionRecord]] = relationship(back_populates="quality_finding")


class CleaningDecisionSet(Base):
    """Immutable, monotonically versioned user decision snapshot."""

    __tablename__ = "cleaning_decision_sets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["quality_report_id", "project_id"],
            ["quality_reports.id", "quality_reports.project_id"],
            name="fk_cleaning_decision_sets_report_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["input_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_cleaning_decision_sets_input_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_cleaning_decision_sets_id_project"),
        UniqueConstraint(
            "project_id", "revision_number", name="uq_cleaning_decision_sets_project_revision"
        ),
        CheckConstraint("revision_number >= 1", name="revision_positive"),
        CheckConstraint("decisions_hash ~ '^[0-9a-f]{64}$'", name="decisions_hash_lower_hex"),
        Index("ix_cleaning_decision_sets_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    quality_report_id: Mapped[UUID] = mapped_column(nullable=False)
    input_dataset_version_id: Mapped[UUID] = mapped_column(nullable=False)
    created_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    revision_number: Mapped[int] = mapped_column(nullable=False)
    decisions_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(
        back_populates="cleaning_decision_sets", overlaps="decision_sets"
    )
    quality_report: Mapped[QualityReportRecord] = relationship(
        back_populates="decision_sets", overlaps="cleaning_decision_sets,project"
    )
    input_dataset_version: Mapped[DatasetVersion] = relationship(
        foreign_keys=[input_dataset_version_id]
    )
    created_by: Mapped[User | None] = relationship(
        back_populates="created_cleaning_decision_sets",
        foreign_keys=[created_by_user_id],
    )
    decisions: Mapped[list[CleaningDecisionRecord]] = relationship(
        back_populates="decision_set",
        cascade="all, delete-orphan",
        passive_deletes=True,
        overlaps="decisions",
    )
    output_dataset_version: Mapped[DatasetVersion | None] = relationship(
        back_populates="cleaning_decision_set",
        foreign_keys="DatasetVersion.cleaning_decision_set_id",
        uselist=False,
    )


class CleaningDecisionRecord(Base):
    """One immutable action against one stable QualityFinding."""

    __tablename__ = "cleaning_decisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_cleaning_decisions_set_same_project",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["quality_finding_id", "project_id"],
            ["quality_findings.id", "quality_findings.project_id"],
            name="fk_cleaning_decisions_finding_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "decision_set_id", "quality_finding_id", name="uq_cleaning_decisions_set_finding"
        ),
        CheckConstraint("action IN ('ignore', 'exclude', 'remove')", name="action"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    decision_set_id: Mapped[UUID] = mapped_column(nullable=False)
    quality_finding_id: Mapped[UUID] = mapped_column(nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    decision_set: Mapped[CleaningDecisionSet] = relationship(
        back_populates="decisions", overlaps="decisions"
    )
    quality_finding: Mapped[QualityFindingRecord] = relationship(
        back_populates="decisions", overlaps="decision_set,decisions"
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
        ForeignKeyConstraint(
            ["cleaning_decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_chart_spec_revisions_decision_set_same_project",
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
    cleaning_decision_set_id: Mapped[UUID | None] = mapped_column(nullable=True)
    spec_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    project: Mapped[Project] = relationship(back_populates="chart_revisions")
    dataset_version: Mapped[DatasetVersion] = relationship(foreign_keys=[dataset_version_id])
    created_by: Mapped[User | None] = relationship(
        back_populates="created_chart_revisions", foreign_keys=[created_by_user_id]
    )
    cleaning_decision_set: Mapped[CleaningDecisionSet | None] = relationship(
        foreign_keys=[cleaning_decision_set_id]
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
        ForeignKeyConstraint(
            ["quality_report_id", "project_id"],
            ["quality_reports.id", "quality_reports.project_id"],
            name="fk_project_revisions_quality_report_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["cleaning_decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_project_revisions_decision_set_same_project",
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
    quality_report_id: Mapped[UUID | None] = mapped_column(nullable=True)
    cleaning_decision_set_id: Mapped[UUID | None] = mapped_column(nullable=True)
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
    quality_report: Mapped[QualityReportRecord | None] = relationship(
        foreign_keys=[quality_report_id]
    )
    cleaning_decision_set: Mapped[CleaningDecisionSet | None] = relationship(
        foreign_keys=[cleaning_decision_set_id]
    )
    created_by: Mapped[User | None] = relationship(
        back_populates="created_project_revisions", foreign_keys=[created_by_user_id]
    )


class ShareLinkRecord(Base):
    """Owner-managed bearer link pinned to one immutable ProjectRevision."""

    __tablename__ = "share_links"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_revision_id", "project_id"],
            ["project_revisions.id", "project_revisions.project_id"],
            name="fk_share_links_revision_same_project",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "id", "project_id", "project_revision_id", name="uq_share_links_identity_scope"
        ),
        CheckConstraint("token_digest ~ '^[0-9a-f]{64}$'", name="token_digest_lower_hex"),
        CheckConstraint("token_key_version >= 1", name="token_key_version_positive"),
        CheckConstraint("status IN ('active', 'revoked')", name="status"),
        CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL AND revoked_by_user_id IS NULL) OR "
            "(status = 'revoked' AND revoked_at IS NOT NULL)",
            name="revocation_state",
        ),
        CheckConstraint(
            "expires_at IS NULL OR expires_at > created_at", name="expiry_after_create"
        ),
        Index("ix_share_links_project_created", "project_id", "created_at"),
        Index("ix_share_links_status_expires", "status", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    project_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    created_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    revoked_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    token_digest: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    token_key_version: Mapped[int] = mapped_column(nullable=False)
    downloads_enabled: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ShareLinkEvent(Base):
    """Immutable management audit event that survives ShareLink purge."""

    __tablename__ = "share_link_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('create', 'downloads-update', 'revoke')", name="event_type"
        ),
        Index("ix_share_link_events_share_created", "share_uuid_snapshot", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    share_link_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("share_links.id", ondelete="SET NULL"), nullable=True
    )
    share_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    project_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    project_revision_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ExportJobRecord(Base):
    """Mutable execution state for one publication export request."""

    __tablename__ = "export_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_revision_id", "project_id"],
            ["project_revisions.id", "project_revisions.project_id"],
            name="fk_export_jobs_revision_same_project",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["current_processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_export_jobs_run_same_project",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("id", "project_id", name="uq_export_jobs_id_project"),
        UniqueConstraint(
            "id",
            "project_id",
            "project_revision_id",
            "format",
            name="uq_export_jobs_id_project_revision_format",
        ),
        CheckConstraint(
            "((requested_by_user_id IS NOT NULL)::int + (guest_session_id IS NOT NULL)::int) = 1",
            name="exactly_one_actor",
        ),
        CheckConstraint("status IN ('queued', 'rendering', 'ready', 'failed')", name="status"),
        CheckConstraint("format IN ('png', 'svg', 'pdf')", name="format"),
        CheckConstraint("request_sha256 ~ '^[0-9a-f]{64}$'", name="request_sha256_lower_hex"),
        CheckConstraint(
            "status <> 'ready' OR pending_stored_object_id IS NULL",
            name="ready_has_no_pending_object",
        ),
        CheckConstraint("attempt_count >= 1", name="attempt_count_positive"),
        Index("ix_export_jobs_project_created", "project_id", "created_at"),
        Index("ix_export_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    project_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    requested_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    guest_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="RESTRICT"), nullable=True
    )
    current_processing_run_id: Mapped[UUID | None] = mapped_column(nullable=True)
    pending_stored_object_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    format: Mapped[str] = mapped_column(String(8), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(nullable=False, default=1)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class PublicationExport(Base):
    """Immutable, reproducible publication artifact produced by an ExportJob."""

    __tablename__ = "publication_exports"
    __table_args__ = (
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_publication_exports_dataset_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["cleaning_decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_publication_exports_decision_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["chart_spec_revision_id", "project_id"],
            ["chart_spec_revisions.id", "chart_spec_revisions.project_id"],
            name="fk_publication_exports_chart_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_publication_exports_run_same_project",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_publication_exports_project_id_projects",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "id",
            "project_id",
            "project_revision_id",
            "format",
            name="uq_publication_exports_binding_scope",
        ),
        CheckConstraint("format IN ('png', 'svg', 'pdf')", name="format"),
        CheckConstraint("dpi > 0", name="dpi_positive"),
        CheckConstraint("output_size_bytes >= 0", name="output_size_nonnegative"),
        CheckConstraint("output_sha256 ~ '^[0-9a-f]{64}$'", name="output_sha256_lower_hex"),
        Index("ix_publication_exports_revision_format", "project_revision_id", "format"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    project_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    dataset_version_id: Mapped[UUID] = mapped_column(nullable=False)
    cleaning_decision_set_id: Mapped[UUID | None] = mapped_column(nullable=True)
    chart_spec_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    processing_run_id: Mapped[UUID] = mapped_column(nullable=False)
    stored_object_id: Mapped[UUID] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="RESTRICT"), nullable=False
    )
    format: Mapped[str] = mapped_column(String(8), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    renderer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    render_contract_version: Mapped[str] = mapped_column(String(128), nullable=False)
    render_spec_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    size_preset: Mapped[str] = mapped_column(String(32), nullable=False)
    width: Mapped[float | None] = mapped_column(nullable=True)
    height: Mapped[float | None] = mapped_column(nullable=True)
    unit: Mapped[str] = mapped_column(String(8), nullable=False)
    dpi: Mapped[int] = mapped_column(nullable=False)
    output_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    output_size_bytes: Mapped[int] = mapped_column(nullable=False)
    validation_document: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class StoredObjectWriteIntent(Base):
    """GC hold for a staged object until its immutable business reference commits."""

    __tablename__ = "stored_object_write_intents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["export_job_id", "project_id"],
            ["export_jobs.id", "export_jobs.project_id"],
            name="fk_stored_object_write_intents_export_job_scope",
            ondelete="CASCADE",
        ),
        UniqueConstraint("export_job_id", name="uq_stored_object_write_intents_export_job_id"),
        CheckConstraint("operation = 'export'", name="operation"),
        CheckConstraint("status IN ('pending', 'completed')", name="status"),
        CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL)",
            name="completion_state",
        ),
        Index("ix_stored_object_write_intents_status_created", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    export_job_id: Mapped[UUID] = mapped_column(nullable=False)
    stored_object_id: Mapped[UUID] = mapped_column(
        ForeignKey("stored_objects.id", ondelete="RESTRICT"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(24), nullable=False, default="export")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ShareExportBinding(Base):
    """Immutable one-time ShareLink-to-PublicationExport binding for one format."""

    __tablename__ = "share_export_bindings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["share_link_id", "project_id", "project_revision_id"],
            ["share_links.id", "share_links.project_id", "share_links.project_revision_id"],
            name="fk_share_export_bindings_share_scope",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
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
        CheckConstraint("format IN ('png', 'svg', 'pdf')", name="format"),
        Index("ix_share_export_bindings_export", "publication_export_id"),
    )

    share_link_id: Mapped[UUID] = mapped_column(primary_key=True)
    format: Mapped[str] = mapped_column(String(8), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    project_revision_id: Mapped[UUID] = mapped_column(nullable=False)
    publication_export_id: Mapped[UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ProjectClaim(Base):
    """Single successful GuestSession-to-User ownership transfer for a Project."""

    __tablename__ = "project_claims"

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    guest_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    guest_session_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    user_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ProjectOrigin(Base):
    """Duplicate/import provenance that survives deletion of the source aggregate."""

    __tablename__ = "project_origins"
    __table_args__ = (
        CheckConstraint("origin_kind IN ('duplicate', 'local-import')", name="origin_kind"),
    )

    target_project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    source_project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    source_revision_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("project_revisions.id", ondelete="SET NULL"), nullable=True
    )
    source_project_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    source_revision_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    origin_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ProjectLifecycleEvent(Base):
    __tablename__ = "project_lifecycle_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('claim', 'save', 'duplicate', 'delete', 'restore', "
            "'revision-restore', 'purge')",
            name="event_type",
        ),
        Index("ix_project_lifecycle_events_project_created", "project_uuid_snapshot", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    project_uuid_snapshot: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "actor_user_id", "operation", "idempotency_key", name="uq_idempotency_actor_operation"
        ),
        UniqueConstraint(
            "guest_session_id",
            "operation",
            "idempotency_key",
            name="uq_idempotency_guest_operation",
        ),
        CheckConstraint(
            "((actor_user_id IS NOT NULL)::int + (guest_session_id IS NOT NULL)::int) = 1",
            name="exactly_one_actor",
        ),
        CheckConstraint("length(idempotency_key) BETWEEN 1 AND 255", name="key_length"),
        CheckConstraint(
            "request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$'",
            name="request_sha256_lower_hex",
        ),
        Index("ix_idempotency_records_created", "created_at"),
        Index("ix_idempotency_records_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    actor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    guest_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="CASCADE"), nullable=True
    )
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[UUID] = mapped_column(nullable=False)
    response_document: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
