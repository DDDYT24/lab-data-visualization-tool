"""Normalize check-constraint names created before the Alembic 1.19 repair.

Revision ID: 0011_normalize_check_names
Revises: 0010_experiment_runs
"""

from __future__ import annotations

from hashlib import md5

import sqlalchemy as sa
from alembic import op

revision: str = "0011_normalize_check_names"
down_revision: str | None = "0010_experiment_runs"
branch_labels: str | None = None
depends_on: str | None = None


CHECK_SUFFIXES: dict[str, tuple[str, ...]] = {
    "auth_challenges": ("email_normalized", "failed_attempts_nonnegative"),
    "auth_rate_limit_buckets": (
        "client_identity_lower_hex",
        "email_identity_normalized",
        "request_count_nonnegative",
        "scope",
        "window_order",
    ),
    "auth_sessions": ("token_digest_lower_hex",),
    "chart_spec_revisions": (
        "decision_revision_positive",
        "revision_positive",
        "schema_version_positive",
    ),
    "cleaning_decision_sets": ("decisions_hash_lower_hex", "revision_positive"),
    "cleaning_decisions": ("action",),
    "dataset_versions": (
        "column_count_positive",
        "kind",
        "parent_not_self",
        "row_count_nonnegative",
        "version_positive",
    ),
    "datasets": ("header_row_positive", "name_length"),
    "export_jobs": (
        "attempt_count_positive",
        "exactly_one_actor",
        "format",
        "ready_has_no_pending_object",
        "request_sha256_lower_hex",
        "status",
    ),
    "guest_sessions": ("revoked_status_time", "status", "token_digest_lower_hex"),
    "idempotency_records": (
        "exactly_one_actor",
        "key_length",
        "request_sha256_lower_hex",
    ),
    "orphan_staging_candidates": (
        "deletion_completion",
        "observation_count_nonnegative",
        "retry_count_nonnegative",
        "size_bytes_nonnegative",
    ),
    "processing_runs": ("finish_after_start", "operation", "output_succeeded", "status"),
    "project_lifecycle_events": ("event_type",),
    "project_origins": ("origin_kind",),
    "project_revisions": ("revision_positive", "schema_version_positive"),
    "projects": (
        "deletion_window",
        "storage_mode",
        "title_length",
        "worker_fencing_token_nonnegative",
        "worker_lease_pair",
        "worker_quarantine_has_no_lease",
        "worker_retry_count_nonnegative",
    ),
    "publication_exports": (
        "dpi_positive",
        "format",
        "output_sha256_lower_hex",
        "output_size_nonnegative",
    ),
    "quality_findings": ("affected_count_nonnegative", "kind", "severity"),
    "quality_reports": ("profiler_name_nonempty", "revision_positive", "status"),
    "share_export_bindings": ("format",),
    "share_link_events": ("event_type",),
    "share_links": (
        "expiry_after_create",
        "revocation_state",
        "status",
        "token_digest_lower_hex",
        "token_key_version_positive",
    ),
    "source_files": (
        "header_row_positive",
        "original_name_nonempty",
        "sha256_lower_hex",
        "size_nonnegative",
    ),
    "storage_inventory_checkpoints": (
        "completion_state",
        "item_count_nonnegative",
        "page_count_nonnegative",
        "status",
        "task_fencing_token_nonnegative",
    ),
    "stored_object_write_intents": (
        "completion_state",
        "operation",
        "status",
        "worker_fencing_token_nonnegative",
        "worker_lease_pair",
        "worker_quarantine_has_no_lease",
        "worker_retry_count_nonnegative",
    ),
    "stored_objects": (
        "deleted_status_time",
        "object_key_nonempty",
        "purpose",
        "sha256_lower_hex",
        "size_nonnegative",
        "status",
        "worker_fencing_token_nonnegative",
        "worker_lease_pair",
        "worker_quarantine_has_no_lease",
        "worker_retry_count_nonnegative",
    ),
    "users": ("email_length", "email_normalized"),
    "worker_leases": ("fencing_token_nonnegative", "lease_pair", "task"),
}


def _legacy_name(table_name: str, expected_name: str) -> str:
    doubled = f"ck_{table_name}_{expected_name}"
    if len(doubled) <= 63:
        return doubled
    digest = md5(doubled.encode(), usedforsecurity=False).hexdigest()[-4:]
    return f"{doubled[:55]}_{digest}"


def _check_names(table_name: str) -> set[str]:
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema = current_schema()
              AND table_name = :table_name
              AND constraint_type = 'CHECK'
            """
        ),
        {"table_name": table_name},
    )
    return {str(row[0]) for row in rows}


def upgrade() -> None:
    connection = op.get_bind()
    preparer = connection.dialect.identifier_preparer
    for table_name, suffixes in CHECK_SUFFIXES.items():
        names = _check_names(table_name)
        for suffix in suffixes:
            expected_name = f"ck_{table_name}_{suffix}"
            legacy_name = _legacy_name(table_name, expected_name)
            if legacy_name not in names:
                continue
            if expected_name in names:
                raise RuntimeError(
                    f"Both legacy and normalized constraints exist on {table_name}: "
                    f"{legacy_name}, {expected_name}."
                )
            op.execute(
                sa.text(
                    "ALTER TABLE "
                    f"{preparer.quote(table_name)} RENAME CONSTRAINT "
                    f"{preparer.quote(legacy_name)} TO {preparer.quote(expected_name)}"
                )
            )
            names.remove(legacy_name)
            names.add(expected_name)


def downgrade() -> None:
    """Keep normalized names because upgrade may have been a no-op on a fresh database."""
