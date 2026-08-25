"""Add immutable quality and cleaning lineage.

Revision ID: 0003_quality_cleaning_lineage
Revises: 0002_phase2_persistence
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_quality_cleaning_lineage"
down_revision: str | Sequence[str] | None = "0002_phase2_persistence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(none_as_null=True)
TIMESTAMP = sa.DateTime(timezone=True)

IMMUTABLE_TABLES = (
    "dataset_versions",
    "project_revisions",
    "chart_spec_revisions",
    "quality_reports",
    "quality_findings",
    "cleaning_decision_sets",
    "cleaning_decisions",
)


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_processing_runs_id_project", "processing_runs", ["id", "project_id"]
    )

    op.create_table(
        "quality_reports",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("processing_run_id", UUID, nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("profiler_name", sa.String(128), nullable=False),
        sa.Column("profiler_version", sa.String(128), nullable=False),
        sa.Column("algorithm_version", sa.String(128), nullable=False),
        sa.Column("code_version", sa.String(128), nullable=False),
        sa.Column("parameters", JSONB, nullable=False),
        sa.Column("report_document", JSONB, nullable=False),
        sa.Column("completed_at", TIMESTAMP, nullable=True),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "revision_number >= 1", name=op.f("ck_quality_reports_revision_positive")
        ),
        sa.CheckConstraint(
            "status IN ('completed', 'failed')", name=op.f("ck_quality_reports_status")
        ),
        sa.CheckConstraint(
            "length(profiler_name) > 0", name=op.f("ck_quality_reports_profiler_name_nonempty")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_quality_reports_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_quality_reports_dataset_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["processing_run_id", "project_id"],
            ["processing_runs.id", "processing_runs.project_id"],
            name="fk_quality_reports_run_same_project",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_quality_reports"),
        sa.UniqueConstraint("id", "project_id", name="uq_quality_reports_id_project"),
        sa.UniqueConstraint(
            "dataset_version_id",
            "revision_number",
            name="uq_quality_reports_dataset_revision",
        ),
        sa.UniqueConstraint("processing_run_id", name="uq_quality_reports_processing_run"),
    )
    op.create_index(
        "ix_quality_reports_project_created", "quality_reports", ["project_id", "created_at"]
    )

    op.create_table(
        "quality_findings",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("quality_report_id", UUID, nullable=False),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(48), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("column_name", sa.String(512), nullable=True),
        sa.Column("column_identity", JSONB, nullable=True),
        sa.Column("source_record_refs", JSONB, nullable=False),
        sa.Column("affected_count", sa.Integer(), nullable=False),
        sa.Column("evidence_document", JSONB, nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "affected_count >= 0", name=op.f("ck_quality_findings_affected_count_nonnegative")
        ),
        sa.CheckConstraint(
            "kind IN ('missing', 'duplicate', 'type-conflict', 'extreme-value', "
            "'sudden-change', 'outside-range', 'trend-inconsistent')",
            name=op.f("ck_quality_findings_kind"),
        ),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name=op.f("ck_quality_findings_severity"),
        ),
        sa.ForeignKeyConstraint(
            ["quality_report_id", "project_id"],
            ["quality_reports.id", "quality_reports.project_id"],
            name="fk_quality_findings_report_same_project",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_quality_findings"),
        sa.UniqueConstraint("id", "project_id", name="uq_quality_findings_id_project"),
        sa.UniqueConstraint(
            "quality_report_id",
            "external_id",
            name="uq_quality_findings_report_external",
        ),
    )
    op.create_index(
        "ix_quality_findings_report_kind",
        "quality_findings",
        ["quality_report_id", "kind"],
    )

    op.create_table(
        "cleaning_decision_sets",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("quality_report_id", UUID, nullable=False),
        sa.Column("input_dataset_version_id", UUID, nullable=False),
        sa.Column("created_by_user_id", UUID, nullable=True),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("decisions_hash", sa.String(64), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "revision_number >= 1", name=op.f("ck_cleaning_decision_sets_revision_positive")
        ),
        sa.CheckConstraint(
            "decisions_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_cleaning_decision_sets_decisions_hash_lower_hex"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_cleaning_decision_sets_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["quality_report_id", "project_id"],
            ["quality_reports.id", "quality_reports.project_id"],
            name="fk_cleaning_decision_sets_report_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["input_dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name="fk_cleaning_decision_sets_input_same_project",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_cleaning_decision_sets_created_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cleaning_decision_sets"),
        sa.UniqueConstraint("id", "project_id", name="uq_cleaning_decision_sets_id_project"),
        sa.UniqueConstraint(
            "project_id",
            "revision_number",
            name="uq_cleaning_decision_sets_project_revision",
        ),
    )
    op.create_index(
        "ix_cleaning_decision_sets_project_created",
        "cleaning_decision_sets",
        ["project_id", "created_at"],
    )

    op.create_table(
        "cleaning_decisions",
        sa.Column("id", UUID, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("decision_set_id", UUID, nullable=False),
        sa.Column("quality_finding_id", UUID, nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "action IN ('ignore', 'exclude', 'remove')",
            name=op.f("ck_cleaning_decisions_action"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_set_id", "project_id"],
            ["cleaning_decision_sets.id", "cleaning_decision_sets.project_id"],
            name="fk_cleaning_decisions_set_same_project",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["quality_finding_id", "project_id"],
            ["quality_findings.id", "quality_findings.project_id"],
            name="fk_cleaning_decisions_finding_same_project",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cleaning_decisions"),
        sa.UniqueConstraint(
            "decision_set_id",
            "quality_finding_id",
            name="uq_cleaning_decisions_set_finding",
        ),
    )

    op.add_column("dataset_versions", sa.Column("cleaning_decision_set_id", UUID, nullable=True))
    op.create_unique_constraint(
        "uq_dataset_versions_cleaning_decision_set",
        "dataset_versions",
        ["cleaning_decision_set_id"],
    )
    op.create_foreign_key(
        "fk_dataset_versions_decision_set_same_project",
        "dataset_versions",
        "cleaning_decision_sets",
        ["cleaning_decision_set_id", "project_id"],
        ["id", "project_id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "chart_spec_revisions", sa.Column("cleaning_decision_set_id", UUID, nullable=True)
    )
    op.create_foreign_key(
        "fk_chart_spec_revisions_decision_set_same_project",
        "chart_spec_revisions",
        "cleaning_decision_sets",
        ["cleaning_decision_set_id", "project_id"],
        ["id", "project_id"],
        ondelete="RESTRICT",
    )

    op.add_column("project_revisions", sa.Column("quality_report_id", UUID, nullable=True))
    op.add_column("project_revisions", sa.Column("cleaning_decision_set_id", UUID, nullable=True))
    op.create_foreign_key(
        "fk_project_revisions_quality_report_same_project",
        "project_revisions",
        "quality_reports",
        ["quality_report_id", "project_id"],
        ["id", "project_id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_project_revisions_decision_set_same_project",
        "project_revisions",
        "cleaning_decision_sets",
        ["cleaning_decision_set_id", "project_id"],
        ["id", "project_id"],
        ondelete="RESTRICT",
    )

    op.execute(
        """
        CREATE FUNCTION labviz_reject_immutable_update()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'immutable LabViz record cannot be updated: %', TG_TABLE_NAME
                USING ERRCODE = '55000';
        END;
        $$
        """
    )
    for table_name in IMMUTABLE_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table_name}_immutable "
            f"BEFORE UPDATE ON {table_name} FOR EACH ROW "
            "EXECUTE FUNCTION labviz_reject_immutable_update()"
        )


def downgrade() -> None:
    for table_name in IMMUTABLE_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_immutable ON {table_name}")
    op.execute("DROP FUNCTION IF EXISTS labviz_reject_immutable_update()")

    op.drop_constraint(
        "fk_project_revisions_decision_set_same_project",
        "project_revisions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_project_revisions_quality_report_same_project",
        "project_revisions",
        type_="foreignkey",
    )
    op.drop_column("project_revisions", "cleaning_decision_set_id")
    op.drop_column("project_revisions", "quality_report_id")

    op.drop_constraint(
        "fk_chart_spec_revisions_decision_set_same_project",
        "chart_spec_revisions",
        type_="foreignkey",
    )
    op.drop_column("chart_spec_revisions", "cleaning_decision_set_id")

    op.drop_constraint(
        "fk_dataset_versions_decision_set_same_project",
        "dataset_versions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_dataset_versions_cleaning_decision_set",
        "dataset_versions",
        type_="unique",
    )
    op.drop_column("dataset_versions", "cleaning_decision_set_id")

    op.drop_table("cleaning_decisions")
    op.drop_index("ix_cleaning_decision_sets_project_created", table_name="cleaning_decision_sets")
    op.drop_table("cleaning_decision_sets")
    op.drop_index("ix_quality_findings_report_kind", table_name="quality_findings")
    op.drop_table("quality_findings")
    op.drop_index("ix_quality_reports_project_created", table_name="quality_reports")
    op.drop_table("quality_reports")
    op.drop_constraint("uq_processing_runs_id_project", "processing_runs", type_="unique")
