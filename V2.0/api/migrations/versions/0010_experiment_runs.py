"""Add experiment and physical acquisition provenance.

Revision ID: 0010_experiment_runs
Revises: 0009_atomic_auth_rate_limits
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0010_experiment_runs"
down_revision: str | Sequence[str] | None = "0009_atomic_auth_rate_limits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("owner_user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("guest_session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(title) BETWEEN 1 AND 200",
            name=op.f("ck_experiments_title_length"),
        ),
        sa.CheckConstraint(
            "(owner_user_id IS NULL) <> (guest_session_id IS NULL)",
            name=op.f("ck_experiments_single_owner"),
        ),
        sa.ForeignKeyConstraint(
            ["guest_session_id"],
            ["guest_sessions.id"],
            name=op.f("fk_experiments_guest_session_id_guest_sessions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.id"],
            name=op.f("fk_experiments_owner_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiments")),
    )
    op.create_index("ix_experiments_owner_updated", "experiments", ["owner_user_id", "updated_at"])
    op.create_index(
        "ix_experiments_guest_updated", "experiments", ["guest_session_id", "updated_at"]
    )
    op.create_table(
        "experiment_runs",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("experiment_id", UUID(as_uuid=True), nullable=False),
        sa.Column("run_label", sa.String(200), nullable=False),
        sa.Column("replicate_id", sa.String(100), nullable=True),
        sa.Column("batch_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(run_label) BETWEEN 1 AND 200",
            name=op.f("ck_experiment_runs_run_label_length"),
        ),
        sa.CheckConstraint(
            "replicate_id IS NULL OR length(replicate_id) BETWEEN 1 AND 100",
            name=op.f("ck_experiment_runs_replicate_id_length"),
        ),
        sa.CheckConstraint(
            "batch_id IS NULL OR length(batch_id) BETWEEN 1 AND 100",
            name=op.f("ck_experiment_runs_batch_id_length"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_experiment_runs_experiment_id_experiments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiment_runs")),
    )
    op.create_index(
        "ix_experiment_runs_experiment_created",
        "experiment_runs",
        ["experiment_id", "created_at"],
    )
    op.add_column("projects", sa.Column("experiment_run_id", UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        op.f("fk_projects_experiment_run_id_experiment_runs"),
        "projects",
        "experiment_runs",
        ["experiment_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("ix_projects_experiment_run_id", "projects", ["experiment_run_id"])
    op.add_column(
        "publication_exports", sa.Column("experiment_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "publication_exports", sa.Column("experiment_run_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "publication_exports",
        sa.Column("experiment_title_snapshot", sa.String(200), nullable=True),
    )
    op.add_column(
        "publication_exports", sa.Column("run_label_snapshot", sa.String(200), nullable=True)
    )
    op.add_column(
        "publication_exports", sa.Column("replicate_id_snapshot", sa.String(100), nullable=True)
    )
    op.add_column(
        "publication_exports", sa.Column("batch_id_snapshot", sa.String(100), nullable=True)
    )
    op.create_foreign_key(
        op.f("fk_publication_exports_experiment_id_experiments"),
        "publication_exports",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        op.f("fk_publication_exports_experiment_run_id_experiment_runs"),
        "publication_exports",
        "experiment_runs",
        ["experiment_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    connection = op.get_bind()
    retained = connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM experiments)")).scalar_one()
    if retained:
        raise RuntimeError(
            "Experiment provenance exists; export or remove it explicitly before downgrading."
        )
    op.drop_constraint(
        op.f("fk_publication_exports_experiment_run_id_experiment_runs"),
        "publication_exports",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_publication_exports_experiment_id_experiments"),
        "publication_exports",
        type_="foreignkey",
    )
    op.drop_column("publication_exports", "batch_id_snapshot")
    op.drop_column("publication_exports", "replicate_id_snapshot")
    op.drop_column("publication_exports", "run_label_snapshot")
    op.drop_column("publication_exports", "experiment_title_snapshot")
    op.drop_column("publication_exports", "experiment_run_id")
    op.drop_column("publication_exports", "experiment_id")
    op.drop_index("ix_projects_experiment_run_id", table_name="projects")
    op.drop_constraint(
        op.f("fk_projects_experiment_run_id_experiment_runs"),
        "projects",
        type_="foreignkey",
    )
    op.drop_column("projects", "experiment_run_id")
    op.drop_index("ix_experiment_runs_experiment_created", table_name="experiment_runs")
    op.drop_table("experiment_runs")
    op.drop_index("ix_experiments_guest_updated", table_name="experiments")
    op.drop_index("ix_experiments_owner_updated", table_name="experiments")
    op.drop_table("experiments")
