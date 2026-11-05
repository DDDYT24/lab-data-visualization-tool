"""Record the RenderPlan each research chart was drawn from.

Revision ID: 0109_chart_render_plans
Revises: 0108_pack_decision_rows

Charts saved from now on store their RenderPlan as an artifact, and ``plan_sha256`` names
it. Charts saved earlier keep NULL: their frozen files were not drawn from a plan, and no
plan is reconstructed for them.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast

from alembic import op

revision: str = "0109_chart_render_plans"
down_revision: str | Sequence[str] | None = "0108_pack_decision_rows"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    connection.execute("ALTER TABLE research_charts ADD COLUMN plan_sha256 TEXT")


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0108.")
