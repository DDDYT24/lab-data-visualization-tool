"""Record the RenderPlan each composed figure panel was drawn from.

Revision ID: 0110_figure_panel_plans
Revises: 0109_chart_render_plans

Boards composed from now on store one RenderPlan artifact per planned panel. Boards saved
earlier have no rows here: their frozen files were not drawn from plans, and their v1/v2
scenes keep working.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast

from alembic import op

revision: str = "0110_figure_panel_plans"
down_revision: str | Sequence[str] | None = "0109_chart_render_plans"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS research_figure_panel_plans ("
        "board_id TEXT NOT NULL REFERENCES research_figure_boards(id) ON DELETE CASCADE, "
        "panel TEXT NOT NULL, plan_sha256 TEXT NOT NULL, PRIMARY KEY (board_id, panel))"
    )


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0109.")
