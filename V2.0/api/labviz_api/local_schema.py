"""Ordered schema setup for the local SQLite profile.

Baseline tables were previously created by whichever constructor happened to run first,
and later columns were patched in by hand with ``PRAGMA table_info`` checks. This module
makes the order explicit and hands every change after the baseline to Alembic:

1. run the idempotent baseline installers (classic repository + research sidecars);
2. stamp a profile that predates the local revision table at ``0100_local_baseline``;
3. upgrade to head.

A brand-new profile and an existing one therefore take the same path.

A profile that is already current reads one row with ``sqlite3`` and returns, so ordinary
desktop startup never imports Alembic or builds a SQLAlchemy engine.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

BASELINE_REVISION = "0100_local_baseline"
#: The chain head this build expects. ``test_local_schema_migrations`` keeps the two equal,
#: so a new revision cannot be added without updating the fast path that skips Alembic.
CURRENT_REVISION = "0107_cleaned_dataset_versions"
VERSION_TABLE = "alembic_version_local"
#: Kept inside the package so that every route that ships ``labviz_api`` (the container
#: image, the Windows installer and the desktop bundle) ships the migrations with it.
MIGRATIONS_PATH = Path(__file__).resolve().parent / "local_migrations"


def sqlite_url(database_path: Path) -> str:
    """Build a URL that survives spaces and non-ASCII characters in a Windows profile path."""
    return f"sqlite+pysqlite:///{database_path.resolve().as_posix()}"


def require_sqlite_url(url: str) -> str:
    """Keep the local chain off the PostgreSQL schema, which has its own revisions."""
    if not url.startswith("sqlite"):
        raise RuntimeError("The local migration chain only runs against SQLite.")
    return url


def local_schema_revision(database_path: Path) -> str | None:
    """Return the stamped local revision, or None when the profile predates Alembic."""
    if not database_path.exists():
        return None
    with closing(sqlite3.connect(database_path)) as connection:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (VERSION_TABLE,),
        ).fetchone()
        if present is None:
            return None
        row = connection.execute(f"SELECT version_num FROM {VERSION_TABLE}").fetchone()
    return str(row[0]) if row else None


def install_local_baseline(database_path: Path, *, project_ttl_seconds: int) -> None:
    """Create every baseline table, in one place instead of per constructor.

    Imported inside the function so that reading the stamped revision never drags in
    the research stack.
    """
    from labviz_api.repository import ProjectRepository
    from labviz_api.research_chart_store import install_charts
    from labviz_api.research_figure_board import install_boards
    from labviz_api.research_reuse import install_reuse
    from labviz_api.research_store import ResearchStore

    ProjectRepository(database_path, project_ttl_seconds)
    store = ResearchStore(database_path)
    install_boards(store)
    install_charts(store)
    install_reuse(store)


def _migrate(database_path: Path) -> None:
    """Stamp a pre-Alembic profile, then upgrade. Alembic is imported only when used."""
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    config.attributes["database_url"] = sqlite_url(database_path)
    if local_schema_revision(database_path) is None:
        command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")
    # Migrations can drop large columns; SQLite keeps those pages until VACUUM, which
    # cannot run inside the migration's own transaction. In WAL mode the file itself only
    # shrinks once the rewritten pages are checkpointed back into it.
    with closing(sqlite3.connect(database_path, timeout=30, isolation_level=None)) as connection:
        connection.execute("VACUUM")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def prepare_local_database(database_path: Path, *, project_ttl_seconds: int) -> str | None:
    """Install the baseline and bring the local schema to this build's revision."""
    install_local_baseline(database_path, project_ttl_seconds=project_ttl_seconds)
    if local_schema_revision(database_path) != CURRENT_REVISION:
        _migrate(database_path)
    return local_schema_revision(database_path)
