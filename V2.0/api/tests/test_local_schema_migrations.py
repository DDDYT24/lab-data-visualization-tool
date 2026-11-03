from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from labviz_api.local_schema import (
    BASELINE_REVISION,
    CURRENT_REVISION,
    MIGRATIONS_PATH,
    VERSION_TABLE,
    install_local_baseline,
    local_schema_revision,
    prepare_local_database,
    require_sqlite_url,
    sqlite_url,
)

TTL = 7_200
BASELINE_TABLES = {
    "projects",
    "jobs",
    "decisions",
    "shares",
    "exports",
    "figure_snapshots",
    "research_revisions",
    "research_heads",
    "research_drafts",
    "research_figure_boards",
    "research_figure_artifacts",
    "research_figure_jobs",
    "research_charts",
    "research_templates",
    "research_outputs",
}


def tables(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def head_revision() -> str:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    return ScriptDirectory.from_config(config).get_current_head() or ""


def test_the_local_chain_has_exactly_one_head_and_matches_the_build() -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))

    assert len(ScriptDirectory.from_config(config).get_heads()) == 1
    # Adding a revision without updating CURRENT_REVISION would leave the startup fast
    # path believing an old profile is already current.
    assert head_revision() == CURRENT_REVISION


def test_a_current_profile_starts_without_importing_alembic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Desktop cold start must not pay for Alembic once the schema is already current."""
    database = tmp_path / "labviz.db"
    prepare_local_database(database, project_ttl_seconds=TTL)

    for module in ("alembic", "alembic.command", "alembic.config"):
        monkeypatch.setitem(sys.modules, module, None)

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION


def test_a_new_profile_installs_every_baseline_table_and_reaches_head(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"

    revision = prepare_local_database(database, project_ttl_seconds=TTL)

    assert revision == head_revision()
    assert tables(database) >= BASELINE_TABLES
    assert VERSION_TABLE in tables(database)


def test_a_profile_that_predates_alembic_is_stamped_without_losing_data(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    install_local_baseline(database, project_ttl_seconds=TTL)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('keep-me', 'Existing', '{}', 'saved-local', '2026-01-01T00:00:00Z')"
        )
    assert local_schema_revision(database) is None
    assert VERSION_TABLE not in tables(database)

    revision = prepare_local_database(database, project_ttl_seconds=TTL)

    assert revision == head_revision()
    with sqlite3.connect(database) as connection:
        kept = connection.execute("SELECT title FROM projects WHERE id='keep-me'").fetchone()
    assert kept is not None and kept[0] == "Existing"


def test_preparing_twice_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"

    first = prepare_local_database(database, project_ttl_seconds=TTL)
    before = tables(database)
    second = prepare_local_database(database, project_ttl_seconds=TTL)

    assert first == second == head_revision()
    assert tables(database) == before


def test_the_local_chain_is_separate_from_the_postgresql_chain(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"

    prepare_local_database(database, project_ttl_seconds=TTL)

    # The PostgreSQL chain owns "alembic_version"; a local profile must never claim it.
    assert "alembic_version" not in tables(database)
    assert BASELINE_REVISION.startswith("0100_")


def test_no_connection_survives_preparation(tmp_path: Path) -> None:
    """Windows keeps a moved or deleted database locked while any connection is open."""
    database = tmp_path / "labviz.db"
    prepare_local_database(database, project_ttl_seconds=TTL)

    moved = tmp_path / "moved.db"
    database.rename(moved)

    assert moved.is_file()
    assert not database.exists()


def test_a_sqlite_url_survives_spaces_and_non_ascii_paths(tmp_path: Path) -> None:
    directory = tmp_path / "科研 数据"
    directory.mkdir()
    database = directory / "labviz.db"

    assert sqlite_url(database).startswith("sqlite+pysqlite:///")
    revision = prepare_local_database(database, project_ttl_seconds=TTL)

    assert revision == head_revision()


def test_the_local_chain_refuses_a_non_sqlite_url() -> None:
    assert require_sqlite_url("sqlite+pysqlite:///x.db") == "sqlite+pysqlite:///x.db"
    with pytest.raises(RuntimeError):
        require_sqlite_url("postgresql+psycopg://user:pass@127.0.0.1/db")


def test_the_local_environment_uses_the_shared_guard_and_version_table() -> None:
    source = (MIGRATIONS_PATH / "env.py").read_text(encoding="utf-8")

    assert "from labviz_api.local_schema import VERSION_TABLE, require_sqlite_url" in source
    assert source.count("require_sqlite_url(database_url())") == 2
    assert "version_table=VERSION_TABLE" in source


def test_the_chain_ships_inside_the_package_every_installer_copies() -> None:
    """The container, Windows and desktop builds copy ``labviz_api`` but not its siblings.

    A chain beside the package left every packaged app unable to start: the first
    ``create_app()`` raised because Alembic could not find the scripts directory.
    """
    import labviz_api

    package = Path(labviz_api.__file__).resolve().parent
    assert MIGRATIONS_PATH.is_relative_to(package)
    assert (MIGRATIONS_PATH / "env.py").is_file()
    assert (MIGRATIONS_PATH / "versions" / f"{BASELINE_REVISION}.py").is_file()
