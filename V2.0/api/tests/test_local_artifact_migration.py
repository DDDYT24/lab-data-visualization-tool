from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from labviz_api.artifacts import (
    REFERENCES,
    REGISTRY_TABLE,
    local_artifact_store,
    remember,
    sweep,
)
from labviz_api.local_schema import (
    CURRENT_REVISION,
    install_local_baseline,
    prepare_local_database,
)
from labviz_api.repository import ProjectRepository

TTL = 7_200
SVG = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096
PDF = b"%PDF-1.7\n" + b"\x01" * 4096
ZIP = b"PK\x03\x04" + b"\x02" * 4096
THUMBNAIL = b"\x89PNG\r\n\x1a\nthumbnail"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def legacy_profile(database: Path) -> None:
    """A profile written before 0101: every rendered artifact is a BLOB in its row."""
    install_local_baseline(database, project_ttl_seconds=TTL)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO exports (id, project_id, format, status, payload, message, created_at) "
            "VALUES ('export', 'p', 'png', 'ready', ?, '', '2026-01-01')",
            (PNG,),
        )
        connection.execute(
            "INSERT INTO exports (id, project_id, format, status, payload, message, created_at) "
            "VALUES ('failed', 'p', 'pdf', 'failed', NULL, '', '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO figure_snapshots (id, project_id, title, format, chart_json, payload, "
            "thumbnail_png, sha256, byte_size, created_at) "
            "VALUES ('snapshot', 'p', 'Figure', 'png', '{}', ?, ?, ?, ?, '2026-01-01')",
            (PNG, THUMBNAIL, sha(PNG), len(PNG)),
        )
        connection.execute(
            "INSERT INTO research_charts (id, project_id, revision_id, result_json, svg, png, "
            "pdf, created_at) VALUES ('chart', 'p', 'r', '{}', ?, ?, ?, '2026-01-01')",
            (SVG, PNG, PDF),
        )
        connection.execute(
            "INSERT INTO research_figure_artifacts (board_id, svg, png, pdf, scene_json) "
            "VALUES ('board', ?, ?, ?, '{}')",
            (SVG, PNG, PDF),
        )
        connection.execute(
            "INSERT INTO research_outputs (id, project_id, board_id, manifest_json, notes_json, "
            "zip_blob, sha256, created_at) VALUES ('output', 'p', 'board', '{}', '{}', ?, ?, "
            "'2026-01-01')",
            (ZIP, sha(ZIP)),
        )


def test_upgrading_moves_every_blob_into_the_store_and_drops_the_columns(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION

    store = local_artifact_store(database)
    with sqlite3.connect(database) as connection:
        for table, dropped in (
            ("exports", "payload"),
            ("figure_snapshots", "payload"),
            ("figure_snapshots", "thumbnail_png"),
            ("research_charts", "svg"),
            ("research_figure_artifacts", "pdf"),
            ("research_outputs", "zip_blob"),
        ):
            assert dropped not in columns(connection, table)
        export, failed = connection.execute(
            "SELECT payload_sha256 FROM exports ORDER BY id"
        ).fetchall()
        snapshot = connection.execute(
            "SELECT sha256, thumbnail_sha256 FROM figure_snapshots"
        ).fetchone()
        chart = connection.execute(
            "SELECT svg_sha256, png_sha256, pdf_sha256 FROM research_charts"
        ).fetchone()
        board = connection.execute(
            "SELECT svg_sha256, png_sha256, pdf_sha256 FROM research_figure_artifacts"
        ).fetchone()
        output = connection.execute("SELECT sha256 FROM research_outputs").fetchone()

    assert store.read(export[0]) == PNG
    assert failed == (None,)
    assert store.read(snapshot[0]) == PNG
    assert store.read(snapshot[1]) == THUMBNAIL
    assert [store.read(digest) for digest in chart] == [SVG, PNG, PDF]
    assert tuple(board) == tuple(chart)
    assert store.read(output[0]) == ZIP


def test_identical_renders_share_one_file_and_one_registry_row(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)

    prepare_local_database(database, project_ttl_seconds=TTL)

    with sqlite3.connect(database) as connection:
        registered = dict(connection.execute(f"SELECT sha256, byte_size FROM {REGISTRY_TABLE}"))
    # The PNG appears in an export, a snapshot, a chart and a board but is stored once.
    assert registered == {sha(payload): len(payload) for payload in (SVG, PNG, PDF, ZIP, THUMBNAIL)}
    stored = [path for path in (tmp_path / "objects/artifacts").rglob("*") if path.is_file()]
    assert sorted(path.name for path in stored if len(path.name) == 64) == sorted(registered)


def test_upgrading_reclaims_the_space_the_blobs_used(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT INTO research_charts (id, project_id, revision_id, result_json, svg, png, "
            "pdf, created_at) VALUES (?, 'p', 'r', '{}', ?, ?, ?, '2026-01-01')",
            [(f"bulk-{index}", SVG, PNG + bytes([index]) * 65_536, PDF) for index in range(32)],
        )
    # The profile runs in WAL mode, so the bytes may sit in the -wal file until checkpoint.
    before = sum(path.stat().st_size for path in tmp_path.glob("labviz.db*"))

    prepare_local_database(database, project_ttl_seconds=TTL)

    after = sum(path.stat().st_size for path in tmp_path.glob("labviz.db*"))
    assert after < before // 4


def test_a_current_profile_upgrades_once(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    with sqlite3.connect(database) as connection:
        first = connection.execute(f"SELECT COUNT(*) FROM {REGISTRY_TABLE}").fetchone()

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION
    with sqlite3.connect(database) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {REGISTRY_TABLE}").fetchone() == first


def test_every_reference_column_exists_after_upgrade(tmp_path: Path) -> None:
    """The sweep trusts REFERENCES; a missing column would hide live artifacts from it."""
    database = tmp_path / "labviz.db"

    prepare_local_database(database, project_ttl_seconds=TTL)

    with sqlite3.connect(database) as connection:
        for table, column in REFERENCES:
            assert column in columns(connection, table), (table, column)


def test_sweep_deletes_only_unreferenced_artifacts(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    store = local_artifact_store(database)
    with sqlite3.connect(database) as connection:
        orphan = remember(connection, store, b"orphan", media_type="image/png")
        connection.execute("DELETE FROM research_outputs")

    assert sweep(database, store) == 2

    assert not store.exists(orphan)
    assert not store.exists(sha(ZIP))
    assert store.exists(sha(PNG)) and store.exists(sha(THUMBNAIL))
    with sqlite3.connect(database) as connection:
        remaining = {row[0] for row in connection.execute(f"SELECT sha256 FROM {REGISTRY_TABLE}")}
    assert orphan not in remaining and sha(ZIP) not in remaining
    assert sweep(database, store) == 0


def test_sweep_leaves_an_unmigrated_profile_alone(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)

    assert sweep(database, local_artifact_store(database)) == 0


def test_an_export_reads_back_from_the_store_and_is_swept_with_its_project(
    tmp_path: Path,
) -> None:
    database = tmp_path / "labviz.db"
    prepare_local_database(database, project_ttl_seconds=TTL)
    repository = ProjectRepository(database, TTL)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('p', 'Kept', '{}', 'local', '2026-01-01T00:00:00Z')"
        )
    repository.save_export(
        export_id="export",
        project_id="p",
        format_name="png",
        payload=PNG,
        expires_at="2999-01-01T00:00:00Z",
        message="ready",
        figure_snapshot={
            "id": "snapshot",
            "title": "Figure",
            "chart_json": "{}",
            "sha256": sha(PNG),
            "thumbnail_png": THUMBNAIL,
        },
    )

    export = repository.get_export("export")
    snapshot = repository.get_figure_snapshot("p", "snapshot")
    assert export is not None and export["payload"] == PNG
    assert snapshot is not None and snapshot["payload"] == PNG
    assert snapshot["thumbnail_png"] == THUMBNAIL

    assert repository.delete_project("p")
    repository.cleanup_expired()

    assert not repository.artifacts.exists(sha(PNG))
    assert not repository.artifacts.exists(sha(THUMBNAIL))
