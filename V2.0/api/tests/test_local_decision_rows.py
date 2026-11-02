from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config

from labviz_api import processing
from labviz_api.local_schema import (
    CURRENT_REVISION,
    MIGRATIONS_PATH,
    install_local_baseline,
    prepare_local_database,
    sqlite_url,
)
from labviz_api.persistence.sqlite import SqliteProjectStore
from labviz_api.processing import (
    apply_chart_decisions,
    build_preview,
    build_quality_report,
    default_chart_spec,
)
from labviz_api.repository import ProjectRepository

TTL = 7_200


def measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [0, 1, 2, 3, 4, 5],
            "Signal": [1.5, 2.25, np.nan, 4.0, 5.125, 6.5],
            "Group": ["a", "b", "a", None, "b", "a"],
        }
    )


def uploaded(database: Path, frame: pd.DataFrame | None = None) -> SqliteProjectStore:
    prepare_local_database(database, project_ttl_seconds=TTL)
    projects = SqliteProjectStore(ProjectRepository(database, TTL))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('p', 'Decided', '{}', 'local', '2026-01-01T00:00:00Z')"
        )
    complete(projects, measurements() if frame is None else frame)
    return projects


def complete(projects: SqliteProjectStore, frame: pd.DataFrame) -> None:
    projects.complete_project(
        project_id="p",
        source={},
        frame=frame,
        preview=build_preview("p", frame),
        quality=build_quality_report("p", frame),
        chart=default_chart_spec(frame),
    )


def frozen_sets(database: Path) -> list[tuple[str, int, str]]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT finding_id, row_count, frozen_at FROM decision_row_sets ORDER BY finding_id"
        ).fetchall()


def detector_finds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        processing,
        "_quality_finding_mask",
        lambda frame, finding: np.zeros(len(frame), dtype=bool),
    )


def test_a_decision_keeps_removing_the_rows_it_froze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)

    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])
    assert [row[:2] for row in frozen_sets(database)] == [("missing:Signal", 1)]

    # A changed detector no longer changes what the saved decision removes.
    detector_finds_nothing(monkeypatch)
    assert projects.load_chart_dataframe("p")["Time"].tolist() == [0, 1, 3, 4, 5]
    assert len(projects.load_cleaned_dataframe("p")) == 6


def test_actions_share_one_frozen_set_until_the_decision_is_ignored(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)

    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])
    [first] = frozen_sets(database)
    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "remove"}])
    assert frozen_sets(database) == [first]
    assert projects.load_cleaned_dataframe("p")["Time"].tolist() == [0, 1, 3, 4, 5]
    assert projects.load_chart_dataframe("p")["Time"].tolist() == [0, 1, 3, 4, 5]

    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "ignore"}])
    assert frozen_sets(database) == []
    assert len(projects.load_chart_dataframe("p")) == 6

    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "remove"}])
    projects.save_decisions("p", [])
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM decision_rows").fetchone() == (0,)
    assert frozen_sets(database) == []


def test_reprocessing_drops_decisions_with_the_version_they_froze(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])

    reprocessed = measurements()
    reprocessed.loc[2, "Signal"] = 3.0
    reprocessed.loc[4, "Signal"] = np.nan
    complete(projects, reprocessed)

    assert projects.get_decisions("p") == []
    assert frozen_sets(database) == []
    assert len(projects.load_chart_dataframe("p")) == 6


def test_a_decision_without_frozen_rows_is_refused_rather_than_detected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO decisions (project_id, finding_id, action, updated_at) "
            "VALUES ('p', 'missing:Signal', 'exclude', '2026-01-01T00:00:00Z')"
        )

    with pytest.raises(ValueError, match="no frozen rows"):
        projects.load_chart_dataframe("p")


def migrate(database: Path, target: str) -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    config.attributes["database_url"] = sqlite_url(database)
    command.upgrade(config, target)


def legacy_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": list(range(12)),
            "Signal": [1.5, 2.25, np.nan, 4.0, 5.125, 6.5, 7.0, 8.1, 9.0, 95.0, 11.2, 12.0],
            "Group": ["a", "b", "a", None, "b", "a", "b", "a", "b", "a", "b", "a"],
            "Batch": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        }
    )


LEGACY_DECISIONS = [
    ("missing:Signal", "exclude"),
    ("extreme-value:Signal", "remove"),
    ("sudden-change:Batch", "remove"),
    ("missing:Group", "ignore"),
    # Saved against an earlier report; detection matched no rows, so it froze none.
    ("outside-range:Signal", "exclude"),
]


def test_upgrading_freezes_legacy_decisions_without_changing_any_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "labviz.db"
    install_local_baseline(database, project_ttl_seconds=TTL)
    migrate(database, "0105_decision_row_sets")
    projects = SqliteProjectStore(ProjectRepository(database, TTL))
    frame = legacy_measurements()
    quality = build_quality_report("p", frame)
    with sqlite3.connect(database) as connection:
        for project_id in ("p", "pending"):
            connection.execute(
                "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
                "VALUES (?, 'Legacy', '{}', 'local', '2026-01-01T00:00:00Z')",
                (project_id,),
            )
    complete(projects, frame)
    with sqlite3.connect(database) as connection:
        for project_id in ("p", "pending"):
            connection.executemany(
                "INSERT INTO decisions (project_id, finding_id, action, updated_at) "
                "VALUES (?, ?, ?, '2026-01-01T00:00:00Z')",
                [(project_id, finding, action) for finding, action in LEGACY_DECISIONS],
            )
    decisions = [{"findingId": finding, "action": action} for finding, action in LEGACY_DECISIONS]
    # What every read returned before the upgrade: the findings detected again.
    stored = projects.load_quality_dataframe("p")
    before_chart = apply_chart_decisions(stored, quality, decisions)
    before_cleaned = apply_chart_decisions(stored, quality, decisions, actions={"remove"})

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION
    detector_finds_nothing(monkeypatch)

    pd.testing.assert_frame_equal(projects.load_chart_dataframe("p"), before_chart)
    pd.testing.assert_frame_equal(projects.load_cleaned_dataframe("p"), before_cleaned)
    assert before_chart["Time"].tolist() == [0, 1, 3, 5, 6, 7, 10, 11]
    assert before_cleaned["Time"].tolist() == [0, 1, 2, 3, 5, 6, 7, 10, 11]
    with sqlite3.connect(database) as connection:
        sets = connection.execute(
            "SELECT project_id, finding_id, row_count, migrated_from FROM decision_row_sets "
            "ORDER BY finding_id"
        ).fetchall()
    # The project still uploading has no data to freeze against, and ignore removes nothing.
    assert sets == [
        ("p", "extreme-value:Signal", 1, "0106_freeze_legacy_decisions"),
        ("p", "missing:Signal", 1, "0106_freeze_legacy_decisions"),
        ("p", "outside-range:Signal", 0, "0106_freeze_legacy_decisions"),
        ("p", "sudden-change:Batch", 2, "0106_freeze_legacy_decisions"),
    ]


def test_a_saved_decision_is_not_marked_as_migrated(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])

    with sqlite3.connect(database) as connection:
        marks = connection.execute("SELECT migrated_from FROM decision_row_sets").fetchall()
    assert marks == [(None,)]


def test_a_copy_keeps_the_frozen_rows_after_the_original_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "remove"}])

    projects.repository.duplicate_project(
        source_project_id="p", project_id="copy", job_id="job", owner_user_id="user"
    )
    assert projects.repository.delete_project("p")
    detector_finds_nothing(monkeypatch)

    assert projects.load_cleaned_dataframe("copy")["Time"].tolist() == [0, 1, 3, 4, 5]
    with sqlite3.connect(database) as connection:
        copied = connection.execute(
            "SELECT s.dataset_version_id = p.dataset_version_id FROM decision_row_sets s "
            "JOIN projects p ON p.id = s.project_id WHERE s.project_id = 'copy'"
        ).fetchone()
    assert copied == (1,)
