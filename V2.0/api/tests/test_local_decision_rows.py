from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labviz_api import processing
from labviz_api.local_schema import prepare_local_database
from labviz_api.persistence.sqlite import SqliteProjectStore
from labviz_api.processing import build_preview, build_quality_report, default_chart_spec
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


def test_a_decision_saved_before_freezing_is_still_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO decisions (project_id, finding_id, action, updated_at) "
            "VALUES ('p', 'missing:Signal', 'exclude', '2026-01-01T00:00:00Z')"
        )

    assert projects.load_chart_dataframe("p")["Time"].tolist() == [0, 1, 3, 4, 5]
    detector_finds_nothing(monkeypatch)
    assert len(projects.load_chart_dataframe("p")) == 6


def test_rows_frozen_against_another_version_are_not_applied(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    projects = uploaded(database)
    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])

    # The same finding covers another row once the project is reprocessed with new data.
    reprocessed = measurements()
    reprocessed.loc[2, "Signal"] = 3.0
    reprocessed.loc[4, "Signal"] = np.nan
    complete(projects, reprocessed)
    assert projects.load_chart_dataframe("p")["Time"].tolist() == [0, 1, 2, 3, 5]

    projects.save_decisions("p", [{"findingId": "missing:Signal", "action": "exclude"}])
    with sqlite3.connect(database) as connection:
        version = connection.execute("SELECT dataset_version_id FROM projects").fetchone()
        frozen = connection.execute("SELECT dataset_version_id FROM decision_row_sets").fetchone()
        rows = connection.execute("SELECT row_number FROM decision_rows").fetchall()
    assert frozen == version
    assert rows == [(4,)]


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
