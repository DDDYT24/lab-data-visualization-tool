from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from labviz_api.artifacts import local_artifact_store
from labviz_api.local_schema import (
    CURRENT_REVISION,
    install_local_baseline,
    prepare_local_database,
)
from labviz_api.models import ChartSpec
from labviz_api.parquet import read_parquet
from labviz_api.persistence.sqlite import SqliteProjectStore
from labviz_api.processing import (
    analyze_chart,
    build_preview,
    build_quality_report,
    default_chart_spec,
    serialize_dataframe,
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


def legacy_profile(database: Path, frame: pd.DataFrame) -> dict[str, object]:
    """A profile written before 0102: the parsed table is a zlib JSON BLOB on the project."""
    install_local_baseline(database, project_ttl_seconds=TTL)
    preview = build_preview("p", frame)
    preview["columns"][1]["unit"] = "mV"
    chart = default_chart_spec(frame)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at, chart_json, "
            "preview_json, quality_json, data_blob) VALUES ('p', 'Kept', '{}', 'local', "
            "'2026-01-01T00:00:00Z', ?, ?, ?, ?)",
            (
                json.dumps(chart),
                json.dumps(preview),
                json.dumps(build_quality_report("p", frame)),
                serialize_dataframe(frame),
            ),
        )
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('pending', 'Uploading', '{}', 'local', '2026-01-01T00:00:00Z')"
        )
    return chart


def store(database: Path) -> SqliteProjectStore:
    return SqliteProjectStore(ProjectRepository(database, TTL))


def test_upgrading_turns_each_data_blob_into_a_parquet_dataset_version(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database, measurements())

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        columns = {row[1] for row in connection.execute("PRAGMA table_info(projects)")}
        projects = {
            row["id"]: row["dataset_version_id"]
            for row in connection.execute("SELECT id, dataset_version_id FROM projects")
        }
        version = connection.execute("SELECT * FROM dataset_versions").fetchone()
    assert "data_blob" not in columns
    assert projects["pending"] is None
    assert projects["p"] == version["id"]
    assert (version["kind"], version["version_number"]) == ("parsed", 1)
    assert (version["row_count"], version["column_count"]) == (6, 3)
    units = {item["name"]: item["unit"] for item in json.loads(version["schema_json"])["columns"]}
    assert units == {"Time": None, "Signal": "mV", "Group": None}

    payload = local_artifact_store(database).read(version["content_sha256"])
    frame = read_parquet(payload, expected_sha256=version["content_sha256"])
    expected = measurements()
    assert frame["Time"].tolist() == expected["Time"].tolist()
    assert frame["Signal"].isna().tolist() == expected["Signal"].isna().tolist()
    assert frame["Signal"].dropna().tolist() == expected["Signal"].dropna().tolist()
    assert frame["Group"].isna().tolist() == expected["Group"].isna().tolist()


def test_a_migrated_project_opens_and_draws_the_same_chart(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chart = ChartSpec.model_validate(legacy_profile(database, measurements()))
    before = analyze_chart(measurements(), chart)

    prepare_local_database(database, project_ttl_seconds=TTL)
    projects = store(database)

    project = projects.get_project("p")
    assert project is not None and project["ready"]
    assert analyze_chart(projects.load_quality_dataframe("p"), chart) == before


def test_new_uploads_are_versioned_and_reprocessing_adds_a_version(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    prepare_local_database(database, project_ttl_seconds=TTL)
    projects = store(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('p', 'New', '{}', 'local', '2026-01-01T00:00:00Z')"
        )
    frame = measurements()
    for _ in range(2):
        projects.complete_project(
            project_id="p",
            source={},
            frame=frame,
            preview=build_preview("p", frame),
            quality=build_quality_report("p", frame),
            chart=default_chart_spec(frame),
        )

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT version_number, content_sha256 FROM dataset_versions")
        numbers, digests = zip(*rows, strict=True)
    assert sorted(numbers) == [1, 2]
    # Parquet v1 bytes are canonical, so the same table always has the same digest.
    assert len(set(digests)) == 1
    assert projects.get_project("p")["ready"]  # type: ignore[index]
    assert projects.load_quality_dataframe("p")["Time"].tolist() == frame["Time"].tolist()


def test_a_copy_shares_the_bytes_and_survives_deleting_the_original(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database, measurements())
    prepare_local_database(database, project_ttl_seconds=TTL)
    repository = ProjectRepository(database, TTL)
    digest = repository.dataset_digest("p")
    assert digest is not None

    repository.duplicate_project(
        source_project_id="p", project_id="copy", job_id="job", owner_user_id="user"
    )
    assert repository.dataset_digest("copy") == digest

    assert repository.delete_project("p")
    repository.cleanup_expired()
    assert repository.artifacts.exists(digest)
    assert store(database).load_quality_dataframe("copy")["Time"].tolist() == [0, 1, 2, 3, 4, 5]

    assert repository.delete_project("copy")
    repository.cleanup_expired()
    assert not repository.artifacts.exists(digest)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM dataset_versions").fetchone() == (0,)
