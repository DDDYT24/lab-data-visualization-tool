from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from labviz_api.config import Settings
from labviz_api.db.models import (
    CleaningDecisionRecord,
    CleaningDecisionSet,
    DatasetVersion,
    ProcessingRun,
    Project,
    ProjectRevision,
    QualityFindingRecord,
    QualityReportRecord,
    StoredObject,
)
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.parquet import read_parquet
from labviz_api.persistence.exceptions import (
    ObjectConfirmationPending,
    PersistenceUnavailable,
)
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import build_preview, build_quality_report
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.storage import LocalObjectStorage, ObjectInfo, StagedObject

API_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL = os.environ.get(
    "LABVIZ_TEST_POSTGRES_URL",
    "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test",
)


def alembic_config(url: str) -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = url
    return config


@pytest.fixture(scope="module")
def postgres_database() -> Iterator[Database]:
    database = Database(POSTGRES_URL)
    if not database.health().ready:
        database.dispose()
        pytest.skip("Local PostgreSQL is not running; start it with docker compose.")
    config = alembic_config(POSTGRES_URL)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    try:
        yield database
    finally:
        database.dispose()


@pytest.fixture(autouse=True)
def empty_postgres(postgres_database: Database) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "signal": [1.0, None, 3.0, 4.0],
            "note": ["a", "remove-me", "c", "d"],
        }
    )


def chart_document() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": "Signal over time",
        "xAxis": {"field": "time", "title": "Time", "unit": "s"},
        "yAxis": {"field": "signal", "title": "Signal", "unit": "V"},
        "series": [{"field": "signal", "label": "Signal", "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": "png",
            "dpi": 300,
            "sizePreset": "double-column",
            "grayscalePreview": False,
        },
    }


def complete_store(
    database: Database,
    storage: LocalObjectStorage,
) -> tuple[PostgresProjectStore, str, str, pd.DataFrame]:
    store = PostgresProjectStore(database, storage, 7_200)
    project_id = uuid4().hex
    job_id = uuid4().hex
    payload = b"time,signal,note\n0,1,a\n1,,remove-me\n2,3,c\n3,4,d\n"
    source = {
        "name": "quality.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="quality",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest="a" * 64,
    )
    frame = sample_frame()
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=build_preview(project_id, frame),
        quality=build_quality_report(project_id, frame),
        chart=chart_document(),
    )
    return store, project_id, job_id, frame


def test_quality_report_tracks_dataset_run_and_stable_finding_evidence(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, _job_id, _frame = complete_store(
        postgres_database, LocalObjectStorage(tmp_path / "objects")
    )
    with postgres_database.session() as session:
        project = session.get(Project, project_id)
        assert project is not None and project.current_revision is not None
        report = project.current_revision.quality_report
        version = project.current_revision.active_dataset_version
        assert report is not None
        assert report.dataset_version_id == version.id
        assert report.status == "completed"
        assert report.profiler_name == "labviz-quality"
        assert report.profiler_version == "1"
        assert report.algorithm_version == "quality-v1"
        assert report.code_version == "v2-phase3"
        assert report.processing_run.operation == "profile"
        assert report.processing_run.status == "succeeded"
        assert report.processing_run.input_dataset_version_id == version.id
        finding = session.scalar(
            select(QualityFindingRecord).where(
                QualityFindingRecord.quality_report_id == report.id,
                QualityFindingRecord.external_id == "missing:signal",
            )
        )
        assert finding is not None
        assert finding.column_identity is not None
        assert finding.column_identity["name"] == "signal"
        assert finding.column_identity["ordinal"] == 1
        assert finding.source_record_refs == [
            {
                "datasetVersionId": version.id.hex,
                "rowOrdinal": 2,
                "rowFingerprint": finding.source_record_refs[0]["rowFingerprint"],
            }
        ]
        assert len(finding.source_record_refs[0]["rowFingerprint"]) == 64
        assert finding.affected_count == 1
    quality = store.get_project(project_id, touch=False)
    assert quality is not None and '"missing:signal"' in quality["quality_json"]


def test_ignore_exclude_and_remove_create_immutable_sibling_versions(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, _job_id, baseline = complete_store(
        postgres_database, LocalObjectStorage(tmp_path / "objects")
    )
    finding_id = "missing:signal"

    store.save_decisions(project_id, [{"findingId": finding_id, "action": "ignore"}])
    assert len(store.load_cleaned_dataframe(project_id)) == 4
    assert len(store.load_chart_dataframe(project_id)) == 4

    store.save_decisions(project_id, [{"findingId": finding_id, "action": "exclude"}])
    assert len(store.load_cleaned_dataframe(project_id)) == 4
    assert len(store.load_chart_dataframe(project_id)) == 3

    store.save_decisions(project_id, [{"findingId": finding_id, "action": "remove"}])
    cleaned = store.load_cleaned_dataframe(project_id)
    assert len(cleaned) == 3
    assert "remove-me" not in cleaned["note"].tolist()
    assert len(store.load_chart_dataframe(project_id)) == 3

    with postgres_database.session() as session:
        sets = list(
            session.scalars(
                select(CleaningDecisionSet)
                .where(CleaningDecisionSet.project_id == project_id)
                .order_by(CleaningDecisionSet.revision_number)
            )
        )
        assert [item.revision_number for item in sets] == [1, 2, 3]
        actions = [
            session.scalar(
                select(CleaningDecisionRecord.action).where(
                    CleaningDecisionRecord.decision_set_id == item.id
                )
            )
            for item in sets
        ]
        assert actions == ["ignore", "exclude", "remove"]
        versions = list(
            session.scalars(
                select(DatasetVersion)
                .where(DatasetVersion.project_id == project_id)
                .order_by(DatasetVersion.version_number)
            )
        )
        baseline_id = versions[0].id
        assert [item.version_number for item in versions] == [1, 2, 3, 4]
        assert all(item.parent_version_id == baseline_id for item in versions[1:])
        assert [item.row_count for item in versions] == [4, 4, 4, 3]
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        store._load_version_dataframe(baseline_id).reset_index(drop=True),
        check_dtype=False,
    )


def test_derived_parquet_run_project_and_chart_lineage_are_reproducible(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store, project_id, _job_id, _baseline = complete_store(postgres_database, storage)
    store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "remove"}])

    with postgres_database.session() as session:
        project = session.get(Project, project_id)
        assert project is not None and project.current_revision is not None
        current = project.current_revision
        derived = current.active_dataset_version
        decision_set = current.cleaning_decision_set
        assert derived.kind == "cleaned"
        assert derived.parent_version is not None
        assert derived.parent_version.version_number == 1
        assert decision_set is not None
        assert derived.cleaning_decision_set_id == decision_set.id
        run = derived.output_of_run
        assert run is not None
        assert run.operation == "clean"
        assert run.status == "succeeded"
        assert run.input_dataset_version_id == derived.parent_version_id
        assert run.output_dataset_version_id == derived.id
        assert run.algorithm_version == "cleaning-decisions-v1"
        assert run.code_version == "v2-phase3"
        assert run.parameters["cleaningDecisionSetId"] == decision_set.id.hex
        assert current.chart_spec_revision.dataset_version_id == derived.id
        assert current.chart_spec_revision.cleaning_decision_set_id == decision_set.id
        first_revision = session.scalar(
            select(ProjectRevision).where(
                ProjectRevision.project_id == project.id,
                ProjectRevision.revision_number == 1,
            )
        )
        assert first_revision is not None
        assert first_revision.active_dataset_version_id == derived.parent_version_id
        assert first_revision.chart_spec_revision.dataset_version_id == derived.parent_version_id
        assert current.spec_document["cleaning"] == {
            "decisionSetId": str(decision_set.id),
            "revision": 1,
        }
        object_key = derived.stored_object.object_key
        expected_hash = derived.content_sha256
        assert expected_hash == derived.stored_object.sha256
    with storage.open(object_key) as stream:
        payload = stream.read()
    assert hashlib.sha256(payload).hexdigest() == expected_hash
    assert len(read_parquet(payload, expected_sha256=expected_hash)) == 3


def test_project_revision_restore_switches_dataset_chart_quality_and_decisions(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, _job_id, _frame = complete_store(
        postgres_database, LocalObjectStorage(tmp_path / "objects")
    )
    store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "remove"}])
    assert len(store.load_dataframe(project_id)) == 3
    assert store.get_decisions(project_id)[0]["action"] == "remove"

    store.restore_project_revision(project_id, 1)
    assert len(store.load_dataframe(project_id)) == 4
    assert store.get_decisions(project_id) == []

    store.restore_project_revision(project_id, 2)
    assert len(store.load_dataframe(project_id)) == 3
    assert store.get_decisions(project_id)[0]["action"] == "remove"
    with postgres_database.session() as session:
        revisions = list(
            session.scalars(select(ProjectRevision).where(ProjectRevision.project_id == project_id))
        )
        assert len(revisions) == 2


def test_quality_reprofile_creates_new_report_and_resets_to_its_input_version(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, _job_id, _frame = complete_store(
        postgres_database, LocalObjectStorage(tmp_path / "objects")
    )
    store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "remove"}])
    baseline = store.load_quality_dataframe(project_id)
    quality = build_quality_report(project_id, baseline, {"signal": (0, 2)})
    store.save_quality_report(
        project_id,
        quality,
        parameters={"validRanges": [{"field": "signal", "minimum": 0, "maximum": 2}]},
    )
    assert len(store.load_dataframe(project_id)) == 4
    assert store.get_decisions(project_id) == []

    with postgres_database.session() as session:
        reports = list(
            session.scalars(
                select(QualityReportRecord)
                .where(QualityReportRecord.project_id == project_id)
                .order_by(QualityReportRecord.revision_number)
            )
        )
        assert [item.revision_number for item in reports] == [1, 2]
        assert reports[0].report_document["findings"][0]["id"] == "missing:signal"
        assert any(
            item["id"] == "outside-range:signal" for item in reports[1].report_document["findings"]
        )
        project = session.get(Project, project_id)
        assert project is not None and project.current_revision is not None
        assert project.current_revision.quality_report_id == reports[1].id
        assert project.current_revision.cleaning_decision_set_id is None
        assert project.current_revision.active_dataset_version.version_number == 1


def test_immutable_lineage_trigger_and_decision_revision_uniqueness(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, _job_id, _frame = complete_store(
        postgres_database, LocalObjectStorage(tmp_path / "objects")
    )
    store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "ignore"}])
    with pytest.raises(DBAPIError), postgres_database.session() as session:
        session.execute(
            update(CleaningDecisionSet)
            .where(CleaningDecisionSet.project_id == project_id)
            .values(decisions_hash="f" * 64)
        )

    with pytest.raises(DBAPIError), postgres_database.session() as session:
        existing = session.scalar(
            select(CleaningDecisionSet).where(CleaningDecisionSet.project_id == project_id)
        )
        assert existing is not None
        duplicate = CleaningDecisionSet(
            project_id=existing.project_id,
            quality_report_id=existing.quality_report_id,
            input_dataset_version_id=existing.input_dataset_version_id,
            revision_number=existing.revision_number,
            decisions_hash="e" * 64,
        )
        session.add(duplicate)
        session.flush()


def test_cleaning_database_failure_discards_staged_object(
    tmp_path: Path,
    postgres_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store, project_id, _job_id, _frame = complete_store(postgres_database, storage)

    def fail_database(**_kwargs: Any) -> bool:
        raise SQLAlchemyError("injected cleaning transaction failure")

    monkeypatch.setattr(store, "_persist_cleaning_result", fail_database)
    with pytest.raises(PersistenceUnavailable):
        store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "remove"}])
    assert not any((tmp_path / "objects").rglob("*.part"))
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(DatasetVersion)) == 1
        assert connection.scalar(select(func.count()).select_from(CleaningDecisionSet)) == 0
        assert connection.scalar(select(func.count()).select_from(StoredObject)) == 1


class ControlledConfirmStorage(LocalObjectStorage):
    fail_next_confirm = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        if self.fail_next_confirm:
            self.fail_next_confirm = False
            raise OSError("injected cleaned object confirmation failure")
        return super().confirm(staged)


def test_cleaning_confirmation_failure_is_recoverable(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ControlledConfirmStorage(tmp_path / "objects")
    store, project_id, _job_id, _frame = complete_store(postgres_database, storage)
    storage.fail_next_confirm = True
    with pytest.raises(ObjectConfirmationPending):
        store.save_decisions(project_id, [{"findingId": "missing:signal", "action": "remove"}])
    pending = store.get_project(project_id, touch=False)
    assert pending is not None and pending["ready"] is False
    with postgres_database.engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count())
                .select_from(StoredObject)
                .where(StoredObject.status == "pending")
            )
            == 1
        )
    assert store.recover_pending_objects() == 1
    recovered = store.get_project(project_id, touch=False)
    assert recovered is not None and recovered["ready"] is True
    assert len(store.load_dataframe(project_id)) == 3
    with postgres_database.session() as session:
        clean_run = session.scalar(
            select(ProcessingRun).where(
                ProcessingRun.project_id == project_id,
                ProcessingRun.operation == "clean",
            )
        )
        assert clean_run is not None
        assert clean_run.status == "succeeded"
        assert clean_run.output_dataset_version_id is not None


def _api_settings(root: Path, *, backend: str) -> Settings:
    return Settings(
        database_path=root / "reference.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
        environment="test",
        postgres_url=POSTGRES_URL,
        object_storage_root=root / "objects",
        persistence_backend=backend,
    )


def _exercise_quality_api(settings: Settings) -> dict[str, Any]:
    payload = b"time,signal,note\n0,1,a\n1,,remove-me\n2,3,c\n3,4,d\n"
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/projects",
            files={"file": ("quality.csv", payload, "text/csv")},
        )
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        quality = client.get(f"/api/v1/projects/{project_id}/quality")
        assert quality.status_code == 200
        finding_id = next(
            item["id"] for item in quality.json()["findings"] if item["kind"] == "missing"
        )
        excluded = client.patch(
            f"/api/v1/projects/{project_id}/cleaning-decisions",
            json={"decisions": [{"findingId": finding_id, "action": "exclude"}]},
        )
        excluded_csv = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
        removed = client.patch(
            f"/api/v1/projects/{project_id}/cleaning-decisions",
            json={"decisions": [{"findingId": finding_id, "action": "remove"}]},
        )
        removed_csv = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
        reprofiled = client.put(
            f"/api/v1/projects/{project_id}/quality-rules",
            json={"ranges": [{"field": "signal", "minimum": 0, "maximum": 2}]},
        )
        assert excluded.status_code == removed.status_code == reprofiled.status_code == 200
        assert excluded_csv.status_code == removed_csv.status_code == 200
        return {
            "quality": {**quality.json(), "projectId": "dynamic"},
            "excluded": {
                **excluded.json(),
                "projectId": "dynamic",
                "updatedAt": "dynamic",
            },
            "removed": {
                **removed.json(),
                "projectId": "dynamic",
                "updatedAt": "dynamic",
            },
            "reprofiled": {**reprofiled.json(), "projectId": "dynamic"},
            "excludedContainsRow": "remove-me" in excluded_csv.text,
            "removedContainsRow": "remove-me" in removed_csv.text,
        }


def test_sqlite_and_postgresql_quality_cleaning_api_responses_match(
    tmp_path: Path,
) -> None:
    sqlite_settings = _api_settings(tmp_path / "sqlite", backend="sqlite")
    postgres_settings = _api_settings(tmp_path / "postgres", backend="postgresql")
    sqlite_result = _exercise_quality_api(sqlite_settings)
    postgres_result = _exercise_quality_api(postgres_settings)
    assert sqlite_result == postgres_result
    assert sqlite_result["excludedContainsRow"] is True
    assert sqlite_result["removedContainsRow"] is False
    with sqlite3.connect(postgres_settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_real_xlsx_quality_cleaning_can_reopen_after_restart(
    tmp_path: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Measurements"
    sheet.append(["time_s", "signal_v", "note"])
    sheet.append([0, 1.2, "keep"])
    sheet.append([1, None, "remove-me"])
    sheet.append([2, 1.8, "keep"])
    payload = io.BytesIO()
    workbook.save(payload)
    settings = _api_settings(tmp_path / "restart", backend="postgresql")
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            files={
                "file": (
                    "measurements.xlsx",
                    payload.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={"sheetName": "Measurements", "headerRow": "1"},
        )
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        cookie = client.cookies.get("labviz_guest")
        assert cookie is not None
        quality = client.get(f"/api/v1/projects/{project_id}/quality").json()
        finding_id = next(item["id"] for item in quality["findings"] if item["kind"] == "missing")
        decision = client.patch(
            f"/api/v1/projects/{project_id}/cleaning-decisions",
            json={"decisions": [{"findingId": finding_id, "action": "remove"}]},
        )
        assert decision.status_code == 200

    restarted_store = PostgresProjectStore(
        Database(POSTGRES_URL),
        LocalObjectStorage(settings.object_storage_root),
        settings.project_ttl_seconds,
    )
    reference = SqliteReferenceRepository(settings.database_path, settings.project_ttl_seconds)
    try:
        with TestClient(
            create_app(settings, repository=reference, project_store=restarted_store)
        ) as restarted:
            restarted.cookies.set("labviz_guest", cookie)
            reopened = restarted.get(f"/api/v1/projects/{project_id}")
            quality = restarted.get(f"/api/v1/projects/{project_id}/quality")
            cleaned = restarted.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
            assert reopened.status_code == quality.status_code == cleaned.status_code == 200
            assert "remove-me" not in cleaned.text
            assert restarted_store.load_dataframe(project_id).shape == (2, 3)
            assert restarted_store.get_decisions(project_id) == [
                {"findingId": finding_id, "action": "remove"}
            ]
    finally:
        restarted_store.dispose()
