from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from labviz_api.config import Settings
from labviz_api.db.models import (
    ChartSpecRevision,
    DatasetVersion,
    Project,
    ProjectRevision,
    StoredObject,
    User,
)
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.parquet import ParquetContractError, read_parquet, write_parquet
from labviz_api.persistence.contracts import ProjectStore
from labviz_api.persistence.exceptions import (
    ObjectConfirmationPending,
    PersistenceNotFound,
    PersistenceUnavailable,
)
from labviz_api.persistence.postgres import (
    PostgresProjectStore,
    SqlAlchemyUnitOfWork,
)
from labviz_api.persistence.sqlite import SqliteProjectStore
from labviz_api.processing import build_preview, build_quality_report, default_chart_spec
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
    with database.engine.begin() as connection:
        if "projects" in inspect(connection).get_table_names():
            connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
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


@pytest.fixture(params=["sqlite", "postgresql"])
def project_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    postgres_database: Database,
) -> ProjectStore:
    if request.param == "sqlite":
        reference = SqliteReferenceRepository(tmp_path / "reference.db", 7_200)
        return SqliteProjectStore(reference)
    return PostgresProjectStore(
        postgres_database,
        LocalObjectStorage(tmp_path / "objects"),
        7_200,
    )


def chart_document(title: str = "Response over time") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": title,
        "xAxis": {"field": "time", "title": "Time", "unit": "s"},
        "yAxis": {"field": "response", "title": "Response", "unit": "V"},
        "series": [{"field": "response", "label": "Response", "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": "png",
            "dpi": 300,
            "sizePreset": "double-column",
            "grayscalePreview": False,
        },
    }


def complete_store(
    store: ProjectStore,
    *,
    project_id: str | None = None,
    job_id: str | None = None,
) -> tuple[str, str, pd.DataFrame]:
    resolved_project_id = project_id or uuid4().hex
    resolved_job_id = job_id or uuid4().hex
    payload = b"time,response\n0,1.2\n1,1.4\n2,\n"
    source = {
        "name": "experiment.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": None,
    }
    store.create_project(
        project_id=resolved_project_id,
        job_id=resolved_job_id,
        title="experiment",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest="a" * 64,
    )
    store.update_job(
        resolved_job_id,
        stage="parsing",
        progress=45,
        message="Reading the table and identifying columns.",
    )
    frame = pd.DataFrame({"time": [0, 1, 2], "response": [1.2, 1.4, None]})
    processed_source = {**source, "headerRow": 1}
    store.complete_project(
        project_id=resolved_project_id,
        source=processed_source,
        frame=frame,
        preview=build_preview(resolved_project_id, frame),
        quality=build_quality_report(resolved_project_id, frame),
        chart=default_chart_spec(frame),
    )
    store.update_job(
        resolved_job_id,
        stage="ready",
        progress=100,
        message="Your data is ready to inspect.",
    )
    return resolved_project_id, resolved_job_id, frame


def test_same_project_repository_contract_runs_on_both_backends(
    project_store: ProjectStore,
) -> None:
    project_id, job_id, _frame = complete_store(project_store)
    project = project_store.get_project(project_id, touch=False)
    assert project is not None
    assert project["ready"] is True
    assert project["storage_mode"] == "temporary-cloud"
    assert project["guest_token_digest"] == "a" * 64
    assert project_store.get_job(job_id)["stage"] == "ready"  # type: ignore[index]
    preview = project["preview_json"]
    assert preview is not None and '"totalRows": 3' in preview

    updated_at = project_store.save_chart(project_id, chart_document("Changed title"))
    assert updated_at.endswith("Z")
    changed = project_store.get_project(project_id, touch=False)
    assert changed is not None and '"Changed title"' in changed["chart_json"]


def test_unit_of_work_rolls_back_without_explicit_commit(
    postgres_database: Database,
) -> None:
    project_id = uuid4()
    with SqlAlchemyUnitOfWork(postgres_database) as uow:
        assert uow.session is not None
        uow.session.add(
            Project(
                id=project_id,
                storage_mode="local",
                title="rolled back",
                description="",
            )
        )
    with postgres_database.engine.connect() as connection:
        count = connection.scalar(
            select(func.count()).select_from(Project).where(Project.id == project_id)
        )
    assert count == 0


def test_duplicate_completion_is_idempotent(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, _job_id, frame = complete_store(store)
    project = store.get_project(project_id, touch=False)
    assert project is not None
    source = cast(dict[str, Any], __import__("json").loads(project["source_json"]))
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=build_preview(project_id, frame),
        quality=build_quality_report(project_id, frame),
        chart=default_chart_spec(frame),
    )
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(DatasetVersion)) == 1
        assert connection.scalar(select(func.count()).select_from(ProjectRevision)) == 1
        assert connection.scalar(select(func.count()).select_from(StoredObject)) == 1


def test_staged_object_is_discarded_when_database_transaction_fails(
    tmp_path: Path,
    postgres_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id = uuid4().hex
    job_id = uuid4().hex
    payload = b"time,response\n0,1\n"
    source = {
        "name": "failed.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="failed",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest="b" * 64,
    )

    def fail_database(**_kwargs: Any) -> bool:
        raise SQLAlchemyError("injected transaction failure")

    monkeypatch.setattr(store, "_persist_processed_project", fail_database)
    frame = pd.DataFrame({"time": [0], "response": [1.0]})
    with pytest.raises(PersistenceUnavailable):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=default_chart_spec(frame),
        )
    assert not any((tmp_path / "objects").rglob("*.part"))
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(StoredObject)) == 0


def test_staged_object_is_discarded_when_parquet_validation_fails(
    tmp_path: Path,
    postgres_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id = uuid4().hex
    job_id = uuid4().hex
    source = {
        "name": "invalid.csv",
        "size": 18,
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="invalid",
        source=source,
        source_sha256="d" * 64,
        guest_token_digest="e" * 64,
    )

    def fail_validation(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise ParquetContractError("injected validation failure")

    monkeypatch.setattr("labviz_api.persistence.postgres.read_parquet", fail_validation)
    frame = pd.DataFrame({"time": [0], "response": [1.0]})
    with pytest.raises(PersistenceUnavailable):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=default_chart_spec(frame),
        )
    assert not any((tmp_path / "objects").rglob("*.part"))
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(StoredObject)) == 0


class FailFirstConfirmStorage(LocalObjectStorage):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failures_remaining = 1

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("injected final confirmation failure")
        return super().confirm(staged)


def test_database_commit_survives_confirmation_failure_and_recovers(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailFirstConfirmStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id = uuid4().hex
    job_id = uuid4().hex
    payload = b"time,response\n0,1\n"
    source = {
        "name": "recover.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="recover",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest="c" * 64,
    )
    frame = pd.DataFrame({"time": [0], "response": [1.0]})
    with pytest.raises(ObjectConfirmationPending):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=default_chart_spec(frame),
        )
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


def test_parquet_v1_round_trip_units_missing_values_and_hash() -> None:
    frame = pd.DataFrame(
        {
            "count": pd.Series([1, None, 3], dtype="Int64"),
            "signal": [1.25, float("nan"), 3.5],
            "label": ["a", None, "c"],
            "accepted": pd.Series([True, None, False], dtype="boolean"),
            "recorded_at": pd.to_datetime(
                ["2026-01-01T00:00:00Z", None, "2026-01-03T00:00:00Z"], utc=True
            ),
        }
    )
    artifact = write_parquet(frame, units={"signal": "mV"})
    reopened = read_parquet(artifact.payload, expected_sha256=artifact.sha256)
    assert len(reopened) == 3
    assert pd.isna(reopened.loc[1, "signal"])
    assert pd.isna(reopened.loc[1, "label"])
    signal = next(
        column for column in artifact.schema_document["columns"] if column["name"] == "signal"
    )
    assert signal["unit"] == "mV"
    assert artifact.schema_document["missingValues"] == "arrow-null"
    assert artifact.provenance == {
        "provenanceVersion": "1",
        "pandasVersion": pd.__version__,
        "pyarrowVersion": pa.__version__,
        "parquetWriter": "pyarrow.parquet.write_table",
        "parquetWriterVersion": pa.__version__,
    }
    corrupted = artifact.payload[:-1] + bytes([artifact.payload[-1] ^ 1])
    with pytest.raises(ParquetContractError, match="hash"):
        read_parquet(corrupted, expected_sha256=artifact.sha256)


def test_dataset_reopen_and_project_revision_restore(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, _job_id, original = complete_store(store)
    store.save_chart(project_id, chart_document("Revision two"))
    store.save_chart(project_id, chart_document("Revision three"))
    restored_at = store.restore_project_revision(project_id, 2)
    assert restored_at.endswith("Z")
    restored = store.get_project(project_id, touch=False)
    assert restored is not None and '"Revision two"' in restored["chart_json"]
    pd.testing.assert_frame_equal(
        store.load_dataframe(project_id).reset_index(drop=True),
        original.convert_dtypes().reset_index(drop=True),
        check_dtype=False,
    )
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(DatasetVersion)) == 1
        assert connection.scalar(select(func.count()).select_from(ProjectRevision)) == 3
        assert connection.scalar(select(func.count()).select_from(ChartSpecRevision)) == 3


def test_restore_service_enforces_time_window_beyond_database_equality_constraint(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    now = datetime.now(UTC)
    owner = User(email="restore-window@example.com")
    recoverable = Project(
        owner=owner,
        storage_mode="saved-cloud",
        title="recoverable",
        description="",
        deleted_at=now,
        purge_after=now + timedelta(hours=24),
    )
    expired_deleted_at = now - timedelta(hours=25)
    expired = Project(
        owner=owner,
        storage_mode="saved-cloud",
        title="expired",
        description="",
        deleted_at=expired_deleted_at,
        purge_after=expired_deleted_at + timedelta(hours=24),
    )
    with postgres_database.session() as session:
        session.add_all([recoverable, expired])
    assert store.get_project(recoverable.id.hex, touch=False) is None
    assert store.restore_deleted_project(recoverable.id.hex).endswith("Z")
    with pytest.raises(PersistenceNotFound, match="expired"):
        store.restore_deleted_project(expired.id.hex)


def test_database_rejects_a_saved_project_recovery_window_other_than_24_hours(
    postgres_database: Database,
) -> None:
    now = datetime.now(UTC)
    with pytest.raises(IntegrityError), postgres_database.session() as session:
        session.add(
            Project(
                owner=User(email="wrong-window@example.com"),
                storage_mode="saved-cloud",
                title="wrong window",
                description="",
                deleted_at=now,
                purge_after=now + timedelta(hours=23),
            )
        )


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


def _exercise_migrated_api(settings: Settings) -> dict[str, Any]:
    with TestClient(create_app(settings)) as client:
        created_response = client.post(
            "/api/v1/projects",
            files={"file": ("api.csv", b"time,response\n0,1.2\n1,1.4\n", "text/csv")},
        )
        assert created_response.status_code == 202
        created = created_response.json()
        project_id = created["projectId"]
        opened_response = client.get(f"/api/v1/projects/{project_id}")
        preview_response = client.get(f"/api/v1/projects/{project_id}/preview")
        chart_response = client.put(
            f"/api/v1/projects/{project_id}/chart",
            json={"chart": chart_document("API chart")},
        )
        analysis_response = client.post(
            f"/api/v1/projects/{project_id}/chart-analysis",
            json={"chart": chart_document("API chart")},
        )
        assert opened_response.status_code == preview_response.status_code == 200
        assert chart_response.status_code == 200
        assert analysis_response.status_code == 200
        return {
            "created": created,
            "opened": opened_response.json(),
            "preview": preview_response.json(),
            "chart": chart_response.json(),
            "analysis": analysis_response.json(),
        }


def _without_dynamic_fields(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("projectId", None)
    normalized.pop("expiresAt", None)
    normalized.pop("updatedAt", None)
    job = normalized.get("job")
    if isinstance(job, dict):
        normalized["job"] = {
            key: value for key, value in job.items() if key not in {"id", "projectId"}
        }
    rows = normalized.get("rows")
    if isinstance(rows, list):
        normalized["rows"] = rows
    return normalized


def test_sqlite_and_postgresql_api_responses_remain_compatible(
    tmp_path: Path,
) -> None:
    sqlite_settings = _api_settings(tmp_path / "sqlite", backend="sqlite")
    postgres_settings = _api_settings(tmp_path / "postgres", backend="postgresql")
    sqlite = _exercise_migrated_api(sqlite_settings)
    postgres = _exercise_migrated_api(postgres_settings)
    for key in ("created", "opened", "preview", "chart", "analysis"):
        assert _without_dynamic_fields(sqlite[key]) == _without_dynamic_fields(postgres[key])
    with sqlite3.connect(postgres_settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_postgresql_chart_analysis_preserves_access_and_missing_project_errors(
    tmp_path: Path,
) -> None:
    settings = _api_settings(tmp_path / "postgres-analysis", backend="postgresql")
    app = create_app(settings)
    with TestClient(app) as owner_client:
        created = owner_client.post(
            "/api/v1/projects",
            files={"file": ("analysis.csv", b"time,response\n0,1.2\n1,1.4\n", "text/csv")},
        )
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        analysis = owner_client.post(
            f"/api/v1/projects/{project_id}/chart-analysis",
            json={"chart": chart_document()},
        )
        assert analysis.status_code == 200
        assert analysis.json()["projectId"] == project_id

    with TestClient(app) as other_client:
        denied = other_client.post(
            f"/api/v1/projects/{project_id}/chart-analysis",
            json={"chart": chart_document()},
        )
        missing = other_client.post(
            f"/api/v1/projects/{uuid4().hex}/chart-analysis",
            json={"chart": chart_document()},
        )
        assert denied.status_code == 403
        assert denied.json()["code"] == "project-access-denied"
        assert missing.status_code == 404
        assert missing.json()["code"] == "project-not-found"

    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_real_xlsx_upload_can_be_reopened_after_application_restart(
    tmp_path: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Measurements"
    sheet.append(["time_s", "signal_v"])
    sheet.append([0, 1.2])
    sheet.append([1, 1.5])
    payload = io.BytesIO()
    workbook.save(payload)
    settings = _api_settings(tmp_path / "restart", backend="postgresql")

    with TestClient(create_app(settings)) as first_client:
        created = first_client.post(
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
        guest_cookie = first_client.cookies.get("labviz_guest")
        assert guest_cookie

    restarted_store = PostgresProjectStore(
        Database(POSTGRES_URL),
        LocalObjectStorage(settings.object_storage_root),
        settings.project_ttl_seconds,
    )
    reference = SqliteReferenceRepository(settings.database_path, settings.project_ttl_seconds)
    try:
        with TestClient(
            create_app(settings, repository=reference, project_store=restarted_store)
        ) as restarted_client:
            restarted_client.cookies.set("labviz_guest", guest_cookie)
            reopened = restarted_client.get(f"/api/v1/projects/{project_id}")
            preview = restarted_client.get(f"/api/v1/projects/{project_id}/preview")
            assert reopened.status_code == preview.status_code == 200
            assert reopened.json()["source"]["sheetName"] == "Measurements"
            assert [column["field"] for column in preview.json()["columns"]] == [
                "time_s",
                "signal_v",
            ]
            assert restarted_store.load_dataframe(project_id).shape == (2, 2)
    finally:
        restarted_store.dispose()
