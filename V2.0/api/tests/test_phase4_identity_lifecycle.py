from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.db.models import (
    DatasetVersion,
    GuestSession,
    IdempotencyRecord,
    ProcessingRun,
    Project,
    ProjectClaim,
    ProjectLifecycleEvent,
    ProjectOrigin,
    ProjectRevision,
    QualityFindingRecord,
    QualityReportRecord,
    StoredObject,
    User,
)
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.persistence.exceptions import PersistenceConflict, PersistenceNotFound
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import build_preview, build_quality_report, default_chart_spec
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.storage import LocalObjectStorage

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


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "signal": [1.0, None, 3.0, 4.0],
            "note": ["a", "remove-me", "c", "d"],
        }
    )


def _complete_project(
    store: PostgresProjectStore,
    *,
    guest_digest: str,
) -> tuple[str, str]:
    project_id = uuid4().hex
    job_id = uuid4().hex
    payload = b"time,signal,note\n0,1,a\n1,,remove-me\n2,3,c\n3,4,d\n"
    source = {
        "name": "phase4.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="phase4",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest=guest_digest,
    )
    frame = _frame()
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=build_preview(project_id, frame),
        quality=build_quality_report(project_id, frame),
        chart=default_chart_spec(frame),
    )
    store.update_job(
        job_id,
        stage="ready",
        progress=100,
        message="Project is ready.",
    )
    return project_id, job_id


def _add_user(database: Database, email: str) -> UUID:
    user_id = uuid4()
    with database.session() as session:
        session.add(User(id=user_id, email=email))
    return user_id


def _project_object_ids(database: Database, project_id: str) -> set[UUID]:
    with database.session() as session:
        return set(
            session.scalars(
                select(DatasetVersion.stored_object_id).where(
                    DatasetVersion.project_id == UUID(project_id)
                )
            )
        )


def _chart(title: str) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": title,
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


def test_guest_claim_and_save_are_locked_in_place_and_idempotent(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, _job_id = _complete_project(store, guest_digest="a" * 64)
    owner_id = _add_user(postgres_database, "owner@example.com")
    other_owner_id = _add_user(postgres_database, "other@example.com")
    object_ids_before = _project_object_ids(postgres_database, project_id)

    with pytest.raises(PersistenceConflict, match="GuestSession"):
        store.save_project(
            project_id,
            owner_id.hex,
            guest_token_digest="b" * 64,
        )

    store.save_project(project_id, owner_id.hex, guest_token_digest="a" * 64)
    store.save_project(project_id, owner_id.hex, guest_token_digest=None)
    with pytest.raises(PersistenceConflict, match="another user"):
        store.save_project(project_id, other_owner_id.hex, guest_token_digest="a" * 64)

    with postgres_database.session() as session:
        project = session.get(Project, UUID(project_id))
        assert project is not None
        assert project.storage_mode == "saved-cloud"
        assert project.owner_user_id == owner_id
        assert project.guest_session_id is None
        assert project.expires_at is None
        assert project.saved_at is not None
        claim = session.get(ProjectClaim, project.id)
        assert claim is not None
        assert claim.user_uuid_snapshot == owner_id
        assert claim.guest_session_uuid_snapshot is not None
        assert session.get(GuestSession, claim.guest_session_uuid_snapshot) is not None
        event_types = list(
            session.scalars(
                select(ProjectLifecycleEvent.event_type)
                .where(ProjectLifecycleEvent.project_uuid_snapshot == project.id)
                .order_by(ProjectLifecycleEvent.created_at)
            )
        )
        assert event_types == ["claim", "save"]

    assert _project_object_ids(postgres_database, project_id) == object_ids_before


def test_dedup_scope_isolated_across_guests_and_content_keys_ignore_project_id(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    first_id, _ = _complete_project(store, guest_digest="a" * 64)
    second_id, _ = _complete_project(store, guest_digest="b" * 64)
    first_objects = _project_object_ids(postgres_database, first_id)
    second_objects = _project_object_ids(postgres_database, second_id)
    assert first_objects.isdisjoint(second_objects)

    with postgres_database.session() as session:
        first = session.get(Project, UUID(first_id))
        second = session.get(Project, UUID(second_id))
        assert first is not None and second is not None
        assert first.guest_session_id is not None and second.guest_session_id is not None
        assert first.guest_session_id != second.guest_session_id
        rows = list(session.scalars(select(StoredObject).order_by(StoredObject.created_at)))
        assert {row.dedup_scope for row in rows} == {
            f"guest:{first.guest_session_id.hex}",
            f"guest:{second.guest_session_id.hex}",
        }
        assert all(
            first_id not in row.object_key and second_id not in row.object_key for row in rows
        )
        assert len({row.object_key for row in rows}) == 2


def test_duplicate_copies_only_current_reproducible_closure_and_is_idempotent(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    source_id, _ = _complete_project(store, guest_digest="a" * 64)
    store.save_decisions(
        source_id,
        [{"findingId": "missing:signal", "action": "remove"}],
    )
    store.save_chart(source_id, _chart("Current source chart"))
    owner_id = _add_user(postgres_database, "duplicate-owner@example.com")
    store.save_project(source_id, owner_id.hex, guest_token_digest="a" * 64)
    source_object_ids = _project_object_ids(postgres_database, source_id)

    duplicate_id = uuid4().hex
    returned_id = store.duplicate_project(
        source_project_id=source_id,
        project_id=duplicate_id,
        job_id=uuid4().hex,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
        idempotency_key="duplicate-current-revision",
    )
    repeated_id = store.duplicate_project(
        source_project_id=source_id,
        project_id=uuid4().hex,
        job_id=uuid4().hex,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
        idempotency_key="duplicate-current-revision",
    )
    assert returned_id == repeated_id == duplicate_id
    assert _project_object_ids(postgres_database, duplicate_id) == source_object_ids

    with postgres_database.session() as session:
        source = session.get(Project, UUID(source_id))
        duplicate = session.get(Project, UUID(duplicate_id))
        assert source is not None and source.current_revision is not None
        assert duplicate is not None and duplicate.current_revision is not None
        assert duplicate.storage_mode == "saved-cloud"
        assert duplicate.owner_user_id == owner_id
        assert duplicate.id != source.id
        assert duplicate.current_revision.id != source.current_revision.id
        assert duplicate.current_revision.revision_number == 1
        assert duplicate.current_revision.active_dataset_version_id != (
            source.current_revision.active_dataset_version_id
        )
        assert duplicate.current_revision.chart_spec_revision_id != (
            source.current_revision.chart_spec_revision_id
        )
        assert duplicate.current_revision.quality_report_id != (
            source.current_revision.quality_report_id
        )
        assert duplicate.current_revision.cleaning_decision_set_id != (
            source.current_revision.cleaning_decision_set_id
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(ProjectRevision)
                .where(ProjectRevision.project_id == duplicate.id)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(QualityReportRecord)
                .where(QualityReportRecord.project_id == duplicate.id)
            )
            == 1
        )
        finding_count = session.scalar(
            select(func.count())
            .select_from(QualityFindingRecord)
            .where(QualityFindingRecord.project_id == duplicate.id)
        )
        assert finding_count is not None and finding_count > 0
        reused_runs = list(
            session.scalars(select(ProcessingRun).where(ProcessingRun.project_id == duplicate.id))
        )
        assert reused_runs
        assert all(run.execution_mode == "reused-result" for run in reused_runs)
        assert all(run.origin_run_uuid_snapshot is not None for run in reused_runs)
        origin = session.get(ProjectOrigin, duplicate.id)
        assert origin is not None
        assert origin.source_project_id == source.id
        assert origin.source_revision_id == source.current_revision.id
        assert origin.source_project_uuid_snapshot == source.id
        assert origin.source_revision_uuid_snapshot == source.current_revision.id
        assert (
            session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(IdempotencyRecord.actor_user_id == owner_id)
            )
            == 1
        )


def test_soft_delete_restore_purge_and_gc_respect_shared_foreign_keys(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    source_id, _ = _complete_project(store, guest_digest="a" * 64)
    owner_id = _add_user(postgres_database, "recovery-owner@example.com")
    store.save_project(source_id, owner_id.hex, guest_token_digest="a" * 64)
    duplicate_id = uuid4().hex
    store.duplicate_project(
        source_project_id=source_id,
        project_id=duplicate_id,
        job_id=uuid4().hex,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )
    shared_object_ids = _project_object_ids(postgres_database, source_id)
    assert _project_object_ids(postgres_database, duplicate_id) == shared_object_ids

    assert store.delete_project(
        source_id,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )
    assert store.get_project(source_id, touch=False) is None
    assert [item["id"] for item in store.list_deleted_projects(owner_id.hex)] == [source_id]
    store.restore_deleted_project(source_id, owner_id.hex)
    assert store.get_project(source_id, touch=False) is not None

    store.delete_project(source_id, owner_user_id=owner_id.hex, guest_token_digest=None)
    expired_deleted_at = datetime.now(UTC) - timedelta(hours=25)
    with postgres_database.session() as session:
        source = session.get(Project, UUID(source_id))
        assert source is not None
        source.deleted_at = expired_deleted_at
        source.purge_after = expired_deleted_at + timedelta(hours=24)
    store.cleanup_expired()

    with postgres_database.session() as session:
        assert session.get(Project, UUID(source_id)) is None
        origin = session.get(ProjectOrigin, UUID(duplicate_id))
        assert origin is not None
        assert origin.source_project_id is None
        assert origin.source_revision_id is None
        assert origin.source_project_uuid_snapshot == UUID(source_id)
        objects = list(
            session.scalars(select(StoredObject).where(StoredObject.id.in_(shared_object_ids)))
        )
        assert objects and all(item.status == "available" for item in objects)
        object_keys = [item.object_key for item in objects]
    assert all(storage.exists(key) for key in object_keys)

    store.delete_project(duplicate_id, owner_user_id=owner_id.hex, guest_token_digest=None)
    with postgres_database.session() as session:
        duplicate = session.get(Project, UUID(duplicate_id))
        assert duplicate is not None and duplicate.deleted_at is not None
        duplicate.deleted_at = expired_deleted_at
        duplicate.purge_after = expired_deleted_at + timedelta(hours=24)
    store.cleanup_expired()
    with postgres_database.session() as session:
        objects = list(
            session.scalars(select(StoredObject).where(StoredObject.id.in_(shared_object_ids)))
        )
        assert objects and all(item.status == "deleted" for item in objects)
    assert all(not storage.exists(key) for key in object_keys)


def test_duplicate_origin_snapshots_are_database_immutable(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    source_id, _ = _complete_project(store, guest_digest="a" * 64)
    owner_id = _add_user(postgres_database, "immutable-origin@example.com")
    store.save_project(source_id, owner_id.hex, guest_token_digest="a" * 64)
    duplicate_id = uuid4().hex
    store.duplicate_project(
        source_project_id=source_id,
        project_id=duplicate_id,
        job_id=uuid4().hex,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )

    with pytest.raises(DBAPIError), postgres_database.session() as session:
        session.execute(
            update(ProjectOrigin)
            .where(ProjectOrigin.target_project_id == UUID(duplicate_id))
            .values(source_project_uuid_snapshot=uuid4())
        )


def test_workspace_is_assembled_from_current_revision_and_revision_restore(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, _ = _complete_project(store, guest_digest="a" * 64)
    owner_id = _add_user(postgres_database, "workspace-owner@example.com")
    store.save_project(project_id, owner_id.hex, guest_token_digest="a" * 64)
    original_workspace = store.get_workspace(project_id)
    original_title = original_workspace["chart"]["title"]
    store.save_chart(project_id, _chart("Second revision"))
    assert store.get_workspace(project_id)["chart"]["title"] == "Second revision"

    store.restore_project_revision(project_id, 1, owner_id.hex)
    restored = store.get_workspace(project_id)
    assert restored["chart"]["title"] == original_title
    assert restored["preview"]["projectId"] == project_id
    assert restored["quality"]["projectId"] == project_id
    assert restored["shares"] == []
    with postgres_database.session() as session:
        project = session.get(Project, UUID(project_id))
        assert project is not None and project.current_revision is not None
        assert project.current_revision.revision_number == 1
        assert (
            session.scalar(
                select(func.count())
                .select_from(ProjectRevision)
                .where(ProjectRevision.project_id == project.id)
            )
            == 2
        )


def _postgres_settings(root: Path) -> Settings:
    return Settings(
        database_path=root / "reference.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
        environment="test",
        postgres_url=POSTGRES_URL,
        object_storage_root=root / "objects",
        persistence_backend="postgresql",
    )


def _sign_in(client: TestClient, sender: MemoryEmailSender) -> None:
    requested = client.post(
        "/api/v1/auth/email-code",
        json={"email": "phase4-api@example.com"},
    )
    assert requested.status_code == 200
    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={
            "challengeId": requested.json()["challengeId"],
            "code": sender.messages[-1][1],
        },
    )
    assert verified.status_code == 200


def test_postgresql_api_identity_history_duplicate_delete_restore_workspace_closure(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    settings = _postgres_settings(tmp_path)
    store = PostgresProjectStore(
        postgres_database,
        LocalObjectStorage(settings.object_storage_root),
        settings.project_ttl_seconds,
        guest_session_ttl_seconds=settings.session_ttl_seconds,
    )
    reference = SqliteReferenceRepository(settings.database_path, settings.project_ttl_seconds)
    sender = MemoryEmailSender()
    auth = AuthService(sender, settings.session_ttl_seconds, store)
    app = create_app(
        settings,
        repository=reference,
        project_store=store,
        auth_service=auth,
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/samples/thermal-response/projects")
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        _sign_in(client, sender)

        saved = client.post(f"/api/v1/projects/{project_id}/save")
        assert saved.status_code == 200
        assert saved.json()["projectId"] == project_id
        assert saved.json()["storageMode"] == "saved-cloud"
        history = client.get("/api/v1/projects")
        assert history.status_code == 200
        assert [item["id"] for item in history.json()["projects"]] == [project_id]

        first_duplicate = client.post(
            f"/api/v1/projects/{project_id}/duplicate",
            headers={"Idempotency-Key": "phase4-api-duplicate"},
        )
        second_duplicate = client.post(
            f"/api/v1/projects/{project_id}/duplicate",
            headers={"Idempotency-Key": "phase4-api-duplicate"},
        )
        assert first_duplicate.status_code == second_duplicate.status_code == 200
        duplicate_id = first_duplicate.json()["projectId"]
        assert second_duplicate.json()["projectId"] == duplicate_id

        workspace = client.get(f"/api/v1/projects/{duplicate_id}/workspace")
        assert workspace.status_code == 200
        assert set(workspace.json()) == {
            "apiVersion",
            "session",
            "preview",
            "quality",
            "decisions",
            "chart",
            "shares",
        }
        assert workspace.json()["session"]["projectId"] == duplicate_id
        assert workspace.json()["shares"] == []

        deleted = client.delete(f"/api/v1/projects/{duplicate_id}")
        assert deleted.status_code == 204
        recovery = client.get("/api/v1/recovery/projects")
        assert recovery.status_code == 200
        assert [item["id"] for item in recovery.json()["projects"]] == [duplicate_id]
        restored = client.post(f"/api/v1/projects/{duplicate_id}/restore")
        assert restored.status_code == 200
        assert restored.json()["projectId"] == duplicate_id
        assert client.get(f"/api/v1/projects/{duplicate_id}/workspace").status_code == 200

    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_restore_rejects_wrong_owner_and_expired_window(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, _ = _complete_project(store, guest_digest="a" * 64)
    owner_id = _add_user(postgres_database, "restore-owner@example.com")
    other_id = _add_user(postgres_database, "restore-other@example.com")
    store.save_project(project_id, owner_id.hex, guest_token_digest="a" * 64)
    store.delete_project(project_id, owner_user_id=owner_id.hex, guest_token_digest=None)
    with pytest.raises(PersistenceNotFound):
        store.restore_deleted_project(project_id, other_id.hex)

    expired_deleted_at = datetime.now(UTC) - timedelta(hours=25)
    with postgres_database.session() as session:
        project = session.get(Project, UUID(project_id))
        assert project is not None
        project.deleted_at = expired_deleted_at
        project.purge_after = expired_deleted_at + timedelta(hours=24)
    with pytest.raises(PersistenceNotFound, match="expired"):
        store.restore_deleted_project(project_id, owner_id.hex)
