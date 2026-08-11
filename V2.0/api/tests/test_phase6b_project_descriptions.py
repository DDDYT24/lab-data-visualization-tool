from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select, text

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.db.models import ExportJobRecord, Project, ProjectRevision, ShareLinkRecord, User
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.models import UpdateProjectDescriptionRequest
from labviz_api.persistence.exceptions import PersistenceNotFound, ProjectRevisionConflict
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.persistence.sqlite import SqliteProjectStore
from labviz_api.processing import build_preview, build_quality_report, default_chart_spec
from labviz_api.project_spec import ProjectSpecV1
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.storage import LocalObjectStorage

API_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL = os.environ.get(
    "LABVIZ_TEST_POSTGRES_URL",
    "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test",
)


def alembic_config() -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = POSTGRES_URL
    return config


@pytest.fixture(scope="module")
def postgres_database() -> Iterator[Database]:
    database = Database(POSTGRES_URL)
    if not database.health().ready:
        database.dispose()
        pytest.fail("PostgreSQL is required for Phase 6B description tests.")
    command.upgrade(alembic_config(), "head")
    try:
        yield database
    finally:
        database.dispose()


@pytest.fixture(autouse=True)
def empty_postgres(postgres_database: Database) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))


def settings(root: Path, backend: str) -> Settings:
    return Settings(
        database_path=root / f"{backend}.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
        environment="test",
        postgres_url=POSTGRES_URL,
        object_storage_root=root / f"{backend}-objects",
        persistence_backend=backend,
    )


def sign_in(client: TestClient, sender: MemoryEmailSender, email: str) -> None:
    requested = client.post("/api/v1/auth/email-code", json={"email": email})
    assert requested.status_code == 200
    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={
            "challengeId": requested.json()["challengeId"],
            "code": sender.messages[-1][1],
        },
    )
    assert verified.status_code == 200


def app_for_backend(
    backend: str,
    root: Path,
    database: Database,
) -> tuple[Any, MemoryEmailSender]:
    configured = settings(root, backend)
    reference = SqliteReferenceRepository(
        configured.database_path,
        configured.project_ttl_seconds,
    )
    store: PostgresProjectStore | SqliteProjectStore
    auth_repository: Any
    if backend == "postgresql":
        store = PostgresProjectStore(
            database,
            LocalObjectStorage(configured.object_storage_root),
            configured.project_ttl_seconds,
            guest_session_ttl_seconds=configured.session_ttl_seconds,
        )
        auth_repository = store
    else:
        store = SqliteProjectStore(reference)
        auth_repository = reference
    sender = MemoryEmailSender()
    auth = AuthService(sender, configured.session_ttl_seconds, auth_repository)
    return (
        create_app(
            configured,
            repository=reference,
            project_store=store,
            auth_service=auth,
        ),
        sender,
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgresql"])
def test_description_api_contract_authorization_and_pinned_share(
    backend: str,
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    app, sender = app_for_backend(backend, tmp_path, postgres_database)
    with TestClient(app) as owner:
        created = owner.post("/api/v1/samples/thermal-response/projects")
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        assert created.json()["description"] == ""
        ready = owner.get(f"/api/v1/projects/{project_id}")
        assert ready.status_code == 200
        assert ready.json()["currentRevisionId"] is not None

        unauthenticated = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": "not allowed",
                "expectedRevisionId": ready.json()["currentRevisionId"],
            },
        )
        assert unauthenticated.status_code == 401

        sign_in(owner, sender, f"owner-{backend}@example.com")
        saved = owner.post(f"/api/v1/projects/{project_id}/save")
        assert saved.status_code == 200
        assert saved.json()["description"] == ""
        initial_revision = saved.json()["currentRevisionId"]

        workspace = owner.get(f"/api/v1/projects/{project_id}/workspace")
        assert workspace.status_code == 200
        changed_chart = dict(workspace.json()["chart"])
        changed_chart["title"] = "Chart changed before description"
        chart_update = owner.put(
            f"/api/v1/projects/{project_id}/chart",
            json={"chart": changed_chart},
        )
        assert chart_update.status_code == 200

        first_description = "<b>Plain text only</b> — 第一版"
        first = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": first_description,
                "expectedRevisionId": initial_revision,
            },
        )
        assert first.status_code == 200
        assert first.json()["description"] == first_description
        assert first.json()["revisionId"] != initial_revision

        no_op = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": first_description,
                "expectedRevisionId": first.json()["revisionId"],
            },
        )
        assert no_op.status_code == 200
        assert no_op.json()["revisionId"] == first.json()["revisionId"]
        assert no_op.json()["revisionNumber"] == first.json()["revisionNumber"]

        share = owner.post(
            f"/api/v1/projects/{project_id}/shares",
            json={"downloadsEnabled": False},
        )
        assert share.status_code == 200
        token = share.json()["token"]

        second = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": "Second description",
                "expectedRevisionId": first.json()["revisionId"],
            },
        )
        assert second.status_code == 200

        stale = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": "stale overwrite",
                "expectedRevisionId": first.json()["revisionId"],
            },
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "project-revision-conflict"

        shared = owner.get(f"/api/v1/shares/{token}")
        assert shared.status_code == 200
        assert shared.json()["description"] == first_description

        duplicate = owner.post(f"/api/v1/projects/{project_id}/duplicate")
        assert duplicate.status_code == 200
        assert duplicate.json()["description"] == "Second description"
        duplicate_workspace = owner.get(
            f"/api/v1/projects/{duplicate.json()['projectId']}/workspace"
        )
        assert duplicate_workspace.status_code == 200
        assert duplicate_workspace.json()["session"]["description"] == "Second description"

        too_large = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": "界" * 1_334,
                "expectedRevisionId": second.json()["revisionId"],
            },
        )
        assert too_large.status_code == 422
        assert too_large.json()["code"] == "validation-error"

        other_app, other_sender = app_for_backend(backend, tmp_path, postgres_database)
        with TestClient(other_app) as other:
            sign_in(other, other_sender, f"other-{backend}@example.com")
            forbidden = other.patch(
                f"/api/v1/projects/{project_id}/description",
                json={
                    "description": "not mine",
                    "expectedRevisionId": second.json()["revisionId"],
                },
            )
            assert forbidden.status_code == 403
        deleted = owner.delete(f"/api/v1/projects/{project_id}")
        assert deleted.status_code == 204
        after_delete = owner.patch(
            f"/api/v1/projects/{project_id}/description",
            json={
                "description": "deleted",
                "expectedRevisionId": second.json()["revisionId"],
            },
        )
        assert after_delete.status_code == 404


def ready_saved_store(
    backend: str,
    root: Path,
    database: Database,
) -> tuple[PostgresProjectStore | SqliteProjectStore, str, str, str]:
    owner_id = uuid4().hex
    configured = settings(root, backend)
    if backend == "postgresql":
        store: PostgresProjectStore | SqliteProjectStore = PostgresProjectStore(
            database,
            LocalObjectStorage(configured.object_storage_root),
            configured.project_ttl_seconds,
        )
        with database.session() as session:
            session.add(User(id=UUID(owner_id), email=f"{owner_id}@example.test"))
    else:
        store = SqliteProjectStore(
            SqliteReferenceRepository(configured.database_path, configured.project_ttl_seconds)
        )
    project_id = uuid4().hex
    job_id = uuid4().hex
    payload = b"time,signal\n0,1\n1,2\n2,3\n"
    source = {
        "name": "description.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    frame = pd.DataFrame({"time": [0, 1, 2], "signal": [1.0, 2.0, 3.0]})
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="description",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest="a" * 64,
    )
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=build_preview(project_id, frame),
        quality=build_quality_report(project_id, frame),
        chart=default_chart_spec(frame),
    )
    store.update_job(job_id, stage="ready", progress=100, message="ready")
    store.save_project(project_id, owner_id, guest_token_digest="a" * 64)
    project = store.get_project(project_id, touch=False)
    assert project is not None and project["current_revision_id"] is not None
    return store, project_id, owner_id, str(project["current_revision_id"])


@pytest.mark.parametrize("backend", ["sqlite", "postgresql"])
def test_concurrent_description_edits_admit_exactly_one(
    backend: str,
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, owner_id, expected_revision_id = ready_saved_store(
        backend,
        tmp_path,
        postgres_database,
    )
    barrier = Barrier(2)

    def edit(label: str) -> str:
        barrier.wait(timeout=10)
        try:
            store.update_project_description(
                project_id=project_id,
                owner_user_id=owner_id,
                description=label,
                expected_revision_id=expected_revision_id,
            )
        except ProjectRevisionConflict:
            return "conflict"
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(edit, ("first", "second")))
    assert sorted(results) == ["conflict", "saved"]


def test_postgres_description_revision_preserves_lineage_restore_and_export(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store, project_id, owner_id, initial_revision_id = ready_saved_store(
        "postgresql",
        tmp_path,
        postgres_database,
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = json.loads(project["chart_json"])
    share = store.create_share(
        project_id=project_id,
        owner_user_id=owner_id,
        downloads_enabled=False,
    )
    export = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=initial_revision_id,
        chart=chart,
        payload=b"\x89PNG\r\n\x1a\npublication-bytes",
        owner_user_id=owner_id,
        guest_token_digest=None,
    )
    updated = store.update_project_description(
        project_id=project_id,
        owner_user_id=owner_id,
        description="Durable description",
        expected_revision_id=initial_revision_id,
    )

    with postgres_database.session() as session:
        revisions = list(
            session.scalars(
                select(ProjectRevision)
                .where(ProjectRevision.project_id == UUID(project_id))
                .order_by(ProjectRevision.revision_number)
            )
        )
        link = session.scalar(
            select(ShareLinkRecord).where(ShareLinkRecord.project_id == UUID(project_id))
        )
        export_job = session.get(ExportJobRecord, UUID(export["id"]))
        current_project = session.get(Project, UUID(project_id))
    assert len(revisions) == 2
    assert ProjectSpecV1.model_validate(revisions[0].spec_document).description == ""
    assert (
        ProjectSpecV1.model_validate(revisions[1].spec_document).description
        == "Durable description"
    )
    assert link is not None and link.project_revision_id == revisions[0].id
    assert export_job is not None and export_job.project_revision_id == revisions[0].id
    assert current_project is not None and current_project.current_revision_id == UUID(
        str(updated["revisionId"])
    )

    shared = store.get_shared_project(share["token"])
    assert shared is not None and shared[0]["description"] == ""
    store.restore_project_revision(project_id, revisions[0].revision_number, owner_id)
    restored = store.get_project(project_id, touch=False)
    assert restored is not None and restored["description"] == ""

    store.delete_project(project_id, owner_user_id=owner_id, guest_token_digest=None)
    with pytest.raises(PersistenceNotFound):
        store.update_project_description(
            project_id=project_id,
            owner_user_id=owner_id,
            description="blocked while deleted",
            expected_revision_id=initial_revision_id,
        )
    store.restore_deleted_project(project_id, owner_id)
    assert store.get_project(project_id, touch=False) is not None


def test_description_limit_is_utf8_bytes_and_rejects_nulls() -> None:
    accepted = UpdateProjectDescriptionRequest(
        description="界" * 1_333,
        expected_revision_id=uuid4(),
    )
    assert len(accepted.description.encode("utf-8")) == 3_999
    with pytest.raises(ValidationError, match="4000 UTF-8 bytes"):
        UpdateProjectDescriptionRequest(
            description="界" * 1_334,
            expected_revision_id=uuid4(),
        )
    with pytest.raises(ValidationError, match="null character"):
        UpdateProjectDescriptionRequest(
            description="invalid\x00text",
            expected_revision_id=uuid4(),
        )
