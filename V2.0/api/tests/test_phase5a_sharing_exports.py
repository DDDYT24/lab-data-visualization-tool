from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.db.models import (
    DatasetVersion,
    ExportJobRecord,
    IdempotencyRecord,
    ProcessingRun,
    Project,
    ProjectRevision,
    PublicationExport,
    ShareExportBinding,
    ShareLinkRecord,
    SourceFile,
    StoredObject,
    StoredObjectWriteIntent,
    User,
)
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.models import ChartSpec
from labviz_api.persistence.exceptions import (
    ObjectConfirmationPending,
    PersistenceConflict,
    PersistenceUnavailable,
)
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import (
    build_preview,
    build_quality_report,
    render_chart,
)
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.share_tokens import ShareTokenCodec
from labviz_api.storage import LocalObjectStorage, ObjectInfo, StagedObject

API_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL = os.environ.get(
    "LABVIZ_TEST_POSTGRES_URL",
    "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test",
)


def _alembic_config() -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = POSTGRES_URL
    return config


@pytest.fixture(scope="module")
def postgres_database() -> Iterator[Database]:
    database = Database(POSTGRES_URL)
    if not database.health().ready:
        database.dispose()
        pytest.skip("Local PostgreSQL is not running; start it with docker compose.")
    with database.engine.begin() as connection:
        if "projects" in inspect(connection).get_table_names():
            connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
    command.downgrade(_alembic_config(), "base")
    command.upgrade(_alembic_config(), "head")
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
            "signal": [1.0, 2.0, 3.5, 5.0],
            "note": ["a", "b", "c", "d"],
        }
    )


def _chart(format_name: str = "png", title: str = "Pinned figure") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": title,
        "xAxis": {"field": "time", "title": "Time", "unit": "s"},
        "yAxis": {"field": "signal", "title": "Signal", "unit": "V"},
        "series": [{"field": "signal", "label": "Signal", "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": format_name,
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }


def _complete_saved_project(
    store: PostgresProjectStore,
    database: Database,
    *,
    email: str,
) -> tuple[str, UUID, pd.DataFrame]:
    project_id, guest_digest, frame = _complete_guest_project(store)
    owner_id = uuid4()
    with database.session() as session:
        session.add(User(id=owner_id, email=email))
    store.save_project(project_id, owner_id.hex, guest_token_digest=guest_digest)
    return project_id, owner_id, frame


def _complete_guest_project(
    store: PostgresProjectStore,
    *,
    guest_digest: str | None = None,
) -> tuple[str, str, pd.DataFrame]:
    project_id = uuid4().hex
    job_id = uuid4().hex
    frame = _frame()
    payload = frame.to_csv(index=False).encode()
    source = {
        "name": "phase5.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    guest_digest = guest_digest or hashlib.sha256(uuid4().bytes).hexdigest()
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="Phase 5",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest=guest_digest,
    )
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=build_preview(project_id, frame),
        quality=build_quality_report(project_id, frame),
        chart=_chart(),
    )
    store.update_job(job_id, stage="ready", progress=100, message="Project is ready.")
    return project_id, guest_digest, frame


def _render(frame: pd.DataFrame, chart: dict[str, Any]) -> bytes:
    return render_chart(frame, ChartSpec.model_validate(chart))


def test_share_token_format_digest_and_key_rotation() -> None:
    codec = ShareTokenCodec.from_strings(((1, "a" * 32), (2, "b" * 32)), 2)
    public_id = uuid4()
    token = codec.issue(public_id)
    assert len(token) == 79
    assert codec.parse_public_id(token) == public_id
    digest = codec.digest(token)
    assert token not in digest
    assert codec.verify(
        token,
        public_id=public_id,
        key_version=2,
        stored_digest=digest,
    )
    tampered_suffix = "A" if token[-1] != "A" else "B"
    assert not codec.verify(
        f"{token[:-1]}{tampered_suffix}",
        public_id=public_id,
        key_version=2,
        stored_digest=digest,
    )
    old_token = codec.issue(public_id, 1)
    assert codec.verify(
        old_token,
        public_id=public_id,
        key_version=1,
        stored_digest=codec.digest(old_token),
    )


def test_export_job_scope_foreign_keys_reject_mismatched_publication_and_intent(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    first_project_id, first_owner_id, _frame_value = _complete_saved_project(
        store, postgres_database, email="scope-one@example.com"
    )
    second_project_id, _second_owner_id, _second_frame = _complete_saved_project(
        store, postgres_database, email="scope-two@example.com"
    )
    job_id = uuid4()
    with postgres_database.session() as session:
        first_project = session.get(Project, UUID(first_project_id))
        assert first_project is not None and first_project.current_revision_id is not None
        revision = session.get(ProjectRevision, first_project.current_revision_id)
        assert revision is not None
        version = session.get(DatasetVersion, revision.active_dataset_version_id)
        assert version is not None
        stored = session.get(StoredObject, version.stored_object_id)
        assert stored is not None
        run = session.scalar(
            select(ProcessingRun)
            .where(ProcessingRun.project_id == first_project.id)
            .order_by(ProcessingRun.created_at)
            .limit(1)
        )
        assert run is not None
        session.add(
            ExportJobRecord(
                id=job_id,
                project_id=first_project.id,
                project_revision_id=revision.id,
                requested_by_user_id=first_owner_id,
                current_processing_run_id=run.id,
                status="ready",
                format="png",
                request_sha256="1" * 64,
                message="ready",
                attempt_count=1,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        publication_values = {
            "id": job_id,
            "project_id": first_project.id,
            "project_revision_id": revision.id,
            "dataset_version_id": revision.active_dataset_version_id,
            "cleaning_decision_set_id": revision.cleaning_decision_set_id,
            "chart_spec_revision_id": revision.chart_spec_revision_id,
            "processing_run_id": run.id,
            "stored_object_id": stored.id,
            "format": "svg",
            "media_type": "image/svg+xml",
            "renderer_name": "test",
            "renderer_version": "1",
            "render_contract_version": "publication-export-v1",
            "render_spec_document": {},
            "size_preset": "single-column",
            "unit": "in",
            "dpi": 300,
            "output_sha256": stored.sha256,
            "output_size_bytes": stored.size_bytes,
            "validation_document": {},
            "created_at": datetime.now(UTC),
        }
        stored_object_id = stored.id

    with pytest.raises(IntegrityError), postgres_database.session() as session:
        session.add(PublicationExport(**publication_values))

    with pytest.raises(IntegrityError), postgres_database.session() as session:
        session.add(
            StoredObjectWriteIntent(
                project_id=UUID(second_project_id),
                export_job_id=job_id,
                stored_object_id=stored_object_id,
                operation="export",
                status="pending",
                created_at=datetime.now(UTC),
            )
        )


def test_guest_export_idempotency_uses_logical_request_and_allows_expired_key_reuse(
    tmp_path: Path,
    postgres_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresProjectStore(
        postgres_database,
        LocalObjectStorage(tmp_path / "objects"),
        604_800,
    )
    project_id, guest_digest, frame = _complete_guest_project(store)
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    payload = _render(frame, chart)
    first = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=payload,
        owner_user_id=None,
        guest_token_digest=guest_digest,
        idempotency_key="guest-logical-request",
    )
    replay = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=payload + b"different-render-bytes",
        owner_user_id=None,
        guest_token_digest=guest_digest,
        idempotency_key="guest-logical-request",
    )
    assert replay["id"] == first["id"]

    clock = datetime.now(UTC) + timedelta(hours=25)
    monkeypatch.setattr("labviz_api.persistence.postgres._now", lambda: clock)

    replacement = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=payload + b"new-render-after-expiry",
        owner_user_id=None,
        guest_token_digest=guest_digest,
        idempotency_key="guest-logical-request",
    )
    assert replacement["id"] != first["id"]

    # Create one already-expired immutable record, then let the worker decide with
    # PostgreSQL time. IdempotencyRecord itself cannot be updated by design.
    clock = datetime.now(UTC) - timedelta(hours=25)
    cleanup_job = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=payload,
        owner_user_id=None,
        guest_token_digest=guest_digest,
        idempotency_key="guest-expired-cleanup",
    )
    assert cleanup_job["status"] == "ready"
    store.cleanup_expired()
    with postgres_database.session() as session:
        assert (
            session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.idempotency_key == "guest-expired-cleanup"
                )
            )
            is None
        )


def test_identical_export_bytes_do_not_cross_deduplication_scopes(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    first_project_id, first_owner_id, frame = _complete_saved_project(
        store, postgres_database, email="dedup-one@example.com"
    )
    second_project_id, second_owner_id, _second_frame = _complete_saved_project(
        store, postgres_database, email="dedup-two@example.com"
    )
    chart = _chart()
    payload = _render(frame, chart)
    exports: list[dict[str, Any]] = []
    for project_id, owner_id in (
        (first_project_id, first_owner_id),
        (second_project_id, second_owner_id),
    ):
        project = store.get_project(project_id, touch=False)
        assert project is not None
        exports.append(
            store.create_publication_export(
                project_id=project_id,
                expected_revision_id=project["current_revision_id"],
                chart=chart,
                payload=payload,
                owner_user_id=owner_id.hex,
                guest_token_digest=None,
            )
        )
    with postgres_database.session() as session:
        first = session.get(PublicationExport, UUID(exports[0]["id"]))
        second = session.get(PublicationExport, UUID(exports[1]["id"]))
        assert first is not None and second is not None
        assert first.stored_object_id != second.stored_object_id


def test_revision_pinned_shares_fixed_bindings_and_exact_byte_reuse(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        store, postgres_database, email="phase5-owner@example.com"
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    pinned_revision_id = project["current_revision_id"]
    chart = _chart()
    payload = _render(frame, chart)
    first = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=pinned_revision_id,
        chart=chart,
        payload=payload,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
        idempotency_key="first-png",
    )
    replay = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=pinned_revision_id,
        chart=chart,
        payload=payload,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
        idempotency_key="first-png",
    )
    assert replay["id"] == first["id"]
    changed_request = _chart(title="Different idempotent request")
    with pytest.raises(PersistenceConflict, match="different request"):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=pinned_revision_id,
            chart=changed_request,
            payload=_render(frame, changed_request),
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
            idempotency_key="first-png",
        )
    share = store.create_share(
        project_id=project_id,
        owner_user_id=owner_id.hex,
        downloads_enabled=True,
    )
    share_id = store.share_tokens.parse_public_id(share["token"])
    assert share_id is not None

    class CleanupReportingFailure(LocalObjectStorage):
        def discard(self, staged: StagedObject) -> None:
            super().discard(staged)
            raise OSError("simulated post-commit staging cleanup failure")

    cleanup_store = PostgresProjectStore(
        postgres_database,
        CleanupReportingFailure(storage.root),
        7_200,
    )
    second = cleanup_store.create_publication_export(
        project_id=project_id,
        expected_revision_id=pinned_revision_id,
        chart=chart,
        payload=payload,
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )
    assert second["id"] != first["id"]
    for format_name in ("svg", "pdf"):
        format_chart = deepcopy(chart)
        format_chart["export"]["format"] = format_name
        result = store.create_publication_export(
            project_id=project_id,
            expected_revision_id=pinned_revision_id,
            chart=format_chart,
            payload=_render(frame, format_chart),
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )
        assert result["project_revision_id"] == pinned_revision_id

    with postgres_database.session() as session:
        publications = list(
            session.scalars(
                select(PublicationExport)
                .where(PublicationExport.project_id == UUID(project_id))
                .order_by(PublicationExport.created_at, PublicationExport.id)
            )
        )
        png_exports = [item for item in publications if item.format == "png"]
        assert len(png_exports) == 2
        assert png_exports[0].stored_object_id == png_exports[1].stored_object_id
        binding = session.get(ShareExportBinding, (share_id, "png"))
        assert binding is not None
        assert binding.publication_export_id == UUID(first["id"])
        assert (
            session.scalar(
                select(func.count())
                .select_from(ShareExportBinding)
                .where(ShareExportBinding.share_link_id == share_id)
            )
            == 3
        )
        share_record = session.get(ShareLinkRecord, share_id)
        assert share_record is not None
        assert share_record.token_digest != share["token"]
        assert len(share_record.token_digest) == 64

    with pytest.raises(DBAPIError, match="immutable"), postgres_database.session() as session:
        session.execute(
            update(PublicationExport)
            .where(PublicationExport.id == UUID(first["id"]))
            .values(renderer_version="mutated")
        )
    with pytest.raises(DBAPIError, match="immutable"), postgres_database.session() as session:
        session.execute(
            update(ShareExportBinding)
            .where(
                ShareExportBinding.share_link_id == share_id,
                ShareExportBinding.format == "png",
            )
            .values(publication_export_id=UUID(second["id"]))
        )

    shared = store.get_shared_project(share["token"])
    assert shared is not None
    assert shared[0]["title"] == "Pinned figure"
    assert shared[0]["download_formats"] == ["pdf", "png", "svg"]
    assert store.get_shared_export(share["token"], "png")["export"]["id"] == first["id"]
    tampered_suffix = "A" if share["token"][-1] != "A" else "B"
    assert store.get_shared_project(f"{share['token'][:-1]}{tampered_suffix}") is None

    changed_chart = _chart(title="Later working revision")
    current = store.get_project(project_id, touch=False)
    assert current is not None
    changed = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=current["current_revision_id"],
        chart=changed_chart,
        payload=_render(frame, changed_chart),
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )
    assert changed["project_revision_id"] != pinned_revision_id
    assert store.get_shared_project(share["token"])[0]["title"] == "Pinned figure"
    stale_chart = _chart(title="Stale concurrent export")
    with pytest.raises(PersistenceConflict, match="Project changed"):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=pinned_revision_id,
            chart=stale_chart,
            payload=_render(frame, stale_chart),
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )
    staging_root = storage.root / ".staging"
    assert not staging_root.exists() or not any(staging_root.iterdir())

    assert store.delete_project(project_id, owner_user_id=owner_id.hex, guest_token_digest=None)
    assert store.get_shared_project(share["token"]) is None
    store.restore_deleted_project(project_id, owner_id.hex)
    assert store.get_shared_project(share["token"]) is not None
    assert store.revoke_share(
        token=share["token"], project_id=project_id, owner_user_id=owner_id.hex
    )
    assert store.revoke_share(
        token=share["token"], project_id=project_id, owner_user_id=owner_id.hex
    )
    assert store.get_shared_project(share["token"]) is None
    with postgres_database.session() as session:
        publication = session.get(PublicationExport, UUID(first["id"]))
        assert publication is not None
        export_object_id = publication.stored_object_id
    assert store.delete_project(project_id, owner_user_id=owner_id.hex, guest_token_digest=None)
    deleted_at = datetime.now(UTC) - timedelta(hours=25)
    with postgres_database.session() as session:
        project_row = session.get(Project, UUID(project_id))
        assert project_row is not None
        project_row.deleted_at = deleted_at
        project_row.purge_after = deleted_at + timedelta(hours=24)
    store.cleanup_expired()
    with postgres_database.session() as session:
        assert session.get(Project, UUID(project_id)) is None
        stored = session.get(StoredObject, export_object_id)
        assert stored is not None and stored.status == "deleted"


def test_concurrent_exports_fill_one_immutable_share_binding(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, LocalObjectStorage(tmp_path / "objects"), 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        store, postgres_database, email="binding-race@example.com"
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    share = store.create_share(
        project_id=project_id,
        owner_user_id=owner_id.hex,
        downloads_enabled=True,
    )
    share_id = store.share_tokens.parse_public_id(share["token"])
    assert share_id is not None
    chart = _chart()
    payload = _render(frame, chart)

    def create(payload_value: bytes) -> dict[str, Any]:
        return store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=payload_value,
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, (payload + b"race-a", payload + b"race-b")))

    with postgres_database.session() as session:
        bindings = list(
            session.scalars(
                select(ShareExportBinding).where(
                    ShareExportBinding.share_link_id == share_id,
                    ShareExportBinding.format == "png",
                )
            )
        )
        assert len(bindings) == 1
        assert bindings[0].publication_export_id in {UUID(item["id"]) for item in results}


class FailNextConfirmStorage(LocalObjectStorage):
    fail_next = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated object confirmation outage")
        return super().confirm(staged)


def test_temporary_project_immediate_delete_marks_complete_object_closure_for_gc(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailNextConfirmStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id, guest_digest, frame = _complete_guest_project(store)
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    payload = _render(frame, chart)
    published = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=payload,
        owner_user_id=None,
        guest_token_digest=guest_digest,
    )

    source_payload = b"original source bytes"
    source_key = f"sources/{project_id}/source.csv"
    source_info = storage.put(source_key, io.BytesIO(source_payload))
    source_object_id = uuid4()
    with postgres_database.session() as session:
        source = session.scalar(
            select(SourceFile).where(SourceFile.project_id == UUID(project_id)).limit(1)
        )
        assert source is not None
        session.add(
            StoredObject(
                id=source_object_id,
                storage_backend="local",
                object_key=source_key,
                purpose="source-upload",
                status="available",
                media_type="text/csv",
                size_bytes=source_info.size_bytes,
                sha256=source_info.sha256,
                dedup_scope=f"guest:{project_id}",
                format_contract_version="source-v1",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        source.stored_object_id = source_object_id

    storage.fail_next = True
    with pytest.raises(ObjectConfirmationPending):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=payload + b"pending-render",
            owner_user_id=None,
            guest_token_digest=guest_digest,
        )

    with postgres_database.session() as session:
        project_row = session.get(Project, UUID(project_id))
        assert project_row is not None and project_row.current_revision_id is not None
        revision = session.get(ProjectRevision, project_row.current_revision_id)
        assert revision is not None
        version = session.get(DatasetVersion, revision.active_dataset_version_id)
        publication = session.get(PublicationExport, UUID(published["id"]))
        intent = session.scalar(
            select(StoredObjectWriteIntent).where(
                StoredObjectWriteIntent.project_id == project_row.id,
                StoredObjectWriteIntent.status == "pending",
            )
        )
        pending_job = session.scalar(
            select(ExportJobRecord).where(
                ExportJobRecord.project_id == project_row.id,
                ExportJobRecord.pending_stored_object_id.is_not(None),
            )
        )
        assert version is not None and publication is not None
        assert intent is not None and pending_job is not None
        assert pending_job.pending_stored_object_id == intent.stored_object_id
        object_ids = {
            source_object_id,
            version.stored_object_id,
            publication.stored_object_id,
            intent.stored_object_id,
        }

    assert store.delete_project(
        project_id,
        owner_user_id=uuid4().hex,
        guest_token_digest=guest_digest,
    )
    with postgres_database.session() as session:
        assert session.get(Project, UUID(project_id)) is None
        objects = list(session.scalars(select(StoredObject).where(StoredObject.id.in_(object_ids))))
        assert {item.id for item in objects} == object_ids
        assert all(item.gc_candidate_at is not None for item in objects)


def test_write_intent_recovers_export_across_store_restart(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailNextConfirmStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        store, postgres_database, email="recovery-owner@example.com"
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    payload = _render(frame, chart)
    storage.fail_next = True
    with pytest.raises(ObjectConfirmationPending):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=payload,
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )

    with postgres_database.session() as session:
        job = session.scalar(select(ExportJobRecord))
        intent = session.scalar(select(StoredObjectWriteIntent))
        assert job is not None and job.status == "rendering"
        assert job.pending_stored_object_id is not None
        assert intent is not None and intent.status == "pending"
        job_id = job.id

    restarted = PostgresProjectStore(postgres_database, storage, 7_200)
    assert restarted.recover_pending_objects() == 1
    reopened = restarted.get_export(job_id.hex)
    assert reopened is not None
    assert reopened["payload"] == payload
    with postgres_database.session() as session:
        job = session.get(ExportJobRecord, job_id)
        intent = session.scalar(
            select(StoredObjectWriteIntent).where(StoredObjectWriteIntent.export_job_id == job_id)
        )
        assert job is not None and job.status == "ready"
        assert job.pending_stored_object_id is None
        assert intent is not None and intent.status == "completed"
        assert session.get(PublicationExport, job_id) is not None


class FailNextFinalizeStore(PostgresProjectStore):
    fail_next_finalize = True

    def _finalize_export_object(self, staged: StagedObject) -> dict[str, Any]:
        if self.fail_next_finalize:
            self.fail_next_finalize = False
            raise PersistenceUnavailable("simulated final database transaction outage")
        return super()._finalize_export_object(staged)


def test_confirmed_object_recovers_after_final_database_transaction_failure(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    failing = FailNextFinalizeStore(postgres_database, storage, 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        failing, postgres_database, email="finalize-recovery@example.com"
    )
    project = failing.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    payload = _render(frame, chart)
    with pytest.raises(ObjectConfirmationPending):
        failing.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=payload,
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )

    with postgres_database.session() as session:
        job = session.scalar(
            select(ExportJobRecord).where(ExportJobRecord.project_id == UUID(project_id))
        )
        assert job is not None and job.pending_stored_object_id is not None
        stored = session.get(StoredObject, job.pending_stored_object_id)
        assert stored is not None and stored.status == "pending"
        assert storage.exists(stored.object_key)
        job_id = job.id

    restarted = PostgresProjectStore(postgres_database, storage, 7_200)
    assert restarted.recover_pending_objects() == 1
    reopened = restarted.get_export(job_id.hex)
    assert reopened is not None and reopened["payload"] == payload


@pytest.mark.parametrize("dangerous_state", ["publication", "pending-write"])
def test_migration_0005_downgrade_fails_closed_with_tracked_artifacts(
    dangerous_state: str,
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailNextConfirmStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        store, postgres_database, email=f"downgrade-{dangerous_state}@example.com"
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    if dangerous_state == "pending-write":
        storage.fail_next = True
        with pytest.raises(ObjectConfirmationPending):
            store.create_publication_export(
                project_id=project_id,
                expected_revision_id=project["current_revision_id"],
                chart=chart,
                payload=_render(frame, chart),
                owner_user_id=owner_id.hex,
                guest_token_digest=None,
            )
    else:
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=_render(frame, chart),
            owner_user_id=owner_id.hex,
            guest_token_digest=None,
        )

    with pytest.raises(DBAPIError, match="Phase 5A downgrade refused"):
        command.downgrade(_alembic_config(), "0004_identity_project_lifecycle")
    with postgres_database.engine.connect() as connection:
        # Alembic runs the requested multi-revision downgrade transactionally;
        # 0005's fail-closed guard therefore restores the 0006 starting head too.
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0007_phase5b2_orphan_staging"
        )


def test_empty_database_migrates_0005_to_0004_and_back(
    postgres_database: Database,
) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
    config = _alembic_config()
    command.downgrade(config, "0004_identity_project_lifecycle")
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0004_identity_project_lifecycle"
        )
    command.upgrade(config, "head")
    command.check(config)
    with postgres_database.engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0007_phase5b2_orphan_staging"
        )


def _settings(root: Path) -> Settings:
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
    requested = client.post("/api/v1/auth/email-code", json={"email": "phase5-api@example.com"})
    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={
            "challengeId": requested.json()["challengeId"],
            "code": sender.messages[-1][1],
        },
    )
    assert verified.status_code == 200


class CountingOpenStorage(LocalObjectStorage):
    open_calls = 0

    def open(self, key: str) -> BinaryIO:
        self.open_calls += 1
        return super().open(key)


def test_unauthorized_private_download_does_not_open_object_storage(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    settings = _settings(tmp_path)
    storage = CountingOpenStorage(settings.object_storage_root)
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id, owner_id, frame = _complete_saved_project(
        store, postgres_database, email="private-download@example.com"
    )
    project = store.get_project(project_id, touch=False)
    assert project is not None
    chart = _chart()
    exported = store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=chart,
        payload=_render(frame, chart),
        owner_user_id=owner_id.hex,
        guest_token_digest=None,
    )
    reference = SqliteReferenceRepository(settings.database_path, settings.project_ttl_seconds)
    app = create_app(
        settings,
        repository=reference,
        project_store=store,
        auth_service=AuthService(MemoryEmailSender(), settings.session_ttl_seconds, store),
    )
    with TestClient(app) as client:
        storage.open_calls = 0
        response = client.get(f"/api/v1/exports/{exported['id']}/download")
        assert response.status_code == 401
        assert storage.open_calls == 0


def test_postgresql_api_png_svg_pdf_fixed_download_permissions_and_recovery(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    settings = _settings(tmp_path)
    storage = LocalObjectStorage(settings.object_storage_root)
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    reference = SqliteReferenceRepository(settings.database_path, settings.project_ttl_seconds)
    sender = MemoryEmailSender()
    app = create_app(
        settings,
        repository=reference,
        project_store=store,
        auth_service=AuthService(sender, settings.session_ttl_seconds, store),
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/samples/thermal-response/projects")
        project_id = created.json()["projectId"]
        _sign_in(client, sender)
        assert client.post(f"/api/v1/projects/{project_id}/save").status_code == 200
        workspace = client.get(f"/api/v1/projects/{project_id}/workspace").json()
        chart = workspace["chart"]
        shared = client.post(
            f"/api/v1/projects/{project_id}/shares",
            json={"downloadsEnabled": True},
        )
        assert shared.status_code == 200
        token = shared.json()["token"]
        assert client.get(f"/api/v1/shares/{token}").json()["downloads"] == {
            "png": None,
            "svg": None,
            "pdf": None,
        }

        export_ids: dict[str, str] = {}
        for format_name, signature in {
            "png": b"\x89PNG\r\n\x1a\n",
            "svg": b"<?xml",
            "pdf": b"%PDF-",
        }.items():
            requested_chart = deepcopy(chart)
            requested_chart["export"]["format"] = format_name
            exported = client.post(
                f"/api/v1/projects/{project_id}/exports",
                json={"chart": requested_chart},
                headers={"Idempotency-Key": f"phase5-{format_name}"},
            )
            assert exported.status_code == 200
            assert exported.json()["status"] == "ready"
            export_ids[format_name] = exported.json()["id"]
            downloaded = client.get(exported.json()["downloadUrl"])
            assert downloaded.status_code == 200
            assert downloaded.content.startswith(signature)

        public = client.get(f"/api/v1/shares/{token}")
        assert public.status_code == 200
        assert all(public.json()["downloads"].values())
        public_png = client.get(f"/api/v1/shares/{token}/downloads/png")
        assert public_png.status_code == 200
        assert public_png.headers["cache-control"] == "no-store"

        share_id = store.share_tokens.parse_public_id(token)
        assert share_id is not None
        with postgres_database.session() as session:
            first_binding = session.get(ShareExportBinding, (share_id, "png"))
            assert first_binding is not None
            first_bound_export = first_binding.publication_export_id
        duplicate_png = client.post(
            f"/api/v1/projects/{project_id}/exports",
            json={"chart": chart},
        )
        assert duplicate_png.status_code == 200
        with postgres_database.session() as session:
            binding = session.get(ShareExportBinding, (share_id, "png"))
            assert binding is not None
            assert binding.publication_export_id == first_bound_export

        disabled = client.patch(
            f"/api/v1/projects/{project_id}/shares/{token}",
            json={"downloadsEnabled": False},
        )
        assert disabled.status_code == 200
        assert client.get(f"/api/v1/shares/{token}/downloads/png").status_code == 403
        assert client.delete(f"/api/v1/projects/{project_id}").status_code == 204
        assert client.get(f"/api/v1/shares/{token}").status_code == 404
        assert client.post(f"/api/v1/projects/{project_id}/restore").status_code == 200
        assert client.get(f"/api/v1/shares/{token}").status_code == 200
        assert client.delete(f"/api/v1/projects/{project_id}/shares/{token}").status_code == 204
        assert client.delete(f"/api/v1/projects/{project_id}/shares/{token}").status_code == 204
        assert client.get(f"/api/v1/shares/{token}").status_code == 404

    restarted_database = Database(POSTGRES_URL)
    try:
        restarted = PostgresProjectStore(restarted_database, storage, 7_200)
        assert restarted.get_export(export_ids["png"]) is not None
    finally:
        restarted_database.dispose()
