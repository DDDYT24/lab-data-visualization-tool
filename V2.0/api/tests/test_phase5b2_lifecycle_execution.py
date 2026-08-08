from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text

from labviz_api.config import Settings
from labviz_api.db.models import (
    AuthChallenge,
    AuthRequest,
    DatasetVersion,
    ExportJobRecord,
    GuestSession,
    IdempotencyRecord,
    OrphanStagingCandidate,
    Project,
    PublicationExport,
    SourceFile,
    StoredObject,
    StoredObjectWriteIntent,
    User,
)
from labviz_api.db.session import Database
from labviz_api.models import ChartSpec
from labviz_api.persistence.exceptions import ObjectConfirmationPending
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import build_preview, build_quality_report, render_chart
from labviz_api.storage import LocalObjectStorage, StagedObject
from labviz_api.storage.base import ObjectInfo
from labviz_api.workers.garbage_collection import StoredObjectGarbageCollector
from labviz_api.workers.leases import (
    METADATA_CLEANUP,
    ORPHAN_STAGING_INVENTORY,
    LeaseStore,
    RetryPolicy,
)
from labviz_api.workers.lifecycle import ProjectLifecycleHandler
from labviz_api.workers.metadata_cleanup import MetadataCleanupHandler
from labviz_api.workers.orphan_staging import OrphanStagingHandler
from labviz_api.workers.reconciliation import PendingObjectReconciler
from labviz_api.workers.runner import RunnerConfig, WorkerRunner
from labviz_api.workers.safety import MaintenanceSafety

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
        if "storage_inventory_checkpoints" in inspect(connection).get_table_names():
            connection.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
        if "orphan_staging_candidates" in inspect(connection).get_table_names():
            connection.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
    command.downgrade(_alembic_config(), "base")
    command.upgrade(_alembic_config(), "head")
    try:
        yield database
    finally:
        with database.engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
            connection.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
        command.upgrade(_alembic_config(), "head")
        database.dispose()


@pytest.fixture(autouse=True)
def empty_postgres(postgres_database: Database) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
        connection.execute(
            text("TRUNCATE TABLE auth_challenges, auth_requests, guest_sessions CASCADE")
        )
        connection.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
        connection.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
        connection.execute(
            text(
                "UPDATE worker_leases SET lease_owner = NULL, lease_until = NULL, "
                "heartbeat_at = NULL, updated_at = clock_timestamp()"
            )
        )


def test_local_storage_heads_and_lists_only_staging_objects(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    staged = storage.stage("datasets/final.parquet", io.BytesIO(b"parquet"))
    storage.put("exports/final.png", io.BytesIO(b"png"))

    info = storage.head(staged.staging_key)
    assert info is not None
    assert info.key == staged.staging_key
    assert info.size_bytes == len(b"parquet")
    assert info.sha256 == hashlib.sha256(b"parquet").hexdigest()
    assert [item.key for item in storage.list_staged().items] == [staged.staging_key]
    assert storage.head("missing") is None


def test_0007_upgrade_downgrade_reupgrade_and_schema_drift(
    postgres_database: Database,
) -> None:
    config = _alembic_config()
    inspector = inspect(postgres_database.engine)
    assert "orphan_staging_candidates" in inspector.get_table_names()

    command.downgrade(config, "0006_worker_leases")
    assert "orphan_staging_candidates" not in inspect(postgres_database.engine).get_table_names()

    command.upgrade(config, "head")
    assert "orphan_staging_candidates" in inspect(postgres_database.engine).get_table_names()
    command.check(config)


def _stored_object(
    database: Database,
    *,
    status: str = "available",
    gc_candidate: bool = True,
    created_at: datetime | None = None,
    key: str | None = None,
    staging_key: str | None = None,
) -> UUID:
    object_id = uuid4()
    with database.session() as session:
        session.add(
            StoredObject(
                id=object_id,
                storage_backend="local",
                object_key=key or f"exports/{object_id}.png",
                staging_key=staging_key,
                purpose="export",
                status=status,
                media_type="image/png",
                size_bytes=3,
                sha256=hashlib.sha256(b"png").hexdigest(),
                dedup_scope="owner:test",
                format_contract_version="publication-export-v1",
                gc_candidate_at=datetime.now(UTC) if gc_candidate else None,
                created_at=created_at or datetime.now(UTC),
            )
        )
    return object_id


def _guest_project(
    database: Database,
    *,
    expired: bool = True,
    saved_deleted: bool = False,
) -> tuple[UUID, UUID]:
    guest_id = uuid4()
    project_id = uuid4()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            GuestSession(
                id=guest_id,
                token_digest=hashlib.sha256(guest_id.bytes).hexdigest(),
                expires_at=now - timedelta(hours=1) if expired else now + timedelta(hours=1),
            )
        )
        if saved_deleted:
            user = User(id=uuid4(), email=f"{uuid4().hex}@example.test")
            session.add(user)
            session.add(
                Project(
                    id=project_id,
                    storage_mode="saved-cloud",
                    title="Saved purge candidate",
                    owner_user_id=user.id,
                    expires_at=None,
                    deleted_at=now - timedelta(hours=25),
                    purge_after=now - timedelta(hours=1),
                )
            )
        else:
            session.add(
                Project(
                    id=project_id,
                    storage_mode="temporary-cloud",
                    title="Temporary purge candidate",
                    guest_session_id=guest_id,
                    expires_at=now - timedelta(seconds=1) if expired else now + timedelta(hours=1),
                )
            )
    return guest_id, project_id


def _chart(format_name: str = "png") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": "Worker recovery",
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


class ConfirmThenFailOnceStorage(LocalObjectStorage):
    fail_after_confirm = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        info = super().confirm(staged)
        if self.fail_after_confirm:
            self.fail_after_confirm = False
            raise OSError("process stopped after confirm")
        return info


class FailBeforeConfirmOnceStorage(LocalObjectStorage):
    fail_before_confirm = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        if self.fail_before_confirm:
            self.fail_before_confirm = False
            raise OSError("confirmation unavailable before publish")
        return super().confirm(staged)


def _pending_dataset(
    database: Database,
    storage: FailBeforeConfirmOnceStorage | ConfirmThenFailOnceStorage,
) -> tuple[PostgresProjectStore, str, UUID, dict[str, Any], pd.DataFrame]:
    store = PostgresProjectStore(database, storage, 7_200)
    project_id = uuid4().hex
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    frame = pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})
    payload = frame.to_csv(index=False).encode()
    source = {
        "name": "pending-dataset.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=uuid4().hex,
        title="Pending dataset",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest=guest_digest,
    )
    if isinstance(storage, FailBeforeConfirmOnceStorage):
        storage.fail_before_confirm = True
    else:
        storage.fail_after_confirm = True
    with pytest.raises(ObjectConfirmationPending):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=_chart(),
        )
    with database.session() as session:
        object_id = session.scalar(
            select(DatasetVersion.stored_object_id).where(
                DatasetVersion.project_id == UUID(project_id)
            )
        )
        assert object_id is not None
    return store, project_id, object_id, source, frame


def _pending_export(
    database: Database,
    storage: ConfirmThenFailOnceStorage,
) -> tuple[PostgresProjectStore, UUID, str]:
    store = PostgresProjectStore(database, storage, 7_200)
    project_id = uuid4().hex
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    frame = pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})
    csv_payload = frame.to_csv(index=False).encode()
    source = {
        "name": "pending.csv",
        "size": len(csv_payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=uuid4().hex,
        title="Pending recovery",
        source=source,
        source_sha256=hashlib.sha256(csv_payload).hexdigest(),
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
    project = store.get_project(project_id, touch=False)
    assert project is not None
    storage.fail_after_confirm = True
    with pytest.raises(ObjectConfirmationPending):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=_chart(),
            payload=render_chart(frame, ChartSpec.model_validate(_chart())),
            owner_user_id=None,
            guest_token_digest=guest_digest,
        )
    with database.session() as session:
        intent_id = session.scalar(
            select(StoredObjectWriteIntent.id).where(StoredObjectWriteIntent.status == "pending")
        )
        assert intent_id is not None
    return store, intent_id, project_id


def test_default_worker_safety_is_non_destructive(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "test.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
    )
    assert settings.worker_dry_run is True
    assert settings.worker_delete_enabled is False
    assert MaintenanceSafety().may_delete is False


def test_reconciliation_recovers_confirmed_bytes_after_process_restart(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    store, intent_id, _project_id = _pending_export(postgres_database, storage)
    leases = LeaseStore(postgres_database)
    claim = leases.claim_write_intents("reconcile-after-restart", batch_size=1, lease_seconds=60)[0]
    assert claim.item_id == intent_id

    # A new service instance represents recovery after the original API process exited.
    restarted_store = PostgresProjectStore(
        postgres_database,
        LocalObjectStorage(tmp_path / "objects"),
        7_200,
    )
    PendingObjectReconciler(restarted_store)(claim, leases)

    with postgres_database.session() as session:
        intent = session.get(StoredObjectWriteIntent, intent_id)
        publication = session.get(PublicationExport, intent.export_job_id if intent else uuid4())
        assert intent is not None and intent.status == "completed"
        assert intent.lease_owner is None
        assert publication is not None
        stored = session.get(StoredObject, publication.stored_object_id)
        assert stored is not None and stored.status == "available"
    store.dispose()


def test_reconciliation_confirms_staged_dataset_and_finishes_processing_run(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailBeforeConfirmOnceStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id = uuid4().hex
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    frame = pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})
    payload = frame.to_csv(index=False).encode()
    source = {
        "name": "pending-dataset.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=uuid4().hex,
        title="Pending dataset",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest=guest_digest,
    )
    storage.fail_before_confirm = True
    with pytest.raises(ObjectConfirmationPending):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=_chart(),
        )
    leases = LeaseStore(postgres_database)
    claims = leases.claim_pending_objects("dataset-reconcile", batch_size=1, lease_seconds=60)
    assert len(claims) == 1 and claims[0].expected_state == "pending"
    PendingObjectReconciler(store)(claims[0], leases)
    with postgres_database.session() as session:
        stored = session.get(StoredObject, claims[0].item_id)
        version = session.scalar(
            select(DatasetVersion).where(DatasetVersion.stored_object_id == claims[0].item_id)
        )
        assert stored is not None and stored.status == "available"
        assert version is not None and version.output_of_run is not None
        assert version.output_of_run.status == "succeeded"


def test_dataset_request_and_compat_recovery_cannot_bypass_a_worker_lease(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailBeforeConfirmOnceStorage(tmp_path / "objects")
    store, project_id, object_id, source, frame = _pending_dataset(postgres_database, storage)
    leases = LeaseStore(postgres_database)
    claimed = leases.claim_pending_object("leased-worker", object_id=object_id, lease_seconds=60)
    assert claimed is not None

    assert store.recover_pending_objects() == 0
    with pytest.raises(ObjectConfirmationPending):
        store.complete_project(
            project_id=project_id,
            source=source,
            frame=frame,
            preview=build_preview(project_id, frame),
            quality=build_quality_report(project_id, frame),
            chart=_chart(),
        )

    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        version = session.scalar(
            select(DatasetVersion).where(DatasetVersion.stored_object_id == object_id)
        )
        assert stored is not None
        assert (stored.status, stored.lease_owner, stored.fencing_token) == (
            "pending",
            "leased-worker",
            claimed.fencing_token,
        )
        assert version is not None
    assert leases.claim_stored_objects("gc-worker", batch_size=1, lease_seconds=60) == []


def test_dataset_expired_lease_takeover_fences_old_worker_and_clears_lease(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = FailBeforeConfirmOnceStorage(tmp_path / "objects")
    store, _project_id, object_id, _source, _frame = _pending_dataset(postgres_database, storage)
    leases = LeaseStore(postgres_database)
    old = leases.claim_pending_object("old-worker", object_id=object_id, lease_seconds=60)
    assert old is not None
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_objects SET lease_until = clock_timestamp() - INTERVAL '1 second' "
                "WHERE id = :id"
            ),
            {"id": object_id},
        )
    replacement = leases.claim_pending_object(
        "replacement-worker", object_id=object_id, lease_seconds=60
    )
    assert replacement is not None
    assert replacement.fencing_token == old.fencing_token + 1

    PendingObjectReconciler(store)(old, leases)
    assert leases.heartbeat_item(old, 60) is None
    assert not leases.release_item(old)
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None
        assert (stored.status, stored.lease_owner) == ("pending", "replacement-worker")

    PendingObjectReconciler(store)(replacement, leases)
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None
        assert (stored.status, stored.lease_owner, stored.lease_until) == (
            "available",
            None,
            None,
        )
    assert store.recover_pending_objects() == 0


def test_dataset_recovery_is_idempotent_after_confirm_before_sql_finalize_crash(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    _store, project_id, object_id, _source, _frame = _pending_dataset(postgres_database, storage)
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None
        assert stored.status == "pending"
        assert stored.lease_owner is None
        assert stored.fencing_token == 1

    restarted = PostgresProjectStore(
        postgres_database,
        LocalObjectStorage(tmp_path / "objects"),
        7_200,
    )
    assert restarted.recover_pending_objects() == 1
    assert restarted.recover_pending_objects() == 0
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        version = session.scalar(
            select(DatasetVersion).where(DatasetVersion.stored_object_id == object_id)
        )
        assert stored is not None
        assert (stored.status, stored.lease_owner, stored.lease_until) == (
            "available",
            None,
            None,
        )
        assert version is not None and version.project_id == UUID(project_id)


def test_reconciliation_retry_handles_crash_after_external_confirm(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    store, intent_id, _project_id = _pending_export(postgres_database, storage)
    leases = LeaseStore(postgres_database)
    claim = leases.claim_write_intents("first-worker", batch_size=1, lease_seconds=60)[0]

    storage.fail_after_confirm = True
    runner = WorkerRunner(
        task="pending-reconciliation",
        owner="first-worker-runner",
        leases=leases,
        config=RunnerConfig(batch_size=1, retry_policy=RetryPolicy(base_seconds=1)),
        item_handler=PendingObjectReconciler(store),
    )
    # Release the manually claimed row so the runner can own the persisted item.
    assert leases.release_item(claim)
    assert runner.run_once() == 1
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_object_write_intents SET next_attempt_at = "
                "clock_timestamp() - INTERVAL '1 second' WHERE id = :id"
            ),
            {"id": intent_id},
        )
    restarted = WorkerRunner(
        task="pending-reconciliation",
        owner="replacement-worker",
        leases=LeaseStore(postgres_database),
        config=RunnerConfig(batch_size=1),
        item_handler=PendingObjectReconciler(
            PostgresProjectStore(
                postgres_database,
                LocalObjectStorage(tmp_path / "objects"),
                7_200,
            )
        ),
    )
    assert restarted.run_once() == 1
    with postgres_database.session() as session:
        intent = session.get(StoredObjectWriteIntent, intent_id)
        assert intent is not None and intent.status == "completed"


def test_two_workers_compete_safely_for_write_intent_and_project(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    _store, intent_id, _project_id = _pending_export(postgres_database, storage)

    def claim_intent(owner: str) -> list[UUID]:
        return [
            lease.item_id
            for lease in LeaseStore(postgres_database).claim_write_intents(
                owner, batch_size=1, lease_seconds=60
            )
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        intent_results = list(pool.map(claim_intent, ("intent-a", "intent-b")))
    assert [item for result in intent_results for item in result] == [intent_id]

    # Clear the winning lease before exercising the independent Project authority.
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_object_write_intents SET lease_owner = NULL, lease_until = NULL "
                "WHERE id = :id"
            ),
            {"id": intent_id},
        )
    _guest_id, project_id = _guest_project(postgres_database)

    def claim_project(owner: str) -> list[UUID]:
        return [
            lease.item_id
            for lease in LeaseStore(postgres_database).claim_projects(
                owner, batch_size=1, lease_seconds=60
            )
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        project_results = list(pool.map(claim_project, ("project-a", "project-b")))
    assert [item for result in project_results for item in result] == [project_id]


def test_replacement_worker_fences_old_write_intent_owner(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    store, intent_id, _project_id = _pending_export(postgres_database, storage)
    leases = LeaseStore(postgres_database)
    old = leases.claim_write_intents("stale-intent", batch_size=1, lease_seconds=60)[0]
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_object_write_intents SET lease_until = "
                "clock_timestamp() - INTERVAL '1 second' WHERE id = :id"
            ),
            {"id": intent_id},
        )
    replacement = leases.claim_write_intents("replacement-intent", batch_size=1, lease_seconds=60)[
        0
    ]
    assert replacement.fencing_token == old.fencing_token + 1
    PendingObjectReconciler(store)(old, leases)
    with postgres_database.session() as session:
        intent = session.get(StoredObjectWriteIntent, intent_id)
        assert intent is not None and intent.status == "pending"
    PendingObjectReconciler(store)(replacement, leases)
    with postgres_database.session() as session:
        intent = session.get(StoredObjectWriteIntent, intent_id)
        assert intent is not None and intent.status == "completed"


def test_project_purge_records_candidates_before_foreign_key_cascade(
    postgres_database: Database,
) -> None:
    _guest_id, project_id = _guest_project(postgres_database)
    object_id = _stored_object(postgres_database, gc_candidate=False)
    with postgres_database.session() as session:
        session.add(
            SourceFile(
                id=uuid4(),
                project_id=project_id,
                stored_object_id=object_id,
                original_name="source.csv",
                media_type="text/csv",
                size_bytes=3,
                sha256=hashlib.sha256(b"csv").hexdigest(),
                available_sheets=[],
                parser_name="csv",
                parser_version="1",
            )
        )
    leases = LeaseStore(postgres_database)
    claim = leases.claim_projects("project-purge", batch_size=1, lease_seconds=60)[0]
    ProjectLifecycleHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(claim, leases)
    with postgres_database.session() as session:
        assert session.get(Project, project_id) is None
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.gc_candidate_at is not None


def test_project_purge_records_every_authoritative_object_reference_type(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = ConfirmThenFailOnceStorage(tmp_path / "objects")
    store = PostgresProjectStore(postgres_database, storage, 7_200)
    project_id = uuid4().hex
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    frame = pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})
    source_payload = frame.to_csv(index=False).encode()
    source = {
        "name": "closure.csv",
        "size": len(source_payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    store.create_project(
        project_id=project_id,
        job_id=uuid4().hex,
        title="Full deletion closure",
        source=source,
        source_sha256=hashlib.sha256(source_payload).hexdigest(),
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
    project = store.get_project(project_id, touch=False)
    assert project is not None
    store.create_publication_export(
        project_id=project_id,
        expected_revision_id=project["current_revision_id"],
        chart=_chart("png"),
        payload=render_chart(frame, ChartSpec.model_validate(_chart("png"))),
        owner_user_id=None,
        guest_token_digest=guest_digest,
    )
    storage.fail_after_confirm = True
    with pytest.raises(ObjectConfirmationPending):
        store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=_chart("svg"),
            payload=render_chart(frame, ChartSpec.model_validate(_chart("svg"))),
            owner_user_id=None,
            guest_token_digest=guest_digest,
        )
    source_object_id = uuid4()
    project_uuid = UUID(project_id)
    with postgres_database.session() as session:
        source_row = session.scalar(select(SourceFile).where(SourceFile.project_id == project_uuid))
        assert source_row is not None
        session.add(
            StoredObject(
                id=source_object_id,
                storage_backend="local",
                object_key=f"sources/{source_object_id}.csv",
                purpose="source-upload",
                status="available",
                media_type="text/csv",
                size_bytes=len(source_payload),
                sha256=hashlib.sha256(source_payload).hexdigest(),
                dedup_scope=f"guest:{source_row.project.guest_session_id}",
                format_contract_version="source-v1",
            )
        )
        session.flush()
        source_row.stored_object_id = source_object_id
        dataset_ids = set(
            session.scalars(
                select(DatasetVersion.stored_object_id).where(
                    DatasetVersion.project_id == project_uuid
                )
            )
        )
        publication_ids = set(
            session.scalars(
                select(PublicationExport.stored_object_id).where(
                    PublicationExport.project_id == project_uuid
                )
            )
        )
        intent_ids = set(
            session.scalars(
                select(StoredObjectWriteIntent.stored_object_id).where(
                    StoredObjectWriteIntent.project_id == project_uuid,
                    StoredObjectWriteIntent.status == "pending",
                )
            )
        )
        pending_job_ids = set(
            session.scalars(
                select(ExportJobRecord.pending_stored_object_id).where(
                    ExportJobRecord.project_id == project_uuid,
                    ExportJobRecord.pending_stored_object_id.is_not(None),
                )
            )
        )
        expected_ids = {
            source_object_id,
            *dataset_ids,
            *publication_ids,
            *intent_ids,
            *pending_job_ids,
        }
        assert dataset_ids and publication_ids and intent_ids and pending_job_ids
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE projects SET expires_at = clock_timestamp() - INTERVAL '1 second' "
                "WHERE id = :id"
            ),
            {"id": project_uuid},
        )
    leases = LeaseStore(postgres_database)
    claim = leases.claim_projects("full-project-purge", batch_size=1, lease_seconds=60)[0]
    ProjectLifecycleHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(claim, leases)
    with postgres_database.session() as session:
        rows = list(session.scalars(select(StoredObject).where(StoredObject.id.in_(expected_ids))))
        assert {row.id for row in rows} == expected_ids
        assert all(row.gc_candidate_at is not None for row in rows)
        assert all(
            row.status == "deleting" and row.staging_key is None
            for row in rows
            if row.id in intent_ids
        )


def test_restore_wins_before_fenced_purge_and_project_survives(
    postgres_database: Database,
) -> None:
    _guest_id, project_id = _guest_project(postgres_database, saved_deleted=True)
    leases = LeaseStore(postgres_database)
    claim = leases.claim_projects("purge-race", batch_size=1, lease_seconds=60)[0]
    with postgres_database.session() as session:
        project = session.get(Project, project_id)
        assert project is not None
        project.deleted_at = None
        project.purge_after = None
    ProjectLifecycleHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(claim, leases)
    with postgres_database.session() as session:
        assert session.get(Project, project_id) is not None


class DeleteFailureStorage(LocalObjectStorage):
    transient_failures = 0
    permanent_failure = False
    delete_calls = 0

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        self.delete_calls += 1
        if self.permanent_failure:
            raise PermissionError("object policy denies deletion")
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise OSError("temporary object-store outage")
        return super().delete(key, expected=expected)


class DeleteThenFailOnceStorage(LocalObjectStorage):
    failed = False

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        existed = super().delete(key, expected=expected)
        if not self.failed:
            self.failed = True
            raise OSError("process stopped after external delete")
        return existed


def _run_gc(
    database: Database,
    storage: LocalObjectStorage,
    *,
    owner: str,
    safety: MaintenanceSafety | None = None,
    policy: RetryPolicy | None = None,
) -> UUID:
    leases = LeaseStore(database, gc_orphan_age_seconds=1)
    claim = leases.claim_stored_objects(owner, batch_size=1, lease_seconds=60)[0]
    StoredObjectGarbageCollector(
        storage,
        safety or MaintenanceSafety(dry_run=False, delete_enabled=True),
        policy or RetryPolicy(),
    )(claim, leases)
    return claim.item_id


def test_gc_404_is_idempotent_success_and_never_leaves_a_database_reference(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    object_id = _stored_object(postgres_database)
    deleted_id = _run_gc(
        postgres_database,
        LocalObjectStorage(tmp_path / "empty-objects"),
        owner="gc-404",
    )
    assert deleted_id == object_id
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "deleted"


def test_gc_transient_retry_then_success_and_permanent_quarantine(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    transient_storage = DeleteFailureStorage(tmp_path / "transient")
    info = transient_storage.put("exports/transient.png", io.BytesIO(b"png"))
    object_id = _stored_object(postgres_database, key=info.key)
    transient_storage.transient_failures = 1
    _run_gc(
        postgres_database,
        transient_storage,
        owner="gc-transient",
        policy=RetryPolicy(max_retries=3, base_seconds=1),
    )
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "deleting"
        assert stored.retry_count == 1 and stored.next_attempt_at is not None
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_objects SET next_attempt_at = "
                "clock_timestamp() - INTERVAL '1 second' WHERE id = :id"
            ),
            {"id": object_id},
        )
    _run_gc(postgres_database, transient_storage, owner="gc-retry")
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "deleted"

    permanent_storage = DeleteFailureStorage(tmp_path / "permanent")
    permanent_info = permanent_storage.put("exports/permanent.png", io.BytesIO(b"png"))
    permanent_id = _stored_object(postgres_database, key=permanent_info.key)
    permanent_storage.permanent_failure = True
    _run_gc(postgres_database, permanent_storage, owner="gc-permanent")
    with postgres_database.session() as session:
        stored = session.get(StoredObject, permanent_id)
        assert stored is not None
        assert stored.status == "deleting" and stored.quarantined_at is not None


def test_gc_recovers_when_process_stops_after_external_delete(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = DeleteThenFailOnceStorage(tmp_path / "objects")
    info = storage.put("exports/crash.png", io.BytesIO(b"png"))
    object_id = _stored_object(postgres_database, key=info.key)
    _run_gc(
        postgres_database,
        storage,
        owner="gc-crash-after-delete",
        policy=RetryPolicy(max_retries=3, base_seconds=1),
    )
    assert not storage.exists(info.key)
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "deleting"
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_objects SET next_attempt_at = "
                "clock_timestamp() - INTERVAL '1 second' WHERE id = :id"
            ),
            {"id": object_id},
        )
    _run_gc(postgres_database, storage, owner="gc-crash-replacement")
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "deleted"


def test_gc_discovers_historic_orphan_without_gc_candidate_at(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    object_id = _stored_object(
        postgres_database,
        gc_candidate=False,
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    deleted_id = _run_gc(
        postgres_database,
        LocalObjectStorage(tmp_path / "objects"),
        owner="gc-historic",
    )
    assert deleted_id == object_id


def test_gc_rechecks_soft_deleted_project_reference_and_does_not_open_delete_path(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    _guest_id, project_id = _guest_project(postgres_database, expired=False)
    object_id = _stored_object(postgres_database)
    storage = DeleteFailureStorage(tmp_path / "objects")
    storage.put(f"exports/{object_id}.png", io.BytesIO(b"png"))
    with postgres_database.session() as session:
        session.add(
            SourceFile(
                project_id=project_id,
                stored_object_id=object_id,
                original_name="kept.csv",
                media_type="text/csv",
                size_bytes=3,
                sha256=hashlib.sha256(b"csv").hexdigest(),
                available_sheets=[],
                parser_name="csv",
                parser_version="1",
            )
        )
    _run_gc(postgres_database, storage, owner="gc-referenced")
    with postgres_database.session() as session:
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "available"
        assert stored.gc_candidate_at is None
    assert storage.delete_calls == 0


def test_dry_run_never_purges_project_or_deletes_object(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    _guest_id, project_id = _guest_project(postgres_database)
    object_id = _stored_object(postgres_database)
    storage = DeleteFailureStorage(tmp_path / "objects")
    key = f"exports/{object_id}.png"
    storage.put(key, io.BytesIO(b"png"))
    leases = LeaseStore(postgres_database)
    project_claim = leases.claim_projects("dry-project", batch_size=1, lease_seconds=60)[0]
    ProjectLifecycleHandler(MaintenanceSafety())(project_claim, leases)
    _run_gc(
        postgres_database,
        storage,
        owner="dry-gc",
        safety=MaintenanceSafety(),
    )
    with postgres_database.session() as session:
        assert session.get(Project, project_id) is not None
        stored = session.get(StoredObject, object_id)
        assert stored is not None and stored.status == "available"
    assert storage.delete_calls == 0


def test_orphan_staging_requires_two_inventories_and_grace_before_delete(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    staged = storage.stage("datasets/final.parquet", io.BytesIO(b"parquet"))
    leases = LeaseStore(postgres_database)
    handler = OrphanStagingHandler(
        storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
        grace_seconds=60,
    )
    first = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "inventory-1", 60)
    assert first is not None
    assert handler(first, leases) == 0
    assert leases.release_task(first)
    assert storage.exists(staged.staging_key)
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE orphan_staging_candidates SET first_seen_at = "
                "clock_timestamp() - INTERVAL '2 minutes' WHERE staging_key = :key"
            ),
            {"key": staged.staging_key},
        )
    second = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "inventory-2", 60)
    assert second is not None
    assert handler(second, leases) == 1
    assert not storage.exists(staged.staging_key)
    with postgres_database.session() as session:
        tombstone = session.get(
            OrphanStagingCandidate,
            (storage.backend_name, storage.inventory_scope, staged.staging_key),
        )
        assert tombstone is not None and tombstone.deletion_completed_at is not None


def test_orphan_staging_active_database_owner_is_never_candidate(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    storage = DeleteFailureStorage(tmp_path / "objects")
    staged = storage.stage("datasets/final.parquet", io.BytesIO(b"parquet"))
    object_id = uuid4()
    with postgres_database.session() as session:
        session.add(
            StoredObject(
                id=object_id,
                storage_backend="local",
                object_key=staged.key,
                staging_key=staged.staging_key,
                purpose="dataset",
                status="pending",
                media_type="application/vnd.apache.parquet",
                size_bytes=staged.size_bytes,
                sha256=staged.sha256,
                dedup_scope="guest:test",
                format_contract_version="parquet-v1",
            )
        )
    leases = LeaseStore(postgres_database)
    task = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "active-inventory", 60)
    assert task is not None
    handler = OrphanStagingHandler(
        storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
        grace_seconds=1,
    )
    assert handler(task, leases) == 0
    assert storage.exists(staged.staging_key)
    assert storage.delete_calls == 0


def test_metadata_cleanup_preserves_guest_until_project_is_purged(
    postgres_database: Database,
) -> None:
    referenced_guest_id, project_id = _guest_project(postgres_database)
    free_guest_id = uuid4()
    user_id = uuid4()
    now = datetime.now(UTC)
    with postgres_database.session() as session:
        session.add(User(id=user_id, email=f"{user_id.hex}@example.test"))
        session.flush()
        session.add_all(
            [
                GuestSession(
                    id=free_guest_id,
                    token_digest=hashlib.sha256(free_guest_id.bytes).hexdigest(),
                    expires_at=now - timedelta(hours=1),
                ),
                AuthChallenge(
                    id=uuid4(),
                    email="expired@example.test",
                    salt="a" * 64,
                    code_digest="b" * 64,
                    expires_at=now - timedelta(seconds=1),
                    resend_at=now - timedelta(seconds=1),
                ),
                AuthRequest(
                    id=uuid4(),
                    client_key="client",
                    email="expired@example.test",
                    requested_at=now - timedelta(hours=2),
                ),
                IdempotencyRecord(
                    id=uuid4(),
                    actor_user_id=user_id,
                    operation="export",
                    idempotency_key="expired-key",
                    request_sha256="c" * 64,
                    resource_id=uuid4(),
                    response_document={},
                    expires_at=now - timedelta(seconds=1),
                ),
            ]
        )
    leases = LeaseStore(postgres_database)
    task = leases.acquire_task(METADATA_CLEANUP, "metadata-before-purge", 60)
    assert task is not None
    cleaned = MetadataCleanupHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(
        task, leases
    )
    assert cleaned >= 4
    assert leases.release_task(task)
    with postgres_database.session() as session:
        assert session.get(GuestSession, referenced_guest_id) is not None
        assert session.get(GuestSession, free_guest_id) is None
        assert session.scalar(select(IdempotencyRecord)) is None

    project_claim = leases.claim_projects("project-before-guest", batch_size=1, lease_seconds=60)[0]
    assert project_claim.item_id == project_id
    ProjectLifecycleHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(
        project_claim, leases
    )
    task = leases.acquire_task(METADATA_CLEANUP, "metadata-after-purge", 60)
    assert task is not None
    MetadataCleanupHandler(MaintenanceSafety(dry_run=False, delete_enabled=True))(task, leases)
    with postgres_database.session() as session:
        assert session.get(GuestSession, referenced_guest_id) is None


def test_metadata_and_orphan_dry_run_have_no_destructive_side_effects(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    free_guest_id = uuid4()
    with postgres_database.session() as session:
        session.add(
            GuestSession(
                id=free_guest_id,
                token_digest=hashlib.sha256(free_guest_id.bytes).hexdigest(),
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )
    storage = DeleteFailureStorage(tmp_path / "objects")
    staged = storage.stage("exports/final.png", io.BytesIO(b"png"))
    leases = LeaseStore(postgres_database)
    metadata_task = leases.acquire_task(METADATA_CLEANUP, "metadata-dry", 60)
    assert metadata_task is not None
    assert MetadataCleanupHandler(MaintenanceSafety())(metadata_task, leases) == 1
    assert leases.release_task(metadata_task)
    orphan_task = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "orphan-dry", 60)
    assert orphan_task is not None
    assert (
        OrphanStagingHandler(
            storage,
            MaintenanceSafety(),
            RetryPolicy(),
            grace_seconds=1,
        )(orphan_task, leases)
        == 1
    )
    with postgres_database.session() as session:
        assert session.get(GuestSession, free_guest_id) is not None
        assert (
            session.get(
                OrphanStagingCandidate,
                (storage.backend_name, storage.inventory_scope, staged.staging_key),
            )
            is not None
        )
    assert storage.exists(staged.staging_key)
    assert storage.delete_calls == 0
