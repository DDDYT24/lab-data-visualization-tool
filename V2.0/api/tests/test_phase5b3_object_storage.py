from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import boto3
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from labviz_api.config import Settings
from labviz_api.db.models import (
    DatasetVersion,
    ExportJobRecord,
    OrphanStagingCandidate,
    StorageInventoryCheckpoint,
    StoredObject,
    StoredObjectWriteIntent,
)
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.models import ChartSpec
from labviz_api.persistence.exceptions import ObjectConfirmationPending
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import (
    build_preview,
    build_quality_report,
    default_chart_spec,
    render_chart,
)
from labviz_api.storage import ObjectInfo, ObjectStorage, StagedObject, StagingPage
from labviz_api.storage.factory import build_object_storage
from labviz_api.storage.s3 import S3ObjectStorage
from labviz_api.workers.garbage_collection import StoredObjectGarbageCollector
from labviz_api.workers.leases import ORPHAN_STAGING_INVENTORY, LeaseStore, RetryPolicy
from labviz_api.workers.orphan_staging import OrphanStagingHandler
from labviz_api.workers.references import lock_staging_key
from labviz_api.workers.safety import MaintenanceSafety, PermanentWorkerFailure

API_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL = os.environ.get(
    "LABVIZ_TEST_POSTGRES_URL",
    "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test",
)
MINIO_ENDPOINT = os.environ.get("LABVIZ_TEST_MINIO_ENDPOINT", "http://127.0.0.1:59000")
MINIO_BUCKET = os.environ.get("LABVIZ_TEST_MINIO_BUCKET", "labviz-test")
MINIO_ACCESS_KEY = os.environ.get("LABVIZ_TEST_MINIO_ACCESS_KEY", "labviz-minio")
MINIO_SECRET_KEY = os.environ.get(
    "LABVIZ_TEST_MINIO_SECRET_KEY",
    "labviz-minio-local-only",
)
FIVE_MIB = 5 * 1024 * 1024


def _alembic_config() -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = POSTGRES_URL
    return config


def _minio_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 1}),
    )


def _ensure_minio() -> Any:
    client = _minio_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", "Unknown"))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            pytest.fail(f"Real MinIO is unavailable: {code}")
        client.create_bucket(Bucket=MINIO_BUCKET)
    return client


@pytest.fixture(scope="module")
def postgres_database() -> Iterator[Database]:
    database = Database(POSTGRES_URL)
    if not database.health().ready:
        database.dispose()
        pytest.fail("Real PostgreSQL is required for Phase 5B-3 integration tests.")
    command.upgrade(_alembic_config(), "head")
    try:
        yield database
    finally:
        with database.engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
            connection.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
            connection.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
        database.dispose()


@pytest.fixture(autouse=True)
def clean_postgres(postgres_database: Database) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
        connection.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
        connection.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
        connection.execute(
            text(
                "UPDATE worker_leases SET lease_owner = NULL, lease_until = NULL, "
                "heartbeat_at = NULL, updated_at = clock_timestamp()"
            )
        )


@pytest.fixture
def minio_storage() -> Iterator[S3ObjectStorage]:
    client = _ensure_minio()
    prefix = f"phase5b3/{uuid4().hex}"
    storage = S3ObjectStorage(
        bucket=MINIO_BUCKET,
        prefix=prefix,
        endpoint_url=MINIO_ENDPOINT,
        region="us-east-1",
        multipart_threshold=FIVE_MIB,
        multipart_part_size=FIVE_MIB,
        client=client,
    )
    yield storage
    continuation: str | None = None
    while True:
        parameters: dict[str, Any] = {
            "Bucket": MINIO_BUCKET,
            "Prefix": f"{prefix}/",
        }
        if continuation is not None:
            parameters["ContinuationToken"] = continuation
        listed = client.list_objects_v2(**parameters)
        objects = [{"Key": item["Key"]} for item in listed.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=MINIO_BUCKET, Delete={"Objects": objects})
        if not listed.get("IsTruncated"):
            break
        continuation = str(listed["NextContinuationToken"])


def _settings(tmp_path: Path, prefix: str) -> Settings:
    return Settings(
        database_path=tmp_path / "reference.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
        environment="test",
        postgres_url=POSTGRES_URL,
        persistence_backend="postgresql",
        object_storage_backend="s3",
        s3_bucket=MINIO_BUCKET,
        s3_prefix=prefix,
        s3_region="us-east-1",
        s3_endpoint_url=MINIO_ENDPOINT,
        s3_multipart_threshold_bytes=FIVE_MIB,
        s3_multipart_part_size_bytes=FIVE_MIB,
    )


def _aws_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _chart(frame: pd.DataFrame) -> dict[str, Any]:
    return ChartSpec.model_validate(default_chart_spec(frame)).model_dump(
        mode="json",
        by_alias=True,
    )


def _complete_guest_project(
    store: PostgresProjectStore,
) -> tuple[str, str, pd.DataFrame, dict[str, Any]]:
    project_id = uuid4().hex
    job_id = uuid4().hex
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    payload = b"time,response\n0,1.2\n1,1.4\n2,1.8\n"
    source = {
        "name": "experiment.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": None,
    }
    frame = pd.DataFrame({"time": [0, 1, 2], "response": [1.2, 1.4, 1.8]})
    preview = build_preview(project_id, frame)
    quality = build_quality_report(project_id, frame)
    chart = _chart(frame)
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="S3 integration",
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        guest_token_digest=guest_digest,
    )
    store.update_job(job_id, stage="parsing", progress=45, message="Parsing")
    store.complete_project(
        project_id=project_id,
        source=source,
        frame=frame,
        preview=preview,
        quality=quality,
        chart=chart,
    )
    store.update_job(job_id, stage="ready", progress=100, message="Ready")
    return project_id, guest_digest, frame, chart


class DelegatingStorage:
    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    @property
    def backend_name(self) -> str:
        return self.storage.backend_name

    @property
    def inventory_scope(self) -> str:
        return self.storage.inventory_scope

    def __getattr__(self, name: str) -> Any:
        return getattr(self.storage, name)


class ConfirmThenFailStorage(DelegatingStorage):
    def __init__(self, storage: ObjectStorage) -> None:
        super().__init__(storage)
        self.fail_after_confirm = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        confirmed = self.storage.confirm(staged)
        if self.fail_after_confirm:
            self.fail_after_confirm = False
            raise OSError("injected process stop after provider confirm")
        return confirmed


class FailBeforeConfirmStorage(DelegatingStorage):
    def confirm(self, staged: StagedObject) -> ObjectInfo:
        raise OSError("injected provider interruption before confirm")


class TransactionProbeStorage(DelegatingStorage):
    def __init__(self, storage: ObjectStorage, database: Database) -> None:
        super().__init__(storage)
        self.database = database
        self.probe_calls = 0

    def _assert_no_open_transaction(self) -> None:
        with self.database.engine.connect() as connection:
            count = connection.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                    "AND state = 'idle in transaction'"
                )
            )
        assert count == 0
        self.probe_calls += 1

    def open(self, key: str) -> BinaryIO:
        self._assert_no_open_transaction()
        return self.storage.open(key)

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        self._assert_no_open_transaction()
        return self.storage.confirm(staged)

    def list_staged(self, *, page_size: int = 250, cursor: str | None = None) -> StagingPage:
        self._assert_no_open_transaction()
        return self.storage.list_staged(page_size=page_size, cursor=cursor)

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        self._assert_no_open_transaction()
        return self.storage.delete(key, expected=expected)


class FailListOnceStorage(DelegatingStorage):
    def __init__(self, storage: ObjectStorage) -> None:
        super().__init__(storage)
        self.failed = False

    def list_staged(self, *, page_size: int = 250, cursor: str | None = None) -> StagingPage:
        if not self.failed:
            self.failed = True
            raise OSError("injected transient inventory failure")
        return self.storage.list_staged(page_size=page_size, cursor=cursor)


def test_api_factory_postgres_minio_round_trip_export_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_database: Database,
) -> None:
    _ensure_minio()
    _aws_test_credentials(monkeypatch)
    prefix = f"api/{uuid4().hex}"
    settings = _settings(tmp_path, prefix)
    provider = build_object_storage(settings)
    assert provider.backend_name == "s3"
    assert provider.inventory_scope == f"s3://{MINIO_BUCKET}/{prefix}/.staging"

    cookies: dict[str, str]
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/v1/samples/thermal-response/projects")
        assert created.status_code == 202
        project_id = created.json()["projectId"]
        workspace = client.get(f"/api/v1/projects/{project_id}/workspace")
        assert workspace.status_code == 200
        chart = workspace.json()["chart"]
        analysis = client.post(
            f"/api/v1/projects/{project_id}/chart-analysis",
            json={"chart": chart},
        )
        assert analysis.status_code == 200
        exported = client.post(
            f"/api/v1/projects/{project_id}/exports",
            json={"chart": chart},
            headers={"Idempotency-Key": "phase5b3-real-minio"},
        )
        assert exported.status_code == 200
        replay = client.post(
            f"/api/v1/projects/{project_id}/exports",
            json={"chart": chart},
            headers={"Idempotency-Key": "phase5b3-real-minio"},
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == exported.json()["id"]
        download = client.get(exported.json()["downloadUrl"])
        assert download.status_code == 200
        assert download.content.startswith(b"\x89PNG\r\n\x1a\n")
        cookies = {name: value for name, value in client.cookies.items()}

    with postgres_database.session() as session:
        objects = list(session.scalars(select(StoredObject)))
        assert len(objects) >= 2
        assert all(item.storage_backend == "s3" for item in objects)
        assert all(item.size_bytes > 0 and len(item.sha256) == 64 for item in objects)
        for item in objects:
            info = provider.head(item.object_key)
            assert info is not None
            assert info.sha256 == item.sha256 and info.size_bytes == item.size_bytes
            assert info.metadata.get("labviz-format-version") == item.format_contract_version

    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0

    with TestClient(create_app(settings)) as restarted:
        restarted.cookies.update(cookies)
        reopened = restarted.get(f"/api/v1/projects/{project_id}/workspace")
        assert reopened.status_code == 200
        assert reopened.json()["session"]["projectId"] == project_id


def test_real_minio_dataset_confirm_crash_recovers_with_same_fenced_worker_path(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    failing = ConfirmThenFailStorage(minio_storage)
    failing.fail_after_confirm = True
    store = PostgresProjectStore(postgres_database, failing, 7_200)
    with pytest.raises(ObjectConfirmationPending):
        _complete_guest_project(store)
    with postgres_database.session() as session:
        stored = session.scalar(select(StoredObject).where(StoredObject.status == "pending"))
        assert stored is not None and stored.storage_backend == "s3"
        assert stored.staging_key is not None
        pending_key = stored.staging_key
        version = session.scalar(select(DatasetVersion))
        assert version is not None and version.stored_object_id == stored.id
        assert stored.lease_owner is None and stored.lease_until is None
    assert minio_storage.head(pending_key) is None

    restarted = PostgresProjectStore(postgres_database, minio_storage, 7_200)
    assert restarted.recover_pending_objects() == 1
    with postgres_database.session() as session:
        stored = session.scalar(select(StoredObject))
        assert stored is not None and stored.status == "available"
        assert stored.staging_key is None
        assert stored.lease_owner is None and stored.lease_until is None
        assert minio_storage.head(stored.object_key) is not None


def test_minio_io_runs_after_postgres_transactions_commit(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    probe = TransactionProbeStorage(minio_storage, postgres_database)
    store = PostgresProjectStore(postgres_database, probe, 7_200)
    project_id, _guest_digest, _frame, _chart_document = _complete_guest_project(store)
    frame = store.load_chart_dataframe(project_id)
    assert list(frame.columns) == ["time", "response"]
    minio_storage.stage("datasets/orphan.parquet", io.BytesIO(b"orphan"))
    leases = LeaseStore(postgres_database, storage_backend="s3")
    handler = OrphanStagingHandler(
        probe,
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
    )
    assert _acquire_and_scan(handler, leases, "transaction-probe") == 1
    assert probe.probe_calls >= 3


def test_write_intent_is_explicit_inventory_root_and_lock_fences_delete(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    base_store = PostgresProjectStore(postgres_database, minio_storage, 7_200)
    project_id, guest_digest, frame, chart = _complete_guest_project(base_store)
    project = base_store.get_project(project_id, touch=False)
    assert project is not None
    failing_store = PostgresProjectStore(
        postgres_database,
        FailBeforeConfirmStorage(minio_storage),
        7_200,
    )
    with pytest.raises(ObjectConfirmationPending):
        failing_store.create_publication_export(
            project_id=project_id,
            expected_revision_id=project["current_revision_id"],
            chart=chart,
            payload=render_chart(frame, ChartSpec.model_validate(chart)),
            owner_user_id=None,
            guest_token_digest=guest_digest,
        )
    with postgres_database.session() as session:
        intent = session.scalar(select(StoredObjectWriteIntent))
        assert intent is not None and intent.status == "pending"
        stored = session.get(StoredObject, intent.stored_object_id)
        assert stored is not None and stored.staging_key is not None
        staging_key = stored.staging_key
    current = minio_storage.head(staging_key)
    assert current is not None and current.last_modified is not None
    now = datetime.now(UTC)
    with postgres_database.session() as session:
        session.add(
            OrphanStagingCandidate(
                backend_name="s3",
                inventory_scope=minio_storage.inventory_scope,
                staging_key=staging_key,
                size_bytes=current.size_bytes,
                sha256=current.sha256,
                provider_last_modified=current.last_modified,
                provider_etag=current.etag,
                first_seen_at=now - timedelta(minutes=2),
                last_seen_at=now,
                observation_count=2,
            )
        )
    leases = LeaseStore(postgres_database, storage_backend="s3")
    task = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "intent-race", 60)
    assert task is not None
    handler = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
        grace_seconds=1,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.session() as session:
            lock_staging_key(session, "s3", staging_key)
            future = executor.submit(handler._claim_deletion, task, leases, current)
            time.sleep(0.1)
            assert not future.done()
        assert future.result(timeout=5) is False
    assert minio_storage.head(staging_key) is not None
    with postgres_database.session() as session:
        assert (
            session.get(
                OrphanStagingCandidate,
                ("s3", minio_storage.inventory_scope, staging_key),
            )
            is None
        )
        intent = session.scalar(select(StoredObjectWriteIntent))
        assert intent is not None and intent.status == "pending"
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_object_write_intents SET next_attempt_at = "
                "clock_timestamp() - INTERVAL '1 second'"
            )
        )
    recovered = PostgresProjectStore(postgres_database, minio_storage, 7_200)
    assert recovered.recover_pending_objects() == 1
    with postgres_database.session() as session:
        intent = session.scalar(select(StoredObjectWriteIntent))
        job = session.scalar(select(ExportJobRecord))
        assert intent is not None and intent.status == "completed"
        assert job is not None and job.status == "ready"


def _acquire_and_scan(
    handler: OrphanStagingHandler,
    leases: LeaseStore,
    owner: str,
) -> int:
    task = leases.acquire_task(ORPHAN_STAGING_INVENTORY, owner, 60)
    assert task is not None
    try:
        return handler(task, leases)
    finally:
        leases.release_task(task)


def test_inventory_checkpoint_resumes_pages_and_counts_only_complete_generations(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    staged = [
        minio_storage.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
        for index in range(5)
    ]
    leases = LeaseStore(postgres_database, storage_backend="s3")
    dry_handler = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
        batch_size=2,
    )
    assert _acquire_and_scan(dry_handler, leases, "page-1") == 0
    with postgres_database.session() as session:
        checkpoint = session.scalar(select(StorageInventoryCheckpoint))
        assert checkpoint is not None and checkpoint.status == "running"
        assert checkpoint.page_count == 1 and checkpoint.cursor is not None
        assert session.scalar(select(OrphanStagingCandidate.observation_count)) == 0
    assert _acquire_and_scan(dry_handler, leases, "page-2") == 0
    assert _acquire_and_scan(dry_handler, leases, "page-3") == 5
    with postgres_database.session() as session:
        checkpoint = session.scalar(select(StorageInventoryCheckpoint))
        assert checkpoint is not None and checkpoint.status == "completed"
        assert checkpoint.page_count == 3 and checkpoint.item_count == 5
        observations = {
            item.observation_count for item in session.scalars(select(OrphanStagingCandidate))
        }
        assert observations == {1}
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE orphan_staging_candidates SET first_seen_at = "
                "clock_timestamp() - INTERVAL '2 seconds'"
            )
        )
    deleting = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
        grace_seconds=1,
        batch_size=2,
    )
    assert _acquire_and_scan(deleting, leases, "delete-page-1") == 0
    assert _acquire_and_scan(deleting, leases, "delete-page-2") == 0
    assert _acquire_and_scan(deleting, leases, "delete-page-3") == 2
    assert sum(minio_storage.head(item.staging_key) is None for item in staged) == 2
    assert _acquire_and_scan(deleting, leases, "delete-page-4") == 0
    assert _acquire_and_scan(deleting, leases, "delete-page-5") == 2
    assert _acquire_and_scan(deleting, leases, "delete-page-6") == 1


def test_inventory_takeover_fences_old_page_and_resumes_committed_cursor(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    for index in range(3):
        minio_storage.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
    leases = LeaseStore(postgres_database, storage_backend="s3")
    handler = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
        batch_size=2,
    )
    old = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "old", 60)
    assert old is not None
    old_claim = handler._claim_page(old, leases)
    old_page = minio_storage.list_staged(page_size=2, cursor=old_claim.cursor)
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE worker_leases SET lease_until = clock_timestamp() - INTERVAL '1 second' "
                "WHERE task = :task"
            ),
            {"task": ORPHAN_STAGING_INVENTORY},
        )
    takeover = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "takeover", 60)
    assert takeover is not None and takeover.fencing_token > old.fencing_token
    assert handler(takeover, leases) == 0
    assert handler._persist_page(
        old,
        leases,
        old_claim.generation_id,
        old_page.items,
        old_page.next_cursor,
        old_page.has_more,
    ) == (False, 0)
    assert leases.release_task(takeover)

    resumed = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "resumed", 60)
    assert resumed is not None
    assert handler(resumed, leases) == 3
    with postgres_database.session() as session:
        checkpoint = session.scalar(select(StorageInventoryCheckpoint))
        assert checkpoint is not None and checkpoint.status == "completed"
        assert checkpoint.page_count == 2 and checkpoint.item_count == 3
        assert checkpoint.task_fencing_token == resumed.fencing_token
        candidates = list(session.scalars(select(OrphanStagingCandidate)))
        assert len(candidates) == 3
        assert all(item.observation_count == 1 for item in candidates)


def test_inventory_invalid_persisted_cursor_fails_closed_without_restart(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    for index in range(3):
        minio_storage.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
    leases = LeaseStore(postgres_database, storage_backend="s3")
    handler = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
        batch_size=2,
    )
    assert _acquire_and_scan(handler, leases, "valid-page") == 0
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text("UPDATE storage_inventory_checkpoints SET cursor = 'invalid-cursor'"),
        )
    task = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "invalid-resume", 60)
    assert task is not None
    with pytest.raises(PermanentWorkerFailure, match="persisted inventory cursor"):
        handler(task, leases)
    with postgres_database.session() as session:
        checkpoint = session.scalar(select(StorageInventoryCheckpoint))
        assert checkpoint is not None and checkpoint.status == "failed"
        assert checkpoint.page_count == 1 and checkpoint.item_count == 2
        assert checkpoint.last_error_code == "InvalidStorageCursor"
        assert checkpoint.cursor == "invalid-cursor"


def test_inventory_transient_provider_failure_retries_committed_checkpoint(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    minio_storage.stage("datasets/retry.parquet", io.BytesIO(b"retry"))
    leases = LeaseStore(postgres_database, storage_backend="s3")
    failing = OrphanStagingHandler(
        FailListOnceStorage(minio_storage),
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
    )
    with pytest.raises(OSError, match="transient inventory failure"):
        _acquire_and_scan(failing, leases, "transient-failure")
    with postgres_database.session() as session:
        checkpoint = session.scalar(select(StorageInventoryCheckpoint))
        assert checkpoint is not None and checkpoint.status == "running"
        assert checkpoint.cursor is None and checkpoint.page_count == 0
    recovered = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(),
        RetryPolicy(),
        grace_seconds=1,
    )
    assert _acquire_and_scan(recovered, leases, "transient-retry") == 1


def test_orphan_delete_claim_recovers_before_and_after_external_io(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    leases = LeaseStore(postgres_database, storage_backend="s3")
    handler = OrphanStagingHandler(
        minio_storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
        grace_seconds=1,
    )

    def eligible_candidate(staged: StagedObject) -> ObjectInfo:
        current = minio_storage.head(staged.staging_key)
        assert current is not None and current.last_modified is not None
        now = datetime.now(UTC)
        with postgres_database.session() as session:
            session.add(
                OrphanStagingCandidate(
                    backend_name="s3",
                    inventory_scope=minio_storage.inventory_scope,
                    staging_key=staged.staging_key,
                    size_bytes=current.size_bytes,
                    sha256=current.sha256,
                    provider_last_modified=current.last_modified,
                    provider_etag=current.etag,
                    first_seen_at=now - timedelta(minutes=2),
                    last_seen_at=now,
                    observation_count=2,
                )
            )
        return current

    before_io = minio_storage.stage("datasets/before.parquet", io.BytesIO(b"before"))
    before_info = eligible_candidate(before_io)
    crashed_before = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "crash-before", 60)
    assert crashed_before is not None
    assert handler._claim_deletion(crashed_before, leases, before_info)
    assert leases.release_task(crashed_before)
    assert _acquire_and_scan(handler, leases, "recover-before") == 1
    assert minio_storage.head(before_io.staging_key) is None
    assert not handler._finish_deleted(
        crashed_before,
        leases,
        before_io.staging_key,
    )

    after_io = minio_storage.stage("datasets/after.parquet", io.BytesIO(b"after"))
    after_info = eligible_candidate(after_io)
    crashed_after = leases.acquire_task(ORPHAN_STAGING_INVENTORY, "crash-after", 60)
    assert crashed_after is not None
    assert handler._claim_deletion(crashed_after, leases, after_info)
    assert minio_storage.delete(after_io.staging_key, expected=after_info)
    assert leases.release_task(crashed_after)
    assert _acquire_and_scan(handler, leases, "recover-after") == 1
    with postgres_database.session() as session:
        tombstones = list(session.scalars(select(OrphanStagingCandidate)))
        assert len(tombstones) == 2
        assert all(item.deletion_completed_at is not None for item in tombstones)
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE orphan_staging_candidates SET deletion_completed_at = "
                "clock_timestamp() - INTERVAL '2 days'"
            )
        )
    assert _acquire_and_scan(handler, leases, "expire-tombstones") == 0
    with postgres_database.session() as session:
        assert session.scalar(select(OrphanStagingCandidate)) is None


def test_real_minio_gc_deletes_only_unreferenced_backend_objects(
    minio_storage: S3ObjectStorage,
    postgres_database: Database,
) -> None:
    store = PostgresProjectStore(postgres_database, minio_storage, 7_200)
    _complete_guest_project(store)
    orphan_info = minio_storage.put(
        f"exports/{uuid4().hex}.png",
        io.BytesIO(b"orphan"),
        metadata={
            "labviz-format-version": "publication-export-v1",
            "labviz-media-type": "image/png",
        },
    )
    orphan_id = uuid4()
    with postgres_database.session() as session:
        referenced = session.scalar(select(StoredObject))
        assert referenced is not None
        referenced.gc_candidate_at = datetime.now(UTC) - timedelta(seconds=2)
        session.add(
            StoredObject(
                id=orphan_id,
                storage_backend="s3",
                object_key=orphan_info.key,
                purpose="export",
                status="available",
                media_type="image/png",
                size_bytes=orphan_info.size_bytes,
                sha256=orphan_info.sha256,
                dedup_scope="guest:orphan",
                format_contract_version="publication-export-v1",
                gc_candidate_at=datetime.now(UTC) - timedelta(seconds=2),
            )
        )
        referenced_key = referenced.object_key
    leases = LeaseStore(
        postgres_database,
        gc_orphan_age_seconds=1,
        storage_backend="s3",
    )
    claims = leases.claim_stored_objects("s3-gc", batch_size=10, lease_seconds=60)
    assert orphan_id in {claim.item_id for claim in claims}
    collector = StoredObjectGarbageCollector(
        minio_storage,
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        RetryPolicy(),
    )
    for claim in claims:
        collector(claim, leases)
    assert minio_storage.head(orphan_info.key) is None
    assert minio_storage.head(referenced_key) is not None
    with postgres_database.session() as session:
        orphan = session.get(StoredObject, orphan_id)
        assert orphan is not None and orphan.status == "deleted"
        illegal = session.scalar(
            select(StoredObject.id).where(
                StoredObject.status.in_(("available", "deleted")),
                StoredObject.lease_owner.is_not(None),
            )
        )
        assert illegal is None


def test_0008_fail_closed_and_empty_round_trip(postgres_database: Database) -> None:
    now = datetime.now(UTC)
    with postgres_database.session() as session:
        session.add(
            StorageInventoryCheckpoint(
                backend_name="s3",
                inventory_scope="s3://test/.staging",
                generation_id=uuid4(),
                status="completed",
                cursor=None,
                started_at=now,
                last_checkpoint_at=now,
                completed_at=now,
                lease_owner="migration-test",
                task_fencing_token=1,
            )
        )
    with pytest.raises(RuntimeError, match="inventory metadata exists"):
        command.downgrade(_alembic_config(), "0007_phase5b2_orphan_staging")
    with postgres_database.session() as session:
        session.execute(text("TRUNCATE TABLE storage_inventory_checkpoints"))
        session.execute(text("TRUNCATE TABLE orphan_staging_candidates"))
        for backend in ("local", "s3"):
            session.add(
                StoredObject(
                    id=uuid4(),
                    storage_backend=backend,
                    object_key="datasets/shared-logical-key.parquet",
                    purpose="dataset",
                    status="available",
                    media_type="application/vnd.apache.parquet",
                    size_bytes=4,
                    sha256=hashlib.sha256(b"same").hexdigest(),
                    dedup_scope="guest:backend-scope",
                    format_contract_version="parquet-v1",
                )
            )
    with pytest.raises(RuntimeError, match="provider-scoped object keys collide"):
        command.downgrade(_alembic_config(), "0007_phase5b2_orphan_staging")
    with postgres_database.session() as session:
        session.execute(text("TRUNCATE TABLE stored_objects CASCADE"))
    command.downgrade(_alembic_config(), "0007_phase5b2_orphan_staging")
    table_names = inspect(postgres_database.engine).get_table_names()
    assert "storage_inventory_checkpoints" not in table_names
    command.upgrade(_alembic_config(), "head")
    assert "storage_inventory_checkpoints" in inspect(postgres_database.engine).get_table_names()
    command.check(_alembic_config())
