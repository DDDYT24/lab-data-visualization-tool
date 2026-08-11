from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import DBAPIError

from labviz_api.db.models import (
    GuestSession,
    Project,
    StoredObject,
    StoredObjectWriteIntent,
    WorkerLease,
)
from labviz_api.db.session import Database
from labviz_api.models import ChartSpec
from labviz_api.persistence.exceptions import ObjectConfirmationPending
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.processing import build_preview, build_quality_report, render_chart
from labviz_api.storage import LocalObjectStorage, ObjectInfo, StagedObject
from labviz_api.workers import leases as leases_module
from labviz_api.workers.leases import (
    STORED_OBJECT_GC,
    TASKS,
    LeaseStore,
    RetryPolicy,
    TaskLease,
    WorkItemLease,
)
from labviz_api.workers.runner import RunnerConfig, WorkerRunner

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
        tables = set(inspect(connection).get_table_names())
        if "projects" in tables:
            connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
        if "auth_rate_limit_buckets" in tables:
            connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))
    command.downgrade(_alembic_config(), "base")
    command.upgrade(_alembic_config(), "head")
    try:
        yield database
    finally:
        command.upgrade(_alembic_config(), "head")
        database.dispose()


@pytest.fixture(autouse=True)
def empty_postgres(postgres_database: Database) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE users, stored_objects, projects CASCADE"))
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))
        connection.execute(
            text(
                "UPDATE worker_leases SET lease_owner = NULL, lease_until = NULL, "
                "heartbeat_at = NULL, updated_at = clock_timestamp()"
            )
        )


def _gc_object(database: Database, *, key: str | None = None) -> UUID:
    object_id = uuid4()
    with database.session() as session:
        session.add(
            StoredObject(
                id=object_id,
                storage_backend="local",
                object_key=key or f"exports/{object_id}.png",
                purpose="export",
                status="available",
                media_type="image/png",
                size_bytes=3,
                sha256=hashlib.sha256(b"png").hexdigest(),
                dedup_scope="owner:test",
                format_contract_version="publication-export-v1",
                gc_candidate_at=datetime.now(UTC),
            )
        )
    return object_id


def _claim_gc(store: LeaseStore, owner: str, lease_seconds: int = 60) -> WorkItemLease:
    claims = store.claim_stored_objects(owner, batch_size=1, lease_seconds=lease_seconds)
    assert len(claims) == 1
    return claims[0]


def test_two_workers_cannot_claim_the_same_valid_work_item(
    postgres_database: Database,
) -> None:
    object_id = _gc_object(postgres_database)

    def claim(owner: str) -> list[WorkItemLease]:
        return LeaseStore(postgres_database).claim_stored_objects(
            owner, batch_size=1, lease_seconds=60
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("worker-a", "worker-b")))
    claimed = [lease for result in results for lease in result]
    assert [lease.item_id for lease in claimed] == [object_id]


def test_expired_lease_is_taken_over_and_old_worker_is_fenced(
    postgres_database: Database,
) -> None:
    object_id = _gc_object(postgres_database)
    store = LeaseStore(postgres_database)
    original = _claim_gc(store, "worker-old")
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_objects SET lease_until = clock_timestamp() - INTERVAL '1 second' "
                "WHERE id = :id"
            ),
            {"id": object_id},
        )

    replacement = _claim_gc(store, "worker-new")
    assert replacement.fencing_token == original.fencing_token + 1
    assert not store.release_item(original)
    assert not store.record_failure(
        original,
        error_code="stale-worker",
        error_message="must not overwrite the new owner",
        policy=RetryPolicy(),
    )


def test_heartbeat_extends_only_the_current_owner_lease(
    postgres_database: Database,
) -> None:
    _gc_object(postgres_database)
    store = LeaseStore(postgres_database)
    lease = _claim_gc(store, "worker-owner", lease_seconds=30)
    renewed = store.heartbeat_item(lease, 90)
    assert renewed is not None
    assert renewed.lease_until > lease.lease_until

    impostor = WorkItemLease(
        lease.kind,
        lease.item_id,
        "worker-other",
        lease.fencing_token,
        lease.lease_until,
        lease.expected_state,
    )
    assert store.heartbeat_item(impostor, 90) is None
    assert not store.release_item(impostor)
    assert store.release_item(renewed)


def test_task_lease_heartbeat_release_and_takeover_use_fencing(
    postgres_database: Database,
) -> None:
    store = LeaseStore(postgres_database)
    lease = store.acquire_task(STORED_OBJECT_GC, "scanner-a", 30)
    assert lease is not None
    assert store.acquire_task(STORED_OBJECT_GC, "scanner-b", 30) is None
    renewed = store.heartbeat_task(lease, 90)
    assert renewed is not None and renewed.lease_until > lease.lease_until
    impostor = TaskLease(lease.task, "scanner-b", lease.fencing_token, lease.lease_until)
    assert store.heartbeat_task(impostor, 90) is None
    assert not store.release_task(impostor)

    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE worker_leases SET lease_until = clock_timestamp() - INTERVAL '1 second' "
                "WHERE task = :task"
            ),
            {"task": lease.task},
        )
    takeover = store.acquire_task(STORED_OBJECT_GC, "scanner-b", 30)
    assert takeover is not None and takeover.fencing_token == lease.fencing_token + 1
    assert not store.release_task(renewed)


def test_runner_commits_claim_before_handler_external_io(
    postgres_database: Database,
) -> None:
    object_id = _gc_object(postgres_database)
    callback_observed_unlocked_row: list[bool] = []

    def simulated_external_io(lease: WorkItemLease, _store: LeaseStore) -> None:
        with postgres_database.session() as session:
            locked = session.scalar(
                select(StoredObject)
                .where(StoredObject.id == lease.item_id)
                .with_for_update(nowait=True)
            )
            callback_observed_unlocked_row.append(locked is not None)

    runner = WorkerRunner(
        task=STORED_OBJECT_GC,
        owner="runner-io",
        leases=LeaseStore(postgres_database),
        config=RunnerConfig(
            batch_size=1,
            lease_seconds=60,
            heartbeat_seconds=20,
            destructive_maintenance=True,
        ),
        item_handler=simulated_external_io,
    )
    assert runner.run_once() == 1
    assert callback_observed_unlocked_row == [True]
    with postgres_database.session() as session:
        row = session.get(StoredObject, object_id)
        assert row is not None and row.lease_owner is None


def test_runner_without_a_phase5b2_handler_does_not_claim_business_rows(
    postgres_database: Database,
) -> None:
    object_id = _gc_object(postgres_database)
    runner = WorkerRunner(
        task=STORED_OBJECT_GC,
        owner="runner-disabled",
        leases=LeaseStore(postgres_database),
        config=RunnerConfig(),
    )
    assert runner.run_once() == 0
    with postgres_database.session() as session:
        row = session.get(StoredObject, object_id)
        assert row is not None
        assert row.lease_owner is None
        assert row.fencing_token == 0
        assert row.last_attempt_at is None


def test_graceful_stop_releases_unprocessed_batch_leases(
    postgres_database: Database,
) -> None:
    first_id = _gc_object(postgres_database, key=f"exports/{uuid4()}.png")
    second_id = _gc_object(postgres_database, key=f"exports/{uuid4()}.png")
    runner: WorkerRunner

    def stop_after_first(_lease: WorkItemLease, _store: LeaseStore) -> None:
        runner.request_stop()

    runner = WorkerRunner(
        task=STORED_OBJECT_GC,
        owner="runner-graceful-stop",
        leases=LeaseStore(postgres_database),
        config=RunnerConfig(batch_size=2, destructive_maintenance=True),
        item_handler=stop_after_first,
    )
    assert runner.run_once() == 1
    with postgres_database.session() as session:
        rows = list(
            session.scalars(select(StoredObject).where(StoredObject.id.in_({first_id, second_id})))
        )
        assert len(rows) == 2
        assert all(row.lease_owner is None and row.lease_until is None for row in rows)


def test_runner_heartbeats_during_long_external_io(postgres_database: Database) -> None:
    _gc_object(postgres_database)
    lease_was_extended: list[bool] = []

    def long_external_io(lease: WorkItemLease, _store: LeaseStore) -> None:
        Event().wait(1.2)
        with postgres_database.session() as session:
            row = session.get(StoredObject, lease.item_id)
            assert row is not None and row.lease_until is not None
            lease_was_extended.append(row.lease_until > lease.lease_until)

    runner = WorkerRunner(
        task=STORED_OBJECT_GC,
        owner="runner-heartbeat",
        leases=LeaseStore(postgres_database),
        config=RunnerConfig(
            batch_size=1,
            lease_seconds=2,
            heartbeat_seconds=1,
            destructive_maintenance=True,
        ),
        item_handler=long_external_io,
    )
    assert runner.run_once() == 1
    assert lease_was_extended == [True]


def test_worker_decisions_ignore_skewed_host_clock(
    monkeypatch: pytest.MonkeyPatch,
    postgres_database: Database,
) -> None:
    _gc_object(postgres_database)

    class BrokenHostClock:
        @classmethod
        def now(cls, *_args: object, **_kwargs: object) -> datetime:
            raise AssertionError("host clock must not decide lease eligibility")

    monkeypatch.setattr(leases_module, "datetime", BrokenHostClock)
    assert _claim_gc(LeaseStore(postgres_database), "database-clock").owner == "database-clock"


def test_retry_backoff_truncation_and_quarantine_are_bounded(
    postgres_database: Database,
) -> None:
    object_id = _gc_object(postgres_database)
    store = LeaseStore(postgres_database)
    policy = RetryPolicy(max_retries=2, base_seconds=10, max_seconds=20)
    first = _claim_gc(store, "retry-worker")
    assert store.record_failure(
        first,
        error_code="E" * 200,
        error_message="detail" * 400,
        policy=policy,
    )
    with postgres_database.session() as session:
        row = session.get(StoredObject, object_id)
        assert row is not None
        assert row.retry_count == 1
        assert len(row.last_error_code or "") == 128
        assert len(row.last_error_message or "") == 1024
        assert (
            row.next_attempt_at is not None
            and row.last_attempt_at is not None
            and row.next_attempt_at > row.last_attempt_at
        )
        assert row.quarantined_at is None and row.lease_owner is None
    assert store.claim_stored_objects("too-soon", batch_size=1, lease_seconds=60) == []

    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE stored_objects SET next_attempt_at = "
                "clock_timestamp() - INTERVAL '1 second' "
                "WHERE id = :id"
            ),
            {"id": object_id},
        )
    second = _claim_gc(store, "retry-worker")
    assert store.record_failure(
        second,
        error_code="still-failing",
        error_message="quarantine now",
        policy=policy,
    )
    with postgres_database.session() as session:
        row = session.get(StoredObject, object_id)
        assert row is not None
        assert row.retry_count == 2
        assert row.quarantined_at is not None
        assert row.next_attempt_at is None and row.lease_owner is None
    assert store.claim_stored_objects("after-quarantine", batch_size=1, lease_seconds=60) == []


def test_fencing_tokens_cannot_move_backwards_in_the_database(
    postgres_database: Database,
) -> None:
    _gc_object(postgres_database)
    lease = _claim_gc(LeaseStore(postgres_database), "monotonic")
    with (
        pytest.raises(DBAPIError, match="fencing_token cannot decrease"),
        postgres_database.engine.begin() as connection,
    ):
        connection.execute(
            text("UPDATE stored_objects SET fencing_token = :token WHERE id = :id"),
            {"token": lease.fencing_token - 1, "id": lease.item_id},
        )


def test_project_lifecycle_claim_uses_project_as_authoritative_work_item(
    postgres_database: Database,
) -> None:
    guest_id = uuid4()
    project_id = uuid4()
    with postgres_database.session() as session:
        session.add(
            GuestSession(
                id=guest_id,
                token_digest=hashlib.sha256(b"phase5b-guest").hexdigest(),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        session.add(
            Project(
                id=project_id,
                storage_mode="temporary-cloud",
                title="Expired lease-only project",
                guest_session_id=guest_id,
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
    claims = LeaseStore(postgres_database).claim_projects(
        "lifecycle-worker", batch_size=5, lease_seconds=60
    )
    assert [(claim.item_id, claim.expected_state) for claim in claims] == [
        (project_id, "temporary-expired")
    ]


class FailNextConfirmStorage(LocalObjectStorage):
    fail_next = False

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated object confirmation outage")
        return super().confirm(staged)


def _chart() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": "Pending export",
        "xAxis": {"field": "time", "title": "Time", "unit": "s"},
        "yAxis": {"field": "signal", "title": "Signal", "unit": "V"},
        "series": [{"field": "signal", "label": "Signal", "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": "png",
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }


def _pending_write_intent(
    database: Database, storage: FailNextConfirmStorage
) -> StoredObjectWriteIntent:
    store = PostgresProjectStore(database, storage, 7_200)
    project_id = uuid4().hex
    job_id = uuid4().hex
    frame = pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})
    payload = frame.to_csv(index=False).encode()
    source = {
        "name": "pending.csv",
        "size": len(payload),
        "mediaType": "text/csv",
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
    }
    guest_digest = hashlib.sha256(uuid4().bytes).hexdigest()
    store.create_project(
        project_id=project_id,
        job_id=job_id,
        title="Pending write",
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
    project = store.get_project(project_id, touch=False)
    assert project is not None
    storage.fail_next = True
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
        intent = session.scalar(
            select(StoredObjectWriteIntent).where(StoredObjectWriteIntent.status == "pending")
        )
        assert intent is not None
        session.expunge(intent)
        return intent


def test_write_intent_is_the_authoritative_export_reconciliation_lease(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    intent = _pending_write_intent(postgres_database, FailNextConfirmStorage(tmp_path / "objects"))
    store = LeaseStore(postgres_database)
    claims = store.claim_write_intents("reconcile-worker", batch_size=5, lease_seconds=60)
    assert [(claim.item_id, claim.expected_state) for claim in claims] == [(intent.id, "pending")]
    assert store.claim_write_intents("other-worker", batch_size=5, lease_seconds=60) == []


def test_0006_upgrade_downgrade_reupgrade_and_schema_drift(
    postgres_database: Database,
) -> None:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))
    config = _alembic_config()
    command.downgrade(config, "0005_share_publication_exports")
    inspector = inspect(postgres_database.engine)
    assert "worker_leases" not in inspector.get_table_names()
    assert "lease_owner" not in {
        column["name"] for column in inspector.get_columns("stored_objects")
    }

    command.upgrade(config, "head")
    inspector = inspect(postgres_database.engine)
    assert "worker_leases" in inspector.get_table_names()
    assert {row.task for row in _worker_lease_rows(postgres_database)} == set(TASKS)
    command.check(config)


def _worker_lease_rows(database: Database) -> list[WorkerLease]:
    with database.session() as session:
        rows = list(session.scalars(select(WorkerLease).order_by(WorkerLease.task)))
        for row in rows:
            session.expunge(row)
        return rows
