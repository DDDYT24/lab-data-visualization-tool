from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select, text

from labviz_api.db.models import AuthRateLimitBucket, AuthRequest
from labviz_api.db.session import Database
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.storage import LocalObjectStorage
from labviz_api.workers.leases import METADATA_CLEANUP, LeaseStore
from labviz_api.workers.metadata_cleanup import MetadataCleanupHandler
from labviz_api.workers.safety import MaintenanceSafety

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
        pytest.fail("PostgreSQL is required for Phase 6B atomic limiter tests.")
    command.upgrade(alembic_config(), "head")
    try:
        yield database
    finally:
        command.upgrade(alembic_config(), "head")
        database.dispose()


@pytest.fixture(autouse=True)
def empty_limiter_state(postgres_database: Database) -> Iterator[None]:
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))
    yield
    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets, auth_requests"))


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def postgres_store(database: Database, root: Path) -> PostgresProjectStore:
    return PostgresProjectStore(database, LocalObjectStorage(root), 7_200)


def test_populated_0008_upgrade_downgrade_guard_and_reupgrade(
    postgres_database: Database,
) -> None:
    config = alembic_config()
    command.downgrade(config, "0008_phase5b3_storage_inventory")
    assert "auth_rate_limit_buckets" not in inspect(postgres_database.engine).get_table_names()

    client_digest = digest("migration-client")
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO auth_requests (id, client_key, email, requested_at)
                VALUES
                    (:first_id, :client_digest, 'migration@example.test', statement_timestamp()),
                    (
                        :second_id, 'legacy-raw-client',
                        'migration@example.test', statement_timestamp()
                    )
                """
            ),
            {
                "first_id": uuid4(),
                "second_id": uuid4(),
                "client_digest": client_digest,
            },
        )

    command.upgrade(config, "head")
    with postgres_database.engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT scope, identity_key, request_count
                FROM auth_rate_limit_buckets
                ORDER BY scope, identity_key
                """
            )
        ).all()
    assert [tuple(row) for row in rows] == [
        ("client", client_digest, 1),
        ("email", "migration@example.test", 2),
    ]

    with pytest.raises(RuntimeError, match="rate-limit buckets exist"):
        command.downgrade(config, "0008_phase5b3_storage_inventory")

    with postgres_database.engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE auth_rate_limit_buckets"))
    command.downgrade(config, "0008_phase5b3_storage_inventory")
    assert "auth_rate_limit_buckets" not in inspect(postgres_database.engine).get_table_names()
    command.upgrade(config, "head")
    command.check(config)


def test_postgres_concurrency_admits_exactly_the_client_limit(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = postgres_store(postgres_database, tmp_path / "objects")
    client_key = digest(f"concurrent-client-{uuid4()}")
    email = f"concurrent-{uuid4().hex}@example.test"
    attempts = 20
    barrier = Barrier(attempts)

    def attempt() -> bool:
        barrier.wait(timeout=10)
        return store.allow_auth_request(
            client_key=client_key,
            email=email,
            client_limit=5,
            email_limit=20,
            window_seconds=3_600,
        )

    with ThreadPoolExecutor(max_workers=attempts) as executor:
        admitted = list(executor.map(lambda _index: attempt(), range(attempts)))

    assert admitted.count(True) == 5
    assert admitted.count(False) == 15
    with postgres_database.session() as session:
        buckets = list(
            session.scalars(
                select(AuthRateLimitBucket).where(
                    AuthRateLimitBucket.identity_key.in_((client_key, email))
                )
            )
        )
        request_count = session.scalar(
            select(func.count())
            .select_from(AuthRequest)
            .where(AuthRequest.client_key == client_key)
        )
    assert {(bucket.scope, bucket.request_count) for bucket in buckets} == {
        ("client", 5),
        ("email", 5),
    }
    assert request_count == 5


def test_count_then_insert_negative_probe_demonstrates_the_removed_race(
    postgres_database: Database,
) -> None:
    client_key = digest(f"unsafe-client-{uuid4()}")
    attempts = 8
    after_count = Barrier(attempts)

    def unsafe_attempt(index: int) -> bool:
        with postgres_database.engine.begin() as connection:
            observed = int(
                connection.scalar(
                    text("SELECT count(*) FROM auth_requests WHERE client_key = :client_key"),
                    {"client_key": client_key},
                )
                or 0
            )
            after_count.wait(timeout=10)
            if observed >= 1:
                return False
            connection.execute(
                text(
                    """
                    INSERT INTO auth_requests (id, client_key, email, requested_at)
                    VALUES (:id, :client_key, :email, statement_timestamp())
                    """
                ),
                {
                    "id": uuid4(),
                    "client_key": client_key,
                    "email": f"unsafe-{index}@example.test",
                },
            )
            return True

    with ThreadPoolExecutor(max_workers=attempts) as executor:
        unsafe_results = list(executor.map(unsafe_attempt, range(attempts)))

    assert unsafe_results.count(True) == attempts
    assert attempts > 1


def test_rejection_does_not_partially_consume_the_other_dimension(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = postgres_store(postgres_database, tmp_path / "objects")
    limited_client = digest(f"limited-{uuid4()}")
    alternate_client = digest(f"alternate-{uuid4()}")
    first_email = f"first-{uuid4().hex}@example.test"
    untouched_email = f"untouched-{uuid4().hex}@example.test"

    assert store.allow_auth_request(
        client_key=limited_client,
        email=first_email,
        client_limit=1,
        email_limit=1,
    )
    assert not store.allow_auth_request(
        client_key=limited_client,
        email=untouched_email,
        client_limit=1,
        email_limit=1,
    )
    assert store.allow_auth_request(
        client_key=alternate_client,
        email=untouched_email,
        client_limit=1,
        email_limit=1,
    )

    with postgres_database.session() as session:
        untouched = session.scalar(
            select(AuthRateLimitBucket).where(
                AuthRateLimitBucket.scope == "email",
                AuthRateLimitBucket.identity_key == untouched_email,
            )
        )
    assert untouched is not None
    assert untouched.request_count == 1


def test_postgres_window_boundary_and_restart_use_durable_database_state(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = postgres_store(postgres_database, tmp_path / "objects")
    client_key = digest(f"boundary-{uuid4()}")
    email = f"boundary-{uuid4().hex}@example.test"

    with postgres_database.engine.connect() as connection:
        phase = float(
            connection.scalar(text("SELECT mod(extract(epoch FROM clock_timestamp()), 2)"))
        )
        connection.execute(text("SELECT pg_sleep(:seconds)"), {"seconds": 2.1 - phase})
    assert store.allow_auth_request(
        client_key=client_key,
        email=email,
        client_limit=1,
        email_limit=1,
        window_seconds=2,
    )

    restarted_database = Database(POSTGRES_URL)
    try:
        restarted = postgres_store(restarted_database, tmp_path / "restart-objects")
        assert not restarted.allow_auth_request(
            client_key=client_key,
            email=email,
            client_limit=1,
            email_limit=1,
            window_seconds=2,
        )
        with restarted_database.engine.connect() as connection:
            connection.execute(text("SELECT pg_sleep(2.1)"))
        assert restarted.allow_auth_request(
            client_key=client_key,
            email=email,
            client_limit=1,
            email_limit=1,
            window_seconds=2,
        )
    finally:
        restarted_database.dispose()


def test_sqlite_reference_path_is_atomic_across_connections(tmp_path: Path) -> None:
    repository = SqliteReferenceRepository(tmp_path / "sqlite-limiter.db", 7_200)
    client_key = digest(f"sqlite-{uuid4()}")
    email = f"sqlite-{uuid4().hex}@example.test"
    attempts = 12
    barrier = Barrier(attempts)

    def attempt() -> bool:
        barrier.wait(timeout=10)
        return repository.allow_auth_request(
            client_key=client_key,
            email=email,
            client_limit=3,
            email_limit=10,
            window_seconds=3_600,
        )

    with ThreadPoolExecutor(max_workers=attempts) as executor:
        admitted = list(executor.map(lambda _index: attempt(), range(attempts)))

    assert admitted.count(True) == 3
    assert admitted.count(False) == 9
    with repository._connect() as connection:
        counts = connection.execute(
            """
            SELECT scope, request_count FROM auth_rate_limit_buckets
            WHERE identity_key IN (?, ?) ORDER BY scope
            """,
            (client_key, email),
        ).fetchall()
    assert [(row["scope"], row["request_count"]) for row in counts] == [
        ("client", 3),
        ("email", 3),
    ]


def test_leased_metadata_cleanup_deletes_only_expired_buckets(
    postgres_database: Database,
) -> None:
    expired_id = uuid4()
    active_id = uuid4()
    with postgres_database.session() as session:
        now = session.scalar(select(func.clock_timestamp()))
        assert isinstance(now, datetime)
        session.add_all(
            [
                AuthRateLimitBucket(
                    id=expired_id,
                    scope="client",
                    identity_key=digest(f"expired-{uuid4()}"),
                    window_started_at=now - timedelta(hours=2),
                    expires_at=now - timedelta(hours=1),
                    request_count=1,
                    updated_at=now - timedelta(hours=2),
                ),
                AuthRateLimitBucket(
                    id=active_id,
                    scope="client",
                    identity_key=digest(f"active-{uuid4()}"),
                    window_started_at=now - timedelta(minutes=1),
                    expires_at=now + timedelta(minutes=59),
                    request_count=1,
                    updated_at=now,
                ),
            ]
        )

    leases = LeaseStore(postgres_database)
    task = leases.acquire_task(METADATA_CLEANUP, f"limiter-cleanup-{uuid4()}", 60)
    assert task is not None
    cleaned = MetadataCleanupHandler(
        MaintenanceSafety(dry_run=False, delete_enabled=True),
        batch_size=1_000,
    )(task, leases)
    assert cleaned >= 1
    assert leases.release_task(task)

    with postgres_database.session() as session:
        assert session.get(AuthRateLimitBucket, expired_id) is None
        assert session.get(AuthRateLimitBucket, active_id) is not None
