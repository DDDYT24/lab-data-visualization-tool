"""Command-line entry point for independent PostgreSQL maintenance workers."""

from __future__ import annotations

import argparse
import json
import logging
import signal
from collections.abc import Sequence
from datetime import UTC, datetime
from types import FrameType
from uuid import uuid4

from labviz_api.config import Settings
from labviz_api.db.session import Database
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.storage import LocalObjectStorage

from .garbage_collection import StoredObjectGarbageCollector
from .leases import (
    METADATA_CLEANUP,
    ORPHAN_STAGING_INVENTORY,
    PENDING_RECONCILIATION,
    PROJECT_LIFECYCLE,
    STORED_OBJECT_GC,
    TASKS,
    LeaseStore,
    RetryPolicy,
)
from .lifecycle import ProjectLifecycleHandler
from .metadata_cleanup import MetadataCleanupHandler
from .orphan_staging import OrphanStagingHandler
from .reconciliation import PendingObjectReconciler
from .runner import RunnerConfig, ScannerHandler, WorkerRunner, WorkItemHandler
from .safety import MaintenanceSafety


class JsonLogFormatter(logging.Formatter):
    """Emit bounded worker metadata without object paths, secrets, or content."""

    _fields = (
        "task",
        "owner",
        "count",
        "item_kind",
        "fencing_token",
        "reason",
        "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in self._fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _logger() -> logging.Logger:
    logger = logging.getLogger("labviz.worker")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _handlers(
    task: str,
    *,
    project_store: PostgresProjectStore,
    safety: MaintenanceSafety,
    retry_policy: RetryPolicy,
    orphan_grace_seconds: int,
    batch_size: int,
) -> tuple[WorkItemHandler | None, ScannerHandler | None]:
    if task == PENDING_RECONCILIATION:
        return PendingObjectReconciler(project_store), None
    if task == PROJECT_LIFECYCLE:
        return ProjectLifecycleHandler(safety), None
    if task == STORED_OBJECT_GC:
        return StoredObjectGarbageCollector(project_store.storage, safety, retry_policy), None
    if task == ORPHAN_STAGING_INVENTORY:
        return None, OrphanStagingHandler(
            project_store.storage,
            safety,
            retry_policy,
            grace_seconds=orphan_grace_seconds,
            batch_size=batch_size,
        )
    if task == METADATA_CLEANUP:
        return None, MetadataCleanupHandler(safety, batch_size=batch_size)
    raise ValueError(f"Unsupported worker task: {task}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one LabViz PostgreSQL maintenance worker.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    settings = Settings.from_env()
    if settings.persistence_backend != "postgresql" or settings.postgres_url is None:
        raise SystemExit("Workers require LABVIZ_PERSISTENCE_BACKEND=postgresql.")

    database = Database(settings.postgres_url, echo=settings.postgres_echo)
    storage = LocalObjectStorage(settings.object_storage_root)
    project_store = PostgresProjectStore(
        database,
        storage,
        settings.project_ttl_seconds,
        guest_session_ttl_seconds=settings.session_ttl_seconds,
    )
    retry_policy = RetryPolicy(
        max_retries=settings.worker_max_retries,
        base_seconds=settings.worker_backoff_base_seconds,
        max_seconds=settings.worker_backoff_max_seconds,
    )
    config = RunnerConfig(
        batch_size=settings.worker_batch_size,
        lease_seconds=settings.worker_lease_seconds,
        heartbeat_seconds=settings.worker_heartbeat_seconds,
        poll_seconds=settings.worker_poll_seconds,
        retry_policy=retry_policy,
        destructive_maintenance=settings.worker_destructive_maintenance,
        dry_run=settings.worker_dry_run,
        delete_enabled=settings.worker_delete_enabled,
    )
    safety = MaintenanceSafety(
        dry_run=settings.worker_dry_run,
        delete_enabled=settings.worker_delete_enabled,
    )
    item_handler, scanner_handler = _handlers(
        args.task,
        project_store=project_store,
        safety=safety,
        retry_policy=retry_policy,
        orphan_grace_seconds=settings.worker_orphan_staging_grace_seconds,
        batch_size=settings.worker_batch_size,
    )
    runner = WorkerRunner(
        task=args.task,
        owner=f"worker-{uuid4()}",
        leases=LeaseStore(
            database,
            gc_orphan_age_seconds=settings.worker_gc_orphan_age_seconds,
        ),
        config=config,
        item_handler=item_handler,
        scanner_handler=scanner_handler,
        logger=_logger(),
    )

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        runner.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        if args.once:
            runner.run_once()
        else:
            runner.run_forever()
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
