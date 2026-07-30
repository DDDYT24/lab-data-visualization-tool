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
from .metadata_cleanup import inspect_metadata
from .orphan_staging import inventory_only
from .runner import RunnerConfig, ScannerHandler, WorkerRunner, WorkItemHandler


class JsonLogFormatter(logging.Formatter):
    """Emit bounded worker metadata without object paths, secrets, or content."""

    _fields = ("task", "owner", "count", "item_kind", "fencing_token", "error_type")

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


def _handlers(task: str) -> tuple[WorkItemHandler | None, ScannerHandler | None]:
    if task in {PENDING_RECONCILIATION, PROJECT_LIFECYCLE, STORED_OBJECT_GC}:
        # Phase 5B-1 intentionally does not claim business rows from the CLI.
        return None, None
    if task == ORPHAN_STAGING_INVENTORY:
        return None, inventory_only
    if task == METADATA_CLEANUP:
        return None, inspect_metadata
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
    )
    item_handler, scanner_handler = _handlers(args.task)
    runner = WorkerRunner(
        task=args.task,
        owner=f"worker-{uuid4()}",
        leases=LeaseStore(database),
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
