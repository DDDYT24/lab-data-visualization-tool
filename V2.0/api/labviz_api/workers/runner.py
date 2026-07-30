"""Standalone worker runner; callbacks execute only after claim transactions commit."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread

from .leases import (
    METADATA_CLEANUP,
    ORPHAN_STAGING_INVENTORY,
    PENDING_RECONCILIATION,
    PROJECT_LIFECYCLE,
    STORED_OBJECT_GC,
    LeaseStore,
    RetryPolicy,
    TaskLease,
    WorkItemLease,
)

WorkItemHandler = Callable[[WorkItemLease, LeaseStore], None]
ScannerHandler = Callable[[TaskLease, LeaseStore], int]
DESTRUCTIVE_TASKS = {
    PROJECT_LIFECYCLE,
    STORED_OBJECT_GC,
    ORPHAN_STAGING_INVENTORY,
    METADATA_CLEANUP,
}


@dataclass(frozen=True)
class RunnerConfig:
    batch_size: int = 25
    lease_seconds: int = 60
    heartbeat_seconds: int = 20
    poll_seconds: int = 5
    retry_policy: RetryPolicy = RetryPolicy()
    destructive_maintenance: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.poll_seconds < 1:
            raise ValueError("batch_size and poll_seconds must be positive")
        if self.lease_seconds < 2:
            raise ValueError("lease_seconds must be at least 2")
        if not 0 < self.heartbeat_seconds < self.lease_seconds:
            raise ValueError("heartbeat_seconds must be positive and shorter than the lease")


class WorkerRunner:
    """Coordinate one task lease and its fenced work-item leases."""

    def __init__(
        self,
        *,
        task: str,
        owner: str,
        leases: LeaseStore,
        config: RunnerConfig,
        item_handler: WorkItemHandler | None = None,
        scanner_handler: ScannerHandler | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.task = task
        self.owner = owner
        self.leases = leases
        self.config = config
        self.item_handler = item_handler
        self.scanner_handler = scanner_handler
        self.logger = logger or logging.getLogger("labviz.worker")
        self._stop = Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        self.logger.info("worker-started", extra={"task": self.task, "owner": self.owner})
        try:
            while not self._stop.is_set():
                self.run_once()
                self._stop.wait(self.config.poll_seconds)
        finally:
            self.logger.info("worker-stopped", extra={"task": self.task, "owner": self.owner})

    def run_once(self) -> int:
        task_lease = self.leases.acquire_task(self.task, self.owner, self.config.lease_seconds)
        if task_lease is None:
            return 0
        processed = 0
        try:
            if self.task in DESTRUCTIVE_TASKS and not self.config.destructive_maintenance:
                self.logger.info(
                    "worker-destructive-task-disabled",
                    extra={"task": self.task, "owner": self.owner},
                )
                return 0
            if self.task in {ORPHAN_STAGING_INVENTORY, METADATA_CLEANUP}:
                if self.scanner_handler is not None:
                    processed = self._run_scanner(task_lease)
                return processed

            if self.item_handler is None:
                self.logger.info(
                    "worker-task-disabled",
                    extra={"task": self.task, "owner": self.owner},
                )
                return 0
            claims = self._claim_items()
            self.logger.info(
                "worker-batch-claimed",
                extra={"task": self.task, "owner": self.owner, "count": len(claims)},
            )
            for index, lease in enumerate(claims):
                if self._stop.is_set():
                    self._release_unprocessed(claims[index:])
                    break
                try:
                    # The claim session has committed before any handler or external I/O runs.
                    self._run_item_handler(task_lease, lease)
                except Exception as exc:
                    self.logger.warning(
                        "worker-item-failed",
                        extra={
                            "task": self.task,
                            "owner": self.owner,
                            "item_kind": lease.kind,
                            "fencing_token": lease.fencing_token,
                            "error_type": type(exc).__name__,
                        },
                    )
                    self.leases.record_failure(
                        lease,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                        policy=self.config.retry_policy,
                    )
                else:
                    self.leases.release_item(lease)
                processed += 1
                refreshed = self.leases.heartbeat_task(task_lease, self.config.lease_seconds)
                if refreshed is None:
                    self._release_unprocessed(claims[index + 1 :])
                    break
                task_lease = refreshed
            return processed
        finally:
            self.leases.release_task(task_lease)

    def _run_scanner(self, task_lease: TaskLease) -> int:
        assert self.scanner_handler is not None
        heartbeat_stop, heartbeat = self._start_heartbeat(task_lease, None)
        try:
            return self.scanner_handler(task_lease, self.leases)
        finally:
            heartbeat_stop.set()
            heartbeat.join()

    def _run_item_handler(self, task_lease: TaskLease, item_lease: WorkItemLease) -> None:
        assert self.item_handler is not None
        heartbeat_stop, heartbeat = self._start_heartbeat(task_lease, item_lease)
        try:
            self.item_handler(item_lease, self.leases)
        finally:
            heartbeat_stop.set()
            heartbeat.join()

    def _start_heartbeat(
        self, task_lease: TaskLease, item_lease: WorkItemLease | None
    ) -> tuple[Event, Thread]:
        stop = Event()

        def heartbeat_loop() -> None:
            while not stop.wait(self.config.heartbeat_seconds):
                try:
                    task_renewed = self.leases.heartbeat_task(task_lease, self.config.lease_seconds)
                    item_renewed = (
                        self.leases.heartbeat_item(item_lease, self.config.lease_seconds)
                        if item_lease is not None
                        else True
                    )
                except Exception as exc:
                    self.logger.warning(
                        "worker-heartbeat-failed",
                        extra={
                            "task": self.task,
                            "owner": self.owner,
                            "error_type": type(exc).__name__,
                        },
                    )
                    return
                if task_renewed is None or item_renewed is None:
                    self.logger.warning(
                        "worker-lease-lost",
                        extra={
                            "task": self.task,
                            "owner": self.owner,
                            "item_kind": item_lease.kind if item_lease else None,
                            "fencing_token": (
                                item_lease.fencing_token
                                if item_lease is not None
                                else task_lease.fencing_token
                            ),
                        },
                    )
                    return

        thread = Thread(
            target=heartbeat_loop,
            name=f"labviz-heartbeat-{self.task}",
            daemon=True,
        )
        thread.start()
        return stop, thread

    def _claim_items(self) -> list[WorkItemLease]:
        if self.task == PENDING_RECONCILIATION:
            return self.leases.claim_write_intents(
                self.owner,
                batch_size=self.config.batch_size,
                lease_seconds=self.config.lease_seconds,
            )
        if self.task == PROJECT_LIFECYCLE:
            return self.leases.claim_projects(
                self.owner,
                batch_size=self.config.batch_size,
                lease_seconds=self.config.lease_seconds,
            )
        if self.task == STORED_OBJECT_GC:
            return self.leases.claim_stored_objects(
                self.owner,
                batch_size=self.config.batch_size,
                lease_seconds=self.config.lease_seconds,
            )
        raise ValueError(f"Unsupported worker task: {self.task}")

    def _release_unprocessed(self, claims: list[WorkItemLease]) -> None:
        for lease in claims:
            self.leases.release_item(lease)
