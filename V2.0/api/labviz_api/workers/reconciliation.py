"""Recover persisted object-confirmation and finalization work after process failure."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import func, select

from labviz_api.db.models import (
    DatasetVersion,
    ExportJobRecord,
    ProcessingRun,
    PublicationExport,
    StoredObject,
)
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.storage import ObjectInfo, StagedObject
from labviz_api.storage.local import ObjectAlreadyExists, ObjectIntegrityError

from .leases import LeaseStore, WorkItemLease
from .safety import PermanentWorkerFailure


class PendingObjectReconciler:
    """Confirm bytes outside transactions, then finalize through a fenced transaction."""

    def __init__(self, project_store: PostgresProjectStore) -> None:
        self.project_store = project_store
        self.database = project_store.database
        self.storage = project_store.storage

    def __call__(self, lease: WorkItemLease, leases: LeaseStore) -> None:
        if lease.kind == "write-intent":
            self._reconcile_export(lease, leases)
            return
        if lease.kind == "pending-object":
            self._reconcile_dataset(lease, leases)
            return
        raise ValueError(f"Unsupported reconciliation item: {lease.kind}")

    def _reconcile_export(self, lease: WorkItemLease, leases: LeaseStore) -> None:
        with self.database.session() as session:
            intent = leases.lock_owned_item(session, lease)
            if intent is None:
                return
            stored = session.get(StoredObject, intent.stored_object_id)
            if stored is None:
                raise PermanentWorkerFailure("WriteIntent StoredObject is missing.")
            staged = self._staged(stored)
        info = self._confirm(staged)
        with self.database.session() as session:
            intent = leases.lock_owned_item(session, lease)
            if intent is None:
                return
            stored = session.scalar(
                select(StoredObject)
                .where(StoredObject.id == intent.stored_object_id)
                .with_for_update()
            )
            if stored is None or not self._matches(stored, info):
                raise PermanentWorkerFailure("Confirmed export metadata no longer matches SQL.")
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            publication = session.get(PublicationExport, intent.export_job_id)
            if publication is None:
                intent.lease_owner = None
                intent.lease_until = None
                job = self.project_store._complete_export_intent(session, stored, now)  # noqa: SLF001
                if job is None:
                    raise PermanentWorkerFailure("Pending export WriteIntent disappeared.")
            else:
                job = session.scalar(
                    select(ExportJobRecord)
                    .where(ExportJobRecord.id == intent.export_job_id)
                    .with_for_update()
                )
                if job is None:
                    raise PermanentWorkerFailure("PublicationExport has no ExportJob.")
                run = (
                    session.get(ProcessingRun, job.current_processing_run_id)
                    if job.current_processing_run_id is not None
                    else None
                )
                stored.status = "available"
                stored.staging_key = None
                stored.updated_at = now
                intent.status = "completed"
                intent.completed_at = now
                intent.lease_owner = None
                intent.lease_until = None
                job.pending_stored_object_id = None
                job.status = "ready"
                job.finished_at = now
                job.updated_at = now
                if run is not None:
                    run.status = "succeeded"
                    run.finished_at = now
                self.project_store._bind_export_to_shares(session, publication, now)  # noqa: SLF001

    def _reconcile_dataset(self, lease: WorkItemLease, leases: LeaseStore) -> None:
        with self.database.session() as session:
            stored = leases.lock_owned_item(session, lease)
            if stored is None:
                return
            staged = self._staged(stored)
        info = self._confirm(staged)
        with self.database.session() as session:
            stored = leases.lock_owned_item(session, lease)
            if stored is None:
                return
            if not self._matches(stored, info):
                raise PermanentWorkerFailure("Confirmed dataset metadata no longer matches SQL.")
            version = session.scalar(
                select(DatasetVersion).where(DatasetVersion.stored_object_id == stored.id)
            )
            if version is None:
                raise PermanentWorkerFailure("Pending dataset object has no DatasetVersion.")
            runs = list(
                session.scalars(
                    select(ProcessingRun)
                    .where(
                        ProcessingRun.project_id == version.project_id,
                        ProcessingRun.status.in_(("queued", "running")),
                    )
                    .with_for_update()
                )
            )
            run = next(
                (
                    item
                    for item in runs
                    if item.parameters.get("pendingDatasetVersionId") == version.id.hex
                ),
                None,
            )
            if run is None:
                raise PermanentWorkerFailure("Pending dataset object has no active ProcessingRun.")
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            stored.status = "available"
            stored.staging_key = None
            stored.updated_at = now
            stored.lease_owner = None
            stored.lease_until = None
            run.output_dataset_version_id = version.id
            run.status = "succeeded"
            run.finished_at = now
            run.error_code = None
            run.error_message = None
            if run.operation == "parse":
                run.parameters = {
                    **run.parameters,
                    "apiJob": {
                        "stage": "ready",
                        "progress": 100,
                        "message": "Your data is ready to inspect.",
                    },
                }

    def _confirm(self, staged: StagedObject) -> ObjectInfo:
        try:
            return self.storage.confirm(staged)
        except (
            FileNotFoundError,
            PermissionError,
            ObjectAlreadyExists,
            ObjectIntegrityError,
        ) as exc:
            raise PermanentWorkerFailure(str(exc)) from exc

    @staticmethod
    def _staged(stored: StoredObject) -> StagedObject:
        if stored.staging_key is None:
            if stored.status == "available":
                return StagedObject(
                    key=stored.object_key,
                    staging_key=".staging/already-confirmed.part",
                    size_bytes=stored.size_bytes,
                    sha256=stored.sha256,
                )
            raise PermanentWorkerFailure("Pending StoredObject has no staging key.")
        return StagedObject(
            key=stored.object_key,
            staging_key=stored.staging_key,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
        )

    @staticmethod
    def _matches(stored: StoredObject, info: ObjectInfo) -> bool:
        return (
            stored.object_key == info.key
            and stored.sha256 == info.sha256
            and stored.size_bytes == info.size_bytes
        )
