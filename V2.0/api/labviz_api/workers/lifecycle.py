"""Fenced project expiry and saved-project purge execution."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import func, select

from labviz_api.db.models import ProjectLifecycleEvent, StoredObject

from .leases import LeaseStore, WorkItemLease
from .references import collect_project_object_ids, finalize_purged_project_objects
from .safety import MaintenanceSafety


class ProjectLifecycleHandler:
    def __init__(self, safety: MaintenanceSafety) -> None:
        self.safety = safety

    def __call__(self, lease: WorkItemLease, leases: LeaseStore) -> None:
        if not self.safety.may_delete:
            return
        with leases.database.session() as session:
            project = leases.lock_owned_item(session, lease)
            if project is None:
                return
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            object_ids = collect_project_object_ids(session, project.id)
            stored_objects: list[StoredObject] = []
            if object_ids:
                stored_objects = list(
                    session.scalars(
                        select(StoredObject)
                        .where(StoredObject.id.in_(object_ids))
                        .with_for_update()
                    )
                )
                for stored in stored_objects:
                    stored.gc_candidate_at = now
            session.add(
                ProjectLifecycleEvent(
                    project_id=project.id,
                    project_uuid_snapshot=project.id,
                    actor_user_id=project.owner_user_id,
                    event_type="purge",
                    details={"worker": True, "objectCandidateCount": len(object_ids)},
                    created_at=now,
                )
            )
            session.delete(project)
            finalize_purged_project_objects(session, stored_objects, now)
