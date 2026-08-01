"""Single authoritative StoredObject reachability definition for purge and GC."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from labviz_api.db.models import (
    DatasetVersion,
    ExportJobRecord,
    PublicationExport,
    SourceFile,
    StoredObject,
    StoredObjectWriteIntent,
)


def object_is_referenced(object_id: Any = StoredObject.id) -> ColumnElement[bool]:
    """Return the SQL predicate used by every physical-deletion decision."""

    return or_(
        exists(select(1).where(SourceFile.stored_object_id == object_id)),
        exists(select(1).where(DatasetVersion.stored_object_id == object_id)),
        exists(select(1).where(PublicationExport.stored_object_id == object_id)),
        exists(
            select(1).where(
                StoredObjectWriteIntent.stored_object_id == object_id,
                or_(
                    StoredObjectWriteIntent.status == "pending",
                    StoredObjectWriteIntent.quarantined_at.is_not(None),
                ),
            )
        ),
        exists(select(1).where(ExportJobRecord.pending_stored_object_id == object_id)),
    )


def has_object_references(session: Session, stored_object_id: UUID) -> bool:
    """Evaluate authoritative reachability for a locked StoredObject row."""

    return bool(session.scalar(select(object_is_referenced(stored_object_id))))


def collect_project_object_ids(session: Session, project_id: UUID) -> set[UUID]:
    """Collect every object whose last project-scoped reference purge may remove."""

    values: set[UUID | None] = {
        *session.scalars(
            select(SourceFile.stored_object_id).where(
                SourceFile.project_id == project_id,
                SourceFile.stored_object_id.is_not(None),
            )
        ),
        *session.scalars(
            select(DatasetVersion.stored_object_id).where(DatasetVersion.project_id == project_id)
        ),
        *session.scalars(
            select(PublicationExport.stored_object_id).where(
                PublicationExport.project_id == project_id
            )
        ),
        *session.scalars(
            select(StoredObjectWriteIntent.stored_object_id).where(
                StoredObjectWriteIntent.project_id == project_id
            )
        ),
        *session.scalars(
            select(ExportJobRecord.pending_stored_object_id).where(
                ExportJobRecord.project_id == project_id,
                ExportJobRecord.pending_stored_object_id.is_not(None),
            )
        ),
    }
    return {value for value in values if value is not None}


def finalize_purged_project_objects(
    session: Session,
    stored_objects: list[StoredObject],
    now: datetime,
) -> None:
    """Make orphaned pending writes GC-able after their project FKs are gone."""

    session.flush()
    for stored in stored_objects:
        stored.gc_candidate_at = now
        if stored.status == "pending" and not has_object_references(session, stored.id):
            # The staging key becomes a two-pass orphan-inventory candidate. The
            # possibly confirmed final key is removed through the fenced GC state machine.
            stored.status = "deleting"
            stored.staging_key = None
            stored.updated_at = now


def staging_key_has_database_owner(session: Session, staging_key: str) -> bool:
    """Return whether a staging key belongs to a persisted active object operation."""

    return bool(
        session.scalar(
            select(
                exists(
                    select(1).where(
                        StoredObject.staging_key == staging_key,
                        StoredObject.status == "pending",
                    )
                )
            )
        )
    )
