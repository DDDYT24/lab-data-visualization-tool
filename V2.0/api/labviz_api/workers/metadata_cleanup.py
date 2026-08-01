"""Database-time cleanup of bounded authentication and request metadata."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import delete, exists, func, or_, select

from labviz_api.db.models import (
    AuthChallenge,
    AuthRequest,
    AuthSession,
    GuestSession,
    IdempotencyRecord,
    Project,
)

from .leases import LeaseStore, TaskLease
from .safety import MaintenanceSafety


class MetadataCleanupHandler:
    def __init__(self, safety: MaintenanceSafety, *, batch_size: int = 25) -> None:
        if batch_size < 1:
            raise ValueError("metadata cleanup batch size must be positive")
        self.safety = safety
        self.batch_size = batch_size

    def __call__(self, lease: TaskLease, leases: LeaseStore) -> int:
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return 0
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            predicates = (
                (AuthChallenge, AuthChallenge.id, AuthChallenge.expires_at <= now),
                (AuthSession, AuthSession.token_digest, AuthSession.expires_at <= now),
                (AuthRequest, AuthRequest.id, AuthRequest.requested_at < now - timedelta(hours=1)),
                (
                    IdempotencyRecord,
                    IdempotencyRecord.id,
                    IdempotencyRecord.expires_at.is_not(None)
                    & (IdempotencyRecord.expires_at <= now),
                ),
            )
            guest_predicate = or_(
                GuestSession.expires_at <= now, GuestSession.status == "revoked"
            ) & ~exists(select(1).where(Project.guest_session_id == GuestSession.id))
            selections = (*predicates, (GuestSession, GuestSession.id, guest_predicate))
            remaining = self.batch_size
            selected: list[tuple[Any, Any, list[Any]]] = []
            for model, key_column, predicate in selections:
                if remaining == 0:
                    break
                keys = list(
                    session.scalars(
                        select(key_column).select_from(model).where(predicate).limit(remaining)
                    )
                )
                selected.append((model, key_column, keys))
                remaining -= len(keys)
            count = self.batch_size - remaining
            if not self.safety.may_delete:
                return count
            for selected_model, selected_key, keys in selected:
                if keys:
                    session.execute(delete(selected_model).where(selected_key.in_(keys)))
            return count
