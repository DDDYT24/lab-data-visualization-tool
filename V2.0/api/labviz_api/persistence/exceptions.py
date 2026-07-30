"""Stable domain failures independent of database and object-store vendors."""


class PersistenceError(RuntimeError):
    code = "persistence-error"


class PersistenceNotFound(PersistenceError):
    code = "persistence-not-found"


class PersistenceConflict(PersistenceError):
    code = "persistence-conflict"


class PersistenceUnavailable(PersistenceError):
    code = "persistence-unavailable"


class ObjectConfirmationPending(PersistenceUnavailable):
    code = "object-confirmation-pending"


class FeatureNotMigrated(PersistenceError):
    code = "feature-not-migrated"
