"""Content-addressed artifact store for rendered figures and exports.

Rendered PNG, SVG and PDF bytes were stored as SQLite BLOBs. That grows the database
with every saved figure, needs twice the disk during ``VACUUM``, forces a backup to copy
the whole data directory as one unit, and keeps two copies of identical renders. Storing
them beside the database under their own SHA-256 keeps the database small, lets identical
renders share one file, and makes every read verifiable.

The store reuses the existing ``ObjectStorage`` provider, so the local filesystem backend's
atomic same-directory replacement, fsync and digest checks apply unchanged.
"""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .storage.base import ObjectAlreadyExists as StorageObjectAlreadyExists
from .storage.base import ObjectIntegrityError, ObjectStorage
from .storage.local import LocalObjectStorage

ARTIFACT_PREFIX = "artifacts"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
#: Every artifact a local profile stores, so cleanup never needs to list the filesystem.
REGISTRY_TABLE = "local_artifacts"
#: Each (table, column) that holds an artifact digest. Anything registered and absent
#: from all of them is unreferenced.
REFERENCES = (
    ("exports", "payload_sha256"),
    ("figure_snapshots", "sha256"),
    ("figure_snapshots", "thumbnail_sha256"),
    ("research_charts", "svg_sha256"),
    ("research_charts", "png_sha256"),
    ("research_charts", "pdf_sha256"),
    ("research_figure_artifacts", "svg_sha256"),
    ("research_figure_artifacts", "png_sha256"),
    ("research_figure_artifacts", "pdf_sha256"),
    ("research_outputs", "sha256"),
    ("dataset_versions", "content_sha256"),
)
MEDIA_TYPES = {
    "svg": "image/svg+xml",
    "png": "image/png",
    "pdf": "application/pdf",
    "zip": "application/zip",
    "parquet": "application/vnd.apache.parquet",
}


@dataclass(frozen=True)
class StoredArtifact:
    """What the database records instead of the bytes themselves."""

    sha256: str
    byte_size: int


def artifact_key(sha256: str) -> str:
    """Return the object key for a digest, fanned out so no directory grows unbounded."""
    digest = sha256.lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError("An artifact digest must be 64 lower-case hexadecimal characters.")
    return f"{ARTIFACT_PREFIX}/{digest[:2]}/{digest}"


class ArtifactStore:
    """Write-once artifact bytes addressed by their own SHA-256."""

    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    def put(self, payload: bytes, *, media_type: str) -> StoredArtifact:
        digest = hashlib.sha256(payload).hexdigest()
        key = artifact_key(digest)
        try:
            self.storage.put(
                key,
                io.BytesIO(payload),
                expected_sha256=digest,
                metadata={"labviz-media-type": media_type},
            )
        except StorageObjectAlreadyExists:
            # Under content addressing an existing key means the identical bytes are
            # already stored, so a collision is a deduplicated write rather than a
            # conflict. Confirm the stored length before trusting it.
            existing = self.storage.head(key)
            if existing is None or existing.size_bytes != len(payload):
                raise ObjectIntegrityError("A stored artifact disagrees with its digest.") from None
        return StoredArtifact(sha256=digest, byte_size=len(payload))

    def read(self, sha256: str) -> bytes:
        key = artifact_key(sha256)
        with self.storage.open(key) as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != sha256.lower():
            raise ObjectIntegrityError("A stored artifact does not match its recorded digest.")
        return payload

    def exists(self, sha256: str) -> bool:
        return self.storage.exists(artifact_key(sha256))

    def delete(self, sha256: str) -> bool:
        """Remove one artifact. The caller owns the decision that nothing references it."""
        return self.storage.delete(artifact_key(sha256))


def local_artifact_store(database_path: Path) -> ArtifactStore:
    """Artifacts of a local profile live in the ``objects`` directory beside its database.

    Installed, portable and desktop profiles already back up exactly ``labviz.db`` and
    ``objects/``, so placing artifacts there keeps every backup complete.
    """
    return ArtifactStore(LocalObjectStorage(database_path.parent / "objects"))


def remember(
    connection: sqlite3.Connection, store: ArtifactStore, payload: bytes, *, media_type: str
) -> str:
    """Store bytes for a row the caller is about to write in the same transaction.

    Registering first takes SQLite's write lock before the file is written. ``sweep``
    runs under the same lock, so it can never delete a deduplicated file between this
    call and the commit of the row that references it.
    """
    digest = hashlib.sha256(payload).hexdigest()
    connection.execute(
        f"INSERT OR IGNORE INTO {REGISTRY_TABLE} (sha256, media_type, byte_size, created_at) "
        "VALUES (?, ?, ?, ?)",
        (digest, media_type, len(payload), datetime.now(UTC).isoformat()),
    )
    store.put(payload, media_type=media_type)
    return digest


def sweep(database_path: Path, store: ArtifactStore) -> int:
    """Delete registered artifacts that no row references, in a transaction of its own.

    The transaction only removes registry rows, so if it fails to commit after files were
    deleted, no reference comes back with it; the next sweep drops the leftover rows.
    A profile without the registry (one that has not been migrated) is left untouched.
    """
    with closing(sqlite3.connect(database_path, timeout=30, isolation_level=None)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if REGISTRY_TABLE not in tables:
                connection.execute("ROLLBACK")
                return 0
            referenced = " UNION ALL ".join(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
                for table, column in REFERENCES
                if table in tables
            )
            rows = connection.execute(
                f"SELECT sha256 FROM {REGISTRY_TABLE} WHERE sha256 NOT IN ({referenced})"
            ).fetchall()
            for (digest,) in rows:
                store.delete(digest)
                connection.execute(f"DELETE FROM {REGISTRY_TABLE} WHERE sha256 = ?", (digest,))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return len(rows)
