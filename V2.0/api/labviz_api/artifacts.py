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
from dataclasses import dataclass

from .storage.base import ObjectAlreadyExists as StorageObjectAlreadyExists
from .storage.base import ObjectIntegrityError, ObjectStorage

ARTIFACT_PREFIX = "artifacts"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
