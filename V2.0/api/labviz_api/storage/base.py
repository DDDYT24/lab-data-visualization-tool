"""Provider-neutral object storage contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class ObjectInfo:
    """Integrity metadata returned after an object is committed."""

    key: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StagedObject:
    """Recoverable object write prepared before its database transaction commits."""

    key: str
    staging_key: str
    size_bytes: int
    sha256: str


class ObjectStorage(Protocol):
    """Minimum interface required by the dataset and export pipelines."""

    def put(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> ObjectInfo:
        """Atomically commit a streamed object after integrity validation."""
        ...

    def open(self, key: str) -> BinaryIO:
        """Open an existing object for binary streaming."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether an exact object key exists."""
        ...

    def head(self, key: str) -> ObjectInfo | None:
        """Return integrity metadata without returning object bytes."""
        ...

    def list_staged(self) -> list[ObjectInfo]:
        """Inventory provider-owned staging objects without deleting them."""
        ...

    def delete(self, key: str) -> bool:
        """Delete one exact object key and report whether it existed."""
        ...

    def stage(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
    ) -> StagedObject:
        """Write and validate recoverable bytes without publishing the final key."""
        ...

    def open_staged(self, staged: StagedObject) -> BinaryIO:
        """Open staged bytes for format validation before database commit."""
        ...

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        """Publish a staged write; repeated confirmation must be safe."""
        ...

    def discard(self, staged: StagedObject) -> bool:
        """Remove uncommitted staged bytes during transaction compensation."""
        ...
