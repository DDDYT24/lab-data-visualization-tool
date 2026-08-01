"""Provider-neutral object storage contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import BinaryIO, Protocol

DEFAULT_INVENTORY_PAGE_SIZE = 100
MAX_INVENTORY_PAGE_SIZE = 1_000


class InvalidObjectKey(ValueError):
    """Raised when a key could escape the configured provider scope."""


class InvalidStorageCursor(ValueError):
    """Raised when an inventory cursor is invalid for the current provider scope."""


class ObjectAlreadyExists(FileExistsError):
    """Raised when an immutable object key would be overwritten."""


class ObjectIntegrityError(ValueError):
    """Raised when provider bytes or metadata fail application integrity checks."""


@dataclass(frozen=True)
class ObjectInfo:
    """Integrity metadata returned after an object is committed."""

    key: str
    size_bytes: int
    sha256: str
    last_modified: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    etag: str | None = None
    version_id: str | None = None


@dataclass(frozen=True)
class StagedObject:
    """Recoverable object write prepared before its database transaction commits."""

    key: str
    staging_key: str
    size_bytes: int
    sha256: str
    last_modified: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StagingPage:
    """One bounded provider inventory page with an opaque continuation cursor."""

    items: tuple[ObjectInfo, ...]
    next_cursor: str | None
    has_more: bool


class ObjectStorage(Protocol):
    """Minimum interface required by the dataset and export pipelines."""

    @property
    def backend_name(self) -> str:
        """Return the stable database identifier for this provider implementation."""
        ...

    @property
    def inventory_scope(self) -> str:
        """Return a stable, non-secret identity for the provider staging namespace."""
        ...

    def put(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, str] | None = None,
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

    def list_staged(
        self,
        *,
        page_size: int = DEFAULT_INVENTORY_PAGE_SIZE,
        cursor: str | None = None,
    ) -> StagingPage:
        """Return one bounded staging inventory page and an opaque continuation cursor."""
        ...

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        """Delete one exact object key and report whether it existed."""
        ...

    def stage(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        metadata: Mapping[str, str] | None = None,
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
