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

    def delete(self, key: str) -> bool:
        """Delete one exact object key and report whether it existed."""
        ...
