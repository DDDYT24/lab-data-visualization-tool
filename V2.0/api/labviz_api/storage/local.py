"""Filesystem object storage backend for local development and contract tests."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .base import ObjectInfo

CHUNK_SIZE = 1024 * 1024


class InvalidObjectKey(ValueError):
    """Raised when a key could escape the configured object root."""


class ObjectAlreadyExists(FileExistsError):
    """Raised when an immutable object key would be overwritten."""


class ObjectIntegrityError(ValueError):
    """Raised when the committed bytes do not match the expected digest."""


class LocalObjectStorage:
    """Store objects below one root using atomic same-directory replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_key(key: str) -> PurePosixPath:
        if not key or "\\" in key:
            raise InvalidObjectKey("Object keys must be non-empty POSIX paths.")
        normalized = PurePosixPath(key)
        if normalized.is_absolute() or any(part in {"", ".", ".."} for part in normalized.parts):
            raise InvalidObjectKey("Object keys cannot be absolute or traverse parent paths.")
        return normalized

    def _path(self, key: str) -> Path:
        normalized = self._normalize_key(key)
        candidate = self.root.joinpath(*normalized.parts).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise InvalidObjectKey("Object key resolves outside the configured root.")
        return candidate

    def put(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> ObjectInfo:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise ObjectAlreadyExists(key)

        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=".labviz-upload-",
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := source.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha256 = digest.hexdigest()
            if expected_sha256 and sha256 != expected_sha256.lower():
                raise ObjectIntegrityError("Object SHA-256 does not match the expected digest.")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        return ObjectInfo(key=key, size_bytes=size_bytes, sha256=sha256)

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> bool:
        target = self._path(key)
        if not target.is_file():
            return False
        target.unlink()
        return True
