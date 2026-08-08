"""Filesystem object storage backend for local development and contract tests."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from .base import (
    DEFAULT_INVENTORY_PAGE_SIZE,
    MAX_INVENTORY_PAGE_SIZE,
    InvalidObjectKey,
    InvalidStorageCursor,
    ObjectAlreadyExists,
    ObjectInfo,
    ObjectIntegrityError,
    StagedObject,
    StagingPage,
)
from .cursor import decode_cursor, encode_cursor

CHUNK_SIZE = 1024 * 1024
ALLOWED_CALLER_METADATA = {
    "labviz-format-version",
    "labviz-media-type",
    "labviz-pandas-version",
    "labviz-pyarrow-version",
    "labviz-parquet-writer",
    "labviz-parquet-writer-version",
}
PERSISTED_METADATA_FIELDS = ALLOWED_CALLER_METADATA | {"labviz-created-at"}


class LocalObjectStorage:
    """Store objects below one root using atomic same-directory replacement."""

    backend_name = "local"

    def __init__(self, root: Path, *, cursor_ttl_seconds: int = 86_400) -> None:
        if cursor_ttl_seconds < 1:
            raise ValueError("Storage cursor TTL must be positive.")
        self.root = root.resolve()
        self.cursor_ttl_seconds = cursor_ttl_seconds
        root_digest = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:32]
        self._inventory_scope = f"local-staging-v1:{root_digest}"
        self._inventory_lock = threading.Lock()
        self._inventory_clock = datetime.min.replace(tzinfo=UTC)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def inventory_scope(self) -> str:
        return self._inventory_scope

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
        candidate = self.root.joinpath(*normalized.parts)
        resolved_root = self._comparable_path(self.root.resolve())
        resolved_candidate = self._comparable_path(candidate.resolve())
        try:
            common = os.path.commonpath((resolved_root, resolved_candidate))
        except ValueError as exc:
            raise InvalidObjectKey("Object key resolves outside the configured root.") from exc
        if os.path.normcase(common) != os.path.normcase(resolved_root):
            raise InvalidObjectKey("Object key resolves outside the configured root.")
        return candidate

    @staticmethod
    def _comparable_path(path: Path) -> str:
        value = str(path)
        if os.name == "nt" and value.startswith("\\\\?\\"):
            value = value[4:]
        return os.path.normcase(os.path.abspath(value))

    def put(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        application_metadata = self._validate_metadata(metadata)
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
            # Publish bytes and their inventory timestamp under one local-provider
            # boundary.  Windows wall-clock timestamps may repeat within one tick;
            # the logical clock keeps post-snapshot writes strictly newer.
            with self._inventory_lock:
                created_at = self._next_inventory_timestamp_locked()
                persisted_metadata = {
                    "labviz-created-at": created_at.isoformat(),
                    **application_metadata,
                }
                if overwrite:
                    os.replace(temporary, destination)
                else:
                    try:
                        os.link(temporary, destination)
                    except FileExistsError as exc:
                        raise ObjectAlreadyExists(key) from exc
                self._write_metadata(key, persisted_metadata, overwrite=overwrite)
        finally:
            if temporary.exists():
                temporary.unlink()

        return self._object_info(destination, key)

    def stage(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StagedObject:
        self._path(key)
        staging_key = f".staging/{uuid4().hex}.part"
        info = self.put(
            staging_key,
            source,
            expected_sha256=expected_sha256,
            overwrite=False,
            metadata=metadata,
        )
        return StagedObject(
            key=key,
            staging_key=staging_key,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
            last_modified=info.last_modified,
            metadata=info.metadata,
        )

    def open_staged(self, staged: StagedObject) -> BinaryIO:
        return self.open(staged.staging_key)

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        staging_path = self._path(staged.staging_key)
        destination = self._path(staged.key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            existing = self._object_info(destination, staged.key)
            if existing.sha256 != staged.sha256 or existing.size_bytes != staged.size_bytes:
                raise ObjectAlreadyExists(staged.key)
            missing_metadata = {
                name: value
                for name, value in staged.metadata.items()
                if name in ALLOWED_CALLER_METADATA and name not in existing.metadata
            }
            if missing_metadata:
                with suppress(ObjectAlreadyExists):
                    self._write_metadata(
                        staged.key,
                        {
                            "labviz-created-at": datetime.now(UTC).isoformat(),
                            **missing_metadata,
                        },
                        overwrite=False,
                    )
                existing = self._object_info(destination, staged.key)
            self._require_metadata(existing.metadata, staged.metadata, staged.key)
            self.discard(staged)
            return existing
        if not staging_path.is_file():
            raise FileNotFoundError(staged.staging_key)
        try:
            os.link(staging_path, destination)
        except FileExistsError:
            return self.confirm(staged)
        self._write_metadata(
            staged.key,
            {
                "labviz-created-at": datetime.now(UTC).isoformat(),
                **{
                    name: value
                    for name, value in staged.metadata.items()
                    if name in ALLOWED_CALLER_METADATA
                },
            },
            overwrite=False,
        )
        self.discard(staged)
        return self._object_info(destination, staged.key)

    def discard(self, staged: StagedObject) -> bool:
        return self.delete(staged.staging_key)

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def head(self, key: str) -> ObjectInfo | None:
        target = self._path(key)
        if not target.is_file():
            return None
        return self._object_info(target, key)

    def list_staged(
        self,
        *,
        page_size: int = DEFAULT_INVENTORY_PAGE_SIZE,
        cursor: str | None = None,
    ) -> StagingPage:
        if not 1 <= page_size <= MAX_INVENTORY_PAGE_SIZE:
            raise ValueError(
                f"Inventory page size must be between 1 and {MAX_INVENTORY_PAGE_SIZE}."
            )
        now = datetime.now(UTC)
        last_key = ""
        with self._inventory_lock:
            snapshot_at = max(now, self._inventory_clock)
            self._inventory_clock = snapshot_at
        if cursor is not None:
            state = decode_cursor(
                cursor,
                backend_name=self.backend_name,
                inventory_scope=self.inventory_scope,
                now=now,
            )
            last_key = state.get("lastKey", "")
            try:
                snapshot_at = datetime.fromisoformat(state["snapshotAt"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidStorageCursor("Local inventory cursor state is malformed.") from exc
            if not isinstance(last_key, str) or snapshot_at.tzinfo is None:
                raise InvalidStorageCursor("Local inventory cursor state is malformed.")
        staging_root = self._path(".staging")
        if not staging_root.is_dir():
            return StagingPage((), None, False)
        selected = heapq.nsmallest(
            page_size + 1,
            self._inventory_entries(staging_root, last_key=last_key, snapshot_at=snapshot_at),
            key=lambda item: item[0],
        )
        has_more = len(selected) > page_size
        selected = selected[:page_size]
        items = tuple(
            info
            for key, target in selected
            if (info := self._object_info_if_present(target, key)) is not None
        )
        next_cursor = None
        if has_more and selected:
            next_cursor = encode_cursor(
                backend_name=self.backend_name,
                inventory_scope=self.inventory_scope,
                state={"lastKey": selected[-1][0], "snapshotAt": snapshot_at.isoformat()},
                issued_at=now,
                ttl_seconds=self.cursor_ttl_seconds,
            )
        return StagingPage(items, next_cursor, has_more)

    def _next_inventory_timestamp_locked(self) -> datetime:
        created_at = datetime.now(UTC)
        if created_at <= self._inventory_clock:
            created_at = self._inventory_clock + timedelta(microseconds=1)
        self._inventory_clock = created_at
        return created_at

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        target = self._path(key)
        if not target.is_file():
            return False
        if expected is not None:
            current = self._object_info(target, key)
            if (
                current.sha256 != expected.sha256
                or current.size_bytes != expected.size_bytes
                or (
                    expected.last_modified is not None
                    and current.last_modified != expected.last_modified
                )
            ):
                raise ObjectIntegrityError("Object changed before conditional deletion.")
        target.unlink()
        metadata_path = self._metadata_path(key)
        if metadata_path.is_file():
            metadata_path.unlink()
        return True

    def _inventory_entries(
        self,
        staging_root: Path,
        *,
        last_key: str,
        snapshot_at: datetime,
    ) -> Iterator[tuple[str, Path]]:
        for target in staging_root.rglob("*.part"):
            try:
                stat = target.stat()
            except FileNotFoundError:
                continue
            key = target.relative_to(self.root).as_posix()
            modified = datetime.fromtimestamp(stat.st_mtime, UTC)
            metadata = self._read_metadata(key)
            created_at = metadata.get("labviz-created-at")
            try:
                created = datetime.fromisoformat(created_at) if created_at else modified
            except ValueError as exc:
                raise ObjectIntegrityError("Local object has malformed creation metadata.") from exc
            if created.tzinfo is None:
                raise ObjectIntegrityError("Local object creation metadata lacks a timezone.")
            if target.is_file() and key > last_key and created <= snapshot_at:
                yield key, target

    def _object_info_if_present(self, target: Path, key: str) -> ObjectInfo | None:
        try:
            return self._object_info(target, key)
        except FileNotFoundError:
            return None

    def _object_info(
        self,
        target: Path,
        key: str,
    ) -> ObjectInfo:
        with target.open("rb") as source:
            sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        stat = target.stat()
        application_metadata = {
            "labviz-metadata-version": "1",
            "labviz-sha256": sha256,
            "labviz-size": str(stat.st_size),
            **self._read_metadata(key),
        }
        return ObjectInfo(
            key=key,
            size_bytes=stat.st_size,
            sha256=sha256,
            last_modified=datetime.fromtimestamp(stat.st_mtime, UTC),
            metadata=application_metadata,
        )

    def _metadata_path(self, key: str) -> Path:
        normalized = self._normalize_key(key).as_posix()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.root / ".metadata" / f"{digest}.json"

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, str] | None) -> dict[str, str]:
        supplied = {str(key): str(value) for key, value in (metadata or {}).items()}
        if set(supplied) - ALLOWED_CALLER_METADATA:
            raise ValueError("Object metadata contains unsupported application fields.")
        return supplied

    def _write_metadata(
        self,
        key: str,
        metadata: Mapping[str, str],
        *,
        overwrite: bool,
    ) -> None:
        supplied = {str(key): str(value) for key, value in metadata.items()}
        if set(supplied) - PERSISTED_METADATA_FIELDS:
            raise ValueError("Object metadata contains unsupported persisted fields.")
        path = self._metadata_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".labviz-meta-")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(supplied, output, sort_keys=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError as exc:
                    if self._read_metadata(key) != supplied:
                        raise ObjectAlreadyExists(key) from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_metadata(self, key: str) -> dict[str, str]:
        path = self._metadata_path(key)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ObjectIntegrityError("Local object metadata is unreadable.") from exc
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
        ):
            raise ObjectIntegrityError("Local object metadata is malformed.")
        try:
            normalized = {str(key): str(item) for key, item in value.items()}
            if set(normalized) - PERSISTED_METADATA_FIELDS:
                raise ValueError
            return normalized
        except ValueError as exc:
            raise ObjectIntegrityError("Local object metadata is unsupported.") from exc

    @staticmethod
    def _require_metadata(
        existing: Mapping[str, str],
        expected: Mapping[str, str],
        key: str,
    ) -> None:
        for name in ALLOWED_CALLER_METADATA:
            if name in expected and existing.get(name) != expected[name]:
                raise ObjectAlreadyExists(key)
