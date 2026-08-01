"""AWS S3 and MinIO compatible object storage provider."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, BinaryIO, cast
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

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
MIN_MULTIPART_PART_SIZE = 5 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CALLER_METADATA = {
    "labviz-format-version",
    "labviz-media-type",
}
EXPLICIT_CONTINUATION_ERROR_CODES = {
    "expiredcontinuationtoken",
    "invalidcontinuationtoken",
    "invalidcontinuationtokenexception",
}
CONTEXTUAL_CONTINUATION_ERROR_CODES = {
    "expiredtoken",
    "invalidargument",
    "invalidtoken",
}


class S3ObjectStorage:
    """S3-compatible immutable object storage using conditional writes."""

    backend_name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        multipart_threshold: int = 16 * 1024 * 1024,
        multipart_part_size: int = 8 * 1024 * 1024,
        connect_timeout_seconds: int = 5,
        read_timeout_seconds: int = 60,
        cursor_ttl_seconds: int = 86_400,
        client: Any | None = None,
        validate_bucket: bool = True,
    ) -> None:
        if not bucket or len(bucket) > 255:
            raise ValueError("S3 bucket must be configured.")
        if multipart_part_size < MIN_MULTIPART_PART_SIZE:
            raise ValueError("S3 multipart part size must be at least 5 MiB.")
        if multipart_threshold < multipart_part_size:
            raise ValueError("S3 multipart threshold must be at least the multipart part size.")
        if min(connect_timeout_seconds, read_timeout_seconds, cursor_ttl_seconds) < 1:
            raise ValueError("S3 timeouts and cursor TTL must be positive.")
        self.bucket = bucket
        self.prefix = self._normalize_prefix(prefix)
        self.multipart_threshold = multipart_threshold
        self.multipart_part_size = multipart_part_size
        self.cursor_ttl_seconds = cursor_ttl_seconds
        self._inventory_scope = f"s3://{bucket}/{self.prefix}.staging"
        self.client = client or boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            config=Config(
                signature_version="s3v4",
                connect_timeout=connect_timeout_seconds,
                read_timeout=read_timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        if validate_bucket:
            self._call("head_bucket", Bucket=self.bucket)

    @property
    def inventory_scope(self) -> str:
        return self._inventory_scope

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        value = prefix.strip("/")
        if value and ("\\" in value or any(part in {"", ".", ".."} for part in value.split("/"))):
            raise InvalidObjectKey("S3 prefix must be a normalized POSIX path.")
        return f"{value}/" if value else ""

    @staticmethod
    def _normalize_key(key: str) -> str:
        if not key or "\\" in key or key.startswith("/"):
            raise InvalidObjectKey("Object keys must be non-empty relative POSIX paths.")
        parts = key.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise InvalidObjectKey("Object keys cannot escape the configured S3 prefix.")
        return "/".join(parts)

    def _provider_key(self, key: str) -> str:
        return f"{self.prefix}{self._normalize_key(key)}"

    def _logical_key(self, provider_key: str) -> str:
        if not provider_key.startswith(self.prefix):
            raise InvalidObjectKey("Provider key is outside the configured S3 prefix.")
        return self._normalize_key(provider_key[len(self.prefix) :])

    def put(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        provider_key = self._provider_key(key)
        with tempfile.SpooledTemporaryFile(max_size=self.multipart_threshold, mode="w+b") as spool:
            digest = hashlib.sha256()
            size_bytes = 0
            while chunk := source.read(CHUNK_SIZE):
                spool.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
            sha256 = digest.hexdigest()
            if expected_sha256 is not None and sha256 != expected_sha256.lower():
                raise ObjectIntegrityError("Object SHA-256 does not match the expected digest.")
            multipart = size_bytes >= self.multipart_threshold
            application_metadata = self._metadata(
                sha256,
                size_bytes,
                metadata,
                upload_mode="multipart" if multipart else "single",
            )
            spool.seek(0)
            if multipart:
                spool.rollover()
                self._multipart_put(
                    provider_key,
                    cast(BinaryIO, spool),
                    application_metadata,
                    overwrite=overwrite,
                )
            else:
                parameters: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "Key": provider_key,
                    "Body": spool,
                    "ContentLength": size_bytes,
                    "Metadata": application_metadata,
                }
                if not overwrite:
                    parameters["IfNoneMatch"] = "*"
                self._conditional_write("put_object", key, **parameters)
        info = self.head(key)
        if info is None or info.sha256 != sha256 or info.size_bytes != size_bytes:
            raise ObjectIntegrityError("S3 object metadata differs after upload.")
        return info

    def _multipart_put(
        self,
        provider_key: str,
        source: BinaryIO,
        metadata: Mapping[str, str],
        *,
        overwrite: bool,
    ) -> None:
        created = self._call(
            "create_multipart_upload",
            Bucket=self.bucket,
            Key=provider_key,
            Metadata=dict(metadata),
        )
        upload_id = cast(str, created["UploadId"])
        parts: list[dict[str, Any]] = []
        try:
            part_number = 1
            while chunk := source.read(self.multipart_part_size):
                response = self._call(
                    "upload_part",
                    Bucket=self.bucket,
                    Key=provider_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                    ContentLength=len(chunk),
                )
                parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
                part_number += 1
            parameters: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": provider_key,
                "UploadId": upload_id,
                "MultipartUpload": {"Parts": parts},
            }
            if not overwrite:
                parameters["IfNoneMatch"] = "*"
            self._conditional_write("complete_multipart_upload", provider_key, **parameters)
        except Exception:
            with suppress(OSError):
                self._call(
                    "abort_multipart_upload",
                    Bucket=self.bucket,
                    Key=provider_key,
                    UploadId=upload_id,
                )
            raise

    def stage(
        self,
        key: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StagedObject:
        self._normalize_key(key)
        staging_key = f".staging/{uuid4().hex}.part"
        info = self.put(
            staging_key,
            source,
            expected_sha256=expected_sha256,
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

    def confirm(self, staged: StagedObject) -> ObjectInfo:
        existing = self.head(staged.key)
        if existing is not None:
            self._require_identity(existing, staged)
            self.discard(staged)
            return existing
        with self.open_staged(staged) as source:
            try:
                committed = self.put(
                    staged.key,
                    source,
                    expected_sha256=staged.sha256,
                    overwrite=False,
                    metadata={
                        key: value
                        for key, value in staged.metadata.items()
                        if key in ALLOWED_CALLER_METADATA
                    },
                )
            except ObjectAlreadyExists:
                existing_after_race = self.head(staged.key)
                if existing_after_race is None:
                    raise
                self._require_identity(existing_after_race, staged)
                committed = existing_after_race
        self.discard(staged)
        return committed

    @staticmethod
    def _require_identity(info: ObjectInfo, staged: StagedObject) -> None:
        if info.sha256 != staged.sha256 or info.size_bytes != staged.size_bytes:
            raise ObjectAlreadyExists(staged.key)
        for name in ALLOWED_CALLER_METADATA:
            if name in staged.metadata and info.metadata.get(name) != staged.metadata[name]:
                raise ObjectAlreadyExists(staged.key)

    def open(self, key: str) -> BinaryIO:
        response = self._call("get_object", Bucket=self.bucket, Key=self._provider_key(key))
        return cast(BinaryIO, response["Body"])

    def open_staged(self, staged: StagedObject) -> BinaryIO:
        return self.open(staged.staging_key)

    def exists(self, key: str) -> bool:
        return self.head(key) is not None

    def head(self, key: str) -> ObjectInfo | None:
        provider_key = self._provider_key(key)
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=provider_key)
        except ClientError as exc:
            if self._error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise self._safe_error(exc) from exc
        except BotoCoreError as exc:
            raise OSError("S3 head operation failed.") from exc
        metadata = {
            str(name).lower(): str(value) for name, value in response.get("Metadata", {}).items()
        }
        sha256 = metadata.get("labviz-sha256")
        recorded_size = metadata.get("labviz-size")
        size_bytes = int(response["ContentLength"])
        if (
            sha256 is None
            or SHA256_PATTERN.fullmatch(sha256) is None
            or recorded_size is None
            or not recorded_size.isdigit()
            or int(recorded_size) != size_bytes
        ):
            raise ObjectIntegrityError("S3 object is missing valid LabViz integrity metadata.")
        modified = response.get("LastModified")
        return ObjectInfo(
            key=key,
            size_bytes=size_bytes,
            sha256=sha256,
            last_modified=modified.astimezone(UTC) if isinstance(modified, datetime) else None,
            metadata=metadata,
            etag=str(response.get("ETag", "")).strip('"') or None,
            version_id=cast(str | None, response.get("VersionId")),
        )

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
        snapshot_at = now
        token: str | None = None
        if cursor is not None:
            state = decode_cursor(
                cursor,
                backend_name=self.backend_name,
                inventory_scope=self.inventory_scope,
                now=now,
            )
            token = state.get("providerToken")
            try:
                snapshot_at = datetime.fromisoformat(state["snapshotAt"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidStorageCursor("S3 inventory cursor state is malformed.") from exc
            if not isinstance(token, str) or not token or snapshot_at.tzinfo is None:
                raise InvalidStorageCursor("S3 inventory cursor state is malformed.")
        parameters: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": f"{self.prefix}.staging/",
            "MaxKeys": page_size,
        }
        if token:
            parameters["ContinuationToken"] = token
        response = self._list_staged_page(
            parameters,
            has_continuation_token=token is not None,
        )
        items: list[ObjectInfo] = []
        for item in response.get("Contents", []):
            modified = item.get("LastModified")
            if isinstance(modified, datetime) and modified.astimezone(UTC) > snapshot_at:
                continue
            logical_key = self._logical_key(str(item["Key"]))
            if not logical_key.endswith(".part"):
                continue
            info = self.head(logical_key)
            if info is None:
                continue
            created_at = info.metadata.get("labviz-created-at")
            try:
                created = datetime.fromisoformat(created_at) if created_at else None
            except ValueError as exc:
                raise ObjectIntegrityError(
                    "S3 object has malformed LabViz creation metadata."
                ) from exc
            if created is not None and created.tzinfo is None:
                raise ObjectIntegrityError("S3 object creation metadata lacks a timezone.")
            if created is not None and created > snapshot_at:
                continue
            items.append(info)
        has_more = bool(response.get("IsTruncated"))
        next_cursor = None
        if has_more:
            next_token = response.get("NextContinuationToken")
            if not isinstance(next_token, str) or not next_token:
                raise InvalidStorageCursor("S3 did not return a valid continuation token.")
            next_cursor = encode_cursor(
                backend_name=self.backend_name,
                inventory_scope=self.inventory_scope,
                state={"providerToken": next_token, "snapshotAt": snapshot_at.isoformat()},
                issued_at=now,
                ttl_seconds=self.cursor_ttl_seconds,
            )
        return StagingPage(tuple(items), next_cursor, has_more)

    def _list_staged_page(
        self,
        parameters: Mapping[str, Any],
        *,
        has_continuation_token: bool,
    ) -> Any:
        try:
            return self.client.list_objects_v2(**parameters)
        except ClientError as exc:
            if has_continuation_token and self._is_rejected_continuation_token(exc):
                raise InvalidStorageCursor(
                    "Inventory continuation cursor was rejected and cannot be resumed."
                ) from exc
            raise self._safe_error(exc) from exc
        except BotoCoreError as exc:
            raise OSError("S3 list_objects_v2 operation failed.") from exc

    @classmethod
    def _is_rejected_continuation_token(cls, error: ClientError) -> bool:
        details = error.response.get("Error", {})
        if not isinstance(details, Mapping):
            return False
        code = cls._error_code(error).casefold()
        if code in EXPLICIT_CONTINUATION_ERROR_CODES:
            return True
        if code not in CONTEXTUAL_CONTINUATION_ERROR_CODES:
            return False
        context = " ".join(
            str(details.get(name, "")) for name in ("Message", "ArgumentName", "ParameterName")
        ).casefold()
        normalized = context.replace("-", " ").replace("_", " ")
        return "continuation" in normalized and "token" in normalized

    def delete(self, key: str, *, expected: ObjectInfo | None = None) -> bool:
        current = self.head(key)
        if current is None:
            return False
        if expected is not None:
            if current.sha256 != expected.sha256 or current.size_bytes != expected.size_bytes:
                raise ObjectIntegrityError("S3 object changed before conditional deletion.")
            if (
                expected.last_modified is not None
                and current.last_modified != expected.last_modified
            ):
                raise ObjectIntegrityError("S3 object changed before conditional deletion.")
        parameters: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._provider_key(key),
        }
        if current.etag is not None:
            parameters["IfMatch"] = current.etag
        if current.version_id is not None:
            parameters["VersionId"] = current.version_id
        try:
            self._call("delete_object", **parameters)
        except OSError as exc:
            if "PreconditionFailed" in str(exc):
                raise ObjectIntegrityError(
                    "S3 object changed before conditional deletion."
                ) from exc
            raise
        return True

    def discard(self, staged: StagedObject) -> bool:
        return self.delete(staged.staging_key)

    @staticmethod
    def _metadata(
        sha256: str,
        size_bytes: int,
        caller: Mapping[str, str] | None,
        *,
        upload_mode: str,
    ) -> dict[str, str]:
        supplied = dict(caller or {})
        unexpected = set(supplied) - ALLOWED_CALLER_METADATA
        if unexpected:
            raise ValueError("Object metadata contains unsupported application fields.")
        return {
            "labviz-metadata-version": "1",
            "labviz-sha256": sha256,
            "labviz-size": str(size_bytes),
            "labviz-upload-mode": upload_mode,
            "labviz-created-at": datetime.now(UTC).isoformat(),
            **supplied,
        }

    def _conditional_write(self, operation: str, logical_key: str, **parameters: Any) -> Any:
        try:
            return getattr(self.client, operation)(**parameters)
        except ClientError as exc:
            if self._error_code(exc) in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise ObjectAlreadyExists(logical_key) from exc
            raise self._safe_error(exc) from exc
        except BotoCoreError as exc:
            raise OSError(f"S3 {operation} operation failed.") from exc

    def _call(self, operation: str, **parameters: Any) -> Any:
        try:
            return getattr(self.client, operation)(**parameters)
        except ClientError as exc:
            raise self._safe_error(exc) from exc
        except BotoCoreError as exc:
            raise OSError(f"S3 {operation} operation failed.") from exc

    @staticmethod
    def _error_code(error: ClientError) -> str:
        return str(error.response.get("Error", {}).get("Code", "Unknown"))[:128]

    def _safe_error(self, error: ClientError) -> OSError:
        return OSError(f"S3 operation failed ({self._error_code(error)}).")
