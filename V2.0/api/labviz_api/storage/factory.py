"""Build exactly one object storage provider from application settings."""

from __future__ import annotations

from labviz_api.config import Settings

from .base import ObjectStorage
from .local import LocalObjectStorage
from .s3 import S3ObjectStorage


def build_object_storage(settings: Settings) -> ObjectStorage:
    if settings.object_storage_backend == "local":
        return LocalObjectStorage(
            settings.object_storage_root,
            cursor_ttl_seconds=settings.object_storage_cursor_ttl_seconds,
        )
    if settings.s3_bucket is None:
        raise ValueError("S3 object storage requires LABVIZ_S3_BUCKET.")
    return S3ObjectStorage(
        bucket=settings.s3_bucket,
        prefix=settings.s3_prefix,
        region=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
        multipart_threshold=settings.s3_multipart_threshold_bytes,
        multipart_part_size=settings.s3_multipart_part_size_bytes,
        connect_timeout_seconds=settings.s3_connect_timeout_seconds,
        read_timeout_seconds=settings.s3_read_timeout_seconds,
        cursor_ttl_seconds=settings.object_storage_cursor_ttl_seconds,
    )
