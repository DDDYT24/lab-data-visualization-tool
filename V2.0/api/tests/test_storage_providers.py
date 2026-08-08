from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from labviz_api.storage import (
    InvalidObjectKey,
    InvalidStorageCursor,
    LocalObjectStorage,
    ObjectAlreadyExists,
    ObjectIntegrityError,
    ObjectStorage,
)
from labviz_api.storage.cursor import encode_cursor
from labviz_api.storage.s3 import S3ObjectStorage

MINIO_ENDPOINT = os.environ.get("LABVIZ_TEST_MINIO_ENDPOINT", "http://127.0.0.1:59000")
MINIO_BUCKET = os.environ.get("LABVIZ_TEST_MINIO_BUCKET", "labviz-test")
MINIO_ACCESS_KEY = os.environ.get("LABVIZ_TEST_MINIO_ACCESS_KEY", "labviz-minio")
MINIO_SECRET_KEY = os.environ.get(
    "LABVIZ_TEST_MINIO_SECRET_KEY",
    "labviz-minio-local-only",
)
FIVE_MIB = 5 * 1024 * 1024


def _minio_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 1},
        ),
    )


def _ensure_minio_bucket() -> Any:
    client = _minio_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            pytest.fail(f"Real MinIO is unavailable: {code}")
        client.create_bucket(Bucket=MINIO_BUCKET)
    return client


@pytest.fixture(params=("local", "minio"))
def storage_provider(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[ObjectStorage]:
    if request.param == "local":
        yield LocalObjectStorage(tmp_path / "objects")
        return
    client = _ensure_minio_bucket()
    prefix = f"contract/{uuid4().hex}"
    storage = S3ObjectStorage(
        bucket=MINIO_BUCKET,
        prefix=prefix,
        endpoint_url=MINIO_ENDPOINT,
        region="us-east-1",
        multipart_threshold=FIVE_MIB,
        multipart_part_size=FIVE_MIB,
        client=client,
    )
    yield storage
    listed = client.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=f"{prefix}/")
    objects = [{"Key": item["Key"]} for item in listed.get("Contents", [])]
    if objects:
        client.delete_objects(Bucket=MINIO_BUCKET, Delete={"Objects": objects})


def test_provider_contract_stage_confirm_open_metadata_and_idempotent_cleanup(
    storage_provider: ObjectStorage,
) -> None:
    payload = b"provider-contract-bytes"
    sha256 = hashlib.sha256(payload).hexdigest()
    staged = storage_provider.stage(
        "datasets/final.parquet",
        io.BytesIO(payload),
        expected_sha256=sha256,
        metadata={"labviz-format-version": "parquet-v1"},
    )
    with storage_provider.open_staged(staged) as source:
        assert source.read() == payload
    committed = storage_provider.confirm(staged)
    repeated = storage_provider.confirm(staged)

    assert committed == repeated
    assert committed.sha256 == sha256
    assert committed.size_bytes == len(payload)
    assert committed.last_modified is not None
    assert committed.metadata["labviz-sha256"] == sha256
    assert committed.metadata["labviz-size"] == str(len(payload))
    with storage_provider.open(committed.key) as source:
        assert source.read() == payload
    assert not storage_provider.discard(staged)
    assert storage_provider.delete(committed.key, expected=committed)
    assert not storage_provider.delete(committed.key)


def test_provider_contract_put_rejects_escape_and_overwrite(
    storage_provider: ObjectStorage,
) -> None:
    storage_provider.put("exports/final.png", io.BytesIO(b"first"))
    with pytest.raises(ObjectAlreadyExists):
        storage_provider.put("exports/final.png", io.BytesIO(b"second"))
    with pytest.raises(InvalidObjectKey):
        storage_provider.put("../outside", io.BytesIO(b"unsafe"))
    with pytest.raises(InvalidObjectKey):
        storage_provider.head("C:\\outside")


def test_provider_contract_bounded_cursor_resume_and_scope_binding(
    storage_provider: ObjectStorage,
    tmp_path: Path,
) -> None:
    for index in range(5):
        storage_provider.put(f".staging/{index}.part", io.BytesIO(str(index).encode()))
    keys: list[str] = []
    cursor = None
    while True:
        page = storage_provider.list_staged(page_size=2, cursor=cursor)
        assert len(page.items) <= 2
        keys.extend(item.key for item in page.items)
        if not page.has_more:
            assert page.next_cursor is None
            break
        assert page.next_cursor is not None
        cursor = page.next_cursor
    assert keys == [f".staging/{index}.part" for index in range(5)]
    assert len(keys) == len(set(keys))

    first = storage_provider.list_staged(page_size=1)
    assert first.next_cursor is not None
    with pytest.raises(InvalidStorageCursor):
        storage_provider.list_staged(page_size=1, cursor=f"{first.next_cursor}x")
    if storage_provider.backend_name == "local":
        other: ObjectStorage = LocalObjectStorage(tmp_path / "other")
    else:
        other = S3ObjectStorage(
            bucket=MINIO_BUCKET,
            prefix=f"other/{uuid4().hex}",
            endpoint_url=MINIO_ENDPOINT,
            region="us-east-1",
            multipart_threshold=FIVE_MIB,
            multipart_part_size=FIVE_MIB,
            client=_minio_client(),
        )
    with pytest.raises(InvalidStorageCursor):
        other.list_staged(page_size=1, cursor=first.next_cursor)


def test_provider_contract_concurrent_confirm_is_immutable(
    storage_provider: ObjectStorage,
) -> None:
    payload = b"same-immutable-output"
    staged = [
        storage_provider.stage("exports/final.png", io.BytesIO(payload)) for _index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(storage_provider.confirm, staged))
    assert results[0].sha256 == results[1].sha256 == hashlib.sha256(payload).hexdigest()

    conflicting = storage_provider.stage("exports/final.png", io.BytesIO(b"different"))
    with pytest.raises(ObjectAlreadyExists):
        storage_provider.confirm(conflicting)
    storage_provider.discard(conflicting)


def test_provider_contract_metadata_persists_and_conflicts_are_rejected(
    storage_provider: ObjectStorage,
) -> None:
    first = storage_provider.stage(
        "datasets/metadata.parquet",
        io.BytesIO(b"same"),
        metadata={
            "labviz-format-version": "parquet-v1",
            "labviz-media-type": "application/vnd.apache.parquet",
            "labviz-pandas-version": "pandas-test",
            "labviz-pyarrow-version": "pyarrow-test",
            "labviz-parquet-writer": "pyarrow.parquet.write_table",
            "labviz-parquet-writer-version": "writer-test",
        },
    )
    committed = storage_provider.confirm(first)
    reopened = storage_provider.head(committed.key)
    assert reopened is not None
    assert reopened.metadata["labviz-format-version"] == "parquet-v1"
    assert reopened.metadata["labviz-media-type"] == "application/vnd.apache.parquet"
    assert reopened.metadata["labviz-pandas-version"] == "pandas-test"
    assert reopened.metadata["labviz-pyarrow-version"] == "pyarrow-test"
    assert reopened.metadata["labviz-parquet-writer"] == "pyarrow.parquet.write_table"
    assert reopened.metadata["labviz-parquet-writer-version"] == "writer-test"

    conflicting = storage_provider.stage(
        committed.key,
        io.BytesIO(b"same"),
        metadata={
            "labviz-format-version": "publication-export-v1",
            "labviz-media-type": "image/png",
        },
    )
    with pytest.raises(ObjectAlreadyExists):
        storage_provider.confirm(conflicting)
    storage_provider.discard(conflicting)


def test_provider_contract_inventory_snapshot_excludes_new_and_tolerates_delete(
    storage_provider: ObjectStorage,
) -> None:
    staged = [
        storage_provider.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
        for index in range(4)
    ]
    first = storage_provider.list_staged(page_size=2)
    assert first.next_cursor is not None
    first_keys = {item.key for item in first.items}
    delete_target = next(item for item in staged if item.staging_key not in first_keys)
    added = storage_provider.stage("datasets/new.parquet", io.BytesIO(b"new"))
    storage_provider.discard(delete_target)

    keys = [item.key for item in first.items]
    cursor: str | None = first.next_cursor
    while cursor is not None:
        page = storage_provider.list_staged(page_size=2, cursor=cursor)
        keys.extend(item.key for item in page.items)
        cursor = page.next_cursor
    expected = {item.staging_key for item in staged if item != delete_target}
    assert set(keys) == expected
    assert len(keys) == len(set(keys))
    assert added.staging_key not in keys


def test_provider_contract_empty_boundary_and_expired_cursor(
    storage_provider: ObjectStorage,
) -> None:
    empty = storage_provider.list_staged(page_size=2)
    assert empty.items == () and empty.next_cursor is None and not empty.has_more
    staged = [
        storage_provider.stage(f"datasets/{index}.parquet", io.BytesIO(b"x")) for index in range(2)
    ]
    boundary = storage_provider.list_staged(page_size=2)
    assert len(boundary.items) == 2
    assert boundary.next_cursor is None and not boundary.has_more

    expired = encode_cursor(
        backend_name=storage_provider.backend_name,
        inventory_scope=storage_provider.inventory_scope,
        state={"snapshotAt": datetime.now(UTC).isoformat()},
        issued_at=datetime.now(UTC) - timedelta(seconds=2),
        ttl_seconds=1,
    )
    with pytest.raises(InvalidStorageCursor, match="expired"):
        storage_provider.list_staged(page_size=2, cursor=expired)
    for item in staged:
        storage_provider.discard(item)


def test_provider_contract_inventory_excludes_future_overwrite(
    storage_provider: ObjectStorage,
) -> None:
    staged = [
        storage_provider.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
        for index in range(4)
    ]
    first = storage_provider.list_staged(page_size=2)
    first_keys = {item.key for item in first.items}
    overwrite_target = next(item for item in staged if item.staging_key not in first_keys)
    storage_provider.put(
        overwrite_target.staging_key,
        io.BytesIO(b"overwritten"),
        overwrite=True,
    )
    keys = list(first_keys)
    cursor = first.next_cursor
    while cursor is not None:
        page = storage_provider.list_staged(page_size=2, cursor=cursor)
        keys.extend(item.key for item in page.items)
        cursor = page.next_cursor
    assert overwrite_target.staging_key not in keys
    assert len(keys) == len(set(keys))
    next_generation = storage_provider.list_staged(page_size=10)
    assert overwrite_target.staging_key in {item.key for item in next_generation.items}


def test_real_minio_small_and_multipart_paths_write_authoritative_metadata() -> None:
    client = _ensure_minio_bucket()
    prefix = f"multipart/{uuid4().hex}"
    storage = S3ObjectStorage(
        bucket=MINIO_BUCKET,
        prefix=prefix,
        endpoint_url=MINIO_ENDPOINT,
        region="us-east-1",
        multipart_threshold=FIVE_MIB,
        multipart_part_size=FIVE_MIB,
        client=client,
    )
    small = storage.put("small.bin", io.BytesIO(b"small"))
    large_payload = b"x" * (FIVE_MIB + 1)
    large = storage.put("large.bin", io.BytesIO(large_payload))

    assert small.metadata["labviz-upload-mode"] == "single"
    assert large.metadata["labviz-upload-mode"] == "multipart"
    assert large.sha256 == hashlib.sha256(large_payload).hexdigest()
    assert large.size_bytes == len(large_payload)
    assert large.etag is not None and large.etag != large.sha256
    storage.delete(small.key, expected=small)
    storage.delete(large.key, expected=large)


class FailSecondPartClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.upload_calls = 0
        self.abort_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def upload_part(self, **parameters: Any) -> Any:
        self.upload_calls += 1
        if self.upload_calls == 2:
            raise ClientError(
                {"Error": {"Code": "InjectedPartFailure", "Message": "injected"}},
                "UploadPart",
            )
        return self.client.upload_part(**parameters)

    def abort_multipart_upload(self, **parameters: Any) -> Any:
        self.abort_calls += 1
        return self.client.abort_multipart_upload(**parameters)


def test_real_minio_multipart_failure_aborts_upload() -> None:
    base_client = _ensure_minio_bucket()
    client = FailSecondPartClient(base_client)
    prefix = f"abort/{uuid4().hex}"
    storage = S3ObjectStorage(
        bucket=MINIO_BUCKET,
        prefix=prefix,
        endpoint_url=MINIO_ENDPOINT,
        region="us-east-1",
        multipart_threshold=FIVE_MIB,
        multipart_part_size=FIVE_MIB,
        client=client,
    )

    with pytest.raises(OSError, match="InjectedPartFailure"):
        storage.put("large.bin", io.BytesIO(b"x" * (FIVE_MIB * 2 + 1)))
    assert client.upload_calls == 2
    assert client.abort_calls == 1
    uploads = base_client.list_multipart_uploads(
        Bucket=MINIO_BUCKET,
        Prefix=f"{prefix}/large.bin",
    )
    assert uploads.get("Uploads", []) == []


def test_real_minio_rejects_object_without_application_integrity_metadata() -> None:
    client = _ensure_minio_bucket()
    prefix = f"legacy/{uuid4().hex}"
    client.put_object(
        Bucket=MINIO_BUCKET,
        Key=f"{prefix}/legacy.bin",
        Body=b"legacy",
    )
    storage = S3ObjectStorage(
        bucket=MINIO_BUCKET,
        prefix=prefix,
        endpoint_url=MINIO_ENDPOINT,
        region="us-east-1",
        multipart_threshold=FIVE_MIB,
        multipart_part_size=FIVE_MIB,
        client=client,
    )
    with pytest.raises(ObjectIntegrityError, match="integrity metadata"):
        storage.head("legacy.bin")
    client.delete_object(Bucket=MINIO_BUCKET, Key=f"{prefix}/legacy.bin")
