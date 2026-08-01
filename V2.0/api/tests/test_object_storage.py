from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from labviz_api.storage import InvalidStorageCursor
from labviz_api.storage.local import (
    InvalidObjectKey,
    LocalObjectStorage,
    ObjectAlreadyExists,
    ObjectIntegrityError,
)


def test_local_object_storage_streams_and_verifies_content(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    storage = LocalObjectStorage(root)
    payload = b"immutable dataset bytes"
    expected = hashlib.sha256(payload).hexdigest()

    info = storage.put(
        "datasets/project-1/version-1.parquet",
        io.BytesIO(payload),
        expected_sha256=expected,
    )

    assert info.size_bytes == len(payload)
    assert info.sha256 == expected
    assert storage.exists(info.key)
    with storage.open(info.key) as stored:
        assert stored.read() == payload
    assert storage.delete(info.key)
    assert not storage.delete(info.key)


def test_local_object_storage_rejects_overwrite_and_path_traversal(
    tmp_path: Path,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    storage.put("exports/figure.png", io.BytesIO(b"first"))

    with pytest.raises(ObjectAlreadyExists):
        storage.put("exports/figure.png", io.BytesIO(b"second"))
    with pytest.raises(InvalidObjectKey):
        storage.put("../outside", io.BytesIO(b"unsafe"))
    with pytest.raises(InvalidObjectKey):
        storage.exists("C:\\outside")


def test_local_object_storage_removes_failed_integrity_write(
    tmp_path: Path,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")

    with pytest.raises(ObjectIntegrityError):
        storage.put("datasets/bad.parquet", io.BytesIO(b"payload"), expected_sha256="0" * 64)

    assert not storage.exists("datasets/bad.parquet")


def test_local_object_storage_metadata_and_bounded_resumable_inventory(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    storage = LocalObjectStorage(root)
    staged = [
        storage.stage(f"datasets/{index}.parquet", io.BytesIO(f"value-{index}".encode()))
        for index in range(5)
    ]

    cursor = None
    keys: list[str] = []
    while True:
        page = LocalObjectStorage(root).list_staged(page_size=2, cursor=cursor)
        assert len(page.items) <= 2
        for info in page.items:
            assert info.last_modified is not None
            assert info.metadata["labviz-sha256"] == info.sha256
            assert info.metadata["labviz-size"] == str(info.size_bytes)
        keys.extend(item.key for item in page.items)
        if not page.has_more:
            assert page.next_cursor is None
            break
        assert page.next_cursor is not None
        cursor = page.next_cursor

    assert keys == sorted(item.staging_key for item in staged)
    assert len(keys) == len(set(keys)) == 5


def test_local_inventory_cursor_rejects_tampering_and_another_root(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path / "one")
    for index in range(2):
        storage.stage(f"datasets/{index}.parquet", io.BytesIO(str(index).encode()))
    page = storage.list_staged(page_size=1)
    assert page.next_cursor is not None

    with pytest.raises(InvalidStorageCursor):
        storage.list_staged(page_size=1, cursor=f"{page.next_cursor}x")
    with pytest.raises(InvalidStorageCursor):
        LocalObjectStorage(tmp_path / "two").list_staged(
            page_size=1,
            cursor=page.next_cursor,
        )


def test_local_immutable_put_has_one_concurrent_winner(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")

    def write(payload: bytes) -> str:
        try:
            return storage.put("exports/final.png", io.BytesIO(payload)).sha256
        except ObjectAlreadyExists:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, (b"first", b"second")))

    assert results.count("exists") == 1
    with storage.open("exports/final.png") as source:
        assert source.read() in {b"first", b"second"}


def test_legacy_local_object_without_sidecar_is_still_fully_hashed(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    legacy = root / "datasets" / "legacy.parquet"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-bytes")

    info = LocalObjectStorage(root).head("datasets/legacy.parquet")

    assert info is not None
    assert info.sha256 == hashlib.sha256(b"legacy-bytes").hexdigest()
    assert info.size_bytes == len(b"legacy-bytes")
    assert info.metadata["labviz-sha256"] == info.sha256
    assert "labviz-format-version" not in info.metadata
