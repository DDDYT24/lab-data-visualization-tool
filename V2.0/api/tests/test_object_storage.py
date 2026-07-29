from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

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
