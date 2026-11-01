from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from labviz_api.artifacts import ARTIFACT_PREFIX, ArtifactStore, artifact_key
from labviz_api.storage.base import ObjectIntegrityError
from labviz_api.storage.local import LocalObjectStorage

PNG = b"\x89PNG\r\n\x1a\n" + b"local artifact bytes" * 16


def store(root: Path) -> ArtifactStore:
    return ArtifactStore(LocalObjectStorage(root))


def test_a_digest_fans_out_into_a_bounded_directory() -> None:
    digest = hashlib.sha256(PNG).hexdigest()

    assert artifact_key(digest) == f"{ARTIFACT_PREFIX}/{digest[:2]}/{digest}"
    assert artifact_key(digest.upper()) == artifact_key(digest)


@pytest.mark.parametrize(
    "value",
    ["", "zz", "../escape", "a" * 63, "a" * 65, "A" * 64 + "0"],
)
def test_an_invalid_digest_never_becomes_a_key(value: str) -> None:
    with pytest.raises(ValueError):
        artifact_key(value)


def test_writing_then_reading_round_trips_and_records_identity(tmp_path: Path) -> None:
    artifacts = store(tmp_path / "objects")

    stored = artifacts.put(PNG, media_type="image/png")

    assert stored.sha256 == hashlib.sha256(PNG).hexdigest()
    assert stored.byte_size == len(PNG)
    assert artifacts.exists(stored.sha256)
    assert artifacts.read(stored.sha256) == PNG


def test_identical_renders_share_one_file_instead_of_conflicting(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    artifacts = store(root)

    first = artifacts.put(PNG, media_type="image/png")
    second = artifacts.put(PNG, media_type="image/png")

    assert first == second
    stored_files = [path for path in (root / ARTIFACT_PREFIX).rglob("*") if path.is_file()]
    payloads = [path for path in stored_files if path.name == first.sha256]
    assert len(payloads) == 1


def test_different_bytes_stay_separate(tmp_path: Path) -> None:
    artifacts = store(tmp_path / "objects")

    first = artifacts.put(PNG, media_type="image/png")
    second = artifacts.put(PNG + b"tail", media_type="image/png")

    assert first.sha256 != second.sha256
    assert artifacts.read(first.sha256) == PNG
    assert artifacts.read(second.sha256) == PNG + b"tail"


def test_a_tampered_artifact_is_refused_rather_than_returned(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    artifacts = store(root)
    stored = artifacts.put(PNG, media_type="image/png")
    (root / artifact_key(stored.sha256)).write_bytes(PNG + b"tampered")

    with pytest.raises(ObjectIntegrityError):
        artifacts.read(stored.sha256)


def test_a_missing_artifact_raises_instead_of_returning_empty_bytes(tmp_path: Path) -> None:
    artifacts = store(tmp_path / "objects")
    absent = hashlib.sha256(b"never stored").hexdigest()

    assert not artifacts.exists(absent)
    with pytest.raises(OSError):
        artifacts.read(absent)


def test_delete_removes_only_the_named_artifact(tmp_path: Path) -> None:
    artifacts = store(tmp_path / "objects")
    kept = artifacts.put(PNG, media_type="image/png")
    removed = artifacts.put(PNG + b"tail", media_type="image/png")

    assert artifacts.delete(removed.sha256)
    assert not artifacts.exists(removed.sha256)
    assert artifacts.read(kept.sha256) == PNG
