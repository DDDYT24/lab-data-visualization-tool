from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "backup" / "logical_backup.py"
SPEC = importlib.util.spec_from_file_location("logical_backup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
logical_backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logical_backup)


def test_sha256_streams_the_expected_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "backup.dump"
    artifact.write_bytes(b"labviz-backup" * 1000)

    assert logical_backup._sha256(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_required_rejects_missing_or_blank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LABVIZ_TEST_REQUIRED", raising=False)
    with pytest.raises(RuntimeError, match="LABVIZ_TEST_REQUIRED is required"):
        logical_backup._required("LABVIZ_TEST_REQUIRED")

    monkeypatch.setenv("LABVIZ_TEST_REQUIRED", " value ")
    assert logical_backup._required("LABVIZ_TEST_REQUIRED") == "value"
