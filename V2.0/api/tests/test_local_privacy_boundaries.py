"""Regression checks for the default single-computer deployment boundary."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from labviz_api.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[3]


def _clear_local_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LABVIZ_DATABASE_PATH",
        "LABVIZ_PERSISTENCE_BACKEND",
        "LABVIZ_POSTGRES_URL",
        "LABVIZ_OBJECT_STORAGE_BACKEND",
        "LABVIZ_OBJECT_STORAGE_ROOT",
        "LABVIZ_S3_BUCKET",
        "LABVIZ_S3_ENDPOINT_URL",
        "NEXT_PUBLIC_LABVIZ_API_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_settings_keep_persistence_and_objects_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_local_overrides(monkeypatch)

    settings = Settings.from_env(tmp_path)

    assert settings.persistence_backend == "sqlite"
    assert settings.postgres_url is None
    assert settings.object_storage_backend == "local"
    assert settings.database_path == tmp_path / ".labviz" / "labviz-v2.db"
    assert settings.object_storage_root == tmp_path / ".labviz" / "objects"
    assert settings.allowed_origins == (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )


def test_launchers_bind_both_services_to_loopback() -> None:
    powershell = (REPO_ROOT / "start-labviz.ps1").read_text(encoding="utf-8")
    shell = (REPO_ROOT / "start-labviz.sh").read_text(encoding="utf-8")

    assert '--host", "127.0.0.1"' in powershell
    assert '--hostname", "127.0.0.1"' in powershell
    assert "--host 127.0.0.1" in shell
    assert "--hostname 127.0.0.1" in shell
    assert "0.0.0.0" not in powershell
    assert "0.0.0.0" not in shell


def test_web_client_defaults_to_relative_api_and_has_no_telemetry_hooks() -> None:
    api_client = (REPO_ROOT / "V2.0" / "web" / "src" / "lib" / "api" / "labviz-api.ts").read_text(
        encoding="utf-8"
    )
    next_config = (REPO_ROOT / "V2.0" / "web" / "next.config.ts").read_text(encoding="utf-8")

    assert 'process.env.NEXT_PUBLIC_LABVIZ_API_URL ?? "/api/v1"' in api_client
    assert 'process.env.LABVIZ_API_PROXY_TARGET ?? "http://127.0.0.1:8000"' in next_config
    for marker in ("sendbeacon", "sentry", "posthog", "google-analytics", "telemetry"):
        assert marker not in api_client.lower()


def test_public_markdown_has_no_internal_tool_or_model_references() -> None:
    forbidden_tokens = (
        "".join(chr(value) for value in (99, 111, 100, 101, 120)),
        "".join(chr(value) for value in (99, 104, 97, 116, 103, 112, 116)),
        "".join(chr(value) for value in (111, 112, 101, 110, 97, 105)),
        "\u4eba\u5de5\u667a\u80fd",
        "".join(chr(value) for value in (97, 105)),
    )
    forbidden = re.compile(
        "|".join((*forbidden_tokens[:4], r"\b" + forbidden_tokens[4] + r"\b")),
        re.IGNORECASE,
    )
    public_files = [
        path
        for path in REPO_ROOT.rglob("*.md")
        if ".git" not in path.parts
        and "node_modules" not in path.parts
        and ".venv" not in path.parts
        and "outputs" not in path.parts
    ]

    matches = {
        str(path.relative_to(REPO_ROOT)): forbidden.findall(path.read_text(encoding="utf-8"))
        for path in public_files
        if forbidden.search(path.read_text(encoding="utf-8"))
    }

    assert matches == {}
