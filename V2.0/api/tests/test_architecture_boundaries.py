from __future__ import annotations

import json
from pathlib import Path

from labviz_api.config import Settings

V2_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_INFRASTRUCTURE = {"celery", "kafka-python", "redis"}


def test_sqlite_remains_the_local_reference_default() -> None:
    settings = Settings(
        database_path=V2_ROOT / "api" / ".labviz" / "test.db",
        allowed_origins=("http://127.0.0.1:3000",),
        public_web_url="http://127.0.0.1:3000",
    )

    assert settings.persistence_backend == "sqlite"


def test_deferred_queue_infrastructure_is_not_declared() -> None:
    python_inputs = {
        line.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower()
        for path in (V2_ROOT / "api" / "requirements.txt", V2_ROOT / "api" / "requirements-dev.txt")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("-")
    }
    package = json.loads((V2_ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    javascript_inputs = set(package["dependencies"]) | set(package["devDependencies"])

    assert FORBIDDEN_INFRASTRUCTURE.isdisjoint(python_inputs)
    assert FORBIDDEN_INFRASTRUCTURE.isdisjoint(javascript_inputs)
