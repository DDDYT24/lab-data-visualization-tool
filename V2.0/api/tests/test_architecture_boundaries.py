from __future__ import annotations

import json
from pathlib import Path

from labviz_api.config import Settings

V2_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_INFRASTRUCTURE = {"celery", "kafka-python", "redis"}
DEPLOY_ROOT = V2_ROOT / "deploy" / "aws"


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


def test_phase6a_images_and_ecs_examples_keep_runtime_boundaries() -> None:
    api_dockerfile = (V2_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    web_dockerfile = (V2_ROOT / "web" / "Dockerfile").read_text(encoding="utf-8")
    next_config = (V2_ROOT / "web" / "next.config.ts").read_text(encoding="utf-8")

    assert "USER labviz" in api_dockerfile
    assert "USER node" in web_dockerfile
    assert 'output: "standalone"' in next_config
    assert "requirements.lock.txt" in api_dockerfile
    assert "http://127.0.0.1:8000/health" in api_dockerfile
    assert "http://127.0.0.1:8000/api/v1/ready" not in api_dockerfile

    definitions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(DEPLOY_ROOT.glob("*-task-definition.example.json"))
    ]
    assert len(definitions) == 3
    serialized = json.dumps(definitions)
    assert "AWS_ACCESS_KEY_ID" not in serialized
    assert "AWS_SECRET_ACCESS_KEY" not in serialized
    assert "${ACCOUNT_ID}" in serialized

    api_definition = next(item for item in definitions if item["family"].endswith("-api"))
    api_container = api_definition["containerDefinitions"][0]
    api_health_command = " ".join(api_container["healthCheck"]["command"])
    secret_names = {item["name"] for item in api_container["secrets"]}
    assert "http://127.0.0.1:8000/health" in api_health_command
    assert "/api/v1/ready" not in api_health_command
    assert {
        "LABVIZ_POSTGRES_URL",
        "LABVIZ_SHARE_TOKEN_KEYS",
        "LABVIZ_SMTP_PASSWORD",
    } <= secret_names

    worker_definition = next(item for item in definitions if "-worker-" in item["family"])
    worker_container = worker_definition["containerDefinitions"][0]
    worker_secret_names = {item["name"] for item in worker_container["secrets"]}
    worker_environment = {item["name"]: item["value"] for item in worker_container["environment"]}
    assert "healthCheck" not in worker_container
    assert worker_secret_names == {"LABVIZ_POSTGRES_URL"}
    assert worker_environment["LABVIZ_RUNTIME_ROLE"] == "worker"

    deployment_contract = (DEPLOY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "`/api/v1/ready`" in deployment_contract
    assert "ECS restart storms" in deployment_contract
    assert "do not attach a PostgreSQL/S3 dependency probe" in deployment_contract
