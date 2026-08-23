from __future__ import annotations

import json
import re
from pathlib import Path

from labviz_api.config import Settings

V2_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_INFRASTRUCTURE = {"celery", "kafka-python", "redis"}
DEPLOY_ROOT = V2_ROOT / "deploy" / "aws"
PHASE6_GATE_CONTRACT = V2_ROOT / "contracts" / "phase6-gates-v1.json"
TERRAFORM_ROOT = DEPLOY_ROOT / "terraform"


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


def test_phase6_gate_contract_is_acyclic_and_assigns_real_cloud_evidence_to_6c() -> None:
    contract = json.loads(PHASE6_GATE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["schemaVersion"] == 1

    stages = contract["stages"]
    assert set(stages) == {
        "phase6_pre",
        "phase6b_application",
        "phase6c_cloud",
        "phase6d_launch",
    }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        assert stage_id in stages
        assert stage_id not in visiting, f"cyclic Phase 6 gate at {stage_id}"
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for required_stage in stages[stage_id]["requires"]:
            visit(required_stage)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id, stage in stages.items():
        visit(stage_id)
        for unlocked_stage in stage["unlocks"]:
            assert stage_id in stages[unlocked_stage]["requires"]

    assert stages["phase6b_application"]["requires"] == ["phase6_pre"]
    assert stages["phase6c_cloud"]["requires"] == ["phase6b_application"]
    assert stages["phase6d_launch"]["requires"] == ["phase6c_cloud"]

    evidence = contract["transferredEvidence"]
    assert set(evidence) == {
        "alb_forwarding_chain",
        "deployed_iam_roles",
        "ses_delivery_feedback",
        "ses_operational_alarms",
    }
    assert all(item["implementedBy"] == "phase6b_application" for item in evidence.values())
    assert all(item["acceptedBy"] == "phase6c_cloud" for item in evidence.values())
    assert all(item["realServiceRequired"] is True for item in evidence.values())

    for document in (
        "PERSISTENCE_PHASE6.md",
        "PERSISTENCE_PHASE6B.md",
        "PERSISTENCE_PHASE6C.md",
        "PERSISTENCE_PHASE6D.md",
        "PHASE6_AWS_DEPENDENCY_MATRIX.md",
    ):
        assert "phase6-gates-v1.json" in (V2_ROOT / document).read_text(encoding="utf-8")


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
        "LABVIZ_CLIENT_IDENTITY_KEY",
    } <= secret_names
    assert not {name for name in secret_names if "SMTP" in name or "SES" in name}
    api_environment = {item["name"]: item["value"] for item in api_container["environment"]}
    assert api_environment["LABVIZ_AUTH_MODE"] == "ses"
    assert api_environment["LABVIZ_SES_REGION"] == "${AWS_REGION}"
    assert api_environment["LABVIZ_SES_CONFIGURATION_SET"] == "${SES_CONFIGURATION_SET}"

    ses_policy = json.loads(
        (DEPLOY_ROOT / "api-ses-task-role-policy.example.json").read_text(encoding="utf-8")
    )
    statements = ses_policy["Statement"]
    assert len(statements) == 1
    assert statements[0]["Action"] == "ses:SendEmail"
    assert statements[0]["Resource"].endswith(":identity/${SES_IDENTITY}")
    assert set(statements[0]["Condition"]["StringEquals"]) == {"ses:FromAddress"}

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


def test_phase6c0_terraform_layout_and_state_boundaries_are_tracked() -> None:
    expected_modules = {
        "network",
        "security",
        "data",
        "compute",
        "edge",
        "email",
        "observability",
        "backup",
    }
    assert {path.name for path in (TERRAFORM_ROOT / "modules").iterdir() if path.is_dir()} == (
        expected_modules
    )

    roots = {
        "bootstrap": TERRAFORM_ROOT / "bootstrap",
        "account": TERRAFORM_ROOT / "account",
        "staging": TERRAFORM_ROOT / "environments" / "staging",
        "production": TERRAFORM_ROOT / "environments" / "production",
    }
    assert all(path.is_dir() for path in roots.values())

    backends = {
        name: (path / "versions.tf").read_text(encoding="utf-8")
        for name, path in roots.items()
        if name != "bootstrap"
    }
    expected_keys = {
        "account": "account/terraform.tfstate",
        "staging": "staging/terraform.tfstate",
        "production": "production/terraform.tfstate",
    }
    for name, config in backends.items():
        assert f'key          = "{expected_keys[name]}"' in config
        assert 'region       = "ap-southeast-1"' in config
        assert "encrypt      = true" in config
        assert "use_lockfile = true" in config
        assert "bucket" not in config
        assert "access_key" not in config
        assert "secret_key" not in config


def test_phase6c0_state_bootstrap_is_protected_and_state_access_is_least_privilege() -> None:
    bootstrap_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TERRAFORM_ROOT / "bootstrap").glob("*.tf"))
    )

    assert 'resource "aws_s3_bucket" "terraform_state"' in bootstrap_text
    assert "force_destroy = false" in bootstrap_text
    assert "prevent_destroy = true" in bootstrap_text
    assert 'sse_algorithm = "AES256"' in bootstrap_text
    assert 'status = "Enabled"' in bootstrap_text
    assert all(
        setting in bootstrap_text
        for setting in (
            "block_public_acls       = true",
            "block_public_policy     = true",
            "ignore_public_acls      = true",
            "restrict_public_buckets = true",
        )
    )
    assert 'variable = "aws:SecureTransport"' in bootstrap_text
    assert 'values   = ["false"]' in bootstrap_text

    assert 'url = "https://token.actions.githubusercontent.com"' in bootstrap_text
    assert 'client_id_list = ["sts.amazonaws.com"]' in bootstrap_text
    assert 'variable = "token.actions.githubusercontent.com:aud"' in bootstrap_text
    assert 'variable = "token.actions.githubusercontent.com:sub"' in bootstrap_text
    assert "repo:${var.github_repository}:environment:staging" in bootstrap_text
    assert "repo:${var.github_repository}:environment:production" in bootstrap_text
    assert "max_session_duration = 3600" in bootstrap_text

    delete_object_lines = [
        line for line in bootstrap_text.splitlines() if "s3:DeleteObject" in line
    ]
    assert len(delete_object_lines) == 1
    lock_statement = bootstrap_text.split('sid     = "ManageSelectedLockFiles"', maxsplit=1)[1]
    assert "s3:DeleteObject" in lock_statement
    assert "terraform.tfstate.tflock" in lock_statement


def test_phase6c0_account_guardrails_and_secret_hygiene() -> None:
    terraform_files = sorted(TERRAFORM_ROOT.rglob("*.tf"))
    terraform_text = "\n".join(path.read_text(encoding="utf-8") for path in terraform_files)
    account_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TERRAFORM_ROOT / "account").glob("*.tf"))
    )

    assert 'name         = "labviz-monthly-cost-usd-30"' in account_text
    assert 'limit_amount = "30"' in account_text
    assert "include_credit = false" in account_text
    assert "include_refund = false" in account_text
    assert 'resource "aws_ce_anomaly_subscription" "daily"' in account_text
    assert 'resource "aws_cloudtrail" "account"' in account_text
    assert "is_multi_region_trail         = true" in account_text
    assert "enable_log_file_validation    = true" in account_text
    assert 'variable = "aws:MultiFactorAuthPresent"' in account_text
    assert "max_session_duration = 3600" in account_text

    assert "AWS_ACCESS_KEY_ID" not in terraform_text
    assert "AWS_SECRET_ACCESS_KEY" not in terraform_text
    assert re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", terraform_text) is None
    assert re.search(r"\b\d{12}\b", terraform_text) is None

    gitignore = (V2_ROOT.parent / ".gitignore").read_text(encoding="utf-8")
    assert "*.tfstate" in gitignore
    assert "*.tfplan" in gitignore
    assert "*.tfvars" in gitignore
    assert ".terraform.lock.hcl" not in gitignore
