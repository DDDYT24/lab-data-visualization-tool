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


def _terraform_module_text(name: str) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TERRAFORM_ROOT / "modules" / name).glob("*.tf"))
    )


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


def test_phase6c1_network_and_security_keep_workloads_private() -> None:
    network = _terraform_module_text("network")
    security = _terraform_module_text("security")

    assert 'resource "aws_vpc" "this"' in network
    assert "slice(sort(data.aws_availability_zones.available.names), 0, 2)" in network
    assert network.count("map_public_ip_on_launch = false") == 3
    assert network.count('destination_cidr_block = "0.0.0.0/0"') == 1
    assert 'resource "aws_nat_gateway"' not in network
    assert 'resource "aws_eip"' not in network

    for service in ("ecr.api", "ecr.dkr", "logs", "secretsmanager", "kms", "email"):
        assert f'"{service}" =' in security or f"{service} =" in security
    assert 'vpc_endpoint_type   = "Interface"' in security
    assert "private_dns_enabled = true" in security
    assert 'service_name        = "com.amazonaws.${var.aws_region}.${each.key}"' in security
    assert 'ip_protocol = "-1"' not in security
    assert security.count('cidr_ipv4         = "0.0.0.0/0"') == 2
    assert "database_from_api" in security
    assert "database_from_worker" in security


def test_phase6c1_data_controls_encrypt_retain_and_prevent_public_access() -> None:
    data = _terraform_module_text("data")

    assert 'engine_version = "17"' in data
    assert 'family = "postgres17"' in data
    assert 'name         = "rds.force_ssl"' in data
    assert re.search(r"manage_master_user_password\s*=\s*true", data)
    assert "publicly_accessible    = false" in data
    assert "storage_encrypted     = true" in data
    assert re.search(r"backup_retention_period\s*=\s*var\.database_backup_retention_days", data)
    assert re.search(r"monitoring_interval\s*=\s*60", data)
    assert re.search(r"performance_insights_enabled\s*=\s*true", data)
    assert re.search(r"deletion_protection\s*=\s*var\.database_deletion_protection", data)
    assert re.search(r"skip_final_snapshot\s*=\s*false", data)
    assert data.count("prevent_destroy = true") == 2

    assert 'sse_algorithm     = "aws:kms"' in data
    assert "bucket_key_enabled = true" in data
    assert 'status = "Enabled"' in data
    assert "abort_incomplete_multipart_upload" in data
    assert "noncurrent_version_expiration" in data
    assert 'variable = "aws:SecureTransport"' in data
    assert all(
        setting in data
        for setting in (
            "block_public_acls       = true",
            "block_public_policy     = true",
            "ignore_public_acls      = true",
            "restrict_public_buckets = true",
        )
    )
    assert 'vpc_endpoint_type = "Gateway"' in data
    assert "prod-${var.aws_region}-starport-layer-bucket" in data
    assert '"${aws_s3_bucket.data.arn}/${var.object_prefix}*"' in data


def test_phase6c1_edge_and_environment_differences_are_explicit() -> None:
    edge = _terraform_module_text("edge")
    staging_root = (TERRAFORM_ROOT / "environments" / "staging" / "main.tf").read_text(
        encoding="utf-8"
    )
    production_root = (TERRAFORM_ROOT / "environments" / "production" / "main.tf").read_text(
        encoding="utf-8"
    )
    production_variables = (
        TERRAFORM_ROOT / "environments" / "production" / "variables.tf"
    ).read_text(encoding="utf-8")

    assert 'validation_method = "DNS"' in edge
    assert 'resource "aws_route53_record" "certificate_validation"' in edge
    assert 'resource "aws_route53_record" "application"' in edge
    assert "drop_invalid_header_fields       = true" in edge
    assert 'desync_mitigation_mode           = "strictest"' in edge
    assert 'xff_header_processing_mode       = "append"' in edge
    assert "enable_waf_fail_open             = false" in edge
    assert 'path                = "/api/v1/ready"' in edge
    assert 'values = ["/api/v1/*", "/health"]' in edge
    assert 'status_code = "HTTP_301"' in edge
    assert "default_action {\n    block {}" in edge

    assert re.search(r'vpc_cidr\s*=\s*"10\.20\.0\.0/16"', staging_root)
    assert re.search(r"database_multi_az\s*=\s*false", staging_root)
    assert re.search(r"database_backup_retention_days\s*=\s*7", staging_root)
    assert re.search(r'vpc_cidr\s*=\s*"10\.30\.0\.0/16"', production_root)
    assert re.search(r"database_multi_az\s*=\s*true", production_root)
    assert re.search(r"database_backup_retention_days\s*=\s*14", production_root)
    assert "default     = true" in production_variables.split('variable "enable_waf"', 1)[1]

    for environment in ("staging", "production"):
        test_file = TERRAFORM_ROOT / "environments" / environment / "phase6c1.tftest.hcl"
        assert test_file.is_file()
        assert "command = plan" in test_file.read_text(encoding="utf-8")


def test_phase6c2_images_tasks_and_services_fail_closed() -> None:
    compute = _terraform_module_text("compute")

    assert compute.count('image_tag_mutability = "IMMUTABLE"') == 2
    assert compute.count("scan_on_push = true") == 2
    assert 'encryption_type = "KMS"' in compute
    assert 'identifiers = ["logs.${var.aws_region}.amazonaws.com"]' in compute
    assert 'variable = "kms:EncryptionContext:aws:logs:arn"' in compute
    assert '"${aws_ecr_repository.api.repository_url}@${var.api_image_digest}"' in compute
    assert '"${aws_ecr_repository.web.repository_url}@${var.web_image_digest}"' in compute
    assert "readonlyRootFilesystem = true" in compute
    assert 'containerPath = "/tmp"' in compute
    assert "http://127.0.0.1:8000/health" in compute
    assert "http://127.0.0.1:8000/api/v1/ready" not in compute
    assert '["python", "-m", "alembic", "upgrade", "head"]' in compute
    assert len(re.findall(r"desired_count\s*=\s*var\.activate_services \? 1 : 0", compute)) == 3
    assert compute.count("assign_public_ip = false") == 3
    assert compute.count("deployment_circuit_breaker") == 3
    assert compute.count("rollback = true") >= 3


def test_phase6c2_workers_and_runtime_permissions_are_separated() -> None:
    compute = _terraform_module_text("compute")

    for worker in (
        "metadata-cleanup",
        "orphan-staging-inventory",
        "pending-reconciliation",
        "project-lifecycle",
        "stored-object-gc",
    ):
        assert f'"{worker}"' in compute

    assert (
        'command                = ["python", "-m", "labviz_api.workers.cli", each.key]' in compute
    )
    assert '{ name = "LABVIZ_WORKER_DESTRUCTIVE_MAINTENANCE", value = "false" }' in compute
    assert '{ name = "LABVIZ_WORKER_DRY_RUN", value = "true" }' in compute
    assert '{ name = "LABVIZ_WORKER_DELETE_ENABLED", value = "false" }' in compute
    assert 'each.key == "worker" && var.enable_worker_delete_permission' in compute
    assert 'role   = aws_iam_role.task["api"].id' in compute
    assert compute.count('actions   = ["ses:SendEmail"]') == 1


def test_phase6c2_secret_values_never_enter_terraform() -> None:
    compute = _terraform_module_text("compute")
    terraform_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(TERRAFORM_ROOT.rglob("*.tf"))
    )

    assert compute.count('resource "aws_secretsmanager_secret"') == 3
    assert 'resource "aws_secretsmanager_secret_version"' not in terraform_text
    assert "secret_string" not in terraform_text
    assert "secret_binary" not in terraform_text
    assert 'name      = "LABVIZ_POSTGRES_URL"' in compute
    assert "AWS_ACCESS_KEY_ID" not in terraform_text
    assert "AWS_SECRET_ACCESS_KEY" not in terraform_text


def test_phase6c3_email_and_alarm_paths_are_operationally_scoped() -> None:
    email = _terraform_module_text("email")
    observability = _terraform_module_text("observability")

    assert 'tls_policy = "REQUIRE"' in email
    assert 'suppressed_reasons = ["BOUNCE", "COMPLAINT"]' in email
    assert 'dimension_name          = "environment"' in email
    assert 'identifiers = ["ses.amazonaws.com"]' in email
    assert 'variable = "AWS:SourceAccount"' in email
    for event in ("BOUNCE", "COMPLAINT", "DELIVERY", "DELIVERY_DELAY", "REJECT", "SEND"):
        assert f'"{event}"' in email

    for signal in (
        "TargetResponseTime",
        "UnHealthyHostCount",
        "HTTPCode_ELB_5XX_Count",
        "RejectedConnectionCount",
        "DesiredTaskCount",
        "RunningTaskCount",
        "CPUUtilization",
        "MemoryUtilization",
        "DatabaseConnections",
        "FreeStorageSpace",
        "NumberOfBackupJobsFailed",
        "Reputation.BounceRate",
        "Reputation.ComplaintRate",
    ):
        assert f'"{signal}"' in observability
    assert 'source        = ["aws.s3", "aws.kms"]' in observability
    assert 'errorCode = [{ prefix = "AccessDenied" }]' in observability


def test_phase6c3_backup_has_continuous_daily_and_isolated_restore_paths() -> None:
    backup = _terraform_module_text("backup")
    data = _terraform_module_text("data")
    script = (DEPLOY_ROOT / "backup" / "logical_backup.py").read_text(encoding="utf-8")

    assert re.search(r"enable_continuous_backup\s*=\s*true", backup)
    assert "delete_after = 30" in backup
    assert "resources    = [var.database_arn, var.data_bucket_arn]" in backup
    assert 'resource "aws_backup_restore_testing_plan" "quarterly"' in backup
    assert "count = var.enable_restore_testing ? 1 : 0" in backup
    assert 'description = "No-ingress isolated RDS restore-testing group"' in backup
    assert re.search(r'PubliclyAccessible\s*=\s*"false"', backup)
    assert 'resource "aws_scheduler_schedule" "logical_backup"' in backup
    assert "count = var.activate_logical_backup_schedule ? 1 : 0" in backup
    assert "readonlyRootFilesystem = true" in backup
    assert "assign_public_ip = false" in backup
    assert (
        '"${aws_ecr_repository.logical_backup.repository_url}@${var.backup_image_digest}"' in backup
    )
    assert 'prefix = "backups/logical/"' in data
    assert data.count("noncurrent_days = 30") == 1
    assert 'resource "aws_s3_bucket_metric" "entire_bucket"' in data

    assert "PGPASSWORD" in script
    assert '"--no-owner"' in script
    assert '"--no-acl"' in script
    assert '"--sse-kms-key-id"' in script
    assert "logical-backup-failed" in script
    assert "raw_url" not in script.split("print", maxsplit=1)[-1]


def test_phase6c3_runbooks_cover_required_incidents_and_measured_objectives() -> None:
    runbook_root = V2_ROOT / "runbooks"
    runbook_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(runbook_root.glob("*.md"))
    )

    assert "availability `>=99.9%`" in runbook_text
    assert "RPO `<=15 minutes`" in runbook_text
    assert "RTO `<=4 hours`" in runbook_text
    for topic in (
        "Migration failure",
        "RDS",
        "S3",
        "SES",
        "KMS",
        "task exhaustion",
        "Worker quarantine",
        "Credential",
        "Backup restoration",
        "Security incident",
        "Enable destructive work",
    ):
        assert topic.lower() in runbook_text.lower()


def test_phase6c3_application_emits_non_sensitive_rate_limit_signal() -> None:
    main = (V2_ROOT / "api" / "labviz_api" / "main.py").read_text(encoding="utf-8")
    assert 'LOGGER.warning("authentication-request-rate-limited")' in main


def test_phase6c4_oidc_state_and_permissions_boundaries_fail_closed() -> None:
    bootstrap = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TERRAFORM_ROOT / "bootstrap").glob("*.tf"))
    )
    runtime_modules = "\n".join(
        _terraform_module_text(name) for name in ("backup", "compute", "data")
    )

    assert "repo:${var.github_repository}:pull_request" not in bootstrap
    assert "repo:${var.github_repository}:ref:refs/heads/main" in bootstrap
    assert "repo:${var.github_repository}:environment:staging" in bootstrap
    assert "repo:${var.github_repository}:environment:production" in bootstrap
    assert 'resource "aws_iam_policy" "runtime_boundary"' in bootstrap
    assert 'variable = "iam:PermissionsBoundary"' in bootstrap
    assert 'data "aws_iam_policy_document" "github_plan_read"' in bootstrap
    assert "ReadOnlyAccess" not in bootstrap
    assert 'sid       = "ChangeApprovedHostedZoneOnly"' in bootstrap
    assert "var.route53_zone_ids[each.key]" in bootstrap
    regional_permissions = bootstrap.split("ManageRegionalLabVizInfrastructure", maxsplit=1)[1]
    regional_permissions = regional_permissions.split(
        "ManageEnvironmentDataBucketsOnly", maxsplit=1
    )[0]
    assert '"s3:*"' not in regional_permissions
    assert runtime_modules.count("permissions_boundary = var.permissions_boundary_arn") == 8
    assert "AdministratorAccess" not in bootstrap


def test_phase6c4_workflows_enforce_ci_migration_before_apply_and_rollback() -> None:
    workflows = V2_ROOT.parent / ".github" / "workflows"
    infrastructure = (workflows / "phase6c-infrastructure.yml").read_text(encoding="utf-8")
    foundation = (workflows / "phase6c-foundation.yml").read_text(encoding="utf-8")
    release = (workflows / "phase6c-release.yml").read_text(encoding="utf-8")

    assert "pull_request:" in infrastructure
    assert "init -backend=false" in infrastructure
    assert "id-token: write" in infrastructure
    assert "github.event_name != 'pull_request'" in infrastructure
    assert "role-to-assume:" in infrastructure

    assert 'TF_VAR_activate_services: "false"' in foundation
    assert 'TF_VAR_activate_logical_backup_schedule: "false"' in foundation
    assert "workloadsDisabled:true" in foundation

    order = [
        "Require successful full CI for the exact commit",
        "Build and push immutable candidate images",
        "Scan API candidate",
        "Sign accepted digests with GitHub OIDC",
        "Create the reviewed release plan",
        "Capture current services and run candidate migration",
        "Apply only after successful migration",
        "Wait for every service and verify HTTPS",
        "Fail on active operational alarms",
        "Require no unexplained post-apply drift",
    ]
    positions = [release.index(item) for item in order]
    assert positions == sorted(positions)
    assert "cosign verify" in release
    assert '--image-ids imageTag="$GITHUB_SHA"' in release
    assert 'terraform -chdir="$TF_ROOT" apply' in release
    assert "if: failure()" in release
    assert "aws ecs update-service" in release
    assert "AWS_ACCESS_KEY_ID" not in infrastructure + foundation + release
    assert "AWS_SECRET_ACCESS_KEY" not in infrastructure + foundation + release


def test_phase6c4_live_evidence_contract_cannot_be_satisfied_by_code() -> None:
    evidence_path = V2_ROOT / "contracts" / "phase6c-evidence-v1.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["schemaVersion"] == 1
    assert evidence["phase"] == "6C"
    assert evidence["codeEvidenceIsAcceptance"] is False
    required = evidence["requiredLiveEvidence"]
    assert set(required) == {
        "terraform_apply",
        "private_network",
        "tls_dns_alb",
        "deployment_rollback",
        "backup_restore",
        "ses",
        "deployed_iam",
        "application",
    }
    assert all(item["required"] is True and item["artifacts"] for item in required.values())
    assert "terraform-state" in evidence["redactions"]


def test_phase6_cost_model_blocks_billable_apply_by_default() -> None:
    cost_path = V2_ROOT / "contracts" / "phase6-cost-model-v1.json"
    model = json.loads(cost_path.read_text(encoding="utf-8"))
    foundation = (V2_ROOT.parent / ".github" / "workflows" / "phase6c-foundation.yml").read_text(
        encoding="utf-8"
    )
    release = (V2_ROOT.parent / ".github" / "workflows" / "phase6c-release.yml").read_text(
        encoding="utf-8"
    )

    assert model["defaultDeploymentMode"] == "local-only"
    assert model["estimates"]["foundationWorkloadsDisabled"]["pricedMinimum"] > 30
    assert model["estimates"]["servicesActivated"]["pricedMinimum"] > 30
    assert model["acceptance"]["budgetIsHardCap"] is False
    assert model["acceptance"]["billableApplyAllowedByDefault"] is False
    for workflow in (foundation, release):
        assert "Require explicit recurring cost approval" in workflow
        assert "COST_APPROVAL_REFERENCE" in workflow
        assert 'test "$COST_APPROVAL_REFERENCE" != "local-only"' in workflow


def test_phase6_github_governance_is_complete_but_not_applied() -> None:
    governance_path = V2_ROOT / "contracts" / "phase6-github-governance-v1.json"
    governance = json.loads(governance_path.read_text(encoding="utf-8"))

    assert governance["repository"] == "DDDYT24/lab-data-visualization-tool"
    assert governance["defaultBranch"] == "main"
    assert governance["status"] == "prepared-not-applied"
    assert governance["externalMutationApplied"] is False
    assert governance["secretValuesStoredInRepository"] is False
    assert set(governance["environments"]) == {"staging", "production"}
    assert governance["repositorySecrets"] == ["NOTIFICATION_EMAIL"]
    assert "AWS_ACCESS_KEY_ID" not in governance["repositoryVariables"]
    assert "AWS_SECRET_ACCESS_KEY" not in governance["repositoryVariables"]
    for environment in governance["environments"].values():
        assert environment["deploymentBranch"] == "main"
        assert environment["preventSelfReview"] is True
        assert environment["requiredReviewers"] >= 1
        assert "COST_APPROVAL_REFERENCE" in environment["variables"]
        assert environment["secrets"] == ["NOTIFICATION_EMAIL"]

    protection = governance["branchProtection"]
    assert protection["enforceAdmins"] is True
    assert protection["allowForcePushes"] is False
    assert protection["allowDeletions"] is False
    required_jobs = {
        job for workflow in protection["requiredWorkflowJobs"] for job in workflow["jobs"]
    }
    assert required_jobs == {"python", "frontend", "v2-api", "v2-containers", "static"}


def test_phase6d_drafts_are_prepared_without_bypassing_phase6c() -> None:
    contracts = V2_ROOT / "contracts"
    preparation = json.loads(
        (contracts / "phase6d-preparation-v1.json").read_text(encoding="utf-8")
    )
    quotas = json.loads((contracts / "phase6d-quota-retention-v1.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (contracts / "phase6d-data-inventory-v1.json").read_text(encoding="utf-8")
    )
    load_recovery = json.loads(
        (contracts / "phase6d-load-recovery-v1.json").read_text(encoding="utf-8")
    )

    assert preparation["status"] == "draft-blocked-by-phase6c"
    assert preparation["phase6cAcceptanceRequired"] is True
    assert preparation["entryAllowed"] is False
    assert preparation["codeEvidenceIsAcceptance"] is False
    assert "phase6d-pass" in preparation["prohibitedClaims"]

    assert quotas["status"] == "candidate-not-enforced"
    assert quotas["acceptedExistingLimits"]["uploadBytes"] == 50 * 1024 * 1024
    assert quotas["candidateProductQuotas"]["savedProjectsPerUser"] > 0
    assert quotas["candidateProductQuotas"]["retainedObjectBytesPerUser"] > 0
    assert quotas["accountingRequirements"]["reservationMustBeAtomic"] is True
    assert quotas["accountingRequirements"]["releaseMustBeAtomic"] is True

    assert inventory["status"] == "draft-not-legal-advice"
    assert inventory["complianceCertificationClaimed"] is False
    inventory_ids = {item["id"] for item in inventory["dataClasses"]}
    assert {"uploaded-source-bytes", "user-email", "client-identity-digest"} <= inventory_ids
    assert {"database-object-and-logical-backups", "incident-and-support-evidence"} <= inventory_ids

    assert load_recovery["status"] == "draft-local-preflight-only"
    assert load_recovery["loadProfiles"]["localIntegrity"]["acceptanceAuthority"] is False
    assert load_recovery["loadProfiles"]["stagingQualification"]["acceptanceAuthority"] is True
    assert load_recovery["disasterRecovery"]["rpoMinutesMaximum"] == 15
    assert load_recovery["disasterRecovery"]["rtoMinutesMaximum"] == 240
    assert load_recovery["disasterRecovery"]["localSimulationIsAcceptance"] is False
