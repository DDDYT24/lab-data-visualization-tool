# Phase 6C-2 Implementation Record

**Status:** Code is tracked on `main`; no AWS environment was applied and Phase 6C acceptance
remains pending.

**Current backlog and completion status:** [`TODO.md`](TODO.md)

## Boundary

- Direct parent and Phase 6C-1 candidate: `e215f262a884aaa8243128b3d87db57bd0410b89`.
- Scope: Phase 6C-2 compute, deployment, IAM, and secret-metadata code only.
- No AWS resource was created, changed, or deleted by this unit.
- DNS authority, ACM validation, SES verification, secret values, image publication, migration
  execution, service activation, and real staging evidence remain external gates.

## Implemented

- KMS-encrypted, immutable, scan-on-push ECR repositories for API and Web.
- Digest-only ECS/Fargate task definitions with non-root application images, read-only root filesystems,
  writable `/tmp` volumes, and retained KMS-encrypted logs.
- Independent API, Web, five accepted worker services, and one-shot Alembic migration task.
- API liveness on `/health`; ALB readiness remains `/api/v1/ready`; workers have no HTTP probes.
- ECS circuit-breaker rollback and ALB unhealthy-target alarm rollback for request-serving services.
- Separate execution/task roles and role-scoped S3, KMS, SES, log, image, and secret permissions.
- Three Secrets Manager containers with recovery windows but no Terraform-managed secret versions.
- Fail-closed service activation and worker-deletion switches.

## Verification

- `terraform fmt -check -recursive V2.0/deploy/aws/terraform`
- `terraform validate` for staging and production with AWS provider `6.61.0`
- Backend-independent native Terraform plan tests for staging and production
- Phase 6 architecture-boundary regression tests
- Repository secret/account/e-mail hygiene scan and staged-scope audit

Passing local tests establish code behavior only. They are not an AWS apply, migration, deployment,
ALB, IAM, SES, CloudWatch, DNS, backup, or browser acceptance claim.
