# Phase 6C-4 Implementation Record

**Status:** CI/CD code is tracked on remote `main`. The generic CI and Phase 6C infrastructure
static workflow have run; the protected foundation/release workflows have not run. AWS execution
and Phase 6C acceptance remain pending.

**Last reviewed:** 2026-09-10

**Current backlog and completion status:** [`TODO.md`](TODO.md)

## Boundary

- Direct parent and Phase 6C-3 candidate: `7074266ef77f12858c5da6a7ff043aa469955d82`.
- Scope: Phase 6C-4 GitHub Actions, OIDC deployment permissions, release sequencing, rollback, and
  machine-readable evidence requirements.
- No AWS/GitHub environment, resource, image, DNS record, secret, or deployment was changed here.

## Implemented

- Pull-request Terraform format/validate/misconfiguration/secret scan with no AWS/state access.
- Main-branch OIDC plans and one-day machine-readable plan artifacts.
- Protected, main-only foundation workflow with all workloads and paid schedules fail-closed.
- Protected release workflow that requires exact-commit full CI; builds immutable API/Web/backup
  images; scans, signs, and verifies digests; plans; runs migration; applies; waits; checks HTTPS,
  alarms, and drift; and rolls services back after failure.
- Environment-specific GitHub OIDC roles and runtime permissions boundaries; no static AWS keys.
- Machine-readable live-evidence contract and exact GitHub/DNS setup handoff.

## Local verification

- Workflow YAML parse and static contract tests.
- Terraform formatting/validation and native staging/production graph plans.
- Architecture-boundary tests, pre-commit hooks, secret/account/e-mail scan, and staged-scope audit.

Local workflow inspection cannot replace a successful protected GitHub run or any real staging
evidence in `contracts/phase6c-evidence-v1.json`.
