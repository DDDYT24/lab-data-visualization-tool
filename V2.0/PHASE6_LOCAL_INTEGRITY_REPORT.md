# Phase 6 local integrity report

Status: **LOCAL PASS**

Verified: 2026-08-25 (Asia/Shanghai)

Application revision: `6f173aa`

This report verifies the application and release artifacts locally. It is not Phase 6C
AWS/staging acceptance and does not unlock Phase 6D. Real SES, IAM, ALB, CloudWatch, DNS,
staging rollback, and production evidence remain external environment gates.

## Results

| Surface | Result | Evidence |
| --- | --- | --- |
| V1.1 regression | PASS | Ruff check/format, mypy, 33 pytest tests; 72% total coverage |
| V2 API static checks | PASS | Ruff check/format and mypy |
| V2 API regression | PASS | 224/224 pytest tests against real PostgreSQL and MinIO |
| V2 web verification | PASS | ESLint, TypeScript, 41/41 Vitest tests, Next.js production build |
| Browser regression | PASS | 22 Playwright tests passed; 2 live-only cases skipped by the default suite |
| Live application flow | PASS | Real FastAPI + PostgreSQL + MinIO processed a generated 1.87 MB CSV through upload, quality review, chart creation, PNG export, and download |
| Private workbook case | NOT RUN | No `LABVIZ_E2E_FILE` was supplied; the public generated-file live case passed |
| Terraform | PASS | `fmt`, offline `init`, and `validate` for all four roots; staging and production mock tests passed |
| Repository hooks | PASS | All pre-commit hooks passed |
| Production images | PASS | API, web, and logical-backup Docker images built successfully |
| Container smoke | PASS | Built API `/health` and built web `/` both returned HTTP 200 |

The first API integration invocation used `labviz-minio-local` instead of the Compose
secret `labviz-minio-local-only`, so MinIO correctly returned HTTP 403. The credential
was corrected and the complete 224-test suite was rerun successfully. No application
change was needed.

## Local artifacts

- `labviz-api:local-integrity`
- `labviz-web:local-integrity`
- `labviz-logical-backup:local-integrity`

The PostgreSQL and MinIO test containers were stopped after verification. Named local
development volumes were retained so the services can be restarted without rebuilding
the data layer.

## Acceptance boundary

This local pass proves that the checked-in application, Terraform syntax, browser flow,
and image builds are internally coherent. It does not prove that AWS resources exist or
that public DNS, TLS, SES delivery, alarms, deployment, rollback, quota, privacy, load,
or recovery acceptance has occurred. Those claims require the evidence listed in the
Phase 6C and Phase 6D contracts.
