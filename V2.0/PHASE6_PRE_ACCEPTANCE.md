PASS

# Phase 6-PRE Acceptance

- **Accepted on:** 2026-08-11
- **Branch:** local development branch
- **Base:** `f345e1adfc45dfe263ab11521c7969b6b8c7e326`
- **PRE-1:** `c5ce14e985d15c0a0adc3b6a21c6603eae8f699f`
- **PRE-2 candidate:** `96f9a8b409557985127c7e16a83ae28e88da40db`
- **Accepted implementation range:** `f345e1adfc45dfe263ab11521c7969b6b8c7e326..96f9a8b409557985127c7e16a83ae28e88da40db`

The parent chain is exact: PRE-2 directly follows PRE-1, and PRE-1 directly follows the accepted
Phase 6A commit. The range contains only the Phase 6 acceptance contracts, roadmap updates,
health-boundary correction, deployment examples, documentation, and its regression test. It adds
no migration, API/schema change, provider, queue, billing, credential, or Phase 6B implementation.

## Requirement results

- `AGENT_ACCEPTANCE_STANDARD.md` and detailed 6B/6C/6D implementation, evidence, failure, and
  rollback contracts are tracked and linked.
- The API image and ECS API task use dependency-free `/health` for container liveness.
- The deployment contract assigns `/api/v1/ready` to the ALB target group.
- The essential Worker process owns task lifetime; `runtime_health` remains a bounded deployment
  preflight/operator diagnostic and is not an ECS container-health check.
- A real MinIO outage returned liveness `200` and readiness `503`; API Docker health remained
  `healthy`, restart count remained `0`, and readiness returned to `200` after recovery.
- Incomplete production settings exited nonzero with the expected fail-closed validation.

## Independently rerun verification

| Area | Command/evidence | Result |
| --- | --- | --- |
| V2 API static | `ruff check`, `ruff format --check`, `mypy` over `labviz_api tests migrations scripts` | PASS, 65 files |
| V2 API | `python -m pytest` with real PostgreSQL/MinIO | PASS, 165 tests |
| Migration | `alembic upgrade head`, `alembic current`, `alembic check` | PASS, `0008` head, no drift |
| V1.1 | CI Ruff, format, MyPy, `pytest -q --cov --cov-report=term-missing` | PASS, 33 tests |
| Web | `npm run verify` | PASS, ESLint, TypeScript, 39 Vitest tests, production build |
| Browser | `npm run test:e2e` | PASS, 16 tests; 2 documented opt-in live tests skipped |
| Live browser/API | generated 1.87 MB CSV live test against the containerized API | PASS, 1 test |
| Images | API and Web production builds, config inspection, HTTP probes | PASS, UIDs 999/1000, both healthy |
| Worker | `python -m labviz_api.runtime_health` in the API image against real dependencies | PASS |
| Repository | `pre-commit run --all-files`, JSON/YAML, tracked Markdown link checks, `git diff --check` | PASS |

The private-workbook Playwright case was not run because no private workbook was supplied; it is an
explicit opt-in data check, not a product acceptance substitute. The generated-file live flow was
run instead. No unexpected skip, `xfail`, retry, or weakened assertion was introduced.

## Environment and cleanup

- Docker Engine `29.6.2`; PostgreSQL `17.10`; MinIO
  `RELEASE.2025-04-22T22-12-26Z`; Python `3.12.13`; Node `24.17.0`; npm `11.13.0`.
- Deleted the one test project and the dedicated `pre3-runtime/` object prefix created by the live
  browser test.
- Removed only the API/Web verification containers and images and the Compose containers/network
  started for this review. Ports `3000`, `13000`, `18000`, `54329`, `59000`, and `59001` are free.
- Preserved the PostgreSQL and MinIO named volumes; no `docker compose down -v` was used.
- Preserved the pre-existing `labviz-v2-minio-test` container and the two pre-existing untracked
  files `V2.0/PERSISTENCE_PHASE5B3_ADMISSION_PRECHECK.md` and `healing_notes.py`.

Phase 6-PRE is complete. Phase 6B is admitted but was not started by this acceptance.
