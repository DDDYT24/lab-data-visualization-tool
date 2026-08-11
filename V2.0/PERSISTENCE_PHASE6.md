# Phase 6 Production Runtime and Operations

**Status:** Phase 6-0 and Phase 6A were verified on 2026-08-09. Phase 6-PRE was accepted on
2026-08-11; Phase 6B local implementation is active under the AWS dependency matrix. See
[`PHASE6_PRE_ACCEPTANCE.md`](PHASE6_PRE_ACCEPTANCE.md).

Phase 6 turns the verified PostgreSQL/S3 application into a production-deployable website without
changing `/api/v1`, scientific contracts, immutable revision semantics, or the SQLite reference
adapter. It does not add Redis, Kafka, Celery, billing, regulated-compliance claims, or speculative
scientific entities.

## Phase 6-0 — fact reconciliation and decisions

The repository establishes these current facts:

- PostgreSQL is the production metadata and coordination store; SQLite remains local/reference.
- S3-compatible storage is the production object route. PostgreSQL dataset versions and
  publication exports already use `StoredObject`; the older publication-export backlog item was
  stale. SQLite BLOB exports intentionally remain part of the reference adapter.
- Authenticated saved-project description editing is implemented in Phase 6B-4 with immutable
  ProjectRevision updates, pinned-share semantics, and bilingual plain-text UI behavior.
- PostgreSQL leases already run lifecycle, reconciliation, object GC, staging inventory, and
  metadata cleanup independently from FastAPI. Multi-host authentication still needs a trusted
  client-IP boundary and atomic distributed limiting before API horizontal scaling.
- The SES v2 application adapter is implemented; real delivery/feedback evidence and monitoring,
  quotas, restore drills, and target-market compliance remain later Phase 6 work.

The historical untracked `PERSISTENCE_PHASE5B3_ADMISSION_PRECHECK.md` is superseded by the
implemented and verified [`PERSISTENCE_PHASE5B3.md`](PERSISTENCE_PHASE5B3.md). It is not a current
phase gate and is left untouched because it is outside the tracked project baseline.

## Approved production topology

The initial provider is AWS in one selected region:

| Concern | Decision |
| --- | --- |
| Web, API, Workers | Independent ECS services/tasks on Fargate |
| Routing and TLS | ALB or equivalent managed ingress; HTTPS is the only public production scheme |
| Database | Amazon RDS for PostgreSQL 17, private networking, managed encryption, PITR >= 7 days |
| Objects | Amazon S3, Block Public Access, encryption at rest, versioning, lifecycle policy |
| Email | Amazon SES, implemented and accepted in Phase 6B |
| Secrets and keys | AWS Secrets Manager and KMS; no deployment secret in Git or task JSON |
| Logs and metrics | CloudWatch; alarms and SLOs are Phase 6C |
| Schema changes | A one-shot ECS migration task before an API rollout |

Provider-specific resource creation stays outside application code. The application continues to
depend on PostgreSQL, the `ObjectStorage`/`EmailSender` contracts, and environment-backed configuration.
This keeps another managed provider possible without runtime dual-write or a second scientific
processing path.

References: [Amazon ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html),
[RDS backups](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html),
[S3 encryption](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-encryption.html),
[S3 versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/versioning-workflows.html),
and [Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html).

## Phase 6A — production runtime foundation

Required deliverables:

1. Production settings fail closed unless PostgreSQL and S3 are configured. The API role also
   requires HTTPS public/CORS origins, a production email provider, secure cookies, and an explicit
   share-token key ring; the Worker role does not receive unused email or share-token settings.
2. API readiness verifies both the selected database and object provider. Liveness does not
   depend on external services.
3. API and Worker expose bounded process health checks suitable for container orchestration.
4. Reproducible non-root API and Next.js standalone container images exist. One API image supports
   the web API, one-shot Alembic migration, and each existing Worker task.
5. Deployment examples contain placeholders and secret references only—never credentials.
6. Unit, real PostgreSQL/MinIO integration, container build, frontend, browser, static-analysis,
   hooks, and diff checks pass.

Phase 6A does not provision an AWS account or claim a live production release. Cloud-resource IaC,
SES integration, multi-host abuse protection, backup automation/restore evidence, quotas,
compliance review, dashboards, alerts, and SLOs remain explicitly deferred.

### Phase 6A implementation

- `Settings` validates an explicit API/Worker runtime role and the production PostgreSQL/S3/TLS
  boundary without allowing Local, SQLite, insecure origins, custom S3 endpoints, or default API
  secrets to become production fallbacks.
- `/health` remains dependency-free liveness. `/api/v1/ready` checks PostgreSQL and the configured
  object provider with stable, bounded 503 codes. After Phase 6-PRE-2, the API container calls
  `/health`, the ALB target group calls `/api/v1/ready`, and
  `python -m labviz_api.runtime_health` is a Worker deployment preflight or operator diagnostic,
  not an ECS container-health check.
- The API/Worker image installs the hash-locked Python 3.12 runtime and runs as `labviz`. The Web
  image builds Next.js standalone on Node 24.17 and runs as `node`. Writable local-reference and
  Matplotlib paths stay under the non-root API user's home.
- CI builds both images. ECS API, Worker, and Web templates use immutable image-digest placeholders,
  Secrets Manager references, CloudWatch logging, and role-specific health checks. Worker tasks do
  not receive email-provider or share-token settings.

### Phase 6A verification

The final local matrix used Docker Engine 29.6.2, PostgreSQL 17, and the pinned MinIO image:

- Ruff, format checking, and strict MyPy passed across 65 Python files.
- 165 API tests passed against real PostgreSQL/MinIO; Alembic remained at `0008` head and
  `alembic check` reported no schema drift.
- ESLint, TypeScript, 39 Vitest tests, and the Next.js standalone production build passed.
- Playwright reported 16 passed and two explicitly opt-in live-file/live-API tests skipped.
- Both container images built. API and Web started as non-root users and returned healthy; API and
  Worker probes reached real PostgreSQL/MinIO. A missing Worker database configuration returned a
  non-zero `postgresql-required` result, and production configuration refused a MinIO endpoint.
- Repository hooks, deployment JSON parsing/secret-boundary tests, and `git diff --check` passed.

MinIO verifies the shared S3 provider contract but is not presented as an AWS production test. A
real AWS deployment, SES delivery, managed-resource settings, backups, alerts, and restore drill
remain later-phase acceptance evidence.

## Later Phase 6 ownership

- **Phase 6B:** trusted-proxy client identity, atomic multi-host abuse limits, SES delivery,
  bounce/complaint monitoring, and project-description editing. The exact contract is
  [`PERSISTENCE_PHASE6B.md`](PERSISTENCE_PHASE6B.md).
- **Phase 6C:** IaC for network/ECS/RDS/S3/KMS/Secrets Manager, backup and restore automation,
  CloudWatch dashboards/alarms, runbooks, measured SLOs, CI/CD, and rollback. The exact contract is
  [`PERSISTENCE_PHASE6C.md`](PERSISTENCE_PHASE6C.md).
- **Phase 6D:** configurable project/storage quotas, retention review, target-market privacy and
  compliance assessment, staging load tests, disaster-recovery drill, and launch acceptance. The
  exact contract is [`PERSISTENCE_PHASE6D.md`](PERSISTENCE_PHASE6D.md).

## Phase 6-PRE admission sequence

1. **PRE-1:** track [`AGENT_ACCEPTANCE_STANDARD.md`](AGENT_ACCEPTANCE_STANDARD.md), establish the
   detailed 6B/6C/6D contracts, and update the repository roadmap without changing runtime code.
2. **PRE-2:** separate process liveness from dependency readiness. API container health uses
   `/health`, ALB target readiness uses `/api/v1/ready`, and Worker dependency probes do not create
   restart storms. This boundary is implemented and regression-tested.
3. **PRE-3:** independently rerun the complete repository, real PostgreSQL/MinIO, migration,
   frontend/browser, image/runtime, hook, scope, and cleanup gates. The recorded verdict is `PASS`
   in [`PHASE6_PRE_ACCEPTANCE.md`](PHASE6_PRE_ACCEPTANCE.md).

Phase 6B may not begin until all three admission steps pass. Every numbered implementation unit is
an independent commit and must be accepted against
[`AGENT_ACCEPTANCE_STANDARD.md`](AGENT_ACCEPTANCE_STANDARD.md). A later phase may add stricter
evidence but may not weaken existing API, persistence, integrity, recovery, or security invariants.
