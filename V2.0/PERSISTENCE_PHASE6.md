# Phase 6 Production Runtime and Operations

**Status:** Phase 6-0 approved on 2026-08-09; Phase 6A implementation follows as a separate commit.

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
- `Project.description` already exists in persistence, ProjectSpec, and shared-chart reads. Only
  an authenticated editing API and matching UI remain.
- PostgreSQL leases already run lifecycle, reconciliation, object GC, staging inventory, and
  metadata cleanup independently from FastAPI. Multi-host authentication still needs a trusted
  client-IP boundary and atomic distributed limiting before API horizontal scaling.
- Production SMTP delivery and monitoring, quotas, restore drills, and target-market compliance
  remain later Phase 6 work.

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
depend on PostgreSQL, the `ObjectStorage` contract, SMTP, and environment-backed configuration.
This keeps another managed provider possible without runtime dual-write or a second scientific
processing path.

References: [Amazon ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html),
[RDS backups](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html),
[S3 encryption](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-encryption.html),
[S3 versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/versioning-workflows.html),
and [Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html).

## Phase 6A — production runtime foundation

Required deliverables:

1. Production settings fail closed unless PostgreSQL, S3, HTTPS public/CORS origins, SMTP,
   secure cookies, and an explicit share-token key ring are configured.
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

## Later Phase 6 ownership

- **Phase 6B:** trusted-proxy client identity, atomic multi-host abuse limits, SES delivery,
  bounce/complaint monitoring, and project-description editing.
- **Phase 6C:** IaC for network/ECS/RDS/S3/KMS/Secrets Manager, backup and restore automation,
  CloudWatch dashboards/alarms, runbooks, and measured SLOs.
- **Phase 6D:** configurable project/storage quotas, retention review, target-market privacy and
  compliance assessment, staging load tests, disaster-recovery drill, and launch acceptance.

Every subphase is an independent commit and must be accepted against
[`AGENT_ACCEPTANCE_STANDARD.md`](AGENT_ACCEPTANCE_STANDARD.md). A later phase may add stricter
evidence but may not weaken existing API, persistence, integrity, recovery, or security invariants.
