# Phase 6C AWS Infrastructure and Operations

**Status:** Phase 6C IaC, operations, backup/restore, runbook, and CI/CD code is present in the
repository and on `main`. Final acceptance still requires the complete real staging evidence
defined below. Current status is tracked only in [`TODO.md`](TODO.md).

The versioned [`contracts/phase6-gates-v1.json`](contracts/phase6-gates-v1.json) contract makes this
stage depend on the Phase 6B application gate and makes its `PASS` the only path into Phase 6D.

Phase 6C creates the reproducible AWS staging and production runtime for the accepted application.
The selected production/data Region is `ap-southeast-1`. Staging and production must use isolated
Terraform state, databases, buckets/prefixes, secrets, keys, services, and log groups. A missing
domain, DNS authority, SES sender identity, alarm destination, AWS credential, or approved budget
blocks live apply; no value is guessed and no placeholder is presented as deployed evidence.

## Admission and Git object

- The accepted Phase 6B application range plus the tracked phase-boundary correction is the exact
  base. Record the correction commit as the first 6C unit's direct parent, then record every unit's
  base, candidate, and direct parent before acceptance.
- Each numbered unit is an independent local commit and is verified before the next starts.
- Terraform and AWS provider versions are constrained and locked at implementation time; dependency
  upgrades are separate reviewed changes.
- Final acceptance is read-only over the complete 6C range under
  [`AGENT_ACCEPTANCE_STANDARD.md`](AGENT_ACCEPTANCE_STANDARD.md).

## 6C-0 — state, identity, and repository layout

Create this tracked layout under `deploy/aws/terraform`:

```text
bootstrap/
modules/{network,security,data,compute,edge,email,observability,backup}/
environments/{staging,production}/
```

Bootstrap provisions only the encrypted, Block-Public-Access, versioned S3 Terraform-state bucket
and least-privilege state access. The S3 backend uses `use_lockfile = true`, separate environment
keys, and no credentials or secret values in backend configuration, `.tfvars`, plans, outputs, or
Git. Break-glass and deployment access are documented, time-bounded roles; normal CI uses GitHub
Actions OIDC and no static AWS access key.

Required account guardrails include AWS Budgets/cost anomaly notification, CloudTrail, required
resource tags, and an explicit monthly budget owner. State bootstrap and application infrastructure
cannot share a destroy operation.

## 6C-1 — network, edge, and data services

Terraform provisions:

- one VPC spanning at least two Availability Zones;
- public subnets only for the ALB and required egress, private application subnets for ECS, and
  isolated database subnets for RDS;
- least-privilege security groups, S3 gateway endpoint, and the required ECR, CloudWatch Logs,
  Secrets Manager, KMS, and SES connectivity without public RDS/ECS addresses;
- Route 53 records when AWS owns DNS, ACM certificate validation, HTTPS-only ALB listeners, HTTP to
  HTTPS redirect, invalid-header dropping, ALB append-mode forwarding, and WAF fail-closed when WAF
  is enabled;
- RDS PostgreSQL 17 with KMS encryption, TLS enforcement, Multi-AZ production, deletion protection,
  automated PITR of at least seven days, monitoring, and a final-snapshot policy;
- S3 Block Public Access, TLS-only bucket policy, versioning, SSE-KMS, incomplete-multipart cleanup,
  lifecycle controls, and task-role access limited to the selected prefix.

Staging may use smaller capacity but may not weaken identity, TLS, encryption, public-access,
secret, migration, or restore semantics.

## 6C-2 — compute, deployment, and secrets

Provision immutable ECR repositories and independent ECS services for Web, API, and each accepted
Worker task. Requirements:

- images are referenced by digest, run non-root, use read-only roots where compatible, and receive
  only role-specific settings/secrets;
- API container liveness uses `/health`; ALB target readiness uses `/api/v1/ready`; dependency
  outages do not cause container restart storms;
- the essential Worker process owns process lifetime; dependency preflight and Worker heartbeat,
  backlog, lease, retry, quarantine, and failure metrics are monitored separately;
- one-shot Alembic migration runs and succeeds before a service revision that needs it receives
  traffic; migrations never run concurrently in every API task;
- rolling deployment circuit breakers and alarm-based rollback restore the last completed revision;
- API/Worker task roles use only their S3 prefix, KMS key, SES send permission where applicable,
  log/metric actions, and required secret ARNs; Web receives none of those credentials;
- Secrets Manager and KMS separation, rotation ownership, and emergency recovery are documented.

## 6C-3 — observability, backup, restore, and runbooks

Create retained structured logs, dashboards, alarms, and operator runbooks for:

- ALB request volume, p95 latency, target health, 4xx/5xx, and rejected requests;
- ECS desired/running task count, CPU/memory, restarts, failed deployments, and rollback;
- RDS CPU, connections, free storage, latency, failover, and backup failures;
- S3/KMS access failures and Worker reconciliation, backlog, quarantine, and deletion failures;
- SES sends, delivery delays, rejects, bounces, complaints, and reputation risk;
- application readiness, authentication limiting, email delivery, processing failures, and jobs that
  exceed the documented age threshold.

Production objectives are monthly public availability at least 99.9%, RPO at most 15 minutes, and
RTO at most four hours. Phase 6D validates or tightens user-facing latency and processing targets
with representative load evidence.

RDS PITR is supplemented by an encrypted daily logical backup retained for 30 days. Quarterly
restore drills recover PITR and logical backup into isolated infrastructure, restore required S3
object versions, apply migrations, compare schema and durable record/object inventories, exercise
representative downloads/exports/shares, and delete only drill-created resources after evidence is
retained.

Runbooks cover deployment, rollback, migration failure, RDS/S3/SES/KMS outage, task exhaustion,
Worker quarantine, credential rotation, backup restoration, security incident triage, and controlled
destructive-Worker enablement.

## 6C-4 — CI/CD and live staging proof

GitHub Actions must:

1. run the existing backend, frontend, browser, migration, hook, and container gates;
2. scan and sign accepted images, push them to ECR, and retain immutable digests;
3. run `terraform fmt -check -recursive`, initialization, validation, security/static checks, and a
   saved staging/production plan using environment-protected OIDC roles;
4. require the approved production environment before apply;
5. run the one-shot migration, deploy by digest, wait for steady state, and run HTTPS smoke tests;
6. automatically fail and roll back when migration, deployment, alarm, or smoke evidence fails.

Acceptance requires a real staging apply, a post-apply plan with no unexplained drift, private
network reachability proof, TLS/DNS and trusted-client-header proof, forced failed-deployment
rollback, alarm delivery, RDS PITR and logical restore evidence, S3 version recovery, SES event
evidence, and successful Web/API/Worker smoke flows.

The following evidence was intentionally transferred from the Phase 6B application gate because
only the Phase 6C staging environment can produce it. Every item is blocking for Phase 6C `PASS`:

1. accepted SES v2 send through the staging API ECS task role, with MessageId and delivery event;
2. separate SES bounce and complaint simulator events, account suppression, and alarm delivery;
3. deployed-role inspection proving identity-scoped `ses:SendEmail` only on the API role, no email
   permission on Worker roles, and no unrelated permissions;
4. real append-mode ALB forwarding-chain proof for the configured private trusted peer/hop path,
   including rejection of forged and malformed forwarding headers.

These requirements may not be satisfied by mocks, local task-definition examples, policy text, or
provider SDK-model tests. Missing AWS authentication or staging resources returns `BLOCKED` for
Phase 6C acceptance and does not reopen or invalidate the accepted Phase 6B application code.

## Required commands and prohibited scope

The implementation documents the exact locked tool versions and runnable commands. At minimum:

```powershell
terraform fmt -check -recursive
terraform -chdir=deploy/aws/terraform/environments/staging init
terraform -chdir=deploy/aws/terraform/environments/staging validate
terraform -chdir=deploy/aws/terraform/environments/staging plan -out=staging.tfplan
terraform -chdir=deploy/aws/terraform/environments/production init
terraform -chdir=deploy/aws/terraform/environments/production validate
terraform -chdir=deploy/aws/terraform/environments/production plan -out=production.tfplan
```

Saved plans and generated state never enter Git. Phase 6C must not import user-owned resources,
copy local test data into production, create static access keys, enable Worker deletion before the
runbook gate, add queues/billing, or destroy retained production/state resources during tests.

Rollback uses immutable task revisions and Terraform-reviewed changes; state is never edited by
hand. A failed migration stops deployment before traffic moves. Destructive infrastructure rollback
requires an explicit resource-by-resource recovery plan and cannot use broad `destroy` as a release
mechanism.
