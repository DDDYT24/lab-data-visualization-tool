# Phase 6B Public-Edge and Account Completion

**Status:** Phase 6-PRE accepted. AWS-independent implementation is active under
[`PHASE6_AWS_DEPENDENCY_MATRIX.md`](PHASE6_AWS_DEPENDENCY_MATRIX.md); final Phase 6B acceptance
remains pending real SES and staging ALB evidence.

Phase 6B makes the existing application safe and complete at the public application boundary. It
does not provision long-lived AWS infrastructure; Phase 6C owns Terraform and live service
deployment. Phase 6B must preserve `/api/v1` compatibility, immutable ProjectRevision and share
semantics, PostgreSQL as the production coordination store, and SQLite as the local/reference
adapter.

## Local execution status

- [x] **6B-1:** implemented and locally verified; real staging ALB chain evidence remains pending.
- [ ] **6B-2:** not started.
- [ ] **6B-4:** not started.
- [ ] **6B-3 offline portion:** not started; live AWS evidence remains mandatory.

Exact immutable commit IDs and rerun evidence are recorded in the local acceptance report after
the independent unit commits exist.

## Admission and Git object

- The accepted Phase 6-PRE closeout commit must be the direct base, contain Phase 6-PRE-2, and link
  a Phase 6-PRE-3 `PASS` report before work starts.
- Record the exact base and candidate commit for every numbered unit below. Each unit is one local
  commit, is verified before the next unit starts, and is never pushed by an implementation agent.
- Final Phase 6B acceptance is read-only over the complete ordered 6B range and follows
  [`AGENT_ACCEPTANCE_STANDARD.md`](AGENT_ACCEPTANCE_STANDARD.md).
- If AWS CLI access, the selected SES Region, a verified sender identity, or a real PostgreSQL 17
  dependency is unavailable, the affected acceptance claim is `BLOCKED`; no mock may replace it.
- The temporary local execution order and the prohibition on entering Phase 6C/6D early are defined
  by [`PHASE6_AWS_DEPENDENCY_MATRIX.md`](PHASE6_AWS_DEPENDENCY_MATRIX.md).

## 6B-1 — trusted public client identity

The API must derive one canonical abuse-control client identity without trusting arbitrary request
headers:

- accept proxy headers only when the immediate peer is inside an explicit production trusted-proxy
  CIDR allowlist and the configured proxy-hop count is positive;
- reject globally routed/default trusted-proxy ranges in production; the allowlist represents only
  the private network path to the application;
- use the nearest untrusted address from the right side of the ALB `X-Forwarded-For` chain when the
  ALB is configured in append mode;
- normalize IPv4, IPv4-with-port, bracketed IPv6, and IPv4-mapped IPv6 before use;
- ignore proxy headers from an untrusted peer and reject malformed, ambiguous, or insufficient
  trusted-hop chains without falling back to a caller-controlled value;
- store only a keyed digest of the normalized address in authentication-abuse records. A dedicated
  environment-backed key must be at least 32 bytes and must not reuse share-token material;
- retain a deterministic development/test path that does not make production trust permissive.

Required negative tests cover forged leftmost and rightmost values, duplicate headers, excessive
hop counts, malformed ports, IPv6, an untrusted direct peer, and a missing production trust
configuration. Live ALB header behavior is rechecked after Phase 6C staging apply.

## 6B-2 — atomic multi-host authentication limits

Replace count-then-insert authorization with an atomic PostgreSQL limiter:

- migration `0009` introduces bounded client and normalized-email buckets with a unique scope,
  keyed identity, and database-time window;
- create or lock both buckets in a deterministic order, evaluate both limits, and update them in one
  transaction so concurrent requests cannot over-admit or partially consume one dimension;
- all production window decisions use PostgreSQL time; application-server time is not authoritative;
- limits and window length are positive production settings with explicit defaults and upper bounds;
- database or limiter failure returns a stable unavailable response and never fails open;
- cleanup is a bounded leased Worker responsibility and cannot delete the active window;
- SQLite keeps equivalent local/reference semantics without becoming the production proof.

Acceptance requires genuinely overlapping requests from separate connections/processes against
PostgreSQL 17, exact admitted-count assertions at the limit, boundary-window tests, restart tests,
and proof that removing the atomic guard makes the concurrency test fail.

## 6B-3 — Amazon SES delivery and feedback contract

Production email uses an `EmailSender` adapter backed by the SES v2 API and the ECS task role.
Development may keep console delivery and non-production compatibility tests may keep SMTP, but
production must not require static SES SMTP credentials.

Required behavior:

- production requires an explicit SES Region, verified sender, and configuration-set name;
- send one recipient per request, attach non-PII purpose tags, and record the returned SES message
  identifier without logging verification codes or raw credentials;
- use the standard AWS credential chain and least-privilege `ses:SendEmail`; static access keys are
  prohibited in settings, task definitions, Git, and CI;
- delete an unsent authentication challenge on synchronous delivery failure and return the existing
  stable `email-delivery-failed` contract;
- configure-set events must cover sends, deliveries, rejects, rendering failures, delays, hard
  bounces, and complaints; Phase 6C codifies the SES/SNS/CloudWatch resources in Terraform;
- account-level suppression and operational alarms are mandatory before public launch.

Acceptance uses the real SES mailbox simulator or verified test recipient in `ap-southeast-1`,
observes accepted delivery plus bounce/complaint events, proves that the console sender cannot be
selected in production, and verifies that API/Worker task roles receive no unrelated permissions.

## 6B-4 — authenticated project-description editing

Add a plain-text description editor without weakening revision immutability:

- only the authenticated owner of an active saved project may edit;
- an edit creates a new immutable ProjectRevision and atomically advances the working project;
- existing revision-pinned shares and publication exports remain unchanged;
- duplicate, delete/restore, history, workspace, and shared-chart reads retain their existing
  semantics;
- the API and generated frontend contract use an additive field/endpoint and one documented UTF-8
  length limit; HTML is never interpreted;
- the bilingual Web UI exposes saving, empty, loading, conflict, unauthorized, deleted, and
  recoverable-error states.

Acceptance covers PostgreSQL and SQLite contract parity, concurrent stale edits, authorization,
share immutability, duplicate/delete/restore behavior, schema fixtures, Vitest, and Playwright.

## Required verification

From `V2.0/api`, with the shared PostgreSQL 17 and pinned MinIO services running:

```powershell
ruff check labviz_api tests migrations scripts
ruff format --check labviz_api tests migrations scripts
mypy labviz_api tests migrations scripts
python -m pytest
python -m alembic upgrade head
python -m alembic check
```

Migration acceptance must additionally perform populated `0008 -> 0009`, downgrade to `0008`,
re-upgrade, schema-drift, concurrent limiter, and cleanup-boundary probes on an isolated database.

From `V2.0/web`:

```powershell
npm run verify
npm run test:e2e
```

Run the real SES probe added by 6B against the selected test identity without printing recipients,
message bodies, credentials, or authorization URLs. Run repository hooks, JSON/YAML and Markdown
link checks, container builds, `git diff --check`, and final status/scope audits. No new skip or
`xfail` is permitted unless this document is amended before implementation.

## Prohibited scope and rollback

Phase 6B must not add Redis, Kafka, Celery, billing, Terraform resources, a second production data
route, mutable shares, regulated-compliance claims, or long-lived AWS credentials.

Rollback is the reverse ordered deployment of 6B-4 through 6B-1. Code requiring migration `0009`
must be removed before downgrading to `0008`; downgrade is allowed only after the active and retained
limiter rows are confirmed disposable. SES application rollback restores the previous sender only
in non-production; production fails closed rather than reverting to console delivery.
