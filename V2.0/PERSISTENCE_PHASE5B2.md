# Phase 5B-2 Lifecycle, Reconciliation, and GC Execution

Status: implemented and verified on 2026-08-01.

Phase 5B-2 activates the Phase 5B-1 lease infrastructure for PostgreSQL. It does not change any
`/api/v1` JSON contract, modify the SQLite reference Repository, introduce dual-write, add an S3
provider, or begin Phase 5B-3/Phase 6.

## Migration 0007

`0007_phase5b2_orphan_staging` follows `0006_worker_leases`. Migrations `0004`, `0005`, and
`0006` remain byte-for-byte unchanged.

The only new table is `orphan_staging_candidates`, because safe two-pass inventory must survive
process restarts. Its primary key is the provider staging key. It records bounded integrity
metadata, first/last observations, observation count, retry scheduling, bounded diagnostics, and
quarantine. No public business entity or API schema is added.

The empty-schema path supports `0007 -> 0006 -> 0007`, and Alembic reports no model/schema drift.
Phase 5A's fail-closed downgrade guard remains in force: tests explicitly clear only their isolated
database before a full-chain downgrade; production artifacts are never silently removed.

## Safety and process boundary

PostgreSQL maintenance runs through `python -m labviz_api.workers.cli <task>` as an independent
process. FastAPI's PostgreSQL lifespan no longer executes recovery or cleanup. The old explicit
`cleanup_expired()` compatibility method delegates to the same leased handlers, so there is no
unfenced maintenance path.

Defaults are:

```text
LABVIZ_WORKER_DRY_RUN=true
LABVIZ_WORKER_DELETE_ENABLED=false
```

Project purge, metadata deletion, physical StoredObject deletion, and staging deletion require
`dry_run=false` and `delete_enabled=true` simultaneously. Dry-run may persist staging inventory
observations, but never invokes object deletion, purges a Project, or deletes metadata. Structured
logs report task, count, item kind, fencing token, and reason without tokens, object content, or
sensitive paths.

Pending confirmation/finalization is not destructive maintenance and remains active: it completes
an already persisted, recoverable write. It does not retry chart rendering. Rendering still occurs
synchronously before the ExportJob is created.

## Reconciliation state machine

Export finalization uses `StoredObjectWriteIntent` as its only authoritative work item:

```text
claim pending WriteIntent in a short transaction
-> reconstruct StagedObject metadata
-> confirm outside the transaction
-> lock and fence the WriteIntent
-> lock StoredObject
-> verify key, SHA-256, and size
-> make StoredObject available
-> create immutable PublicationExport if absent
-> complete WriteIntent and ExportJob/ProcessingRun
-> fill eligible immutable ShareExportBinding rows
-> commit once
```

If confirmation succeeds and the process stops before the final transaction, the next Worker sees
the existing final object, verifies identical bytes, and safely completes SQL state. Dataset writes
without a WriteIntent use the pending `StoredObject` as their work item; the Worker confirms the
Parquet object and completes the associated parse/clean ProcessingRun under the same fencing rule.

Failures before confirmation retain staging bytes and retry with bounded backoff. Integrity,
missing-metadata, or permission failures quarantine the authoritative row. WriteIntent remains a GC
root while pending or quarantined and is cleared only in the transaction that establishes the real
PublicationExport reference.

## Project lifecycle ordering

Eligible Project rows are temporary projects past `expires_at` and saved projects past
`purge_after`. Claim and final eligibility use PostgreSQL time. The final purge transaction:

1. validates owner, fencing token, lease, storage mode, and current expiry/delete state;
2. collects every project-scoped StoredObject reference;
3. locks those StoredObject rows with `FOR UPDATE`;
4. records `gc_candidate_at` before any FK cascade;
5. records an immutable purge lifecycle event;
6. deletes the Project and commits.

After the FK cascade is flushed, an unreferenced `pending` object is moved to `deleting` and its
staging key is released to the two-pass orphan inventory. This prevents a purged project's
interrupted upload/export from remaining an uncollectable pending row.

If restore clears `deleted_at/purge_after` before the purge transaction obtains the row, the claim's
state condition fails and purge cannot proceed. If purge owns the row first, restore cannot mutate
a project that has already been permanently deleted. Metadata cleanup removes an expired/revoked
GuestSession only when no Project FK references it, so Guest expiry can never bypass Project purge.

## Authoritative GC reachability

One shared query in `workers/references.py` is used by physical GC and the PostgreSQL compatibility
paths. A StoredObject is reachable if any of these references exist:

- `SourceFile.stored_object_id`;
- `DatasetVersion.stored_object_id`;
- `PublicationExport.stored_object_id`;
- pending or quarantined `StoredObjectWriteIntent.stored_object_id`;
- `ExportJob.pending_stored_object_id`.

Project soft deletion does not remove any FK, so all objects remain reachable throughout the
24-hour restore period. `gc_candidate_at` is only a discovery hint. The scanner also discovers
old `available` objects with no hint and no authoritative reference after the configured safety
age.

## StoredObject deletion and lock protocol

All application paths that reuse an immutable object (dataset creation, cleaning, duplicate, and
publication export) lock the StoredObject and require `status=available`. GC uses that identical row
lock:

```text
short transaction: lock row + recheck all references
-> available to deleting + persist item lease
-> commit
-> external delete (no SQL transaction held)
-> final transaction: verify owner + fencing token + deleting state
-> recheck all references
-> deleted, or restore to available if a reference exists
```

Provider missing/404 is idempotent success. Transient I/O failure leaves `deleting`, releases the
lease, and schedules bounded retry. Permanent permission failure quarantines the row with bounded
diagnostics. A process stop after external deletion but before SQL completion is recovered by the
next attempt's idempotent missing-object result. A stale Worker cannot finalize after another owner
increments the fencing token.

## Orphan staging and metadata cleanup

Staging cleanup never deletes from a single listing. First observation inserts a candidate. A later
inventory must observe identical SHA-256 and size after the configured grace period, and SQL must
show no pending StoredObject using the staging key. Only then may deletion occur. Changed bytes
reset the observation window. Missing provider keys remove stale candidate metadata. Failures use
bounded retry/quarantine and every mutation is fenced by the task lease.

Metadata cleanup uses PostgreSQL time for expired AuthChallenge, AuthSession, IdempotencyRecord,
hour-bounded AuthRequest, and unreferenced expired/revoked GuestSession rows. Operational order is:

```text
pending-reconciliation -> project-lifecycle -> stored-object-gc
-> orphan-staging-inventory -> metadata-cleanup
```

## Verification and fault injection

The dedicated Phase 5B-2 suite covers migration round trips, dataset/export cross-process recovery,
failure immediately before/after object I/O, Project restore/purge races, the complete pre-purge
object closure, historic-orphan discovery, authoritative reference rechecks, missing-object delete,
transient retry, permanent quarantine, two-pass staging cleanup, GuestSession ordering, and dry-run
zero-destructive-effect behavior. Phase 5B-1 tests continue to cover two-Worker claim exclusion,
lease expiry takeover, heartbeat, database-time decisions, and stale-Worker fencing.

Final verification:

- backend: 106 tests passed, including 21 Phase 5B-2 and 14 Phase 5B-1 tests;
- Ruff: all API source, tests, scripts, and migrations passed;
- MyPy strict mode: 34 source files passed;
- Alembic: `0007 -> 0006 -> 0007` passed and `alembic check` reported no drift;
- frontend: ESLint and TypeScript passed, 8 Vitest files / 33 tests passed, and the Next.js
  production build completed successfully.

Phase 5B-3 remains responsible for a final S3-compatible adapter and provider contract tests.
Production KMS, backup, compliance, and formal retention configuration remain Phase 6 work.

## Phase 5B-2.1 admission closure

Phase 5B-2.1 closes the three blockers found by the Phase 5B-3 admission precheck without changing
the database schema, public API contract, SQLite reference Repository, or Phase 5B-3 scope.
Migrations `0001` through `0007` remain unchanged and no `0008` was created.

### Backend-independent chart analysis

`POST /api/v1/projects/{project_id}/chart-analysis` now resolves the configured `ProjectStore`
instead of the SQLite-only `ProjectRepository`. The route first performs the unchanged ownership
and readiness checks, then calls the existing backend-independent `load_chart_dataframe()`
contract. SQLite reconstructs the frame from its reference data, while PostgreSQL resolves the
current immutable DatasetVersion and quality/decision metadata in short transactions and reads the
Parquet object after those transactions have closed. PostgreSQL never falls back to or writes the
SQLite project table.

Real API integration tests prove that a PostgreSQL-created project returns a successful analysis,
an unrelated GuestSession is rejected, a missing project retains `project-not-found`, PostgreSQL
and SQLite success responses have the same JSON shape, and the SQLite reference project table
stays empty in PostgreSQL mode.

### One Dataset finalization authority

The unfenced PostgreSQL `_mark_object_confirmed()` path was removed. Initial Dataset confirmation,
an explicit request retry, the compatibility `recover_pending_objects()` entry point, and the
standalone Worker now all use `LeaseStore` claims and `PendingObjectReconciler` finalization:

```text
PostgreSQL-time pending selection
-> short transaction claims StoredObject and increments fencing_token
-> commit
-> object confirm outside SQL transaction
-> new transaction verifies owner + fencing token + pending state
-> lock StoredObject and validate SHA-256/size
-> finalize DatasetVersion ProcessingRun
-> set StoredObject available and clear lease atomically
```

The request path may target its own pending object, but cannot claim an item already leased by a
Worker. The compatibility recovery entry delegates to `WorkerRunner`; it no longer scans and
mutates pending rows itself. If confirmation has already completed externally, a replacement
Worker repeats the idempotent confirmation and completes SQL. A stale owner cannot finalize,
heartbeat, or release after takeover. A pending DatasetVersion FK remains an authoritative GC root,
and GC cannot claim its `pending` StoredObject.

Real PostgreSQL transaction tests cover request/compatibility-entry exclusion under an active
lease, database-time lease expiry, fencing-token takeover, stale-owner rejection, crash after
external confirm but before SQL finalization, repeated recovery/confirm idempotency, atomic lease
clearing, absence of `available + lease_owner`, and DatasetVersion/GC reachability.

### WCAG Ready status

The workspace Ready Chip now uses the existing outlined success style. This preserves the design
palette and interaction while providing compliant foreground/background contrast. The WCAG test
runs normally—there is no skip, exemption, or test-only style override.

Phase 5B-2.1 final verification:

- backend: 110 tests passed; the Phase 5B-1/5B-2 focused suites passed 38 tests;
- Ruff: all API files passed;
- MyPy strict: 34 source files passed;
- Alembic: `0007 -> 0006 -> 0007` passed; `alembic check` found no schema drift;
- frontend: ESLint and TypeScript passed; 8 Vitest files / 33 tests passed; production build passed;
- Playwright: 15 tests passed and 2 live-API tests were skipped because
  `LABVIZ_E2E_LIVE=1` and the optional private `LABVIZ_E2E_FILE` were not configured;
- WCAG principal-state test passed without exceptions.

Phase 5B-3 still owns S3/MinIO compatibility, multipart upload, staging pagination and stable
cursors, provider metadata, and any storage `backend_name` work. Phase 6 still owns production KMS,
backup, region, quotas, retention, SMTP, and compliance configuration. Other product backlog items
remain unchanged.
