# Phase 5B-1 Worker Lease and Fencing Infrastructure

Status: implemented and verified on 2026-07-30.

Phase 5B-1 adds the safe scheduling foundation for later PostgreSQL maintenance. It deliberately
does not purge Projects, delete StoredObjects, remove staging data, use S3, or alter any
`/api/v1` JSON contract. The SQLite reference Repository remains unchanged and there is no
SQLite/PostgreSQL runtime dual-write.

## Migration 0006

`0006_worker_leases` follows `0005_share_publication_exports`; migrations `0004` and `0005`
remain unchanged.

The migration creates `worker_leases`, with one seeded row for each scanner responsibility:

- `pending-reconciliation`;
- `project-lifecycle`;
- `stored-object-gc`;
- `orphan-staging-inventory`;
- `metadata-cleanup`.

Each task row stores the owner, expiry, monotonically increasing fencing token, heartbeat, and
created/updated timestamps. `Project`, `StoredObject`, and `StoredObjectWriteIntent` each receive
the complete work-item lease and retry envelope:

- `lease_owner`, `lease_until`, and `fencing_token`;
- `next_attempt_at`, `last_attempt_at`, and `retry_count`;
- bounded `last_error_code` (128 characters) and `last_error_message` (1024 characters);
- `quarantined_at`.

Checks require paired owner/expiry values, nonnegative fencing/retry counters, and no active lease
on a quarantined row. Claim indexes support the three responsibilities without changing business
identity. A shared database trigger rejects any fencing-token decrease. `0006` also replaces the
`0005` WriteIntent update guard only far enough to permit lease metadata changes while status is
pending; downgrade restores the exact `0005` transition rule.

The migration is reversible on an empty/non-destructively leased database and passes
`0006 -> 0005 -> 0006` plus Alembic schema-drift validation.

## Authoritative lease locations

Task leases coordinate scanners or task shards; they never replace item ownership.

| Responsibility | Authoritative work-item row | Reason |
| --- | --- | --- |
| Pending object confirmation and export finalization | `StoredObjectWriteIntent` | The intent is the immutable GC hold spanning both operations; a second ExportJob lease would create competing authorities. |
| Temporary expiry and saved-project purge | `Project` | Project ownership, storage mode, expiry, delete, and purge state are checked together. |
| StoredObject deletion | `StoredObject` | The same row owns lifecycle state and will be locked for final FK reachability checks in 5B-2. |
| Orphan staging inventory | task lease only in 5B-1 | Provider inventory and cleanup are deferred. |
| Auth/session/idempotency cleanup | task lease only in 5B-1 | Concrete cleanup work-item selection is deferred. |

## Claim, heartbeat, and fencing protocol

Every work claim is a short PostgreSQL transaction:

1. read `clock_timestamp()` from PostgreSQL;
2. select due, unquarantined, eligible rows with `FOR UPDATE SKIP LOCKED`;
3. assign owner and expiry, increment the fencing token, and record the attempt;
4. commit;
5. only then invoke a handler or simulated external I/O.

No database transaction remains open during a handler. The runner uses a separate heartbeat loop
to renew both task and item leases during long work. Renewal, release, and failure recording all
require the current owner, fencing token, unexpired lease, and unchanged business
state. Once a lease expires and another worker increments the token, the old worker cannot update
the row. Nonowners cannot renew or release it.

Eligibility and backoff use PostgreSQL time, never the worker host clock. Retry delay is bounded
exponential backoff. The terminal configured attempt clears the lease and sets `quarantined_at`;
quarantined rows are no longer claimable. Error strings are truncated before persistence and by
the database column bounds.

## Independent runner and safety boundary

`labviz_api.workers.cli` runs independently of FastAPI and supports one-shot or polling operation,
SIGINT/SIGTERM stop requests, natural lease expiry after process loss, configurable batch/lease/
heartbeat/poll/backoff values, and structured logs. Logs contain task/owner/fencing/error-type
metadata only; they exclude share tokens, secrets, object bytes, and object paths.

The default `LABVIZ_WORKER_DESTRUCTIVE_MAINTENANCE=false` is explicit. More importantly, every
5B-1 handler is inspection-only regardless of that flag. The CLI does not claim Project,
StoredObject, or WriteIntent business rows until a real 5B-2 handler is installed, avoiding
no-op retry churn. Current Phase 5A synchronous/startup
WriteIntent recovery is not removed until Phase 5B-2 supplies and verifies the real leased
replacement. Contract-test handlers cannot commit a stale lease result because release and
failure paths are state- and fencing-conditional.

## Verification

The Phase 5B-1 suite covers:

- concurrent claim exclusion and expired-lease takeover;
- stale-worker fencing and the database monotonic-token guard;
- task/item heartbeat, nonowner rejection, and long-I/O renewal;
- proof that the claim transaction commits before the handler begins;
- PostgreSQL-time behavior under a deliberately unusable host clock;
- bounded retry, backoff, error truncation, and quarantine;
- Project, StoredObject, and pending WriteIntent authority selection;
- `0006 -> 0005 -> 0006` migration and schema-drift checks.

Final verification:

- backend and PostgreSQL suites: 85 passed;
- dedicated Phase 5B-1 suite: 14 passed;
- Ruff: all source, test, and Migration lint checks passed;
- MyPy strict mode: 32 source files passed;
- Alembic: `0006 -> 0005 -> 0006` passed and `alembic check` reported no drift;
- frontend: ESLint and TypeScript passed, 8 Vitest files / 33 tests passed, and the Next.js
  production build completed successfully.

## Phase 5B-2 handoff

Phase 5B-2 may now implement real behavior behind these leases. It still must:

- move pending confirmation/finalization into the fenced WriteIntent handler, including
  cross-process recovery, before removing PostgreSQL lifespan recovery;
- recheck every real FK GC root while holding the existing StoredObject row lock, then perform
  physical delete outside the transaction and fence the final state update;
- implement Project expiry/purge selection, pre-delete GC candidate inventory, and idempotent
  retries without bypassing the 24-hour saved-project window;
- inventory and clean orphan staging objects with provider-neutral age and ownership rules;
- define concrete metadata cleanup batches;
- add the final S3-compatible adapter and provider failure tests.

Phase 5B-1 stops before every item above.
