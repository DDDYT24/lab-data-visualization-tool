# Phase 4 Pre-implementation Architecture

Status: approved for implementation on 2026-07-30.

Phase 4 implements the PostgreSQL-backed identity, ownership, project lifecycle, history,
duplicate, restore, and workspace slice without changing the existing `/api/v1` JSON request or
response shapes. The SQLite reference repository remains unchanged and runtime dual-write is
forbidden. Sharing and durable publication-export migration are outside this phase.

## Approved ownership model

- A browser guest is represented by a server-side `GuestSession`. Only a digest of its opaque,
  HttpOnly cookie token is persisted.
- A `temporary-cloud` Project has exactly one owner: `guest_session_id` or `owner_user_id`, and it
  has an expiry.
- A `saved-cloud` Project has an `owner_user_id`, no guest owner, and no temporary expiry.
- Web guest claim and Save to Cloud mutate the existing Project in one locked database
  transaction. DatasetVersion, ProjectRevision, ChartSpecRevision, and Parquet bytes are not
  copied or modified by Save.
- Windows local identity and future local/cloud synchronization remain a separate adapter and
  mapping concern; PostgreSQL does not runtime-dual-write local projects.

## Phase 4 deduplication scope

The first implementation isolates deduplication domains as follows:

- `saved-cloud`: `dedup_scope = user:<owner_user_id>`;
- `temporary-cloud`: `dedup_scope = guest:<guest_session_id>`;
- objects from different users or different GuestSessions must not participate in observable
  cross-domain deduplication.

Reusable immutable object keys are content-addressed within the deduplication scope and must not
depend on the originating `project_id`. Exact physical reuse additionally requires matching byte
SHA-256, size, purpose/media contract, format contract version, encryption boundary, and an
`available` object state. A logical DatasetVersion content hash alone is not sufficient to prove
that two physical Parquet objects are byte-identical.

## StoredObject reference and GC lock protocol

Every new DatasetVersion or future Export reference and the GC worker use the same StoredObject
row-lock protocol:

1. Begin a database transaction.
2. Select the StoredObject row `FOR UPDATE`.
3. A writer may add a reference only while the object is `available`.
4. GC rechecks all authoritative foreign-key references while holding the same lock.
5. GC may transition `available -> deleting` only when no reference exists.
6. After commit, object deletion is attempted outside the database transaction; success marks the
   row `deleted`, while failure remains retryable and is reconciled.

Reference counts may be used only as observability caches. They are never authoritative for
physical deletion. Soft-deleted projects retain their foreign-key references for the complete
24-hour recovery window.

## Duplicate Project closure and provenance

Duplicate creates a new saved Project from the source Project's current ProjectRevision and the
minimum transitive scientific closure needed to reproduce it. It does not copy unrelated history,
shares, or durable publication exports.

The copied closure receives new Project, SourceFile, Dataset, DatasetVersion, ProjectRevision,
ChartSpecRevision, QualityReport, QualityFinding, CleaningDecisionSet, and CleaningDecision
identities. Immutable StoredObject bytes may be reused under the same deduplication scope.

Only the QualityReport, Findings, and DecisionSet reachable from the current ProjectRevision are
copied. Reused results are recorded through `ProcessingRun.execution_mode = 'reused-result'` with
an origin-run UUID; they must not be represented as newly executed analysis.

Duplicate provenance must never prevent source purge. Relational source Project and Revision
foreign keys are nullable with `ON DELETE SET NULL`, while immutable source Project and Revision
UUID snapshots remain after the source aggregate is physically deleted.

## Save to Cloud transaction boundary

The Save database transaction is limited to:

- validating the authenticated User and GuestSession ownership;
- locking and claiming the existing Project;
- changing `storage_mode`, owner, guest owner, and expiry fields;
- writing claim/lifecycle audit records and idempotent outcome state.

Save never copies Parquet and never treats an object-store lifecycle update as part of the
database atomic transaction. Any required post-commit retention or lifecycle update is
idempotently retryable and is repaired by periodic reconciliation.

## Workspace and lifecycle

Workspace is assembled from `Project.current_revision_id` and the referenced immutable
ProjectRevision graph. Saved-project delete sets `deleted_at` and `purge_after = deleted_at + 24
hours`; all references remain in place. Restore only clears those fields after an owner-authorized,
locked time-window check. After the window, a leased worker purges the aggregate and submits
newly unreachable StoredObjects to the locked GC protocol.

## Explicit Phase 4 boundary

- Preserve all existing `/api/v1` JSON request and response structures.
- Do not modify or delete the SQLite reference repository.
- Select exactly one persistence backend; never runtime dual-write.
- Do not migrate sharing or durable publication exports in this phase.
- Do not introduce Redis, Kafka, or microservices.
- Stop after identity ownership, claim, Save, History, Duplicate, Delete, Restore, and Workspace
  form a tested PostgreSQL closure.
