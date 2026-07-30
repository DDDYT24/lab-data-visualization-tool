# Phase 4 Identity and Project Lifecycle Persistence

Status: implemented and verified on 2026-07-30.

Phase 4 moves identity, ownership, Save to Cloud, saved-project history, Duplicate, delete,
restore, and workspace assembly onto the selected PostgreSQL backend. It preserves the existing
`/api/v1` JSON request and response documents, keeps the SQLite reference repository intact, and
does not runtime-dual-write the two backends.

## Migration

The migration chain is:

1. `0001_core_foundation`
2. `0002_phase2_persistence`
3. `0003_quality_cleaning_lineage`
4. `0004_identity_project_lifecycle`

Migration `0004` adds GuestSession and persisted authentication state, replaces the Project guest
token column with `guest_session_id`, adds claim/provenance/lifecycle/idempotency records, adds
StoredObject deduplication and GC metadata, permits multiple DatasetVersions to reference one
StoredObject, and records reused ProcessingRuns explicitly. Existing StoredObjects are backfilled
to user, GuestSession, or isolated legacy scopes. Alembic upgrade, empty/pre-sharing downgrade,
re-upgrade, and model-drift checks are automated.

Once Phase 4 has created shared StoredObject references, a direct downgrade to Phase 3 is not a
lossless production rollback because Phase 3 requires one StoredObject per DatasetVersion. A
production rollback therefore needs a deliberate object-unsharing data migration; the normal
recovery path is a forward fix. The migration does not silently delete or fabricate object bytes.

## Identity and ownership

- A temporary web Project is owned by an active GuestSession and expires normally.
- Save validates and locks the Project, validates the GuestSession bearer digest, records one
  immutable ProjectClaim, assigns the authenticated User, changes the Project to `saved-cloud`,
  and clears temporary expiry in place.
- Repeated Save by the same owner is idempotent. A different user or GuestSession cannot claim the
  Project.
- Authentication challenges, sessions, request-rate records, and Users use PostgreSQL whenever
  PostgreSQL project persistence is selected.

Save does not copy Parquet or replace DatasetVersion, ProjectRevision, or ChartSpecRevision.
Lifecycle records flag external retention reconciliation as post-commit work rather than claiming
object-store operations are part of the database transaction.

## Duplicate closure and provenance

Duplicate creates a new saved Project from exactly the source Project's current ProjectRevision
closure. It creates new SourceFile, Dataset, DatasetVersion, QualityReport/Finding,
CleaningDecisionSet/Decision, ChartSpecRevision, and ProjectRevision identities, but does not copy
unrelated history, shares, or publication exports.

Every reused analytical run has `execution_mode = reused-result`, an origin run UUID snapshot, and
where available a nullable origin-run FK. ProjectOrigin uses nullable `ON DELETE SET NULL` source
FKs plus immutable project/revision UUID snapshots, so the duplicate never prevents source purge.
An optional HTTP `Idempotency-Key` makes Duplicate replay return the first target Project.

## StoredObject and GC protocol

- Object keys are content-addressed by dedup scope and byte SHA-256 and never contain a Project ID.
- New temporary objects use `guest:<guest_session_id>`; user-owned objects use
  `user:<owner_user_id>`.
- Reuse requires matching backend, scope/authorized ownership transition, purpose, media type,
  format contract, byte hash, byte size, encryption-key boundary, and `available` status.
- Reusing an existing StoredObject locks that row before creating the new foreign-key reference.
- The previously approved staged first-write flow still owns a newly inserted `pending` row until
  confirmation; it is not eligible for reuse or GC and is recovered by reconciliation.
- GC locks the same row, counts real SourceFile and DatasetVersion FKs, and may enter `deleting`
  only with zero references. Physical deletion happens after commit and is retryable.
- Soft-deleted saved Projects keep every object FK for the full 24-hour recovery period.

## Project lifecycle and workspace

- History lists only the authenticated user's active saved Projects.
- Delete sets `deleted_at` and exactly `purge_after = deleted_at + 24 hours`.
- The recovery list shows only owner-authorized Projects whose recovery window is still open.
- Restore locks the Project and clears deletion fields only before expiry.
- The lifecycle worker purges expired temporary or soft-deleted Projects and then runs FK-based
  object GC.
- Workspace is rebuilt on every request from `Project.current_revision` and validates the active
  DatasetVersion, QualityReport, CleaningDecisionSet, and ChartSpecRevision graph.
- Restoring a historical revision only moves `Project.current_revision`; it never mutates the
  historical immutable objects.

## API coverage

Migrated or completed on PostgreSQL:

- email-code authentication and session lookup;
- `POST /api/v1/projects/{projectId}/save`;
- `GET /api/v1/projects`;
- `POST /api/v1/projects/{projectId}/duplicate`;
- `DELETE /api/v1/projects/{projectId}`;
- `GET /api/v1/recovery/projects`;
- `POST /api/v1/projects/{projectId}/restore`;
- `POST /api/v1/projects/{projectId}/revisions/{revisionNumber}/restore`;
- `GET /api/v1/projects/{projectId}/workspace`.

Sharing and permanent publication-export persistence remain outside Phase 4. The SQLite reference
Repository remains unchanged; its adapter only satisfies the common application boundary.

## Verification

Automated coverage includes migration upgrade/downgrade/drift, GuestSession isolation, claim and
Save idempotency, cross-domain content-key isolation, minimal Duplicate closure, reused-result
provenance, immutable origin snapshots, duplicate request replay, 24-hour delete/restore,
shared-object retention and final GC, workspace reconstruction, historical revision restore, a
real PostgreSQL API flow, SQLite/PostgreSQL contract regression, Ruff, and MyPy.

## Deferred after Phase 4

- revision-pinned sharing and permanent publication exports;
- selected SMTP provider integration and delivery monitoring;
- leased multi-host reconciliation/GC with retry budgets, quarantine, metrics, and alerts;
- exact pandas/PyArrow writer provenance and a future Parquet schema-v2 policy;
- cloud provider, region, quota, KMS, backup, and compliance production decisions;
- Windows local/cloud bidirectional synchronization.
