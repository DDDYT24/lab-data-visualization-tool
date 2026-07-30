# Phase 2 Project Persistence Contract

**Status:** Phase 2A and the approved Phase 2B vertical slice implemented
**Date:** 2026-07-30

## Recovery boundary

The PostgreSQL column `projects.purge_after` is the current physical name for the saved-project
`restore_expires_at` concept. The database check constraint guarantees only these facts:

- a non-deleted project has both `deleted_at` and `purge_after` set to `NULL`;
- only a `saved-cloud` project may enter the recovery area;
- `purge_after = deleted_at + INTERVAL '24 hours'`.

The constraint does not authorize a restore. Normal project queries exclude every row with a
non-null `deleted_at` immediately. The restore service uses a locked query and permits recovery
only while `purge_after > current UTC time`. A later cleanup worker will purge rows and objects
when `purge_after <= current UTC time`.

## Repository and Unit of Work boundaries

`ProjectStore` is the application-facing contract used by the migrated API routes. The existing
SQLite repository is unchanged and is exposed through `SqliteProjectStore`. PostgreSQL uses:

- `SqlAlchemyProjectRepository` for database-only aggregate queries;
- `SqlAlchemyUnitOfWork` for exactly one caller-controlled PostgreSQL transaction;
- `PostgresProjectStore` for application operations and database/object-store coordination.

`LABVIZ_PERSISTENCE_BACKEND=sqlite|postgresql` selects one project backend. The four migrated
routes and their processing-job support write only that backend. There is no SQLite/PostgreSQL
project dual-write. Authentication, quality editing, cleaning, sharing, and exports remain on the
SQLite reference path until their own migration slices are implemented.

The current POST contract has no idempotency key, so two intentional HTTP uploads create two
projects even when their bytes match. Idempotency applies to retrying completion for the same
Project and ProcessingRun: row locks, immutable revision constraints, and content-addressed object
keys ensure that a retry creates at most one baseline DatasetVersion and one initial revision.

## Object-state and compensation flow

| State | Object storage | PostgreSQL | Recovery action |
| --- | --- | --- | --- |
| Staged | Private `.staging` key, hash and Parquet verified | No object row yet | Database failure discards staging |
| Pending | Staging key retained | StoredObject=`pending`; immutable revisions committed | Confirm can be retried after restart |
| Available | Final immutable key exists | StoredObject=`available`; staging key cleared | Project becomes readable |
| Confirmation uncertain | Staging or final key may exist | StoredObject remains `pending` | Idempotent confirm verifies final hash, then marks available |

The database transaction never attempts to roll back object storage. It records enough staging
metadata to make confirmation retryable. A failure before database commit discards the staged
object. A failure after database commit leaves a recoverable pending record; startup recovery
confirms the same key and completes the ProcessingRun.

## ProjectSpec single source

`labviz_api.project_spec.ProjectSpecV1` is the authoritative contract source. The generation
script exports `contracts/project-spec-v1.schema.json`. The frontend constructs its Zod validator
from that generated JSON Schema. Pydantic and Zod both run against the same valid and invalid
fixtures under `contracts/fixtures`.

ProjectSpec v1 stores immutable Project, SourceFile, Dataset, DatasetVersion, optional cleaning
revision, and complete ChartSpec references. `cleaning: null` means that no immutable cleaning
decision set has been created yet; it is not a placeholder identifier.
