# LabViz V2.0 Database and Storage Design

**Status:** Production data route confirmed; phase 1 database foundation implemented, business cutover pending
**Decision date:** 2026-07-29
**Phase 1 implementation:** 2026-07-29
**Product source of truth:** [`prd.md`](prd.md)
**Architecture source of truth:** [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
**Design process:** [`DATABASE_PLAN.md`](DATABASE_PLAN.md)

## 1. Confirmed Production Direction

The website production path uses:

- PostgreSQL for users, ownership, project lifecycle, immutable revisions, processing lineage,
  permissions, shares, and artifact metadata.
- S3-compatible object storage for canonical datasets and downloadable artifacts.
- Parquet for immutable parsed and derived tabular dataset versions.
- A durable processing worker backed initially by PostgreSQL job state. Redis or a separate
  queue is introduced only after measured concurrency requires it.
- SQLAlchemy as the persistence boundary and Alembic for every schema change.

The existing FastAPI and SQLite service remains a contract and test reference. SQLite plus
local files or Parquet is the intended direction for the deferred Windows local mode, not the
production website database.

The phase 1 implementation adds the PostgreSQL connection/session boundary, provider-neutral
object-storage interface, the first nine metadata and lineage tables, and reversible Alembic
migration `0001_core_foundation`. It intentionally does not route existing `/api/v1` business
operations through PostgreSQL yet.

TimescaleDB is not required because LabViz currently imports bounded tabular files rather than
continuous time-series streams. DuckDB may later be used as an embedded analytical execution
engine, but it is not the cloud system of record.

## 2. Confirmed Business Decisions

### 2.1 Source retention

The PRD takes precedence: cloud uploads do not retain the original binary file after successful
parsing and integrity validation.

LabViz retains:

- immutable source metadata, including original name, media type, byte size, sheet selection,
  header row, and SHA-256;
- the immutable parsed baseline as DatasetVersion 1 in Parquet;
- the parser, algorithm, and code versions required to explain how that baseline was produced.

This supports reproducibility from the parsed baseline. It does not claim byte-identical
re-parsing of the original XLSX or text file after the binary has been deleted.

Temporary raw upload objects must be deleted immediately after the parsed baseline has been
committed successfully. Failed and abandoned uploads receive a short lifecycle expiry as a
safety net.

### 2.2 Project, Experiment, and run semantics

`Project` remains the aggregate root for the current V2.0 workflow. A project owns its source
metadata, logical dataset, dataset versions, processing history, cleaning decisions, chart
revisions, exports, and share links.

`Experiment` and `ExperimentRun` are not introduced in the current schema or UI because the
approved product flow does not yet collect their identity, hierarchy, or repeated-acquisition
semantics. Creating them now would produce speculative tables and duplicate the meaning of
Project and Dataset.

`ProcessingRun` is introduced now and means a software execution such as parse, profile, clean,
analyze, or export. It must not later be reused to mean a physical laboratory run.

The current product does need an optional project/experiment description because the PRD
allows it on shared charts. This is a `Project.description` field, not a reason to create an
Experiment entity.

Experiment and physical ExperimentRun become eligible only when a confirmed workflow requires
one or more of the following:

- multiple acquisitions grouped into one scientific experiment;
- multiple source files or instruments in one project;
- run-to-run comparison, replicate identity, or batch metadata;
- experiment-level permissions or reporting distinct from project ownership.

### 2.3 Saved-project deletion and recovery

Deleting a saved project performs a soft delete:

- `deleted_at` is set immediately;
- `purge_after` is set to 24 hours after deletion;
- normal project, export, and share access becomes unavailable immediately;
- the owner may restore the project during the 24-hour recovery window;
- restoration reactivates share links that were active before deletion, except links that were
  explicitly revoked;
- after `purge_after`, a worker permanently deletes dependent database records and stored
  objects.

Temporary guest projects are not placed in the recovery area. They follow their two-hour
expiry and are physically purged.

Disaster-recovery backups are not a user-facing extension of the 24-hour recovery window.

### 2.4 Share version and lifetime

A share link is pinned to one immutable ProjectRevision. Later edits to the working project do
not change an existing shared result.

- Default lifetime: no automatic expiry for a saved project.
- The owner can revoke the link at any time.
- The owner can create a new link for a newer revision or explicitly move a link to a new
  revision through a future audited operation.
- The raw source file is never exposed.
- The database stores only a digest of the share token.

### 2.5 Export retention

Exports belonging to a saved ProjectRevision are retained while the saved project exists. They
are not subject to the two-hour temporary export TTL.

- A saved export is immutable and addressed through StoredObject metadata.
- An export records its DatasetVersion, cleaning decision set, ChartSpec revision, renderer
  version, format, dimensions, DPI, checksum, and creation time.
- When a saved project is soft-deleted, its exports become inaccessible immediately and are
  physically deleted after the 24-hour recovery window.
- Guest-project exports expire with the temporary project.

“Permanent” therefore means durable for the lifetime of the saved project, not retention after
the user has permanently deleted the project.

## 3. ProjectSpec v1

### 3.1 What the current gap means

The current frontend and API can reopen a workspace, but project state is split across source
JSON, quality JSON, mutable cleaning rows, mutable ChartSpec JSON, and export records. There is
no single versioned contract that identifies exactly which dataset, decisions, and chart make up
a reproducible project state.

`ChartSpec` solves only the visualization portion. It cannot identify the parsed dataset or the
cleaning revision used to produce the figure.

### 3.2 Resolution

Define the same `ProjectSpec` version in frontend Zod and backend Pydantic. API contract tests
must validate that both representations accept and reject the same fixtures.

Conceptual ProjectSpec v1:

```json
{
  "schemaVersion": 1,
  "projectId": "uuid",
  "title": "Thermal response",
  "description": "Optional experiment description",
  "source": {
    "sourceFileId": "uuid",
    "datasetId": "uuid",
    "datasetVersionId": "uuid",
    "sheetName": "Measurements",
    "headerRow": 1
  },
  "cleaning": {
    "decisionSetId": "uuid",
    "revision": 3
  },
  "chart": {
    "schemaVersion": 1
  }
}
```

The abbreviated `chart` value above represents the complete validated ChartSpec. Its existing
`export` section remains the project-specific export configuration; ProjectSpec does not store a
second conflicting copy. Browser-level default preferences remain a separate UserPreference
concern.

ProjectSpec is a domain model and API DTO, not the database schema. A ProjectRevision stores an
immutable ProjectSpec snapshot for exact contract reproduction while also storing relational
foreign keys to DatasetVersion, CleaningDecisionSet, and ChartSpecRevision. This intentional
redundancy provides both referential integrity and an exact historical API snapshot; both forms
are written in one transaction and revisions are never updated in place.

Future ProjectSpec changes require an explicit schema version and a tested `vN -> vN+1`
migration function. The frontend must never read database entities directly.

## 4. Initial Persistent Model

### Core relational entities

| Entity | Responsibility |
| --- | --- |
| `users` | Normalized email identity and account lifecycle. |
| `projects` | Ownership, title, description, storage mode, current revision, activity, soft deletion, and purge time. |
| `project_revisions` | Immutable ProjectSpec snapshot and references to the active data, decisions, and chart revisions. |
| `source_files` | Original file metadata, SHA-256, parser selection, and proof that the temporary binary was deleted. |
| `datasets` | Stable logical dataset identity created from a source selection. |
| `dataset_versions` | Immutable parsed or derived tabular versions stored as Parquet objects. |
| `processing_runs` | Operation, input, parameters, algorithm/code version, status, timing, error, and output. |
| `quality_reports` | Profile result tied to one DatasetVersion and ProcessingRun. |
| `quality_findings` | Stable findings that cleaning decisions can reference. |
| `cleaning_decision_sets` | Immutable, numbered snapshots of user-approved decisions. |
| `cleaning_decisions` | Finding/action pairs belonging to one decision set. |
| `chart_spec_revisions` | Immutable validated ChartSpec JSON bound to data and cleaning revisions. |
| `exports` | Durable or temporary artifact metadata and reproducibility references. |
| `share_links` | Hashed access token, fixed ProjectRevision, download permission, revocation, and optional expiry. |
| `stored_objects` | Object key, purpose, byte size, media type, SHA-256, encryption key ID, lifecycle, and deletion state. |

Authentication challenges, sessions, and abuse-limit records are operational data. They remain
separate from the scientific lineage model and use short lifecycles.

### Version rules

- DatasetVersion, ProjectRevision, CleaningDecisionSet, and ChartSpecRevision are immutable.
- User edits create a new revision rather than updating history in place.
- A ProcessingRun records exactly one operation and may produce a new DatasetVersion or another
  derived artifact.
- Dataset content and exports stay outside PostgreSQL. PostgreSQL stores verified object keys and
  checksums only.
- JSONB is limited to validated specifications, parameters, column schemas, and bounded derived
  details. Large row arrays are not stored in JSONB.
- All timestamps use UTC. IDs use UUIDs. Object integrity uses SHA-256.

## 5. Production Safety Baseline

These controls must be designed before the backend storage closure, even when the exact cloud
provider is selected later.

### Quotas

- Keep the approved 50 MB cloud upload limit.
- Implement quota policy and usage accounting as configurable backend capabilities, not hard-coded
  plan entitlements.
- Exact saved-project count, total storage, and paid-plan values remain a product decision before
  public launch.

### Data region

- Use one deployment region initially.
- PostgreSQL, object storage, backups, and workers must be co-located in that region.
- Do not promise a specific residency jurisdiction until the provider and target market are
  approved.

### Encryption and secrets

- Require TLS for browser, API, database, and object-storage traffic in production.
- Require managed encryption at rest for PostgreSQL, objects, and backups.
- Use separate encryption keys and credentials for development, staging, and production.
- Store secrets in the deployment secret manager, never in Git or database rows.
- Record the storage encryption key identifier in metadata; do not store raw key material.
- Start with provider-managed KMS keys. Customer-managed keys are introduced only when a contract
  or compliance requirement justifies their additional operations burden.

### Backup and recovery

- PostgreSQL point-in-time recovery window: at least 7 days.
- Daily logical database backup retention: 30 days.
- Saved object versioning or an equivalent recoverable-object policy must protect against
  accidental administrative deletion.
- Run a restore drill before production launch and at least quarterly afterward.
- Backups inherit the same encryption and region requirements as primary data.

### Compliance boundary

The first production baseline includes data minimization, access control, audit events, documented
retention, deletion, and incident recovery. LabViz must not claim HIPAA, GxP, GLP, GDPR, PIPL, or
other regulated certification without a confirmed target use case, provider review, and legal or
compliance approval.

## 6. Migration and Cutover Direction

The reference SQLite database currently contains no user business records, so the production
schema can begin cleanly without a one-time user-data migration.

Implementation order:

1. Add SQLAlchemy persistence interfaces and Alembic without changing the frontend API contract.
2. Create PostgreSQL core metadata and revision tables through the first migration.
3. Add the object-storage adapter, temporary upload lifecycle, SHA-256 validation, and Parquet
   DatasetVersion writing.
4. Add ProjectSpec v1 in Zod and Pydantic with cross-contract fixtures.
5. Replace mutable decisions and ChartSpec writes with immutable revisions.
6. Move saved exports and canonical datasets out of SQLite BLOBs.
7. Pin shares to ProjectRevision and implement 24-hour saved-project recovery.
8. Validate expiry, recovery, reproducibility, concurrent writes, authorization, backup restore,
   and real 50 MB boundary behavior.
9. Remove the SQLite BLOB production path after parity tests pass. Keep SQLite as a local/test
   adapter only.
