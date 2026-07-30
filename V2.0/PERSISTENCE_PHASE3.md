# Phase 3 Quality and Cleaning Persistence Contract

**Status:** Quality report, cleaning decision, and derived DatasetVersion slice implemented
**Date:** 2026-07-30

## Immutable lineage

Migration `0003_quality_cleaning_lineage` adds `quality_reports`, `quality_findings`,
`cleaning_decision_sets`, and `cleaning_decisions`. It also links the active quality report and
decision set from ProjectRevision, and links derived DatasetVersion and copied ChartSpecRevision
to the decision set that produced them.

PostgreSQL rejects updates to DatasetVersion, ProjectRevision, ChartSpecRevision, QualityReport,
QualityFinding, CleaningDecisionSet, and CleaningDecision rows. New scientific state is always an
insert. Project.current_revision remains the mutable pointer used to select one immutable history
snapshot. Project revision restoration changes only that pointer.

## Quality reports and stable findings

Every QualityReport is bound to one immutable input DatasetVersion and one completed `profile`
ProcessingRun. It records profiler name/version, algorithm version, code version, parameters,
completion state, completion time, and the exact API report document.

The API keeps the existing semantic finding identifiers such as `missing:signal`. The database
also assigns an immutable UUID. A finding persists:

- column name, ordinal, and Parquet/Arrow column identity;
- issue kind and severity;
- bounded evidence and affected count;
- up to 100 source-record references;
- each reference's immutable DatasetVersion ID, one-based source ordinal, and SHA-256 row
  fingerprint.

The source reference is therefore not an unqualified frontend row number. Truncation and total
affected count remain explicit.

## Decision semantics and undo

Each PATCH creates a monotonically numbered CleaningDecisionSet unless it is an exact retry of
the current set. Decisions are immutable and reference QualityFinding UUIDs from the bound report.

| Action | Derived Parquet | Chart input |
| --- | --- | --- |
| `ignore` | Record retained | Record retained |
| `exclude` | Record retained | Record excluded |
| `remove` | Record removed | Record excluded |

Every decision set is reapplied to the QualityReport's immutable input DatasetVersion. Derived
versions are siblings of that input rather than destructive descendants of the previous cleaned
copy. Changing `remove` back to `ignore` can therefore restore the source record without editing or
reconstructing old objects.

Applying a decision set creates a `clean` ProcessingRun, a new immutable Parquet object, a new
DatasetVersion with a parent and decision-set reference, a copied ChartSpecRevision bound to that
new version, and a new ProjectRevision. The previous chart and project revisions remain valid.

## Object compensation

Derived Parquet uses the Phase 2 staging protocol: stage, hash and format validation, database
commit as `pending`, idempotent final confirmation, then StoredObject and ProcessingRun completion.
Failure before the database commit discards staging. Failure after commit preserves the pending
row and is recoverable by the same startup reconciliation path.

## Migrated API slice

- `GET /api/v1/projects/{projectId}/quality`
- `PUT /api/v1/projects/{projectId}/quality-rules`
- `PATCH /api/v1/projects/{projectId}/cleaning-decisions`
- `GET /api/v1/projects/{projectId}/exports/cleaned-data.csv`

The cleaned-data path is migrated only as the current cleaning workflow's dataset download. Saved
publication exports, identity, sharing, and cloud project lifecycle are not part of this phase.

## Recorded follow-up designs

- HTTP upload idempotency needs an `Idempotency-Key` record scoped to actor, route, and normalized
  request hash, with replayed status/result and a bounded expiry. It must not deduplicate two
  intentional uploads based only on file SHA-256.
- Pending StoredObject reconciliation should become a periodic leased worker that retries
  confirmation, verifies final hash, records attempts, alarms on age thresholds, and quarantines
  irreconcilable rows. Startup recovery remains the current safety path.
- The Parquet schema version identifies logical writer rules. Exact pandas, PyArrow, and Parquet
  writer versions should be recorded as processing provenance and object metadata. Embedding new
  mandatory fields in Parquet v1 is forbidden; a byte-contract change requires Parquet schema v2.
