# Phase 5A Sharing and Publication Export Persistence

Status: implemented and verified on 2026-07-30.

Phase 5A migrates revision-pinned sharing and PNG/SVG/PDF publication exports to the selected
PostgreSQL backend. It preserves every existing `/api/v1` JSON request and response shape, keeps
the SQLite reference Repository unchanged, and never runtime-dual-writes the two backends.

## Migration

The migration chain now ends with `0005_share_publication_exports`. Migration `0005` adds:

- `share_links` and immutable `share_link_events`;
- immutable `share_export_bindings` keyed by ShareLink and format;
- mutable `export_jobs` and immutable `publication_exports`;
- `stored_object_write_intents` for staged-export GC holds;
- GuestSession-aware request hashes and expiry metadata on `idempotency_records`.

PublicationExport has same-project foreign keys to ProjectRevision, DatasetVersion,
CleaningDecisionSet, ChartSpecRevision, and ProcessingRun. An insert trigger verifies those
lineage fields equal the pinned ProjectRevision and that the referenced StoredObject is available
with the same final SHA-256 and byte size. PublicationExport and ShareExportBinding reject updates.

Database scope constraints also make the ExportJob authoritative for the final artifact identity:

- `publication_exports(id, project_id, project_revision_id, format)` references the same four
  columns on `export_jobs`;
- `stored_object_write_intents(export_job_id, project_id)` references the same ExportJob project
  scope;
- the required ExportJob candidate keys are explicit and named consistently in Alembic and the
  SQLAlchemy metadata.

Downgrading to `0004_identity_project_lifecycle` now fails closed before changing schema whenever
there is a PublicationExport, nonterminal WriteIntent or ExportJob, live export StoredObject, or
publication-export idempotency record. Operators must inventory and explicitly complete or clean
both database and object-storage lifecycle first. An empty database remains reversibly migratable.
Migration `0004` remains byte-for-byte identical to the Phase 4 baseline commit.

## Share identity and permission model

Share tokens use the strict form `s1.<public-id-hex>.<base64url-hmac>`. The database stores only
the public UUID, HMAC key version, and SHA-256 digest of the complete token. Validation loads the
row by public ID, selects the recorded key version, reconstructs the complete expected token, and
uses constant-time comparisons for both token and digest.

`LABVIZ_SHARE_TOKEN_KEY_VERSION` selects the only key used for new links.
`LABVIZ_SHARE_TOKEN_KEYS` is a comma-separated `version=secret` validation key ring. An old key
must remain configured while any ShareLink records that version. Phase 6 will move these secrets
to production KMS; Phase 5A does not claim production key management.

Public tokens authorize only the immutable shared view and explicitly enabled figure downloads.
Create, update, and revoke operations independently require an authenticated Project owner.
Invalid, malformed, expired, revoked, deleted-project, and missing public links all resolve to the
same unavailable response. Share routes use `no-referrer` and `no-store` response policies, and
request logging records the route template rather than the raw token.

Project soft deletion leaves ShareLink state and every FK intact but makes public access
unavailable. Restore reactivates links that remain active. Explicitly revoked links remain revoked
after restore. Project purge cascades ShareLink records; immutable event UUID snapshots remain.

## Fixed download bindings

A ShareLink never asks for the latest export. `share_export_bindings` contains at most one row for
each ShareLink and format and binds it to one exact PublicationExport from the same Project and
ProjectRevision. Creation, enabling downloads, or the first successful export fills only missing
bindings while holding the ShareLink row. Later exports never update existing bindings.

The public download permission exposes only bound PNG, SVG, and PDF PublicationExports. It never
exposes source uploads, Parquet, cleaned CSV, another revision, or an unbound later artifact.

## Export and object transaction

ExportJob owns mutable execution and error state. Every real render attempt has a
`ProcessingRun(operation='export')`. PublicationExport is created only after validated bytes are
available and records renderer name/version, render-contract version, exact ExportSpec, size
preset, dimensions, unit, DPI, media type, signature validation, final SHA-256, and byte size.

Format, DPI, and output-size-only requests are PublicationExport parameters and do not create a
new ProjectRevision. A change to the visual ChartSpec creates a new ChartSpecRevision and
ProjectRevision before the export is fixed to it.

Fresh bytes follow this sequence:

1. render, stage, validate signature, SHA-256, and byte count;
2. persist a pending StoredObject, rendering ExportJob, ProcessingRun, and pending WriteIntent;
3. confirm the external object;
4. in one transaction lock the StoredObject, mark it available, create PublicationExport, complete
   the WriteIntent, mark the run and job successful, and fill missing ShareExportBindings.

The pending WriteIntent and ExportJob StoredObject FK remain GC roots until step 4 commits. A
database failure before step 2 discards staging. A confirmation or finalization failure leaves the
recoverable pending rows. Startup reconciliation repeats confirmation and the final transaction,
including after process restart.

Physical byte reuse occurs only inside the same deduplication scope and only after the final output
SHA-256 and byte count match an available StoredObject. Request/render hashes can guide a future
cache lookup but never establish byte identity. Multiple logical PublicationExports may reference
the same immutable StoredObject through real foreign keys.

An idempotency request digest contains only the actor, project, fixed ProjectRevision, normalized
ChartSpec, and API/export contract versions. Final output hashes and byte sizes are excluded and
remain physical-object identity only. Expired records are ignored and deleted under the same
actor/operation/key row lock before key reuse; normal expiry cleanup also deletes them. Database
unique constraints remain the concurrent-creation backstop.

Temporary-project deletion calls the same locked purge implementation as expiry and saved-project
purge. Before deleting real FKs it records every SourceFile, DatasetVersion, PublicationExport,
pending WriteIntent, and pending ExportJob StoredObject as a GC candidate. Soft deletion continues
to retain all references.

## Migrated API surface

- `POST /api/v1/projects/{projectId}/shares`;
- `PATCH /api/v1/projects/{projectId}/shares/{token}`;
- `DELETE /api/v1/projects/{projectId}/shares/{token}`;
- `GET /api/v1/shares/{token}`;
- `POST /api/v1/projects/{projectId}/exports`;
- `GET /api/v1/exports/{exportId}/download`;
- `GET /api/v1/shares/{token}/downloads/{format}`;
- Share summaries in `GET /api/v1/projects/{projectId}/workspace`.

The export POST accepts the optional `Idempotency-Key` header without changing JSON. Reusing a key
with the same actor and request returns the original ExportJob; a different request returns a
conflict. Cleaned CSV remains the Phase 3 DatasetVersion download and is not a PublicationExport.

Private download first reads only PublicationExport metadata, verifies the User or GuestSession
against the Project, and only then opens and hashes object bytes. Rejected requests cannot call the
object-storage `open()` path.

## Verification

- all 71 backend tests pass;
- the 14 Phase 5A tests cover database scope rejection, Guest idempotency and expiry, logical
  request replay with different render bytes, dedup-scope isolation, complete temporary purge
  closure, fail-closed and empty downgrade paths, concurrent fixed binding, both confirmation
  recovery boundaries, authorization-before-open, and real PostgreSQL PNG/SVG/PDF restart flows;
- Ruff and MyPy pass for the backend;
- all 33 frontend tests, ESLint, TypeScript, and the production build pass;
- Alembic upgrades from `0004` to `0005`, downgrades from `0005` to `0004`, and reports no pending
  schema operations at `0005`.

## Phase 5B boundary

Phase 5A deliberately retains synchronous rendering and startup reconciliation. Phase 5B remains
responsible for task-level and item-level leases, retry budgets, fencing tokens, multi-host worker
claims, orphan staging inventory, final lifecycle/GC batching, and the S3-compatible adapter.
Redis, Kafka, microservices, a final S3 vendor, production KMS, backup policy, and compliance
claims remain outside Phase 5A.
