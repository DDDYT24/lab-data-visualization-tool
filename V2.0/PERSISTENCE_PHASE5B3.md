# Phase 5B-3 Object Storage Providers and Resumable Inventory

Status: implemented and verified on 2026-08-01.

Phase 5B-3 adds S3/MinIO compatibility without changing `/api/v1`, the SQLite reference
Repository, immutable PublicationExport/ShareLink rules, or the Phase 5B lease and GC protocols.
Exactly one object provider is selected at runtime; there is no Local/S3 fallback or dual-write.

## Provider contract and backend identity

`labviz_api.storage.ObjectStorage` is the only interface used by persistence and Workers. It
defines stable `backend_name` and `inventory_scope` identities plus `stage`, `put`, `head`, `open`,
`open_staged`, immutable `confirm`, conditional `delete`, idempotent `discard`, and bounded
`list_staged` pages.

- Local rows retain `storage_backend=local`.
- AWS S3 and compatible services use the stable database identity `s3`; endpoint URLs are not part
  of the backend name.
- `inventory_scope` binds a cursor/checkpoint to its Local root or S3 bucket/prefix and contains no
  credentials.
- API construction and the independent Worker CLI both use `storage.factory.build_object_storage`.
- Creation, dedup lookup, read, claim, reconciliation, GC, and purge validate the current backend.
  A configured provider never probes another provider.
- StoredObject final and staging keys are unique per `(storage_backend, key)`, so historical Local
  and S3 rows can coexist without cross-backend reuse or false uniqueness conflicts.

The Local provider keeps existing object keys and atomic hard-link publication. Optional
format/media metadata is stored in atomic sidecars below its root. Legacy Local objects without a
sidecar remain readable only after SHA-256 and size are recomputed from the bytes; missing optional
format metadata is explicit, not treated as an integrity bypass.

## S3 and MinIO configuration

Set `LABVIZ_OBJECT_STORAGE_BACKEND=s3` and configure bucket, optional key prefix, region, optional
S3-compatible endpoint, multipart threshold/part size, timeouts, and cursor TTL. Credentials use
boto3's standard AWS provider chain. Application settings contain no access key, secret, session
token, or signed URL. Bounded errors report an operation/error code, not credentials or object
content. Invalid S3 initialization fails closed.

All logical keys are normalized relative POSIX paths. Absolute paths, backslashes, empty segments,
`.` and `..` are rejected before a provider call. Every physical key stays within the configured
bucket and prefix.

`compose.yaml` provides pinned PostgreSQL, MinIO, and bucket-initializer services for local tests.
Its credentials are development-only placeholders. Production provider, KMS, backup, region,
quota, retention, and compliance configuration remain Phase 6.

## Integrity metadata and immutable confirm

Both providers return the same authoritative core:

- application SHA-256 and exact byte size;
- application metadata version;
- optional format-contract version and media type;
- provider `last_modified`;
- optional ETag/version ID for diagnostics or a conditional operation only.

S3 stores SHA-256, size, metadata version, upload mode, creation timestamp, format version, and
media type as object metadata. `head()` uses HEAD and never downloads bytes. Missing or malformed
S3 integrity metadata fails closed. ETag is never treated as SHA-256; multipart ETags are expected
to differ.

Staging and final writes use `If-None-Match: *`. A small write uses conditional PutObject; a
multipart write uses conditional CompleteMultipartUpload. If another writer wins, LabViz HEADs
the final object and accepts it only when SHA-256, size, format version, and media type match.
Different bytes or metadata cannot overwrite the key. References:

- <https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html>
- <https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html>
- <https://www.min.io/blog/leading-the-way-minios-conditional-write-feature-for-modern-data-workloads>

Real MinIO tests verify the same behavior, including simultaneous confirm.

## Multipart upload

Input is copied through a bounded `SpooledTemporaryFile` while SHA-256 and size are computed. Data
at or above the threshold is forced to disk before network upload, so a large object is not retained
in memory. Parts stream at a configured size of at least 5 MiB. Final metadata is known before
multipart creation. Any part or complete failure triggers AbortMultipartUpload; a real MinIO
fault-injection test verifies both the abort call and absence of an outstanding multipart session.
Small objects continue to use conditional PutObject.

## Staging pagination and cursor semantics

Pages default to 100 and are capped at 1,000. Business code treats the cursor as opaque. Its
base64url envelope binds version, backend, inventory scope, issue/expiry time, scan snapshot, and
provider state. Malformed, expired, tampered, backend-mismatched, or scope-mismatched cursors fail
instead of continuing elsewhere.

- Local uses deterministic keyset pagination and keeps only `page_size + 1` candidates in memory;
  it never materializes the complete inventory.
- S3 uses `ListObjectsV2` continuation tokens and bounded HEAD calls for authoritative metadata.
- With an unchanged inventory, traversal returns every staging key exactly once.
- An object created or overwritten after `snapshot_at` is deferred to the next generation.
- An object deleted during a scan may be absent from later pages. Deletion still requires a fresh
  HEAD and fingerprint comparison.
- A key returned before an overwrite remains an observation of the old fingerprint. The next
  complete generation sees the new fingerprint and resets its safety window.

Provider contract tests fix empty, exact-boundary, multi-page, resume, mutation, prefix isolation,
path rejection, invalid/expired cursor, and bounded-page behavior for Local and real MinIO.

## Migration 0008 and inventory checkpoint

`0008_phase5b3_storage_inventory` leaves `0001` through `0007` unchanged. It provider-scopes
orphan candidates and records provider last-modified/ETag, generation observation, and deletion
claim state. It adds `storage_inventory_checkpoints`, keyed by backend and inventory scope, with
generation UUID, status, opaque cursor, timestamps, task owner/fencing token, page/item counts, and
bounded diagnostics.

One scanner invocation processes one page:

```text
short transaction: lock task lease and checkpoint, bind owner/fence, commit
-> provider list page outside SQL
-> short transaction: revalidate task owner/fence and generation
-> upsert page fingerprints and persist the next cursor
-> on the last page, increment observations once and complete the generation
```

A stop before checkpoint commit repeats that page safely. A stop after commit resumes the next
page. Lease-expiry takeover changes task fencing, so the old Worker cannot update the checkpoint.
Page replay does not count as a second inventory; only a complete generation increments
`observation_count`. An invalid persisted cursor marks the checkpoint `failed` with bounded
diagnostics and requires operator action. It never silently restarts and converts one scan into
two. A completed checkpoint starts a fresh generation on the next invocation.

Downgrade `0008 -> 0007` fails closed while any checkpoint or candidate exists. An empty test
database supports downgrade, re-upgrade, and schema-drift validation.

## WriteIntent protection and orphan deletion

The authoritative staging-owner query explicitly checks the current provider's pending
StoredObject and active/quarantined StoredObjectWriteIntent. Through that StoredObject identity it
checks SourceFile, DatasetVersion, PublicationExport, ExportJob pending-object, and every real GC
root. Inventory removes a stale candidate when any owner appears.

Creation of a pending Dataset or Export root and the final deletion claim use the same
provider/staging-key PostgreSQL advisory transaction lock. Creation commits the StoredObject and
WriteIntent while holding the lock. Deletion holds it while rechecking ownership, task fencing,
grace, two complete observations, and current provider fingerprint, then records
`deletion_started_at` and commits. Provider delete runs after the transaction. The final short
transaction revalidates task owner/fence and SQL ownership before removing the candidate.

If a process stops after the claim or after provider deletion, a later task-fenced Worker retries
the same conditional operation. Successful deletion retains a 24-hour candidate tombstone, so a
request that staged bytes before the claim but reaches SQL late still sees the deletion claim and
cannot create a reference to missing bytes. A later complete inventory removes the aged tombstone.

Either the SQL root wins and deletion refuses, or the deletion claim wins and application creation
fails safely before recording a reference. A real PostgreSQL concurrency test holds the shared
advisory lock while a competing delete waits, then proves an active WriteIntent prevents deletion.
Defaults remain:

```text
LABVIZ_WORKER_DRY_RUN=true
LABVIZ_WORKER_DELETE_ENABLED=false
```

## State writes and guards

| State write | Required protection |
| --- | --- |
| new pending StoredObject | current backend, staging advisory lock, no deletion claim, DB transaction |
| pending reconciliation claim | PostgreSQL time, short row claim, owner, lease, incremented fencing token |
| pending to available | confirm outside SQL; new transaction verifies owner/fence/state/hash/size and clears lease atomically |
| available to deleting | StoredObject row lock, current backend, complete FK reachability recheck, owner/fence |
| deleting to deleted/available | provider delete outside SQL; new transaction verifies owner/fence/state and references |
| inventory checkpoint | task row lock plus checkpoint owner, task fence, generation and running state |
| orphan delete claim/finalize | task fence, advisory lock, owner/fingerprint/grace recheck; provider I/O outside SQL |

DatasetVersion remains a GC root throughout pending confirmation. WriteIntent remains a root until
the transaction that makes StoredObject available, creates immutable PublicationExport, completes
the intent, and marks ExportJob ready. Real MinIO crash tests cover provider confirm followed by a
stop before SQL finalization and subsequent idempotent fenced recovery.

## Operations and residual scope

Run API and Workers with identical storage environment. Routine ordering remains:

```text
pending-reconciliation -> project-lifecycle -> stored-object-gc
-> orphan-staging-inventory -> metadata-cleanup
```

Use `python -m labviz_api.workers.cli <task> --once` for one bounded pass. Provider I/O is outside
SQL transactions; integration probes assert no `idle in transaction` session during S3 open,
confirm, list, or delete. Structured errors/logs omit credentials, tokens, object content, and
sensitive paths. Failed checkpoints, integrity mismatches, permission failures, and quarantine
remain diagnostic state for explicit operator handling; there is no automatic destruction.

Real tests require PostgreSQL plus MinIO. Configure `LABVIZ_TEST_POSTGRES_URL`,
`LABVIZ_TEST_MINIO_ENDPOINT`, `LABVIZ_TEST_MINIO_BUCKET`, and local test credentials, then run:

```powershell
python -m pytest tests/test_storage_providers.py tests/test_phase5b3_object_storage.py
python -m pytest
python -m alembic check
```

Phase 6 still owns production KMS, credential lifecycle, formal backup/restore, region and
replication policy, quotas, retention, compliance, SMTP, and production observability/SLOs.

## Final verification

The final matrix ran against PostgreSQL 17 and a real MinIO server. Docker Desktop was not used
because its local WSL integration was unavailable; the equivalent standalone services were used,
so no provider or database test was skipped.

* Backend: 144 tests passed, including 19 Local/MinIO provider cases and 11 real
  PostgreSQL-plus-MinIO integration cases.
* Local inventory snapshot ordering was repeated 30 times after hardening its logical timestamp
  against Windows wall-clock tick collisions.
* Ruff and MyPy strict passed with zero findings.
* Alembic `0008 -> 0007 -> 0008` passed, and `alembic check` reported no schema drift.
* Frontend ESLint, TypeScript, 33 Vitest tests, and the Next.js production build passed.
* Playwright completed with 15 passed and two existing live-file/live-API cases skipped because
  `LABVIZ_E2E_LIVE` and `LABVIZ_E2E_FILE` were not configured. WCAG checks passed without an
  exemption.
* `git diff --check` passed. No production credential, environment file, database, object data,
  log, cache, or test artifact is part of the Phase 5B-3 change set.
