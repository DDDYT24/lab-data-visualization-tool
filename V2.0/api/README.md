# LabViz V2.0 API

SQLite remains the default reference repository. Phase 2 adds an opt-in PostgreSQL project
persistence slice for project creation, reopening, preview, and immutable chart revisions without
changing their `/api/v1` request or response models. The two project backends are selected, never
dual-written. Phase 3 extends that selected backend through quality reports, immutable cleaning
decisions, derived Parquet DatasetVersions, and cleaned-data download. Phase 4 adds identity and
project lifecycle persistence. Phase 5A adds revision-pinned HMAC shares and immutable publication
exports without changing the SQLite reference repository. Phase 5B-1 adds independent PostgreSQL
worker lease, fencing, heartbeat, retry, and quarantine infrastructure. Phase 5B-2 activates
fenced reconciliation, lifecycle purge, metadata cleanup, StoredObject GC, and two-pass orphan
staging cleanup. The SQLite reference repository remains unchanged.
Phase 5B-3 adds the shared Local/S3 provider contract, real MinIO verification, bounded multipart
uploads, provider metadata, and resumable provider-scoped staging inventory.

## Local PostgreSQL

Copy `.env.example` to `.env` if local overrides are needed, then start PostgreSQL and MinIO:

```powershell
docker compose up -d postgres minio minio-init
docker compose ps
```

The defaults expose the development database on `127.0.0.1:54329` and create a separate
`labviz_test` database. These credentials are local-only.

Install dependencies and apply migrations:

```powershell
python -m pip install -r requirements-dev.txt
$env:LABVIZ_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz"
python -m alembic upgrade head
python -m alembic current
```

Select PostgreSQL for the migrated project slice:

```powershell
$env:LABVIZ_PERSISTENCE_BACKEND = "postgresql"
$env:LABVIZ_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz"
$env:LABVIZ_OBJECT_STORAGE_ROOT = ".labviz/objects"
$env:LABVIZ_SHARE_TOKEN_KEY_VERSION = "1"
$env:LABVIZ_SHARE_TOKEN_KEYS = "1=replace-with-at-least-32-random-characters"
python -m uvicorn labviz_api.main:app --reload
```

Share key rotation keeps old `version=secret` entries in `LABVIZ_SHARE_TOKEN_KEYS` for validation
while `LABVIZ_SHARE_TOKEN_KEY_VERSION` selects the only version used for new links. Do not remove
an old key while a ShareLink still records that `token_key_version`.

Omit `LABVIZ_PERSISTENCE_BACKEND` or set it to `sqlite` to run the unchanged reference path.

Select S3-compatible storage for PostgreSQL. AWS credentials are resolved through the standard
AWS provider chain and must not be committed to `.env`:

```powershell
$env:LABVIZ_OBJECT_STORAGE_BACKEND = "s3"
$env:LABVIZ_S3_BUCKET = "labviz-test"
$env:LABVIZ_S3_PREFIX = "labviz/dev"
$env:LABVIZ_S3_REGION = "us-east-1"
$env:LABVIZ_S3_ENDPOINT_URL = "http://127.0.0.1:59000" # omit for AWS S3
$env:AWS_ACCESS_KEY_ID = "labviz-minio"                 # local Compose only
$env:AWS_SECRET_ACCESS_KEY = "labviz-minio-local-only" # local Compose only
```

Choosing `s3` with an invalid bucket, endpoint, or credential fails startup; it never silently
falls back to the Local provider.

Rollback only Phase 5A while retaining the Phase 4 schema, then reapply it:

```powershell
python -m alembic downgrade 0004_identity_project_lifecycle
python -m alembic upgrade head
```

Rollback and reapply only the Phase 5B-1 lease schema:

```powershell
python -m alembic downgrade 0005_share_publication_exports
python -m alembic upgrade head
```

Rollback and reapply the Phase 5B-3 inventory schema only after its checkpoint and candidate
manifest is empty. Migration 0008 fails closed while that diagnostic state exists:

```powershell
python -m alembic downgrade 0007_phase5b2_orphan_staging
python -m alembic upgrade head
```

Rollback and reapply only the Phase 5B-2 staging-inventory schema:

```powershell
python -m alembic downgrade 0006_worker_leases
python -m alembic upgrade head
```

Run a worker as an independent process. Reconciliation may finalize already-persisted writes;
lifecycle, GC, metadata, and staging tasks default to audited dry-run:

```powershell
$env:LABVIZ_PERSISTENCE_BACKEND = "postgresql"
$env:LABVIZ_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz"
python -m labviz_api.workers.cli pending-reconciliation --once
```

Available tasks are `pending-reconciliation`, `project-lifecycle`, `stored-object-gc`,
`orphan-staging-inventory`, and `metadata-cleanup`. All eligibility, lease, retry, and grace-period
decisions use PostgreSQL time. FastAPI does not run PostgreSQL maintenance from its lifespan.

Destructive execution requires both settings; changing only one is insufficient:

```powershell
$env:LABVIZ_WORKER_DRY_RUN = "false"
$env:LABVIZ_WORKER_DELETE_ENABLED = "true"
python -m labviz_api.workers.cli project-lifecycle --once
python -m labviz_api.workers.cli stored-object-gc --once
```

Run `project-lifecycle` before `metadata-cleanup`, so expired temporary Projects are purged before
their now-unreferenced GuestSessions. Run `pending-reconciliation` before GC during routine
operations. `orphan-staging-inventory` requires two observations separated by
`LABVIZ_WORKER_ORPHAN_STAGING_GRACE_SECONDS`; a single provider listing never deletes bytes.

Use `downgrade base` only against an isolated disposable test database when verifying the full
migration chain.

Run all API and PostgreSQL foundation tests:

```powershell
$env:LABVIZ_TEST_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test"
python -m pytest
```

Local object storage defaults to `.labviz/objects`. Local and S3 are selected, never dual-written,
and both are accessed only through the provider-neutral contract. Real MinIO tests require the
Compose service and the dedicated PostgreSQL test database.

Regenerate the ProjectSpec v1 JSON Schema after an intentional Pydantic contract change:

```powershell
python -m scripts.export_project_spec_schema
```

The generated schema and shared fixtures live in `V2.0/contracts`. See
`PERSISTENCE_PHASE2.md` and `PARQUET_FORMAT.md` for the transaction, compensation, and Parquet v1
rules. See `PERSISTENCE_PHASE3.md` for quality lineage, decision semantics, undo, and derived
dataset behavior. See `PERSISTENCE_PHASE5B1.md` for the lease foundation and
`PERSISTENCE_PHASE5B2.md` for lifecycle state machines and operations, and
`PERSISTENCE_PHASE5B3.md` for provider, multipart, metadata, cursor, checkpoint, and recovery
semantics.
