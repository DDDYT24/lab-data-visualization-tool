# LabViz V2.0 API

SQLite remains the default reference repository. Phase 2 adds an opt-in PostgreSQL project
persistence slice for project creation, reopening, preview, and immutable chart revisions without
changing their `/api/v1` request or response models. The two project backends are selected, never
dual-written. Phase 3 extends that selected backend through quality reports, immutable cleaning
decisions, derived Parquet DatasetVersions, and cleaned-data download. Phase 4 adds identity and
project lifecycle persistence. Phase 5A adds revision-pinned HMAC shares and immutable publication
exports without changing the SQLite reference repository. Phase 5B-1 adds independent,
non-destructive PostgreSQL worker lease, fencing, heartbeat, retry, and quarantine infrastructure;
it does not yet execute lifecycle or object-storage maintenance.

## Local PostgreSQL

Copy `.env.example` to `.env` if local overrides are needed, then start PostgreSQL:

```powershell
docker compose up -d postgres
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

Run a non-destructive worker scan as an independent process:

```powershell
$env:LABVIZ_PERSISTENCE_BACKEND = "postgresql"
$env:LABVIZ_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz"
python -m labviz_api.workers.cli pending-reconciliation --once
```

Available tasks are `pending-reconciliation`, `project-lifecycle`, `stored-object-gc`,
`orphan-staging-inventory`, and `metadata-cleanup`. In Phase 5B-1 the CLI only exercises task-level
leases; business-row claims are covered by the worker contract tests and remain disabled until a
real 5B-2 handler is installed. It does not purge Projects, delete StoredObjects, or clean staging
data, even if destructive maintenance is configured. Phase 5B-2 must add those operations behind
explicit safety checks.

Use `downgrade base` only against an isolated disposable test database when verifying the full
migration chain.

Run all API and PostgreSQL foundation tests:

```powershell
$env:LABVIZ_TEST_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test"
python -m pytest
```

Local object storage defaults to `.labviz/objects` and is accessed only through the storage
interface. No S3 vendor SDK is part of this phase.

Regenerate the ProjectSpec v1 JSON Schema after an intentional Pydantic contract change:

```powershell
python -m scripts.export_project_spec_schema
```

The generated schema and shared fixtures live in `V2.0/contracts`. See
`PERSISTENCE_PHASE2.md` and `PARQUET_FORMAT.md` for the transaction, compensation, and Parquet v1
rules. See `PERSISTENCE_PHASE3.md` for quality lineage, decision semantics, undo, and derived
dataset behavior. See `PERSISTENCE_PHASE5B1.md` for worker lease and fencing guarantees.
