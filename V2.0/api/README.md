# LabViz V2.0 API

SQLite remains the default reference repository. Phase 2 adds an opt-in PostgreSQL project
persistence slice for project creation, reopening, preview, and immutable chart revisions without
changing their `/api/v1` request or response models. The two project backends are selected, never
dual-written.

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
python -m uvicorn labviz_api.main:app --reload
```

Omit `LABVIZ_PERSISTENCE_BACKEND` or set it to `sqlite` to run the unchanged reference path.

Rollback only the Phase 2 migration while retaining the Phase 1 tables, then reapply it:

```powershell
python -m alembic downgrade 0001_core_foundation
python -m alembic upgrade head
```

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
rules.
