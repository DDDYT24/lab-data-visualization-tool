# LabViz V2.0 API

The current `/api/v1` application continues to use the SQLite reference repository. The
PostgreSQL package and migrations in this directory are the production persistence foundation;
business endpoints are not switched to it in the first infrastructure phase.

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

Rollback the first migration and reapply it:

```powershell
python -m alembic downgrade base
python -m alembic upgrade head
```

Run all API and PostgreSQL foundation tests:

```powershell
$env:LABVIZ_TEST_POSTGRES_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test"
python -m pytest
```

Local object storage defaults to `.labviz/objects` and is accessed only through the storage
interface. No S3 vendor SDK is part of this phase.
