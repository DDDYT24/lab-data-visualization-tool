# V2.0 Technology Stack Audit

**Closed:** 2026-08-09

## Decision

The current direct dependencies are justified by implemented V2.0 behavior. Removing one would
either break a runtime path, a supported file/chart format, the persistence boundary, or a
verification gate. No Redis, Kafka, Celery, microservice framework, or duplicate state/chart
library is present.

## Frontend

- Next.js and React provide the website runtime; TypeScript is the compile-time contract gate.
- MUI Core, icons, Data Grid, and the MUI Next.js integration provide the application shell,
  controls, virtualized preview table, and App Router style cache. Emotion packages are the
  required MUI styling/SSR peers, including the server package used by the Next.js integration.
- ECharts, its React adapter, and ECharts GL provide the interactive 2D and optional 3D preview.
- TanStack Query owns remote state; Zustand owns only local workspace state. They are not
  competing stores.
- Zod validates every remote and persisted frontend contract; next-intl owns English/Chinese
  rendering.
- Playwright, Vitest, ESLint, axe-core, JSDOM, and type packages are verification-only tools.

## API

- FastAPI, Uvicorn, Pydantic, email-validator, and python-multipart provide the HTTP runtime,
  validated email models, and multipart upload/form parsing.
- pandas, NumPy, SciPy, Matplotlib, openpyxl, and PyArrow implement parsing, scientific analysis,
  publication export, XLSX support, and immutable Parquet datasets.
- SQLAlchemy, Alembic, psycopg, and boto3 implement the selectable persistence boundary,
  migrations, PostgreSQL, and S3-compatible storage.
- pytest, Ruff, and MyPy are verification-only tools. `httpx2` remains intentional because the
  installed Starlette TestClient selects it.
- Python 3.12 runtime and development lock files are hash-pinned; range files remain the reviewed
  dependency inputs.

## Architecture Boundaries

- SQLite is the default local/reference repository and a candidate for the deferred desktop
  mode. It is not the production website system of record.
- The production website data route is PostgreSQL metadata plus S3-compatible objects and
  immutable Parquet versions. Exactly one persistence backend is selected; there is no runtime
  dual-write.
- ProjectSpec v1 is generated from Pydantic into committed JSON Schema and consumed by Zod with
  shared valid/invalid fixtures.
- ECharts preview and Matplotlib export share `contracts/chart-render-v1.json` for palette, line,
  grid, and confidence-band semantics. Pixel equality is not promised across SVG/canvas and
  Matplotlib; data, labels, series assignment, grayscale precedence, and styling intent are.
- PostgreSQL leases remain the worker/queue mechanism. Redis, Kafka, and Celery stay deferred
  until measured load demonstrates a need.

## Revisit Rule

Add or replace a dependency only for an implemented requirement, record the boundary change,
regenerate the relevant lock file, and run the complete API/frontend gates.
