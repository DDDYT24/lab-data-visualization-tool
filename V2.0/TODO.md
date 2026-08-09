# LabViz Product Backlog

This file records explicitly deferred work so it does not expand the V2.0 prototype scope.

## V2.0 Closeout Gate

**Checkpoint:** 2026-08-09

- [x] P0-1: close frontend runtime-error gaps, enforce pageerror/console-error Playwright guards, and keep UTC rendering deterministic.
- [x] P0-2: repair Ruff/MyPy gates and commit hash-pinned Python 3.12 runtime/development locks.
- [x] P0-3: run PostgreSQL 17 and MinIO from the shared Compose definition in CI, initialize the test bucket, and clean up volumes.
- [x] P0-4: document one API verification gate and keep the remaining production decisions explicitly deferred below.
- [x] P1-1: record exact pandas, PyArrow, and Parquet writer provenance in PostgreSQL processing records and immutable object metadata without changing Parquet v1 bytes.
- [x] P1-2: bound PostgreSQL and MinIO integration probes so missing local services fail explicitly instead of hanging.
- [x] P1-3: return stable quality-finding message codes and parameters, and localize their summaries and reasons in the frontend with legacy-text fallback.
- [x] P1-4: persist HTTP upload `Idempotency-Key` requests, replay the original project/job, reject changed requests, and serialize concurrent first attempts.
- [x] P2-1: audit every direct frontend/API dependency, retain `httpx2` with an explicit reason, and document why the current stack is not redundant.
- [x] P2-2: keep ProjectSpec v1 generated from Pydantic and consumed by Zod with shared cross-runtime fixtures.
- [x] P2-3: define SQLite as the local/reference backend, PostgreSQL plus S3-compatible storage as the production website data route, and prohibit runtime dual-write.
- [x] P2-4: version shared ECharts/Matplotlib render semantics and test grayscale, grouped-series, fit, confidence-band, line-style, and palette precedence.
- [x] P2-5: keep Redis, Kafka, and Celery absent until measured load requires a separate queue, with an automated dependency-boundary guard.

The frontend gates are `npm run verify` and `npm run test:e2e`. The API gate is documented in
[`api/README.md`](api/README.md) and requires the Compose PostgreSQL/MinIO services for the full
integration suite. Production hosting, SMTP, cloud-provider, security, quota, and compliance
decisions remain intentionally open.

The P2 dependency and architecture evidence is recorded in
[`TECH_STACK_AUDIT.md`](TECH_STACK_AUDIT.md). P2 rendering consistency means shared scientific
and style semantics, not pixel-identical output from different rendering engines.

## Website Frontend Progress

**Last checkpoint:** 2026-08-09
**Architecture:** [`PROJECT_PLAN.md`](PROJECT_PLAN.md)

Completed:

- [x] Approve website-first delivery and defer the desktop runtime.
- [x] Select Next.js, strict TypeScript, MUI Core, MUI X Community, ECharts, next-intl, Zustand, and Zod.
- [x] Create the semantic light theme, bilingual application shell, and responsive Home screen.
- [x] Create the Import, Inspect, Chart, and Export frontend workflow without generated scientific data.
- [x] Add website file-policy, versioned chart contract, workspace state, and unit tests.
- [x] Verify 1440 px, 1024 px, and 390 px layouts without text overlap or browser-console errors.
- [x] Add frontend lint, type-check, unit-test, production-build, README, and CI coverage.
- [x] Add the versioned `/api/v1` frontend contract and validate every remote response with Zod.
- [x] Remove generated workspace rows; table, quality, chart, history, sharing, authentication, and export now use API responses or explicit edge states.
- [x] Add History, Settings, Help, Shared Chart, email-code authentication, not-found, loading, empty, expired, and recoverable error views.
- [x] Implement the matching FastAPI/OpenAPI upload, progress, preview, quality, decisions, history, auth, sharing, and publication-export endpoints.
- [x] Complete and integration-test real CSV and XLSX vertical slices against the checked frontend contract.
- [x] Complete the scientific chart controls for seven chart types, fitting, equations, R², error bars, confidence bands, dual axes, grouping, and up to four panels.
- [x] Complete PNG/SVG/PDF publication settings for 300/600 DPI, four size modes, mm/cm/in units, fonts, line styles, legend, grid, background, transparency, and grayscale review.
- [x] Add worksheet/header re-import, user-defined valid ranges, issue filtering, undo/redo, and complete cleaned-data download.
- [x] Add History search/filter/continue/duplicate/delete/export/share entry points and browser-persisted figure defaults.
- [x] Add English and Simplified Chinese interface coverage for the principal workflow, settings, help, history, and shared-chart states.
- [x] Add Playwright desktop/mobile browser flows and live API tests, including a real 1.87 MB XLSX upload and same-origin export download.
- [x] Add screenshot regression coverage for the approved 1440 × 1024, 1024 × 1024, and 390 × 844 breakpoints.
- [x] Complete final website visual QA and automated serious/critical WCAG checks across the principal workflow and supporting pages.

Next:

- [x] Localize processing-service quality summaries and reasons by returning stable message codes and parameters in the API and rendering them in the selected frontend locale.
- [ ] Move to ESLint 10 after the Next.js React lint plugins are compatible; current audit findings are confined to the development-only minimatch/brace-expansion chain.

## Backend Production Hardening

**Last checkpoint:** 2026-08-08

- [x] Approve the production data route: PostgreSQL metadata and revisions, S3-compatible object storage, and immutable Parquet dataset versions. See [`DATABASE_DESIGN.md`](DATABASE_DESIGN.md).
- [ ] Approve the remaining production backend runtime, worker, hosting, and provider selections; the current FastAPI service remains a local API-contract reference implementation.
- [x] Implement versioned `ProjectSpec` v1 in frontend Zod and backend Pydantic, backed by immutable ProjectRevision records and cross-contract fixtures.
- [ ] Add optional project/experiment description persistence and editing for shared-chart context without introducing speculative Experiment entities.
- [x] Add the phase 1 SQLAlchemy 2 persistence boundary, PostgreSQL development/test environment, provider-neutral object-storage interface, first nine core models, and reversible Alembic migration.
- [x] Add Phase 2 Repository/Unit of Work boundaries, selectable SQLite/PostgreSQL project persistence, ProjectSpec v1, Parquet v1, and the approved six-entity project revision slice without runtime dual-write.
- [x] Add Phase 3 immutable QualityReport/Finding and CleaningDecisionSet/Decision lineage, derived Parquet DatasetVersions, copied chart revisions, API parity, and object compensation.
- [x] Add Phase 4 PostgreSQL GuestSession/User ownership, authentication, in-place claim/Save, history/workspace assembly, current-revision Duplicate, 24-hour delete/restore, provenance, idempotency, and FK-authoritative object GC.
- [x] Complete Phase 5A PostgreSQL revision-pinned HMAC sharing, fixed ShareLink/export bindings, immutable permanent publication exports, and recoverable StoredObjectWriteIntent flow.
- [x] Complete Phase 5B-1 independent Worker runner, task/work-item leases, PostgreSQL-time heartbeat, fencing, bounded retry/backoff, and quarantine infrastructure without destructive maintenance.
- [x] Complete Phase 5B-2 lifecycle/reconciliation execution, final FK reachability GC, metadata cleanup, and two-pass orphan staging cleanup with safe defaults.
- [x] Complete Phase 5B-3 provider-neutral S3-compatible adapter and provider contract/failure tests.
- [x] Persist HTTP upload `Idempotency-Key` requests and replay semantics without treating identical file hashes as the same intentional project.
- [x] Activate pending StoredObject confirmation/finalization in the leased reconciliation worker and retire PostgreSQL lifespan recovery after the Phase 5B-2 cross-restart path passes.
- [x] Record exact pandas, PyArrow, and Parquet writer versions as processing/object provenance; require Parquet schema v2 for any change to the v1 byte contract.
- [x] Select authentication persistence with the configured SQLite/PostgreSQL backend without runtime dual-write.
- [ ] Add multi-host authentication abuse controls and a leased lifecycle/reconciliation worker before horizontally scaling the API.
- [ ] Integrate and test the selected production SMTP provider, abuse limits, and delivery monitoring.
- [ ] Move publication exports to managed object storage; saved PostgreSQL datasets already use the provider-neutral object-storage boundary.
- [x] Add an explicit authenticated “Save to Cloud” endpoint independent of link sharing.
- [x] Implement immutable share snapshots pinned to ProjectRevision; links do not expire by default, remain revocable, and never change when the working project is edited.
- [x] Retain saved-project publication exports for the lifetime of the project and remove them only after permanent project purge.
- [x] Implement 24-hour saved-project soft-delete recovery; suspend access immediately and purge database/object data after the window.
- [ ] Enforce the production baseline for TLS, managed encryption at rest, environment-separated secrets and KMS keys, seven-day-or-longer PITR, 30-day daily backup retention, and quarterly restore drills.
- [ ] Select the initial cloud provider and single deployment/data region before public production launch; keep database, objects, workers, and backups co-located.
- [ ] Finalize configurable saved-project count and total-storage quota values before public launch; keep the approved 50 MB per-upload limit.
- [ ] Complete target-market privacy and compliance review before making GDPR, PIPL, HIPAA, GLP, GxP, or similar claims.

## Deferred Scientific Domain Model

- [ ] Introduce `Experiment` and physical `ExperimentRun` entities, UI, and business rules only after a validated workflow requires repeated acquisitions, multi-file grouping, replicate identity, batch comparison, or experiment-level permissions.
- [ ] Keep software `ProcessingRun` separate from any future physical laboratory `ExperimentRun`.

## Figma Prototype Progress

**Last checkpoint:** 2026-07-28
**Figma file:** [LabViz V2.0 — Product Prototype](https://www.figma.com/design/xKdEwynLhAj2dyiqeEw58n)
**File key:** `xKdEwynLhAj2dyiqeEw58n`

Completed:

- [x] Create the Figma file and import the approved LabViz logo and synthetic table reference.
- [x] Create local color, dimension, and typography tokens.
- [x] Create reusable buttons, status chips, select fields, and issue cards.
- [x] Create two desktop Home directions; Direction A is the recommended starting point.
- [x] Create Import Data, Inspect and Clean, Create Chart, and Customize and Export desktop screens.
- [x] Represent fitted curves, error bars, 95% confidence intervals, grayscale preview, and PNG/SVG/PDF export controls.
- [x] Create email-code authentication, cloud-save, sharing settings, and read-only shared-chart screens.
- [x] Create History, Settings, mobile shared-chart, and required edge-state screens.

Resume after the Figma MCP quota resets:

- [ ] Run a node-level typography and clipping audit on every principal screen.
- [ ] Fix any overlapping, truncated, or undersized text in place, with special attention to the compact inspection table and dialogs.
- [ ] Recheck the mobile chart, description card, and bottom guidance at 390 × 844.
- [ ] Arrange playable prototype screens as top-level frames on one Figma page.
- [ ] Connect Home → Import → Inspect → Chart → Export and email verification → cloud save → sharing.
- [ ] Capture final screenshots and complete visual QA before approval.

## Platform Support

- [ ] Build and package the deferred Windows desktop application after the website workflow is stable.
- [ ] Define optional local/cloud two-way synchronization, stable identity mapping, conflict resolution, offline edits, deletion propagation, and encryption before implementing desktop cloud sync.
- [ ] Add a macOS installer and complete macOS compatibility testing.
- [ ] Add a Linux installer and complete Linux distribution testing.

## Authentication

- [ ] Add Google sign-in after email one-time-code authentication is stable.
- [ ] Add Microsoft sign-in after email one-time-code authentication is stable.

## Appearance and Accessibility

- [ ] Design and implement a high-quality dark theme.
- [ ] Test additional chart palettes for color-vision deficiencies and print workflows.

## Advanced Analysis

- [ ] Evaluate bootstrap confidence intervals.
- [ ] Evaluate simultaneous confidence bands and prediction intervals.
- [ ] Evaluate weighted and robust regression.
- [ ] Add residual diagnostics and multiple-comparison correction where scientifically appropriate.
- [ ] Evaluate user-defined fitting models with strong validation and guidance.

## Cloud and Commercial Features

- [ ] Define free, paid researcher, and team cloud plans using observed product usage and operating cost.
- [ ] Decide cloud project, storage, file-size, and sharing-link quotas.
- [ ] Evaluate six-panel figures as part of a paid cloud workflow.
- [ ] Evaluate batch processing, batch export, journal presets, and priority processing.
- [ ] Add billing only after the free workflow and activation metrics are validated.
- [ ] Add team workspaces, roles, organization administration, and institutional controls.

## Mobile

- [ ] Evaluate advanced mobile data cleaning after the desktop workflow is stable.
- [ ] Evaluate mobile multi-panel editing and advanced 3D controls.
