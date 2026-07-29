# LabViz Product Backlog

This file records explicitly deferred work so it does not expand the V2.0 prototype scope.

## Website Frontend Progress

**Last checkpoint:** 2026-07-29
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

- [ ] Localize processing-service quality summaries and reasons by returning stable message codes and parameters in the future production API.
- [ ] Move to ESLint 10 after the Next.js React lint plugins are compatible; current audit findings are confined to the development-only minimatch/brace-expansion chain.

## Backend Production Hardening

- [x] Approve the production data route: PostgreSQL metadata and revisions, S3-compatible object storage, and immutable Parquet dataset versions. See [`DATABASE_DESIGN.md`](DATABASE_DESIGN.md).
- [ ] Approve the remaining production backend runtime, worker, hosting, and provider selections; the current FastAPI service remains a local API-contract reference implementation.
- [ ] Implement versioned `ProjectSpec` v1 in frontend Zod and backend Pydantic, backed by immutable ProjectRevision records and cross-contract fixtures.
- [ ] Add optional project/experiment description persistence and editing for shared-chart context without introducing speculative Experiment entities.
- [ ] Add SQLAlchemy persistence boundaries and Alembic migrations for the production PostgreSQL schema.
- [ ] Move reference-service SQLite authentication and project state to production-grade shared services before running multiple hosts.
- [ ] Integrate and test the selected production SMTP provider, abuse limits, and delivery monitoring.
- [ ] Move saved cloud datasets and exports from SQLite blobs to managed object storage with a scheduled expiry worker.
- [ ] Add an explicit authenticated “Save to Cloud” endpoint independent of link sharing.
- [ ] Implement immutable share snapshots pinned to ProjectRevision; links do not expire by default, remain revocable, and never change when the working project is edited.
- [ ] Retain saved-project exports for the lifetime of the project and remove them only after permanent project purge.
- [ ] Implement 24-hour saved-project soft-delete recovery; suspend access immediately and purge database/object data after the window.
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
