# LabViz Product Backlog (V2.1.1 local release)

This file is the single source of truth for future work, completion status, release gates,
compatibility issues, and deferred decisions. Update it first whenever the product status
changes. Other Markdown files may explain a stable contract or preserve historical evidence,
but they must not create a competing backlog.

- **Current release:** V2.1.1 local self-hosted release, published on GitHub
- **Status review:** 2026-09-10
- **Version history:** [`../VERSION_BASELINE.md`](../VERSION_BASELINE.md)

This file records the V2.0 baseline, the ordered V2.1 work, and explicitly deferred work.

> **Release note:** V2.1.1 is distributed as a local self-hosted application. The AWS Phase 6C/6D
> items below are optional maintainer work and do not block the V2.1.1 local release. Local use relies
> on automatically created SQLite and local object storage.

## V2.1 Roadmap

**Planning checkpoint:** 2026-09-10

**Release intent:** Improve the local-first workflow for non-programmer laboratory users and
make structured 3D data scientifically legible. Keep the V2.0 local release stable while
shipping the work below in order. Public-cloud operations, desktop packaging, billing, and team
features remain separate tracks.

### P0 — 3D surface correctness (complete before P1)

- [x] **V21-P0-1: Add a structured-grid and surface-field contract.** Detect numeric X/Y/Z
  candidates, regular rectangular grids, duplicate coordinate pairs, missing grid cells, and
  non-finite values. Expose explicit X, Y, and Z roles for `surface3d`; keep the existing
  generic series editor for other chart types.
  - **Acceptance:** `07_surface3d.csv` is identified as a 21 x 21 grid with 441 usable points;
    irregular, duplicate, missing, and collinear fixtures produce actionable messages; the
    recommendation never silently changes a user's chart.
  - **Evidence (2026-09-08, V2.1 local development branch):** the actual `07_surface3d.csv` returns a valid 21 x 21
    grid with 441 usable points; the API blocks invalid surface exports with stable error codes;
    the chart editor uses explicit Y/Z selectors; duplicate, missing, irregular, collinear, and
    non-finite cases pass the focused API regression. Browser and quality-recommendation gates
    remain open below.
- [x] **V21-P0-2: Make quality checks and chart recommendations grid-aware.** Do not treat the
  row-boundary reset in a flattened grid as a sudden change. Warn when a line chart would join
  repeated X values or different surface slices, and recommend surface, heatmap, or scatter
  views with a plain-language explanation.
  - **Acceptance:** the clean `07_surface3d.csv` fixture has no false row-order anomaly
    findings, while a true injected discontinuity is still reported; the recommendation is
    localized in English and Simplified Chinese.
  - **Evidence (2026-09-09, V2.1 local development branch):** the actual 441-row fixture reports no sudden-change
    findings; a deterministic 21 x 21 regression with a 1,000-unit local spike still reports the
    affected Z rows. Line charts over a complete grid receive a localized 3D-surface suggestion,
    while incomplete, duplicated, or irregular grid-like data receives a localized scatter
    suggestion. Suggestions are informational and never change the selected chart.
- [x] **V21-P0-3: Verify interactive and export surface parity.** Add a browser regression for
  the real surface workflow and assert the serialized `ChartSpec`, `surfacePoints`, 3D axes,
  ECharts-GL readiness, camera controls, and PNG/SVG/PDF export path. Keep backend Matplotlib
  and frontend ECharts semantics aligned.
  - **Acceptance:** the browser test selects X=`x`, Y=`y`, Z=`z`, receives 441 points, renders
    `grid3D`/`xAxis3D`/`yAxis3D`/`zAxis3D`, and never falls back to a `y over x` line chart.
  - **Evidence (2026-09-09, V2.1 local development branch):** the Chromium regression selects the explicit fields,
    observes all four ECharts-GL 3D components, renders all 441 analyzed points, exercises mouse
    rotation and zoom, and verifies that the saved chart and PNG/SVG/PDF requests retain the same
    surface fields. The focused backend surface render/export regression and frontend render-contract
    unit test pass. Local Playwright now starts an isolated server on port 3100 instead of reusing an
    unrelated process on port 3000.

### P1 — user and scientific workflow (start after all P0 gates pass)

- [x] **V21-P1-1: Add beginner guidance and progressive disclosure.** Provide a recommended
  chart card for detected data shapes, X/Y/Z help, short explanations for cleaning, grouping,
  uncertainty, and fitting, plus surface-specific empty/error states. Preserve the no-code,
  reversible-cleaning principles and the desktop/mobile responsibility split.
  - **Acceptance:** a first-time user can import the surface fixture, understand the suggested
    chart, correct a wrong chart type, and reach export without reading developer terminology;
    Playwright, keyboard, localization, and serious/critical WCAG checks pass.
  - **Evidence (2026-09-09, V2.1 local development branch):** the browser flow uploads a deterministic surface CSV,
    uses the localized recommendation card with keyboard Enter, explains cleaning/grouping/fitting/
    uncertainty in context, checks the dedicated surface setup state, switches English/Simplified
    Chinese, verifies the mobile desktop handoff, reaches export, and finds no serious or critical
    Axe violations. The full frontend verify gate passes with 42 Vitest tests.
- [x] **V21-P1-2: Ship a scientifically defensible analysis slice.** Define and validate the
  first supported bootstrap confidence interval, prediction/simultaneous-band, residual
  diagnostic, robust/weighted fitting, and multiple-comparison guidance paths. Every method
  must state assumptions, sample size, exclusions, and limitations in the result and export.
  - **Acceptance:** independent answer-keyed fixtures cover valid, insufficient, and misleading
    cases; the UI does not present a statistically weaker default as more authoritative; the
    remaining advanced methods are explicitly listed as deferred rather than implied complete.
  - **Evidence (2026-09-09, V2.1 local development branch):** the additive v1 analysis contract now records fit
    method, interval method, sample size, exclusions, assumptions, residual RMSE/MAE, possible
    residual structure, and method limitations in each analyzed series. Ordinary and weighted
    least-squares fits are supported; Student-t and deterministic 400-resample residual Bootstrap
    pointwise mean bands are supported. Prediction intervals, simultaneous bands, robust fitting,
    and multiplicity correction remain explicitly deferred in both the result card and exported
    figure footer. Independent valid, insufficient, and misleading answer-keyed fixtures pass;
    the browser regression verifies the method selector, evidence disclosure, and deferred-method
    copy. `ruff`, `mypy`, targeted API tests, frontend verify, and focused Chromium tests pass.
- [x] **V21-P1-3: Introduce the minimum experiment-level model.** Add validated `Experiment`
  and physical `ExperimentRun` concepts for repeated acquisitions, replicate identity, batch
  identification/history comparison, and experiment-level permissions while keeping software
  `ProcessingRun` separate.
  - **Acceptance:** one multi-file/replicate workflow has a persisted schema, API contract,
    history view, and export provenance; migration and deletion/restore behavior are tested
    before broader team features are considered.
  - **Evidence (2026-09-09, V2.1 local development branch):** optional upload metadata groups repeated files by
    experiment for the same browser or signed-in owner; each file receives a distinct physical
    `ExperimentRun` with run, replicate, and batch identity. SQLite and PostgreSQL persistence,
    additive API contracts, history search/cards, export response metadata, download headers, and
    figure footers carry the lineage while software `ProcessingRun` remains separate. The local
    two-file workflow, ownership claim, history, export snapshot, and orphan cleanup tests pass;
    a real PostgreSQL 17 run also passes upgrade/downgrade, schema-column, downgrade-protection,
    and soft-delete/restore checks. Alembic offline PostgreSQL DDL, `ruff`, `mypy`, API regression,
    frontend verify, and focused Chromium flows pass. Repository-wide `alembic check` still reports
    pre-existing name-only drift for historical check constraints; no P1-3 table or column drift was
    reported, and the baseline issue is tracked as `V21-C-4` below.

### Compatibility and release hardening

- [x] **V21-C-1: Normalize UTF-8 BOM headers** and add a regression that expects the first
  column name without `\ufeff` for TXT/CSV imports.
  - **Evidence (2026-09-09, V2.1 local development branch):** CSV, TSV, and delimiter-detected TXT parsing now
    explicitly uses UTF-8 with BOM handling. API regressions verify all three formats expose
    `time_min`, never `\ufefftime_min`; the original `E10_BOM表头兼容性.txt` fixture also parses
    to two clean columns and three preserved rows. The four focused cases and all 23 API tests
    pass, followed by clean `ruff` and `mypy` checks.
- [x] **V21-C-2: Make cleaned-data downloads Unicode-safe.** Encode `Content-Disposition` for
  Chinese and other non-ASCII filenames and add a 200-status download regression.
  - **Evidence (2026-09-09, V2.1 local development branch):** cleaned CSV responses use an ASCII fallback plus an
    RFC 5987 UTF-8 `filename*` value. The original `E11_中文文件名.csv` uploads and downloads with
    status 200, and the API regression checks the encoded Chinese filename and BOM CSV body.
- [x] **V21-C-3: Re-run the complete fixture matrix** after P0/P1 changes, including the 23-file
  package, answer-keyed surface checks, API checks, browser checks, and export checks. Record
  the scope explicitly when cloud, live mail, or private-file tests are not configured.
  - **Evidence (2026-09-09, V2.1 local development branch):** all 23 fixture datasets produced 95 passing checks;
    seven chart types each exported PNG, SVG, and PDF (21 exports). The 243-test API/PostgreSQL/
    MinIO gate, 45 frontend unit tests, production build, and 27 configured browser tests pass.
    A separate live-service browser flow processes and exports a generated 1.87 MB CSV. The
    private-workbook, live-mail, AWS, and external GPU/OS combinations remain explicitly unverified.
  - **Current verification note (2026-09-09):** PostgreSQL 17 and MinIO were started from the
    repository Compose definition, the test bucket was initialized, and all 243 API tests passed in
    143.96 seconds. Ruff, formatting, MyPy, frontend verification, production build, and the
    configured browser suite also passed; the temporary Compose services were stopped afterward.
- [x] **V21-C-4: Reconcile the Alembic check-constraint naming baseline.** Remove the historical
  name-only drift between SQLAlchemy metadata and the first nine migrations, then require a clean
  `alembic check` against a fresh PostgreSQL 17 database without renaming live constraints blindly.
  - **Evidence (2026-09-09, V2.1 local development branch):** migration `0011_normalize_check_names` recognizes only
    the exact historical SQLAlchemy double-prefixed names, renames 100 affected constraints in the
    persisted development database without rebuilding them, and is a no-op on a fresh database.
    Fresh and legacy PostgreSQL 17 upgrade paths both end at head with `alembic check` reporting no
    new upgrade operations; targeted round-trip and normalization regressions pass.
- [x] **V21-C-5: Publish the local privacy boundary.** Document the default SQLite/local-object
  path, dependency installation, backup and restore procedure, and the conditions that would send
  data outside the machine. Add regression checks for loopback binding, local defaults, relative API
  routing, and the absence of telemetry hooks.
  - **Evidence (2026-09-09, V2.1.0 local release):** `PRIVACY_DATA_BOUNDARY.md`,
    `OFFLINE_INSTALL.md`, `BACKUP_RESTORE.md`, `SECURITY.md`, `CONTRIBUTING.md`, and the public
    release checklist are present; the focused privacy test passes alongside Ruff, MyPy, frontend
    verify, and browser visual/accessibility coverage.
- [x] **V21-C-6: Make the user-facing storage language network-neutral.** Explain temporary and
  saved projects as local states in both locales, keep the save/share flow explicit, and remove
  stale cloud wording from the help and settings views without changing persisted API compatibility.
  - **Evidence (2026-09-09, V2.1.0 local release):** the localized frontend verify and
    visual/accessibility browser suite pass; legacy wire values remain accepted for existing projects.

### Later optimization (after V2.1 acceptance)

- [ ] Replace the legacy persisted `temporary-cloud`/`saved-cloud` state names with network-neutral
  values in a separately versioned migration after confirming API and saved-history compatibility.
- [ ] Add a true no-store mode only if users require processing without local persistence; define
  its refresh, history, export, and crash-recovery semantics before implementation.
- [ ] Evaluate large-surface performance, adaptive sampling, irregular-surface triangulation,
  accessible data-table alternatives, color-vision-safe palettes, and mobile 3D controls.
- [ ] Resume the optional AWS Phase 6C/6D track only with account-owned credentials, staging
  evidence, backups, monitoring, quotas, privacy/compliance review, load/recovery drills, and
  rollback evidence.
- [ ] Build native installers, external identity providers, team workspaces, billing, and other
  commercial features only after the local workflow and usage evidence justify them.

### V2.1 definition of done

- [x] All three P0 gates pass on the answer-keyed surface fixtures and the browser regression.
- [x] All three P1 tracks have implementation, localized UX copy, versioned API/schema changes,
  reproducible fixtures, and documented limitations.
- [x] The two known compatibility regressions are closed and remain in automated regression
  coverage.
- [x] Local privacy boundary, offline installation, backup/restore, security, contribution, and
  public-release documents are present; default local routing and public-text hygiene have focused
  regression checks.
- [x] A fresh PostgreSQL 17 database reaches Alembic head and returns a clean `alembic check`.
- [x] `npm run verify`, `npm run test:e2e`, and the documented API gate pass; skipped live/cloud
  checks are reported as skipped and do not count as evidence.

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
integration suite. The AWS single-region production runtime is approved in
[`PERSISTENCE_PHASE6.md`](PERSISTENCE_PHASE6.md). Real SES delivery, live cloud deployment and
restore evidence, quotas, observability/SLOs, and compliance remain explicit later Phase 6 work. The
Phase 6 code and workflow definitions may exist in the repository without constituting live AWS
acceptance.

The P2 dependency and architecture evidence is recorded in
[`TECH_STACK_AUDIT.md`](TECH_STACK_AUDIT.md). P2 rendering consistency means shared scientific
and style semantics, not pixel-identical output from different rendering engines.

## Website Frontend Progress

**Last checkpoint:** 2026-09-08
**Architecture:** [`PROJECT_PLAN.md`](PROJECT_PLAN.md)

The latest checkpoint includes the chart-inspector spacing regression fix and the current general
desktop/mobile browser gates. The V2.1 `surface3d` field/grid contract, grid-aware quality and
recommendation behavior, dedicated browser preview/export regression, and beginner guidance are
complete under `V21-P0-1` through `V21-P0-3` and `V21-P1-1`. P1 analysis/model work remains open.

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

**Last checkpoint:** 2026-08-09

- [x] Approve the production data route: PostgreSQL metadata and revisions, S3-compatible object storage, and immutable Parquet dataset versions. See [`DATABASE_DESIGN.md`](DATABASE_DESIGN.md).
- [x] Approve AWS ECS/Fargate, RDS PostgreSQL 17, S3, SES, Secrets Manager/KMS, CloudWatch, and one-region deployment as the initial production runtime. See [`PERSISTENCE_PHASE6.md`](PERSISTENCE_PHASE6.md).
- [x] Implement versioned `ProjectSpec` v1 in frontend Zod and backend Pydantic, backed by immutable ProjectRevision records and cross-contract fixtures.
- [x] Add authenticated project-description editing API/UI with immutable revision updates, pinned-share behavior, UTF-8 limits, and no speculative Experiment entities.
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
- [x] Add trusted-proxy client identity and atomic multi-host authentication abuse limits before horizontally scaling the API; real staging ALB chain evidence is a Phase 6C acceptance requirement.
- [ ] Complete real-account SES accepted/bounce/complaint, suppression, deployed-IAM, and alarm evidence in Phase 6C; the SES v2 adapter and abuse limits are accepted application code.
- [x] Store PostgreSQL publication exports and datasets through the selected provider-neutral object-storage boundary; keep SQLite BLOBs only in the local/reference adapter.
- [x] Add an explicit authenticated “Save to Cloud” endpoint independent of link sharing.
- [x] Implement immutable share snapshots pinned to ProjectRevision; links do not expire by default, remain revocable, and never change when the working project is edited.
- [x] Retain saved-project publication exports for the lifetime of the project and remove them only after permanent project purge.
- [x] Implement 24-hour saved-project soft-delete recovery; suspend access immediately and purge database/object data after the window.
- [ ] Enforce the production baseline for TLS, managed encryption at rest, environment-separated secrets and KMS keys, seven-day-or-longer PITR, 30-day daily backup retention, and quarterly restore drills.
- [x] Select AWS `ap-southeast-1` as the initial deployment/data Region; keep database, objects, workers, and backups co-located.
- [ ] Finalize configurable saved-project count and total-storage quota values before public launch; keep the approved 50 MB per-upload limit.
- [ ] Complete target-market privacy and compliance review before making GDPR, PIPL, HIPAA, GLP, GxP, or similar claims.

### Phase 6 execution

- [x] Phase 6-0: reconcile repository facts, approve the AWS production topology, preserve cloud-provider boundaries, and assign every remaining responsibility to a subphase.
- [x] Phase 6A: add fail-closed production settings, dependency readiness, process health checks, non-root API/Worker and standalone Next.js images, and deployment-safe examples.
- [x] Phase 6-PRE-1: track the acceptance standard and establish complete Phase 6B/6C/6D implementation, evidence, failure, and rollback contracts.
- [x] Phase 6-PRE-2: separate process liveness, dependency readiness, and Worker operational probes.
- [x] Phase 6-PRE-3: independently rerun the complete admission matrix and record `PASS` in [`PHASE6_PRE_ACCEPTANCE.md`](PHASE6_PRE_ACCEPTANCE.md).
- [x] Phase 6 AWS dependency boundary: accept 6B application code, move infrastructure-dependent SES/IAM/ALB/CloudWatch evidence to the blocking 6C staging gate, and prohibit 6D before 6C acceptance. See [`PHASE6_AWS_DEPENDENCY_MATRIX.md`](PHASE6_AWS_DEPENDENCY_MATRIX.md).
- [x] Phase 6B: add multi-host abuse protection, the SES v2 application boundary, and project-description editing. See [`PERSISTENCE_PHASE6B.md`](PERSISTENCE_PHASE6B.md).
- [x] Phase 6C implementation: commit the production IaC, encrypted backup/restore automation,
  CloudWatch operations, runbooks, alerts, SLO definitions, CI/CD workflows, and rollback logic.
  Static checks and the infrastructure workflow are present in the repository; this does not
  claim a live AWS deployment. See [`PERSISTENCE_PHASE6C.md`](PERSISTENCE_PHASE6C.md).
- [ ] Phase 6C live acceptance: configure account-owned AWS/GitHub/DNS inputs, deploy one exact
  staging candidate, and collect real SES, IAM, ALB, CloudWatch, backup/restore, rollback, and
  cost evidence.
- [ ] Phase 6D: add quotas, privacy/compliance review, load and recovery drills, and production-launch acceptance. See [`PERSISTENCE_PHASE6D.md`](PERSISTENCE_PHASE6D.md).

## Deferred Scientific Domain Model

> V2.1 implements the minimum repeated-acquisition/replicate workflow in `V21-P1-3` above.
> Broader experiment editing, cross-experiment comparison, membership roles, and team permissions
> remain deferred until this vertical slice is validated through user studies.

- [x] Introduce the minimum `Experiment` and physical `ExperimentRun` entities for repeated
  acquisitions, multi-file grouping, replicate identity, batch identity, and owner-scoped access.
- [x] Keep software `ProcessingRun` separate from the physical laboratory `ExperimentRun`; this
  boundary is part of the accepted V21-P1-3 implementation.

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

> V2.1 tracks the first scientifically defensible methods slice under `V21-P1-2` above. These
> individual backlog items remain open until their assumptions, answer-keyed fixtures, and
> export disclosures are accepted.

- [x] Evaluate bootstrap confidence intervals for the first supported residual-bootstrap slice.
- [ ] Evaluate simultaneous confidence bands and prediction intervals.
- [ ] Evaluate robust regression; weighted least squares is supported only with a scientifically
  justified error column and remains bounded by the P1-2 disclosure contract.
- [x] Add residual diagnostics and document multiple-comparison correction as explicitly deferred
  until a comparison design and correction policy are accepted.
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
