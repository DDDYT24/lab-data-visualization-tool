# LabViz Product Backlog (V2.2 active development; V2.1.1 current release)

This file is the single source of truth for future work, completion status, release gates,
compatibility issues, and deferred decisions. Update it first whenever the product status
changes. Other Markdown files may explain a stable contract or preserve historical evidence,
but they must not create a competing backlog.

- **Current release:** V2.1.1 local self-hosted release, published on GitHub
- **Status review:** 2026-09-12
- **Working line:** V2.2.0-dev local-first development; not a release tag or installer
- **Version history:** [`../VERSION_BASELINE.md`](../VERSION_BASELINE.md)

This file records the V2.0 baseline, completed V2.1 work, the ordered V2.2 plan, and explicitly
deferred work.

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

Large-surface performance, accessible alternatives, mobile 3D, native packaging, cloud sync,
external identity, and team workspaces have been promoted to the numbered V2.2 tracks below. AWS
Phase 6C/6D remains an optional source of live infrastructure evidence for `V22-P6`; billing remains
post-V2.2 work.

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

## V2.2 Roadmap (active; V2.2.0-dev)

**Planning checkpoint:** 2026-09-12

**Release intent:** Make the local-first product easier to learn, more comfortable on desktop and
mobile, and scientifically stronger without weakening the V2.1 data and privacy contracts. V2.2
must ship user-facing examples, contextual guidance, a privacy-safe feedback path, dark mode,
advanced mobile 3D, and a bounded advanced-analysis slice. A Windows installer is the first native
packaging target. Cloud synchronization, team collaboration, and external identity are an optional
server-backed preview track and must not block or silently change the local release.

### P0 - freeze contracts and expand the test foundation

- [x] **V22-P0-1: Define one versioned example catalog and fixture manifest.** Each entry must have
  a stable slug, bilingual title and description, format, row/column shape, field roles, intended
  chart, learning goal, expected quality findings, and answer-keyed output facts. Bundle examples
  with the application so they work offline and label every value as synthetic.
  - **Acceptance:** at least seven user-facing examples cover time-series lines, grouped repeated
    runs with uncertainty, X/Y scatter and fitting, categorical comparison, distributions,
    correlation heatmaps, and a regular X/Y/Z surface. Format mirrors cover CSV, TSV/TXT, JSON,
    and multi-sheet XLSX; negative fixtures cover malformed headers, mixed types, missing values,
  outliers, duplicate surface coordinates, incomplete grids, non-finite values, BOM headers, and
  Unicode filenames.
  - **Evidence (2026-09-12):** `V2.0/api/samples/v22/manifest.json` contains seven public and nine
    edge synthetic entries. `tests/test_v22_samples.py` validates the version, formats, schemas,
    quality expectations, and tracked byte sizes.
- [x] **V22-P0-2: Turn the data corpus into one repeatable matrix runner.** Keep the existing
  23-dataset V2.1 matrix, add every public example and V2.2 edge fixture, and run the same manifest
  through import, preview, quality, cleaning, recommendation, chart analysis, and PNG/SVG/PDF plus
  cleaned-data export. Store machine-readable results and fail when an answer key changes without
  an explicit review.
  - **Acceptance:** every dataset records pass/fail/skip with a reason; public examples complete the
    browser workflow; invalid inputs fail with localized, actionable errors; the runner never counts
    an unconfigured cloud, live-mail, private-file, operating-system, or GPU check as passed.
  - **Evidence and boundary (2026-09-13):** `python -m scripts.run_v22_matrix --output <ignored-json>`
    passes all seven public fixtures, all nine V2.2 edge fixtures, and all 23 historical V2.1
    datasets. Valid datasets run import validation, preview, quality, cleaning, recommendation,
    chart analysis, cleaned CSV, and PNG/SVG/PDF export; expected-invalid datasets compare stable
    error codes. The historical corpus remains an ignored local synthetic test package rather than
    tracked user data, so a clean release machine must provide it with `--historical-root`; a missing
    corpus is reported as `skip`, never `pass`.
- [ ] **V22-P0-3: Establish user-comfort performance and compatibility budgets before feature work.**
  Measure import-to-ready time, first-chart render, chart interaction, export, and peak memory on a
  small example, a medium dataset, a large dataset near the documented local limit, and 21 x 21 and
  101 x 101 surfaces. Record the reference machine and set reviewed regression thresholds rather
  than relying on subjective impressions.
  - **Acceptance:** Chromium, Firefox, and WebKit desktop smoke flows pass; the supported mobile
    viewport and touch flow pass; large-data sampling is disclosed; no supported case crashes,
    freezes without progress, or hides a failure behind an empty chart.
  - **Current evidence and limitation (2026-09-13):** on Windows 11 / Python 3.12.14 / pandas 3.0.5,
    tables with 24/1,000/10,000 rows load in 10.08/1.92/6.12 ms, run quality checks in
    12.20/10.34/20.16 ms, analyze in 4.66/8.95/11.22 ms, and render PNG in
    190.08/174.62/172.10 ms. The 21×21 and 101×101 surfaces complete quality/analysis/PNG in
    35.88/43.37/303.82 ms and 127.23/80.81/319.69 ms. Chromium, mobile Chromium, Firefox, and
    WebKit smoke flows pass, including disclosed large-surface sampling. Process-level peak memory,
    browser interaction timing, and reviewed pass/fail budgets remain open, so this item stays
    unchecked.

### P1 - example gallery and first-run learning

- [x] **V22-P1-1: Replace the single fixed sample button with an example chooser.** Present compact
  cards grouped by learning goal, with a tiny schema preview, recommended visualization, difficulty,
  expected quality lesson, and an explicit `Open example` action. Keep normal file upload equally
  prominent and allow users to reopen the chooser from Help and an empty workspace.
  - **Evidence (2026-09-12):** `example-gallery.spec.ts` verifies seven cards, bilingual content,
    keyboard activation, and a selected sample route; the catalog is bundled and served by the
    local API.
- [x] **V22-P1-2: Generalize the sample API and preserve offline behavior.** Replace the hard-coded
  thermal-response route with a catalog endpoint and stable per-slug project creation while keeping
  the old route as a compatibility alias for one release. Example projects must pass through the
  real processing pipeline and must not contain a privileged shortcut unavailable to uploaded data.
  - **Evidence (2026-09-12):** `test_v22_samples.py` exercises every public catalog entry through
    import, preview, quality, cleaning, recommendation, analysis, cleaned-data export, and PNG/SVG/PDF
    exports. The old sample route remains covered by the API compatibility tests.
- [x] **V22-P1-3: Test every example as a teaching workflow.** For each card, assert its metadata,
  intended recommendation, important quality findings, editable chart, export provenance, browser
  back/refresh behavior, English/Chinese copy, keyboard access, and mobile layout. Add a regression
  preventing a card from pointing to a missing or mismatched fixture.
  - **Evidence (2026-09-13):** `test_v22_samples.py` validates every card-to-fixture mapping,
    bilingual metadata, recommendation/quality answer keys, real processing, export provenance, and
    all four output formats. `example-gallery.spec.ts` opens each of the seven cards by keyboard and
    verifies metadata, route, persisted refresh state, and browser back behavior; the chooser
    renders all seven Chinese titles, and `mobile.spec.ts` opens all seven cards without horizontal
    overflow. Browser tests mock transport while the API suite independently exercises the real
    processing path; no sample-only processing shortcut is counted.

### P2 - help, contextual tips, and feedback center

- [x] **V22-P2-1: Restructure Help around user tasks.** Add searchable sections for choosing data,
  fixing import problems, understanding quality warnings, selecting a chart, fitting and intervals,
  3D surfaces, exporting, privacy/storage, and troubleshooting. Link relevant example cards from
  each section and show the supported table shape next to the explanation.
  - **Evidence (2026-09-12):** Help search, workflow explanation, examples link, and localized
    guidance cards are covered by the Help browser tests.
- [x] **V22-P2-2: Add short contextual tips without interrupting work.** Show dismissible tips only
  where a decision is made, such as sheet/header selection, keep/exclude/remove, X/Y/Z assignment,
  confidence versus prediction intervals, sampling, and export resolution. Tips must be bilingual,
  keyboard accessible, locally stored, and recoverable from Help after dismissal.
  - **Evidence (2026-09-12):** contextual-tip unit/browser tests verify dismissal and recovery;
    tips are stored in localStorage and expose accessible controls.
- [x] **V22-P2-3: Add a privacy-safe local feedback center.** Let users choose bug, usability,
  scientific-method, or feature-request feedback; preview and copy a diagnostic summary; and open a
  prefilled GitHub issue when the user explicitly continues. By default include only app version,
  operating system/browser class, current screen, and a user-written description - never source
  rows, chart values, filenames, email addresses, project titles, or persistent identifiers.
  - **Acceptance:** the center works without an account, remains useful offline through copy/save,
     warns before opening an external site, and has an automated privacy regression proving that
     protected fields and raw data are absent. A future server submission endpoint requires separate
     consent, retention, abuse-protection, and deletion contracts.
  - **Evidence (2026-09-12):** `help-feedback.spec.ts` and diagnostic unit tests verify offline copy,
    external-site warning behavior, and exclusion of project IDs, filenames, source rows, and other
    protected fields from the diagnostic preview.

### P3 - appearance, accessibility, and advanced mobile 3D

- [x] **V22-P3-1: Implement a complete dark theme.** Provide system, light, and dark choices; persist
  the selection locally; theme every loading/empty/error/dialog/table/chart state; and define dark
  chart palettes independently from publication export defaults. PNG/SVG/PDF must remain readable
  and must not unexpectedly inherit a dark background.
  - **Evidence (2026-09-12):** `appearance.spec.ts` verifies explicit dark persistence and system
    scheme changes; the theme uses a separate dark chart palette and export settings remain backend
    controlled. The browser appearance suite passes.
- [x] **V22-P3-2: Harden color and non-visual access.** Verify WCAG AA text and control contrast,
  serious/critical Axe results, visible focus, reduced motion, color-vision-safe chart palettes,
  grayscale/print behavior, and an accessible data-table alternative for every chart family.
  - **Evidence (2026-09-12):** the dark action contrast regression passes at the 4.5:1 AA threshold;
    the visual/accessibility suite reports no serious or critical Axe violations; the grayscale
    render contract and chart data-table tests pass. Full manual screen-reader certification is not
    claimed.
- [x] **V22-P3-3: Make 3D usable on supported mobile screens.** Add documented one-finger rotate,
  two-finger zoom/pan, reset-view, axis/legend toggles, loading/progress, and a lower-cost fallback
  for devices that cannot render the full surface smoothly. Never replace scientific points without
  disclosing the sampling or fallback.
  - **Acceptance:** touch gestures do not conflict with page scrolling, controls meet the minimum
    touch-target size, orientation changes preserve the chart specification, and the regular,
    incomplete, duplicated, and larger-surface fixtures pass mobile browser tests.
  - **Evidence and limitation (2026-09-13):** mobile Chromium covers the regular 21×21 surface,
    duplicate-coordinate and missing-cell diagnostics, and a 101×101 surface with disclosed
    low-cost sampling. It sends a real two-touch browser event, checks `pan-y` page-scroll
    coexistence, 44×44 controls, rotate/reset/axis/legend controls, orientation preservation, and
    the bilingual gesture guidance. This is automated browser-device emulation, not a claim of
    certification on every physical phone or GPU.

### P4 - bounded advanced statistics

- [x] **V22-P4-1: Add prediction intervals as a separate result from mean-response confidence
  intervals.** Define supported models and assumptions first; expose sample size, exclusions,
  confidence level, interval meaning, and limitations in the UI and exported figure.
  - **Evidence (2026-09-12):** the prediction answer key and API/frontend/export tests verify the
    separate interval kind, wider prediction width, sample/exclusion evidence, and Student-t method.
- [x] **V22-P4-2: Add the first simultaneous confidence-band method.** Name the method, document the
  family of values it covers, and prevent it from being presented as interchangeable with pointwise
  bootstrap bands or prediction intervals.
  - **Evidence (2026-09-12):** Working–Hotelling is named in the API, UI, export footer, contract,
    and answer-keyed tests; prediction, pointwise, and simultaneous disclosures are distinct.
- [x] **V22-P4-3: Add one robust-regression method only after its contract is accepted.** Document the
  loss/influence rule, convergence behavior, scaling, unsupported cases, and the fact that robust
  fitting does not make biased or poorly designed data valid.
  - **Evidence (2026-09-12):** Huber IRLS is restricted to linear fitting, records iterations and
    convergence, rejects unsupported models and confidence bands, and is covered by the robust
    answer-key case and failure-code tests.
- [x] **V22-P4-4: Require independent answer keys for every shipped method.** Compare coefficients,
  intervals, residual facts, and failure states against an independently implemented reference on
  clean, noisy, outlier-heavy, heteroscedastic, insufficient, singular, and misleading datasets.
  The API, localized UI, chart preview, and all export formats must disclose the same method facts.
  - **Evidence (2026-09-12):** `statistics_answer_keys.json` is synthetic-only and hand-authored;
    `test_v22_statistics.py` checks deterministic coefficients/intervals, residual and failure
    cases, API payloads, and PNG/SVG/PDF signatures. The answer key is not a substitute for a
    domain expert review of a user's study design.

> User-defined fitting models and broad multiple-comparison automation remain post-V2.2 research
> until safe expression validation, reproducibility, and scientifically defensible guidance are
> specified.

### P5 - local packaging and upgrade experience

- [x] **V22-P5-1: Decide and document the native packaging architecture.** Prefer a bundled local
  runtime that opens LabViz without requiring Git, Python, Node.js, or PowerShell knowledge. Define
  ports, process lifecycle, updates, logs, data location, backup, crash recovery, and uninstall data
  handling before selecting the packager.
  - **Evidence (2026-09-12):** `V2.0/packaging/windows/package-manifest.json`, the packaging README,
    portable launcher contract, icon sources, and PowerShell validator define per-user paths,
    loopback ports, bundled-runtime expectations, health checks, upgrade/rollback/uninstall data
    rules, and forbidden user-data artifacts. Contract tests and PowerShell parser checks pass.
- [ ] **V22-P5-2: Ship and test the Windows installer first.** Verify clean install, offline launch,
  upgrade from V2.1.1, rollback, repair/reinstall, non-administrator behavior where supported,
  antivirus/signing expectations, paths with spaces and non-ASCII characters, and uninstall with an
  explicit keep/delete-local-data choice.
  - **Current limitation (2026-09-12):** no installer compiler, bundled CPython/Node runtimes,
    signed artifact, or clean Windows machine is available in this repository. This item remains
    open; the portable launcher is a contract, not an installer or offline-install evidence.
- [ ] **V22-P5-3: Reuse the accepted packaging contract for macOS and Linux.** Do not advertise a
  platform package until it passes clean-machine installation, launch, upgrade, export, backup, and
  removal tests on named supported versions. macOS/Linux packaging may follow V2.2 Windows GA as a
  V2.2.x deliverable if signing hardware or test machines are unavailable.
  - **Current limitation (2026-09-12):** no macOS or Linux native package is implemented or
    advertised; support remains the developer checkout path only.

### P6 - optional server-backed collaboration preview

This track increases hosting, storage, identity, security, abuse-prevention, monitoring, support,
and compliance cost. It is not part of the offline local definition of done and must stay behind an
explicit deployment/profile boundary.

- [ ] **V22-P6-1: Approve a measured cloud cost and data-governance model** before enabling cloud
  synchronization. Define storage and project quotas, retention/deletion, encryption, backup and
  restore, regional placement, audit events, incident response, and who operates the service.
- [ ] **V22-P6-2: Define two-way synchronization semantics** for stable identities, offline edits,
  conflicts, duplicate devices, deletion propagation, partial uploads, retries, encryption, and
  downgrade back to local-only use. Build destructive conflict and recovery tests before rollout.
- [ ] **V22-P6-3: Add Google and Microsoft OIDC only after the server session model is accepted.**
  Test account linking, provider-email changes, revoked consent, duplicate identities, token expiry,
  CSRF/state/nonce validation, logout, and account deletion. Login must not imply Drive or Graph file
  access unless separately requested and consented.
- [ ] **V22-P6-4: Add team workspaces through least-privilege roles.** Define owner/admin/editor/
  viewer capabilities, invitations, removal, ownership transfer, project isolation, audit history,
  and concurrent-edit conflict behavior. Cross-tenant access tests and real staging evidence are
  mandatory; mocks alone do not count as acceptance.

### P7 - release candidate and V2.2 definition of done

**Local checkpoint (2026-09-13, based on `292bdad`):** see
[`docs/V2.2_P7_TEST_REPORT.md`](docs/V2.2_P7_TEST_REPORT.md). Frontend verification, 50 Playwright
checks across Chromium/mobile Chromium/Firefox/WebKit, all 254 API tests with real local
PostgreSQL 17 and MinIO, static Python gates, and the 7 + 9 + 23 data matrix pass. Two live-API
browser tests were intentionally skipped in the mock-backed browser matrix. Native installer,
clean-machine/platform gates, and reviewed performance/memory budgets remain unavailable.
Therefore P7 remains open.

- [ ] All seven or more public examples are bundled offline, have bilingual teaching metadata, and
  complete import-to-export browser workflows with answer-keyed results.
- [ ] The existing 23-dataset V2.1 regression matrix and every V2.2 example, edge, format-parity,
  statistics, mobile, theme, privacy, and packaging gate pass with a machine-readable report.
- [ ] `ruff`, formatting, MyPy, the complete API/PostgreSQL/MinIO gate, `npm run verify`, desktop and
  mobile Playwright, visual/accessibility checks, and clean-install tests pass at the release commit.
- [ ] No serious or critical accessibility issue remains; reviewed performance budgets pass or an
  explicit limitation and fallback is shown before the user starts the expensive operation.
- [ ] Help, contextual tips, feedback privacy, theme behavior, every new scientific method, and all
  supported package/upgrade instructions are complete in English and Simplified Chinese.
- [ ] Release notes and `VERSION_BASELINE.md` distinguish shipped local features, optional preview
  features, skipped external checks, and deferred work. V2.2 is not marked complete until this list
  is supported by reproducible evidence from the release commit.

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
recommendation behavior, dedicated browser preview/export regression, beginner guidance,
scientific-method slice, and experiment-level model are complete under `V21-P0-1` through
`V21-P1-3`.

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

Native packaging is now tracked only by `V22-P5`. Optional local/cloud synchronization is tracked
only by `V22-P6-1` and `V22-P6-2`.

## Authentication

Google and Microsoft sign-in are now tracked only by `V22-P6-3` and remain optional server-backed
preview work.

## Appearance and Accessibility

Dark mode, accessible alternatives, color-vision palettes, and print behavior are now tracked only
by `V22-P3`.

## Advanced Analysis

> V2.1 tracks the first scientifically defensible methods slice under `V21-P1-2` above. These
> individual backlog items remain open until their assumptions, answer-keyed fixtures, and
> export disclosures are accepted.

- [x] Evaluate bootstrap confidence intervals for the first supported residual-bootstrap slice.
- [x] Add residual diagnostics and document multiple-comparison correction as explicitly deferred
  until a comparison design and correction policy are accepted.
- [ ] Evaluate user-defined fitting models with strong validation and guidance.

Prediction intervals, simultaneous confidence bands, robust regression, and their answer-keyed
evidence are now tracked only by `V22-P4`.

## Cloud and Commercial Features

- [ ] Define free, paid researcher, and team cloud plans using observed product usage and operating cost.
- [ ] Decide cloud project, storage, file-size, and sharing-link quotas.
- [ ] Evaluate six-panel figures as part of a paid cloud workflow.
- [ ] Evaluate batch processing, batch export, journal presets, and priority processing.
- [ ] Add billing only after the free workflow and activation metrics are validated.
- [ ] Evaluate organization administration and institutional controls after the bounded
  `V22-P6-4` team-workspace preview has real usage and security evidence.

## Mobile

- [ ] Evaluate advanced mobile data cleaning after the desktop workflow is stable.
- [ ] Evaluate mobile multi-panel editing after the V2.2 advanced-3D controls are accepted.
