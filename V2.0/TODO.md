# LabViz Product Backlog (V2.2.0 local release; V2.3 separate development)

This file is the single source of truth for future work, completion status, release gates,
compatibility issues, and deferred decisions. Update it first whenever the product status
changes. Other Markdown files may explain a stable contract or preserve historical evidence,
but they must not create a competing backlog.

- **Current release:** V2.2.0 Windows local release, single ordinary Windows profile
- **Status review:** 2026-10-07
- **Frozen release line:** V2.2.0; see publication decision and release assets below
- **Next development roadmap:** [V2.3 Local Research Workspace](#v23-roadmap), planned on 2026-09-27; no V2.3 implementation or release is claimed
- **Version history:** [`../VERSION_BASELINE.md`](../VERSION_BASELINE.md)

This file records the V2.0 baseline, completed V2.1 work, the ordered V2.2 plan, the V2.3
development scope, and explicitly deferred work. The V2.3 planning date does not refresh the
V2.2 verification checkpoint or change current application/release versions.


## V2.2.0 publication decision — 2026-10-07 (current authority)

The owner reports the preceding candidate's clean/offline/restart acceptance complete and
explicitly authorizes source/version freeze, complete bilingual README and installer upload.
Record this as owner testimony, not independently observed/hash-attested final-EXE evidence.
Two real Windows accounts were not tested; publication is limited to one ordinary Windows
profile and does not claim filesystem/concurrent-loopback isolation between accounts.
Authentic V2.1.1 installer upgrade/rollback acceptance remains owner-waived. Cloud remains deferred.

API/Web/package versions are promoted to 2.2.0; the release is rebuilt from one frozen commit.
Exact source, build ID, installer SHA-256, fresh packaged checks and local antivirus result
are attached to the [v2.2.0 release](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0).
The installer is unsigned; signing and universal antivirus reputation are not claimed.
The final metadata rebuild is distinct from the preceding owner-tested development package.

This decision supersedes earlier open-release summaries below for the supported single-user
scope. Historical counts, failure records and original artifact boundaries are retained.


### Histogram preview/export content repair (2026-10-04)

- The owner reported that downloads succeeded but the 05 histogram preview and exported PNG
  differed. The earlier general "testing passed" report remains recorded at its scope; it did
  not establish chart-content parity. The confirmed defect was in browser interval geometry:
  API bins were collapsed to center/count pairs and drawn as ordinary bars with inferred width,
  gaps and an unwanted zero-inclusive X scale. Counts agreed; preview interval widths did not.
- The browser now draws full API start/end/count intervals with a custom rectangle renderer,
  preserves empty-bin gaps, uses the shared 0.55 opacity and 5% axis padding, and centers axis
  titles while suppressing crowded range-end/floating-point labels. API bins remain authoritative;
  the client no longer computes a different fallback histogram. Multi-panel histogram series
  use their specified panel. Python exports use the same explicit padding/opacity contract.
- Real packaged Chromium E2E detects the defect in the prior candidate for both raw uploaded
  files 04/05. Final repaired candidate `outputs/v22-histogram-parity-candidate-20261003`
  (Web build ID `gBSNfDEov26rbEkW4xEnB`) passes **4/4** live workflows in **2.3 minutes**:
  raw 04 and 05 upload/preview/reopen/export-content checks, all seven bundled examples with
  PNG/SVG/PDF, and local history/gallery/privacy workflow. Checks independently count raw rows
  within API bins, inspect actual preview SVG rectangle widths/heights, exported SVG intervals,
  PNG pixel heights, and PDF vector intervals/frequencies. Reports and actual downloaded figures
  are retained under `outputs/v22-histogram-parity-final-confirmation-20261004/browser-results`.
  An initial repaired run passed content checks but failed an incorrect test expectation that
  reopening without `?step=export` stays on the export step; the test now explicitly reopens that
  route. Final formatting normalization preserved Python AST and the four workflows were rerun.
- Frontend TypeScript, ESLint, production build and **60/60** Vitest passed; Python render-
  contract and sample tests **6/6**, Ruff check and formatting passed. The sample test's first
  invocation could not access the shared system temporary directory; a new workspace-scoped
  temporary root resolved that environment error. Source/staged Python bytes match; verification
  summary is `outputs/v22-histogram-parity-final-confirmation-20261004/verification-summary.json`.
  Replacement installer compilation and delivery preparation passed.
  Current-host isolated-data results do not close clean/offline installation, two-account
  isolation, signing/distribution decisions or final frozen-release P7.
- Owner workflow requirement: after every completed feature/fix, immediately run the relevant
  real-API end-to-end path, including actual exported content where applicable; file signatures
  and successful download buttons alone are insufficient. Preserve V2.2/V2.3 data isolation.
- New unsigned repair installer: `outputs/v22-histogram-parity-installer-20261004/LabViz-Setup-2.2.0.exe`,
  148,080,528 bytes, SHA-256
  `7890089F912E0F0A4450B5B871F18B0D2527739E1DA109B52CF004895AAF092E`.
  `LabViz-Export-Fix-20261004.zip` in the same directory bundles this EXE, 04/05 data, instructions,
  hash verification tools, source/build provenance and the verification summary. This replaces
  the 2026-10-03 single-value repair candidate for future owner acceptance; historical reports
  remain at their original artifact scope. It is still a `2.2.0-dev` candidate, not a release.

### Owner repair follow-up and remaining release gates (2026-10-03)

- After the new 04/05 repair package and same-path update instructions were supplied, the user
  reported "testing passed" and asked what remains before release. Record owner-reported
  repair/update acceptance in this conversation's new-candidate context. The report did not
  enumerate individual checks or provide a package-verification JSON; do not infer every
  export format, old-history recovery, disconnected Windows reboot, duplicate-launch behavior,
  clean/offline first installation or second-account isolation from this general statement.
- Remaining for the current replacement candidate: installer identity and scoped owner
  offline/reboot evidence; clean/offline install/first start; real two-account file/loopback
  isolation before claiming that boundary; final real-API controls/error regressions and P7
  rerun from one frozen source/artifact; public-distribution signing/antivirus decision and
  release metadata. Previously accepted V2.1.1 waiver and deferred cloud scope are unchanged.

### Owner offline follow-up and 04/05 repair (2026-10-03)

- The user reported normal disconnected operation on their own computer, including viewing
  history and exporting images. This is owner-reported development-host offline success for
  their installed copy. The earlier agent hash report identifies the supplied EXE on this
  host, not independently the installed files; no clean/offline first-install or two-account
  pass is inferred. The screenshot confirms 04's invalid line chart still selected
  `measurement` for both X and response. The user explicitly requested repair of 04/05.
- Source repair and replacement candidate built: backend single-value defaults now select histogram; frontend direct
  uploads and reopened legacy X/response-conflict projects use a valid histogram configuration.
  Distribution controls omit the independent X selector, keep the response selectable and
  explain pooled values. One-value line/scatter/bar selections are disabled. User titles,
  colors and export settings are retained where possible; source data is not changed.
- Categorical group comparisons remain a separate limit: histogram/box values are pooled and
  the UI now says so. The repair is for the reported invalid default, not an implementation of
  grouped distributions. The earlier launcher bootstrap repair will accompany the new package.
  TypeScript, ESLint, Python/PowerShell parsing, isolated production compilation and package
  structure checks passed. Inno Setup compilation and delivery ZIP integrity checks passed.
  No runtime tests have been added or run for this repair; new-package installation and
  functional acceptance remain pending. Keep final P7 open.
- Replacement installer: `outputs/v22-single-value-fix-installer-20261003/LabViz-Setup-2.2.0.exe`,
  147,967,756 bytes, SHA-256
  `CE940E028FC6BC5EA34F5CC6E03190273290EBA7C4485B1BB869331F19D6D334`.
  It remains an unsigned `2.2.0-dev` candidate (installer display version `2.2.0`), with Web
  build ID `MrgUPaA6MCOcthBPeSfSo`, bundled Python 3.13.7 and Node 24.17.0. The same folder's
  `LabViz-04-05-Fix-20261003.zip` includes the EXE, 04/05 files, update/retest instructions,
  new-hash verification scripts and source/build provenance. Historical owner reports are
  retained; they do not automatically become functional passes for this replacement hash.

### School owner acceptance: exports and colors (2026-10-02)

- After the earlier import recovery and 04/05 field-selection guidance, the user reported that
  everything was working on the school computer, explicitly including export and color
  adjustments. Record these as owner-reported manual successes. No new agent-run test was
  executed for this update, and individual export formats, datasets or chart configurations
  were not enumerated by the user.
- Follow-up clarification: the school computer remained connected by Ethernet; no disconnected
  test was attempted. In response to the Windows-reboot history/data/figure/open/download
  question, the user confirmed the other checks worked. Record that as owner-reported online
  reboot/recovery success, retaining the earlier first-import failure. The user did not run
  package verification on the school computer. Its installed-artifact hash, Windows/browser
  versions and two-account isolation remain unconfirmed.
- The owner can supplement disconnected operation on their own development computer, but
  that does not establish clean-machine disconnected installation/first-start acceptance.
  The verification script generates JSON automatically; no handwritten report is required.
  The agent ran that script on the current development host and obtained SHA-256 MATCH for the
  copied 2026-09-26 EXE, Windows build `26200.9457`, saved under
  `outputs/v22-acceptance-delivery-20261001/LabViz-V22-Acceptance/results/package-verification-20261002-222259.json`.
  This identifies the copied installer on this host only, not the school installation or a
  disconnected functional result. Do not close L5/P7 or repeat the confirmed school online
  export/color/recovery checks as if they had not been done.
- Retain the initial 403 failure and the unsafe single-value default configuration as separate
  records. The later manual success does not establish that either source defect has been
  repaired in the delivered installer. The launcher source repair remains unbundled; direct-
  upload default and categorical group-comparison repairs/regressions remain pending.

### Single-value uploaded examples 04/05 (2026-10-02)

- The user reported "X field must be different from every response field" when generating
  output for uploaded `04_group_comparison.txt` and `05_distribution.json`. Their exact selected
  chart types were not captured. Both datasets contain only one numeric column, `measurement`.
  Source inspection confirms that direct-upload defaults can select that same column for X and
  response while leaving the chart type as line; the API rejects this configuration. Built-in
  sample recommendations do not validate this direct-upload path.
- Owner workaround to verify: choose box for 04 or histogram for 05, response `measurement`,
  grouping none, fitting none and uncertainty none. Histogram/box validation does not require
  a different X field. Current box/histogram implementations pool the selected response column
  and do not support categorical `groupField`; therefore a successful pooled export does not
  prove the examples' between-group comparison goal. Earlier user-facing instructions to select
  `group` for 05's box plot were incorrect for this candidate.
- Keep direct-upload single-value defaults and categorical group-comparison coverage open for
  repair and final candidate regression. The later owner export/color success above is broad
  and does not identify the 04/05 configurations individually. No source repair for this defect
  or rebuilt installer has been recorded yet.

### School-computer import follow-up (2026-10-02)

- The user reported that uploading `00_basic_acceptance.csv` on the school computer, using
  the browser opened by the LabViz shortcut, showed "Project data is unavailable" and
  "Open LabViz from the launcher for this Windows account." This message corresponds to the
  API's 403 `local-session-required` response. The user subsequently reported that import
  worked after restarting. Preserve both the initial failure and the recovery.
- This is owner-reported import recovery only. The restart type (application or Windows),
  Windows/browser versions, disconnected state and hash of the installed artifact were not
  confirmed in this report. It does not close exact-installer clean/offline, history/export,
  two-account or final P7 acceptance.
- A separate confirmed source defect was repaired: the installed launcher's already-running
  branch now decrypts the current Windows user's DPAPI credential and opens the bootstrap URL,
  instead of opening a bare URL without authorization. It is a possible cause, not a confirmed
  explanation of the school incident. PowerShell parsing passed; launcher/browser runtime
  regression and packaging of this change remain pending. The previously delivered installer
  and its SHA-256 have not changed. The optional school session diagnostic need not be run
  while the user's import is working.

### User acceptance update (2026-09-15)

- At that checkpoint, the user reported disconnected operation on a clean school computer,
  keyboard-only import, and Narrator checks passed. These remain owner-reported results for the
  build tested there; no exact candidate hash or Windows version was attached.
- At that checkpoint, migration/replacement and phone testing were unresolved. The 2026-09-22
  update below records the user's waiver of V2.1.1 replacement/rollback for personal use and
  clarifies that the supplied “phone” screenshot is desktop device emulation, not a physical phone.
  Other reported manual checks are not formal statistical certification.
- User confirms the color issue is resolved. The supplied screenshot describes configured colors
  working on ungrouped charts including dark mode, and disabled color inputs with explanations
  when grouping or grayscale overrides them. Corresponding source edits are present in the
  staged portable candidate; appearance/theme regressions pass in the current automated browser
  suite. This does not claim a hash-linked human visual review on a clean machine.
- At that checkpoint, signing/antivirus had not been done. They are distribution-only gates for
  the current private single-user use; the current candidate and remaining release gates are
  summarized in the 2026-09-23 checkpoint below.

### User acceptance update (2026-09-22)

- User clarified that the reported "phone test" was Chrome DevTools device emulation on a
  desktop browser. The supplied screenshot shows an Asus Zenbook Fold preset at 853 x 1280
  against 127.0.0.1:3000, not a physical phone. Count this as desktop responsive emulation;
  real-phone/touch/mobile-GPU behavior remains unverified. Do not describe it as a real-device pass.
- For the current single-user, local-only V2.2 scope, the user explicitly waives a real
  V2.1.1-to-V2.2 replacement and rollback test. This is a scope decision, not passing evidence:
  do not promise supported migration or rollback until an actual V2.1.1 path is tested. Restore
  that gate before a wider installer release or a claim of backward-compatible upgrades.
- Code signing and antivirus review have not been performed. A purchased/trusted signing route
  is optional for the user's private unsigned use; do not label the installer signed or the
  release security-reviewed. Exact-candidate packaging and release regression remain open.
- Synthetic chaos-data rerun on Python 3.12.14: 100/100 deterministic datasets passed the
  real processing functions for upload validation, import, preview, quality, cleaning,
  analysis and expected export behavior. The matrix includes 80 surface datasets
  (52 valid, 28 intentionally invalid), 20 2D datasets, 24 CSV, 23 TSV, 25 TXT, 24 JSON,
  and 4 multi-sheet XLSX. All 72 valid charts produced PNG/SVG/PDF; the 28 invalid
  charts were rejected with the expected diagnostic or numeric-field code. Cases cover
  regular/large grids (including 101 x 101), shuffled/reversed coordinates, anisotropic
  scales, noise, BOM and Chinese headers/filenames, missing/mixed values, duplicates,
  incomplete/irregular grids, constant and collinear axes, and planar outliers.
  Machine-readable evidence and the synthetic-only ZIP are ignored under
  `outputs/v22-chaos-100-20260922/`; reusable generation/verification code is
  `V2.0/api/scripts/chaos_100_matrix.py` with its XLSX builder. No experimental
  user data is included.
  The existing 7 public + 9 edge + 23 historical matrix also passed 39/39;
  `tests/test_v22_samples.py` and `tests/test_api.py` passed 28/28.
  This run does not establish browser/GPU visual parity, real-device behavior, or
  release-candidate packaging; the screenshot establishes only desktop device emulation.

### V2.2 audit and repairs (2026-09-15)

- Repaired surface and correlation-heatmap preview/export palette parity using versioned render
  colors. Correlation remains fixed at -1..1; surface color limits come from all points even when
  preview sampling omits an extreme. Grayscale follows the same contract.
- Replaced flattened-array surface sampling with Cartesian-axis sampling, preserving edge
  coordinates and complete grid connectivity. Added a 101x101 regression with an unsampled peak.
- Increased the default camera distance and reserved layout space for 3D labels/color scale.
  Axis visibility now includes labels; grid and color-scale visibility respect their controls.
- Disabled ineffective series-color inputs for surfaces/heatmaps, with bilingual explanations.
  Removed the redundant preview color wrapper; retained the existing custom-color repair.
- Corrected the obsolete console-OTP E2E text assertion and the stale P7 peak-memory value.
- Removed the unused dark-only chart palette. Public-document privacy scanning now prunes the
  same excluded dependency/output directories before traversal and reads each document once;
  it no longer walks bundled runtimes only to discard them afterward. Four regression checks pass.
- Current validation: 52 frontend unit tests, ESLint/TypeScript, Ruff and strict MyPy pass.
  Final serial browser run: 55 pass, 5 opt-in live tests skipped; desktop/mobile 3D targeted run:
  8 pass. Local API/render/examples/statistics/privacy/packaging selection: 42 pass. Production
  build passes and the repaired 3D preview was visually checked. An earlier overlapping rebuild
  invalidated two WebKit results; the final serial run supersedes that interrupted evidence.
  The 7+9+23 data matrix passed; the final palette contract and seven-example export checks
  passed separately after the heatmap repair. Full API attempt: 124 passed, 95 skipped,
  40 setup errors and 4 failures, all unexecuted/failing paths require unavailable local
  PostgreSQL/MinIO. That attempt did not pass; it is superseded by the integration rerun below.
- **Integration rerun after Docker recovery (2026-09-15): 263 passed, 0 skipped, 0 failures,
  0 errors in 204.03 seconds**, using Python 3.12.14 and a fresh dedicated Compose project
  `labviz-audit-20260915-integration` with PostgreSQL 17 and MinIO. This covers the current
  uncommitted audit repairs as well as the previously blocked database/storage tests. Evidence:
  `V2.0/api/outputs/v22-audit-20260915-integration.xml` (ignored local JUnit report).
  The temporary project and its synthetic-data volumes were removed after verification.
- Existing installer artifacts predate these repairs. Final live-API browser and
  rebuilt-package verification remain required before release; see the user acceptance above
  for the clean-school-machine, keyboard and Narrator feedback.

### Historical automated verification checkpoint (2026-09-14)

Historical automated checkpoint; the user acceptance update above supersedes its statements
about unavailable clean-machine, keyboard and screen-reader checks.

This checkpoint supersedes older statements below about proposed budgets, missing CJK assets,
and browser coverage, but does not close P7 or declare V2.2 released.

- Performance implementation enforces the versioned budgets, records PNG/SVG/PDF stages,
  median/range and process peak memory, and fails if peak-memory measurement is unavailable.
  Python 3.12.14 completed three repetitions for 24/1,000/10,000 rows and 21×21/101×101
  surfaces with zero violations; the recorded peak was 212.38 MiB. The budget file is
  `status: enforced` for this reference machine, not a guarantee for every machine.
- The real seven-example workflow passed through FastAPI in Chromium, Firefox, and WebKit.
  Each engine checked recommendations, editable charts, PNG/SVG/PDF downloads, cleaned CSV
  downloads, keyboard access, refresh/back behavior, and bilingual gallery content. The two
  live API compatibility flows also passed.
- The mock browser suite passed 55 checks with 5 intentional opt-in live skips. Repeated
  browser performance passed 12/12 across Chromium, Firefox, WebKit, and mobile Chromium.
  Frontend verification passed 48 Vitest tests, TypeScript, ESLint and production build.
- The full API/PostgreSQL/MinIO suite passed 263 tests on an independent local Compose project.
  The 7 public + 9 edge + 23 historical synthetic matrix also passed. No experimental user
  data was uploaded.
- A disposable candidate with Python 3.13.7, Node.js 24.17.0, standalone Next output,
  production API dependencies and bundled OFL Noto Sans SC passed package validation. Unsigned
  Inno Setup test installers passed 2.2.0 install, 2.2.1 upgrade, rollback, repair/reinstall,
  path-with-spaces health checks, default local-data retention and explicit local-data deletion
  on this Windows machine.
- Remaining P5-2 limits are a clean/disconnected machine, a real V2.1.1 upgrade, signing and
  antivirus review. P5-3 has no native macOS/Linux package or named test machine. P7 still
  requires an exact release-commit rerun and final release metadata. Cloud sync, teams,
  external login, AWS deployment and GitHub push remain out of scope for this local pass.
- E2E builds use ignored `.next-e2e`; the existing `V2.0/web/next-env.d.ts` modification was
  preserved and remains the only protected uncommitted working-tree file.

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

**Planning checkpoint:** 2026-09-14 (stages 1–4 implementation and current-machine evidence complete; release gates open)

**Release intent:** V2.2.0 is a local-first Windows product. Users install bundled runtimes,
learn with offline examples, import/clean/analyze/export data, and keep projects on their computer.

### Current scope and acceptance

**Full current-host audit (2026-10-07):** the updated dependency environment passes all
271 API tests with PostgreSQL/MinIO and no skips, frontend verify with 60 unit tests,
65 mock/browser workflows, the 39-fixture matrix on Python 3.12 and bundled 3.13,
100 chaos fixtures, and three enforced performance repeats. Eight final packaged live-API
workflows and two additional scientific-control/error-recovery cases have final passing
evidence. Navigation prefetch errors were repaired and production dependencies patched.
The raw failed runs, test-harness corrections, modern-standby interruptions, Windows path
limitations, and scoped security exceptions remain recorded in the
[latest P7 checkpoint](docs/V2.2_P7_TEST_REPORT.md). Final candidate Web build is
`Xz35RBu6doArE3EZJmbU6`; user EXE SHA-256 is
`D14EF97B2505668284FE519ACCA7792016052D7F3FAF0C135E4CD0CEA4845DEE`.
L5/P7 remain open for exact-EXE clean/offline/reboot acceptance, real Windows-account
isolation if claimed, and final release-source/metadata freeze. Existing school/owner reports
do not close these gates for the new hash. No release promotion or cloud deployment is implied.

- [ ] **Windows path compatibility follow-up:** deep custom installation/test-storage paths
  can exceed Win32 file-operation limits (observed SciPy bytecode installation rollback and
  local-object hard-link errors). Verify a bounded supported install/data-root policy or add
  early path-length guidance before claiming arbitrary custom-directory support. Keep the
  standard per-user path as the documented installation route.
- [ ] **Development dependency review follow-up:** the full npm audit retains seven high
  findings in tooling paths; production audit is separately clean after the listed patches.
  Review compatible tooling updates and rerun their lint/build/test gates; do not describe
  the production result as a clean scan of every dependency.

| Workstream | Status | Remaining acceptance |
| --- | --- | --- |
| P0 data and performance | Implemented; reference-machine evidence recorded | Preserve 7 + 9 + 23 matrix and enforced budgets; rerun affected gates at release. |
| P1 examples | Implemented; live desktop-engine evidence recorded | Offline installed-product verification. |
| P2 help and feedback | Help, local feedback and bilingual Markdown About implemented | Agent-run route/content/feedback regressions pass; no paid email service is used. |
| P3 appearance and mobile 3D | Implemented; automated coverage recorded | Installed-product visual/keyboard review; name real-device coverage separately. |
| P4 statistics | Implemented; answer tests recorded | Verify installed previews/exports and independent answers. |
| Local UX and history | V22-L1 through V22-L4 implementation and agent-run verification complete | V22-L5 remains open for exact-candidate clean/offline acceptance and Windows-account isolation; user-reported manual checks are recorded separately. |
| P5 Windows | Self-contained Python 3.13/Node 24 candidate staged and unsigned Inno Setup installer compiled; package validation, Windows PowerShell 5.1 launcher smoke, and same-host native install/Web/API/repair/default-retention/explicit-deletion tests pass | Exact-installer clean/offline owner acceptance and second-Windows-account isolation (only if multi-user separation is claimed) remain open. V2.1.1 upgrade/rollback is waived for personal use; signing/antivirus is distribution-only. |
| P5 macOS/Linux | Deferred to V2.2.x or later | Native-machine build and lifecycle evidence before announcing packages. |
| P6 cloud and online identity | Long-term, feedback-gated | Begin only after real users establish recurring cross-device or external-sharing demand and a cost/privacy model is approved. |
| P7 release | Open; V2.2.0-dev remains unreleased | Freeze one source/artifact candidate, close remaining L5 acceptance, and rerun final metadata/tests on that exact candidate. |

### Execution stages

1. **Baseline and status:** preserve working-tree changes, reconcile this file and test evidence.
   Python 3.12.14 is the current validated development/candidate runtime. Python 3.13 remains the
   final packaging target from the accepted plan; a 3.12 candidate does not satisfy that target.
   Keep the staging script's explicit Python 3.13 assertion for final packaging.
2. **Windows experience:** bilingual startup/maintenance guidance, per-user shortcuts, free-port
   selection, single-instance ownership, health timeouts, logs, graceful shutdown and child cleanup.
   Validate with Windows PowerShell 5.1 as used by the installed shortcut, not just PowerShell 7.
3. **Migration and recovery:** import a stopped V2.1.1 SQLite/object directory into an empty
   installed data directory; never merge or change source data. Snapshot before version activation;
   test health failures, interrupted upgrades, compatible program/data rollback and repair.
   Imported guest ownership/browser sessions need explicit end-to-end evidence before closure.
4. **Clean Windows:** standard user, no developer tools, disconnected first launch, Chinese/spaced
   paths, restart, upgrades, repair and both uninstall choices; signing/antivirus evidence separately.
5. **Product review:** theme/logo, all principal states, CJK exports, seven examples, statistical
   parity, privacy, keyboard/accessibility and regular/invalid/large mobile 3D.
6. **P7 candidate gate:** complete API/integration, synthetic matrices, static/frontend, live/mock
   browser and performance tests plus packaging lifecycle. Record actual counts and all skips.
7. **Delivery:** bilingual README, CHANGELOG, version baseline, this TODO and test report; align
   final versions and artifact hashes, then local commit. GitHub publication needs explicit request.

**Current implementation checkpoint (2026-09-14):** stages 1–4 implementation and current-machine
evidence are complete; release gates remain open. Stage 2 launcher implementation passes the host-level
experience checks: occupied preferred port,
duplicate start, graceful stop, forced-parent termination with child cleanup, and restart after an
abandoned mutex. These checks use real API/Web processes and Windows PowerShell 5.1 with an
isolated Chinese/spaced data path; they are not clean-machine installation evidence.

- Python 3.12.14: packaging contracts and synthetic snapshot/import tests **9 passed**. Ruff checks pass.
- All packaging PowerShell scripts parse under Windows PowerShell 5.1. Inno Setup 6.7.3 compiles
  the bilingual installer source, and the latest unsigned 2.2.0/2.2.1 test EXEs pass the current
  host lifecycle harness. Maintenance dialogs have not had manual visual acceptance.
- `outputs/v22-stage12-candidate-20260914` was staged and passed package structure validation.
  It is a Python 3.12.14 development candidate, not the required final Python 3.13 release.
- A cold API import during investigation took about 208 seconds and exceeded the 90-second
  startup health deadline; warm host checks passed. Cold/clean-machine startup remains open.
- Stage 3 local migration/recovery evidence passes: the real installed launcher completed upgrade,
  data rollback, injected startup failure recovery, interrupted-upgrade recovery, and a synthetic
  V2.1.1-style import from a Chinese/spaced `.labviz` path. The import retained a guest project and
  object bytes, left source hashes unchanged, reset the old browser session, and reopened through
  the real API. A real historical V2.1.1 installation and browser-cookie/guest-token journey are
  still outside this fixture and remain a release limitation.
- Stage 4 current-machine installer evidence passes with bundled Python 3.13.7 and Node.js:
  install, loopback health, version coexistence, upgrade, rollback, repair/reinstall, a non-elevated
  user token, Chinese/spaced install and data paths, default data retention, and explicit deletion
  of data plus retained/failed transaction copies. The source-only Inno compiler check and the
  test-only deletion build are reproducible; signed/antivirus, disconnected clean-machine and
  independent standard-user installation evidence are still unavailable.
- The 3.13 candidate and unsigned test installers are disposable artifacts under `outputs` and are
  not release files or GitHub uploads. The current developer candidate remains Python 3.12.14, while
  the final installer target is Python 3.13.x.

Stages 3/4 were committed as `698daa7`; stage 5/6 test updates and evidence were committed as
`215a9f4`. The protected next-env.d.ts edit is preserved. The latest consolidated checkpoint
supersedes the older d5ce311 evidence; an exact final-release rerun remains open.

**Stage 7 delivery checkpoint (2026-09-14):** bilingual README, CHANGELOG, version baseline and
P7 report are reconciled for the unreleased candidate. Installer SHA-256 identities are recorded
in the P7 report. API `2.2.0.dev0` and Web `2.2.0-dev` retain equivalent development versions.
Installer labels 2.2.0/2.2.1 exercise lifecycle transitions and are not public release versions.
Final promotion to 2.2.0, a release tag and final artifact build remain pending P5-2/P7 evidence.
Delivery validation: four local-privacy boundary tests pass on Python 3.12; `git diff --check`
passes, and all three listed installer SHA-256 values were read from the local artifacts.
No clean-machine, disconnected, signed or real historical V2.1.1 pass is claimed.

Cloud synchronization, teams, Google/Microsoft login and AWS are excluded from these stages.
Post-V2.2 local follow-ups remain: native macOS/Linux packages, custom formulas, advanced mobile
cleaning/multi-panel editing, no-store mode, ESLint 10 and Figma cleanup. Network-neutral local
storage-state names are now part of V22-L2 rather than a deferred follow-up.

### Revised V2.2 local product plan (2026-09-22; decision, not implementation evidence)

The user's accepted default flow is **open the locally installed application -> import data ->
make a chart -> export**. No account, verification email, cloud service, or remote server is
required. The bundled Web and FastAPI processes still run on this computer. Real uploaded projects
should be saved automatically in the current Windows user's local SQLite/object directory; the
user should not have to press a separate Save button to make local history durable. Offline sample
projects may stay temporary unless explicitly retained, so exploring examples does not fill history.

Email-code login, account controls, online sharing, cloud storage controls, and Google/Microsoft
identity move to a long-term feedback-gated track. Keep the About-page `mailto:` contact as a
user-initiated email-client link; it does not call an email API or attach project data. Do not
remove the existing auth/share backend merely to hide local-product UI: preserve compatibility
while changing local storage and access contracts. The cloud track starts only after real user
feedback demonstrates a recurring need for cross-device synchronization or external sharing and
its cost/privacy model is approved.

The Windows installer already locates data beneath `%LOCALAPPDATA%\LabViz\data` for each Windows
profile. File ownership and ACLs must be checked. Loopback binding limits network reach but does
not identify the Windows user. Before removing browser guest-token checks, introduce an internal
per-user launch credential or equivalent local IPC boundary and enforce it on data/history/export
endpoints. This is invisible to the user and must survive changing browsers within the same
Windows profile. Verify that another logged-in Windows account cannot read the first account's
Web/API or local data, including during concurrent sessions. The development checkout's `.labviz`
path is separate from the installed per-user data path and needs a migration/backup explanation.

- [x] **V22-L1: Inventory and repair the local UI controls.** Record the control group, expected
  action, local dependency, and empty/loading/error behavior in the acceptance table below.
  Remove duplicate/inert local actions and keep cloud-only controls out of the local profile.
- [x] **V22-L2: Make local history durable without login.** Real imports and chart snapshots are
  automatically stored in the local SQLite/object store and reopened through a per-launch local
  credential, not an email account or browser guest cookie. Same-Windows-profile browsers can
  access history after launcher bootstrap. Versioned `local-storage-v1` mapping, backup/restore,
  restart and privacy boundaries are tested. Existing email-account projects are not silently
  reassigned; users must re-import their original files to create local-profile records.
- [x] **V22-L3: Make history useful for data and figures.** The local gallery lists bounded PNG
  thumbnails and saved PNG/SVG/PDF snapshots; users can preview, download and explicitly delete
  figures, review processed data, and download cleaned CSV. Missing-thumbnail/PDF-preview,
  retention, storage counts, and cascade/delete behavior have defined states and tests.
- [x] **V22-L4: Ship a readable bilingual About page.** Editable English/Chinese Markdown explains
  verified advantages, audience, privacy and limitations; it embeds synthetic 2D/3D figures
  generated by the real LabViz renderer and links to try examples. Feedback uses a direct
  `mailto:liyutao982@gmail.com`; no paid email or cloud account is involved.
- [x] **V22-L5: Close the supported single-user acceptance scope by owner decision.**
  Owner reports clean/offline/restart acceptance complete for the preceding candidate. This
  is not independent final-hash attestation. Two-account isolation remains untested and excluded
  from release guarantees; V2.1.1 installer upgrade/rollback is owner-waived. Final source/hash
  and fresh automated checks are published as release assets.
- [x] **V22-L6: Reconcile release documentation and metadata.** English/Chinese README,
  CHANGELOG, VERSION_BASELINE, TODO, release notes and P7 state V2.2.0's single-user scope,
  unsigned distribution and distinct historical/current evidence. Release assets identify the
  exact commit and EXE; future work remains in this backlog.
- [ ] **Post-release: verify two ordinary Windows accounts before claiming isolation.**
- [ ] **Post-release: certificate signing and broader antivirus/reputation validation.**

**Current L1-L6 evidence checkpoint (2026-09-23; working tree, not a release sign-off).**

L1-L4 implementation and agent-run acceptance are complete; L5 remains open only for
environment-specific/current-artifact acceptance; L6 documentation is reconciled. Earlier
dated checkpoints below are history, not current status. Detailed run provenance and prior
attempts are in the P7 evidence report.

| Local UI control group | Expected behavior, dependency and failure state | Current automated coverage |
| --- | --- | --- |
| Header/sidebar, Home, New Analysis, History, Settings, Help, About | Local routes; New Analysis focuses the chooser; route errors remain local | `about-local-navigation`, `core-flow`, `help-feedback`; default Playwright matrix. |
| Language, theme and figure defaults | Browser-local preferences; no account/API requirement; safe fallback if storage is unavailable | `appearance`, settings hydration test, Vitest and accessibility checks. |
| File chooser, drag/drop, metadata and examples | FastAPI import/sample APIs; invalid files show actionable errors | `core-flow`, `experiment-metadata`, `live-api`; seven examples through Chromium/Firefox/WebKit. |
| Quality, reversible cleaning, chart/statistics controls and exports | Local API; unsupported/pending options have explained disabled states; failures retain recovery actions | `beginner-guidance`, `analysis-guidance`, `statistics`, `surface3d`, `core-flow`, live sample exports. |
| History search/filter/duplicate/delete and figure gallery | SQLite list/reopen; processed-data preview, cleaned CSV, saved PNG/SVG/PDF preview/download and confirmed deletion | `history-share`, `live-local-history`, API persistence/gallery/privacy tests. |
| Help tips, feedback and About examples | Offline content and local feedback; `mailto:` opens the user's mail client only | `help-feedback`, `about-local-navigation`; no email API/account. |
| Cloud/save/share controls | Not part of local profile; hidden after local-profile detection and retained only for legacy account mode | `live-local-history` verifies sign-in/share are absent from local history/export pages. |

**Current reproducible gates:**

- Python 3.12.14 full API/PostgreSQL 17/MinIO suite: **270 passed, 0 failed, 0 skipped** in
  358.71 seconds. Ruff, format check (72 files), and strict MyPy (43 source files) pass.
- Web ESLint, TypeScript and **57/57 Vitest** pass. Isolated production Webpack build succeeds;
  default Playwright passes **62** with **6 intentional opt-in live-service skips** across desktop
  Chromium, emulated mobile Chromium, Firefox and WebKit.
- Real-API history/gallery workflow: **1/1**. Seven real-API example workflows pass in **3/3
  engines** with PNG/SVG/PDF and cleaned-CSV downloads. Two live upload tests pass for a generated
  large CSV and a multi-sheet workbook. The optimized staged Windows candidate separately passes
  **2/2** Chromium seven-example/export and local-history workflows through bundled Python 3.13.
- The 7 public + 9 edge + 23 historical matrix passes. A separate **100/100** chaos matrix covers
  80 surface profiles (including invalid grids), 20 2D cases and CSV/TSV/TXT/JSON/multi-sheet XLSX.
  Test fixtures and reports stay ignored under `outputs/`.
  - **Bundled-runtime rerun (2026-09-24):** the optimized Windows candidate's Python 3.13.7 and
    packaged API processing code passed the same **100/100** existing synthetic files with zero
    failures. The copied inputs are byte-identical to `outputs/v22-chaos-100-20260922/data`;
    machine-readable results are in ignored
    `outputs/v22-chaos-100-bundled-recheck-20260924/test-report.json`. Compared with the
    separately generated 2026-09-23 matrix, statuses, surface/error classifications and export
    format outcomes agree; its four XLSX containers have different binary hashes despite the same
    tested semantics. This exercises packaged processing functions, not 100 browser uploads or a
    clean/offline installed EXE. No new test files are required for this gate unless a new format,
    defect, or changed data contract warrants targeted fixtures.
- Three-run Windows performance budgets pass with zero violations: 24/1,000/10,000-row tables,
  21×21/101×101 surfaces, and PNG/SVG/PDF; process peak memory **208.49 MiB**. The 10,000-row
  total is 1.79 s max; the 101×101 all-stage total is 5.67 s max.

**L5 remaining owner/environment evidence (2026-09-23–24):** the current optimized portable candidate is
`outputs/l1l6-final-native-candidate-20260924` (Web build ID `TY91QRcKwYgUEdU8r3q5L`;
Python 3.13.7; Node 24.17.0). Its package validator and Windows PowerShell 5.1 launcher
smoke pass. The corresponding unsigned native test installer is
`outputs/l1l6-final-native-installer-20260924/LabViz-Setup-2.2.0.exe` (SHA-256
`CDF7236F31E950B14A3CBDDA75F84B712C1D8AF6543160FF9762A16AB0E0BC6E`). It passes
current-host install, Web/API health, same-version repair, and default data-retaining uninstall.
A test-only deletion variant from the same candidate (SHA-256
`0469EC6219F438AF04C77D2D70A6AA59A7E9E9463959F1485E36AB64212A093E`) passes
explicit deletion of synthetic data, logs, backups, and retained/failed data copies. Packaged real-API
browser rerun for this optimized candidate is recorded separately in the P7 report. The user's
earlier clean-computer offline operation and keyboard/Narrator checks are not hash-linked to this
installer. Test this exact EXE on the clean/offline computer; test a second Windows account before
claiming multi-user separation. Authentic V2.1.1 upgrade/rollback is waived for personal local
use. Signing/antivirus is optional until public distribution; native macOS/Linux packages are
later work. Cloud, teams, external login and AWS remain feedback-gated. Keep V2.2 at
`2.2.0-dev`; no release tag or GitHub push is implied.

The native-candidate follow-up also reran Web ESLint, TypeScript and **57/57 Vitest**; focused
packaging contract tests pass **4/4** with Ruff/format. The earlier full API/integration and
default Playwright totals below remain valid for their recorded source checkpoint but have not
been rerun on a frozen final release commit. P7 therefore remains unchecked.

**Superseded implementation snapshot (historical; before L3/L4 completion).**

- L1/L4 partial: `/about` now has bilingual product/audience/privacy/limits/feedback content
  and explicitly illustrative synthetic 2D/3D SVGs; About navigation reaches the page.
  History's New Analysis goes to the focused upload area, the duplicate save/share menu
  action is gone, two disabled Settings controls are removed, and History/Settings retention
  copy now distinguishes durable real imports from temporary examples/exports. A full visible-control
  inventory and actual exported sample figures/Markdown source remain open.
- L2 partial: the Windows source and portable launchers now pass a random local credential
  to the SQLite API; the portable launcher stores only a DPAPI CurrentUser-protected copy for
  reopening another browser. API data routes reject requests without the corresponding
  HttpOnly local session cookie. On successful **real** import, parsed data/metadata and
  decisions use the existing SQLite store with permanent `local` mode; bundled samples stay
  temporary. The local browser hides sign-in/share controls; existing account-owned projects
  are *not* silently exposed or migrated. The source launcher now rejects occupied/equal
  ports before starting, avoiding a new browser credential paired with an old API process.
  Source-checkout and packaged paths differ, so
  migration/backup, simultaneous Windows-account/ACL validation and packaged launcher
  lifecycle tests remain open. The macOS/Linux source launcher still uses the legacy session
  path; extend and test it separately before claiming an account-free cross-platform flow.
  Never equate loopback binding alone with user isolation.
- Tested on the current Windows checkout using Python 3.12.14: focused local/API/privacy
  pytest checks passed (including 8 local-package/profile cases), Ruff and MyPy passed from
  the API directory, `npm run verify` passed (lint, typecheck, 52 Vitest cases, build),
  PowerShell parsing passed, 12 mock Chromium UI flows passed, and one **real API**
  Playwright flow passed for import -> local history -> denied second browser -> launcher-key
  bootstrap -> same history. DPAPI CurrentUser protect/unprotect passed on this machine.
  These tests do not prove a clean packaged machine, cross-Windows-account isolation,
  offline restart, Firefox/WebKit/mobile, full UI control inventory or P7 candidate status.
- Baseline whole API command was stopped after PostgreSQL-dependent tests errored; the
  first `pytest -x` error was `Real PostgreSQL is required for Phase 5B-3 integration tests`.
  Therefore no full-suite pass is claimed from this run. No new package was staged or signed.
- L3 remains open: local thumbnails, saved figure gallery, export retention and deletion
  semantics are not implemented. L6 remains open: CHANGELOG/VERSION_BASELINE/P7 reconciliation
  and one-candidate final verification have not happened. V2.2 stays `2.2.0-dev`.

**Historical L1/L2/L5/L6 follow-up snapshot (superseded by current evidence above).**

| Visible local control group | Intended action and dependency | Error/verification evidence and remaining limit |
| --- | --- | --- |
| Header/sidebar Home, New analysis, History, Settings, Help, About | Local route navigation; New analysis focuses the chooser | About/New analysis Chromium checks pass; full keyboard/route sweep still open. Sign-in/share stay hidden only after local auth detection. |
| Theme and language | Browser-local preference / locale cookie, no API account | Theme Chromium and unit checks pass; both languages across every screen still need a real-API sweep. |
| Home file chooser, drag/drop, experiment metadata, example gallery | Upload and sample API, then project workspace; invalid files show local error | Real upload -> history Playwright passes; all input permutations and drag/drop failure states not yet audited. |
| History search, filters, clear, continue, export, duplicate, delete | List/duplicate/delete API; export opens workspace; filters run locally | Mock history actions pass. A real local API/browser flow now confirms import -> history -> export and confirms sign-in/share controls stay hidden; a second browser is denied until same-user launcher bootstrap. Duplicate-failure UI is covered; deletion/recovery use SQLite and an offline stopped-app snapshot. Full history chart gallery belongs to L3. |
| Settings figure defaults, local-history and retention links | Browser preference storage / local navigation | Unit/appearance tests cover key choices; every selector and error/storage-disabled state still need browser review. |
| Help search, contextual tips and feedback; About example/feedback links | Local content and user-initiated `mailto:` only | About navigation/links pass; full Help control sweep remains open. No paid email API. |
| Workspace import, quality/cleaning, chart controls, exports | Real local API; validation errors must remain actionable | Core mock and selected real-API flows pass, but the exhaustive per-control inventory and failure-state matrix remain open. |

- L2: fixed SQLite connection lifetime (`with sqlite3.Connection` alone did not close the
  handle), which had blocked Windows directory restore. Added a real synthetic local-project
  backup -> delete -> restore regression, chart/cleaning/restart/export checks, and a test that
  copied legacy account records remain private. Failed snapshot lock no longer leaves a partial
  directory. The Windows import UI/README now state that legacy guest/account projects are
  preserved but require re-import of original files for account-free history; no ownership
  is silently changed. Source launcher now uses a per-user mutex and DPAPI-protected launch
  marker, allows a second same-account invocation to reopen the running app, and uses a Job
  object to close its children. A source-checkout launch and second-invocation check passed on
  isolated synthetic data; hard termination left a stale encrypted marker, which the next
  owner safely discards. Packaged and cross-account confirmation remain open.
  - L5: on this worktree, the full API/PostgreSQL/MinIO suite completed **269 passed** with
  Python 3.12.14 and a disposable local Compose project; `npm run verify` passed lint,
  typecheck, **52** Vitest tests and build. The complete default mock Playwright matrix passed
  **59** with **6 intentional live skips**; the 7 + 9 + 23 synthetic matrix passed with the
  historical corpus present, rerun on the current tree with 0 failures and the machine-readable
  report ignored under `V2.0/api/outputs/`. The repeated 24/1,000/10,000-row and 21×21/101×101 surface
  performance budget passed with zero violations (207.86 MiB peak). One real-API Chromium
  local-history flow passed; its focused rerun also passed the export-page check that login/share
  controls stay hidden in local mode. The seven-example real-API flow passed in Chromium, Firefox and
  WebKit after test bootstrap/navigation synchronization fixes; first attempts failed and are
  retained in the P7 report. An earlier
  Chromium mock sweep exposed a duplicate-error menu issue; the corrected full sweep is green.
  These checks do not establish two-Windows-account isolation, an exhaustive real-API control/error
  sweep, or the final release-commit pass.
- L2: added the versioned `local-storage-v1` wire contract. In local-profile responses, legacy
  temporary rows serialize as `temporary-local`, durable local rows as `saved-local`, and a legacy
  `saved-cloud` row maps only when its owner is already `local-profile`. Reads do not rewrite rows
  or ownership; other account owners remain inaccessible. A contract regression checks the
  version and ownership rule. This is an API compatibility mapping, not a bulk database migration.
- L2/L5 still open: actual two-Windows-account filesystem and loopback isolation, packaged
  offline/restart and comprehensive visible-action coverage. A current-tree Python 3.13.7 /
  Node.js 24.17.0 Windows package has now passed package validation and a host-level fresh-install
  lifecycle on a path containing spaces and Chinese characters: the installed launcher returned
  Web/API 200, stopped gracefully, default uninstall retained synthetic local data, and the
  separate delete-data installer removed its synthetic data directory. Both artifacts are
  unsigned test packages; hashes and limits are in the P7 report. This does not prove clean/offline
  operation for this exact artifact, Windows-account isolation, or a full visible-control sweep.
  Source/installed backup paths must not be treated as interchangeable. L6 remains open until all
  required evidence is reconciled on one candidate; `2.2.0-dev` stays unchanged. The user waived the
  authentic V2.1.1 installer upgrade/rollback check for single-user local use only, not
  migration/backup safety or a future public-upgrade claim.

**Historical UI/L2 refresh (superseded by current evidence above; not release sign-off).**

- The existing web build `GWcQQnZhy3lWBrFtQOB9U` passed frontend lint, typecheck and all **52**
  Vitest tests; the three focused local-history/package/privacy pytest modules passed **16** tests.
  The default Playwright matrix passed **60** with **6 intentional live-service skips** across
  Chromium, mobile Chromium, Firefox and WebKit. A focused navigation/theme/help/history subset
  passed **14/14**. The real local-history Playwright flow passed **1/1** against an isolated
  SQLite/object store: real CSV import, saved-history read/export, hidden sign-in/share actions,
  denial in an unbootstrapped browser, and access after same-user launcher-key bootstrap.
- The browser UI run served the existing `.next` output with `next start`; Next prints its
  standalone-output advisory. Treat this as current-build UI regression evidence, not a substitute
  for running the exact packaged standalone server or installer. No source rebuild or `next dev`
  was used; the protected `V2.0/web/next-env.d.ts` was left untouched.
- User reports successful offline use on a clean school computer, keyboard-only import, and Windows
  Narrator. The user also reports mobile testing complete; the supplied screenshot shows Chrome
  DevTools emulation (Asus Zenbook Fold, 853×1280), not proof of a physical phone/GPU run. These
  manual results are owner-reported and are not tied to an OS/browser version or either current
  unsigned installer hash.
- L1 remains open for a complete control-by-control/error-state inventory and real-API sweep;
  keyboard-only import alone does not verify every route/control. L2 remains open for actual
  two-Windows-account ACL/loopback isolation and exact-package offline/restart validation. L5/L6
  remain open for these gates and same-candidate P7 reconciliation; the user-waived authentic
  V2.1.1 upgrade/rollback test is excluded from this personal local-use scope.

**Historical post-fix frontend gate (superseded by current 62/6 run above; not release sign-off).** A settings refresh
reproduced React hydration error #418 because the server and first client render could read
different localStorage preferences. Settings now renders the same defaults on both sides, then
loads saved preferences after hydration. A server-render/hydration regression test was added.
ESLint, TypeScript, and all **53/53 Vitest tests** pass. A layout-preserving temporary copy of the
current source built successfully with `next build --webpack`; its production server passed the
full default Playwright matrix: **61 passed, 6 intentional opt-in live-service skips (67 total,
3.4 minutes)** across Chromium, emulated mobile Chromium, Firefox, and WebKit. This includes the
settings persistence/privacy-navigation regression. The temporary copy kept the repository's
relative API/contract paths and did not write the protected working-tree `web/next-env.d.ts`.
An exploratory `next dev` run was excluded from the release result; only the production-mode run
is counted. The browser run is source-snapshot evidence, not an installer or final-commit pass.
L1/L2/L5/L6 remain open for exhaustive real-API visible-control coverage, two-Windows-account
isolation, exact-installer offline/restart acceptance, and final same-candidate reconciliation.

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
- [x] **V22-P0-3: Establish user-comfort performance and compatibility budgets before feature work.**
  Measure import-to-ready time, first-chart render, chart interaction, export, and peak memory on a
  small example, a medium dataset, a large dataset near the documented local limit, and 21 x 21 and
  101 x 101 surfaces. Record the reference machine and set reviewed regression thresholds rather
  than relying on subjective impressions.
  - **Acceptance:** Chromium, Firefox, and WebKit desktop smoke flows pass; the supported mobile
    viewport and touch flow pass; large-data sampling is disclosed; no supported case crashes,
    freezes without progress, or hides a failure behind an empty chart.
  - **Evidence and limitation (2026-09-14):** the repeated local runner executes each case
    three times on Windows 11 / Python 3.12.14 / pandas 3.0.5 and records median/range and
    process peak memory for PNG/SVG/PDF.
    Conservative maxima are: 24 rows load/preview/quality/analysis/PNG `7.32/3.97/17.46/12.41/364.24`
    ms; 1,000 rows `2.71/22.97/18.72/24.95/316.72` ms; 10,000 rows
    `8.89/27.54/37.92/24.88/326.85` ms; 21×21 surface quality/analysis/PNG
    `61.58/102.54/532.92` ms; and 101×101 `226.35/244.04/804.92` ms. Maximum whole-case times
    are `1136.03/1240.62/1187.58/2145.54/3937.08` ms respectively, and process peak memory is
    `212.38 MiB`. The browser responsiveness run measured sample-import-to-ready `1,943 ms`,
    first-chart `846 ms`, and chart interaction `478 ms`, all below the enforced
    reference-machine limits. The versioned performance report returns `pass` with zero
    violations, and `performance_budgets.json` is `status: enforced`. The repeated browser run
    passed 12/12 across Chromium, Firefox, WebKit, and mobile Chromium; the real seven-example
    flow passed in each desktop engine. These thresholds remain reference-machine limits and
    are not a guarantee for every machine.

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
  - **Evidence (2026-09-14):** `test_v22_samples.py` validates every card-to-fixture mapping,
    bilingual metadata, recommendation/quality answer keys, real processing, export provenance, and
    all four output formats. `example-gallery.spec.ts` opens each of the seven cards by keyboard and
    verifies metadata, route, persisted refresh state, and browser back behavior; the chooser
    renders all seven Chinese titles, and `mobile.spec.ts` opens all seven cards without horizontal
    overflow. The separately enabled `live-samples.spec.ts` now opens all seven cards through the
    real FastAPI path, checks each recommended chart type, renders the editable chart, prepares the
    export, and downloads real PNG/SVG/PDF plus cleaned CSV files successfully. Browser tests mock
    transport only where stated; no sample-only processing shortcut is counted. The live flow
    passed in Chromium, Firefox, and WebKit, while exact release-commit rerun remains a P7 gate.

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
  - **Evidence (2026-09-14):** `appearance.spec.ts` verifies explicit light/dark switching from
    the header, local persistence and system-scheme behavior; the inline SVG LabViz wordmark uses
    contrast-safe theme fills so “Lab” remains readable in both themes. The theme uses a separate
    dark chart palette and export settings remain backend controlled. The browser appearance suite
    passes.
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
  - **Current limitation (2026-09-14):** an Inno Setup 6.7.3 compiler was available on this
    Windows development machine and produced unsigned 2.2.0/2.2.1 test installers from a
    disposable candidate. The source and lifecycle harness are tracked, but no signed/public
    artifact or clean/disconnected Windows machine is available. This item remains open for
    real V2.1.1 upgrade, offline/clean-install, signing/antivirus, and release-packaging evidence.
  - **Candidate and current-machine evidence (2026-09-14):** `stage-candidate.ps1 -RequirePython313`
    produced a Python 3.13.7 candidate from Next standalone output, locked API dependencies,
    Node.js 24.17.0 and bundled Noto Sans SC; `validate-package.ps1 -RequireBundledRuntimes`
    passed. The non-elevated host lifecycle run passed 2.2.0 install, 2.2.1 upgrade, rollback,
    repair/reinstall, health checks from Chinese/spaced program and data paths, and default
    local-data retention. A separate test-only deletion build removed data, logs, backups and
    retained/failed transaction copies. These results are current-machine lifecycle evidence, not
    clean-machine, disconnected, signed, antivirus or real-V2.1.1-upgrade evidence.
  - **Earlier current-tree installer rerun (2026-09-23; superseded candidate):** Python 3.13.7 / Node.js 24.17.0 candidate staging and
    package validation passed. The unsigned installer and a test-only delete-data variant each
    installed under a new path containing Chinese characters and spaces; the installed launcher
    returned Web/API 200 and stopped gracefully. Default uninstall retained a synthetic data file;
    the explicit delete-data branch removed its synthetic data directory. Each installer is
    142.46 MiB; SHA-256 and exact artifact paths are recorded in
    [`docs/V2.2_P7_TEST_REPORT.md`](docs/V2.2_P7_TEST_REPORT.md). User-reported offline use and
    keyboard/Narrator checks were on a prior build, not bound to those installer hashes. A later
    staged candidate, `outputs/l1l6-runtime-proxy-candidate-20260924`, was a validated portable
    directory. The later optimized candidate was compiled into an unsigned installer and passed
    same-host install/health/same-version repair/default-retention uninstall, while a test-only
  deletion variant passes explicit local-data/logs/backups/copies removal; artifact hashes and
  limits are in the latest P7 checkpoint. Clean/offline acceptance of the exact user-facing EXE
  and cross-account testing remain open. The user waived authentic V2.1.1
  upgrade/rollback for personal use; this is not an upgrade guarantee for other users. Signing
  and antivirus review are only required before public distribution.
  - **About-asset packaging fix and same-version replacement (2026-09-26):** the installed
    2.2.0 had the compiled `/about` route but lacked `web/public/about`, causing the Markdown
    request to fail. `stage-candidate.ps1` now copies Next.js `public` assets and requires the
    bilingual Markdown plus both SVG figures; `validate-package.ps1` enforces all four files.
    Candidate `outputs/l1l6-about-fix-candidate-20260926` passed staging/package validation
    (Web build `TY91QRcKwYgUEdU8r3q5L`, Python 3.13.7, Node 24.17.0). A production-identity
    unsigned 2.2.0 installer is at
    `outputs/l1l6-about-fix-installer-20260926/LabViz-Setup-2.2.0.exe`, SHA-256
    `6BEF885126DD5F7898631AC9FC4AF7144C66AA39A2EE423B09405C304A299A2A`.
    The isolated same-version lifecycle test installed the affected old candidate and then this
    candidate's test-identity build into the same `versions\2.2.0` path: no duplicate version
    directory; Chinese About Markdown and 3D SVG returned HTTP 200; synthetic local data survived
    replacement and default uninstall. The test used a dedicated AppId and no shortcuts so it did
    not modify the real installation/registry. The current user's exact production-identity EXE
    has not yet been installed over their current copy; exact-artifact owner acceptance and the
    clean/offline, signing/antivirus and final P7 gates remain open.
- [ ] **V22-P5-3: Reuse the accepted packaging contract for macOS and Linux.** Do not advertise a
  platform package until it passes clean-machine installation, launch, upgrade, export, backup, and
  removal tests on named supported versions. macOS/Linux packaging may follow V2.2 Windows GA as a
  V2.2.x deliverable if signing hardware or test machines are unavailable.
  - **Current limitation and release boundary (2026-09-13):** no macOS or Linux native package is
    implemented or advertised; support remains the developer checkout path only. This item is a
    V2.2.x-or-later follow-up and does not block V2.2.0 Windows GA.

### P6 - long-term server-backed work, gated by user feedback

This track increases hosting, storage, identity, security, abuse-prevention, monitoring, support,
and compliance cost. It is not part of the offline local definition of done and must stay behind an
explicit deployment/profile boundary. Do not start P6 merely because the code has prepared
interfaces: first collect real user feedback showing recurring cross-device synchronization or
external sharing needs, then approve the cost/privacy model and a separate online product scope.

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

**Local checkpoint (2026-09-14, current working tree):** see
[`docs/V2.2_P7_TEST_REPORT.md`](docs/V2.2_P7_TEST_REPORT.md). Frontend verification includes
48 Vitest tests, lint, typecheck, production build, 55 mock browser passes with 5 intentional
opt-in skips, and 12/12 repeated browser performance passes across desktop/mobile projects. The
7 + 9 + 23 data matrix and full 263-test API/PostgreSQL/MinIO gate pass on fresh local test services.
The real seven-example flow passes through FastAPI in Chromium, Firefox, and WebKit with all
PNG/SVG/PDF and cleaned CSV downloads. The enforced performance budgets have zero violations.
An unsigned Inno Setup test installer passes current Windows install/upgrade/rollback/repair/
uninstall-retention/deletion checks from a disposable candidate. This paragraph is historical
candidate evidence; later user acceptance is recorded above and the revised local work remains open.

**Current private local release blockers:** L1-L4 implementation and current-source automated
acceptance are complete. The optimized candidate passes package validation and an unsigned
native installer passes same-host install/health/same-version repair/default-retention uninstall;
a test-only branch passes explicit deletion. L5 still requires
owner acceptance of that exact installer on a clean/offline computer and second-Windows-account
file/loopback isolation if multi-user separation is claimed. P7 also requires a frozen
source/artifact rerun and release metadata alignment. Signing/antivirus is a public-distribution
gate, not a blocker for private unsigned use. Authentic V2.1.1 upgrade/rollback is waived for this
one-person scope.
The user's clean-school-computer offline, keyboard-only import and Narrator results remain valid
user-reported acceptance for the build tested there, but are not hash-linked to today's installer.
The user waived authentic V2.1.1 installer upgrade/rollback for personal use; do not imply an
upgrade guarantee. Signing and antivirus review remain separate gates before claiming a signed or
security-reviewed public installer.
`V22-P5-3`, all P6 items, no-store mode, public AWS acceptance, billing, and broader mobile
editing remain non-blocking follow-up work.

- [ ] All seven or more public examples are bundled offline, have bilingual teaching metadata, and
  complete import-to-export browser workflows with answer-keyed results.
  - **Current evidence:** bundled metadata and API answer keys pass; the live workflow passes all
    seven cards through FastAPI in Chromium, Firefox, and WebKit with editable charts, PNG/SVG/PDF
    exports and cleaned CSV downloads. The exact release-commit rerun remains required before
    this final P7 checklist item can be checked.
- [ ] The existing 23-dataset V2.1 regression matrix and every V2.2 example, edge, format-parity,
  statistics, mobile, theme, privacy, and packaging gate pass with a machine-readable report.
  - **Current evidence:** the 7 + 9 + 23 data matrix passes; enforced repeated performance results
    have zero violations; the latest complete API/PostgreSQL/MinIO gate passes **270 tests**; and the
    staged portable candidate passes package validation and Windows PowerShell 5.1 launcher smoke;
    the corresponding native installer passes same-host install/health/same-version repair/
    default-retention uninstall and a test-only variant passes explicit deletion.
    Optimized candidate Chromium example/history/export flows pass 2/2; its bundled Python/API
    processing code also passes the existing 100-file synthetic matrix. Exact release-commit and
    exact-installer clean/offline acceptance remain open; the installer is unsigned and not public.
- [ ] `ruff`, formatting, MyPy, the complete API/PostgreSQL/MinIO gate, `npm run verify`, desktop and
  mobile Playwright, visual/accessibility checks, and clean-install tests pass at the release commit.
  - **Current evidence and limitation:** current static, frontend, browser, and full API gates pass;
    they must be rerun on the exact release commit, and clean/disconnected install evidence is absent.
- [ ] No serious or critical accessibility issue remains; reviewed performance budgets pass or an
  explicit limitation and fallback is shown before the user starts the expensive operation.
  - **Remaining:** automated accessibility and enforced reference-machine timing/memory budgets pass.
    The user reports keyboard-only import and Narrator checks passed on the clean school computer;
    this is user acceptance for that tested build, not independent assistive-technology certification.
    Physical-phone/touch and device-GPU checks remain open (the supplied mobile screenshot was desktop
    DevTools emulation).
- [ ] Help, contextual tips, feedback privacy, theme behavior, every new scientific method, and all
  supported package/upgrade instructions are complete in English and Simplified Chinese.
  - **Remaining:** application copy and method contracts pass; current Windows install/upgrade/
    uninstall instructions and bundled-font regression pass. Clean-machine, signed-installer and
    cross-platform CJK visual verification remain.
- [ ] Release notes and `VERSION_BASELINE.md` distinguish shipped local features, optional preview
  features, skipped external checks, and deferred work. V2.2 is not marked complete until this list
  is supported by reproducible evidence from the release commit.
  - **Remaining:** README, CHANGELOG and VERSION_BASELINE now describe the current local evidence
    and limitations; final release metadata and notes wait for all blocking gates.

<a id="v23-roadmap"></a>

## V2.3 Roadmap — Local Research Workspace (planned development / 开发规划)

**Planning date:** 2026-09-27. **Status:** recorded development plan; all new work below is
unimplemented and unverified. This section owns the researcher-workflow proposals discussed
for V2.3; linked historical sections retain their existing implementation evidence.

**Product goal:** help a researcher connect samples and measurements to reusable processing,
mixed-chart figures, and traceable research outputs, entirely on their own computer.

**范围决定：V2.3 以本地科研工作台为主线。V2.3.0 先完成有边界的实验处理与出图流程；
V2.3.x 按用户反馈逐项扩展；专业统计、仪器直连和在线协作不要求在 V2.3.0 一次性完成。**

Existing foundations include seven chart types, uncertainty and fitting controls, grouping,
up to four panels sharing one chart type, immutable revision contracts, Experiment/ExperimentRun
metadata, local history and figure snapshots, and PNG/SVG/PDF export. Extend these foundations;
do not count them as newly delivered V2.3 features. Current reference contracts are
[`web/src/domain/chart-spec.ts`](web/src/domain/chart-spec.ts),
[`api/labviz_api/models.py`](api/labviz_api/models.py), and
[`docs/STATISTICS_CONTRACT_V2.2.md`](docs/STATISTICS_CONTRACT_V2.2.md).

### Delivery sequence and scope control

| Delivery | Intended result | Boundary |
| --- | --- | --- |
| Current V2.2 closeout | Accept one frozen Windows local candidate through existing L5/P7 gates | Planning V2.3 neither closes those gates nor changes `2.2.0-dev`; substantive V2.3 implementation must be isolated from the frozen V2.2 candidate. |
| V2.3.0 core | Sample-aware import and QC, bounded transformations, three new chart families, up to four independent 2D panels, reusable local templates, and figure source-data export | Only P0, C1-C8, UI1-UI8 and the acceptance gates below are V2.3.0 commitments. |
| V2.3.x candidates | Additional scientific charts, richer processing, batch automation, project exchange, and publication conveniences | The E register is ordered candidate work, not a promise to ship every item in patch releases. Reassess scope, compatibility and the appropriate version before each release. |
| After V2.3 / separate research tracks | Domain-specific methods, advanced inference, experimental-design assistance, natural-language assistance, and extensibility | The F register requires separate method contracts, demand evidence and validation; it does not block V2.3.0. |
| Optional cloud track | Synchronization, remote collaboration, shared administration and optional remote execution | Governed by existing V22-P6 and Phase 6 dependencies; not required for local V2.3 acceptance. |

Do not attempt the whole register in one release. Sample semantics, computation, plot rendering,
storage migration and workspace interaction each introduce different failure modes. Ship a
complete, limited workflow first. Freeze that slice before implementation; move additional
ideas to E/F rather than silently enlarging V2.3.0 or weakening its scientific/export checks.
These are dependency stages, not calendar estimates or approved staffing commitments.

<a id="v23-local-cloud"></a>

### Local versus cloud ownership

| Capability | Local product | Optional cloud contribution |
| --- | --- | --- |
| Charts, layered plots, mixed figures, six-or-more-panel layouts, journal styles | Local computation and export; larger layouts follow local performance/readability evidence | Reuse the same figure contract; panel count is not an inherent cloud requirement. |
| Samples, repeats, QC, transforms, statistics and source-data reports | Local snapshots, calculation and provenance | Shared ownership and remote execution only when explicitly configured. |
| Templates, batch processing/export, instrument-file parsing and folder monitoring | Local capabilities; use bounded jobs and user-selected paths | Optional shared template distribution or remote jobs, not prerequisites. |
| Project exchange, backup and frozen publication versions | User-directed files and local storage | Optional remote backup/sync; copying a package is not automatic two-way synchronization. |
| Cross-device synchronization, shared comments, concurrent editing and team roles | Local projects remain usable independently | Requires identity, permissions, conflict/deletion semantics, audit events and operational ownership. |
| Natural-language configuration assistance | Later optional local execution if hardware and packaging permit | An external service is a separate opt-in data transfer with visible payload, cost and failure behavior. |
| Hosted large jobs and commercial plans | Existing local features remain available within tested resource limits | Hosting, service operations and shared-resource costs can inform later plans; pricing is not decided here. |

Reuse one versioned scientific calculation/figure contract across profiles. No mandatory online
login, paid email, remote model, public website, network font, or remote chart library may be
introduced into the local workflow. Bundle required assets. A locally served browser interface
does not by itself make the product a cloud service.

Cloud prerequisites and authoritative owners remain `V22-P6-1` (cost/governance), `V22-P6-2`
(sync/conflicts), `V22-P6-3` (online identity), and `V22-P6-4` (teams), together with the existing
Phase 6 deployment gates. Local scope is not evidence that any hosted service is operational.

<a id="v23-core"></a>

### P0 — scope, contracts and reusable verification foundation

- [ ] **V23-P0-1: Freeze the first researcher workflow and its limits.** Use synthetic independent
  samples measured repeatedly across two batches, with an explicit paired subset and known QC
  failures. Define supported file count, aggregate bytes, rows and job concurrency before coding;
  retain the existing 50 MB single-file ceiling unless separately measured and revised.
  - **Acceptance:** every core task has a named input, expected result, failure case and owner;
    the fixture manifest distinguishes independent samples, technical repeats and observations.
    Retain existing 7+9+23 and 100-case synthetic fixtures; add cases only for new contracts.
- [ ] **V23-P0-2: Version the research and figure contracts before migrations.** Specify stable
  sample/observation IDs, physical acquisition versus software processing, source-row lineage,
  measurement states, unit metadata, analysis results and a figure containing independent panels.
  Decide schema versions explicitly; generate/check frontend and backend contracts together.
  - **Acceptance:** legacy projects and snapshots still open. Unknown legacy sample identity stays
    unknown; migration never guesses independent sample counts or reassigns ownership. Rehearse
    backup, migration failure and restore on disposable SQLite data; preserve PostgreSQL adapter
    compatibility without making PostgreSQL a local-user prerequisite.
- [ ] **V23-P0-3: Define computation and rendering budgets.** Record baseline and target timings
  for import, first usable preview, selection, recomputation, save/reopen and four-panel export,
  plus memory on a named reference machine and named datasets. Set numeric budgets in that
  manifest before feature work; do not promise arbitrary million-row or real-time performance.
  - **Acceptance:** previews disclose sampling/aggregation; scientific results use the full selected
    valid dataset. Cache keys include data revision, parameters, methods and deterministic seeds.
    Sampling preserves relevant extrema and structure; it never changes source data or sample n.

### V2.3.0 core development content

- [ ] **V23-C1: Add a sample-aware experiment view.** Extend current experiment/run metadata with
  sample ID, optional parent/aliquot ID, group, time point, batch, measurement type, units and
  user-declared repeat relationships. Offer optional notes and links to local experimental records.
  - **Acceptance:** show independent experimental units, distinct samples and measurement counts
    separately where declared. Six independently treated samples measured three times display
    six experimental units and eighteen observations, not an inferred n of eighteen. Different
    experimental designs can declare different units; ordinary single-file plotting still works
    without completing an experimental inventory.
- [ ] **V23-C2: Support a bounded multi-file preparation workflow.** Begin with same-schema file
  stacking, explicit source/batch columns, user-confirmed field-role mapping, wide/long reshape,
  filtering, and a small documented set of compatible unit conversions. Match paired observations
  only through explicit keys; general arbitrary joins and formula execution are later work.
  - **Acceptance:** preview changes and row counts before applying them. Retain parsed raw
    snapshots and original row references. Flag duplicate keys, missing partners, incompatible
    units and ambiguous repeats; do not silently aggregate or interpolate them. Record operation
    order and preserve existing cleaning undo/redo behavior.
- [ ] **V23-C3: Record measurement meaning and QC decisions.** Distinguish observed zero, missing,
  not measured, failed, below detection/quantification limits, out of range and saturated readings.
  Add optional blank/standard/control roles, limit values and experimental exclusion notes.
  - **Acceptance:** retain the reported value/qualifier; never convert a limit-qualified value to
    zero or mean-fill it as ordinary missing data. Initially flag and withhold unsupported
    calculations with an explicit reason; special censored-data methods are outside the core.
    Show all included/excluded counts, reasons and the selected analysis unit. Reuse validated
    existing summaries only where their assumptions fit; exploratory changes create a recorded
    revision. A saved plan is a local record, not a claim of external preregistration.
- [ ] **V23-C4: Introduce a mixed-chart figure.** Give each panel its own chart type, dataset
  revision, mappings and analysis binding. Start with one-, two- and four-panel 2D layouts,
  A-D labels, independent axes by default, optional compatible shared axes, and figure-wide style.
  - **Acceptance:** compose existing line/scatter/box or correlation panels with supported new
    charts, reopen the figure and export it as one PNG/SVG/PDF. Preserve legacy single-type panel
    assignments and standalone 3D behavior; mixed 3D composition and freeform layouts are later
    work. Shared limits, categories, units and legend meanings agree across preview and export.
- [ ] **V23-C5: Deliver three new chart families end to end.** Add grouped raw-point/strip plots,
  paired-change plots, and violin plots with optional observed points and explicit summaries.
  - **Acceptance:** stable sample IDs determine pairing; missing/duplicate partners are reported.
    Fix or record jitter seeds. Compute violin density once, with a documented bandwidth, scale,
    support and degenerate/small-sample behavior, then share its result between renderers.
    Expose raw observations and n; density is not additional data. Each chart has bilingual
    examples, numeric answer keys, invalid cases, accessible tables, saved history and all exports.
- [ ] **V23-C6: Link figures to the observations they describe.** Selecting a point highlights its
  source row/sample and corresponding observations in other panels. Aggregate marks identify the
  contributing rows; density marks disclose their input subset rather than inventing a source row.
  - **Acceptance:** distinguish highlight, filter and exclusion actions visibly. Selection alone
    changes neither the analysis nor n. Explain sampled previews; maintain correct IDs through
    sorting, filtering, duplicate measurements, rerendering and reopened projects.
- [ ] **V23-C7: Save reusable local workflow and figure templates.** Store the supported preparation
  steps, field roles, sample/group mapping, charts, styles and export defaults. Apply a template
  to a replacement dataset through an explicit mapping/validation preview.
  - **Acceptance:** preserve category-to-color/marker mapping, method parameters and units;
    unresolved fields stop only dependent steps with actionable guidance. Templates contain no
    hidden experiment data or executable scripts. Begin with one replacement dataset/workflow;
    large batch scheduling and folder watching remain E7.
- [ ] **V23-C8: Deliver traceable research outputs.** Export the composed figure, panel-labelled
  source-data CSV files, a manifest of revisions/parameters/counts, and editable bilingual figure
  legend/method notes derived solely from operations actually executed. Save a frozen local output
  revision while allowing further work on a new revision.
  - **Acceptance:** every panel traces back to source observations and method inputs. The export
    dialog previews included data/identifiers and defines raw-versus-derived contents. A ZIP may
    package the CSV/manifest/notes; its declared role is research output, not yet a round-trip
    project format. Later edits never overwrite a frozen result. No generated causal or
    significance claim may be inferred from chart appearance; unsupported methods remain explicit.

<a id="v23-ui"></a>

### UI development requirements — a researcher workspace

Retain a short **Quick plot / 快速出图** path for one-file tasks. Add a **Research workspace /
研究工作台** for repeat experiments and mixed figures. Both views use the same underlying project,
data and figure records; switching views must not duplicate the project or reset its state.

Proposed desktop layout (wireframe requirement, not an implemented screen):

```text
Experiment / figure name       Local save state + version       Undo   Redo   Export
Data | Quality | Analysis | Figures | Outputs
+--------------------+--------------------------------------+----------------------+
| Project contents   | Active table / analysis / figure     | Selected item        |
| Samples & batches  |                                      | Data and mappings    |
| Data revisions     |   [A: grouped points] [B: violin]    | Method and summary   |
| Processing steps   |   [C: paired change ] [D: trend ]    | Appearance           |
| Saved figures      |                                      | Relevant help        |
+--------------------+--------------------------------------+----------------------+
Selected observations / source rows / QC details / processing record (collapsible)
```

- [ ] **V23-UI1: Organize navigation by research task.** Make recent experiments and figures easy
  to resume; use Data, Quality, Analysis, Figures and Outputs within a project. Keep Help/About/
  Settings accessible but secondary. Preserve existing history links and direct entry points.
  - **Acceptance:** a novice can import and export a basic chart without declaring a complex
    experiment. A repeat user can reopen the previous figure and identify its data revision.
    Use one global navigation plus one project navigation, with no competing deep sidebar trees.
- [ ] **V23-UI2: Keep the active work visible.** Use a collapsible project list, a flexible central
  canvas/table and a contextual right inspector. Show settings for the selected panel or mark;
  separate Data, Method and Appearance instead of one long all-purpose settings form.
  - **Acceptance:** at 1440 px the three regions are usable together; at 1280/1024 px collapse
    secondary regions into drawers before squeezing the plot. At 390 px provide a single active
    region with accessible switches and preserve existing supported quick-plot/3D operations;
    full multi-panel touch authoring is deferred. At 200% zoom, controls and messages remain usable.
    Genuine two-dimensional tables/canvases may scroll within their own region.
- [ ] **V23-UI3: Make figure editing direct and predictable.** Add panel selection, duplicate,
  type-compatible add/replace, move, delete, axis-link controls and A-D labels. Support editing
  titles/labels in context; indicate clearly whether a setting applies to a panel or the figure.
  - **Acceptance:** point/label selection also has a list/table route. Dragging is optional;
    clickable move/assign controls and keyboard equivalents complete the same work. Changing a
    chart type previews incompatible mappings instead of silently discarding configuration.
    Undo restores the previous mapping/layout; deletion affects the selected object only.
- [ ] **V23-UI4: Make data and QC legible.** Keep sample IDs/units visible with sticky key columns,
  field-role chips, grouped issue counts, filters, a source/processed toggle and point-to-row
  navigation. Provide mapping previews before file stacking or pairing.
  - **Acceptance:** each message names the affected rows/field, explains the consequence and
    offers a valid next action. Observations, independent-unit counts, missing partners and
    exclusions remain distinguishable without relying on color. Raw data is visibly protected;
    edits create derived versions. Expanded details must remain available to screen readers.
- [ ] **V23-UI5: Show trustworthy save and computation states.** Distinguish Unsaved, Saving,
  Saved locally, Save failed, Computing, Cancelled, Needs recomputation and Frozen output.
  Keep a compact status area and a recoverable job/detail drawer; preserve drafts during errors.
  - **Acceptance:** report Saved only after durable persistence succeeds. Reopen restores the
    last successful state; failed/cancelled jobs cannot replace valid results. Mark stale results
    when inputs change, and export an explicitly selected consistent revision. Debounce expensive
    controls without moving keyboard focus, resetting scroll or presenting stale output as current.
    Surface existing data-location/backup/restore functions with clear scope and failure messages.
- [ ] **V23-UI6: Establish consistent scientific presentation.** Maintain one group color/marker
  mapping across an experiment; add readable legends, units, n and named error definitions.
  Preserve the current light/dark/system theme, CJK export fonts and grayscale review.
  - **Acceptance:** workspace theme and publication-page background are separately controlled.
    A paper-size preview shows clipping/overlap before export. Defaults use restrained decoration,
    readable text and real data; icons have labels/tooltips and relevant disabled-state reasons.
    Interactive preview and output agree on scientific values and styling semantics, without
    claiming pixel-identical rasterization across engines.
- [ ] **V23-UI7: Preserve accessible and localized operation.** Target WCAG 2.2 AA for the changed
  flow, with visible focus, semantic labels, tabular chart alternatives, sufficient contrast,
  reduced-motion support and English/Simplified Chinese parity.
  - **Acceptance:** pointer targets satisfy the 24 CSS-pixel minimum or the standard's permitted
    spacing/equivalent exceptions; aim for 44 px for primary touch actions. Keyboard-only and
    click/tap-only users can map fields and reorder panels. Test long translations, empty/error
    states, zoom and assistive-technology announcements, not just the happy-path screenshot.
- [ ] **V23-UI8: Validate complete researcher tasks.** Evaluate first import-to-figure, finding an
  anomalous sample, applying a saved workflow to a new batch, and exporting figure source data.
  - **Acceptance:** record time, errors, backtracking and assistance against the current workflow
    using a small pilot target of 3-5 researchers/learners. Resolve silent data changes and data-loss
    issues before release. If participants are unavailable, record usability as unverified rather
    than replacing it with an automated pass. Prototype/visual review precedes implementation of
    major navigation changes; freeze approved responsive screenshots after functional validation.

<a id="v23-extensions"></a>

### V2.3.x extension candidates — retained ideas, outside the V2.3.0 gate

Select one complete workflow at a time using pilot evidence and implementation cost. Each item
below requires its own bounded acceptance before moving into an announced release scope.

| ID | Candidate development content | Required boundary or evidence |
| --- | --- | --- |
| V23-E1 | Raincloud, box-plus-points and other compatible layers; richer grouped summaries and raw/fit overlays | Share data/units and named summary definitions; density and interval calculations come from one checked result. |
| V23-E2 | General numeric/XYZ heatmaps and contour/filled-contour views alongside existing correlation heatmaps and 3D surfaces | Keep measured values distinct from correlations; specify grids, interpolation, missing cells, color limits and preview/export parity. |
| V23-E3 | ECDF, QQ, residual and distribution comparison views; purpose-based chart recommendations | Explain recommended roles and assumptions; do not automatically choose an inferential test or interpret a QQ view as proof of normality. |
| V23-E4 | Hexbin/2D density, 3D scatter and other dense-observation views | Verify bin/density definitions, camera behavior, full-data summaries and accessible alternatives; label aggregation. |
| V23-E5 | Stacked spectra/curves, waterfall, horizontal/stacked/percentage bars and area views | Preserve category/order/units and baseline meaning; pie/donut/radar remain lower-priority optional general-purpose charts. |
| V23-E6 | Reusable blank/baseline correction, normalization, limited calculated columns, calibration inversion, dilution correction, alignment, peaks and integrals | Introduce methods separately with units, operation order, valid ranges, uncertainty assumptions and independent answers. No unrestricted formula evaluation or silent extrapolation. |
| V23-E7 | Batch template application/export, saved instrument-file parsing profiles and optional local folder watching | Bounded sequential jobs first; preview inferred metadata; wait for complete files, prevent duplicate ingestion, support cancellation/restart and never overwrite frozen outputs. |
| V23-E8 | Cross-assay/batch/experiment comparison, completeness matrices, richer parent/aliquot relationships and drift/control views | Validate join cardinality, compatible units, expected versus missing measurements and planned versus observed groups; comparison alone is not automatic batch correction. |
| V23-E9 | Portable project import/export, richer experiment notes/attachments, revision comparison and reviewer-change records | Define manifest versions, checksums, bounded archive extraction, conflicting IDs, ownership and local-path portability. Inspect contents before import; do not execute embedded code. |
| V23-E10 | Publication report packages, reusable Python/R reproduction scripts and editable methods/legend exports | Generate only supported recorded operations, include dependencies/seeds and verify the script reproduces the declared results. PDF/DOCX reports are separate from existing figure export. |
| V23-E11 | Six-or-more-panel figures, nested/freeform layouts, insets, annotations, math labels, log/custom axes, journal-style presets and mixed 3D placement | Remain local features; require resource/readability budgets, valid scale behavior, compatible axes and complete figure export. Presets are editable styles, not journal acceptance guarantees. |
| V23-E12 | Saved workspace views, command search, richer keyboard shortcuts, tagged experiment/figure search and user-defined default layouts | Preserve text-editing shortcuts, focus and recoverability; derive priorities from repeated user actions rather than adding permanent toolbar clutter. |

### Longer-term specialist development register

| ID | Direction | Entry requirement |
| --- | --- | --- |
| V23-F1 | Forest/estimation plots, Bland-Altman, ROC and survival workflows | Separate input/model contracts and independent answers for each method, including pairing/censoring/validation where relevant; these are not merely new plot skins. |
| V23-F2 | PCA, volcano plots and high-dimensional research views | Define preprocessing, scaling, missing-data and multiplicity policies and interpretation limits; validate each domain workflow. |
| V23-F3 | Broader group comparisons, repeated-measures/hierarchical models, custom fitting, measurement-uncertainty propagation and sensitivity comparisons | Require study-design inputs, method-specific assumptions, convergence/error behavior and independent fixtures; retain planned and exploratory analyses with their decisions. |
| V23-F4 | Domain packs: plate layouts/repeat wells and assay QC; chemical spectra/calibration; material stress-strain/cycling; sensor time-series/events | Select an actual target workflow and supported input formats first. Biological normalization, metrology and signal processing each need their own validated methods. |
| V23-F5 | Experimental-design assistance: planned groups/time points, allocation, inclusion/exclusion rules, anonymized group views and sample-size scenarios | Record supplied assumptions and any random seed; masking labels alone is not secure blinding. Separate access to allocation keys where needed; do not infer a defensible sample size without study inputs. |
| V23-F6 | Natural-language field mapping, chart configuration and factual method/legend drafts | Produce inspectable configuration before applying changes. Scientific computation remains deterministic. Local execution is optional; any external service requires explicit data-transfer choice and visible limitations/cost. |
| V23-F7 | Curated importer/chart/method extensions, instrument connections and optional large-job execution | Version extension interfaces and resource/permission boundaries, use reproducible fixtures, and add infrastructure only after measured demand; a plugin must not bypass provenance or export checks. |
| V23-F8 | Teaching mode with synthetic experiments and processing-before/after explanations | State what is synthetic, preserve an unobstructed expert workflow and validate the teaching answers. |

<a id="v23-acceptance"></a>

### V2.3.0 release acceptance and evidence ownership

**Reference workflow:** import synthetic observations from three treatment groups, each with six
independently treated samples measured three times, distributed across two acquisition batches.
Declare the design explicitly; add a separate known paired example. Identify a flagged sample,
apply the supported preparation steps, compose a mixed 2D figure, save/reopen offline, apply its
template to a replacement dataset, and export the figure plus panel source data and method notes.
Neither the treatment effect nor the number of independent units is inferred from file count.

- [ ] **V23-A1: Scope and compatibility.** P0, C1-C8 and UI1-UI8 have implementation evidence;
  old projects, seven existing chart types, local history and exports continue to work. No E/F or
  cloud item is silently advertised as shipped or counted as a core prerequisite.
- [ ] **V23-A2: Scientific correctness.** Independently answer-keyed fixtures verify pairing,
  experimental-unit/observation counts, exclusions, unit conversions, summaries and density output.
  Invalid/insufficient/ambiguous inputs return actionable failures. Compare semantic values across
  preview, figure export and source data, including grouped colors, axes, intervals and sampling.
- [ ] **V23-A3: Persistence and offline operation.** Validate successful/failed autosave,
  close/reopen, interrupted calculation, backup/restore and schema migration. Repeat the reference
  workflow in the packaged local UI with networking disconnected; retain exact artifact hashes.
- [ ] **V23-A4: UI, accessibility and performance.** Complete the changed-screen desktop/compact
  layouts, keyboard and pointer alternatives, CJK/light/dark/print checks, named numeric budgets
  and researcher task review. Separate browser emulation from physical-device evidence.
- [ ] **V23-A5: Regression and documentation.** Run affected feature tests during development,
  then the complete API/PostgreSQL/MinIO and frontend gates for the frozen candidate, including
  real-API flows and the reused synthetic corpus plus new cases. Update bilingual help, limits,
  migration/recovery instructions and release metadata only to match delivered behavior.
- [ ] **V23-A6: Freeze the release.** Bind source commit, package hash, test results and manual
  acceptance to the same candidate. Keep source-code, bundled-code, installed-UI and formal
  release evidence separate; do not reuse historical V2.2 acceptance for a changed V2.3 artifact.

| Evidence | Responsible party | Completion rule |
| --- | --- | --- |
| Contracts, implementation, independent fixtures and automated checks | Maintainer/development work | Save reproducible commands/results tied to the candidate. |
| Researcher task fit and scientific workflow review | Maintainer plus participating researchers | Record actual tasks, findings and fixes; unavailable participation remains an explicit gap. |
| Clean/offline installed use and claimed device/account boundaries | Maintainer plus a person with the required machine/accounts | Validate the exact package; do not fabricate environmental evidence or inherit old waivers as new guarantees. |
| Hosted identity, collaboration, cost and service operations | Future cloud product/operator | Existing P6/Phase 6 acceptance applies; no dependency on this evidence for local V2.3.0. |

### Design and scientific reference material

These references inform requirements; they do not certify implementation or every research design.

- [NC3Rs: experimental units](https://eda.nc3rs.org.uk/experimental-design-unit) and
  [inclusion/exclusion rules](https://nc3rs.org.uk/3rs-resources/driver-recommendations/experimental-groups-and-exclusions).
- [Nature Communications: submission and source data](https://www.nature.com/ncomms/submit/how-to-submit).
- [ECharts custom series](https://echarts.apache.org/handbook/en/how-to/custom-series/) and
  [Matplotlib figure composition](https://matplotlib.org/stable/users/explain/axes/mosaic.html).
- [WCAG 2.2: dragging alternatives](https://www.w3.org/WAI/WCAG22/Understanding/dragging-movements.html)
  and [minimum pointer target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html).

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
> Local sample/repeat/QC development is now planned under [V2.3 core](#v23-core), with broader
> cross-assay/experiment comparison in V23-E8 and specialist models in the V2.3 F register.
> These are plans, not implementation evidence. Membership roles and team permissions remain
> under the separate V22-P6-4 cloud track.

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

The existing dark-mode, accessible-alternative, color-vision and print baseline is tracked by
`V22-P3`. Research-workspace navigation, mixed-figure editing, save states and new accessibility
acceptance are planned in [V23-UI1 through V23-UI8](#v23-ui).

## Advanced Analysis

> V2.1 tracks the first scientifically defensible methods slice under `V21-P1-2` above. These
> individual backlog items remain open until their assumptions, answer-keyed fixtures, and
> export disclosures are accepted.

- [x] Evaluate bootstrap confidence intervals for the first supported residual-bootstrap slice.
- [x] Add residual diagnostics and document multiple-comparison correction as explicitly deferred
  until a comparison design and correction policy are accepted.
User-defined fitting models and broader inference/uncertainty work are now tracked by V23-F3;
chart diagnostic extensions are tracked by V23-E3. Both remain planned, not implemented.

Prediction intervals, simultaneous confidence bands, robust regression, and their answer-keyed
evidence are now tracked only by `V22-P4`.

## Cloud and Commercial Features

Local/cloud capability ownership is defined in [the V2.3 scope decision](#v23-local-cloud).
Six-or-more-panel figures, batch processing/export and journal-style presets are local-capable
work in V23-E7/V23-E11; this supersedes the older idea of assigning them to a paid cloud workflow.
Future commercial packaging is undecided and must not be presented as a technical cloud dependency.

- [ ] Define free, paid researcher, and team cloud plans using observed product usage and operating cost.
- [ ] Decide cloud project, storage, file-size, and sharing-link quotas.
- [ ] Evaluate optional hosted priority/large-job processing only after measured demand and the
  V22-P6 cost/privacy/operations model; retain a bounded local processing route.
- [ ] Add billing only after the free workflow and activation metrics are validated.
- [ ] Evaluate organization administration and institutional controls after the bounded
  `V22-P6-4` team-workspace preview has real usage and security evidence.

## Mobile

V2.3 compact-screen navigation and preservation of current supported operations are tracked in
[V23-UI2 and V23-UI7](#v23-ui). The following remain later work rather than V2.3.0 requirements.

- [ ] Evaluate advanced mobile data cleaning after the desktop workflow is stable.
- [ ] Evaluate mobile multi-panel editing after the V2.2 advanced-3D controls are accepted.
