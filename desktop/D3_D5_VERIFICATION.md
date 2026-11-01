# D3–D5 background implementation and verification

Date: 2026-10-20.

Draft persistence, activity draining, pause/resume supervision and service optimizations are implemented on the isolated desktop branch. This round uses private synthetic profiles, background processes and a headless production browser. The native executable is compiled but is not launched during this round. Native tray behavior, WebView2 destruction and the complete native process-tree budget remain acceptance gates.

The application base is `b587db848ba5f1930657d5a316e0160c71864ead` plus the recorded desktop source changes. The private build receipt binds the actual payload, executable and five runtime modules. `outputs/desktop-d34/verification.json` records final results and artifact hashes. It is the detailed evidence index; task status remains in the authoritative V2.3 TODO.

## Drafts and recovery

The versioned checkpoint stores navigation, import and inspection controls, pending source files, preparation mappings, classic chart settings and cleaning undo, scientific chart settings and selection, figure layout undo/redo, figure inspection controls, template mappings and output notes. Completed import references and saved object identifiers are retained. Outstanding jobs remain in the backend's durable queues and must finish before pause can commit.

Source files are staged by content hash. Recovery verifies their bytes and reconstructs actual `File` objects before mounting import hooks. Snapshot JSON is checksummed and written through a temporary file, fsync and atomic replacement. The previous checkpoint and invalid or conflicting material are retained. Storage limits are explicit in the session contract; exceeding them blocks pause.

Checkpointing does not apply preparation, save scientific revisions or freeze outputs. Cached results reload from the database. Preview/replacement capabilities and output-review consent are renewed. The restore checks compare project and research revisions; selected immutable charts and layouts are fetched by their saved identifiers. Missing projects, changed revisions, unsupported versions and damaged checkpoints produce warnings and preserve the original material.

The project baseline endpoint uses the same UUID spelling as the public API. This prevents a hex UUID in SQLite and its hyphenated API representation from being mistaken for different revisions. New service generations also refresh profile preferences when an old port is reused, while navigation within one generation preserves current preferences. Pause requires an acknowledged preference flush as well as the draft acknowledgement.

Ordinary browser server rendering retains its original content. Only the local desktop runtime delays import hooks while restoring staged files.

## Activity and pause protocol

The ASGI gate counts the full request lifetime, including upload receive, the final streaming response and Starlette background tasks. It also checks queued/running imports and figures, owned artifact callbacks and maintenance work. The host uses individual download identifiers so duplicate completion notifications cannot release another download's guard. Uncertain activity is a reason to retain services.

After 45 quiet seconds while hidden, the controller closes admission to new API work, captures and acknowledges preferences and drafts, then rechecks idle status and the same drain generation. It requests page destruction and waits for its acknowledgement before stopping owned services normally. A show request cancels a pending transition. Once destruction has committed, shutdown completes and one fresh authenticated cycle restores the workspace.

The desktop profile mutex remains held while services are suspended. A second controller cannot rotate credentials or adopt that profile during the pause. Service startup retains the installed profile's existing mutex names and data lifecycle. Credentials use current-user DPAPI. API control calls verify executable, PID creation time, loopback listener ownership and profile identity before sending a launch credential. Each service cycle receives a new key.

The background tests recreate the page and restart services; they do not count a hide/show operation as recovery. They verify draft titles, undo/redo, notes, source bytes, selected saved charts, unchanged scientific revision heads/counts and unchanged frozen export bytes. Other tests cover cancellation during checkpointing, injected save failures, a real multipart upload held beyond the idle deadline, concurrent download guards and repeated recovery. Corrupt recovery material is retained until the user explicitly chooses saved state.

## D5 service decision

Keep the current Next service for this candidate, subject to the native budget gate. The measured primary costs were Python's default calculation-library threads, eager library imports and service bootstrap. The desktop now limits BLAS/OpenMP/NumExpr threads, loads calculation libraries and optional cloud SDKs when needed, avoids loading PostgreSQL implementation models for SQLite startup and starts API/Next through a small Python runner. Hash-checked bytecode caches use portable filenames and the bundled interpreter. Numerical functions, statistical contracts and frozen scientific outputs are preserved.

The runner replaces the active PowerShell launcher process while retaining program/profile identity, data recovery, random reserved-port checks, listener ownership, DPAPI and owned JobObjects. The controller stays small during suspension; the runner, API and Next process are stopped. Node remains bundled and its disk and runtime costs must be included in the comparison.

In the controlled calculation-module probe, inherited settings used 1360.43 MiB of private commit and 45 threads. One-thread settings used 137.83 MiB and seven threads. This is a module/process measurement, not a whole-application resident-memory claim. Private commit differs from physical working set.

The final evidence index includes individual service-ready and editable-workspace restore times, active process roles, suspended process samples and payload sizes. The 5-second restore and 100-MiB suspended-tree targets remain unchanged. Headless measurements are reported with their scope; the native host and WebView2 still need direct acceptance before the full desktop budget can be signed off. ZIP or directory size does not replace installer/dependency-download measurement.

On the final candidate, four small-workspace recoveries took 4116, 4956, 4898 and 4044 ms. The suspended controller used 17.28–18.35 MiB of private commit across those checks. These measurements include actual page destruction and owned-service restart; they exclude a native tray host. A small-workflow idle sample measured 653.28 MiB including the headless browser: API 199.65, Next 117.11, runner 18.70, controller 17.38 and browser 300.45 MiB. This is a different browser boundary from D1's native measurements, so the totals are not a direct before/after comparison.

The payload contains 556,931,596 bytes; Node accounts for 94,085,536 bytes, Python and its libraries for 415,227,464 bytes. The full preview directory contains 562,382,150 bytes. The D1 common payload was 550,466,639 bytes; boot caches add disk cost while reducing startup work. Installer size and missing-WebView2 downloads remain unmeasured for this candidate. Retaining Node is conditional on native acceptance, not a claim that all D5 gates have closed.

## Validation scope

Validation includes filesystem failure and corruption cases, profile/credential/port ownership, activity lifetime, deferred-import concurrency, preference generation/flush, browser-rendering compatibility, scientific API regression and the production workflow. The test counts and pass/failure scope are recorded in the evidence index rather than combined into one coverage percentage.

The final checks passed 31 Python desktop tests, four adapter tests and 132 frontend tests. A 158-test desktop API regression preceded the last optional-SDK deferral; the subsequent 96-test core regression and 29-test SDK regression passed. The production small workflow passed 15 checks and retrieved 17 artifacts. Independent verification matched observation values, units, independent counts, paired deltas and violin density grids against the D1 native baseline at a numeric tolerance of 1e-12. Checkpoint/recovery also retained the exact frozen PDF bytes.

The 100,000-observation, 50,000-independent-unit workflow passed seven checks and retained seven artifacts, including a four-panel composition. Its independently checked mean was 500.5. Corruption recovery retained the invalid checkpoint bytes, displayed the warning and required an explicit choice before using saved state. Both owned headless sessions exited with no surviving owned processes.

Switching away from the analysis task originally reset the selected saved chart. The fix retains the explicit selection across task refreshes. The roundtrip test now asserts both the checkpoint's selection and the reconstructed editor's selection. It also covers figure undo/redo, preparation files, output notes and unchanged scientific revision heads/counts.

The first broad API run also attempted PostgreSQL/MinIO gates without those services and source-CLI recovery without the new worktree's test interpreter. Those diagnostics are retained. The local recovery interpreter was prepared and the applicable API regression rerun successfully. Cloud-dependent failures are not claimed as passed.

Earlier candidates exceeded the five-second restore target, including an approximately eleven-second run. The final candidate adds boot caches and optional-SDK deferral; its four recorded restore times are listed above without removing slower historical diagnostics. The first large-workflow run hit the test's default five-second readiness wait. The editor subsequently became ready; the large-workflow readiness wait is now bounded at 120 seconds. This does not change the separate five-second recovery target. Small and large artifact folders are separated to retain both sets of frozen files.

No original stable, development or RC source is adopted by the desktop runtime. Their preservation check and the append-only status update have separate receipts. Native window/tray/dialog checks, long hidden-window timer behavior and the final complete desktop resource measurements remain open under the user's no-desktop-control scope. Native transfers whose completion cannot be matched retain their activity guard; retry/exit behavior needs native acceptance. Installer coordination, desktop-session backup during maintenance and clean-machine/physical-offline acceptance remain later-stage work.
