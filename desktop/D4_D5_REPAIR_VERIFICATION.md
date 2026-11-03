# D4 and D5 repair verification

Date: 2026-10-23 UTC. The isolated desktop checkout is based on `b587db848ba5f1930657d5a316e0160c71864ead` plus its existing work. These checks use fresh synthetic profiles, bundled services and headless Chromium. No native application window, tray, system power setting or installed desktop shortcut was operated.

## Shutdown and suspension

The runner now writes a shutdown receipt bound to its profile and service session. Normal suspension requires an authenticated shutdown acknowledgement, a zero API exit code, completed owned children, a matching normal runner receipt, a zero runner exit code and removal of the running record. Missing or stale receipts, timeouts and abnormal exits report failure and retain the durable checkpoint. They cannot emit `suspended` or authorize data maintenance.

Next has no authenticated shutdown endpoint. After the API has drained and exited normally, the runner terminates the disposable Next listener and waits for it. The receipt records this planned action separately from emergency JobObject termination. Unexpected web exit remains a failure.

Maintenance submissions and the final suspension check share a lock. A late accepted backup cancels ordinary suspension and captures a new maintenance checkpoint. Once suspension commits, new maintenance submissions are rejected until reopen. A scheduler gap longer than five seconds or a backwards clock restarts the quiet interval and cancels an uncommitted checkpoint. The [shutdown contract](contracts/shutdown-v1.json) states the Windows lock, sleep, resume and shutdown policies. Simulated clock checks do not accept actual Windows events.

## Checks

| Check | Fresh evidence |
| --- | --- |
| Desktop tests | 76 passed. Seven new tests exercise the actual controller/runner paths with injected shutdown failures, missing or stale receipts, concurrent maintenance, and clock gaps. Individual cases include several failure variants. |
| Frontend tests | 138 passed in 23 files. Frontend, API calculation code, production website and native host are unchanged by these repairs. The host uses the previously verified locked release binary; it was not launched or rebuilt this round. |
| Small scientific workflow | 15 checks, 17 exports. Independent values, experimental units, paired deltas and frozen ZIP contents passed. |
| Large scientific workflow | Seven checks, seven exports. Independent checks confirm 100,000 observations, 50,000 units, mean 500.5 and four panels. |
| Normal recreation | Five checks retain drafts, source hashes, undo/redo, chart selection, output notes, scientific heads/counts and a frozen export. |
| Large recreation | Four checks retain the editable large draft, scientific table hashes and seven frozen exports. Restore took 3657.17 ms. All project fields were compared except the documented access timestamp, whose exact change is recorded. |
| Maintenance race | Five checks pass with a real HTTP backup request accepted during a held checkpoint. Ordinary suspension cancels; a fresh checkpoint, normal stop, verified backup and service recreation follow. |
| Repeated recreation | Ten consecutive cycles pass without duplicate scientific work or draft loss. Editable restore times are 3918, 3822, 3819, 4353, 3744, 3677, 3842, 3797, 4004 and 3937 ms. Stable suspension has one owned controller process per cycle. |
| Busy and failure guards | Reopening cancels a pending checkpoint; a save failure retains services; duplicate completion cannot release a different download; a real multipart upload remains active beyond the idle deadline and finishes with HTTP 202. Both scientific workflows also finish composed generation while hidden. Request, save, artifact-callback and background-job boundaries have automated coverage. |
| Actual service crash and recovery | Terminating only the identity-verified synthetic API produces a failed session with no false suspension event or owned survivors. Durable checkpoint bytes and scientific tables remain unchanged. A fresh authenticated session restores the editable draft and all 24 frozen small/large exports; a fenced explicit exit then stops normally. |

The failed crash run intentionally exits nonzero. Its diagnostic log and receipt remain separate from the successful recovery. Other retained diagnostics cover the initial test-fixture path/clock setup, an early DOM visibility assumption, access-timestamp comparison and rejection of JavaScript-rounded Windows creation ticks. The ownership probe was corrected to read the raw Python receipt; ownership checks were preserved.

## Resource scope and decision

The report utility records Working Set, Private Commit and CPU separately for the owned controller/services plus the headless test browser. CPU percentages use one core and can exceed 100 for a process tree. Driver phase labels can persist through later checks; their peaks are not steady idle measurements. Native host and WebView2 are excluded.

During the ten consecutive cycles, stable private commit ranges from 21.85 to 22.17 MiB and finishes at 21.89 MiB. The first-to-last median change is about 0.04 MiB, with intervening decreases. This observation does not prove the absence of every long-term leak. A separate fresh-profile launch reaches an enabled home upload control in 4835.03 ms. The earlier H candidate's 22014 ms download-guard outlier remains retained in its original audit; these new samples do not erase it.

The unfrozen preview contains 562,465,962 logical bytes: Node 94,085,536, Python/libraries 415,227,464, and application/host files 53,152,962. Node is included in both disk and active process accounting. The D3/D5 preview's earlier 562,382,150-byte total is historical context; it has a different source set. This directory measurement does not represent a new installer, complete installed program or missing-WebView2 download.

Keep Next conditionally. The native host limit of 32 MiB, complete suspended tree limit of 100 MiB and small editable restore target of 5000 ms remain unchanged. Final native measurements and mandatory reports determine acceptance or a static-service migration. Static migration is a conditional branch and has not been triggered by these background checks.

Exact payload, host/runtime hashes, reports, resource samples and cleanup receipts are indexed in `outputs/desktop-repair-d4-d5-20261023/verification.json`. The application payload remains `8914606e3a9ac1678a04a076e9b528777db87c4a23d3fdf3377aaf312e35c1aa`; separate runtime bindings identify these repairs. The frozen H installer and D2 shortcut remain unchanged. The original V2.3 TODO is the status authority.

## Remaining acceptance

Actual tray interaction, WebView2 destruction, native download/dialog failures, full native resource measurements and Windows session/power events still need direct acceptance. Final installer/component-download measurements, build provenance, licenses and external-environment acceptance remain D6/D7 work. This repair does not accept all of D4/D5 or publish a release.
