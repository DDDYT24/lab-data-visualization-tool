# Desktop shell comparison

Tauri is selected for the Windows desktop preview. Both candidates ran the same V2.3 production payload and scientific workflows. Tauri used less memory in the host process and added fewer files. The common Python service, launcher and Next service account for most of the application's resource use.

The baseline is `b587db848ba5f1930657d5a316e0160c71864ead`. The common payload candidate is `ec67b32b403cf67367f05af2adf88134444c8d8036f76fca5af5f8c21902e039`. Neither candidate changes scientific calculations or adopts an existing user profile.

## Measurements

These results describe this Windows computer. Each shell had one fresh-profile start and two process-cold starts using the same profile. Runs alternated between candidates. A fresh profile is not a reboot or a clean-machine installation.

| Measurement | Tauri | pywebview |
| --- | ---: | ---: |
| Fresh-profile home ready | 22.30 s | 21.48 s |
| Warm-profile home ready, median of two | 21.17 s | 22.26 s |
| Host private commit, median | 4.90 MiB | 71.85 MiB |
| Complete home-idle private commit, median | 1969.24 MiB | 2042.27 MiB |
| Complete hidden private commit, median | 1974.96 MiB | 2049.74 MiB |
| Host files on disk | 3,435,008 bytes | 38,289,610 bytes |
| Host comparison ZIP | 1,453,840 bytes | 19,647,573 bytes |

The complete process tree includes the host, its WebView2 processes, controller, PowerShell launcher, Node and Python. Private commit is not physical resident memory. Summed working sets can count shared pages more than once. Samples include startup, home, research, generation, hidden and restored phases. Hiding does not release services in this stage.

In the first Tauri home-idle run, Python accounted for about 1415 MiB of private commit, Node 171 MiB, the launcher 138 MiB and WebView2 226 MiB. Removing shell overhead alone cannot produce a small background application.

The shared payload contains 550,466,639 bytes and its comparison ZIP contains 199,230,439 bytes. ZIPs use the same deflate level 9; they are not installers. The previous RC installer is 132,271,120 bytes, built with a different packaging process, so its size is not a shell comparison. This computer already had WebView2; missing-component download size and installer behavior remain D6/D7 measurements.

## Compatibility evidence

Both actual WebView2 hosts passed 15 workflow checks with 17 retrieved frozen artifacts. Inputs covered TSV, XLSX and a CSV in a Chinese path. The synthetic data has 12 observations, six independent units, a mean of 4.5 and six paired changes of 2. Independent content verification checked these values, all recorded hashes and ZIP contents.

The workflows cover preparation, grouped raw points, paired change, violin density, two-panel composition, template replacement, history, saved-project reopen, both interface modes, themes and Chinese. Actual application-created WebGL contexts remained live, and surface rotation and zoom worked. PNG, SVG, PDF, JSON, CSV and ZIP each passed a native Save dialog and exact-byte comparison in both shells. Native file selection with the Chinese CSV and save cancellation were also exercised.

Eight serial measurement runs completed normally with no surviving owned processes: three small-workload runs per shell and one 100,000-observation, four-panel run per shell. Closing a window preserved an unsaved figure title and allowed queued composition to finish. This proves hide/show behavior; it does not prove draft recovery after destroying a window.

The initial WebGL diagnostic probed contexts itself and was replaced by passive observation of contexts created by the application. Its failed output remains available. One functional Tauri run was interrupted after its content checks; its timings and shutdown are excluded. A separate native Tauri run exited normally with no owned survivors.

Private evidence lives under ignored `outputs/desktop-d01`: `d0`, `benchmark/comparison.json`, `benchmark/sizes.json` and per-run receipts. Synthetic trial directories contain workflow reports, downloaded bytes, native-save hashes and process identities. These files are evidence, not release assets.

## Budget and next-stage implications

The D2 host budget is 32 MiB of private commit. The future D4 suspended-process-tree target is 100 MiB, with a target of five seconds to restore a small saved workspace. These are engineering targets, not achieved whole-application results. A fresh start currently takes about 21–22 seconds and the complete idle process tree commits about 2 GiB.

D2 keeps the proven service topology and adds supervision, a startup window, profile-backed preferences and single-instance behavior. D3/D4 must establish durable drafts and global task draining before automatic suspension can be enabled. Any later service-topology change must preserve the same scientific and file-transfer checks.

## Resource policy after the D0–D7 audit

The three numeric gates above remain unchanged. The measurement policy in [resource-budget-v1.json](contracts/resource-budget-v1.json) now distinguishes those gates from mandatory reports for installation size, cold start, active/peak memory, busy-tray use and large-workspace recovery. Those reports have no implicit numeric pass threshold. They must be complete before the final topology is accepted; a new numeric target must be set before testing a changed candidate.

This clarification applies prospectively from 2026-10-23. It does not accept the existing preview or discard its slow recovery samples. Exact native resource measurements remain open.
