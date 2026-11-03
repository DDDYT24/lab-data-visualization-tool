# D6 background and installer verification

Date: 2026-10-21 UTC. This receipt covers D6 implementation and checks that can run without controlling the desktop. Tests use private synthetic profiles, background services, headless Chromium and silent installers. The native host was compiled but was not launched in this round. Native file/window acceptance remains open, as do the D4/D5 native resource gates and D7 release acceptance. The original V2.3 TODO remains the status authority.

## Data maintenance

The desktop data panel provides backup, verification and independent restore without terminal commands. Before backup it saves a checkpoint, stops accepting new work, waits for activity to finish, destroys the page and normally stops the owned services. It then restarts the services and restores the editor. Installer preparation follows the same sequence and exits after the backup completes. Failed saves retain the editor and services.

A backup includes the SQLite database, object files, preferences, checkpoint and staged draft sources. Access credentials and WebView cookies/cache are excluded. Verification checks the complete file set, hashes, SQLite integrity, installed data epoch and desktop format. Restore creates an independent profile, preserves scientific identifiers and frozen content, and rebinds the checkpoint and preferences to the new profile ID. A partial copy lacks a completion receipt and cannot be selected. The panel selects a verified copy or the original profile for the next launch using bounded identifiers; actual native launcher switching remains untested.

Installer preparation requests carry an expiry and the requester's process identity. An abandoned request cannot close a later desktop session. Full package hashes are checked in parallel, with the same required file set and mismatch handling.

## Evidence and candidate binding

| Check | Result and scope |
| --- | --- |
| Desktop Python | 52 passed, including authorization, corruption, failed writes, partial restore, incompatible epoch, installer transactions, requester lifetime and Chinese paths beyond 260 characters. |
| Frontend | 132 tests in 22 files passed; production build and lint passed. |
| Production scientific workflow | 15 checks and 17 downloaded artifacts passed, with independent scientific and file-content checks. The scientific API and web payload are byte-identical across the subsequent maintenance candidates. |
| Desktop maintenance | Seven headless production-UI checks passed: inline frozen PDF URL/bytes, live-editor backup/restart, verification, independent restore, copy/original selection, restored-profile service health, and preparation that exits the live controller. Scientific revision heads and object counts remained unchanged. |
| Installer lifecycle | Eight checks passed on candidate G: actual per-user installation and exact files, private GUI shortcut target, same-version repair, compatible upgrade/rollback, failed service activation recovery, future-epoch refusal, uninstall retention and reinstall/reopen. Bundled-service activation ran with only Windows system paths available. |
| Data created after upgrade | An authenticated CSV import created a new project after a compatible upgrade. A compatible rollback retained that new project and the existing projects. |
| Chinese long paths | Bundled Node read input and saved a frozen PDF on 262-character Chinese paths. Authenticated import/reopen returned 12 rows; PDF bytes matched their frozen SHA256. Native dialogs were not involved. |
| Final installer wrapper | Final H installed successfully, then returned failure code 20 for the deliberately damaged candidate. The previous program, scientific data and checkpoint were retained, and its bundled services remained healthy. Final silent uninstall passed and removed the test registration. |
| WebView2 | Actual registry detection found `155.0.4283.62`. Six isolated tests passed for existing/missing runtime, download failure, signature failure, installer failure and failed post-install detection. Component download/installation was mocked; no system component was installed. |
| Native host | Locked release build and format checks passed. PE import inspection found only Windows OS DLL imports for the GUI host; clean-machine compatibility still requires testing. |

Candidate G supplied the complete runtime and lifecycle checks. Final H changes only `LabVizDesktop.iss` to report activation failure correctly, prevent launch after failure and suppress user shortcuts in private acceptance runs. Every runtime, host, web and API file is byte-identical between G and H. Final H passed its own actual positive installation and injected failure; its final uninstall and cleanup are recorded in the evidence index. Earlier scientific evidence is reused only for the unchanged scientific payload. Completed headless runs exited without owned process survivors.

Private evidence is indexed by `outputs/desktop-d06/verification.json`; original reports and failed runs are retained. `final-bundle-audit.json` verifies all 12,569 listed files, absence of extra files, the seven runtime source files and exact installer sources.

| Final H artifact | Value |
| --- | --- |
| Package ID | `08dde4ce016aca8f0c87ad98c9febb58e22edf7c8d5b555ceee9f929c04accef` |
| Scientific payload candidate | `4e5fbf0d963094634a80ca77d5f85156668b08fa6ae071fbcb75378f24f09d95` |
| Source base | `b587db848ba5f1930657d5a316e0160c71864ead`, plus recorded uncommitted desktop changes |
| Host | Tauri `2.12.4`, application `2.3.0-desktop.1` |
| Installer | `outputs/desktop-d06/build-h/LabViz-Desktop-2.3.0-desktop.1-Private-Setup.exe` |
| Installer bytes | `146963724` |
| Installer SHA256 | `5dcd3142019e65d1839e7fabed36fe115c7ca585c8a2c00b11cf74a734c28f7b` |
| Staged bundle bytes | `566901456`, excluding any system component download |

The outer package manifest binds the source base, dirty source state, host version, web build ID, checkpoint schema, data epoch, WebView2 policy, every payload file and installer source. Build receipts also bind the support archive, background evidence and final installer hash. The package includes inventories for 51 Python distributions and 236 Rust packages in the Windows dependency closure, plus notices collected for 157 installed production web packages. The support archive was exercised through its actual script and subsequent silent installs. This is an unsigned private candidate; public release acceptance is false.

## Installation behavior and remaining gates

The installer uses a distinct per-user desktop identity, stable profile paths, a GUI executable entry point and immutable version directories. Update preparation checkpoints and backs up the profile before activation. A candidate must pass bundled-service health checks; failed activation restores the previous program, scientific data and checkpoint. Compatible rollback preserves data created after upgrade. Same-version repair and uninstall retention are covered by the lifecycle checks.

The WebView2 helper follows [Microsoft's detection and deployment flow](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution). It checks the Microsoft signature before executing a downloaded bootstrapper, verifies the installed version afterward and offers retry on failure. The policy floor is `120.0.0.0`; that oldest version has not been physically tested. This machine downloaded zero component bytes because the runtime was present. A missing-runtime installation's complete download size remains unknown and must be recorded separately from the application installer.

An injected failure exposed an installer bug: the helper restored the previous program/data correctly, but Setup returned success. The wrapper now tracks activation failure, uses [GetCustomSetupExitCode](https://jrsoftware.org/ishelp/topic_scriptevents.htm) and guards the post-install launch. The original failing run is retained; final H repeated the same deployed-file fault and returned 20 while retaining the previous program, scientific data and checkpoint.

Remaining D6 native work includes installed GUI shortcut use, next-launch profile switching, open/save/overwrite/cancel behavior, native Chinese long paths, format selection, PDF rendering, external links, clipboard, keyboard and high DPI. The headless PDF check certifies the inline URL and returned bytes only. D4/D5 still require actual tray/WebView destruction and rebuild, long hidden timing, the complete suspended process budget, host budget and native editable recovery measurement. D7 clean Windows, missing-component installation, physical offline use and human acceptance remain separate.

The real Desktop Preview shortcut still targets D2. Final cleanup found no owned process survivors, test installation registration or test Start Menu entry; synthetic profiles and diagnostics are retained. Stable and RC source/index contents remain protected. Only an append to the original development TODO records this round's status.
