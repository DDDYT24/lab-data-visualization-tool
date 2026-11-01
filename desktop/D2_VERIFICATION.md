# Desktop host verification

The Windows GUI host and desktop adapter are implemented on the isolated desktop branch. The reference application remains V2.3 commit `b587db848ba5f1930657d5a316e0160c71864ead`; its API and web source files are unchanged. The formal host is a 3,533,824-byte Windows GUI executable. It uses Tauri and the existing bundled Python and Next services.

## Verified behavior

The final release build passed the 15-check scientific workflow in its actual WebView2 window, with 17 frozen artifacts and independent checks of values, file hashes and ZIP contents. Two-panel generation completed while the window was hidden. A 100,000-observation, four-panel workload had already passed in each D1 candidate using the same application payload.

The native window Close button hid the main window. Launching the executable again exited the duplicate process, restored the original host, retained its unsaved figure title and left controller/API/web process identities unchanged. The user interface runs without a terminal window. Opening an internal `target=_blank` link navigated the existing desktop window without creating another WebView.

Profile-backed preferences survived a normal exit and restart with a different random web port. The profile ID remained the same, and Chinese, dark appearance and overview mode were restored. Six HTTP-service tests cover persistence, wrong origin, wrong token, oversized or unknown fields, profile identity, and retention of corrupt originals. Only the two documented preference keys and language are stored; this is not an unsaved scientific-draft checkpoint.

Real bundled-controller tests passed normal start/stop, a one-second startup deadline, unrelated-directory refusal and unexpected controller termination. Every case left no owned descendant processes. The controller uses an outer JobObject so partial startup and controller loss cannot leave its service tree running.

A GUI deadline test displayed Retry, Open logs and Exit after cleanup. Clicking Retry in the native startup window started services successfully: two controller starts, one main window, and a passing hide/restore smoke workflow. Deliberately stopping the exact owned API process changed the main title to **Connection lost**. The editing WebView and its unsaved figure title remained intact. Explicit acceptance shutdowns returned exit code 0 with no surviving owned processes.

The formal host's observed private commit was 6.14 MiB, below its 32 MiB host budget. This excludes WebView2 and services; the complete D1 idle baseline remains about 2 GiB of private commit. Automatic suspension is disabled.

The final executable's native PDF Save dialog produced bytes identical to the frozen export, and cancelling the dialog was exercised. D1 separately exercised all six native formats and cancellation in both candidate hosts. The application's same-origin API, image and download URLs are preserved. External HTTP, HTTPS and mail links are delegated to the system; other schemes are denied. The desktop adapter exposes metadata and a bounded preference service, with no application IPC commands for arbitrary pages.

## Build and evidence

The executable SHA256 is `6eeb126f324981a3b516d965df2d6e1309ddafc444ddd87f0409fa6c17b49633`. Its runtime manifest also binds the three controller modules. Staging independently verified 12,041 common-payload hashes. Cargo's locked release build, Rust formatting and all six preference tests passed.

Private receipts are in ignored `outputs/desktop-d02` and the synthetic `LabVizD2` acceptance directories. They include the build manifest, full workflow, content verification, single-instance checks, preference restoration, service-loss checks, controller lifecycle cases, retry checks and exit identities. D1 source snapshots confirm that the stable checkout, original development checkout and frozen RC retained their original source hashes and indexes before the append-only status update.

Failed diagnostics remain available. The bundled interpreter initially excluded sibling controller modules; the controller now adds only its own verified runtime directory. A blocked inherited stdin handle was removed from child services by giving the launcher a null input handle. Two repeat-workflow assumptions were corrected: wait for restored preferences to hydrate before switching modes, and return to the home route before expecting an import control. The final build then passed the full workflow.

## Remaining stage boundaries

D2 provides the GUI entry point, supervision, tray callbacks, duplicate-launch restore, native dialogs and profile-backed preferences. Physical notification-area icon interaction and the explicit-exit confirmation need user-facing manual acceptance alongside D7 checks; the shared show/exit handlers were exercised by the acceptance adapter. This preview is not a signed installer or a stable release. Missing-WebView2 installation, clean-machine and physical-offline use, upgrades and maintenance remain D6/D7 work.

D3/D4 still own durable drafts, global activity/draining, automatic tray suspension and recovery after destroying a WebView. Closing the current preview keeps all services running. Explicit exit can discard unsaved edits, so the normal tray Exit action asks for confirmation.
