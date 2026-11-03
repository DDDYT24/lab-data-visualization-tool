# Windows desktop preview

2026-10-28 第二部分原生阶段暂停：CSV／XLSX／TSV与中文长路径打开、六格式原生保存已有实测。PDF预览及跨项目状态防护已作源码修复，156项前端测试通过；最新暂存为 `outputs/desktop-d06/p2final28/bundle`，尚无匹配新Setup。按用户要求停止，当前从[阶段记录与暂停点](PART2_NATIVE_PROGRESS_20261028.md)继续，第二部分未整体验收。


2026-10-28 第二部分后台工作：下载失败与同源并发保护、缩放快捷键、资料副本选择和剪贴板反馈已修复；新私有候选为 `outputs/desktop-d06/p228/bundle`，有匹配未签名Setup。自动结果及证据见[本轮报告](D6_PART2_BACKGROUND_20261028.md)，后续按[第二部分原生清单](PART2_NATIVE_CHECKLIST.md)验收。第一部分既有十轮不重复；本轮没有操作用户桌面。


2026-10-25 原生验证片段：最新测试 bundle 为 `outputs/desktop-d06/native-wake-final-20261025/bundle`，修复休眠后重复唤醒无响应。按用户收窄范围完成五轮片段，实际保留六轮完整记录；正式十轮及完整生命周期仍未签收，Next 暂时保留。此次宿主修复尚无新 Setup；下文 f832 的旧安装器不包含它。见[原生片段验证](D5_NATIVE_SLICE_VERIFICATION.md)，当前任务状态仍由原开发区 V2.3 TODO 管理。

The latest private delivery is `outputs/desktop-d06/ps-module-fix-20261025/bundle`, with its matching Setup under `build`. It repairs inherited PowerShell module loading and directly invokes the trusted Windows signature cmdlet. All 107 desktop tests, four observer unit tests, seven real-engine signature controls, four headless fault classifications and the exact updated silent installer lifecycle passed. See [remaining-repair verification](D7_REMAINING_VERIFICATION.md), the [preceding delivery](D7_DELIVERY_VERIFICATION.md) and [operator acceptance](D7_OPERATOR_ACCEPTANCE.md). Two upstream notice holds, the original unconfirmed disconnect, native/external acceptance and intended signing/public release remain open. The real desktop shortcut still targets D2.

The 2026-10-24 memory-repair snapshot used `outputs/desktop-d06/memoryfix-20261024/bundle`. It fixes unbounded DPAPI ctypes type retention. All 91 desktop tests, 1,001 embedded-Python decryptions, 20 headless rebuilds and fresh scientific/maintenance/fault checks passed. See [scoped memory verification](D5_MEMORY_REPAIR_VERIFICATION.md). That snapshot preceded Setup compilation. The delivery above supersedes its packaging and notice-review status. The real desktop shortcut still targets D2.

The preceding private installer candidate is `outputs/desktop-d06/closeout-final-20261024/bundle`, with its matching installer under `build`. It adds protected calculation-cache cleanup and closes scoped non-native crash, port-race, maintenance-race and service-coexistence checks. Its explicit UTF-8 packaging fix uses a separate manifest with every execution member byte-identical to the tested precompile package. Ten headless rebuilds, final-package service health and the exact final silent installer lifecycle passed. See [closeout verification](D7_CLOSEOUT_VERIFICATION.md) for hashes, evidence lineage and remaining gates. Five web notice reviews and native/external acceptance remain open; the real desktop shortcut still targets D2.

The D6 source repairs use `outputs/desktop-d06/repair-20261024/bundle`, with their private installer under `build` in the same output directory. Build-time host/web source snapshots, logs and output hashes are required before staging and are bound into the frozen package. The license inventory includes `LICENCE` files, version-pinned upstream Rust notices and explicit web notice review items. See [D6 repair verification](D6_REPAIR_VERIFICATION.md). The real desktop shortcut still uses D2; private tests do not replace it.

The D4/D5 repairs use `outputs/desktop-repair-d4-d5-20261023/bundle`. Service shutdown now produces a session-bound receipt; a timeout or abnormal exit cannot be reported as successful suspension. Maintenance submissions and suspension commit share an admission lock. See [repair verification](D4_D5_REPAIR_VERIFICATION.md) and [shutdown policy](contracts/shutdown-v1.json) for recovery and Windows session/power behavior. `desktop/scripts/report_resources.py` reports background working set, private commit, CPU, component sizes and repeated suspension samples; native resource acceptance remains open.

The D1–D3 source repairs have a separate unfrozen preview under `outputs/desktop-repair-d1-d3-20261023/bundle`. They cover draft navigation, settings recovery, bounded staging reclamation and a task-aware exit protocol. See [repair verification](D1_D3_REPAIR_VERIFICATION.md). The D6 installer below remains frozen; these source repairs do not replace that installer or the D2 desktop shortcut.

The historical D6 H candidate is staged under `outputs/desktop-d06/bundle`, with its private installer under `outputs/desktop-d06/build-h`. It adds in-app backup, verification, independent restore and checkpoint-aware installer preparation. Silent installation, repair, compatible update/rollback, failed activation, uninstall retention and reinstall have been exercised with bundled runtimes and synthetic profiles. See [D6 verification](D6_VERIFICATION.md) for the exact candidate and evidence scope. Native file/window acceptance remains open. The existing desktop shortcut continues to use D2.

The historical D3–D5 preview is staged under `outputs/desktop-d34/bundle`. Its GUI entry point is `LabViz.exe`. The existing desktop shortcut still points to the previously tested D2 build. This round validates the new code through isolated background services and a headless browser; native tray, dialog and WebView2 acceptance remains separate.

Closing the current implementation hides the window. After 45 quiet seconds, it blocks new API work, saves the editable state and source files, verifies the same drain generation, destroys the page and normally stops its API and Next services. Reopening starts a fresh authenticated service cycle and restores the draft. Busy work, active downloads, unknown activity and failed saves keep the editor and services alive. Explicit Exit offers waiting for active work, cancelling the exit, or a separately confirmed interruption. The controller fences admission and checks activity again before approving a waiting exit; edits since the last checkpoint can still be discarded. The revised native dialogs require direct acceptance.

The checkpoint includes preparation controls, pending source files, classic chart settings and cleaning undo, scientific chart controls, figure layout undo/redo, template mappings and output notes. Stored sources are addressed and verified by SHA256. Result caches reload from saved data; preview tokens and output-review consent are renewed. A changed revision or damaged checkpoint produces a recovery warning and preserves the original material.

The desktop keeps Tauri, Python and Next. D5 limits scientific-library threads, loads calculation libraries on demand and starts the two services through a small Python runner. The runner retains the existing profile mutex names, data lifecycle, DPAPI protection and process/port ownership checks. See [D3–D5 verification](D3_D5_VERIFICATION.md) and [session contract](contracts/session-v1.json).

Build current components through `desktop/scripts/build_verified.py`, using a private Cargo cache and the verified bundled Node executable. The wrapper records source snapshots before and after compilation, tool versions, logs and output hashes. It restores the generated `next-env.d.ts` bytes after the web build. A changed source, build tool, output or log invalidates the receipt.

```powershell
python desktop/scripts/build_verified.py host --output outputs/desktop-d06/my-build/provenance --cargo-home $env:CARGO_HOME
python desktop/scripts/build_verified.py web --output outputs/desktop-d06/my-build/provenance --node outputs/desktop-d01/payload/runtime/node/node.exe
python desktop/scripts/stage_d6.py --prepare --output outputs/desktop-d06/my-build/bundle --provenance outputs/desktop-d06/my-build/provenance
```

Set `CARGO_HOME` to the private cache before running the commands. Choose a fresh build directory; receipts and frozen candidates are preserved. Stop every process using an unfrozen preview before restaging it. Freezing and installer compilation use `desktop/packaging/build_candidate.py`; matching background evidence is required for compilation. Desktop tests run with `python -m unittest discover -s desktop/tests -v`. WebView2 branch tests require a fresh explicit output: `powershell -NoProfile -File desktop/tests/test_webview2.ps1 -OutputRoot outputs/desktop-d06/my-component-tests`. They reject existing or escaping outputs and never overwrite historical component receipts. The D1 measurement policy remains [resource-budget-v1.json](contracts/resource-budget-v1.json).

This remains a private preview. D6 includes WebView2 detection and signed component installation when needed; missing-runtime branches have isolated tests. Actual component installation on a clean machine, physical-offline use and native acceptance remain open.

## D2 verification snapshot

The desktop entry point is `LabViz.exe` in the staged preview folder. Keep the folder together and double-click the executable. No terminal is required. Closing the window keeps the workspace and local services running in the tray; reopening the executable brings back that instance. Choose **Exit LabViz** in the tray menu to stop it. Explicit exit asks for confirmation because unsaved edits are not yet recoverable across shutdown.

The preview uses its own `Local AppData/LabVizDesktopPreview` directory. Its WebView data, application preferences, logs, database and object files stay outside replaceable program files. The stable application and V2.3 RC retain their existing profiles. Startup failures offer Retry, Open logs and Exit. A service failure retains the editing window instead of silently discarding its state.

Tauri is selected; see [D1 comparison](D1_COMPARISON.md). The formal host lives in `host`, with the narrow preference bridge and process controller in `runtime`. The unchanged production interface still uses its loopback origin for API calls, images and downloads. Internal new-window links reuse the desktop window. External HTTP, HTTPS and mail links open through the system. Native open/save dialogs preserve the selected file format.

Build the host with `cargo build --release --locked` from `desktop/host`, using a Windows MSVC toolchain. `desktop/scripts/stage_desktop.py --prepare` copies the common verified payload into a new output folder; running it again without `--prepare` verifies the copied files and stages the host and runtime manifest. The script currently reads the shared comparison target directory for `LabViz.exe`, so set `CARGO_TARGET_DIR` to `desktop/prototypes/tauri/target` when building. Python preference tests run with `python -m unittest discover -s desktop/tests -v`. Real controller and GUI acceptance use fresh synthetic profiles under `LabVizD2`; their receipts and binaries remain private and ignored.

This is a private preview folder, not a signed installer. It requires WebView2 on the computer. Installer dependency handling, upgrade/maintenance behavior, clean-machine and physical-offline acceptance belong to D6/D7. Automatic suspension, global task draining and durable unsaved-draft recovery belong to D3/D4 and remain disabled.

## Comparison prototypes

The D0/D1 prototypes load the V2.3 production website and bundled Python service in an independent window. Closing the window hides it to the tray and keeps its state and services alive. Automatic suspension remains disabled until draft persistence and global task draining are implemented.

Both candidates use the same production payload, synthetic inputs, startup controller and measurement boundaries. Tauri and pywebview are compared through the complete application process tree. The comparison includes actual scientific workflows, file transfers, tray behavior, startup and memory.

The behavior contract is in `contracts/lifecycle-v1.json`. Private source snapshots, synthetic files, logs, measurements and binaries belong under the ignored `outputs/desktop-d01` directory. Do not use default user profiles or modify the released RC assets.

The reference source is commit `b587db848ba5f1930657d5a316e0160c71864ead`. Runtime and payload hashes are recorded with each trial. These prototypes are private comparison builds; installation and stable desktop acceptance follow in later stages.
