# Windows desktop operator acceptance

Use the exact Setup and bundle named in `package.json` inside the private acceptance kit. Current project status remains in the V2.3 TODO. These steps require a person, a real application window or an independent Windows environment. Preparing the kit does not pass any gate.

## Start and identify the candidate

1. Copy the private Setup and acceptance kit to a disposable Windows test account. Keep existing personal data and installed V2.2/RC profiles out of the test.
2. Check the Setup SHA-256 before installation. In PowerShell, run `Get-FileHash -Algorithm SHA256 -LiteralPath .\LabViz-Desktop-2.3.0-desktop.1-Private-Setup.exe` and compare the complete value with the kit's `package.json`. Record `Get-AuthenticodeSignature` separately; the current private Setup is unsigned.
3. Record Windows build, architecture, available memory, WebView2 version, display scaling, input method, network state and any other LabViz process. Leave unknown observations blank and keep the gate pending.
4. Install in the disposable account. Record component downloads separately. On a machine without WebView2, the first installation may use the network; ordinary workflows must subsequently work offline.
5. Open the new executable from its verified installation directory. The original development PC's desktop shortcut still targets D2 and must not be used as evidence for this candidate.

For a development machine, the existing `desktop/scripts/run_desktop_acceptance.py` can launch the native binary with a fresh synthetic `--label` and explicit `--bundle`. Its Python environment needs psutil. Running that command opens the actual native application; it has not been run in this delivery. Use the default 45-second idle policy for final resource acceptance. The installed clean-machine checks must also use the installed result.

The kit's `verify.py` checks the bundle, manifest identity and Setup without starting the application. After installation, run it with the Python executable inside the selected installed generation:

```powershell
& '<generation>\app\runtime\python\python.exe' -B -X utf8 .\verify.py --package-info .\package.json --bundle '<generation>' --setup '.\LabViz-Desktop-2.3.0-desktop.1-Private-Setup.exe' --output '.\integrity-observation.json'
```

Replace `<generation>` with the full verified `versions` directory for the expected package. The output file must be fresh. `artifactIntegrityPassed` verifies the recorded bytes; native and release acceptance remain false until their own checks pass.

## Scientific fixtures

The kit includes CSV/TSV/XLSX small fixtures and a 100,000-row CSV. Map measurement to `value`, sample to `sample`, independent unit and pair to `pair`, time to `time`, group to `group`, and unit to `unit`.

- Small: 12 observations, six independent units, overall observation mean 4.5 and six paired differences of 2.
- Large: 100,000 observations, 50,000 independent units, overall observation mean 500.5 and paired difference 2. Use four panels for the composed figure.

Map fields explicitly. Do not interpret row count as independent n. Freeze exports, retain their SHA-256 values and compare the downloaded bytes after every recovery or maintenance step.

## Required observations

| Gate | Action | Pass criterion and record |
| --- | --- | --- |
| Window and tray | Edit a draft, close the window, reopen from the tray and launch the same entry again. Exercise wait, cancel and separately confirmed interrupt on exit. | The active task is retained; one same-profile instance is used; closing the window leaves accepted work running. Record every dialog and result. |
| Native files | Use native open/save dialogs for CSV, TSV and XLSX; cancel dialogs; try a missing source, Chinese names and paths longer than 260 characters. | Supported imports retain the expected counts. Cancellation/errors retain the draft. PNG/SVG/PDF file types and downloaded bytes match the chosen format. |
| Draft reconstruction | Edit chart title, selected saved chart, figure panels, undo/redo, replacement mapping and output notes. Leave a source file unsubmitted, hide until suspension, then reopen. | The same editable values and staged file return. Scientific revision heads/counts and existing export hashes stay unchanged. |
| Native suspension | Observe the actual Tauri/WebView2, controller, API and Next process identities before and after idle suspension. | Durable checkpoint and normal service receipt precede suspension. WebView2 and services are released; reopening creates one replacement path. No unrelated process is stopped. |
| Native resource gates | Measure the complete owned tree through at least ten consecutive cycles. Record Working Set, Private Commit and CPU separately. | Native host Private Commit <=32 MiB; complete suspended tree <=100 MiB; small-workspace intent-to-editable restore <=5,000 ms. Retain every sample and investigate growth or slow samples. Headless results do not pass these gates. |
| Mandatory measurements | Record installer bytes, complete new installation, retained-version overhead, component download, cold home/project start, large restore, foreground/peak and busy-tray resources. | Every field has an observation and scope. Missing observations keep acceptance pending; there are no additional implicit numeric thresholds. |
| Busy tray and interruption | Start the large composed render, hide the window, reopen during work, then exercise exit choices and a controlled owned-service failure. | Completed artifacts retain correct values and unique publications. Waiting/cancelling exit retains work. Confirmed interruption and recovery retain durable data; do not promise unfinished work survives process termination. |
| Data tools and PDF | Back up, verify, restore an independent copy, select that copy, clean the calculation cache and preview a saved PDF. Test native clipboard and external links with inert examples. | Original data remain unchanged, the selected copy is independent, drafts return, cache rebuilds and PDF bytes/rendering agree. Clipboard and approved system handlers work. Do not send messages from a mail handler. |
| Windows session/power | On the disposable device, lock/unlock, sleep/resume and test logoff/shutdown followed by reopening. | Lock does not cancel accepted work. Resume requires fresh quiet/activity checks. Windows sleep need not execute tasks. Logoff/shutdown recover durable state; uncheckpointed edits and unfinished in-memory tasks have no durability promise. |
| Clean Windows/components | Test a machine without global Node/Python and, separately, one missing the required WebView2 runtime. | Bundled runtimes suffice. The component's Microsoft signature and final installed version are checked; failures remain retryable and preserve the previous program/data. Record actual download size. |
| Physical offline | After first installation, physically disconnect the network and repeat import, prepare, chart/figure, export, history, backup, restart and restore. | All ordinary workflows complete with the expected data and files. A denied proxy is not physical-offline evidence. |
| Version coexistence | Run the declared current V2.2, RC and desktop native entries with distinct synthetic profiles. | Their versions, roots, ports, cookies and keys are recorded; own requests authenticate and cross-session access fails. Copied historical services alone do not pass native/current-RC coexistence. |
| Accessibility and trial | Test keyboard-only navigation, focus after reopening/dialogs, IME, supported DPI scaling and a screen reader. Give the same fixture task to a researcher. | Essential actions are operable and understandable; record blockers and actual participant/device observations. Automated DOM checks do not substitute for these observations. |
| Distribution | Resolve remaining upstream notice holds, sign with the intended identity, then verify the exact signed bytes and complete release acceptance. | No declaration-only notice hold is silently cleared. Signature identity, timestamp and signed-file hash are recorded. Any post-test byte change needs an explicit evidence mapping or rerun. |

## Recording

Fill the kit's `results-template.json`. For each gate record the exact package and Setup hash, machine/account, action, expected result, actual result, evidence files, outcome, operator and UTC time. Use `pass`, `fail`, `not-run` or `blocked`; never default to pass. Hash screenshots, recordings, process samples and exported files. Keep the raw failure before repair or rerun.

The repository also provides `desktop/scripts/release_preflight.py` and `desktop/contracts/release-record-template-v1.json`. Copy the template into a private evidence directory, bind the exact package and Setup hashes, and fill it from actual observations. Its fourteen gate IDs match this checklist. Evidence paths are portable relative paths beneath that record's directory; each file needs its SHA-256. Provide the six mandatory report references and at least ten native small-workspace restore samples. The existing 32 MiB, 100 MiB and 5,000 ms limits are unchanged.

Run the checker from this repository:

```powershell
python desktop/scripts/release_preflight.py --bundle '<generation>' --build '<matching-build-directory>' --acceptance-record '<private-evidence-directory>\release-record.json' --expected-signer-thumbprint '<intended-certificate-thumbprint>' --output '<fresh-report-path>'
```

The checker reads the artifact, its actual Windows signature and the supplied record. It verifies file hashes, gate identities/scopes, evidence references, recorded numeric limits and mandatory report references. It rejects explicitly headless JSON/JSONL evidence as the sole proof for a native gate. A record and its attachments still require human review for truthful scope and sufficient observations; this tool does not perform measurements or authenticate a person's account of a physical test. Exit 0 means the supplied release checks pass, exit 1 means artifact verification failed, and exit 2 means verified bytes still have release blockers or incomplete inputs. The tool does not sign, publish, operate windows, change trust stores or replace shortcuts. Signing needs an explicitly selected intended identity.

Do not edit the original fixture files or their expected values to make a check pass. Do not replace the real desktop shortcut until the intended installed candidate and user-facing entry have been accepted. The kit does not authorize signing with an arbitrary certificate or publishing an unaccepted package.
