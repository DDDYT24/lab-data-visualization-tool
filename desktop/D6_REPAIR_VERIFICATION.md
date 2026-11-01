# D6 repair verification

Date: 2026-10-24 UTC. These repairs use the isolated desktop checkout at `b587db848ba5f1930657d5a316e0160c71864ead` plus its existing work. Native windows and the real desktop shortcut were not operated. The original V2.3 TODO remains the status authority.

## Build lineage

`build_verified.py` captures host/web input file hashes before and after compilation, build-tool hashes, tool versions, commands, complete build logs and output file hashes. A fresh final-crate Rust compile is requested without cleaning the shared dependency cache. The web build preserves the original generated `next-env.d.ts` bytes. A source or build-tool change during compilation fails the receipt.

Host inputs include Rust/configuration/lock files, boot content, the embedded adapter and icon. Web inputs include application source, public assets, translations, scripts, configuration, the package lock and installed npm lock. Generated `next-env.d.ts` is restored separately. These local records establish the tested build's input/output relationship; reproduction on another machine or toolchain remains untested.

Staging requires successful matching component receipts and logs. It checks current input and output bytes, records API/contracts/assets/tool hashes, rechecks inputs after copying, and binds the runtime files and payload identity. Freezing packages the receipts and logs. Compilation checks the same lineage, exact package members, installer sources and matching background evidence. Old frozen candidates are refused before mutation.

The new host source digest is `975136dc84c97c7fbe8206f1ba745325db455e7a47cf10d34610978295f8f919`; the web source digest is `4535b6edc28e5002f9d2d599c98328af639cc7627397d8a6ed1e359692bb3cbd`. The first experimental wrapper receipts are retained separately; the frozen candidate uses only `provenance-final`, which verifies that its build tools remained unchanged. Static PE imports match historical H. The native executable was compiled and not launched.

## Dependency notices

Python collection now recognizes both LICENSE and LICENCE and follows safe wheel RECORD references to additional notices outside dist-info. All 51 distributions have collected notice files, including the original openpyxl and et_xmlfile LICENCE texts.

All 236 Rust dependencies in the locked Windows normal/build closure have copied, hash-bound notice files. The plist LICENCE is included. Five crates without standalone notices use version- and commit-matched upstream material: alloc-stdlib, selectors and the three webview2-com packages. Their original declarations and applicable source notices are preserved. Cargo graph roles are recorded; a normal dependency edge does not itself establish final binary linkage. Supplement sources and retrieval hashes are in `desktop/packaging/license-supplements/sources.json`.

The web inventory records all 225 production lockfile entries: 168 installed packages and 57 uninstalled entries with optional/platform metadata. Complete notices embedded in README files are collected, same-release Next notices are retained, and the exact icu-type-parser upstream notice is included. Entries are no longer silently skipped.

Five installed releases still provide only a license declaration without a package-specific full notice: `@swc/counter@0.1.3`, `buffer-from@0.1.2`, `client-only@0.0.1`, `object-keys@0.4.0`, and `size-sensor@1.0.3`. Their metadata and registry integrity values are recorded as review issues; copyright holders were not inferred. This private candidate does not close those upstream notice questions or accept public distribution.

## Verification

| Check | Result and scope |
| --- | --- |
| Desktop Python | 83 tests passed, including seven new packaging tests for changed source/output/logs, unstable builds, LICENCE/RECORD collection, upstream supplement identity, missing web notices and frozen-output protection. |
| Fresh component builds | Locked/offline Rust host and production web builds passed with before/after input equality and unchanged build tools. |
| Scientific production | Small workflow: 15 checks and 17 exports. Large workflow: seven checks and seven exports. Independent checks confirm 100,000 observations, 50,000 units, mean 500.5, four panels and recorded hashes. |
| Application maintenance | Seven checks passed on this exact package: inline PDF bytes, live-editor backup/restart, verification, independent restore, original/copy selection, restored-profile health and installation preparation that checkpoints and exits. |
| WebView2 | Six isolated branches passed. Existing-runtime detection reads the registry; download, signature and component installation are mocked. No system component was installed. |
| Installer lifecycle | Eight checks passed: exact private installation and service activation, a shortcut created only inside the fixture, same-version repair, compatible candidate update/rollback, failed activation recovery, future-epoch refusal, uninstall byte retention and reinstall/reopen. Test registration was removed. |
| Data after upgrade | A real authenticated import adds one project after a compatible update. Rolling back retains the original five projects and the new one. |
| Setup failure reporting | The exact installer returns 20 after a fault is injected into the deployed candidate. The previous program, scientific data and checkpoint remain available and its services pass health checks. |
| Final reopen | Three checks retain an editable chart draft, all 24 frozen export hashes and the post-upgrade project after rollback and Setup failure. Fenced explicit exit then completes normally with no owned survivors. |
| Negative evidence checks | Changed component input/output/logs and invalid supplement identities are rejected. CLI checks reject frozen staging, mismatched package evidence, overwriting component reports and component outputs outside the checkout's outputs directory. |

The WebView2 test helper now requires a fresh explicit output directory. This round preserved the historical component receipt and recorded its hash. The earlier audit's historical-output incident remains recorded separately; no missing historical bytes were reconstructed.

The final reopen driver initially connected before the new session was ready, then searched for a control hidden by the correctly restored maintenance drawer. Both diagnostics are retained. The corrected driver waits for the new readiness event and returns to the active task before checking the visible editor. Application code and scientific assertions were unchanged for these rechecks.

## Candidate

| Artifact | Binding |
| --- | --- |
| Package ID | `157aae3995255a7746d1b973d6c4c50b3322d7075a4a032bdc294af43e79ce0f` |
| Payload ID | `86b85060e11b9ce4278ba938b4424107e0a8e0b22b880ef71801d5ca86afffe6` |
| Host SHA256 | `f13e8e94353b236a2320808c6881fef808fe3a831257ac031dfc43807565a7a7` |
| Installer SHA256 | `cc596186f6b0d9305995fa0c9d537866f5e5b076f9eac2c74fc083bfa7810f72` |
| Installer bytes | 147,136,633 |
| Frozen bundle bytes | 569,884,318 |

The private output is `outputs/desktop-d06/repair-20261024`. Its final evidence index binds the installer lifecycle, installation footprint and cleanup records. The complete first program installation contains 594,908,966 logical bytes in 13,182 files. With two retained candidates, the program root contains 1,169,646,990 bytes in 26,354 files. These totals include the stable launcher, maintenance support, retained candidates and uninstall files; profile data and system WebView2 are excluded. The missing-runtime download remains unmeasured on a computer without WebView2. The setup is unsigned and public release acceptance is false.

## Remaining acceptance

The five web notice review items remain open. Native tray/WebView2/resource acceptance, installed GUI/profile switching and native file/dialog/PDF/clipboard/DPI/keyboard behavior remain untested. Clean Windows, actual missing-component installation, physical offline use, the remaining D7 fault matrix and human acceptance are separate gates. The original H package and D2 desktop entry remain historical artifacts.
