# Private Windows desktop delivery verification

Date: 2026-10-25 UTC. This snapshot covers the isolated desktop checkout at `b587db848ba5f1930657d5a316e0160c71864ead` plus its existing work. Current status remains in the original V2.3 TODO. The user confirmed that this delivery excludes desktop control. No native LabViz window was launched; the real desktop shortcut was inspected without modification.

## Delivered candidate

| Artifact | Identity |
| --- | --- |
| Bundle | `outputs/desktop-d06/delivery-notices-20261025/bundle` |
| Package | `6328d05fad70394fa7ac7fddec1aaa762dede977c70e761c31a1f79ced13b428` |
| Payload | `6e1325ec24c799804d856e67655873cd6fee00cc02e056efc092acd651e208f9` |
| Setup | `outputs/desktop-d06/delivery-notices-20261025/build/LabViz-Desktop-2.3.0-desktop.1-Private-Setup.exe` |
| Setup SHA-256 | `c979b92b235f4a9ab4ed0c14254b4f01759ced9e7b5f6d028fce18e3ddd5a536` |
| Setup bytes | 147145710 |
| Bundled members | 13174 |
| Signature | NotSigned; private delivery only |
| Evidence | `outputs/desktop-delivery-20261025/verification.json` and `outputs/desktop-notice-delivery-20261025` |

The controller contains the bounded DPAPI type fix verified in [the memory slice](D5_MEMORY_REPAIR_VERIFICATION.md). The intermediate memory package `e511fa0ee357cdc62d1e064fc2cd3e09e08755bbd18047600bbaa90d047e8d8b` was compiled and passed its exact silent lifecycle. Three authenticated upstream notice supplements then required a separate frozen package and Setup. All app, host and runtime bytes are identical between those two packages. Three notice files were added; three aggregate notice records changed. The outer installer-source hashes reflect the collector and supplements. `notice-execution-equivalence.json` records the complete difference; original memory/workflow reports retain their executed identity.

The final package has fresh service health, full small/large scientific workflows, maintenance and CDP diagnostics. Its own exact Setup passed the complete silent lifecycle, rollback/retention and activation-failure checks, long Unicode paths and final reopening. The prior 20 headless restores and DPAPI stress evidence remain attached to their original package and transfer only through the verified execution-byte equivalence. No headless result accepts native resource gates.

## Results

| Check | Observed result |
| --- | --- |
| Binary header | AMD64 PE32+ Windows GUI subsystem (2) verified without launch. Actual native window/console behavior remains an operator check. |
| Desktop tests | 97 passed, including two notice-provenance cases and four retention cases that protect durable/unexpired projects while permitting recorded temporary-demo expiry. |
| Component branches | Six isolated WebView2 branches passed: already installed, signed install/recheck, untrusted signature, failed download, failed installer and false success without a runtime. No actual component installation was performed. |
| Scientific workflows | Fresh final package: small 15 checks/17 exports; large seven checks/seven exports, retaining 100,000 observations, 50,000 units, mean 500.5 and four panels. |
| Exact final Setup | Eight checks passed: full-file installation/activation, fixture-only shortcut, repair, compatible update/rollback, failed activation recovery, future-epoch refusal, uninstall retention and reinstall. |
| Data after upgrade | New authenticated imports survive compatible rollback; data blob, metadata and current-revision fingerprints are unchanged. |
| Injected Setup failure | The exact final Setup returns 20 and retains the previous usable program, scientific data, checkpoint and healthy services. |
| Long Unicode paths | Bundled Node/API import a 12-row CSV and write/reread the matching frozen PDF through Chinese paths longer than 260 characters. |
| Final reopen | Editable draft, all 24 frozen downloads and post-upgrade imports remain available. Exit is normal with no owned survivors. |
| Cache rebuild | Bundled Matplotlib Agg regenerates font cache and a valid PNG; all 295 non-cache profile files retain their hashes. |
| Complete new installation | 595035606 logical bytes and 13185 files; excludes profile and system WebView2. |
| Two retained generations | 1169884954 logical bytes and 26360 files; the overhead is reported separately. |
| Operator verifier | The embedded Python verifies all 13174 bundle members and exact Setup. Wrong checksums and an output inside the bundle are refused. This check launches no application. |

An initial retention assertion included a temporary 3D sample whose recorded expiry had already passed. Its removal occurred before rollback; the prepared rollback backup and the final project rows matched exactly, and every original local project remained present. The acceptance checker now permits only recorded expired temporary projects to disappear, still rejects durable/unexpired loss and unexpected projects, and records any allowed removal. Four focused cases cover that boundary. The first failed assertion, database observations and diagnosis are retained. Product cleanup and rollback behavior were unchanged.

The profile-hash acceptance driver now hashes the same complete file set with eight workers and rejects linked descendants before hashing. Its coverage matches the previous enumeration; no preservation assertion was removed. Post-install automation waits for restored client readiness before selecting Analysis and closing an old drawer. The runtime and user interface were unchanged by those driver repairs. The intermediate fault wrapper recorded the expected Setup exit code 20 and data/program preservation, then exceeded its parent 800-second wait during cleanup. Independent follow-up verified that its registration, entry point, uninstaller and running marker were gone, with project IDs and checkpoint retained; an unobserved cleanup exit code was not inferred. The final delivery uses longer total orchestration waits with the same product assertions and unchanged native resource targets. A private-fixture archive retry used extended Windows paths within the same checked trial boundary; no profile directory was moved or deleted.

## Upstream notices

Three full texts were copied from maintainer commits and bound to the exact installed metadata and byte-identical implementation files. The original tarballs' missing notices remain explicit. The collector does not infer a copyright holder.

- `@swc/counter@0.1.3`: [maintainer notice addition](https://github.com/swc-project/pkgs/commit/3030e6758766ba166aada785c8fdfc91dd449cfb); the source package remains 0.1.3 and its implementation matches.
- `object-keys@0.4.0`: [maintainer MIT notice addition](https://github.com/ljharb/object-keys/commit/28a7c3ada8a84cb88dfa4d17daf2d24a3a440dc4); version 0.4.0 and all four implementation files match.
- `buffer-from@0.1.2`: [maintainer MIT notice addition](https://github.com/LinusU/buffer-from/commit/286425afdd44bbb6cadccf1bc8fce07522da52c4); that source package is 1.0.0, while its implementation is byte-identical to 0.1.2. Both declare MIT. The later notice origin is preserved explicitly.

The original npm tarballs for all five reviewed packages were also downloaded and checked against both registry and lockfile SHA-512 values. Every installed release file matched, and none of the five tarballs contained a notice-named file. Two unsent upstream requests are included in the operator kit.

Two notice holds remain: `client-only@0.0.1` and `size-sensor@1.0.3`. Reviewed release files and pinned source do not supply a package-specific complete permission/copyright notice. Their declared MIT/ISC identifiers and publisher/author metadata are retained without inventing a notice. Full registry/source receipts and the exact reviewed file lists are in `license-review.json` and `license-research`. This is a collection/provenance result, not a legal determination. Public release remains unaccepted.

## Headless connection investigation

The diagnostic host now records page closure/crash, context closure, browser disconnection and controller exit, with a per-cycle expected-close flag. It logs no credentials. The original maintenance source was reconstructed to its prior SHA-256 before adapting only its repository path. Both original and fixed clients passed a fresh full/large workflow sequence. Twelve independent CDP client exits left the production page alive. One deliberately injected main-page closure was captured as unexpected; planned context/browser teardown was classified as expected. The adapter and owned services exited normally.

The initial historical disconnection was not reproduced. Its cause remains unconfirmed. These results rule out automatic page closure from the tested client exits in this sequence; they do not establish why the earlier attempt disconnected. Original failures and raw lifetime events remain available for a recurrence.

## Remaining acceptance and operator kit

The private kit at `outputs/desktop-delivery-20261025/operator-acceptance-kit.zip` contains candidate/Setup hashes, the read-only verifier, synthetic CSV/TSV/XLSX fixtures, expected values, resource/shutdown policies and a 14-gate blank result record. Its [operator instructions](D7_OPERATOR_ACCEPTANCE.md) specify steps and pass criteria. All native/manual result entries remain `not-run`.

Real tray/WebView2 behavior, full native 32 MiB / 100 MiB / 5,000 ms budgets, native files/PDF/clipboard/links, profile switching, Windows session/power events, clean Windows/missing-component installation, physical offline use, current RC/native entry coexistence, accessibility/device/researcher trials, signing and public release remain unaccepted. The two upstream notice holds and the historical disconnect's unconfirmed cause remain explicit. D0–D7 are not accepted as a whole.

The real desktop shortcut still targets D2. Test installations were confined to the owned synthetic fixture and their registration was removed. Its previous program fixtures were archived with all profile bytes verified unchanged. Stable, development and RC sources, Git HEADs and logical indexes are preserved except for the authorized append to the status TODO. Previous frozen packages and the 116/158-file evidence indexes remain unchanged.
