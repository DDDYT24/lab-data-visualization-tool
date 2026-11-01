# Remaining desktop repairs and verification

Date: 2026-10-25 UTC. Scope: the isolated desktop checkout at `b587db848ba5f1930657d5a316e0160c71864ead` plus its existing work. The user authorized continued repairs while excluding desktop control. No native LabViz window, tray control, physical network/power action, signing or public release was performed. The original V2.3 TODO remains the current-status authority.

## Current private candidate

| Artifact | Identity |
| --- | --- |
| Bundle | `outputs/desktop-d06/ps-module-fix-20261025/bundle` |
| Package | `f83250bcba777c1aeb5bbc3a33f383a3229e1779d16a402a57a8a402182144e6` |
| Payload | `6e1325ec24c799804d856e67655873cd6fee00cc02e056efc092acd651e208f9` |
| Setup | `outputs/desktop-d06/ps-module-fix-20261025/build/LabViz-Desktop-2.3.0-desktop.1-Private-Setup.exe` |
| Setup SHA-256 | `d32ae25d251813f3ea415dec2bbb9989937a78a3894d933d3fd31b8646fa6970` |
| Setup bytes | 147149293 |
| Bundled members | 13,174 |
| Signature | NotSigned; private candidate |
| Evidence index | `outputs/desktop-remaining-20261025/verification.json` |

Every bundled member is byte-identical to the preceding notice delivery `6328d05fad70394fa7ac7fddec1aaa762dede977c70e761c31a1f79ced13b428`. Only the outer manifest's installer-source hashes differ, for `prepare-install.ps1` and `webview2.ps1`. The new Setup embeds those repaired scripts and has its own fresh installer acceptance. Earlier runtime, large-workspace and 20-cycle memory evidence retain their original executed identities; `package-equivalence.json` supplies the complete byte comparison. The previous frozen bundles, installers and sealed evidence remain unchanged.

## Installation repair

Fresh signature inspection exposed a real Windows PowerShell 5.1 module-load failure when it inherited PowerShell 7's `PSModulePath`. The original error and raw diagnostic are retained. A separate output-decoding failure was also corrected in the new preflight reader: Windows errors need not be UTF-8, so it captures bytes and explicitly decodes the UTF-8 JSON output.

The installation wrapper now uses the current engine's module directory for its process. WebView2 signature inspection imports the absolute Windows security module and invokes its exported cmdlet directly, avoiding a same-named function. It still requires a valid Microsoft signature before component installation. No system module, persistent environment setting, certificate store or trust setting was changed.

Seven real-engine regression scenarios passed: the inherited-path failure was reproduced in the old helper; the fixed helper accepted the actual Microsoft-signed bootstrap and rejected an unsigned file; old and fixed helpers were compared against an injected same-name function; and simulated missing-runtime branches both accepted the real signature and rejected the unsigned replacement before installation. The official 1,876,736-byte bootstrap was downloaded and signature-checked, never executed. Component detection/installation in those two branch controls was simulated. The separate six existing WebView2 branch controls also passed. These checks do not accept an actual missing-runtime installation on a clean Windows machine.

## Verified results

| Check | Observed result |
| --- | --- |
| Python desktop suite | 107 tests passed, including ten release-preflight cases. |
| Observer unit suite | Four tests passed; classification, bounded metadata, redaction and captured-failure preservation covered. |
| Exact updated Setup | Eight silent installation/repair/update/rollback/failure/version/uninstall/reinstall checks passed; test registration removed. |
| Data retention | Newly imported scientific data, metadata and revision fingerprints survived compatible-generation rollback; durable projects retained. |
| Actual activation failure | A fault in the owned newly deployed controller made the exact Setup return 20; the previous program, scientific data and checkpoint survived and previous services were healthy. Cleanup completed. |
| Long path | Actual bundled Node/API/PDF operations passed in the owned Chinese path longer than 260 characters. |
| Final reopening | The updated candidate restored its editable draft, retained the fixture and newly created projects, and preserved all 17 frozen small-workflow exports. Owned services exited normally without survivors. |
| Calculation-cache regeneration | The bundled Agg renderer regenerated its font/cache outputs; all recorded other profile bytes remained unchanged. |
| Fresh small fixture | 15 scientific checks, 17 exports and seven maintenance checks passed on the preceding byte-identical runtime, with original executed identity recorded. No new large-workflow run is claimed for this fixture. |
| Instrumented maintenance | Three race/maintenance checks and all 24 existing small/large frozen exports passed with the observer integrated, on the preceding byte-identical runtime. |
| Portable operator verifier | All 13,174 current bundle members, manifest identity and exact updated Setup hash passed. |
| Release preflight | Current artifact integrity passed. Release acceptance stayed false with 26 explicit pending/blocking entries, including fourteen unrun operator gates. |

## Headless connection diagnostics

The client observer retains up to 32 metadata events, correlated host lifecycle events and cycle identity. It records no URLs, credentials, request bodies or raw error text. Failures preserve their captured evidence through disposal. Its output explicitly declares that it cannot accept native gates. The maintenance client captures a failed attempt and may probe the existing page read-only; it does not resubmit scientific operations.

Four controlled live cases passed: client transport disconnection while an independent page probe succeeded, an actual renderer-crash event, whole-browser closure, and planned suspension/rebuild. Each headless adapter and its owned services exited with no survivors. A renderer can emit a crash event and later answer a read, so that control reports actual recovered-read status rather than fabricating an exception. The earlier over-strict test assumption and its failure are retained. These controls and the successful maintenance run improve recurrence diagnosis; the original historical disconnect was not reproduced and its cause remains unconfirmed.

## Release records and remaining external work

`release_preflight.py` checks the canonical manifest, exact Setup, actual Windows signature, intended signer, timestamp, remaining notice issues and the supplied fourteen-gate record. Records need exact artifact identities, machine/operator/time, actions and expected/actual observations, scoped relative evidence with hashes, six mandatory report references, the existing 32 MiB / 100 MiB limits and at least ten native small-restore samples no greater than 5,000 ms each. Invalid hashes, escapes, booleans/non-finite numbers and explicitly headless evidence standing alone for native gates are rejected. Input validation does not authenticate a human account of a physical test or establish that the attached observation is sufficient; human review remains required.

The refreshed kit `outputs/desktop-remaining-20261025/operator-acceptance-kit.zip` includes the standalone byte verifier, synthetic CSV/TSV/XLSX data, expected values, policies, exact updated identities, operator instructions, blank manual records and unsent notice requests. The repository preflight needs this repository's runtime and policy dependencies; it is not presented as a standalone kit program. The [operator guide](D7_OPERATOR_ACCEPTANCE.md) gives the command and pass criteria. All fourteen native/manual gates remain unrun.

Fresh official registry/source checks still did not supply complete package-specific upstream notice text for `client-only@0.0.1` or `size-sensor@1.0.3`. Their release integrity is unchanged. No copyright holder was inferred, no statement was fabricated and no maintainer message was sent. The three previously authenticated supplements remain intact. These two notice holds remain open.

Read-only inspection found zero currently valid code-signing certificates with private keys in CurrentUser/My and no configured intended task signer. Other stores/files were not exhaustively searched. No arbitrary or self-signed identity was used. The Setup remains unsigned and unpublished.

Real tray/WebView2 behavior, complete native memory/recovery budgets, native files/PDF/clipboard/links, system events, actual clean Windows/component installation, physical offline use, current-version native coexistence, accessibility/device/researcher trials, intended signing and release remain open. The historical disconnect cause also remains unresolved. D0-D7 are not accepted as a whole. The actual desktop shortcut was hash-checked without modification and still targets D2.

Stable, original development and RC source bytes, all four HEADs and logical indexes are preserved except for the authorized append to the original V2.3 TODO. This snapshot makes no commit, push or entry-point replacement. Prior sealed evidence is checked again by the current evidence index.
