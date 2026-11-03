# D1–D3 source repair verification

Date: 2026-10-23 UTC. These repairs are based on the isolated desktop checkout at `b587db848ba5f1930657d5a316e0160c71864ead` plus its existing uncommitted work. Verification uses synthetic profiles, a private headless browser and bundled services. The revised native host was compiled and not launched. The D6 H installer and the real D2 shortcut remain unchanged.

## Changes

D1 now distinguishes the existing three numeric gates from mandatory resource reports in [resource-budget-v1.json](contracts/resource-budget-v1.json). The host, suspended tree and small-workspace targets remain 32 MiB, 100 MiB and five seconds. Installation size, active/peak memory, cold start, large-workspace recovery and busy-tray measurements are required reports with no implicit numeric pass threshold. New numeric targets must be set prospectively. Native measurements still determine the final D5 topology decision; this clarification does not retroactively accept a candidate.

D2 explicit exit offers waiting, cancellation and a separately confirmed interruption. Waiting keeps accepted uploads, calculations, saves and downloads running. Once activity is verified idle, the controller closes new-work admission and notifies the host. A final confirmation checks current activity, downloads, maintenance and drain generation again. Cancellation reopens admission. Maintenance submissions share an atomic exit fence, so a new backup cannot slip between the idle check and exit approval. Unknown activity cannot approve exit. The host rejects new downloads while exit is pending, ignores old-controller notifications and explains that uncheckpointed edits may be discarded. Interrupted exit still needs explicit confirmation. Revised native dialogs require direct acceptance.

Oversized preference writes now drain a bounded rejected body before returning 413 on Windows. This fixes a connection-reset failure observed by the original HTTP regression test without accepting oversized or unknown settings.

D3 preserves editable records across normal editor unmount/remount and checks their project/revision before applying them. A changed saved revision retains the prior draft and requires an explicit decision. Project switching keeps drafts separate. Successful user-confirmed deletion removes only that project's in-memory draft references. Settings is now an allowed recovery route. Saved table and chart pagination is restored by fetching the saved revision again; result rows and preview capabilities are not serialized into the checkpoint.

Staging reclaims only owned, unreferenced source files after a durable checkpoint ACK. The current, previous and retained recovery snapshots protect their references. Unknown/corrupt snapshots, unfamiliar files and links block reclamation. The steady staging ceiling remains 512 MiB. An active capture with verified recovery material may use a temporary reserve of at most 100 MiB only when enough known-unreferenced data can be reclaimed after its ACK; the hard transient ceiling is 612 MiB. No source is deleted to create that reserve before the write succeeds. A failed write retains old and newly staged sources. Protected recovery material can still cause a safe refusal to add files.

Staging supports a fresh `--output` directory and refuses a frozen candidate before mutation. The README now uses the current D6 staging entry instead of the frozen D3–D5 entry.

## Verification

| Check | Result and scope |
| --- | --- |
| Desktop Python | 69 tests passed, including waiting/cancel/reconfirmation, late downloads and maintenance, settings, retention, atomic failure, capture reserve, unknown recovery material, authorization and preferences. |
| Frontend | 138 tests in 23 files passed. Six new real-provider/hook/component regressions cover navigation, the real chart form, revision conflict, project switching, explicit deletion and chart pagination. Type checking, lint and production build passed. |
| Native host | Locked release build passed with the Windows common-controls feature needed for named dialog buttons. Native dialogs were not exercised. |
| Real staging pressure | 512 MiB of old sources remained present while a 1 MiB replacement used the bounded reserve. After its durable ACK only the referenced 1 MiB remained; new source access and settings checkpoint were valid. |
| Scientific production | Small workflow: 15 checks and 17 exports. Large workflow: seven checks and seven exports. Independent checks retain 100,000 observations, 50,000 independent units, mean 500.5 and four panels, plus all recorded export hashes. |
| Real navigation and recovery | Five checks passed: a scientific title survives normal Next navigation; an observation table returns to rows 101–200 of 100,000; settings can checkpoint, destroy the test page/services and reopen; scientific heads/counts stay unchanged; all 17 small frozen export hashes remain identical. |
| Real exit protocol | Seven checks passed with a multipart upload held open: waiting does not interrupt it, cancelling lets it finish, idle exit fences admission, a late download blocks confirmation, cancellation reopens admission, a ready exit rejects new maintenance submissions atomically, and fresh confirmation stops services normally with no owned survivors. |

The tested preview is under `outputs/desktop-repair-d1-d3-20261023/bundle`. Its payload candidate is `8914606e3a9ac1678a04a076e9b528777db87c4a23d3fdf3377aaf312e35c1aa`; the exact host and runtime hashes are retained in its build receipt and the repair evidence index. This is an unfrozen preview directory, not an installed update or a new release installer. The payload ID does not replace the separate host/runtime bindings.

The final controller/maintenance fence was restaged and rechecked through settings destruction/recreation and the seven real exit checks. The scientific and navigation evidence reuses the byte-identical application payload, native host, session and preference modules; only the final exit-fence paths changed and were rechecked.

The evidence index is `outputs/desktop-repair-d1-d3-20261023/verification.json`. Earlier HTTP-reset and ref-access lint diagnostics are retained alongside successful rechecks. The source and process preservation receipts describe the final cleanup; synthetic data and test records remain private.

## Remaining acceptance

Native exit dialog selection, tray interaction and exact WebView behavior remain untested in this repair. D4's abnormal automatic-stop reporting and runner shutdown-result propagation remain open. D5 needs final native resource acceptance, including the previously retained slow samples. D6 license collection, build-source provenance and final installer delivery, and D7 clean-machine, physical-offline and human acceptance remain separate work. The original V2.3 TODO is the status authority.
