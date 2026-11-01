# Desktop controller memory repair: scoped verification

Date: 2026-10-24 UTC. This snapshot closes the current memory and recovery slice after the user requested a bounded finish. Current status remains in the V2.3 TODO. All live checks use isolated synthetic profiles. No native application or real desktop shortcut was operated.

## Repair and evidence

The preceding frozen controller declared a ctypes structure inside every DPAPI read. Python's global pointer-type cache retained those classes after garbage collection: 100, 200, 300 and 400 successful decryptions retained 100, 200, 300 and 400 local classes. This establishes an unbounded type retention defect; it does not explain every earlier private-commit increase.

The structure is now declared once and reused. Decryption and `LocalFree` cleanup remain in the same control flow. A new regression performs 400 real reads and requires the warmed pointer cache to stay constant. It rejects the preceding frozen controller as expected. All 91 desktop tests passed. The repaired candidate's embedded Python completed 1,001 correct decryptions, with zero pointer-cache growth after warm-up.

Recovery test drivers now wait for a ready, idle client heartbeat after navigation, then close restored drawers and select the task explicitly. A visible workspace can precede the asynchronous restoration of its task and drawer. The suspension driver also takes the expected chart ID from the selected menu option and waits for the asynchronous chart fetch before reading the selector. Its previous immediate read had captured the old Violin ID after clicking Raw. Failed locator reports, checkpoint bytes and screenshots remain in the evidence directory. An initial CDP page-disconnection attempt is retained without a claimed root cause; the complete later recovery sequence ran in a fresh owned session.

| Fresh check | Result |
| --- | --- |
| Desktop unit/integration cases | 91 passed |
| Embedded-Python DPAPI reads | 1,001 correct; warmed pointer-cache growth 0 |
| Small scientific workflow | 15 checks; 17 frozen exports |
| 100,000-observation workflow | Seven checks; seven frozen exports |
| Draft roundtrip | Five checks passed, including selected chart, staged file, undo/redo and notes |
| Continuous headless rebuilds | 20 passed; scientific counts and revision heads unchanged |
| Editable headless restore | 3993–4405 ms, with all samples retained |
| D6 maintenance | Seven checks passed, including backup, verification, independent restore and graceful preparation for update |
| Cache and maintenance race | Three checks passed; all 24 frozen downloads retained their hashes |
| Actual service faults | Eight checks passed, including API loss, one complete recovered publication, bind-race refusal and cache preservation |
| Shutdown | Normal exit; no owned survivors |

There are 21 stable suspended intervals with only the controller. Median private commit changed from 22.301 to 22.262 MiB. Raw samples and every interval are retained. These observations establish the stated test scope, without certifying arbitrary-duration behavior or the complete native resource budget.

## Candidate boundary

The repaired preview is `outputs/desktop-d06/memoryfix-20261024/bundle`, package `e511fa0ee357cdc62d1e064fc2cd3e09e08755bbd18047600bbaa90d047e8d8b`. The controller SHA-256 is `366d2792fb342a2cc4a3bb701fdf6b5fe1105169a163f66ec78f31ea01cb07ea`. All app payload and host bytes match the preceding package; only the controller and three hash-binding records changed. The staged candidate is frozen and has matching background verification.

**No new Setup has been compiled or installed for this repair.** Full installer compilation, repair/update/rollback/retention/failure tests and final post-install reopening are deferred to the next slice. Prepared delivery scripts are not completion evidence. The latest previously verified Setup remains the closeout package `9015fb4e25c6878da113c6fca5d4c24a04179ccd407cdab47ed6c74e1ed3d8bc` and contains the old controller. Its eight lifecycle checks, long-path checks and cache-renderer checks remain historical evidence for those exact bytes.

The preceding 158-file evidence index and frozen package are preserved and hash-checked again. The actual desktop shortcut still targets D2, so opening that shortcut does not run the new preview. Stable, development and RC source trees remain unchanged apart from the authorized append to the V2.3 status TODO; Git HEADs and logical indexes are unchanged.

## Remaining work

- Compile an installer for this exact repaired preview, run its complete silent lifecycle, rollback/retention and activation-failure checks, then reopen the installed result and repeat long-path/cache checks as appropriate. The frozen manifest and matching background proof are ready.
- Resolve the five package-specific upstream web notice reviews. Public release remains unaccepted.
- Track the initial unexplained headless CDP disconnection if it recurs. Later successful recovery tests do not establish its cause; all original logs remain available.
- Accept the real tray/WebView2 lifecycle, complete native 32 MiB / 100 MiB memory and 5,000 ms restore targets, native file/PDF/clipboard/link behavior, profile switching and actual Windows session/power events.
- Complete clean-Windows and missing-WebView2 installation, physical offline use, current RC/native launcher coexistence, keyboard/DPI/screen-reader/device and researcher trials, then signing and release acceptance.

D0–D7 are not accepted as a whole. The confirmed DPAPI type retention defect and the scoped background recovery checks above are complete. Broader tests remain explicitly separate.
