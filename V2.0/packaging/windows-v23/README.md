# Rebuild the Windows V2.3 RC

This entry builds the V2.3 identity. V2.2 packaging remains in its existing directory and retains its release identity.

Use Windows 10 or later, a full Git checkout with the published `v2.2.0` tag, Python 3.13 for build tooling, and Inno Setup 6. The pipeline downloads hash-pinned CPython and Node.js archives, installs hash-locked Python wheels and the npm lock, and keeps its output under the checkout's ignored `outputs/v23-release` directory. A network connection is needed to prepare those build inputs.

```powershell
py -3.13 V2.0/packaging/windows-v23/build_rc.py
```

Commit the reviewed source before producing public assets. Build tooling and data fixtures use private directories. The installer tests refuse to adopt an existing V2.3 uninstall registration; use a disposable Windows runner or a machine without an installed V2.3 identity.

The pipeline phases can also be run separately with `--phase`:

| Phase | Result |
| --- | --- |
| `prepare` | Pinned runtime archives, isolated build dependencies, and Chromium |
| `verify-source` | API lint/types/local regression, frontend checks and build, fresh identity/data and native receipts |
| `stage` | Self-contained payload, production dependency notices, complete hashes, and source input binding |
| `portable` | Actual bundled browser/scientific workflows, occupied ports, process recovery, backup/restore, released-V2.2 import, upgrade/failure rollback, and explicit deletion checks |
| `installer` | Inno compilation, installed-file integrity, actual installed browser workflows, same-version reinstall, uninstall retention, and reopen after reinstall |
| `artifacts` | Installer, synthetic examples, source/payload/provenance records, verification summary, and SHA256 sums |

The local API phase explicitly records excluded PostgreSQL/MinIO test files. The existing Linux CI integration job runs those tests with private services. Both evidence scopes are required before publishing.

Use a new `--output` directory for a changed candidate or after a failed phase, preserving the earlier logs and fixture. `--runtime-cache` can reuse already-downloaded archives after their pinned hashes are checked. `--development` permits a private uncommitted installer experiment; the public asset phase still refuses dirty or changed source.

Do not run source recovery tests concurrently with live scientific acceptance on the same profile. Native fixtures belong under the current user's `LabVizV23Acceptance` directory. They are separate from default V2.2/V2.3 user data. Failed fixtures are retained for inspection.

The pipeline never signs or publishes a release. After PR checks finish, create a draft pre-release for the verified commit, attach the complete asset set, download it again, verify hashes and the installer workflow, then publish the RC without replacing the stable V2.2 release. Changed bytes require a new candidate and fresh applicable evidence.

The RC is unsigned. Actual clean-machine/offline use, assistive speech, physical-phone use, and researcher-pilot observations remain separate from these automated checks. Independent builds may have different Web build IDs and installer hashes.
