# Shared Windows packaging inputs (formerly the V2.3 RC build)

The V2.3 RC build that lived here is retired: V2.3.0 and later ship the desktop installer from
`desktop/`. Its CI workflow, `build_rc.py`, and the upgrade check from the old RC
(`tests/verify_previous_upgrade.py`, `previous-release.json`) were removed in V2.5 M1. Git history
and the `v2.3.0-rc.1` and `v2.3.0-rc.2` tags keep them. The dependency audit they ran is now
`.github/scripts/npm_audit_gate.py`, in CI's `frontend` job.

The desktop build still reads these files, so they stay:

| File | Read by |
| --- | --- |
| `stage_candidate.py` | `desktop/scripts/stage_d34.py` and `desktop/scripts/stage_comparison.py` (payload staging); its hash is part of the stage provenance |
| `bin/` | copied into the payload by the same scripts; hashed by `desktop/packaging/provenance.py` |
| `runtimes.json` | `desktop/packaging/build_candidate.py` (pinned Python and Node.js runtimes) |
| `identity.json` | copied by `stage_candidate.py`; its data epoch is checked by `V2.0/api/tests/test_installed_v23_lifecycle.py` |

The other files (`LabVizV23.iss`, `compile_candidate.py`, `release_assets.py`,
`verify_components.py`, `stage-candidate.ps1`, `collect-web-licenses.mjs` and `tests/`) belonged
to the RC build. Moving the shared files under `desktop/packaging` and removing the rest is a
later clean-up.
