<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="V2.0/assets/labviz-logo-dark.svg">
  <img src="V2.0/assets/labviz-logo.svg" alt="LabViz logo" width="320">
</picture>

# LabViz

**Scientific Data Visualization**

**Import experimental tables, inspect data quality, create figures, and keep your work locally.**

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v2.2.0-0f766e)](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [Download V2.2](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [Release notes](V2.0/RELEASE_NOTES_V2.2.md) · [Report an issue](https://github.com/DDDYT24/lab-data-visualization-tool/issues)

</div>

LabViz is a local web application for researchers, students, and teaching labs. Its Windows
V2.2 installer includes the website, Python processing service, and runtimes. You use it in a
browser; data processing and storage run on your own computer. No LabViz account or email
verification is required. Ordinary use needs no Python installation, Node.js, Docker,
PostgreSQL, S3, AWS account, or hosted website.

## Install and start — Windows users

1. Download [LabViz-Setup-2.2.0.exe](https://github.com/DDDYT24/lab-data-visualization-tool/releases/download/v2.2.0/LabViz-Setup-2.2.0.exe) from the official release. Target: **Windows x64**.
2. Optionally check the file against [SHA256SUMS.txt](https://github.com/DDDYT24/lab-data-visualization-tool/releases/download/v2.2.0/SHA256SUMS.txt):

   ```powershell
   Get-FileHash -LiteralPath '.\LabViz-Setup-2.2.0.exe' -Algorithm SHA256
   ```

3. Run the installer as your normal Windows user, choose English or Simplified Chinese,
   and keep the default installation directory. Administrator privileges are not required.
4. Use the **LabViz** desktop/Start Menu shortcut. It starts the local services and opens an
   authorized browser page. **There is no sign-in step.** Do not use a stale bookmark as the first launch.
5. Import your table or try a bundled synthetic example. The installer and normal workflow
   work offline once the EXE has been downloaded or transferred to the computer.
6. Use **Stop LabViz / 退出** in the Start Menu when you want to stop the background services.

The public installer is **unsigned**. Windows may display a publisher/SmartScreen warning;
verify its origin and SHA-256 and follow your institution's software policy. Signing and
universal antivirus/SmartScreen approval are not claimed. See the release's verification
report for the actual local scan result. Do not bypass a school or employer's installation policy.

### Updating an existing installation

Stop LabViz, run the new EXE using the **same Windows account and default program directory**,
then use the existing shortcut. The installer preserves the separate data directory by default;
an update or same-version repair does not intentionally clear history. Check History after
updating and keep an offline backup before important changes. You do not need to delete the
old version first. Uninstalling asks whether to keep or permanently delete data; choose **Keep**
if you want to reinstall with history retained. Authentic V2.1.1 installer upgrade/rollback
acceptance was waived for this single-user release; this is not a guarantee for every old build.

## Typical research workflow

```text
CSV / TSV / TXT / JSON / XLSX
            ↓
Parse and validate → Inspect quality → Confirm cleaning
            ↓
Choose chart, fields, statistics, labels and appearance
            ↓
Preview → PNG / SVG / PDF export
            ↘ Local project history, processed data and saved figures
```

| Capability | What you can do |
| --- | --- |
| Import | Read CSV, TSV, delimited TXT, JSON and XLSX; select workbook sheets/header rows where applicable |
| Quality review | Inspect missing/non-numeric values, duplicates and grid problems; review proposed cleaning before applying it |
| Charts | Line, scatter, bar, histogram, box plot, heatmap and structured 3D surface |
| Scientific controls | Choose X/response/group fields, fitting, uncertainty bands, axis ranges/units, titles, fonts, colors and legend options |
| Statistics | Ordinary/weighted fitting, residual diagnostics, documented prediction intervals, Working–Hotelling mean bands and linear Huber robust fitting |
| Export | Download PNG/SVG/PDF with configurable size and resolution, and cleaned CSV data |
| Local history | Automatically retain real imports; reopen/edit projects, review processed data, view/download saved figures and explicitly delete work |
| Usability | English/Chinese UI, light/dark/system theme, accessible chart data tables, keyboard/accessibility checks and mobile 3D controls |
| Examples | Seven versioned synthetic teaching datasets and a bilingual Help/About page |

Real imports are saved after successful processing. Bundled examples are temporary by default;
they are teaching data, not experimental evidence. Group comparisons can use categorical X;
single-value distributions use the measurement field. Histogram interval boundaries and numeric
axis ranges are shared between the browser preview and Python exports. Export verifies figure
content as well as successful downloading; browser styling and Python typography need not be
pixel-identical.

### Figures for papers

LabViz reduces repeated spreadsheet setup by keeping the import → quality → analysis → figure
workflow together, supporting structured 3D data, explicit statistical assumptions and reusable
project history. Excel remains useful for spreadsheet calculations and manual table editing.
LabViz is not a replacement for reviewing your experimental design or a complete statistical package.

PNG, SVG and PDF figures **can be used in a manuscript when they meet the journal's requirements**.
Check final dimensions/DPI, accepted vector formats, font embedding, readable labels, color/accessibility,
statistical assumptions, exclusions and captions. Confirm the exported file itself before submission.
The software does not certify scientific validity, peer review, or compliance with every publisher.
See the [statistics contract](V2.0/docs/STATISTICS_CONTRACT_V2.2.md) for the exact supported models
and limits; Huber fitting is linear-only and does not offer confidence bands.

## Architecture and technology

```text
Browser (Next.js / React / TypeScript / MUI / ECharts)
    → local Next.js API proxy → FastAPI / Python
    → parsing, quality, cleaning, numerical analysis and Matplotlib export
    → SQLite metadata + local files (processed objects / figures)
```

| Layer | V2.2 local implementation |
| --- | --- |
| Website | Next.js 16.3.6, React 19, TypeScript, MUI, TanStack Query, Zustand, ECharts/ECharts GL |
| API / science | FastAPI; pandas, NumPy, SciPy, Matplotlib, PyArrow and workbook readers |
| Persistence | SQLite for project/processing metadata; local filesystem for objects and figures |
| Packaged runtime | CPython 3.13.7 and Node.js 24.17.0, bundled in the Windows installer |
| Installer | Inno Setup, per-user installation, PowerShell launcher, local-session bootstrap and lifecycle/recovery scripts |
| Verification | pytest, Ruff, MyPy, Vitest, ESLint, TypeScript and Playwright |
| Optional integration route | PostgreSQL and S3-compatible/MinIO adapters, worker and container/infrastructure code |

FastAPI remains the backend. SQLite replaces the need for a database **server** in the local
deployment; local file storage performs the object-storage role. PostgreSQL and S3-compatible
adapters remain in the repository for integration testing and future shared deployments.
**Cloud collaboration, hosted sync, team accounts and public cloud operation are deferred**;
their presence in source is not a deployed or released cloud service.

## Data, privacy, backup and migration

| Installed Windows location | Purpose |
| --- | --- |
| `%LOCALAPPDATA%\Programs\LabViz` | Program and version directories |
| `%LOCALAPPDATA%\LabViz\data` | SQLite database, local objects and local-access credential |
| `%LOCALAPPDATA%\LabViz\logs` | Startup and maintenance diagnostics |
| `%LOCALAPPDATA%\LabViz\backups` | Managed maintenance/upgrade backups |

The Windows **source checkout** instead uses `V2.0/api/.labviz`; installing the EXE does not
automatically import that history. With both copies stopped, use **Import old data / 迁移旧数据**
to import a compatible old data directory into an empty installed destination. The source and an
integrity-checked backup are retained. Old guest/email-account records are not silently reassigned
to the new local profile: re-import their original files if they are not accessible in local history.

For backup, stop LabViz and copy the **entire data directory**, not just the SQLite file. Database
records and object files belong together. Keep original experimental files separately; the project
store is not an archival copy of every original upload. Deleting local projects is permanent and
requires a prior backup to undo. See [backup and restore](V2.0/docs/BACKUP_RESTORE.md).

Normal use binds services to `127.0.0.1` and performs processing locally. Opening external help,
GitHub or email links is your explicit action. There is no automatic cloud sync. This release's
supported scope is **one ordinary Windows profile**. Two real Windows accounts' filesystem and
concurrent loopback isolation have **not** been validated; do not claim that privacy guarantee or
treat this package as a shared multi-user service. Do not expose its ports publicly. See
[privacy boundaries](V2.0/docs/PRIVACY_DATA_BOUNDARY.md) and [security policy](SECURITY.md).

## Verification and release scope

The 2026-10-07 full source/candidate audit passed **271 API tests**, **60 Vitest tests**,
**65 browser regressions**, ten distinct packaged real-API workflows, a **39-case fixture matrix**,
**100/100 chaos cases**, and three enforced performance runs. PostgreSQL/MinIO were real isolated
integration services; they are not user prerequisites. Installer/launcher lifecycle checks used
disposable data and a test AppId, not the personal production installation.

The owner reported completion of the preceding candidate's clean/offline/restart acceptance and
authorized this single-user release. That testimony is separate from agent-observed checks and
does not provide an independently measured hash attestation for the final EXE. The final release
rebuild changes version/publication metadata and documentation; its exact commit, build ID, EXE
hash and fresh packaged verification are in the release assets. Historical failed runs and their
repairs are retained in the [P7 report](V2.0/docs/V2.2_P7_TEST_REPORT.md).

Production dependency findings were patched/reviewed; seven development-tool findings remain
documented. An Arrow exception applies only to the reviewed Python/Parquet path. This is not a
claim that all dependencies are vulnerability-free. See the
[dependency security review](V2.0/docs/DEPENDENCY_SECURITY_REVIEW.md).

## Development from source

The code remains under `V2.0/` for compatibility with existing paths. Source setup needs Git,
Python **3.12 or 3.13**, and Node.js **22.22.2+** (24 used in verification). Python 3.14 is not
supported. First setup downloads dependencies; this differs from the ready-to-run Windows EXE.

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
.\start-labviz.cmd
# After dependency files change:
.\start-labviz.cmd -RefreshDependencies
```

On macOS/Linux, `chmod +x start-labviz.sh && ./start-labviz.sh` runs the source application.
The Unix source launcher retains its existing session behavior; Windows account-free launcher
parity and native macOS/Linux installers are not part of this release. For separate API/web
terminals and full integration commands, read [API development](V2.0/api/README.md).

```powershell
Set-Location -LiteralPath .\V2.0\web
$env:NEXT_TELEMETRY_DISABLED = '1'
npm ci
npm run verify
npx playwright install chromium firefox webkit
npm run test:e2e
```

Default browser tests include mocked API checks; live-service tests are opt-in. Use
[`test-live-candidate.ps1`](V2.0/packaging/windows/test-live-candidate.ps1) for a disposable
packaged real-API run and [Windows packaging](V2.0/packaging/windows/README.md) for rebuilding.
The complete API suite needs the documented isolated PostgreSQL/MinIO services; use the locked
development dependencies. Tests never need actual experimental data.

## Troubleshooting

- **Project data unavailable / local access denied:** stop and reopen from the LabViz shortcut;
  use the automatically opened page. The launcher's internal credential unlocks local history.
  If recovery fails, inspect `%LOCALAPPDATA%\LabViz\logs`; do not publish credentials or raw logs.
- **Ports busy:** the packaged launcher selects free loopback ports. Use the page it opens.
- **After sleep/reboot:** reopen the shortcut; stale browser tabs may refer to stopped services.
- **Wrong fields:** use a categorical X for group comparisons, a numeric response for measurements,
  and the recommended histogram/box chart for single-value data. Update to V2.2 if 04/05 still
  select identical line-chart X/response fields.
- **Export differs:** inspect the downloaded file, field roles, histogram bins, axis limits and
  units. Report a synthetic reproduction; preview and export are different rendering engines.
- **Installation/path errors:** keep the default short installation path. Arbitrarily long custom
  paths and every sleep/resume policy are not validated.
- **History missing after install:** check Windows account and data location; source/installed
  histories are separate. Use the migration instructions instead of deleting old data.
- **Source setup:** check `py -0p` and `node --version`, reopen the terminal after installing tools,
  and refresh dependencies when their files change.

## Repository and documentation

| Path | Contents |
| --- | --- |
| `V2.0/web/` | Frontend, local API proxy, unit and browser tests |
| `V2.0/api/` | Processing/rendering API, persistence/storage adapters and tests |
| `V2.0/contracts/` | Versioned rendering and local-storage contracts |
| `V2.0/api/samples/v22/` | Public synthetic examples and edge fixtures |
| `V2.0/packaging/windows/` | Installer source, staging, validation and lifecycle checks |
| `V2.0/docs/` | Privacy, offline, backup, statistics, security and acceptance evidence |
| `V1.1/` | Legacy Python/Streamlit application |

[Version history](VERSION_BASELINE.md) · [Changelog](CHANGELOG.md) ·
[Future work/status authority](V2.0/TODO.md) · [Offline guide](V2.0/docs/OFFLINE_INSTALL.md) ·
[Contributing](CONTRIBUTING.md) · [Public release checklist](V2.0/docs/PUBLIC_RELEASE_CHECKLIST.md)

## Feedback and license

Report reproducible issues through [GitHub Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues)
or the app's feedback link to [liyutao982@gmail.com](mailto:liyutao982@gmail.com). Remove private
data, local credentials and identifying paths before sharing diagnostics. Claims and release
evidence are documented by scope.

LabViz source is licensed under [MIT](LICENSE). Bundled third-party components retain their
own notices/licenses, including the [OFL CJK font](V2.0/assets/fonts/OFL.txt).
