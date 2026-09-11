<div align="center">

# 🧪 Lab Data Visualization Tool

**Turn raw experiment tables into clear, exportable visuals — locally.**

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
![Release](https://img.shields.io/badge/release-V2.1.1%20local-0f766e)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [Version baseline](VERSION_BASELINE.md) · [Future backlog](V2.0/TODO.md) · [Privacy boundary](V2.0/docs/PRIVACY_DATA_BOUNDARY.md) · [Offline install](V2.0/docs/OFFLINE_INSTALL.md)

</div>

LabViz is a local-first web application for importing, checking, cleaning,
visualizing, and exporting experimental data. V2.1.1 combines a Next.js website
with a FastAPI processing service. The default setup stores data on the user's
computer and does not require a hosted website, AWS, Docker, PostgreSQL, or MinIO.

## Quick start — V2.1.1

Install the three tools below. Click a link to open the official download page:

| Tool | Windows | macOS / Linux |
| --- | --- | --- |
| Git | [Download Git for Windows](https://git-scm.com/download/win) | [Git downloads](https://git-scm.com/downloads) |
| Python | [Python 3.12 or 3.13](https://www.python.org/downloads/windows/) | [Python downloads](https://www.python.org/downloads/) |
| Node.js | [Node.js 22.22.2 or newer](https://nodejs.org/en/download) | [Node.js 22.22.2 or newer](https://nodejs.org/en/download) |

### Windows

Open PowerShell in the folder where you want the project, then run:

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
.\start-labviz.cmd
```

If the repository is already open in PowerShell, skip the first two lines. The
launcher creates `V2.0/api/.venv`, installs the local Python and website
dependencies on the first run, and starts both services. Then open
`http://127.0.0.1:3000`.

### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
chmod +x start-labviz.sh
./start-labviz.sh
```

Open `http://127.0.0.1:3000`. Keep the terminal open; local sign-in codes are
printed by the API process. Press `Ctrl+C` to stop both services.

After dependency files change, refresh them with:

```powershell
.\start-labviz.cmd -RefreshDependencies
```

```bash
./start-labviz.sh --refresh-dependencies
```

## What the local app stores

The default local mode uses SQLite and a local object directory. Both are
created automatically under `V2.0/api/.labviz/`; SQLite comes from Python's
standard library. Parsed snapshots, project history, quality results, and
exports remain on the same computer. The default launcher binds the website and
API to `127.0.0.1`.

The normal local workflow does not need an account, email service, database
server, or cloud storage. PostgreSQL, MinIO/S3, Docker, and AWS files remain in
the repository for maintainers who choose a larger self-hosted or integration
testing setup.

## Main workflow

```text
Import data → Inspect quality → Confirm cleaning → Create chart → Export
```

Supported inputs include CSV, TSV, delimited TXT, JSON, and XLSX. The web app
provides quality summaries, safe cleaning decisions, seven 2D/3D chart types,
project history, local sign-in, read-only sharing, and PNG/SVG/PDF exports.

V1.1 remains available as a Python-only Streamlit application for users who need
the smaller legacy interface: [`V1.1/`](V1.1/).

## Supported environment

| Area | Supported and verified |
| --- | --- |
| Operating systems | Windows, macOS, and Linux; CI verifies Ubuntu |
| Python | 3.12 verified by CI; 3.13 supported; 3.14 is not supported yet |
| Node.js | 22.22.2 minimum; Node.js 24 verified by CI |
| Hardware | CPU only; no GPU or external database server required |
| Browser | A current modern browser; website port `3000`, API port `8000` |

## Manual startup and development

The launcher is the recommended way to run LabViz. If you need separate API
and website terminals, follow the commands in [`V2.0/api/README.md`](V2.0/api/README.md)
and keep the website environment variable `NEXT_TELEMETRY_DISABLED=1`.

The frontend checks are run from `V2.0/web`:

```bash
npm ci
npm run verify
npm run test:e2e
```

The full API integration suite is run from `V2.0/api` and needs the PostgreSQL
and MinIO Compose services. The complete command set is documented in
[`V2.0/docs/PUBLIC_RELEASE_CHECKLIST.md`](V2.0/docs/PUBLIC_RELEASE_CHECKLIST.md).

## Troubleshooting

- If `Set-Location` fails, run it from the parent directory of the clone, or
  skip it when PowerShell is already in the repository.
- If Python is not found, install Python 3.12 or 3.13 and reopen PowerShell.
- If Node.js is too old, install 22.22.2 or newer and reopen PowerShell.
- If port `3000` or `8000` is busy, stop the process using it and run the launcher again.
- If dependencies changed, use the refresh command shown above.

## Project layout

| Path | Purpose |
| --- | --- |
| `V2.0/web/` | Next.js frontend and browser tests |
| `V2.0/api/` | FastAPI processing service, SQLite persistence, and API tests |
| `V1.1/` | Legacy Streamlit application |
| `V2.0/docs/` | Privacy, offline installation, backup, and release guidance |
| `VERSION_BASELINE.md` | What V1.0, V1.1, V2.0, and V2.1 delivered |
| `V2.0/TODO.md` | Single source of truth for future work and status |

## License

LabViz is released under the [MIT License](LICENSE).

For Chinese installation instructions, see [`README.zh-CN.md`](README.zh-CN.md).
