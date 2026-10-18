# LabViz

Import experimental tables, inspect data quality, and create traceable figures on your computer.

[English](README.md) · [简体中文](README.zh-CN.md) · [Stable V2.2.0](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [V2.3 RC notes](V2.0/RELEASE_NOTES_V2.3.md) · [MIT license](LICENSE)

## V2.3 preview

This branch contains V2.3.0-rc.1. V2.2.0 remains the stable release. The Windows RC is unsigned and has explicit remaining acceptance limits; use copies or synthetic data while evaluating it.

The local research workspace adds sample-aware preparation, grouped raw-point, paired-change and violin charts, up to four independent 2D figure panels, reusable templates, and frozen research outputs. Classic remains the default interface; the overview interface is available in Settings. Existing chart families, local history, quality checks, cleaning decisions, and PNG/SVG/PDF exports are retained.

Preparation previews show the selected mappings and exclusions before a revision is applied. Samples, observations, and independent experimental units remain distinct. Frozen research outputs include source data and calculation/revision records. Advanced scientific methods outside the supported contracts are not inferred automatically.

## Windows installer

Download the V2.3 RC installer and `SHA256SUMS.txt` from [Releases](https://github.com/DDDYT24/lab-data-visualization-tool/releases). Compare the installer hash:

```powershell
Get-FileHash .\LabViz-V23-2.3.0-rc.1-Setup.exe -Algorithm SHA256
```

The installer includes Python, Node.js, scientific libraries, the website, fonts, and synthetic examples. Installed use does not require a separate Python, Node.js, Docker, database server, hosted account, or email login. Keep Windows security controls enabled when evaluating the unsigned RC.

Start **LabViz V2.3** from the Start menu. It opens a loopback browser interface. Stop it with **Stop LabViz V2.3** before backup or maintenance. V2.3 has separate program and data directories from V2.2. Importing old data is explicit and creates an independent copy; uninstall retains profiles.

See [installation, V2.2 import, backup, restore, and uninstall](V2.0/docs/INSTALL_V2.3.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Run published RC source

Source use needs Git, Python 3.12 or 3.13, and a compatible Node.js version. The Windows build pipeline uses Python 3.13 and pinned Node.js 24.22.0.

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
git checkout v2.3.0-rc.1
.\start-labviz.cmd
```

The Windows source launcher uses ports 3001/8001, private dependencies, and `V2.0/api/.labviz/v23-development`. Its first setup can download dependencies. Source setup is separate from the self-contained Windows installer.

macOS/Linux source launchers remain available for development. Native packages for those systems are outside this RC.

## Verify and rebuild

[Windows RC build instructions](V2.0/packaging/windows-v23/README.md) describe a clean-checkout pipeline for pinned inputs, component tests, actual bundled browser workflows, legacy import, installation, rollback, and data retention.

CI checks the existing application and frontend alongside PostgreSQL/MinIO integration. Windows verification covers the bundled Python 3.13 runtime and installer lifecycle. Release provenance identifies the source commit, build, installer hash, and observed verification scope. A fresh rebuild can have a different build ID and binary hash.

The source remains a Next.js/MUI frontend and a FastAPI scientific API. SQLite and local files are the default data stores. Docker and hosted persistence are optional development/deployment paths. Build the API image from the V2.0 context so contracts, examples, and fonts are included:

```sh
docker build -f V2.0/api/Dockerfile -t labviz-api:rc V2.0
```

## Remaining RC acceptance

Clean-machine use, a complete physically disconnected workflow, actual desktop 200% zoom and assistive speech, physical iPhone use, and a researcher pilot with 3 to 5 participants remain pending. Automated accessibility checks and browser emulation have narrower evidence scopes. This preview does not claim stable-release acceptance.

The reviewed production npm tree has no reported audit findings. Five development lint-chain findings remain documented in the [RC notes](V2.0/RELEASE_NOTES_V2.3.md). Report installation failures, data/recovery problems, and scientific discrepancies through [Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues), using synthetic examples and removing personal data from attachments.

## License

LabViz source is MIT licensed. Bundled third-party components retain their own licenses. Original instrument files should be archived separately from the application's project backups.
