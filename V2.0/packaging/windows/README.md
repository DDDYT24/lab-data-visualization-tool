# LabViz Windows packaging contract

This directory defines the V2.2 local-first Windows packaging contract. It is an architecture and
validation aid, not a released installer. The repository currently does not contain bundled Python
or Node runtimes, a signed installer, or clean-machine installation evidence.

## Target architecture

The first native package is a per-user Windows installer. It will install the immutable application
under `%LOCALAPPDATA%\Programs\LabViz` and keep user-owned state outside that directory:

- data and SQLite objects: `%LOCALAPPDATA%\LabViz\data`
- logs and crash diagnostics: `%LOCALAPPDATA%\LabViz\logs`
- user-created backups: `%LOCALAPPDATA%\LabViz\backups`

The package will contain a pinned CPython 3.13.x API runtime, the locked Python wheels, a Node.js
22.22.2-or-newer runtime, and the Next.js standalone web server. It will launch both processes on
`127.0.0.1` (web `3000`, API `8000` by default), run a local health check, and open the browser.
The packaged launch path must not run `git`, create a virtual environment, run `pip install`, or
run `npm ci` on the user's first launch.

The current developer launcher remains the supported V2.2.0-dev path until that bundle exists:

```powershell
Set-Location -LiteralPath 'C:\path\to\lab-data-visualization-tool'
.\start-labviz.cmd
```

The first developer launch may download dependencies locally. That behavior is not evidence of a
cloud deployment and is intentionally separate from the future offline installer.

## Upgrade, rollback, and uninstall rules

An upgrade must leave `%LOCALAPPDATA%\LabViz\data` in place and keep the previous program version
until the new API and web health check passes. A rollback changes only the program directory. The
uninstaller must present an explicit “keep local data” or “delete local data” choice and default to
keeping it. The package must support paths with spaces and non-ASCII characters and must not require
administrator privileges when installed per-user.

## Verification status

`package-manifest.json` is the machine-readable contract. `validate-package.ps1` checks a candidate
package for required launch metadata and prohibited user-data/runtime artifacts. Pass
`-RequireBundledRuntimes` only after the actual runtime bundle has been assembled. The icon source
files are reusable: `V2.0/assets/labviz-logo.svg` is the vector source,
`V2.0/assets/labviz-logo.png` is the existing application source, and
`V2.0/assets/labviz-logo.ico` is generated from the PNG for the Windows shell.

To validate a staged candidate package, run this from the repository's packaging directory. The
candidate root must contain the paths listed by `package-manifest.json`; the repository checkout
itself is not a candidate package:

```powershell
Set-Location -LiteralPath 'C:\path\to\lab-data-visualization-tool\V2.0\packaging\windows'
.\validate-package.ps1 -PackageRoot 'C:\path\to\staged\LabViz'
```

After the bundled runtime exists, add `-RequireBundledRuntimes`. The portable launcher performs
both web and API loopback health checks before opening the browser and keeps user data below
`%LOCALAPPDATA%\LabViz`.

P5-1 (architecture and lifecycle contract) is implemented here. P5-2 (a real installer and clean
machine tests) remains open until a packager, bundled runtimes, signing decision, and clean Windows
test machine are available. macOS/Linux packages are not advertised by this contract.
