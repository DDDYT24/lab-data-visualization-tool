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

`package-manifest.json` is the machine-readable contract. `stage-candidate.ps1` assembles a staged
portable candidate from an already-built Next.js standalone tree and explicit runtime directories;
`validate-package.ps1` then checks required launch metadata and prohibited user-data/runtime
artifacts. The standalone web server includes its traced production dependency tree under
`V2.0\web\node_modules`; development checkout dependencies elsewhere remain prohibited. Pass
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

To stage a local candidate after `npm run build` has produced `.next\standalone\web`, provide a
base CPython 3.12/3.13 directory, a clean target containing only the locked API dependencies, and
a Node.js 22.22.2-or-newer executable. Do not pass the development virtual environment directly:
it may contain pytest, build tools, or unrelated workspace dependencies. The output directory and
dependency target are disposable and must not contain user data:

```powershell
$py = 'C:\path\to\lab-data-visualization-tool\V2.0\api\.venv\Scripts\python.exe'
$pythonRoot = & $py -c "import sys; print(sys.base_prefix)"
$sitePackages = 'C:\path\to\staging\LabViz-api-site-packages'
& $py -m pip install --require-hashes --no-compile --target $sitePackages `
  -r 'C:\path\to\lab-data-visualization-tool\V2.0\api\requirements.lock.txt'
.\stage-candidate.ps1 `
  -OutputRoot 'C:\path\to\staged\LabViz' `
  -PythonRuntimeRoot $pythonRoot `
  -PythonSitePackagesRoot $sitePackages `
  -NodeExecutable 'C:\Program Files\nodejs\node.exe'
```

The staging script validates Python 3.12/3.13 and Node.js 22.22.2-or-newer, but it is not a native
installer builder. A release candidate still needs a selected installer technology, bundled-runtime
review, signing decision, and clean-machine lifecycle tests.

After the bundled runtime exists, add `-RequireBundledRuntimes`. The portable launcher performs
both web and API loopback health checks before opening the browser; automated smoke tests may add
`-SkipOpenBrowser`. It keeps user data below
`%LOCALAPPDATA%\LabViz`.

P5-1 (architecture and lifecycle contract) is implemented here. P5-2 (a real installer and clean
machine tests) remains open until a packager, bundled runtimes, signing decision, and clean Windows
test machine are available. macOS/Linux packages are not advertised by this contract.
