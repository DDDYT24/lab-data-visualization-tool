# LabViz Windows packaging contract

## V2.2.0 public release — current scope

[Download the unsigned Windows x64 installer](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0).
The source/API/Web/installer version is 2.2.0. The supported scope is one ordinary Windows
profile; real two-account isolation remains open and is not claimed. Owner-reported
clean/offline/reboot acceptance of the preceding candidate is recorded separately from fresh
final-build automation and its hash. Release assets contain source/build provenance, SHA-256,
test results and local scan scope. No signing certificate or universal antivirus approval.

User install/update/history/backup instructions are in the complete
[English README](../../../README.md) / [中文 README](../../../README.zh-CN.md).
The P5-2 prototype and development-candidate references below describe historical build
checkpoints; they do not change this current release status. Keep the default short path.



This directory contains the reproducible Inno Setup source and packaging/lifecycle scripts.
The unsigned V2.2.0 installer and bundled runtimes are distributed as GitHub Release assets,
not committed binaries. Owner acceptance and exact-artifact automation have separate scopes.

## Historical prototype specification and reproducible build guidance

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

The developer launcher remains available alongside the staged Windows test bundle:

```powershell
Set-Location -LiteralPath 'C:\path\to\lab-data-visualization-tool'
.\start-labviz.cmd
```

The first developer launch may download dependencies locally. That behavior is not evidence of a
cloud deployment and is intentionally separate from the future offline installer.

## Upgrade, rollback, and uninstall rules

An upgrade must leave `%LOCALAPPDATA%\LabViz\data` in place and keep the previous program version
until the new API and web health check passes. Activation snapshots SQLite and local files; a
rollback requires the corresponding pre-upgrade data snapshot and retains displaced data. The
uninstaller must present an explicit “keep local data” or “delete local data” choice and default to
keeping it. The package must support paths with spaces and non-ASCII characters and must not require
administrator privileges when installed per-user. An import copies guest-owned projects and local
objects without changing the source; it clears old browser sessions and transient authentication
state, so a user may need to sign in again. An explicit data deletion also removes retained and
failed-transaction data copies.

## Verification status

`package-manifest.json` is the machine-readable contract. `stage-candidate.ps1` assembles a staged
portable candidate from an already-built Next.js standalone tree and explicit runtime directories;
it copies both `.next/static` and the app's `public` assets (including the bilingual About content
and its 2D/3D SVG figures). `validate-package.ps1` then checks these required launch assets, metadata,
and prohibited user-data/runtime artifacts. The standalone web server includes its traced production dependency tree under
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

When building Next.js in an isolated web copy to preserve a modified checkout, pass that copy's
`V2.0\web` directory with `-WebBuildRoot`. The staged API, packaging scripts and README still come
from this repository; the provided web directory must already contain the standalone `.next` build.

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
installer builder. Inno Setup is the selected test installer technology. The current Windows
candidate bundles Python 3.13.7 and Node.js 24.17.0 and uses `-RequirePython313`. Staging omits
the Python `Doc`, `include`, and standard-library `Lib/test` trees because the application does
not need developer documentation, headers, or interpreter tests at runtime. The package
validator rejects these trees if they reappear. Clean-machine lifecycle tests remain open.

After the bundled runtime exists, add `-RequireBundledRuntimes`. The portable launcher performs
both web and API loopback health checks before opening the browser; automated smoke tests may add
`-SkipOpenBrowser`. It keeps user data below
`%LOCALAPPDATA%\LabViz`.

### Local test installer

`LabViz.iss` installs each candidate under a versioned directory below
`%LOCALAPPDATA%\Programs\LabViz\versions\<version>`. The stable launcher records the active
version. Installation writes a pending marker; the launcher snapshots data and checks both
services before activating it. `set-labviz-version.ps1` requests validation of an installed version.
Downgrade requires the matching snapshot; rollback retains the newer data tree separately. The
application data remains under `%LOCALAPPDATA%\LabViz`; uninstall asks whether to keep it and
defaults to keeping it.

Compile a disposable test installer from the packaging directory after staging a candidate:

```powershell
& 'C:\Users\<user>\AppData\Local\Programs\Inno Setup 6\ISCC.exe' `
  '/DCandidateRoot=C:\path\to\staged\LabViz' `
  '/DOutputDir=C:\path\to\outputs\v22-installer' `
  .\LabViz.iss
```

The lifecycle harness accepts a second setup executable when testing an upgrade:

```powershell
.\test-installer.ps1 `
  -SetupExe 'C:\path\to\LabViz-Setup-2.2.0.exe' `
  -RepairSameVersion `
  -UpgradeSetupExe 'C:\path\to\LabViz-Setup-2.2.1.exe' `
  -DeleteDataSetupExe 'C:\path\to\LabViz-Setup-delete-test.exe' `
  -TestRoot 'C:\path with spaces\labviz-installer-test'
```

To verify a same-version replacement of an already-installed candidate (for example, a packaging
fix that keeps version `2.2.0`), pass the affected old installer as `-SetupExe` and the repaired
installer as `-SameVersionUpdateSetupExe`. **If LabViz is already installed for the current Windows
user, compile both disposable test installers with the same dedicated test-only `/DAppId=...`;
do not use the production AppId in this harness on that account.** `/NOICONS` prevents test
shortcuts but does not isolate uninstall registration. Use a dedicated empty `TestRoot`. The
harness checks the shared install path, About Markdown/SVG HTTP responses, absence of duplicate
version folders, and preservation of local data:

```powershell
.\test-installer.ps1 `
  -SetupExe 'C:\path\to\old\LabViz-Setup-2.2.0.exe' `
  -SameVersionUpdateSetupExe 'C:\path\to\fixed\LabViz-Setup-2.2.0.exe' `
  -TestRoot 'C:\path with spaces\labviz-same-version-update-test'
```

It checks installation and loopback health, coexistence of two versions, rollback,
repair/reinstall (including an optional same-version repair), and silent uninstall with local
data retained by default. The harness
also accepts a test-only installer compiled with `/DTestDeleteData=1` to exercise explicit
data deletion, including retained/failed transaction copies. The harness does not replace
clean-machine, disconnected, signing, antivirus, or real V2.1.1 upgrade evidence.

P5-1 (architecture and lifecycle contract) and the installer source are implemented here. An
unsigned 2.2.0 test installer compiled with Inno Setup 6.7.3 and passed isolated current-host
installation, Web/API health, same-version repair and default data-retaining uninstall. A
test-only delete-data variant from the same candidate passed explicit deletion of data, logs,
backups and retained/failed data copies. P5-2 remains open for clean/offline acceptance of the
exact user-facing installer. The user
waived authentic V2.1.1 upgrade/rollback for single-person use. Signing and antivirus review
apply before public distribution. macOS/Linux packages are not advertised by this contract.

## Current Windows launch experience

The default shortcuts use Windows PowerShell 5.1, with UTF-8 BOM for Chinese script messages.
Preferred ports are 3000/8000; occupied ports fall back to available loopback ports, and sharing
URLs/origin configuration follow the selected web port. The standalone web server resolves its API
proxy from the launcher's selected API port at request time, rather than baking port 8000 into the
build. The proxy accepts only HTTP loopback targets. A per-user mutex prevents duplicate writers.
The launcher prints readiness progress and uses a 90-second startup health deadline. Missing files,
timeouts and child failures are reported with the local log location. An OS job object closes the
API/Web process group if the launcher is forcibly terminated.

Start-menu entries provide **Stop LabViz**, **LabViz logs**, **Import old data** and **Rollback**.
Import prompts for the stopped old application's data directory and only accepts an empty target.
For V2.1.1 source installations, select V2.0/api/.labviz. Custom database/object locations and
browser guest-session recovery are not supported. Guest and account-owned project rows are
retained in the copied database, but remain inaccessible from the new account-free profile;
they are **not** silently assigned to the Windows user. Re-import the original files to create
new local projects. Keep the old checkout and source data until you have checked the new
projects and exports. Import into an empty destination only; the source and an integrity-checked
backup are retained. Stop both versions before importing, snapshotting, or restoring.

`languages/ChineseSimplified.isl` is vendored from the Inno Setup source repository's
[Simplified Chinese translation](https://raw.githubusercontent.com/jrsoftware/issrc/main/Files/Languages/ChineseSimplified.isl)
(retrieved 2026-09-23, SHA-256 `E0B0B350E2245F3C5E65586DFE43D574F6E7F06F2261149ABA284954B3FC9A8D`).
Its 281 message keys match the installed Inno Setup 6.7.3 `Default.isl`; LabViz-specific
installer prompts remain in `LabViz.iss`. Installer language visual review remains required.

A reproducible developer-machine launch and migration test (not a clean-machine installer test):

~~~powershell
.\test-launcher.ps1 -CandidateRoot 'C:\path\to\candidate' -TestRoot 'C:\new test folder' -ExperienceOnly
~~~

This creates disposable version wrappers and read-only-use runtime directory junctions to the
candidate. It verifies occupied ports, single-instance behavior, stop, forced termination cleanup
and recovery using Windows PowerShell 5.1. Keep the test directory under ignored outputs.
The full mode additionally checks a synthetic V2.1.1-style source with a guest project, an old
browser session, and an object file: the source remains unchanged, the guest project is present
after import, the old session is absent, and the imported database can be reopened by the API.

To run the packaged Chromium real-API examples and local-history flows against a disposable
per-user data directory, with the checkout's installed Playwright browsers:

~~~powershell
.\test-live-candidate.ps1 -CandidateRoot 'C:\path\to\candidate' -TestRoot 'C:\new empty test folder'
~~~

The harness uses the same per-user DPAPI local-access bootstrap as the launcher and removes its
owned processes on exit. It does not send data off the machine or validate a clean Windows host.
