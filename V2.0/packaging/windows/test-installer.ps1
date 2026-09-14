[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SetupExe,
    [Parameter(Mandatory = $true)]
    [string]$TestRoot,
    [string]$UpgradeSetupExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$setupPath = (Resolve-Path -LiteralPath $SetupExe).Path
$testRootAbsolute = [IO.Path]::GetFullPath($TestRoot)
if (Test-Path -LiteralPath $testRootAbsolute) {
    if (@(Get-ChildItem -LiteralPath $testRootAbsolute -Force).Count -gt 0) {
        throw "TestRoot must be new or empty: $testRootAbsolute"
    }
} else {
    New-Item -ItemType Directory -Force -Path $testRootAbsolute | Out-Null
}

$installRoot = Join-Path $testRootAbsolute "LabViz"
$installArguments = @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    ('/DIR="{0}"' -f $installRoot)
)
$installer = Start-Process -FilePath $setupPath -ArgumentList $installArguments -Wait -PassThru
if ($installer.ExitCode -ne 0) {
    throw "Installer returned exit code $($installer.ExitCode)."
}

$required = @(
    "current-version.txt",
    "unins000.exe",
    "bin\start-labviz-installed.cmd",
    "bin\set-labviz-version.ps1",
    "versions\2.2.0\runtime\python\python.exe",
    "versions\2.2.0\runtime\node\node.exe",
    "versions\2.2.0\V2.0\assets\fonts\NotoSansSC-VF.ttf"
)
foreach ($relativePath in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $installRoot $relativePath) -PathType Leaf)) {
        throw "Installed package is missing: $relativePath"
    }
}
if ((Get-Content -LiteralPath (Join-Path $installRoot "current-version.txt") -Raw).Trim() -ne "2.2.0") {
    throw "The installed version marker is incorrect."
}

$oldLocalAppData = $env:LOCALAPPDATA
$localAppData = Join-Path $testRootAbsolute "LocalAppData"
New-Item -ItemType Directory -Force -Path $localAppData | Out-Null
$env:LOCALAPPDATA = $localAppData

function Assert-InstalledHealth {
    param(
        [int]$WebPort,
        [int]$ApiPort,
        [string]$Label
    )

    $launcher = Join-Path $installRoot "bin\start-labviz-installed.ps1"
    $launcherArguments = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        ('"{0}"' -f $launcher),
        "-WebPort",
        "$WebPort",
        "-ApiPort",
        "$ApiPort",
        "-SkipOpenBrowser"
    )
    $launcherProcess = $null
    try {
        $launcherProcess = Start-Process -FilePath (Join-Path $PSHOME "pwsh.exe") `
            -ArgumentList $launcherArguments -WorkingDirectory $installRoot `
            -WindowStyle Hidden -PassThru
        $deadline = (Get-Date).AddSeconds(60)
        $webReady = $false
        $apiReady = $false
        do {
            Start-Sleep -Seconds 1
            try {
                $web = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort/" -TimeoutSec 2
                $webReady = $web.StatusCode -eq 200
            } catch {
                $webReady = $false
            }
            try {
                $api = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
                $apiReady = $api.StatusCode -eq 200
            } catch {
                $apiReady = $false
            }
            $launcherProcess.Refresh()
            if ($launcherProcess.HasExited -and -not ($webReady -and $apiReady)) {
                throw "$Label exited with code $($launcherProcess.ExitCode)."
            }
        } while (-not ($webReady -and $apiReady) -and (Get-Date) -lt $deadline)
        if (-not ($webReady -and $apiReady)) {
            throw "$Label health checks timed out."
        }
        Write-Host "$Label passed: Web $($web.StatusCode), API $($api.StatusCode)" -ForegroundColor Green
    } finally {
        if ($launcherProcess -and -not $launcherProcess.HasExited) {
            Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
        }
        foreach ($port in @($WebPort, $ApiPort)) {
            $owners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique)
            foreach ($owner in $owners) {
                Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Assert-InstalledHealth -WebPort 3340 -ApiPort 8340 -Label "Installed launcher"

$dataRoot = Join-Path $localAppData "LabViz\data"
$logRoot = Join-Path $localAppData "LabViz\logs"
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
Set-Content -LiteralPath (Join-Path $dataRoot "keep-me.txt") -Value "local test data" -Encoding utf8
if (-not (Test-Path -LiteralPath $logRoot -PathType Container)) {
    throw "Installed launcher did not create the local log directory."
}

if ($UpgradeSetupExe) {
    $upgradePath = (Resolve-Path -LiteralPath $UpgradeSetupExe).Path
    $upgradeArguments = @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        ('/DIR="{0}"' -f $installRoot)
    )
    $upgrade = Start-Process -FilePath $upgradePath -ArgumentList $upgradeArguments -Wait -PassThru
    if ($upgrade.ExitCode -ne 0) {
        throw "Upgrade installer returned exit code $($upgrade.ExitCode)."
    }
    foreach ($relativePath in @(
        "versions\2.2.0\bin\start-labviz-portable.ps1",
        "versions\2.2.1\bin\start-labviz-portable.ps1"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $installRoot $relativePath) -PathType Leaf)) {
            throw "Upgrade did not retain both versioned program trees: $relativePath"
        }
    }
    if ((Get-Content -LiteralPath (Join-Path $installRoot "current-version.txt") -Raw).Trim() -ne "2.2.1") {
        throw "Upgrade did not select version 2.2.1."
    }

    $selector = Join-Path $installRoot "bin\set-labviz-version.ps1"
    & (Join-Path $PSHOME "pwsh.exe") -NoProfile -ExecutionPolicy Bypass `
        -File $selector -Version "2.2.0"
    if ((Get-Content -LiteralPath (Join-Path $installRoot "current-version.txt") -Raw).Trim() -ne "2.2.0") {
        throw "Rollback selector did not select version 2.2.0."
    }
    Assert-InstalledHealth -WebPort 3341 -ApiPort 8341 -Label "Rollback launcher"

    & (Join-Path $PSHOME "pwsh.exe") -NoProfile -ExecutionPolicy Bypass `
        -File $selector -Version "2.2.1"
    $repair = Start-Process -FilePath $upgradePath -ArgumentList $upgradeArguments -Wait -PassThru
    if ($repair.ExitCode -ne 0) {
        throw "Repair/reinstall returned exit code $($repair.ExitCode)."
    }
    if ((Get-Content -LiteralPath (Join-Path $installRoot "current-version.txt") -Raw).Trim() -ne "2.2.1") {
        throw "Repair/reinstall did not restore version 2.2.1."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $localAppData "LabViz\data\keep-me.txt") -PathType Leaf)) {
        throw "Upgrade or repair changed local data."
    }
    Write-Host "Upgrade, rollback, repair/reinstall passed." -ForegroundColor Green
}

try {
    $uninstaller = Join-Path $installRoot "unins000.exe"
    $uninstall = Start-Process -FilePath $uninstaller -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
    ) -Wait -PassThru
    if ($uninstall.ExitCode -ne 0) {
        throw "Uninstaller returned exit code $($uninstall.ExitCode)."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $localAppData "LabViz\data\keep-me.txt") -PathType Leaf)) {
        throw "Silent uninstall did not preserve local data by default."
    }
    if (Test-Path -LiteralPath $installRoot) {
        throw "Uninstaller left the program directory behind."
    }
    Write-Host "Uninstall passed: local data was retained by default." -ForegroundColor Green
} finally {
    $env:LOCALAPPDATA = $oldLocalAppData
}
