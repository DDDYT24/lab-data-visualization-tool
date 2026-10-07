[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CandidateRoot,
    [Parameter(Mandatory = $true)][string]$TestRoot,
    [int]$WebPort = 3381,
    [int]$ApiPort = 8381,
    [string[]]$TestFiles = @('e2e/live-samples.spec.ts', 'e2e/live-local-history.spec.ts')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$candidate = (Resolve-Path -LiteralPath $CandidateRoot).Path
$testRootAbsolute = [IO.Path]::GetFullPath($TestRoot)
if (Test-Path -LiteralPath $testRootAbsolute) {
    if (@(Get-ChildItem -LiteralPath $testRootAbsolute -Force).Count -gt 0) {
        throw "TestRoot must be new or empty: $testRootAbsolute"
    }
} else {
    New-Item -ItemType Directory -Force -Path $testRootAbsolute | Out-Null
}

$launcher = Join-Path $candidate 'bin\start-labviz-portable.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Candidate launcher is missing: $launcher"
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$webRoot = Join-Path $repoRoot 'V2.0\web'
$playwright = Join-Path $webRoot 'node_modules\.bin\playwright.cmd'
if (-not (Test-Path -LiteralPath $playwright -PathType Leaf)) {
    throw "Install the checkout's Web test dependencies first: $playwright"
}

$oldLocalAppData = $env:LOCALAPPDATA
$oldBaseUrl = $env:PLAYWRIGHT_BASE_URL
$oldLive = $env:LABVIZ_E2E_LIVE
$oldTestKey = $env:LABVIZ_E2E_LOCAL_ACCESS_KEY
$oldBrowsersPath = $env:PLAYWRIGHT_BROWSERS_PATH
$launcherProcess = $null
$localAppData = Join-Path $testRootAbsolute '本地数据 space'
if ([string]::IsNullOrWhiteSpace($oldBrowsersPath)) {
    $browserCache = Join-Path $oldLocalAppData 'ms-playwright'
    if (Test-Path -LiteralPath $browserCache -PathType Container) {
        $env:PLAYWRIGHT_BROWSERS_PATH = $browserCache
    }
}
$env:LOCALAPPDATA = $localAppData
New-Item -ItemType Directory -Force -Path $localAppData | Out-Null

try {
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $launcher),
        '-WebPort', "$WebPort", '-ApiPort', "$ApiPort", '-SkipOpenBrowser'
    )
    $launcherProcess = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') `
        -ArgumentList $arguments -WorkingDirectory $candidate -WindowStyle Hidden -PassThru

    $readyFile = Join-Path $localAppData 'LabViz\logs\running.json'
    $tokenFile = Join-Path $localAppData 'LabViz\data\local-access.dpapi'
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        $launcherProcess.Refresh()
        if ($launcherProcess.HasExited) {
            throw "Candidate launcher exited with code $($launcherProcess.ExitCode)."
        }
        if ((Test-Path -LiteralPath $readyFile -PathType Leaf) -and
            (Test-Path -LiteralPath $tokenFile -PathType Leaf)) {
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path -LiteralPath $readyFile -PathType Leaf)) {
        throw "Candidate did not become ready; inspect logs under $localAppData"
    }

    $ready = Get-Content -LiteralPath $readyFile -Raw | ConvertFrom-Json
    if ([int]$ready.launcherPid -ne $launcherProcess.Id -or [int]$ready.apiPort -ne $ApiPort) {
        throw 'Ready marker does not identify this candidate launch.'
    }
    Add-Type -AssemblyName System.Security
    $protected = [IO.File]::ReadAllBytes($tokenFile)
    $unprotected = [Security.Cryptography.ProtectedData]::Unprotect(
        $protected, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    $env:LABVIZ_E2E_LOCAL_ACCESS_KEY = [Convert]::ToBase64String($unprotected).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    [Array]::Clear($unprotected, 0, $unprotected.Length)
    $env:PLAYWRIGHT_BASE_URL = [string]$ready.webUrl
    $env:LABVIZ_E2E_LIVE = '1'

    Push-Location $webRoot
    try {
        & $playwright test @TestFiles `
            --project=chromium --workers=1 --reporter=line
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged real-API browser checks failed with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
    Write-Host 'Packaged real-API examples, exports, and local history passed.' -ForegroundColor Green
} finally {
    if ($launcherProcess -and -not $launcherProcess.HasExited) {
        $stopFile = Join-Path $localAppData 'LabViz\logs\stop.request'
        [IO.File]::WriteAllText($stopFile, 'stop')
        if (-not $launcherProcess.WaitForExit(60000)) {
            Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
            throw 'Candidate launcher did not stop gracefully.'
        }
    }
    $env:LOCALAPPDATA = $oldLocalAppData
    foreach ($entry in @(
        @{ Name = 'PLAYWRIGHT_BASE_URL'; Value = $oldBaseUrl },
        @{ Name = 'PLAYWRIGHT_BROWSERS_PATH'; Value = $oldBrowsersPath },
        @{ Name = 'LABVIZ_E2E_LIVE'; Value = $oldLive },
        @{ Name = 'LABVIZ_E2E_LOCAL_ACCESS_KEY'; Value = $oldTestKey }
    )) {
        if ($null -eq $entry.Value) {
            Remove-Item -LiteralPath "Env:$($entry.Name)" -ErrorAction SilentlyContinue
        } else {
            Set-Item -LiteralPath "Env:$($entry.Name)" -Value $entry.Value
        }
    }
}
