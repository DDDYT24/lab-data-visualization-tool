[CmdletBinding()]
param(
    [switch]$RefreshDependencies,
    [switch]$SkipOpenBrowser,
    [switch]$CheckConfiguration,
    [string]$DataDirectory,
    [ValidateRange(1, 65535)]
    [int]$WebPort = 3001,
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Security

$repoRoot = $PSScriptRoot
$apiRoot = Join-Path $repoRoot "V2.0\api"
$webRoot = Join-Path $repoRoot "V2.0\web"
$venvRoot = Join-Path $apiRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

# This workspace is exclusively for V2.3 source development.
if ($WebPort -eq $ApiPort -or $WebPort -in @(3000, 8000) -or $ApiPort -in @(3000, 8000)) {
    throw 'V2.3 requires distinct ports; 3000 and 8000 are reserved for V2.2.'
}
$developmentDataRoot = Join-Path $apiRoot '.labviz\v23-development'
if ($DataDirectory) {
    $restoredParent = [IO.Path]::GetFullPath((Join-Path $apiRoot '.labviz\v23-restored')).TrimEnd('\') + '\'
    $candidateDataRoot = [IO.Path]::GetFullPath($DataDirectory).TrimEnd('\')
    if (-not $candidateDataRoot.StartsWith($restoredParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Only an independently restored V2.3 source copy can be selected.'
    }
    $restoreRecordPath = Join-Path $candidateDataRoot 'restore-manifest.json'
    if (-not (Test-Path -LiteralPath $restoreRecordPath -PathType Leaf)) { throw 'Missing restore verification record.' }
    $restoreRecord = Get-Content -LiteralPath $restoreRecordPath -Raw | ConvertFrom-Json
    if ($restoreRecord.format -ne 'labviz-v23-restored-copy' -or $restoreRecord.schemaVersion -ne 1) {
        throw 'Unsupported restore verification record.'
    }
    Push-Location $apiRoot
    try {
    $candidateResolved = & $venvPython -X utf8 -c "import sys; from pathlib import Path; from labviz_api.source_data import check_database,safe_files; p=Path(sys.argv[1]); base=Path(sys.argv[2]).resolve(); assert p.resolve().is_relative_to(base); check_database(p/'labviz.db'); assert (p/'objects').is_dir(); safe_files(p/'objects'); print(str(p.resolve()))" $candidateDataRoot $restoredParent.TrimEnd("\")
    if ($LASTEXITCODE -ne 0) { throw 'Restored source data failed its startup integrity or path check.' }
    $developmentDataRoot = $candidateResolved
    } finally { Pop-Location }
}
$developmentEnvironment = @{
    LABVIZ_DEVELOPMENT_LINE = '2.3'
    LABVIZ_ENVIRONMENT = 'development'
    LABVIZ_RUNTIME_ROLE = 'api'
    LABVIZ_PERSISTENCE_BACKEND = 'sqlite'
    LABVIZ_DATABASE_PATH = Join-Path $developmentDataRoot 'labviz.db'
    LABVIZ_OBJECT_STORAGE_BACKEND = 'local'
    LABVIZ_OBJECT_STORAGE_ROOT = Join-Path $developmentDataRoot 'objects'
    LABVIZ_AUTH_MODE = 'console'
    LABVIZ_COOKIE_SECURE = 'false'
    LABVIZ_PUBLIC_WEB_URL = "http://127.0.0.1:$WebPort"
    LABVIZ_ALLOWED_ORIGINS = "http://127.0.0.1:$WebPort,http://localhost:$WebPort"
    NEXT_PUBLIC_LABVIZ_API_URL = '/api/v1'
}
if ($CheckConfiguration) {
    [ordered]@{
        developmentLine = '2.3'
        webPort = $WebPort
        apiPort = $ApiPort
        database = $developmentEnvironment.LABVIZ_DATABASE_PATH
        objects = $developmentEnvironment.LABVIZ_OBJECT_STORAGE_ROOT
        apiProxyTarget = "http://127.0.0.1:$ApiPort"
        browserApi = '/api/v1'
        cookieSuffix = '_v23'
        pythonEnvironment = $venvRoot
        nodeModules = Join-Path $webRoot 'node_modules'
        sessionRoot = Join-Path $env:LOCALAPPDATA 'LabVizV23Development\source-sessions'
        productionPackagingEnabled = $false
    } | ConvertTo-Json
    return
}

function Test-PythonCandidate {
    param(
        [string]$Executable,
        [string[]]$Prefix
    )

    $arguments = @($Prefix) + @(
        "-c",
        "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)"
    )
    try {
        $null = & $Executable @arguments 2>$null
        $exitCode = $LASTEXITCODE
    }
    catch {
        # Windows PowerShell 5.1 can promote native stderr to a terminating
        # error even when stderr is redirected. Treat an unavailable selector
        # as a failed candidate so the next supported Python version is tried.
        return $false
    }
    return $exitCode -eq 0
}

$basePython = $null
$pythonPrefix = @()
$pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($pyLauncher) {
    foreach ($selector in @("-3.12", "-3.13")) {
        if (Test-PythonCandidate -Executable $pyLauncher.Source -Prefix @($selector)) {
            $basePython = $pyLauncher.Source
            $pythonPrefix = @($selector)
            break
        }
    }
}

if (-not $basePython) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and (Test-PythonCandidate -Executable $pythonCommand.Source -Prefix @())) {
        $basePython = $pythonCommand.Source
    }
}

if (-not $basePython) {
    throw "Python 3.12 or 3.13 is required. Install it and run this launcher again."
}

$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $nodeCommand -or -not $npmCommand) {
    throw "Node.js 22.22.2 or newer is required. Install Node.js and run this launcher again."
}
$null = & $nodeCommand.Source -e "const [a,b,c]=process.versions.node.split('.').map(Number); process.exit(a>22||(a===22&&(b>22||(b===22&&c>=2)))?0:1)"
if ($LASTEXITCODE -ne 0) {
    throw "Node.js 22.22.2 or newer is required."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating the LabViz Python environment..." -ForegroundColor Cyan
    $venvArguments = @($pythonPrefix) + @("-m", "venv", $venvRoot)
    & $basePython @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python virtual environment creation failed."
    }
    $RefreshDependencies = $true
}

if ($RefreshDependencies) {
    Write-Host "Installing API dependencies..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $venvPython -m pip install -r (Join-Path $apiRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "API dependency installation failed." }
}

if ($RefreshDependencies -or -not (Test-Path -LiteralPath (Join-Path $webRoot "node_modules"))) {
    Write-Host "Installing website dependencies..." -ForegroundColor Cyan
    Push-Location $webRoot
    try {
        & $npmCommand.Source ci
        if ($LASTEXITCODE -ne 0) { throw "Website dependency installation failed." }
    }
    finally {
        Pop-Location
    }
}

$sessionRoot = Join-Path $env:LOCALAPPDATA 'LabViz\source-sessions'
New-Item -ItemType Directory -Force -Path $sessionRoot | Out-Null
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $workspaceId = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($repoRoot.ToLowerInvariant())))).Replace('-', '').Substring(0, 24)
} finally { $sha.Dispose() }
$sessionFile = Join-Path $sessionRoot "$workspaceId.json"
$sessionTemporary = Join-Path $sessionRoot "$workspaceId.new.json"
$logRoot = Join-Path $sessionRoot "$workspaceId-logs"
$userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$sessionMutex = New-Object Threading.Mutex($false, "Local\LabViz-source-$workspaceId-$userSid")
$ownsSessionMutex = $false
try { $ownsSessionMutex = $sessionMutex.WaitOne(0) }
catch [Threading.AbandonedMutexException] { $ownsSessionMutex = $true }
if (-not $ownsSessionMutex) {
    try {
        if ($RefreshDependencies) { throw 'Stop LabViz before refreshing dependencies.' }
        if (-not (Test-Path -LiteralPath $sessionFile)) { throw 'LabViz is still starting. Try again shortly.' }
        $session = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json
        if ($session.webPort -lt 1 -or $session.webPort -gt 65535 -or
            $session.apiPort -lt 1 -or $session.apiPort -gt 65535) {
            throw 'The existing LabViz session is invalid.'
        }
        $sessionDataRoot = Join-Path $apiRoot '.labviz\v23-development'
        if ($session.PSObject.Properties.Name -contains 'dataDirectory') { $sessionDataRoot = $session.dataDirectory }
        if (-not [string]::Equals($sessionDataRoot, $developmentDataRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Stop the active source launcher before switching to a restored copy.'
        }
        $null = Get-Process -Id $session.webPid -ErrorAction Stop
        $null = Get-Process -Id $session.apiPid -ErrorAction Stop
        $encrypted = [Convert]::FromBase64String($session.protectedKey)
        $keyBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $encrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        $key = [Convert]::ToBase64String($keyBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        if (-not $SkipOpenBrowser) {
            Start-Process "http://127.0.0.1:$($session.webPort)/#labviz-access=$key"
        }
        Write-Host 'Opened the running LabViz session.'
        return
    } finally { $sessionMutex.Dispose() }
}
# Owning the mutex proves that no prior source launcher is active. A hard
# termination may have left an encrypted marker behind; never reuse its key.
if (Test-Path -LiteralPath $sessionFile) { Remove-Item -LiteralPath $sessionFile -Force }
if (Test-Path -LiteralPath $sessionTemporary) { Remove-Item -LiteralPath $sessionTemporary -Force }

$apiProcess = $null
$webProcess = $null
$previousProxyTarget = $env:LABVIZ_API_PROXY_TARGET
$previousNextTelemetryDisabled = $env:NEXT_TELEMETRY_DISABLED
$previousLocalAccessKey = $env:LABVIZ_LOCAL_ACCESS_KEY
$processJob = $null
try {
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    . (Join-Path $repoRoot 'V2.0\packaging\windows\bin\process-job.ps1')
    $processJob = New-Object LabViz.ProcessJob
    if ($WebPort -eq $ApiPort) { throw "WebPort and ApiPort must be different." }
    foreach ($port in @($WebPort, $ApiPort)) {
        $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $port)
        try { $listener.Start() }
        catch { throw "Local port $port is already in use. Close the previous LabViz window or choose another port." }
        finally { $listener.Stop() }
    }
    Write-Host "Starting LabViz V2.2 local-first on http://127.0.0.1:$WebPort" -ForegroundColor Green
    Write-Host "Keep this window open; no account is needed for local projects." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop both services." -ForegroundColor Yellow
    [byte[]]$accessBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($accessBytes) } finally { $rng.Dispose() }
    $localAccessKey = [Convert]::ToBase64String($accessBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $env:LABVIZ_LOCAL_ACCESS_KEY = $localAccessKey
    $env:NEXT_TELEMETRY_DISABLED = "1"
    $apiProcess = Start-Process -FilePath $venvPython `
        -ArgumentList @("-m", "uvicorn", "labviz_api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
        -WorkingDirectory $apiRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logRoot 'api.log') `
        -RedirectStandardError (Join-Path $logRoot 'api.error.log')
    $processJob.Add($apiProcess)
    $env:LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:$ApiPort"
    $quotedNext = '"{0}"' -f (Join-Path $webRoot 'node_modules\next\dist\bin\next')
    $webProcess = Start-Process -FilePath $nodeCommand.Source `
        -ArgumentList @($quotedNext, "dev", "--hostname", "127.0.0.1", "--port", "$WebPort") `
        -WorkingDirectory $webRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logRoot 'web.log') `
        -RedirectStandardError (Join-Path $logRoot 'web.error.log')
    $processJob.Add($webProcess)

    $deadline = (Get-Date).AddSeconds(90)
    do {
        try {
            $webReady = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort/" -TimeoutSec 2).StatusCode -eq 200
        } catch { $webReady = $false }
        if (-not $webReady) { Start-Sleep -Seconds 1 }
    } while (-not $webReady -and (Get-Date) -lt $deadline)
    if (-not $webReady) { throw "LabViz did not become ready at http://127.0.0.1:$WebPort/. Logs: $logRoot" }
    $sessionRecord = @{
        dataDirectory = $developmentDataRoot
        webPort = $WebPort
        apiPort = $ApiPort
        webPid = $webProcess.Id
        apiPid = $apiProcess.Id
        protectedKey = [Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Protect(
            $accessBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
        ))
    } | ConvertTo-Json
    [IO.File]::WriteAllText($sessionTemporary, $sessionRecord, [Text.Encoding]::UTF8)
    Move-Item -LiteralPath $sessionTemporary -Destination $sessionFile -Force
    if (-not $SkipOpenBrowser) {
        Start-Process "http://127.0.0.1:$WebPort/#labviz-access=$localAccessKey"
    }

    while ($true) {
        Start-Sleep -Seconds 1
        $apiProcess.Refresh()
        $webProcess.Refresh()
        if ($apiProcess.HasExited) {
            throw "The LabViz API stopped with exit code $($apiProcess.ExitCode)."
        }
        if ($webProcess.HasExited) {
            throw "The LabViz website stopped with exit code $($webProcess.ExitCode)."
        }
    }
}
finally {
    if (Test-Path -LiteralPath $sessionFile) { Remove-Item -LiteralPath $sessionFile -Force }
    if (Test-Path -LiteralPath $sessionTemporary) { Remove-Item -LiteralPath $sessionTemporary -Force }
    $sessionMutex.ReleaseMutex()
    $sessionMutex.Dispose()
    $env:LABVIZ_API_PROXY_TARGET = $previousProxyTarget
    $env:NEXT_TELEMETRY_DISABLED = $previousNextTelemetryDisabled
    $env:LABVIZ_LOCAL_ACCESS_KEY = $previousLocalAccessKey
    foreach ($process in @($apiProcess, $webProcess)) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id
        }
    }
    if ($processJob) { $processJob.Dispose() }
}
