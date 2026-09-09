[CmdletBinding()]
param(
    [switch]$RefreshDependencies,
    [ValidateRange(1, 65535)]
    [int]$WebPort = 3000,
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$apiRoot = Join-Path $repoRoot "V2.0\api"
$webRoot = Join-Path $repoRoot "V2.0\web"
$venvRoot = Join-Path $apiRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

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

Write-Host "Starting LabViz V2.0 on http://127.0.0.1:$WebPort" -ForegroundColor Green
Write-Host "Keep this window open. Local sign-in codes appear in the API output." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both services." -ForegroundColor Yellow

$apiProcess = $null
$webProcess = $null
$previousProxyTarget = $env:LABVIZ_API_PROXY_TARGET
try {
    $apiProcess = Start-Process -FilePath $venvPython `
        -ArgumentList @("-m", "uvicorn", "labviz_api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
        -WorkingDirectory $apiRoot -NoNewWindow -PassThru
    $env:LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:$ApiPort"
    $webProcess = Start-Process -FilePath $npmCommand.Source `
        -ArgumentList @("run", "dev", "--", "--hostname", "127.0.0.1", "--port", "$WebPort") `
        -WorkingDirectory $webRoot -NoNewWindow -PassThru

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
    $env:LABVIZ_API_PROXY_TARGET = $previousProxyTarget
    foreach ($process in @($apiProcess, $webProcess)) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id
        }
    }
}
