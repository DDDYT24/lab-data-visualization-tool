[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$WebPort = 3000,
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000,
    [switch]$SkipOpenBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $packageRoot "runtime\python\python.exe"
$node = Join-Path $packageRoot "runtime\node\node.exe"
$apiRoot = Join-Path $packageRoot "V2.0\api"
$webServer = Join-Path $packageRoot "V2.0\web\server.js"
$dataRoot = Join-Path $env:LOCALAPPDATA "LabViz\data"
$logRoot = Join-Path $env:LOCALAPPDATA "LabViz\logs"

foreach ($path in @($python, $node, $webServer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "This portable launcher needs the bundled runtime and standalone web server. Missing: $path"
    }
}
New-Item -ItemType Directory -Force -Path $dataRoot, $logRoot | Out-Null

$previousDatabasePath = $env:LABVIZ_DATABASE_PATH
$previousObjectRoot = $env:LABVIZ_OBJECT_STORAGE_ROOT
$previousProxyTarget = $env:LABVIZ_API_PROXY_TARGET
$previousPort = $env:PORT
$previousHostname = $env:HOSTNAME
$apiProcess = $null
$webProcess = $null
try {
    $env:LABVIZ_DATABASE_PATH = Join-Path $dataRoot "labviz-v2.db"
    $env:LABVIZ_OBJECT_STORAGE_ROOT = Join-Path $dataRoot "objects"
    $env:LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:$ApiPort"
    $env:PORT = "$WebPort"
    $env:HOSTNAME = "127.0.0.1"
    $apiProcess = Start-Process -FilePath $python -ArgumentList @(
        "-m", "uvicorn", "labviz_api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort"
    ) -WorkingDirectory $apiRoot -RedirectStandardOutput (Join-Path $logRoot "api.log") `
        -RedirectStandardError (Join-Path $logRoot "api.error.log") -NoNewWindow -PassThru
    $quotedWebServer = '"{0}"' -f $webServer
    $webProcess = Start-Process -FilePath $node -ArgumentList @($quotedWebServer) -WorkingDirectory $packageRoot `
        -RedirectStandardOutput (Join-Path $logRoot "web.log") `
        -RedirectStandardError (Join-Path $logRoot "web.error.log") -NoNewWindow -PassThru
    $webReady = $false
    $apiReady = $false
    $deadline = (Get-Date).AddSeconds(30)
    do {
        try {
            $webHealth = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort/" -TimeoutSec 2
            $webReady = $webHealth.StatusCode -ge 200 -and $webHealth.StatusCode -lt 500
        }
        catch {
            $webReady = $false
        }
        try {
            $apiHealth = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
            $apiReady = $apiHealth.StatusCode -ge 200 -and $apiHealth.StatusCode -lt 500
        }
        catch {
            $apiReady = $false
        }
        $apiProcess.Refresh()
        $webProcess.Refresh()
        if ($apiProcess.HasExited) { throw "LabViz API stopped with exit code $($apiProcess.ExitCode)." }
        if ($webProcess.HasExited) { throw "LabViz web server stopped with exit code $($webProcess.ExitCode)." }
        if (-not ($webReady -and $apiReady)) { Start-Sleep -Seconds 1 }
    } while (-not ($webReady -and $apiReady) -and (Get-Date) -lt $deadline)
    if ((Get-Date) -ge $deadline) { throw "LabViz web server health check timed out." }
    if (-not $SkipOpenBrowser) {
        Start-Process "http://127.0.0.1:$WebPort"
    }
    while ($true) {
        Start-Sleep -Seconds 1
        $apiProcess.Refresh()
        $webProcess.Refresh()
        if ($apiProcess.HasExited) { throw "LabViz API stopped with exit code $($apiProcess.ExitCode)." }
        if ($webProcess.HasExited) { throw "LabViz web server stopped with exit code $($webProcess.ExitCode)." }
    }
}
finally {
    foreach ($process in @($apiProcess, $webProcess)) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    $env:LABVIZ_DATABASE_PATH = $previousDatabasePath
    $env:LABVIZ_OBJECT_STORAGE_ROOT = $previousObjectRoot
    $env:LABVIZ_API_PROXY_TARGET = $previousProxyTarget
    $env:PORT = $previousPort
    $env:HOSTNAME = $previousHostname
}
