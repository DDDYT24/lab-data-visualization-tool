[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$WebPort = 3000,
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000,
    [switch]$SkipOpenBrowser,
    [switch]$HealthCheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Security

$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $packageRoot "runtime\python\python.exe"
$node = Join-Path $packageRoot "runtime\node\node.exe"
$apiRoot = Join-Path $packageRoot "V2.0\api"
$webServer = Join-Path $packageRoot "V2.0\web\server.js"
$dataRoot = Join-Path $env:LOCALAPPDATA "LabViz\data"
$logRoot = Join-Path $env:LOCALAPPDATA "LabViz\logs"
$tokenPath = Join-Path $dataRoot "local-access.dpapi"

foreach ($path in @($python, $node, $webServer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "This portable launcher needs the bundled runtime and standalone web server. Missing: $path"
    }
}
New-Item -ItemType Directory -Force -Path $dataRoot, $logRoot | Out-Null

# Only one LabViz process group may write this user's packaged data.
$mutex = New-Object Threading.Mutex($false, ("Local\LabViz-" + [Security.Principal.WindowsIdentity]::GetCurrent().User.Value))
$ownsMutex = $false
try { $ownsMutex = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
if (-not $ownsMutex) {
    $mutex.Dispose()
    $readyPath = Join-Path $logRoot "running.json"
    if ((Test-Path -LiteralPath $readyPath) -and (Test-Path -LiteralPath $tokenPath)) {
        $running = Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json
        $webUri = [Uri]$running.webUrl
        if ($webUri.Scheme -ne 'http' -or $webUri.Host -ne '127.0.0.1' -or $webUri.Port -lt 1) {
            throw 'The running LabViz address is not a local loopback URL.'
        }
        $protectedBytes = [IO.File]::ReadAllBytes($tokenPath)
        $accessBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $protectedBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        $accessKey = [Convert]::ToBase64String($accessBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        Start-Process "$($webUri.GetLeftPart([UriPartial]::Authority))/#labviz-access=$accessKey"
        return
    }
    throw "LabViz is already running / LabViz 已在运行。"
}
function Get-AvailablePort([int]$Preferred) {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $Preferred)
    try { $listener.Start(); return $listener.LocalEndpoint.Port }
    catch {
        $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        return $listener.LocalEndpoint.Port
    } finally { $listener.Stop() }
}
$stopFile = Join-Path $logRoot "stop.request"
$readyFile = Join-Path $logRoot "running.json"

$previousDatabasePath = $env:LABVIZ_DATABASE_PATH
$previousObjectRoot = $env:LABVIZ_OBJECT_STORAGE_ROOT
$previousProxyTarget = $env:LABVIZ_API_PROXY_TARGET
$previousPort = $env:PORT
$previousHostname = $env:HOSTNAME
$previousOrigins = $env:LABVIZ_ALLOWED_ORIGINS
$previousPublicUrl = $env:LABVIZ_PUBLIC_WEB_URL
$previousLocalAccessKey = $env:LABVIZ_LOCAL_ACCESS_KEY
$apiProcess = $null
$webProcess = $null
$processJob = $null
try {
    . (Join-Path $PSScriptRoot 'process-job.ps1')
    $processJob = New-Object LabViz.ProcessJob
    $WebPort = Get-AvailablePort $WebPort
    $ApiPort = Get-AvailablePort $ApiPort
    while ($ApiPort -eq $WebPort) { $ApiPort = Get-AvailablePort 0 }
    if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile }
    if (Test-Path -LiteralPath $readyFile) { Remove-Item -LiteralPath $readyFile }
    Write-Host "Starting LabViz / 正在启动 LabViz..."
    [byte[]]$accessBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($accessBytes) } finally { $rng.Dispose() }
    $localAccessKey = [Convert]::ToBase64String($accessBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $protectedBytes = [Security.Cryptography.ProtectedData]::Protect(
        $accessBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    [IO.File]::WriteAllBytes($tokenPath, $protectedBytes)
    $env:LABVIZ_LOCAL_ACCESS_KEY = $localAccessKey
    $env:LABVIZ_DATABASE_PATH = Join-Path $dataRoot "labviz-v2.db"
    $env:LABVIZ_OBJECT_STORAGE_ROOT = Join-Path $dataRoot "objects"
    $env:LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:$ApiPort"
    $env:PORT = "$WebPort"
    $env:HOSTNAME = "127.0.0.1"
    $env:LABVIZ_ALLOWED_ORIGINS = "http://127.0.0.1:$WebPort"
    $env:LABVIZ_PUBLIC_WEB_URL = "http://127.0.0.1:$WebPort"
    $apiProcess = Start-Process -FilePath $python -ArgumentList @(
        "-m", "uvicorn", "labviz_api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort"
    ) -WorkingDirectory $apiRoot -RedirectStandardOutput (Join-Path $logRoot "api.log") `
        -RedirectStandardError (Join-Path $logRoot "api.error.log") -WindowStyle Hidden -PassThru
    $processJob.Add($apiProcess)
    $quotedWebServer = '"{0}"' -f $webServer
    $webProcess = Start-Process -FilePath $node -ArgumentList @($quotedWebServer) -WorkingDirectory $packageRoot `
        -RedirectStandardOutput (Join-Path $logRoot "web.log") `
        -RedirectStandardError (Join-Path $logRoot "web.error.log") -WindowStyle Hidden -PassThru
    $processJob.Add($webProcess)
    $webReady = $false
    $apiReady = $false
    $deadline = (Get-Date).AddSeconds(90)
    $progressAt = Get-Date
    do {
        try {
            $webHealth = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort/" -TimeoutSec 2
            $webReady = $webHealth.StatusCode -eq 200
        }
        catch {
            $webReady = $false
        }
        try {
            $apiHealth = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
            $apiReady = $apiHealth.StatusCode -eq 200
        }
        catch {
            $apiReady = $false
        }
        $apiProcess.Refresh()
        $webProcess.Refresh()
        if ($apiProcess.HasExited) { throw "LabViz API stopped with exit code $($apiProcess.ExitCode)." }
        if ($webProcess.HasExited) { throw "LabViz web server stopped with exit code $($webProcess.ExitCode)." }
        if ((Get-Date) -ge $progressAt -and -not ($webReady -and $apiReady)) {
            Write-Host "Waiting for local services / 正在等待本地服务 (Web=$webReady, API=$apiReady)..."
            $progressAt = (Get-Date).AddSeconds(5)
        }
        if (-not ($webReady -and $apiReady)) { Start-Sleep -Seconds 1 }
    } while (-not ($webReady -and $apiReady) -and (Get-Date) -lt $deadline)
    if (-not ($webReady -and $apiReady)) {
        throw "LabViz startup health check timed out (Web=$webReady, API=$apiReady). See logs / 启动超时，请查看日志: $logRoot"
    }
    if ($HealthCheckOnly) { return }
    @{ webUrl = "http://127.0.0.1:$WebPort"; apiPort = $ApiPort; launcherPid = $PID } |
        ConvertTo-Json | Set-Content -LiteralPath $readyFile -Encoding UTF8
    Write-Host "LabViz ready / 已启动。Close this window to stop / 关闭此窗口以退出。"
    if (-not $SkipOpenBrowser) {
        Start-Process "http://127.0.0.1:$WebPort/#labviz-access=$localAccessKey"
    }
    while (-not (Test-Path -LiteralPath $stopFile)) {
        Start-Sleep -Seconds 1
        $apiProcess.Refresh()
        $webProcess.Refresh()
        if ($apiProcess.HasExited) { throw "LabViz API stopped with exit code $($apiProcess.ExitCode)." }
        if ($webProcess.HasExited) { throw "LabViz web server stopped with exit code $($webProcess.ExitCode)." }
    }
}
finally {
    if ($processJob) { $processJob.Dispose() }
    foreach ($process in @($apiProcess, $webProcess)) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
    }
    foreach ($stateFile in @($readyFile, $stopFile)) {
        if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
    }
    $mutex.ReleaseMutex()
    $mutex.Dispose()
    $env:LABVIZ_DATABASE_PATH = $previousDatabasePath
    $env:LABVIZ_OBJECT_STORAGE_ROOT = $previousObjectRoot
    $env:LABVIZ_API_PROXY_TARGET = $previousProxyTarget
    $env:PORT = $previousPort
    $env:HOSTNAME = $previousHostname
    $env:LABVIZ_ALLOWED_ORIGINS = $previousOrigins
    $env:LABVIZ_PUBLIC_WEB_URL = $previousPublicUrl
    $env:LABVIZ_LOCAL_ACCESS_KEY = $previousLocalAccessKey
}
