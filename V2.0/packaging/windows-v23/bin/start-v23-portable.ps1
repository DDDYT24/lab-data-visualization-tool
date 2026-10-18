[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InstallRoot,
    [string]$ProfileRoot = (Join-Path $env:LOCALAPPDATA 'LabVizV23'),
    [ValidateRange(0,65535)][int]$WebPort = 3301,
    [ValidateRange(0,65535)][int]$ApiPort = 8301,
    [switch]$SkipOpenBrowser,
    [switch]$HealthCheckOnly
)
. (Join-Path $PSScriptRoot 'common.ps1')
Add-Type -AssemblyName System.Security
$reserved = @(3000,8000,3001,8001,3305,8305)
if ($WebPort -in $reserved -or $ApiPort -in $reserved) { throw 'A V2.2, source, or acceptance port is reserved.' }
$InstallRoot = Get-V23ProgramRoot $InstallRoot
$packageRoot = Assert-Unlinked (Join-Path $PSScriptRoot '..')
$metadata = Get-Content -LiteralPath (Join-Path $packageRoot 'candidate-build.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($metadata.appId -ne $V23AppId -or $metadata.dataEpoch -ne 1) { throw 'Invalid portable identity.' }
$profile = Invoke-V23Data $packageRoot 'init' $ProfileRoot
$ProfileRoot = $profile.root
$profileMutex = Enter-V23Mutex (Get-V23MutexName $ProfileRoot)
$applicationMutex = New-Object Threading.Mutex($false, 'Local\LabVizV23-App')
$allocationMutex = $null; $allocationOwned = $false
$apiProcess = $null; $webProcess = $null; $job = $null
$apiReservation = $null; $webReservation = $null
$apiSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$session = [guid]::NewGuid().ToString('N')
$logs = Assert-Unlinked (Join-Path $ProfileRoot 'logs')
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$readyFile = Join-Path $logs 'running.json'
$stopFile = Join-Path $logs 'stop.request'
$keys = @('LABVIZ_DATABASE_PATH','LABVIZ_OBJECT_STORAGE_ROOT','LABVIZ_PERSISTENCE_BACKEND','LABVIZ_OBJECT_STORAGE_BACKEND','LABVIZ_ENVIRONMENT','LABVIZ_AUTH_MODE','LABVIZ_RUNTIME_ROLE','LABVIZ_INSTALL_LINE','LABVIZ_DEVELOPMENT_LINE','LABVIZ_PROFILE_ROOT','LABVIZ_PROFILE_ID','LABVIZ_INSTALL_ROOT','LABVIZ_API_PROXY_TARGET','LABVIZ_ALLOWED_ORIGINS','LABVIZ_PUBLIC_WEB_URL','LABVIZ_LOCAL_ACCESS_KEY','PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','NODE_PATH','PORT','HOSTNAME','NODE_ENV','NEXT_TELEMETRY_DISABLED')
$previous = @{}; foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }

function Reserve-Port([int]$Preferred) {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $Preferred)
    $listener.Server.ExclusiveAddressUse = $true
    try { $listener.Start() }
    catch {
        $listener.Stop()
        $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
    }
    try { [LabViz.ProcessJob]::PreventInheritance($listener.Server.Handle) }
    catch { $listener.Stop(); throw }
    if ($listener.LocalEndpoint.Port -in $reserved) { $listener.Stop(); return Reserve-Port 0 }
    return $listener
}

try {
    $null = Invoke-V23Data $packageRoot 'recover' $ProfileRoot
    if ($profile.dataEpoch -gt $metadata.dataEpoch) { throw 'A newer data schema requires a compatible runtime or a matching rollback backup.' }
    foreach ($state in @($readyFile,$stopFile)) { if (Test-Path -LiteralPath $state) { Remove-Item -LiteralPath $state } }
    . (Join-Path $PSScriptRoot 'process-job.ps1')
    $job = New-Object LabViz.ProcessJob
    $allocationMutex = New-Object Threading.Mutex($false, 'Local\LabVizV23-PortAllocation')
    try { $allocationOwned = $allocationMutex.WaitOne(90000) } catch [Threading.AbandonedMutexException] { $allocationOwned = $true }
    if (-not $allocationOwned) { throw 'Another installed instance is still allocating ports.' }
    $webReservation = Reserve-Port $WebPort; $WebPort = $webReservation.LocalEndpoint.Port
    $apiReservation = Reserve-Port $ApiPort; $ApiPort = $apiReservation.LocalEndpoint.Port
    [byte[]]$random = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($random) } finally { $rng.Dispose() }
    $accessKey = [Convert]::ToBase64String($random).TrimEnd('=').Replace('+','-').Replace('/','_')
    [IO.File]::WriteAllBytes((Join-Path $ProfileRoot 'access.dpapi'), [Security.Cryptography.ProtectedData]::Protect($random,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser))
    $values = @{
        LABVIZ_DATABASE_PATH=(Join-Path $ProfileRoot 'data\labviz.db'); LABVIZ_OBJECT_STORAGE_ROOT=(Join-Path $ProfileRoot 'data\objects')
        LABVIZ_PERSISTENCE_BACKEND='sqlite'; LABVIZ_OBJECT_STORAGE_BACKEND='local'; LABVIZ_ENVIRONMENT='development'; LABVIZ_AUTH_MODE='console'
        LABVIZ_INSTALL_LINE='2.3'; LABVIZ_DEVELOPMENT_LINE='2.3'; LABVIZ_PROFILE_ROOT=$ProfileRoot; LABVIZ_PROFILE_ID=$profile.profileId; LABVIZ_INSTALL_ROOT=$InstallRoot
        LABVIZ_API_PROXY_TARGET="http://127.0.0.1:$ApiPort"; LABVIZ_ALLOWED_ORIGINS="http://127.0.0.1:$WebPort"; LABVIZ_PUBLIC_WEB_URL="http://127.0.0.1:$WebPort"
        LABVIZ_LOCAL_ACCESS_KEY=$accessKey; PORT="$WebPort"; HOSTNAME='127.0.0.1'; NODE_ENV='production'; NEXT_TELEMETRY_DISABLED='1'; PYTHONDONTWRITEBYTECODE='1'; MPLCONFIGDIR=(Join-Path $ProfileRoot 'cache\matplotlib'); LABVIZ_RUNTIME_ROLE='api'
    }
    foreach ($key in $values.Keys) { [Environment]::SetEnvironmentVariable($key,$values[$key],'Process') }
    foreach ($key in @('PYTHONPATH','PYTHONHOME','NODE_PATH')) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $python = Join-Path $packageRoot 'runtime\python\python.exe'
    $node = Join-Path $packageRoot 'runtime\node\node.exe'
    $apiReservation.Stop(); $apiReservation = $null
    $apiProcess = Start-Process -FilePath $python -ArgumentList @('-B','-X','utf8','-m','labviz_api.installed_server','--port',"$ApiPort") -WorkingDirectory (Join-Path $packageRoot 'V2.0\api') -RedirectStandardOutput (Join-Path $logs 'api.log') -RedirectStandardError (Join-Path $logs 'api.error.log') -WindowStyle Hidden -PassThru
    $job.Add($apiProcess)
    $deadline = (Get-Date).AddSeconds(120); $apiReady = $false
    do {
        $apiProcess.Refresh(); if ($apiProcess.HasExited) { throw 'Owned API exited. See this profile logs; foreign ports were not stopped.' }
        try {
            Assert-V23PortOwner $ApiPort $apiProcess
            $null = Invoke-RestMethod -UseBasicParsing -Method Post -Uri "http://127.0.0.1:$ApiPort/api/v1/local/session" -Headers @{'X-LabViz-Local-Key'=$accessKey} -WebSession $apiSession -TimeoutSec 2
            $instance = Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/api/v1/local/instance" -Headers @{'X-LabViz-Local-Key'=$accessKey} -WebSession $apiSession -TimeoutSec 2
            $apiReady = $instance.appId -eq $V23AppId -and $instance.profileId -eq $profile.profileId -and $instance.pid -eq $apiProcess.Id
        } catch { $apiReady = $false }
        if (-not $apiReady) { Start-Sleep -Milliseconds 300 }
    } while (-not $apiReady -and (Get-Date) -lt $deadline)
    if (-not $apiReady) { throw 'Installed API identity/health check timed out.' }
    $webReservation.Stop(); $webReservation = $null
    $quotedServer = '"{0}"' -f (Join-Path $packageRoot 'V2.0\web\server.js')
    $webProcess = Start-Process -FilePath $node -ArgumentList @($quotedServer) -WorkingDirectory $packageRoot -RedirectStandardOutput (Join-Path $logs 'web.log') -RedirectStandardError (Join-Path $logs 'web.error.log') -WindowStyle Hidden -PassThru
    $job.Add($webProcess)
    $webReady = $false
    do {
        $webProcess.Refresh(); if ($webProcess.HasExited) { throw 'Owned Web server exited.' }
        try {
            Assert-V23PortOwner $WebPort $webProcess
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort/" -TimeoutSec 3
            $webReady = $response.StatusCode -eq 200
        } catch { $webReady = $false }
        if (-not $webReady) { Start-Sleep -Milliseconds 300 }
    } while (-not $webReady -and (Get-Date) -lt $deadline)
    if (-not $webReady) { throw 'Installed Web health check timed out.' }
    $allocationMutex.ReleaseMutex(); $allocationOwned = $false
    $running = @{ appId=$V23AppId; profileId=$profile.profileId; sessionId=$session; candidateId=$metadata.candidateId; webUrl="http://127.0.0.1:$WebPort"; apiPort=$ApiPort; launcher=(Get-V23ProcessRecord (Get-Process -Id $PID)); api=(Get-V23ProcessRecord $apiProcess); web=(Get-V23ProcessRecord $webProcess) }
    if ($HealthCheckOnly) { Write-Host 'V2.3 installed health and identity checks passed.'; return }
    Write-V23Json $readyFile $running
    if (-not $SkipOpenBrowser) { Start-Process "http://127.0.0.1:$WebPort/#labviz-access=$accessKey" }
    Write-Host 'LabViz V2.3 ready / 已启动。Use the installed Stop shortcut to exit.'
    while ($true) {
        if (Test-Path -LiteralPath $stopFile) {
            $request = Get-Content -LiteralPath $stopFile -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($request.profileId -eq $profile.profileId -and $request.sessionId -eq $session) { break }
        }
        $apiProcess.Refresh(); $webProcess.Refresh()
        if ($apiProcess.HasExited -or $webProcess.HasExited) { throw 'An owned service exited unexpectedly; durable jobs recover on the next start.' }
        Start-Sleep -Milliseconds 500
    }
} finally {
    if ($apiReservation) { $apiReservation.Stop() }; if ($webReservation) { $webReservation.Stop() }
    if ($apiProcess -and -not $apiProcess.HasExited -and (Get-Variable accessKey -ErrorAction SilentlyContinue)) {
        try {
            Assert-V23PortOwner $ApiPort $apiProcess
            $null = Invoke-RestMethod -UseBasicParsing -Method Post -Uri "http://127.0.0.1:$ApiPort/api/v1/local/shutdown" -Headers @{'X-LabViz-Local-Key'=$accessKey} -WebSession $apiSession -TimeoutSec 3
            $null = $apiProcess.WaitForExit(45000)
        } catch { Write-Host 'Closing the owned process job after shutdown did not complete.' }
    }
    if ($job) { $job.Dispose() }
    foreach ($process in @($apiProcess,$webProcess)) { if ($process) { $null = $process.WaitForExit(10000) } }
    foreach ($state in @($readyFile,$stopFile)) { if (Test-Path -LiteralPath $state) { Remove-Item -LiteralPath $state -Force } }
    if ($allocationOwned) { $allocationMutex.ReleaseMutex() }; if ($allocationMutex) { $allocationMutex.Dispose() }
    $profileMutex.ReleaseMutex(); $profileMutex.Dispose(); $applicationMutex.Dispose()
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
