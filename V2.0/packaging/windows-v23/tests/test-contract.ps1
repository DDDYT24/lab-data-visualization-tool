[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$TestRoot)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot '..\bin\common.ps1')
$TestRoot = [IO.Path]::GetFullPath($TestRoot)
if (Test-Path -LiteralPath $TestRoot) { throw 'The test root must be new.' }
New-Item -ItemType Directory -Path $TestRoot | Out-Null
$previousLocal = $env:LOCALAPPDATA; $env:LOCALAPPDATA = Join-Path $TestRoot 'appdata'
$checks = [ordered]@{}
function Must-Reject([scriptblock]$Operation) { $rejected=$false; try { & $Operation | Out-Null } catch { $rejected=$true }; if (-not $rejected) { throw 'Expected rejection did not happen.' } }
try {
    $program = Join-Path $env:LOCALAPPDATA 'Programs\LabVizV23'
    New-Item -ItemType Directory -Force -Path $program | Out-Null
    Write-V23Json (Join-Path $program 'installation.json') @{appId=$V23AppId;format='labviz-installed-v1'}
    if ((Get-V23ProgramRoot $program) -ne $program) { throw 'Own root did not validate.' }; $checks.ownIdentity=$true
    foreach ($relative in @('Programs\LabViz','LabViz')) {
        $foreign = Join-Path $env:LOCALAPPDATA $relative
        New-Item -ItemType Directory -Force -Path $foreign | Out-Null
        Write-V23Json (Join-Path $foreign 'installation.json') @{appId=$V23AppId;format='labviz-installed-v1'}
        Must-Reject { Get-V23ProgramRoot $foreign }
    }; $checks.foreignRootsRejected=$true
    $name = Get-V23MutexName (Join-Path $env:LOCALAPPDATA 'LabVizV23')
    $otherName = Get-V23MutexName (Join-Path $env:LOCALAPPDATA 'LabVizV23Profiles\different')
    if ($name -eq $otherName -or $name -notmatch '^Local\\LabVizV23-') { throw 'Mutex namespaces collided.' }; $checks.profileMutexesDisjoint=$true
    $mutex = Enter-V23Mutex $name
    try {
        $code = '$m=New-Object Threading.Mutex($false,"' + $name + '"); try { if($m.WaitOne(0)){ $m.ReleaseMutex(); exit 2 }; exit 0 } finally {$m.Dispose()}'
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code))
        $child = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-EncodedCommand',$encoded) -WindowStyle Hidden -Wait -PassThru
        if ($child.ExitCode -ne 0) { throw 'A second process acquired the active profile mutex.' }
    } finally { $mutex.ReleaseMutex(); $mutex.Dispose() }; $checks.crossProcessExclusion=$true
    $record = Get-V23ProcessRecord (Get-Process -Id $PID)
    $null = Assert-V23Process ([pscustomobject]$record)
    $record.startedTicks = [long]$record.startedTicks-1
    Must-Reject { Assert-V23Process ([pscustomobject]$record) }; $checks.staleProcessIdentityRejected=$true
    Must-Reject { Get-V23VersionRoot $program '..\foreign' }; $checks.invalidCandidateIdRejected=$true
    $fullId = 'a' * 64
    $slot = Join-Path $program ('versions\' + $fullId.Substring(0,24))
    New-Item -ItemType Directory -Force -Path $slot | Out-Null
    Write-V23Json (Join-Path $slot 'candidate-build.json') @{appId=$V23AppId;candidateId=$fullId;dataEpoch=1}
    foreach ($file in @('runtime\python\python.exe','runtime\node\node.exe','bin\start-v23-portable.ps1','V2.0\web\server.js')) {
        $path = Join-Path $slot $file; New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null; [IO.File]::WriteAllText($path,'synthetic contract fixture')
    }
    if ((Get-V23VersionRoot $program $fullId) -ne $slot) { throw 'Short version directory did not retain full identity.' }; $checks.shortDirectoryUsesFullIdentity=$true
    $collision = ('a' * 24) + ('b' * 40)
    Must-Reject { Get-V23VersionRoot $program $collision }; $checks.hashPrefixCollisionRejected=$true
    foreach ($port in @(3000,8000,3001,8001,3305,8305)) {
        $rejected = $false
        try { & (Join-Path $PSScriptRoot '..\bin\start-v23-portable.ps1') -InstallRoot $program -WebPort $port -SkipOpenBrowser }
        catch { if ($_.Exception.Message -notlike '*reserved*') { throw }; $rejected = $true }
        if (-not $rejected) { throw 'Reserved port was accepted.' }
    }; $checks.reservedPortsRejectedBeforeRuntime=$true
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback,0)
    try {
        $listener.Start(); Assert-V23PortOwner $listener.LocalEndpoint.Port (Get-Process -Id $PID)
        $other = Get-Process | Where-Object { $_.Id -ne $PID } | Select-Object -First 1
        Must-Reject { Assert-V23PortOwner $listener.LocalEndpoint.Port $other }
        if (-not $listener.Server.IsBound) { throw 'Port-owner rejection disturbed the foreign listener.' }
    } finally { $listener.Stop() }; $checks.portOwnerCheckedWithoutStoppingForeignListener=$true
    $target = Join-Path $TestRoot 'foreign-linked-target'; New-Item -ItemType Directory -Path $target | Out-Null
    [IO.File]::WriteAllText((Join-Path $target 'keep'),'untouched')
    $link = Join-Path $TestRoot 'junction'
    $null = New-Item -ItemType Junction -Path $link -Target $target
    try { Must-Reject { Assert-Unlinked (Join-Path $link 'keep') } } finally {
        if (([IO.Path]::GetFullPath($link)) -ne (Join-Path $TestRoot 'junction')) { throw 'Unexpected test link path.' }
        [IO.Directory]::Delete($link)
    }
    if ([IO.File]::ReadAllText((Join-Path $target 'keep')) -ne 'untouched') { throw 'Junction target changed.' }; $checks.junctionRejectedWithoutChangingTarget=$true
    $empty = Join-Path $TestRoot 'new-install-destination'
    if ((Assert-V23Destination $empty) -ne $empty -or (Test-Path -LiteralPath $empty)) { throw 'Empty-destination preflight changed the file system.' }
    $unknown = Join-Path $TestRoot 'unknown-install'; New-Item -ItemType Directory -Path $unknown | Out-Null
    [IO.File]::WriteAllText((Join-Path $unknown 'keep'),'untouched')
    Must-Reject { Assert-V23Destination $unknown }
    if ([IO.File]::ReadAllText((Join-Path $unknown 'keep')) -ne 'untouched') { throw 'Unknown directory was changed.' }
    $checks.destinationPreflightRetainsUnrelatedDirectories=$true
    . (Join-Path $PSScriptRoot '..\bin\process-job.ps1')
    $reservation=New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback,0)
    $reservation.Server.ExclusiveAddressUse=$true; $reservation.Start()
    [LabViz.ProcessJob]::PreventInheritance($reservation.Server.Handle)
    $port=$reservation.LocalEndpoint.Port
    $job=New-Object LabViz.ProcessJob; $replacement=$null
    try {
        $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes('Start-Sleep -Seconds 10'))
        $child=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-EncodedCommand',$encoded) -RedirectStandardOutput (Join-Path $TestRoot 'socket-child.log') -WindowStyle Hidden -PassThru
        $job.Add($child);$reservation.Stop()
        $replacement=New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback,$port)
        $replacement.Server.ExclusiveAddressUse=$true;$replacement.Start()
        $child.Refresh();if($child.HasExited){throw 'Child exited before socket inheritance could be checked.'}
        $checks.portReservationNotInheritedByChild=$true
    } finally {$reservation.Stop();if($replacement){$replacement.Stop()};$job.Dispose()}
    $result=@{passed=$true;checks=$checks;scope='Native path, identity, mutex, reserved-port, socket inheritance and process-owner contracts; not an installer or real offline UI acceptance.'}
    Write-V23Json (Join-Path $TestRoot 'verification.json') $result
    $result | ConvertTo-Json -Depth 10
} finally { $env:LOCALAPPDATA = $previousLocal }
