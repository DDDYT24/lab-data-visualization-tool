Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$V23AppId = '55B46BB7-101E-46DC-951A-3DE7E402EDD0'

function Assert-Unlinked([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $walk = $absolute
    while ($walk) {
        if (Test-Path -LiteralPath $walk) {
            $item = Get-Item -LiteralPath $walk -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Linked path refused: $walk" }
        }
        $parent = Split-Path -Parent $walk
        if ($parent -eq $walk) { break }
        $walk = $parent
    }
    return $absolute.TrimEnd('\')
}

function Write-V23Json([string]$Path, $Value) {
    $null = Assert-Unlinked $Path
    $temporary = "$Path.new-$([guid]::NewGuid().ToString('N'))"
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 15), (New-Object Text.UTF8Encoding($false)))
    if (Test-Path -LiteralPath $Path) {
        $backup = "$Path.previous-$([guid]::NewGuid().ToString('N'))"
        [IO.File]::Replace($temporary, $Path, $backup)
        Remove-Item -LiteralPath $backup -Force
    } else { [IO.File]::Move($temporary, $Path) }
}

function Get-V23MutexName([string]$ProfileRoot) {
    $root = Assert-Unlinked $ProfileRoot
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $key = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes("$sid|$($root.ToLowerInvariant())")))).Replace('-', '') }
    finally { $hash.Dispose() }
    return "Local\LabVizV23-$key"
}

function Enter-V23Mutex([string]$Name) {
    $mutex = New-Object Threading.Mutex($false, $Name)
    $owned = $false
    try { $owned = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned = $true }
    if (-not $owned) { $mutex.Dispose(); throw 'This installed profile is running or being maintained. Stop it first.' }
    return $mutex
}

function Assert-V23Destination([string]$Root, [switch]$RequireIdentity) {
    $root = Assert-Unlinked $Root
    foreach ($relative in @('Programs\LabViz','LabViz')) {
        $foreign = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA $relative)).TrimEnd('\')
        if ($root -eq $foreign -or $root.StartsWith("$foreign\", [StringComparison]::OrdinalIgnoreCase)) { throw 'The inherited V2.2 program/data directory is protected.' }
    }
    if ($root -match '(?i)(^|[\\/])\.labviz([\\/]|$)') { throw 'Source data directories cannot be installation roots.' }
    $marker = Join-Path $root 'installation.json'
    $null = Assert-Unlinked $marker
    if (-not (Test-Path -LiteralPath $marker)) {
        if ($RequireIdentity -or ((Test-Path -LiteralPath $root) -and @(Get-ChildItem -LiteralPath $root -Force).Count -gt 0)) { throw 'An existing unrelated directory cannot be adopted as an installation.' }
        return $root
    }
    $identity = Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($identity.appId -ne $V23AppId -or $identity.format -ne 'labviz-installed-v1') { throw 'Invalid V2.3 installation identity.' }
    return $root
}

function Get-V23ProgramRoot([string]$Root) { return Assert-V23Destination $Root -RequireIdentity }

function Get-V23VersionRoot([string]$InstallRoot, [string]$CandidateId) {
    if ($CandidateId -notmatch '^[a-f0-9]{64}$') { throw 'Invalid immutable candidate ID.' }
    $directoryId = $CandidateId.Substring(0,24)
    $root = Assert-Unlinked (Join-Path $InstallRoot "versions\$directoryId")
    $metadata = Get-Content -LiteralPath (Join-Path $root 'candidate-build.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($metadata.appId -ne $V23AppId -or $metadata.candidateId -ne $CandidateId -or $metadata.dataEpoch -ne 1) { throw 'Candidate identity or supported data epoch differs.' }
    foreach ($relative in @('runtime\python\python.exe','runtime\node\node.exe','bin\start-v23-portable.ps1','V2.0\web\server.js')) {
        $path = Assert-Unlinked (Join-Path $root $relative)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Incomplete candidate: $relative" }
    }
    return $root
}

function Invoke-V23Data([string]$VersionRoot, [string]$Action, [string]$ProfileRoot, [string]$Source = '', [string]$ConfirmId = '') {
    $python = Join-Path $VersionRoot 'runtime\python\python.exe'
    $arguments = @('-B','-X','utf8','-m','labviz_api.installed_lifecycle',$Action,'--profile',$ProfileRoot)
    if ($Source) { $arguments += @('--source',$Source) }
    if ($ConfirmId) { $arguments += @('--confirm-profile-id',$ConfirmId) }
    $oldPythonPath = $env:PYTHONPATH; $oldPythonHome = $env:PYTHONHOME
    try {
        Remove-Item Env:PYTHONPATH,Env:PYTHONHOME -ErrorAction SilentlyContinue
        $output = & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "V2.3 data operation failed: $Action. Data copies have been retained." }
        return ($output | ConvertFrom-Json)
    } finally { $env:PYTHONPATH = $oldPythonPath; $env:PYTHONHOME = $oldPythonHome }
}

function Assert-V23Process($Record) {
    $process = Get-Process -Id ([int]$Record.pid) -ErrorAction SilentlyContinue
    if (-not $process -or $process.HasExited -or $process.StartTime.ToUniversalTime().Ticks -ne [long]$Record.startedTicks -or $process.Path -ne $Record.executable) { throw 'Process identity changed; no stop signal was sent.' }
    return $process
}

function Get-V23ProcessRecord($Process) {
    $Process.Refresh()
    return @{ pid=$Process.Id; startedTicks=$Process.StartTime.ToUniversalTime().Ticks; executable=$Process.Path }
}

function Assert-V23PortOwner([int]$Port, $Process) {
    $Process.Refresh()
    if ($Process.HasExited) { throw 'Owned service exited.' }
    $owners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1 -or $owners[0] -ne $Process.Id) { throw 'Port belongs to another process; credentials were not sent.' }
}
