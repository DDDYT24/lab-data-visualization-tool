[CmdletBinding()]
param(
    [ValidateSet('Info','Stop','Backup','Verify','Restore','Import','Rollback','RemoveData','Recover')][string]$Action='Info',
    [string]$ProfileRoot=(Join-Path $env:LOCALAPPDATA 'LabVizV23'),
    [string]$DataDirectory,
    [string]$BackupDirectory,
    [string]$SourceDirectory,
    [string]$ConfirmProfileId,
    [switch]$SkipOpenBrowser
)
. (Join-Path $PSScriptRoot 'common.ps1')
$installRoot = Get-V23ProgramRoot (Join-Path $PSScriptRoot '..')
$currentFile = Join-Path $installRoot 'current-candidate.json'
if (-not (Test-Path -LiteralPath $currentFile)) { $currentFile = Join-Path $installRoot 'pending-candidate.json' }
$candidate = Get-Content -LiteralPath $currentFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($candidate.appId -ne $V23AppId) { throw 'Foreign installation marker.' }
$versionRoot = Get-V23VersionRoot $installRoot $candidate.candidateId
if ($Action -eq 'Import') {
    if (-not $SourceDirectory) { throw 'SourceDirectory is required; imports are explicit and copy-only.' }
    if (-not $PSBoundParameters.ContainsKey('ProfileRoot')) { $ProfileRoot = Join-Path (Join-Path $env:LOCALAPPDATA 'LabVizV23Profiles') ([guid]::NewGuid().ToString('N')) }
    $profile = Invoke-V23Data $versionRoot 'init' $ProfileRoot
} elseif ($Action -eq 'Restore') { $profile = Invoke-V23Data $versionRoot 'init' $ProfileRoot }
else { $profile = Invoke-V23Data $versionRoot 'info' $ProfileRoot }
$ProfileRoot = $profile.root
if ($DataDirectory -and (Assert-Unlinked $DataDirectory) -ne (Join-Path $ProfileRoot 'data')) { throw 'The data directory does not match this installed profile.' }
if ($Action -eq 'Stop') {
    $ready = Join-Path $ProfileRoot 'logs\running.json'
    if (-not (Test-Path -LiteralPath $ready)) { Write-Host 'This profile is already stopped.'; return }
    $running = Get-Content -LiteralPath $ready -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($running.appId -ne $V23AppId -or $running.profileId -ne $profile.profileId) { throw 'Foreign running record refused.' }
    $null = Assert-V23Process $running.launcher
    $null = Assert-V23Process $running.api
    $null = Assert-V23Process $running.web
    Write-V23Json (Join-Path $ProfileRoot 'logs\stop.request') @{profileId=$profile.profileId;sessionId=$running.sessionId}
    return
}
if ($Action -eq 'Rollback') {
    & (Join-Path $PSScriptRoot 'start-v23-installed.ps1') -ProfileRoot $ProfileRoot -Rollback -SkipOpenBrowser:$SkipOpenBrowser
    return
}
$mutex = Enter-V23Mutex (Get-V23MutexName $ProfileRoot)
$sourceMutex = $null
try {
    switch ($Action) {
        'Info' { $profile | ConvertTo-Json }
        'Backup' { Invoke-V23Data $versionRoot 'backup' $ProfileRoot | ConvertTo-Json -Depth 10 }
        'Verify' { if (-not $BackupDirectory) { throw 'BackupDirectory is required.' }; Invoke-V23Data $versionRoot 'verify' $ProfileRoot $BackupDirectory | ConvertTo-Json -Depth 15 }
        'Restore' {
            if (-not $BackupDirectory) { throw 'BackupDirectory is required.' }
            $result = Invoke-V23Data $versionRoot 'restore' $ProfileRoot $BackupDirectory
            $result | ConvertTo-Json -Depth 10
            Write-Host ("Start independent copy / 启动独立副本: & '{0}' -ProfileRoot '{1}'" -f (Join-Path $PSScriptRoot 'start-v23-installed.ps1').Replace("'","''"),$result.restoredProfileRoot.Replace("'","''"))
        }
        'Import' {
            if (-not $SourceDirectory) { throw 'SourceDirectory is required; imports are explicit and copy-only.' }
            $source = Assert-Unlinked $SourceDirectory
            $legacyRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'LabViz')).TrimEnd('\')
            if ($source.StartsWith("$legacyRoot\",[StringComparison]::OrdinalIgnoreCase)) {
                $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
                $sourceMutex = Enter-V23Mutex "Local\LabViz-$sid"
            } elseif ($source -match '^(.*)[\\/]V2\.0[\\/]api[\\/]\.labviz(?:[\\/].*)?$') {
                $workspace = (Assert-Unlinked $Matches[1]).ToLowerInvariant()
                $hash = [Security.Cryptography.SHA256]::Create()
                try { $workspaceId = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspace)))).Replace('-','').Substring(0,24) } finally { $hash.Dispose() }
                $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
                $sourceMutex = Enter-V23Mutex "Local\LabViz-source-$workspaceId-$sid"
            }
            $result = Invoke-V23Data $versionRoot 'import' $ProfileRoot $source
            $result | ConvertTo-Json -Depth 10
            Write-Host ("Start imported independent copy / 启动导入的独立副本: & '{0}' -ProfileRoot '{1}'" -f (Join-Path $PSScriptRoot 'start-v23-installed.ps1').Replace("'","''"),$ProfileRoot.Replace("'","''"))
        }
        'Recover' { Invoke-V23Data $versionRoot 'recover' $ProfileRoot | ConvertTo-Json }
        'RemoveData' {
            if (-not $ConfirmProfileId) { throw 'ConfirmProfileId must equal the exact owned profile ID.' }
            Invoke-V23Data $versionRoot 'remove' $ProfileRoot '' $ConfirmProfileId | ConvertTo-Json
        }
    }
} finally {
    if ($sourceMutex) { $sourceMutex.ReleaseMutex(); $sourceMutex.Dispose() }
    $mutex.ReleaseMutex(); $mutex.Dispose()
}
