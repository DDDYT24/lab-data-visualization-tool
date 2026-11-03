[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Backup','Verify','Restore')][string]$Action,
    [string]$DataDirectory,
    [string]$BackupDirectory
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $workspaceId = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($repoRoot.ToLowerInvariant())))).Replace('-', '').Substring(0, 24)
} finally { $sha.Dispose() }
$userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$sourceDataMutex = New-Object Threading.Mutex($false, "Local\LabViz-source-$workspaceId-$userSid")
$ownsSourceDataMutex = $false
try {
    try { $ownsSourceDataMutex = $sourceDataMutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsSourceDataMutex = $true }
    if (-not $ownsSourceDataMutex) { throw 'Stop this V2.3 source launcher with Ctrl+C before using data tools.' }
    # A hard-killed launcher may leave its private child processes; refuse them too.
    $sourceWriters = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and ($_.CommandLine.Contains($repoRoot) -and
        ($_.CommandLine -match 'uvicorn|next[\\/].*(start|dev)|next (start|dev)'))
    })
    if ($sourceWriters.Count -gt 0) { throw 'A V2.3 source service is still running. Stop it before using data tools.' }
    $sourceDataPython = Join-Path $repoRoot 'V2.0\api\.venv\Scripts\python.exe'
    $sourceDataArguments = @('-X','utf8','-m','scripts.v23_source_data',$Action.ToLowerInvariant())
    if ($DataDirectory) { $sourceDataArguments += @('--data-directory',[IO.Path]::GetFullPath($DataDirectory).TrimEnd('\')) }
    if ($BackupDirectory) { $sourceDataArguments += @('--backup-directory',[IO.Path]::GetFullPath($BackupDirectory).TrimEnd('\')) }
    Push-Location (Join-Path $repoRoot 'V2.0\api')
    try {
        & $sourceDataPython @sourceDataArguments
        if ($LASTEXITCODE -ne 0) { throw 'Source data operation failed; the active data directory was not replaced.' }
    } finally { Pop-Location }
} finally {
    if ($ownsSourceDataMutex) { $sourceDataMutex.ReleaseMutex() }
    $sourceDataMutex.Dispose()
}
