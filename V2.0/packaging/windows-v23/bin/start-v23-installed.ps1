[CmdletBinding()]
param(
    [string]$ProfileRoot = (Join-Path $env:LOCALAPPDATA 'LabVizV23'),
    [ValidateRange(0,65535)][int]$WebPort = 3301,
    [ValidateRange(0,65535)][int]$ApiPort = 8301,
    [switch]$SkipOpenBrowser,
    [switch]$HealthCheckOnly,
    [switch]$Rollback
)
. (Join-Path $PSScriptRoot 'common.ps1')
$installRoot = Get-V23ProgramRoot (Join-Path $PSScriptRoot '..')
$currentFile = Join-Path $installRoot 'current-candidate.json'
$pendingFile = Join-Path $installRoot 'pending-candidate.json'
$journalFile = Join-Path $installRoot 'upgrade.json'
$receiptFile = Join-Path $installRoot 'last-upgrade.json'
$applicationMutex = New-Object Threading.Mutex($false, 'Local\LabVizV23-App')
$mutex = $null

function Read-Candidate([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return '' }
    $record = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($record.appId -ne $V23AppId -or $record.candidateId -notmatch '^[a-f0-9]{64}$') { throw 'Invalid installed candidate marker.' }
    return $record.candidateId
}
function Set-Candidate([string]$Path, [string]$Id) { Write-V23Json $Path @{appId=$V23AppId;candidateId=$Id} }
function Assert-OwnBackup([string]$Backup) {
    $absolute = Assert-Unlinked $Backup
    $parent = (Assert-Unlinked (Join-Path $ProfileRoot 'backups')) + '\'
    if (-not $absolute.StartsWith($parent,[StringComparison]::OrdinalIgnoreCase)) { throw 'Upgrade backup is outside this profile.' }
    return $absolute
}
function Undo-Transaction($Transaction, [string]$Helper) {
    if ($Transaction.appId -ne $V23AppId -or $Transaction.profileId -ne $profile.profileId) { throw 'Foreign upgrade transaction refused.' }
    if ($Transaction.backup) {
        $backup = Assert-OwnBackup $Transaction.backup
        $null = Invoke-V23Data $Helper 'replace' $ProfileRoot $backup
    }
    if ($Transaction.previous) { Set-Candidate $currentFile $Transaction.previous }
    elseif (Test-Path -LiteralPath $currentFile) { Remove-Item -LiteralPath $currentFile }
    if (Test-Path -LiteralPath $pendingFile) { Remove-Item -LiteralPath $pendingFile }
    if (Test-Path -LiteralPath $journalFile) { Remove-Item -LiteralPath $journalFile }
    Write-V23Json (Join-Path $installRoot 'failed-upgrade.json') $Transaction
}

try {
    $current = Read-Candidate $currentFile
    $pending = Read-Candidate $pendingFile
    $selected = if ($pending) { $pending } else { $current }
    if (-not $selected) { throw 'No installed candidate is available.' }
    $versionRoot = Get-V23VersionRoot $installRoot $selected
    $profile = Invoke-V23Data $versionRoot 'init' $ProfileRoot
    $ProfileRoot = $profile.root
    try { $mutex = Enter-V23Mutex (Get-V23MutexName $ProfileRoot) }
    catch {
        if ($Rollback -or $HealthCheckOnly -or $pending) { throw }
        if (-not $SkipOpenBrowser) {
            Add-Type -AssemblyName System.Security
            $running = Get-Content -LiteralPath (Join-Path $ProfileRoot 'logs\running.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($running.appId -ne $V23AppId -or $running.profileId -ne $profile.profileId) { throw 'Running identity mismatch.' }
            $web = Assert-V23Process $running.web; $null = Assert-V23Process $running.api
            $uri = [Uri]$running.webUrl
            if ($uri.Scheme -ne 'http' -or $uri.Host -ne '127.0.0.1' -or $uri.AbsolutePath -ne '/') { throw 'Invalid running loopback URL.' }
            Assert-V23PortOwner $uri.Port $web
            $raw = [Security.Cryptography.ProtectedData]::Unprotect([IO.File]::ReadAllBytes((Join-Path $ProfileRoot 'access.dpapi')),$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
            $key = [Convert]::ToBase64String($raw).TrimEnd('=').Replace('+','-').Replace('/','_')
            Start-Process "$($uri.GetLeftPart([UriPartial]::Authority))/#labviz-access=$key"
        }
        Write-Host 'LabViz V2.3 is already running / 已在运行。'
        return
    }
    $null = Invoke-V23Data $versionRoot 'recover' $ProfileRoot
    if (Test-Path -LiteralPath $journalFile) {
        $transaction = Get-Content -LiteralPath $journalFile -Raw -Encoding UTF8 | ConvertFrom-Json
        Undo-Transaction $transaction $versionRoot
        $current = Read-Candidate $currentFile
        if (-not $current) { throw 'Interrupted first activation recovered. Reinstall the candidate to retry.' }
        $versionRoot = Get-V23VersionRoot $installRoot $current
        $pending = ''
    }
    if ($Rollback) {
        $receipt = Get-Content -LiteralPath $receiptFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($receipt.appId -ne $V23AppId -or $receipt.profileId -ne $profile.profileId -or -not $receipt.previous -or -not $receipt.backup) { throw 'No matching rollback snapshot is available.' }
        $pending = $receipt.previous
        $versionRoot = Get-V23VersionRoot $installRoot $pending
        $null = Assert-OwnBackup $receipt.backup
    }
    if ($pending) {
        $profile = Invoke-V23Data $versionRoot 'info' $ProfileRoot
        if ($profile.dataEpoch -gt 1) { throw 'Automatic rollback to an incompatible data schema is refused. Use a compatible runtime; current data is unchanged.' }
        $backup = ''
        if (Test-Path -LiteralPath (Join-Path $ProfileRoot 'data\labviz.db')) { $backup = (Invoke-V23Data $versionRoot 'backup' $ProfileRoot).backupDirectory }
        $transaction = @{ appId=$V23AppId; profileId=$profile.profileId; previous=$current; candidate=$pending; backup=$backup }
        Write-V23Json $journalFile $transaction
        try {
            if ($Rollback) { $null = Invoke-V23Data $versionRoot 'replace' $ProfileRoot $receipt.backup }
            & (Join-Path $versionRoot 'bin\start-v23-portable.ps1') -InstallRoot $installRoot -ProfileRoot $ProfileRoot -WebPort $WebPort -ApiPort $ApiPort -SkipOpenBrowser -HealthCheckOnly
            Set-Candidate $currentFile $pending
            Write-V23Json $receiptFile $transaction
            if (Test-Path -LiteralPath $pendingFile) { Remove-Item -LiteralPath $pendingFile }
            Remove-Item -LiteralPath $journalFile
        } catch {
            $failure = $_
            Undo-Transaction ([pscustomobject]$transaction) $versionRoot
            if (-not $current) { throw $failure }
            $versionRoot = Get-V23VersionRoot $installRoot $current
            Write-Host 'Activation failed; previous candidate and verified data restored / 已恢复旧候选和数据。'
        }
    }
    & (Join-Path $versionRoot 'bin\start-v23-portable.ps1') -InstallRoot $installRoot -ProfileRoot $ProfileRoot -WebPort $WebPort -ApiPort $ApiPort -SkipOpenBrowser:$SkipOpenBrowser -HealthCheckOnly:$HealthCheckOnly
} finally {
    if ($mutex) { $mutex.ReleaseMutex(); $mutex.Dispose() }
    $applicationMutex.Dispose()
}
