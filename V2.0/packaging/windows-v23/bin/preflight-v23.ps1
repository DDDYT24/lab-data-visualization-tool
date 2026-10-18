[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$InstallRoot,[switch]$RequireIdentity,[switch]$ScanTree,[string]$CandidateId)
. (Join-Path $PSScriptRoot 'common.ps1')
$root = Assert-V23Destination $InstallRoot -RequireIdentity:$RequireIdentity
if ($CandidateId) {
    if ($CandidateId -notmatch '^[a-f0-9]{64}$') { throw 'Invalid full candidate identity.' }
    if ($root.Length + 10 + 24 + 1 + 121 -ge 259) { throw 'Choose a shorter installation directory for Windows path compatibility.' }
    $slot = Join-Path $root ('versions\' + $CandidateId.Substring(0,24))
    if (Test-Path -LiteralPath $slot) {
        $existing = Get-Content -LiteralPath (Join-Path $slot 'candidate-build.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($existing.appId -ne $V23AppId -or $existing.candidateId -ne $CandidateId) { throw 'A different or incomplete candidate already occupies this version directory.' }
    }
}
if ($ScanTree -and (Test-Path -LiteralPath $root)) {
    $pending = New-Object 'Collections.Generic.Stack[string]'
    $pending.Push($root)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or -not $item.FullName.StartsWith("$root\",[StringComparison]::OrdinalIgnoreCase)) { throw 'A linked or external installation entry was refused.' }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
        }
    }
}
Write-Host 'V2.3 installation path and identity verified.'
