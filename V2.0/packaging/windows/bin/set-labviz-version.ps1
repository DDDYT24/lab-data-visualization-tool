[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$') {
    throw "Version must use a safe semantic version such as 2.2.0."
}
$installRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$versionRoot = Join-Path $installRoot "versions\$Version"
if (-not (Test-Path -LiteralPath (Join-Path $versionRoot "bin\start-labviz-portable.ps1") -PathType Leaf)) {
    throw "The requested LabViz version is not installed: $Version"
}
Set-Content -LiteralPath (Join-Path $installRoot "current-version.txt") -Value $Version -Encoding ascii
Write-Host "LabViz will use version $Version on the next launch. Local data was not changed." -ForegroundColor Green
