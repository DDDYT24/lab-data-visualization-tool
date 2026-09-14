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

$installRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$versionsRoot = Join-Path $installRoot "versions"
$currentVersionFile = Join-Path $installRoot "current-version.txt"

function Get-VersionDirectory {
    param([string]$Version)

    if ([string]::IsNullOrWhiteSpace($Version) -or $Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$') {
        throw "The installed LabViz version marker is invalid."
    }
    $candidate = Join-Path $versionsRoot $Version
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "The selected LabViz version is not installed: $Version"
    }
    return $candidate
}

if (Test-Path -LiteralPath $currentVersionFile -PathType Leaf) {
    $version = (Get-Content -LiteralPath $currentVersionFile -Raw).Trim()
} else {
    $version = @(
        Get-ChildItem -LiteralPath $versionsRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1 -ExpandProperty Name
    ) | Select-Object -First 1
}

$versionRoot = Get-VersionDirectory -Version $version
$portableLauncher = Join-Path $versionRoot "bin\start-labviz-portable.ps1"
if (-not (Test-Path -LiteralPath $portableLauncher -PathType Leaf)) {
    throw "The selected LabViz version is incomplete: $versionRoot"
}

$portableArguments = @{
    WebPort = $WebPort
    ApiPort = $ApiPort
}
if ($SkipOpenBrowser) {
    $portableArguments.SkipOpenBrowser = $true
}
& $portableLauncher @portableArguments
exit $LASTEXITCODE
