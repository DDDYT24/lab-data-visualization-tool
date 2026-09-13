[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,
    [switch]$RequireBundledRuntimes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
$manifestPath = Join-Path $resolvedRoot "package-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "package-manifest.json is missing from the candidate package."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.contractVersion -ne "v1") {
    throw "Unsupported package contract version: $($manifest.contractVersion)"
}
if ($manifest.localFirst -ne $true) {
    throw "The Windows package must remain local-first."
}

$requiredFiles = @(
    "package-manifest.json",
    "bin\start-labviz-portable.cmd",
    "bin\start-labviz-portable.ps1",
    "V2.0\assets\labviz-logo.ico",
    "V2.0\assets\labviz-logo.svg"
)
$missing = @(
    $requiredFiles | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $resolvedRoot $_) -PathType Leaf)
    }
)
if ($missing.Count -gt 0) {
    throw "Candidate package is missing: $($missing -join ', ')"
}

$forbiddenPatterns = @(
    "(^|[\\/])\.env(?:\.|$)",
    "(^|[\\/])\.labviz(?:[\\/]|$)",
    "(^|[\\/])\.venv(?:[\\/]|$)",
    "(^|[\\/])node_modules(?:[\\/]|$)",
    "(^|[\\/])\.next[\\/]cache(?:[\\/]|$)",
    "(^|[\\/])outputs(?:[\\/]|$)",
    "(^|[\\/])playwright-report(?:[\\/]|$)",
    "(^|[\\/])test-results(?:[\\/]|$)"
)
$violations = @(
    Get-ChildItem -LiteralPath $resolvedRoot -Recurse -Force -File |
        ForEach-Object {
            $relative = $_.FullName.Substring($resolvedRoot.Length).TrimStart('\\', '/')
            foreach ($pattern in $forbiddenPatterns) {
                if ($relative -match $pattern) {
                    [pscustomobject]@{ Path = $relative; Rule = $pattern }
                    break
                }
            }
        }
)
if ($violations.Count -gt 0) {
    $paths = $violations | ForEach-Object { $_.Path }
    throw "Candidate package contains prohibited runtime or user-data paths: $($paths -join ', ')"
}

if ($RequireBundledRuntimes) {
    $runtimeFiles = @(
        "runtime\python\python.exe",
        "runtime\node\node.exe",
        "V2.0\web\server.js"
    )
    $runtimeMissing = @(
        $runtimeFiles | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $resolvedRoot $_) -PathType Leaf)
        }
    )
    if ($runtimeMissing.Count -gt 0) {
        throw "Bundled runtime verification requested, but these files are missing: $($runtimeMissing -join ', ')"
    }
}

Write-Host "LabViz Windows package contract passed: $resolvedRoot" -ForegroundColor Green
if (-not $RequireBundledRuntimes) {
    Write-Host "Runtime bundle check was not requested; this validates metadata and data boundaries only." -ForegroundColor Yellow
}
