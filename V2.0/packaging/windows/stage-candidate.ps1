[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonRuntimeRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonSitePackagesRoot,
    [Parameter(Mandatory = $true)]
    [string]$NodeExecutable,
    [switch]$BuildWeb,
    [switch]$RequirePython313
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$webRoot = Join-Path $repoRoot "V2.0\web"
$apiRoot = Join-Path $repoRoot "V2.0\api"
$packagingRoot = $PSScriptRoot

function Resolve-ExistingDirectory {
    param([string]$Path, [string]$Label)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $resolved -or -not (Test-Path -LiteralPath $resolved.Path -PathType Container)) {
        throw "$Label does not exist or is not a directory: $Path"
    }
    return $resolved.Path
}

function Resolve-ExistingFile {
    param([string]$Path, [string]$Label)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $resolved -or -not (Test-Path -LiteralPath $resolved.Path -PathType Leaf)) {
        throw "$Label does not exist or is not a file: $Path"
    }
    return $resolved.Path
}

function Copy-DirectoryContents {
    param([string]$Source, [string]$Destination)

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

function Copy-PythonRuntime {
    param([string]$Source, [string]$Destination)

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force |
        Where-Object { $_.Name -ine "Lib" } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
        }

    $sourceLib = Join-Path $Source "Lib"
    $destinationLib = Join-Path $Destination "Lib"
    New-Item -ItemType Directory -Force -Path $destinationLib | Out-Null
    Get-ChildItem -LiteralPath $sourceLib -Force |
        Where-Object { $_.Name -ine "site-packages" } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $destinationLib -Recurse -Force
        }
}

function Copy-RelativeEntry {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot,
        [string]$RelativePath
    )

    $source = Join-Path $SourceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required source is missing: $source"
    }
    $destination = Join-Path $DestinationRoot $RelativePath
    $sourceItem = Get-Item -LiteralPath $source
    if ($sourceItem.PSIsContainer) {
        Copy-DirectoryContents -Source $sourceItem.FullName -Destination $destination
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $sourceItem.FullName -Destination $destination -Force
    }
}

$pythonRoot = Resolve-ExistingDirectory -Path $PythonRuntimeRoot -Label "Python runtime root"
$pythonSitePackages = Resolve-ExistingDirectory -Path $PythonSitePackagesRoot -Label "Python site-packages root"
$nodePath = Resolve-ExistingFile -Path $NodeExecutable -Label "Node executable"
$nestedNodeModules = @(
    Get-ChildItem -LiteralPath $pythonSitePackages -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ieq "node_modules" }
)
if ($nestedNodeModules.Count -gt 0) {
    $paths = $nestedNodeModules | ForEach-Object { $_.FullName }
    throw "Python site-packages contains unrelated node_modules and cannot be bundled: $($paths -join ', ')"
}
$pythonExecutable = Join-Path $pythonRoot "python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Python runtime root must contain python.exe: $pythonExecutable"
}

$pythonVersion = (& $pythonExecutable -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not query the supplied Python runtime: $pythonExecutable"
}
$pythonParts = $pythonVersion.Split('.')
$pythonMajorMinor = "$($pythonParts[0]).$($pythonParts[1])"
if (@("3.12", "3.13") -notcontains $pythonMajorMinor) {
    throw "The package runtime must be Python 3.12 or 3.13; received $pythonVersion."
}
if ($RequirePython313 -and $pythonMajorMinor -ne "3.13") {
    throw "The release packaging gate requires Python 3.13.x; received $pythonVersion."
}

$nodeVersionText = (& $nodePath --version).Trim().TrimStart("v")
try {
    $nodeVersion = [version]$nodeVersionText
} catch {
    throw "Could not parse the supplied Node.js version: $nodeVersionText"
}
if ($nodeVersion -lt [version]"22.22.2") {
    throw "The package runtime requires Node.js 22.22.2 or newer; received $nodeVersionText."
}

$outputAbsolute = [IO.Path]::GetFullPath($OutputRoot)
$protectedRoots = @(
    [IO.Path]::GetFullPath((Join-Path $repoRoot "V1.1")),
    [IO.Path]::GetFullPath((Join-Path $repoRoot "V2.0"))
)
foreach ($protectedRoot in $protectedRoots) {
    if ($outputAbsolute.Equals($protectedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $outputAbsolute.StartsWith("$protectedRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "OutputRoot must not be inside a source tree: $outputAbsolute"
    }
}
$existingOutput = Get-Item -LiteralPath $outputAbsolute -ErrorAction SilentlyContinue
if ($existingOutput) {
    if (-not $existingOutput.PSIsContainer) {
        throw "OutputRoot must be a directory: $outputAbsolute"
    }
    $existingEntries = @(Get-ChildItem -LiteralPath $outputAbsolute -Force)
    if ($existingEntries.Count -gt 0) {
        throw "OutputRoot must be new or empty to prevent stale files: $outputAbsolute"
    }
} else {
    New-Item -ItemType Directory -Force -Path $outputAbsolute | Out-Null
}

if ($BuildWeb) {
    $npm = Join-Path (Split-Path -Parent $nodePath) "npm.cmd"
    if (-not (Test-Path -LiteralPath $npm -PathType Leaf)) {
        throw "-BuildWeb requires npm.cmd beside the supplied node.exe: $npm"
    }
    $previousBuildFlag = $env:LABVIZ_E2E_BUILD
    try {
        Remove-Item Env:LABVIZ_E2E_BUILD -ErrorAction SilentlyContinue
        Push-Location $webRoot
        try {
            & $npm run build
            if ($LASTEXITCODE -ne 0) {
                throw "Next.js standalone build failed."
            }
        } finally {
            Pop-Location
        }
    } finally {
        if ($null -eq $previousBuildFlag) {
            Remove-Item Env:LABVIZ_E2E_BUILD -ErrorAction SilentlyContinue
        } else {
            $env:LABVIZ_E2E_BUILD = $previousBuildFlag
        }
    }
}

$standaloneWebRoot = Join-Path $webRoot ".next\standalone\web"
$webStaticRoot = Join-Path $webRoot ".next\static"
if (-not (Test-Path -LiteralPath (Join-Path $standaloneWebRoot "server.js") -PathType Leaf)) {
    throw "Next.js standalone output is missing. Run 'npm run build' in V2.0/web first."
}
if (-not (Test-Path -LiteralPath $webStaticRoot -PathType Container)) {
    throw "Next.js static output is missing: $webStaticRoot"
}

Copy-Item -LiteralPath (Join-Path $packagingRoot "package-manifest.json") -Destination (Join-Path $outputAbsolute "package-manifest.json") -Force
Copy-DirectoryContents -Source $standaloneWebRoot -Destination (Join-Path $outputAbsolute "V2.0\web")
Copy-DirectoryContents -Source $webStaticRoot -Destination (Join-Path $outputAbsolute "V2.0\web\.next\static")

$apiEntries = @(
    "labviz_api",
    "migrations",
    "samples\v22",
    "alembic.ini",
    "requirements.txt",
    "pyproject.toml"
)
foreach ($entry in $apiEntries) {
    Copy-RelativeEntry -SourceRoot $apiRoot -DestinationRoot (Join-Path $outputAbsolute "V2.0\api") -RelativePath $entry
}

Copy-RelativeEntry -SourceRoot $repoRoot -DestinationRoot $outputAbsolute -RelativePath "README.md"
Copy-RelativeEntry -SourceRoot $repoRoot -DestinationRoot $outputAbsolute -RelativePath "README.zh-CN.md"
Copy-RelativeEntry -SourceRoot $repoRoot -DestinationRoot $outputAbsolute -RelativePath "LICENSE"
Copy-RelativeEntry -SourceRoot $repoRoot -DestinationRoot $outputAbsolute -RelativePath "V2.0\assets\labviz-logo.ico"
Copy-RelativeEntry -SourceRoot $repoRoot -DestinationRoot $outputAbsolute -RelativePath "V2.0\assets\labviz-logo.svg"
Copy-RelativeEntry -SourceRoot $packagingRoot -DestinationRoot $outputAbsolute -RelativePath "bin\start-labviz-portable.cmd"
Copy-RelativeEntry -SourceRoot $packagingRoot -DestinationRoot $outputAbsolute -RelativePath "bin\start-labviz-portable.ps1"

Copy-PythonRuntime -Source $pythonRoot -Destination (Join-Path $outputAbsolute "runtime\python")
Copy-DirectoryContents -Source $pythonSitePackages -Destination (Join-Path $outputAbsolute "runtime\python\Lib\site-packages")
New-Item -ItemType Directory -Force -Path (Join-Path $outputAbsolute "runtime\node") | Out-Null
Copy-Item -LiteralPath $nodePath -Destination (Join-Path $outputAbsolute "runtime\node\node.exe") -Force

$metadata = [ordered]@{
    contractVersion = "v1"
    stagedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    pythonVersion = $pythonVersion
    nodeVersion = $nodeVersionText
    nextStandalone = $true
    runtimeDependencyInstall = $false
    networkRequiredAfterInstall = $false
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputAbsolute "candidate-build.json") -Encoding utf8

& (Join-Path $packagingRoot "validate-package.ps1") -PackageRoot $outputAbsolute -RequireBundledRuntimes
Write-Host "LabViz Windows candidate staged and validated: $outputAbsolute" -ForegroundColor Green
Write-Host "This is a staged portable candidate, not a native installer or clean-machine acceptance result." -ForegroundColor Yellow
