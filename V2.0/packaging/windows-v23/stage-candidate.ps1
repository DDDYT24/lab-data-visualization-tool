[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][string]$PythonRuntimeRoot,
    [Parameter(Mandatory=$true)][string]$PythonSitePackagesRoot,
    [Parameter(Mandatory=$true)][string]$NodeExecutable,
    [Parameter(Mandatory=$true)][string]$VerificationReceipt,
    [Parameter(Mandatory=$true)][string]$BuildPython
)
$ErrorActionPreference='Stop'
& $BuildPython -B -X utf8 (Join-Path $PSScriptRoot 'stage_candidate.py') --output $OutputRoot --python-runtime $PythonRuntimeRoot --site-packages $PythonSitePackagesRoot --node $NodeExecutable --receipt $VerificationReceipt
if ($LASTEXITCODE -ne 0) { throw 'Independent V2.3 staging failed. Original source and data are retained.' }
