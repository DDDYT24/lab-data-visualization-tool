[CmdletBinding()]
param([ValidateSet('Prepare','Activate','Abort','Uninstall')][string]$Action,
 [Parameter(Mandatory=$true)][string]$InstallRoot,
 [Parameter(Mandatory=$true)][string]$ProfileBase,
 [Parameter(Mandatory=$true)][string]$SupportZip,
 [string]$PackageId,[switch]$SkipWebView2)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
# This is a dedicated maintenance process. Do not inherit incompatible or
# user-supplied module implementations from its launching shell.
$env:PSModulePath=Join-Path $PSHOME 'Modules'
$completed=$false
$temporary=$null
try {
    if($SkipWebView2){throw 'Component validation cannot be disabled in the installer.'}
    $temporary=Join-Path ([IO.Path]::GetTempPath()) ('LabVizDesktopSetup-'+[guid]::NewGuid().ToString('N'))
    $null=New-Item -ItemType Directory -Path $temporary
    Expand-Archive -LiteralPath $SupportZip -DestinationPath $temporary
    $payload=Join-Path $temporary 'app'
    $commonArguments=@('--install-root',$InstallRoot,'--profile-base',$ProfileBase,'--payload',$payload)
    $null=& (Join-Path $payload 'runtime\python\python.exe') '-B' '-X' 'utf8' (Join-Path $temporary 'runtime\installation.py') 'check' @commonArguments
    if($LASTEXITCODE -ne 0){throw 'The desktop destination or profile is unrelated. No component installer was run.'}
    if($Action -eq 'Prepare') {
        . (Join-Path $PSScriptRoot 'webview2.ps1')
        $component=Ensure-DesktopWebView2 -PolicyPath (Join-Path $PSScriptRoot 'webview2.json') -DownloadDirectory $temporary
        $component | ConvertTo-Json -Compress | Write-Output
    }
    $arguments=@('-B','-X','utf8',(Join-Path $temporary 'runtime\installation.py'),$Action.ToLowerInvariant(),'--install-root',$InstallRoot,'--profile-base',$ProfileBase,'--payload',$payload)
    if($PackageId){$arguments+=@('--package',$PackageId)}
    Remove-Item Env:PYTHONPATH,Env:PYTHONHOME,Env:PYTHONPYCACHEPREFIX -ErrorAction SilentlyContinue
    $result=& (Join-Path $payload 'runtime\python\python.exe') @arguments
    if($LASTEXITCODE -ne 0){throw 'Desktop maintenance failed. Your data and earlier program files were retained. Finish active work and retry setup.'}
    $result | Write-Output
    if($Action -eq 'Activate' -and (($result|ConvertFrom-Json).PSObject.Properties.Name -contains 'rolledBack')) {throw 'The new candidate failed. The previous candidate and data were restored. Run setup again to retry.'}
    $completed=$true
} catch { Write-Error $_;exit 1 }
finally {
    if($completed -and $temporary -and (Test-Path -LiteralPath $temporary)) {
        $resolved=[IO.Path]::GetFullPath($temporary)
        $expectedParent=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
        if([IO.Path]::GetDirectoryName($resolved) -ne $expectedParent -or [IO.Path]::GetFileName($resolved) -notmatch '^LabVizDesktopSetup-[a-f0-9]{32}$'){throw 'Unsafe temporary cleanup path refused.'}
        $links=@(Get-ChildItem -LiteralPath $resolved -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
        if($links.Count -eq 0){Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue}
    }
}
