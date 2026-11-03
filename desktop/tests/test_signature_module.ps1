[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Helper,
 [Parameter(Mandatory=$true)][string]$TargetFile,
 [Parameter(Mandatory=$true)][string]$OutputRoot,
 [Parameter(Mandatory=$true)][string]$PolicyPath,
 [ValidateSet('Inspect','Shadow','Ensure','EnsureShadow')][string]$Mode='Inspect')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$boundary=(Join-Path $workspace 'outputs').TrimEnd('\')+'\'
$folder=[IO.Path]::GetFullPath($OutputRoot)
if(-not $folder.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $folder)){throw 'Use a fresh private signature test output'}
$null=New-Item -ItemType Directory -Path $folder
. $Helper
if($Mode -in @('Shadow','EnsureShadow')) {
    $fake=Join-Path $folder 'shadow\Microsoft.PowerShell.Security\99.0.0'
    $null=New-Item -ItemType Directory -Path $fake -Force
    @'
function Get-AuthenticodeSignature {
 param([string]$LiteralPath)
 [pscustomobject]@{Status='Valid';SignerCertificate=[pscustomobject]@{Subject='CN=Microsoft Corporation'}}
}
Export-ModuleMember -Function Get-AuthenticodeSignature
'@ | Set-Content -LiteralPath (Join-Path $fake 'shadow.psm1') -Encoding UTF8
    "@{RootModule='shadow.psm1';ModuleVersion='99.0.0';FunctionsToExport=@('Get-AuthenticodeSignature')}" | Set-Content -LiteralPath (Join-Path $fake 'Microsoft.PowerShell.Security.psd1') -Encoding UTF8
    $null=Import-Module -Name (Join-Path $fake 'Microsoft.PowerShell.Security.psd1') -Force
}
$script:InstallCalls=0;$script:Detected=$false
$result=[ordered]@{success=$false;mode=$Mode;engineMajor=$PSVersionTable.PSVersion.Major;componentExecuted=$false;installSimulated=($Mode -like 'Ensure*');installCalls=0;errorClass=$null}
try {
    if($Mode -like 'Ensure*') {
        $null=Ensure-DesktopWebView2 -PolicyPath $PolicyPath -DownloadDirectory (Join-Path $folder 'download') `
            -Detect {if($script:Detected){return [version]'999.0.0.0'};return $null} `
            -Download {param($Url,$Destination) Copy-Item -LiteralPath $TargetFile -Destination $Destination} `
            -Install {param($File) $script:InstallCalls++;$script:Detected=$true;return 0}
    } else {Assert-DesktopMicrosoftSignature $TargetFile}
    $result.success=$true
} catch {
    $result.errorClass=if($_.Exception.Message -like '*signature is invalid*'){'SignatureRejected'}elseif($_.FullyQualifiedErrorId -like '*Autoload*' -or $_.Exception.Message -like '*Microsoft.PowerShell.Security*'){'ModuleLoadFailure'}else{'OtherFailure'}
}
$result.installCalls=$script:InstallCalls
$mods=@(Get-Module -Name Microsoft.PowerShell.Security)
$trustedPath=Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1'
$result.trustedSecurityModuleLoaded=(@($mods | Where-Object {$_.Path -eq $trustedPath -and $_.ExportedCmdlets.ContainsKey('Get-AuthenticodeSignature')}).Count -gt 0)
$result | ConvertTo-Json -Compress
