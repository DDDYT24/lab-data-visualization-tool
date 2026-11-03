Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Get-DesktopWebView2Version {
    $found=@()
    foreach($hive in @([Microsoft.Win32.RegistryHive]::CurrentUser,[Microsoft.Win32.RegistryHive]::LocalMachine)) {
        foreach($view in @([Microsoft.Win32.RegistryView]::Registry32,[Microsoft.Win32.RegistryView]::Registry64)) {
            $base=$null;$key=$null
            try {
                $base=[Microsoft.Win32.RegistryKey]::OpenBaseKey($hive,$view)
                $key=$base.OpenSubKey('Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}')
                if($key) { $value=[version]$key.GetValue('pv');if($value -gt [version]'0.0.0.0'){$found+=$value} }
            } catch { } finally {if($key){$key.Dispose()};if($base){$base.Dispose()}}
        }
    }
    return ($found | Sort-Object -Descending | Select-Object -First 1)
}

function Assert-DesktopMicrosoftSignature([string]$Path) {
    # Bind the cmdlet to this engine's trusted module, even when a PowerShell 7
    # parent passed its module search path to Windows PowerShell 5.1.
    $modulePath=Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1'
    $security=Import-Module -Name $modulePath -Force -PassThru
    $signature=& $security.ExportedCmdlets['Get-AuthenticodeSignature'] -LiteralPath $Path
    if($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or $signature.SignerCertificate.Subject -notmatch '(^|,\s*)CN=Microsoft Corporation(,|$)') {
        throw 'The WebView2 installer signature is invalid. No component installer was run.'
    }
}

function Ensure-DesktopWebView2 {
    param([string]$PolicyPath,[string]$DownloadDirectory,
        [scriptblock]$Detect={Get-DesktopWebView2Version},
        [scriptblock]$Download={param($Url,$Path) Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Path},
        [scriptblock]$CheckSignature={param($Path) Assert-DesktopMicrosoftSignature $Path},
        [scriptblock]$Install={param($Path) $process=Start-Process -FilePath $Path -ArgumentList '/silent','/install' -WindowStyle Hidden -Wait -PassThru;return $process.ExitCode})
    $policy=Get-Content -LiteralPath $PolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if($policy.format -ne 'labviz-desktop-webview2-v1' -or $policy.bootstrapperUrl -ne 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'){throw 'Invalid WebView2 installation policy.'}
    $version=& $Detect
    if($version -and [version]$version -ge [version]$policy.minimumVersion){return @{installed=$true;downloadedBootstrapperBytes=0;version=$version.ToString();runtimePayloadDownloadBytes=$null}}
    $directory=[IO.Path]::GetFullPath($DownloadDirectory)
    $null=New-Item -ItemType Directory -Path $directory -Force
    $file=Join-Path $directory ('WebView2-'+[guid]::NewGuid().ToString('N')+'.exe')
    & $Download $policy.bootstrapperUrl $file
    & $CheckSignature $file
    $bytes=(Get-Item -LiteralPath $file).Length
    $code=& $Install $file
    if($code -notin @(0,3010)){throw "WebView2 setup failed ($code). Check the connection and select Retry in setup."}
    $version=& $Detect
    if(-not $version -or [version]$version -lt [version]$policy.minimumVersion){throw 'WebView2 is unavailable after setup. Restart Windows if requested, then retry setup.'}
    return @{installed=$true;downloadedBootstrapperBytes=$bytes;runtimePayloadDownloadBytes=$null;version=$version.ToString();exitCode=$code}
}
