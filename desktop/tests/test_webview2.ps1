param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskOutput=[IO.Path]::GetFullPath([IO.Path]::Combine($root,$OutputRoot))
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'outputs'))+[IO.Path]::DirectorySeparatorChar
if(-not $taskOutput.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Use a dedicated output below this checkout outputs/.'}
if(Test-Path -LiteralPath $taskOutput){throw 'Use a fresh component-test output; existing evidence is preserved.'}
$parent=[IO.DirectoryInfo]$taskOutput
while($parent){if($parent.Exists -and ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Linked component-test outputs are not supported.'};$parent=$parent.Parent}
[IO.Directory]::CreateDirectory($taskOutput) | Out-Null
. (Join-Path $root 'desktop\packaging\webview2.ps1')
$policy=Join-Path $root 'desktop\packaging\webview2.json'
$directory=Join-Path $taskOutput 'downloads'
$results=@()
function Check([string]$Name,[scriptblock]$Test) { & $Test; $script:results+=@{name=$Name;passed=$true} }
function MustFail([scriptblock]$Test) { $failed=$false;try{& $Test}catch{$failed=$true};if(-not $failed){throw 'Expected failure was accepted.'} }
Check 'already-installed-does-not-download' {
 $result=Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {[version]'140.0.0.0'} -Download {throw 'Unexpected download'}
 if($result.downloadedBootstrapperBytes -ne 0){throw 'Existing runtime downloaded bytes'}
}
Check 'missing-runtime-rechecks-after-signed-install' {
 $script:detected=0
 $result=Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {$script:detected++;if($script:detected -gt 1){[version]'140.0.0.0'}} -Download {param($u,$p)[IO.File]::WriteAllBytes($p,[byte[]](1,2,3))} -CheckSignature {param($p)if(-not(Test-Path -LiteralPath $p)){throw 'Missing signed fixture'}} -Install {0}
 if($result.downloadedBootstrapperBytes -ne 3 -or $script:detected -ne 2){throw 'Installation was not rechecked'}
}
Check 'untrusted-signature-never-runs-installer' {
 MustFail {Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {$null} -Download {param($u,$p)[IO.File]::WriteAllBytes($p,[byte[]](1))} -CheckSignature {throw 'Invalid publisher'} -Install {throw 'Installer must not run'}}
}
Check 'download-failure-never-runs-installer' {
 MustFail {Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {$null} -Download {throw 'Offline'} -Install {throw 'Installer must not run'}}
}
Check 'installer-failure-is-retryable' {
 MustFail {Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {[version]'110.0.0.0'} -Download {param($u,$p)[IO.File]::WriteAllBytes($p,[byte[]](1))} -CheckSignature {} -Install {1603}}
}
Check 'success-code-without-runtime-is-not-accepted' {
 MustFail {Ensure-DesktopWebView2 -PolicyPath $policy -DownloadDirectory $directory -Detect {$null} -Download {param($u,$p)[IO.File]::WriteAllBytes($p,[byte[]](1))} -CheckSignature {} -Install {0}}
}
$actual=Get-DesktopWebView2Version
@{passed=$true;checks=$results;actualVersion=if($actual){$actual.ToString()}else{$null};scope='Read installed registry; mocked download/signature/install paths. No system component was installed.'} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskOutput 'component-verification.json') -Encoding UTF8
$results | ConvertTo-Json -Compress
