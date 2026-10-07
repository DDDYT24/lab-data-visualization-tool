[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$WebPort = 3000,
    [ValidateRange(1, 65535)][int]$ApiPort = 8000,
    [switch]$SkipOpenBrowser,
    [switch]$HealthCheckOnly,
    [string]$ImportData
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$installRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$currentFile = Join-Path $installRoot "current-version.txt"
$pendingFile = Join-Path $installRoot "pending-version.txt"
$journalFile = Join-Path $installRoot "upgrade.json"
$receiptFile = Join-Path $installRoot "last-upgrade.json"
$userRoot = Join-Path $env:LOCALAPPDATA "LabViz"
$dataRoot = Join-Path $userRoot "data"
$logRoot = Join-Path $userRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$mutex = New-Object Threading.Mutex($false, ("Local\LabViz-" + [Security.Principal.WindowsIdentity]::GetCurrent().User.Value))
$ownsMutex = $false
try { $ownsMutex = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
if (-not $ownsMutex) {
    $mutex.Dispose()
    if ($ImportData) { throw "Stop LabViz before importing / 请先退出 LabViz。" }
    $running = Join-Path $logRoot "running.json"
    if ((Test-Path -LiteralPath $running) -and -not $SkipOpenBrowser) {
        $url = (Get-Content -LiteralPath $running -Raw | ConvertFrom-Json).webUrl
        if ($url -notmatch '^http://127\.0\.0\.1:[0-9]+$') {
            throw "The running LabViz address is not a local loopback URL."
        }
        $tokenPath = Join-Path $dataRoot "local-access.dpapi"
        if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
            throw "Local launch credential is missing; stop and restart LabViz / 缺少本地启动凭据，请完全退出后重启 LabViz。"
        }
        Add-Type -AssemblyName System.Security
        $protectedBytes = [IO.File]::ReadAllBytes($tokenPath)
        $accessBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $protectedBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        $accessKey = [Convert]::ToBase64String($accessBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        Start-Process "$url/#labviz-access=$accessKey"
    }
    Write-Host "LabViz is already running or starting / LabViz 已在运行或正在启动。"
    return
}
function Get-VersionRoot([string]$Version) {
    if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$') {
        throw "Invalid version / 版本标记无效。"
    }
    $root = Join-Path $installRoot "versions\$Version"
    if (-not (Test-Path -LiteralPath (Join-Path $root "bin\start-labviz-portable.ps1"))) {
        throw "Incomplete installation; reinstall LabViz / 安装不完整，请重新安装。"
    }
    return $root
}
function Write-AtomicFile([string]$Path, [string]$Value, [Text.Encoding]$Encoding) {
    $temporary = "$Path.new-" + [guid]::NewGuid().ToString("N")
    if (Test-Path -LiteralPath $Path) {
        # File.Replace requires a real same-volume backup path on Windows.
        # Keep that backup beside the marker until the replacement succeeds.
        $backup = "$Path.replace-" + [guid]::NewGuid().ToString("N")
        try {
            [IO.File]::WriteAllText($temporary, $Value, $Encoding)
            [IO.File]::Replace($temporary, $Path, $backup)
        } finally {
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            }
            if (Test-Path -LiteralPath $backup) {
                Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
            }
        }
    } else {
        try {
            [IO.File]::WriteAllText($temporary, $Value, $Encoding)
            [IO.File]::Move($temporary, $Path)
        } finally {
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
function Write-Marker([string]$Path, [string]$Value) {
    Write-AtomicFile $Path $Value ([Text.Encoding]::ASCII)
}
function Write-Transaction([string]$Path, $Transaction) {
    Write-AtomicFile $Path ($Transaction | ConvertTo-Json -Depth 5) ([Text.Encoding]::UTF8)
}
function Invoke-DataTool([string]$Root, [string]$Action, [string]$Source, [string]$Destination) {
    & (Join-Path $Root "runtime\python\python.exe") (Join-Path $PSScriptRoot "local-data.py") $Action $Source $Destination
    if ($LASTEXITCODE -ne 0) { throw "Data operation failed; originals retained / 数据操作失败，原数据已保留。" }
}
function Restore-Transaction($Transaction) {
    $backupRoot = [IO.Path]::GetFullPath((Join-Path $userRoot "backups")) + "\"
    if ($Transaction.backup -and -not ([IO.Path]::GetFullPath($Transaction.backup).StartsWith($backupRoot, [StringComparison]::OrdinalIgnoreCase))) {
        throw "Invalid backup path / 备份路径无效。"
    }
    $runtimeRoot = Get-VersionRoot $Transaction.candidate
    if ($Transaction.backup) {
        Invoke-DataTool $runtimeRoot "restore" $Transaction.backup $dataRoot
    } elseif (Test-Path -LiteralPath $dataRoot) {
        # Preserve candidate-created data without exposing it to the previous runtime.
        Move-Item -LiteralPath $dataRoot -Destination (Join-Path $userRoot ("data.failed-" + [guid]::NewGuid().ToString("N")))
    }
    if ($Transaction.previous) { Write-Marker $currentFile $Transaction.previous }
    if (Test-Path -LiteralPath $pendingFile) { Remove-Item -LiteralPath $pendingFile }
    Remove-Item -LiteralPath $journalFile
    Write-Host "Upgrade restored; failed data retained / 已恢复升级前状态，失败版本的数据副本已保留。"
}
try {
    # A crash between snapshot and successful activation is recovered before another start.
    if (Test-Path -LiteralPath $journalFile) {
        Restore-Transaction (Get-Content -LiteralPath $journalFile -Raw | ConvertFrom-Json)
    }
    $previous = if (Test-Path -LiteralPath $currentFile) { (Get-Content -LiteralPath $currentFile -Raw).Trim() } else { "" }
    $version = if (Test-Path -LiteralPath $pendingFile) { (Get-Content -LiteralPath $pendingFile -Raw).Trim() } else { $previous }
    $versionRoot = Get-VersionRoot $version
    if ($ImportData) {
        Invoke-DataTool $versionRoot "import" $ImportData $dataRoot
        Write-Host "Data copied; old account/guest projects remain private and require re-import for account-free history / 数据已复制；旧账户和访客项目仍受原有访问限制，需重新导入才会进入免登录历史。"
        return
    }
    $arguments = @{ WebPort = $WebPort; ApiPort = $ApiPort; SkipOpenBrowser = $true; HealthCheckOnly = $true }
    if (Test-Path -LiteralPath $pendingFile) {
        $backup = ""
        if (Test-Path -LiteralPath (Join-Path $dataRoot "labviz-v2.db")) {
            $backup = Join-Path $userRoot ("backups\upgrade-" + [guid]::NewGuid().ToString("N"))
            Invoke-DataTool $versionRoot "snapshot" $dataRoot $backup
        } elseif ((Test-Path -LiteralPath $dataRoot) -and @(Get-ChildItem -LiteralPath $dataRoot -Force).Count -gt 0) {
            throw "Unrecognized local data; backup and repair first / 本地数据状态异常，请先备份并修复。"
        }
        $transaction = @{ previous = $previous; candidate = $version; backup = $backup }
        Write-Transaction $journalFile $transaction
        try {
            if ($previous -and ([version]($version.Split('-')[0]) -lt [version]($previous.Split('-')[0]))) {
                if (-not (Test-Path -LiteralPath $receiptFile)) { throw "Rollback snapshot unavailable / 缺少回滚快照。" }
                $receipt = Get-Content -LiteralPath $receiptFile -Raw | ConvertFrom-Json
                if ($receipt.previous -ne $version -or -not $receipt.backup) { throw "No matching rollback snapshot / 无匹配回滚快照。" }
                Invoke-DataTool $versionRoot "restore" $receipt.backup $dataRoot
            }
            & (Join-Path $versionRoot "bin\start-labviz-portable.ps1") @arguments
            Write-Marker $currentFile $version
            Write-Transaction $receiptFile $transaction
            Remove-Item -LiteralPath $pendingFile
            Remove-Item -LiteralPath $journalFile
        } catch {
            $failure = $_
            Restore-Transaction ([pscustomobject]$transaction)
            if (-not $previous) { throw $failure }
            $versionRoot = Get-VersionRoot $previous
        }
    }
    $arguments.HealthCheckOnly = [bool]$HealthCheckOnly
    $arguments.SkipOpenBrowser = [bool]$SkipOpenBrowser
    & (Join-Path $versionRoot "bin\start-labviz-portable.ps1") @arguments
} catch {
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $logRoot "launcher.error.log")
    Write-Host "LabViz could not start / LabViz 启动失败。Logs / 日志: $logRoot" -ForegroundColor Red
    throw
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
