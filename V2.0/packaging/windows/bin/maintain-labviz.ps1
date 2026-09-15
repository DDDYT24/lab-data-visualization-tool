[CmdletBinding()]
param([ValidateSet('Import', 'Stop', 'Logs', 'Rollback')][string]$Action = 'Import')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$userRoot = Join-Path $env:LOCALAPPDATA 'LabViz'
try {
    switch ($Action) {
        'Logs' {
            $logs = Join-Path $userRoot 'logs'
            New-Item -ItemType Directory -Force -Path $logs | Out-Null
            Start-Process explorer.exe -ArgumentList ('"{0}"' -f $logs)
        }
        'Stop' {
            $logs = Join-Path $userRoot 'logs'
            New-Item -ItemType Directory -Force -Path $logs | Out-Null
            [IO.File]::WriteAllText((Join-Path $logs 'stop.request'), 'stop')
        }
        'Import' {
            $answer = [Windows.Forms.MessageBox]::Show('Close the old LabViz app first. Select its V2.0\api\.labviz folder. Source files stay unchanged. Import requires an empty installed data directory. Browser sign-in may be required again. / 请先退出旧版 LabViz，再选择旧版 V2.0\api\.labviz 目录。保留原数据，仅导入到空数据目录，可能需重新登录。', 'LabViz', 'OKCancel')
            if ($answer -ne 'OK') { return }
            $dialog = New-Object Windows.Forms.FolderBrowserDialog
            $dialog.Description = 'Select old LabViz data / 选择旧版 .labviz 数据目录'
            if ($dialog.ShowDialog() -ne 'OK') { return }
            & (Join-Path $PSScriptRoot 'start-labviz-installed.ps1') -ImportData $dialog.SelectedPath
            [Windows.Forms.MessageBox]::Show('Import complete. Source and backup retained. / 迁移完成，原数据及备份已保留。', 'LabViz') | Out-Null
        }
        'Rollback' {
            $receipt = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\last-upgrade.json') -Raw | ConvertFrom-Json
            if (-not $receipt.previous -or -not $receipt.backup) { throw 'No rollback snapshot / 无可用回滚快照。' }
            $answer = [Windows.Forms.MessageBox]::Show('Close LabViz first. Rollback restores the previous data snapshot; newer data is retained separately. / 请先退出 LabViz。回滚将恢复升级前的数据，新数据另存保留。', 'LabViz', 'OKCancel')
            if ($answer -ne 'OK') { return }
            & (Join-Path $PSScriptRoot 'set-labviz-version.ps1') -Version $receipt.previous
            & (Join-Path $PSScriptRoot 'start-labviz-installed.ps1')
        }
    }
} catch {
    [Windows.Forms.MessageBox]::Show($_.Exception.Message, 'LabViz', 'OK', 'Error') | Out-Null
    exit 1
}
