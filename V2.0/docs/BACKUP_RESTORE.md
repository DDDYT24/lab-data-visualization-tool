# 本地数据备份与恢复

## V2.2.0 Windows 安装版：备份与恢复

安装版的数据位于 `%LOCALAPPDATA%\LabViz\data`，与程序目录
`%LOCALAPPDATA%\Programs\LabViz` 分开。更新、修复和默认保留数据的卸载不主动删除历史。
源码版 `.labviz` 的备份命令见下文，两者不能混用。

### 备份

1. 从开始菜单执行 **Stop LabViz / 退出**，确认服务已停止。
2. 将整个 `%LOCALAPPDATA%\LabViz\data` 复制到你控制的备份目录。SQLite、对象文件和相关
   状态需要配套；不能只复制数据库。原始实验文件应另行归档。
3. 检查文件数量/大小，重要备份核对哈希，保留时间和软件版本标识。

```powershell
# 先停止 LabViz。备份目录不能是正在使用的数据目录。
$labvizBackup = Join-Path $env:USERPROFILE ('Documents\LabViz-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $labvizBackup | Out-Null
robocopy (Join-Path $env:LOCALAPPDATA 'LabViz\data') $labvizBackup /E /COPY:DAT /R:2 /W:2
if ($LASTEXITCODE -gt 7) { throw 'LabViz backup failed' }
```

### 恢复

保持服务停止，用同一 Windows 账户操作。先将当前 data 改名保存为带时间的恢复前副本，
再将完整备份复制到原 data 位置；不要把备份直接混入已有对象目录。
启动快捷方式后检查历史、数据预览和实际导出。`local-access.dpapi` 绑定 Windows 身份，
不得假定跨账户复制即可获得相同授权；本版本不提供跨账户迁移保证。

开始菜单 **Import old data / 迁移旧数据** 是将兼容旧 `.labviz` 复制到空安装版目录的维护入口，
它保留源目录和完整性校验备份；旧 guest/邮箱记录不会静默改归本地用户。
自动升级备份在 `%LOCALAPPDATA%\LabViz\backups`，不代替你单独保存的离线备份。
删除项目或卸载时选择“删除数据”属于永久删除；先备份后再决定。

## Historical source-checkout guidance


本文面向保存实验记录的用户和实验室管理员。默认本地部署的可恢复数据位于
`V2.0/api/.labviz/`，其中包含 SQLite 数据库和本地对象目录。

## 备份内容

停止 LabViz 后，备份整个目录：

```text
V2.0/api/.labviz/
├─ labviz-v2.db
└─ objects/
```

原始上传文件不会以原文件形式作为项目资产长期保存。如果实验室需要保留原始数据，请把
原始 CSV、TSV、TXT、JSON 或 XLSX 文件按自己的数据保留制度单独备份。

### Windows PowerShell

```powershell
# 请先关闭启动脚本窗口，确认网站和 API 已停止。
New-Item -ItemType Directory -Force -Path 'D:\LabViz-backups' | Out-Null
robocopy '.\V2.0\api\.labviz' 'D:\LabViz-backups\labviz-v2' /E /COPY:DAT /R:2 /W:2
if ($LASTEXITCODE -gt 7) { throw "备份失败，robocopy exit code: $LASTEXITCODE" }
```

### macOS / Linux

```bash
mkdir -p "$HOME/LabViz-backups"
cp -a V2.0/api/.labviz "$HOME/LabViz-backups/labviz-v2"
```

备份介质应使用实验室批准的磁盘加密和访问控制。不要把备份目录提交到 Git，也不要放入
自动公开同步的目录。

## 恢复步骤

1. 停止网站和 API。
2. 将现有 `V2.0/api/.labviz/` 改名为临时目录，以便需要时回退。
3. 把备份中的 `labviz-v2/` 恢复为 `V2.0/api/.labviz/`。
4. 确认当前用户对目录具有读写权限。
5. 按原来的启动脚本启动，并检查项目历史、图表和导出是否可读。

Windows 示例：

```powershell
Rename-Item '.\V2.0\api\.labviz' '.labviz.before-restore' -ErrorAction SilentlyContinue
robocopy 'D:\LabViz-backups\labviz-v2' '.\V2.0\api\.labviz' /E /COPY:DAT /R:2 /W:2
if ($LASTEXITCODE -gt 7) { throw "恢复失败，robocopy exit code: $LASTEXITCODE" }
```

恢复后若应用版本跨越了数据库迁移版本，先在 API 虚拟环境中运行项目规定的迁移命令，
再启动网站。不要把生产或他人电脑上的 `.env`、密钥和登录会话复制到共享目录。
