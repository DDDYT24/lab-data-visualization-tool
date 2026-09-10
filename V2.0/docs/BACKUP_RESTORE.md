# 本地数据备份与恢复

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
