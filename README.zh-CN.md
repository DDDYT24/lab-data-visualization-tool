<div align="center">

# 🧪 LabViz 实验数据可视化工具

**把实验表格转换为清晰、可导出的科研图表，全程在本地运行。**

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
![Release](https://img.shields.io/badge/release-V2.1%20local-0f766e)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [隐私边界](V2.0/docs/PRIVACY_DATA_BOUNDARY.md) · [离线安装](V2.0/docs/OFFLINE_INSTALL.md) · [问题反馈](https://github.com/DDDYT24/lab-data-visualization-tool/issues)

</div>

LabViz V2.1.1 是当前本地发布版本，由 Next.js 网站和 FastAPI 科研处理
服务组成。它支持数据导入、质量检查、清洗、绘图、项目历史、本地验证码登录、
只读分享，以及 PNG、SVG、PDF 科研图像导出。

本项目不提供官方在线网站。普通使用者不需要购买域名，也不需要 AWS、DNS、
Docker、PostgreSQL、MinIO 或付费邮件服务。

## 快速开始 V2.1.1

V2.1 继续使用 `V2.0/` 作为代码目录。先点击下面的官方下载链接安装三个工具：

- [Git for Windows](https://git-scm.com/download/win)；macOS/Linux 使用 [Git downloads](https://git-scm.com/downloads)
- [Python 3.12 或 3.13](https://www.python.org/downloads/)
- [Node.js 22.22.2 或更新版本](https://nodejs.org/en/download)

### Windows

请在准备存放项目的目录中打开 PowerShell，然后按顺序运行：

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
.\start-labviz.cmd
```

如果已经在 `lab-data-visualization-tool` 仓库目录内，跳过前两行即可。第一次启动会自动创建
`V2.0/api/.venv`，安装 API 和网站依赖，然后启动两个本地服务。打开
`http://127.0.0.1:3000` 即可使用。

### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
chmod +x start-labviz.sh
./start-labviz.sh
```

启动完成后打开：

- LabViz 网站：`http://127.0.0.1:3000`
- API 文档：`http://127.0.0.1:8000/docs`

保持终端窗口开启；六位本地登录验证码会显示在 API 输出中，不会发送真实邮件。按
`Ctrl+C` 可停止网站和 API。

依赖文件更新后可强制刷新：

```powershell
.\start-labviz.cmd -RefreshDependencies
```

```bash
./start-labviz.sh --refresh-dependencies
```

## 使用流程

```text
导入数据 → 检查质量 → 确认清洗方式 → 创建图表 → 导出结果
```

打开网站后可点击“使用示例数据”，也可以上传 CSV、TSV、分隔符 TXT、JSON 或
XLSX 文件。网页端单文件上限为 50 MB。

主要功能包括：

- 缺失值、重复行、字段类型、唯一值和内存占用检查；
- 删除重复行，以及保留、删除、前向填充、后向填充、均值填充或中位数填充；
- 折线图、散点图、柱状图、直方图、箱线图、相关性热力图和 3D 曲面图；
- 完整清洗数据 CSV，以及 PNG、SVG、PDF 图像导出；
- 本地项目历史、描述修订、本地登录和只读分享链接。

## 本地数据保存在哪里

| 组件 | 默认行为 |
| --- | --- |
| SQLite | 自动创建于 `V2.0/api/.labviz/labviz-v2.db`，保存项目元数据、会话、历史和修订 |
| 本地对象目录 | 自动创建于 `V2.0/api/.labviz/`，保存本地处理结果 |
| 原始上传文件 | 不作为原文件长期保存；解析后的数据快照会留在本地以支持历史、清洗和导出 |
| 登录验证码 | 显示在本地 API 终端，不发送邮件 |

`.labviz` 运行数据已被 Git 忽略，不会随正常提交上传到 GitHub。公开分享仓库前，
仍请确认没有手动添加实验数据、密钥、个人路径或本机生成的 `outputs/` 文件。

## 哪些组件不是本地运行必需的

| 组件 | 普通本地使用 | 适用情况 |
| --- | --- | --- |
| PostgreSQL | 不需要 | 多 API/Worker 进程或多人共享服务 |
| MinIO / S3 | 不需要 | 共享或远程对象存储 |
| Docker | 不需要 | 贡献者完整集成测试或容器部署 |
| AWS、DNS、TLS、SES | 不需要 | 维护者主动建设公网服务时 |

请保持默认服务绑定在 `127.0.0.1`。如果没有配置 HTTPS、安全 Cookie、生产邮件、
备份、监控和多用户存储，不要把端口 `3000` 或 `8000` 暴露到公网。

更多说明：[`隐私边界`](V2.0/docs/PRIVACY_DATA_BOUNDARY.md) ·
[`离线安装`](V2.0/docs/OFFLINE_INSTALL.md) · [`备份与恢复`](V2.0/docs/BACKUP_RESTORE.md)。

贡献代码请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，安全问题请阅读
[`SECURITY.md`](SECURITY.md)，版本变化见 [`CHANGELOG.md`](CHANGELOG.md)，维护者发布前可参阅
[`公开发布清单`](V2.0/docs/PUBLIC_RELEASE_CHECKLIST.md)。

## 手动启动

如果需要分别查看两个服务的输出，可使用两个终端。

终端一（API）：

```powershell
Set-Location .\V2.0\api
# 推荐 Python 3.12；如果未安装 3.12，请将下一行的 py -3.12 改为 py -3.13。
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn labviz_api.main:app --reload --host 127.0.0.1 --port 8000
```

终端二（网站）：

```powershell
Set-Location .\V2.0\web
$env:NEXT_TELEMETRY_DISABLED = "1"
npm ci
npm run dev -- --hostname 127.0.0.1 --port 3000
```

## 常见问题

- **`Set-Location` 找不到路径：** 该命令假设当前目录是克隆目标的父目录。如果提示符已经显示在
  `lab-data-visualization-tool` 内，请跳过它；否则使用实际路径，例如
  `Set-Location -LiteralPath 'C:\Users\你的用户名\lab-data-visualization-tool'`，再用
  `Test-Path .\start-labviz.cmd` 确认返回 `True`。
- **出现 `No suitable Python runtime found`：** 启动器支持 Python 3.12 或 3.13，暂不支持
  Python 3.14。运行 `py -0p` 和 `py -3.13 --version` 检查版本；如果 3.13 可以运行但旧启动器
  仍失败，请更新到最新仓库版本。

- **找不到 Python 3.12/3.13：** 安装支持的 Python 后重新打开终端；Windows 可用
  `py -0p` 查看已安装版本。
- **Node.js 版本过低：** 运行 `node --version`；升级到 22.22.2 或更新版本。
- **端口已占用：** 关闭正在使用 `3000` 或 `8000` 的程序，再重新启动 LabViz。
- **修改依赖后运行异常：** 使用上文的刷新依赖命令。
- **预览提示 API 不可用：** 检查启动窗口中 API 是否成功监听
  `http://127.0.0.1:8000`，然后点击“重试预览”。

## 开发与验证

网站验证：

```powershell
Set-Location .\V2.0\web
$env:NEXT_TELEMETRY_DISABLED = "1"
npm ci
npm run verify
npm run test:e2e
```

API 的完整验证需要 PostgreSQL 和 MinIO 测试容器，详细命令见
[`V2.0/api/README.md`](V2.0/api/README.md)。普通本地使用不需要这些容器。

## V1.1 旧版

[`V1.1/`](V1.1/) 中保留了只需 Python 的 Streamlit 旧版，适合只需要基础绘图和
命令行流程的使用者；新功能和主要维护以 V2.0 为准。

## MIT 许可证

LabViz 使用 [MIT 许可证](LICENSE)发布。任何人都可以使用、复制、修改、分发、
再许可或销售本软件，包括商业用途；条件是保留版权和许可证声明。本软件不提供
任何担保。
