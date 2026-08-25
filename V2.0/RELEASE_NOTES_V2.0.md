# LabViz V2.0 / 本地自托管正式版

LabViz V2.0 is the first supported release of the Next.js and FastAPI workflow.
It is distributed as a local self-hosted application and does not require an AWS
account, domain, DNS, Docker, external database, object-storage server, or paid
email provider for ordinary use.

LabViz V2.0 是 Next.js 与 FastAPI 工作流的首个正式支持版本。它以本地自托管
应用发布；普通使用不需要 AWS 账号、域名、DNS、Docker、外部数据库、对象存储
服务器或付费邮件服务。

## Highlights / 主要内容

- One-command launchers for Windows, macOS, and Linux.
- CSV, TSV, delimited TXT, JSON, and XLSX imports up to 50 MB.
- Explained quality checks and explicit cleaning decisions.
- Seven 2D/3D chart types with PNG, SVG, and PDF exports.
- Local SQLite history, project descriptions, email-code sessions, and read-only sharing.
- Bilingual English and Simplified Chinese documentation and interface.
- Windows、macOS 和 Linux 一键启动脚本。
- 支持最大 50 MB 的 CSV、TSV、分隔符 TXT、JSON 和 XLSX 文件。
- 可解释的数据质量检查和明确的清洗决策。
- 七类 2D/3D 图表，以及 PNG、SVG、PDF 导出。
- 本地 SQLite 历史、项目描述、验证码会话和只读分享。
- 英文与简体中文文档及界面。

## Start / 启动

Windows:

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location .\lab-data-visualization-tool
.\start-labviz.cmd
```

macOS / Linux:

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
chmod +x start-labviz.sh
./start-labviz.sh
```

Open `http://127.0.0.1:3000`. See [`README.md`](../README.md) or
[`README.zh-CN.md`](../README.zh-CN.md) for prerequisites, manual startup, data
storage, and troubleshooting.

## License / 许可证

MIT. See [`LICENSE`](../LICENSE).
