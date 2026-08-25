<div align="center">

# 🧪 Lab Data Visualization Tool

**Turn raw experiment tables into clear, exportable visuals — locally.**

从导入、检查、清洗到 2D/3D 可视化与导出，一套面向实验数据的轻量工作流。

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
![Release](https://img.shields.io/badge/release-V2.0-0f766e)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776ab?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-22.22.2%2B-339933?logo=node.js&logoColor=white)

[English](README.md) · [简体中文](README.zh-CN.md) · [Quick Start](#-quick-start--v20) · [Features](#-features) · [Development](#-development-and-verification)

</div>

Lab Data Visualization Tool is a local-first web application for loading, validating, cleaning,
visualizing, and exporting experimental data. V2.0 combines a guided Next.js interface with a
FastAPI scientific-processing service. Raw uploads stay on the local machine and are not written
to the project database.

> **Current release: V2.0** — supports CSV, TSV, delimited TXT, JSON, and XLSX data; guided
> quality review; seven 2D/3D chart types; saved history; local sign-in; sharing; and PNG/SVG/PDF
> publication exports.

> **Distribution model:** this repository does not operate an official hosted website. Clone it
> from GitHub and run it on your own computer. The default local paths require no AWS account,
> domain, DNS, paid email provider, Docker, PostgreSQL server, or MinIO server.

## Repository Versions

| Version | Status | Location |
| --- | --- | --- |
| V2.0 | Current local self-hosted Next.js + FastAPI release | [`V2.0/`](V2.0/) |
| V1.1 | Legacy Python-only Streamlit application | [`V1.1/`](V1.1/) |

Repository-wide automation and documentation remain at the root. Deferred V2.0 work is tracked in [`V2.0/TODO.md`](V2.0/TODO.md).

V2.0 needs Python and Node.js, but its default SQLite database and local object directory are
created automatically. V1.1 remains available when a smaller Python-only legacy interface is
preferred.

## ⚡ Quick Start — V2.0

Install [Git](https://git-scm.com/downloads), Python 3.12 or 3.13, and Node.js 22.22.2 or newer.
The first launch creates the Python environment and installs all required packages.

### Windows PowerShell

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location .\lab-data-visualization-tool
.\start-labviz.cmd
```

### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
chmod +x start-labviz.sh
./start-labviz.sh
```

Open `http://127.0.0.1:3000`. Keep the terminal open; local sign-in codes appear in the API
output. Press `Ctrl+C` to stop both services. Run `start-labviz.cmd -RefreshDependencies` on
Windows or `./start-labviz.sh --refresh-dependencies` on macOS/Linux after dependency files change.

```text
Load data  →  Inspect quality  →  Clean safely  →  Visualize  →  Export
```

### Manual startup

V2.0 includes a runnable FastAPI contract/reference service for frontend
development and end-to-end testing. It accepts CSV, TSV, delimited
TXT, JSON, and XLSX uploads; builds bounded previews and explained quality
findings; records user cleaning decisions; renders PNG, SVG, and PDF figures;
and supports temporary projects, email-code sign-in, saved history, authenticated plain-text
project descriptions, and read-only revision-pinned share links. Raw uploaded bytes are processed in memory and are not
written to the SQLite project database.

V2.0 is distributed for local self-hosting. Its default single-computer route
uses SQLite plus local object storage. PostgreSQL 17,
S3-compatible storage, workers, and the AWS deployment files remain available
for advanced multi-process deployments and integration testing, but they are
not required to run the application locally.

Prerequisites: Python 3.12 or 3.13 and Node.js 22.22.2 or newer. Python 3.12
and Node.js 24 are used by CI and are the recommended development versions.

Start the API in the first PowerShell window:

```powershell
Set-Location .\V2.0\api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn labviz_api.main:app --reload --port 8000
```

Start the website in a second PowerShell window:

```powershell
Set-Location .\V2.0\web
npm ci
npm run dev
```

Open `http://localhost:3000` and select **Try sample data**, or upload a real
table. The Next.js development server proxies `/api/v1` to
`http://127.0.0.1:8000` by default. API documentation is available at
`http://localhost:8000/docs`.

Local development uses console email delivery: request a sign-in code in the
website, then copy the six-digit code printed in the API terminal. No email is
sent and no email account is required in this mode. SMTP and Amazon SES adapters
are retained only for maintainers who choose to build an externally hosted
deployment.
For a separately hosted API, set `NEXT_PUBLIC_LABVIZ_API_URL` before starting
Next.js and include the website origin in `LABVIZ_ALLOWED_ORIGINS`.

### What local users do and do not need

| Component | Default local use | When it becomes useful |
| --- | --- | --- |
| SQLite | Automatically included with Python | Stores project metadata, sessions, history, and revisions locally |
| Local object directory | Automatically created under `V2.0/api/.labviz/` | Stores local processed artifacts |
| PostgreSQL | Not required | Multiple API/worker processes or a shared self-hosted server |
| MinIO / S3 | Not required | Shared or remote object storage |
| Docker | Not required | Contributor integration tests or container-based deployment |
| AWS, DNS, TLS, SES | Not required | Only if a maintainer intentionally publishes an Internet-facing service |

Keep the default services bound to `127.0.0.1`. Do not expose ports 3000 or
8000 to the Internet without adding HTTPS, secure cookies, production email,
backups, monitoring, and an appropriate multi-user persistence configuration.

## ✨ Features

| Stage | Capabilities |
| --- | --- |
| Load | CSV, TSV, delimited TXT, JSON, and XLSX; up to 50 MB in the web app |
| Inspect | Row and column counts, data types, missing cells, duplicates, unique values, and memory use |
| Clean | Exact-duplicate removal; keep, drop, forward-fill, backward-fill, mean-fill, or median-fill missing values |
| Visualize | Line, scatter, bar, histogram, box, correlation heatmap, and 3D surface plots |
| Export | Complete cleaned table as CSV and publication figures as PNG, SVG, or PDF |
| Track | Compact SQLite plot history without storing raw table contents |
| Scale | Deterministic preview limits and evenly spaced sampling for large plots |

Large previews are limited to 200 rows. Plots above the selected threshold use an evenly spaced sample, while the cleaned CSV download always contains every remaining row.

For a 3D surface, select three numeric columns representing X, Y, and Z. The points must span a surface rather than a single straight line.

## 🧱 Architecture

```text
V1.1/app.py                  Streamlit front end
V1.1/main.py                 CLI compatibility entry point
V1.1/labviz/core.py          Loading, validation, cleaning, and profiling
V1.1/labviz/plotting.py      Validated 2D/3D rendering and bounded sampling
V1.1/labviz/database.py      SQLite metadata/history repository
V1.1/labviz/cli.py           Scriptable command-line workflow
V1.1/tests/                  Core, plotting, database, CLI, and UI smoke tests
V2.0/prd.md                  Approved product and Figma prototype requirements
V2.0/PROJECT_PLAN.md         Website-first architecture, boundaries, and phases
V2.0/web/                    Next.js, TypeScript, MUI, and ECharts frontend
V2.0/api/                    FastAPI, SQLite/PostgreSQL persistence, local/S3 storage, and Matplotlib
V2.0/TODO.md                 Prototype checkpoints and explicitly deferred work
```

When V1.1 is launched from its version directory, the SQLite file is created at `V1.1/.labviz/history.db` and is ignored by Git. Set `LABVIZ_DB_PATH` to use another location. Only the filename, SHA-256 fingerprint, size, quality counts, and plot configuration are recorded.

## 🧰 Supported Development Environment

| Area | Supported and verified environment |
| --- | --- |
| Operating system | Designed for Windows, macOS, and Linux. CI verifies `ubuntu-latest`; V1.1 was also verified locally on Windows. |
| Python | Python 3.12 is verified by CI; Python 3.13 is supported. V2.0 intentionally excludes Python 3.14 until its scientific stack is verified. |
| Node.js | Node.js 24 is verified by CI for the V2.0 frontend; minimum supported version is 22.22.2. |
| Hardware | CPU-only; no GPU or external database server is required. |
| Web runtime | A modern browser; V1.1 uses port `8501`; V2.0 uses website port `3000` and API port `8000`. |

### Runtime Dependencies

Install these from `V1.1/requirements.txt` when you only need to use the application:

| Dependency | Supported range | Purpose |
| --- | --- | --- |
| Streamlit | `>=1.49,<2` | Web interface, uploads, controls, caching, and downloads |
| pandas | `>=2.2,<3` | Tabular loading, cleaning, profiling, and CSV export |
| NumPy | `>=1.26,<3` | Numeric arrays, deterministic sampling, and plotting support |
| Matplotlib | `>=3.9,<4` | 2D/3D figure rendering and PNG export |
| openpyxl | `>=3.1,<4` | XLSX import support |

SQLite is provided by Python's standard library, so no database package or service is required.

### Development and CI Dependencies

Contributors should install both `V1.1/requirements.txt` and `V1.1/requirements-dev.txt`:

| Dependency | Supported range | Purpose |
| --- | --- | --- |
| pytest | `>=8.3,<9` | Automated tests |
| pytest-cov | `>=5,<7` | Coverage reporting |
| pre-commit | `>=3.8,<5` | Repository checks before commits |
| mypy | `>=1.11,<2` | Static type checking |
| Ruff | `==0.6.9` | Linting and formatting; pinned to match the pre-commit hook and CI |

Runtime packages use bounded version ranges so compatible updates can be installed without silently crossing a major-version boundary. Upgrade a major version separately and run the complete validation suite before changing these bounds.

### V2.0 Frontend Dependencies

The frontend lockfile is authoritative. Its core stack is Next.js 16, React 19, TypeScript 5.9, Material UI 9, MUI X Data Grid Community, Apache ECharts 6, next-intl, Zustand, Zod, Vitest, and ESLint 9. ESLint intentionally remains on the latest compatible 9.x release because the React lint rules used by the current Next.js configuration are not yet compatible with ESLint 10.

### V2.0 API Dependencies and Configuration

Install `V2.0/api/requirements.txt` for runtime use. The service uses FastAPI,
Uvicorn, Pydantic, pandas, NumPy, Matplotlib, openpyxl, PostgreSQL support, and
Python's built-in SQLite driver. Install `requirements-dev.txt` for pytest,
Ruff, and mypy. The committed `requirements.lock.txt` and
`requirements-dev.lock.txt` are hash-pinned Python 3.12 installs used for
reproducible checks and CI; the range files remain their update inputs.

The API defaults are suitable for one local development process: SQLite data
is stored in `V2.0/api/.labviz/labviz-v2.db`, temporary projects and exports
expire two hours after the last project operation, uploads are limited to 50
MB, and email codes are printed only in the API terminal. Relevant variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LABVIZ_DATABASE_PATH` | `V2.0/api/.labviz/labviz-v2.db` | SQLite database location |
| `LABVIZ_ALLOWED_ORIGINS` | localhost and 127.0.0.1 on port 3000 | Comma-separated CORS origins |
| `LABVIZ_PUBLIC_WEB_URL` | `http://localhost:3000` | Base URL for generated share links |
| `LABVIZ_MAX_UPLOAD_BYTES` | `52428800` | Cloud upload limit in bytes |
| `LABVIZ_AUTH_MODE` | `console` | `console` locally, `smtp` for non-production compatibility, or production `ses` |
| `LABVIZ_AUTH_RATE_LIMIT_WINDOW_SECONDS` | `3600` | Fixed authentication limit window; allowed range is 60–86400 seconds |
| `LABVIZ_AUTH_CLIENT_REQUEST_LIMIT` | `30` | Maximum requests per keyed client identity and window |
| `LABVIZ_AUTH_EMAIL_REQUEST_LIMIT` | `10` | Maximum requests per normalized email and window |
| `LABVIZ_SMTP_HOST`, `LABVIZ_SMTP_PORT` | unset, `587` | SMTP connection |
| `LABVIZ_SMTP_USERNAME`, `LABVIZ_SMTP_PASSWORD` | unset | Optional SMTP credentials |
| `LABVIZ_SMTP_FROM` | `LabViz <noreply@localhost>` | Sender displayed in verification emails |
| `LABVIZ_SES_REGION` | unset | Required production SES Region; must match the S3 Region |
| `LABVIZ_SES_FROM` | unset | Required verified production sender |
| `LABVIZ_SES_CONFIGURATION_SET` | unset | Required delivery/feedback configuration set |
| `LABVIZ_COOKIE_SECURE` | `false` | Set to `true` behind production HTTPS |

Verification challenges, rate-limit records, and login sessions are persisted with expiry. The
SQLite reference adapter serializes limiter writes with `BEGIN IMMEDIATE`; production PostgreSQL
uses database-time fixed-window buckets and row locks to atomically enforce both client and email
limits across API hosts. Limiter storage failures return 503 and never send a verification code.

## 💻 Command Line

```bash
python main.py \
  --input experiment.csv \
  --x time \
  --y temperature pressure \
  --type line \
  --missing median \
  --out results/experiment.png \
  --cleaned-out results/experiment_cleaned.csv \
  --summary
```

3D example:

```bash
python main.py -i surface.csv -x x -y y z --type surface3d -o surface.png
```

Run `python main.py --help` for every option.

## 🛠 Troubleshooting

### `No such file or directory: 'requirements.txt'`

The command is running outside the V1.1 application directory. In PowerShell, enter it and confirm the file exists:

```powershell
Set-Location .\lab-data-visualization-tool\V1.1
Test-Path .\requirements.txt
```

`Test-Path` must return `True` before installing dependencies.

### `streamlit` is not recognized

This usually means dependency installation did not finish, or `streamlit` belongs to a different Python environment. From the repository directory, run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Using `python -m streamlit` guarantees that Streamlit is started by the currently active Python environment.

### `Cache entry deserialization failed, entry ignored`

This is a pip cache warning, not the cause of a missing `requirements.txt`. Pip ignores that cache entry and continues. If it repeats frequently, clear only pip's download cache and retry:

```powershell
python -m pip cache purge
python -m pip install -r requirements.txt
```

### PowerShell blocks environment activation

Allow scripts only for the current PowerShell process, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 🧪 Development and Verification

Run these commands from the repository root:

```bash
cd V1.1
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
ruff check app.py main.py labviz tests
ruff format --check app.py main.py labviz tests
mypy app.py main.py labviz
pytest -q --cov
```

CI runs the same lint, format, type, and test checks on Python 3.12.

Verify the V2.0 frontend separately:

```bash
cd V2.0/web
npm ci
npm run verify
npm run test:e2e
```

The first command runs ESLint, strict TypeScript checking, Vitest, and a production Next.js build. The second runs the Playwright desktop and mobile product flows, approved-breakpoint screenshot regression, and serious/critical WCAG checks. CI repeats both on Node.js 24.

Verify the V2.0 API from its own Python 3.12 environment:

```powershell
cd V2.0/api
python -m pip install -r requirements-dev.lock.txt
docker compose up -d --wait postgres minio
docker compose run --rm minio-init
ruff check labviz_api tests migrations scripts
ruff format --check labviz_api tests migrations scripts
mypy labviz_api tests migrations scripts
python -m pytest
docker compose down -v --remove-orphans
```

The API integration suite exercises real CSV processing, data-quality
findings, cleaning decisions, PNG/SVG/PDF generation, email-code sessions,
history, sharing permissions, MinIO-backed object storage, and stable error
responses. CI runs the same checks on Python 3.12; the Compose services are
required for the full integration suite.

## License

LabViz is released under the [MIT License](LICENSE). You may use, copy, modify,
distribute, sublicense, or sell the software, including in commercial projects,
provided that the copyright and license notice remain included. The software is
provided without warranty.

## 📚 Design References

- The large-file preview, data-quality summary, and numeric-column guidance were informed by [The-Schultz-Lab/plottle](https://github.com/The-Schultz-Lab/plottle) (MIT).
- The README's strong project statement, visible status badges, and quick-start-first hierarchy were inspired by [msitarzewski/agency-agents](https://github.com/msitarzewski/agency-agents).

This project keeps an intentionally smaller scope and contains an independent implementation. It does not include source code or copied documentation from either reference project.

---

## 中文说明

这是一个轻量、本地运行的实验数据处理与可视化工具，面向需要“上传数据后直接检查、清洗、绘图并导出”的实验者。项目不提供官方在线网站，用户从 GitHub 克隆后在自己的电脑上运行；实验原始数据不会写入历史数据库。

V2.0 是当前正式发布的本地自托管版本，由 Next.js 前端和 FastAPI 科研处理服务组成，
提供真实的数据处理、工作区、历史、分享、登录和导出接口。V2.0 默认使用
SQLite 和本地对象目录，二者都会自动创建；不需要 AWS、域名、DNS、Docker、
PostgreSQL、MinIO 或付费邮件服务。原始上传文件只在处理期间保留于内存，不写入
SQLite。V1.1 作为仅需 Python 的旧版界面继续保留。

### 🚀 快速开始

请先安装 [Git](https://git-scm.com/downloads)、Python 3.12 或 3.13，以及 Node.js
22.22.2 或更新版本。第一次启动会自动创建 Python 虚拟环境并安装所需依赖。

#### Windows PowerShell

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location .\lab-data-visualization-tool
.\start-labviz.cmd
```

#### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
chmod +x start-labviz.sh
./start-labviz.sh
```

打开 `http://127.0.0.1:3000`。请保持终端窗口开启；本地登录验证码会显示在 API
输出中。按 `Ctrl+C` 可同时停止网站和 API。依赖文件变化后，Windows 可运行
`.\start-labviz.cmd -RefreshDependencies`，macOS/Linux 可运行
`./start-labviz.sh --refresh-dependencies`。

#### 手动启动 V2.0 网站与 API

请安装 Python 3.12（也支持 3.13）以及 Node.js 22.22.2 或更新版本，推荐使用
Node.js 24。先在第一个 PowerShell 窗口启动 API：

```powershell
Set-Location .\V2.0\api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn labviz_api.main:app --reload --port 8000
```

再在第二个 PowerShell 窗口启动网站：

```powershell
Set-Location .\V2.0\web
npm ci
npm run dev
```

打开 `http://localhost:3000`。网站默认把 `/api/v1` 代理到
`http://127.0.0.1:8000`，接口文档位于 `http://localhost:8000/docs`。点击
“使用示例数据”会请求服务端明确提供的示例项目；也可以上传真实表格。开发模式
下邮箱验证码会显示在 API 终端，接口响应不会返回验证码。

#### 本地部署需要哪些组件

| 组件 | 普通本地使用 | 什么时候才需要 |
| --- | --- | --- |
| SQLite | Python 自带、自动创建 | 本地保存项目元数据、会话、历史和版本 |
| 本地对象目录 | 自动创建在 `V2.0/api/.labviz/` | 保存本地处理结果 |
| PostgreSQL | 不需要 | 多个 API/Worker 进程或多人共享服务器 |
| MinIO / S3 | 不需要 | 共享或远程对象存储 |
| Docker | 不需要 | 贡献者完整集成测试或容器部署 |
| AWS、DNS、TLS、SES | 不需要 | 维护者主动建设公网服务时才需要 |

普通用户只需保持网站和 API 绑定在 `127.0.0.1`。如果没有配置 HTTPS、安全
Cookie、生产邮件、备份、监控和多用户数据库，请不要把 3000 或 8000 端口暴露到
公网。

### ✨ 核心能力

- 支持 CSV、TSV、分隔符 TXT、JSON 和 XLSX，网页端单文件上限为 50 MB。
- 自动统计行列数、字段类型、缺失值、重复行、唯一值和内存占用。
- 支持删除重复行，以及保留、删除、前向填充、后向填充、均值填充和中位数填充。
- 支持折线图、散点图、柱状图、直方图、箱线图、相关性热力图和 3D 曲面模型。
- 可下载完整清洗数据，以及 PNG、SVG 或 PDF 图像。
- SQLite 只记录绘图历史和数据质量摘要，不保存实验原始数据。
- 大数据绘图使用等距抽样，避免浏览器卡顿；数据下载仍保留全部清洗结果。

### 🧰 支持的开发环境

| 项目 | 支持与验证情况 |
| --- | --- |
| 操作系统 | 设计上支持 Windows、macOS 和 Linux；CI 验证 `ubuntu-latest`，V1.1 也已在 Windows 本地验证。 |
| Python | CI 验证 Python 3.12，同时支持 3.13；V2.0 暂不支持尚未完成科学计算依赖验证的 Python 3.14。 |
| Node.js | V2.0 前端最低支持 22.22.2；CI 使用 Node.js 24。 |
| 硬件 | 仅需 CPU，不需要 GPU，也不需要外部数据库服务器。 |
| 网页运行 | 现代浏览器；V1.1 使用端口 `8501`；V2.0 网站使用 `3000`，API 使用 `8000`。 |

#### 运行依赖

普通用户只需要安装 `V1.1/requirements.txt`：

| 依赖 | 支持范围 | 用途 |
| --- | --- | --- |
| Streamlit | `>=1.49,<2` | 网页界面、上传、交互控件、缓存和下载 |
| pandas | `>=2.2,<3` | 表格读取、清洗、质量分析和 CSV 导出 |
| NumPy | `>=1.26,<3` | 数值数组、确定性抽样和绘图支持 |
| Matplotlib | `>=3.9,<4` | 2D/3D 绘图和 PNG 导出 |
| openpyxl | `>=3.1,<4` | XLSX 文件读取 |

SQLite 来自 Python 标准库，因此不需要额外安装数据库软件或 Python 数据库包。

#### 开发与 CI 依赖

开发者需要同时安装 `V1.1/requirements.txt` 和 `V1.1/requirements-dev.txt`：

| 依赖 | 支持范围 | 用途 |
| --- | --- | --- |
| pytest | `>=8.3,<9` | 自动化测试 |
| pytest-cov | `>=5,<7` | 测试覆盖率 |
| pre-commit | `>=3.8,<5` | 提交前检查 |
| mypy | `>=1.11,<2` | 静态类型检查 |
| Ruff | `==0.6.9` | 代码检查和格式化；固定版本以确保本地、pre-commit 与 CI 一致 |

运行依赖采用带主版本上限的范围，能够获取兼容更新，同时避免自动跨越可能带来破坏性变更的主版本。升级主版本时应单独修改并运行完整验证。

#### V2.0 API 依赖与默认行为

运行服务安装 `V2.0/api/requirements.txt`；开发和测试安装
`V2.0/api/requirements-dev.txt`。核心依赖包括 FastAPI、Uvicorn、Pydantic、
pandas、NumPy、Matplotlib、openpyxl 和 PostgreSQL 支持，SQLite 由 Python
自带。提交的 `requirements.lock.txt` 与 `requirements-dev.lock.txt` 是带哈希的
Python 3.12 锁文件，CI 和可复现验证使用它们；两个范围文件仍是升级输入。

本地默认数据库为 `V2.0/api/.labviz/labviz-v2.db`；临时项目和导出结果按最后
一次项目操作保留 2 小时；网站上传上限为 50 MB；验证码使用终端输出模式。
本地运行不需要配置 SMTP 或 SES；验证码直接显示在 API 终端。SMTP、SES、
PostgreSQL、S3/MinIO 和 AWS 文件仅供选择高级自托管方案的维护者使用。验证码
挑战、原子频率限制和登录会话默认由 SQLite 持久化；只有多进程、多主机部署才
需要 PostgreSQL 协调。

### 💻 命令行示例

```powershell
python main.py -i experiment.csv -x time -y temperature --type line -o result.png --summary
```

运行 `python main.py --help` 可查看全部参数。

### 🛠 常见问题

- 出现 `No such file or directory: 'requirements.txt'`：当前不在 V1.1 应用目录中。先运行 `Set-Location .\lab-data-visualization-tool\V1.1`，并用 `Test-Path .\requirements.txt` 确认返回 `True`。
- 出现“无法识别 `streamlit`”：依赖没有成功安装或 Python 环境不一致。重新安装依赖后使用 `python -m streamlit run app.py`。
- 出现 `Cache entry deserialization failed`：这是 pip 缓存警告，不是找不到项目文件的原因；通常可以忽略。
- PowerShell 禁止运行激活脚本：先执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，它只对当前窗口生效。

### 🧪 运行测试

以下命令从仓库根目录开始执行：

```powershell
Set-Location .\V1.1
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files
pytest -q --cov
```

V2.0 前端验证命令：

```powershell
Set-Location .\V2.0\web
npm ci
npm run verify
npm run test:e2e
```

浏览器测试同时覆盖桌面与移动端主流程、批准断点的截图回归，以及主要页面的严重/关键 WCAG 问题检查。

V2.0 API 验证命令：

```powershell
Set-Location .\V2.0\api
python -m pip install -r requirements-dev.lock.txt
docker compose up -d --wait postgres minio
docker compose run --rm minio-init
ruff check labviz_api tests migrations scripts
ruff format --check labviz_api tests migrations scripts
mypy labviz_api tests migrations scripts
python -m pytest
docker compose down -v --remove-orphans
```

完整 API 集成测试需要 Compose 提供 PostgreSQL 和 MinIO；CI 使用同一套锁文件、服务和验证命令。

### 许可证

LabViz 使用 [MIT 许可证](LICENSE)发布。任何人都可以使用、复制、修改、分发、
再许可或销售本软件，包括商业用途；条件是保留原始版权和许可证声明。本软件不
提供任何担保。

---

<div align="center">

**From raw measurements to readable results.**

[⭐ Star](https://github.com/DDDYT24/lab-data-visualization-tool) · [🐛 Report an issue](https://github.com/DDDYT24/lab-data-visualization-tool/issues) · [↩ Back to top](#-lab-data-visualization-tool)

</div>
