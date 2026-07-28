<div align="center">

# 🧪 Lab Data Visualization Tool

**Turn raw experiment tables into clear, exportable visuals — locally.**

从导入、检查、清洗到 2D/3D 可视化与导出，一套面向实验数据的轻量工作流。

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
![Release](https://img.shields.io/badge/release-V1.1-6f42c1)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-ff4b4b?logo=streamlit&logoColor=white)

[Quick Start](#-quick-start) · [Features](#-features) · [CLI](#-command-line) · [Development](#-development-and-verification) · [中文](#中文说明)

</div>

Lab Data Visualization Tool is a local-first application for loading, validating, cleaning, visualizing, and exporting experimental data. Researchers can use the Streamlit interface for interactive work or the CLI for repeatable scripts. Raw experiment data stays on the local machine and is never written to the history database.

> **Current release: V1.1** — supports CSV, TSV, delimited TXT, JSON, and XLSX data, with seven 2D/3D visualization types.

## ⚡ Quick Start

Prerequisites: [Git](https://git-scm.com/downloads) and Python 3.12 or 3.13. Run the following commands from the folder where you want to download the project.

### Windows PowerShell

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location .\lab-data-visualization-tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the application in the browser, normally at `http://localhost:8501`. Upload a table in the sidebar, choose cleaning rules, inspect data quality, generate a chart, and download the result.

```text
Load data  →  Inspect quality  →  Clean safely  →  Visualize  →  Export
```

## ✨ Features

| Stage | Capabilities |
| --- | --- |
| Load | CSV, TSV, delimited TXT, JSON, and XLSX; up to 50 MB in the web app |
| Inspect | Row and column counts, data types, missing cells, duplicates, unique values, and memory use |
| Clean | Exact-duplicate removal; keep, drop, forward-fill, backward-fill, mean-fill, or median-fill missing values |
| Visualize | Line, scatter, bar, histogram, box, correlation heatmap, and 3D surface plots |
| Export | Complete cleaned table as CSV and the current figure as PNG |
| Track | Compact SQLite plot history without storing raw table contents |
| Scale | Deterministic preview limits and evenly spaced sampling for large plots |

Large previews are limited to 200 rows. Plots above the selected threshold use an evenly spaced sample, while the cleaned CSV download always contains every remaining row.

For a 3D surface, select three numeric columns representing X, Y, and Z. The points must span a surface rather than a single straight line.

## 🧱 Architecture

```text
app.py                  Streamlit front end
main.py                 CLI compatibility entry point
labviz/core.py          Loading, validation, cleaning, and profiling
labviz/plotting.py      Validated 2D/3D rendering and bounded sampling
labviz/database.py      SQLite metadata/history repository
labviz/cli.py           Scriptable command-line workflow
tests/                  Core, plotting, database, CLI, and UI smoke tests
```

The SQLite file is created at `.labviz/history.db` on first launch and is ignored by Git. Set `LABVIZ_DB_PATH` to use another location. Only the filename, SHA-256 fingerprint, size, quality counts, and plot configuration are recorded.

## 🧰 Supported Development Environment

| Area | Supported and verified environment |
| --- | --- |
| Operating system | Designed for Windows, macOS, and Linux. CI verifies `ubuntu-latest`; V1.1 was also verified locally on Windows. |
| Python | Python 3.12 is verified by CI and Python 3.13 was verified locally. Other versions are not part of the release checks. |
| Hardware | CPU-only; no GPU or external database server is required. |
| Web runtime | A modern browser and local access to Streamlit's default port `8501`. |

### Runtime Dependencies

Install these from `requirements.txt` when you only need to use the application:

| Dependency | Supported range | Purpose |
| --- | --- | --- |
| Streamlit | `>=1.49,<2` | Web interface, uploads, controls, caching, and downloads |
| pandas | `>=2.2,<3` | Tabular loading, cleaning, profiling, and CSV export |
| NumPy | `>=1.26,<3` | Numeric arrays, deterministic sampling, and plotting support |
| Matplotlib | `>=3.9,<4` | 2D/3D figure rendering and PNG export |
| openpyxl | `>=3.1,<4` | XLSX import support |

SQLite is provided by Python's standard library, so no database package or service is required.

### Development and CI Dependencies

Contributors should install both `requirements.txt` and `requirements-dev.txt`:

| Dependency | Supported range | Purpose |
| --- | --- | --- |
| pytest | `>=8.3,<9` | Automated tests |
| pytest-cov | `>=5,<7` | Coverage reporting |
| pre-commit | `>=3.8,<5` | Repository checks before commits |
| mypy | `>=1.11,<2` | Static type checking |
| Ruff | `==0.6.9` | Linting and formatting; pinned to match the pre-commit hook and CI |

Runtime packages use bounded version ranges so compatible updates can be installed without silently crossing a major-version boundary. Upgrade a major version separately and run the complete validation suite before changing these bounds.

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

The command is running outside the repository. In PowerShell, enter the cloned project directory and confirm the file exists:

```powershell
Set-Location .\lab-data-visualization-tool
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

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
ruff check app.py main.py labviz tests
ruff format --check app.py main.py labviz tests
mypy app.py main.py labviz
pytest -q --cov
```

CI runs the same lint, format, type, and test checks on Python 3.12.

## 📚 Design References

- The large-file preview, data-quality summary, and numeric-column guidance were informed by [The-Schultz-Lab/plottle](https://github.com/The-Schultz-Lab/plottle) (MIT).
- The README's strong project statement, visible status badges, and quick-start-first hierarchy were inspired by [msitarzewski/agency-agents](https://github.com/msitarzewski/agency-agents).

This project keeps an intentionally smaller scope and contains an independent implementation. It does not include source code or copied documentation from either reference project.

---

## 中文说明

这是一个轻量、本地运行的实验数据处理与可视化工具，面向需要“上传数据后直接检查、清洗、绘图并导出”的实验者。网页端使用 Streamlit，命令行端适合重复实验和批处理脚本；实验原始数据不会写入历史数据库。

### 🚀 快速开始

请先安装 [Git](https://git-scm.com/downloads) 和 Python 3.12 或 3.13，然后在准备存放项目的目录中执行完整命令。**克隆后必须先进入项目目录**，否则系统找不到 `requirements.txt` 和 `app.py`。

#### Windows PowerShell

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location .\lab-data-visualization-tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

#### macOS / Linux

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

启动后浏览器通常会打开 `http://localhost:8501`。在侧边栏上传表格，选择清洗规则，在数据质量页检查结果，然后生成并下载图像。

### ✨ 核心能力

- 支持 CSV、TSV、分隔符 TXT、JSON 和 XLSX，网页端单文件上限为 50 MB。
- 自动统计行列数、字段类型、缺失值、重复行、唯一值和内存占用。
- 支持删除重复行，以及保留、删除、前向填充、后向填充、均值填充和中位数填充。
- 支持折线图、散点图、柱状图、直方图、箱线图、相关性热力图和 3D 曲面模型。
- 可下载完整清洗数据和 PNG 图像。
- SQLite 只记录绘图历史和数据质量摘要，不保存实验原始数据。
- 大数据绘图使用等距抽样，避免浏览器卡顿；数据下载仍保留全部清洗结果。

### 🧰 支持的开发环境

| 项目 | 支持与验证情况 |
| --- | --- |
| 操作系统 | 设计上支持 Windows、macOS 和 Linux；CI 验证 `ubuntu-latest`，V1.1 也已在 Windows 本地验证。 |
| Python | CI 验证 Python 3.12，本地验证 Python 3.13；其他版本不属于当前发布检查范围。 |
| 硬件 | 仅需 CPU，不需要 GPU，也不需要外部数据库服务器。 |
| 网页运行 | 现代浏览器，并允许本机访问 Streamlit 默认端口 `8501`。 |

#### 运行依赖

普通用户只需要安装 `requirements.txt`：

| 依赖 | 支持范围 | 用途 |
| --- | --- | --- |
| Streamlit | `>=1.49,<2` | 网页界面、上传、交互控件、缓存和下载 |
| pandas | `>=2.2,<3` | 表格读取、清洗、质量分析和 CSV 导出 |
| NumPy | `>=1.26,<3` | 数值数组、确定性抽样和绘图支持 |
| Matplotlib | `>=3.9,<4` | 2D/3D 绘图和 PNG 导出 |
| openpyxl | `>=3.1,<4` | XLSX 文件读取 |

SQLite 来自 Python 标准库，因此不需要额外安装数据库软件或 Python 数据库包。

#### 开发与 CI 依赖

开发者需要同时安装 `requirements.txt` 和 `requirements-dev.txt`：

| 依赖 | 支持范围 | 用途 |
| --- | --- | --- |
| pytest | `>=8.3,<9` | 自动化测试 |
| pytest-cov | `>=5,<7` | 测试覆盖率 |
| pre-commit | `>=3.8,<5` | 提交前检查 |
| mypy | `>=1.11,<2` | 静态类型检查 |
| Ruff | `==0.6.9` | 代码检查和格式化；固定版本以确保本地、pre-commit 与 CI 一致 |

运行依赖采用带主版本上限的范围，能够获取兼容更新，同时避免自动跨越可能带来破坏性变更的主版本。升级主版本时应单独修改并运行完整验证。

### 💻 命令行示例

```powershell
python main.py -i experiment.csv -x time -y temperature --type line -o result.png --summary
```

运行 `python main.py --help` 可查看全部参数。

### 🛠 常见问题

- 出现 `No such file or directory: 'requirements.txt'`：当前不在项目目录中。先运行 `Set-Location .\lab-data-visualization-tool`，并用 `Test-Path .\requirements.txt` 确认返回 `True`。
- 出现“无法识别 `streamlit`”：依赖没有成功安装或 Python 环境不一致。重新安装依赖后使用 `python -m streamlit run app.py`。
- 出现 `Cache entry deserialization failed`：这是 pip 缓存警告，不是找不到项目文件的原因；通常可以忽略。
- PowerShell 禁止运行激活脚本：先执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，它只对当前窗口生效。

### 🧪 运行测试

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files
pytest -q --cov
```

---

<div align="center">

**From raw measurements to readable results.**

[⭐ Star](https://github.com/DDDYT24/lab-data-visualization-tool) · [🐛 Report an issue](https://github.com/DDDYT24/lab-data-visualization-tool/issues) · [↩ Back to top](#-lab-data-visualization-tool)

</div>
