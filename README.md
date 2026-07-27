# Lab Data Visualization Tool

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)

A lightweight local tool for loading, checking, cleaning, visualizing, and exporting experimental data. It provides a Streamlit interface for researchers and a CLI for repeatable scripts. Raw experiment data is processed locally and is never written to the history database.

Current release: **V1.1**

中文说明见 [中文](#中文).

## Features

- Import CSV, TSV, delimited TXT, JSON, and XLSX tables (up to 50 MB in the web app).
- Inspect row/column counts, data types, missing cells, duplicates, unique values, and memory use.
- Remove exact duplicates and keep, drop, forward-fill, backward-fill, mean-fill, or median-fill missing values.
- Create line, scatter, bar, histogram, box, correlation heatmap, and 3D surface plots.
- Export the complete cleaned table as CSV and the current figure as PNG.
- Keep compact plot history in SQLite without storing raw table contents.
- Sample large plots evenly while keeping previews and exports deterministic.

## Architecture

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

## Installation

Python 3.12 or 3.13 is recommended.

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Web interface

```bash
streamlit run app.py
```

Upload a table in the sidebar, choose cleaning rules, inspect the data-quality tab, then generate and download a visualization. Large previews are limited to 200 rows, and plots above the selected threshold use an evenly spaced sample; the cleaned CSV export always contains every remaining row.

For a 3D surface, select three numeric columns representing X, Y, and Z. The points must span a surface rather than a single straight line.

## CLI

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

## Development and verification

```bash
python -m pip install -r requirements-dev.txt
ruff check app.py main.py labviz tests
ruff format --check app.py main.py labviz tests
mypy app.py main.py labviz
pytest -q --cov
```

CI runs the same lint, format, type, and test checks on Python 3.12.

## Design reference

The large-file preview, data-quality summary, and numeric-column guidance were informed by [The-Schultz-Lab/plottle](https://github.com/The-Schultz-Lab/plottle) (MIT). This project keeps an intentionally smaller scope and contains an independent implementation; it does not include Plottle source code or its heavier plugin/format stack.

## 中文

这是一个轻量、本地运行的实验数据处理与可视化工具，面向需要“上传数据后直接检查、清洗、绘图并导出”的实验者。网页端使用 Streamlit，命令行端适合重复实验和批处理脚本。

### 核心能力

- 支持 CSV、TSV、分隔符 TXT、JSON 和 XLSX，网页端单文件上限为 50 MB。
- 自动统计行列数、字段类型、缺失值、重复行、唯一值和内存占用。
- 支持删除重复行，以及保留、删除、前向填充、后向填充、均值填充和中位数填充。
- 支持折线图、散点图、柱状图、直方图、箱线图、相关性热力图和 3D 曲面模型。
- 可下载完整清洗数据和 PNG 图像。
- SQLite 只记录绘图历史和数据质量摘要，不保存实验原始数据。
- 大数据绘图使用等距抽样，避免浏览器卡顿；数据下载仍保留全部清洗结果。

### 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

命令行示例：

```powershell
python main.py -i experiment.csv -x time -y temperature --type line -o result.png --summary
```

运行测试：

```powershell
python -m pip install -r requirements-dev.txt
pytest -q --cov
```
