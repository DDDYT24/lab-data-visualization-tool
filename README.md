# Lab Data Visualization Tool

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/DDDYT24/lab-data-visualization-tool/main?logo=codecov)](https://app.codecov.io/gh/DDDYT24/lab-data-visualization-tool)

A Python tool for cleaning and visualizing lab CSV data.  
This repository provides:

- A **Streamlit app** (`app.py`) for interactive exploration
- A **CLI** (`labviz/cli.py`) for scriptable terminal workflows

## Features

- CSV loading with column validation
- Data cleaning: duplicate removal + optional missing-value handling
- Plotting: **line / scatter / bar**
- Export cleaned CSV from Streamlit
- Automated quality checks with Ruff, MyPy, Pytest, GitHub Actions, and Codecov

## Installation

1. Clone repository:

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
```

2. Create and activate virtual environment:

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS (bash):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Notes (from current implementation):
- File upload: CSV only
- Cleaning option: `Drop rows containing NaN` (default enabled)
- Plot types: `line`, `scatter`, `bar`

### CLI

Current CLI entry is `labviz/cli.py` (not `main.py`):

```bash
python labviz/cli.py -i INPUT.csv -x X_COLUMN -y Y1 [Y2 ...]
```

Common options:
- `--type {line,scatter,bar}` (default: `line`)
- `-o, --out OUTPUT_PATH` (default: `plot.png`, auto-adds `.png` if no suffix)
- `--method {ffill,bfill}` fill missing values before optional dropping
- `--dropna` drop rows containing NaN (CLI default is off unless this flag is set)
- `--show` show interactive plot window if environment supports GUI

Example:

```bash
python labviz/cli.py -i data.csv -x time -y temperature humidity --type line -o result.png --method ffill --dropna
```

## Testing / Lint / Type Check

Same commands as CI:

```bash
ruff check .
mypy app.py main.py
pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing
```

---

## 中文说明（与当前代码实现对齐）

这是一个用于实验数据清理与可视化的 Python 工具，包含两种使用方式：

- `app.py`：Streamlit 交互界面
- `labviz/cli.py`：命令行批处理

### 主要功能

- 读取 CSV 并校验列名
- 数据清理（去重、缺失值处理）
- 绘图类型：折线图 / 散点图 / 柱状图
- 在 Streamlit 页面下载清理后的 CSV

### 安装

```bash
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
cd lab-data-visualization-tool
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 使用

Streamlit：

```bash
streamlit run app.py
```

CLI（当前正确入口）：

```bash
python labviz/cli.py -i 输入.csv -x 横轴列 -y 纵轴列1 [纵轴列2 ...]
```

示例：

```bash
python labviz/cli.py -i data.csv -x time -y value --type scatter -o plot.png --dropna
```
