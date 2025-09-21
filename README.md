Lab Data Visualization Tool

(This is my first project. If there are any questions, I hope you can email me and I will modify it as soon as possible. Thank you so much !!!!)
--- email: liyutao982@gmail.com

![CI](https://github.com/DanielROG/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)

[![codecov](https://codecov.io/gh/DanielROG/lab-data-visualization-tool/branch/main/graph/badge.svg)](https://codecov.io/gh/DanielROG/lab-data-visualization-tool)


A Python-based tool to ‘clean, visualize, and explore experimental datasets’,which is designed to support scientific research workflows.
Using the 'Streamlit' for interactive dashboards and 'CLI' for fast data processing.

Features
- Clean datasets (remove duplicates, address missing values, check CSV format).
- Interactive data visualizable (histograms, scatter plots, line charts).
- Export processed datasets as CSV.
- CLI tool for quick analysis in the terminal.
- Automated testing, CI/CD, and coverage reporting with GitHub Actions and Codecov.

---
Installation

1. Clone the repository:
git clone https://github.com/DanielROG/lab-data-visualization-tool>.git
cd lab-data-visualization-tool

2. Create and activate a virtual environment:

'Windows (PowerShell)':
python -m venv venv
venv\Scripts\activate

For 'Linux/macOS (bash)':
python3 -m venv venv
source venv/bin/activate

3.Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q  ###Testing

---
Run the Streamlit app:
streamlit run app.py

Run the CLI tool:
python main.py --csv data.csv

pytest -q
pytest --cov=. --cov-report=term-missing

中文版本：

一个用于实验数据清理、可视化和分析的工具，支持自动化数据处理和结果展示。
适合科研人员、工程师以及学习数据分析的同学使用。

功能特点:
数据清理：支持缺失值处理、重复值删除等常见操作。
可视化：基于 matplotlib 和 seaborn 生成常见图表。
自动化：内置脚本可快速运行分析流程。
测试覆盖：通过 pytest + coverage 保证代码质量。
CI/CD 集成：支持 GitHub Actions 自动测试与 Codecov 覆盖率上传。

---
安装步骤

1. 克隆仓库
git clone https://github.com/DanielROG/lab-data-visualization-tool.git
cd lab-data-visualization-tool

2. 创建虚拟环境:
Windows (PowerShell):
python -m venv venv
venv\Scripts\activate

Linux/macOS (bash):
python3 -m venv venv
source venv/bin/activate

3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 开发环境依赖

pytest -q

---
使用方法
python main.py --input data.csv --output result.png

或者 streamlit run app.py

Run the CLI tool:
python main.py --csv data.csv

测试与覆盖率
pytest -q

生成覆盖率报告：
pytest --cov=. --cov-report=term-missing
