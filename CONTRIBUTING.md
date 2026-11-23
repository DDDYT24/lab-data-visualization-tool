# 贡献指南

本文面向希望修改 LabViz 源代码、文档或测试的贡献者。普通用户只需要阅读根目录 README
和 `V2.0/docs/` 下的本地部署文档。

项目状态规则：开发待办（`V2.0/TODO.md`）和版本计划（`V2.0/V<版本>_PLAN.md`）由维护者保存在本地，
已列入 `.gitignore`，不进入仓库。版本交付范围记录在 [`VERSION_BASELINE.md`](VERSION_BASELINE.md)。
仓库内的说明文件只描述稳定契约、用户操作或历史证据，不写待办列表。

## 开发环境

- Python 3.12 或 3.13
- Node.js 22.22.2 或更新版本
- 现代浏览器
- Docker 仅在运行 PostgreSQL/MinIO 集成测试时需要

先运行根目录启动脚本，或分别安装 `V2.0/api/requirements-dev.txt` 和
`V2.0/web/package-lock.json` 对应的依赖。

## 提交前验证

前端：

```powershell
Set-Location .\V2.0\web
$env:NEXT_TELEMETRY_DISABLED = "1"
npm ci
npm run verify
npm run test:e2e
```

API：

```powershell
Set-Location .\V2.0\api
python -m pip install -r requirements-dev.lock.txt
ruff check labviz_api tests migrations scripts
ruff format --check labviz_api tests migrations scripts
mypy labviz_api tests migrations scripts
python -m pytest
```

需要 PostgreSQL/MinIO 的测试必须按 `V2.0/api/README.md` 启动测试服务，并在结束后清理
临时卷。不要把 `.labviz/`、`outputs/`、测试结果、真实实验数据或密钥加入提交。

依赖审计（CI 的 `frontend` 任务也会运行，需要联网；只放行已审查的问题）：

```powershell
python -m unittest discover -s .github/scripts -p "test_*.py"
python .github/scripts/npm_audit_gate.py
```

## 桌面候选包验收

先按 `desktop/README.md` 构建并冻结候选包，再从仓库根目录运行下面的驱动。`<候选>` 是含
`bundle/` 和 `build/` 的文件夹，`<标签>` 每次用新名字。运行驱动的 Python 需要装有 `psutil`。
全程不打开原生窗口，也不能代替原生窗口验收。

```powershell
python -B -X utf8 desktop/scripts/release_acceptance.py session --candidate <候选> --label <标签>
# session 通过后写出 <候选>/background-evidence.json，编译安装包时用 --evidence 传入
python -B -X utf8 desktop/scripts/release_acceptance.py installer --candidate <候选> --label <标签>
python -B -X utf8 desktop/scripts/release_acceptance.py large --candidate <候选> --label <另一标签>
python -B -X utf8 desktop/scripts/release_upgrade.py --old-build <旧候选>/build --new-build <候选>/build --old-package <旧包编号> --old-setup-sha <旧安装包 SHA-256> --output <新文件夹>
```

## 代码和文档约定

- 面向用户的文档使用产品名称、功能名称和公开的部署说明。
- 不在公开文档、截图、测试数据或提交消息中写入本地编辑工具、个人路径、内部会话信息、
  访问令牌或未公开的服务地址。
- 文件内容和路径、提交消息、提交作者、分支名、PR 标题和正文中，都不出现代码生成工具或对话助手的名称，
  也不为这类工具添加联合作者署名。受限词表不放进仓库：CI（`.github/workflows/public-text.yml`）
  从仓库 secret `RESTRICTED_TERMS` 读取；本地从 `.git/info/restricted-terms` 读取，每行一个词，
  词表请向仓库所有者索取。运行一次 `pre-commit install` 后，每次提交都会用
  `.github/scripts/check_public_text.py` 检查暂存的改动、提交消息和分支名；命中时只报告位置和词的编号。
- 新增功能必须说明默认行为、数据保存位置、失败处理和验证命令。
- 保持本地部署路径简单；远程数据库、对象存储和多主机方案应明确标记为维护者扩展。
- 保留与真实兼容性问题对应的回归测试，不要通过放宽断言来绕过失败。
