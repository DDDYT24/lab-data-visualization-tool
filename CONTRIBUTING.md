# 贡献指南

本文面向希望修改 LabViz 源代码、文档或测试的贡献者。普通用户只需要阅读根目录 README
和 `V2.0/docs/` 下的本地部署文档。

项目状态规则：未来功能、缺陷、兼容性问题、发布门槛和延期决定只记录在
[`V2.0/TODO.md`](V2.0/TODO.md)，它只写“接下来做什么”。从 V2.4 起，每个版本另有一份
`V2.0/V<版本>_PLAN.md`（例如 [`V2.4_PLAN.md`](V2.0/V2.4_PLAN.md)），写详细计划：每一步改哪些文件、
验收门槛，以及实际执行过的验收命令和结果。完成或调整一步时，在同一个提交里更新这份计划文件
和 TODO.md。版本交付范围记录在 [`VERSION_BASELINE.md`](VERSION_BASELINE.md)。
其他说明文件只能描述稳定契约、用户操作或历史证据，不能复制一份新的待办列表。

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
