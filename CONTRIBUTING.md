# 贡献指南

本文面向希望修改 LabViz 源代码、文档或测试的贡献者。普通用户只需要阅读根目录 README
和 `V2.0/docs/` 下的本地部署文档。

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
- 新增功能必须说明默认行为、数据保存位置、失败处理和验证命令。
- 保持本地部署路径简单；远程数据库、对象存储和多主机方案应明确标记为维护者扩展。
- 保留与真实兼容性问题对应的回归测试，不要通过放宽断言来绕过失败。
