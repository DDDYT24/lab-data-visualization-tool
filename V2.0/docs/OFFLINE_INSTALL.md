# 本地离线运行指南

本文面向最终用户和负责实验室电脑部署的管理员。LabViz 的运行路径是本地自托管：网站、API、
SQLite 数据库和处理结果都在同一台电脑上。

## 首次安装

首次安装需要联网获取源代码和依赖。安装完成后，正常处理数据不需要 AWS、PostgreSQL、
MinIO、Docker 或外部邮件服务。

### Windows

在仓库根目录运行：

```powershell
.\start-labviz.cmd
```

脚本会自动创建 `V2.0/api/.venv`，检查 Python 3.12/3.13，安装 FastAPI 等 API 依赖，并在
缺少 `V2.0/web/node_modules` 时运行 `npm ci`。启动网站时会设置
`NEXT_TELEMETRY_DISABLED=1`。浏览器打开 `http://127.0.0.1:3000`。

### macOS / Linux

```bash
chmod +x start-labviz.sh
./start-labviz.sh
```

脚本会创建 Python 虚拟环境并安装 API 和网站依赖，同时禁用 Next.js 遥测。浏览器打开
`http://127.0.0.1:3000`。

## 已安装依赖后的运行

首次安装完成后，启动脚本不会重复安装依赖。应用运行时只使用本机的 Next.js、FastAPI、
SQLite 和本地对象目录。若电脑处于隔离网络，需提前准备仓库、Python wheel 缓存和 npm 缓存；
仓库提供的 `requirements.lock.txt` 和 `package-lock.json` 可用于复现已验证的依赖版本，
但离线缓存的制作属于实验室自己的软件分发流程。

依赖文件发生有意变更时，维护者才需要强制刷新：

```powershell
.\start-labviz.cmd -RefreshDependencies
```

```bash
./start-labviz.sh --refresh-dependencies
```

## 本地服务地址

| 服务 | 默认地址 | 作用 |
| --- | --- | --- |
| 网站 | `http://127.0.0.1:3000` | 浏览器界面 |
| API | `http://127.0.0.1:8000` | 文件解析、质量检查、绘图和导出 |
| API 文档 | `http://127.0.0.1:8000/docs` | 本机接口查看 |
| SQLite | `V2.0/api/.labviz/labviz-v2.db` | 本地项目和处理记录 |
| 对象目录 | `V2.0/api/.labviz/objects/` | 本地处理对象和导出结果 |

保持启动窗口开启，按 `Ctrl+C` 停止服务。详细的数据边界见
[`PRIVACY_DATA_BOUNDARY.md`](PRIVACY_DATA_BOUNDARY.md)。
