# LabViz V2.2 本地优先与离线运行指南

## V2.2.0 Windows 安装版：当前方法

从 [官方 Release](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0)
下载 `LabViz-Setup-2.2.0.exe` 和 `SHA256SUMS.txt`，或将两者复制到断网电脑。
用 `Get-FileHash -LiteralPath '.\LabViz-Setup-2.2.0.exe' -Algorithm SHA256` 核对哈希。
EXE 内置 Python 3.13.7、Node.js 24.17.0、锁定 API 依赖和 standalone 网页：
安装和首次启动都不需要 pip/npm、Docker、数据库服务器、邮箱登录或网络。

使用普通 Windows 账户、默认较短安装路径完成安装，然后点 LabViz 快捷方式，
使用自动打开的页面。浏览器和服务之间使用本机回环；端口被占用时启动器选择空闲端口。
真实导入成功后自动保存于 `%LOCALAPPDATA%\LabViz\data`；重启后再点快捷方式可打开历史。
开始菜单的 **Stop LabViz / 退出** 用于停止服务，**LabViz logs / 日志** 用于排查启动问题。

安装包未签名，请遵守机构的软件政策。本次仅支持单普通 Windows 用户；用户报告此前候选的
断网/重启验收通过，最终哈希对应的自动化验证范围见 Release 附件，双账户隔离未验证。
下文保留的是需要提前下载依赖的源码部署方法，不能与内置运行时的 EXE 混淆。

## Historical source-checkout guidance


本文面向最终用户和负责实验室电脑部署的管理员。LabViz 的运行路径是本地自托管：网站、API、
SQLite 数据库和处理结果都在同一台电脑上。

当前 V2.2.0-dev 仍使用开发代码仓库启动器；Windows 原生安装包只有架构契约和校验脚本，
尚未发布绑定 Python/Node 运行时的安装器。未来安装器的生命周期规则见
[`../packaging/windows/README.md`](../packaging/windows/README.md)。

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
