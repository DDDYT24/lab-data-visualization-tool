# LabViz

在本机导入实验表格、检查数据质量，并生成可追溯的科研图。

[English](README.md) · [简体中文](README.zh-CN.md) · [稳定版 V2.2.0](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [V2.3 RC 说明](V2.0/RELEASE_NOTES_V2.3.md) · [MIT 许可证](LICENSE)

## V2.3 预览版

本分支为 V2.3.0-rc.1。V2.2.0 仍是稳定版。本次 Windows RC 未签名，实际使用验收尚有未完成项，建议使用合成数据或数据副本评估。

新增本地研究工作台：按样本映射和预览处理数据，生成分组原始点、配对变化与小提琴图，将最多四个独立二维面板排成组图，复用模板并冻结科研输出。默认保留经典界面，可在设置切换概览界面。现有图表、历史记录、质量检查、清洗决定及 PNG／SVG／PDF 导出继续保留。

处理预览会显示映射、排除和计数，确认后才保存修订；样本、观测和独立实验单元分别计数。冻结输出附带源数据、计算和修订记录。超出已支持契约的专业统计方法不会被自动推断。

## Windows 安装版

从 [Releases](https://github.com/DDDYT24/lab-data-visualization-tool/releases) 下载 RC 安装器及 `SHA256SUMS.txt`，先核对哈希：

```powershell
Get-FileHash .\LabViz-V23-2.3.0-rc.1-Setup.exe -Algorithm SHA256
```

安装器自带 Python、Node.js、科研依赖、网页、字体和合成样例。安装后无需另装这些开发工具，也无需 Docker、数据库服务器、在线账号或邮箱登录。评估未签名 RC 时请保留 Windows 安全防护。

从开始菜单打开 **LabViz V2.3**，它会启动仅监听本机的浏览器界面。备份或维护前，用 **Stop LabViz V2.3** 停止服务。V2.3 与 V2.2 的程序和资料目录分离；旧数据需要显式导入为独立副本，卸载默认保留资料。

详细操作见[安装、V2.2 导入、备份恢复和卸载说明](V2.0/docs/INSTALL_V2.3.md)，依赖许可见[第三方声明](THIRD_PARTY_NOTICES.md)。

## 运行已发布 RC 的源码

源码运行需要 Git、Python 3.12 或 3.13，以及兼容的 Node.js。Windows 打包流程使用 Python 3.13 和锁定的 Node.js 24.22.0。

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
git checkout v2.3.0-rc.1
.\start-labviz.cmd
```

Windows 源码入口使用 3001／8001 端口，依赖和资料保留在本检出目录，默认数据在 `V2.0/api/.labviz/v23-development`。首次源码配置可能下载依赖；这与自包含安装版不同。

macOS／Linux 源码入口保留作开发用途，本轮不提供这些系统的原生安装包。

## 验证和重建

[Windows RC 构建说明](V2.0/packaging/windows-v23/README.md)提供从干净检出开始的流程：锁定输入、契约测试、真实载荷浏览器流程、旧版数据导入、安装、回滚及数据保留验证。

CI 覆盖既有应用、前端与 PostgreSQL／MinIO 集成；Windows 验证覆盖自带的 Python 3.13 和安装生命周期。发布附件记录源提交、构建、安装包哈希与实际验证范围。重新构建可能产生不同的构建 ID 和文件哈希。

默认数据存储仍是 SQLite 与本地文件。Docker 和托管持久化是可选开发／部署路径。API 镜像需要从 V2.0 上下文构建，以带入契约、样例和字体：

```sh
docker build -f V2.0/api/Dockerfile -t labviz-api:rc V2.0
```

## 尚未完成的 RC 验收

干净机器、完整物理断网流程、真实桌面 200% 缩放及读屏听读、iPhone 实机，以及 3–5 人研究者试用仍待完成。自动无障碍检查和浏览器模拟各有独立证据范围，不能替代这些实际观察；本 RC 不代表稳定版验收完成。

已审计的 npm 生产树无已报告告警，开发 lint 链仍有 5 项，原因见 [RC 说明](V2.0/RELEASE_NOTES_V2.3.md)。可通过 [Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues)报告安装、恢复或科学结果问题；请使用合成样例并移除附件中的个人资料。

## 许可证

LabViz 源码采用 MIT，随包第三方组件保留各自许可。原始仪器文件应与应用项目备份分开归档。
