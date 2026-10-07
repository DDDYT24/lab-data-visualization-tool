<div align="center">

# LabViz — 科研数据可视化工具

**导入实验表格、检查数据质量、生成图表，在本机保留项目和导出结果。**

[![CI](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/DDDYT24/lab-data-visualization-tool/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v2.2.0-0f766e)](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [下载 V2.2](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [发布说明](V2.0/RELEASE_NOTES_V2.2.md) · [问题反馈](https://github.com/DDDYT24/lab-data-visualization-tool/issues)

</div>

LabViz 面向研究人员、写论文的学生和教学实验室。V2.2 Windows 安装包内置网页界面、
Python 科研处理服务和运行环境：你在浏览器中操作，计算和保存都在自己的电脑上完成。
**无需 LabViz 登录或邮箱验证码**，普通使用者无需另装 Python、Node.js、Docker、
PostgreSQL 或 S3，也不需要 AWS 账户、域名或在线网站。

## 安装与启动：Windows 用户

1. 从官方发布页面下载 [LabViz-Setup-2.2.0.exe](https://github.com/DDDYT24/lab-data-visualization-tool/releases/download/v2.2.0/LabViz-Setup-2.2.0.exe)。支持范围为 **Windows x64**。
2. 可用 PowerShell 计算哈希，与同页 [SHA256SUMS.txt](https://github.com/DDDYT24/lab-data-visualization-tool/releases/download/v2.2.0/SHA256SUMS.txt) 核对：

   ```powershell
   Get-FileHash -LiteralPath '.\LabViz-Setup-2.2.0.exe' -Algorithm SHA256
   ```

3. 使用平时的 Windows 账户运行安装包，选择中文或英文，保留默认安装目录；无需管理员权限。
4. 双击桌面或开始菜单的 **LabViz** 快捷方式，使用它自动打开的页面。**没有登录步骤**。
   首次打开请用快捷方式，旧书签或手动输入地址可能没有本地授权。
5. 导入自己的数据，或先试内置合成示例。EXE 下载或拷贝到电脑后，安装和正常使用可在断网时完成。
6. 需要结束后台服务时，在开始菜单选择 **Stop LabViz / 退出**。

当前公开安装包**未签名**，Windows 可能出现发布者/SmartScreen 提示。请核对官方来源和
SHA-256，并遵守学校或单位的软件安装规定。我们没有宣称已获得代码签名或所有杀毒服务的认证；
本机实际扫描结果见发布附件中的验证报告。不要绕过机构的软件安装策略。

### 已安装旧版：如何更新、历史是否还在

先退出 LabViz，再用**同一 Windows 账户、相同默认程序目录**运行新 EXE，完成后点原有快捷方式。
**不需要先删除旧版本**。数据保存在程序目录之外，更新或同版本修复默认保留历史，不主动清空数据。
更新后进入“历史”检查，重要数据请先离线备份。卸载时会询问保留还是永久删除数据；
打算重装并继续使用历史时请选择“保留”。真实 V2.1.1 安装器升级/回滚验收已按用户决定豁免，
因此不能保证每一个历史构建都经过同样的升级验证。

## 科研使用流程

```text
CSV / TSV / TXT / JSON / XLSX
            ↓
解析与校验 → 检查数据质量 → 确认清洗
            ↓
选择图表、字段、统计方法、标签和样式
            ↓
预览 → 导出 PNG / SVG / PDF
            ↘ 本地历史、处理后数据和已保存图表
```

| 功能 | 能做什么 |
| --- | --- |
| 导入 | 支持 CSV、TSV、分隔符 TXT、JSON、XLSX；按适用格式选择工作表和表头行 |
| 质量检查 | 查看缺失值、非数值、重复和网格问题；在应用清洗前确认处理选择 |
| 图表 | 折线图、散点图、柱状图、直方图、箱线图、热图、规则网格 3D 曲面 |
| 科研控件 | X/响应/分组字段、拟合、不确定性区间、坐标范围/单位、标题、字体、配色和图例 |
| 统计 | 普通/加权拟合、残差诊断、预测区间、Working–Hotelling 同时均值带、线性 Huber 稳健拟合 |
| 导出 | PNG/SVG/PDF，配置尺寸和分辨率；下载清洗后的 CSV |
| 本地历史 | 真实导入自动保存，可重新打开和编辑、预览处理数据、查看/下载已保存图表、确认删除 |
| 界面 | 中英文、浅色/暗色/跟随系统主题、图表数据表、键盘/可访问性检查、移动端 3D 控制 |
| 示例与帮助 | 7 个版本化合成教学示例，以及中英文帮助和关于页面 |

真实文件成功处理后自动保存；内置示例默认是临时项目，示例结果不能作为真实实验依据。
分组比较支持类别作为 X；单测量字段可使用直方图/箱线图。浏览器预览与 Python 导出共享
直方图区间边界和数值坐标范围，并核验实际导出图内容。两种渲染器的字体和样式不保证逐像素相同。

### 对写论文的人有什么帮助？与 Excel 比较如何？

LabViz 把导入、质量检查、分析、绘图、历史和导出连接起来，减少重复配置；支持规则 3D 网格，
明确统计方法的假设和限制，并方便重新打开项目调整论文图。Excel 仍适合电子表格计算和手工编辑。
LabViz 不代替实验设计审查，也不是覆盖所有统计方法的完整统计软件。

**导出的 PNG、SVG、PDF 满足目标期刊要求时可以用于论文。** 提交前请检查最终尺寸/DPI、
期刊接受的矢量格式、字体、标签和单位、颜色可读性、统计假设、数据排除规则及图注，
并直接检查下载的文件。软件不会自动证明科学结论有效，也不保证满足所有出版社的规范。
具体模型和限制见 [统计契约](V2.0/docs/STATISTICS_CONTRACT_V2.2.md)；Huber 当前仅支持线性拟合，
不提供置信带。

## 架构与技术栈

```text
浏览器：Next.js / React / TypeScript / MUI / ECharts
    → 本机 Next.js API 代理 → FastAPI / Python
    → 解析、质量检查、清洗、数值分析、Matplotlib 导出
    → SQLite 元数据 + 本地文件（处理对象与图表）
```

| 层次 | V2.2 Local 使用的技术 |
| --- | --- |
| 前端 | Next.js 16.3.6、React 19、TypeScript、MUI、TanStack Query、Zustand、ECharts/ECharts GL |
| 后端与科研处理 | FastAPI、pandas、NumPy、SciPy、Matplotlib、PyArrow、工作簿读取库 |
| 数据保存 | SQLite 保存项目/处理元数据；本地文件系统保存对象和图表 |
| 安装包运行时 | 内置 CPython 3.13.7、Node.js 24.17.0 |
| Windows 分发 | Inno Setup、按用户安装、PowerShell 启动器、本地授权、生命周期/恢复脚本 |
| 验证 | pytest、Ruff、MyPy、Vitest、ESLint、TypeScript、Playwright |
| 可选集成路线 | PostgreSQL、S3 兼容/MinIO 适配器、Worker、容器与基础设施源码 |

**FastAPI 没有去掉**，仍负责后端处理。本地版用 SQLite 避免用户部署数据库服务器，
本地文件目录承担对象存储的作用。PostgreSQL 和 S3 适配器仍在仓库中，用于集成测试和未来共享部署。
**Cloud 云协作、托管同步、团队账户和公开云服务仍延期**；保留这些源码不等于已经上线云协作版。

## 数据、隐私、备份与旧数据迁移

| 安装版 Windows 路径 | 用途 |
| --- | --- |
| `%LOCALAPPDATA%\Programs\LabViz` | 程序和版本目录 |
| `%LOCALAPPDATA%\LabViz\data` | SQLite、对象文件和本地访问凭据 |
| `%LOCALAPPDATA%\LabViz\logs` | 启动与维护日志 |
| `%LOCALAPPDATA%\LabViz\backups` | 维护/升级产生的备份 |

Windows **源码启动版**使用 `V2.0/api/.labviz`，安装 EXE 不会自动导入源码历史。
两边都停止后，可通过开始菜单 **Import old data / 迁移旧数据** 将兼容目录导入空的安装版数据目录；
保留源目录和通过完整性校验的备份。旧 guest/邮箱账户项目不会自动改归新的本地身份；
若在本地历史看不到，需要重新导入原文件。

备份时先停止 LabViz，再复制**整个 data 目录**，不能只复制 SQLite：数据库记录需要配套对象文件。
原始实验文件请单独保存，项目存储不等于完整原文件档案。删除本地项目是永久操作，
要恢复必须有删除前的备份。详细步骤见 [备份与恢复](V2.0/docs/BACKUP_RESTORE.md)。

普通使用的服务只绑定 `127.0.0.1`，处理在本机完成，无自动云同步；打开 GitHub、外部帮助或
邮件链接由你主动触发。本次发布支持**一个普通 Windows 用户配置**。
两个真实 Windows 账户的文件权限和同时运行的 loopback 隔离**未验证**，不能宣称这一隐私保证，
也不要把它当作多人共享服务或暴露端口到公网。见 [隐私边界](V2.0/docs/PRIVACY_DATA_BOUNDARY.md)
与 [安全说明](SECURITY.md)。

## 测试、验收与发布范围

2026-10-07 全量源码/候选包审查通过：**271 项 API、60 项 Vitest、65 项浏览器回归**，
十个不同的打包后真实 API 工作流、**39 个 fixture、100/100 chaos 数据案例**以及三轮强制性能检查。
PostgreSQL/MinIO 是独立的真实集成测试服务，普通用户无需部署。安装器/启动器生命周期使用
临时数据与 test AppId，没有覆盖本机个人生产安装。

用户报告已完成此前候选包的干净/断网/重启验收，并授权按单用户范围发布。
用户报告与代理实际观察的自动化结果分开记录，未提供对最终 EXE 的独立哈希验收证明。
正式构建只调整版本/发布元数据和文档；精确 commit、构建 ID、EXE 哈希、重新执行的打包后验证
随 Release 附件提供。历史失败及修复保留在 [P7 报告](V2.0/docs/V2.2_P7_TEST_REPORT.md)。

生产依赖风险已修补/审查；还有 7 个开发工具依赖问题记录在案。
Arrow 排除项只适用于已审查的 Python/Parquet 路径，不代表所有依赖都没有漏洞。
见 [依赖安全审查](V2.0/docs/DEPENDENCY_SECURITY_REVIEW.md)。

## 从源码开发

为保持旧路径兼容，V2.2 的代码仍放在 `V2.0/`。源码方式需要 Git、**Python 3.12/3.13**、
**Node.js 22.22.2+**（验证使用 24），暂不支持 Python 3.14。
首次会下载依赖；这与无需另装运行时的 Windows EXE 不同。

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
.\start-labviz.cmd
# 依赖文件更新后：
.\start-labviz.cmd -RefreshDependencies
```

macOS/Linux 可用 `chmod +x start-labviz.sh && ./start-labviz.sh` 运行源码；Unix 启动器仍沿用
原会话方式，与 Windows 免登录启动器完全对齐及 macOS/Linux 原生安装包不在本次发布范围内。
分开启动 API/网站及完整集成测试命令见 [API 开发说明](V2.0/api/README.md)。

```powershell
Set-Location -LiteralPath .\V2.0\web
$env:NEXT_TELEMETRY_DISABLED = '1'
npm ci
npm run verify
npx playwright install chromium firefox webkit
npm run test:e2e
```

默认浏览器回归包含模拟 API 检查，真实服务测试需要显式启用。
[`test-live-candidate.ps1`](V2.0/packaging/windows/test-live-candidate.ps1) 可以对临时配置执行打包后真实 API 测试；
重建方式见 [Windows 打包说明](V2.0/packaging/windows/README.md)。完整 API 测试需独立 PostgreSQL/MinIO
与锁定开发依赖，不需要真实实验数据。

## 常见问题

- **项目数据不可用 / local access denied：** 退出后重新点 LabViz 快捷方式，使用自动打开的页面。
  启动器内部授权用于读取本地历史；仍失败时查看 `%LOCALAPPDATA%\LabViz\logs`，不要公开凭据或原始日志。
- **端口被占用：** 安装版自动选择可用的回环端口，请使用启动器打开的页面。
- **休眠或重启后打不开旧页面：** 重新点快捷方式，旧标签页可能仍指向已停止的服务。
- **04/05 字段不对：** 分组比较选择类别 X 和数值响应；单测量值选择推荐的直方图/箱线图。
  若旧版仍把折线图 X/响应都设成 measurement，请更新到 V2.2。
- **导出与预览有差异：** 检查下载文件的字段、分箱、坐标范围、单位，使用合成数据报告复现步骤。
  浏览器与 Python 使用不同渲染器，字体样式不保证逐像素相同。
- **路径或安装失败：** 使用默认较短安装路径；任意超长自定义路径及所有休眠策略未验证。
- **更新后历史为空：** 核对 Windows 账户和数据位置；源码与安装版目录独立，先查迁移指南，不要删除旧数据。
- **源码环境问题：** 检查 `py -0p`、`node --version`，安装工具后重开终端；依赖变化后执行刷新命令。

## 仓库结构与完整文档

| 路径 | 内容 |
| --- | --- |
| `V2.0/web/` | 前端、本地 API 代理、单元/浏览器测试 |
| `V2.0/api/` | 科研处理/渲染、持久化/存储适配器、API 测试 |
| `V2.0/contracts/` | 渲染与本地存储契约 |
| `V2.0/api/samples/v22/` | 公开合成示例和边界案例 |
| `V2.0/packaging/windows/` | 安装器源码、构建、校验和生命周期测试 |
| `V2.0/docs/` | 隐私、离线、备份、统计、安全与验收证据 |
| `V1.1/` | 旧版 Python/Streamlit 应用 |

[版本基准](VERSION_BASELINE.md) · [更新记录](CHANGELOG.md) · [待办与状态唯一来源](V2.0/TODO.md) ·
[离线指南](V2.0/docs/OFFLINE_INSTALL.md) · [贡献指南](CONTRIBUTING.md) ·
[公开发布清单](V2.0/docs/PUBLIC_RELEASE_CHECKLIST.md)

## 反馈与许可

通过 [GitHub Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues) 或应用反馈链接
[liyutao982@gmail.com](mailto:liyutao982@gmail.com) 报告可复现问题。分享诊断前移除真实数据、
本地凭据和个人路径。开发使用 AI 编程辅助，工程结论和发布证据按实际验证范围记录。

源码使用 [MIT 许可证](LICENSE)，保留版权与许可声明即可使用、修改和分发；不提供任何担保。
随包第三方组件遵守各自的许可，包括 [OFL 中文字体](V2.0/assets/fonts/OFL.txt)。
