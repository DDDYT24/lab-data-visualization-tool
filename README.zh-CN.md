<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="V2.0/assets/labviz-logo-dark.svg">
  <img src="V2.0/assets/labviz-logo.svg" alt="LabViz 标志" width="320">
</picture>

# LabViz

[English](README.md) · [简体中文](README.zh-CN.md) · [稳定版 V2.2.0](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [V2.3 RC 说明](V2.0/RELEASE_NOTES_V2.3.md) · [MIT 许可证](LICENSE)

</div>

我是 LabViz，一个在你自己电脑上运行的科研制图工具。把 CSV 或 Excel 交给我，我先帮你查数据：缺失值、重复行、异常点都会标出来，保留还是剔除由你决定。然后我把它画成能放进论文的图。

## 我现在的版本

- **V2.2.0**：稳定版，日常使用请选它。
- **V2.3.0-rc.1**：预览版。安装器未签名，验收还没做完，请用合成数据或数据副本试用。
- **桌面版**（`desktop/`）：我正在学着不靠浏览器、用独立窗口运行，还没有发布。

## 我能做什么

- 读取 CSV、TSV、TXT、JSON、XLSX，导入前先给你看预览。
- 检查缺失、重复、类型冲突和异常值。我只标记，不替你删；每一次排除都会记下理由。
- 画折线、散点、柱状、直方、箱线、热图和 3D 曲面；V2.3 新增分组原始点、配对变化和小提琴图，最多四个面板拼成一张组图。
- 导出 PNG、SVG、PDF。V2.3 还能把图、各面板源数据、计算参数和修订记录一起冻结成研究输出。
- 分清样本、观测和独立实验单元，不会把重复测量算成更大的 n。

所有处理都在本机完成，不需要账号、数据库服务器或在线服务。

## 安装（Windows）

从 [Releases](https://github.com/DDDYT24/lab-data-visualization-tool/releases) 下载安装器和 `SHA256SUMS.txt`，先核对哈希：

```powershell
Get-FileHash .\LabViz-V23-2.3.0-rc.1-Setup.exe -Algorithm SHA256
```

我自带 Python、Node.js、科研依赖和字体，装好即可使用。从开始菜单打开 **LabViz V2.3**；备份前请用 **Stop LabViz V2.3** 让我停下。V2.3 和 V2.2 的程序与资料互不干扰，旧数据需要你显式导入，卸载时我默认保留资料。

详见[安装、导入、备份与卸载](V2.0/docs/INSTALL_V2.3.md)。

## 从源码运行

需要 Git、Python 3.12 或 3.13，以及 Node.js 22.22.2 以上（打包使用 24.22.0）。

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
git checkout v2.3.0-rc.1
.\start-labviz.cmd
```

源码模式使用 3001／8001 端口，数据存放在 `V2.0/api/.labviz/v23-development`。重新打包请看 [Windows RC 构建说明](V2.0/packaging/windows-v23/README.md)。

## 我还没做到的

- 干净机器、完整物理断网、真实 200% 缩放与读屏、iPhone 实机的验收。
- 代码签名。Windows 可能提示风险；请保留安全防护，不要绕过。
- 开发用的 lint 依赖链还有 5 项 npm 告警，生产依赖没有，原因见 [RC 说明](V2.0/RELEASE_NOTES_V2.3.md)。

我导出的图能否投稿，取决于你是否核对过尺寸、字体、统计假设和图注。实验设计是否成立，我不替你下结论。

## 反馈与许可

问题请提到 [Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues)，附件请使用合成数据并去掉个人信息。我的源码采用 MIT 许可，随包的第三方组件保留各自许可，见[第三方声明](THIRD_PARTY_NOTICES.md)。
