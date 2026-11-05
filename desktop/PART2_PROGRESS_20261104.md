# 原生桌面第二部分：实现、构建与暂停记录

2026-11-04（America/Edmonton）。用户要求“先停到这更新文件”，本轮已停止。

**第二部分未完成。新候选的原生窗口验收尚未开始，未执行实际安装。** 本轮完成的是安装回滚修复、当前源码构建、候选冻结和后台验证。第一部分已沿用的十轮没有重跑；第三、四部分没有签收。

## 工作位置与源码

- 分支：`desktop/part2-acceptance`，起点 `4e779c5`。
- `4ec6a06`：安装回滚、退出回执身份、暂存来源修复及回归测试。
- `f9161b9`：Windows 检出时保持上游许可补充文件的精确字节。
- 主目录保留 `main`；详细计划和待办继续保存在主目录本地忽略的 `V2.0/V2.3_PLAN.md`、`V2.0/TODO.md`，并与 `V2.0/V2.4_PLAN.md` 的第二部分记录同步。
- 包外的 `desktop/tests/part2_headless_acceptance.mjs` 还有未提交的 PDF 验收驱动修改；该驱动尚未通过，不得当作已完成修复。

## 新冻结候选

| 项目 | 身份 |
| --- | --- |
| Bundle | `outputs/desktop-d06/part2-20261104/bundle` |
| Package ID | `29db6edf66bd908c9fb3fdbbf852c932305adac9447e362303577fcffa48866a` |
| Payload ID | `0838947f05efb412fb584ac53925a51e791cdb5be89bc410fc096f735c7139f7` |
| Host SHA-256 | `e149318f9e698d57d6b72c5657750874b0c05a59fc044776e5d2d1dda429fe83` |
| Source commit | `f9161b9af8bf1d48415b3e12d8358d5fb153e368` |
| Data epoch | 2 |
| 文件完整性 | 13,495 个文件与冻结清单匹配；来源与编译回执匹配 |
| 匹配 Setup | 编译按用户停止要求中断，尚无可验收的匹配安装器 |

包已冻结，不得在其内部修改、补文件或重做暂存。`build-final/build-verification.json` 保留非零编译退出码 `4294967295`；残留 EXE 已改名为 `.exe.incomplete`，不是可用安装包。文件大小、哈希和保留路径见 `outputs/desktop-part2-20261104/user-stop.json`。

旧 `p2final28` 包的 13,174 个文件，以及旧 Setup 的 `7cb4b6082603d228527bf83f00cac0024ca3c822eb331e4099285668066d7f7e`／147,435,156 字节均复核保持。旧原生结果仍只属于旧执行包。

## 已完成的实现与检查

1. 新增显式安装 `rollback`：绑定当前／前代候选、所选资料、备份身份与 epoch；先保留当前资料，再恢复升级前科学数据和完整 session／preferences，验证前代健康后切换候选。失败会恢复当前资料。新工作保留在备份与 retained 目录，不静默丢弃。
2. 修复同版本 repair 覆盖原升级凭据的问题；独立复核了 `epoch1 → epoch2 → repair → rollback` 的临时 SQLite／草稿夹具。健康检查在这些单元测试中使用替身，不能代替真实旧安装包运行。
3. 原生启动驱动的退出回执加入 `runId`、`shell`；暂存载荷不再硬编码历史 source commit。
4. 当前 host、Web 均重新生产构建，来源前后相同，输出与实际权限清单绑定。
5. 首次冻结被许可文件的 Windows 换行转换阻止；已让八份上游文本保持原字节，并逐份对照固定 SHA-256。失败日志保留，未放宽哈希要求。

| 检查 | 实际结果与范围 |
| --- | --- |
| 桌面 Python unittest | 最终 123 项通过，0 失败、0 跳过；`python-unittest-final.log` |
| 前端 Vitest | 163 项通过；`web-unit.json` |
| TypeScript | 通过；`web-types.log` |
| Rust 下载规则 | 6 项通过，覆盖取消、同源兄弟下载、失败与终态等；不代签原生回调 |
| 打包来源测试 | 17 项通过 |
| 冻结包科研流程 | 15 项通过、17 份冻结导出；为无头真实 API 工作流，非原生操作 |
| 六格式与中文长路径 | 后台字节及哈希检查通过；不是原生保存框验收 |
| 包与旧证据保护 | `integrity.json` 通过 |
| 代理工具离线检查 | 最后 7 项通过；不包含原生启动或下载验收 |

上述本轮结果位于 `outputs/desktop-part2-20261104/`；后台合成试验为 `~/LabVizD2/p2bg1104`。没有把紧凑 viewport、无头 WebGL、HTTP 上传或文件系统写入当作原生通过。

## 当前阻点与未运行项目

### PDF 辅助驱动

`part2_headless_acceptance.mjs` 在六格式检查后仍失败，尚未进入本轮备份／两副本检查。

- 原驱动使用 HTTP 请求读取 `blob:` 地址，因协议不支持而失败。
- 已修改为同源临时文档读取并校验 PDF 类型／文件头及真实源内容；重跑在捕获到的源 HTTP body 为 0 字节时失败，尚未执行后续 blob 校验。
- 这仍是未解决的驱动／响应观察问题，尚无证据判定为产品 PDF 渲染故障；本轮 PDF 不记通过。
- 原失败保留在 `part2-headless.failed-blob.json` 与 `headless-part2.failed-blob.log`；最新失败见 trial 的 `part2-headless.json` 和输出目录 `headless-part2.retry-blob.log`。

### 原生下载故障注入工具

已准备透明代理、受控目标文件锁与短时剪贴板占用工具，均放在输出目录的 `tools/`。方案保留精确宿主与真实控制器，代理只改变测试传输；即使后续通过，也须与直连整包证据区分。

当前 `download-proxy-variant/` 仅为半成品，没有 `variant-manifest.json`；`~/LabVizD2/p2proxy1104` 没有准备完成标记，不能启动。第一次长路径复制失败和封存半成品保留；重试准备进程已按用户要求停止。

最后离线工具报告在 `~/LabVizD2/proxy-selftest-0ef7d86746c6/offline-verification.json`。`tools/tool-verification.json` 仍是更早的六项及旧工具哈希快照，不代表当前版本的完整证明。剪贴板占用工具未执行。

### 安装与原生矩阵

- 匹配 Setup 未完成编译，实际安装、修复、跨 epoch 升级／回滚、重装及安装后原生重开均未执行。
- 真实 epoch 往返驱动 `tools/verify_epoch_install.py` 仅完成语法与入口检查，未运行安装。
- `tools/prepare_native.py`、`launch_native.py`、`native_support.py` 已准备；尚未建立或启动新的原生资料环境。
- 打开取消、六格式原生取消／覆盖、写入失败恢复、同源并发、校验后选择较早副本、实际副本文件框导入、剪贴板失败、外链、原生 3D、完整键盘和真实 200% 仍须逐项执行。

## 用户停止时的收尾

- 后台试验通过正常维护协议保存、停服、退出；host 退出码 0，未残留其所属进程。
- 服务回执为 `normal=true`、`forced=false`、`servicesStopped=true`；原始 API／Web 退出值为 0／1，原回执保留，未改写为全部 0。
- 系统剪贴板的原七种格式已从内存恢复；原内容未打印、未写入文件。没有执行应用内复制／粘贴验收。
- 仅停止了本轮安装器编译和未完成的代理准备进程；没有强制终止应用服务。
- 没有启动新的原生测试窗口、实际安装、变更真实快捷方式、签名、推送或发布。

## 继续入口

先读 `outputs/desktop-part2-20261104/stop-evidence-index.json` 与主目录本地 `V2.0/V2.3_PLAN.md`。

1. 审阅并完成未提交的 PDF 驱动修正，保留已记录失败，不放宽内容一致性要求。
2. 已冻结包保持不动；继续编译时使用新的 build 输出目录，复制并校验既有 candidate／support 输入，保留本次非零编译回执和 `.incomplete` 文件。
3. 用通过的合成科研资料准备全新原生试验；实际执行原生清单，逐项绑定候选身份及证据。
4. 代理变体先封存本轮半成品，再用新目录／label 完成 prepare 和 verify；机制故障注入另记范围。
5. 匹配 Setup 编译成功后再运行精确旧 epoch1 包到新 epoch2 包的往返驱动，以及安装后真正原生重开。
6. 全部第二部分门槛通过后才能签收。第一部分十轮及第三、四部分保持原有证据边界。
