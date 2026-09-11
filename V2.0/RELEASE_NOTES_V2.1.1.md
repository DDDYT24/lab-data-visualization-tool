# LabViz V2.1.1 / 本地安全更新

V2.1.1 是 V2.1.0 的安全补丁，不改变科研工作流、项目数据格式或本地保存路径。

版本功能总览见 [`../VERSION_BASELINE.md`](../VERSION_BASELINE.md)，后续待办和完成状态只以
[`TODO.md`](TODO.md) 为准。

## 更新内容

- Next.js 从 16.2.12 更新至 16.3.4，修复影响 Windows 本地服务的远程代码执行公告。
- Sharp 从 0.35.3 更新至 0.35.4，修复图像处理依赖中的高危公告。
- Nanoid 从 3.3.16 更新至 3.3.18，修复自定义生成器在零长度输入下可能无限循环的问题。
- 更新 Vitest、brace-expansion 和 js-yaml，关闭开发与测试工具链中的已知公告。
- API 和网站版本元数据统一更新为 2.1.1。

## 部署与数据边界

仍使用根目录启动脚本运行。SQLite、本地对象目录和 `127.0.0.1` 回环绑定保持不变，
不需要 AWS、PostgreSQL、MinIO 或 Docker。现有 V2.1.0 本地数据无需迁移。

## 验证

- 完整 npm 安全审计无已知漏洞；
- 前端 lint、TypeScript、45 项单元测试、生产构建和 Playwright 浏览器回归通过；
- 243 项 API/PostgreSQL/MinIO 测试、Ruff、格式和 MyPy 检查通过；
- GitHub 主分支 CI 和容器镜像构建通过。
