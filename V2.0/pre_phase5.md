# Phase 5 开工前架构确认

Status: Phase 5A approved for implementation on 2026-07-30.

## 结论

Phase 5 建议按 Phase 5A 和 Phase 5B 拆分推进。现有 Phase 4 架构可以承接，不需要重写。

核心选择如下：

- 使用版本化 HMAC 分享令牌。
- 分离可变的 `ExportJob` 和不可变的 `PublicationExport`。
- 使用 PostgreSQL 租约 Worker。
- 使用真实外键引用判断 `StoredObject` 的 GC 可达性。

Phase 5A is approved for implementation. Phase 5B remains deferred until Phase 5A is separately
verified and accepted.

## 一、现状确认

当前代码与设计之间有四个明确缺口：

1. 数据库设计已要求 `ShareLink` 固定到不可变的 `ProjectRevision`，并且数据库只保存 Token 摘要。参见 [`DATABASE_DESIGN.md`](DATABASE_DESIGN.md)。
2. `StoredObject` 已支持 `purpose=export`，但当前正式关系只有 `SourceFile` 和 `DatasetVersion`。
3. 当前 PostgreSQL GC 只统计 `SourceFile`、`DatasetVersion`，尚未统计 `Export` 和进行中的对象写入。
4. 分享和出版物导出仍走 SQLite `ProjectRepository`：保存原始 Token、读取当前项目版本、按整个项目寻找“最新导出”，不满足不可变 Revision 复现要求。

另外，当前工作区 Git 状态与“Phase 4 已提交”不一致：`0004`、Phase 4 文档和实现仍显示为未提交，最新提交是 Phase 3。Phase 5 实施前应先确认是否处在错误分支或缺少提交，并形成明确的可回退基线。

## 二、ShareLink 推荐方案

### 2.1 固定版本

`share_links` 必须同时保存：

- `project_id`
- `project_revision_id`
- 两者之间的同项目复合外键
- 创建人和创建时间
- 下载开关
- 撤销状态
- 可选过期时间

`project_revision_id` 创建后不可修改。项目后续产生新 Revision，不影响旧链接。需要分享新版本时，应创建新的 `ShareLink`，而不是移动旧链接。

公开页面必须从该 `ProjectRevision` 实时组装：

- `DatasetVersion`
- `CleaningDecisionSet`
- `ChartSpecRevision`

不能读取 `Project.current_revision`。

### 2.2 Token

常规方案是生成 32 字节随机 Token，只保存 SHA-256 摘要。但它无法兼容现有 workspace JSON：当前响应需要反复返回 `token` 和完整 `url`，数据库又不能保存原始 Token。

因此推荐采用可重建、版本化 HMAC Token：

```text
s1.<随机share_uuid>.<HMAC-SHA256(SHARE_TOKEN_KEY, share_uuid)>
```

数据库保存：

- `public_id` 或 `share_uuid`
- `token_digest = SHA-256(完整Token)`
- `token_key_version`
- 不保存完整 Token

该方案具有以下优点：

- 仍然满足“数据库只保存摘要”。
- 服务端可使用密钥重新生成 Token，兼容现有 JSON。
- 支持未来密钥轮换：新链接使用新版本，旧版本通过 key ring 验证。
- Phase 6 再将密钥迁移到正式 KMS；Phase 5 使用环境密钥，但不宣称生产 KMS 已完成。

验证流程：

1. 限制 Token 最大长度并解析版本。
2. 始终执行一次固定结构的数据库查询。
3. 使用对应版本密钥重新计算 Token。
4. 使用常量时间比较函数比较摘要。
5. 对不存在、格式错误、已撤销、已过期、项目已删除等状态统一返回相同的 404。

Token 使用严格规范化格式 `s1.<32位小写UUID十六进制>.<43位无填充base64url MAC>`。
服务端先从 Token 解析 `public_id`，再根据数据库记录的 `token_key_version` 选择密钥，
重建完整预期 Token，并与提交值做常量时间比较。项目内创建、修改和撤销分享始终要求
登录用户通过项目所有权检查；持有公开 Token 不授予管理权限。

密钥轮换采用“当前版本只签发新 Token、旧版本仅验证”的 key ring。只要仍有使用某版本
的 ShareLink，该旧密钥就不能移除；撤销或 purge 最后一个旧版本链接后，才可以在一次
受审计部署中移除旧密钥。Phase 5 使用环境变量提供 key ring，Phase 6 再接入生产 KMS。

同时应当：

- 禁止在访问日志、错误日志和分析事件中记录完整 Token。
- 为分享页面设置 `Referrer-Policy: no-referrer`。
- 为分享页面设置 `Cache-Control: no-store`。

### 2.3 状态关系

| ShareLink 状态 | Project 状态 | 访问结果 |
| --- | --- | --- |
| `active`、未过期 | 正常 | 可访问 |
| `active` | 软删除 | 立即不可访问，但不修改 ShareLink |
| `active` | 恢复 | 自动重新可访问 |
| `revoked` | 删除后恢复 | 仍不可访问 |
| `expired` | 删除后恢复 | 仍不可访问 |
| 任意状态 | 永久 purge | ShareLink 随项目删除 |

不建议增加 `suspended-by-delete` 状态，因为项目删除本身已经能够表达暂停。重复状态容易产生恢复竞态。

主动撤销是不可逆状态变化。重复 DELETE 可以幂等返回 204，但不能恢复已撤销链接。

### 2.4 下载权限

`downloads_enabled` 只允许下载固定到同一个 `project_revision_id` 的 PNG、SVG、PDF 出版物导出。

不允许公开下载：

- 原始上传文件
- Parquet
- `cleaned-data.csv`
- 其他 `ProjectRevision` 的导出

公开下载不能再使用当前的“项目最新导出”，也不能在请求时动态选择某个 Revision
下的最新产物。`ShareLink` 通过不可变绑定表固定到具体 `PublicationExport`：

- 每个 ShareLink 和格式最多绑定一个产物；
- 创建分享、开启下载或首次生成该格式时，只能在持锁事务中填补尚不存在的绑定；
- 已存在的绑定不可更新，也不会因后来生成新 Export 而自动切换；
- 下载只解析绑定表，且 PublicationExport 必须与 ShareLink 属于同一个 Project 和
  ProjectRevision。

### 2.5 过期时间

Schema 支持 nullable `expires_at`：

- `saved-cloud` 默认永久，即 `expires_at = NULL`。
- 当前 JSON 请求没有过期字段，因此 Phase 5A 不增加用户可配置过期功能。
- 未来可以通过可选字段进行向后兼容扩展。
- 当前创建分享要求登录；若项目仍是 `temporary-cloud`，应复用 Phase 4 的原地认领/Save 流程，再锁定 Revision 创建 ShareLink。

## 三、永久出版物导出

### 3.1 数据模型

不建议把运行状态和最终不可变产物混在同一张表中。推荐拆分为以下两个对象。

#### ExportJob：可变运行记录

- 项目和固定 `ProjectRevision`
- 请求者 User 或 Guest
- 请求摘要和 `Idempotency-Key`
- `queued`、`running`、`ready`、`failed` 状态
- 重试次数和下次重试时间
- 错误分类
- 当前 `ProcessingRun`
- 临时 `StoredObject` 写入意图

#### PublicationExport：不可变成功结果

- 一对一关联 `ExportJob`
- `project_revision_id`
- `dataset_version_id`
- nullable `cleaning_decision_set_id`
- `chart_spec_revision_id`
- 成功的 `processing_run_id`
- `stored_object_id`
- renderer 与完整格式信息

数据库约束或插入触发器应确认 `PublicationExport` 保存的 `DatasetVersion`、`CleaningDecisionSet`、`ChartSpecRevision`，正好等于 `ProjectRevision` 所引用的对象。

现有 `ProcessingRun.operation='export'` 已经存在，可以直接用于记录每次实际渲染尝试。

### 3.2 导出请求与 Revision

现有导出接口携带完整 `ChartSpec`，并且当前行为会保存图表。为了保持语义兼容：

- 如果提交的可视化配置与当前 Revision 相同，复用当前 Revision。仅格式、DPI、尺寸等
  PublicationExport 渲染参数不同，不创建新 Revision，而是记录在不可变 Export 上。
- 如果坐标轴、系列、标题、拟合或其他可视化配置不同，先创建新的 `ChartSpecRevision`
  和 `ProjectRevision`。
- `ExportJob` 固定到最终得到的 `ProjectRevision`。
- 后续项目编辑不影响已经生成的 `PublicationExport`。

### 3.3 格式和复现信息

`PublicationExport` 至少记录：

- `format` 和 MIME type
- renderer 名称、版本和构建版本
- render-contract 版本
- 完整、规范化的 `ExportSpec`
- 宽、高、原始单位、最终像素或页面尺寸
- DPI
- 字体、线宽、marker、legend、灰度和透明背景等设置
- 输入 `DatasetVersion` 内容哈希
- 输出 SHA-256、字节数和格式校验结果

PNG 的 DPI 是实际像素密度。SVG/PDF 的 DPI 主要约束栅格元素和兼容输出，因此仍必须单独保存物理页面尺寸。

### 3.4 保留策略

- `saved-cloud`：`expires_at = NULL`，保留到项目永久 purge。
- `temporary-cloud`：过期时间不晚于项目 `expires_at`。
- 项目软删除：导出立即不可访问，但外键和 `StoredObject` 引用继续存在。
- 项目恢复：导出重新可访问。
- Guest 项目后来 Save：不复制字节，项目关系继续有效。原 Guest `dedup_scope` 可以保留为更窄的安全域，后续对生命周期元数据进行提交后 reconciliation。

### 3.5 字节复用

允许多个 `PublicationExport` 引用同一个 `StoredObject`。渲染配置摘要只能用来寻找候选，
最终物理复用必须同时满足：

- 相同 `dedup_scope`
- 最终输出 SHA-256 完全相同
- 最终输出字节数完全相同
- `StoredObject` 已锁定且状态为 `available`

逻辑 Export 记录不复用身份，只复用不可变物理字节。

不同用户、不同 GuestSession 之间不复用，也不使用引用计数作为删除依据。对象键继续使用安全域和内容寻址信息，不能依赖 `project_id`。

### 3.6 失败、重试和幂等

建议为现有 POST export 增加可选 `Idempotency-Key` Header，不修改 JSON。

- 相同 actor、operation、key 和 request hash：返回同一个 `ExportJob`。
- 相同 key、不同请求：返回 409。
- 无 key：创建新的逻辑 `ExportJob`，但仍可能复用物理字节。
- 每次实际重试创建新的 `ProcessingRun`。
- 同一个 `ExportJob` 最终只能对应一个成功的 `PublicationExport`。

对象流程：

1. 渲染并暂存字节。
2. 校验格式、大小和 SHA-256。
3. 创建 pending `StoredObject` 和 `stored_object_write_intent`。
4. 提交数据库。
5. 确认最终对象。
6. 锁定 `StoredObject`，确认或设置为 `available`。
7. 在同一个数据库事务中锁定 available StoredObject、创建 `PublicationExport`、完成
   WriteIntent、将 ExportJob 标记为 ready，并填补尚不存在的 ShareLink 格式绑定。

数据库失败时删除暂存对象。对象确认失败时保留 pending intent，由 reconciliation 重试。

WriteIntent 在上述事务提交前始终是 GC 保留根。只有当 `StoredObject` 已经 `available`，
并且不可变 Export 行成功提交后，接口才能返回 `ready`。

## 四、生命周期 Worker

### 4.1 并发模型

推荐使用“数据库租约 + 行级抢占”：

- `worker_leases` 维护任务级租约、owner、数据库时间、过期时间和递增 fencing token。
- 任务级租约只控制扫描器；工作项拥有独立的 `lease_owner`、`lease_until` 和
  `fencing_token`。
- `FOR UPDATE SKIP LOCKED` 只在短事务中用于抢占并写入工作项租约。
- 长任务定期 heartbeat。
- 外部对象存储 I/O 不持有长数据库事务。
- 完成状态更新必须同时校验工作项 lease owner 和 fencing token。
- Worker 崩溃后，租约过期即可被另一实例接管。

相比 PostgreSQL advisory lock，租约表更易观察、审计和恢复，也能防止旧 Worker 在暂停后错误提交结果。

Worker 仍属于同一应用部署和代码库，只是独立运行命令，不是微服务。

### 4.2 任务行为

- **temporary expiry**：锁定 Project，重新确认 storage mode 和 expiry，再执行 purge。
- **saved purge**：锁定 Project，重新确认 `purge_after <= DB now()`，再永久删除。
- **pending reconciliation**：HEAD 暂存和最终对象，核对 SHA/size；确认、重试或 quarantine。
- **deleting retry**：锁定 `StoredObject`，确认零引用；对象存储返回 404 时视为删除成功。
- **orphan staging**：只删除超过安全宽限期，且不存在 `StoredObject.staging_key` 或 WriteIntent 的对象。
- **auth/session cleanup**：批量删除过期 challenge、session 和 GuestSession。
- **idempotency cleanup**：建议默认保留 24 小时后清理。
- **export retry**：只重试被分类为 transient 的失败；永久输入错误或渲染错误不自动重试。

所有任务都必须采用“至少执行一次，但效果幂等”的语义，不能假设 PostgreSQL 与对象存储之间存在原子事务。

### 4.3 S3 兼容适配

Phase 5B 扩展对象存储契约：

- stage/upload
- head
- confirm/copy-if-absent
- paginated list staging
- delete
- metadata SHA-256/size 校验

不能把 multipart ETag 当成 SHA-256。最终对象通过自有 SHA metadata 和 size 校验。

建议开发和测试使用 MinIO 验证 S3 兼容性。生产厂商、KMS、备份和合规仍留到 Phase 6。

## 五、GC 可达性

GC 的权威引用集合应包括：

- `source_files.stored_object_id`
- `dataset_versions.stored_object_id`
- `publication_exports.stored_object_id`
- 活跃的 `stored_object_write_intents.stored_object_id`
- `ExportJob` 的 pending 对象引用

规则如下：

1. pending/compensating 对象是保留根，不能进入普通 GC。
2. `SourceFile` 只要外键非空就是有效引用；原文件确认删除时，应同时清空外键并写入 `binary_deleted_at`。
3. 项目软删除期间不忽略任何引用。
4. 不维护权威引用计数。
5. purge 删除真实外键后，才把对象标记为 GC candidate。

清理顺序固定为：先 purge 已符合条件的 Project，再清理仍被 Project 引用的
GuestSession；Project 永久删除前记录其 StoredObject GC 候选；软删除期间全部真实 FK
继续算有效引用；GuestSession 过期本身不得绕过这些 FK 触发对象删除。

统一锁协议：

1. 新引用：`SELECT StoredObject FOR UPDATE`。
2. 确认对象状态为 `available` 且安全域正确。
3. 插入真实外键。
4. GC 锁定同一行并重新查询全部真实外键。
5. 零引用时执行 `available -> deleting`。
6. 状态提交后再执行外部删除。
7. 删除完成后重新锁行，再次确认零引用且状态为 `deleting`，最后标记为 `deleted`。

该协议确保“创建新引用”和“GC 删除”无法同时越过对方。

## 六、Phase 5 切片

| 切片 | 内容 | 停止条件 |
| --- | --- | --- |
| Phase 5A | ShareLink、ExportJob、PublicationExport、WriteIntent、现有分享/导出接口迁移 | 真实项目跨重启完成固定版本分享、PNG/SVG/PDF 永久导出、下载权限、删除/恢复/撤销闭环 |
| Phase 5B | 租约 Worker、完整 GC、pending/deleting/orphan 恢复、S3 兼容适配 | 两个并发 Worker 在崩溃、重试、网络失败下最终收敛，数据库引用与对象存储一致 |

### 6.1 Phase 5A Migration

建议命名为 `0005_share_publication_exports`。

新增或修改：

- `share_links`
- `share_link_events`
- `export_jobs`
- `publication_exports`
- `stored_object_write_intents`
- 扩展 `idempotency_records`：Guest actor、request hash、`expires_at`、部分唯一索引
- 不可变触发器和同项目复合外键
- `PublicationExport`/Revision 内容一致性约束

迁移接口：

- `POST /api/v1/projects/{projectId}/shares`
- `PATCH /api/v1/projects/{projectId}/shares/{token}`
- `DELETE /api/v1/projects/{projectId}/shares/{token}`
- `GET /api/v1/shares/{token}`
- `POST /api/v1/projects/{projectId}/exports`
- `GET /api/v1/exports/{exportId}/download`
- `GET /api/v1/shares/{token}/downloads/{format}`
- workspace 中的 shares 聚合

`cleaned-data.csv` 继续使用 Phase 3 的 `DatasetVersion` 数据闭环，不把它误归类为出版物导出。

测试：

- Token 摘要、不落原文、固定时间比较和统一错误
- Revision 固定与后续编辑隔离
- revoke/delete/restore/expiry 状态矩阵
- 下载权限和跨 Revision 隔离
- 完整导出 lineage、格式校验和不可变性
- Guest/saved 保留期
- 幂等冲突、重试和对象失败补偿
- 相同或不同安全域的对象复用
- SQLite/PostgreSQL JSON 响应兼容
- 原有全部回归测试

### 6.2 Phase 5B Migration

建议命名为 `0006_lifecycle_worker_gc`。

新增或修改：

- `worker_leases`
- `worker_checkpoints`，用于保存 S3 inventory cursor
- `StoredObject`、WriteIntent、ExportJob 的 retry、lease、`last_error`、`next_attempt` 字段
- 必要的 Worker 运行审计字段

不新增公开 JSON 接口，只增加内部 Worker 启动命令以及健康和监控信息。

测试：

- 双 Worker 抢占、`SKIP LOCKED` 和 fencing
- 租约过期接管和进程中断恢复
- 临时项目和已保存项目 purge
- pending confirm、deleting retry、404 幂等删除
- orphan staging 安全宽限期
- 新引用与 GC 并发竞态
- 软删除引用保留
- MinIO S3 合同测试
- 本地/S3 适配器一致性
- 全部 PostgreSQL、SQLite 和 API 回归

## 七、开工前需人工确认

进入实施前，建议确认以下三点：

1. 接受版本化 HMAC Token，以兼容“数据库只保存摘要”和现有 workspace JSON；否则必须接受 Token 只显示一次的接口语义变化。
2. 接受将可变 `ExportJob` 与不可变 `PublicationExport` 分表。
3. 先解决当前工作区 Phase 4 未形成 Git 提交基线的问题。

实施 Phase 5、真正编写代码前，需要先读取并遵循指定的 `coding_skills/SKILL.md`。
