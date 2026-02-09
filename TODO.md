# PaperView Zotero 插件集成 TODO

**已锁定假设（中文备注）**
- [x] 使用 Zotero `item key` 作为 `paper_id`
- [x] OCR 输出沿用 `papers.pages.jsonl`（必要时可添加 Zotero 相关键值）
- [x] 结果以 `http://localhost:<port>/...` 提供

**阶段 0 - 范围确认**
- [x] 确认 Zotero 8 插件技术路线（WebExtension + bootstrap/background）
- [x] 明确最低支持平台：macOS
- [x] 本地服务启动方式：手动启动

**阶段 1 - 最小 Demo 验证**
- [x] 插件能被加载与卸载（验证可安装与调试链路）
- [x] 右键菜单出现 `Query`（验证 UI 注入）
- [x] 能拿到选中条目 key（验证数据可读性）

**阶段 2 - 本地服务骨架**
- [x] 本地服务启动并可访问（`/health`）
- [x] 插件调用服务并打开返回 URL（验证端到端链路）

**阶段 3 - Zotero 数据接入**
- [x] 插件收集元信息 + PDF 路径并发送
- [x] 服务端保存入库快照（用于后续 OCR/查询）

**阶段 4 - OCR 接入**
- [x] 复用现有 OCR 流程产出 `papers.pages.jsonl`
- [x] OCR 支持增量更新（resume + source_hash）

**阶段 5 - Query Pipeline**
- [x] 插件弹出查询输入框并发送 query
- [x] 后台接收 query 并打印查询内容
- [x] 为每次查询创建 job 文件夹（`store/zotero/query/<job_id>`），写入查询与条目快照
- [x] OCR 结果只缓存到 `store/zotero/ocr`，查询时复用并增量补齐
- [x] 复用 `query_papers.py` 生成查询结果并写入 job 文件夹
- [x] 结果页面通过本地服务展示（`/result/<job_id>`）
- [x] 结果页使用 `query_view.html` 完整 UI（默认当前查询）
- [x] 历史查询列表 + 切换（`/history` + 前端下拉）
- [x] 端到端验证：Zotero 右键 → 查询 → 生成结果页

**阶段 7 - 配置与稳定性**
- [x] 插件配置项（并发数、模型、base_url 等）
- [x] 插件配置项：服务端口 / service_base_url
- [x] 基本错误处理与提示

**阶段 8 - 打包与文档**
- [x] 打包 `.xpi`
- [x] 简要安装与运行说明

**阶段 9 - 增强优化**
- [ ] 验证 OCR 并发是否正确实现（正确性 + 性能收益）
- [ ] 查询输入支持多行文本输入
- [ ] 最终结果网页支持 Markdown 渲染

**待确认问题**
- [ ] 是否需要按 Zotero collection 进行结果分桶（可选）
- [ ] 是否需要插件自动启动本地服务（可选）
