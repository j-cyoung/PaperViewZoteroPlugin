# PaperView

## 中文

**项目简介**
PaperView 是一个面向论文检索与批处理的本地流水线项目，集成 Zotero 插件用于选中文献发起查询，并在本地服务中完成 OCR、LLM 查询与结果可视化。

**功能概览**
- Zotero 右键菜单 `Query` 发起查询
- 可选查询章节（如 `abstract`、`introduction`、`methods`）
- 不指定章节时默认查询全文（`full_text`）
- 进度条展示查询状态
- 结果页面支持历史查询切换
- OCR 缓存与增量更新（复用已生成结果）

**环境要求**
- macOS
- Zotero 8.x
- Python 3.10+
- LLM API Key（`SILICONFLOW_API_KEY` 或 `OPENAI_API_KEY`）

**快速开始**
1. 启动后端服务（推荐端口 20341）
```bash
python local_service.py --port 20341
```
2. 打包插件并安装
```bash
./scripts/build_xpi.sh
```
3. 在 Zotero 插件管理器中拖入 `paperview-query.xpi` 安装
4. 设置服务地址：Zotero 顶部菜单 `Tools` → `PaperView: Set Service URL`
   例如：`http://127.0.0.1:20341`
5. 在 Zotero 文献列表中右键选择 `Query` 发起查询

**查询输入格式**
- 直接输入问题（默认全文）：
  `请总结方法`
- 指定章节：
  `[method] 总结方法`
  `[abstract,introduction] 请翻译成英文`

**章节名称示例**
- `abstract` / `摘要`
- `introduction` / `引言`
- `related_work` / `background` / `相关工作`
- `methods` / `method` / `approach` / `方法`
- `experiments` / `evaluation` / `results` / `实验`
- `conclusion` / `discussion` / `总结`
- `full_text` / `全文` / `全部`

**数据与结果路径**
- `store/zotero/items.jsonl`：Zotero 元信息快照
- `store/zotero/items.csv`：OCR 输入
- `store/zotero/ocr/papers.pages.jsonl`：OCR 输出
- `store/zotero/query/<job_id>/`：查询结果与中间文件

**OCR 流程**
```bash
./scripts/ocr_from_zotero.sh
```
输出：`store/zotero/ocr/papers.pages.jsonl`

**结果可视化**
- 查询完成后自动打开 `http://127.0.0.1:20341/result/<job_id>`
- 历史查询页面：`http://127.0.0.1:20341/query_view.html`

**配置项**
- 服务地址：`Tools` → `PaperView: Set Service URL`
- 本地服务端口：`local_service.py --port <PORT>`（默认 20341）
**API Key（推荐用插件设置）**
- Zotero 菜单：`Tools` → `PaperView: Set API Key`
**LLM 配置（文件 + 菜单）**
- Zotero 菜单：`Tools` → `PaperView: LLM Settings`
- 配置文件：`<ZoteroProfile>/paperview/llm_config.json`
**插件日志（Profile 目录）**
- 服务输出：`<ZoteroProfile>/paperview/logs/service.log`
- 环境安装：`<ZoteroProfile>/paperview/logs/env-install.log`
- pip 详细日志：`<ZoteroProfile>/paperview/logs/pip-install.log`

**常见问题**
- 端口被占用：`lsof -nP -iTCP:20341 -sTCP:LISTEN` 后 `kill <PID>`
- 浏览器未自动打开：检查服务是否运行、端口是否一致
- 查询进度不更新：确认 `local_service.py` 与 `query_papers.py` 已更新

**许可证**
- MIT License（见 `LICENSE`）

**项目说明**
- 本项目诞生于我对文献进行批量查询的需求（比如查询每篇文章使用了什么样的算力等），用于前期的文献调研、文献综述（帮老师写本子）等任务。完全通过vibe coding开发，随缘更新与迭代～

---

## English

**Overview**
PaperView is a local pipeline for paper retrieval and batch analysis. It integrates a Zotero plugin to trigger queries, runs OCR and LLM querying in a local service, and visualizes results in a web page.

**Key Features**
- Zotero context menu `Query`
- Optional section targeting (`abstract`, `introduction`, `methods`)
- Defaults to full text (`full_text`) when no section is specified
- Progress window during query
- History page to switch between past queries
- OCR caching with incremental updates

**Requirements**
- macOS
- Zotero 8.x
- Python 3.10+
- LLM API Key (`SILICONFLOW_API_KEY` or `OPENAI_API_KEY`)

**Quick Start**
1. Start the backend service (recommended port 20341)
```bash
python local_service.py --port 20341
```
2. Build and install the plugin
```bash
./scripts/build_xpi.sh
```
3. Drag `paperview-query.xpi` into Zotero’s Add-ons manager
4. Set the service URL in Zotero: `Tools` → `PaperView: Set Service URL`
   Example: `http://127.0.0.1:20341`
5. Right-click items in Zotero and choose `Query`

**Query Input Format**
- Direct question (defaults to full text):
  `Summarize the method`
- With section prefix:
  `[method] Summarize the method`
  `[abstract,introduction] Please translate to English`

**Section Names**
- `abstract`
- `introduction`
- `related_work` / `background`
- `methods` / `method` / `approach`
- `experiments` / `evaluation` / `results`
- `conclusion` / `discussion`
- `full_text` / `all`

**Data Paths**
- `store/zotero/items.jsonl` ingest snapshot
- `store/zotero/items.csv` OCR input
- `store/zotero/ocr/papers.pages.jsonl` OCR output
- `store/zotero/query/<job_id>/` query outputs

**OCR Pipeline**
```bash
./scripts/ocr_from_zotero.sh
```
Output: `store/zotero/ocr/papers.pages.jsonl`

**Visualization**
- Result page: `http://127.0.0.1:20341/result/<job_id>`
- History page: `http://127.0.0.1:20341/query_view.html`

**Configuration**
- Service URL: `Tools` → `PaperView: Set Service URL`
- Service port: `local_service.py --port <PORT>` (default 20341)
**API Key (recommended via plugin)**
- Zotero menu: `Tools` → `PaperView: Set API Key`
**LLM Config (file + menu)**
- Zotero menu: `Tools` → `PaperView: LLM Settings`
- Config file: `<ZoteroProfile>/paperview/llm_config.json`
**Plugin Logs (Profile directory)**
- Service output: `<ZoteroProfile>/paperview/logs/service.log`
- Env setup: `<ZoteroProfile>/paperview/logs/env-install.log`
- pip details: `<ZoteroProfile>/paperview/logs/pip-install.log`

**Troubleshooting**
- Port in use: `lsof -nP -iTCP:20341 -sTCP:LISTEN` then `kill <PID>`
- Browser not opening: verify service is running and URL matches
- Progress not updating: ensure `local_service.py` and `query_papers.py` are updated

**License**
- MIT License (see `LICENSE`)
