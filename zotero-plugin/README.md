# PaperView Zotero 插件 / PaperView Zotero Plugin

## 中文

**Demo 行为**
- 在 Zotero 文献列表右键菜单增加 `Query`
- 向本地服务 `/ingest` 发送元信息与 PDF 路径（仅存储型附件）
- 调用 `/query` 并打开返回的 `result_url`
- 查询过程中显示进度条（轮询 `/status/<job_id>`）

**安装（打包成 .xpi）**
1. 在仓库根目录执行：
```bash
./scripts/build_xpi.sh
```
2. 在 Zotero 插件管理器中拖入 `paperview-query.xpi` 安装

**服务启动/停止（插件内）**
- Zotero 顶部菜单：`Tools` → `PaperView: Start Service` / `PaperView: Stop Service`
- 插件加载时会自动准备 Python 环境（首次可能较慢）
- Zotero 退出时会自动停止服务
**API Key 设置**
- Zotero 顶部菜单：`Tools` → `PaperView: Set API Key`
- 插件会将 API Key 传入服务进程（等价于设置 `OPENAI_API_KEY`/`SILICONFLOW_API_KEY`）
**LLM 配置（文件 + 菜单）**
- 菜单：`Tools` → `PaperView: LLM Settings`
- 配置文件：`<ZoteroProfile>/paperview/llm_config.json`（默认基于当前配置自动生成）

**日志位置（便于调试）**
- 服务输出：`<ZoteroProfile>/paperview/logs/service.log`
- 环境安装：`<ZoteroProfile>/paperview/logs/env-install.log`
- pip 详细日志：`<ZoteroProfile>/paperview/logs/pip-install.log`

**手动启动后端服务（可选）**
```bash
python local_service.py --port 20341
```

**设置端口/服务地址**
Zotero 顶部菜单：`Tools` → `PaperView: Set Service URL`  
示例：`http://127.0.0.1:20341`

**查询输入格式**
- 默认全文：`请总结方法`
- 指定章节：`[method] 总结方法`

---

## English

**Demo Behavior**
- Adds `Query` to Zotero item context menu
- Sends metadata + PDF path to `/ingest`
- Calls `/query` and opens `result_url`
- Shows a progress window (polls `/status/<job_id>`)

**Install (build .xpi)**
1. From repo root:
```bash
./scripts/build_xpi.sh
```
2. Drag `paperview-query.xpi` into Zotero Add-ons manager

**Start/Stop Service (in Zotero)**
- Zotero menu: `Tools` → `PaperView: Start Service` / `PaperView: Stop Service`
- Env bootstrap runs automatically after install (first run may take time)
- Service stops automatically when Zotero quits
**API Key**
- Zotero menu: `Tools` → `PaperView: Set API Key`
- The key is injected into the service process as `OPENAI_API_KEY` / `SILICONFLOW_API_KEY`
**LLM Config (file + menu)**
- Menu: `Tools` → `PaperView: LLM Settings`
- Config file: `<ZoteroProfile>/paperview/llm_config.json` (auto-generated from current settings)

**Logs (for debugging)**
- Service: `<ZoteroProfile>/paperview/logs/service.log`
- Env setup: `<ZoteroProfile>/paperview/logs/env-install.log`
- pip details: `<ZoteroProfile>/paperview/logs/pip-install.log`

**Manual Start (optional)**
```bash
python local_service.py --port 20341
```

**Configure Service URL**
Zotero menu: `Tools` → `PaperView: Set Service URL`  
Example: `http://127.0.0.1:20341`

**Query Input Format**
- Default full text: `Summarize the method`
- With section prefix: `[method] Summarize the method`
