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

**启动后端服务**
```bash
uv run python local_service.py --port 20341
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

**Start Backend Service**
```bash
uv run python local_service.py --port 20341
```

**Configure Service URL**
Zotero menu: `Tools` → `PaperView: Set Service URL`  
Example: `http://127.0.0.1:20341`

**Query Input Format**
- Default full text: `Summarize the method`
- With section prefix: `[method] Summarize the method`
