# PaperView Zotero 插件（最小 Demo）

## 目录结构
- `manifest.json`
- `bootstrap.js`
- `locale/en-US/paperview.ftl`

## Demo 行为
- 在 Zotero 文献列表右键菜单增加 `Query`
- 点击后在 Zotero 调试输出中打印选中条目的 item key
- 向本地服务 `/ingest` 发送元信息与 PDF 路径（仅存储型附件）
- 调用 `/query` 并打开返回的 `result_url`
- 查询过程中显示进度条（轮询本地服务 `/status/<job_id>`）

## 编译与安装（打包成 .xpi）
1. 在仓库根目录执行以下命令生成安装包：

```bash
./scripts/build_xpi.sh
```

2. 在 Zotero 中打开插件管理器，然后把 `paperview-query.xpi` 拖入即可安装。

## 本地服务（最小 Demo）
在仓库根目录启动本地服务：

```bash
uv run python local_service.py --port 20341
```

服务启动后，右键点击 `Query` 会打开本地演示页面。

### 配置服务地址（端口）
在 Zotero 顶部菜单：`Tools` → `PaperView: Set Service URL`  
例如：`http://127.0.0.1:20341`

### 数据落盘位置
`/ingest` 接收到的数据会覆盖写入：`store/zotero/items.jsonl`

## OCR 流程（Zotero 数据）
将 `items.jsonl` 转为 OCR 输入 CSV，并生成兼容的 `papers.pages.jsonl`：

```bash
./scripts/ocr_from_zotero.sh
```

输出位置：
- `store/zotero/items.csv`
- `store/zotero/ocr/papers.pages.jsonl`

## 开发模式（可选）
如果你希望直接从源码加载（不用每次打包）：
1. 在 Zotero 配置文件的 `extensions` 目录下新建文件，文件名为插件 ID：

```
paperview-query@local
```

2. 文件内容是一行插件目录的绝对路径，例如：

```
/Users/you/Documents/work/study/PaperView/zotero-plugin
```

3. 重新启动 Zotero（必要时清理缓存）。

## 日志查看
- 打开 Zotero 的调试输出窗口，查看 `[PaperView]` 前缀日志。
