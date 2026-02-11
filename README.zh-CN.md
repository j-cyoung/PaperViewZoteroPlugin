# PaperView

英文版: [README.md](README.md)

## 项目简介

PaperView 是一个面向论文检索与批处理的本地流水线项目，集成 Zotero 插件用于选中文献发起查询，并在本地服务中完成 OCR、LLM 查询与结果可视化。

## 功能概览

- Zotero 右键菜单 `Query` / `Concat Query` / `OCR Cache`
- 可选查询章节（如 `abstract`、`introduction`、`methods`）
- 不指定章节时默认查询全文（`full_text`）
- 右键操作会自动启动后台服务（若当前未运行）
- 查询输入支持多行文本（换行保留）
- 进度条展示查询状态
- 结果页面支持 Markdown 渲染/原文切换
- 结果页面支持历史查询切换、删除单条、清空历史
- OCR 缓存与增量更新（复用已生成结果）
- OCR 默认并行执行，支持配置并发度

## 环境要求

- macOS
- Zotero 8.x
- Python 3.10+（用于插件创建 venv）
- LLM API Key（`SILICONFLOW_API_KEY` 或 `OPENAI_API_KEY`）

## 快速开始

1. 打包插件并安装。

```bash
./scripts/build_xpi.sh
```

2. 在 Zotero 插件管理器中拖入 `paperview-query.xpi` 安装并重启。
3. 设置服务地址：Zotero 顶部菜单 `Tools` -> `PaperView: Set Service URL`。
   例如：`http://127.0.0.1:20341`
4. 设置 API Key：`Tools` -> `PaperView: Set API Key`
5. （可选）手动启动服务：`Tools` -> `PaperView: Start Service`
6. 在 Zotero 文献列表中右键选择 `Query` / `OCR Cache` 发起任务（若服务未启动会自动拉起）

## 查询输入格式

- 支持多行输入（可直接换行，`Ctrl/Cmd + Enter` 提交）
- 直接输入问题（默认全文）：
  `请总结方法`
- 指定章节：
  `[method] 总结方法`
  `[abstract,introduction] 请翻译成英文`

## 章节名称示例

- `abstract` / `摘要`
- `introduction` / `引言`
- `related_work` / `background` / `相关工作`
- `methods` / `method` / `approach` / `方法`
- `experiments` / `evaluation` / `results` / `实验`
- `conclusion` / `discussion` / `总结`
- `full_text` / `全文` / `全部`

## 数据与结果路径

- `store/zotero/items.jsonl`：Zotero 元信息快照
- `store/zotero/items.csv`：OCR 输入
- `store/zotero/ocr/papers.pages.jsonl`：OCR 输出
- `store/zotero/query/<job_id>/`：查询结果与中间文件

## OCR 流程

```bash
./scripts/ocr_from_zotero.sh
```

输出：`store/zotero/ocr/papers.pages.jsonl`

## 结果可视化

- 查询完成后自动打开 `http://127.0.0.1:20341/result/<job_id>`
- 历史查询页面：`http://127.0.0.1:20341/query_view.html`
- 回答支持 Markdown 渲染/原文切换查看
- 历史支持删除当前记录与清空全部记录

## 配置项

- 服务地址：`Tools` -> `PaperView: Set Service URL`
- 本地服务端口：`local_service.py --port <PORT>`（默认 20341）

## API Key（推荐用插件设置）

- Zotero 菜单：`Tools` -> `PaperView: Set API Key`

## LLM 配置（文件 + 菜单）

- Zotero 菜单：`Tools` -> `PaperView: LLM Settings`
- 配置文件：`<ZoteroProfile>/paperview/llm_config.json`
- 可配置 `ocr_concurrency`（OCR 并发度，默认 `4`）

## 支持的 API 风格

- OpenAI-compatible Chat Completions（`/chat/completions`）

## 插件日志（Profile 目录）

- 服务输出：`<ZoteroProfile>/paperview/logs/service.log`
- 环境安装：`<ZoteroProfile>/paperview/logs/env-install.log`
- pip 详细日志：`<ZoteroProfile>/paperview/logs/pip-install.log`

## 常见问题

- 端口被占用：`lsof -nP -iTCP:20341 -sTCP:LISTEN` 后 `kill <PID>`
- 浏览器未自动打开：检查服务是否运行、端口是否一致
- 查询进度不更新：确认 `local_service.py` 与 `query_papers.py` 已更新
- OCR 看起来变慢：检查 `service.log` 是否出现 `switch force-serial mode`（并发错误触发后会降级串行）

## 许可证

- MIT License（见 `LICENSE`）
