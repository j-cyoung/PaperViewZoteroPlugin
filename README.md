# PaperView

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Overview

PaperView is a local pipeline for paper retrieval and batch analysis. It integrates a Zotero plugin to trigger queries, runs OCR and LLM querying in a local service, and visualizes results in a web page.

## Key Features

- Zotero context menu `Query` / `Concat Query` / `OCR Cache`
- Optional section targeting (`abstract`, `introduction`, `methods`)
- Defaults to full text (`full_text`) when no section is specified
- Right-click actions auto-start backend service when needed
- Multi-line query input support
- Progress window during query
- Result page supports Markdown rendered/raw toggle
- History page supports switch/delete/clear
- OCR caching with incremental updates
- OCR runs in parallel by default with configurable concurrency

## Requirements

- macOS
- Zotero 8.x
- Python 3.10+ (used for plugin venv)
- LLM API Key (`SILICONFLOW_API_KEY` or `OPENAI_API_KEY`)

## Quick Start

1. Build and install the plugin.

```bash
./scripts/build_xpi.sh
```

2. Drag `paperview-query.xpi` into Zotero's Add-ons manager and restart.
3. Set the service URL in Zotero: `Tools` -> `PaperView: Set Service URL`.
   Example: `http://127.0.0.1:20341`
4. Set API Key: `Tools` -> `PaperView: Set API Key`
5. (Optional) Start service manually: `Tools` -> `PaperView: Start Service`
6. Right-click items and run `Query` / `OCR Cache` (service auto-starts if not running)

## Query Input Format

- Multi-line input is supported (`Ctrl/Cmd + Enter` to submit)
- Direct question (defaults to full text):
  `Summarize the method`
- With section prefix:
  `[method] Summarize the method`
  `[abstract,introduction] Please translate to English`

## Section Names

- `abstract`
- `introduction`
- `related_work` / `background`
- `methods` / `method` / `approach`
- `experiments` / `evaluation` / `results`
- `conclusion` / `discussion`
- `full_text` / `all`

## Data Paths

- `store/zotero/items.jsonl` ingest snapshot
- `store/zotero/items.csv` OCR input
- `store/zotero/ocr/papers.pages.jsonl` OCR output
- `store/zotero/query/<job_id>/` query outputs

## OCR Pipeline

```bash
./scripts/ocr_from_zotero.sh
```

Output: `store/zotero/ocr/papers.pages.jsonl`

## Visualization

- Result page: `http://127.0.0.1:20341/result/<job_id>`
- History page: `http://127.0.0.1:20341/query_view.html`
- Markdown rendered/raw toggle for responses
- Delete one history record or clear all history from the history panel

## Configuration

- Service URL: `Tools` -> `PaperView: Set Service URL`
- Service port: `local_service.py --port <PORT>` (default 20341)

## API Key (recommended via plugin)

- Zotero menu: `Tools` -> `PaperView: Set API Key`

## LLM Config (file + menu)

- Zotero menu: `Tools` -> `PaperView: LLM Settings`
- Config file: `<ZoteroProfile>/paperview/llm_config.json`
- `ocr_concurrency` controls OCR worker parallelism (default `4`)

## Supported API Style

- OpenAI-compatible Chat Completions (`/chat/completions`)

## Plugin Logs (Profile directory)

- Service output: `<ZoteroProfile>/paperview/logs/service.log`
- Env setup: `<ZoteroProfile>/paperview/logs/env-install.log`
- pip details: `<ZoteroProfile>/paperview/logs/pip-install.log`

## Troubleshooting

- Port in use: `lsof -nP -iTCP:20341 -sTCP:LISTEN` then `kill <PID>`
- Browser not opening: verify service is running and URL matches
- Progress not updating: ensure `local_service.py` and `query_papers.py` are updated
- OCR looks slow: check `service.log` for `switch force-serial mode` (parallel failures can trigger serial fallback)

## License

- MIT License (see `LICENSE`)
