# Changelog

All notable changes to this project will be documented in this file.

## v0.3.0 - 2026-02-11

**Added**
- OCR now runs in parallel by default, with configurable `ocr_concurrency` from menu/config/request.
- Query prompt upgraded to multi-line input (newline support, `Ctrl/Cmd + Enter` submit, `Esc` cancel).
- Result page now supports Markdown rendering with rendered/raw toggle.
- History page now supports deleting one record and clearing all history (`/history/delete`, `/history/clear`).
- Right-click actions (`Query`, `Concat Query`, `OCR Cache`, `Query History`) now auto-start the backend service when unavailable.

**Changed**
- OCR logging is more detailed (thread name, start/end, parallel retry, serial fallback events) for debugging performance and stability.
- OCR progress/status reporting is more consistent.

**Fixed**
- `full_text` queries now fall back to `md_text`/`abstract` when `md_pages` is missing, reducing `no_sections` failures.
- Multi-line query modal input box no longer overflows its container.
- History/result page interaction fixes (button state, copy behavior, load fallback).

## v0.2.0 - 2026-02-05

**Added**
- Built-in service lifecycle in the Zotero plugin (start/stop, auto-shutdown on exit).
- Automatic Python venv bootstrap and dependency installation on first run.
- LLM configuration file support at `<ZoteroProfile>/paperview/llm_config.json`.
- Menu-based LLM Settings and API Key configuration with write-back to config file.
- Service and environment logs in `<ZoteroProfile>/paperview/logs/`.
- OCR cache and concat query actions in the Zotero context menu.

**Changed**
- Service default port is now `20341`.
- Service and query scripts run via `sys.executable` (no `uv` requirement).
- LLM query parameters are configurable via CLI, config file, or menu.

**Fixed**
- Progress windows now close on query/OCR errors and surface failure messages.
- Chrome registration and file handling updated for Zotero 8 compatibility.

## v0.1.0 - 2026-02-05

**Added**
- Initial Zotero plugin with Query action and progress window.
- Local service endpoints for ingest and query.
- Basic OpenAI-compatible LLM querying with model/base URL options.
