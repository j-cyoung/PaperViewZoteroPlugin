# Changelog

All notable changes to this project will be documented in this file.

## v0.5.1 - 2026-02-12

**Added**
- Added a Zotero auto-update feed (`updates.json`) and wired `manifest.json#update_url` to enable automatic plugin updates.

## v0.5.0 - 2026-02-12

**Added**
- Added a unified plugin-side error taxonomy and alert flow across Query/Concat/OCR/History and service startup actions.
- Added runtime diagnostics APIs (`/runtime/check`) with key-source visibility (`request`/`prefs`/`env`/`none`) and remote connectivity probing.
- Added settings actions: `Check Runtime`, `Test Connection`, and `Restart Service`.
- Added plugin-side automated tests focused on error classification and runtime-check mapping.

**Changed**
- Updated settings-page status output to a dedicated multi-line area under action buttons, with line wrapping for long diagnostics.
- Standardized settings-page validation/status messages to English and aligned runtime hints with current UI behavior.
- Enhanced service job status payloads with structured `error_code` values for better client-side mapping.

**Fixed**
- Improved failure handling for no-service, timeout, HTTP 4xx/5xx, remote auth failure, and remote unreachable scenarios.
- Added remote API probe fallback when query subprocess fails, producing more actionable error reasons.

## v0.4.0 - 2026-02-11

**Added**
- Introduced a full PaperView settings pane with explicit fields for service URL, API key, model runtime parameters, and reset/save actions.
- Added integration and contract tests for service endpoints, runtime config resolution, and preferences pane wiring.

**Changed**
- Moved PaperView to a more stable plugin-first runtime flow with synchronized settings/runtime config before query execution.
- Improved OCR cache behavior to process only selected items and merge selected-run outputs back into global cache incrementally.
- Upgraded query flow with multiline input support, markdown-capable result rendering, and stronger progress/status polling.

**Fixed**
- Fixed API key propagation and query subprocess environment injection to avoid missing-key runtime failures.
- Fixed preferences default rendering and normalization so blank/invalid values are auto-seeded and persisted safely.
- Fixed progress/error behavior to close windows on failure and display actionable diagnostics.

## v0.3.8 - 2026-02-11

**Changed**
- Query runtime now synchronizes PaperView settings to runtime config/environment right before execution, reducing stale config issues.
- Ingest summary logging now reports service-native fields (`written`/`received`) to avoid misleading `ingested_keys=0` diagnostics.

**Fixed**
- Local query pipeline now resolves API key from request/config/environment consistently and injects it into query subprocess runtime.
- Query failure dialogs now use a reliable prompt path and show concrete error messages instead of blank/ambiguous alerts.

**Added**
- Runtime config tests for API key resolution priority and environment fallback behavior.

## v0.3.7 - 2026-02-11

**Changed**
- Expanded the explicit PaperView settings pane from a minimal subset to full runtime configuration coverage, including temperature, max output tokens, query concurrency, retry wait, and retry-on-429.
- Updated pane state normalization and persistence so all supported settings are validated and written consistently to Zotero prefs and `llm_config.json`.

**Added**
- UI contract tests for the preferences pane structure and explicit onload controller hook.
- Runtime configuration resolution tests for OCR concurrency fallback precedence.
- Project skill `paperview-reliability-playbook` documenting proven fixes for right-click menu initialization and settings default rendering issues.

## v0.3.6 - 2026-02-11

**Changed**
- Replaced the PaperView preferences pane with a minimal explicit form (`Service Base URL`, `API Key`, `LLM Base URL`, `Model`, `OCR Concurrency`) plus `Save` and `Reset Defaults`.
- Preferences pane initialization now follows an explicit `onload` controller pattern (`Zotero.PaperViewPrefsPane.init(window)`), similar to Better BibTeX style, instead of relying on implicit field binding.

**Fixed**
- Defaults are now rendered directly into input fields by pane controller code and persisted immediately, avoiding blank-field behavior caused by binding timing/type inconsistencies.

## v0.3.5 - 2026-02-11

**Fixed**
- Preferences pane now explicitly loads and normalizes all field values via `prefs/prefs.js`, ensuring default values are visible even when Zotero preference binding leaves inputs blank.
- Preference fields now auto-correct empty/invalid values on pane load and on edit, then persist sanitized values back to Zotero prefs.

## v0.3.4 - 2026-02-11

**Fixed**
- Default settings seeding now treats empty/invalid preference values as missing and rewrites them with safe defaults, so the Settings panel shows prefilled values consistently.
- LLM config normalization now enforces positive `max_output_tokens` and non-negative `retry_wait_s`, avoiding empty numeric fields caused by invalid zero values.
- Added detailed startup logs listing which preference keys were seeded, to simplify preference initialization debugging.

## v0.3.3 - 2026-02-11

**Fixed**
- Settings pane now auto-seeds default PaperView runtime preferences when values are missing or invalid, so fields are prefilled on first use.
- Default preference seeding now runs both at plugin startup and on main window load to avoid timing-related initialization misses.
- Preferences pane XML label escaping fixed to prevent settings page load failures.

## v0.3.1 - 2026-02-11

**Added**
- Integration test suite for local service endpoints, including ingest/query/ocr/status/history and path traversal checks.
- Repository skills for post-edit XPI build and commit workflow with changelog/README sync requirements.

**Changed**
- `OCR Cache` now processes only currently selected Zotero items instead of refreshing all ingested items.
- OCR cache update now merges selected-run outputs back into the global OCR cache file incrementally.
- Query/OCR progress polling and query command flow in `bootstrap.js` were refactored to reduce duplicated logic.

**Fixed**
- Environment bootstrap can recover from a failed initialization and retry later instead of staying stuck.
- CLI flags `--page_chunks` and `--dedupe` now support explicit disable forms (`--no-page_chunks`, `--no-dedupe`).
- Query result normalization fixed redundant token/elapsed fallback expressions.

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
