/* global Zotero, Cc, Ci, Services, ChromeUtils */

var pluginID = null;
var SERVICE_BASE_URL = "http://127.0.0.1:20341";
var PREF_SERVICE_URL = "extensions.paperview.service_base_url";
var PREF_API_KEY = "extensions.paperview.api_key";
var PREF_LLM_CONFIG = "extensions.paperview.llm_config";
var PREF_LLM_BASE_URL = "extensions.paperview.llm.base_url";
var PREF_LLM_MODEL = "extensions.paperview.llm.model";
var PREF_LLM_TEMPERATURE = "extensions.paperview.llm.temperature";
var PREF_LLM_MAX_OUTPUT_TOKENS = "extensions.paperview.llm.max_output_tokens";
var PREF_LLM_CONCURRENCY = "extensions.paperview.llm.concurrency";
var PREF_LLM_OCR_CONCURRENCY = "extensions.paperview.llm.ocr_concurrency";
var PREF_LLM_RETRY_ON_429 = "extensions.paperview.llm.retry_on_429";
var PREF_LLM_RETRY_WAIT_S = "extensions.paperview.llm.retry_wait_s";
var cleanupHandlers = [];
var chromeHandle = null;
var serviceProcess = null;
var serviceStarting = false;
var serviceEnvPromise = null;
var serviceReadyPromise = null;
var preferencePaneID = null;
const ENABLE_CONTEXT_MENUS = true;
const ENABLE_TOOLS_SERVICE_MENU = true;
const PAPER_VIEW_ICON_FALLBACK_URL =
  "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/zotero.svg";
var addonRootURI = null;
var paperViewIconURL = null;
const SERVICE_FILES = [
  "local_service.py",
  "query_papers.py",
  "zotero_items_to_csv.py",
  "pdf_to_md_pymupdf4llm.py",
  "paper_utils.py",
  "query_view.html",
  "requirements.txt"
];
var MENU_LABELS = {
  query: "PaperView: Query",
  concat: "PaperView: Concat Query",
  ocr: "PaperView: OCR Cache",
  history: "PaperView: Query History",
  startService: "PaperView: Start Service",
  stopService: "PaperView: Stop Service"
};

const ENV_LOG_FILENAME = "env-install.log";
const PIP_LOG_FILENAME = "pip-install.log";
const LLM_CONFIG_FILENAME = "llm_config.json";
const PREF_PANE_SRC = "prefs/prefs.xhtml";

const DEFAULT_LLM_CONFIG = {
  base_url: "https://api.siliconflow.cn/v1",
  model: "Qwen/Qwen2.5-72B-Instruct",
  api_key: "",
  temperature: 0.0,
  max_output_tokens: 2048,
  concurrency: 5,
  ocr_concurrency: 4,
  retry_on_429: false,
  retry_wait_s: 300
};
const DEFAULT_SERVICE_REQUEST_TIMEOUT_MS = 15000;
const DEFAULT_STATUS_POLL_TIMEOUT_MS = 4000;
const DEFAULT_STATUS_POLL_MAX_ERRORS = 8;
const DEFAULT_RUNTIME_TEST_TIMEOUT_S = 20;
const PREFS_FIELD_IDS = [
  "paperview-service-url",
  "paperview-api-key",
  "paperview-llm-base-url",
  "paperview-llm-model",
  "paperview-llm-temperature",
  "paperview-llm-max-output-tokens",
  "paperview-llm-concurrency",
  "paperview-llm-ocr-concurrency",
  "paperview-llm-retry-wait"
];
function getPaperViewIconURL() {
  if (paperViewIconURL) return paperViewIconURL;
  if (addonRootURI) {
    try {
      const localURL = `${addonRootURI}skin/paperview.svg`;
      const svg = Zotero.File.getContentsFromURL(localURL);
      if (svg) {
        paperViewIconURL =
          "data:image/svg+xml;utf8," + encodeURIComponent(svg);
        return paperViewIconURL;
      }
    } catch (err) {
      Zotero.debug(`[PaperView] load local icon failed: ${err}`);
    }
  }
  paperViewIconURL = PAPER_VIEW_ICON_FALLBACK_URL;
  return paperViewIconURL;
}

function applyMenuIcon(menuitem) {
  if (!menuitem) return;
  const iconURL = getPaperViewIconURL();
  if (!iconURL) return;
  if (menuitem.classList) {
    menuitem.classList.add("menuitem-iconic");
  } else {
    const existing = menuitem.getAttribute("class") || "";
    menuitem.setAttribute("class", `${existing} menuitem-iconic`.trim());
  }
  menuitem.setAttribute("image", iconURL);
  menuitem.style.listStyleImage = `url("${iconURL}")`;
}

function getServiceBaseUrl() {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.get === "function") {
      const pref = Zotero.Prefs.get(PREF_SERVICE_URL);
      if (pref && typeof pref === "string") return pref;
    }
  } catch (err) {
    // ignore pref errors
  }
  if (SERVICE_BASE_URL && typeof SERVICE_BASE_URL === "string") {
    return SERVICE_BASE_URL;
  }
  return "http://127.0.0.1:20341";
}

function setServiceBaseUrl(value) {
  SERVICE_BASE_URL = value;
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.set === "function") {
      Zotero.Prefs.set(PREF_SERVICE_URL, value);
    }
  } catch (err) {
    // ignore pref errors
  }
}

function getApiKey() {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.get === "function") {
      const pref = Zotero.Prefs.get(PREF_API_KEY);
      if (pref && typeof pref === "string") return pref;
    }
  } catch (err) {
    // ignore pref errors
  }
  return "";
}

function setApiKey(value) {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.set === "function") {
      Zotero.Prefs.set(PREF_API_KEY, value || "");
    }
  } catch (err) {
    // ignore pref errors
  }
}

function showPaperViewAlert(message, title) {
  const alertTitle = title || "PaperView";
  const alertMessage = String(message || "").trim() || "Unknown error";
  try {
    if (Services && Services.prompt && typeof Services.prompt.alert === "function") {
      const win = Zotero.getMainWindow ? Zotero.getMainWindow() : null;
      Services.prompt.alert(win || null, alertTitle, alertMessage);
      return;
    }
  } catch (err) {
    // ignore and fallback
  }
  try {
    const win = Zotero.getMainWindow ? Zotero.getMainWindow() : null;
    if (win && typeof win.alert === "function") {
      win.alert(`${alertTitle}\n${alertMessage}`);
    }
  } catch (err) {
    // ignore fallback failures
  }
}

function asErrorText(err) {
  if (!err) return "";
  try {
    if (err.stack) return String(err.stack);
  } catch (stackErr) {
    // ignore
  }
  try {
    if (err.message) return String(err.message);
  } catch (messageErr) {
    // ignore
  }
  try {
    return String(err);
  } catch (stringErr) {
    return "<unprintable error>";
  }
}

function createPaperViewError(code, message, detail) {
  const err = new Error(message || code || "PaperView error");
  err.name = "PaperViewError";
  err.paperView = Object.assign({ code: code || "UNKNOWN_ERROR" }, detail || {});
  return err;
}

function isPaperViewError(err) {
  return !!(err && err.paperView && err.paperView.code);
}

function safeParseJson(text) {
  const payload = String(text || "").trim();
  if (!payload) return null;
  try {
    return JSON.parse(payload);
  } catch (err) {
    return null;
  }
}

function extractHttpStatus(err) {
  const candidates = [
    err && err.status,
    err && err.statusCode,
    err && err.responseStatus,
    err && err.xmlhttp && err.xmlhttp.status
  ];
  for (const candidate of candidates) {
    const status = Number(candidate);
    if (Number.isFinite(status) && status > 0) {
      return status;
    }
  }
  return null;
}

function extractResponseText(err) {
  const candidates = [
    err && err.responseText,
    err && err.response,
    err && err.xmlhttp && err.xmlhttp.responseText,
    err && err.xmlhttp && err.xmlhttp.response
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate;
    }
  }
  return "";
}

function isTimeoutLikeError(err) {
  const text = asErrorText(err).toLowerCase();
  return /timeout|timed out|time out|ns_error_net_timeout|etimedout/.test(text);
}

function isConnectionLikeError(err) {
  const text = asErrorText(err).toLowerCase();
  return /connection refused|econnrefused|networkerror|failed to fetch|name or service not known|enotfound|dns|ns_error_unknown_host|connection reset|econnreset/.test(text);
}

function mapServiceErrorCodeToPluginCode(serviceCode) {
  const code = String(serviceCode || "").trim().toLowerCase();
  if (!code) return null;
  const mapping = {
    api_key_missing: "API_KEY_MISSING",
    remote_api_timeout: "REMOTE_API_TIMEOUT",
    remote_api_unreachable: "REMOTE_API_UNREACHABLE",
    remote_api_auth: "REMOTE_API_AUTH",
    remote_api_http_error: "REMOTE_API_HTTP_ERROR",
    ocr_start_failed: "OCR_START_FAILED",
    ingest_required: "INGEST_REQUIRED",
    selection_empty: "NO_MATCHING_ITEMS",
    query_pipeline_failed: "QUERY_PIPELINE_FAILED",
    ocr_pipeline_failed: "OCR_PIPELINE_FAILED",
    bad_request: "BAD_REQUEST",
    internal_error: "INTERNAL_ERROR"
  };
  return mapping[code] || null;
}

function classifyServiceRequestError(err, options) {
  if (isPaperViewError(err)) return err;
  const opts = options || {};
  const status = extractHttpStatus(err);
  const payload = safeParseJson(extractResponseText(err)) || {};
  const serviceCode = payload.code || payload.error_code;
  const mappedCode = mapServiceErrorCodeToPluginCode(serviceCode);
  const payloadMessage =
    (typeof payload.error === "string" && payload.error) ||
    (typeof payload.message === "string" && payload.message) ||
    "";

  if (mappedCode) {
    return createPaperViewError(
      mappedCode,
      payloadMessage || asErrorText(err),
      { status, serviceCode, operation: opts.operation || "", raw: asErrorText(err) }
    );
  }
  if (status) {
    return createPaperViewError(
      status >= 500 ? "SERVICE_HTTP_5XX" : "SERVICE_HTTP_4XX",
      payloadMessage || `HTTP ${status}`,
      { status, operation: opts.operation || "", raw: asErrorText(err) }
    );
  }
  if (isTimeoutLikeError(err)) {
    return createPaperViewError("SERVICE_TIMEOUT", asErrorText(err), {
      operation: opts.operation || "",
      raw: asErrorText(err)
    });
  }
  if (isConnectionLikeError(err)) {
    return createPaperViewError("SERVICE_UNREACHABLE", asErrorText(err), {
      operation: opts.operation || "",
      raw: asErrorText(err)
    });
  }
  return createPaperViewError("REQUEST_FAILED", asErrorText(err), {
    operation: opts.operation || "",
    raw: asErrorText(err)
  });
}

function formatPaperViewError(err, options) {
  const opts = options || {};
  const wrapped = isPaperViewError(err)
    ? err
    : createPaperViewError("UNKNOWN_ERROR", asErrorText(err));
  const detail = wrapped.paperView || {};
  const code = detail.code || "UNKNOWN_ERROR";
  const status = Number(detail.status);
  const serviceUrl = getServiceBaseUrl();
  const source = detail.apiKeySource || "unknown";
  const logHint = `服务日志：${getLogPath()}`;
  const envHint = `环境安装日志：${getEnvLogPath()}`;
  const manualStartHint = "可在 Tools -> PaperView: Start Service 手动启动服务。";

  let title = opts.title || "PaperView 错误";
  let message = String(wrapped.message || "未知错误").trim() || "未知错误";

  switch (code) {
    case "API_KEY_MISSING":
      title = opts.title || "PaperView 配置缺失";
      message =
        "未检测到 API Key。\n请打开 Settings -> PaperView，填写 API Key 后点击 Save，然后重试。\n" +
        `当前检测来源：${source}`;
      break;
    case "SERVICE_START_FAILED":
      title = opts.title || "PaperView 服务启动失败";
      message =
        "本地服务未能成功启动。\n" +
        `${manualStartHint}\n${logHint}\n${envHint}`;
      break;
    case "SERVICE_NOT_READY":
      title = opts.title || "PaperView 服务未就绪";
      message =
        `本地服务在超时时间内未就绪（${serviceUrl}）。\n` +
        `${manualStartHint}\n${logHint}`;
      break;
    case "SERVICE_UNREACHABLE":
      title = opts.title || "PaperView 无法连接本地服务";
      message =
        `无法连接本地服务：${serviceUrl}\n` +
        `${manualStartHint}\n请检查 Service Base URL 配置。`;
      break;
    case "SERVICE_TIMEOUT":
      title = opts.title || "PaperView 服务响应超时";
      message = `本地服务请求超时：${serviceUrl}\n请稍后重试，或检查服务状态。\n${logHint}`;
      break;
    case "REMOTE_API_UNREACHABLE":
      title = opts.title || "PaperView 无法连接远程 API";
      message = "无法连接远程 API。请检查 LLM Base URL、网络和代理配置。";
      break;
    case "REMOTE_API_TIMEOUT":
      title = opts.title || "PaperView 远程 API 超时";
      message = "远程 API 响应超时。请检查网络质量，或增大重试等待时间后重试。";
      break;
    case "REMOTE_API_AUTH":
      title = opts.title || "PaperView 远程 API 鉴权失败";
      message = "远程 API 返回鉴权错误（通常是 API Key 无效或权限不足）。请检查 API Key。";
      break;
    case "REMOTE_API_HTTP_ERROR":
      title = opts.title || "PaperView 远程 API 错误";
      message = status ? `远程 API 返回 HTTP ${status}。请检查 Base URL、Model 与服务状态。` : "远程 API 返回异常状态。";
      break;
    case "OCR_START_FAILED":
      title = opts.title || "PaperView OCR 启动失败";
      message = `OCR 任务启动失败，请检查 OCR 依赖和输入文件。\n${logHint}`;
      break;
    case "OCR_PIPELINE_FAILED":
      title = opts.title || "PaperView OCR 失败";
      message = `OCR 处理失败，请检查输入 PDF 和日志。\n${logHint}`;
      break;
    case "QUERY_PIPELINE_FAILED":
      title = opts.title || "PaperView 查询失败";
      message = `查询任务执行失败，请检查远程 API 配置和服务日志。\n${logHint}`;
      break;
    case "INGEST_REQUIRED":
      title = opts.title || "PaperView 缺少索引数据";
      message = "未找到 items 快照，请先在插件中执行一次 ingest/查询后重试。";
      break;
    case "NO_MATCHING_ITEMS":
      title = opts.title || "PaperView 未找到匹配条目";
      message = "当前选择条目未写入索引，或 key 不匹配。请重新选择条目后重试。";
      break;
    case "SERVICE_HTTP_4XX":
    case "BAD_REQUEST":
      title = opts.title || "PaperView 请求参数错误";
      message = status ? `服务返回 HTTP ${status}。\n请检查当前输入配置后重试。` : message;
      break;
    case "SERVICE_HTTP_5XX":
    case "INTERNAL_ERROR":
      title = opts.title || "PaperView 服务内部错误";
      message = `本地服务返回内部错误。\n${logHint}`;
      break;
    case "RUNTIME_CHECK_FAILED":
      title = opts.title || "PaperView Runtime 检查失败";
      message = "Runtime 检查失败，请确认 Service Base URL 和 API 配置。";
      break;
    default:
      break;
  }

  if (detail.hint) {
    message += `\n${detail.hint}`;
  }
  return { title, message, code, detail };
}

function showPaperViewError(err, options) {
  const normalized = formatPaperViewError(err, options);
  const detailText = asErrorText(err);
  Zotero.debug(
    `[PaperView][error] code=${normalized.code} detail=${detailText}`
  );
  showPaperViewAlert(normalized.message, normalized.title);
}

async function runWithPaperViewErrorAlert(action, options) {
  try {
    return await action();
  } catch (err) {
    showPaperViewError(err, options);
    return null;
  }
}

function resolveApiKeyFromPrefsOrEnv() {
  const prefApiKey = String(getApiKey() || "").trim();
  if (prefApiKey) {
    return { apiKey: prefApiKey, source: "prefs" };
  }
  let envApiKey = "";
  try {
    envApiKey = String(
      (
        (Services.env && Services.env.get("SILICONFLOW_API_KEY")) ||
        (Services.env && Services.env.get("OPENAI_API_KEY")) ||
        ""
      )
    ).trim();
  } catch (err) {
    envApiKey = "";
  }
  if (envApiKey) {
    return { apiKey: envApiKey, source: "env" };
  }
  return { apiKey: "", source: "none" };
}

function syncRuntimeConfigAndEnv(options) {
  const opts = options || {};
  const requireApiKey = !!opts.requireApiKey;
  const cfg = normalizeLlmConfig(getCurrentLlmConfig());
  const resolved = resolveApiKeyFromPrefsOrEnv();
  if (resolved.apiKey) {
    cfg.api_key = resolved.apiKey;
  }
  Zotero.debug(`[PaperView] api_key_source=${resolved.source}`);
  writeLlmConfigFile(cfg);
  try {
    Services.env.set("PAPERVIEW_LLM_CONFIG", getLlmConfigPath());
  } catch (err) {
    Zotero.debug(`[PaperView] set config env failed: ${err}`);
  }
  if (cfg.api_key) {
    try {
      Services.env.set("OPENAI_API_KEY", cfg.api_key);
      Services.env.set("SILICONFLOW_API_KEY", cfg.api_key);
    } catch (err) {
      Zotero.debug(`[PaperView] set api key env failed: ${err}`);
    }
  }
  if (requireApiKey && !cfg.api_key) {
    throw createPaperViewError(
      "API_KEY_MISSING",
      "missing api key",
      {
        apiKeySource: resolved.source,
        hint: "如果你在 macOS 从 Dock/Finder 启动 Zotero，shell 环境变量通常不可见，建议直接在设置页填写 API Key。"
      }
    );
  }
  return cfg;
}

function hasUserPref(key) {
  try {
    return !!(Services && Services.prefs && Services.prefs.prefHasUserValue(key));
  } catch (err) {
    return false;
  }
}

function getPrefValue(key, fallback) {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.get === "function") {
      const value = Zotero.Prefs.get(key);
      if (value !== undefined && value !== null) return value;
    }
  } catch (err) {
    // ignore pref errors
  }
  return fallback;
}

function setPrefValue(key, value) {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.set === "function") {
      Zotero.Prefs.set(key, value);
    }
  } catch (err) {
    // ignore pref errors
  }
}

function getStringPref(key, fallback) {
  const value = getPrefValue(key, fallback);
  if (typeof value === "string") {
    const text = value.trim();
    return text ? text : fallback;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return fallback;
}

function getNumberPref(key, fallback) {
  const value = getPrefValue(key, fallback);
  if (typeof value === "string" && value.trim() === "") {
    return fallback;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getPositiveIntPref(key, fallback) {
  const n = Math.floor(getNumberPref(key, fallback));
  return n > 0 ? n : fallback;
}

function getBooleanPref(key, fallback) {
  const value = getPrefValue(key, fallback);
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(text)) return true;
    if (["0", "false", "no", "off"].includes(text)) return false;
  }
  return fallback;
}

function getLlmConfigPref() {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.get === "function") {
      const pref = Zotero.Prefs.get(PREF_LLM_CONFIG);
      if (pref && typeof pref === "string") {
        return JSON.parse(pref);
      }
    }
  } catch (err) {
    // ignore pref errors
  }
  return null;
}

function setLlmConfigPref(config) {
  try {
    if (Zotero.Prefs && typeof Zotero.Prefs.set === "function") {
      Zotero.Prefs.set(PREF_LLM_CONFIG, JSON.stringify(config));
    }
  } catch (err) {
    // ignore pref errors
  }
}

function normalizeLlmConfig(raw) {
  const normalizePositiveInt = (value, fallback) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(1, Math.floor(n));
  };
  const cfg = Object.assign({}, DEFAULT_LLM_CONFIG, raw || {});
  cfg.base_url = String(cfg.base_url || DEFAULT_LLM_CONFIG.base_url);
  cfg.model = String(cfg.model || DEFAULT_LLM_CONFIG.model);
  cfg.api_key = String(cfg.api_key || getApiKey() || "");
  cfg.temperature = Number.isFinite(Number(cfg.temperature))
    ? Number(cfg.temperature)
    : DEFAULT_LLM_CONFIG.temperature;
  cfg.max_output_tokens = normalizePositiveInt(
    cfg.max_output_tokens,
    DEFAULT_LLM_CONFIG.max_output_tokens
  );
  cfg.concurrency = normalizePositiveInt(
    cfg.concurrency,
    DEFAULT_LLM_CONFIG.concurrency
  );
  cfg.ocr_concurrency = normalizePositiveInt(
    cfg.ocr_concurrency,
    DEFAULT_LLM_CONFIG.ocr_concurrency
  );
  cfg.retry_on_429 = typeof cfg.retry_on_429 === "boolean"
    ? cfg.retry_on_429
    : DEFAULT_LLM_CONFIG.retry_on_429;
  cfg.retry_wait_s = Number.isFinite(Number(cfg.retry_wait_s))
    ? Math.max(0, Number(cfg.retry_wait_s))
    : DEFAULT_LLM_CONFIG.retry_wait_s;
  return cfg;
}

function loadLlmConfigFile() {
  const text = readTextFile(getLlmConfigPath());
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (err) {
    return null;
  }
}

function loadLegacyLlmConfig() {
  const prefCfg = getLlmConfigPref();
  if (prefCfg) return normalizeLlmConfig(prefCfg);
  const fileCfg = loadLlmConfigFile();
  if (fileCfg) return normalizeLlmConfig(fileCfg);
  return null;
}

function syncLlmPrefsFromConfig(config, options) {
  const cfg = normalizeLlmConfig(config);
  const onlyMissing = !!(options && options.onlyMissing);

  const maybeSet = (key, value) => {
    if (onlyMissing && hasUserPref(key)) return;
    setPrefValue(key, value);
  };

  maybeSet(PREF_LLM_BASE_URL, cfg.base_url);
  maybeSet(PREF_LLM_MODEL, cfg.model);
  maybeSet(PREF_LLM_TEMPERATURE, cfg.temperature);
  maybeSet(PREF_LLM_MAX_OUTPUT_TOKENS, cfg.max_output_tokens);
  maybeSet(PREF_LLM_CONCURRENCY, cfg.concurrency);
  maybeSet(PREF_LLM_OCR_CONCURRENCY, cfg.ocr_concurrency);
  maybeSet(PREF_LLM_RETRY_ON_429, !!cfg.retry_on_429);
  maybeSet(PREF_LLM_RETRY_WAIT_S, cfg.retry_wait_s);
}

function migrateLegacyLlmPrefs() {
  const legacy = loadLegacyLlmConfig();
  if (!legacy) return;
  syncLlmPrefsFromConfig(legacy, { onlyMissing: true });
}

function getCurrentLlmConfig() {
  const legacy = loadLegacyLlmConfig() || normalizeLlmConfig({});
  const cfg = normalizeLlmConfig({
    base_url: getStringPref(PREF_LLM_BASE_URL, legacy.base_url),
    model: getStringPref(PREF_LLM_MODEL, legacy.model),
    api_key: getApiKey() || legacy.api_key || "",
    temperature: getNumberPref(PREF_LLM_TEMPERATURE, legacy.temperature),
    max_output_tokens: getPositiveIntPref(
      PREF_LLM_MAX_OUTPUT_TOKENS,
      legacy.max_output_tokens
    ),
    concurrency: getPositiveIntPref(PREF_LLM_CONCURRENCY, legacy.concurrency),
    ocr_concurrency: getPositiveIntPref(
      PREF_LLM_OCR_CONCURRENCY,
      legacy.ocr_concurrency
    ),
    retry_on_429: getBooleanPref(PREF_LLM_RETRY_ON_429, legacy.retry_on_429),
    retry_wait_s: getNumberPref(PREF_LLM_RETRY_WAIT_S, legacy.retry_wait_s)
  });
  return cfg;
}

function writeLlmConfigFile(config) {
  const cfg = normalizeLlmConfig(config);
  ensureDir(getDataDirFile());
  writeTextFile(getLlmConfigPath(), JSON.stringify(cfg, null, 2), false);
  setLlmConfigPref(cfg);
  syncLlmPrefsFromConfig(cfg, { onlyMissing: false });
  if (cfg.api_key) {
    setApiKey(cfg.api_key);
  }
}

function ensureLlmConfigFile() {
  migrateLegacyLlmPrefs();
  writeLlmConfigFile(getCurrentLlmConfig());
}

function ensurePreferenceDefaults() {
  const defaults = normalizeLlmConfig({});
  const changed = [];

  const ensureString = (key, fallback) => {
    const current = getPrefValue(key, "");
    const text = typeof current === "string" ? current.trim() : String(current || "").trim();
    if (text) return;
    setPrefValue(key, String(fallback || ""));
    changed.push(key);
  };

  const ensureNumber = (key, fallback, validate) => {
    const n = Number(getPrefValue(key, NaN));
    if (Number.isFinite(n) && (!validate || validate(n))) return;
    setPrefValue(key, Number(fallback));
    changed.push(key);
  };

  const ensureBoolean = (key, fallback) => {
    const current = getPrefValue(key, null);
    if (typeof current === "boolean") return;
    if (typeof current === "number") return;
    if (typeof current === "string") {
      const text = current.trim().toLowerCase();
      if (["1", "0", "true", "false", "yes", "no", "on", "off"].includes(text)) {
        return;
      }
    }
    setPrefValue(key, !!fallback);
    changed.push(key);
  };

  // Service URL and runtime parameters must always have visible defaults.
  ensureString(PREF_SERVICE_URL, SERVICE_BASE_URL || "http://127.0.0.1:20341");
  ensureString(PREF_LLM_BASE_URL, defaults.base_url);
  ensureString(PREF_LLM_MODEL, defaults.model);
  ensureNumber(PREF_LLM_TEMPERATURE, defaults.temperature);
  ensureNumber(PREF_LLM_MAX_OUTPUT_TOKENS, defaults.max_output_tokens, (n) => n >= 1);
  ensureNumber(PREF_LLM_CONCURRENCY, defaults.concurrency, (n) => n >= 1);
  ensureNumber(PREF_LLM_OCR_CONCURRENCY, defaults.ocr_concurrency, (n) => n >= 1);
  ensureNumber(PREF_LLM_RETRY_WAIT_S, defaults.retry_wait_s, (n) => n >= 0);
  ensureBoolean(PREF_LLM_RETRY_ON_429, defaults.retry_on_429);

  // API key default intentionally allows empty; seed only when pref key does not exist.
  if (!hasUserPref(PREF_API_KEY)) {
    setPrefValue(PREF_API_KEY, "");
    changed.push(PREF_API_KEY);
  }

  // Keep the legacy JSON pref consistent with current normalized values.
  const normalized = getCurrentLlmConfig();
  setLlmConfigPref(normalized);
  if (!hasUserPref(PREF_LLM_CONFIG)) {
    changed.push(PREF_LLM_CONFIG);
  }

  if (changed.length) {
    Zotero.debug(`[PaperView] seeded default prefs: ${changed.length} (${changed.join(", ")})`);
  } else {
    Zotero.debug("[PaperView] default prefs already initialized");
  }
}

function getPrefsPaneDefaults() {
  return {
    serviceURL: "http://127.0.0.1:20341",
    apiKey: "",
    llmBaseURL: DEFAULT_LLM_CONFIG.base_url,
    llmModel: DEFAULT_LLM_CONFIG.model,
    temperature: DEFAULT_LLM_CONFIG.temperature,
    maxOutputTokens: DEFAULT_LLM_CONFIG.max_output_tokens,
    concurrency: DEFAULT_LLM_CONFIG.concurrency,
    ocrConcurrency: DEFAULT_LLM_CONFIG.ocr_concurrency,
    retryWaitS: DEFAULT_LLM_CONFIG.retry_wait_s,
    retryOn429: DEFAULT_LLM_CONFIG.retry_on_429
  };
}

function normalizePrefsPaneState(raw) {
  const defaults = getPrefsPaneDefaults();
  const state = Object.assign({}, defaults, raw || {});
  const parseNumber = (value, fallback) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  };
  const parseBoolean = (value, fallback) => {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
      const text = value.trim().toLowerCase();
      if (["1", "true", "yes", "on"].includes(text)) return true;
      if (["0", "false", "no", "off"].includes(text)) return false;
    }
    return fallback;
  };
  state.serviceURL = String(state.serviceURL || defaults.serviceURL).trim() || defaults.serviceURL;
  state.apiKey = String(state.apiKey || "").trim();
  state.llmBaseURL = String(state.llmBaseURL || defaults.llmBaseURL).trim() || defaults.llmBaseURL;
  state.llmModel = String(state.llmModel || defaults.llmModel).trim() || defaults.llmModel;
  state.temperature = parseNumber(state.temperature, defaults.temperature);
  state.maxOutputTokens = Math.max(
    1,
    Math.floor(parseNumber(state.maxOutputTokens, defaults.maxOutputTokens))
  );
  state.concurrency = Math.max(
    1,
    Math.floor(parseNumber(state.concurrency, defaults.concurrency))
  );
  state.ocrConcurrency = Math.max(1, Math.floor(Number(state.ocrConcurrency) || defaults.ocrConcurrency));
  state.retryWaitS = Math.max(
    0,
    Math.floor(parseNumber(state.retryWaitS, defaults.retryWaitS))
  );
  state.retryOn429 = parseBoolean(state.retryOn429, defaults.retryOn429);
  return state;
}

function getPrefsPaneState() {
  const defaults = getPrefsPaneDefaults();
  const cfg = getCurrentLlmConfig();
  return normalizePrefsPaneState({
    serviceURL: getStringPref(PREF_SERVICE_URL, defaults.serviceURL),
    apiKey: getApiKey(),
    llmBaseURL: getStringPref(PREF_LLM_BASE_URL, cfg.base_url),
    llmModel: getStringPref(PREF_LLM_MODEL, cfg.model),
    temperature: getNumberPref(PREF_LLM_TEMPERATURE, cfg.temperature),
    maxOutputTokens: getPositiveIntPref(PREF_LLM_MAX_OUTPUT_TOKENS, cfg.max_output_tokens),
    concurrency: getPositiveIntPref(PREF_LLM_CONCURRENCY, cfg.concurrency),
    ocrConcurrency: getPositiveIntPref(PREF_LLM_OCR_CONCURRENCY, cfg.ocr_concurrency),
    retryWaitS: getNumberPref(PREF_LLM_RETRY_WAIT_S, cfg.retry_wait_s),
    retryOn429: getBooleanPref(PREF_LLM_RETRY_ON_429, cfg.retry_on_429)
  });
}

function applyPrefsPaneState(doc, state) {
  const s = normalizePrefsPaneState(state);
  const serviceNode = doc.getElementById("paperview-service-url");
  const apiKeyNode = doc.getElementById("paperview-api-key");
  const baseURLNode = doc.getElementById("paperview-llm-base-url");
  const modelNode = doc.getElementById("paperview-llm-model");
  const temperatureNode = doc.getElementById("paperview-llm-temperature");
  const maxTokensNode = doc.getElementById("paperview-llm-max-output-tokens");
  const concurrencyNode = doc.getElementById("paperview-llm-concurrency");
  const ocrNode = doc.getElementById("paperview-llm-ocr-concurrency");
  const retryWaitNode = doc.getElementById("paperview-llm-retry-wait");
  const retry429Node = doc.getElementById("paperview-llm-retry-on-429");
  if (serviceNode) serviceNode.value = s.serviceURL;
  if (apiKeyNode) apiKeyNode.value = s.apiKey;
  if (baseURLNode) baseURLNode.value = s.llmBaseURL;
  if (modelNode) modelNode.value = s.llmModel;
  if (temperatureNode) temperatureNode.value = String(s.temperature);
  if (maxTokensNode) maxTokensNode.value = String(s.maxOutputTokens);
  if (concurrencyNode) concurrencyNode.value = String(s.concurrency);
  if (ocrNode) ocrNode.value = String(s.ocrConcurrency);
  if (retryWaitNode) retryWaitNode.value = String(s.retryWaitS);
  if (retry429Node) retry429Node.checked = !!s.retryOn429;
}

function readPrefsPaneRawState(doc) {
  const getValue = (id) => {
    const node = doc.getElementById(id);
    return node ? String(node.value || "") : "";
  };
  const getChecked = (id) => {
    const node = doc.getElementById(id);
    return !!(node && node.checked);
  };
  return {
    serviceURL: getValue("paperview-service-url"),
    apiKey: getValue("paperview-api-key"),
    llmBaseURL: getValue("paperview-llm-base-url"),
    llmModel: getValue("paperview-llm-model"),
    temperature: getValue("paperview-llm-temperature"),
    maxOutputTokens: getValue("paperview-llm-max-output-tokens"),
    concurrency: getValue("paperview-llm-concurrency"),
    ocrConcurrency: getValue("paperview-llm-ocr-concurrency"),
    retryWaitS: getValue("paperview-llm-retry-wait"),
    retryOn429: getChecked("paperview-llm-retry-on-429")
  };
}

function readPrefsPaneState(doc) {
  return normalizePrefsPaneState(readPrefsPaneRawState(doc));
}

function validatePrefsPaneRawState(raw) {
  const issues = [];
  const addIssue = (fieldId, message, level) => {
    issues.push({ fieldId, message, level: level || "error" });
  };

  const requireHttpUrl = (fieldId, label, value) => {
    const text = String(value || "").trim();
    if (!text) {
      addIssue(fieldId, `${label} is required`, "error");
      return;
    }
    try {
      const url = new URL(text);
      if (!/^https?:$/i.test(url.protocol)) {
        addIssue(fieldId, `${label} must use http/https`, "error");
      }
    } catch (err) {
      addIssue(fieldId, `${label} is not a valid URL`, "error");
    }
  };

  const requireNumberInRange = (fieldId, label, value, min, max, integerOnly) => {
    const text = String(value || "").trim();
    if (!text) {
      addIssue(fieldId, `${label} is required`, "error");
      return;
    }
    const n = Number(text);
    if (!Number.isFinite(n)) {
      addIssue(fieldId, `${label} must be a number`, "error");
      return;
    }
    if (integerOnly && !Number.isInteger(n)) {
      addIssue(fieldId, `${label} must be an integer`, "error");
      return;
    }
    if (n < min || n > max) {
      addIssue(fieldId, `${label} must be between ${min} and ${max}`, "error");
    }
  };

  requireHttpUrl("paperview-service-url", "Service Base URL", raw.serviceURL);
  requireHttpUrl("paperview-llm-base-url", "LLM Base URL", raw.llmBaseURL);
  if (!String(raw.llmModel || "").trim()) {
    addIssue("paperview-llm-model", "Model is required", "error");
  }
  requireNumberInRange("paperview-llm-temperature", "Temperature", raw.temperature, 0, 2, false);
  requireNumberInRange("paperview-llm-max-output-tokens", "Max Output Tokens", raw.maxOutputTokens, 1, 32768, true);
  requireNumberInRange("paperview-llm-concurrency", "Query Concurrency", raw.concurrency, 1, 64, true);
  requireNumberInRange("paperview-llm-ocr-concurrency", "OCR Concurrency", raw.ocrConcurrency, 1, 64, true);
  requireNumberInRange("paperview-llm-retry-wait", "Retry Wait (s)", raw.retryWaitS, 0, 7200, true);

  if (!String(raw.apiKey || "").trim()) {
    addIssue(
      "paperview-api-key",
      "API Key is empty. Runtime requests may fail unless another valid key source is available.",
      "warning"
    );
  }
  return issues;
}

function applyPrefsPaneValidationUI(doc, issues) {
  const errorFields = new Set(
    issues.filter((issue) => issue.level === "error").map((issue) => issue.fieldId)
  );
  const warningFields = new Set(
    issues.filter((issue) => issue.level === "warning").map((issue) => issue.fieldId)
  );
  for (const id of PREFS_FIELD_IDS) {
    const node = doc.getElementById(id);
    if (!node || !node.style) continue;
    if (errorFields.has(id)) {
      node.style.borderColor = "#dc2626";
    } else if (warningFields.has(id)) {
      node.style.borderColor = "#d97706";
    } else {
      node.style.borderColor = "#cbd5e1";
    }
  }
}

function collectPrefsPaneValidation(doc) {
  const raw = readPrefsPaneRawState(doc);
  const normalized = normalizePrefsPaneState(raw);
  const issues = validatePrefsPaneRawState(raw);
  applyPrefsPaneValidationUI(doc, issues);
  const errors = issues.filter((issue) => issue.level === "error");
  const warnings = issues.filter((issue) => issue.level === "warning");
  return { raw, normalized, issues, errors, warnings };
}

function setPrefsPaneStatus(doc, message, level) {
  const node = doc.getElementById("paperview-pref-status");
  if (!node) return;
  let mode = level;
  if (mode === true) mode = "error";
  if (mode === false || !mode) mode = "info";
  const palette = {
    info: "#475569",
    success: "#166534",
    warning: "#92400e",
    error: "#842029"
  };
  const text = String(message || "");
  if (node.localName === "label") {
    node.setAttribute("value", text);
  } else if (typeof node.textContent === "string" || node.textContent === "") {
    node.textContent = text;
  } else {
    node.setAttribute("value", text);
  }
  node.style.color = palette[mode] || palette.info;
  node.style.whiteSpace = "pre-wrap";
  node.style.lineHeight = "1.35";
}

function persistPrefsPaneState(state) {
  const s = normalizePrefsPaneState(state);
  setServiceBaseUrl(s.serviceURL);
  setApiKey(s.apiKey);
  setPrefValue(PREF_LLM_BASE_URL, s.llmBaseURL);
  setPrefValue(PREF_LLM_MODEL, s.llmModel);
  setPrefValue(PREF_LLM_TEMPERATURE, s.temperature);
  setPrefValue(PREF_LLM_MAX_OUTPUT_TOKENS, s.maxOutputTokens);
  setPrefValue(PREF_LLM_CONCURRENCY, s.concurrency);
  setPrefValue(PREF_LLM_OCR_CONCURRENCY, s.ocrConcurrency);
  setPrefValue(PREF_LLM_RETRY_WAIT_S, s.retryWaitS);
  setPrefValue(PREF_LLM_RETRY_ON_429, s.retryOn429);

  // Keep runtime config and file in sync with the explicit settings pane.
  const cfg = normalizeLlmConfig(
    Object.assign({}, getCurrentLlmConfig(), {
      base_url: s.llmBaseURL,
      model: s.llmModel,
      api_key: s.apiKey,
      temperature: s.temperature,
      max_output_tokens: s.maxOutputTokens,
      concurrency: s.concurrency,
      ocr_concurrency: s.ocrConcurrency,
      retry_wait_s: s.retryWaitS,
      retry_on_429: s.retryOn429
    })
  );
  writeLlmConfigFile(cfg);
}

function formatPrefsPaneErrorSummary(err) {
  if (isPaperViewError(err)) {
    const code = err.paperView.code;
    switch (code) {
      case "API_KEY_MISSING":
        return "API Key is missing. Fill API Key and click Save.";
      case "SERVICE_UNREACHABLE":
        return "Cannot reach local PaperView service.";
      case "SERVICE_TIMEOUT":
        return "Local PaperView service request timed out.";
      case "SERVICE_NOT_READY":
        return "Local PaperView service is not ready.";
      case "SERVICE_START_FAILED":
        return "Failed to start local PaperView service.";
      case "REMOTE_API_UNREACHABLE":
        return "Cannot reach remote API.";
      case "REMOTE_API_TIMEOUT":
        return "Remote API request timed out.";
      case "REMOTE_API_AUTH":
        return "Remote API authentication failed.";
      case "REMOTE_API_HTTP_ERROR":
        return "Remote API returned an HTTP error.";
      default:
        break;
    }
  }
  const text = asErrorText(err).trim();
  if (!text) return "Unexpected error.";
  return text.split("\n")[0];
}

function initPrefsPaneWindow(win) {
  try {
    const doc = win && win.document;
    if (!doc) return;
    const root = doc.getElementById("paperview-pref-root");
    if (!root) return;

    const current = getPrefsPaneState();
    applyPrefsPaneState(doc, current);
    persistPrefsPaneState(current);
    setPrefsPaneStatus(doc, "Defaults loaded. Click Save after changes.", "info");

    if (root.getAttribute("data-paperview-bound") === "true") {
      return;
    }

    const runPaneAction = async (buttonId, handler) => {
      const btn = buttonId ? doc.getElementById(buttonId) : null;
      const originalDisabled = btn ? !!btn.disabled : false;
      if (btn) btn.disabled = true;
      try {
        await handler();
      } catch (err) {
        setPrefsPaneStatus(doc, `Failed: ${formatPrefsPaneErrorSummary(err)}`, "error");
        Zotero.debug(`[PaperView] settings action failed: ${asErrorText(err)}`);
      } finally {
        if (btn) btn.disabled = originalDisabled;
      }
    };

    const refreshValidationHint = (preferInfo) => {
      const validation = collectPrefsPaneValidation(doc);
      if (validation.errors.length) {
        setPrefsPaneStatus(doc, `Validation failed: ${validation.errors[0].message}`, "error");
      } else if (validation.warnings.length) {
        setPrefsPaneStatus(doc, validation.warnings[0].message, "warning");
      } else if (preferInfo) {
        setPrefsPaneStatus(doc, "Configuration looks good.", "info");
      }
      return validation;
    };

    const onSave = async () => {
      const validation = refreshValidationHint(false);
      if (validation.errors.length) {
        return;
      }
      persistPrefsPaneState(validation.normalized);
      applyPrefsPaneState(doc, validation.normalized);
      const needsRestart = isServiceRunning();
      if (needsRestart) {
        setPrefsPaneStatus(
          doc,
          "Saved.\nService is running. Click Restart Service to apply changes.",
          validation.warnings.length ? "warning" : "success"
        );
      } else {
        setPrefsPaneStatus(doc, "Saved.", validation.warnings.length ? "warning" : "success");
      }
      Zotero.debug("[PaperView] preferences saved");
    };

    const onReset = () => {
      try {
        const defaults = getPrefsPaneDefaults();
        applyPrefsPaneState(doc, defaults);
        persistPrefsPaneState(defaults);
        refreshValidationHint(false);
        setPrefsPaneStatus(doc, "Defaults restored.", "success");
      } catch (err) {
        setPrefsPaneStatus(doc, `Reset failed: ${err}`, "error");
      }
    };

    const onCheckRuntime = async () => {
      const validation = refreshValidationHint(false);
      if (validation.errors.length) return;
      setPrefsPaneStatus(doc, "Checking runtime...", "info");
      const runtime = await runRuntimeCheck(validation.normalized, { checkRemote: false });
      const info = runtime && runtime.runtime ? runtime.runtime : {};
      const source = formatApiKeySource(info.api_key_source);
      const envVisible = info.env_visible ? "visible" : "not-visible";
      const profile = info.profile || "";
      const platform = info.platform || "unknown";
      let message = `Runtime check passed.\nAPI key source: ${source}\nEnvironment: ${envVisible}\nPlatform: ${platform}`;
      if (profile) {
        message += `\nProfile: ${profile}`;
      }
      setPrefsPaneStatus(doc, message, "success");
    };

    const onTestConnection = async () => {
      const validation = refreshValidationHint(false);
      if (validation.errors.length) return;
      setPrefsPaneStatus(doc, "Testing API connection...", "info");
      const result = await runRuntimeCheck(validation.normalized, { checkRemote: true });
      const runtime = result && result.runtime ? result.runtime : {};
      const remote = result && result.remote ? result.remote : {};
      const source = formatApiKeySource(runtime.api_key_source);
      const latency = Number(remote.latency_ms);
      const latencyText = Number.isFinite(latency) ? `${latency}ms` : "n/a";
      setPrefsPaneStatus(
        doc,
        `Connection test passed.\nSource: ${source}\nLatency: ${latencyText}\nModel: ${remote.model || validation.normalized.llmModel}`,
        "success"
      );
    };

    const onRestartService = async () => {
      setPrefsPaneStatus(doc, "Restarting service...", "info");
      stopService();
      await startService();
      await ensureServiceReady();
      setPrefsPaneStatus(doc, "Service restarted.", "success");
    };

    const saveBtn = doc.getElementById("paperview-pref-save");
    const resetBtn = doc.getElementById("paperview-pref-reset");
    const runtimeBtn = doc.getElementById("paperview-pref-check-runtime");
    const testBtn = doc.getElementById("paperview-pref-test-connection");
    const restartBtn = doc.getElementById("paperview-pref-restart-service");
    if (saveBtn) {
      saveBtn.addEventListener("command", () => {
        void runPaneAction("paperview-pref-save", onSave);
      });
    }
    if (resetBtn) resetBtn.addEventListener("command", onReset);
    if (runtimeBtn) {
      runtimeBtn.addEventListener("command", () => {
        void runPaneAction("paperview-pref-check-runtime", onCheckRuntime);
      });
    }
    if (testBtn) {
      testBtn.addEventListener("command", () => {
        void runPaneAction("paperview-pref-test-connection", onTestConnection);
      });
    }
    if (restartBtn) {
      restartBtn.addEventListener("command", () => {
        void runPaneAction("paperview-pref-restart-service", onRestartService);
      });
    }

    for (const id of PREFS_FIELD_IDS) {
      const input = doc.getElementById(id);
      if (!input || typeof input.addEventListener !== "function") continue;
      input.addEventListener("input", () => {
        refreshValidationHint(true);
      });
      input.addEventListener("change", () => {
        refreshValidationHint(true);
      });
    }

    refreshValidationHint(true);

    root.setAttribute("data-paperview-bound", "true");
    Zotero.debug("[PaperView] preferences pane initialized");
  } catch (err) {
    Zotero.debug(`[PaperView] init prefs pane failed: ${err}`);
  }
}

function registerPrefsPaneController() {
  try {
    if (!Zotero.PaperViewPrefsPane) {
      Zotero.PaperViewPrefsPane = {};
    }
    Zotero.PaperViewPrefsPane.init = initPrefsPaneWindow;
  } catch (err) {
    Zotero.debug(`[PaperView] register prefs pane controller failed: ${err}`);
  }
}

function unregisterPrefsPaneController() {
  try {
    if (Zotero.PaperViewPrefsPane && Zotero.PaperViewPrefsPane.init) {
      delete Zotero.PaperViewPrefsPane.init;
    }
  } catch (err) {
    // ignore
  }
}

function isWindows() {
  return Services && Services.appinfo && Services.appinfo.OS === "WINNT";
}

function getProfileDirFile() {
  return Services.dirsvc.get("ProfD", Ci.nsIFile);
}

function appendPath(baseFile, parts) {
  const file = baseFile.clone();
  for (const part of parts) {
    file.append(part);
  }
  return file;
}

function getProfileDirPath() {
  return getProfileDirFile().path;
}

function getDataDirFile() {
  return appendPath(getProfileDirFile(), ["paperview"]);
}

function getServiceDirFile() {
  return appendPath(getDataDirFile(), ["service"]);
}

function getEnvDirFile() {
  return appendPath(getDataDirFile(), ["venv"]);
}

function getLogDirFile() {
  return appendPath(getDataDirFile(), ["logs"]);
}

function getDataDirPath() {
  return getDataDirFile().path;
}

function getServiceDirPath() {
  return getServiceDirFile().path;
}

function getEnvDirPath() {
  return getEnvDirFile().path;
}

function getLogDirPath() {
  return getLogDirFile().path;
}

function getLogPath() {
  return appendPath(getLogDirFile(), ["service.log"]).path;
}

function getEnvLogPath() {
  return appendPath(getLogDirFile(), [ENV_LOG_FILENAME]).path;
}

function getPipLogPath() {
  return appendPath(getLogDirFile(), [PIP_LOG_FILENAME]).path;
}

function getLlmConfigPath() {
  return appendPath(getDataDirFile(), [LLM_CONFIG_FILENAME]).path;
}

function getServiceScriptPath() {
  return appendPath(getServiceDirFile(), ["local_service.py"]).path;
}

function getRequirementsPath() {
  return appendPath(getServiceDirFile(), ["requirements.txt"]).path;
}

function getVenvPythonPath() {
  const envDir = getEnvDirFile();
  const pyFile = envDir.clone();
  if (isWindows()) {
    pyFile.append("Scripts");
    pyFile.append("python.exe");
  } else {
    pyFile.append("bin");
    pyFile.append("python");
  }
  return pyFile.path;
}

function getServiceHostPort() {
  try {
    const url = new URL(getServiceBaseUrl());
    const port = Number(url.port) || 20341;
    const host = url.hostname || "127.0.0.1";
    return { host, port };
  } catch (err) {
    return { host: "127.0.0.1", port: 20341 };
  }
}

function createLocalFile(path) {
  const file = Cc["@mozilla.org/file/local;1"].createInstance(Ci.nsIFile);
  file.initWithPath(path);
  return file;
}

function ensureDir(file) {
  if (file.exists()) return;
  const parent = file.parent;
  if (parent && !parent.exists()) {
    ensureDir(parent);
  }
  file.create(Ci.nsIFile.DIRECTORY_TYPE, 0o755);
}

function ensureDirPath(path) {
  ensureDir(createLocalFile(path));
}

function fileExists(path) {
  try {
    return createLocalFile(path).exists();
  } catch (err) {
    return false;
  }
}

function writeTextFile(path, text, append) {
  const file = createLocalFile(path);
  if (file.parent) {
    ensureDir(file.parent);
  }
  const flags = 0x02 | 0x08 | (append ? 0x10 : 0x20);
  const stream = Cc["@mozilla.org/network/file-output-stream;1"].createInstance(
    Ci.nsIFileOutputStream
  );
  stream.init(file, flags, 0o644, 0);
  const converter = Cc["@mozilla.org/intl/converter-output-stream;1"].createInstance(
    Ci.nsIConverterOutputStream
  );
  converter.init(stream, "UTF-8");
  converter.writeString(text);
  converter.close();
}

function appendTextFile(path, text) {
  writeTextFile(path, text, true);
}

function readTextFile(path) {
  try {
    const file = createLocalFile(path);
    if (!file.exists()) return null;
    const stream = Cc["@mozilla.org/network/file-input-stream;1"].createInstance(
      Ci.nsIFileInputStream
    );
    stream.init(file, 0x01, 0o444, 0);
    const converter = Cc["@mozilla.org/intl/converter-input-stream;1"].createInstance(
      Ci.nsIConverterInputStream
    );
    converter.init(stream, "UTF-8");
    let data = "";
    const buffer = {};
    while (converter.readString(0xffffffff, buffer) !== 0) {
      data += buffer.value;
    }
    converter.close();
    return data;
  } catch (err) {
    return null;
  }
}

function runProcess(exePath, args) {
  return new Promise((resolve, reject) => {
    try {
      const file = createLocalFile(exePath);
      const proc = Cc["@mozilla.org/process/util;1"].createInstance(Ci.nsIProcess);
      proc.init(file);
      const observer = {
        observe() {
          resolve(proc.exitValue);
        }
      };
      proc.runAsync(args, args.length, observer, false);
    } catch (err) {
      reject(err);
    }
  });
}

async function runProcessChecked(exePath, args) {
  Zotero.debug(`[PaperView] exec: ${exePath} ${args.join(" ")}`);
  const code = await runProcess(exePath, args);
  if (code !== 0) {
    throw new Error(`Command failed (${code})`);
  }
}

async function findExecutable(candidates) {
  const pathVar = (Services.env && Services.env.get("PATH")) || "";
  const sep = isWindows() ? ";" : ":";
  const dirs = pathVar.split(sep).filter(Boolean);
  const suffix = isWindows() ? ".exe" : "";
  for (const dir of dirs) {
    for (const name of candidates) {
      const exe = name.endsWith(suffix) ? name : name + suffix;
      let full = null;
      try {
        const f = createLocalFile(dir);
        f.append(exe);
        full = f.path;
      } catch (err) {
        full = null;
      }
      if (full && fileExists(full)) return full;
    }
  }
  return null;
}

async function ensureServiceFiles() {
  if (!addonRootURI) throw new Error("addonRootURI not set");
  ensureDir(getDataDirFile());
  ensureDir(getServiceDirFile());
  for (const relPath of SERVICE_FILES) {
    const srcURL = `${addonRootURI}service/${relPath}`;
    const destPath = appendPath(getServiceDirFile(), [relPath]).path;
    const contents = Zotero.File.getContentsFromURL(srcURL);
    writeTextFile(destPath, contents, false);
  }
}

async function appendEnvLog(message) {
  try {
    ensureDir(getLogDirFile());
    const line = `[${new Date().toISOString()}] ${message}\n`;
    appendTextFile(getEnvLogPath(), line);
  } catch (err) {
    Zotero.debug(`[PaperView] append env log failed: ${err}`);
  }
}

async function ensureEnvReady() {
  if (serviceEnvPromise) return serviceEnvPromise;
  serviceEnvPromise = (async () => {
    await ensureServiceFiles();
    ensureDir(getLogDirFile());
    await appendEnvLog("env init start");
    const envPython = getVenvPythonPath();
    try {
      if (!fileExists(envPython)) {
        ensureDir(getDataDirFile());
        const candidates = isWindows()
          ? ["python", "py"]
          : ["python3", "python"];
        const pythonExe = await findExecutable(candidates);
        if (!pythonExe) {
          throw new Error("Python not found in PATH");
        }
        await appendEnvLog(`create venv using ${pythonExe}`);
        await runProcessChecked(pythonExe, ["-m", "venv", getEnvDirPath()]);
        await appendEnvLog(`pip upgrade (log: ${getPipLogPath()})`);
        await runProcessChecked(envPython, [
          "-m",
          "pip",
          "install",
          "--upgrade",
          "pip",
          "--log",
          getPipLogPath()
        ]);
        await appendEnvLog(`pip install requirements (log: ${getPipLogPath()})`);
        await runProcessChecked(envPython, [
          "-m",
          "pip",
          "install",
          "-r",
          getRequirementsPath(),
          "--log",
          getPipLogPath()
        ]);
      } else {
        await appendEnvLog("venv exists; skip install");
      }
      await appendEnvLog("env init done");
      return envPython;
    } catch (err) {
      await appendEnvLog(`env init error: ${err}`);
      throw err;
    }
  })();
  try {
    return await serviceEnvPromise;
  } catch (err) {
    // Allow a later retry after a failed init.
    serviceEnvPromise = null;
    throw err;
  }
}

async function startService() {
  if (serviceProcess && serviceProcess.isRunning) return true;
  if (serviceStarting) return false;
  serviceStarting = true;
  try {
    // Sync runtime config from Zotero preferences before launching service.
    syncRuntimeConfigAndEnv({ requireApiKey: false });
    const envPython = await ensureEnvReady();
    ensureDir(getDataDirFile());
    ensureDir(getLogDirFile());
    const { host, port } = getServiceHostPort();
    const args = [
      "-u",
      getServiceScriptPath(),
      "--host",
      host,
      "--port",
      String(port),
      "--log-file",
      getLogPath(),
      "--llm-config",
      getLlmConfigPath()
    ];
    const file = createLocalFile(envPython);
    const proc = Cc["@mozilla.org/process/util;1"].createInstance(Ci.nsIProcess);
    proc.init(file);
    const observer = {
      observe() {
        Zotero.debug("[PaperView] service exited");
        serviceProcess = null;
      }
    };
    proc.runAsync(args, args.length, observer, false);
    serviceProcess = proc;
    Zotero.debug(`[PaperView] service started at ${host}:${port}`);
    Zotero.debug(`[PaperView] service log: ${getLogPath()}`);
    return true;
  } catch (err) {
    Zotero.debug(`[PaperView] start service error: ${err}`);
    serviceProcess = null;
    throw createPaperViewError("SERVICE_START_FAILED", asErrorText(err), {
      logPath: getLogPath(),
      envLogPath: getEnvLogPath(),
      pipLogPath: getPipLogPath()
    });
  } finally {
    serviceStarting = false;
  }
}

function stopService() {
  if (serviceProcess && serviceProcess.isRunning) {
    try {
      serviceProcess.kill();
    } catch (err) {
      Zotero.debug(`[PaperView] stop service error: ${err}`);
    }
  }
  serviceProcess = null;
}

function isServiceRunning() {
  return !!(serviceProcess && serviceProcess.isRunning);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pingServiceHealth() {
  const baseUrl = getServiceBaseUrl();
  try {
    const resp = await Zotero.HTTP.request("GET", `${baseUrl}/health`, {
      headers: { "Content-Type": "application/json" },
      timeout: 2000
    });
    const data = parseJsonResponse(resp);
    return !!(data && data.ok);
  } catch (err) {
    return false;
  }
}

async function waitServiceReady(timeoutMs, intervalMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await pingServiceHealth()) {
      return true;
    }
    await delay(intervalMs);
  }
  return false;
}

async function ensureServiceReady() {
  if (await pingServiceHealth()) {
    return true;
  }
  if (serviceReadyPromise) {
    return serviceReadyPromise;
  }
  serviceReadyPromise = (async () => {
    Zotero.debug("[PaperView] service not ready, starting background service...");
    await startService();
    const ready = await waitServiceReady(90000, 800);
    if (!ready) {
      throw createPaperViewError("SERVICE_NOT_READY", "service not ready in time", {
        logPath: getLogPath(),
        baseUrl: getServiceBaseUrl()
      });
    }
    Zotero.debug("[PaperView] service is ready");
    return true;
  })();
  try {
    return await serviceReadyPromise;
  } finally {
    serviceReadyPromise = null;
  }
}

function getStoredPdfAttachment(item) {
  const attachmentIDs = item.getAttachments();
  for (const id of attachmentIDs) {
    const attachment = Zotero.Items.get(id);
    if (!attachment || !attachment.isAttachment()) continue;
    if (attachment.attachmentContentType !== "application/pdf") continue;
    if (
      attachment.attachmentLinkMode !==
      Zotero.Attachments.LINK_MODE_IMPORTED_FILE
    ) {
      continue;
    }
    let path = null;
    if (typeof attachment.getFilePath === "function") {
      path = attachment.getFilePath();
    }
    if (!path && typeof attachment.getFile === "function") {
      const file = attachment.getFile();
      if (file) path = file.path;
    }
    if (path) {
      return {
        pdf_path: path,
        attachment_key: attachment.key,
        attachment_item_id: attachment.id,
        attachment_link_mode: attachment.attachmentLinkMode
      };
    }
  }
  return null;
}

function extractYear(dateStr) {
  const m = (dateStr || "").match(/\b(\d{4})\b/);
  return m ? m[1] : null;
}

function buildItemPayload(item) {
  const creators = item.getCreators ? item.getCreators() : [];
  const date = item.getField ? item.getField("date") : "";
  const attachment = getStoredPdfAttachment(item);
  const publicationTitle = item.getField ? item.getField("publicationTitle") : "";
  const conferenceName = item.getField ? item.getField("conferenceName") : "";
  const proceedingsTitle = item.getField ? item.getField("proceedingsTitle") : "";
  const bookTitle = item.getField ? item.getField("bookTitle") : "";
  const series = item.getField ? item.getField("series") : "";
  const volume = item.getField ? item.getField("volume") : "";
  const issue = item.getField ? item.getField("issue") : "";
  const pages = item.getField ? item.getField("pages") : "";
  const publisher = item.getField ? item.getField("publisher") : "";
  const place = item.getField ? item.getField("place") : "";
  const abstractNote = item.getField ? item.getField("abstractNote") : "";
  const extra = item.getField ? item.getField("extra") : "";
  const language = item.getField ? item.getField("language") : "";
  const venue =
    conferenceName ||
    proceedingsTitle ||
    publicationTitle ||
    bookTitle ||
    series ||
    "";
  return {
    item_key: item.key,
    item_id: item.id,
    library_id: item.libraryID,
    item_type: item.itemType,
    title: item.getField ? item.getField("title") : "",
    creators,
    year: extractYear(date),
    date,
    doi: item.getField ? item.getField("DOI") : "",
    url: item.getField ? item.getField("url") : "",
    venue,
    publication_title: publicationTitle,
    conference_name: conferenceName,
    proceedings_title: proceedingsTitle,
    book_title: bookTitle,
    series,
    volume,
    issue,
    pages,
    publisher,
    place,
    abstract: abstractNote,
    extra,
    language,
    pdf_path: attachment ? attachment.pdf_path : null,
    pdf_missing: !attachment,
    attachment_key: attachment ? attachment.attachment_key : null,
    attachment_item_id: attachment ? attachment.attachment_item_id : null,
    attachment_link_mode: attachment ? attachment.attachment_link_mode : null
  };
}

async function promptQueryText() {
  try {
    const win = Zotero.getMainWindow();
    if (!win || !win.document) return null;
    const doc = win.document;
    const mount = doc.body || doc.documentElement;
    if (!mount) return null;

    const old = doc.getElementById("paperview-query-backdrop");
    if (old && old.parentNode) {
      old.parentNode.removeChild(old);
    }

    return await new Promise((resolve) => {
      let closed = false;
      const close = (value) => {
        if (closed) return;
        closed = true;
        if (backdrop && backdrop.parentNode) {
          backdrop.parentNode.removeChild(backdrop);
        }
        resolve(value && value.trim() ? value.trim() : null);
      };

      const backdrop = doc.createElement("div");
      backdrop.setAttribute("id", "paperview-query-backdrop");
      backdrop.style.position = "fixed";
      backdrop.style.inset = "0";
      backdrop.style.background = "rgba(15, 23, 42, 0.32)";
      backdrop.style.zIndex = "2147483647";
      backdrop.style.display = "flex";
      backdrop.style.alignItems = "center";
      backdrop.style.justifyContent = "center";

      const panel = doc.createElement("div");
      panel.style.width = "min(760px, calc(100vw - 40px))";
      panel.style.maxHeight = "calc(100vh - 80px)";
      panel.style.background = "#ffffff";
      panel.style.border = "1px solid #cbd5e1";
      panel.style.borderRadius = "12px";
      panel.style.boxShadow = "0 20px 44px rgba(15,23,42,.25)";
      panel.style.padding = "14px";
      panel.style.boxSizing = "border-box";
      panel.style.display = "flex";
      panel.style.flexDirection = "column";
      panel.style.gap = "10px";

      const title = doc.createElement("div");
      title.textContent = "请输入查询内容（支持多行）";
      title.style.fontSize = "14px";
      title.style.fontWeight = "700";
      title.style.color = "#0f172a";

      const hint = doc.createElement("div");
      hint.textContent = "提示：可直接换行；Ctrl/Cmd + Enter 提交，Esc 取消。";
      hint.style.fontSize = "12px";
      hint.style.color = "#475569";

      const textarea = doc.createElement("textarea");
      textarea.value = "";
      textarea.style.width = "100%";
      textarea.style.maxWidth = "100%";
      textarea.style.boxSizing = "border-box";
      textarea.style.minHeight = "220px";
      textarea.style.maxHeight = "55vh";
      textarea.style.resize = "vertical";
      textarea.style.padding = "10px";
      textarea.style.border = "1px solid #cbd5e1";
      textarea.style.borderRadius = "10px";
      textarea.style.fontSize = "13px";
      textarea.style.lineHeight = "1.45";
      textarea.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      textarea.style.background = "#f8fafc";
      textarea.style.color = "#0f172a";

      const btnRow = doc.createElement("div");
      btnRow.style.display = "flex";
      btnRow.style.justifyContent = "flex-end";
      btnRow.style.gap = "8px";

      const btnCancel = doc.createElement("button");
      btnCancel.textContent = "取消";
      btnCancel.style.padding = "7px 12px";
      btnCancel.style.border = "1px solid #cbd5e1";
      btnCancel.style.background = "#ffffff";
      btnCancel.style.borderRadius = "9px";
      btnCancel.style.cursor = "pointer";

      const btnOk = doc.createElement("button");
      btnOk.textContent = "查询";
      btnOk.style.padding = "7px 12px";
      btnOk.style.border = "1px solid #4f46e5";
      btnOk.style.background = "#4f46e5";
      btnOk.style.color = "#ffffff";
      btnOk.style.borderRadius = "9px";
      btnOk.style.cursor = "pointer";

      const onKeyDown = (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
          e.preventDefault();
          close(textarea.value);
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          close(null);
        }
      };

      btnCancel.addEventListener("click", () => close(null));
      btnOk.addEventListener("click", () => close(textarea.value));
      textarea.addEventListener("keydown", onKeyDown);
      backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) close(null);
      });

      btnRow.appendChild(btnCancel);
      btnRow.appendChild(btnOk);
      panel.appendChild(title);
      panel.appendChild(hint);
      panel.appendChild(textarea);
      panel.appendChild(btnRow);
      backdrop.appendChild(panel);
      mount.appendChild(backdrop);

      win.setTimeout(() => {
        try {
          textarea.focus();
          textarea.select();
        } catch (err) {
          // ignore
        }
      }, 0);
    });
  } catch (err) {
    Zotero.debug(`[PaperView] prompt error: ${err}`);
    return null;
  }
}

function parseJsonResponse(resp) {
  const text = resp.responseText || resp.response || "";
  if (!String(text || "").trim()) {
    return {};
  }
  const parsed = safeParseJson(text);
  if (parsed === null) {
    throw createPaperViewError("INVALID_JSON_RESPONSE", "invalid json response", {
      responseText: String(text || "").slice(0, 600)
    });
  }
  return parsed;
}

async function requestServiceJSON(method, path, payload, options) {
  const opts = options || {};
  const baseUrl = opts.baseUrl || getServiceBaseUrl();
  const url = `${baseUrl}${path}`;
  const reqOptions = {
    headers: { "Content-Type": "application/json" },
    timeout: Number(opts.timeoutMs) || DEFAULT_SERVICE_REQUEST_TIMEOUT_MS
  };
  if (payload !== undefined && payload !== null) {
    reqOptions.body = JSON.stringify(payload);
  }
  try {
    const resp = await Zotero.HTTP.request(method, url, reqOptions);
    return parseJsonResponse(resp);
  } catch (err) {
    throw classifyServiceRequestError(err, { operation: opts.operation || path });
  }
}

function formatApiKeySource(source) {
  const key = String(source || "").trim().toLowerCase();
  if (key === "prefs") return "prefs";
  if (key === "env") return "env";
  if (key === "request") return "request";
  return "none";
}

async function runRuntimeCheck(state, options) {
  const opts = options || {};
  await ensureServiceReady();
  const payload = {
    base_url: state.llmBaseURL,
    model: state.llmModel,
    api_key: state.apiKey,
    timeout_s: DEFAULT_RUNTIME_TEST_TIMEOUT_S,
    check_remote: !!opts.checkRemote
  };
  const data = await requestServiceJSON("POST", "/runtime/check", payload, {
    timeoutMs: Math.max(8000, DEFAULT_RUNTIME_TEST_TIMEOUT_S * 1000 + 4000),
    operation: opts.checkRemote ? "runtime_test_connection" : "runtime_check"
  });
  if (data && data.ok) {
    return data;
  }
  const serviceCode = data && data.code;
  const mappedCode = mapServiceErrorCodeToPluginCode(serviceCode) || "RUNTIME_CHECK_FAILED";
  throw createPaperViewError(mappedCode, (data && data.error) || "runtime check failed", {
    serviceCode,
    status: Number(data && data.status) || null,
    apiKeySource: data && data.runtime && data.runtime.api_key_source
  });
}

function summarizeIngestResult(ingest) {
  if (!ingest || typeof ingest !== "object") return "ingest=unknown";
  const ingested = Array.isArray(ingest.ingested_keys)
    ? ingest.ingested_keys.length
    : (Number.isFinite(Number(ingest.written)) ? Number(ingest.written) : 0);
  const items = Number.isFinite(Number(ingest.items_total))
    ? Number(ingest.items_total)
    : (Number.isFinite(Number(ingest.received)) ? Number(ingest.received) : null);
  const missingPdf = Number.isFinite(Number(ingest.missing_pdf))
    ? Number(ingest.missing_pdf)
    : null;
  return `ingested=${ingested}${items !== null ? ` items_total=${items}` : ""}${missingPdf !== null ? ` missing_pdf=${missingPdf}` : ""}`;
}

async function ingestItems(items) {
  const payload = {
    items: items.map(buildItemPayload),
    client: {
      plugin_id: pluginID,
      zotero_version: Zotero.version
    }
  };
  return requestServiceJSON("POST", "/ingest", payload, {
    timeoutMs: DEFAULT_SERVICE_REQUEST_TIMEOUT_MS,
    operation: "ingest"
  });
}

async function queryService(itemKeys, queryText, sectionsText, queryMode) {
  const payload = {
    item_keys: itemKeys,
    query: queryText,
    sections: sectionsText || "",
    query_mode: queryMode || "single"
  };
  const data = await requestServiceJSON("POST", "/query", payload, {
    timeoutMs: DEFAULT_SERVICE_REQUEST_TIMEOUT_MS,
    operation: "query_submit"
  });
  if (!data || !data.result_url) {
    throw createPaperViewError("BAD_RESPONSE", "missing result_url in /query response");
  }
  return data;
}

async function ocrService(itemKeys) {
  const cfg = getCurrentLlmConfig();
  const payload = {
    item_keys: itemKeys,
    ocr_concurrency: cfg.ocr_concurrency
  };
  const data = await requestServiceJSON("POST", "/ocr", payload, {
    timeoutMs: DEFAULT_SERVICE_REQUEST_TIMEOUT_MS,
    operation: "ocr_submit"
  });
  if (!data || !data.job_id) {
    throw createPaperViewError("OCR_START_FAILED", "missing job_id in /ocr response");
  }
  return data;
}

async function runQueryForSelectedItems(queryMode) {
  syncRuntimeConfigAndEnv({ requireApiKey: true });
  await ensureServiceReady();
  const pane = Zotero.getActiveZoteroPane();
  const items = pane && typeof pane.getSelectedItems === "function"
    ? pane.getSelectedItems()
    : [];
  if (!items || items.length === 0) return;

  const keys = items.map((item) => item.key);
  const modeText = queryMode === "merge" ? "concat" : "query";
  Zotero.debug(
    `[PaperView] Selected ${keys.length} item(s) for ${modeText}: ${keys.join(", ")}`
  );
  const rawQuery = await promptQueryText();
  if (!rawQuery) {
    Zotero.debug(`[PaperView] ${modeText} cancelled`);
    return;
  }

  const ingest = await ingestItems(items);
  Zotero.debug(`[PaperView] Ingested: ${summarizeIngestResult(ingest)}`);
  const result = await queryService(keys, rawQuery, "", queryMode);
  if (result && result.job_id && result.result_url) {
    await showQueryProgress(result.job_id, result.result_url);
  } else if (result && result.result_url) {
    Zotero.launchURL(result.result_url);
  }
}

// Shared job polling UI for query/ocr tasks.
async function showJobProgress(jobId, options) {
  const opts = options || {};
  const win = Zotero.getMainWindow();
  if (!win) return;

  const pw = new Zotero.ProgressWindow({ closeOnClick: false });
  pw.changeHeadline(opts.headline || "PaperView 处理中");
  const icon = getPaperViewIconURL();
  const progress = new pw.ItemProgress(icon, "准备中...");
  progress.setProgress(0);
  pw.show();

  let stopped = false;
  let timer = null;
  let pollErrorCount = 0;
  const stopPolling = () => {
    if (stopped) return;
    stopped = true;
    if (timer !== null) {
      win.clearInterval(timer);
      timer = null;
    }
  };

  const update = async () => {
    if (stopped) return;
    try {
      const data = await requestServiceJSON("GET", `/status/${jobId}`, null, {
        timeoutMs: DEFAULT_STATUS_POLL_TIMEOUT_MS,
        operation: "job_status_poll"
      });
      pollErrorCount = 0;
      const stage = data.stage || opts.defaultStage || "job";
      const done = Number(data.done || 0);
      const total = Number(data.total || 0);
      const rawMsg = typeof data.message === "string" ? data.message.trim() : "";
      const msg = rawMsg ? ` ${rawMsg}` : "";
      let percent = 0;
      if (total > 0) {
        percent = Math.min(100, Math.round((done * 100) / total));
      } else if (stage === "done") {
        percent = 100;
      }
      progress.setProgress(percent);
      progress.setText(`${stage} ${total > 0 ? `${done}/${total}` : ""}${msg}`.trim());

      if (stage === "done") {
        stopPolling();
        progress.setProgress(100);
        progress.setText("完成");
        pw.close();
        if (typeof opts.onDone === "function") {
          await opts.onDone();
        }
        return;
      }
      if (stage === "error") {
        stopPolling();
        progress.setProgress(100);
        const serviceCode = data.error_code || data.code || "";
        const mappedCode = mapServiceErrorCodeToPluginCode(serviceCode);
        const finalMsg = rawMsg || `未知错误，请检查日志：${getLogPath()}`;
        progress.setText(`失败 ${finalMsg}`.trim());
        pw.close();
        const error = createPaperViewError(
          mappedCode || (opts.defaultStage === "ocr" ? "OCR_PIPELINE_FAILED" : "QUERY_PIPELINE_FAILED"),
          finalMsg,
          { serviceCode, stage }
        );
        showPaperViewError(error, { title: opts.errorPrefix || "任务失败" });
      }
    } catch (err) {
      pollErrorCount += 1;
      if (pollErrorCount >= DEFAULT_STATUS_POLL_MAX_ERRORS) {
        stopPolling();
        pw.close();
        showPaperViewError(err, { title: opts.errorPrefix || "任务失败" });
        return;
      }
      progress.setText(`等待服务响应...(${pollErrorCount}/${DEFAULT_STATUS_POLL_MAX_ERRORS})`);
    }
  };

  timer = win.setInterval(() => {
    void update();
  }, 1000);
  void update();
}

async function showQueryProgress(jobId, resultUrl) {
  return showJobProgress(jobId, {
    headline: "PaperView 查询中",
    defaultStage: "query",
    errorPrefix: "查询失败",
    onDone: () => Zotero.launchURL(resultUrl)
  });
}

async function showOcrProgress(jobId) {
  return showJobProgress(jobId, {
    headline: "PaperView OCR 中",
    defaultStage: "ocr",
    errorPrefix: "OCR 失败"
  });
}

function attachMenuToWindow(win) {
  try {
    if (!win || !win.document) return;
    const doc = win.document;
    const menu = doc.getElementById("zotero-itemmenu");
    if (!menu) return;
    let menuitem = doc.getElementById("paperview-query-menuitem");
    if (!menuitem) {
      menuitem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      menuitem.setAttribute("id", "paperview-query-menuitem");
      menuitem.setAttribute("label", MENU_LABELS.query);
      applyMenuIcon(menuitem);

      const onCommand = async () => {
        await runWithPaperViewErrorAlert(
          () => runQueryForSelectedItems("single"),
          { title: "PaperView 查询失败" }
        );
      };

      const onPopupShowing = () => {
        const items = Zotero.getActiveZoteroPane().getSelectedItems();
        const hidden = !items || items.length === 0;
        menuitem.hidden = hidden;
        const concatNode = doc.getElementById("paperview-concat-query-menuitem");
        if (concatNode) concatNode.hidden = hidden;
      };

      menuitem.addEventListener("command", onCommand);
      menu.addEventListener("popupshowing", onPopupShowing);
      menu.appendChild(menuitem);

      cleanupHandlers.push(() => {
        menu.removeEventListener("popupshowing", onPopupShowing);
        menuitem.removeEventListener("command", onCommand);
        if (menuitem.parentNode) menuitem.parentNode.removeChild(menuitem);
      });
    }

    let concatItem = doc.getElementById("paperview-concat-query-menuitem");
    if (!concatItem) {
      concatItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      concatItem.setAttribute("id", "paperview-concat-query-menuitem");
      concatItem.setAttribute("label", MENU_LABELS.concat);
      applyMenuIcon(concatItem);

      const onConcatCommand = async () => {
        await runWithPaperViewErrorAlert(
          () => runQueryForSelectedItems("merge"),
          { title: "PaperView Concat 查询失败" }
        );
      };

      concatItem.addEventListener("command", onConcatCommand);
      menu.appendChild(concatItem);

      cleanupHandlers.push(() => {
        concatItem.removeEventListener("command", onConcatCommand);
        if (concatItem.parentNode) concatItem.parentNode.removeChild(concatItem);
      });
    }

    let ocrItem = doc.getElementById("paperview-ocr-menuitem");
    if (!ocrItem) {
      ocrItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      ocrItem.setAttribute("id", "paperview-ocr-menuitem");
      ocrItem.setAttribute("label", MENU_LABELS.ocr);
      applyMenuIcon(ocrItem);

      const onOcrCommand = async () => {
        await runWithPaperViewErrorAlert(async () => {
          await ensureServiceReady();
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] OCR selected ${keys.length} item(s): ${keys.join(", ")}`
          );
          if (!items || items.length === 0) return;
          const ingest = await ingestItems(items);
          Zotero.debug(`[PaperView] Ingested: ${summarizeIngestResult(ingest)}`);
          const result = await ocrService(keys);
          if (result && result.job_id) {
            await showOcrProgress(result.job_id);
          }
        }, { title: "PaperView OCR 失败" });
      };

      const onOcrPopupShowing = () => {
        const items = Zotero.getActiveZoteroPane().getSelectedItems();
        ocrItem.hidden = !items || items.length === 0;
      };

      ocrItem.addEventListener("command", onOcrCommand);
      menu.addEventListener("popupshowing", onOcrPopupShowing);
      menu.appendChild(ocrItem);

      cleanupHandlers.push(() => {
        menu.removeEventListener("popupshowing", onOcrPopupShowing);
        ocrItem.removeEventListener("command", onOcrCommand);
        if (ocrItem.parentNode) ocrItem.parentNode.removeChild(ocrItem);
      });
    }

    let historyItem = doc.getElementById("paperview-query-history-menuitem");
    if (!historyItem) {
      historyItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      historyItem.setAttribute("id", "paperview-query-history-menuitem");
      historyItem.setAttribute("label", MENU_LABELS.history);
      applyMenuIcon(historyItem);

      const onHistoryCommand = () => {
        void runWithPaperViewErrorAlert(async () => {
          await ensureServiceReady();
          const baseUrl = getServiceBaseUrl();
          Zotero.launchURL(`${baseUrl}/query_view.html`);
        }, { title: "PaperView 历史查询失败" });
      };

      historyItem.addEventListener("command", onHistoryCommand);
      menu.appendChild(historyItem);

      cleanupHandlers.push(() => {
        historyItem.removeEventListener("command", onHistoryCommand);
        if (historyItem.parentNode) historyItem.parentNode.removeChild(historyItem);
      });
    }
  } catch (err) {
    Zotero.debug(`[PaperView] attachMenuToWindow error: ${err}`);
  }
}

function attachToolsMenuToWindow(win) {
  try {
    if (!win || !win.document) return;
    const doc = win.document;
    const toolsMenu = doc.getElementById("menu_ToolsPopup");
    if (!toolsMenu) return;

    let startServiceItem = doc.getElementById("paperview-tools-start-service");
    if (!startServiceItem) {
      startServiceItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      startServiceItem.setAttribute("id", "paperview-tools-start-service");
      startServiceItem.setAttribute("label", MENU_LABELS.startService);
      applyMenuIcon(startServiceItem);
      const onStartService = async () => {
        await runWithPaperViewErrorAlert(() => startService(), {
          title: "PaperView 服务启动失败"
        });
      };
      startServiceItem.addEventListener("command", onStartService);
      toolsMenu.appendChild(startServiceItem);
      cleanupHandlers.push(() => {
        startServiceItem.removeEventListener("command", onStartService);
        if (startServiceItem.parentNode) {
          startServiceItem.parentNode.removeChild(startServiceItem);
        }
      });
    }

    let stopServiceItem = doc.getElementById("paperview-tools-stop-service");
    if (!stopServiceItem) {
      stopServiceItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      stopServiceItem.setAttribute("id", "paperview-tools-stop-service");
      stopServiceItem.setAttribute("label", MENU_LABELS.stopService);
      applyMenuIcon(stopServiceItem);
      const onStopService = () => {
        stopService();
      };
      stopServiceItem.addEventListener("command", onStopService);
      toolsMenu.appendChild(stopServiceItem);
      cleanupHandlers.push(() => {
        stopServiceItem.removeEventListener("command", onStopService);
        if (stopServiceItem.parentNode) {
          stopServiceItem.parentNode.removeChild(stopServiceItem);
        }
      });
    }

    if (!toolsMenu.getAttribute("data-paperview-bound")) {
      const onToolsPopupShowing = () => {
        const running = isServiceRunning();
        const startNode = doc.getElementById("paperview-tools-start-service");
        const stopNode = doc.getElementById("paperview-tools-stop-service");
        if (startNode) {
          startNode.disabled = running || serviceStarting;
        }
        if (stopNode) {
          stopNode.disabled = !running;
        }
      };
      toolsMenu.addEventListener("popupshowing", onToolsPopupShowing);
      toolsMenu.setAttribute("data-paperview-bound", "true");
      cleanupHandlers.push(() => {
        toolsMenu.removeEventListener("popupshowing", onToolsPopupShowing);
        toolsMenu.removeAttribute("data-paperview-bound");
      });
    }
  } catch (err) {
    Zotero.debug(`[PaperView] attachToolsMenuToWindow error: ${err}`);
  }
}

function initMenus() {
  try {
    const windows = Zotero.getMainWindows();
    for (const win of windows) {
      if (ENABLE_CONTEXT_MENUS) {
        attachMenuToWindow(win);
      }
      if (ENABLE_TOOLS_SERVICE_MENU) {
        attachToolsMenuToWindow(win);
      }
    }
  } catch (err) {
    Zotero.debug(`[PaperView] initMenus error: ${err}`);
  }
}

function ensureChrome(rootURI) {
  if (chromeHandle || !rootURI) return;
  try {
    const aomStartup = Cc["@mozilla.org/addons/addon-manager-startup;1"].getService(
      Ci.amIAddonManagerStartup
    );
    let manifestURI = null;
    if (Services && Services.io && typeof Services.io.newURI === "function") {
      manifestURI = Services.io.newURI(`${rootURI}manifest.json`);
    } else {
      const ioService = Cc["@mozilla.org/network/io-service;1"].getService(
        Ci.nsIIOService
      );
      manifestURI = ioService.newURI(`${rootURI}manifest.json`);
    }
    chromeHandle = aomStartup.registerChrome(manifestURI, [
      ["skin", "paperview-query", "classic/1.0", "skin/"],
      ["locale", "paperview-query", "en-US", "locale/en-US/"]
    ]);
  } catch (err) {
    Zotero.debug(`[PaperView] registerChrome failed: ${err}`);
  }
}

async function registerPreferencePane() {
  if (!addonRootURI || !pluginID) return;
  if (preferencePaneID) return;
  if (!Zotero.PreferencePanes || typeof Zotero.PreferencePanes.register !== "function") {
    Zotero.debug("[PaperView] PreferencePanes API unavailable");
    return;
  }
  try {
    preferencePaneID = await Zotero.PreferencePanes.register({
      pluginID,
      src: `${addonRootURI}${PREF_PANE_SRC}`,
      label: "PaperView",
      image: getPaperViewIconURL()
    });
    Zotero.debug(`[PaperView] preference pane registered: ${preferencePaneID}`);
  } catch (err) {
    Zotero.debug(`[PaperView] register preference pane failed: ${err}`);
  }
}

function unregisterPreferencePane() {
  if (!preferencePaneID) return;
  if (!Zotero.PreferencePanes || typeof Zotero.PreferencePanes.unregister !== "function") {
    preferencePaneID = null;
    return;
  }
  try {
    Zotero.PreferencePanes.unregister(preferencePaneID);
    Zotero.debug("[PaperView] preference pane unregistered");
  } catch (err) {
    Zotero.debug(`[PaperView] unregister preference pane failed: ${err}`);
  }
  preferencePaneID = null;
}

async function startup({ id, resourceURI, rootURI }) {
  pluginID = id;
  const resolvedRoot =
    rootURI || (resourceURI && resourceURI.spec) || null;
  addonRootURI = resolvedRoot;
  registerPrefsPaneController();
  try {
    ensurePreferenceDefaults();
  } catch (err) {
    Zotero.debug(`[PaperView] ensure default prefs failed: ${err}`);
  }
  // Keep startup path minimal to avoid chrome registration incompatibilities.
  await registerPreferencePane();
  Zotero.debug(`[PaperView] service_base_url=${getServiceBaseUrl()}`);
  try {
    ensureLlmConfigFile();
  } catch (err) {
    Zotero.debug(`[PaperView] llm config init failed: ${err}`);
  }
  if (ENABLE_CONTEXT_MENUS || ENABLE_TOOLS_SERVICE_MENU) {
    initMenus();
  }
  ensureEnvReady().catch((err) => {
    Zotero.debug(`[PaperView] env init failed: ${err}`);
  });
}

function shutdown() {
  unregisterPreferencePane();
  unregisterPrefsPaneController();
  stopService();
  for (const cleanup of cleanupHandlers) {
    try {
      cleanup();
    } catch (err) {
      // ignore cleanup errors
    }
  }
  cleanupHandlers = [];
  if (chromeHandle) {
    try {
      chromeHandle.destruct();
    } catch (err) {
      // ignore
    }
    chromeHandle = null;
  }
}

function install() {}
function uninstall() {}

function onMainWindowLoad({ window }) {
  try {
    ensurePreferenceDefaults();
  } catch (err) {
    Zotero.debug(`[PaperView] ensure default prefs (window) failed: ${err}`);
  }
  if (!preferencePaneID) {
    registerPreferencePane().catch((err) => {
      Zotero.debug(`[PaperView] preference pane late init failed: ${err}`);
    });
  }
  if (ENABLE_CONTEXT_MENUS) {
    attachMenuToWindow(window);
  }
  if (ENABLE_TOOLS_SERVICE_MENU) {
    attachToolsMenuToWindow(window);
  }
}

this.PaperViewPlugin = {
  startup,
  shutdown,
  install,
  uninstall,
  onMainWindowLoad
};
