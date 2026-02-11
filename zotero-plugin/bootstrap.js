/* global Zotero, Cc, Ci, Services, ChromeUtils */

var pluginID = null;
var SERVICE_BASE_URL = "http://127.0.0.1:20341";
var LAST_QUERY_SECTIONS = "full_text";
var PREF_SERVICE_URL = "extensions.paperview.service_base_url";
var PREF_API_KEY = "extensions.paperview.api_key";
var PREF_LLM_CONFIG = "extensions.paperview.llm_config";
var cleanupHandlers = [];
var chromeHandle = null;
var serviceProcess = null;
var serviceStarting = false;
var serviceEnvPromise = null;
var serviceReadyPromise = null;
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
  settings: "PaperView: Set Service URL",
  apiKey: "PaperView: Set API Key",
  llmConfig: "PaperView: LLM Settings",
  startService: "PaperView: Start Service",
  stopService: "PaperView: Stop Service"
};

const ENV_LOG_FILENAME = "env-install.log";
const PIP_LOG_FILENAME = "pip-install.log";
const LLM_CONFIG_FILENAME = "llm_config.json";

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
  cfg.max_output_tokens = Number.isFinite(Number(cfg.max_output_tokens))
    ? Number(cfg.max_output_tokens)
    : DEFAULT_LLM_CONFIG.max_output_tokens;
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
    ? Number(cfg.retry_wait_s)
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

function getCurrentLlmConfig() {
  const fileCfg = loadLlmConfigFile();
  if (fileCfg) return normalizeLlmConfig(fileCfg);
  const prefCfg = getLlmConfigPref();
  if (prefCfg) return normalizeLlmConfig(prefCfg);
  return normalizeLlmConfig({});
}

function writeLlmConfigFile(config) {
  const cfg = normalizeLlmConfig(config);
  ensureDir(getDataDirFile());
  writeTextFile(getLlmConfigPath(), JSON.stringify(cfg, null, 2), false);
  setLlmConfigPref(cfg);
  if (cfg.api_key) {
    setApiKey(cfg.api_key);
  }
}

function ensureLlmConfigFile() {
  const path = getLlmConfigPath();
  if (fileExists(path)) return;
  writeLlmConfigFile(getCurrentLlmConfig());
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
  return serviceEnvPromise;
}

async function startService() {
  if (serviceStarting) return;
  if (serviceProcess && serviceProcess.isRunning) return;
  serviceStarting = true;
  try {
    ensureLlmConfigFile();
    const llmConfig = getCurrentLlmConfig();
    const apiKey = llmConfig.api_key || getApiKey();
    if (apiKey) {
      try {
        Services.env.set("OPENAI_API_KEY", apiKey);
        Services.env.set("SILICONFLOW_API_KEY", apiKey);
      } catch (err) {
        Zotero.debug(`[PaperView] set api key env failed: ${err}`);
      }
    }
    try {
      Services.env.set("PAPERVIEW_LLM_CONFIG", getLlmConfigPath());
    } catch (err) {
      Zotero.debug(`[PaperView] set config env failed: ${err}`);
    }
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
  } catch (err) {
    Zotero.debug(`[PaperView] start service error: ${err}`);
    try {
      const win = Zotero.getMainWindow();
      if (win) {
        win.alert(
          "PaperView",
          `服务启动失败：${err}\n请检查日志：${getLogPath()}`
        );
      }
    } catch (err2) {
      // ignore alert failures
    }
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
    const text = resp.responseText || resp.response || "";
    const data = text ? JSON.parse(text) : {};
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
      throw new Error(`Service not ready in time. Check log: ${getLogPath()}`);
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

async function ingestItems(items) {
  const baseUrl = getServiceBaseUrl();
  const payload = {
    items: items.map(buildItemPayload),
    client: {
      plugin_id: pluginID,
      zotero_version: Zotero.version
    }
  };
  const resp = await Zotero.HTTP.request("POST", `${baseUrl}/ingest`, {
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" }
  });
  const text = resp.responseText || resp.response || "";
  return JSON.parse(text);
}

async function queryService(itemKeys, queryText, sectionsText, queryMode) {
  const baseUrl = getServiceBaseUrl();
  const payload = {
    item_keys: itemKeys,
    query: queryText,
    sections: sectionsText || "",
    query_mode: queryMode || "single"
  };
  const resp = await Zotero.HTTP.request("POST", `${baseUrl}/query`, {
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" }
  });
  const text = resp.responseText || resp.response || "";
  const data = JSON.parse(text);
  if (!data || !data.result_url) {
    throw new Error("Missing result_url in response");
  }
  return data;
}

async function ocrService(itemKeys) {
  const baseUrl = getServiceBaseUrl();
  const cfg = getCurrentLlmConfig();
  const payload = {
    item_keys: itemKeys,
    ocr_concurrency: cfg.ocr_concurrency
  };
  const resp = await Zotero.HTTP.request("POST", `${baseUrl}/ocr`, {
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" }
  });
  const text = resp.responseText || resp.response || "";
  return JSON.parse(text);
}

async function showQueryProgress(jobId, resultUrl) {
  const baseUrl = getServiceBaseUrl();
  const win = Zotero.getMainWindow();
  const pw = new Zotero.ProgressWindow({ closeOnClick: false });
  pw.changeHeadline("PaperView 查询中");
  const icon = getPaperViewIconURL();
  const progress = new pw.ItemProgress(icon, "准备中...");
  progress.setProgress(0);
  pw.show();

  let stopped = false;
  const update = async () => {
    if (stopped) return;
    try {
      const resp = await Zotero.HTTP.request("GET", `${baseUrl}/status/${jobId}`, {
        headers: { "Content-Type": "application/json" }
      });
      const text = resp.responseText || resp.response || "";
      const data = JSON.parse(text);
      const stage = data.stage || "query";
      const done = Number(data.done || 0);
      const total = Number(data.total || 0);
      const msg = data.message ? ` ${data.message}` : "";
      let percent = 0;
      if (total > 0) {
        percent = Math.min(100, Math.round((done * 100) / total));
      } else if (stage === "done") {
        percent = 100;
      }
      progress.setProgress(percent);
      progress.setText(`${stage} ${total > 0 ? `${done}/${total}` : ""}${msg}`.trim());
      if (stage === "done") {
        progress.setProgress(100);
        progress.setText("完成");
        stopped = true;
        win.clearInterval(timer);
        pw.close();
        Zotero.launchURL(resultUrl);
      }
      if (stage === "error") {
        progress.setProgress(100);
        progress.setText(`失败${msg}`);
        stopped = true;
        win.clearInterval(timer);
        try {
          pw.close();
          win.alert("PaperView", `查询失败${msg}`.trim());
        } catch (err) {
          // ignore
        }
      }
    } catch (err) {
      progress.setText("等待服务响应...");
    }
  };

  const timer = win.setInterval(update, 1000);
  update();
}

async function showOcrProgress(jobId) {
  const baseUrl = getServiceBaseUrl();
  const win = Zotero.getMainWindow();
  const pw = new Zotero.ProgressWindow({ closeOnClick: false });
  pw.changeHeadline("PaperView OCR 中");
  const icon = getPaperViewIconURL();
  const progress = new pw.ItemProgress(icon, "准备中...");
  progress.setProgress(0);
  pw.show();

  let stopped = false;
  const update = async () => {
    if (stopped) return;
    try {
      const resp = await Zotero.HTTP.request("GET", `${baseUrl}/status/${jobId}`, {
        headers: { "Content-Type": "application/json" }
      });
      const text = resp.responseText || resp.response || "";
      const data = JSON.parse(text);
      const stage = data.stage || "ocr";
      const done = Number(data.done || 0);
      const total = Number(data.total || 0);
      const msg = data.message ? ` ${data.message}` : "";
      let percent = 0;
      if (total > 0) {
        percent = Math.min(100, Math.round((done * 100) / total));
      } else if (stage === "done") {
        percent = 100;
      }
      progress.setProgress(percent);
      progress.setText(`${stage} ${total > 0 ? `${done}/${total}` : ""}${msg}`.trim());
      if (stage === "done") {
        progress.setProgress(100);
        progress.setText("完成");
        stopped = true;
        win.clearInterval(timer);
        pw.close();
      }
      if (stage === "error") {
        progress.setProgress(100);
        progress.setText(`失败${msg}`);
        stopped = true;
        win.clearInterval(timer);
        try {
          pw.close();
          win.alert("PaperView", `OCR 失败${msg}`.trim());
        } catch (err) {
          // ignore
        }
      }
    } catch (err) {
      progress.setText("等待服务响应...");
    }
  };

  const timer = win.setInterval(update, 1000);
  update();
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
        try {
          await ensureServiceReady();
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] Selected ${keys.length} item(s): ${keys.join(", ")}`
          );
          const rawQuery = await promptQueryText();
          if (!rawQuery) {
            Zotero.debug("[PaperView] Query cancelled");
            return;
          }
          const queryText = rawQuery;
          const sectionsText = "";
          const ingest = await ingestItems(items);
          Zotero.debug(`[PaperView] Ingested: ${JSON.stringify(ingest)}`);
          const result = await queryService(keys, queryText, sectionsText, "single");
          if (result && result.job_id && result.result_url) {
            await showQueryProgress(result.job_id, result.result_url);
          } else if (result && result.result_url) {
            Zotero.launchURL(result.result_url);
          }
        } catch (err) {
          Zotero.debug(`[PaperView] onCommand error: ${err}`);
        }
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
        try {
          await ensureServiceReady();
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] Selected ${keys.length} item(s) for concat: ${keys.join(", ")}`
          );
          const rawQuery = await promptQueryText();
          if (!rawQuery) {
            Zotero.debug("[PaperView] Concat query cancelled");
            return;
          }
          const queryText = rawQuery;
          const sectionsText = "";
          const ingest = await ingestItems(items);
          Zotero.debug(`[PaperView] Ingested: ${JSON.stringify(ingest)}`);
          const result = await queryService(keys, queryText, sectionsText, "merge");
          if (result && result.job_id && result.result_url) {
            await showQueryProgress(result.job_id, result.result_url);
          } else if (result && result.result_url) {
            Zotero.launchURL(result.result_url);
          }
        } catch (err) {
          Zotero.debug(`[PaperView] onConcatCommand error: ${err}`);
        }
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
        try {
          await ensureServiceReady();
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] OCR selected ${keys.length} item(s): ${keys.join(", ")}`
          );
          if (!items || items.length === 0) return;
          const ingest = await ingestItems(items);
          Zotero.debug(`[PaperView] Ingested: ${JSON.stringify(ingest)}`);
          const result = await ocrService(keys);
          if (result && result.job_id) {
            await showOcrProgress(result.job_id);
          }
        } catch (err) {
          Zotero.debug(`[PaperView] OCR command error: ${err}`);
        }
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
        ensureServiceReady()
          .then(() => {
          const baseUrl = getServiceBaseUrl();
          Zotero.launchURL(`${baseUrl}/query_view.html`);
          })
          .catch((err) => {
            Zotero.debug(`[PaperView] history command error: ${err}`);
          });
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

    let settingsItem = doc.getElementById("paperview-tools-settings");
    if (!settingsItem) {
      settingsItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      settingsItem.setAttribute("id", "paperview-tools-settings");
      settingsItem.setAttribute("label", MENU_LABELS.settings);
      applyMenuIcon(settingsItem);
      const onSettings = () => {
        try {
          const current = getServiceBaseUrl();
          const input = win.prompt("请输入服务地址（如 http://127.0.0.1:20341）", current);
          if (!input) return;
          setServiceBaseUrl(input.trim());
          Zotero.debug(`[PaperView] service_base_url set to ${getServiceBaseUrl()}`);
        } catch (err) {
          Zotero.debug(`[PaperView] settings error: ${err}`);
        }
      };
      settingsItem.addEventListener("command", onSettings);
      toolsMenu.appendChild(settingsItem);
      cleanupHandlers.push(() => {
        settingsItem.removeEventListener("command", onSettings);
        if (settingsItem.parentNode) settingsItem.parentNode.removeChild(settingsItem);
      });
    }

    let apiKeyItem = doc.getElementById("paperview-tools-api-key");
    if (!apiKeyItem) {
      apiKeyItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      apiKeyItem.setAttribute("id", "paperview-tools-api-key");
      apiKeyItem.setAttribute("label", MENU_LABELS.apiKey);
      applyMenuIcon(apiKeyItem);
      const onApiKey = () => {
        try {
          const current = getCurrentLlmConfig().api_key || "";
          const input = win.prompt("请输入 API Key（留空则清除）", current);
          if (input === null) return;
          const next = getCurrentLlmConfig();
          next.api_key = input.trim();
          writeLlmConfigFile(next);
          Zotero.debug("[PaperView] api key updated");
        } catch (err) {
          Zotero.debug(`[PaperView] api key error: ${err}`);
        }
      };
      apiKeyItem.addEventListener("command", onApiKey);
      toolsMenu.appendChild(apiKeyItem);
      cleanupHandlers.push(() => {
        apiKeyItem.removeEventListener("command", onApiKey);
        if (apiKeyItem.parentNode) apiKeyItem.parentNode.removeChild(apiKeyItem);
      });
    }

    let llmConfigItem = doc.getElementById("paperview-tools-llm-config");
    if (!llmConfigItem) {
      llmConfigItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      llmConfigItem.setAttribute("id", "paperview-tools-llm-config");
      llmConfigItem.setAttribute("label", MENU_LABELS.llmConfig);
      applyMenuIcon(llmConfigItem);
      const onLlmConfig = () => {
        try {
          const current = getCurrentLlmConfig();
          const ask = (label, value) => {
            const input = win.prompt(`${label}（留空保持不变）`, String(value));
            if (input === null) return null;
            const trimmed = input.trim();
            return trimmed === "" ? value : trimmed;
          };
          const baseUrl = ask("API Base URL", current.base_url);
          if (baseUrl === null) return;
          const model = ask("Model", current.model);
          if (model === null) return;
          const temperatureRaw = ask("Temperature", current.temperature);
          if (temperatureRaw === null) return;
          const maxTokensRaw = ask("Max Output Tokens", current.max_output_tokens);
          if (maxTokensRaw === null) return;
          const concurrencyRaw = ask("Concurrency", current.concurrency);
          if (concurrencyRaw === null) return;
          const ocrConcurrencyRaw = ask("OCR Concurrency", current.ocr_concurrency);
          if (ocrConcurrencyRaw === null) return;
          const retryOn429Raw = ask("Retry On 429 (true/false)", current.retry_on_429);
          if (retryOn429Raw === null) return;
          const retryWaitRaw = ask("Retry Wait Seconds", current.retry_wait_s);
          if (retryWaitRaw === null) return;

          const next = Object.assign({}, current, {
            base_url: String(baseUrl),
            model: String(model),
            temperature: Number(temperatureRaw),
            max_output_tokens: Number(maxTokensRaw),
            concurrency: Number(concurrencyRaw),
            ocr_concurrency: Number(ocrConcurrencyRaw),
            retry_on_429:
              String(retryOn429Raw).toLowerCase() === "true" ||
              String(retryOn429Raw).toLowerCase() === "1",
            retry_wait_s: Number(retryWaitRaw)
          });
          writeLlmConfigFile(next);
          Zotero.debug("[PaperView] llm config updated");
        } catch (err) {
          Zotero.debug(`[PaperView] llm config error: ${err}`);
        }
      };
      llmConfigItem.addEventListener("command", onLlmConfig);
      toolsMenu.appendChild(llmConfigItem);
      cleanupHandlers.push(() => {
        llmConfigItem.removeEventListener("command", onLlmConfig);
        if (llmConfigItem.parentNode) llmConfigItem.parentNode.removeChild(llmConfigItem);
      });
    }

    let startServiceItem = doc.getElementById("paperview-tools-start-service");
    if (!startServiceItem) {
      startServiceItem = doc.createXULElement
        ? doc.createXULElement("menuitem")
        : doc.createElement("menuitem");
      startServiceItem.setAttribute("id", "paperview-tools-start-service");
      startServiceItem.setAttribute("label", MENU_LABELS.startService);
      applyMenuIcon(startServiceItem);
      const onStartService = () => {
        startService();
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
      attachMenuToWindow(win);
      attachToolsMenuToWindow(win);
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
      manifestURI = Services.io.newURI(`${rootURI}chrome.manifest`);
    } else {
      const ioService = Cc["@mozilla.org/network/io-service;1"].getService(
        Ci.nsIIOService
      );
      manifestURI = ioService.newURI(`${rootURI}chrome.manifest`);
    }
    chromeHandle = aomStartup.registerChrome(manifestURI, [
      ["skin", "paperview-query", "classic/1.0", "skin/"],
      ["locale", "paperview-query", "en-US", "locale/en-US/"]
    ]);
  } catch (err) {
    Zotero.debug(`[PaperView] registerChrome failed: ${err}`);
  }
}

function startup({ id, resourceURI, rootURI }) {
  pluginID = id;
  const resolvedRoot =
    rootURI || (resourceURI && resourceURI.spec) || null;
  addonRootURI = resolvedRoot;
  ensureChrome(resolvedRoot);
  Zotero.debug(`[PaperView] service_base_url=${getServiceBaseUrl()}`);
  try {
    ensureLlmConfigFile();
  } catch (err) {
    Zotero.debug(`[PaperView] llm config init failed: ${err}`);
  }
  initMenus();
  ensureEnvReady().catch((err) => {
    Zotero.debug(`[PaperView] env init failed: ${err}`);
  });
}

function shutdown() {
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
  attachMenuToWindow(window);
  attachToolsMenuToWindow(window);
}
