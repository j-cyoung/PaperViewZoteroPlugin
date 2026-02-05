/* global Zotero */

var pluginID = null;
var SERVICE_BASE_URL = "http://127.0.0.1:20341";
var LAST_QUERY_SECTIONS = "full_text";
var PREF_SERVICE_URL = "extensions.paperview.service_base_url";
var cleanupHandlers = [];

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

function promptQueryText() {
  try {
    const win = Zotero.getMainWindow();
    const text = win.prompt("请输入查询内容，例：[method] 总结方法", "");
    return text && text.trim() ? text.trim() : null;
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
  const payload = {
    item_keys: itemKeys
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
  const icon = `chrome://zotero/skin/treesource-unfiled${Zotero.hiDPI ? "@2x" : ""}.png`;
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
  const icon = `chrome://zotero/skin/treesource-unfiled${Zotero.hiDPI ? "@2x" : ""}.png`;
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
      menuitem.setAttribute("label", "Query");

      const onCommand = async () => {
        try {
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] Selected ${keys.length} item(s): ${keys.join(", ")}`
          );
          const rawQuery = promptQueryText();
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
      concatItem.setAttribute("label", "Concat Query");

      const onConcatCommand = async () => {
        try {
          const items = Zotero.getActiveZoteroPane().getSelectedItems();
          const keys = items.map((item) => item.key);
          Zotero.debug(
            `[PaperView] Selected ${keys.length} item(s) for concat: ${keys.join(", ")}`
          );
          const rawQuery = promptQueryText();
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
      ocrItem.setAttribute("label", "OCR Cache");

      const onOcrCommand = async () => {
        try {
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
      historyItem.setAttribute("label", "Query History");

      const onHistoryCommand = () => {
        try {
          const baseUrl = getServiceBaseUrl();
          Zotero.launchURL(`${baseUrl}/query_view.html`);
        } catch (err) {
          Zotero.debug(`[PaperView] history command error: ${err}`);
        }
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
      settingsItem.setAttribute("label", "PaperView: Set Service URL");
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

function startup({ id }) {
  pluginID = id;
  Zotero.debug(`[PaperView] service_base_url=${getServiceBaseUrl()}`);
  initMenus();
}

function shutdown() {
  for (const cleanup of cleanupHandlers) {
    try {
      cleanup();
    } catch (err) {
      // ignore cleanup errors
    }
  }
  cleanupHandlers = [];
}

function install() {}
function uninstall() {}

function onMainWindowLoad({ window }) {
  attachMenuToWindow(window);
  attachToolsMenuToWindow(window);
}
