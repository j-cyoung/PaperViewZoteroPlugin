/* global Zotero */

var menuRegistrationID = null;
var pluginID = null;
var FTL_FILE = "paperview.ftl";
var SERVICE_BASE_URL = "http://127.0.0.1:20341";

async function queryService(itemKeys) {
  const payload = { item_keys: itemKeys };
  const resp = await Zotero.HTTP.request("POST", `${SERVICE_BASE_URL}/query`, {
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" }
  });
  const text = resp.responseText || resp.response || "";
  const data = JSON.parse(text);
  if (!data || !data.result_url) {
    throw new Error("Missing result_url in response");
  }
  Zotero.launchURL(data.result_url);
}

function insertFTL(win) {
  try {
    if (win && win.MozXULElement) {
      win.MozXULElement.insertFTLIfNeeded(FTL_FILE);
    }
  } catch (err) {
    Zotero.debug(`[PaperView] insertFTL error: ${err}`);
  }
}

function registerMenu(pluginID) {
  if (menuRegistrationID) return;
  menuRegistrationID = Zotero.MenuManager.registerMenu({
    menuID: "paperview-query",
    pluginID,
    target: "main/library/item",
    menus: [
      {
        menuType: "menuitem",
        l10nID: "paperview-menu-query",
        onCommand: async (event, context) => {
          try {
            const items =
              context && context.items && context.items.length
                ? context.items
                : Zotero.getActiveZoteroPane().getSelectedItems();
            const keys = items.map((item) => item.key);
            Zotero.debug(
              `[PaperView] Selected ${keys.length} item(s): ${keys.join(", ")}`
            );
            await queryService(keys);
          } catch (err) {
            Zotero.debug(`[PaperView] onCommand error: ${err}`);
          }
        }
      }
    ]
  });
}

function startup({ id }) {
  pluginID = id;
  registerMenu(id);
  try {
    const windows = Zotero.getMainWindows();
    for (const win of windows) {
      if (!win || !win.ZoteroPane) continue;
      insertFTL(win);
    }
  } catch (err) {
    Zotero.debug(`[PaperView] startup window scan error: ${err}`);
  }
}

function shutdown() {
  if (menuRegistrationID) {
    Zotero.MenuManager.unregisterMenu(menuRegistrationID);
    menuRegistrationID = null;
  }
}

function install() {}
function uninstall() {}

function onMainWindowLoad({ window }) {
  insertFTL(window);
  registerMenu(pluginID || "paperview-query@local");
}
