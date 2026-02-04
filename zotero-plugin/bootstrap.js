/* global Zotero */

var menuRegistrationID = null;
var pluginID = null;
var FTL_FILE = "paperview.ftl";

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
        onCommand: (event, context) => {
          try {
            const items =
              context && context.items && context.items.length
                ? context.items
                : Zotero.getActiveZoteroPane().getSelectedItems();
            const keys = items.map((item) => item.key);
            Zotero.debug(
              `[PaperView] Selected ${keys.length} item(s): ${keys.join(", ")}`
            );
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
