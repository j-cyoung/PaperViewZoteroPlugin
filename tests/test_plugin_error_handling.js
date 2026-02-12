const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function createNsIFile(initialPath) {
  return {
    path: initialPath || "/tmp/profile",
    clone() {
      return createNsIFile(this.path);
    },
    append(part) {
      const suffix = String(part || "");
      if (!suffix) return;
      this.path = `${this.path}/${suffix}`;
    },
    get parent() {
      const idx = this.path.lastIndexOf("/");
      const parentPath = idx > 0 ? this.path.slice(0, idx) : "/";
      return createNsIFile(parentPath);
    },
    exists() {
      return true;
    },
    create() {}
  };
}

function loadPluginContext() {
  const pluginPath = path.resolve(__dirname, "..", "modules", "paperview-plugin.js");
  const code = fs.readFileSync(pluginPath, "utf8");
  const envStore = new Map();
  const prefStore = new Map();

  const context = {
    console,
    URL,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    ChromeUtils: {},
    Ci: {},
    Cc: {},
    Services: {
      prompt: { alert() {} },
      env: {
        get(key) {
          return envStore.get(String(key)) || "";
        },
        set(key, value) {
          envStore.set(String(key), String(value || ""));
        }
      },
      prefs: {
        prefHasUserValue(key) {
          return prefStore.has(String(key));
        }
      },
      appinfo: { OS: "Darwin" },
      dirsvc: {
        get() {
          return createNsIFile("/tmp/profile");
        }
      }
    },
    Zotero: {
      debug() {},
      version: "8.0.0",
      Prefs: {
        get(key) {
          return prefStore.get(String(key));
        },
        set(key, value) {
          prefStore.set(String(key), value);
        }
      },
      HTTP: {
        async request() {
          return { responseText: "{}" };
        }
      },
      File: {
        getContentsFromURL() {
          return "";
        }
      },
      getMainWindow() {
        return {
          alert() {},
          setInterval,
          clearInterval
        };
      },
      getMainWindows() {
        return [];
      },
      getActiveZoteroPane() {
        return { getSelectedItems: () => [] };
      },
      Items: { get() { return null; } },
      Attachments: { LINK_MODE_IMPORTED_FILE: 1 },
      PreferencePanes: {
        async register() {
          return "pref-pane-id";
        },
        unregister() {}
      },
      launchURL() {}
    }
  };

  vm.createContext(context);
  vm.runInContext(code, context, { filename: "paperview-plugin.js" });

  // Keep tests isolated from filesystem writes.
  context.writeLlmConfigFile = () => {};
  context.getLlmConfigPath = () => "/tmp/profile/paperview/llm_config.json";
  context.getCurrentLlmConfig = () => ({
    base_url: "https://api.siliconflow.cn/v1",
    model: "Qwen/Qwen2.5-72B-Instruct",
    api_key: "",
    temperature: 0,
    max_output_tokens: 2048,
    concurrency: 5,
    ocr_concurrency: 4,
    retry_on_429: false,
    retry_wait_s: 300
  });
  return context;
}

test("syncRuntimeConfigAndEnv throws API_KEY_MISSING when key is absent", () => {
  const ctx = loadPluginContext();
  ctx.getApiKey = () => "";
  const originalGet = ctx.Services.env.get;
  ctx.Services.env.get = (key) => {
    if (key === "SILICONFLOW_API_KEY" || key === "OPENAI_API_KEY") {
      return "";
    }
    return originalGet(key);
  };

  assert.throws(
    () => ctx.syncRuntimeConfigAndEnv({ requireApiKey: true }),
    (err) => {
      assert.equal(err.name, "PaperViewError");
      assert.equal(err.paperView.code, "API_KEY_MISSING");
      assert.equal(err.paperView.apiKeySource, "none");
      return true;
    }
  );
});

test("classifyServiceRequestError maps no-service errors to SERVICE_UNREACHABLE", () => {
  const ctx = loadPluginContext();
  const err = new Error("Connection refused: 127.0.0.1:20341");
  const wrapped = ctx.classifyServiceRequestError(err, { operation: "query_submit" });
  assert.equal(wrapped.paperView.code, "SERVICE_UNREACHABLE");
});

test("classifyServiceRequestError maps timeout errors to SERVICE_TIMEOUT", () => {
  const ctx = loadPluginContext();
  const err = new Error("Request timed out after 15000ms");
  const wrapped = ctx.classifyServiceRequestError(err, { operation: "query_submit" });
  assert.equal(wrapped.paperView.code, "SERVICE_TIMEOUT");
});

test("classifyServiceRequestError maps HTTP 4xx responses to SERVICE_HTTP_4XX", () => {
  const ctx = loadPluginContext();
  const err = {
    status: 404,
    responseText: JSON.stringify({ error: "not found" })
  };
  const wrapped = ctx.classifyServiceRequestError(err, { operation: "status_poll" });
  assert.equal(wrapped.paperView.code, "SERVICE_HTTP_4XX");
});

test("classifyServiceRequestError maps HTTP 5xx responses to SERVICE_HTTP_5XX", () => {
  const ctx = loadPluginContext();
  const err = {
    status: 503,
    responseText: JSON.stringify({ error: "service unavailable" })
  };
  const wrapped = ctx.classifyServiceRequestError(err, { operation: "status_poll" });
  assert.equal(wrapped.paperView.code, "SERVICE_HTTP_5XX");
});

test("classifyServiceRequestError prioritizes service error code mapping", () => {
  const ctx = loadPluginContext();
  const err = {
    status: 401,
    responseText: JSON.stringify({ code: "remote_api_auth", error: "auth failed" })
  };
  const wrapped = ctx.classifyServiceRequestError(err, { operation: "runtime_test_connection" });
  assert.equal(wrapped.paperView.code, "REMOTE_API_AUTH");
});

test("runRuntimeCheck maps runtime/check error codes to plugin errors", async () => {
  const ctx = loadPluginContext();
  ctx.ensureServiceReady = async () => true;
  ctx.requestServiceJSON = async () => ({
    ok: false,
    code: "remote_api_timeout",
    error: "upstream timeout",
    runtime: { api_key_source: "prefs" }
  });

  await assert.rejects(
    () =>
      ctx.runRuntimeCheck(
        {
          llmBaseURL: "https://api.siliconflow.cn/v1",
          llmModel: "Qwen/Qwen2.5-72B-Instruct",
          apiKey: "dummy"
        },
        { checkRemote: true }
      ),
    (err) => {
      assert.equal(err.name, "PaperViewError");
      assert.equal(err.paperView.code, "REMOTE_API_TIMEOUT");
      return true;
    }
  );
});
