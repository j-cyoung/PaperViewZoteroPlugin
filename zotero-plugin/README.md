# PaperView Zotero 插件（最小 Demo）

## 目录结构
- `manifest.json`
- `bootstrap.js`
- `locale/en-US/paperview.ftl`

## Demo 行为
- 在 Zotero 文献列表右键菜单增加 `Query`
- 点击后在 Zotero 调试输出中打印选中条目的 item key
- 同时调用本地服务 `/query`，并打开返回的 `result_url`

## 编译与安装（打包成 .xpi）
1. 在仓库根目录执行以下命令生成安装包：

```bash
cd zotero-plugin
zip -r ../paperview-query.xpi .
```

2. 在 Zotero 中打开插件管理器，然后把 `paperview-query.xpi` 拖入即可安装。

## 本地服务（最小 Demo）
在仓库根目录启动本地服务：

```bash
python local_service.py --port 23119
```

服务启动后，右键点击 `Query` 会打开本地演示页面。

## 开发模式（可选）
如果你希望直接从源码加载（不用每次打包）：
1. 在 Zotero 配置文件的 `extensions` 目录下新建文件，文件名为插件 ID：

```
paperview-query@local
```

2. 文件内容是一行插件目录的绝对路径，例如：

```
/Users/you/Documents/work/study/PaperView/zotero-plugin
```

3. 重新启动 Zotero（必要时清理缓存）。

## 日志查看
- 打开 Zotero 的调试输出窗口，查看 `[PaperView]` 前缀日志。
