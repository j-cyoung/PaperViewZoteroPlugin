## PaperView

这是一个简单的论文处理流水线项目：从论文列表解析元信息，检索摘要与下载 PDF，PDF 转 Markdown，抽取算力线索，并用 LLM 做翻译与结构化总结。

访问 https://j-cyoung.github.io/Embodied-AI-Paper-TopConf-Pages/ 查看可视化结果

### 数据来源声明
本项目的论文列表基于：  
https://github.com/Songwxuan/Embodied-AI-Paper-TopConf

### 项目定位
- 论文可视化 pipeline（用于快速整理与分析）
- 面向批量处理与增量更新（支持 resume）

### 核心脚本
- `parse_paperlist.py`：解析论文列表 Markdown，生成结构化 CSV/JSONL
- `enrich_papers.py`：补全元信息并下载 PDF
- `pdf_to_md_pymupdf4llm.py`：PDF 转 Markdown
- `extract_compute_from_pages.py`：从 Markdown 中提取算力相关线索
- `llm_enrich.py`：LLM 翻译摘要 + 结构化算力总结

