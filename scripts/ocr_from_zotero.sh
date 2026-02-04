#!/bin/bash
set -euo pipefail

# 将 Zotero 导出的 items.jsonl 转为 OCR 需要的 CSV
uv run python zotero_items_to_csv.py \
  --jsonl ./store/zotero/items.jsonl \
  --csv_out ./store/zotero/items.csv

# 基于 CSV 进行 OCR，输出保持兼容的 papers.pages.jsonl
uv run python pdf_to_md_pymupdf4llm.py \
  --csv_in ./store/zotero/items.csv \
  --base_output_dir ./store/zotero/ocr \
  --out_jsonl papers.pages.jsonl \
  --resume \
  --resume_from ./store/zotero/ocr/papers.pages.jsonl
