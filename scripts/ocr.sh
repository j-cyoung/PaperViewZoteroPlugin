#!/bin/bash

mkdir -p ./store/ocr

uv run python pdf_to_md_pymupdf4llm.py \
    --csv_in ./store/enrich/papers.enriched.csv \
    --page_chunks