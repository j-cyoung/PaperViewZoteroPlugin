#!/bin/bash

mkdir -p ./store/compute
uv run python extract_compute_from_pages.py --pages_jsonl ./store/ocr/papers.pages.jsonl \
--csv_in ./store/enrich/papers.enriched.csv