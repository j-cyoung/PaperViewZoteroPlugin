#!/bin/bash
mkdir -p ./store/enrich
mkdir -p ./store/enrich/pdfs
uv run python enrich_papers.py ./store/parse/papers.csv --download_pdf --resume --resume_from ./store/enrich/papers.enriched.csv