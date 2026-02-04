#!/bin/bash
mkdir -p ./store/parse
uv run python parse_paperlist.py paper_list.md --resume --resume_from ./store/parse/papers.jsonl