#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List

from paper_utils import sha1_text, write_csv


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def pick_venue(item: Dict[str, Any]) -> str:
    for key in (
        "venue",
        "conference_name",
        "proceedings_title",
        "publication_title",
        "book_title",
        "series",
    ):
        val = (item.get(key) or "").strip()
        if val:
            return val
    return ""


def compute_source_hash(item_key: str, title: str, pdf_path: str) -> str:
    if pdf_path and os.path.exists(pdf_path):
        st = os.stat(pdf_path)
        base = f"{item_key}|{pdf_path}|{st.st_size}|{int(st.st_mtime)}"
    else:
        base = f"{item_key}|{title}"
    return sha1_text(base)


def normalize_row(item: Dict[str, Any]) -> Dict[str, Any]:
    item_key = item.get("item_key") or item.get("paper_id") or ""
    title = item.get("title") or ""
    pdf_path = item.get("pdf_path") or ""
    venue = pick_venue(item)
    source_hash = compute_source_hash(item_key, title, pdf_path)

    row = {
        "paper_id": item_key,
        "arxiv_id": item_key,
        "title": title,
        "abstract": item.get("abstract") or "",
        "venue": venue,
        "category": item.get("item_type") or "",
        "alias": item_key,
        "paper_url": item.get("url") or "",
        "page_url": item.get("url") or "",
        "doi": item.get("doi") or "",
        "year": item.get("year") or "",
        "date": item.get("date") or "",
        "pdf_path": pdf_path,
        "pdf_missing": item.get("pdf_missing"),
        "source_hash": source_hash,
        "raw_line": f"{item_key}|{title}",
        "zotero_item_key": item_key,
        "library_id": item.get("library_id"),
        "item_type": item.get("item_type"),
        "publication_title": item.get("publication_title") or "",
        "conference_name": item.get("conference_name") or "",
        "proceedings_title": item.get("proceedings_title") or "",
        "book_title": item.get("book_title") or "",
        "series": item.get("series") or "",
        "volume": item.get("volume") or "",
        "issue": item.get("issue") or "",
        "pages": item.get("pages") or "",
        "publisher": item.get("publisher") or "",
        "place": item.get("place") or "",
        "language": item.get("language") or "",
        "creators": item.get("creators") or [],
        "extra": item.get("extra") or "",
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="Convert Zotero items.jsonl to CSV for OCR")
    ap.add_argument("--jsonl", default="./store/zotero/items.jsonl")
    ap.add_argument("--csv_out", default="./store/zotero/items.csv")
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    if not rows:
        raise SystemExit(f"No rows found in {args.jsonl}")

    if args.dedupe:
        by_key: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            k = r.get("item_key") or r.get("paper_id")
            if not k:
                continue
            by_key[k] = r
        rows = list(by_key.values())

    out_rows = [normalize_row(r) for r in rows]
    write_csv(args.csv_out, out_rows, preferred_fields=[
        "paper_id",
        "arxiv_id",
        "title",
        "venue",
        "year",
        "date",
        "doi",
        "paper_url",
        "pdf_path",
        "pdf_missing",
        "source_hash",
    ])
    print(f"Wrote {len(out_rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
