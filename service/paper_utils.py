#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.() ]+", "_", (s or "").strip())
    if len(s) > max_len:
        s = s[:max_len]
    return s or "paper"


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_id_text(s: str) -> str:
    return normalize_space(s).lower()


def compute_paper_id(
    venue: Optional[str],
    category: Optional[str],
    alias: Optional[str],
    title: Optional[str],
    paper_url: Optional[str],
    page_url: Optional[str],
) -> str:
    parts = [
        normalize_id_text(venue or ""),
        normalize_id_text(category or ""),
        normalize_id_text(alias or ""),
        normalize_id_text(title or ""),
        normalize_id_text(paper_url or ""),
        normalize_id_text(page_url or ""),
    ]
    text = "|".join(parts).strip("|")
    return sha1_text(text or "empty")


def compute_source_hash(venue: Optional[str], category: Optional[str], raw_line: Optional[str]) -> str:
    parts = [
        normalize_space(venue or ""),
        normalize_space(category or ""),
        normalize_space(raw_line or ""),
    ]
    return sha1_text("|".join(parts))


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stringify_value(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def write_csv(path: str, rows: List[Dict[str, Any]], preferred_fields: Optional[List[str]] = None) -> None:
    if not rows:
        return

    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    fields: List[str] = []
    if preferred_fields:
        for k in preferred_fields:
            if k in field_set:
                fields.append(k)
                field_set.remove(k)
    fields += sorted(field_set)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: _stringify_value(v) for k, v in r.items()}
            w.writerow(out)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
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


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def expected_pdf_name(arxiv_id: str, title: str) -> str:
    base = safe_filename(f"{arxiv_id}_{title}".strip("_"))
    return base + ".pdf"


def list_pdfs(pdf_dir: str) -> List[str]:
    if not os.path.isdir(pdf_dir):
        return []
    out = []
    for fn in os.listdir(pdf_dir):
        if fn.lower().endswith(".pdf"):
            out.append(os.path.join(pdf_dir, fn))
    return out


def build_pdf_indices(pdf_paths: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    by_arxiv: Dict[str, List[str]] = {}
    by_token: Dict[str, List[str]] = {}

    for p in pdf_paths:
        bn = os.path.basename(p)
        m = ARXIV_ID_RE.search(bn)
        if m:
            aid = m.group(1) + (m.group(2) or "")
            by_arxiv.setdefault(aid, []).append(p)

        tokens = re.split(r"[_\-\s]+", bn.lower())
        for t in tokens:
            if len(t) >= 8 and t.isalnum():
                by_token.setdefault(t, []).append(p)

    return by_arxiv, by_token


def find_pdf_for_row(
    row: Dict[str, Any],
    pdf_dir: str,
    by_arxiv: Optional[Dict[str, List[str]]] = None,
    by_token: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[str], str]:
    title = (row.get("title") or "").strip()
    arxiv_id = (row.get("arxiv_id") or "").strip()
    pdf_path = (row.get("pdf_path") or "").strip()

    if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
        return pdf_path, "row_pdf_path"

    paper_url = (row.get("paper_url") or "").strip()
    pdf_url = (row.get("pdf_url") or "").strip()
    if not title and not paper_url and not pdf_url and not arxiv_id:
        return None, "skip_row"

    exp = expected_pdf_name(arxiv_id, title) if (arxiv_id or title) else ""
    if exp:
        cand = os.path.join(pdf_dir, exp)
        if os.path.exists(cand) and os.path.getsize(cand) > 1024:
            return cand, "exact_name"

    if by_arxiv is not None and arxiv_id and arxiv_id in by_arxiv:
        cands = sorted(by_arxiv[arxiv_id], key=lambda p: os.path.getsize(p), reverse=True)
        return cands[0], "arxiv_in_name"

    if by_token is not None and title:
        stitle = safe_filename(title).lower()
        tokens = [t for t in re.split(r"[_\-\s]+", stitle) if len(t) >= 8 and t.isalnum()]
        hits = []
        for t in tokens[:8]:
            hits.extend(by_token.get(t, []))
        hits = list(dict.fromkeys(hits))
        if hits:
            hits = sorted(hits, key=lambda p: os.path.getsize(p), reverse=True)
            return hits[0], "title_token_match"

    return None, "not_found"


def md_path_for_row(row: Dict[str, Any], md_dir: str) -> str:
    title = (row.get("title") or "").strip()
    arxiv_id = (row.get("arxiv_id") or "").strip()
    doc_id = arxiv_id if arxiv_id else safe_filename(title or "paper")
    return os.path.join(md_dir, safe_filename(doc_id) + ".md")
