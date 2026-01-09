#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")

def safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.() ]+", "_", (s or "").strip())
    if len(s) > max_len:
        s = s[:max_len]
    return s or "paper"

def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_pages_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    """
    建一个索引：优先用 arxiv_id 做 key；没有 arxiv_id 的用 title 做 key。
    这样能检测 papers.pages.jsonl 里哪些论文没有 md_pages / status 不对。
    """
    if not os.path.exists(path):
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            arxiv_id = (obj.get("arxiv_id") or "").strip()
            title = (obj.get("title") or "").strip()
            key = arxiv_id if arxiv_id else title
            if key:
                idx[key] = obj
    return idx

def list_pdfs(pdf_dir: str) -> List[str]:
    if not os.path.isdir(pdf_dir):
        return []
    out = []
    for fn in os.listdir(pdf_dir):
        if fn.lower().endswith(".pdf"):
            out.append(os.path.join(pdf_dir, fn))
    return out

def build_pdf_indices(pdf_paths: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    - by_arxiv: arxiv_id -> [pdf_paths] （文件名包含 2506.01953 的）
    - by_title_token: token -> [pdf_paths] （用于弱匹配）
    """
    by_arxiv: Dict[str, List[str]] = {}
    by_token: Dict[str, List[str]] = {}

    for p in pdf_paths:
        bn = os.path.basename(p)
        m = ARXIV_ID_RE.search(bn)
        if m:
            aid = m.group(1) + (m.group(2) or "")
            by_arxiv.setdefault(aid, []).append(p)

        # token index（非常弱匹配：只取一些长 token）
        tokens = re.split(r"[_\-\s]+", bn.lower())
        for t in tokens:
            if len(t) >= 8 and t.isalnum():
                by_token.setdefault(t, []).append(p)

    return by_arxiv, by_token

def expected_pdf_name(arxiv_id: str, title: str) -> str:
    base = safe_filename(f"{arxiv_id}_{title}".strip("_"))
    return base + ".pdf"

def find_pdf_for_row(row: Dict[str, str], pdf_dir: str,
                     by_arxiv: Dict[str, List[str]], by_token: Dict[str, List[str]]) -> Tuple[Optional[str], str]:
    """
    返回 (pdf_path, reason)
    reason:
      - exact_name
      - arxiv_in_name
      - title_token_match
      - not_found
      - skip_row
    """
    title = (row.get("title") or "").strip()
    arxiv_id = (row.get("arxiv_id") or "").strip()

    # 跳过明显无效行（比如 title/paper_url 都空的 TOC 垃圾行）
    paper_url = (row.get("paper_url") or "").strip()
    pdf_url = (row.get("pdf_url") or "").strip()
    if not title and not paper_url and not pdf_url and not arxiv_id:
        return None, "skip_row"

    # 1) exact expected name
    exp = expected_pdf_name(arxiv_id, title) if (arxiv_id or title) else ""
    if exp:
        cand = os.path.join(pdf_dir, exp)
        if os.path.exists(cand) and os.path.getsize(cand) > 1024:
            return cand, "exact_name"

    # 2) arxiv id appears in filename
    if arxiv_id and arxiv_id in by_arxiv:
        # 若多个候选，取文件最大者（一般是完整版）
        cands = sorted(by_arxiv[arxiv_id], key=lambda p: os.path.getsize(p), reverse=True)
        return cands[0], "arxiv_in_name"

    # 3) title token weak match (only if title exists)
    if title:
        # 用 safe_filename 后的 token 去匹配
        stitle = safe_filename(title).lower()
        tokens = [t for t in re.split(r"[_\-\s]+", stitle) if len(t) >= 8 and t.isalnum()]
        hits = []
        for t in tokens[:8]:
            hits.extend(by_token.get(t, []))
        hits = list(dict.fromkeys(hits))  # dedup preserve order

        if hits:
            hits = sorted(hits, key=lambda p: os.path.getsize(p), reverse=True)
            return hits[0], "title_token_match"

    return None, "not_found"

def md_path_for_row(row: Dict[str, str], md_dir: str) -> str:
    title = (row.get("title") or "").strip()
    arxiv_id = (row.get("arxiv_id") or "").strip()
    doc_id = arxiv_id if arxiv_id else safe_filename(title or "paper")
    return os.path.join(md_dir, safe_filename(doc_id) + ".md")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", default="papers.enriched.csv")
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--md_dir", default="mds")
    ap.add_argument("--pages_jsonl", default="papers.pages.jsonl")
    ap.add_argument("--out_csv", default="audit_report.csv")
    ap.add_argument("--top_k", type=int, default=0, help="只审计前K行（0=全部）")
    ap.add_argument("--no_tqdm", action="store_true", default=False)
    args = ap.parse_args()

    rows = load_csv(args.csv_in)
    if args.top_k and args.top_k > 0:
        rows = rows[:args.top_k]

    pages_idx = load_pages_jsonl(args.pages_jsonl)

    pdf_paths = list_pdfs(args.pdf_dir)
    by_arxiv, by_token = build_pdf_indices(pdf_paths)

    it = rows
    if not args.no_tqdm:
        it = tqdm(rows, desc="Auditing", unit="row")

    report: List[Dict[str, Any]] = []

    for i, row in enumerate(it, start=1):
        title = (row.get("title") or "").strip()
        arxiv_id = (row.get("arxiv_id") or "").strip()
        pdf_url = (row.get("pdf_url") or "").strip()
        paper_url = (row.get("paper_url") or "").strip()

        found_pdf, find_reason = find_pdf_for_row(row, args.pdf_dir, by_arxiv, by_token)
        md_path = md_path_for_row(row, args.md_dir)
        md_exists = os.path.exists(md_path) and os.path.getsize(md_path) > 256

        # pages_jsonl status
        key = arxiv_id if arxiv_id else title
        pinfo = pages_idx.get(key) or {}
        md_status = pinfo.get("md_status")
        has_md_pages = bool(pinfo.get("md_pages"))
        parse_ok = pinfo.get("parse_ok")
        md_written = pinfo.get("md_written")
        json_written = pinfo.get("json_written")

        # 分类 status（用于你手动排查）
        if find_reason == "skip_row":
            status = "skip_row"
        elif not pdf_url and not paper_url and not arxiv_id:
            status = "no_pdf_info_in_csv"
        elif found_pdf:
            # 有PDF文件
            if not md_exists:
                status = "pdf_found_but_md_missing"
            elif md_status != "ok" or (md_status == "ok" and not has_md_pages):
                status = "pdf_found_but_pages_jsonl_incomplete"
            else:
                status = "ok"
        else:
            # 没找到PDF文件
            if pdf_url or arxiv_id:
                status = "pdf_expected_but_file_missing"
            else:
                status = "pdf_unknown_no_arxiv_id"

        report.append({
            "row_idx": i,
            "status": status,
            "venue": row.get("venue"),
            "category": row.get("category"),
            "title": title,
            "arxiv_id": arxiv_id,
            "paper_url": paper_url,
            "pdf_url": pdf_url,
            "found_pdf_path": found_pdf or "",
            "find_reason": find_reason,
            "expected_pdf_name": expected_pdf_name(arxiv_id, title) if (arxiv_id or title) else "",
            "md_path": md_path,
            "md_exists": md_exists,
            "pages_jsonl_status": md_status or "",
            "has_md_pages": has_md_pages,
            "parse_ok": parse_ok if parse_ok is not None else "",
            "md_written": md_written if md_written is not None else "",
            "json_written": json_written if json_written is not None else "",
        })

    # 写 audit CSV
    fieldnames = list(report[0].keys()) if report else []
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report:
            w.writerow(r)

    # 打印摘要统计
    from collections import Counter
    c = Counter([r["status"] for r in report])
    print("Audit summary:")
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    print(f"Report written to: {args.out_csv}")

if __name__ == "__main__":
    main()