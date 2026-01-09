#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from paper_utils import (
    compute_paper_id,
    compute_source_hash,
    load_csv,
    load_jsonl,
    normalize_space,
    write_csv,
    write_jsonl,
)

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
H1_RE = re.compile(r"^#\s+(?!#)(.+?)\s*$")       # "# NeuIPS2025"
H2_RE = re.compile(r"^##\s+(?!#)(.+?)\s*$")      # "## Vision-Language-Action Model"
BULLET_RE = re.compile(r"^\s*-\s+(.*)$")

def parse_year_from_venue(venue: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", venue)
    return int(m.group(0)) if m else None

def strip_markdown_emphasis(s: str) -> str:
    # remove **bold** and *italic* markers but keep content
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    return s

def extract_alias_and_title(text_wo_links: str) -> Tuple[Optional[str], str]:
    """
    Handles patterns like:
      **Fast-in-Slow**: A Dual-System VLA Model ...
      *Hyper-GoalNet*: ...
      Plain Title ...
    Returns (alias, title_text)
    """
    t = text_wo_links.strip()

    # alias in **...**
    m = re.match(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$", t)
    if m:
        alias = normalize_space(m.group(1))
        rest = normalize_space(m.group(2)) if m.group(2) else ""
        title = rest if rest else alias
        return alias, title

    # alias in *...*
    m = re.match(r"^\*(.+?)\*\s*:?\s*(.*)$", t)
    if m:
        alias = normalize_space(m.group(1))
        rest = normalize_space(m.group(2)) if m.group(2) else ""
        title = rest if rest else alias
        return alias, title

    return None, normalize_space(strip_markdown_emphasis(t))

def pick_urls(links: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str], Dict[str, List[str]]]:
    """
    Prefer:
      paper_url: label in {"paper","pdf"} or arxiv abs/pdf
      page_url:  label in {"page","project","website","home"}
    others: everything else grouped by label
    """
    if not links:
        return None, None, {}

    # group by lower label
    grouped: Dict[str, List[str]] = {}
    for label, url in links:
        key = label.strip().lower()
        grouped.setdefault(key, []).append(url.strip())

    def first_of(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in grouped and grouped[k]:
                return grouped[k][0]
        return None

    paper_url = first_of(["paper", "pdf"])
    page_url = first_of(["page", "project", "website", "home"])

    # arXiv fallback for paper_url
    if paper_url is None:
        for _, u in links:
            ul = u.lower()
            if "arxiv.org/abs/" in ul or "arxiv.org/pdf/" in ul:
                paper_url = u
                break

    # still none? use first link as paper_url
    if paper_url is None and links:
        paper_url = links[0][1]

    # other urls
    other = {k: v for k, v in grouped.items() if k not in {"paper", "pdf", "page", "project", "website", "home"}}
    return paper_url, page_url, other

def is_toc_heading(text: str) -> bool:
    t = normalize_space(text).lower()
    return "paper list" in t

def is_toc_line(bullet_text: str) -> bool:
    """
    Skip table-of-contents bullets like:
      - [📖 NeuIPS2025](#neuips2025)
    """
    # if it's just a single internal anchor link (and contains "#...") it's likely TOC
    links = LINK_RE.findall(bullet_text)
    if len(links) == 1:
        label, url = links[0]
        if url.startswith("#"):
            return True
    # also skip bullets that are purely a link list with emojis
    if bullet_text.strip().startswith("[") and "(#" in bullet_text:
        return True
    return False

def title_from_links(text_wo_links: str, links: List[Tuple[str, str]]) -> str:
    if text_wo_links:
        return text_wo_links
    if not links:
        return ""
    label = (links[0][0] or "").strip()
    if not label:
        return ""
    if label.lower() in {"paper", "pdf", "page", "project", "website", "home"}:
        return ""
    return normalize_space(label)

@dataclass
class PaperRow:
    paper_id: str
    source_hash: str
    venue: str
    year: Optional[int]
    category: Optional[str]
    alias: Optional[str]
    title: str
    paper_url: Optional[str]
    page_url: Optional[str]
    other_urls: str
    raw_line: str
    parse_status: str
    parse_error: Optional[str]

@dataclass
class ParseIssue:
    line_no: int
    reason: str
    raw_line: str
    venue: Optional[str]
    category: Optional[str]

def parse_markdown(md_text: str) -> Tuple[List[PaperRow], List[ParseIssue]]:
    rows: List[PaperRow] = []
    issues: List[ParseIssue] = []
    venue: Optional[str] = None
    category: Optional[str] = None
    in_toc = False

    for line_no, line in enumerate(md_text.splitlines(), start=1):
        line = line.rstrip()

        h1 = H1_RE.match(line)
        if h1:
            venue = normalize_space(h1.group(1))
            category = None
            in_toc = False
            continue

        h2 = H2_RE.match(line)
        if h2:
            heading = normalize_space(h2.group(1))
            if is_toc_heading(heading):
                in_toc = True
                category = None
                continue
            if not in_toc:
                category = heading
            continue

        b = BULLET_RE.match(line)
        if not b:
            continue

        bullet_text = b.group(1).strip()
        if not bullet_text:
            continue
        if in_toc:
            issues.append(ParseIssue(
                line_no=line_no,
                reason="toc_block",
                raw_line=line.strip(),
                venue=venue,
                category=category,
            ))
            continue
        if not venue:
            issues.append(ParseIssue(
                line_no=line_no,
                reason="missing_venue",
                raw_line=line.strip(),
                venue=venue,
                category=category,
            ))
            continue
        if is_toc_line(bullet_text):
            issues.append(ParseIssue(
                line_no=line_no,
                reason="toc_anchor",
                raw_line=line.strip(),
                venue=venue,
                category=category,
            ))
            continue

        # extract links
        links = LINK_RE.findall(bullet_text)
        text_wo_links = LINK_RE.sub("", bullet_text)
        text_wo_links = normalize_space(text_wo_links)
        text_wo_links = title_from_links(text_wo_links, links)

        if not text_wo_links:
            issues.append(ParseIssue(
                line_no=line_no,
                reason="missing_title",
                raw_line=line.strip(),
                venue=venue,
                category=category,
            ))
            continue

        alias, title = extract_alias_and_title(text_wo_links)
        paper_url, page_url, other = pick_urls(links)

        # If venue is still None, keep but mark unknown (you也可以选择直接跳过)
        v = venue or "UNKNOWN"
        y = parse_year_from_venue(v)
        source_hash = compute_source_hash(v, category, line.strip())
        paper_id = compute_paper_id(v, category, alias, title, paper_url, page_url)

        rows.append(
            PaperRow(
                paper_id=paper_id,
                source_hash=source_hash,
                venue=v,
                year=y,
                category=category,
                alias=alias,
                title=title,
                paper_url=paper_url,
                page_url=page_url,
                other_urls=json.dumps(other, ensure_ascii=False),
                raw_line=line.strip(),
                parse_status="ok",
                parse_error=None,
            )
        )

    return rows, issues

def load_existing_rows(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
    else:
        rows = load_csv(path)
    idx: Dict[str, Dict[str, str]] = {}
    for r in rows:
        pid = r.get("paper_id")
        if not pid:
            pid = compute_paper_id(
                r.get("venue"),
                r.get("category"),
                r.get("alias"),
                r.get("title"),
                r.get("paper_url"),
                r.get("page_url"),
            )
        if pid:
            idx[pid] = r
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path", help="Input markdown file (paper list)")
    ap.add_argument("--out_csv", default="papers.csv")
    ap.add_argument("--out_jsonl", default="papers.jsonl")
    ap.add_argument("--issues_out", default="papers.parse_issues.jsonl")
    ap.add_argument("--base_output_dir", default="./store/parse", help="输出根目录（相对路径会拼接到该目录）")
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--resume_from", default="", help="Optional path to prior parse output (csv/jsonl)")
    args = ap.parse_args()

    def apply_base_dir(path: str, base_dir: str) -> str:
        if not base_dir or os.path.isabs(path):
            return path
        base_norm = os.path.normpath(base_dir)
        path_norm = os.path.normpath(path)
        if path_norm == base_norm or path_norm.startswith(base_norm + os.sep):
            return path
        return os.path.join(base_dir, path)

    if args.base_output_dir:
        os.makedirs(os.path.dirname(args.base_output_dir), exist_ok=True)
        args.out_csv = apply_base_dir(args.out_csv, args.base_output_dir)
        args.out_jsonl = apply_base_dir(args.out_jsonl, args.base_output_dir)
        args.issues_out = apply_base_dir(args.issues_out, args.base_output_dir)

    with open(args.md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    rows, issues = parse_markdown(md_text)

    preferred_fields = [
        "paper_id", "source_hash",
        "venue", "year", "category", "alias", "title",
        "paper_url", "page_url", "other_urls", "raw_line",
        "parse_status", "parse_error",
    ]

    out_rows: List[Dict[str, str]] = []
    existing_idx: Dict[str, Dict[str, str]] = {}
    if args.resume:
        resume_path = args.resume_from or (args.out_jsonl if os.path.exists(args.out_jsonl) else args.out_csv)
        existing_idx = load_existing_rows(resume_path)

    for r in rows:
        row = asdict(r)
        if args.resume and row.get("paper_id") in existing_idx:
            old = existing_idx[row["paper_id"]]
            if old.get("source_hash") == row.get("source_hash") and old.get("parse_status", "ok") == "ok":
                merged = dict(old)
                merged.update(row)
                out_rows.append(merged)
                continue
        out_rows.append(row)

    write_csv(args.out_csv, out_rows, preferred_fields=preferred_fields)
    write_jsonl(args.out_jsonl, out_rows)
    if issues:
        write_jsonl(args.issues_out, [asdict(i) for i in issues])

    print(f"Parsed {len(out_rows)} papers.")
    if issues:
        print(f"Parse issues: {len(issues)} -> {args.issues_out}")
    print(f"CSV  -> {args.out_csv}")
    print(f"JSONL-> {args.out_jsonl}")

if __name__ == "__main__":
    main()
