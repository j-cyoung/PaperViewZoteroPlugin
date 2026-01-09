#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
H1_RE = re.compile(r"^#\s+(?!#)(.+?)\s*$")       # "# NeuIPS2025"
H2_RE = re.compile(r"^##\s+(?!#)(.+?)\s*$")      # "## Vision-Language-Action Model"
BULLET_RE = re.compile(r"^\s*-\s+(.*)$")

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

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

@dataclass
class PaperRow:
    venue: str
    year: Optional[int]
    category: Optional[str]
    alias: Optional[str]
    title: str
    paper_url: Optional[str]
    page_url: Optional[str]
    other_urls: str
    raw_line: str

def parse_markdown(md_text: str) -> List[PaperRow]:
    rows: List[PaperRow] = []
    venue: Optional[str] = None
    category: Optional[str] = None

    for line in md_text.splitlines():
        line = line.rstrip()

        h1 = H1_RE.match(line)
        if h1:
            venue = normalize_space(h1.group(1))
            category = None
            continue

        h2 = H2_RE.match(line)
        if h2:
            category = normalize_space(h2.group(1))
            continue

        b = BULLET_RE.match(line)
        if not b:
            continue

        bullet_text = b.group(1).strip()
        if not bullet_text:
            continue
        if is_toc_line(bullet_text):
            continue

        # extract links
        links = LINK_RE.findall(bullet_text)
        text_wo_links = LINK_RE.sub("", bullet_text)
        text_wo_links = normalize_space(text_wo_links)

        alias, title = extract_alias_and_title(text_wo_links)
        paper_url, page_url, other = pick_urls(links)

        # If venue is still None, keep but mark unknown (you也可以选择直接跳过)
        v = venue or "UNKNOWN"
        y = parse_year_from_venue(v)

        rows.append(
            PaperRow(
                venue=v,
                year=y,
                category=category,
                alias=alias,
                title=title,
                paper_url=paper_url,
                page_url=page_url,
                other_urls=json.dumps(other, ensure_ascii=False),
                raw_line=line.strip(),
            )
        )

    return rows

def write_csv(rows: List[PaperRow], path: str) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "venue", "year", "category", "alias", "title",
                "paper_url", "page_url", "other_urls", "raw_line"
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

def write_jsonl(rows: List[PaperRow], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path", help="Input markdown file (paper list)")
    ap.add_argument("--out_csv", default="papers.csv")
    ap.add_argument("--out_jsonl", default="papers.jsonl")
    args = ap.parse_args()

    with open(args.md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    rows = parse_markdown(md_text)
    write_csv(rows, args.out_csv)
    write_jsonl(rows, args.out_jsonl)

    print(f"Parsed {len(rows)} papers.")
    print(f"CSV  -> {args.out_csv}")
    print(f"JSONL-> {args.out_jsonl}")

if __name__ == "__main__":
    main()