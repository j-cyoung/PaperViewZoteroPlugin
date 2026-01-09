#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import hashlib
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import requests
from tqdm import tqdm

ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)(?P<id>\d{4}\.\d{4,5})(?P<v>v\d+)?",
    re.IGNORECASE
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

def safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[^\w\-.() ]+", "_", s).strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def is_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return u.endswith(".pdf") or "/pdf/" in u or "download" in u and "pdf" in u

def arxiv_pdf_url(arxiv_id: str) -> str:
    # arXiv supports /pdf/<id>.pdf and version suffixes
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def extract_arxiv_id_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = ARXIV_ID_RE.search(url)
    if not m:
        return None
    _id = m.group("id")
    v = m.group("v") or ""
    return f"{_id}{v}"

def extract_doi_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = DOI_RE.search(text)
    return m.group(0) if m else None

def requests_get(url: str, headers: Dict[str, str], params: Dict[str, Any] = None, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r

# -------------------------
# arXiv API (ATOM feed)
# -------------------------
def arxiv_api_query_by_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Query arXiv API using id_list. Returns a dict with keys:
      title, abstract, authors, published, doi (if present), arxiv_id, canonical_url, pdf_url
    """
    # arXiv API expects id_list without the "arxiv:" prefix
    # Keep version if provided.
    api_url = "https://export.arxiv.org/api/query"
    params = {"id_list": arxiv_id}
    r = requests.get(api_url, params=params, timeout=30)
    r.raise_for_status()
    xml = r.text

    # very light XML parsing via regex (enough for title/summary/authors/published/id/doi)
    # if you want stricter parsing, use feedparser or xml.etree.
    def pick(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
        if not m:
            return None
        return re.sub(r"\s+", " ", m.group(1)).strip()

    # entry id is like: http://arxiv.org/abs/2506.01953v1
    entry_id = pick("id")
    title = pick("title")
    abstract = pick("summary")
    published = pick("published")

    authors = re.findall(r"<name>(.*?)</name>", xml, re.DOTALL)
    authors = [re.sub(r"\s+", " ", a).strip() for a in authors]

    doi = None
    # arXiv API sometimes includes doi in arxiv:doi tag; try common patterns
    mdoi = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", xml, re.DOTALL)
    if mdoi:
        doi = re.sub(r"\s+", " ", mdoi.group(1)).strip()

    canonical_url = entry_id
    # normalize to https://arxiv.org/abs/...
    if canonical_url:
        canonical_url = canonical_url.replace("http://", "https://")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": published,
        "doi": doi,
        "canonical_url": canonical_url,
        "pdf_url": arxiv_pdf_url(arxiv_id),
        "source": "arxiv_api",
    }

# -------------------------
# Semantic Scholar Graph API
# -------------------------
def s2_headers() -> Dict[str, str]:
    h = {"User-Agent": "paper-enricher/0.1"}
    key = os.environ.get("S2_API_KEY", "").strip()
    if key:
        # S2 supports x-api-key for higher quotas
        h["x-api-key"] = key
    return h

def s2_get_paper_by_identifier(identifier: str, fields: str) -> Optional[Dict[str, Any]]:
    # identifier can be: "arXiv:2506.01953" or DOI or URL-encoded paperId
    url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}"
    r = requests_get(url, headers=s2_headers(), params={"fields": fields})
    return r.json()

def s2_search_paper(query: str, fields: str, limit: int = 1) -> Optional[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    r = requests_get(url, headers=s2_headers(), params={"query": query, "limit": limit, "fields": fields})
    return r.json()

def s2_pick_ids(s2: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ext = s2.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv") or ext.get("arXiv") or ext.get("arxiv")
    doi = ext.get("DOI") or ext.get("doi")
    return arxiv_id, doi

# -------------------------
# Crossref
# -------------------------
def crossref_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.crossref.org/works/{doi}"
    r = requests.get(url, headers={"User-Agent": "paper-enricher/0.1"}, timeout=30)
    r.raise_for_status()
    msg = r.json().get("message") or {}
    # Crossref abstract is sometimes in msg["abstract"] (often HTML-ish)
    return {
        "doi": doi,
        "title": (msg.get("title") or [None])[0],
        "abstract": msg.get("abstract"),
        "published": msg.get("created", {}).get("date-time"),
        "canonical_url": msg.get("URL"),
        "source": "crossref",
    }

# -------------------------
# PDF download
# -------------------------
def download_pdf(pdf_url: str, out_path: str) -> Tuple[bool, Optional[str]]:
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with requests.get(pdf_url, stream=True, timeout=60, headers={"User-Agent": "paper-enricher/0.1"}) as r:
            r.raise_for_status()
            # quick check content-type
            ct = (r.headers.get("Content-Type") or "").lower()
            if "pdf" not in ct and not out_path.lower().endswith(".pdf"):
                # some servers don't set content-type; we still try
                pass
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------------
# Main enrichment logic
# -------------------------
FIELDS_S2 = ",".join([
    "title",
    "abstract",
    "url",
    "year",
    "venue",
    "authors",
    "externalIds",
    "isOpenAccess",
    "openAccessPdf",
])

def enrich_one(row: Dict[str, str], cache: Dict[str, Any], sleep_s: float = 0.8) -> Dict[str, Any]:
    title = (row.get("title") or "").strip()
    paper_url = (row.get("paper_url") or "").strip()
    page_url = (row.get("page_url") or "").strip()

    out: Dict[str, Any] = dict(row)
    out.update({
        "arxiv_id": None,
        "doi": None,
        "canonical_url": None,
        "abstract": None,
        "authors": None,
        "published": None,
        "pdf_url": None,
        "source": None,
        "status": "ok",
        "error": None,
    })

    # 0) Direct DOI from URLs/text
    out["doi"] = extract_doi_from_text(" ".join([paper_url, page_url, title])) or None

    # 1) Direct arXiv id from paper_url if possible
    arxiv_id = extract_arxiv_id_from_url(paper_url)
    out["arxiv_id"] = arxiv_id

    # 2) If paper_url already pdf (e.g., physicalintelligence.company/download/*.pdf), keep as candidate
    if paper_url and is_pdf_url(paper_url):
        out["pdf_url"] = paper_url

    # 3) arXiv API first if we have arXiv id
    if arxiv_id:
        key = f"arxiv:{arxiv_id}"
        if key in cache:
            meta = cache[key]
        else:
            meta = arxiv_api_query_by_id(arxiv_id)
            cache[key] = meta
            time.sleep(sleep_s)

        if meta:
            out["canonical_url"] = meta.get("canonical_url") or out["canonical_url"]
            out["abstract"] = meta.get("abstract") or out["abstract"]
            out["authors"] = meta.get("authors") or out["authors"]
            out["published"] = meta.get("published") or out["published"]
            out["doi"] = out["doi"] or meta.get("doi")
            out["pdf_url"] = out["pdf_url"] or meta.get("pdf_url")
            out["source"] = out["source"] or meta.get("source")

    # 4) Semantic Scholar by identifier (arXiv/DOI) or by title search
    if not out["abstract"] or not out["doi"] or not out["arxiv_id"] or not out["pdf_url"]:
        s2_meta = None

        # 4.1 Try by arXiv id
        if out["arxiv_id"]:
            ident = f"arXiv:{out['arxiv_id']}"
            key = f"s2:{ident}"
            if key in cache:
                s2_meta = cache[key]
            else:
                try:
                    s2_meta = s2_get_paper_by_identifier(ident, fields=FIELDS_S2)
                    cache[key] = s2_meta
                    time.sleep(sleep_s)
                except Exception:
                    s2_meta = None

        # 4.2 Try by DOI
        if not s2_meta and out["doi"]:
            ident = f"DOI:{out['doi']}"
            key = f"s2:{ident}"
            if key in cache:
                s2_meta = cache[key]
            else:
                try:
                    s2_meta = s2_get_paper_by_identifier(ident, fields=FIELDS_S2)
                    cache[key] = s2_meta
                    time.sleep(sleep_s)
                except Exception:
                    s2_meta = None

        # 4.3 Fallback: search by title
        if not s2_meta and title:
            key = f"s2search:{sha1(title)}"
            if key in cache:
                sr = cache[key]
            else:
                try:
                    sr = s2_search_paper(title, fields=FIELDS_S2, limit=1)
                    cache[key] = sr
                    time.sleep(sleep_s)
                except Exception:
                    sr = None

            if sr and (sr.get("data") or []):
                s2_meta = sr["data"][0]

        if s2_meta:
            s2_arxiv, s2_doi = s2_pick_ids(s2_meta)
            out["arxiv_id"] = out["arxiv_id"] or s2_arxiv
            out["doi"] = out["doi"] or s2_doi
            out["canonical_url"] = out["canonical_url"] or s2_meta.get("url")
            out["abstract"] = out["abstract"] or s2_meta.get("abstract")
            # authors list -> names
            authors = s2_meta.get("authors")
            if authors and isinstance(authors, list):
                out["authors"] = out["authors"] or [a.get("name") for a in authors if a.get("name")]
            # openAccessPdf
            oap = s2_meta.get("openAccessPdf") or {}
            if not out["pdf_url"] and isinstance(oap, dict):
                out["pdf_url"] = oap.get("url") or out["pdf_url"]
            out["source"] = out["source"] or "semantic_scholar"

    # 5) Crossref by DOI if still no abstract
    if out["doi"] and not out["abstract"]:
        key = f"crossref:{out['doi']}"
        if key in cache:
            cr = cache[key]
        else:
            try:
                cr = crossref_by_doi(out["doi"])
                cache[key] = cr
                time.sleep(sleep_s)
            except Exception:
                cr = None
        if cr:
            out["canonical_url"] = out["canonical_url"] or cr.get("canonical_url")
            out["abstract"] = out["abstract"] or cr.get("abstract")
            out["published"] = out["published"] or cr.get("published")
            out["source"] = out["source"] or "crossref"

    # Final: if arxiv_id present and still no pdf_url, use arXiv pdf
    if out["arxiv_id"] and not out["pdf_url"]:
        out["pdf_url"] = arxiv_pdf_url(out["arxiv_id"])

    if not out["abstract"]:
        out["status"] = "missing_abstract"

    return out

def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # 取所有行 key 的并集，保证不会因为“后面行多了字段”而报错
    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    # 给字段一个稳定顺序：优先常用字段，其余按字母序
    preferred = [
        "venue","year","category","alias","title",
        "paper_url","page_url","other_urls","raw_line",
        "arxiv_id","doi","canonical_url",
        "abstract","authors","published",
        "pdf_url","pdf_download_error",
        "source","status","error",
    ]
    fields = []
    for k in preferred:
        if k in field_set:
            fields.append(k)
            field_set.remove(k)
    fields += sorted(field_set)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_cache(cache_path: str) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["k"]] = obj["v"]
    return cache

def append_cache(cache_path: str, k: str, v: Any) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("--out_csv", default="papers.enriched.csv")
    ap.add_argument("--out_jsonl", default="papers.enriched.jsonl")
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--download_pdf", action="store_true", default=False,
                    help="Download pdfs (only when pdf_url is available).")
    ap.add_argument("--sleep", type=float, default=0.8, help="Sleep between API calls to reduce rate-limit risk.")
    ap.add_argument("--cache", default=".cache_s2.jsonl")
    args = ap.parse_args()

    rows = load_csv(args.csv_in)
    cache = load_cache(args.cache)

    enriched: List[Dict[str, Any]] = []

    # We'll track cache keys we add during run and append to file (simple append-only)
    cache_added: List[Tuple[str, Any]] = []

    def cache_set(k: str, v: Any):
        if k not in cache:
            cache[k] = v
            cache_added.append((k, v))

    # monkey patch: wrap cache assignment inside enrich_one usage by capturing dict reference
    # (we keep enrich_one as-is but manually record new cache keys by diffing at the end)
    before_keys = set(cache.keys())

    for r in tqdm(rows, desc="Enriching"):
        try:
            out = enrich_one(r, cache=cache, sleep_s=args.sleep)
        except Exception as e:
            out = dict(r)
            out.update({
                "arxiv_id": None, "doi": None, "canonical_url": None, "abstract": None,
                "authors": None, "published": None, "pdf_url": None,
                "source": None, "status": "error", "error": str(e)
            })
        enriched.append(out)

    # Append newly added cache entries
    after_keys = set(cache.keys())
    new_keys = list(after_keys - before_keys)
    if new_keys:
        for k in new_keys:
            append_cache(args.cache, k, cache[k])

    # optional PDF download
    if args.download_pdf:
        os.makedirs(args.pdf_dir, exist_ok=True)
        for out in tqdm(enriched, desc="Downloading PDFs"):
            pdf_url = out.get("pdf_url")
            if not pdf_url:
                continue
            # file name
            title = out.get("title") or "paper"
            arxiv_id = out.get("arxiv_id") or ""
            base = safe_filename(f"{arxiv_id}_{title}".strip("_")) or "paper"
            out_path = os.path.join(args.pdf_dir, base + ".pdf")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                continue
            ok, err = download_pdf(pdf_url, out_path)
            if not ok:
                out["pdf_download_error"] = err

    write_csv(args.out_csv, enriched)
    write_jsonl(args.out_jsonl, enriched)

    print(f"Done. Enriched rows: {len(enriched)}")
    print(f"CSV:   {args.out_csv}")
    print(f"JSONL: {args.out_jsonl}")
    if args.download_pdf:
        print(f"PDF dir: {args.pdf_dir}")

if __name__ == "__main__":
    main()