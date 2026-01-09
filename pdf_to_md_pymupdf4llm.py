#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from typing import Dict, Any, List, Optional, Union

import pymupdf4llm
from tqdm import tqdm


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_filename(s: str, max_len: int = 180) -> str:
    import re
    s = re.sub(r"[^\w\-.() ]+", "_", (s or "").strip())
    return (s[:max_len] if len(s) > max_len else s) or "paper"


def pick_pdf_path(row: Dict[str, str], pdf_dir: str) -> Optional[str]:
    """
    更鲁棒的 PDF 匹配：
    1) 先按 enrich 的 expected name 精确匹配
    2) 再在 pdf_dir 里扫描：文件名包含 arxiv_id
    3) 再做 title token 弱匹配
    """
    arxiv_id = (row.get("arxiv_id") or "").strip()
    title = (row.get("title") or "").strip()

    # 1) exact expected name
    base = safe_filename(f"{arxiv_id}_{title}".strip("_"))
    cand = os.path.join(pdf_dir, base + ".pdf")
    if os.path.exists(cand) and os.path.getsize(cand) > 1024:
        return cand

    # 2) scan by arxiv_id substring in filename
    if arxiv_id:
        for fn in os.listdir(pdf_dir):
            if fn.lower().endswith(".pdf") and arxiv_id in fn:
                p = os.path.join(pdf_dir, fn)
                if os.path.getsize(p) > 1024:
                    return p

    # 3) weak match by title tokens
    stitle = safe_filename(title).lower()
    tokens = [t for t in re.split(r"[_\-\s]+", stitle) if len(t) >= 8 and t.isalnum()]
    if tokens:
        pdfs = [fn for fn in os.listdir(pdf_dir) if fn.lower().endswith(".pdf")]
        for fn in pdfs:
            low = fn.lower()
            hit = sum(1 for t in tokens[:8] if t in low)
            if hit >= 2:  # 阈值可调
                p = os.path.join(pdf_dir, fn)
                if os.path.getsize(p) > 1024:
                    return p

    # 4) fallback: title-only exact
    cand2 = os.path.join(pdf_dir, safe_filename(title) + ".pdf")
    if os.path.exists(cand2) and os.path.getsize(cand2) > 1024:
        return cand2

    return None


def convert_one_pdf(
    pdf_path: str,
    page_chunks: bool,
    write_images: bool,
    image_path: Optional[str],
    image_format: str,
    dpi: int,
    extract_words: bool,
) -> Union[str, List[Dict[str, Any]]]:
    return pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=page_chunks,
        write_images=write_images,
        image_path=image_path,
        image_format=image_format,
        dpi=dpi,
        extract_words=extract_words,
    )


def sanitize_md_pages(md_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    关键修复点：
    PyMuPDF4LLM 返回的 page_chunks 里可能包含不可 JSON 序列化对象。
    我们只保留后续分析必需的字段：
      - metadata.page
      - text
    这样 papers.pages.jsonl 一定能稳定写入，md_status 不会“假 error”。
    """
    out: List[Dict[str, Any]] = []
    for c in md_pages:
        meta = c.get("metadata") or {}
        out.append({
            "metadata": {"page": meta.get("page")},
            "text": c.get("text") or "",
        })
    return out


def build_md_text_from_pages(md_pages_sanitized: List[Dict[str, Any]]) -> str:
    joined = []
    for c in md_pages_sanitized:
        meta = c.get("metadata", {})
        p = meta.get("page")
        joined.append(f"\n\n---\n\n# Page {p}\n\n")
        joined.append(c.get("text") or "")
    return "".join(joined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True, help="papers.enriched.csv")
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--md_dir", default="mds")
    ap.add_argument("--out_jsonl", default="papers.pages.jsonl")

    ap.add_argument("--page_chunks", action="store_true", default=True,
                    help="page_chunks=True: 输出按页 chunks（推荐）；否则输出整篇 markdown 字符串")
    ap.add_argument("--write_images", action="store_true", default=False,
                    help="是否抽取图片并在 markdown 中插入引用（会写文件）")
    ap.add_argument("--image_dir", default="images")
    ap.add_argument("--image_format", default="jpg")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--extract_words", action="store_true", default=False)

    # ✅ 调试/加速功能
    ap.add_argument("--top_k", type=int, default=0,
                    help="只处理前 K 篇论文（0 表示处理全部）")
    ap.add_argument("--verbose", action="store_true", default=False,
                    help="打印每篇论文的处理信息（title/arxiv_id/status/path）")
    ap.add_argument("--no_tqdm", action="store_true", default=False,
                    help="Disable tqdm progress bar")

    args = ap.parse_args()

    rows = load_csv(args.csv_in)
    if args.top_k and args.top_k > 0:
        rows = rows[:args.top_k]

    safe_mkdir(args.md_dir)
    if args.write_images:
        safe_mkdir(args.image_dir)

    # 写 jsonl
    out_f = open(args.out_jsonl, "w", encoding="utf-8")

    iterator = rows
    if not args.no_tqdm:
        iterator = tqdm(rows, desc="PDF→MD", unit="paper")

    n_ok = 0
    n_missing = 0
    n_error = 0

    for idx, row in enumerate(iterator, start=1):
        title = (row.get("title") or "").strip()
        arxiv_id = (row.get("arxiv_id") or "").strip()
        doc_id = arxiv_id or safe_filename(title or "paper")
        md_path = os.path.join(args.md_dir, safe_filename(doc_id) + ".md")

        pdf_path = pick_pdf_path(row, args.pdf_dir)
        if not pdf_path:
            n_missing += 1
            obj = dict(row)
            obj.update({
                "pdf_path": None,
                "md_path": md_path,
                "md_status": "missing_pdf",
                "parse_ok": False,
                "md_written": False,
                "json_written": True,   # jsonl 这行写得出去
                "md_error": None,
            })
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if args.verbose:
                print(f"[{idx}] missing_pdf | arxiv_id={arxiv_id} | title={title[:80]}")
            continue

        # 默认先设为失败，后续逐步置 True（避免“写 md 成功但 json 失败”的错觉）
        parse_ok = False
        md_written = False
        json_written = False
        md_error = None

        obj = dict(row)
        obj.update({
            "pdf_path": pdf_path,
            "md_path": md_path,
            "md_status": "init",
            "parse_ok": False,
            "md_written": False,
            "json_written": False,
            "md_error": None,
        })

        try:
            md_or_chunks = convert_one_pdf(
                pdf_path=pdf_path,
                page_chunks=args.page_chunks,
                write_images=args.write_images,
                image_path=args.image_dir if args.write_images else None,
                image_format=args.image_format,
                dpi=args.dpi,
                extract_words=args.extract_words,
            )
            parse_ok = True

            if args.page_chunks:
                if not isinstance(md_or_chunks, list):
                    raise TypeError(f"page_chunks=True but got {type(md_or_chunks)}")
                md_pages_s = sanitize_md_pages(md_or_chunks)
                md_text = build_md_text_from_pages(md_pages_s)

                # 写 md 文件
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_text)
                md_written = True

                # 写 jsonl：只写可序列化 md_pages
                obj["md_pages"] = md_pages_s

            else:
                if not isinstance(md_or_chunks, str):
                    raise TypeError(f"page_chunks=False but got {type(md_or_chunks)}")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_or_chunks)
                md_written = True
                obj["md_text"] = md_or_chunks

            # 统一写状态（现在 parse_ok/md_written 都已确定）
            obj["parse_ok"] = parse_ok
            obj["md_written"] = md_written
            obj["md_status"] = "ok" if (parse_ok and md_written) else "partial_ok"

            # 写 jsonl（保证可序列化）
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            json_written = True
            obj["json_written"] = True

            n_ok += 1

            if args.verbose:
                print(f"[{idx}] ok | arxiv_id={arxiv_id} | md={md_path} | pdf={os.path.basename(pdf_path)} | title={title[:80]}")

        except Exception as e:
            md_error = str(e)

            # 尽量把当时的状态写清楚：可能 parse_ok=True 但 json 序列化失败（现在理论上不会）
            obj["parse_ok"] = parse_ok
            obj["md_written"] = md_written
            obj["json_written"] = json_written
            obj["md_error"] = md_error

            # 更细的状态：如果 md 写了但后面失败，标成 partial_ok（比 error 更贴近事实）
            if parse_ok and md_written:
                obj["md_status"] = "partial_ok"
            else:
                obj["md_status"] = "error"

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_error += 1

            if args.verbose:
                print(f"[{idx}] {obj['md_status']} | arxiv_id={arxiv_id} | title={title[:80]} | err={md_error}")

    out_f.close()
    print(f"Done. ok={n_ok}, error={n_error}, missing_pdf={n_missing}, out={args.out_jsonl}, md_dir={args.md_dir}")


if __name__ == "__main__":
    main()