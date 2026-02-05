#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

import pymupdf4llm
from tqdm import tqdm

from paper_utils import (
    build_pdf_indices,
    compute_paper_id,
    compute_source_hash,
    find_pdf_for_row,
    list_pdfs,
    load_csv,
    load_jsonl,
    md_path_for_row,
)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

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


def ensure_ids(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row.get("paper_id"):
        row["paper_id"] = compute_paper_id(
            row.get("venue"),
            row.get("category"),
            row.get("alias"),
            row.get("title"),
            row.get("paper_url"),
            row.get("page_url"),
        )
    if not row.get("source_hash"):
        raw_line = row.get("raw_line") or row.get("title") or ""
        row["source_hash"] = compute_source_hash(
            row.get("venue"), row.get("category"), raw_line
        )
    return row


def load_existing_pages(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    rows = load_jsonl(path)
    idx: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        r = ensure_ids(r)
        pid = r.get("paper_id")
        if pid:
            idx[pid] = r
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True, help="papers.enriched.csv")
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--md_dir", default="mds")
    ap.add_argument("--out_jsonl", default="papers.pages.jsonl")
    ap.add_argument("--issues_out", default="papers.md_issues.csv")
    ap.add_argument("--base_output_dir", default="./store/ocr", help="输出根目录（相对路径会拼接到该目录）")

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
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--resume_from", default="", help="Optional path to prior pages jsonl")
    ap.add_argument("--progress_path", default="", help="Optional progress json path")

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
        args.pdf_dir = apply_base_dir(args.pdf_dir, args.base_output_dir)
        args.md_dir = apply_base_dir(args.md_dir, args.base_output_dir)
        args.out_jsonl = apply_base_dir(args.out_jsonl, args.base_output_dir)
        args.issues_out = apply_base_dir(args.issues_out, args.base_output_dir)
        args.image_dir = apply_base_dir(args.image_dir, args.base_output_dir)
        if args.progress_path:
            args.progress_path = apply_base_dir(args.progress_path, args.base_output_dir)

    def write_progress(done: int, total: int) -> None:
        if not args.progress_path:
            return
        payload = {
            "done": done,
            "total": total,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = args.progress_path + ".tmp"
        try:
            parent = os.path.dirname(args.progress_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, args.progress_path)
        except Exception:
            pass

    rows = load_csv(args.csv_in)
    if args.top_k and args.top_k > 0:
        rows = rows[:args.top_k]
    total = len(rows)
    done_count = 0

    def mark_progress() -> None:
        nonlocal done_count
        done_count += 1
        write_progress(done_count, total)

    safe_mkdir(args.md_dir)
    if args.write_images:
        safe_mkdir(args.image_dir)

    # Build PDF indices for faster lookup
    pdf_paths = list_pdfs(args.pdf_dir)
    by_arxiv, by_token = build_pdf_indices(pdf_paths)

    existing_idx: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        resume_path = args.resume_from or args.out_jsonl
        existing_idx = load_existing_pages(resume_path)

    # 写 jsonl
    out_f = open(args.out_jsonl, "w", encoding="utf-8")

    iterator = rows
    if not args.no_tqdm:
        iterator = tqdm(rows, desc="PDF→MD", unit="paper")

    n_ok = 0
    n_missing = 0
    n_error = 0
    issues: List[Dict[str, Any]] = []
    write_progress(done_count, total)

    for idx, row in enumerate(iterator, start=1):
        row = ensure_ids(dict(row))
        title = (row.get("title") or "").strip()
        arxiv_id = (row.get("arxiv_id") or "").strip()
        md_path = md_path_for_row(row, args.md_dir)

        if args.resume and row.get("paper_id") in existing_idx:
            old = existing_idx[row["paper_id"]]
            if old.get("source_hash") == row.get("source_hash") and old.get("md_status") == "ok":
                md_exists = os.path.exists(md_path) and os.path.getsize(md_path) > 256
                if md_exists:
                    merged = dict(old)
                    merged.update(row)
                    merged["md_path"] = md_path
                    merged["json_written"] = True
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    n_ok += 1
                    if args.verbose:
                        print(f"[{idx}] resume_ok | arxiv_id={arxiv_id} | md={md_path} | title={title[:80]}")
                    mark_progress()
                    continue

        pdf_path, find_reason = find_pdf_for_row(row, args.pdf_dir, by_arxiv, by_token)
        if not pdf_path:
            n_missing += 1
            obj = dict(row)
            obj.update({
                "pdf_path": None,
                "pdf_find_reason": find_reason,
                "md_path": md_path,
                "md_status": "missing_pdf",
                "parse_ok": False,
                "md_written": False,
                "json_written": True,   # jsonl 这行写得出去
                "md_error": None,
            })
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            issues.append({
                "paper_id": obj.get("paper_id"),
                "title": title,
                "md_status": obj.get("md_status"),
                "md_error": obj.get("md_error"),
                "pdf_find_reason": find_reason,
            })
            if args.verbose:
                print(f"[{idx}] missing_pdf | arxiv_id={arxiv_id} | title={title[:80]}")
            mark_progress()
            continue

        # 默认先设为失败，后续逐步置 True（避免“写 md 成功但 json 失败”的错觉）
        parse_ok = False
        md_written = False
        json_written = False
        md_error = None

        obj = dict(row)
        obj.update({
            "pdf_path": pdf_path,
            "pdf_find_reason": find_reason,
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
            issues.append({
                "paper_id": obj.get("paper_id"),
                "title": title,
                "md_status": obj.get("md_status"),
                "md_error": obj.get("md_error"),
                "pdf_find_reason": obj.get("pdf_find_reason"),
            })

            if args.verbose:
                print(f"[{idx}] {obj['md_status']} | arxiv_id={arxiv_id} | title={title[:80]} | err={md_error}")
        mark_progress()

    out_f.close()
    if issues:
        # small issue report for quick inspection
        fieldnames = ["paper_id", "title", "md_status", "md_error", "pdf_find_reason"]
        with open(args.issues_out, "w", encoding="utf-8", newline="") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in issues:
                w.writerow(r)

    print(f"Done. ok={n_ok}, error={n_error}, missing_pdf={n_missing}, out={args.out_jsonl}, md_dir={args.md_dir}")
    if issues:
        print(f"Issues: {len(issues)} -> {args.issues_out}")
    write_progress(total, total)


if __name__ == "__main__":
    main()
