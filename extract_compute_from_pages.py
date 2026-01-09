#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from paper_utils import (
    compute_paper_id,
    compute_source_hash,
    load_csv,
    load_jsonl,
    write_csv,
    write_jsonl,
)

# ============================================================
# 1. GPU / Compute 关键词与正则
# ============================================================

GPU_KEYWORDS_RE = re.compile(
    r"\b(gpu|gpus|cuda|nvidia|accelerator|tpus?|a100|h100|v100|t4|a6000|l40s?|rtx\s?4090|4090|3090)\b",
    re.IGNORECASE
)

GPU_MODEL_PATTERNS = [
    r"A100(?:\s*[-–]?\s*80GB|\s*80GB)?",
    r"H100(?:\s*(?:SXM|PCIE|PCIe))?",
    r"V100",
    r"T4",
    r"P100",
    r"(?:RTX\s*)?A6000",
    r"L40S?",
    r"(?:GeForce\s*)?(?:RTX\s*)?4090D?",
    r"(?:GeForce\s*)?(?:RTX\s*)?3090",
]

GPU_MODEL_RE = re.compile(
    r"\b(" + "|".join(GPU_MODEL_PATTERNS) + r")\b",
    re.IGNORECASE
)

MEM_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(GB|G|GiB|MB|M|MiB|TB|T|TiB)\b",
    re.IGNORECASE
)

COMPUTE_KEYWORDS_RE = re.compile(
    r"\b(compute|computational|training time|gpu[- ]?hours|gpu[- ]?days|flops?|"
    r"tflops?|pflops?|throughput|runtime|memory|vram)\b",
    re.IGNORECASE
)

# 数量 × 型号
COUNT_X_MODEL_RE = re.compile(
    r"\b(\d+)\s*([x×\*])\s*(" + "|".join(GPU_MODEL_PATTERNS) + r")\b",
    re.IGNORECASE
)

MODEL_X_COUNT_RE = re.compile(
    r"\b(" + "|".join(GPU_MODEL_PATTERNS) + r")\s*([x×\*])\s*(\d+)\b",
    re.IGNORECASE
)

COUNT_MODEL_GPUS_RE = re.compile(
    r"\b(\d+)\s+(" + "|".join(GPU_MODEL_PATTERNS) + r")\s+gpus?\b",
    re.IGNORECASE
)

COUNT_GPUS_RE = re.compile(
    r"\b(on|with|using|trained on|train on)\s+(\d+)\s+gpus?\b",
    re.IGNORECASE
)

# ============================================================
# 2. 工具函数
# ============================================================

def page_no(chunk: Dict[str, Any]) -> Optional[int]:
    meta = chunk.get("metadata") or {}
    p = meta.get("page")
    try:
        return int(p)
    except Exception:
        return None

def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        return paras
    return [l.strip() for l in text.splitlines() if l.strip()]

def line_window(lines: List[str], i: int, r: int = 2) -> str:
    lo = max(0, i - r)
    hi = min(len(lines), i + r + 1)
    return " ".join(lines[lo:hi])

def mem_to_gb(value: float, unit: str) -> float:
    u = unit.lower()
    if u in {"mb", "m", "mib"}:
        return value / 1024.0
    if u in {"tb", "t", "tib"}:
        return value * 1024.0
    return value

def extract_mem_gb(text: str) -> List[float]:
    mems = []
    for m in MEM_RE.finditer(text):
        try:
            val = float(m.group(1))
        except Exception:
            continue
        mems.append(mem_to_gb(val, m.group(2)))
    return mems

def detect_coarse_hits(text: str) -> List[str]:
    hits = []
    if GPU_KEYWORDS_RE.search(text):
        hits.append("gpu_keyword")
    if GPU_MODEL_RE.search(text):
        hits.append("gpu_model")
    if MEM_RE.search(text):
        hits.append("memory")
    if COMPUTE_KEYWORDS_RE.search(text):
        hits.append("compute_keyword")
    return hits

def detect_fine_hits(text: str) -> List[str]:
    hits = []
    if GPU_MODEL_RE.search(text):
        hits.append("gpu_model")
    if COUNT_X_MODEL_RE.search(text) or MODEL_X_COUNT_RE.search(text) or COUNT_MODEL_GPUS_RE.search(text):
        hits.append("gpu_count_model")
    if COUNT_GPUS_RE.search(text):
        hits.append("gpu_count")
    if MEM_RE.search(text) and (GPU_KEYWORDS_RE.search(text) or GPU_MODEL_RE.search(text)):
        hits.append("gpu_memory")
    return hits

def extract_match_contexts(text: str, window: int = 60) -> List[Dict[str, str]]:
    if not text:
        return []
    specs = [
        ("gpu_keyword", GPU_KEYWORDS_RE),
        ("gpu_model", GPU_MODEL_RE),
        ("memory", MEM_RE),
        ("compute_keyword", COMPUTE_KEYWORDS_RE),
        ("count_x_model", COUNT_X_MODEL_RE),
        ("model_x_count", MODEL_X_COUNT_RE),
        ("count_model_gpus", COUNT_MODEL_GPUS_RE),
        ("count_gpus", COUNT_GPUS_RE),
    ]
    out: List[Dict[str, str]] = []
    seen = set()
    for label, regex in specs:
        for m in regex.finditer(text):
            start, end = m.span()
            key = (label, start, end)
            if key in seen:
                continue
            seen.add(key)
            ctx = text[max(0, start - window):min(len(text), end + window)]
            out.append({
                "type": label,
                "match": m.group(0),
                "context": ctx,
            })
    return out

def collect_context_strings(
    snippets: List[Dict[str, Any]],
    max_items: int = 12,
    context_limit: int = 200,
) -> List[str]:
    out: List[str] = []
    for snip in snippets:
        page = snip.get("page")
        page_tag = f"p{page}" if page is not None else "p?"
        for ctx in snip.get("match_contexts") or []:
            text = ctx.get("context") or ""
            if len(text) > context_limit:
                text = text[:context_limit] + "..."
            item = f"{page_tag}:{ctx.get('type')}:{ctx.get('match')}:{text}"
            out.append(item)
            if len(out) >= max_items:
                return out
    return out

def collect_coarse_snippets(text: str, max_snips: int = 20) -> List[Dict[str, Any]]:
    """
    召回所有可能与算力相关的片段：
    - GPU 关键词
    - 显存/内存单位 (MB/GB/TB)
    - Compute / FLOPs / runtime 等关键词
    """
    if not text:
        return []

    snips: List[Dict[str, Any]] = []

    # 1) 段落级召回
    paras = split_paragraphs(text)
    for p in paras:
        hits = detect_coarse_hits(p)
        if hits:
            snips.append({"text": p[:1800], "hits": hits})
            if len(snips) >= max_snips:
                return snips

    # 2) 行窗口召回（markdown 崩坏兜底）
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        w = line_window(lines, i)
        hits = detect_coarse_hits(w)
        if hits:
            snips.append({"text": w[:1800], "hits": hits})
            if len(snips) >= max_snips:
                break

    # 去重
    out, seen = [], set()
    for s in snips:
        key = s["text"]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

def norm_model(m: str) -> str:
    m = re.sub(r"\s+", " ", m.strip())
    m = m.replace("PCIe", "PCIE")
    m = re.sub(r"\s*[-–]?\s*\d{2,3}\s*(GB|G)\b", "", m, flags=re.IGNORECASE)
    return m.strip()

# ============================================================
# 3. 单 snippet GPU 解析
# ============================================================

def parse_gpu_from_snippet(snippet: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def add(count, model, mem, conf):
        items.append({
            "count": count,
            "model": model,
            "mem_gb": mem,
            "confidence": conf
        })

    s = snippet

    # 1) 数量 × 型号
    for m in COUNT_X_MODEL_RE.finditer(s):
        cnt = int(m.group(1))
        model_raw = m.group(3)
        mem = None
        mm = MEM_RE.search(model_raw)
        if mm:
            mem = mem_to_gb(float(mm.group(1)), mm.group(2))
        add(cnt, norm_model(model_raw), mem, "high")

    for m in MODEL_X_COUNT_RE.finditer(s):
        model_raw = m.group(1)
        cnt = int(m.group(3))
        mem = None
        mm = MEM_RE.search(model_raw)
        if mm:
            mem = mem_to_gb(float(mm.group(1)), mm.group(2))
        add(cnt, norm_model(model_raw), mem, "high")

    for m in COUNT_MODEL_GPUS_RE.finditer(s):
        cnt = int(m.group(1))
        model_raw = m.group(2)
        mem = None
        mm = MEM_RE.search(s[m.start():m.end()+40])
        if mm:
            mem = mem_to_gb(float(mm.group(1)), mm.group(2))
        add(cnt, norm_model(model_raw), mem, "high")

    # 2) on N GPUs + 型号在同 snippet
    for m in COUNT_GPUS_RE.finditer(s):
        cnt = int(m.group(2))
        models = [norm_model(x.group(1)) for x in GPU_MODEL_RE.finditer(s)]
        if models:
            add(cnt, models[0], None, "medium")
        else:
            add(cnt, None, None, "low")

    # 3) 只有型号（无数量）
    if not items:
        for m in GPU_MODEL_RE.finditer(s):
            model_raw = m.group(1)
            mem = None
            mm = MEM_RE.search(s)
            if mm:
                mem = mem_to_gb(float(mm.group(1)), mm.group(2))
            add(None, norm_model(model_raw), mem, "low")

    # 去重
    seen, out = set(), []
    for it in items:
        key = (it["count"], it["model"], it["mem_gb"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def merge_gpu_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        key = it["model"] or "UNKNOWN"
        by_model.setdefault(key, []).append(it)

    merged = []
    for model, lst in by_model.items():
        counts = [x["count"] for x in lst if isinstance(x["count"], int)]
        mems = [x["mem_gb"] for x in lst if isinstance(x["mem_gb"], int)]
        confs = [x["confidence"] for x in lst]

        merged.append({
            "model": model,
            "count": max(counts) if counts else None,
            "mem_gb": max(set(mems), key=mems.count) if mems else None,
            "confidence": "high" if "high" in confs else ("medium" if "medium" in confs else "low")
        })
    return merged

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

def load_existing(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
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

def merge_meta(obj: Dict[str, Any], meta_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not meta_map:
        return obj
    key = obj.get("paper_id") or obj.get("arxiv_id") or obj.get("title")
    if not key:
        return obj
    meta = meta_map.get(key)
    if not meta:
        return obj
    merged = dict(meta)
    merged.update(obj)
    return merged

def build_meta_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        r = ensure_ids(r)
        if r.get("paper_id"):
            idx[r["paper_id"]] = r
        if r.get("arxiv_id"):
            idx[r["arxiv_id"]] = r
        if r.get("title"):
            idx[r["title"]] = r
    return idx

def iter_page_texts(obj: Dict[str, Any], max_pages: int = 0) -> List[Tuple[Optional[int], str]]:
    pages = []
    if obj.get("md_pages"):
        for ch in obj.get("md_pages") or []:
            p = page_no(ch)
            if max_pages and p is not None and p > max_pages:
                continue
            pages.append((p, ch.get("text") or ""))
    elif obj.get("md_text"):
        pages.append((None, obj.get("md_text") or ""))
    else:
        md_path = obj.get("md_path")
        if md_path and os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    pages.append((None, f.read()))
            except Exception:
                pass
    return pages

# ============================================================
# 4. 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_jsonl", default="papers.pages.jsonl", help="papers.pages.jsonl")
    ap.add_argument("--in_jsonl", default="", help="兼容旧参数名")
    ap.add_argument("--csv_in", default="papers.enriched.csv")
    ap.add_argument("--out_jsonl", default="./store/compute/gpu_compute.jsonl")
    ap.add_argument("--out_csv", default="./store/compute/gpu_compute.csv")
    ap.add_argument("--issues_out", default="./store/compute/gpu_compute_issues.csv")
    ap.add_argument("--max_pages", type=int, default=0, help="只扫描前N页（0=全部）")
    ap.add_argument("--max_coarse", type=int, default=20)
    ap.add_argument("--max_fine", type=int, default=12)
    ap.add_argument("--context_window", type=int, default=200, help="上下文窗口字符数")
    ap.add_argument("--keep_pages", action="store_true", default=False)
    ap.add_argument("--max_contexts_csv", type=int, default=12, help="CSV里最多写入的上下文条数")
    ap.add_argument("--context_char_limit", type=int, default=500, help="CSV单条上下文最长字符数")
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--resume_from", default="", help="Optional path to prior output jsonl")
    args = ap.parse_args()

    pages_path = args.pages_jsonl or args.in_jsonl
    if not pages_path:
        raise SystemExit("Missing --pages_jsonl/--in_jsonl")

    page_rows = load_jsonl(pages_path)
    meta_map = build_meta_index(load_csv(args.csv_in)) if os.path.exists(args.csv_in) else {}

    existing_idx: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        resume_path = args.resume_from or args.out_jsonl
        existing_idx = load_existing(resume_path)

    results = []
    issues: List[Dict[str, Any]] = []

    for obj in tqdm(page_rows, desc="Extract GPU compute", unit="paper"):
        obj = merge_meta(obj, meta_map)
        obj = ensure_ids(dict(obj))
        pid = obj.get("paper_id")
        source_hash = obj.get("source_hash")

        if args.resume and pid in existing_idx:
            old = existing_idx[pid]
            if old.get("source_hash") == source_hash and old.get("compute_status") == "ok":
                results.append(old)
                continue

        all_snips = []
        all_items = []
        fine_snips = []
        mems = []

        compute_status = "ok"
        compute_error = None

        try:
            pages = iter_page_texts(obj, max_pages=args.max_pages)
            if not pages:
                compute_status = "missing_md"
            else:
                for pno, text in pages:
                    snips = collect_coarse_snippets(text, max_snips=args.max_coarse)
                    for s in snips:
                        all_snips.append({
                            "page": pno,
                            "text": s["text"],
                            "hits": s["hits"],
                            "match_contexts": extract_match_contexts(
                                s["text"], window=args.context_window
                            ),
                        })
                        mems.extend(extract_mem_gb(s["text"]))
                        fine_hits = detect_fine_hits(s["text"])
                        if fine_hits:
                            fine_snips.append({
                                "page": pno,
                                "text": s["text"],
                                "hits": fine_hits,
                                "match_contexts": extract_match_contexts(
                                    s["text"], window=args.context_window
                                ),
                            })
                            all_items.extend(parse_gpu_from_snippet(s["text"]))
        except Exception as e:
            compute_status = "error"
            compute_error = str(e)

        merged = merge_gpu_items(all_items)
        gpu_arch = sorted({x["model"] for x in merged if x["model"] and x["model"] != "UNKNOWN"})
        gpu_counts = [x["count"] for x in merged if isinstance(x["count"], int)]
        gpu_mem_gb = [x["mem_gb"] for x in merged if isinstance(x["mem_gb"], (int, float))]
        if not gpu_mem_gb and mems:
            gpu_mem_gb = sorted({round(x, 1) for x in mems})
        gpu_detected = bool(merged) or any("gpu_keyword" in s.get("hits", []) for s in all_snips)

        if compute_status == "ok" and not all_snips:
            compute_status = "no_signal"

        record = {
            "paper_id": pid,
            "source_hash": source_hash,
            "title": obj.get("title"),
            "arxiv_id": obj.get("arxiv_id"),
            "pdf_path": obj.get("pdf_path"),
            "md_path": obj.get("md_path"),
            "gpu_detected": gpu_detected,
            "coarse_snippets": all_snips[:args.max_coarse],
            "fine_snippets": fine_snips[:args.max_fine],
            "gpu_hardware": merged,
            "gpu_arch": gpu_arch,
            "gpu_counts": gpu_counts,
            "gpu_mem_gb": gpu_mem_gb,
            "evidence_pages": sorted({x["page"] for x in all_snips if x["page"] is not None}),
            "compute_status": compute_status,
            "compute_error": compute_error,
        }
        if args.keep_pages:
            if obj.get("md_pages") is not None:
                record["md_pages"] = obj.get("md_pages")
            if obj.get("md_text") is not None:
                record["md_text"] = obj.get("md_text")
        results.append(record)

        if compute_status in {"missing_md", "error"}:
            issues.append({
                "paper_id": pid,
                "title": obj.get("title"),
                "compute_status": compute_status,
                "compute_error": compute_error,
            })

    write_jsonl(args.out_jsonl, results)

    summary_rows = []
    for r in results:
        context_samples = collect_context_strings(
            r.get("fine_snippets") or r.get("coarse_snippets") or [],
            max_items=args.max_contexts_csv,
            context_limit=args.context_char_limit,
        )
        summary_rows.append({
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "arxiv_id": r.get("arxiv_id"),
            "gpu_detected": r.get("gpu_detected"),
            "gpu_arch": " | ".join(r.get("gpu_arch") or []),
            "gpu_counts": " | ".join([str(x) for x in r.get("gpu_counts") or []]),
            "gpu_mem_gb": " | ".join([f"{x:.1f}" for x in r.get("gpu_mem_gb") or []]),
            "coarse_snippets": len(r.get("coarse_snippets") or []),
            "fine_snippets": len(r.get("fine_snippets") or []),
            "evidence_pages": " | ".join(map(str, r.get("evidence_pages") or [])),
            "context_samples": " || ".join(context_samples),
            "compute_status": r.get("compute_status"),
        })
    preferred = [
        "paper_id", "title", "arxiv_id",
        "gpu_detected", "gpu_arch", "gpu_counts", "gpu_mem_gb",
        "coarse_snippets", "fine_snippets", "evidence_pages",
        "context_samples",
        "compute_status",
    ]
    write_csv(args.out_csv, summary_rows, preferred_fields=preferred)

    if issues:
        write_csv(args.issues_out, issues, preferred_fields=["paper_id", "title", "compute_status", "compute_error"])

    print(f"Done. JSONL -> {args.out_jsonl}, CSV -> {args.out_csv}")
    if issues:
        print(f"Issues -> {args.issues_out}")

if __name__ == "__main__":
    main()
