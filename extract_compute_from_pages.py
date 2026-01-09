#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import csv
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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

MEM_RE = re.compile(r"\b(\d{2,3})\s*(GB|G)\b", re.IGNORECASE)

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

def collect_gpu_snippets(text: str, max_snips: int = 12) -> List[str]:
    """
    召回所有可能与 GPU 相关的文本片段：
    - 含 GPU 关键词
    - 含 GPU 型号
    """
    if not text:
        return []

    snips: List[str] = []

    # 1) 段落级召回
    paras = split_paragraphs(text)
    for p in paras:
        if GPU_KEYWORDS_RE.search(p) or GPU_MODEL_RE.search(p):
            snips.append(p[:1800])
            if len(snips) >= max_snips:
                return snips

    # 2) 行窗口召回（markdown 崩坏兜底）
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        if GPU_KEYWORDS_RE.search(l) or GPU_MODEL_RE.search(l):
            snips.append(line_window(lines, i)[:1800])
            if len(snips) >= max_snips:
                break

    # 去重
    out, seen = [], set()
    for s in snips:
        if s not in seen:
            seen.add(s)
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
            mem = int(mm.group(1))
        add(cnt, norm_model(model_raw), mem, "high")

    for m in MODEL_X_COUNT_RE.finditer(s):
        model_raw = m.group(1)
        cnt = int(m.group(3))
        mem = None
        mm = MEM_RE.search(model_raw)
        if mm:
            mem = int(mm.group(1))
        add(cnt, norm_model(model_raw), mem, "high")

    for m in COUNT_MODEL_GPUS_RE.finditer(s):
        cnt = int(m.group(1))
        model_raw = m.group(2)
        mem = None
        mm = MEM_RE.search(s[m.start():m.end()+40])
        if mm:
            mem = int(mm.group(1))
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
                mem = int(mm.group(1))
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

# ============================================================
# 4. 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="papers.pages.jsonl")
    ap.add_argument("--out_jsonl", default="gpu_compute.jsonl")
    ap.add_argument("--out_csv", default="gpu_compute.csv")
    ap.add_argument("--top_pages", type=int, default=4)
    args = ap.parse_args()

    results = []

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Extract GPU compute", unit="paper"):
        obj = json.loads(line)
        pages = obj.get("md_pages") or []

        all_snips = []
        all_items = []

        for ch in pages:
            pno = page_no(ch)
            text = ch.get("text") or ""
            snips = collect_gpu_snippets(text)
            for s in snips:
                all_snips.append({"page": pno, "text": s})
                all_items.extend(parse_gpu_from_snippet(s))

        merged = merge_gpu_items(all_items)

        results.append({
            "title": obj.get("title"),
            "arxiv_id": obj.get("arxiv_id"),
            "pdf_path": obj.get("pdf_path"),
            "gpu_reported": bool(merged),
            "gpu_hardware": merged,
            "evidence_pages": sorted({x["page"] for x in all_snips if x["page"] is not None}),
            "evidence_snippets": all_snips[:12],
        })

    # JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV（简化版，便于筛选）
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "title", "arxiv_id", "gpu_reported",
                "gpu_models", "gpu_counts", "gpu_mem_gb",
                "evidence_pages"
            ]
        )
        w.writeheader()
        for r in results:
            models = [x["model"] for x in r["gpu_hardware"]]
            counts = [str(x["count"]) for x in r["gpu_hardware"] if x["count"]]
            mems = [str(x["mem_gb"]) for x in r["gpu_hardware"] if x["mem_gb"]]
            w.writerow({
                "title": r["title"],
                "arxiv_id": r["arxiv_id"],
                "gpu_reported": r["gpu_reported"],
                "gpu_models": " | ".join(models),
                "gpu_counts": " | ".join(counts),
                "gpu_mem_gb": " | ".join(mems),
                "evidence_pages": " | ".join(map(str, r["evidence_pages"])),
            })

    print(f"Done. JSONL -> {args.out_jsonl}, CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()