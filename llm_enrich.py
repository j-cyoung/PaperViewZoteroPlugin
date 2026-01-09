#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Condition
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from paper_utils import (
    compute_paper_id,
    compute_source_hash,
    load_csv,
    load_jsonl,
    write_csv,
    write_jsonl,
)


DEFAULT_TRANSLATE_PROMPT = (
    "You are a professional translator for machine learning papers. "
    "Translate the abstract into Simplified Chinese. "
    "Keep technical terms, acronyms, and citations unchanged. "
    "Do not add explanations. Output only the translation.\n\n"
    "Title: {title}\n\nAbstract:\n{abstract}\n"
)

DEFAULT_COMPUTE_PROMPT = (
    "You are an expert at reading compute descriptions in research papers. "
    "Given the extracted context snippets, summarize the compute requirements. "
    "Use only the provided context; if unknown, use null or \"unknown\". "
    "Return ONLY a single-line JSON object with keys: "
    "gpu_models (array of strings), gpu_count (number or null), gpu_memory_gb (number or null), "
    "training_time (string or null), gpu_hours (number or null), "
    "tasks (array of strings), other_resources (array of strings), "
    "notes (string), confidence (low/medium/high), summary_zh (string, Chinese natural description).\n\n"
    "Title: {title}\n"
    "ArXiv: {arxiv_id}\n\n"
    "Compute Contexts:\n{contexts}\n"
)


@dataclass
class LLMUsage:
    prompt_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Dict[str, Any]


class LLMProvider:
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""

        usage_meta = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_meta.get("prompt_tokens") or 0),
            output_tokens=int(usage_meta.get("completion_tokens") or 0),
            total_tokens=int(usage_meta.get("total_tokens") or 0),
        )
        return LLMResponse(text=text, model=self.model, usage=usage, raw=data)


@dataclass
class TaskSpec:
    name: str
    prompt_template: str
    output_field: str
    status_field: str
    error_field: str
    usage_field: str
    model_field: str
    max_output_tokens: int = 1024
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None


class RateLimitController:
    def __init__(self, max_inflight: int, wait_s: int):
        self.default_max = max(1, max_inflight)
        self.max_inflight = self.default_max
        self.in_flight = 0
        self.pause_until = 0.0
        self.rate_limited = False
        self.wait_s = max(1, int(wait_s))
        self.waiting_logged = False
        self.cond = Condition()

    def acquire(self) -> None:
        with self.cond:
            while True:
                now = time.time()
                if self.pause_until and now < self.pause_until:
                    wait_s = self.pause_until - now
                    if not self.waiting_logged:
                        self.waiting_logged = True
                        tqdm.write(
                            f"[rate-limit] waiting {int(wait_s)}s, concurrency={self.max_inflight}"
                        )
                    self.cond.wait(timeout=wait_s)
                    continue
                if self.pause_until and now >= self.pause_until:
                    self.pause_until = 0.0
                    self.waiting_logged = False
                if self.in_flight < self.max_inflight:
                    self.in_flight += 1
                    return
                self.cond.wait()

    def release(self) -> None:
        with self.cond:
            self.in_flight = max(0, self.in_flight - 1)
            self.cond.notify_all()

    def on_rate_limit(self) -> None:
        with self.cond:
            self.rate_limited = True
            self.max_inflight = 1
            now = time.time()
            self.pause_until = max(self.pause_until, now + self.wait_s)
            self.waiting_logged = False
            self.cond.notify_all()
            tqdm.write(
                f"[rate-limit] 429 detected, pause {self.wait_s}s, "
                f"concurrency set to 1"
            )

    def on_success(self) -> None:
        with self.cond:
            if self.rate_limited and time.time() >= self.pause_until:
                self.rate_limited = False
                self.max_inflight = self.default_max
                self.cond.notify_all()
                tqdm.write(
                    f"[rate-limit] recovered, concurrency={self.default_max}"
                )


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


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    t = strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        return None


def format_contexts(
    compute_row: Dict[str, Any],
    max_items: int,
    context_char_limit: int,
) -> str:
    snippets = compute_row.get("fine_snippets") or compute_row.get("coarse_snippets") or []
    lines: List[str] = []
    for snip in snippets:
        page = snip.get("page")
        page_tag = f"p{page}" if page is not None else "p?"
        match_ctxs = snip.get("match_contexts") or []
        if match_ctxs:
            for ctx in match_ctxs:
                text = ctx.get("context") or ""
                if len(text) > context_char_limit:
                    text = text[:context_char_limit] + "..."
                lines.append(f"[{page_tag}] {ctx.get('type')} {ctx.get('match')}: {text}")
                if len(lines) >= max_items:
                    return "\n".join(lines)
        else:
            text = snip.get("text") or ""
            if len(text) > context_char_limit:
                text = text[:context_char_limit] + "..."
            lines.append(f"[{page_tag}] {text}")
            if len(lines) >= max_items:
                return "\n".join(lines)
    return "\n".join(lines)


def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_default_path(primary: str, fallback: str) -> str:
    if primary and os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    return primary or fallback


def build_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    if args.tasks_config:
        raw = load_prompt_from_file(args.tasks_config)
        data = json.loads(raw)
        tasks: List[TaskSpec] = []
        for t in data:
            tasks.append(TaskSpec(
                name=t["name"],
                prompt_template=t["prompt_template"],
                output_field=t.get("output_field", t["name"]),
                status_field=t.get("status_field", f"{t['name']}_status"),
                error_field=t.get("error_field", f"{t['name']}_error"),
                usage_field=t.get("usage_field", f"{t['name']}_usage"),
                model_field=t.get("model_field", f"{t['name']}_model"),
                max_output_tokens=int(t.get("max_output_tokens", 1024)),
                temperature=float(t.get("temperature", 0.0)),
                system_prompt=t.get("system_prompt"),
                response_format=t.get("response_format"),
            ))
        return tasks

    translate_prompt = DEFAULT_TRANSLATE_PROMPT
    if args.translate_prompt:
        translate_prompt = args.translate_prompt
    if args.translate_prompt_file:
        translate_prompt = load_prompt_from_file(args.translate_prompt_file)

    compute_prompt = DEFAULT_COMPUTE_PROMPT
    if args.compute_prompt:
        compute_prompt = args.compute_prompt
    if args.compute_prompt_file:
        compute_prompt = load_prompt_from_file(args.compute_prompt_file)

    tasks: List[TaskSpec] = []
    if not args.skip_translate:
        tasks.append(TaskSpec(
            name="translate",
            prompt_template=translate_prompt,
            output_field="abstract_zh",
            status_field="translate_status",
            error_field="translate_error",
            usage_field="translate_usage",
            model_field="translate_model",
            max_output_tokens=args.translate_max_tokens,
            temperature=args.translate_temperature,
        ))
    if not args.skip_compute:
        tasks.append(TaskSpec(
            name="compute",
            prompt_template=compute_prompt,
            output_field="compute_structured",
            status_field="compute_status",
            error_field="compute_error",
            usage_field="compute_usage",
            model_field="compute_model",
            max_output_tokens=args.compute_max_tokens,
            temperature=args.compute_temperature,
            response_format={"type": "json_object"} if args.compute_json_mode else None,
        ))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", default="./store/enrich/papers.enriched.csv")
    ap.add_argument("--compute_jsonl", default="./store/compute/gpu_compute.jsonl")
    ap.add_argument("--out_jsonl", default="papers.llm.jsonl")
    ap.add_argument("--out_csv", default="papers.llm.csv")
    ap.add_argument("--issues_out", default="papers.llm_issues.csv")
    ap.add_argument("--base_output_dir", default="./store/llm", help="输出根目录（相对路径会拼接到该目录）")

    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--base_url", default="https://api.siliconflow.cn/v1")
    ap.add_argument("--api_key", default="")
    ap.add_argument("--tasks_config", default="", help="JSON file defining custom tasks")

    ap.add_argument("--translate_prompt", default="")
    ap.add_argument("--translate_prompt_file", default="")
    ap.add_argument("--compute_prompt", default="")
    ap.add_argument("--compute_prompt_file", default="")

    ap.add_argument("--skip_translate", action="store_true", default=False)
    ap.add_argument("--skip_compute", action="store_true", default=False)
    ap.add_argument("--translate_temperature", type=float, default=0.0)
    ap.add_argument("--compute_temperature", type=float, default=0.0)
    ap.add_argument("--translate_max_tokens", type=int, default=8192)
    ap.add_argument("--compute_max_tokens", type=int, default=8192)
    ap.add_argument("--compute_json_mode", action="store_true", default=True)
    ap.add_argument("--no_compute_json_mode", action="store_false", dest="compute_json_mode")

    ap.add_argument("--max_context_items", type=int, default=100)
    ap.add_argument("--context_char_limit", type=int, default=8192)

    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--resume_from", default="", help="Optional prior output jsonl")
    ap.add_argument("--top_k", type=int, default=0, help="只处理前K条（0=全部）")
    ap.add_argument("--concurrency", type=int, default=1, help="并发请求数量")
    ap.add_argument("--retry_on_429", action="store_true", default=False, help="遇到429时持续重试")
    ap.add_argument("--retry_wait_s", type=int, default=300, help="429重试等待秒数")

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
        args.out_jsonl = apply_base_dir(args.out_jsonl, args.base_output_dir)
        args.out_csv = apply_base_dir(args.out_csv, args.base_output_dir)
        args.issues_out = apply_base_dir(args.issues_out, args.base_output_dir)

    csv_path = resolve_default_path(args.csv_in, "./store/enrich/papers.enriched.csv")
    compute_path = resolve_default_path(args.compute_jsonl, "./store/compute/gpu_compute.jsonl")

    meta_rows = load_csv(csv_path) if os.path.exists(csv_path) else []
    compute_rows = load_jsonl(compute_path) if os.path.exists(compute_path) else []
    meta_map = build_meta_index(meta_rows)
    compute_map = build_meta_index(compute_rows)

    api_key = (
        args.api_key
        or os.environ.get("SILICONFLOW_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        raise SystemExit("Missing SILICONFLOW_API_KEY/OPENAI_API_KEY or --api_key")

    provider = OpenAICompatProvider(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
    )
    tasks = build_tasks(args)

    existing_idx: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        resume_path = args.resume_from or args.out_jsonl
        existing_idx = {r.get("paper_id"): r for r in load_jsonl(resume_path)}

    rows_iter = meta_rows
    if args.top_k and args.top_k > 0:
        rows_iter = meta_rows[:args.top_k]

    def process_one(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        row = ensure_ids(dict(row))
        pid = row.get("paper_id")
        source_hash = row.get("source_hash")

        compute_row = compute_map.get(pid) or compute_map.get(row.get("arxiv_id")) or compute_map.get(row.get("title"))
        if compute_row:
            row = merge_meta(row, {pid: compute_row})

        out = dict(row)
        out["paper_id"] = pid
        out["source_hash"] = source_hash
        out.setdefault("task_logs", [])

        existing = existing_idx.get(pid) if args.resume else None
        if existing and existing.get("source_hash") == source_hash:
            out.update(dict(existing))

        contexts = ""
        if compute_row:
            contexts = format_contexts(
                compute_row,
                max_items=args.max_context_items,
                context_char_limit=args.context_char_limit,
            )

        for task in tasks:
            if existing and existing.get(task.status_field) == "ok" and existing.get("source_hash") == source_hash:
                continue

            prompt_vars = {
                "title": out.get("title") or "",
                "abstract": out.get("abstract") or "",
                "abstract_zh": out.get("abstract_zh") or "",
                "arxiv_id": out.get("arxiv_id") or "",
                "contexts": contexts,
                "gpu_hardware": json.dumps(out.get("gpu_hardware") or [], ensure_ascii=False),
            }
            prompt = task.prompt_template.format(**prompt_vars)

            start_ts = time.time()
            retries = 0
            try:
                while True:
                    rate_state.acquire()
                    try:
                        resp = provider.generate(
                            prompt=prompt,
                            system=task.system_prompt,
                            temperature=task.temperature,
                            max_output_tokens=task.max_output_tokens,
                            response_format=task.response_format,
                        )
                        rate_state.on_success()
                        break
                    except requests.HTTPError as e:
                        status = getattr(e.response, "status_code", None)
                        if status == 429 and args.retry_on_429:
                            retries += 1
                            rate_state.on_rate_limit()
                            continue
                        raise
                    finally:
                        rate_state.release()
                elapsed_ms = int((time.time() - start_ts) * 1000)
                out[task.model_field] = resp.model
                out[task.usage_field] = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                out[f"{task.name}_elapsed_ms"] = elapsed_ms

                if task.name == "compute":
                    parsed = parse_json_from_text(resp.text)
                    if parsed is not None:
                        out[task.output_field] = parsed
                        out["compute_summary_zh"] = parsed.get("summary_zh")
                        out[task.status_field] = "ok"
                    else:
                        out[task.output_field] = None
                        out["compute_summary_zh"] = None
                        out["compute_summary_raw"] = resp.text.strip()
                        out[task.status_field] = "parse_error"
                else:
                    out[task.output_field] = resp.text.strip()
                    out[task.status_field] = "ok"

                out[task.error_field] = None
                out["task_logs"].append({
                    "task": task.name,
                    "model": resp.model,
                    "status": out.get(task.status_field),
                    "elapsed_ms": elapsed_ms,
                    "retries": retries,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.total_tokens,
                    "started_at": start_ts,
                })
            except Exception as e:
                elapsed_ms = int((time.time() - start_ts) * 1000)
                out[task.status_field] = "error"
                out[task.error_field] = str(e)
                out[f"{task.name}_elapsed_ms"] = elapsed_ms
                out["task_logs"].append({
                    "task": task.name,
                    "model": out.get(task.model_field),
                    "status": "error",
                    "elapsed_ms": elapsed_ms,
                    "retries": retries,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "started_at": start_ts,
                    "error": str(e),
                })
        return {"idx": idx, "data": out}

    results: List[Dict[str, Any]] = [None] * len(rows_iter)
    rate_state = RateLimitController(args.concurrency, args.retry_wait_s)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(process_one, i, row) for i, row in enumerate(rows_iter)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM enrich", unit="paper"):
            res = fut.result()
            results[res["idx"]] = res["data"]

    issues: List[Dict[str, Any]] = []
    for out in results:
        if any(out.get(t.status_field) not in {None, "ok"} for t in tasks):
            reasons = []
            for t in tasks:
                if out.get(t.status_field) not in {None, "ok"}:
                    msg = out.get(t.error_field) or out.get(f"{t.name}_error") or ""
                    if msg:
                        reasons.append(f"{t.name}: {msg}")
                    else:
                        reasons.append(f"{t.name}: {out.get(t.status_field)}")
            issues.append({
                "paper_id": out.get("paper_id"),
                "title": out.get("title"),
                "status": "error",
                "reason": " | ".join(reasons),
            })

    write_jsonl(args.out_jsonl, results)

    summary_rows = []
    for r in results:
        compute_structured = r.get("compute_structured") if isinstance(r.get("compute_structured"), dict) else {}
        summary_rows.append({
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "arxiv_id": r.get("arxiv_id"),
            "abstract_zh": r.get("abstract_zh"),
            "compute_summary_zh": r.get("compute_summary_zh"),
            "compute_gpu_models": " | ".join(compute_structured.get("gpu_models") or []),
            "compute_gpu_count": compute_structured.get("gpu_count"),
            "compute_gpu_memory_gb": compute_structured.get("gpu_memory_gb"),
            "compute_training_time": compute_structured.get("training_time"),
            "compute_gpu_hours": compute_structured.get("gpu_hours"),
            "compute_confidence": compute_structured.get("confidence"),
            "translate_elapsed_ms": r.get("translate_elapsed_ms"),
            "compute_elapsed_ms": r.get("compute_elapsed_ms"),
            "translate_status": r.get("translate_status"),
            "compute_status": r.get("compute_status"),
            "translate_prompt_tokens": (r.get("translate_usage") or {}).get("prompt_tokens"),
            "translate_output_tokens": (r.get("translate_usage") or {}).get("output_tokens"),
            "compute_prompt_tokens": (r.get("compute_usage") or {}).get("prompt_tokens"),
            "compute_output_tokens": (r.get("compute_usage") or {}).get("output_tokens"),
            "translate_model": r.get("translate_model"),
            "compute_model": r.get("compute_model"),
        })

    preferred = [
        "paper_id", "title", "arxiv_id",
        "abstract_zh", "compute_summary_zh",
        "compute_gpu_models", "compute_gpu_count", "compute_gpu_memory_gb",
        "compute_training_time", "compute_gpu_hours", "compute_confidence",
        "translate_elapsed_ms", "compute_elapsed_ms",
        "translate_status", "compute_status",
        "translate_prompt_tokens", "translate_output_tokens",
        "compute_prompt_tokens", "compute_output_tokens",
        "translate_model", "compute_model",
    ]
    write_csv(args.out_csv, summary_rows, preferred_fields=preferred)

    if issues:
        write_csv(args.issues_out, issues, preferred_fields=["paper_id", "title", "status", "reason"])

    print(f"Done. JSONL -> {args.out_jsonl}, CSV -> {args.out_csv}")
    if issues:
        print(f"Issues -> {args.issues_out}")


if __name__ == "__main__":
    main()
