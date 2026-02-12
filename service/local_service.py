#!/usr/bin/env python3
# Minimal local service for Zotero demo

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
import re
import threading
import time
import requests
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_OCR_CONCURRENCY = 4
DEFAULT_RUNTIME_TEST_TIMEOUT_S = 20
DEFAULT_REMOTE_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_REMOTE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
OCR_CACHE_LOCK = threading.Lock()


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "PaperViewDemo/0.1"

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status, html):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _result_url(self):
        host = "localhost"
        port = self.server.server_address[1]
        return f"http://{host}:{port}/result/demo"

    def _base_dir(self):
        return Path(__file__).resolve().parent

    def _store_dir(self):
        return self._base_dir() / "store" / "zotero"

    def _ocr_dir(self):
        return self._store_dir() / "ocr"

    def _query_dir(self):
        return self._store_dir() / "query"

    def _items_jsonl(self):
        return self._store_dir() / "items.jsonl"

    def _items_csv(self):
        return self._store_dir() / "items.csv"

    def _pages_jsonl(self):
        return self._ocr_dir() / "papers.pages.jsonl"

    def _write_ingest(self, items):
        store_dir = self._store_dir()
        store_dir.mkdir(parents=True, exist_ok=True)
        out_file = self._items_jsonl()
        now = datetime.now(timezone.utc).isoformat()
        existing_by_key = {}
        for row in self._load_jsonl(out_file):
            if not isinstance(row, dict):
                continue
            key = row.get("item_key") or row.get("paper_id")
            if not key:
                continue
            existing_by_key[key] = row

        incoming = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            item.setdefault("ingest_time", now)
            key = item.get("item_key") or item.get("paper_id")
            if not key:
                continue
            incoming += 1
            existing_by_key[key] = item
        if existing_by_key:
            lines = [json.dumps(v, ensure_ascii=False) for v in existing_by_key.values()]
            with out_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return incoming

    def _run_cmd(self, cmd, cwd=None):
        cmd_display = " ".join(cmd)
        print(f"[cmd] {cmd_display}")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, cwd=cwd, check=True, env=env)

    def _pick_positive_int(self, *values, default=1):
        for val in values:
            if val is None:
                continue
            try:
                n = int(float(val))
            except Exception:
                continue
            if n > 0:
                return n
        return default

    def _pick_non_empty_str(self, *values, default=""):
        for val in values:
            if val is None:
                continue
            text = str(val).strip()
            if text:
                return text
        return default

    def _load_runtime_config(self):
        config_path = LLM_CONFIG_PATH or os.environ.get("PAPERVIEW_LLM_CONFIG", "")
        if not config_path:
            return {}
        try:
            return json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _resolve_ocr_concurrency(self, payload=None):
        cfg = self._load_runtime_config()
        req = payload if isinstance(payload, dict) else {}
        return self._pick_positive_int(
            req.get("ocr_concurrency"),
            cfg.get("ocr_concurrency"),
            cfg.get("concurrency"),
            os.environ.get("PAPERVIEW_OCR_CONCURRENCY"),
            default=DEFAULT_OCR_CONCURRENCY,
        )

    def _resolve_api_key_with_source(self, payload=None):
        cfg = self._load_runtime_config()
        req = payload if isinstance(payload, dict) else {}

        req_key = self._pick_non_empty_str(req.get("api_key"), default="")
        if req_key:
            return req_key, "request"

        cfg_key = self._pick_non_empty_str(cfg.get("api_key"), default="")
        if cfg_key:
            return cfg_key, "prefs"

        env_sf = self._pick_non_empty_str(os.environ.get("SILICONFLOW_API_KEY"), default="")
        if env_sf:
            return env_sf, "env"

        env_openai = self._pick_non_empty_str(os.environ.get("OPENAI_API_KEY"), default="")
        if env_openai:
            return env_openai, "env"

        return "", "none"

    def _resolve_api_key(self, payload=None):
        key, _source = self._resolve_api_key_with_source(payload)
        return key

    def _resolve_runtime_base_url(self, payload=None):
        cfg = self._load_runtime_config()
        req = payload if isinstance(payload, dict) else {}
        value = self._pick_non_empty_str(
            req.get("base_url"),
            cfg.get("base_url"),
            os.environ.get("PAPERVIEW_LLM_BASE_URL"),
            default=DEFAULT_REMOTE_BASE_URL,
        )
        return value.rstrip("/")

    def _resolve_runtime_model(self, payload=None):
        cfg = self._load_runtime_config()
        req = payload if isinstance(payload, dict) else {}
        return self._pick_non_empty_str(
            req.get("model"),
            cfg.get("model"),
            os.environ.get("PAPERVIEW_LLM_MODEL"),
            default=DEFAULT_REMOTE_MODEL,
        )

    def _resolve_runtime_timeout_s(self, payload=None):
        req = payload if isinstance(payload, dict) else {}
        timeout = self._pick_positive_int(
            req.get("timeout_s"),
            default=DEFAULT_RUNTIME_TEST_TIMEOUT_S,
        )
        return max(3, min(timeout, 120))

    def _classify_exception(self, err, stage=""):
        text = str(err or "").strip() or repr(err)
        lower = text.lower()
        stage_l = str(stage or "").lower()
        code = "internal_error"

        if "remote_api_timeout" in lower:
            code = "remote_api_timeout"
        elif "remote_api_unreachable" in lower:
            code = "remote_api_unreachable"
        elif "remote_api_auth" in lower:
            code = "remote_api_auth"
        elif "remote_api_http_error" in lower:
            code = "remote_api_http_error"
        elif "missing api key" in lower:
            code = "api_key_missing"
        elif "items.jsonl not found" in lower:
            code = "ingest_required"
        elif "no matching items" in lower:
            code = "selection_empty"
        elif "pdf_to_md_pymupdf4llm failed" in lower:
            code = "ocr_start_failed"
        elif "timed out" in lower or "timeout" in lower:
            code = "remote_api_timeout"
        elif "connection refused" in lower or "failed to establish a new connection" in lower:
            code = "remote_api_unreachable"
        elif "name or service not known" in lower or "temporary failure in name resolution" in lower:
            code = "remote_api_unreachable"
        elif "max retries exceeded" in lower and ("http" in lower or "https" in lower):
            code = "remote_api_unreachable"
        elif "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower:
            code = "remote_api_auth"
        elif "query_papers failed with code" in lower:
            code = "query_pipeline_failed"
        elif stage_l == "ocr":
            code = "ocr_pipeline_failed"
        elif stage_l == "query":
            code = "query_pipeline_failed"
        return code, text

    def _row_item_key(self, row):
        return (
            row.get("paper_id")
            or row.get("zotero_item_key")
            or row.get("item_key")
        )

    def _selected_ocr_stats(self, selected_keys):
        pages_path = self._pages_jsonl()
        if not pages_path.exists() or not selected_keys:
            return {"processed": 0, "ok": 0, "error": 0}
        processed = set()
        ok = set()
        error = set()
        try:
            with pages_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    pid = self._row_item_key(row)
                    if pid not in selected_keys:
                        continue
                    processed.add(pid)
                    md_status = str(row.get("md_status") or "").lower()
                    has_text = bool(row.get("md_pages")) or bool(str(row.get("md_text") or "").strip())
                    if md_status == "ok" and has_text:
                        ok.add(pid)
                    elif md_status in {"error", "missing_pdf"} or not has_text:
                        error.add(pid)
        except Exception:
            return {"processed": 0, "ok": 0, "error": 0}
        return {"processed": len(processed), "ok": len(ok), "error": len(error)}

    def _select_ingested_items(self, selected_item_keys):
        rows = self._load_jsonl(self._items_jsonl())
        if not selected_item_keys:
            return rows
        key_set = {k for k in selected_item_keys if k}
        if not key_set:
            return []
        selected = []
        for row in rows:
            key = row.get("item_key") or row.get("paper_id")
            if key in key_set:
                selected.append(row)
        return selected

    def _merge_pages_cache(self, new_rows):
        pages_path = self._pages_jsonl()
        existing_rows = self._load_jsonl(pages_path)
        by_key = {}
        ordered_keys = []
        extras = []

        def add_row(row):
            if not isinstance(row, dict):
                return
            key = self._row_item_key(row)
            if key:
                if key not in by_key:
                    ordered_keys.append(key)
                by_key[key] = row
            else:
                extras.append(row)

        for row in existing_rows:
            add_row(row)
        for row in new_rows:
            add_row(row)

        merged = [by_key[k] for k in ordered_keys] + extras
        self._write_jsonl(pages_path, merged)

    def _ensure_ocr_cache(
        self,
        progress_path=None,
        status_cb=None,
        ocr_concurrency=DEFAULT_OCR_CONCURRENCY,
        selected_item_keys=None,
    ):
        with OCR_CACHE_LOCK:
            base_dir = self._base_dir()
            ocr_dir = self._ocr_dir()
            ocr_dir.mkdir(parents=True, exist_ok=True)
            selected_rows = self._select_ingested_items(selected_item_keys)
            if not selected_rows:
                if status_cb:
                    status_cb(0, 0)
                return

            run_id = uuid.uuid4().hex[:8]
            selected_items_jsonl = ocr_dir / f"items.selected.{run_id}.jsonl"
            selected_items_csv = ocr_dir / f"items.selected.{run_id}.csv"
            selected_pages_jsonl = ocr_dir / f"papers.pages.selected.{run_id}.jsonl"

            try:
                self._write_jsonl(selected_items_jsonl, selected_rows)
                self._run_cmd(
                    [
                        sys.executable,
                        "zotero_items_to_csv.py",
                        "--jsonl",
                        str(selected_items_jsonl),
                        "--csv_out",
                        str(selected_items_csv),
                    ],
                    cwd=base_dir,
                )

                cmd = [
                    sys.executable,
                    "pdf_to_md_pymupdf4llm.py",
                    "--csv_in",
                    str(selected_items_csv),
                    "--base_output_dir",
                    str(ocr_dir),
                    "--out_jsonl",
                    str(selected_pages_jsonl),
                    "--resume",
                    "--resume_from",
                    str(self._pages_jsonl()),
                    "--concurrency",
                    str(max(1, int(ocr_concurrency))),
                ]
                if progress_path:
                    cmd += ["--progress_path", str(progress_path)]

                print(
                    f"[ocr] start ensure cache selected={len(selected_rows)} "
                    f"concurrency={ocr_concurrency} progress_path={progress_path}"
                )
                print(f"[cmd] {' '.join(cmd)}")

                if not status_cb:
                    self._run_cmd(cmd, cwd=base_dir)
                else:
                    proc = subprocess.Popen(cmd, cwd=base_dir)
                    out_jsonl = selected_pages_jsonl
                    last_done = -1
                    last_total = -1
                    last_tick = 0.0
                    while proc.poll() is None:
                        done = None
                        total = None
                        if progress_path:
                            try:
                                progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                                done = int(progress.get("done") or 0)
                                total = int(progress.get("total") or 0)
                            except Exception:
                                done = None
                                total = None
                        if done is None and out_jsonl.exists():
                            with out_jsonl.open("r", encoding="utf-8") as f:
                                done = sum(1 for line in f if line.strip())
                        if done is not None and (done != last_done or total != last_total):
                            status_cb(done, total)
                            now = time.time()
                            if now - last_tick >= 10:
                                print(f"[ocr] progress done={done}, total={total}")
                                last_tick = now
                            last_done = done
                            last_total = total
                        time.sleep(1)
                    if proc.returncode != 0:
                        raise RuntimeError(f"pdf_to_md_pymupdf4llm failed with code {proc.returncode}")

                new_rows = self._load_jsonl(selected_pages_jsonl)
                self._merge_pages_cache(new_rows)

                if status_cb:
                    if progress_path:
                        try:
                            progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                            done = int(progress.get("done") or 0)
                            total = int(progress.get("total") or 0)
                            status_cb(done, total)
                            return
                        except Exception:
                            pass
                    status_cb(len(new_rows), len(selected_rows))
            finally:
                for path in (selected_items_jsonl, selected_items_csv, selected_pages_jsonl):
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        pass

    def _load_jsonl(self, path):
        rows = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def _write_jsonl(self, path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _filter_pages(self, item_keys, out_path):
        pages = self._load_jsonl(self._pages_jsonl())
        key_set = set(item_keys)
        selected = []
        for row in pages:
            paper_id = self._row_item_key(row)
            if paper_id in key_set:
                selected.append(row)
        self._write_jsonl(out_path, selected)
        return len(selected)

    def _select_items_by_keys(self, items, item_keys):
        key_set = {k for k in item_keys if k}
        if not key_set:
            return []
        return [item for item in items if item.get("item_key") in key_set]

    def _count_selected_ocr_done(self, selected_keys):
        return self._selected_ocr_stats(selected_keys).get("processed", 0)

    def _query_job_dir(self, job_id):
        job_dir = self._query_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _status_path(self, job_dir):
        return job_dir / "status.json"

    def _write_status(
        self,
        job_dir,
        stage,
        done=None,
        total=None,
        message=None,
        error_code=None,
        extra=None,
    ):
        payload = {
            "stage": stage,
            "done": done,
            "total": total,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error_code:
            payload["error_code"] = error_code
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        self._status_path(job_dir).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _read_status(self, job_id):
        job_dir = self._query_dir() / job_id
        path = self._status_path(job_dir)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _history_entries(self):
        query_dir = self._query_dir()
        if not query_dir.exists():
            return []
        entries = []
        for child in query_dir.iterdir():
            if not child.is_dir():
                continue
            entry = {"job_id": child.name}
            meta_path = child / "query.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                entry["created_at"] = meta.get("created_at")
                entry["query"] = meta.get("query")
                entry["item_count"] = meta.get("item_count")
            entries.append(entry)

        def sort_key(item):
            ts = item.get("created_at") or ""
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        entries.sort(key=sort_key, reverse=True)
        return entries

    def _resolve_history_job_dir(self, job_id):
        job = str(job_id or "").strip()
        if not job or not re.fullmatch(r"[A-Za-z0-9._-]+", job):
            return None
        root = self._query_dir().resolve()
        target = (self._query_dir() / job).resolve()
        if not str(target).startswith(str(root) + os.sep):
            return None
        return target

    def _delete_history_job(self, job_id):
        target = self._resolve_history_job_dir(job_id)
        if not target:
            return False
        if not target.exists() or not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def _clear_history(self):
        query_dir = self._query_dir()
        if not query_dir.exists():
            return 0
        removed = 0
        for child in list(query_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except Exception as e:
                print(f"[history] clear skip path={child} err={e}")
        return removed

    def _write_query_snapshot(self, job_dir, payload, items):
        sections = payload.get("sections") or ""
        snapshot = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": payload.get("query") or "",
            "sections": sections,
            "query_mode": payload.get("query_mode") or "single",
            "item_keys": payload.get("item_keys") or [],
            "item_count": len(items),
        }
        (job_dir / "query.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_jsonl(job_dir / "items.jsonl", items)

    def _run_query_pipeline(self, payload, job_id):
        item_keys = payload.get("item_keys") or []
        query_text = payload.get("query") or ""
        query_mode = (payload.get("query_mode") or "single").strip().lower()
        if query_mode not in {"single", "merge"}:
            query_mode = "single"
        sections = payload.get("sections")
        if (not sections or str(sections).strip() == "") and query_text:
            match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", str(query_text))
            if match:
                sections = match.group(1).strip()
                query_text = (match.group(2) or "").strip()
                payload["query"] = query_text
        if isinstance(sections, list):
            sections = ",".join([str(s).strip() for s in sections if str(s).strip()])
        sections = str(sections).strip() if sections is not None else ""
        tokens = [t.strip() for t in sections.split(",") if t.strip()]
        if not tokens:
            sections = "full_text"
        else:
            lowered = [t.lower() for t in tokens]
            if any(t in {"full_text", "full", "all", "*"} for t in lowered) or any(
                t in {"全文", "全部"} for t in tokens
            ):
                sections = "full_text"
            else:
                sections = ",".join(tokens)
        payload["sections"] = sections
        payload["query_mode"] = query_mode
        if not item_keys or not query_text:
            raise ValueError("missing item_keys or query text")

        job_dir = self._query_job_dir(job_id)
        self._write_status(job_dir, "queued", 0, len(item_keys) or 0)

        items_path = self._items_jsonl()
        if not items_path.exists():
            raise ValueError("items.jsonl not found; run ingest first")
        items = self._load_jsonl(items_path)
        selected = self._select_items_by_keys(items, item_keys)
        if not selected:
            raise ValueError("no matching items in items.jsonl")

        api_key, api_key_source = self._resolve_api_key_with_source(payload)
        print(f"[runtime] query api_key_source={api_key_source}")
        if not api_key:
            raise ValueError(
                "missing API key (set PaperView API Key in Settings, then Save)"
            )

        self._write_status(job_dir, "prepare", 0, len(selected))
        self._write_query_snapshot(job_dir, payload, selected)

        # ensure OCR cache (incremental)
        ocr_total = len(selected)
        self._write_status(job_dir, "ocr", 0, ocr_total)
        progress_path = job_dir / "ocr.progress.json"
        selected_keys = {item.get("item_key") for item in selected if item.get("item_key")}
        ocr_concurrency = self._resolve_ocr_concurrency(payload)
        print(
            f"[ocr-job] query pipeline job={job_id} selected={len(selected)} "
            f"ocr_concurrency={ocr_concurrency}"
        )

        def ocr_update(_done, _total):
            stats = self._selected_ocr_stats(selected_keys)
            message = None
            if stats["error"] > 0:
                message = f"ocr error={stats['error']}, ok={stats['ok']}"
            self._write_status(job_dir, "ocr", stats["processed"], ocr_total, message)

        self._ensure_ocr_cache(
            progress_path=progress_path,
            status_cb=ocr_update,
            ocr_concurrency=ocr_concurrency,
            selected_item_keys=selected_keys,
        )
        final_stats = self._selected_ocr_stats(selected_keys)
        final_msg = None
        if final_stats["error"] > 0:
            final_msg = f"ocr error={final_stats['error']}, ok={final_stats['ok']}"
        self._write_status(job_dir, "ocr", final_stats["processed"], ocr_total, final_msg)

        # filter OCR pages for selected items
        filtered_pages = job_dir / "pages.selected.jsonl"
        self._filter_pages(item_keys, filtered_pages)

        # run query on selected pages with progress
        self._write_status(job_dir, "query", 0, len(selected))
        out_jsonl = job_dir / "papers.query.jsonl"
        progress_path = job_dir / "progress.json"
        cmd = [
            sys.executable,
            "query_papers.py",
            "--pages_jsonl",
            str(filtered_pages),
            "--base_output_dir",
            str(job_dir),
            "--question",
            query_text,
            "--sections",
            sections,
            "--progress_path",
            str(progress_path),
            "--query_mode",
            query_mode,
        ]
        if LLM_CONFIG_PATH:
            cmd += ["--config", LLM_CONFIG_PATH]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["SILICONFLOW_API_KEY"] = api_key
        env["OPENAI_API_KEY"] = api_key
        proc = subprocess.Popen(cmd, cwd=self._base_dir(), env=env)
        while proc.poll() is None:
            done = 0
            total = len(selected)
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    done = int(progress.get("done") or 0)
                    total = int(progress.get("total") or total)
                except Exception:
                    done = 0
            elif out_jsonl.exists():
                with out_jsonl.open("r", encoding="utf-8") as f:
                    done = sum(1 for line in f if line.strip())
            self._write_status(job_dir, "query", done, total)
            time.sleep(1)
        if proc.returncode != 0:
            probe = self._runtime_check(
                {
                    "api_key": api_key,
                    "base_url": self._resolve_runtime_base_url(payload),
                    "model": self._resolve_runtime_model(payload),
                    "check_remote": True,
                    "timeout_s": 8,
                }
            )
            if isinstance(probe, dict) and not probe.get("ok"):
                probe_code = str(probe.get("code") or "query_pipeline_failed").strip()
                probe_error = str(probe.get("error") or "").strip()
                if probe_error:
                    raise RuntimeError(f"{probe_code}: {probe_error}")
                raise RuntimeError(probe_code)
            raise RuntimeError(f"query_papers failed with code {proc.returncode}")
        done = 0
        total = len(selected)
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                done = int(progress.get("done") or done)
                total = int(progress.get("total") or total)
            except Exception:
                done = 0
        elif out_jsonl.exists():
            with out_jsonl.open("r", encoding="utf-8") as f:
                done = sum(1 for line in f if line.strip())
        self._write_status(job_dir, "done", done, total)
        return job_id

    def _run_ocr_pipeline(self, payload, job_id):
        item_keys = payload.get("item_keys") or []
        if not item_keys:
            raise ValueError("missing item_keys")
        job_dir = self._query_job_dir(job_id)
        self._write_status(job_dir, "queued", 0, len(item_keys) or 0)

        items_path = self._items_jsonl()
        if not items_path.exists():
            raise ValueError("items.jsonl not found; run ingest first")
        items = self._load_jsonl(items_path)
        selected = self._select_items_by_keys(items, item_keys)
        if not selected:
            raise ValueError("no matching items in items.jsonl")

        ocr_total = len(selected)
        self._write_status(job_dir, "ocr", 0, ocr_total)
        progress_path = job_dir / "ocr.progress.json"
        selected_keys = {item.get("item_key") for item in selected if item.get("item_key")}
        ocr_concurrency = self._resolve_ocr_concurrency(payload)
        print(
            f"[ocr-job] ocr pipeline job={job_id} selected={len(selected)} "
            f"ocr_concurrency={ocr_concurrency}"
        )

        def ocr_update(_done, _total):
            stats = self._selected_ocr_stats(selected_keys)
            message = None
            if stats["error"] > 0:
                message = f"ocr error={stats['error']}, ok={stats['ok']}"
            self._write_status(job_dir, "ocr", stats["processed"], ocr_total, message)

        self._ensure_ocr_cache(
            progress_path=progress_path,
            status_cb=ocr_update,
            ocr_concurrency=ocr_concurrency,
            selected_item_keys=selected_keys,
        )
        final_stats = self._selected_ocr_stats(selected_keys)
        final_msg = None
        if final_stats["error"] > 0:
            final_msg = f"ocr error={final_stats['error']}, ok={final_stats['ok']}"
        self._write_status(job_dir, "done", final_stats["processed"], ocr_total, final_msg)
        return job_id

    def _serve_query_result(self, job_id):
        return self._send_redirect(f"/query_view.html?job={job_id}")

    def _serve_query_view(self):
        view_path = self._base_dir() / "query_view.html"
        if not view_path.exists():
            return self._send_json(404, {"error": "query_view.html not found"})
        html = view_path.read_text(encoding="utf-8")
        return self._send_html(200, html)

    def _runtime_info(self, payload=None):
        key, source = self._resolve_api_key_with_source(payload)
        base_url = self._resolve_runtime_base_url(payload)
        model = self._resolve_runtime_model(payload)
        timeout_s = self._resolve_runtime_timeout_s(payload)
        runtime = {
            "api_key_source": source,
            "env_visible": bool(
                self._pick_non_empty_str(
                    os.environ.get("SILICONFLOW_API_KEY"),
                    os.environ.get("OPENAI_API_KEY"),
                    default="",
                )
            ),
            "llm_config_path": LLM_CONFIG_PATH or "",
            "base_url": base_url,
            "model": model,
            "timeout_s": timeout_s,
            "platform": sys.platform,
            "profile": str(self._base_dir()),
        }
        return key, source, base_url, model, timeout_s, runtime

    def _runtime_check(self, payload=None):
        req = payload if isinstance(payload, dict) else {}
        check_remote = bool(req.get("check_remote"))
        api_key, source, base_url, model, timeout_s, runtime = self._runtime_info(req)

        print(
            f"[runtime] check_remote={check_remote} api_key_source={source} "
            f"base_url={base_url} model={model} timeout_s={timeout_s}"
        )

        if not api_key:
            return {
                "ok": False,
                "code": "api_key_missing",
                "error": "missing API key (set PaperView API Key in Settings, then Save)",
                "runtime": runtime,
            }

        if not check_remote:
            return {"ok": True, "runtime": runtime}

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Return only: OK"}],
            "temperature": 0.0,
            "max_tokens": 4,
        }

        start = time.time()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
            latency_ms = int((time.time() - start) * 1000)
            status = int(resp.status_code)
            if status in {401, 403}:
                return {
                    "ok": False,
                    "code": "remote_api_auth",
                    "status": status,
                    "error": f"remote api auth failed (HTTP {status})",
                    "runtime": runtime,
                    "remote": {"reachable": False, "url": url, "latency_ms": latency_ms},
                }
            if status >= 400:
                snippet = (resp.text or "").strip().replace("\n", " ")[:240]
                return {
                    "ok": False,
                    "code": "remote_api_http_error",
                    "status": status,
                    "error": f"remote api returned HTTP {status}: {snippet}",
                    "runtime": runtime,
                    "remote": {"reachable": False, "url": url, "latency_ms": latency_ms},
                }

            data = {}
            try:
                data = resp.json()
            except Exception:
                data = {}
            remote_model = data.get("model") if isinstance(data, dict) else None
            return {
                "ok": True,
                "runtime": runtime,
                "remote": {
                    "reachable": True,
                    "status": status,
                    "latency_ms": latency_ms,
                    "model": remote_model or model,
                    "url": url,
                },
            }
        except requests.exceptions.Timeout as e:
            return {
                "ok": False,
                "code": "remote_api_timeout",
                "error": str(e),
                "runtime": runtime,
                "remote": {"reachable": False, "url": url},
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "ok": False,
                "code": "remote_api_unreachable",
                "error": str(e),
                "runtime": runtime,
                "remote": {"reachable": False, "url": url},
            }
        except requests.exceptions.RequestException as e:
            status = None
            if getattr(e, "response", None) is not None:
                status = int(e.response.status_code)
            code = "remote_api_http_error" if status else "remote_api_unreachable"
            if status in {401, 403}:
                code = "remote_api_auth"
            return {
                "ok": False,
                "code": code,
                "status": status,
                "error": str(e),
                "runtime": runtime,
                "remote": {"reachable": False, "url": url},
            }

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._send_json(200, {"ok": True})
        if path == "/history":
            return self._send_json(200, {"history": self._history_entries()})
        if path == "/query_view.html":
            return self._serve_query_view()
        if path.startswith("/status/"):
            job_id = path.split("/status/")[-1].strip("/")
            status = self._read_status(job_id)
            if not status:
                return self._send_json(404, {"error": "not found"})
            return self._send_json(200, status)
        if path.startswith("/result/"):
            job_id = path.split("/result/")[-1].strip("/")
            if job_id == "demo":
                return self._send_redirect("/query_view.html")
            return self._serve_query_result(job_id)
        if path.startswith("/data/"):
            rel = path[len("/data/"):].lstrip("/")
            target = (self._query_dir() / rel).resolve()
            if not str(target).startswith(str(self._query_dir().resolve())):
                return self._send_json(403, {"error": "forbidden"})
            if not target.exists():
                return self._send_json(404, {"error": "not found"})
            if target.is_dir():
                return self._send_json(400, {"error": "not a file"})
            body = target.read_bytes()
            self.send_response(200)
            if target.suffix.lower() == ".jsonl":
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/query":
            return self._send_json(200, {"result_url": self._result_url()})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if path == "/query":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            query_text = payload.get("query") or ""
            item_keys = payload.get("item_keys") or []
            if query_text:
                print(f"[query] {query_text} | items={len(item_keys)}")
            job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            job_dir = self._query_job_dir(job_id)
            self._write_status(job_dir, "queued", 0, len(item_keys) or 0)
            result_url = f"http://localhost:{self.server.server_address[1]}/result/{job_id}"

            def run_job():
                try:
                    self._run_query_pipeline(payload, job_id)
                except Exception as e:
                    code, text = self._classify_exception(e, stage="query")
                    self._write_status(
                        job_dir,
                        "error",
                        0,
                        len(item_keys) or 0,
                        text,
                        error_code=code,
                    )

            threading.Thread(target=run_job, daemon=True).start()
            return self._send_json(200, {"result_url": result_url, "job_id": job_id})
        if path == "/ocr":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            item_keys = payload.get("item_keys") or []
            job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            job_dir = self._query_job_dir(job_id)
            self._write_status(job_dir, "queued", 0, len(item_keys) or 0)

            def run_job():
                try:
                    self._run_ocr_pipeline(payload, job_id)
                except Exception as e:
                    code, text = self._classify_exception(e, stage="ocr")
                    self._write_status(
                        job_dir,
                        "error",
                        0,
                        len(item_keys) or 0,
                        text,
                        error_code=code,
                    )

            threading.Thread(target=run_job, daemon=True).start()
            return self._send_json(200, {"job_id": job_id})
        if path == "/runtime/check":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            try:
                return self._send_json(200, self._runtime_check(payload))
            except Exception as e:
                code, text = self._classify_exception(e, stage="runtime")
                return self._send_json(
                    500,
                    {"ok": False, "code": code, "error": text},
                )
        if path == "/ingest":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "invalid json"})
            items = payload.get("items") or []
            written = self._write_ingest(items)
            missing_pdf = sum(1 for item in items if item.get("pdf_missing"))
            return self._send_json(
                200,
                {
                    "ok": True,
                    "received": len(items),
                    "written": written,
                    "missing_pdf": missing_pdf,
                },
            )
        if path == "/history/delete":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            job_id = (payload.get("job_id") or "").strip()
            if not job_id:
                return self._send_json(400, {"ok": False, "error": "missing job_id"})
            try:
                deleted = self._delete_history_job(job_id)
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e), "job_id": job_id})
            print(f"[history] delete job_id={job_id} deleted={deleted}")
            return self._send_json(200, {"ok": True, "job_id": job_id, "deleted": bool(deleted)})
        if path == "/history/clear":
            try:
                removed = self._clear_history()
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})
            print(f"[history] clear removed={removed}")
            return self._send_json(200, {"ok": True, "removed": int(removed)})
        return self._send_json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="PaperView local demo service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20341)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--llm-config", default="")
    args = parser.parse_args()

    global LLM_CONFIG_PATH
    LLM_CONFIG_PATH = args.llm_config or os.environ.get("PAPERVIEW_LLM_CONFIG", "")

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        os.dup2(log_fh.fileno(), 1)
        os.dup2(log_fh.fileno(), 2)
        sys.stdout = log_fh
        sys.stderr = log_fh
        print(f"[boot] logging to {log_path}")

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"PaperView demo service running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
LLM_CONFIG_PATH = ""
