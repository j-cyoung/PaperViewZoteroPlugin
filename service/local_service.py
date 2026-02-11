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
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_OCR_CONCURRENCY = 4
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

    def _ensure_ocr_cache(self, progress_path=None, status_cb=None, ocr_concurrency=DEFAULT_OCR_CONCURRENCY):
        with OCR_CACHE_LOCK:
            base_dir = self._base_dir()
            self._ocr_dir().mkdir(parents=True, exist_ok=True)
            # update items.csv from items.jsonl
            self._run_cmd(
                [
                    sys.executable,
                    "zotero_items_to_csv.py",
                    "--jsonl",
                    str(self._items_jsonl()),
                    "--csv_out",
                    str(self._items_csv()),
                ],
                cwd=base_dir,
            )
            # run OCR with resume to reuse cached results
            cmd = [
                sys.executable,
                "pdf_to_md_pymupdf4llm.py",
                "--csv_in",
                str(self._items_csv()),
                "--base_output_dir",
                str(self._ocr_dir()),
                "--out_jsonl",
                "papers.pages.jsonl",
                "--resume",
                "--resume_from",
                str(self._pages_jsonl()),
                "--concurrency",
                str(max(1, int(ocr_concurrency))),
            ]
            if progress_path:
                cmd += ["--progress_path", str(progress_path)]
            print(f"[ocr] start ensure cache, concurrency={ocr_concurrency}, progress_path={progress_path}")
            print(f"[cmd] {' '.join(cmd)}")
            if not status_cb:
                self._run_cmd(cmd, cwd=base_dir)
                return
            proc = subprocess.Popen(cmd, cwd=base_dir)
            out_jsonl = self._pages_jsonl()
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
                if done is not None:
                    if done != last_done or total != last_total:
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
            # final update
            if progress_path:
                try:
                    progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                    done = int(progress.get("done") or 0)
                    total = int(progress.get("total") or 0)
                    status_cb(done, total)
                    return
                except Exception:
                    pass
            if out_jsonl.exists():
                with out_jsonl.open("r", encoding="utf-8") as f:
                    done = sum(1 for line in f if line.strip())
                status_cb(done, None)

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

    def _count_selected_ocr_done(self, selected_keys):
        return self._selected_ocr_stats(selected_keys).get("processed", 0)

    def _query_job_dir(self, job_id):
        job_dir = self._query_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _status_path(self, job_dir):
        return job_dir / "status.json"

    def _write_status(self, job_dir, stage, done=None, total=None, message=None):
        payload = {
            "stage": stage,
            "done": done,
            "total": total,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
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
        selected = [item for item in items if item.get("item_key") in set(item_keys)]
        if not selected:
            raise ValueError("no matching items in items.jsonl")

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
        selected = [item for item in items if item.get("item_key") in set(item_keys)]
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
                    self._write_status(job_dir, "error", 0, len(item_keys) or 0, str(e))

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
                    self._write_status(job_dir, "error", 0, len(item_keys) or 0, str(e))

            threading.Thread(target=run_job, daemon=True).start()
            return self._send_json(200, {"job_id": job_id})
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
