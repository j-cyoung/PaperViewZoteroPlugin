#!/usr/bin/env python3
# Minimal local service for Zotero demo

import argparse
import json
import subprocess
import uuid
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


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
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item.setdefault("ingest_time", now)
            lines.append(json.dumps(item, ensure_ascii=False))
        if lines:
            with out_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return len(lines)

    def _run_cmd(self, cmd, cwd=None):
        cmd_display = " ".join(cmd)
        print(f"[cmd] {cmd_display}")
        subprocess.run(cmd, cwd=cwd, check=True)

    def _ensure_ocr_cache(self):
        base_dir = self._base_dir()
        self._ocr_dir().mkdir(parents=True, exist_ok=True)
        # update items.csv from items.jsonl
        self._run_cmd(
            [
                "uv",
                "run",
                "python",
                "zotero_items_to_csv.py",
                "--jsonl",
                str(self._items_jsonl()),
                "--csv_out",
                str(self._items_csv()),
            ],
            cwd=base_dir,
        )
        # run OCR with resume to reuse cached results
        self._run_cmd(
            [
                "uv",
                "run",
                "python",
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
            ],
            cwd=base_dir,
        )

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
            paper_id = row.get("paper_id") or row.get("zotero_item_key")
            if paper_id in key_set:
                selected.append(row)
        self._write_jsonl(out_path, selected)
        return len(selected)

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

    def _write_query_snapshot(self, job_dir, payload, items):
        sections = payload.get("sections") or ""
        snapshot = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": payload.get("query") or "",
            "sections": sections,
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
        self._write_status(job_dir, "ocr", 0, len(selected))
        self._ensure_ocr_cache()
        self._write_status(job_dir, "ocr", len(selected), len(selected))

        # filter OCR pages for selected items
        filtered_pages = job_dir / "pages.selected.jsonl"
        self._filter_pages(item_keys, filtered_pages)

        # run query on selected pages with progress
        self._write_status(job_dir, "query", 0, len(selected))
        out_jsonl = job_dir / "papers.query.jsonl"
        progress_path = job_dir / "progress.json"
        cmd = [
            "uv",
            "run",
            "python",
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
        ]
        proc = subprocess.Popen(cmd, cwd=self._base_dir())
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
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/query":
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
        if self.path == "/ingest":
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
        return self._send_json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="PaperView local demo service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23119)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"PaperView demo service running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
