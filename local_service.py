#!/usr/bin/env python3
# Minimal local service for Zotero demo

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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

    def _result_url(self):
        host = "localhost"
        port = self.server.server_address[1]
        return f"http://{host}:{port}/result/demo"

    def _write_ingest(self, items):
        base_dir = Path(__file__).resolve().parent
        store_dir = base_dir / "store" / "zotero"
        store_dir.mkdir(parents=True, exist_ok=True)
        out_file = store_dir / "items.jsonl"
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

    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"ok": True})
        if self.path.startswith("/result/"):
            html = """<!doctype html>
<html lang=\"en\">
<head><meta charset=\"utf-8\"><title>PaperView Demo</title></head>
<body>
  <h2>PaperView Demo Result</h2>
  <p>Local service is running and returned this page.</p>
</body>
</html>
"""
            return self._send_html(200, html)
        if self.path == "/query":
            return self._send_json(200, {"result_url": self._result_url()})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/query":
            return self._send_json(200, {"result_url": self._result_url()})
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
