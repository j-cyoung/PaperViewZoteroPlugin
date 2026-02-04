#!/usr/bin/env python3
# Minimal local service for Zotero demo

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
        if self.path != "/query":
            return self._send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            _ = self.rfile.read(length)
        return self._send_json(200, {"result_url": self._result_url()})


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
