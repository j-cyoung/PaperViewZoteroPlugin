import json
import sys
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple
from email.message import Message


SERVICE_DIR = Path(__file__).resolve().parents[1] / "service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import local_service  # noqa: E402


class IntegrationTestHandler(local_service.DemoHandler):
    """Test-only handler that keeps endpoint behavior but stubs heavy pipelines."""

    def _base_dir(self):
        return Path(self.server.test_base_dir)

    def log_message(self, _fmt, *_args):
        return

    def _run_query_pipeline(self, payload, job_id):
        if payload.get("query") == "boom":
            raise RuntimeError("query_papers failed with code 2")
        item_keys = payload.get("item_keys") or []
        selected = [{"item_key": k, "title": f"title-{k}"} for k in item_keys]
        job_dir = self._query_job_dir(job_id)
        self._write_query_snapshot(job_dir, payload, selected)
        self._write_jsonl(
            job_dir / "papers.query.jsonl",
            [
                {
                    "paper_id": k,
                    "title": f"title-{k}",
                    "query_status": "ok",
                    "query_response": "stubbed response",
                }
                for k in item_keys
            ],
        )
        self._write_status(job_dir, "done", len(item_keys), len(item_keys))
        return job_id

    def _run_ocr_pipeline(self, payload, job_id):
        if payload.get("force_fail"):
            raise RuntimeError("pdf_to_md_pymupdf4llm failed with code 1")
        item_keys = payload.get("item_keys") or []
        job_dir = self._query_job_dir(job_id)
        self._write_status(job_dir, "done", len(item_keys), len(item_keys))
        return job_id

    def _runtime_check(self, payload=None):
        req = payload if isinstance(payload, dict) else {}
        api_key = str(req.get("api_key") or "").strip()
        if not api_key:
            return {
                "ok": False,
                "code": "api_key_missing",
                "error": "missing API key",
                "runtime": {
                    "api_key_source": "none",
                    "env_visible": False,
                    "platform": "test",
                },
            }
        runtime = {
            "api_key_source": "request",
            "env_visible": False,
            "platform": "test",
        }
        if req.get("check_remote"):
            return {
                "ok": True,
                "runtime": runtime,
                "remote": {
                    "reachable": True,
                    "status": 200,
                    "latency_ms": 12,
                    "model": req.get("model") or "test-model",
                },
            }
        return {"ok": True, "runtime": runtime}


class _DummyServer:
    def __init__(self, base_dir: str):
        self.test_base_dir = Path(base_dir)
        self.server_address = ("127.0.0.1", 20341)


class NoSocketIntegrationHandler(IntegrationTestHandler):
    """Drive BaseHTTPRequestHandler routes without opening a real TCP socket."""

    def __init__(self, server: _DummyServer, method: str, path: str, body: bytes = b""):
        self.server = server
        self.command = method
        self.path = path
        self.request_version = "HTTP/1.1"
        self.requestline = f"{method} {path} HTTP/1.1"
        self.close_connection = True

        self._status = None
        self._headers: Dict[str, str] = {}
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        if body:
            self.headers["Content-Type"] = "application/json"

    def send_response(self, code, message=None):
        self._status = code

    def send_header(self, key, value):
        self._headers[key] = value

    def end_headers(self):
        return


class ServiceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = _DummyServer(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _request_handler(self, method: str, path: str, payload: Dict[str, Any] = None):
        body = b""
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        handler = NoSocketIntegrationHandler(self.server, method, path, body=body)
        if method.upper() == "GET":
            handler.do_GET()
        elif method.upper() == "POST":
            handler.do_POST()
        else:
            raise ValueError(f"unsupported method: {method}")
        return handler

    def request_json(
        self,
        method: str,
        path: str,
        payload: Dict[str, Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        handler = self._request_handler(method, path, payload)
        body = handler.wfile.getvalue().decode("utf-8")
        return handler._status, json.loads(body) if body else {}

    def request_text(self, method: str, path: str) -> Tuple[int, str]:
        handler = self._request_handler(method, path)
        body = handler.wfile.getvalue().decode("utf-8")
        return handler._status, body

    def wait_for_job_done(self, job_id: str, timeout_s: float = 3.0) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status, data = self.request_json("GET", f"/status/{job_id}")
            if status == 200 and data.get("stage") in {"done", "error"}:
                return data
            time.sleep(0.05)
        self.fail(f"job did not finish in time: {job_id}")

    def test_health_endpoint(self):
        status, data = self.request_json("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("ok"), True)

    def test_ingest_endpoint(self):
        payload = {
            "items": [
                {"item_key": "A1", "title": "Paper A", "pdf_missing": False},
                {"item_key": "B2", "title": "Paper B", "pdf_missing": True},
            ]
        }
        status, data = self.request_json("POST", "/ingest", payload)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("ok"), True)
        self.assertEqual(data.get("received"), 2)
        self.assertEqual(data.get("written"), 2)
        self.assertEqual(data.get("missing_pdf"), 1)

    def test_query_status_and_history_lifecycle(self):
        status, data = self.request_json(
            "POST",
            "/query",
            {"item_keys": ["K1", "K2"], "query": "what is the contribution?"},
        )
        self.assertEqual(status, 200)
        job_id = data.get("job_id")
        self.assertTrue(job_id)
        self.assertIn("/result/", data.get("result_url", ""))

        done_status = self.wait_for_job_done(job_id)
        self.assertEqual(done_status.get("stage"), "done")
        self.assertEqual(done_status.get("done"), 2)

        status, history = self.request_json("GET", "/history")
        self.assertEqual(status, 200)
        jobs = [h.get("job_id") for h in history.get("history", [])]
        self.assertIn(job_id, jobs)

        status, text = self.request_text("GET", f"/data/{job_id}/papers.query.jsonl")
        self.assertEqual(status, 200)
        self.assertIn("stubbed response", text)

        status, result = self.request_json("POST", "/history/delete", {"job_id": job_id})
        self.assertEqual(status, 200)
        self.assertEqual(result.get("ok"), True)
        self.assertEqual(result.get("deleted"), True)

    def test_ocr_job_status(self):
        status, data = self.request_json("POST", "/ocr", {"item_keys": ["OCR1"]})
        self.assertEqual(status, 200)
        job_id = data.get("job_id")
        self.assertTrue(job_id)
        done_status = self.wait_for_job_done(job_id)
        self.assertEqual(done_status.get("stage"), "done")
        self.assertEqual(done_status.get("done"), 1)

    def test_history_clear(self):
        first = self.request_json("POST", "/query", {"item_keys": ["A"], "query": "q1"})[1]["job_id"]
        second = self.request_json("POST", "/query", {"item_keys": ["B"], "query": "q2"})[1]["job_id"]
        self.wait_for_job_done(first)
        self.wait_for_job_done(second)

        status, result = self.request_json("POST", "/history/clear", {})
        self.assertEqual(status, 200)
        self.assertEqual(result.get("ok"), True)
        self.assertGreaterEqual(int(result.get("removed", 0)), 2)

        status, history = self.request_json("GET", "/history")
        self.assertEqual(status, 200)
        self.assertEqual(history.get("history"), [])

    def test_data_path_traversal_is_forbidden(self):
        status, data = self.request_json("GET", "/data/../../etc/passwd")
        self.assertEqual(status, 403)
        self.assertEqual(data.get("error"), "forbidden")

    def test_runtime_check_endpoint(self):
        status, data = self.request_json(
            "POST",
            "/runtime/check",
            {"api_key": "k1", "base_url": "https://example.com/v1", "model": "m1"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data.get("ok"), True)
        self.assertEqual(data.get("runtime", {}).get("api_key_source"), "request")

        status, data = self.request_json(
            "POST",
            "/runtime/check",
            {"api_key": "k1", "check_remote": True, "model": "m2"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data.get("ok"), True)
        self.assertEqual(data.get("remote", {}).get("reachable"), True)
        self.assertEqual(data.get("remote", {}).get("model"), "m2")

        status, data = self.request_json("POST", "/runtime/check", {"api_key": ""})
        self.assertEqual(status, 200)
        self.assertEqual(data.get("ok"), False)
        self.assertEqual(data.get("code"), "api_key_missing")

    def test_query_status_reports_structured_error_code(self):
        status, data = self.request_json(
            "POST",
            "/query",
            {"item_keys": ["K1"], "query": "boom"},
        )
        self.assertEqual(status, 200)
        job_id = data.get("job_id")
        done_status = self.wait_for_job_done(job_id)
        self.assertEqual(done_status.get("stage"), "error")
        self.assertEqual(done_status.get("error_code"), "query_pipeline_failed")


if __name__ == "__main__":
    unittest.main()
