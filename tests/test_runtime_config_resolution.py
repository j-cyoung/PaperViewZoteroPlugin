import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1] / "service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import local_service  # noqa: E402


class _RuntimeResolver:
    _pick_positive_int = local_service.DemoHandler._pick_positive_int
    _pick_non_empty_str = local_service.DemoHandler._pick_non_empty_str
    _load_runtime_config = local_service.DemoHandler._load_runtime_config
    _resolve_ocr_concurrency = local_service.DemoHandler._resolve_ocr_concurrency
    _resolve_api_key_with_source = local_service.DemoHandler._resolve_api_key_with_source
    _resolve_api_key = local_service.DemoHandler._resolve_api_key


class RuntimeConfigResolutionTests(unittest.TestCase):
    def setUp(self):
        self._orig_llm_config_path = local_service.LLM_CONFIG_PATH
        self._orig_ocr_env = os.environ.get("PAPERVIEW_OCR_CONCURRENCY")
        self._orig_sf_api_key = os.environ.get("SILICONFLOW_API_KEY")
        self._orig_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        local_service.LLM_CONFIG_PATH = self._orig_llm_config_path
        if self._orig_ocr_env is None:
            os.environ.pop("PAPERVIEW_OCR_CONCURRENCY", None)
        else:
            os.environ["PAPERVIEW_OCR_CONCURRENCY"] = self._orig_ocr_env
        if self._orig_sf_api_key is None:
            os.environ.pop("SILICONFLOW_API_KEY", None)
        else:
            os.environ["SILICONFLOW_API_KEY"] = self._orig_sf_api_key
        if self._orig_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._orig_openai_api_key
        self.tmp.cleanup()

    def _write_cfg(self, payload):
        path = Path(self.tmp.name) / "llm_config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        local_service.LLM_CONFIG_PATH = str(path)
        return path

    def test_ocr_concurrency_resolution_priority(self):
        self._write_cfg({"ocr_concurrency": 6, "concurrency": 5})
        os.environ["PAPERVIEW_OCR_CONCURRENCY"] = "7"
        resolver = _RuntimeResolver()

        # request payload should have highest priority
        self.assertEqual(
            resolver._resolve_ocr_concurrency({"ocr_concurrency": 9}),
            9,
        )
        # then config.ocr_concurrency
        self.assertEqual(
            resolver._resolve_ocr_concurrency({}),
            6,
        )

    def test_ocr_concurrency_fallback_chain(self):
        resolver = _RuntimeResolver()

        # fallback to config.concurrency when ocr_concurrency is missing
        self._write_cfg({"concurrency": 8})
        os.environ["PAPERVIEW_OCR_CONCURRENCY"] = "11"
        self.assertEqual(resolver._resolve_ocr_concurrency({}), 8)

        # fallback to env when config has no usable values
        self._write_cfg({})
        self.assertEqual(resolver._resolve_ocr_concurrency({}), 11)

        # fallback to built-in default when env is invalid
        os.environ["PAPERVIEW_OCR_CONCURRENCY"] = "not-a-number"
        self.assertEqual(
            resolver._resolve_ocr_concurrency({}),
            local_service.DEFAULT_OCR_CONCURRENCY,
        )

    def test_api_key_resolution_priority(self):
        self._write_cfg({"api_key": "cfg-key"})
        os.environ["SILICONFLOW_API_KEY"] = "env-sf-key"
        os.environ["OPENAI_API_KEY"] = "env-openai-key"
        resolver = _RuntimeResolver()

        # request payload should have highest priority
        self.assertEqual(
            resolver._resolve_api_key({"api_key": "payload-key"}),
            "payload-key",
        )
        # then runtime config
        self.assertEqual(
            resolver._resolve_api_key({}),
            "cfg-key",
        )

        value, source = resolver._resolve_api_key_with_source({})
        self.assertEqual(value, "cfg-key")
        self.assertEqual(source, "prefs")

        value, source = resolver._resolve_api_key_with_source({"api_key": "payload-key"})
        self.assertEqual(value, "payload-key")
        self.assertEqual(source, "request")

    def test_api_key_resolution_env_fallback(self):
        self._write_cfg({})
        resolver = _RuntimeResolver()

        os.environ["SILICONFLOW_API_KEY"] = "env-sf-key"
        os.environ.pop("OPENAI_API_KEY", None)
        self.assertEqual(resolver._resolve_api_key({}), "env-sf-key")

        os.environ["SILICONFLOW_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = "env-openai-key"
        self.assertEqual(resolver._resolve_api_key({}), "env-openai-key")

        os.environ["SILICONFLOW_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = ""
        self.assertEqual(resolver._resolve_api_key({}), "")

        value, source = resolver._resolve_api_key_with_source({})
        self.assertEqual(value, "")
        self.assertEqual(source, "none")

if __name__ == "__main__":
    unittest.main()
