import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFS_XHTML = ROOT / "prefs" / "prefs.xhtml"


class PrefsUIContractTests(unittest.TestCase):
    def test_settings_pane_contains_all_expected_fields(self):
        tree = ET.parse(PREFS_XHTML)
        root = tree.getroot()
        self.assertEqual(root.attrib.get("id"), "paperview-pref-root")

        expected_ids = [
            "paperview-service-url",
            "paperview-api-key",
            "paperview-llm-base-url",
            "paperview-llm-model",
            "paperview-llm-temperature",
            "paperview-llm-max-output-tokens",
            "paperview-llm-concurrency",
            "paperview-llm-ocr-concurrency",
            "paperview-llm-retry-wait",
            "paperview-llm-retry-on-429",
            "paperview-pref-save",
            "paperview-pref-reset",
            "paperview-pref-check-runtime",
            "paperview-pref-test-connection",
            "paperview-pref-restart-service",
            "paperview-pref-status",
        ]

        for elem_id in expected_ids:
            node = root.find(f".//*[@id='{elem_id}']")
            self.assertIsNotNone(node, f"missing settings field: {elem_id}")

    def test_settings_pane_uses_explicit_onload_controller(self):
        tree = ET.parse(PREFS_XHTML)
        root = tree.getroot()
        onload = root.attrib.get("onload", "")
        self.assertIn("Zotero.PaperViewPrefsPane", onload)
        self.assertIn(".init(window)", onload)


if __name__ == "__main__":
    unittest.main()
