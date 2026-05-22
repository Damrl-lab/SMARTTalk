import unittest

from smarttalk.inference.output_parser import (
    best_effort_prediction_payload,
    extract_first_json_block,
    normalize_status,
)


class OutputParserTests(unittest.TestCase):
    def test_extract_json_block(self) -> None:
        payload = extract_first_json_block('prefix {"status":"AT_RISK","ttf_bucket":"<7"} suffix')
        self.assertEqual(payload["status"], "AT_RISK")

    def test_extract_json_block_from_fenced_payload(self) -> None:
        payload = extract_first_json_block(
            '```json\n{"status":"HEALTHY","ttf_bucket":"NONE"}\n```'
        )
        self.assertEqual(payload["status"], "HEALTHY")

    def test_normalize_status(self) -> None:
        self.assertEqual(normalize_status("AT_RISK"), 1)
        self.assertEqual(normalize_status("healthy"), 0)

    def test_best_effort_payload(self) -> None:
        payload = best_effort_prediction_payload(
            "The drive appears AT_RISK and likely within 7 days. Replace soon."
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "AT_RISK")
        self.assertEqual(payload["ttf_bucket"], "<7")


if __name__ == "__main__":
    unittest.main()
