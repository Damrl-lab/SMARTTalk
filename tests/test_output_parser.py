import unittest

from smarttalk.inference.output_parser import extract_first_json_block, normalize_status


class OutputParserTests(unittest.TestCase):
    def test_extract_json_block(self) -> None:
        payload = extract_first_json_block('prefix {"status":"AT_RISK","ttf_bucket":"<7"} suffix')
        self.assertEqual(payload["status"], "AT_RISK")

    def test_normalize_status(self) -> None:
        self.assertEqual(normalize_status("AT_RISK"), 1)
        self.assertEqual(normalize_status("healthy"), 0)


if __name__ == "__main__":
    unittest.main()
