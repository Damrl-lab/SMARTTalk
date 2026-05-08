from pathlib import Path
import unittest

from smarttalk.data.schema import summarize_npz_split


class SchemaTests(unittest.TestCase):
    def test_sample_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = summarize_npz_split(root / "data" / "sample_data" / "MB1_round1_test_sample.npz")
        self.assertEqual(summary["window_days"], 30)
        self.assertEqual(summary["num_attributes"], 15)
        self.assertGreater(summary["num_windows"], 0)


if __name__ == "__main__":
    unittest.main()
