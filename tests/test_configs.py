from pathlib import Path
import unittest

from smarttalk.common.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "default_mb2.yaml")
        self.assertEqual(cfg["dataset"], "MB2")
        self.assertEqual(cfg["window_size"], 30)


if __name__ == "__main__":
    unittest.main()
