import unittest

from smarttalk.common.paths import ROOT
from smarttalk.common.runner import _base_env


class RunnerEnvTests(unittest.TestCase):
    def test_base_env_includes_repo_root_in_pythonpath(self) -> None:
        env = _base_env()
        pythonpath = env.get("PYTHONPATH", "")
        self.assertIn(str(ROOT), pythonpath.split(":"))


if __name__ == "__main__":
    unittest.main()
