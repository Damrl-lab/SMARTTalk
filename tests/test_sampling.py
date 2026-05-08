import unittest

import numpy as np

from smarttalk.data.imbalanced_sampling import build_sampled_test_indices


class SamplingTests(unittest.TestCase):
    def test_sampling_contains_all_failures(self) -> None:
        y = np.array([1] * 4 + [0] * 100)
        selected = build_sampled_test_indices(y, healthy_per_failed=2, seed=2026)
        self.assertTrue(set(range(4)).issubset(set(selected.tolist())))


if __name__ == "__main__":
    unittest.main()
