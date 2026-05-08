import unittest

from smarttalk.evaluation.status_metrics import compute_status_metrics


class StatusMetricTests(unittest.TestCase):
    def test_status_metrics_values(self) -> None:
        metrics = compute_status_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics.tp, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.tn, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertAlmostEqual(metrics.fpr, 0.5)
        self.assertAlmostEqual(metrics.fnr, 0.5)


if __name__ == "__main__":
    unittest.main()
