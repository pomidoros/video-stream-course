"""Opt-in integration suite: it recomputes predictions and metrics from saved models."""

import os
import unittest
from pathlib import Path
from project.common import ROOT
from project.pipeline import evaluate
from project.thresholds import checks


@unittest.skipUnless(
    os.environ.get("RUN_METRIC_TESTS") == "1",
    "Run: python scripts/check.py metrics --models artifacts",
)
class PublicMetricTests(unittest.TestCase):
    def test_published_thresholds(self):
        models = Path(os.environ.get("COURSE_MODELS", ROOT / "artifacts"))
        result = evaluate(ROOT, models, ROOT / "outputs/metric_tests")
        for row in checks(result):
            with self.subTest(metric=row["metric"]):
                self.assertTrue(
                    row["passed"],
                    f"{row['value']} {row['operator']} {row['threshold']}",
                )


if __name__ == "__main__":
    unittest.main()
