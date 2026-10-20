"""Запуск проверок. Ненулевой код завершения означает ошибку."""

from pathlib import Path
import argparse
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument("suite", choices=["environment", "contracts", "metrics", "all"])
p.add_argument("--models", type=Path, default=ROOT / "artifacts")
a = p.parse_args()
if (
    not (ROOT / "data/recordings.json").is_file()
    or not (ROOT / "players/data/manifest.json").is_file()
):
    p.error(
        "Сначала установите данные: python scripts/setup_data.py --archive ПУТЬ_К_ZIP. См. DATA.md"
    )
if a.suite in ("metrics", "all"):
    os.environ["RUN_METRIC_TESTS"] = "1"
    os.environ["COURSE_MODELS"] = str(a.models.resolve())
patterns = {
    "environment": [
        "test_environment.py",
        "test_infrastructure.py",
        "test_data_setup.py",
    ],
    "contracts": ["test_contracts.py", "test_practice.py"],
    "metrics": ["test_metrics.py"],
    "all": ["test_*.py"],
}[a.suite]
suite = unittest.TestSuite()
for pattern in patterns:
    suite.addTests(
        unittest.defaultTestLoader.discover(
            str(ROOT / "tests"), pattern=pattern, top_level_dir=str(ROOT)
        )
    )
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
