import csv
import json
from pathlib import Path

import pytest

TRACK_COLUMNS = {"frame", "track_id", "x", "y", "width", "height", "source", "confirmed"}


def validate_results(path: Path) -> None:
    with (path / "tracks.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "tracks.csv is empty"
    assert TRACK_COLUMNS.issubset(rows[0])
    assert {row["source"] for row in rows} <= {"measurement", "prediction"}
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert {"entered", "exited", "current", "recovered_tracks"}.issubset(summary)
    assert (path / "events.csv").exists()
    assert (path / "demo.mp4").stat().st_size > 0


@pytest.mark.parametrize("scene", ["debug", "development", "challenge"])
def test_student_output_format(scene: str) -> None:
    """Проверить файлы студента после запуска соответствующей сцены."""
    results = Path(__file__).resolve().parents[1] / "results" / scene
    if not results.exists():
        pytest.skip(f"Сначала запустите src.main для сцены {scene}")
    validate_results(results)
