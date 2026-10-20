"""Проверки готовой подготовки данных: исходники сохраняются, интервалы точны."""

import csv
from pathlib import Path

import pytest

from src.prepare_gaps import FILES, GAPS, prepare_gaps

SOURCE = Path(__file__).resolve().parents[2] / "data/debug"


def read_rows(path):
    """Прочитать измерения без округления координат."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_gap_variants_preserve_source_and_remove_only_requested_frames(tmp_path):
    before = {name: (SOURCE / name).read_bytes() for name in (*FILES, "measurements.csv")}
    rows = read_rows(SOURCE / "measurements.csv")
    variants = prepare_gaps(SOURCE, tmp_path / "variants")
    assert len(variants) == len(GAPS)
    for gap, target in zip(GAPS, variants):
        assert target.name == f"gap_{gap:02d}"
        expected = [row for row in rows if not 40 <= int(row["frame"]) < 40 + gap]
        assert read_rows(target / "measurements.csv") == expected
        for name in FILES:
            assert (target / name).read_bytes() == before[name]
    assert (variants[0] / "measurements.csv").read_bytes() == before["measurements.csv"]
    assert before == {name: (SOURCE / name).read_bytes() for name in before}


def test_existing_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "student_work.txt"
    marker.write_text("Сохранить")
    with pytest.raises(FileExistsError):
        prepare_gaps(SOURCE, tmp_path)
    assert marker.read_text() == "Сохранить"


@pytest.mark.parametrize("start_frame", [0, 100])
def test_invalid_interval_creates_no_output(tmp_path, start_frame):
    target = tmp_path / "variants"
    with pytest.raises(ValueError):
        prepare_gaps(SOURCE, target, start_frame)
    assert not target.exists()


def test_output_inside_source_is_rejected():
    target = SOURCE / "nested_variants"
    with pytest.raises(ValueError):
        prepare_gaps(SOURCE, target)
    assert not target.exists()
