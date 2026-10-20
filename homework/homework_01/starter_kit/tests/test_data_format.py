import json
from pathlib import Path

import cv2
import pytest

from src.data import load_measurements, load_sequence_info, load_zones


PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["debug", "development", "challenge"])
def test_dataset_contract(name: str) -> None:
    data = PROJECT / "data" / name
    info = load_sequence_info(data / "seqinfo.yaml")
    measurements = load_measurements(data / "measurements.csv")
    zones = load_zones(data / "zones.json")
    capture = cv2.VideoCapture(str(data / "video.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == info.frames
    assert measurements and min(measurements) >= 1 and max(measurements) <= info.frames
    assert len(zones["entry_line"]) == 2
    capture.release()
    if name != "challenge":
        counts = json.loads((data / "expected_counts.json").read_text(encoding="utf-8"))
        assert set(counts) == {"entered", "exited"}
        assert all(isinstance(value, int) and value >= 0 for value in counts.values())


def test_challenge_ground_truth_is_not_public() -> None:
    assert not (PROJECT / "data" / "challenge" / "ground_truth.csv").exists()
    assert not (PROJECT / "data" / "challenge" / "expected_counts.json").exists()
