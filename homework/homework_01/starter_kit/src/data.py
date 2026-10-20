from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    frames: int
    fps: float
    width: int
    height: int


def load_measurements(path: Path) -> dict[int, list[Detection]]:
    result: dict[int, list[Detection]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "measurement_id", "x", "y", "width", "height", "confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"measurements columns must contain {sorted(required)}")
        for row in reader:
            frame = int(row["frame"])
            result.setdefault(frame, []).append(
                Detection(float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"]), float(row["confidence"]))
            )
    return result


def load_sequence_info(path: Path) -> SequenceInfo:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SequenceInfo(raw["name"], int(raw["frames"]), float(raw["fps"]), int(raw["width"]), int(raw["height"]))


def load_zones(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "entry_line" not in raw or len(raw["entry_line"]) != 2:
        raise ValueError("zones.json must contain a two-point entry_line")
    return raw


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
