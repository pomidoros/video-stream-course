"""Служебные загрузка данных и подключение студенческих функций."""

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def solutions():
    return (
        importlib.import_module(
            os.environ.get("COURSE_OUTCOME_MODULE", "shots.outcome_student")
        ),
        importlib.import_module(
            os.environ.get("COURSE_PEOPLE_MODULE", "players.people_student")
        ),
    )


def read_detections(path):
    frames = {}
    times = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            f = int(row["frame"])
            times[f] = float(row["time_s"])
            frames.setdefault(f, [])
            if row["class_name"]:
                frames[f].append(
                    dict(
                        class_name=row["class_name"],
                        confidence=float(row["confidence"]),
                        box=np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2")]),
                    )
                )
    return [(f, times[f], frames[f]) for f in sorted(frames)]


def checked_detections(root, row):
    p = Path(root) / row["detections"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != row["detections_sha256"]:
        raise ValueError("Detection checksum mismatch")
    return read_detections(p)


def stress_input(data):
    dropped = data["frames"][[20 <= i % 45 < 25 for i in range(len(data["frames"]))]]
    mask = ~np.isin(data["det_frame"], dropped)
    return {k: v.copy() if k == "frames" else v[mask].copy() for k, v in data.items()}
