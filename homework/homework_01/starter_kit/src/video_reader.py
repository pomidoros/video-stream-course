from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def read_video(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frame = 1
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield frame, image
            frame += 1
    finally:
        capture.release()


def create_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {path}")
    return writer
