from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from .data import Detection
from .tracker import TrackView


class Visualizer:
    def __init__(self, trail_length: int = 25) -> None:
        self.trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))

    def draw(
        self,
        image: np.ndarray,
        detections: list[Detection],
        tracks: list[TrackView],
        line: list[list[float]],
        entered: int,
        exited: int,
        current: int,
        recovered: int,
    ) -> np.ndarray:
        canvas = image.copy()
        for detection in detections:
            x, y, w, h = map(int, (detection.x, detection.y, detection.width, detection.height))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (170, 170, 170), 1)
        a, b = tuple(map(int, line[0])), tuple(map(int, line[1]))
        cv2.line(canvas, a, b, (255, 210, 30), 2)
        for track in tracks:
            x, y, w, h = map(int, track.bbox)
            color = (60, 220, 90) if track.source == "measurement" else (40, 150, 255)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.putText(canvas, f"ID {track.track_id} {track.source[0].upper()}", (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            point = tuple(map(int, track.center))
            self.trails[track.track_id].append(point)
            trail = list(self.trails[track.track_id])
            for old, new in zip(trail, trail[1:]):
                cv2.line(canvas, old, new, color, 1)
        label = f"IN {entered}  OUT {exited}  NOW {current}  RECOVERED {recovered}"
        cv2.rectangle(canvas, (8, image.shape[0] - 35), (520, image.shape[0] - 6), (20, 20, 20), -1)
        cv2.putText(canvas, label, (16, image.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        return canvas
