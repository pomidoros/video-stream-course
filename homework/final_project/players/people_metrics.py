import numpy as np
import motmetrics as mm


def iou(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    inter = np.maximum(
        0,
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2]),
    ).prod(-1)
    area = np.maximum(0, a[:, 2:] - a[:, :2]).prod(-1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area[:, None] + area_b[None, :] - inter, 1e-12)


def accumulator(rows, gt):
    acc = mm.MOTAccumulator(auto_id=True)
    for r in rows:
        truth = [g for g in gt if g["frame"] == r["frame"]]
        pred = [t for t in r["tracks"] if t["confirmed"]]
        overlap = iou([g["bbox"] for g in truth], [t["bbox"] for t in pred])
        dist = 1 - overlap
        dist[overlap < 0.5] = np.nan
        acc.update(
            [g["person_id"] for g in truth], [int(t["track_id"]) for t in pred], dist
        )
    return acc


def evaluate(rows, gt):
    fields = [
        "num_frames",
        "num_objects",
        "idf1",
        "mota",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "precision",
        "recall",
    ]
    return (
        mm.metrics.create()
        .compute(accumulator(rows, gt), metrics=fields)
        .iloc[0]
        .to_dict()
    )
