"""Готовая оценка треков и сравнение счётчиков с контрольными значениями."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_boxes(path: Path, id_column: str) -> dict[int, list[tuple[int, np.ndarray]]]:
    frames: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("confirmed", "1") == "0":
                continue
            center = np.array([float(row["x"]) + float(row["width"]) / 2, float(row["y"]) + float(row["height"]) / 2])
            frames[int(row["frame"])].append((int(row[id_column]), center))
    return frames


def evaluate_tracks(gt_path: Path, tracks_path: Path, threshold: float = 60.0) -> dict:
    gt, predicted = read_boxes(gt_path, "track_id"), read_boxes(tracks_path, "track_id")
    matches = 0
    errors: list[float] = []
    id_switches = 0
    last_prediction: dict[int, int] = {}
    seen_gt: set[int] = set()
    matched_gt: set[int] = set()
    total_gt = sum(len(items) for items in gt.values())
    for frame in sorted(gt):
        truth, estimate = gt[frame], predicted.get(frame, [])
        seen_gt.update(item[0] for item in truth)
        if not truth or not estimate:
            continue
        costs = np.linalg.norm(np.stack([item[1] for item in truth])[:, None, :] - np.stack([item[1] for item in estimate])[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(costs)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if costs[row, col] > threshold:
                continue
            gt_id, pred_id = truth[row][0], estimate[col][0]
            if gt_id in last_prediction and last_prediction[gt_id] != pred_id:
                id_switches += 1
            last_prediction[gt_id] = pred_id
            matched_gt.add(gt_id)
            matches += 1
            errors.append(float(costs[row, col]))
    return {
        "matched_boxes": matches,
        "gt_boxes": total_gt,
        "center_recall": round(matches / total_gt, 4) if total_gt else 0.0,
        "mean_center_error": round(float(np.mean(errors)), 3) if errors else None,
        "id_switches": id_switches,
        "lost_gt_tracks": len(seen_gt - matched_gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--expected-counts", type=Path)
    args = parser.parse_args()
    gt_path = args.ground_truth or args.data / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError("Разметка challenge закрыта. Локально оценивайте debug и development.")
    metrics = evaluate_tracks(gt_path, args.results / "tracks.csv")
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    counts_path = args.expected_counts or args.data / "expected_counts.json"
    expected = json.loads(counts_path.read_text(encoding="utf-8"))
    expected_entered, expected_exited = expected["entered"], expected["exited"]
    metrics.update({
        "expected_entered": expected_entered,
        "expected_exited": expected_exited,
        "entry_count_error": abs(summary["entered"] - expected_entered),
        "exit_count_error": abs(summary["exited"] - expected_exited),
    })
    metrics.update({f"reported_{key}": value for key, value in summary.items()})
    output = args.results / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
