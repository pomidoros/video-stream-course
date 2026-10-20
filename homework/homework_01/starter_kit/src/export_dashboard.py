"""Export tracker results into a browser-friendly JavaScript payload."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .data import load_measurements, load_sequence_info, load_zones


def grouped_csv(path: Path, converter) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["frame"]].append(converter(row))
    return dict(grouped)


def build_scene(project: Path, results_root: Path, name: str) -> dict:
    data = project / "data" / name
    results = results_root / name
    info = load_sequence_info(data / "seqinfo.yaml")
    measurements = load_measurements(data / "measurements.csv")
    payload = {
        "fps": info.fps,
        "frames": info.frames,
        "width": info.width,
        "height": info.height,
        "video": f"../../data/{name}/video.mp4",
        "zones": load_zones(data / "zones.json"),
        "measurements": {
            str(frame): [[round(d.x, 2), round(d.y, 2), round(d.width, 2), round(d.height, 2)] for d in items]
            for frame, items in measurements.items()
        },
        "tracks": grouped_csv(
            results / "tracks.csv",
            lambda row: [int(row["track_id"]), round(float(row["x"]), 2), round(float(row["y"]), 2), round(float(row["width"]), 2), round(float(row["height"]), 2), row["source"][0], int(row["confirmed"])],
        ),
        "events": [],
        "summary": json.loads((results / "summary.json").read_text(encoding="utf-8")),
        "metrics": json.loads((results / "metrics.json").read_text(encoding="utf-8")) if (results / "metrics.json").exists() else {},
    }
    with (results / "events.csv").open(newline="", encoding="utf-8") as stream:
        payload["events"] = [[int(row["frame"]), int(row["track_id"]), row["event"]] for row in csv.DictReader(stream)]
    gt_path = data / "ground_truth.csv"
    if gt_path.exists():
        payload["groundTruth"] = grouped_csv(
            gt_path,
            lambda row: [int(row["track_id"]), round(float(row["x"]), 2), round(float(row["y"]), 2), round(float(row["width"]), 2), round(float(row["height"]), 2)],
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("starter_kit/interactive/data.js"))
    args = parser.parse_args()
    payload = {"scenes": {name: build_scene(args.project, args.results_root, name) for name in ("debug", "development", "challenge")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("window.HW1_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
