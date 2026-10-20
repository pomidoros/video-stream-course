"""Готовая обработка видео. Допишите один блок TODO внутри цикла треков."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .counter import LineCounter
from .data import load_config, load_measurements, load_sequence_info, load_zones
from .tracker import MultiObjectTracker
from .video_reader import create_writer, read_video
from .visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/easy.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = load_sequence_info(args.data / "seqinfo.yaml")
    zones = load_zones(args.data / "zones.json")
    config = load_config(args.config)
    measurements = load_measurements(args.data / "measurements.csv")
    tracker = MultiObjectTracker(**config["tracker"], **config["kalman"])
    counter = LineCounter(tuple(zones["entry_line"][0]), tuple(zones["entry_line"][1]), zones["entry_direction"], zones.get("initial_occupancy", 0))
    visualizer = Visualizer()
    args.output.mkdir(parents=True, exist_ok=True)
    writer = create_writer(args.output / "demo.mp4", info.fps, (info.width, info.height))
    track_stream = (args.output / "tracks.csv").open("w", newline="", encoding="utf-8")
    event_stream = (args.output / "events.csv").open("w", newline="", encoding="utf-8")
    track_csv, event_csv = csv.writer(track_stream), csv.writer(event_stream)
    track_csv.writerow(["frame", "track_id", "x", "y", "width", "height", "source", "confirmed"])
    event_csv.writerow(["frame", "track_id", "event"])
    recovered_ids: set[int] = set()
    try:
        for frame, image in read_video(args.data / "video.mp4"):
            detections = measurements.get(frame, [])
            tracks = tracker.step(detections, dt=1.0)
            for track in tracks:
                x, y, width, height = track.bbox
                track_csv.writerow([frame, track.track_id, x, y, width, height, track.source, int(track.confirmed)])
                if track.recovered:
                    recovered_ids.add(track.track_id)
                if track.confirmed:
                    # TODO(student): передайте центр и признак измерения в counter.observe;
                    # запишите полученное событие: frame, track_id, event.
                    pass
            canvas = visualizer.draw(image, detections, tracks, zones["entry_line"], counter.entered, counter.exited, counter.current, len(recovered_ids))
            writer.write(canvas)
    finally:
        writer.release()
        track_stream.close()
        event_stream.close()
    summary = {"entered": counter.entered, "exited": counter.exited, "current": counter.current, "recovered_tracks": len(recovered_ids)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
