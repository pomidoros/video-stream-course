import argparse
import json
from pathlib import Path
from project.common import ROOT, dump
from project import pipeline
from project.thresholds import checks


def main():
    p = argparse.ArgumentParser(description="Student basketball video project")
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "run",
        "train",
        "evaluate",
        "predict",
        "video",
        "render-people",
        "render-ball",
    ]:
        s = sub.add_parser(name)
        s.add_argument("--data", type=Path, default=ROOT)
        s.add_argument("--output", type=Path, default=ROOT / "outputs")
        s.add_argument("--models", type=Path, default=ROOT / "artifacts")
        if name == "train":
            s.add_argument("--epochs", type=int, default=120)
            s.add_argument("--seed", type=int, default=17)
        if name == "predict":
            s.add_argument("--split", default="validation")
            s.add_argument("--kind", choices=["last", "mean", "gru"], default="gru")
        if name.startswith("render-"):
            s.add_argument("--results", type=Path, default=ROOT / "outputs")
            s.add_argument("--max-frames", type=int, default=None)
        if name == "render-people":
            s.add_argument("--clip", default="basket_S6T2_post")
            s.add_argument(
                "--mode", choices=["prediction", "comparison"], default="prediction"
            )
        if name == "render-ball":
            s.add_argument("--recording", default="indoor_09")
    a = p.parse_args()
    if a.command == "run":
        from project.visualization import render_people, render_ball

        from project.practice import video_exercise

        exercise = video_exercise(a.data, a.output / "video")
        training = pipeline.train(a.data, a.models)
        metrics = pipeline.evaluate(a.data, a.models, a.output)
        thresholds = checks(metrics)
        dump(a.output / "threshold_checks.json", thresholds)
        videos = [
            render_ball(a.data, a.output, a.output / "videos", "indoor_09"),
            render_people(a.data, a.output, a.output / "videos", "basket_S6T2_post"),
        ]
        result = dict(
            training=training,
            thresholds=thresholds,
            videos=videos,
            video_exercise=exercise,
        )
        dump(a.output / "run.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if all(row["passed"] for row in thresholds) else 1)
    elif a.command == "video":
        from project.practice import video_exercise

        result = video_exercise(a.data, a.output / "video")
    elif a.command == "train":
        result = pipeline.train(a.data, a.models, a.seed, a.epochs)
    elif a.command == "predict":
        result = {
            k: len(v)
            for k, v in pipeline.predict(
                a.data, a.models, a.output, a.split, a.kind
            ).items()
        }
    elif a.command == "evaluate":
        result = pipeline.evaluate(a.data, a.models, a.output)
        out = checks(result)
        dump(a.output / "threshold_checks.json", out)
        for r in out:
            print(
                ("PASS" if r["passed"] else "FAIL"),
                r["metric"],
                r["value"],
                r["operator"],
                r["threshold"],
            )
        raise SystemExit(0 if all(r["passed"] for r in out) else 1)
    else:
        from project.visualization import render_people, render_ball

        if a.command == "render-people":
            result = render_people(
                a.data, a.results, a.output, a.clip, a.mode, a.max_frames
            )
        else:
            result = render_ball(a.data, a.results, a.output, a.recording, a.max_frames)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
