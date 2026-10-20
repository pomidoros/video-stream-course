"""Готовый запуск: видео → три клипа → модель → таблица, график и видео."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

import homework

ROOT = Path(__file__).resolve().parent
VARIANTS = ("original", "shuffled", "frozen")


def read_video(path):
    """Прочитать RGB-кадры и фактическую частоту видео через OpenCV."""
    capture = cv2.VideoCapture(str(path))
    frames, fps = [], capture.get(cv2.CAP_PROP_FPS)
    try:
        if not capture.isOpened() or fps <= 0:
            raise ValueError(f"Не удалось открыть видео: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"В видео нет кадров: {path}")
    return np.stack(frames), fps


def save_csv(path, rows):
    """Сохранить словари с одинаковыми полями в CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visualize(examples, results, output):
    """Нарисовать сравнение и подписать исходные видео ответами модели."""
    targets = [example[0]["target"] for example in examples]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(targets))
    for variant, offset, title in [("original", -0.25, "Исходный порядок"),
                                   ("shuffled", 0, "Перемешанные кадры"),
                                   ("frozen", 0.25, "Повтор одного кадра")]:
        values = [r["target_score"] for r in results if r["variant"] == variant]
        bars = axis.bar(positions + offset, values, 0.24, label=title)
        axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set(xticks=positions, xticklabels=targets, ylim=(0, 1.18),
             ylabel="Softmax-оценка правильного класса",
             title="Как временная структура влияет на ответ модели?")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)

    writer = cv2.VideoWriter(str(output / "annotated_examples.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать MP4")
    try:
        for item, frames, fps in examples:
            predicted = next(r for r in results if r["video"] == item["file"]
                             and r["variant"] == "original")
            # Частота вывода фиксирована; исходные кадры выбираются по времени.
            for index in range(round(len(frames) / fps * 15)):
                frame = frames[min(round(index / 15 * fps), len(frames) - 1)]
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                canvas[120:] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 360))
                lines = [f"Expected: {item['target']}",
                         f"Predicted: {predicted['top1']} ({predicted['top1_score']:.2f})",
                         "Video-level answer from 3 clips; not event boundaries"]
                for row, line in enumerate(lines):
                    cv2.putText(canvas, line, (12, 28 + row * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.51,
                                (230, 230, 230), 1, cv2.LINE_AA)
                writer.write(canvas)
    finally:
        writer.release()


def render_comparison(comparisons, path):
    """Показать именно поданные сети клипы с их отдельными предсказаниями."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1152, 360))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать сравнительное видео")
    titles = {"original": "ORIGINAL", "shuffled": "SHUFFLED", "frozen": "FROZEN"}
    groups = dict.fromkeys((item["file"], number) for item, number, *_ in comparisons)
    try:
        for video, number in groups:
            panels = {variant: (item, clip, row) for item, n, variant, clip, row in comparisons
                      if item["file"] == video and n == number}
            length = panels["original"][1].shape[0]
            # Два повтора только для просмотра: модель увидела каждый клип один раз.
            for step in range(length * 2):
                canvas = np.zeros((360, 1152, 3), dtype=np.uint8)
                for column, variant in enumerate(VARIANTS):
                    item, clip, row = panels[variant]
                    left = column * 384
                    frame = clip[step % length].permute(1, 2, 0).numpy()
                    canvas[104:320, left:left + 384] = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (384, 216))
                    lines = [titles[variant] + f" | clip {number + 1}/3",
                             f"Top-1: {row['top1']}",
                             f"P({item['target']}): {row['target_score']:.4f}"]
                    for line_number, line in enumerate(lines):
                        cv2.putText(canvas, line, (left + 10, 24 + 29 * line_number),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
                    footer = f"Source {row['start_s']:.2f}s | loop {step // length + 1}/2"
                    cv2.putText(canvas, footer, (left + 10, 346),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
                writer.write(canvas)
    finally:
        writer.release()


def main():
    """Выполнить эксперимент с реализацией студента и сохранить артефакты."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed перемешивания кадров")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--weights", type=Path, help="Локальный файл весов для запуска без сети")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    weights = R3D_18_Weights.KINETICS400_V1
    if args.weights:
        model = r3d_18(weights=None)
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    else:
        model = r3d_18(weights=weights)
    preprocess, categories = weights.transforms(), weights.meta["categories"]
    model = model.cpu()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    results, window_results, examples, comparisons = [], [], [], []
    started = time.perf_counter()
    for item in manifest["videos"]:
        frames, fps = read_video(ROOT / "data" / item["file"])
        examples.append((item, frames, fps))
        last_start = (len(frames) - 1) / fps - 15 / 15
        if last_start < 0:
            raise ValueError("Видео короче одного клипа из 16 кадров")
        starts = np.linspace(0, last_start, 3)
        clips = []
        for start in starts:
            indices = homework.sample_indices(len(frames), fps, float(start))
            clips.append(torch.from_numpy(frames[indices]).permute(0, 3, 1, 2))
        for variant in VARIANTS:
            logits = []
            for number, clip in enumerate(clips):
                if variant != "original":
                    clip = homework.transform_clip(clip, variant, seed=args.seed)
                batch = preprocess(clip).unsqueeze(0)  # 1 × C × T × H × W
                scores = homework.predict_logits(model, batch)
                if scores.shape != (1, len(categories)):
                    raise ValueError("predict_logits должна вернуть тензор формы (B, K)")
                logits.append(scores[0])
                index = int(scores[0].argmax())
                clip_probabilities = scores[0].softmax(dim=0)
                window_results.append({"video": item["file"], "variant": variant,
                                       "clip": number, "start_s": float(starts[number]),
                                       "top1": categories[index],
                                       "top1_score": float(clip_probabilities[index]),
                                       "target_score": float(clip_probabilities[categories.index(item["target"])])})
                comparisons.append((item, number, variant, clip.clone(), window_results[-1]))
            probabilities = homework.aggregate_logits(torch.stack(logits))
            if probabilities.shape != (len(categories),) or not torch.isfinite(probabilities).all():
                raise ValueError("aggregate_logits должна вернуть вероятности формы (K,)")
            top_indices = probabilities.topk(3).indices.tolist()
            top1 = categories[top_indices[0]]
            results.append({"video": item["file"], "target": item["target"],
                            "variant": variant, "top1": top1,
                            "top1_score": float(probabilities[top_indices[0]]),
                            "target_score": float(probabilities[categories.index(item["target"])]),
                            "correct": int(top1 == item["target"]),
                            "top3": " | ".join(categories[i] for i in top_indices)})
            print(f"{item['file']:14s} {variant:8s} → {top1}", flush=True)
    inference_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "predictions.csv", results)
    save_csv(args.output / "clips.csv", window_results)
    metrics = {variant: {"correct": sum(r["correct"] for r in results if r["variant"] == variant),
                         "total": len(examples)} for variant in VARIANTS}
    metrics["processing_seconds"] = round(inference_seconds, 3)
    metrics["seed"] = args.seed
    metrics["device"] = "cpu"
    metrics["weights"] = "R3D_18_Weights.KINETICS400_V1"
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    visualize(examples, results, args.output)
    render_comparison(comparisons, args.output / "comparison.mp4")
    print(f"Готово: {args.output}. Обработка без загрузки весов: {inference_seconds:.1f} с.")


if __name__ == "__main__":
    main()
