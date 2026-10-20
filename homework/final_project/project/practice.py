"""Готовые запуск и иллюстрации дополнительных учебных экспериментов."""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from project.common import ROOT, dump, video_solution
from players.people_io import load_manifest


def video_exercise(root=ROOT, output=ROOT / "outputs/video"):
    """Применить студенческие функции к двум коротким фрагментам видео."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    solution = video_solution()
    report = {}
    for clip in load_manifest(root / "players")["clips"][:2]:
        frames = solution.read_frames(
            root / "players" / clip["source_video"], stride=2, width=640, max_frames=30
        )
        previous = None
        summary = []
        for index, current in enumerate(frames):
            if index >= 30 or current["frame"] != index * 2:
                raise ValueError("Неверные номера кадров или ограничение чтения")
            if not np.isclose(current["time_s"], current["frame"] / clip["fps"]):
                raise ValueError("Используйте FPS исходного видео")
            if current["image"].shape != (360, 640, 3):
                raise ValueError("Неверный размер уменьшенного кадра")
            if previous is not None:
                mask = solution.motion_mask(previous["image"], current["image"])
                flow = solution.dense_flow(previous["image"], current["image"])
                if (
                    mask.shape != (360, 640)
                    or mask.dtype != np.uint8
                    or not np.isin(mask, [0, 255]).all()
                ):
                    raise ValueError("Ожидается бинарная маска uint8")
                if flow.shape != (360, 640, 2) or not np.isfinite(flow).all():
                    raise ValueError("Неверный формат потока")
                metrics = solution.summarize_motion(
                    mask, flow, current["time_s"] - previous["time_s"]
                )
                if (
                    not 0 <= metrics["moving_fraction"] <= 1
                    or not np.isfinite(metrics["median_speed_px_s"])
                    or metrics["median_speed_px_s"] < 0
                ):
                    raise ValueError("Неверные показатели движения")
                summary.append(
                    dict(frame=current["frame"], time_s=current["time_s"], **metrics)
                )
                if index == 15:
                    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
                    axes[0].imshow(cv2.cvtColor(current["image"], cv2.COLOR_BGR2RGB))
                    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
                    axes[2].imshow(np.linalg.norm(flow, axis=-1), cmap="magma")
                    for axis, title in zip(
                        axes,
                        ["Исходный кадр", "Маска движения", "Смещение, пиксели за шаг"],
                    ):
                        axis.set_title(title)
                        axis.axis("off")
                    figure.text(
                        0.01,
                        0.01,
                        "TrackID3x3 · Yamada et al. · CC BY 4.0 · Учебная обработка",
                        fontsize=8,
                    )
                    figure.tight_layout(rect=(0, 0.05, 1, 1))
                    figure.savefig(output / f"{clip['name']}.motion.png", dpi=120)
                    plt.close(figure)
            previous = current
        if len(summary) != 29:
            raise ValueError("Нужно прочитать 30 кадров каждого фрагмента")
        report[clip["name"]] = summary
    dump(output / "motion.json", report)
    return dict(clips=len(report), pairs_per_clip=29, output=str(output))


def save_statistics(statistics, output):
    """Сохранить вычисленные студентом числа и нарисовать все тепловые карты."""
    output = Path(output)
    dump(Path(str(output) + ".json"), statistics)
    if not statistics:
        return
    identities = sorted(statistics, key=int)
    figure, axes = plt.subplots(
        (len(identities) + 3) // 4,
        4,
        figsize=(12, 3 * ((len(identities) + 3) // 4)),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, identity in zip(axes.flat, identities):
        item = statistics[identity]
        heatmap = np.asarray(item["heatmap"])
        if (
            heatmap.shape != (8, 12)
            or not np.isfinite(heatmap).all()
            or (heatmap < 0).any()
        ):
            raise ValueError(
                "Тепловая карта должна иметь форму [8, 12] и конечные счётчики"
            )
        if heatmap.sum() != item["observed_frames"]:
            raise ValueError("Число посещений должно совпадать с числом наблюдений")
        axis.imshow(heatmap, cmap="magma", interpolation="nearest")
        axis.set_title(
            f"ID {identity}: {item['observed_time_s']:.1f} с\nПуть {item['distance_px']:.0f} px; {item['mean_speed_px_s']:.0f} px/с"
        )
    figure.suptitle("Посещения в плоскости изображения; это не координаты площадки")
    figure.text(
        0.01,
        0.01,
        "Светлее — чаще; масштаб каждого ID свой. Источник: TrackID3x3 · CC BY 4.0",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(Path(str(output) + ".png"), dpi=120)
    plt.close(figure)
