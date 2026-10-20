"""Сохранить сравнение OpenCV PyrLK без пирамиды и с ней."""

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from lk import arrows, corners, opencv_lk

DIRECTORY = Path(__file__).resolve().parent


def compare_video(path, gap=1):
    """Обработать пары кадров с заданным интервалом и сохранить MP4."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()

    if gap < 1 or len(frames) <= gap:
        raise ValueError("В видео должно быть больше кадров, чем заданный интервал")

    output = DIRECTORY / "outputs" / f"{path.stem}_gap{gap}.mp4"
    output.parent.mkdir(exist_ok=True)
    height, width = frames[0].shape[:2]
    # H.264 воспроизводится прямо в notebook; две панели стоят рядом.
    writer = imageio_ffmpeg.write_frames(
        str(output), (width * 2, height), fps=fps,
        pix_fmt_in="bgr24", codec="libx264", macro_block_size=1,
    )
    writer.send(None)  # Запускаем процесс кодирования.
    try:
        for index in range(len(frames) - gap):
            previous = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
            current = cv2.cvtColor(frames[index + gap], cv2.COLOR_BGR2GRAY)
            points = corners(previous)

            # Все методы получают одни и те же начальные точки.
            cv_points, cv_valid = opencv_lk(previous, current, points, levels=0)
            pyr_points, pyr_valid = opencv_lk(previous, current, points, levels=3)
            results = [
                ("OpenCV L0", cv_points, cv_valid),
                ("OpenCV L3", pyr_points, pyr_valid),
            ]

            panels = []
            for name, next_points, valid in results:
                label = f"{name}: {valid.sum()}/{len(points)}"
                panels.append(arrows(frames[index], points, next_points, valid, label))
            comparison = np.hstack(panels)
            writer.send(comparison)
            if index == 10:
                cv2.imwrite(str(output.with_suffix(".jpg")), comparison)
    finally:
        writer.close()
    return output


if __name__ == "__main__":
    # Измените gap на 5, чтобы увеличить смещение между кадрами.
    for path in sorted((DIRECTORY / "data").glob("*.mp4")):
        print(compare_video(path, gap=1))
