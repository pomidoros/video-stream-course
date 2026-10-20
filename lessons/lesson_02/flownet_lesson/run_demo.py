"""Сохранить сравнения Farnebäck, FlowNetS и FlowNetC на двух видео."""

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from flow_utils import farneback, flow_rgb
from inference import load_model, predict

DIRECTORY = Path(__file__).resolve().parent


def label(image, text):
    """Добавить подпись над RGB-изображением."""
    result = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(result, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return result


def compare_video(path, model_s, model_c, gap=5, stride=3, max_motion=8,
                  div_flow_s=20, div_flow_c=20):
    """Обработать каждый stride-й кадр; модели передаются уже загруженными."""
    if gap < 1 or stride < 1:
        raise ValueError("Интервалы gap и stride должны быть положительными")
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) <= gap:
        raise ValueError("В видео недостаточно кадров для указанного gap")

    output = DIRECTORY / "outputs" / f"{path.stem}_flownet_gap{gap}.mp4"
    output.parent.mkdir(exist_ok=True)
    height, width = frames[0].shape[:2]
    # Уменьшаем fps вместе с числом кадров, чтобы сохранить примерно исходный темп.
    writer = imageio_ffmpeg.write_frames(
        str(output), (width * 2, (height + 26) * 2 + 30), fps=fps / stride,
        codec="libx264", macro_block_size=1,
    )
    writer.send(None)
    try:
        for index in range(0, len(frames) - gap, stride):
            first, second = frames[index], frames[index + gap]
            previous = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
            current = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
            # Одна и та же пара для всех трёх методов.
            flow_cv = farneback(previous, current)
            flow_s = predict(model_s, first, second, div_flow_s)
            flow_c = predict(model_c, first, second, div_flow_c)

            panels = [
                label(first, f'Frame {index} -> {index + gap}'),
                label(flow_rgb(flow_cv, max_motion), 'Farneback'),
                label(flow_rgb(flow_s, max_motion), 'FlowNetS'),
                label(flow_rgb(flow_c, max_motion), 'FlowNetC'),
            ]
            comparison = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            caption = f'Hue=direction | brightness 0..{max_motion:g} px/pair'
            comparison = cv2.copyMakeBorder(comparison, 0, 30, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(comparison, caption, (7, comparison.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            writer.send(comparison)
            if index == 30:
                cv2.imwrite(str(output.with_suffix('.jpg')), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    finally:
        writer.close()
    return output


if __name__ == '__main__':
    # Загружаем веса один раз, а не для каждой пары кадров.
    model_s, div_flow_s = load_model('flownets')
    model_c, div_flow_c = load_model('flownetc')
    for path in sorted((DIRECTORY / 'data').glob('*.mp4')):
        print(compare_video(path, model_s, model_c, gap=5,
                            div_flow_s=div_flow_s, div_flow_c=div_flow_c))
