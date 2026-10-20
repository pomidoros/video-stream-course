"""Сохранить сравнения Farnebäck для двух видео."""

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from dense import farneback, flow_arrows, flow_rgb

DIRECTORY = Path(__file__).resolve().parent


def label(image, text):
    """Добавить подпись над BGR-изображением."""
    result = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(result, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    return result


def compare_video(path, gap=5, max_motion=8):
    """Обработать пары с заданным gap и сохранить четыре панели в H.264 MP4."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            frames.append(image)
    finally:
        capture.release()
    if gap < 1 or len(frames) <= gap:
        raise ValueError("В видео должно быть больше кадров, чем заданный gap")

    output = DIRECTORY / 'outputs' / f'{path.stem}_dense_gap{gap}.mp4'
    output.parent.mkdir(exist_ok=True)
    height, width = frames[0].shape[:2]
    # Кодируем сразу в H.264: временный MP4 и повторное преобразование не нужны.
    writer = imageio_ffmpeg.write_frames(
        str(output), (width * 2, (height + 26) * 2 + 30), fps=fps,
        pix_fmt_in='bgr24', codec='libx264', macro_block_size=1,
    )
    writer.send(None)
    try:
        for index in range(len(frames) - gap):
            previous = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
            current = cv2.cvtColor(frames[index + gap], cv2.COLOR_BGR2GRAY)
            flow_one = farneback(previous, current, levels=1)
            flow_pyramid = farneback(previous, current, levels=3)

            # Цветовая шкала одинакова на всех панелях и кадрах.
            panels = [
                label(flow_arrows(frames[index], flow_pyramid), 'Farneback arrows 1:1'),
                label(frames[index], 'Source'),
                label(cv2.cvtColor(flow_rgb(flow_one, max_motion), cv2.COLOR_RGB2BGR), 'Farneback L1'),
                label(cv2.cvtColor(flow_rgb(flow_pyramid, max_motion), cv2.COLOR_RGB2BGR), 'Farneback L3'),
            ]
            comparison = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            comparison = cv2.copyMakeBorder(comparison, 0, 30, 0, 0, cv2.BORDER_CONSTANT)
            caption = f'Hue=direction | brightness 0..{max_motion:g} px/pair'
            cv2.putText(comparison, caption, (7, comparison.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            writer.send(comparison)
            if index == 10:
                cv2.imwrite(str(output.with_suffix('.jpg')), comparison)
    finally:
        writer.close()
    return output


if __name__ == '__main__':
    # Измените gap на 1 для сравнения соседних кадров.
    for path in sorted((DIRECTORY / 'data').glob('*.mp4')):
        print(compare_video(path, gap=5))
