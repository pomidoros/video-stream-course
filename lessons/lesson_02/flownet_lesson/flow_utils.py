"""Чтение кадров, классический поток и визуализация смещений."""

import cv2
import numpy as np


def farneback(previous, current):
    """Плотный поток OpenCV для сравнения с нейросетями."""
    return cv2.calcOpticalFlowFarneback(
        previous, current, None,  # Два серых кадра; выходной массив создаёт OpenCV.
        pyr_scale=0.5,   # Каждый следующий масштаб вдвое меньше.
        levels=3,       # До трёх уровней, включая исходный.
        winsize=21,     # Размер локального окна усреднения.
        iterations=5,   # Число уточнений на каждом уровне.
        poly_n=5,       # Размер окрестности полиномиального приближения.
        poly_sigma=1.1, # Сглаживание для этого приближения.
        flags=0,
    )


def flow_rgb(flow, max_motion=8):
    """Цвет — направление, яркость — модуль от 0 до max_motion px за пару."""
    if max_motion <= 0:
        raise ValueError("Верхняя граница шкалы должна быть положительной")
    valid = np.isfinite(flow).all(axis=-1)
    safe_flow = np.where(valid[..., None], flow, 0)
    angle = np.mod(np.arctan2(safe_flow[..., 1], safe_flow[..., 0]), 2 * np.pi)
    magnitude = np.linalg.norm(safe_flow, axis=-1)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    # В uint8-представлении OpenCV тон H занимает диапазон 0…179.
    hsv[..., 0] = np.floor(angle * 180 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    # Шкала общая для кадров: не нормируем каждый результат на свой максимум.
    hsv[..., 2] = np.rint(np.minimum(magnitude / max_motion, 1) * 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    image[~valid] = (65, 65, 65)
    return image


def flow_arrows(image, flow, step=12):
    """Нарисовать каждую step-ю стрелку на RGB-кадре в масштабе 1:1."""
    result = image.copy()
    height, width = flow.shape[:2]
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if not np.isfinite(flow[y, x]).all():
                continue
            end = tuple(np.rint([x, y] + flow[y, x]).astype(int))
            cv2.arrowedLine(result, (x, y), end, (40, 255, 70), 1, tipLength=0.3)
    return result


def texture():
    """Создать одну воспроизводимую текстуру для проверки переноса."""
    rng = np.random.default_rng(7)
    noise = rng.uniform(0, 255, (180, 240)).astype(np.float32)
    image = cv2.GaussianBlur(noise, (0, 0), 1.4)
    return np.clip((image - 128) * 3 + 128, 0, 255).astype(np.uint8)


def read_pair(path, frame=0, gap=1):
    """Вернуть два RGB-кадра с номерами frame и frame + gap."""
    if frame < 0 or gap < 1:
        raise ValueError("Нужны frame >= 0 и gap >= 1")
    capture = cv2.VideoCapture(str(path))
    images = []
    try:
        for index in (frame, frame + gap):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)  # Номер исходного кадра.
            success, image = capture.read()
            if not success:
                raise ValueError(f"Не удалось прочитать кадр {index}: {path}")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return images
