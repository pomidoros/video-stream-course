"""Farnebäck и функции для работы с полями смещений."""

import cv2
import numpy as np


def farneback(previous, current, window_size=21, levels=3, iterations=5):
    """Готовый плотный поток OpenCV; у результата нет отдельного status."""
    return cv2.calcOpticalFlowFarneback(
        previous, current, None,  # Два серых кадра; выходной массив создаёт OpenCV.
        pyr_scale=0.5,            # Следующий уровень вдвое меньше по каждой оси.
        levels=levels,           # Число уровней, включая исходный; 1 — без пирамиды.
        winsize=window_size,     # Окно усреднения оценок движения.
        iterations=iterations,   # Число уточнений на каждом уровне.
        poly_n=5,                # Окрестность полиномиального приближения яркости.
        poly_sigma=1.1,          # Сглаживание производных для этого приближения.
        flags=0,                 # Без начального поля и дополнительного Gaussian-флага.
    )


def flow_rgb(flow, max_motion=8, valid=None):
    """Тон — направление, яркость — модуль; отвергнутые пиксели серые."""
    if max_motion <= 0:
        raise ValueError("Верхняя граница шкалы должна быть положительной")
    finite = np.isfinite(flow).all(axis=-1)
    safe_flow = np.where(finite[..., None], flow, 0)
    angle = np.mod(np.arctan2(safe_flow[..., 1], safe_flow[..., 0]), 2 * np.pi)
    magnitude = np.linalg.norm(safe_flow, axis=-1)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    # OpenCV хранит тон uint8 в диапазоне 0…179, а насыщенность и яркость — 0…255.
    hsv[..., 0] = np.floor(angle * (180 / (2 * np.pi))).astype(np.uint8)
    hsv[..., 1] = 255
    # Шкала одна для всех кадров; значения выше max_motion насыщаются.
    hsv[..., 2] = np.rint(np.minimum(magnitude / max_motion, 1) * 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask = finite if valid is None else finite & valid
    image[~mask] = (65, 65, 65)
    return image


def flow_arrows(image, flow, valid=None, step=12):
    """Нарисовать разреженную сетку стрелок масштаба 1:1; вернуть BGR."""
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    height, width = flow.shape[:2]
    # Прореживаем только отрисовку: само поле остаётся плотным.
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if valid is not None and not valid[y, x]:
                continue
            if not np.isfinite(flow[y, x]).all():
                continue
            end = tuple(np.rint([x, y] + flow[y, x]).astype(int))
            cv2.arrowedLine(result, (x, y), end, (40, 255, 70), 1, tipLength=0.3)
            cv2.circle(result, (x, y), 1, (0, 220, 255), -1)
    return result


def warp_next(current, flow):
    """Собрать первый кадр, выбирая яркости второго по координатам x+u, y+v."""
    height, width = current.shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    inside = (
        np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0) & (map_x < width - 1)
        & (map_y >= 0) & (map_y < height - 1)
    )
    warped = cv2.remap(
        current, map_x, map_y,          # Откуда брать значение для каждой выходной позиции.
        interpolation=cv2.INTER_LINEAR, # Билинейная интерполяция дробных координат.
        borderMode=cv2.BORDER_CONSTANT, # Вышедшие за кадр позиции заполняются нулями.
    )
    return warped, inside


def synthetic(dx=0.5, dy=0.25, kind="texture", brightness=0):
    """Создать два серых кадра с известным переносом и изменением яркости."""
    height, width = 180, 240
    if kind == "texture":
        rng = np.random.default_rng(7)  # Одна и та же текстура при смене параметров.
        noise = rng.uniform(0, 255, (height, width)).astype(np.float32)
        previous = cv2.GaussianBlur(noise, (0, 0), 1.4)
        previous = np.clip((previous - 128) * 3 + 128, 0, 255).astype(np.uint8)
    elif kind in ("corner", "edge"):
        previous = np.full((height, width), 35, dtype=np.uint8)
        if kind == "corner":
            previous[90:, 120:] = 220
        else:
            previous[:, 120:] = 220
        previous = cv2.GaussianBlur(previous, (0, 0), 1.2)
    elif kind == "flat":
        previous = np.full((height, width), 128, dtype=np.uint8)
    else:
        raise ValueError("Неизвестный тип изображения")

    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    current = cv2.warpAffine(
        previous, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101
    )
    current = np.clip(current.astype(float) + brightness, 0, 255).astype(np.uint8)
    return previous, current


def read_pair(path, frame=0, gap=1):
    """Вернуть BGR-кадры с номерами frame и frame + gap."""
    if frame < 0 or gap < 1:
        raise ValueError("Нужны frame >= 0 и gap >= 1")
    capture = cv2.VideoCapture(str(path))
    images = []
    try:
        for index in (frame, frame + gap):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, image = capture.read()
            if not success:
                raise ValueError(f"Не удалось прочитать кадр {index}: {path}")
            images.append(image)
    finally:
        capture.release()
    return images
