"""Разреженный оптический поток Lucas–Kanade и функции для примеров."""

import cv2
import numpy as np


def corners(image, count=80, margin=14):
    """Найти углы и вернуть координаты в массиве формы (N, 2)."""
    # Не выбираем точки у края: вокруг каждой должно помещаться окно.
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[margin:-margin, margin:-margin] = 255
    points = cv2.goodFeaturesToTrack(
        image,              # Серый кадр uint8.
        maxCorners=count,   # Верхняя граница числа точек.
        qualityLevel=0.02,  # Порог относительно качества самого сильного угла.
        minDistance=8,      # Минимальное расстояние между точками, px.
        mask=mask,          # Не выбирать точки у края кадра.
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2)


def lucas_kanade(previous, current, points, window_size=21):
    """Один линейный шаг LK для выбранных точек, без пирамиды и итераций."""
    first = previous.astype(np.float32) / 255
    second = current.astype(np.float32) / 255
    # Собель: float32, производные по x и y, ядро 3×3, нормировка 1/8.
    gx = cv2.Sobel(first, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(first, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    temporal = second - first
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    next_points = points.copy()
    valid = np.zeros(len(points), dtype=bool)
    radius = window_size // 2 + 1
    height, width = first.shape
    for index, (x, y) in enumerate(points):
        # Полное окно и производные должны помещаться в исходный кадр.
        if not (radius <= x < width - radius and radius <= y < height - radius):
            continue
        center = (float(x), float(y))
        size = (window_size, window_size)
        # getRectSubPix вырезает окно с дробным центром, интерполируя значения.
        ix = cv2.getRectSubPix(gx, size, center).ravel()
        iy = cv2.getRectSubPix(gy, size, center).ravel()
        it = cv2.getRectSubPix(temporal, size, center).ravel()
        A = np.column_stack([ix, iy])  # По уравнению Ix*u + Iy*v = -It на пиксель.
        H = A.T @ A                   # Матрица системы 2×2.
        if np.linalg.eigvalsh(H)[0] / window_size**2 < 1e-5:
            continue  # Ровный участок или граница не задают обе компоненты.
        movement = np.linalg.solve(H, -A.T @ it)  # Наименьшие квадраты, без inverse.
        end = points[index] + movement
        if np.isfinite(end).all() and 0 <= end[0] < width and 0 <= end[1] < height:
            next_points[index] = end
            valid[index] = True
    return next_points, valid


def opencv_lk(previous, current, points, window_size=21, levels=3, iterations=30):
    """Найти координаты через PyrLK; вернуть (N, 2) и маску принятых точек."""
    # OpenCV ожидает float32 и форму (N, 1, 2).
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return points.reshape(-1, 2), np.zeros(0, dtype=bool)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None,  # Два серых кадра, точки, выход создаёт OpenCV.
        winSize=(window_size, window_size),  # Локальное окно на каждом уровне.
        maxLevel=levels,  # Максимальный индекс: 0 отключает пирамиду.
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            iterations,  # Предел числа уточнений на уровне.
            0.01,        # Порог длины поправки позиции, px текущего уровня.
        ),
    )
    next_points = next_points.reshape(-1, 2)
    valid = status.ravel() == 1

    # Дополнительно отбрасываем координаты вне второго кадра.
    height, width = current.shape
    valid &= np.isfinite(next_points).all(axis=1)
    valid &= (next_points[:, 0] >= 0) & (next_points[:, 0] < width)
    valid &= (next_points[:, 1] >= 0) & (next_points[:, 1] < height)
    return next_points, valid


def synthetic(dx=1, dy=0.5, kind="texture", brightness=0):
    """Создать пару кадров с известным сдвигом (dx, dy)."""
    height, width = 180, 240
    if kind == "texture":
        # Один seed сохраняет текстуру при изменении параметров.
        rng = np.random.default_rng(7)
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

    # Положительные dx и dy сдвигают изображение вправо и вниз.
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    current = cv2.warpAffine(
        previous, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101
    )
    current = np.clip(current.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
    return previous, current


def read_pair(path, frame=0, gap=1):
    """Прочитать BGR-кадры с номерами frame и frame + gap."""
    if frame < 0 or gap < 1:
        raise ValueError("Номер кадра должен быть >= 0, интервал — >= 1")

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


def arrows(image, points, next_points, valid, label=""):
    """Нарисовать принятые точки и векторы смещения; вернуть BGR-изображение."""
    result = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2 else image.copy()
    )
    # Начальные точки жёлтые, смещения зелёные. Масштаб стрелок — 1:1.
    for start, end in zip(points[valid], next_points[valid]):
        start = tuple(np.rint(start).astype(int))
        end = tuple(np.rint(end).astype(int))
        cv2.circle(result, start, 2, (0, 220, 255), -1)
        cv2.arrowedLine(result, start, end, (70, 255, 90), 1, tipLength=0.3)

    if label:
        cv2.rectangle(result, (0, 0), (result.shape[1], 25), (25, 25, 25), -1)
        cv2.putText(
            result, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return result
