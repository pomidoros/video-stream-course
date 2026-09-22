"""Готовые загрузчики бросков. Детектор и его веса для них не нужны."""

import hashlib
import json
from pathlib import Path
import numpy as np


def load_recordings(root):
    """Прочитать список записей, их пути и разделение train/validation.

    Аргументы:
        root (Path): корень проекта с папкой data.

    Возвращает:
        list[dict]: описания записей из recordings.json.
    """
    return json.loads((Path(root) / "data/recordings.json").read_text())["recordings"]


def load_features(root, row):
    """Загрузить признаки одной записи и проверить целостность файла.

    Аргументы:
        root (Path): корень проекта.
        row (dict): описание записи из load_recordings.

    Возвращает:
        dict: массивы frame, time_s и features [кадры, 70].
    """
    path = Path(root) / row["features"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["feature_sha256"]:
        raise ValueError("Контрольная сумма признаков не совпадает")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in ("frame", "time_s", "features")}
    x, t = result["features"], result["time_s"]
    if (
        x.shape != (len(t), 70)
        or not np.isfinite(x).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("Неверный формат признаков или временных меток")
    return result


def load_attempts(root, row):
    """Прочитать размеченные попытки одной записи.

    Аргументы:
        root (Path): корень проекта.
        row (dict): описание записи из load_recordings.

    Возвращает:
        list[dict]: попытки с временем и исходом. Закрытые метки недоступны.
    """
    if row["annotations"] is None:
        raise ValueError("Метки закрытого теста недоступны")
    path = Path(root) / row["annotations"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["annotation_sha256"]:
        raise ValueError("Контрольная сумма разметки не совпадает")
    return json.loads(path.read_text())["attempts"]


def windows(features, length):
    """Перебирать окна признаков без заглядывания в будущие шаги.

    Аргументы:
        features (np.ndarray): признаки [кадры, признаки].
        length (int): число шагов в окне.

    Генерирует:
        tuple: индекс последнего шага и окно [length, признаки].
    """
    if length < 1:
        raise ValueError("Длина окна должна быть положительной")
    for end in range(length - 1, len(features)):
        yield end, features[end - length + 1 : end + 1]
