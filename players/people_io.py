"""Готовые загрузчики игровых данных. Изменять их не требуется."""

import hashlib
import json
from pathlib import Path
import numpy as np


def load_manifest(root):
    """Прочитать описание игровых клипов.

    Аргументы:
        root (Path): папка players.

    Возвращает:
        dict: описание набора; список клипов находится в ключе clips.
    """
    return json.loads((Path(root) / "data/manifest.json").read_text())


def load_clip(root, clip):
    """Загрузить детекции и признаки внешности одного клипа.

    Аргументы:
        root (Path): папка players.
        clip (dict): описание клипа из manifest.

    Возвращает:
        dict: frames, det_frame, bbox, confidence, embeddings.
    """
    p = Path(root) / "data" / f"{clip['name']}.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != clip["cache_sha256"]:
        raise ValueError("Контрольная сумма детекций не совпадает")
    return dict(np.load(p, allow_pickle=False))


def load_labels(root, clip):
    """Загрузить правильные рамки и ID для внешней оценки.

    Аргументы:
        root (Path): папка players.
        clip (dict): описание клипа из manifest.

    Возвращает:
        list[dict]: разметка кадров. Не передавайте её трекеру.
    """
    p = Path(root) / "data" / clip["gt_path"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != clip["gt_sha256"]:
        raise ValueError("Контрольная сумма разметки не совпадает")
    return json.loads(p.read_text())
