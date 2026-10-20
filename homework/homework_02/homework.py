"""Модуль 5: распознавание действий и проверка роли движения.

Допишите четыре TODO. Чтение видео, загрузка весов и визуализация готовы.
"""

import numpy as np
import torch


def sample_indices(
    frame_count: int, fps: float, start_s: float,
    clip_len: int = 16, sample_fps: float = 15.0,
) -> np.ndarray:
    """Выбрать номера кадров клипа по временным меткам.

    Параметры
    ----------
    frame_count : int
        Число кадров исходного видео. Нумерация начинается с нуля.
    fps : float
        Частота исходного видео.
    start_s : float
        Время первого кадра клипа в секундах.
    clip_len : int
        Число выбираемых кадров.
    sample_fps : float
        Частота выборки кадров клипа, кадров/с.

    Возвращает
    ----------
    np.ndarray
        Целочисленный массив формы (clip_len,): индексы равномерно
        выбранных по времени кадров начиная с start_s.
        Округление до ближайшего кадра — по правилу np.rint.
        При выходе индексов за границы видео выбросите ValueError.
    """
    if frame_count < 1 or fps <= 0 or sample_fps <= 0 or clip_len < 1 or start_s < 0:
        raise ValueError("Некорректные параметры клипа")
    # TODO 1: выбор кадров.
    raise NotImplementedError


def predict_logits(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Получить логиты классов без обучения модели.

    Параметры
    ----------
    model : torch.nn.Module
        Готовая сеть; должна работать в режиме оценки.
    batch : torch.Tensor
        Уже нормализованные клипы формы (B, C, T, H, W).

    Возвращает
    ----------
    torch.Tensor
        Логиты формы (B, K), без softmax и вычисления градиентов.
    """
    # TODO 2: инференс.
    raise NotImplementedError


def aggregate_logits(clip_logits: torch.Tensor) -> torch.Tensor:
    """Объединить ответы нескольких клипов одного видео.

    Параметры
    ----------
    clip_logits : torch.Tensor
        Логиты формы (N, K): N клипов, K классов.

    Возвращает
    ----------
    torch.Tensor
        Вероятности формы (K,): softmax усреднённых по клипам логитов.
    """
    # TODO 3: ответ для видео.
    raise NotImplementedError


def transform_clip(clip: torch.Tensor, mode: str, seed: int = 42) -> torch.Tensor:
    """Изменить временную структуру клипа для контрольного эксперимента.

    Параметры
    ----------
    clip : torch.Tensor
        RGB-клип формы (T, C, H, W), до нормализации.
    mode : str
        "shuffled": переставить те же кадры вдоль оси T.
        "frozen": повторить центральный кадр T раз.
    seed : int
        Seed локального генератора PyTorch для воспроизводимой перестановки.
        Перестановку формируйте через torch.randperm.

    Возвращает
    ----------
    torch.Tensor
        Клип той же формы и типа. Исходный тензор не изменяется.
        В shuffled каждый исходный кадр встречается ровно один раз;
        глобальное состояние генератора случайных чисел не меняется.
        В frozen используется кадр с индексом T // 2.
        Для неизвестного mode выбросите ValueError.
    """
    # TODO 4: временные преобразования.
    raise NotImplementedError
