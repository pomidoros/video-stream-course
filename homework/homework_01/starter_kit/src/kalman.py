from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class KalmanCV:
    """Фильтр Калмана с постоянной скоростью для состояния [cx, cy, vx, vy].

    Параметры
    ----------
    center : Iterable[float]
        Начальный центр объекта в пикселях.
    process_var : float
        Дисперсия шума движения.
    measurement_var : float
        Дисперсия ошибки измерения координат.

    Атрибуты
    --------
    x : np.ndarray
        Состояние формы (4,); скорости измеряются в пикселях за кадр.
    P : np.ndarray
        Ковариация состояния формы (4, 4).
    """

    def __init__(self, center: Iterable[float], process_var: float, measurement_var: float) -> None:
        cx, cy = center
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([100.0, 100.0, 400.0, 400.0])
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(2) * measurement_var
        self.process_var = float(process_var)

    @property
    def center(self) -> np.ndarray:
        """Вернуть копию центра объекта формы (2,)."""
        return self.x[:2].copy()

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Спрогнозировать состояние через dt кадров и вернуть центр (2,).

        Обновите x и P. Матрицы перехода F и шума Q зависят от dt.
        """
        # TODO(student): составьте F и Q, выполните прогноз x и P.
        raise NotImplementedError

    def update(self, measurement: Iterable[float]) -> np.ndarray:
        """Уточнить состояние по измеренному центру и вернуть центр (2,).

        measurement содержит две координаты в пикселях. Обновите x и P;
        для ковариации используйте форму Джозефа из материала курса.
        """
        # TODO(student): невязка, S, коэффициент K, коррекция x и P.
        raise NotImplementedError
