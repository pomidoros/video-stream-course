from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data import Detection


@dataclass(frozen=True)
class TrackView:
    """Состояние трека для счётчика, записи CSV и визуализации.

    bbox — (x, y, width, height), center — (cx, cy), координаты в пикселях.
    source — "measurement" или "prediction"; confirmed — подтверждён ли трек.
    missed — число последовательных пропусков; recovered — вернулось ли
    измерение после пропуска именно в текущем кадре.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    source: str
    confirmed: bool
    missed: int
    recovered: bool = False


class MultiObjectTracker:
    """Сопровождать объекты по центрам обнаружений.

    Параметры
    ----------
    min_hits : int
        Число успешных наблюдений для подтверждения трека, включая первое.
    max_missed : int
        Допустимое число последовательных кадров без измерения.
        Удаляйте трек, когда missed становится больше max_missed.
    max_distance : float
        Максимальное расстояние между прогнозом и измерением в пикселях.
    process_var, measurement_var : float
        Параметры фильтра Калмана.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_missed: int = 15,
        max_distance: float = 80.0,
        process_var: float = 25.0,
        measurement_var: float = 36.0,
    ) -> None:
        # TODO(student): проверьте и сохраните параметры, создайте треки и счётчик ID.
        raise NotImplementedError

    def step(self, detections: Iterable[Detection], dt: float = 1.0) -> list[TrackView]:
        """Обработать один кадр и вернуть все активные треки, включая прогнозы.

        detections — обнаружения текущего кадра; пустой список означает пропуск.
        dt — число кадров с прошлого шага. Каждое измерение сопоставляется
        не более чем одному треку. ID удалённых треков не используйте повторно.
        """
        # TODO(student): прогноз, сопоставление, обновление, создание и удаление.
        raise NotImplementedError
