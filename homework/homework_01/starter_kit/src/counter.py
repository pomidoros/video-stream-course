from __future__ import annotations

from collections.abc import Iterable


class LineCounter:
    """Считать входы и выходы через конечный отрезок start → end.

    start, end — координаты концов в пикселях. entry_direction задаёт вход:
    "negative_to_positive" или "positive_to_negative" по знаку векторного
    произведения (end - start) и (center - start). Для горизонтальной линии
    слева направо первое значение соответствует движению сверху вниз.
    initial_occupancy — начальная заполненность. Создайте открытые счётчики
    entered и exited; начальное значение обоих равно нулю.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        entry_direction: str = "negative_to_positive",
        initial_occupancy: int = 0,
    ) -> None:
        # TODO(student): проверьте параметры, создайте историю и счётчики.
        raise NotImplementedError

    @property
    def current(self) -> int:
        """Вернуть баланс initial_occupancy + entered - exited без обрезания."""
        # TODO(student): вычислите баланс пересечений.
        raise NotImplementedError

    def observe(self, track_id: int, center: Iterable[float], measured: bool = True) -> str | None:
        """Принять центр трека и вернуть "entered", "exited" либо None.

        measured=False означает прогноз: он не подтверждает пересечение.
        Считайте каждое направление не более одного раза для одного ID.
        Точка точно на линии не должна стирать последнее положение вне неё.
        """
        # TODO(student): пересечение отрезка, направление и защита от повторов.
        raise NotImplementedError
