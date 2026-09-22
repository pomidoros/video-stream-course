"""Часть 1: реализуйте функции для мяча, бросков и обучения модели.

Начните с track_ball и find_candidates, затем перейдите к обучению.
Загрузка данных, сохранение весов, оценка и видео уже готовы.
"""

from torch import nn


class FrameBaseline(nn.Module):
    """Готовый пример: модель исхода по последнему кадру окна."""

    def __init__(self):
        """Создать небольшую сеть: 70 признаков → 32 числа → один logit."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        """Преобразовать тензор [пачка, время, 70] в logits [пачка]."""
        last_frame = features[:, -1]
        return self.net(last_frame).squeeze(-1)


class TemporalOutcomeModel(nn.Module):
    """Модель исхода броска: последний кадр, среднее или GRU."""

    def __init__(self, kind="gru"):
        """Создать слои выбранной модели.

        Аргументы:
            kind (str): «last», «mean» или «gru». Для «last» можно
                использовать готовый FrameBaseline.
        """
        super().__init__()
        raise NotImplementedError("Создайте слои для трёх вариантов модели")

    def forward(self, features):
        """Вычислить исход для каждого окна.

        Аргументы:
            features (torch.Tensor): окна [пачка, время, 70].

        Возвращает:
            torch.Tensor: logits [пачка], без sigmoid.
        """
        raise NotImplementedError("Реализуйте прямой проход модели")


def build_training_windows(features, time_s, attempts, length):
    """Собрать окна с известными исходами для обучения.

    Аргументы:
        features (np.ndarray): признаки [кадры, признаки], уже нормализованы.
        time_s (np.ndarray): время каждого шага в секундах.
        attempts (list[dict]): попытки с reference_time_s и outcome.
        length (int): число шагов в окне.

    Возвращает:
        tuple: X [окна, length, признаки] и y [окна], оба float32.
            made = 1, missed = 0; uncertain пропускается.

    Концы окон: range(length - 1, len(features), 2). Конец должен попасть
    в интервал [reference_time_s + 0.8, reference_time_s + 1.8].
    Если примеров нет, верните пустые массивы с теми же размерностями.
    """
    raise NotImplementedError("Соберите окна и соответствующие им метки")


def train_model(model, train_data, validation_data, config):
    """Обучить переданную модель на CPU и восстановить её лучшие веса.

    Аргументы:
        model (nn.Module): модель, которую нужно обучить.
        train_data (tuple): обучающие массивы (X, y).
        validation_data (tuple): проверочные массивы (X, y).
        config (dict): epochs — число эпох, lr — шаг обучения, seed.

    Возвращает:
        dict: curve — список {epoch, train_loss, validation_brier};
            best_epoch — номер лучшей эпохи, начиная с 1.

    Используйте BCEWithLogitsLoss. Лучшие веса выбирайте по минимальному
    validation Brier. Сохранять файлы здесь не нужно: это делает runner.
    """
    raise NotImplementedError("Реализуйте цикл обучения и выбор лучших весов")


def track_ball(detections, rim):
    """Сопроводить мяч фильтром Калмана, используя только доступное прошлое.

    Аргументы:
        detections (list): шаги (frame, time_s, объекты). Каждый объект —
            словарь с class_name, confidence и box = [x1, y1, x2, y2].
        rim (np.ndarray): прямоугольник кольца [x1, y1, x2, y2].

    Возвращает:
        list[dict]: по одному состоянию на шаг: frame, time_s, observed,
            x, y, vx, vy. Координаты — в пикселях, скорости — в пикселях/с.
            До появления состояния x, y, vx, vy равны None.

    Обработайте пропуски и выбросы. При прогнозе без принятого наблюдения
    observed = False. Входные данные не изменяйте.
    """
    raise NotImplementedError("Подключите Калман из материалов курса")


def find_candidates(tracks, rim):
    """Найти попытки броска по движению мяча около кольца.

    Аргументы:
        tracks (list[dict]): состояния из track_ball.
        rim (np.ndarray): прямоугольник кольца [x1, y1, x2, y2].

    Возвращает:
        list[float]: возрастающие времена кандидатов в секундах.
            Используйте время наблюдаемого кадра, без интерполяции назад.

    Подавляйте повторы. Правило пересечения и численные параметры
    для базового сравнения описаны в docs/API.md.
    """
    raise NotImplementedError("Найдите кандидатов по траектории")
