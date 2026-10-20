"""Загрузка весов и вычисление плотного потока на CPU."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flownets_simple import SimpleFlowNetS
from vendor.models.FlowNetC import FlowNetC

DIRECTORY = Path(__file__).resolve().parent
# Для небольших кадров слишком много CPU-потоков создаёт лишние накладные расходы.
torch.set_num_threads(min(torch.get_num_threads(), 4))


def checkpoint(name):
    """Прочитать checkpoint; пути и источники весов перечислены в assets.json."""
    weights = json.loads((DIRECTORY / "assets.json").read_text())["weights"]
    path = DIRECTORY / "weights" / weights[name]["filename"]
    if not path.exists():
        raise FileNotFoundError("Сначала выполните python fetch_assets.py")
    # На CPU читаем только данные весов, без выполнения произвольного pickle-кода.
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(name):
    """Создать одну модель, загрузить веса и вернуть её вместе с div_flow."""
    if name == "flownets":
        model = SimpleFlowNetS()
    elif name == "flownetc":
        model = FlowNetC(batchNorm=False)
    else:
        raise ValueError("Выберите flownets или flownetc")

    weights = checkpoint(name)
    # strict=True требует совпадения всех имён и размеров параметров.
    model.load_state_dict(weights["state_dict"], strict=True)
    model.eval()
    return model, float(weights.get("div_flow", 20))


def prepare_pair(first, second):
    """Преобразовать два RGB uint8-кадра в тензор (1, 6, H, W)."""
    if first.shape != second.shape:
        raise ValueError("Кадры должны иметь одинаковые размеры")
    # Соединяем RGB-каналы: (H, W, 3) + (H, W, 3) → (H, W, 6).
    pair = np.concatenate([first, second], axis=2)
    # permute: каналы вперёд; unsqueeze: добавляем размерность batch.
    pair = torch.from_numpy(pair).permute(2, 0, 1).float().unsqueeze(0) / 255
    # Эти средние RGB соответствуют выбранным PyTorch checkpoints.
    mean = torch.tensor([0.411, 0.432, 0.450, 0.411, 0.432, 0.450]).view(1, 6, 1, 1)
    pair = pair - mean

    height, width = first.shape[:2]
    pad_height = (-height) % 64
    pad_width = (-width) % 64
    # Добавляем только справа и снизу; replicate повторяет крайние пиксели.
    return F.pad(pair, (0, pad_width, 0, pad_height), mode="replicate")


def restore_flow(raw_flow, padded_shape, original_shape, div_flow=20):
    """Вернуть поток на исходную сетку в единицах px за пару кадров."""
    flow = F.interpolate(
        raw_flow,
        size=padded_shape,     # Высота и ширина входа после padding.
        mode="bilinear",      # Плавная интерполяция пространственной сетки.
        align_corners=False,  # Соглашение о координатах как в исходном инференсе.
    )
    # Модель предсказывает векторы в пикселях полного входа, делённых на div_flow.
    # Дополнительно умножать на 4 из-за сетки H/4 × W/4 не нужно.
    flow = flow * div_flow
    height, width = original_shape
    flow = flow[0, :, :height, :width]  # Убираем batch и добавленные края.
    return flow.permute(1, 2, 0).cpu().numpy()


@torch.inference_mode()  # Не строим граф градиентов при предсказании.
def predict(model, first, second, div_flow=20):
    """Модель загружается снаружи один раз, затем используется для всех пар."""
    pair = prepare_pair(first, second)
    raw_flow = model(pair)
    return restore_flow(raw_flow, pair.shape[-2:], first.shape[:2], div_flow)
