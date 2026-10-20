"""Сопоставление признаков FlowNetC на чистом PyTorch."""

import torch
import torch.nn.functional as F


def cost_volume(first, second, radius=10, step=2):
    """Вернуть сходство для каждого смещения: (B, (2r+1)², H, W)."""
    if first.shape != second.shape:
        raise ValueError("Карты признаков должны иметь одинаковую форму")

    height, width = first.shape[-2:]
    padding = radius * step
    # F.pad получает отступы в порядке: слева, справа, сверху, снизу.
    padded = F.pad(second, (padding, padding, padding, padding))
    channels = []
    # Порядок dy, затем dx должен совпадать с порядком при обучении модели.
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            top = padding + dy * step
            left = padding + dx * step
            shifted = padded[:, :, top:top + height, left:left + width]
            # Среднее произведение признаков по каналам: (B, C, H, W) → (B, H, W).
            similarity = (first * shifted).mean(dim=1)
            channels.append(similarity)
    # Новый канал соответствует одному смещению второго feature map.
    return torch.stack(channels, dim=1)


def correlate(first, second):
    """Параметры слоя соответствуют выбранным предобученным весам FlowNetC."""
    # 21×21 смещений с шагом 2: от −20 до +20 feature-пикселей по каждой оси.
    volume = cost_volume(first, second, radius=10, step=2)
    return F.leaky_relu(volume, negative_slope=0.1)
