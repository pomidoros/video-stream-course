"""Проверить соответствие весам, correlation и единицы выходного потока."""

import numpy as np
import torch
import torch.nn.functional as F

from correlation import cost_volume
from flownets_simple import SimpleFlowNetS
from inference import checkpoint, load_model, predict, prepare_pair, restore_flow
from vendor.models.FlowNetS import FlowNetS


def check_architecture():
    weights = checkpoint('flownets')['state_dict']
    model = SimpleFlowNetS().eval()
    reference = FlowNetS(batchNorm=False).eval()
    model.load_state_dict(weights, strict=True)
    reference.load_state_dict(weights, strict=True)
    # Одинаковые веса должны давать ровно тот же результат, в том числе на нечётном входе.
    with torch.inference_mode():
        for height, width in [(64, 64), (95, 157), (192, 320)]:
            pair = torch.randn(1, 6, height, width)
            assert torch.equal(model(pair), reference(pair))
        model.train()
        scales = model(torch.zeros(1, 6, 128, 192))
    assert [tuple(flow.shape[-2:]) for flow in scales] == [(32, 48), (16, 24), (8, 12), (4, 6), (2, 3)]
    print('FlowNetS совпадает с исходной реализацией на трёх размерах')


def check_correlation():
    # Независимое сравнение: unfold сразу собирает все смещённые окна.
    first = torch.randn(2, 3, 5, 7, requires_grad=True)
    second = torch.randn(2, 3, 5, 7, requires_grad=True)
    volume = cost_volume(first, second, radius=2, step=2)
    windows = F.unfold(second, kernel_size=5, dilation=2, padding=4).reshape(2, 3, 25, 5, 7)
    reference = (first.unsqueeze(2) * windows).mean(dim=1)
    assert torch.allclose(volume, reference, atol=1e-6)
    volume.sum().backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()

    # Положительный dx ищет соответствие справа во втором feature map.
    first = torch.zeros(1, 1, 7, 7)
    second = torch.zeros_like(first)
    first[0, 0, 3, 3] = 1
    second[0, 0, 3, 5] = 1
    assert cost_volume(first, second, radius=1, step=2)[0, :, 3, 3].argmax() == 5
    print('Correlation: значения, порядок каналов и градиенты корректны')


def check_inference():
    image = np.full((180, 320, 3), [255, 128, 0], dtype=np.uint8)
    pair = prepare_pair(image, image)
    assert pair.shape == (1, 6, 192, 320)
    assert np.allclose(pair[0, :3, 0, 0], [1 - .411, 128 / 255 - .432, -.450])
    # Постоянный выход проверяет div_flow без ошибочного дополнительного ×4.
    raw = torch.zeros(1, 2, 48, 80)
    raw[:, 0] = .1
    raw[:, 1] = -.2
    flow = restore_flow(raw, (192, 320), (180, 320), div_flow=20)
    assert flow.shape == (180, 320, 2) and np.allclose(flow, [2, -4], atol=1e-6)
    for name in ['flownets', 'flownetc']:
        model, div_flow = load_model(name)
        flow = predict(model, image, image, div_flow)
        assert flow.shape == (180, 320, 2) and np.isfinite(flow).all()
    print('RGB, padding, единицы потока и инференс обеих моделей корректны')


if __name__ == '__main__':
    check_architecture()
    check_correlation()
    check_inference()
    print('Все проверки пройдены')
