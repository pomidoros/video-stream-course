"""Численные проверки готового потока OpenCV и компенсации движения."""

import cv2
import numpy as np

from dense import farneback, flow_rgb, synthetic, warp_next


def check_translations():
    for dx, dy in [(0, 0), (.25, -.25), (.5, .25), (-.5, .25), (1, .5)]:
        previous, current = synthetic(dx, dy)
        flow = farneback(previous, current)
        assert flow.shape == (*previous.shape, 2) and flow.dtype == np.float32
        # Измеряем ошибку внутри кадра, где нет дополненных границ.
        center = flow[32:-32, 32:-32]
        error = np.linalg.norm(center - [dx, dy], axis=-1)
        assert np.median(error) < .08
        print(f'Сдвиг ({dx}, {dy}): медианная EPE {np.median(error):.4f} px')


def check_warp_and_color():
    # Заданный точный поток проверяет знак и порядок координат remap.
    previous, current = synthetic(dx=3, dy=-2)
    flow = np.empty((*previous.shape, 2), dtype=np.float32)
    flow[:] = [3, -2]
    warped, inside = warp_next(current, flow)
    assert np.array_equal(previous[32:-32, 32:-32], warped[32:-32, 32:-32])
    assert not inside[-1, -1]

    # Нулевое движение чёрное; отвергнутое измерение серое.
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    assert not flow_rgb(flow).any()
    invalid = np.zeros((2, 2), dtype=bool)
    assert np.array_equal(flow_rgb(flow, valid=invalid)[0, 0], [65, 65, 65])
    flow[:] = [2, 0]
    assert np.array_equal(flow_rgb(flow, max_motion=8)[0, 0], [64, 0, 0])

    # Пирамида восстанавливает большой перенос на фиксированной текстуре.
    previous, current = synthetic(dx=12, dy=6)
    flow = farneback(previous, current)
    error = np.linalg.norm(flow[32:-32, 32:-32] - [12, 6], axis=-1)
    assert np.median(error) < .1
    print('Направление remap, цветовая шкала и большой перенос: OK')


if __name__ == '__main__':
    check_translations()
    check_warp_and_color()
    print('Все проверки пройдены')
