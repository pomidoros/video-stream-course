"""Проверить обёртку OpenCV на известных переносах и неинформативных точках."""

import numpy as np

from lk import corners, lucas_kanade, opencv_lk, synthetic


def check_translations():
    for dx, dy in [(0, 0), (0.5, -0.5), (4, 2), (-2, 1), (12, 6)]:
        previous, current = synthetic(dx, dy)
        points = corners(previous)
        next_points, valid = opencv_lk(previous, current, points)
        error = np.linalg.norm(next_points[valid] - points[valid] - [dx, dy], axis=1)
        assert valid.mean() > 0.9
        assert np.median(error) < 0.05
        # Обёртка дополнительно исключает выходящие за кадр координаты.
        assert ((next_points[valid] >= 0) & (next_points[valid] < [240, 180])).all()
        print(f"Сдвиг ({dx}, {dy}): медианная EPE {np.median(error):.4f} px")


def check_invalid_points():
    for kind in ["flat", "edge"]:
        previous, current = synthetic(kind=kind)
        # Один масштаб: окно не получает случайной информации с грубых уровней.
        _, valid = opencv_lk(previous, current, [[120, 90]], levels=0)
        assert not valid.any()

    previous, current = synthetic()
    next_points, valid = opencv_lk(previous, current, [])
    assert next_points.shape == (0, 2) and next_points.dtype == np.float32
    assert valid.shape == (0,) and valid.dtype == bool


def check_simple_lk():
    # Собственная версия делает один шаг: проверяем только малые переносы.
    for dx, dy in [(0, 0), (.5, .25), (-.5, .25)]:
        previous, current = synthetic(dx, dy)
        points = corners(previous)
        found, valid = lucas_kanade(previous, current, points)
        assert valid.mean() > .9
        error = np.linalg.norm(found[valid] - points[valid] - [dx, dy], axis=1)
        assert np.median(error) < .05
    for kind in ['flat', 'edge']:
        previous, current = synthetic(kind=kind)
        _, valid = lucas_kanade(previous, current, [[120, 90]])
        assert not valid.any()
    previous, current = synthetic()
    found, valid = lucas_kanade(previous, current, [])
    assert found.shape == (0, 2) and valid.shape == (0,)
    print('Своя LK: малые переносы, вырожденные окна и пустой набор — OK')


if __name__ == "__main__":
    check_simple_lk()
    check_translations()
    check_invalid_points()
    print("Все проверки пройдены")
