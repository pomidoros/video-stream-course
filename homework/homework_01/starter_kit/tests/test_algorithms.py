import numpy as np

from src.counter import LineCounter
from src.data import Detection
from src.kalman import KalmanCV
from src.tracker import MultiObjectTracker


def detection(cx: float, cy: float) -> Detection:
    return Detection(cx - 5, cy - 10, 10, 20)


def test_kalman_predict_and_update() -> None:
    kf = KalmanCV((0, 0), process_var=1, measurement_var=1)
    kf.predict()
    center = kf.update((10, 0))
    assert 0 < center[0] < 10
    assert np.allclose(kf.P, kf.P.T)


def test_track_survives_two_missing_frames() -> None:
    tracker = MultiObjectTracker(min_hits=1, max_missed=2, max_distance=50)
    track_id = tracker.step([detection(0, 0)])[0].track_id
    tracker.step([detection(5, 0)])
    tracker.step([])
    tracker.step([])
    view = tracker.step([detection(20, 0)])[0]
    assert view.track_id == track_id
    assert view.recovered


def test_prediction_crossing_is_not_counted() -> None:
    counter = LineCounter((0, 0), (10, 0))
    counter.observe(1, (5, -2), measured=True)
    assert counter.observe(1, (5, 2), measured=False) is None
    assert counter.entered == 0


def test_exact_line_point_preserves_crossing() -> None:
    counter = LineCounter((0, 0), (10, 0), initial_occupancy=2)
    counter.observe(1, (5, -2))
    counter.observe(1, (5, 0))
    assert counter.observe(1, (5, 2)) == "entered"
    assert counter.current == 3



def test_kalman_prediction_uses_velocity_and_dt() -> None:
    kf = KalmanCV((3, 4), process_var=1, measurement_var=1)
    kf.x[2:] = (2, -1)
    np.testing.assert_allclose(kf.predict(dt=2), (7, 2))


def test_new_track_needs_confirmation() -> None:
    tracker = MultiObjectTracker(min_hits=2, max_distance=50)
    first = tracker.step([detection(0, 0)])[0]
    assert not first.confirmed
    second = tracker.step([detection(1, 0)])[0]
    assert second.track_id == first.track_id and second.confirmed


def test_expired_track_is_deleted_and_id_is_not_reused() -> None:
    tracker = MultiObjectTracker(min_hits=1, max_missed=1)
    old_id = tracker.step([detection(0, 0)])[0].track_id
    assert tracker.step([])[0].source == "prediction"
    assert tracker.step([]) == []
    assert tracker.step([detection(0, 0)])[0].track_id != old_id


def test_one_measurement_cannot_update_two_tracks() -> None:
    tracker = MultiObjectTracker(min_hits=1, max_distance=50)
    tracker.step([detection(0, 0), detection(20, 0)])
    views = tracker.step([detection(10, 0)])
    assert len(views) == 2
    assert sum(view.source == "measurement" for view in views) == 1
    assert len({view.track_id for view in views}) == 2


def test_crossing_outside_segment_does_not_count() -> None:
    counter = LineCounter((0, 0), (10, 0))
    counter.observe(1, (20, -2))
    assert counter.observe(1, (20, 2)) is None
    assert counter.entered == counter.exited == 0


def test_same_direction_counts_once_per_id() -> None:
    counter = LineCounter((0, 0), (10, 0))
    for y in (-2, 2, -2, 2):
        counter.observe(1, (5, y))
    assert counter.entered == counter.exited == 1
