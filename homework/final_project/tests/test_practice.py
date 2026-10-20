"""Проверки учебных блоков видео, простого трекера и статистики."""

import unittest
from pathlib import Path
import numpy as np
from project.common import ROOT, solutions, video_solution
from project.contracts import people_rows


class PracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.people = solutions()
        cls.video = video_solution()

    def test_video_keeps_original_time_and_limits_frames(self):
        source = ROOT / "players/videos/basket_S1T1_pre.mp4"
        iterator = self.video.read_frames(source, stride=3, width=320, max_frames=4)
        self.assertIs(iter(iterator), iterator)
        frames = list(iterator)
        self.assertEqual([r["frame"] for r in frames], [0, 3, 6, 9])
        np.testing.assert_allclose(
            [r["time_s"] for r in frames], np.array([0, 3, 6, 9]) / 19.98
        )
        self.assertTrue(
            all(
                r["image"].shape == (180, 320, 3) and r["image"].dtype == np.uint8
                for r in frames
            )
        )

    def test_video_rejects_invalid_inputs(self):
        for path, kwargs in [
            (ROOT / "absent.mp4", {}),
            (ROOT / "players/videos/basket_S1T1_pre.mp4", {"stride": 0}),
        ]:
            with self.subTest(path=path, kwargs=kwargs), self.assertRaises(ValueError):
                list(self.video.read_frames(path, **kwargs))

    def test_motion_mask_removes_small_components(self):
        first = np.zeros((24, 24, 3), np.uint8)
        second = first.copy()
        second[4:8, 4:8] = 100
        second[18, 18] = 255
        original = second.copy()
        mask = self.video.motion_mask(first, second, threshold=25, min_area=8)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, first.shape[:2])
        self.assertEqual(int((mask == 255).sum()), 16)
        self.assertEqual(mask[18, 18], 0)
        np.testing.assert_array_equal(second, original)
        self.assertFalse(self.video.motion_mask(first, first).any())

    def test_flow_detects_translation(self):
        rng = np.random.default_rng(17)
        first = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        second = np.roll(first, 3, axis=1)
        flow = self.video.dense_flow(first, second)
        self.assertEqual(flow.shape, (80, 100, 2))
        self.assertEqual(flow.dtype, np.float32)
        self.assertTrue(np.isfinite(flow).all())
        self.assertGreater(float(np.median(flow[20:-20, 20:-20, 0])), 0.5)
        self.assertLess(float(np.median(flow[20:-20, 20:-20, 0])), 6)
        still = self.video.dense_flow(first, first)
        self.assertLess(float(np.median(np.linalg.norm(still, axis=-1))), 0.1)

    def test_motion_summary_uses_mask_and_time(self):
        mask = np.array([[255, 0], [0, 255]], np.uint8)
        flow = np.zeros((2, 2, 2), np.float32)
        flow[mask != 0] = [3, 4]
        flow[mask == 0] = [100, 100]
        result = self.video.summarize_motion(mask, flow, 0.5)
        self.assertAlmostEqual(result["moving_fraction"], 0.5)
        self.assertAlmostEqual(result["median_speed_px_s"], 10)
        self.assertEqual(
            self.video.summarize_motion(mask * 0, flow, 0.5)["median_speed_px_s"], 0
        )
        with self.assertRaises(ValueError):
            self.video.summarize_motion(mask, flow, 0)
        with self.assertRaises(ValueError):
            self.video.summarize_motion(mask, flow * np.nan, 0.5)

    def tracker_input(self):
        return dict(
            frames=np.arange(7),
            det_frame=np.array([0, 1, 5, 6]),
            bbox=np.array(
                [[0, 0, 20, 20], [1, 0, 21, 20], [1, 0, 21, 20], [2, 0, 22, 20]], float
            ),
            confidence=np.ones(4),
            embeddings=np.ones((4, 512)),
        )

    def test_iou_tracker_expires_tracks_and_preserves_prefix(self):
        data = self.tracker_input()
        rows = self.people.track_with_iou(data, 10, max_age_s=0.2)
        people_rows(rows, data, 10)
        first = rows[0]["tracks"][0]["track_id"]
        self.assertEqual(rows[1]["tracks"][0]["track_id"], first)
        self.assertFalse(rows[2]["tracks"][0]["observed"])
        self.assertEqual(rows[4]["tracks"], [])
        self.assertNotEqual(rows[5]["tracks"][0]["track_id"], first)
        short = {
            key: value[:4] if key == "frames" else value[:2]
            for key, value in data.items()
        }
        self.assertEqual(rows[:4], self.people.track_with_iou(short, 10, max_age_s=0.2))
        altered = {key: value.copy() for key, value in data.items()}
        altered["embeddings"] *= -3
        self.assertEqual(rows, self.people.track_with_iou(altered, 10, max_age_s=0.2))

    def test_iou_tracker_assigns_detections_once_and_handles_empty(self):
        data = dict(
            frames=np.arange(2),
            det_frame=np.array([0, 0, 1]),
            bbox=np.array([[0, 0, 20, 20], [2, 0, 22, 20], [1, 0, 21, 20]], float),
            confidence=np.ones(3),
            embeddings=np.ones((3, 512)),
        )
        rows = self.people.track_with_iou(data, 10)
        people_rows(rows, data, 10)
        self.assertEqual(sum(t["observed"] for t in rows[1]["tracks"]), 1)
        self.assertEqual(len(rows[0]["tracks"]), 2)
        empty = {key: value[:0] for key, value in data.items()}
        self.assertEqual(self.people.track_with_iou(empty, 10), [])

    def test_statistics_does_not_bridge_missing_observation(self):
        rows = []
        for index, x in enumerate([0, 3, None, 90, 94]):
            tracks = (
                []
                if x is None
                else [
                    dict(
                        track_id="7",
                        bbox=[x, 0, x + 2, 10],
                        confirmed=True,
                        observed=True,
                    )
                ]
            )
            # Предсказанная позиция и неподтверждённый трек не входят в статистику.
            tracks.append(
                dict(track_id="99", bbox=[0, 0, 10, 10], confirmed=True, observed=False)
            )
            tracks.append(
                dict(
                    track_id="100", bbox=[0, 0, 10, 10], confirmed=False, observed=True
                )
            )
            rows.append(dict(frame=index, time_s=index * 0.1, tracks=tracks))
        stats = self.people.player_statistics(rows, 10, (100, 100), (5, 5))
        self.assertEqual(set(stats), {"7"})
        item = stats["7"]
        self.assertEqual(item["observed_frames"], 4)
        self.assertAlmostEqual(item["observed_time_s"], 0.4)
        self.assertAlmostEqual(item["distance_px"], 7)
        self.assertAlmostEqual(item["mean_speed_px_s"], 35)
        self.assertAlmostEqual(item["max_speed_px_s"], 40)
        heatmap = np.array(item["heatmap"])
        self.assertEqual(heatmap.shape, (5, 5))
        self.assertEqual(heatmap.sum(), 4)
        self.assertEqual(heatmap[0, 0], 2)
        self.assertEqual(heatmap[0, 4], 2)
        self.assertEqual(self.people.player_statistics([], 10, (100, 100)), {})

    def test_heatmap_clips_image_boundary(self):
        rows = [
            dict(
                frame=0,
                time_s=0,
                tracks=[
                    dict(
                        track_id=3,
                        bbox=[90, 90, 110, 110],
                        confirmed=True,
                        observed=True,
                    )
                ],
            )
        ]
        stats = self.people.player_statistics(rows, 10, (100, 100), (5, 5))["3"]
        self.assertEqual(np.array(stats["heatmap"])[4, 4], 1)
        self.assertEqual(stats["mean_speed_px_s"], 0)


if __name__ == "__main__":
    unittest.main()
