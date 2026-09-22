import unittest
from pathlib import Path
import cv2
from project.common import ROOT, checked_detections
from shots.outcome_io import load_recordings, load_features, load_attempts
from players.people_io import load_manifest, load_clip, load_labels


class EnvironmentTests(unittest.TestCase):
    def test_ball_data_and_split(self):
        rows = load_recordings(ROOT)
        self.assertEqual({r["split"] for r in rows}, {"train", "validation"})
        self.assertEqual(len({r["source_group"] for r in rows}), len(rows))
        known = {s: 0 for s in ["train", "validation"]}
        for r in rows:
            load_features(ROOT, r)
            checked_detections(ROOT, r)
            known[r["split"]] += sum(
                a["outcome"] != "uncertain" for a in load_attempts(ROOT, r)
            )
        self.assertEqual(known, dict(train=55, validation=22))

    def test_people_data_video_and_dependencies(self):
        from deep_sort_realtime.deep_sort.tracker import Tracker
        import imageio_ffmpeg

        self.assertTrue(Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file())
        C = ROOT / "players"
        for c in load_manifest(C)["clips"]:
            data = load_clip(C, c)
            self.assertEqual(len(data["frames"]), c["frame_count"])
            self.assertEqual(len(load_labels(C, c)), c["dense_gt_boxes"])
            cap = cv2.VideoCapture(str(C / c["source_video"]))
            ok, im = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(im.shape[:2], (720, 1280))


if __name__ == "__main__":
    unittest.main()
