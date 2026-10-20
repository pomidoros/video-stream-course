import unittest
import numpy as np
import torch
from project.common import solutions, ROOT
from project.contracts import ball_rows, people_rows


class StudentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcome, cls.people = solutions()
        torch.set_num_threads(2)

    def test_cosine_values_and_normalization(self):
        q = np.array([[2.0, 0.0], [0.0, 3.0]])
        g = np.array([[4.0, 0.0], [-2.0, 0.0]])
        np.testing.assert_allclose(
            self.people.cosine_distances(q, g), [[0, 2], [1, 1]], atol=1e-6
        )
        np.testing.assert_array_equal(q, [[2, 0], [0, 3]])

    def test_cosine_invalid_and_empty(self):
        for q, g in [
            (np.zeros((1, 2)), np.ones((1, 2))),
            (np.array([[np.nan, 1]]), np.ones((1, 2))),
            (np.ones(2), np.ones((1, 2))),
            (np.ones((2, 3)), np.ones((2, 2))),
        ]:
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    self.people.cosine_distances(q, g)
        self.assertEqual(
            self.people.cosine_distances(np.empty((0, 2)), np.ones((3, 2))).shape,
            (0, 3),
        )

    def test_retrieval_unknown_tie_and_empty(self):
        g = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([[3.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(
            self.people.identify_people(q, g, np.array([7, 19]), 0.3), [7, -1, 19]
        )
        np.testing.assert_array_equal(
            self.people.identify_people(q, np.empty((0, 2)), np.array([], int), 0.3),
            [-1] * 3,
        )
        np.testing.assert_array_equal(
            self.people.identify_people(
                g[:1], np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([7, 19]), 0
            ),
            [7],
        )

    def test_model_shapes_and_gradients(self):
        for kind in ["last", "mean", "gru"]:
            model = self.outcome.TemporalOutcomeModel(kind)
            x = torch.randn(3, 16, 70)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (3,))
            self.assertTrue(torch.isfinite(logits).all())
            logits.sum().backward()
            self.assertTrue(
                any(
                    p.grad is not None
                    and torch.isfinite(p.grad).all()
                    and p.grad.abs().sum() > 0
                    for p in model.parameters()
                )
            )

    def test_temporal_model_uses_history(self):
        torch.manual_seed(17)
        model = self.outcome.TemporalOutcomeModel("gru").eval()
        x = torch.randn(4, 16, 70)
        altered = x.clone()
        altered[:, :-1] *= -2
        with torch.no_grad():
            self.assertFalse(
                torch.allclose(model(x), model(altered), atol=1e-7, rtol=0)
            )

    def test_training_windows_labels_and_endpoints(self):
        t = np.arange(40) / 10
        x = np.repeat(np.arange(40)[:, None], 70, axis=1).astype(np.float32)
        attempts = [
            dict(reference_time_s=0.0, outcome="made"),
            dict(reference_time_s=2.0, outcome="uncertain"),
        ]
        xx, yy = self.outcome.build_training_windows(x, t, attempts, 4)
        ends = [i for i in range(3, len(t), 2) if 0.8 <= t[i] <= 1.8]
        np.testing.assert_array_equal(xx, np.stack([x[i - 3 : i + 1] for i in ends]))
        np.testing.assert_array_equal(yy, np.ones(len(ends)))
        xx, yy = self.outcome.build_training_windows(x, t, [attempts[1]], 4)
        self.assertEqual(xx.shape, (0, 4, 70))
        self.assertEqual(yy.shape, (0,))

    def test_training_changes_weights(self):
        torch.manual_seed(17)
        model = self.outcome.TemporalOutcomeModel("gru")
        x = torch.randn(32, 16, 70)
        y = (x[:, -1, 0] > 0).float()
        before = [p.detach().clone() for p in model.parameters()]
        history = self.outcome.train_model(
            model,
            (x.numpy(), y.numpy()),
            (x.numpy(), y.numpy()),
            dict(epochs=3, lr=0.01, seed=17),
        )
        self.assertTrue(history["curve"])
        self.assertGreater(history["best_epoch"], 0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        )

    def test_ball_missing_frames_and_prefix(self):
        frames = []
        for i in range(25):
            ds = (
                []
                if 10 <= i < 12
                else [
                    dict(
                        class_name="ball",
                        confidence=0.9,
                        box=np.array([50 + i, 20 + 2 * i, 54 + i, 24 + 2 * i]),
                    )
                ]
            )
            frames.append((i, i * 0.05, ds))
        rim = np.array([40, 80, 100, 90])
        rows = self.outcome.track_ball(frames, rim)
        ball_rows(rows, frames)
        self.assertEqual(rows[:16], self.outcome.track_ball(frames[:16], rim))
        self.assertFalse(rows[10]["observed"])
        self.assertTrue(rows[15]["observed"])

    def test_candidates_observed_time_and_causality(self):
        rim = np.array([40, 80, 100, 90])
        ys = [35, 40, 48, 55, 65, 85, 110]
        tracks = [
            dict(
                frame=i,
                time_s=i * 0.1,
                observed=True,
                x=70.0,
                y=float(y),
                vx=0.0,
                vy=70.0,
            )
            for i, y in enumerate(ys)
        ]
        candidates = self.outcome.find_candidates(tracks, rim)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0], 0.3)
        self.assertEqual(self.outcome.find_candidates(tracks[:4], rim), candidates)

    def test_people_complete_timeline_and_prefix(self):
        data = dict(np.load(ROOT / "players/data/basket_S1T1_pre.npz"))
        mask = data["det_frame"] <= 40
        short = {k: v[:21] if k == "frames" else v[mask] for k, v in data.items()}
        rows = self.people.track_with_deepsort(data, 19.98)
        people_rows(rows, data, 19.98)
        self.assertEqual(rows[:21], self.people.track_with_deepsort(short, 19.98))

    def test_people_empty_observations(self):
        data = dict(
            frames=np.arange(0, 20, 2),
            det_frame=np.empty(0, int),
            bbox=np.empty((0, 4)),
            confidence=np.empty(0),
            embeddings=np.empty((0, 512)),
        )
        rows = self.people.track_with_deepsort(data, 20.0)
        people_rows(rows, data, 20.0)
        self.assertTrue(all(not r["tracks"] for r in rows))


if __name__ == "__main__":
    unittest.main()
