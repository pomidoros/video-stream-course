import unittest
import numpy as np
from shots.stream_utils import predict_stream, evaluate_stream
from project.thresholds import checks
from project.contracts import events


class InfrastructureTests(unittest.TestCase):
    def test_queue_has_no_future_input(self):
        times = np.arange(20) / 10
        x = np.arange(20)[:, None]
        seen = []
        result = predict_stream(
            x,
            times,
            [0.0],
            lambda w: seen.append(w.copy()) or 0.5,
            length=4,
            delay_s=1.8,
        )
        np.testing.assert_array_equal(seen[0].ravel(), [15, 16, 17, 18])
        self.assertEqual(result[0]["decision_index"], 18)

    def test_unknown_at_end_is_not_removed(self):
        r = predict_stream(np.ones((2, 70)), np.array([0.0, 0.1]), [0.1], lambda w: 0.5)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["outcome"], "unknown")

    def test_duplicate_events_count_as_false_positive(self):
        a = [dict(attempt_id="a", reference_time_s=0.0, outcome="made")]
        p = [
            dict(
                candidate_time_s=0.0, time_s=1.8, made_probability=0.9, outcome="made"
            ),
            dict(
                candidate_time_s=0.1, time_s=1.9, made_probability=0.9, outcome="made"
            ),
        ]
        self.assertEqual(evaluate_stream(a, p)["event"]["fp"], 1)

    def test_missing_nan_and_empty_metrics_fail(self):
        for d in [{}, {"x": float("nan")}, {"x": None}, {"x": True}]:
            self.assertFalse(checks(d, {"x": (">=", 0.5)})[0]["passed"])

    def test_backdated_decision_rejected(self):
        p = [
            dict(
                candidate_time_s=0.0,
                time_s=0.1,
                decision_index=1,
                outcome="made",
                made_probability=0.9,
            )
        ]
        with self.assertRaises(ValueError):
            events(p, np.arange(30) / 10)


if __name__ == "__main__":
    unittest.main()
