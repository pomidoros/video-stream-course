"""Проверить формат предсказаний перед оценкой и отрисовкой."""

import numpy as np


def ball_rows(rows, frames):
    if len(rows) != len(frames):
        raise ValueError("Ball tracker must return every time step")
    for r, (f, t, _) in zip(rows, frames):
        if r["frame"] != f or not np.isclose(r["time_s"], t):
            raise ValueError("Ball frame/time mismatch")
        if type(r["observed"]) is not bool:
            raise ValueError("observed must be bool")
        values = [r[k] for k in ("x", "y", "vx", "vy")]
        if any(v is None for v in values):
            if not all(v is None for v in values) or r["observed"]:
                raise ValueError("Missing state must contain four nulls")
        elif not np.isfinite(values).all():
            raise ValueError("Non-finite ball state")


def people_rows(rows, data, fps):
    if len(rows) != len(data["frames"]):
        raise ValueError("Return every frame, including empty observations")
    for r, f in zip(rows, data["frames"]):
        if r["frame"] != int(f) or not np.isclose(r["time_s"], f / fps):
            raise ValueError("Person frame/time mismatch")
        ids = []
        assigned = []
        for t in r["tracks"]:
            identity = int(t["track_id"])
            ids.append(identity)
            if identity < 1:
                raise ValueError(
                    "track_id must be a positive integer or its string representation"
                )
            if type(t["observed"]) is not bool or type(t["confirmed"]) is not bool:
                raise ValueError("Boolean track flags required")
            if type(t["missed_steps"]) is not int or t["missed_steps"] < 0:
                raise ValueError("Invalid missed_steps")
            b = t["bbox"]
            if b is None:
                if t["observed"]:
                    raise ValueError("Observed track requires a bbox")
            elif (
                np.asarray(b).shape != (4,)
                or not np.isfinite(b).all()
                or b[2] <= b[0]
                or b[3] <= b[1]
            ):
                raise ValueError("Invalid xyxy bbox")
            i = t.get("detection_index")
            if t["observed"]:
                if (
                    i is None
                    or int(i) != i
                    or not 0 <= i < len(data["det_frame"])
                    or data["det_frame"][int(i)] != f
                ):
                    raise ValueError(
                        "Observed track must point to a current input detection"
                    )
                assigned.append(i)
        if len(ids) != len(set(ids)) or len(assigned) != len(set(assigned)):
            raise ValueError("Duplicate ID or detection assignment in a frame")


def events(rows, times, length=16, delay=1.8, observed_times=None):
    candidates = [r["candidate_time_s"] for r in rows]
    if not np.isfinite(candidates).all() or any(
        b <= a for a, b in zip(candidates, candidates[1:])
    ):
        raise ValueError("Invalid candidate order")
    for r in rows:
        c = r["candidate_time_s"]
        if not len(times) or c < times[0] or c > times[-1]:
            raise ValueError("Candidate outside recording")
        if observed_times is not None and not np.any(
            np.isclose(observed_times, c, rtol=0, atol=1e-6)
        ):
            raise ValueError("Candidate must be an observed input timestamp")
        end = int(np.searchsorted(times, c + delay))
        if end >= len(times):
            if (
                r["time_s"] is not None
                or r["decision_index"] is not None
                or r["outcome"] != "unknown"
                or r["made_probability"] is not None
            ):
                raise ValueError("End-of-stream event must remain unknown")
        else:
            if (
                r["decision_index"] != end
                or r["time_s"] is None
                or not np.isclose(r["time_s"], times[end])
            ):
                raise ValueError("Decision is not on the scheduled frame")
            if end < length - 1 and r["outcome"] != "unknown":
                raise ValueError("Insufficient history")
        if r["outcome"] == "unknown":
            if r["made_probability"] is not None:
                raise ValueError("Unknown cannot carry a probability")
        else:
            p = r["made_probability"]
            if (
                p is None
                or not np.isfinite(p)
                or not 0 <= p <= 1
                or r["outcome"] != ("made" if p >= 0.5 else "missed")
            ):
                raise ValueError("Invalid probability/outcome")
