"""Готовые очередь решений и оценка бросков. Изменять этот модуль не нужно."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, f1_score


def predict_stream(
    features, time_s, candidates, predict_probability, length=16, delay_s=1.8
):
    """Выдать исходы после ожидания нужного числа шагов.

    Аргументы:
        features: признаки [кадры, признаки], нормализованные по train.
        time_s: возрастающие временные метки шагов, в секундах.
        candidates: времена кандидатов из алгоритма, не из разметки.
        predict_probability: функция окно [T, D] → вероятность попадания.
        length: число шагов в окне сети.
        delay_s: ожидание после кандидата, в секундах.

    Возвращает:
        list[dict]: журнал исходов; unknown при нехватке истории или конца
            записи. Модель получает копию только уже доступного окна.
    """
    x, times, candidates = (
        np.asarray(features),
        np.asarray(time_s),
        np.asarray(candidates, dtype=float),
    )
    if (
        times.ndim != 1
        or x.ndim != 2
        or len(x) != len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(x).all()
        or not np.all(np.diff(times) > 0)
        or candidates.ndim != 1
        or not np.isfinite(candidates).all()
        or not np.all(np.diff(candidates) > 0)
        or (
            len(candidates)
            and (
                not len(times) or candidates[0] < times[0] or candidates[-1] > times[-1]
            )
        )
        or type(length) is not int
        or length < 1
        or not np.isfinite(delay_s)
        or delay_s < 0
    ):
        raise ValueError("Invalid stream, candidate chronology, length or delay")
    pending, result, cursor = [], [], 0
    for end, t in enumerate(times):
        while cursor < len(candidates) and candidates[cursor] <= t:
            pending.append(float(candidates[cursor]))
            cursor += 1
        while pending and pending[0] + delay_s <= t:
            candidate = pending.pop(0)
            row = dict(
                candidate_time_s=candidate,
                time_s=float(t),
                decision_index=end,
                outcome="unknown",
                made_probability=None,
            )
            if end < length - 1:
                row["reason"] = "insufficient_history"
            else:
                p = float(predict_probability(x[end - length + 1 : end + 1].copy()))
                if not np.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError(
                        "Callback must return a finite probability in [0,1]"
                    )
                row.update(made_probability=p, outcome="made" if p >= 0.5 else "missed")
            result.append(row)
    result.extend(
        dict(
            candidate_time_s=c,
            time_s=None,
            decision_index=None,
            outcome="unknown",
            made_probability=None,
            reason="recording_ended_before_decision",
        )
        for c in pending
    )
    return result


def evaluate_stream(attempts, predictions, delay_s=1.8):
    """Сопоставить попытки и ответы один-к-одному и вычислить метрики.

    Аргументы:
        attempts: правильные попытки с reference_time_s и outcome.
        predictions: журнал из predict_stream.
        delay_s: ожидание решения после кандидата.

    Возвращает:
        dict: метрики событий, известных исходов и список разобранных случаев.

    Окно сопоставления: от reference_time_s до +2,5 с. Для unknown в конце
    записи candidate + delay служит только для сопоставления, не как
    измеренная задержка. Unknown не считается правильным исходом.
    """
    truth_times = np.asarray([a["reference_time_s"] for a in attempts])
    prediction_times = np.asarray(
        [
            p["time_s"] if p["time_s"] is not None else p["candidate_time_s"] + delay_s
            for p in predictions
        ]
    )
    matches = {}
    if len(attempts) and len(predictions):
        delta = prediction_times[None, :] - truth_times[:, None]
        valid = (delta >= 0) & (delta <= 2.5)
        ai, pi = linear_sum_assignment(np.where(valid, np.abs(delta - 1.3), 1e6))
        matches = {int(a): int(p) for a, p in zip(ai, pi) if valid[a, p]}
    y, prob, delays, cases = [], [], [], []
    known = correct = unknown_known = 0
    for i, a in enumerate(attempts):
        if a["outcome"] not in ("made", "missed", "uncertain"):
            raise ValueError("Invalid outcome label")
        is_known = a["outcome"] != "uncertain"
        known += is_known
        p = predictions[matches[i]] if i in matches else None
        if p is not None:
            if p["time_s"] is not None:
                delays.append(p["time_s"] - a["reference_time_s"])
            if is_known and p["outcome"] == "unknown":
                unknown_known += 1
            if p["outcome"] != "unknown":
                v = p["made_probability"]
                if v is None or not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError("Resolved result needs a probability")
                if p["outcome"] != ("made" if v >= 0.5 else "missed"):
                    raise ValueError("Outcome must use threshold 0.5")
                if is_known:
                    y.append(int(a["outcome"] == "made"))
                    prob.append(v)
                    correct += p["outcome"] == a["outcome"]
        cases.append(
            dict(attempt_id=a["attempt_id"], outcome=a["outcome"], prediction=p)
        )
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(attempts) - tp
    yy, pp = np.asarray(y), np.asarray(prob)
    return dict(
        event=dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else 0,
            recall=tp / (tp + fn) if tp + fn else 0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        ),
        known_attempts=int(known),
        uncertain_attempts=len(attempts) - int(known),
        correct_detected_and_classified=int(correct),
        end_to_end_accuracy=correct / known if known else None,
        unknown_predictions=sum(p["outcome"] == "unknown" for p in predictions),
        unknown_matched_known=int(unknown_known),
        labelled_resolved=len(y),
        brier=float(np.mean((pp - yy) ** 2)) if len(y) else None,
        macro_f1=(
            float(
                f1_score(yy, pp >= 0.5, labels=[0, 1], average="macro", zero_division=0)
            )
            if len(y)
            else None
        ),
        confusion_missed_made=(
            confusion_matrix(yy, pp >= 0.5, labels=[0, 1]).tolist()
            if len(y)
            else [[0, 0], [0, 0]]
        ),
        median_delay_s=float(np.median(delays)) if delays else None,
        cases=cases,
    )
