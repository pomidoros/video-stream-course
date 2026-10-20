"""Открытые пороги. Оценка пересчитывается из предсказаний, а не берётся из отчёта студента."""

import math

PUBLIC = {
    "outcome.gru.aggregate.event_f1": (">=", 0.85),
    "outcome.gru.aggregate.end_to_end_accuracy": (">=", 0.65),
    "outcome.gru.aggregate.resolved_coverage": (">=", 0.85),
    "outcome.gru.aggregate.brier": ("<=", 0.30),
    "outcome.gru.aggregate.macro_f1": (">=", 0.60),
    "people.aggregate.idf1": (">=", 0.35),
    "people.aggregate.recall": (">=", 0.75),
    "people.aggregate.retrieval_top1": (">=", 0.75),
}
# Same formulas and published thresholds; hidden recordings, not hidden requirements.
HIDDEN = {
    key.split("outcome.gru.aggregate.")[1]: value
    for key, value in PUBLIC.items()
    if key.startswith("outcome.gru.aggregate.")
}


def checks(metrics, rules=PUBLIC):
    result = []
    for name, (op, threshold) in rules.items():
        try:
            value = metrics
            for part in name.split("."):
                value = value[part]
            valid = type(value) in (int, float) and math.isfinite(value)
            passed = bool(
                valid and (value >= threshold if op == ">=" else value <= threshold)
            )
        except (KeyError, TypeError):
            value = None
            passed = False
        result.append(
            dict(
                metric=name,
                value=value,
                operator=op,
                threshold=threshold,
                passed=passed,
            )
        )
    return result
