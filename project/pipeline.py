"""Готовый запуск: студенческие функции, обучение, предсказание и оценка."""

from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import torch
import motmetrics as mm
from project.common import ROOT, dump, solutions, checked_detections, stress_input
from project import contracts
from shots.outcome_io import load_recordings, load_features, load_attempts
from shots.stream_utils import predict_stream, evaluate_stream
from players.people_io import load_manifest, load_clip, load_labels
from players.people_metrics import accumulator
from project.retrieval import score as retrieval_score


def load_ball(root, row, solution):
    data = load_features(root, row)
    detections = checked_detections(root, row)
    tracks = solution.track_ball(detections, np.asarray(row["rim_bbox"], float))
    contracts.ball_rows(tracks, detections)
    candidates = solution.find_candidates(tracks, np.asarray(row["rim_bbox"], float))
    return data, tracks, candidates


def normalize(features, scaler):
    mean = np.asarray(scaler["mean"], np.float32)
    scale = np.asarray(scaler["scale"], np.float32)
    if (
        mean.shape != (70,)
        or scale.shape != (70,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Invalid train scaler")
    return np.clip((features - mean) / scale, -6, 6).astype(np.float32)


def train(root=ROOT, output=ROOT / "artifacts", seed=17, epochs=120):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    solution, _ = solutions()
    torch.set_num_threads(2)
    recordings = load_recordings(root)
    loaded = []
    for row in recordings:
        if row["split"] not in ("train", "validation"):
            continue
        data, tracks, candidates = load_ball(root, row, solution)
        loaded.append(
            dict(
                row=row,
                data=data,
                tracks=tracks,
                candidates=candidates,
                attempts=load_attempts(root, row),
            )
        )
    training = [r for r in loaded if r["row"]["split"] == "train"]
    validation = [r for r in loaded if r["row"]["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Need separate train/validation recordings")
    values = np.concatenate([r["data"]["features"] for r in training])
    scaler = dict(
        mean=values.mean(0).tolist(), scale=np.maximum(values.std(0), 1e-3).tolist()
    )
    for r in loaded:
        r["x"] = normalize(r["data"]["features"], scaler)
    pairs = [
        solution.build_training_windows(r["x"], r["data"]["time_s"], r["attempts"], 16)
        for r in training
    ]
    tx = np.concatenate([x for x, y in pairs])
    ty = np.concatenate([y for x, y in pairs])
    vx = []
    vy = []
    if (
        tx.ndim != 3
        or tx.shape[1:] != (16, 70)
        or ty.shape != (len(tx),)
        or not np.isfinite(tx).all()
        or not np.isin(ty, [0, 1]).all()
    ):
        raise ValueError("Invalid training windows")
    for r in validation:
        pred = predict_stream(
            r["x"], r["data"]["time_s"], r["candidates"], lambda x: 0.5, 16
        )
        for case in evaluate_stream(r["attempts"], pred)["cases"]:
            p = case["prediction"]
            if case["outcome"] == "uncertain" or p is None or p["outcome"] == "unknown":
                continue
            end = p["decision_index"]
            vx.append(r["x"][end - 15 : end + 1])
            vy.append(float(case["outcome"] == "made"))
    if not len(vx):
        raise ValueError("No validation windows: check ball candidates first")
    vx = np.stack(vx)
    vy = np.asarray(vy, np.float32)
    summary = {}
    for kind in ["last", "mean", "gru"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = solution.TemporalOutcomeModel(kind)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        start = time.perf_counter()
        history = solution.train_model(
            model, (tx, ty), (vx, vy), dict(seed=seed, epochs=epochs, lr=0.001)
        )
        if (
            not isinstance(history, dict)
            or not history.get("curve")
            or "best_epoch" not in history
        ):
            raise ValueError("Training must return curve and best_epoch")
        for entry in history["curve"]:
            if not np.isfinite([entry["train_loss"], entry["validation_brier"]]).all():
                raise ValueError("Non-finite learning curve")
        delta = float(
            sum(
                ((model.state_dict()[k] - v) ** 2).sum() for k, v in before.items()
            ).sqrt()
        )
        if not np.isfinite(delta) or delta <= 1e-8:
            raise ValueError("Weights did not change; actual training is required")
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(vx))
            if logits.shape != (len(vx),) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite logits [B]")
        torch.save(model.state_dict(), output / f"{kind}.pt")
        history.update(
            seed=seed,
            weight_delta_l2=delta,
            measured_seconds=time.perf_counter() - start,
        )
        dump(output / f"{kind}.history.json", history)
        summary[kind] = {
            k: history[k] for k in ["best_epoch", "weight_delta_l2", "measured_seconds"]
        }
    dump(
        output / "metadata.json",
        dict(
            schema_version=1,
            length=16,
            delay_s=1.8,
            seed=seed,
            scaler=scaler,
            train_recordings=[r["row"]["sample_id"] for r in training],
            train_windows=len(tx),
            validation_windows=len(vx),
            training=summary,
        ),
    )
    return summary


def predict(root, models, output, split="validation", kind="gru"):
    """Выполнить предсказание без чтения правильных ответов."""
    root = Path(root)
    models = Path(models)
    output = Path(output)
    solution, _ = solutions()
    torch.set_num_threads(2)
    metadata = json.loads((models / "metadata.json").read_text())
    if metadata["length"] != 16 or metadata["delay_s"] != 1.8:
        raise ValueError("Required protocol is length16 / delay1.8")
    model = solution.TemporalOutcomeModel(kind)
    model.load_state_dict(
        torch.load(models / f"{kind}.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    def probability(window):
        with torch.inference_mode():
            return float(model(torch.from_numpy(window)[None]).sigmoid()[0])

    result = {}
    for row in load_recordings(root):
        if split != "all" and row["split"] != split:
            continue
        data, tracks, candidates = load_ball(root, row, solution)
        x = normalize(data["features"], metadata["scaler"])
        events = predict_stream(x, data["time_s"], candidates, probability, 16, 1.8)
        contracts.events(
            events, data["time_s"], observed_times=[r["time_s"] for r in tracks]
        )
        name = row["sample_id"]
        dump(output / f"{name}.events.json", events)
        dump(output / f"{name}.ball.json", tracks)
        result[name] = events
    if not result:
        raise ValueError("No recordings match the requested split")
    return result


def outcome_metrics(root, predictions):
    """Оценить предсказания отдельно для каждой записи и собрать общие метрики."""
    reports = {}
    for row in load_recordings(root):
        name = row["sample_id"]
        if name not in predictions:
            continue
        times = load_features(root, row)["time_s"]
        contracts.events(
            predictions[name],
            times,
            observed_times=[t for f, t, d in checked_detections(root, row)],
        )
        reports[name] = evaluate_stream(load_attempts(root, row), predictions[name])
    if not reports:
        raise ValueError("No predictions to score")
    tp = sum(r["event"]["tp"] for r in reports.values())
    fp = sum(r["event"]["fp"] for r in reports.values())
    fn = sum(r["event"]["fn"] for r in reports.values())
    known = sum(r["known_attempts"] for r in reports.values())
    resolved = sum(r["labelled_resolved"] for r in reports.values())
    correct = sum(r["correct_detected_and_classified"] for r in reports.values())
    confusion = np.sum([r["confusion_missed_made"] for r in reports.values()], axis=0)
    f1 = []
    for k in (0, 1):
        denominator = confusion[k, :].sum() + confusion[:, k].sum()
        f1.append(2 * confusion[k, k] / denominator if denominator else 0.0)
    aggregate = dict(
        event_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        event_tp=tp,
        event_fp=fp,
        event_fn=fn,
        end_to_end_accuracy=correct / known if known else 0.0,
        known_attempts=known,
        resolved_coverage=resolved / known if known else 0.0,
        brier=(
            sum(
                r["brier"] * r["labelled_resolved"]
                for r in reports.values()
                if r["brier"] is not None
            )
            / resolved
            if resolved
            else 1.0
        ),
        macro_f1=float(np.mean(f1)),
        confusion=confusion.tolist(),
        uncertain_attempts=sum(r["uncertain_attempts"] for r in reports.values()),
    )
    return dict(aggregate=aggregate, recordings=reports)


def people(root, output):
    root = Path(root)
    output = Path(output)
    _, solution = solutions()
    C = root / "players"
    result = {}
    all_acc = []
    stress_acc = []
    natural = []
    for clip in load_manifest(C)["clips"]:
        data = load_clip(C, clip)
        gt = load_labels(C, clip)
        name = clip["name"]
        scores = {}
        for mode in ["normal", "gaps"]:
            inp = data if mode == "normal" else stress_input(data)
            rows = solution.track_with_deepsort(inp, clip["fps"])
            contracts.people_rows(rows, inp, clip["fps"])
            dump(output / f"{name}.{mode}.tracks.json", rows)
            visible = [
                r | dict(tracks=[t for t in r["tracks"] if t["bbox"] is not None])
                for r in rows
            ]
            acc = accumulator(visible, gt)
            fields = [
                "idf1",
                "mota",
                "num_switches",
                "num_false_positives",
                "num_misses",
                "recall",
            ]
            scores[mode] = (
                mm.metrics.create().compute(acc, metrics=fields).iloc[0].to_dict()
            )
            if clip["split"] == "evaluation":
                (all_acc if mode == "normal" else stress_acc).append(acc)
        scores["retrieval"] = retrieval_score(solution, data, gt)
        if clip["split"] == "evaluation":
            natural.append(scores["retrieval"]["natural"])
        result[name] = scores
    names = [str(i) for i in range(len(all_acc))]
    agg = (
        mm.metrics.create()
        .compute_many(all_acc, names=names, metrics=fields, generate_overall=True)
        .loc["OVERALL"]
        .to_dict()
    )
    gaps = (
        mm.metrics.create()
        .compute_many(stress_acc, names=names, metrics=fields, generate_overall=True)
        .loc["OVERALL"]
        .to_dict()
    )
    agg["retrieval_top1"] = sum(r["top1_correct"] for r in natural) / sum(
        r["known"] for r in natural
    )
    result = dict(aggregate=agg, gaps_aggregate=gaps, clips=result)
    dump(output / "people_metrics.json", result)
    return result


def evaluate(root=ROOT, models=ROOT / "artifacts", output=ROOT / "outputs"):
    output = Path(output)
    outcomes = {}
    for kind in ["last", "mean", "gru"]:
        pred = predict(root, models, output / kind, kind=kind)
        outcomes[kind] = outcome_metrics(root, pred)
    persons = people(root, output / "people")
    result = dict(schema_version=1, outcome=outcomes, people=persons)
    dump(output / "metrics.json", result)
    return result
