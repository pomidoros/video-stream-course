"""Готовая оценка Re-ID. Студенческая функция возвращает ID, а не значения метрик."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from players.people_metrics import iou


def cases(data, gt):
    labels = np.full(len(data["det_frame"]), -1, int)
    for f in data["frames"]:
        inds = np.flatnonzero(data["det_frame"] == f)
        truth = [g for g in gt if g["frame"] == f]
        if not len(inds) or not truth:
            continue
        overlap = iou(data["bbox"][inds], [g["bbox"] for g in truth])
        rr, cc = linear_sum_assignment(1 - overlap)
        for r, c in zip(rr, cc):
            if overlap[r, c] >= 0.5:
                labels[inds[r]] = truth[c]["person_id"]
    stop = data["frames"][-1] + (data["frames"][1] - data["frames"][0])
    early = (data["det_frame"] < stop / 3) & (labels >= 0)
    late = (data["det_frame"] >= 2 * stop / 3) & (labels >= 0)
    ids = np.array(sorted(set(labels[early])))
    gallery = np.stack([data["embeddings"][early & (labels == i)].mean(0) for i in ids])
    query = data["embeddings"][late]
    return dict(
        gallery=gallery,
        gallery_ids=ids,
        queries=query,
        truth=labels[late],
        matched_detections=int((labels >= 0).sum()),
        unmatched_detections=int((labels < 0).sum()),
    )


def score(solution, data, gt, threshold=0.3):
    c = cases(data, gt)
    result = {}
    for mode, selection in [
        ("natural", slice(None)),
        ("controlled_unknown", slice(1, None)),
    ]:
        ids = c["gallery_ids"][selection]
        gallery = c["gallery"][selection]
        pred = np.asarray(
            solution.identify_people(
                c["queries"].copy(), gallery.copy(), ids.copy(), threshold
            )
        )
        if pred.shape != (len(c["truth"]),) or not np.isin(pred, np.r_[ids, -1]).all():
            raise ValueError("Return one gallery ID or -1 per query")
        dist = np.asarray(
            solution.cosine_distances(c["queries"].copy(), gallery.copy())
        )
        if dist.shape != (len(pred), len(ids)) or not np.isfinite(dist).all():
            raise ValueError("Invalid cosine matrix")
        top = ids[dist.argmin(1)]
        known = np.isin(c["truth"], ids)
        result[mode] = dict(
            known=int(known.sum()),
            unknown=int((~known).sum()),
            top1_correct=int(((top == c["truth"]) & known).sum()),
            accepted_correct_known=int(((pred == c["truth"]) & known).sum()),
            rejected_unknown=int(((pred == -1) & ~known).sum()),
            false_accept_unknown=int(((pred != -1) & ~known).sum()),
        )
    result.update(
        threshold=threshold,
        matched_detections=c["matched_detections"],
        unmatched_detections=c["unmatched_detections"],
    )
    return result
