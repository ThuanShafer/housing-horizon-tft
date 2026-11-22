"""Classification scores for the five tiers."""

from __future__ import annotations

import numpy as np

from housing_tft.spec import TIERS


def scores(y_true, y_pred, n: int = 5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    recalls, f1s, precisions = [], [], []
    for cls in range(n):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    confusion = np.zeros((n, n), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "precision": [float(v) for v in precisions],
        "recall": [float(v) for v in recalls],
        "f1": [float(v) for v in f1s],
        "support": [int(np.sum(y_true == cls)) for cls in range(n)],
        "confusion": confusion.tolist(),
        "labels": [tier["name"] for tier in TIERS],
    }


def binary_auc(y_true, scores) -> float | None:
    """One-vs-rest ROC AUC. None when a class has no positive or no negative cases."""
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + 1 + end + 1) / 2.0
        start = end + 1
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(y_true, scores, limit: int = 40) -> list[list[float]]:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    tp = 0
    fp = 0
    points = [[0.0, 0.0]]
    for index in order:
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
        points.append([fp / n_neg, tp / n_pos])
    if len(points) > limit:
        step = max(1, len(points) // limit)
        points = points[::step]
        if points[-1] != [1.0, 1.0]:
            points.append([1.0, 1.0])
    return [[round(a, 4), round(b, 4)] for a, b in points]


def performance(y_true, y_pred, probabilities) -> dict:
    """Accuracy, macro precision, recall, specificity, F1, and one-vs-rest AUC."""
    base = scores(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    specifics = []
    aucs = []
    curves = []
    per_class = []
    for cls, tier in enumerate(TIERS):
        negative = y_true != cls
        true_negative = int(np.sum(negative & (y_pred != cls)))
        false_positive = int(np.sum(negative & (y_pred == cls)))
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else None
        auc = binary_auc(y_true == cls, probabilities[:, cls])
        if specificity is not None:
            specifics.append(specificity)
        if auc is not None:
            aucs.append(auc)
        per_class.append({
            "tier_id": tier["id"],
            "name": tier["short"],
            "support": base["support"][cls],
            "precision": base["precision"][cls],
            "recall": base["recall"][cls],
            "specificity": specificity,
            "f1": base["f1"][cls],
            "auc": auc,
        })
        curves.append({"name": tier["short"], "auc": auc, "points": roc_curve(y_true == cls, probabilities[:, cls])})
    return {
        "n": base["n"],
        "accuracy": base["accuracy"],
        "precision_macro": float(np.mean(base["precision"])),
        "recall_macro": float(np.mean(base["recall"])),
        "specificity_macro": float(np.mean(specifics)) if specifics else None,
        "f1_macro": base["macro_f1"],
        "auc_macro": float(np.mean(aucs)) if aucs else None,
        "per_class": per_class,
        "roc": curves,
        "confusion": base["confusion"],
        "labels": base["labels"],
    }
