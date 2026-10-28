from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

N_TIERS = 5


def confusion_matrix(y_true: IntArray, y_pred: IntArray, n: int) -> IntArray:
    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        matrix[t, p] += 1
    return matrix


def macro_f1(y_true: IntArray, y_pred: IntArray, n: int = N_TIERS) -> float:
    matrix = confusion_matrix(y_true, y_pred, n)
    scores = []
    for k in range(n):
        tp = matrix[k, k]
        fp = matrix[:, k].sum() - tp
        fn = matrix[k, :].sum() - tp
        if tp == 0 and (fp == 0 or fn == 0):
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores))


def ordinal_accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    return float(np.mean(y_true == y_pred))


def ordinal_mae(y_true: IntArray, y_pred: IntArray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def auroc(scores: FloatArray, labels: IntArray) -> float:
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("scores and labels must have equal lengths and finite values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("AUROC labels must be binary")
    ranks = _rankdata(scores) + 1.0
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman_rho(a: FloatArray, b: FloatArray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return 0.0 if denom == 0 else float((ra * rb).sum() / denom)


def _rankdata(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(len(counts))
    acc = 0.0
    sorted_unique_ranks: list[float] = []
    for count in counts:
        sorted_unique_ranks.append(acc + (count - 1) / 2.0)
        acc += count
    tie_mean = np.asarray(sorted_unique_ranks)
    return tie_mean[inverse]


def weighted_kappa(y_true: IntArray, y_pred: IntArray, n: int = N_TIERS) -> float:
    matrix = confusion_matrix(y_true, y_pred, n).astype(np.float64)
    total = matrix.sum()
    if total == 0:
        return 0.0
    observed = matrix / total
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)
    idx = np.arange(n)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n - 1) ** 2
    num = (weights * observed).sum()
    den = (weights * expected).sum()
    return 1.0 if den == 0 else float(1.0 - num / den)


def expected_calibration_error(probs: FloatArray, y_true: IntArray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def coverage(prediction_sets: list[set[int]], y_true: IntArray) -> float:
    hits = [1.0 if int(y) in s else 0.0 for s, y in zip(prediction_sets, y_true, strict=False)]
    return float(np.mean(hits)) if hits else 0.0


def average_set_size(prediction_sets: list[set[int]]) -> float:
    return float(np.mean([len(s) for s in prediction_sets])) if prediction_sets else 0.0
