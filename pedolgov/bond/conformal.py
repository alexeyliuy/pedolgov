from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def nonconformity_scores(probs: FloatArray, labels: IntArray) -> FloatArray:
    rows = np.arange(len(labels))
    return 1.0 - probs[rows, labels]


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    ordered = np.sort(scores)
    return float(ordered[rank - 1])


def prediction_set(prob_row: FloatArray, qhat: float) -> set[int]:
    return {int(k) for k in range(len(prob_row)) if prob_row[k] >= 1.0 - qhat}


@dataclass
class SplitConformal:
    alpha: float
    ood_percentile: float = 95.0
    qhat: float = 1.0
    _calibration_scores: FloatArray | None = None

    def fit(self, probs: FloatArray, labels: IntArray) -> SplitConformal:
        scores = nonconformity_scores(probs, labels)
        self._calibration_scores = scores
        self.qhat = conformal_quantile(scores, self.alpha)
        return self

    def predict_set(self, prob_row: FloatArray) -> set[int]:
        result = prediction_set(prob_row, self.qhat)
        if not result:
            result = {int(np.argmax(prob_row))}
        return result

    def predict_sets(self, probs: FloatArray) -> list[set[int]]:
        return [self.predict_set(row) for row in probs]

    def ood_distance(self, prob_row: FloatArray, label: int) -> bool:
        if self._calibration_scores is None:
            return False
        threshold = float(np.percentile(self._calibration_scores, self.ood_percentile))
        return float(1.0 - prob_row[label]) > threshold


def empirical_coverage(prediction_sets: list[set[int]], labels: IntArray) -> float:
    hits = [1.0 if int(y) in s else 0.0 for s, y in zip(prediction_sets, labels, strict=False)]
    return float(np.mean(hits)) if hits else 0.0


def average_set_size(prediction_sets: list[set[int]]) -> float:
    return float(np.mean([len(s) for s in prediction_sets])) if prediction_sets else 0.0
