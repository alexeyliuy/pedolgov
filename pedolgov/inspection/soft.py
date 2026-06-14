from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pedolgov.bond.conformal import SplitConformal
from pedolgov.lanes.contracts import EvidenceSummary

FloatArray = npt.NDArray[np.float64]

_DISAGREEMENT_SPREAD = 0.4


def large_set_flag(prediction_set: set[int], threshold: int) -> bool:
    return len(prediction_set) >= threshold


def source_disagreement(source_risk: dict[str, float]) -> bool:
    if len(source_risk) < 2:
        return False
    values = list(source_risk.values())
    return (max(values) - min(values)) > _DISAGREEMENT_SPREAD


def calibration_distance(conformal: SplitConformal, prob_row: FloatArray, predicted: int) -> bool:
    return conformal.ood_distance(prob_row, predicted)


class SoftGuardrails:
    def __init__(self, large_set_threshold: int) -> None:
        self.large_set_threshold = large_set_threshold

    def evaluate(
        self,
        summary: EvidenceSummary,
        prediction_set: set[int],
        conformal: SplitConformal,
        prob_row: FloatArray,
    ) -> list[str]:
        flags: list[str] = []
        if large_set_flag(prediction_set, self.large_set_threshold):
            flags.append("large_prediction_set")
        if source_disagreement(summary.source_risk):
            flags.append("source_disagreement")
        if calibration_distance(conformal, prob_row, int(np.argmax(prob_row))):
            flags.append("calibration_distance")
        return flags
