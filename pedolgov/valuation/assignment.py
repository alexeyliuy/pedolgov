from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pedolgov.manifest.declaration import RiskTier

FloatArray = npt.NDArray[np.float64]


def as_thresholds(values: tuple[float, float, float, float]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.diff(arr) > 0):
        raise ValueError("thresholds must be strictly increasing")
    return arr


def assign_tier(risk: float, thresholds: FloatArray) -> RiskTier:
    return RiskTier(int(np.searchsorted(thresholds, risk, side="right")))
