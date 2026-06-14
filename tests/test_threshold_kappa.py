from __future__ import annotations

import numpy as np

from pedolgov.audit.metrics import weighted_kappa
from pedolgov.valuation.assignment import as_thresholds, assign_tier
from pedolgov.valuation.fitting import calibrate_thresholds


def test_calibration_beats_naive_thresholds() -> None:
    rng = np.random.default_rng(1)
    n = 800
    risk = rng.random(n)
    true_thresholds = as_thresholds((0.18, 0.36, 0.6, 0.82))
    labels = np.array([int(assign_tier(float(r), true_thresholds)) for r in risk], dtype=np.int64)

    calibrated = calibrate_thresholds(risk, labels, grid=41)
    naive = as_thresholds((0.5, 0.6, 0.7, 0.8))

    pred_cal = np.array([int(assign_tier(float(r), calibrated)) for r in risk], dtype=np.int64)
    pred_naive = np.array([int(assign_tier(float(r), naive)) for r in risk], dtype=np.int64)

    assert np.all(np.diff(calibrated) > 0)
    assert 0.55 <= calibrated[2] <= 0.65
    assert weighted_kappa(labels, pred_cal) > weighted_kappa(labels, pred_naive)
