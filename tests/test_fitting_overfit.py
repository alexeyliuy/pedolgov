from __future__ import annotations

import numpy as np

from pedolgov.manifest.config import FitConfig
from pedolgov.valuation.assignment import as_thresholds, assign_tier
from pedolgov.valuation.fitting import fit_weights, project_capped_simplex


def test_capped_simplex_projection() -> None:
    w = project_capped_simplex(np.array([10.0, 0.0, 0.0, 0.0]), cap=0.25)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w <= 0.25 + 1e-9)


def test_fit_recovers_dominant_factor() -> None:
    rng = np.random.default_rng(0)
    n = 400
    factors = rng.random((n, 8))
    true = np.zeros(8)
    true[2] = 0.7
    true[5] = 0.3
    thresholds = as_thresholds((0.2, 0.38, 0.58, 0.78))
    risk = factors @ true
    labels = np.array([int(assign_tier(float(r), thresholds)) for r in risk], dtype=np.int64)
    cfg = FitConfig(learning_rate=0.1, steps=200, kappa_grid=9)
    weights, scale, history = fit_weights(factors, labels, thresholds, cfg, cap=0.25)
    assert history[-1] < history[0]
    top_two = {int(i) for i in np.argsort(weights)[-2:]}
    assert top_two == {2, 5}
    assert np.all(weights <= 0.25 + 1e-9)
