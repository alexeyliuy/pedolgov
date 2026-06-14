from __future__ import annotations

import numpy as np

from pedolgov.valuation.assignment import as_thresholds, assign_tier
from pedolgov.valuation.scoring import composite_risk


def test_composite_risk_monotone_in_each_factor() -> None:
    weights = np.full(8, 0.125)
    base = np.full(8, 0.3)
    for j in range(8):
        low = base.copy()
        high = base.copy()
        low[j] = 0.1
        high[j] = 0.9
        assert composite_risk(high, weights) >= composite_risk(low, weights)


def test_assignment_is_ordered() -> None:
    thresholds = as_thresholds((0.2, 0.4, 0.6, 0.8))
    tiers = [int(assign_tier(r, thresholds)) for r in [0.05, 0.3, 0.5, 0.7, 0.95]]
    assert tiers == [0, 1, 2, 3, 4]


def test_thresholds_must_increase() -> None:
    import pytest

    with pytest.raises(ValueError):
        as_thresholds((0.2, 0.2, 0.6, 0.8))
