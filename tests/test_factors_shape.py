from __future__ import annotations

import numpy as np
from conftest import make_summary

from pedolgov.valuation.factors import N_FACTORS, factor_vector


def test_factor_vector_shape_and_range() -> None:
    vector = factor_vector(make_summary())
    assert vector.shape == (N_FACTORS,)
    assert vector.dtype == np.float64
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


def test_age_factor_ordering() -> None:
    from pedolgov.manifest.declaration import AgeGroup

    values = [factor_vector(make_summary(age_group=a))[3] for a in AgeGroup]
    assert values[0] > values[1] > values[2] > values[3]


def test_binary_factors_extremes() -> None:
    high = factor_vector(make_summary(on_kids_list=True, narrow_ti=True, label_gap=True))
    low = factor_vector(make_summary(on_kids_list=False, narrow_ti=False, label_gap=False))
    assert high[6] == 1.0 and low[6] == 0.0
    assert high[7] == 1.0 and low[7] == 0.0
    assert high[5] > low[5]
