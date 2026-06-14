from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from pedolgov.audit.metrics import (
    auroc,
    expected_calibration_error,
    macro_f1,
    spearman_rho,
    weighted_kappa,
)


def test_macro_f1_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 300)
    y_pred = rng.integers(0, 5, 300)
    mine = macro_f1(y_true, y_pred)
    ref = f1_score(y_true, y_pred, average="macro", labels=list(range(5)), zero_division=0)
    assert abs(mine - ref) < 1e-9


def test_auroc_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    scores = rng.random(200) + labels * 0.5
    assert abs(auroc(scores, labels) - roc_auc_score(labels, scores)) < 1e-9


def test_weighted_kappa_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, 300)
    y_pred = rng.integers(0, 5, 300)
    ref = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5)))
    assert abs(weighted_kappa(y_true, y_pred) - ref) < 1e-9


def test_spearman_perfect_monotone() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(spearman_rho(a, b) - 1.0) < 1e-9


def test_ece_is_zero_for_perfect_confidence() -> None:
    probs = np.eye(5)[np.array([0, 1, 2, 3, 4])]
    y = np.array([0, 1, 2, 3, 4])
    assert expected_calibration_error(probs, y) < 1e-9
