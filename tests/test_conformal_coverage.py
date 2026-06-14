from __future__ import annotations

import numpy as np

from pedolgov.bond.conformal import (
    SplitConformal,
    conformal_quantile,
    empirical_coverage,
    nonconformity_scores,
)


def _draw(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, k, n)
    logits = rng.normal(0.0, 1.0, (n, k))
    logits[np.arange(n), y] += 2.0
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, y


def test_quantile_rank_is_finite_sample() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    q = conformal_quantile(scores, alpha=0.1)
    assert 0.88 <= q <= 0.92


def test_marginal_coverage_holds_on_average() -> None:
    rng = np.random.default_rng(7)
    alpha = 0.1
    covs = []
    for _ in range(150):
        pc, yc = _draw(rng, 400, 5)
        pt, yt = _draw(rng, 400, 5)
        conformal = SplitConformal(alpha=alpha).fit(pc, yc)
        covs.append(empirical_coverage(conformal.predict_sets(pt), yt))
    assert float(np.mean(covs)) >= 1.0 - alpha - 0.02


def test_nonconformity_definition() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    labels = np.array([0, 2])
    scores = nonconformity_scores(probs, labels)
    assert np.allclose(scores, [0.3, 0.7])
