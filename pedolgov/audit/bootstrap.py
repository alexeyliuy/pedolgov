from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def paired_bootstrap(
    a: FloatArray, b: FloatArray, resamples: int, seed: int
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        idx = rng.integers(0, n, n)
        diffs[r] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi), float(min(1.0, p))


def bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvalues)
    return [p <= alpha / m for p in pvalues]


def cohens_d(a: FloatArray, b: FloatArray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def cliffs_delta(a: FloatArray, b: FloatArray) -> float:
    greater = 0
    less = 0
    for x in a:
        greater += int((x > b).sum())
        less += int((x < b).sum())
    n = len(a) * len(b)
    return 0.0 if n == 0 else float((greater - less) / n)
