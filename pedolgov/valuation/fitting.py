from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from pedolgov.audit.metrics import weighted_kappa
from pedolgov.manifest.config import FitConfig
from pedolgov.valuation.assignment import assign_tier
from pedolgov.valuation.scoring import RiskHead

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_TAU3_BAND = (0.55, 0.65)


def ordinal_cross_entropy(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    picked = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -torch.log(picked).mean()


def project_capped_simplex(weights: FloatArray, cap: float) -> FloatArray:
    w = np.clip(weights, 0.0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    for _ in range(100):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        if not free.any():
            break
        free_mass = w[free].sum()
        if free_mass <= 0:
            w[free] += excess / int(free.sum())
        else:
            w[free] += excess * w[free] / free_mass
    return np.asarray(w, dtype=np.float64)


def fit_weights(
    factors: FloatArray,
    labels: IntArray,
    thresholds: FloatArray,
    cfg: FitConfig,
    cap: float,
) -> tuple[FloatArray, float, list[float]]:
    x = torch.tensor(factors, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    tau = torch.tensor(thresholds, dtype=torch.float32)
    head = RiskHead(factors.shape[1])
    optimizer = torch.optim.Adam(
        head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[float] = []
    for _ in range(cfg.steps):
        optimizer.zero_grad()
        probs = head(x, tau)
        loss = ordinal_cross_entropy(probs, y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        raw = head.weights().numpy().astype(np.float64)
        scale = float(torch.exp(head.log_scale))
    return project_capped_simplex(raw, cap), scale, history


def _objective(risk: FloatArray, labels: IntArray, thresholds: FloatArray) -> float:
    preds = np.array([int(assign_tier(float(r), thresholds)) for r in risk], dtype=np.int64)
    return weighted_kappa(labels, preds)


def calibrate_thresholds(
    risk: FloatArray,
    labels: IntArray,
    grid: int,
    n_tiers: int = 5,
) -> FloatArray:
    freq = np.bincount(labels, minlength=n_tiers).astype(np.float64)
    cum = np.cumsum(freq / freq.sum())[:-1]
    thresholds = np.clip(np.quantile(risk, cum), 1e-3, 1.0 - 1e-3)
    thresholds = np.sort(thresholds)
    thresholds += np.linspace(0, 1e-4, len(thresholds))
    for _ in range(3):
        for k in range(len(thresholds)):
            lower = thresholds[k - 1] + 1e-3 if k > 0 else 1e-3
            upper = thresholds[k + 1] - 1e-3 if k + 1 < len(thresholds) else 1.0 - 1e-3
            if k == 2:
                lower = max(lower, _TAU3_BAND[0])
                upper = min(upper, _TAU3_BAND[1])
            if upper <= lower:
                continue
            candidates = np.linspace(lower, upper, grid)
            best_val = -2.0
            best = thresholds[k]
            for cand in candidates:
                trial = thresholds.copy()
                trial[k] = cand
                if not np.all(np.diff(trial) > 0):
                    continue
                val = _objective(risk, labels, trial)
                if val > best_val:
                    best_val = val
                    best = cand
            thresholds[k] = best
    return thresholds
