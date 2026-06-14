from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

FloatArray = npt.NDArray[np.float64]

N_TIERS = 5


def composite_risk(factors: FloatArray, weights: FloatArray) -> float:
    return float(np.clip(float(factors @ weights), 0.0, 1.0))


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


def class_probabilities(risk: float, thresholds: FloatArray, scale: float) -> FloatArray:
    cumulative = _sigmoid((thresholds - risk) / max(scale, 1e-6))
    bounds = np.concatenate(([0.0], cumulative, [1.0]))
    probs = np.diff(bounds)
    probs = np.clip(probs, 1e-9, None)
    return np.asarray(probs / probs.sum(), dtype=np.float64)


class RiskHead(nn.Module):
    def __init__(self, n_factors: int, scale: float = 0.15) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_factors))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale)))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, factors: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        risk = factors @ self.weights()
        scale = torch.exp(self.log_scale).clamp_min(1e-3)
        cumulative = torch.sigmoid((thresholds.unsqueeze(0) - risk.unsqueeze(1)) / scale)
        zeros = torch.zeros(risk.shape[0], 1)
        ones = torch.ones(risk.shape[0], 1)
        bounds = torch.cat([zeros, cumulative, ones], dim=1)
        probs = bounds[:, 1:] - bounds[:, :-1]
        return probs.clamp_min(1e-9)


class RiskScorer:
    def __init__(self, weights: FloatArray, thresholds: FloatArray, scale: float) -> None:
        self.weights = weights
        self.thresholds = thresholds
        self.scale = scale

    def risk(self, factors: FloatArray) -> float:
        return composite_risk(factors, self.weights)

    def probabilities(self, factors: FloatArray) -> FloatArray:
        return class_probabilities(self.risk(factors), self.thresholds, self.scale)
