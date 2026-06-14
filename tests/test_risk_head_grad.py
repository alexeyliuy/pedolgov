from __future__ import annotations

import torch

from pedolgov.valuation.scoring import RiskHead


def test_risk_head_gradients_are_finite() -> None:
    head = RiskHead(n_factors=8)
    factors = torch.rand(16, 8)
    thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8])
    probs = head(factors, thresholds)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)
    loss = -torch.log(probs[:, 0]).mean()
    loss.backward()
    for param in head.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
