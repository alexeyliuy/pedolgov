from __future__ import annotations

from pedolgov.clearance.governance import GovernanceModule, requires_human_review
from pedolgov.manifest.config import GovernanceConfig
from pedolgov.manifest.declaration import AgeGroup, RiskTier


def test_hitl_truth_table() -> None:
    cfg = GovernanceConfig()
    assert requires_human_review(RiskTier.CRITICAL, {4}, [], AgeGroup.CHILD, 3, cfg)
    assert requires_human_review(RiskTier.LOW, {0, 1, 2, 3}, [], AgeGroup.CHILD, 3, cfg)
    assert requires_human_review(RiskTier.LOW, {0}, ["drug_name"], AgeGroup.CHILD, 3, cfg)
    assert requires_human_review(RiskTier.LOW, {0}, [], AgeGroup.NEONATE, 3, cfg)
    assert not requires_human_review(RiskTier.LOW, {0}, [], AgeGroup.CHILD, 3, cfg)


def test_recommendation_mapping() -> None:
    module = GovernanceModule(GovernanceConfig(), set_size_target=3)
    release = module.govern(RiskTier.LOW, {1}, [], [], AgeGroup.CHILD)
    assert release.recommendation == "release" and not release.human_review

    warn = module.govern(RiskTier.HIGH, {3}, [], [], AgeGroup.CHILD)
    assert warn.recommendation == "warn" and warn.human_review

    defer = module.govern(RiskTier.CRITICAL, {4}, [], [], AgeGroup.CHILD)
    assert defer.recommendation == "defer" and defer.human_review
